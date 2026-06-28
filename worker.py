"""
Off-box macOS build worker.

Run this on a Mac. It connects to the API server's `/workers/macbuild`
WebSocket, waits for build requests, reconstructs the compile sandbox sent by
the server, runs the native `clang++`, and streams the resulting Mach-O binary
back in base64 chunks. The API server (this host has no macOS toolchain) then
bundles it into the user's `.sim` / serves it as a standalone build.

    pip install websockets pydantic
    # optionally point at a local server / set the shared secret:
    export MACBUILD_SERVER_URL="ws://localhost:7000/workers/macbuild"
    export MACBUILD_WORKER_TOKEN="<same value the API server has>"
    python worker.py

Env:
  MACBUILD_SERVER_URL    default wss://api.simulizer.net/workers/macbuild
  MACBUILD_WORKER_TOKEN  shared secret; appended as ?token=… (must match server)
  MACBUILD_CXX           C++ driver, default "clang++"
  WORKER_PRIORITY        integer, lower = higher priority. default 0

Requires the `workers/` package (workers.worker + workers.macbuild) to sit
beside this file, exactly as it does in backend-api.
"""

import asyncio
import base64
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from urllib.parse import quote

from workers.macbuild import MacBuildRequest, MacBuildResponse
from workers.worker import Worker

DEFAULT_SERVER_URL = "wss://api.simulizer.net/workers/macbuild"
# Stream the binary back in chunks small enough to stay well under any
# WebSocket frame-size cap on the server side.
CHUNK_BYTES = 256 * 1024

# clang -v prints the cc1 invocation, then the linker (`ld`) invocation; we use
# these to surface a couple of coarse progress stages to the user.
_CC1_RE = re.compile(r"-cc1\b")
_LD_RE = re.compile(r'(^|[ "/\\])ld(\.exe)?[ "]')


def _build_argv(req: MacBuildRequest, build_dir: Path, exe: Path) -> list[str]:
    """Native macOS compile line — mirrors routes/compile.py's macOS recipe
    minus the docker/osxcross wrapper. No -lrt / static-libstdc++: shm_open and
    the C++ runtime live in libSystem/libc++ on macOS."""
    cxx = os.environ.get("MACBUILD_CXX", "clang++")
    return [
        cxx,
        str(build_dir / req.entry_rel),
        str(build_dir / req.data_cpp_rel),
        req.std_flag,
        "-o", str(exe),
        f"-I{build_dir / req.system_rel}",
        f"-I{build_dir / req.project_rel}",
        req.opt_flag,
        *req.define_flags,
        "-pthread",
        # GCC (the Windows/Linux targets) treats narrowing inside a braced
        # init list as a *warning*, but Apple clang makes -Wc++11-narrowing a
        # hard error by default. Demote it so a .sim that compiles for
        # Windows/Linux also compiles for macOS from the same source.
        "-Wno-c++11-narrowing",
        "-v",
    ]


def _run_build_blocking(req: MacBuildRequest, emit, stop: threading.Event) -> None:
    """Materialize the sandbox, compile, and stream the binary back. `emit` is
    a thread-safe callback (dict payload, is_final) that marshals onto the
    asyncio loop. Always terminates the stream with one is_final message."""
    build_dir = Path(tempfile.mkdtemp(prefix="macbuild_"))
    try:
        for f in req.files:
            # Defense in depth: keep reconstructed paths inside build_dir.
            rel = Path(f.path)
            if rel.is_absolute() or ".." in rel.parts:
                emit({"type": "error", "message": f"잘못된 파일 경로: {f.path!r}"}, True)
                return
            dst = build_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(base64.b64decode(f.content_b64))

        exe = build_dir / req.out_name
        argv = _build_argv(req, build_dir, exe)

        emit({"type": "progress", "message": "컴파일 시작"})
        proc = subprocess.Popen(
            argv, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True
        )
        stderr_buf: list[str] = []
        seen_cc1 = seen_ld = False
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_buf.append(line)
            if stop.is_set():
                proc.terminate()
                break
            if not seen_cc1 and _CC1_RE.search(line):
                seen_cc1 = True
                emit({"type": "progress", "message": "컴파일 중"})
            elif not seen_ld and _LD_RE.search(line):
                seen_ld = True
                emit({"type": "progress", "message": "링킹 중"})
        proc.wait()

        if stop.is_set():
            emit({"type": "error", "message": "빌드가 취소되었습니다."}, True)
            return
        if proc.returncode != 0:
            emit({"type": "error",
                  "message": "[macos] 빌드 실패\n" + "".join(stderr_buf)}, True)
            return

        data = exe.read_bytes()
        for i in range(0, len(data), CHUNK_BYTES):
            chunk = data[i:i + CHUNK_BYTES]
            emit({
                "type": "chunk",
                "seq": i // CHUNK_BYTES,
                "data": base64.b64encode(chunk).decode("ascii"),
            })
        emit({"type": "done"}, True)
    except Exception as e:  # noqa: BLE001
        emit({"type": "error", "message": f"macOS 빌드 오류: {e}"}, True)
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


async def _bridge_cancel(cancel_event: asyncio.Event, stop: threading.Event) -> None:
    try:
        await cancel_event.wait()
        stop.set()
    except asyncio.CancelledError:
        pass


async def handle(req: MacBuildRequest, send, cancel_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    stop = threading.Event()
    bridge = asyncio.create_task(_bridge_cancel(cancel_event, stop))

    def emit(msg: dict, is_final: bool = False) -> None:
        resp = MacBuildResponse(**msg)
        asyncio.run_coroutine_threadsafe(send(resp, is_final), loop).result()

    try:
        await asyncio.to_thread(_run_build_blocking, req, emit, stop)
    except Exception as e:  # noqa: BLE001
        try:
            await send(MacBuildResponse(type="error", message=str(e)), is_final=True)
        except Exception:
            pass
    finally:
        bridge.cancel()


def _server_url() -> str:
    url = os.environ.get("MACBUILD_SERVER_URL", DEFAULT_SERVER_URL)
    token = os.environ.get("MACBUILD_WORKER_TOKEN", "")
    if token:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={quote(token)}"
    return url


def main() -> None:
    url = _server_url()
    priority = int(os.environ.get("WORKER_PRIORITY", "0"))
    worker = Worker(MacBuildRequest, MacBuildResponse, priority=priority, handler=handle)
    # Don't log the token-bearing URL.
    shown = url.split("?", 1)[0]
    print(f"[macbuild-worker pid={os.getpid()}] connecting to {shown}  priority={priority}",
          flush=True)
    asyncio.run(worker.run(url))


if __name__ == "__main__":
    main()

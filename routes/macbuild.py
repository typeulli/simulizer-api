"""
macOS build dispatch.

This host has no local macOS toolchain. Instead an off-box build worker
(worker.py, run on a Mac) connects over WebSocket and produces the macOS
binary on demand. This module owns:

  - the worker-facing WebSocket endpoint `/workers/macbuild` (token-gated),
  - the `Server[MacBuildRequest, MacBuildResponse]` singleton, shared with
    routes/compile.py so the user-facing `/compile/build` can dispatch the
    macOS job to a connected worker,
  - `run_mac_build_job`, a *synchronous* bridge that lets compile.py's
    threadpool-run SSE generator drive an async worker stream and reassemble
    the returned binary.

Auth: a worker must present `?token=<MACBUILD_WORKER_TOKEN>`. When that env var
is unset (local dev) the endpoint accepts any connection and logs a warning —
in production a public endpoint without a token would let anyone receive user
source and return arbitrary binaries.
"""

import asyncio
import base64
import os
import queue
from pathlib import Path
from typing import Iterator, Optional

from fastapi import APIRouter, WebSocket

from workers.macbuild import MacBuildRequest, MacBuildResponse
from workers.server import NoWorkerAvailable, Server, WorkerDisconnected

router = APIRouter()

macbuild_server: Server[MacBuildRequest, MacBuildResponse] = Server(
    MacBuildRequest, MacBuildResponse
)

# The shared secret a connecting worker must echo back. Empty → no auth (dev).
WORKER_TOKEN = os.environ.get("MACBUILD_WORKER_TOKEN", "")

# The FastAPI event loop, captured at startup. compile.py's build SSE runs in a
# threadpool (sync generator) and needs this handle to drive the async server
# via run_coroutine_threadsafe.
_app_loop: Optional[asyncio.AbstractEventLoop] = None


def capture_app_loop() -> None:
    """Record the running event loop. Call once from the app lifespan."""
    global _app_loop
    _app_loop = asyncio.get_running_loop()


def get_app_loop() -> asyncio.AbstractEventLoop:
    if _app_loop is None:
        raise RuntimeError("app event loop not captured (lifespan not started?)")
    return _app_loop


def has_macbuild_worker() -> bool:
    """Whether at least one build worker is currently connected. Safe to call
    from any thread — it only reads the length of the worker list."""
    return macbuild_server.worker_count > 0


@router.websocket("/workers/macbuild")
async def macbuild_ws(ws: WebSocket):
    # Reject before accept() so a bad token gets an HTTP 403 handshake failure
    # rather than an open-then-closed socket.
    if WORKER_TOKEN:
        if ws.query_params.get("token") != WORKER_TOKEN:
            await ws.close(code=1008)  # policy violation
            return
    else:
        print("[macbuild] WARNING: MACBUILD_WORKER_TOKEN unset - worker endpoint "
              "is unauthenticated.", flush=True)
    await macbuild_server.handle_worker(ws)


class MacBuildError(Exception):
    """Raised by run_mac_build_job when the remote build fails / no worker /
    the worker disconnects. `detail` is surfaced to the user."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def run_mac_build_job(
    req: MacBuildRequest, exe_file: Path, *, loop: asyncio.AbstractEventLoop
) -> Iterator[str]:
    """Drive a remote macOS build from a synchronous (threadpool) context.

    Yields human-readable progress labels as the worker reports them. On
    success the assembled binary is written to `exe_file` after iteration
    completes; on any failure a `MacBuildError` is raised at the end of
    iteration. Intended to be consumed by compile.py's SSE generator.
    """
    out_q: "queue.Queue[Optional[tuple[str, Optional[str]]]]" = queue.Queue()
    chunks: list[bytes] = []

    async def driver() -> None:
        try:
            request_id = await macbuild_server.submit(req)
        except NoWorkerAvailable:
            out_q.put(("error", "macOS 빌드 서버(worker)가 연결되어 있지 않습니다."))
            out_q.put(None)
            return
        try:
            async for resp in macbuild_server.stream(request_id):
                if resp.type == "progress":
                    out_q.put(("progress", resp.message or ""))
                elif resp.type == "chunk":
                    if resp.data:
                        chunks.append(base64.b64decode(resp.data))
                elif resp.type == "error":
                    out_q.put(("error", resp.message or "macOS 빌드 실패"))
                    out_q.put(None)
                    return
                # "done" simply ends the stream (is_final on the wire).
        except WorkerDisconnected:
            out_q.put(("error", "macOS 빌드 worker 연결이 끊어졌습니다."))
            out_q.put(None)
            return
        out_q.put(("ok", None))
        out_q.put(None)

    fut = asyncio.run_coroutine_threadsafe(driver(), loop)

    error: Optional[str] = None
    while True:
        item = out_q.get()
        if item is None:
            break
        kind, payload = item
        if kind == "progress":
            yield payload or ""
        elif kind == "error":
            error = payload

    # driver() always enqueues a terminating None, so it has returned by now;
    # surface any unexpected exception it raised.
    try:
        fut.result()
    except Exception as e:  # noqa: BLE001
        if error is None:
            error = f"macOS 빌드 내부 오류: {e}"

    if error is not None:
        raise MacBuildError(error)

    exe_file.parent.mkdir(parents=True, exist_ok=True)
    exe_file.write_bytes(b"".join(chunks))

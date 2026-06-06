import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/lsp")

path_here = Path(__file__).parent.parent
LSP_ROOT = path_here / "lsp_root"
LSP_ROOT.mkdir(parents=True, exist_ok=True)

# Each WebSocket session gets its own subdirectory under SESSIONS_ROOT — so
# that didOpen/didChange events can materialize the user's bundle files on
# disk (clangd's preprocessor resolves `#include "src/util.hpp"` via the
# file system, not via open documents). A copy of the project-wide .clangd
# is dropped into each session dir at start.
SESSIONS_ROOT = LSP_ROOT / "sessions"
SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)

SYSTEM_INCLUDE_DIR = path_here / "bin" / "include"
SIMSTD_PATH = SYSTEM_INCLUDE_DIR / "simstd.hpp"

# URI namespace the client uses; the backend rewrites both ways so clangd sees
# the real per-session path while the client stays workspace-agnostic.
CLIENT_PREFIX = "file:///workspace"

# Second mapping pair for read-only system headers. clangd resolves
# #include "simstd.hpp" to bin/include/simstd.hpp via -I; we expose that to the
# client as a `simulizer:` scheme URI so monaco-vscode's FileSystemProvider can
# handle it without competing with the built-in `file:` handler. The frontend
# uses the same scheme on the wire so no client-side URI translation is needed.
SYSTEM_CLIENT_PREFIX = "simulizer:/"
_system_posix = SYSTEM_INCLUDE_DIR.resolve().as_posix()
SYSTEM_SERVER_PREFIX = "file:///" + quote(_system_posix.lstrip("/"), safe="/:") + "/"


def _resolve_clangd() -> str:
    path = shutil.which("clangd")
    if path:
        return path
    candidates = [
        r"C:\Program Files\LLVM\bin\clangd.exe",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin\clangd.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise RuntimeError("clangd executable not found in PATH")


def _make_server_prefix(session_dir: Path) -> str:
    """Build the file:/// URI prefix that points at this session's root."""
    posix = session_dir.resolve().as_posix()
    return "file:///" + quote(posix.lstrip("/"), safe="/:")


def _make_rewriters(server_prefix: str):
    """Return (server→client, client→server) string rewriters for this session.

    The system-header mapping is shared across sessions; the workspace mapping
    is per-session (so files written to one session's tmpdir don't leak into
    another's URI namespace).
    """

    def server_to_client(text: str) -> str:
        text = text.replace(SYSTEM_SERVER_PREFIX, SYSTEM_CLIENT_PREFIX)
        return text.replace(server_prefix, CLIENT_PREFIX)

    def client_to_server(text: str) -> str:
        text = text.replace(SYSTEM_CLIENT_PREFIX, SYSTEM_SERVER_PREFIX)
        return text.replace(CLIENT_PREFIX, server_prefix)

    return server_to_client, client_to_server


# Match `"uri": "file:///<session_dir>/some/path.cpp"` after client-to-server
# rewrite. We use this to extract the on-disk path that an LSP notification
# refers to, so we can mirror its text content to disk.
def _extract_workspace_path(server_uri: str, server_prefix: str) -> Path | None:
    if not server_uri.startswith(server_prefix + "/"):
        return None
    rel = unquote(server_uri[len(server_prefix) + 1:])
    # Defense in depth: refuse traversal that climbs above the session root.
    if ".." in rel.split("/"):
        return None
    return Path(rel)


# clangd's preprocessor resolves `#include "..."` against the filesystem; if
# we let user-controlled escapes through, the LSP becomes a read-anything-
# on-disk oracle (diagnostics + go-to-definition leak content). We neuter
# offending lines with `#error` so clangd reports the issue back to the user
# instead of silently resolving the include.
#
# Intra-project relative paths like `src/foo.cpp` doing `#include "../bar.hpp"`
# are *allowed*; we only block paths that, once resolved against the file's
# own directory, step above the project root.
_INCLUDE_RE = re.compile(r'^(\s*#\s*include(?:_next)?\s*)[<"]([^>"]+)[>"]', re.MULTILINE)


def _include_is_unsafe(file_dir: str, inc: str) -> bool:
    if not inc:
        return True
    if inc.startswith("/") or inc.startswith("\\"):
        return True
    if len(inc) >= 2 and inc[1] == ":":
        return True
    inc_parts = re.split(r"[/\\]", inc)
    dir_parts = file_dir.split("/") if file_dir else []
    stack = list(dir_parts)
    for p in inc_parts:
        if p == "" or p == ".":
            continue
        if p == "..":
            if not stack:
                return True
            stack.pop()
            continue
        stack.append(p)
    return False


def _pos_to_offset(text: str, line: int, character: int) -> int:
    """Translate an LSP {line, character} position into a string offset
    inside `text`. LSP uses UTF-16 code units for `character`, but Python
    strings index by code points; for ASCII-only C++ source they are
    identical, which covers our actual use (looking at `#include` lines)."""
    offset = 0
    cur_line = 0
    while cur_line < line:
        nl = text.find("\n", offset)
        if nl == -1:
            return len(text)
        offset = nl + 1
        cur_line += 1
    nl = text.find("\n", offset)
    line_end = nl if nl != -1 else len(text)
    return min(offset + character, line_end)


def _apply_change(text: str, change: dict) -> str:
    """Apply a single LSP TextDocumentContentChangeEvent to `text`. Supports
    both incremental (range present) and full-document (range absent) forms."""
    if "text" not in change:
        return text
    new = change["text"]
    rng = change.get("range")
    if rng is None:
        return new
    try:
        s = rng["start"]; e = rng["end"]
        start = _pos_to_offset(text, s["line"], s["character"])
        end = _pos_to_offset(text, e["line"], e["character"])
    except Exception:
        return text
    return text[:start] + new + text[end:]


_INCLUDE_PREFIX_RE = re.compile(r'^\s*#\s*include(?:_next)?\s*[<"]')


def _handle_include_completion(
    msg: dict | None, session_dir: Path, documents: dict[str, str],
) -> list[dict] | None:
    """If `msg` is an `textDocument/completion` request whose cursor sits
    inside an `#include "..."` (or `<...>`) directive, return a list of
    LSP CompletionItems we want to answer the request with — namely the
    user's own bundle headers, filtered to .hpp. Returning None means the
    request should be forwarded to clangd unchanged.

    Suppressing clangd here keeps it from enumerating system include paths
    (Windows SDK, libc, etc.) and leaking that file layout to the client.
    """
    if not isinstance(msg, dict):
        return None
    if msg.get("method") != "textDocument/completion":
        return None
    params = msg.get("params") or {}
    doc = params.get("textDocument") or {}
    pos = params.get("position") or {}
    line_num = pos.get("line")
    cursor_ch = pos.get("character")
    if not isinstance(line_num, int) or line_num < 0:
        return None
    if not isinstance(cursor_ch, int) or cursor_ch < 0:
        return None
    server_prefix = _make_server_prefix(session_dir)
    rel = _extract_workspace_path(doc.get("uri") or "", server_prefix)
    if rel is None:
        return None
    text = documents.get(rel.as_posix())
    if text is None:
        return None
    lines = text.split("\n")
    if line_num >= len(lines):
        return None
    line_text = lines[line_num]
    m = _INCLUDE_PREFIX_RE.match(line_text)
    if m is None:
        return None
    # Replacement range = from right-after-delimiter to current cursor. clangd
    # would normally compute this from the partial path; we mirror that so
    # the typed-so-far prefix is replaced, not appended to.
    path_start = m.end()
    self_path = rel.as_posix()
    items: list[dict] = []
    for p in sorted(documents.keys()):
        if p == self_path:
            continue
        if not p.lower().endswith(".hpp"):
            continue
        items.append({
            "label": p,
            "kind": 17,  # CompletionItemKind.File
            "filterText": p,
            "insertText": p,
            "textEdit": {
                "range": {
                    "start": {"line": line_num, "character": path_start},
                    "end":   {"line": line_num, "character": cursor_ch},
                },
                "newText": p,
            },
        })
    return items


def _neuter_unsafe_includes(rel_path: str, text: str) -> str:
    slash = rel_path.replace("\\", "/").rfind("/")
    file_dir = rel_path[:slash] if slash >= 0 else ""

    def repl(m: "re.Match[str]") -> str:
        p = m.group(2).strip()
        if _include_is_unsafe(file_dir, p):
            return f'#error "External include path is not allowed: {p}"'
        return m.group(0)
    return _INCLUDE_RE.sub(repl, text)


def _mirror_to_disk(session_dir: Path, server_msg: dict, documents: dict[str, str]):
    """Write text-document state changes from an LSP message to disk so that
    clangd's preprocessor can resolve `#include "..."` between user files,
    and keep an in-memory mirror that other helpers (completion suppression)
    can consult without re-reading the disk.

    Handles didOpen/didChange (both incremental and full-document forms) and
    didClose. Best-effort — failures here shouldn't break LSP traffic.
    """
    try:
        method = server_msg.get("method")
        params = server_msg.get("params") or {}
        doc = params.get("textDocument") or {}
        uri = doc.get("uri") or ""
        server_prefix = _make_server_prefix(session_dir)
        rel = _extract_workspace_path(uri, server_prefix)
        if rel is None:
            return
        target = session_dir / rel
        rel_str = rel.as_posix()
        if method == "textDocument/didOpen":
            text = _neuter_unsafe_includes(rel_str, doc.get("text", ""))
            documents[rel_str] = text
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        elif method == "textDocument/didChange":
            current = documents.get(rel_str, "")
            for ch in params.get("contentChanges") or []:
                if isinstance(ch, dict):
                    current = _apply_change(current, ch)
            # Re-neuter on every change in case the user just typed a `..`
            # include — we don't want stale escapes leaking through.
            current = _neuter_unsafe_includes(rel_str, current)
            documents[rel_str] = current
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(current, encoding="utf-8")
        elif method == "textDocument/didClose":
            # Leave the file on disk and the in-memory snapshot so cross-
            # file includes still resolve after a tab close. The whole
            # session dir is removed when the WebSocket disconnects.
            pass
    except Exception as e:  # noqa: BLE001
        print(f"[lsp] mirror failed: {type(e).__name__}: {e}", flush=True)


def _read_lsp_frame(stream) -> bytes | None:
    """Blocking read of one LSP frame (Content-Length header + body) from a
    binary stream. Returns the JSON body bytes, or None on EOF."""
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        line = line.rstrip(b"\r\n")
        if line == b"":
            break
        if b":" in line:
            k, _, v = line.partition(b":")
            headers[k.decode("ascii").strip().lower()] = v.decode("ascii").strip()
    length_str = headers.get("content-length")
    if length_str is None:
        return b""
    length = int(length_str)
    body = b""
    while len(body) < length:
        chunk = stream.read(length - len(body))
        if not chunk:
            return None
        body += chunk
    return body


async def _stdout_to_ws(proc: subprocess.Popen, ws: WebSocket, server_to_client):
    """Pump frames from clangd stdout (blocking I/O in a worker thread)
    to the WebSocket."""
    loop = asyncio.get_running_loop()
    while True:
        body = await loop.run_in_executor(None, _read_lsp_frame, proc.stdout)
        if body is None:
            print("[lsp] clangd stdout closed", flush=True)
            return
        if not body:
            continue
        text = server_to_client(body.decode("utf-8"))
        try:
            await ws.send_text(text)
        except Exception as e:
            print(f"[lsp] ws.send_text failed: {type(e).__name__}: {e}", flush=True)
            return


async def _ws_to_stdin(proc: subprocess.Popen, ws: WebSocket, session_dir: Path, client_to_server):
    """Read JSON-RPC messages from the WebSocket and write them to clangd
    stdin framed with Content-Length headers (blocking write in a worker
    thread so the event loop never blocks). Along the way we mirror open/
    change notifications to disk under session_dir."""
    loop = asyncio.get_running_loop()
    assert proc.stdin is not None
    # rel posix path → current text. Kept in sync with clangd by
    # _mirror_to_disk so request interceptors can inspect document state
    # without re-reading the disk on every keystroke.
    documents: dict[str, str] = {}
    while True:
        try:
            message = await ws.receive_text()
        except WebSocketDisconnect:
            print("[lsp] ws client disconnected", flush=True)
            return
        except Exception as e:
            print(f"[lsp] ws.receive_text failed: {type(e).__name__}: {e}", flush=True)
            return
        try:
            parsed = json.loads(message)
        except Exception:
            logger.warning("LSP client sent non-JSON frame; ignoring")
            continue
        rewritten = client_to_server(message)
        # Mirror after rewrite so the URIs in the message already point at
        # the session directory.
        try:
            mirrored = json.loads(rewritten)
            _mirror_to_disk(session_dir, mirrored, documents)
        except Exception:
            mirrored = None

        # Suppress `#include "..."` completion at the bridge so clangd
        # never gets a chance to enumerate header files in `-I` paths and
        # leak them back to the client. We answer the request ourselves
        # with the bundle's own `.hpp` files instead.
        include_items = _handle_include_completion(mirrored, session_dir, documents)
        if include_items is not None:
            req_id = parsed.get("id") if isinstance(parsed, dict) else None
            if req_id is not None:
                response = json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"items": include_items, "isIncomplete": False},
                })
                try:
                    await ws.send_text(response)
                except Exception as e:
                    print(f"[lsp] ws.send_text (include suppression) failed: {type(e).__name__}: {e}", flush=True)
                    return
            continue

        encoded = rewritten.encode("utf-8")
        header = f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii")

        def _write(data: bytes):
            try:
                assert proc.stdin is not None
                proc.stdin.write(data)
                proc.stdin.flush()
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                print(f"[lsp] write to clangd stdin failed: {type(e).__name__}: {e}", flush=True)
                return False
            return True

        ok = await loop.run_in_executor(None, _write, header + encoded)
        if not ok:
            return


async def _stderr_to_console(proc: subprocess.Popen):
    """Surface clangd stderr on the backend console so startup errors,
    missing-DLL warnings, and crash reports don't vanish silently."""
    loop = asyncio.get_running_loop()
    assert proc.stderr is not None
    while True:
        line = await loop.run_in_executor(None, proc.stderr.readline)
        if not line:
            return
        try:
            text = line.decode("utf-8", errors="replace").rstrip()
        except Exception:
            continue
        if text:
            print(f"[clangd] {text}", flush=True)


@router.get("/simstd.hpp", response_class=PlainTextResponse)
async def lsp_simstd_header():
    """Serve the simstd.hpp system header to the editor so it can show a
    read-only tab when the user navigates into it via go-to-definition."""
    if not SIMSTD_PATH.is_file():
        raise HTTPException(status_code=404, detail="simstd.hpp not found")
    return PlainTextResponse(
        SIMSTD_PATH.read_text(encoding="utf-8"),
        media_type="text/x-c++hdr; charset=utf-8",
    )


_SAFE_SESSION_RE = re.compile(r"^[A-Za-z0-9-]+$")


_CLANGD_TEMPLATE = """CompileFlags:
  Add:
    - -std=c++17
    - -xc++
    - -D__EMSCRIPTEN__
    - -fno-exceptions
    - -I{system_include}
    - -I.
  Compiler: clang++

Diagnostics:
  UnusedIncludes: None
  Suppress:
    - pp_file_not_found
"""


def _make_session_dir() -> Path:
    sid = uuid.uuid4().hex
    if not _SAFE_SESSION_RE.match(sid):
        raise RuntimeError("unexpected session id format")
    d = SESSIONS_ROOT / sid
    d.mkdir(parents=True, exist_ok=False)
    # The template at LSP_ROOT/.clangd hard-codes a relative include path
    # (`../bin/include`) that assumes .clangd sits at LSP_ROOT. With per-
    # session subdirs we instead write a config whose system-include is
    # absolute — that way clangd resolves simstd.hpp regardless of how deep
    # the session dir lives. Use forward slashes (as_posix) because clangd's
    # YAML config parser handles `\` inconsistently on Windows.
    (d / ".clangd").write_text(
        _CLANGD_TEMPLATE.format(system_include=SYSTEM_INCLUDE_DIR.resolve().as_posix()),
        encoding="utf-8",
    )
    return d


def _clangd_args(clangd_path: str) -> list[str]:
    return [
        clangd_path,
        "--background-index=true",
        "--pch-storage=memory",
        "--header-insertion=never",
        "--clang-tidy=false",
        "--completion-style=bundled",
        "--limit-results=20",
        "-j=4",
        "--log=error",
    ]


def _spawn_clangd(session_dir: Path) -> subprocess.Popen:
    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    return subprocess.Popen(
        _clangd_args(_resolve_clangd()),
        cwd=str(session_dir),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creation_flags,
    )


# --- warm clangd pool -------------------------------------------------------
# A WebSocket connect otherwise pays the full cold cost of launching clangd:
# loading the (large) clangd.exe + its DLLs, plus an AV scan on Windows. That
# is the dominant "connection is slow" latency. We keep a few clangd processes
# pre-spawned and idle (a clangd does nothing until it receives `initialize`,
# so an idle one is cheap and holds no preamble yet) and hand one off on
# connect, refilling the pool in the background.
#
# NOTE: this removes process-launch from the connect path and lets the
# preamble build start sooner, but it does NOT pre-build the STL preamble:
# clangd's preamble is keyed per main-file path and can't be built before the
# client's own `initialize` arrives, so each session still parses the STL on
# its first request. Eliminating that would need a prebuilt PCH (a separate
# change).

CLANGD_POOL_SIZE = int(os.environ.get("CLANGD_POOL_SIZE", "2"))


class _PooledClangd:
    __slots__ = ("proc", "session_dir")

    def __init__(self, proc: subprocess.Popen, session_dir: Path):
        self.proc = proc
        self.session_dir = session_dir


_pool: list[_PooledClangd] = []
_pool_lock = asyncio.Lock()
_replenishing = False


def _spawn_pooled() -> _PooledClangd:
    """Blocking: create a session dir and launch a clangd bound to it. The
    process and its session dir travel together (cwd + URI namespace)."""
    session_dir = _make_session_dir()
    proc = _spawn_clangd(session_dir)
    print(f"[lsp] pooled clangd pid={proc.pid} cwd={session_dir}", flush=True)
    return _PooledClangd(proc, session_dir)


def _discard_pooled(entry: _PooledClangd):
    """Blocking: terminate a pooled process we won't use and remove its dir."""
    try:
        if entry.proc.poll() is None:
            entry.proc.terminate()
            try:
                entry.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                entry.proc.kill()
                entry.proc.wait()
    except Exception:  # noqa: BLE001
        pass
    shutil.rmtree(entry.session_dir, ignore_errors=True)


async def _replenish_pool():
    """Top the pool back up to CLANGD_POOL_SIZE, one spawn at a time. Guarded
    so concurrent connects don't overshoot the target."""
    global _replenishing
    if _replenishing or CLANGD_POOL_SIZE <= 0:
        return
    _replenishing = True
    loop = asyncio.get_running_loop()
    try:
        while True:
            async with _pool_lock:
                need = CLANGD_POOL_SIZE - len(_pool)
            if need <= 0:
                return
            try:
                entry = await loop.run_in_executor(None, _spawn_pooled)
            except Exception as e:  # noqa: BLE001
                print(f"[lsp] pool spawn failed: {type(e).__name__}: {e}", flush=True)
                return
            async with _pool_lock:
                if len(_pool) < CLANGD_POOL_SIZE:
                    _pool.append(entry)
                    entry = None
            if entry is not None:  # raced past the target; drop it
                await loop.run_in_executor(None, _discard_pooled, entry)
    finally:
        _replenishing = False


def _schedule_replenish():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_replenish_pool())


async def _acquire_clangd() -> _PooledClangd:
    """Return a ready clangd + its session dir. Pops a warm one if available;
    otherwise spawns synchronously (off the event loop). Either way the pool
    is refilled in the background. May raise if clangd can't be launched."""
    loop = asyncio.get_running_loop()
    async with _pool_lock:
        while _pool:
            entry = _pool.pop()
            if entry.proc.poll() is None:
                _schedule_replenish()
                return entry
            # A pooled process died before use — clean it and try the next.
            await loop.run_in_executor(None, _discard_pooled, entry)
    entry = await loop.run_in_executor(None, _spawn_pooled)
    _schedule_replenish()
    return entry


async def warm_clangd_pool():
    """Fill the pool at startup so the first connect is fast too. No-op (with
    a log) if pooling is disabled or clangd isn't found."""
    if CLANGD_POOL_SIZE <= 0:
        return
    try:
        _resolve_clangd()
    except RuntimeError as e:
        print(f"[lsp] clangd not found; warm pool disabled: {e}", flush=True)
        return
    await _replenish_pool()


async def shutdown_clangd_pool():
    """Terminate any still-pooled processes and remove their session dirs."""
    loop = asyncio.get_running_loop()
    async with _pool_lock:
        entries = list(_pool)
        _pool.clear()
    for e in entries:
        await loop.run_in_executor(None, _discard_pooled, e)


def _sweep_stale_sessions_blocking() -> int:
    """Remove every session dir under SESSIONS_ROOT. Safe to call only at
    startup: no session is active yet, so anything here is an orphan left by
    a prior hard-kill (graceful disconnect already rmtree's its own dir).
    Returns the number of dirs removed."""
    removed = 0
    for child in SESSIONS_ROOT.iterdir():
        if not child.is_dir():
            continue
        shutil.rmtree(child, ignore_errors=True)
        # ignore_errors hides Windows .cache locks; count only true removals.
        if not child.exists():
            removed += 1
    return removed


async def sweep_stale_sessions():
    """Clear orphaned clangd session dirs at startup. MUST run before
    warm_clangd_pool, which creates fresh dirs we don't want swept."""
    loop = asyncio.get_running_loop()
    try:
        removed = await loop.run_in_executor(None, _sweep_stale_sessions_blocking)
    except Exception as e:  # noqa: BLE001
        print(f"[lsp] session sweep failed: {type(e).__name__}: {e}", flush=True)
        return
    if removed:
        print(f"[lsp] swept {removed} stale session dir(s)", flush=True)


@router.websocket("/cpp")
async def lsp_cpp(ws: WebSocket):
    await ws.accept()
    client = f"{ws.client.host if ws.client else '?'}"
    print(f"[lsp] /cpp accept from {client}", flush=True)

    try:
        pooled = await _acquire_clangd()
    except Exception as e:  # noqa: BLE001 — clangd missing or launch failure
        print(f"[lsp] clangd unavailable: {type(e).__name__}: {e}", flush=True)
        await ws.close(code=1011, reason="clangd unavailable")
        return

    proc = pooled.proc
    session_dir = pooled.session_dir
    server_prefix = _make_server_prefix(session_dir)
    server_to_client, client_to_server = _make_rewriters(server_prefix)
    print(f"[lsp] session dir: {session_dir} (clangd pid={proc.pid})", flush=True)
    # The three pumps don't have symmetric lifetimes: _ws_to_stdin returns as
    # soon as the client disconnects, but _stdout_to_ws and _stderr_to_console
    # are blocked on executor threads doing pipe reads that only unblock when
    # clangd dies. Using asyncio.gather here would wait for ALL three — and
    # since clangd doesn't exit on its own, gather would hang forever, the
    # finally block never runs, and clangd + 2 executor threads leak per
    # disconnect. Once the default ThreadPoolExecutor (~16 threads) fills up
    # with zombie readers, new sessions can't get a thread for their own
    # _stdout_to_ws and the LSP appears broken until the server restarts.
    #
    # Use FIRST_COMPLETED so a single disconnect (or any pump error) triggers
    # immediate clangd termination; the pipe EOF then unblocks the other two
    # executor threads.
    tasks = [
        asyncio.create_task(_stdout_to_ws(proc, ws, server_to_client)),
        asyncio.create_task(_ws_to_stdin(proc, ws, session_dir, client_to_server)),
        asyncio.create_task(_stderr_to_console(proc)),
    ]
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            exc = t.exception()
            if exc is not None:
                print(f"[lsp] bridge pump raised: {type(exc).__name__}: {exc}", flush=True)
    except Exception as e:
        print(f"[lsp] bridge error: {type(e).__name__}: {e}", flush=True)
    finally:
        print(f"[lsp] tearing down pid={proc.pid} returncode={proc.poll()}", flush=True)
        # Kill clangd FIRST. Until its pipes close, the executor threads behind
        # _stdout_to_ws / _stderr_to_console stay parked on .readline()/.read()
        # and the surrounding tasks can't actually finish even after .cancel().
        if proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: proc.wait(timeout=3)
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        # Now cancel and drain the remaining pumps. With clangd gone the pipe
        # reads return EOF, the executor threads finish, and these awaits
        # complete (CancelledError on the cancelled ones, normally otherwise).
        for t in tasks:
            if not t.done():
                t.cancel()
        for t in tasks:
            try:
                await t
            except BaseException:
                pass
        try:
            await ws.close()
        except Exception:
            pass
        # Clean up the session's working dir; safe to ignore failures (e.g.,
        # the index subdir being held by a transient child process on Windows).
        shutil.rmtree(session_dir, ignore_errors=True)

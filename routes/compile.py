import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from block2cpp import cppize


router = APIRouter(prefix="/compile")

path_here = Path(__file__).parent.parent
path_temp = path_here / "temp"
path_bin = path_here / "bin"


ALLOWED_BUNDLE_EXTENSIONS = (".cpp", ".hpp")


# Headers the build needs to see when compiling user code. We stage them into
# a per-build `_system/` directory and `-I` only that — so user code resolves
# `#include "simstd.hpp"` to the full (inline-implementation) version that
# `debug_log` / `show_mat` need, *without* the rest of backend-api/ becoming
# reachable via `#include "ocr.py"`, `#include "key/..."`, etc.
#
# The LSP side keeps using `bin/include/simstd.hpp` (declaration-only) for
# its own `.clangd` -I, which is intentionally minimal for indexing speed.
_BUILD_SYSTEM_FILES = [
    (path_here / "simstd.hpp",            "simstd.hpp"),
    (path_here / "lib" / "banner.hpp",    "lib/banner.hpp"),
    (path_here / "lib" / "httplib.h",     "lib/httplib.h"),
]


def _stage_build_system(system_dir: Path):
    """Copy the headers above into `system_dir`. Skips entries that don't
    exist on disk so a partial install doesn't break compilation when the
    user's code doesn't actually need the missing piece."""
    for src, rel in _BUILD_SYSTEM_FILES:
        if not src.is_file():
            continue
        dst = system_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _validate_segment(name: str):
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise HTTPException(status_code=400, detail=f"Invalid path segment: {name!r}")


# Matches `#include "..."` and `#include <...>`. Captures the path between
# the delimiters. Note this is *best-effort textual* defence — it won't catch
# macro-expanded includes (e.g. `#define X "..."` followed by `#include X`).
# A defence-in-depth sandbox would still be needed for hostile inputs, but
# this raises the bar to "needs to fight the preprocessor" which is enough
# for a casual playground.
_INCLUDE_RE = re.compile(r'^\s*#\s*include(?:_next)?\s*[<"]([^>"]+)[>"]', re.MULTILINE)


def _classify_include(file_dir: str, inc: str) -> str | None:
    """Return a human-readable reason if `inc`, resolved against the directory
    of the including file (`file_dir`, relative to the project root), escapes
    the project root. Otherwise None.

    Legitimate intra-project uses like `src/foo.cpp` doing `#include "../common.hpp"`
    pass — they resolve to `common.hpp` at the project root.
    """
    if not inc:
        return "empty include path"
    # Reject absolute / drive-letter / UNC paths outright. These have no
    # in-project interpretation.
    if inc.startswith("/") or inc.startswith("\\"):
        return f"absolute include path not allowed: {inc!r}"
    if len(inc) >= 2 and inc[1] == ":":
        return f"absolute include path not allowed: {inc!r}"
    # Tokenize on BOTH separators so a user can't smuggle `..` past us by
    # mixing slashes (`src\..\..\..\etc`).
    inc_parts = re.split(r"[/\\]", inc)
    dir_parts = file_dir.split("/") if file_dir else []
    stack = list(dir_parts)
    for p in inc_parts:
        if p == "" or p == ".":
            continue
        if p == "..":
            if not stack:
                # Would step above project root.
                return f"include escapes project root: {inc!r}"
            stack.pop()
            continue
        stack.append(p)
    return None


def _check_user_source(rel_path: str, content: str):
    """Reject user-supplied translation units that try to escape the
    project sandbox via `#include`. `rel_path` is the file's path relative
    to the project root (e.g. `"src/util.cpp"`)."""
    slash = rel_path.replace("\\", "/").rfind("/")
    file_dir = rel_path[:slash] if slash >= 0 else ""
    for m in _INCLUDE_RE.finditer(content):
        reason = _classify_include(file_dir, m.group(1).strip())
        if reason is not None:
            raise HTTPException(
                status_code=400,
                detail=f"{rel_path}: {reason}",
            )


def _materialize_tree(nodes: list, root: Path, parent_rel: str = ""):
    """Write the bundle's tree into `root`. Raises HTTPException on a malformed
    node (unknown type, illegal name, file extension outside the allowlist)
    or on a user file whose `#include` lines escape the project sandbox.
    """
    for node in nodes:
        if not isinstance(node, dict):
            raise HTTPException(status_code=400, detail="Tree node must be an object")
        name = node.get("name", "")
        _validate_segment(name)
        kind = node.get("type")
        target = root / name
        rel = f"{parent_rel}/{name}" if parent_rel else name
        if kind == "file":
            ext = Path(name).suffix.lower()
            if ext not in ALLOWED_BUNDLE_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Disallowed file extension: {name!r}",
                )
            content = node.get("content", "")
            _check_user_source(rel, content)
            target.write_text(content, encoding="utf-8")
        elif kind == "folder":
            target.mkdir(parents=True, exist_ok=True)
            _materialize_tree(node.get("contents", []), target, rel)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tree node type: {kind!r}")


def _resolve_entry(tree: list, entry: str, root: Path) -> Path:
    if not entry:
        raise HTTPException(status_code=400, detail="entry is required")
    parts = entry.split("/")
    for p in parts:
        _validate_segment(p)
    ext = Path(parts[-1]).suffix.lower()
    if ext != ".cpp":
        raise HTTPException(status_code=400, detail="entry must be a .cpp file")
    p = root / Path(*parts)
    if not p.is_file():
        raise HTTPException(status_code=400, detail=f"entry not found in tree: {entry!r}")
    return p


class CompileRequest(BaseModel):
    lang: str
    code: str
    main_fn: str = "main"


SUPPORTED_LANGS = ("cpp", "py", "js")


@router.post("")
def compile(body: CompileRequest):
    if body.lang not in SUPPORTED_LANGS:
        raise HTTPException(status_code=400, detail="Unsupported language")
    try:
        result = cppize(body.code, target=body.lang, main_fn_name=body.main_fn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"result": result}


class BuildRequest(BaseModel):
    # lang="blocks": legacy Blockly JSON in `code`.
    # lang="cpp":    multi-file bundle in `tree` + `entry`.
    lang: str = "blocks"
    code: Optional[str] = None
    tree: Optional[list] = None
    entry: Optional[str] = None


def _write_status(out_dir: Path, status: str, **extra):
    (out_dir / "status.json").write_text(
        json.dumps({"status": status, **extra}), encoding="utf-8"
    )

def _read_status(out_dir: Path) -> dict:
    try:
        return json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


BUILD_STAGES = [
    (re.compile(r'cc1plus\.exe'),           "컴파일 시작"),
    (re.compile(r'GNU C\+\+\d+.*version'),  "컴파일러 초기화"),
    (re.compile(r'End of search list\.'),   "헤더 탐색 완료"),
    (re.compile(r'as\.exe'),                "어셈블 중"),
    (re.compile(r'GNU assembler version'),  "어셈블 시작"),
    (re.compile(r'collect2\.exe'),          "링킹 중"),
]
BUILD_TOTAL = len(BUILD_STAGES) + 1


def _stream_build(file_uuid: str, out_dir: Path, cpp_file: Path, exe_file: Path, project_root: Path, system_dir: Path):
    res_file = path_bin / "resource.o"
    bin_file = path_bin / "binary_data.o"
    BUILD_COMMAND = [
        "C:/mingw64/bin/g++.exe",
        str(cpp_file),
        str(res_file), str(bin_file),
        "-std=c++17",
        "-o", str(exe_file),
        # Restricted to a per-build staging dir so user code cannot reach
        # backend Python sources, the auth `key/` dir, prompts, etc. via
        # #include. Only the headers in _BUILD_SYSTEM_FILES land here.
        f"-I{system_dir}",
        f"-I{project_root}",
        "-O3",
        "-pthread", "-lpdh", "-lws2_32", "-lgdi32",
        "-static-libgcc", "-static-libstdc++",
        "-Wl,-Bstatic", "-lstdc++", "-lpthread", "-Wl,-Bdynamic",
        "-v",
    ]

    yield f"data: {json.dumps({'uuid': file_uuid, 'step': 0, 'total': BUILD_TOTAL, 'message': '빌드 준비'})}\n\n"

    proc = subprocess.Popen(
        BUILD_COMMAND,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
    )

    step = 0
    stderr_buf = []
    assert proc.stderr is not None
    for line in proc.stderr:
        stderr_buf.append(line)
        for idx, (pattern, msg) in enumerate(BUILD_STAGES, start=1):
            if idx > step and pattern.search(line):
                step = idx
                yield f"data: {json.dumps({'step': step, 'total': BUILD_TOTAL, 'message': msg})}\n\n"
                break

    proc.wait()
    if proc.returncode != 0:
        stderr_text = "".join(stderr_buf)
        _write_status(out_dir, "error", detail=stderr_text)
        yield f"event: error\ndata: {json.dumps({'detail': stderr_text})}\n\n"
        return

    _write_status(out_dir, "ready")
    yield f"data: {json.dumps({'step': BUILD_TOTAL, 'total': BUILD_TOTAL, 'message': '완료'})}\n\n"
    yield f"event: done\ndata: {json.dumps({'uuid': file_uuid})}\n\n"


_MAIN_SHIM = """
int main() {
    simulizer::init();
    std::thread t([]() {
        worker();
        std::cout << "\\nPress enter to exit..." << std::flush;
        std::cin.get();
        simulizer::svr.stop();
    });
    simulizer::svr.listen("localhost", 8080);
    t.join();
    return 0;
}
"""


@router.post("/build")
def build(body: BuildRequest):
    file_uuid = str(uuid.uuid4())
    out_dir = path_temp / file_uuid
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_status(out_dir, "compiling")

    # The build root is where we drop user files and the main shim; for the
    # blocks pathway we just write a single output.cpp at the root. For the
    # cpp bundle pathway we materialize the whole tree and append the main
    # shim to a copy of the entry file. Either way the build command sees
    # `output.cpp` as the translation unit and -I<project_root> resolves
    # `#include "..."`s for sibling/nested headers.
    project_root = out_dir / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    system_dir = out_dir / "_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    _stage_build_system(system_dir)

    if body.lang == "blocks":
        if not body.code:
            raise HTTPException(status_code=400, detail="code is required for blocks build")
        try:
            cpp_code = cppize(body.code, "worker")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        full_source = '#include <iostream>\n' + cpp_code + _MAIN_SHIM
    elif body.lang == "cpp":
        if not isinstance(body.tree, list) or not body.entry:
            raise HTTPException(status_code=400, detail="tree and entry are required for cpp build")
        _materialize_tree(body.tree, project_root)
        entry_path = _resolve_entry(body.tree, body.entry, project_root)
        original = entry_path.read_text(encoding="utf-8")
        full_source = '#include <iostream>\n' + original + _MAIN_SHIM
        # Overwrite the entry file with the wrapped source so include search
        # paths and relative includes still resolve from its original location.
        entry_path.write_text(full_source, encoding="utf-8")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported lang: {body.lang}")

    if body.lang == "blocks":
        cpp_file = project_root / "output.cpp"
        cpp_file.write_text(full_source, encoding="utf-8")
    else:
        cpp_file = entry_path
    exe_file = out_dir / "output.exe"

    return StreamingResponse(
        _stream_build(file_uuid, out_dir, cpp_file, exe_file, project_root, system_dir),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class EmccRequest(BaseModel):
    tree: list
    entry: str


_WORKER_ENTRY_SHIM = (
    '\n#include <type_traits>\n'
    'template<typename F> int __sim_invoke_or_zero(F f) {\n'
    '    if constexpr (std::is_void_v<decltype(f())>) { f(); return 0; }\n'
    '    else { return f(); }\n'
    '}\n'
    'extern "C" __attribute__((used)) int __sim_worker_entry() { return __sim_invoke_or_zero(worker); }\n'
)


@router.post("/emcc")
def emcc(body: EmccRequest):
    file_uuid = str(uuid.uuid4())
    out_dir = path_temp / file_uuid
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        project_root = out_dir / "project"
        project_root.mkdir(parents=True, exist_ok=True)
        system_dir = out_dir / "_system"
        system_dir.mkdir(parents=True, exist_ok=True)
        _stage_build_system(system_dir)
        _materialize_tree(body.tree, project_root)
        entry_path = _resolve_entry(body.tree, body.entry, project_root)

        # C++ name mangling renames `worker()` to `_Z6workerv`, so append an
        # extern "C" shim with a fixed symbol the worker can look up. The shim
        # is added to the entry file in-place so its include search paths and
        # any relative `#include "..."`s still resolve correctly.
        entry_path.write_text(
            entry_path.read_text(encoding="utf-8") + _WORKER_ENTRY_SHIM,
            encoding="utf-8",
        )

        wasm_file = out_dir / "user.wasm"

        # Standalone WASM: libc/libc++ statically linked into the module,
        # only `env` (our __sim_* bridges) and `wasi_snapshot_preview1` imports
        # need to be supplied by the worker at instantiate time.
        EMCC_COMMAND = [
            "em++",
            str(entry_path),
            "-std=c++17",
            "-o", str(wasm_file),
            f"-I{system_dir}",
            f"-I{project_root}",
            "-O3",
            "-sSTANDALONE_WASM=1",
            "-sERROR_ON_UNDEFINED_SYMBOLS=0",
            "-sEXPORTED_FUNCTIONS=['___sim_worker_entry']",
            "-fno-exceptions",
            "-Wl,--allow-undefined",
        ]

        proc = subprocess.run(
            EMCC_COMMAND,
            capture_output=True,
            text=True,
            shell=True,
        )
        if proc.returncode != 0:
            raise HTTPException(status_code=422, detail=proc.stderr)

        return Response(
            content=wasm_file.read_bytes(),
            media_type="application/wasm",
        )
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


@router.get("/build/download/{uuid}")
def build_download(uuid: str):
    out_dir = path_temp / uuid
    status = _read_status(out_dir)

    if not status:
        raise HTTPException(status_code=404, detail="File not found")
    if status.get("status") == "compiling":
        raise HTTPException(status_code=409, detail="Still compiling")
    if status.get("status") == "error":
        raise HTTPException(status_code=422, detail=status.get("detail", "Compile error"))
    if status.get("status") != "ready":
        raise HTTPException(status_code=404, detail="File not found")

    exe_file = out_dir / "output.exe"
    if not exe_file.exists():
        raise HTTPException(status_code=404, detail="File not found")

    content = exe_file.read_bytes()
    shutil.rmtree(out_dir, ignore_errors=True)
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=output.exe"},
    )

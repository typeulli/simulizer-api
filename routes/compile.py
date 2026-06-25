import base64
import io
import json
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from block2cpp import cppize
from debug.instrument import instrument_tu


router = APIRouter(prefix="/compile")

path_here = Path(__file__).parent.parent
path_temp = path_here / "temp"
path_bin = path_here / "bin"
path_res = path_here / "res"


ALLOWED_BUNDLE_EXTENSIONS = (".cpp", ".hpp")
# Image files usable as the exe icon. .ico is used as-is; the rest are converted
# to .ico server-side (Pillow). Kept in sync with the frontend BINARY_EXTENSIONS.
_IMAGE_EXTENSIONS = (".ico", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
# Files that ride along in the bundle tree (so the server can read the project
# config and assets) but are NOT translation units — skipped when materializing
# the compile sandbox. config.json is parsed for compile options; image files
# are resolved as the exe icon.
_NONSOURCE_SKIP_EXTENSIONS = (".json",) + _IMAGE_EXTENSIONS


# ── Compile options (from the project's compile.json, sent by the frontend) ──
#
# The frontend parses compile.json and forwards the resolved options here. We
# re-validate everything server-side (defense in depth) before it reaches a
# compiler command line: optimization/std are closed enums, and each define
# must be a bare identifier (optionally `=value`) — no spaces or shell-special
# characters — so nothing user-controlled can inject extra arguments.
_OPT_LEVELS = {"O0", "O1", "O2", "O3", "Os"}
_STDS = {"c++17", "c++20", "c++23"}
_DEFINE_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(=[A-Za-z0-9_.]+)?$')


# Internal holder for the resolved compile options. No longer a request field:
# the server parses these from the bundle's config.json (single source of truth)
# rather than trusting client-sent values.
class CompileOptions(BaseModel):
    system: Optional[str] = None       # auto/None → sniff User-Agent (Build only)
    optimization: str = "O3"
    std: str = "c++17"
    defines: list[str] = []
    icon: str = ""                     # relative .ico path (compile.icon); Build+Windows only
    alone: bool = False                # build.alone: standalone SSE+browser .exe vs
                                       # client-driven shared-memory .sim (default)


@dataclass(frozen=True)
class ResolvedFlags:
    std_flag: str          # e.g. "-std=c++17"
    opt_flag: str          # e.g. "-O3"
    define_flags: list[str]  # e.g. ["-DDEBUG", "-DVERSION=2"]


def _resolve_flags(opts: CompileOptions) -> ResolvedFlags:
    if opts.optimization not in _OPT_LEVELS:
        raise HTTPException(status_code=400, detail=f"Invalid optimization: {opts.optimization!r}")
    if opts.std not in _STDS:
        raise HTTPException(status_code=400, detail=f"Invalid std: {opts.std!r}")
    define_flags: list[str] = []
    for d in opts.defines:
        if not isinstance(d, str) or not _DEFINE_RE.match(d):
            raise HTTPException(status_code=400, detail=f"Invalid define: {d!r}")
        define_flags.append(f"-D{d}")
    if opts.alone:
        # Standalone build: simstd.hpp serves the SSE console + opens a browser
        # instead of writing to shared memory.
        define_flags.append("-DSIMULIZER_ALONE")
    return ResolvedFlags(std_flag=f"-std={opts.std}", opt_flag=f"-{opts.optimization}", define_flags=define_flags)


# ── config.json (server-side parse) ────────────────────────────────────────
#
# The frontend ships config.json inside the bundle tree as a normal file; the
# server is the source of truth for what reaches the compiler. Settings are
# namespaced into sections: `build` (system + icon, Build-only) and `compile`
# (optimization/std/defines). Missing/invalid values fall back to defaults (the
# editor surfaces JSON errors to the user separately). _resolve_flags then
# re-validates as defense in depth before anything hits a command line.
_SYSTEMS = {"auto", "windows", "linux", "macos"}
CONFIG_FILENAME = "config.json"


def _read_config(tree: Optional[list]) -> dict:
    """Parse the bundle's root config.json and return the whole object, or {}
    when it's absent / unparseable / wrong-shaped (lenient — a broken config
    falls back to defaults rather than failing the build)."""
    if not isinstance(tree, list):
        return {}
    node = next(
        (n for n in tree
         if isinstance(n, dict) and n.get("type") == "file" and n.get("name") == CONFIG_FILENAME),
        None,
    )
    if node is None:
        return {}
    try:
        data = json.loads(node.get("content") or "")
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _section(cfg: dict, name: str) -> dict:
    sec = cfg.get(name)
    return sec if isinstance(sec, dict) else {}


def _compile_options_from_tree(tree: Optional[list]) -> CompileOptions:
    """Resolve compile options from the bundle's config.json — `system`/`icon`
    from the `build` section, optimization/std/defines from `compile`. Unknown/
    out-of-range values silently fall back to their defaults."""
    cfg = _read_config(tree)
    build = _section(cfg, "build")
    comp = _section(cfg, "compile")
    opts = CompileOptions()
    if isinstance(build.get("system"), str) and build["system"] in _SYSTEMS:
        opts.system = build["system"]
    if isinstance(build.get("icon"), str):
        opts.icon = build["icon"].strip()
    if isinstance(build.get("alone"), bool):
        opts.alone = build["alone"]
    if isinstance(comp.get("optimization"), str) and comp["optimization"] in _OPT_LEVELS:
        opts.optimization = comp["optimization"]
    if isinstance(comp.get("std"), str) and comp["std"] in _STDS:
        opts.std = comp["std"]
    if isinstance(comp.get("defines"), list):
        opts.defines = [d for d in comp["defines"] if isinstance(d, str) and _DEFINE_RE.match(d)]
    return opts


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
    (path_here / "lib" / "sim_shm.hpp",   "lib/sim_shm.hpp"),
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


# ── Per-OS build targets ──────────────────────────────────────────────────
#
# We build the user's program for the *requesting* OS, not the server's. The
# Windows host has no native Linux/Darwin compiler, so:
#   - windows: native MinGW g++ (links the prebuilt resource.o icon/manifest
#              and binary_data.o, plus the Win32 PDH/Winsock/GDI libs).
#   - linux  : g++ inside a Docker container.
#   - macos  : osxcross' clang inside a Docker container.
# For non-Windows targets the Win32 resource object and PDH/Winsock/GDI libs
# are dropped (simstd.hpp gates those behind `_WIN32`) and the portable
# embedded-asset data is recompiled from source inside the container.
WINDOWS_GXX = "C:/mingw64/bin/g++.exe"
WINDOWS_WINDRES = "C:/mingw64/bin/windres.exe"
LINUX_BUILD_IMAGE = "gcc:13"
# Apple's SDK can't ship in a public image, so this points at a locally-built
# osxcross image; `o64-clang++` is its C++ driver. Build the image separately
# before serving macOS targets.
MACOS_BUILD_IMAGE = "simulizer/osxcross:latest"
MACOS_GXX = "o64-clang++"


@dataclass(frozen=True)
class BuildTarget:
    key: str            # canonical OS key
    out_name: str       # filename of the produced binary inside out_dir
    download_name: str  # filename presented to the client


TARGETS = {
    "windows": BuildTarget("windows", "output.exe", "output.exe"),
    "linux":   BuildTarget("linux",   "output",     "output"),
    "macos":   BuildTarget("macos",   "output",     "output"),
}

_SYSTEM_ALIASES = {
    "windows": "windows", "win": "windows", "win32": "windows", "win64": "windows",
    "linux": "linux", "ubuntu": "linux", "debian": "linux",
    "macos": "macos", "mac": "macos", "osx": "macos", "darwin": "macos",
}


def _resolve_target_os(system: Optional[str], user_agent: str) -> str:
    """Pick the target OS. An explicit `system=` query param wins; otherwise we
    sniff the User-Agent. Defaults to windows when nothing matches."""
    if system:
        key = _SYSTEM_ALIASES.get(system.strip().lower())
        if key is None:
            raise HTTPException(status_code=400, detail=f"Unsupported system: {system!r}")
        return key
    ua = (user_agent or "").lower()
    if "windows" in ua:
        return "windows"
    if "mac" in ua or "darwin" in ua or "iphone" in ua or "ipad" in ua:
        return "macos"
    if "linux" in ua or "android" in ua or "x11" in ua:
        return "linux"
    return "windows"


# Portable embedded-asset sources (pure byte arrays — no Win32 API). Windows
# links the prebuilt bin/binary_data.o; other targets recompile these inside
# the build container.
_BUILD_DATA_FILES = (
    "binary_data.cpp", "console.hpp", "console.html.hpp", "index.html.hpp",
)


def _stage_build_data(data_dir: Path) -> Path:
    """Copy the embedded-asset sources into `data_dir` (all relative includes
    resolve from there since the headers sit beside binary_data.cpp) and return
    the path to binary_data.cpp."""
    src_dir = path_here / "lib" / "bin"
    for name in _BUILD_DATA_FILES:
        shutil.copy2(src_dir / name, data_dir / name)
    return data_dir / "binary_data.cpp"


def _build_argv(os_key: str, *, cpp_file: Path, exe_file: Path, system_dir: Path,
                project_root: Path, out_dir: Path, data_cpp: Optional[Path],
                flags: ResolvedFlags, res_override: Optional[Path] = None) -> list[str]:
    """Assemble the compiler command line for the given target OS. Windows runs
    natively; Linux/macOS wrap g++/clang in `docker run`, mounting out_dir at
    /work so every path the compiler touches lives under the bind mount.

    `res_override` is a per-build resource object (font + custom icon) used in
    place of the prebuilt bin/resource.o when the user set an exe icon; it only
    applies to Windows (other targets don't embed a Win32 resource)."""
    if os_key == "windows":
        res_file = res_override if res_override is not None else (path_bin / "resource.o")
        bin_file = path_bin / "binary_data.o"
        return [
            WINDOWS_GXX,
            str(cpp_file),
            str(res_file), str(bin_file),
            flags.std_flag,
            "-o", str(exe_file),
            f"-I{system_dir}",
            f"-I{project_root}",
            flags.opt_flag,
            *flags.define_flags,
            "-pthread", "-lpdh", "-lws2_32", "-lgdi32",
            "-static-libgcc", "-static-libstdc++",
            "-Wl,-Bstatic", "-lstdc++", "-lpthread", "-Wl,-Bdynamic",
            "-v",
        ]

    assert data_cpp is not None, "non-Windows builds need the staged data source"

    def c(p: Path) -> str:
        # Host path under out_dir → container path under /work.
        return "/work/" + Path(p).relative_to(out_dir).as_posix()

    common = [
        c(cpp_file), c(data_cpp),
        flags.std_flag,
        "-o", c(exe_file),
        f"-I{c(system_dir)}",
        f"-I{c(project_root)}",
        flags.opt_flag,
        *flags.define_flags,
        "-pthread",
        "-v",
    ]
    if os_key == "linux":
        # -lrt: POSIX shm_open/mmap for the shared-memory output mirror
        # (lib/sim_shm.hpp); harmless on glibc >=2.34 where it folded into libc.
        inner = ["g++", *common, "-lrt", "-static-libgcc", "-static-libstdc++"]
        image = LINUX_BUILD_IMAGE
    else:  # macos
        inner = [MACOS_GXX, *common]
        image = MACOS_BUILD_IMAGE
    return [
        "docker", "run", "--rm",
        "-v", f"{out_dir}:/work",
        "-w", "/work",
        image,
        *inner,
    ]


# ── Windows exe icon ───────────────────────────────────────────────────────
#
# The bundle ships the chosen icon image (base64) as a normal tree file. The
# server resolves the relative compile.icon path, converts the image to a
# Windows .ico when it isn't one already (Pillow), and embeds it into a
# per-build resource object alongside the runtime font — then links that instead
# of the prebuilt bin/resource.o. Windows-only (other targets carry no resource).
_MAX_ICON_BYTES = 1 * 1024 * 1024          # .ico passthrough cap (1 MB)
_MAX_ICON_SOURCE_BYTES = 4 * 1024 * 1024   # source image cap, pre-conversion (4 MB)


def _decode_b64(data: str, cap: int) -> bytes:
    try:
        raw = base64.b64decode(data, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="아이콘 이미지를 디코딩할 수 없어요 (base64 형식 오류).")
    if len(raw) > cap:
        raise HTTPException(status_code=400, detail="아이콘 이미지가 너무 큽니다.")
    return raw


def _image_to_ico(raw: bytes) -> bytes:
    """Convert a raster image (PNG/JPG/GIF/BMP/WebP/…) to a multi-size Windows
    .ico via Pillow. Non-square inputs are padded to a transparent square so
    they aren't distorted. Pillow is imported lazily so a missing optional
    dependency only affects icon conversion, not regular builds."""
    try:
        from PIL import Image
    except ImportError:
        raise HTTPException(status_code=500, detail="이미지를 .ico 로 변환하려면 Pillow(PIL) 가 필요해요.")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        img = img.convert("RGBA")
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 파일을 읽을 수 없어요.")
    w, h = img.size
    side = max(w, h)
    if w != h:  # pad to a transparent square so non-square art isn't stretched
        square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        square.paste(img, ((side - w) // 2, (side - h) // 2))
        img = square
    if side != 256:
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    return buf.getvalue()


def _icon_to_ico_bytes(content_b64: str, ext: str) -> bytes:
    """Resolve a bundle image file (base64) to Windows .ico bytes. A .ico is
    validated and passed through; any other supported raster image is
    converted. Raises HTTPException(400/500) on bad/unreadable input."""
    if ext == ".ico":
        raw = _decode_b64(content_b64, _MAX_ICON_BYTES)
        # ICONDIR header: reserved(2)=0, type(2)=1 (icon), little-endian.
        if raw[:4] != b"\x00\x00\x01\x00":
            raise HTTPException(status_code=400, detail="유효한 .ico 파일이 아니에요.")
        return raw
    return _image_to_ico(_decode_b64(content_b64, _MAX_ICON_SOURCE_BYTES))


def _resolve_icon_from_tree(tree: Optional[list], icon_path: str) -> Optional[bytes]:
    """Resolve the relative `compile.icon` path within the bundle tree to
    Windows .ico bytes, server-side (converting non-.ico images on the fly).
    Returns None when unset / not a supported image / not found / not base64.

    Traversal-safe: the path is walked against the in-memory tree (it can never
    reach the filesystem), and absolute / drive-letter / '.' / '..' segments are
    rejected outright. A found-but-unreadable image raises."""
    if not icon_path or not isinstance(tree, list):
        return None
    p = icon_path.strip().replace("\\", "/")
    ext = Path(p).suffix.lower()
    if ext not in _IMAGE_EXTENSIONS:
        return None
    if p.startswith("/") or (len(p) >= 2 and p[1] == ":"):  # absolute / drive-letter
        return None
    parts = [seg for seg in p.split("/") if seg != ""]
    if not parts or any(seg in (".", "..") for seg in parts):
        return None
    nodes: object = tree
    for i, seg in enumerate(parts):
        if not isinstance(nodes, list):
            return None
        match = next((n for n in nodes if isinstance(n, dict) and n.get("name") == seg), None)
        if match is None:
            return None
        if i == len(parts) - 1:
            if match.get("type") != "file" or match.get("encoding") != "base64":
                return None
            return _icon_to_ico_bytes(match.get("content") or "", ext)
        if match.get("type") != "folder":
            return None
        nodes = match.get("contents", [])
    return None


def _build_windows_resource(out_dir: Path, icon_bytes: bytes) -> Path:
    """Compile a per-build resource object that embeds BOTH the runtime font
    (RCDATA 101 — simstd.hpp loads it via FindResource) and the user's icon
    (ICON 1 — picked up by Explorer for the .exe). Returns the .o path.

    windres runs with cwd=res_dir so the relative resource filenames resolve.
    """
    res_dir = out_dir / "_res"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "app.ico").write_bytes(icon_bytes)

    rc_lines: list[str] = []
    font_src = path_res / "JetBrainsMono-Medium.ttf"
    if font_src.is_file():
        shutil.copy2(font_src, res_dir / "JetBrainsMono-Medium.ttf")
        rc_lines.append('101 RCDATA "JetBrainsMono-Medium.ttf"')
    rc_lines.append('1 ICON "app.ico"')
    (res_dir / "simulizer.rc").write_text("\n".join(rc_lines) + "\n", encoding="utf-8")

    res_o = res_dir / "resource.o"
    proc = subprocess.run(
        [WINDOWS_WINDRES, "simulizer.rc", "-o", "resource.o", "--output-format=coff"],
        cwd=str(res_dir),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"아이콘 리소스 컴파일 실패 (windres):\n{(proc.stderr or proc.stdout or '').strip()}",
        )
    return res_o


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
            # config/data JSON and binary assets (icons) ride along in the tree
            # but aren't translation units — skip them so only .cpp/.hpp ever
            # land in the compile sandbox. They're read separately (config parse
            # / icon resolution) straight from the tree data.
            if ext in _NONSOURCE_SKIP_EXTENSIONS:
                continue
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
    # lang="cpp":    multi-file bundle in `tree` + `entry`. Compile options and
    #               the exe icon are read server-side from the tree's config.json
    #               and .ico files — not sent as separate fields.
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


# `.exe` is optional so the same patterns match both MinGW (cc1plus.exe) and
# the Linux/macOS toolchains (cc1plus); clang prints `cc1` rather than the
# collect2/as banners, so some stages simply won't fire there — progress is
# cosmetic and the build still completes.
BUILD_STAGES = [
    (re.compile(r'cc1(?:plus)?(?:\.exe)?'),  "컴파일 시작"),
    (re.compile(r'GNU C\+\+\d+.*version'),   "컴파일러 초기화"),
    (re.compile(r'End of search list\.'),    "헤더 탐색 완료"),
    (re.compile(r'[\\/]as(?:\.exe)?[ "\t]'), "어셈블 중"),
    (re.compile(r'GNU assembler version'),   "어셈블 시작"),
    (re.compile(r'collect2(?:\.exe)?'),      "링킹 중"),
]
BUILD_TOTAL = len(BUILD_STAGES) + 1


def _stream_build(file_uuid: str, out_dir: Path, cpp_file: Path, exe_file: Path,
                  project_root: Path, system_dir: Path, target: BuildTarget,
                  data_cpp: Optional[Path], flags: ResolvedFlags,
                  res_override: Optional[Path] = None):
    # The -I paths are restricted to per-build staging dirs so user code cannot
    # reach backend Python sources, the auth `key/` dir, prompts, etc. via
    # #include. Only the headers in _BUILD_SYSTEM_FILES land in system_dir.
    BUILD_COMMAND = _build_argv(
        target.key,
        cpp_file=cpp_file,
        exe_file=exe_file,
        system_dir=system_dir,
        project_root=project_root,
        out_dir=out_dir,
        data_cpp=data_cpp,
        flags=flags,
        res_override=res_override,
    )

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

    _write_status(out_dir, "ready", out_name=target.out_name, download_name=target.download_name)
    yield f"data: {json.dumps({'step': BUILD_TOTAL, 'total': BUILD_TOTAL, 'message': '완료'})}\n\n"
    yield f"event: done\ndata: {json.dumps({'uuid': file_uuid, 'name': target.download_name})}\n\n"


_MAIN_SHIM = """
int main() {
    simulizer::init();
    std::thread t([]() {
        worker();
#ifdef SIMULIZER_ALONE
        std::cout << "\\nPress enter to exit..." << std::flush;
        std::cin.get();
        simulizer::svr.stop();
#endif
    });
    simulizer::serve();   // alone: serve SSE (blocks until stop); else: returns now
    t.join();
    return 0;
}
"""


@router.post("/build")
def build(body: BuildRequest, request: Request, system: Optional[str] = None):
    # Resolve compile options from the bundle's config.json (server-side source
    # of truth), then validate them (optimization/std/defines) before they reach
    # the command line.
    opts = _compile_options_from_tree(body.tree)
    flags = _resolve_flags(opts)

    # Build for the requesting client's OS. Precedence: explicit `?system=`
    # query → config.json `system` → User-Agent sniff. "auto"/None means sniff.
    requested_system = system
    if not requested_system and opts.system and opts.system != "auto":
        requested_system = opts.system
    os_key = _resolve_target_os(requested_system, request.headers.get("user-agent", ""))
    target = TARGETS[os_key]
    if not opts.alone:
        # Client-driven build: ship as output.sim — a renamed native executable
        # the Simulizer desktop client launches as a subprocess and reads over
        # shared memory. The on-disk out_name (what `-o` produced) is unchanged.
        target = BuildTarget(target.key, target.out_name, "output.sim")

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

    # Windows links the prebuilt bin/binary_data.o; non-Windows targets compile
    # the portable asset data from source inside the build container.
    data_cpp = None
    if os_key != "windows":
        data_dir = out_dir / "_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_cpp = _stage_build_data(data_dir)

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
    exe_file = out_dir / target.out_name

    # Custom exe icon: resolve the relative compile.icon path within the tree to
    # .ico bytes (traversal-safe), then embed it. Only Windows links a Win32
    # resource, so the icon is applied there and ignored for Linux/macOS.
    res_override: Optional[Path] = None
    if os_key == "windows":
        icon_bytes = _resolve_icon_from_tree(body.tree, opts.icon)
        if icon_bytes:
            res_override = _build_windows_resource(out_dir, icon_bytes)

    return StreamingResponse(
        _stream_build(file_uuid, out_dir, cpp_file, exe_file, project_root, system_dir, target, data_cpp, flags, res_override),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class EmccRequest(BaseModel):
    # Compile options are read server-side from the tree's config.json.
    tree: list
    entry: str


_WORKER_ENTRY_SHIM = (
    '\n#include <type_traits>\n'
    '#include <cstdio>\n'
    'template<typename F> int __sim_invoke_or_zero(F f) {\n'
    '    if constexpr (std::is_void_v<decltype(f())>) { f(); return 0; }\n'
    '    else { return f(); }\n'
    '}\n'
    # worker() returns without calling exit(), so the WASI libc never flushes
    # its (fully-buffered, non-tty) stdout — output would be lost. Make the
    # streams unbuffered so printf/cout appear immediately (incremental output
    # is also what a debugger wants), and flush once more on the way out.
    'extern "C" __attribute__((used)) int __sim_worker_entry() {\n'
    '    static bool __sim_io_init = []{ setvbuf(stdout, nullptr, _IONBF, 0); setvbuf(stderr, nullptr, _IONBF, 0); return true; }();\n'
    '    (void)__sim_io_init;\n'
    '    int __sim_r = __sim_invoke_or_zero(worker);\n'
    '    std::fflush(nullptr);\n'
    '    return __sim_r;\n'
    '}\n'
)


@router.post("/emcc")
def emcc(body: EmccRequest):
    # Compile options come from the bundle's config.json (server-side). `system`
    # is a Build-only concept; Run honors optimization/std/defines.
    flags = _resolve_flags(_compile_options_from_tree(body.tree))
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

        # If the program reads interactive input (sim_input_*), build with
        # Asyncify so those host imports can suspend the run until the UI
        # supplies a value (same machinery as the debugger); malloc/free are
        # exported so the worker can allocate the Asyncify stack. Only the user's
        # own sources are scanned — simstd.hpp lives in system_dir, not here.
        uses_input = any(
            "sim_input" in p.read_text(encoding="utf-8", errors="ignore")
            for p in project_root.rglob("*")
            if p.suffix in (".cpp", ".hpp", ".h", ".cc", ".cxx")
        )
        exported = ("['___sim_worker_entry','_malloc','_free']"
                    if uses_input else "['___sim_worker_entry']")
        asyncify_flags = (
            ["-sASYNCIFY=1", "-sASYNCIFY_IMPORTS=['__sim_input_i32','__sim_input_f64']"]
            if uses_input else []
        )

        # Standalone WASM: libc/libc++ statically linked into the module,
        # only `env` (our __sim_* bridges) and `wasi_snapshot_preview1` imports
        # need to be supplied by the worker at instantiate time.
        EMCC_COMMAND = [
            "em++",
            str(entry_path),
            flags.std_flag,
            "-o", str(wasm_file),
            f"-I{system_dir}",
            f"-I{project_root}",
            flags.opt_flag,
            *flags.define_flags,
            "-sSTANDALONE_WASM=1",
            "-sERROR_ON_UNDEFINED_SYMBOLS=0",
            *asyncify_flags,
            f"-sEXPORTED_FUNCTIONS={exported}",
            "-fno-exceptions",
            # Build a WASI *reactor* (no main/_start): the worker calls
            # __sim_worker_entry directly, so we just need `_initialize` to run
            # the global constructors (e.g. std::ios_base::Init for std::cout).
            "--no-entry",
            "-Wl,--allow-undefined",
        ]

        # `shell=True` is retained because `em++` is a launcher script (.bat /
        # python shim) that CreateProcess can't exec directly on Windows. It is
        # safe here: every interpolated token is server-validated — std/opt are
        # closed enums, defines match a strict identifier regex (no spaces or
        # shell metacharacters), and all paths are server-generated uuid dirs —
        # so no user input can inject additional shell arguments.
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


@router.post("/emcc/debug")
def emcc_debug(body: EmccRequest):
    """Debug counterpart of /compile/emcc. Builds an *instrumented*, Asyncify-
    enabled standalone WASM at -O0 -g plus a "rich sidecar" of type/line/variable
    metadata, so the Web Worker can offer breakpoints, stepping, a call stack and
    variable inspection over the user's own C++. Returns a JSON envelope
    `{"wasm": <base64>, "sidecar": {...}}` (Run still returns raw wasm)."""
    opts = _compile_options_from_tree(body.tree)
    flags = _resolve_flags(opts)
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

        # Instrument every user file the entry TU touches (rewrites them in place)
        # and capture the sidecar BEFORE appending the worker-entry shim, so the
        # shim itself isn't instrumented and the parse sees the user's real code.
        try:
            sidecar = instrument_tu(
                entry_path, project_root, system_dir, flags.std_flag, flags.define_flags
            )
        except Exception as e:  # parse produced no usable AST
            raise HTTPException(status_code=422, detail=f"디버그 계측 실패: {e}")

        entry_path.write_text(
            entry_path.read_text(encoding="utf-8") + _WORKER_ENTRY_SHIM,
            encoding="utf-8",
        )

        wasm_file = out_dir / "user.wasm"

        # Same standalone-wasm shape as Run, but: -O0 -g for faithful stepping,
        # ASYNCIFY so __sim_dbg_line can unwind/rewind to pause, and malloc/free
        # exported so the worker can allocate the Asyncify stack region. Only
        # __sim_dbg_line is an async import (the other hooks never unwind).
        EMCC_COMMAND = [
            "em++",
            str(entry_path),
            flags.std_flag,
            "-o", str(wasm_file),
            f"-I{system_dir}",
            f"-I{project_root}",
            "-O0", "-g",
            *flags.define_flags,
            "-sSTANDALONE_WASM=1",
            "-sERROR_ON_UNDEFINED_SYMBOLS=0",
            "-sASYNCIFY=1",
            "-sASYNCIFY_IMPORTS=['__sim_dbg_line']",
            "-sEXPORTED_FUNCTIONS=['___sim_worker_entry','_malloc','_free']",
            "-fno-exceptions",
            # WASI reactor (see /compile/emcc): `_initialize` runs the global
            # constructors so std::cout & user global objects work.
            "--no-entry",
            "-Wl,--allow-undefined",
        ]

        # shell=True for the same reason as /compile/emcc (em++ is a launcher
        # script). Every interpolated token is server-generated or server-
        # validated (std/opt enums, identifier-only defines, uuid paths).
        proc = subprocess.run(
            EMCC_COMMAND,
            capture_output=True,
            text=True,
            shell=True,
        )
        if proc.returncode != 0:
            raise HTTPException(status_code=422, detail=proc.stderr)

        wasm_b64 = base64.b64encode(wasm_file.read_bytes()).decode("ascii")
        return {"wasm": wasm_b64, "sidecar": sidecar}
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


class EmccBlockRequest(BaseModel):
    # Legacy Blockly JSON (same shape the /build lang="blocks" path consumes).
    code: str


@router.post("/emcc/blocks")
def emcc_blocks(body: EmccBlockRequest):
    """Interactive run path for Block programs that contain input blocks.

    Transpiles the blocks to C++ (cppize → worker()) and builds an Asyncify-
    enabled standalone WASM whose sim_input_int()/sim_input_float() suspend via
    the __sim_input_* host imports, so the worker can pause for user input and
    resume. Returns raw wasm, like /compile/emcc."""
    flags = _resolve_flags(_compile_options_from_tree(None))
    file_uuid = str(uuid.uuid4())
    out_dir = path_temp / file_uuid
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        project_root = out_dir / "project"
        project_root.mkdir(parents=True, exist_ok=True)
        system_dir = out_dir / "_system"
        system_dir.mkdir(parents=True, exist_ok=True)
        _stage_build_system(system_dir)

        try:
            cpp_code = cppize(body.code, "worker")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Same worker-entry shim as /compile/emcc (provides __sim_worker_entry
        # calling worker()); the cppize output already #includes "simstd.hpp".
        cpp_file = project_root / "output.cpp"
        cpp_file.write_text(
            '#include <iostream>\n' + cpp_code + _WORKER_ENTRY_SHIM,
            encoding="utf-8",
        )

        wasm_file = out_dir / "user.wasm"

        # Standalone wasm (see /compile/emcc) + ASYNCIFY so the input host
        # imports can unwind/rewind, and malloc/free exported so the worker can
        # allocate the Asyncify stack region.
        EMCC_COMMAND = [
            "em++",
            str(cpp_file),
            flags.std_flag,
            "-o", str(wasm_file),
            f"-I{system_dir}",
            f"-I{project_root}",
            flags.opt_flag,
            *flags.define_flags,
            "-sSTANDALONE_WASM=1",
            "-sERROR_ON_UNDEFINED_SYMBOLS=0",
            "-sASYNCIFY=1",
            "-sASYNCIFY_IMPORTS=['__sim_input_i32','__sim_input_f64']",
            "-sEXPORTED_FUNCTIONS=['___sim_worker_entry','_malloc','_free']",
            "-fno-exceptions",
            "--no-entry",
            "-Wl,--allow-undefined",
        ]

        # shell=True for the same reason as /compile/emcc (em++ is a launcher
        # script); every interpolated token is server-generated or validated.
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

    out_name = status.get("out_name", "output.exe")
    download_name = status.get("download_name", out_name)
    exe_file = out_dir / out_name
    if not exe_file.exists():
        raise HTTPException(status_code=404, detail="File not found")

    content = exe_file.read_bytes()
    shutil.rmtree(out_dir, ignore_errors=True)
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={download_name}"},
    )

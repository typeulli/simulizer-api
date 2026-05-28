"""Generate ``bin/include/simstd.hpp`` for clangd/LSP consumption.

Three things happen relative to the source ``simstd.hpp``:

* Only the ``__EMSCRIPTEN__`` branch is kept — the native (``#else``)
  branch and the surrounding ``#ifdef`` / ``#endif`` directives are
  stripped from the output entirely.
* The ``namespace simulizer { ... }`` block is removed entirely.
* Every remaining function body is replaced with ``;`` so the file is
  cheap for clangd to parse for completion / signature help.

The output is intentionally *not* compilable.
"""
from __future__ import annotations

import os
import pathlib

import clang.cindex
from clang.cindex import Config, CursorKind, TokenKind, TranslationUnit


LIBCLANG_DLL = r"C:/Program Files/LLVM/bin/libclang.dll"

FUNCTION_KINDS = {
    CursorKind.FUNCTION_DECL,
    CursorKind.CXX_METHOD,
    CursorKind.CONSTRUCTOR,
    CursorKind.DESTRUCTOR,
    CursorKind.CONVERSION_FUNCTION,
    CursorKind.FUNCTION_TEMPLATE,
}


def _ensure_libclang() -> None:
    if not Config.loaded:
        Config.set_library_file(LIBCLANG_DLL)


# ─────────────────────────────────────────────────────────────────────────────
# Source-byte scanning helpers.
#
# We can't rely on ``cursor.is_definition()`` or ``CompoundStmt`` children
# because the Python libclang bindings do not expose function bodies for
# ``FUNCTION_TEMPLATE`` cursors or for member functions defined inline
# inside a class template — both ubiquitous in simstd.hpp. Instead we
# locate bodies directly from the source bytes around each cursor.
# ─────────────────────────────────────────────────────────────────────────────


def _skip_string(src: bytes, i: int) -> int:
    n = len(src)
    i += 1
    while i < n:
        c = src[i:i+1]
        if c == b"\\":
            i += 2
        elif c == b'"':
            return i + 1
        else:
            i += 1
    return n


def _skip_char(src: bytes, i: int) -> int:
    n = len(src)
    i += 1
    while i < n:
        c = src[i:i+1]
        if c == b"\\":
            i += 2
        elif c == b"'":
            return i + 1
        else:
            i += 1
    return n


def _skip_line_comment(src: bytes, i: int) -> int:
    n = len(src)
    while i < n and src[i:i+1] != b"\n":
        i += 1
    return i


def _skip_block_comment(src: bytes, i: int) -> int:
    n = len(src)
    i += 2
    while i < n - 1 and src[i:i+2] != b"*/":
        i += 1
    return i + 2


def _find_matching_close_brace(src: bytes, open_pos: int) -> int | None:
    depth = 0
    i = open_pos
    n = len(src)
    while i < n:
        c = src[i:i+1]
        if c == b'"':
            i = _skip_string(src, i)
        elif c == b"'":
            i = _skip_char(src, i)
        elif src[i:i+2] == b"//":
            i = _skip_line_comment(src, i)
        elif src[i:i+2] == b"/*":
            i = _skip_block_comment(src, i)
        elif c == b"{":
            depth += 1
            i += 1
        elif c == b"}":
            depth -= 1
            i += 1
            if depth == 0:
                return i
        else:
            i += 1
    return None


def _skip_parens(src: bytes, open_pos: int) -> int:
    depth = 0
    i = open_pos
    n = len(src)
    while i < n:
        c = src[i:i+1]
        if c == b'"':
            i = _skip_string(src, i)
        elif c == b"'":
            i = _skip_char(src, i)
        elif src[i:i+2] == b"//":
            i = _skip_line_comment(src, i)
        elif src[i:i+2] == b"/*":
            i = _skip_block_comment(src, i)
        elif c == b"(":
            depth += 1
            i += 1
        elif c == b")":
            depth -= 1
            i += 1
            if depth == 0:
                return i
        else:
            i += 1
    return n


def _find_first_top_brace(src: bytes, start: int, end: int) -> int | None:
    i = start
    paren = 0
    bracket = 0
    while i < end:
        c = src[i:i+1]
        if c == b'"':
            i = _skip_string(src, i)
        elif c == b"'":
            i = _skip_char(src, i)
        elif src[i:i+2] == b"//":
            i = _skip_line_comment(src, i)
        elif src[i:i+2] == b"/*":
            i = _skip_block_comment(src, i)
        elif c == b"(":
            paren += 1; i += 1
        elif c == b")":
            paren -= 1; i += 1
        elif c == b"[":
            bracket += 1; i += 1
        elif c == b"]":
            bracket -= 1; i += 1
        elif c == b"{" and paren == 0 and bracket == 0:
            return i
        else:
            i += 1
    return None


def _find_top_colon(src: bytes, start: int, end: int) -> int | None:
    i = start
    paren = 0
    bracket = 0
    while i < end:
        c = src[i:i+1]
        if c == b'"':
            i = _skip_string(src, i)
        elif c == b"'":
            i = _skip_char(src, i)
        elif src[i:i+2] == b"//":
            i = _skip_line_comment(src, i)
        elif src[i:i+2] == b"/*":
            i = _skip_block_comment(src, i)
        elif c == b"(":
            paren += 1; i += 1
        elif c == b")":
            paren -= 1; i += 1
        elif c == b"[":
            bracket += 1; i += 1
        elif c == b"]":
            bracket -= 1; i += 1
        elif c == b":" and paren == 0 and bracket == 0:
            if src[i:i+2] == b"::":
                i += 2
            elif src[i-1:i] == b":":
                i += 1
            else:
                return i
        else:
            i += 1
    return None


def _scan_after_signature(
    src: bytes, sig_end: int, is_ctor: bool
) -> tuple[int, int] | None:
    n = len(src)
    i = sig_end
    init_colon: int | None = None

    while i < n:
        c = src[i:i+1]
        if c in (b" ", b"\t", b"\n", b"\r"):
            i += 1
            continue
        if src[i:i+2] == b"//":
            i = _skip_line_comment(src, i)
            continue
        if src[i:i+2] == b"/*":
            i = _skip_block_comment(src, i)
            continue
        if c == b";" or c == b"=":
            return None
        if c == b"{":
            close = _find_matching_close_brace(src, i)
            if close is None:
                return None
            return (init_colon if init_colon is not None else i), close
        if c == b":" and src[i:i+2] != b"::":
            if is_ctor and init_colon is None:
                init_colon = i
                i += 1
                continue
            return None
        if c == b"(":
            i = _skip_parens(src, i)
            continue
        i += 1
    return None


def _locate_body(cursor, src: bytes) -> tuple[int, int] | None:
    ext_start = cursor.extent.start.offset
    ext_end = cursor.extent.end.offset
    is_ctor = cursor.kind == CursorKind.CONSTRUCTOR

    if ext_end > ext_start and src[ext_end - 1:ext_end] == b"}":
        body_open = _find_first_top_brace(src, ext_start, ext_end)
        if body_open is not None:
            strip_start = body_open
            if is_ctor:
                colon = _find_top_colon(src, ext_start, body_open)
                if colon is not None:
                    strip_start = colon
            return strip_start, ext_end

    return _scan_after_signature(src, ext_end, is_ctor)


def _find_inline_keyword(cursor, src: bytes) -> tuple[int, int] | None:
    """If the cursor has an ``inline`` specifier on its declaration, return
    the byte range covering ``inline`` plus the single trailing whitespace
    so that removal leaves no awkward gap. Returns ``None`` otherwise."""
    for tok in cursor.get_tokens():
        if tok.kind == TokenKind.KEYWORD and tok.spelling == "inline":
            start = tok.extent.start.offset
            end = tok.extent.end.offset
            while end < len(src) and src[end:end+1] in (b" ", b"\t"):
                end += 1
            return start, end
    return None


def _collect_body_edits(
    tu: TranslationUnit, src_path: str, src: bytes
) -> list[tuple[int, int, bytes]]:
    """Walk ``tu`` and collect the edits needed for body-stripped
    functions:

    * the body itself (and ctor init list, if any) → ``;``
    * the ``inline`` specifier on the same declaration → removed entirely
    """
    src_norm = os.path.normcase(os.path.abspath(src_path))
    edits: list[tuple[int, int, bytes]] = []

    def visit(cursor) -> None:
        loc = cursor.location
        if loc.file is not None:
            if os.path.normcase(os.path.abspath(loc.file.name)) != src_norm:
                return

        if cursor.kind in FUNCTION_KINDS:
            body = _locate_body(cursor, src)
            if body is not None:
                bs, be = body
                # Pull the strip start backward across any whitespace so the
                # replacement `;` butts directly up against the signature.
                while bs > 0 and src[bs-1:bs] in (b" ", b"\t", b"\n", b"\r"):
                    bs -= 1
                edits.append((bs, be, b";"))
                inline_r = _find_inline_keyword(cursor, src)
                if inline_r is not None:
                    edits.append((inline_r[0], inline_r[1], b""))
                return

        for child in cursor.get_children():
            visit(child)

    for child in tu.cursor.get_children():
        visit(child)
    return edits


def _collect_namespace_ranges(
    tu: TranslationUnit, src_path: str, ns_name: str
) -> list[tuple[int, int]]:
    """Find every ``namespace <ns_name> { ... }`` block originating from
    ``src_path`` and return its byte range (including the closing ``}``).
    """
    src_norm = os.path.normcase(os.path.abspath(src_path))
    ranges: list[tuple[int, int]] = []

    def visit(cursor) -> None:
        loc = cursor.location
        if loc.file is not None:
            if os.path.normcase(os.path.abspath(loc.file.name)) != src_norm:
                return
        if cursor.kind == CursorKind.NAMESPACE and cursor.spelling == ns_name:
            ranges.append((cursor.extent.start.offset, cursor.extent.end.offset))
            return
        for child in cursor.get_children():
            visit(child)

    for child in tu.cursor.get_children():
        visit(child)
    return ranges


def _find_emscripten_directive_ranges(src: bytes) -> list[tuple[int, int]]:
    """Return ranges covering the ``#ifdef __EMSCRIPTEN__`` line and the
    matching ``#else  // !__EMSCRIPTEN__`` … ``#endif  // __EMSCRIPTEN__``
    block so they can be elided from the output."""
    ranges: list[tuple[int, int]] = []

    def _line_end(off: int) -> int:
        eol = src.find(b"\n", off)
        return len(src) if eol < 0 else eol + 1

    idx = src.find(b"#ifdef __EMSCRIPTEN__")
    if idx < 0:
        raise RuntimeError("'#ifdef __EMSCRIPTEN__' 토큰을 찾지 못했습니다.")
    ranges.append((idx, _line_end(idx)))

    else_idx = src.find(b"#else  // !__EMSCRIPTEN__")
    if else_idx < 0:
        raise RuntimeError("'#else  // !__EMSCRIPTEN__' 토큰을 찾지 못했습니다.")
    endif_idx = src.find(b"#endif  // __EMSCRIPTEN__", else_idx)
    if endif_idx < 0:
        raise RuntimeError("'#endif  // __EMSCRIPTEN__' 토큰을 찾지 못했습니다.")
    ranges.append((else_idx, _line_end(endif_idx)))

    return ranges


def _parse(src_path: str, extra_args: list[str]) -> TranslationUnit:
    index = clang.cindex.Index.create()
    project_root = str(pathlib.Path(src_path).resolve().parent)
    args = [
        "-x", "c++",
        "-std=c++17",
        f"-I{project_root}",
    ] + extra_args
    return index.parse(
        src_path,
        args=args,
        options=TranslationUnit.PARSE_INCOMPLETE,
    )


def _merge_edits(
    edits: list[tuple[int, int, bytes]]
) -> list[tuple[int, int, bytes]]:
    """Sort edits by (start, -end). An edit fully contained in the most
    recently kept one is dropped (its replacement is subsumed).
    """
    sorted_edits = sorted(set(edits), key=lambda e: (e[0], -e[1]))
    kept: list[tuple[int, int, bytes]] = []
    for s, e, rep in sorted_edits:
        if kept and s >= kept[-1][0] and e <= kept[-1][1]:
            continue
        kept.append((s, e, rep))
    return kept


def strip_simstd(src_path: pathlib.Path, out_path: pathlib.Path) -> None:
    _ensure_libclang()
    src_path = pathlib.Path(src_path)
    out_path = pathlib.Path(out_path)

    src_bytes = src_path.read_bytes()

    tu = _parse(str(src_path), ["-D__EMSCRIPTEN__"])

    edits: list[tuple[int, int, bytes]] = []
    edits.extend(_collect_body_edits(tu, str(src_path), src_bytes))
    for s, e in _collect_namespace_ranges(tu, str(src_path), "simulizer"):
        edits.append((s, e, b""))
    for s, e in _find_emscripten_directive_ranges(src_bytes):
        edits.append((s, e, b""))

    if not edits:
        raise RuntimeError(
            "편집할 범위를 한 건도 찾지 못했습니다 — libclang 파싱이 실패했을 수 있습니다."
        )

    merged = _merge_edits(edits)

    out = bytearray(src_bytes)
    for start, end, rep in reversed(merged):
        out[start:end] = rep

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(out))


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent
    strip_simstd(here / "simstd.hpp", here / "bin/include/simstd.hpp")

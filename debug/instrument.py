"""libclang-based debug instrumentation + "rich sidecar" emitter.

For a Debug build we parse the user's entry translation unit with libclang
(targeting the wasm32 emscripten ABI so type sizes/field offsets match what
em++ produces), then source-to-source rewrite every *user* file it touches to
inject four hooks the Web Worker provides as `env` imports:

  __sim_dbg_enter(funcId)        - function entry  (push a shadow frame)
  __sim_dbg_exit(funcId)         - function exit   (pop a shadow frame)
  __sim_dbg_line(locId)          - statement boundary (possible pause point)
  __sim_dbg_local(varId, &var)   - register a local's runtime address

Taking the address of every local forces it to be memory-resident at -O0, so
the worker can read its value straight from linear memory using the layout in
the emitted sidecar (files / functions / locations / variables / types). The
sidecar is shipped to the frontend alongside the wasm; no DWARF parser needed.

Instrumentation is best-effort: non-fatal parse diagnostics are tolerated (the
user code is compiled for real by em++ afterwards, which is the source of truth
for build errors). A statement is only hooked when it sits directly inside a
CompoundStmt, so brace-less `if`/`for` bodies are left intact (a v1 limitation).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import clang.cindex as CI


# ── Toolchain paths ─────────────────────────────────────────────────────────
# The pip `clang.cindex` ships a libclang 16 DLL, which is too old to parse
# emscripten's (clang-23) libc++ — it lacks builtins like __builtin_ctzg, so
# std::vector & friends fail to instantiate and degrade to error types. We
# instead load the locally-installed LLVM libclang (v20), which understands
# those builtins, and pair it with LLVM's matching clang builtin-header dir.
# Override the LLVM root via SIM_LLVM_DIR.
_LLVM_DIR = Path(os.environ.get("SIM_LLVM_DIR", r"C:/Program Files/LLVM"))
_LLVM_LIBCLANG = _LLVM_DIR / "bin" / "libclang.dll"
_LLVM_VER = os.environ.get("SIM_LLVM_VER", "20")
_RESOURCE_INC = _LLVM_DIR / "lib" / "clang" / _LLVM_VER / "include"

# emscripten sysroot supplies libc / libc++ headers (the ABI we target).
_EMSDK_UPSTREAM = Path(os.environ.get("SIM_EMSDK_UPSTREAM", r"C:/dev/emsdk/upstream"))
_SYSROOT_INC = _EMSDK_UPSTREAM / "emscripten" / "cache" / "sysroot" / "include"

# Point the cindex bindings at LLVM's libclang before any Index is created.
if _LLVM_LIBCLANG.is_file():
    try:
        CI.Config.set_library_file(str(_LLVM_LIBCLANG))
    except Exception:
        pass  # already initialised elsewhere; tolerate


def build_parse_args(system_dir: Path, project_root: Path,
                     std_flag: str, define_flags: list[str]) -> list[str]:
    """clang args mirroring the em++ Debug compile so the AST resolves the same
    headers and ABI. `system_dir`/`project_root` match the -I dirs the compiler
    uses; std/defines match the resolved compile options."""
    return [
        std_flag,
        "-target", "wasm32-unknown-emscripten",
        "-fno-exceptions",
        "-D__EMSCRIPTEN__",
        "-nostdinc",
        "-nostdinc++",
        "-isystem", str(_SYSROOT_INC / "c++" / "v1"),
        "-isystem", str(_RESOURCE_INC),
        "-isystem", str(_SYSROOT_INC),
        "-isystem", str(_SYSROOT_INC / "compat"),  # emscripten xlocale.h etc.
        f"-I{system_dir}",
        f"-I{project_root}",
        *define_flags,
    ]


# extern "C" declarations of the four hooks, prepended to each rewritten file.
_PROLOGUE = (
    'extern "C" {\n'
    'void __sim_dbg_enter(int);\n'
    'void __sim_dbg_exit(int);\n'
    'void __sim_dbg_line(int);\n'
    'void __sim_dbg_local(int, void*);\n'
    '}\n'
)


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


_FUNC_DEF_KINDS = {
    CI.CursorKind.FUNCTION_DECL,
    CI.CursorKind.CXX_METHOD,
    CI.CursorKind.CONSTRUCTOR,
    CI.CursorKind.DESTRUCTOR,
    CI.CursorKind.CONVERSION_FUNCTION,
}

# Containers we descend into looking for function definitions. We deliberately
# do NOT recurse into a function body here (that's done during instrumentation),
# nor into lambdas.
_CONTAINER_KINDS = {
    CI.CursorKind.NAMESPACE,
    CI.CursorKind.STRUCT_DECL,
    CI.CursorKind.CLASS_DECL,
    CI.CursorKind.CLASS_TEMPLATE,
    CI.CursorKind.UNEXPOSED_DECL,        # extern "C" {...}, linkage specs
}


class Instrumenter:
    def __init__(self, user_files: set[str]):
        self.user_files = {_norm(f) for f in user_files}
        self.files: list[str] = []
        self._file_ix: dict[str, int] = {}
        self.functions: list[dict] = []
        self.locations: list[dict] = []
        self.variables: list[dict] = []
        self.types: list[dict] = []
        self._type_ix: dict[str, int] = {}
        self._unknown_tid: Optional[int] = None
        # user CLASS_TEMPLATE name -> {params: [str], fields: [(name, Type)]},
        # used to reconstruct fields of template specializations (libclang
        # doesn't expose instantiated fields of a specialization).
        self._templates: dict[str, dict] = {}
        self.inserts: dict[str, list[tuple[int, int, str]]] = {}  # path -> (offset, order, text)
        self._order = 0
        self.srcs: dict[str, bytes] = {}

    # ── id tables ────────────────────────────────────────────────────────
    def file_id(self, path: str) -> int:
        n = _norm(path)
        if n not in self._file_ix:
            self._file_ix[n] = len(self.files)
            self.files.append(path.replace("\\", "/"))
        return self._file_ix[n]

    def loc_id(self, file_id: int, line: int) -> int:
        i = len(self.locations)
        self.locations.append({"id": i, "file": file_id, "line": line})
        return i

    def add_insert(self, path: str, offset: int, text: str):
        # `order` preserves insertion order when several inserts share an offset
        # (e.g. enter-hook then guard then param registrations at the '{').
        self.inserts.setdefault(_norm(path), []).append((offset, self._order, text))
        self._order += 1

    def _in_user_file(self, cursor) -> bool:
        f = cursor.location.file
        return f is not None and _norm(f.name) in self.user_files

    # ── type interning ───────────────────────────────────────────────────
    def _unknown_type_id(self) -> int:
        if self._unknown_tid is None:
            self._unknown_tid = len(self.types)
            self.types.append({"id": self._unknown_tid, "name": "<unknown>", "size": -1, "kind": "unknown"})
        return self._unknown_tid

    def intern_type(self, t) -> int:
        # libclang/cindex version skew can throw on some template-heavy types
        # (e.g. "Unknown template argument kind N"). Never let that abort the
        # build — degrade the offending type to <unknown> (renders as raw).
        try:
            return self._intern_type(t)
        except Exception:
            return self._unknown_type_id()

    def _intern_type(self, t) -> int:
        c = t.get_canonical()
        key = c.spelling
        if key in self._type_ix:
            return self._type_ix[key]
        tid = len(self.types)
        self._type_ix[key] = tid
        # Reserve a *valid* (unknown) slot before recursing, so recursive types
        # terminate and a mid-way failure still leaves a usable entry.
        entry: dict = {"id": tid, "name": key, "size": -1, "kind": "unknown"}
        self.types.append(entry)
        try:
            entry["size"] = c.get_size()
        except Exception:
            entry["size"] = -1
        try:
            self._fill_type(entry, c)
        except Exception:
            entry["kind"] = "unknown"
        return tid

    def _fill_type(self, entry: dict, c) -> None:
        K = CI.TypeKind
        scalar = {
            K.BOOL: "bool", K.CHAR_S: "char", K.SCHAR: "schar", K.UCHAR: "uchar",
            K.CHAR_U: "uchar", K.CHAR16: "ushort", K.CHAR32: "uint",
            K.SHORT: "short", K.USHORT: "ushort", K.INT: "int", K.UINT: "uint",
            K.LONG: "int", K.ULONG: "uint", K.LONGLONG: "longlong",
            K.ULONGLONG: "ulonglong", K.FLOAT: "float", K.DOUBLE: "double",
            K.LONGDOUBLE: "double", K.WCHAR: "int",
        }
        k = c.kind
        if k in scalar:
            entry["kind"] = "scalar"
            entry["scalar"] = scalar[k]
        elif k == K.POINTER:
            entry["kind"] = "pointer"
            entry["pointee"] = self.intern_type(c.get_pointee())
        elif k == K.CONSTANTARRAY:
            entry["kind"] = "array"
            entry["elem"] = self.intern_type(c.get_array_element_type())
            entry["count"] = c.get_array_size()
        elif k == K.RECORD:
            entry["kind"] = "record"
            decl = c.get_declaration()
            fields = []
            for f in decl.get_children():
                if f.kind == CI.CursorKind.FIELD_DECL:
                    try:
                        off = c.get_offset(f.spelling)  # bits
                    except Exception:
                        off = -1
                    fields.append({
                        "name": f.spelling,
                        "offset": off // 8 if off >= 0 else -1,
                        "type": self.intern_type(f.type),
                    })
            entry["fields"] = fields
            self._maybe_stl(entry, c)
            # libclang doesn't expose a template specialization's instantiated
            # fields — reconstruct user template structs (pair<int>, Box<T>, …).
            if not fields and "stl" not in entry:
                self._template_fields(c, entry)
        elif k == K.ENUM:
            entry["kind"] = "scalar"
            entry["scalar"] = "int"
        else:
            entry["kind"] = "unknown"

    def _maybe_stl(self, entry: dict, c) -> None:
        """Tag libc++ std::vector / std::string records with an `stl` descriptor
        so the frontend can pretty-print them (their private members aren't
        enumerable). The element type is interned recursively, so vector<Point>
        and vector<vector<int>> render through the normal machinery. The
        template-argument APIs are the cindex/DLL skew risk — guarded."""
        sp = entry.get("name", "")
        K = CI.TypeKind
        try:
            if "vector<" in sp:
                elem = c.get_template_argument_type(0)
                entry["stl"] = {"kind": "vector", "elem": self.intern_type(elem)}
            elif "basic_string<" in sp or sp == "std::string" or sp.endswith("::string"):
                arg = c.get_template_argument_type(0).get_canonical()
                if arg.kind in (K.CHAR_S, K.CHAR_U, K.SCHAR, K.UCHAR):  # char strings only (v1)
                    entry["stl"] = {"kind": "string"}
        except Exception:
            pass  # not enrichable — stays a raw record

    def _synth_pointer(self, pointee_id: int) -> int:
        key = f"__ptr_{pointee_id}"
        if key in self._type_ix:
            return self._type_ix[key]
        tid = len(self.types)
        self._type_ix[key] = tid
        self.types.append({
            "id": tid, "name": self.types[pointee_id]["name"] + " *",
            "size": 4, "kind": "pointer", "pointee": pointee_id,
        })
        return tid

    def _subst_field_type(self, ftype, param_map: dict) -> int:
        """Resolve a template field's written type to a concrete sidecar type:
        bare param `T` -> its argument; `T*` -> synthesized pointer; otherwise a
        concrete (non-dependent) type is interned directly."""
        written = ftype.spelling.strip()
        if written in param_map:
            return self.intern_type(param_map[written])
        if written.endswith("*"):
            base = written[:-1].strip()
            if base in param_map:
                return self._synth_pointer(self.intern_type(param_map[base]))
        if ftype.kind != CI.TypeKind.UNEXPOSED:  # concrete field (e.g. `int tag`)
            return self.intern_type(ftype)
        return self._unknown_type_id()

    def _template_fields(self, c, entry: dict) -> None:
        """Reconstruct fields of a user template specialization (e.g. pair<int>)
        from the primary template's field names + the specialization's offsets."""
        sp = c.spelling
        if "<" not in sp:
            return
        key = sp.split("<", 1)[0].rsplit("::", 1)[-1]  # unqualified template name
        tmpl = self._templates.get(key)
        if not tmpl:
            return
        try:
            nargs = c.get_num_template_arguments()
            args = [c.get_template_argument_type(i) for i in range(nargs)] if nargs and nargs > 0 else []
        except Exception:
            return
        param_map = {p: args[i] for i, p in enumerate(tmpl["params"]) if i < len(args)}
        fields = []
        for fname, ftype in tmpl["fields"]:
            try:
                off = c.get_offset(fname)
                if off < 0:
                    continue
                fields.append({"name": fname, "offset": off // 8,
                               "type": self._subst_field_type(ftype, param_map)})
            except Exception:
                continue
        if fields:
            entry["fields"] = fields

    # ── walking ──────────────────────────────────────────────────────────
    def run(self, tu, srcs: dict[str, bytes]):
        self.srcs = srcs
        try:
            self._collect_templates(tu.cursor)
        except Exception:
            pass
        self._walk_decls(tu.cursor)

    def _collect_templates(self, cursor):
        for c in cursor.walk_preorder():
            if c.kind == CI.CursorKind.CLASS_TEMPLATE and c.is_definition() and self._in_user_file(c):
                params = [ch.spelling for ch in c.get_children()
                          if ch.kind == CI.CursorKind.TEMPLATE_TYPE_PARAMETER]
                flds = [(ch.spelling, ch.type) for ch in c.get_children()
                        if ch.kind == CI.CursorKind.FIELD_DECL and ch.spelling]
                if params and flds:
                    self._templates[c.spelling] = {"params": params, "fields": flds}

    def _snapshot(self):
        """Capture table/insert sizes so a function whose instrumentation
        throws partway can be rolled back cleanly (atomic per function)."""
        return (
            len(self.functions), len(self.variables), len(self.locations),
            {p: len(v) for p, v in self.inserts.items()},
        )

    def _rollback(self, snap):
        nf, nv, nl, ins = snap
        del self.functions[nf:]
        del self.variables[nv:]
        del self.locations[nl:]
        for p in list(self.inserts.keys()):
            if p in ins:
                del self.inserts[p][ins[p]:]
            else:
                del self.inserts[p]

    def _walk_decls(self, cursor):
        try:
            children = list(cursor.get_children())
        except Exception:
            return
        for ch in children:
            try:
                if ch.kind in _FUNC_DEF_KINDS:
                    if ch.is_definition() and self._in_user_file(ch):
                        snap = self._snapshot()
                        try:
                            self._instrument_function(ch)
                        except Exception:
                            # cindex skew or an odd construct — skip just this
                            # function (it won't be debuggable) but keep the build.
                            self._rollback(snap)
                elif ch.kind in _CONTAINER_KINDS:
                    self._walk_decls(ch)
            except Exception:
                continue

    def _instrument_function(self, fn):
        path = fn.location.file.name
        fid = self.file_id(path)
        func_id = len(self.functions)
        self.functions.append({
            "id": func_id,
            "name": fn.spelling or fn.displayname,
            "file": fid,
            "line": fn.location.line,
        })
        body = next((c for c in fn.get_children()
                     if c.kind == CI.CursorKind.COMPOUND_STMT), None)
        if body is None:
            return
        prologue = (
            f"__sim_dbg_enter({func_id});"
            f"struct __SimG{func_id}{{~__SimG{func_id}(){{__sim_dbg_exit({func_id});}}}}"
            f"__sim_g{func_id};"
        )
        for p in fn.get_children():
            try:
                if p.kind == CI.CursorKind.PARM_DECL and p.spelling:
                    var_id = self._add_var(func_id, p.spelling, p.type, fn.location.line)
                    prologue += f"__sim_dbg_local({var_id},(void*)&{p.spelling});"
            except Exception:
                continue  # best-effort: skip a parameter we can't read
        self.add_insert(path, body.extent.start.offset + 1, prologue)  # just after '{'
        self._instrument_compound(body, func_id, path, fid)

    def _add_var(self, func_id, name, type_cursor, line) -> int:
        var_id = len(self.variables)
        self.variables.append({
            "id": var_id, "func": func_id, "name": name,
            "type": self.intern_type(type_cursor), "line": line,
        })
        return var_id

    def _instrument_compound(self, comp, func_id, path, fid):
        # Per-statement best-effort: a single statement whose cursor/type trips
        # the cindex version skew is skipped, the rest of the function still
        # gets its pause hooks.
        try:
            stmts = list(comp.get_children())
        except Exception:
            return
        for stmt in stmts:
            # 1) line hook (essential — breakpoints/stepping). Skip the stmt if
            #    we can't even place it.
            try:
                f = stmt.location.file
                if f is None or _norm(f.name) not in self.user_files:
                    continue
                lid = self.loc_id(fid, stmt.extent.start.line)
                self.add_insert(path, stmt.extent.start.offset, f"__sim_dbg_line({lid});")
                is_decl = stmt.kind == CI.CursorKind.DECL_STMT
                decl_end = stmt.extent.end.offset if is_decl else 0
            except Exception:
                continue
            # 2) local registrations (best-effort, independent of the line hook).
            if is_decl:
                try:
                    for d in stmt.get_children():
                        if d.kind == CI.CursorKind.VAR_DECL and d.spelling:
                            var_id = self._add_var(func_id, d.spelling, d.type, d.location.line)
                            self.add_insert(path, decl_end,
                                            f"__sim_dbg_local({var_id},(void*)&{d.spelling});")
                except Exception:
                    pass
            # 3) nested blocks (best-effort).
            try:
                self._recurse_nested(stmt, func_id, path, fid)
            except Exception:
                pass

    def _recurse_nested(self, cur, func_id, path, fid):
        try:
            children = list(cur.get_children())
        except Exception:
            return
        for ch in children:
            try:
                if ch.kind == CI.CursorKind.LAMBDA_EXPR:
                    continue  # don't instrument lambda bodies in v1
                if ch.kind == CI.CursorKind.COMPOUND_STMT:
                    self._instrument_compound(ch, func_id, path, fid)
                else:
                    self._recurse_nested(ch, func_id, path, fid)
            except Exception:
                continue

    # ── emit ─────────────────────────────────────────────────────────────
    def rewrite_file(self, path: str, src_bytes: bytes) -> bytes:
        ins = sorted(self.inserts.get(_norm(path), []), key=lambda x: (x[0], x[1]))
        out = bytearray(_PROLOGUE.encode())
        prev = 0
        for off, _order, text in ins:
            out += src_bytes[prev:off]
            out += text.encode()
            prev = off
        out += src_bytes[prev:]
        return bytes(out)

    def touched_files(self) -> set[str]:
        return set(self.inserts.keys())

    def sidecar(self) -> dict:
        return {
            "version": 1,
            "files": self.files,
            "functions": self.functions,
            "locations": self.locations,
            "variables": self.variables,
            "types": self.types,
        }


def _gather_user_files(project_root: Path) -> set[str]:
    """All .cpp/.hpp under the materialized project root are 'user files' the
    instrumenter may rewrite. Sysroot/system headers never appear here."""
    out: set[str] = set()
    for p in project_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".cpp", ".hpp", ".h", ".cc", ".cxx"):
            out.add(str(p))
    return out


def instrument_tu(entry_path: Path, project_root: Path, system_dir: Path,
                  std_flag: str, define_flags: list[str]) -> dict:
    """Parse the entry TU, rewrite every touched user file in place with the
    debug hooks, and return the sidecar dict. Raises RuntimeError only if the
    parse produces no usable AST at all."""
    user_files = _gather_user_files(project_root)
    args = build_parse_args(system_dir, project_root, std_flag, define_flags)

    index = CI.Index.create()
    tu = index.parse(str(entry_path), args=args,
                     options=CI.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
    if tu is None or tu.cursor is None:
        raise RuntimeError("libclang failed to parse the entry translation unit")

    # Read sources once (the rewrite needs the original bytes).
    srcs = {_norm(f): Path(f).read_bytes() for f in user_files}

    inst = Instrumenter(user_files)
    inst.run(tu, srcs)

    for norm_path in inst.touched_files():
        # norm_path is normcased; map back via the original user_files set.
        original = next((f for f in user_files if _norm(f) == norm_path), None)
        if original is None:
            continue
        rewritten = inst.rewrite_file(original, srcs[norm_path])
        Path(original).write_bytes(rewritten)

    sidecar = inst.sidecar()
    # Rewrite file paths to be project-root-relative so the frontend can match
    # them against editor/bundle paths (the absolute temp dir is server-only).
    rel_files = []
    for f in sidecar["files"]:
        try:
            rel_files.append(Path(f).resolve().relative_to(project_root.resolve()).as_posix())
        except Exception:
            rel_files.append(Path(f).name)
    sidecar["files"] = rel_files
    return sidecar

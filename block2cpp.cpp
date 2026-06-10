/*
 * block2cpp.cpp — Blockly JSON → target source (simstd subset)
 *
 * Usage:  reads from stdin:
 *           line 1 : main function name (e.g. "main" or "worker")
 *           rest   : Blockly workspace JSON
 *         Writes the generated source to stdout.
 *
 *         Target language is chosen on the command line:
 *           --target=cpp   (default)  — C++ (simstd subset)
 *           --target=py                — Python
 *           --target=js                — JavaScript
 *
 * The tree-traversal and precedence-aware expression machinery is shared
 * across all targets; everything that differs per language is funnelled
 * through the `Backend` interface (see below).
 *
 * This started as a direct port of the former block2cpp.py converter. The
 * Python `cppize(json, main_fn_name, target)` is a thin subprocess wrapper
 * around this binary (see block2cpp.py).
 */

#include <deque>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

using J = nlohmann::ordered_json;

/* ------------------------------------------------------------------ *
 *  C++ operator precedence (higher = binds tighter)
 *
 *  Shared across all backends. C++/JS/Python precedence differs in
 *  detail, but using this single scheme stays *correct* everywhere —
 *  the worst case is a few redundant parentheses in the non-C++ output.
 * ------------------------------------------------------------------ */
enum {
    P_TERNARY = 2,
    P_LOR     = 3,
    P_LAND    = 4,
    P_BOR     = 5,
    P_BXOR    = 6,
    P_BAND    = 7,
    P_EQ      = 8,
    P_REL     = 9,
    P_SHIFT   = 10,
    P_ADD     = 11,
    P_MUL     = 12,
    P_UNARY   = 13,
    P_POSTFIX = 14,
    P_PRIMARY = 15,
};

struct Expr {
    std::string code;
    int prec = P_PRIMARY;
    Expr() = default;
    Expr(std::string c, int p) : code(std::move(c)), prec(p) {}
};

/* ------------------------------------------------------------------ *
 *  Small string helpers
 * ------------------------------------------------------------------ */
static std::string indent(const std::string &code, int level = 1) {
    std::string pre;
    for (int i = 0; i < level; ++i) pre += "    ";

    // Emulate Python str.splitlines(): '\n' is a line terminator, so a
    // trailing newline does not produce an extra empty line.
    std::vector<std::string> lines;
    std::string cur;
    for (char c : code) {
        if (c == '\n') { lines.push_back(cur); cur.clear(); }
        else cur += c;
    }
    lines.push_back(cur);
    if (!code.empty() && code.back() == '\n' && !lines.empty() && lines.back().empty())
        lines.pop_back();

    std::string out;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i) out += "\n";
        out += pre + lines[i];
    }
    return out;
}

static std::string strip(const std::string &s) {
    size_t a = 0, b = s.size();
    auto isws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; };
    while (a < b && isws(s[a])) ++a;
    while (b > a && isws(s[b - 1])) --b;
    return s.substr(a, b - a);
}

static std::string join(const std::vector<std::string> &parts, const std::string &sep) {
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out += sep;
        out += parts[i];
    }
    return out;
}

static std::string replace_all(std::string s, const std::string &from, const std::string &to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

static bool starts_with(const std::string &s, const std::string &p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

/* ------------------------------------------------------------------ *
 *  Expression constructors (precedence-aware)
 *
 *  These describe purely syntactic shapes that are identical in C++,
 *  JavaScript and Python (infix operators, calls, indexing, member
 *  access). Constructs that genuinely differ per language (casts,
 *  ternaries, statements) live behind the Backend interface instead.
 * ------------------------------------------------------------------ */
static Expr e_binop(const Expr &lhs, const std::string &op, const Expr &rhs, int prec) {
    std::string l = lhs.prec >= prec ? lhs.code : "(" + lhs.code + ")";
    std::string r = rhs.prec > prec ? rhs.code : "(" + rhs.code + ")";
    return Expr(l + " " + op + " " + r, prec);
}

static Expr e_unary(const std::string &op, const Expr &val) {
    std::string v = val.prec > P_UNARY ? val.code : "(" + val.code + ")";
    return Expr(op + v, P_UNARY);
}

static Expr e_cast(const std::string &typ, const Expr &val) {
    std::string v = val.prec >= P_UNARY ? val.code : "(" + val.code + ")";
    return Expr("(" + typ + ")" + v, P_UNARY);
}

static Expr e_ternary(const Expr &cond, const Expr &then, const Expr &els) {
    std::string c = cond.prec > P_TERNARY ? cond.code : "(" + cond.code + ")";
    std::string e = els.prec >= P_TERNARY ? els.code : "(" + els.code + ")";
    return Expr(c + " ? " + then.code + " : " + e, P_TERNARY);
}

static Expr e_member(const Expr &obj, const std::string &field) {
    std::string o = obj.prec >= P_POSTFIX ? obj.code : "(" + obj.code + ")";
    return Expr(o + "." + field, P_POSTFIX);
}

static Expr e_index(const Expr &arr, const Expr &idx) {
    std::string a = arr.prec >= P_POSTFIX ? arr.code : "(" + arr.code + ")";
    return Expr(a + "[" + idx.code + "]", P_POSTFIX);
}

static Expr e_call(const std::string &fn, const std::vector<Expr> &args) {
    std::vector<std::string> codes;
    for (auto &a : args) codes.push_back(a.code);
    return Expr(fn + "(" + join(codes, ", ") + ")", P_POSTFIX);
}

static Expr e_stmt(const std::string &code) { return Expr(code, P_PRIMARY); }

/* ------------------------------------------------------------------ *
 *  Operator tables
 * ------------------------------------------------------------------ */
struct OpEntry { std::string sym; int prec; };

static const std::unordered_map<std::string, OpEntry> I32_BINOP = {
    {"add", {"+", P_ADD}},  {"sub", {"-", P_ADD}},   {"mul", {"*", P_MUL}},
    {"div_s", {"/", P_MUL}}, {"rem_s", {"%", P_MUL}}, {"and", {"&", P_BAND}},
    {"or", {"|", P_BOR}},   {"xor", {"^", P_BXOR}},  {"shl", {"<<", P_SHIFT}},
    {"shr_s", {">>", P_SHIFT}},
    // legacy aliases
    {"div", {"/", P_MUL}},  {"mod", {"%", P_MUL}},   {"lsh", {"<<", P_SHIFT}},
    {"rsh", {">>", P_SHIFT}},
};

static const std::unordered_map<std::string, OpEntry> F64_BINOP = {
    {"add", {"+", P_ADD}}, {"sub", {"-", P_ADD}}, {"mul", {"*", P_MUL}}, {"div", {"/", P_MUL}},
};

static const std::unordered_map<std::string, OpEntry> BOOL_BINOP = {
    {"and", {"&&", P_LAND}}, {"or", {"||", P_LOR}}, {"xor", {"^", P_BXOR}},
};

static const std::unordered_map<std::string, OpEntry> CMP_OP = {
    {"eq", {"==", P_EQ}},   {"ne", {"!=", P_EQ}},   {"lt", {"<", P_REL}},
    {"le", {"<=", P_REL}},  {"gt", {">", P_REL}},   {"ge", {">=", P_REL}},
    {"lt_s", {"<", P_REL}}, {"le_s", {"<=", P_REL}}, {"gt_s", {">", P_REL}},
    {"ge_s", {">=", P_REL}},
};

static const OpEntry &lookup(const std::unordered_map<std::string, OpEntry> &m,
                             const std::string &k, const char *what) {
    auto it = m.find(k);
    if (it == m.end()) throw std::runtime_error(std::string("Unknown ") + what + ": " + k);
    return it->second;
}

/* ------------------------------------------------------------------ *
 *  Backend interface
 *
 *  Everything that differs between target languages is expressed here.
 *  compile_block() drives the shared logic and delegates the
 *  language-specific spellings to the active backend.
 * ------------------------------------------------------------------ */
struct Backend {
    virtual ~Backend() = default;

    // File header emitted once, before any function.
    virtual std::string prelude() const = 0;
    // Statement terminator: ";" for C-like languages, "" for Python.
    virtual std::string terminate(const std::string &s) const = 0;

    // Literals / value-producing expressions.
    virtual std::string boolean(bool v) const = 0;
    virtual Expr cast_i32(const Expr &v) const = 0;
    virtual Expr cast_f64(const Expr &v) const = 0;
    virtual Expr ternary(const Expr &c, const Expr &t, const Expr &e) const = 0;

    // `kind` is a logical type tag: i32, f64, vec2, vec3, tensor,
    // array_i32, array_f64, bd2, bd3.
    virtual std::string decl(const std::string &kind, const std::string &name,
                             const std::string &init) const = 0;          // typed binding with initializer
    virtual std::string decl_default(const std::string &kind, const std::string &name) const = 0;  // default-constructed
    virtual std::string array_alloc(const std::string &elem, const std::string &name,
                                    const std::string &size) const = 0;   // zero-filled array of `size`
    virtual std::string array_literal(const std::vector<std::string> &elems,
                                      const std::string &elem) const = 0;
    virtual Expr array_len(const Expr &arr) const = 0;

    // Tensor / matrix value-producing constructs (numpy on the Python target).
    virtual Expr tensor_new(const std::vector<std::string> &dims) const = 0;
    virtual std::string tensor_index(const std::string &name,
                                     const std::vector<std::string> &idx) const = 0;
    virtual Expr matrix_create(const Expr &rows, const Expr &cols) const = 0;

    // Statements / control flow (return the statement text, no trailing
    // terminator — callers add it where appropriate).
    virtual std::string func(const std::string &ret, const std::string &name,
                             const std::string &body) const = 0;
    // Custom function with parameters; `params` is a list of (name, type-kind).
    virtual std::string func_params(const std::string &ret, const std::string &name,
                                    const std::vector<std::pair<std::string, std::string>> &params,
                                    const std::string &body) const = 0;
    // Struct/record type definition; `fields` is a list of (name, "i32"|"f64").
    virtual std::string struct_def(const std::string &name,
                                   const std::vector<std::pair<std::string, std::string>> &fields) const = 0;
    // Default-constructed struct value expression (e.g. `Pair()` / `new Pair()`).
    virtual std::string struct_value(const std::string &name) const = 0;
    virtual std::string if_stmt(const std::string &cond, const std::string &body, bool has_body) const = 0;
    virtual std::string if_else(const std::string &cond, const std::string &then_body,
                                const std::string &else_body) const = 0;
    virtual std::string for_range(const std::string &var, const std::string &start,
                                  const std::string &end, const std::string &body) const = 0;
    virtual std::string while_stmt(const std::string &cond, const std::string &body) const = 0;
    virtual std::string region(const std::string &name, const std::string &body, bool has_body) const = 0;

    // Maps a logical builtin key (e.g. "sqrt", "clz", "min") to the
    // language's spelling. Defaults to identity in each backend's table.
    virtual std::string builtin(const std::string &key) const = 0;
    // Lets a backend remap an i32 binop symbol (e.g. Python integer "//").
    virtual std::string i32_sym(const std::string &op, const std::string &sym) const { return sym; }
};

/* ---- C++ backend (reference behaviour) ---- */
struct CppBackend : Backend {
    std::unordered_map<std::string, std::string> blt;
    std::unordered_map<std::string, std::string> tymap;

    CppBackend() {
        blt = {
            {"clz", "__builtin_clz"}, {"ctz", "__builtin_ctz"}, {"popcnt", "__builtin_popcount"},
            {"abs", "std::abs"},   {"sqrt", "std::sqrt"},  {"ceil", "std::ceil"},
            {"floor", "std::floor"}, {"trunc", "std::trunc"}, {"nearest", "std::round"},
            {"exp", "std::exp"},   {"ln", "std::log"},     {"cos", "std::cos"},
            {"sin", "std::sin"},   {"min", "std::fmin"},   {"max", "std::fmax"},
        };
        tymap = {
            {"i32", "i32"}, {"f64", "f64"}, {"vec2", "vec2"}, {"vec3", "vec3"},
            {"tensor", "Tensor<f64>"}, {"array_i32", "std::vector<i32>"},
            {"array_f64", "std::vector<f64>"}, {"bd2", "Boundary2D"}, {"bd3", "Boundary3D"},
        };
    }

    std::string spell(const std::string &k) const {
        auto it = tymap.find(k);
        return it == tymap.end() ? k : it->second;
    }

    std::string prelude() const override { return "#include \"simstd.hpp\"\n\n"; }
    std::string terminate(const std::string &s) const override { return s + ";"; }
    std::string boolean(bool v) const override { return v ? "true" : "false"; }
    Expr cast_i32(const Expr &v) const override { return e_cast("i32", v); }
    Expr cast_f64(const Expr &v) const override { return e_cast("f64", v); }
    Expr ternary(const Expr &c, const Expr &t, const Expr &e) const override { return e_ternary(c, t, e); }

    std::string decl(const std::string &kind, const std::string &name, const std::string &init) const override {
        return spell(kind) + " " + name + " = " + init;
    }
    std::string decl_default(const std::string &kind, const std::string &name) const override {
        return spell(kind) + " " + name;
    }
    std::string array_alloc(const std::string &elem, const std::string &name, const std::string &size) const override {
        std::string fill = elem == "i32" ? "0" : "0.0";
        return "std::vector<" + elem + "> " + name + "(" + size + ", " + fill + ")";
    }
    std::string array_literal(const std::vector<std::string> &elems, const std::string &) const override {
        return "{" + join(elems, ", ") + "}";
    }
    Expr array_len(const Expr &arr) const override { return e_member(arr, "size()"); }

    Expr tensor_new(const std::vector<std::string> &dims) const override {
        return Expr("Tensor<f64>({" + join(dims, ", ") + "})", P_POSTFIX);
    }
    std::string tensor_index(const std::string &name, const std::vector<std::string> &idx) const override {
        return name + "(" + join(idx, ", ") + ")";
    }
    Expr matrix_create(const Expr &rows, const Expr &cols) const override {
        return e_call("matrix_create", {rows, cols});
    }

    std::string func(const std::string &ret, const std::string &name, const std::string &body) const override {
        return ret + " " + name + "() {\n" + indent(body) + "\n}";
    }
    std::string func_params(const std::string &ret, const std::string &name,
                            const std::vector<std::pair<std::string, std::string>> &params,
                            const std::string &body) const override {
        std::string ps;
        for (size_t i = 0; i < params.size(); ++i) {
            if (i) ps += ", ";
            ps += spell(params[i].second) + " " + params[i].first;
        }
        return ret + " " + name + "(" + ps + ") {\n" + indent(body) + "\n}";
    }
    std::string struct_def(const std::string &name,
                           const std::vector<std::pair<std::string, std::string>> &fields) const override {
        std::string body;
        for (auto &f : fields)
            body += "    " + spell(f.second) + " " + f.first + " = " + (f.second == "i32" ? "0" : "0.0") + ";\n";
        return "struct " + name + " {\n" + body + "};";
    }
    std::string struct_value(const std::string &name) const override { return name + "()"; }
    std::string if_stmt(const std::string &cond, const std::string &body, bool has_body) const override {
        if (has_body) return "if (" + cond + ") {\n" + indent(body) + "\n}";
        return "if (" + cond + ") {\n}";
    }
    std::string if_else(const std::string &cond, const std::string &then_body, const std::string &else_body) const override {
        return "if (" + cond + ") {\n" + indent(then_body) + "\n} else {\n" + indent(else_body) + "\n}";
    }
    virtual std::string loop_kw() const { return "i32"; }
    std::string for_range(const std::string &var, const std::string &start,
                          const std::string &end, const std::string &body) const override {
        return "for (" + loop_kw() + " " + var + " = " + start + "; " + var + " < " + end +
               "; " + var + "++) {\n" + indent(body) + "\n}";
    }
    std::string while_stmt(const std::string &cond, const std::string &body) const override {
        return "while (" + cond + ") {\n" + indent(body) + "\n}";
    }
    std::string region(const std::string &name, const std::string &body, bool has_body) const override {
        if (has_body) return "#pragma region " + name + "\n" + body + "\n#pragma endregion";
        return "#pragma region " + name + "\n#pragma endregion";
    }
    std::string builtin(const std::string &key) const override {
        auto it = blt.find(key);
        return it == blt.end() ? key : it->second;
    }
};

/* ---- JavaScript backend (reuses the C-like structure) ---- */
struct JsBackend : CppBackend {
    JsBackend() {
        blt = {
            {"clz", "Math.clz32"}, {"ctz", "ctz"}, {"popcnt", "popcount"},
            {"abs", "Math.abs"},   {"sqrt", "Math.sqrt"},  {"ceil", "Math.ceil"},
            {"floor", "Math.floor"}, {"trunc", "Math.trunc"}, {"nearest", "Math.round"},
            {"exp", "Math.exp"},   {"ln", "Math.log"},     {"cos", "Math.cos"},
            {"sin", "Math.sin"},   {"min", "Math.min"},    {"max", "Math.max"},
        };
        // tymap inherited; used only for default-constructed object names.
    }

    std::string prelude() const override { return "import * as simstd from \"./simstd.js\";\n\n"; }
    Expr cast_i32(const Expr &v) const override { return e_binop(v, "|", Expr("0", P_PRIMARY), P_BOR); }
    Expr cast_f64(const Expr &v) const override { return v; }  // JS numbers are already double

    std::string decl(const std::string &, const std::string &name, const std::string &init) const override {
        return "let " + name + " = " + init;
    }
    std::string decl_default(const std::string &kind, const std::string &name) const override {
        return "let " + name + " = new " + spell(kind) + "()";
    }
    std::string array_alloc(const std::string &elem, const std::string &name, const std::string &size) const override {
        std::string fill = elem == "i32" ? "0" : "0.0";
        return "let " + name + " = new Array(" + size + ").fill(" + fill + ")";
    }
    std::string array_literal(const std::vector<std::string> &elems, const std::string &) const override {
        return "[" + join(elems, ", ") + "]";
    }
    std::string func(const std::string &, const std::string &name, const std::string &body) const override {
        return "function " + name + "() {\n" + indent(body) + "\n}";
    }
    std::string func_params(const std::string &, const std::string &name,
                            const std::vector<std::pair<std::string, std::string>> &params,
                            const std::string &body) const override {
        std::string ps;
        for (size_t i = 0; i < params.size(); ++i) { if (i) ps += ", "; ps += params[i].first; }
        return "function " + name + "(" + ps + ") {\n" + indent(body) + "\n}";
    }
    std::string struct_def(const std::string &name,
                           const std::vector<std::pair<std::string, std::string>> &fields) const override {
        std::string ctor;
        for (auto &f : fields)
            ctor += "        this." + f.first + " = " + (f.second == "i32" ? "0" : "0.0") + ";\n";
        return "class " + name + " {\n    constructor() {\n" + ctor + "    }\n}";
    }
    std::string struct_value(const std::string &name) const override { return "new " + name + "()"; }
    std::string loop_kw() const override { return "let"; }
    std::string region(const std::string &name, const std::string &body, bool has_body) const override {
        if (has_body) return "// #region " + name + "\n" + body + "\n// #endregion";
        return "// #region " + name + "\n// #endregion";
    }
};

/* ---- Python backend (indentation-based, no braces) ---- */
struct PyBackend : Backend {
    std::unordered_map<std::string, std::string> blt;
    std::unordered_map<std::string, std::string> ctormap;

    PyBackend() {
        blt = {
            // Math → numpy. clz/ctz/popcnt have no numpy form, so they fall
            // back to the simstd helpers (also imported below).
            {"clz", "clz"}, {"ctz", "ctz"}, {"popcnt", "popcount"},
            {"abs", "np.abs"},   {"sqrt", "np.sqrt"},  {"ceil", "np.ceil"},
            {"floor", "np.floor"}, {"trunc", "np.trunc"}, {"nearest", "np.round"},
            {"exp", "np.exp"},   {"ln", "np.log"},     {"cos", "np.cos"},
            {"sin", "np.sin"},   {"min", "np.minimum"}, {"max", "np.maximum"},
            // Matrix ops that are plain name swaps onto numpy.
            {"matrix_identity", "np.eye"},      {"matrix_matmul", "np.matmul"},
            {"matrix_transpose", "np.transpose"}, {"matrix_inverse", "np.linalg.inv"},
            {"matrix_det", "np.linalg.det"},    {"matrix_trace", "np.trace"},
        };
        ctormap = { {"bd2", "Boundary2D"}, {"bd3", "Boundary3D"} };
    }

    // numpy for arrays/tensors/matrices/math; simstd for vec/boundary/debug.
    std::string prelude() const override { return "import numpy as np\nfrom simstd import *\n\n"; }
    std::string terminate(const std::string &s) const override { return s; }
    std::string boolean(bool v) const override { return v ? "True" : "False"; }
    Expr cast_i32(const Expr &v) const override { return e_call("int", {v}); }
    Expr cast_f64(const Expr &v) const override { return e_call("float", {v}); }
    Expr ternary(const Expr &c, const Expr &t, const Expr &e) const override {
        return Expr("(" + t.code + " if " + c.code + " else " + e.code + ")", P_PRIMARY);
    }

    std::string decl(const std::string &, const std::string &name, const std::string &init) const override {
        return name + " = " + init;
    }
    std::string decl_default(const std::string &kind, const std::string &name) const override {
        auto it = ctormap.find(kind);
        std::string ctor = it == ctormap.end() ? kind : it->second;
        return name + " = " + ctor + "()";
    }
    std::string array_alloc(const std::string &elem, const std::string &name, const std::string &size) const override {
        if (elem == "i32") return name + " = np.zeros(" + size + ", dtype=np.int32)";
        return name + " = np.zeros(" + size + ")";
    }
    std::string array_literal(const std::vector<std::string> &elems, const std::string &elem) const override {
        if (elem == "i32") return "np.array([" + join(elems, ", ") + "], dtype=np.int32)";
        return "np.array([" + join(elems, ", ") + "])";
    }
    Expr array_len(const Expr &arr) const override { return e_call("len", {arr}); }

    Expr tensor_new(const std::vector<std::string> &dims) const override {
        return Expr("np.zeros((" + join(dims, ", ") + "))", P_POSTFIX);
    }
    std::string tensor_index(const std::string &name, const std::vector<std::string> &idx) const override {
        return name + "[" + join(idx, ", ") + "]";
    }
    Expr matrix_create(const Expr &rows, const Expr &cols) const override {
        return Expr("np.zeros((" + rows.code + ", " + cols.code + "))", P_POSTFIX);
    }

    std::string func(const std::string &, const std::string &name, const std::string &body) const override {
        return "def " + name + "():\n" + indent(body);
    }
    std::string func_params(const std::string &, const std::string &name,
                            const std::vector<std::pair<std::string, std::string>> &params,
                            const std::string &body) const override {
        std::string ps;
        for (size_t i = 0; i < params.size(); ++i) { if (i) ps += ", "; ps += params[i].first; }
        return "def " + name + "(" + ps + "):\n" + indent(body);
    }
    std::string struct_def(const std::string &name,
                           const std::vector<std::pair<std::string, std::string>> &fields) const override {
        std::string body;
        for (auto &f : fields)
            body += "        self." + f.first + " = " + (f.second == "i32" ? "0" : "0.0") + "\n";
        if (fields.empty()) body = "        pass\n";
        if (!body.empty() && body.back() == '\n') body.pop_back();
        return "class " + name + ":\n    def __init__(self):\n" + body;
    }
    std::string struct_value(const std::string &name) const override { return name + "()"; }
    std::string if_stmt(const std::string &cond, const std::string &body, bool has_body) const override {
        return "if " + cond + ":\n" + indent(has_body ? body : "pass");
    }
    std::string if_else(const std::string &cond, const std::string &then_body, const std::string &else_body) const override {
        return "if " + cond + ":\n" + indent(then_body) + "\nelse:\n" + indent(else_body);
    }
    std::string for_range(const std::string &var, const std::string &start,
                          const std::string &end, const std::string &body) const override {
        return "for " + var + " in range(" + start + ", " + end + "):\n" + indent(body);
    }
    std::string while_stmt(const std::string &cond, const std::string &body) const override {
        return "while " + cond + ":\n" + indent(body);
    }
    std::string region(const std::string &name, const std::string &body, bool has_body) const override {
        if (has_body) return "# region " + name + "\n" + body + "\n# endregion";
        return "# region " + name + "\n# endregion";
    }
    std::string builtin(const std::string &key) const override {
        auto it = blt.find(key);
        return it == blt.end() ? key : it->second;
    }
    std::string i32_sym(const std::string &op, const std::string &sym) const override {
        if (op == "div" || op == "div_s") return "//";  // Python integer division
        return sym;
    }
};

/* ------------------------------------------------------------------ *
 *  JSON accessors
 * ------------------------------------------------------------------ */
static std::string node_type(const J &n) {
    if (!n.is_object() || !n.contains("type")) return "";
    return n["type"].get<std::string>();
}

// inputs[key]["block"]  ->  pointer to child block (or nullptr)
static const J *child(const J &n, const char *key) {
    if (!n.contains("inputs")) return nullptr;
    const J &ins = n["inputs"];
    if (!ins.is_object() || !ins.contains(key)) return nullptr;
    const J &v = ins[key];
    if (!v.is_object() || !v.contains("block")) return nullptr;
    return &v["block"];
}

static std::string fstr(const J &n, const char *key) {
    if (!n.contains("fields")) return "";
    const J &f = n["fields"];
    if (!f.is_object() || !f.contains(key)) return "";
    const J &v = f[key];
    if (v.is_string()) return v.get<std::string>();
    return v.dump();  // numbers render like Python str() (shortest round-trip)
}

static long fint(const J &n, const char *key) {
    if (!n.contains("fields")) return 0;
    const J &f = n["fields"];
    if (!f.is_object() || !f.contains(key)) return 0;
    return f[key].get<long>();
}

// Child blocks in execution order: inputs first (in order), then next.
static std::vector<const J *> children(const J &b) {
    std::vector<const J *> out;
    if (b.contains("inputs") && b["inputs"].is_object()) {
        for (auto it = b["inputs"].begin(); it != b["inputs"].end(); ++it) {
            const J &v = it.value();
            if (v.is_object() && v.contains("block")) out.push_back(&v["block"]);
        }
    }
    if (b.contains("next") && b["next"].is_object() && b["next"].contains("block"))
        out.push_back(&b["next"]["block"]);
    return out;
}

/* ------------------------------------------------------------------ *
 *  Pre-order scan: which tensor_save blocks first introduce a name.
 * ------------------------------------------------------------------ */
static std::unordered_set<const J *> scan_tensor_decls(const J &root) {
    std::unordered_set<std::string> seen;
    std::unordered_set<const J *> first;
    std::deque<const J *> stack;
    stack.push_back(&root);
    while (!stack.empty()) {
        const J *cur = stack.back();
        stack.pop_back();
        if (node_type(*cur) == "tensor_save") {
            std::string name = fstr(*cur, "NAME");
            if (!seen.count(name)) {
                seen.insert(name);
                first.insert(cur);
            }
        }
        auto ch = children(*cur);
        for (auto it = ch.rbegin(); it != ch.rend(); ++it) stack.push_back(*it);
    }
    return first;
}

/* ------------------------------------------------------------------ *
 *  Per-block compilation. Children are already compiled & cached.
 * ------------------------------------------------------------------ */
using Results = std::unordered_map<const J *, Expr>;

// A custom function's call-block type is `custom_func_<funcId>`, where funcId is
// the definition block's Blockly id with every non-alphanumeric char mapped to
// '_' (mirrors funcIdOf() in the frontend). Lets call blocks resolve to the
// function's user-facing name + arity.
static std::string funcIdOf(const std::string &blocklyId) {
    std::string out = "f";
    for (char c : blocklyId) {
        bool alnum = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        out += alnum ? c : '_';
    }
    return out;
}

struct FuncInfo { std::string name; std::string ret; int paramCount; };
using FuncTable = std::unordered_map<std::string, FuncInfo>;

// A struct's per-instance block type is `struct_<structId>_{decl,ref,get,set}`,
// where structId is the definition block's Blockly id sanitized with 's' prefix
// (mirrors structIdOf() in the frontend). Lets instance blocks resolve to the
// struct's user-facing name + field layout.
static std::string structIdOf(const std::string &blocklyId) {
    std::string out = "s";
    for (char c : blocklyId) {
        bool alnum = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        out += alnum ? c : '_';
    }
    return out;
}

struct StructInfo { std::string name; std::vector<std::pair<std::string, std::string>> fields; };
using StructTable = std::unordered_map<std::string, StructInfo>;  // key = structIdOf(blockId)

static int extra_param_count(const J &node) {
    if (node.contains("extraState") && node["extraState"].is_object()
        && node["extraState"].contains("paramCount"))
        return (int)node["extraState"]["paramCount"].get<long>();
    return 0;
}

static Expr compile_block(const J &node, const Results &results,
                          const std::string &main_fn_name,
                          const std::unordered_set<const J *> &first_decls,
                          const FuncTable &funcs,
                          const StructTable &structs,
                          const Backend &B) {
    const std::string T = node_type(node);

    auto D = [&](const J *b) -> const Expr & {
        if (!b) throw std::runtime_error("missing required child block in '" + T + "'");
        auto it = results.find(b);
        if (it == results.end()) throw std::runtime_error("child block not compiled in '" + T + "'");
        return it->second;
    };
    auto C = [&](const char *k) -> const J * { return child(node, k); };
    auto F = [&](const char *k) -> std::string { return fstr(node, k); };

    // _putnext: code of the `next` block, prefixed with a newline.
    std::string putnext;
    if (node.contains("next") && node["next"].is_object() && node["next"].contains("block"))
        putnext = "\n" + D(&node["next"]["block"]).code;

    // Statement helpers: term() appends the language's terminator,
    // S() builds a terminated statement followed by any `next` block.
    auto term = [&](const std::string &s) { return B.terminate(s); };
    auto S = [&](const std::string &s) { return e_stmt(term(s) + putnext); };

    /* ---- function root ---- */
    if (T == "wasm_func_main") {
        std::string ret_type = F("RET_TYPE");
        if (ret_type != "i32" && ret_type != "f64" && ret_type != "void")
            throw std::runtime_error("Unsupported RET_TYPE: " + ret_type);
        std::string body_code = D(C("BODY")).code;
        return e_stmt(B.func(ret_type, main_fn_name, body_code));
    }

    /* ---- custom function definition ---- */
    if (T == "custom_func_def") {
        std::string ret = F("RET");
        if (starts_with(ret, "struct_")) {
            // Struct return: resolve the struct's user-facing type name.
            std::string sid = ret.substr(std::string("struct_").size());
            auto sit = structs.find(sid);
            ret = (sit != structs.end()) ? sit->second.name : sid;
        } else if (ret != "i32" && ret != "f64" && ret != "void") {
            throw std::runtime_error("Unsupported custom function RET: " + ret);
        }
        std::string fname = F("NAME");
        int pc = extra_param_count(node);
        std::vector<std::pair<std::string, std::string>> params;
        for (int i = 0; i < pc; ++i) {
            std::string pname = fstr(node, ("PNAME" + std::to_string(i)).c_str());
            std::string ptype = fstr(node, ("PTYPE" + std::to_string(i)).c_str());
            if (starts_with(ptype, "struct_")) {
                // Struct param: resolve to the struct's user-facing type name.
                std::string sid = ptype.substr(std::string("struct_").size());
                auto sit = structs.find(sid);
                ptype = (sit != structs.end()) ? sit->second.name : sid;
            }
            params.push_back({ pname, ptype });
        }
        std::string body_code = D(C("BODY")).code;
        return e_stmt(B.func_params(ret, fname, params, body_code));
    }

    /* ---- custom function call (`custom_func_<id>`) ---- */
    if (starts_with(T, "custom_func_") && T != "custom_func_def") {
        std::string id = T.substr(std::string("custom_func_").size());
        auto it = funcs.find(id);
        std::string fname = (it != funcs.end()) ? it->second.name : id;
        int pc = (it != funcs.end()) ? it->second.paramCount : 0;
        std::vector<Expr> args;
        for (int i = 0; i < pc; ++i)
            args.push_back(D(C(("ARG" + std::to_string(i)).c_str())));
        if (it != funcs.end() && it->second.ret == "void")
            return S(e_call(fname, args).code);
        return e_call(fname, args);
    }

    /* ---- struct definition ---- */
    if (T == "struct_def") {
        std::string name = F("NAME");
        std::vector<std::pair<std::string, std::string>> fields;
        const J *f = C("FIELDS");
        while (f) {
            if (node_type(*f) == "struct_field")
                fields.push_back({ fstr(*f, "FNAME"), fstr(*f, "FTYPE") });
            f = (f->contains("next") && (*f)["next"].is_object() && (*f)["next"].contains("block"))
                ? &(*f)["next"]["block"] : nullptr;
        }
        return e_stmt(B.struct_def(name, fields));
    }
    if (T == "struct_field") return e_stmt("");  // consumed by struct_def

    /* ---- struct instance ops (`struct_<id>_{decl,ref,get,set}`) ---- */
    if (starts_with(T, "struct_") && T != "struct_def" && T != "struct_field") {
        size_t last = T.rfind('_');
        std::string op  = T.substr(last + 1);
        std::string sid = T.substr(std::string("struct_").size(),
                                   last - std::string("struct_").size());
        auto it = structs.find(sid);
        std::string sname = (it != structs.end()) ? it->second.name : sid;
        std::string var = F("VAR");
        if (op == "ref") return Expr(var, P_PRIMARY);
        if (op == "get") return e_member(Expr(var, P_PRIMARY), F("FIELD"));
        if (op == "set") return S(var + "." + F("FIELD") + " = " + D(C("VALUE")).code);
        if (op == "decl") {
            const J *init = C("INIT");
            if (init) return S(B.decl(sname, var, D(init).code));
            return S(B.decl_default(sname, var));
        }
    }

    /* ---- empty / default value ---- */
    if (T == "empty_value") {
        std::string ty = F("TYPE");
        if (ty == "f64") return Expr("0.0", P_PRIMARY);
        if (starts_with(ty, "struct_")) {
            std::string sid = ty.substr(std::string("struct_").size());
            auto it = structs.find(sid);
            std::string sname = (it != structs.end()) ? it->second.name : sid;
            return Expr(B.struct_value(sname), P_POSTFIX);
        }
        return Expr("0", P_PRIMARY);
    }

    /* ---- constants ---- */
    if (T == "i32_const") return Expr(F("VALUE"), P_PRIMARY);
    if (T == "f64_const") return Expr(F("VALUE"), P_PRIMARY);
    if (T == "bool_const") {
        const J &v = node["fields"]["VALUE"];
        bool tv = (v.is_boolean() && v.get<bool>()) || (v.is_string() && v.get<std::string>() == "true");
        return Expr(B.boolean(tv), P_PRIMARY);
    }

    /* ---- binary / comparison operators ---- */
    if (T == "bool_binop") {
        const OpEntry &o = lookup(BOOL_BINOP, F("OP"), "bool binop");
        return e_binop(D(C("LHS")), o.sym, D(C("RHS")), o.prec);
    }
    if (T == "i32_binop") {
        std::string op = F("OP");
        const OpEntry &o = lookup(I32_BINOP, op, "i32 binop");
        return e_binop(D(C("LHS")), B.i32_sym(op, o.sym), D(C("RHS")), o.prec);
    }
    if (T == "f64_binop") {
        std::string op = F("OP");
        if (op == "min") return e_call(B.builtin("min"), {D(C("LHS")), D(C("RHS"))});
        if (op == "max") return e_call(B.builtin("max"), {D(C("LHS")), D(C("RHS"))});
        const OpEntry &o = lookup(F64_BINOP, op, "f64 binop");
        return e_binop(D(C("LHS")), o.sym, D(C("RHS")), o.prec);
    }
    if (T == "i32_cmp" || T == "f64_cmp") {
        const OpEntry &o = lookup(CMP_OP, F("OP"), "compare");
        return e_binop(D(C("LHS")), o.sym, D(C("RHS")), o.prec);
    }

    /* ---- locals ---- */
    if (T == "local_get_i32" || T == "local_get_f64") return Expr(F("NAME"), P_PRIMARY);

    if (T == "local_decl_i32") return S(B.decl("i32", F("NAME"), D(C("INIT")).code));
    if (T == "local_set_i32")  return S(F("NAME") + " = " + D(C("VALUE")).code);
    if (T == "local_decl_f64") return S(B.decl("f64", F("NAME"), D(C("INIT")).code));
    if (T == "local_set_f64")  return S(F("NAME") + " = " + D(C("VALUE")).code);

    if (T == "local_array_decl_i32") return S(B.array_alloc("i32", F("NAME"), D(C("SIZE")).code));
    if (T == "local_array_decl_f64") return S(B.array_alloc("f64", F("NAME"), D(C("SIZE")).code));

    if (T == "local_array_get_i32" || T == "local_array_get_f64") return Expr(F("NAME"), P_PRIMARY);

    if (T == "array_get_i32" || T == "array_get_f64")
        return e_index(D(C("ARRAY")), D(C("INDEX")));

    if (T == "array_set_i32" || T == "array_set_f64")
        return S(e_index(D(C("ARRAY")), D(C("INDEX"))).code + " = " + D(C("VALUE")).code);

    if (T == "array_assign_i32") return S(B.decl("array_i32", F("NAME"), D(C("PTR")).code));
    if (T == "array_assign_f64") return S(B.decl("array_f64", F("NAME"), D(C("PTR")).code));

    if (T == "array_len_i32" || T == "array_len_f64")
        return B.array_len(D(C("ARRAY")));

    /* ---- conversions / unary ---- */
    if (T == "i32_from_f64") return B.cast_i32(D(C("VALUE")));
    if (T == "f64_from_i32") return B.cast_f64(D(C("VALUE")));
    if (T == "bool_not") return e_unary("!", D(C("VALUE")));

    if (T == "i32_unop") {
        std::string op = F("OP");
        const Expr &v = D(C("VALUE"));
        if (op == "neg") return e_unary("-", v);
        if (op == "clz") return e_call(B.builtin("clz"), {v});
        if (op == "ctz") return e_call(B.builtin("ctz"), {v});
        if (op == "popcnt") return e_call(B.builtin("popcnt"), {v});
        if (op == "eqz") return e_binop(v, "==", Expr("0", P_PRIMARY), P_EQ);
        throw std::runtime_error("Unsupported i32 unop: " + op);
    }
    if (T == "f64_unop") {
        std::string op = F("OP");
        const Expr &v = D(C("VALUE"));
        if (op == "neg") return e_unary("-", v);
        static const std::unordered_set<std::string> known = {
            "abs", "sqrt", "ceil", "floor", "trunc", "nearest", "exp", "ln", "cos", "sin",
        };
        if (!known.count(op)) throw std::runtime_error("Unsupported f64 unop: " + op);
        return e_call(B.builtin(op), {v});
    }

    /* ---- random ---- */
    if (T == "i32_random")       return e_call("sim_rand_int",   {D(C("MIN")), D(C("MAX"))});
    if (T == "f64_random_range") return e_call("sim_rand_range", {D(C("MIN")), D(C("MAX"))});
    if (T == "f64_random")       return e_call("sim_rand",       {});

    /* ---- input (single-function form; recovered 1:1 by cpp2block) ---- */
    if (T == "input_i32") return e_call("sim_input_int",   {});
    if (T == "input_f64") return e_call("sim_input_float", {});

    /* ---- control flow ---- */
    if (T == "flow_if") {
        const J *cond = C("COND");
        if (!cond) throw std::runtime_error("Unsupported block type: flow_if (no COND)");
        std::string cond_code = D(cond).code;
        const J *then_block = C("THEN");
        if (then_block)
            return e_stmt(B.if_stmt(cond_code, D(then_block).code, true) + putnext);
        return e_stmt(B.if_stmt(cond_code, "", false) + putnext);
    }
    if (T == "flow_if_else") {
        const J *cond = C("COND");
        if (!cond) throw std::runtime_error("Unsupported block type: flow_if_else (no COND)");
        std::string cond_code = D(cond).code;
        const J *then_block = C("THEN");
        const J *else_block = C("ELSE");
        if (then_block && else_block)
            return e_stmt(B.if_else(cond_code, D(then_block).code, D(else_block).code) + putnext);
        if (then_block)
            return e_stmt(B.if_stmt(cond_code, D(then_block).code, true) + putnext);
        return e_stmt(B.if_stmt(cond_code, "", false) + putnext);
    }
    if (T == "flow_for") {
        std::string var = F("VAR");
        return e_stmt(B.for_range(var, D(C("START")).code, D(C("END")).code, D(C("BODY")).code) + putnext);
    }
    if (T == "flow_break") return S("break");
    if (T == "flow_while")
        return e_stmt(B.while_stmt(D(C("COND")).code, D(C("BODY")).code) + putnext);
    if (T == "flow_fold_region") {
        std::string name = F("NAME");
        const J *body = C("BODY");
        if (body) return e_stmt(B.region(name, D(body).code, true) + putnext);
        return e_stmt(B.region(name, "", false) + putnext);
    }

    /* ---- returns ---- */
    if (T == "wasm_return_i32") {
        const J *value = C("VALUE");
        if (value) return S("return " + D(value).code);
        return S("return 0");
    }
    if (T == "wasm_return_f64")
        return S("return " + D(C("VALUE")).code);
    if (T == "wasm_return_struct")
        return S("return " + D(C("VALUE")).code);

    /* ---- debug ---- */
    if (T == "debug_graph_array") {
        const J *inner = C("VALUE");
        std::string it = inner ? node_type(*inner) : "";
        if (it == "local_array_get_i32")
            return S("graph_arr_i32(" + fstr(*inner, "NAME") + ")");
        if (it == "local_array_get_f64")
            return S("graph_arr_f64(" + fstr(*inner, "NAME") + ")");
        throw std::runtime_error("Unsupported block type: debug_graph_array");
    }
    if (T == "debug_graph_array_range") {
        const J *inner = C("VALUE");
        std::string it = inner ? node_type(*inner) : "";
        if (it == "local_array_get_i32")
            return S("graph_arr_range_i32(" + fstr(*inner, "NAME") + ", " +
                     D(C("MIN")).code + ", " + D(C("MAX")).code + ")");
        if (it == "local_array_get_f64")
            return S("graph_arr_range_f64(" + fstr(*inner, "NAME") + ", " +
                     D(C("MIN")).code + ", " + D(C("MAX")).code + ")");
        throw std::runtime_error("Unsupported block type: debug_graph_array_range");
    }
    if (T == "debug_log")
        return S("debug_log(" + D(C("VALUE")).code + ")");
    if (T == "debug_series")
        return Expr("debug_series()", P_POSTFIX);
    if (T == "debug_set_holder")
        return S("debug_set_holder(" + D(C("ID")).code + ")");
    if (T == "debug_bar")
        return e_call("debug_bar", {D(C("MIN")), D(C("MAX"))});
    if (T == "debug_bar_set")
        return S("debug_bar_set(" + D(C("ID")).code + ", " + D(C("VALUE")).code + ")");

    /* ---- tensors ---- */
    if (T == "tensor_random") {
        std::string dist = F("DIST");
        std::string fn = dist == "0" ? "tensor_uniform" : dist == "1" ? "tensor_normal" : "";
        if (fn.empty()) throw std::runtime_error("Unsupported tensor_random DIST: " + dist);
        return e_call(fn, {D(C("ARRAY")), D(C("PARAM1")), D(C("PARAM2"))});
    }
    if (T == "tensor_new") {
        long dim = fint(node, "DIM");
        std::vector<std::string> parts;
        for (long i = 0; i < dim; ++i)
            parts.push_back(D(C(("DIM_" + std::to_string(i)).c_str())).code);
        return B.tensor_new(parts);
    }
    if (T == "tensor_get") return Expr(F("NAME"), P_PRIMARY);
    if (T == "tensor_save") {
        std::string name = F("NAME");
        if (first_decls.count(&node))
            return S(B.decl("tensor", name, D(C("EXPR")).code));
        return S(name + " = " + D(C("EXPR")).code);
    }
    if (T == "tensor_binop") {
        std::string op = F("OP");
        const Expr &l = D(C("LHS"));
        const Expr &r = D(C("RHS"));
        if (op == "matmul") return e_binop(l, "@", r, P_MUL);
        if (op == "add") return e_binop(l, "+", r, P_ADD);
        if (op == "sub") return e_binop(l, "-", r, P_ADD);
        if (op == "elemul") return e_binop(l, "*", r, P_MUL);
        throw std::runtime_error("Unsupported tensor binary operator: " + op);
    }
    if (T == "tensor_unop") {
        std::string op = F("OP");
        if (op == "neg") return e_unary("-", D(C("TENSOR")));
        throw std::runtime_error("Unsupported tensor unary operator: " + op);
    }
    if (T == "tensor_grad") return e_call("tensor_grad", {D(C("TENSOR"))});
    if (T == "tensor_curl") return e_call("tensor_curl", {D(C("TENSOR"))});
    if (T == "tensor_lapl") return e_call("tensor_lapl", {D(C("TENSOR"))});
    if (T == "tensor_scale")
        return e_binop(D(C("TENSOR")), "*", D(C("SCALAR")), P_MUL);

    /* ---- matrices ---- */
    if (T == "matrix_create") return B.matrix_create(D(C("ROWS")), D(C("COLS")));
    if (T == "matrix_identity") return e_call(B.builtin("matrix_identity"), {D(C("N"))});
    if (T == "matrix_matmul") return e_call(B.builtin("matrix_matmul"), {D(C("LHS")), D(C("RHS"))});
    if (T == "matrix_transpose") return e_call(B.builtin("matrix_transpose"), {D(C("M"))});
    if (T == "matrix_inverse") return e_call(B.builtin("matrix_inverse"), {D(C("M"))});
    if (T == "matrix_det") return e_call(B.builtin("matrix_det"), {D(C("M"))});
    if (T == "matrix_trace") return e_call(B.builtin("matrix_trace"), {D(C("M"))});

    /* ---- selects ---- */
    if (T == "i32_select" || T == "f64_select")
        return B.ternary(D(C("COND")), D(C("THEN")), D(C("ELSE")));

    if (T == "tensor_show_mat")
        return S("show_mat(" + D(C("TENSOR_ID")).code + ")");

    /* ---- array literals ---- */
    if (T == "array_literal_i32" || T == "array_literal_f64") {
        long size = fint(node, "SIZE");
        std::vector<std::string> elements;
        for (long i = 0; i < size; ++i) {
            const J *blk = C(("VAL_" + std::to_string(i)).c_str());
            if (blk) elements.push_back(D(blk).code);
        }
        std::string elem = (T == "array_literal_i32") ? "i32" : "f64";
        return Expr(B.array_literal(elements, elem), P_PRIMARY);
    }

    /* ---- tensor index ops ---- */
    if (T == "tensor_set_by_index" || T == "tensor_get_by_index") {
        std::string tname = F("TENSOR_NAME");
        std::vector<std::string> idx_parts;
        if (node.contains("inputs") && node["inputs"].is_object()) {
            for (auto it = node["inputs"].begin(); it != node["inputs"].end(); ++it) {
                if (starts_with(it.key(), "INDEX_")) {
                    const J &v = it.value();
                    if (v.is_object() && v.contains("block")) idx_parts.push_back(D(&v["block"]).code);
                }
            }
        }
        std::string acc = B.tensor_index(tname, idx_parts);
        if (T == "tensor_set_by_index")
            return S(acc + "=" + D(C("VALUE")).code);
        return Expr(acc, P_POSTFIX);
    }

    /* ---- vec2 ---- */
    if (T == "vec2_literal") return e_call("vec2", {D(C("X")), D(C("Y"))});
    if (T == "vec2_get") return Expr(F("NAME"), P_PRIMARY);
    if (T == "vec2_decl") return S(B.decl("vec2", F("NAME"), D(C("VEC")).code));
    if (T == "vec2_set") return S(F("NAME") + " = " + D(C("VEC")).code);
    if (T == "vec2_component_set")
        return S(F("NAME") + "." + F("AXIS") + " = " + D(C("VAL")).code);
    if (T == "vec2_x") return e_member(D(C("VEC")), "x");
    if (T == "vec2_y") return e_member(D(C("VEC")), "y");
    if (T == "vec2_len") return e_call("vec2_len", {D(C("VEC"))});
    if (T == "vec2_len_sq") return e_call("vec2_len_sq", {D(C("VEC"))});
    if (T == "vec2_add") return e_binop(D(C("A")), "+", D(C("B")), P_ADD);
    if (T == "vec2_sub") return e_binop(D(C("A")), "-", D(C("B")), P_ADD);
    if (T == "vec2_scale") return e_binop(D(C("VEC")), "*", D(C("S")), P_MUL);
    if (T == "vec2_neg") return e_unary("-", D(C("VEC")));
    if (T == "vec2_normalize") return e_call("vec2_normalize", {D(C("VEC"))});
    if (T == "vec2_dot") return e_call("vec2_dot", {D(C("A")), D(C("B"))});
    if (T == "vec2_proj_scalar") return e_call("vec2_proj_scalar", {D(C("A")), D(C("B"))});
    if (T == "vec2_proj_vec") return e_call("vec2_proj_vec", {D(C("A")), D(C("B"))});

    /* ---- vec3 ---- */
    if (T == "vec3_literal") return e_call("vec3", {D(C("X")), D(C("Y")), D(C("Z"))});
    if (T == "vec3_get") return Expr(F("NAME"), P_PRIMARY);
    if (T == "vec3_decl") return S(B.decl("vec3", F("NAME"), D(C("VEC")).code));
    if (T == "vec3_component_set")
        return S(F("NAME") + "." + F("AXIS") + " = " + D(C("VAL")).code);
    if (T == "vec3_set") return S(F("NAME") + " = " + D(C("VEC")).code);
    if (T == "vec3_x") return e_member(D(C("VEC")), "x");
    if (T == "vec3_y") return e_member(D(C("VEC")), "y");
    if (T == "vec3_z") return e_member(D(C("VEC")), "z");
    if (T == "vec3_len") return e_call("vec3_len", {D(C("VEC"))});
    if (T == "vec3_len_sq") return e_call("vec3_len_sq", {D(C("VEC"))});
    if (T == "vec3_add") return e_binop(D(C("A")), "+", D(C("B")), P_ADD);
    if (T == "vec3_sub") return e_binop(D(C("A")), "-", D(C("B")), P_ADD);
    if (T == "vec3_scale") return e_binop(D(C("VEC")), "*", D(C("S")), P_MUL);
    if (T == "vec3_neg") return e_unary("-", D(C("VEC")));
    if (T == "vec3_normalize") return e_call("vec3_normalize", {D(C("VEC"))});
    if (T == "vec3_dot") return e_call("vec3_dot", {D(C("A")), D(C("B"))});
    if (T == "vec3_cross") return e_call("vec3_cross", {D(C("A")), D(C("B"))});
    if (T == "vec3_proj_scalar") return e_call("vec3_proj_scalar", {D(C("A")), D(C("B"))});
    if (T == "vec3_proj_vec") return e_call("vec3_proj_vec", {D(C("A")), D(C("B"))});

    /* ---- boundary2d ---- */
    if (T == "local_decl_bd2") return e_stmt(term(B.decl_default("bd2", F("NAME"))));
    if (T == "local_get_bd2") return Expr(F("NAME") + "." + F("FIELD"), P_POSTFIX);
    if (T == "local_get_bd2_point") {
        std::string n = F("NAME");
        return Expr("vec2(" + n + ".x, " + n + ".y)", P_POSTFIX);
    }
    if (T == "local_get_bd2_tangent") {
        std::string n = F("NAME");
        return Expr("vec2(" + n + ".tx, " + n + ".ty)", P_POSTFIX);
    }
    if (T == "local_get_bd2_normal") {
        std::string n = F("NAME");
        return Expr("vec2(" + n + ".nx, " + n + ".ny)", P_POSTFIX);
    }

    /* ---- boundary3d ---- */
    if (T == "local_decl_bd3") return e_stmt(term(B.decl_default("bd3", F("NAME"))));
    if (T == "local_get_bd3") return Expr(F("NAME") + "." + F("FIELD"), P_POSTFIX);
    if (T == "local_get_bd3_point") {
        std::string n = F("NAME");
        return Expr("vec3(" + n + ".x, " + n + ".y, " + n + ".z)", P_POSTFIX);
    }
    if (T == "local_get_bd3_normal") {
        std::string n = F("NAME");
        return Expr("vec3(" + n + ".nx, " + n + ".ny, " + n + ".nz)", P_POSTFIX);
    }

    /* ---- latex ---- */
    if (T == "latex_expr") {
        std::string esc = replace_all(replace_all(F("LATEX"), "\\", "\\\\"), "\"", "\\\"");
        return S("eval_latex(\"" + esc + "\")");
    }
    if (T == "latex_value") {
        std::string latex = F("LATEX");
        std::cerr << latex << "\n";  // mirrors the Python print(latex) debug output
        std::string esc = replace_all(replace_all(latex, "\\", "\\\\"), "\"", "\\\"");
        return Expr("eval_latex(\"" + esc + "\")", P_POSTFIX);
    }

    throw std::runtime_error("Unsupported block type: " + T);
}

/* ------------------------------------------------------------------ *
 *  Iterative post-order compile of a block tree.
 * ------------------------------------------------------------------ */
static Expr dfs(const J &root, const std::string &main_fn_name,
                const FuncTable &funcs, const StructTable &structs, const Backend &B) {
    auto first_decls = scan_tensor_decls(root);
    Results results;
    std::deque<std::pair<const J *, bool>> stack;
    stack.push_back({&root, false});

    while (!stack.empty()) {
        auto [block, visited] = stack.back();
        stack.pop_back();
        if (visited) {
            results[block] = compile_block(*block, results, main_fn_name, first_decls, funcs, structs, B);
            continue;
        }
        if (results.count(block)) continue;
        stack.push_back({block, true});
        for (const J *c : children(*block))
            if (!results.count(c)) stack.push_back({c, false});
    }
    return results[&root];
}

// Build the id → {name, ret, paramCount} table from the canvas definition blocks
// so call blocks can resolve to a name + arity.
static FuncTable collect_funcs(const J &blocks) {
    FuncTable funcs;
    for (const J &block : blocks["blocks"]) {
        if (node_type(block) != "custom_func_def") continue;
        if (!block.contains("id")) continue;
        std::string id = funcIdOf(block["id"].get<std::string>());
        funcs[id] = { fstr(block, "NAME"), fstr(block, "RET"), extra_param_count(block) };
    }
    return funcs;
}

// Build the id → {name, fields} table from the canvas struct_def blocks so
// instance blocks can resolve to a type name + field layout.
static StructTable collect_structs(const J &blocks) {
    StructTable structs;
    for (const J &block : blocks["blocks"]) {
        if (node_type(block) != "struct_def") continue;
        if (!block.contains("id")) continue;
        std::string id = structIdOf(block["id"].get<std::string>());
        StructInfo si;
        si.name = fstr(block, "NAME");
        const J *f = child(block, "FIELDS");
        while (f) {
            if (node_type(*f) == "struct_field")
                si.fields.push_back({ fstr(*f, "FNAME"), fstr(*f, "FTYPE") });
            f = (f->contains("next") && (*f)["next"].is_object() && (*f)["next"].contains("block"))
                ? &(*f)["next"]["block"] : nullptr;
        }
        structs[id] = si;
    }
    return structs;
}

/* ------------------------------------------------------------------ *
 *  Top-level: workspace JSON -> target source string.
 * ------------------------------------------------------------------ */
static std::string cppize(const J &json, const std::string &main_fn_name, const Backend &B) {
    if (!json.contains("blocks")) throw std::runtime_error("missing 'blocks'");
    const J &blocks = json["blocks"];
    if (!blocks.contains("languageVersion") || blocks["languageVersion"].get<int>() != 0)
        throw std::runtime_error("languageVersion must be 0");
    if (!blocks.contains("blocks") || !blocks["blocks"].is_array())
        throw std::runtime_error("blocks.blocks must be an array");

    const FuncTable funcs = collect_funcs(blocks);
    const StructTable structs = collect_structs(blocks);

    std::string out = B.prelude();
    // Emit struct definitions first so functions can reference the types.
    for (const J &block : blocks["blocks"]) {
        if (node_type(block) == "struct_def")
            out += dfs(block, main_fn_name, funcs, structs, B).code + "\n\n";
    }
    // Then custom function definitions so C++ sees them before main uses
    // them (no forward declarations needed for the common case).
    for (const J &block : blocks["blocks"]) {
        if (node_type(block) == "custom_func_def")
            out += dfs(block, main_fn_name, funcs, structs, B).code + "\n\n";
    }
    // Then main (and legacy wasm_func_def roots). Orphan top-level blocks that
    // live outside a function are ignored.
    for (const J &block : blocks["blocks"]) {
        const std::string t = node_type(block);
        if (t != "wasm_func_main" && !starts_with(t, "wasm_func_def_"))
            continue;
        out += dfs(block, main_fn_name, funcs, structs, B).code + "\n\n";
    }
    return strip(out);
}

/* ------------------------------------------------------------------ *
 *  Target selection.
 * ------------------------------------------------------------------ */
static std::unique_ptr<Backend> make_backend(const std::string &target) {
    if (target == "cpp" || target == "c++") return std::make_unique<CppBackend>();
    if (target == "py" || target == "python") return std::make_unique<PyBackend>();
    if (target == "js" || target == "javascript") return std::make_unique<JsBackend>();
    return nullptr;
}

int main(int argc, char **argv) {
    std::ios::sync_with_stdio(false);

    // Parse the target language from argv. Accepts:
    //   --target=cpp | --target cpp | cpp   (also c++, py, python, js, javascript)
    std::string target = "cpp";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (starts_with(a, "--target=")) {
            target = a.substr(std::string("--target=").size());
        } else if (a == "--target" && i + 1 < argc) {
            target = argv[++i];
        } else if (!starts_with(a, "-")) {
            target = a;
        }
    }

    std::unique_ptr<Backend> backend = make_backend(target);
    if (!backend) {
        std::cerr << "Unknown target: " << target << " (expected cpp, py, or js)";
        return 1;
    }

    std::string main_fn_name;
    std::getline(std::cin, main_fn_name);
    while (!main_fn_name.empty() && (main_fn_name.back() == '\r' || main_fn_name.back() == '\n'))
        main_fn_name.pop_back();
    if (main_fn_name.empty()) main_fn_name = "main";

    std::string body((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());

    try {
        J json = J::parse(body);
        std::cout << cppize(json, main_fn_name, *backend);
        std::cout.flush();
    } catch (const std::exception &e) {
        std::cerr << e.what();
        return 1;
    }
    return 0;
}

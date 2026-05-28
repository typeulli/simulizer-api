/*
 * cpp2block.cpp — C++ (simstd subset) → Blockly JSON
 *
 * Usage:  cpp2block
 *         First line of stdin: N (number of code lines)
 *         Next N lines: C++ source code
 *         Outputs Blockly JSON to stdout.
 *
 * Build:  see build.bat  (requires LLVM/libclang)
 */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_NONSTDC_NO_DEPRECATE

#include <clang-c/Index.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#if defined(_WIN32)
#  include <windows.h>
#else
#  include <unistd.h>
#endif

using std::string;

/* ═══════════════════════════════════════════════════════════════════
   String formatting helper
   ═══════════════════════════════════════════════════════════════════ */
static string fmt(const char *f, ...) {
    va_list ap;
    va_start(ap, f);
    int n = vsnprintf(nullptr, 0, f, ap);
    va_end(ap);
    string r(static_cast<size_t>(n), '\0');
    va_start(ap, f);
    vsnprintf(&r[0], static_cast<size_t>(n) + 1, f, ap);
    va_end(ap);
    return r;
}

/* ═══════════════════════════════════════════════════════════════════
   ID generator
   ═══════════════════════════════════════════════════════════════════ */
static unsigned long long g_id_ctr = 1;
static string new_id() {
    char buf[17];
    snprintf(buf, 17, "%016llx", g_id_ctr++);
    return buf;
}

/* ═══════════════════════════════════════════════════════════════════
   Symbol table
   ═══════════════════════════════════════════════════════════════════ */
enum VarType {
    TY_I32, TY_F64, TY_BOOL,
    TY_I32_VEC, TY_F64_VEC,
    TY_VEC2, TY_VEC3,
    TY_TENSOR, TY_BD2, TY_BD3,
    TY_UNKNOWN
};

struct Var { char name[256]; VarType type; };
static const int MAX_VARS = 1024;
static Var g_vars[MAX_VARS];
static int g_nv = 0;

static void sym_set(const char *n, VarType t) {
    for (int i = 0; i < g_nv; i++)
        if (!strcmp(g_vars[i].name, n)) { g_vars[i].type = t; return; }
    if (g_nv < MAX_VARS) {
        strncpy(g_vars[g_nv].name, n, 255);
        g_vars[g_nv++].type = t;
    }
}
static VarType sym_get(const char *n) {
    for (int i = 0; i < g_nv; i++)
        if (!strcmp(g_vars[i].name, n)) return g_vars[i].type;
    return TY_UNKNOWN;
}

/* ═══════════════════════════════════════════════════════════════════
   #pragma region tracking
   ═══════════════════════════════════════════════════════════════════ */
struct Region {
    string name;
    unsigned start_line;
    unsigned end_line;
};
static const int MAX_REGIONS = 256;
static Region g_regions[MAX_REGIONS];
static int g_nreg = 0;
static bool g_region_claimed[MAX_REGIONS] = {false};
static bool g_had_error = false;

static bool scan_regions(const string &src) {
    struct Pending { string name; unsigned line; };
    Pending stack[MAX_REGIONS];
    int sp = 0;
    bool ok = true;

    unsigned line_no = 1;
    size_t i = 0;
    while (i <= src.size()) {
        size_t eol = src.find('\n', i);
        size_t line_end = (eol == string::npos) ? src.size() : eol;

        size_t p = i;
        while (p < line_end && (src[p] == ' ' || src[p] == '\t')) p++;

        if (p < line_end && src[p] == '#') {
            p++;
            while (p < line_end && (src[p] == ' ' || src[p] == '\t')) p++;
            if (p + 6 <= line_end && src.compare(p, 6, "pragma") == 0 &&
                (p + 6 == line_end || src[p+6] == ' ' || src[p+6] == '\t')) {
                p += 6;
                while (p < line_end && (src[p] == ' ' || src[p] == '\t')) p++;

                bool is_end =
                    p + 9 <= line_end &&
                    src.compare(p, 9, "endregion") == 0 &&
                    (p + 9 == line_end || src[p+9] == ' ' ||
                     src[p+9] == '\t' || src[p+9] == '\r');
                bool is_begin =
                    !is_end &&
                    p + 6 <= line_end &&
                    src.compare(p, 6, "region") == 0 &&
                    (p + 6 == line_end || src[p+6] == ' ' ||
                     src[p+6] == '\t' || src[p+6] == '\r');

                if (is_end) {
                    if (sp == 0) {
                        fprintf(stderr,
                            "Error: stray #pragma endregion at line %u\n",
                            line_no);
                        ok = false;
                    } else {
                        sp--;
                        if (g_nreg < MAX_REGIONS) {
                            g_regions[g_nreg].name = stack[sp].name;
                            g_regions[g_nreg].start_line = stack[sp].line;
                            g_regions[g_nreg].end_line = line_no;
                            g_nreg++;
                        }
                    }
                } else if (is_begin) {
                    p += 6;
                    while (p < line_end && (src[p] == ' ' || src[p] == '\t')) p++;
                    size_t name_start = p;
                    while (p < line_end && src[p] != ' ' &&
                           src[p] != '\t' && src[p] != '\r') p++;
                    if (name_start == p) {
                        fprintf(stderr,
                            "Error: #pragma region without name at line %u\n",
                            line_no);
                        ok = false;
                    } else if (sp < MAX_REGIONS) {
                        stack[sp].name =
                            src.substr(name_start, p - name_start);
                        stack[sp].line = line_no;
                        sp++;
                    }
                }
            }
        }

        line_no++;
        if (eol == string::npos) break;
        i = eol + 1;
    }

    while (sp > 0) {
        sp--;
        fprintf(stderr,
            "Error: unclosed #pragma region '%s' starting at line %u\n",
            stack[sp].name.c_str(), stack[sp].line);
        ok = false;
    }

    return ok;
}

/* ═══════════════════════════════════════════════════════════════════
   libclang helpers
   ═══════════════════════════════════════════════════════════════════ */

struct Kids { CXCursor c[128]; int n; };

static enum CXChildVisitResult _collect(CXCursor c, CXCursor, CXClientData d) {
    Kids *k = static_cast<Kids*>(d);
    if (k->n < 128) k->c[k->n++] = c;
    return CXChildVisit_Continue;
}
static Kids get_kids(CXCursor c) {
    Kids k; k.n = 0;
    clang_visitChildren(c, _collect, &k);
    return k;
}

static string cx_str(CXString s) {
    string r = clang_getCString(s);
    clang_disposeString(s);
    return r;
}

static unsigned cursor_start_line(CXCursor c) {
    CXFile f; unsigned line, col, off;
    clang_getExpansionLocation(
        clang_getRangeStart(clang_getCursorExtent(c)),
        &f, &line, &col, &off);
    return line;
}
static unsigned cursor_end_line(CXCursor c) {
    CXFile f; unsigned line, col, off;
    clang_getExpansionLocation(
        clang_getRangeEnd(clang_getCursorExtent(c)),
        &f, &line, &col, &off);
    return line;
}

static string first_token(CXCursor c, CXTranslationUnit tu) {
    CXToken *toks; unsigned n;
    clang_tokenize(tu, clang_getCursorExtent(c), &toks, &n);
    string r = (n > 0) ? cx_str(clang_getTokenSpelling(tu, toks[0])) : "0";
    clang_disposeTokens(tu, toks, n);
    return r;
}

static string binop_tok(CXCursor c, CXTranslationUnit tu,
                        CXCursor left, CXCursor right) {
    unsigned lend = 0, rstart = 0, line, col;
    CXFile f;
    clang_getExpansionLocation(clang_getRangeEnd(clang_getCursorExtent(left)),
                               &f, &line, &col, &lend);
    clang_getExpansionLocation(clang_getRangeStart(clang_getCursorExtent(right)),
                               &f, &line, &col, &rstart);
    CXToken *toks; unsigned n;
    clang_tokenize(tu, clang_getCursorExtent(c), &toks, &n);
    string op;
    for (unsigned i = 0; i < n && op.empty(); i++) {
        unsigned off;
        clang_getExpansionLocation(clang_getTokenLocation(tu, toks[i]),
                                   &f, &line, &col, &off);
        if (off >= lend && off < rstart &&
            clang_getTokenKind(toks[i]) == CXToken_Punctuation)
            op = cx_str(clang_getTokenSpelling(tu, toks[i]));
    }
    clang_disposeTokens(tu, toks, n);
    return op.empty() ? "?" : op;
}

static string unop_tok(CXCursor c, CXTranslationUnit tu, CXCursor child) {
    unsigned cstart = 0, line, col;
    CXFile f;
    clang_getExpansionLocation(clang_getRangeStart(clang_getCursorExtent(child)),
                               &f, &line, &col, &cstart);
    CXToken *toks; unsigned n;
    clang_tokenize(tu, clang_getCursorExtent(c), &toks, &n);
    string op;
    for (unsigned i = 0; i < n && op.empty(); i++) {
        unsigned off;
        clang_getExpansionLocation(clang_getTokenLocation(tu, toks[i]),
                                   &f, &line, &col, &off);
        if (off < cstart &&
            clang_getTokenKind(toks[i]) == CXToken_Punctuation)
            op = cx_str(clang_getTokenSpelling(tu, toks[i]));
    }
    clang_disposeTokens(tu, toks, n);
    return op.empty() ? "?" : op;
}

static VarType cx_vartype(CXType t) {
    CXType canon = clang_getCanonicalType(t);
    if (canon.kind != CXType_Invalid && canon.kind != CXType_Unexposed)
        t = canon;
    switch (t.kind) {
    case CXType_Int: case CXType_Long: case CXType_LongLong:
    case CXType_UInt: case CXType_ULong: case CXType_ULongLong:
    case CXType_Short: case CXType_SChar: case CXType_Char_S:
        return TY_I32;
    case CXType_Double: case CXType_Float: case CXType_LongDouble:
        return TY_F64;
    case CXType_Bool:
        return TY_BOOL;
    default: {
        string sp = cx_str(clang_getTypeSpelling(t));
        if (sp.find("vector") != string::npos &&
            (sp.find("int") != string::npos || sp.find("i32") != string::npos))
            return TY_I32_VEC;
        if (sp.find("vector") != string::npos &&
            (sp.find("double") != string::npos || sp.find("f64") != string::npos))
            return TY_F64_VEC;
        if (sp.find("vec2")      != string::npos) return TY_VEC2;
        if (sp.find("vec3")      != string::npos) return TY_VEC3;
        if (sp.find("Tensor")    != string::npos) return TY_TENSOR;
        if (sp.find("Boundary2D")!= string::npos) return TY_BD2;
        if (sp.find("Boundary3D")!= string::npos) return TY_BD3;
        return TY_UNKNOWN;
    }
    }
}

static CXCursor unwrap(CXCursor c) {
    for (;;) {
        if (c.kind == CXCursor_ParenExpr || c.kind == CXCursor_UnexposedExpr) {
            Kids k = get_kids(c);
            if (k.n == 1) { c = k.c[0]; continue; }
        }
        break;
    }
    return c;
}

/* ═══════════════════════════════════════════════════════════════════
   JSON block builder helpers
   ═══════════════════════════════════════════════════════════════════ */

static string wrap_block(const string &block_json) {
    return "{\"block\":" + block_json + "}";
}

static string make_block(const char *type, const char *flds, const char *inps) {
    string r = fmt("{\"type\":\"%s\",\"id\":\"%s\"", type, new_id().c_str());
    if (flds && flds[0]) r += fmt(",\"fields\":{%s}", flds);
    if (inps && inps[0]) r += fmt(",\"inputs\":{%s}", inps);
    r += "}";
    return r;
}

/* ═══════════════════════════════════════════════════════════════════
   Forward declarations
   ═══════════════════════════════════════════════════════════════════ */
static string expr_json(CXCursor c, CXTranslationUnit tu);
static string stmt_json(CXCursor c, CXTranslationUnit tu);
static string body_json(CXCursor compound, CXTranslationUnit tu);

static CXCursor find_init_list(CXCursor c, int depth = 0) {
    if (depth > 12) return clang_getNullCursor();
    if (c.kind == CXCursor_InitListExpr) return c;
    Kids k = get_kids(c);
    for (int i = 0; i < k.n; i++) {
        CXCursor r = find_init_list(k.c[i], depth + 1);
        if (!clang_Cursor_isNull(r)) return r;
    }
    return clang_getNullCursor();
}

/* Type-aware placeholder for expressions we couldn't translate.
   Returning i32_const 0 unconditionally would create a type mismatch
   inside f64_binop / bool slots, because coerce_f64 only inspects the
   cursor type and would see "no conversion needed" while the actual
   block is i32_const. */
static string default_const_for(CXCursor c) {
    VarType t = cx_vartype(clang_getCursorType(unwrap(c)));
    if (t == TY_F64)  return make_block("f64_const",  "\"VALUE\":0.0",     nullptr);
    if (t == TY_BOOL) return make_block("bool_const", "\"VALUE\":\"false\"", nullptr);
    return make_block("i32_const", "\"VALUE\":0", nullptr);
}

/* Wrap json in f64_from_i32 if the cursor's type is i32/bool */
static string coerce_f64(const string& json, CXCursor c) {
    VarType t = cx_vartype(clang_getCursorType(unwrap(c)));
    if (t == TY_I32 || t == TY_BOOL) {
        // Fold f64_from_i32(i32_const(N)) → f64_const(N)
        if (json.find("\"type\":\"i32_const\"") != string::npos) {
            size_t vp = json.find("\"VALUE\":");
            if (vp != string::npos) {
                vp += 8;
                size_t end = json.find_first_of(",}", vp);
                string val = json.substr(vp, end - vp);
                return make_block("f64_const", fmt("\"VALUE\":%s", val.c_str()).c_str(), nullptr);
            }
        }
        string inps = fmt("\"VALUE\":%s", wrap_block(json).c_str());
        return make_block("f64_from_i32", nullptr, inps.c_str());
    }
    return json;
}

/* ═══════════════════════════════════════════════════════════════════
   Expression → JSON
   ═══════════════════════════════════════════════════════════════════ */
static string expr_json(CXCursor c, CXTranslationUnit tu) {
    c = unwrap(c);

    switch (c.kind) {

    /* ── integer literal ── */
    case CXCursor_IntegerLiteral: {
        string num = first_token(c, tu);
        while (!num.empty() && (num.back()=='u'||num.back()=='U'||
                                num.back()=='l'||num.back()=='L'))
            num.pop_back();
        return make_block("i32_const", fmt("\"VALUE\":%s", num.c_str()).c_str(), nullptr);
    }

    /* ── floating literal ── */
    case CXCursor_FloatingLiteral: {
        string num = first_token(c, tu);
        while (!num.empty() && (num.back()=='f'||num.back()=='F'))
            num.pop_back();
        return make_block("f64_const", fmt("\"VALUE\":%s", num.c_str()).c_str(), nullptr);
    }

    /* ── bool literal ── */
    case CXCursor_CXXBoolLiteralExpr: {
        string txt = first_token(c, tu);
        bool is_true = (txt == "true");
        return make_block("bool_const",
                          fmt("\"VALUE\":\"%s\"", is_true ? "true" : "false").c_str(),
                          nullptr);
    }

    /* ── variable reference ── */
    case CXCursor_DeclRefExpr: {
        string name = cx_str(clang_getCursorSpelling(c));
        VarType t = sym_get(name.c_str());
        string fld = fmt("\"NAME\":\"%s\"", name.c_str());
        if (t == TY_I32 || t == TY_BOOL)   return make_block("local_get_i32",      fld.c_str(), nullptr);
        if (t == TY_F64)                   return make_block("local_get_f64",      fld.c_str(), nullptr);
        if (t == TY_I32_VEC)               return make_block("local_array_get_i32",fld.c_str(), nullptr);
        if (t == TY_F64_VEC)               return make_block("local_array_get_f64",fld.c_str(), nullptr);
        if (t == TY_VEC2)                  return make_block("vec2_get",           fld.c_str(), nullptr);
        if (t == TY_VEC3)                  return make_block("vec3_get",           fld.c_str(), nullptr);
        if (t == TY_TENSOR)                return make_block("tensor_get",         fld.c_str(), nullptr);
        return make_block("local_get_i32", fld.c_str(), nullptr);
    }

    /* ── array subscript: arr[idx] ── */
    case CXCursor_ArraySubscriptExpr: {
        Kids k = get_kids(c);
        if (k.n < 2) return "{}";
        string arr = expr_json(k.c[0], tu);
        string idx = expr_json(k.c[1], tu);
        VarType et = TY_UNKNOWN;
        CXCursor arr_c = unwrap(k.c[0]);
        if (arr_c.kind == CXCursor_DeclRefExpr) {
            string n = cx_str(clang_getCursorSpelling(arr_c));
            VarType at = sym_get(n.c_str());
            et = (at == TY_I32_VEC) ? TY_I32 : TY_F64;
        }
        string inps = fmt("\"ARRAY\":%s,\"INDEX\":%s",
                          wrap_block(arr).c_str(), wrap_block(idx).c_str());
        return make_block(et == TY_F64 ? "array_get_f64" : "array_get_i32",
                          nullptr, inps.c_str());
    }

    /* ── C-style cast ── */
    case CXCursor_CStyleCastExpr: {
        Kids k = get_kids(c);
        if (k.n < 1) return "{}";
        CXCursor child = k.c[k.n-1];
        string val = expr_json(child, tu);
        VarType child_type = cx_vartype(clang_getCursorType(unwrap(child)));
        string cast_sp = cx_str(clang_getTypeSpelling(clang_getCursorType(c)));
        bool to_f64 = (cast_sp.find("double") != string::npos ||
                       cast_sp.find("f64")    != string::npos);
        
        // f64로 캐스트하는데 이미 f64면 그냥 반환
        if (to_f64 && child_type == TY_F64) {
            return val;
        }
        // f64로 캐스트하는데 i32/bool이면 f64_from_i32
        if (to_f64) {
            string inps = fmt("\"VALUE\":%s", wrap_block(val).c_str());
            return make_block("f64_from_i32", nullptr, inps.c_str());
        }
        // i32로 캐스트하는데 f64면 i32_from_f64
        if (child_type == TY_F64) {
            string inps = fmt("\"VALUE\":%s", wrap_block(val).c_str());
            return make_block("i32_from_f64", nullptr, inps.c_str());
        }
        // 같은 타입이면 그냥 반환
        return val;
    }

    /* ── unary operator ── */
    case CXCursor_UnaryOperator: {
        Kids k = get_kids(c);
        if (k.n < 1) return "{}";
        CXCursor child = k.c[0];
        string op  = unop_tok(c, tu, child);
        string val = expr_json(child, tu);
        string wv  = wrap_block(val);
        VarType t  = cx_vartype(clang_getCursorType(child));
        string inps = fmt("\"VALUE\":%s", wv.c_str());
        if (op == "!")
            return make_block("bool_not", nullptr, inps.c_str());
        if (op == "-")
            return make_block(t == TY_F64 ? "f64_unop" : "i32_unop",
                              "\"OP\":\"neg\"", inps.c_str());
        return make_block("i32_unop", "\"OP\":\"neg\"", inps.c_str());
    }

    /* ── binary operator ── */
    case CXCursor_BinaryOperator: {
        Kids k = get_kids(c);
        if (k.n < 2) return "{}";
        CXCursor lc = k.c[0], rc = k.c[1];
        string op = binop_tok(c, tu, lc, rc);
        string lj = expr_json(lc, tu);
        string rj = expr_json(rc, tu);
        VarType lt = cx_vartype(clang_getCursorType(unwrap(lc)));
        VarType rt = cx_vartype(clang_getCursorType(unwrap(rc)));
        bool is_f64 = (lt == TY_F64 || rt == TY_F64);
        bool is_bool_op = (op=="&&" || op=="||" || op=="^" ||
                           (lt==TY_BOOL && rt==TY_BOOL));
        string inps = fmt("\"LHS\":%s,\"RHS\":%s",
                          wrap_block(lj).c_str(), wrap_block(rj).c_str());

        if (op=="==" || op=="!=" || op=="<" || op=="<=" || op==">" || op==">=") {
            const char *wop =
                op=="==" ? "eq" : op=="!=" ? "ne" :
                op=="<"  ? (is_f64?"lt":"lt_s") :
                op=="<=" ? (is_f64?"le":"le_s") :
                op==">"  ? (is_f64?"gt":"gt_s") : (is_f64?"ge":"ge_s");
            if (is_f64) {
                lj = coerce_f64(lj, lc);
                rj = coerce_f64(rj, rc);
                inps = fmt("\"LHS\":%s,\"RHS\":%s",
                           wrap_block(lj).c_str(), wrap_block(rj).c_str());
            }
            return make_block(is_f64 ? "f64_cmp" : "i32_cmp",
                              fmt("\"OP\":\"%s\"", wop).c_str(), inps.c_str());
        }
        if (op=="&&" || op=="||" || (is_bool_op && op=="^")) {
            const char *wop = op=="&&"?"and": op=="||"?"or":"xor";
            return make_block("bool_binop",
                              fmt("\"OP\":\"%s\"", wop).c_str(), inps.c_str());
        }
        if (is_f64) {
            lj = coerce_f64(lj, lc);
            rj = coerce_f64(rj, rc);
            inps = fmt("\"LHS\":%s,\"RHS\":%s",
                       wrap_block(lj).c_str(), wrap_block(rj).c_str());
            const char *wop =
                op=="+" ? "add" : op=="-" ? "sub" :
                op=="*" ? "mul" : op=="/" ? "div" : "add";
            return make_block("f64_binop",
                              fmt("\"OP\":\"%s\"", wop).c_str(), inps.c_str());
        } else {
            const char *wop =
                op=="+"  ? "add"   : op=="-"  ? "sub"   :
                op=="*"  ? "mul"   : op=="/"  ? "div_s" :
                op=="%"  ? "rem_s" : op=="&"  ? "and"   :
                op=="|"  ? "or"    : op=="^"  ? "xor"   :
                op=="<<" ? "shl"   : op==">>" ? "shr_s" : "add";
            return make_block("i32_binop",
                              fmt("\"OP\":\"%s\"", wop).c_str(), inps.c_str());
        }
    }

    /* ── ternary / conditional ── */
    case CXCursor_ConditionalOperator: {
        Kids k = get_kids(c);
        if (k.n < 3) return "{}";
        string wc = wrap_block(expr_json(k.c[0], tu));
        string wt = wrap_block(expr_json(k.c[1], tu));
        string we = wrap_block(expr_json(k.c[2], tu));
        VarType t = cx_vartype(clang_getCursorType(c));
        string inps = fmt("\"COND\":%s,\"THEN\":%s,\"ELSE\":%s",
                          wc.c_str(), wt.c_str(), we.c_str());
        return make_block(t == TY_F64 ? "f64_select" : "i32_select",
                          nullptr, inps.c_str());
    }

    /* ── call expression (returns value) ── */
    case CXCursor_CallExpr: {
        string fname = cx_str(clang_getCursorSpelling(c));
        Kids k = get_kids(c);
        int a = 1; /* first arg index */

        auto wrap1 = [&](int i) { return wrap_block(expr_json(k.c[i], tu)); };
        auto wrap2 = [&](int i, int j) {
            return fmt("\"A\":%s,\"B\":%s",
                       wrap_block(expr_json(k.c[i],tu)).c_str(),
                       wrap_block(expr_json(k.c[j],tu)).c_str());
        };

        if (fname == "operator()") {
            // tensor element read: T(i, j, ...)
            // libclang layout: k.c[0]=object(T), k.c[1]=method ref, k.c[2..]=index args
            a = 2;
            CXCursor tensor_obj = k.c[0];
            // fallback: if k.c[0] is a MemberRefExpr, the object is its child
            {
                CXCursor mr = unwrap(k.c[0]);
                if (mr.kind == CXCursor_MemberRefExpr) {
                    Kids ch = get_kids(mr);
                    if (ch.n > 0) { tensor_obj = ch.c[0]; }
                }
            }
            string tname;
            if (unwrap(tensor_obj).kind == CXCursor_DeclRefExpr)
                tname = cx_str(clang_getCursorSpelling(unwrap(tensor_obj)));
            int dim = k.n - a;
            string flds = fmt("\"TENSOR_NAME\":\"%s\",\"DIM\":%d",
                              tname.c_str(), dim);
            string inps;
            for (int i = a; i < k.n; i++)
                inps += fmt("%s\"INDEX_%d\":%s",
                            inps.empty() ? "" : ",",
                            i - a,
                            wrap_block(expr_json(k.c[i], tu)).c_str());
            return make_block("tensor_get_by_index", flds.c_str(),
                              inps.empty() ? nullptr : inps.c_str());
        }
        if (fname == "debug_bar") {
            string inps = fmt("\"MIN\":%s,\"MAX\":%s",
                              wrap1(a).c_str(), wrap1(a+1).c_str());
            return make_block("debug_bar", nullptr, inps.c_str());
        }
        if (fname == "debug_series")
            return make_block("debug_series", nullptr, nullptr);
        if (fname == "__builtin_clz") {
            string inps = fmt("\"VALUE\":%s", wrap1(a).c_str());
            return make_block("i32_unop", "\"OP\":\"clz\"", inps.c_str());
        }
        if (fname == "__builtin_ctz") {
            string inps = fmt("\"VALUE\":%s", wrap1(a).c_str());
            return make_block("i32_unop", "\"OP\":\"ctz\"", inps.c_str());
        }
        if (fname == "__builtin_popcount") {
            string inps = fmt("\"VALUE\":%s", wrap1(a).c_str());
            return make_block("i32_unop", "\"OP\":\"popcnt\"", inps.c_str());
        }
        if (fname=="abs"||fname=="fabs"||fname=="sqrt"||fname=="ceil"||
            fname=="floor"||fname=="trunc"||fname=="round") {
            const char *wop =
                (fname=="abs"||fname=="fabs") ? "abs" :
                fname=="sqrt"  ? "sqrt"  : fname=="ceil"  ? "ceil"  :
                fname=="floor" ? "floor" : fname=="trunc" ? "trunc" : "nearest";
            string inps = fmt("\"VALUE\":%s", wrap1(a).c_str());
            return make_block("f64_unop",
                              fmt("\"OP\":\"%s\"", wop).c_str(), inps.c_str());
        }
        if (fname=="fmin"||fname=="fmax") {
            const char *wop = fname=="fmin" ? "min" : "max";
            string inps = fmt("\"LHS\":%s,\"RHS\":%s",
                              wrap1(a).c_str(), wrap1(a+1).c_str());
            return make_block("f64_binop",
                              fmt("\"OP\":\"%s\"", wop).c_str(), inps.c_str());
        }
        if (fname == "vec2") {
            string x_json = coerce_f64(expr_json(k.c[a], tu), k.c[a]);
            string y_json = coerce_f64(expr_json(k.c[a+1], tu), k.c[a+1]);
            string inps = fmt("\"X\":%s,\"Y\":%s", wrap_block(x_json).c_str(), wrap_block(y_json).c_str());
            return make_block("vec2_literal", nullptr, inps.c_str());
        }
        if (fname == "vec3") {
            string x_json = coerce_f64(expr_json(k.c[a], tu), k.c[a]);
            string y_json = coerce_f64(expr_json(k.c[a+1], tu), k.c[a+1]);
            string z_json = coerce_f64(expr_json(k.c[a+2], tu), k.c[a+2]);
            string inps = fmt("\"X\":%s,\"Y\":%s,\"Z\":%s",
                              wrap_block(x_json).c_str(), wrap_block(y_json).c_str(), wrap_block(z_json).c_str());
            return make_block("vec3_literal", nullptr, inps.c_str());
        }
        if (fname=="vec2_len"||fname=="vec2_len_sq"||fname=="vec2_normalize"||
            fname=="vec3_len"||fname=="vec3_len_sq"||fname=="vec3_normalize") {
            string inps = fmt("\"VEC\":%s", wrap1(a).c_str());
            return make_block(fname.c_str(), nullptr, inps.c_str());
        }
        if (fname=="vec2_dot"||fname=="vec2_proj_scalar"||fname=="vec2_proj_vec"||
            fname=="vec3_dot"||fname=="vec3_cross"||
            fname=="vec3_proj_scalar"||fname=="vec3_proj_vec") {
            string inps = wrap2(a, a+1);
            return make_block(fname.c_str(), nullptr, inps.c_str());
        }
        if (fname == "matrix_create") {
            string inps = fmt("\"ROWS\":%s,\"COLS\":%s",
                              wrap1(a).c_str(), wrap1(a+1).c_str());
            return make_block("matrix_create", nullptr, inps.c_str());
        }
        if (fname == "matrix_identity") {
            string inps = fmt("\"N\":%s", wrap1(a).c_str());
            return make_block("matrix_identity", nullptr, inps.c_str());
        }
        if (fname == "matrix_matmul") {
            string inps = fmt("\"LHS\":%s,\"RHS\":%s",
                              wrap1(a).c_str(), wrap1(a+1).c_str());
            return make_block("matrix_matmul", nullptr, inps.c_str());
        }
        if (fname=="matrix_transpose"||fname=="matrix_inverse"||
            fname=="matrix_det"||fname=="matrix_trace") {
            string inps = fmt("\"M\":%s", wrap1(a).c_str());
            return make_block(fname.c_str(), nullptr, inps.c_str());
        }
        if (fname == "size" && k.n >= 1) {
            Kids mk = get_kids(k.c[0]);
            CXCursor arr_c = (mk.n > 0) ? mk.c[0] : k.c[0];
            VarType at = cx_vartype(clang_getCursorType(unwrap(arr_c)));
            string inps = fmt("\"ARRAY\":%s",
                              wrap_block(expr_json(arr_c, tu)).c_str());
            return make_block(at==TY_F64_VEC?"array_len_f64":"array_len_i32",
                              nullptr, inps.c_str());
        }
        return default_const_for(c);
    }

    /* ── member access: vec.x, vec.y, etc. ── */
    case CXCursor_MemberRefExpr: {
        Kids k = get_kids(c);
        string member = cx_str(clang_getCursorSpelling(c));
        if (k.n >= 1) {
            CXCursor base = k.c[0];
            VarType bt = cx_vartype(clang_getCursorType(unwrap(base)));
            string inps = fmt("\"VEC\":%s",
                              wrap_block(expr_json(base, tu)).c_str());
            if ((bt==TY_VEC2||bt==TY_VEC3) && member=="x")
                return make_block(bt==TY_VEC2?"vec2_x":"vec3_x", nullptr, inps.c_str());
            if ((bt==TY_VEC2||bt==TY_VEC3) && member=="y")
                return make_block(bt==TY_VEC2?"vec2_y":"vec3_y", nullptr, inps.c_str());
            if (bt==TY_VEC3 && member=="z")
                return make_block("vec3_z", nullptr, inps.c_str());
        }
        return default_const_for(c);
    }

    default:
        return default_const_for(c);
    }
}

/* ═══════════════════════════════════════════════════════════════════
   Statement chaining
   ═══════════════════════════════════════════════════════════════════ */
static string chain_next(string *stmts, int n) {
    if (n == 0) return "{}";
    string cur = stmts[n-1];
    for (int i = n-2; i >= 0; i--) {
        /* inject ,"next":{...} before the last '}' */
        string &prev = stmts[i];
        string result = prev.substr(0, prev.size()-1);
        result += ",\"next\":{\"block\":" + cur + "}}";
        cur = result;
    }
    return cur;
}

static string body_json_range(CXCursor *children, int n,
                              unsigned lo, unsigned hi,
                              CXTranslationUnit tu) {
    int direct[MAX_REGIONS]; int ndirect = 0;
    for (int r = 0; r < g_nreg; r++) {
        Region &R = g_regions[r];
        if (R.start_line < lo || R.end_line > hi) continue;
        if (R.start_line == lo && R.end_line == hi) continue;
        bool in_child = false;
        for (int j = 0; j < n; j++) {
            unsigned js = cursor_start_line(children[j]);
            unsigned je = cursor_end_line(children[j]);
            if (R.start_line >= js && R.end_line <= je) { in_child = true; break; }
        }
        if (in_child) continue;
        direct[ndirect++] = r;
    }

    int top[MAX_REGIONS]; int ntop = 0;
    for (int i = 0; i < ndirect; i++) {
        Region &Ri = g_regions[direct[i]];
        bool contained = false;
        for (int j = 0; j < ndirect; j++) {
            if (i == j) continue;
            Region &Rj = g_regions[direct[j]];
            if (Ri.start_line >= Rj.start_line &&
                Ri.end_line   <= Rj.end_line &&
                !(Ri.start_line == Rj.start_line &&
                  Ri.end_line   == Rj.end_line)) {
                contained = true; break;
            }
        }
        if (!contained) top[ntop++] = direct[i];
    }

    for (int i = 0; i < ntop; i++) {
        Region &R = g_regions[top[i]];
        for (int j = 0; j < n; j++) {
            unsigned js = cursor_start_line(children[j]);
            unsigned je = cursor_end_line(children[j]);
            bool fully_in  = (js >= R.start_line && je <= R.end_line);
            bool fully_out = (je < R.start_line || js > R.end_line);
            if (!fully_in && !fully_out) {
                fprintf(stderr,
                    "Error: #pragma region '%s' (lines %u-%u) crosses block boundary at line %u\n",
                    R.name.c_str(), R.start_line, R.end_line, js);
                g_had_error = true;
            }
        }
    }

    bool child_handled[128] = {false};
    for (int i = 0; i < ntop; i++) {
        Region &R = g_regions[top[i]];
        g_region_claimed[top[i]] = true;
        for (int j = 0; j < n; j++) {
            unsigned js = cursor_start_line(children[j]);
            unsigned je = cursor_end_line(children[j]);
            if (js >= R.start_line && je <= R.end_line)
                child_handled[j] = true;
        }
    }

    struct Item { bool is_reg; int idx; unsigned start; };
    Item items[512]; int nitems = 0;
    for (int i = 0; i < ntop; i++) {
        Region &R = g_regions[top[i]];
        items[nitems].is_reg = true;
        items[nitems].idx = top[i];
        items[nitems].start = R.start_line;
        nitems++;
    }
    for (int j = 0; j < n; j++) {
        if (child_handled[j]) continue;
        items[nitems].is_reg = false;
        items[nitems].idx = j;
        items[nitems].start = cursor_start_line(children[j]);
        nitems++;
    }
    for (int i = 1; i < nitems; i++) {
        Item key = items[i];
        int k = i - 1;
        while (k >= 0 && items[k].start > key.start) {
            items[k+1] = items[k]; k--;
        }
        items[k+1] = key;
    }

    string *stmts = new string[nitems + 1];
    int ns = 0;
    for (int i = 0; i < nitems; i++) {
        if (items[i].is_reg) {
            Region &R = g_regions[items[i].idx];
            CXCursor inner[128]; int nin = 0;
            for (int j = 0; j < n; j++) {
                unsigned js = cursor_start_line(children[j]);
                unsigned je = cursor_end_line(children[j]);
                if (js >= R.start_line && je <= R.end_line) {
                    if (nin < 128) inner[nin++] = children[j];
                }
            }
            string sub = body_json_range(inner, nin,
                                         R.start_line, R.end_line, tu);
            string flds = fmt("\"NAME\":\"%s\"", R.name.c_str());
            string inps;
            if (sub != "{}" && !sub.empty())
                inps = fmt("\"BODY\":%s", wrap_block(sub).c_str());
            string fold = make_block("flow_fold_region", flds.c_str(),
                                     inps.empty() ? nullptr : inps.c_str());
            stmts[ns++] = fold;
        } else {
            string s = stmt_json(children[items[i].idx], tu);
            if (!s.empty()) stmts[ns++] = s;
        }
    }
    string r = (ns == 0) ? "{}" : chain_next(stmts, ns);
    delete[] stmts;
    return r;
}

static string body_json(CXCursor compound, CXTranslationUnit tu) {
    Kids k = get_kids(compound);
    if (k.n == 0) return "{}";
    unsigned lo = cursor_start_line(compound);
    unsigned hi = cursor_end_line(compound);
    return body_json_range(k.c, k.n, lo, hi, tu);
}

/* ═══════════════════════════════════════════════════════════════════
   Statement → JSON
   ═══════════════════════════════════════════════════════════════════ */
static string stmt_json(CXCursor c, CXTranslationUnit tu) {
    switch (c.kind) {

    /* ── variable declaration ── */
    case CXCursor_VarDecl: {
        string name = cx_str(clang_getCursorSpelling(c));
        VarType vt = cx_vartype(clang_getCursorType(c));
        sym_set(name.c_str(), vt);
        string fld = fmt("\"NAME\":\"%s\"", name.c_str());

        Kids k = get_kids(c);

        auto find_init = [&](bool skip_bool) -> CXCursor {
            for (int i = k.n-1; i >= 0; i--) {
                auto knd = k.c[i].kind;
                if (knd==CXCursor_TypeRef||knd==CXCursor_NamespaceRef||
                    knd==CXCursor_TemplateRef) continue;
                if (skip_bool && knd==CXCursor_CXXBoolLiteralExpr) continue;
                return k.c[i];
            }
            return clang_getNullCursor();
        };

        if (vt == TY_I32 || vt == TY_F64) {
            CXCursor init_c = find_init(false);
            string init_j;
            if (!clang_Cursor_isNull(init_c)) {
                init_j = expr_json(init_c, tu);
                if (vt == TY_F64) init_j = coerce_f64(init_j, init_c);
            } else {
                init_j = make_block(vt==TY_F64?"f64_const":"i32_const",
                                    vt==TY_F64?"\"VALUE\":0.0":"\"VALUE\":0", nullptr);
            }
            string inps = fmt("\"INIT\":%s", wrap_block(init_j).c_str());
            return make_block(vt==TY_F64?"local_decl_f64":"local_decl_i32",
                              fld.c_str(), inps.c_str());
        }
        if (vt == TY_I32_VEC || vt == TY_F64_VEC) {
            CXCursor size_c = clang_getNullCursor();
            for (int i = 0; i < k.n; i++) {
                auto knd = k.c[i].kind;
                if (knd==CXCursor_TypeRef||knd==CXCursor_NamespaceRef||
                    knd==CXCursor_TemplateRef||knd==CXCursor_CXXBoolLiteralExpr)
                    continue;
                size_c = k.c[i]; break;
            }
            string inps;
            if (!clang_Cursor_isNull(size_c))
                inps = fmt("\"SIZE\":%s", wrap_block(expr_json(size_c, tu)).c_str());
            return make_block(vt==TY_F64_VEC?"local_array_decl_f64":"local_array_decl_i32",
                              fld.c_str(), inps.empty()?nullptr:inps.c_str());
        }
        if (vt == TY_VEC2 || vt == TY_VEC3) {
            CXCursor init_c = find_init(false);
            if (!clang_Cursor_isNull(init_c)) {
                string inps = fmt("\"VEC\":%s",
                                  wrap_block(expr_json(init_c, tu)).c_str());
                return make_block(vt==TY_VEC2?"vec2_decl":"vec3_decl",
                                  fld.c_str(), inps.c_str());
            }
            return make_block(vt==TY_VEC2?"vec2_decl":"vec3_decl",
                              fld.c_str(), nullptr);
        }
        if (vt == TY_TENSOR) {
            CXCursor init_c = find_init(false);
            if (!clang_Cursor_isNull(init_c)) {
                string expr_j;
                CXCursor ilist = find_init_list(init_c);
                if (!clang_Cursor_isNull(ilist)) {
                    Kids dims = get_kids(ilist);
                    int ndim = dims.n;
                    string flds_new = fmt("\"DIM\":%d", ndim);
                    string inps_new;
                    for (int i = 0; i < ndim; i++) {
                        string dj = expr_json(dims.c[i], tu);
                        inps_new += fmt("%s\"DIM_%d\":%s",
                                        inps_new.empty() ? "" : ",",
                                        i, wrap_block(dj).c_str());
                    }
                    expr_j = make_block("tensor_new", flds_new.c_str(),
                                        inps_new.empty() ? nullptr : inps_new.c_str());
                } else {
                    expr_j = expr_json(init_c, tu);
                }
                string inps = fmt("\"EXPR\":%s", wrap_block(expr_j).c_str());
                return make_block("tensor_save", fld.c_str(), inps.c_str());
            }
            return make_block("tensor_save", fld.c_str(), nullptr);
        }
        if (vt == TY_BD2) return make_block("local_decl_bd2", fld.c_str(), nullptr);
        if (vt == TY_BD3) return make_block("local_decl_bd3", fld.c_str(), nullptr);
        return make_block("i32_const", "\"VALUE\":0", nullptr);
    }

    /* ── DeclStmt wraps VarDecl ── */
    case CXCursor_DeclStmt: {
        Kids k = get_kids(c);
        if (k.n > 0) return stmt_json(k.c[0], tu);
        return "";
    }

    /* ── if statement ── */
    case CXCursor_IfStmt: {
        Kids k = get_kids(c);
        if (k.n < 2) return make_block("flow_if", nullptr, nullptr);

        /* Detect 'else' keyword in IfStmt tokens at brace-depth 0. */
        unsigned else_off = 0;
        bool has_else_kw = false;
        {
            CXToken *toks; unsigned ntoks;
            clang_tokenize(tu, clang_getCursorExtent(c), &toks, &ntoks);
            int depth = 0;
            CXFile f; unsigned line, col;
            for (unsigned i = 0; i < ntoks; i++) {
                CXTokenKind tk = clang_getTokenKind(toks[i]);
                string s = cx_str(clang_getTokenSpelling(tu, toks[i]));
                if (tk == CXToken_Punctuation) {
                    if (s == "{") depth++;
                    else if (s == "}") depth--;
                } else if (tk == CXToken_Keyword && s == "else" && depth == 0) {
                    clang_getExpansionLocation(clang_getTokenLocation(tu, toks[i]),
                                               &f, &line, &col, &else_off);
                    has_else_kw = true;
                    break;
                }
            }
            clang_disposeTokens(tu, toks, ntoks);
        }

        /* Identify cond / then / else children by source location relative to 'else'. */
        CXCursor cond_c = k.c[0];
        CXCursor then_c = clang_getNullCursor();
        CXCursor else_c = clang_getNullCursor();
        for (int i = 1; i < k.n; i++) {
            unsigned off = 0, line, col; CXFile f;
            clang_getExpansionLocation(clang_getRangeStart(clang_getCursorExtent(k.c[i])),
                                       &f, &line, &col, &off);
            if (has_else_kw && off > else_off) {
                else_c = k.c[i];
            } else {
                then_c = k.c[i];
            }
        }
        if (clang_Cursor_isNull(then_c)) then_c = k.c[1];

        string wc = wrap_block(expr_json(cond_c, tu));
        string bj = (then_c.kind==CXCursor_CompoundStmt)
                    ? body_json(then_c, tu) : stmt_json(then_c, tu);
        string wb = wrap_block(bj);

        if (has_else_kw && !clang_Cursor_isNull(else_c)) {
            string ej = (else_c.kind==CXCursor_CompoundStmt)
                        ? body_json(else_c, tu) : stmt_json(else_c, tu);
            string inps = fmt("\"COND\":%s,\"THEN\":%s,\"ELSE\":%s",
                              wc.c_str(), wb.c_str(), wrap_block(ej).c_str());
            return make_block("flow_if_else", nullptr, inps.c_str());
        }
        string inps = fmt("\"COND\":%s,\"THEN\":%s", wc.c_str(), wb.c_str());
        return make_block("flow_if", nullptr, inps.c_str());
    }

    /* ── for statement ── */
    case CXCursor_ForStmt: {
        Kids k = get_kids(c);
        char var[256] = "i";
        string start_j, end_j, body_j;

        CXCursor init_decl = k.n >= 1 ? k.c[0] : clang_getNullCursor();
        if (!clang_Cursor_isNull(init_decl) &&
            init_decl.kind == CXCursor_DeclStmt) {
            Kids dk = get_kids(init_decl);
            if (dk.n > 0) init_decl = dk.c[0];
        }
        if (!clang_Cursor_isNull(init_decl) &&
            init_decl.kind == CXCursor_VarDecl) {
            string n = cx_str(clang_getCursorSpelling(init_decl));
            strncpy(var, n.c_str(), 255);
            sym_set(var, TY_I32);
            Kids vk = get_kids(init_decl);
            for (int i = 0; i < vk.n; i++) {
                if (vk.c[i].kind != CXCursor_TypeRef) {
                    start_j = expr_json(vk.c[i], tu); break;
                }
            }
        }
        if (k.n >= 2) {
            Kids ck = get_kids(k.c[1]);
            if (ck.n >= 2) end_j = expr_json(ck.c[1], tu);
        }
        if (k.n >= 1) {
            CXCursor bd = k.c[k.n-1];
            body_j = (bd.kind==CXCursor_CompoundStmt)
                     ? body_json(bd, tu) : stmt_json(bd, tu);
        }

        if (start_j.empty()) start_j = make_block("i32_const","\"VALUE\":0",nullptr);
        if (end_j.empty())   end_j   = make_block("i32_const","\"VALUE\":0",nullptr);
        if (body_j.empty())  body_j  = "{}";

        string fld  = fmt("\"VAR\":\"%s\"", var);
        string inps = fmt("\"START\":%s,\"END\":%s,\"BODY\":%s",
                          wrap_block(start_j).c_str(),
                          wrap_block(end_j).c_str(),
                          wrap_block(body_j).c_str());
        return make_block("flow_for", fld.c_str(), inps.c_str());
    }

    /* ── while statement ── */
    case CXCursor_WhileStmt: {
        Kids k = get_kids(c);
        if (k.n < 2) return make_block("flow_while", nullptr, nullptr);
        string wc = wrap_block(expr_json(k.c[0], tu));
        string bj = (k.c[1].kind==CXCursor_CompoundStmt)
                    ? body_json(k.c[1], tu) : stmt_json(k.c[1], tu);
        string inps = fmt("\"COND\":%s,\"BODY\":%s",
                          wc.c_str(), wrap_block(bj).c_str());
        return make_block("flow_while", nullptr, inps.c_str());
    }

    /* ── break ── */
    case CXCursor_BreakStmt:
        return make_block("flow_break", nullptr, nullptr);

    /* ── return ── */
    case CXCursor_ReturnStmt: {
        Kids k = get_kids(c);
        if (k.n == 0) return make_block("wasm_return_i32", nullptr, nullptr);
        string vj = expr_json(k.c[0], tu);
        VarType t = cx_vartype(clang_getCursorType(unwrap(k.c[0])));
        string inps = fmt("\"VALUE\":%s", wrap_block(vj).c_str());
        return make_block(t==TY_F64?"wasm_return_f64":"wasm_return_i32",
                          nullptr, inps.c_str());
    }

    /* ── call as statement ── */
    case CXCursor_CallExpr: {
        string fname = cx_str(clang_getCursorSpelling(c));
        Kids k = get_kids(c);
        int a = 1;

        if (fname=="log_i32"||fname=="log_f64"||fname=="debug_log") {
            string inps = fmt("\"VALUE\":%s",
                              wrap_block(expr_json(k.c[a], tu)).c_str());
            return make_block("debug_log", nullptr, inps.c_str());
        }
        if (fname == "debug_bar_set") {
            string inps = fmt("\"ID\":%s,\"VALUE\":%s",
                              wrap_block(expr_json(k.c[a],   tu)).c_str(),
                              wrap_block(expr_json(k.c[a+1], tu)).c_str());
            return make_block("debug_bar_set", nullptr, inps.c_str());
        }
        if (fname == "debug_set_holder") {
            string inps = fmt("\"ID\":%s",
                              wrap_block(expr_json(k.c[a], tu)).c_str());
            return make_block("debug_set_holder", nullptr, inps.c_str());
        }
        if (fname == "show_mat") {
            string inps = fmt("\"TENSOR_ID\":%s",
                              wrap_block(expr_json(k.c[a], tu)).c_str());
            return make_block("tensor_show_mat", nullptr, inps.c_str());
        }
        return "";  // Unhandled function call - skip statement
    }

    /* ── assignment / binary operator as statement ── */
    case CXCursor_BinaryOperator: {
        Kids k = get_kids(c);
        if (k.n < 2) return "";
        CXCursor lhs = unwrap(k.c[0]);
        CXCursor rhs = k.c[1];
        string op = binop_tok(c, tu, k.c[0], rhs);
        if (op != "=") return "";

        if (lhs.kind == CXCursor_CallExpr &&
            cx_str(clang_getCursorSpelling(lhs)) == "operator()") {
            // tensor element write: T(i, j, ...) = val
            Kids ck = get_kids(lhs);
            int ca = 2;
            CXCursor tensor_obj = ck.c[0];
            // fallback: if ck.c[0] is a MemberRefExpr, the object is its child
            {
                CXCursor mr = unwrap(ck.c[0]);
                if (mr.kind == CXCursor_MemberRefExpr) {
                    Kids ch = get_kids(mr);
                    if (ch.n > 0) tensor_obj = ch.c[0];
                }
            }
            string tensor_j = expr_json(tensor_obj, tu);
            string tname;
            if (unwrap(tensor_obj).kind == CXCursor_DeclRefExpr)
                tname = cx_str(clang_getCursorSpelling(unwrap(tensor_obj)));
            int dim = ck.n - ca;
            string flds = fmt("\"TENSOR_NAME\":\"%s\",\"DIM\":%d",
                              tname.c_str(), dim);
            // coerce rhs: tensor element type from return type of operator()
            CXType lhs_ret = clang_getCanonicalType(clang_getCursorType(lhs));
            bool is_f64t = lhs_ret.kind == CXType_Double ||
                           lhs_ret.kind == CXType_Float;
            string rhs_j = expr_json(rhs, tu);
            if (is_f64t) rhs_j = coerce_f64(rhs_j, rhs);
            string inps;
            for (int i = ca; i < ck.n; i++)
                inps += fmt("%s\"INDEX_%d\":%s",
                            inps.empty() ? "" : ",",
                            i - ca,
                            wrap_block(expr_json(ck.c[i], tu)).c_str());
            inps += fmt("%s\"VALUE\":%s",
                        inps.empty() ? "" : ",",
                        wrap_block(rhs_j).c_str());
            return make_block("tensor_set_by_index", flds.c_str(), inps.c_str());
        }
        if (lhs.kind == CXCursor_ArraySubscriptExpr) {
            Kids ak = get_kids(lhs);
            if (ak.n < 2) return "";
            CXCursor arr_c = unwrap(ak.c[0]);
            VarType at = TY_I32_VEC;
            if (arr_c.kind == CXCursor_DeclRefExpr) {
                string n = cx_str(clang_getCursorSpelling(arr_c));
                at = sym_get(n.c_str());
            }
            string inps = fmt("\"ARRAY\":%s,\"INDEX\":%s,\"VALUE\":%s",
                              wrap_block(expr_json(ak.c[0], tu)).c_str(),
                              wrap_block(expr_json(ak.c[1], tu)).c_str(),
                              wrap_block(expr_json(rhs,     tu)).c_str());
            return make_block(at==TY_F64_VEC?"array_set_f64":"array_set_i32",
                              nullptr, inps.c_str());
        }
        if (lhs.kind == CXCursor_MemberRefExpr) {
            // Member access assignment: v.x = val
            Kids mk = get_kids(lhs);
            if (mk.n > 0) {
                CXCursor obj = mk.c[0];
                string member = cx_str(clang_getCursorSpelling(lhs));
                string obj_name = cx_str(clang_getCursorSpelling(obj));
                
                VarType vt = sym_get(obj_name.c_str());
                if ((vt == TY_VEC2 || vt == TY_VEC3) && (member == "x" || member == "y" || member == "z")) {
                    string rhs_j = expr_json(rhs, tu);
                    rhs_j = coerce_f64(rhs_j, rhs);  // vec components are f64
                    string wv = wrap_block(rhs_j);
                    string fld = fmt("\"NAME\":\"%s\",\"AXIS\":\"%s\"", obj_name.c_str(), member.c_str());
                    string inps = fmt("\"VAL\":%s", wv.c_str());
                    const char *btype = (vt == TY_VEC2) ? "vec2_component_set" : "vec3_component_set";
                    return make_block(btype, fld.c_str(), inps.c_str());
                }
            }
        }
        if (lhs.kind == CXCursor_DeclRefExpr) {
            string name = cx_str(clang_getCursorSpelling(lhs));
            VarType vt = sym_get(name.c_str());
            string rhs_j = expr_json(rhs, tu);
            if (vt == TY_F64) rhs_j = coerce_f64(rhs_j, rhs);
            string wv = wrap_block(rhs_j);
            string fld = fmt("\"NAME\":\"%s\"", name.c_str());
            string inps;
            const char *btype = nullptr;
            if (vt==TY_F64)  { btype="local_set_f64"; inps=fmt("\"VALUE\":%s",wv.c_str()); }
            else if (vt==TY_VEC2) { btype="vec2_set"; inps=fmt("\"VEC\":%s",wv.c_str()); }
            else if (vt==TY_VEC3) { btype="vec3_set"; inps=fmt("\"VEC\":%s",wv.c_str()); }
            else { btype="local_set_i32"; inps=fmt("\"VALUE\":%s",wv.c_str()); }
            return make_block(btype, fld.c_str(), inps.c_str());
        }
        return "";
    }

    case CXCursor_CompoundStmt:
        return body_json(c, tu);

    default:
        return "";
    }
}

/* ═══════════════════════════════════════════════════════════════════
   Top-level function visitor
   ═══════════════════════════════════════════════════════════════════ */
struct TopCtx {
    CXTranslationUnit tu;
    string out;
    bool first = true;
};

static enum CXChildVisitResult visit_top(CXCursor c, CXCursor, CXClientData data) {
    TopCtx *ctx = static_cast<TopCtx*>(data);
    if (c.kind != CXCursor_FunctionDecl) return CXChildVisit_Continue;

    string name = cx_str(clang_getCursorSpelling(c));
    if (name != "main") return CXChildVisit_Continue;

    CXType ret = clang_getResultType(clang_getCursorType(c));
    VarType rvt = cx_vartype(ret);
    const char *ret_str =
        (ret.kind == CXType_Void) ? "void" :
        (rvt == TY_F64)           ? "f64"  :
        (rvt == TY_BOOL)          ? "bool" : "i32";

    Kids k = get_kids(c);
    CXCursor body = clang_getNullCursor();
    bool has_body = false;
    for (int i = 0; i < k.n; i++) {
        if (k.c[i].kind == CXCursor_CompoundStmt) {
            body = k.c[i]; has_body = true; break;
        }
    }

    string blk = fmt("{\"type\":\"wasm_func_main\",\"id\":\"%s\","
                     "\"x\":40,\"y\":40,\"fields\":{\"RET_TYPE\":\"%s\"}",
                     new_id().c_str(), ret_str);
    if (has_body)
        blk += fmt(",\"inputs\":{\"BODY\":%s}",
                   wrap_block(body_json(body, ctx->tu)).c_str());
    blk += "}";

    if (!ctx->first) ctx->out += ",";
    ctx->out += blk;
    ctx->first = false;

    return CXChildVisit_Continue;
}

/* ═══════════════════════════════════════════════════════════════════
   main
   ═══════════════════════════════════════════════════════════════════ */
int main(int /*argc*/, char ** /*argv*/) {
    int n = 0;
    if (scanf("%d", &n) != 1 || n <= 0) {
        fprintf(stderr, "Error: invalid line count\n");
        return 1;
    }
    /* consume the newline after N */
    {
        int ch;
        while ((ch = getchar()) != '\n' && ch != EOF) {}
    }

    string src;
    src.reserve(static_cast<size_t>(n) * 80);
    char line_buf[65536];
    for (int i = 0; i < n; i++) {
        if (!fgets(line_buf, sizeof(line_buf), stdin)) break;
        src += line_buf;
    }

    if (!scan_regions(src)) {
        return 1;
    }

    /* write source to a temp file so libclang can parse it */
    char tmp_path[512];
#if defined(_WIN32)
    {
        char tmp_dir[256];
        DWORD len = GetTempPathA(sizeof(tmp_dir), tmp_dir);
        if (len == 0) strcpy(tmp_dir, ".");
        snprintf(tmp_path, sizeof(tmp_path), "%scpp2block_%lu.cpp",
                 tmp_dir, (unsigned long)GetCurrentProcessId());
    }
#else
    snprintf(tmp_path, sizeof(tmp_path), "/tmp/cpp2block_%d.cpp", (int)getpid());
#endif

    FILE *tf = fopen(tmp_path, "w");
    if (!tf) {
        fprintf(stderr, "Error: cannot create temp file %s\n", tmp_path);
        return 1;
    }
    fwrite(src.c_str(), 1, src.size(), tf);
    fclose(tf);

    /* Locate simstd.hpp relative to this executable so user code that uses
       i32, f64, vec2, etc. can be parsed even without an explicit include. */
    char include_arg[1024] = {0};
    char include_dir_arg[768] = {0};
    bool have_include = false;
#if defined(_WIN32)
    {
        char exe_path[768];
        DWORD el = GetModuleFileNameA(nullptr, exe_path, sizeof(exe_path));
        if (el > 0 && el < sizeof(exe_path)) {
            char *slash = strrchr(exe_path, '\\');
            if (!slash) slash = strrchr(exe_path, '/');
            if (slash) *slash = '\0';
            char candidates[2][768];
            snprintf(candidates[0], sizeof(candidates[0]), "%s\\..\\simstd.hpp", exe_path);
            snprintf(candidates[1], sizeof(candidates[1]), "%s\\simstd.hpp", exe_path);
            for (int i = 0; i < 2; i++) {
                FILE *probe = fopen(candidates[i], "r");
                if (probe) {
                    fclose(probe);
                    snprintf(include_arg, sizeof(include_arg), "-include%s", candidates[i]);
                    char *s2 = strrchr(candidates[i], '\\');
                    if (s2) {
                        *s2 = '\0';
                        snprintf(include_dir_arg, sizeof(include_dir_arg), "-I%s", candidates[i]);
                    }
                    have_include = true;
                    break;
                }
            }
        }
    }
#endif

    const char *clang_args_no_inc[] = { "-std=c++17", "-I.", "-w" };
    const char *clang_args_inc[]    = { "-std=c++17", "-I.", "-w",
                                        include_dir_arg, include_arg };

    CXIndex index = clang_createIndex(0, 0);
    CXTranslationUnit tu = clang_parseTranslationUnit(
        index, tmp_path,
        have_include ? clang_args_inc : clang_args_no_inc,
        have_include ? 5 : 3,
        nullptr, 0, CXTranslationUnit_None);

    if (!tu) {
        fprintf(stderr, "Failed to parse input\n");
        remove(tmp_path);
        clang_disposeIndex(index);
        return 1;
    }

    unsigned ndiag = clang_getNumDiagnostics(tu);
    for (unsigned i = 0; i < ndiag; i++) {
        CXDiagnostic d = clang_getDiagnostic(tu, i);
        if (clang_getDiagnosticSeverity(d) >= CXDiagnostic_Error) {
            CXString msg = clang_getDiagnosticSpelling(d);
            const char *msg_str = clang_getCString(msg);
            if (strstr(msg_str, "'main' must return 'int'") == nullptr)
                fprintf(stderr, "Error: %s\n", msg_str);
            clang_disposeString(msg);
        }
        clang_disposeDiagnostic(d);
    }

    TopCtx ctx;
    ctx.tu = tu;

    clang_visitChildren(clang_getTranslationUnitCursor(tu), visit_top, &ctx);

    for (int i = 0; i < g_nreg; i++) {
        if (!g_region_claimed[i]) {
            fprintf(stderr,
                "Error: #pragma region '%s' (lines %u-%u) is not inside the main function body\n",
                g_regions[i].name.c_str(),
                g_regions[i].start_line, g_regions[i].end_line);
            g_had_error = true;
        }
    }
    if (g_had_error) {
        clang_disposeTranslationUnit(tu);
        clang_disposeIndex(index);
        remove(tmp_path);
        return 1;
    }

    printf("{\"blocks\":{\"languageVersion\":0,\"blocks\":[%s]}}\n",
           ctx.out.c_str());

    clang_disposeTranslationUnit(tu);
    clang_disposeIndex(index);
    remove(tmp_path);
    return 0;
}

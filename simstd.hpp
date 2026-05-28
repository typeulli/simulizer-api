#pragma once
// simstd.h — Simulizer standard declarations for C++ transpiled code
//
// Inline: vec2, vec3 math (from vector blocks)
// Extern: debug/logging and tensor ops (from debug & tensor blocks)

#ifdef __EMSCRIPTEN__
// =====================================================================
// EMSCRIPTEN BRANCH — Side-module build for Simulizer web runtime.
// All Tensor<f64> ops and debug/log functions delegate to host imports
// resolved by the main module's --js-library bindings.
// =====================================================================

#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <initializer_list>

using i32 = int32_t;
using f64 = double;

// ── vec2 ────────────────────────────────────────────────────────────────────
struct vec2 {
    f64 x, y;
    constexpr vec2(f64 x = 0.0, f64 y = 0.0) : x(x), y(y) {}
};
inline vec2 operator+(vec2 a, vec2 b)         { return {a.x + b.x, a.y + b.y}; }
inline vec2 operator-(vec2 a, vec2 b)         { return {a.x - b.x, a.y - b.y}; }
inline vec2 operator*(vec2 v, f64 s)          { return {v.x * s,   v.y * s};   }
inline vec2 operator*(f64 s, vec2 v)          { return {v.x * s,   v.y * s};   }
inline vec2 operator-(vec2 v)                 { return {-v.x, -v.y};            }
inline f64  vec2_dot(vec2 a, vec2 b)          { return a.x*b.x + a.y*b.y; }
inline f64  vec2_len_sq(vec2 v)               { return v.x*v.x + v.y*v.y; }
inline f64  vec2_len(vec2 v)                  { return std::sqrt(vec2_len_sq(v)); }
inline vec2 vec2_normalize(vec2 v)            { f64 l = vec2_len(v); return {v.x/l, v.y/l}; }
inline f64  vec2_cross_scalar(vec2 a, vec2 b) { return a.x*b.y - a.y*b.x; }
inline f64  vec2_proj_scalar(vec2 a, vec2 b)  { return vec2_dot(a, b) / vec2_len(b); }
inline vec2 vec2_proj_vec(vec2 a, vec2 b)     { f64 s = vec2_dot(a, b) / vec2_len_sq(b); return b * s; }

// ── vec3 ────────────────────────────────────────────────────────────────────
struct vec3 {
    f64 x, y, z;
    constexpr vec3(f64 x = 0.0, f64 y = 0.0, f64 z = 0.0) : x(x), y(y), z(z) {}
};
inline vec3 operator+(vec3 a, vec3 b)  { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline vec3 operator-(vec3 a, vec3 b)  { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline vec3 operator*(vec3 v, f64 s)   { return {v.x*s,   v.y*s,   v.z*s};   }
inline vec3 operator*(f64 s, vec3 v)   { return {v.x*s,   v.y*s,   v.z*s};   }
inline vec3 operator-(vec3 v)          { return {-v.x, -v.y, -v.z};          }
inline f64  vec3_dot(vec3 a, vec3 b)   { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline f64  vec3_len_sq(vec3 v)        { return v.x*v.x + v.y*v.y + v.z*v.z; }
inline f64  vec3_len(vec3 v)           { return std::sqrt(vec3_len_sq(v)); }
inline vec3 vec3_normalize(vec3 v)     { f64 l = vec3_len(v); return {v.x/l, v.y/l, v.z/l}; }
inline vec3 vec3_cross(vec3 a, vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline f64  vec3_proj_scalar(vec3 a, vec3 b) { return vec3_dot(a, b) / vec3_len(b); }
inline vec3 vec3_proj_vec(vec3 a, vec3 b)    { f64 s = vec3_dot(a, b) / vec3_len_sq(b); return b * s; }

// ── Boundary2D / Boundary3D ─────────────────────────────────────────────────
struct Boundary2D {
    f64 t, x, y;
    f64 tx, ty;
    f64 nx, ny;
};
struct Boundary3D {
    f64 t, x, y, z;
    f64 nx, ny, nz;
};

// ── Host imports (resolved by main module --js-library) ─────────────────────
extern "C" {
    void __sim_log_i32(int val);
    void __sim_log_f64(double val);
    void __sim_log_vec2(double x, double y);
    void __sim_log_vec3(double x, double y, double z);
    void __sim_log_arr_i32(const int* ptr, int cap);
    void __sim_log_arr_f64(const double* ptr, int cap);
    void __sim_log_tensor(int tensor_id);

    int  __sim_debug_bar(int min, int max);
    void __sim_debug_bar_set(int bar_id, int val);
    int  __sim_debug_series();
    void __sim_debug_set_holder(int id);
    void __sim_show_mat(int tensor_id);

    void __sim_graph_arr_i32(const int* ptr, int cap);
    void __sim_graph_arr_f64(const double* ptr, int cap);
    void __sim_graph_arr_range_i32(const int* ptr, int cap, double mn, double mx);
    void __sim_graph_arr_range_f64(const double* ptr, int cap, double mn, double mx);

    int    __sim_tensor_create(int varid, const int* shape, int dim);
    int    __sim_tensor_random(int varid, int dist_type, double p1, double p2, const int* shape, int dim);
    int    __sim_tensor_add(int lhs, int rhs);
    int    __sim_tensor_sub(int lhs, int rhs);
    int    __sim_tensor_matmul(int lhs, int rhs);
    int    __sim_tensor_neg(int v);
    int    __sim_tensor_elemul(int lhs, int rhs);
    int    __sim_tensor_scale(int v, double s);
    int    __sim_tensor_save(int out_varid, int tensor_id);
    int    __sim_tensor_set(int tensor_id, int n, int i0, int i1, int i2, int i3, int i4, int i5, double value);
    double __sim_tensor_get(int tensor_id, int n, int i0, int i1, int i2, int i3, int i4, int i5);
    int    __sim_tensor_perlin(int varid, int rows, int cols);
    int    __sim_tensor_clone(int dst_varid, int src_varid);
    int    __sim_tensor_dispose(int varid);

    int    __sim_matrix_create(int varid, int rows, int cols);
    int    __sim_matrix_matmul(int lhs, int rhs);
    int    __sim_matrix_transpose(int v);
    int    __sim_matrix_inverse(int v);
    double __sim_matrix_det(int v);
    double __sim_matrix_trace(int v);
    int    __sim_matrix_identity(int n);
}

namespace simulizer {
    inline void sync(size_t = 1) {}
    inline void init() {}
    inline bool openBrowser(const std::string&) { return false; }
}

// ── Tensor — varid handle bound to a JS-side TF.js Tensor ───────────────────
template<typename T>
class Tensor {
    int varid_;

    static int& _next_varid() {
        static int v = 1;
        return v;
    }

public:
    static int alloc_varid() { return _next_varid()++; }

    Tensor() : varid_(-1) {}
    explicit Tensor(int existing) : varid_(existing) {}

    Tensor(const std::vector<size_t>& shape) {
        varid_ = alloc_varid();
        std::vector<int> shape_i(shape.begin(), shape.end());
        __sim_tensor_create(varid_, shape_i.data(), (int)shape_i.size());
    }

    Tensor(std::initializer_list<i32> shape) {
        varid_ = alloc_varid();
        std::vector<int> shape_i(shape.begin(), shape.end());
        __sim_tensor_create(varid_, shape_i.data(), (int)shape_i.size());
    }

    // Deep-copy semantics to match the native branch: copying a Tensor
    // produces an independent JS-side tensor (via __sim_tensor_clone), so
    // mutating the copy does not affect the source. Move steals the varid
    // without cloning. Destructor disposes the owned JS-side tensor.
    ~Tensor() { if (varid_ >= 0) __sim_tensor_dispose(varid_); }

    Tensor(const Tensor& other) : varid_(-1) {
        if (other.varid_ >= 0) {
            varid_ = alloc_varid();
            __sim_tensor_clone(varid_, other.varid_);
        }
    }
    Tensor& operator=(const Tensor& other) {
        if (this == &other) return *this;
        if (other.varid_ < 0) {
            if (varid_ >= 0) __sim_tensor_dispose(varid_);
            varid_ = -1;
        } else {
            if (varid_ < 0) varid_ = alloc_varid();
            __sim_tensor_clone(varid_, other.varid_);
        }
        return *this;
    }
    Tensor(Tensor&& other) noexcept : varid_(other.varid_) { other.varid_ = -1; }
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (varid_ >= 0) __sim_tensor_dispose(varid_);
            varid_ = other.varid_;
            other.varid_ = -1;
        }
        return *this;
    }

    int  id()  const           { return varid_; }
    void set_id(int v)         { varid_ = v; }

    struct Proxy {
        int varid;
        int n;
        int idx[6];

        operator T() const {
            return (T)__sim_tensor_get(varid, n, idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]);
        }
        Proxy& operator=(T value) {
            __sim_tensor_set(varid, n, idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], (double)value);
            return *this;
        }
        Proxy& operator=(const Proxy& other) { return *this = (T)other; }
        Proxy& operator+=(T value) { return *this = (T)(*this) + value; }
        Proxy& operator-=(T value) { return *this = (T)(*this) - value; }
        Proxy& operator*=(T value) { return *this = (T)(*this) * value; }
        Proxy& operator/=(T value) { return *this = (T)(*this) / value; }
    };

    template<typename... Args>
    Proxy operator()(Args... args) {
        static_assert(sizeof...(Args) <= 6, "Tensor supports up to 6 dimensions");
        Proxy p{varid_, (int)sizeof...(Args), {0, 0, 0, 0, 0, 0}};
        int i = 0;
        ((p.idx[i++] = (int)args), ...);
        return p;
    }

    template<typename... Args>
    Proxy operator()(Args... args) const {
        static_assert(sizeof...(Args) <= 6, "Tensor supports up to 6 dimensions");
        Proxy p{varid_, (int)sizeof...(Args), {0, 0, 0, 0, 0, 0}};
        int i = 0;
        ((p.idx[i++] = (int)args), ...);
        return p;
    }
};

// ── tensor_random helpers ───────────────────────────────────────────────────
inline Tensor<f64> tensor_uniform(const std::vector<i32>& shape, f64 lo, f64 hi) {
    Tensor<f64> t;
    int v = Tensor<f64>::alloc_varid();
    t.set_id(v);
    __sim_tensor_random(v, 0, lo, hi, shape.data(), (int)shape.size());
    return t;
}
inline Tensor<f64> tensor_normal(const std::vector<i32>& shape, f64 mean, f64 stddev) {
    Tensor<f64> t;
    int v = Tensor<f64>::alloc_varid();
    t.set_id(v);
    __sim_tensor_random(v, 1, mean, stddev, shape.data(), (int)shape.size());
    return t;
}

// ── matrix blocks ───────────────────────────────────────────────────────────
inline Tensor<f64> matrix_create(i32 rows, i32 cols) {
    Tensor<f64> t;
    int v = Tensor<f64>::alloc_varid();
    t.set_id(v);
    __sim_matrix_create(v, rows, cols);
    return t;
}
inline Tensor<f64> matrix_identity(i32 n)                                { return Tensor<f64>(__sim_matrix_identity(n)); }
inline Tensor<f64> matrix_transpose(const Tensor<f64>& a)                { return Tensor<f64>(__sim_matrix_transpose(a.id())); }
inline Tensor<f64> matrix_matmul(const Tensor<f64>& a, const Tensor<f64>& b) { return Tensor<f64>(__sim_matrix_matmul(a.id(), b.id())); }
inline f64         matrix_trace(const Tensor<f64>& a)                    { return __sim_matrix_trace(a.id()); }
inline f64         matrix_det(const Tensor<f64>& a)                      { return __sim_matrix_det(a.id()); }
inline Tensor<f64> matrix_inverse(const Tensor<f64>& a)                  { return Tensor<f64>(__sim_matrix_inverse(a.id())); }

// ── debug_log overloads ─────────────────────────────────────────────────────
inline void debug_log(i32 val)                  { __sim_log_i32(val); }
inline void debug_log(f64 val)                  { __sim_log_f64(val); }
inline void debug_log(vec2 v)                   { __sim_log_vec2(v.x, v.y); }
inline void debug_log(vec3 v)                   { __sim_log_vec3(v.x, v.y, v.z); }
inline void debug_log(i32* ptr, i32 cap)        { __sim_log_arr_i32(ptr, cap); }
inline void debug_log(f64* ptr, i32 cap)        { __sim_log_arr_f64(ptr, cap); }
inline void debug_log(Tensor<f64>& t)           { __sim_log_tensor(t.id()); }
inline void debug_log(Tensor<i32>& t)           { __sim_log_tensor(t.id()); }

// ── show_mat / debug_bar / debug_series ─────────────────────────────────────
inline void show_mat(Tensor<f64>& t)            { __sim_show_mat(t.id()); }
inline i32  debug_bar(i32 mn, i32 mx)           { return __sim_debug_bar(mn, mx); }
inline void debug_bar_set(i32 id, i32 val)      { __sim_debug_bar_set(id, val); }
inline i32  debug_series()                      { return __sim_debug_series(); }
inline void debug_set_holder(i32 id)            { __sim_debug_set_holder(id); }

// ── graph_arr_* ─────────────────────────────────────────────────────────────
inline void graph_arr_i32(const std::vector<i32>& arr) {
    __sim_graph_arr_i32(arr.data(), (i32)arr.size());
}
inline void graph_arr_f64(const std::vector<f64>& arr) {
    __sim_graph_arr_f64(arr.data(), (i32)arr.size());
}
inline void graph_arr_range_i32(const std::vector<i32>& arr, f64 mn, f64 mx) {
    __sim_graph_arr_range_i32(arr.data(), (i32)arr.size(), mn, mx);
}
inline void graph_arr_range_f64(const std::vector<f64>& arr, f64 mn, f64 mx) {
    __sim_graph_arr_range_f64(arr.data(), (i32)arr.size(), mn, mx);
}

#else  // !__EMSCRIPTEN__

#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#undef WINVER
#define WINVER 0x0A00
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <random>
#include <atomic>
#include <chrono>
#include <string>
#include "lib/httplib.h"
#include "lib/banner.hpp"

#if defined(_WIN32)
#  include <windows.h>
#  include <psapi.h>
#  include <pdh.h>
#  include <pdhmsg.h>
#  if defined(_MSC_VER)
#    pragma comment(lib, "psapi.lib")
#    pragma comment(lib, "pdh.lib")
#  endif
#elif defined(__APPLE__)
#  include <mach/mach.h>
#  include <sys/resource.h>
#  include <sys/time.h>
#  include <unistd.h>
#else
#  include <sys/resource.h>
#  include <sys/time.h>
#  include <unistd.h>
#  include <cstdio>
#endif

#ifdef _WIN32

#include <windows.h>

bool openBrowser(const std::string& url) {
    HINSTANCE result = ShellExecuteA(
        nullptr,
        "open",
        url.c_str(),
        nullptr,
        nullptr,
        SW_SHOWNORMAL
    );

    return reinterpret_cast<intptr_t>(result) > 32;
}

#elif __APPLE__

#include <spawn.h>
#include <sys/wait.h>

extern char** environ;

bool openBrowser(const std::string& url) {
    pid_t pid;

    const char* argv[] = {
        "open",
        url.c_str(),
        nullptr
    };

    int status = posix_spawnp(
        &pid,
        "open",
        nullptr,
        nullptr,
        const_cast<char* const*>(argv),
        environ
    );

    return status == 0;
}

#else

#include <spawn.h>
#include <sys/wait.h>

extern char** environ;

bool openBrowser(const std::string& url) {
    pid_t pid;

    const char* argv[] = {
        "xdg-open",
        url.c_str(),
        nullptr
    };

    int status = posix_spawnp(
        &pid,
        "xdg-open",
        nullptr,
        nullptr,
        const_cast<char* const*>(argv),
        environ
    );

    return status == 0;
}

#endif

extern const unsigned char _binary_assets_index_html[];
extern const unsigned char _binary_assets_console_html[];
extern const unsigned char _binary_assets_console_js[];
extern const unsigned int _binary_assets_index_html_len;
extern const unsigned int _binary_assets_console_html_len;
extern const unsigned int _binary_assets_console_js_len;

namespace simulizer {


inline void sync(size_t weight = 1) {
    std::this_thread::yield();
}
bool initailized = false;
std::vector<std::string> message_queue;
std::shared_mutex queue_mutex;
std::condition_variable_any queue_cv;
httplib::Server svr;

inline void push_message(const std::string& msg) {
    {
        std::unique_lock<std::shared_mutex> lock(queue_mutex);
        message_queue.push_back(msg);
    }
    queue_cv.notify_all();
    simulizer::sync();
}

inline std::string _json_str(const std::string& s) {
    std::string r = "\"";
    for (char c : s) {
        if (c == '"') r += "\\\"";
        else if (c == '\\') r += "\\\\";
        else if (c == '\n') r += "\\n";
        else r += c;
    }
    return r + "\"";
}

inline std::string make_log_msg(const std::string& text) {
    return "{\"type\":\"log\",\"kind\":\"info\",\"text\":" + _json_str(text) + "}";
}

template<typename T>
inline std::string make_array_msg(const std::string& label, const T* ptr, size_t n) {
    std::string s = "{\"type\":\"array\",\"label\":" + _json_str(label) + ",\"data\":[";
    char buf[64];
    for (size_t i = 0; i < n; ++i) {
        if (i) s += ',';
        std::snprintf(buf, sizeof(buf), "%g", (double)ptr[i]);
        s += buf;
    }
    return s + "]}";
}

inline std::string make_matrix_msg(size_t rows, size_t cols, const double* ptr) {
    std::string s = "{\"type\":\"matrix\",\"rows\":" + std::to_string(rows)
                  + ",\"cols\":" + std::to_string(cols) + ",\"data\":[";
    char buf[64];
    for (size_t i = 0, n = rows * cols; i < n; ++i) {
        if (i) s += ',';
        std::snprintf(buf, sizeof(buf), "%g", ptr[i]);
        s += buf;
    }
    return s + "]}";
}

// ─────────────────────────────────────────────────────────────────────────────
// Process / runtime statistics
// ─────────────────────────────────────────────────────────────────────────────

inline std::atomic<std::uint64_t> live_buffers{0};
inline std::atomic<std::uint64_t> live_buffer_bytes{0};
inline std::atomic<std::uint64_t> total_buffers{0};
inline std::atomic<std::uint64_t> total_buffer_bytes{0};

inline std::chrono::steady_clock::time_point& start_time() {
    static std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    return t;
}

inline std::uint64_t sample_rss_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return static_cast<std::uint64_t>(pmc.WorkingSetSize);
    return 0;
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
        return static_cast<std::uint64_t>(info.resident_size);
    return 0;
#else
    long rss = 0;
    if (FILE* f = std::fopen("/proc/self/statm", "r")) {
        long size = 0;
        if (std::fscanf(f, "%ld %ld", &size, &rss) != 2) rss = 0;
        std::fclose(f);
    }
    return static_cast<std::uint64_t>(rss) * static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
#endif
}

struct cpu_sample {
    std::uint64_t proc_ns = 0;
    std::uint64_t wall_ns = 0;
};

inline cpu_sample sample_cpu() {
    cpu_sample s{};
    auto wall = std::chrono::steady_clock::now().time_since_epoch();
    s.wall_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(wall).count());
#if defined(_WIN32)
    FILETIME c, e, k, u;
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) {
        auto to_ns = [](FILETIME ft) {
            ULARGE_INTEGER li;
            li.LowPart = ft.dwLowDateTime;
            li.HighPart = ft.dwHighDateTime;
            return static_cast<std::uint64_t>(li.QuadPart) * 100ULL; // 100ns units
        };
        s.proc_ns = to_ns(k) + to_ns(u);
    }
#else
    struct rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        auto to_ns = [](const struct timeval& tv) {
            return static_cast<std::uint64_t>(tv.tv_sec) * 1'000'000'000ULL
                 + static_cast<std::uint64_t>(tv.tv_usec) * 1'000ULL;
        };
        s.proc_ns = to_ns(ru.ru_utime) + to_ns(ru.ru_stime);
    }
#endif
    return s;
}

inline unsigned cpu_count() {
    unsigned n = std::thread::hardware_concurrency();
    return n ? n : 1u;
}

#if defined(_WIN32)
struct gpu_pdh_state {
    bool ok = false;
    PDH_HQUERY query = nullptr;
    PDH_HCOUNTER counter = nullptr;
};

inline gpu_pdh_state& gpu_state() {
    static gpu_pdh_state s;
    return s;
}

inline void gpu_init() {
    auto& g = gpu_state();
    if (g.ok) return;
    if (PdhOpenQueryW(NULL, 0, &g.query) != ERROR_SUCCESS) return;
    if (PdhAddEnglishCounterW(g.query, L"\\GPU Engine(*)\\Utilization Percentage", 0, &g.counter)
        != ERROR_SUCCESS) {
        PdhCloseQuery(g.query);
        g.query = nullptr;
        return;
    }
    PdhCollectQueryData(g.query);
    g.ok = true;
}

inline double sample_gpu_pct() {
    auto& g = gpu_state();
    if (!g.ok) return -1.0;
    if (PdhCollectQueryData(g.query) != ERROR_SUCCESS) return -1.0;
    DWORD buf_size = 0, item_count = 0;
    PDH_STATUS st = PdhGetFormattedCounterArrayW(
        g.counter, PDH_FMT_DOUBLE, &buf_size, &item_count, nullptr);
    if (st != PDH_MORE_DATA || buf_size == 0) return -1.0;
    std::vector<unsigned char> buf(buf_size);
    auto* arr = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buf.data());
    if (PdhGetFormattedCounterArrayW(
            g.counter, PDH_FMT_DOUBLE, &buf_size, &item_count, arr) != ERROR_SUCCESS)
        return -1.0;
    double sum = 0.0;
    for (DWORD i = 0; i < item_count; ++i) sum += arr[i].FmtValue.doubleValue;
    if (sum > 100.0) sum = 100.0;
    if (sum < 0.0) sum = 0.0;
    return sum;
}
#else
inline void gpu_init() {}
inline double sample_gpu_pct() { return -1.0; }
#endif

inline std::string make_stat_msg(std::uint64_t uptime_ms,
                                 double cpu_pct,
                                 std::uint64_t rss_bytes,
                                 double gpu_pct,
                                 std::uint64_t buffers,
                                 std::uint64_t buffer_bytes,
                                 std::uint64_t total_alloc_n,
                                 std::uint64_t total_alloc_bytes) {
    char buf[96];
    std::string s = "{\"type\":\"stat\"";
    std::snprintf(buf, sizeof(buf), ",\"uptime_ms\":%llu",
                  static_cast<unsigned long long>(uptime_ms));      s += buf;
    std::snprintf(buf, sizeof(buf), ",\"cpu_pct\":%.2f", cpu_pct);  s += buf;
    std::snprintf(buf, sizeof(buf), ",\"rss_bytes\":%llu",
                  static_cast<unsigned long long>(rss_bytes));      s += buf;
    std::snprintf(buf, sizeof(buf), ",\"gpu_pct\":%.2f", gpu_pct);  s += buf;
    std::snprintf(buf, sizeof(buf), ",\"buffers\":%llu",
                  static_cast<unsigned long long>(buffers));        s += buf;
    std::snprintf(buf, sizeof(buf), ",\"buffer_bytes\":%llu",
                  static_cast<unsigned long long>(buffer_bytes));   s += buf;
    std::snprintf(buf, sizeof(buf), ",\"total_alloc\":%llu",
                  static_cast<unsigned long long>(total_alloc_n));  s += buf;
    std::snprintf(buf, sizeof(buf), ",\"total_alloc_bytes\":%llu",
                  static_cast<unsigned long long>(total_alloc_bytes)); s += buf;
    return s + "}";
}

inline std::atomic<bool> sampler_running{false};

inline void sampler_loop() {
    gpu_init();
    cpu_sample prev = sample_cpu();
    const unsigned cores = cpu_count();
    while (sampler_running.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        cpu_sample cur = sample_cpu();
        double cpu_pct = 0.0;
        if (cur.wall_ns > prev.wall_ns) {
            double dw = static_cast<double>(cur.wall_ns - prev.wall_ns);
            double dp = static_cast<double>(
                cur.proc_ns > prev.proc_ns ? cur.proc_ns - prev.proc_ns : 0);
            cpu_pct = (dp / dw) * 100.0 / static_cast<double>(cores);
            if (cpu_pct < 0.0) cpu_pct = 0.0;
            if (cpu_pct > 100.0) cpu_pct = 100.0;
        }
        prev = cur;
        auto uptime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time()).count();
        push_message(make_stat_msg(
            static_cast<std::uint64_t>(uptime_ms),
            cpu_pct,
            sample_rss_bytes(),
            sample_gpu_pct(),
            live_buffers.load(),
            live_buffer_bytes.load(),
            total_buffers.load(),
            total_buffer_bytes.load()));
    }
}

inline std::string make_result_msg(const std::string& label, const std::string& value) {
    return "{\"type\":\"result\",\"label\":" + _json_str(label)
         + ",\"value\":" + _json_str(value) + "}";
}

inline void result(std::int64_t v, const std::string& label = "result") {
    push_message(make_result_msg(label, std::to_string(v)));
}
inline void result(double v, const std::string& label = "result") {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%g", v);
    push_message(make_result_msg(label, buf));
}
inline void result(const std::string& v, const std::string& label = "result") {
    push_message(make_result_msg(label, v));
}

#ifdef _WIN32
    // Enable ANSI escape codes on Windows 10 cmd.exe
#include <windows.h>
#include <locale>
#include <windows.h>
#include <iostream>
void enable_ansi() {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }

    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    setlocale(LC_ALL, ".UTF-8");
}

void load_font() {
    // 1. 리소스 찾기 (ID 101번 RCDATA)
    HRSRC hRes = FindResource(NULL, MAKEINTRESOURCE(101), RT_RCDATA);
    if (hRes == NULL) return;

    // 2. 리소스 로드 및 데이터 포인터 가져오기
    HGLOBAL hData = LoadResource(NULL, hRes);
    void* pFontData = LockResource(hData);
    DWORD nFontLen = SizeofResource(NULL, hRes);

    // 3. 메모리에 있는 데이터를 시스템 폰트 테이블에 추가
    DWORD nFonts = 0;
    HANDLE hFontRes = AddFontMemResourceEx(pFontData, nFontLen, NULL, &nFonts);

    if (hFontRes != NULL) {
        // 4. 추가된 폰트를 콘솔에 적용
        CONSOLE_FONT_INFOEX cfi;
        cfi.cbSize = sizeof(cfi);
        cfi.nFont = 0;
        cfi.dwFontSize.X = 10;
        cfi.dwFontSize.Y = 20;
        cfi.FontFamily = FF_MODERN;
        cfi.FontWeight = FW_NORMAL;
        
        // 폰트 파일의 실제 '이름'을 적어야 합니다 (파일명 X)
        wcscpy_s(cfi.FaceName, L"JetBrainsMono Medium"); 

        SetCurrentConsoleFontEx(GetStdHandle(STD_OUTPUT_HANDLE), FALSE, &cfi);
    }
}

void resize_console(SHORT cols, SHORT rows) {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) return;

    SMALL_RECT minRect = {0, 0, 0, 0};
    SetConsoleWindowInfo(hOut, TRUE, &minRect);

    COORD bufSize = {cols, rows};
    SetConsoleScreenBufferSize(hOut, bufSize);

    SMALL_RECT winRect = {0, 0, static_cast<SHORT>(cols - 1), static_cast<SHORT>(rows - 1)};
    SetConsoleWindowInfo(hOut, TRUE, &winRect);
}
#else
void enable_ansi() {}
void load_font() {}
void resize_console(short, short) {}
#endif

void init() {
    if (initailized) return;
    enable_ansi();
    load_font();
    std::cout << BANNER << std::endl;
    (void)start_time();
    if (!sampler_running.exchange(true)) {
        std::thread(sampler_loop).detach();
    }
    svr.new_task_queue = [] {
        return new httplib::ThreadPool(8);
    };
    svr.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("pong", "text/plain");
    });
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::string content(_binary_assets_index_html, _binary_assets_index_html + _binary_assets_index_html_len);
        res.set_content(content, "text/html; charset=utf-8");
    });
    svr.Get("/console", [](const httplib::Request&, httplib::Response& res) {
        std::string content(_binary_assets_console_html, _binary_assets_console_html + _binary_assets_console_html_len);
        res.set_content(content, "text/html; charset=utf-8");
    });
    svr.Get("/scripts/console.js", [](const httplib::Request&, httplib::Response& res) {
        std::string content(_binary_assets_console_js, _binary_assets_console_js + _binary_assets_console_js_len);
        res.set_content(content, "application/javascript");
    });
    svr.Get("/events", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_chunked_content_provider(
            "text/event-stream",
            [sent_count = size_t(0)](size_t, httplib::DataSink& sink) mutable -> bool {
                std::vector<std::string> pending;
                {
                    std::shared_lock<std::shared_mutex> lock(simulizer::queue_mutex);
                    simulizer::queue_cv.wait_for(lock, std::chrono::seconds(5),
                        [&]{ return !sink.is_writable() || simulizer::message_queue.size() > sent_count; });
                    if (!sink.is_writable()) return false;
                    while (sent_count < simulizer::message_queue.size())
                        pending.push_back(simulizer::message_queue[sent_count++]);
                }
                for (const auto& msg : pending) {
                    std::string sse = "data: " + msg + "\n\n";
                    if (!sink.write(sse.c_str(), sse.size())) return false;
                }
                const char* heartbeat = ": heartbeat\n\n";
                if (!sink.write(heartbeat, std::strlen(heartbeat))) return false;
                return true;
            }
        );
    });
    openBrowser("http://localhost:8080/");
    initailized = true;
}
}

// ─────────────────────────────────────────────────────────────────────────────
// Primitive type aliases
// ─────────────────────────────────────────────────────────────────────────────

using i32 = int32_t;
using f64 = double;

// ─────────────────────────────────────────────────────────────────────────────
// vec2 — 2D double-precision vector  [vector blocks]
// ─────────────────────────────────────────────────────────────────────────────

struct vec2 {
    f64 x, y;
    constexpr vec2(f64 x = 0.0, f64 y = 0.0) : x(x), y(y) {}
};

inline vec2 operator+(vec2 a, vec2 b)  { return {a.x + b.x, a.y + b.y}; }
inline vec2 operator-(vec2 a, vec2 b)  { return {a.x - b.x, a.y - b.y}; }
inline vec2 operator*(vec2 v, f64 s)   { return {v.x * s,   v.y * s};   }
inline vec2 operator*(f64 s, vec2 v)   { return {v.x * s,   v.y * s};   }
inline vec2 operator-(vec2 v)          { return {-v.x, -v.y};            }

inline f64  vec2_dot(vec2 a, vec2 b)          { return a.x*b.x + a.y*b.y; }
inline f64  vec2_len_sq(vec2 v)               { return v.x*v.x + v.y*v.y; }
inline f64  vec2_len(vec2 v)                  { return std::sqrt(vec2_len_sq(v)); }
inline vec2 vec2_normalize(vec2 v)            { f64 l = vec2_len(v); return {v.x/l, v.y/l}; }
inline f64  vec2_cross_scalar(vec2 a, vec2 b) { return a.x*b.y - a.y*b.x; }
inline f64  vec2_proj_scalar(vec2 a, vec2 b)  { return vec2_dot(a, b) / vec2_len(b); }
inline vec2 vec2_proj_vec(vec2 a, vec2 b)     { f64 s = vec2_dot(a, b) / vec2_len_sq(b); return b * s; }

// ─────────────────────────────────────────────────────────────────────────────
// vec3 — 3D double-precision vector  [vector blocks]
// ─────────────────────────────────────────────────────────────────────────────

struct vec3 {
    f64 x, y, z;
    constexpr vec3(f64 x = 0.0, f64 y = 0.0, f64 z = 0.0) : x(x), y(y), z(z) {}
};

inline vec3 operator+(vec3 a, vec3 b)  { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline vec3 operator-(vec3 a, vec3 b)  { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline vec3 operator*(vec3 v, f64 s)   { return {v.x*s,   v.y*s,   v.z*s};   }
inline vec3 operator*(f64 s, vec3 v)   { return {v.x*s,   v.y*s,   v.z*s};   }
inline vec3 operator-(vec3 v)          { return {-v.x, -v.y, -v.z};          }

inline f64  vec3_dot(vec3 a, vec3 b)   { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline f64  vec3_len_sq(vec3 v)        { return v.x*v.x + v.y*v.y + v.z*v.z; }
inline f64  vec3_len(vec3 v)           { return std::sqrt(vec3_len_sq(v)); }
inline vec3 vec3_normalize(vec3 v)     { f64 l = vec3_len(v); return {v.x/l, v.y/l, v.z/l}; }
inline vec3 vec3_cross(vec3 a, vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline f64  vec3_proj_scalar(vec3 a, vec3 b) { return vec3_dot(a, b) / vec3_len(b); }
inline vec3 vec3_proj_vec(vec3 a, vec3 b)    { f64 s = vec3_dot(a, b) / vec3_len_sq(b); return b * s; }


// ─────────────────────────────────────────────────────────────────────────────
// Bondary 2D
// ─────────────────────────────────────────────────────────────────────────────
struct Boundary2D {
    f64 t, x, y;
    f64 tx, ty;
    f64 nx, ny;
};
struct Boundary3D {
    f64 t, x, y, z;
    f64 nx, ny, nz;
};

// ─────────────────────────────────────────────────────────────────────────────
// Tensor — arbitrary-dimensional tensor with runtime storage  [tensor blocks]
// ─────────────────────────────────────────────────────────────────────────────

// =========================
// Buffer
// =========================

template <typename T>
class Buffer {
    T* ptr_;
    size_t size_;

public:
    Buffer() : ptr_(nullptr), size_(0) {}

    explicit Buffer(size_t n) : ptr_(n ? new T[n] : nullptr), size_(n) {
        if (n) {
            const std::uint64_t b = static_cast<std::uint64_t>(n) * sizeof(T);
            simulizer::live_buffers.fetch_add(1, std::memory_order_relaxed);
            simulizer::live_buffer_bytes.fetch_add(b, std::memory_order_relaxed);
            simulizer::total_buffers.fetch_add(1, std::memory_order_relaxed);
            simulizer::total_buffer_bytes.fetch_add(b, std::memory_order_relaxed);
        }
    }

    Buffer(size_t n, const T& value) : Buffer(n) {
        for (size_t i = 0; i < size_; ++i) ptr_[i] = value;
    }

    ~Buffer() {
        if (size_) {
            simulizer::live_buffers.fetch_sub(1, std::memory_order_relaxed);
            simulizer::live_buffer_bytes.fetch_sub(
                static_cast<std::uint64_t>(size_) * sizeof(T),
                std::memory_order_relaxed);
        }
        delete[] ptr_;
    }

    Buffer(const Buffer& other) : Buffer(other.size_) {
        for (size_t i = 0; i < size_; ++i) ptr_[i] = other.ptr_[i];
    }

    Buffer& operator=(const Buffer& other) {
        if (this == &other) return *this;
        if (size_) {
            simulizer::live_buffers.fetch_sub(1, std::memory_order_relaxed);
            simulizer::live_buffer_bytes.fetch_sub(
                static_cast<std::uint64_t>(size_) * sizeof(T),
                std::memory_order_relaxed);
        }
        delete[] ptr_;
        size_ = other.size_;
        ptr_ = size_ ? new T[size_] : nullptr;
        if (size_) {
            const std::uint64_t b = static_cast<std::uint64_t>(size_) * sizeof(T);
            simulizer::live_buffers.fetch_add(1, std::memory_order_relaxed);
            simulizer::live_buffer_bytes.fetch_add(b, std::memory_order_relaxed);
            simulizer::total_buffers.fetch_add(1, std::memory_order_relaxed);
            simulizer::total_buffer_bytes.fetch_add(b, std::memory_order_relaxed);
        }
        for (size_t i = 0; i < size_; ++i) ptr_[i] = other.ptr_[i];
        return *this;
    }

    Buffer(Buffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this == &other) return *this;
        if (size_) {
            simulizer::live_buffers.fetch_sub(1, std::memory_order_relaxed);
            simulizer::live_buffer_bytes.fetch_sub(
                static_cast<std::uint64_t>(size_) * sizeof(T),
                std::memory_order_relaxed);
        }
        delete[] ptr_;
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
        return *this;
    }

    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }

    size_t size() const { return size_; }
};

// =========================
// Tensor
// =========================

template <typename T>
class Tensor {
    std::vector<size_t> shape_;   // [d0, d1, ...]
    std::vector<size_t> stride_;  // row-major stride
    Buffer<T> data_;

public:
    // =========================
    // ctor / dtor
    // =========================

    Tensor() {}

    Tensor(const std::vector<size_t>& shape)
        : shape_(shape) {

        size_t dim = shape_.size();
        if (dim == 0)
            throw std::invalid_argument("shape must have at least 1 dimension");

        stride_.resize(dim);

        size_t acc = 1;
        for (size_t i = dim; i-- > 0;) {
            stride_[i] = acc;
            acc *= shape_[i];
        }

        data_ = Buffer<T>(acc);
    }

    ~Tensor() {}

    Tensor(const Tensor& other)
        : shape_(other.shape_),
          stride_(other.stride_),
          data_(other.data_) {}

    Tensor& operator=(const Tensor& other) {
        if (this == &other) return *this;
        shape_ = other.shape_;
        stride_ = other.stride_;
        data_ = other.data_;
        return *this;
    }

    // =========================
    // properties
    // =========================

    size_t dim() const { return shape_.size(); }

    const size_t* shape() const { return shape_.data(); }

    size_t size() const { return data_.size(); }

    // =========================
    // indexing (variadic)
    // =========================

private:
    template <typename... Args>
    size_t offset(Args... args) const {
        size_t idx[] = { static_cast<size_t>(args)... };
        size_t dim = shape_.size();

        if (sizeof...(Args) != dim)
            throw std::out_of_range("invalid dimension");

        size_t off = 0;
        for (size_t i = 0; i < dim; ++i) {
            if (idx[i] >= shape_[i])
                throw std::out_of_range("index out of range");
            off += idx[i] * stride_[i];
        }
        return off;
    }

public:
    template <typename... Args>
    T& operator()(Args... args) {
        return data_[offset(args...)];
    }

    template <typename... Args>
    const T& operator()(Args... args) const {
        return data_[offset(args...)];
    }

    // =========================
    // dynamic indexing
    // =========================

    T& at(const size_t* idx) {
        size_t dim = shape_.size();
        size_t off = 0;
        for (size_t i = 0; i < dim; ++i) {
            if (idx[i] >= shape_[i])
                throw std::out_of_range("index out of range");
            off += idx[i] * stride_[i];
        }
        return data_[off];
    }

    // =========================
    // linear access
    // =========================

    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
};

// ── tensor_random helpers ─────────────────────────────────────────────────────

inline Tensor<f64> tensor_uniform(const std::vector<i32>& shape, f64 lo, f64 hi) {
    std::vector<size_t> s(shape.begin(), shape.end());
    Tensor<f64> t(s);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<f64> dist(lo, hi);
    for (size_t i = 0; i < t.size(); ++i) t[i] = dist(rng);
    return t;
}

inline Tensor<f64> tensor_normal(const std::vector<i32>& shape, f64 mean, f64 stddev) {
    std::vector<size_t> s(shape.begin(), shape.end());
    Tensor<f64> t(s);
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<f64> dist(mean, stddev);
    for (size_t i = 0; i < t.size(); ++i) t[i] = dist(rng);
    return t;
}

// ── matrix blocks ─────────────────────────────────────────────────────────────
// All matrices are 2D Tensor<f64> in row-major storage.

inline Tensor<f64> matrix_create(i32 rows, i32 cols) {
    return Tensor<f64>({ static_cast<size_t>(rows), static_cast<size_t>(cols) });
}

inline Tensor<f64> matrix_identity(i32 n) {
    Tensor<f64> m({ static_cast<size_t>(n), static_cast<size_t>(n) });
    for (i32 i = 0; i < n; ++i) m(i, i) = 1.0;
    return m;
}

inline Tensor<f64> matrix_transpose(const Tensor<f64>& a) {
    size_t r = a.shape()[0], c = a.shape()[1];
    Tensor<f64> t({ c, r });
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            t(j, i) = a(i, j);
    return t;
}

inline Tensor<f64> matrix_matmul(const Tensor<f64>& a, const Tensor<f64>& b) {
    size_t n = a.shape()[0], k = a.shape()[1], m = b.shape()[1];
    if (b.shape()[0] != k)
        throw std::invalid_argument("matrix_matmul: inner dimensions mismatch");
    Tensor<f64> c({ n, m });
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j) {
            f64 s = 0.0;
            for (size_t p = 0; p < k; ++p) s += a(i, p) * b(p, j);
            c(i, j) = s;
        }
    return c;
}

inline f64 matrix_trace(const Tensor<f64>& a) {
    size_t n = a.shape()[0];
    f64 s = 0.0;
    for (size_t i = 0; i < n; ++i) s += a(i, i);
    return s;
}

inline f64 matrix_det(const Tensor<f64>& a) {
    size_t n = a.shape()[0];
    std::vector<f64> m(n * n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) m[i * n + j] = a(i, j);
    f64 det = 1.0;
    for (size_t i = 0; i < n; ++i) {
        size_t piv = i;
        for (size_t r = i + 1; r < n; ++r)
            if (std::fabs(m[r * n + i]) > std::fabs(m[piv * n + i])) piv = r;
        if (std::fabs(m[piv * n + i]) < 1e-12) return 0.0;
        if (piv != i) {
            for (size_t c = 0; c < n; ++c) std::swap(m[i * n + c], m[piv * n + c]);
            det = -det;
        }
        det *= m[i * n + i];
        for (size_t r = i + 1; r < n; ++r) {
            f64 f = m[r * n + i] / m[i * n + i];
            for (size_t c = i; c < n; ++c) m[r * n + c] -= f * m[i * n + c];
        }
    }
    return det;
}

inline Tensor<f64> matrix_inverse(const Tensor<f64>& a) {
    size_t n = a.shape()[0];
    size_t w = 2 * n;
    std::vector<f64> m(n * w, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) m[i * w + j] = a(i, j);
        m[i * w + n + i] = 1.0;
    }
    for (size_t i = 0; i < n; ++i) {
        size_t piv = i;
        for (size_t r = i + 1; r < n; ++r)
            if (std::fabs(m[r * w + i]) > std::fabs(m[piv * w + i])) piv = r;
        if (piv != i)
            for (size_t c = 0; c < w; ++c) std::swap(m[i * w + c], m[piv * w + c]);
        f64 d = m[i * w + i];
        if (std::fabs(d) < 1e-12)
            throw std::runtime_error("matrix_inverse: singular matrix");
        for (size_t c = 0; c < w; ++c) m[i * w + c] /= d;
        for (size_t r = 0; r < n; ++r) {
            if (r == i) continue;
            f64 f = m[r * w + i];
            for (size_t c = 0; c < w; ++c) m[r * w + c] -= f * m[i * w + c];
        }
    }
    Tensor<f64> inv({ n, n });
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) inv(i, j) = m[i * w + n + j];
    return inv;
}

// ── debug_log block ───────────────────────────────────────────────────────────

inline void debug_log(i32 val) {
    simulizer::push_message(simulizer::make_log_msg(std::string("[log] ") + std::to_string(val)));
}
inline void debug_log(f64 val) {
    simulizer::push_message(simulizer::make_log_msg(std::string("[log] ") + std::to_string(val)));
}
inline void debug_log(vec2 v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "[log] vec2(%g, %g)", v.x, v.y);
    simulizer::push_message(simulizer::make_log_msg(buf));
}
inline void debug_log(vec3 v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "[log] vec3(%g, %g, %g)", v.x, v.y, v.z);
    simulizer::push_message(simulizer::make_log_msg(buf));
}

inline void debug_log(i32* ptr, i32 cap) {
    simulizer::push_message(simulizer::make_array_msg(
        "i32[" + std::to_string(cap) + "]", ptr, (size_t)cap));
}

inline void debug_log(f64* ptr, i32 cap) {
    simulizer::push_message(simulizer::make_array_msg(
        "f64[" + std::to_string(cap) + "]", ptr, (size_t)cap));
}

// Forward-declared so debug_log(Tensor) and show_mat below can tag their
// messages with the current holder; the actual definition lives next to
// debug_set_holder / debug_series further down.
namespace _sim_detail { inline i32& current_holder(); }

// Mirrors the web's debug_log(Tensor): emit a multi-line text log via
// {"type":"log"} rather than a heatmap. Tagged with the current holder so
// debug_set_holder routes the dump to the right series, same as the web
// side does in clang-worker.ts (__sim_log_tensor → log() with currentHolderId).
namespace _sim_detail {
    template <typename T>
    inline void _print_tensor_recur(std::string& out, const Tensor<T>& t,
                                    size_t offset, size_t dim_idx, int indent,
                                    const char* fmt) {
        const size_t* shape = t.shape();
        const size_t dim = t.dim();
        const size_t this_dim = shape[dim_idx];
        size_t stride = 1;
        for (size_t i = dim_idx + 1; i < dim; ++i) stride *= shape[i];

        if (dim_idx + 1 == dim) {
            out += "[";
            char buf[64];
            for (size_t i = 0; i < this_dim; ++i) {
                if (i) out += ",";
                std::snprintf(buf, sizeof(buf), fmt, t[offset + i]);
                out += buf;
            }
            out += "]";
            return;
        }
        out += "[\n";
        const std::string next_pad((indent + 1) * 2, ' ');
        const std::string this_pad(indent * 2, ' ');
        for (size_t i = 0; i < this_dim; ++i) {
            out += next_pad;
            _print_tensor_recur(out, t, offset + i * stride, dim_idx + 1, indent + 1, fmt);
            if (i + 1 < this_dim) out += ",";
            out += "\n";
        }
        out += this_pad + "]";
    }

    template <typename T>
    inline std::string _format_tensor_text(const Tensor<T>& t, const char* dtype, const char* fmt) {
        std::string s = "\xF0\x9F\xA7\xA0 Tensor(shape: [";  // 🧠
        for (size_t i = 0; i < t.dim(); ++i) {
            if (i) s += ", ";
            s += std::to_string(t.shape()[i]);
        }
        s += "], dtype: ";
        s += dtype;
        s += ")";
        if (t.dim() > 0 && t.size() > 0) {
            s += "\n";
            _print_tensor_recur(s, t, 0, 0, 0, fmt);
        }
        return s;
    }

    template <typename T>
    inline void _emit_tensor_log(const Tensor<T>& t, const char* dtype, const char* fmt) {
        const std::string text = _format_tensor_text(t, dtype, fmt);
        std::string msg = "{\"type\":\"log\",\"kind\":\"info\",\"text\":" + simulizer::_json_str(text);
        i32 hid = current_holder();
        if (hid >= 0) msg += ",\"holder\":" + std::to_string(hid);
        msg += "}";
        simulizer::push_message(msg);
    }
}

inline void debug_log(Tensor<i32>& t) { _sim_detail::_emit_tensor_log(t, "i32", "%d"); }
inline void debug_log(Tensor<f64>& t) { _sim_detail::_emit_tensor_log(t, "f64", "%g"); }

// ── show_mat block ────────────────────────────────────────────────────────────

inline void show_mat(Tensor<f64>& t) {
    if (t.dim() != 2) return;
    size_t rows = t.shape()[0], cols = t.shape()[1];
    std::string s = "{\"type\":\"matrix\",\"rows\":" + std::to_string(rows)
                  + ",\"cols\":" + std::to_string(cols) + ",\"data\":[";
    char buf[64];
    for (size_t i = 0, n = t.size(); i < n; ++i) {
        if (i) s += ',';
        std::snprintf(buf, sizeof(buf), "%g", t[i]);
        s += buf;
    }
    s += "]";
    i32 hid = _sim_detail::current_holder();
    if (hid >= 0) s += ",\"holder\":" + std::to_string(hid);
    simulizer::push_message(s + "}");
}


// ── debug_bar / debug_bar_set blocks ─────────────────────────────────────────

namespace _sim_detail {
    struct BarState { i32 min, max; };
    inline std::vector<BarState>& bar_registry() {
        static std::vector<BarState> v;
        return v;
    }
}

inline i32 debug_bar(i32 min, i32 max) {
    auto& reg = _sim_detail::bar_registry();
    i32 id = static_cast<i32>(reg.size());
    reg.push_back({min, max});
    simulizer::push_message(
        "{\"type\":\"bar\",\"id\":" + std::to_string(id) +
        ",\"value\":" + std::to_string(min) +
        ",\"min\":" + std::to_string(min) +
        ",\"max\":" + std::to_string(max) + "}"
    );
    return id;
}

inline void debug_bar_set(i32 bar_id, i32 val) {
    auto& reg = _sim_detail::bar_registry();
    if (bar_id < 0 || bar_id >= static_cast<i32>(reg.size())) return;
    simulizer::push_message(
        "{\"type\":\"bar\",\"id\":" + std::to_string(bar_id) +
        ",\"value\":" + std::to_string(val) + "}"
    );
}

// ── debug_series / debug_set_holder blocks ───────────────────────────────────

namespace _sim_detail {
    inline i32& series_next_id() {
        static i32 id = 1;
        return id;
    }
    inline i32& current_holder() {
        static i32 id = -1;
        return id;
    }
}

inline i32 debug_series() {
    i32 id = _sim_detail::series_next_id()++;
    simulizer::push_message(
        "{\"type\":\"series\",\"id\":" + std::to_string(id) + "}"
    );
    return id;
}

inline void debug_set_holder(i32 holder_id) {
    _sim_detail::current_holder() = holder_id;
}

// ── graph_arr / graph_arr_range blocks ────────────────────────────────────────

template<typename T>
inline std::string _make_graph_msg(const std::vector<T>& arr) {
    std::string s = "{\"type\":\"graph\",\"data\":[";
    char buf[64];
    for (size_t i = 0; i < arr.size(); ++i) {
        if (i) s += ',';
        std::snprintf(buf, sizeof(buf), "%g", (double)arr[i]);
        s += buf;
    }
    s += "]";
    i32 hid = _sim_detail::current_holder();
    if (hid >= 0) s += ",\"holder\":" + std::to_string(hid);
    return s + "}";
}

template<typename T>
inline std::string _make_graph_range_msg(const std::vector<T>& arr, f64 mn, f64 mx) {
    std::string s = "{\"type\":\"graph\",\"data\":[";
    char buf[64];
    for (size_t i = 0; i < arr.size(); ++i) {
        if (i) s += ',';
        std::snprintf(buf, sizeof(buf), "%g", (double)arr[i]);
        s += buf;
    }
    std::snprintf(buf, sizeof(buf), "%g", mn);
    s += std::string("],\"min\":") + buf;
    std::snprintf(buf, sizeof(buf), "%g", mx);
    s += std::string(",\"max\":") + buf;
    i32 hid = _sim_detail::current_holder();
    if (hid >= 0) s += ",\"holder\":" + std::to_string(hid);
    return s + "}";
}

inline void graph_arr_i32(const std::vector<i32>& arr) {
    simulizer::push_message(_make_graph_msg(arr));
}

inline void graph_arr_f64(const std::vector<f64>& arr) {
    simulizer::push_message(_make_graph_msg(arr));
}

inline void graph_arr_range_i32(const std::vector<i32>& arr, f64 mn, f64 mx) {
    simulizer::push_message(_make_graph_range_msg(arr, mn, mx));
}

inline void graph_arr_range_f64(const std::vector<f64>& arr, f64 mn, f64 mx) {
    simulizer::push_message(_make_graph_range_msg(arr, mn, mx));
}

#endif  // __EMSCRIPTEN__
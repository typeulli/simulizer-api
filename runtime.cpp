// runtime.cpp — Simulizer emscripten MAIN_MODULE entry point.
//
// Builds with `-sMAIN_MODULE=2 --js-library simulizer_lib.js` to produce
// runtime.js + runtime.wasm. The main module is loaded once on page mount;
// every user-submitted C++ program is built as a SIDE_MODULE and linked
// against this runtime at dlopen time.
//
// _keepalive_imports references each host function via a (volatile-gated,
// dead-at-runtime) call so emcc keeps the JS library binding without
// forcing a WASM function-pointer wrapper for it (which would trigger
// addFunction → convertJsFunctionToWasm at init).

#include "simstd.hpp"

__attribute__((noinline, used))
static int _keepalive_imports() {
    volatile int gate = 0;
    if (!gate) return 0;

    int    di = 0;
    double dd = 0.0;
    int    da_i[1] = {0};
    double da_d[1] = {0.0};
    int sink = 0;

    __sim_log_i32(di);
    __sim_log_f64(dd);
    __sim_log_vec2(dd, dd);
    __sim_log_vec3(dd, dd, dd);
    __sim_log_arr_i32(da_i, di);
    __sim_log_arr_f64(da_d, di);
    __sim_log_tensor(di);

    sink += __sim_debug_bar(di, di);
    __sim_debug_bar_set(di, di);
    sink += __sim_debug_series();
    __sim_debug_set_holder(di);
    __sim_show_mat(di);

    __sim_graph_arr_i32(da_i, di);
    __sim_graph_arr_f64(da_d, di);
    __sim_graph_arr_range_i32(da_i, di, dd, dd);
    __sim_graph_arr_range_f64(da_d, di, dd, dd);

    sink += __sim_tensor_create(di, da_i, di);
    sink += __sim_tensor_random(di, di, dd, dd, da_i, di);
    sink += __sim_tensor_add(di, di);
    sink += __sim_tensor_sub(di, di);
    sink += __sim_tensor_matmul(di, di);
    sink += __sim_tensor_neg(di);
    sink += __sim_tensor_elemul(di, di);
    sink += __sim_tensor_scale(di, dd);
    sink += __sim_tensor_save(di, di);
    sink += __sim_tensor_set(di, di, di, di, di, di, di, di, dd);
    sink += (int)__sim_tensor_get(di, di, di, di, di, di, di, di);
    sink += __sim_tensor_perlin(di, di, di);

    sink += __sim_matrix_create(di, di, di);
    sink += __sim_matrix_matmul(di, di);
    sink += __sim_matrix_transpose(di);
    sink += __sim_matrix_inverse(di);
    sink += (int)__sim_matrix_det(di);
    sink += (int)__sim_matrix_trace(di);
    sink += __sim_matrix_identity(di);

    return sink;
}

int main() {
    return _keepalive_imports();
}

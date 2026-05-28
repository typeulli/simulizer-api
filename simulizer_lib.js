// simulizer_lib.js — Emscripten JS library bridging __sim_* C exports to the
// host worker via Module.bridge. Linked into runtime.js by --js-library.
//
// Each function declares a __sig (emcc signature: v=void, i=i32, d=f64) so
// addFunction can wrap it as a WASM-callable function pointer for the
// keepalive table in runtime.cpp.

mergeInto(LibraryManager.library, {
    __sim_log_i32__sig: 'vi',
    __sim_log_i32: function(val) {
        Module.bridge.log_i32(val);
    },
    __sim_log_f64__sig: 'vd',
    __sim_log_f64: function(val) {
        Module.bridge.log_f64(val);
    },
    __sim_log_vec2__sig: 'vdd',
    __sim_log_vec2: function(x, y) {
        Module.bridge.log_vec2(x, y);
    },
    __sim_log_vec3__sig: 'vddd',
    __sim_log_vec3: function(x, y, z) {
        Module.bridge.log_vec3(x, y, z);
    },
    __sim_log_arr_i32__sig: 'vii',
    __sim_log_arr_i32: function(ptr, cap) {
        Module.bridge.log_arr_i32(Module.wasmMemory, ptr, cap);
    },
    __sim_log_arr_f64__sig: 'vii',
    __sim_log_arr_f64: function(ptr, cap) {
        Module.bridge.log_arr_f64(Module.wasmMemory, ptr, cap);
    },
    __sim_log_tensor__sig: 'vi',
    __sim_log_tensor: function(id) {
        Module.bridge.log_tensor(id);
    },

    __sim_debug_bar__sig: 'iii',
    __sim_debug_bar: function(mn, mx) {
        return Module.bridge.debug_bar(mn, mx);
    },
    __sim_debug_bar_set__sig: 'vii',
    __sim_debug_bar_set: function(id, val) {
        Module.bridge.debug_bar_set(id, val);
    },
    __sim_debug_series__sig: 'i',
    __sim_debug_series: function() {
        return Module.bridge.debug_series();
    },
    __sim_debug_set_holder__sig: 'vi',
    __sim_debug_set_holder: function(id) {
        Module.bridge.debug_set_holder(id);
    },
    __sim_show_mat__sig: 'vi',
    __sim_show_mat: function(id) {
        Module.bridge.show_mat(id);
    },

    __sim_graph_arr_i32__sig: 'vii',
    __sim_graph_arr_i32: function(ptr, cap) {
        Module.bridge.graph_arr_i32(Module.wasmMemory, ptr, cap);
    },
    __sim_graph_arr_f64__sig: 'vii',
    __sim_graph_arr_f64: function(ptr, cap) {
        Module.bridge.graph_arr_f64(Module.wasmMemory, ptr, cap);
    },
    __sim_graph_arr_range_i32__sig: 'viidd',
    __sim_graph_arr_range_i32: function(ptr, cap, mn, mx) {
        Module.bridge.graph_arr_range_i32(Module.wasmMemory, ptr, cap, mn, mx);
    },
    __sim_graph_arr_range_f64__sig: 'viidd',
    __sim_graph_arr_range_f64: function(ptr, cap, mn, mx) {
        Module.bridge.graph_arr_range_f64(Module.wasmMemory, ptr, cap, mn, mx);
    },

    __sim_tensor_create__sig: 'iiii',
    __sim_tensor_create: function(varid, ptr, dim) {
        return Module.bridge.js_tensor_create(Module.wasmMemory, varid, ptr, dim);
    },
    __sim_tensor_random__sig: 'iiiddii',
    __sim_tensor_random: function(varid, distType, p1, p2, ptr, dim) {
        return Module.bridge.js_tensor_random(Module.wasmMemory, varid, distType, p1, p2, ptr, dim);
    },
    __sim_tensor_add__sig: 'iii',
    __sim_tensor_add: function(lhs, rhs)         { return Module.bridge.js_tensor_add(lhs, rhs); },
    __sim_tensor_sub__sig: 'iii',
    __sim_tensor_sub: function(lhs, rhs)         { return Module.bridge.js_tensor_sub(lhs, rhs); },
    __sim_tensor_matmul__sig: 'iii',
    __sim_tensor_matmul: function(lhs, rhs)      { return Module.bridge.js_tensor_matmul(lhs, rhs); },
    __sim_tensor_neg__sig: 'ii',
    __sim_tensor_neg: function(v)                { return Module.bridge.js_tensor_neg(v); },
    __sim_tensor_elemul__sig: 'iii',
    __sim_tensor_elemul: function(lhs, rhs)      { return Module.bridge.js_tensor_elemul(lhs, rhs); },
    __sim_tensor_scale__sig: 'iid',
    __sim_tensor_scale: function(v, s)           { return Module.bridge.js_tensor_scale(v, s); },
    __sim_tensor_save__sig: 'iii',
    __sim_tensor_save: function(out, src)        { return Module.bridge.js_tensor_save(out, src); },
    __sim_tensor_set__sig: 'iiiiiiiiid',
    __sim_tensor_set: function(id, n, i0, i1, i2, i3, i4, i5, val) {
        return Module.bridge.js_tensor_set(id, n, i0, i1, i2, i3, i4, i5, val);
    },
    __sim_tensor_get__sig: 'diiiiiiii',
    __sim_tensor_get: function(id, n, i0, i1, i2, i3, i4, i5) {
        return Module.bridge.js_tensor_get(id, n, i0, i1, i2, i3, i4, i5);
    },
    __sim_tensor_perlin__sig: 'iiii',
    __sim_tensor_perlin: function(varid, rows, cols) {
        return Module.bridge.js_tensor_perlin(varid, rows, cols);
    },

    __sim_matrix_create__sig: 'iiii',
    __sim_matrix_create: function(varid, rows, cols) { return Module.bridge.js_matrix_create(varid, rows, cols); },
    __sim_matrix_matmul__sig: 'iii',
    __sim_matrix_matmul: function(lhs, rhs)          { return Module.bridge.js_matrix_matmul(lhs, rhs); },
    __sim_matrix_transpose__sig: 'ii',
    __sim_matrix_transpose: function(v)              { return Module.bridge.js_matrix_transpose(v); },
    __sim_matrix_inverse__sig: 'ii',
    __sim_matrix_inverse: function(v)                { return Module.bridge.js_matrix_inverse(v); },
    __sim_matrix_det__sig: 'di',
    __sim_matrix_det: function(v)                    { return Module.bridge.js_matrix_det(v); },
    __sim_matrix_trace__sig: 'di',
    __sim_matrix_trace: function(v)                  { return Module.bridge.js_matrix_trace(v); },
    __sim_matrix_identity__sig: 'ii',
    __sim_matrix_identity: function(n)               { return Module.bridge.js_matrix_identity(n); },
});

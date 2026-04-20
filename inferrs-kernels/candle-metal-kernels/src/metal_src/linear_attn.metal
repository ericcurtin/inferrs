#include <metal_stdlib>
using namespace metal;

/// Fused decay gate for GatedDeltaNet SSM layers.
///
/// Computes: g[i] = exp(-a_exp[h] * softplus(a_input[i] + dt_bias[h]))
/// where h = i % n_heads, softplus(x) = max(x,0) + log(1 + exp(-|x|))
///
/// Replaces ~8 element-wise candle dispatches (broadcast_add, abs, neg, exp,
/// ones_like, log, add, mul, neg, exp) with a single kernel launch.
///
/// Grid: 1D, one thread per element (b*t*n_heads total).
/// Input a_input may be F32 or BF16 (separate kernel variants).

inline float softplus_gate(float a_val, float dt_b, float a_e) {
    float x = a_val + dt_b;
    float abs_x = abs(x);
    // numerically stable softplus: max(x,0) + log(1 + exp(-|x|))
    float sp = fmax(x, 0.0f) + log(1.0f + exp(-abs_x));
    return exp(-a_e * sp);
}

kernel void kernel_compute_decay_gate_f32(
    device const float* a_input  [[buffer(0)]],
    device const float* dt_bias  [[buffer(1)]],
    device const float* a_exp    [[buffer(2)]],
    device float*       out_g    [[buffer(3)]],
    constant uint&      n_heads  [[buffer(4)]],
    constant uint&      n_total  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n_total) return;
    const uint h = tid % n_heads;
    out_g[tid] = softplus_gate(a_input[tid], dt_bias[h], a_exp[h]);
}

kernel void kernel_compute_decay_gate_bf16f32(
    device const bfloat* a_input [[buffer(0)]],
    device const float*  dt_bias [[buffer(1)]],
    device const float*  a_exp   [[buffer(2)]],
    device float*        out_g   [[buffer(3)]],
    constant uint&       n_heads [[buffer(4)]],
    constant uint&       n_total [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n_total) return;
    const uint h = tid % n_heads;
    out_g[tid] = softplus_gate((float)a_input[tid], dt_bias[h], a_exp[h]);
}

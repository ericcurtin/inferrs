// Fused decay gate for GatedDeltaNet SSM layers — CUDA port of linear_attn.metal.
//
// Computes: g[i] = exp(-a_exp[h] * softplus(a_input[i] + dt_bias[h]))
// where h = i % n_heads, softplus(x) = max(x,0) + log(1 + exp(-|x|))
//
// Replaces ~8 element-wise candle dispatches (broadcast_add, abs, neg, exp,
// ones_like, log, add, mul, neg, exp) with a single kernel launch.
//
// Grid: 1D, one thread per element (b*t*n_heads total).
// Input a_input may be F32 or BF16 (separate kernel variants).

#include "cuda_bf16.h"
#include <stdint.h>

// Numerically stable softplus composed with the decay exp:
//   sp = max(x,0) + log(1 + exp(-|x|))   with x = a_val + dt_b
//   g  = exp(-a_e * sp)
// exp(-|x|) <= 1 so log(1 + exp(-|x|)) stays finite and >= 0, avoiding the
// overflow that log(1 + exp(x)) would have for large positive x (FTZ-safe on
// CUDA even with subnormal-flushing, since exp(-|x|) for large |x| underflows
// to 0 and log(1) = 0 exactly).
__device__ __forceinline__ float softplus_gate(float a_val, float dt_b, float a_e) {
    float x = a_val + dt_b;
    float abs_x = fabsf(x);
    float sp = fmaxf(x, 0.0f) + logf(1.0f + __expf(-abs_x));
    return __expf(-a_e * sp);
}

// `__launch_bounds__(256)` matches the block size selected in cuda_linear_attn.rs
// and lets nvcc tighten register allocation (the kernel is trivially arith-heavy
// with no shared memory, so occupancy benefits from explicit bounds).
extern "C" __global__ __launch_bounds__(256) void compute_decay_gate_f32(
    const float* __restrict__ a_input,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_exp,
    float* __restrict__ out_g,
    const uint32_t n_heads,
    const uint32_t n_total
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_total) return;
    const uint32_t h = tid % n_heads;
    out_g[tid] = softplus_gate(a_input[tid], dt_bias[h], a_exp[h]);
}

extern "C" __global__ __launch_bounds__(256) void compute_decay_gate_bf16f32(
    const __nv_bfloat16* __restrict__ a_input,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_exp,
    float* __restrict__ out_g,
    const uint32_t n_heads,
    const uint32_t n_total
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_total) return;
    const uint32_t h = tid % n_heads;
    out_g[tid] = softplus_gate(__bfloat162float(a_input[tid]), dt_bias[h], a_exp[h]);
}

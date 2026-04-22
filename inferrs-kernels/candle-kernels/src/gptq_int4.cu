// GPTQ-Int4 direct GEMV kernel.
// Layout (HuggingFace AutoGPTQ v1):
//   qweight[in_dim/8, out_dim] int32 — 8 int4 nibbles per int32, LSB-first
//   scales[n_groups, out_dim]  bf16  — per-group scales
//   qzeros[n_groups, out_dim/8] int32 — packed zero-points (same packing as qweight)
//
// Each warp computes one output neuron (column of weight matrix).
// Grid=(ceil_div(out_dim, 1), 1, 1), Block=(WARP_SIZE=32, 1, 1).

#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include <stdint.h>

#define WARP_SIZE 32

extern "C" __global__ void dequantize_mul_mat_vec_gptq_int4_bf16in(
    const int32_t *__restrict__ qweight,  // [in_dim/8, out_dim]
    const nv_bfloat16 *__restrict__ scales, // [n_groups, out_dim]
    const int32_t *__restrict__ qzeros,   // [n_groups, out_dim/8]
    const nv_bfloat16 *__restrict__ x,    // [in_dim]
    float         *__restrict__ dst,      // [out_dim]
    int in_dim,
    int out_dim,
    int group_size)
{
    const int col = blockIdx.x;
    if (col >= out_dim) return;

    float sum = 0.f;

    for (int row = threadIdx.x; row < in_dim; row += WARP_SIZE) {
        const int group    = row / group_size;
        const float scale  = __bfloat162float(scales[group * out_dim + col]);

        // Unpack weight nibble: qweight layout is [in_dim/8, out_dim]
        const int qw_row   = row / 8;
        const int qw_bit   = (row % 8) * 4;
        const int nibble   = (qweight[qw_row * out_dim + col] >> qw_bit) & 0xF;

        // Unpack zero nibble: qzeros layout is [n_groups, out_dim/8]
        const int qz_col   = col / 8;
        const int qz_bit   = (col % 8) * 4;
        const int zero     = (qzeros[group * (out_dim / 8) + qz_col] >> qz_bit) & 0xF;

        const float w = (nibble - zero) * scale;
        sum += w * __bfloat162float(x[row]);
    }

    // Butterfly warp reduction (matches codebase convention)
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }

    if (threadIdx.x == 0) dst[col] = sum;
}

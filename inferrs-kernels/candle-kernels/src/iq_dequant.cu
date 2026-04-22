// IQ2_XS / IQ3_XXS / IQ4_XS CUDA support derived from llama.cpp ggml-cuda/convert.cu
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include <stdint.h>

#define QK_K 256

#include "ggml_iq_tables_device.inc"

typedef struct {
    half d;
    uint16_t qs[QK_K / 8];
    uint8_t scales[QK_K / 32];
} block_iq2_xs;

typedef struct {
    half d;
    uint8_t qs[3 * QK_K / 8];
} block_iq3_xxs;

typedef struct {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[QK_K / 64];
    uint8_t qs[QK_K / 2];
} block_iq4_xs;

static __device__ __forceinline__ int2 get_int_from_table_16_iq(const int & q4, const int8_t * table) {
    const uint32_t * table32 = (const uint32_t *) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// ---- full-tensor dequantization kernels ----
// Grid=(num_blocks, 1, 1), Block=(32, 1, 1): one warp per quantized block.
// Each thread handles one (il, ib) cell → 8 output elements.

extern "C" __global__ void dequantize_block_iq2xs_f32(const void *__restrict__ vx, float *__restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq2_xs *x = (const block_iq2_xs *)vx;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8;
    const int64_t ib = tid % 8;
    float *y = yy + i * QK_K + 32 * ib + 8 * il;
    const uint16_t *q2 = x[i].qs + 4 * ib;
    const uint8_t *grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d =
        __half2float(x[i].d) * (0.5f + ((x[i].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (float)grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    }
}

extern "C" __global__ void dequantize_block_iq2xs_f16(const void *__restrict__ vx, half *__restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq2_xs *x = (const block_iq2_xs *)vx;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8;
    const int64_t ib = tid % 8;
    half *y = yy + i * QK_K + 32 * ib + 8 * il;
    const uint16_t *q2 = x[i].qs + 4 * ib;
    const uint8_t *grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d =
        __half2float(x[i].d) * (0.5f + ((x[i].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) {
        y[j] = __float2half_rn(d * (float)grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

extern "C" __global__ void dequantize_block_iq3xxs_f32(const void *__restrict__ vx, float *__restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq3_xxs *x = (const block_iq3_xxs *)vx;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8;
    const int64_t ib = tid % 8;
    float *y = yy + i * QK_K + 32 * ib + 8 * il;
    const uint8_t *q3 = x[i].qs + 8 * ib;
    const uint16_t *gas = (const uint16_t *)(x[i].qs + QK_K / 4) + 2 * ib;
    const uint8_t *grid1 = (const uint8_t *)(iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t *grid2 = (const uint8_t *)(iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = __half2float(x[i].d) * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = d * (float)grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
        y[j + 4] = d * (float)grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
    }
}

extern "C" __global__ void dequantize_block_iq3xxs_f16(const void *__restrict__ vx, half *__restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq3_xxs *x = (const block_iq3_xxs *)vx;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8;
    const int64_t ib = tid % 8;
    half *y = yy + i * QK_K + 32 * ib + 8 * il;
    const uint8_t *q3 = x[i].qs + 8 * ib;
    const uint16_t *gas = (const uint16_t *)(x[i].qs + QK_K / 4) + 2 * ib;
    const uint8_t *grid1 = (const uint8_t *)(iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t *grid2 = (const uint8_t *)(iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = __half2float(x[i].d) * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = __float2half_rn(d * (float)grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = __float2half_rn(d * (float)grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}

extern "C" __global__ void dequantize_block_iq4xs_f32(const void *__restrict__ vx, float *__restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq4_xs *x = (const block_iq4_xs *)vx;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8;
    const int64_t ib = tid % 8;
    float *y = yy + i * QK_K + 32 * ib + 8 * il;
    const uint8_t *q4 = x[i].qs + 16 * ib + 4 * il;
    const float d = __half2float(x[i].d) *
        ((((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x[i].scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        const int2 v = get_int_from_table_16_iq(q4[j], kvalues_iq4nl);
        const int8_t *v8 = (const int8_t *)&v;
        y[j + 0] = d * (float)v8[0];
        y[j + 4] = d * (float)v8[1];
    }
}

extern "C" __global__ void dequantize_block_iq4xs_f16(const void *__restrict__ vx, half *__restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq4_xs *x = (const block_iq4_xs *)vx;
    const int64_t tid = threadIdx.x;
    const int64_t il = tid / 8;
    const int64_t ib = tid % 8;
    half *y = yy + i * QK_K + 32 * ib + 8 * il;
    const uint8_t *q4 = x[i].qs + 16 * ib + 4 * il;
    const float d = __half2float(x[i].d) *
        ((((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x[i].scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        const int2 v = get_int_from_table_16_iq(q4[j], kvalues_iq4nl);
        const int8_t *v8 = (const int8_t *)&v;
        y[j + 0] = __float2half_rn(d * (float)v8[0]);
        y[j + 4] = __float2half_rn(d * (float)v8[1]);
    }
}

// ---- fused GEMV: dst[row] = sum_k W[row,k] * y[k] ----
// Butterfly warp reduction (matches codebase convention in quantized.cu).
// All 32 lanes end up with the final sum — we let threadIdx.x == 0 write.
static __device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, mask);
    }
    return v;
}

static __device__ float iq2xs_row_dot(const block_iq2_xs *row_blocks, const float *yy, int nblk) {
    float sum = 0.f;
    const int tid = threadIdx.x;
    const int il = tid / 8;
    const int ibw = tid % 8;
    for (int ib = 0; ib < nblk; ++ib) {
        const block_iq2_xs *xi = &row_blocks[ib];
        const float *yb = yy + (int64_t)ib * QK_K;
        const uint16_t *q2 = xi->qs + 4 * ibw;
        const uint8_t *grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
        const float d =
            __half2float(xi->d) * (0.5f + ((xi->scales[ibw] >> 4 * (il / 2)) & 0xf)) * 0.25f;
        const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
        const int base = 32 * ibw + 8 * il;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float w = d * (float)grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            sum += w * yb[base + j];
        }
    }
    return warp_reduce_sum(sum);
}

static __device__ float iq2xs_row_dot_bf16(const block_iq2_xs *row_blocks, const nv_bfloat16 *yy, int nblk) {
    float sum = 0.f;
    const int tid = threadIdx.x;
    const int il = tid / 8;
    const int ibw = tid % 8;
    for (int ib = 0; ib < nblk; ++ib) {
        const block_iq2_xs *xi = &row_blocks[ib];
        const nv_bfloat16 *yb = yy + (int64_t)ib * QK_K;
        const uint16_t *q2 = xi->qs + 4 * ibw;
        const uint8_t *grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
        const float d =
            __half2float(xi->d) * (0.5f + ((xi->scales[ibw] >> 4 * (il / 2)) & 0xf)) * 0.25f;
        const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
        const int base = 32 * ibw + 8 * il;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float w = d * (float)grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            sum += w * __bfloat162float(yb[base + j]);
        }
    }
    return warp_reduce_sum(sum);
}

static __device__ float iq3xxs_row_dot(const block_iq3_xxs *row_blocks, const float *yy, int nblk) {
    float sum = 0.f;
    const int tid = threadIdx.x;
    const int il = tid / 8;
    const int ibw = tid % 8;
    for (int ib = 0; ib < nblk; ++ib) {
        const block_iq3_xxs *xi = &row_blocks[ib];
        const float *yb = yy + (int64_t)ib * QK_K;
        const uint8_t *q3 = xi->qs + 8 * ibw;
        const uint16_t *gas = (const uint16_t *)(xi->qs + QK_K / 4) + 2 * ibw;
        const uint8_t *grid1 = (const uint8_t *)(iq3xxs_grid + q3[2 * il + 0]);
        const uint8_t *grid2 = (const uint8_t *)(iq3xxs_grid + q3[2 * il + 1]);
        const uint32_t aux32 = gas[0] | (gas[1] << 16);
        const float d = __half2float(xi->d) * (0.5f + (aux32 >> 28)) * 0.5f;
        const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
        const int base = 32 * ibw + 8 * il;
        for (int j = 0; j < 4; ++j) {
            float w0 = d * (float)grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
            float w1 = d * (float)grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            sum += w0 * yb[base + j + 0];
            sum += w1 * yb[base + j + 4];
        }
    }
    return warp_reduce_sum(sum);
}

static __device__ float iq3xxs_row_dot_bf16(const block_iq3_xxs *row_blocks, const nv_bfloat16 *yy, int nblk) {
    float sum = 0.f;
    const int tid = threadIdx.x;
    const int il = tid / 8;
    const int ibw = tid % 8;
    for (int ib = 0; ib < nblk; ++ib) {
        const block_iq3_xxs *xi = &row_blocks[ib];
        const nv_bfloat16 *yb = yy + (int64_t)ib * QK_K;
        const uint8_t *q3 = xi->qs + 8 * ibw;
        const uint16_t *gas = (const uint16_t *)(xi->qs + QK_K / 4) + 2 * ibw;
        const uint8_t *grid1 = (const uint8_t *)(iq3xxs_grid + q3[2 * il + 0]);
        const uint8_t *grid2 = (const uint8_t *)(iq3xxs_grid + q3[2 * il + 1]);
        const uint32_t aux32 = gas[0] | (gas[1] << 16);
        const float d = __half2float(xi->d) * (0.5f + (aux32 >> 28)) * 0.5f;
        const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
        const int base = 32 * ibw + 8 * il;
        for (int j = 0; j < 4; ++j) {
            float w0 = d * (float)grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
            float w1 = d * (float)grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
            sum += w0 * __bfloat162float(yb[base + j + 0]);
            sum += w1 * __bfloat162float(yb[base + j + 4]);
        }
    }
    return warp_reduce_sum(sum);
}

static __device__ float iq4xs_row_dot(const block_iq4_xs *row_blocks, const float *yy, int nblk) {
    float sum = 0.f;
    const int tid = threadIdx.x;
    const int il = tid / 8;
    const int ibw = tid % 8;
    for (int ib = 0; ib < nblk; ++ib) {
        const block_iq4_xs *xi = &row_blocks[ib];
        const float *yb = yy + (int64_t)ib * QK_K;
        const uint8_t *q4 = xi->qs + 16 * ibw + 4 * il;
        const float d = __half2float(xi->d) *
            ((((xi->scales_l[ibw / 2] >> 4 * (ibw % 2)) & 0xf) | (((xi->scales_h >> 2 * ibw) & 3) << 4)) - 32);
        const int base = 32 * ibw + 8 * il;
        for (int j = 0; j < 4; ++j) {
            const int2 v = get_int_from_table_16_iq(q4[j], kvalues_iq4nl);
            const int8_t *v8 = (const int8_t *)&v;
            sum += (d * (float)v8[0]) * yb[base + j + 0];
            sum += (d * (float)v8[1]) * yb[base + j + 4];
        }
    }
    return warp_reduce_sum(sum);
}

static __device__ float iq4xs_row_dot_bf16(const block_iq4_xs *row_blocks, const nv_bfloat16 *yy, int nblk) {
    float sum = 0.f;
    const int tid = threadIdx.x;
    const int il = tid / 8;
    const int ibw = tid % 8;
    for (int ib = 0; ib < nblk; ++ib) {
        const block_iq4_xs *xi = &row_blocks[ib];
        const nv_bfloat16 *yb = yy + (int64_t)ib * QK_K;
        const uint8_t *q4 = xi->qs + 16 * ibw + 4 * il;
        const float d = __half2float(xi->d) *
            ((((xi->scales_l[ibw / 2] >> 4 * (ibw % 2)) & 0xf) | (((xi->scales_h >> 2 * ibw) & 3) << 4)) - 32);
        const int base = 32 * ibw + 8 * il;
        for (int j = 0; j < 4; ++j) {
            const int2 v = get_int_from_table_16_iq(q4[j], kvalues_iq4nl);
            const int8_t *v8 = (const int8_t *)&v;
            sum += (d * (float)v8[0]) * __bfloat162float(yb[base + j + 0]);
            sum += (d * (float)v8[1]) * __bfloat162float(yb[base + j + 4]);
        }
    }
    return warp_reduce_sum(sum);
}

// Launch layout: one output row per (blockIdx.x * blockDim.y + threadIdx.y).
// Grid=(ceil_div(nrows, GGML_CUDA_MMV_Y), 1, 1), Block=(WARP_SIZE=32, GGML_CUDA_MMV_Y=1, 1).

extern "C" __global__ void dequantize_mul_mat_vec_iq2xs_cuda(
    const void *__restrict__ vx, const float *__restrict__ yy, float *__restrict__ dst, int ncols, int nrows) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) return;
    const int nblk = ncols / QK_K;
    const block_iq2_xs *row_blocks = (const block_iq2_xs *)vx + (size_t)row * (size_t)nblk;
    const float dot = iq2xs_row_dot(row_blocks, yy, nblk);
    if (threadIdx.x == 0) dst[row] = dot;
}

extern "C" __global__ void dequantize_mul_mat_vec_iq2xs_bf16in_cuda(
    const void *__restrict__ vx, const nv_bfloat16 *__restrict__ yy, float *__restrict__ dst, int ncols, int nrows) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) return;
    const int nblk = ncols / QK_K;
    const block_iq2_xs *row_blocks = (const block_iq2_xs *)vx + (size_t)row * (size_t)nblk;
    const float dot = iq2xs_row_dot_bf16(row_blocks, yy, nblk);
    if (threadIdx.x == 0) dst[row] = dot;
}

extern "C" __global__ void dequantize_mul_mat_vec_iq3xxs_cuda(
    const void *__restrict__ vx, const float *__restrict__ yy, float *__restrict__ dst, int ncols, int nrows) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) return;
    const int nblk = ncols / QK_K;
    const block_iq3_xxs *row_blocks = (const block_iq3_xxs *)vx + (size_t)row * (size_t)nblk;
    const float dot = iq3xxs_row_dot(row_blocks, yy, nblk);
    if (threadIdx.x == 0) dst[row] = dot;
}

extern "C" __global__ void dequantize_mul_mat_vec_iq3xxs_bf16in_cuda(
    const void *__restrict__ vx, const nv_bfloat16 *__restrict__ yy, float *__restrict__ dst, int ncols, int nrows) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) return;
    const int nblk = ncols / QK_K;
    const block_iq3_xxs *row_blocks = (const block_iq3_xxs *)vx + (size_t)row * (size_t)nblk;
    const float dot = iq3xxs_row_dot_bf16(row_blocks, yy, nblk);
    if (threadIdx.x == 0) dst[row] = dot;
}

extern "C" __global__ void dequantize_mul_mat_vec_iq4xs_cuda(
    const void *__restrict__ vx, const float *__restrict__ yy, float *__restrict__ dst, int ncols, int nrows) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) return;
    const int nblk = ncols / QK_K;
    const block_iq4_xs *row_blocks = (const block_iq4_xs *)vx + (size_t)row * (size_t)nblk;
    const float dot = iq4xs_row_dot(row_blocks, yy, nblk);
    if (threadIdx.x == 0) dst[row] = dot;
}

extern "C" __global__ void dequantize_mul_mat_vec_iq4xs_bf16in_cuda(
    const void *__restrict__ vx, const nv_bfloat16 *__restrict__ yy, float *__restrict__ dst, int ncols, int nrows) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows) return;
    const int nblk = ncols / QK_K;
    const block_iq4_xs *row_blocks = (const block_iq4_xs *)vx + (size_t)row * (size_t)nblk;
    const float dot = iq4xs_row_dot_bf16(row_blocks, yy, nblk);
    if (threadIdx.x == 0) dst[row] = dot;
}

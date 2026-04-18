// Flash Attention decode kernel for GQA (Grouped Query Attention).
//
// Flash Decoding: single-token query attending to a KV cache.
// Supports GQA where n_kv_groups Q heads share one KV head.
//
// Tensor layout (candle convention, transposed from standard):
//   Q:   [1, n_q_heads, 1, head_dim]   - BF16, stride [n_q*head_dim, head_dim, head_dim, 1]
//   K:   [1, n_kv_heads, kv_len, head_dim] - BF16
//   V:   [1, n_kv_heads, kv_len, head_dim] - BF16
//   Out: [1, n_kv_heads*n_kv_groups, head_dim] - F32 (GQA output, reshaped by caller)
//
// Grid:  (n_kv_heads * n_kv_groups, 1, 1)  — one block per Q head
// Block: (D, 1, 1) — one thread per output dimension

#include "cuda_bf16.h"
#include <stdint.h>
#include <float.h>

#define FA_TILE 32

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

template <int D>
static __device__ void flash_attn_decode_clean(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    float* __restrict__ out,
    int n_kv_groups,
    int kv_len,
    float scale
) {
    const int q_head  = blockIdx.x;
    const int kv_head = q_head / n_kv_groups;
    const int d       = threadIdx.x;

    const __nv_bfloat16* q_ptr = Q + (size_t)q_head * D;
    const __nv_bfloat16* k_ptr = K + (size_t)kv_head * kv_len * D;
    const __nv_bfloat16* v_ptr = V + (size_t)kv_head * kv_len * D;

    float q_val = bf16_to_f32(q_ptr[d]);

    // Shared memory for warp sums and scores.
    __shared__ float smem_scores[FA_TILE];  // scores[j]
    extern __shared__ float smem_warp[];    // warp sums: (D/32) floats

    float acc   = 0.0f;
    float m_old = -FLT_MAX;
    float l_old = 0.0f;

    const int n_warps = (D + 31) / 32;

    for (int tile_start = 0; tile_start < kv_len; tile_start += FA_TILE) {
        int tile_end  = min(tile_start + FA_TILE, kv_len);
        int tile_size = tile_end - tile_start;

        // Compute scores[j] = (Q · K[j]) * scale  for j in tile.
        for (int j = 0; j < tile_size; j++) {
            int pos = tile_start + j;
            float k_val = bf16_to_f32(k_ptr[(size_t)pos * D + d]);
            float partial = q_val * k_val;

            // Step 1: warp reduce within each warp.
            float warp_sum = warp_reduce_sum(partial);
            // Step 2: warp lane 0 writes to shared memory.
            if ((d & 31) == 0) smem_warp[d / 32] = warp_sum;
            __syncthreads();
            // Step 3: thread 0 reduces warp sums.
            if (d == 0) {
                float total = 0.0f;
                for (int w = 0; w < n_warps; w++) total += smem_warp[w];
                smem_scores[j] = total * scale;
            }
            __syncthreads();
        }

        // Online softmax update.
        float m_new = m_old;
        for (int j = 0; j < tile_size; j++) m_new = fmaxf(m_new, smem_scores[j]);

        float exp_diff = __expf(m_old - m_new);
        float l_new = l_old * exp_diff;
        acc = acc * exp_diff;

        for (int j = 0; j < tile_size; j++) {
            float e = __expf(smem_scores[j] - m_new);
            l_new += e;
            acc += e * bf16_to_f32(v_ptr[(size_t)(tile_start + j) * D + d]);
        }

        m_old = m_new;
        l_old = l_new;
        __syncthreads();
    }

    out[q_head * D + d] = (l_old > 0.0f) ? acc / l_old : 0.0f;
}

// Kernel wrappers: one per supported head dimension.
// Each kernel is a separate global function (no template in extern "C").
// Dynamic shared memory size = (D/32) * sizeof(float) for warp sums.

#define DEF_FA_KERNEL(D_VAL)                                                    \
extern "C" __global__                                                           \
__launch_bounds__(D_VAL)                                                        \
void flash_attn_decode_bf16_d##D_VAL(                                           \
    const __nv_bfloat16* Q,                                                     \
    const __nv_bfloat16* K,                                                     \
    const __nv_bfloat16* V,                                                     \
    float* out,                                                                 \
    int n_kv_groups,                                                            \
    int kv_len,                                                                 \
    float scale                                                                 \
) {                                                                             \
    flash_attn_decode_clean<D_VAL>(Q, K, V, out, n_kv_groups, kv_len, scale);  \
}

DEF_FA_KERNEL(64)
DEF_FA_KERNEL(128)
DEF_FA_KERNEL(256)
DEF_FA_KERNEL(512)

// ── Flash Attention prefill kernel ───────────────────────────────────────────
//
// Handles t > 1 (prefill / chunked-prefill). Causal, GQA.
//
// Tensor layout (contiguous, head-major):
//   Q:   [batch * n_q_heads,  q_len,  D]  BF16
//   K:   [batch * n_kv_heads, kv_len, D]  BF16
//   V:   [batch * n_kv_heads, kv_len, D]  BF16
//   out: [batch * n_q_heads,  q_len,  D]  BF16
//
// Grid:  (ceil(q_len / Br), batch * n_q_heads, 1)
// Block: (D, 1, 1)
//
// SMEM:
//   Static  ~42 KB: q/k/v tiles, scores, m/l/rsc  (D=256, Br=16, Bk=32)
//   Dynamic  16 KB: warp_sums_dyn[Br][Bk][n_warps]
//   Combined ~58 KB — requires PREFERRED_SHARED_MEMORY_CARVEOUT=100 and
//             MAX_DYNAMIC_SHARED_SIZE_BYTES=96KB set on the function before launch.
//
// Sync budget per KV tile: 5 (vs 35 in the per-row-serial design).
//   P2 fix: all Br rows' warp sums are written atomically before one cross-warp
//   reduction sync, eliminating the 2-sync-per-row structure (2×Br → 1 sync).
//   P3 fix: score reduction is parallel across D threads (2 scores/thread for
//   D=256, Br=16, Bk=32); softmax runs Br=16 threads in parallel.
//
// __expf is ~10-bit mantissa (fast approximation). Acceptable here because all
// exp outputs are immediately renormalised by l. Switch to expf() if precision
// tests show divergence vs the naive path.
template <int D, int Br, int Bk>
static __device__ void flash_attn_prefill_bf16_impl(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ out,
    int n_kv_groups,
    int q_len,
    int kv_len,
    int seqlen_offset,
    float scale
) {
    const int q_tile  = blockIdx.x;
    const int head    = blockIdx.y;
    const int d       = threadIdx.x;
    const int kv_head = head / n_kv_groups;
    const int q_start = q_tile * Br;
    constexpr int n_warps = D / 32;
    const int warp_id = d / 32;

    // Static SMEM: Q/K/V tiles + per-tile score matrix + softmax stats.
    __shared__ __nv_bfloat16 q_smem[Br][D];
    __shared__ __nv_bfloat16 k_smem[Bk][D];
    __shared__ __nv_bfloat16 v_smem[Bk][D];
    __shared__ float scores[Br][Bk];
    __shared__ float m_smem[Br];
    __shared__ float l_smem[Br];
    __shared__ float rsc_smem[Br];

    // Dynamic SMEM: per-row warp partial sums [Br][Bk][n_warps].
    // Keeping this dynamic avoids pushing static SMEM above 48 KB.
    // Flat index: i * Bk * n_warps + j * n_warps + w.
    extern __shared__ float warp_sums_dyn[];

    // Load Q tile (each thread loads its dim d for all Br rows).
    #pragma unroll
    for (int i = 0; i < Br; i++) {
        int q_pos = q_start + i;
        q_smem[i][d] = (q_pos < q_len)
            ? Q[(size_t)head * q_len * D + (size_t)q_pos * D + d]
            : __float2bfloat16(0.f);
    }
    if (d < Br) {
        m_smem[d] = -FLT_MAX;
        l_smem[d] = 0.f;
    }
    __syncthreads();  // SYNC 0: Q tile + m/l ready

    // Per-thread output accumulators (registers, one float per Q row).
    float acc[Br];
    #pragma unroll
    for (int i = 0; i < Br; i++) acc[i] = 0.f;

    for (int kv_s = 0; kv_s < kv_len; kv_s += Bk) {

        // ── SYNC 1: load K and V tiles ────────────────────────────────────────
        #pragma unroll
        for (int j = 0; j < Bk; j++) {
            int kv_pos = kv_s + j;
            k_smem[j][d] = (kv_pos < kv_len)
                ? K[(size_t)kv_head * kv_len * D + (size_t)kv_pos * D + d]
                : __float2bfloat16(0.f);
            v_smem[j][d] = (kv_pos < kv_len)
                ? V[(size_t)kv_head * kv_len * D + (size_t)kv_pos * D + d]
                : __float2bfloat16(0.f);
        }
        __syncthreads();  // SYNC 1

        // ── SYNC 2: compute all Br × Bk warp partial sums in one pass ─────────
        // No sync between Q rows: each row writes to its own warp_sums_dyn[i]
        // slice, eliminating the per-row race condition that forced 2 syncs/row.
        #pragma unroll
        for (int i = 0; i < Br; i++) {
            float q_val = __bfloat162float(q_smem[i][d]);
            #pragma unroll
            for (int j = 0; j < Bk; j++) {
                float p = q_val * __bfloat162float(k_smem[j][d]);
                p = warp_reduce_sum(p);
                if ((d & 31) == 0)
                    warp_sums_dyn[i * Bk * n_warps + j * n_warps + warp_id] = p;
            }
        }
        __syncthreads();  // SYNC 2: all warp_sums_dyn entries written

        // ── SYNC 3: parallel score reduction ──────────────────────────────────
        // Thread d reduces (Br*Bk/D) scores, covering every (i,j) pair exactly.
        // With Br=16, Bk=32, D=256: each thread handles 2 scores (rows d/32 and
        // d/32+8; col = d%32), so all 256 threads work and none idles.
        #pragma unroll
        for (int k = 0; k < Br * Bk / D; k++) {
            const int row = d / Bk + k * (D / Bk);  // d/32 + k*8  for D=256, Bk=32
            const int col = d % Bk;                  // d%32
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < n_warps; w++)
                s += warp_sums_dyn[row * Bk * n_warps + col * n_warps + w];
            s *= scale;
            const int abs_kv = kv_s + col;
            if (abs_kv >= kv_len || abs_kv > seqlen_offset + q_start + row)
                s = -FLT_MAX;
            scores[row][col] = s;
        }
        __syncthreads();  // SYNC 3: all scores[Br][Bk] written

        // ── SYNC 4: parallel softmax — thread i owns row i ────────────────────
        // Br threads execute in parallel; threads Br..D-1 skip this section.
        if (d < Br) {
            float m_new = m_smem[d];
            #pragma unroll
            for (int j = 0; j < Bk; j++) m_new = fmaxf(m_new, scores[d][j]);
            const float rsc = __expf(m_smem[d] - m_new);
            float sum_e = 0.f;
            #pragma unroll
            for (int j = 0; j < Bk; j++) sum_e += __expf(scores[d][j] - m_new);
            l_smem[d]   = l_smem[d] * rsc + sum_e;
            m_smem[d]   = m_new;
            rsc_smem[d] = rsc;
        }
        __syncthreads();  // SYNC 4: m / l / rsc broadcast to all threads

        // ── SYNC 5: accumulate O += exp(S - m) * V ────────────────────────────
        #pragma unroll
        for (int i = 0; i < Br; i++) {
            acc[i] *= rsc_smem[i];
            const float m_i = m_smem[i];
            #pragma unroll
            for (int j = 0; j < Bk; j++) {
                if (kv_s + j < kv_len)
                    acc[i] += __expf(scores[i][j] - m_i) * __bfloat162float(v_smem[j][d]);
            }
        }
        __syncthreads();  // SYNC 5: before k_smem / v_smem reload
    }

    // Normalize and write BF16 output.
    #pragma unroll
    for (int i = 0; i < Br; i++) {
        const int q_pos = q_start + i;
        if (q_pos < q_len) {
            const float l = l_smem[i];
            out[(size_t)head * q_len * D + (size_t)q_pos * D + d] =
                __float2bfloat16(l > 0.f ? acc[i] / l : 0.f);
        }
    }
}

#define DEF_FA_PREFILL_KERNEL(D_VAL)                                              \
extern "C" __global__                                                             \
__launch_bounds__(D_VAL)                                                          \
void flash_attn_prefill_bf16_d##D_VAL(                                            \
    const __nv_bfloat16* Q,                                                       \
    const __nv_bfloat16* K,                                                       \
    const __nv_bfloat16* V,                                                       \
    __nv_bfloat16* out,                                                           \
    int n_kv_groups,                                                              \
    int q_len,                                                                    \
    int kv_len,                                                                   \
    int seqlen_offset,                                                            \
    float scale                                                                   \
) {                                                                               \
    flash_attn_prefill_bf16_impl<D_VAL, 16, 32>(                                  \
        Q, K, V, out, n_kv_groups, q_len, kv_len, seqlen_offset, scale);          \
}

DEF_FA_PREFILL_KERNEL(128)
DEF_FA_PREFILL_KERNEL(256)

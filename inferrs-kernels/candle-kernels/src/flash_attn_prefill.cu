// FlashAttention-2 prefill kernel: BF16 Q/K/V, F32 accumulators, causal, GQA.
//
// SM80+ required (BF16 native arithmetic; no WMMA — accumulation via cuda cores).
//
// Layout (candle convention): [batch=1, heads, seq, head_dim]
//   Q:   [1, n_q_heads,  q_len,  D]  BF16
//   K:   [1, n_kv_heads, kv_len, D]  BF16
//   V:   [1, n_kv_heads, kv_len, D]  BF16
//   out: [1, n_q_heads,  q_len,  D]  F32  (caller casts to BF16)
//
// Grid:  (ceil(q_len / 32), n_kv_heads * ceil(gqa_factor / CHUNK), 1)
// Block: (BLOCK_THREADS=1024, 1, 1)   — 32 warps, 1 warp = 1 Q-row.
//
// Each warp walks its Q-row against the K/V tiles; within that walk, the warp
// iterates sequentially over CHUNK Q-heads that share the same KV-head. The
// K/V tile is therefore loaded once per (block, kv-tile) and reused for
// 32 × CHUNK dot-products → minimal K/V global-memory traffic.
//
// BK: K/V tile width (32 for D=128, 16 for D=256) — keeps smem(K+V) = 16 KB.
// CHUNK: GQA heads multiplexed per block.
//        CHUNK=4 for D=128 (≈50 regs/thread), CHUNK=2 for D=256 (≈46 regs).
//
// kv_offset = kv_len - q_len: absolute position of Q-row 0 in the full sequence.
// Causal mask: k_pos visible to q_row iff k_pos <= q_row + kv_offset.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "flash_attn_prefill requires SM80+ (Ampere / Ada, RTX 30xx+)"
#endif

#include "cuda_bf16.h"
#include <float.h>
#include <stdint.h>

#define BQ              32   // Q-rows per block (= warps per block)
#define BLOCK_THREADS   1024 // 32 warps × 32 lanes

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// Core implementation, templated on head_dim D, KV-tile width BK, and CHUNK
// (GQA Q-heads multiplexed per block).
template <int D, int BK, int CHUNK>
static __device__ void flash_attn_prefill_impl(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    float*               __restrict__ out,
    int gqa_factor,
    int n_kv_heads,
    int q_len,
    int kv_len,
    int kv_offset,
    float scale
) {
    // D-elements owned by each lane within its warp.
    // 32 lanes per warp: D=128 → 4, D=256 → 8.
    constexpr int EPT = D / 32;

    // Elements each thread loads per K (or V) tile pass.
    // BK*D total / 1024 threads: D=128,BK=32 → 4; D=256,BK=16 → 4.
    constexpr int LOAD_EPT = BK * D / BLOCK_THREADS;

    // Block → (kv_head, chunk_idx) mapping. chunks_per_kv covers the whole
    // GQA group even when gqa_factor is not a multiple of CHUNK (last chunk
    // runs partial via h_count).
    const int chunks_per_kv = (gqa_factor + CHUNK - 1) / CHUNK;
    const int kv_head       = blockIdx.y / chunks_per_kv;
    const int chunk_idx     = blockIdx.y % chunks_per_kv;
    const int h_start       = kv_head * gqa_factor + chunk_idx * CHUNK;
    const int h_remaining   = gqa_factor - chunk_idx * CHUNK;
    const int h_count       = (h_remaining < CHUNK) ? h_remaining : CHUNK;

    const int q_tile_start = blockIdx.x * BQ;
    const int warp_id      = threadIdx.x / 32;   // [0, 32)
    const int lane_id      = threadIdx.x % 32;   // [0, 32)

    const int  q_row   = q_tile_start + warp_id;
    const bool q_valid = (q_row < q_len);

    // K/V global pointers: one KV slab per kv_head.
    const __nv_bfloat16* k_base = K + kv_head * kv_len * D;
    const __nv_bfloat16* v_base = V + kv_head * kv_len * D;

    // Shared memory: K_tile[BK][D] + V_tile[BK][D], both BF16 → 16 KB.
    extern __shared__ char smem_bytes[];
    __nv_bfloat16* K_tile = reinterpret_cast<__nv_bfloat16*>(smem_bytes);
    __nv_bfloat16* V_tile = K_tile + BK * D;

    // Per-thread registers: one Q / acc set per head in the CHUNK.
    float q_reg [CHUNK][EPT];
    float acc   [CHUNK][EPT];
    float m_val [CHUNK];
    float l_val [CHUNK];

#pragma unroll
    for (int h = 0; h < CHUNK; h++) {
        m_val[h] = -FLT_MAX;
        l_val[h] = 0.0f;
#pragma unroll
        for (int i = 0; i < EPT; i++) {
            acc  [h][i] = 0.0f;
            q_reg[h][i] = 0.0f;
        }
    }

    // Load Q for all CHUNK heads once, reused across every KV tile.
    if (q_valid) {
#pragma unroll
        for (int h = 0; h < CHUNK; h++) {
            if (h < h_count) {
                const int q_head = h_start + h;
                const __nv_bfloat16* q_ptr = Q + (q_head * q_len + q_row) * D;
#pragma unroll
                for (int i = 0; i < EPT; i++) {
                    q_reg[h][i] = bf16_to_f32(q_ptr[lane_id * EPT + i]);
                }
            }
        }
    }

    // KV tile loop
    for (int k_start = 0; k_start < kv_len; k_start += BK) {
        const int tile_size = min(BK, kv_len - k_start);

        // Collaborative coalesced load of K and V tiles.
        // Thread t loads LOAD_EPT contiguous elements starting at t*LOAD_EPT.
        const int base = threadIdx.x * LOAD_EPT;
#pragma unroll
        for (int i = 0; i < LOAD_EPT; i++) {
            int idx = base + i;           // flat index into [BK][D]
            int row = idx / D;            // row within tile
            int col = idx % D;
            int gpos = k_start + row;     // absolute KV position
            __nv_bfloat16 zero = __float2bfloat16(0.0f);
            K_tile[idx] = (row < tile_size) ? k_base[gpos * D + col] : zero;
            V_tile[idx] = (row < tile_size) ? v_base[gpos * D + col] : zero;
        }
        __syncthreads();

        // Per-warp online softmax + V-accumulation over the KV tile.
        // For each K position, update CHUNK independent softmax states.
        if (q_valid) {
#pragma unroll 4
            for (int kj = 0; kj < BK; kj++) {
                if (kj >= tile_size) break;

                const int k_pos   = k_start + kj;
                const bool masked = (k_pos > q_row + kv_offset);

                // K / V lane-local values: read once per kj, reused across CHUNK.
                float k_local[EPT];
                float v_local[EPT];
#pragma unroll
                for (int i = 0; i < EPT; i++) {
                    k_local[i] = bf16_to_f32(K_tile[kj * D + lane_id * EPT + i]);
                    v_local[i] = bf16_to_f32(V_tile[kj * D + lane_id * EPT + i]);
                }

#pragma unroll
                for (int h = 0; h < CHUNK; h++) {
                    if (h >= h_count) break;

                    // Partial dot product: each lane contributes EPT terms.
                    float partial = 0.0f;
                    if (!masked) {
#pragma unroll
                        for (int i = 0; i < EPT; i++) {
                            partial += q_reg[h][i] * k_local[i];
                        }
                    }
                    float score = warp_reduce_sum(partial) * scale;
                    if (masked) score = -FLT_MAX;

                    // Online softmax update (FA-2 recipe, in registers).
                    // Skip masked positions entirely: when score == -FLT_MAX and
                    // m_val == -FLT_MAX, exp(score-m_new)=exp(0)=1 would incorrectly
                    // accumulate into l_val. After the first valid score m_val is
                    // a real value and exp(-FLT_MAX - real) ≈ 0 anyway, but the
                    // branch makes correctness unconditional.
                    if (score > -FLT_MAX) {
                        const float m_new = fmaxf(m_val[h], score);
                        const float alpha = __expf(m_val[h] - m_new);
                        const float p     = __expf(score - m_new);
                        l_val[h] = l_val[h] * alpha + p;
#pragma unroll
                        for (int i = 0; i < EPT; i++) {
                            acc[h][i] = acc[h][i] * alpha + p * v_local[i];
                        }
                        m_val[h] = m_new;
                    }
                }
            }
        }
        __syncthreads();  // guard before next tile load
    }

    // Write normalised output for each Q-head in the CHUNK.
    if (q_valid) {
#pragma unroll
        for (int h = 0; h < CHUNK; h++) {
            if (h >= h_count) break;
            if (!(l_val[h] > 0.0f)) continue;
            const int q_head = h_start + h;
            float* out_ptr = out + (q_head * q_len + q_row) * D;
            const float inv_l = 1.0f / l_val[h];
#pragma unroll
            for (int i = 0; i < EPT; i++) {
                out_ptr[lane_id * EPT + i] = acc[h][i] * inv_l;
            }
        }
    }

    (void)n_kv_heads;  // kept in ABI for future bounds checks / debug
}

// Kernel wrappers: one per supported (D, BK, CHUNK) triple.
// shared_mem_bytes = 2 * BK * D * sizeof(bf16):
//   D=128, BK=32 → 16 384 B  (16 KB)
//   D=256, BK=16 → 16 384 B  (16 KB)
//
// Grid:  (ceil(q_len / 32), n_kv_heads * ceil(gqa_factor / CHUNK), 1).
// Block: (BLOCK_THREADS=1024, 1, 1).

#define DEF_FA_PREFILL_KERNEL(D_VAL, BK_VAL, CHUNK_VAL)                         \
extern "C" __global__                                                            \
__launch_bounds__(BLOCK_THREADS)                                                 \
void flash_attn_prefill_bf16_d##D_VAL(                                          \
    const __nv_bfloat16* Q,                                                      \
    const __nv_bfloat16* K,                                                      \
    const __nv_bfloat16* V,                                                      \
    float*               out,                                                    \
    int gqa_factor,                                                              \
    int n_kv_heads,                                                              \
    int q_len,                                                                   \
    int kv_len,                                                                  \
    int kv_offset,                                                               \
    float scale                                                                  \
) {                                                                              \
    flash_attn_prefill_impl<D_VAL, BK_VAL, CHUNK_VAL>(                           \
        Q, K, V, out, gqa_factor, n_kv_heads,                                    \
        q_len, kv_len, kv_offset, scale);                                        \
}

DEF_FA_PREFILL_KERNEL(128, 32, 4)   // Qwen3 all sizes
DEF_FA_PREFILL_KERNEL(256, 16, 2)   // Qwen3.5 all sizes

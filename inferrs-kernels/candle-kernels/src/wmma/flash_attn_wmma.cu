// Flash Attention prefill — WMMA + Q-from-global. Requires SM 80+ (Ampere).
//
// Compiled separately with -arch=compute_80 to isolate sm_80-only PTX from the
// main flash_attn module: on SM < 80, cuModuleLoadData would fail for the entire
// module containing WMMA instructions, breaking the existing decode kernels too.
//
// Differences vs flash_attn_prefill_bf16_impl:
//   - q_smem[Br][D] added back as q_stage (8 KB) for partial last tile only;
//     complete tiles still load Q directly from global (L2-hot after first KV tile).
//   - SYNC 2 replaced by WMMA mma_sync: 4 ops/warp/tile vs 512 FMA+shuffles.
//   - SMEM D=256: ~42 KB static + 16 KB dynamic = 58 KB → 1 block/SM on RTX 3090.
//     (was 50 KB / 2 blocks before q_stage was added for OOB safety)
//   - __launch_bounds__(D): register hint without minBlocks (2-block occupancy lost).
//
// cp.async for K/V is NOT used here: __pipeline_memcpy_async requires 4+ byte
// transfers; BF16 (2 bytes) does not qualify. cg::memcpy_async would require
// zero-padding the last partial tile (extra branching). Benefit is marginal vs
// the occupancy + WMMA gains already achieved.

#include "cuda_bf16.h"
#include <float.h>
#include <mma.h>
using namespace nvcuda;

template <int D, int Br, int Bk>
static __device__ void flash_attn_prefill_wmma_bf16_impl(
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
    // K_TILES_PER_WARP = (D/16) / n_warps = (D/16) / (D/32) = 2, always.
    constexpr int K_TILES_PER_WARP = 2;

    // Static SMEM: K/V tiles + score matrix + softmax stats + Q staging buffer.
    __shared__ __nv_bfloat16 k_smem[Bk][D];
    __shared__ __nv_bfloat16 v_smem[Bk][D];
    __shared__ float scores[Br][Bk];
    __shared__ float m_smem[Br];
    __shared__ float l_smem[Br];
    __shared__ float rsc_smem[Br];
    // q_stage: used only for the last (partial) Q-tile to zero-pad OOB rows before
    // wmma::load_matrix_sync, avoiding undefined behaviour on global OOB reads.
    __shared__ __nv_bfloat16 q_stage[Br][D];

    // Dynamic SMEM: WMMA partial results, layout [n_warps][Br][Bk].
    // Each warp writes its partial Q×K^T accumulation here before SYNC 3 reduces.
    extern __shared__ float warp_sums_dyn[];

    if (d < Br) {
        m_smem[d] = -FLT_MAX;
        l_smem[d] = 0.f;
    }

    float acc[Br];
    #pragma unroll
    for (int i = 0; i < Br; i++) acc[i] = 0.f;

    // Populate q_stage for the last partial tile (q_start + Br > q_len).
    // Thread d fills all Br rows of column d with bounds-checked Q values.
    // Complete tiles (common case) load directly from global inside the KV loop.
    const bool partial_tile = (q_start + Br > q_len);
    if (partial_tile) {
        #pragma unroll
        for (int i = 0; i < Br; i++) {
            const int q_pos = q_start + i;
            q_stage[i][d] = (q_pos < q_len)
                ? Q[(size_t)head * q_len * D + (size_t)q_pos * D + d]
                : __float2bfloat16(0.f);
        }
    }
    __syncthreads();  // SYNC 0: q_stage ready (if partial_tile) before KV loop

    for (int kv_s = 0; kv_s < kv_len; kv_s += Bk) {

        // ── SYNC 1: load K and V tiles ────────────────────────────────────────
        // Out-of-bounds positions are zeroed; SYNC 3 masks their scores to -inf.
        #pragma unroll
        for (int j = 0; j < Bk; j++) {
            const int kv_pos = kv_s + j;
            k_smem[j][d] = (kv_pos < kv_len)
                ? K[(size_t)kv_head * kv_len * D + (size_t)kv_pos * D + d]
                : __float2bfloat16(0.f);
            v_smem[j][d] = (kv_pos < kv_len)
                ? V[(size_t)kv_head * kv_len * D + (size_t)kv_pos * D + d]
                : __float2bfloat16(0.f);
        }
        __syncthreads();  // SYNC 1

        // ── SYNC 2: WMMA Q×K^T → warp_sums_dyn[warp][Br][Bk] ──────────────
        //
        // Layout trick (K^T without transposing):
        //   A (Q):   q_stage[Br][D] (partial tile) or Q global (complete tile), row_major, stride D
        //            → load_matrix_sync(q_frag, q_src, D) : matrix_a row_major
        //   B (K^T): k_smem[Bk][D] row_major stored with stride D
        //            → k_smem[n_off][k_off] col_major, stride D gives K^T correctly
        //              (ptr[col*D+row] = k_smem[n_off+col][k_off+row] = K^T[k+row][n+col])
        //
        // Warp partitioning: each warp handles K_TILES_PER_WARP=2 K-tiles of D,
        // covering 2×16=32 consecutive dimensions. n_warps=8 (D=256) or 4 (D=128)
        // warps cover all D dimensions together.
        {
            wmma::fragment<wmma::matrix_a,    16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b,    16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[Bk / 16];
            #pragma unroll
            for (int n = 0; n < Bk / 16; n++)
                wmma::fill_fragment(c_frag[n], 0.f);

            #pragma unroll
            for (int kk = 0; kk < K_TILES_PER_WARP; kk++) {
                const int k_off = (warp_id * K_TILES_PER_WARP + kk) * 16;
                // Complete tiles: load Q directly from global (L2-hot, fast path).
                // Partial last tile: load from q_stage (zero-padded, set in SYNC 0).
                const __nv_bfloat16* q_src = partial_tile
                    ? &q_stage[0][k_off]
                    : &Q[(size_t)head * q_len * D + (size_t)q_start * D + k_off];
                wmma::load_matrix_sync(q_frag, q_src, D);
                #pragma unroll
                for (int n = 0; n < Bk / 16; n++) {
                    const int n_off = n * 16;
                    wmma::load_matrix_sync(k_frag, &k_smem[n_off][k_off], D);
                    wmma::mma_sync(c_frag[n], q_frag, k_frag, c_frag[n]);
                }
            }
            // Store partial [Br][Bk] result into warp's slice of warp_sums_dyn.
            // Row stride = Bk so element [i][j] is at warp_id*Br*Bk + i*Bk + j.
            #pragma unroll
            for (int n = 0; n < Bk / 16; n++) {
                wmma::store_matrix_sync(
                    warp_sums_dyn + warp_id * Br * Bk + n * 16,
                    c_frag[n], Bk, wmma::mem_row_major);
            }
        }
        __syncthreads();  // SYNC 2: all warp_sums_dyn written

        // ── SYNC 3: parallel score reduction (warp-first layout) ─────────────
        // warp_sums_dyn layout [n_warps][Br][Bk]: each entry [w][row][col] holds
        // warp w's partial dot-product for Q row `row` against K^T column `col`.
        // Thread d reduces Br*Bk/D scores in parallel (2 scores for D=256, Bk=32).
        #pragma unroll
        for (int k = 0; k < Br * Bk / D; k++) {
            const int row = d / Bk + k * (D / Bk);  // d/32 + k*8 for D=256,Bk=32
            const int col = d % Bk;
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < n_warps; w++)
                s += warp_sums_dyn[w * Br * Bk + row * Bk + col];
            s *= scale;
            const int abs_kv = kv_s + col;
            if (abs_kv >= kv_len || abs_kv > seqlen_offset + q_start + row)
                s = -FLT_MAX;
            scores[row][col] = s;
        }
        __syncthreads();  // SYNC 3

        // ── SYNC 4: parallel softmax (threads 0..Br-1) ───────────────────────
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
        __syncthreads();  // SYNC 4

        // ── SYNC 5: accumulate O += exp(S-m)*V ───────────────────────────────
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
        __syncthreads();  // SYNC 5: before next k/v_smem load
    }

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

#define DEF_FA_PREFILL_WMMA_KERNEL(D_VAL)                                              \
extern "C" __global__                                                                  \
__launch_bounds__(D_VAL)                                                               \
void flash_attn_prefill_wmma_bf16_d##D_VAL(                                            \
    const __nv_bfloat16* Q,                                                            \
    const __nv_bfloat16* K,                                                            \
    const __nv_bfloat16* V,                                                            \
    __nv_bfloat16* out,                                                                \
    int n_kv_groups,                                                                   \
    int q_len,                                                                         \
    int kv_len,                                                                        \
    int seqlen_offset,                                                                 \
    float scale                                                                        \
) {                                                                                    \
    flash_attn_prefill_wmma_bf16_impl<D_VAL, 16, 32>(                                  \
        Q, K, V, out, n_kv_groups, q_len, kv_len, seqlen_offset, scale);               \
}

DEF_FA_PREFILL_WMMA_KERNEL(128)
DEF_FA_PREFILL_WMMA_KERNEL(256)

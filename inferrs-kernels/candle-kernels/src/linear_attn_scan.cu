// 3-kernel FLA-style GatedDeltaNet chunked scan (prefill).
//
// Replaces the monolithic per-(batch,head) kernel with three specialised kernels
// that expose C-level parallelism to the GPU scheduler:
//
//   K1  linear_attn_intra   grid(B*NH*C)  — KKT + fwd-subst + WY per chunk
//   K2  linear_attn_state   grid(B*NH)    — sequential state scan, state in regs
//   K3  linear_attn_output  grid(B*NH*C)  — tiled qk + matmul per chunk
//
// Intermediate buffers (all F32, allocated by Rust caller):
//   w    [B*NH*C, S, HK]
//   u    [B*NH*C, S, HV]
//   gc   [B*NH*C, S]
//   inter[B*NH*C, S, HV]   (q_exp @ state snapshot, computed in K2)
//   vnew [B*NH*C, S, HV]   (u − w @ state, computed in K2)
//
// Public entry points follow the naming convention:
//   linear_attn_intra_{f32|bf16}_hk{HK}_hv{HV}
//   linear_attn_state_{f32|bf16}_hk{HK}_hv{HV}
//   linear_attn_output_{f32|bf16}_hk{HK}_hv{HV}

#include <stdint.h>
#include <float.h>
#include <cuda_bf16.h>

// ── Type helpers ──────────────────────────────────────────────────────────────

template<typename T>
__device__ __forceinline__ float load_as_f32(const T* ptr, int i);

template<>
__device__ __forceinline__ float load_as_f32<float>(const float* ptr, int i) {
    return ptr[i];
}

template<>
__device__ __forceinline__ float load_as_f32<__nv_bfloat16>(const __nv_bfloat16* ptr, int i) {
    return __bfloat162float(ptr[i]);
}

// Block-wide inclusive prefix sum in smem[0..S).
// All threads must call — no guard at call site.
__device__ __forceinline__ void prefix_sum_inplace(float* smem, int tid, int S) {
    float v = (tid < S) ? smem[tid] : 0.0f;
    __syncthreads();
    for (int step = 1; step < S; step <<= 1) {
        float prev = (tid >= step && tid < S) ? smem[tid - step] : 0.0f;
        __syncthreads();
        v += prev;
        if (tid < S) smem[tid] = v;
        __syncthreads();
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// K1 — linear_attn_intra
// Grid : (B*NH*C, 1, 1)   Block : (256, 1, 1)
//
// Inputs  (slice for ONE chunk):
//   q_ci, k_ci, v_ci : [S, HK/HV]  dtype T
//   log_g_ci, beta_ci : [S]         F32
// Outputs (slice for ONE chunk):
//   w_ci  : [S, HK]  F32   = (I−a_mat)^{-1} @ (k*beta*exp(gc))
//   u_ci  : [S, HV]  F32   = (I−a_mat)^{-1} @ (v*beta)
//   gc_ci : [S]      F32   = inclusive prefix sum of log_g
//
// Shared memory layout (phased, peak ~49 KB for HK=HV=128):
//   s_attn [S*S]  16 KB  — (I−a_mat)^{-1} after fwd subst
//   s_a_row[S]   256 B   — scratch row for fwd subst phase A
//   s_gcsum[S]   256 B   — g_cumsum
//   s_tile [S*BK] up to 16 KB  — staging for KKT / WY tiles
//   s_tile2[S*BK] up to 16 KB  — second tile slot (KKT phase)
// ═════════════════════════════════════════════════════════════════════════════

template<int HK, int HV, int S = 64, int BK = 64, typename T = float>
static __device__ void linear_attn_intra_impl(
    const T*     __restrict__ q_ci,
    const T*     __restrict__ k_ci,
    const T*     __restrict__ v_ci,
    const float* __restrict__ log_g_ci,
    const float* __restrict__ beta_ci,
    float*       __restrict__ w_ci,
    float*       __restrict__ u_ci,
    float*       __restrict__ gc_ci
) {
    const int tid  = threadIdx.x;
    const int NTHR = blockDim.x; // 256

    extern __shared__ float smem[];
    // Layout (offsets in floats):
    //   [0          .. S*S)     s_attn
    //   [S*S        .. S*S+S)   s_a_row
    //   [S*S+S      .. S*S+2S)  s_gcsum
    //   [S*S+2S     .. S*S+2S+S*BK) s_tile   (reused for tile1 and WY pass)
    //   [S*S+2S+S*BK .. S*S+2S+2*S*BK) s_tile2
    float* const s_attn  = smem;
    float* const s_a_row = smem + S * S;
    float* const s_gcsum = s_a_row + S;
    float* const s_tile  = s_gcsum + S;
    float* const s_tile2 = s_tile + S * BK;

    // ── Step 1: g_cumsum ───────────────────────────────────────────────────
    if (tid < S) s_gcsum[tid] = log_g_ci[tid];
    __syncthreads();
    prefix_sum_inplace(s_gcsum, tid, S);
    // Write gc to global
    if (tid < S) gc_ci[tid] = s_gcsum[tid];

    // ── Step 2: Init s_attn = I ────────────────────────────────────────────
    for (int idx = tid; idx < S * S; idx += NTHR) {
        int r = idx / S, c = idx % S;
        s_attn[idx] = (r == c) ? 1.0f : 0.0f;
    }
    __syncthreads();

    // ── Step 3: KKT + forward substitution ────────────────────────────────
    //
    // a_mat[i,j] = -dot(k_beta[i,:], k[j,:]) * exp(gc[i]-gc[j])  for j < i
    // (I − a_mat)^{-1} built row by row via forward substitution.
    //
    // For each row i=1..S-1:
    //   Phase A: thread tid (for tid < i) computes s_a_row[tid] via tiled dot.
    //   Phase B: thread tid (for tid < S) updates s_attn[i, tid].

    for (int i = 1; i < S; i++) {
        // Phase A: tiled dot product over HK dimension.
        // Thread tid < i computes dot(k_beta_i[:], k_tid[:]) * decay.
        // k_beta[i, hk] = k_ci[i*HK+hk] * beta_ci[i]  (computed on the fly).
        // We tile over BK-wide slices of HK.
        float dot_val = 0.0f;
        for (int bk = 0; bk < HK; bk += BK) {
            // Load k_beta tile for row i into s_tile[0..BK).
            // Load k tile for rows 0..i-1 into s_tile2[row*BK..].
            // Use all threads to stage data, then compute.
            // Stage k*beta for row i: threads 0..BK-1 load
            if (tid < BK && (bk + tid) < HK) {
                s_tile[tid] = load_as_f32(k_ci + i * HK, bk + tid) * beta_ci[i];
            }
            // Stage k rows 0..i-1 into s_tile2[row*BK + col]
            // Distribute loading: thread (row*BK + col) loads k_ci[row*HK + bk+col]
            for (int idx = tid; idx < i * BK; idx += NTHR) {
                int row = idx / BK;
                int col = idx % BK;
                if (bk + col < HK)
                    s_tile2[idx] = load_as_f32(k_ci + row * HK, bk + col);
                else
                    s_tile2[idx] = 0.0f;
            }
            __syncthreads();

            // Accumulate dot product for thread tid (if tid < i)
            if (tid < i) {
                for (int col = 0; col < BK && (bk + col) < HK; col += 2) {
                    float2 kb = make_float2(s_tile[col], s_tile[col + 1]);
                    float2 kv = make_float2(s_tile2[tid * BK + col],
                                            s_tile2[tid * BK + col + 1]);
                    dot_val += kb.x * kv.x + kb.y * kv.y;
                }
            }
            __syncthreads();
        }

        // Write s_a_row[tid] = -dot_val * decay
        if (tid < i) {
            float decay = __expf(s_gcsum[i] - s_gcsum[tid]);
            s_a_row[tid] = -dot_val * decay;
        }
        __syncthreads();

        // Phase B: update row i of s_attn
        if (tid < S) {
            float acc = 0.0f;
            for (int j = 0; j < i; j++) {
                acc += s_a_row[j] * s_attn[j * S + tid];
            }
            s_attn[i * S + tid] += acc;
        }
        __syncthreads();
    }
    // s_attn now holds (I − a_mat)^{-1}.

    // ── Step 4: WY — w = s_attn @ (k*beta*exp(gc)) ────────────────────────
    // Tile over HK (BK-wide passes).
    for (int bk = 0; bk < HK; bk += BK) {
        // Stage k*beta*exp(gc) for rows 0..S into s_tile[row*BK+col]
        for (int idx = tid; idx < S * BK; idx += NTHR) {
            int row = idx / BK;
            int col = idx % BK;
            int hk  = bk + col;
            if (hk < HK)
                s_tile[idx] = load_as_f32(k_ci + row * HK, hk)
                              * beta_ci[row] * __expf(s_gcsum[row]);
            else
                s_tile[idx] = 0.0f;
        }
        __syncthreads();

        // Each thread computes w[s1, bk+col] = Σ_{s2} s_attn[s1,s2] * s_tile[s2,col]
        for (int idx = tid; idx < S * BK; idx += NTHR) {
            int s1  = idx / BK;
            int col = idx % BK;
            int hk  = bk + col;
            if (hk < HK) {
                float acc = 0.0f;
                for (int s2 = 0; s2 < S; s2++)
                    acc += s_attn[s1 * S + s2] * s_tile[s2 * BK + col];
                w_ci[s1 * HK + hk] = acc;
            }
        }
        __syncthreads();
    }

    // ── Step 5: WY — u = s_attn @ (v*beta) ───────────────────────────────
    // Tile over HV (BK-wide passes, reusing BK constant).
    for (int bv = 0; bv < HV; bv += BK) {
        // Stage v*beta for rows 0..S into s_tile[row*BK+col]
        for (int idx = tid; idx < S * BK; idx += NTHR) {
            int row = idx / BK;
            int col = idx % BK;
            int hv  = bv + col;
            if (hv < HV)
                s_tile[idx] = load_as_f32(v_ci + row * HV, hv) * beta_ci[row];
            else
                s_tile[idx] = 0.0f;
        }
        __syncthreads();

        for (int idx = tid; idx < S * BK; idx += NTHR) {
            int s1  = idx / BK;
            int col = idx % BK;
            int hv  = bv + col;
            if (hv < HV) {
                float acc = 0.0f;
                for (int s2 = 0; s2 < S; s2++)
                    acc += s_attn[s1 * S + s2] * s_tile[s2 * BK + col];
                u_ci[s1 * HV + hv] = acc;
            }
        }
        __syncthreads();
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// K2 — linear_attn_state
// Grid : (B*NH, 1, 1)   Block : (256, 1, 1)
//
// Template params:
//   HK, HV  — head dims (both supported: 64 or 128, must be equal)
//   S       — chunk size (64)
//   HPG     — HK values owned per thread = HK * HV / 256
//             (64 for HK=HV=128, 16 for HK=HV=64)
//
// Thread decomposition (all 256 threads active):
//   bv_local = tid % HV   — column of state owned by this thread (0..HV-1)
//   hk_group = tid / HV   — which HPG-wide strip of HK (0..N_GROUPS-1)
//   N_GROUPS = 256 / HV   — (2 for HV=128, 4 for HV=64)
//
//   Each thread holds HPG floats in registers:
//     state_reg[j] = state[(hk_group*HPG + j), bv_local]  for j=0..HPG-1
//
// For HK=HV=128: HPG=64, N_GROUPS=2 → state_reg[64], no idle threads.
// For HK=HV=64:  HPG=16, N_GROUPS=4 → state_reg[16], no idle threads.
//
// Shared memory (~34 KB for HK=HV=128):
//   s_row       [HK]       — staging for w/k/q rows
//   s_partial   [256]      — reduction buffer (N_GROUPS * HV = 256, constant)
//   s_vnew_cache[S * HV]   — cache of the full vnew chunk to avoid S global
//                            re-reads and S __syncthreads() in Step B
// ═════════════════════════════════════════════════════════════════════════════

template<int HK, int HV, int S = 64, int HPG = 64, typename T = float>
static __device__ void linear_attn_state_impl(
    const float* __restrict__ w,
    const float* __restrict__ u,
    const float* __restrict__ gc,
    const T*     __restrict__ k,
    const T*     __restrict__ q,
    float*       __restrict__ state,   // read at start, overwritten at end (in-place)
    float*       __restrict__ inter,
    float*       __restrict__ vnew,
    int C
) {
    const int bh  = blockIdx.x;
    const int tid = threadIdx.x;
    constexpr int N_GROUPS = 256 / HV;   // number of HK strips

    const int bv_local      = tid % HV;          // 0..HV-1
    const int hk_group      = tid / HV;          // 0..N_GROUPS-1
    const int hk_local_base = hk_group * HPG;    // first HK index for this thread

    // HPG floats of state in registers — sized exactly, no waste.
    float state_reg[HPG];

    extern __shared__ float smem2[];
    float* const s_row        = smem2;                         // [HK]
    float* const s_partial    = s_row        + HK;             // [256]  (N_GROUPS * HV)
    float* const s_vnew_cache = s_partial    + N_GROUPS * HV;  // [S * HV]

    // ── Load state into registers ─────────────────────────────────────────
    float* my_state = state + (long)bh * HK * HV;
    for (int j = 0; j < HPG; j++) {
        state_reg[j] = my_state[(hk_local_base + j) * HV + bv_local];
    }

    const float* w_bh  = w  + (long)bh * C * S * HK;
    const float* u_bh  = u  + (long)bh * C * S * HV;
    const float* gc_bh = gc + (long)bh * C * S;
    const T*     k_bh  = k  + (long)bh * C * S * HK;
    const T*     q_bh  = q  + (long)bh * C * S * HK;
    float* inter_bh    = inter + (long)bh * C * S * HV;
    float* vnew_bh     = vnew  + (long)bh * C * S * HV;

    for (int ci = 0; ci < C; ci++) {
        const float* w_ci  = w_bh  + ci * S * HK;
        const float* u_ci  = u_bh  + ci * S * HV;
        const float* gc_ci = gc_bh + ci * S;
        const T*     k_ci  = k_bh  + ci * S * HK;
        const T*     q_ci  = q_bh  + ci * S * HK;
        float* inter_ci    = inter_bh + ci * S * HV;
        float* vnew_ci     = vnew_bh  + ci * S * HV;

        float gc_last = gc_ci[S - 1];

        // ── Step A: inter and vnew ────────────────────────────────────────
        //
        // inter[s, bv_local] = exp(gc[s]) * Σ_{j} q[s, hk_local_base+j] * state_reg[j]
        // vnew[s, bv_local]  = u[s, bv_local] − Σ_{j} w[s, hk_local_base+j] * state_reg[j]
        //
        // Both require a reduction over N_GROUPS via s_partial[256].
        for (int s = 0; s < S; s++) {
            float gc_s = gc_ci[s];

            // — inter —
            for (int idx = tid; idx < HK; idx += 256)
                s_row[idx] = load_as_f32(q_ci + s * HK, idx);
            __syncthreads();

            float inter_p = 0.0f;
            for (int j = 0; j < HPG; j++)
                inter_p += s_row[hk_local_base + j] * state_reg[j];
            s_partial[hk_group * HV + bv_local] = inter_p * __expf(gc_s);
            __syncthreads();

            if (hk_group == 0) {
                float sum = 0.f;
                for (int g = 0; g < N_GROUPS; g++)
                    sum += s_partial[g * HV + bv_local];
                inter_ci[s * HV + bv_local] = sum;
            }
            __syncthreads();

            // — vnew —
            for (int idx = tid; idx < HK; idx += 256)
                s_row[idx] = w_ci[s * HK + idx];
            __syncthreads();

            float w_p = 0.0f;
            for (int j = 0; j < HPG; j++)
                w_p += s_row[hk_local_base + j] * state_reg[j];
            s_partial[hk_group * HV + bv_local] = w_p;
            __syncthreads();

            if (hk_group == 0) {
                float sum = 0.f;
                for (int g = 0; g < N_GROUPS; g++)
                    sum += s_partial[g * HV + bv_local];
                float vn = u_ci[s * HV + bv_local] - sum;
                vnew_ci[s * HV + bv_local]        = vn;  // global write (K3 reads)
                s_vnew_cache[s * HV + bv_local]   = vn;  // smem cache for Step B
            }
            __syncthreads();
        }
        // s_vnew_cache[S, HV] is now fully populated for this chunk.

        // ── Step B: state update ──────────────────────────────────────────
        //
        // state_reg *= exp(gc_last)
        // for s2 in 0..S: state_reg[j] += k[s2, hk_local_base+j] * decay * vnew[s2, bv_local]
        //
        // vnew is read from s_vnew_cache instead of global memory, avoiding S
        // global loads and S __syncthreads() per chunk.
        float g_end = __expf(gc_last);
        for (int j = 0; j < HPG; j++) state_reg[j] *= g_end;

        for (int s2 = 0; s2 < S; s2++) {
            for (int idx = tid; idx < HK; idx += 256)
                s_row[idx] = load_as_f32(k_ci + s2 * HK, idx);
            __syncthreads();

            float decay = __expf(gc_last - gc_ci[s2]);
            float vn    = s_vnew_cache[s2 * HV + bv_local];
            for (int j = 0; j < HPG; j++)
                state_reg[j] += s_row[hk_local_base + j] * decay * vn;
            __syncthreads();
        }
    } // end chunk loop

    // ── Write updated state back in-place ────────────────────────────────
    for (int j = 0; j < HPG; j++) {
        my_state[(hk_local_base + j) * HV + bv_local] = state_reg[j];
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// K3 — linear_attn_output
// Grid : (B*NH*C, 1, 1)   Block : (256, 1, 1)
//
// Inputs (slice for ONE chunk):
//   q_ci, k_ci  : [S, HK]  dtype T
//   vnew_ci     : [S, HV]  F32
//   inter_ci    : [S, HV]  F32
//   gc_ci       : [S]      F32
// Output:
//   out_ci      : [S, HV]  F32
//
// Algorithm:
//   1. Load inter into register accumulator
//   2. Tiled qk: s_attn[S,S] = q @ k^T  (tiled over HK in BK-wide passes)
//   3. Causal decay mask: s_attn[i,j] *= exp(gc[i]-gc[j]) for j≤i, else 0
//   4. Tiled matmul: out += s_attn @ vnew  (tiled over HV in BV-wide passes)
//   5. Write out
//
// Shared memory (peak ~49 KB for S=64, BK=BV=64):
//   s_attn [S*S]  16 KB
//   s_q    [S*BK] 16 KB  (reused with s_k in different sub-steps)
//   s_k    [S*BK] 16 KB
//   s_gc   [S]   256 B
// ═════════════════════════════════════════════════════════════════════════════

template<int HK, int HV, int S = 64, int BK = 64, int BV = 64, typename T = float>
static __device__ void linear_attn_output_impl(
    const T*     __restrict__ q_ci,
    const T*     __restrict__ k_ci,
    const float* __restrict__ vnew_ci,
    const float* __restrict__ inter_ci,
    const float* __restrict__ gc_ci,
    float*       __restrict__ out_ci
) {
    const int tid  = threadIdx.x;
    const int NTHR = blockDim.x; // 256

    extern __shared__ float smem3[];
    float* const s_attn = smem3;             // [S*S]  16 KB
    float* const s_q    = smem3 + S * S;     // [S*BK] 16 KB
    float* const s_k    = s_q + S * BK;      // [S*BK] 16 KB
    float* const s_gc   = s_k + S * BK;      // [S]   256 B

    // Load gc
    if (tid < S) s_gc[tid] = gc_ci[tid];
    __syncthreads();

    // ── Step 1: Init out accumulator from inter ───────────────────────────
    // Each thread writes a range of out[s,hv] starting from inter.
    // We'll accumulate in registers. With S*HV=8192 elements and 256 threads,
    // each thread handles 32 elements.
    // Store in s_attn temporarily after qk phase (it's free then).
    // For now load inter into global output directly; we add to it later.
    // Strategy: accumulate into out_ci directly. First write inter, then add.
    for (int idx = tid; idx < S * HV; idx += NTHR) {
        out_ci[idx] = inter_ci[idx];
    }
    __syncthreads();

    // ── Step 2: s_attn = q @ k^T  (tiled over HK) ────────────────────────
    // Init s_attn = 0
    for (int idx = tid; idx < S * S; idx += NTHR) s_attn[idx] = 0.0f;
    __syncthreads();

    for (int bk = 0; bk < HK; bk += BK) {
        // Load s_q[s, col] = q_ci[s, bk+col]  and  s_k[s, col] = k_ci[s, bk+col]
        for (int idx = tid; idx < S * BK; idx += NTHR) {
            int s   = idx / BK;
            int col = idx % BK;
            int hk  = bk + col;
            s_q[idx] = (hk < HK) ? load_as_f32(q_ci + s * HK, hk) : 0.0f;
            s_k[idx] = (hk < HK) ? load_as_f32(k_ci + s * HK, hk) : 0.0f;
        }
        __syncthreads();

        // Outer product accumulation: s_attn[s1, s2] += Σ_col s_q[s1,col]*s_k[s2,col]
        // Distribute (s1, s2) pairs across threads.
        // S*S = 4096 pairs, 256 threads → 16 pairs/thread.
        for (int idx = tid; idx < S * S; idx += NTHR) {
            int s1 = idx / S;
            int s2 = idx % S;
            float acc = 0.0f;
            for (int col = 0; col < BK; col++) {
                acc += s_q[s1 * BK + col] * s_k[s2 * BK + col];
            }
            s_attn[idx] += acc;
        }
        __syncthreads();
    }

    // ── Step 3: Causal decay mask ─────────────────────────────────────────
    for (int idx = tid; idx < S * S; idx += NTHR) {
        int s1 = idx / S;
        int s2 = idx % S;
        if (s2 > s1) {
            s_attn[idx] = 0.0f;
        } else {
            s_attn[idx] *= __expf(s_gc[s1] - s_gc[s2]);
        }
    }
    __syncthreads();

    // ── Step 4: out += s_attn @ vnew  (tiled over HV) ────────────────────
    for (int bv = 0; bv < HV; bv += BV) {
        // Load s_k (reused as s_v) = vnew_ci[:, bv..bv+BV]
        for (int idx = tid; idx < S * BV; idx += NTHR) {
            int s   = idx / BV;
            int col = idx % BV;
            int hv  = bv + col;
            s_k[idx] = (hv < HV) ? vnew_ci[s * HV + hv] : 0.0f;
        }
        __syncthreads();

        // Accumulate: out[s1, bv+col] += Σ_{s2} s_attn[s1,s2] * s_k[s2,col]
        for (int idx = tid; idx < S * BV; idx += NTHR) {
            int s1  = idx / BV;
            int col = idx % BV;
            int hv  = bv + col;
            if (hv < HV) {
                float acc = 0.0f;
                for (int s2 = 0; s2 < S; s2++)
                    acc += s_attn[s1 * S + s2] * s_k[s2 * BV + col];
                out_ci[s1 * HV + hv] += acc;
            }
        }
        __syncthreads();
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Entry points
// ═════════════════════════════════════════════════════════════════════════════

// Shared memory sizes (in bytes):
//   K1: S*S*4 + 2*S*4 + 2*S*BK*4       = 16384 + 512 + 32768  = 49664 B (~49 KB, needs 96 KB carveout)
//   K2: (HK + 256 + S*HV)*4             = (128 + 256 + 8192)*4 = 34304 B (~34 KB, fits in default 48 KB)
//   K3: S*S*4 + 2*S*BK*4 + S*4         = 16384 + 32768 + 256  = 49408 B (~49 KB, needs 96 KB carveout)

#define K1_SMEM(S, BK)      ((S)*(S)*4 + 2*(S)*4 + 2*(S)*(BK)*4)
#define K2_SMEM(HK, HV, S)  (((HK) + 256 + (S)*(HV))*4)
#define K3_SMEM(S, BK)      ((S)*(S)*4 + 2*(S)*(BK)*4 + (S)*4)

// ── K1 ───────────────────────────────────────────────────────────────────────

#define DEF_INTRA_KERNEL(DTYPE_NAME, HK_VAL, HV_VAL, T_TYPE)                 \
extern "C" __global__                                                         \
__launch_bounds__(256, 2)                                                     \
void linear_attn_intra_##DTYPE_NAME##_hk##HK_VAL##_hv##HV_VAL(              \
    const T_TYPE* q,   const T_TYPE* k,   const T_TYPE* v,                   \
    const float*  log_g, const float* beta,                                   \
    float* w, float* u, float* gc                                             \
) {                                                                           \
    int bh_chunk = blockIdx.x;   /* flat index into b_nh × C */              \
    /* The caller lays out q/k/v as [B*NH*C, S, dim], so slice by bh_chunk */\
    linear_attn_intra_impl<HK_VAL, HV_VAL, 64, 64, T_TYPE>(                 \
        q   + (long)bh_chunk * 64 * HK_VAL,                                  \
        k   + (long)bh_chunk * 64 * HK_VAL,                                  \
        v   + (long)bh_chunk * 64 * HV_VAL,                                  \
        log_g + (long)bh_chunk * 64,                                          \
        beta  + (long)bh_chunk * 64,                                          \
        w   + (long)bh_chunk * 64 * HK_VAL,                                  \
        u   + (long)bh_chunk * 64 * HV_VAL,                                  \
        gc  + (long)bh_chunk * 64                                             \
    );                                                                        \
}

DEF_INTRA_KERNEL(f32,  64,  64,  float)
DEF_INTRA_KERNEL(f32,  128, 128, float)
DEF_INTRA_KERNEL(bf16, 64,  64,  __nv_bfloat16)
DEF_INTRA_KERNEL(bf16, 128, 128, __nv_bfloat16)

// ── K2 ───────────────────────────────────────────────────────────────────────

// HPG = HK * HV / 256  (hk values per thread, ensures all 256 threads active)
//   (64,64):   HPG = 64*64/256 = 16
//   (128,128): HPG = 128*128/256 = 64
#define DEF_STATE_KERNEL(DTYPE_NAME, HK_VAL, HV_VAL, HPG_VAL, T_TYPE)        \
extern "C" __global__                                                         \
__launch_bounds__(256, 2)                                                     \
void linear_attn_state_##DTYPE_NAME##_hk##HK_VAL##_hv##HV_VAL(              \
    const float* w,  const float* u,  const float* gc,                       \
    const T_TYPE* k, const T_TYPE* q,                                         \
    float* state,                                                             \
    float* inter, float* vnew, int C                                          \
) {                                                                           \
    linear_attn_state_impl<HK_VAL, HV_VAL, 64, HPG_VAL, T_TYPE>(            \
        w, u, gc, k, q, state, inter, vnew, C);                              \
}

DEF_STATE_KERNEL(f32,  64,  64,  16, float)
DEF_STATE_KERNEL(f32,  128, 128, 64, float)
DEF_STATE_KERNEL(bf16, 64,  64,  16, __nv_bfloat16)
DEF_STATE_KERNEL(bf16, 128, 128, 64, __nv_bfloat16)

// ── K3 ───────────────────────────────────────────────────────────────────────

#define DEF_OUTPUT_KERNEL(DTYPE_NAME, HK_VAL, HV_VAL, T_TYPE)                \
extern "C" __global__                                                         \
__launch_bounds__(256, 2)                                                     \
void linear_attn_output_##DTYPE_NAME##_hk##HK_VAL##_hv##HV_VAL(             \
    const T_TYPE* q, const T_TYPE* k,                                         \
    const float* vnew, const float* inter, const float* gc,                   \
    float* out                                                                \
) {                                                                           \
    int bh_chunk = blockIdx.x;                                                \
    linear_attn_output_impl<HK_VAL, HV_VAL, 64, 64, 64, T_TYPE>(            \
        q    + (long)bh_chunk * 64 * HK_VAL,                                 \
        k    + (long)bh_chunk * 64 * HK_VAL,                                 \
        vnew + (long)bh_chunk * 64 * HV_VAL,                                 \
        inter+ (long)bh_chunk * 64 * HV_VAL,                                 \
        gc   + (long)bh_chunk * 64,                                           \
        out  + (long)bh_chunk * 64 * HV_VAL                                  \
    );                                                                        \
}

DEF_OUTPUT_KERNEL(f32,  64,  64,  float)
DEF_OUTPUT_KERNEL(f32,  128, 128, float)
DEF_OUTPUT_KERNEL(bf16, 64,  64,  __nv_bfloat16)
DEF_OUTPUT_KERNEL(bf16, 128, 128, __nv_bfloat16)

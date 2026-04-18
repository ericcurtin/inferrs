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

// ═════════════════════════════════════════════════════════════════════════════
// K2a — linear_attn_ops
// Grid : (B*NH*C, 1, 1)   Block : (256, 1, 1)
//
// Computes per-chunk linear recurrence operators (A_i, b_i):
//   K_d[s,j] = k[s,j] * exp(gc_last - gc[s])
//   A_i = exp(gc_last) * I - K_d^T @ W     [HK, HK]
//   b_i = K_d^T @ U                         [HK, HV]
//
// Outputs:
//   P_buf : [B*NH*C, HK, HK] row-major  — A_i per chunk
//   q_buf : [B*NH*C, HK, HV] col-major  — b_i per chunk (same layout as state)
//
// Algorithm: Tiled GEMM over the S (timestep) dimension.
//   BK=32 tiles over HK for the output tiles.
//   Thread layout: 16x16 block, each thread owns 2x2 sub-tile.
//
// Shared memory:
//   s_kd  [BK, S]   — K_d tile (BK rows of the S-column K_d matrix)
//   s_x   [S, BK]   — W or U tile
//   s_gc  [S]       — gc values for the chunk
//   Total: (BK*S + S*BK + S)*4 = (2048+2048+64)*4 = 16640 B (~16 KB)
// ═════════════════════════════════════════════════════════════════════════════

template<int HK, int HV, int S = 64, int BK = 32, typename T = float>
static __device__ void linear_attn_ops_impl(
    const float* __restrict__ w,
    const float* __restrict__ u,
    const float* __restrict__ gc,
    const T*     __restrict__ k,
    float*       __restrict__ P_buf,
    float*       __restrict__ q_buf,
    int C_real,
    int C_padded
) {
    const int tid = threadIdx.x;
    const int bh_chunk = blockIdx.x;
    const int bh = bh_chunk / C_real;
    const int ci = bh_chunk % C_real;

    const float* w_ci  = w  + (long)bh_chunk * S * HK;
    const float* u_ci  = u  + (long)bh_chunk * S * HV;
    const float* gc_ci = gc + (long)bh_chunk * S;
    const T*     k_ci  = k  + (long)bh_chunk * S * HK;

    float* P_ci = P_buf + ((long)bh * C_padded + ci) * HK * HK;
    float* q_ci = q_buf + ((long)bh * C_padded + ci) * HK * HV;

    extern __shared__ float smem_k2a[];
    float* const s_kd = smem_k2a;
    float* const s_x  = s_kd + BK * S;
    float* const s_gc_local = s_x + S * BK;

    for (int idx = tid; idx < S; idx += 256)
        s_gc_local[idx] = gc_ci[idx];
    __syncthreads();

    float gc_last = s_gc_local[S - 1];
    float g_end = __expf(gc_last);

    // Thread owns 2x2 sub-tile within a BKxBK tile
    // 16x16 threads = 256
    const int tx = tid % 16;
    const int ty = tid / 16;

    // ── Phase 1: Compute A_i = g_end * I - K_d^T @ W ─────────────────────
    // Output is [HK, HK] row-major
    for (int rt = 0; rt < HK; rt += BK) {
        for (int ct = 0; ct < HK; ct += BK) {
            float acc[2][2];
            acc[0][0] = 0.f; acc[0][1] = 0.f;
            acc[1][0] = 0.f; acc[1][1] = 0.f;

            for (int s_start = 0; s_start < S; s_start += S) {
                // Load K_d rows [rt..rt+BK) into s_kd[BK, S]
                for (int idx = tid; idx < BK * S; idx += 256) {
                    int br = idx / S;
                    int sc = idx % S;
                    int hk = rt + br;
                    if (hk < HK) {
                        float kv = load_as_f32(k_ci + sc * HK, hk);
                        float decay = __expf(gc_last - s_gc_local[sc]);
                        s_kd[br * S + sc] = kv * decay;
                    } else {
                        s_kd[br * S + sc] = 0.f;
                    }
                }
                __syncthreads();

                // Load W rows [0..S) cols [ct..ct+BK) into s_x[S, BK]
                for (int idx = tid; idx < S * BK; idx += 256) {
                    int sc = idx / BK;
                    int bc = idx % BK;
                    int hk = ct + bc;
                    s_x[idx] = (hk < HK) ? w_ci[sc * HK + hk] : 0.f;
                }
                __syncthreads();

                // Each thread computes 2x2 = 4 elements of the output tile
                // A[rt+ty*2+dy, ct+tx*2+dx] = -sum_s s_kd[br, s] * s_x[s, bc]
                for (int s = 0; s < S; s++) {
                    float kd0 = s_kd[(ty * 2 + 0) * S + s];
                    float kd1 = s_kd[(ty * 2 + 1) * S + s];
                    float xs0 = s_x[s * BK + tx * 2 + 0];
                    float xs1 = s_x[s * BK + tx * 2 + 1];
                    acc[0][0] -= kd0 * xs0;
                    acc[0][1] -= kd0 * xs1;
                    acc[1][0] -= kd1 * xs0;
                    acc[1][1] -= kd1 * xs1;
                }
                __syncthreads();
            }

            // Add identity * g_end and write
            int row0 = rt + ty * 2 + 0;
            int row1 = rt + ty * 2 + 1;
            int col0 = ct + tx * 2 + 0;
            int col1 = ct + tx * 2 + 1;
            if (row0 < HK && col0 < HK) {
                acc[0][0] += (row0 == col0) ? g_end : 0.f;
                P_ci[row0 * HK + col0] = acc[0][0];
            }
            if (row0 < HK && col1 < HK) {
                P_ci[row0 * HK + col1] = acc[0][1];
            }
            if (row1 < HK && col0 < HK) {
                P_ci[row1 * HK + col0] = acc[1][0];
            }
            if (row1 < HK && col1 < HK) {
                acc[1][1] += (row1 == col1) ? g_end : 0.f;
                P_ci[row1 * HK + col1] = acc[1][1];
            }
            __syncthreads();
        }
    }

    // ── Phase 2: Compute b_i = K_d^T @ U ─────────────────────────────────
    // Output is [HK, HV] — same col-major layout as state: [hk, hv] -> offset = hk * HV + hv
    for (int rt = 0; rt < HK; rt += BK) {
        for (int ct = 0; ct < HV; ct += BK) {
            float acc[2][2];
            acc[0][0] = 0.f; acc[0][1] = 0.f;
            acc[1][0] = 0.f; acc[1][1] = 0.f;

            for (int s_start = 0; s_start < S; s_start += S) {
                // Re-load K_d rows [rt..rt+BK) into s_kd
                for (int idx = tid; idx < BK * S; idx += 256) {
                    int br = idx / S;
                    int sc = idx % S;
                    int hk = rt + br;
                    if (hk < HK) {
                        float kv = load_as_f32(k_ci + sc * HK, hk);
                        float decay = __expf(gc_last - s_gc_local[sc]);
                        s_kd[br * S + sc] = kv * decay;
                    } else {
                        s_kd[br * S + sc] = 0.f;
                    }
                }
                __syncthreads();

                // Load U rows [0..S) cols [ct..ct+BK) into s_x[S, BK]
                for (int idx = tid; idx < S * BK; idx += 256) {
                    int sc = idx / BK;
                    int bc = idx % BK;
                    int hv = ct + bc;
                    s_x[idx] = (hv < HV) ? u_ci[sc * HV + hv] : 0.f;
                }
                __syncthreads();

                for (int s = 0; s < S; s++) {
                    float kd0 = s_kd[(ty * 2 + 0) * S + s];
                    float kd1 = s_kd[(ty * 2 + 1) * S + s];
                    float xs0 = s_x[s * BK + tx * 2 + 0];
                    float xs1 = s_x[s * BK + tx * 2 + 1];
                    acc[0][0] += kd0 * xs0;
                    acc[0][1] += kd0 * xs1;
                    acc[1][0] += kd1 * xs0;
                    acc[1][1] += kd1 * xs1;
                }
                __syncthreads();
            }

            int row0 = rt + ty * 2 + 0;
            int row1 = rt + ty * 2 + 1;
            int col0 = ct + tx * 2 + 0;
            int col1 = ct + tx * 2 + 1;
            if (row0 < HK && col0 < HV)
                q_ci[row0 * HV + col0] = acc[0][0];
            if (row0 < HK && col1 < HV)
                q_ci[row0 * HV + col1] = acc[0][1];
            if (row1 < HK && col0 < HV)
                q_ci[row1 * HV + col0] = acc[1][0];
            if (row1 < HK && col1 < HV)
                q_ci[row1 * HV + col1] = acc[1][1];
            __syncthreads();
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// K2b — linear_attn_prefix_scan (up-sweep)
// Grid : variable (see Rust dispatch)   Block : (256, 1, 1)
//
// One launch per up-sweep level. Each block composes two adjacent chunks:
//   P[i] = P[i] ∘ P[i-stride]
//   q[i] = P[i] * q[i-stride] + q[i]
//   Saved left values: A_buf[i-stride] = old P[i-stride], b_buf[i-stride] = old q[i-stride]
//
// BK=16 for smem. Thread layout: 16x16, each thread owns 2x2 sub-tile.
//
// Shared memory:
//   s_a [BK, BK]  — left tile (from P[i-stride])
//   s_b [BK, BK]  — right tile (from P[i])
//   Total: 2*BK*BK*4 = 2048 B (~2 KB)
// ═════════════════════════════════════════════════════════════════════════════

template<int HK, int HV, int BK = 16>
static __device__ void linear_attn_scan_up_impl(
    float* __restrict__ P_buf,
    float* __restrict__ q_buf,
    float* __restrict__ A_buf,
    float* __restrict__ b_buf,
    int stride,
    int C_padded
) {
    const int tid = threadIdx.x;
    const int pair_id = blockIdx.x;
    const int pairs_per_bh = C_padded / stride;
    const int bh = pair_id / pairs_per_bh;
    const int pair = pair_id % pairs_per_bh;
    const int i = bh * C_padded + pair * stride + (stride - 1);
    const int left = i - stride / 2;

    if (pair * stride + (stride - 1) >= C_padded || left < bh * C_padded) return;

    float* P_left  = P_buf + (long)left * HK * HK;
    float* P_right = P_buf + (long)i    * HK * HK;
    float* q_left  = q_buf + (long)left * HK * HV;
    float* q_right = q_buf + (long)i    * HK * HV;
    float* A_save  = A_buf + (long)left * HK * HK;
    float* b_save  = b_buf + (long)left * HK * HV;

    const int tx = tid % 16;
    const int ty = tid / 16;

    // ── Save left values before composition ──────────────────────────────
    for (int rt = 0; rt < HK; rt += BK) {
        for (int ct = 0; ct < HK; ct += BK) {
            int r0 = rt + ty * 2 + 0, r1 = rt + ty * 2 + 1;
            int c0 = ct + tx * 2 + 0, c1 = ct + tx * 2 + 1;
            if (r0 < HK && c0 < HK) A_save[r0 * HK + c0] = P_left[r0 * HK + c0];
            if (r0 < HK && c1 < HK) A_save[r0 * HK + c1] = P_left[r0 * HK + c1];
            if (r1 < HK && c0 < HK) A_save[r1 * HK + c0] = P_left[r1 * HK + c0];
            if (r1 < HK && c1 < HK) A_save[r1 * HK + c1] = P_left[r1 * HK + c1];
        }
    }

    for (int rt = 0; rt < HK; rt += BK) {
        for (int ct = 0; ct < HV; ct += BK) {
            int r0 = rt + ty * 2 + 0, r1 = rt + ty * 2 + 1;
            int c0 = ct + tx * 2 + 0, c1 = ct + tx * 2 + 1;
            if (r0 < HK && c0 < HV) b_save[r0 * HV + c0] = q_left[r0 * HV + c0];
            if (r0 < HK && c1 < HV) b_save[r0 * HV + c1] = q_left[r0 * HV + c1];
            if (r1 < HK && c0 < HV) b_save[r1 * HV + c0] = q_left[r1 * HV + c0];
            if (r1 < HK && c1 < HV) b_save[r1 * HV + c1] = q_left[r1 * HV + c1];
        }
    }

    __syncthreads();

    // ── Compose: q[i] = P_old[i] @ q[left] + q[i], then P[i] = P[i] @ P[left] ─
    // q composition must happen FIRST because it uses P_right before overwrite.
    //
    // Smem layout: s_a[BK*BK] | s_b[BK*BK] | s_pr[BK*HK]
    //   s_pr caches one row-block of P_right so the P composition can tile over
    //   output columns (ct) without reading cells that were already overwritten.
    extern __shared__ float smem_up[];
    float* const s_a  = smem_up;
    float* const s_b  = s_a + BK * BK;
    float* const s_pr = s_b + BK * BK;   // [BK * HK]

    // q composition: q_right = P_right_old @ q_left + q_right
    for (int rt = 0; rt < HK; rt += BK) {
        for (int ct = 0; ct < HV; ct += BK) {
            float acc[2][2] = {{0.f, 0.f}, {0.f, 0.f}};

            for (int kt = 0; kt < HK; kt += BK) {
                for (int idx = tid; idx < BK * BK; idx += 256) {
                    int r = kt + idx / BK, c = ct + idx % BK;
                    s_a[idx] = (r < HK && c < HV) ? q_left[r * HV + c] : 0.f;
                }
                for (int idx = tid; idx < BK * BK; idx += 256) {
                    int r = rt + idx / BK, c = kt + idx % BK;
                    s_b[idx] = (r < HK && c < HK) ? P_right[r * HK + c] : 0.f;
                }
                __syncthreads();

                for (int kk = 0; kk < BK; kk++) {
                    float b0 = s_b[(ty * 2 + 0) * BK + kk];
                    float b1 = s_b[(ty * 2 + 1) * BK + kk];
                    float a0 = s_a[kk * BK + tx * 2 + 0];
                    float a1 = s_a[kk * BK + tx * 2 + 1];
                    acc[0][0] += b0 * a0;
                    acc[0][1] += b0 * a1;
                    acc[1][0] += b1 * a0;
                    acc[1][1] += b1 * a1;
                }
                __syncthreads();
            }

            int r0 = rt + ty * 2 + 0, r1 = rt + ty * 2 + 1;
            int c0 = ct + tx * 2 + 0, c1 = ct + tx * 2 + 1;
            if (r0 < HK && c0 < HV) q_right[r0 * HV + c0] = acc[0][0] + q_right[r0 * HV + c0];
            if (r0 < HK && c1 < HV) q_right[r0 * HV + c1] = acc[0][1] + q_right[r0 * HV + c1];
            if (r1 < HK && c0 < HV) q_right[r1 * HV + c0] = acc[1][0] + q_right[r1 * HV + c0];
            if (r1 < HK && c1 < HV) q_right[r1 * HV + c1] = acc[1][1] + q_right[r1 * HV + c1];
            __syncthreads();
        }
    }

    // P composition: P_right = P_right @ P_left
    //
    // Without care, the tiled in-place write to P_right[rt, ct] would corrupt
    // subsequent tiles (rt, ct') that still need to read P_right[rt, ct] from
    // the kt loop.  Fix: for each rt block, cache the full P_right[rt:rt+BK, :]
    // row into s_pr before computing any output columns for that block.
    for (int rt = 0; rt < HK; rt += BK) {
        // Load P_right[rt:rt+BK, 0:HK] into s_pr[BK, HK] before any writes.
        for (int idx = tid; idx < BK * HK; idx += 256) {
            int br = idx / HK, bc = idx % HK;
            int r  = rt + br;
            s_pr[br * HK + bc] = (r < HK) ? P_right[r * HK + bc] : 0.f;
        }
        __syncthreads();

        for (int ct = 0; ct < HK; ct += BK) {
            float acc[2][2] = {{0.f, 0.f}, {0.f, 0.f}};

            for (int kt = 0; kt < HK; kt += BK) {
                for (int idx = tid; idx < BK * BK; idx += 256) {
                    int r = kt + idx / BK, c = ct + idx % BK;
                    s_a[idx] = (r < HK && c < HK) ? P_left[r * HK + c] : 0.f;
                }
                __syncthreads();

                for (int kk = 0; kk < BK; kk++) {
                    float pr0 = s_pr[(ty * 2 + 0) * HK + kt + kk];
                    float pr1 = s_pr[(ty * 2 + 1) * HK + kt + kk];
                    float pl0 = s_a[kk * BK + tx * 2 + 0];
                    float pl1 = s_a[kk * BK + tx * 2 + 1];
                    acc[0][0] += pr0 * pl0;
                    acc[0][1] += pr0 * pl1;
                    acc[1][0] += pr1 * pl0;
                    acc[1][1] += pr1 * pl1;
                }
                __syncthreads();
            }

            int r0 = rt + ty * 2 + 0, r1 = rt + ty * 2 + 1;
            int c0 = ct + tx * 2 + 0, c1 = ct + tx * 2 + 1;
            if (r0 < HK && c0 < HK) P_right[r0 * HK + c0] = acc[0][0];
            if (r0 < HK && c1 < HK) P_right[r0 * HK + c1] = acc[0][1];
            if (r1 < HK && c0 < HK) P_right[r1 * HK + c0] = acc[1][0];
            if (r1 < HK && c1 < HK) P_right[r1 * HK + c1] = acc[1][1];
            __syncthreads();
        }
        // s_pr will be overwritten at the top of the next rt iteration,
        // which is safe because all ct writes for this rt block are done.
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// K2b — linear_attn_prefix_scan (down-sweep)
// Grid : variable   Block : (256, 1, 1)
//
// Non-commutative Blelloch down-sweep. Each block:
//   old = P[i]
//   P[i-stride] = old                    (right child gets parent prefix)
//   P[i] = saved_left[i-stride] @ old    (left child gets left∘parent)
//
// Same for q_buf: q[i-stride] = old_q[i], q[i] = saved_b[i-stride] + A[i-stride] @ old_q[i]
// ═════════════════════════════════════════════════════════════════════════════

template<int HK, int HV, int BK = 16>
static __device__ void linear_attn_scan_down_impl(
    float* __restrict__ P_buf,
    float* __restrict__ q_buf,
    const float* __restrict__ A_buf,
    const float* __restrict__ b_buf,
    int stride,
    int C_padded
) {
    const int tid = threadIdx.x;
    const int pair_id = blockIdx.x;
    const int pairs_per_bh = C_padded / stride;
    const int bh = pair_id / pairs_per_bh;
    const int pair = pair_id % pairs_per_bh;
    const int i = bh * C_padded + pair * stride + (stride - 1);
    const int left = i - stride / 2;

    if (pair * stride + (stride - 1) >= C_padded || left < bh * C_padded) return;

    float* P_left  = P_buf + (long)left * HK * HK;
    float* P_right = P_buf + (long)i    * HK * HK;
    float* q_left  = q_buf + (long)left * HK * HV;
    float* q_right = q_buf + (long)i    * HK * HV;
    const float* A_saved = A_buf + (long)left * HK * HK;
    const float* b_saved = b_buf + (long)left * HK * HV;

    const int tx = tid % 16;
    const int ty = tid / 16;

    extern __shared__ float smem_down[];
    float* const s_a = smem_down;
    float* const s_b = s_a + BK * BK;

    // ── Step 1: Save P_right (old) → P_left, q_right (old) → q_left ────
    for (int rt = 0; rt < HK; rt += BK) {
        for (int ct = 0; ct < HK; ct += BK) {
            int r0 = rt + ty * 2 + 0, r1 = rt + ty * 2 + 1;
            int c0 = ct + tx * 2 + 0, c1 = ct + tx * 2 + 1;
            float v00 = 0.f, v01 = 0.f, v10 = 0.f, v11 = 0.f;
            if (r0 < HK && c0 < HK) { v00 = P_right[r0 * HK + c0]; P_left[r0 * HK + c0] = v00; }
            if (r0 < HK && c1 < HK) { v01 = P_right[r0 * HK + c1]; P_left[r0 * HK + c1] = v01; }
            if (r1 < HK && c0 < HK) { v10 = P_right[r1 * HK + c0]; P_left[r1 * HK + c0] = v10; }
            if (r1 < HK && c1 < HK) { v11 = P_right[r1 * HK + c1]; P_left[r1 * HK + c1] = v11; }
        }
    }

    for (int rt = 0; rt < HK; rt += BK) {
        for (int ct = 0; ct < HV; ct += BK) {
            int r0 = rt + ty * 2 + 0, r1 = rt + ty * 2 + 1;
            int c0 = ct + tx * 2 + 0, c1 = ct + tx * 2 + 1;
            if (r0 < HK && c0 < HV) q_left[r0 * HV + c0] = q_right[r0 * HV + c0];
            if (r0 < HK && c1 < HV) q_left[r0 * HV + c1] = q_right[r0 * HV + c1];
            if (r1 < HK && c0 < HV) q_left[r1 * HV + c0] = q_right[r1 * HV + c0];
            if (r1 < HK && c1 < HV) q_left[r1 * HV + c1] = q_right[r1 * HV + c1];
        }
    }

    __syncthreads();

    // ── Step 2: P_right = A_saved @ old_P_right (tiled GEMM) ─────────────
    // P_right is the old value we just saved to P_left. Now overwrite P_right.
    for (int rt = 0; rt < HK; rt += BK) {
        for (int ct = 0; ct < HK; ct += BK) {
            float acc[2][2] = {{0.f, 0.f}, {0.f, 0.f}};

            for (int kt = 0; kt < HK; kt += BK) {
                for (int idx = tid; idx < BK * BK; idx += 256) {
                    int r = kt + idx / BK, c = ct + idx % BK;
                    s_a[idx] = (r < HK && c < HK) ? P_left[r * HK + c] : 0.f;
                }
                for (int idx = tid; idx < BK * BK; idx += 256) {
                    int r = rt + idx / BK, c = kt + idx % BK;
                    s_b[idx] = (r < HK && c < HK) ? A_saved[r * HK + c] : 0.f;
                }
                __syncthreads();

                for (int kk = 0; kk < BK; kk++) {
                    float b0 = s_b[(ty * 2 + 0) * BK + kk];
                    float b1 = s_b[(ty * 2 + 1) * BK + kk];
                    float a0 = s_a[kk * BK + tx * 2 + 0];
                    float a1 = s_a[kk * BK + tx * 2 + 1];
                    acc[0][0] += b0 * a0;
                    acc[0][1] += b0 * a1;
                    acc[1][0] += b1 * a0;
                    acc[1][1] += b1 * a1;
                }
                __syncthreads();
            }

            int r0 = rt + ty * 2 + 0, r1 = rt + ty * 2 + 1;
            int c0 = ct + tx * 2 + 0, c1 = ct + tx * 2 + 1;
            if (r0 < HK && c0 < HK) P_right[r0 * HK + c0] = acc[0][0];
            if (r0 < HK && c1 < HK) P_right[r0 * HK + c1] = acc[0][1];
            if (r1 < HK && c0 < HK) P_right[r1 * HK + c0] = acc[1][0];
            if (r1 < HK && c1 < HK) P_right[r1 * HK + c1] = acc[1][1];
            __syncthreads();
        }
    }

    // ── Step 3: q_right = A_saved @ old_q_right + b_saved ────────────────
    for (int rt = 0; rt < HK; rt += BK) {
        for (int ct = 0; ct < HV; ct += BK) {
            float acc[2][2] = {{0.f, 0.f}, {0.f, 0.f}};

            for (int kt = 0; kt < HK; kt += BK) {
                for (int idx = tid; idx < BK * BK; idx += 256) {
                    int r = kt + idx / BK, c = ct + idx % BK;
                    s_a[idx] = (r < HK && c < HV) ? q_left[r * HV + c] : 0.f;
                }
                for (int idx = tid; idx < BK * BK; idx += 256) {
                    int r = rt + idx / BK, c = kt + idx % BK;
                    s_b[idx] = (r < HK && c < HK) ? A_saved[r * HK + c] : 0.f;
                }
                __syncthreads();

                for (int kk = 0; kk < BK; kk++) {
                    float b0 = s_b[(ty * 2 + 0) * BK + kk];
                    float b1 = s_b[(ty * 2 + 1) * BK + kk];
                    float a0 = s_a[kk * BK + tx * 2 + 0];
                    float a1 = s_a[kk * BK + tx * 2 + 1];
                    acc[0][0] += b0 * a0;
                    acc[0][1] += b0 * a1;
                    acc[1][0] += b1 * a0;
                    acc[1][1] += b1 * a1;
                }
                __syncthreads();
            }

            int r0 = rt + ty * 2 + 0, r1 = rt + ty * 2 + 1;
            int c0 = ct + tx * 2 + 0, c1 = ct + tx * 2 + 1;
            if (r0 < HK && c0 < HV) q_right[r0 * HV + c0] = acc[0][0] + b_saved[r0 * HV + c0];
            if (r0 < HK && c1 < HV) q_right[r0 * HV + c1] = acc[0][1] + b_saved[r0 * HV + c1];
            if (r1 < HK && c0 < HV) q_right[r1 * HV + c0] = acc[1][0] + b_saved[r1 * HV + c0];
            if (r1 < HK && c1 < HV) q_right[r1 * HV + c1] = acc[1][1] + b_saved[r1 * HV + c1];
            __syncthreads();
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// K2c — linear_attn_apply
// Grid : (B*NH*C, 1, 1)   Block : (256, 1, 1)
//
// Reconstructs per-chunk state from prefix scan result, then computes
// inter/vnew (Step A), and optionally new_state for ci==C-1.
//
// Inputs per chunk:
//   P_buf[ci] : [HK, HK] — exclusive prefix A composition
//   q_buf[ci] : [HK, HV] — exclusive prefix b composition
//   state_0   : [HK, HV] — initial state (read-only)
//   w_ci, u_ci, gc_ci, q_ci, k_ci — per-chunk data
//
// Thread decomposition: same as K2 (bv_local=tid%HV, hk_group=tid/HV, HPG regs)
//
// Shared memory:
//   s_p_tile  [HK, BK]  — full-HK column tile of P_buf (all groups share one load)
//   s_st_tile [BK, HV]  — row tile of state_0 (all columns, avoids per-group aliasing)
//   s_gc      [S]       — gc values
//   s_row_qw  [HK]      — staging for q/w rows
//   s_partial [256]     — cross-group reduction
// ═════════════════════════════════════════════════════════════════════════════

template<int HK, int HV, int S = 64, int HPG = 64, int BK = 16, typename T = float>
static __device__ void linear_attn_apply_impl(
    const float* __restrict__ w,
    const float* __restrict__ u,
    const float* __restrict__ gc,
    const T*     __restrict__ q_tensor,
    const T*     __restrict__ k,
    const float* __restrict__ state_0,   // initial state — read by ALL blocks concurrently
    float*       __restrict__ new_state, // updated state — written ONLY by ci==C_real-1 block
    const float* __restrict__ P_buf,
    const float* __restrict__ q_buf,
    float*       __restrict__ inter,
    float*       __restrict__ vnew,
    int C_real,
    int C_padded
) {
    const int tid = threadIdx.x;
    const int bh_chunk = blockIdx.x;
    const int ci = bh_chunk % C_padded;

    if (ci >= C_real) return;

    const long bh = bh_chunk / C_padded;

    constexpr int N_GROUPS = 256 / HV;
    const int bv_local      = tid % HV;
    const int hk_group      = tid / HV;
    const int hk_local_base = hk_group * HPG;

    float state_reg[HPG];

    extern __shared__ float smem_k2c[];
    float* const s_p_tile   = smem_k2c;
    float* const s_st_tile  = s_p_tile  + HK * BK;
    float* const s_gc_local = s_st_tile + BK * HV;
    float* const s_row_qw   = s_gc_local + S;
    float* const s_partial  = s_row_qw  + HK;

    const long real_chunk = (long)bh * C_real + ci;

    const float* w_ci  = w  + real_chunk * S * HK;
    const float* u_ci  = u  + real_chunk * S * HV;
    const float* gc_ci = gc + real_chunk * S;
    const T*     q_ci  = q_tensor + real_chunk * S * HK;
    float* inter_ci    = inter + real_chunk * S * HV;
    float* vnew_ci     = vnew  + real_chunk * S * HV;

    const float* P_ci = P_buf + ((long)bh * C_padded + ci) * HK * HK;
    const float* q_prefix = q_buf + ((long)bh * C_padded + ci) * HK * HV;
    const float* my_state = state_0 + (long)bh * HK * HV;

    // Load gc
    for (int idx = tid; idx < S; idx += 256)
        s_gc_local[idx] = gc_ci[idx];
    __syncthreads();

    // ── Step 1: state_in = P[ci] @ state_0 + q_prefix ─────────────────────
    // All thread groups cooperatively load full-HK tiles so every group reads
    // from its correct row range — the old per-group s_tile caused groups to
    // overwrite each other's entries (aliased indices) and s_row_st held a
    // diagonal of the state rather than a single column.
    for (int j = 0; j < HPG; j++) state_reg[j] = 0.f;

    for (int kt = 0; kt < HK; kt += BK) {
        // Load P_ci[:, kt:kt+BK] into s_p_tile[HK, BK] — all groups collaborate
        for (int idx = tid; idx < HK * BK; idx += 256) {
            int r = idx / BK;
            int c = kt + idx % BK;
            s_p_tile[idx] = (c < HK) ? P_ci[r * HK + c] : 0.f;
        }
        // Load state_0[kt:kt+BK, :] into s_st_tile[BK, HV] — all groups collaborate
        for (int idx = tid; idx < BK * HV; idx += 256) {
            int kk = idx / HV;
            int bv = idx % HV;
            int k  = kt + kk;
            s_st_tile[idx] = (k < HK) ? my_state[k * HV + bv] : 0.f;
        }
        __syncthreads();

        for (int j = 0; j < HPG; j++) {
            for (int kk = 0; kk < BK; kk++) {
                state_reg[j] += s_p_tile[(hk_local_base + j) * BK + kk]
                              * s_st_tile[kk * HV + bv_local];
            }
        }
        __syncthreads();
    }

    // Add q_prefix[hk_local_base..hk_local_base+HPG, bv_local]
    for (int j = 0; j < HPG; j++) {
        state_reg[j] += q_prefix[(hk_local_base + j) * HV + bv_local];
    }

    // ── Step 2: inter and vnew (Step A — same as K2) ──────────────────────
    for (int s = 0; s < S; s++) {
        float gc_s = s_gc_local[s];

        // inter
        for (int idx = tid; idx < HK; idx += 256)
            s_row_qw[idx] = load_as_f32(q_ci + s * HK, idx);
        __syncthreads();

        float inter_p = 0.0f;
        for (int j = 0; j < HPG; j++)
            inter_p += s_row_qw[hk_local_base + j] * state_reg[j];
        s_partial[hk_group * HV + bv_local] = inter_p * __expf(gc_s);
        __syncthreads();

        if (hk_group == 0) {
            float sum = 0.f;
            for (int g = 0; g < N_GROUPS; g++)
                sum += s_partial[g * HV + bv_local];
            inter_ci[s * HV + bv_local] = sum;
        }
        __syncthreads();

        // vnew
        for (int idx = tid; idx < HK; idx += 256)
            s_row_qw[idx] = w_ci[s * HK + idx];
        __syncthreads();

        float w_p = 0.0f;
        for (int j = 0; j < HPG; j++)
            w_p += s_row_qw[hk_local_base + j] * state_reg[j];
        s_partial[hk_group * HV + bv_local] = w_p;
        __syncthreads();

        if (hk_group == 0) {
            float sum = 0.f;
            for (int g = 0; g < N_GROUPS; g++)
                sum += s_partial[g * HV + bv_local];
            vnew_ci[s * HV + bv_local] = u_ci[s * HV + bv_local] - sum;
        }
        __syncthreads();
    }

    // ── Step 3: For ci==C_real-1, compute new_state via Step B ─────────────
    // new_state = A_{C-1} @ state_in + b_{C-1}
    // But we don't have A_{C-1}/b_{C-1} separately — we have the raw state update:
    // new_state = exp(gc_last)*state_in + sum_s k_d[s] * decay * vnew[s]
    if (ci == C_real - 1) {
        float gc_last = s_gc_local[S - 1];
        float g_end = __expf(gc_last);
        for (int j = 0; j < HPG; j++) state_reg[j] *= g_end;

        const T* k_ci = k + real_chunk * S * HK;
        for (int s2 = 0; s2 < S; s2++) {
            for (int idx = tid; idx < HK; idx += 256)
                s_row_qw[idx] = load_as_f32(k_ci + s2 * HK, idx);
            __syncthreads();

            float decay = __expf(gc_last - s_gc_local[s2]);
            float vn = vnew_ci[s2 * HV + bv_local];
            for (int j = 0; j < HPG; j++)
                state_reg[j] += s_row_qw[hk_local_base + j] * decay * vn;
            __syncthreads();
        }

        float* my_new_state = new_state + (long)bh * HK * HV;
        for (int j = 0; j < HPG; j++) {
            my_new_state[(hk_local_base + j) * HV + bv_local] = state_reg[j];
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// K2b — Init identity operators for padded chunks + clear root
// Grid : (B*NH, 1, 1)   Block : (256, 1, 1)
//
// For each ci in [C_real, C_padded): sets P[ci] = I, q[ci] = 0.
// Also sets P[C_padded-1] = I (root of up-sweep tree).
// When C_real == C_padded, only clears the root.
// ═════════════════════════════════════════════════════════════════════════════

template<int HK, int HV>
static __device__ void linear_attn_scan_init_impl(
    float* __restrict__ P_buf,
    float* __restrict__ q_buf,
    int C_real,
    int C_padded
) {
    const int tid = threadIdx.x;
    const int bh = blockIdx.x;

    for (int ci = C_real; ci < C_padded; ci++) {
        float* P_ci = P_buf + ((long)bh * C_padded + ci) * HK * HK;
        float* q_ci = q_buf + ((long)bh * C_padded + ci) * HK * HV;

        for (int idx = tid; idx < HK * HK; idx += 256) {
            int r = idx / HK, c = idx % HK;
            P_ci[idx] = (r == c) ? 1.f : 0.f;
        }
        for (int idx = tid; idx < HK * HV; idx += 256) {
            q_ci[idx] = 0.f;
        }
    }
    __syncthreads();

    // Always clear root (C_padded-1) to identity for the up-sweep result
    {
        float* P_root = P_buf + ((long)bh * C_padded + C_padded - 1) * HK * HK;
        float* q_root = q_buf + ((long)bh * C_padded + C_padded - 1) * HK * HV;
        for (int idx = tid; idx < HK * HK; idx += 256) {
            int r = idx / HK, c = idx % HK;
            P_root[idx] = (r == c) ? 1.f : 0.f;
        }
        for (int idx = tid; idx < HK * HV; idx += 256) {
            q_root[idx] = 0.f;
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Entry points for K2a, K2b_up, K2b_down, K2b_clear, K2c
// ═════════════════════════════════════════════════════════════════════════════

// Smem sizes:
//   K2a: (BK*S + S*BK + S)*4 = (2048+2048+64)*4 = 16640 B  (BK=32)
//   K2b_up/down: 2*BK*BK*4 = 8192 B (BK=32)
//   K2c: (HPG*BK + BK + S + HK + 256)*4  — varies by config
//   K2b_clear: 0

#define K2A_SMEM(BK, S)         (((BK)*(S) + (S)*(BK) + (S))*4)
// Up-sweep needs s_pr[BK*HK] in addition to s_a+s_b to cache P_right row-blocks
// before tiling over output columns (avoids in-place read-write conflict).
#define K2B_UP_SMEM(BK, HK)    ((2*(BK)*(BK) + (BK)*(HK))*4)
#define K2B_DOWN_SMEM(BK)      (2*(BK)*(BK)*4)
#define K2C_SMEM(BK, S, HK, HV)  (((HK)*(BK) + (BK)*(HV) + (S) + (HK) + 256)*4)

#define DEF_OPS_KERNEL(DTYPE_NAME, HK_VAL, HV_VAL, T_TYPE)                     \
extern "C" __global__                                                           \
__launch_bounds__(256, 2)                                                       \
void linear_attn_ops_##DTYPE_NAME##_hk##HK_VAL##_hv##HV_VAL(                  \
    const float* w,  const float* u,  const float* gc,                         \
    const T_TYPE* k,                                                            \
    float* P_buf, float* q_buf, int C_real, int C_padded                        \
) {                                                                             \
    linear_attn_ops_impl<HK_VAL, HV_VAL, 64, 32, T_TYPE>(                     \
        w, u, gc, k, P_buf, q_buf, C_real, C_padded);                          \
}

DEF_OPS_KERNEL(f32,  64,  64,  float)
DEF_OPS_KERNEL(f32,  128, 128, float)
DEF_OPS_KERNEL(bf16, 64,  64,  __nv_bfloat16)
DEF_OPS_KERNEL(bf16, 128, 128, __nv_bfloat16)

#define DEF_SCAN_UP_KERNEL(HK_VAL, HV_VAL)                                     \
extern "C" __global__                                                           \
__launch_bounds__(256, 2)                                                       \
void linear_attn_scan_up_hk##HK_VAL##_hv##HV_VAL(                             \
    float* P_buf, float* q_buf, float* A_buf, float* b_buf,                    \
    int stride, int C_padded                                                    \
) {                                                                             \
    linear_attn_scan_up_impl<HK_VAL, HV_VAL, 32>(                             \
        P_buf, q_buf, A_buf, b_buf, stride, C_padded);                        \
}

DEF_SCAN_UP_KERNEL(64,  64)
DEF_SCAN_UP_KERNEL(128, 128)

#define DEF_SCAN_DOWN_KERNEL(HK_VAL, HV_VAL)                                   \
extern "C" __global__                                                           \
__launch_bounds__(256, 2)                                                       \
void linear_attn_scan_down_hk##HK_VAL##_hv##HV_VAL(                           \
    float* P_buf, float* q_buf,                                                \
    const float* A_buf, const float* b_buf,                                    \
    int stride, int C_padded                                                    \
) {                                                                             \
    linear_attn_scan_down_impl<HK_VAL, HV_VAL, 32>(                           \
        P_buf, q_buf, A_buf, b_buf, stride, C_padded);                        \
}

DEF_SCAN_DOWN_KERNEL(64,  64)
DEF_SCAN_DOWN_KERNEL(128, 128)

#define DEF_CLEAR_ROOT_KERNEL(HK_VAL, HV_VAL)                                  \
extern "C" __global__                                                           \
__launch_bounds__(256, 2)                                                       \
void linear_attn_scan_clear_root_hk##HK_VAL##_hv##HV_VAL(                     \
    float* P_buf, float* q_buf, int C_real, int C_padded                        \
) {                                                                             \
    linear_attn_scan_init_impl<HK_VAL, HV_VAL>(                               \
        P_buf, q_buf, C_real, C_padded);                                       \
}

DEF_CLEAR_ROOT_KERNEL(64,  64)
DEF_CLEAR_ROOT_KERNEL(128, 128)

#define DEF_APPLY_KERNEL(DTYPE_NAME, HK_VAL, HV_VAL, HPG_VAL, T_TYPE)         \
extern "C" __global__                                                           \
__launch_bounds__(256, 2)                                                       \
void linear_attn_apply_##DTYPE_NAME##_hk##HK_VAL##_hv##HV_VAL(                \
    const float* w,  const float* u,  const float* gc,                         \
    const T_TYPE* q_tensor, const T_TYPE* k,                                   \
    const float* state_0, float* new_state,                                    \
    const float* P_buf, const float* q_buf,                                    \
    float* inter, float* vnew,                                                 \
    int C_real, int C_padded                                                    \
) {                                                                             \
    linear_attn_apply_impl<HK_VAL, HV_VAL, 64, HPG_VAL, 16, T_TYPE>(          \
        w, u, gc, q_tensor, k, state_0, new_state, P_buf, q_buf,              \
        inter, vnew, C_real, C_padded);                                        \
}

DEF_APPLY_KERNEL(f32,  64,  64,  16, float)
DEF_APPLY_KERNEL(f32,  128, 128, 64, float)
DEF_APPLY_KERNEL(bf16, 64,  64,  16, __nv_bfloat16)
DEF_APPLY_KERNEL(bf16, 128, 128, 64, __nv_bfloat16)

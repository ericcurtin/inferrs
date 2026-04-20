/// CUDA dispatch for the 3-kernel FLA-style GatedDeltaNet chunked scan.
///
/// Three kernels run sequentially on the same CUDA stream:
///   K1  linear_attn_intra   grid(B*NH*C) — KKT + fwd-subst + WY per chunk
///   K2  linear_attn_state   grid(B*NH)   — sequential state scan, state in regs
///   K3  linear_attn_output  grid(B*NH*C) — tiled qk + matmul per chunk
///
/// Supports F32 and BF16 inputs for q/k/v.  log_g, beta, state are always F32.
/// Output tensors (out, new_state) are always F32.
///
/// All input tensors must be contiguous and shaped as `[B*NH, C, S, dim]`
/// (caller is responsible for reshaping before calling this function).
/// State is `[B*NH, HK, HV]`.
///
/// Returns `(out [B*NH, C, S, HV], new_state [B*NH, HK, HV])` — both F32.
use crate::{op::BackpropOp, DType, Result, Storage, Tensor};

pub fn cuda_linear_attn_scan(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    use candle_kernels as kernels;
    use cudarc::driver::PushKernelArg;

    let cuda_dev = match q.device() {
        crate::Device::Cuda(d) => d.clone(),
        _ => crate::bail!("cuda_linear_attn_scan: requires CUDA device"),
    };

    // q: [b_nh, C, S, HK]
    let (b_nh, c, s, hk) = q.dims4()?;
    let hv = v.dim(3)?;

    if s != 64 {
        crate::bail!(
            "cuda_linear_attn_scan: chunk_size={s} != 64 (only S=64 is supported)"
        );
    }

    let dtype_tag = match q.dtype() {
        DType::F32  => "f32",
        DType::BF16 => "bf16",
        dt => crate::bail!(
            "cuda_linear_attn_scan: unsupported dtype {dt:?} — only F32 or BF16"
        ),
    };

    let (hk_tag, hv_tag) = match (hk, hv) {
        (64,  64)  => ("64",  "64"),
        (128, 128) => ("128", "128"),
        _ => crate::bail!(
            "cuda_linear_attn_scan: unsupported (hk={hk}, hv={hv}) — \
             only (64,64) and (128,128)"
        ),
    };

    let k1_name = format!("linear_attn_intra_{dtype_tag}_hk{hk_tag}_hv{hv_tag}");
    let k2_name = format!("linear_attn_state_{dtype_tag}_hk{hk_tag}_hv{hv_tag}");
    let k3_name = format!("linear_attn_output_{dtype_tag}_hk{hk_tag}_hv{hv_tag}");

    // Shared memory sizes (bytes):
    //   K1: s_attn[S*S] + s_a_row[S] + s_gcsum[S] + s_tile[S*64] + s_tile2[S*64]
    //       = (4096 + 64 + 64 + 4096 + 4096) * 4 = 49664 B
    //   K2: s_row[HK] = HK * 4
    //   K3: s_attn[S*S] + s_q[S*64] + s_k[S*64] + s_gc[S]
    //       = (4096 + 4096 + 4096 + 64) * 4 = 49408 B
    let k1_smem = ((s * s + 2 * s + 2 * s * 64) * std::mem::size_of::<f32>()) as u32; // 64 = BK
    // K2: s_row[HK] + s_partial[256] + s_vnew_cache[S*HV]
    // s_partial has 256 elements always (N_GROUPS * HV = 256).
    // s_vnew_cache caches the full vnew chunk in smem to avoid S global re-reads
    // and S __syncthreads() in Step B.  Total: (128+256+8192)*4 = 34 KB < 48 KB.
    let k2_smem = ((hk + 256 + s * hv) * std::mem::size_of::<f32>()) as u32;
    let k3_smem = ((s * s + 2 * s * 64 + s) * std::mem::size_of::<f32>()) as u32; // 64 = BK=BV

    // Load and configure all three kernels.
    let load_fn = |name: &str, smem: u32| -> Result<_> {
        let func = cuda_dev
            .get_or_load_func(name, &kernels::LINEAR_ATTN_SCAN)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
        if smem > 48 * 1024 {
            func.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                96 * 1024,
            )
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
        }
        Ok((func, smem))
    };
    let (f_k1, smem_k1) = load_fn(&k1_name, k1_smem)?;
    let (f_k2, smem_k2) = load_fn(&k2_name, k2_smem)?;
    let (f_k3, smem_k3) = load_fn(&k3_name, k3_smem)?;

    // ── One workspace for all 5 intermediate F32 buffers ─────────────────────
    // The CUDA driver allocator (cuMemAlloc) is not cached; a single allocation
    // carved with split_at_mut avoids 5 separate round-trips.
    let w_n   = b_nh * c * s * hk;
    let u_n   = b_nh * c * s * hv;
    let gc_n  = b_nh * c * s;
    let ihv_n = b_nh * c * s * hv; // inter and vnew have identical shape
    let mut workspace = unsafe {
        cuda_dev
            .alloc::<f32>(w_n + u_n + gc_n + ihv_n * 2)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?
    };
    // Non-overlapping mutable views — safe because ranges are disjoint.
    let (mut w_v,  mut rest) = workspace.split_at_mut(w_n);
    let (mut u_v,  mut rest) = rest.split_at_mut(u_n);
    let (mut gc_v, mut rest) = rest.split_at_mut(gc_n);
    let (mut inter_v, mut vnew_v) = rest.split_at_mut(ihv_n);

    // ── Output + state buffers (F32) ──────────────────────────────────────────
    let out_buf = unsafe {
        cuda_dev.alloc::<f32>(b_nh * c * s * hv)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?
    };

    // Copy input state into mutable buffer (K2 reads and writes it).
    let state_buf = {
        let (st_stor, st_lay) = state.storage_and_layout();
        let (st_o1, st_o2) = st_lay
            .contiguous_offsets()
            .ok_or_else(|| crate::Error::msg("state not contiguous"))?;
        let src = match &*st_stor {
            Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(st_o1..st_o2),
            _ => crate::bail!("expected Cuda storage for state"),
        };
        let mut buf = unsafe {
            cuda_dev.alloc::<f32>(b_nh * hk * hv)
                .map_err(|e| crate::Error::Cuda(Box::new(e)))?
        };
        cuda_dev
            .memcpy_dtod(&src, &mut buf)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
        buf
    };

    // ── Extract log_g and beta slices (always F32) ────────────────────────────
    let (lg_stor, lg_lay) = log_g.storage_and_layout();
    let (lg_o1, lg_o2) = lg_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("log_g not contiguous"))?;
    let lg_sl = match &*lg_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(lg_o1..lg_o2),
        _ => crate::bail!("expected Cuda storage for log_g"),
    };

    let (bt_stor, bt_lay) = beta.storage_and_layout();
    let (bt_o1, bt_o2) = bt_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("beta not contiguous"))?;
    let bt_sl = match &*bt_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(bt_o1..bt_o2),
        _ => crate::bail!("expected Cuda storage for beta"),
    };

    let c_i = c as i32;

    // ── Dispatch by dtype ─────────────────────────────────────────────────────
    match q.dtype() {
        DType::F32 => {
            let (q_stor, q_lay) = q.storage_and_layout();
            let (q_o1, q_o2) = q_lay
                .contiguous_offsets()
                .ok_or_else(|| crate::Error::msg("q not contiguous"))?;
            let q_sl = match &*q_stor {
                Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(q_o1..q_o2),
                _ => crate::bail!("expected Cuda storage for q"),
            };

            let (k_stor, k_lay) = k.storage_and_layout();
            let (k_o1, k_o2) = k_lay
                .contiguous_offsets()
                .ok_or_else(|| crate::Error::msg("k not contiguous"))?;
            let k_sl = match &*k_stor {
                Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(k_o1..k_o2),
                _ => crate::bail!("expected Cuda storage for k"),
            };

            let (v_stor, v_lay) = v.storage_and_layout();
            let (v_o1, v_o2) = v_lay
                .contiguous_offsets()
                .ok_or_else(|| crate::Error::msg("v not contiguous"))?;
            let v_sl = match &*v_stor {
                Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(v_o1..v_o2),
                _ => crate::bail!("expected Cuda storage for v"),
            };

            // K1: grid=(b_nh*c,), produces w, u, gc
            {
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((b_nh * c) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem_k1,
                };
                let mut b = f_k1.builder();
                b.arg(&q_sl);
                b.arg(&k_sl);
                b.arg(&v_sl);
                b.arg(&lg_sl);
                b.arg(&bt_sl);
                b.arg(&mut w_v);
                b.arg(&mut u_v);
                b.arg(&mut gc_v);
                unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
            }

            // K2: grid=(b_nh,), produces inter, vnew, state_new
            {
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: (b_nh as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem_k2,
                };
                let mut b = f_k2.builder();
                b.arg(&mut w_v);
                b.arg(&mut u_v);
                b.arg(&mut gc_v);
                b.arg(&k_sl);
                b.arg(&q_sl);
                b.arg(&state_buf);
                b.arg(&mut inter_v);
                b.arg(&mut vnew_v);
                b.arg(&c_i);
                unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
            }

            // K3: grid=(b_nh*c,), produces out
            {
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((b_nh * c) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem_k3,
                };
                let mut b = f_k3.builder();
                b.arg(&q_sl);
                b.arg(&k_sl);
                b.arg(&mut vnew_v);
                b.arg(&mut inter_v);
                b.arg(&mut gc_v);
                b.arg(&out_buf);
                unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
            }

            drop(q_stor);
            drop(k_stor);
            drop(v_stor);
        }

        DType::BF16 => {
            let (q_stor, q_lay) = q.storage_and_layout();
            let (q_o1, q_o2) = q_lay
                .contiguous_offsets()
                .ok_or_else(|| crate::Error::msg("q not contiguous"))?;
            let q_sl = match &*q_stor {
                Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(q_o1..q_o2),
                _ => crate::bail!("expected Cuda storage for q"),
            };

            let (k_stor, k_lay) = k.storage_and_layout();
            let (k_o1, k_o2) = k_lay
                .contiguous_offsets()
                .ok_or_else(|| crate::Error::msg("k not contiguous"))?;
            let k_sl = match &*k_stor {
                Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(k_o1..k_o2),
                _ => crate::bail!("expected Cuda storage for k"),
            };

            let (v_stor, v_lay) = v.storage_and_layout();
            let (v_o1, v_o2) = v_lay
                .contiguous_offsets()
                .ok_or_else(|| crate::Error::msg("v not contiguous"))?;
            let v_sl = match &*v_stor {
                Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(v_o1..v_o2),
                _ => crate::bail!("expected Cuda storage for v"),
            };

            // K1
            {
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((b_nh * c) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem_k1,
                };
                let mut b = f_k1.builder();
                b.arg(&q_sl);
                b.arg(&k_sl);
                b.arg(&v_sl);
                b.arg(&lg_sl);
                b.arg(&bt_sl);
                b.arg(&mut w_v);
                b.arg(&mut u_v);
                b.arg(&mut gc_v);
                unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
            }

            // K2
            {
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: (b_nh as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem_k2,
                };
                let mut b = f_k2.builder();
                b.arg(&mut w_v);
                b.arg(&mut u_v);
                b.arg(&mut gc_v);
                b.arg(&k_sl);
                b.arg(&q_sl);
                b.arg(&state_buf);
                b.arg(&mut inter_v);
                b.arg(&mut vnew_v);
                b.arg(&c_i);
                unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
            }

            // K3
            {
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((b_nh * c) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem_k3,
                };
                let mut b = f_k3.builder();
                b.arg(&q_sl);
                b.arg(&k_sl);
                b.arg(&mut vnew_v);
                b.arg(&mut inter_v);
                b.arg(&mut gc_v);
                b.arg(&out_buf);
                unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
            }

            drop(q_stor);
            drop(k_stor);
            drop(v_stor);
        }

        dt => crate::bail!("cuda_linear_attn_scan: unsupported dtype {dt:?}"),
    }

    drop(lg_stor);
    drop(bt_stor);

    // ── Wrap raw buffers into candle tensors ──────────────────────────────────
    let out_tensor = {
        let cs = crate::CudaStorage::wrap_cuda_slice(out_buf, cuda_dev.clone());
        let shape = crate::Shape::from_dims(&[b_nh, c, s, hv]);
        Tensor::from_storage(Storage::Cuda(cs), shape, BackpropOp::none(), false)
    };

    let state_tensor = {
        let cs = crate::CudaStorage::wrap_cuda_slice(state_buf, cuda_dev);
        let shape = crate::Shape::from_dims(&[b_nh, hk, hv]);
        Tensor::from_storage(Storage::Cuda(cs), shape, BackpropOp::none(), false)
    };

    Ok((out_tensor, state_tensor))
}

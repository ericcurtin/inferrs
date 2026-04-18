/// CUDA dispatch for the parallelized GatedDeltaNet chunked scan.
///
/// Kernels run sequentially on the same CUDA stream:
///   K1  linear_attn_intra   grid(B*NH*C) — KKT + fwd-subst + WY per chunk
///   K2a linear_attn_ops     grid(B*NH*C) — compute (A_i, b_i) per chunk
///   K2b linear_attn_scan    variable     — Blelloch prefix scan over chunks
///   K2c linear_attn_apply   grid(B*NH*C_padded) — reconstruct state, compute inter/vnew
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

fn next_power_of_2(n: usize) -> usize {
    if n <= 1 {
        return n;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

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

    let c_padded = next_power_of_2(c);
    let c_padded_i = c_padded as i32;
    let c_real_i = c as i32;

    let k1_name = format!("linear_attn_intra_{dtype_tag}_hk{hk_tag}_hv{hv_tag}");
    let k2a_name = format!("linear_attn_ops_{dtype_tag}_hk{hk_tag}_hv{hv_tag}");
    let k2b_up_name = format!("linear_attn_scan_up_hk{hk_tag}_hv{hv_tag}");
    let k2b_down_name = format!("linear_attn_scan_down_hk{hk_tag}_hv{hv_tag}");
    let k2b_clear_name = format!("linear_attn_scan_clear_root_hk{hk_tag}_hv{hv_tag}");
    let k2c_name = format!("linear_attn_apply_{dtype_tag}_hk{hk_tag}_hv{hv_tag}");
    let k3_name = format!("linear_attn_output_{dtype_tag}_hk{hk_tag}_hv{hv_tag}");

    let k1_smem = ((s * s + 2 * s + 2 * s * 64) * std::mem::size_of::<f32>()) as u32;
    let k2a_smem = ((32 * s + s * 32 + s) * std::mem::size_of::<f32>()) as u32;
    // Up-sweep needs an extra s_pr[BK*HK] buffer (BK=32) to cache P_right row-blocks
    // and avoid the in-place read-write conflict in the P composition tiling.
    let k2b_up_smem = ((2 * 32 * 32 + 32 * hk) * std::mem::size_of::<f32>()) as u32;
    let k2b_down_smem = (2 * 32 * 32 * std::mem::size_of::<f32>()) as u32;

    let k2c_smem = ((hk * 16 + 16 * hv + s + hk + 256) * std::mem::size_of::<f32>()) as u32;
    let k3_smem = ((s * s + 2 * s * 64 + s) * std::mem::size_of::<f32>()) as u32;

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
    let (f_k2a, smem_k2a) = load_fn(&k2a_name, k2a_smem)?;
    let (f_k2b_up, smem_k2b_up) = load_fn(&k2b_up_name, k2b_up_smem)?;
    let (f_k2b_down, smem_k2b_down) = load_fn(&k2b_down_name, k2b_down_smem)?;
    let (f_k2b_clear, _smem_k2b_clear) = load_fn(&k2b_clear_name, 0)?;
    let (f_k2c, smem_k2c) = load_fn(&k2c_name, k2c_smem)?;
    let (f_k3, smem_k3) = load_fn(&k3_name, k3_smem)?;

    // ── Workspace for K1 intermediates ─────────────────────────────────────
    let w_n   = b_nh * c * s * hk;
    let u_n   = b_nh * c * s * hv;
    let gc_n  = b_nh * c * s;
    let ihv_n = b_nh * c * s * hv;
    let mut workspace = unsafe {
        cuda_dev
            .alloc::<f32>(w_n + u_n + gc_n + ihv_n * 2)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?
    };
    let (mut w_v,  mut rest) = workspace.split_at_mut(w_n);
    let (mut u_v,  mut rest) = rest.split_at_mut(u_n);
    let (mut gc_v, mut rest) = rest.split_at_mut(gc_n);
    let (mut inter_v, mut vnew_v) = rest.split_at_mut(ihv_n);

    // ── Prefix scan buffers (padded to power of 2) ─────────────────────────
    let p_buf_n = b_nh * c_padded * hk * hk;
    let q_buf_n = b_nh * c_padded * hk * hv;
    let mut p_buf = unsafe {
        cuda_dev.alloc::<f32>(p_buf_n).map_err(|e| crate::Error::Cuda(Box::new(e)))?
    };
    let mut q_prefix_buf = unsafe {
        cuda_dev.alloc::<f32>(q_buf_n).map_err(|e| crate::Error::Cuda(Box::new(e)))?
    };
    let mut a_buf = unsafe {
        cuda_dev.alloc::<f32>(p_buf_n).map_err(|e| crate::Error::Cuda(Box::new(e)))?
    };
    let mut b_buf = unsafe {
        cuda_dev.alloc::<f32>(q_buf_n).map_err(|e| crate::Error::Cuda(Box::new(e)))?
    };

    // ── Output + state buffers (F32) ──────────────────────────────────────────
    let out_buf = unsafe {
        cuda_dev.alloc::<f32>(b_nh * c * s * hv)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?
    };

    // state_0_buf: read-only view of the initial state, shared by all K2c blocks.
    // new_state_buf: written exclusively by the ci==C_real-1 block — keeping these
    // separate eliminates the concurrent read-write data race that would arise if
    // all blocks shared a single state buffer (the last-chunk block could overwrite
    // state_0 before sibling blocks have finished reading it in Step 1).
    let state_0_buf = {
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
    let mut new_state_buf = unsafe {
        cuda_dev.alloc::<f32>(b_nh * hk * hv)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?
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
            let _v_sl = match &*v_stor {
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
                b.arg(&_v_sl);
                b.arg(&lg_sl);
                b.arg(&bt_sl);
                b.arg(&mut w_v);
                b.arg(&mut u_v);
                b.arg(&mut gc_v);
                unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
            }

            // K2a: grid=(b_nh*c,), computes A_i, b_i per chunk
            {
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((b_nh * c) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem_k2a,
                };
                let mut b = f_k2a.builder();
                b.arg(&mut w_v);
                b.arg(&mut u_v);
                b.arg(&mut gc_v);
                b.arg(&k_sl);
                b.arg(&mut p_buf);
                b.arg(&mut q_prefix_buf);
                b.arg(&c_real_i);
                b.arg(&c_padded_i);
                unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
            }

            // K2b: Blelloch prefix scan
            {
                // Up-sweep
                let mut stride = 2usize;
                while stride <= c_padded {
                    let n_pairs = b_nh * (c_padded / stride);
                    let cfg = cudarc::driver::LaunchConfig {
                        grid_dim: (n_pairs as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: smem_k2b_up,
                    };
                    let mut b = f_k2b_up.builder();
                    b.arg(&mut p_buf);
                    b.arg(&mut q_prefix_buf);
                    b.arg(&mut a_buf);
                    b.arg(&mut b_buf);
                    let stride_i = stride as i32; b.arg(&stride_i);
                    b.arg(&c_padded_i);
                    unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                    stride <<= 1;
                }

                // Clear root: P[C_padded-1] = I, q[C_padded-1] = 0
                {
                    let cfg = cudarc::driver::LaunchConfig {
                        grid_dim: (b_nh as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b = f_k2b_clear.builder();
                    b.arg(&mut p_buf);
                    b.arg(&mut q_prefix_buf);
                    b.arg(&c_real_i);
                    b.arg(&c_padded_i);
                    unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                }

                // Down-sweep
                stride = c_padded;
                while stride >= 2 {
                    let n_pairs = b_nh * (c_padded / stride);
                    let cfg = cudarc::driver::LaunchConfig {
                        grid_dim: (n_pairs as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: smem_k2b_down,
                    };
                    let mut b = f_k2b_down.builder();
                    b.arg(&mut p_buf);
                    b.arg(&mut q_prefix_buf);
                    b.arg(&a_buf);
                    b.arg(&b_buf);
                    let stride_i = stride as i32; b.arg(&stride_i);
                    b.arg(&c_padded_i);
                    unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                    stride >>= 1;
                }
            }

            // K2c: grid=(b_nh * c_padded,), reconstructs state, computes inter/vnew
            {
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((b_nh * c_padded) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem_k2c,
                };
                let mut b = f_k2c.builder();
                b.arg(&mut w_v);
                b.arg(&mut u_v);
                b.arg(&mut gc_v);
                b.arg(&q_sl);
                b.arg(&k_sl);
                b.arg(&state_0_buf);
                b.arg(&mut new_state_buf);
                b.arg(&p_buf);
                b.arg(&q_prefix_buf);
                b.arg(&mut inter_v);
                b.arg(&mut vnew_v);
                b.arg(&c_real_i);
                b.arg(&c_padded_i);
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
            let _v_sl = match &*v_stor {
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
                b.arg(&_v_sl);
                b.arg(&lg_sl);
                b.arg(&bt_sl);
                b.arg(&mut w_v);
                b.arg(&mut u_v);
                b.arg(&mut gc_v);
                unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
            }

            // K2a
            {
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((b_nh * c) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem_k2a,
                };
                let mut b = f_k2a.builder();
                b.arg(&mut w_v);
                b.arg(&mut u_v);
                b.arg(&mut gc_v);
                b.arg(&k_sl);
                b.arg(&mut p_buf);
                b.arg(&mut q_prefix_buf);
                b.arg(&c_real_i);
                b.arg(&c_padded_i);
                unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
            }

            // K2b: Blelloch prefix scan
            {
                let mut stride = 2usize;
                while stride <= c_padded {
                    let n_pairs = b_nh * (c_padded / stride);
                    let cfg = cudarc::driver::LaunchConfig {
                        grid_dim: (n_pairs as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: smem_k2b_up,
                    };
                    let mut b = f_k2b_up.builder();
                    b.arg(&mut p_buf);
                    b.arg(&mut q_prefix_buf);
                    b.arg(&mut a_buf);
                    b.arg(&mut b_buf);
                    let stride_i = stride as i32; b.arg(&stride_i);
                    b.arg(&c_padded_i);
                    unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                    stride <<= 1;
                }

                {
                    let cfg = cudarc::driver::LaunchConfig {
                        grid_dim: (b_nh as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b = f_k2b_clear.builder();
                    b.arg(&mut p_buf);
                    b.arg(&mut q_prefix_buf);
                    b.arg(&c_real_i);
                    b.arg(&c_padded_i);
                    unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                }

                stride = c_padded;
                while stride >= 2 {
                    let n_pairs = b_nh * (c_padded / stride);
                    let cfg = cudarc::driver::LaunchConfig {
                        grid_dim: (n_pairs as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: smem_k2b_down,
                    };
                    let mut b = f_k2b_down.builder();
                    b.arg(&mut p_buf);
                    b.arg(&mut q_prefix_buf);
                    b.arg(&a_buf);
                    b.arg(&b_buf);
                    let stride_i = stride as i32; b.arg(&stride_i);
                    b.arg(&c_padded_i);
                    unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
                    stride >>= 1;
                }
            }

            // K2c
            {
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((b_nh * c_padded) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: smem_k2c,
                };
                let mut b = f_k2c.builder();
                b.arg(&mut w_v);
                b.arg(&mut u_v);
                b.arg(&mut gc_v);
                b.arg(&q_sl);
                b.arg(&k_sl);
                b.arg(&state_0_buf);
                b.arg(&mut new_state_buf);
                b.arg(&p_buf);
                b.arg(&q_prefix_buf);
                b.arg(&mut inter_v);
                b.arg(&mut vnew_v);
                b.arg(&c_real_i);
                b.arg(&c_padded_i);
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

    let out_tensor = {
        let cs = crate::CudaStorage::wrap_cuda_slice(out_buf, cuda_dev.clone());
        let shape = crate::Shape::from_dims(&[b_nh, c, s, hv]);
        Tensor::from_storage(Storage::Cuda(cs), shape, BackpropOp::none(), false)
    };

    let state_tensor = {
        let cs = crate::CudaStorage::wrap_cuda_slice(new_state_buf, cuda_dev);
        let shape = crate::Shape::from_dims(&[b_nh, hk, hv]);
        Tensor::from_storage(Storage::Cuda(cs), shape, BackpropOp::none(), false)
    };

    Ok((out_tensor, state_tensor))
}

/// CUDA dispatch for the monolithic GatedDeltaNet chunked scan kernel.
///
/// Calls `gated_delta_net_scan_f32_hkHK_hvHV` from
/// `candle-kernels/src/linear_attn_scan.cu`.
///
/// All input tensors must be F32 and contiguous, shaped as `[B*NH, C, S, dim]`
/// (caller is responsible for reshaping before calling this function).
/// State is `[B*NH, HK, HV]`.
///
/// Returns `(out [B*NH, C, S, HV], new_state [B*NH, HK, HV])` as new F32 tensors.
///
/// Shared memory exceeds 48 KB for all supported configs; the kernel opts in to
/// the 96 KB carveout via `set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)`.
use crate::{op::BackpropOp, Result, Storage, Tensor};

/// Launch the monolithic scan kernel.
///
/// Arguments (all F32, contiguous):
///   q, k : `[b_nh, C, S, HK]`
///   v    : `[b_nh, C, S, HV]`
///   log_g: `[b_nh, C, S]`
///   beta : `[b_nh, C, S]`
///   state: `[b_nh, HK, HV]`
///
/// Returns `(out [b_nh, C, S, HV], new_state [b_nh, HK, HV])`.
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

    let kernel_name = match (hk, hv) {
        (64, 64) => "gated_delta_net_scan_f32_hk64_hv64",
        (128, 128) => "gated_delta_net_scan_f32_hk128_hv128",
        _ => crate::bail!(
            "cuda_linear_attn_scan: unsupported (hk={hk}, hv={hv}) — only (64,64) and (128,128)"
        ),
    };

    // Shared memory: (S*S + 3*S + S*HK + S*HV) * sizeof(float)
    let shared_bytes =
        ((s * s + 3 * s + s * hk + s * hv) * std::mem::size_of::<f32>()) as u32;

    let func = cuda_dev
        .get_or_load_func(kernel_name, &kernels::LINEAR_ATTN_SCAN)
        .map_err(|e| crate::Error::Cuda(Box::new(e)))?;

    // Opt-in to 96 KB dynamic shared memory (required for all supported configs).
    func.set_attribute(
        cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        96 * 1024,
    )
    .map_err(|e| crate::Error::Cuda(Box::new(e)))?;

    // ── Extract read-only F32 slices (pattern from cuda_flash_attn.rs) ────────
    // Each guard (`_q_s`, `_k_s`, ...) must be held until after the kernel
    // launch so the underlying slice remains valid.

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

    // ── Allocate output buffer ────────────────────────────────────────────────
    let out_buf = unsafe {
        cuda_dev
            .alloc::<f32>(b_nh * c * s * hv)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?
    };

    // ── Copy input state into a fresh mutable buffer ──────────────────────────
    // The kernel reads and writes state in-place across chunks.
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
            cuda_dev
                .alloc::<f32>(b_nh * hk * hv)
                .map_err(|e| crate::Error::Cuda(Box::new(e)))?
        };
        cuda_dev
            .memcpy_dtod(&src, &mut buf)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?;
        buf // st_stor guard dropped here; memcpy is on the same stream so it's safe
    };

    // ── Launch ────────────────────────────────────────────────────────────────
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (b_nh as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    let c_i = c as i32;
    {
        let mut b = func.builder();
        b.arg(&q_sl);
        b.arg(&k_sl);
        b.arg(&v_sl);
        b.arg(&lg_sl);
        b.arg(&bt_sl);
        b.arg(&state_buf);
        b.arg(&out_buf);
        b.arg(&c_i);
        unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
    }

    drop(q_stor);
    drop(k_stor);
    drop(v_stor);
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

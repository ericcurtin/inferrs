/// CUDA fused decay gate for GatedDeltaNet SSM layers.
///
/// Dispatches `compute_decay_gate_{f32, bf16f32}` from
/// `candle-kernels/linear_attn.cu`.  Mirrors the Metal implementation at
/// `candle-metal-kernels/metal_src/linear_attn.metal`.
///
/// Computes: `g[i] = exp(-a_exp[h] * softplus(a_input[i] + dt_bias[h]))`
/// where `h = i % n_heads`, `softplus(x) = max(x,0) + log(1 + exp(-|x|))`.
///
/// # Arguments
/// * `a_input` — `[..., n_heads]`, F32 or BF16, contiguous, zero offset.
/// * `dt_bias` — `[n_heads]`, F32, contiguous, zero offset.
/// * `a_exp`   — `[n_heads]`, F32, contiguous, zero offset (`A_log.exp()` precomputed).
///
/// # Returns
/// An F32 tensor with the same shape as `a_input`.
///
/// # Errors
/// Fails if any input is not on CUDA, not contiguous, has a non-zero offset,
/// or has a dtype outside the supported set.
use crate::{op::BackpropOp, DType, Result, Storage, Tensor};

pub fn compute_decay_gate_cuda(
    a_input: &Tensor,
    dt_bias: &Tensor,
    a_exp: &Tensor,
) -> Result<Tensor> {
    use candle_kernels as kernels;
    use cudarc::driver::PushKernelArg;

    let cuda_dev = match a_input.device() {
        crate::Device::Cuda(d) => d.clone(),
        _ => crate::bail!("compute_decay_gate_cuda requires CUDA device"),
    };

    let bf16_input = match a_input.dtype() {
        DType::BF16 => true,
        DType::F32 => false,
        dt => crate::bail!(
            "compute_decay_gate_cuda: expected BF16 or F32 a_input, got {:?}",
            dt
        ),
    };
    if dt_bias.dtype() != DType::F32 || a_exp.dtype() != DType::F32 {
        crate::bail!("compute_decay_gate_cuda: dt_bias and a_exp must be F32");
    }

    if !a_input.is_contiguous() || !dt_bias.is_contiguous() || !a_exp.is_contiguous() {
        crate::bail!("compute_decay_gate_cuda: all inputs must be contiguous");
    }

    let dims = a_input.dims();
    if dims.is_empty() {
        crate::bail!("compute_decay_gate_cuda: a_input must have at least 1 dim");
    }
    let n_heads_usize = dims[dims.len() - 1];
    let n_total_usize = a_input.elem_count();

    // Empty input (any dim == 0): skip the kernel launch — grid_dim = 0 is an
    // invalid CUDA launch config — and hand back a zero-element tensor with the
    // same shape.
    if n_total_usize == 0 {
        return Tensor::zeros(a_input.shape(), DType::F32, a_input.device());
    }

    // The kernel indexes elements with u32. Realistic decay-gate tensors are
    // small (b × seq × n_heads ≪ 2³² for any sane config), but bail explicitly
    // rather than silently truncate if the invariant ever breaks.
    if n_total_usize > u32::MAX as usize {
        crate::bail!(
            "compute_decay_gate_cuda: n_total ({}) exceeds u32::MAX; kernel indexing would overflow",
            n_total_usize
        );
    }
    let n_heads = n_heads_usize as u32;
    let n_total = n_total_usize as u32;

    // Extract CUDA slices. Hold the read-guards for the duration of the launch.
    let (a_stor, a_lay) = a_input.storage_and_layout();
    let (a_o1, a_o2) = a_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("a_input not contiguous"))?;
    if a_o1 != 0 {
        crate::bail!("compute_decay_gate_cuda: a_input must have zero offset");
    }

    let (dt_stor, dt_lay) = dt_bias.storage_and_layout();
    let (dt_o1, dt_o2) = dt_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("dt_bias not contiguous"))?;
    if dt_o1 != 0 {
        crate::bail!("compute_decay_gate_cuda: dt_bias must have zero offset");
    }

    let (ae_stor, ae_lay) = a_exp.storage_and_layout();
    let (ae_o1, ae_o2) = ae_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("a_exp not contiguous"))?;
    if ae_o1 != 0 {
        crate::bail!("compute_decay_gate_cuda: a_exp must have zero offset");
    }

    let dt_slice = match &*dt_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(dt_o1..dt_o2),
        _ => crate::bail!("expected Cuda storage for dt_bias"),
    };
    let ae_slice = match &*ae_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(ae_o1..ae_o2),
        _ => crate::bail!("expected Cuda storage for a_exp"),
    };

    let out_buf = unsafe {
        cuda_dev
            .alloc::<f32>(n_total as usize)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?
    };

    // grid = ceil(n_total / 256), block = 256
    const BLOCK: u32 = 256;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (n_total.div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let kernel_name = if bf16_input {
        "compute_decay_gate_bf16f32"
    } else {
        "compute_decay_gate_f32"
    };
    let func = cuda_dev
        .get_or_load_func(kernel_name, &kernels::LINEAR_ATTN)
        .map_err(|e| crate::Error::Cuda(Box::new(e)))?;

    if bf16_input {
        let a_slice = match &*a_stor {
            Storage::Cuda(cs) => cs.as_cuda_slice::<half::bf16>()?.slice(a_o1..a_o2),
            _ => crate::bail!("expected Cuda storage for a_input"),
        };
        let mut b = func.builder();
        b.arg(&a_slice);
        b.arg(&dt_slice);
        b.arg(&ae_slice);
        b.arg(&out_buf);
        b.arg(&n_heads);
        b.arg(&n_total);
        unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
    } else {
        let a_slice = match &*a_stor {
            Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(a_o1..a_o2),
            _ => crate::bail!("expected Cuda storage for a_input"),
        };
        let mut b = func.builder();
        b.arg(&a_slice);
        b.arg(&dt_slice);
        b.arg(&ae_slice);
        b.arg(&out_buf);
        b.arg(&n_heads);
        b.arg(&n_total);
        unsafe { b.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;
    }

    drop(a_stor);
    drop(dt_stor);
    drop(ae_stor);

    let out_cs = crate::CudaStorage::wrap_cuda_slice(out_buf, cuda_dev);
    Ok(Tensor::from_storage(
        Storage::Cuda(out_cs),
        a_input.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

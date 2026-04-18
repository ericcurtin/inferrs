/// CUDA dispatch for depthwise conv1d in BTC ([b, t, c]) layout.
///
/// Avoids the two `transpose + contiguous` copies that the standard BCL path requires.
/// The caller is responsible for pre-padding the input (causal state concatenation).
///
/// padded: [b, l_in, c] contiguous F32 — already includes causal pad tokens.
/// kernel: [c, 1, k_size] F32 — standard depthwise conv weight (squeezed internally).
///
/// Output: [b, l_out, c] contiguous F32, where l_out = l_in - k_size + 1.
use crate::{op::BackpropOp, DType, Result, Storage, Tensor};
use cudarc::driver::{LaunchConfig, PushKernelArg};

pub fn cuda_conv1d_depthwise_btc(padded: &Tensor, kernel: &Tensor) -> Result<Tensor> {
    use candle_kernels as kernels;

    let (b, l_in, c) = padded.dims3()?;
    let (c_k, _one, k_size) = kernel.dims3()?;
    if c != c_k {
        crate::bail!(
            "cuda_conv1d_depthwise_btc: channel mismatch padded c={c} kernel c_out={c_k}"
        );
    }
    if padded.dtype() != DType::F32 {
        crate::bail!(
            "cuda_conv1d_depthwise_btc: only F32 supported, got {:?}",
            padded.dtype()
        );
    }

    let l_out = l_in
        .checked_sub(k_size - 1)
        .ok_or_else(|| crate::Error::msg("cuda_conv1d_depthwise_btc: l_in < k_size - 1"))?;

    let cuda_dev = match padded.device() {
        crate::Device::Cuda(d) => d.clone(),
        _ => crate::bail!("cuda_conv1d_depthwise_btc: requires CUDA device"),
    };

    // Squeeze kernel [c, 1, k_size] → [c, k_size]; contiguous() is a no-op if already so.
    let kernel_sq = kernel.squeeze(1)?.contiguous()?;

    let (pad_stor, pad_lay) = padded.storage_and_layout();
    let (pad_o1, pad_o2) = pad_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("cuda_conv1d_depthwise_btc: padded not contiguous"))?;
    let pad_sl = match &*pad_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(pad_o1..pad_o2),
        _ => crate::bail!("cuda_conv1d_depthwise_btc: expected Cuda storage for padded"),
    };

    let (ker_stor, ker_lay) = kernel_sq.storage_and_layout();
    let (ker_o1, ker_o2) = ker_lay
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::msg("cuda_conv1d_depthwise_btc: kernel not contiguous"))?;
    let ker_sl = match &*ker_stor {
        Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(ker_o1..ker_o2),
        _ => crate::bail!("cuda_conv1d_depthwise_btc: expected Cuda storage for kernel"),
    };

    let dst_el = b * l_out * c;
    let out_buf = unsafe {
        cuda_dev
            .alloc::<f32>(dst_el)
            .map_err(|e| crate::Error::Cuda(Box::new(e)))?
    };

    let block_dim: u32 = 256;
    let n_blocks = dst_el.div_ceil(block_dim as usize) as u32;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = cuda_dev
        .get_or_load_func("conv1d_depthwise_btc_f32", &kernels::CONV)
        .map_err(|e| crate::Error::Cuda(Box::new(e)))?;

    // Strides for [b, l_in, c] contiguous layout.
    let src_b_stride: usize = l_in * c;
    let src_t_stride: usize = c;
    let src_c_stride: usize = 1usize;
    // stride=1, padding=0, dilation=1: caller pre-pads, no internal padding needed.
    let stride: usize = 1usize;
    let padding: usize = 0usize;
    let dilation: usize = 1usize;

    let mut builder = func.builder();
    builder.arg(&b);
    builder.arg(&c);
    builder.arg(&l_in);
    builder.arg(&l_out);
    builder.arg(&k_size);
    builder.arg(&stride);
    builder.arg(&padding);
    builder.arg(&dilation);
    builder.arg(&src_b_stride);
    builder.arg(&src_t_stride);
    builder.arg(&src_c_stride);
    builder.arg(&pad_sl);
    builder.arg(&ker_sl);
    builder.arg(&out_buf);
    unsafe { builder.launch(cfg) }.map_err(|e| crate::Error::Cuda(Box::new(e)))?;

    let cs = crate::CudaStorage::wrap_cuda_slice(out_buf, cuda_dev);
    let shape = crate::Shape::from_dims(&[b, l_out, c]);
    Ok(Tensor::from_storage(
        Storage::Cuda(cs),
        shape,
        BackpropOp::none(),
        false,
    ))
}

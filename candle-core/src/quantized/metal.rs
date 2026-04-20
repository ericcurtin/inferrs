use super::{GgmlDType, QStorage};
use crate::backend::BackendStorage;
use crate::{DType, MetalDevice, MetalStorage, Result, Shape, D};
use candle_metal_kernels::metal::Buffer;
use std::sync::Arc;

pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
    buffer: Arc<Buffer>,
    /// Byte offset within `buffer` where this tensor's data starts.
    /// For standalone allocations (legacy path), this is always 0.
    offset: usize,
}

impl QMetalStorage {
    pub fn zeros(device: &MetalDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        let buffer = device.allocate_zeros(size)?;
        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
            offset: 0,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<MetalStorage> {
        use crate::quantized::k_quants::GgmlType;

        let buffer = self.device.allocate_buffer(self.buffer.length())?;
        let blit = self.device.blit_command_encoder()?;
        blit.set_label("blit_to_cpu");
        blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        blit.end_encoding();
        self.device.wait_until_completed()?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => {
                let vec: Vec<f32> = read_to_vec(&buffer, block_len);
                f32::to_float(&vec, &mut out);
            }
            GgmlDType::F16 => {
                let vec: Vec<half::f16> = read_to_vec(&buffer, block_len);
                half::f16::to_float(&vec, &mut out);
            }
            GgmlDType::BF16 => {
                let vec: Vec<half::bf16> = read_to_vec(&buffer, block_len);
                half::bf16::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_0 => {
                let vec: Vec<crate::quantized::BlockQ4_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_1 => {
                let vec: Vec<crate::quantized::BlockQ4_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_0 => {
                let vec: Vec<crate::quantized::BlockQ5_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_1 => {
                let vec: Vec<crate::quantized::BlockQ5_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_0 => {
                let vec: Vec<crate::quantized::BlockQ8_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_1 => {
                let vec: Vec<crate::quantized::BlockQ8_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q2K => {
                let vec: Vec<crate::quantized::BlockQ2K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ2K::to_float(&vec, &mut out);
            }
            GgmlDType::Q3K => {
                let vec: Vec<crate::quantized::BlockQ3K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ3K::to_float(&vec, &mut out);
            }
            GgmlDType::Q4K => {
                let vec: Vec<crate::quantized::BlockQ4K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4K::to_float(&vec, &mut out);
            }
            GgmlDType::Q5K => {
                let vec: Vec<crate::quantized::BlockQ5K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5K::to_float(&vec, &mut out);
            }
            GgmlDType::Q6K => {
                let vec: Vec<crate::quantized::BlockQ6K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ6K::to_float(&vec, &mut out);
            }
            GgmlDType::Q8K => {
                let vec: Vec<crate::quantized::BlockQ8K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8K::to_float(&vec, &mut out);
            }
            GgmlDType::TQ2_0 => {
                let vec: Vec<crate::quantized::BlockTq2_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockTq2_0::to_float(&vec, &mut out);
            }
        }

        let buffer = self.device.new_buffer_with_data(&out)?;
        Ok(MetalStorage::new(
            buffer,
            self.device.clone(),
            elem_count,
            DType::F32,
        ))
    }

    pub fn quantize(&mut self, src: &MetalStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &MetalStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.buffer.length()
    }

    fn fwd_mv(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();

        // We always use a single batch dimension and stack all the tensors in the batch on the
        // second dimension as the implementation in candle-metal-kernels doesn't handle batch
        // properly.
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            n => crate::bail!("Invalid rank {n} for quantized matmul metal"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul")?;
        let encoder = device.command_encoder()?;
        // In some cases it would be better to use the mm variant, though it has its drawbacks
        // around memory alignment.
        for batch_id in 0..m {
            candle_metal_kernels::call_quantized_matmul_mv_t(
                device.device(),
                &encoder,
                device.kernels(),
                self.dtype.into(),
                (1, 1, n, k),
                storage.buffer(),
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes(),
                &self.buffer,
                self.offset,
                batch_id * n * DType::F32.size_in_bytes(),
                &dst,
            )
            .map_err(MetalError::from)?;
        }
        let dst_storage = crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    /// Q4K GEMV with F32 input and BF16 output.
    /// Uses `kernel_mul_mv_q4_K_f32_bf16o` — eliminates F32→BF16 to_dtype after down_proj/PLI.
    pub fn fwd_mv_q4k_bf16o(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q4K {
            crate::bail!("fwd_mv_q4k_bf16o only supports Q4K weights");
        }
        if storage.dtype() != DType::F32 {
            crate::bail!("fwd_mv_q4k_bf16o requires F32 input");
        }
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv_q4k_bf16o: input not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            crate::bail!("fwd_mv_q4k_bf16o: input has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            r => crate::bail!("fwd_mv_q4k_bf16o: invalid rank {r}"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv_q4k_bf16o: input incompatible with weight shape")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::BF16, "qmatmul_q4k_bf16o")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_offset = (layout.start_offset() + batch_id * k) * DType::F32.size_in_bytes();
            let dst_offset = batch_id * n * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16o(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                lhs_offset,
                &self.buffer,
                self.offset,
                dst_offset,
                &dst,
            )
            .map_err(MetalError::from)?;
        }
        let dst_storage =
            crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::BF16);
        Ok((dst_storage, dst_shape))
    }

    /// Q4K GEMV: F32 input → BF16 output, writing into a pre-allocated destination buffer.
    /// Returns `true` if the dispatch succeeded; caller reuses `dst`.
    /// Saves 1 Metal buffer allocation per decode step vs `fwd_mv_q4k_bf16o`.
    pub fn fwd_mv_q4k_bf16o_prealloc(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        dst: &Buffer,
        dst_offset_bytes: usize,
    ) -> Result<bool> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q4K {
            return Ok(false);
        }
        if storage.dtype() != DType::F32 {
            return Ok(false);
        }
        if !layout.is_contiguous() {
            return Ok(false);
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            return Ok(false);
        }
        let (n, k) = self_shape.dims2()?;
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        let device = storage.device();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_offset = (layout.start_offset() + batch_id * k) * DType::F32.size_in_bytes();
            let d_off = dst_offset_bytes + batch_id * n * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16o(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                lhs_offset,
                &self.buffer,
                self.offset,
                d_off,
                dst,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    /// Q4K GEMV v2: float4 dequantization + dot() hardware instruction.
    /// Accepts F32 input, outputs F32. Uses `kernel_mul_mv_q4_K_f32_v2`.
    /// Alternative to standard Q4K GEMV — may be faster due to vectorized dot().
    pub fn fwd_mv_q4k_v2(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q4K {
            crate::bail!("fwd_mv_q4k_v2: Q4K only");
        }
        if storage.dtype() != DType::F32 {
            crate::bail!("fwd_mv_q4k_v2: F32 input required");
        }
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv_q4k_v2: input not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            crate::bail!("fwd_mv_q4k_v2: input rank < 2")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            r => crate::bail!("fwd_mv_q4k_v2: invalid rank {r}"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv_q4k_v2: shape mismatch")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul_q4k_v2")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_offset = (layout.start_offset() + batch_id * k) * DType::F32.size_in_bytes();
            let dst_offset = batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q4k_v2(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                lhs_offset,
                &self.buffer,
                self.offset,
                dst_offset,
                &dst,
            )
            .map_err(MetalError::from)?;
        }
        let dst_storage = crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    /// Q4K GEMV (F32 input) with in-place scaled accumulate into a pre-allocated BF16 buffer.
    ///
    /// Computes: `result_bf16[i] += w * GEMV(self, xs_f32)[i]` in one Metal dispatch.
    ///
    /// Eliminates 2 dispatches vs: `out = GEMV(self, xs)`, `result += w * out`
    /// (scale + accumulate). For MoE expert dispatch (8 experts × N layers), this
    /// saves 2 × 8 × N dispatches per decode step.
    ///
    /// Returns `true` if the kernel was dispatched, `false` if preconditions failed.
    pub fn fwd_mv_q4k_f32_saxpy_bf16(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        result: &crate::MetalStorage,
        result_offset_bytes: usize,
        w: f32,
    ) -> Result<bool> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q4K {
            return Ok(false);
        }
        if storage.dtype() != DType::F32 {
            return Ok(false);
        }
        if result.dtype() != DType::BF16 {
            return Ok(false);
        }
        if !layout.is_contiguous() {
            return Ok(false);
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            return Ok(false);
        }
        let (n, k) = self_shape.dims2()?;
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        let device = storage.device().clone();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let rhs_offset = (layout.start_offset() + batch_id * k) * DType::F32.size_in_bytes();
            let result_off = result_offset_bytes + batch_id * n * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q4k_f32_saxpy_bf16(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                rhs_offset,
                &self.buffer,
                self.offset,
                result.buffer(),
                result_off,
                w,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    /// Q4K GEMV (BF16 input) with in-place scaled accumulate into a pre-allocated BF16 buffer.
    ///
    /// Computes: `result_bf16[i] += w * GEMV(self, xs_bf16)[i]` in one Metal dispatch.
    ///
    /// Same as `fwd_mv_q4k_f32_saxpy_bf16` but takes BF16 input directly, eliminating
    /// the BF16→F32 cast dispatch before SAXPY. Saves 1 dispatch per expert per MoE layer
    /// (8 experts × 35 layers = 280 dispatches per decode step).
    ///
    /// Returns `true` if the kernel was dispatched, `false` if preconditions failed.
    pub fn fwd_mv_q4k_bf16i_saxpy_bf16(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        result: &crate::MetalStorage,
        result_offset_bytes: usize,
        w: f32,
    ) -> Result<bool> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q4K {
            return Ok(false);
        }
        if storage.dtype() != DType::BF16 {
            return Ok(false);
        }
        if result.dtype() != DType::BF16 {
            return Ok(false);
        }
        if !layout.is_contiguous() {
            return Ok(false);
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            return Ok(false);
        }
        let (n, k) = self_shape.dims2()?;
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        let device = storage.device().clone();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let rhs_offset = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let result_off = result_offset_bytes + batch_id * n * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16i_saxpy_bf16(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                rhs_offset,
                &self.buffer,
                self.offset,
                result.buffer(),
                result_off,
                w,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    /// Q8_0 GEMV with F32 input and BF16 output.
    /// Uses `kernel_mul_mv_q8_0_f32_to_bf16` — eliminates the F32→BF16 `to_dtype`
    /// dispatch after down_proj / pli_projection.
    pub fn fwd_mv_q8_0_bf16o(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q8_0 {
            crate::bail!("fwd_mv_q8_0_bf16o only supports Q8_0 weights");
        }
        if storage.dtype() != DType::F32 {
            crate::bail!("fwd_mv_q8_0_bf16o requires F32 input");
        }
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv_q8_0_bf16o: input not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            crate::bail!("fwd_mv_q8_0_bf16o: input has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            r => crate::bail!("fwd_mv_q8_0_bf16o: invalid rank {r}"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv_q8_0_bf16o: input incompatible with weight shape")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::BF16, "qmatmul_q8_bf16o")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_offset = (layout.start_offset() + batch_id * k) * DType::F32.size_in_bytes();
            let dst_offset = batch_id * n * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q8_0_bf16o(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                lhs_offset,
                &self.buffer,
                self.offset,
                dst_offset,
                &dst,
            )
            .map_err(MetalError::from)?;
        }
        let dst_storage =
            crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::BF16);
        Ok((dst_storage, dst_shape))
    }

    /// Q8_0 GEMV: F32 input → BF16 output, into a pre-allocated BF16 buffer.
    /// Eliminates the Metal buffer allocation for down_proj per decode step.
    pub fn fwd_mv_q8_0_bf16o_prealloc(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        dst: &Buffer,
        dst_offset_bytes: usize,
    ) -> Result<bool> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q8_0 {
            return Ok(false);
        }
        if storage.dtype() != DType::F32 {
            return Ok(false);
        }
        if !layout.is_contiguous() {
            return Ok(false);
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            return Ok(false);
        }
        let (n, k) = self_shape.dims2()?;
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        let device = storage.device();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_offset = (layout.start_offset() + batch_id * k) * DType::F32.size_in_bytes();
            let d_off = dst_offset_bytes + batch_id * n * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q8_0_bf16o(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                lhs_offset,
                &self.buffer,
                self.offset,
                d_off,
                dst,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    /// Q8_0 GEMV: BF16 input → F32 output.
    /// Eliminates the BF16→F32 pre-cast for E2B PLI gate and other F32-output GEMVs.
    pub fn fwd_mv_q8_0_bf16i_f32(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q8_0 {
            crate::bail!("fwd_mv_q8_0_bf16i_f32: Q8_0 required");
        }
        if storage.dtype() != DType::BF16 {
            crate::bail!("fwd_mv_q8_0_bf16i_f32: BF16 input required");
        }
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv_q8_0_bf16i_f32: not contiguous")
        }
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            r => crate::bail!("fwd_mv_q8_0_bf16i_f32: rank {r}"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv_q8_0_bf16i_f32: shape mismatch")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmv_q8_bf16i_f32")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_off = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let dst_off = batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q8_0_bf16i(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                lhs_off,
                &self.buffer,
                self.offset,
                dst_off,
                &dst,
            )
            .map_err(MetalError::from)?;
        }
        let dst_storage = crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    /// Q8_0 GEMV: BF16 input → F32 output, into a pre-allocated F32 buffer.
    pub fn fwd_mv_q8_0_bf16i_f32_prealloc(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        dst: &Buffer,
        dst_offset_bytes: usize,
    ) -> Result<bool> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q8_0 {
            return Ok(false);
        }
        if storage.dtype() != DType::BF16 {
            return Ok(false);
        }
        if !layout.is_contiguous() {
            return Ok(false);
        }
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        let device = storage.device();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_off = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let d_off = dst_offset_bytes + batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q8_0_bf16i(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                lhs_off,
                &self.buffer,
                self.offset,
                d_off,
                dst,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    /// Q8_0 GEMV fused with GELU × BF16 elementwise multiply: BF16 input → F32 output.
    ///
    /// Computes `gelu_tanh(GEMV(xs_bf16, self_q8)) * pli_embed[i]` for each output
    /// element in one kernel dispatch.  Saves 1 dispatch vs the two-step path
    /// (bf16i GEMV then gelu_mul_f32_bf16i).
    ///
    /// Returns `Ok(true)` when the kernel was dispatched; `Ok(false)` otherwise.
    pub fn fwd_mv_q8_0_bf16i_gelu_mul_bf16i_f32_prealloc(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        pli_embed_storage: &MetalStorage,
        pli_embed_layout: &crate::Layout,
        dst: &Buffer,
        dst_offset_bytes: usize,
    ) -> Result<bool> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q8_0 {
            return Ok(false);
        }
        if storage.dtype() != DType::BF16 || pli_embed_storage.dtype() != DType::BF16 {
            return Ok(false);
        }
        if !layout.is_contiguous() || !pli_embed_layout.is_contiguous() {
            return Ok(false);
        }
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        // pli_embed must have exactly n elements
        if pli_embed_layout.shape().elem_count() != n {
            return Ok(false);
        }
        let device = storage.device();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_off = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let pli_off = pli_embed_layout.start_offset() * DType::BF16.size_in_bytes();
            let d_off = dst_offset_bytes + batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q8_0_bf16i_gelu_mul_bf16i(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                lhs_off,
                pli_embed_storage.buffer(),
                pli_off,
                &self.buffer,
                self.offset,
                d_off,
                dst,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    /// Q4K BF16i GEMV fused with gelu_tanh × pli_embed (BF16) → F32 output.
    ///
    /// For the Gemma-4 E4B PLI gate: computes `gelu(GEMV(self, xs)) × pli_embed` in one dispatch.
    /// Saves 1 Metal dispatch vs separate Q4K BF16i GEMV + gelu_mul (42 dispatch savings for E4B).
    pub fn fwd_mv_q4k_bf16i_gelu_mul_bf16i_f32_prealloc(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        pli_embed_storage: &MetalStorage,
        pli_embed_layout: &crate::Layout,
        dst: &Buffer,
        dst_offset_bytes: usize,
    ) -> Result<bool> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q4K {
            return Ok(false);
        }
        if storage.dtype() != DType::BF16 || pli_embed_storage.dtype() != DType::BF16 {
            return Ok(false);
        }
        if !layout.is_contiguous() || !pli_embed_layout.is_contiguous() {
            return Ok(false);
        }
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        // pli_embed must have exactly n elements.
        if pli_embed_layout.shape().elem_count() != n {
            return Ok(false);
        }
        let device = storage.device();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_off = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let pli_off = pli_embed_layout.start_offset() * DType::BF16.size_in_bytes();
            let d_off = dst_offset_bytes + batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16i_gelu_mul_bf16i(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                lhs_off,
                pli_embed_storage.buffer(),
                pli_off,
                &self.buffer,
                self.offset,
                d_off,
                dst,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    /// Paired Q8_0 BF16-input GEMV with fused gelu_mul → F32 output.
    ///
    /// Computes `gelu_tanh(GEMV(self=gate_proj, xs)) * GEMV(up=up_proj, xs)` in one dispatch,
    /// writing the result into `dst` at `dst_offset_bytes`.
    ///
    /// Returns `Ok(true)` when the kernel was dispatched; `Ok(false)` when conditions
    /// are not met (wrong dtype, non-contiguous, or shape mismatch).
    pub fn fwd_mv2_q8_0_bf16i_gelu_mul_f32_prealloc(
        &self,
        self_shape: &Shape,      // gate_proj shape
        up_storage: &QMetalStorage, // up_proj QTensor
        up_shape: &Shape,
        storage: &MetalStorage,  // BF16 activation input
        layout: &crate::Layout,
        dst: &Buffer,
        dst_offset_bytes: usize,
    ) -> Result<bool> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q8_0 || up_storage.dtype != GgmlDType::Q8_0 {
            return Ok(false);
        }
        if storage.dtype() != DType::BF16 {
            return Ok(false);
        }
        if !layout.is_contiguous() {
            return Ok(false);
        }
        let (n, k) = self_shape.dims2()?;
        let (n_up, k_up) = up_shape.dims2()?;
        if n != n_up || k != k_up {
            return Ok(false);
        }
        let src_shape = layout.shape();
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        let device = storage.device();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_off = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let d_off = dst_offset_bytes + batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv2_q8_0_bf16i_gelu_mul_f32(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                &self.buffer,
                self.offset,
                &up_storage.buffer,
                up_storage.offset,
                storage.buffer(),
                lhs_off,
                d_off,
                dst,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    /// Q6K GEMV: BF16 input → F32 output.
    /// Saves the BF16→F32 pre-cast dispatch for Q6K layers (e.g. lm_head in E4B).
    pub fn fwd_mv_q6k_bf16i_f32(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q6K {
            crate::bail!("fwd_mv_q6k_bf16i_f32: Q6K required");
        }
        if storage.dtype() != DType::BF16 {
            crate::bail!("fwd_mv_q6k_bf16i_f32: BF16 input required");
        }
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv_q6k_bf16i_f32: not contiguous")
        }
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            r => crate::bail!("fwd_mv_q6k_bf16i_f32: rank {r}"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv_q6k_bf16i_f32: shape mismatch")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmv_q6k_bf16i_f32")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_off = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let dst_off = batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q6k_bf16i(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                lhs_off,
                &self.buffer,
                self.offset,
                dst_off,
                &dst,
            )
            .map_err(MetalError::from)?;
        }
        let dst_storage =
            crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    /// Q4K GEMV: BF16 input → BF16 output. For E4B decode.
    pub fn fwd_mv_q4k_bf16i_to_bf16(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q4K {
            crate::bail!("fwd_mv_q4k_bf16i_to_bf16: Q4K only");
        }
        if storage.dtype() != DType::BF16 {
            crate::bail!("fwd_mv_q4k_bf16i_to_bf16: BF16 input required");
        }
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv_q4k_bf16i_to_bf16: not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            r => crate::bail!("fwd_mv_q4k_bf16i_to_bf16: rank {r}"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv_q4k_bf16i_to_bf16: shape mismatch")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::BF16, "qmv_q4k_b2b")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_offset = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let dst_offset = batch_id * n * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16i_to_bf16(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                lhs_offset,
                &self.buffer,
                self.offset,
                dst_offset,
                &dst,
            )
            .map_err(MetalError::from)?;
        }
        let dst_storage =
            crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::BF16);
        Ok((dst_storage, dst_shape))
    }

    /// Q4K GEMV: BF16 input → BF16 output, writing into a pre-allocated destination buffer.
    /// Returns `true` if the dispatch succeeded; caller reuses `dst`.
    /// Saves 1 Metal buffer allocation per decode step vs `fwd_mv_q4k_bf16i_to_bf16`.
    pub fn fwd_mv_q4k_bf16i_to_bf16_prealloc(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        dst: &Buffer,
        dst_offset_bytes: usize,
    ) -> Result<bool> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q4K {
            return Ok(false);
        }
        if storage.dtype() != DType::BF16 {
            return Ok(false);
        }
        if !layout.is_contiguous() {
            return Ok(false);
        }
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        let device = storage.device();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_offset = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let d_off = dst_offset_bytes + batch_id * n * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16i_to_bf16(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                lhs_offset,
                &self.buffer,
                self.offset,
                d_off,
                dst,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    /// Q8_0 GEMV: BF16 input → BF16 output.
    /// Eliminates both the pre-cast (BF16→F32) and post-cast (F32→BF16).
    pub fn fwd_mv_q8_0_bf16i_to_bf16(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q8_0 {
            crate::bail!("fwd_mv_q8_0_bf16i_to_bf16: requires Q8_0 weights");
        }
        if storage.dtype() != DType::BF16 {
            crate::bail!("fwd_mv_q8_0_bf16i_to_bf16: requires BF16 input");
        }
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv_q8_0_bf16i_to_bf16: not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            crate::bail!("fwd_mv_q8_0_bf16i_to_bf16: rank < 2 {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            r => crate::bail!("fwd_mv_q8_0_bf16i_to_bf16: invalid rank {r}"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv_q8_0_bf16i_to_bf16: shape mismatch")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::BF16, "qmv_q8_b2b")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_offset = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let dst_offset = batch_id * n * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q8_0_bf16i_to_bf16(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                lhs_offset,
                &self.buffer,
                self.offset,
                dst_offset,
                &dst,
            )
            .map_err(MetalError::from)?;
        }
        let dst_storage =
            crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::BF16);
        Ok((dst_storage, dst_shape))
    }

    /// Q8_0 GEMV: BF16 input → BF16 output, writing into a caller-supplied buffer.
    ///
    /// Like `fwd_mv_q8_0_bf16i_to_bf16` but avoids allocating a new Metal buffer.
    /// `dst` must already be allocated with at least `n` BF16 elements.
    /// Returns `true` on success, `false` if preconditions are not met.
    pub fn fwd_mv_q8_0_bf16i_to_bf16_prealloc(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        dst: &Buffer,
        dst_offset_bytes: usize,
    ) -> Result<bool> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q8_0 {
            return Ok(false);
        }
        if storage.dtype() != DType::BF16 {
            return Ok(false);
        }
        if !layout.is_contiguous() {
            return Ok(false);
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            return Ok(false);
        }
        let (n, k) = self_shape.dims2()?;
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        let device = storage.device();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_offset = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let d_off = dst_offset_bytes + batch_id * n * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv_q8_0_bf16i_to_bf16(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                lhs_offset,
                &self.buffer,
                self.offset,
                d_off,
                dst,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    /// GEMV with BF16 input: calls kernel_mul_mv_q4_K_bf16i_f32, avoiding a separate
    /// BF16->F32 conversion dispatch.  Only valid for Q4K weights on Metal.
    pub fn fwd_mv_bf16i(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q4K && self.dtype != GgmlDType::Q8_0 {
            crate::bail!("fwd_mv_bf16i only supports Q4K and Q8_0 weights");
        }
        if storage.dtype() != DType::BF16 {
            crate::bail!("fwd_mv_bf16i requires BF16 input");
        }
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            r => crate::bail!("Invalid rank {r} for quantized matmul metal"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul_bf16i")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_offset = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let dst_offset = batch_id * n * DType::F32.size_in_bytes();
            match self.dtype {
                GgmlDType::Q8_0 => {
                    candle_metal_kernels::call_quantized_matmul_mv_q8_0_bf16i(
                        device.device(),
                        &encoder,
                        device.kernels(),
                        (1, 1, n, k),
                        storage.buffer(),
                        lhs_offset,
                        &self.buffer,
                        self.offset,
                        dst_offset,
                        &dst,
                    )
                    .map_err(MetalError::from)?;
                }
                _ => {
                    candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16i(
                        device.device(),
                        &encoder,
                        device.kernels(),
                        (1, 1, n, k),
                        storage.buffer(),
                        lhs_offset,
                        &self.buffer,
                        self.offset,
                        dst_offset,
                        &dst,
                    )
                    .map_err(MetalError::from)?;
                }
            }
        }
        let dst_storage = crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    /// BF16 input → F32 output GEMV into a pre-allocated F32 buffer.
    /// Supports both Q4K and Q8_0. Returns `true` on success; caller reuses `dst`.
    pub fn fwd_mv_bf16i_prealloc(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        dst: &Buffer,
        dst_offset_bytes: usize,
    ) -> Result<bool> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q4K && self.dtype != GgmlDType::Q8_0 {
            return Ok(false);
        }
        if storage.dtype() != DType::BF16 {
            return Ok(false);
        }
        if !layout.is_contiguous() {
            return Ok(false);
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            return Ok(false);
        }
        let (n, k) = self_shape.dims2()?;
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        let device = storage.device();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_offset = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let d_off = dst_offset_bytes + batch_id * n * DType::F32.size_in_bytes();
            match self.dtype {
                GgmlDType::Q8_0 => {
                    candle_metal_kernels::call_quantized_matmul_mv_q8_0_bf16i(
                        device.device(),
                        &encoder,
                        device.kernels(),
                        (1, 1, n, k),
                        storage.buffer(),
                        lhs_offset,
                        &self.buffer,
                        self.offset,
                        d_off,
                        dst,
                    )
                    .map_err(MetalError::from)?;
                }
                _ => {
                    candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16i(
                        device.device(),
                        &encoder,
                        device.kernels(),
                        (1, 1, n, k),
                        storage.buffer(),
                        lhs_offset,
                        &self.buffer,
                        self.offset,
                        d_off,
                        dst,
                    )
                    .map_err(MetalError::from)?;
                }
            }
        }
        Ok(true)
    }

    /// Fused double GEMV for Q4K: compute `out_a = self @ xs` and `out_b = other @ xs`
    /// in a single Metal dispatch, halving kernel-launch overhead and improving
    /// input-vector cache reuse.  Only valid when both tensors are Q4K with the same shape.
    pub fn fwd_mv2_q4k(
        &self,
        other: &QMetalStorage,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<((MetalStorage, Shape), (MetalStorage, Shape))> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        if self.dtype != GgmlDType::Q4K || other.dtype != GgmlDType::Q4K {
            crate::bail!("fwd_mv2_q4k requires both tensors to be Q4K")
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            rank => crate::bail!("Invalid rank {rank} for fwd_mv2_q4k"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);

        let device = storage.device().clone();
        let dst_a = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul_a")?;
        let dst_b = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul_b")?;
        let encoder = device.command_encoder()?;

        for batch_id in 0..m {
            let src1_offset =
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes();
            let dst_offset = batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv2_q4k(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                src1_offset,
                &self.buffer,
                self.offset,
                dst_offset,
                &dst_a,
                &other.buffer,
                other.offset,
                dst_offset,
                &dst_b,
            )
            .map_err(MetalError::from)?;
        }

        let da_storage =
            crate::MetalStorage::new(dst_a, device.clone(), dst_shape.elem_count(), DType::F32);
        let db_storage =
            crate::MetalStorage::new(dst_b, device, dst_shape.elem_count(), DType::F32);
        Ok(((da_storage, dst_shape.clone()), (db_storage, dst_shape)))
    }

    /// Paired Q4K GEMV into pre-allocated F32 buffers.
    /// Returns `true` if dispatch succeeded; caller reuses `dst_a` and `dst_b`.
    pub fn fwd_mv2_q4k_prealloc(
        &self,
        other: &QMetalStorage,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        dst_a: &Buffer,
        dst_b: &Buffer,
        dst_elem_count: usize,
    ) -> Result<bool> {
        use crate::MetalError;
        if !layout.is_contiguous() {
            return Ok(false);
        }
        if self.dtype != GgmlDType::Q4K || other.dtype != GgmlDType::Q4K {
            return Ok(false);
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            return Ok(false);
        }
        let (n, k) = self_shape.dims2()?;
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        if dst_elem_count != m * n {
            return Ok(false);
        }
        let device = storage.device();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let src1_offset =
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes();
            let dst_offset = batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv2_q4k(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                src1_offset,
                &self.buffer,
                self.offset,
                dst_offset,
                dst_a,
                &other.buffer,
                other.offset,
                dst_offset,
                dst_b,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    /// Fused paired Q4K GEMV + gelu_mul: F32 input → single F32 gelu_mul output.
    /// Saves 1 dispatch per MLP layer vs paired GEMV + separate gelu_mul (E4B decode hot path).
    pub fn fwd_mv2_q4k_gelu_mul_f32_prealloc(
        &self,
        other: &QMetalStorage,
        self_shape: &Shape,
        storage: &MetalStorage, // F32 activation input
        layout: &crate::Layout,
        dst: &Buffer,
        dst_offset_bytes: usize,
    ) -> Result<bool> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q4K || other.dtype != GgmlDType::Q4K {
            return Ok(false);
        }
        if storage.dtype() != DType::F32 {
            return Ok(false);
        }
        if !layout.is_contiguous() {
            return Ok(false);
        }
        let (n, k) = self_shape.dims2()?;
        let src_shape = layout.shape();
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        let device = storage.device();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let src1_off = (layout.start_offset() + batch_id * k) * DType::F32.size_in_bytes();
            let d_off = dst_offset_bytes + batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv2_q4k_gelu_mul_f32(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                src1_off,
                &self.buffer,
                self.offset,
                &other.buffer,
                other.offset,
                d_off,
                dst,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    /// Fused paired Q4K BF16i GEMV + gelu_mul: BF16 input → single F32 gelu_mul output.
    /// Saves 1 dispatch vs paired Q4K bf16i GEMV + separate gelu_mul.
    pub fn fwd_mv2_q4k_bf16i_gelu_mul_f32_prealloc(
        &self,
        self_shape: &Shape,         // gate_proj shape
        up_storage: &QMetalStorage, // up_proj QTensor
        up_shape: &Shape,
        storage: &MetalStorage,     // BF16 activation input
        layout: &crate::Layout,
        dst: &Buffer,
        dst_offset_bytes: usize,
    ) -> Result<bool> {
        use crate::MetalError;
        if self.dtype != GgmlDType::Q4K || up_storage.dtype != GgmlDType::Q4K {
            return Ok(false);
        }
        if storage.dtype() != DType::BF16 {
            return Ok(false);
        }
        if !layout.is_contiguous() {
            return Ok(false);
        }
        let (n, k) = self_shape.dims2()?;
        let (n_up, k_up) = up_shape.dims2()?;
        if n != n_up || k != k_up {
            return Ok(false);
        }
        let src_shape = layout.shape();
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        let device = storage.device();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let lhs_off = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let d_off = dst_offset_bytes + batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv2_q4k_bf16i_gelu_mul_f32(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                lhs_off,
                &self.buffer,
                self.offset,
                &up_storage.buffer,
                up_storage.offset,
                d_off,
                dst,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    /// Paired Q8_0 GEMV: computes (self @ xs, other @ xs) in one Metal dispatch.
    /// Used for fused gate+up MLP with Q8_0 weights (E2B Q8_0_full recipe).
    pub fn fwd_mv2_q8_0(
        &self,
        other: &QMetalStorage,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<((MetalStorage, Shape), (MetalStorage, Shape))> {
        use crate::MetalError;
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv2_q8_0: input not contiguous {layout:?}")
        }
        if self.dtype != GgmlDType::Q8_0 || other.dtype != GgmlDType::Q8_0 {
            crate::bail!("fwd_mv2_q8_0: both tensors must be Q8_0")
        }
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            rank => crate::bail!("fwd_mv2_q8_0: invalid rank {rank}"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv2_q8_0: input incompatible with weight shape")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst_a = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul_q8_a")?;
        let dst_b = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul_q8_b")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let src1_offset =
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes();
            let dst_offset = batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv2_q8_0(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                src1_offset,
                &self.buffer,
                self.offset,
                dst_offset,
                &dst_a,
                &other.buffer,
                other.offset,
                dst_offset,
                &dst_b,
            )
            .map_err(MetalError::from)?;
        }
        let da =
            crate::MetalStorage::new(dst_a, device.clone(), dst_shape.elem_count(), DType::F32);
        let db = crate::MetalStorage::new(dst_b, device, dst_shape.elem_count(), DType::F32);
        Ok(((da, dst_shape.clone()), (db, dst_shape)))
    }

    /// Paired Q8_0 GEMV with BF16 activation — avoids the BF16→F32 cast dispatch.
    pub fn fwd_mv2_q8_0_bf16i(
        &self,
        other: &QMetalStorage,
        self_shape: &Shape,
        storage: &MetalStorage, // BF16 activation
        layout: &crate::Layout,
    ) -> Result<((MetalStorage, Shape), (MetalStorage, Shape))> {
        use crate::MetalError;
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv2_q8_0_bf16i: not contiguous")
        }
        if self.dtype != GgmlDType::Q8_0 || other.dtype != GgmlDType::Q8_0 {
            crate::bail!("fwd_mv2_q8_0_bf16i: both must be Q8_0")
        }
        if storage.dtype() != DType::BF16 {
            crate::bail!("fwd_mv2_q8_0_bf16i: input must be BF16")
        }
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            r => crate::bail!("rank {r}"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv2_q8_0_bf16i: shape mismatch")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst_a = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmv2_q8_bf16i_a")?;
        let dst_b = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmv2_q8_bf16i_b")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let src1_offset = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let dst_offset = batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv2_q8_0_bf16i_paired(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                src1_offset,
                &self.buffer,
                self.offset,
                dst_offset,
                &dst_a,
                &other.buffer,
                other.offset,
                dst_offset,
                &dst_b,
            )
            .map_err(MetalError::from)?;
        }
        let da =
            crate::MetalStorage::new(dst_a, device.clone(), dst_shape.elem_count(), DType::F32);
        let db = crate::MetalStorage::new(dst_b, device, dst_shape.elem_count(), DType::F32);
        Ok(((da, dst_shape.clone()), (db, dst_shape)))
    }

    /// Paired Q8_0 bf16i GEMV into pre-allocated F32 buffers.
    /// Eliminates 2 Metal buffer allocations per decode step for E2B gate+up.
    pub fn fwd_mv2_q8_0_bf16i_prealloc(
        &self,
        other: &QMetalStorage,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        dst_a: &Buffer,
        dst_b: &Buffer,
    ) -> Result<bool> {
        use crate::MetalError;
        if !layout.is_contiguous() {
            return Ok(false);
        }
        if self.dtype != GgmlDType::Q8_0 || other.dtype != GgmlDType::Q8_0 {
            return Ok(false);
        }
        if storage.dtype() != DType::BF16 {
            return Ok(false);
        }
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        let device = storage.device();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let src1_offset = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let dst_offset = batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv2_q8_0_bf16i_paired(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                src1_offset,
                &self.buffer,
                self.offset,
                dst_offset,
                dst_a,
                &other.buffer,
                other.offset,
                dst_offset,
                dst_b,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    /// Paired Q3K GEMV: computes (self @ xs, other @ xs) in one Metal dispatch.
    /// Used for fused gate+up MLP with Q3K weights.
    pub fn fwd_mv2_q3k(
        &self,
        other: &QMetalStorage,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<((MetalStorage, Shape), (MetalStorage, Shape))> {
        use crate::MetalError;
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv2_q3k: input not contiguous {layout:?}")
        }
        if self.dtype != GgmlDType::Q3K || other.dtype != GgmlDType::Q3K {
            crate::bail!("fwd_mv2_q3k: both tensors must be Q3K")
        }
        let src_shape = layout.shape();
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            rank => crate::bail!("Invalid rank {rank}"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv2_q3k: size mismatch")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst_a = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul_q3k_a")?;
        let dst_b = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul_q3k_b")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let src1_offset =
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes();
            let dst_offset = batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv2_q3k(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                src1_offset,
                &self.buffer,
                self.offset,
                dst_offset,
                &dst_a,
                &other.buffer,
                other.offset,
                dst_offset,
                &dst_b,
            )
            .map_err(MetalError::from)?;
        }
        let da =
            crate::MetalStorage::new(dst_a, device.clone(), dst_shape.elem_count(), DType::F32);
        let db = crate::MetalStorage::new(dst_b, device, dst_shape.elem_count(), DType::F32);
        Ok(((da, dst_shape.clone()), (db, dst_shape)))
    }

    /// Paired BF16-input Q4K GEMV: like `fwd_mv2_q4k` but accepts BF16 input directly.
    /// Saves the BF16→F32 `to_dtype` dispatch.
    pub fn fwd_mv2_q4k_bf16i(
        &self,
        other: &QMetalStorage,
        self_shape: &Shape,
        storage: &MetalStorage, // BF16 input
        layout: &crate::Layout,
    ) -> Result<((MetalStorage, Shape), (MetalStorage, Shape))> {
        use crate::MetalError;
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv2_q4k_bf16i: input not contiguous {layout:?}")
        }
        if self.dtype != GgmlDType::Q4K || other.dtype != GgmlDType::Q4K {
            crate::bail!("fwd_mv2_q4k_bf16i: both tensors must be Q4K")
        }
        if storage.dtype() != DType::BF16 {
            crate::bail!(
                "fwd_mv2_q4k_bf16i: input must be BF16, got {:?}",
                storage.dtype()
            )
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            crate::bail!("fwd_mv2_q4k_bf16i: input has only one dimension")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            rank => crate::bail!("Invalid rank {rank} for fwd_mv2_q4k_bf16i"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv2_q4k_bf16i: input incompatible with weight shape")
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst_a = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul_bf16i_a")?;
        let dst_b = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul_bf16i_b")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let src1_offset =
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes();
            let dst_offset = batch_id * n * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv2_q4k_bf16i(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n, k),
                storage.buffer(),
                src1_offset,
                &self.buffer,
                self.offset,
                dst_offset,
                &dst_a,
                &other.buffer,
                other.offset,
                dst_offset,
                &dst_b,
            )
            .map_err(MetalError::from)?;
        }
        let da =
            crate::MetalStorage::new(dst_a, device.clone(), dst_shape.elem_count(), DType::F32);
        let db = crate::MetalStorage::new(dst_b, device, dst_shape.elem_count(), DType::F32);
        Ok(((da, dst_shape.clone()), (db, dst_shape)))
    }

    /// Fused QKV triple Q4K GEMV on Metal.
    /// `self`=q_weight, `kw`=k_weight, `vw`=v_weight; all must be Q4K.
    pub fn fwd_mv3_q4k(
        &self,
        kw: &QMetalStorage,
        vw: &QMetalStorage,
        self_shape: &Shape,
        kv_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(
        (MetalStorage, Shape),
        (MetalStorage, Shape),
        (MetalStorage, Shape),
    )> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        if self.dtype != GgmlDType::Q4K || kw.dtype != GgmlDType::Q4K || vw.dtype != GgmlDType::Q4K
        {
            crate::bail!("fwd_mv3_q4k requires all tensors to be Q4K")
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n_q, k) = self_shape.dims2()?;
        let (n_kv, _) = kv_shape.dims2()?;

        let mut dst_dims = src_shape.dims().to_vec();
        let m = match dst_dims.len() {
            3 => dst_dims[0] * dst_dims[1],
            2 => dst_dims[0],
            rank => crate::bail!("Invalid rank {rank} for fwd_mv3_q4k"),
        };
        let last_k = dst_dims.pop().unwrap();
        if last_k != k {
            crate::bail!(
                "input tensor {layout:?} incompatible with q_shape {:?}",
                self_shape
            )
        }
        let mut dst_dims_kv = dst_dims.clone();
        dst_dims.push(n_q);
        dst_dims_kv.push(n_kv);
        let dst_shape_q = Shape::from(dst_dims);
        let dst_shape_kv = Shape::from(dst_dims_kv);

        let device = storage.device().clone();
        let dst_q = device.new_buffer(dst_shape_q.elem_count(), DType::F32, "qmatmul_q")?;
        let dst_k = device.new_buffer(dst_shape_kv.elem_count(), DType::F32, "qmatmul_k")?;
        let dst_v = device.new_buffer(dst_shape_kv.elem_count(), DType::F32, "qmatmul_v")?;
        let encoder = device.command_encoder()?;

        for batch_id in 0..m {
            let src1_offset =
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes();
            let dst_q_offset = batch_id * n_q * DType::F32.size_in_bytes();
            let dst_kv_offset = batch_id * n_kv * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv3_q4k(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n_q, n_kv, k),
                storage.buffer(),
                src1_offset,
                &self.buffer,
                self.offset,
                dst_q_offset,
                &dst_q,
                &kw.buffer,
                kw.offset,
                dst_kv_offset,
                &dst_k,
                &vw.buffer,
                vw.offset,
                dst_kv_offset,
                &dst_v,
            )
            .map_err(MetalError::from)?;
        }

        let dq =
            crate::MetalStorage::new(dst_q, device.clone(), dst_shape_q.elem_count(), DType::F32);
        let dk =
            crate::MetalStorage::new(dst_k, device.clone(), dst_shape_kv.elem_count(), DType::F32);
        let dv = crate::MetalStorage::new(dst_v, device, dst_shape_kv.elem_count(), DType::F32);
        Ok((
            (dq, dst_shape_q),
            (dk, dst_shape_kv.clone()),
            (dv, dst_shape_kv),
        ))
    }

    /// Triple Q4K GEMV with inline BF16 output.
    /// Uses `kernel_mul_mv3_q4_K_f32_to_bf16` which converts F32→BF16 directly
    /// in the accumulator write — no intermediate F32 buffers, no extra cast dispatch.
    pub fn fwd_mv3_q4k_bf16o(
        &self,
        kw: &QMetalStorage,
        vw: &QMetalStorage,
        self_shape: &Shape,
        kv_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(
        (MetalStorage, Shape),
        (MetalStorage, Shape),
        (MetalStorage, Shape),
    )> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("fwd_mv3_q4k_bf16o: input not contiguous {layout:?}")
        }
        if self.dtype != GgmlDType::Q4K || kw.dtype != GgmlDType::Q4K || vw.dtype != GgmlDType::Q4K
        {
            crate::bail!("fwd_mv3_q4k_bf16o: all tensors must be Q4K")
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            crate::bail!("fwd_mv3_q4k_bf16o: input has only one dimension {layout:?}")
        }
        let (n_q, k) = self_shape.dims2()?;
        let (n_kv, _) = kv_shape.dims2()?;

        let mut dst_dims = src_shape.dims().to_vec();
        let m = match dst_dims.len() {
            3 => dst_dims[0] * dst_dims[1],
            2 => dst_dims[0],
            rank => crate::bail!("fwd_mv3_q4k_bf16o: invalid rank {rank}"),
        };
        let last_k = dst_dims.pop().unwrap();
        if last_k != k {
            crate::bail!(
                "fwd_mv3_q4k_bf16o: input {layout:?} incompatible with q_shape {:?}",
                self_shape
            )
        }
        let mut dst_dims_kv = dst_dims.clone();
        dst_dims.push(n_q);
        dst_dims_kv.push(n_kv);
        let dst_shape_q = Shape::from(dst_dims);
        let dst_shape_kv = Shape::from(dst_dims_kv);

        let device = storage.device().clone();
        // Allocate BF16 output buffers directly — no F32 intermediates.
        let dst_q = device.new_buffer(dst_shape_q.elem_count(), DType::BF16, "q4k_bf16o_q")?;
        let dst_k = device.new_buffer(dst_shape_kv.elem_count(), DType::BF16, "q4k_bf16o_k")?;
        let dst_v = device.new_buffer(dst_shape_kv.elem_count(), DType::BF16, "q4k_bf16o_v")?;
        let encoder = device.command_encoder()?;

        for batch_id in 0..m {
            let src1_offset =
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes();
            let dst_q_offset = batch_id * n_q * DType::BF16.size_in_bytes();
            let dst_kv_offset = batch_id * n_kv * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv3_q4k_bf16o(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n_q, n_kv, k),
                storage.buffer(),
                src1_offset,
                &self.buffer,
                self.offset,
                dst_q_offset,
                &dst_q,
                &kw.buffer,
                kw.offset,
                dst_kv_offset,
                &dst_k,
                &vw.buffer,
                vw.offset,
                dst_kv_offset,
                &dst_v,
            )
            .map_err(MetalError::from)?;
        }

        let rq =
            crate::MetalStorage::new(dst_q, device.clone(), dst_shape_q.elem_count(), DType::BF16);
        let rk = crate::MetalStorage::new(
            dst_k,
            device.clone(),
            dst_shape_kv.elem_count(),
            DType::BF16,
        );
        let rv = crate::MetalStorage::new(dst_v, device, dst_shape_kv.elem_count(), DType::BF16);
        Ok((
            (rq, dst_shape_q),
            (rk, dst_shape_kv.clone()),
            (rv, dst_shape_kv),
        ))
    }

    /// Triple Q4K GEMV: BF16 input → BF16 output.
    pub fn fwd_mv3_q4k_bf16i_to_bf16(
        &self,
        kw: &QMetalStorage,
        vw: &QMetalStorage,
        self_shape: &Shape,
        kv_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(
        (MetalStorage, Shape),
        (MetalStorage, Shape),
        (MetalStorage, Shape),
    )> {
        use crate::MetalError;
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv3_q4k_bf16i_to_bf16: not contiguous")
        }
        if self.dtype != GgmlDType::Q4K || kw.dtype != GgmlDType::Q4K || vw.dtype != GgmlDType::Q4K
        {
            crate::bail!("fwd_mv3_q4k_bf16i_to_bf16: all Q4K required")
        }
        if storage.dtype() != DType::BF16 {
            crate::bail!("fwd_mv3_q4k_bf16i_to_bf16: BF16 input required")
        }
        let src_shape = layout.shape();
        let (n_q, k) = self_shape.dims2()?;
        let (n_kv, _) = kv_shape.dims2()?;
        let mut dst_dims = src_shape.dims().to_vec();
        let m = match dst_dims.len() {
            3 => dst_dims[0] * dst_dims[1],
            2 => dst_dims[0],
            r => crate::bail!("rank {r}"),
        };
        let last_k = dst_dims.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv3_q4k_bf16i_to_bf16: shape mismatch")
        }
        let mut dst_dims_kv = dst_dims.clone();
        dst_dims.push(n_q);
        dst_dims_kv.push(n_kv);
        let sq = Shape::from(dst_dims);
        let skv = Shape::from(dst_dims_kv);
        let device = storage.device().clone();
        let dst_q = device.new_buffer(sq.elem_count(), DType::BF16, "q4k_b2b_q")?;
        let dst_k = device.new_buffer(skv.elem_count(), DType::BF16, "q4k_b2b_k")?;
        let dst_v = device.new_buffer(skv.elem_count(), DType::BF16, "q4k_b2b_v")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let src1_off = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let dq_off = batch_id * n_q * DType::BF16.size_in_bytes();
            let dkv_off = batch_id * n_kv * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv3_q4k_bf16i_to_bf16(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n_q, n_kv, k),
                storage.buffer(),
                src1_off,
                &self.buffer,
                self.offset,
                dq_off,
                &dst_q,
                &kw.buffer,
                kw.offset,
                dkv_off,
                &dst_k,
                &vw.buffer,
                vw.offset,
                dkv_off,
                &dst_v,
            )
            .map_err(MetalError::from)?;
        }
        let rq = crate::MetalStorage::new(dst_q, device.clone(), sq.elem_count(), DType::BF16);
        let rk = crate::MetalStorage::new(dst_k, device.clone(), skv.elem_count(), DType::BF16);
        let rv = crate::MetalStorage::new(dst_v, device, skv.elem_count(), DType::BF16);
        Ok(((rq, sq), (rk, skv.clone()), (rv, skv)))
    }

    /// Triple Q4K GEMV: BF16 → BF16, writing into caller-supplied buffers.
    /// Eliminates 3 Metal buffer allocations per decode step for Q4K (E4B) models.
    #[allow(clippy::too_many_arguments)]
    pub fn fwd_mv3_q4k_bf16i_to_bf16_prealloc(
        &self,
        kw: &QMetalStorage,
        vw: &QMetalStorage,
        self_shape: &Shape,
        kv_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        dst_q: &Buffer,
        dst_q_offset: usize,
        dst_k: &Buffer,
        dst_k_offset: usize,
        dst_v: &Buffer,
        dst_v_offset: usize,
    ) -> Result<bool> {
        use crate::MetalError;
        if !layout.is_contiguous() {
            return Ok(false);
        }
        if self.dtype != GgmlDType::Q4K || kw.dtype != GgmlDType::Q4K || vw.dtype != GgmlDType::Q4K
        {
            return Ok(false);
        }
        if storage.dtype() != DType::BF16 {
            return Ok(false);
        }
        let src_shape = layout.shape();
        let (n_q, k) = self_shape.dims2()?;
        let (n_kv, _) = kv_shape.dims2()?;
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        let device = storage.device();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let src1_off = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let dq_off = dst_q_offset + batch_id * n_q * DType::BF16.size_in_bytes();
            let dkv_off_k = dst_k_offset + batch_id * n_kv * DType::BF16.size_in_bytes();
            let dkv_off_v = dst_v_offset + batch_id * n_kv * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv3_q4k_bf16i_to_bf16(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n_q, n_kv, k),
                storage.buffer(),
                src1_off,
                &self.buffer,
                self.offset,
                dq_off,
                dst_q,
                &kw.buffer,
                kw.offset,
                dkv_off_k,
                dst_k,
                &vw.buffer,
                vw.offset,
                dkv_off_v,
                dst_v,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    /// Triple Q8_0 GEMV: computes Q, K, V projections in one Metal dispatch.
    pub fn fwd_mv3_q8_0(
        &self,
        kw: &QMetalStorage,
        vw: &QMetalStorage,
        self_shape: &Shape,
        kv_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(
        (MetalStorage, Shape),
        (MetalStorage, Shape),
        (MetalStorage, Shape),
    )> {
        use crate::MetalError;
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv3_q8_0: input not contiguous")
        }
        if self.dtype != GgmlDType::Q8_0
            || kw.dtype != GgmlDType::Q8_0
            || vw.dtype != GgmlDType::Q8_0
        {
            crate::bail!("fwd_mv3_q8_0: all tensors must be Q8_0")
        }
        let src_shape = layout.shape();
        let (n_q, k) = self_shape.dims2()?;
        let (n_kv, _) = kv_shape.dims2()?;
        let mut dst_dims = src_shape.dims().to_vec();
        let m = match dst_dims.len() {
            3 => dst_dims[0] * dst_dims[1],
            2 => dst_dims[0],
            rank => crate::bail!("fwd_mv3_q8_0: invalid rank {rank}"),
        };
        let last_k = dst_dims.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv3_q8_0: input shape mismatch")
        }
        let mut dst_dims_kv = dst_dims.clone();
        dst_dims.push(n_q);
        dst_dims_kv.push(n_kv);
        let dst_shape_q = Shape::from(dst_dims);
        let dst_shape_kv = Shape::from(dst_dims_kv);
        let device = storage.device().clone();
        let dst_q = device.new_buffer(dst_shape_q.elem_count(), DType::F32, "qmv3_q8_q")?;
        let dst_k = device.new_buffer(dst_shape_kv.elem_count(), DType::F32, "qmv3_q8_k")?;
        let dst_v = device.new_buffer(dst_shape_kv.elem_count(), DType::F32, "qmv3_q8_v")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let src1_offset =
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes();
            let dst_q_offset = batch_id * n_q * DType::F32.size_in_bytes();
            let dst_kv_offset = batch_id * n_kv * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv3_q8_0(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n_q, n_kv, k),
                storage.buffer(),
                src1_offset,
                &self.buffer,
                self.offset,
                dst_q_offset,
                &dst_q,
                &kw.buffer,
                kw.offset,
                dst_kv_offset,
                &dst_k,
                &vw.buffer,
                vw.offset,
                dst_kv_offset,
                &dst_v,
            )
            .map_err(MetalError::from)?;
        }
        let dq =
            crate::MetalStorage::new(dst_q, device.clone(), dst_shape_q.elem_count(), DType::F32);
        let dk =
            crate::MetalStorage::new(dst_k, device.clone(), dst_shape_kv.elem_count(), DType::F32);
        let dv = crate::MetalStorage::new(dst_v, device, dst_shape_kv.elem_count(), DType::F32);
        Ok((
            (dq, dst_shape_q),
            (dk, dst_shape_kv.clone()),
            (dv, dst_shape_kv),
        ))
    }

    /// Triple Q8_0 GEMV with BF16 OUTPUT — F32 in, BF16 Q/K/V out (saves 3 back-casts).
    pub fn fwd_mv3_q8_0_bf16o(
        &self,
        kw: &QMetalStorage,
        vw: &QMetalStorage,
        self_shape: &Shape,
        kv_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(
        (MetalStorage, Shape),
        (MetalStorage, Shape),
        (MetalStorage, Shape),
    )> {
        use crate::MetalError;
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv3_q8_0_bf16o: not contiguous")
        }
        if self.dtype != GgmlDType::Q8_0
            || kw.dtype != GgmlDType::Q8_0
            || vw.dtype != GgmlDType::Q8_0
        {
            crate::bail!("fwd_mv3_q8_0_bf16o: all must be Q8_0")
        }
        if storage.dtype() != DType::F32 {
            crate::bail!("fwd_mv3_q8_0_bf16o: input must be F32")
        }
        let src_shape = layout.shape();
        let (n_q, k) = self_shape.dims2()?;
        let (n_kv, _) = kv_shape.dims2()?;
        let mut dd = src_shape.dims().to_vec();
        let m = match dd.len() {
            3 => dd[0] * dd[1],
            2 => dd[0],
            r => crate::bail!("rank {r}"),
        };
        let last_k = dd.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv3_q8_0_bf16o: shape mismatch")
        }
        let mut dd_kv = dd.clone();
        dd.push(n_q);
        dd_kv.push(n_kv);
        let sq = Shape::from(dd);
        let skv = Shape::from(dd_kv);
        let device = storage.device().clone();
        let dst_q = device.new_buffer(sq.elem_count(), DType::BF16, "qmv3_q8_bf16o_q")?;
        let dst_k = device.new_buffer(skv.elem_count(), DType::BF16, "qmv3_q8_bf16o_k")?;
        let dst_v = device.new_buffer(skv.elem_count(), DType::BF16, "qmv3_q8_bf16o_v")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let src1_off = (layout.start_offset() + batch_id * k) * DType::F32.size_in_bytes();
            let dq_off = batch_id * n_q * DType::BF16.size_in_bytes();
            let dkv_off = batch_id * n_kv * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv3_q8_0_f32_to_bf16(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n_q, n_kv, k),
                storage.buffer(),
                src1_off,
                &self.buffer,
                self.offset,
                dq_off,
                &dst_q,
                &kw.buffer,
                kw.offset,
                dkv_off,
                &dst_k,
                &vw.buffer,
                vw.offset,
                dkv_off,
                &dst_v,
            )
            .map_err(MetalError::from)?;
        }
        let dq = crate::MetalStorage::new(dst_q, device.clone(), sq.elem_count(), DType::BF16);
        let dk = crate::MetalStorage::new(dst_k, device.clone(), skv.elem_count(), DType::BF16);
        let dv = crate::MetalStorage::new(dst_v, device, skv.elem_count(), DType::BF16);
        Ok(((dq, sq), (dk, skv.clone()), (dv, skv)))
    }

    /// Triple Q8_0 GEMV with BF16 activation — avoids BF16→F32 cast dispatch.
    pub fn fwd_mv3_q8_0_bf16i(
        &self,
        kw: &QMetalStorage,
        vw: &QMetalStorage,
        self_shape: &Shape,
        kv_shape: &Shape,
        storage: &MetalStorage, // BF16 activation
        layout: &crate::Layout,
    ) -> Result<(
        (MetalStorage, Shape),
        (MetalStorage, Shape),
        (MetalStorage, Shape),
    )> {
        use crate::MetalError;
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv3_q8_0_bf16i: not contiguous")
        }
        if self.dtype != GgmlDType::Q8_0
            || kw.dtype != GgmlDType::Q8_0
            || vw.dtype != GgmlDType::Q8_0
        {
            crate::bail!("fwd_mv3_q8_0_bf16i: all must be Q8_0")
        }
        if storage.dtype() != DType::BF16 {
            crate::bail!("fwd_mv3_q8_0_bf16i: input must be BF16")
        }
        let src_shape = layout.shape();
        let (n_q, k) = self_shape.dims2()?;
        let (n_kv, _) = kv_shape.dims2()?;
        let mut dd = src_shape.dims().to_vec();
        let m = match dd.len() {
            3 => dd[0] * dd[1],
            2 => dd[0],
            r => crate::bail!("rank {r}"),
        };
        let last_k = dd.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv3_q8_0_bf16i: shape mismatch")
        }
        let mut dd_kv = dd.clone();
        dd.push(n_q);
        dd_kv.push(n_kv);
        let sq = Shape::from(dd);
        let skv = Shape::from(dd_kv);
        let device = storage.device().clone();
        let dst_q = device.new_buffer(sq.elem_count(), DType::F32, "qmv3_q8_bf16i_q")?;
        let dst_k = device.new_buffer(skv.elem_count(), DType::F32, "qmv3_q8_bf16i_k")?;
        let dst_v = device.new_buffer(skv.elem_count(), DType::F32, "qmv3_q8_bf16i_v")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let src1_off = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let dq_off = batch_id * n_q * DType::F32.size_in_bytes();
            let dkv_off = batch_id * n_kv * DType::F32.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv3_q8_0_bf16i(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n_q, n_kv, k),
                storage.buffer(),
                src1_off,
                &self.buffer,
                self.offset,
                dq_off,
                &dst_q,
                &kw.buffer,
                kw.offset,
                dkv_off,
                &dst_k,
                &vw.buffer,
                vw.offset,
                dkv_off,
                &dst_v,
            )
            .map_err(MetalError::from)?;
        }
        let dq = crate::MetalStorage::new(dst_q, device.clone(), sq.elem_count(), DType::F32);
        let dk = crate::MetalStorage::new(dst_k, device.clone(), skv.elem_count(), DType::F32);
        let dv = crate::MetalStorage::new(dst_v, device, skv.elem_count(), DType::F32);
        Ok(((dq, sq), (dk, skv.clone()), (dv, skv)))
    }

    /// Triple Q8_0 GEMV: BF16 input → BF16 output (Q, K, V).
    /// Eliminates both the pre-cast (BF16→F32) and post-casts (3×F32→BF16),
    /// saving 1 dispatch vs fwd_mv3_q8_0_bf16o and 4 vs the F32 path.
    pub fn fwd_mv3_q8_0_bf16i_to_bf16(
        &self,
        kw: &QMetalStorage,
        vw: &QMetalStorage,
        self_shape: &Shape,
        kv_shape: &Shape,
        storage: &MetalStorage, // BF16 activation
        layout: &crate::Layout,
    ) -> Result<(
        (MetalStorage, Shape),
        (MetalStorage, Shape),
        (MetalStorage, Shape),
    )> {
        use crate::MetalError;
        if !layout.is_contiguous() {
            crate::bail!("fwd_mv3_q8_0_bf16i_to_bf16: not contiguous")
        }
        if self.dtype != GgmlDType::Q8_0
            || kw.dtype != GgmlDType::Q8_0
            || vw.dtype != GgmlDType::Q8_0
        {
            crate::bail!("fwd_mv3_q8_0_bf16i_to_bf16: all must be Q8_0")
        }
        if storage.dtype() != DType::BF16 {
            crate::bail!("fwd_mv3_q8_0_bf16i_to_bf16: input must be BF16")
        }
        let src_shape = layout.shape();
        let (n_q, k) = self_shape.dims2()?;
        let (n_kv, _) = kv_shape.dims2()?;
        let mut dd = src_shape.dims().to_vec();
        let m = match dd.len() {
            3 => dd[0] * dd[1],
            2 => dd[0],
            r => crate::bail!("rank {r}"),
        };
        let last_k = dd.pop().unwrap();
        if last_k != k {
            crate::bail!("fwd_mv3_q8_0_bf16i_to_bf16: shape mismatch")
        }
        let mut dd_kv = dd.clone();
        dd.push(n_q);
        dd_kv.push(n_kv);
        let sq = Shape::from(dd);
        let skv = Shape::from(dd_kv);
        let device = storage.device().clone();
        let dst_q = device.new_buffer(sq.elem_count(), DType::BF16, "qmv3_q8_b2b_q")?;
        let dst_k = device.new_buffer(skv.elem_count(), DType::BF16, "qmv3_q8_b2b_k")?;
        let dst_v = device.new_buffer(skv.elem_count(), DType::BF16, "qmv3_q8_b2b_v")?;
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let src1_off = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let dq_off = batch_id * n_q * DType::BF16.size_in_bytes();
            let dkv_off = batch_id * n_kv * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv3_q8_0_bf16i_to_bf16(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n_q, n_kv, k),
                storage.buffer(),
                src1_off,
                &self.buffer,
                self.offset,
                dq_off,
                &dst_q,
                &kw.buffer,
                kw.offset,
                dkv_off,
                &dst_k,
                &vw.buffer,
                vw.offset,
                dkv_off,
                &dst_v,
            )
            .map_err(MetalError::from)?;
        }
        let dq = crate::MetalStorage::new(dst_q, device.clone(), sq.elem_count(), DType::BF16);
        let dk = crate::MetalStorage::new(dst_k, device.clone(), skv.elem_count(), DType::BF16);
        let dv = crate::MetalStorage::new(dst_v, device, skv.elem_count(), DType::BF16);
        Ok(((dq, sq), (dk, skv.clone()), (dv, skv)))
    }

    /// Triple Q8_0 GEMV: BF16 input → BF16 outputs, writing into caller-supplied buffers.
    ///
    /// Eliminates 3 Metal buffer allocations per decode step (one per Q/K/V output).
    /// `dst_q`, `dst_k`, `dst_v` must be pre-allocated BF16 buffers with correct sizes.
    /// Returns `true` on success, `false` if preconditions not met.
    #[allow(clippy::too_many_arguments)]
    pub fn fwd_mv3_q8_0_bf16i_to_bf16_prealloc(
        &self,
        kw: &QMetalStorage,
        vw: &QMetalStorage,
        self_shape: &Shape,
        kv_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        dst_q: &Buffer,
        dst_q_offset: usize,
        dst_k: &Buffer,
        dst_k_offset: usize,
        dst_v: &Buffer,
        dst_v_offset: usize,
    ) -> Result<bool> {
        use crate::MetalError;
        if !layout.is_contiguous() {
            return Ok(false);
        }
        if self.dtype != GgmlDType::Q8_0
            || kw.dtype != GgmlDType::Q8_0
            || vw.dtype != GgmlDType::Q8_0
        {
            return Ok(false);
        }
        if storage.dtype() != DType::BF16 {
            return Ok(false);
        }
        let src_shape = layout.shape();
        let (n_q, k) = self_shape.dims2()?;
        let (n_kv, _) = kv_shape.dims2()?;
        let m = match src_shape.rank() {
            3 => src_shape.dims()[0] * src_shape.dims()[1],
            2 => src_shape.dims()[0],
            _ => return Ok(false),
        };
        let last_k = src_shape.dims()[src_shape.rank() - 1];
        if last_k != k {
            return Ok(false);
        }
        let device = storage.device();
        let encoder = device.command_encoder()?;
        for batch_id in 0..m {
            let src1_off = (layout.start_offset() + batch_id * k) * DType::BF16.size_in_bytes();
            let dq_off = dst_q_offset + batch_id * n_q * DType::BF16.size_in_bytes();
            let dkv_off_k = dst_k_offset + batch_id * n_kv * DType::BF16.size_in_bytes();
            let dkv_off_v = dst_v_offset + batch_id * n_kv * DType::BF16.size_in_bytes();
            candle_metal_kernels::call_quantized_matmul_mv3_q8_0_bf16i_to_bf16(
                device.device(),
                &encoder,
                device.kernels(),
                (1, 1, n_q, n_kv, k),
                storage.buffer(),
                src1_off,
                &self.buffer,
                self.offset,
                dq_off,
                dst_q,
                &kw.buffer,
                kw.offset,
                dkv_off_k,
                dst_k,
                &vw.buffer,
                vw.offset,
                dkv_off_v,
                dst_v,
            )
            .map_err(MetalError::from)?;
        }
        Ok(true)
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let n = self_shape.dim(D::Minus2)?;
        let k = self_shape.dim(D::Minus1)?;
        let mut dst_shape = src_shape.dims().to_vec();

        if src_shape.rank() < self_shape.rank() {
            crate::bail!(
                "input rank ({}) must be >= weight rank ({})",
                src_shape.rank(),
                self_shape.rank()
            )
        }

        if src_shape.dim(D::Minus2)? == 1 {
            return self.fwd_mv(self_shape, storage, layout);
        }

        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        // TQ2_0 has no matrix-matrix (prefill) Metal kernel.  Dequantize the
        // weight to F32 on CPU and re-upload so that the F32 GEMM path handles
        // it.  This only affects per_layer_model_proj (one small tensor) so the
        // performance impact is negligible.
        if self.dtype == GgmlDType::TQ2_0 {
            use crate::quantized::k_quants::GgmlType;
            // Read raw blocks from GPU → CPU.
            let buffer = device.allocate_buffer(self.buffer.length())?;
            {
                let blit = device.blit_command_encoder()?;
                blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
                blit.end_encoding();
                device.wait_until_completed()?;
            }
            let block_len = (n * k) / GgmlDType::TQ2_0.block_size();
            let blocks: Vec<crate::quantized::BlockTq2_0> = read_to_vec(&buffer, block_len);
            let mut f32_data = vec![0.0f32; n * k];
            crate::quantized::BlockTq2_0::to_float(&blocks, &mut f32_data);
            // Upload F32 data back to GPU as a new buffer.
            let (f32_buf, _offset) = device.new_buffer_with_data_untracked_offset(&f32_data)?;
            let f32_q = QMetalStorage {
                dtype: GgmlDType::F32,
                device: device.clone(),
                buffer: f32_buf,
                offset: 0,
            };
            return f32_q.fwd(self_shape, storage, layout);
        }

        let dst = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul")?;
        let encoder = device.command_encoder()?;

        assert_eq!(storage.dtype(), DType::F32);

        if self_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", self_shape.rank())
        }
        let src0_l = crate::Layout::contiguous(
            [vec![1; 4 - self_shape.rank()], self_shape.dims().to_vec()].concat(),
        );
        let src0_stride = src0_l
            .stride()
            .iter()
            .map(|x| {
                (*x as f32 * (self.dtype.type_size() as f32 / self.dtype.block_size() as f32))
                    as usize
            })
            .collect::<Vec<_>>();

        if src_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", src_shape.rank())
        }
        let src1_l = crate::Layout::contiguous(
            [vec![1; 4 - src_shape.rank()], src_shape.dims().to_vec()].concat(),
        );

        candle_metal_kernels::call_quantized_matmul_mm_t(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            src0_l.dims(),
            &src0_stride,
            &self.buffer,
            src1_l.dims(),
            &src1_l
                .stride()
                .iter()
                .map(|x| x * DType::F32.size_in_bytes())
                .collect::<Vec<_>>(),
            storage.buffer(),
            src1_l.start_offset() * storage.dtype().size_in_bytes(),
            dst_shape.dims(),
            0,
            &dst,
        )
        .map_err(MetalError::from)?;

        let dst_storage = crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let buffer = self.device.allocate_buffer(self.buffer.length())?;
        {
            let blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
            blit.end_encoding();
        }
        self.device.wait_until_completed()?;
        Ok(read_to_vec::<u8>(&buffer, self.storage_size_in_bytes()))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &MetalDevice,
    data: &[T],
) -> Result<QStorage> {
    // Use arena allocation when available (batch mode): single Metal buffer for all weights.
    // Returns the shared buffer + byte offset where this tensor's data starts.
    let (buffer, offset) = device.new_buffer_with_data_untracked_offset(data)?;
    let device = device.clone();
    Ok(QStorage::Metal(QMetalStorage {
        dtype: T::DTYPE,
        device,
        buffer,
        offset,
    }))
}

/// Load quantized tensor using zero-copy from mmap'd data.
///
/// # Safety
///
/// `data` must remain valid for the entire lifetime of the returned `QStorage`
/// and any tensors derived from it.  Metal will **not** copy the memory — it
/// creates a buffer that points directly into the caller's slice.  Using the
/// tensor after the backing slice is freed is undefined behaviour (use-after-free).
///
/// In practice this is only safe when `data` comes from a memory-mapped file
/// that lives for the duration of the process, which is how it is used in
/// `candle_core::quantized::gguf_file`.
pub unsafe fn load_quantized_no_copy<T: super::GgmlType + Send + Sync + 'static>(
    device: &MetalDevice,
    data: &[T],
) -> Result<QStorage> {
    let buffer = device.new_buffer_no_copy(data)?;
    let device = device.clone();
    Ok(QStorage::Metal(QMetalStorage {
        dtype: T::DTYPE,
        device,
        buffer,
        offset: 0,
    }))
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

impl From<GgmlDType> for candle_metal_kernels::GgmlDType {
    fn from(value: GgmlDType) -> Self {
        match value {
            GgmlDType::Q4_0 => candle_metal_kernels::GgmlDType::Q4_0,
            GgmlDType::Q4_1 => candle_metal_kernels::GgmlDType::Q4_1,
            GgmlDType::Q5_0 => candle_metal_kernels::GgmlDType::Q5_0,
            GgmlDType::Q5_1 => candle_metal_kernels::GgmlDType::Q5_1,
            GgmlDType::Q8_0 => candle_metal_kernels::GgmlDType::Q8_0,
            GgmlDType::Q8_1 => candle_metal_kernels::GgmlDType::Q8_1,
            GgmlDType::Q2K => candle_metal_kernels::GgmlDType::Q2K,
            GgmlDType::Q3K => candle_metal_kernels::GgmlDType::Q3K,
            GgmlDType::Q4K => candle_metal_kernels::GgmlDType::Q4K,
            GgmlDType::Q5K => candle_metal_kernels::GgmlDType::Q5K,
            GgmlDType::Q6K => candle_metal_kernels::GgmlDType::Q6K,
            GgmlDType::Q8K => candle_metal_kernels::GgmlDType::Q8K,
            GgmlDType::F16 => candle_metal_kernels::GgmlDType::F16,
            GgmlDType::F32 => candle_metal_kernels::GgmlDType::F32,
            GgmlDType::BF16 => candle_metal_kernels::GgmlDType::F16,
            GgmlDType::TQ2_0 => candle_metal_kernels::GgmlDType::TQ2_0,
        }
    }
}

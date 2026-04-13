//! Shared quantized linear layer, GGUF VarBuilder, and construction helpers.
//!
//! Extracted from gemma4.rs so that multiple model architectures (Gemma4, Qwen3.5,
//! etc.) can reuse `QLinear`, `QGgufVarBuilder`, and `qlinear_b` without code
//! duplication.

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// QLinear: a Linear layer backed by either a standard Tensor or a QMatMul.
//
// When weights are loaded from a GGUF file with --quantize, using QMatMul keeps
// the weights in their compressed quantized form (e.g. Q4K) and dispatches to
// Metal's optimised quantized GEMV kernel (call_quantized_matmul_mv_t) during
// the decode step.  This is the same kernel that llama.cpp/ggml uses and gives
// ~3-4× higher decode throughput compared to dequantizing to bf16 first.
//
// For safetensors (bf16) models, QLinear falls back to a standard Linear
// (identical to the previous behaviour).
// ---------------------------------------------------------------------------

/// A linear projection layer backed by `QMatMul`.
///
/// ## Memory-efficient quantized linear (GGUF path)
///
/// Stores only the `QMatMul::QTensor` — the weight stays compressed in Metal
/// memory (Q4K ≈ 4.5 bits/param).  No second bf16 copy is kept.
///
/// * **Decode** (seq_len = 1): Metal's `kernel_mul_mv_q4_K_f32` GEMV.
///   Input is cast bf16→f32 (the kernel requires f32), output cast back.
///   Q4K is 4× smaller than bf16 → ~3-4× faster GEMV.
///
/// * **Prefill** (seq_len > 1): `forward_via_f16` dequantizes the QTensor to
///   f16 on-the-fly, runs the standard f16 GEMM, then converts back to the
///   original dtype.  The dequantization is a single fast Metal kernel; its
///   cost is negligible compared with the GEMM for any realistic sequence length.
///   Memory stays at QTensor-only — no permanent second copy.
///
/// ## Safetensors path
///
/// `inner` is `QMatMul::Tensor` (plain bf16).  Both paths use the standard
/// matmul, identical to `candle_nn::Linear`.
#[derive(Debug, Clone)]
pub struct QLinear {
    inner: QMatMul,
    pub(crate) bias: Option<Tensor>,
}

impl QLinear {
    /// Build from a quantized tensor (GGUF path).
    pub fn from_qtensor(
        qtensor: Arc<candle_core::quantized::QTensor>,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        let inner = QMatMul::from_arc(qtensor)?;
        Ok(Self { inner, bias })
    }

    /// Build from a regular tensor (safetensors path).
    pub fn from_tensor(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            inner: QMatMul::Tensor(weight),
            bias,
        }
    }

    /// Returns true when the underlying weight is a quantized QTensor (GGUF path).
    /// Returns false for the dense BF16 safetensors path (`QMatMul::Tensor`).
    pub fn is_quantized(&self) -> bool {
        matches!(self.inner, QMatMul::QTensor(_))
    }
}

impl Module for QLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.inner {
            QMatMul::QTensor(_) => {
                let orig_dtype = xs.dtype();
                // On CUDA with BF16 activations, the patched `dequantize_matmul_vec`
                // has a BF16 fast path that fuses BF16→Q8_1 in one kernel dispatch
                // (vs the old two-dispatch BF16→F32 + F32→Q8_1 path).
                // Pass BF16 input directly and save one kernel launch per GEMV.
                // Non-CUDA or non-BF16: keep the standard F32 conversion path.
                let r = if matches!(xs.device(), candle_core::Device::Cuda(_))
                    && orig_dtype == DType::BF16
                {
                    // Direct BF16 path: skip the BF16→F32 conversion kernel.
                    self.inner.forward(xs)?
                } else {
                    let xs_f32 = if orig_dtype == DType::F32 {
                        xs.clone()
                    } else {
                        xs.to_dtype(DType::F32)?
                    };
                    self.inner.forward(&xs_f32)?
                };
                // GEMV output is always F32; convert back to orig_dtype if needed.
                let result = if orig_dtype == DType::F32 || r.dtype() == orig_dtype {
                    r
                } else {
                    r.to_dtype(orig_dtype)?
                };
                match &self.bias {
                    None => Ok(result),
                    Some(b) => result.broadcast_add(b),
                }
            }
            QMatMul::Tensor(w) | QMatMul::TensorF16(w) => {
                // Dense path: standard matmul.
                // Some external GGUFs (e.g. bartowski Gemma4) store certain
                // projections (per_layer_model_projection) as unquantized F32.
                // QMatMul::from_arc then creates QMatMul::Tensor(F32) rather than
                // a QTensor, and Tensor::matmul requires both operands to share the
                // same dtype.  Cast xs to the weight dtype before the multiply, then
                // cast the result back so the caller sees the expected activation dtype.
                let orig_dtype = xs.dtype();
                let w_dtype = w.dtype();
                let (xs_cast, need_cast_back) = if orig_dtype != w_dtype {
                    (xs.to_dtype(w_dtype)?, true)
                } else {
                    (xs.clone(), false)
                };
                let result = self.inner.forward(&xs_cast)?;
                let result = if need_cast_back && result.dtype() != orig_dtype {
                    result.to_dtype(orig_dtype)?
                } else {
                    result
                };
                match &self.bias {
                    None => Ok(result),
                    Some(b) => result.broadcast_add(b),
                }
            }
        }
    }
}

impl QLinear {
    /// Forward pass that takes an already-F32 input and returns F32 output.
    ///
    /// When the underlying QMatMul is a QTensor (quantized GGUF path), the
    /// CUDA/Metal GEMV kernel requires F32 input.  If the caller has already
    /// converted the activation to F32 (e.g. to amortise the cost across
    /// multiple QLinear calls that share the same input), calling this method
    /// directly skips the two dtype-conversion kernel launches that
    /// `forward()` would add.
    ///
    /// For the Dense path (QMatMul::Tensor), this falls through to a
    /// standard matmul and then converts the result to F32 if needed.
    #[allow(dead_code)]
    pub fn forward_f32(&self, xs_f32: &Tensor) -> Result<Tensor> {
        debug_assert_eq!(xs_f32.dtype(), DType::F32, "forward_f32 requires F32 input");
        match &self.inner {
            QMatMul::QTensor(_) => {
                // No conversion needed — the GEMV kernel already takes F32.
                let r = self.inner.forward(xs_f32)?;
                // r is F32; return F32 to the caller.
                match &self.bias {
                    None => Ok(r),
                    Some(b) => {
                        let b_f32 = if b.dtype() == DType::F32 {
                            b.clone()
                        } else {
                            b.to_dtype(DType::F32)?
                        };
                        r.broadcast_add(&b_f32)
                    }
                }
            }
            _ => {
                // Dense path: standard matmul, result may be bf16; cast to F32.
                let result = self.inner.forward(xs_f32)?;
                let result = if result.dtype() == DType::F32 {
                    result
                } else {
                    result.to_dtype(DType::F32)?
                };
                match &self.bias {
                    None => Ok(result),
                    Some(b) => {
                        let b_f32 = b.to_dtype(DType::F32)?;
                        result.broadcast_add(&b_f32)
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// QGgufVarBuilder: a lazy VarBuilder for quantized GGUF tensors.
//
// Stores the GGUF file handle and metadata; loads a QTensor into device
// memory only when qlinear_weight is called for a specific projection.
// A shared cache (Mutex<HashMap>) ensures each tensor is loaded at most once.
// ---------------------------------------------------------------------------

/// A lazy VarBuilder that loads tensors from a GGUF file on demand, retaining
/// their quantized form (`Arc<QTensor>`).
///
/// Tensors are loaded only when explicitly requested via `qlinear_weight` or
/// `get_qtensor`, and cached so that each tensor is read from disk at most once.
/// Tensors that are never requested (norms, biases, embeddings) are never
/// loaded, eliminating the double-memory issue of an eager approach.
///
/// Key renaming (GGUF → HuggingFace tensor name mapping) is stored as a pure
/// metadata table and applied lazily at load time, so `rename_keys` never
/// touches device memory or reads from the file.
#[derive(Clone)]
pub struct QGgufVarBuilder {
    file: Arc<std::sync::Mutex<std::fs::File>>,
    content: Arc<candle_core::quantized::gguf_file::Content>,
    cache: Arc<
        std::sync::Mutex<std::collections::HashMap<String, Arc<candle_core::quantized::QTensor>>>,
    >,
    /// Maps HuggingFace tensor names → GGUF tensor names.
    /// Populated by `rename_keys`; empty when no renaming is needed.
    name_remap: Arc<std::collections::HashMap<String, String>>,
    device: Device,
    path: Vec<String>,
}

impl QGgufVarBuilder {
    /// Open a GGUF file and read its metadata. No tensors are loaded into device
    /// memory until `qlinear_weight` or `get_qtensor` is called.
    pub fn from_gguf<P: AsRef<std::path::Path>>(
        p: P,
        device: &Device,
    ) -> candle_core::Result<Self> {
        use candle_core::quantized::gguf_file;
        let mut file = std::fs::File::open(p.as_ref()).map_err(candle_core::Error::from)?;
        let content = gguf_file::Content::read(&mut file)?;
        Ok(Self {
            file: Arc::new(std::sync::Mutex::new(file)),
            content: Arc::new(content),
            cache: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            name_remap: Arc::new(std::collections::HashMap::new()),
            device: device.clone(),
            path: Vec::new(),
        })
    }

    /// Enter a sub-namespace (mirrors `VarBuilder::pp`).
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            file: self.file.clone(),
            content: self.content.clone(),
            cache: self.cache.clone(),
            name_remap: self.name_remap.clone(),
            device: self.device.clone(),
            path,
        }
    }

    /// Build the fully-qualified name for a tensor under the current namespace.
    pub fn full_name(&self, name: &str) -> String {
        if self.path.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.path.join("."), name)
        }
    }

    /// Load a tensor by name, returning the cached copy if already loaded.
    ///
    /// `name` is the HuggingFace-style key; if a `name_remap` table is present
    /// (built by `rename_keys`) it is consulted to find the actual GGUF key
    /// before reading from the file.
    fn load_qtensor(
        &self,
        name: &str,
    ) -> candle_core::Result<Option<Arc<candle_core::quantized::QTensor>>> {
        {
            let cache = self.cache.lock().unwrap();
            if let Some(qt) = cache.get(name) {
                return Ok(Some(qt.clone()));
            }
        }
        // Resolve the GGUF tensor name: use the remap table if present,
        // otherwise assume the name is already in GGUF format.
        let gguf_name: std::borrow::Cow<str> = if let Some(g) = self.name_remap.get(name) {
            std::borrow::Cow::Borrowed(g.as_str())
        } else {
            std::borrow::Cow::Borrowed(name)
        };
        if !self.content.tensor_infos.contains_key(gguf_name.as_ref()) {
            return Ok(None);
        }
        let qt = {
            let mut file = self.file.lock().unwrap();
            self.content
                .tensor(&mut *file, gguf_name.as_ref(), &self.device)?
        };
        let qt = Arc::new(qt);
        self.cache
            .lock()
            .unwrap()
            .insert(name.to_string(), qt.clone());
        Ok(Some(qt))
    }

    /// Retrieve the raw `Arc<QTensor>` for the "weight" tensor at the current path.
    ///
    /// Returns `None` if the tensor is not present in the GGUF file.
    pub fn get_qtensor(&self) -> Option<Arc<candle_core::quantized::QTensor>> {
        let name = self.full_name("weight");
        self.load_qtensor(&name).ok().flatten()
    }

    /// Retrieve the raw `Arc<QTensor>` for a tensor with an arbitrary suffix
    /// (not necessarily "weight") at the current path.
    ///
    /// Used for HF parameters stored without the `.weight` suffix, such as
    /// `experts.gate_up_proj` and `experts.down_proj` in Gemma4 MoE.
    pub fn get_qtensor_named(&self, suffix: &str) -> Option<Arc<candle_core::quantized::QTensor>> {
        let name = self.full_name(suffix);
        self.load_qtensor(&name).ok().flatten()
    }

    /// Build a bias-free `QLinear` from the "weight" tensor at the current path.
    ///
    /// Errors if the tensor is absent from the GGUF file.
    pub fn qlinear_weight(&self) -> Result<QLinear> {
        let name = self.full_name("weight");
        match self.load_qtensor(&name)? {
            Some(qt) => QLinear::from_qtensor(qt, None),
            None => candle_core::bail!("QGgufVarBuilder: tensor not found: {name}"),
        }
    }

    /// Try to build a `QLinear`; returns `None` if the tensor is absent.
    pub fn try_qlinear_weight(&self) -> Option<Result<QLinear>> {
        let name = self.full_name("weight");
        match self.load_qtensor(&name) {
            Ok(Some(qt)) => Some(QLinear::from_qtensor(qt, None)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    /// Re-key all tensors using a mapping function that translates raw GGUF
    /// tensor names to HF-style names.
    ///
    /// Because the lazy loader looks up tensors by name in the GGUF content
    /// metadata, this method eagerly loads all tensors under their mapped names
    /// Build a key-remapping table that maps HuggingFace tensor names to GGUF
    /// tensor names.  No tensors are loaded from disk; the mapping is applied
    /// lazily inside `load_qtensor` when a tensor is first requested.
    ///
    /// This replaces the previous eager implementation that loaded every GGUF
    /// tensor into device memory upfront, causing a large transient RSS spike
    /// and preventing parallelisation with other startup work.
    pub fn rename_keys<F: Fn(&str) -> String>(&self, map_fn: F) -> Result<Self> {
        let remap: std::collections::HashMap<String, String> = self
            .content
            .tensor_infos
            .keys()
            .map(|gguf_name| (map_fn(gguf_name), gguf_name.clone()))
            .collect();
        Ok(Self {
            file: self.file.clone(),
            content: self.content.clone(),
            cache: self.cache.clone(),
            name_remap: Arc::new(remap),
            device: self.device.clone(),
            path: self.path.clone(),
        })
    }
}

/// Build a bias-free QLinear layer.
///
/// If `qvb` is `Some`, keeps the weight as QTensor (quantized GGUF path).
/// If `qvb` is `None`, loads the dequantized tensor from `vb`, with NVFP4
/// dequantization applied automatically when the weight is stored in that format.
///
/// Both `vb` and `qvb` are already `.pp("layer_name")` scoped.
pub fn qlinear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
    qvb: Option<&QGgufVarBuilder>,
) -> Result<QLinear> {
    // Load bias from the dense VarBuilder when requested.  The GGUF path also
    // uses vb for bias since bias vectors are stored at F16 (not quantized).
    let b = if bias {
        Some(vb.get(out_dim, "bias")?)
    } else {
        None
    };
    if let Some(q) = qvb {
        let mut ql = q.qlinear_weight()?;
        ql.bias = b;
        Ok(ql)
    } else {
        // Check for NVFP4 quantized weight (U8 packed FP4 + F8E4M3 block scales).
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let weight = if let Some(w) =
            crate::nvfp4::try_load_from_varbuilder(&vb, out_dim, in_dim, dtype, &device)?
        {
            w
        } else {
            vb.get((out_dim, in_dim), "weight")?
        };

        // Online Q4K quantization: convert BF16 weights to a QTensor so that
        // Metal's quantized GEMV kernel (call_quantized_matmul_mv_t) is used at
        // decode time instead of a full dense matmul.  Q4K (Q4_K_M) uses 256-
        // element blocks; all major projection dimensions are multiples of 256.
        // For any tensor whose element count is not a multiple of 256 (e.g. small
        // adapter layers) we fall back to Q8_0 (block_size=32), and if that also
        // fails we keep the original dense BF16 path.
        //
        // Q4K is ~4.5 bits/weight — 4× less bandwidth than BF16 GEMV — matching
        // the decode throughput of a pre-built Q4K GGUF without requiring --quantize.
        let elem_count = weight.elem_count();
        let quant_dtype = if elem_count % 256 == 0 {
            Some(candle_core::quantized::GgmlDType::Q4K)
        } else if elem_count % 32 == 0 {
            Some(candle_core::quantized::GgmlDType::Q8_0)
        } else {
            None
        };
        if let Some(dtype) = quant_dtype {
            match candle_core::quantized::QTensor::quantize(&weight, dtype) {
                Ok(qt) => return QLinear::from_qtensor(Arc::new(qt), b),
                Err(e) => {
                    tracing::debug!(
                        "online {dtype:?} quantization failed for [{out_dim}×{in_dim}], \
                         falling back to BF16: {e}"
                    );
                }
            }
        }
        Ok(QLinear::from_tensor(weight, b))
    }
}

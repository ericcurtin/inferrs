//! Gemma 4 text-only language model implementation (google variant).
//!
//! This implements the simplified Gemma 4 text model as represented in the
//! `google/gemma-4-E2B-it` checkpoint.  The full Gemma 3n model includes
//! AltUp, Laurel, and KV-sharing components; this variant omits them and uses
//! a straightforward transformer decoder with per-layer input residuals.
//!
//! Key differences from Gemma 3:
//!
//! * **Dual head dims**: sliding layers use `head_dim=256`, global (full)
//!   attention layers use `global_head_dim=512`.
//! * **Dual RoPE**: sliding uses `rope_theta=10_000` (full rotation), global
//!   uses `rope_theta=1_000_000` with `partial_rotary_factor=0.25`.
//! * **Per-layer input residual**: each layer receives a per-token embedding
//!   from `embed_tokens_per_layer`, combined with a projection from the hidden
//!   state via `per_layer_model_projection`, gated and added to the hidden
//!   state after the MLP sub-layer.
//! * **Double-wide MLP**: the second half of layers uses `intermediate_size*2`.
//! * **Embedding scale**: `embed_tokens` is scaled by `sqrt(hidden_size)`,
//!   `embed_tokens_per_layer` is scaled by `sqrt(hidden_size_per_layer_input)`.
//! * All language-model weights live under `model.language_model.*`.

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{rms_norm, Activation, RmsNorm, VarBuilder};
use std::sync::Arc;

use crate::turbo_quant::{TurboQuantConfig, TurboQuantKvCache, MIN_KV_BUFFER_CAP};

// ---------------------------------------------------------------------------
// PLI embedding cache size
//
// The per-layer input (PLI) embedding is a pure function of the token ID,
// so the result of the CPU lookup + CPU→GPU DMA + scale multiply can be
// reused across decode steps.  We cache at most this many entries; older
// entries are evicted in LRU order.
const PLI_EMBED_CACHE_SIZE: usize = 1024;

// ---------------------------------------------------------------------------
// PliEmbeddingTable: memory-efficient quantized PLI embedding lookup
//
// The `embed_tokens_per_layer` table is [vocab_size, num_layers * pli_dim]
// and is 4.7 GB in BF16 for Gemma4-E2B-it.  Loading it fully dequantized
// consumes most of the process RSS.
//
// `PliEmbeddingTable` stores the table in its GGUF-quantized form (Q6K,
// ~1.93 GB for E2B) and dequantizes one row at a time on demand.  This
// reduces peak memory by ~2.8 GB.
//
// The per-row dequantization (Q6K → BF16 for 8960 elements) takes ~20 µs
// on a modern CPU — a one-time cost per unique token, amortised by the
// PLI embed cache above.
// ---------------------------------------------------------------------------

/// Memory-efficient per-row embedding lookup for the PLI embedding table.
///
/// The `embed_tokens_per_layer` table is [vocab_size, num_layers * pli_dim]
/// and is 4.7 GB in BF16 for Gemma4-E2B-it.  When the GGUF path is active,
/// this struct keeps the table in its quantized form (Q6K, ~1.9 GB) and
/// dequantizes one row at a time on demand — reducing peak CPU RAM by ~2.8 GB.
///
/// When loaded from safetensors (no --quantize), the full BF16 tensor is
/// wrapped and row lookups use `index_select`.
#[derive(Clone, Debug)]
enum PliEmbeddingTable {
    /// GGUF-file-backed on-demand row lookup (zero CPU RAM for the table).
    ///
    /// Rows are read directly from the GGUF file as needed, with results
    /// cached by token ID (via the model-level `pli_embed_cache`).  This
    /// keeps the PLI embedding table out of CPU RAM entirely, reducing peak
    /// RSS by ~1.9 GB compared to the `Quantized` variant.
    GgufFile {
        /// GGUF file, shared among all model components.
        file: Arc<std::sync::Mutex<std::fs::File>>,
        /// Absolute file offset where tensor data starts.
        tensor_offset: u64,
        #[allow(dead_code)]
        vocab_size: usize,
        #[allow(dead_code)]
        embed_dim: usize,
        /// Bytes per row (determined by the quantization format).
        row_bytes: usize,
        /// GGUF quantization type for the PLI embedding.
        dtype_q: candle_core::quantized::GgmlDType,
    },
    /// Quantized form (GGUF path): raw bytes + shape info for the full table.
    /// `qtensor` is kept alive so the data pointer remains valid.
    Quantized {
        qtensor: Arc<candle_core::quantized::QTensor>,
        #[allow(dead_code)]
        vocab_size: usize,
        #[allow(dead_code)]
        embed_dim: usize,
        /// Bytes per row in the raw quantized layout.
        row_bytes: usize,
    },
    /// Dequantized BF16 tensor (safetensors path).
    Dense(candle_core::Tensor),
}

impl PliEmbeddingTable {
    /// Build from an open GGUF file (zero-copy, minimal RAM usage).
    ///
    /// Reads individual rows directly from the GGUF file on demand.
    /// The PLI embedding table is NOT loaded into CPU RAM.
    fn from_gguf_file(gguf_path: &std::path::Path) -> candle_core::Result<Option<Self>> {
        use candle_core::quantized::gguf_file;

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::from)?;
        let content = gguf_file::Content::read(&mut file)?;

        // Find the PLI embedding tensor.
        let tensor_name = "model.language_model.embed_tokens_per_layer.weight";
        let info = match content.tensor_infos.get(tensor_name) {
            Some(t) => t,
            None => return Ok(None),
        };

        let embed_dim = info.shape.dims()[1];
        let vocab_size = info.shape.dims()[0];
        let dtype_q = info.ggml_dtype;
        let block_size = dtype_q.block_size();
        let type_size = dtype_q.type_size();

        if embed_dim % block_size != 0 {
            return Ok(None); // non-aligned, can't safely slice by row
        }

        let blocks_per_row = embed_dim / block_size;
        let row_bytes = blocks_per_row * type_size;
        let tensor_offset = content.tensor_data_offset + info.offset;

        Ok(Some(PliEmbeddingTable::GgufFile {
            file: Arc::new(std::sync::Mutex::new(file)),
            tensor_offset,
            vocab_size,
            embed_dim,
            row_bytes,
            dtype_q,
        }))
    }

    /// Build from an `Arc<QTensor>` (GGUF quantized path, ~1.9 GB RAM).
    fn from_qtensor(qt: Arc<candle_core::quantized::QTensor>) -> candle_core::Result<Self> {
        let (vocab_size, embed_dim) = qt.shape().dims2()?;
        let total_bytes = qt.data()?.len();
        let row_bytes = total_bytes / vocab_size;
        Ok(PliEmbeddingTable::Quantized {
            qtensor: qt,
            vocab_size,
            embed_dim,
            row_bytes,
        })
    }

    /// Build from a dequantized CPU tensor (safetensors path).
    fn from_tensor(t: candle_core::Tensor) -> Self {
        PliEmbeddingTable::Dense(t)
    }

    /// Dequantize or look up a batch of token IDs.
    ///
    /// `token_ids`: flat slice of token IDs (one per token in the batch).
    ///
    /// Returns a CPU BF16 tensor of shape `[n_tokens, embed_dim]`.
    fn lookup_batch(
        &self,
        token_ids: &[u32],
        embed_dim: usize,
        dtype: candle_core::DType,
    ) -> candle_core::Result<candle_core::Tensor> {
        use candle_core::quantized::{QStorage, QTensor};
        use std::borrow::Cow;
        use std::io::{Read, Seek, SeekFrom};

        match self {
            PliEmbeddingTable::GgufFile {
                file,
                tensor_offset,
                row_bytes,
                dtype_q,
                ..
            } => {
                let n = token_ids.len();
                let mut row_data: Vec<u8> = vec![0u8; n * row_bytes];
                let mut f = file.lock().expect("GGUF file lock poisoned");
                for (i, &tok) in token_ids.iter().enumerate() {
                    let file_pos = tensor_offset + tok as u64 * *row_bytes as u64;
                    f.seek(SeekFrom::Start(file_pos))
                        .map_err(candle_core::Error::from)?;
                    f.read_exact(&mut row_data[i * row_bytes..(i + 1) * row_bytes])
                        .map_err(candle_core::Error::from)?;
                }
                let storage =
                    QStorage::from_data(Cow::Owned(row_data), &candle_core::Device::Cpu, *dtype_q)?;
                let row_qt = QTensor::new(storage, (n, embed_dim))?;
                row_qt
                    .dequantize(&candle_core::Device::Cpu)?
                    .to_dtype(dtype)
            }
            PliEmbeddingTable::Quantized {
                qtensor, row_bytes, ..
            } => {
                let raw = qtensor.data()?;
                let n = token_ids.len();
                let dtype_q = qtensor.dtype();

                // Collect the raw bytes for all requested rows.
                let mut row_data: Vec<u8> = Vec::with_capacity(n * row_bytes);
                for &tok in token_ids {
                    let start = tok as usize * row_bytes;
                    let end = start + row_bytes;
                    row_data.extend_from_slice(&raw[start..end]);
                }

                // Build a QTensor of shape [n, embed_dim] from the gathered bytes.
                let storage =
                    QStorage::from_data(Cow::Owned(row_data), &candle_core::Device::Cpu, dtype_q)?;
                let row_qt = QTensor::new(storage, (n, embed_dim))?;
                row_qt
                    .dequantize(&candle_core::Device::Cpu)?
                    .to_dtype(dtype)
            }
            PliEmbeddingTable::Dense(t) => {
                // Use index_select for the dense (safetensors) path.
                let ids = candle_core::Tensor::new(token_ids, &candle_core::Device::Cpu)?;
                t.index_select(&ids, 0)?.to_dtype(dtype)
            }
        }
    }

    /// Shorthand: look up a single token.
    fn lookup_single(
        &self,
        token_id: u32,
        embed_dim: usize,
        dtype: candle_core::DType,
    ) -> candle_core::Result<candle_core::Tensor> {
        self.lookup_batch(&[token_id], embed_dim, dtype)
            .and_then(|t| t.reshape((1usize, 1usize, embed_dim)))
    }
}

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
            _ => {
                // Dense path (safetensors bf16): standard matmul.
                let result = self.inner.forward(xs)?;
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

// NVFP4 dequantization is implemented in crate::nvfp4 (shared with quantize.rs).

// ---------------------------------------------------------------------------
// QGgufVarBuilder: a VarBuilder for quantized GGUF tensors.
//
// Provides Arc<QTensor> access for named tensors, enabling QMatMul::QTensor
// construction that uses the Metal quantized GEMV kernel path.
// ---------------------------------------------------------------------------

/// A VarBuilder that holds tensors from a GGUF file in their original
/// quantized form (rather than dequantizing them upfront to bf16).
///
/// Used alongside the standard `VarBuilder` (for non-quantized tensors such
/// as norms and embeddings) to provide `QLinear`-backed projections.
#[derive(Clone)]
pub struct QGgufVarBuilder {
    data: Arc<std::collections::HashMap<String, Arc<candle_core::quantized::QTensor>>>,
    path: Vec<String>,
}

impl QGgufVarBuilder {
    /// Load projection tensors from a GGUF file, retaining their quantized format.
    ///
    /// Only tensors that are used as linear projection weights (i.e. those that
    /// will be called via `QLinear` for GEMV/GEMM) are loaded into Metal memory
    /// here.  Embedding tables (`embed_tokens`, `embed_tokens_per_layer`) are
    /// deliberately excluded: they are used only for index-select lookups and
    /// do not benefit from the quantized GEMV kernel path.  They are loaded
    /// separately via the standard `VarBuilder` (dequantized to bf16) and can
    /// be very large (e.g. `embed_tokens_per_layer` is 4.7 GB in bf16 for E2B).
    /// Loading them twice would double their memory footprint.
    ///
    /// Exception: `embed_tokens.weight` (the tied lm_head) IS loaded here
    /// because it is also used as the output projection (lm_head GEMV), which
    /// is the single most expensive per-token operation.
    pub fn from_gguf<P: AsRef<std::path::Path>>(
        p: P,
        device: &Device,
    ) -> candle_core::Result<Self> {
        use candle_core::quantized::gguf_file;
        let mut file = std::fs::File::open(p.as_ref()).map_err(candle_core::Error::from)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut data = std::collections::HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            // Skip the PLI embedding table — it is loaded via the memory-efficient
            // GGUF file-backed PliEmbeddingTable::GgufFile which reads rows on
            // demand without loading 1.9 GB into CPU RAM.
            if tensor_name.contains("embed_tokens_per_layer") {
                continue;
            }
            let qt = content.tensor(&mut file, tensor_name, device)?;
            data.insert(tensor_name.to_string(), Arc::new(qt));
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
        })
    }

    /// Enter a sub-namespace (mirrors `VarBuilder::pp`).
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
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

    /// Retrieve the raw `Arc<QTensor>` for the "weight" tensor at the current path.
    ///
    /// Returns `None` if the tensor is not found in the GGUF data map.
    pub fn get_qtensor(&self) -> Option<Arc<candle_core::quantized::QTensor>> {
        let name = self.full_name("weight");
        self.data.get(&name).cloned()
    }

    /// Build a bias-free `QLinear` for the "weight" tensor at the current path.
    ///
    /// Returns `QMatMul::QTensor` (fast Metal quantized GEMV) when the tensor
    /// is present; errors if absent.
    /// Build a bias-free `QLinear` from the "weight" tensor at the current path.
    pub fn qlinear_weight(&self) -> Result<QLinear> {
        let name = self.full_name("weight");
        match self.data.get(&name) {
            Some(qt) => QLinear::from_qtensor(qt.clone(), None),
            None => candle_core::bail!("QGgufVarBuilder: tensor not found: {name}"),
        }
    }

    /// Try to build a `QLinear`; returns `None` if the tensor is absent.
    pub fn try_qlinear_weight(&self) -> Option<Result<QLinear>> {
        let name = self.full_name("weight");
        self.data
            .get(&name)
            .map(|qt| QLinear::from_qtensor(qt.clone(), None))
    }
}

/// Build a bias-free QLinear layer.
///
/// If `qvb` is `Some`, keeps the weight as QTensor (quantized GGUF path).
/// If `qvb` is `None`, loads the dequantized tensor from `vb`, with NVFP4
/// dequantization applied automatically when the weight is stored in that format.
///
/// Both `vb` and `qvb` are already `.pp("layer_name")` scoped.
fn qlinear_b(
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
        Ok(QLinear::from_tensor(weight, b))
    }
}

/// Configuration for the Gemma 4 language model.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Gemma4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// KV heads for global (full) attention layers.
    /// Defaults to `num_key_value_heads` when absent from config.
    pub num_global_key_value_heads: usize,
    /// Head dimension for sliding-window attention layers.
    pub head_dim: usize,
    /// Head dimension for global (full) attention layers.
    pub global_head_dim: usize,
    /// Per-layer residual embedding dimension (hidden_size_per_layer_input).
    pub hidden_size_per_layer_input: usize,
    pub rms_norm_eps: f64,
    pub rope_theta_sliding: f64,
    pub rope_theta_global: f64,
    /// Fraction of head_dim that is rotated in global attention RoPE.
    pub partial_rotary_factor_global: f64,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub max_position_embeddings: usize,
    pub final_logit_softcapping: Option<f64>,
    pub attn_logit_softcapping: Option<f64>,
    pub query_pre_attn_scalar: usize,
    pub attention_bias: bool,
    /// When true, global attention layers share V with K (no separate v_proj).
    pub attention_k_eq_v: bool,
    pub hidden_activation: Activation,
    pub tie_word_embeddings: bool,
    /// `true` for each layer that uses global (full) attention.
    pub layer_is_full_attention: Vec<bool>,
    /// Layer index from which the MLP uses `intermediate_size * 2`.
    /// Set to `num_hidden_layers` to disable.
    pub double_wide_mlp_start_layer: usize,
    /// First layer index that shares K,V from a donor layer.
    /// Equal to `num_hidden_layers - num_kv_shared_layers`.
    /// Set to `num_hidden_layers` when there is no KV sharing.
    pub first_kv_shared_idx: usize,
    /// When `Some(bits)`, KV cache vectors are quantized using TurboQuant at the given bit-width.
    pub turbo_quant_bits: Option<u8>,
    pub dtype: DType,
    pub device: Device,
}

// ---------------------------------------------------------------------------
// Helpers for applying RmsNorm to multi-dimensional tensors
// ---------------------------------------------------------------------------

/// Apply `candle_nn::RmsNorm` to the last dimension of a 4-D tensor
/// `[b, h, t, d]` using the fused Metal/CUDA kernel path.
///
/// The Metal `rmsnorm` kernel operates on `elem_count / last_dim` independent
/// vectors of length `last_dim`, regardless of the number of leading dimensions.
/// This means we can pass the 4-D tensor directly after making it contiguous —
/// no reshape to `[b*h*t, d]` and back is required.  Removing the two reshape
/// calls saves two Tensor metadata allocations per invocation.
///
/// Precondition: `norm` must have a 1-D weight of length equal to the last
/// dimension of `x` (standard `rms_norm(d, eps, vb)` construction ensures this).
#[inline]
#[allow(dead_code)]
fn apply_rms_norm_4d(x: &Tensor, norm: &RmsNorm) -> Result<Tensor> {
    norm.forward(&x.contiguous()?)
}

/// Apply a weight-free (scale=1) RMSNorm to a 4-D tensor `[b, h, t, d]` using
/// the fused `candle_nn::ops::rms_norm` kernel.
///
/// Same rationale as `apply_rms_norm_4d`: the Metal kernel handles arbitrary
/// leading dimensions, so the reshape to 2-D and back is unnecessary.
///
/// `weight` must be a 1-D all-ones tensor of length `d` (pre-allocated at
/// construction time to avoid runtime allocations).
#[inline]
fn apply_rms_norm_4d_with_weight(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    candle_nn::ops::rms_norm(&x.contiguous()?, weight, eps)
}

// ---------------------------------------------------------------------------
// Rotary Embedding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    /// F32 copies for the full-F32 decode fast-path (avoids per-step dtype conversion).
    sin_f32: Tensor,
    cos_f32: Tensor,
    /// Number of dimensions that are rotated per head.
    rotary_dim: usize,
}

impl RotaryEmbedding {
    /// Standard (full) RoPE — all `head_dim` features are rotated.
    fn new_standard(
        dtype: DType,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let rotary_dim = head_dim;
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;
        let sin_f32 = sin.to_dtype(DType::F32)?;
        let cos_f32 = cos.to_dtype(DType::F32)?;
        Ok(Self {
            sin,
            cos,
            sin_f32,
            cos_f32,
            rotary_dim,
        })
    }

    /// Partial RoPE — only the first `round(head_dim * factor)` (even) features are rotated.
    ///
    /// The reference implementation (`_compute_proportional_rope_parameters`) uses
    /// `head_dim` (not `rotary_dim`) as the denominator when computing inv_freq:
    ///
    ///   rope_angles = int(partial_rotary_factor * head_dim / 2)  # number of freq pairs
    ///   inv_freq[k] = 1 / (rope_theta ^ (2k / head_dim))         # divide by head_dim
    ///
    /// The remaining `head_dim/2 - rope_angles` pairs get inv_freq=0 (no rotation).
    fn new_partial(
        dtype: DType,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        dev: &Device,
    ) -> Result<Self> {
        // Number of (cos, sin) frequency pairs that are actually rotated.
        let rope_angles =
            ((partial_rotary_factor * head_dim as f64 / 2.0).floor() as usize).min(head_dim / 2);
        // rotary_dim = 2 * rope_angles (the number of scalar features that get RoPE)
        let rotary_dim = rope_angles * 2;

        // Frequencies use head_dim in the exponent denominator (not rotary_dim).
        let inv_freq: Vec<f32> = (0..rope_angles)
            .map(|k| 1f32 / rope_theta.powf(2.0 * k as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;
        let sin_f32 = sin.to_dtype(DType::F32)?;
        let cos_f32 = cos.to_dtype(DType::F32)?;
        Ok(Self {
            sin,
            cos,
            sin_f32,
            cos_f32,
            rotary_dim,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, head_dim) = q.dims4()?;
        // Use F32 cos/sin when q is F32 (full-F32 decode fast-path).
        let (cos_src, sin_src) = if q.dtype() == DType::F32 {
            (&self.cos_f32, &self.sin_f32)
        } else {
            (&self.cos, &self.sin)
        };
        let cos = cos_src.narrow(0, seqlen_offset, seq_len)?;
        let sin = sin_src.narrow(0, seqlen_offset, seq_len)?;

        if self.rotary_dim == head_dim {
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        } else {
            // Partial RoPE: split into [rotated | passthrough]
            let q_rot = q.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
            let q_pass = q.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
            let k_rot = k.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
            let k_pass = k.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;

            let q_rot = candle_nn::rotary_emb::rope(&q_rot, &cos, &sin)?;
            let k_rot = candle_nn::rotary_emb::rope(&k_rot, &cos, &sin)?;

            let q_embed = Tensor::cat(&[q_rot, q_pass.contiguous()?], D::Minus1)?;
            let k_embed = Tensor::cat(&[k_rot, k_pass.contiguous()?], D::Minus1)?;
            Ok((q_embed, k_embed))
        }
    }

    /// Apply RoPE only to the query tensor (used for KV-sharing layers where K is reused).
    fn apply_rotary_emb_q(&self, q: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_sz, _h, seq_len, head_dim) = q.dims4()?;
        let (cos_src, sin_src) = if q.dtype() == DType::F32 {
            (&self.cos_f32, &self.sin_f32)
        } else {
            (&self.cos, &self.sin)
        };
        let cos = cos_src.narrow(0, seqlen_offset, seq_len)?;
        let sin = sin_src.narrow(0, seqlen_offset, seq_len)?;

        if self.rotary_dim == head_dim {
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)
        } else {
            let q_rot = q.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
            let q_pass = q.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
            let q_rot = candle_nn::rotary_emb::rope(&q_rot, &cos, &sin)?;
            Tensor::cat(&[q_rot, q_pass.contiguous()?], D::Minus1)
        }
    }
}

// ---------------------------------------------------------------------------
// MLP (SwiGLU)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
    act_fn: Activation,
}

impl Mlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        bias: bool,
        act_fn: Activation,
        vb: VarBuilder,
        qvb: Option<&QGgufVarBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: qlinear_b(
                hidden_size,
                intermediate_size,
                bias,
                vb.pp("gate_proj"),
                qvb.map(|q| q.pp("gate_proj")).as_ref(),
            )?,
            up_proj: qlinear_b(
                hidden_size,
                intermediate_size,
                bias,
                vb.pp("up_proj"),
                qvb.map(|q| q.pp("up_proj")).as_ref(),
            )?,
            down_proj: qlinear_b(
                intermediate_size,
                hidden_size,
                bias,
                vb.pp("down_proj"),
                qvb.map(|q| q.pp("down_proj")).as_ref(),
            )?,
            act_fn,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // The BF16-native GEMV kernel handles BF16 input directly, so no
        // pre-conversion to F32 is needed. Each QLinear::forward call uses
        // the fused BF16→Q8_1 path internally.
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ---------------------------------------------------------------------------
// KV Cache (normal or rotating)
// ---------------------------------------------------------------------------

/// A retaining rotating KV cache for sliding-window attention layers.
///
/// This is a drop-in replacement for `candle_nn::kv_cache::RotatingKvCache`
/// that **retains its pre-allocated Metal buffer across sequence resets**.
///
/// ## Problem with `RotatingKvCache::reset()`
///
/// `candle_nn::kv_cache::RotatingCache::reset()` sets `all_data = None`, which
/// drops the Metal buffer and forces a fresh `Tensor::zeros(…, max_seq_len, …)`
/// allocation on the next decode step.  For Gemma4-E2B-it with `sliding_window=512`
/// and `head_dim=256`, each K or V sliding buffer is 512 × 256 × 2 bytes (bf16)
/// = 256 KiB.  With 28 sliding layers that is 28 × 2 × 256 KiB ≈ 14 MiB of
/// Metal buffer allocations (+ zero-fills) issued on every sequence reset.
///
/// By retaining the buffer and only resetting the write-position and
/// sequence-length counters, the expensive allocation+zero-fill path is paid
/// at most once (on the first decode step of the first ever sequence).
/// All subsequent sequences reuse the existing Metal buffer, improving
/// TTFT and per-token decode latency.
///
/// ## Rotation semantics
///
/// The buffer is a fixed-size circular store of `max_seq_len` tokens.  When the
/// sequence length exceeds `max_seq_len` the oldest tokens are overwritten.
/// The returned tensor always covers `min(seq_len, max_seq_len)` tokens from the
/// circular buffer (a contiguous view when the write pointer has not yet wrapped,
/// or the full buffer when it has).
#[derive(Debug, Clone)]
struct RetainingRotatingKvCache {
    k_buf: Option<candle_core::Tensor>,
    v_buf: Option<candle_core::Tensor>,
    /// Current write position (mod max_seq_len).
    offset: usize,
    /// Total number of tokens seen (grows unboundedly, not clamped).
    current_seq_len: usize,
    /// Maximum size of the circular buffer.
    max_seq_len: usize,
}

impl RetainingRotatingKvCache {
    fn new(max_seq_len: usize) -> Self {
        Self {
            k_buf: None,
            v_buf: None,
            offset: 0,
            current_seq_len: 0,
            max_seq_len,
        }
    }

    /// Append `k` / `v` (shape `[b, n_kv, t, d]`) to the rotating cache.
    ///
    /// Returns the accumulated `(k, v)` tensors of shape
    /// `[b, n_kv, min(seq_len, max_seq_len), d]`.
    fn append(
        &mut self,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
    ) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
        let t = k.dim(2)?; // number of new tokens

        // Lazily allocate the rotating buffer on the first call.
        if self.k_buf.is_none() {
            let mut shape = k.dims().to_vec();
            shape[2] = self.max_seq_len;
            self.k_buf = Some(candle_core::Tensor::zeros(
                shape.as_slice(),
                k.dtype(),
                k.device(),
            )?);
            let mut shape = v.dims().to_vec();
            shape[2] = self.max_seq_len;
            self.v_buf = Some(candle_core::Tensor::zeros(
                shape.as_slice(),
                v.dtype(),
                v.device(),
            )?);
        }

        let kb = self.k_buf.as_mut().expect("k_buf initialised above");
        let vb = self.v_buf.as_mut().expect("v_buf initialised above");

        // Write the new tokens into the circular buffer using slice_set.
        // When the new tokens fit without wrapping, a single slice_set suffices.
        // When they wrap around the end of the buffer, we split into two writes.
        self.current_seq_len += t;

        if t >= self.max_seq_len {
            // Rare: new tokens fill or overflow the entire buffer.
            // Write the last max_seq_len tokens starting at position 0.
            let start = t - self.max_seq_len;
            let k_tail = k.narrow(2, start, self.max_seq_len)?.contiguous()?;
            let v_tail = v.narrow(2, start, self.max_seq_len)?.contiguous()?;
            kb.slice_set(&k_tail, 2, 0)?;
            vb.slice_set(&v_tail, 2, 0)?;
            self.offset = 0;
            return Ok((kb.clone(), vb.clone()));
        }

        let rem = self.max_seq_len - self.offset;
        if t <= rem {
            // All new tokens fit before the end of the buffer — single write.
            kb.slice_set(&k.contiguous()?, 2, self.offset)?;
            vb.slice_set(&v.contiguous()?, 2, self.offset)?;
            self.offset = (self.offset + t) % self.max_seq_len;
        } else {
            // New tokens wrap around — two writes.
            let k1 = k.narrow(2, 0, rem)?.contiguous()?;
            let v1 = v.narrow(2, 0, rem)?.contiguous()?;
            kb.slice_set(&k1, 2, self.offset)?;
            vb.slice_set(&v1, 2, self.offset)?;

            let k2 = k.narrow(2, rem, t - rem)?.contiguous()?;
            let v2 = v.narrow(2, rem, t - rem)?.contiguous()?;
            kb.slice_set(&k2, 2, 0)?;
            vb.slice_set(&v2, 2, 0)?;
            self.offset = t - rem;
        }

        // Return a view of the valid portion of the circular buffer.
        let valid = self.current_seq_len.min(self.max_seq_len);
        let k_out = if valid == self.max_seq_len {
            kb.clone()
        } else {
            kb.narrow(2, 0, valid)?
        };
        let v_out = if valid == self.max_seq_len {
            vb.clone()
        } else {
            vb.narrow(2, 0, valid)?
        };
        Ok((k_out, v_out))
    }

    /// Reset the write-position and sequence-length counters **without dropping
    /// the Metal buffer**.
    ///
    /// The next `append` call will overwrite from position 0, so stale data
    /// beyond `current_seq_len` is never exposed to the attention kernel.
    fn reset(&mut self) {
        self.offset = 0;
        self.current_seq_len = 0;
        // Intentionally retain k_buf / v_buf so the Metal allocation is reused.
    }
}

/// A KV cache for a single K or V tensor that retains its pre-allocated Metal
/// buffer across sequence resets and grows it lazily by doubling.
///
/// ## Original design — problem
///
/// The previous implementation allocated `Tensor::zeros([b, n_kv, max_seq_len, d])`
/// on the *first decode step ever*, where `max_seq_len = 131_072` and `head_dim = 512`
/// for Gemma4 global layers.  That is 128 MiB per K or V buffer.  With 7 global
/// attention layers (3 non-shared + 4 KV-sharing) the first request triggered up to
/// **1.75 GiB** of Metal `newBuffer + zero-fill` commands before a single token was
/// generated.
///
/// ## This design — solution
///
/// Between requests, `reset()` zeroes only the sequence-length counter; the Metal
/// buffer is retained so the next request reuses the same allocation (as before).
/// If the next request is longer than the current capacity, the buffer is grown
/// then — only one copy is needed per doubling, so long conversations pay O(log N)
/// grow operations total.
#[derive(Debug, Clone)]
struct RetainingKvCache {
    k_buf: Option<candle_core::Tensor>,
    v_buf: Option<candle_core::Tensor>,
    /// Number of valid tokens currently stored in the buffer.
    seq_len: usize,
    /// Number of tokens the current buffer can hold (always a power of two ≥ 256).
    buf_cap: usize,
    /// Hard upper limit (from `max_position_embeddings`).
    max_seq_len: usize,
}

impl RetainingKvCache {
    fn new(max_seq_len: usize) -> Self {
        Self {
            k_buf: None,
            v_buf: None,
            seq_len: 0,
            buf_cap: 0,
            max_seq_len,
        }
    }

    /// Append `k` / `v` (shape `[b, n_kv, t, d]`) to the cache.
    ///
    /// Returns the accumulated `(k, v)` tensors of shape `[b, n_kv, seq_len, d]`.
    fn append(
        &mut self,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
    ) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
        let t = k.dim(2)?; // number of new tokens
        let needed = self.seq_len + t;

        if needed > self.max_seq_len {
            candle_core::bail!(
                "RetainingKvCache: above max-seq-len {}+{}>{}",
                self.seq_len,
                t,
                self.max_seq_len
            );
        }

        // Grow the buffer if necessary (double until capacity ≥ needed).
        if needed > self.buf_cap {
            let new_cap = needed
                .next_power_of_two()
                .max(MIN_KV_BUFFER_CAP)
                .min(self.max_seq_len);

            let mut k_shape = k.dims().to_vec();
            k_shape[2] = new_cap;
            let new_k_buf = candle_core::Tensor::zeros(k_shape.as_slice(), k.dtype(), k.device())?;
            let mut v_shape = v.dims().to_vec();
            v_shape[2] = new_cap;
            let new_v_buf = candle_core::Tensor::zeros(v_shape.as_slice(), v.dtype(), v.device())?;

            // Copy existing valid tokens into the new (larger) buffer.
            if self.seq_len > 0 {
                if let (Some(kb_old), Some(vb_old)) = (&self.k_buf, &self.v_buf) {
                    let k_valid = kb_old.narrow(2, 0, self.seq_len)?;
                    let v_valid = vb_old.narrow(2, 0, self.seq_len)?;
                    new_k_buf.slice_set(&k_valid.contiguous()?, 2, 0)?;
                    new_v_buf.slice_set(&v_valid.contiguous()?, 2, 0)?;
                }
            }

            self.k_buf = Some(new_k_buf);
            self.v_buf = Some(new_v_buf);
            self.buf_cap = new_cap;
        }

        let kb = self.k_buf.as_mut().expect("k_buf allocated above");
        let vb = self.v_buf.as_mut().expect("v_buf allocated above");

        kb.slice_set(&k.contiguous()?, 2, self.seq_len)?;
        vb.slice_set(&v.contiguous()?, 2, self.seq_len)?;
        self.seq_len += t;

        let k_out = kb.narrow(2, 0, self.seq_len)?;
        let v_out = vb.narrow(2, 0, self.seq_len)?;
        Ok((k_out, v_out))
    }

    /// Reset the sequence-length counter **without dropping the Metal buffer**.
    ///
    /// The next `append` call will overwrite from position 0, so stale data
    /// beyond `seq_len` is never read.  The buffer capacity is retained for reuse.
    fn reset(&mut self) {
        self.seq_len = 0;
        // Intentionally retain k_buf / v_buf and buf_cap so the Metal allocation
        // is reused on the next sequence without reallocation.
    }
}

#[derive(Debug, Clone)]
enum KvCache {
    Normal(RetainingKvCache),
    Rotating(RetainingRotatingKvCache),
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    /// All-ones weight for the scale-free value RMSNorm (model dtype = BF16).
    v_norm_weight: Tensor,
    /// All-ones weight in F32 for the F32 decode fast path.
    v_norm_weight_f32: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    /// Attention logit soft-capping value (optional).
    attn_logit_softcapping: Option<f64>,
    rotary_emb: Arc<RotaryEmbedding>,
    /// Standard (unquantized) KV cache.
    kv_cache: KvCache,
    /// TurboQuant compressed KV cache (used instead of `kv_cache` when enabled).
    tq_cache: Option<TurboQuantKvCache>,
    /// Whether to use the fused `candle_nn::ops::sdpa` kernel.
    /// True when the device is Metal and head_dim is in {32, 64, 96, 128, 256}.
    /// When true, the decode path (q_seq=1) uses the optimised vector SDPA kernel,
    /// skipping the separate `repeat_kv` + matmul + softmax + matmul sequence.
    use_sdpa: bool,
    /// Pre-allocated output buffers for the partial-RoPE decode path.
    ///
    /// For global attention layers (head_dim=512, rotary_dim=128), applying RoPE
    /// requires splitting Q/K into a rotated part and a passthrough part, rotating
    /// the first part, then re-joining them with `Tensor::cat`.  Each `cat` allocates
    /// a new Metal buffer on every decode step.
    ///
    /// By pre-allocating fixed-size decode buffers ([1, num_heads, 1, head_dim] and
    /// [1, num_kv_heads, 1, head_dim]) and using two `slice_set` calls to fill them,
    /// we eliminate the per-step `Tensor::cat` allocation (10 allocations per decode
    /// step across all 7 global attention layers).
    ///
    /// `None` for layers that use full RoPE (sliding layers, head_dim == rotary_dim).
    /// Lazily allocated on the first decode step.
    partial_rope_q_out: Option<Tensor>,
    partial_rope_k_out: Option<Tensor>,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        is_sliding: bool,
        cfg: &Gemma4Config,
        head_dim: usize,
        tq_cfg: Option<&TurboQuantConfig>,
        vb: VarBuilder,
        qvb: Option<&QGgufVarBuilder>,
    ) -> Result<Self> {
        let hs = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = if is_sliding {
            cfg.num_key_value_heads
        } else {
            cfg.num_global_key_value_heads
        };
        let num_kv_groups = num_heads / num_kv_heads;
        let bias = cfg.attention_bias;

        // Global layers in 31B-style models tie V to K (no separate v_proj weight).
        let k_eq_v = !is_sliding && cfg.attention_k_eq_v;
        let q_proj = qlinear_b(
            hs,
            num_heads * head_dim,
            bias,
            vb.pp("q_proj"),
            qvb.map(|q| q.pp("q_proj")).as_ref(),
        )?;
        let k_proj = qlinear_b(
            hs,
            num_kv_heads * head_dim,
            bias,
            vb.pp("k_proj"),
            qvb.map(|q| q.pp("k_proj")).as_ref(),
        )?;
        let v_proj = if k_eq_v {
            k_proj.clone()
        } else {
            qlinear_b(
                hs,
                num_kv_heads * head_dim,
                bias,
                vb.pp("v_proj"),
                qvb.map(|q| q.pp("v_proj")).as_ref(),
            )?
        };
        let o_proj = qlinear_b(
            num_heads * head_dim,
            hs,
            bias,
            vb.pp("o_proj"),
            qvb.map(|q| q.pp("o_proj")).as_ref(),
        )?;
        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        // All-ones weight for the scale-free value RMSNorm — allocated once at
        // construction so the fused `candle_nn::ops::rms_norm` kernel can be
        // used at each forward pass without allocating a new tensor each time.
        let v_norm_weight = Tensor::ones(head_dim, cfg.dtype, &cfg.device)?;
        let v_norm_weight_f32 = Tensor::ones(head_dim, DType::F32, &cfg.device)?;

        let kv_cache = if is_sliding {
            KvCache::Rotating(RetainingRotatingKvCache::new(cfg.sliding_window))
        } else {
            // Use RetainingKvCache instead of candle's KvCache so that the
            // pre-allocated Metal buffer (up to 128 MiB per layer for global
            // attention) is reused across sequence resets rather than being
            // dropped and re-allocated on every new request.
            KvCache::Normal(RetainingKvCache::new(cfg.max_position_embeddings))
        };

        let tq_cache =
            tq_cfg.map(|c| TurboQuantKvCache::new(c, num_kv_heads, cfg.dtype, cfg.device.clone()));

        // Enable the fused SDPA kernel for Metal when the head dim is supported.
        // The Metal SDPA vector kernel (q_seq=1) supports head dims {32,64,96,128,256}
        // and handles GQA, eliminating the separate repeat_kv + matmul sequence.
        // Enable the fused SDPA vector kernel for all supported head dims on Metal.
        // head_dim=512 is now enabled via our patched candle-nn and candle-metal-kernels
        // that add sdpa_vector_{type}_512 instantiations and update the supported_head_dim
        // check to include 512.  This fuses attention for the 7 global attention layers
        // that previously required 4–5 separate Metal kernel dispatches.
        let use_sdpa = matches!(cfg.device, Device::Metal(_))
            && matches!(head_dim, 32 | 64 | 96 | 128 | 256 | 512);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm_weight,
            v_norm_weight_f32,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            attn_logit_softcapping: cfg.attn_logit_softcapping,
            rotary_emb,
            kv_cache,
            tq_cache,
            use_sdpa,
            // Lazily allocated on first decode step; None until then.
            partial_rope_q_out: None,
            partial_rope_k_out: None,
        })
    }

    /// Apply partial RoPE to Q and K for the decode path (q_len=1) using pre-allocated
    /// output buffers to avoid `Tensor::cat` allocations.
    ///
    /// For global attention layers where `rotary_dim < head_dim`, RoPE is applied to only
    /// the first `rotary_dim` features.  The standard implementation joins the rotated and
    /// passthrough parts with `Tensor::cat`, allocating a new Metal buffer each call.
    ///
    /// This method instead writes the two parts into a pre-allocated buffer via `slice_set`,
    /// eliminating the per-step allocation.  On the first call the buffers are lazily
    /// allocated.
    ///
    /// Falls back to `apply_rotary_emb_qkv` (which uses `Tensor::cat`) for the prefill path
    /// (`q_len > 1`) since the buffer is sized for a single decode step.
    fn apply_rope_qkv_buffered(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (b_sz, _n_heads, q_len, head_dim) = q.dims4()?;
        let rotary_dim = self.rotary_emb.rotary_dim;

        // Fast path: full RoPE (no partial), or prefill (multiple tokens).
        // For prefill, fall back to the standard cat-based implementation.
        if rotary_dim == head_dim || q_len != 1 {
            return self.rotary_emb.apply_rotary_emb_qkv(q, k, seqlen_offset);
        }

        // Decode path (q_len == 1) with partial RoPE: use pre-allocated output buffers.
        // Use F32 cos/sin when q is F32 (full-F32 decode fast-path).
        let (cos_src, sin_src) = if q.dtype() == DType::F32 {
            (&self.rotary_emb.cos_f32, &self.rotary_emb.sin_f32)
        } else {
            (&self.rotary_emb.cos, &self.rotary_emb.sin)
        };
        let cos = cos_src.narrow(0, seqlen_offset, 1)?;
        let sin = sin_src.narrow(0, seqlen_offset, 1)?;
        let pass_len = head_dim - rotary_dim;

        // Lazily allocate (or reallocate if batch size or dtype changed) the output buffers.
        let needs_alloc = self
            .partial_rope_q_out
            .as_ref()
            .is_none_or(|t| t.dim(0).unwrap_or(0) != b_sz || t.dtype() != q.dtype());
        if needs_alloc {
            self.partial_rope_q_out = Some(Tensor::zeros(
                (b_sz, self.num_heads, 1, head_dim),
                q.dtype(),
                q.device(),
            )?);
            self.partial_rope_k_out = Some(Tensor::zeros(
                (b_sz, self.num_kv_heads, 1, head_dim),
                k.dtype(),
                k.device(),
            )?);
        }
        let q_out = self.partial_rope_q_out.as_mut().unwrap();
        let k_out = self.partial_rope_k_out.as_mut().unwrap();

        // Apply RoPE to the first `rotary_dim` features, write passthrough unchanged.
        let q_rot = candle_nn::rotary_emb::rope(
            &q.narrow(D::Minus1, 0, rotary_dim)?.contiguous()?,
            &cos,
            &sin,
        )?;
        q_out.slice_set(&q_rot, D::Minus1, 0)?;
        q_out.slice_set(
            &q.narrow(D::Minus1, rotary_dim, pass_len)?.contiguous()?,
            D::Minus1,
            rotary_dim,
        )?;

        let k_rot = candle_nn::rotary_emb::rope(
            &k.narrow(D::Minus1, 0, rotary_dim)?.contiguous()?,
            &cos,
            &sin,
        )?;
        k_out.slice_set(&k_rot, D::Minus1, 0)?;
        k_out.slice_set(
            &k.narrow(D::Minus1, rotary_dim, pass_len)?.contiguous()?,
            D::Minus1,
            rotary_dim,
        )?;

        // Return the pre-allocated buffers directly.
        // `clone()` shares the underlying Metal buffer (ref-counted); no new
        // allocation occurs.  The caller consumes the clones for attention
        // computation within the same `forward_returning_kv` call.  By the time
        // the *next* decode step calls this method and overwrites the buffers via
        // `slice_set`, the previous clones have already been dropped — so there
        // is no aliasing hazard.
        Ok((q_out.clone(), k_out.clone()))
    }

    /// Apply partial RoPE to Q only for the decode path (q_len=1), using the pre-allocated
    /// Q output buffer.  Used by `forward_with_shared_kv` (KV-sharing layers).
    fn apply_rope_q_buffered(&mut self, q: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_sz, _n_heads, q_len, head_dim) = q.dims4()?;
        let rotary_dim = self.rotary_emb.rotary_dim;

        if rotary_dim == head_dim || q_len != 1 {
            return self.rotary_emb.apply_rotary_emb_q(q, seqlen_offset);
        }

        let (cos_src, sin_src) = if q.dtype() == DType::F32 {
            (&self.rotary_emb.cos_f32, &self.rotary_emb.sin_f32)
        } else {
            (&self.rotary_emb.cos, &self.rotary_emb.sin)
        };
        let cos = cos_src.narrow(0, seqlen_offset, 1)?;
        let sin = sin_src.narrow(0, seqlen_offset, 1)?;
        let pass_len = head_dim - rotary_dim;

        let needs_alloc = self
            .partial_rope_q_out
            .as_ref()
            .is_none_or(|t| t.dim(0).unwrap_or(0) != b_sz || t.dtype() != q.dtype());
        if needs_alloc {
            self.partial_rope_q_out = Some(Tensor::zeros(
                (b_sz, self.num_heads, 1, head_dim),
                q.dtype(),
                q.device(),
            )?);
        }
        let q_out = self.partial_rope_q_out.as_mut().unwrap();

        let q_rot = candle_nn::rotary_emb::rope(
            &q.narrow(D::Minus1, 0, rotary_dim)?.contiguous()?,
            &cos,
            &sin,
        )?;
        q_out.slice_set(&q_rot, D::Minus1, 0)?;
        q_out.slice_set(
            &q.narrow(D::Minus1, rotary_dim, pass_len)?.contiguous()?,
            D::Minus1,
            rotary_dim,
        )?;

        // Same aliasing rationale as `apply_rope_qkv_buffered`: the clone is
        // consumed within this decode step before the next step overwrites q_out.
        Ok(q_out.clone())
    }

    /// Standard forward pass.  Returns `(attn_output, post_cache_key, post_cache_value)`.
    ///
    /// The `post_cache_key` and `post_cache_value` are the accumulated K,V tensors
    /// returned from the KV cache (shape `[b, n_kv_heads, total_kv_len, head_dim]`).
    /// These can be forwarded to KV-sharing layers that reuse this layer's K,V.
    fn forward_returning_kv(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, q_len, _) = xs.dims3()?;

        // Project Q, K, V and apply per-head norms **before** the transpose so
        // that the norm inputs are contiguous (no GPU copy needed by contiguous()).
        //
        // Layout before transpose: [b, q_len, n_heads, head_dim] — row-major ✓
        // Layout after  transpose: [b, n_heads, q_len, head_dim] — NOT contiguous ✗
        //
        // The BF16-native GEMV kernel (quantize_q8_1_bf16) now handles BF16
        // activations directly, so there is no need to pre-convert to F32.
        // Each QLinear::forward call uses the fused BF16→Q8_1 path internally.
        let q_raw =
            self.q_proj
                .forward(xs)?
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
        let k_raw =
            self.k_proj
                .forward(xs)?
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?;
        let v_raw =
            self.v_proj
                .forward(xs)?
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?;
        // Apply norms on contiguous pre-transpose shape, then transpose.
        let query_states = self.q_norm.forward(&q_raw)?.transpose(1, 2)?;
        let key_states = self.k_norm.forward(&k_raw)?.transpose(1, 2)?;
        let v_norm_w = if v_raw.dtype() == DType::F32 {
            &self.v_norm_weight_f32
        } else {
            &self.v_norm_weight
        };
        let value_states =
            apply_rms_norm_4d_with_weight(&v_raw, v_norm_w, 1e-6_f32)?.transpose(1, 2)?;

        // query_states and key_states are now [b, n_heads, q_len, head_dim] (post-transpose).
        // RoPE — use the buffer-based path to avoid Tensor::cat allocations for
        // partial-RoPE global attention layers during decode.
        let (query_states, key_states) =
            self.apply_rope_qkv_buffered(&query_states, &key_states, seqlen_offset)?;

        // value_states is already normalized and transposed above.

        // KV cache — TurboQuant-compressed or plain.
        // Returns accumulated K,V (donor layer stores these for KV-sharing layers).
        let (key_states, value_states) = if let Some(tq) = &mut self.tq_cache {
            tq.append(&key_states, &value_states)
                .map_err(candle_core::Error::wrap)?;
            tq.dequantize().map_err(candle_core::Error::wrap)?
        } else {
            match &mut self.kv_cache {
                KvCache::Normal(c) => c.append(&key_states, &value_states)?,
                KvCache::Rotating(c) => c.append(&key_states, &value_states)?,
            }
        };

        // Attention computation.
        //
        // During decode (q_len=1) on Metal with a supported head_dim, use the
        // fused `sdpa` vector kernel which handles GQA internally (no repeat_kv
        // needed) and fuses the QK^T + optional softcap + softmax + @V into a
        // single kernel call.
        //
        // During prefill (q_len>1) or when SDPA is not available, fall back to
        // the manual path: expand KV for GQA, matmul, optional softcap, mask,
        // softmax, matmul.
        let attn_output = if self.use_sdpa && q_len == 1 && attention_mask.is_none() {
            // Fused decode path: pass unexpanded K/V directly to SDPA.
            // `sdpa(q, k, v, scale=1.0, softcapping)`:
            //   - scale=1.0  matches the reference (no per-element scaling)
            //   - softcapping=sc if softcapping is set, else 1.0 (no-op)
            let softcapping = self.attn_logit_softcapping.unwrap_or(1.0) as f32;
            candle_nn::ops::sdpa(
                &query_states,
                &key_states,
                &value_states,
                None,
                false,
                1.0_f32,
                softcapping,
            )?
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)?
        } else {
            // Manual GQA path: use Q-reshape to avoid materializing expanded K/V.
            //
            // For decode (q_len=1), `gqa_attention_no_expand` returns a result
            // already shaped [b, q_len, n_q*head_dim] — no transpose needed,
            // avoiding a non-contiguous tensor and the GPU contiguous() copy.
            // For prefill (q_len>1), it returns [b, n_q, q_len, head_dim] which
            // needs the standard transpose+reshape path.
            {
                let attn_out = gqa_attention_no_expand(
                    &query_states,
                    &key_states,
                    &value_states,
                    self.num_kv_groups,
                    self.attn_logit_softcapping,
                    attention_mask,
                )?;
                if q_len == 1 {
                    // Decode fast path: already [b, q_len, n_q*d] — apply o_proj directly.
                    attn_out.apply(&self.o_proj)?
                } else {
                    // Prefill path: [b, n_q, q_len, d] → transpose → reshape → o_proj.
                    attn_out
                        .transpose(1, 2)?
                        .reshape((b_sz, q_len, ()))?
                        .apply(&self.o_proj)?
                }
            }
        };

        Ok((attn_output, key_states, value_states))
    }

    /// Forward pass for KV-sharing layers: reuses K,V from the donor layer.
    ///
    /// `shared_key` and `shared_value` are the accumulated K,V tensors from the
    /// donor layer (already passed through the donor's KV cache).  This layer
    /// computes its own Q, then attends to the shared K,V.
    fn forward_with_shared_kv(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        shared_key: &Tensor,
        shared_value: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        // Compute Q only (K,V are shared from the donor layer).
        // Apply q_norm BEFORE transpose so the input is contiguous (avoids a GPU copy).
        let q_raw =
            self.q_proj
                .forward(xs)?
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
        let query_states = self.q_norm.forward(&q_raw)?.transpose(1, 2)?;

        // RoPE — use buffer-based path for partial-RoPE global layers during decode.
        let query_states = self.apply_rope_q_buffered(&query_states, seqlen_offset)?;

        // Use shared K,V directly (no cache update for this layer).
        // Use fused SDPA for decode (q_len=1) when available.
        if self.use_sdpa && q_len == 1 && attention_mask.is_none() {
            let softcapping = self.attn_logit_softcapping.unwrap_or(1.0) as f32;
            return candle_nn::ops::sdpa(
                &query_states,
                shared_key,
                shared_value,
                None,
                false,
                1.0_f32,
                softcapping,
            )?
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj);
        }

        // Use Q-reshape GQA path: avoids materializing expanded K/V copies.
        // For decode (q_len=1), the decode fast path returns [b, q_len, n_q*d].
        let attn_out = gqa_attention_no_expand(
            &query_states,
            shared_key,
            shared_value,
            self.num_kv_groups,
            self.attn_logit_softcapping,
            attention_mask,
        )?;
        if q_len == 1 {
            attn_out.apply(&self.o_proj)
        } else {
            attn_out
                .transpose(1, 2)?
                .reshape((b_sz, q_len, ()))?
                .apply(&self.o_proj)
        }
    }

    /// Paged forward for a **global (full-attention) donor layer**.
    ///
    /// Writes the new K/V tokens into the paged KV store at `layer_paged_idx`
    /// and gathers the full context from it, then runs attention.
    ///
    /// Returns `(attn_output, key_states_full, value_states_full)` where the
    /// last two tensors are the full accumulated K/V for reuse by KV-sharing layers.
    fn forward_returning_kv_paged(
        &mut self,
        xs: &Tensor,
        seqlen_offset: usize,
        block_table: &crate::kv_cache::BlockTable,
        kv_store: &mut crate::kv_cache::PagedKvStore,
        layer_paged_idx: usize,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, q_len, _) = xs.dims3()?;

        // Project Q/K/V and apply per-head norms (contiguous before transpose).
        let q_raw =
            self.q_proj
                .forward(xs)?
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
        let k_raw =
            self.k_proj
                .forward(xs)?
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?;
        let v_raw =
            self.v_proj
                .forward(xs)?
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?;

        let query_states = self.q_norm.forward(&q_raw)?.transpose(1, 2)?;
        let key_states = self.k_norm.forward(&k_raw)?.transpose(1, 2)?;
        let v_norm_w = if v_raw.dtype() == DType::F32 {
            &self.v_norm_weight_f32
        } else {
            &self.v_norm_weight
        };
        let value_states =
            apply_rms_norm_4d_with_weight(&v_raw, v_norm_w, 1e-6_f32)?.transpose(1, 2)?;

        // RoPE.
        let (query_states, key_states) =
            self.apply_rope_qkv_buffered(&query_states, &key_states, seqlen_offset)?;

        // Write new K/V into paged store and gather full context.
        let total_tokens = seqlen_offset + q_len;
        let all_slot_ids: Vec<u32> = (0..total_tokens)
            .map(|pos| {
                block_table.slot_for(pos).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "Gemma4 paged attn: no slot for position {pos}"
                    ))
                })
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let new_slot_ids = &all_slot_ids[seqlen_offset..];
        let new_slots_t = Tensor::new(new_slot_ids, key_states.device())?;

        // k/v: [b=1, num_kv_heads, q_len, head_dim] → [q_len, num_kv_heads, head_dim]
        let k_new = key_states.squeeze(0)?.transpose(0, 1)?.contiguous()?;
        let v_new = value_states.squeeze(0)?.transpose(0, 1)?.contiguous()?;

        kv_store.key_caches[layer_paged_idx] =
            kv_store.key_caches[layer_paged_idx].index_add(&new_slots_t, &k_new, 0)?;
        kv_store.value_caches[layer_paged_idx] =
            kv_store.value_caches[layer_paged_idx].index_add(&new_slots_t, &v_new, 0)?;

        let (k_gathered, v_gathered) = kv_store.gather_slots(layer_paged_idx, &all_slot_ids)?;

        // Reshape gathered: [total_tokens, num_kv_heads, head_dim] → [b, num_kv_heads, total, head_dim]
        // `.contiguous()` is required: the transpose produces a non-contiguous layout and
        // `gqa_attention_no_expand` will call `.transpose(2,3)` on k_full again inside matmul,
        // which requires contiguous input on CUDA.
        let k_full = k_gathered
            .reshape((b_sz, total_tokens, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?; // [b, n_kv, total, d] — contiguous for matmul
        let v_full = v_gathered
            .reshape((b_sz, total_tokens, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Attention (no sliding-window mask for global layers; apply causal mask for prefill).
        let attention_mask = if q_len > 1 {
            use crate::models::attention_utils::causal_mask;
            let m = causal_mask(q_len, total_tokens, seqlen_offset, xs.device(), xs.dtype())
                .map_err(candle_core::Error::wrap)?;
            Some(m)
        } else {
            None
        };

        let attn_output = gqa_attention_no_expand(
            &query_states,
            &k_full,
            &v_full,
            self.num_kv_groups,
            self.attn_logit_softcapping,
            attention_mask.as_ref(),
        )?;

        let attn_output = if q_len == 1 {
            attn_output.apply(&self.o_proj)?
        } else {
            attn_output
                .transpose(1, 2)?
                .reshape((b_sz, q_len, ()))?
                .apply(&self.o_proj)?
        };

        Ok((attn_output, k_full, v_full))
    }

    /// Paged forward for a **global KV-sharing layer**.
    ///
    /// Only computes Q (with RoPE), then attends to the shared K/V from the donor.
    fn forward_with_shared_kv_paged(
        &mut self,
        xs: &Tensor,
        seqlen_offset: usize,
        shared_key: &Tensor,
        shared_value: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q_raw =
            self.q_proj
                .forward(xs)?
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
        let query_states = self.q_norm.forward(&q_raw)?.transpose(1, 2)?;
        let query_states = self.apply_rope_q_buffered(&query_states, seqlen_offset)?;

        let attention_mask = if q_len > 1 {
            use crate::models::attention_utils::causal_mask;
            let kv_len = shared_key.dim(2)?;
            let m = causal_mask(q_len, kv_len, seqlen_offset, xs.device(), xs.dtype())
                .map_err(candle_core::Error::wrap)?;
            Some(m)
        } else {
            None
        };

        let attn_out = gqa_attention_no_expand(
            &query_states,
            shared_key,
            shared_value,
            self.num_kv_groups,
            self.attn_logit_softcapping,
            attention_mask.as_ref(),
        )?;

        if q_len == 1 {
            attn_out.apply(&self.o_proj)
        } else {
            attn_out
                .transpose(1, 2)?
                .reshape((b_sz, q_len, ()))?
                .apply(&self.o_proj)
        }
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kv_cache {
            KvCache::Normal(c) => c.reset(),
            KvCache::Rotating(c) => c.reset(),
        }
        if let Some(tq) = &mut self.tq_cache {
            tq.clear();
        }
    }
}

/// GQA attention without materializing expanded K/V tensors.
///
/// When `n_kv_groups > 1`, instead of expanding K/V from `[b, n_kv, kv_len, d]`
/// to `[b, n_q, kv_len, d]` (which forces a full contiguous copy), this function
/// reshapes Q to merge the group dimension with the query sequence:
///
///   Q:        [b, n_kv_groups * n_kv, q_len, d]
///   →reshape: [b, n_kv, n_kv_groups * q_len, d]
///   K^T:      [b, n_kv, d, kv_len]
///   QK^T:     [b, n_kv, n_kv_groups * q_len, kv_len]
///   →reshape: [b, n_q, q_len, kv_len]        ← apply mask/softcap here
///   softmax:  [b, n_q, q_len, kv_len]
///   →reshape: [b, n_kv, n_kv_groups * q_len, kv_len]
///   @V:       [b, n_kv, n_kv_groups * q_len, d]
///   →reshape: [b, n_q, q_len, d]
///
/// K and V are read exactly once from memory (no duplication), saving
/// `(n_kv_groups - 1) * kv_len * head_dim * 2` bytes per K and V tensor.
///
/// Arguments:
/// - `q`  : `[b, n_q_heads, q_len, head_dim]`
/// - `k`  : `[b, n_kv_heads, kv_len, head_dim]`
/// - `v`  : `[b, n_kv_heads, kv_len, head_dim]`
/// - `n_kv_groups`: `n_q_heads / n_kv_heads`
/// - `softcap`: optional attention logit soft-capping value
/// - `mask`: optional causal attention mask `[b, 1, q_len, kv_len]`
///
/// Returns `[b, n_q_heads, q_len, head_dim]` (ready for transpose + o_proj).
#[inline]
fn gqa_attention_no_expand(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_kv_groups: usize,
    softcap: Option<f64>,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, n_q_heads, q_len, head_dim) = q.dims4()?;
    let (_, n_kv_heads, kv_len, _) = k.dims4()?;

    if n_kv_groups == 1 {
        // No GQA: standard batched matmul.
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = match softcap {
            None => attn_weights,
            Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
        };
        let attn_weights = match mask {
            None => attn_weights,
            Some(m) => attn_weights.broadcast_add(m)?,
        };
        return candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(v);
    }

    // Reshape Q: [b, n_q, q_len, d] → [b, n_kv, n_kv_groups * q_len, d]
    // This groups each set of n_kv_groups query heads with the sequence dimension,
    // allowing a single matmul against the unexpanded K (no data duplication).
    let q_r = q.reshape((b, n_kv_heads, n_kv_groups * q_len, head_dim))?;

    // QK^T: [b, n_kv, n_kv_groups * q_len, kv_len]
    let attn_weights = q_r.matmul(&k.transpose(2, 3)?)?;

    // Apply optional softcapping (element-wise, shape-agnostic).
    let attn_weights = match softcap {
        None => attn_weights,
        Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
    };

    // Reshape to [b, n_q_heads, q_len, kv_len] to apply the per-head causal mask.
    // The reshape is valid because query head h maps to kv head h/n_kv_groups and
    // the corresponding attention rows are laid out contiguously.
    let attn_weights = attn_weights.reshape((b, n_q_heads, q_len, kv_len))?;

    let attn_weights = match mask {
        None => attn_weights,
        Some(m) => attn_weights.broadcast_add(m)?,
    };

    // Softmax over the kv_len dimension.
    let attn = candle_nn::ops::softmax_last_dim(&attn_weights)?;

    // Reshape back to [b, n_kv, n_kv_groups * q_len, kv_len] for the V matmul.
    let attn_r = attn.reshape((b, n_kv_heads, n_kv_groups * q_len, kv_len))?;

    // @V: [b, n_kv, n_kv_groups * q_len, head_dim]
    let out = attn_r.matmul(v)?;

    // For decode (q_len=1): the matmul output is [b, n_kv, n_kv_groups, d]
    // = [b, 1, 8, 256] which is contiguous.  We can skip the reshape to
    // [b, 8, 1, 256] and let the caller directly reshape to [b, 1, n_q*d].
    // This avoids the outer transpose(1,2) which would make the tensor
    // non-contiguous and force a contiguous() GPU copy before o_proj.
    if q_len == 1 {
        // Already [b, n_kv, n_kv_groups, head_dim] = [b, 1, n_q, d] — contiguous.
        // Return as-is; the caller (forward_returning_kv) will reshape directly
        // to [b, q_len, n_q_heads * head_dim] via a single contiguous reshape.
        return out.reshape((b, q_len, n_q_heads * head_dim));
    }

    // Reshape to [b, n_q_heads, q_len, head_dim]
    out.reshape((b, n_q_heads, q_len, head_dim))
}

// ---------------------------------------------------------------------------
// Decoder Layer
// ---------------------------------------------------------------------------

/// Per-layer input (PLI) fields — only present when `hidden_size_per_layer_input > 0`
/// (the efficient E2B/E4B variants). The 31B standard model omits these entirely.
#[derive(Debug, Clone)]
struct LayerPli {
    /// hidden_size -> hidden_size_per_layer_input
    gate: QLinear,
    /// hidden_size_per_layer_input -> hidden_size
    projection: QLinear,
    norm: RmsNorm,
    act_fn: Activation,
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    /// Scalar weight (shape [1]) multiplying the hidden state after MLP+PLI.
    /// Pre-cast to model dtype at construction time.
    layer_scalar: Tensor,
    /// F32 version of layer_scalar for the F32 decode fast-path.
    layer_scalar_f32: Tensor,
    /// PLI fields; `None` for models without per-layer input (e.g. 31B).
    pli: Option<LayerPli>,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb_sliding: Arc<RotaryEmbedding>,
        rotary_emb_global: Arc<RotaryEmbedding>,
        is_full_attention: bool,
        intermediate_size: usize,
        cfg: &Gemma4Config,
        tq_cfg: Option<&TurboQuantConfig>,
        vb: VarBuilder,
        qvb: Option<&QGgufVarBuilder>,
    ) -> Result<Self> {
        let head_dim = if is_full_attention {
            cfg.global_head_dim
        } else {
            cfg.head_dim
        };
        let rotary_emb = if is_full_attention {
            rotary_emb_global
        } else {
            rotary_emb_sliding
        };
        // Build a per-layer TurboQuantConfig with the correct head_dim for this layer type.
        let layer_tq_cfg: Option<TurboQuantConfig> = tq_cfg.map(|c| TurboQuantConfig {
            bits: c.bits,
            head_dim,
        });
        let self_attn = Attention::new(
            rotary_emb,
            !is_full_attention,
            cfg,
            head_dim,
            layer_tq_cfg.as_ref(),
            vb.pp("self_attn"),
            qvb.map(|q| q.pp("self_attn")).as_ref(),
        )?;
        let mlp = Mlp::new(
            cfg.hidden_size,
            intermediate_size,
            cfg.attention_bias,
            cfg.hidden_activation,
            vb.pp("mlp"),
            qvb.map(|q| q.pp("mlp")).as_ref(),
        )?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let pre_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        let pli = if cfg.hidden_size_per_layer_input > 0 {
            let gate = qlinear_b(
                cfg.hidden_size,
                cfg.hidden_size_per_layer_input,
                false,
                vb.pp("per_layer_input_gate"),
                qvb.map(|q| q.pp("per_layer_input_gate")).as_ref(),
            )?;
            let projection = qlinear_b(
                cfg.hidden_size_per_layer_input,
                cfg.hidden_size,
                false,
                vb.pp("per_layer_projection"),
                qvb.map(|q| q.pp("per_layer_projection")).as_ref(),
            )?;
            let norm = rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_per_layer_input_norm"),
            )?;
            Some(LayerPli {
                gate,
                projection,
                norm,
                act_fn: cfg.hidden_activation,
            })
        } else {
            None
        };

        // Pre-cast layer_scalar to the model dtype so the forward pass never
        // needs a runtime to_dtype() conversion (called once per token per layer).
        let layer_scalar_raw = vb.get(1, "layer_scalar")?;
        let layer_scalar_f32 = layer_scalar_raw.to_dtype(DType::F32)?;
        let layer_scalar = layer_scalar_raw.to_dtype(cfg.dtype)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_attention_layernorm,
            layer_scalar,
            layer_scalar_f32,
            pli,
        })
    }

    /// Forward pass for a **donor** layer (non-shared).
    ///
    /// Returns `(output, post_cache_key, post_cache_value)`.  The accumulated
    /// K,V tensors (after the KV cache) are returned so that KV-sharing layers
    /// that derive from this donor can reuse them.
    ///
    /// `per_layer_input`: [b, s, hidden_size_per_layer_input], or `None` for non-PLI models.
    fn forward_donor(
        &mut self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let residual = xs;
        let normed = self.input_layernorm.forward(xs)?;
        let (attn_out, k, v) =
            self.self_attn
                .forward_returning_kv(&normed, attention_mask, seqlen_offset)?;
        let attn_out = attn_out.apply(&self.post_attention_layernorm)?;
        let xs = (attn_out + residual)?;

        let xs = self.apply_mlp_and_pli(xs, per_layer_input)?;
        Ok((xs, k, v))
    }

    /// Forward pass for a **KV-sharing** layer.
    ///
    /// Instead of computing new K,V, this layer uses the K,V tensors from its
    /// donor layer (already accumulated through the donor's KV cache).
    ///
    /// `per_layer_input`: [b, s, hidden_size_per_layer_input], or `None` for non-PLI models.
    /// `shared_key`, `shared_value`: accumulated K,V from the donor layer.
    fn forward_shared(
        &mut self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        shared_key: &Tensor,
        shared_value: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs;
        let normed = self.input_layernorm.forward(xs)?;
        let attn_out = self.self_attn.forward_with_shared_kv(
            &normed,
            attention_mask,
            seqlen_offset,
            shared_key,
            shared_value,
        )?;
        let attn_out = attn_out.apply(&self.post_attention_layernorm)?;
        let xs = (attn_out + residual)?;

        self.apply_mlp_and_pli(xs, per_layer_input)
    }

    /// Paged forward for a **global donor** layer.
    ///
    /// Uses the paged KV store for K/V storage instead of the internal concat cache.
    fn forward_donor_paged(
        &mut self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        seqlen_offset: usize,
        block_table: &crate::kv_cache::BlockTable,
        kv_store: &mut crate::kv_cache::PagedKvStore,
        layer_paged_idx: usize,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let residual = xs;
        let normed = self.input_layernorm.forward(xs)?;
        let (attn_out, k, v) = self.self_attn.forward_returning_kv_paged(
            &normed,
            seqlen_offset,
            block_table,
            kv_store,
            layer_paged_idx,
        )?;
        let attn_out = attn_out.apply(&self.post_attention_layernorm)?;
        let xs = (attn_out + residual)?;
        let xs = self.apply_mlp_and_pli(xs, per_layer_input)?;
        Ok((xs, k, v))
    }

    /// Paged forward for a **global KV-sharing** layer.
    fn forward_shared_paged(
        &mut self,
        xs: &Tensor,
        per_layer_input: Option<&Tensor>,
        seqlen_offset: usize,
        shared_key: &Tensor,
        shared_value: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let residual = xs;
        let normed = self.input_layernorm.forward(xs)?;
        let attn_out = self.self_attn.forward_with_shared_kv_paged(
            &normed,
            seqlen_offset,
            shared_key,
            shared_value,
        )?;
        let attn_out = attn_out.apply(&self.post_attention_layernorm)?;
        let xs = (attn_out + residual)?;
        self.apply_mlp_and_pli(xs, per_layer_input)
    }

    /// Applies the MLP sub-layer, the optional PLI residual, and layer_scalar.
    fn apply_mlp_and_pli(
        &mut self,
        xs: Tensor,
        per_layer_input: Option<&Tensor>,
    ) -> Result<Tensor> {
        debug_assert_eq!(
            self.pli.is_some(),
            per_layer_input.is_some(),
            "PLI presence mismatch: layer has pli={}, but per_layer_input={}",
            self.pli.is_some(),
            per_layer_input.is_some()
        );
        // Standard Gemma MLP sub-layer
        let residual = &xs;
        let mlp_out = self
            .pre_feedforward_layernorm
            .forward(&xs)
            .and_then(|n| n.apply(&self.mlp))?;
        let mlp_out = mlp_out.apply(&self.post_feedforward_layernorm)?;
        let xs = (residual + mlp_out)?;

        // Per-layer input residual path — only for efficient variants (E2B/E4B).
        // Absent on standard models (e.g. 31B) where hidden_size_per_layer_input == 0.
        //
        // Reference (Gemma4DecoderLayer.forward):
        //   residual = hidden_states
        //   hidden_states = per_layer_input_gate(hidden_states)
        //   hidden_states = act_fn(hidden_states)
        //   hidden_states = hidden_states * per_layer_input   (element-wise)
        //   hidden_states = per_layer_projection(hidden_states)
        //   hidden_states = post_per_layer_input_norm(hidden_states)
        //   hidden_states = residual + hidden_states
        //   if layer_scalar is not None:
        //       hidden_states *= layer_scalar                 ← applied to WHOLE state
        let xs = if let (Some(pli), Some(pli_input)) = (&self.pli, per_layer_input) {
            let residual = &xs;
            // The BF16-native GEMV kernel handles BF16 input directly — no
            // explicit F32 conversion needed. Each QLinear::forward call uses
            // the fused BF16→Q8_1 path on CUDA.
            let gate = xs.apply(&pli.gate)?.apply(&pli.act_fn)?;
            let pli_out = gate.broadcast_mul(pli_input)?;
            let pli_out = pli_out.apply(&pli.projection)?;
            let pli_out = pli.norm.forward(&pli_out)?;
            (residual + pli_out)?
        } else {
            xs
        };

        // layer_scalar multiplies the entire hidden state (not just the contribution).
        let scalar = if xs.dtype() == DType::F32 {
            &self.layer_scalar_f32
        } else {
            &self.layer_scalar
        };
        xs.broadcast_mul(scalar)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ---------------------------------------------------------------------------
// Top-level Model
// ---------------------------------------------------------------------------

/// Top-level PLI tensors — only present when `hidden_size_per_layer_input > 0`.
struct ModelPli {
    /// PLI embedding table — memory-efficient per-row lookup.
    ///
    /// When `--quantize` is active: Q6K quantized (~1.9 GB for E2B) instead of
    /// the 4.7 GB BF16 dequantized form.  Individual rows are dequantized on
    /// demand and cached by token ID for zero-copy decode steps.
    ///
    /// When loading from safetensors: plain BF16 tensor (original behaviour).
    pli_embed_table: PliEmbeddingTable,
    /// embed_dim = num_hidden_layers * pli_dim.
    embed_dim: usize,
    /// GPU device to transfer lookup results to.
    gpu_device: Device,
    /// hidden_size -> num_layers * pli_dim
    per_layer_model_projection: QLinear,
    /// RMS norm applied per pli_dim slice.
    per_layer_projection_norm: candle_nn::RmsNorm,
    /// Fused scale for embed_tokens_per_layer: `sqrt(pli_dim) / sqrt(2)`.
    embed_combined_scale: f64,
    pli_dim: usize,
    /// Cache mapping token_id → scaled GPU pli_embed tensor `[1, 1, num_layers*pli_dim]`.
    ///
    /// During single-token decode the PLI embedding is a pure function of the
    /// token ID, so the GPU tensor (after CPU lookup + DMA + scale multiply) can
    /// be reused across requests and decode steps without recomputation.
    ///
    /// This eliminates:
    ///   - GPU→CPU synchronisation to transfer the token ID back (stalls pipeline)
    ///   - CPU `index_select` into the 4.7 GB table (random-access, cache-miss heavy)
    ///   - CPU→GPU DMA (17.5 KB per step for E2B)
    ///   - One GPU `mul` kernel (the embed_combined_scale multiply)
    ///
    /// The cache is bounded to at most `PLI_EMBED_CACHE_SIZE` entries via LRU
    /// eviction, implemented with a `VecDeque` of recently used token IDs.
    pli_embed_cache: std::collections::HashMap<u32, Tensor>,
    /// LRU order for `pli_embed_cache`; front = least recently used.
    pli_embed_cache_lru: std::collections::VecDeque<u32>,
    /// Cache mapping token_id → full `pli_all` tensor `[1, 1, num_layers, pli_dim]` on GPU.
    ///
    /// `pli_all = norm(per_layer_model_projection(embed(token_id))) + pli_embed(token_id) * scale`.
    ///
    /// Both `per_layer_model_projection` and `pli_embed_table` are pure functions of the
    /// token ID.  Caching `pli_all` by token ID eliminates, per decode step:
    ///   - 1 QLinear GEMV (per_layer_model_projection, 2×2048→8960 conversions + GEMV)
    ///   - 1 RmsNorm over [1,1,35,256]
    ///   - 1 pli_embed lookup (from pli_embed_cache) + dtype convert
    ///   - 1 pli_embed scale multiply
    ///   - 1 reshape + 1 add + 1 contiguous
    ///
    /// Total: ~6–8 fewer GPU kernel dispatches per decode step.
    pli_all_cache: std::collections::HashMap<u32, Vec<Tensor>>,
    /// LRU order for `pli_all_cache`.
    pli_all_cache_lru: std::collections::VecDeque<u32>,
}

pub struct Gemma4Model {
    embed_tokens: candle_nn::Embedding,
    /// PLI model-level tensors; `None` for standard models (e.g. 31B).
    pli: Option<ModelPli>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    /// Output projection (lm_head).
    ///
    /// For GGUF models where `embed_tokens.weight` is quantized, this is a
    /// `QLinear` backed by `QMatMul::QTensor` for decode and a pre-dequantized
    /// bf16 tensor for prefill, reducing the dominant lm_head GEMV cost 3×.
    /// For safetensors models this is an unquantized `QLinear::Tensor`.
    lm_head: QLinear,
    final_logit_softcapping: Option<f64>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    num_hidden_layers: usize,
    sliding_window: usize,
    /// KV-sharing map: `kv_sharing_map[i] = Some(donor_idx)` if layer `i` is a KV-sharing
    /// layer whose K,V come from layer `donor_idx`.  `None` means the layer computes its own K,V.
    kv_sharing_map: Vec<Option<usize>>,
    /// Reusable buffer for donor K,V tensors, avoiding a HashMap allocation on every
    /// forward call.  Indexed by layer index; `None` until populated in the layer loop.
    kv_donor_buf: Vec<Option<(Tensor, Tensor)>>,
    /// Pre-computed per-layer sliding flag: `true` if layer `i` uses sliding-window attention.
    /// Avoids an enum pattern-match on the KvCache variant on every forward call.
    is_sliding_per_layer: Vec<bool>,
    /// Cached attention masks to avoid rebuilding the same mask on every prefill call.
    /// Keyed by `(tgt_len, kv_len, seqlen_offset, is_sliding)`.
    mask_cache: std::collections::HashMap<(usize, usize, usize, bool), Tensor>,
    /// Token ID hint set by `hint_decode_token` before each decode step.
    ///
    /// When set, `forward_transformer` uses this token ID directly (no GPU→CPU
    /// sync) to look up the PLI embedding cache.  Cleared after each forward call.
    pending_decode_token_id: Option<u32>,
    /// When `true`, the next `forward` result will be sampled greedily (argmax).
    /// The model can skip monotonic final-logit transformations (e.g. softcapping)
    /// that do not affect argmax, saving up to 3 GPU kernel dispatches over the
    /// full vocab (262K elements) per decode step.
    skip_final_softcap: bool,
}

impl Gemma4Model {
    /// Build the model.
    ///
    /// `qvb` — when `Some`, projection weights are loaded as `QMatMul::QTensor`
    /// (quantized GGUF path, fast Metal GEMV); when `None`, loaded as plain
    /// bf16 tensors from `vb` (safetensors path or dequantized GGUF).
    pub fn new(
        cfg: &Gemma4Config,
        vb: VarBuilder,
        qvb: Option<&QGgufVarBuilder>,
        gguf_path: Option<&std::path::Path>,
    ) -> Result<Self> {
        let vb_lm = vb.pp("model").pp("language_model");
        let qvb_lm = qvb.map(|q| q.pp("model").pp("language_model"));

        // embed_tokens: vocab -> hs, scaled by sqrt(hs) inside the embedding
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_lm.pp("embed_tokens"))?;

        // embed_tokens_per_layer: vocab -> num_layers * pli_dim
        // PLI tensors — only for efficient variants (hidden_size_per_layer_input > 0).
        let model_pli = if cfg.hidden_size_per_layer_input > 0 {
            let pli_dim = cfg.hidden_size_per_layer_input;
            let embed_dim = cfg.num_hidden_layers * pli_dim;

            // Load embed_tokens_per_layer as a memory-efficient PliEmbeddingTable.
            //
            // GGUF path (--quantize): Use a GGUF-file-backed variant that reads
            // individual rows directly from the GGUF file without loading the full
            // ~1.9 GB Q6K tensor into CPU RAM.  This reduces peak RSS by ~1.9 GB,
            // bringing inferrs closer to llama-server's memory footprint.
            //
            // Safetensors path: Load dequantized BF16 on CPU as before.
            let pli_embed_table = if let Some(gguf) = gguf_path {
                // Try the zero-copy file-backed variant first.
                match PliEmbeddingTable::from_gguf_file(gguf)? {
                    Some(t) => {
                        tracing::info!(
                            "PLI embedding: using GGUF file-backed zero-copy lookup \
                             (no CPU RAM for the {:.1} GB table)",
                            (cfg.vocab_size * embed_dim * 2) as f64 / 1e9,
                        );
                        t
                    }
                    None => {
                        tracing::warn!(
                            "PLI embedding: file-backed lookup failed, falling back to QTensor"
                        );
                        if let Some(ref qvb) = qvb_lm {
                            if let Some(qt) = qvb.pp("embed_tokens_per_layer").get_qtensor() {
                                PliEmbeddingTable::from_qtensor(qt.clone())?
                            } else {
                                let cpu_vb =
                                    vb_lm.pp("embed_tokens_per_layer").set_device(Device::Cpu);
                                let t = cpu_vb.get((cfg.vocab_size, embed_dim), "weight")?;
                                PliEmbeddingTable::from_tensor(t)
                            }
                        } else {
                            let cpu_vb = vb_lm.pp("embed_tokens_per_layer").set_device(Device::Cpu);
                            let t = cpu_vb.get((cfg.vocab_size, embed_dim), "weight")?;
                            PliEmbeddingTable::from_tensor(t)
                        }
                    }
                }
            } else {
                // Safetensors path: load BF16 on CPU.
                let cpu_vb = vb_lm.pp("embed_tokens_per_layer").set_device(Device::Cpu);
                let t = cpu_vb.get((cfg.vocab_size, embed_dim), "weight")?;
                PliEmbeddingTable::from_tensor(t)
            };

            let per_layer_model_projection = qlinear_b(
                cfg.hidden_size,
                embed_dim,
                false,
                vb_lm.pp("per_layer_model_projection"),
                qvb_lm
                    .as_ref()
                    .map(|q| q.pp("per_layer_model_projection"))
                    .as_ref(),
            )?;
            // Pre-scale the norm weight by `1/sqrt(2)` so the forward pass can compute
            // `normed_pli_proj + pli_embed` without an extra scalar-multiply dispatch.
            let per_layer_input_scale = 1.0_f64 / 2.0_f64.sqrt();
            let per_layer_projection_norm = {
                let raw_norm = rms_norm(
                    pli_dim,
                    cfg.rms_norm_eps,
                    vb_lm.pp("per_layer_projection_norm"),
                )?;
                let layer_norm = raw_norm.into_inner();
                let scaled_weight = (layer_norm.weight() * per_layer_input_scale)?;
                candle_nn::RmsNorm::new(scaled_weight, cfg.rms_norm_eps)
            };
            Some(ModelPli {
                pli_embed_table,
                embed_dim,
                gpu_device: cfg.device.clone(),
                per_layer_model_projection,
                per_layer_projection_norm,
                embed_combined_scale: (pli_dim as f64).sqrt() / 2.0_f64.sqrt(),
                pli_dim,
                pli_embed_cache: std::collections::HashMap::new(),
                pli_embed_cache_lru: std::collections::VecDeque::new(),
                pli_all_cache: std::collections::HashMap::new(),
                pli_all_cache_lru: std::collections::VecDeque::new(),
            })
        } else {
            None
        };

        let dtype = vb.dtype();
        let dev = vb.device();

        // Rotary embedding tables (shared across same-type layers)
        let rotary_emb_sliding = Arc::new(RotaryEmbedding::new_standard(
            dtype,
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta_sliding,
            dev,
        )?);
        let rotary_emb_global = Arc::new(RotaryEmbedding::new_partial(
            dtype,
            cfg.global_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta_global,
            cfg.partial_rotary_factor_global,
            dev,
        )?);

        // Build TurboQuant config if requested.  head_dim is set per-layer inside
        // DecoderLayer::new since sliding and global layers use different head dims.
        let tq_cfg: Option<TurboQuantConfig> = cfg.turbo_quant_bits.map(|bits| {
            tracing::info!("TurboQuant KV cache enabled: {bits} bits/coord, absmax quantization");
            TurboQuantConfig {
                bits,
                head_dim: cfg.head_dim, // placeholder; overridden per layer in DecoderLayer::new
            }
        });

        let vb_layers = vb_lm.pp("layers");
        let qvb_layers = qvb_lm.as_ref().map(|q| q.pp("layers"));
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let is_full = cfg
                .layer_is_full_attention
                .get(layer_idx)
                .copied()
                .unwrap_or(false);
            let intermediate_size = if layer_idx >= cfg.double_wide_mlp_start_layer {
                cfg.intermediate_size * 2
            } else {
                cfg.intermediate_size
            };
            layers.push(DecoderLayer::new(
                rotary_emb_sliding.clone(),
                rotary_emb_global.clone(),
                is_full,
                intermediate_size,
                cfg,
                tq_cfg.as_ref(),
                vb_layers.pp(layer_idx),
                qvb_layers.as_ref().map(|q| q.pp(layer_idx)).as_ref(),
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_lm.pp("norm"))?;

        // lm_head: weight-tied to embed_tokens.
        //
        // lm_head: weight-tied to embed_tokens.
        //
        // GGUF path: use the quantized (Q6K) embed_tokens QTensor for the GEMV.
        // Prefill dequantizes on-the-fly via forward_via_f16 (no second copy).
        // Safetensors path: plain bf16 clone (no change).
        let lm_head = {
            let dense = embed_tokens.embeddings().clone();
            let built = qvb_lm
                .as_ref()
                .and_then(|q| q.pp("embed_tokens").try_qlinear_weight());
            match built {
                Some(Ok(ql)) => {
                    tracing::info!("lm_head: using quantized embed_tokens QTensor (Q6K)");
                    ql
                }
                Some(Err(e)) => {
                    tracing::warn!("lm_head: quantized build failed ({e}), using bf16");
                    QLinear::from_tensor(dense, None)
                }
                None => QLinear::from_tensor(dense, None),
            }
        };

        // Build the KV-sharing map.
        //
        // Layers with index >= first_kv_shared_layer_idx = num_layers - num_kv_shared_layers
        // reuse the K,V from the **last non-shared layer of the same attention type**.
        //
        // Example with num_layers=35, num_kv_shared_layers=20:
        //   first_kv_shared_layer_idx = 15
        //   Layer types (sliding=S, full=F): S S S S F S S S S F S S S S F | S S S S F ...
        //   Non-shared (0-14): last sliding = 13, last full = 14
        //   Shared (15-34): sliding layers → donor=13, full layers → donor=14
        let kv_sharing_map = {
            let first_kv_shared_idx = cfg.first_kv_shared_idx;

            // For each layer type, find the last non-shared layer index.
            let last_non_shared_sliding = (0..first_kv_shared_idx)
                .rev()
                .find(|&i| !cfg.layer_is_full_attention[i]);
            let last_non_shared_full = (0..first_kv_shared_idx)
                .rev()
                .find(|&i| cfg.layer_is_full_attention[i]);

            (0..cfg.num_hidden_layers)
                .map(|i| {
                    if i < first_kv_shared_idx {
                        None // non-shared layer; computes its own K,V
                    } else if cfg.layer_is_full_attention[i] {
                        last_non_shared_full // shared full-attention layer
                    } else {
                        last_non_shared_sliding // shared sliding layer
                    }
                })
                .collect::<Vec<_>>()
        };

        let num_hidden_layers = cfg.num_hidden_layers;

        // Pre-compute is_sliding per layer: true when the layer uses sliding-window (rotating)
        // attention. Derived from layer_is_full_attention to avoid per-forward enum matches.
        let is_sliding_per_layer: Vec<bool> = cfg
            .layer_is_full_attention
            .iter()
            .map(|&is_full| !is_full)
            .collect();

        Ok(Self {
            embed_tokens,
            pli: model_pli,
            layers,
            norm,
            lm_head,
            final_logit_softcapping: cfg.final_logit_softcapping,
            device: dev.clone(),
            dtype,
            hidden_size: cfg.hidden_size,
            num_hidden_layers,
            sliding_window: cfg.sliding_window,
            kv_sharing_map,
            kv_donor_buf: vec![None; num_hidden_layers],
            is_sliding_per_layer,
            mask_cache: std::collections::HashMap::new(),
            pending_decode_token_id: None,
            skip_final_softcap: false,
        })
    }

    fn prepare_decoder_attention_mask(
        &mut self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
        is_sliding: bool,
    ) -> Result<Tensor> {
        // For sliding-window layers the rotating KV cache holds at most
        // `sliding_window` tokens.  Clamp kv_len so the mask's last dimension
        // matches the actual KV tensor returned by `RetainingRotatingKvCache::append`.
        // Without this, a prefill prompt longer than `sliding_window` tokens would
        // produce attn_weights of shape [..., tgt_len, sliding_window] but a mask of
        // shape [..., tgt_len, tgt_len], causing a broadcast/dimension panic.
        let unclamped_kv_len = tgt_len + seqlen_offset;
        let kv_len = if is_sliding {
            unclamped_kv_len.min(self.sliding_window)
        } else {
            unclamped_kv_len
        };
        let key = (tgt_len, kv_len, seqlen_offset, is_sliding);

        if let Some(cached) = self.mask_cache.get(&key) {
            // Mask is already stored in model dtype — only expand batch dim.
            // Eliminates the `to_dtype` kernel call on cache hits, which is paid
            // on every repeated prefill of the same prompt length.
            return cached.expand((b_size, 1, tgt_len, kv_len));
        }

        let window = if is_sliding {
            self.sliding_window
        } else {
            usize::MAX
        };

        // Build the mask over the visible KV context.
        //
        // For global layers: kv_len == tgt_len + seqlen_offset, so we first
        // build a [tgt_len, tgt_len] causal square and then prepend seqlen_offset
        // zero-columns for the cached prefix (all visible, no masking needed).
        //
        // For sliding layers: kv_len is clamped to sliding_window.  The oldest
        // tokens have already been evicted from the rotating cache, so they are
        // simply absent — no prefix columns are needed.  We build a
        // [tgt_len, min(tgt_len, kv_len)] window where each query position i
        // can attend to KV positions within the sliding window.
        let mask = if is_sliding {
            // Number of KV slots from the current prefill chunk that are visible.
            let mask: Vec<f32> = (0..tgt_len)
                .flat_map(|i| {
                    // For each query row, the KV columns run over the last `kv_len`
                    // positions of the sequence.  Column j in the clamped mask
                    // corresponds to absolute position (unclamped_kv_len - kv_len + j).
                    let kv_start_abs = unclamped_kv_len - kv_len; // oldest visible KV position
                    (0..kv_len).map(move |j| {
                        let abs_j = kv_start_abs + j; // absolute position of this KV slot
                        let abs_i = seqlen_offset + i; // absolute position of this query
                        if abs_j > abs_i || (window != usize::MAX && abs_j + window < abs_i) {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            Tensor::from_slice(&mask, (tgt_len, kv_len), &self.device)?
        } else {
            // Global (full) attention: standard causal mask + cached-prefix columns.
            let mask: Vec<f32> = (0..tgt_len)
                .flat_map(|i| {
                    (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0.0 })
                })
                .collect();
            let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
            if seqlen_offset > 0 {
                let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
                Tensor::cat(&[&mask0, &mask], D::Minus1)?
            } else {
                mask
            }
        };

        // Convert to model dtype once, then cache as [1, 1, tgt_len, kv_len].
        // Storing in model dtype means cache hits only need a cheap `expand`,
        // eliminating the `to_dtype` GPU kernel on every repeated prefill call.
        let mask_model_dtype = mask.to_dtype(self.dtype)?;
        let mask_1 = mask_model_dtype.unsqueeze(0)?.unsqueeze(0)?;
        self.mask_cache.insert(key, mask_1.clone());
        mask_1.expand((b_size, 1, tgt_len, kv_len))
    }

    /// Forward pass with pre-computed audio embeddings.
    ///
    /// `audio_embeds`:   `[N, hidden_size]` — output of the audio encoder in LM space
    /// `audio_positions`: positions in `input_ids` that are `<|audio|>` soft tokens
    ///
    /// Audio embeddings are in the unscaled token embedding space; they are
    /// inserted before the `sqrt(hidden_size)` scale is applied, exactly as in
    /// the reference `Gemma4ForConditionalGeneration.forward`.
    pub fn forward_with_audio(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        audio_embeds: Tensor,
        audio_positions: Vec<usize>,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;

        // Replace audio soft token IDs with 0 (pad) to avoid OOB embedding lookup.
        let safe_ids = {
            let ids_data = input_ids.to_vec2::<u32>()?;
            let mut safe: Vec<u32> = ids_data.into_iter().flatten().collect();
            for &pos in &audio_positions {
                if pos < safe.len() {
                    safe[pos] = 0;
                }
            }
            Tensor::from_vec(safe, (b_size, seq_len), input_ids.device())?
        };

        // Embed tokens and scale by sqrt(hidden_size) — same as in forward().
        // This matches the reference Gemma4TextScaledWordEmbedding which bakes
        // the scale into the embedding lookup.
        let xs = self.embed_tokens.forward(&safe_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;

        // Inject audio embeddings into the already-scaled embedding tensor.
        // Audio features from embedding_projection are trained to live in the
        // scaled embedding space (the reference injects them after embed_tokens,
        // which includes the sqrt scale).
        //
        // Save text-only xs (PAD at audio positions) for PLI computation.
        // The reference uses llm_inputs_embeds (PAD at audio) for per_layer_model_projection,
        // not the audio-injected embeddings.  Cloning before injection preserves that.
        let xs_for_pli = xs.clone();
        let audio_embeds = audio_embeds.to_device(xs.device())?.to_dtype(xs.dtype())?;
        let h = self.hidden_size;
        tracing::info!(
            "forward_with_audio: injecting {} audio embeddings at {} positions",
            audio_embeds.dim(0).unwrap_or(0),
            audio_positions.len()
        );
        for (audio_idx, &pos) in audio_positions.iter().enumerate() {
            if audio_idx >= audio_embeds.dim(0)? {
                break;
            }
            let emb = audio_embeds.narrow(0, audio_idx, 1)?.unsqueeze(0)?; // [1, 1, H]
            xs = xs.slice_assign(&[0..b_size, pos..pos + 1, 0..h], &emb)?;
        }

        self.forward_transformer(
            b_size,
            seq_len,
            seqlen_offset,
            &safe_ids,
            Some(&xs_for_pli),
            xs,
        )
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;

        // Main token embeddings (scaled by sqrt(hidden_size) via embed_tokens convention)
        // Note: the Gemma embedding is raw; we scale here as in Gemma3.
        let xs = self.embed_tokens.forward(input_ids)?;
        let xs = (xs * (self.hidden_size as f64).sqrt())?;

        self.forward_transformer(b_size, seq_len, seqlen_offset, input_ids, None, xs)
    }

    /// Compute per-layer PLI inputs for all transformer layers.
    ///
    /// `b_size` / `seq_len`: batch and sequence dimensions.
    /// `ids_for_pli`: token IDs used for PLI embedding lookup (audio positions zeroed for audio path).
    /// `xs_for_pli`: text-only embeddings for `per_layer_model_projection`.  When `Some`, used
    ///               instead of `xs` so that audio embeddings do not corrupt the PLI projection
    ///               (matches reference: PLI uses `llm_inputs_embeds` with PAD at audio positions).
    /// `xs`: the current hidden states (used for projection when `xs_for_pli` is `None`).
    fn compute_pli_per_layer(
        &mut self,
        b_size: usize,
        seq_len: usize,
        ids_for_pli: &Tensor,
        xs_for_pli: Option<&Tensor>,
        xs: &Tensor,
    ) -> Result<Vec<Option<Tensor>>> {
        if let Some(model_pli) = &mut self.pli {
            // For the single-token decode path (seq_len == 1, no audio):
            // `pli_all = norm(per_layer_model_projection(embed(token))) + scaled_pli_embed(token)`
            // is a pure function of the token ID.  Cache it by token ID to avoid:
            //   1. GPU→CPU sync (token ID hint skips this)
            //   2. PLI embedding lookup from file / CPU RAM
            //   3. CPU→GPU DMA (17.5 KB)
            //   4. per_layer_model_projection GEMV (large 1536→8960 matrix)
            //   5. per_layer_projection_norm
            //   6. pli_embed scale multiply + add + contiguous
            if seq_len == 1 && xs_for_pli.is_none() && b_size == 1 {
                // Get token ID (prefer hint to avoid GPU→CPU sync)
                let token_id = if let Some(id) = self.pending_decode_token_id.take() {
                    id
                } else {
                    let ids_cpu = ids_for_pli.to_device(&Device::Cpu)?;
                    ids_cpu.flatten_all()?.to_vec1::<u32>()?[0]
                };

                if let Some(cached_layers) = model_pli.pli_all_cache.get(&token_id) {
                    // Cache hit: return clones of the pre-sliced per-layer tensors.
                    Ok(cached_layers
                        .iter()
                        .map(|t| Some(t.clone()))
                        .collect::<Vec<_>>())
                } else {
                    // Cache miss: compute pli_all and slice into per-layer tensors.
                    // 1. Get pli_embed (from pli_embed_cache or compute)
                    let pli_embed = if let Some(cached) = model_pli.pli_embed_cache.get(&token_id) {
                        cached.clone()
                    } else {
                        let embed_dim = model_pli.embed_dim;
                        let pli_embed_cpu = model_pli
                            .pli_embed_table
                            .lookup_single(token_id, embed_dim, self.dtype)?;
                        let pli_embed_gpu = pli_embed_cpu.to_device(&model_pli.gpu_device)?;
                        let scaled = (pli_embed_gpu * model_pli.embed_combined_scale)?;
                        while model_pli.pli_embed_cache.len() >= PLI_EMBED_CACHE_SIZE {
                            if let Some(evict_id) = model_pli.pli_embed_cache_lru.pop_front() {
                                model_pli.pli_embed_cache.remove(&evict_id);
                            } else {
                                break;
                            }
                        }
                        model_pli.pli_embed_cache.insert(token_id, scaled.clone());
                        model_pli.pli_embed_cache_lru.push_back(token_id);
                        scaled
                    };

                    // 2. pli_proj = norm(per_layer_model_projection(embed(token)))
                    let pli_proj = xs.apply(&model_pli.per_layer_model_projection)?.reshape((
                        1usize,
                        1usize,
                        self.num_hidden_layers,
                        model_pli.pli_dim,
                    ))?;
                    let pli_proj = model_pli.per_layer_projection_norm.forward(&pli_proj)?;
                    // pli_embed may be BF16 (cached); match xs dtype for the addition.
                    let pli_embed_cast = if pli_embed.dtype() != xs.dtype() {
                        pli_embed.to_dtype(xs.dtype())?
                    } else {
                        pli_embed.clone()
                    };
                    let pli_embed2 = pli_embed_cast.reshape((
                        1usize,
                        1usize,
                        self.num_hidden_layers,
                        model_pli.pli_dim,
                    ))?;
                    let pli_all = (pli_proj + pli_embed2)?.contiguous()?;

                    // 3. Slice into per-layer tensors and cache.
                    let per_layer: Vec<Tensor> = (0..self.num_hidden_layers)
                        .map(|i| pli_all.narrow(2, i, 1).and_then(|t| t.squeeze(2)))
                        .collect::<candle_core::Result<Vec<_>>>()?;

                    // Store in pli_all_cache (LRU eviction)
                    while model_pli.pli_all_cache.len() >= PLI_EMBED_CACHE_SIZE {
                        if let Some(evict_id) = model_pli.pli_all_cache_lru.pop_front() {
                            model_pli.pli_all_cache.remove(&evict_id);
                        } else {
                            break;
                        }
                    }
                    model_pli.pli_all_cache.insert(token_id, per_layer.clone());
                    model_pli.pli_all_cache_lru.push_back(token_id);

                    Ok(per_layer.into_iter().map(Some).collect::<Vec<_>>())
                }
            } else {
                // Prefill/audio path: compute from scratch for all tokens.
                let ids_cpu = ids_for_pli.to_device(&Device::Cpu)?;
                let ids_flat_vec = ids_cpu.flatten_all()?.to_vec1::<u32>()?;
                let embed_dim = model_pli.embed_dim;
                let pli_embed_cpu =
                    model_pli
                        .pli_embed_table
                        .lookup_batch(&ids_flat_vec, embed_dim, self.dtype)?;
                let pli_embed_cpu = pli_embed_cpu.reshape((b_size, seq_len, embed_dim))?;
                let pli_embed_gpu = pli_embed_cpu.to_device(&model_pli.gpu_device)?;
                let pli_embed = (pli_embed_gpu * model_pli.embed_combined_scale)?;

                let proj_input = xs_for_pli.unwrap_or(xs);
                let pli_proj = proj_input.apply(&model_pli.per_layer_model_projection)?;
                let pli_proj = pli_proj.reshape((
                    b_size,
                    seq_len,
                    self.num_hidden_layers,
                    model_pli.pli_dim,
                ))?;
                let pli_proj = model_pli.per_layer_projection_norm.forward(&pli_proj)?;
                let pli_embed = pli_embed.reshape((
                    b_size,
                    seq_len,
                    self.num_hidden_layers,
                    model_pli.pli_dim,
                ))?;
                let pli_all = (pli_proj + pli_embed)?.contiguous()?;
                Ok((0..self.num_hidden_layers)
                    .map(|i| pli_all.narrow(2, i, 1).and_then(|t| t.squeeze(2)).map(Some))
                    .collect::<candle_core::Result<_>>()?)
            }
        } else {
            Ok(vec![None; self.num_hidden_layers])
        }
    }

    /// Shared transformer body — runs PLI, attention layers, and lm_head.
    ///
    /// `ids_for_pli`: the token IDs used for PLI embedding lookup.  For the
    /// audio path this is the safe (audio positions zeroed) IDs tensor.
    fn forward_transformer(
        &mut self,
        b_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
        ids_for_pli: &Tensor,
        // Text-only embeddings for PLI projection (PAD at audio positions).
        // When Some, used for per_layer_model_projection instead of xs.
        // Matches reference behavior: PLI uses llm_inputs_embeds (PAD at audio).
        xs_for_pli: Option<&Tensor>,
        mut xs: Tensor,
    ) -> Result<Tensor> {
        // Per-layer inputs (PLI) — only for efficient variants.
        //
        // `pli_all` has shape [b, seq_len, num_hidden_layers, pli_dim].
        // We slice it into per-layer [b, seq_len, pli_dim] tensors using a single
        // `narrow + squeeze` per layer.  The contiguous `pli_all` tensor is computed
        // once; each `narrow(2, i, 1)` is a zero-copy metadata-only view on Metal,
        // and `squeeze(2)` removes the singleton dim.
        //
        // The previous code applied `pli_proj + pli_embed` then looped
        // `narrow(...).squeeze(...)` 35 times — identical semantically.  Making the
        // loop explicit here keeps the rest of the forward pass unchanged.
        let pli_per_layer =
            self.compute_pli_per_layer(b_size, seq_len, ids_for_pli, xs_for_pli, &xs)?;

        // Build attention masks (per attention type)
        let sliding_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset, true)?)
        };
        let global_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset, false)?)
        };

        // Clear only the donor-layer slots (those that are non-shared and may have been
        // written on the previous forward call).  Shared layers never write to kv_donor_buf,
        // so their slots are always None and skipping them avoids unnecessary work.
        for (slot, sharing) in self.kv_donor_buf.iter_mut().zip(self.kv_sharing_map.iter()) {
            if sharing.is_none() {
                *slot = None;
            }
        }

        for (layer_idx, pli) in pli_per_layer.iter().enumerate() {
            let mask = if self.is_sliding_per_layer[layer_idx] {
                sliding_mask.as_ref()
            } else {
                global_mask.as_ref()
            };

            match self.kv_sharing_map[layer_idx] {
                None => {
                    // Non-shared (donor candidate): run forward and capture K,V
                    let (new_xs, k, v) = self.layers[layer_idx].forward_donor(
                        &xs,
                        pli.as_ref(),
                        mask,
                        seqlen_offset,
                    )?;
                    xs = new_xs;
                    // Store K,V for any later layers that share from this layer
                    self.kv_donor_buf[layer_idx] = Some((k, v));
                }
                Some(donor_idx) => {
                    // KV-sharing layer: use K,V from the donor
                    let (shared_k, shared_v) = self.kv_donor_buf[donor_idx]
                        .as_ref()
                        .ok_or_else(|| {
                            candle_core::Error::msg(format!(
                                "KV sharing: donor layer {} has no stored K,V when processing layer {}",
                                donor_idx, layer_idx
                            ))
                        })?;
                    xs = self.layers[layer_idx].forward_shared(
                        &xs,
                        pli.as_ref(),
                        mask,
                        seqlen_offset,
                        shared_k,
                        shared_v,
                    )?;
                }
            }
        }

        let last_hidden = xs.narrow(1, seq_len - 1, 1)?.apply(&self.norm)?;

        // Apply final logit softcapping only when it can affect the result.
        //
        // softcapping(x) = tanh(x / sc) * sc is a monotonic function (tanh is
        // monotone, sc > 0), so argmax(softcapping(x)) == argmax(x).  When the
        // sampler will use argmax (temperature ≈ 0, no repetition penalty), the
        // three GPU kernel dispatches (div + tanh + mul) over the full vocab
        // (~262K elements for Gemma4-E2B-it) can be skipped safely.
        let greedy = self.skip_final_softcap;
        self.skip_final_softcap = false; // always reset

        // For greedy sampling on CUDA with a quantized (GGUF) lm_head, keep the
        // output as F32 (no F32→BF16 conversion needed since argmax works on any
        // numeric dtype).  This saves one GPU kernel per decode step.
        //
        // The F32 fast path is only valid when lm_head is a QTensor: the GGUF GEMV
        // kernel accepts F32 input and outputs F32.  For the dense safetensors path
        // (QMatMul::Tensor with BF16 weights), xs.matmul(&w) requires matching dtypes
        // so we must NOT pass F32 — doing so triggers "dtype mismatch in matmul".
        let logits = if greedy
            && self.lm_head.is_quantized()
            && matches!(last_hidden.device(), candle_core::Device::Cuda(_))
        {
            // Pass F32 input to lm_head to bypass the F32→BF16 output conversion.
            // The lm_head GEMV (quantize_q8_1_bf16 + mul_mat_vec) outputs F32;
            // with F32 input the existing QLinear code keeps it as F32 (no conversion).
            let h_f32 = last_hidden.to_dtype(DType::F32)?;
            h_f32.apply(&self.lm_head)? // Output stays F32
        } else {
            last_hidden.apply(&self.lm_head)?
        };

        let logits = if greedy {
            logits // Skip monotonic softcap
        } else {
            match self.final_logit_softcapping {
                None => logits,
                Some(sc) => ((logits / sc)?.tanh()? * sc)?,
            }
        };

        Ok(logits)
    }

    /// Paged-attention forward pass with pre-computed audio embeddings.
    ///
    /// Mirrors `forward_with_audio` but routes through the paged KV store for
    /// global (full-attention) layers.  Audio embeddings are injected into the
    /// hidden-state tensor before the transformer body runs, and the text-only
    /// embeddings are passed to `compute_pli_per_layer` so that PLI projection
    /// is not corrupted by the audio features (matches reference behaviour).
    pub fn forward_paged_with_audio(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        block_table: &crate::kv_cache::BlockTable,
        kv_store: &mut crate::kv_cache::PagedKvStore,
        audio_embeds: Tensor,
        audio_positions: Vec<usize>,
    ) -> candle_core::Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;

        // Zero audio soft-token IDs to avoid OOB embedding lookup.
        let safe_ids = {
            let ids_data = input_ids.to_vec2::<u32>()?;
            let mut safe: Vec<u32> = ids_data.into_iter().flatten().collect();
            for &pos in &audio_positions {
                if pos < safe.len() {
                    safe[pos] = 0;
                }
            }
            Tensor::from_vec(safe, (b_size, seq_len), input_ids.device())?
        };

        let xs = self.embed_tokens.forward(&safe_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;

        // Save text-only embed for PLI (same rationale as forward_with_audio).
        let xs_for_pli = xs.clone();
        let audio_embeds = audio_embeds.to_device(xs.device())?.to_dtype(xs.dtype())?;
        let h = self.hidden_size;
        tracing::info!(
            "forward_paged_with_audio: injecting {} audio embeddings at {} positions",
            audio_embeds.dim(0).unwrap_or(0),
            audio_positions.len()
        );
        for (audio_idx, &pos) in audio_positions.iter().enumerate() {
            if audio_idx >= audio_embeds.dim(0)? {
                break;
            }
            let emb = audio_embeds.narrow(0, audio_idx, 1)?.unsqueeze(0)?;
            xs = xs.slice_assign(&[0..b_size, pos..pos + 1, 0..h], &emb)?;
        }

        self.forward_paged_inner(
            b_size,
            seq_len,
            seqlen_offset,
            &safe_ids,
            Some(&xs_for_pli),
            xs,
            block_table,
            kv_store,
        )
    }

    /// Paged-attention forward pass.
    ///
    /// Global (full-attention) layers store their K/V in the paged KV store.
    /// Sliding-window layers continue using their existing rotating concat cache.
    ///
    /// This allows long-context serving without pre-allocating contiguous per-sequence
    /// buffers for the dominant full-attention KV tensors.
    pub fn forward_paged(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        block_table: &crate::kv_cache::BlockTable,
        kv_store: &mut crate::kv_cache::PagedKvStore,
    ) -> candle_core::Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;

        // Embed tokens — same as the non-paged path.
        let xs = self.embed_tokens.forward(input_ids)?;
        let xs = (xs * (self.hidden_size as f64).sqrt())?;

        self.forward_paged_inner(
            b_size,
            seq_len,
            seqlen_offset,
            input_ids,
            None,
            xs,
            block_table,
            kv_store,
        )
    }

    /// Shared paged-attention transformer body.
    ///
    /// Global layers use the paged KV store; sliding-window layers keep the
    /// existing rotating concat cache.
    #[allow(clippy::too_many_arguments)]
    fn forward_paged_inner(
        &mut self,
        b_size: usize,
        seq_len: usize,
        seqlen_offset: usize,
        ids_for_pli: &Tensor,
        xs_for_pli: Option<&Tensor>,
        mut xs: Tensor,
        block_table: &crate::kv_cache::BlockTable,
        kv_store: &mut crate::kv_cache::PagedKvStore,
    ) -> candle_core::Result<Tensor> {
        let pli_per_layer =
            self.compute_pli_per_layer(b_size, seq_len, ids_for_pli, xs_for_pli, &xs)?;

        // Sliding-window attention masks (used for sliding layers only).
        let sliding_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset, true)?)
        };

        // Clear donor K,V slots for this forward pass.
        for (slot, sharing) in self.kv_donor_buf.iter_mut().zip(self.kv_sharing_map.iter()) {
            if sharing.is_none() {
                *slot = None;
            }
        }

        // Track which full-attention donor layer maps to which paged-store slot.
        // The paged KV store has one entry per non-shared global layer (donor layers only).
        // KV-sharing layers reuse the donor's gathered K/V from kv_donor_buf.
        let mut paged_layer_idx: usize = 0;

        for (layer_idx, pli) in pli_per_layer.iter().enumerate() {
            let is_sliding = self.is_sliding_per_layer[layer_idx];

            match self.kv_sharing_map[layer_idx] {
                None => {
                    // Donor layer.
                    if is_sliding {
                        // Sliding donor: use existing rotating KV cache.
                        let mask = sliding_mask.as_ref();
                        let (new_xs, k, v) = self.layers[layer_idx].forward_donor(
                            &xs,
                            pli.as_ref(),
                            mask,
                            seqlen_offset,
                        )?;
                        xs = new_xs;
                        self.kv_donor_buf[layer_idx] = Some((k, v));
                    } else {
                        // Global donor: use paged KV store.
                        let (new_xs, k, v) = self.layers[layer_idx].forward_donor_paged(
                            &xs,
                            pli.as_ref(),
                            seqlen_offset,
                            block_table,
                            kv_store,
                            paged_layer_idx,
                        )?;
                        xs = new_xs;
                        self.kv_donor_buf[layer_idx] = Some((k, v));
                        paged_layer_idx += 1;
                    }
                }
                Some(donor_idx) => {
                    // KV-sharing layer: use donor's accumulated K/V.
                    let (shared_k, shared_v) =
                        self.kv_donor_buf[donor_idx].as_ref().ok_or_else(|| {
                            candle_core::Error::msg(format!(
                                "KV sharing (paged): donor layer {} has no K,V for layer {}",
                                donor_idx, layer_idx
                            ))
                        })?;
                    let (shared_k, shared_v) = (shared_k.clone(), shared_v.clone());

                    if is_sliding {
                        // Sliding KV-sharing: use non-paged shared-KV path.
                        let mask = sliding_mask.as_ref();
                        xs = self.layers[layer_idx].forward_shared(
                            &xs,
                            pli.as_ref(),
                            mask,
                            seqlen_offset,
                            &shared_k,
                            &shared_v,
                        )?;
                    } else {
                        // Global KV-sharing: paged path (only Q is computed here).
                        xs = self.layers[layer_idx].forward_shared_paged(
                            &xs,
                            pli.as_ref(),
                            seqlen_offset,
                            &shared_k,
                            &shared_v,
                        )?;
                    }
                }
            }
        }

        let last_hidden = xs.narrow(1, seq_len - 1, 1)?.apply(&self.norm)?;

        let greedy = self.skip_final_softcap;
        self.skip_final_softcap = false;

        let logits = if greedy
            && self.lm_head.is_quantized()
            && matches!(last_hidden.device(), candle_core::Device::Cuda(_))
        {
            let h_f32 = last_hidden.to_dtype(DType::F32)?;
            h_f32.apply(&self.lm_head)?
        } else {
            last_hidden.apply(&self.lm_head)?
        };

        let logits = if greedy {
            logits
        } else {
            match self.final_logit_softcapping {
                None => logits,
                Some(sc) => ((logits / sc)?.tanh()? * sc)?,
            }
        };

        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }

    /// Hint that the next `forward()` call is a single-token decode for `token_id`.
    ///
    /// This allows `forward_transformer` to look up the PLI embedding cache using
    /// the raw `u32` token ID directly, avoiding a GPU→CPU device transfer of
    /// the `input_ids` tensor.  The hint is consumed and cleared after one call.
    pub fn hint_decode_token(&mut self, token_id: u32) {
        self.pending_decode_token_id = Some(token_id);
    }

    /// Hint that the next `forward()` result will be sampled with `temperature`.
    ///
    /// When `temperature < ε` (greedy), the final logit softcapping is skipped
    /// because `tanh` is monotonic and argmax is invariant to it.
    pub fn hint_sampling_temperature(&mut self, temperature: f64) {
        const SAMPLING_EPS: f64 = 1e-5;
        self.skip_final_softcap = temperature < SAMPLING_EPS;
    }
}

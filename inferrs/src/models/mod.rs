//! Model implementations.
//!
//! We use candle-transformers' model implementations directly, wrapping them
//! with a unified trait for the engine to use.

pub mod attention_utils;
pub mod audio_encoder;
pub mod gemma4;
pub mod qwen3;
pub mod qwen3_5;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

use crate::config::{ModelArchitecture, RawConfig};
use crate::kv_cache::{BlockTable, PagedKvStore};
use gemma4::QGgufVarBuilder;

/// Unified model interface for the engine.
pub trait CausalLM: Send {
    /// Run a forward pass on the given input token IDs.
    /// Returns logits for the last token position: shape (batch_size, 1, vocab_size).
    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor>;

    /// Hint: the next `forward()` call will be a single-token decode step for
    /// this `token_id`.  Models that cache per-token state (e.g. PLI embeddings)
    /// can use this to pre-populate the cache without a GPU→CPU device transfer.
    ///
    /// The default implementation is a no-op.  Must be called before `forward`.
    fn hint_decode_token(&mut self, _token_id: u32) {}

    /// Hint: the next `forward()` call result will be sampled with `temperature`.
    ///
    /// When `temperature < ε` (greedy decoding), models can skip monotonic
    /// final-logit transformations (e.g. softcapping) that do not affect argmax.
    /// The default implementation is a no-op.
    fn hint_sampling_temperature(&mut self, _temperature: f64) {}

    /// Run a paged-attention forward pass.
    ///
    /// The default implementation falls back to `forward`, ignoring the paged
    /// store.  Models that support paged attention override this.
    ///
    /// The default clears the model's internal KV cache at the start of each
    /// new sequence (`seqlen_offset == 0`), matching the behaviour of the
    /// non-paged path in `cb_prefill`.  This prevents stale cache entries from
    /// a previous sequence from corrupting attention weight shapes.
    fn forward_paged(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        block_table: &BlockTable,
        kv_store: &mut PagedKvStore,
    ) -> Result<Tensor> {
        let _ = (block_table, kv_store); // unused in default impl
        if seqlen_offset == 0 {
            self.clear_kv_cache();
        }
        self.forward(input_ids, seqlen_offset)
    }

    /// Clear all KV caches (for starting a new sequence).
    fn clear_kv_cache(&mut self);

    // ── Audio ────────────────────────────────────────────────────────────────

    /// Returns `true` if this model has an audio encoder.
    #[allow(dead_code)]
    fn has_audio_tower(&self) -> bool {
        false
    }

    /// Encode a log-mel spectrogram (f32, shape `[1, T, 128]`) to LM-space
    /// embeddings of shape `[T/4, lm_hidden_size]`.
    ///
    /// Returns an error for models without an audio encoder.
    fn encode_audio(&mut self, _mel: &Tensor) -> Result<Tensor> {
        anyhow::bail!("this model does not have an audio encoder")
    }

    /// Store audio embeddings to be injected during the next `forward()` call.
    ///
    /// `embeds`:    `[N, lm_hidden_size]` — output of `encode_audio`
    /// `positions`: indices in the upcoming `input_ids` that hold audio soft tokens
    fn set_pending_audio(&mut self, _embeds: Tensor, _positions: Vec<usize>) {}
}

/// Implement `CausalLM` for a simple newtype wrapper whose `inner` field
/// exposes `.forward(input_ids, seqlen_offset)` and `.clear_kv_cache()`.
macro_rules! impl_causal_lm_wrapper {
    ($wrapper:ident, $inner_ty:ty) => {
        struct $wrapper {
            inner: $inner_ty,
        }

        impl CausalLM for $wrapper {
            fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
                self.inner
                    .forward(input_ids, seqlen_offset)
                    .map_err(Into::into)
            }

            fn clear_kv_cache(&mut self) {
                self.inner.clear_kv_cache();
            }
        }
    };
}

impl_causal_lm_wrapper!(
    Qwen2Model,
    candle_transformers::models::qwen2::ModelForCausalLM
);
impl_causal_lm_wrapper!(Gemma2Model, candle_transformers::models::gemma2::Model);
impl_causal_lm_wrapper!(Gemma3Model, candle_transformers::models::gemma3::Model);

/// A Qwen3 model wrapper.
struct Qwen3ModelWrapper {
    inner: qwen3::Qwen3Model,
}

impl CausalLM for Qwen3ModelWrapper {
    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.inner.forward(input_ids, seqlen_offset)
    }

    fn forward_paged(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        block_table: &BlockTable,
        kv_store: &mut PagedKvStore,
    ) -> Result<Tensor> {
        self.inner
            .forward_paged(input_ids, seqlen_offset, block_table, kv_store)
    }

    fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }
}

/// A Gemma4 model wrapper (with optional audio encoder).
struct Gemma4ModelWrapper {
    inner: gemma4::Gemma4Model,
    audio_encoder: Option<crate::models::audio_encoder::AudioEncoder>,
    /// Pending audio: embeddings + positions of audio soft tokens in input_ids.
    pending_audio: Option<(Tensor, Vec<usize>)>,
}

impl CausalLM for Gemma4ModelWrapper {
    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        if let Some((audio_embeds, positions)) = self.pending_audio.take() {
            Ok(self
                .inner
                .forward_with_audio(input_ids, seqlen_offset, audio_embeds, positions)?)
        } else {
            Ok(self.inner.forward(input_ids, seqlen_offset)?)
        }
    }

    fn forward_paged(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        block_table: &BlockTable,
        kv_store: &mut PagedKvStore,
    ) -> Result<Tensor> {
        if seqlen_offset == 0 {
            // Clear the sliding-window concat KV caches at the start of each sequence.
            self.inner.clear_kv_cache();
        }
        if let Some((audio_embeds, positions)) = self.pending_audio.take() {
            Ok(self.inner.forward_paged_with_audio(
                input_ids,
                seqlen_offset,
                block_table,
                kv_store,
                audio_embeds,
                positions,
            )?)
        } else {
            Ok(self
                .inner
                .forward_paged(input_ids, seqlen_offset, block_table, kv_store)?)
        }
    }

    fn hint_decode_token(&mut self, token_id: u32) {
        self.inner.hint_decode_token(token_id);
    }

    fn hint_sampling_temperature(&mut self, temperature: f64) {
        self.inner.hint_sampling_temperature(temperature);
    }

    fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }

    fn has_audio_tower(&self) -> bool {
        self.audio_encoder.is_some()
    }

    fn encode_audio(&mut self, mel: &Tensor) -> Result<Tensor> {
        let enc = self
            .audio_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gemma4 model was loaded without an audio tower"))?;
        enc.encode(mel)
    }

    fn set_pending_audio(&mut self, embeds: Tensor, positions: Vec<usize>) {
        self.pending_audio = Some((embeds, positions));
    }
}

/// A Qwen3.5 model wrapper.
struct Qwen35ModelWrapper {
    inner: qwen3_5::Qwen35Model,
}

impl CausalLM for Qwen35ModelWrapper {
    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.inner.forward(input_ids, seqlen_offset)
    }

    fn forward_paged(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        block_table: &BlockTable,
        kv_store: &mut PagedKvStore,
    ) -> Result<Tensor> {
        self.inner
            .forward_paged(input_ids, seqlen_offset, block_table, kv_store)
    }

    fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }
}

/// A lazy [`candle_nn::var_builder::SimpleBackend`] backed by a GGUF file.
///
/// Tensors are dequantized on demand — only when the model calls
/// `VarBuilder::get` for that specific weight.  This avoids the huge memory
/// spike and slow startup of the previous eager approach (loading all 2 000+
/// tensors including multi-gigabyte embedding tables upfront).
///
/// The GGUF file is kept open for the lifetime of the backend; a `Mutex`
/// around the `BufReader` satisfies the `Sync` requirement of `SimpleBackend`.
struct GgufBackend {
    content: candle_core::quantized::gguf_file::Content,
    reader: std::sync::Mutex<std::io::BufReader<std::fs::File>>,
    device: Device,
}

impl candle_nn::var_builder::SimpleBackend for GgufBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let mut reader = self.reader.lock().expect("gguf reader lock poisoned");
        // Use `dev` (the VarBuilder's device) for loading the quantized tensor
        // so that if the caller requests CPU placement (e.g. for the enormous
        // embed_tokens_per_layer table) the data never touches GPU memory.
        let load_dev = if matches!(dev, Device::Cpu) {
            dev
        } else {
            &self.device
        };
        let qt = self
            .content
            .tensor(&mut *reader, name, load_dev)
            .map_err(|e| {
                candle_core::Error::CannotFindTensor {
                    path: format!("{name}: {e}"),
                }
                .bt()
            })?;

        let tensor = qt.dequantize(dev)?.to_dtype(dtype)?;

        // Validate shape — same contract as VarBuilder::from_tensors.
        if tensor.shape() != &s {
            candle_core::bail!(
                "shape mismatch for {name}: expected {s:?}, got {:?}",
                tensor.shape()
            );
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let mut reader = self.reader.lock().expect("gguf reader lock poisoned");
        let load_dev = if matches!(dev, Device::Cpu) {
            dev
        } else {
            &self.device
        };
        let qt = self
            .content
            .tensor(&mut *reader, name, load_dev)
            .map_err(|e| {
                candle_core::Error::CannotFindTensor {
                    path: format!("{name}: {e}"),
                }
                .bt()
            })?;
        qt.dequantize(dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }
}

/// Build a [`VarBuilder`] backed by a GGUF file.
///
/// Tensors are dequantized lazily — only on first access — so startup is
/// fast and peak memory is bounded by the model's actual weight usage rather
/// than the full file size.
fn var_builder_from_gguf(
    gguf_path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    use candle_core::quantized::gguf_file;

    let file = std::fs::File::open(gguf_path)
        .with_context(|| format!("Cannot open GGUF {}", gguf_path.display()))?;
    let mut reader = std::io::BufReader::new(file);

    let content = gguf_file::Content::read(&mut reader)
        .with_context(|| format!("Failed to parse GGUF header in {}", gguf_path.display()))?;

    tracing::info!(
        "Opened GGUF with {} tensors: {}",
        content.tensor_infos.len(),
        gguf_path.display()
    );

    let backend = GgufBackend {
        content,
        reader: std::sync::Mutex::new(reader),
        device: device.clone(),
    };

    Ok(VarBuilder::from_backend(
        Box::new(backend),
        dtype,
        device.clone(),
    ))
}

/// Load a model from weight files.
pub fn load_model(
    raw_config: &RawConfig,
    arch: &ModelArchitecture,
    weight_paths: &[impl AsRef<Path>],
    gguf_path: Option<&Path>,
    dtype: DType,
    device: &Device,
    turbo_quant_bits: Option<u8>,
) -> Result<Box<dyn CausalLM>> {
    tracing::info!("Loading model weights ({:?} architecture)...", arch);

    // When a GGUF is present, load weights from it (dequantizing each tensor
    // to `dtype`).  Otherwise fall back to the standard mmap'd safetensors path.
    let vb: VarBuilder<'static> = if let Some(gguf) = gguf_path {
        var_builder_from_gguf(gguf, dtype, device)?
    } else {
        let paths_ref: Vec<&Path> = weight_paths.iter().map(|p| p.as_ref()).collect();
        // SAFETY: the mmap lifetime is extended to 'static by the unsafe block.
        // The VarBuilder (and the model built from it) keep the mmap alive.
        unsafe { VarBuilder::from_mmaped_safetensors(&paths_ref, dtype, device)? }
    };

    // For Gemma4 loaded from GGUF, also build a QGgufVarBuilder that keeps
    // weights in their quantized form (e.g. Q4K) so that projection layers
    // use QMatMul::QTensor → Metal's quantized GEMV kernel during decode.
    // This is the same strategy llama.cpp uses and gives ~3-4× decode speedup.
    let gemma4_qvb: Option<QGgufVarBuilder> = if matches!(arch, ModelArchitecture::Gemma4) {
        gguf_path.and_then(|p| {
            match QGgufVarBuilder::from_gguf(p, device) {
                Ok(qvb) => {
                    tracing::info!("Gemma4: using quantized weight projection (QMatMul) for GGUF model");
                    Some(qvb)
                }
                Err(e) => {
                    tracing::warn!("Gemma4: failed to build QGgufVarBuilder, falling back to dequantized weights: {e}");
                    None
                }
            }
        })
    } else {
        None
    };

    // TurboQuant is on by default; warn if this architecture doesn't support it.
    if turbo_quant_bits.is_some() {
        match arch {
            ModelArchitecture::Qwen3 | ModelArchitecture::Gemma4 => {} // supported
            other => {
                tracing::warn!(
                    "--turbo-quant is not supported for {:?} and will be ignored. \
                     TurboQuant KV cache compression is currently only available for Qwen3 and Gemma4. \
                     Pass --turbo-quant=false to suppress this warning.",
                    other
                );
            }
        }
    }

    let model: Box<dyn CausalLM> = match arch {
        ModelArchitecture::Qwen3 => {
            let config = raw_config.to_qwen3_config(dtype, device.clone(), turbo_quant_bits);
            tracing::info!(
                "Qwen3 config: {} layers, {} heads, {} hidden, {} kv_heads, head_dim={}",
                config.num_hidden_layers,
                config.num_attention_heads,
                config.hidden_size,
                config.num_key_value_heads,
                config.head_dim,
            );
            Box::new(Qwen3ModelWrapper {
                inner: qwen3::Qwen3Model::new(&config, vb)?,
            })
        }
        ModelArchitecture::Qwen2 => {
            let config = raw_config.to_qwen2_config();
            tracing::info!(
                "Qwen2 config: {} layers, {} heads, {} hidden, {} kv_heads",
                config.num_hidden_layers,
                config.num_attention_heads,
                config.hidden_size,
                config.num_key_value_heads
            );
            Box::new(Qwen2Model {
                inner: candle_transformers::models::qwen2::ModelForCausalLM::new(&config, vb)?,
            })
        }
        ModelArchitecture::Gemma2 => {
            let config = raw_config.to_gemma2_config();
            tracing::info!(
                "Gemma2 config: {} layers, {} heads, {} hidden, {} head_dim",
                config.num_hidden_layers,
                config.num_attention_heads,
                config.hidden_size,
                config.head_dim
            );
            Box::new(Gemma2Model {
                inner: candle_transformers::models::gemma2::Model::new(false, &config, vb)?,
            })
        }
        ModelArchitecture::Gemma3 => {
            let config = raw_config.to_gemma3_config();
            tracing::info!(
                "Gemma3 config: {} layers, {} heads, {} hidden, {} head_dim",
                config.num_hidden_layers,
                config.num_attention_heads,
                config.hidden_size,
                config.head_dim
            );
            Box::new(Gemma3Model {
                inner: candle_transformers::models::gemma3::Model::new(false, &config, vb)?,
            })
        }
        ModelArchitecture::Qwen35 => {
            let config = raw_config.to_qwen35_config(dtype, device.clone());
            tracing::info!(
                "Qwen3.5 config: {} layers, {} attn heads, {} hidden, {} kv_heads",
                config.num_hidden_layers,
                config.num_attention_heads,
                config.hidden_size,
                config.num_key_value_heads,
            );
            Box::new(Qwen35ModelWrapper {
                inner: qwen3_5::Qwen35Model::new(&config, vb)?,
            })
        }
        ModelArchitecture::Gemma4 => {
            let config = raw_config.to_gemma4_config(dtype, device.clone(), turbo_quant_bits);
            tracing::info!(
                "Gemma4 config: {} layers, {} heads, {} hidden, {} kv_heads",
                config.num_hidden_layers,
                config.num_attention_heads,
                config.hidden_size,
                config.num_key_value_heads,
            );
            let inner =
                gemma4::Gemma4Model::new(&config, vb.clone(), gemma4_qvb.as_ref(), gguf_path)?;

            // Load audio encoder if audio_config is present in the model config.
            let audio_encoder = if let Some(audio_cfg) = &raw_config.audio_config {
                tracing::info!(
                    "Gemma4 audio encoder: {} layers, hidden={}, output_dims={}",
                    audio_cfg.num_hidden_layers,
                    audio_cfg.hidden_size,
                    audio_cfg.output_proj_dims,
                );
                let enc = audio_encoder::AudioEncoder::load(
                    vb.pp("model"),
                    audio_cfg,
                    config.hidden_size,
                    device,
                    dtype,
                )
                .context("Failed to load Gemma4 audio encoder")?;
                tracing::info!("Audio encoder loaded successfully");
                Some(enc)
            } else {
                None
            };

            Box::new(Gemma4ModelWrapper {
                inner,
                audio_encoder,
                pending_audio: None,
            })
        }
    };
    tracing::info!("Model loaded successfully");
    Ok(model)
}

//! Model implementations.
//!
//! We use candle-transformers' model implementations directly, wrapping them
//! with a unified trait for the engine to use.

pub mod attention_utils;
pub mod audio_encoder;
pub mod gemma4;
mod gemma4_moe;
pub mod quantized_linear;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_5_linear_attn_scan;
pub mod vision_encoder;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

use crate::config::{ModelArchitecture, RawConfig, VisionConfig};
use inferrs_models::kv_cache::{BlockTable, PagedKvStore};
use quantized_linear::QGgufVarBuilder;

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

    // ── Vision ───────────────────────────────────────────────────────────────

    /// Returns `true` if this model has a vision encoder.
    #[allow(dead_code)]
    fn has_vision_tower(&self) -> bool {
        false
    }

    /// Encode pre-patchified pixel values to LM-space embeddings.
    ///
    /// `pixel_values`:  `[N_patches, patch_pixels]` f32 in [0, 1].
    /// `position_ids`:  `[N_patches, 2]`           i64 (x, y) coordinates.
    /// `n_soft_tokens`: requested output soft-token count.
    ///
    /// Returns `[n_soft_tokens, lm_hidden_size]`.
    fn encode_image(
        &mut self,
        _pixel_values: &Tensor,
        _position_ids: &Tensor,
        _n_soft_tokens: usize,
    ) -> Result<Tensor> {
        anyhow::bail!("this model does not have a vision encoder")
    }

    /// Store image embeddings to be injected during the next `forward()` call.
    ///
    /// `embeds`:    `[N_soft, lm_hidden_size]` — output of `encode_image`
    /// `positions`: indices in `input_ids` that hold image soft tokens
    fn set_pending_image(&mut self, _embeds: Tensor, _positions: Vec<usize>) {}

    /// Populate the paged KV store from the model's internal KV cache after a
    /// non-paged prefill.
    ///
    /// This is the key to "hybrid prefill": run `forward` (fast, contiguous, no
    /// scatter overhead) for the prompt, then copy the resulting K/V tensors
    /// from the internal cache into the paged store before decode begins.
    /// Decode steps then use `forward_paged` as usual.
    ///
    /// The default implementation is a no-op; models that support paged
    /// attention should override this.
    ///
    /// `block_table`: maps logical positions to physical paged slots for this sequence.
    /// `kv_store`: the physical paged KV store to populate.
    /// `prompt_len`: number of prompt tokens (positions 0..prompt_len to write).
    fn populate_paged_from_cache(
        &mut self,
        _block_table: &BlockTable,
        _kv_store: &mut PagedKvStore,
        _prompt_len: usize,
    ) -> Result<()> {
        Ok(()) // default: no-op (model does not support hybrid prefill)
    }
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
impl_causal_lm_wrapper!(Phi3Model, candle_transformers::models::phi3::Model);

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

    fn populate_paged_from_cache(
        &mut self,
        block_table: &BlockTable,
        kv_store: &mut PagedKvStore,
        prompt_len: usize,
    ) -> Result<()> {
        self.inner
            .populate_paged_from_cache(block_table, kv_store, prompt_len)
    }
}

/// A Gemma4 model wrapper (with optional audio and vision encoders).
struct Gemma4ModelWrapper {
    inner: gemma4::Gemma4Model,
    audio_encoder: Option<crate::models::audio_encoder::AudioEncoder>,
    /// Pending audio: embeddings + positions of audio soft tokens in input_ids.
    pending_audio: Option<(Tensor, Vec<usize>)>,
    vision_encoder: Option<crate::models::vision_encoder::VisionEncoder>,
    /// Pending image: embeddings + positions of image soft tokens in input_ids.
    pending_image: Option<(Tensor, Vec<usize>)>,
}

impl CausalLM for Gemma4ModelWrapper {
    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        if let Some((audio_embeds, positions)) = self.pending_audio.take() {
            Ok(self
                .inner
                .forward_with_audio(input_ids, seqlen_offset, audio_embeds, positions)?)
        } else if let Some((image_embeds, positions)) = self.pending_image.take() {
            Ok(self
                .inner
                .forward_with_image(input_ids, seqlen_offset, image_embeds, positions)?)
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
        } else if let Some((image_embeds, positions)) = self.pending_image.take() {
            Ok(self.inner.forward_paged_with_image(
                input_ids,
                seqlen_offset,
                block_table,
                kv_store,
                image_embeds,
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

    fn has_vision_tower(&self) -> bool {
        self.vision_encoder.is_some()
    }

    fn encode_image(
        &mut self,
        pixel_values: &Tensor,
        position_ids: &Tensor,
        n_soft_tokens: usize,
    ) -> Result<Tensor> {
        let enc = self
            .vision_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gemma4 model was loaded without a vision tower"))?;
        enc.encode(pixel_values, position_ids, Some(n_soft_tokens))
    }

    fn set_pending_image(&mut self, embeds: Tensor, positions: Vec<usize>) {
        self.pending_image = Some((embeds, positions));
    }

    fn populate_paged_from_cache(
        &mut self,
        block_table: &BlockTable,
        kv_store: &mut PagedKvStore,
        prompt_len: usize,
    ) -> Result<()> {
        self.inner
            .populate_paged_from_cache(block_table, kv_store, prompt_len)
            .map_err(Into::into)
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

/// A [`candle_nn::var_builder::SimpleBackend`] that wraps [`GgufBackend`] and
/// subtracts 1.0 from any tensor whose name ends in `"norm.weight"`.
///
/// llama.cpp/GGUF stores the actual RMSNorm scale value directly, but
/// candle-transformers (following HuggingFace convention) expects the stored
/// weight `w` to be applied as `1 + w`.  So llama.cpp stores `s`, HF stores
/// `s - 1.0`.  This wrapper corrects the mismatch for Gemma2 and Gemma3
/// external GGUFs.
struct GemmaNormFixBackend {
    inner: GgufBackend,
}

impl candle_nn::var_builder::SimpleBackend for GemmaNormFixBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.inner.get(s, name, h, dtype, dev)?;
        if name.ends_with("norm.weight") {
            tensor - 1.0f64
        } else {
            Ok(tensor)
        }
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let tensor = self.inner.get_unchecked(name, dtype, dev)?;
        if name.ends_with("norm.weight") {
            tensor - 1.0f64
        } else {
            Ok(tensor)
        }
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.contains_tensor(name)
    }
}

/// Build a [`VarBuilder`] backed by a GGUF file.
///
/// Tensors are dequantized lazily — only on first access — so startup is
/// fast and peak memory is bounded by the model's actual weight usage rather
/// than the full file size.
///
/// When `arch` is `Gemma2` or `Gemma3` and the file is an external GGUF
/// (detected by the presence of `token_embd.weight`), the backend is wrapped
/// in [`GemmaNormFixBackend`] to subtract 1.0 from all `*norm.weight` tensors,
/// correcting the RMSNorm scale convention difference between llama.cpp and HF.
fn var_builder_from_gguf(
    gguf_path: &Path,
    dtype: DType,
    device: &Device,
    arch: &ModelArchitecture,
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
    let top_keys = content.tensor_infos.keys().take(10).collect::<Vec<_>>();
    tracing::debug!("Top 10 GGUF keys: {:?}", top_keys);

    let is_external_gguf = content.tensor_infos.contains_key("token_embd.weight")
        && !content
            .tensor_infos
            .contains_key("model.embed_tokens.weight");

    let backend = GgufBackend {
        content,
        reader: std::sync::Mutex::new(reader),
        device: device.clone(),
    };

    let boxed: Box<dyn candle_nn::var_builder::SimpleBackend + 'static> = if is_external_gguf
        && matches!(arch, ModelArchitecture::Gemma2 | ModelArchitecture::Gemma3)
    {
        tracing::info!(
            "{arch:?}: applying GemmaNormFixBackend to correct RMSNorm scale for external GGUF"
        );
        Box::new(GemmaNormFixBackend { inner: backend })
    } else {
        Box::new(backend)
    };

    Ok(VarBuilder::from_backend(boxed, dtype, device.clone()))
}

/// Map a HuggingFace safetensors tensor name to its llama.cpp canonical GGUF
/// equivalent.  This is the inverse of llama.cpp's `tensor_mapping.py`.
///
/// Called from `rename_f` when an external GGUF file is detected (i.e. the file
/// contains `token_embd.weight` rather than `model.embed_tokens.weight`).
fn gguf_rename_tensor(name: &str, arch: &ModelArchitecture) -> String {
    // Architectures that nest under `model.language_model.*`:
    //   Qwen3.5, Gemma4
    // All others nest under `model.*`.
    let layer_prefix = match arch {
        ModelArchitecture::Qwen35 | ModelArchitecture::Gemma4 => "model.language_model.",
        _ => "model.",
    };

    // ── Top-level tensors ────────────────────────────────────────────────

    // embed_tokens → token_embd
    let embed_path = format!("{layer_prefix}embed_tokens.weight");
    if name == embed_path {
        return "token_embd.weight".into();
    }

    // final norm → output_norm
    let norm_path = format!("{layer_prefix}norm.weight");
    if name == norm_path {
        return "output_norm.weight".into();
    }

    // lm_head → output  (only models with untied heads have this tensor)
    if name == "lm_head.weight" {
        return "output.weight".into();
    }

    // ── Gemma4 PLI global tensors ────────────────────────────────────────
    if matches!(arch, ModelArchitecture::Gemma4) {
        let pli_prefix = layer_prefix.to_string();
        if name == format!("{pli_prefix}embed_tokens_per_layer.weight") {
            return "per_layer_token_embd.weight".into();
        }
        if name == format!("{pli_prefix}per_layer_model_projection.weight") {
            return "per_layer_model_proj.weight".into();
        }
        if name == format!("{pli_prefix}per_layer_projection_norm.weight") {
            return "per_layer_proj_norm.weight".into();
        }
    }

    // ── Per-layer tensors ────────────────────────────────────────────────
    let layers_prefix = format!("{layer_prefix}layers.");
    if let Some(rest) = name.strip_prefix(&layers_prefix) {
        // rest = "0.self_attn.q_proj.weight" → idx="0", suffix="self_attn.q_proj.weight"
        if let Some((idx, suffix)) = rest.split_once('.') {
            let mapped = gguf_rename_layer_suffix(suffix, arch);
            return format!("blk.{idx}.{mapped}");
        }
    }

    // Unmapped — return as-is (happens for inferrs-internal tensors).
    name.to_string()
}

/// Map a single layer suffix (e.g. `self_attn.q_proj.weight`) to its GGUF
/// equivalent (e.g. `attn_q.weight`).
fn gguf_rename_layer_suffix(suffix: &str, arch: &ModelArchitecture) -> String {
    // ── Attention projections ────────────────────────────────────────────
    let mapped = match suffix {
        // Separate Q/K/V (Qwen2/3/3.5, Gemma2/3/4)
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.q_proj.bias" => "attn_q.bias",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.k_proj.bias" => "attn_k.bias",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.v_proj.bias" => "attn_v.bias",
        // Fused QKV (Phi3)
        "self_attn.qkv_proj.weight" => "attn_qkv.weight",
        // Output projection
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.o_proj.bias" => "attn_output.bias",
        // QK-norm (Qwen3, Gemma3, Gemma4)
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",

        // ── MLP ──────────────────────────────────────────────────────────
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        // Phi3 fused gate_up_proj
        "mlp.gate_up_proj.weight" => "ffn_up.weight",

        // ── Norms ────────────────────────────────────────────────────────
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => {
            // Gemma2/3/4 use this as "post_attention_norm", everyone else as "ffn_norm"
            match arch {
                ModelArchitecture::Gemma2
                | ModelArchitecture::Gemma3
                | ModelArchitecture::Gemma4 => "post_attention_norm.weight",
                _ => "ffn_norm.weight",
            }
        }
        // Gemma-specific extra norms
        "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
        "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",

        // ── Gemma4 PLI per-layer tensors ─────────────────────────────────
        "per_layer_input_gate.weight" => "inp_gate.weight",
        "per_layer_projection.weight" => "proj.weight",
        "post_per_layer_input_norm.weight" => "post_norm.weight",
        // layer_scalar is loaded via vb.get(1, "layer_scalar") (no .weight suffix)
        "layer_scalar" => "layer_output_scale.weight",

        // ── Qwen3.5 linear attention tensors ─────────────────────────────
        "linear_attn.in_proj_qkv.weight" => "linear_attn_in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight" => "linear_attn_in_proj_z.weight",
        "linear_attn.in_proj_a.weight" => "linear_attn_in_proj_a.weight",
        "linear_attn.in_proj_b.weight" => "linear_attn_in_proj_b.weight",
        "linear_attn.out_proj.weight" => "linear_attn_out_proj.weight",

        // Fallback: pass through the suffix unchanged so that `blk.{idx}.`
        // prefix is still applied.  This handles any tensors not explicitly
        // listed above without causing a lookup failure.
        other => return other.to_string(),
    };
    mapped.to_string()
}

/// Reverse-map a llama.cpp canonical GGUF tensor name back to its HuggingFace
/// safetensors equivalent.  Used to re-key `QGgufVarBuilder` data so that
/// model code looking up HF-style paths can find the right tensors.
fn gguf_reverse_rename_tensor(gguf_name: &str, arch: &ModelArchitecture) -> String {
    let layer_prefix = match arch {
        ModelArchitecture::Qwen35 | ModelArchitecture::Gemma4 => "model.language_model.",
        _ => "model.",
    };

    // ── Top-level tensors ────────────────────────────────────────────────
    match gguf_name {
        "token_embd.weight" => return format!("{layer_prefix}embed_tokens.weight"),
        "output_norm.weight" => return format!("{layer_prefix}norm.weight"),
        "output.weight" => return "lm_head.weight".into(),
        _ => {}
    }

    // ── Gemma4 PLI global tensors ────────────────────────────────────────
    if matches!(arch, ModelArchitecture::Gemma4) {
        match gguf_name {
            "per_layer_token_embd.weight" => {
                return format!("{layer_prefix}embed_tokens_per_layer.weight")
            }
            "per_layer_model_proj.weight" => {
                return format!("{layer_prefix}per_layer_model_projection.weight")
            }
            "per_layer_proj_norm.weight" => {
                return format!("{layer_prefix}per_layer_projection_norm.weight")
            }
            _ => {}
        }
    }

    // ── Per-layer tensors: blk.{idx}.{gguf_suffix} ───────────────────────
    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        if let Some((idx, gguf_suffix)) = rest.split_once('.') {
            let hf_suffix = gguf_reverse_layer_suffix(gguf_suffix, arch);
            return format!("{layer_prefix}layers.{idx}.{hf_suffix}");
        }
    }

    gguf_name.to_string()
}

/// Reverse-map a GGUF block suffix to its HF equivalent.
fn gguf_reverse_layer_suffix(suffix: &str, arch: &ModelArchitecture) -> String {
    let mapped = match suffix {
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_q.bias" => "self_attn.q_proj.bias",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_k.bias" => "self_attn.k_proj.bias",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_v.bias" => "self_attn.v_proj.bias",
        "attn_qkv.weight" => "self_attn.qkv_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_output.bias" => "self_attn.o_proj.bias",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => match arch {
            ModelArchitecture::Phi3 => "mlp.gate_up_proj.weight",
            _ => "mlp.up_proj.weight",
        },
        "ffn_down.weight" => "mlp.down_proj.weight",
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => match arch {
            ModelArchitecture::Gemma2 | ModelArchitecture::Gemma3 | ModelArchitecture::Gemma4 => {
                "pre_feedforward_layernorm.weight"
            }
            _ => "post_attention_layernorm.weight",
        },
        "post_attention_norm.weight" => "post_attention_layernorm.weight",
        "post_ffw_norm.weight" => "post_feedforward_layernorm.weight",
        "inp_gate.weight" => "per_layer_input_gate.weight",
        "proj.weight" => "per_layer_projection.weight",
        "post_norm.weight" => "post_per_layer_input_norm.weight",
        "layer_output_scale.weight" => "layer_scalar",
        // Passthrough for unknown suffixes
        other => return other.to_string(),
    };
    mapped.to_string()
}

/// Load a model from weight files.
#[allow(clippy::too_many_arguments)]
pub fn load_model(
    raw_config: &RawConfig,
    arch: &ModelArchitecture,
    weight_paths: &[impl AsRef<Path>],
    gguf_path: Option<&Path>,
    dtype: DType,
    device: &Device,
    turbo_quant_bits: Option<u8>,
    config_path: &Path,
) -> Result<Box<dyn CausalLM>> {
    tracing::info!("Loading model weights ({:?} architecture)...", arch);

    // When a GGUF is present, load weights from it (dequantizing each tensor
    // to `dtype`).  Otherwise fall back to the standard mmap'd safetensors path.
    let vb: VarBuilder<'static> = if let Some(gguf) = gguf_path {
        let mut base_vb = var_builder_from_gguf(gguf, dtype, device, arch)?;

        // Map safetensors names (expected by candle-transformers) to the
        // llama.cpp canonical GGUF names if the user downloaded an external
        // GGUF file.  Detection: external GGUFs always have `token_embd.weight`
        // while inferrs-quantized GGUFs retain the original HF names.
        if base_vb.contains_tensor("token_embd.weight")
            && !base_vb.contains_tensor("model.embed_tokens.weight")
        {
            tracing::info!(
                "Detected external GGUF format: applying llama.cpp tensor name mappings for {:?}",
                arch
            );
            let arch_clone = arch.clone();
            base_vb = base_vb.rename_f(move |name: &str| gguf_rename_tensor(name, &arch_clone));
        }

        base_vb
    } else {
        let paths_ref: Vec<&Path> = weight_paths.iter().map(|p| p.as_ref()).collect();
        // SAFETY: the mmap lifetime is extended to 'static by the unsafe block.
        // The VarBuilder (and the model built from it) keep the mmap alive.
        unsafe { VarBuilder::from_mmaped_safetensors(&paths_ref, dtype, device)? }
    };

    // Detect external GGUF format for QGgufVarBuilder re-keying below.
    let is_external_gguf = gguf_path.is_some() && vb.contains_tensor("token_embd.weight");

    // For Gemma4 loaded from GGUF, also build a QGgufVarBuilder that keeps
    // weights in their quantized form (e.g. Q4K) so that projection layers
    // use QMatMul::QTensor → quantized GEMV kernel during decode.
    // This is the same strategy llama.cpp uses and gives ~3-4× decode speedup.
    // NOTE: Only enabled for architectures that have been updated to use
    // qlinear_b. Enabling it before the model uses QLinear would load weights
    // twice (quantized + dequantized), doubling memory usage.
    let qvb: Option<QGgufVarBuilder> = if matches!(
        arch,
        ModelArchitecture::Gemma4 | ModelArchitecture::Qwen35
    ) {
        gguf_path.and_then(|p| {
            match QGgufVarBuilder::from_gguf(p, device) {
                Ok(qvb) => {
                    tracing::info!(
                        "{arch:?}: using quantized weight projection (QMatMul) for GGUF model"
                    );
                    // For external GGUFs, re-key from llama.cpp names to HF names
                    // so that model code can find tensors via HF-style paths.
                    let qvb = if is_external_gguf {
                        match qvb
                            .rename_keys(|gguf_key| gguf_reverse_rename_tensor(gguf_key, arch))
                        {
                            Ok(q) => q,
                            Err(e) => {
                                tracing::warn!(
                                    "{arch:?}: failed to re-key QGgufVarBuilder, falling back to dequantized weights: {e}"
                                );
                                return None;
                            }
                        }
                    } else {
                        qvb
                    };
                    Some(qvb)
                }
                Err(e) => {
                    tracing::warn!(
                        "{arch:?}: failed to build QGgufVarBuilder, falling back to dequantized weights: {e}"
                    );
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
            ModelArchitecture::Qwen3 | ModelArchitecture::Qwen35 | ModelArchitecture::Gemma4 => {}
            other => {
                tracing::warn!(
                    "--turbo-quant is not supported for {:?} and will be ignored. \
                     TurboQuant KV cache compression is currently only available for Qwen3, Qwen3.5, and Gemma4. \
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
            let config = raw_config.to_qwen35_config(dtype, device.clone(), turbo_quant_bits);
            tracing::info!(
                "Qwen3.5 config: {} layers, {} attn heads, {} hidden, {} kv_heads",
                config.num_hidden_layers,
                config.num_attention_heads,
                config.hidden_size,
                config.num_key_value_heads,
            );
            Box::new(Qwen35ModelWrapper {
                inner: qwen3_5::Qwen35Model::new(&config, vb, qvb.as_ref())?,
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
            let inner = gemma4::Gemma4Model::new(&config, vb.clone(), qvb.as_ref(), gguf_path)?;

            // Load audio encoder if audio_config is present in the model config.
            // GGUF files typically only contain language model weights, so skip
            // audio/vision encoder loading if the tensors are absent.
            let audio_encoder = if let Some(audio_cfg) = &raw_config.audio_config {
                tracing::info!(
                    "Gemma4 audio encoder: {} layers, hidden={}, output_dims={}",
                    audio_cfg.num_hidden_layers,
                    audio_cfg.hidden_size,
                    audio_cfg.output_proj_dims,
                );
                match audio_encoder::AudioEncoder::load(
                    vb.pp("model"),
                    audio_cfg,
                    config.hidden_size,
                    device,
                    dtype,
                ) {
                    Ok(enc) => {
                        tracing::info!("Audio encoder loaded successfully");
                        Some(enc)
                    }
                    Err(e)
                        if gguf_path.is_some()
                            && format!("{e:#}").contains("cannot find tensor") =>
                    {
                        tracing::warn!("Audio encoder weights not found in GGUF, skipping: {e:#}");
                        None
                    }
                    Err(e) => return Err(e).context("Failed to load Gemma4 audio encoder"),
                }
            } else {
                None
            };

            // Load vision encoder if vision_config is present in the model config.
            let vision_encoder = if let Some(vision_cfg) = &raw_config.vision_config {
                match vision_cfg {
                    VisionConfig::Gemma4(cfg) => {
                        tracing::info!(
                            "Gemma4 vision encoder: {} layers, hidden={}, patch_size={}, output_length={}",
                            cfg.num_hidden_layers,
                            cfg.hidden_size,
                            cfg.patch_size,
                            cfg.default_output_length,
                        );
                        match vision_encoder::VisionEncoder::load(
                            vb.pp("model"),
                            cfg,
                            config.hidden_size,
                            device,
                            dtype,
                        ) {
                            Ok(enc) => {
                                tracing::info!("Vision encoder loaded successfully");
                                Some(enc)
                            }
                            Err(e)
                                if gguf_path.is_some()
                                    && format!("{e:#}").contains("cannot find tensor") =>
                            {
                                tracing::warn!(
                                    "Vision encoder weights not found in GGUF, skipping: {e:#}"
                                );
                                None
                            }
                            Err(e) => {
                                return Err(e).context("Failed to load Gemma4 vision encoder")
                            }
                        }
                    }
                    VisionConfig::Qwen(_) => {
                        tracing::info!(
                            "Qwen vision encoder detected but not yet supported, skipping"
                        );
                        None
                    }
                }
            } else {
                None
            };

            Box::new(Gemma4ModelWrapper {
                inner,
                audio_encoder,
                pending_audio: None,
                vision_encoder,
                pending_image: None,
            })
        }
        ModelArchitecture::Phi3 => {
            let content = std::fs::read_to_string(config_path)
                .context("Failed to read config.json for Phi3")?;
            let config: candle_transformers::models::phi3::Config =
                serde_json::from_str(&content).context("Failed to parse Phi3 config")?;
            tracing::info!(
                "Phi3 config: {} layers, {} heads, {} hidden, {} kv_heads, head_dim={}",
                config.num_hidden_layers,
                config.num_attention_heads,
                config.hidden_size,
                config.num_key_value_heads,
                config.head_dim(),
            );
            Box::new(Phi3Model {
                inner: candle_transformers::models::phi3::Model::new(&config, vb)?,
            })
        }
    };
    tracing::info!("Model loaded successfully");
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelArchitecture;

    /// Helper: assert that `gguf_rename_tensor(input, arch) == expected` for every pair.
    fn check(arch: &ModelArchitecture, pairs: &[(&str, &str)]) {
        for (input, expected) in pairs {
            let got = gguf_rename_tensor(input, arch);
            assert_eq!(
                got, *expected,
                "gguf_rename_tensor({:?}, {:?}): got {:?}, want {:?}",
                input, arch, got, expected
            );
        }
    }

    #[test]
    fn gguf_gemma2_rename_table() {
        check(
            &ModelArchitecture::Gemma2,
            &[
                // ── globals (tied lm_head — no output.weight) ──────────────
                ("model.embed_tokens.weight", "token_embd.weight"),
                ("model.norm.weight", "output_norm.weight"),
                // ── attention ──────────────────────────────────────────────
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    "blk.0.attn_q.weight",
                ),
                (
                    "model.layers.1.self_attn.k_proj.weight",
                    "blk.1.attn_k.weight",
                ),
                (
                    "model.layers.1.self_attn.v_proj.weight",
                    "blk.1.attn_v.weight",
                ),
                (
                    "model.layers.1.self_attn.o_proj.weight",
                    "blk.1.attn_output.weight",
                ),
                // ── 4-norm per layer ───────────────────────────────────────
                (
                    "model.layers.2.input_layernorm.weight",
                    "blk.2.attn_norm.weight",
                ),
                (
                    "model.layers.2.post_attention_layernorm.weight",
                    "blk.2.post_attention_norm.weight",
                ),
                (
                    "model.layers.2.pre_feedforward_layernorm.weight",
                    "blk.2.ffn_norm.weight",
                ),
                (
                    "model.layers.2.post_feedforward_layernorm.weight",
                    "blk.2.post_ffw_norm.weight",
                ),
                // ── MLP ────────────────────────────────────────────────────
                (
                    "model.layers.3.mlp.gate_proj.weight",
                    "blk.3.ffn_gate.weight",
                ),
                ("model.layers.3.mlp.up_proj.weight", "blk.3.ffn_up.weight"),
                (
                    "model.layers.3.mlp.down_proj.weight",
                    "blk.3.ffn_down.weight",
                ),
                // ── passthrough ────────────────────────────────────────────
                (
                    "model.some_future_tensor.weight",
                    "model.some_future_tensor.weight",
                ),
            ],
        );
    }

    #[test]
    fn gguf_gemma3_rename_table() {
        check(
            &ModelArchitecture::Gemma3,
            &[
                // ── globals ────────────────────────────────────────────────
                ("model.embed_tokens.weight", "token_embd.weight"),
                ("model.norm.weight", "output_norm.weight"),
                // ── attention ──────────────────────────────────────────────
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    "blk.0.attn_q.weight",
                ),
                (
                    "model.layers.4.self_attn.k_proj.weight",
                    "blk.4.attn_k.weight",
                ),
                (
                    "model.layers.4.self_attn.v_proj.weight",
                    "blk.4.attn_v.weight",
                ),
                (
                    "model.layers.4.self_attn.o_proj.weight",
                    "blk.4.attn_output.weight",
                ),
                (
                    "model.layers.6.self_attn.q_norm.weight",
                    "blk.6.attn_q_norm.weight",
                ),
                (
                    "model.layers.6.self_attn.k_norm.weight",
                    "blk.6.attn_k_norm.weight",
                ),
                // ── norms (all 4 per layer) ────────────────────────────────
                (
                    "model.layers.2.input_layernorm.weight",
                    "blk.2.attn_norm.weight",
                ),
                (
                    "model.layers.2.pre_feedforward_layernorm.weight",
                    "blk.2.ffn_norm.weight",
                ),
                (
                    "model.layers.2.post_feedforward_layernorm.weight",
                    "blk.2.post_ffw_norm.weight",
                ),
                (
                    "model.layers.2.post_attention_layernorm.weight",
                    "blk.2.post_attention_norm.weight",
                ),
                // ── MLP ────────────────────────────────────────────────────
                (
                    "model.layers.9.mlp.gate_proj.weight",
                    "blk.9.ffn_gate.weight",
                ),
                ("model.layers.9.mlp.up_proj.weight", "blk.9.ffn_up.weight"),
                (
                    "model.layers.9.mlp.down_proj.weight",
                    "blk.9.ffn_down.weight",
                ),
                // ── lm_head (Gemma3 has tied embeddings; but the rename function
                // maps any lm_head.weight → output.weight regardless of arch)
                ("lm_head.weight", "output.weight"),
                // ── passthrough ────────────────────────────────────────────
                (
                    "model.some_future_tensor.weight",
                    "model.some_future_tensor.weight",
                ),
            ],
        );
    }

    #[test]
    fn gguf_gemma4_rename_table() {
        check(
            &ModelArchitecture::Gemma4,
            &[
                // ── globals ────────────────────────────────────────────────
                (
                    "model.language_model.embed_tokens.weight",
                    "token_embd.weight",
                ),
                (
                    "model.language_model.embed_tokens_per_layer.weight",
                    "per_layer_token_embd.weight",
                ),
                (
                    "model.language_model.per_layer_model_projection.weight",
                    "per_layer_model_proj.weight",
                ),
                (
                    "model.language_model.per_layer_projection_norm.weight",
                    "per_layer_proj_norm.weight",
                ),
                ("model.language_model.norm.weight", "output_norm.weight"),
                // ── attention ──────────────────────────────────────────────
                (
                    "model.language_model.layers.0.self_attn.q_proj.weight",
                    "blk.0.attn_q.weight",
                ),
                (
                    "model.language_model.layers.5.self_attn.k_proj.weight",
                    "blk.5.attn_k.weight",
                ),
                (
                    "model.language_model.layers.12.self_attn.v_proj.weight",
                    "blk.12.attn_v.weight",
                ),
                (
                    "model.language_model.layers.3.self_attn.o_proj.weight",
                    "blk.3.attn_output.weight",
                ),
                (
                    "model.language_model.layers.7.self_attn.q_norm.weight",
                    "blk.7.attn_q_norm.weight",
                ),
                (
                    "model.language_model.layers.7.self_attn.k_norm.weight",
                    "blk.7.attn_k_norm.weight",
                ),
                // ── norms ──────────────────────────────────────────────────
                (
                    "model.language_model.layers.2.input_layernorm.weight",
                    "blk.2.attn_norm.weight",
                ),
                (
                    "model.language_model.layers.2.pre_feedforward_layernorm.weight",
                    "blk.2.ffn_norm.weight",
                ),
                (
                    "model.language_model.layers.2.post_feedforward_layernorm.weight",
                    "blk.2.post_ffw_norm.weight",
                ),
                (
                    "model.language_model.layers.2.post_attention_layernorm.weight",
                    "blk.2.post_attention_norm.weight",
                ),
                // ── MLP ────────────────────────────────────────────────────
                (
                    "model.language_model.layers.10.mlp.gate_proj.weight",
                    "blk.10.ffn_gate.weight",
                ),
                (
                    "model.language_model.layers.10.mlp.up_proj.weight",
                    "blk.10.ffn_up.weight",
                ),
                (
                    "model.language_model.layers.10.mlp.down_proj.weight",
                    "blk.10.ffn_down.weight",
                ),
                // ── Gemma4 PLI per-layer tensors ───────────────────────────
                (
                    "model.language_model.layers.1.per_layer_input_gate.weight",
                    "blk.1.inp_gate.weight",
                ),
                (
                    "model.language_model.layers.1.per_layer_projection.weight",
                    "blk.1.proj.weight",
                ),
                (
                    "model.language_model.layers.1.post_per_layer_input_norm.weight",
                    "blk.1.post_norm.weight",
                ),
                (
                    "model.language_model.layers.1.layer_scalar",
                    "blk.1.layer_output_scale.weight",
                ),
                // ── lm_head (mapped globally to output.weight regardless of arch) ──
                ("lm_head.weight", "output.weight"),
                // ── passthrough ────────────────────────────────────────────
                (
                    "model.some_future_tensor.weight",
                    "model.some_future_tensor.weight",
                ),
            ],
        );
    }

    #[test]
    fn gguf_phi3_rename_table() {
        check(
            &ModelArchitecture::Phi3,
            &[
                // ── globals ────────────────────────────────────────────────
                ("model.embed_tokens.weight", "token_embd.weight"),
                ("model.norm.weight", "output_norm.weight"),
                ("lm_head.weight", "output.weight"),
                // ── fused QKV and output projection ───────────────────────
                (
                    "model.layers.0.self_attn.qkv_proj.weight",
                    "blk.0.attn_qkv.weight",
                ),
                (
                    "model.layers.0.self_attn.o_proj.weight",
                    "blk.0.attn_output.weight",
                ),
                // ── norms ──────────────────────────────────────────────────
                (
                    "model.layers.1.input_layernorm.weight",
                    "blk.1.attn_norm.weight",
                ),
                (
                    "model.layers.1.post_attention_layernorm.weight",
                    "blk.1.ffn_norm.weight",
                ),
                // ── fused gate+up and down projection ─────────────────────
                (
                    "model.layers.2.mlp.gate_up_proj.weight",
                    "blk.2.ffn_up.weight",
                ),
                (
                    "model.layers.2.mlp.down_proj.weight",
                    "blk.2.ffn_down.weight",
                ),
                // ── passthrough ────────────────────────────────────────────
                (
                    "model.some_future_tensor.weight",
                    "model.some_future_tensor.weight",
                ),
            ],
        );
    }

    #[test]
    fn gguf_qwen3_rename_table() {
        check(
            &ModelArchitecture::Qwen3,
            &[
                // ── globals ────────────────────────────────────────────────
                ("model.embed_tokens.weight", "token_embd.weight"),
                ("model.norm.weight", "output_norm.weight"),
                ("lm_head.weight", "output.weight"),
                // ── attention ──────────────────────────────────────────────
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    "blk.0.attn_q.weight",
                ),
                (
                    "model.layers.3.self_attn.k_proj.weight",
                    "blk.3.attn_k.weight",
                ),
                (
                    "model.layers.3.self_attn.v_proj.weight",
                    "blk.3.attn_v.weight",
                ),
                (
                    "model.layers.3.self_attn.o_proj.weight",
                    "blk.3.attn_output.weight",
                ),
                (
                    "model.layers.5.self_attn.q_norm.weight",
                    "blk.5.attn_q_norm.weight",
                ),
                (
                    "model.layers.5.self_attn.k_norm.weight",
                    "blk.5.attn_k_norm.weight",
                ),
                // ── norms ──────────────────────────────────────────────────
                (
                    "model.layers.2.input_layernorm.weight",
                    "blk.2.attn_norm.weight",
                ),
                // Qwen3: post_attention_layernorm → ffn_norm (not Gemma)
                (
                    "model.layers.2.post_attention_layernorm.weight",
                    "blk.2.ffn_norm.weight",
                ),
                // ── MLP ────────────────────────────────────────────────────
                (
                    "model.layers.8.mlp.gate_proj.weight",
                    "blk.8.ffn_gate.weight",
                ),
                ("model.layers.8.mlp.up_proj.weight", "blk.8.ffn_up.weight"),
                (
                    "model.layers.8.mlp.down_proj.weight",
                    "blk.8.ffn_down.weight",
                ),
                // ── passthrough — top-level (not a layer tensor) ───────────
                (
                    "model.some_future_tensor.weight",
                    "model.some_future_tensor.weight",
                ),
                // ── passthrough — unknown layer suffix (blk.{idx}. prefix applied) ──
                (
                    "model.layers.4.some_unknown_tensor.weight",
                    "blk.4.some_unknown_tensor.weight",
                ),
            ],
        );
    }

    #[test]
    fn gguf_qwen35_rename_table() {
        check(
            &ModelArchitecture::Qwen35,
            &[
                // ── globals ────────────────────────────────────────────────
                (
                    "model.language_model.embed_tokens.weight",
                    "token_embd.weight",
                ),
                ("model.language_model.norm.weight", "output_norm.weight"),
                // ── full-attention layer ───────────────────────────────────
                (
                    "model.language_model.layers.0.self_attn.q_proj.weight",
                    "blk.0.attn_q.weight",
                ),
                (
                    "model.language_model.layers.0.self_attn.k_proj.weight",
                    "blk.0.attn_k.weight",
                ),
                (
                    "model.language_model.layers.0.self_attn.v_proj.weight",
                    "blk.0.attn_v.weight",
                ),
                (
                    "model.language_model.layers.0.self_attn.o_proj.weight",
                    "blk.0.attn_output.weight",
                ),
                (
                    "model.language_model.layers.3.self_attn.q_norm.weight",
                    "blk.3.attn_q_norm.weight",
                ),
                (
                    "model.language_model.layers.3.self_attn.k_norm.weight",
                    "blk.3.attn_k_norm.weight",
                ),
                // ── norms ──────────────────────────────────────────────────
                (
                    "model.language_model.layers.1.input_layernorm.weight",
                    "blk.1.attn_norm.weight",
                ),
                // Qwen3.5 (not Gemma): post_attention_layernorm → ffn_norm
                (
                    "model.language_model.layers.1.post_attention_layernorm.weight",
                    "blk.1.ffn_norm.weight",
                ),
                // ── MLP ────────────────────────────────────────────────────
                (
                    "model.language_model.layers.5.mlp.gate_proj.weight",
                    "blk.5.ffn_gate.weight",
                ),
                (
                    "model.language_model.layers.5.mlp.up_proj.weight",
                    "blk.5.ffn_up.weight",
                ),
                (
                    "model.language_model.layers.5.mlp.down_proj.weight",
                    "blk.5.ffn_down.weight",
                ),
                // ── SSM / linear-attention tensors (Qwen3.5-specific) ──────
                (
                    "model.language_model.layers.2.linear_attn.in_proj_qkv.weight",
                    "blk.2.linear_attn_in_proj_qkv.weight",
                ),
                (
                    "model.language_model.layers.2.linear_attn.in_proj_z.weight",
                    "blk.2.linear_attn_in_proj_z.weight",
                ),
                (
                    "model.language_model.layers.2.linear_attn.in_proj_a.weight",
                    "blk.2.linear_attn_in_proj_a.weight",
                ),
                (
                    "model.language_model.layers.2.linear_attn.in_proj_b.weight",
                    "blk.2.linear_attn_in_proj_b.weight",
                ),
                (
                    "model.language_model.layers.2.linear_attn.out_proj.weight",
                    "blk.2.linear_attn_out_proj.weight",
                ),
                // ── passthrough ────────────────────────────────────────────
                (
                    "model.some_future_tensor.weight",
                    "model.some_future_tensor.weight",
                ),
            ],
        );
    }
}

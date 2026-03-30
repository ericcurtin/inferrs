//! Model implementations.
//!
//! We use candle-transformers' model implementations directly, wrapping them
//! with a unified trait for the engine to use.

pub mod attention_utils;
pub mod qwen3;
pub mod qwen3_5;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

use crate::config::{ModelArchitecture, RawConfig};
use crate::kv_cache::{BlockTable, PagedKvStore};

/// Unified model interface for the engine.
pub trait CausalLM: Send {
    /// Run a forward pass on the given input token IDs.
    /// Returns logits for the last token position: shape (batch_size, 1, vocab_size).
    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor>;

    /// Run a paged-attention forward pass.
    ///
    /// The default implementation falls back to `forward`, ignoring the paged
    /// store.  Models that support paged attention override this.
    fn forward_paged(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        block_table: &BlockTable,
        kv_store: &mut PagedKvStore,
    ) -> Result<Tensor> {
        let _ = (block_table, kv_store); // unused in default impl
        self.forward(input_ids, seqlen_offset)
    }

    /// Clear all KV caches (for starting a new sequence).
    fn clear_kv_cache(&mut self);
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

/// Load a model from weight files.
pub fn load_model(
    raw_config: &RawConfig,
    arch: &ModelArchitecture,
    weight_paths: &[impl AsRef<Path>],
    dtype: DType,
    device: &Device,
    turbo_quant_bits: Option<u8>,
) -> Result<Box<dyn CausalLM>> {
    tracing::info!("Loading model weights ({:?} architecture)...", arch);

    let paths_ref: Vec<&Path> = weight_paths.iter().map(|p| p.as_ref()).collect();

    // Load safetensors into a VarBuilder
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths_ref, dtype, device)? };

    // Warn if --turbo-quant was requested but this architecture doesn't support it.
    if turbo_quant_bits.is_some() {
        match arch {
            ModelArchitecture::Qwen3 => {} // supported
            other => {
                tracing::warn!(
                    "--turbo-quant is not supported for {:?} and will be ignored. \
                     TurboQuant KV cache compression is currently only available for Qwen3.",
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
    };
    tracing::info!("Model loaded successfully");
    Ok(model)
}

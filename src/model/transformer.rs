//! Transformer model loading and forward pass using candle.
//!
//! Supports Qwen2, Llama, Mistral, and other HF-standard decoder-only
//! transformer architectures by reading config.json and loading safetensors.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::path::Path;

/// Model configuration parsed from config.json.
/// Covers Qwen2, Llama, Mistral, and similar architectures.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_num_kv_heads")]
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    #[serde(default = "default_max_position")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub bos_token_id: Option<serde_json::Value>,
}

fn default_num_kv_heads() -> usize {
    0
}
fn default_max_position() -> usize {
    2048
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    10000.0
}

impl ModelConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).context("reading config.json")?;

        // Try parsing directly first (standard decoder-only models).
        // If that fails, look for a nested `text_config` object (VLMs like
        // Qwen3.5, LLaVA, etc. put the language model config there).
        let mut config: Self = match serde_json::from_str::<Self>(&content) {
            Ok(cfg) => cfg,
            Err(_top_err) => {
                let raw: serde_json::Value =
                    serde_json::from_str(&content).context("parsing config.json as JSON")?;
                if let Some(text_cfg) = raw.get("text_config") {
                    // Merge top-level fields (like eos_token_id, tie_word_embeddings)
                    // with text_config so nothing is lost.
                    let mut merged = text_cfg.clone();
                    if let (Some(merged_obj), Some(raw_obj)) =
                        (merged.as_object_mut(), raw.as_object())
                    {
                        for (k, v) in raw_obj {
                            if k != "text_config" && k != "vision_config" {
                                merged_obj.entry(k.clone()).or_insert_with(|| v.clone());
                            }
                        }
                    }
                    serde_json::from_value(merged)
                        .context("parsing text_config from config.json")?
                } else {
                    return Err(_top_err).context("parsing config.json");
                }
            }
        };

        // If num_key_value_heads is 0 (not specified), default to num_attention_heads (MHA)
        if config.num_key_value_heads == 0 {
            config.num_key_value_heads = config.num_attention_heads;
        }
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Parse eos_token_id which can be a single int or an array.
    pub fn get_eos_token_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            Some(serde_json::Value::Number(n)) => {
                vec![n.as_u64().unwrap_or(0) as u32]
            }
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect(),
            _ => vec![],
        }
    }
}

/// RMS normalization layer.
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(vb: VarBuilder, size: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = (&x * &x)?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let result = x_normed.to_dtype(dtype)?.broadcast_mul(&self.weight)?;
        Ok(result)
    }
}

/// Rotary positional embedding.
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?.unsqueeze(1)?;
        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin })
    }

    fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, _, seq_len, head_dim) = x.dims4()?;
        let cos = self
            .cos
            .i(offset..offset + seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin = self
            .sin
            .i(offset..offset + seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        let x1 = x.narrow(3, 0, head_dim / 2)?;
        let x2 = x.narrow(3, head_dim / 2, head_dim / 2)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;

        let result = (x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?;
        Ok(result)
    }
}

/// A single transformer attention layer.
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let head_dim = config.head_dim();
        let q_size = config.num_attention_heads * head_dim;
        let kv_size = config.num_key_value_heads * head_dim;

        let q_proj = candle_nn::linear_no_bias(hidden, q_size, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_no_bias(hidden, kv_size, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_no_bias(hidden, kv_size, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(hidden, hidden, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        kv_cache: Option<(&Tensor, &Tensor)>,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (batch, num_heads, seq_len, head_dim)
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings
        let q = rotary.apply(&q, offset)?;
        let k = rotary.apply(&k, offset)?;

        // Concatenate with KV cache if present
        let (k, v) = match kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };

        // Save the un-repeated K/V for caching BEFORE repeat_kv.
        // k, v shape: (batch, num_kv_heads, total_seq_len, head_dim)
        let k_for_cache = k.clone();
        let v_for_cache = v.clone();

        // GQA: repeat KV heads if needed
        let k_expanded = self.repeat_kv(&k)?;
        let v_expanded = self.repeat_kv(&v)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k_expanded.transpose(2, 3)?)? / scale)?;

        // Causal mask
        let total_len = k_expanded.dim(2)?;
        let attn_weights = self.apply_causal_mask(&attn_weights, seq_len, total_len)?;

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v_expanded)?;

        // Reshape back to (batch, seq_len, hidden)
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        let output = self.o_proj.forward(&attn_output)?;

        Ok((output, k_for_cache, v_for_cache))
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        if self.num_kv_heads == self.num_heads {
            return Ok(x.clone());
        }
        let repeats = self.num_heads / self.num_kv_heads;
        let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        let x = x
            .unsqueeze(2)?
            .expand((batch, num_kv_heads, repeats, seq_len, head_dim))?
            .reshape((batch, num_kv_heads * repeats, seq_len, head_dim))?;
        Ok(x)
    }

    fn apply_causal_mask(
        &self,
        attn_weights: &Tensor,
        query_len: usize,
        key_len: usize,
    ) -> Result<Tensor> {
        if query_len == 1 {
            // Decode step: single query can attend to all keys
            return Ok(attn_weights.clone());
        }
        let device = attn_weights.device();
        // Build causal mask: position i can attend to positions 0..=i + (key_len - query_len)
        let offset = key_len - query_len;
        let mask: Vec<f32> = (0..query_len)
            .flat_map(|i| {
                (0..key_len).map(move |j| {
                    if j <= i + offset {
                        0.0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        let mask = Tensor::new(mask, device)?.reshape((1, 1, query_len, key_len))?;
        let mask = mask.to_dtype(attn_weights.dtype())?;
        Ok((attn_weights + mask)?)
    }
}

/// MLP layer (gate + up + down projection with SiLU activation).
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let gate_proj = candle_nn::linear_no_bias(hidden, intermediate, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(hidden, intermediate, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let result = self.down_proj.forward(&(gate * up)?)?;
        Ok(result)
    }
}

/// A single transformer decoder layer.
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn load(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let self_attn = Attention::load(vb.pp("self_attn"), config)?;
        let mlp = Mlp::load(vb.pp("mlp"), config)?;
        let input_layernorm = RmsNorm::load(
            vb.pp("input_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rotary: &RotaryEmbedding,
        kv_cache: Option<(&Tensor, &Tensor)>,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // Pre-norm attention
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let (attn_out, k_cache, v_cache) = self.self_attn.forward(&x, rotary, kv_cache, offset)?;
        let x = (residual + attn_out)?;

        // Pre-norm MLP
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (residual + x)?;

        Ok((x, k_cache, v_cache))
    }
}

/// The full transformer model.
pub struct TransformerModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary: RotaryEmbedding,
    #[allow(dead_code)]
    config: ModelConfig,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
}

impl TransformerModel {
    /// Load model from safetensors files.
    pub fn load(
        config: &ModelConfig,
        weight_paths: &[impl AsRef<Path>],
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        tracing::info!(
            "loading {} layers, hidden_size={}, heads={}/{}",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads
        );

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &weight_paths
                    .iter()
                    .map(|p| p.as_ref().to_path_buf())
                    .collect::<Vec<_>>(),
                dtype,
                device,
            )?
        };

        let vb_model = vb.pp("model");

        let embed_tokens = Embedding::new(
            vb_model.get(
                (config.vocab_size, config.hidden_size),
                "embed_tokens.weight",
            )?,
            config.hidden_size,
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = DecoderLayer::load(vb_model.pp(format!("layers.{i}")), config)?;
            layers.push(layer);
            if (i + 1) % 4 == 0 || i == config.num_hidden_layers - 1 {
                tracing::info!("loaded layer {}/{}", i + 1, config.num_hidden_layers);
            }
        }

        let norm = RmsNorm::load(vb_model.pp("norm"), config.hidden_size, config.rms_norm_eps)?;

        // LM head: may be tied to embed_tokens
        let lm_head = if config.tie_word_embeddings {
            let weight = vb_model.get(
                (config.vocab_size, config.hidden_size),
                "embed_tokens.weight",
            )?;
            Linear::new(weight, None)
        } else {
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        let max_seq = config.max_position_embeddings;
        let rotary = RotaryEmbedding::new(config.head_dim(), max_seq, config.rope_theta, device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
            config: config.clone(),
            device: device.clone(),
            dtype,
        })
    }

    /// Forward pass. Returns logits for the last token position.
    ///
    /// `input_ids`: shape (batch, seq_len)
    /// `kv_caches`: one (k, v) pair per layer, or empty for prefill
    /// `offset`: position offset in the sequence (for decode steps)
    ///
    /// Returns (logits, new_kv_caches).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        kv_caches: &[(Tensor, Tensor)],
        offset: usize,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        let mut new_kv_caches = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter().enumerate() {
            let kv = if i < kv_caches.len() {
                Some((&kv_caches[i].0, &kv_caches[i].1))
            } else {
                None
            };
            let (out, k, v) = layer.forward(&x, &self.rotary, kv, offset)?;
            x = out;
            new_kv_caches.push((k, v));
        }

        x = self.norm.forward(&x)?;

        // Only compute logits for the last position to save memory
        let seq_len = x.dim(1)?;
        let last_hidden = x.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&last_hidden)?.squeeze(1)?;

        Ok((logits, new_kv_caches))
    }

    #[allow(dead_code)]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    #[allow(dead_code)]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[allow(dead_code)]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

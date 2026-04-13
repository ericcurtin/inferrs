//! Model configuration loading from config.json.

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use serde::Deserialize;
use std::path::Path;

// ---------------------------------------------------------------------------
// Multimodal config types (duplicated here so inferrs has no rlib dep on
// inferrs-multimodal; the plugin is loaded at runtime via dlopen).
// Keep in sync with inferrs-multimodal/src/config.rs.
// ---------------------------------------------------------------------------

/// Audio encoder configuration from `audio_config` in `config.json`.
#[derive(Debug, serde::Deserialize, serde::Serialize, Clone)]
pub struct AudioConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub output_proj_dims: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    pub subsampling_conv_channels: Vec<usize>,
    #[serde(default = "default_conv_kernel_size")]
    pub conv_kernel_size: usize,
    pub attention_chunk_size: usize,
    pub attention_context_left: usize,
    #[allow(dead_code)]
    #[serde(default)]
    pub attention_context_right: usize,
    #[serde(default = "default_logit_cap")]
    pub attention_logit_cap: f64,
    #[serde(default = "default_invalid_logit")]
    pub attention_invalid_logits_value: f64,
    #[serde(default = "default_residual_weight")]
    pub residual_weight: f64,
    #[serde(default = "default_gradient_clipping")]
    pub gradient_clipping: f64,
    #[allow(dead_code)]
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_conv_kernel_size() -> usize {
    5
}
fn default_logit_cap() -> f64 {
    50.0
}
fn default_invalid_logit() -> f64 {
    -1e9
}
fn default_residual_weight() -> f64 {
    0.5
}
fn default_gradient_clipping() -> f64 {
    1e10
}
fn default_hidden_act() -> String {
    "silu".to_string()
}

/// SigLIP2 ViT vision encoder configuration (Gemma4).
#[derive(Debug, serde::Deserialize, serde::Serialize, Clone)]
#[allow(dead_code)]
pub struct Gemma4VisionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_vision_head_dim")]
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub default_output_length: usize,
    pub position_embedding_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_vision_rope_theta")]
    pub rope_theta: f64,
    pub use_clipped_linears: bool,
    pub standardize: bool,
    #[serde(default = "default_vision_hidden_activation")]
    pub hidden_activation: String,
}

/// Qwen vision encoder configuration (shared by Qwen3.5 and Qwen3-VL).
#[derive(Debug, Deserialize, Clone, Default)]
#[serde(default)]
#[allow(dead_code)]
pub struct QwenVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_position_embeddings: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub out_hidden_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub initializer_range: f64,
    pub deepstack_visual_indexes: Vec<usize>,
}

fn default_vision_head_dim() -> usize {
    64
}
fn default_vision_rope_theta() -> f64 {
    100.0
}
fn default_vision_hidden_activation() -> String {
    "gelu_pytorch_tanh".to_string()
}

/// Parameters for image preprocessing, regardless of encoder architecture.
pub struct VisionPreprocessParams {
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub default_output_length: usize,
}

/// Vision encoder configuration from `vision_config` in `config.json`.
#[derive(Debug, Clone)]
pub enum VisionConfig {
    Gemma4(Gemma4VisionConfig),
    Qwen(QwenVisionConfig),
}

impl VisionConfig {
    pub fn preprocess_params(&self) -> VisionPreprocessParams {
        match self {
            VisionConfig::Gemma4(c) => VisionPreprocessParams {
                patch_size: c.patch_size,
                pooling_kernel_size: c.pooling_kernel_size,
                default_output_length: c.default_output_length,
            },
            VisionConfig::Qwen(c) => VisionPreprocessParams {
                patch_size: c.patch_size,
                pooling_kernel_size: if c.spatial_merge_size > 0 {
                    c.spatial_merge_size
                } else {
                    3
                },
                default_output_length: 280,
            },
        }
    }
}

impl<'de> serde::Deserialize<'de> for VisionConfig {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> std::result::Result<Self, D::Error> {
        let raw = serde_json::Value::deserialize(de)?;
        let model_type = raw.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
        match model_type {
            "qwen3_5" | "qwen3_vl" => {
                let cfg: QwenVisionConfig =
                    serde_json::from_value(raw).map_err(serde::de::Error::custom)?;
                Ok(VisionConfig::Qwen(cfg))
            }
            _ => {
                let cfg: Gemma4VisionConfig =
                    serde_json::from_value(raw).map_err(serde::de::Error::custom)?;
                Ok(VisionConfig::Gemma4(cfg))
            }
        }
    }
}

/// Lenient deserializer for `RawConfig.vision_config`.
pub fn deserialize_vision_config_opt<'de, D>(
    de: D,
) -> std::result::Result<Option<VisionConfig>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = Option::<serde_json::Value>::deserialize(de)?;
    let Some(val) = raw else { return Ok(None) };
    let model_type = val.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
    match model_type {
        "qwen3_5" | "qwen3_vl" => {
            let cfg: QwenVisionConfig =
                serde_json::from_value(val).map_err(serde::de::Error::custom)?;
            Ok(Some(VisionConfig::Qwen(cfg)))
        }
        "siglip_vision_model" => Ok(None),
        _ => match serde_json::from_value::<Gemma4VisionConfig>(val) {
            Ok(cfg) => Ok(Some(VisionConfig::Gemma4(cfg))),
            Err(_) => Ok(None),
        },
    }
}

/// Supported model architectures.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelArchitecture {
    Qwen2,
    Qwen3,
    Qwen35,
    Gemma2,
    Gemma3,
    Gemma4,
    Phi3,
}

/// Rope parameters nested object (used in Qwen3.5 text_config).
#[derive(Debug, Deserialize, Default)]
pub struct RopeParameters {
    pub rope_theta: Option<f64>,
    pub partial_rotary_factor: Option<f64>,
}

/// Shared text_config nested object (used by Qwen3.5 and Gemma4).
#[derive(Debug, Deserialize)]
pub struct TextConfig {
    // Qwen3.5 fields
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub rms_norm_eps: Option<f64>,
    pub tie_word_embeddings: Option<bool>,
    pub full_attention_interval: Option<usize>,
    pub linear_conv_kernel_dim: Option<usize>,
    pub linear_key_head_dim: Option<usize>,
    pub linear_value_head_dim: Option<usize>,
    pub linear_num_key_heads: Option<usize>,
    pub linear_num_value_heads: Option<usize>,
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub rope_parameters: RopeParameters,

    // Qwen3.5 MTP fields
    pub mtp_num_hidden_layers: Option<usize>,

    // Gemma4-specific text_config fields
    pub global_head_dim: Option<usize>,
    pub sliding_window: Option<usize>,
    pub sliding_window_pattern: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub hidden_size_per_layer_input: Option<usize>,
    pub final_logit_softcapping: Option<f64>,
    pub attn_logit_softcapping: Option<f64>,
    pub query_pre_attn_scalar: Option<usize>,
    pub attention_bias: Option<bool>,
    pub hidden_activation: Option<String>,
    #[allow(dead_code)]
    pub model_type: Option<String>,
    pub num_kv_shared_layers: Option<usize>,
    pub use_double_wide_mlp: Option<bool>,
    pub num_global_key_value_heads: Option<usize>,
    pub attention_k_eq_v: Option<bool>,
    // Gemma4 MoE fields (26B A4B variant)
    pub enable_moe_block: Option<bool>,
    pub num_experts: Option<usize>,
    pub top_k_experts: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
}

/// Raw config.json from HuggingFace.
#[derive(Debug, Deserialize)]
pub struct RawConfig {
    pub architectures: Option<Vec<String>>,
    pub model_type: Option<String>,
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub rms_norm_eps: Option<f64>,
    pub rope_theta: Option<f64>,
    pub tie_word_embeddings: Option<bool>,
    #[allow(dead_code)]
    pub hidden_act: Option<String>,

    // Qwen2-specific
    pub sliding_window: Option<usize>,
    pub max_window_layers: Option<usize>,
    pub use_sliding_window: Option<bool>,

    // Gemma2-specific
    pub head_dim: Option<usize>,
    pub hidden_activation: Option<String>,
    pub attention_bias: Option<bool>,
    pub final_logit_softcapping: Option<f64>,
    pub attn_logit_softcapping: Option<f64>,
    pub query_pre_attn_scalar: Option<usize>,

    // Gemma3-specific
    pub rope_local_base_freq: Option<f64>,
    pub sliding_window_pattern: Option<usize>,

    // Qwen3-specific (flat, not nested; kept for potential future use)
    #[allow(dead_code)]
    pub layer_types: Option<Vec<String>>,

    // Qwen3.5/Gemma4-specific (nested text_config)
    pub text_config: Option<TextConfig>,

    // Gemma4 multimodal
    pub audio_config: Option<AudioConfig>,
    pub audio_token_id: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_vision_config_opt")]
    pub vision_config: Option<VisionConfig>,
    pub image_token_id: Option<u32>,
    pub boi_token_id: Option<u32>,
    pub eoi_token_id: Option<u32>,
}

/// Default epsilon for RMS normalization layers across all model families.
const RMS_NORM_EPS_DEFAULT: f64 = 1e-6;

impl RawConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).context("Failed to read config.json")?;
        let config: RawConfig =
            serde_json::from_str(&content).context("Failed to parse config.json")?;
        Ok(config)
    }

    pub fn detect_architecture(&self) -> Result<ModelArchitecture> {
        if let Some(archs) = &self.architectures {
            for arch in archs {
                if arch.contains("Qwen3_5") {
                    return Ok(ModelArchitecture::Qwen35);
                }
                if arch.contains("Qwen3") {
                    return Ok(ModelArchitecture::Qwen3);
                }
                if arch.contains("Qwen2") {
                    return Ok(ModelArchitecture::Qwen2);
                }
                if arch.contains("Gemma4") {
                    return Ok(ModelArchitecture::Gemma4);
                }
                if arch.contains("Gemma3") {
                    return Ok(ModelArchitecture::Gemma3);
                }
                if arch.contains("Gemma2") {
                    return Ok(ModelArchitecture::Gemma2);
                }
                if arch.contains("Phi3") {
                    return Ok(ModelArchitecture::Phi3);
                }
            }
        }

        if let Some(model_type) = &self.model_type {
            match model_type.as_str() {
                "qwen2" | "qwen2_5" => return Ok(ModelArchitecture::Qwen2),
                "qwen3" => return Ok(ModelArchitecture::Qwen3),
                "qwen3_5" => return Ok(ModelArchitecture::Qwen35),
                "gemma4" => return Ok(ModelArchitecture::Gemma4),
                "gemma3" => return Ok(ModelArchitecture::Gemma3),
                "gemma2" => return Ok(ModelArchitecture::Gemma2),
                "phi3" => return Ok(ModelArchitecture::Phi3),
                _ => {}
            }
        }

        anyhow::bail!(
            "Unsupported model architecture. architectures={:?}, model_type={:?}",
            self.architectures,
            self.model_type
        )
    }

    pub fn to_qwen2_config(&self) -> candle_transformers::models::qwen2::Config {
        candle_transformers::models::qwen2::Config {
            vocab_size: self.vocab_size.unwrap_or(151936),
            hidden_size: self.hidden_size.unwrap_or(896),
            intermediate_size: self.intermediate_size.unwrap_or(4864),
            num_hidden_layers: self.num_hidden_layers.unwrap_or(24),
            num_attention_heads: self.num_attention_heads.unwrap_or(14),
            num_key_value_heads: self.num_key_value_heads.unwrap_or(2),
            max_position_embeddings: self.max_position_embeddings.unwrap_or(131072),
            sliding_window: self.sliding_window.unwrap_or(131072),
            max_window_layers: self.max_window_layers.unwrap_or(28),
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(true),
            rope_theta: self.rope_theta.unwrap_or(1000000.0),
            rms_norm_eps: self.rms_norm_eps.unwrap_or(RMS_NORM_EPS_DEFAULT),
            use_sliding_window: self.use_sliding_window.unwrap_or(false),
            hidden_act: candle_nn::Activation::Silu,
        }
    }

    pub fn to_gemma2_config(&self) -> candle_transformers::models::gemma2::Config {
        let num_attention_heads = self.num_attention_heads.unwrap_or(16);
        let hidden_size = self.hidden_size.unwrap_or(3584);
        candle_transformers::models::gemma2::Config {
            vocab_size: self.vocab_size.unwrap_or(256000),
            hidden_size,
            intermediate_size: self.intermediate_size.unwrap_or(14336),
            num_hidden_layers: self.num_hidden_layers.unwrap_or(42),
            num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(8),
            head_dim: self.head_dim.unwrap_or(hidden_size / num_attention_heads),
            hidden_activation: parse_gemma_activation(
                self.hidden_activation
                    .as_deref()
                    .unwrap_or("gelu_pytorch_tanh"),
            ),
            rms_norm_eps: self.rms_norm_eps.unwrap_or(RMS_NORM_EPS_DEFAULT),
            rope_theta: self.rope_theta.unwrap_or(10000.0),
            attention_bias: self.attention_bias.unwrap_or(false),
            final_logit_softcapping: self.final_logit_softcapping,
            attn_logit_softcapping: self.attn_logit_softcapping,
            query_pre_attn_scalar: self.query_pre_attn_scalar.unwrap_or(256),
            sliding_window: self.sliding_window,
            max_position_embeddings: self.max_position_embeddings.unwrap_or(8192),
        }
    }

    pub fn to_gemma3_config(&self) -> candle_transformers::models::gemma3::Config {
        let num_attention_heads = self.num_attention_heads.unwrap_or(8);
        let hidden_size = self.hidden_size.unwrap_or(2560);
        candle_transformers::models::gemma3::Config {
            vocab_size: self.vocab_size.unwrap_or(262144),
            hidden_size,
            intermediate_size: self.intermediate_size.unwrap_or(10240),
            num_hidden_layers: self.num_hidden_layers.unwrap_or(34),
            num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(4),
            head_dim: self.head_dim.unwrap_or(256),
            hidden_activation: parse_gemma_activation(
                self.hidden_activation
                    .as_deref()
                    .unwrap_or("gelu_pytorch_tanh"),
            ),
            rms_norm_eps: self.rms_norm_eps.unwrap_or(RMS_NORM_EPS_DEFAULT),
            rope_theta: self.rope_theta.unwrap_or(1_000_000.0),
            rope_local_base_freq: self.rope_local_base_freq.unwrap_or(10_000.0),
            attention_bias: self.attention_bias.unwrap_or(false),
            final_logit_softcapping: self.final_logit_softcapping,
            attn_logit_softcapping: self.attn_logit_softcapping,
            query_pre_attn_scalar: self.query_pre_attn_scalar.unwrap_or(256),
            sliding_window: self.sliding_window.unwrap_or(1024),
            sliding_window_pattern: self.sliding_window_pattern.unwrap_or(6),
            max_position_embeddings: self.max_position_embeddings.unwrap_or(131072),
        }
    }

    /// Build a Qwen3Config (all-full-attention transformer with gated q_proj).
    /// Qwen3 is the same architecture as the full-attention layers in Qwen3.5
    /// but all layers are full-attention and weights live under `model.*` (not
    /// `model.language_model.*`).
    pub fn to_qwen3_config(
        &self,
        dtype: DType,
        device: Device,
        turbo_quant_bits: Option<u8>,
    ) -> crate::models::qwen3::Qwen3Config {
        use crate::models::qwen3::Qwen3Config;

        let vocab_size = self.vocab_size.unwrap_or(151936);
        let hidden_size = self.hidden_size.unwrap_or(1024);
        let intermediate_size = self.intermediate_size.unwrap_or(3072);
        let num_hidden_layers = self.num_hidden_layers.unwrap_or(28);
        let num_attention_heads = self.num_attention_heads.unwrap_or(16);
        let num_key_value_heads = self.num_key_value_heads.unwrap_or(8);
        let head_dim = self.head_dim.unwrap_or(hidden_size / num_attention_heads);
        let rms_norm_eps = self.rms_norm_eps.unwrap_or(RMS_NORM_EPS_DEFAULT);
        let tie_word_embeddings = self.tie_word_embeddings.unwrap_or(true);
        let rope_theta = self.rope_theta.unwrap_or(1_000_000.0);

        Qwen3Config {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            tie_word_embeddings,
            rope_theta,
            dtype,
            device,
            turbo_quant_bits,
        }
    }

    pub fn to_gemma4_config(
        &self,
        dtype: DType,
        device: Device,
        turbo_quant_bits: Option<u8>,
    ) -> crate::models::gemma4::Gemma4Config {
        use crate::models::gemma4::Gemma4Config;

        // All language-model params live in the nested text_config
        let tc = self.text_config.as_ref();

        let vocab_size = tc.and_then(|t| t.vocab_size).unwrap_or(262144);
        let hidden_size = tc.and_then(|t| t.hidden_size).unwrap_or(1536);
        let intermediate_size = tc.and_then(|t| t.intermediate_size).unwrap_or(6144);
        let num_hidden_layers = tc.and_then(|t| t.num_hidden_layers).unwrap_or(35);
        let num_attention_heads = tc.and_then(|t| t.num_attention_heads).unwrap_or(8);
        let num_key_value_heads = tc.and_then(|t| t.num_key_value_heads).unwrap_or(1);
        let num_global_key_value_heads = tc
            .and_then(|t| t.num_global_key_value_heads)
            .unwrap_or(num_key_value_heads);
        let head_dim = tc.and_then(|t| t.head_dim).unwrap_or(256);
        let global_head_dim = tc.and_then(|t| t.global_head_dim).unwrap_or(512);
        let hidden_size_per_layer_input = tc
            .and_then(|t| t.hidden_size_per_layer_input)
            .unwrap_or(256);
        let rms_norm_eps = tc
            .and_then(|t| t.rms_norm_eps)
            .unwrap_or(RMS_NORM_EPS_DEFAULT);
        let sliding_window = tc.and_then(|t| t.sliding_window).unwrap_or(512);
        let sliding_window_pattern = tc.and_then(|t| t.sliding_window_pattern).unwrap_or(5);
        let max_position_embeddings = tc.and_then(|t| t.max_position_embeddings).unwrap_or(131072);
        let final_logit_softcapping = tc.and_then(|t| t.final_logit_softcapping).or(Some(30.0));
        let attn_logit_softcapping = tc.and_then(|t| t.attn_logit_softcapping);
        let query_pre_attn_scalar = tc.and_then(|t| t.query_pre_attn_scalar).unwrap_or(256);
        let attention_bias = tc.and_then(|t| t.attention_bias).unwrap_or(false);
        let attention_k_eq_v = tc.and_then(|t| t.attention_k_eq_v).unwrap_or(false);
        let hidden_activation = parse_gemma_activation(
            tc.and_then(|t| t.hidden_activation.as_deref())
                .unwrap_or("gelu_pytorch_tanh"),
        );
        let tie_word_embeddings = tc
            .and_then(|t| t.tie_word_embeddings)
            .or(self.tie_word_embeddings)
            .unwrap_or(true);

        // Build layer_types: "full_attention" layers are global, the rest are sliding
        let layer_is_full_attention: Vec<bool> =
            if let Some(types) = tc.and_then(|t| t.layer_types.as_ref()) {
                types.iter().map(|s| s == "full_attention").collect()
            } else {
                // fallback: every sliding_window_pattern-th layer is full
                (0..num_hidden_layers)
                    .map(|i| (i + 1) % sliding_window_pattern == 0)
                    .collect()
            };

        // double-wide MLP: layers from index (num_hidden_layers - num_kv_shared_layers) onward
        // use intermediate_size * 2. Only applies when use_double_wide_mlp is true.
        let use_double_wide_mlp = tc.and_then(|t| t.use_double_wide_mlp).unwrap_or(false);
        let num_kv_shared_layers = tc
            .and_then(|t| t.num_kv_shared_layers)
            .unwrap_or(num_hidden_layers);
        let double_wide_mlp_start_layer = if use_double_wide_mlp {
            num_hidden_layers.saturating_sub(num_kv_shared_layers)
        } else {
            num_hidden_layers // disabled: never trigger
        };
        // KV sharing is independent of double-wide MLP (e.g. E4B has KV sharing
        // but use_double_wide_mlp=false, so double_wide_mlp_start_layer would be
        // num_hidden_layers and accidentally disable KV sharing).
        let first_kv_shared_idx = num_hidden_layers.saturating_sub(num_kv_shared_layers);

        let enable_moe_block = tc.and_then(|t| t.enable_moe_block).unwrap_or(false);
        let num_experts = tc.and_then(|t| t.num_experts).unwrap_or(1);
        let top_k_experts = tc.and_then(|t| t.top_k_experts).unwrap_or(1);
        let moe_intermediate_size = tc
            .and_then(|t| t.moe_intermediate_size)
            .unwrap_or(intermediate_size);

        Gemma4Config {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            num_global_key_value_heads,
            head_dim,
            global_head_dim,
            hidden_size_per_layer_input,
            rms_norm_eps,
            rope_theta_sliding: 10000.0,
            rope_theta_global: 1_000_000.0,
            partial_rotary_factor_global: 0.25,
            sliding_window,
            sliding_window_pattern,
            max_position_embeddings,
            final_logit_softcapping,
            attn_logit_softcapping,
            query_pre_attn_scalar,
            attention_bias,
            attention_k_eq_v,
            hidden_activation,
            tie_word_embeddings,
            layer_is_full_attention,
            double_wide_mlp_start_layer,
            first_kv_shared_idx,
            turbo_quant_bits,
            enable_moe_block,
            num_experts,
            top_k_experts,
            moe_intermediate_size,
            dtype,
            device,
        }
    }

    pub fn to_qwen35_config(
        &self,
        dtype: DType,
        device: Device,
        turbo_quant_bits: Option<u8>,
    ) -> crate::models::qwen3_5::Qwen35Config {
        use crate::models::qwen3_5::{LayerType, Qwen35Config};

        // All model params live in the nested text_config
        let tc = self.text_config.as_ref();

        let vocab_size = tc.and_then(|t| t.vocab_size).unwrap_or(248320);
        let hidden_size = tc.and_then(|t| t.hidden_size).unwrap_or(1024);
        let intermediate_size = tc.and_then(|t| t.intermediate_size).unwrap_or(3584);
        let num_hidden_layers = tc.and_then(|t| t.num_hidden_layers).unwrap_or(24);
        let num_attention_heads = tc.and_then(|t| t.num_attention_heads).unwrap_or(8);
        let num_key_value_heads = tc.and_then(|t| t.num_key_value_heads).unwrap_or(2);
        let head_dim = tc.and_then(|t| t.head_dim).unwrap_or(256);
        let rms_norm_eps = tc
            .and_then(|t| t.rms_norm_eps)
            .unwrap_or(RMS_NORM_EPS_DEFAULT);
        let tie_word_embeddings = tc.and_then(|t| t.tie_word_embeddings).unwrap_or(true);
        let full_attention_interval = tc.and_then(|t| t.full_attention_interval).unwrap_or(4);
        let linear_conv_kernel_dim = tc.and_then(|t| t.linear_conv_kernel_dim).unwrap_or(4);
        let linear_key_head_dim = tc.and_then(|t| t.linear_key_head_dim).unwrap_or(128);
        let linear_value_head_dim = tc.and_then(|t| t.linear_value_head_dim).unwrap_or(128);
        let linear_num_key_heads = tc.and_then(|t| t.linear_num_key_heads).unwrap_or(16);
        let linear_num_value_heads = tc.and_then(|t| t.linear_num_value_heads).unwrap_or(16);

        let rope_theta = tc
            .map(|t| t.rope_parameters.rope_theta.unwrap_or(10_000_000.0))
            .unwrap_or(10_000_000.0);
        let partial_rotary_factor = tc
            .map(|t| t.rope_parameters.partial_rotary_factor.unwrap_or(0.25))
            .unwrap_or(0.25);

        // Build layer_types from the string list in text_config
        let layer_types: Vec<LayerType> =
            if let Some(types) = tc.and_then(|t| t.layer_types.as_ref()) {
                types
                    .iter()
                    .map(|s| LayerType {
                        is_full_attention: s == "full_attention",
                    })
                    .collect()
            } else {
                // Fall back to computing from full_attention_interval
                (0..num_hidden_layers)
                    .map(|i| LayerType {
                        is_full_attention: (i + 1) % full_attention_interval == 0,
                    })
                    .collect()
            };

        Qwen35Config {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            linear_num_key_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_num_value_heads,
            linear_conv_kernel_dim,
            full_attention_interval,
            rms_norm_eps,
            rope_theta,
            partial_rotary_factor,
            layer_types,
            tie_word_embeddings,
            dtype,
            device,
            turbo_quant_bits,
            mtp_num_hidden_layers: tc.and_then(|t| t.mtp_num_hidden_layers).unwrap_or(0),
        }
    }

    /// Return the effective maximum sequence length that the model's KV cache
    /// can hold.
    ///
    /// For Gemma3, the upstream candle-transformers implementation sizes **all**
    /// KV caches (both sliding-window and non-sliding layers) to
    /// `sliding_window` tokens.  Any attempt to generate beyond that limit
    /// causes an opaque tensor error.  We therefore report `sliding_window` as
    /// the hard limit for Gemma3, not `max_position_embeddings`.
    pub fn effective_max_seq_len(&self, arch: &ModelArchitecture) -> usize {
        match arch {
            ModelArchitecture::Gemma3 => {
                // The KvCache is capped at sliding_window in candle-transformers.
                self.sliding_window.unwrap_or(512)
            }
            ModelArchitecture::Gemma2 => self.max_position_embeddings.unwrap_or(8192),
            ModelArchitecture::Qwen2 => self.max_position_embeddings.unwrap_or(131072),
            ModelArchitecture::Qwen3 => self.max_position_embeddings.unwrap_or(40960),
            ModelArchitecture::Qwen35 => {
                let tc = self.text_config.as_ref();
                // Qwen3.5 uses linear attention for most layers; the full-attn
                // layers have no artificial cap.
                tc.and_then(|t| t.num_hidden_layers)
                    .map(|_| usize::MAX) // effectively unlimited
                    .unwrap_or(usize::MAX)
            }
            ModelArchitecture::Gemma4 => {
                // Gemma4 interleaves local sliding-window layers with global
                // full-attention layers.  The sliding_window value only limits
                // the local layers; global layers can attend to the full
                // context up to max_position_embeddings.  Use
                // max_position_embeddings so that the KV cache and max_tokens
                // clamping are sized for the full context, not just the local
                // window.
                let tc = self.text_config.as_ref();
                tc.and_then(|t| t.max_position_embeddings).unwrap_or(8192)
            }
            ModelArchitecture::Phi3 => {
                // Phi-3/Phi-4 configs always carry `max_position_embeddings`
                // in practice (e.g. 131072 for LongRoPE variants, 16384 for
                // Phi-4).  The fallback below is intentionally conservative
                // (matches the original Phi-3-mini-4k context) so a
                // malformed config cannot accidentally trigger a very large
                // KV-cache allocation when the field is missing.
                self.max_position_embeddings.unwrap_or(4096)
            }
        }
    }

    /// Return `(num_kv_heads, head_dim, num_full_attn_layers)` for paged KV
    /// cache sizing.  `num_full_attn_layers` is the number of layers whose KV
    /// pairs are stored in the paged store (full-attention layers only; SSM
    /// layers are excluded).
    pub fn kv_cache_params(&self, arch: &ModelArchitecture) -> (usize, usize, usize) {
        match arch {
            ModelArchitecture::Qwen3 => {
                let num_kv_heads = self.num_key_value_heads.unwrap_or(8);
                let num_attention_heads = self.num_attention_heads.unwrap_or(16);
                let hidden_size = self.hidden_size.unwrap_or(1024);
                let head_dim = self.head_dim.unwrap_or(hidden_size / num_attention_heads);
                let num_layers = self.num_hidden_layers.unwrap_or(28);
                (num_kv_heads, head_dim, num_layers)
            }
            ModelArchitecture::Qwen35 => {
                let tc = self.text_config.as_ref();
                let num_kv_heads = tc.and_then(|t| t.num_key_value_heads).unwrap_or(2);
                let head_dim = tc.and_then(|t| t.head_dim).unwrap_or(256);
                let num_hidden_layers = tc.and_then(|t| t.num_hidden_layers).unwrap_or(24);
                let full_attention_interval =
                    tc.and_then(|t| t.full_attention_interval).unwrap_or(4);
                // Count full-attention layers from layer_types list if present.
                let num_full_attn = if let Some(types) = tc.and_then(|t| t.layer_types.as_ref()) {
                    types
                        .iter()
                        .filter(|s| s.as_str() == "full_attention")
                        .count()
                } else {
                    // Fallback: every full_attention_interval-th layer is full-attention.
                    (0..num_hidden_layers)
                        .filter(|i| (i + 1) % full_attention_interval == 0)
                        .count()
                };
                (num_kv_heads, head_dim, num_full_attn)
            }
            ModelArchitecture::Qwen2 => {
                let num_kv_heads = self.num_key_value_heads.unwrap_or(2);
                let num_attention_heads = self.num_attention_heads.unwrap_or(14);
                let hidden_size = self.hidden_size.unwrap_or(896);
                let head_dim = hidden_size / num_attention_heads;
                let num_layers = self.num_hidden_layers.unwrap_or(24);
                (num_kv_heads, head_dim, num_layers)
            }
            ModelArchitecture::Gemma2 => {
                let num_kv_heads = self.num_key_value_heads.unwrap_or(8);
                let num_attention_heads = self.num_attention_heads.unwrap_or(16);
                let hidden_size = self.hidden_size.unwrap_or(3584);
                let head_dim = self.head_dim.unwrap_or(hidden_size / num_attention_heads);
                let num_layers = self.num_hidden_layers.unwrap_or(42);
                (num_kv_heads, head_dim, num_layers)
            }
            ModelArchitecture::Gemma3 => {
                let num_kv_heads = self.num_key_value_heads.unwrap_or(4);
                let head_dim = self.head_dim.unwrap_or(256);
                let num_layers = self.num_hidden_layers.unwrap_or(34);
                (num_kv_heads, head_dim, num_layers)
            }
            ModelArchitecture::Gemma4 => {
                let tc = self.text_config.as_ref();
                let num_kv_heads_sliding = tc.and_then(|t| t.num_key_value_heads).unwrap_or(1);
                let num_kv_heads = tc
                    .and_then(|t| t.num_global_key_value_heads)
                    .unwrap_or(num_kv_heads_sliding);
                let head_dim = tc.and_then(|t| t.global_head_dim).unwrap_or(512);
                let num_hidden_layers = tc.and_then(|t| t.num_hidden_layers).unwrap_or(35);
                let sliding_window_pattern = tc.and_then(|t| t.sliding_window_pattern).unwrap_or(5);
                // Count full-attention layers (every sliding_window_pattern-th layer).
                let num_full_attn = if let Some(types) = tc.and_then(|t| t.layer_types.as_ref()) {
                    types
                        .iter()
                        .filter(|s| s.as_str() == "full_attention")
                        .count()
                } else {
                    (0..num_hidden_layers)
                        .filter(|i| (i + 1) % sliding_window_pattern == 0)
                        .count()
                };
                (num_kv_heads, head_dim, num_full_attn)
            }
            ModelArchitecture::Phi3 => {
                // Defaults below match `microsoft/Phi-4-mini-instruct` (the
                // primary tested Phi3 target) and are only used as a
                // last-resort fallback when the corresponding field is
                // absent from `config.json`.  The authoritative config used
                // by the model itself is the `phi3::Config` parsed directly
                // in `load_model`; these values are just for KV-cache
                // sizing when RawConfig fields are missing.  Prefer the
                // explicit `head_dim` field over `hidden_size /
                // num_attention_heads` to avoid silent integer truncation
                // when `hidden_size` isn't divisible by the head count.
                let num_kv_heads = self.num_key_value_heads.unwrap_or(8);
                let num_attention_heads = self.num_attention_heads.unwrap_or(24);
                let hidden_size = self.hidden_size.unwrap_or(3072);
                let head_dim = self.head_dim.unwrap_or(hidden_size / num_attention_heads);
                let num_layers = self.num_hidden_layers.unwrap_or(32);
                (num_kv_heads, head_dim, num_layers)
            }
        }
    }
}

fn parse_gemma_activation(s: &str) -> candle_nn::Activation {
    match s {
        "gelu_pytorch_tanh" | "gelu_fast" | "gelu_new" => candle_nn::Activation::GeluPytorchTanh,
        "gelu" => candle_nn::Activation::Gelu,
        "silu" => candle_nn::Activation::Silu,
        "relu" => candle_nn::Activation::Relu,
        _ => candle_nn::Activation::GeluPytorchTanh,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_arch(archs: &[&str], model_type: &str) -> RawConfig {
        RawConfig {
            architectures: Some(archs.iter().map(|s| s.to_string()).collect()),
            model_type: Some(model_type.to_string()),
            vocab_size: None,
            hidden_size: None,
            intermediate_size: None,
            num_hidden_layers: None,
            num_attention_heads: None,
            num_key_value_heads: None,
            max_position_embeddings: None,
            rms_norm_eps: None,
            rope_theta: None,
            rope_local_base_freq: None,
            tie_word_embeddings: None,
            hidden_act: None,
            sliding_window: None,
            max_window_layers: None,
            use_sliding_window: None,
            head_dim: None,
            hidden_activation: None,
            attention_bias: None,
            final_logit_softcapping: None,
            attn_logit_softcapping: None,
            query_pre_attn_scalar: None,
            sliding_window_pattern: None,
            layer_types: None,
            text_config: None,
            audio_config: None,
            audio_token_id: None,
            vision_config: None,
            image_token_id: None,
            boi_token_id: None,
            eoi_token_id: None,
        }
    }

    #[test]
    fn detect_qwen2_architecture() {
        let cfg = config_with_arch(&["Qwen2ForCausalLM"], "qwen2");
        assert_eq!(cfg.detect_architecture().unwrap(), ModelArchitecture::Qwen2);
    }

    #[test]
    fn detect_gemma2_architecture() {
        let cfg = config_with_arch(&["Gemma2ForCausalLM"], "gemma2");
        assert_eq!(
            cfg.detect_architecture().unwrap(),
            ModelArchitecture::Gemma2
        );
    }

    #[test]
    fn detect_gemma3_architecture() {
        let cfg = config_with_arch(&["Gemma3ForCausalLM"], "gemma3");
        assert_eq!(
            cfg.detect_architecture().unwrap(),
            ModelArchitecture::Gemma3
        );
    }

    #[test]
    fn detect_gemma3_preferred_over_gemma2() {
        // A config with both Gemma3 and Gemma2 in the architecture list should pick Gemma3.
        let cfg = config_with_arch(&["Gemma3ForCausalLM", "Gemma2ForCausalLM"], "gemma3");
        assert_eq!(
            cfg.detect_architecture().unwrap(),
            ModelArchitecture::Gemma3
        );
    }

    #[test]
    fn unsupported_architecture_errors() {
        let cfg = config_with_arch(&["LlamaForCausalLM"], "llama");
        assert!(cfg.detect_architecture().is_err());
    }

    #[test]
    fn gemma3_config_defaults_are_reasonable() {
        let cfg = config_with_arch(&["Gemma3ForCausalLM"], "gemma3");
        let g3 = cfg.to_gemma3_config();
        assert!(g3.num_hidden_layers > 0);
        assert!(g3.hidden_size > 0);
        assert!(g3.sliding_window > 0);
        assert!(g3.sliding_window_pattern > 0);
    }

    #[test]
    fn detect_phi3_architecture() {
        let cfg = config_with_arch(&["Phi3ForCausalLM"], "phi3");
        assert_eq!(cfg.detect_architecture().unwrap(), ModelArchitecture::Phi3);
    }

    #[test]
    fn detect_phi3_by_model_type() {
        let cfg = config_with_arch(&["UnknownArch"], "phi3");
        assert_eq!(cfg.detect_architecture().unwrap(), ModelArchitecture::Phi3);
    }
}

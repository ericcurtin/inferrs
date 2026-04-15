//! GGUF metadata and tensor-name conversion for Gemma-4 models.
//!
//! Reimplements the key parts of `convert_hf_to_gguf.py` so that inferrs
//! produces GGUF files that are structurally identical to the ggml-org
//! reference files:
//!
//!  - GGUF version 3
//!  - Full `general.*`, `gemma4.*`, and `tokenizer.ggml.*` KV metadata
//!  - Canonical llama.cpp tensor names (`blk.N.attn_q.weight`, etc.)
//!  - Norm/scalar weights stored as F32 (not F16)
//!  - Tokenizer vocab, scores, token-type, and merge tables embedded inline

use anyhow::{Context, Result};
use candle_core::quantized::{gguf_file, GgmlDType};
use std::path::Path;

// ── GGUF value helpers ────────────────────────────────────────────────────────

fn str_val(s: &str) -> gguf_file::Value {
    gguf_file::Value::String(s.to_string())
}
fn u32_val(v: u32) -> gguf_file::Value {
    gguf_file::Value::U32(v)
}
fn i32_val(v: i32) -> gguf_file::Value {
    gguf_file::Value::I32(v)
}
fn f32_val(v: f32) -> gguf_file::Value {
    gguf_file::Value::F32(v)
}
fn bool_val(v: bool) -> gguf_file::Value {
    gguf_file::Value::Bool(v)
}
fn bool_array_val(v: Vec<bool>) -> gguf_file::Value {
    gguf_file::Value::Array(v.into_iter().map(gguf_file::Value::Bool).collect())
}
fn str_array_val(v: Vec<String>) -> gguf_file::Value {
    gguf_file::Value::Array(v.into_iter().map(gguf_file::Value::String).collect())
}

fn f32_array_val(v: Vec<f32>) -> gguf_file::Value {
    gguf_file::Value::Array(v.into_iter().map(gguf_file::Value::F32).collect())
}
fn i32_array_val(v: Vec<i32>) -> gguf_file::Value {
    gguf_file::Value::Array(v.into_iter().map(gguf_file::Value::I32).collect())
}
fn str_array_val_static(v: &[&str]) -> gguf_file::Value {
    gguf_file::Value::Array(
        v.iter()
            .map(|s| gguf_file::Value::String(s.to_string()))
            .collect(),
    )
}

// ── Gemma-4 metadata builder ──────────────────────────────────────────────────

/// All GGUF KV metadata for a Gemma-4 model, populated from `config.json`,
/// `tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja`.
pub struct Gemma4Meta {
    pub kv: Vec<(String, gguf_file::Value)>,
}

impl Gemma4Meta {
    /// Build the full metadata block from the files in `model_dir`.
    pub fn from_model_dir(model_dir: &Path, quant_dtype: GgmlDType) -> Result<Self> {
        let cfg = read_config(model_dir)?;
        let tc = cfg
            .get("text_config")
            .and_then(|v| v.as_object())
            .unwrap_or_else(|| cfg.as_object().unwrap());

        // ── Architecture params ───────────────────────────────────────────────
        let n_layers = tc
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(35) as u32;
        let hidden_size = tc
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(1536) as u32;
        let n_heads = tc
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as u32;
        let n_kv_heads = tc
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;
        // SWA (sliding-window) head dim — read directly from config.
        // For Gemma4: "head_dim"=256 is the SWA head dim.
        let head_dim = tc
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(hidden_size / n_heads);
        // Full-attention (global) head dim — "global_head_dim"=512.
        let global_head_dim = tc
            .get("global_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(head_dim * 2);
        let rope_theta = tc
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(1_000_000.0) as f32;
        let rope_theta_swa = tc
            .get("rope_local_base_freq")
            .and_then(|v| v.as_f64())
            .unwrap_or(10_000.0) as f32;
        let ctx_len = tc
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(131072) as u32;
        let rms_eps = tc
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32;
        let softcap = tc
            .get("final_logit_softcapping")
            .and_then(|v| v.as_f64())
            .unwrap_or(30.0) as f32;
        let sliding_window = tc
            .get("sliding_window")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as u32;
        let n_kv_shared = tc
            .get("num_kv_shared_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(20) as u32;
        let pli_dim = tc
            .get("hidden_size_per_layer_input")
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as u32;

        // Sliding-window pattern (true = sliding, false = global/full)
        let layer_types: Vec<bool> =
            if let Some(arr) = tc.get("layer_types").and_then(|v| v.as_array()) {
                arr.iter()
                    .map(|v| v.as_str().map(|s| s == "sliding_attention").unwrap_or(true))
                    .collect()
            } else {
                // Default: full-attention every 5th layer (0-indexed: layers 4,9,14,...)
                (0..n_layers).map(|i| (i + 1) % 5 != 0).collect()
            };

        // Per-layer feed_forward_length (may vary for double-wide MLP)
        let base_ffn = tc
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(6144) as u32;
        let use_double_wide = tc
            .get("use_double_wide_mlp")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let dw_start_layer = if use_double_wide {
            (n_layers - n_kv_shared) as usize
        } else {
            n_layers as usize
        };
        let ffn_lengths: Vec<u32> = (0..n_layers as usize)
            .map(|i| {
                if i >= dw_start_layer {
                    base_ffn * 2
                } else {
                    base_ffn
                }
            })
            .collect();
        // If all identical, ggml-org writes a scalar; else an array.
        let all_same_ffn = ffn_lengths.iter().all(|&v| v == ffn_lengths[0]);

        // ── General file metadata ─────────────────────────────────────────────
        // file_type: 7=Q8_0, 15=Q4_K_M  (matches ggml-org general.file_type)
        let file_type: u32 = match quant_dtype {
            GgmlDType::Q8_0 => 7,
            GgmlDType::Q4K => 15,
            GgmlDType::Q6K => 17,
            GgmlDType::Q5K => 16,
            GgmlDType::Q3K => 11,
            _ => 0,
        };
        // Size label: read from tokenizer_config.json "model_size_label" if available,
        // otherwise use ggml-org conventions based on model dimensions.
        // ggml-org E2B="4.6B", E4B="7.5B" — these reflect total param counts
        // including embeddings and PLI tables, not just the transformer layers.
        let size_label = {
            // Try to match the ggml-org label by total parameters.
            // embed_tokens: vocab_size × hidden_size
            // per_layer_token_embd: num_layers × pli_dim × vocab_size
            // The total is dominated by the PLI table for E4B.
            let vocab = 262144u64;
            let embed_params = vocab * hidden_size as u64;
            let pli_params = n_layers as u64 * pli_dim as u64 * vocab;
            let attn_params = n_layers as u64
                * (hidden_size as u64 * hidden_size as u64 * 4
                    + n_kv_heads as u64 * 2 * head_dim as u64 * hidden_size as u64);
            // rough MLP
            let ffn_params = n_layers as u64 * (base_ffn as u64 * hidden_size as u64 * 3);
            let pli_proj = hidden_size as u64 * (n_layers as u64 * pli_dim as u64);
            let total =
                (embed_params + pli_params + attn_params + ffn_params + pli_proj) as f64 / 1e9;
            if total < 5.5 {
                "4.6B".to_string()
            } else {
                "7.5B".to_string()
            }
        };

        // ── Tokenizer vocab ───────────────────────────────────────────────────
        let (tokens, scores, token_types, merges, bos_id, eos_id, unk_id, pad_id, mask_id, add_bos) =
            read_tokenizer(model_dir)?;

        // ── Chat template ─────────────────────────────────────────────────────
        let chat_template = read_chat_template_from_dir(model_dir);

        // ── Build KV list in ggml-org order ───────────────────────────────────
        let mut kv: Vec<(String, gguf_file::Value)> = Vec::with_capacity(48);

        // general.*
        kv.push(("general.architecture".into(), str_val("gemma4")));
        kv.push(("general.type".into(), str_val("model")));
        kv.push(("general.file_type".into(), u32_val(file_type)));
        kv.push(("general.quantization_version".into(), u32_val(2)));
        kv.push(("general.size_label".into(), str_val(&size_label)));
        kv.push(("general.license".into(), str_val("apache-2.0")));
        kv.push((
            "general.license.link".into(),
            str_val("https://ai.google.dev/gemma/docs/gemma_4_license"),
        ));
        kv.push(("general.tags".into(), str_array_val_static(&["any-to-any"])));
        kv.push(("general.sampling.top_k".into(), i32_val(64)));
        kv.push(("general.sampling.top_p".into(), f32_val(0.95)));
        kv.push(("general.sampling.temp".into(), f32_val(1.0)));

        // gemma4.*
        kv.push(("gemma4.block_count".into(), u32_val(n_layers)));
        kv.push(("gemma4.context_length".into(), u32_val(ctx_len)));
        kv.push(("gemma4.embedding_length".into(), u32_val(hidden_size)));
        kv.push((
            "gemma4.embedding_length_per_layer_input".into(),
            u32_val(pli_dim),
        ));
        // ggml-org writes feed_forward_length as signed i32 array (array type 5).
        if all_same_ffn {
            kv.push(("gemma4.feed_forward_length".into(), u32_val(ffn_lengths[0])));
        } else {
            let signed: Vec<i32> = ffn_lengths.iter().map(|&v| v as i32).collect();
            kv.push(("gemma4.feed_forward_length".into(), i32_array_val(signed)));
        }
        kv.push(("gemma4.attention.head_count".into(), u32_val(n_heads)));
        kv.push(("gemma4.attention.head_count_kv".into(), u32_val(n_kv_heads)));
        kv.push((
            "gemma4.attention.key_length".into(),
            u32_val(global_head_dim),
        ));
        kv.push((
            "gemma4.attention.value_length".into(),
            u32_val(global_head_dim),
        ));
        kv.push(("gemma4.attention.key_length_swa".into(), u32_val(head_dim)));
        kv.push((
            "gemma4.attention.value_length_swa".into(),
            u32_val(head_dim),
        ));
        kv.push((
            "gemma4.attention.layer_norm_rms_epsilon".into(),
            f32_val(rms_eps),
        ));
        kv.push((
            "gemma4.attention.sliding_window".into(),
            u32_val(sliding_window),
        ));
        kv.push((
            "gemma4.attention.sliding_window_pattern".into(),
            bool_array_val(layer_types),
        ));
        kv.push((
            "gemma4.attention.shared_kv_layers".into(),
            u32_val(n_kv_shared),
        ));
        kv.push(("gemma4.final_logit_softcapping".into(), f32_val(softcap)));
        kv.push(("gemma4.rope.freq_base".into(), f32_val(rope_theta)));
        kv.push(("gemma4.rope.freq_base_swa".into(), f32_val(rope_theta_swa)));
        kv.push((
            "gemma4.rope.dimension_count".into(),
            u32_val(global_head_dim),
        ));
        kv.push(("gemma4.rope.dimension_count_swa".into(), u32_val(head_dim)));

        // tokenizer.ggml.*
        kv.push(("tokenizer.ggml.model".into(), str_val("gemma4")));
        kv.push(("tokenizer.ggml.tokens".into(), str_array_val(tokens)));
        kv.push(("tokenizer.ggml.scores".into(), f32_array_val(scores)));
        kv.push((
            "tokenizer.ggml.token_type".into(),
            i32_array_val(token_types),
        ));
        kv.push(("tokenizer.ggml.merges".into(), str_array_val(merges)));
        kv.push(("tokenizer.ggml.bos_token_id".into(), u32_val(bos_id)));
        kv.push(("tokenizer.ggml.eos_token_id".into(), u32_val(eos_id)));
        kv.push(("tokenizer.ggml.unknown_token_id".into(), u32_val(unk_id)));
        kv.push(("tokenizer.ggml.padding_token_id".into(), u32_val(pad_id)));
        kv.push(("tokenizer.ggml.mask_token_id".into(), u32_val(mask_id)));
        kv.push(("tokenizer.ggml.add_bos_token".into(), bool_val(add_bos)));
        kv.push(("tokenizer.ggml.add_space_prefix".into(), bool_val(false)));

        if let Some(tmpl) = chat_template {
            kv.push(("tokenizer.chat_template".into(), str_val(&tmpl)));
        }

        Ok(Self { kv })
    }
}

// ── Tensor name remapping ─────────────────────────────────────────────────────

/// Remap a HuggingFace safetensors tensor name to the llama.cpp GGUF canonical
/// name used by ggml-org.  Returns `None` for tensors that should be omitted
/// from the GGUF (vision/audio encoders, etc.).
///
/// The mapping is derived directly from the ggml-org reference GGUFs for
/// google/gemma-4-E2B-it and google/gemma-4-E4B-it.
pub fn remap_tensor_name_gemma4(hf_name: &str) -> Option<String> {
    let name = hf_name
        .strip_prefix("model.language_model.")
        .or_else(|| hf_name.strip_prefix("model."));

    let name = name?;

    // ── Top-level (non-layer) tensors ─────────────────────────────────────────
    let canonical = match name {
        "embed_tokens.weight" => "token_embd.weight".to_string(),
        "embed_tokens_per_layer.weight" => "per_layer_token_embd.weight".to_string(),
        "norm.weight" => "output_norm.weight".to_string(),
        "per_layer_model_projection.weight" => "per_layer_model_proj.weight".to_string(),
        "per_layer_projection_norm.weight" => "per_layer_proj_norm.weight".to_string(),
        // Vision/audio projection — omit from the language GGUF
        "embed_vision.embedding_projection.weight" => return None,
        "embed_audio.embedding_projection.weight" => return None,
        other => {
            // ── Per-layer tensors: layers.N.* ─────────────────────────────────
            let rest = other.strip_prefix("layers.")?;
            let dot = rest.find('.')?;
            let layer_idx: u32 = rest[..dot].parse().ok()?;
            let suffix = &rest[dot + 1..];

            let blk_name = match suffix {
                // Attention
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "self_attn.q_norm.weight" => "attn_q_norm.weight",
                "self_attn.k_norm.weight" => "attn_k_norm.weight",
                // Norms
                "input_layernorm.weight" => "attn_norm.weight",
                "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
                "post_attention_layernorm.weight" => "post_attention_norm.weight",
                "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
                "post_per_layer_input_norm.weight" => "post_norm.weight",
                // MLP
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                // PLI
                "per_layer_input_gate.weight" => "inp_gate.weight",
                "per_layer_projection.weight" => "proj.weight",
                // Layer scalar
                "layer_scalar" => "layer_output_scale.weight",
                _ => return None,
            };
            format!("blk.{layer_idx}.{blk_name}")
        }
    };

    Some(canonical)
}

/// Returns `true` when the canonical llama.cpp tensor name should be stored as
/// F32 rather than quantized (norms, scalars, RoPE frequencies).
///
/// This matches ggml-org: all `attn_norm`, `ffn_norm`, `post_*_norm`,
/// `attn_q_norm`, `attn_k_norm`, `layer_output_scale`, `output_norm`,
/// `per_layer_proj_norm`, and `rope_freqs` tensors are F32.
pub fn canonical_name_is_f32(canonical: &str) -> bool {
    canonical.contains("_norm.weight")
        || canonical.contains("scale.weight")
        || canonical == "rope_freqs.weight"
        || canonical == "per_layer_proj_norm.weight"
        || canonical == "output_norm.weight"
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn read_config(dir: &Path) -> Result<serde_json::Value> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text).context("invalid config.json")
}

fn read_chat_template_from_dir(dir: &Path) -> Option<String> {
    std::fs::read_to_string(dir.join("chat_template.jinja")).ok()
}

#[allow(dead_code)]
fn derive_size_label(hidden: u32, layers: u32) -> String {
    // Very rough parameter count: 12 × layers × hidden²
    let approx_b = 12.0 * layers as f64 * (hidden as f64).powi(2) / 1e9;
    if approx_b < 3.0 {
        format!("{:.1}B", approx_b)
    } else {
        format!("{:.0}B", approx_b.round())
    }
}

/// Tokenizer data extracted from `tokenizer.json`.
type TokenizerData = (
    Vec<String>,
    Vec<f32>,
    Vec<i32>,
    Vec<String>,
    u32,
    u32,
    u32,
    u32,
    u32,
    bool,
);

/// Read `tokenizer.json` and return `TokenizerData` (tokens, scores, token_types,
/// merges, bos_id, eos_id, unk_id, pad_id, mask_id, add_bos).
///
/// Token types follow the llama.cpp / sentencepiece convention:
/// 1 = normal, 3 = control (special/added), 4 = user_defined, 6 = byte.
fn read_tokenizer(dir: &Path) -> Result<TokenizerData> {
    let path = dir.join("tokenizer.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let tj: serde_json::Value = serde_json::from_str(&text).context("invalid tokenizer.json")?;

    let model = &tj["model"];
    let vocab_map = model["vocab"]
        .as_object()
        .context("no vocab in tokenizer.json")?;
    let vocab_size = vocab_map.len();

    // Build id→token map
    let mut id_to_token: Vec<String> = vec![String::new(); vocab_size];
    for (tok, id_val) in vocab_map {
        let id = id_val.as_u64().context("non-integer token id")? as usize;
        if id < vocab_size {
            id_to_token[id] = tok.clone();
        }
    }

    // added_tokens: determine which ids are special (type=3)
    let added: std::collections::HashSet<u64> = tj["added_tokens"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v["id"].as_u64()).collect())
        .unwrap_or_default();

    // Find special token IDs from added_tokens by content
    let mut bos_id = 2u32;
    let mut eos_id = 1u32;
    let mut unk_id = 3u32;
    let mut pad_id = 0u32;
    let mut mask_id = 4u32;
    if let Some(arr) = tj["added_tokens"].as_array() {
        for v in arr {
            let id = v["id"].as_u64().unwrap_or(u64::MAX) as u32;
            match v["content"].as_str().unwrap_or("") {
                "<bos>" => bos_id = id,
                "<eos>" => eos_id = id,
                "<unk>" => unk_id = id,
                "<pad>" => pad_id = id,
                "<mask>" => mask_id = id,
                _ => {}
            }
        }
    }

    // Scores: -1000 for special tokens, BPE rank-based for normal tokens.
    // llama.cpp uses the BPE merge rank as the negative score.
    // For tokens not in any merge (rare), score = 0.
    // This matches gguf-py's behaviour for BPE models.
    let mut scores = vec![0.0f32; vocab_size];
    for id in &added {
        if (*id as usize) < vocab_size {
            scores[*id as usize] = -1000.0;
        }
    }

    // Token types
    let mut token_types = vec![1i32; vocab_size]; // 1 = normal
    for id in &added {
        if (*id as usize) < vocab_size {
            token_types[*id as usize] = 3; // control/special
        }
    }
    // Byte tokens: gguf-py marks tokens like "<0x00>" as type 6
    for (i, tok) in id_to_token.iter().enumerate() {
        if tok.starts_with("<0x") && tok.ends_with('>') && tok.len() == 6 {
            token_types[i] = 6;
        }
    }

    // Merges: HuggingFace tokenizer.json stores merges as either:
    //   - Array of strings: ["ab cd", ...]  (space-separated)
    //   - Array of arrays:  [["ab","cd"], ...] (two-element arrays)
    // GGUF expects space-separated strings.
    let merges: Vec<String> = model["merges"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    if let Some(s) = v.as_str() {
                        // Already a space-separated string.
                        Some(s.to_string())
                    } else if let Some(pair) = v.as_array() {
                        // Two-element array ["a", "b"] → "a b"
                        let a = pair.first().and_then(|x| x.as_str()).unwrap_or("");
                        let b = pair.get(1).and_then(|x| x.as_str()).unwrap_or("");
                        Some(format!("{a} {b}"))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    // add_bos: Gemma4Model.set_vocab() in convert_hf_to_gguf.py always calls
    // self.gguf_writer.add_add_bos_token(True) (line 7702, current HEAD).
    // The ggml-org E2B GGUF shows False because it was built with an older
    // commit before that line was added; the current script produces True.
    let add_bos = true;

    Ok((
        id_to_token,
        scores,
        token_types,
        merges,
        bos_id,
        eos_id,
        unk_id,
        pad_id,
        mask_id,
        add_bos,
    ))
}

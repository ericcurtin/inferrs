//! Weight quantization: converts safetensors model weights to a GGUF file on disk.
//!
//! ## Usage
//!
//! ```text
//! inferrs serve google/gemma-4-E2B-it --quantize          # uses model-default recipe
//! inferrs serve google/gemma-4-E2B-it --quantize=Q8_0     # explicit override
//! ```
//!
//! On first invocation the weights are read from the HuggingFace cache, quantized
//! on the CPU, and written to a `.gguf` file next to the safetensors shards.
//! Subsequent invocations find the file already present and skip the conversion.
//!
//! ## Per-tensor policy
//!
//! Tensors whose names contain any of the "keep" substrings (norm, bias, rope,
//! layer_scalar) are stored at F16.  All other tensors are quantized according to
//! either the user-supplied format or the model-specific recipe below.
//!
//! ## Model-specific recipes
//!
//! Several models have a known-good per-tensor recipe derived by inspecting the
//! ggml-org reference GGUFs.  When no explicit `--quantize` format is supplied,
//! the recipe is used automatically; otherwise the user's format is applied to
//! every tensor (overriding the recipe).
//!
//! | Model                       | Recipe      | Key strategy                         |
//! |-----------------------------|-------------|--------------------------------------|
//! | google/gemma-4-E2B-it       | Q8_0        | All weights Q8_0 (near-lossless)     |
//! | google/gemma-4-E4B-it       | Q4K_M       | Layer weights Q4K; embeddings Q6K    |
//! | google/gemma-4-26B-A4B-it   | Q4K_M       | Experts Q4K; attn Q8_0; embeddings Q6K |
//! | google/gemma-4-31B-it       | Q4K_M       | Layer weights Q4K; embeddings Q6K    |
//!
//! ## Supported formats
//!
//! Candle 0.8 `GgmlDType` variants, matched case-insensitively:
//!
//! | String     | Meaning                              |
//! |------------|--------------------------------------|
//! | Q4_0       | 4-bit, block=32, delta per block     |
//! | Q4_1       | 4-bit, block=32, delta+min per block |
//! | Q5_0       | 5-bit, block=32                      |
//! | Q5_1       | 5-bit, block=32, +min                |
//! | Q8_0       | 8-bit, block=32 (fast, near-lossless)|
//! | Q2K        | 2-bit k-quant                        |
//! | Q3K        | 3-bit k-quant                        |
//! | Q4K / Q4_K_M | 4-bit k-quant (default)            |
//! | Q5K        | 5-bit k-quant                        |
//! | Q6K        | 6-bit k-quant                        |

use anyhow::{Context, Result};
use candle_core::{quantized::GgmlDType, DType, Device};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use crate::gguf_meta::{canonical_name_is_f32, remap_tensor_name_gemma4, Gemma4Meta};

// ── Format parsing ────────────────────────────────────────────────────────────

/// Parse a user-supplied quantization format string into a [`GgmlDType`].
///
/// Matching is case-insensitive and accepts several aliases:
/// `Q4K` / `Q4_K` / `Q4_K_M` are all treated as [`GgmlDType::Q4K`].
pub fn parse_format(s: &str) -> Result<GgmlDType> {
    match s.to_uppercase().replace('-', "_").as_str() {
        "Q4_0" => Ok(GgmlDType::Q4_0),
        "Q4_1" => Ok(GgmlDType::Q4_1),
        "Q5_0" => Ok(GgmlDType::Q5_0),
        "Q5_1" => Ok(GgmlDType::Q5_1),
        "Q8_0" => Ok(GgmlDType::Q8_0),
        "Q2K" | "Q2_K" | "Q2_K_S" => Ok(GgmlDType::Q2K),
        "Q3K" | "Q3_K" | "Q3_K_M" | "Q3_K_S" | "Q3_K_L" => Ok(GgmlDType::Q3K),
        // Q4K is the default and is advertised as "Q4_K_M" to match llama.cpp naming.
        "Q4K" | "Q4_K" | "Q4_K_M" | "Q4_K_S" => Ok(GgmlDType::Q4K),
        "Q5K" | "Q5_K" | "Q5_K_M" | "Q5_K_S" => Ok(GgmlDType::Q5K),
        "Q6K" | "Q6_K" => Ok(GgmlDType::Q6K),
        "TQ2_0" | "TQ2" => Ok(GgmlDType::TQ2_0),
        other => anyhow::bail!(
            "Unknown quantization format {:?}. \
             Accepted: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2K, Q3K, Q4K (default/Q4_K_M), Q5K, Q6K.",
            other
        ),
    }
}

// ── Model-specific quantization recipes ─────────────────────────────────────

/// A per-tensor quantization recipe for a specific model.
///
/// When inferrs quantizes a model it calls [`Recipe::dtype_for`] for each
/// tensor name to determine the target format.  Recipes are derived from the
/// ggml-org reference GGUFs so that inferrs produces quantizations that are
/// byte-compatible in strategy (if not in tensor naming) with those files.
pub struct Recipe {
    /// Human-readable label used in log messages.
    pub label: &'static str,
    /// The "headline" dtype advertised in the GGUF filename / `--quantize` flag.
    pub default_dtype: GgmlDType,
    /// Per-tensor classifier.  Returns the dtype to use for the tensor with the
    /// given name.  `None` means "keep at F16 (do not quantize)".
    /// Per-tensor classifier.  `n_layers` is the total number of transformer
    /// blocks (from config.json) needed for per-layer decisions like Q4_K_M.
    pub classifier: fn(name: &str, default: GgmlDType, n_layers: usize) -> Option<GgmlDType>,
}

impl Recipe {
    /// Choose the quantization dtype for `tensor_name`, or `None` to keep unquantized.
    #[allow(dead_code)]
    pub fn dtype_for(&self, tensor_name: &str, n_layers: usize) -> Option<GgmlDType> {
        (self.classifier)(tensor_name, self.default_dtype, n_layers)
    }
}

// ── Shared helpers used by multiple classifiers ───────────────────────────────

/// Substrings whose tensors should NOT be quantized (stored as F16 or F32).
const KEEP_UNQUANTIZED: &[&str] = &["norm", ".bias", "rope", "layer_scalar"];

/// Returns `true` when the tensor should be kept unquantized regardless of recipe.
fn is_f16_only(name: &str) -> bool {
    KEEP_UNQUANTIZED.iter().any(|&s| name.contains(s))
}

/// Returns `true` for embedding tables and tied output projections.
/// These are quantized to Q6K rather than the layer default for accuracy:
/// the lm_head GEMV is the hottest operation per decode step (~80% of memory
/// bandwidth) and Q6K keeps vocabulary logits within 0.1% of F16 quality.
fn is_embedding(name: &str) -> bool {
    name.contains("embed_tokens") || name.contains("lm_head")
}

// ── Per-model classifiers ─────────────────────────────────────────────────────

/// **E2B classifier** — Q8_0 for language model weights (near-lossless).
///
/// `per_layer_model_projection` is stored as F16 (matches ggml-org exactly).
/// Vision/audio encoder weights are excluded entirely.
fn classify_e2b(name: &str, _default: GgmlDType, _n_layers: usize) -> Option<GgmlDType> {
    if is_f16_only(name) {
        return None; // → F32 for Gemma4, F16 for others (handled in convert_to_gguf)
    }
    // per_layer_model_projection → F16 (ggml-org stores this at F16 for E2B)
    if name.contains("per_layer_model_projection") {
        return None; // F16 (Gemma4 path keeps F32 for None, but see below)
    }
    Some(GgmlDType::Q8_0)
}

/// Returns `true` when a tensor should be **omitted from the GGUF** entirely
/// and loaded from safetensors on demand instead.
///
/// This applies to multimodal encoder weights (vision, audio) which:
///  1. Are not on the text-decode critical path.
///  2. Are lazy-loaded by the engine; if absent from the GGUF the engine
///     falls back to the original safetensors file.
///  3. Would add ~475 MB to the GGUF for E2B if quantized, bloating the
///     file size beyond the reference ggml-org GGUF (~4.6 GB).
fn should_skip_in_gguf(name: &str) -> bool {
    name.contains("vision_tower") || name.contains("audio_tower")
}

/// **E4B / 31B classifier** — Q4K for most weights, Q6K for embeddings and
/// quality-critical tensors (attn_v, ffn_down) to match ggml-org's Q4_K_M recipe.
///
/// Matches the ggml-org `Q4_K_M` recipe:
/// - `embed_tokens` (token embeddings / tied lm_head): Q6K  
/// - `embed_tokens_per_layer` (PLI table, ~4 GB for E4B): Q6K  
///   This is the dominant bandwidth tensor in E4B; Q6K adds ~50% size vs Q4K
///   but keeps PLI quality high and matches ggml-org exactly.
/// - All other layer weights (attention Q/K/V/O, MLP gate/up/down): Q4K
fn classify_q4k_m(name: &str, _default: GgmlDType, n_layers: usize) -> Option<GgmlDType> {
    if is_f16_only(name) {
        return None;
    }
    if is_embedding(name) {
        return Some(GgmlDType::Q6K);
    }
    // For attn_v and ffn_down, apply llama.cpp's `use_more_bits` heuristic from
    // llama-quant.cpp (Q4_K_M case):
    //   use_more_bits(i, n) = i < n/8  OR  i >= 7*n/8  OR  (i - n/8) % 3 == 2
    //
    // The layer index is parsed from the HuggingFace tensor name
    // "model.language_model.layers.N.{self_attn.v_proj,mlp.down_proj}.weight".
    let is_attn_v = name.contains("self_attn.v_proj");
    let is_ffn_down = name.contains("mlp.down_proj");
    if is_attn_v || is_ffn_down {
        if let Some(layer_idx) = parse_layer_index(name) {
            // n_layers is not known here; use a sentinel of 42 (E4B) and fall
            // back to "all Q6K" for unknown sizes. The recipe is only invoked
            // for models where this classifier is registered, so the layer count
            // is implicitly known; we encode the rule parametrically and the
            // caller passes the total layer count via the recipe's `use_more_bits`
            // closure.  Since we don't have n_layers here, we rely on the fact
            // that both registered models (E4B: 42, E31B: ?) both benefit from
            // Q6K for the same subset. We detect n_layers from the name itself
            // (not available) so we conservatively check up to n=42.
            //
            // To match ggml-org exactly we need the true n_layers. We infer it
            // from the model in the recipe table via a separate helper.
            // For now use Q6K for the same boolean pattern as n=42.
            // Use the actual layer count; default to 42 (E4B) if 0 was passed.
            let nl = if n_layers == 0 { 42 } else { n_layers };
            if use_more_bits(layer_idx, nl) {
                return Some(GgmlDType::Q6K);
            }
            return Some(GgmlDType::Q4K);
        }
        // If we can't parse the layer index, default to Q4K.
        return Some(GgmlDType::Q4K);
    }
    Some(GgmlDType::Q4K)
}

/// Returns true when llama.cpp Q4_K_M would bump this layer to higher precision.
/// Mirrors llama-quant.cpp:  i < n/8 || i >= 7*n/8 || (i - n/8) % 3 == 2
fn use_more_bits(i_layer: usize, n_layers: usize) -> bool {
    i_layer < n_layers / 8
        || i_layer >= 7 * n_layers / 8
        || (i_layer.saturating_sub(n_layers / 8)) % 3 == 2
}

/// Parse the layer index from a HuggingFace tensor name like
/// "model.language_model.layers.N.suffix".
fn parse_layer_index(name: &str) -> Option<usize> {
    let after = name
        .strip_prefix("model.language_model.layers.")
        .or_else(|| name.strip_prefix("model.layers."))?;
    let dot = after.find('.')?;
    after[..dot].parse().ok()
}

/// **26B MoE classifier** — Q4K for experts, Q8_0 for attention, Q6K for embed.
///
/// Derived from the ggml-org `Q4_K_M` GGUF for gemma-4-26B-A4B which uses a
/// three-tier mixed strategy:
///
/// | Tensor kind                          | dtype | Rationale                    |
/// |--------------------------------------|-------|------------------------------|
/// | embed_tokens, lm_head                | Q6K   | Hot path; accuracy-critical  |
/// | Self-attention q/k/v/o projections   | Q8_0  | Small, high-quality needed   |
/// | MoE expert gate/up/down projections  | Q4K   | Bulk of params; bandwidth-bound |
/// | Shared-expert projections            | Q8_0  | Always-active; high quality  |
/// | Everything else                      | Q4K   | Default                      |
fn classify_26b_moe(name: &str, _default: GgmlDType, _n_layers: usize) -> Option<GgmlDType> {
    if is_f16_only(name) {
        return None;
    }
    if is_embedding(name) {
        return Some(GgmlDType::Q6K);
    }
    // Attention projections (q_proj, k_proj, v_proj, o_proj) → Q8_0.
    // These are small (hidden×head_dim each) but always active and
    // quality-sensitive; Q8_0 keeps them near-lossless.
    if name.contains(".self_attn.") {
        return Some(GgmlDType::Q8_0);
    }
    // Shared expert MLP (always active, like a dense layer) → Q8_0.
    if name.contains("shared_expert") {
        return Some(GgmlDType::Q8_0);
    }
    // MoE routed expert MLP weights → Q4K (bandwidth-bound, sparsely active).
    // Everything else (cross-layer projections, PLI, etc.) → Q4K.
    Some(GgmlDType::Q4K)
}

// ── Recipe registry ──────────────────────────────────────────────────────────

/// Per-model quantization recipes, keyed by HuggingFace model ID.
///
/// | Model ID                   | Recipe label |
/// |----------------------------|--------------|
/// | google/gemma-4-E2B-it      | Q8_0_full    |
/// | google/gemma-4-E4B-it      | Q4K_M        |
/// | google/gemma-4-26B-A4B-it  | Q4K_M_moe    |
/// | google/gemma-4-31B-it      | Q4K_M        |
static RECIPES: &[(&str, Recipe)] = &[
    (
        "google/gemma-4-e2b-it",
        Recipe {
            label: "Q8_0_full",
            default_dtype: GgmlDType::Q8_0,
            classifier: classify_e2b,
        },
    ),
    (
        "google/gemma-4-e4b-it",
        Recipe {
            label: "Q4K_M",
            default_dtype: GgmlDType::Q4K,
            classifier: classify_q4k_m,
        },
    ),
    (
        "google/gemma-4-26b-a4b-it",
        Recipe {
            label: "Q4K_M_moe",
            default_dtype: GgmlDType::Q4K,
            classifier: classify_26b_moe,
        },
    ),
    (
        "google/gemma-4-31b-it",
        Recipe {
            label: "Q4K_M",
            default_dtype: GgmlDType::Q4K,
            classifier: classify_q4k_m,
        },
    ),
];

/// Look up a model-specific quantization recipe by HuggingFace model ID.
///
/// Matching is case-insensitive.  The `google/` prefix is optional — both
/// `"google/gemma-4-E2B-it"` and `"gemma-4-E2B-it"` match the E2B recipe.
pub fn recipe_for(model_id: &str) -> Option<&'static Recipe> {
    let needle = model_id.to_lowercase();
    // Try exact match first (with org prefix), then strip the prefix.
    for (key, recipe) in RECIPES {
        if needle == *key {
            return Some(recipe);
        }
        // Also match without the leading "org/" segment.
        // Also match the bare model name without an org prefix, anchored at a
        // path-separator boundary so "myorg/custom-prefix-gemma-4-e4b-it" does
        // NOT accidentally match the "google/gemma-4-e4b-it" recipe.
        if let Some(short) = key.split_once('/').map(|(_, m)| m) {
            if needle == short || needle.ends_with(&format!("/{short}")) {
                return Some(recipe);
            }
        }
    }
    None
}

// ── Per-tensor policy (generic, used when no recipe matches) ──────────────────

/// Generic per-tensor classifier used when no model-specific recipe is
/// registered.  Matches the historical inferrs behaviour:
/// - Normalisation / bias / RoPE / layer_scalar → F16
/// - embed_tokens / lm_head → Q6K
/// - Everything else → `quant_dtype`
fn classify_generic(name: &str, quant_dtype: GgmlDType, _n_layers: usize) -> Option<GgmlDType> {
    if is_f16_only(name) {
        return None;
    }
    if is_embedding(name) {
        return Some(GgmlDType::Q6K);
    }
    Some(quant_dtype)
}

// Keep old public helpers for any callers outside this module.
/// Return true when a tensor should be kept at F16 (not quantized).
#[allow(dead_code)]
pub fn should_keep_f16(name: &str) -> bool {
    is_f16_only(name)
}

/// Return true when a tensor should be quantized to Q6K.
#[allow(dead_code)]
pub fn should_use_q6k(name: &str) -> bool {
    is_embedding(name)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify every registered model resolves to the expected headline dtype,
    /// and that key tensors get the right quantization.
    #[test]
    fn recipe_lookup() {
        let cases: &[(&str, GgmlDType, &[(&str, Option<GgmlDType>)])] = &[
            (
                "google/gemma-4-E2B-it",
                GgmlDType::Q8_0,
                &[
                    ("model.layers.0.mlp.gate_proj.weight", Some(GgmlDType::Q8_0)),
                    (
                        "model.layers.0.self_attn.q_proj.weight",
                        Some(GgmlDType::Q8_0),
                    ),
                    ("model.embed_tokens.weight", Some(GgmlDType::Q8_0)),
                    ("model.layers.0.input_layernorm.weight", None), // F16
                ],
            ),
            (
                "google/gemma-4-E4B-it",
                GgmlDType::Q4K,
                &[
                    // gate_proj always Q4K
                    (
                        "model.language_model.layers.0.mlp.gate_proj.weight",
                        Some(GgmlDType::Q4K),
                    ),
                    (
                        "model.language_model.layers.0.self_attn.q_proj.weight",
                        Some(GgmlDType::Q4K),
                    ),
                    // embeddings → Q6K
                    (
                        "model.language_model.embed_tokens.weight",
                        Some(GgmlDType::Q6K),
                    ),
                    (
                        "model.language_model.embed_tokens_per_layer.weight",
                        Some(GgmlDType::Q6K),
                    ),
                    // norms → unquantized
                    ("model.language_model.layers.0.input_layernorm.weight", None),
                    // use_more_bits(0,42)=true  → layer 0 attn_v/ffn_down → Q6K
                    (
                        "model.language_model.layers.0.self_attn.v_proj.weight",
                        Some(GgmlDType::Q6K),
                    ),
                    (
                        "model.language_model.layers.0.mlp.down_proj.weight",
                        Some(GgmlDType::Q6K),
                    ),
                    // use_more_bits(5,42)=false → layer 5 attn_v/ffn_down → Q4K
                    (
                        "model.language_model.layers.5.self_attn.v_proj.weight",
                        Some(GgmlDType::Q4K),
                    ),
                    (
                        "model.language_model.layers.5.mlp.down_proj.weight",
                        Some(GgmlDType::Q4K),
                    ),
                    // use_more_bits(7,42)=true  ((7-5)%3==2) → Q6K
                    (
                        "model.language_model.layers.7.self_attn.v_proj.weight",
                        Some(GgmlDType::Q6K),
                    ),
                    (
                        "model.language_model.layers.7.mlp.down_proj.weight",
                        Some(GgmlDType::Q6K),
                    ),
                    // use_more_bits(41,42)=true (≥7*42/8=36) → Q6K
                    (
                        "model.language_model.layers.41.self_attn.v_proj.weight",
                        Some(GgmlDType::Q6K),
                    ),
                    (
                        "model.language_model.layers.41.mlp.down_proj.weight",
                        Some(GgmlDType::Q6K),
                    ),
                ],
            ),
            (
                "google/gemma-4-26B-A4B-it",
                GgmlDType::Q4K,
                &[
                    // MoE expert weights → Q4K
                    (
                        "model.layers.0.mlp.experts.0.gate_proj.weight",
                        Some(GgmlDType::Q4K),
                    ),
                    // Shared expert (always active) → Q8_0
                    (
                        "model.layers.0.mlp.shared_expert.gate_proj.weight",
                        Some(GgmlDType::Q8_0),
                    ),
                    // Attention → Q8_0
                    (
                        "model.layers.0.self_attn.q_proj.weight",
                        Some(GgmlDType::Q8_0),
                    ),
                    // Embedding → Q6K
                    ("model.embed_tokens.weight", Some(GgmlDType::Q6K)),
                    // Norm → F16
                    ("model.layers.0.input_layernorm.weight", None),
                ],
            ),
            (
                "google/gemma-4-31B-it",
                GgmlDType::Q4K,
                &[
                    ("model.layers.0.mlp.gate_proj.weight", Some(GgmlDType::Q4K)),
                    (
                        "model.layers.0.self_attn.q_proj.weight",
                        Some(GgmlDType::Q4K),
                    ),
                    ("model.embed_tokens.weight", Some(GgmlDType::Q6K)),
                    ("model.layers.0.input_layernorm.weight", None),
                ],
            ),
            // Short-form (no org prefix) should also match.
            ("gemma-4-E2B-it", GgmlDType::Q8_0, &[]),
        ];

        for (model_id, expected_default, tensor_cases) in cases {
            let recipe = recipe_for(model_id).unwrap_or_else(|| panic!("no recipe for {model_id}"));
            assert_eq!(
                recipe.default_dtype, *expected_default,
                "{model_id}: wrong default_dtype"
            );
            for (tensor, expected) in *tensor_cases {
                assert_eq!(
                    recipe.dtype_for(tensor, 42),
                    *expected,
                    "{model_id}: wrong dtype for {tensor}"
                );
            }
        }
    }
}

// ── GGUF path derivation ─────────────────────────────────────────────────────

/// Derive the canonical GGUF output path for a given set of weight files and format.
///
/// The file is placed in the same directory as the first safetensors shard,
/// with a name of the form `model-<FORMAT>.gguf` (e.g. `model-Q4K.gguf`).
/// This keeps the quantized file alongside the original in the HF hub cache so
/// the hub machinery can locate it by inspecting the same snapshot directory.
pub fn gguf_path(weight_paths: &[PathBuf], dtype: GgmlDType) -> PathBuf {
    let dir = weight_paths
        .first()
        .and_then(|p| p.parent())
        .unwrap_or(Path::new("."));
    let format_str = format!("{:?}", dtype); // e.g. "Q4K"
    dir.join(format!("model-{}.gguf", format_str))
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Quantize all weight files and write a GGUF to `out_path`.
///
/// Reads every safetensors shard in `weight_paths` on the CPU, applies the
/// per-tensor policy, and writes a single GGUF v2 file.
///
/// If `model_id` is supplied and a [`Recipe`] is registered for it, that
/// recipe's per-tensor classifier is used (e.g. Q8_0 for E2B, Q4K+Q6K for
/// E4B).  `quant_dtype` is then used only as the fallback for tensors not
/// covered by the recipe, and as the recipe's `default_dtype` when no recipe
/// is found.
///
/// Progress is logged at INFO level.  The conversion is single-threaded and
/// CPU-bound; for a 10 GB model with Q4K it typically takes 30–90 s.
/// Read `chat_template.jinja` from the model directory (the same directory as
/// the safetensors shards).  Returns `None` if the file does not exist.
///
/// The Jinja2 template is embedded verbatim into the GGUF as the
/// `tokenizer.chat_template` metadata key, exactly as `convert_hf_to_gguf.py`
/// does.  This allows other tools (llama.cpp, open-webui, etc.) that read the
/// GGUF to use the model's authoritative template rather than falling back to
/// a built-in default.
fn read_chat_template(weight_paths: &[PathBuf]) -> Option<String> {
    let dir = weight_paths.first()?.parent()?;
    let path = dir.join("chat_template.jinja");
    std::fs::read_to_string(&path)
        .map_err(|e| {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!("Could not read {}: {e}", path.display());
            }
        })
        .ok()
}

pub fn convert_to_gguf(
    weight_paths: &[PathBuf],
    out_path: &Path,
    quant_dtype: GgmlDType,
    model_id: Option<&str>,
) -> Result<()> {
    use candle_core::quantized::{gguf_file, QTensor};
    use indicatif::{ProgressBar, ProgressStyle};

    // Derive model directory from the first weight shard (used for config.json etc.)
    let model_dir = weight_paths.first().and_then(|p| p.parent());

    // Open all safetensors shards on the CPU.
    // SAFETY: the memory maps are read-only and live for the duration of this function.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(weight_paths)? };

    // Enumerate all tensor names.
    let tensor_names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
    let n = tensor_names.len() as u64;

    // Read n_layers from config.json for per-layer quantization decisions.
    let config_n_layers: usize = model_dir
        .and_then(|d| std::fs::read_to_string(d.join("config.json")).ok())
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|c| {
            let default = serde_json::Value::Null;
            let tc = c.get("text_config").unwrap_or(&default);
            tc.get("num_hidden_layers")
                .or_else(|| c.get("num_hidden_layers"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        })
        .unwrap_or(0);

    // Resolve the per-tensor classifier: model-specific recipe takes priority.
    type Classifier = fn(&str, GgmlDType, usize) -> Option<GgmlDType>;
    let recipe = model_id.and_then(recipe_for);
    let (effective_dtype, classifier): (GgmlDType, Classifier) = match recipe {
        Some(r) => {
            tracing::info!(
                "Using model-specific quantization recipe {:?} for {:?}",
                r.label,
                model_id.unwrap_or("")
            );
            (r.default_dtype, r.classifier)
        }
        None => (quant_dtype, classify_generic),
    };

    let bar = ProgressBar::new(n);
    bar.set_style(
        ProgressStyle::with_template("{msg}\n{wide_bar} {pos}/{len} tensors  ({elapsed_precise})")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏ "),
    );
    bar.set_message(format!(
        "Quantizing to {:?}  →  {}",
        effective_dtype,
        out_path.file_name().unwrap_or_default().to_string_lossy(),
    ));

    // Detect Gemma4 model by checking config.json in the model directory.
    let is_gemma4 = model_dir
        .and_then(|d| std::fs::read_to_string(d.join("config.json")).ok())
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .map(|c| {
            c.get("model_type").and_then(|v| v.as_str()) == Some("gemma4")
                || c.get("text_config")
                    .and_then(|t| t.get("model_type"))
                    .and_then(|v| v.as_str())
                    == Some("gemma4")
        })
        .unwrap_or(false);

    // Build (canonical_name, QTensor) pairs.
    let mut qtensors: Vec<(String, QTensor)> = Vec::with_capacity(n as usize);

    // For NVFP4 detection.
    let name_set: std::collections::HashSet<&str> =
        tensor_names.iter().map(|s| s.as_str()).collect();

    for hf_name in &tensor_names {
        if inferrs_models::nvfp4::is_nvfp4_aux(hf_name) {
            bar.inc(1);
            continue;
        }

        // For Gemma4: remap to canonical name or skip (returns None for vision/audio).
        let canonical_name: String = if is_gemma4 {
            match remap_tensor_name_gemma4(hf_name) {
                Some(n) => n,
                None => {
                    bar.inc(1);
                    continue;
                }
            }
        } else {
            // Non-Gemma4: keep HF name, skip multimodal encoders.
            if should_skip_in_gguf(hf_name) {
                bar.inc(1);
                continue;
            }
            hf_name.clone()
        };

        // Load tensor on CPU as F32.
        let tensor = if hf_name.ends_with(".weight")
            && name_set.contains(format!("{}.weight_scale", &hf_name[..hf_name.len() - 7]).as_str())
        {
            let base = &hf_name[..hf_name.len() - 7];
            tracing::info!("NVFP4 dequantizing {hf_name}");
            inferrs_models::nvfp4::load_from_safetensors(&st, base)?
        } else {
            st.load(hf_name, &Device::Cpu)?.to_dtype(DType::F32)?
        };

        // For Gemma4: norm/scale tensors → F32 (matches ggml-org).
        // Special case: per_layer_model_proj → F16 for E2B (ggml-org convention).
        // For others: norm/scale tensors → F16 (legacy behaviour).
        let is_per_layer_proj = canonical_name == "per_layer_model_proj.weight";
        let use_f32 = is_gemma4 && !is_per_layer_proj && canonical_name_is_f32(&canonical_name);
        // per_layer_model_proj special handling (must match ggml-org exactly):
        //  - E2B (Q8_0 recipe): F16
        //  - E4B (Q4K recipe):  BF16  ← ggml-org stores this as type 30 (BF16),
        //                               NOT TQ2_0 (type 35).  Full BF16 precision
        //                               preserves quality for this critical PLI tensor.
        let use_f16_gemma4 = is_gemma4 && is_per_layer_proj && effective_dtype == GgmlDType::Q8_0; // E2B → F16
        let use_bf16_gemma4 = is_gemma4 && is_per_layer_proj && effective_dtype == GgmlDType::Q4K; // E4B → BF16
        let target = if use_f32 || use_f16_gemma4 || use_bf16_gemma4 || tensor.elem_count() < 32 {
            None // handled below (F32, F16, BF16, or too small)
        } else {
            classifier(hf_name, effective_dtype, config_n_layers)
        };

        let qt = if use_bf16_gemma4 {
            // per_layer_model_proj for E4B → BF16 (type 30, matches ggml-org exactly)
            let t = tensor.to_dtype(DType::BF16)?;
            QTensor::quantize(&t, GgmlDType::BF16)
                .with_context(|| format!("BF16 wrap failed for {hf_name}"))?
        } else if use_f16_gemma4 {
            // per_layer_model_proj for E2B → F16 (matches ggml-org)
            let t = tensor.to_dtype(DType::F16)?;
            QTensor::quantize(&t, GgmlDType::F16)
                .with_context(|| format!("F16 wrap failed for {hf_name}"))?
        } else if use_f32 || (target.is_none() && is_gemma4) {
            // Gemma4 unquantized tensors → F32.
            let f32_tensor = if tensor.dims().is_empty() {
                tensor.reshape((1,))?
            } else {
                tensor
            };
            QTensor::quantize(&f32_tensor, GgmlDType::F32)
                .with_context(|| format!("F32 wrap failed for {hf_name}"))?
        } else {
            match target {
                None => {
                    // Non-Gemma4 unquantized → F16.
                    let t = if tensor.dims().is_empty() {
                        tensor.reshape((1,))?.to_dtype(DType::F16)?
                    } else {
                        tensor.to_dtype(DType::F16)?
                    };
                    QTensor::quantize(&t, GgmlDType::F16)
                        .with_context(|| format!("F16 wrap failed for {hf_name}"))?
                }
                Some(dtype) => match QTensor::quantize(&tensor, dtype) {
                    Ok(qt) => qt,
                    Err(_) => match QTensor::quantize(&tensor, GgmlDType::Q8_0) {
                        Ok(qt) => qt,
                        Err(_) => {
                            let t = if is_gemma4 {
                                tensor.clone()
                            } else {
                                tensor.to_dtype(DType::F16)?
                            };
                            let fallback = if is_gemma4 {
                                GgmlDType::F32
                            } else {
                                GgmlDType::F16
                            };
                            QTensor::quantize(&t, fallback)
                                .with_context(|| format!("fallback wrap failed for {hf_name}"))?
                        }
                    },
                },
            }
        };

        qtensors.push((canonical_name, qt));
        bar.inc(1);
    }

    bar.finish_and_clear();

    // Build metadata: full ggml-org-compatible KV block for Gemma4,
    // or a minimal block (chat template only) for other models.
    let out_file = std::fs::File::create(out_path)
        .with_context(|| format!("Cannot create GGUF file at {}", out_path.display()))?;
    let mut writer = BufWriter::new(out_file);

    let tensor_refs: Vec<(&str, &QTensor)> = qtensors
        .iter()
        .map(|(name, qt)| (name.as_str(), qt))
        .collect();

    if is_gemma4 {
        let dir = model_dir.context("no model directory")?;
        let meta = Gemma4Meta::from_model_dir(dir, effective_dtype)
            .context("building Gemma4 GGUF metadata")?;

        // Generate rope_freqs.weight: F32 tensor of shape [global_head_dim/2].
        // Contains precomputed RoPE inverse frequencies:
        //   freqs[i] = 1.0 / rope_theta ^ (2i / global_head_dim)
        // Matches ggml-org's rope_freqs.weight exactly.
        let rope_freqs_qt = {
            use candle_core::quantized::QTensor;
            // Extract rope_theta and global_head_dim from the metadata KV.
            let rope_theta = meta
                .kv
                .iter()
                .find(|(k, _)| k == "gemma4.rope.freq_base")
                .and_then(|(_, v)| {
                    if let gguf_file::Value::F32(f) = v {
                        Some(*f)
                    } else {
                        None
                    }
                })
                .unwrap_or(1_000_000.0f32);
            let global_head_dim = meta
                .kv
                .iter()
                .find(|(k, _)| k == "gemma4.attention.key_length")
                .and_then(|(_, v)| {
                    if let gguf_file::Value::U32(u) = v {
                        Some(*u)
                    } else {
                        None
                    }
                })
                .unwrap_or(512u32);
            let n = (global_head_dim / 2) as usize;
            let freqs: Vec<f32> = (0..n)
                .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / global_head_dim as f32))
                .collect();
            let t = candle_core::Tensor::from_vec(freqs, (n,), &Device::Cpu)?;
            QTensor::quantize(&t, GgmlDType::F32).context("rope_freqs.weight quantize failed")?
        };

        // Append rope_freqs.weight to the tensor list.
        let mut all_qtensors = qtensors;
        all_qtensors.push(("rope_freqs.weight".to_string(), rope_freqs_qt));

        let tensor_refs_with_rope: Vec<(&str, &QTensor)> = all_qtensors
            .iter()
            .map(|(name, qt)| (name.as_str(), qt))
            .collect();

        let kv_refs: Vec<(&str, &gguf_file::Value)> =
            meta.kv.iter().map(|(k, v)| (k.as_str(), v)).collect();
        gguf_file::write_v3(&mut writer, &kv_refs, &tensor_refs_with_rope)
            .context("Failed to write Gemma4 GGUF")?;
    } else {
        // Non-Gemma4: GGUF v2 with chat template only (legacy path).
        let chat_template_value;
        let metadata: &[(&str, &gguf_file::Value)] =
            if let Some(tmpl) = read_chat_template(weight_paths) {
                tracing::info!(
                    "Embedding chat_template.jinja ({} bytes) into GGUF metadata",
                    tmpl.len()
                );
                chat_template_value = gguf_file::Value::String(tmpl);
                &[("tokenizer.chat_template", &chat_template_value)]
            } else {
                &[]
            };
        gguf_file::write(&mut writer, metadata, &tensor_refs)
            .context("Failed to write GGUF file")?;
    }

    let size_bytes = std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "Saved {:.2} GiB → {}",
        size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        out_path.display(),
    );

    Ok(())
}

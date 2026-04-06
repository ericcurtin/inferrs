//! Weight quantization: converts safetensors model weights to a GGUF file on disk.
//!
//! ## Usage
//!
//! ```text
//! inferrs serve google/gemma-4-E2B-it --quantize          # default Q4_K_M (= Q4K)
//! inferrs serve google/gemma-4-E2B-it --quantize=Q8_0     # explicit format
//! ```
//!
//! On first invocation the weights are read from the HuggingFace cache, quantized
//! on the CPU, and written to a `.gguf` file next to the safetensors shards.
//! Subsequent invocations find the file already present and skip the conversion.
//!
//! ## Per-tensor policy
//!
//! Tensors whose names contain any of the "keep" substrings (embed, norm, head,
//! bias) are stored at F16; every other tensor (primarily the large linear weight
//! matrices) is quantized to the requested dtype.  This mirrors the llama.cpp
//! `convert_hf_to_gguf.py` convention of keeping embedding and normalisation
//! layers in full precision for accuracy.
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
        other => anyhow::bail!(
            "Unknown quantization format {:?}. \
             Accepted: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2K, Q3K, Q4K (default/Q4_K_M), Q5K, Q6K.",
            other
        ),
    }
}

// ── Per-tensor quantization policy ───────────────────────────────────────────

/// Substrings in a tensor name that indicate it should be kept at F16 rather
/// than quantized.
///
/// - ALL normalisation weights — any tensor whose name contains `norm`
///   (catches `.norm`, `_norm`, `layernorm`, `input_layernorm`,
///   `pre_feedforward_layernorm`, `post_feedforward_layernorm`, etc.)
///   Norm weights are 1-D vectors and often contain large outlier values;
///   quantizing them causes catastrophic accuracy loss.
/// - bias vectors
/// - RoPE tables (`rope`)
/// - per-layer scalar gates (`layer_scalar`)
///
/// Note: `embed_tokens` and `lm_head` are intentionally NOT in this list.
/// They are instead handled by `should_use_q6k`, which quantizes them to Q6K
/// (higher accuracy than the default format) rather than keeping them at F16.
/// This is the same strategy used by llama.cpp for tied-embedding models:
/// the output projection GEMV benefits greatly from quantization (4× bandwidth
/// reduction) and Q6K is accurate enough that vocabulary logits are unaffected.
const KEEP_F16_SUBSTRINGS: &[&str] = &[
    "norm", // catches layernorm, input_layernorm, pre/post_feedforward_layernorm, etc.
    ".bias",
    "rope",
    "layer_scalar",
];

/// Tensor name substrings that should be quantized to Q6K (high accuracy)
/// regardless of the user-requested format.
///
/// - Embedding tables (`embed_tokens`, `embed_tokens_per_layer`): these are
///   used both for token lookup and (for `embed_tokens`) as the tied lm_head
///   GEMV weight.  Q6K gives near-F16 accuracy (6 bits/param) while reducing
///   the 805 MiB `embed_tokens` weight to ~270 MiB — a 3× bandwidth reduction
///   for the lm_head GEMV, which is the single most expensive operation per
///   decode step (≈80% of memory bandwidth).
/// - `lm_head` (if present as a separate tensor in non-tied models)
const USE_Q6K_SUBSTRINGS: &[&str] = &["embed_tokens", "lm_head"];

/// Return true when a tensor should be kept at F16 (not quantized).
fn should_keep_f16(name: &str) -> bool {
    KEEP_F16_SUBSTRINGS.iter().any(|&sub| name.contains(sub))
}

/// Return true when a tensor should be quantized to Q6K regardless of the
/// user-requested format.
fn should_use_q6k(name: &str) -> bool {
    USE_Q6K_SUBSTRINGS.iter().any(|&sub| name.contains(sub))
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
/// per-tensor policy (F16 for embed/norm/head, `quant_dtype` for everything
/// else), and writes a single GGUF v2 file.
///
/// Progress is logged at INFO level so the user can see which tensors are being
/// processed.  The conversion is single-threaded and CPU-bound; for a 10 GB
/// model with Q4K it typically takes 30–90 s on a modern laptop.
pub fn convert_to_gguf(
    weight_paths: &[PathBuf],
    out_path: &Path,
    quant_dtype: GgmlDType,
) -> Result<()> {
    use candle_core::quantized::{gguf_file, QTensor};
    use indicatif::{ProgressBar, ProgressStyle};

    // Open all safetensors shards on the CPU.
    // SAFETY: the memory maps are read-only and live for the duration of this function.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(weight_paths)? };

    // Enumerate all tensor names.
    let tensor_names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
    let n = tensor_names.len() as u64;

    let bar = ProgressBar::new(n);
    bar.set_style(
        ProgressStyle::with_template("{msg}\n{wide_bar} {pos}/{len} tensors  ({elapsed_precise})")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏ "),
    );
    bar.set_message(format!(
        "Quantizing to {:?}  →  {}",
        quant_dtype,
        out_path.file_name().unwrap_or_default().to_string_lossy(),
    ));

    // Build (name, QTensor) pairs — collected eagerly to pass a slice to the writer.
    let mut qtensors: Vec<(String, QTensor)> = Vec::with_capacity(n as usize);

    // Build a set of all tensor names for NVFP4 detection.
    // An NVFP4 weight tensor is one whose ".weight" sibling has a ".weight_scale"
    // present in the file.  We detect this by checking the full name set.
    let name_set: std::collections::HashSet<&str> =
        tensor_names.iter().map(|s| s.as_str()).collect();

    for name in &tensor_names {
        // Skip NVFP4 auxiliary tensors — they are consumed when dequantizing the
        // corresponding weight tensor and must not appear as independent entries.
        if crate::nvfp4::is_nvfp4_aux(name) {
            bar.inc(1);
            continue;
        }

        // Detect NVFP4 weight tensors: a ".weight" whose sibling ".weight_scale"
        // exists in the file.  Dequantize them via the nvfp4 module so that the
        // resulting GGUF stores the correct full-width float matrix.
        let tensor = if name.ends_with(".weight")
            && name_set.contains(format!("{}.weight_scale", &name[..name.len() - 7]).as_str())
        {
            let base = &name[..name.len() - 7]; // strip trailing ".weight"
            tracing::info!("NVFP4 dequantizing {name}");
            crate::nvfp4::load_from_safetensors(&st, base)?
        } else {
            // Load tensor on CPU in f32 — quantization requires f32 input.
            st.load(name, &Device::Cpu)?.to_dtype(DType::F32)?
        };

        // All GGML quantization formats require a non-scalar shape and a last
        // dimension that is a multiple of the block size (32 for Q8_0, 256 for
        // k-quants).  Scalars and very small tensors are stored at F16 instead.
        //
        // QTensor::quantize also rejects scalars for F16, so we reshape any 0-d
        // tensor to [1] before wrapping it.
        let keep_f16 = should_keep_f16(name) || tensor.elem_count() < 32;
        // embed_tokens and lm_head are quantized to Q6K for high accuracy.
        let use_q6k = !keep_f16 && should_use_q6k(name);

        let qt = if keep_f16 {
            // Keep at F16. Scalars (rank 0) must be reshaped to [1] first
            // because QTensor::quantize rejects empty dims even for F16.
            let f16_tensor = if tensor.dims().is_empty() {
                tensor.reshape((1,))?.to_dtype(DType::F16)?
            } else {
                tensor.to_dtype(DType::F16)?
            };
            QTensor::quantize(&f16_tensor, GgmlDType::F16)
                .with_context(|| format!("F16 wrap failed for {name}"))?
        } else {
            // Select the target dtype for this tensor.
            let target_dtype = if use_q6k { GgmlDType::Q6K } else { quant_dtype };

            // Quantize to the requested dtype.
            // k-quants require the last dim to be a multiple of 256; Q8_0
            // requires a multiple of 32.  Fall back down the chain as needed.
            match QTensor::quantize(&tensor, target_dtype) {
                Ok(qt) => qt,
                Err(_) => match QTensor::quantize(&tensor, GgmlDType::Q8_0) {
                    Ok(qt) => qt,
                    Err(_) => {
                        let f16_tensor = tensor.to_dtype(DType::F16)?;
                        QTensor::quantize(&f16_tensor, GgmlDType::F16)
                            .with_context(|| format!("F16 wrap failed for {name}"))?
                    }
                },
            }
        };

        qtensors.push((name.to_string(), qt));
        bar.inc(1);
    }

    bar.finish_and_clear();

    // Write the GGUF file.
    // Empty metadata slice: inferrs rebuilds config from config.json, so no
    // arch/context keys are needed in the GGUF.
    let out_file = std::fs::File::create(out_path)
        .with_context(|| format!("Cannot create GGUF file at {}", out_path.display()))?;
    let mut writer = BufWriter::new(out_file);

    let tensor_refs: Vec<(&str, &QTensor)> = qtensors
        .iter()
        .map(|(name, qt)| (name.as_str(), qt))
        .collect();

    let no_metadata: &[(&str, &gguf_file::Value)] = &[];
    gguf_file::write(&mut writer, no_metadata, &tensor_refs)
        .context("Failed to write GGUF file")?;

    let size_bytes = std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "Saved {:.2} GiB → {}",
        size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        out_path.display(),
    );

    Ok(())
}

//! HuggingFace Hub model downloading.

use anyhow::{Context, Result};
use candle_core::quantized::GgmlDType;
use hf_hub::api::sync::Api;
use std::path::PathBuf;

/// GGUF metadata key that stores the HuggingFace repo ID of the source model
/// (e.g. "google/gemma-4-E2B-it").  Set by ggml-org conversion scripts.
const GGUF_SOURCE_REPO_KEY: &str = "general.source.repo_id";

/// Files needed to load a model.
pub struct ModelFiles {
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub tokenizer_config_path: Option<PathBuf>,
    /// Original safetensors shards (may be empty when loading a pre-existing
    /// GGUF from a CNCF ModelPack bundle).
    pub weight_paths: Vec<PathBuf>,
    /// Path to a GGUF file.  Populated when `--quantize` was given or when
    /// loading a CNCF ModelPack bundle that contains a GGUF model.
    /// When `Some`, callers should load weights from this GGUF instead of
    /// `weight_paths`.
    pub gguf_path: Option<PathBuf>,
}

/// Load model files from a local directory (no network required).
pub fn load_local_model(path: &std::path::Path) -> Result<ModelFiles> {
    tracing::info!("Loading model from local path: {}", path.display());

    let config_path = path.join("config.json");
    anyhow::ensure!(
        config_path.exists(),
        "config.json not found in {}",
        path.display()
    );

    let tokenizer_path = path.join("tokenizer.json");
    anyhow::ensure!(
        tokenizer_path.exists(),
        "tokenizer.json not found in {}",
        path.display()
    );

    let tokenizer_config_path = {
        let p = path.join("tokenizer_config.json");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    };

    // Prefer model.safetensors, then scan for shards
    let weight_paths = if path.join("model.safetensors").exists() {
        vec![path.join("model.safetensors")]
    } else {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(path)
            .with_context(|| format!("Cannot read {}", path.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
            .collect();
        shards.sort();
        anyhow::ensure!(
            !shards.is_empty(),
            "No safetensors files found in {}",
            path.display()
        );
        shards
    };

    Ok(ModelFiles {
        config_path,
        tokenizer_path,
        tokenizer_config_path,
        weight_paths,
        gguf_path: None,
    })
}

/// Download model files from HuggingFace Hub.
///
/// When `gguf_file` is `Some`, the repo is treated as a GGUF-only repo and
/// that specific file is downloaded.  When `None`, the repo is expected to
/// contain safetensors weights and the usual config/tokenizer files.
pub fn download_model(
    model_id: &str,
    revision: &str,
    gguf_file: Option<&str>,
    tokenizer_source: Option<&str>,
) -> Result<ModelFiles> {
    // If the model_id looks like a local path, load directly without network.
    let as_path = std::path::Path::new(model_id);
    if as_path.is_absolute()
        || model_id.starts_with("./")
        || model_id.starts_with("../")
        || as_path.exists()
    {
        return load_local_model(as_path);
    }

    // ── OCI model resolution ────────────────────────────────────────────
    // For OCI references (single word → docker.io/ai, or explicit registry),
    // first check the Docker Model Runner store (~/.docker/models) for a
    // cached bundle.  If not found, auto-pull via the Go helper.
    // This allows `inferrs run gemma3` and `inferrs serve docker.io/org/model`
    // to work seamlessly — reusing DMR's cache or pulling on demand.
    if let crate::pull::RefKind::Oci = crate::pull::classify_reference(model_id) {
        // --revision is only meaningful for HuggingFace references; warn when
        // it is set to a non-default value for an OCI model so the user knows
        // it is being ignored.
        if revision != "main" {
            tracing::warn!(
                "--revision '{}' is ignored for OCI model '{}'; \
                 use an OCI tag instead (e.g. docker.io/org/model:v2)",
                revision,
                model_id,
            );
        }

        let bundle_path = match crate::pull::oci_bundle_path(model_id) {
            Some(p) => {
                tracing::info!(
                    "Found OCI model in local store: {} → {}",
                    model_id,
                    p.display()
                );
                p
            }
            None => {
                tracing::info!("OCI model not cached, pulling: {}", model_id);
                crate::pull::oci_pull_model(model_id)?
            }
        };
        return crate::modelpack::load_bundle(&bundle_path);
    }

    tracing::info!("Downloading model {} (revision: {})", model_id, revision);

    let api = Api::new().context("Failed to create HuggingFace API client")?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));

    // Fast-path: caller explicitly asked for a specific GGUF file.
    if let Some(fname) = gguf_file {
        return download_gguf_only_repo(&repo, &api, model_id, fname, tokenizer_source);
    }

    // Probe for config.json.  If it is missing the repo is likely GGUF-only
    // (e.g. ggml-org/gemma-4-E2B-it-GGUF).  Auto-detect and handle it.
    let config_result = repo.get("config.json");
    if config_result.is_err() {
        tracing::info!("config.json not found in {model_id} — checking for GGUF-only repo");
        let gguf_fname = pick_best_gguf_file(&repo, model_id)?;
        return download_gguf_only_repo(&repo, &api, model_id, &gguf_fname, tokenizer_source);
    }
    let config_path = config_result.expect("checked above");
    tracing::info!("Downloaded config.json");

    // Download tokenizer.json
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("Failed to download tokenizer.json")?;
    tracing::info!("Downloaded tokenizer.json");

    // Try to download tokenizer_config.json (optional)
    let tokenizer_config_path = repo.get("tokenizer_config.json").ok();
    if tokenizer_config_path.is_some() {
        tracing::info!("Downloaded tokenizer_config.json");
    }

    // Download safetensors weight files
    let weight_paths = download_safetensors(&repo)?;

    Ok(ModelFiles {
        config_path,
        tokenizer_path,
        tokenizer_config_path,
        weight_paths,
        gguf_path: None,
    })
}

/// List `.gguf` files in `repo` (excluding vision projectors like `mmproj-*`)
/// and pick the best one: prefer Q4K, then Q8_0, then the first one found.
fn pick_best_gguf_file(repo: &hf_hub::api::sync::ApiRepo, model_id: &str) -> Result<String> {
    let info = repo.info().with_context(|| {
        format!("Failed to list files in {model_id}: no config.json and repo.info() failed")
    })?;

    let gguf_files: Vec<String> = info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .filter(|name| {
            name.ends_with(".gguf") && !name.starts_with("mmproj-") && !name.contains("mmproj")
        })
        .collect();

    anyhow::ensure!(
        !gguf_files.is_empty(),
        "No .gguf files found in {model_id} (and no config.json present)"
    );

    // Prefer by quality/size trade-off: Q4K > Q5K > Q6K > Q8_0 > anything
    // else (avoids accidentally picking a massive bf16 or f32 file).
    let preferred = gguf_files
        .iter()
        .find(|f| {
            let lower = f.to_lowercase();
            lower.contains("q4_k") || lower.contains("q4k")
        })
        .or_else(|| {
            gguf_files.iter().find(|f| {
                let lower = f.to_lowercase();
                lower.contains("q5_k") || lower.contains("q5k")
            })
        })
        .or_else(|| {
            gguf_files.iter().find(|f| {
                let lower = f.to_lowercase();
                lower.contains("q6_k") || lower.contains("q6k")
            })
        })
        .or_else(|| {
            gguf_files.iter().find(|f| {
                let lower = f.to_lowercase();
                lower.contains("q8_0") || lower.contains("q8-0")
            })
        })
        .unwrap_or(&gguf_files[0]);

    tracing::info!(
        "GGUF-only repo detected. Available files: {gguf_files:?}. Selected: {preferred}"
    );
    Ok(preferred.clone())
}

/// Download a GGUF file from `repo`, then fetch `config.json` and
/// `tokenizer.json` from the source model referenced in the GGUF metadata.
fn download_gguf_only_repo(
    repo: &hf_hub::api::sync::ApiRepo,
    api: &Api,
    model_id: &str,
    gguf_filename: &str,
    tokenizer_source: Option<&str>,
) -> Result<ModelFiles> {
    tracing::info!("Downloading GGUF file: {gguf_filename}");
    let gguf_path = repo
        .get(gguf_filename)
        .with_context(|| format!("Failed to download {gguf_filename} from {model_id}"))?;
    tracing::info!("Downloaded {gguf_filename}");

    // Read GGUF metadata to find the source model repo.
    let source_repo_id = if let Some(ts) = tokenizer_source {
        Some(ts.to_string())
    } else {
        // Use `unwrap_or_else` so a parse error (e.g. unknown GGUF metadata
        // type in a future format version) falls through to the HF model card
        // fallback instead of aborting the whole pull.
        let from_gguf = read_gguf_source_repo(&gguf_path).unwrap_or_else(|e| {
            tracing::debug!("Could not read GGUF source repo from metadata: {e:#}");
            None
        });
        if from_gguf.is_some() {
            from_gguf
        } else {
            // GGUF metadata lacked the source repo key.  Many ggml-org repos
            // do not embed it but do advertise the original model via the
            // HuggingFace model card (`cardData.base_model`).  Try that next.
            read_hf_base_model(model_id)
        }
    };

    let (config_path, tokenizer_path, tokenizer_config_path) = match source_repo_id {
        Some(ref src) => {
            tracing::info!("Fetching config and tokenizer from source model: {src}");
            // Always use the default branch for the source model: the
            // `revision` argument belongs to the GGUF repo and is very
            // unlikely to exist in the original model repo.
            let src_repo = api.repo(hf_hub::Repo::new(src.clone(), hf_hub::RepoType::Model));
            let config_path = src_repo.get("config.json").with_context(|| {
                format!("Failed to download config.json from source model {src}")
            })?;
            tracing::info!("Downloaded config.json from {src}");
            let tokenizer_path = src_repo.get("tokenizer.json").with_context(|| {
                format!("Failed to download tokenizer.json from source model {src}")
            })?;
            tracing::info!("Downloaded tokenizer.json from {src}");
            let tokenizer_config_path = src_repo.get("tokenizer_config.json").ok();
            (config_path, tokenizer_path, tokenizer_config_path)
        }
        None => {
            // No source repo in GGUF metadata — try the GGUF repo itself.
            tracing::warn!(
                "GGUF metadata does not contain '{GGUF_SOURCE_REPO_KEY}'. \
                 Trying config.json and tokenizer.json from the same repo."
            );
            let config_path = repo
                .get("config.json")
                .context("Failed to download config.json (not in GGUF repo and no source repo found in GGUF metadata)")?;
            let tokenizer_path = repo
                .get("tokenizer.json")
                .context("Failed to download tokenizer.json (not in GGUF repo and no source repo found in GGUF metadata)")?;
            let tokenizer_config_path = repo.get("tokenizer_config.json").ok();
            (config_path, tokenizer_path, tokenizer_config_path)
        }
    };

    Ok(ModelFiles {
        config_path,
        tokenizer_path,
        tokenizer_config_path,
        // GGUF-only repos have no safetensors; weight_paths stays empty.
        weight_paths: vec![],
        gguf_path: Some(gguf_path),
    })
}

/// Open a GGUF file and extract the value of `general.source.repo_id` from
/// its metadata, returning `None` if the key is absent or the file cannot be
/// read.
fn read_gguf_source_repo(gguf_path: &std::path::Path) -> Result<Option<String>> {
    use candle_core::quantized::gguf_file;

    let file = std::fs::File::open(gguf_path)
        .with_context(|| format!("Cannot open GGUF {}", gguf_path.display()))?;
    let mut reader = std::io::BufReader::new(file);

    let content = gguf_file::Content::read(&mut reader)
        .with_context(|| format!("Failed to parse GGUF header in {}", gguf_path.display()))?;

    let repo_id = content
        .metadata
        .get(GGUF_SOURCE_REPO_KEY)
        .and_then(|v| v.to_string().ok())
        .cloned();

    if let Some(ref id) = repo_id {
        tracing::info!("GGUF source repo: {id}");
    }

    Ok(repo_id)
}

/// Query the HuggingFace model card API for the `base_model` field.
///
/// Many GGUF-only repos (e.g. `ggml-org/*-GGUF`) do not embed
/// `general.source.repo_id` in the GGUF metadata but do declare the original
/// model via the repo's model card (`cardData.base_model`).  This function
/// fetches `https://huggingface.co/api/models/{model_id}` and extracts the
/// first entry from `cardData.base_model`, which is the canonical source repo
/// ID (e.g. `"google/gemma-4-E2B-it"`).
///
/// Returns `None` (never errors) when the API is unreachable or the field is
/// absent, so the caller can fall through to further fallbacks.
fn read_hf_base_model(model_id: &str) -> Option<String> {
    // Build the endpoint URL, respecting HF_ENDPOINT overrides.
    let hf_endpoint =
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    let url = format!("{hf_endpoint}/api/models/{model_id}");

    // Resolve the HF token the same way hf-hub does: $HF_TOKEN env var first,
    // then the token file in the cache directory (respects HF_HOME and
    // platform-specific paths via Cache::from_env()).
    let token: Option<String> = std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| hf_hub::Cache::from_env().token());

    let mut request = ureq::get(&url);
    if let Some(tok) = token {
        request = request.set("Authorization", &format!("Bearer {tok}"));
    }

    let response: ureq::Response = match request.call() {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!("HF model card API request failed for {model_id}: {e}");
            return None;
        }
    };

    let json: serde_json::Value = match response.into_json() {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!("Failed to parse HF model card API response for {model_id}: {e}");
            return None;
        }
    };

    // cardData.base_model is either a string or an array of strings.
    let base_model = json
        .get("cardData")
        .and_then(|cd| cd.get("base_model"))
        .and_then(|bm| {
            if let Some(s) = bm.as_str() {
                Some(s.to_string())
            } else if let Some(arr) = bm.as_array() {
                arr.first().and_then(|v| v.as_str()).map(|s| s.to_string())
            } else {
                None
            }
        });

    if let Some(ref src) = base_model {
        tracing::info!("Found source model from HF model card: {src}");
    }

    base_model
}

/// Download the model (same as [`download_model`]) and, when `quant_dtype` is
/// `Some`, ensure a quantized GGUF is present on disk.
///
/// The GGUF is written next to the safetensors shards in the HF hub cache.
/// If the file already exists it is reused without re-running the conversion.
/// Quantization happens on the CPU and can take up to a few minutes for large
/// models; progress is logged at INFO level.
///
/// `gguf_file` is forwarded to [`download_model`] to support GGUF-only repos.
pub fn download_and_maybe_quantize(
    model_id: &str,
    revision: &str,
    gguf_file: Option<&str>,
    tokenizer_source: Option<&str>,
    quant_dtype: Option<GgmlDType>,
) -> Result<ModelFiles> {
    let mut files = download_model(model_id, revision, gguf_file, tokenizer_source)?;

    let Some(dtype) = quant_dtype else {
        return Ok(files);
    };

    // GGUF-only repos have no safetensors shards to convert.  The GGUF was
    // already downloaded; skip the quantize step.
    if files.weight_paths.is_empty() {
        tracing::warn!(
            "--quantize is ignored for GGUF-only repos (the downloaded GGUF is used as-is)"
        );
        return Ok(files);
    }

    // When the model was loaded from a CNCF ModelPack bundle that already
    // contains a pre-quantized GGUF, skip the quantization step — the
    // weights are already in the desired format.
    if files.gguf_path.is_some() {
        tracing::info!(
            "Model already has a GGUF file (e.g. from a ModelPack bundle); \
             skipping --quantize"
        );
        return Ok(files);
    }

    let gguf = crate::quantize::gguf_path(&files.weight_paths, dtype);

    if gguf.exists() {
        tracing::info!("Reusing cached GGUF at {} ({:?})", gguf.display(), dtype);
    } else {
        tracing::info!(
            "Quantizing model to {:?} — this runs once and is then cached…",
            dtype
        );
        // Write to a temp path then atomically rename so that an interrupted
        // conversion (OOM, Ctrl-C, disk-full) never leaves a truncated file
        // that would be silently reused on the next run.
        let tmp = gguf.with_extension("gguf.tmp");
        crate::quantize::convert_to_gguf(&files.weight_paths, &tmp, dtype)?;
        std::fs::rename(&tmp, &gguf)
            .with_context(|| format!("Failed to rename {} → {}", tmp.display(), gguf.display()))?;
    }

    files.gguf_path = Some(gguf);
    Ok(files)
}

fn download_safetensors(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<PathBuf>> {
    // Try model.safetensors first (single file models)
    if let Ok(path) = repo.get("model.safetensors") {
        tracing::info!("Downloaded model.safetensors");
        return Ok(vec![path]);
    }

    // Try model.safetensors.index.json for sharded models
    let index_path = repo
        .get("model.safetensors.index.json")
        .context("No model.safetensors or model.safetensors.index.json found")?;

    let index_content =
        std::fs::read_to_string(&index_path).context("Failed to read safetensors index")?;
    let index: serde_json::Value =
        serde_json::from_str(&index_content).context("Failed to parse safetensors index")?;

    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .context("No weight_map in safetensors index")?;

    // Collect unique filenames
    let mut filenames: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    filenames.sort();
    filenames.dedup();

    let mut paths = Vec::new();
    for filename in &filenames {
        let path = repo
            .get(filename)
            .with_context(|| format!("Failed to download {filename}"))?;
        tracing::info!("Downloaded {}", filename);
        paths.push(path);
    }

    Ok(paths)
}

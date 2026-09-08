//! HuggingFace Hub API calls and file-selection logic. Originally
//! ported from the Go shim's `hfFetchModelInfo`/`hfFetchFiles`/
//! `selectGGUF`/`selectMMProj`/`selectLicenseFile`/
//! `safetensorsMediaType`, since deleted — this is the only
//! implementation now.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::client::HttpStatusError;
use super::oci;

/// One entry from the HuggingFace tree API — mirrors `hfFile`.
#[derive(Debug, Clone, Deserialize)]
pub struct HfFile {
    pub path: String,
    #[serde(default)]
    pub size: i64,
    #[serde(default, rename = "type")]
    pub kind: String, // "file" or "directory"
}

/// Issues an authenticated GET and decodes JSON; see `client::probe` for
/// the (lack of) retries.
async fn get_json<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
    token: Option<&str>,
) -> Result<T> {
    let body = super::client::probe(&format!("GET {url}"), || async {
        let mut req = client.get(url);
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        let resp = req.send().await.with_context(|| format!("GET {url}"))?;
        let status = resp.status();
        if status != reqwest::StatusCode::OK {
            let headers = resp.headers().clone();
            return Err(
                HttpStatusError::new(format!("GET {url}"), status.as_u16(), &headers).into(),
            );
        }
        resp.bytes()
            .await
            .with_context(|| format!("read body of GET {url}"))
    })
    .await?;
    serde_json::from_slice(&body).with_context(|| format!("decode JSON from {url}"))
}

/// The subset of `GET /api/models/{owner}/{repo}` this needs — mirrors `hfModelInfo`.
#[derive(Debug, Deserialize, Default)]
pub struct ModelInfo {
    #[serde(default)]
    sha: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default, rename = "cardData")]
    card_data: CardData,
}

#[derive(Debug, Deserialize, Default)]
struct CardData {
    #[serde(default)]
    license: String,
}

impl ModelInfo {
    /// The commit SHA to pin resolve URLs to, falling back to "main".
    pub fn commit(&self) -> &str {
        if self.sha.is_empty() {
            "main"
        } else {
            &self.sha
        }
    }

    /// Best-effort SPDX license expression, or `None` if the repo
    /// doesn't declare one usable at all.
    pub fn license(&self) -> Option<String> {
        if !self.card_data.license.is_empty() {
            if let Some(id) = normalize_spdx_license(&self.card_data.license) {
                return Some(id);
            }
        }
        for tag in &self.tags {
            if let Some(slug) = tag.strip_prefix("license:") {
                if !slug.is_empty() {
                    if let Some(id) = normalize_spdx_license(slug) {
                        return Some(id);
                    }
                }
            }
        }
        None
    }
}

/// Maps a HuggingFace license slug to its SPDX expression — mirrors
/// `spdxLicenseIDs`/`normalizeSPDXLicense`. "other"/"unknown" are HF's
/// catch-alls for "not a real license identifier" and report as unusable
/// (`None`) rather than fabricating a bogus SPDX id. Anything else not
/// listed falls through unchanged.
fn normalize_spdx_license(slug: &str) -> Option<String> {
    let slug = slug.trim().to_lowercase();
    match slug.as_str() {
        "apache-2.0" => Some("Apache-2.0".to_string()),
        "mit" => Some("MIT".to_string()),
        "bsd-2-clause" => Some("BSD-2-Clause".to_string()),
        "bsd-3-clause" => Some("BSD-3-Clause".to_string()),
        "gpl-2.0" => Some("GPL-2.0-only".to_string()),
        "gpl-3.0" => Some("GPL-3.0-only".to_string()),
        "lgpl-2.1" => Some("LGPL-2.1-only".to_string()),
        "lgpl-3.0" => Some("LGPL-3.0-only".to_string()),
        "mpl-2.0" => Some("MPL-2.0".to_string()),
        "cc-by-4.0" => Some("CC-BY-4.0".to_string()),
        "cc-by-sa-4.0" => Some("CC-BY-SA-4.0".to_string()),
        "cc0-1.0" => Some("CC0-1.0".to_string()),
        "other" | "unknown" => None,
        other => Some(other.to_string()),
    }
}

pub async fn fetch_model_info(
    client: &reqwest::Client,
    endpoint: &str,
    owner: &str,
    repo: &str,
    token: Option<&str>,
) -> Result<ModelInfo> {
    let url = format!("{endpoint}api/models/{owner}/{repo}");
    get_json(client, &url, token).await.context("HF model info")
}

pub async fn fetch_files(
    client: &reqwest::Client,
    endpoint: &str,
    owner: &str,
    repo: &str,
    commit: &str,
    token: Option<&str>,
) -> Result<Vec<HfFile>> {
    let url = format!("{endpoint}api/models/{owner}/{repo}/tree/{commit}?recursive=true");
    get_json(client, &url, token).await.context("HF file list")
}

// ---------------------------------------------------------------------------
// GGUF file selection (mirrors llama.cpp find_best_model)
// ---------------------------------------------------------------------------

const QUANT_PREFERENCE: &[&str] = &[
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "Q4_0", "Q6_K", "Q2_K",
];

/// True for GGUF files that are primary model weights (not mmproj
/// projectors or imatrix importance files).
fn is_model_gguf(path: &str) -> bool {
    let lower = path.to_lowercase();
    lower.ends_with(".gguf") && !lower.contains("mmproj") && !lower.contains("imatrix")
}

/// Parses llama.cpp's gguf-split naming convention —
/// "&lt;name&gt;-NNNNN-of-MMMMM.gguf" — without a regex dependency.
/// Returns `(prefix, index, total)`. Case-insensitive (matches
/// `is_model_gguf`'s own lowercasing), so a split isn't silently missed
/// over a `.GGUF`/`-OF-` casing difference.
fn parse_gguf_shard(path: &str) -> Option<(String, u32, u32)> {
    let base = path.rsplit('/').next().unwrap_or(path).to_lowercase();
    let base = base.strip_suffix(".gguf")?;
    let (rest, total_str) = base.rsplit_once("-of-")?;
    let total: u32 = total_str.parse().ok()?;
    let (prefix, index_str) = rest.rsplit_once('-')?;
    let index: u32 = index_str.parse().ok()?;
    Some((prefix.to_string(), index, total))
}

fn dirname(path: &str) -> &str {
    path.rsplit_once('/').map(|(d, _)| d).unwrap_or("")
}

/// Returns every shard of the same multi-part split as `chosen`, in
/// shard order — a manifest built from only some of a split's shards
/// silently produces a model no GGUF-reading runtime can actually load,
/// so this errors rather than return an incomplete set. Returns
/// `[chosen]` unchanged for a file that isn't part of a split.
///
/// Groups by directory as well as name/total: some repos put the same
/// split basenames under multiple per-quantization subdirectories, and
/// grouping by basename alone could merge or miscount across them.
fn gguf_shards(models: &[HfFile], chosen: &HfFile) -> Result<Vec<HfFile>> {
    let Some((prefix, _, total)) = parse_gguf_shard(&chosen.path) else {
        return Ok(vec![chosen.clone()]);
    };
    let chosen_dir = dirname(&chosen.path);
    let mut shards: Vec<(u32, HfFile)> = models
        .iter()
        .filter_map(|f| {
            let (p, idx, t) = parse_gguf_shard(&f.path)?;
            (p == prefix && t == total && dirname(&f.path) == chosen_dir).then(|| (idx, f.clone()))
        })
        .collect();
    shards.sort_by_key(|(idx, _)| *idx);
    shards.dedup_by_key(|(idx, _)| *idx);
    if shards.len() as u32 != total {
        anyhow::bail!(
            "incomplete GGUF split {prefix:?}: found {} of {total} shards",
            shards.len()
        );
    }
    Ok(shards.into_iter().map(|(_, f)| f).collect())
}

/// Picks the best GGUF quant from the file listing, returning every
/// shard of a multi-part split together. `tag` is the user-supplied
/// quantization hint (e.g. "Q4_K_M") or empty for auto.
pub fn select_gguf(files: &[HfFile], tag: &str) -> Result<Vec<HfFile>> {
    let models: Vec<HfFile> = files
        .iter()
        .filter(|f| f.kind == "file" && is_model_gguf(&f.path))
        .cloned()
        .collect();
    if models.is_empty() {
        anyhow::bail!("no GGUF model files found in repository");
    }

    if !tag.is_empty() && tag != "latest" {
        let upper = tag.to_uppercase();
        if let Some(f) = models
            .iter()
            .find(|f| f.path.to_uppercase().contains(&upper))
        {
            return gguf_shards(&models, f);
        }
        let list: String = models.iter().map(|f| format!("  {}\n", f.path)).collect();
        anyhow::bail!("no GGUF file matching {tag:?} found; available:\n{list}");
    }

    for pref in QUANT_PREFERENCE {
        if let Some(f) = models.iter().find(|f| f.path.to_uppercase().contains(pref)) {
            return gguf_shards(&models, f);
        }
    }

    // Fallback: smallest file (most compressed).
    let smallest = models
        .iter()
        .min_by_key(|f| f.size)
        .expect("models is non-empty");
    gguf_shards(&models, smallest)
}

// ---------------------------------------------------------------------------
// Latent diffusion repositories (mirrors llama.cpp common/download.cpp)
// ---------------------------------------------------------------------------

fn is_weights_file(lower: &str) -> bool {
    lower.ends_with(".gguf") || lower.ends_with(".safetensors")
}

/// True when the repo ships a diffusion model: a transformer GGUF next to
/// VAE / text projection sidecars (e.g. `unsloth/LTX-2.3-GGUF`).
pub fn is_diffusion_repo(files: &[HfFile]) -> bool {
    files.iter().any(|f| {
        let name = basename_lower(&f.path);
        f.kind == "file"
            && is_weights_file(&name)
            && (name.contains("video_vae")
                || name.contains("_vae.")
                || name.contains("embeddings_connectors"))
    })
}

fn basename_lower(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_lowercase()
}

/// A Diffusers pipeline in safetensors (a root `model_index.json`, e.g.
/// `nvidia/Cosmos3-Edge`): pulled as safetensors, served by vLLM-Omni.
pub fn is_diffusers_repo(files: &[HfFile]) -> bool {
    files
        .iter()
        .any(|f| f.kind == "file" && f.path == crate::modelpack::DIFFUSERS_MODEL_INDEX)
        && files
            .iter()
            .any(|f| f.kind == "file" && basename_lower(&f.path).ends_with(".safetensors"))
}

/// Substrings of the `_class_name`s vLLM-Omni registers with
/// `final_output_type="video"` (its `diffusion/registry.py`), lowercased.
const VIDEO_PIPELINE_HINTS: &[&str] = &[
    "video",
    "omni",
    "cosmos",
    "wan",
    "ltx",
    "magi",
    "helios",
    "lingbot",
    "longcat",
    "minimaxh3",
];

/// A Diffusers pipeline's `outputTypes` from its `_class_name`: the video
/// families vLLM-Omni serves also make video, anything else only images.
pub fn diffusers_outputs(class_name: Option<&str>) -> Vec<&'static str> {
    let lower = class_name.unwrap_or("").to_lowercase();
    if VIDEO_PIPELINE_HINTS.iter().any(|h| lower.contains(h)) {
        vec!["image", "video"]
    } else {
        vec!["image"]
    }
}

/// [`diffusers_pipeline_class`] for a transfer, which keeps no layers:
/// fetches the index from the Hub. `None` on failure.
pub async fn fetch_diffusers_pipeline_class(
    client: &reqwest::Client,
    endpoint: &str,
    owner: &str,
    repo: &str,
    commit: &str,
    token: Option<&str>,
) -> Option<String> {
    let url = format!(
        "{endpoint}{owner}/{repo}/resolve/{commit}/{}",
        crate::modelpack::DIFFUSERS_MODEL_INDEX
    );
    pipeline_class(&get_json(client, &url, token).await.ok()?)
}

fn pipeline_class(index: &serde_json::Value) -> Option<String> {
    index.get("_class_name")?.as_str().map(str::to_string)
}

/// The `_class_name` of the pipeline index among `layers`, already
/// downloaded into the OCI layout at `layout_dir`.
pub fn diffusers_pipeline_class(
    layout_dir: &std::path::Path,
    layers: &[oci::Descriptor],
) -> Option<String> {
    let index = layers.iter().find(|d| {
        d.annotations
            .as_ref()
            .and_then(|a| a.get(oci::ANNOTATION_FILEPATH))
            .is_some_and(|p| p == crate::modelpack::DIFFUSERS_MODEL_INDEX)
    })?;
    let hex = index.digest.strip_prefix("sha256:")?;
    let bytes = std::fs::read(layout_dir.join("blobs").join("sha256").join(hex)).ok()?;
    pipeline_class(&serde_json::from_slice(&bytes).ok()?)
}

/// Like [`select_gguf`], but when no quant is requested prefers the fast
/// `distilled` variant that diffusion repos ship next to the `dev` one.
pub fn select_diffusion_gguf(files: &[HfFile], tag: &str) -> Result<Vec<HfFile>> {
    // sidecars may be GGUFs too
    let files: Vec<HfFile> = files
        .iter()
        .filter(|f| !is_diffusion_sidecar(&basename_lower(&f.path)))
        .cloned()
        .collect();
    if tag.is_empty() || tag == "latest" {
        // one file per transformer: a split cannot be served
        let models: Vec<&HfFile> = files
            .iter()
            .filter(|f| {
                f.kind == "file" && is_model_gguf(&f.path) && parse_gguf_shard(&f.path).is_none()
            })
            .collect();
        for pref in QUANT_PREFERENCE {
            let matching: Vec<&&HfFile> = models
                .iter()
                .filter(|f| f.path.to_uppercase().contains(pref))
                .collect();
            if let Some(f) = matching
                .iter()
                .find(|f| f.path.to_lowercase().contains("distilled"))
                .or(matching.first())
            {
                return Ok(vec![(**f).clone()]);
            }
        }
    }
    select_gguf(&files, tag)
}

fn is_diffusion_sidecar(name: &str) -> bool {
    ["vae", "connector", "text_encoder", "embeddings"]
        .iter()
        .any(|k| name.contains(k))
}

/// Everything a diffusion transformer needs next to it.
pub struct DiffusionPlan {
    /// `(role, file)` from the same repo: `vae`, `audio_vae`, `text_proj`.
    pub sidecars: Vec<(&'static str, HfFile)>,
    /// The `image` / `video` / `audio` capabilities the sidecars enable.
    pub outputs: Vec<&'static str>,
    /// `owner/repo:quant` of the text encoder the family was trained with.
    pub text_encoder: Option<&'static str>,
}

/// See llama.cpp's `common_download_get_hf_plan` for the same resolution.
pub fn diffusion_plan(files: &[HfFile], model_path: &str) -> DiffusionPlan {
    let pick = |k| select_diffusion_sidecar(files, model_path, k);
    let candidates = [
        (
            "vae",
            pick("video_vae")
                .or_else(|| pick("_vae.").filter(|f| !f.path.to_lowercase().contains("audio_vae"))),
        ),
        ("audio_vae", pick("audio_vae")),
        ("text_proj", pick("embeddings_connectors")),
    ];
    let mut plan = DiffusionPlan {
        sidecars: Vec::new(),
        outputs: Vec::new(),
        text_encoder: diffusion_default_text_encoder(model_path),
    };
    for (role, file) in candidates {
        let Some(file) = file else { continue };
        match role {
            "vae" => plan.outputs.extend(["image", "video"]),
            "audio_vae" => plan.outputs.push("audio"),
            _ => {}
        }
        plan.sidecars.push((role, file));
    }
    plan
}

/// Media type and annotations of a sidecar layer.
pub fn sidecar_layer(mut d: oci::Descriptor, file: &HfFile, role: &str) -> oci::Descriptor {
    d.media_type = safetensors_media_type(&file.path).to_string();
    let name = file
        .path
        .rsplit('/')
        .next()
        .unwrap_or(&file.path)
        .to_string();
    d.annotations = Some(BTreeMap::from([
        (oci::ANNOTATION_FILEPATH.to_string(), name),
        (oci::ANNOTATION_ROLE.to_string(), role.to_string()),
    ]));
    d
}

/// The sidecar whose file name contains `keyword` and shares the longest
/// prefix with the chosen transformer's file name — so the `distilled`
/// VAE goes with the `distilled` transformer.
pub fn select_diffusion_sidecar(
    files: &[HfFile],
    model_path: &str,
    keyword: &str,
) -> Option<HfFile> {
    let model_name = basename_lower(model_path);
    let mut best: Option<(usize, &HfFile)> = None;
    for f in files.iter().filter(|f| f.kind == "file") {
        let name = basename_lower(&f.path);
        if !is_weights_file(&name) || !name.contains(keyword) {
            continue;
        }
        let common = name
            .bytes()
            .zip(model_name.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        if best.is_none_or(|(c, _)| common > c) {
            best = Some((common, f));
        }
    }
    best.map(|(_, f)| f.clone())
}

/// The text encoder repository a diffusion model family is trained with.
/// LTX-2.x uses Gemma 3 12B; LTX-2.5 needs a fine-tuned Gemma 4 that
/// ships with the model, so there is no generic default for it.
pub fn diffusion_default_text_encoder(model_path: &str) -> Option<&'static str> {
    let name = basename_lower(model_path);
    if name.contains("ltx-2.5") || name.contains("ltx-2-5") {
        return None;
    }
    if name.contains("ltx-2") || name.contains("ltx2") {
        return Some("ggml-org/gemma-3-12b-it-GGUF:Q4_K_M");
    }
    None
}

const MMPROJ_PREFERENCE: &[&str] = &["F16", "BF16", "F32"];

/// Returns the repo's multimodal projector file, if it has one.
pub fn select_mmproj(files: &[HfFile]) -> Option<HfFile> {
    let candidates: Vec<&HfFile> = files
        .iter()
        .filter(|f| {
            f.kind == "file"
                && f.path.to_lowercase().contains("mmproj")
                && f.path.to_lowercase().ends_with(".gguf")
        })
        .collect();
    for pref in MMPROJ_PREFERENCE {
        for f in &candidates {
            let base = f.path.rsplit('/').next().unwrap_or(&f.path).to_uppercase();
            if base == format!("{pref}.GGUF") || base.ends_with(&format!("-{pref}.GGUF")) {
                return Some((*f).clone());
            }
        }
    }
    candidates.first().map(|f| (*f).clone())
}

const LICENSE_FILENAMES: &[&str] = &["LICENSE", "LICENSE.txt", "LICENSE.md"];

/// Returns the repo's root-level LICENSE file, if it has one.
pub fn select_license_file(files: &[HfFile]) -> Option<HfFile> {
    for want in LICENSE_FILENAMES {
        if let Some(f) = files
            .iter()
            .find(|f| f.kind == "file" && f.path.eq_ignore_ascii_case(want))
        {
            return Some(f.clone());
        }
    }
    None
}

/// Maps a file extension to the appropriate CNCF model layer media type
/// — mirrors `safetensorsMediaType`. ".jinja" is config, not doc: many
/// repos ship a standalone chat_template.jinja, and doc-type layers are
/// dropped before serving, which would silently hide the chat template.
pub fn safetensors_media_type(path: &str) -> &'static str {
    let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();
    match ext.as_str() {
        "safetensors" | "bin" | "pt" | "pth" => oci::MEDIA_TYPE_MODEL_WEIGHT_RAW,
        "json" | "model" | "txt" | "tiktoken" | "jinja" => oci::MEDIA_TYPE_MODEL_WEIGHT_CONFIG_RAW,
        _ => oci::MEDIA_TYPE_MODEL_DOC_RAW,
    }
}

/// True for files that belong in a local model directory. Deliberately
/// safetensors-only: a GGUF repo is selected by `select_gguf` in a
/// separate pass, so weight formats other than safetensors must not be
/// swept up here. `crate::sources::should_pack` is the wider equivalent
/// for sources that have no such second pass.
fn should_download_safetensors(path: &str) -> bool {
    let base = path.rsplit('/').next().unwrap_or(path).to_lowercase();
    if base.starts_with('.') {
        return false;
    }
    let ext = base.rsplit('.').next().unwrap_or("");
    if matches!(
        ext,
        "safetensors" | "bin" | "pt" | "pth" | "json" | "model" | "txt" | "tiktoken" | "jinja"
    ) {
        return true;
    }
    matches!(
        base.as_str(),
        "readme.md" | "license" | "licence" | "license.txt" | "licence.txt"
    )
}

/// Filters `files` down to the plain files `should_download_safetensors`
/// accepts, ignoring directories.
pub fn select_downloadable_hf_files(files: &[HfFile]) -> Vec<HfFile> {
    files
        .iter()
        .filter(|f| f.kind == "file" && should_download_safetensors(&f.path))
        .cloned()
        .collect()
}

/// Splits a (possibly `:latest`-normalized) HF reference
/// "host/owner/repo[:tag]" into its four components.
pub fn parse_hf_ref(reference: &str) -> Result<(String, String, String, String)> {
    let last_colon = reference.rfind(':').map(|i| i as isize).unwrap_or(-1);
    let last_slash = reference.rfind('/').map(|i| i as isize).unwrap_or(-1);
    let (rest, tag) = if last_colon > last_slash {
        (
            &reference[..last_colon as usize],
            reference[last_colon as usize + 1..].to_string(),
        )
    } else {
        (reference, String::new())
    };
    let parts: Vec<&str> = rest.splitn(3, '/').collect();
    let [host, owner, repo] = parts[..] else {
        anyhow::bail!("invalid HuggingFace reference {reference:?}: expected host/owner/repo");
    };
    Ok((host.to_string(), owner.to_string(), repo.to_string(), tag))
}

#[allow(dead_code)]
pub type Annotations = BTreeMap<String, String>;

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, size: i64) -> HfFile {
        HfFile {
            path: path.to_string(),
            size,
            kind: "file".to_string(),
        }
    }

    #[test]
    fn select_gguf_prefers_q4_k_m() {
        let files = vec![file("model-Q8_0.gguf", 100), file("model-Q4_K_M.gguf", 50)];
        let got = select_gguf(&files, "").unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].path, "model-Q4_K_M.gguf");
    }

    #[test]
    fn select_gguf_falls_back_to_smallest() {
        let files = vec![
            file("model-weird1.gguf", 100),
            file("model-weird2.gguf", 50),
        ];
        let got = select_gguf(&files, "").unwrap();
        assert_eq!(got[0].path, "model-weird2.gguf");
    }

    #[test]
    fn select_gguf_excludes_mmproj_and_imatrix() {
        let files = vec![
            file("mmproj-F16.gguf", 10),
            file("model.imatrix.gguf", 10),
            file("model-Q4_K_M.gguf", 50),
        ];
        let got = select_gguf(&files, "").unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].path, "model-Q4_K_M.gguf");
    }

    #[test]
    fn select_gguf_by_explicit_tag() {
        let files = vec![file("model-Q4_K_M.gguf", 50), file("model-Q8_0.gguf", 100)];
        let got = select_gguf(&files, "Q8_0").unwrap();
        assert_eq!(got[0].path, "model-Q8_0.gguf");
    }

    #[test]
    fn select_gguf_returns_every_shard_of_a_split_in_order() {
        let files = vec![
            file("model-Q4_K_M-00002-of-00003.gguf", 10),
            file("model-Q4_K_M-00001-of-00003.gguf", 10),
            file("model-Q4_K_M-00003-of-00003.gguf", 10),
            file("model-Q8_0-00001-of-00002.gguf", 10),
        ];
        let got = select_gguf(&files, "Q4_K_M").unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].path, "model-Q4_K_M-00001-of-00003.gguf");
        assert_eq!(got[1].path, "model-Q4_K_M-00002-of-00003.gguf");
        assert_eq!(got[2].path, "model-Q4_K_M-00003-of-00003.gguf");
    }

    #[test]
    fn select_gguf_errors_on_no_gguf_files() {
        let files = vec![file("config.json", 10)];
        assert!(select_gguf(&files, "").is_err());
    }

    #[test]
    fn select_mmproj_prefers_f16() {
        let files = vec![file("mmproj-BF16.gguf", 10), file("mmproj-F16.gguf", 10)];
        assert_eq!(select_mmproj(&files).unwrap().path, "mmproj-F16.gguf");
    }

    #[test]
    fn select_mmproj_absent_returns_none() {
        assert!(select_mmproj(&[file("model.gguf", 10)]).is_none());
    }

    #[test]
    fn select_license_file_case_insensitive() {
        let files = vec![file("license", 10)];
        assert!(select_license_file(&files).is_some());
    }

    #[test]
    fn normalize_spdx_license_maps_known_slugs() {
        assert_eq!(
            normalize_spdx_license("apache-2.0"),
            Some("Apache-2.0".to_string())
        );
        assert_eq!(normalize_spdx_license("other"), None);
        assert_eq!(normalize_spdx_license("unknown"), None);
        assert_eq!(
            normalize_spdx_license("some-custom-license"),
            Some("some-custom-license".to_string())
        );
    }

    #[test]
    fn parse_hf_ref_splits_host_owner_repo_tag() {
        let (host, owner, repo, tag) = parse_hf_ref("huggingface.co/owner/repo:Q4_K_M").unwrap();
        assert_eq!(
            (host.as_str(), owner.as_str(), repo.as_str(), tag.as_str()),
            ("huggingface.co", "owner", "repo", "Q4_K_M")
        );
    }

    #[test]
    fn parse_hf_ref_without_tag() {
        let (host, owner, repo, tag) = parse_hf_ref("huggingface.co/owner/repo").unwrap();
        assert_eq!(
            (host.as_str(), owner.as_str(), repo.as_str(), tag.as_str()),
            ("huggingface.co", "owner", "repo", "")
        );
    }

    #[test]
    fn parse_hf_ref_rejects_malformed_ref() {
        assert!(parse_hf_ref("not-enough-parts").is_err());
    }

    #[test]
    fn safetensors_media_type_classifies_chat_template_jinja_as_config() {
        assert_eq!(
            safetensors_media_type("chat_template.jinja"),
            oci::MEDIA_TYPE_MODEL_WEIGHT_CONFIG_RAW
        );
    }

    fn ltx_repo() -> Vec<HfFile> {
        vec![
            file("README.md", 10),
            file("ltx-2.3-22b-dev-Q4_K_M.gguf", 14),
            file("ltx-2.3-22b-dev-Q8_0.gguf", 22),
            file("distilled/ltx-2.3-22b-distilled-Q4_K_M.gguf", 14),
            file("distilled-1.1/ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf", 14),
            file("vae/ltx-2.3-22b-dev_video_vae.safetensors", 1),
            file("vae/ltx-2.3-22b-distilled_video_vae.safetensors", 1),
            file("vae/ltx-2.3-22b-dev_audio_vae.safetensors", 1),
            file("vae/ltx-2.3-22b-distilled_audio_vae.safetensors", 1),
            file(
                "text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors",
                2,
            ),
            file(
                "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
                2,
            ),
        ]
    }

    #[test]
    fn diffusion_repo_is_detected_by_its_sidecars() {
        assert!(is_diffusion_repo(&ltx_repo()));
        assert!(!is_diffusion_repo(&[
            file("model-Q4_K_M.gguf", 1),
            file("mmproj-F16.gguf", 1)
        ]));
    }

    #[test]
    fn diffusion_gguf_prefers_distilled_without_a_tag() {
        let picked = select_diffusion_gguf(&ltx_repo(), "").unwrap();
        assert_eq!(picked.len(), 1);
        assert!(picked[0].path.contains("distilled"), "{}", picked[0].path);
        assert!(picked[0].path.contains("Q4_K_M"));
        // an explicit tag still wins
        let dev = select_diffusion_gguf(&ltx_repo(), "dev-Q8_0").unwrap();
        assert_eq!(dev[0].path, "ltx-2.3-22b-dev-Q8_0.gguf");
    }

    #[test]
    fn diffusion_sidecars_follow_the_transformer_variant() {
        let files = ltx_repo();
        let model = "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf";
        let vae = select_diffusion_sidecar(&files, model, "video_vae").unwrap();
        assert_eq!(vae.path, "vae/ltx-2.3-22b-distilled_video_vae.safetensors");
        let conn = select_diffusion_sidecar(&files, model, "embeddings_connectors").unwrap();
        assert_eq!(
            conn.path,
            "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
        );
        let dev_vae =
            select_diffusion_sidecar(&files, "ltx-2.3-22b-dev-Q8_0.gguf", "audio_vae").unwrap();
        assert_eq!(dev_vae.path, "vae/ltx-2.3-22b-dev_audio_vae.safetensors");
        assert!(select_diffusion_sidecar(&files, model, "nothing").is_none());
    }

    #[test]
    fn ltx2_defaults_to_gemma3_text_encoder() {
        assert_eq!(
            diffusion_default_text_encoder("distilled-1.1/ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"),
            Some("ggml-org/gemma-3-12b-it-GGUF:Q4_K_M")
        );
        assert_eq!(
            diffusion_default_text_encoder("ltx-2.5-22b-dev-Q8_0.gguf"),
            None
        );
        assert_eq!(diffusion_default_text_encoder("flux-dev-Q8_0.gguf"), None);
    }

    /// `nvidia/Cosmos3-Edge`'s listing (weights only; assets elided).
    fn cosmos3_repo() -> Vec<HfFile> {
        vec![
            file("README.md", 10),
            file("config.json", 1),
            file("model_index.json", 1),
            file("model.safetensors.index.json", 1),
            file("scheduler/scheduler_config.json", 1),
            file("transformer/config.json", 1),
            file(
                "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
                5,
            ),
            file(
                "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
                2,
            ),
            file("vae/config.json", 1),
            file("vae/diffusion_pytorch_model.safetensors", 1),
            file("vision_encoder/model.safetensors", 1),
            file("assets/example_i2v_input.jpg", 1),
        ]
    }

    #[test]
    fn diffusers_repo_is_detected_by_its_root_pipeline_index() {
        let files = cosmos3_repo();
        assert!(is_diffusers_repo(&files));
        // Not the GGUF-sidecar kind: nothing in it is named like an LTX VAE,
        // so it must fall through to the safetensors pull, nested paths intact.
        assert!(!is_diffusion_repo(&files));
        assert!(select_gguf(&files, "").is_err());
        let picked = select_downloadable_hf_files(&files);
        let paths: Vec<&str> = picked.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"model_index.json"));
        assert!(paths.contains(&"vae/diffusion_pytorch_model.safetensors"));
        assert!(paths.contains(&"transformer/config.json"));
        assert!(!paths.contains(&"assets/example_i2v_input.jpg"));
        // A plain LLM repo, and an index without weights, are not one.
        assert!(!is_diffusers_repo(&[
            file("config.json", 1),
            file("model.safetensors", 1)
        ]));
        assert!(!is_diffusers_repo(&[file("model_index.json", 1)]));
        // A nested index does not make the repo a pipeline.
        assert!(!is_diffusers_repo(&[
            file("model.safetensors", 1),
            file("demo/model_index.json", 1)
        ]));
    }

    #[test]
    fn diffusers_outputs_follow_the_pipeline_class() {
        assert_eq!(
            diffusers_outputs(Some("Cosmos3OmniPipeline")),
            vec!["image", "video"]
        );
        for video in [
            "WanPipeline",
            "LTX2Pipeline",
            "HunyuanVideo15Pipeline",
            "Magi2Pipeline",
        ] {
            assert_eq!(
                diffusers_outputs(Some(video)),
                vec!["image", "video"],
                "{video}"
            );
        }
        for image in ["QwenImagePipeline", "FluxPipeline", "LancePipeline"] {
            assert_eq!(diffusers_outputs(Some(image)), vec!["image"], "{image}");
        }
        assert_eq!(
            diffusers_outputs(None),
            vec!["image"],
            "unknown: at least an image model"
        );
    }
}

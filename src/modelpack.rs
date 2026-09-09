//! Reusable "resolve a local OCI-store reference to servable model files"
//! logic — the CNCF ModelPack (<https://github.com/modelpack/model-spec>)
//! equivalent of `huggingface_hub.snapshot_download`.
//!
//! Originally private to `cmd::serve` (which uses it to decide whether to
//! spawn `llama-server`, `vllm`, or (Apple Silicon macOS) `mlx_lm.server`
//! as its backend for a given model), this
//! module is `pub` so it also backs `cmd::resolve` (`llmman resolve`) — a
//! standalone, scriptable entry point that other tools (e.g. a vLLM plugin
//! that wants vLLM itself, not `llmman`, to be the one serving the model)
//! can shell out to, without needing `llmman serve`'s HTTP daemon or its
//! opinions about which inference backend to launch.
//!
//! Everything here assumes the reference has already been pulled into the
//! local `OciStore` at `store_path` (see `crate::ffi::pull`) — this module
//! only resolves+extracts, it never talks to a registry itself.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};

use crate::storage::OciStore;

const HF_GGUF_MEDIA_TYPE: &str = "application/vnd.docker.ai.gguf.v3";

/// What kind of model did we find in the OCI store?
pub enum ModelPath {
    /// A GGUF file — serve with llama-server. The second field, when
    /// present, is a companion `--mmproj` projector GGUF (see
    /// [`is_mmproj_layer`]) needed for vision/audio support.
    Gguf(PathBuf, Option<PathBuf>),
    /// A safetensors directory — serve with vllm, or (Apple Silicon
    /// macOS) `mlx_lm.server` — see `cmd::serve::backend::use_mlx_for_safetensors`.
    SafeTensors(PathBuf),
    /// A latent diffusion model (image / video / audio generation) —
    /// served in-process by `crate::mediagen` from the transformer GGUF
    /// plus the sidecars pulled next to it (see
    /// `crate::hf::oci::ANNOTATION_ROLE`).
    Diffusion(DiffusionPaths),
    /// A Diffusers-layout safetensors directory (a root `model_index.json`,
    /// weights in `transformer/`, `vae/`, ...) such as `nvidia/Cosmos3-Edge`
    /// — served by vLLM-Omni (`vllm serve --omni`); plain vllm cannot.
    Omni(PathBuf),
}

/// The Diffusers pipeline index that marks a [`ModelPath::Omni`] repo.
pub const DIFFUSERS_MODEL_INDEX: &str = "model_index.json";

/// The files of a resolved [`ModelPath::Diffusion`] model.
#[derive(Debug, Clone, Default)]
pub struct DiffusionPaths {
    /// The diffusion transformer GGUF (`--model`).
    pub model: PathBuf,
    /// Video VAE (`--vae`).
    pub vae: Option<PathBuf>,
    /// Audio VAE + vocoder (`--audio-vae`).
    pub audio_vae: Option<PathBuf>,
    /// Text embedding projection / connectors (`--text-proj`).
    pub text_proj: Option<PathBuf>,
    /// Text encoder GGUF (`--text-encoder`).
    pub text_encoder: Option<PathBuf>,
    /// Every raw layer by its `org.cncf.model.filepath` (tokenizer, configs, ...).
    pub files: std::collections::BTreeMap<String, PathBuf>,
}

impl ModelPath {
    /// The local filesystem path this variant resolved to — either a
    /// single `.gguf` file, or the model directory (parent of
    /// `config.json`) for a safetensors checkout.
    pub fn path(&self) -> &Path {
        match self {
            ModelPath::Gguf(p, _) => p,
            ModelPath::SafeTensors(p) | ModelPath::Omni(p) => p,
            ModelPath::Diffusion(d) => &d.model,
        }
    }

    /// The companion `--mmproj` projector file resolved alongside a
    /// `Gguf` model, if any — always `None` for `SafeTensors` (neither
    /// vllm nor mlx_lm.server has an equivalent separate-projector-file
    /// convention).
    pub fn mmproj(&self) -> Option<&Path> {
        match self {
            ModelPath::Gguf(_, mmproj) => mmproj.as_deref(),
            ModelPath::SafeTensors(_) | ModelPath::Diffusion(_) | ModelPath::Omni(_) => None,
        }
    }

    /// A short, stable string identifying which variant this is — used by
    /// `cmd::resolve`'s JSON output and any other consumer that wants to
    /// branch on format without matching the enum directly.
    pub fn format(&self) -> &'static str {
        match self {
            ModelPath::Gguf(..) => "gguf",
            ModelPath::SafeTensors(_) => "safetensors",
            ModelPath::Diffusion(_) => "diffusion",
            ModelPath::Omni(_) => "omni",
        }
    }
}

/// Splits an OCI digest ("sha256:abcd...") down to just its hex portion,
/// which is what the blob store's on-disk layout uses as the filename.
fn digest_hex(digest: &str) -> anyhow::Result<&str> {
    digest
        .split_once(':')
        .map(|(_, hex)| hex)
        .filter(|h| h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()))
        .ok_or_else(|| anyhow!("malformed digest: {digest}"))
}

fn layer_filepath(l: &crate::storage::oci::Descriptor) -> Option<&str> {
    l.annotations.as_ref().and_then(|a| {
        a.get("org.cncf.model.filepath")
            .or_else(|| a.get("org.opencontainers.image.title"))
            .map(|s| s.as_str())
    })
}

/// The diffusion sidecar role of a layer — see
/// `crate::hf::oci::ANNOTATION_ROLE`.
fn layer_role(l: &crate::storage::oci::Descriptor) -> Option<&str> {
    l.annotations
        .as_ref()
        .and_then(|a| a.get(crate::hf::oci::ANNOTATION_ROLE))
        .map(|s| s.as_str())
}

fn is_gguf_layer(l: &crate::storage::oci::Descriptor) -> bool {
    if l.media_type == HF_GGUF_MEDIA_TYPE {
        return true;
    }
    layer_filepath(l)
        .map(|p| p.to_lowercase().ends_with(".gguf"))
        .unwrap_or(false)
}

fn is_safetensors_layer(l: &crate::storage::oci::Descriptor) -> bool {
    layer_filepath(l)
        .map(|p| p.to_lowercase().ends_with(".safetensors"))
        .unwrap_or(false)
}

/// The layer holding the repo's *root* `model_index.json` (a Diffusers
/// pipeline index; a nested one is some vendored pipeline's, not this
/// repo's).
fn is_diffusers_index_layer(l: &crate::storage::oci::Descriptor) -> bool {
    layer_filepath(l) == Some(DIFFUSERS_MODEL_INDEX)
}

/// A Diffusers-layout repo: a root `model_index.json` next to safetensors.
fn is_diffusers_manifest(manifest: &crate::storage::oci::Manifest) -> bool {
    manifest.layers.iter().any(is_diffusers_index_layer)
        && manifest.layers.iter().any(is_safetensors_layer)
}

/// True if a (already-confirmed-GGUF) layer looks like a multimodal
/// projector rather than the main model — matched by filename containing
/// "mmproj", the de facto convention GGUF repos use, since there's no
/// media-type or metadata distinction to key off instead.
fn is_mmproj_layer(l: &crate::storage::oci::Descriptor) -> bool {
    layer_filepath(l)
        .map(|p| p.to_lowercase().contains("mmproj"))
        .unwrap_or(false)
}

// gguf_architecture/gguf_context_length_override (a GGUF metadata reader
// + --override-kv builder that let --ctx-size force a context above a
// model's own trained length) were tried and removed: llama-server's own
// capping of --ctx-size back down to a model's trained context — see
// cmd::serve::config::context_length_from_env's doc comment — is deliberate, not
// a bug to work around. Defeating that safety net via --override-kv
// produces a real NaN/incoherent-output risk for out-of-distribution
// RoPE positions that llama-server's own warning exists to prevent, for
// a use case (fitting a real agent's system prompt) that a model
// whose trained context is that tight was never going to serve well
// regardless — see docker/sandboxes' own llmmanCtxSize doc comment for the
// model-selection fix that replaced this instead.

/// Extracts a single GGUF layer to a local path, caching under
/// `cache_path` — shared by [`resolve_model`] for both the primary model
/// GGUF and, when present, a companion `--mmproj` GGUF (see
/// [`is_mmproj_layer`]), since both are extracted exactly the same way.
fn extract_gguf_layer(
    store: &OciStore,
    store_path: &Path,
    cache_path: &Path,
    layer: &crate::storage::oci::Descriptor,
) -> anyhow::Result<PathBuf> {
    let title = layer_filepath(layer).unwrap_or("model.gguf").to_owned();
    let layer_hex = digest_hex(&layer.digest)?;

    // HF blobs are stored as raw GGUF — use directly. The same goes for
    // the CNCF raw-weight layers `crate::hf::pull` writes: copying a
    // multi-GiB GGUF into the cache would double its disk footprint for
    // nothing.
    if layer.media_type == HF_GGUF_MEDIA_TYPE
        || layer.media_type == crate::hf::oci::MEDIA_TYPE_MODEL_WEIGHT_RAW
    {
        let blob_path = store_path.join("blobs").join("sha256").join(layer_hex);
        if blob_path.exists() && blob_is_gguf(&blob_path) {
            eprintln!("[llmman] using blob directly: {}", blob_path.display());
            return Ok(blob_path);
        }
    }

    // Otherwise extract from tar layer.
    let cached_dir = cache_path.join(layer_hex);
    if cached_dir.exists() {
        for e in std::fs::read_dir(&cached_dir)?.flatten() {
            let p = e.path();
            if p.extension().and_then(|e| e.to_str()) == Some("gguf") {
                return Ok(p);
            }
        }
    }
    std::fs::create_dir_all(&cached_dir)?;
    let blob = store
        .read_blob(&layer.digest)
        .with_context(|| format!("read blob {}", layer.digest))?;
    if blob.len() >= 4 && &blob[..4] == b"GGUF" {
        let name = Path::new(&title)
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("model.gguf"));
        let p = cached_dir.join(name);
        std::fs::write(&p, &blob)?;
        return Ok(p);
    }
    let mut archive = tar::Archive::new(std::io::Cursor::new(&blob));
    for entry in archive.entries()? {
        let mut entry = entry?;
        let ep = entry.path()?.to_path_buf();
        if ep.extension().and_then(|e| e.to_str()) == Some("gguf") {
            let name = ep
                .file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("model.gguf"));
            let d = cached_dir.join(name);
            entry.unpack(&d)?;
            return Ok(d);
        }
    }
    Err(anyhow!("no .gguf in tar layer {}", layer.digest))
}

/// True if the file at `path` starts with the GGUF magic.
fn blob_is_gguf(path: &Path) -> bool {
    use std::io::Read as _;
    let mut magic = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map(|()| &magic == b"GGUF")
        .unwrap_or(false)
}

/// The content-addressed blob of a raw (non-tar) layer, used as-is.
fn raw_blob_path(
    store_path: &Path,
    layer: &crate::storage::oci::Descriptor,
) -> anyhow::Result<PathBuf> {
    let p = store_path
        .join("blobs")
        .join("sha256")
        .join(digest_hex(&layer.digest)?);
    if !p.exists() {
        anyhow::bail!(
            "missing blob {} for layer {:?}",
            layer.digest,
            layer_filepath(layer)
        );
    }
    Ok(p)
}

/// A raw layer's blob under its original file name, as a symlink
/// `<cache>/<digest>/<filename>`: the name carries the variant
/// (`distilled` selects the sampling schedule).
fn named_blob_path(
    store_path: &Path,
    cache_path: &Path,
    layer: &crate::storage::oci::Descriptor,
) -> anyhow::Result<PathBuf> {
    let blob = raw_blob_path(store_path, layer)?;
    let Some(name) = layer_filepath(layer).and_then(|p| Path::new(p).file_name()) else {
        return Ok(blob);
    };
    // absolute: LLMMAN_MODELS may be relative
    let blob = blob.canonicalize()?;
    let dir = cache_path.join(digest_hex(&layer.digest)?);
    std::fs::create_dir_all(&dir)?;
    let link = dir.join(name);
    if !link.exists() {
        // a link left dangling by blob GC
        let _ = std::fs::remove_file(&link);
        #[cfg(unix)]
        std::os::unix::fs::symlink(&blob, &link)
            .with_context(|| format!("link {} -> {}", link.display(), blob.display()))?;
        #[cfg(not(unix))]
        std::fs::hard_link(&blob, &link)
            .with_context(|| format!("link {} -> {}", link.display(), blob.display()))?;
    }
    Ok(link)
}

/// Ollama's `model.Capability` values reported in `/api/show`'s
/// `capabilities` array (`ollama run` reads them to set `opts.MultiModal`).
pub const CAPABILITY_COMPLETION: &str = "completion";
pub const CAPABILITY_VISION: &str = "vision";
/// Generates images (a latent diffusion model) — ollama's
/// `CapabilityImage`. Such a model has no `"completion"`.
pub const CAPABILITY_IMAGE: &str = "image";
pub const CAPABILITY_VIDEO: &str = "video";
pub const CAPABILITY_AUDIO: &str = "audio";

/// The part of ollama's `Model.Capabilities()` (server/images.go) a
/// manifest alone can answer, without extracting or opening a GGUF:
/// `"completion"` always, plus `"vision"` when a companion mmproj layer
/// is present (`projectorCapabilities`) or the CNCF config's
/// `config.capabilities.inputTypes` lists `"image"` (`configCapabilities`).
pub fn capabilities(store: &OciStore, manifest: &crate::storage::oci::Manifest) -> Vec<String> {
    // Shape written by crate::hf::oci::build_cncf_manifest.
    let config: Option<serde_json::Value> = store
        .read_blob(&manifest.config.digest)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok());
    let config_types = |key: &str| -> Vec<String> {
        config
            .as_ref()
            .and_then(|v| {
                v.get("config")?
                    .get("capabilities")?
                    .get(key)?
                    .as_array()
                    .cloned()
            })
            .unwrap_or_default()
            .iter()
            .filter_map(|t| t.as_str().map(str::to_string))
            .collect()
    };

    // A diffusion model generates media instead of text: its capabilities
    // are exactly the output types it was pulled with. A Diffusers-layout
    // repo pulled before those were recorded still generates images.
    let outputs = config_types("outputTypes");
    if outputs.iter().any(|t| t == CAPABILITY_IMAGE)
        || manifest.layers.iter().any(|l| layer_role(l).is_some())
        || is_diffusers_manifest(manifest)
    {
        let mut caps: Vec<String> = outputs
            .into_iter()
            .filter(|t| t == CAPABILITY_IMAGE || t == CAPABILITY_VIDEO || t == CAPABILITY_AUDIO)
            .collect();
        if caps.is_empty() {
            caps.push(CAPABILITY_IMAGE.to_string());
        }
        return caps;
    }

    let mut caps = vec![CAPABILITY_COMPLETION.to_string()];

    let has_mmproj = gguf_layers(manifest).is_some_and(|(_, mmproj)| mmproj.is_some());
    let config_says_image = config_types("inputTypes").iter().any(|t| t == "image");

    if has_mmproj || config_says_image {
        caps.push(CAPABILITY_VISION.to_string());
    }
    caps
}

/// The manifest's primary GGUF layer and its companion mmproj layer, if
/// any. Prefers a non-mmproj-named layer as primary so an mmproj file
/// that sorts first isn't taken for the model; falls back to the first
/// layer if every one looks like mmproj.
fn gguf_layers(
    manifest: &crate::storage::oci::Manifest,
) -> Option<(
    &crate::storage::oci::Descriptor,
    Option<&crate::storage::oci::Descriptor>,
)> {
    let layers: Vec<&crate::storage::oci::Descriptor> = manifest
        .layers
        .iter()
        .filter(|l| is_gguf_layer(l) && layer_role(l).is_none())
        .collect();
    let primary = *layers
        .iter()
        .find(|l| !is_mmproj_layer(l))
        .or(layers.first())?;
    let mmproj = layers
        .iter()
        .copied()
        .find(|l| is_mmproj_layer(l) && l.digest != primary.digest);
    Some((primary, mmproj))
}

/// Which [`ModelPath`] variant a manifest resolves to — see [`stored_format`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Gguf,
    SafeTensors,
    Diffusion,
    /// Diffusers-layout safetensors — see [`ModelPath::Omni`].
    Omni,
}

/// [`resolve_model`]'s classification: diffusion > GGUF > Diffusers
/// (omni) > safetensors, `None` for "no servable model layer".
fn manifest_format(manifest: &crate::storage::oci::Manifest) -> Option<ModelFormat> {
    if manifest.layers.iter().any(|l| layer_role(l).is_some()) {
        Some(ModelFormat::Diffusion)
    } else if gguf_layers(manifest).is_some() {
        Some(ModelFormat::Gguf)
    } else if is_diffusers_manifest(manifest) {
        Some(ModelFormat::Omni)
    } else if manifest.layers.iter().any(is_safetensors_layer) {
        Some(ModelFormat::SafeTensors)
    } else {
        None
    }
}

/// The "nothing servable" error, naming the file extensions that were there.
fn no_servable_layer(model_ref: &str, manifest: &crate::storage::oci::Manifest) -> anyhow::Error {
    let exts: std::collections::HashSet<String> = manifest
        .layers
        .iter()
        .filter_map(|l| layer_filepath(l))
        .filter_map(|p| Path::new(p).extension()?.to_str().map(|e| e.to_lowercase()))
        .collect();
    if exts.is_empty() {
        anyhow!("no servable model layer found in {model_ref}")
    } else {
        anyhow!(
            "no servable model layer in {model_ref} — found {exts:?} files; \
             llmman serve supports GGUF (llama-server) and safetensors (vllm/vllm-omni/mlx)"
        )
    }
}

/// What [`resolve_model`] would resolve `model_ref` to, read off its
/// manifest without extracting anything (for `--pull-oci`, which only
/// needs to know which engine's image to pull).
pub fn stored_format(store_path: &Path, model_ref: &str) -> anyhow::Result<ModelFormat> {
    let store = OciStore::open(store_path)?;
    let desc = store
        .find(model_ref)
        .with_context(|| format!("model not found in store: {model_ref}"))?;
    let manifest = store.read_manifest(&desc.digest)?;
    manifest_format(&manifest).ok_or_else(|| no_servable_layer(model_ref, &manifest))
}

/// Resolve `model_ref` (already present in the `OciStore` at `store_path`)
/// to either a `.gguf` file or an extracted safetensors directory, caching
/// any extraction under `cache_path`.
pub fn resolve_model(
    store_path: &Path,
    cache_path: &Path,
    model_ref: &str,
) -> anyhow::Result<ModelPath> {
    let store = OciStore::open(store_path)?;
    let desc = store
        .find(model_ref)
        .with_context(|| format!("model not found in store: {model_ref}"))?;
    let manifest = store.read_manifest(&desc.digest)?;

    let Some(format) = manifest_format(&manifest) else {
        return Err(no_servable_layer(model_ref, &manifest));
    };

    // ── diffusion → crate::mediagen ───────────────────────────────────────
    if format == ModelFormat::Diffusion {
        let (primary, _) = gguf_layers(&manifest)
            .ok_or_else(|| anyhow!("{model_ref}: diffusion model has no transformer GGUF layer"))?;
        // Every file keeps its name (see named_blob_path); a GGUF in a tar
        // layer is extracted.
        let named = |l: &crate::storage::oci::Descriptor| -> anyhow::Result<PathBuf> {
            let raw = raw_blob_path(store_path, l);
            if is_gguf_layer(l) && !raw.as_ref().is_ok_and(|p| blob_is_gguf(p)) {
                extract_gguf_layer(&store, store_path, cache_path, l)
            } else if raw.is_ok()
                && (is_gguf_layer(l) || l.media_type == crate::hf::oci::MEDIA_TYPE_MODEL_WEIGHT_RAW)
            {
                named_blob_path(store_path, cache_path, l)
            } else {
                anyhow::bail!(
                    "{model_ref}: unsupported layer {} ({})",
                    l.digest,
                    l.media_type
                )
            }
        };
        let mut paths = DiffusionPaths {
            model: named(primary)?,
            ..Default::default()
        };
        for l in &manifest.layers {
            // raw layers only: a tar layer's blob is the archive, not the file
            if let (Some(fp), true) = (layer_filepath(l), l.media_type.ends_with(".raw")) {
                paths
                    .files
                    .insert(fp.to_string(), named_blob_path(store_path, cache_path, l)?);
            }
            let Some(role) = layer_role(l) else { continue };
            let p = named(l)?;
            match role {
                "vae" => paths.vae = Some(p),
                "audio_vae" => paths.audio_vae = Some(p),
                "text_proj" => paths.text_proj = Some(p),
                "text_encoder" => paths.text_encoder = Some(p),
                // Cosmos3 sidecars its generation path does not use
                "vision_encoder" | "sound_tokenizer" => {}
                other => {
                    eprintln!("[llmman] {model_ref}: ignoring layer with unknown role {other:?}")
                }
            }
        }
        return Ok(ModelPath::Diffusion(paths));
    }

    // ── GGUF → llama-server ────────────────────────────────────────────────
    if let Some((primary, mmproj)) = gguf_layers(&manifest) {
        let primary_path = extract_gguf_layer(&store, store_path, cache_path, primary)?;
        let mmproj_path = mmproj
            .map(|l| extract_gguf_layer(&store, store_path, cache_path, l))
            .transpose()
            .context("extracting companion mmproj file")?;
        if mmproj_path.is_some() {
            eprintln!("[llmman] {model_ref}: found companion mmproj file");
        }
        return Ok(ModelPath::Gguf(primary_path, mmproj_path));
    }

    // ── safetensors → vllm / vllm-omni / mlx_lm.server ──────────────────
    let model_dir = extract_safetensors_dir(store_path, cache_path, &desc.digest, &manifest)?;
    Ok(match format {
        ModelFormat::Omni => ModelPath::Omni(model_dir),
        _ => {
            debug_assert_eq!(format, ModelFormat::SafeTensors);
            ModelPath::SafeTensors(model_dir)
        }
    })
}

/// The model directory of an extracted checkout: the parent of the
/// shallowest `config.json` or `model_index.json` (a Diffusers repo has a
/// `config.json` per component, so "first in layer order" is wrong), else
/// the checkout root.
fn safetensors_model_dir(cache_dir: &Path, rel_paths: &[&str]) -> PathBuf {
    rel_paths
        .iter()
        .filter(|p| {
            Path::new(p)
                .file_name()
                .is_some_and(|n| n == "config.json" || n == DIFFUSERS_MODEL_INDEX)
        })
        .min_by_key(|p| p.matches('/').count())
        .and_then(|p| Path::new(p).parent())
        // `join("")` would leave a trailing separator on the root case.
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(|parent| cache_dir.join(parent))
        .unwrap_or_else(|| cache_dir.to_path_buf())
}

/// Extract CNCF-format safetensors layers to a cache directory and return the
/// model directory (see [`safetensors_model_dir`]).
fn extract_safetensors_dir(
    store_path: &Path,
    cache_path: &Path,
    manifest_digest: &str,
    manifest: &crate::storage::oci::Manifest,
) -> anyhow::Result<PathBuf> {
    let hex = digest_hex(manifest_digest)?;
    let cache_dir = cache_path.join(hex);

    for layer in &manifest.layers {
        // Only extract config and weight files; skip code/docs.
        let include = matches!(
            layer.media_type.as_str(),
            "application/vnd.cncf.model.weight.config.v1.raw"
                | "application/vnd.cncf.model.weight.v1.raw"
        );
        if !include {
            continue;
        }

        let Some(rel_path) = layer_filepath(layer) else {
            continue;
        };
        // The annotation came from whatever produced the manifest, so it
        // is not this process's to trust: joining "../../id_rsa" onto
        // cache_dir would escape it and overwrite an arbitrary file.
        // `crate::sources` rejects these at pack time; this covers a
        // layer already in the store, or pulled by any other path.
        if !crate::sources::is_safe_relative_path(rel_path) {
            eprintln!("[llmman] skipping layer with unsafe filepath {rel_path:?}");
            continue;
        }
        let dest = cache_dir.join(rel_path);
        // exists() is not complete: a killed copy can leave a short dest.
        if dest
            .metadata()
            .map(|m| m.is_file() && m.len() == layer.size)
            .unwrap_or(false)
        {
            continue;
        }

        std::fs::create_dir_all(dest.parent().context("no parent")?)?;
        let layer_hex = digest_hex(&layer.digest)?;
        let blob = store_path.join("blobs").join("sha256").join(layer_hex);
        std::fs::copy(&blob, &dest).with_context(|| format!("copy {rel_path} from blob store"))?;
        eprintln!("[llmman] extracted {rel_path}");
    }

    let rel_paths: Vec<&str> = manifest
        .layers
        .iter()
        .filter_map(layer_filepath)
        .filter(|p| crate::sources::is_safe_relative_path(p))
        .collect();
    Ok(safetensors_model_dir(&cache_dir, &rel_paths))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_matches_variant() {
        assert_eq!(
            ModelPath::Gguf(PathBuf::from("/x/m.gguf"), None).format(),
            "gguf"
        );
        assert_eq!(
            ModelPath::SafeTensors(PathBuf::from("/x")).format(),
            "safetensors"
        );
    }

    #[test]
    fn path_returns_inner_pathbuf() {
        let p = ModelPath::SafeTensors(PathBuf::from("/models/foo"));
        assert_eq!(p.path(), Path::new("/models/foo"));
    }

    #[test]
    fn mmproj_is_none_unless_explicitly_set() {
        let no_mmproj = ModelPath::Gguf(PathBuf::from("/x/m.gguf"), None);
        assert_eq!(no_mmproj.mmproj(), None);

        let with_mmproj = ModelPath::Gguf(
            PathBuf::from("/x/m.gguf"),
            Some(PathBuf::from("/x/mmproj-f16.gguf")),
        );
        assert_eq!(with_mmproj.mmproj(), Some(Path::new("/x/mmproj-f16.gguf")));

        // SafeTensors (vllm/mlx) has no equivalent separate-projector-file
        // convention.
        assert_eq!(ModelPath::SafeTensors(PathBuf::from("/x")).mmproj(), None);
    }

    fn descriptor(digest: &str, filepath: &str) -> crate::storage::oci::Descriptor {
        let mut ann = std::collections::HashMap::new();
        ann.insert("org.cncf.model.filepath".to_string(), filepath.to_string());
        crate::storage::oci::Descriptor {
            media_type: "application/vnd.cncf.model.weight.v1.tar".into(),
            digest: digest.to_string(),
            size: 123,
            annotations: Some(ann),
        }
    }

    #[test]
    fn is_mmproj_layer_matches_filename_regardless_of_case_or_position() {
        assert!(is_mmproj_layer(&descriptor("sha256:a", "mmproj-F16.gguf")));
        assert!(is_mmproj_layer(&descriptor(
            "sha256:b",
            "Qwen3-VL-mmproj.gguf"
        )));
        assert!(!is_mmproj_layer(&descriptor(
            "sha256:c",
            "model.Q4_K_M.gguf"
        )));
    }

    fn manifest_with(
        layers: Vec<crate::storage::oci::Descriptor>,
    ) -> (OciStore, crate::storage::oci::Manifest) {
        // parallel tests can share a clock tick; the counter keeps their stores apart
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "llmman-modelpack-caps-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let store = OciStore::open(&dir).unwrap();
        let config = store
            .write_blob("application/vnd.cncf.model.config.v1+json", b"{}")
            .unwrap();
        let manifest = crate::storage::oci::Manifest {
            schema_version: 2,
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            artifact_type: None,
            config,
            layers,
            annotations: None,
        };
        (store, manifest)
    }

    #[test]
    fn capabilities_is_completion_only_without_a_projector() {
        let (store, m) = manifest_with(vec![descriptor("sha256:a", "model.Q4_K_M.gguf")]);
        assert_eq!(capabilities(&store, &m), vec!["completion"]);
    }

    #[test]
    fn capabilities_adds_vision_when_a_companion_mmproj_layer_is_present() {
        // Mirrors ollama's projectorCapabilities: any projector ⇒ vision.
        let (store, m) = manifest_with(vec![
            descriptor("sha256:a", "mmproj-F16.gguf"),
            descriptor("sha256:b", "model.Q4_K_M.gguf"),
        ]);
        assert_eq!(capabilities(&store, &m), vec!["completion", "vision"]);
    }

    #[test]
    fn capabilities_a_lone_mmproj_named_gguf_is_the_model_not_a_projector() {
        let (store, m) = manifest_with(vec![descriptor("sha256:a", "mmproj-only.gguf")]);
        assert_eq!(capabilities(&store, &m), vec!["completion"]);
    }

    #[test]
    fn capabilities_two_mmproj_named_ggufs_load_with_a_projector_so_report_vision() {
        // resolve_model's fallback: first is primary, second is its mmproj.
        let (store, m) = manifest_with(vec![
            descriptor("sha256:a", "mmproj-model.gguf"),
            descriptor("sha256:b", "mmproj-F16.gguf"),
        ]);
        assert_eq!(capabilities(&store, &m), vec!["completion", "vision"]);
    }

    #[test]
    fn capabilities_honours_cncf_config_input_types_image() {
        // The exact shape hf::oci::build_cncf_manifest writes.
        let (store, mut m) = manifest_with(vec![descriptor("sha256:a", "model.gguf")]);
        m.config = store
            .write_blob(
                "application/vnd.cncf.model.config.v1+json",
                br#"{"descriptor":{},"modelfs":{"type":"layers","diffIds":[]},"config":{"format":"gguf","capabilities":{"inputTypes":["text","image"],"outputTypes":["text"]}}}"#,
            )
            .unwrap();
        assert_eq!(capabilities(&store, &m), vec!["completion", "vision"]);
    }

    #[test]
    fn capabilities_of_a_diffusion_model_are_its_output_types() {
        // A diffusion pull: the transformer GGUF plus role-annotated
        // sidecars, config listing what it generates. No "completion".
        let mut vae = descriptor("sha256:b", "ltx_video_vae.safetensors");
        vae.annotations.get_or_insert_with(Default::default).insert(
            crate::hf::oci::ANNOTATION_ROLE.to_string(),
            "vae".to_string(),
        );
        let mut te = descriptor("sha256:c", "gemma-3-12b-it-Q4_K_M.gguf");
        te.annotations.get_or_insert_with(Default::default).insert(
            crate::hf::oci::ANNOTATION_ROLE.to_string(),
            "text_encoder".to_string(),
        );
        let (store, mut m) =
            manifest_with(vec![descriptor("sha256:a", "ltx-Q4_K_M.gguf"), vae, te]);
        m.config = store
            .write_blob(
                "application/vnd.cncf.model.config.v1+json",
                br#"{"config":{"format":"gguf","capabilities":{"inputTypes":["text"],"outputTypes":["image","video","audio"]}}}"#,
            )
            .unwrap();
        assert_eq!(capabilities(&store, &m), vec!["image", "video", "audio"]);
        // the text encoder GGUF is a sidecar, not the model
        let (primary, mmproj) = gguf_layers(&m).unwrap();
        assert_eq!(primary.digest, "sha256:a");
        assert!(mmproj.is_none());
    }

    #[test]
    fn capabilities_ignores_input_types_at_the_document_root() {
        let (store, mut m) = manifest_with(vec![descriptor("sha256:a", "model.gguf")]);
        m.config = store
            .write_blob(
                "application/vnd.cncf.model.config.v1+json",
                br#"{"capabilities":{"inputTypes":["text","image"]}}"#,
            )
            .unwrap();
        assert_eq!(capabilities(&store, &m), vec!["completion"]);
    }

    #[test]
    fn manifest_format_follows_resolve_models_precedence() {
        let (_, m) = manifest_with(vec![
            descriptor("sha256:a", "model.Q4_K_M.gguf"),
            descriptor("sha256:b", "extra.safetensors"),
        ]);
        assert_eq!(manifest_format(&m), Some(ModelFormat::Gguf));
        let (_, m) = manifest_with(vec![
            descriptor("sha256:a", "config.json"),
            descriptor("sha256:b", "model-00001-of-00002.safetensors"),
        ]);
        assert_eq!(manifest_format(&m), Some(ModelFormat::SafeTensors));
        let mut vae = descriptor("sha256:b", "vae.safetensors");
        vae.annotations.get_or_insert_with(Default::default).insert(
            crate::hf::oci::ANNOTATION_ROLE.to_string(),
            "vae".to_string(),
        );
        let (_, m) = manifest_with(vec![descriptor("sha256:a", "ltx.gguf"), vae]);
        assert_eq!(manifest_format(&m), Some(ModelFormat::Diffusion));
        let (_, m) = manifest_with(vec![descriptor("sha256:a", "README.md")]);
        assert_eq!(manifest_format(&m), None);
    }

    /// The layers `llmman pull nvidia/Cosmos3-Edge` records (the
    /// Diffusers layout: a root pipeline index, per-component subdirs).
    fn cosmos3_layers() -> Vec<crate::storage::oci::Descriptor> {
        vec![
            descriptor("sha256:a", "config.json"),
            descriptor("sha256:b", "model_index.json"),
            descriptor("sha256:c", "scheduler/scheduler_config.json"),
            descriptor("sha256:d", "transformer/config.json"),
            descriptor(
                "sha256:e",
                "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            ),
            descriptor("sha256:f", "vae/config.json"),
            descriptor("sha256:g", "vae/diffusion_pytorch_model.safetensors"),
            descriptor("sha256:h", "vision_encoder/model.safetensors"),
        ]
    }

    #[test]
    fn a_root_model_index_makes_a_safetensors_repo_omni() {
        let (_, m) = manifest_with(cosmos3_layers());
        assert_eq!(manifest_format(&m), Some(ModelFormat::Omni));
        // a GGUF transformer with role-annotated sidecars still wins
        let mut layers = cosmos3_layers();
        let mut vae = descriptor("sha256:z", "vae.safetensors");
        vae.annotations.get_or_insert_with(Default::default).insert(
            crate::hf::oci::ANNOTATION_ROLE.to_string(),
            "vae".to_string(),
        );
        layers.push(descriptor("sha256:y", "ltx.gguf"));
        layers.push(vae);
        let (_, m) = manifest_with(layers);
        assert_eq!(manifest_format(&m), Some(ModelFormat::Diffusion));
    }

    #[test]
    fn a_nested_model_index_or_one_without_weights_is_not_omni() {
        // A pipeline vendored inside an LLM repo does not make it one.
        let (_, m) = manifest_with(vec![
            descriptor("sha256:a", "config.json"),
            descriptor("sha256:b", "model.safetensors"),
            descriptor("sha256:c", "examples/pipeline/model_index.json"),
        ]);
        assert_eq!(manifest_format(&m), Some(ModelFormat::SafeTensors));
        // An index with nothing to serve is still nothing to serve.
        let (_, m) = manifest_with(vec![descriptor("sha256:a", "model_index.json")]);
        assert_eq!(manifest_format(&m), None);
    }

    #[test]
    fn capabilities_of_an_omni_model_are_media_even_without_recorded_outputs() {
        // Pulled before the pull recorded outputTypes: still not a chat model.
        let (store, m) = manifest_with(cosmos3_layers());
        assert_eq!(capabilities(&store, &m), vec!["image"]);
        let (store, mut m) = manifest_with(cosmos3_layers());
        m.config = store
            .write_blob(
                "application/vnd.cncf.model.config.v1+json",
                br#"{"config":{"format":"safetensors","capabilities":{"inputTypes":["text"],"outputTypes":["image","video"]}}}"#,
            )
            .unwrap();
        assert_eq!(capabilities(&store, &m), vec!["image", "video"]);
    }

    #[test]
    fn model_dir_is_the_parent_of_the_shallowest_config_json() {
        let cache = Path::new("/cache/abc");
        // Diffusers: transformer/config.json is listed before the root one.
        let dir = safetensors_model_dir(
            cache,
            &[
                "transformer/config.json",
                "vae/config.json",
                "config.json",
                "model_index.json",
            ],
        );
        assert_eq!(dir, cache);
        // A plain LLM checkout, as before.
        assert_eq!(
            safetensors_model_dir(cache, &["config.json", "model.safetensors"]),
            cache
        );
        // A repo whose only config.json is nested still resolves into it.
        assert_eq!(
            safetensors_model_dir(cache, &["sub/config.json", "sub/model.safetensors"]),
            cache.join("sub")
        );
        // A pure Diffusers repo: no root config.json, only the pipeline
        // index there and a config.json per component.
        assert_eq!(
            safetensors_model_dir(cache, &["vae/config.json", "model_index.json"]),
            cache
        );
        assert_eq!(safetensors_model_dir(cache, &["a.safetensors"]), cache);
    }

    #[test]
    fn omni_variant_reports_its_format_and_path() {
        let p = ModelPath::Omni(PathBuf::from("/cache/cosmos"));
        assert_eq!(p.format(), "omni");
        assert_eq!(p.path(), Path::new("/cache/cosmos"));
        assert_eq!(p.mmproj(), None);
    }

    #[test]
    fn extract_safetensors_dir_replaces_dest_shorter_than_layer_size() {
        let weights = b"complete-weights-bytes";
        let layer_hex = "aa".repeat(32);
        let mut layer = descriptor(&format!("sha256:{layer_hex}"), "model.safetensors");
        layer.media_type = "application/vnd.cncf.model.weight.v1.raw".into();
        layer.size = weights.len() as u64;
        let (store, manifest) = manifest_with(vec![layer]);

        let blob = store.root().join("blobs").join("sha256").join(&layer_hex);
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::write(&blob, weights).unwrap();

        let cache = store.root().join("cache");
        let dest = cache.join("bb".repeat(32)).join("model.safetensors");
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        std::fs::write(&dest, b"trunc").unwrap();

        let digest = format!("sha256:{}", "bb".repeat(32));
        extract_safetensors_dir(store.root(), &cache, &digest, &manifest).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), weights);

        std::fs::remove_file(&blob).unwrap();
        extract_safetensors_dir(store.root(), &cache, &digest, &manifest).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), weights);
    }

    #[test]
    fn is_gguf_layer_matches_both_mmproj_and_primary_model_files() {
        // is_mmproj_layer only narrows *within* the GGUF files a manifest
        // has — is_gguf_layer itself must still say "yes" for an mmproj
        // file, or resolve_model's gguf_layers filter would silently
        // drop it instead of finding a companion.
        assert!(is_gguf_layer(&descriptor("sha256:a", "mmproj-F16.gguf")));
        assert!(is_gguf_layer(&descriptor("sha256:b", "model.Q4_K_M.gguf")));
        assert!(!is_gguf_layer(&descriptor("sha256:c", "model.safetensors")));
    }
}

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
    /// macOS) `mlx_lm.server` — see `cmd::serve::use_mlx_for_safetensors`.
    SafeTensors(PathBuf),
}

impl ModelPath {
    /// The local filesystem path this variant resolved to — either a
    /// single `.gguf` file, or the model directory (parent of
    /// `config.json`) for a safetensors checkout.
    pub fn path(&self) -> &Path {
        match self {
            ModelPath::Gguf(p, _) => p,
            ModelPath::SafeTensors(p) => p,
        }
    }

    /// The companion `--mmproj` projector file resolved alongside a
    /// `Gguf` model, if any — always `None` for `SafeTensors` (neither
    /// vllm nor mlx_lm.server has an equivalent separate-projector-file
    /// convention).
    pub fn mmproj(&self) -> Option<&Path> {
        match self {
            ModelPath::Gguf(_, mmproj) => mmproj.as_deref(),
            ModelPath::SafeTensors(_) => None,
        }
    }

    /// A short, stable string identifying which variant this is — used by
    /// `cmd::resolve`'s JSON output and any other consumer that wants to
    /// branch on format without matching the enum directly.
    pub fn format(&self) -> &'static str {
        match self {
            ModelPath::Gguf(..) => "gguf",
            ModelPath::SafeTensors(_) => "safetensors",
        }
    }
}

/// Splits an OCI digest ("sha256:abcd...") down to just its hex portion,
/// which is what the blob store's on-disk layout uses as the filename.
fn digest_hex(digest: &str) -> anyhow::Result<&str> {
    digest
        .split_once(':')
        .map(|(_, hex)| hex)
        .ok_or_else(|| anyhow!("malformed digest: {digest}"))
}

fn layer_filepath(l: &crate::storage::oci::Descriptor) -> Option<&str> {
    l.annotations.as_ref().and_then(|a| {
        a.get("org.cncf.model.filepath")
            .or_else(|| a.get("org.opencontainers.image.title"))
            .map(|s| s.as_str())
    })
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

    // HF blobs are stored as raw GGUF — use directly.
    if layer.media_type == HF_GGUF_MEDIA_TYPE {
        let blob_path = store_path.join("blobs").join("sha256").join(layer_hex);
        if blob_path.exists() {
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

/// Ollama's `model.Capability` values reported in `/api/show`'s
/// `capabilities` array (`ollama run` reads them to set `opts.MultiModal`).
pub const CAPABILITY_COMPLETION: &str = "completion";
pub const CAPABILITY_VISION: &str = "vision";

/// The part of ollama's `Model.Capabilities()` (server/images.go) a
/// manifest alone can answer, without extracting or opening a GGUF:
/// `"completion"` always, plus `"vision"` when a companion mmproj layer
/// is present (`projectorCapabilities`) or the CNCF config's
/// `config.capabilities.inputTypes` lists `"image"` (`configCapabilities`).
pub fn capabilities(store: &OciStore, manifest: &crate::storage::oci::Manifest) -> Vec<String> {
    let mut caps = vec![CAPABILITY_COMPLETION.to_string()];

    let has_mmproj = gguf_layers(manifest).is_some_and(|(_, mmproj)| mmproj.is_some());

    // Shape written by crate::hf::oci::build_cncf_manifest.
    let config_says_image = store
        .read_blob(&manifest.config.digest)
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| {
            v.get("config")?
                .get("capabilities")?
                .get("inputTypes")?
                .as_array()
                .map(|a| a.iter().any(|t| t.as_str() == Some("image")))
        })
        .unwrap_or(false);

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
        .filter(|l| is_gguf_layer(l))
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

    // ── safetensors → vllm / mlx_lm.server ──────────────────────────────
    if manifest.layers.iter().any(is_safetensors_layer) {
        let model_dir = extract_safetensors_dir(store_path, cache_path, &desc.digest, &manifest)?;
        return Ok(ModelPath::SafeTensors(model_dir));
    }

    // Nothing usable found — report what was present.
    let exts: std::collections::HashSet<String> = manifest
        .layers
        .iter()
        .filter_map(|l| layer_filepath(l))
        .filter_map(|p| Path::new(p).extension()?.to_str().map(|e| e.to_lowercase()))
        .collect();
    if exts.is_empty() {
        anyhow::bail!("no servable model layer found in {model_ref}");
    } else {
        anyhow::bail!(
            "no servable model layer in {model_ref} — found {exts:?} files; \
             llmman serve supports GGUF (llama-server) and safetensors (vllm/mlx)"
        );
    }
}

/// Extract CNCF-format safetensors layers to a cache directory and return the
/// model directory (parent of `config.json`).
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
        if dest.exists() {
            continue;
        }

        std::fs::create_dir_all(dest.parent().context("no parent")?)?;
        let layer_hex = digest_hex(&layer.digest)?;
        let blob = store_path.join("blobs").join("sha256").join(layer_hex);
        std::fs::copy(&blob, &dest).with_context(|| format!("copy {rel_path} from blob store"))?;
        eprintln!("[llmman] extracted {rel_path}");
    }

    // Model dir = parent of config.json
    for layer in &manifest.layers {
        let Some(rel_path) = layer_filepath(layer) else {
            continue;
        };
        if Path::new(rel_path)
            .file_name()
            .map(|n| n == "config.json")
            .unwrap_or(false)
        {
            let config = cache_dir.join(rel_path);
            return config
                .parent()
                .map(|p| p.to_path_buf())
                .ok_or_else(|| anyhow!("config.json has no parent directory"));
        }
    }
    Ok(cache_dir)
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
        let dir = std::env::temp_dir().join(format!(
            "llmman-modelpack-caps-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
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

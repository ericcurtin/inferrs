//! Local OCI layout types and read/write helpers — a Rust equivalent of
//! go-shim/shared_oci.go's blob helpers and manifest_ref.go, plus the
//! CNCF ModelPack manifest construction the Go shim's own (since
//! deleted) `buildCNCFManifest` used to do.
//!
//! Writes the exact same on-disk format Go's `push`/`inspect` and
//! `crate::storage`'s `build`/`tag`/`list`/`run`/`serve` already read
//! and write (content-addressed `blobs/<alg>/<hex>`, `manifests/<ref>`
//! pointer files, `oci-layout`) — a from-scratch equivalent, not a
//! replacement, so a native pull never needs to call into Go.
//!
//! Shared with `crate::sources`, which stores the ModelScope/NGC/S3/GCS/
//! local-directory sources in exactly this format too.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

// ---------------------------------------------------------------------------
// OCI image-spec types — just enough of the public spec to round-trip
// what this codebase actually produces/reads (mirrors
// opencontainers/image-spec's Go structs field-for-field).
// ---------------------------------------------------------------------------

pub const MEDIA_TYPE_IMAGE_MANIFEST: &str = "application/vnd.oci.image.manifest.v1+json";
pub const ANNOTATION_REF_NAME: &str = "org.opencontainers.image.ref.name";
pub const ANNOTATION_TITLE: &str = "org.opencontainers.image.title";

/// Matches `ocispec.Descriptor`'s JSON shape exactly.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Descriptor {
    #[serde(rename = "mediaType")]
    pub media_type: String,
    pub digest: String,
    pub size: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub urls: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<BTreeMap<String, String>>,
    #[serde(
        rename = "artifactType",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub artifact_type: Option<String>,
}

impl Descriptor {
    pub fn annotation(&self, key: &str) -> Option<&str> {
        self.annotations.as_ref()?.get(key).map(String::as_str)
    }
}

/// Matches `ocispec.Manifest`'s JSON shape exactly.
#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    #[serde(rename = "schemaVersion")]
    pub schema_version: i32,
    #[serde(rename = "mediaType")]
    pub media_type: String,
    #[serde(
        rename = "artifactType",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub artifact_type: Option<String>,
    pub config: Descriptor,
    pub layers: Vec<Descriptor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<BTreeMap<String, String>>,
}

// ---------------------------------------------------------------------------
// CNCF ModelPack config (github.com/modelpack/model-spec) — just the
// fields buildCNCFManifest actually populates.
// ---------------------------------------------------------------------------

pub const ARTIFACT_TYPE_MODEL_MANIFEST: &str = "application/vnd.cncf.model.manifest.v1+json";
pub const MEDIA_TYPE_MODEL_CONFIG: &str = "application/vnd.cncf.model.config.v1+json";
pub const MEDIA_TYPE_MODEL_WEIGHT_RAW: &str = "application/vnd.cncf.model.weight.v1.raw";
pub const MEDIA_TYPE_MODEL_WEIGHT_CONFIG_RAW: &str =
    "application/vnd.cncf.model.weight.config.v1.raw";
pub const MEDIA_TYPE_MODEL_DOC_RAW: &str = "application/vnd.cncf.model.doc.v1.raw";
/// Inference/handler scripts shipped alongside the weights. Only
/// `crate::sources` produces these — a HuggingFace repo's `.py` files
/// aren't among the ones `select_downloadable_hf_files` fetches.
pub const MEDIA_TYPE_MODEL_CODE_RAW: &str = "application/vnd.cncf.model.code.v1.raw";
pub const ANNOTATION_FILEPATH: &str = "org.cncf.model.filepath";

#[derive(Serialize, Default)]
struct ModelDescriptor {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    licenses: Vec<String>,
}

#[derive(Serialize)]
struct ModelFS {
    #[serde(rename = "type")]
    kind: String,
    #[serde(rename = "diffIds")]
    diff_ids: Vec<String>,
}

#[derive(Serialize, Default)]
struct ModelCapabilities {
    #[serde(rename = "inputTypes", default, skip_serializing_if = "Vec::is_empty")]
    input_types: Vec<&'static str>,
    #[serde(rename = "outputTypes", default, skip_serializing_if = "Vec::is_empty")]
    output_types: Vec<&'static str>,
}

#[derive(Serialize, Default)]
struct ModelConfig {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    format: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    capabilities: Option<ModelCapabilities>,
}

#[derive(Serialize)]
struct Model {
    #[serde(rename = "descriptor")]
    descriptor: ModelDescriptor,
    modelfs: ModelFS,
    #[serde(default, skip_serializing_if = "is_default_config")]
    config: ModelConfig,
}

fn is_default_config(c: &ModelConfig) -> bool {
    c.format.is_empty() && c.capabilities.is_none()
}

/// Mirrors `modelMeta` in hf.go: the optional-but-valuable metadata
/// `build_cncf_manifest` populates beyond the bare config.format+modelfs.
#[derive(Default)]
pub struct ModelMeta {
    pub format: String,
    pub licenses: Vec<String>,
    pub vision: bool,
    /// A latent diffusion model: generates media instead of text. The
    /// listed output types (`"image"`, `"video"`, `"audio"`) become the
    /// model's capabilities.
    pub diffusion_outputs: Vec<&'static str>,
}

/// Layer annotation naming the part a sidecar plays in a diffusion
/// model: `"vae"`, `"audio_vae"`, `"text_proj"` or `"text_encoder"`.
/// Layers without it are the model itself (plus mmproj / license).
pub const ANNOTATION_ROLE: &str = "org.llmman.role";

/// Builds a conformant CNCF model-spec config blob and manifest
/// referencing `layers`, writing each into `layout_dir`, and returns the
/// manifest's descriptor. `filepath_annotation` sets the manifest-level
/// `org.cncf.model.filepath` annotation for the single-weight-file case
/// (GGUF); pass `""` for the multi-layer safetensors case.
pub fn build_cncf_manifest(
    layout_dir: &Path,
    meta: &ModelMeta,
    model_repo: &str,
    filepath_annotation: &str,
    layers: Vec<Descriptor>,
) -> Result<Descriptor> {
    let model = Model {
        descriptor: ModelDescriptor {
            licenses: meta.licenses.clone(),
        },
        modelfs: ModelFS {
            kind: "layers".to_string(),
            diff_ids: layers.iter().map(|l| l.digest.clone()).collect(),
        },
        config: ModelConfig {
            format: meta.format.clone(),
            capabilities: if !meta.diffusion_outputs.is_empty() {
                Some(ModelCapabilities {
                    input_types: vec!["text"],
                    output_types: meta.diffusion_outputs.clone(),
                })
            } else {
                meta.vision.then(|| ModelCapabilities {
                    input_types: vec!["text", "image"],
                    output_types: vec!["text"],
                })
            },
        },
    };
    let cfg_data = serde_json::to_vec(&model).context("marshal CNCF model config")?;
    let config_desc = write_blob(layout_dir, MEDIA_TYPE_MODEL_CONFIG, &cfg_data)
        .context("store CNCF model config")?;

    let mut annotations = BTreeMap::new();
    annotations.insert("ai.model.repo".to_string(), model_repo.to_string());
    if !filepath_annotation.is_empty() {
        annotations.insert(
            ANNOTATION_FILEPATH.to_string(),
            filepath_annotation.to_string(),
        );
    }
    let manifest = Manifest {
        schema_version: 2,
        media_type: MEDIA_TYPE_IMAGE_MANIFEST.to_string(),
        artifact_type: Some(ARTIFACT_TYPE_MODEL_MANIFEST.to_string()),
        config: config_desc,
        layers,
        annotations: Some(annotations),
    };
    let manifest_data = serde_json::to_vec(&manifest).context("marshal CNCF manifest")?;
    write_blob(layout_dir, MEDIA_TYPE_IMAGE_MANIFEST, &manifest_data).context("store CNCF manifest")
}

// ---------------------------------------------------------------------------
// Blob storage — mirrors shared_oci.go's blobPath/readBlob/writeBlob/
// writeBlobStream/blobExists.
// ---------------------------------------------------------------------------

/// Process-wide, so no two temp paths written here can ever collide.
fn next_counter() -> u64 {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn digest_to_path(layout_dir: &Path, digest: &str) -> Result<PathBuf> {
    let (alg, hex) = digest
        .split_once(':')
        .with_context(|| format!("malformed digest {digest:?}"))?;
    Ok(layout_dir.join("blobs").join(alg).join(hex))
}

pub fn blob_exists(layout_dir: &Path, desc: &Descriptor) -> bool {
    match digest_to_path(layout_dir, &desc.digest) {
        Ok(p) => std::fs::metadata(&p)
            .map(|m| m.len() as i64 == desc.size)
            .unwrap_or(false),
        Err(_) => false,
    }
}

pub fn read_blob(layout_dir: &Path, digest: &str) -> Result<Vec<u8>> {
    std::fs::read(digest_to_path(layout_dir, digest)?).context("read blob")
}

/// Atomically writes `data` to the layout's blobs directory, keyed by its
/// own sha256 — mirrors `writeBlob`.
pub fn write_blob(layout_dir: &Path, media_type: &str, data: &[u8]) -> Result<Descriptor> {
    let digest = format!("sha256:{:x}", Sha256::digest(data));
    let dir = layout_dir.join("blobs").join("sha256");
    std::fs::create_dir_all(&dir)?;
    let dest = dir.join(digest.trim_start_matches("sha256:"));
    if let Ok(m) = std::fs::metadata(&dest) {
        if m.len() as usize == data.len() {
            return Ok(Descriptor {
                media_type: media_type.to_string(),
                digest,
                size: data.len() as i64,
                ..Default::default()
            });
        }
    }
    // A counter, not a name derived from the digest: two concurrent
    // writes of byte-identical content (same digest) would otherwise
    // share a tmp path and could truncate each other mid-write.
    let tmp = dir.join(format!("{}-{}.tmp", std::process::id(), next_counter()));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(data)?;
    }
    std::fs::rename(&tmp, &dest)?;
    Ok(Descriptor {
        media_type: media_type.to_string(),
        digest,
        size: data.len() as i64,
        ..Default::default()
    })
}

/// Stores a file already on local disk as a blob, hashing a chunk at a
/// time rather than reading it whole into memory — the input here is a
/// multi-gigabyte safetensors shard.
///
/// `consume` moves the file in (a temp file this process owns and that
/// nothing else can be writing). Otherwise it is copied and hashed in
/// one pass into a temp file, so the bytes stored under the returned
/// digest are the exact bytes that were hashed even if the caller's own
/// file changes underneath; either way that file is left untouched.
///
/// A file whose digest is already stored costs nothing to re-offer, so
/// the caller still owns any staging file after this returns.
pub fn write_blob_from_file(
    layout_dir: &Path,
    media_type: &str,
    path: &Path,
    consume: bool,
) -> Result<Descriptor> {
    let dir = layout_dir.join("blobs").join("sha256");
    std::fs::create_dir_all(&dir)?;
    let tmp = dir.join(format!("{}-{}.tmp", std::process::id(), next_counter()));
    let mut source =
        std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();

    // Owning the file means nobody else is writing it, so hashing it
    // where it lies and then renaming is safe — and free.
    let (size, staged) = if consume {
        let size = std::io::copy(&mut source, &mut hasher)
            .with_context(|| format!("hash {}", path.display()))?;
        (size, path.to_path_buf())
    } else {
        let mut out = std::fs::File::create(&tmp)?;
        let size = std::io::copy(&mut source, &mut HashingWriter(&mut out, &mut hasher))
            .with_context(|| format!("copy {} into the blob store", path.display()))?;
        (size, tmp.clone())
    };
    drop(source);
    let digest = format!("sha256:{:x}", hasher.finalize());
    let dest = dir.join(digest.trim_start_matches("sha256:"));

    // Content-addressed, so an existing blob of this digest already
    // holds exactly these bytes. Renaming (never writing to `dest`
    // directly) is what keeps a concurrent writer of the same digest
    // from ever seeing a half-written blob at its final path.
    if !dest.exists() {
        match std::fs::rename(&staged, &dest) {
            Ok(()) => {}
            // A sibling won the race and published the identical blob
            // first; POSIX overwrites silently, Windows errors.
            Err(_) if dest.exists() => {}
            Err(e) => {
                if !consume {
                    let _ = std::fs::remove_file(&tmp);
                }
                return Err(e).with_context(|| format!("publish blob for {}", path.display()));
            }
        }
    }
    if !consume {
        let _ = std::fs::remove_file(&tmp);
    }
    Ok(Descriptor {
        media_type: media_type.to_string(),
        digest,
        size: size as i64,
        ..Default::default()
    })
}

/// Tees writes into a hasher, so a copy hashes exactly the bytes it
/// stores rather than re-reading the source a second time.
struct HashingWriter<'a>(&'a mut std::fs::File, &'a mut Sha256);

impl std::io::Write for HashingWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.0.write(buf)?;
        self.1.update(&buf[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

// ---------------------------------------------------------------------------
// Manifest-ref bookkeeping — mirrors manifest_ref.go.
// ---------------------------------------------------------------------------

const MANIFESTS_DIR_NAME: &str = "manifests";

fn sanitize_ref_segment(s: &str) -> String {
    if s.is_empty() || s == "." || s == ".." {
        return "__".to_string();
    }
    s.replace([':', '\\'], "_")
}

/// Splits `reference` into the path segments used to lay it out under
/// `manifests/` — mirrors `refPathSegments`. A `:` only starts a tag if
/// it comes after the last `/` (so a registry port, e.g.
/// "host:5000/owner/repo", isn't mistaken for one); otherwise the tag
/// defaults to "latest".
fn ref_path_segments(reference: &str) -> Vec<String> {
    let last_colon = reference.rfind(':').map(|i| i as isize).unwrap_or(-1);
    let last_slash = reference.rfind('/').map(|i| i as isize).unwrap_or(-1);
    let (name, tag) = if last_colon > last_slash {
        (
            &reference[..last_colon as usize],
            &reference[last_colon as usize + 1..],
        )
    } else {
        (reference, "latest")
    };
    let mut segs: Vec<String> = name
        .split('/')
        .filter(|s| !s.is_empty())
        .map(sanitize_ref_segment)
        .collect();
    segs.push(sanitize_ref_segment(tag));
    segs
}

fn manifest_ref_path(layout_dir: &Path, reference: &str) -> PathBuf {
    let mut p = layout_dir.join(MANIFESTS_DIR_NAME);
    for seg in ref_path_segments(reference) {
        p.push(seg);
    }
    p
}

pub fn read_manifest_ref(layout_dir: &Path, reference: &str) -> Result<Descriptor> {
    let data = std::fs::read(manifest_ref_path(layout_dir, reference))?;
    Ok(serde_json::from_slice(&data)?)
}

/// Atomically records that `reference` now points at `manifest_desc` —
/// mirrors `writeManifestRef`.
pub fn write_manifest_ref(
    layout_dir: &Path,
    reference: &str,
    mut manifest_desc: Descriptor,
) -> Result<()> {
    let mut ann = manifest_desc.annotations.clone().unwrap_or_default();
    ann.insert(ANNOTATION_REF_NAME.to_string(), reference.to_string());
    manifest_desc.annotations = Some(ann);

    let path = manifest_ref_path(layout_dir, reference);
    let dir = path.parent().context("manifest ref path has no parent")?;
    std::fs::create_dir_all(dir)?;
    let data = serde_json::to_vec_pretty(&manifest_desc)?;

    let tmp = dir.join(format!(
        "{}.{}-{}.tmp",
        path.file_name().unwrap().to_string_lossy(),
        std::process::id(),
        next_counter()
    ));
    std::fs::write(&tmp, &data)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Records `alias` as a second name for whatever `reference` already
/// points at. Needed when staging a pull into a throwaway layout that is
/// then pushed under a *different* reference: `llmman_push` resolves the
/// manifest by an exact ref lookup, so the staged model has to be
/// findable under the destination's own name.
pub fn alias_manifest_ref(layout_dir: &Path, reference: &str, alias: &str) -> Result<()> {
    if alias == reference {
        return Ok(());
    }
    let desc = read_manifest_ref(layout_dir, reference)
        .with_context(|| format!("read staged manifest for {reference}"))?;
    write_manifest_ref(layout_dir, alias, desc)
}

/// Initializes the OCI layout marker files and `manifests/` directory if
/// not already present — mirrors `ensureLayout`.
pub fn ensure_layout(layout_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(layout_dir)?;
    std::fs::create_dir_all(layout_dir.join(MANIFESTS_DIR_NAME))?;
    let marker = layout_dir.join("oci-layout");
    if !marker.exists() {
        std::fs::write(marker, br#"{"imageLayoutVersion":"1.0.0"}"#)?;
    }
    Ok(())
}

/// Returns the cached layer's filename if `reference` is already fully
/// cached in `layout_dir` (manifest blob + every layer blob present), or
/// `None` — mirrors `cachedLayerName`.
pub fn cached_layer_name(layout_dir: &Path, reference: &str) -> Option<String> {
    let m = read_manifest_ref(layout_dir, reference).ok()?;
    if !blob_exists(layout_dir, &m) {
        return None;
    }
    let data = read_blob(layout_dir, &m.digest).ok()?;
    let manifest: Manifest = serde_json::from_slice(&data).ok()?;
    for layer in &manifest.layers {
        if !blob_exists(layout_dir, layer) {
            return None;
        }
    }
    if let Some(layer) = manifest.layers.first() {
        for key in [ANNOTATION_FILEPATH, ANNOTATION_TITLE] {
            if let Some(name) = layer.annotation(key) {
                if !name.is_empty() {
                    return Path::new(name)
                        .file_name()
                        .map(|f| f.to_string_lossy().into_owned());
                }
            }
        }
    }
    Some(reference.to_string())
}

/// Prints "Cached &lt;label&gt;" and returns true if `reference` is
/// already fully cached — mirrors `reportCached`.
pub fn report_cached(layout_dir: &Path, reference: &str) -> bool {
    match cached_layer_name(layout_dir, reference) {
        Some(name) => {
            eprintln!("Cached   {name}");
            true
        }
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ref_path_segments_defaults_a_missing_tag_to_latest() {
        assert_eq!(
            ref_path_segments("owner/repo"),
            vec!["owner", "repo", "latest"]
        );
    }

    #[test]
    fn ref_path_segments_splits_an_explicit_tag() {
        assert_eq!(
            ref_path_segments("owner/repo:v1"),
            vec!["owner", "repo", "v1"]
        );
    }

    #[test]
    fn ref_path_segments_sanitizes_unsafe_characters() {
        assert_eq!(
            ref_path_segments("owner/repo:"),
            vec!["owner", "repo", "__"]
        );
    }

    #[test]
    fn write_blob_then_read_blob_round_trips() {
        let dir = tempfile();
        let desc = write_blob(&dir, "text/plain", b"hello world").unwrap();
        assert_eq!(desc.size, 11);
        assert!(blob_exists(&dir, &desc));
        assert_eq!(read_blob(&dir, &desc.digest).unwrap(), b"hello world");
    }

    #[test]
    fn write_blob_from_file_moves_a_file_it_owns_and_copies_one_it_does_not() {
        let dir = tempfile();
        let src = dir.join("weights.safetensors");
        std::fs::write(&src, b"weights").unwrap();

        let copied = write_blob_from_file(&dir, MEDIA_TYPE_MODEL_WEIGHT_RAW, &src, false).unwrap();
        assert_eq!(copied.size, 7);
        assert!(
            src.exists(),
            "consume=false must leave a user's own file alone"
        );
        assert_eq!(read_blob(&dir, &copied.digest).unwrap(), b"weights");

        // Same content, so the blob already exists and nothing is
        // written — but the caller still owns the staging file.
        let staged = dir.join("staged.part");
        std::fs::write(&staged, b"weights").unwrap();
        let moved = write_blob_from_file(&dir, MEDIA_TYPE_MODEL_WEIGHT_RAW, &staged, true).unwrap();
        assert_eq!(moved.digest, copied.digest);

        // Distinct content: consume=true really does move it away.
        let staged2 = dir.join("staged2.part");
        std::fs::write(&staged2, b"other weights").unwrap();
        let moved2 =
            write_blob_from_file(&dir, MEDIA_TYPE_MODEL_WEIGHT_RAW, &staged2, true).unwrap();
        assert!(!staged2.exists(), "consume=true must move, not copy");
        assert_eq!(read_blob(&dir, &moved2.digest).unwrap(), b"other weights");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn alias_manifest_ref_makes_a_staged_pull_findable_under_another_name() {
        let dir = tempfile();
        ensure_layout(&dir).unwrap();
        let desc = Descriptor {
            media_type: MEDIA_TYPE_IMAGE_MANIFEST.to_string(),
            digest: "sha256:abc".to_string(),
            size: 3,
            ..Default::default()
        };
        write_manifest_ref(&dir, "s3://bucket/model:latest", desc).unwrap();
        alias_manifest_ref(&dir, "s3://bucket/model:latest", "ghcr.io/me/model:v1").unwrap();
        assert_eq!(
            read_manifest_ref(&dir, "ghcr.io/me/model:v1")
                .unwrap()
                .digest,
            "sha256:abc"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_blob_is_idempotent() {
        let dir = tempfile();
        let d1 = write_blob(&dir, "text/plain", b"same content").unwrap();
        let d2 = write_blob(&dir, "text/plain", b"same content").unwrap();
        assert_eq!(d1.digest, d2.digest);
    }

    #[test]
    fn manifest_ref_round_trips_and_records_ref_name_annotation() {
        let dir = tempfile();
        ensure_layout(&dir).unwrap();
        let desc = Descriptor {
            media_type: MEDIA_TYPE_IMAGE_MANIFEST.to_string(),
            digest: "sha256:abc".to_string(),
            size: 3,
            ..Default::default()
        };
        write_manifest_ref(&dir, "owner/repo:tag", desc).unwrap();
        let got = read_manifest_ref(&dir, "owner/repo:tag").unwrap();
        assert_eq!(got.annotation(ANNOTATION_REF_NAME), Some("owner/repo:tag"));
    }

    #[test]
    fn cached_layer_name_is_none_when_nothing_cached() {
        let dir = tempfile();
        ensure_layout(&dir).unwrap();
        assert!(cached_layer_name(&dir, "owner/repo:tag").is_none());
    }

    #[test]
    fn cached_layer_name_returns_filename_once_fully_cached() {
        let dir = tempfile();
        ensure_layout(&dir).unwrap();
        let layer = write_blob(&dir, MEDIA_TYPE_MODEL_WEIGHT_RAW, b"weights").unwrap();
        let mut layer = layer;
        layer.annotations = Some(BTreeMap::from([(
            ANNOTATION_FILEPATH.to_string(),
            "model.gguf".to_string(),
        )]));
        let manifest_desc = build_cncf_manifest(
            &dir,
            &ModelMeta {
                format: "gguf".to_string(),
                ..Default::default()
            },
            "owner/repo",
            "model.gguf",
            vec![layer],
        )
        .unwrap();
        write_manifest_ref(&dir, "owner/repo:tag", manifest_desc).unwrap();
        assert_eq!(
            cached_layer_name(&dir, "owner/repo:tag").as_deref(),
            Some("model.gguf")
        );
    }

    fn tempfile() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "llmman-oci-test-{}-{}",
            std::process::id(),
            rand_suffix()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn rand_suffix() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

//! Local OCI Image Layout store.
//!
//! Implements a subset of the OCI Image Layout spec
//! (<https://github.com/opencontainers/image-spec/blob/main/image-layout.md>)
//! sufficient for llmman's local operations: build, list, rm, tag, inspect-local.
//!
//! Layout on disk:
//! ```text
//! <store-root>/
//!   oci-layout             {"imageLayoutVersion":"1.0.0"}
//!   manifests/<...>/<tag>  one small JSON OCI descriptor (mediaType/
//!                          digest/size) per stored reference, pointing at
//!                          the manifest blob under blobs/ — see
//!                          `ref_path_segments`. One file per model,
//!                          mirroring Ollama's per-model manifest layout,
//!                          rather than a single shared index.json (a
//!                          breaking change from older llmman versions —
//!                          no migration is provided; re-pull instead).
//!   blobs/
//!     sha256/
//!       <hex>              one file per blob — config, layers, and each
//!                          manifest's own raw JSON bytes
//! ```

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest as Sha2Digest, Sha256};

/// Per-process counter used only to make `OciStore::write_ref`'s temp
/// file name unique across concurrent writers — see there.
static WRITE_REF_COUNTER: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// Minimal OCI spec types (no external crate needed)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Descriptor {
    pub media_type: String,
    pub digest: String,
    pub size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    pub schema_version: u32,
    pub media_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_type: Option<String>,
    pub config: Descriptor,
    pub layers: Vec<Descriptor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<std::collections::HashMap<String, String>>,
}

/// `application/vnd.cncf.model.config.v1+json` — the CNCF Model Format Spec
/// (<https://github.com/modelpack/model-spec>) config document. The
/// deserializing counterpart of what `crate::hf::oci::build_cncf_manifest`
/// writes for HuggingFace/cloud-source pulls; this is the equivalent for
/// `llmman build`'s local-directory packaging path.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CncfConfigDescriptor {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CncfConfigConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CncfModelFs {
    #[serde(rename = "type")]
    pub fs_type: String,
    pub diff_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CncfModelConfig {
    pub descriptor: CncfConfigDescriptor,
    pub config: CncfConfigConfig,
    pub modelfs: CncfModelFs,
}

/// `find`'s "no such reference" error, distinct from a broken store so
/// callers (e.g. `/api/show`) can answer 404 for one and 500 for the other.
#[derive(Debug)]
pub struct NotFound(pub String);

impl std::fmt::Display for NotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "image not found: {}", self.0)
    }
}

impl std::error::Error for NotFound {}

/// Summary of a locally stored image shown by `list`.
#[derive(Debug)]
pub struct ImageSummary {
    pub reference: String,
    pub digest: String,
    #[allow(dead_code)]
    pub media_type: String,
    pub size: u64,
    pub modified_at: Option<std::time::SystemTime>,
}

// ---------------------------------------------------------------------------
// OciStore
// ---------------------------------------------------------------------------

pub struct OciStore {
    root: PathBuf,
}

impl OciStore {
    /// Open (or create) an OCI layout store at `root`.
    pub fn open(root: impl AsRef<Path>) -> anyhow::Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        fs::create_dir_all(root.join("blobs").join("sha256"))?;
        fs::create_dir_all(root.join("manifests"))?;

        // Write oci-layout marker if absent
        let marker = root.join("oci-layout");
        if !marker.exists() {
            fs::write(&marker, r#"{"imageLayoutVersion":"1.0.0"}"#)?;
        }

        Ok(Self { root })
    }

    #[allow(dead_code)]
    pub fn root(&self) -> &Path {
        &self.root
    }

    // ------------------------------------------------------------------
    // Per-model manifest references
    // ------------------------------------------------------------------
    //
    // Each tagged/pulled/built model gets its own small JSON file (an OCI
    // `Descriptor`) recording which manifest blob it points to — see the
    // module doc comment. `ref_path` derives the file's path directly
    // from the reference, so the common lookup is an O(1) read, no index
    // to scan. `find`/`remove` fall back to a `list_refs` scan through
    // `matching_index` only for a reference spelled differently than it
    // was stored (see `ref_matches_precise`).

    /// The path `reference`'s manifest-pointer file lives (or would live)
    /// at, under `manifests/` — see `ref_path_segments`.
    fn ref_path(&self, reference: &str) -> PathBuf {
        let mut path = self.root.join("manifests");
        for seg in ref_path_segments(reference) {
            path.push(seg);
        }
        path
    }

    fn read_ref(&self, reference: &str) -> anyhow::Result<Descriptor> {
        let data = fs::read(self.ref_path(reference))
            .with_context(|| format!("read manifest ref {reference}"))?;
        serde_json::from_slice(&data).context("parse manifest ref")
    }

    fn write_ref(&self, reference: &str, desc: &Descriptor) -> anyhow::Result<()> {
        let path = self.ref_path(reference);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        // A uniquely-named temp file (pid + a per-process counter), not
        // a fixed one: two concurrent writers of the same reference must
        // not truncate/interleave each other's write. Appended as a
        // suffix, not via `with_extension` (which would eat whatever
        // follows the last '.' in a dotted tag, e.g. ".../0.8b" becoming
        // ".../0.tmp").
        let n = WRITE_REF_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut tmp = path.clone().into_os_string();
        tmp.push(format!(".{}.{n}.tmp", std::process::id()));
        let tmp = PathBuf::from(tmp);
        if let Err(e) = fs::write(&tmp, serde_json::to_string_pretty(desc)?) {
            let _ = fs::remove_file(&tmp);
            return Err(e.into());
        }
        if let Err(e) = fs::rename(&tmp, &path) {
            let _ = fs::remove_file(&tmp);
            return Err(e.into());
        }
        Ok(())
    }

    fn remove_ref(&self, reference: &str) -> anyhow::Result<()> {
        fs::remove_file(self.ref_path(reference))
            .with_context(|| format!("remove manifest ref {reference}"))
    }

    /// Every stored manifest descriptor, read from `manifests/` — each
    /// still carries its own reference name in its
    /// `org.opencontainers.image.ref.name` annotation. A single
    /// unreadable or unparsable entry is skipped rather than failing the
    /// whole walk.
    pub(crate) fn list_refs(&self) -> Vec<Descriptor> {
        let root = self.root.join("manifests");
        let mut out = Vec::new();
        collect_refs(&root, &mut out);
        out
    }

    /// Like [`list_refs`], but aborts on the first unreadable subtree or
    /// unparsable pointer file instead of silently skipping it. Required by
    /// the destructive GC path ([`crate::storage::gc::referenced_digests`]):
    /// a reference dropped by the lossy `list_refs` would make that model's
    /// blobs look unreferenced and get swept, so GC must instead refuse to
    /// run at all when it can't enumerate every surviving reference. A
    /// missing `manifests/` directory (a fresh store) is an empty list, not
    /// an error.
    pub(crate) fn list_refs_strict(&self) -> anyhow::Result<Vec<Descriptor>> {
        let root = self.root.join("manifests");
        let mut out = Vec::new();
        collect_refs_strict(&root, &mut out)?;
        Ok(out)
    }

    // ------------------------------------------------------------------
    // Blobs
    // ------------------------------------------------------------------

    fn blob_path(&self, digest: &str) -> anyhow::Result<PathBuf> {
        let (algo, hex) = split_digest(digest)?;
        Ok(self.root.join("blobs").join(algo).join(hex))
    }

    /// Write `data` as a blob.  Returns its `Descriptor`.
    pub fn write_blob(&self, media_type: &str, data: &[u8]) -> anyhow::Result<Descriptor> {
        let hex = hex::encode(Sha256::digest(data));
        let digest = format!("sha256:{}", hex);
        let path = self.root.join("blobs").join("sha256").join(&hex);
        if !path.exists() {
            let tmp = path.with_extension("tmp");
            fs::write(&tmp, data)?;
            fs::rename(tmp, &path)?;
        }
        Ok(Descriptor {
            media_type: media_type.into(),
            digest,
            size: data.len() as u64,
            annotations: None,
        })
    }

    /// Write a large file as a blob, streaming to avoid buffering the whole file.
    #[allow(dead_code)]
    pub fn write_blob_file(
        &self,
        media_type: &str,
        path: impl AsRef<Path>,
    ) -> anyhow::Result<Descriptor> {
        let path = path.as_ref();
        let mut hasher = Sha256::new();
        let mut size = 0u64;
        let tmp = self
            .root
            .join("blobs")
            .join("sha256")
            .join(format!("tmp-{}", std::process::id()));
        {
            let mut src =
                fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
            let mut dst = fs::File::create(&tmp)?;
            let mut buf = vec![0u8; 1 << 20]; // 1 MiB chunks
            loop {
                let n = src.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
                dst.write_all(&buf[..n])?;
                size += n as u64;
            }
        }
        let hex = hex::encode(hasher.finalize());
        let digest = format!("sha256:{}", hex);
        let dest = self.root.join("blobs").join("sha256").join(&hex);
        if dest.exists() {
            fs::remove_file(&tmp)?;
        } else {
            fs::rename(tmp, &dest)?;
        }
        Ok(Descriptor {
            media_type: media_type.into(),
            digest,
            size,
            annotations: None,
        })
    }

    /// Read a blob's raw bytes.
    pub fn read_blob(&self, digest: &str) -> anyhow::Result<Vec<u8>> {
        fs::read(self.blob_path(digest)?).with_context(|| format!("read blob {}", digest))
    }

    // ------------------------------------------------------------------
    // Manifest helpers
    // ------------------------------------------------------------------

    pub fn write_manifest(&self, manifest: &Manifest) -> anyhow::Result<Descriptor> {
        let data = serde_json::to_vec(manifest)?;
        self.write_blob("application/vnd.oci.image.manifest.v1+json", &data)
    }

    pub fn read_manifest(&self, digest: &str) -> anyhow::Result<Manifest> {
        let data = self.read_blob(digest)?;
        serde_json::from_slice(&data).context("parse manifest")
    }

    // ------------------------------------------------------------------
    // Tag operations
    // ------------------------------------------------------------------

    /// Records that `reference` now points at `desc`, replacing any prior
    /// entry stored under that exact reference. The full `reference`
    /// string is stored in the annotation so `list` shows it verbatim —
    /// see `ref_path_segments` for how it's also encoded (with some loss:
    /// a tagless reference and its `:latest` spelling collapse to the
    /// same file, same as `ref_matches_precise` already treats them) into
    /// the file's own path.
    pub fn tag(&self, mut desc: Descriptor, reference: &str) -> anyhow::Result<()> {
        let mut ann = desc.annotations.take().unwrap_or_default();
        ann.insert(
            "org.opencontainers.image.ref.name".into(),
            reference.to_string(),
        );
        desc.annotations = Some(ann);
        self.write_ref(reference, &desc)
    }

    /// Find the descriptor stored for `reference` — see this type's own
    /// doc comment on the fast path (`read_ref`) and its fallback
    /// (`matching_index` over every stored reference).
    pub fn find(&self, reference: &str) -> anyhow::Result<Descriptor> {
        if self.ref_path(reference).exists() {
            // Present but unparsable is a broken store, not a typo —
            // surface that error rather than falling through to "not
            // found".
            let desc = self.read_ref(reference)?;
            // A digest reference names content, and the pointer at its
            // path answers for it only while it holds that content. One
            // that holds another digest, from a `cp` of an older llmman
            // to a digest-shaped destination or from a hand edit, is
            // passed over, and the lookup by content below decides.
            match split_ref_digest(reference).1 {
                Some(digest) if !desc.digest.eq_ignore_ascii_case(digest) => {}
                _ => return Ok(desc),
            }
        }
        let refs = self.list_refs();
        matching_index(&refs, reference)
            .map(|i| refs[i].clone())
            .ok_or_else(|| NotFound(reference.to_string()).into())
    }

    /// The real total size of an image: the sum of its layer sizes, not
    /// `desc.size` (a `Descriptor`'s own `size` field is the *manifest
    /// blob's* size — just a few hundred bytes of JSON — not the image's
    /// actual content size). Falls back to `desc.size` if the manifest
    /// can't be read, so callers still get *something* rather than an
    /// error over what's only ever used for display.
    pub fn total_size(&self, desc: &Descriptor) -> u64 {
        self.read_manifest(&desc.digest)
            .map(|manifest| manifest.layers.iter().map(|l| l.size).sum())
            .unwrap_or(desc.size)
    }

    // ------------------------------------------------------------------
    // List / Remove
    // ------------------------------------------------------------------

    pub fn list(&self) -> anyhow::Result<Vec<ImageSummary>> {
        let refs = self.list_refs();
        Ok(refs
            .into_iter()
            .map(|m| {
                let reference = m
                    .annotations
                    .as_ref()
                    .and_then(|a| a.get("org.opencontainers.image.ref.name"))
                    .cloned()
                    .unwrap_or_else(|| m.digest.clone());
                let modified_at = self
                    .blob_path(&m.digest)
                    .ok()
                    .and_then(|p| fs::metadata(p).ok())
                    .and_then(|meta| meta.modified().ok());
                let size = self.total_size(&m);
                ImageSummary {
                    reference,
                    digest: m.digest,
                    media_type: m.media_type,
                    size,
                    modified_at,
                }
            })
            .collect())
    }

    /// Remove the same single entry `find` would return for `reference`
    /// (see `find`'s own doc comment on its fast path and fallback), and
    /// say which: the stored reference that went, since for a digest
    /// that is a tag the caller did not spell. Two differences from
    /// `find`: the pointer at the reference's own path goes whatever
    /// digest it holds, since a pointer `find` passes over for holding
    /// other content is exactly what `rm` has to be able to take away;
    /// and a digest held by several tags is refused with the tags listed,
    /// the way `docker rmi` refuses an image several tags point at,
    /// unless the reference spells one of them (`m:v9@sha256:…`), rather
    /// than taking whichever the tree is walked to first. A bare
    /// `m@sha256:…` spells no tag here, even though `find` reads it as
    /// `:latest` for a lookup: a removal takes the explicit form. Does not
    /// GC blobs.
    pub fn remove(&self, reference: &str) -> anyhow::Result<String> {
        if self.remove_ref(reference).is_ok() {
            return Ok(reference.to_owned());
        }
        let refs = self.list_refs();
        let (base, digest) = split_ref_digest(reference);
        if digest.is_some() {
            // `default_tag` leaves a reference alone only when it already
            // carries a tag: that is the spelled one.
            let spelled = (default_tag(base) == base).then_some(base);
            let held_by: Vec<&str> = refs
                .iter()
                .filter(|m| ref_matches_precise(m, reference))
                .filter_map(stored_ref_name)
                .collect();
            if held_by.len() > 1
                && !held_by
                    .iter()
                    .any(|s| spelled.is_some_and(|t| default_tag(s) == t))
            {
                return Err(anyhow!(
                    "{reference} is held by more than one tag: {}; name the tag to remove",
                    held_by.join(", ")
                ));
            }
        }
        let Some(i) = matching_index(&refs, reference) else {
            return Err(anyhow!("image not found: {}", reference));
        };
        let Some(name) = stored_ref_name(&refs[i]) else {
            return Err(anyhow!("image not found: {}", reference));
        };
        self.remove_ref(name)?;
        Ok(name.to_owned())
    }

    // ------------------------------------------------------------------
    // Build helpers
    // ------------------------------------------------------------------

    /// Package all files in `src_dir` as a CNCF Model Format Spec
    /// (<https://github.com/modelpack/model-spec>) OCI artifact stored in
    /// this layout. Each file becomes one uncompressed tar layer, classified
    /// into the appropriate `application/vnd.cncf.model.*` media type by
    /// extension (mirroring [`crate::sources::classify_file`], the
    /// equivalent classifier used when pulling from HuggingFace/cloud
    /// sources), with `org.cncf.model.filepath` recording its path.
    /// Returns the manifest descriptor.
    pub fn build(
        &self,
        src_dir: impl AsRef<Path>,
        reference: &str,
        labels: &std::collections::HashMap<String, String>,
    ) -> anyhow::Result<Descriptor> {
        use walkdir::WalkDir;

        let src_dir = src_dir.as_ref();

        // Canonicalize the source directory so that symlinks, NTFS
        // junctions, and volume mount points all resolve to the real
        // path before any files are walked.  `dunce::canonicalize`
        // avoids the `\\?\` verbatim-path prefix on Windows that would
        // break `strip_prefix` below.
        let canonical_src = dunce::canonicalize(src_dir)
            .with_context(|| format!("canonicalize source dir {}", src_dir.display()))?;

        let mut layers: Vec<Descriptor> = Vec::new();
        let mut format: Option<&'static str> = None;

        // One layer per file (uncompressed tar, filename preserved via annotations)
        for entry in WalkDir::new(&canonical_src).follow_links(true) {
            let entry = entry?;
            if !entry.file_type().is_file() {
                continue;
            }

            // Canonicalize each entry to resolve nested symlinks and
            // detect any that escape the source root.
            let canonical_entry = dunce::canonicalize(entry.path())
                .with_context(|| format!("canonicalize {}", entry.path().display()))?;
            if !canonical_entry.starts_with(&canonical_src) {
                return Err(anyhow!(
                    "path {} resolves to {} which is outside source dir {}",
                    entry.path().display(),
                    canonical_entry.display(),
                    canonical_src.display()
                ));
            }

            let rel = canonical_entry
                .strip_prefix(&canonical_src)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");

            let media_type = classify_model_layer(&rel);
            if media_type == WEIGHT_TAR_MEDIA_TYPE {
                let lower = rel.to_lowercase();
                if lower.ends_with(".gguf") || lower.ends_with(".ggml") {
                    format = Some("gguf");
                } else if lower.ends_with(".safetensors") && format.is_none() {
                    format = Some("safetensors");
                }
            }

            // Build a minimal tar with a single entry
            let tar_data = make_single_file_tar(&canonical_entry, &rel)?;
            let mut desc = self.write_blob(media_type, &tar_data)?;
            desc.annotations = Some({
                let mut m = std::collections::HashMap::new();
                m.insert("org.cncf.model.filepath".into(), rel.clone());
                m.insert("org.opencontainers.image.title".into(), rel);
                m
            });
            layers.push(desc);
        }

        if layers.is_empty() {
            return Err(anyhow!("no files found in {}", src_dir.display()));
        }

        // `application/vnd.cncf.model.config.v1+json` config. Since every
        // layer above is an uncompressed tar, each layer's DiffID (hash of
        // its uncompressed content) is simply its own digest.
        let cncf_config = CncfModelConfig {
            descriptor: CncfConfigDescriptor {
                created_at: Some(chrono::Utc::now().to_rfc3339()),
            },
            config: CncfConfigConfig {
                format: format.map(str::to_string),
            },
            modelfs: CncfModelFs {
                fs_type: "layers".into(),
                diff_ids: layers.iter().map(|l| l.digest.clone()).collect(),
            },
        };
        let config_data = serde_json::to_vec(&cncf_config)?;
        let config_desc =
            self.write_blob("application/vnd.cncf.model.config.v1+json", &config_data)?;

        // `--label key=value` pairs have no dedicated slot in the model
        // config schema, so they're carried as manifest annotations instead.
        let manifest_annotations = if labels.is_empty() {
            None
        } else {
            Some(labels.clone())
        };

        // Manifest
        let manifest = Manifest {
            schema_version: 2,
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            artifact_type: Some("application/vnd.cncf.model.manifest.v1+json".into()),
            config: config_desc,
            layers,
            annotations: manifest_annotations,
        };
        let manifest_desc = self.write_manifest(&manifest)?;
        self.tag(manifest_desc.clone(), reference)?;
        Ok(manifest_desc)
    }
}

// ---------------------------------------------------------------------------
// CNCF Model Format Spec layer classification
// ---------------------------------------------------------------------------

const WEIGHT_TAR_MEDIA_TYPE: &str = "application/vnd.cncf.model.weight.v1.tar";
const WEIGHT_CONFIG_TAR_MEDIA_TYPE: &str = "application/vnd.cncf.model.weight.config.v1.tar";
const DOC_TAR_MEDIA_TYPE: &str = "application/vnd.cncf.model.doc.v1.tar";
const CODE_TAR_MEDIA_TYPE: &str = "application/vnd.cncf.model.code.v1.tar";

/// Maps a file's relative path to the appropriate CNCF model layer media
/// type by extension, mirroring [`crate::sources::classify_file`] (used for
/// HuggingFace/cloud-source pulls). Uses the `.tar` variants because
/// `build()` wraps each file in its own uncompressed tar archive, rather
/// than the `.raw` variants a pulled, un-archived blob gets.
fn classify_model_layer(rel_path: &str) -> &'static str {
    let lower = rel_path.to_lowercase();
    let base = Path::new(&lower)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(lower.as_str());
    let ext = Path::new(base)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext {
        "safetensors" | "bin" | "pt" | "pth" | "gguf" | "ggml" | "ot" | "engine" | "trt"
        | "onnx" => WEIGHT_TAR_MEDIA_TYPE,
        // "jinja": a standalone chat_template.jinja file.
        "json" | "yaml" | "yml" | "toml" | "ini" | "cfg" | "conf" | "model" | "tiktoken"
        | "vocab" | "merges" | "spm" | "jinja" => WEIGHT_CONFIG_TAR_MEDIA_TYPE,
        "txt" => {
            if base.contains("vocab") || base.contains("merges") {
                WEIGHT_CONFIG_TAR_MEDIA_TYPE
            } else {
                DOC_TAR_MEDIA_TYPE
            }
        }
        "py" | "sh" | "js" | "ts" => CODE_TAR_MEDIA_TYPE,
        "md" | "rst" | "pdf" => DOC_TAR_MEDIA_TYPE,
        _ => {
            // README, LICENSE, and anything unrecognized default to doc,
            // same as the Go classifier's fallback.
            DOC_TAR_MEDIA_TYPE
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// True if the descriptor's ref annotation matches `reference` precisely
/// — exact match, tag-only match (a bare `reference` against a stored
/// `"reg/repo:tag"`), tagless match (`reference` with an implicit
/// `:latest` appended), or, for a `reference` carrying `@<digest>`, a
/// digest match (the branch below says what that compares). The tag-only
/// branch is unambiguous only when `reference` is a bare tag that's
/// unique across stored tags — two different repos sharing the same tag
/// (e.g. both `:0.8b`) can still both match here. A digest match is
/// ambiguous across the tags of one repository in the same way, and
/// `matching_index` is what settles that.
fn ref_matches_precise(desc: &Descriptor, reference: &str) -> bool {
    let Some(stored) = stored_ref_name(desc) else {
        return false;
    };

    // `<name>[:tag]@<digest>` names content, not a tag: it matches the
    // stored model with that manifest digest under the same repository,
    // whatever tag it was stored under. Without this a digest reference
    // for a model already in the store was "not found", and the daemon
    // pulled the same bytes again under a second reference and started a
    // second server for them. The hex is compared without regard to case
    // because `shortnames::validate_reference` accepts either while the
    // store records lowercase, so an uppercase spelling matched nothing.
    // Unlike the tag branches below, this one has no tag-only fallback: a
    // bare `m@sha256:…` against a stored `docker.io/ai/m:v9` misses on the
    // repository. Latent today, since each caller resolves a reference
    // through `shortnames::resolve_ollama_api` before it gets here.
    if let (base, Some(digest)) = split_ref_digest(reference) {
        return desc.digest.eq_ignore_ascii_case(digest) && repo_name(stored) == repo_name(base);
    }

    if stored == reference || tag_from_ref(stored) == reference {
        return true;
    }
    // `default_tag` also guards non-taggable sources (URI schemes, absolute
    // paths), so this can't accidentally match one of those against some
    // unrelated stored `"...:latest"`.
    if stored == default_tag(reference) {
        return true;
    }
    false
}

/// Splits `<name>[:tag]@<digest>` into the part before `@` and the digest.
/// The `@` has to follow the last `/`, so a host or namespace component
/// cannot be mistaken for one; a reference without a digest comes back
/// whole. `shortnames::parse_registry_ref` splits at the first `@`
/// instead; the two agree on any reference it has accepted, since no
/// part before the model admits an `@`. This one is also applied to
/// stored names, which reach `ref_matches_precise` from the annotation
/// and not through that parser.
pub fn split_ref_digest(reference: &str) -> (&str, Option<&str>) {
    // The same guard as `default_tag`: an absolute path or a URI is
    // opaque to `shortnames::validate_reference`, so an `@` in its last
    // component is part of the name, not a digest, and splitting it
    // would let `s3://bucket/model@sha256:…` resolve to a stored
    // `s3://bucket/model` of that digest instead of importing the key.
    if reference.starts_with('/') || reference.contains("://") {
        return (reference, None);
    }
    let last_slash = reference.rfind('/').map_or(0, |i| i + 1);
    match reference[last_slash..].find('@') {
        Some(offset) => {
            let at = last_slash + offset;
            (&reference[..at], Some(&reference[at + 1..]))
        }
        None => (reference, None),
    }
}

/// The repository part of a reference: everything before a tag or digest,
/// so `docker.io/ai/m:v9`, `docker.io/ai/m` and `docker.io/ai/m@sha256:…`
/// all give `docker.io/ai/m`. The tag is cut by the rule
/// `ref_path_segments` lays paths out with, a `:` after the last `/`, so
/// a host port stays.
pub fn repo_name(reference: &str) -> &str {
    let (base, _) = split_ref_digest(reference);
    let last_slash = base.rfind('/').map_or(0, |i| i + 1);
    match base[last_slash..].find(':') {
        Some(offset) => &base[..last_slash + offset],
        None => base,
    }
}

/// The index of the entry in `manifests` that `find`/`remove` should
/// treat as matching `reference`, or `None`.
///
/// One manifest can sit under several tags, so a digest reference can
/// match more than one entry. The tag the reference spells (`:latest`
/// when it spells none) wins: `m:v9@<digest>` with `m:v9` and `m:latest`
/// both at that digest picks `m:v9`, and a stored name with no tag
/// counts as `:latest`, as it does everywhere else in this store.
/// Between tags it does not spell, the first in the order `list_refs`
/// walks the tree wins, which is readdir order and not sorted; `remove`
/// refuses that case rather than take one (see its doc comment), `find`
/// answers with it.
fn matching_index(manifests: &[Descriptor], reference: &str) -> Option<usize> {
    let (base, digest) = split_ref_digest(reference);
    if digest.is_some() {
        let spelled = default_tag(base);
        let own_tag = manifests.iter().position(|m| {
            ref_matches_precise(m, reference)
                && stored_ref_name(m).is_some_and(|stored| default_tag(stored) == spelled)
        });
        if own_tag.is_some() {
            return own_tag;
        }
    }
    manifests
        .iter()
        .position(|m| ref_matches_precise(m, reference))
}

fn stored_ref_name(desc: &Descriptor) -> Option<&str> {
    desc.annotations
        .as_ref()
        .and_then(|a| a.get("org.opencontainers.image.ref.name"))
        .map(|s| s.as_str())
}

/// Splits `reference` into the path segments used to lay it out under
/// `manifests/` — one directory per "/"-delimited path segment, with an
/// explicit (defaulted, if absent — see `default_tag`) tag as the final
/// segment. Mirrors Ollama's `model.Name.Filepath()`
/// (`types/model/name.go`: `{host}/{namespace}/{model}/{tag}`),
/// generalized to also cover llmman's broader reference space
/// (HuggingFace-style `host/owner/repo:tag` references — already the same
/// shape — and non-registry sources, which `default_tag` already knows to
/// leave untouched since they never carry a tag).
///
/// Each segment is sanitized for safe use as a single path component: an
/// empty, "." or ".." segment (e.g. an empty tag from "repo:") becomes
/// "__" instead, since any of those would otherwise produce a malformed
/// or unintended path — see `sanitize_ref_segment`.
fn ref_path_segments(reference: &str) -> Vec<String> {
    let tagged = default_tag(reference);
    let (name, tag): (&str, &str) = match tagged.rfind(':') {
        Some(pos) if pos > tagged.rfind('/').unwrap_or(0) => (&tagged[..pos], &tagged[pos + 1..]),
        _ => (tagged.as_str(), "latest"),
    };
    name.split('/')
        .filter(|s| !s.is_empty())
        .map(sanitize_ref_segment)
        .chain(std::iter::once(sanitize_ref_segment(tag)))
        .collect()
}

/// Neutralizes characters that would be unsafe or misleading as a single
/// path segment: a literal `:` or `\` (a Windows path separator, or a
/// drive-letter colon, that could otherwise land inside what's meant to
/// be one segment — e.g. an absolute Windows path, or a URI scheme's
/// `://`) becomes `_`, and a segment that's exactly `..` is neutralized,
/// so no reference can ever escape the `manifests/` tree it's rooted
/// under.
fn sanitize_ref_segment(s: &str) -> String {
    if s.is_empty() || s == "." || s == ".." {
        return "__".to_string();
    }
    s.replace([':', '\\'], "_")
}

/// Recursively collects every manifest-pointer file under `dir` into
/// `out` — see `OciStore::list_refs`.
fn collect_refs(dir: &Path, out: &mut Vec<Descriptor>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return; // an unreadable subtree must not hide every other model
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            collect_refs(&path, out);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("tmp") {
            continue; // an in-progress or abandoned write — see write_ref
        }
        let Ok(data) = fs::read(&path) else { continue };
        if let Ok(desc) = serde_json::from_slice::<Descriptor>(&data) {
            out.push(desc);
        }
    }
}

/// Strict counterpart to [`collect_refs`] for the GC path: any error that
/// `collect_refs` would swallow (an unreadable directory, an unreadable or
/// unparsable pointer file) is propagated instead, so an incomplete
/// enumeration can never be mistaken for "these blobs are unreferenced". A
/// non-existent `dir` (a store with no `manifests/` yet) is treated as
/// empty — that's a genuinely empty reference set, not a read failure.
fn collect_refs_strict(dir: &Path, out: &mut Vec<Descriptor>) -> anyhow::Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e).with_context(|| format!("read {}", dir.display())),
    };
    for entry in entries {
        let entry = entry.with_context(|| format!("read entry in {}", dir.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("stat {}", path.display()))?;
        if file_type.is_dir() {
            collect_refs_strict(&path, out)?;
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("tmp") {
            continue; // an in-progress or abandoned write — see write_ref
        }
        let data = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let desc = serde_json::from_slice::<Descriptor>(&data)
            .with_context(|| format!("parse manifest ref {}", path.display()))?;
        out.push(desc);
    }
    Ok(())
}

/// reference and its `:latest` spelling are one string wherever a
/// reference is used as a map/lock key. A no-op on anything that isn't a
/// taggable registry reference — a URI-scheme source (`s3://`, `ngc://`,
/// `ms://`, ...) or an absolute local path — since those never carry a
/// tag and appending one would corrupt them.
pub fn default_tag(reference: &str) -> String {
    if reference.starts_with('/') || reference.contains("://") {
        return reference.to_owned();
    }
    let after_slash = &reference[reference.rfind('/').unwrap_or(0)..];
    if after_slash.contains(':') {
        reference.to_owned()
    } else {
        format!("{reference}:latest")
    }
}

fn split_digest(digest: &str) -> anyhow::Result<(&str, &str)> {
    let mut parts = digest.splitn(2, ':');
    let algo = parts
        .next()
        .ok_or_else(|| anyhow!("invalid digest: {}", digest))?;
    let hex = parts
        .next()
        .ok_or_else(|| anyhow!("invalid digest: {}", digest))?;
    Ok((algo, hex))
}

pub fn tag_from_ref(reference: &str) -> &str {
    if let Some(pos) = reference.rfind(':') {
        if pos > reference.rfind('/').unwrap_or(0) {
            return &reference[pos + 1..];
        }
    }
    "latest"
}

/// Build an in-memory uncompressed tar archive containing a single file.
fn make_single_file_tar(path: &Path, name: &str) -> anyhow::Result<Vec<u8>> {
    let file_data = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut buf = Vec::new();
    {
        let mut archive = tar::Builder::new(&mut buf);
        let mut header = tar::Header::new_gnu();
        header.set_path(name)?;
        header.set_size(file_data.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        archive.append(&header, file_data.as_slice())?;
        archive.finish()?;
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc_with_ref(ref_name: &str) -> Descriptor {
        let mut ann = std::collections::HashMap::new();
        ann.insert(
            "org.opencontainers.image.ref.name".to_string(),
            ref_name.to_string(),
        );
        Descriptor {
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            digest: "sha256:deadbeef".into(),
            size: 123,
            annotations: Some(ann),
        }
    }

    /// The docker/containerd backend's own writer (go-shim/manifest_ref.go's
    /// writeManifestRef) always stores the full reference it was called with.
    #[test]
    fn ref_matches_precise_a_full_reference_stored_verbatim() {
        let d = desc_with_ref("docker.io/ai/qwen3.5:0.8b");
        assert!(ref_matches_precise(&d, "docker.io/ai/qwen3.5:0.8b"));
        assert!(!ref_matches_precise(&d, "docker.io/ai/qwen3.5:1.5b"));
        assert!(!ref_matches_precise(&d, "docker.io/ai/other:0.8b"));
    }

    /// Pins the digest branch of `ref_matches_precise` (see its own
    /// comment for why it exists): same digest and same repository match
    /// regardless of tag, and a different digest or repository does not.
    #[test]
    fn ref_matches_precise_resolves_a_digest_reference_by_content() {
        let mut d = desc_with_ref("docker.io/ai/m:v9");
        d.digest = "sha256:aaaa".into();
        assert!(ref_matches_precise(&d, "docker.io/ai/m@sha256:aaaa"));
        assert!(ref_matches_precise(&d, "docker.io/ai/m:latest@sha256:aaaa"));
        assert!(!ref_matches_precise(&d, "docker.io/ai/m@sha256:bbbb"));
        assert!(!ref_matches_precise(&d, "docker.io/ai/other@sha256:aaaa"));
        // A tag beside the digest does not narrow the match: the grammar
        // in docker/distribution's `reference.go` allows both, and there
        // the digest is what names content. `matching_index` is where the
        // tag counts, as a tie-breaker.
        assert!(ref_matches_precise(&d, "docker.io/ai/m:other@sha256:aaaa"));
        assert!(ref_matches_precise(&d, "docker.io/ai/m@sha256:AAAA"));
    }

    /// Two tags at one digest: the spelled tag is chosen, and without one
    /// the first entry is. See `matching_index`'s doc comment.
    #[test]
    fn matching_index_prefers_the_tag_a_digest_reference_spells() {
        let mut latest = desc_with_ref("docker.io/ai/m:latest");
        latest.digest = "sha256:aaaa".into();
        let mut v9 = desc_with_ref("docker.io/ai/m:v9");
        v9.digest = "sha256:aaaa".into();
        let manifests = [latest, v9];
        assert_eq!(
            matching_index(&manifests, "docker.io/ai/m:v9@sha256:aaaa"),
            Some(1)
        );
        assert_eq!(
            matching_index(&manifests, "docker.io/ai/m@sha256:aaaa"),
            Some(0)
        );
        assert_eq!(
            matching_index(&manifests, "docker.io/ai/m:v8@sha256:aaaa"),
            Some(0)
        );
        assert_eq!(
            matching_index(&manifests, "docker.io/ai/m@sha256:bbbb"),
            None
        );

        // A tagless stored name is `:latest` here too, ahead of `:v9`
        // whatever order the tree is walked in.
        let mut tagless = desc_with_ref("docker.io/ai/m");
        tagless.digest = "sha256:aaaa".into();
        let mut v9 = desc_with_ref("docker.io/ai/m:v9");
        v9.digest = "sha256:aaaa".into();
        assert_eq!(
            matching_index(&[v9, tagless], "docker.io/ai/m@sha256:aaaa"),
            Some(1)
        );
    }

    #[test]
    fn split_ref_digest_only_splits_after_the_last_slash() {
        assert_eq!(
            split_ref_digest("docker.io/ai/m@sha256:aa"),
            ("docker.io/ai/m", Some("sha256:aa"))
        );
        assert_eq!(
            split_ref_digest("docker.io/ai/m:v9@sha256:aa"),
            ("docker.io/ai/m:v9", Some("sha256:aa"))
        );
        assert_eq!(
            split_ref_digest("docker.io/ai/m:v9"),
            ("docker.io/ai/m:v9", None)
        );
        // Opaque sources keep their `@`, as `default_tag` keeps their `:`.
        assert_eq!(
            split_ref_digest("s3://bucket/model@sha256:aa"),
            ("s3://bucket/model@sha256:aa", None)
        );
        assert_eq!(
            split_ref_digest("/models/m@sha256:aa"),
            ("/models/m@sha256:aa", None)
        );
        assert_eq!(repo_name("docker.io/ai/m:v9@sha256:aa"), "docker.io/ai/m");
        assert_eq!(repo_name("localhost:5000/m"), "localhost:5000/m");
        assert_eq!(repo_name("localhost:5000/m:v1"), "localhost:5000/m");
    }

    /// A pointer at a digest-shaped path that holds another digest is not
    /// what a lookup by that digest gets; the content decides, and the
    /// pointer itself can still be removed by name.
    #[test]
    fn find_passes_over_a_digest_named_pointer_holding_other_content() {
        let dir = temp_store_dir("digest-pointer");
        let store = OciStore::open(&dir).unwrap();
        let mut lying = desc_with_ref("unused");
        lying.digest = "sha256:aaaa".into();
        store.tag(lying, "docker.io/ai/m@sha256:bbbb").unwrap();

        assert!(store.find("docker.io/ai/m@sha256:bbbb").is_err());
        assert_eq!(
            store.find("docker.io/ai/m@sha256:aaaa").unwrap().digest,
            "sha256:aaaa",
            "by content, the pointer is the entry for the digest it holds"
        );
        let mut real = desc_with_ref("unused");
        real.digest = "sha256:bbbb".into();
        store.tag(real, "docker.io/ai/m:v9").unwrap();
        assert_eq!(
            store
                .find("docker.io/ai/m@sha256:bbbb")
                .unwrap()
                .annotations
                .unwrap()["org.opencontainers.image.ref.name"],
            "docker.io/ai/m:v9"
        );
        store.remove("docker.io/ai/m@sha256:bbbb").unwrap();
        assert!(store.find("docker.io/ai/m@sha256:aaaa").is_err());
        assert!(store.find("docker.io/ai/m:v9").is_ok());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A digest held by several tags: the removal is refused with every
    /// holder listed, a bare `m@sha256:…` included even with `:latest`
    /// among them, unless the reference spells one of the tags; then that
    /// one goes and is named. Nothing goes in readdir order.
    #[test]
    fn remove_by_digest_names_what_went_and_refuses_an_ambiguous_one() {
        let dir = temp_store_dir("digest-remove");
        let store = OciStore::open(&dir).unwrap();
        let desc = |digest: &str| {
            let mut d = desc_with_ref("unused");
            d.digest = digest.into();
            d
        };
        store.tag(desc("sha256:aaaa"), "docker.io/ai/m:v8").unwrap();
        store.tag(desc("sha256:aaaa"), "docker.io/ai/m:v9").unwrap();
        store
            .tag(desc("sha256:aaaa"), "docker.io/ai/m:latest")
            .unwrap();

        let err = store.remove("docker.io/ai/m@sha256:aaaa").unwrap_err();
        let msg = err.to_string();
        assert!(
            [
                "docker.io/ai/m:v8",
                "docker.io/ai/m:v9",
                "docker.io/ai/m:latest"
            ]
            .iter()
            .all(|t| msg.contains(t)),
            "{msg}"
        );
        assert!(store.find("docker.io/ai/m:latest").is_ok());

        assert_eq!(
            store.remove("docker.io/ai/m:latest@sha256:aaaa").unwrap(),
            "docker.io/ai/m:latest"
        );
        assert_eq!(
            store.remove("docker.io/ai/m:v9@sha256:aaaa").unwrap(),
            "docker.io/ai/m:v9"
        );
        assert_eq!(
            store.remove("docker.io/ai/m@sha256:aaaa").unwrap(),
            "docker.io/ai/m:v8",
            "one holder left is not ambiguous, and the name that went is reported"
        );
        assert!(store.remove("docker.io/ai/m@sha256:aaaa").is_err());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Same branch through `find`: stored under a tag, looked up by digest.
    #[test]
    fn find_resolves_a_digest_reference_to_the_tagged_entry() {
        let dir = temp_store_dir("digest-find");
        let store = OciStore::open(&dir).unwrap();
        let mut d = desc_with_ref("unused");
        d.digest = "sha256:cafe".into();
        store.tag(d, "docker.io/ai/m:latest").unwrap();

        let found = store.find("docker.io/ai/m@sha256:cafe").unwrap();
        assert_eq!(
            found.annotations.unwrap()["org.opencontainers.image.ref.name"],
            "docker.io/ai/m:latest"
        );
        assert!(store.find("docker.io/ai/m@sha256:dead").is_err());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn ref_matches_precise_defaults_a_tagless_reference_to_latest() {
        let d = desc_with_ref("docker.io/ai/qwen3.5:latest");
        assert!(ref_matches_precise(&d, "docker.io/ai/qwen3.5"));
    }

    #[test]
    fn default_tag_appends_latest_only_when_no_tag_is_present() {
        assert_eq!(
            default_tag("docker.io/ai/gemma4"),
            "docker.io/ai/gemma4:latest"
        );
        assert_eq!(
            default_tag("docker.io/ai/gemma4:latest"),
            "docker.io/ai/gemma4:latest"
        );
        assert_eq!(
            default_tag("docker.io/ai/gemma4:e4b"),
            "docker.io/ai/gemma4:e4b"
        );
        // A colon before the last '/' (a port in a registry host) must not
        // be mistaken for a tag separator.
        assert_eq!(
            default_tag("localhost:5000/gemma4"),
            "localhost:5000/gemma4:latest"
        );
    }

    #[test]
    fn default_tag_leaves_non_registry_sources_untouched() {
        // These never carry a tag; appending one would corrupt them.
        assert_eq!(default_tag("ngc://org/model"), "ngc://org/model");
        assert_eq!(default_tag("s3://bucket/key"), "s3://bucket/key");
        assert_eq!(default_tag("gs://bucket/key"), "gs://bucket/key");
        assert_eq!(default_tag("ms://owner/repo"), "ms://owner/repo");
        assert_eq!(default_tag("/abs/path/model.gguf"), "/abs/path/model.gguf");
    }

    #[test]
    fn ref_matches_precise_returns_false_without_a_ref_name_annotation() {
        let d = Descriptor {
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            digest: "sha256:deadbeef".into(),
            size: 123,
            annotations: None,
        };
        assert!(!ref_matches_precise(&d, "docker.io/ai/qwen3.5:0.8b"));
    }

    /// An unrelated model stored under a bare "latest" tag must not
    /// hijack a precisely-matching tagless reference (which also
    /// defaults to "latest").
    #[test]
    fn matching_index_prefers_a_precise_match_over_an_unrelated_bare_latest() {
        let unrelated_bare_latest = desc_with_ref("latest");
        let inferact = desc_with_ref("hf.co/Inferact/Qwen3.8-27B-NVFP4:latest");
        let manifests = vec![unrelated_bare_latest, inferact.clone()];

        let i = matching_index(&manifests, "hf.co/Inferact/Qwen3.8-27B-NVFP4")
            .expect("must find a match");
        assert_eq!(manifests[i].digest, inferact.digest);

        let i = matching_index(&manifests, "hf.co/Inferact/Qwen3.8-27B-NVFP4:latest")
            .expect("must find a match");
        assert_eq!(manifests[i].digest, inferact.digest);
    }

    #[test]
    fn tag_from_ref_extracts_the_tag_after_the_last_slash() {
        assert_eq!(tag_from_ref("docker.io/ai/qwen3.5:0.8b"), "0.8b");
        assert_eq!(tag_from_ref("docker.io/ai/qwen3.5"), "latest");
        assert_eq!(tag_from_ref("0.8b"), "latest");
    }

    /// Ported from ollama's types/model/name_test.go
    /// ("host:port/namespace/model:tag" and the tagless default): a colon
    /// inside the host component must never be mistaken for a tag
    /// separator, and a reference without a tag defaults to "latest".
    #[test]
    fn tag_from_ref_ignores_a_port_in_the_host_component() {
        assert_eq!(tag_from_ref("example.com:5000/ns/model"), "latest");
        assert_eq!(tag_from_ref("example.com:5000/ns/model:tag"), "tag");
        assert_eq!(tag_from_ref("localhost:11434/library/mistral:7b"), "7b");
    }

    // ---------------------------------------------------------------
    // One-file-per-model storage (OciStore itself)
    // ---------------------------------------------------------------

    fn temp_store_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "llmman-oci-test-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn ref_path_segments_mirrors_ollama_host_namespace_model_tag() {
        assert_eq!(
            ref_path_segments("docker.io/ai/qwen3.5:0.8b"),
            vec!["docker.io", "ai", "qwen3.5", "0.8b"]
        );
        assert_eq!(
            ref_path_segments("docker.io/ai/qwen3.5"),
            vec!["docker.io", "ai", "qwen3.5", "latest"]
        );
    }

    #[test]
    fn ref_path_segments_sanitizes_unsafe_characters() {
        // A URI-scheme source's "://" and a bare ".." must never produce
        // a path component that could escape manifests/ or break on
        // Windows.
        assert_eq!(
            ref_path_segments("s3://bucket/key"),
            vec!["s3_", "bucket", "key", "latest"]
        );
        assert_eq!(
            ref_path_segments("../../etc/passwd"),
            vec!["__", "__", "etc", "passwd", "latest"]
        );
        // An empty tag ("repo:") must not vanish: an empty final segment
        // would collapse into the repo directory itself.
        assert_eq!(
            ref_path_segments("docker.io/ai/x:"),
            vec!["docker.io", "ai", "x", "__"]
        );
    }

    /// Core regression test for this change: two different references
    /// must resolve to two different files, so a torn write to one
    /// model's manifest file can never affect any other model's own file
    /// — unlike the old shared index.json, where every model's entry
    /// lived in the same file.
    #[test]
    fn tag_and_find_round_trip_one_file_per_model() {
        let dir = temp_store_dir("round-trip");
        let store = OciStore::open(&dir).unwrap();

        store
            .tag(desc_with_ref("unused"), "docker.io/ai/qwen3.5:0.8b")
            .unwrap();
        store
            .tag(desc_with_ref("unused"), "docker.io/ai/gemma4:latest")
            .unwrap();

        let a = store.ref_path("docker.io/ai/qwen3.5:0.8b");
        let b = store.ref_path("docker.io/ai/gemma4:latest");
        assert_ne!(a, b, "distinct references must live at distinct paths");
        assert!(a.is_file());
        assert!(b.is_file());

        let found = store.find("docker.io/ai/qwen3.5:0.8b").unwrap();
        assert_eq!(
            found.annotations.unwrap()["org.opencontainers.image.ref.name"],
            "docker.io/ai/qwen3.5:0.8b"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn write_ref_is_atomic_and_leaves_no_tmp_file() {
        let dir = temp_store_dir("atomic");
        let store = OciStore::open(&dir).unwrap();
        store
            .tag(desc_with_ref("unused"), "docker.io/ai/gemma4:latest")
            .unwrap();

        let path = store.ref_path("docker.io/ai/gemma4:latest");
        assert!(path.is_file());
        let leftover_tmp = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().is_some_and(|ext| ext == "tmp"));
        assert!(
            !leftover_tmp,
            "no .tmp file should survive a successful write"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn list_and_remove_work_across_many_models() {
        let dir = temp_store_dir("list-remove");
        let store = OciStore::open(&dir).unwrap();
        for i in 0..5 {
            store
                .tag(
                    desc_with_ref("unused"),
                    &format!("docker.io/ai/model-{i}:latest"),
                )
                .unwrap();
        }

        let images = store.list().unwrap();
        assert_eq!(images.len(), 5);

        store.remove("docker.io/ai/model-2:latest").unwrap();
        let images = store.list().unwrap();
        assert_eq!(images.len(), 4);
        assert!(!images
            .iter()
            .any(|i| i.reference == "docker.io/ai/model-2:latest"));

        assert!(store.remove("docker.io/ai/model-2:latest").is_err());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Regression test: one unreadable directory under manifests/ must
    /// not hide every other model from `list`.
    #[test]
    #[cfg(unix)]
    fn list_skips_an_unreadable_subtree() {
        use std::os::unix::fs::PermissionsExt;

        let dir = temp_store_dir("unreadable-subtree");
        let store = OciStore::open(&dir).unwrap();
        store
            .tag(desc_with_ref("unused"), "docker.io/ai/ok:latest")
            .unwrap();

        let blocked = dir.join("manifests").join("unreadable");
        std::fs::create_dir_all(&blocked).unwrap();
        std::fs::set_permissions(&blocked, std::fs::Permissions::from_mode(0o0)).unwrap();

        let images = store.list().unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].reference, "docker.io/ai/ok:latest");

        std::fs::set_permissions(&blocked, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Concurrent writers of *distinct* references must never lose an
    /// entry — the whole point of one file per model is that there's no
    /// shared read-modify-write cycle left for them to race on at all,
    /// unlike the old shared index.json (see this repo's git history).
    #[test]
    fn concurrent_tag_calls_across_distinct_refs_lose_nothing() {
        let dir = temp_store_dir("concurrent");
        let store = std::sync::Arc::new(OciStore::open(&dir).unwrap());

        let handles: Vec<_> = (0..25)
            .map(|i| {
                let store = store.clone();
                std::thread::spawn(move || {
                    store
                        .tag(
                            desc_with_ref("unused"),
                            &format!("docker.io/ai/model-{i}:latest"),
                        )
                        .unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        let images = store.list().unwrap();
        assert_eq!(images.len(), 25);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Regression test: two concurrent writers of the *same* reference
    /// must each produce a valid, parsable file — never a truncated or
    /// interleaved one from racing on a shared temp file name.
    #[test]
    fn concurrent_writes_to_the_same_ref_never_corrupt_it() {
        let dir = temp_store_dir("concurrent-same-ref");
        let store = std::sync::Arc::new(OciStore::open(&dir).unwrap());

        let handles: Vec<_> = (0..25)
            .map(|i| {
                let store = store.clone();
                std::thread::spawn(move || {
                    let mut desc = desc_with_ref("unused");
                    desc.digest = format!("sha256:{i:064x}");
                    store.tag(desc, "docker.io/ai/same:latest").unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        // Whichever writer's rename won last, the result must be one
        // complete, valid entry — not a parse failure from interleaved
        // bytes.
        assert!(store.find("docker.io/ai/same:latest").is_ok());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Regression test for a real bug: `Path::with_extension("tmp")`
    /// replaces whatever follows the last '.' in the file name — for a
    /// tag like "0.8b" that turns the temp path into ".../0.tmp",
    /// clobbering another tag's temp file (e.g. "0.5b" would collide on
    /// the same ".../0.tmp"). write_ref must append ".tmp" as a suffix
    /// instead.
    #[test]
    fn tag_with_a_dot_does_not_collide_with_a_different_tag() {
        let dir = temp_store_dir("dotted-tag");
        let store = OciStore::open(&dir).unwrap();
        store
            .tag(desc_with_ref("unused"), "docker.io/ai/x:0.8b")
            .unwrap();
        store
            .tag(desc_with_ref("unused"), "docker.io/ai/x:0.5b")
            .unwrap();

        let images = store.list().unwrap();
        assert_eq!(images.len(), 2, "distinct dotted tags must not collide");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Regression test: an empty tag ("repo:") must not clobber the
    /// repo's own directory, or every subsequent tag of that repo would
    /// fail.
    #[test]
    fn tag_with_an_empty_tag_does_not_break_other_tags() {
        let dir = temp_store_dir("empty-tag");
        let store = OciStore::open(&dir).unwrap();
        store
            .tag(desc_with_ref("unused"), "docker.io/ai/x:")
            .unwrap();
        store
            .tag(desc_with_ref("unused"), "docker.io/ai/x:latest")
            .unwrap();

        assert!(store.find("docker.io/ai/x:latest").is_ok());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A corrupt (present but unparsable) ref file must not look like a
    /// typo — `find` should surface the parse error, not "not found".
    #[test]
    fn find_distinguishes_a_corrupt_ref_from_a_missing_one() {
        let dir = temp_store_dir("corrupt-ref");
        let store = OciStore::open(&dir).unwrap();
        let path = store.ref_path("docker.io/ai/broken:latest");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not json").unwrap();

        let err = store.find("docker.io/ai/broken:latest").unwrap_err();
        assert!(!err.to_string().contains("not found"), "got: {err}");

        std::fs::remove_dir_all(&dir).unwrap();
    }
}

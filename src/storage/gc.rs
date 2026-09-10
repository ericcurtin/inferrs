//! Content-addressed garbage collection for the blob store and the
//! extracted-model cache.
//!
//! Because the store is fully content-addressed and every tag is just a
//! small pointer file under `manifests/` (see [`crate::storage::oci`]),
//! "what's still needed" can always be recomputed from scratch by reading
//! every *surviving* manifest — there's no refcount or journal to keep in
//! sync (a crash, a manual edit, an interrupted pull can never desync it,
//! because the live set is rebuilt fresh on every sweep).
//!
//! [`referenced_digests`] builds that live set; [`prune_blobs`] and
//! [`prune_cache`] sweep everything not in it. Both are grace-gated the
//! same way [`crate::storage::repair`] gates its stale-temp-file sweep: a
//! blob is written before its manifest/tag pointer, so a blob can be
//! legitimately unreferenced for a moment mid-pull — anything younger than
//! `grace` is left alone. Both `rm` and the `serve` startup catch-all pass
//! the hour-long [`GC_GRACE_PERIOD`]: `pull` runs in the long-lived `serve`
//! daemon, so a concurrent `rm` in a separate process can race a pull that
//! has written a layer blob but not yet tagged its manifest — only the
//! grace window, not `rm` being synchronous, protects that blob.

use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, SystemTime};

use anyhow::Context;

use super::OciStore;

/// Grace window for the startup catch-all sweep — matches
/// [`crate::storage::repair::STALE_TMP_FILE_AGE`], so there's one duration
/// to reason about for "how long might a just-written blob be legitimately
/// unreferenced".
pub const GC_GRACE_PERIOD: Duration = Duration::from_secs(60 * 60);

/// What a sweep freed, for reporting.
#[derive(Debug, Default, Clone, Copy)]
pub struct GcStats {
    pub count: usize,
    pub bytes: u64,
}

/// Every blob digest ("sha256:<hex>") still reachable from a surviving
/// tag: each manifest's own digest, its config digest, and every layer
/// digest. Built by walking [`OciStore::list_refs_strict`] and reading each
/// manifest — the same traversal `resolve_model` does per-reference, just
/// over every reference at once.
///
/// This is the authority for a *destructive* sweep, so it fails closed:
/// any reference that can't be enumerated (an unreadable pointer file or
/// subtree) or any manifest that can't be read/parsed aborts the whole
/// computation with an error. The alternative — skipping the unreadable
/// entry, as the display-oriented `list_refs` does — would make that
/// model's config/layers look unreferenced and get deleted, possibly
/// destroying blobs shared with other healthy models. When we can't prove
/// what's still referenced, callers must delete nothing.
pub fn referenced_digests(store: &OciStore) -> anyhow::Result<HashSet<String>> {
    let mut live = HashSet::new();
    for desc in store.list_refs_strict()? {
        live.insert(desc.digest.clone());
        let manifest = store
            .read_manifest(&desc.digest)
            .with_context(|| format!("read manifest {} for GC reference scan", desc.digest))?;
        live.insert(manifest.config.digest.clone());
        live.extend(manifest.layers.iter().map(|l| l.digest.clone()));
    }
    Ok(live)
}

/// Deletes every blob file under `blobs/sha256/` whose `sha256:<name>`
/// digest isn't in `live`, skipping in-progress temp writes (`tmp-`/
/// `.tmp`, same as [`crate::storage::repair`]) and anything younger than
/// `grace`. A missing blobs directory is a no-op.
pub fn prune_blobs(
    store_root: &Path,
    live: &HashSet<String>,
    grace: Duration,
) -> anyhow::Result<GcStats> {
    let blobs_dir = store_root.join("blobs").join("sha256");
    let mut stats = GcStats::default();
    let entries = match std::fs::read_dir(&blobs_dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(stats),
        Err(e) => return Err(e).with_context(|| format!("read {}", blobs_dir.display())),
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // In-progress or abandoned write — left to repair's own sweep.
        if name.starts_with("tmp-") || name.ends_with(".tmp") {
            continue;
        }
        if live.contains(&format!("sha256:{name}")) {
            continue;
        }
        if !is_older_than(&path, grace) {
            continue;
        }
        let size = freed_file_size(&path);
        if let Err(e) = std::fs::remove_file(&path) {
            eprintln!("[llmman] couldn't remove unreferenced blob {name}: {e:#}");
            continue;
        }
        stats.count += 1;
        stats.bytes += size;
    }
    Ok(stats)
}

/// Deletes every cache subdirectory under `cache_path` whose name (a layer
/// hex for GGUF, a manifest hex for safetensors — see
/// `modelpack::extract_gguf_layer` / `extract_safetensors_dir`) doesn't
/// correspond to a live digest, skipping anything younger than `grace`. A
/// missing cache directory is a no-op.
pub fn prune_cache(
    cache_path: &Path,
    live: &HashSet<String>,
    grace: Duration,
) -> anyhow::Result<GcStats> {
    let live_hex: HashSet<&str> = live
        .iter()
        .filter_map(|d| d.strip_prefix("sha256:"))
        .collect();
    let mut stats = GcStats::default();
    let entries = match std::fs::read_dir(cache_path) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(stats),
        Err(e) => return Err(e).with_context(|| format!("read {}", cache_path.display())),
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if live_hex.contains(name) {
            continue;
        }
        if !is_older_than(&path, grace) {
            continue;
        }
        let size = dir_size(&path);
        if let Err(e) = std::fs::remove_dir_all(&path) {
            eprintln!("[llmman] couldn't remove unreferenced cache dir {name}: {e:#}");
            continue;
        }
        stats.count += 1;
        stats.bytes += size;
    }
    Ok(stats)
}

/// True if `path`'s mtime is at least `grace` old. A zero `grace` makes
/// this always true (used only by tests that want to sweep regardless of
/// age). A file whose mtime can't be read is treated as not-yet-old, so
/// it's left alone rather than deleted on a metadata hiccup.
fn is_older_than(path: &Path, grace: Duration) -> bool {
    if grace.is_zero() {
        return true;
    }
    let Ok(modified) = std::fs::metadata(path).and_then(|m| m.modified()) else {
        return false;
    };
    SystemTime::now()
        .duration_since(modified)
        .is_ok_and(|age| age >= grace)
}

/// Total size of the files directly inside `dir` (cache dirs are flat —
/// extracted GGUF/safetensors files, no nesting). Best-effort: unreadable
/// entries contribute 0.
fn dir_size(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries.flatten().map(|e| freed_file_size(&e.path())).sum()
}

fn freed_file_size(path: &Path) -> u64 {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return 0;
    };
    if meta.is_file() && !has_multiple_links(&meta) {
        meta.len()
    } else {
        0
    }
}

#[cfg(unix)]
fn has_multiple_links(meta: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt as _;
    meta.nlink() > 1
}

#[cfg(not(unix))]
fn has_multiple_links(_meta: &std::fs::Metadata) -> bool {
    false
}

/// Skips both the post-`rm` and startup GC sweeps when `LLMMAN_NOPRUNE` is
/// set to anything other than an explicit falsy value — an escape hatch
/// for shared/read-mostly stores or scripts that `rm` in a loop and would
/// rather prune once at the end themselves. Read fresh at each call site,
/// like every other `LLMMAN_*` var.
pub fn noprune_from_env() -> bool {
    crate::env_flag_set("LLMMAN_NOPRUNE")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "llmman-gc-test-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn prune_blobs_removes_only_unreferenced_blobs() {
        let root = temp_dir("prune-blobs");
        let blobs = root.join("blobs").join("sha256");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("aaaa"), b"referenced").unwrap();
        std::fs::write(blobs.join("bbbb"), b"orphan").unwrap();
        std::fs::write(blobs.join("tmp-123"), b"in-progress").unwrap();

        let mut live = HashSet::new();
        live.insert("sha256:aaaa".to_string());

        let stats = prune_blobs(&root, &live, Duration::ZERO).unwrap();
        assert_eq!(stats.count, 1);
        assert!(blobs.join("aaaa").exists(), "referenced blob must survive");
        assert!(!blobs.join("bbbb").exists(), "orphan blob must be removed");
        assert!(
            blobs.join("tmp-123").exists(),
            "in-progress temp file must be left to repair's sweep"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The concurrent-pull safety property: an unreferenced blob younger
    /// than the grace window is left alone (it may be a layer an in-flight
    /// pull has already written but not yet tagged), while an equally
    /// unreferenced but older blob is swept. Without this, a concurrent
    /// `rm` could delete a live pull's just-written layer.
    #[test]
    #[cfg(unix)]
    fn prune_blobs_respects_the_grace_window() {
        let root = temp_dir("prune-blobs-grace");
        let blobs = root.join("blobs").join("sha256");
        std::fs::create_dir_all(&blobs).unwrap();
        let fresh = blobs.join("cccc");
        let stale = blobs.join("dddd");
        std::fs::write(&fresh, b"just written by an in-flight pull").unwrap();
        std::fs::write(&stale, b"long-abandoned orphan").unwrap();
        // Back-date the stale blob past the grace window.
        let old = SystemTime::now() - GC_GRACE_PERIOD - Duration::from_secs(60);
        filetime_set(&stale, old);

        let live = HashSet::new(); // neither is referenced
        let stats = prune_blobs(&root, &live, GC_GRACE_PERIOD).unwrap();

        assert_eq!(stats.count, 1);
        assert!(
            fresh.exists(),
            "a fresh unreferenced blob must survive — it may be a live pull's untagged layer"
        );
        assert!(!stale.exists(), "a stale unreferenced blob must be swept");

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn prune_cache_removes_only_unreferenced_dirs() {
        let cache = temp_dir("prune-cache");
        std::fs::create_dir_all(cache.join("aaaa")).unwrap();
        std::fs::write(cache.join("aaaa").join("model.gguf"), b"kept").unwrap();
        std::fs::create_dir_all(cache.join("bbbb")).unwrap();
        std::fs::write(cache.join("bbbb").join("model.gguf"), b"orphan").unwrap();

        let mut live = HashSet::new();
        live.insert("sha256:aaaa".to_string());

        let stats = prune_cache(&cache, &live, Duration::ZERO).unwrap();
        assert_eq!(stats.count, 1);
        assert!(cache.join("aaaa").exists(), "referenced cache dir survives");
        assert!(!cache.join("bbbb").exists(), "orphan cache dir removed");

        std::fs::remove_dir_all(&cache).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn prune_counts_hardlinked_blob_and_cache_bytes_once() {
        let root = temp_dir("prune-hardlinked-cache");
        let blobs = root.join("blobs").join("sha256");
        let cache = root.join("cache");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(cache.join("bbbb")).unwrap();
        let blob = blobs.join("aaaa");
        let cache_file = cache.join("bbbb").join("model.safetensors");
        let weights = b"complete-weights-bytes";
        std::fs::write(&blob, weights).unwrap();
        std::fs::hard_link(&blob, &cache_file).unwrap();

        let live = HashSet::new();
        let blob_stats = prune_blobs(&root, &live, Duration::ZERO).unwrap();
        let cache_stats = prune_cache(&cache, &live, Duration::ZERO).unwrap();

        assert_eq!(blob_stats.count, 1);
        assert_eq!(cache_stats.count, 1);
        assert_eq!(blob_stats.bytes + cache_stats.bytes, weights.len() as u64);
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A corrupt (unparsable) manifest pointer file must abort the live-set
    /// computation rather than silently omitting that model — otherwise its
    /// blobs would look unreferenced and a destructive sweep would delete
    /// them. Fail closed: an error, so callers prune nothing.
    #[test]
    fn referenced_digests_aborts_on_an_unreadable_reference() {
        let root = temp_dir("referenced-digests-strict");
        let store = OciStore::open(&root).unwrap();

        // A real, healthy tagged model.
        let cfg = store
            .write_blob("application/vnd.cncf.model.config.v1+json", b"{}")
            .unwrap();
        let manifest = crate::storage::oci::Manifest {
            schema_version: 2,
            media_type: "application/vnd.oci.image.manifest.v1+json".into(),
            artifact_type: None,
            config: cfg,
            layers: vec![],
            annotations: None,
        };
        let desc = store.write_manifest(&manifest).unwrap();
        store.tag(desc, "hf.co/ai/healthy:latest").unwrap();

        // A corrupt pointer file elsewhere in manifests/ — enumeration
        // can't parse it, so the whole scan must fail rather than proceed
        // with an incomplete live set.
        let corrupt = root
            .join("manifests")
            .join("hf.co")
            .join("ai")
            .join("broken")
            .join("latest");
        std::fs::create_dir_all(corrupt.parent().unwrap()).unwrap();
        std::fs::write(&corrupt, b"not json").unwrap();

        assert!(
            referenced_digests(&store).is_err(),
            "an unparsable reference must abort the live-set scan, not be skipped"
        );

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn prune_is_a_no_op_on_missing_directories() {
        let root = temp_dir("missing");
        let live = HashSet::new();
        assert_eq!(prune_blobs(&root, &live, Duration::ZERO).unwrap().count, 0);
        assert_eq!(
            prune_cache(&root.join("cache"), &live, Duration::ZERO)
                .unwrap()
                .count,
            0
        );
    }

    /// Minimal mtime-backdating helper — mirrors the one in
    /// `storage::repair`'s tests, avoiding a `filetime` dependency just for
    /// this. Unix-only, like that one.
    #[cfg(unix)]
    fn filetime_set(path: &Path, when: SystemTime) {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::open(path).unwrap();
        let d = when.duration_since(SystemTime::UNIX_EPOCH).unwrap();
        let tv = libc::timeval {
            tv_sec: d.as_secs() as _,
            tv_usec: d.subsec_micros() as _,
        };
        let times = [tv, tv];
        unsafe {
            libc::futimes(file.as_raw_fd(), times.as_ptr());
        }
    }
}

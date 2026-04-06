//! Shared utility helpers.

use std::path::{Path, PathBuf};

use anyhow::Result;

/// Format a byte count as a human-readable string (GiB / MiB / KiB / B).
pub fn format_bytes(bytes: u64) -> String {
    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    const KIB: u64 = 1 << 10;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Resolve the hf-hub cache root: `$HF_HOME/hub` or `~/.cache/huggingface/hub`.
pub fn cache_root() -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        PathBuf::from(hf_home).join("hub")
    } else if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(xdg_cache).join("huggingface/hub")
    } else {
        home_dir().join(".cache/huggingface/hub")
    }
}

/// Portable home directory without pulling in the `dirs` crate.
///
/// Checks `HOME` (Unix) then `USERPROFILE` (Windows), falling back to `/`.
pub fn home_dir() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/"))
}

/// Recursively sum the size of all files under `path`.
///
/// Uses `symlink_metadata` so that symbolic links (e.g. the HuggingFace
/// `snapshots/` → `blobs/` indirection) are not followed and their targets
/// are not double-counted.
pub fn dir_size(path: &Path) -> Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.path().symlink_metadata()?;
        if metadata.is_dir() {
            total += dir_size(&entry.path()).unwrap_or(0);
        } else if metadata.is_file() {
            total += metadata.len();
        }
    }
    Ok(total)
}

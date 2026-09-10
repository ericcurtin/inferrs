//! Whole-file writes a reader never sees half done.

use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Distinguishes the temp files of concurrent writers in one process; the
/// pid does the same across processes.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Writes `bytes` to `path` through a temp file beside it and a rename,
/// so a reader sees the old file or the new one and not a partial one;
/// the temp name carries the pid and a counter, so no two writers share
/// one.
/// A symlink whose target exists is written through to the target, the
/// old file's permissions go on the temp before it holds anything, and
/// the temp is synced before the rename.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let path = dunce::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let mut tmp = path.clone().into_os_string();
    tmp.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let written = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        if let Ok(meta) = std::fs::metadata(&path) {
            file.set_permissions(meta.permissions())?;
        }
        file.write_all(bytes)?;
        file.sync_all()
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, &path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "llmman-fsutil-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn write_atomic_replaces_the_content_and_leaves_no_temp_behind() {
        let dir = temp_dir("replace");
        let path = dir.join("f.tar.gz");
        write_atomic(&path, b"one").unwrap();
        write_atomic(&path, b"two").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"two");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Writers in one process racing on one file each land whole.
    #[test]
    fn write_atomic_concurrent_writers_each_land_a_whole_file() {
        let dir = temp_dir("race");
        let path = dir.join("f.json");
        let threads: Vec<_> = (0..8u8)
            .map(|i| {
                let path = path.clone();
                std::thread::spawn(move || write_atomic(&path, &[i; 4096]).unwrap())
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let got = std::fs::read(&path).unwrap();
        assert_eq!(got.len(), 4096);
        assert!(got.iter().all(|b| *b == got[0]));
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A dotfile kept as a symlink stays one: the target changes, the
    /// link does not, and the old mode comes along.
    #[cfg(unix)]
    #[test]
    fn write_atomic_writes_through_a_symlink_and_keeps_the_mode() {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = temp_dir("symlink");
        let target = dir.join("real.json");
        let link = dir.join("link.json");
        std::fs::write(&target, b"old").unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::os::unix::fs::symlink(&target, &link).unwrap();
        write_atomic(&link, b"new").unwrap();
        assert!(std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(std::fs::read(&target).unwrap(), b"new");
        assert_eq!(
            std::fs::metadata(&target).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

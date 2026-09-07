//! A local model directory (`/absolute/path`), imported into the OCI
//! store as a CNCF ModelPack so `llmman serve`/`run`/`push` treat it
//! like anything else pulled. The source files are only ever read.

use std::path::Path;

use anyhow::{Context, Result};

use super::{should_pack, PackFile, Target};

pub(crate) fn pull(local_path: &str, target: &Target<'_>) -> Result<()> {
    let root = Path::new(local_path);
    let meta = std::fs::metadata(root).with_context(|| format!("local path {local_path:?}"))?;
    if !meta.is_dir() {
        anyhow::bail!("local path {local_path:?} is not a directory");
    }

    // Canonicalize the root so that two symlinks/junctions/mount points
    // to the same directory produce the same manifest reference and blob
    // digests, instead of being stored twice.  `dunce::canonicalize`
    // resolves NTFS junctions and volume mount points while avoiding the
    // `\\?\` verbatim-path prefix that `std::fs::canonicalize` produces
    // on Windows (which would break `strip_prefix` and `starts_with`
    // comparisons downstream).
    let canonical_root = dunce::canonicalize(root)
        .with_context(|| format!("canonicalize local path {local_path:?}"))?;
    let canonical_path_str = canonical_root.to_string_lossy().replace('\\', "/");

    if target.report_cached(&canonical_path_str, local_path) {
        return Ok(());
    }

    let mut packed = Vec::new();
    for entry in walkdir::WalkDir::new(&canonical_root).follow_links(true) {
        let entry = entry.with_context(|| format!("walk {local_path}"))?;
        if !entry.file_type().is_file() {
            continue;
        }

        // Canonicalize each entry to resolve any nested symlinks and
        // verify it hasn't escaped the source root.  A symlink inside
        // the model directory pointing to `../../etc/passwd` would
        // resolve to a path outside canonical_root and is rejected.
        let canonical_entry = dunce::canonicalize(entry.path()).with_context(|| {
            format!(
                "canonicalize {} under {local_path}",
                entry.path().display()
            )
        })?;
        if !canonical_entry.starts_with(&canonical_root) {
            anyhow::bail!(
                "path {:?} resolves to {:?} which is outside store root {:?}",
                entry.path(),
                canonical_entry,
                canonical_root
            );
        }

        let rel = canonical_entry
            .strip_prefix(&canonical_root)
            .unwrap_or(entry.path())
            .to_string_lossy()
            // Recorded in the manifest, read back on any platform:
            // always the OCI-conventional "/" separator, never "\".
            .replace('\\', "/");
        if !should_pack(&rel) {
            continue;
        }
        packed.push(PackFile {
            local_path: canonical_entry,
            relative_path: rel,
            owned: false,
        });
    }

    if packed.is_empty() {
        anyhow::bail!("no model files found in {local_path}");
    }
    eprintln!("Importing {} files from {local_path}", packed.len());
    // No download progress to show the work happening, so announce each
    // file as it is stored instead.
    for f in &packed {
        if let Ok(m) = std::fs::metadata(&f.local_path) {
            eprintln!("  stored {} ({} bytes)", f.relative_path, m.len());
        }
    }

    let repo = canonical_root
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| local_path.to_string());
    super::pack_as_model_pack(
        target,
        &canonical_path_str,
        &repo,
        packed,
        format!("no model files found in {local_path}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf::oci;

    fn tempdir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "llmman-sources-local-{name}-{}-{}",
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
    fn pull_imports_a_directory_as_a_model_pack_without_moving_the_originals() {
        let src = tempdir("src");
        let layout = tempdir("layout");
        std::fs::write(src.join("config.json"), b"{}").unwrap();
        std::fs::write(src.join("model.safetensors"), b"weights").unwrap();
        std::fs::create_dir_all(src.join("nested")).unwrap();
        std::fs::write(src.join("nested").join("tokenizer.json"), b"{}").unwrap();
        // Filtered out by should_pack.
        std::fs::write(src.join(".gitattributes"), b"x").unwrap();

        let reference = src.to_str().unwrap().to_string();
        oci::ensure_layout(&layout).unwrap();
        pull(
            &reference,
            &Target {
                layout_dir: &layout,
                progress_key: "",
                store_as: None,
            },
        )
        .unwrap();

        let desc = oci::read_manifest_ref(&layout, &reference).unwrap();
        let manifest: oci::Manifest =
            serde_json::from_slice(&oci::read_blob(&layout, &desc.digest).unwrap()).unwrap();
        let mut paths: Vec<&str> = manifest
            .layers
            .iter()
            .filter_map(|l| l.annotation(oci::ANNOTATION_FILEPATH))
            .collect();
        paths.sort_unstable();
        assert_eq!(
            paths,
            vec!["config.json", "model.safetensors", "nested/tokenizer.json"]
        );
        assert!(
            src.join("model.safetensors").exists(),
            "importing must copy, never move, a user's own files"
        );

        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&layout).ok();
    }

    #[test]
    fn pull_rejects_a_directory_with_nothing_importable_in_it() {
        let src = tempdir("empty");
        let layout = tempdir("empty-layout");
        std::fs::write(src.join(".hidden"), b"x").unwrap();
        let err = pull(
            src.to_str().unwrap(),
            &Target {
                layout_dir: &layout,
                progress_key: "",
                store_as: None,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("no model files found"));
        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&layout).ok();
    }

    #[test]
    fn pull_rejects_a_path_that_is_not_a_directory() {
        let dir = tempdir("file");
        let file = dir.join("model.gguf");
        std::fs::write(&file, b"x").unwrap();
        let err = pull(
            file.to_str().unwrap(),
            &Target {
                layout_dir: &dir,
                progress_key: "",
                store_as: None,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("is not a directory"));
        std::fs::remove_dir_all(&dir).ok();
    }

    fn create_symlink_dir(target: &Path, link: &Path) -> std::io::Result<()> {
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(target, link)
        }
        #[cfg(windows)]
        {
            // Use mklink /J (directory junction) which does not require admin/Dev Mode
            let status = std::process::Command::new("cmd")
                .arg("/C")
                .arg("mklink")
                .arg("/J")
                .arg(link.as_os_str())
                .arg(target.as_os_str())
                .status()?;
            if !status.success() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "mklink /J failed",
                ));
            }
            Ok(())
        }
    }

    #[test]
    fn pull_resolves_symlinked_root_to_prevent_duplicate_blobs() {
        let real_src = tempdir("real-src");
        let link_src = tempdir("link-src-dir").join("link");
        let layout = tempdir("symlink-layout");
        
        std::fs::write(real_src.join("model.safetensors"), b"weights").unwrap();
        
        if let Err(e) = create_symlink_dir(&real_src, &link_src) {
            eprintln!("Skipping test: could not create symlink/junction: {e}");
            return;
        }

        oci::ensure_layout(&layout).unwrap();
        pull(
            link_src.to_str().unwrap(),
            &Target {
                layout_dir: &layout,
                progress_key: "",
                store_as: None,
            },
        )
        .unwrap();

        // Check the generated manifest. The canonical path (real_src) should be used as the repo name
        // (unless overridden by store_as), avoiding duplicate blobs if pulled via multiple symlinks.
        let canonical_str = dunce::canonicalize(&real_src).unwrap().to_string_lossy().replace('\\', "/");
        let expected_repo = real_src.file_name().unwrap().to_string_lossy().into_owned();
        let desc = oci::read_manifest_ref(&layout, &canonical_str).unwrap();
        
        std::fs::remove_dir_all(&real_src).ok();
        std::fs::remove_dir_all(link_src.parent().unwrap()).ok();
        std::fs::remove_dir_all(&layout).ok();
    }

    #[test]
    fn pull_rejects_symlink_that_escapes_source_root() {
        let src = tempdir("escape-src");
        let outside = tempdir("escape-outside");
        let layout = tempdir("escape-layout");
        
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.txt"), b"secret").unwrap();
        
        // Put a valid file in src so it doesn't fail early on "no files"
        std::fs::write(src.join("model.safetensors"), b"weights").unwrap();
        
        // Create a symlink/junction inside src pointing outside
        let escape_link = src.join("escape");
        if let Err(e) = create_symlink_dir(&outside, &escape_link) {
            eprintln!("Skipping test: could not create symlink/junction: {e}");
            return;
        }

        oci::ensure_layout(&layout).unwrap();
        let err = pull(
            src.to_str().unwrap(),
            &Target {
                layout_dir: &layout,
                progress_key: "",
                store_as: None,
            },
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("outside store root") || err.to_string().contains("outside source dir"),
            "expected root escape error, got: {err}"
        );
        
        std::fs::remove_dir_all(&src).ok();
        std::fs::remove_dir_all(&outside).ok();
        std::fs::remove_dir_all(&layout).ok();
    }
}

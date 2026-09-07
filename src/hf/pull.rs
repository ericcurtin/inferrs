//! Top-level HuggingFace pull — originally ported from the Go shim's
//! `pullHF`/`pullHFSafetensors`/`downloadHFBlob` (since deleted), and
//! writing straight into the local OCI layout without ever calling into
//! the Go shim.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use futures::stream::{self, StreamExt, TryStreamExt};
use sha2::{Digest as _, Sha256};

use super::api::{self, HfFile};
use super::download::{self, FetchRequest};
use super::oci::{self, Descriptor, ModelMeta};
use super::progress;
use crate::xet_fetch::{self, XetFileRef};

/// Pulls `reference` (already stripped of any `hf://`/`huggingface://`
/// scheme prefix) into `layout_dir`. `progress_key` is the exact,
/// original ref the daemon's `/api/pull` handler was given, for
/// `progress::poll` — see that module's own doc comment.
pub async fn pull(reference: &str, layout_dir: &Path, progress_key: &str) -> Result<()> {
    let _guard = progress::DoneGuard(progress_key);
    let (host, owner, repo, tag) = api::parse_hf_ref(reference)?;

    oci::ensure_layout(layout_dir)?;
    if oci::report_cached(layout_dir, reference) {
        return Ok(());
    }

    let endpoint = super::hf_endpoint(&host);
    let token = super::token();
    let api_client = super::api_client()?;
    let dl_client = super::download_client()?;
    let head_client = super::head_client()?;

    progress::set_status(progress_key, "pulling");
    let info = api::fetch_model_info(&api_client, &endpoint, &owner, &repo, token.as_deref())
        .await
        .context("HF model info")?;
    let commit = info.commit().to_string();
    let mut meta = ModelMeta {
        licenses: info.license().into_iter().collect(),
        ..Default::default()
    };

    let files = api::fetch_files(
        &api_client,
        &endpoint,
        &owner,
        &repo,
        &commit,
        token.as_deref(),
    )
    .await
    .context("HF file list")?;

    let diffusion = api::is_diffusion_repo(&files);
    let selected = if diffusion {
        api::select_diffusion_gguf(&files, &tag)
    } else {
        api::select_gguf(&files, &tag)
    };
    let manifest_desc = match selected {
        Ok(shards) if diffusion && shards.len() != 1 => {
            anyhow::bail!("{owner}/{repo}: split diffusion transformers are not supported");
        }
        Ok(shards) => {
            meta.format = "gguf".to_string();
            let filepath_annotation = if shards.len() == 1 {
                basename(&shards[0].path)
            } else {
                String::new()
            };

            let mut layers = Vec::new();
            for f in &shards {
                layers.push(
                    download_layer(
                        &dl_client,
                        &head_client,
                        &endpoint,
                        &owner,
                        &repo,
                        &commit,
                        f,
                        token.as_deref(),
                        layout_dir,
                        progress_key,
                    )
                    .await?,
                );
            }
            if diffusion {
                let plan = api::diffusion_plan(&files, &shards[0].path);
                meta.diffusion_outputs = plan.outputs;
                for (role, file) in &plan.sidecars {
                    let d = download_layer(
                        &dl_client,
                        &head_client,
                        &endpoint,
                        &owner,
                        &repo,
                        &commit,
                        file,
                        token.as_deref(),
                        layout_dir,
                        progress_key,
                    )
                    .await?;
                    layers.push(api::sidecar_layer(d, file, role));
                }
                if let Some(te_ref) = plan.text_encoder {
                    let mut d = pull_text_encoder(
                        &api_client,
                        &dl_client,
                        &head_client,
                        &host,
                        te_ref,
                        token.as_deref(),
                        layout_dir,
                        progress_key,
                    )
                    .await?;
                    d.annotations
                        .get_or_insert_with(BTreeMap::new)
                        .insert(oci::ANNOTATION_ROLE.to_string(), "text_encoder".to_string());
                    layers.push(d);
                }
            } else if let Some(mmproj) = api::select_mmproj(&files) {
                layers.push(
                    download_layer(
                        &dl_client,
                        &head_client,
                        &endpoint,
                        &owner,
                        &repo,
                        &commit,
                        &mmproj,
                        token.as_deref(),
                        layout_dir,
                        progress_key,
                    )
                    .await?,
                );
                meta.vision = true;
            }
            if let Some(lic) = api::select_license_file(&files) {
                let mut d = download_layer(
                    &dl_client,
                    &head_client,
                    &endpoint,
                    &owner,
                    &repo,
                    &commit,
                    &lic,
                    token.as_deref(),
                    layout_dir,
                    progress_key,
                )
                .await?;
                d.media_type = oci::MEDIA_TYPE_MODEL_DOC_RAW.to_string();
                layers.push(d);
            }
            oci::build_cncf_manifest(
                layout_dir,
                &meta,
                &format!("{owner}/{repo}"),
                &filepath_annotation,
                layers,
            )?
        }
        // Only fall back to safetensors when the repo has no GGUF files
        // at all — a user-requested tag/quant that doesn't exist should
        // be a hard error, not a silent format switch.
        Err(e) if !tag.is_empty() && tag != "latest" => return Err(e),
        Err(_) => {
            meta.format = "safetensors".to_string();
            let to_download = api::select_downloadable_hf_files(&files);
            if to_download.is_empty() {
                anyhow::bail!("no model files found in repository {owner}/{repo}");
            }
            // Concurrent, bounded by LLMMAN_MAX_TRANSFER_STREAMS (the
            // GGUF branch above stays sequential). Indexed so `layers`'
            // order doesn't depend on finish order.
            let max_streams = super::max_transfer_streams();
            let mut indexed_layers: Vec<(usize, Descriptor)> =
                stream::iter(to_download.iter().enumerate())
                    .map(|(i, f)| {
                        let dl_client = &dl_client;
                        let head_client = &head_client;
                        let endpoint = &endpoint;
                        let owner = &owner;
                        let repo = &repo;
                        let commit = &commit;
                        let token = token.as_deref();
                        async move {
                            let mut d = download_layer(
                                dl_client,
                                head_client,
                                endpoint,
                                owner,
                                repo,
                                commit,
                                f,
                                token,
                                layout_dir,
                                progress_key,
                            )
                            .await?;
                            d.media_type = api::safetensors_media_type(&f.path).to_string();
                            d.annotations = Some(BTreeMap::from([(
                                oci::ANNOTATION_FILEPATH.to_string(),
                                f.path.clone(),
                            )]));
                            Ok::<(usize, Descriptor), anyhow::Error>((i, d))
                        }
                    })
                    .buffer_unordered(max_streams)
                    .try_collect()
                    .await?;
            indexed_layers.sort_by_key(|(i, _)| *i);
            let layers: Vec<Descriptor> = indexed_layers.into_iter().map(|(_, d)| d).collect();
            oci::build_cncf_manifest(layout_dir, &meta, &format!("{owner}/{repo}"), "", layers)?
        }
    };

    oci::write_manifest_ref(layout_dir, reference, manifest_desc)
}

fn basename(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

/// Downloads the text encoder GGUF of a diffusion model from its own
/// repository (`owner/repo[:quant]`) into the same blob store, returning
/// its layer descriptor. Blobs are content-addressed, so a text encoder
/// shared by several diffusion models is stored once.
#[allow(clippy::too_many_arguments)]
async fn pull_text_encoder(
    api_client: &reqwest::Client,
    dl_client: &reqwest::Client,
    head_client: &reqwest::Client,
    host: &str,
    reference: &str,
    token: Option<&str>,
    layout_dir: &Path,
    progress_key: &str,
) -> Result<Descriptor> {
    let (_, owner, repo, tag) = api::parse_hf_ref(&format!("{host}/{reference}"))?;
    let endpoint = super::hf_endpoint(host);
    let info = api::fetch_model_info(api_client, &endpoint, &owner, &repo, token)
        .await
        .with_context(|| format!("HF model info for text encoder {reference}"))?;
    let commit = info.commit().to_string();
    let files = api::fetch_files(api_client, &endpoint, &owner, &repo, &commit, token)
        .await
        .with_context(|| format!("HF file list for text encoder {reference}"))?;
    let shards = api::select_gguf(&files, &tag)?;
    if shards.len() != 1 {
        anyhow::bail!("text encoder {reference}: split GGUFs are not supported");
    }
    let mut d = download_layer(
        dl_client,
        head_client,
        &endpoint,
        &owner,
        &repo,
        &commit,
        &shards[0],
        token,
        layout_dir,
        progress_key,
    )
    .await?;
    d.annotations = Some(BTreeMap::from([(
        oci::ANNOTATION_FILEPATH.to_string(),
        basename(&shards[0].path),
    )]));
    Ok(d)
}

/// Deletes its `.part` path on drop unless [`disarm`](Self::disarm) was
/// called first — so a concurrent download cancelled mid-flight (a
/// sibling in the same `buffer_unordered` batch failing first) still
/// cleans up its temp file, not just a normal error return.
struct TmpFileGuard<'a>(&'a Path, bool);

impl TmpFileGuard<'_> {
    fn disarm(&mut self) {
        self.1 = false;
    }
}

impl Drop for TmpFileGuard<'_> {
    fn drop(&mut self) {
        if self.1 {
            let _ = std::fs::remove_file(self.0);
        }
    }
}

/// Moves `tmp_path` to `dest` in the content-addressed blob store.
/// Returns `false` (not moved) if `dest` already had this content — a
/// concurrent sibling download of identical bytes can win the rename
/// first now that safetensors files download in parallel; POSIX
/// overwrites silently, but Windows can error, so `dest` is rechecked
/// rather than failing the whole pull over that race.
fn move_into_place(tmp_path: &Path, dest: &Path) -> Result<bool> {
    if dest.exists() {
        return Ok(false);
    }
    match std::fs::rename(tmp_path, dest) {
        Ok(()) => Ok(true),
        Err(_) if dest.exists() => Ok(false),
        Err(e) => Err(e).with_context(|| format!("move downloaded blob to {}", dest.display())),
    }
}

/// Downloads one file straight into `layout_dir`'s content-addressed
/// blob store, returning its descriptor — mirrors `downloadHFBlob`.
/// Unlike the Go version, a retry always restarts this file from byte
/// zero rather than resuming a local `.part` file; see this migration's
/// own PR description for why that trade-off was accepted.
#[allow(clippy::too_many_arguments)]
async fn download_layer(
    client: &reqwest::Client,
    head_client: &reqwest::Client,
    endpoint: &str,
    owner: &str,
    repo: &str,
    commit: &str,
    file: &HfFile,
    token: Option<&str>,
    layout_dir: &Path,
    progress_key: &str,
) -> Result<Descriptor> {
    let url = format!("{endpoint}{owner}/{repo}/resolve/{commit}/{}", file.path);
    let label = format!("Pulling {}", basename(&file.path));

    // Single attempt: this is a cheap metadata probe, so a bad or
    // nonexistent host must fail fast rather than run the retry backoff.
    // Failure is tolerated anyway (falls back to a plain, un-Xet'd GET
    // self-hashed after download).
    let meta = super::client::once(&format!("HEAD {}", file.path), || {
        download::head_metadata(head_client, url.clone(), token)
    })
    .await
    .unwrap_or_default();
    // One size for both the progress total and the Xet download, so the
    // bytes download_to_path credits can never exceed what was added.
    let size = if meta.size > 0 { meta.size } else { file.size };
    let xet = meta.xet_hash.map(|hash| XetFileRef {
        endpoint: endpoint.to_string(),
        repo_type: "models".to_string(),
        owner_repo: format!("{owner}/{repo}"),
        revision: commit.to_string(),
        hash,
        size: size as u64,
        sha256: meta.digest.clone(),
        hf_token: token.map(str::to_string),
    });
    progress::add_total(progress_key, size);

    // The blob store is content-addressed and HF's X-Linked-Etag is the
    // file's sha256, so a blob already present (the same text encoder
    // shared by two diffusion models, a re-pull after a manifest change)
    // needs no download at all.
    if let Some(want) = &meta.digest {
        let dest = layout_dir.join("blobs").join("sha256").join(want);
        if std::fs::metadata(&dest)
            .map(|m| m.is_file() && m.len() as i64 == size)
            .unwrap_or(false)
        {
            progress::add_completed(progress_key, size);
            return Ok(Descriptor {
                media_type: oci::MEDIA_TYPE_MODEL_WEIGHT_RAW.to_string(),
                digest: format!("sha256:{want}"),
                size,
                annotations: Some(BTreeMap::from([(
                    oci::ANNOTATION_FILEPATH.to_string(),
                    basename(&file.path),
                )])),
                ..Default::default()
            });
        }
    }

    let tmp_dir = layout_dir.join("blobs").join("tmp");
    std::fs::create_dir_all(&tmp_dir)?;
    // pid + a per-call counter, not just pid: two concurrent pulls of
    // different repos that happen to share a filename (e.g. both have a
    // "model.safetensors") would otherwise share this same path.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp_path = tmp_dir.join(format!(
        "hf-{}-{unique}-{}.part",
        std::process::id(),
        sanitize(&file.path)
    ));
    let mut tmp_guard = TmpFileGuard(&tmp_path, true);

    let req = FetchRequest {
        url,
        token,
        xet,
        label: label.clone(),
    };
    let mut last_err: Option<anyhow::Error> = None;
    let mut next_delay = None;
    let digest = 'attempts: {
        for attempt in 0..super::client::MAX_ATTEMPTS {
            if let Some(delay) = next_delay.take() {
                eprintln!(
                    "\n[llmman] retrying {label} (attempt {}/{}, wait {delay:?})",
                    attempt + 1,
                    super::client::MAX_ATTEMPTS
                );
                tokio::time::sleep(delay).await;
            }
            // A fresh file each attempt, not a reopen: File::create
            // truncates, so a partial write from an earlier failed
            // attempt can never linger underneath this one's bytes.
            // (Progress reporting isn't undone on a retry — a retried
            // file's bar may briefly look ahead of itself; a cosmetic
            // wrinkle only, not worth this migration's added complexity.)
            let got = match &req.xet {
                // Xet: hf-xet owns the file so progress tracks the wire
                // (see xet_fetch::download_to_path); hash it afterwards.
                Some(xet) => {
                    match xet_fetch::download_to_path(xet, &tmp_path, |n| {
                        progress::add_completed(progress_key, n as i64)
                    })
                    .await
                    .with_context(|| label.clone())
                    {
                        Ok(()) => sha256_of_file(&tmp_path).await,
                        Err(e) => Err(e),
                    }
                }
                None => {
                    let f = std::fs::File::create(&tmp_path)?;
                    let mut writer = HashingProgressWriter {
                        file: f,
                        hasher: Sha256::new(),
                        progress_key,
                    };
                    download::fetch_once(client, &req, &mut writer)
                        .await
                        .and_then(|_| {
                            writer.file.flush()?;
                            Ok(format!("sha256:{:x}", writer.hasher.finalize()))
                        })
                }
            };
            // meta.digest (from X-Linked-Etag) is a trusted claim about
            // the content ahead of downloading it — worth checking the
            // bytes we actually got still match, in case of transient
            // corruption, same as a failed fetch.
            let result = got.and_then(|got| match &meta.digest {
                Some(want) if format!("sha256:{want}") != got => {
                    anyhow::bail!("digest mismatch: expected sha256:{want}, got {got}")
                }
                _ => Ok(got),
            });
            match result {
                Ok(digest) => break 'attempts digest,
                Err(e) => {
                    eprintln!("[llmman] {label} error: {e:#}");
                    match download::should_retry(&e, attempt + 1) {
                        Some(delay) => next_delay = Some(delay),
                        None => {
                            last_err = Some(e);
                            break;
                        }
                    }
                    last_err = Some(e);
                }
            }
        }
        // tmp_guard's drop removes tmp_path.
        return Err(last_err.unwrap()).with_context(|| {
            format!(
                "download {} failed after {} attempts",
                file.path,
                super::client::MAX_ATTEMPTS
            )
        });
    };

    let dest_dir = layout_dir.join("blobs").join("sha256");
    std::fs::create_dir_all(&dest_dir)?;
    let dest = dest_dir.join(digest.trim_start_matches("sha256:"));
    let total = std::fs::metadata(&tmp_path)?.len() as i64;
    if move_into_place(&tmp_path, &dest)? {
        tmp_guard.disarm(); // already moved away; nothing left to clean up
    }
    // else: same content already at `dest` — tmp_guard's drop removes
    // this now-redundant copy.

    Ok(Descriptor {
        media_type: oci::MEDIA_TYPE_MODEL_WEIGHT_RAW.to_string(),
        digest,
        size: total,
        annotations: Some(BTreeMap::from([(
            oci::ANNOTATION_FILEPATH.to_string(),
            basename(&file.path),
        )])),
        ..Default::default()
    })
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// `sha256:<hex>` of `path`'s content, for the one download path
/// (`xet_fetch::download_to_path`) that bypasses [`HashingProgressWriter`].
/// Off the async runtime: a full read of a possibly hundreds-of-GB file.
async fn sha256_of_file(path: &Path) -> Result<String> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || -> Result<String> {
        use std::io::Read;
        let mut f = std::fs::File::open(&path)
            .with_context(|| format!("open {} for hashing", path.display()))?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            match f.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => hasher.update(&buf[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(format!("sha256:{:x}", hasher.finalize()))
    })
    .await
    .context("sha256 hashing task panicked")?
}

/// Writes to a local file while hashing every byte and reporting it to
/// `progress`.
struct HashingProgressWriter<'a> {
    file: std::fs::File,
    hasher: Sha256,
    progress_key: &'a str,
}

impl Write for HashingProgressWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.file.write(buf)?;
        self.hasher.update(&buf[..n]);
        progress::add_completed(self.progress_key, n as i64);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tmp_file_guard_removes_the_file_on_drop_by_default() {
        let dir = std::env::temp_dir().join(format!("llmman-tmpguard-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("armed.part");
        std::fs::write(&path, b"partial").unwrap();

        {
            let _guard = TmpFileGuard(&path, true);
        }
        assert!(
            !path.exists(),
            "an armed guard must remove its file on drop"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tmp_file_guard_leaves_the_file_alone_once_disarmed() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-tmpguard-test-disarm-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("disarmed.part");
        std::fs::write(&path, b"done").unwrap();

        {
            let mut guard = TmpFileGuard(&path, true);
            guard.disarm();
        }
        assert!(path.exists(), "a disarmed guard must not remove its file");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn move_into_place_moves_the_file_when_dest_is_absent() {
        let dir = std::env::temp_dir().join(format!("llmman-move-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tmp = dir.join("a.part");
        let dest = dir.join("dest");
        std::fs::write(&tmp, b"content").unwrap();

        assert!(move_into_place(&tmp, &dest).unwrap());
        assert!(!tmp.exists());
        assert!(dest.exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regression test: a concurrent sibling download of identical
    /// content winning the rename first (`dest` already exists) must
    /// not be treated as an error — the caller's own `.part` file is
    /// just redundant, not a failure.
    #[test]
    fn move_into_place_tolerates_a_destination_that_already_exists() {
        let dir =
            std::env::temp_dir().join(format!("llmman-move-test-existing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tmp = dir.join("a.part");
        let dest = dir.join("dest");
        std::fs::write(&tmp, b"content").unwrap();
        std::fs::write(&dest, b"content").unwrap(); // a sibling already won

        assert!(!move_into_place(&tmp, &dest).unwrap());
        assert!(
            tmp.exists(),
            "move_into_place itself must leave the redundant tmp file for the caller's guard to clean up"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}

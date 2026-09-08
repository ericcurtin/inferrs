//! Top-level HuggingFace transfer.
//!
//! **`docker` build**: fetches each file (plain HTTP or `hf-xet`, as
//! [`super::pull`] does) and streams it straight into a registry push,
//! never touching local disk. The registry-push protocol itself still
//! runs in Go (only containerd speaks it) — see
//! [`crate::ffi::push_stream_open`]: Go creates the pipe and hands over
//! its write end's raw fd/HANDLE, so only that one integer crosses the
//! FFI boundary per file, not a callback per chunk.
//!
//! **`podman` build**: `copy.Image` works on whole images, not
//! individual blobs, so there's no per-blob streaming primitive to use.
//! Instead this pulls into a throwaway local OCI layout (via
//! [`super::pull`]) and pushes that with the existing `ffi::push`.

use anyhow::{Context, Result};

use crate::ffi::TransferOutcome;

/// Transfers `reference` (already stripped of any `hf://`/
/// `huggingface://` scheme prefix) directly to `destination`. Reports
/// what was pushed and the manifest digest that now sits at the
/// destination — mirrors `ffi::transfer`'s own contract.
pub async fn transfer(reference: &str, destination: &str) -> Result<TransferOutcome> {
    #[cfg(feature = "docker")]
    {
        docker::transfer(reference, destination).await
    }
    #[cfg(not(feature = "docker"))]
    {
        via_temp_pull(reference, destination).await
    }
}

/// The `podman`-build fallback: pull the whole model into a throwaway
/// local layout, then push that layout the ordinary way. `changed` is
/// always true on success — `ffi::push` (podman's `copy.Image`) doesn't
/// report whether the destination actually changed, unlike the docker
/// path's real per-blob answer.
#[cfg_attr(feature = "docker", allow(dead_code))]
async fn via_temp_pull(reference: &str, destination: &str) -> Result<TransferOutcome> {
    let tmp = std::env::temp_dir().join(format!(
        "llmman-hf-transfer-{}-{}",
        std::process::id(),
        rand_suffix()
    ));
    // pull() stages the manifest under `reference` itself, but
    // `ffi::push` resolves what to push by an *exact* ref lookup — so
    // the staged model has to also be findable under `destination`
    // before the push can find it at all.
    let result = super::pull::pull(reference, &tmp, "")
        .await
        .and_then(|()| super::oci::alias_manifest_ref(&tmp, reference, destination))
        .and_then(|()| {
            crate::ffi::push(
                tmp.to_str()
                    .context("temp layout path is not valid UTF-8")?,
                destination,
            )
        })
        // Read back from the staged layout rather than re-resolving the
        // destination tag: this is the manifest that was just pushed, so
        // it is what `--sign-key` must sign.
        .and_then(|()| super::oci::read_manifest_ref(&tmp, destination))
        .map(|desc| TransferOutcome::new(true, desc.digest));
    let _ = std::fs::remove_dir_all(&tmp);
    result
}

fn rand_suffix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(feature = "docker")]
mod docker {
    use std::io::Write;

    use anyhow::{Context, Result};
    use sha2::Digest as _;

    use super::super::api::{self, HfFile};
    use super::super::download::{self, FetchRequest};
    use super::super::oci::{self, Descriptor, ModelMeta};
    use super::super::progress;
    use crate::xet_fetch::XetFileRef;

    pub async fn transfer(reference: &str, destination: &str) -> Result<super::TransferOutcome> {
        let _guard = progress::DoneGuard(destination);
        let (host, owner, repo, tag) = api::parse_hf_ref(reference)?;
        let endpoint = super::super::hf_endpoint(&host);
        let token = super::super::token();
        let api_client = super::super::api_client()?;
        let dl_client = super::super::download_client()?;
        let head_client = super::super::head_client()?;

        // One resolver/pusher for every blob (and the manifest) in this
        // transfer — see push_stream.go's own doc comment for why
        // resolving a destination is too expensive to redo per blob.
        let session = crate::ffi::push_session_open(destination)?;

        progress::set_status(destination, "pushing");
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

        let mut changed = false;
        let diffusion = api::is_diffusion_repo(&files);
        let selected = if diffusion {
            api::select_diffusion_gguf(&files, &tag)
        } else {
            api::select_gguf(&files, &tag)
        };
        let (layers, filepath_annotation) = match selected {
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
                        transfer_layer(
                            &dl_client,
                            &head_client,
                            &endpoint,
                            &owner,
                            &repo,
                            &commit,
                            f,
                            token.as_deref(),
                            &session,
                            &mut changed,
                        )
                        .await?,
                    );
                }
                if diffusion {
                    let plan = api::diffusion_plan(&files, &shards[0].path);
                    meta.diffusion_outputs = plan.outputs;
                    for (role, file) in &plan.sidecars {
                        let d = transfer_layer(
                            &dl_client,
                            &head_client,
                            &endpoint,
                            &owner,
                            &repo,
                            &commit,
                            file,
                            token.as_deref(),
                            &session,
                            &mut changed,
                        )
                        .await?;
                        layers.push(api::sidecar_layer(d, file, role));
                    }
                    if let Some(te_ref) = plan.text_encoder {
                        let d = transfer_text_encoder(
                            &api_client,
                            &dl_client,
                            &head_client,
                            &host,
                            te_ref,
                            token.as_deref(),
                            &session,
                            &mut changed,
                        )
                        .await?;
                        layers.push(d);
                    }
                } else if let Some(mmproj) = api::select_mmproj(&files) {
                    layers.push(
                        transfer_layer(
                            &dl_client,
                            &head_client,
                            &endpoint,
                            &owner,
                            &repo,
                            &commit,
                            &mmproj,
                            token.as_deref(),
                            &session,
                            &mut changed,
                        )
                        .await?,
                    );
                    meta.vision = true;
                }
                if let Some(lic) = api::select_license_file(&files) {
                    let mut d = transfer_layer(
                        &dl_client,
                        &head_client,
                        &endpoint,
                        &owner,
                        &repo,
                        &commit,
                        &lic,
                        token.as_deref(),
                        &session,
                        &mut changed,
                    )
                    .await?;
                    d.media_type = oci::MEDIA_TYPE_MODEL_DOC_RAW.to_string();
                    layers.push(d);
                }
                (layers, filepath_annotation)
            }
            // Only fall back to safetensors when the repo has no GGUF
            // files at all — a user-requested tag/quant that doesn't
            // exist should be a hard error, not a silent format switch.
            Err(e) if !tag.is_empty() && tag != "latest" => return Err(e),
            Err(_) => {
                meta.format = "safetensors".to_string();
                let to_download = api::select_downloadable_hf_files(&files);
                if to_download.is_empty() {
                    anyhow::bail!("no model files found in repository {owner}/{repo}");
                }
                let mut layers = Vec::new();
                for f in &to_download {
                    let mut d = transfer_layer(
                        &dl_client,
                        &head_client,
                        &endpoint,
                        &owner,
                        &repo,
                        &commit,
                        f,
                        token.as_deref(),
                        &session,
                        &mut changed,
                    )
                    .await?;
                    d.media_type = api::safetensors_media_type(&f.path).to_string();
                    d.annotations = Some(std::collections::BTreeMap::from([(
                        oci::ANNOTATION_FILEPATH.to_string(),
                        f.path.clone(),
                    )]));
                    layers.push(d);
                }
                // See the same step in pull.rs.
                if api::is_diffusers_repo(&files) {
                    let class = api::fetch_diffusers_pipeline_class(
                        &api_client,
                        &endpoint,
                        &owner,
                        &repo,
                        &commit,
                        token.as_deref(),
                    )
                    .await;
                    meta.diffusion_outputs = api::diffusers_outputs(class.as_deref());
                }
                (layers, String::new())
            }
        };

        let manifest_digest = push_cncf_manifest(
            &session,
            &meta,
            &format!("{owner}/{repo}"),
            &filepath_annotation,
            layers,
            &mut changed,
        )
        .await?;
        Ok(super::TransferOutcome::new(changed, manifest_digest))
    }

    fn basename(path: &str) -> String {
        path.rsplit('/').next().unwrap_or(path).to_string()
    }

    /// The text encoder GGUF from its own repository, as a `text_encoder` layer.
    #[allow(clippy::too_many_arguments)]
    async fn transfer_text_encoder(
        api_client: &reqwest::Client,
        dl_client: &reqwest::Client,
        head_client: &reqwest::Client,
        host: &str,
        reference: &str,
        token: Option<&str>,
        session: &crate::ffi::PushSession,
        changed: &mut bool,
    ) -> Result<Descriptor> {
        let (_, owner, repo, tag) = api::parse_hf_ref(&format!("{host}/{reference}"))?;
        let endpoint = super::super::hf_endpoint(host);
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
        let mut d = transfer_layer(
            dl_client,
            head_client,
            &endpoint,
            &owner,
            &repo,
            &commit,
            &shards[0],
            token,
            session,
            changed,
        )
        .await?;
        d.annotations = Some(std::collections::BTreeMap::from([
            (
                oci::ANNOTATION_FILEPATH.to_string(),
                basename(&shards[0].path),
            ),
            (oci::ANNOTATION_ROLE.to_string(), "text_encoder".to_string()),
        ]));
        Ok(d)
    }

    /// Fetches one file and streams it directly into a registry push,
    /// setting `*changed` if anything was actually pushed for it.
    #[allow(clippy::too_many_arguments)]
    async fn transfer_layer(
        client: &reqwest::Client,
        head_client: &reqwest::Client,
        endpoint: &str,
        owner: &str,
        repo: &str,
        commit: &str,
        file: &HfFile,
        token: Option<&str>,
        session: &crate::ffi::PushSession,
        changed: &mut bool,
    ) -> Result<Descriptor> {
        let url = format!("{endpoint}{owner}/{repo}/resolve/{commit}/{}", file.path);
        let label = format!("Transferring {}", basename(&file.path));

        let meta = super::super::client::probe(&format!("HEAD {}", file.path), || {
            download::head_metadata(head_client, url.clone(), token)
        })
        .await
        .context("resolve file metadata")?;
        let size = if meta.size > 0 { meta.size } else { file.size };
        let annotations = Some(std::collections::BTreeMap::from([(
            oci::ANNOTATION_FILEPATH.to_string(),
            basename(&file.path),
        )]));

        let (desc, pushed) = match meta.digest {
            // Known digest ahead of time (LFS/Xet-tracked — effectively
            // every real weight file): stream straight through with no
            // buffering, however large.
            Some(digest) => {
                // A known digest with an unknown (0) size would push a
                // Descriptor claiming 0 bytes while the stream writes
                // the real body — the registry only notices the
                // mismatch after the whole upload finishes.
                if size <= 0 {
                    anyhow::bail!("{}: digest known but size unknown", file.path);
                }
                let xet = meta.xet_hash.map(|hash| XetFileRef {
                    endpoint: endpoint.trim_end_matches('/').to_string(),
                    repo_type: "models".to_string(),
                    owner_repo: format!("{owner}/{repo}"),
                    revision: commit.to_string(),
                    hash,
                    size: size as u64,
                    sha256: Some(digest.clone()),
                    hf_token: token.map(str::to_string),
                });
                // FileMetadata::digest is bare hex — an OCI Descriptor's
                // own digest field must carry the "sha256:" algorithm
                // prefix (see FileMetadata's own doc comment).
                let desc = Descriptor {
                    media_type: oci::MEDIA_TYPE_MODEL_WEIGHT_RAW.to_string(),
                    digest: format!("sha256:{digest}"),
                    size,
                    annotations,
                    ..Default::default()
                };
                let req = FetchRequest {
                    url,
                    token,
                    xet,
                    label,
                };
                let pushed = fetch_and_push_stream(client, &req, session, &desc).await?;
                (desc, pushed)
            }
            // No usable digest ahead of time (a small, plain git-blob
            // file — config.json, a tokenizer file, ...; the ETag there
            // is a git blob sha1, not a sha256 of the content): buffer
            // it in memory and hash it ourselves. These are always tiny,
            // so buffering costs nothing.
            None => {
                let req = FetchRequest {
                    url,
                    token,
                    xet: None,
                    label,
                };
                let mut buf = Vec::new();
                download::fetch_once(client, &req, &mut buf)
                    .await
                    .with_context(|| format!("download {}", file.path))?;
                let digest = format!("sha256:{:x}", sha2::Sha256::digest(&buf));
                let desc = Descriptor {
                    media_type: oci::MEDIA_TYPE_MODEL_WEIGHT_RAW.to_string(),
                    digest,
                    size: buf.len() as i64,
                    annotations,
                    ..Default::default()
                };
                let pushed = push_bytes(session, &desc, &buf).await?;
                (desc, pushed)
            }
        };
        *changed |= pushed;
        Ok(desc)
    }

    /// Builds the CNCF model-spec config+manifest referencing `layers`
    /// and pushes both directly, without ever writing them to a local
    /// layout — the same JSON `oci::build_cncf_manifest` writes, reached
    /// via a throwaway local layout purely to reuse its exact
    /// construction (config/manifest JSON is at most a few KB, so that
    /// local round trip costs nothing measurable), then pushing those two
    /// blobs via [`push_stream`] instead of leaving them on disk.
    ///
    /// Returns the manifest's digest, which is what `--sign-key` signs.
    async fn push_cncf_manifest(
        session: &crate::ffi::PushSession,
        meta: &ModelMeta,
        model_repo: &str,
        filepath_annotation: &str,
        layers: Vec<Descriptor>,
        changed: &mut bool,
    ) -> Result<String> {
        let tmp = std::env::temp_dir().join(format!(
            "llmman-hf-transfer-manifest-{}-{}",
            std::process::id(),
            super::rand_suffix()
        ));
        std::fs::create_dir_all(&tmp)?;
        let manifest_desc =
            oci::build_cncf_manifest(&tmp, meta, model_repo, filepath_annotation, layers)?;
        let manifest_data = oci::read_blob(&tmp, &manifest_desc.digest)?;
        let config_digest = {
            let m: oci::Manifest = serde_json::from_slice(&manifest_data)?;
            m.config.digest
        };
        let config_data = oci::read_blob(&tmp, &config_digest)?;
        let _ = std::fs::remove_dir_all(&tmp);

        let config_desc = Descriptor {
            media_type: oci::MEDIA_TYPE_MODEL_CONFIG.to_string(),
            digest: config_digest,
            size: config_data.len() as i64,
            ..Default::default()
        };
        let pushed = push_bytes(session, &config_desc, &config_data)
            .await
            .context("push CNCF model config")?;
        *changed |= pushed;

        let pushed = push_bytes(session, &manifest_desc, &manifest_data)
            .await
            .context("push CNCF manifest")?;
        *changed |= pushed;
        if *changed {
            eprintln!("Writing manifest to image destination");
        }
        Ok(manifest_desc.digest)
    }

    /// Fetches `req` and streams it directly into a fresh push-stream,
    /// retrying the *whole* open-fetch-close-wait cycle (never reusing a
    /// pipe across attempts — see `download::fetch_once`'s own doc
    /// comment on why that would corrupt rather than actually retry).
    async fn fetch_and_push_stream(
        client: &reqwest::Client,
        req: &FetchRequest<'_>,
        session: &crate::ffi::PushSession,
        desc: &Descriptor,
    ) -> Result<bool> {
        retry_push(&req.label, |attempt| {
            let _ = attempt;
            async {
                let (mut writer, stream) = open_push_stream(session, desc)?;
                let write_result = download::fetch_once(client, req, &mut writer).await;
                drop(writer); // close our end first, or push_stream_wait hangs waiting for an EOF we're still holding open
                              // The goroutine's own error is more informative than a
                              // local write symptom like "broken pipe" (usually just
                              // the registry side having already failed and closed
                              // its end first), so it takes priority when both are present.
                let pushed = crate::ffi::push_stream_wait(stream)?;
                write_result?;
                Ok(pushed)
            }
        })
        .await
    }

    async fn push_bytes(
        session: &crate::ffi::PushSession,
        desc: &Descriptor,
        data: &[u8],
    ) -> Result<bool> {
        retry_push("push blob", |_| async {
            let (mut writer, stream) = open_push_stream(session, desc)?;
            let write_result = writer.write_all(data).map_err(anyhow::Error::from);
            drop(writer); // close our end first, or push_stream_wait hangs waiting for an EOF we're still holding open
            let pushed = crate::ffi::push_stream_wait(stream)?;
            write_result?;
            Ok(pushed)
        })
        .await
    }

    /// The push-side retry loop: retries `attempt` (the whole
    /// open-a-fresh-stream-and-push cycle) up to `client::MAX_ATTEMPTS`
    /// times, honoring `Retry-After`/exponential backoff between tries.
    /// It is its own loop rather than a shared helper because `attempt`
    /// needs the attempt number to seed the backoff (see
    /// `download::should_retry`); the metadata path's `client::probe`
    /// retries only a 429 and the blob-download loop lives in
    /// `download.rs`, so there is no shared retry helper to call here.
    async fn retry_push<F, Fut>(label: &str, mut attempt: F) -> Result<bool>
    where
        F: FnMut(u32) -> Fut,
        Fut: std::future::Future<Output = Result<bool>>,
    {
        let mut last_err: Option<anyhow::Error> = None;
        let mut next_delay = None;
        for i in 0..super::super::client::MAX_ATTEMPTS {
            if let Some(delay) = next_delay.take() {
                eprintln!(
                    "\n[llmman] retrying {label} (attempt {}/{}, wait {delay:?})",
                    i + 1,
                    super::super::client::MAX_ATTEMPTS
                );
                tokio::time::sleep(delay).await;
            }
            match attempt(i).await {
                Ok(pushed) => return Ok(pushed),
                Err(e) => {
                    eprintln!("[llmman] {label} error: {e:#}");
                    match download::should_retry(&e, i + 1) {
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
        Err(last_err.unwrap()).with_context(|| {
            format!(
                "{label} failed after {} attempts",
                super::super::client::MAX_ATTEMPTS
            )
        })
    }

    /// Opens a Go-shim pipe for pushing one descriptor within `session`
    /// (see `ffi::push_stream_open`/this module's own top doc comment),
    /// returning its write end wrapped as a `File` to write the blob's
    /// content into. Once done, drop the returned `File` (closing it —
    /// this is what signals EOF to the registry-push goroutine reading
    /// the other end) *before* calling `ffi::push_stream_wait` with the
    /// returned handle, or that call hangs waiting for an EOF that will
    /// never come while this side is still holding its own end open.
    fn open_push_stream(
        session: &crate::ffi::PushSession,
        desc: &Descriptor,
    ) -> Result<(std::fs::File, crate::ffi::PushStream)> {
        let annotations_json = match &desc.annotations {
            Some(a) if !a.is_empty() => serde_json::to_string(a)?,
            _ => String::new(),
        };
        let stream = crate::ffi::push_stream_open(
            session,
            &desc.media_type,
            &desc.digest,
            desc.size,
            &annotations_json,
        )?;

        // SAFETY: `stream.fd` is a fresh, valid, write-only fd/HANDLE
        // that Go just created via its own os.Pipe and handed to us as
        // its sole owner from this point on (see push_stream.go's own
        // doc comment) — nothing else in this process holds or will
        // ever touch it.
        #[cfg(unix)]
        let file = unsafe {
            <std::fs::File as std::os::unix::io::FromRawFd>::from_raw_fd(
                stream.fd as std::os::unix::io::RawFd,
            )
        };
        #[cfg(windows)]
        let file = unsafe {
            <std::fs::File as std::os::windows::io::FromRawHandle>::from_raw_handle(
                stream.fd as *mut std::ffi::c_void,
            )
        };

        Ok((file, stream))
    }
}

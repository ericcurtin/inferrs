//! Real Xet-protocol downloads for HuggingFace files, via `hf-xet`
//! (huggingface/xet-core, Apache-2.0). `huggingface_hub`'s own client
//! refuses to fetch any file over 50GB through a plain HTTP GET at all
//! (`file_download.py`: "Install `hf_xet` ... for xet-powered
//! downloads."), requiring this protocol instead.
//!
//! This crate's own plain-HTTP path (`crate::hf::download::fetch_once`)
//! has no such limit and, since it always sends a `Range` header, works
//! in practice past that threshold too — but it leans on a
//! CloudFront/S3-fronted CAS bridge `huggingface_hub` doesn't trust at
//! that scale, with none of this protocol's real advantages: no
//! chunk-level dedup or resume.
//!
//! This module reconstructs a Xet-backed file the way `hf_xet` itself
//! would and streams it to a writer — no local disk, no full-file
//! buffering — via `hf-xet`'s `xet_session::XetDownloadStreamGroup`
//! streaming API. Called directly from `crate::hf`'s fetch path (a
//! normal function call, no subprocess/FFI, since the caller is
//! already Rust).
//!
//! `hf-xet`'s `xet_session` module is the only dependency needed: it
//! re-exports `XetFileInfo`/`HeaderMap` directly, no need to depend on
//! `xet-client`/`xet-data`/`xet-runtime` separately.

use std::io::Write;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use xet::xet_session::{
    header, HeaderMap, XetFileDownloadGroup, XetFileInfo, XetSession, XetSessionBuilder,
};

/// How often [`download_to_path`] samples hf-xet's network-byte counter.
/// Matches the daemon's own poll cadence (`cmd::serve::stream_ffi_progress`).
const PROGRESS_POLL: Duration = Duration::from_millis(200);

/// Identifies and authenticates one Xet-backed HuggingFace file — just
/// the fields the Go shim's own header-parsing already extracts
/// (`X-Xet-Hash`, `X-Linked-Size`, `X-Linked-Etag`) plus the repo/revision
/// it already knows.
pub struct XetFileRef {
    /// HuggingFace endpoint, e.g. "https://huggingface.co" (see `hf::endpoint()`).
    pub endpoint: String,
    /// Repo type as used in the Hub API path: "models", "datasets", or "spaces".
    pub repo_type: String,
    /// "owner/repo".
    pub owner_repo: String,
    /// Commit this file's hash/size were resolved against — used only to
    /// build the xet-read-token URL; the download itself is addressed by
    /// content hash alone.
    pub revision: String,
    /// The Xet Merkle hash of the file's content (`X-Xet-Hash`).
    pub hash: String,
    /// The file's real size in bytes (`X-Linked-Size`).
    pub size: u64,
    /// The file's SHA-256, if known (`X-Linked-Etag`, unquoted) — lets
    /// `hf-xet` verify the reconstructed content.
    pub sha256: Option<String>,
    /// Bearer token for the Hub API, used only to fetch/refresh the
    /// short-lived Xet CAS token (never sent to CAS itself). `None` for
    /// an anonymous request.
    pub hf_token: Option<String>,
}

impl XetFileRef {
    /// The Hub API URL that hands out (and, via `with_token_refresh_url`,
    /// later refreshes) a short-lived Xet CAS access token — mirrors
    /// `huggingface_hub`'s `xet_connection_info_refresh_url`, and matches
    /// the resolve response's own `Link: <...>; rel="xet-auth"` header.
    fn refresh_route(&self) -> String {
        format!(
            "{}/api/{}/{}/xet-read-token/{}",
            self.endpoint.trim_end_matches('/'),
            self.repo_type,
            self.owner_repo,
            self.revision
        )
    }

    /// Auth headers for the CAS token fetch/refresh (never sent to CAS
    /// itself). Empty for an anonymous request.
    fn refresh_headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        if let Some(token) = &self.hf_token {
            let value = format!("Bearer {token}")
                .parse()
                .context("build Authorization header")?;
            headers.insert(header::AUTHORIZATION, value);
        }
        Ok(headers)
    }

    /// hf-xet's own description of the file to reconstruct.
    fn file_info(&self) -> XetFileInfo {
        match &self.sha256 {
            Some(sha256) => {
                XetFileInfo::new_with_sha256(self.hash.clone(), self.size, sha256.clone())
            }
            None => XetFileInfo::new(self.hash.clone(), self.size),
        }
    }
}

/// A session on the ambient tokio runtime.
fn new_session() -> Result<XetSession> {
    XetSessionBuilder::new()
        .build()
        .context("create xet session")
}

/// Streams `file`'s reconstructed content to `w`, chunk by chunk, with no
/// local disk touched and no full-file buffering — suitable for a
/// multi-hundred-GB file. `with_token_refresh_url` handles both the
/// initial CAS token fetch and any later refresh a long-running stream
/// needs, so this needs no upfront token round trip of its own.
///
/// Chunks arrive in file order while hf-xet fetches many ~64 MiB terms
/// in parallel, so `w` sees long pauses then bursts. Fine for a pipe; a
/// caller writing to a local file should prefer [`download_to_path`],
/// whose progress tracks the wire instead.
pub async fn stream_to_writer(file: &XetFileRef, w: &mut impl Write) -> Result<()> {
    let session = new_session()?;

    let group = session
        .new_download_stream_group()
        .context("create xet download stream group")?
        .with_token_refresh_url(file.refresh_route(), file.refresh_headers()?)
        .build()
        .await
        .context("authenticate xet download")?;

    let mut stream = group
        .download_stream(file.file_info(), None)
        .await
        .context("start xet download stream")?;

    while let Some(chunk) = stream.next().await.context("read xet download stream")? {
        w.write_all(&chunk).context("write downloaded chunk")?;
    }

    Ok(())
}

/// Downloads `file` to `dest` (created or truncated), reporting progress
/// as bytes received from CAS rather than bytes written — via hf-xet's
/// file-download API, the only one exposing its network-side counter.
///
/// `on_progress(delta)` deltas sum to exactly `file.size` on `Ok`: wire
/// bytes are capped at the file size (chunks are compressed on the wire)
/// and any shortfall is reported once the download completes.
///
/// Unlike [`stream_to_writer`] this hashes nothing; verify `dest` after.
pub async fn download_to_path(
    file: &XetFileRef,
    dest: &Path,
    mut on_progress: impl FnMut(u64),
) -> Result<()> {
    let session = new_session()?;

    let group = session
        .new_file_download_group()
        .context("create xet file download group")?
        .with_token_refresh_url(file.refresh_route(), file.refresh_headers()?)
        .build()
        .await
        .context("authenticate xet download")?;
    // hf-xet's download runs detached from this future; make sure
    // dropping it (a sibling in a buffer_unordered failing, say) stops
    // the transfer rather than leaving it writing to `dest`.
    let guard = AbortOnDrop(Some(group.clone()));

    // hf-xet resolves a relative `dest` against the process cwd, which
    // for a daemon is arbitrary.
    let dest = std::path::absolute(dest).context("absolute path of download destination")?;
    group
        .download_file_to_path(file.file_info(), dest)
        .await
        .context("start xet download")?;

    // `finish` consumes the group; poll a clone (lock-free) while it runs.
    let poller = group.clone();
    report_while(
        async { group.finish().await.context("xet download").map(|_| ()) },
        file.size,
        || poller.progress().total_transfer_bytes_completed,
        &mut on_progress,
    )
    .await?;
    guard.disarm();
    Ok(())
}

/// Drives `on_progress` from `wire_bytes()` every [`PROGRESS_POLL`] until
/// `finish` resolves, then tops up so the deltas sum to exactly `size`.
async fn report_while(
    finish: impl std::future::Future<Output = Result<()>>,
    size: u64,
    mut wire_bytes: impl FnMut() -> u64,
    mut on_progress: impl FnMut(u64),
) -> Result<()> {
    let mut finish = std::pin::pin!(finish);
    let mut reported: u64 = 0;
    loop {
        tokio::select! {
            result = &mut finish => {
                result?;
                break;
            }
            _ = tokio::time::sleep(PROGRESS_POLL) => {
                let wire = wire_bytes().min(size);
                if wire > reported {
                    on_progress(wire - reported);
                    reported = wire;
                }
            }
        }
    }
    if reported < size {
        on_progress(size - reported);
    }
    Ok(())
}

/// Aborts an in-flight download group unless disarmed first.
struct AbortOnDrop(Option<XetFileDownloadGroup>);

impl AbortOnDrop {
    fn disarm(mut self) {
        self.0.take();
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if let Some(group) = self.0.take() {
            let _ = group.abort();
        }
    }
}

/// Strips the surrounding quotes an `ETag`/`X-Linked-Etag` value
/// normally carries. A no-op if unquoted.
pub fn strip_etag_quotes(etag: &str) -> String {
    etag.trim_matches('"').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refresh_route_matches_huggingface_hub_xet_read_token_url_shape() {
        let file = XetFileRef {
            endpoint: "https://huggingface.co".to_string(),
            repo_type: "models".to_string(),
            owner_repo: "ornith-ai/Ornith-1.5-397B-GGUF".to_string(),
            revision: "771a73943cafcf88d496c423ea5dd3a1622b1c10".to_string(),
            hash: "d684593ed0cf033c7deef1eb122c2ba4302bf77551d8f8481e5ab5174444a642".to_string(),
            size: 244_309_803_808,
            sha256: None,
            hf_token: None,
        };
        assert_eq!(
            file.refresh_route(),
            "https://huggingface.co/api/models/ornith-ai/Ornith-1.5-397B-GGUF/xet-read-token/771a73943cafcf88d496c423ea5dd3a1622b1c10"
        );
    }

    #[test]
    fn refresh_route_tolerates_a_trailing_slash_on_endpoint() {
        let file = XetFileRef {
            endpoint: "https://huggingface.co/".to_string(),
            repo_type: "models".to_string(),
            owner_repo: "owner/repo".to_string(),
            revision: "main".to_string(),
            hash: "h".to_string(),
            size: 1,
            sha256: None,
            hf_token: None,
        };
        assert_eq!(
            file.refresh_route(),
            "https://huggingface.co/api/models/owner/repo/xet-read-token/main"
        );
    }

    #[test]
    fn strip_etag_quotes_removes_surrounding_quotes() {
        assert_eq!(
            strip_etag_quotes("\"c7775e6fae1a47619c199c81b865df9\""),
            "c7775e6fae1a47619c199c81b865df9"
        );
        assert_eq!(strip_etag_quotes("unquoted"), "unquoted");
    }

    /// Progress must be reported *while* the download runs, not just as
    /// one top-up at the end; the wire count is capped at the file size.
    /// Paused tokio time makes the poll cadence deterministic.
    #[tokio::test(start_paused = true)]
    async fn report_while_emits_deltas_during_the_download_and_caps_at_size() {
        use std::cell::Cell;
        use std::rc::Rc;

        let wire = Rc::new(Cell::new(0u64));
        let deltas = Rc::new(std::cell::RefCell::new(Vec::new()));
        let finish = {
            let wire = wire.clone();
            async move {
                // Bytes arrive across ~5 polls; the wire count then
                // overshoots the file size (compressed-chunk bookkeeping).
                for step in [100u64, 200, 300, 400, 550] {
                    tokio::time::sleep(PROGRESS_POLL).await;
                    wire.set(step);
                }
                tokio::time::sleep(PROGRESS_POLL).await;
                Ok(())
            }
        };
        report_while(finish, 500, || wire.get(), |n| deltas.borrow_mut().push(n))
            .await
            .expect("report_while");

        let deltas = deltas.borrow();
        assert!(
            deltas.len() >= 3,
            "expected several in-flight updates, got {deltas:?}"
        );
        assert_eq!(deltas.iter().sum::<u64>(), 500, "deltas must sum to size");
        assert!(deltas.iter().all(|&d| d > 0), "no zero deltas: {deltas:?}");
    }

    /// A failing download must not be topped up to look complete.
    #[tokio::test(start_paused = true)]
    async fn report_while_propagates_the_error_without_a_top_up() {
        let mut reported = 0u64;
        let err = report_while(
            async { anyhow::bail!("boom") },
            500,
            || 0,
            |n| reported += n,
        )
        .await
        .expect_err("must fail");
        assert!(err.to_string().contains("boom"));
        assert_eq!(reported, 0);
    }

    /// Real streaming download of a small (~450KB), genuinely Xet-backed
    /// file — hf-internal-testing/tiny-random-gpt2's model.safetensors.
    /// Needs real network access (same convention as hf.rs's Go-side
    /// equivalent, hfheadmetadata_test.go). hash/size/sha256 came from:
    ///   curl -sI -H 'Accept-Encoding: identity' <resolve URL>
    #[tokio::test]
    async fn stream_to_writer_downloads_a_real_small_xet_backed_file() {
        let file = XetFileRef {
            endpoint: crate::hf::endpoint(),
            repo_type: "models".to_string(),
            owner_repo: "hf-internal-testing/tiny-random-gpt2".to_string(),
            revision: "71034c5d8bde858ff824298bdedc65515b97d2b9".to_string(),
            hash: "f8accece953fd366d4ce30597b97acc1ccedc3c785187a5ef6ecb4a8e1755122".to_string(),
            size: 453_864,
            sha256: Some(
                "8111d5afb0715dbf5a31396d31432cb56370ba23f6650a035ea0fc8a20b4e500".to_string(),
            ),
            hf_token: crate::hf::token(),
        };

        let mut data = Vec::new();
        stream_to_writer(&file, &mut data)
            .await
            .expect("stream_to_writer");
        assert_eq!(data.len(), 453_864);

        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        hasher.update(&data);
        let got_sha256 = hex::encode(hasher.finalize());
        assert_eq!(
            got_sha256, "8111d5afb0715dbf5a31396d31432cb56370ba23f6650a035ea0fc8a20b4e500",
            "downloaded content doesn't match the expected sha256"
        );
    }

    /// Same file via the to-path API: content must match, and progress
    /// deltas must sum to exactly the file size.
    #[tokio::test]
    async fn download_to_path_downloads_a_real_small_xet_backed_file_and_reports_its_size() {
        let file = XetFileRef {
            endpoint: crate::hf::endpoint(),
            repo_type: "models".to_string(),
            owner_repo: "hf-internal-testing/tiny-random-gpt2".to_string(),
            revision: "71034c5d8bde858ff824298bdedc65515b97d2b9".to_string(),
            hash: "f8accece953fd366d4ce30597b97acc1ccedc3c785187a5ef6ecb4a8e1755122".to_string(),
            size: 453_864,
            sha256: Some(
                "8111d5afb0715dbf5a31396d31432cb56370ba23f6650a035ea0fc8a20b4e500".to_string(),
            ),
            hf_token: crate::hf::token(),
        };

        let dest = std::env::temp_dir().join(format!(
            "llmman-xet-download-to-path-{}.safetensors",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&dest);
        let mut reported = 0u64;
        download_to_path(&file, &dest, |n| reported += n)
            .await
            .expect("download_to_path");
        assert_eq!(
            reported, 453_864,
            "progress deltas must sum to the file size"
        );

        let data = std::fs::read(&dest).expect("read downloaded file");
        let _ = std::fs::remove_file(&dest);
        assert_eq!(data.len(), 453_864);
        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        hasher.update(&data);
        assert_eq!(
            hex::encode(hasher.finalize()),
            "8111d5afb0715dbf5a31396d31432cb56370ba23f6650a035ea0fc8a20b4e500",
            "downloaded content doesn't match the expected sha256"
        );
    }

    /// A malformed hash must fail cleanly, not panic. Uses a real
    /// repo/revision so the xet-read-token exchange succeeds and it's
    /// hf-xet's own local hash validation being tested — still needs
    /// network access for that token exchange.
    #[tokio::test]
    async fn stream_to_writer_rejects_an_invalid_hash_cleanly() {
        let file = XetFileRef {
            endpoint: crate::hf::endpoint(),
            repo_type: "models".to_string(),
            owner_repo: "hf-internal-testing/tiny-random-gpt2".to_string(),
            revision: "71034c5d8bde858ff824298bdedc65515b97d2b9".to_string(),
            hash: "not-a-valid-hex-hash".to_string(),
            size: 1,
            sha256: None,
            hf_token: crate::hf::token(),
        };

        let mut sink = Vec::new();
        let err = stream_to_writer(&file, &mut sink)
            .await
            .expect_err("an invalid hash must not succeed");
        assert!(
            format!("{err:#}").contains("hash"),
            "expected the error to mention the bad hash, got: {err:#}"
        );
    }
}

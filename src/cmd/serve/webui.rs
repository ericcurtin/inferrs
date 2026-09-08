//! The web UI's routes: `/` is the page, `/ui/*` its files, all from
//! [`crate::webui`]. The page talks to the daemon over the same APIs every
//! other client uses, plus `/llmman/shell` (see `super::shell`).
//!
//! `LLMMAN_WEBUI_DIR=webui llmman serve` serves that directory from disk
//! instead, uncompressed and uncached, for working on the UI.

use std::path::{Component, Path, PathBuf};
use std::sync::LazyLock;

use axum::body::Body;
use axum::extract::Path as UrlPath;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};

use crate::webui::{self, content_type, Asset};

/// `LLMMAN_WEBUI_DIR`, canonicalized, if set to an existing directory.
static DEV_DIR: LazyLock<Option<PathBuf>> = LazyLock::new(|| {
    let dir = std::env::var_os("LLMMAN_WEBUI_DIR").filter(|d| !d.is_empty())?;
    match std::fs::canonicalize(&dir) {
        Ok(dir) => {
            eprintln!(
                "[llmman] serving the web UI from {} (LLMMAN_WEBUI_DIR)",
                dir.display()
            );
            Some(dir)
        }
        Err(e) => {
            eprintln!(
                "[llmman] LLMMAN_WEBUI_DIR={}: {e}; serving the built-in UI",
                dir.to_string_lossy()
            );
            None
        }
    }
});

/// `GET /`: the page for a client that accepts HTML, else the liveness
/// line scripts have always seen.
pub(super) async fn handle_root(headers: HeaderMap) -> Response {
    let wants_html = headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|s| s.contains("text/html"));
    if !wants_html {
        return "llmman is running".into_response();
    }
    match DEV_DIR.as_ref() {
        Some(dir) => serve_from_disk(dir, "index.html").await,
        None => serve_asset(&headers, webui::index()),
    }
}

/// Every HTML reply is unframeable: a page that framed the UI would open
/// its WebSocket to /llmman/shell from a trusted localhost origin.
fn finish(mut response: Response) -> Response {
    let is_html = response
        .headers()
        .get(header::CONTENT_TYPE)
        .is_some_and(|v| v.as_bytes().starts_with(b"text/html"));
    if is_html {
        let h = response.headers_mut();
        h.insert(
            header::CONTENT_SECURITY_POLICY,
            "frame-ancestors 'none'".parse().unwrap(),
        );
        h.insert(header::X_FRAME_OPTIONS, "DENY".parse().unwrap());
    }
    response
}

/// `GET /ui/*path`. No fallback to `index.html`: the UI has one page and
/// routes with the fragment. Under `LLMMAN_WEBUI_DIR` only images (the
/// mark build.rs downloads, never in the tree) fall back to the binary;
/// a missing source file stays a 404 rather than a stale embedded copy.
pub(super) async fn handle_asset(UrlPath(path): UrlPath<String>, headers: HeaderMap) -> Response {
    if let Some(dir) = DEV_DIR.as_ref() {
        let response = serve_from_disk(dir, &path).await;
        let is_image = content_type::for_path(&path).is_some_and(|t| t.starts_with("image/"));
        if response.status() != StatusCode::NOT_FOUND || !is_image {
            return response;
        }
    }
    match webui::asset(&path) {
        Some(asset) => serve_asset(&headers, asset),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Gzipped, with a content ETag; a matching `If-None-Match` is a 304.
/// `no-cache` rather than a `max-age`, since the paths are not
/// content-addressed and an upgrade must not leave the old UI cached.
fn serve_asset(headers: &HeaderMap, asset: Asset) -> Response {
    // Weak comparison, as GET allows: `W/` prefixes are ignored and `*`
    // matches any representation.
    let matches = headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(',').any(|tag| {
                let tag = tag.trim();
                tag == "*" || tag.strip_prefix("W/").unwrap_or(tag) == asset.etag
            })
        });
    let response = Response::builder()
        .header(header::ETAG, asset.etag)
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::VARY, "accept-encoding");
    if matches {
        return response
            .status(StatusCode::NOT_MODIFIED)
            .body(Body::empty())
            .unwrap();
    }
    finish(
        response
            .header(header::CONTENT_TYPE, asset.content_type)
            .header(header::CONTENT_ENCODING, "gzip")
            .body(Body::from(asset.gzip))
            .unwrap(),
    )
}

/// The development path: `rel` under `dir`, read on every request. Only
/// plain relative paths, and only files that canonicalize to inside `dir`
/// (so a symlink cannot lead out either).
async fn serve_from_disk(dir: &Path, rel: &str) -> Response {
    let not_found = || StatusCode::NOT_FOUND.into_response();
    let rel_path = Path::new(rel);
    if rel_path
        .components()
        .any(|c| !matches!(c, Component::Normal(_)))
    {
        return not_found();
    }
    let Ok(full) = tokio::fs::canonicalize(dir.join(rel_path)).await else {
        return not_found();
    };
    if !full.starts_with(dir) {
        return not_found();
    }
    let Ok(bytes) = tokio::fs::read(&full).await else {
        return not_found();
    };
    finish(
        Response::builder()
            .header(
                header::CONTENT_TYPE,
                content_type::for_path(rel).unwrap_or("application/octet-stream"),
            )
            .header(header::CACHE_CONTROL, "no-store")
            .body(Body::from(bytes))
            .unwrap(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The development root is the whole reachable tree: parent segments
    /// and symlinks pointing out of it are 404s like any missing file.
    #[tokio::test]
    async fn the_development_directory_cannot_be_escaped() {
        let tmp = std::env::temp_dir().join(format!("llmman-webui-dev-{}", std::process::id()));
        let root = tmp.join("webui");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("app.js"), "ok").unwrap();
        std::fs::write(tmp.join("secret.txt"), "no").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(tmp.join("secret.txt"), root.join("link.txt")).unwrap();
        let root = std::fs::canonicalize(&root).unwrap();

        let ok = serve_from_disk(&root, "app.js").await;
        assert_eq!(ok.status(), StatusCode::OK);
        assert_eq!(ok.headers()["cache-control"], "no-store");
        for bad in ["../secret.txt", "/etc/hosts", "missing.js", "link.txt"] {
            if bad == "link.txt" && !cfg!(unix) {
                continue;
            }
            let r = serve_from_disk(&root, bad).await;
            assert_eq!(r.status(), StatusCode::NOT_FOUND, "{bad}");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }
}

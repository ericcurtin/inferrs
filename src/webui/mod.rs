//! The web UI's files, gzipped into the binary by build.rs as the
//! [`ASSETS`] table: `(path, content-type, etag, bytes)`, keyed by the path
//! relative to `webui/`. Served by `cmd::serve::webui`.

pub mod content_type;

include!(concat!(env!("OUT_DIR"), "/webui_assets.rs"));

/// One embedded file. `etag` is already quoted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Asset {
    pub content_type: &'static str,
    pub etag: &'static str,
    pub gzip: &'static [u8],
}

/// The file at `path` (relative, `/`-separated). A table lookup, so `..`
/// and the like simply never match.
pub fn asset(path: &str) -> Option<Asset> {
    ASSETS
        .binary_search_by(|(p, _, _, _)| (*p).cmp(path))
        .ok()
        .map(|i| {
            let (_, content_type, etag, gzip) = ASSETS[i];
            Asset {
                content_type,
                etag,
                gzip,
            }
        })
}

/// `webui/index.html`, the single page the UI is.
pub fn index() -> Asset {
    asset("index.html").expect("build.rs asserts webui/index.html exists")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_sorted_so_lookups_can_bisect() {
        let paths: Vec<&str> = ASSETS.iter().map(|(p, _, _, _)| *p).collect();
        let mut sorted = paths.clone();
        sorted.sort_unstable();
        assert_eq!(paths, sorted);
    }

    #[test]
    fn every_asset_is_found_by_its_own_path_and_is_gzip() {
        for (path, content_type, etag, gzip) in ASSETS {
            let found = asset(path).unwrap_or_else(|| panic!("{path} not found"));
            assert_eq!(found.content_type, *content_type);
            assert_eq!(found.etag, *etag);
            assert!(
                etag.starts_with('"') && etag.ends_with('"'),
                "{path}: {etag}"
            );
            assert_eq!(&found.gzip[..2], &[0x1f, 0x8b], "{path} is not gzip");
            assert_eq!(found.gzip, *gzip);
        }
    }

    #[test]
    fn a_path_outside_the_table_is_none() {
        for bad in [
            "",
            "/index.html",
            "../index.html",
            "index.html/",
            "INDEX.HTML",
            "vendor",
        ] {
            assert!(asset(bad).is_none(), "{bad:?}");
        }
        assert!(asset("vendor/xterm.js").is_some());
    }

    #[test]
    fn content_types_by_extension() {
        assert_eq!(
            content_type::for_path("a/b.js"),
            Some("application/javascript; charset=utf-8")
        );
        assert_eq!(
            content_type::for_path("VERSIONS"),
            Some("text/plain; charset=utf-8")
        );
        assert_eq!(content_type::for_path("x.wasm"), None);
    }
}

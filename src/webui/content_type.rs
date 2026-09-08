//! `Content-Type` by file extension for the web UI's files. Compiled into
//! both build.rs (which refuses unknown extensions) and the daemon.

pub fn for_path(path: &str) -> Option<&'static str> {
    let ext = path.rsplit_once('.').map(|(_, ext)| ext);
    Some(match ext {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "application/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        Some("json") | Some("webmanifest") => "application/json; charset=utf-8",
        Some("txt") | None => "text/plain; charset=utf-8",
        Some(_) => return None,
    })
}

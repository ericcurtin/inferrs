//! Build script for inferrs.
//!
//! **Web UI compression** — compress `ui/index.html` into `$OUT_DIR/ui.html.gz`
//! using best-level gzip compression.  The server embeds the result via
//! `include_bytes!(concat!(env!("OUT_DIR"), "/ui.html.gz"))` and serves it
//! with `Content-Encoding: gzip` in daemon mode (no model argument).
//! The build fails loudly if the compressed output exceeds 1 MiB so that the
//! size budget is enforced at compile time rather than discovered at runtime.
//!
//! Note: the Go OCI shared library (`libocimodels`) is loaded on demand via
//! `dlopen` at runtime (see `pull.rs`), so no link-time configuration is
//! needed here.

use flate2::{write::GzEncoder, Compression};
use std::{env, fs, io::Write, path::PathBuf};

const SIZE_LIMIT_BYTES: u64 = 1024 * 1024; // 1 MiB

fn main() {
    // -----------------------------------------------------------------------
    // 1. Web UI compression
    // -----------------------------------------------------------------------

    // Re-run this script if the UI source changes.
    println!("cargo:rerun-if-changed=ui/index.html");

    let out_dir: PathBuf = env::var("OUT_DIR").expect("OUT_DIR not set").into();
    let gz_path = out_dir.join("ui.html.gz");

    let html = fs::read("ui/index.html")
        .expect("failed to read inferrs/ui/index.html – run from the crate root");

    let file = fs::File::create(&gz_path).expect("failed to create ui.html.gz in OUT_DIR");
    let mut encoder = GzEncoder::new(file, Compression::best());
    encoder
        .write_all(&html)
        .expect("failed to compress ui/index.html");
    encoder.finish().expect("failed to finalise gzip stream");

    let compressed_size = fs::metadata(&gz_path)
        .expect("failed to stat ui.html.gz")
        .len();

    assert!(
        compressed_size <= SIZE_LIMIT_BYTES,
        "ui.html.gz is {compressed_size} bytes ({:.1} KiB), which exceeds the 1 MiB limit. \
         Reduce the UI size before building.",
        compressed_size as f64 / 1024.0
    );

    println!(
        "cargo:warning=inferrs web UI compressed to {:.1} KiB (budget: 1024 KiB)",
        compressed_size as f64 / 1024.0
    );
}

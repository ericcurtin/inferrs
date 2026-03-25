//! Integration tests for the inferrs HTTP server.
//!
//! These tests start the binary against a real model and verify that
//! `/v1/chat/completions` returns a well-formed OpenAI-compatible response.
//!
//! # Running
//!
//! The tests require network access to download models from HuggingFace Hub
//! the first time they run (subsequent runs use the local cache).
//!
//! ```
//! cargo test --test server_integration -- --nocapture
//! ```
//!
//! Each test starts `inferrs serve <model>` on a random free port, waits for
//! the server to become healthy, sends a chat completion request, and asserts
//! the response structure.

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Finds a free TCP port on localhost by binding to port 0 and reading the
/// assigned port back from the OS.
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind to port 0")
        .local_addr()
        .expect("local_addr")
        .port()
}

/// Starts `inferrs serve <model_id>` on `port` and returns the [`Child`].
///
/// The process inherits stderr (so build / tracing output is visible with
/// `--nocapture`) and has stdout suppressed to keep test output clean.
///
/// `--device auto` is passed, which picks Metal on macOS (always compiled in
/// via the `[target.cfg(macos)]` dependency block in `Cargo.toml`), CUDA on
/// Linux/Windows when available, and falls back to CPU otherwise.
fn spawn_server(model_id: &str, port: u16) -> Child {
    let bin = env!("CARGO_BIN_EXE_inferrs");
    Command::new(bin)
        .args([
            "serve",
            model_id,
            "--port",
            &port.to_string(),
            "--host",
            "127.0.0.1",
            "--max-tokens",
            "64",
            "--dtype",
            "bf16",
            "--device",
            "auto",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to spawn inferrs")
}

/// Polls `/health` until the server responds 200, then returns.  Panics if
/// the timeout is exceeded.
fn wait_for_health(port: u16, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    let url = format!("http://127.0.0.1:{}/health", port);
    loop {
        if Instant::now() > deadline {
            panic!(
                "server on port {} did not become healthy within {:?}",
                port, timeout
            );
        }
        match ureq::get(&url).call() {
            Ok(resp) if resp.status() == 200 => return,
            _ => std::thread::sleep(Duration::from_millis(500)),
        }
    }
}

/// Returns `true` when `text` contains at least one ASCII alphabetic character,
/// meaning the model produced something that looks like a real word rather than
/// pure whitespace, repeated punctuation, or garbage bytes.
fn looks_intelligible(text: &str) -> bool {
    text.chars().any(|c| c.is_ascii_alphabetic())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

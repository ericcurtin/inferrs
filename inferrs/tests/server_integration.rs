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
            "128",
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

/// Starts `inferrs serve <model_id>` with an explicit TurboQuant bit-width.
///
/// TurboQuant is on by default (8-bit); this helper overrides the bit-width via
/// `--turbo-quant=<bits>`.  Uses `require_equals` syntax as defined in the CLI
/// (`--turbo-quant=4` for 4-bit).
fn spawn_server_turbo(model_id: &str, port: u16, bits: u8) -> Child {
    let bin = env!("CARGO_BIN_EXE_inferrs");
    let turbo_flag = format!("--turbo-quant={}", bits);
    Command::new(bin)
        .args([
            "serve",
            model_id,
            "--port",
            &port.to_string(),
            "--host",
            "127.0.0.1",
            "--max-tokens",
            "128",
            "--dtype",
            "bf16",
            "--device",
            "auto",
            &turbo_flag,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to spawn inferrs with TurboQuant")
}

/// Send a single chat-completion request and return the assistant's text.
///
/// Uses 128 max_tokens to allow Qwen3's `<think>…</think>` preamble to
/// complete before the model outputs the actual reply.
fn chat_completion(port: u16, user_message: &str) -> String {
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", port);
    let body = serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": 128,
        "temperature": 0.0
    });
    let resp: serde_json::Value = ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_json(&body)
        .expect("chat completion request failed")
        .into_json()
        .expect("failed to parse chat completion response");
    resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Returns a synthetic prompt of approximately `target_len` tokens by
/// repeating a short sentence.  Each word is roughly 1.3 tokens on average,
/// so `target_len / 6` repetitions ≈ `target_len` tokens.
fn synthetic_prompt(target_tokens: usize) -> String {
    let sentence = "The quick brown fox jumps over the lazy dog. ";
    // Approximately 10 tokens per repetition (9 words × ~1.1 tokens/word).
    let reps = (target_tokens / 10).max(1);
    sentence.repeat(reps)
}

/// Starts `inferrs serve <model_id>` with paged attention enabled.
fn spawn_server_paged(model_id: &str, port: u16) -> std::process::Child {
    let bin = env!("CARGO_BIN_EXE_inferrs");
    std::process::Command::new(bin)
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
            "--paged-attention",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("failed to spawn inferrs with paged attention")
}

/// Verifies that `google/gemma-4-E2B-it` returns a coherent (intelligible)
/// response to a trivial chat message.
///
/// The test is marked `#[ignore]` so that it is skipped by the default
/// `cargo test` run (which has no GPU / big-model download budget).  Run it
/// explicitly with:
///
/// ```
/// cargo test --test server_integration gemma4_e2b_returns_intelligible_output -- --ignored --nocapture
/// ```
#[test]
#[ignore = "requires model download and significant compute; run with --ignored"]
fn gemma4_e2b_returns_intelligible_output() {
    let model_id = "google/gemma-4-E2B-it";
    let port = free_port();

    let mut server = spawn_server(model_id, port);

    // Give the server up to 5 minutes to download, load weights, and start.
    let result = std::panic::catch_unwind(|| {
        wait_for_health(port, Duration::from_secs(300));

        let url = format!("http://127.0.0.1:{}/v1/chat/completions", port);
        let body = serde_json::json!({
            "model": model_id,
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"}
            ],
            "max_tokens": 64,
            "temperature": 0.0
        });

        let resp = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_string(&body.to_string())
            .expect("HTTP request failed");

        assert_eq!(resp.status(), 200, "expected 200 OK from chat completions");

        let json: serde_json::Value = resp.into_json().expect("response is not valid JSON");

        // Validate top-level structure
        assert_eq!(
            json["object"].as_str().unwrap_or(""),
            "chat.completion",
            "unexpected object type: {}",
            json
        );

        let choices = json["choices"]
            .as_array()
            .expect("choices must be an array");
        assert!(!choices.is_empty(), "choices array must not be empty");

        let content = choices[0]["message"]["content"]
            .as_str()
            .expect("choices[0].message.content must be a string");

        assert!(
            !content.is_empty(),
            "model returned an empty response for 'What is 2 + 2?'"
        );

        assert!(
            looks_intelligible(content),
            "model output does not look intelligible (no ASCII alphabetic chars).\
             \nGot: {:?}",
            content
        );

        // For "2 + 2 = 4", the answer should contain "4" somewhere.
        assert!(
            content.contains('4'),
            "expected the answer '4' to appear in the response to 'What is 2 + 2?'.\
             \nGot: {:?}",
            content
        );

        eprintln!("gemma4 response: {:?}", content);
    });

    // Always kill the server child process, even on panic.
    let _ = server.kill();
    let _ = server.wait();

    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// Verifies that `google/gemma-4-E2B-it` handles a prompt longer than the
/// sliding-window size (512 tokens) without crashing.
///
/// This directly exercises the `RetainingRotatingKvCache` wrap-around path and
/// the sliding-window attention mask clamping code, both of which are bypassed
/// by short "Hello"-style prompts.
///
/// Run with:
/// ```
/// cargo test --test server_integration gemma4_e2b_long_context_no_crash -- --ignored --nocapture
/// ```
#[test]
#[ignore = "requires model download and significant compute; run with --ignored"]
fn gemma4_e2b_long_context_no_crash() {
    let model_id = "google/gemma-4-E2B-it";
    let port = free_port();

    let mut server = spawn_server(model_id, port);

    let result = std::panic::catch_unwind(|| {
        wait_for_health(port, Duration::from_secs(300));

        // A prompt of ~600 tokens exercises the sliding-window wrap (window=512).
        let long_prompt = synthetic_prompt(600);
        let prompt = format!(
            "{}. In one word, what animal did the fox jump over?",
            long_prompt
        );

        let url = format!("http://127.0.0.1:{}/v1/chat/completions", port);
        let body = serde_json::json!({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0.0
        });

        let resp = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_string(&body.to_string())
            .expect("HTTP request failed with long prompt");

        assert_eq!(
            resp.status(),
            200,
            "expected 200 OK for long-context request (got {})",
            resp.status()
        );

        let json: serde_json::Value = resp.into_json().expect("response is not valid JSON");
        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");

        assert!(
            looks_intelligible(content),
            "long-context response is not intelligible.\nGot: {:?}",
            content
        );

        eprintln!("gemma4 long-context response: {:?}", content);
    });

    let _ = server.kill();
    let _ = server.wait();

    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// Verifies that `google/gemma-4-E2B-it` with 4-bit TurboQuant compression
/// returns a coherent response to a basic chat message.
///
/// This test exercises the 4-bit KV cache path end-to-end, ensuring that the
/// nibble-packing/unpacking round-trip does not corrupt inference.
///
/// Run with:
/// ```
/// cargo test --test server_integration gemma4_e2b_turbo_quant_4bit_returns_intelligible_output -- --ignored --nocapture
/// ```
#[test]
#[ignore = "requires model download and significant compute; run with --ignored"]
fn gemma4_e2b_turbo_quant_4bit_returns_intelligible_output() {
    let model_id = "google/gemma-4-E2B-it";
    let port = free_port();

    let mut server = spawn_server_turbo(model_id, port, 4);

    let result = std::panic::catch_unwind(|| {
        wait_for_health(port, Duration::from_secs(300));

        let resp = chat_completion(port, "What is 2 + 2?");
        assert!(
            looks_intelligible(&resp),
            "4-bit TurboQuant response is not intelligible.\nGot: {:?}",
            resp
        );
        assert!(
            resp.contains('4'),
            "expected the answer '4' in TurboQuant response.\nGot: {:?}",
            resp
        );
        eprintln!("4-bit TurboQuant response: {:?}", resp);
    });

    let _ = server.kill();
    let _ = server.wait();

    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// Verifies that `google/gemma-4-E2B-it` with `--paged-attention` handles a
/// long prompt (>512 tokens) and that a second request after the first
/// completes successfully (exercises block reuse correctness).
///
/// Run with:
/// ```
/// cargo test --test server_integration gemma4_e2b_paged_long_context_and_second_request -- --ignored --nocapture
/// ```
#[test]
#[ignore = "requires model download and significant compute; run with --ignored"]
fn gemma4_e2b_paged_long_context_and_second_request() {
    let model_id = "google/gemma-4-E2B-it";
    let port = free_port();

    let mut server = spawn_server_paged(model_id, port);

    let result = std::panic::catch_unwind(|| {
        wait_for_health(port, Duration::from_secs(300));

        // First request: long prompt to exercise the paged sliding-window path.
        let long_prompt = synthetic_prompt(600);
        let prompt1 = format!(
            "{}. In one word, what animal did the fox jump over?",
            long_prompt
        );
        let resp1 = chat_completion(port, &prompt1);
        assert!(
            looks_intelligible(&resp1),
            "paged long-context first request not intelligible.\nGot: {:?}",
            resp1
        );
        eprintln!("paged long-context first response: {:?}", resp1);

        // Second request: short, to verify block reuse doesn't corrupt the cache.
        let resp2 = chat_completion(port, "What is 2 + 2?");
        assert!(
            resp2.contains('4'),
            "paged second request (after long-context) expected '4'.\nGot: {:?}",
            resp2
        );
        eprintln!("paged second request response: {:?}", resp2);
    });

    let _ = server.kill();
    let _ = server.wait();

    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// Verifies that `inferrs run google/gemma-4-E2B-it --image <path>`
/// processes an image without error and returns an intelligible response.
///
/// Downloads a public-domain cat JPEG via `curl`, then runs `inferrs run` in
/// non-interactive mode and checks that:
///   1. The process exits cleanly (no panic / error path hit).
///   2. Standard output contains at least one alphabetic character.
///
/// Run with:
/// ```
/// cargo test --test server_integration gemma4_e2b_run_image -- --ignored --nocapture
/// ```
#[test]
#[ignore = "requires model download, network access, and significant compute; run with --ignored"]
fn gemma4_e2b_run_image() {
    let model_id = "google/gemma-4-E2B-it";
    let bin = env!("CARGO_BIN_EXE_inferrs");

    // Full-size public-domain cat photo (~2.8 MB).  The 320 px thumbnails are
    // served as HTML error pages without the right request headers; the original
    // file is always available unconditionally.
    let img_url = "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg";

    let tmp_dir = std::env::temp_dir();
    let img_path = tmp_dir.join("inferrs_test_image.jpg");

    // --fail makes curl exit non-zero on HTTP errors instead of silently
    // saving the error page as the output file.
    let dl_status = std::process::Command::new("curl")
        .args([
            "--silent",
            "--location",
            "--fail",
            "--output",
            img_path.to_str().expect("non-UTF8 temp path"),
            img_url,
        ])
        .status()
        .expect("failed to run curl — is it installed?");
    assert!(
        dl_status.success(),
        "curl failed to download test image from {img_url}"
    );
    assert!(
        img_path.exists() && img_path.metadata().map(|m| m.len()).unwrap_or(0) > 10_000,
        "downloaded image file is empty or too small (likely an error page): {}",
        img_path.display()
    );

    // Run `inferrs run` in non-interactive mode.
    // Flags must come before the positional <MODEL> [PROMPT] arguments.
    // Note: --quantize is now on by default (Q4K), so no need to specify it.
    let output = std::process::Command::new(bin)
        .args([
            "run",
            "--device",
            "auto",
            "--dtype",
            "bf16",
            "--max-tokens",
            "32",
            "--temperature=0.0",
            "--image",
            img_path.to_str().expect("non-UTF8 temp path"),
            model_id,
            "What animal is in this image? Answer in one word.",
        ])
        .output()
        .expect("failed to spawn inferrs run");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    eprintln!("inferrs run stdout: {stdout:?}");
    eprintln!(
        "inferrs run stderr (tail): {:?}",
        &stderr[stderr.len().saturating_sub(500)..]
    );

    assert!(
        output.status.success(),
        "inferrs run exited with non-zero status: {}\nstderr: {}",
        output.status,
        stderr
    );

    assert!(
        looks_intelligible(&stdout),
        "inferrs run --image response is not intelligible (no ASCII alphabetic chars).\nGot: {stdout:?}"
    );

    eprintln!("gemma4 run --image response: {stdout:?}");
}

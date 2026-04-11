//! Interactive REPL for `inferrs run` — talks to a running `inferrs serve`
//! daemon via the Ollama-compatible HTTP API, exactly as `ollama run` does.
//!
//! The server URL is resolved from (in priority order):
//!   1. `--host` / `--port` CLI flags
//!   2. `INFERRS_HOST` environment variable  (e.g. `http://10.0.0.5:17434`)
//!   3. Default: `http://127.0.0.1:17434`
//!
//! When the server is not reachable, `inferrs run` automatically starts
//! `inferrs serve` (no arguments — the bare daemon) in the background and
//! waits for it to become ready, mirroring `ollama run`'s `ensureServerRunning`.
//! Hardware and model-loading configuration belongs on `inferrs serve`, not here.

use anyhow::{Context, Result};
use clap::Parser;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{self, ClearType},
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};

// ─── CLI args ────────────────────────────────────────────────────────────────

/// Default port for the Ollama-compatible API (matches `inferrs serve` default
/// and real Ollama's default port).
const DEFAULT_PORT: u16 = 17434;

#[derive(Parser, Clone)]
pub struct RunArgs {
    /// HuggingFace model ID (e.g. Qwen/Qwen3-0.6B).
    /// Passed to `inferrs serve` when auto-starting the daemon.
    pub model: String,

    /// Optional prompt — when given, run non-interactively and exit
    pub prompt: Option<String>,

    /// System prompt
    #[arg(long)]
    pub system: Option<String>,

    // ── Sampling ──────────────────────────────────────────────────────────────
    /// Default sampling temperature
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f64,

    /// Default nucleus sampling threshold
    #[arg(long, default_value_t = 0.9)]
    pub top_p: f64,

    /// Default top-k sampling
    #[arg(long, default_value_t = 50)]
    pub top_k: usize,

    /// Default max tokens to generate per response
    #[arg(long, default_value_t = 2048)]
    pub max_tokens: usize,

    // ── Server connection ─────────────────────────────────────────────────────
    /// Address of the `inferrs serve` daemon.
    /// Overrides `INFERRS_HOST`.  Defaults to `127.0.0.1`.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port of the `inferrs serve` daemon.
    /// Overrides the port part of `INFERRS_HOST`.  Defaults to 17434.
    #[arg(long, default_value_t = DEFAULT_PORT)]
    pub port: u16,

    // ── Model-loading flags (forwarded to `inferrs serve` on auto-start) ────────
    /// Git branch or tag on HuggingFace Hub
    #[arg(long, default_value = "main")]
    pub revision: String,

    /// Weight data type: f32, f16, bf16
    #[arg(long, default_value = "bf16")]
    pub dtype: String,

    /// Device: cpu, cuda, metal, or auto
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Fraction of GPU/CPU memory to reserve for paged KV blocks.
    /// e.g. `--paged-attention=0.9` reserves 90 % of available memory.
    /// When used as a plain flag the default 0.9 is applied.
    /// Omit entirely to disable paged attention.
    #[arg(long, num_args(0..=1), default_missing_value("0.9"), require_equals(true),
          value_name = "FRACTION")]
    pub paged_attention: Option<f64>,

    /// TurboQuant KV cache compression bit-width (Qwen3/Gemma4).
    /// Enabled by default at 8 bits. Pass a bit-width (1–8) or `false`.
    #[arg(long, default_value = "8", require_equals(true))]
    pub turbo_quant: crate::TurboQuantArg,

    /// Specific GGUF filename to load from a GGUF-only repo.
    ///
    /// Only used when the repo contains GGUF files but no safetensors weights
    /// (e.g. ggml-org/gemma-4-E2B-it-GGUF).  When omitted, inferrs picks the
    /// best available quantization automatically.
    #[arg(long, value_name = "FILENAME")]
    pub gguf_file: Option<String>,

    /// Quantize model weights on first use and cache as GGUF.
    /// Accepted formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K.
    /// Plain `--quantize` defaults to Q4K.
    #[arg(long, num_args(0..=1), default_missing_value("Q4K"), require_equals(true),
          value_name = "FORMAT")]
    pub quantize: Option<String>,

    // ── Media attachments (non-interactive / single-turn only) ────────────────
    /// Path to a WAV audio file to attach to the prompt (Gemma 4 audio models).
    #[arg(long)]
    pub audio: Option<std::path::PathBuf>,

    /// Path to an image file to attach to the prompt (vision models).
    /// Accepts JPEG, PNG, GIF, WebP, and other formats supported by the `image` crate.
    #[arg(long)]
    pub image: Option<std::path::PathBuf>,
}

impl RunArgs {
    /// Resolve the base URL for the server, using the same priority as
    /// `ollama run` uses for `OLLAMA_HOST`:
    ///   CLI flags > `INFERRS_HOST` env var > default.
    fn server_url(&self) -> String {
        // Detect whether the user explicitly overrode the defaults.
        let from_flags = self.host != "127.0.0.1" || self.port != DEFAULT_PORT;
        if from_flags {
            return format!("http://{}:{}", self.host, self.port);
        }

        if let Ok(env) = std::env::var("INFERRS_HOST") {
            let env = env.trim().to_string();
            if !env.is_empty() {
                // Accept plain `host:port` as well as `http://host:port`.
                return if env.starts_with("http://") || env.starts_with("https://") {
                    env
                } else {
                    format!("http://{env}")
                };
            }
        }

        format!("http://{}:{}", self.host, self.port)
    }
}

// ─── Sampling params bundle ──────────────────────────────────────────────────

/// Sampling parameters passed to generation functions.
/// Bundled into a struct to keep function argument counts within clippy limits.
#[derive(Debug, Clone, Copy)]
struct SamplingParams {
    temperature: f64,
    top_p: f64,
    top_k: usize,
    max_tokens: usize,
}

impl SamplingParams {
    fn from_args(args: &RunArgs) -> Self {
        Self {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            max_tokens: args.max_tokens,
        }
    }
}

// ─── Ollama API wire types ───────────────────────────────────────────────────

/// A single message in the `/api/chat` conversation history.
///
/// The `images` field carries base64-encoded image data and is part of the
/// standard Ollama wire format for vision requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiMessage {
    pub role: String,
    pub content: String,
    /// Base64-encoded image data (standard Ollama vision field).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
}

/// `POST /api/chat` request body.
#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ApiMessage],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<ChatOptions>,
}

#[derive(Debug, Serialize)]
struct ChatOptions {
    temperature: f64,
    top_p: f64,
    top_k: usize,
    num_predict: usize,
}

/// One NDJSON line from a streaming `/api/chat` response.
#[derive(Debug, Deserialize)]
struct ChatChunk {
    message: Option<ApiMessage>,
    done: bool,
}

// ─── OpenAI SSE types (used for /v1/chat/completions with audio) ─────────────

/// One SSE data line from a `/v1/chat/completions` streaming response.
#[derive(Debug, Deserialize)]
struct OaiChunkChoice {
    delta: OaiDelta,
}

#[derive(Debug, Deserialize)]
struct OaiDelta {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OaiChunk {
    choices: Vec<OaiChunkChoice>,
}

// ─── HTTP client helpers ─────────────────────────────────────────────────────

/// Probe the server with `HEAD /` (Ollama heartbeat endpoint).
/// Returns `true` if the server responded, `false` if connection was refused.
async fn heartbeat(client: &Client, base_url: &str) -> bool {
    let url = format!("{base_url}/");
    client.head(&url).send().await.is_ok()
}

/// Ensure the server is running, auto-starting it in the background if needed.
///
/// Mirrors `ollama run`'s `ensureServerRunning`: probe the heartbeat, and if
/// the server is not up, spawn `inferrs serve` (no arguments — the bare daemon,
/// just like `ollama serve`) using the same executable that is currently
/// running, then poll every 200 ms until it responds (up to 60 s).
async fn ensure_server_running(client: &Client, base_url: &str) -> Result<()> {
    if heartbeat(client, base_url).await {
        return Ok(());
    }

    // Server not running — spawn `inferrs serve` (no args) in the background.
    let exe = std::env::current_exe().context("Could not determine inferrs executable path")?;

    std::process::Command::new(&exe)
        .arg("serve")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("Failed to spawn `inferrs serve` ({})", exe.display()))?;

    // Poll until the server is ready (up to 60 s).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        if heartbeat(client, base_url).await {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            anyhow::bail!(
                "inferrs serve did not become ready within 60 s at {base_url}.\n\
                 Check its logs for errors."
            );
        }
    }
}

/// Send `POST /api/chat` with streaming enabled and print tokens to stdout as
/// they arrive.  Returns the full assembled response text.
///
/// Uses the Ollama NDJSON streaming format (`application/x-ndjson`).
async fn chat_stream(
    client: &Client,
    base_url: &str,
    model: &str,
    messages: &[ApiMessage],
    params: SamplingParams,
) -> Result<String> {
    let url = format!("{base_url}/api/chat");

    let body = ChatRequest {
        model,
        messages,
        stream: true,
        options: Some(ChatOptions {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            num_predict: params.max_tokens,
        }),
    };

    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("POST {url} failed"))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Server returned {status}: {text}");
    }

    drain_ndjson_stream(response).await
}

/// Send `POST /v1/chat/completions` (OpenAI SSE format) for requests that
/// carry an audio attachment.  The server's `/api/chat` Ollama endpoint does
/// not yet accept audio, so audio requests are routed through the OpenAI
/// endpoint which has full audio preprocessing support.
async fn audio_stream(
    client: &Client,
    base_url: &str,
    model: &str,
    text_messages: &[ApiMessage], // prior conversation (text only)
    prompt: &str,
    audio_b64: &str,
    params: SamplingParams,
) -> Result<String> {
    let url = format!("{base_url}/v1/chat/completions");

    // Rebuild prior turns as plain OpenAI messages, then append the audio turn.
    let mut messages: Vec<serde_json::Value> = text_messages
        .iter()
        .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
        .collect();

    messages.push(serde_json::json!({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
        ]
    }));

    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
    });

    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("POST {url} failed"))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Server returned {status}: {text}");
    }

    drain_openai_sse_stream(response).await
}

// ─── Stream drainers ─────────────────────────────────────────────────────────

/// Drain an Ollama NDJSON response stream, printing and collecting all tokens.
async fn drain_ndjson_stream(response: reqwest::Response) -> Result<String> {
    use futures::StreamExt;

    let mut full_text = String::new();
    let mut stdout = io::stdout();
    let mut byte_stream = response.bytes_stream();
    let mut line_buf = String::new();

    while let Some(chunk) = byte_stream.next().await {
        let chunk = chunk.context("Error reading response stream")?;
        let text = std::str::from_utf8(&chunk).context("Non-UTF-8 bytes in response")?;
        line_buf.push_str(text);

        while let Some(newline_pos) = line_buf.find('\n') {
            let line = line_buf[..newline_pos].trim().to_string();
            line_buf.drain(..=newline_pos);

            if line.is_empty() {
                continue;
            }

            let parsed: ChatChunk = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("Failed to parse NDJSON chunk: {e}: {line}");
                    continue;
                }
            };

            if let Some(msg) = parsed.message {
                full_text.push_str(&msg.content);
                print!("{}", msg.content);
                stdout.flush()?;
            }

            if parsed.done {
                return Ok(full_text);
            }
        }
    }

    Ok(full_text)
}

/// Drain an OpenAI SSE response stream (`text/event-stream`), printing and
/// collecting all content tokens.
async fn drain_openai_sse_stream(response: reqwest::Response) -> Result<String> {
    use futures::StreamExt;

    let mut full_text = String::new();
    let mut stdout = io::stdout();
    let mut byte_stream = response.bytes_stream();
    let mut line_buf = String::new();

    while let Some(chunk) = byte_stream.next().await {
        let chunk = chunk.context("Error reading SSE stream")?;
        let text = std::str::from_utf8(&chunk).context("Non-UTF-8 bytes in SSE response")?;
        line_buf.push_str(text);

        while let Some(newline_pos) = line_buf.find('\n') {
            let line = line_buf[..newline_pos].trim().to_string();
            line_buf.drain(..=newline_pos);

            // SSE lines are `data: <json>` or `data: [DONE]`
            let Some(json) = line.strip_prefix("data: ") else {
                continue;
            };
            if json == "[DONE]" {
                return Ok(full_text);
            }

            let parsed: OaiChunk = match serde_json::from_str(json) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("Failed to parse SSE chunk: {e}: {json}");
                    continue;
                }
            };

            for choice in parsed.choices {
                if let Some(content) = choice.delta.content {
                    full_text.push_str(&content);
                    print!("{content}");
                    stdout.flush()?;
                }
            }
        }
    }

    Ok(full_text)
}

// ─── Entry point ─────────────────────────────────────────────────────────────

/// Loading state shared between the background warm-up task and the REPL.
#[derive(Debug, Clone, PartialEq)]
enum LoadState {
    Loading,
    Ready,
    Failed(String),
}

/// Send an empty-prompt `POST /api/generate` request to trigger server-side
/// model loading without waiting for any generated tokens.  This is the same
/// "warm-up" pattern used by `ollama run` (`loadOrUnloadModel` with no prompt).
///
/// Model-loading options (dtype, device, revision, etc.) are passed in the
/// `options` field and read by the daemon when it spawns the worker process.
async fn warm_up_model(client: &Client, base_url: &str, args: &RunArgs) -> Result<()> {
    let url = format!("{base_url}/api/generate");

    // Build the options object with any non-default model-loading fields.
    let mut options = serde_json::Map::new();
    if args.revision != "main" {
        options.insert("revision".into(), args.revision.clone().into());
    }
    if args.dtype != "bf16" {
        options.insert("dtype".into(), args.dtype.clone().into());
    }
    if args.device != "auto" {
        options.insert("device".into(), args.device.clone().into());
    }
    if let Some(pa) = args.paged_attention {
        options.insert("paged_attention".into(), pa.into());
    }
    if args.turbo_quant.0 != Some(8) {
        options.insert("turbo_quant".into(), args.turbo_quant.to_string().into());
    }
    if let Some(ref q) = args.quantize {
        options.insert("quantize".into(), q.clone().into());
    }
    if let Some(ref f) = args.gguf_file {
        options.insert("gguf_file".into(), f.clone().into());
    }

    let mut body = serde_json::json!({
        "model": args.model,
        "prompt": "",   // empty prompt = load-only signal
        "stream": false,
    });
    if !options.is_empty() {
        body["options"] = serde_json::Value::Object(options);
    }

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("warm-up request failed")?;
    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("warm-up returned error: {text}");
    }
    Ok(())
}

/// Called from `main` (inside the Tokio runtime).
pub async fn run(args: RunArgs) -> Result<()> {
    let base_url = args.server_url();
    let client = Client::new();

    // Ensure the daemon is up, auto-starting it when needed.
    ensure_server_running(&client, &base_url).await?;

    // Kick off model loading immediately in a background task so the server
    // starts downloading/loading weights right now, without blocking the REPL.
    let (load_tx, load_rx) = tokio::sync::watch::channel(LoadState::Loading);
    {
        let client2 = client.clone();
        let base_url2 = base_url.clone();
        let args2 = args.clone();
        tokio::spawn(async move {
            let result = warm_up_model(&client2, &base_url2, &args2).await;
            let state = match result {
                Ok(()) => LoadState::Ready,
                Err(e) => LoadState::Failed(e.to_string()),
            };
            // Ignore send error — REPL may have already exited.
            let _ = load_tx.send(state);
        });
    }

    // Build initial message history (optional system prompt).
    let mut messages: Vec<ApiMessage> = Vec::new();
    if let Some(sys) = &args.system {
        messages.push(ApiMessage {
            role: "system".to_string(),
            content: sys.clone(),
            images: vec![],
        });
    }

    // ── Non-interactive: single prompt then exit ──────────────────────────────

    if let Some(prompt) = args.prompt.clone() {
        // Wait for the model to be ready before sending the single prompt.
        wait_for_load(&mut load_rx.clone()).await?;

        // Audio attachment — route through /v1/chat/completions.
        if let Some(audio_path) = &args.audio {
            let raw = std::fs::read(audio_path)
                .with_context(|| format!("Could not read audio file: {}", audio_path.display()))?;
            let audio_b64 = base64_encode(&raw);
            audio_stream(
                &client,
                &base_url,
                &args.model,
                &messages,
                &prompt,
                &audio_b64,
                SamplingParams::from_args(&args),
            )
            .await?;
            println!();
            return Ok(());
        }

        // Image attachment — include via the standard Ollama `images` field.
        let images = if let Some(image_path) = &args.image {
            let raw = std::fs::read(image_path)
                .with_context(|| format!("Could not read image file: {}", image_path.display()))?;
            vec![base64_encode(&raw)]
        } else {
            vec![]
        };

        messages.push(ApiMessage {
            role: "user".to_string(),
            content: prompt,
            images,
        });
        chat_stream(
            &client,
            &base_url,
            &args.model,
            &messages,
            SamplingParams::from_args(&args),
        )
        .await?;
        println!();
        return Ok(());
    }

    // Interactive REPL — enter immediately; model loads in the background.
    repl(client, base_url, load_rx, args).await
}

/// Wait for the background load to finish, returning an error if it failed.
async fn wait_for_load(rx: &mut tokio::sync::watch::Receiver<LoadState>) -> Result<()> {
    loop {
        {
            let state = rx.borrow();
            match &*state {
                LoadState::Ready => return Ok(()),
                LoadState::Failed(e) => anyhow::bail!("Model failed to load: {e}"),
                LoadState::Loading => {}
            }
        }
        // Wait for the next state change from the background task.
        if rx.changed().await.is_err() {
            // Sender dropped without sending Ready/Failed — treat as ready
            // (server may have responded before the watch was set up).
            return Ok(());
        }
    }
}

/// Encode raw bytes as standard base64 (no line breaks).
fn base64_encode(data: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(data)
}

// ─── Interactive REPL ────────────────────────────────────────────────────────

/// Multiline input state.
enum MultilineState {
    None,
    /// Accumulating a user prompt block opened by `"""`
    Prompt,
}

async fn repl(
    client: Client,
    base_url: String,
    mut load_rx: tokio::sync::watch::Receiver<LoadState>,
    args: RunArgs,
) -> Result<()> {
    let mut messages: Vec<ApiMessage> = Vec::new();
    if let Some(sys) = &args.system {
        messages.push(ApiMessage {
            role: "system".to_string(),
            content: sys.clone(),
            images: vec![],
        });
    }

    let mut temperature = args.temperature;
    let mut top_p = args.top_p;
    let mut top_k = args.top_k;
    let mut max_tokens = args.max_tokens;

    let mut multiline = MultilineState::None;
    let mut buf = String::new();

    loop {
        // Show a loading indicator in the prompt while the model is still loading.
        let prompt_str = match multiline {
            MultilineState::Prompt => ". ",
            MultilineState::None => "> ",
        };
        print!("{prompt_str}");
        io::stdout().flush()?;

        let line = match read_line()? {
            ReadResult::Line(l) => l,
            ReadResult::Interrupt => {
                println!();
                buf.clear();
                multiline = MultilineState::None;
                continue;
            }
            ReadResult::Eof => {
                println!();
                break;
            }
        };

        match multiline {
            MultilineState::Prompt => {
                if let Some(before) = line.strip_suffix("\"\"\"") {
                    buf.push_str(before);
                    multiline = MultilineState::None;
                } else {
                    buf.push_str(&line);
                    buf.push('\n');
                    continue;
                }
            }
            MultilineState::None => {
                if let Some(rest) = line.strip_prefix("\"\"\"") {
                    if let Some(inner) = rest.strip_suffix("\"\"\"") {
                        buf = inner.to_string();
                    } else {
                        buf = rest.to_string();
                        if !buf.is_empty() {
                            buf.push('\n');
                        }
                        multiline = MultilineState::Prompt;
                        continue;
                    }
                } else {
                    buf = line.clone();
                }

                let trimmed = buf.trim().to_string();
                if trimmed.starts_with('/') {
                    handle_command(
                        &trimmed,
                        &mut messages,
                        &mut temperature,
                        &mut top_p,
                        &mut top_k,
                        &mut max_tokens,
                    );
                    buf.clear();
                    continue;
                }

                if trimmed.is_empty() {
                    buf.clear();
                    continue;
                }
            }
        }

        let user_content = buf.trim().to_string();
        buf.clear();

        // If the model is still loading, wait for it before sending the request.
        // Print a notice only if we actually have to wait.
        if matches!(*load_rx.borrow(), LoadState::Loading) {
            if let Err(e) = wait_for_load(&mut load_rx).await {
                eprintln!("{e}");
                continue;
            }
        }

        messages.push(ApiMessage {
            role: "user".to_string(),
            content: user_content,
            images: vec![],
        });

        let assistant_text = match chat_stream(
            &client,
            &base_url,
            &args.model,
            &messages,
            SamplingParams {
                temperature,
                top_p,
                top_k,
                max_tokens,
            },
        )
        .await
        {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Generation error: {e}");
                messages.pop();
                continue;
            }
        };

        println!();

        messages.push(ApiMessage {
            role: "assistant".to_string(),
            content: assistant_text,
            images: vec![],
        });
    }

    Ok(())
}

// ─── Slash command handler ────────────────────────────────────────────────────

fn handle_command(
    cmd: &str,
    messages: &mut Vec<ApiMessage>,
    temperature: &mut f64,
    top_p: &mut f64,
    top_k: &mut usize,
    max_tokens: &mut usize,
) {
    let parts: Vec<&str> = cmd.splitn(3, ' ').collect();
    match parts[0] {
        "/bye" | "/exit" | "/quit" => {
            std::process::exit(0);
        }
        "/clear" => {
            let sys: Vec<ApiMessage> = messages.drain(..).filter(|m| m.role == "system").collect();
            *messages = sys;
            println!("Conversation cleared.");
        }
        "/set" if parts.len() >= 3 => match parts[1] {
            "system" => {
                messages.retain(|m| m.role != "system");
                messages.insert(
                    0,
                    ApiMessage {
                        role: "system".to_string(),
                        content: parts[2].to_string(),
                        images: vec![],
                    },
                );
                println!("System prompt set.");
            }
            "temperature" => {
                if let Ok(v) = parts[2].parse::<f64>() {
                    *temperature = v;
                    println!("temperature set to {v}");
                } else {
                    println!("Invalid value: {}", parts[2]);
                }
            }
            "top_p" => {
                if let Ok(v) = parts[2].parse::<f64>() {
                    *top_p = v;
                    println!("top_p set to {v}");
                } else {
                    println!("Invalid value: {}", parts[2]);
                }
            }
            "top_k" => {
                if let Ok(v) = parts[2].parse::<usize>() {
                    *top_k = v;
                    println!("top_k set to {v}");
                } else {
                    println!("Invalid value: {}", parts[2]);
                }
            }
            "num_predict" | "max_tokens" => {
                if let Ok(v) = parts[2].parse::<usize>() {
                    *max_tokens = v;
                    println!("max_tokens set to {v}");
                } else {
                    println!("Invalid value: {}", parts[2]);
                }
            }
            _ => println!("Unknown /set option: {}", parts[1]),
        },
        "/show" if parts.len() >= 2 => match parts[1] {
            "history" => {
                for (i, m) in messages.iter().enumerate() {
                    println!("[{}] {}: {}", i, m.role, m.content);
                }
            }
            "params" => {
                println!(
                    "temperature={temperature} top_p={top_p} top_k={top_k} max_tokens={max_tokens}"
                );
            }
            _ => println!("Unknown /show option: {}", parts[1]),
        },
        "/help" | "/?" => {
            println!("Commands:");
            println!("  /bye, /exit, /quit          Exit the REPL");
            println!("  /clear                       Clear conversation history");
            println!("  /set system <text>           Set a system prompt");
            println!("  /set temperature <n>         Set sampling temperature");
            println!("  /set top_p <n>               Set nucleus sampling threshold");
            println!("  /set top_k <n>               Set top-k sampling");
            println!("  /set max_tokens <n>          Set max tokens per response");
            println!("  /show history                Print conversation history");
            println!("  /show params                 Print sampling parameters");
            println!("  /help, /?                    Show this help");
            println!();
            println!("Keyboard shortcuts:");
            println!("  Ctrl+D / EOF                 Exit");
            println!("  Ctrl+C                       Cancel current input");
            println!();
            println!("Multiline input:");
            println!("  Start a message with \"\"\" to enter multi-line mode.");
            println!("  End with \"\"\" on its own line to send.");
        }
        other => println!("Unknown command: {other}"),
    }
}

// ─── Raw-mode line reader ────────────────────────────────────────────────────

enum ReadResult {
    Line(String),
    Interrupt,
    Eof,
}

/// Read a single line from stdin using crossterm raw mode.
fn read_line() -> Result<ReadResult> {
    let mut buf: Vec<char> = Vec::new();
    let mut cursor_pos: usize = 0;

    terminal::enable_raw_mode()?;
    let _guard = RawModeGuard;

    let mut stdout = io::stdout();

    loop {
        let ev = event::read()?;

        match ev {
            Event::Key(KeyEvent {
                code, modifiers, ..
            }) => match code {
                KeyCode::Enter => {
                    execute!(stdout, Print("\r\n"))?;
                    let line: String = buf.iter().collect();
                    return Ok(ReadResult::Line(line));
                }

                KeyCode::Char('d') if modifiers.contains(KeyModifiers::CONTROL) => {
                    if buf.is_empty() {
                        execute!(stdout, Print("\r\n"))?;
                        return Ok(ReadResult::Eof);
                    }
                    if cursor_pos < buf.len() {
                        buf.remove(cursor_pos);
                        redraw_from_cursor(&mut stdout, &buf, cursor_pos)?;
                    }
                }

                KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                    execute!(stdout, Print("^C\r\n"))?;
                    return Ok(ReadResult::Interrupt);
                }

                KeyCode::Backspace => {
                    if cursor_pos > 0 {
                        cursor_pos -= 1;
                        buf.remove(cursor_pos);
                        execute!(stdout, cursor::MoveLeft(1))?;
                        redraw_from_cursor(&mut stdout, &buf, cursor_pos)?;
                    }
                }

                KeyCode::Delete => {
                    if cursor_pos < buf.len() {
                        buf.remove(cursor_pos);
                        redraw_from_cursor(&mut stdout, &buf, cursor_pos)?;
                    }
                }

                KeyCode::Left => {
                    if cursor_pos > 0 {
                        cursor_pos -= 1;
                        execute!(stdout, cursor::MoveLeft(1))?;
                    }
                }

                KeyCode::Right => {
                    if cursor_pos < buf.len() {
                        cursor_pos += 1;
                        execute!(stdout, cursor::MoveRight(1))?;
                    }
                }

                KeyCode::Home => {
                    if cursor_pos > 0 {
                        execute!(stdout, cursor::MoveLeft(cursor_pos as u16))?;
                        cursor_pos = 0;
                    }
                }
                KeyCode::Char('a') if modifiers.contains(KeyModifiers::CONTROL) => {
                    if cursor_pos > 0 {
                        execute!(stdout, cursor::MoveLeft(cursor_pos as u16))?;
                        cursor_pos = 0;
                    }
                }

                KeyCode::End => {
                    let remaining = buf.len() - cursor_pos;
                    if remaining > 0 {
                        execute!(stdout, cursor::MoveRight(remaining as u16))?;
                        cursor_pos = buf.len();
                    }
                }
                KeyCode::Char('e') if modifiers.contains(KeyModifiers::CONTROL) => {
                    let remaining = buf.len() - cursor_pos;
                    if remaining > 0 {
                        execute!(stdout, cursor::MoveRight(remaining as u16))?;
                        cursor_pos = buf.len();
                    }
                }

                KeyCode::Char('k') if modifiers.contains(KeyModifiers::CONTROL) => {
                    buf.truncate(cursor_pos);
                    execute!(stdout, terminal::Clear(ClearType::UntilNewLine))?;
                }

                KeyCode::Char('u') if modifiers.contains(KeyModifiers::CONTROL) => {
                    if cursor_pos > 0 {
                        execute!(stdout, cursor::MoveLeft(cursor_pos as u16))?;
                        buf.drain(..cursor_pos);
                        cursor_pos = 0;
                        redraw_from_cursor(&mut stdout, &buf, 0)?;
                    }
                }

                KeyCode::Char('w') if modifiers.contains(KeyModifiers::CONTROL) => {
                    if cursor_pos > 0 {
                        let mut end = cursor_pos;
                        while end > 0 && buf[end - 1] == ' ' {
                            end -= 1;
                        }
                        while end > 0 && buf[end - 1] != ' ' {
                            end -= 1;
                        }
                        let deleted = cursor_pos - end;
                        execute!(stdout, cursor::MoveLeft(deleted as u16))?;
                        buf.drain(end..cursor_pos);
                        cursor_pos = end;
                        redraw_from_cursor(&mut stdout, &buf, cursor_pos)?;
                    }
                }

                KeyCode::Char(c) => {
                    buf.insert(cursor_pos, c);
                    cursor_pos += 1;
                    execute!(stdout, Print(c))?;
                    if cursor_pos < buf.len() {
                        redraw_from_cursor(&mut stdout, &buf, cursor_pos)?;
                    }
                }

                KeyCode::Tab => {
                    for _ in 0..4 {
                        buf.insert(cursor_pos, ' ');
                        cursor_pos += 1;
                    }
                    execute!(stdout, Print("    "))?;
                    if cursor_pos < buf.len() {
                        redraw_from_cursor(&mut stdout, &buf, cursor_pos)?;
                    }
                }

                _ => {}
            },

            Event::Paste(text) => {
                for c in text.chars() {
                    if c == '\n' || c == '\r' {
                        buf.insert(cursor_pos, '\n');
                        cursor_pos += 1;
                    } else {
                        buf.insert(cursor_pos, c);
                        cursor_pos += 1;
                    }
                }
                let before: String = buf[..cursor_pos].iter().collect();
                execute!(
                    stdout,
                    cursor::MoveLeft(cursor_pos as u16),
                    Print(&before),
                    terminal::Clear(ClearType::UntilNewLine)
                )?;
            }

            _ => {}
        }
    }
}

fn redraw_from_cursor(stdout: &mut io::Stdout, buf: &[char], from: usize) -> Result<()> {
    let tail: String = buf[from..].iter().collect();
    let tail_len = tail.chars().count();
    execute!(
        stdout,
        Print(&tail),
        terminal::Clear(ClearType::UntilNewLine),
    )?;
    if tail_len > 0 {
        execute!(stdout, cursor::MoveLeft(tail_len as u16))?;
    }
    Ok(())
}

/// RAII guard that restores normal terminal mode when dropped.
struct RawModeGuard;

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
    }
}

// ─── Display helpers ──────────────────────────────────────────────────────────

/// Print `text` in a muted colour for secondary/status output.
#[allow(dead_code)]
fn print_dim(text: &str) {
    let mut stdout = io::stdout();
    execute!(
        stdout,
        SetForegroundColor(Color::DarkGrey),
        Print(text),
        ResetColor,
    )
    .ok();
}

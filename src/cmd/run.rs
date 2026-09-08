//! `llmman run` — interactive chat or one-shot prompt.
//!
//! Interactive mode uses a raw-mode readline ported directly from ollama's
//! readline package (readline/readline.go, readline/term.go). Paste
//! detection mirrors ollama exactly:
//!
//!   // ollama (Go)
//!   if i.Terminal.reader.Buffered() > 0 { draining = true }
//!
//!   // llmman (Rust)
//!   if !reader.buffer().is_empty() { draining = true; }
//!
//! While draining a paste, '\n' submits the line like Enter; otherwise
//! it's Ctrl-J multiline, same as ollama.
//!
//! Streamed responses are ported from ollama's `cmd/cmd.go`
//! (`generate`/`chat`, `displayResponse`, `thinkingOutputOpeningText`/
//! `...ClosingText`): word-wrapped at the terminal width, "Thinking..."
//! in dim grey/bold ANSI, sent over an async reqwest client racing
//! `tokio::signal::ctrl_c()` so Ctrl-C cancels only the in-flight turn
//! (mirrors ollama's `context.WithCancel` + `signal.Notify`) instead of
//! killing the process. In the interactive REPL, raw mode is released
//! for the duration of each response and re-entered before the next
//! prompt — mirrors `readline.Instance.Readline()`'s own enter/`defer`
//! restore — which is why Ctrl-C while typing is read as byte 3 (ISIG
//! off) but Ctrl-C during a response is a real SIGINT (ISIG back on).
//!
//! Image attachments are ported from ollama's `cmd/interactive.go`: when
//! `/api/show` reports `vision`, image/audio paths in the prompt are
//! read, base64-encoded onto the message's `images`, and removed from
//! the text.

use std::io::{self, IsTerminal, Read, Seek, Write};
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::Context;
use base64::Engine as _;
use clap::Args;
use futures::TryStreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;

use crate::daemon;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Args, Debug)]
pub struct RunArgs {
    #[arg(value_name = "MODEL")]
    pub model: String,
    /// Chat with MODEL at this hosted provider (openai, anthropic,
    /// openrouter, …) instead of locally. The request still goes through
    /// `llmman serve`, which forwards it upstream; MODEL is the
    /// provider's own model id. See `llmman providers`.
    #[arg(long, short = 'p', value_name = "PROVIDER")]
    pub provider: Option<String>,
    /// Send requests too large for the local MODEL to this provider
    /// instead (openai, anthropic, openrouter, ...). Needs
    /// --overflow-model; not combinable with --provider.
    #[arg(long, value_name = "PROVIDER")]
    pub overflow_provider: Option<String>,
    /// The --overflow-provider model that serves requests too large for
    /// the local MODEL. Everything that fits stays on this machine.
    #[arg(long, value_name = "MODEL")]
    pub overflow_model: Option<String>,
    /// Forwarded as Ollama's own top-level `think` field on every request
    /// this sends (see cmd::serve's think_to_chat_template_kwargs) —
    /// `--think false` disables a reasoning model's thinking block
    /// entirely, `--think true` forces it on. Omitted (leaving the
    /// model's own template default in effect) if not passed at all.
    #[arg(long)]
    pub think: Option<bool>,
    /// Forwarded as `options.num_predict` (Ollama's own name for
    /// llama-server's `max_tokens` — see opt_u32 in cmd::serve) on every
    /// request this sends: a hard ceiling on how many tokens a single
    /// reply may generate, regardless of *why* it might otherwise run
    /// away (a real, no-stopping-condition-hit degenerate loop, observed
    /// directly with qwen3.5:0.8b even with `--think false` and a
    /// repeat_penalty already in effect — see this repo's own git history
    /// — is not reliably preventable any other way). Omitted (no ceiling
    /// at all, matching Ollama's own num_predict default of -1) if not
    /// passed.
    #[arg(long)]
    pub num_predict: Option<u32>,
    /// Mirrors ollama's own `run --nowordwrap` — opts out of the
    /// terminal-width word wrap in `wrap_write` below.
    #[arg(long)]
    pub nowordwrap: bool,
    /// Image generation models only (see `crate::imagegen`): output
    /// width in pixels. Hidden, like ollama's own image flags.
    #[arg(long, hide = true, default_value_t = 0)]
    pub width: u32,
    /// Image generation models only: output height in pixels.
    #[arg(long, hide = true, default_value_t = 0)]
    pub height: u32,
    /// Image generation models only: denoising steps (0 = model default).
    #[arg(long, hide = true, default_value_t = 0)]
    pub steps: u32,
    /// Image generation models only: random seed.
    #[arg(long, hide = true)]
    pub seed: Option<u32>,
    /// Image generation models only: guidance scale (>1 enables --negative).
    #[arg(long, hide = true, default_value_t = 0.0)]
    pub cfg_scale: f32,
    /// Image generation models only: negative prompt.
    #[arg(long, hide = true, default_value = "")]
    pub negative: String,
    /// Media generation models only: generate a video (mp4) instead of
    /// an image.
    #[arg(long, hide = true, conflicts_with = "audio")]
    pub video: bool,
    /// Media generation models only: generate audio (wav) instead of an
    /// image.
    #[arg(long, hide = true)]
    pub audio: bool,
    /// Media generation models only: length of a --video / --audio clip.
    #[arg(long, hide = true, default_value_t = 0.0)]
    pub seconds: f32,
    /// Media generation models only: conditioning image (PNG/JPEG) for
    /// image-to-video, or an action run's first frame.
    #[arg(long, hide = true, value_name = "PATH")]
    pub image: Option<std::path::PathBuf>,
    /// Media generation models only: the conditioning video of a
    /// video-to-video or action run.
    #[arg(long, hide = true, value_name = "PATH")]
    pub input_video: Option<std::path::PathBuf>,
    /// Video-to-video: latent frames kept from --input-video (default 0,1).
    #[arg(long, hide = true, value_name = "IDX,IDX", default_value = "")]
    pub condition_frames: String,
    /// Video-to-video: take the conditioning frames from the end of the clip.
    #[arg(long, hide = true)]
    pub condition_keep_last: bool,
    /// Cosmos3 action run: forward_dynamics, inverse_dynamics or policy.
    #[arg(long, hide = true, value_name = "MODE")]
    pub action_mode: Option<String>,
    /// Cosmos3 action run: embodiment domain (bridge_orig_lerobot, av, ...).
    #[arg(long, hide = true, default_value = "")]
    pub action_domain: String,
    /// Cosmos3 action run: JSON file with the [T][D] actions (forward dynamics).
    #[arg(long, hide = true, value_name = "PATH")]
    pub actions: Option<std::path::PathBuf>,
    /// Cosmos3 action run: action transitions in the chunk.
    #[arg(long, hide = true, default_value_t = 0)]
    pub action_chunk: u32,
    /// Cosmos3 action run: conditioning canvas tier (256, 480, 704, 720).
    #[arg(long, hide = true, default_value_t = 480)]
    pub action_tier: u32,
    /// Cosmos3 action run: camera viewpoint (ego_view, third_person_view, ...).
    #[arg(long, hide = true, default_value = "ego_view")]
    pub action_view: String,
    /// Media generation models only: frames per second of the clip.
    #[arg(long, hide = true, default_value_t = 0.0)]
    pub fps: f32,
    /// Cosmos3: scheduler flow shift (the reference action runs use 10).
    #[arg(long, hide = true)]
    pub flow_shift: Option<f32>,
    #[arg(
        value_name = "PROMPT",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    pub prompt: Vec<String>,
}

/// Per-request knobs threaded through unchanged from `RunArgs`.
#[derive(Debug, Clone, Copy)]
struct ChatOptions<'a> {
    think: Option<bool>,
    num_predict: Option<u32>,
    /// Inverse of `RunArgs::nowordwrap`, matching ollama's
    /// `runOptions.WordWrap` (defaults `true`).
    word_wrap: bool,
    /// Provider key to send with every request under `--provider` (see
    /// `provider_model`), `None` for a local model. Applied as a default
    /// header, not a body field — see `chat_client`.
    api_key: Option<&'a str>,
    /// Ollama's `runOptions.MultiModal`, from `/api/show`'s capabilities:
    /// only then is a prompt scanned for image paths.
    multimodal: bool,
}

impl Default for ChatOptions<'_> {
    fn default() -> Self {
        Self {
            think: None,
            num_predict: None,
            word_wrap: true,
            api_key: None,
            multimodal: false,
        }
    }
}

pub fn run(args: &RunArgs) -> anyhow::Result<()> {
    let provider = crate::providers::provider_flag(args.provider.as_deref())?;
    let overflow = crate::hybrid::overflow_flags(
        args.overflow_provider.as_deref(),
        args.overflow_model.as_deref(),
        provider,
    )?;
    let prompt = args.prompt.join(" ");

    // resolve_ollama_api, not resolve: `llmman run` is an /api/chat client
    // (see chat_submit/run_interactive_tty below), so a bare name must
    // resolve the same way it would if requested directly over the Ollama
    // API — otherwise a name resolved here, then handed to every /api/chat
    // request this sends, is no longer "bare" by the time ensure_model
    // resolves it server-side (it already has a "/" and a "."), so the
    // docker.io/ai/ default never fires and this silently falls back to
    // hf.co/<name> instead.
    //
    // Not applied under `--provider`: that MODEL names a model on someone
    // else's servers, so no shortname aliasing, tag defaulting, store
    // lookup or pull applies — the same call ensure_model makes
    // server-side. Resolved before the daemon starts, so a malformed
    // local reference still fails without one.
    let route = match provider {
        Some(provider) => Route::Provider(provider),
        None => Route::Local(crate::shortnames::resolve_ollama_api(&args.model)?),
    };

    // Starts `llmman serve` detached, left running indefinitely, if one
    // isn't already reachable — the same shared helper pull/push/launch
    // use (see daemon::ensure_server's doc comment for why stdio is
    // redirected there: without it, this command would hang forever
    // waiting for the (never-exiting) daemon's inherited stdout/stderr
    // pipes to close). No preload model is passed: the resulting daemon
    // is a plain `llmman serve` with no model argument, so it's shared
    // cleanly across every future `run`/`pull`/`push`/`launch` in this
    // session rather than looking like it's dedicated to whatever model
    // happened to start it first.
    //
    // A provider-routed model has nothing to preload or pull, but needs
    // the daemon sooner still: it owns the catalog `provider_model`
    // validates against, and it forwards the chat upstream.
    crate::daemon::ensure_server("")?;

    let (model, api_key, multimodal) = match route {
        Route::Provider(provider) => {
            let (model, key) = provider_model(provider, &args.model)?;
            // /api/show has no capabilities for a provider-routed model.
            (model, key, false)
        }
        Route::Local(model) => {
            // Fail fast on a bad/unresolvable reference — mirrors ollama's
            // RunHandler, which resolves (Show, falling back to Pull) the
            // model before ever showing its interactive prompt. Without
            // this, an error like an invalid `hf.co/...` reference wouldn't
            // surface until the first message was submitted to /api/chat,
            // well after the `> ` prompt had already been shown and read
            // from. The same Show also answers ollama's `opts.MultiModal`.
            let info = crate::daemon::ensure_model_pulled(&model)?;
            // An image generation model has no chat at all: every prompt
            // is a picture (ollama's `CapabilityImage` branch of
            // RunHandler, before it dropped image generation).
            if info.image() {
                let interactive =
                    prompt.is_empty() && io::stdin().is_terminal() && io::stdout().is_terminal();
                let opts = crate::imagegen::ImageOptions {
                    media: if args.video {
                        crate::imagegen::Media::Video
                    } else if args.audio {
                        crate::imagegen::Media::Audio
                    } else {
                        crate::imagegen::Media::Image
                    },
                    width: args.width,
                    height: args.height,
                    steps: args.steps,
                    seed: args.seed,
                    cfg_scale: args.cfg_scale,
                    negative: args.negative.clone(),
                    seconds: args.seconds,
                    fps: args.fps,
                    flow_shift: args.flow_shift,
                    image: args.image.clone(),
                    input_video: args.input_video.clone(),
                    condition_frames: args.condition_frames.clone(),
                    condition_keep_last: args.condition_keep_last,
                    action: args
                        .action_mode
                        .as_ref()
                        .map(|mode| crate::imagegen::ActionOptions {
                            mode: mode.clone(),
                            domain: args.action_domain.clone(),
                            actions: args.actions.clone(),
                            chunk: args.action_chunk,
                            tier: args.action_tier,
                            view: args.action_view.clone(),
                        }),
                };
                return crate::imagegen::run(&model, &prompt, interactive, opts);
            }
            let multimodal = info.multimodal();
            match overflow {
                // The hosted half is validated and keyed as a bare
                // --provider model is, then paired with the local model.
                Some((provider, hosted)) => {
                    let (remote, key) = provider_model(provider, hosted)?;
                    (
                        crate::hybrid::pair_with_local(&model, &remote)?,
                        key,
                        multimodal,
                    )
                }
                None => (model, None, multimodal),
            }
        }
    };

    // Mirrors ollama's RunHandler: interactive needs *both* ends of the
    // terminal, not just stdin — otherwise a redirected stdout still gets
    // raw-moded and starts emitting ANSI escapes into it.
    let interactive = prompt.is_empty() && io::stdin().is_terminal() && io::stdout().is_terminal();
    let opts = ChatOptions {
        think: args.think,
        num_predict: args.num_predict,
        word_wrap: !args.nowordwrap,
        api_key: api_key.as_deref(),
        multimodal,
    };

    if interactive {
        run_interactive_tty(&model, opts)
    } else {
        let p = if prompt.is_empty() {
            let mut s = String::new();
            io::stdin().read_line(&mut s)?;
            s.trim().to_string()
        } else {
            prompt
        };
        if !p.is_empty() {
            // One-shot "conversation" over /api/chat, same helper as
            // interactive mode. A fresh tokio runtime, not
            // reqwest::blocking, so Ctrl-C can race the response via
            // `tokio::select!` in `chat_submit` — see that fn's doc comment.
            let rt = tokio::runtime::Runtime::new().context("start tokio runtime")?;
            let client = chat_client(opts.api_key)?;
            chat_submit(&rt, &client, &model, &mut Vec::new(), p, opts)?;
        }
        Ok(())
    }
}

/// Where this chat goes: a local reference, already resolved (see `run`),
/// or a provider id `provider_model` still has to validate.
enum Route<'a> {
    Local(String),
    Provider(&'a str),
}

/// Validates `--provider`/MODEL against the daemon's catalog, returning
/// the reference the daemon routes on (see
/// [`crate::providers::REMOTE_PREFIX`]) and the key to send with each
/// request, if this shell has one.
///
/// Mirrors `cmd::launch`'s `resolve_provider_model` minus the
/// integrations: the key travels per request in an `Authorization` header
/// (see `client_api_key` in cmd::serve), never to disk.
fn provider_model(provider: &str, model: &str) -> anyhow::Result<(String, Option<String>)> {
    // Same rule as `launch --provider` (see check_provider_supported in
    // cmd::launch): the daemon has no TLS, so a key sent to one elsewhere
    // on the network would cross it in cleartext. A wildcard bind is
    // fine — that hop is still loopback.
    anyhow::ensure!(
        crate::daemon::connects_over_loopback(),
        "--provider needs a local llmman serve: LLMMAN_HOST points at {}, and the provider \
         key would cross the network in cleartext.\n\
         Export the key where that daemon runs, and run llmman there.",
        crate::daemon::server()
    );

    let entry = daemon::provider(provider)?;
    let model = model.trim();
    anyhow::ensure!(
        !model.is_empty(),
        "--provider {provider} also needs a model\n\n{}",
        crate::providers::example_models(&entry.name, &entry.model_ids())
    );

    entry.warn_unlisted(model);

    // Naming where the missing key goes beats a 401 mid-conversation —
    // unless the daemon has the key, in which case it spends its own.
    let key = entry.api_key();
    anyhow::ensure!(
        key.is_some() || entry.daemon_key_usable(),
        "no API key for {} — {}",
        entry.name,
        crate::providers::key_hint(&entry.id, &entry.key_env)
    );
    Ok((crate::providers::format_remote_ref(provider, model), key))
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Msg {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    /// Ollama's `api.Message.Images`: standard base64, one per image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
}

impl Msg {
    fn text(role: &str, content: String) -> Self {
        Self {
            role: role.into(),
            content,
            thinking: None,
            images: None,
        }
    }
}

/// Async, not `reqwest::blocking`, so a chat turn's response can be raced
/// against `tokio::signal::ctrl_c()` in `chat_submit` — no `.timeout()`
/// needed either, unlike the blocking client's own 30s default.
///
/// A `--provider` key (see `provider_model`) rides along as a default
/// `Authorization` header — this client has one destination, and that
/// header is what `client_api_key` in cmd::serve reads. Sensitive, so a
/// `Debug`-formatted client or request cannot print it.
fn chat_client(api_key: Option<&str>) -> anyhow::Result<Client> {
    let mut builder = Client::builder();
    if let Some(key) = api_key {
        let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
            .context("provider API key is not a valid HTTP header value")?;
        value.set_sensitive(true);
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::AUTHORIZATION, value);
        builder = builder.default_headers(headers);
    }
    builder.build().context("build http client")
}

#[derive(Serialize)]
struct ChatReq<'a> {
    model: &'a str,
    messages: &'a [Msg],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<ChatReqOptions>,
}

#[derive(Serialize)]
struct ChatReqOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
}

#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    message: Option<Msg>,
    #[serde(default)]
    done: bool,
}

// ---------------------------------------------------------------------------
// Image attachments — ported from ollama's cmd/interactive.go
// ---------------------------------------------------------------------------

/// Ollama's `normalizeFilePath`: undoes shell escaping from a drag-and-drop
/// (`My\ Photo.jpg` → `My Photo.jpg`). Same table and order as its
/// `strings.NewReplacer`.
fn normalize_file_path(fp: &str) -> String {
    const PAIRS: &[(&str, &str)] = &[
        ("\\ ", " "),
        ("\\(", "("),
        ("\\)", ")"),
        ("\\[", "["),
        ("\\]", "]"),
        ("\\{", "{"),
        ("\\}", "}"),
        ("\\$", "$"),
        ("\\&", "&"),
        ("\\;", ";"),
        ("\\'", "'"),
        ("\\\\", "\\"),
        ("\\*", "*"),
        ("\\?", "?"),
        ("\\~", "~"),
    ];
    let mut out = String::with_capacity(fp.len());
    let mut rest = fp;
    'outer: while !rest.is_empty() {
        for (from, to) in PAIRS {
            if let Some(tail) = rest.strip_prefix(from) {
                out.push_str(to);
                rest = tail;
                continue 'outer;
            }
        }
        let ch = rest.chars().next().expect("non-empty");
        out.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    out
}

/// Ollama's `extractFileNames`, regex verbatim. Over-matches on purpose;
/// `extract_file_data` drops anything that isn't a real file.
fn extract_file_names(input: &str) -> Vec<&str> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(r"(?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(?i:jpg|jpeg|png|webp|wav)\b")
            .expect("valid regex")
    });
    re.find_iter(input).map(|m| m.as_str()).collect()
}

/// Ollama's `extractFileData`: base64-encodes every existing image/audio
/// file named in `input` and strips the paths (and surrounding single
/// quotes) from the text. A missing path is left in place; an existing
/// file of an unsupported type is an error.
fn extract_file_data(input: &str) -> anyhow::Result<(String, Vec<String>)> {
    let mut out = input.to_string();
    let mut imgs = Vec::new();

    for fp in extract_file_names(input) {
        let nfp = normalize_file_path(fp);
        let data = match get_image_data(&nfp) {
            Ok(data) => data,
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) => {
                eprintln!("Couldn't process file: {:?}", e.to_string());
                return Err(e.into());
            }
        };
        let ext = Path::new(&nfp)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();
        match ext.as_str() {
            "wav" => eprintln!("Added audio '{nfp}'"),
            _ => eprintln!("Added image '{nfp}'"),
        }
        out = out.replace(&format!("'{nfp}'"), "");
        out = out.replace(&format!("'{fp}'"), "");
        out = out.replace(fp, "");
        imgs.push(base64::engine::general_purpose::STANDARD.encode(&data));
    }
    Ok((out.trim().to_string(), imgs))
}

/// Ollama's `getImageData`: sniff the first 512 bytes, cap at 100MB, then
/// read exactly the stat'd size (its `io.ReadFull` into a sized buffer).
fn get_image_data(file_path: &str) -> io::Result<Vec<u8>> {
    let mut file = std::fs::File::open(file_path)?;

    let mut buf = [0u8; 512];
    if file.read(&mut buf)? == 0 {
        // Go's Read on an empty file is io.EOF: an error, not a skip.
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "EOF"));
    }

    let content_type = detect_content_type(&buf);
    const ALLOWED: &[&str] = &[
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "audio/wave",
    ];
    if !ALLOWED.contains(&content_type) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid file type: {content_type}"),
        ));
    }

    let size = file.metadata()?.len();
    const MAX_SIZE: u64 = 100 * 1024 * 1024;
    if size > MAX_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file size exceeds maximum limit (100MB)",
        ));
    }

    let mut data = Vec::with_capacity(size as usize);
    file.seek(io::SeekFrom::Start(0))?;
    file.take(size).read_to_end(&mut data)?;
    if data.len() as u64 != size {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "unexpected EOF",
        ));
    }
    Ok(data)
}

/// The cases of Go's `http.DetectContentType` that `get_image_data` can
/// accept; everything else is the sniffer's `application/octet-stream`.
fn detect_content_type(data: &[u8]) -> &'static str {
    fn masked(data: &[u8], mask: &[u8], pat: &[u8]) -> bool {
        data.len() >= pat.len()
            && data
                .iter()
                .zip(mask)
                .zip(pat)
                .all(|((d, m), p)| d & m == *p)
    }
    if data.starts_with(b"\xFF\xD8\xFF") {
        "image/jpeg"
    } else if data.starts_with(b"\x89PNG\x0D\x0A\x1A\x0A") {
        "image/png"
    } else if masked(
        data,
        b"\xFF\xFF\xFF\xFF\x00\x00\x00\x00\xFF\xFF\xFF\xFF\xFF\xFF",
        b"RIFF\x00\x00\x00\x00WEBPVP",
    ) {
        "image/webp"
    } else if masked(
        data,
        b"\xFF\xFF\xFF\xFF\x00\x00\x00\x00\xFF\xFF\xFF\xFF",
        b"RIFF\x00\x00\x00\x00WAVE",
    ) {
        "audio/wave"
    } else {
        "application/octet-stream"
    }
}

// ---------------------------------------------------------------------------
// Terminal rendering — ported from ollama's cmd.go (displayResponse,
// thinkingOutputOpeningText/ClosingText): the "Thinking..." block is the
// one place `ollama run` colors anything.
// ---------------------------------------------------------------------------

/// ollama's readline.ColorGrey/ColorBold/ColorDefault (readline/types.go).
const COLOR_GREY: &str = "\x1b[38;5;245m";
const COLOR_BOLD: &str = "\x1b[1m";
const COLOR_DEFAULT: &str = "\x1b[0m";

/// Mirrors ollama's `thinkingOutputOpeningText`. Ends re-applying grey
/// (not a full reset) so the streamed thinking text after it can rely on
/// that still-active SGR state instead of coloring itself.
fn thinking_opening_text(plain: bool) -> String {
    let text = "Thinking...\n";
    if plain {
        text.to_string()
    } else {
        format!("{COLOR_GREY}{COLOR_BOLD}{text}{COLOR_DEFAULT}{COLOR_GREY}")
    }
}

/// Mirrors ollama's `thinkingOutputClosingText` — a full reset this time.
fn thinking_closing_text(plain: bool) -> String {
    let text = "...done thinking.\n\n";
    if plain {
        text.to_string()
    } else {
        format!("{COLOR_GREY}{COLOR_BOLD}{text}{COLOR_DEFAULT}")
    }
}

/// Mirrors ollama's `displayResponseState`.
#[derive(Default)]
struct WrapState {
    line_length: usize,
    word_buffer: String,
}

/// Terminal width, falling back to 80 like ollama's `displayResponse`
/// does when `term.GetSize` fails (e.g. not a real terminal).
fn term_width() -> usize {
    #[cfg(unix)]
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 {
            return ws.ws_col as usize;
        }
    }
    80
}

/// Pure computation half of `wrap_write`, split out so it's testable
/// without a real terminal. Direct port of ollama's `displayResponse`:
/// track the current word, and once a line would overflow, backtrack to
/// its start, clear to end of line, and continue on the next one. Every
/// char counts as one column (no `runewidth`-style CJK width handling).
fn wrap_chunk(content: &str, wrap: bool, width: usize, state: &mut WrapState) -> String {
    let mut out = String::new();
    if wrap && width >= 10 {
        for ch in content.chars() {
            if state.line_length + 1 > width - 5 {
                if state.word_buffer.chars().count() > width - 10 {
                    out.push_str(&state.word_buffer);
                    out.push(ch);
                    state.word_buffer.clear();
                    state.line_length = 0;
                    continue;
                }
                let a = state.word_buffer.chars().count();
                if a > 0 {
                    out.push_str(&format!("\x1b[{a}D"));
                }
                out.push_str("\x1b[K\n");
                out.push_str(&state.word_buffer);
                out.push(ch);
                state.line_length = state.word_buffer.chars().count() + 1;
            } else {
                out.push(ch);
                state.line_length += 1;
                match ch {
                    ' ' | '\t' => state.word_buffer.clear(),
                    '\n' | '\r' => {
                        state.line_length = 0;
                        state.word_buffer.clear();
                    }
                    _ => state.word_buffer.push(ch),
                }
            }
        }
    } else {
        out.push_str(&state.word_buffer);
        out.push_str(content);
        state.word_buffer.clear();
    }
    out
}

/// Streams `content` to stdout, word-wrapped via `wrap_chunk` above.
fn wrap_write(content: &str, wrap: bool, state: &mut WrapState) {
    let out = wrap_chunk(content, wrap, term_width(), state);
    print!("{out}");
    io::stdout().flush().ok();
}

/// Mirrors ollama's `progress.NewSpinner("")` (`chat`/`generate` in
/// cmd.go): a braille spinner shown until the first streamed response
/// object arrives, same glyphs and 100ms tick. Unlike ollama, only shown
/// when stderr is a real terminal, so a piped stderr doesn't get raw
/// escape sequences dumped into it.
fn start_spinner() -> Option<ProgressBar> {
    if !io::stderr().is_terminal() {
        return None;
    }
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner} ")
            .unwrap_or_else(|_| ProgressStyle::default_spinner())
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.enable_steady_tick(Duration::from_millis(100));
    Some(pb)
}

/// RAII wrapper around `start_spinner`'s result — mirrors ollama's own
/// `defer p.StopAndClear()`: whichever of `chat_submit_async`'s
/// early-return paths fires, this clears any still-running spinner
/// exactly once.
struct SpinnerGuard(Option<ProgressBar>);

impl SpinnerGuard {
    /// Called on Ctrl-C or once the first response object is decoded —
    /// a no-op after the first call, like ollama's own `Spinner.Stop`.
    fn stop(&mut self) {
        if let Some(sp) = self.0.take() {
            sp.finish_and_clear();
        }
    }
}

impl Drop for SpinnerGuard {
    fn drop(&mut self) {
        self.stop();
    }
}

// ---------------------------------------------------------------------------
// Interactive — TTY path
// ---------------------------------------------------------------------------

fn run_interactive_tty(model: &str, opts: ChatOptions<'_>) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        run_interactive_unix(model, opts)
    }
    #[cfg(not(unix))]
    {
        // Windows fallback: basic cooked-mode loop
        run_interactive_cooked(model, opts)
    }
}

// ---------------------------------------------------------------------------
// Interactive — Unix raw-mode readline (ported from ollama readline package)
// ---------------------------------------------------------------------------

#[cfg(unix)]
fn run_interactive_unix(model: &str, opts: ChatOptions<'_>) -> anyhow::Result<()> {
    use unix_readline::Readline;

    // One tokio runtime, reused across every turn's `block_on` — see
    // `chat_submit`'s doc comment for why this needs async reqwest.
    let rt = tokio::runtime::Runtime::new().context("start tokio runtime")?;
    let client = chat_client(opts.api_key)?;
    let mut messages: Vec<Msg> = Vec::new();
    let mut rl = Readline::new()?;
    let mut multiline: Option<String> = None; // Some while inside """
                                              // Accumulates lines while rl.pasting, mirrors ollama's `sb`.
    let mut paste_sb = String::new();

    loop {
        // ". " is ollama's AltPrompt: shown both inside a """ block
        // and while a bracketed paste is still accumulating.
        let prompt = if multiline.is_some() || !paste_sb.is_empty() {
            ". "
        } else {
            "> "
        };

        let line = match rl.readline(prompt) {
            Ok(Some(l)) => l,
            Ok(None) => break,
            Err(unix_readline::ReadlineError::Interrupted) => {
                multiline = None;
                paste_sb.clear();
                continue;
            }
        };

        // ── Bracketed paste accumulation ────────────────────────────────────
        // Mirrors `case scanner.Pasting: fmt.Fprintln(&sb, line); continue`
        // rl.pasting is true while between \x1b[200~ and \x1b[201~.
        // While pasting, ACCUMULATE into paste_sb WITHOUT submitting.
        // When pasting ends, the final line falls through to normal handling
        // with paste_sb prepended — same as ollama's `default: sb.WriteString`.
        if rl.pasting {
            paste_sb.push_str(&line);
            paste_sb.push('\n');
            continue;
        }

        // Not pasting: prepend any accumulated paste content to this line.
        // (ollama: `default: sb.WriteString(line)` then submit if sb.Len()>0)
        let line = if !paste_sb.is_empty() {
            let mut full = std::mem::take(&mut paste_sb);
            full.push_str(&line);
            full
        } else {
            line
        };

        // ── """ multiline mode ───────────────────────────────────────────────
        if let Some(ref mut buf) = multiline {
            if let Some(content) = line.strip_suffix("\"\"\"") {
                buf.push_str(content);
                let full = std::mem::take(buf).trim_end_matches('\n').to_string();
                multiline = None;
                if !full.trim().is_empty() {
                    submit_turn(&rt, &client, model, &mut rl, &mut messages, full, opts)?;
                }
            } else {
                buf.push_str(&line);
                buf.push('\n');
            }
            continue;
        }

        // ── Slash commands ───────────────────────────────────────────────────
        match line.trim() {
            "" => continue,
            "/bye" | "/exit" => break,
            "/clear" => {
                messages.clear();
                eprintln!("Conversation cleared.");
                continue;
            }
            s if s.starts_with('/') => {
                // Ollama: a line starting with an image path is a prompt,
                // not an unknown command, when the model is multimodal.
                let first_word = s.split_whitespace().next().unwrap_or(s);
                let is_file = opts.multimodal
                    && extract_file_names(s)
                        .iter()
                        .any(|f| f.starts_with(first_word));
                if !is_file {
                    eprintln!("Unknown command '{first_word}'.");
                    eprintln!("Commands: /bye  /clear  \"\"\" (multiline)");
                    continue;
                }
            }
            _ => {}
        }

        // ── Triple-quote multiline opener ────────────────────────────────────
        if line.trim_start().starts_with("\"\"\"") {
            let inner = line.trim_start().trim_start_matches("\"\"\"");
            if let Some(closed) = inner.strip_suffix("\"\"\"") {
                let content = closed.to_string();
                if !content.trim().is_empty() {
                    submit_turn(&rt, &client, model, &mut rl, &mut messages, content, opts)?;
                }
            } else {
                multiline = Some(inner.to_string() + "\n");
            }
            continue;
        }

        if !line.trim().is_empty() {
            submit_turn(&rt, &client, model, &mut rl, &mut messages, line, opts)?;
        }
    }

    Ok(())
}

/// Wraps `chat_submit` with the raw-mode toggle that lets Ctrl-C actually
/// interrupt a streaming response — mirrors ollama's
/// `readline.Instance.Readline()`, which holds raw mode only around
/// reading one line. Without this, raw mode (ISIG off) would stay active
/// through the response too, so Ctrl-C would just sit as an unread byte 3
/// until the next `rl.readline()` call instead of interrupting now.
#[cfg(unix)]
fn submit_turn(
    rt: &tokio::runtime::Runtime,
    client: &Client,
    model: &str,
    rl: &mut unix_readline::Readline,
    messages: &mut Vec<Msg>,
    content: String,
    opts: ChatOptions<'_>,
) -> anyhow::Result<()> {
    rl.leave_raw();
    let result = chat_submit(rt, client, model, messages, content, opts);
    rl.enter_raw();
    result
}

/// Sends one chat turn and streams the response, racing it against
/// Ctrl-C — mirrors ollama's per-turn `context.WithCancel` +
/// `signal.Notify(SIGINT)`: Ctrl-C stops just this turn's response
/// without killing the process. An interrupted turn's partial reply is
/// *not* added to `messages`, matching cmd.go's `chat` returning `nil`
/// on `context.Canceled`.
///
/// Async, not `reqwest::blocking`, so the stream can be raced against
/// `tokio::signal::ctrl_c()` via `tokio::select!` below.
fn chat_submit(
    rt: &tokio::runtime::Runtime,
    client: &Client,
    model: &str,
    messages: &mut Vec<Msg>,
    content: String,
    opts: ChatOptions<'_>,
) -> anyhow::Result<()> {
    rt.block_on(chat_submit_async(client, model, messages, content, opts))
}

async fn chat_submit_async(
    client: &Client,
    model: &str,
    messages: &mut Vec<Msg>,
    content: String,
    opts: ChatOptions<'_>,
) -> anyhow::Result<()> {
    // Ollama: `if opts.MultiModal { ... extractFileData(opts.Prompt) }`.
    // A non-vision model gets the path as plain text, as there.
    let mut user = Msg::text("user", content);
    if opts.multimodal {
        let (text, images) = extract_file_data(&user.content)?;
        user.content = text;
        if !images.is_empty() {
            user.images = Some(images);
        }
    }
    messages.push(user);

    // Mirrors ollama's `progress.NewSpinner("")`: ticks on stderr until
    // the first token/thinking chunk arrives, so `llmman run` doesn't sit
    // silently while a cold model loads server-side.
    let mut spinner = SpinnerGuard(start_spinner());

    // One listener for this whole turn, mirroring ollama's own
    // `signal.Notify(sigChan, syscall.SIGINT)` called once per turn (not
    // per streamed chunk). Recreating `tokio::signal::ctrl_c()` on every
    // loop iteration would subscribe a fresh listener each time, and a
    // SIGINT delivered in the gap between the old one dropping and the
    // new one subscribing could be missed — pinning it once and reusing
    // it via `&mut` closes that gap.
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    let send = client
        .post(format!("{}/api/chat", daemon::server()))
        .json(&ChatReq {
            model,
            messages,
            stream: true,
            think: opts.think,
            options: opts.num_predict.map(|n| ChatReqOptions {
                num_predict: Some(n),
            }),
        })
        .send();

    let resp = tokio::select! {
        r = send => r.context("connect to llmman serve")?,
        // Nothing printed yet, so just stop — `spinner`'s Drop clears it.
        _ = &mut ctrl_c => return Ok(()),
    };

    if !resp.status().is_success() {
        spinner.stop();
        let body = resp.text().await.unwrap_or_default();
        // The daemon's own message, not the JSON envelope around it: a
        // provider's 401 naming the key it rejected is the whole of what
        // is needed here.
        anyhow::bail!("{}", daemon::api_error(&body).unwrap_or(body));
    }

    // Stream NDJSON lines as they arrive, async so each read can be
    // `tokio::select!`ed against Ctrl-C below.
    let byte_stream = resp.bytes_stream().map_err(io::Error::other);
    let mut lines =
        tokio::io::BufReader::new(tokio_util::io::StreamReader::new(byte_stream)).lines();

    // Mirrors ollama's `plainText`: no ANSI codes when stdout is redirected.
    let plain_text = !io::stdout().is_terminal();
    let mut full = String::new();
    let mut thinking_content = String::new();
    let mut thinking_open = false;
    let mut thinking_closed = false;
    let mut wrap = WrapState::default();
    let mut interrupted = false;

    loop {
        let line = tokio::select! {
            l = lines.next_line() => l.context("read response stream")?,
            // Same pinned listener as above, see its comment for why.
            _ = &mut ctrl_c => { interrupted = true; break; }
        };
        let Some(line) = line else { break };
        if line.is_empty() {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<ChatChunk>(&line) else {
            continue;
        };
        // Cleared on the first decoded response object, like ollama's
        // own `fn` callback.
        spinner.stop();
        if let Some(ref msg) = chunk.message {
            if let Some(ref t) = msg.thinking {
                if !t.is_empty() {
                    if !thinking_open {
                        print!("{}", thinking_opening_text(plain_text));
                        thinking_open = true;
                        thinking_closed = false;
                    }
                    thinking_content.push_str(t);
                    wrap_write(t, opts.word_wrap, &mut wrap);
                }
            }
            if thinking_open && !thinking_closed && !msg.content.is_empty() {
                if !thinking_content.ends_with('\n') {
                    println!();
                }
                print!("{}", thinking_closing_text(plain_text));
                thinking_open = false;
                thinking_closed = true;
                wrap = WrapState::default();
            }
            if !msg.content.is_empty() {
                wrap_write(&msg.content, opts.word_wrap, &mut wrap);
                full.push_str(&msg.content);
            }
        }
        if chunk.done {
            break;
        }
    }

    if interrupted {
        return Ok(());
    }

    println!("\n");
    messages.push(Msg::text("assistant", full));
    Ok(())
}

// ---------------------------------------------------------------------------
// Unix raw-mode readline — direct port of ollama readline/readline.go
// ---------------------------------------------------------------------------

#[cfg(unix)]
mod unix_readline {
    use std::io::{BufReader, Read, Stdin, Write};
    use std::os::unix::io::AsRawFd;

    // Character codes — identical to ollama readline/types.go
    const CHAR_INTERRUPT: u8 = 3; // Ctrl-C
    const CHAR_EOF: u8 = 4; // Ctrl-D
    const CHAR_CTRL_J: u8 = 10; // \n  line feed / pasted newline
    const CHAR_ENTER: u8 = 13; // \r  keyboard Enter
    const CHAR_ESC: u8 = 27;
    const CHAR_ESCAPE_EX: u8 = 91; // '[' — second byte of ESC[
    const CHAR_BACKSPACE: u8 = 127;

    pub enum ReadlineError {
        Interrupted,
    }

    // CharBracketedPaste = 50 ('2') — third byte of ESC[ sequence;
    // reading 3 more bytes gives "00~" (paste start) or "01~" (paste end).
    // Mirrors ollama readline/types.go: CharBracketedPaste/Start/End.
    const CHAR_BRACKETED_PASTE: u8 = 50; // '2'
    const PASTE_START: &[u8; 3] = b"00~";
    const PASTE_END: &[u8; 3] = b"01~";

    pub struct Readline {
        reader: BufReader<Stdin>,
        orig: libc::termios,
        fd: std::os::unix::io::RawFd,
        pub pasting: bool, // true while inside \x1b[200~...\x1b[201~
    }

    impl Readline {
        /// Enable raw mode + bracketed paste (mirrors ollama SetRawMode + StartBracketedPaste).
        pub fn new() -> anyhow::Result<Self> {
            let stdin = std::io::stdin();
            let fd = stdin.as_raw_fd();

            let orig = unsafe {
                let mut t: libc::termios = std::mem::zeroed();
                if libc::tcgetattr(fd, &mut t) < 0 {
                    anyhow::bail!("tcgetattr failed");
                }
                t
            };

            let rl = Self {
                reader: BufReader::new(stdin),
                orig,
                fd,
                pasting: false,
            };
            if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &rl.raw_termios()) } < 0 {
                anyhow::bail!("tcsetattr failed");
            }

            // Enable bracketed paste for the whole session (toggled off
            // once, in Drop) — mirrors ollama's own start/end pair
            // wrapping the entire REPL loop, not each `Readline()` call.
            print!("\x1b[?2004h");
            std::io::stdout().flush().ok();

            Ok(rl)
        }

        /// Raw-mode termios derived from `orig`, mirrors `SetRawMode`.
        fn raw_termios(&self) -> libc::termios {
            let mut raw = self.orig;
            raw.c_iflag &= !(libc::IGNBRK
                | libc::BRKINT
                | libc::PARMRK
                | libc::ISTRIP
                | libc::INLCR
                | libc::IGNCR
                | libc::ICRNL
                | libc::IXON);
            raw.c_lflag &= !(libc::ECHO | libc::ECHONL | libc::ICANON | libc::ISIG | libc::IEXTEN);
            raw.c_cflag &= !(libc::CSIZE | libc::PARENB);
            raw.c_cflag |= libc::CS8;
            raw.c_cc[libc::VMIN] = 1;
            raw.c_cc[libc::VTIME] = 0;
            raw
        }

        /// (Re-)enters raw mode — used by `new()` and by `run::submit_turn`
        /// to undo `leave_raw` before the next `readline()` call.
        pub fn enter_raw(&self) {
            unsafe {
                libc::tcsetattr(self.fd, libc::TCSANOW, &self.raw_termios());
            }
        }

        /// Restores cooked (ISIG-enabled) mode for the duration of a
        /// streamed response, so a real SIGINT reaches `chat_submit`'s
        /// `ctrl_c()` instead of raw mode swallowing it as a plain byte 3.
        pub fn leave_raw(&self) {
            unsafe {
                libc::tcsetattr(self.fd, libc::TCSANOW, &self.orig);
            }
        }

        /// Read one logical line from the terminal.
        ///
        /// Paste detection mirrors ollama readline/readline.go exactly:
        ///   - After each read, check reader.buffer() (≡ reader.Buffered() in Go)
        ///   - If non-empty → draining (we are consuming a paste)
        ///   - CharCtrlJ (\n) while draining → submit (same as Enter)
        ///   - CharCtrlJ while NOT draining → Ctrl-J multiline continuation
        ///   - CharEnter (\r) → always submit
        pub fn readline(&mut self, prompt: &str) -> Result<Option<String>, ReadlineError> {
            print!("{prompt}");
            std::io::stdout().flush().ok();

            let mut buf: Vec<u8> = Vec::new();
            let mut pasted_lines: Vec<String> = Vec::new();
            let mut draining = false;
            let mut stop_draining = false;
            let mut esc = false;
            let mut esc_ex = false;

            loop {
                // Apply deferred state from previous iteration (ollama lines 130-134)
                if stop_draining {
                    draining = false;
                    stop_draining = false;
                }

                // Read exactly one byte
                let mut b = [0u8; 1];
                match self.reader.read_exact(&mut b) {
                    Ok(_) => {}
                    Err(_) => return Ok(None),
                }
                let r = b[0];

                // Paste detection: mirrors `if i.Terminal.reader.Buffered() > 0`
                if !self.reader.buffer().is_empty() {
                    draining = true;
                } else if draining {
                    stop_draining = true;
                }

                // ESC sequence handling — mirrors ollama readline.go escex block.
                // Key addition: CharBracketedPaste ('2') reads 3 more bytes to
                // detect "00~" (paste start) or "01~" (paste end).
                if esc_ex {
                    esc_ex = false;
                    match r {
                        CHAR_BRACKETED_PASTE => {
                            // Read 3 more bytes: "00~" or "01~"
                            let mut code = [0u8; 3];
                            if self.reader.read_exact(&mut code).is_ok() {
                                if &code == PASTE_START {
                                    self.pasting = true;
                                } else if &code == PASTE_END {
                                    self.pasting = false;
                                }
                                // Update draining after reading extra bytes
                                if !self.reader.buffer().is_empty() {
                                    draining = true;
                                }
                            }
                        }
                        // Consume the '~' for delete/other 2-byte sequences
                        51 | 53 | 54 => {
                            let mut tilde = [0u8; 1];
                            let _ = self.reader.read_exact(&mut tilde);
                        }
                        _ => {} // arrow keys etc. — just skip
                    }
                    continue;
                } else if esc {
                    esc = false;
                    if r == CHAR_ESCAPE_EX {
                        esc_ex = true;
                    }
                    continue;
                }

                match r {
                    CHAR_INTERRUPT => {
                        pasted_lines.clear();
                        buf.clear();
                        println!();
                        return Err(ReadlineError::Interrupted);
                    }
                    CHAR_EOF => {
                        // Mirrors ollama's `case CharDelete`: only checks
                        // the current line's own buffer, so Ctrl-D on a
                        // fresh empty continuation line exits the whole
                        // REPL even mid-multiline-entry.
                        if buf.is_empty() {
                            println!();
                            return Ok(None);
                        }
                    }
                    CHAR_ESC => {
                        esc = true;
                    }
                    CHAR_BACKSPACE => {
                        if !buf.is_empty() {
                            // Remove last complete UTF-8 codepoint
                            loop {
                                match buf.pop() {
                                    None => break,
                                    Some(b) if (b & 0xC0) != 0x80 => break, // lead byte
                                    Some(_) => {} // continuation byte, keep going
                                }
                            }
                            print!("\x08 \x08");
                            std::io::stdout().flush().ok();
                        } else if !pasted_lines.is_empty() {
                            let prev = pasted_lines.pop().unwrap();
                            print!("\r\x1b[K\x1b[A\r\x1b[K{prompt}{prev}");
                            std::io::stdout().flush().ok();
                            buf = prev.into_bytes();
                        }
                    }
                    CHAR_CTRL_J => {
                        // \n: pasted newline (draining) or Ctrl-J multiline (not draining)
                        // Mirrors ollama case CharCtrlJ
                        if !draining {
                            // Not draining → multiline continuation (Ctrl-J typed)
                            pasted_lines.push(String::from_utf8_lossy(&buf).to_string());
                            buf.clear();
                            println!();
                            print!(". ");
                            std::io::stdout().flush().ok();
                        } else {
                            // Draining → submit (pasted \n acts like Enter)
                            return Ok(Some(Self::assemble(&mut buf, &mut pasted_lines)));
                        }
                    }
                    CHAR_ENTER => {
                        // \r: keyboard Enter → always submit
                        return Ok(Some(Self::assemble(&mut buf, &mut pasted_lines)));
                    }
                    c => {
                        // Printable ASCII, tab, or UTF-8 bytes
                        if c >= 32 || c == 9 || c >= 0x80 {
                            buf.push(c);
                            let _ = std::io::stdout().write_all(&[c]);
                            std::io::stdout().flush().ok();
                        }
                    }
                }
            }
        }

        fn assemble(buf: &mut Vec<u8>, pasted_lines: &mut Vec<String>) -> String {
            let last = String::from_utf8_lossy(buf).to_string();
            buf.clear();
            println!();
            if pasted_lines.is_empty() {
                last
            } else {
                let prefix = pasted_lines.join("\n");
                pasted_lines.clear();
                format!("{prefix}\n{last}")
            }
        }
    }

    impl Drop for Readline {
        fn drop(&mut self) {
            // Disable bracketed paste, restore terminal — mirrors ollama's defer
            print!("\x1b[?2004l");
            std::io::stdout().flush().ok();
            unsafe {
                libc::tcsetattr(self.fd, libc::TCSANOW, &self.orig);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Windows / non-TTY fallback (cooked mode)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn run_interactive_cooked(model: &str, opts: ChatOptions<'_>) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new().context("start tokio runtime")?;
    let client = chat_client(opts.api_key)?;
    let mut messages: Vec<Msg> = Vec::new();
    use std::io::BufRead;
    let stdin = std::io::stdin();
    let mut reader = std::io::BufReader::new(stdin.lock());

    loop {
        print!("> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let line = line
            .trim_end_matches('\n')
            .trim_end_matches('\r')
            .to_string();
        match line.trim() {
            "" => continue,
            "/bye" | "/exit" => break,
            "/clear" => {
                messages.clear();
                continue;
            }
            _ => {}
        }
        if !line.trim().is_empty() {
            chat_submit(&rt, &client, model, &mut messages, line, opts)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_options_default_matches_ollamas_own_wordwrap_true_default() {
        let opts = ChatOptions::default();
        assert_eq!(opts.think, None);
        assert_eq!(opts.num_predict, None);
        assert!(opts.word_wrap);
        assert!(!opts.multimodal);
    }

    // -- image attachments: ports of ollama's cmd/interactive_test.go ------

    /// Ollama's `TestExtractFilenames`, unix-style half.
    #[test]
    fn extract_file_names_unix_style_paths() {
        let input = " some preamble \n \
./relative\\ path/one.png inbetween1 ./not a valid two.jpg inbetween2 ./1.svg\n\
/unescaped space /three.jpeg inbetween3 /valid\\ path/dir/four.png \"./quoted with spaces/five.JPG\n\
/unescaped space /six.webp inbetween6 /valid\\ path/dir/seven.WEBP";
        let res = extract_file_names(input);
        assert_eq!(res.len(), 7, "{res:?}");
        assert!(res[0].contains("one.png"));
        assert!(res[1].contains("two.jpg"));
        assert!(res[2].contains("three.jpeg"));
        assert!(res[3].contains("four.png"));
        assert!(res[4].contains("five.JPG"));
        assert!(res[5].contains("six.webp"));
        assert!(res[6].contains("seven.WEBP"));
        assert!(!res[4].contains('"'));
        assert!(!res.contains(&"inbetween1"));
        assert!(!res.contains(&"./1.svg"));
    }

    /// Ollama's `TestExtractFilenames`, windows-style half.
    #[test]
    fn extract_file_names_windows_style_paths() {
        let input = " some preamble\n \
c:/users/jdoe/one.png inbetween1 c:/program files/someplace/two.jpg inbetween2 \n \
/absolute/nospace/three.jpeg inbetween3 /absolute/with space/four.png inbetween4\n\
./relative\\ path/five.JPG inbetween5 \"./relative with/spaces/six.png inbetween6\n\
d:\\path with\\spaces\\seven.JPEG inbetween7 c:\\users\\jdoe\\eight.png inbetween8 \n \
d:\\program files\\someplace\\nine.png inbetween9 \"E:\\program files\\someplace\\ten.PNG\n\
c:/users/jdoe/eleven.webp inbetween11 c:/program files/someplace/twelve.WebP inbetween12\n\
d:\\path with\\spaces\\thirteen.WEBP some ending\n";
        let res = extract_file_names(input);
        assert_eq!(res.len(), 13, "{res:?}");
        assert!(!res.contains(&"inbetween2"));
        assert!(res[0].contains("one.png") && res[0].contains("c:"));
        assert!(res[1].contains("two.jpg") && res[1].contains("c:"));
        assert!(res[2].contains("three.jpeg"));
        assert!(res[3].contains("four.png"));
        assert!(res[4].contains("five.JPG"));
        assert!(res[5].contains("six.png"));
        assert!(res[6].contains("seven.JPEG") && res[6].contains("d:"));
        assert!(res[7].contains("eight.png") && res[7].contains("c:"));
        assert!(res[8].contains("nine.png") && res[8].contains("d:"));
        assert!(res[9].contains("ten.PNG") && res[9].contains("E:"));
        assert!(res[10].contains("eleven.webp") && res[10].contains("c:"));
        assert!(res[11].contains("twelve.WebP") && res[11].contains("c:"));
        assert!(res[12].contains("thirteen.WEBP") && res[12].contains("d:"));
    }

    #[test]
    fn normalize_file_path_undoes_shell_escapes() {
        assert_eq!(
            normalize_file_path("/My\\ Photos/a\\(1\\)\\[x\\]\\{y\\}\\$\\&\\;\\'\\*\\?\\~.png"),
            "/My Photos/a(1)[x]{y}$&;'*?~.png"
        );
        assert_eq!(normalize_file_path("a\\\\b"), "a\\b");
        assert_eq!(normalize_file_path("/plain/path.jpg"), "/plain/path.jpg");
    }

    #[test]
    fn detect_content_type_matches_gos_sniffer_for_the_allowed_types() {
        assert_eq!(
            detect_content_type(b"\xFF\xD8\xFF\xE0\x00\x10JFIF"),
            "image/jpeg"
        );
        assert_eq!(
            detect_content_type(b"\x89PNG\x0D\x0A\x1A\x0A\x00\x00\x00\x0DIHDR"),
            "image/png"
        );
        assert_eq!(
            detect_content_type(b"RIFF\x24\x00\x00\x00WEBPVP8 "),
            "image/webp"
        );
        assert_eq!(
            detect_content_type(b"RIFF\x58\x02\x00\x00WAVEfmt "),
            "audio/wave"
        );
        assert_eq!(detect_content_type(b"GIF89a"), "application/octet-stream");
        assert_eq!(detect_content_type(b"hello"), "application/octet-stream");
        assert_eq!(detect_content_type(b"RIFF"), "application/octet-stream");
    }

    fn temp_file(name: &str, data: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "llmman-run-img-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let fp = dir.join(name);
        std::fs::write(&fp, data).unwrap();
        fp
    }

    fn jpeg_bytes() -> Vec<u8> {
        let mut data = vec![0u8; 600];
        data[..22].copy_from_slice(&[
            0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, b'J', b'F', b'I', b'F', 0x00, 0x01, 0x01, 0x01,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xd9,
        ]);
        data
    }

    /// Ollama's `TestExtractFileDataRemovesQuotedFilepath`.
    #[test]
    fn extract_file_data_removes_quoted_filepath() {
        let fp = temp_file("img.jpg", &jpeg_bytes());
        let input = format!("before '{}' after", fp.display());
        let (cleaned, imgs) = extract_file_data(&input).unwrap();
        assert_eq!(imgs.len(), 1);
        assert_eq!(cleaned, "before  after");
    }

    /// Ollama's `TestExtractFileDataWAV`.
    #[test]
    fn extract_file_data_wav() {
        let mut data = vec![0u8; 600];
        data[..44].copy_from_slice(&[
            b'R', b'I', b'F', b'F', //
            0x58, 0x02, 0x00, 0x00, // file size - 8
            b'W', b'A', b'V', b'E', //
            b'f', b'm', b't', b' ', //
            0x10, 0x00, 0x00, 0x00, // fmt chunk size
            0x01, 0x00, // PCM
            0x01, 0x00, // mono
            0x80, 0x3e, 0x00, 0x00, // 16000 Hz
            0x00, 0x7d, 0x00, 0x00, // byte rate
            0x02, 0x00, // block align
            0x10, 0x00, // 16-bit
            b'd', b'a', b't', b'a', //
            0x34, 0x02, 0x00, 0x00, // data size
        ]);
        let fp = temp_file("sample.wav", &data);
        let input = format!("before {} after", fp.display());
        let (cleaned, imgs) = extract_file_data(&input).unwrap();
        assert_eq!(imgs.len(), 1);
        assert_eq!(cleaned, "before  after");
    }

    #[test]
    fn extract_file_data_encodes_the_file_as_standard_base64() {
        let bytes = jpeg_bytes();
        let fp = temp_file("img.jpg", &bytes);
        let (_, imgs) = extract_file_data(&format!("look: {}", fp.display())).unwrap();
        assert_eq!(
            imgs,
            vec![base64::engine::general_purpose::STANDARD.encode(&bytes)]
        );
    }

    #[test]
    fn extract_file_data_leaves_a_nonexistent_path_in_the_prompt() {
        let input = "what is in /definitely/not/here.png then";
        let (cleaned, imgs) = extract_file_data(input).unwrap();
        assert!(imgs.is_empty());
        assert_eq!(cleaned, input);
    }

    #[test]
    fn extract_file_data_rejects_an_existing_file_of_the_wrong_type() {
        let fp = temp_file("fake.png", b"this is not a png at all, just text");
        let err = extract_file_data(&format!("see {}", fp.display())).unwrap_err();
        assert!(err.to_string().contains("invalid file type"), "{err:#}");
    }

    #[test]
    fn msg_images_is_omitted_from_json_when_none() {
        let json = serde_json::to_string(&Msg::text("user", "hi".into())).unwrap();
        assert_eq!(json, r#"{"role":"user","content":"hi"}"#);
        let mut with = Msg::text("user", "hi".into());
        with.images = Some(vec!["QUJD".into()]);
        let json = serde_json::to_string(&with).unwrap();
        assert_eq!(json, r#"{"role":"user","content":"hi","images":["QUJD"]}"#);
    }

    #[test]
    fn thinking_opening_text_plain_has_no_ansi_codes() {
        assert_eq!(thinking_opening_text(true), "Thinking...\n");
    }

    #[test]
    fn thinking_opening_text_colored_wraps_in_grey_bold_then_grey_again() {
        let text = thinking_opening_text(false);
        assert!(text.starts_with(&format!("{COLOR_GREY}{COLOR_BOLD}")));
        assert!(text.contains("Thinking...\n"));
        // Re-applies grey rather than a full reset.
        assert!(text.ends_with(&format!("{COLOR_DEFAULT}{COLOR_GREY}")));
    }

    #[test]
    fn thinking_closing_text_plain_has_no_ansi_codes() {
        assert_eq!(thinking_closing_text(true), "...done thinking.\n\n");
    }

    #[test]
    fn thinking_closing_text_colored_ends_in_a_full_reset() {
        let text = thinking_closing_text(false);
        assert!(text.starts_with(&format!("{COLOR_GREY}{COLOR_BOLD}")));
        assert!(text.contains("...done thinking.\n\n"));
        assert!(text.ends_with(COLOR_DEFAULT));
        assert!(!text.ends_with(&format!("{COLOR_DEFAULT}{COLOR_GREY}")));
    }

    #[test]
    fn wrap_chunk_disabled_passes_content_through_unchanged() {
        let mut state = WrapState::default();
        let out = wrap_chunk(
            "hello there, this line is not wrapped at all",
            false,
            20,
            &mut state,
        );
        assert_eq!(out, "hello there, this line is not wrapped at all");
    }

    #[test]
    fn wrap_chunk_too_narrow_terminal_passes_content_through_unchanged() {
        // Mirrors ollama's `wordWrap && termWidth >= 10` guard.
        let mut state = WrapState::default();
        let out = wrap_chunk("hello there", true, 9, &mut state);
        assert_eq!(out, "hello there");
    }

    #[test]
    fn wrap_chunk_breaks_before_a_word_that_would_overflow_the_line() {
        let mut state = WrapState::default();
        // width=20: overflow threshold is line_length+1 > width-5 (>15).
        let out = wrap_chunk("one two three four five", true, 20, &mut state);
        // "four" gets backtracked onto a new line instead of splitting.
        assert!(out.contains("\x1b[K\n"));
        let after_break = out.split("\x1b[K\n").nth(1).unwrap();
        assert!(after_break.starts_with("four"));
    }

    #[test]
    fn wrap_chunk_state_persists_across_calls_like_a_single_stream() {
        // Several small calls (one per streamed token) must wrap the
        // same as one big call, carried via WrapState between them.
        let mut state_streamed = WrapState::default();
        let mut streamed = String::new();
        for word in ["one ", "two ", "three ", "four ", "five"] {
            streamed.push_str(&wrap_chunk(word, true, 20, &mut state_streamed));
        }

        let mut state_whole = WrapState::default();
        let whole = wrap_chunk("one two three four five", true, 20, &mut state_whole);

        assert_eq!(streamed, whole);
    }

    #[test]
    fn wrap_chunk_an_unbroken_word_longer_than_the_line_flushes_without_backtracking() {
        // A word too long to ever fit its own line skips the backtrack
        // and just flushes the buffered prefix again — faithfully
        // mirroring ollama's own `fmt.Printf("%s%c", wordBuffer, ch)`
        // here, the buffered prefix really does appear twice in the
        // output for this rare edge case.
        let mut state = WrapState::default();
        let out = wrap_chunk("supercalifragilisticexpialidocious", true, 20, &mut state);
        assert_eq!(
            out,
            "supercalifragilsupercalifragilisticexpialidocisticexpialidocious"
        );
    }
}

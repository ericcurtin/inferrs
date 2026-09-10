//! `llmman-bench` — measures prefill/decode throughput against one or
//! more models served by an already-running `llmman serve`. The llmman
//! equivalent of Ollama's own standalone `ollama-bench` (`ollama/cmd/
//! bench`): a separate binary, not a subcommand, and — like
//! `ollama-bench` — it never starts the server itself.
//!
//! Talks to `/v1/chat/completions` with `stream_options.include_usage`;
//! `cmd::serve`'s OpenAI proxy forwards both request and response
//! byte-for-byte to the backend `llama-server`, so its own final
//! `usage` chunk gives real prompt/completion token counts here — this
//! file only measures wall-clock time.
//!
//! Several behaviors are ported from `ollama-bench` for parity: varying
//! the prompt per warmup/epoch/retry to defeat KV-cache prefix
//! matching, retrying a short (< `--max-tokens`) timed epoch, an
//! optional `--seed`, and unloading each model once its run finishes.
//! Not ported: `-num-ctx` (context size is fixed at model-load time by
//! a daemon that may already be running), `-image` (no vision-bench
//! path yet), `-k`/keep-alive-between-epochs (every model still ends
//! up unloaded either way), and the `benchstat` format (`--format`
//! stays `text`/`csv`, same information).

use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::Context;
use clap::Parser;
use futures::TryStreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;

use llmman::daemon;
use llmman::shortnames;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Default prompt: long/open-ended enough to reliably fill `--max-tokens`
/// instead of stopping early, mirroring `ollama-bench`'s own default.
const DEFAULT_PROMPT: &str =
    "Write a detailed short story about a robot exploring an abandoned space station.";

/// Retries for a short timed epoch, matching `ollama-bench`'s own `maxRetries`.
const MAX_SHORT_RESPONSE_RETRIES: u32 = 3;

#[derive(Parser, Debug)]
#[command(
    name = "llmman-bench",
    about = "Measure prefill/decode throughput for one or more served models",
    version = env!("LLMMAN_VERSION")
)]
struct Args {
    /// Model(s) to benchmark (repeat, or comma-separate, e.g. `-m a,b`).
    #[arg(
        short = 'm',
        long = "model",
        value_name = "MODEL",
        value_delimiter = ',',
        required = true
    )]
    model: Vec<String>,

    /// Prompt sent on every request, tagged with a `[N]` marker per
    /// request (see `build_prompt`). Ignored when --prompt-tokens > 0.
    #[arg(short = 'p', long, default_value = DEFAULT_PROMPT)]
    prompt: String,

    /// Build a synthetic filler prompt targeting ~N tokens instead of
    /// --prompt. 0 (default) uses --prompt as-is. Only a target — the
    /// real prefill count reported is whatever the backend measures.
    #[arg(long, default_value_t = 0, value_name = "N")]
    prompt_tokens: u32,

    /// Max tokens to generate per request. A timed epoch shorter than
    /// this is retried (up to `MAX_SHORT_RESPONSE_RETRIES` times) with
    /// a varied prompt, since a truncated response skews decode tok/s.
    #[arg(long, default_value_t = 200, value_name = "N")]
    max_tokens: u32,

    /// Timed iterations per model, averaged in the result. Defaults to
    /// 6, matching `ollama-bench`'s own default.
    #[arg(long, default_value_t = 6, value_name = "N")]
    epochs: u32,

    /// Untimed requests before the timed epochs, letting a cold model
    /// finish loading without skewing the timed results.
    #[arg(long, default_value_t = 1, value_name = "N")]
    warmup: u32,

    /// Sampling temperature; 0 (default) is greedy, for reproducible
    /// timing across epochs.
    #[arg(long, default_value_t = 0.0, value_name = "N")]
    temperature: f32,

    /// Random seed forwarded to the backend when non-zero; 0 (default)
    /// omits the field, matching `ollama-bench`'s own `-seed`.
    #[arg(long, default_value_t = 0, value_name = "N")]
    seed: u64,

    /// Per-request timeout, in seconds.
    #[arg(long, default_value_t = 300, value_name = "SECONDS")]
    timeout: u64,

    /// Output format: `text` (aligned table) or `csv`.
    #[arg(long, default_value = "text", value_name = "text|csv")]
    format: String,

    /// Write results to this file instead of stdout.
    #[arg(long, value_name = "FILE")]
    output: Option<String>,

    /// Print basic system info (OS/architecture) to stderr before running.
    #[arg(short = 'v', long)]
    verbose: bool,

    /// Echo each streamed token to stderr as it arrives.
    #[arg(long)]
    debug: bool,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();
    if let Err(e) = run(&args) {
        eprintln!("Error: {:#}", e);
        std::process::exit(1);
    }
}

fn run(args: &Args) -> anyhow::Result<()> {
    if !matches!(args.format.as_str(), "text" | "csv") {
        anyhow::bail!(
            "--format must be \"text\" or \"csv\" (got {:?})",
            args.format
        );
    }
    if args.epochs == 0 {
        // Sample::mean's `len().max(1)` divisor would otherwise silently
        // report an all-zero "success" instead of measuring nothing.
        anyhow::bail!("--epochs must be at least 1");
    }
    // Unlike `llmman bench` (its former subcommand incarnation), this
    // binary never spawns the daemon itself — see this file's own doc
    // comment for why — so it needs an explicit, actionable error
    // instead of a raw connection-refused a few requests in.
    if !daemon::server_alive() {
        anyhow::bail!(
            "llmman serve is not running on {} — start it first (`llmman serve`), then retry",
            daemon::bind_addr()
        );
    }
    if args.verbose {
        eprintln!(
            "os: {} | arch: {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        );
    }

    let rt = tokio::runtime::Runtime::new().context("start tokio runtime")?;
    let client = daemon::async_client_builder()?
        .timeout(Duration::from_secs(args.timeout))
        .build()
        .context("build http client")?;

    let mut results = Vec::with_capacity(args.model.len());
    for raw_model in &args.model {
        let model = shortnames::resolve_ollama_api(raw_model)?;
        // Fail fast on a bad/unresolvable reference rather than only
        // discovering it partway through warmup below.
        daemon::ensure_model_pulled(&model)?;

        let samples = bench_one_model(&rt, &client, &model, args);
        // Unload unconditionally — including on a failed/short-circuited
        // benchmark run — so a model never lingers resident (or
        // contends for VRAM with whatever gets benchmarked next) just
        // because one of its requests errored out. Mirrors
        // `ollama-bench`'s own unconditional `unloadModel` call.
        rt.block_on(unload_model(&client, &model));
        results.push((raw_model.clone(), Sample::mean(&samples?)));
    }

    let mut out: Box<dyn Write> = match &args.output {
        Some(path) => Box::new(
            std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(path)
                .with_context(|| format!("open --output file {path:?}"))?,
        ),
        None => Box::new(std::io::stdout()),
    };
    match args.format.as_str() {
        "csv" => print_csv(&mut out, &results),
        _ => print_table(&mut out, &results),
    }
    .context("write results")
}

/// Runs every warmup and timed epoch for one model, retrying a short
/// response as described on `Args::max_tokens`. Returns one `Sample`
/// per timed epoch (not yet averaged — see `Sample::mean`).
fn bench_one_model(
    rt: &tokio::runtime::Runtime,
    client: &Client,
    model: &str,
    args: &Args,
) -> anyhow::Result<Vec<Sample>> {
    eprintln!(
        "[llmman-bench] {model}: {} warmup + {} timed epoch(s)...",
        args.warmup, args.epochs
    );
    for i in 0..args.warmup {
        // Negative variants so they can never collide with a timed
        // epoch's own variant below.
        let variant = -(i as i64 + 1);
        let prompt = build_prompt(&args.prompt, args.prompt_tokens, variant);
        rt.block_on(one_request(client, model, &prompt, args))?;
    }

    let mut samples = Vec::with_capacity(args.epochs as usize);
    let mut short_count = 0u32;
    for epoch in 0..args.epochs {
        let mut accepted = None;
        for attempt in 0..=MAX_SHORT_RESPONSE_RETRIES {
            // (epoch, attempt) -> a unique variant with no fixed stride
            // to collide across epochs (unlike a plain `epoch +
            // attempt*1000`, which repeats once epoch reaches 1000).
            let variant = epoch as i64 * (MAX_SHORT_RESPONSE_RETRIES as i64 + 1) + attempt as i64;
            let prompt = build_prompt(&args.prompt, args.prompt_tokens, variant);
            let sample = rt.block_on(one_request(client, model, &prompt, args))?;
            let short = args.max_tokens > 0 && sample.completion_tokens < args.max_tokens;
            if short && attempt < MAX_SHORT_RESPONSE_RETRIES {
                eprintln!(
                    "[llmman-bench]   epoch {}/{}: short response ({}/{} tokens), retrying \
                     ({}/{MAX_SHORT_RESPONSE_RETRIES})...",
                    epoch + 1,
                    args.epochs,
                    sample.completion_tokens,
                    args.max_tokens,
                    attempt + 1,
                );
                continue;
            }
            if short {
                short_count += 1;
            }
            accepted = Some(sample);
            break;
        }
        let sample =
            accepted.expect("the attempt loop above always assigns before its last iteration");
        eprintln!(
            "[llmman-bench]   epoch {}/{}: prefill {:.1} tok/s, decode {:.1} tok/s",
            epoch + 1,
            args.epochs,
            sample.prefill_toks_per_sec(),
            sample.decode_toks_per_sec(),
        );
        samples.push(sample);
    }
    if short_count > 0 {
        eprintln!(
            "[llmman-bench] WARNING: {short_count}/{} epoch(s) for {model} had short responses \
             (<{} tokens) even after retrying; decode tok/s may be unreliable.",
            args.epochs, args.max_tokens
        );
    }
    Ok(samples)
}

/// Filler word list for a synthetic `--prompt-tokens` prompt, ported
/// verbatim from `ollama-bench`'s own `promptWordList`.
const PROMPT_WORD_LIST: &[&str] = &[
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "a",
    "bright",
    "sunny",
    "day",
    "in",
    "the",
    "meadow",
    "where",
    "flowers",
    "bloom",
    "and",
    "birds",
    "sing",
    "their",
    "morning",
    "songs",
    "while",
    "gentle",
    "breeze",
    "carries",
    "sweet",
    "scent",
    "of",
    "pine",
    "trees",
    "across",
    "rolling",
    "hills",
    "toward",
    "distant",
    "mountains",
    "covered",
    "with",
    "fresh",
    "snow",
    "beneath",
    "clear",
    "blue",
    "sky",
    "children",
    "play",
    "near",
    "old",
    "stone",
    "bridge",
    "that",
    "crosses",
    "winding",
    "river",
];

/// `ollama-bench`'s own initial (uncalibrated) tokens-per-word ratio —
/// llmman has no local tokenizer to calibrate against, so this fixed
/// value is the whole estimate.
const TOKENS_PER_WORD: f64 = 1.3;

/// Builds one request's prompt. `variant` (distinct per warmup/epoch/
/// retry — see call sites) keeps repeated requests from being
/// byte-identical, so the backend's KV-cache prefix matching can't
/// quietly turn a "cold" prefill measurement into a cache hit.
/// `--prompt-tokens` (when non-zero) always wins over `--prompt`.
fn build_prompt(base_prompt: &str, prompt_tokens: u32, variant: i64) -> String {
    if prompt_tokens > 0 {
        synthetic_prompt(prompt_tokens, variant)
    } else {
        format!("[{variant}] {base_prompt}")
    }
}

/// A synthetic filler prompt targeting roughly `target_tokens` tokens,
/// strided by `variant` — see `build_prompt`.
fn synthetic_prompt(target_tokens: u32, variant: i64) -> String {
    let target_words = ((target_tokens as f64 / TOKENS_PER_WORD) as i64).max(1) as usize;
    let n = PROMPT_WORD_LIST.len() as i64;
    // Stride by a prime (7, matching ollama-bench) so consecutive
    // variants don't just shift the word list by one.
    let offset = (variant * 7).rem_euclid(n) as usize;
    (0..target_words)
        .map(|i| PROMPT_WORD_LIST[(i + offset) % PROMPT_WORD_LIST.len()])
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// One timed request
// ---------------------------------------------------------------------------

/// One epoch's measurements. `ttft` (time to first streamed token) is
/// treated as prefill time — the two are inseparable from a plain HTTP
/// client's view, matching how `ollama-bench` and most black-box LLM
/// benchmarks approximate the same split.
#[derive(Debug, Clone, Copy)]
struct Sample {
    ttft: Duration,
    total: Duration,
    prompt_tokens: u32,
    completion_tokens: u32,
}

impl Sample {
    fn prefill_toks_per_sec(&self) -> f64 {
        let secs = self.ttft.as_secs_f64();
        if secs <= 0.0 {
            0.0
        } else {
            self.prompt_tokens as f64 / secs
        }
    }

    /// Excludes the one token already counted within `ttft`.
    fn decode_toks_per_sec(&self) -> f64 {
        let decode_tokens = self.completion_tokens.saturating_sub(1);
        let secs = (self.total - self.ttft.min(self.total)).as_secs_f64();
        if decode_tokens == 0 || secs <= 0.0 {
            0.0
        } else {
            decode_tokens as f64 / secs
        }
    }

    /// Averages every field across `samples`. Panics on an empty slice
    /// (every call site here always has at least one epoch).
    fn mean(samples: &[Sample]) -> Sample {
        let n = samples.len().max(1) as u32;
        let sum = samples.iter().fold(
            Sample {
                ttft: Duration::ZERO,
                total: Duration::ZERO,
                prompt_tokens: 0,
                completion_tokens: 0,
            },
            |acc, s| Sample {
                ttft: acc.ttft + s.ttft,
                total: acc.total + s.total,
                prompt_tokens: acc.prompt_tokens + s.prompt_tokens,
                completion_tokens: acc.completion_tokens + s.completion_tokens,
            },
        );
        Sample {
            ttft: sum.ttft / n,
            total: sum.total / n,
            prompt_tokens: sum.prompt_tokens / n,
            completion_tokens: sum.completion_tokens / n,
        }
    }
}

#[derive(Serialize)]
struct BenchRequest<'a> {
    model: &'a str,
    messages: [BenchMessage; 1],
    stream: bool,
    temperature: f32,
    max_tokens: u32,
    /// Only present when `--seed` is non-zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    stream_options: StreamOptions,
}

#[derive(Serialize)]
struct BenchMessage {
    role: &'static str,
    content: String,
}

/// Asks llama-server's OpenAI-compatible streaming endpoint to append a
/// final chunk carrying real `prompt_tokens`/`completion_tokens` counts.
#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Deserialize, Default)]
struct BenchChunk {
    #[serde(default)]
    choices: Vec<BenchChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize, Default)]
struct BenchChoice {
    #[serde(default)]
    delta: BenchDelta,
}

#[derive(Deserialize, Default)]
struct BenchDelta {
    #[serde(default)]
    content: Option<String>,
    // A thinking-capable model streams reasoning under one of these two
    // field names instead of `content`, sometimes for its entire token
    // budget — TTFT has to count the first token of any kind, or a
    // thinking-heavy response would misreport generation as prefill.
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
}

impl BenchDelta {
    fn has_any_token(&self) -> bool {
        [&self.content, &self.reasoning_content, &self.thinking]
            .into_iter()
            .any(|f| f.as_deref().is_some_and(|s| !s.is_empty()))
    }

    /// Every non-empty token field, for `--debug`'s raw echo.
    fn tokens(&self) -> impl Iterator<Item = &str> {
        [&self.content, &self.reasoning_content, &self.thinking]
            .into_iter()
            .filter_map(|f| f.as_deref())
            .filter(|s| !s.is_empty())
    }
}

#[derive(Deserialize, Default, Clone, Copy)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

/// Sends one streamed chat completion and measures it — see `Sample`.
async fn one_request(
    client: &Client,
    model: &str,
    prompt: &str,
    args: &Args,
) -> anyhow::Result<Sample> {
    let start = Instant::now();
    let resp = client
        .post(format!("{}/v1/chat/completions", daemon::server()))
        .json(&BenchRequest {
            model,
            messages: [BenchMessage {
                role: "user",
                content: prompt.to_string(),
            }],
            stream: true,
            temperature: args.temperature,
            max_tokens: args.max_tokens,
            seed: (args.seed > 0).then_some(args.seed),
            stream_options: StreamOptions {
                include_usage: true,
            },
        })
        .send()
        .await
        .context("connect to llmman serve")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("{model}: server returned {status}: {body}");
    }

    let byte_stream = resp.bytes_stream().map_err(std::io::Error::other);
    let mut lines =
        tokio::io::BufReader::new(tokio_util::io::StreamReader::new(byte_stream)).lines();

    let mut ttft: Option<Duration> = None;
    let mut usage: Option<Usage> = None;
    // Set only by the `[DONE]` sentinel, not by the loop simply running
    // out of lines (a connection dropped mid-response does that too) —
    // so a truncated stream errors instead of reporting a silently
    // partial/zeroed-out Sample.
    let mut saw_done = false;
    while let Some(line) = lines.next_line().await.context("read response stream")? {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload == "[DONE]" {
            saw_done = true;
            break;
        }
        let Ok(chunk) = serde_json::from_str::<BenchChunk>(payload) else {
            continue;
        };
        if let Some(u) = chunk.usage {
            usage = Some(u);
        }
        if let Some(choice) = chunk.choices.first() {
            if args.debug {
                for token in choice.delta.tokens() {
                    eprint!("{token}");
                }
            }
            if ttft.is_none() && choice.delta.has_any_token() {
                ttft = Some(start.elapsed());
            }
        }
    }
    if args.debug {
        eprintln!();
    }
    let total = start.elapsed();

    if !saw_done {
        anyhow::bail!(
            "{model}: stream ended without a [DONE] terminator (connection dropped mid-response?)"
        );
    }
    let usage = usage.ok_or_else(|| {
        anyhow::anyhow!(
            "{model}: stream completed without a usage summary — backend may not support \
             stream_options.include_usage"
        )
    })?;

    Ok(Sample {
        ttft: ttft.unwrap_or(total),
        total,
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
    })
}

/// Unloads `model` via the same `keep_alive: 0` + empty-prompt signal
/// Ollama's `/api/generate` treats as an explicit unload request.
/// Best-effort and fire-and-forget, like `ollama-bench`'s own
/// `unloadModel` — a failed unload shouldn't fail the whole run.
async fn unload_model(client: &Client, model: &str) {
    let _ = client
        .post(format!("{}/api/generate", daemon::server()))
        .json(&serde_json::json!({ "model": model, "keep_alive": 0 }))
        .send()
        .await;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn print_table(w: &mut dyn Write, results: &[(String, Sample)]) -> anyhow::Result<()> {
    let name_w = results
        .iter()
        .map(|(n, _)| n.len())
        .max()
        .unwrap_or(5)
        .max(5);
    writeln!(
        w,
        "{:<name_w$}  {:>14}  {:>14}  {:>10}  {:>10}  {:>12}  {:>14}",
        "MODEL",
        "PREFILL tok/s",
        "DECODE tok/s",
        "TTFT",
        "TOTAL",
        "PROMPT tok",
        "COMPLETION tok",
        name_w = name_w,
    )?;
    for (name, s) in results {
        writeln!(
            w,
            "{:<name_w$}  {:>14.1}  {:>14.1}  {:>10}  {:>10}  {:>12}  {:>14}",
            name,
            s.prefill_toks_per_sec(),
            s.decode_toks_per_sec(),
            format!("{:.2}s", s.ttft.as_secs_f64()),
            format!("{:.2}s", s.total.as_secs_f64()),
            s.prompt_tokens,
            s.completion_tokens,
            name_w = name_w,
        )?;
    }
    Ok(())
}

fn print_csv(w: &mut dyn Write, results: &[(String, Sample)]) -> anyhow::Result<()> {
    writeln!(
        w,
        "model,prefill_toks_per_sec,decode_toks_per_sec,ttft_ms,total_ms,prompt_tokens,completion_tokens"
    )?;
    for (name, s) in results {
        writeln!(
            w,
            "{},{:.2},{:.2},{},{},{},{}",
            name,
            s.prefill_toks_per_sec(),
            s.decode_toks_per_sec(),
            s.ttft.as_millis(),
            s.total.as_millis(),
            s.prompt_tokens,
            s.completion_tokens,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(ttft_ms: u64, total_ms: u64, prompt_tokens: u32, completion_tokens: u32) -> Sample {
        Sample {
            ttft: Duration::from_millis(ttft_ms),
            total: Duration::from_millis(total_ms),
            prompt_tokens,
            completion_tokens,
        }
    }

    fn args(model: &str, format: &str, epochs: u32) -> Args {
        Args {
            model: vec![model.to_string()],
            prompt: DEFAULT_PROMPT.to_string(),
            prompt_tokens: 0,
            max_tokens: 200,
            epochs,
            warmup: 0,
            temperature: 0.0,
            seed: 0,
            timeout: 300,
            format: format.to_string(),
            output: None,
            verbose: false,
            debug: false,
        }
    }

    #[test]
    fn run_rejects_zero_epochs_before_touching_the_network() {
        let err = run(&args("unused-model", "text", 0)).unwrap_err();
        assert!(err.to_string().contains("--epochs must be at least 1"));
    }

    #[test]
    fn run_rejects_an_unknown_format_before_touching_the_network() {
        let err = run(&args("unused-model", "yaml", 1)).unwrap_err();
        assert!(err.to_string().contains("--format must be"));
    }

    #[test]
    fn prefill_toks_per_sec_divides_prompt_tokens_by_ttft() {
        let s = sample(1000, 5000, 512, 200);
        assert!((s.prefill_toks_per_sec() - 512.0).abs() < 0.01);
    }

    #[test]
    fn decode_toks_per_sec_excludes_the_first_token_already_counted_in_ttft() {
        let s = sample(1000, 5000, 512, 200);
        assert!((s.decode_toks_per_sec() - 49.75).abs() < 0.01);
    }

    #[test]
    fn decode_toks_per_sec_is_zero_for_a_single_completion_token() {
        let s = sample(1000, 1000, 512, 1);
        assert_eq!(s.decode_toks_per_sec(), 0.0);
    }

    #[test]
    fn zero_duration_denominators_report_zero_instead_of_dividing_by_zero() {
        let s = sample(0, 0, 512, 200);
        assert_eq!(s.prefill_toks_per_sec(), 0.0);
        assert_eq!(s.decode_toks_per_sec(), 0.0);
    }

    #[test]
    fn mean_averages_every_field_across_samples() {
        let a = sample(1000, 3000, 100, 50);
        let b = sample(2000, 5000, 200, 150);
        let m = Sample::mean(&[a, b]);
        assert_eq!(m.ttft, Duration::from_millis(1500));
        assert_eq!(m.total, Duration::from_millis(4000));
        assert_eq!(m.prompt_tokens, 150);
        assert_eq!(m.completion_tokens, 100);
    }

    #[test]
    fn mean_of_a_single_sample_is_itself() {
        let a = sample(1234, 5678, 111, 222);
        let m = Sample::mean(&[a]);
        assert_eq!(m.ttft, a.ttft);
        assert_eq!(m.total, a.total);
        assert_eq!(m.prompt_tokens, a.prompt_tokens);
        assert_eq!(m.completion_tokens, a.completion_tokens);
    }

    #[test]
    fn synthetic_prompt_targets_roughly_the_requested_token_count() {
        assert_eq!(synthetic_prompt(10, 0).split(' ').count(), 7);
        assert_eq!(synthetic_prompt(0, 0).split(' ').count(), 1);
    }

    #[test]
    fn synthetic_prompt_varies_with_variant_to_defeat_kv_cache_prefix_matching() {
        let a = synthetic_prompt(50, 0);
        let b = synthetic_prompt(50, 1);
        assert_ne!(a, b);
        assert_eq!(a.split(' ').count(), b.split(' ').count());
    }

    #[test]
    fn build_prompt_prefers_prompt_tokens_over_the_literal_prompt() {
        let p = build_prompt("ignored", 10, 0);
        assert!(!p.contains("ignored"));
    }

    #[test]
    fn build_prompt_tags_the_literal_prompt_with_its_variant() {
        let p = build_prompt("hello", 0, 3);
        assert_eq!(p, "[3] hello");
    }

    #[test]
    fn epoch_retry_variants_never_collide_across_epochs() {
        // The formula `epoch * (MAX_RETRIES + 1) + attempt` (unlike a
        // fixed `epoch + attempt * 1000` stride) never repeats a
        // variant for any (epoch, attempt) pair up to a large epoch
        // count — regression test for the collision CodeRabbit flagged
        // in the original `epoch + attempt * 1000` formula.
        let mut seen = std::collections::HashSet::new();
        for epoch in 0..2000i64 {
            for attempt in 0..=MAX_SHORT_RESPONSE_RETRIES as i64 {
                let variant = epoch * (MAX_SHORT_RESPONSE_RETRIES as i64 + 1) + attempt;
                assert!(seen.insert(variant), "duplicate variant {variant}");
            }
        }
    }
}

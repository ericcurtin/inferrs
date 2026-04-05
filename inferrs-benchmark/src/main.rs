//! HTTP-based comparative benchmark: inferrs vs llama-server.
//!
//! Ports `scripts/benchmark.sh` to Rust so that benchmarks are runnable on
//! macOS, Windows **and** Linux without requiring Bash or Python.
//!
//! The benchmark:
//!   1. Starts `inferrs serve --quantize` and sends timed requests.
//!   2. Starts `inferrs serve --turbo-quant=false --quantize` and sends timed requests.
//!   3. Starts `llama-server -hf <model>` and sends timed requests.
//!   4. Prints a summary table comparing all three backends.

use std::io::BufRead;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::Parser;

// ── CLI arguments ────────────────────────────────────────────────────────────

/// Cross-platform comparative benchmark: inferrs vs llama-server.
#[derive(Parser, Clone)]
#[command(
    name = "inferrs-benchmark",
    about = "Compare inferrs vs llama-server over HTTP"
)]
struct BenchmarkArgs {
    /// Number of timed benchmark runs per backend.
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// Number of warm-up runs (results discarded).
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Approximate number of prompt tokens to generate synthetically.
    #[arg(long, default_value_t = 128)]
    prompt_len: usize,

    /// Maximum tokens to generate per request.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,

    /// Port for the `inferrs serve --quantize` backend.
    #[arg(long, default_value_t = 8080)]
    inferrs_port: u16,

    /// Port for the `inferrs serve --turbo-quant=false --quantize` backend.
    #[arg(long, default_value_t = 8082)]
    inferrs_tq_port: u16,

    /// Port for the `llama-server` backend.
    #[arg(long, default_value_t = 8181)]
    llama_port: u16,

    /// HuggingFace model ID for inferrs.
    #[arg(long, default_value = "google/gemma-4-E2B-it")]
    inferrs_model: String,

    /// GGUF model ID for llama-server.
    #[arg(long, default_value = "ggml-org/gemma-4-E2B-it-GGUF")]
    llama_model: String,

    /// Seconds to wait for a server to become healthy.
    #[arg(long, default_value_t = 120)]
    server_ready_timeout: u64,

    /// Override path to the inferrs binary.
    #[arg(long)]
    inferrs_bin: Option<PathBuf>,
}

// ── Entry point ──────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = BenchmarkArgs::parse();

    let inferrs_bin = resolve_inferrs_bin(&args);
    let prompt = generate_synthetic_prompt(args.prompt_len);

    // ── 1. inferrs serve --quantize ──────────────────────────────────────────
    log_header(&format!(
        "Benchmark 1/3 — inferrs serve --quantize {}",
        args.inferrs_model
    ));
    let summary_inferrs = {
        let mut server = start_inferrs(
            &inferrs_bin,
            &args.inferrs_model,
            args.inferrs_port,
            &["--quantize"],
        )?;
        ok(&format!(
            "inferrs serve --quantize started (pid {})",
            server.id()
        ));

        let health = format!("http://127.0.0.1:{}/health", args.inferrs_port);
        if let Err(e) = wait_for_health(&health, args.server_ready_timeout) {
            err(&format!("inferrs serve --quantize failed to start: {e}"));
            let _ = server.kill();
            let _ = server.wait();
            bail!("server failed to start");
        }

        let mut tracker = PeakMemoryTracker::start(server.id());
        let summary_res = bench_http(
            "127.0.0.1",
            args.inferrs_port,
            args.warmup,
            args.runs,
            args.max_tokens,
            &prompt,
        );
        let peak_mem_mb = tracker.stop();

        let _ = server.kill();
        let _ = server.wait();
        ok("inferrs serve --quantize stopped");
        let mut summary = summary_res?;
        summary.peak_mem_mb = peak_mem_mb;
        summary
    };

    // ── 2. inferrs serve --turbo-quant=false --quantize ─────────────────────
    log_header(&format!(
        "Benchmark 2/3 — inferrs serve --turbo-quant=false --quantize {}",
        args.inferrs_model
    ));
    let summary_inferrs_tq = {
        let mut server = start_inferrs(
            &inferrs_bin,
            &args.inferrs_model,
            args.inferrs_tq_port,
            &["--turbo-quant=false", "--quantize"],
        )?;
        ok(&format!(
            "inferrs serve --turbo-quant=false --quantize started (pid {})",
            server.id()
        ));

        let health = format!("http://127.0.0.1:{}/health", args.inferrs_tq_port);
        if let Err(e) = wait_for_health(&health, args.server_ready_timeout) {
            err(&format!(
                "inferrs serve --turbo-quant=false --quantize failed to start: {e}"
            ));
            let _ = server.kill();
            let _ = server.wait();
            bail!("server failed to start");
        }

        let mut tracker = PeakMemoryTracker::start(server.id());
        let summary_res = bench_http(
            "127.0.0.1",
            args.inferrs_tq_port,
            args.warmup,
            args.runs,
            args.max_tokens,
            &prompt,
        );
        let peak_mem_mb = tracker.stop();

        let _ = server.kill();
        let _ = server.wait();
        ok("inferrs serve --turbo-quant=false --quantize stopped");
        let mut summary = summary_res?;
        summary.peak_mem_mb = peak_mem_mb;
        summary
    };

    // ── 3. llama-server ─────────────────────────────────────────────────────
    log_header(&format!(
        "Benchmark 3/3 — llama-server -hf {}",
        args.llama_model
    ));
    let summary_llama = {
        let mut server = start_llama_server(&args.llama_model, args.llama_port)?;
        ok(&format!("llama-server started (pid {})", server.id()));

        let health = format!("http://127.0.0.1:{}/health", args.llama_port);
        if let Err(e) = wait_for_health(&health, args.server_ready_timeout) {
            err(&format!("llama-server failed to start: {e}"));
            let _ = server.kill();
            let _ = server.wait();
            bail!("server failed to start");
        }

        let mut tracker = PeakMemoryTracker::start(server.id());
        let summary_res = bench_http(
            "127.0.0.1",
            args.llama_port,
            args.warmup,
            args.runs,
            args.max_tokens,
            &prompt,
        );
        let peak_mem_mb = tracker.stop();

        let _ = server.kill();
        let _ = server.wait();
        ok("llama-server stopped");
        let mut summary = summary_res?;
        summary.peak_mem_mb = peak_mem_mb;
        summary
    };

    // ── Summary table ───────────────────────────────────────────────────────
    log_header("Results");
    print_summary(
        &args,
        Some(&summary_llama),
        Some(&summary_inferrs),
        Some(&summary_inferrs_tq),
    );

    Ok(())
}

// ── Server management ────────────────────────────────────────────────────────

/// Resolve the inferrs binary path: explicit override → sibling of the current
/// executable → `inferrs` on PATH.
fn resolve_inferrs_bin(args: &BenchmarkArgs) -> PathBuf {
    if let Some(ref bin) = args.inferrs_bin {
        return bin.clone();
    }

    // Look for an `inferrs` binary next to this executable.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sibling = dir.join("inferrs");
            if sibling.is_file() {
                return sibling;
            }
            // Windows: try with .exe extension.
            let sibling_exe = dir.join("inferrs.exe");
            if sibling_exe.is_file() {
                return sibling_exe;
            }
        }
    }

    PathBuf::from("inferrs")
}

fn start_inferrs(bin: &PathBuf, model: &str, port: u16, extra_args: &[&str]) -> Result<Child> {
    let mut cmd = Command::new(bin);
    cmd.arg("serve")
        .arg(model)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string());
    for arg in extra_args {
        cmd.arg(arg);
    }
    cmd.stdout(Stdio::null()).stderr(Stdio::null());
    let child = cmd
        .spawn()
        .with_context(|| format!("failed to start inferrs: {}", bin.display()))?;
    Ok(child)
}

fn start_llama_server(model: &str, port: u16) -> Result<Child> {
    let child = Command::new("llama-server")
        .arg("-hf")
        .arg(model)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to start llama-server (is it on PATH?)")?;
    Ok(child)
}

fn wait_for_health(url: &str, timeout_secs: u64) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    eprint!("    Waiting for {url} ");

    loop {
        match ureq::get(url).timeout(Duration::from_secs(5)).call() {
            Ok(resp) if resp.status() == 200 => {
                eprintln!(" ready");
                return Ok(());
            }
            _ => {}
        }

        if Instant::now() >= deadline {
            eprintln!(" TIMEOUT");
            bail!("timed out waiting for {url} after {timeout_secs}s");
        }

        eprint!(".");
        std::thread::sleep(Duration::from_secs(2));
    }
}

// ── Synthetic prompt generation ──────────────────────────────────────────────

/// Build a synthetic prompt of approximately `token_count` tokens.
/// Heuristic: ~4 chars per token, cycle through a small vocabulary.
fn generate_synthetic_prompt(token_count: usize) -> String {
    const WORDS: &[&str] = &[
        "The",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "a",
        "lazy",
        "dog",
        "machine",
        "learning",
        "model",
        "performance",
        "benchmark",
        "inference",
        "speed",
        "latency",
        "throughput",
        "token",
        "generation",
        "prefill",
        "decode",
        "attention",
        "transformer",
        "neural",
        "network",
        "parameter",
    ];

    let mut out = String::new();
    for i in 0..token_count {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(WORDS[i % WORDS.len()]);
    }
    // Truncate to ~4 chars per token.
    let max_chars = token_count * 4;
    if out.len() > max_chars {
        out.truncate(max_chars);
    }
    out
}

// ── HTTP benchmark core ──────────────────────────────────────────────────────

/// Summary statistics for one backend.
struct BenchSummary {
    ttft_ms: Option<f64>,
    prefill_tps: Option<f64>,
    decode_tps: Option<f64>,
    peak_mem_mb: Option<f64>,
}

// ── Peak memory tracker ──────────────────────────────────────────────────────

/// Polls the RSS of a process in a background thread and tracks the peak.
/// Call `stop()` to halt polling and retrieve the peak value in MB.
struct PeakMemoryTracker {
    peak_kb: Arc<AtomicU64>,
    stop_flag: Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl PeakMemoryTracker {
    fn start(pid: u32) -> Self {
        let peak_kb = Arc::new(AtomicU64::new(0));
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let peak_kb2 = Arc::clone(&peak_kb);
        let stop2 = Arc::clone(&stop_flag);

        let thread = std::thread::spawn(move || {
            use sysinfo::{Pid, System};
            let mut sys = System::new();
            let sysinfo_pid = Pid::from_u32(pid);
            while !stop2.load(Ordering::Relaxed) {
                sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[sysinfo_pid]), true);
                if let Some(proc) = sys.process(sysinfo_pid) {
                    let kb = proc.memory() / 1024;
                    peak_kb2.fetch_max(kb, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        });

        Self {
            peak_kb,
            stop_flag,
            thread: Some(thread),
        }
    }

    fn stop(&mut self) -> Option<f64> {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        let kb = self.peak_kb.load(Ordering::Relaxed);
        if kb == 0 {
            None
        } else {
            Some(kb as f64 / 1024.0)
        }
    }
}

/// Run the HTTP benchmark against one server.  Returns aggregated metrics.
fn bench_http(
    host: &str,
    port: u16,
    warmup: usize,
    runs: usize,
    max_tokens: usize,
    prompt: &str,
) -> Result<BenchSummary> {
    let url = format!("http://{host}:{port}/v1/chat/completions");

    // Non-streaming probe to get the true prompt token count.
    let n_prompt = {
        let body = serde_json::json!({
            "model": "benchmark",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": false,
        });
        let resp: serde_json::Value = ureq::post(&url)
            .timeout(Duration::from_secs(300))
            .send_json(&body)?
            .into_json()?;
        resp.get("usage")
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize
    };

    let total = warmup + runs;
    let mut ttfts: Vec<f64> = Vec::with_capacity(runs);
    let mut prefills: Vec<f64> = Vec::with_capacity(runs);
    let mut decodes: Vec<f64> = Vec::with_capacity(runs);

    for i in 0..total {
        let is_warmup = i < warmup;
        let (ttft_ms, total_ms, n_gen) = do_stream(&url, prompt, max_tokens)?;

        let decode_ms = (total_ms - ttft_ms).max(1e-6);
        let prefill_tps = if ttft_ms > 0.0 {
            n_prompt as f64 / ttft_ms * 1000.0
        } else {
            0.0
        };
        let decode_tps = if decode_ms > 0.0 {
            n_gen as f64 / decode_ms * 1000.0
        } else {
            0.0
        };

        if is_warmup {
            println!(
                "  [warmup] TTFT={ttft_ms:.1}ms  prefill={prefill_tps:.1}t/s  \
                 decode={decode_tps:.1}t/s  (n_prompt={n_prompt}, n_gen={n_gen})"
            );
        } else {
            let run_num = i - warmup + 1;
            ttfts.push(ttft_ms);
            prefills.push(prefill_tps);
            decodes.push(decode_tps);
            println!(
                "  [run {run_num}/{runs}] TTFT={ttft_ms:.1}ms  prefill={prefill_tps:.1}t/s  \
                 decode={decode_tps:.1}t/s  (n_prompt={n_prompt}, n_gen={n_gen})"
            );
        }
    }

    println!();
    println!("  TTFT    : {}", stats(&ttfts, "ms"));
    println!("  Prefill : {}", stats(&prefills, "tok/s"));
    println!("  Decode  : {}", stats(&decodes, "tok/s"));

    Ok(BenchSummary {
        ttft_ms: mean_or_none(&ttfts),
        prefill_tps: mean_or_none(&prefills),
        decode_tps: mean_or_none(&decodes),
        peak_mem_mb: None,
    })
}

/// Stream one chat-completion request; returns (ttft_ms, total_ms, n_gen).
fn do_stream(url: &str, prompt: &str, max_tokens: usize) -> Result<(f64, f64, usize)> {
    let body = serde_json::json!({
        "model": "benchmark",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": true,
    });

    let t0 = Instant::now();
    let resp = ureq::post(url)
        .timeout(Duration::from_secs(300))
        .send_json(&body)?;

    let reader = std::io::BufReader::new(resp.into_reader());
    let mut ttft_ms: Option<f64> = None;
    let mut n_gen: usize = 0;

    for line in reader.lines() {
        let line = line?;
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }

        let chunk: serde_json::Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let choices = chunk.get("choices").and_then(|c| c.as_array());
        let Some(choices) = choices else { continue };
        if choices.is_empty() {
            continue;
        }

        let delta = choices[0].get("delta");
        let Some(delta) = delta else { continue };

        // Count any generated token: content or reasoning_content.
        let token = delta
            .get("content")
            .and_then(|v| v.as_str())
            .or_else(|| delta.get("reasoning_content").and_then(|v| v.as_str()))
            .unwrap_or("");

        if !token.is_empty() {
            if ttft_ms.is_none() {
                ttft_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
            }
            n_gen += 1;
        }
    }

    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = ttft_ms.unwrap_or(total_ms);
    Ok((ttft_ms, total_ms, n_gen))
}

// ── Statistics helpers ───────────────────────────────────────────────────────

fn mean_or_none(vals: &[f64]) -> Option<f64> {
    if vals.is_empty() {
        return None;
    }
    Some(vals.iter().sum::<f64>() / vals.len() as f64)
}

fn stddev(vals: &[f64]) -> f64 {
    if vals.len() < 2 {
        return 0.0;
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
    var.sqrt()
}

fn stats(vals: &[f64], unit: &str) -> String {
    match mean_or_none(vals) {
        Some(m) => format!("{m:.2} ± {:.2} {unit}", stddev(vals)),
        None => format!("N/A {unit}"),
    }
}

// ── Summary table ────────────────────────────────────────────────────────────

type SummaryRow = (String, Option<f64>, Option<f64>, Option<f64>, Option<f64>);

fn print_summary(
    args: &BenchmarkArgs,
    llama: Option<&BenchSummary>,
    inferrs: Option<&BenchSummary>,
    inferrs_tq: Option<&BenchSummary>,
) {
    fn fmt(v: Option<f64>, unit: &str) -> String {
        match v {
            Some(val) => format!("{val:.2} {unit}"),
            None => "N/A".to_string(),
        }
    }

    let rows: Vec<SummaryRow> = vec![
        (
            format!("llama-server -hf {}", args.llama_model),
            llama.and_then(|s| s.ttft_ms),
            llama.and_then(|s| s.prefill_tps),
            llama.and_then(|s| s.decode_tps),
            llama.and_then(|s| s.peak_mem_mb),
        ),
        (
            format!("inferrs serve --quantize {}", args.inferrs_model),
            inferrs.and_then(|s| s.ttft_ms),
            inferrs.and_then(|s| s.prefill_tps),
            inferrs.and_then(|s| s.decode_tps),
            inferrs.and_then(|s| s.peak_mem_mb),
        ),
        (
            format!(
                "inferrs serve --turbo-quant=false --quantize {}",
                args.inferrs_model
            ),
            inferrs_tq.and_then(|s| s.ttft_ms),
            inferrs_tq.and_then(|s| s.prefill_tps),
            inferrs_tq.and_then(|s| s.decode_tps),
            inferrs_tq.and_then(|s| s.peak_mem_mb),
        ),
    ];

    let w = rows
        .iter()
        .map(|(name, _, _, _, _)| name.len())
        .max()
        .unwrap_or(0)
        .max("Backend".len());
    println!();
    println!(
        "Benchmark settings: prompt_len={} tokens, max_tokens={}, runs={}, warmup={}",
        args.prompt_len, args.max_tokens, args.runs, args.warmup
    );
    println!();
    println!(
        "{:<w$}  {:>12}  {:>14}  {:>13}  {:>14}",
        "Backend",
        "TTFT (ms)",
        "Prefill (t/s)",
        "Decode (t/s)",
        "Peak mem (MB)",
        w = w
    );
    println!("{}", "-".repeat(w + 61));
    for (name, ttft, pfill, dec, mem) in &rows {
        println!(
            "{:<w$}  {:>12}  {:>14}  {:>13}  {:>14}",
            name,
            fmt(*ttft, "ms"),
            fmt(*pfill, "t/s"),
            fmt(*dec, "t/s"),
            fmt(*mem, "MB"),
            w = w
        );
    }
    println!();

    // Relative comparison vs llama-server.
    let base_ttft = llama.and_then(|s| s.ttft_ms);
    let base_pfill = llama.and_then(|s| s.prefill_tps);
    let base_dec = llama.and_then(|s| s.decode_tps);
    let base_mem = llama.and_then(|s| s.peak_mem_mb);

    if let (Some(bt), Some(bp), Some(bd)) = (base_ttft, base_pfill, base_dec) {
        println!(
            "Relative to llama-server (higher prefill/decode is better; lower TTFT/mem is better):"
        );
        for (name, ttft, pfill, dec, mem) in &rows[1..] {
            if let (Some(t), Some(p), Some(d)) = (ttft, pfill, dec) {
                let d_ttft = (t - bt) / bt * 100.0;
                let d_pfill = (p - bp) / bp * 100.0;
                let d_dec = (d - bd) / bd * 100.0;
                let sign = |x: f64| if x >= 0.0 { "+" } else { "" };
                println!("  {name}");
                println!("    TTFT:     {}{d_ttft:.1}%", sign(d_ttft));
                println!("    Prefill:  {}{d_pfill:.1}%", sign(d_pfill));
                println!("    Decode:   {}{d_dec:.1}%", sign(d_dec));
                if let (Some(m), Some(bm)) = (mem, base_mem) {
                    let d_mem = (m - bm) / bm * 100.0;
                    println!("    Peak mem: {}{d_mem:.1}%", sign(d_mem));
                }
            }
        }
    }
}

// ── Logging helpers ──────────────────────────────────────────────────────────

fn log_header(msg: &str) {
    eprintln!("\n\x1b[1;34m==> {msg}\x1b[0m");
}

fn ok(msg: &str) {
    eprintln!("\x1b[1;32m[ok]\x1b[0m {msg}");
}

fn err(msg: &str) {
    eprintln!("\x1b[1;31m[err]\x1b[0m {msg}");
}

//! HTTP-based comparative benchmark: inferrs vs llama-server / vllm.
//!
//! Ports `scripts/benchmark.sh` to Rust so that benchmarks are runnable on
//! macOS, Windows **and** Linux without requiring Bash or Python.
//!
//! Default benchmark (3 backends):
//!   1. Starts `inferrs serve --quantize` and sends timed requests.
//!   2. Starts `inferrs serve --turbo-quant=false --quantize` and sends timed requests.
//!   3. Starts `llama-server -hf <model>` and sends timed requests.
//!   4. Prints a summary table comparing all three backends.
//!
//! DGX Spark benchmark (`--dgx-spark`), runs 5 groups:
//!   Group 1: llama-server 31B GGUF  vs  inferrs --quantize nvidia/Gemma-4-31B-IT-NVFP4
//!   Group 2: llama-server 31B GGUF  vs  inferrs --quantize google/gemma-4-31B-it
//!   Group 3: vllm google/gemma-4-31B-it  vs  inferrs --paged-attention google/gemma-4-31B-it
//!   Group 4: vllm nvidia/Gemma-4-31B-IT-NVFP4  vs  inferrs --paged-attention nvidia/Gemma-4-31B-IT-NVFP4
//!   Group 5: vllm google/gemma-4-E2B-it  vs  inferrs --paged-attention google/gemma-4-E2B-it

use std::io::BufRead;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::Parser;

// ── CLI arguments ────────────────────────────────────────────────────────────

/// Cross-platform comparative benchmark: inferrs vs llama-server / vllm.
#[derive(Parser, Clone)]
#[command(
    name = "inferrs-benchmark",
    about = "Compare inferrs vs llama-server / vllm over HTTP"
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

    /// Port for the `vllm serve` backend (DGX Spark benchmarks).
    #[arg(long, default_value_t = 8000)]
    vllm_port: u16,

    /// HuggingFace model ID for inferrs.
    #[arg(long, default_value = "google/gemma-4-E2B-it")]
    inferrs_model: String,

    /// GGUF model ID for llama-server.
    #[arg(long, default_value = "ggml-org/gemma-4-E2B-it-GGUF")]
    llama_model: String,

    /// Seconds to wait for a server to become healthy.
    #[arg(long, default_value_t = 1800)]
    server_ready_timeout: u64,

    /// Override path to the inferrs binary.
    #[arg(long)]
    inferrs_bin: Option<PathBuf>,

    /// Run the full DGX Spark benchmark suite (5 groups covering 31B and 2B
    /// models with llama-server, vllm, and inferrs --paged-attention backends).
    #[arg(long)]
    dgx_spark: bool,
}

// ── Entry point ──────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = BenchmarkArgs::parse();

    if args.dgx_spark {
        return run_dgx_spark(&args);
    }

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
        let t_bench = Instant::now();
        let summary_res = bench_http(
            "127.0.0.1",
            args.inferrs_port,
            args.warmup,
            args.runs,
            args.max_tokens,
            &args.inferrs_model,
            &prompt,
        );
        let elapsed = t_bench.elapsed();
        let peak_mem_mb = tracker.stop();

        let _ = server.kill();
        let _ = server.wait();
        ok(&format!(
            "inferrs serve --quantize stopped  (benchmark took {:.1}s)",
            elapsed.as_secs_f64()
        ));
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
        let t_bench = Instant::now();
        let summary_res = bench_http(
            "127.0.0.1",
            args.inferrs_tq_port,
            args.warmup,
            args.runs,
            args.max_tokens,
            &args.inferrs_model,
            &prompt,
        );
        let elapsed = t_bench.elapsed();
        let peak_mem_mb = tracker.stop();

        let _ = server.kill();
        let _ = server.wait();
        ok(&format!(
            "inferrs serve --turbo-quant=false --quantize stopped  (benchmark took {:.1}s)",
            elapsed.as_secs_f64()
        ));
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
        let t_bench = Instant::now();
        let summary_res = bench_http(
            "127.0.0.1",
            args.llama_port,
            args.warmup,
            args.runs,
            args.max_tokens,
            &args.llama_model,
            &prompt,
        );
        let elapsed = t_bench.elapsed();
        let peak_mem_mb = tracker.stop();

        let _ = server.kill();
        let _ = server.wait();
        ok(&format!(
            "llama-server stopped  (benchmark took {:.1}s)",
            elapsed.as_secs_f64()
        ));
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

// ── DGX Spark benchmark suite ────────────────────────────────────────────────

/// Run one benchmark backend and return its summary.
///
/// `label` is used for log messages only; `start_fn` spawns the server
/// process; `port` is the port to hit.
fn run_one_backend<F>(
    args: &BenchmarkArgs,
    label: &str,
    port: u16,
    model: &str,
    prompt: &str,
    start_fn: F,
) -> Result<BenchSummary>
where
    F: FnOnce() -> Result<Child>,
{
    run_one_backend_with_cleanup(args, label, port, model, prompt, move || {
        Ok((start_fn()?, None))
    })
}

/// Like [`run_one_backend`] but also accepts an optional cleanup callback that
/// is invoked before waiting for the child to exit.  This is used for Docker
/// backends where killing the client process does not stop the container.
fn run_one_backend_with_cleanup<F>(
    args: &BenchmarkArgs,
    label: &str,
    port: u16,
    model: &str,
    prompt: &str,
    start_fn: F,
) -> Result<BenchSummary>
where
    F: FnOnce() -> Result<(Child, Option<Box<dyn FnOnce()>>)>,
{
    log_header(label);
    let (mut server, cleanup) = start_fn()?;
    ok(&format!("{label} started (pid {})", server.id()));

    let health = format!("http://127.0.0.1:{port}/health");
    if let Err(e) = wait_for_health(&health, args.server_ready_timeout) {
        err(&format!("{label} failed to start: {e}"));
        if let Some(f) = cleanup {
            f();
        }
        let _ = server.kill();
        let _ = server.wait();
        bail!("server failed to start");
    }

    let mut tracker = PeakMemoryTracker::start(server.id());
    let t_bench = Instant::now();
    let summary_res = bench_http(
        "127.0.0.1",
        port,
        args.warmup,
        args.runs,
        args.max_tokens,
        model,
        prompt,
    );
    let elapsed = t_bench.elapsed();
    let peak_mem_mb = tracker.stop();

    // For Docker-based backends the cleanup callback issues `docker stop`,
    // which terminates the container.  The `--rm` flag on `docker run` ensures
    // the container is removed once it exits.  We then kill/wait the client
    // process to reap it.
    if let Some(f) = cleanup {
        f();
    }
    let _ = server.kill();
    let _ = server.wait();
    ok(&format!(
        "{label} stopped  (benchmark took {:.1}s)",
        elapsed.as_secs_f64()
    ));

    let mut summary = summary_res?;
    summary.peak_mem_mb = peak_mem_mb;
    Ok(summary)
}

/// DGX Spark full benchmark suite.
///
/// Runs five groups sequentially. Within each group the reference backend
/// (llama-server or vllm) is the baseline for relative comparisons.
fn run_dgx_spark(args: &BenchmarkArgs) -> Result<()> {
    let inferrs_bin = resolve_inferrs_bin(args);
    let prompt = generate_synthetic_prompt(args.prompt_len);

    // Ports assigned per group to avoid conflicts. We reuse the same three
    // slots the default benchmark uses; servers are killed before the next
    // group starts so there is no overlap.
    let p_ref = args.llama_port; // reference backend (llama-server, groups 1-2)
    let p_vllm = args.vllm_port; // reference backend (vllm, groups 3-5)
    let p_inf = args.inferrs_port; // inferrs (turbo-quant on)
    let p_tq = args.inferrs_tq_port; // inferrs (turbo-quant off)

    // ── Group 1: llama-server 31B GGUF  vs  inferrs --quantize NVFP4 ────────
    log_header("DGX Spark group 1/5 — llama-server 31B GGUF vs inferrs --quantize NVFP4");
    let g1_llama = run_one_backend(
        args,
        "llama-server -hf ggml-org/gemma-4-31B-it-GGUF",
        p_ref,
        "ggml-org/gemma-4-31B-it-GGUF",
        &prompt,
        || start_llama_server("ggml-org/gemma-4-31B-it-GGUF", p_ref),
    )?;
    let g1_inferrs = run_one_backend(
        args,
        "inferrs serve --quantize nvidia/Gemma-4-31B-IT-NVFP4",
        p_inf,
        "nvidia/Gemma-4-31B-IT-NVFP4",
        &prompt,
        || {
            start_inferrs(
                &inferrs_bin,
                "nvidia/Gemma-4-31B-IT-NVFP4",
                p_inf,
                &["--quantize"],
            )
        },
    )?;
    let g1_inferrs_tq = run_one_backend(
        args,
        "inferrs serve --turbo-quant=false --quantize nvidia/Gemma-4-31B-IT-NVFP4",
        p_tq,
        "nvidia/Gemma-4-31B-IT-NVFP4",
        &prompt,
        || {
            start_inferrs(
                &inferrs_bin,
                "nvidia/Gemma-4-31B-IT-NVFP4",
                p_tq,
                &["--turbo-quant=false", "--quantize"],
            )
        },
    )?;

    // ── Group 2: llama-server 31B GGUF  vs  inferrs --quantize google 31B ───
    log_header("DGX Spark group 2/5 — llama-server 31B GGUF vs inferrs --quantize google 31B");
    let g2_llama = run_one_backend(
        args,
        "llama-server -hf ggml-org/gemma-4-31B-it-GGUF",
        p_ref,
        "ggml-org/gemma-4-31B-it-GGUF",
        &prompt,
        || start_llama_server("ggml-org/gemma-4-31B-it-GGUF", p_ref),
    )?;
    let g2_inferrs = run_one_backend(
        args,
        "inferrs serve --quantize google/gemma-4-31B-it",
        p_inf,
        "google/gemma-4-31B-it",
        &prompt,
        || {
            start_inferrs(
                &inferrs_bin,
                "google/gemma-4-31B-it",
                p_inf,
                &["--quantize"],
            )
        },
    )?;
    let g2_inferrs_tq = run_one_backend(
        args,
        "inferrs serve --turbo-quant=false --quantize google/gemma-4-31B-it",
        p_tq,
        "google/gemma-4-31B-it",
        &prompt,
        || {
            start_inferrs(
                &inferrs_bin,
                "google/gemma-4-31B-it",
                p_tq,
                &["--turbo-quant=false", "--quantize"],
            )
        },
    )?;

    // ── Group 3: vllm 31B google  vs  inferrs --paged-attention 31B google ──
    log_header("DGX Spark group 3/5 — vllm google 31B vs inferrs --paged-attention google 31B");
    let g3_vllm = run_one_backend_with_cleanup(
        args,
        "vllm serve google/gemma-4-31B-it",
        p_vllm,
        "google/gemma-4-31B-it",
        &prompt,
        || {
            let (child, name) = start_vllm_server("google/gemma-4-31B-it", p_vllm)?;
            let cleanup: Box<dyn FnOnce()> = Box::new(move || stop_docker_container(&name));
            Ok((child, Some(cleanup)))
        },
    )?;
    let g3_inferrs = run_one_backend(
        args,
        "inferrs serve --paged-attention google/gemma-4-31B-it",
        p_inf,
        "google/gemma-4-31B-it",
        &prompt,
        || {
            start_inferrs(
                &inferrs_bin,
                "google/gemma-4-31B-it",
                p_inf,
                &["--paged-attention"],
            )
        },
    )?;
    let g3_inferrs_tq = run_one_backend(
        args,
        "inferrs serve --turbo-quant=false --paged-attention google/gemma-4-31B-it",
        p_tq,
        "google/gemma-4-31B-it",
        &prompt,
        || {
            start_inferrs(
                &inferrs_bin,
                "google/gemma-4-31B-it",
                p_tq,
                &["--turbo-quant=false", "--paged-attention"],
            )
        },
    )?;

    // ── Group 4: vllm 31B NVFP4  vs  inferrs --paged-attention 31B NVFP4 ───
    log_header("DGX Spark group 4/5 — vllm nvidia 31B vs inferrs --paged-attention nvidia 31B");
    let g4_vllm = run_one_backend_with_cleanup(
        args,
        "vllm serve nvidia/Gemma-4-31B-IT-NVFP4",
        p_vllm,
        "nvidia/Gemma-4-31B-IT-NVFP4",
        &prompt,
        || {
            let (child, name) = start_vllm_server("nvidia/Gemma-4-31B-IT-NVFP4", p_vllm)?;
            let cleanup: Box<dyn FnOnce()> = Box::new(move || stop_docker_container(&name));
            Ok((child, Some(cleanup)))
        },
    )?;
    let g4_inferrs = run_one_backend(
        args,
        "inferrs serve --paged-attention nvidia/Gemma-4-31B-IT-NVFP4",
        p_inf,
        "nvidia/Gemma-4-31B-IT-NVFP4",
        &prompt,
        || {
            start_inferrs(
                &inferrs_bin,
                "nvidia/Gemma-4-31B-IT-NVFP4",
                p_inf,
                &["--paged-attention"],
            )
        },
    )?;
    let g4_inferrs_tq = run_one_backend(
        args,
        "inferrs serve --turbo-quant=false --paged-attention nvidia/Gemma-4-31B-IT-NVFP4",
        p_tq,
        "nvidia/Gemma-4-31B-IT-NVFP4",
        &prompt,
        || {
            start_inferrs(
                &inferrs_bin,
                "nvidia/Gemma-4-31B-IT-NVFP4",
                p_tq,
                &["--turbo-quant=false", "--paged-attention"],
            )
        },
    )?;

    // ── Group 5: vllm 2B google  vs  inferrs --paged-attention 2B google ────
    log_header("DGX Spark group 5/5 — vllm google 2B vs inferrs --paged-attention google 2B");
    let g5_vllm = run_one_backend_with_cleanup(
        args,
        "vllm serve google/gemma-4-E2B-it",
        p_vllm,
        "google/gemma-4-E2B-it",
        &prompt,
        || {
            let (child, name) = start_vllm_server("google/gemma-4-E2B-it", p_vllm)?;
            let cleanup: Box<dyn FnOnce()> = Box::new(move || stop_docker_container(&name));
            Ok((child, Some(cleanup)))
        },
    )?;
    let g5_inferrs = run_one_backend(
        args,
        "inferrs serve --paged-attention google/gemma-4-E2B-it",
        p_inf,
        "google/gemma-4-E2B-it",
        &prompt,
        || {
            start_inferrs(
                &inferrs_bin,
                "google/gemma-4-E2B-it",
                p_inf,
                &["--paged-attention"],
            )
        },
    )?;
    let g5_inferrs_tq = run_one_backend(
        args,
        "inferrs serve --turbo-quant=false --paged-attention google/gemma-4-E2B-it",
        p_tq,
        "google/gemma-4-E2B-it",
        &prompt,
        || {
            start_inferrs(
                &inferrs_bin,
                "google/gemma-4-E2B-it",
                p_tq,
                &["--turbo-quant=false", "--paged-attention"],
            )
        },
    )?;

    // ── Combined results ─────────────────────────────────────────────────────
    log_header("DGX Spark Results — All Groups");
    println!(
        "\nBenchmark settings: prompt_len={} tokens, max_tokens={}, runs={}, warmup={}\n",
        args.prompt_len, args.max_tokens, args.runs, args.warmup
    );

    print_dgx_group(
        "Group 1 — GGUF 31B: llama-server vs inferrs --quantize nvidia NVFP4",
        "llama-server -hf ggml-org/gemma-4-31B-it-GGUF",
        &g1_llama,
        "inferrs serve --quantize nvidia/Gemma-4-31B-IT-NVFP4",
        &g1_inferrs,
        "inferrs serve --turbo-quant=false --quantize nvidia/Gemma-4-31B-IT-NVFP4",
        &g1_inferrs_tq,
    );

    print_dgx_group(
        "Group 2 — GGUF 31B: llama-server vs inferrs --quantize google 31B",
        "llama-server -hf ggml-org/gemma-4-31B-it-GGUF",
        &g2_llama,
        "inferrs serve --quantize google/gemma-4-31B-it",
        &g2_inferrs,
        "inferrs serve --turbo-quant=false --quantize google/gemma-4-31B-it",
        &g2_inferrs_tq,
    );

    print_dgx_group(
        "Group 3 — Paged attn 31B: vllm vs inferrs google 31B",
        "vllm serve google/gemma-4-31B-it",
        &g3_vllm,
        "inferrs serve --paged-attention google/gemma-4-31B-it",
        &g3_inferrs,
        "inferrs serve --turbo-quant=false --paged-attention google/gemma-4-31B-it",
        &g3_inferrs_tq,
    );
    println!("  Note: vllm peak mem reflects the docker client process, not the container.");

    print_dgx_group(
        "Group 4 — Paged attn 31B: vllm vs inferrs nvidia NVFP4",
        "vllm serve nvidia/Gemma-4-31B-IT-NVFP4",
        &g4_vllm,
        "inferrs serve --paged-attention nvidia/Gemma-4-31B-IT-NVFP4",
        &g4_inferrs,
        "inferrs serve --turbo-quant=false --paged-attention nvidia/Gemma-4-31B-IT-NVFP4",
        &g4_inferrs_tq,
    );
    println!("  Note: vllm peak mem reflects the docker client process, not the container.");

    print_dgx_group(
        "Group 5 — Paged attn 2B: vllm vs inferrs google 2B",
        "vllm serve google/gemma-4-E2B-it",
        &g5_vllm,
        "inferrs serve --paged-attention google/gemma-4-E2B-it",
        &g5_inferrs,
        "inferrs serve --turbo-quant=false --paged-attention google/gemma-4-E2B-it",
        &g5_inferrs_tq,
    );
    println!("  Note: vllm peak mem reflects the docker client process, not the container.");

    Ok(())
}

/// Print a summary table for one DGX Spark group, with relative comparison vs
/// the reference backend.
#[allow(clippy::too_many_arguments)]
fn print_dgx_group(
    title: &str,
    ref_name: &str,
    ref_summary: &BenchSummary,
    inferrs_name: &str,
    inferrs_summary: &BenchSummary,
    inferrs_tq_name: &str,
    inferrs_tq_summary: &BenchSummary,
) {
    fn fmt(v: Option<f64>, unit: &str) -> String {
        match v {
            Some(val) => format!("{val:.2} {unit}"),
            None => "N/A".to_string(),
        }
    }

    type Row = (String, Option<f64>, Option<f64>, Option<f64>, Option<f64>);
    let rows: Vec<Row> = vec![
        (
            ref_name.to_string(),
            ref_summary.ttft_ms,
            ref_summary.prefill_tps,
            ref_summary.decode_tps,
            ref_summary.peak_mem_mb,
        ),
        (
            inferrs_name.to_string(),
            inferrs_summary.ttft_ms,
            inferrs_summary.prefill_tps,
            inferrs_summary.decode_tps,
            inferrs_summary.peak_mem_mb,
        ),
        (
            inferrs_tq_name.to_string(),
            inferrs_tq_summary.ttft_ms,
            inferrs_tq_summary.prefill_tps,
            inferrs_tq_summary.decode_tps,
            inferrs_tq_summary.peak_mem_mb,
        ),
    ];

    const W_TTFT: usize = 12;
    const W_PFILL: usize = 14;
    const W_DEC: usize = 13;
    const W_MEM: usize = 14;
    let w = rows
        .iter()
        .map(|(name, _, _, _, _)| name.len())
        .max()
        .unwrap_or(0)
        .max("Backend".len());
    let total_w = w + 2 + W_TTFT + 2 + W_PFILL + 2 + W_DEC + 2 + W_MEM;

    println!("\n{title}");
    println!("{}", "═".repeat(total_w));
    println!(
        "{:<w$}  {:>W_TTFT$}  {:>W_PFILL$}  {:>W_DEC$}  {:>W_MEM$}",
        "Backend", "TTFT (ms)", "Prefill (t/s)", "Decode (t/s)", "Peak mem (MB)",
    );
    println!("{}", "─".repeat(total_w));
    for (name, ttft, pfill, dec, mem) in &rows {
        println!(
            "{:<w$}  {:>W_TTFT$}  {:>W_PFILL$}  {:>W_DEC$}  {:>W_MEM$}",
            name,
            fmt(*ttft, "ms"),
            fmt(*pfill, "t/s"),
            fmt(*dec, "t/s"),
            fmt(*mem, "MB"),
        );
    }
    println!("{}", "═".repeat(total_w));

    // Relative comparison vs reference backend.
    let base_ttft = ref_summary.ttft_ms;
    let base_pfill = ref_summary.prefill_tps;
    let base_dec = ref_summary.decode_tps;
    let base_mem = ref_summary.peak_mem_mb;

    if let (Some(bt), Some(bp), Some(bd)) = (base_ttft, base_pfill, base_dec) {
        println!(
            "\n  Relative to {ref_name} (higher prefill/decode is better; lower TTFT/mem is better):"
        );
        for (name, ttft, pfill, dec, mem) in &rows[1..] {
            if let (Some(t), Some(p), Some(d)) = (ttft, pfill, dec) {
                let d_ttft = (t - bt) / bt * 100.0;
                let d_pfill = (p - bp) / bp * 100.0;
                let d_dec = (d - bd) / bd * 100.0;
                let sign = |x: f64| if x >= 0.0 { "+" } else { "" };
                println!("    {name}");
                println!("      TTFT:     {}{d_ttft:.1}%", sign(d_ttft));
                println!("      Prefill:  {}{d_pfill:.1}%", sign(d_pfill));
                println!("      Decode:   {}{d_dec:.1}%", sign(d_dec));
                if let (Some(m), Some(bm)) = (mem, base_mem) {
                    let d_mem = (m - bm) / bm * 100.0;
                    println!("      Peak mem: {}{d_mem:.1}%", sign(d_mem));
                }
            }
        }
    }
    println!();
}

// ── Server management ────────────────────────────────────────────────────────

/// Resolve the HuggingFace hub cache directory using the same precedence as
/// the HF Python library: `HUGGINGFACE_HUB_CACHE` → `$HF_HOME/hub` →
/// `$XDG_CACHE_HOME/huggingface/hub` → `~/.cache/huggingface/hub`.
fn hf_hub_cache_dir() -> String {
    if let Ok(v) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return v;
    }
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return format!("{hf_home}/hub");
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return format!("{xdg}/huggingface/hub");
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| "/root".to_string());
    format!("{home}/.cache/huggingface/hub")
}

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

/// Start a vllm container and return the `docker run` child process together
/// with the container name so the caller can stop the container explicitly.
///
/// Killing the `docker run` client process does **not** stop the container;
/// the caller must run `docker stop <name>` (or call [`stop_docker_container`])
/// to actually terminate the workload and free GPU memory.
///
/// Memory tracking limitation: the Child returned here is the `docker run`
/// client process (lightweight), not the container workload.
/// PeakMemoryTracker will therefore report near-zero memory for vllm groups;
/// the benchmark output notes this explicitly.
fn start_vllm_server(model: &str, port: u16) -> Result<(Child, String)> {
    // On DGX Spark vllm must run inside its Docker image because the system
    // CUDA stack is too new for the stock vllm wheel.  We mount the host
    // HuggingFace cache read-only so the container uses already-downloaded
    // weights without re-pulling.
    let hf_hub_cache = hf_hub_cache_dir();
    let volume = format!("{hf_hub_cache}:/root/.cache/huggingface/hub");

    // Give the container a unique, predictable name so we can `docker stop` it
    // by name regardless of what happens to the client process.
    let container_name = format!("inferrs-bench-vllm-{port}");

    // Pre-emptively remove any stale container with the same name.  If a
    // previous run crashed before `--rm` could clean up, `docker run --name`
    // would fail immediately and `wait_for_health` would then spin for the
    // full server_ready_timeout before giving up.  `docker rm -f` is a no-op
    // when the container does not exist, so this is always safe.
    stop_docker_container(&container_name);

    let port_mapping = format!("{port}:8000");
    let child = Command::new("docker")
        .arg("run")
        .arg("--rm")
        .arg("--name")
        .arg(&container_name)
        .arg("--gpus=all")
        .arg("-p")
        .arg(&port_mapping)
        .arg("-v")
        .arg(&volume)
        .arg("-e")
        .arg("HF_TOKEN")
        .arg("-e")
        .arg("HUGGING_FACE_HUB_TOKEN")
        .arg("vllm/vllm-openai:gemma4-cu130")
        .arg(model)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to start vllm docker container (is docker on PATH?)")?;
    Ok((child, container_name))
}

/// Stop a Docker container by name, waiting for it to exit.
/// This is the correct way to terminate the vllm workload; killing the
/// `docker run` client process alone leaves the container running.
fn stop_docker_container(name: &str) {
    let _ = Command::new("docker")
        .arg("rm")
        .arg("-f")
        .arg(name)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
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
    model: &str,
    prompt: &str,
) -> Result<BenchSummary> {
    let url = format!("http://{host}:{port}/v1/chat/completions");

    // Non-streaming probe to get the true prompt token count.
    let n_prompt = {
        let body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": false,
        });
        let resp: serde_json::Value = ureq::post(&url)
            .timeout(Duration::from_secs(1800))
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

    // Label width: "[warmup W/W]" or "[run R/RR]" — compute once so columns align.
    let label_w = {
        let run_label = format!("[run {runs}/{runs}]");
        let warmup_label = if warmup > 0 {
            format!("[warmup {warmup}/{warmup}]")
        } else {
            String::new()
        };
        run_label.len().max(warmup_label.len())
    };

    for i in 0..total {
        let is_warmup = i < warmup;
        let (ttft_ms, total_ms, n_gen, output_text) = do_stream(&url, model, prompt, max_tokens)?;

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

        let snippet = truncate_for_display(&output_text, 80);

        if is_warmup {
            let label = format!("[warmup {}/{warmup}]", i + 1);
            eprintln!(
                "  {label:<label_w$}  TTFT={ttft_ms:>8.1}ms  prefill={prefill_tps:>7.1}t/s  \
                 decode={decode_tps:>7.1}t/s  (prompt={n_prompt} tok, gen={n_gen} tok)"
            );
            eprintln!("  {:<label_w$}  output: {snippet}", "");
        } else {
            let run_num = i - warmup + 1;
            ttfts.push(ttft_ms);
            prefills.push(prefill_tps);
            decodes.push(decode_tps);
            let label = format!("[run {run_num}/{runs}]");
            eprintln!(
                "  {label:<label_w$}  TTFT={ttft_ms:>8.1}ms  prefill={prefill_tps:>7.1}t/s  \
                 decode={decode_tps:>7.1}t/s  (prompt={n_prompt} tok, gen={n_gen} tok)"
            );
            eprintln!("  {:<label_w$}  output: {snippet}", "");
        }
    }

    eprintln!();
    eprintln!("  TTFT    : {}", stats(&ttfts, "ms"));
    eprintln!("  Prefill : {}", stats(&prefills, "tok/s"));
    eprintln!("  Decode  : {}", stats(&decodes, "tok/s"));

    Ok(BenchSummary {
        ttft_ms: mean_or_none(&ttfts),
        prefill_tps: mean_or_none(&prefills),
        decode_tps: mean_or_none(&decodes),
        peak_mem_mb: None,
    })
}

/// Stream one chat-completion request; returns (ttft_ms, total_ms, n_gen, output_text).
fn do_stream(
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: usize,
) -> Result<(f64, f64, usize, String)> {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": true,
    });

    let t0 = Instant::now();
    let resp = ureq::post(url)
        .timeout(Duration::from_secs(1800))
        .send_json(&body)?;

    let reader = std::io::BufReader::new(resp.into_reader());
    let mut ttft_ms: Option<f64> = None;
    let mut n_gen: usize = 0;
    let mut output_text = String::new();

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
            output_text.push_str(token);
        }
    }

    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = ttft_ms.unwrap_or(total_ms);
    Ok((ttft_ms, total_ms, n_gen, output_text))
}

/// Truncate `s` to at most `max_chars` characters, replacing newlines with spaces,
/// and appending "…" when truncated.
fn truncate_for_display(s: &str, max_chars: usize) -> String {
    // Collapse whitespace/newlines so the snippet fits on one line.
    let flat: String = s
        .chars()
        .map(|c| if c.is_ascii_whitespace() { ' ' } else { c })
        .collect();
    let flat = flat.trim().to_string();
    if flat.chars().count() <= max_chars {
        flat
    } else {
        let truncated: String = flat.chars().take(max_chars).collect();
        format!("{truncated}…")
    }
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

    // Column widths (fixed for numeric columns, dynamic for backend name).
    const W_TTFT: usize = 12;
    const W_PFILL: usize = 14;
    const W_DEC: usize = 13;
    const W_MEM: usize = 14;
    let w = rows
        .iter()
        .map(|(name, _, _, _, _)| name.len())
        .max()
        .unwrap_or(0)
        .max("Backend".len());
    let total_w = w + 2 + W_TTFT + 2 + W_PFILL + 2 + W_DEC + 2 + W_MEM;

    println!();
    println!(
        "Benchmark settings: prompt_len={} tokens, max_tokens={}, runs={}, warmup={}",
        args.prompt_len, args.max_tokens, args.runs, args.warmup
    );
    println!();
    println!("{}", "═".repeat(total_w));
    println!(
        "{:<w$}  {:>W_TTFT$}  {:>W_PFILL$}  {:>W_DEC$}  {:>W_MEM$}",
        "Backend", "TTFT (ms)", "Prefill (t/s)", "Decode (t/s)", "Peak mem (MB)",
    );
    println!("{}", "─".repeat(total_w));
    for (name, ttft, pfill, dec, mem) in &rows {
        println!(
            "{:<w$}  {:>W_TTFT$}  {:>W_PFILL$}  {:>W_DEC$}  {:>W_MEM$}",
            name,
            fmt(*ttft, "ms"),
            fmt(*pfill, "t/s"),
            fmt(*dec, "t/s"),
            fmt(*mem, "MB"),
        );
    }
    println!("{}", "═".repeat(total_w));
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

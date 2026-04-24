//! Benchmark runner for `inferrs bench`.
//!
//! Runs a configurable number of synthetic generation requests against the
//! local inference engine and reports throughput and latency statistics.
//!
//! Metrics reported per run:
//!   - Prefill throughput  (prompt tokens / GPU-synchronized prefill wall-time)
//!   - Decode  throughput  (output tokens / decode wall-time for all N tokens)
//!   - Time to first token (TTFT)   — prefill + first sample wall-time
//!   - Mean per-token latency       — decode wall-time / output tokens
//!   - End-to-end latency           — total wall-time for the request

use anyhow::Result;

use crate::engine::load_engine;
use crate::sampler::SamplingParams;
use crate::util::format_bytes;
use crate::ServeArgs;

/// Extra options that only apply to the bench subcommand.
#[derive(clap::Args, Clone)]
pub struct BenchArgs {
    /// All options shared with `serve` (model, dtype, device, …).
    #[command(flatten)]
    pub serve: ServeArgs,

    /// Number of warm-up runs (results discarded).
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,

    /// Number of timed benchmark runs.
    #[arg(long, default_value_t = 5)]
    pub runs: usize,

    /// Number of synthetic prompt tokens to feed as input.
    #[arg(long, default_value_t = 128)]
    pub prompt_len: usize,
}

pub fn run(args: BenchArgs) -> Result<()> {
    let serve = &args.serve;

    // Load model, build engine, attach paged KV.
    let ctx = load_engine(serve)?;
    let mut engine = ctx.engine;
    let raw_config = ctx.raw_config;
    let arch = ctx.arch;
    let dtype = ctx.dtype;

    // Reuse the tokenizer already loaded during engine initialisation.
    let tokenizer = ctx.tokenizer;

    // Build a synthetic prompt of the requested length.
    // Use the tokenizer's BOS token id if available, otherwise token id 1.
    let bos_id = tokenizer.bos_token_id().unwrap_or(1);
    let prompt_tokens: Vec<u32> = std::iter::repeat_n(bos_id, args.prompt_len).collect();

    // Clamp max_tokens to the model's effective KV-cache capacity so that
    // models with a sliding-window limit (e.g. Gemma3 at 512 tokens) don't
    // crash mid-generation with an opaque tensor error.
    let max_seq_len = ctx.max_seq_len;
    let max_tokens = {
        let available = if max_seq_len == usize::MAX {
            serve.max_tokens
        } else {
            max_seq_len.saturating_sub(prompt_tokens.len())
        };
        if serve.max_tokens > available {
            tracing::warn!(
                "Clamping max_tokens {} → {} (model KV cache capacity: {}, prompt: {})",
                serve.max_tokens,
                available,
                max_seq_len,
                prompt_tokens.len(),
            );
        }
        serve.max_tokens.min(available)
    };

    let sampling_params = SamplingParams {
        temperature: serve.temperature,
        top_p: serve.top_p,
        top_k: serve.top_k,
        max_tokens,
        // Suppress stop tokens so every run generates exactly max_tokens.
        // Without this, early EOS produces variable n_output and skews throughput.
        bypass_stop_tokens: true,
        ..SamplingParams::default()
    };

    let total_runs = args.warmup + args.runs;
    let mut prefill_ms_samples: Vec<f64> = Vec::with_capacity(args.runs);
    let mut decode_ms_samples: Vec<f64> = Vec::with_capacity(args.runs);
    let mut ttft_ms_samples: Vec<f64> = Vec::with_capacity(args.runs);
    let mut e2e_ms_samples: Vec<f64> = Vec::with_capacity(args.runs);
    let mut prefill_tps_samples: Vec<f64> = Vec::with_capacity(args.runs);
    let mut decode_tps_samples: Vec<f64> = Vec::with_capacity(args.runs);
    let mut output_tok_samples: Vec<usize> = Vec::with_capacity(args.runs);

    println!(
        "Benchmarking {} ({} warm-up + {} timed runs, prompt_len={}, max_tokens={})",
        serve.model.as_deref().unwrap_or("<no model>"),
        args.warmup,
        args.runs,
        args.prompt_len,
        max_tokens,
    );

    for i in 0..total_runs {
        let is_warmup = i < args.warmup;
        let label = if is_warmup {
            format!("warm-up {}/{}", i + 1, args.warmup)
        } else {
            format!("run {}/{}", i - args.warmup + 1, args.runs)
        };

        let wall_start = std::time::Instant::now();
        let (result, prefill_ms, decode_ms, ttft_ms) =
            engine.bench_generate("bench", &prompt_tokens, &sampling_params)?;
        let e2e_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

        let n_prompt = result.prompt_tokens;
        let n_output = result.completion_tokens;

        if is_warmup {
            println!(
                "  [{label}] prefill={prefill_ms:.1}ms  decode={decode_ms:.1}ms  output_tokens={n_output}",
            );
        } else {
            let prefill_tps = if prefill_ms > 0.0 {
                n_prompt as f64 / (prefill_ms / 1000.0)
            } else {
                0.0
            };
            let decode_tps = if decode_ms > 0.0 {
                n_output as f64 / (decode_ms / 1000.0)
            } else {
                0.0
            };

            prefill_ms_samples.push(prefill_ms);
            decode_ms_samples.push(decode_ms);
            ttft_ms_samples.push(ttft_ms);
            e2e_ms_samples.push(e2e_ms);
            prefill_tps_samples.push(prefill_tps);
            decode_tps_samples.push(decode_tps);
            output_tok_samples.push(n_output);

            println!(
                "  [{label}] TTFT={ttft_ms:.1}ms  prefill={prefill_tps:.1}tok/s  decode={decode_tps:.1}tok/s  output_tokens={n_output}",
            );
        }
    }

    if args.runs == 0 {
        return Ok(());
    }

    // ── Aggregate statistics ─────────────────────────────────────────────────
    let n = args.runs as f64;

    let mean_prefill_ms = prefill_ms_samples.iter().sum::<f64>() / n;
    let mean_decode_ms = decode_ms_samples.iter().sum::<f64>() / n;
    let mean_ttft_ms = ttft_ms_samples.iter().sum::<f64>() / n;
    let mean_e2e_ms = e2e_ms_samples.iter().sum::<f64>() / n;
    let mean_output_toks = output_tok_samples.iter().sum::<usize>() as f64 / n;

    // Use mean-of-rates (not rate-of-means) so each run is equally weighted.
    let mean_prefill_tps = prefill_tps_samples.iter().sum::<f64>() / n;
    let mean_decode_tps = decode_tps_samples.iter().sum::<f64>() / n;

    let std_prefill_tps = stddev(&prefill_tps_samples);
    let std_decode_tps = stddev(&decode_tps_samples);

    let mean_per_token_ms = if mean_output_toks > 0.0 {
        mean_decode_ms / mean_output_toks
    } else {
        0.0
    };

    // p50 / p90 for end-to-end latency
    let mut sorted_e2e = e2e_ms_samples.clone();
    sorted_e2e.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&sorted_e2e, 50.0);
    let p90 = percentile(&sorted_e2e, 90.0);

    // ── KV cache memory estimate ─────────────────────────────────────────────
    let kv_mem_str = {
        let (num_kv_heads, head_dim, num_layers) = raw_config.kv_cache_params(&arch);
        let actual_seq_len = args.prompt_len + max_tokens;
        // bytes consumed per token across all layers (K + V combined)
        let bytes_per_token: usize = if let Some(bits) = serve.turbo_quant.0 {
            // PolarQuant: (head_dim-1) packed angle indices + one f32 root norm per token
            let index_bytes = ((head_dim - 1) * bits as usize).div_ceil(8);
            let norm_bytes = 4; // one f32 root norm per token vector
            (index_bytes + norm_bytes) * 2 * num_kv_heads * num_layers
        } else {
            // Regular bf16/f16 or f32 cache
            let bytes_per_element = dtype.size_in_bytes();
            head_dim * 2 * num_kv_heads * num_layers * bytes_per_element
        };
        let total_bytes = bytes_per_token * actual_seq_len;
        format_bytes(total_bytes as u64)
    };

    println!();
    println!(
        "── Results ({} runs) ──────────────────────────────────────────",
        args.runs
    );
    println!("  Output tokens (avg)     : {mean_output_toks:.0}");
    println!("  KV cache memory (est.)  : {kv_mem_str}");
    println!("  Prefill throughput      : {mean_prefill_tps:.1} tok/s  (σ={std_prefill_tps:.1})");
    println!("  Decode  throughput      : {mean_decode_tps:.1} tok/s  (σ={std_decode_tps:.1})");
    println!(
        "  Time to first token     : {mean_ttft_ms:.1} ms  (prefill: {mean_prefill_ms:.1} ms)"
    );
    println!("  Per-token latency (avg) : {mean_per_token_ms:.2} ms/tok");
    println!("  End-to-end latency (avg): {mean_e2e_ms:.1} ms");
    println!("  End-to-end p50          : {p50:.1} ms");
    println!("  End-to-end p90          : {p90:.1} ms");

    Ok(())
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn stddev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let variance = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    variance.sqrt()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

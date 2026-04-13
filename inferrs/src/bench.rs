//! Benchmark runner for `inferrs bench`.
//!
//! Runs a configurable number of synthetic generation requests against the
//! local inference engine and reports throughput and latency statistics.
//!
//! Metrics reported per run:
//!   - Prefill throughput  (prompt tokens / prefill wall-time)
//!   - Decode  throughput  (output tokens / decode  wall-time)
//!   - Time to first token (TTFT)   — prefill wall-time
//!   - Mean per-token latency       — decode wall-time / output tokens
//!   - End-to-end latency           — total wall-time for the request

use anyhow::Result;

use crate::engine::load_engine;
use crate::sampler::SamplingParams;
use crate::tokenizer::Tokenizer;
use crate::util::format_bytes;
use crate::ServeArgs;
use inferrs_models::turbo_quant::GROUP_SIZE;

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

    // Bench only needs a plain tokenizer (no chat template) for the BOS id.
    let tokenizer = Tokenizer::from_file(
        &ctx.model_files.tokenizer_path,
        ctx.model_files.tokenizer_config_path.as_deref(),
    )?;

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
        ..SamplingParams::default()
    };

    let total_runs = args.warmup + args.runs;
    let mut prefill_ms_samples: Vec<f64> = Vec::with_capacity(args.runs);
    let mut decode_ms_samples: Vec<f64> = Vec::with_capacity(args.runs);
    let mut e2e_ms_samples: Vec<f64> = Vec::with_capacity(args.runs);
    let mut prompt_tok_samples: Vec<usize> = Vec::with_capacity(args.runs);
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
        let (result, prefill_ms, decode_ms) =
            engine.bench_generate("bench", &prompt_tokens, &sampling_params)?;
        let e2e_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

        let n_prompt = result.prompt_tokens;
        let n_output = result.completion_tokens;

        if is_warmup {
            println!(
                "  [{label}] prefill={prefill_ms:.1}ms  decode={decode_ms:.1}ms  output_tokens={n_output}",
            );
        } else {
            prefill_ms_samples.push(prefill_ms);
            decode_ms_samples.push(decode_ms);
            e2e_ms_samples.push(e2e_ms);
            prompt_tok_samples.push(n_prompt);
            output_tok_samples.push(n_output);

            let ttft_ms = prefill_ms;
            let decode_tps = if decode_ms > 0.0 {
                n_output as f64 / (decode_ms / 1000.0)
            } else {
                0.0
            };
            let prefill_tps = if prefill_ms > 0.0 {
                n_prompt as f64 / (prefill_ms / 1000.0)
            } else {
                0.0
            };

            println!(
                "  [{label}] TTFT={ttft_ms:.1}ms  decode={decode_tps:.1}tok/s  prefill={prefill_tps:.1}tok/s  output_tokens={n_output}",
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
    let mean_e2e_ms = e2e_ms_samples.iter().sum::<f64>() / n;
    let mean_prompt_toks = prompt_tok_samples.iter().sum::<usize>() as f64 / n;
    let mean_output_toks = output_tok_samples.iter().sum::<usize>() as f64 / n;

    let mean_prefill_tps = mean_prompt_toks / (mean_prefill_ms / 1000.0);
    let mean_decode_tps = mean_output_toks / (mean_decode_ms / 1000.0);
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
        // bytes consumed per token across all layers (K + V combined)
        let bytes_per_token: usize = if let Some(bits) = serve.turbo_quant.0 {
            // TurboQuant: nibble-packed indices + f32 per-group absmax scales
            let index_bytes = if bits <= 4 {
                // two indices packed per byte
                head_dim.div_ceil(2)
            } else {
                // one index per byte
                head_dim
            };
            let n_groups = head_dim.div_ceil(GROUP_SIZE);
            let scale_bytes = n_groups * 4; // f32 per group
                                            // K and V each have index_bytes + scale_bytes, times num_kv_heads, times num_layers
            (index_bytes + scale_bytes) * 2 * num_kv_heads * num_layers
        } else {
            // Regular bf16/f16 or f32 cache
            let bytes_per_element = dtype.size_in_bytes();
            head_dim * 2 * num_kv_heads * num_layers * bytes_per_element
        };
        let effective_seq_len = if max_seq_len == usize::MAX {
            args.prompt_len + max_tokens
        } else {
            max_seq_len
        };
        let total_bytes = bytes_per_token * effective_seq_len;
        format_bytes(total_bytes as u64)
    };

    println!();
    println!(
        "── Results ({} runs) ──────────────────────────────────────────",
        args.runs
    );
    println!("  Prompt tokens (avg)     : {mean_prompt_toks:.0}");
    println!("  Output tokens (avg)     : {mean_output_toks:.0}");
    println!("  KV cache memory         : {kv_mem_str}");
    println!("  Prefill throughput      : {mean_prefill_tps:.1} tok/s");
    println!("  Decode  throughput      : {mean_decode_tps:.1} tok/s");
    println!("  Time to first token     : {mean_prefill_ms:.1} ms");
    println!("  Per-token latency (avg) : {mean_per_token_ms:.2} ms/tok");
    println!("  End-to-end latency (avg): {mean_e2e_ms:.1} ms");
    println!("  End-to-end p50          : {p50:.1} ms");
    println!("  End-to-end p90          : {p90:.1} ms");

    Ok(())
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

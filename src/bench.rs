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
use candle_core::DType;

use crate::config::RawConfig;
use crate::engine::Engine;
use crate::kv_cache::{BlockPool, PagedCacheConfig, PagedKvStore};
use crate::sampler::SamplingParams;
use crate::tokenizer::Tokenizer;
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
    let device = serve.resolve_device()?;
    let dtype = serve.resolve_dtype()?;

    // Download / load model (same path as `serve`)
    let model_files = crate::hub::download_model(&serve.model, &serve.revision)?;
    let raw_config = RawConfig::from_file(&model_files.config_path)?;
    let arch = raw_config.detect_architecture()?;
    tracing::info!("Detected architecture: {:?}", arch);

    let tokenizer = Tokenizer::from_file(
        &model_files.tokenizer_path,
        model_files.tokenizer_config_path.as_deref(),
    )?;

    let model = crate::models::load_model(
        &raw_config,
        &arch,
        &model_files.weight_paths,
        dtype,
        &device,
        Some(4),
    )?;

    let mut engine = Engine::new(
        model,
        Tokenizer::from_file(
            &model_files.tokenizer_path,
            model_files.tokenizer_config_path.as_deref(),
        )?,
        device.clone(),
        serve.max_batch_size,
        serve.max_tokens_per_step,
    );

    // Wire up paged attention if requested (same logic as `serve`)
    if let Some(memory_fraction) = serve.paged_attention {
        let bytes_per_element = match dtype {
            DType::F32 => 4,
            _ => 2,
        };
        let total_memory_bytes: usize = match &device {
            candle_core::Device::Cuda(_) | candle_core::Device::Metal(_) => 8 * 1024 * 1024 * 1024,
            _ => 4 * 1024 * 1024 * 1024,
        };
        let (num_kv_heads, head_dim, num_kv_layers) = raw_config.kv_cache_params(&arch);
        let paged_cfg = PagedCacheConfig::from_memory_fraction(
            total_memory_bytes,
            memory_fraction,
            serve.block_size,
            num_kv_heads,
            head_dim,
            num_kv_layers,
            bytes_per_element,
        );
        tracing::info!(
            "Paged KV store: {} blocks × {} tokens/block",
            paged_cfg.num_blocks,
            paged_cfg.block_size,
        );
        let block_pool = BlockPool::new(paged_cfg.num_blocks, paged_cfg.block_size);
        let kv_store = PagedKvStore::new(paged_cfg, dtype, &device)?;
        engine = engine.with_paged_kv(block_pool, kv_store);
    }

    // Build a synthetic prompt of the requested length.
    // Use the tokenizer's BOS token id if available, otherwise token id 1.
    let bos_id = tokenizer.stop_token_ids.first().copied().unwrap_or(1);
    let prompt_tokens: Vec<u32> = std::iter::repeat_n(bos_id, args.prompt_len).collect();

    // Clamp max_tokens to the model's effective KV-cache capacity so that
    // models with a sliding-window limit (e.g. Gemma3 at 512 tokens) don't
    // crash mid-generation with an opaque tensor error.
    let max_seq_len = raw_config.effective_max_seq_len(&arch);
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
        repetition_penalty: 1.0,
        max_tokens,
    };

    let total_runs = args.warmup + args.runs;
    let mut prefill_ms_samples: Vec<f64> = Vec::with_capacity(args.runs);
    let mut decode_ms_samples: Vec<f64> = Vec::with_capacity(args.runs);
    let mut e2e_ms_samples: Vec<f64> = Vec::with_capacity(args.runs);
    let mut prompt_tok_samples: Vec<usize> = Vec::with_capacity(args.runs);
    let mut output_tok_samples: Vec<usize> = Vec::with_capacity(args.runs);

    println!(
        "Benchmarking {} ({} warm-up + {} timed runs, prompt_len={}, max_tokens={})",
        serve.model, args.warmup, args.runs, args.prompt_len, max_tokens,
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
                "  [{}] prefill={:.1}ms  decode={:.1}ms  output_tokens={}",
                label, prefill_ms, decode_ms, n_output
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
                "  [{}] TTFT={:.1}ms  decode={:.1}tok/s  prefill={:.1}tok/s  output_tokens={}",
                label, ttft_ms, decode_tps, prefill_tps, n_output
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

    println!();
    println!(
        "── Results ({} runs) ──────────────────────────────────────────",
        args.runs
    );
    println!("  Prompt tokens (avg)     : {:.0}", mean_prompt_toks);
    println!("  Output tokens (avg)     : {:.0}", mean_output_toks);
    println!("  Prefill throughput      : {:.1} tok/s", mean_prefill_tps);
    println!("  Decode  throughput      : {:.1} tok/s", mean_decode_tps);
    println!("  Time to first token     : {:.1} ms", mean_prefill_ms);
    println!(
        "  Per-token latency (avg) : {:.2} ms/tok",
        mean_per_token_ms
    );
    println!("  End-to-end latency (avg): {:.1} ms", mean_e2e_ms);
    println!("  End-to-end p50          : {:.1} ms", p50);
    println!("  End-to-end p90          : {:.1} ms", p90);

    Ok(())
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

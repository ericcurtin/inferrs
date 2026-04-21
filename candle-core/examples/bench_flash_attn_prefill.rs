use candle_core::{DType, Device, Result, Tensor};
use std::time::Instant;

struct Cfg {
    name: &'static str,
    head_dim: usize,
    n_q: usize,
    n_kv: usize,
    q_len: usize,
}

const CONFIGS: &[Cfg] = &[
    Cfg { name: "Qwen3-4B (D=128, GQA=4)",    head_dim: 128, n_q:  8, n_kv: 2, q_len:   64 },
    Cfg { name: "Qwen3-4B (D=128, GQA=4)",    head_dim: 128, n_q:  8, n_kv: 2, q_len:  256 },
    Cfg { name: "Qwen3-4B (D=128, GQA=4)",    head_dim: 128, n_q:  8, n_kv: 2, q_len: 1024 },
    Cfg { name: "Qwen3.5-4B (D=256, GQA=4)",  head_dim: 256, n_q:  8, n_kv: 2, q_len:   64 },
    Cfg { name: "Qwen3.5-4B (D=256, GQA=4)",  head_dim: 256, n_q:  8, n_kv: 2, q_len:  256 },
    Cfg { name: "Qwen3.5-4B (D=256, GQA=4)",  head_dim: 256, n_q:  8, n_kv: 2, q_len: 1024 },
    Cfg { name: "Qwen3.5-122B (D=256, GQA=16)", head_dim: 256, n_q: 32, n_kv: 2, q_len:   64 },
    Cfg { name: "Qwen3.5-122B (D=256, GQA=16)", head_dim: 256, n_q: 32, n_kv: 2, q_len:  256 },
    Cfg { name: "Qwen3.5-122B (D=256, GQA=16)", head_dim: 256, n_q: 32, n_kv: 2, q_len: 1024 },
];

const WARMUP: usize = 3;
const RUNS: usize = 10;

fn causal_mask(q_len: usize, kv_len: usize, dev: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..q_len)
        .flat_map(|i| (0..kv_len).map(move |j| if j <= i { 0.0f32 } else { f32::NEG_INFINITY }))
        .collect();
    Tensor::new(mask.as_slice(), dev)?
        .reshape((1, 1, q_len, kv_len))?
        .to_dtype(DType::BF16)
}

fn naive_attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor, scale: f64) -> Result<Tensor> {
    let (b, n_q, q_len, d) = q.dims4()?;
    let (_, n_kv, kv_len, _) = k.dims4()?;
    let gqa = n_q / n_kv;
    let kt = k.transpose(2, 3)?.contiguous()?;
    let vc = v.contiguous()?;

    if gqa == 1 {
        let attn_w = q.contiguous()?.matmul(&kt)?.affine(scale, 0.0)?;
        let attn_w = attn_w.broadcast_add(mask)?;
        let max = attn_w.max_keepdim(3)?;
        let exp = attn_w.broadcast_sub(&max)?.exp()?;
        let sum = exp.sum_keepdim(3)?;
        return exp.broadcast_div(&sum)?.matmul(&vc);
    }

    // GQA: reshape Q into KV-head groups, matmul, then reshape back.
    let q_r = q.contiguous()?.reshape((b, n_kv, gqa * q_len, d))?;
    let attn_w = q_r.matmul(&kt)?.affine(scale, 0.0)?;
    // Reshape to [b, n_q, q_len, kv_len] before applying the mask.
    let attn_w = attn_w.reshape((b, n_q, q_len, kv_len))?;
    let attn_w = attn_w.broadcast_add(mask)?;
    let max = attn_w.max_keepdim(3)?;
    let exp = attn_w.broadcast_sub(&max)?.exp()?;
    let sum = exp.sum_keepdim(3)?;
    let attn = exp.broadcast_div(&sum)?;
    let attn_r = attn.reshape((b, n_kv, gqa * q_len, kv_len))?;
    attn_r.matmul(&vc)?.reshape((b, n_q, q_len, d))
}

fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let out = candle_core::cuda_flash_attn::flash_attn_prefill_cuda(q, k, v, scale)?;
    out.to_dtype(DType::BF16)
}

fn attn_flops(n_q: usize, q_len: usize, d: usize) -> f64 {
    4.0 * n_q as f64 * q_len as f64 * q_len as f64 * d as f64
}

fn median(samples: &[f64]) -> f64 {
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;

    println!("Flash Attention Prefill — Naive vs CUDA Kernel");
    println!("{:=<82}", "");
    println!(
        "{:<30} {:>5} {:>12} {:>12} {:>8} {:>10}",
        "Config", "t", "Naive ms", "Flash ms", "Speedup", "TFLOPS"
    );
    println!("{:-<82}", "");

    for cfg in CONFIGS {
        let q = Tensor::randn(0f32, 1f32, (1, cfg.n_q,  cfg.q_len, cfg.head_dim), &dev)?.to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1f32, (1, cfg.n_kv, cfg.q_len, cfg.head_dim), &dev)?.to_dtype(DType::BF16)?;
        let v = Tensor::randn(0f32, 1f32, (1, cfg.n_kv, cfg.q_len, cfg.head_dim), &dev)?.to_dtype(DType::BF16)?;
        let mask = causal_mask(cfg.q_len, cfg.q_len, &dev)?;
        let scale = 1.0f64 / (cfg.head_dim as f64).sqrt();
        let flops = attn_flops(cfg.n_q, cfg.q_len, cfg.head_dim);

        for _ in 0..WARMUP {
            let _ = naive_attention(&q, &k, &v, &mask, scale)?;
            let _ = flash_attention(&q, &k, &v, scale as f32)?;
            dev.synchronize()?;
        }

        let mut naive_ms = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let t = Instant::now();
            let _ = naive_attention(&q, &k, &v, &mask, scale)?;
            dev.synchronize()?;
            naive_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }

        let mut flash_ms = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let t = Instant::now();
            let _ = flash_attention(&q, &k, &v, scale as f32)?;
            dev.synchronize()?;
            flash_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }

        let naive_med = median(&naive_ms);
        let flash_med = median(&flash_ms);
        let speedup = naive_med / flash_med;
        let tflops = flops / (flash_med * 1e-3) / 1e12;

        println!(
            "{:<30} {:>5} {:>11.2}ms {:>11.2}ms {:>7.2}x {:>9.2}",
            cfg.name, cfg.q_len, naive_med, flash_med, speedup, tflops
        );
    }

    println!("{:=<82}", "");
    Ok(())
}

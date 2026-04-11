use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

/// Benchmark shapes for depthwise conv1d.
/// Tuple: (name, channels, seq_len, kernel_size)
/// The "decode_1tok" shape matches Qwen3.5-2B during single-token decode,
/// which was the original bottleneck (6144 serial CUDA launches → 1 launch).
const BENCH_SHAPES: &[(&str, usize, usize, usize)] = &[
    ("decode_1tok", 6144, 1, 4),   // Qwen3.5-2B single-token decode
    ("decode_8tok", 6144, 8, 4),   // small batch
    ("prefill_128", 6144, 128, 4), // short prefill
    ("small_c64", 64, 32, 3),      // sanity / CPU baseline
];

fn run(x: &Tensor, w: &Tensor, padding: usize, groups: usize) {
    x.conv1d(w, padding, 1, 1, groups).unwrap();
}

fn run_bench(c: &mut Criterion, device: &Device, name: &str, channels: usize, t: usize, k: usize) {
    let dtype = DType::BF16;
    let x = Tensor::zeros(&[1usize, channels, t], dtype, device).unwrap();
    let w = Tensor::zeros(&[channels, 1usize, k], dtype, device).unwrap();
    let padding = k - 1;
    // MAC ops per output element = k; total = channels * t_out * k ≈ channels * t * k
    let flops = channels * t * k;

    let mut group = c.benchmark_group(device.bench_name(format!("conv1d_depthwise_{name}")));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                run(black_box(&x), black_box(&w), padding, channels);
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn conv1d_depthwise_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        for &(name, channels, t, k) in BENCH_SHAPES {
            run_bench(c, &device, name, channels, t, k);
        }
    }
}

criterion_group!(benches, conv1d_depthwise_benchmark);

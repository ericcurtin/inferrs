//! Token sampling: temperature, top-k, top-p, repetition penalty.

use anyhow::Result;
use candle_core::{DType, Tensor};

/// Sampling parameters for a generation request.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f64,
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.0,
            max_tokens: 2048,
        }
    }
}

const SAMPLING_EPS: f64 = 1e-5;

/// Sample a token from logits using the given parameters.
pub fn sample_token(
    logits: &Tensor,
    params: &SamplingParams,
    previous_tokens: &[u32],
) -> Result<u32> {
    // logits shape: (1, 1, vocab_size) or (1, vocab_size) or (vocab_size,)
    // Flatten to 1D
    let logits = logits.squeeze(0)?;
    let logits = if logits.dims().len() > 1 {
        logits.squeeze(0)?
    } else {
        logits
    };
    let logits = logits.to_dtype(DType::F32)?;

    // Apply repetition penalty
    let logits = if params.repetition_penalty != 1.0 && !previous_tokens.is_empty() {
        apply_repetition_penalty(&logits, previous_tokens, params.repetition_penalty)?
    } else {
        logits
    };

    // Greedy sampling if temperature is very low
    if params.temperature < SAMPLING_EPS {
        let token_id = logits.argmax(0)?.to_scalar::<u32>()?;
        return Ok(token_id);
    }

    // Apply temperature
    let logits = (&logits / params.temperature)?;

    // Convert to probabilities
    let probs = candle_nn::ops::softmax_last_dim(&logits)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;

    // Apply top-k filtering
    let mut indexed_probs: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-k
    if params.top_k > 0 && params.top_k < indexed_probs.len() {
        indexed_probs.truncate(params.top_k);
    }

    // Top-p (nucleus sampling)
    if params.top_p < 1.0 {
        let mut cumsum = 0.0f32;
        let mut cutoff_idx = indexed_probs.len();
        for (i, &(_, p)) in indexed_probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= params.top_p as f32 {
                cutoff_idx = i + 1;
                break;
            }
        }
        indexed_probs.truncate(cutoff_idx);
    }

    // Renormalize
    let sum: f32 = indexed_probs.iter().map(|&(_, p)| p).sum();
    if sum <= 0.0 {
        // Fallback to argmax
        return Ok(indexed_probs
            .first()
            .map(|&(idx, _)| idx as u32)
            .unwrap_or(0));
    }

    // Sample from the filtered distribution
    let mut rng_val: f32 = rand_f32();
    for &(idx, prob) in &indexed_probs {
        let normalized = prob / sum;
        if rng_val < normalized {
            return Ok(idx as u32);
        }
        rng_val -= normalized;
    }

    // Fallback
    Ok(indexed_probs
        .last()
        .map(|&(idx, _)| idx as u32)
        .unwrap_or(0))
}

fn apply_repetition_penalty(
    logits: &Tensor,
    previous_tokens: &[u32],
    penalty: f64,
) -> Result<Tensor> {
    // Build the penalty on-device to avoid a GPU→CPU→GPU round-trip every token.
    // Strategy: start from a 1.0 multiplier tensor; for each repeated token set the
    // multiplier to 1/penalty (positive logit) or penalty (negative logit).
    // We use the sign of the logit to decide: multiply by (1/penalty) where logit>0,
    // multiply by penalty where logit<0.
    //
    // Equivalent scalar formula: penalised = logit / penalty   if logit >= 0
    //                                       = logit * penalty   if logit < 0
    // = logit * (1/penalty) * (logit>=0)  +  logit * penalty * (logit<0)
    // = logit * [ (1/penalty - penalty) * (logit>=0) + penalty ]
    //   where (logit>=0) is 1 when >=0, 0 otherwise.
    //
    // We build a scatter mask of size vocab that is 1.0 everywhere except at
    // repeated-token positions where it is set to (penalty_factor(logit)).
    // Because the factor depends on the sign of each logit we do two passes:
    //   pass 1: gather the logits at repeated positions (one GPU read)
    //   pass 2: compute per-index factor on CPU (tiny, only unique token count)
    //   pass 3: scatter back and multiply (two small GPU ops)
    //
    // This is still O(unique_tokens) CPU work but avoids transferring the full
    // vocab tensor (248 K floats) over the bus.

    let vocab = logits.dim(0)?;
    let device = logits.device();

    // Deduplicate previous tokens that fall within vocab range
    let mut seen = std::collections::HashSet::new();
    let unique_ids: Vec<u32> = previous_tokens
        .iter()
        .copied()
        .filter(|&id| (id as usize) < vocab && seen.insert(id))
        .collect();

    if unique_ids.is_empty() {
        return Ok(logits.clone());
    }

    // Gather logits at repeated positions: one small GPU read
    let indices = Tensor::new(unique_ids.as_slice(), device)?;
    let gathered = logits.gather(&indices, 0)?; // [n_unique]
    let gathered_vec: Vec<f32> = gathered.to_vec1()?; // small CPU transfer

    // Compute per-token penalty factor on CPU
    let penalty_f = penalty as f32;
    let factors: Vec<f32> = gathered_vec
        .iter()
        .map(|&s| if s >= 0.0 { 1.0 / penalty_f } else { penalty_f })
        .collect();

    // Build a full vocab multiplier initialised to 1.0, then scatter factors
    let mut multiplier = vec![1.0f32; vocab];
    for (&id, &f) in unique_ids.iter().zip(factors.iter()) {
        multiplier[id as usize] = f;
    }
    let mult = Tensor::from_vec(multiplier, vocab, device)?;

    (logits * mult).map_err(Into::into)
}

/// Random float in [0, 1) using a thread-local xorshift64* PRNG.
///
/// Seeded once per thread from the system clock + thread ID mixed with
/// a splitmix64 step, giving good statistical properties and uniform coverage
/// of [0, 1) without clustering.
fn rand_f32() -> f32 {
    use std::time::SystemTime;

    thread_local! {
        static STATE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        static SEEDED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    SEEDED.with(|seeded| {
        if !seeded.get() {
            // Seed from nanosecond time + thread ID via splitmix64
            let t = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            // Mix thread ID in to avoid identical seeds across threads
            let tid = {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut h = DefaultHasher::new();
                std::thread::current().id().hash(&mut h);
                h.finish()
            };
            let mut seed = t ^ tid;
            // splitmix64 step to avalanche the seed bits
            seed = seed.wrapping_add(0x9e3779b97f4a7c15);
            seed = (seed ^ (seed >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            seed = (seed ^ (seed >> 27)).wrapping_mul(0x94d049bb133111eb);
            seed ^= seed >> 31;
            // Ensure non-zero (xorshift requires non-zero state)
            STATE.with(|s| s.set(if seed == 0 { 0x1234567890abcdef } else { seed }));
            seeded.set(true);
        }
    });

    STATE.with(|s| {
        // xorshift64* — excellent statistical quality, fast
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        // Multiply by a constant and shift to get a well-distributed u32,
        // then map to [0, 1)
        let u = x.wrapping_mul(0x2545f4914f6cdd1d) >> 32;
        u as f32 / (u32::MAX as f32 + 1.0)
    })
}

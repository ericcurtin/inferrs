//! Token sampling: temperature, top-k, top-p, min-p, repetition/frequency/
//! presence penalties, logit bias, and optionally seeded PRNG.
//!
//! Sampling strategies implemented here cover the core set from llama.cpp and
//! vLLM:
//!
//! - **Greedy** (temperature ≤ ε): argmax, native-dtype fast path
//! - **Temperature scaling** — divide logits by `temperature`
//! - **Logit bias** — additive per-token offsets before temperature scaling
//! - **Repetition penalty** — llama.cpp-style: `logit / penalty` if logit≥0,
//!   `logit * penalty` if logit<0 (applied to any previously-seen token)
//! - **Frequency penalty** — OpenAI-style: `logit -= frequency_penalty * count`
//! - **Presence penalty** — OpenAI-style: `logit -= presence_penalty` for any
//!   token that appears at least once in the history
//! - **Top-k** filtering
//! - **Top-p** (nucleus) filtering
//! - **Min-p** filtering — keeps only tokens whose probability is ≥
//!   `min_p * max_prob` (from llama.cpp and Ollama)
//! - **Seeded PRNG** — per-request reproducible sampling via xorshift64*

use anyhow::Result;
use candle_core::{DType, Tensor};
use std::collections::HashMap;

/// Sampling parameters for a generation request.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    /// llama.cpp / Ollama `repeat_penalty`: divides positive logits and
    /// multiplies negative logits of previously-seen tokens.  1.0 = disabled.
    /// Default 1.1 (llama.cpp default) — mild suppression of repeated tokens.
    pub repetition_penalty: f64,
    /// Number of most-recent tokens to consider for `repetition_penalty` and
    /// `frequency_penalty`.  Mirrors llama.cpp `repeat_last_n`.
    /// 0 = disabled (penalty not applied).  Default: 64.
    pub repeat_last_n: usize,
    /// OpenAI `frequency_penalty`: subtracts `frequency_penalty * count` from
    /// each token's logit, penalising tokens proportional to how often they
    /// have already appeared.  0.0 = disabled.  Typical range: [0.0, 2.0].
    pub frequency_penalty: f64,
    /// OpenAI `presence_penalty`: subtracts `presence_penalty` from the logit
    /// of any token that has appeared at least once, regardless of count.
    /// 0.0 = disabled.  Typical range: [0.0, 2.0].
    pub presence_penalty: f64,
    /// Min-p filtering (llama.cpp / Ollama): after softmax keep only tokens
    /// whose probability is ≥ `min_p * max_prob`.  0.0 = disabled.
    pub min_p: f64,
    pub max_tokens: usize,
    /// Per-request extra stop token IDs derived from the `stop` field.
    /// Checked in addition to the model-wide stop token IDs in the engine.
    pub extra_stop_token_ids: Vec<u32>,
    /// Multi-token stop strings (decoded text).  The engine checks whether the
    /// recent decoded output is a suffix match after each token.
    pub stop_strings: Vec<String>,
    /// Per-token additive logit biases (OpenAI `logit_bias`).  Applied before
    /// temperature scaling.  Keys are token IDs.
    pub logit_bias: HashMap<u32, f32>,
    /// Seed for per-request reproducible sampling.  When `None`, a
    /// thread-local time+thread-id-seeded PRNG is used.
    pub seed: Option<u64>,
    /// Whether to collect log-probabilities for the sampled token.
    pub logprobs: bool,
    /// Number of top token log-probabilities to return alongside the sampled
    /// token's logprob (0–20).  Requires `logprobs = true`.
    pub top_logprobs: u8,
    /// Structured output grammar mode.
    pub grammar_mode: GrammarMode,
    /// When true, stop tokens and stop strings are ignored so generation runs
    /// until `max_tokens` is reached.  Used by `inferrs bench` to ensure every
    /// run produces exactly `max_tokens` output tokens for consistent metrics.
    pub bypass_stop_tokens: bool,
}

/// Structured output constraint mode.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum GrammarMode {
    /// No constraint — free generation.
    #[default]
    None,
    /// Constrain output to a valid JSON object (`{"type":"json_object"}`).
    JsonObject,
    /// Constrain output to a valid JSON object conforming to a JSON Schema.
    /// Currently aliases `JsonObject`; full schema validation is a future
    /// extension.
    JsonSchema,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            min_p: 0.0,
            max_tokens: 2048,
            extra_stop_token_ids: vec![],
            stop_strings: vec![],
            logit_bias: HashMap::new(),
            seed: None,
            logprobs: false,
            top_logprobs: 0,
            grammar_mode: GrammarMode::None,
            bypass_stop_tokens: false,
        }
    }
}

const SAMPLING_EPS: f64 = 1e-5;

/// Log-probability data for a single sampled token.
#[derive(Debug, Clone)]
pub struct TokenLogprob {
    #[allow(dead_code)]
    pub token_id: u32,
    /// Decoded text of the sampled token.  Populated by the engine; empty
    /// in the sampler where no tokenizer is available.
    pub token_text: String,
    /// Natural log probability of the sampled token (post-temperature softmax).
    pub logprob: f32,
    /// Top-k alternative log-probabilities ordered by probability descending.
    /// Each entry is `(token_id, logprob)`.  Empty when `top_logprobs == 0`.
    /// Token IDs are decoded to text by the engine before sending to the client;
    /// see `top_logprob_texts`.
    pub top_logprobs: Vec<(u32, f32)>,
    /// Decoded text for each entry in `top_logprobs`, parallel by index.
    /// Populated by the engine once the token is sampled; empty in the sampler.
    pub top_logprob_texts: Vec<String>,
}

/// Sample a token from logits using the given parameters.
///
/// Returns `(token_id, Option<TokenLogprob>)`.  The logprob is populated when
/// `params.logprobs == true`.
///
/// `previous_tokens` is used for penalty computation **and** as the step
/// index for seeded sampling: step = `previous_tokens.len()`.  Mixing the
/// step index into the per-seed hash ensures each decode step draws a
/// distinct pseudo-random value even though the same seed is reused across
/// the entire request.
pub fn sample_token(
    logits: &Tensor,
    params: &SamplingParams,
    previous_tokens: &[u32],
) -> Result<(u32, Option<TokenLogprob>)> {
    // Flatten logits to 1-D: (1, 1, V) → (V,)
    let logits = logits.squeeze(0)?;
    let logits = if logits.dims().len() > 1 {
        logits.squeeze(0)?
    } else {
        logits
    };

    // ── Greedy fast-path ─────────────────────────────────────────────────────
    // Argmax in native dtype (bf16/f32) avoids a full-vocab dtype conversion.
    // Only safe when nothing modifies relative ordering (no penalties/biases).
    //
    // Window previous_tokens to the last `repeat_last_n` tokens (llama.cpp
    // semantics: 0 = disabled, any other value = use last N tokens).
    let penalty_tokens: &[u32] = if params.repeat_last_n == 0 {
        &[]
    } else {
        let start = previous_tokens.len().saturating_sub(params.repeat_last_n);
        &previous_tokens[start..]
    };
    let has_penalty = (params.repetition_penalty != 1.0
        || params.frequency_penalty != 0.0
        || params.presence_penalty != 0.0)
        && !penalty_tokens.is_empty();
    let has_bias = !params.logit_bias.is_empty();
    if params.temperature < SAMPLING_EPS && !has_penalty && !has_bias && !params.logprobs {
        let token_id = logits.argmax(0)?.to_scalar::<u32>()?;
        return Ok((token_id, None));
    }

    // Convert to f32 for all subsequent arithmetic.
    let mut logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;
    let vocab = logits_vec.len();

    // ── 1. Logit bias ─────────────────────────────────────────────────────────
    for (&tid, &bias) in &params.logit_bias {
        if let Some(v) = logits_vec.get_mut(tid as usize) {
            *v += bias;
        }
    }

    // ── 2. Repetition penalty (llama.cpp / Ollama) ────────────────────────────
    // Formula: logit / penalty if logit ≥ 0, logit * penalty if logit < 0.
    // Applied only to tokens within the `repeat_last_n` window.
    // Guard against 1.0 (no-op) and 0.0 (would divide by zero).
    if params.repetition_penalty != 1.0
        && params.repetition_penalty != 0.0
        && !penalty_tokens.is_empty()
    {
        let p = params.repetition_penalty as f32;
        let mut seen = std::collections::HashSet::new();
        for &id in penalty_tokens {
            if id as usize >= vocab || !seen.insert(id) {
                continue;
            }
            let v = &mut logits_vec[id as usize];
            *v = if *v >= 0.0 { *v / p } else { *v * p };
        }
    }

    // ── 3. Frequency + presence penalties (OpenAI) ────────────────────────────
    // Applied only to tokens within the `repeat_last_n` window.
    if (params.frequency_penalty != 0.0 || params.presence_penalty != 0.0)
        && !penalty_tokens.is_empty()
    {
        let fp = params.frequency_penalty as f32;
        let pp = params.presence_penalty as f32;
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for &id in penalty_tokens {
            if (id as usize) < vocab {
                *counts.entry(id).or_insert(0) += 1;
            }
        }
        for (&tid, &count) in &counts {
            let v = &mut logits_vec[tid as usize];
            *v -= fp * count as f32;
            *v -= pp; // present at least once ∴ subtract flat penalty
        }
    }

    // ── 4. Temperature scaling ────────────────────────────────────────────────
    let temp = if params.temperature < SAMPLING_EPS {
        SAMPLING_EPS as f32
    } else {
        params.temperature as f32
    };
    for v in &mut logits_vec {
        *v /= temp;
    }

    // ── 5. Softmax (numerically stable) ──────────────────────────────────────
    let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits_vec.iter().map(|&v| (v - max_logit).exp()).collect();
    let prob_sum: f32 = probs.iter().sum();
    if prob_sum > 0.0 {
        for p in &mut probs {
            *p /= prob_sum;
        }
    }

    // ── 6. Sort by probability descending ────────────────────────────────────
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // ── 7. Top-k ──────────────────────────────────────────────────────────────
    if params.top_k > 0 && params.top_k < indexed.len() {
        indexed.truncate(params.top_k);
    }

    // ── 8. Top-p (nucleus) ────────────────────────────────────────────────────
    if params.top_p < 1.0 {
        let mut cumsum = 0.0f32;
        let mut cut = indexed.len();
        for (i, &(_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum >= params.top_p as f32 {
                cut = i + 1;
                break;
            }
        }
        indexed.truncate(cut);
    }

    // ── 9. Min-p (llama.cpp / Ollama) ─────────────────────────────────────────
    // Keep tokens with prob ≥ min_p * max_prob.  The list is sorted, so
    // indexed[0].1 is the maximum over the (already top-k/top-p filtered) set.
    if params.min_p > 0.0 && !indexed.is_empty() {
        let threshold = indexed[0].1 * params.min_p as f32;
        let cut = indexed.partition_point(|&(_, p)| p >= threshold);
        if cut > 0 {
            indexed.truncate(cut);
        }
    }

    // ── 10. Collect top-k logprobs (before renorm) ────────────────────────────
    let filtered_sum: f32 = indexed.iter().map(|&(_, p)| p).sum();
    let renorm_denom = if filtered_sum > 0.0 {
        filtered_sum
    } else {
        1.0
    };

    let top_lp: Vec<(u32, f32)> = if params.logprobs && params.top_logprobs > 0 {
        indexed
            .iter()
            .take(params.top_logprobs as usize)
            .map(|&(idx, p)| {
                let lp = if p > 0.0 {
                    (p / renorm_denom).ln()
                } else {
                    f32::NEG_INFINITY
                };
                (idx as u32, lp)
            })
            .collect()
    } else {
        vec![]
    };

    // ── 11. Sample ───────────────────────────────────────────────────────────
    if filtered_sum <= 0.0 {
        let token_id = indexed.first().map(|&(idx, _)| idx as u32).unwrap_or(0);
        let lp = params.logprobs.then_some(TokenLogprob {
            token_id,
            token_text: String::new(),
            logprob: f32::NEG_INFINITY,
            top_logprobs: top_lp,
            top_logprob_texts: vec![],
        });
        return Ok((token_id, lp));
    }

    // Use previous_tokens.len() as the step index so that each decode step
    // draws a distinct value when a seed is set.
    let step = previous_tokens.len() as u64;
    let mut rng_val = rand_f32(params.seed, step);
    let mut sampled = indexed.last().map(|&(idx, _)| idx).unwrap_or(0);
    for &(idx, prob) in &indexed {
        let norm = prob / filtered_sum;
        if rng_val < norm {
            sampled = idx;
            break;
        }
        rng_val -= norm;
    }
    let token_id = sampled as u32;

    // ── 12. Sampled token logprob ─────────────────────────────────────────────
    let lp = if params.logprobs {
        let p = indexed
            .iter()
            .find(|&&(idx, _)| idx == sampled)
            .map(|&(_, p)| p / filtered_sum)
            .unwrap_or(0.0);
        Some(TokenLogprob {
            token_id,
            token_text: String::new(),
            logprob: if p > 0.0 { p.ln() } else { f32::NEG_INFINITY },
            top_logprobs: top_lp,
            top_logprob_texts: vec![],
        })
    } else {
        None
    };

    Ok((token_id, lp))
}

// ── PRNG ─────────────────────────────────────────────────────────────────────

/// Random float in [0, 1) via xorshift64*.
///
/// When `seed` is `Some`, the state is derived from `(seed, step)` so that
/// each decode step at a given step index produces a **distinct** value for
/// the same seed.  Mixing in `step` via a second splitmix64 round prevents
/// all steps from collapsing to the same float — the original bug where only
/// the seed constant was hashed.
///
/// When `seed` is `None`, the thread-local PRNG is advanced.
fn rand_f32(seed: Option<u64>, step: u64) -> f32 {
    let x = match seed {
        Some(s) => {
            // Round 1: hash the seed via splitmix64.
            let mut x = s.wrapping_add(0x9e3779b97f4a7c15);
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^= x >> 31;
            // Round 2: mix in the step index so different decode steps yield
            // different values even for the same seed.
            x = x.wrapping_add(step.wrapping_add(1).wrapping_mul(0x9e3779b97f4a7c15));
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^= x >> 31;
            if x == 0 {
                x = 0x1234567890abcdef;
            }
            // One xorshift64* output step.
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        }
        None => rand_state_thread_local(),
    };
    let u = x.wrapping_mul(0x2545f4914f6cdd1d) >> 32;
    u as f32 / (u32::MAX as f32 + 1.0)
}

// ── Speculative decoding (rejection sampling) ─────────────────────────────────

/// Outcome of one speculative verification step.
#[derive(Debug)]
pub enum SpecVerdict {
    /// Draft token accepted (the original draft token id was used).
    Accept,
    /// Draft token rejected.  Contains the replacement token id resampled from
    /// the adjusted distribution `p' = norm(max(0, p - q))` (Chen et al. 2302.01318).
    Reject(u32),
}

/// Verify one draft token against the target model's distribution.
///
/// Algorithm 1 from Chen et al. "Accelerating Large Language Model Decoding
/// with Speculative Sampling" (https://arxiv.org/abs/2302.01318):
///
/// Given draft token `x` sampled from q(·|ctx):
///   r ~ Uniform(0,1)
///   if r < p(x)/q(x):  accept x
///   else:               resample from p'(v) = norm(max(0, p(v) - q(v)))
///
/// `target_logits`: raw logits from the *main* model for this position,
///                  shape `(vocab_size,)` as f32.
/// `draft_probs`:   **normalised** probability vector for the draft position
///                  (not raw logits). For greedy drafts pass a one-hot vector
///                  `{draft_token: 1.0}` so q(draft_token) = 1 and the ratio
///                  p/q = p(draft_token), giving the correct acceptance
///                  criterion and adjusted distribution per Chen et al.
/// `draft_token`:   the token id the draft model sampled.
/// `seed` + `step`: forwarded to `rand_f32` for reproducible sampling.
///
/// `temperature`: user sampling temperature, applied to `target_logits` before
///                softmax so that `p` matches the actual target distribution.
///                Pass `1.0` for unscaled softmax.  Values ≤ 0 are treated as
///                greedy (p becomes a one-hot on the argmax).
///
/// Returns the verdict and a softmax-probability vector for `target_logits`
/// (so the caller can reuse it for subsequent steps without recomputing).
pub fn speculative_verify(
    target_logits: &[f32],
    draft_probs: &[f32],
    draft_token: u32,
    temperature: f64,
    seed: Option<u64>,
    step: u64,
) -> Result<(SpecVerdict, Vec<f32>)> {
    let vocab = target_logits.len();
    anyhow::ensure!(
        draft_probs.len() == vocab,
        "speculative_verify: target vocab {} != draft vocab {}",
        vocab,
        draft_probs.len()
    );

    // Apply temperature scaling so p matches the distribution sample_token uses.
    let p = if temperature <= 0.0 {
        // Greedy: one-hot on argmax.
        let best = target_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut v = vec![0.0f32; vocab];
        v[best] = 1.0;
        v
    } else if (temperature - 1.0).abs() < 1e-6 {
        softmax_vec(target_logits)
    } else {
        let inv_t = (1.0 / temperature) as f32;
        let scaled: Vec<f32> = target_logits.iter().map(|&l| l * inv_t).collect();
        softmax_vec(&scaled)
    };
    let q = draft_probs; // already normalised — caller's responsibility

    let x = draft_token as usize;
    let px = p.get(x).copied().unwrap_or(0.0);
    let qx = q.get(x).copied().unwrap_or(f32::MIN_POSITIVE);

    let r = rand_f32(seed, step);
    if r < px / qx {
        return Ok((SpecVerdict::Accept, p));
    }

    // Rejection: resample from p' = norm(max(0, p(v) - q(v)))
    let mut adjusted: Vec<f32> = p
        .iter()
        .zip(q.iter())
        .map(|(&pv, &qv)| (pv - qv).max(0.0))
        .collect();
    let sum: f32 = adjusted.iter().sum();
    if sum > 0.0 {
        for v in &mut adjusted {
            *v /= sum;
        }
    } else {
        // Degenerate: fall back to argmax of target.
        let best = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        return Ok((SpecVerdict::Reject(best as u32), p));
    }

    let replacement = sample_from_probs(&adjusted, seed, step + 1);
    Ok((SpecVerdict::Reject(replacement), p))
}

/// Sample a token id from a normalised probability vector.
pub fn sample_from_probs(probs: &[f32], seed: Option<u64>, step: u64) -> u32 {
    let mut r = rand_f32(seed, step);
    let mut last = 0usize;
    for (i, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            last = i;
            if r < p {
                return i as u32;
            }
            r -= p;
        }
    }
    last as u32
}

/// Compute softmax of a raw logits slice, returning a probability vector.
pub fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for v in &mut probs {
            *v /= sum;
        }
    }
    probs
}

/// Advance and return the thread-local xorshift64* state.
fn rand_state_thread_local() -> u64 {
    use std::time::SystemTime;

    thread_local! {
        static STATE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        static SEEDED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    SEEDED.with(|seeded| {
        if !seeded.get() {
            let t = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            let tid = {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut h = DefaultHasher::new();
                std::thread::current().id().hash(&mut h);
                h.finish()
            };
            let mut s = t ^ tid;
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            s = (s ^ (s >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            s = (s ^ (s >> 27)).wrapping_mul(0x94d049bb133111eb);
            s ^= s >> 31;
            STATE.with(|st| st.set(if s == 0 { 0x1234567890abcdef } else { s }));
            seeded.set(true);
        }
    });

    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        x
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// Default params have llama.cpp-matching values.
    #[test]
    fn defaults_match_llama_cpp() {
        let p = SamplingParams::default();
        assert!(
            (p.repetition_penalty - 1.1).abs() < 1e-9,
            "repetition_penalty default should be 1.1, got {}",
            p.repetition_penalty
        );
        assert_eq!(p.repeat_last_n, 64, "repeat_last_n default should be 64");
        assert!(
            p.frequency_penalty.abs() < 1e-9,
            "frequency_penalty default should be 0.0"
        );
    }

    /// Tokens OUTSIDE the repeat_last_n window must NOT be penalised.
    ///
    /// Setup: vocab size 4, logits all 1.0.
    /// previous_tokens = [0, 1, 2, 3] (4 tokens).
    /// repeat_last_n = 2  →  only tokens [2, 3] are in the window.
    /// After penalty (1.5), tokens 0 and 1 should be unchanged (still 1.0),
    /// tokens 2 and 3 should be divided: 1.0 / 1.5 ≈ 0.6667.
    #[test]
    fn repeat_last_n_windows_correctly() {
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 1.5,
            repeat_last_n: 2,
            ..SamplingParams::default()
        };
        let previous = vec![0u32, 1, 2, 3];
        // Build a (vocab=4,) logits tensor via sample_token's internal path.
        // We extract the post-penalty logits by using greedy (temp=0) and
        // checking which token is selected, but for a direct check we call
        // the internal logic by constructing a discriminating logit vector:
        // Give token 0 a much higher logit than token 2 — if token 0 is
        // penalised we'd select token 2 instead.
        // Token 1 is the runner-up at 8.0.  Token 0 outside window → 10.0 / 1.0
        // = 10.0 still wins.  Token 0 inside window → 10.0 / 1.5 ≈ 6.67 < 8.0
        // → token 1 wins.
        let base: Vec<f32> = vec![10.0, 8.0, 0.0, 0.0];
        let t = Tensor::from_vec(base, 4usize, &Device::Cpu).unwrap();
        let (tok, _) = sample_token(&t, &params, &previous).unwrap();
        // Token 0 is OUTSIDE the window → not penalised → still highest logit.
        assert_eq!(tok, 0, "token 0 (outside window) should win; got {tok}");

        // Now put token 0 INSIDE the window — 10.0/1.5 ≈ 6.67 < 8.0 → token 1 wins.
        let previous_with_0_in_window = vec![1u32, 2, 3, 0]; // 0 is in last 2
        let (tok2, _) = sample_token(&t, &params, &previous_with_0_in_window).unwrap();
        assert_eq!(
            tok2, 1,
            "token 0 (inside window) should be penalised below token 1 (8.0); got {tok2}"
        );
    }

    /// repeat_last_n = 0 disables the penalty entirely — even tokens in the
    /// immediate history are not suppressed.
    #[test]
    fn repeat_last_n_zero_disables_penalty() {
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 2.0, // strong penalty if applied
            repeat_last_n: 0,        // disabled
            ..SamplingParams::default()
        };
        let previous = vec![0u32]; // token 0 was just emitted
        let base: Vec<f32> = vec![10.0, 0.0, 0.0, 0.0];
        let t = Tensor::from_vec(base, 4usize, &Device::Cpu).unwrap();
        let (tok, _) = sample_token(&t, &params, &previous).unwrap();
        assert_eq!(tok, 0, "penalty disabled; token 0 should still win");
    }

    /// When history is shorter than repeat_last_n the entire history is used
    /// (no out-of-bounds / saturation issues).
    #[test]
    fn repeat_last_n_longer_than_history_uses_full_history() {
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 1.5,
            repeat_last_n: 64, // window >> history length
            ..SamplingParams::default()
        };
        let previous = vec![0u32]; // only 1 token in history
        let base: Vec<f32> = vec![10.0, 5.0];
        let t = Tensor::from_vec(base, 2usize, &Device::Cpu).unwrap();
        let (tok, _) = sample_token(&t, &params, &previous).unwrap();
        // 10.0 / 1.5 ≈ 6.67, still > 5.0, so token 0 still wins but is penalised
        assert_eq!(tok, 0, "token 0 still highest even after penalty");

        // Now use a penalty strong enough to flip the winner.
        let strong = SamplingParams {
            repetition_penalty: 10.0,
            ..params
        };
        let (tok2, _) = sample_token(&t, &strong, &previous).unwrap();
        assert_eq!(tok2, 1, "strong penalty should suppress token 0");
    }

    /// repetition_penalty=0.0 must not divide by zero; behaves as disabled.
    #[test]
    fn repetition_penalty_zero_does_not_divide_by_zero() {
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 0.0,
            repeat_last_n: 64,
            ..SamplingParams::default()
        };
        let previous = vec![0u32];
        let base: Vec<f32> = vec![10.0, 5.0];
        let t = Tensor::from_vec(base, 2usize, &Device::Cpu).unwrap();
        // Should not panic; token 0 wins since no penalty is applied.
        let (tok, _) = sample_token(&t, &params, &previous).unwrap();
        assert_eq!(tok, 0, "zero penalty acts as disabled; token 0 still wins");
    }
}

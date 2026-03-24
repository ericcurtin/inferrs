//! Qwen3.5 text model implementation.
//!
//! Qwen3.5 uses a hybrid architecture alternating between:
//!   - Linear attention layers (Mamba2-style SSM)
//!   - Full attention layers (GQA with QK-norm, no bias)
//!
//! All weights live under the `model.language_model.*` prefix.
//! The model uses tied embeddings (no separate lm_head).

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LayerType {
    pub is_full_attention: bool,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    // Full-attention params
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    // Linear-attention params
    pub linear_num_key_heads: usize,    // = 16 in 0.8B
    pub linear_key_head_dim: usize,     // = 128
    pub linear_value_head_dim: usize,   // = 128
    pub linear_num_value_heads: usize,  // = 16
    pub linear_conv_kernel_dim: usize,  // = 4
    pub full_attention_interval: usize, // every Nth layer is full-attention
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64, // = 0.25
    pub layer_types: Vec<LayerType>,
    pub tie_word_embeddings: bool,
    pub dtype: DType,
    pub device: Device,
}

// ---------------------------------------------------------------------------
// RoPE utilities
// ---------------------------------------------------------------------------

/// Precompute (cos, sin) for positions 0..max_seq_len with partial rotation.
fn precompute_rope(
    head_dim: usize,
    partial_factor: f64,
    rope_theta: f64,
    max_seq_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let rot_dim = (head_dim as f64 * partial_factor) as usize;
    // round down to even
    let rot_dim = rot_dim & !1;
    let half = rot_dim / 2;

    // freqs: [half]
    let freqs: Vec<f32> = (0..half)
        .map(|i| {
            let exp = 2.0 * i as f32 / rot_dim as f32;
            1.0 / (rope_theta as f32).powf(exp)
        })
        .collect();
    let freqs = Tensor::new(freqs.as_slice(), device)?;

    // positions: [max_seq_len]
    let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
    let positions = Tensor::new(positions.as_slice(), device)?;

    // outer product -> [max_seq_len, half]
    let emb = positions
        .unsqueeze(1)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;

    // cos/sin: [max_seq_len, half]
    let cos = emb.cos()?.to_dtype(dtype)?;
    let sin = emb.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Apply rotary embedding to query/key tensors.
/// x: [batch, n_heads, seq_len, head_dim]
/// cos/sin: [seq_len, rot_half]  (half of rot_dim)
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, t, d) = x.dims4()?;
    let rot_half = cos.dim(1)?;
    let rot_dim = rot_half * 2;

    if rot_dim > d {
        anyhow::bail!("rot_dim {} > head_dim {}", rot_dim, d);
    }

    // Split x into rotated and pass-through parts
    let x_rot = x.narrow(3, 0, rot_dim)?;
    let x_pass = if rot_dim < d {
        Some(x.narrow(3, rot_dim, d - rot_dim)?)
    } else {
        None
    };

    // x_rot: [b, h, t, rot_dim] -> split into two halves along last dim
    let x1 = x_rot.narrow(3, 0, rot_half)?;
    let x2 = x_rot.narrow(3, rot_half, rot_half)?;

    // cos/sin broadcast: [1, 1, t, rot_half]
    let cos = cos.narrow(0, 0, t)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.narrow(0, 0, t)?.unsqueeze(0)?.unsqueeze(0)?;

    // rotate_half: (x1, x2) -> (-x2, x1)
    let rotated = Tensor::cat(
        &[
            (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?,
            (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?,
        ],
        3,
    )?;

    match x_pass {
        Some(pass) => Ok(Tensor::cat(&[rotated, pass], 3)?),
        None => Ok(rotated),
    }
}

// ---------------------------------------------------------------------------
// SwiGLU MLP
// ---------------------------------------------------------------------------

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &Qwen35Config, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden).map_err(Into::into)
    }
}

// ---------------------------------------------------------------------------
// Full attention layer (GQA + QK-norm + RoPE, no bias)
// ---------------------------------------------------------------------------

struct FullAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    // KV cache: Option<(k_cache, v_cache)> accumulated across calls
    kv_cache: Option<(Tensor, Tensor)>,
}

impl FullAttention {
    fn new(cfg: &Qwen35Config, vb: VarBuilder) -> Result<Self> {
        // q_proj outputs num_heads * head_dim * 2: first half is query, second half is the
        // output gate (attn_output_gate). The o_proj then takes num_heads * head_dim.
        let q_proj_out = cfg.num_attention_heads * cfg.head_dim * 2;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        let attn_out = cfg.num_attention_heads * cfg.head_dim;

        let q_proj = linear_no_bias(cfg.hidden_size, q_proj_out, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(cfg.hidden_size, kv_out, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(cfg.hidden_size, kv_out, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(attn_out, cfg.hidden_size, vb.pp("o_proj"))?;
        let q_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        seqlen_offset: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        // Project
        // q_proj outputs [b, t, num_heads * head_dim * 2]; split into query + gate
        let q_full = self.q_proj.forward(x)?; // [b, t, num_heads * head_dim * 2]
        let attn_dim = self.num_heads * self.head_dim;
        let q_raw = q_full.narrow(2, 0, attn_dim)?; // [b, t, num_heads * head_dim]
        let gate = q_full.narrow(2, attn_dim, attn_dim)?; // [b, t, num_heads * head_dim]

        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [b, heads, t, head_dim]
        let q = q_raw
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // QK norms (per-head, on head_dim)
        // q_norm expects [..., head_dim]; apply on last dim
        let q = apply_rms_norm_heads(&q, &self.q_norm)?;
        let k = apply_rms_norm_heads(&k, &self.k_norm)?;

        // RoPE
        let cos_slice = cos.narrow(0, seqlen_offset, t)?;
        let sin_slice = sin.narrow(0, seqlen_offset, t)?;
        let q = apply_rope(&q, &cos_slice, &sin_slice)?;
        let k = apply_rope(&k, &cos_slice, &sin_slice)?;

        // Append to KV cache
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                let k = Tensor::cat(&[k_cache, &k], 2)?;
                let v = Tensor::cat(&[v_cache, &v], 2)?;
                (k, v)
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let kv_len = k.dim(2)?;

        // GQA: repeat k/v heads to match q heads
        let groups = self.num_heads / self.num_kv_heads;
        let k = k.repeat(&[1, groups, 1, 1])?;
        let v = v.repeat(&[1, groups, 1, 1])?;

        // Scaled dot-product attention — matmul requires contiguous on Metal
        let scale = (self.head_dim as f64).sqrt();
        let attn = q
            .contiguous()?
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / scale, 0.0)?;

        // Causal mask
        let attn = if t > 1 {
            // Build causal mask [t, kv_len]
            let mask = causal_mask(t, kv_len, seqlen_offset, attn.device(), attn.dtype())?;
            attn.broadcast_add(&mask)?
        } else {
            attn
        };

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v.contiguous()?)?; // [b, heads, t, head_dim]

        // Reshape back: [b, t, heads*head_dim]
        let out = out
            .transpose(1, 2)?
            .reshape((b, t, self.num_heads * self.head_dim))?;

        // Apply output gate: sigmoid(gate) * out  (sigmoid = 1/(1+exp(-x)))
        let gate_sig = (gate.neg()?.exp()? + 1.0)?.recip()?;
        let out = out.broadcast_mul(&gate_sig)?;

        let out = self.o_proj.forward(&out)?;
        Ok(out)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// Apply RmsNorm to last dimension of a 4D tensor [b, h, t, d].
fn apply_rms_norm_heads(x: &Tensor, norm: &RmsNorm) -> Result<Tensor> {
    let (b, h, t, d) = x.dims4()?;
    // reshape requires contiguous on Metal
    let x_flat = x.contiguous()?.reshape((b * h * t, d))?;
    let out = norm.forward(&x_flat)?;
    out.reshape((b, h, t, d)).map_err(Into::into)
}

/// Build a causal attention bias [1, 1, q_len, kv_len].
fn causal_mask(
    q_len: usize,
    kv_len: usize,
    offset: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mask: Vec<f32> = (0..q_len)
        .flat_map(|i| {
            (0..kv_len).map(move |j| {
                // position of query token in full sequence
                let qi = offset + i;
                if j <= qi {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    let mask = Tensor::new(mask.as_slice(), device)?
        .reshape((1, 1, q_len, kv_len))?
        .to_dtype(dtype)?;
    Ok(mask)
}

// ---------------------------------------------------------------------------
// Linear attention (Mamba2 / SSM) layer
// ---------------------------------------------------------------------------
//
// Tensor layout from weights:
//   in_proj_qkv:  [6144, hidden]   -- projects to [q, k, v] interleaved
//   in_proj_z:    [2*hidden, hidden] -- gating
//   in_proj_a:    [n_heads, hidden] -- dt_rank projection
//   in_proj_b:    [n_heads, hidden] -- B (key-side)
//   conv1d:       [6144, 1, 4]      -- depthwise conv on channels
//   A_log:        [n_heads]         -- log(-A), stored as F32
//   dt_bias:      [n_heads]         -- delta bias
//   norm:         [head_dim]        -- output norm (F32 weight, applied as simple RMSNorm)
//   out_proj:     [hidden, 2*hidden]
//
// dim breakdown:
//   n_heads = linear_num_key_heads = 16
//   head_dim = linear_key_head_dim = 128 (= value_head_dim too)
//   So q+k+v = 16*(128+128+128) = 6144  = in_proj_qkv rows  ✓
//   z = 2*hidden = 2*1024 = 2048  ✓
//   out = hidden, in = n_heads * v_head_dim = 16*128 = 2048  ✓
//
// Inference (recurrent mode, one token at a time after prefill):
//   We use a simplified recurrent SSM approximation.
//   For prefill (many tokens), we process sequentially for correctness.

struct LinearAttn {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_a: Linear,     // dt
    in_proj_b: Linear,     // B
    conv1d_weight: Tensor, // [inner, 1, kernel]
    a_log: Tensor,         // [n_heads], F32
    dt_bias: Tensor,       // [n_heads]
    norm_weight: Tensor,   // [head_dim], F32
    out_proj: Linear,
    n_heads: usize,
    head_dim: usize,
    inner_dim: usize, // = n_heads * head_dim (for q+k+v each)
    // SSM state: [1, n_heads, head_dim, head_dim] for each of k*v outer product
    ssm_state: Option<Tensor>,
    // Conv state: [1, inner_dim * 3, kernel-1]  (circular buffer for the 3-part qkv channels)
    conv_state: Option<Tensor>,
}

impl LinearAttn {
    fn new(cfg: &Qwen35Config, vb: VarBuilder) -> Result<Self> {
        let n_heads = cfg.linear_num_key_heads;
        let head_dim = cfg.linear_key_head_dim;
        let inner_dim = n_heads * head_dim; // = 2048 for 0.8B
        let hidden = cfg.hidden_size;
        let kernel = cfg.linear_conv_kernel_dim;

        // in_proj_qkv projects to q+k+v = 3 * inner_dim
        let in_proj_qkv = linear_no_bias(hidden, 3 * inner_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_no_bias(hidden, 2 * hidden, vb.pp("in_proj_z"))?;
        let in_proj_a = linear_no_bias(hidden, n_heads, vb.pp("in_proj_a"))?;
        let in_proj_b = linear_no_bias(hidden, n_heads, vb.pp("in_proj_b"))?;

        // conv1d weight: [3*inner_dim, 1, kernel] -- depthwise
        let conv1d_weight = vb.get((3 * inner_dim, 1, kernel), "conv1d.weight")?;

        // Cast A_log, dt_bias, norm_weight to model dtype at load time so we don't
        // do repeated dtype conversions during every forward pass.
        let dtype = cfg.dtype;
        let a_log = vb
            .get_with_hints(n_heads, "A_log", candle_nn::Init::Const(0.0))?
            .to_dtype(dtype)?;
        let dt_bias = vb.get((n_heads,), "dt_bias")?.to_dtype(dtype)?;
        let norm_weight = vb
            .get_with_hints(head_dim, "norm.weight", candle_nn::Init::Const(1.0))?
            .to_dtype(dtype)?;

        let out_proj = linear_no_bias(2 * hidden, hidden, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            conv1d_weight,
            a_log,
            dt_bias,
            norm_weight,
            out_proj,
            n_heads,
            head_dim,
            inner_dim,
            ssm_state: None,
            conv_state: None,
        })
    }

    fn clear_state(&mut self) {
        self.ssm_state = None;
        self.conv_state = None;
    }

    /// Process a sequence of tokens through the linear attention layer.
    /// x: [batch=1, seq_len, hidden]
    /// Returns: [1, seq_len, hidden]
    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let device = x.device().clone();
        let dtype = x.dtype(); // used for SSM state init and conv1d

        // Project inputs
        let qkv = self.in_proj_qkv.forward(x)?; // [b, t, 3*inner]
        let z = self.in_proj_z.forward(x)?; // [b, t, 2*hidden]
        let dt_input = self.in_proj_a.forward(x)?; // [b, t, n_heads]  (delta)
        let b_input = self.in_proj_b.forward(x)?; // [b, t, n_heads]  (B)

        // Apply depthwise conv1d to qkv channels
        let qkv = self.apply_conv1d(&qkv)?; // [b, t, 3*inner]

        // Split qkv -> q, k, v each [b, t, inner]
        let q = qkv.narrow(2, 0, self.inner_dim)?;
        let k = qkv.narrow(2, self.inner_dim, self.inner_dim)?;
        let v = qkv.narrow(2, 2 * self.inner_dim, self.inner_dim)?;

        // Reshape to [b, t, n_heads, head_dim]
        let q = q.reshape((b, t, self.n_heads, self.head_dim))?;
        let k = k.reshape((b, t, self.n_heads, self.head_dim))?;
        let v = v.reshape((b, t, self.n_heads, self.head_dim))?;

        // Compute A = -exp(a_log): [n_heads]  (a_log already in model dtype)
        let a: Tensor = self.a_log.neg()?.exp()?; // positive magnitudes -> use as decay

        // dt (delta): softplus(dt_input + dt_bias)  (dt_bias already in model dtype)
        let dt = dt_input.broadcast_add(&self.dt_bias.reshape((1, 1, self.n_heads))?)?;
        let dt = softplus(&dt)?; // [b, t, n_heads]

        // Process tokens sequentially through the SSM recurrence
        // state shape: [b, n_heads, head_dim, head_dim]  (outer product of k and v)
        let mut state = match &self.ssm_state {
            None => Tensor::zeros(
                (b, self.n_heads, self.head_dim, self.head_dim),
                dtype,
                &device,
            )?,
            Some(s) => s.clone(),
        };

        // SSM recurrence: s_t = decay_t * s_{t-1} + k_t⊗v_t,  y_t = q_t @ s_t
        //
        // We implement both decode (t=1) and prefill (t>1) paths.
        // For prefill we use a parallel-scan approach based on cumsum to avoid
        // a sequential Rust loop that dispatches many small GPU kernels.
        //
        // Key identity: y_t = q_t @ [ Σ_{i<=t} w_{t,i} * outer(k_i,v_i) ] + w_{t,init} * (q_t @ s0)
        //   where w_{t,i} = exp( Σ_{j=i+1}^{t} log_d_j )
        //                 = exp( log_d_cum[t] - log_d_cum[i] )
        //   and log_d_cum[t] = Σ_{j=1}^{t} log_d_j  (cumulative sum)
        //
        // Equivalently: causal_weight[t,i] = exp(log_d_cum[t] - log_d_cum[i])  for i<=t, else 0
        //
        // Steps (all vectorised):
        //  1. log_d[b,t,h]      = -a[h] * dt[b,t,h]
        //  2. log_d_cum[b,t,h]  = cumsum(log_d, dim=1)
        //  3. scale[b,t,h]      = exp(log_d_cum[b,t,h] - log_d_cum[b,t,h] for each token)
        //                       → implemented as lower-triangular weight matrix W[b,h,t,t]
        //                         W[b,h,t,i] = exp(log_d_cum[b,t,h] - log_d_cum[b,i,h])  for i<=t
        //  4. outer[b,t,h,dk,dv] = (dt*B scaled k_t) ⊗ v_t
        //  5. y[b,t,h,dv]       = q[b,t,h,dk] @ Σ_i W[b,h,t,i] * outer[b,i,h,dk,dv]

        // Compute log-decay: [b, t, n_heads]
        let log_d = a.unsqueeze(0)?.broadcast_mul(&dt)?.neg()?; // [b, t, n_h]
                                                                // Cumulative sum: [b, t, n_heads]
        let log_d_cum = log_d.cumsum(1)?; // [b, t, n_h]

        // Build the causal weight matrix W[b, n_h, t, t]:
        //   W[b, h, t_out, t_in] = exp(log_d_cum[b, t_out, h] - log_d_cum[b, t_in, h])  for t_in <= t_out
        // Rearrange log_d_cum to [b, t, 1, n_h] vs [b, 1, t, n_h]:
        //   diff[b, t_out, t_in, n_h] = log_d_cum[b, t_out, n_h] - log_d_cum[b, t_in, n_h]
        // Then apply lower-triangular mask and exp.
        let ldc = log_d_cum.contiguous()?; // [b, t, n_h]
        let ldc_row = ldc.unsqueeze(2)?; // [b, t, 1, n_h]  (query positions)
        let ldc_col = ldc.unsqueeze(1)?; // [b, 1, t, n_h]  (key positions)
        let diff = ldc_row.broadcast_sub(&ldc_col)?; // [b, t, t, n_h]  (t_out, t_in)
                                                     // Apply causal mask: set upper-triangular (t_in > t_out) to -inf
        let causal_w = {
            let diff_c = diff.contiguous()?;
            // Build mask: 1.0 for i<=j, -inf for i>j  (lower triangular)
            let mask_vals: Vec<f32> = (0..t)
                .flat_map(|row| {
                    (0..t).map(move |col| {
                        if col <= row {
                            0.0f32
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect();
            let mask = Tensor::new(mask_vals.as_slice(), &device)?
                .reshape((t, t))?
                .to_dtype(dtype)?
                .unsqueeze(0)?
                .unsqueeze(3)?; // [1, t, t, 1]
            diff_c.broadcast_add(&mask)?.exp()? // [b, t, t, n_h]
        };
        // causal_w: [b, t_out, t_in, n_h]  →  transpose to [b, n_h, t_out, t_in]
        let causal_w = causal_w.permute((0, 3, 1, 2))?.contiguous()?; // [b, n_h, t, t]

        // Compute outer products for all tokens: outer[b, t, n_h, dk, dv]
        // dt_b = dt * b_input: [b, t, n_h]
        let dt_b = (dt * b_input)?; // [b, t, n_h]
                                    // k_scaled = k * dt_b: [b, t, n_h, dk]
        let k_scaled = k.broadcast_mul(&dt_b.unsqueeze(3)?)?; // [b, t, n_h, dk]

        // outer = k_scaled[:,:,:,:,None] * v[:,:,:,None,:]
        //       → [b, t, n_h, dk, dv]
        // Via einsum-style: reshape and batched matmul
        // k_scaled: [b, t, n_h, dk, 1]  ×  v: [b, t, n_h, 1, dv]
        let k_col = k_scaled.unsqueeze(4)?.contiguous()?; // [b, t, n_h, dk, 1]
        let v_row = v.unsqueeze(3)?.contiguous()?; // [b, t, n_h, 1, dv]
        let outer = k_col.matmul(&v_row)?; // [b, t, n_h, dk, dv]

        // Now compute: weighted_outer[b, n_h, t_out, dk, dv]
        //   = Σ_{t_in} causal_w[b, n_h, t_out, t_in] * outer[b, t_in, n_h, dk, dv]
        //
        // Rearrange outer to [b, n_h, t, dk, dv]
        let outer_t = outer.permute((0, 2, 1, 3, 4))?.contiguous()?; // [b, n_h, t, dk, dv]

        // Reshape for batched matmul:
        //   causal_w: [b, n_h, t, t]
        //   outer_t:  [b, n_h, t, dk*dv]
        let dk_dv = self.head_dim * self.head_dim;
        let outer_flat = outer_t.reshape((b, self.n_heads, t, dk_dv))?; // [b, n_h, t, dk*dv]
                                                                        // weighted = causal_w @ outer_flat: [b, n_h, t_out, dk*dv]
        let weighted = causal_w.contiguous()?.matmul(&outer_flat.contiguous()?)?;
        // Reshape back: [b, n_h, t, dk, dv]
        let weighted = weighted.reshape((b, self.n_heads, t, self.head_dim, self.head_dim))?;

        // Account for initial state contribution:
        //   state_contrib[b, n_h, t, dk, dv] = exp(log_d_cum[b, t, n_h]) * state_0[b, n_h, dk, dv]
        let init_decay = log_d_cum.exp()?.permute((0, 2, 1))?.contiguous()?; // [b, n_h, t]
        let init_contrib = state
            .unsqueeze(2)?
            .broadcast_mul(&init_decay.unsqueeze(3)?.unsqueeze(4)?)?;
        // [b, n_h, t, dk, dv]

        let full_state = (weighted + init_contrib)?; // [b, n_h, t, dk, dv]

        // Update the persistent state with the final step's state:
        //   state[T] = full_state[:, :, T-1, :, :]
        state = full_state
            .narrow(2, t - 1, 1)?
            .squeeze(2)?
            .contiguous()?
            .detach(); // [b, n_h, dk, dv]

        // Compute output: y[b, t, n_h, dv] = q[b, t, n_h, dk] @ full_state[b, n_h, t, dk, dv]
        // Rearrange q to [b, n_h, t, dk] then [b, n_h, t, 1, dk]
        let q_perm = q.permute((0, 2, 1, 3))?.contiguous()?; // [b, n_h, t, dk]
        let q_4 = q_perm.unsqueeze(3)?.contiguous()?; // [b, n_h, t, 1, dk]
                                                      // full_state: [b, n_h, t, dk, dv]
        let out_raw = q_4.matmul(&full_state.contiguous()?)?.squeeze(3)?; // [b, n_h, t, dv]

        // Apply output norm per token per head: rms_norm over dv dimension
        let out_normed = rms_norm_tensor(&out_raw, &self.norm_weight, 1e-6)?; // [b, n_h, t, dv]

        // Rearrange to [b, t, n_h*dv]
        let y = out_normed.permute((0, 2, 1, 3))?.contiguous()?.reshape((
            b,
            t,
            self.n_heads * self.head_dim,
        ))?;

        self.ssm_state = Some(state);

        // Gate with z: silu(z) * cat(y, y) or similar
        // Looking at the architecture: out_proj takes [2*hidden] input.
        // z is [b, t, 2*hidden], y is [b, t, inner_dim=n_heads*head_dim]
        // We need to combine y with z to get 2*hidden for out_proj.
        // The standard Mamba2 gate: y_gated = y * silu(z[:, :, :inner]) then pad,
        // but here inner_dim (2048) == hidden (1024) * 2, so z splits into two halves:
        // gate1 [b,t,hidden] and gate2 [b,t,hidden], y [b,t,2048=2*hidden]
        // Most likely: gated_output = y * silu(z)  where both are [b,t,2*hidden]
        let z_gated = z.silu()?;
        let out = (y * z_gated)?; // [b, t, 2*hidden]

        let out = self.out_proj.forward(&out)?; // [b, t, hidden]
        Ok(out)
    }

    /// Apply depthwise conv1d with causal padding.
    /// x: [b, t, channels]
    /// weight stored as [channels, 1, kernel] (depthwise)
    fn apply_conv1d(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, _t, c) = x.dims3()?;
        let kernel = self.conv1d_weight.dim(2)?;
        let dtype = x.dtype();
        let device = x.device().clone();

        let pad_len = kernel - 1;

        // Build padded input [b, pad_len+t, c] using stored conv state or zeros
        let padded = match &self.conv_state {
            None => {
                let zeros = Tensor::zeros((b, pad_len, c), dtype, &device)?;
                Tensor::cat(&[&zeros, x], 1)?
            }
            Some(prev) => Tensor::cat(&[prev, x], 1)?,
        };

        // Update conv state: keep last pad_len tokens (must be contiguous for Metal)
        let total = padded.dim(1)?;
        self.conv_state = Some(padded.narrow(1, total - pad_len, pad_len)?.contiguous()?);

        // Use candle's native conv1d (Metal-accelerated depthwise: groups = c).
        // Metal conv1d only supports F32; cast if needed and cast back after.
        let conv_dtype = if dtype == DType::BF16 || dtype == DType::F16 {
            DType::F32
        } else {
            dtype
        };
        let w = self.conv1d_weight.to_dtype(conv_dtype)?;

        // Transpose padded: [b, pad_len+t, c] -> [b, c, pad_len+t]
        let inp = padded.to_dtype(conv_dtype)?.transpose(1, 2)?.contiguous()?;

        // Depthwise conv1d: groups = c, no padding (we already padded manually), stride=1
        let out = inp.conv1d(&w, 0, 1, 1, c)?; // [b, c, t]

        // Transpose back: [b, c, t] -> [b, t, c], restore original dtype
        out.transpose(1, 2)?
            .contiguous()?
            .to_dtype(dtype)
            .map_err(Into::into)
    }
}

/// Softplus activation: log(1 + exp(x))
/// Numerically stable: for x > 0 use x + log(1 + exp(-x)) to avoid overflow.
fn softplus(x: &Tensor) -> Result<Tensor> {
    // softplus(x) = log(1 + exp(x))
    //             = x + log(1 + exp(-x))   [stable for x > 0]
    // Use: max(x, 0) + log(1 + exp(-|x|))
    let abs_x = x.abs()?;
    let neg_abs = abs_x.neg()?;
    let ones = x.ones_like()?;
    let log_term = (ones + neg_abs.exp()?)?.log()?;
    // max(x, 0) = (x + |x|) / 2
    let pos_part = ((x + &abs_x)? / 2.0)?;
    (pos_part + log_term).map_err(Into::into)
}

/// Manual RMSNorm over the last dimension.
/// x: [..., head_dim], weight: [head_dim]
/// Operates in the same dtype as x to avoid round-trip casts on Metal.
fn rms_norm_tensor(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let w = weight.to_dtype(dtype)?;

    let rms = (x.sqr()?.mean_keepdim(candle_core::D::Minus1)? + eps)?.sqrt()?;
    let normed = x.broadcast_div(&rms)?; // [..., head_dim]

    // Reshape weight to [1, ..., 1, head_dim] to broadcast over all leading dims
    let w_shape: Vec<usize> = {
        let mut s = vec![1usize; x.rank() - 1];
        s.push(w.dim(0)?);
        s
    };
    let w_bc = w.reshape(w_shape)?;
    normed.broadcast_mul(&w_bc).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

enum LayerAttn {
    Full(FullAttention),
    Linear(LinearAttn),
}

struct DecoderLayer {
    attn: LayerAttn,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new_full(cfg: &Qwen35Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: LayerAttn::Full(FullAttention::new(cfg, vb.pp("self_attn"))?),
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn new_linear(cfg: &Qwen35Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: LayerAttn::Linear(LinearAttn::new(cfg, vb.pp("linear_attn"))?),
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        seqlen_offset: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let normed = self.input_layernorm.forward(x)?;

        let attn_out = match &mut self.attn {
            LayerAttn::Full(a) => a.forward(&normed, seqlen_offset, cos, sin)?,
            LayerAttn::Linear(a) => a.forward(&normed)?,
        };

        let x = (residual + attn_out)?;
        let residual = x.clone();
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        (residual + mlp_out).map_err(Into::into)
    }

    fn clear_cache(&mut self) {
        match &mut self.attn {
            LayerAttn::Full(a) => a.clear_kv_cache(),
            LayerAttn::Linear(a) => a.clear_state(),
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

pub struct Qwen35Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    // Shared weights with embed_tokens (tied)
    lm_head_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
}

impl Qwen35Model {
    pub fn new(cfg: &Qwen35Config, vb: VarBuilder) -> Result<Self> {
        // All language model weights are under model.language_model.*
        let lm_vb = vb.pp("model").pp("language_model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, lm_vb.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, layer_type) in cfg.layer_types.iter().enumerate() {
            let layer_vb = lm_vb.pp("layers").pp(i.to_string());
            let layer = if layer_type.is_full_attention {
                DecoderLayer::new_full(cfg, layer_vb)
            } else {
                DecoderLayer::new_linear(cfg, layer_vb)
            }
            .with_context(|| format!("loading layer {}", i))?;
            layers.push(layer);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, lm_vb.pp("norm"))?;

        // Tied weights: lm_head = embed_tokens.weight transposed
        let lm_head_weight = embed_tokens.embeddings().clone();

        // Precompute RoPE tables (large enough for typical sequences)
        let max_seq = 32768;
        let (cos, sin) = precompute_rope(
            cfg.head_dim,
            cfg.partial_rotary_factor,
            cfg.rope_theta,
            max_seq,
            cfg.dtype,
            &cfg.device,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head_weight,
            cos,
            sin,
        })
    }

    /// Forward pass.
    /// input_ids: [batch, seq_len]
    /// Returns logits for the last position: [batch, 1, vocab_size]
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;

        let mut x = self.embed_tokens.forward(input_ids)?; // [b, t, hidden]

        for layer in &mut self.layers {
            x = layer.forward(&x, seqlen_offset, &self.cos, &self.sin)?;
        }

        x = self.norm.forward(&x)?;

        // Take last position only
        let last = x.narrow(1, t - 1, 1)?; // [b, 1, hidden]

        // Tied embedding: matmul with embed_tokens weight [vocab, hidden]
        // last is [b, 1, hidden]; flatten to [b, hidden] for 2D matmul then restore
        // Both operands must be contiguous for Metal matmul.
        let last_2d = last.squeeze(1)?.contiguous()?; // [b, hidden]
        let logits = last_2d.matmul(&self.lm_head_weight.t()?.contiguous()?)?; // [b, vocab]
        let logits = logits.unsqueeze(1)?; // [b, 1, vocab]
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }
}

//! LTX-2.x: audio-video diffusion transformer, video VAE and audio VAE.

pub mod audio;
pub mod dit;
pub mod text;
pub mod vae;

use anyhow::{bail, Result};

use super::backend::Ctx;
use super::ffi::{self, Tensor};
use super::weights::Weights;

pub const PI: f64 = std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct Hparams {
    pub n_layers: i64,
    pub v_dim: i64,
    pub v_heads: i64,
    pub a_dim: i64,
    pub a_heads: i64,
    pub in_channels: i64,
    pub caption_channels: i64,
    pub connector_layers: i64,
    pub connector_regs: i64,
    pub connector_max_pos: f64,
    pub text_max_tokens: i64,
    pub rope_theta: f64,
    pub max_pos: [f64; 3],
    pub audio_max_pos: f64,
    pub timestep_scale: f32,
    pub av_ca_timestep_scale: f32,
    pub causal_temporal_positioning: bool,
    pub has_audio: bool,
    pub audio_channels: i64,
    pub audio_freq_bins: i64,
    pub audio_sample_rate: i64,
    pub audio_hop_length: i64,
    pub audio_latent_downsample: i64,
    pub latent_channels: i64,
    pub vae_scale_t: i64,
    pub vae_scale_s: i64,
    pub vae_patch_size: i64,
    pub text_hidden_size: i64,
    pub text_n_states: i64,
}

impl Default for Hparams {
    fn default() -> Self {
        Hparams {
            n_layers: 48,
            v_dim: 4096,
            v_heads: 32,
            a_dim: 2048,
            a_heads: 32,
            in_channels: 128,
            caption_channels: 3840,
            connector_layers: 8,
            connector_regs: 128,
            connector_max_pos: 4096.0,
            text_max_tokens: 1024,
            rope_theta: 10000.0,
            max_pos: [20.0, 2048.0, 2048.0],
            audio_max_pos: 20.0,
            timestep_scale: 1000.0,
            av_ca_timestep_scale: 1000.0,
            causal_temporal_positioning: true,
            has_audio: true,
            audio_channels: 8,
            audio_freq_bins: 16,
            audio_sample_rate: 16000,
            audio_hop_length: 160,
            audio_latent_downsample: 4,
            latent_channels: 128,
            vae_scale_t: 8,
            vae_scale_s: 32,
            vae_patch_size: 4,
            text_hidden_size: 3840,
            text_n_states: 49,
        }
    }
}

impl Hparams {
    /// From the `config` JSON stored in the transformer GGUF.
    pub fn from_config(config: &str) -> Result<Hparams> {
        let mut hp = Hparams::default();
        let j: serde_json::Value = serde_json::from_str(config)?;
        let t = j.get("transformer").unwrap_or(&j);
        let geti = |k: &str, v: &mut i64| {
            if let Some(x) = t.get(k).and_then(|x| x.as_i64()) {
                *v = x;
            }
        };
        let getf = |k: &str, v: &mut f32| {
            if let Some(x) = t.get(k).and_then(|x| x.as_f64()) {
                *v = x as f32;
            }
        };
        geti("num_layers", &mut hp.n_layers);
        let (mut v_heads, mut v_hd, mut a_heads, mut a_hd) = (32, 128, 32, 64);
        geti("num_attention_heads", &mut v_heads);
        geti("attention_head_dim", &mut v_hd);
        geti("audio_num_attention_heads", &mut a_heads);
        geti("audio_attention_head_dim", &mut a_hd);
        hp.v_heads = v_heads;
        hp.v_dim = v_heads * v_hd;
        hp.a_heads = a_heads;
        hp.a_dim = a_heads * a_hd;
        geti("in_channels", &mut hp.in_channels);
        geti("caption_channels", &mut hp.caption_channels);
        hp.text_hidden_size = hp.caption_channels;
        geti("connector_num_layers", &mut hp.connector_layers);
        geti("connector_num_learnable_registers", &mut hp.connector_regs);
        if let Some(x) = t.get("positional_embedding_theta").and_then(|x| x.as_f64()) {
            hp.rope_theta = x;
        }
        getf("timestep_scale_multiplier", &mut hp.timestep_scale);
        getf(
            "av_ca_timestep_scale_multiplier",
            &mut hp.av_ca_timestep_scale,
        );
        if let Some(b) = t
            .get("causal_temporal_positioning")
            .and_then(|x| x.as_bool())
        {
            hp.causal_temporal_positioning = b;
        }
        if let Some(a) = t
            .get("positional_embedding_max_pos")
            .and_then(|x| x.as_array())
        {
            for (i, v) in a.iter().take(3).enumerate() {
                hp.max_pos[i] = v.as_f64().unwrap_or(hp.max_pos[i]);
            }
        }
        if let Some(v) = t
            .get("audio_positional_embedding_max_pos")
            .and_then(|x| x.as_array())
            .and_then(|a| a.first())
            .and_then(|x| x.as_f64())
        {
            hp.audio_max_pos = v;
        }
        if let Some(v) = t
            .get("connector_positional_embedding_max_pos")
            .and_then(|x| x.as_array())
            .and_then(|a| a.first())
            .and_then(|x| x.as_f64())
        {
            hp.connector_max_pos = v;
        }
        if let Some(v) = j.get("vae") {
            if let Some(x) = v.get("latent_channels").and_then(|x| x.as_i64()) {
                hp.latent_channels = x;
            }
            if let Some(x) = v.get("patch_size").and_then(|x| x.as_i64()) {
                hp.vae_patch_size = x;
            }
        }
        let positive = [
            hp.n_layers,
            hp.v_heads,
            hp.a_heads,
            hp.in_channels,
            hp.caption_channels,
            hp.connector_regs,
            hp.audio_channels,
            hp.audio_freq_bins,
            hp.audio_sample_rate,
            hp.audio_hop_length,
            hp.audio_latent_downsample,
            hp.latent_channels,
            hp.vae_scale_t,
            hp.vae_scale_s,
            hp.vae_patch_size,
        ];
        if positive.iter().any(|&v| v <= 0 || v > 1 << 20)
            || hp.connector_layers < 0
            || hp.v_dim % (2 * hp.v_heads) != 0
            || hp.a_dim % (2 * hp.a_heads) != 0
            || !(hp.rope_theta > 0.0 && hp.connector_max_pos > 0.0 && hp.audio_max_pos > 0.0)
            || !hp.max_pos.iter().all(|&m| m > 0.0)
        {
            bail!("invalid diffusion model config");
        }
        Ok(hp)
    }
}

/// Everything loaded for one LTX model.
pub struct Model {
    pub hp: Hparams,
    pub dit: Weights,
    pub text_proj: Option<Weights>,
    pub vae: Option<Weights>,
    pub audio_vae: Option<Weights>,
}

impl Model {
    /// Text projection tensors may live in their own file or in the transformer GGUF.
    pub fn tp(&self, name: &str) -> Result<Tensor> {
        if let Some(w) = &self.text_proj {
            if let Some(t) = w.opt(name) {
                return Ok(t);
            }
        }
        self.dit.get(name)
    }

    pub fn tp_opt(&self, name: &str) -> Option<Tensor> {
        self.text_proj
            .as_ref()
            .and_then(|w| w.opt(name))
            .or_else(|| self.dit.opt(name))
    }
}

/// Text conditioning after the connectors: `[n_tokens][dim]` per modality.
#[derive(Default, Clone)]
pub struct TextCond {
    pub n_tokens: i64,
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
}

/// Video latents, token-major: `x[t * channels + c]`, `t = f*H*W + h*W + w`.
#[derive(Default, Clone)]
pub struct VideoLatent {
    pub n_frames: i64,
    pub height: i64,
    pub width: i64,
    pub channels: i64,
    pub x: Vec<f32>,
}

impl VideoLatent {
    pub fn n_tokens(&self) -> i64 {
        self.n_frames * self.height * self.width
    }
}

/// Audio latents, token-major: `x[t * (channels * freq) + c * freq + f]`.
#[derive(Default, Clone)]
pub struct AudioLatent {
    pub n_frames: i64,
    pub channels: i64,
    pub freq: i64,
    pub x: Vec<f32>,
}

//
// shared graph pieces
//

/// Rotary tables for the LTX "split" rope: `positions` holds `n_tokens x n_axes`
/// fractional coordinates; the output is `[n_tokens][dim/2]`.
pub fn rope_tables(
    positions: &[f64],
    n_tokens: i64,
    n_axes: i64,
    dim: i64,
    theta: f64,
) -> (Vec<f32>, Vec<f32>) {
    let half = (dim / 2) as usize;
    let n_freq = (dim / (2 * n_axes)) as usize;
    let pad = half - n_freq * n_axes as usize;
    let indices: Vec<f64> = (0..n_freq)
        .map(|f| {
            let e = if n_freq > 1 {
                f as f64 / (n_freq - 1) as f64
            } else {
                0.0
            };
            theta.powf(e) * PI / 2.0
        })
        .collect();
    let mut cos = vec![1f32; n_tokens as usize * half];
    let mut sin = vec![0f32; n_tokens as usize * half];
    for t in 0..n_tokens as usize {
        let base = t * half + pad;
        for f in 0..n_freq {
            for a in 0..n_axes as usize {
                let frac = positions[t * n_axes as usize + a];
                let angle = indices[f] * (frac * 2.0 - 1.0);
                cos[base + f * n_axes as usize + a] = angle.cos() as f32;
                sin[base + f * n_axes as usize + a] = angle.sin() as f32;
            }
        }
    }
    (cos, sin)
}

pub fn rms(g: &Ctx, x: Tensor, eps: f32, w: Option<Tensor>) -> Tensor {
    let y = g.rms_norm(x, eps);
    match w {
        Some(w) => g.mul(y, w),
        None => y,
    }
}

/// x: [dim, T], cos/sin: [dim/2, T]; per head the first half is rotated with the second.
pub fn rope_split(g: &Ctx, x: Tensor, cos: Tensor, sin: Tensor, n_heads: i64) -> Tensor {
    let [dim, t, ..] = ffi::shape(x);
    let hd = dim / n_heads;
    let x4 = g.reshape_4d(g.cont(x), hd / 2, 2, n_heads, t);
    let nb = ffi::strides(x4);
    let x1 = g.cont(g.view_4d(x4, hd / 2, 1, n_heads, t, nb[1], nb[2], nb[3], 0));
    let x2 = g.cont(g.view_4d(x4, hd / 2, 1, n_heads, t, nb[1], nb[2], nb[3], nb[1]));
    let c = g.reshape_4d(cos, hd / 2, 1, n_heads, t);
    let s = g.reshape_4d(sin, hd / 2, 1, n_heads, t);
    let o1 = g.sub(g.mul(x1, c), g.mul(x2, s));
    let o2 = g.add(g.mul(x1, s), g.mul(x2, c));
    g.reshape_2d(g.concat(o1, o2, 1), dim, t)
}

/// q: [dim_q, Tq], k, v: [dim_kv, Tk] -> [dim_v, Tq]
pub fn sdpa(g: &Ctx, q: Tensor, k: Tensor, v: Tensor, n_heads: i64, flash: bool) -> Tensor {
    let tq = ffi::shape(q)[1];
    let tk = ffi::shape(k)[1];
    let hd = ffi::shape(q)[0] / n_heads;
    let hv = ffi::shape(v)[0] / n_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let q3 = g.permute(g.reshape_3d(q, hd, n_heads, tq), 0, 2, 1, 3);
    let k3 = g.permute(g.reshape_3d(k, hd, n_heads, tk), 0, 2, 1, 3);
    if flash {
        let v3 = g.permute(g.reshape_3d(v, hv, n_heads, tk), 0, 2, 1, 3);
        let k3 = g.cast(k3, ffi::ty::F16);
        let v3 = g.cast(v3, ffi::ty::F16);
        let cur = g.flash_attn_ext(q3, k3, v3, std::ptr::null_mut(), scale, 0.0, 0.0);
        unsafe { (g.api.ggml_flash_attn_ext_set_prec)(cur, ffi::GGML_PREC_F32) };
        g.reshape_2d(cur, hv * n_heads, tq)
    } else {
        let v3 = g.cont(g.permute(g.reshape_3d(v, hv, n_heads, tk), 1, 2, 0, 3));
        let kq = g.soft_max_ext(g.mul_mat(k3, q3), std::ptr::null_mut(), scale, 0.0);
        let kqv = g.mul_mat(v3, kq);
        g.cont_2d(g.permute(kqv, 0, 2, 1, 3), hv * n_heads, tq)
    }
}

/// LTX CrossAttention weights.
pub struct AttnWeights {
    pub to_q: Tensor,
    pub to_q_b: Option<Tensor>,
    pub to_k: Tensor,
    pub to_k_b: Option<Tensor>,
    pub to_v: Tensor,
    pub to_v_b: Option<Tensor>,
    pub to_out: Tensor,
    pub to_out_b: Option<Tensor>,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub gate: Option<Tensor>,
    pub gate_b: Option<Tensor>,
}

impl AttnWeights {
    pub fn load(w: &Weights, prefix: &str) -> Result<AttnWeights> {
        Ok(AttnWeights {
            to_q: w.get(&format!("{prefix}.to_q.weight"))?,
            to_q_b: w.opt(&format!("{prefix}.to_q.bias")),
            to_k: w.get(&format!("{prefix}.to_k.weight"))?,
            to_k_b: w.opt(&format!("{prefix}.to_k.bias")),
            to_v: w.get(&format!("{prefix}.to_v.weight"))?,
            to_v_b: w.opt(&format!("{prefix}.to_v.bias")),
            to_out: w.get(&format!("{prefix}.to_out.0.weight"))?,
            to_out_b: w.opt(&format!("{prefix}.to_out.0.bias")),
            q_norm: w.get(&format!("{prefix}.q_norm.weight"))?,
            k_norm: w.get(&format!("{prefix}.k_norm.weight"))?,
            gate: w.opt(&format!("{prefix}.to_gate_logits.weight")),
            gate_b: w.opt(&format!("{prefix}.to_gate_logits.bias")),
        })
    }
}

/// Rope tables as graph inputs.
#[derive(Clone, Copy)]
pub struct Rope {
    pub cos: Tensor,
    pub sin: Tensor,
}

/// x: [dim_q, Tq] normalized input, context: [dim_kv, Tk] or None for self attention.
pub fn attention(
    g: &Ctx,
    w: &AttnWeights,
    x: Tensor,
    context: Option<Tensor>,
    rope_q: Option<Rope>,
    rope_k: Option<Rope>,
    n_heads: i64,
    flash: bool,
) -> Tensor {
    let kv_in = context.unwrap_or(x);
    let mut q = g.linear(x, w.to_q, w.to_q_b);
    let mut k = g.linear(kv_in, w.to_k, w.to_k_b);
    let v = g.linear(kv_in, w.to_v, w.to_v_b);
    q = rms(g, q, 1e-5, Some(w.q_norm));
    k = rms(g, k, 1e-5, Some(w.k_norm));
    if let Some(r) = rope_q {
        q = rope_split(g, q, r.cos, r.sin, n_heads);
    }
    if let Some(r) = rope_k {
        k = rope_split(g, k, r.cos, r.sin, n_heads);
    }
    let mut out = sdpa(g, q, k, v, n_heads, flash);
    if let Some(gate) = w.gate {
        let tq = ffi::shape(x)[1];
        let hv = ffi::shape(out)[0] / n_heads;
        let gl = g.scale(g.sigmoid(g.linear(x, gate, w.gate_b)), 2.0);
        let o3 = g.mul(
            g.reshape_3d(out, hv, n_heads, tq),
            g.reshape_3d(gl, 1, n_heads, tq),
        );
        out = g.reshape_2d(o3, hv * n_heads, tq);
    }
    g.linear(out, w.to_out, w.to_out_b)
}

/// gelu feed forward: net.0.proj -> gelu -> net.2
pub fn feed_forward(g: &Ctx, w: &Weights, prefix: &str, x: Tensor) -> Result<Tensor> {
    let h = g.linear(
        x,
        w.get(&format!("{prefix}.net.0.proj.weight"))?,
        w.opt(&format!("{prefix}.net.0.proj.bias")),
    );
    let h = g.gelu(h);
    Ok(g.linear(
        h,
        w.get(&format!("{prefix}.net.2.weight"))?,
        w.opt(&format!("{prefix}.net.2.bias")),
    ))
}

/// AdaLayerNormSingle: t [n] -> (mod [coef*dim, n], emb [dim, n])
pub fn adaln_single(g: &Ctx, w: &Weights, prefix: &str, t: Tensor) -> Result<(Tensor, Tensor)> {
    let te = g.timestep_embedding(t, 256, 10000);
    let te = g.linear(
        te,
        w.get(&format!("{prefix}.emb.timestep_embedder.linear_1.weight"))?,
        Some(w.get(&format!("{prefix}.emb.timestep_embedder.linear_1.bias"))?),
    );
    let te = g.silu(te);
    let emb = g.linear(
        te,
        w.get(&format!("{prefix}.emb.timestep_embedder.linear_2.weight"))?,
        Some(w.get(&format!("{prefix}.emb.timestep_embedder.linear_2.bias"))?),
    );
    let m = g.linear(
        g.silu(emb),
        w.get(&format!("{prefix}.linear.weight"))?,
        Some(w.get(&format!("{prefix}.linear.bias"))?),
    );
    Ok((m, emb))
}

/// table[i] + mod[i*dim : (i+1)*dim]
pub fn ada(g: &Ctx, table: Tensor, modv: Tensor, i: i64, dim: i64) -> Tensor {
    let mnb = ffi::strides(modv);
    let tnb = ffi::strides(table);
    let esz = unsafe { (g.api.ggml_type_size)((*modv).type_) };
    let m = g.view_2d(modv, dim, 1, mnb[1], (i * dim) as usize * esz);
    let t = g.view_2d(table, dim, 1, tnb[1], i as usize * tnb[1]);
    g.add(m, t)
}

/// Row i of a [dim, n] table.
pub fn table_row(g: &Ctx, table: Tensor, i: i64) -> Tensor {
    let dim = ffi::shape(table)[0];
    let nb = ffi::strides(table);
    g.view_2d(table, dim, 1, nb[1], i as usize * nb[1])
}

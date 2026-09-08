//! The Cosmos3 omni transformer (diffusers `Cosmos3OmniTransformer`): two token
//! streams per layer that share nothing but attention. The understanding stream
//! (prompt tokens, causal decoder) never sees the generation stream (latents),
//! so it runs once per prompt ([`und_forward`]) and its keys/values feed every
//! denoising step ([`gen_forward`]).

use anyhow::{bail, Result};

use super::Hparams;
use crate::mediagen::backend::{Backend, Ctx, Graph};
use crate::mediagen::ffi::{self, Tensor};
use crate::mediagen::ltx::rms;
use crate::mediagen::weights::Weights;

/// The causal mask is `n x n`; the checkpoints are trained on far shorter prompts.
pub const MAX_PROMPT_TOKENS: i64 = 4096;

/// Per-layer keys (normed, roped) and values of the understanding stream, `[head_dim * n_kv_heads, n_tokens]`.
pub struct UndCache {
    pub n_tokens: i64,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

/// Interleaved 3-D mRoPE cos/sin tables `[n_tokens][head_dim/2]` for `(t, h, w)` positions.
pub fn mrope_tables(hp: &Hparams, positions: &[[f64; 3]]) -> (Vec<f32>, Vec<f32>) {
    let half = (hp.head_dim / 2) as usize;
    let sect = hp.mrope_section;
    let mut cos = Vec::with_capacity(positions.len() * half);
    let mut sin = Vec::with_capacity(positions.len() * half);
    for p in positions {
        for j in 0..half {
            let inv = 1.0 / hp.rope_theta.powf(2.0 * j as f64 / hp.head_dim as f64);
            // [T H W T H W ... T T]: H takes 1, 4, 7, ... and W 2, 5, 8, ... below 3*section
            let axis = if j < 3 * sect[1] as usize && j % 3 == 1 {
                1
            } else if j < 3 * sect[2] as usize && j % 3 == 2 {
                2
            } else {
                0
            };
            let angle = p[axis] * inv;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}

/// Positions of the prompt tokens: the same index on all three axes.
pub fn text_positions(n: i64) -> Vec<[f64; 3]> {
    (0..n).map(|i| [i as f64; 3]).collect()
}

/// Vision token positions (`get_3d_mrope_ids_vae_tokens`): a `[t][h][w]` grid `temporal_margin` after the text.
pub fn vision_positions(hp: &Hparams, n_text: i64, grid: [i64; 3], fps: f32) -> Vec<[f64; 3]> {
    let offset = (n_text + hp.temporal_margin) as f64;
    let [gt, gh, gw] = grid;
    let modulated = hp.enable_fps_modulation && gt > 1;
    let mut out = Vec::with_capacity((gt * gh * gw) as usize);
    for t in 0..gt {
        let tp = if modulated {
            let tps = fps as f64 / hp.vae_scale_t as f64;
            let base_tps = hp.base_fps as f64 / hp.vae_scale_t as f64;
            t as f64 / tps * base_tps + offset
        } else {
            t as f64 + offset
        };
        for h in 0..gh {
            for w in 0..gw {
                out.push([tp, h as f64, w as f64]);
            }
        }
    }
    out
}

/// Sound token positions: one per latent frame at `sound_fps`, same temporal offset as vision.
pub fn sound_positions(hp: &Hparams, n_text: i64, n: i64, sound_fps: f32) -> Vec<[f64; 3]> {
    let offset = (n_text + hp.temporal_margin) as f64;
    (0..n)
        .map(|f| {
            let t = if hp.enable_fps_modulation && n > 1 {
                f as f64 / sound_fps as f64 * hp.base_fps as f64 + offset
            } else {
                f as f64 + offset
            };
            [t, 0.0, 0.0]
        })
        .collect()
}

/// Action token positions: transition `f` sits at video frame `f + 1`.
pub fn action_positions(hp: &Hparams, n_text: i64, n: i64, fps: f32) -> Vec<[f64; 3]> {
    let offset = (n_text + hp.temporal_margin) as f64;
    let base_tps = hp.base_fps as f64 / hp.vae_scale_t as f64;
    (0..n)
        .map(|f| {
            let t = if hp.enable_fps_modulation && n > 1 {
                (f + 1) as f64 / fps as f64 * base_tps + offset
            } else {
                (f + 1) as f64 + offset
            };
            [t, 0.0, 0.0]
        })
        .collect()
}

/// A (cos, sin) pair of mRoPE tables.
pub type Tables = (Vec<f32>, Vec<f32>);

/// diffusers' `Timesteps(256, flip_sin_to_cos=True)`: 128 cosines then 128 sines.
pub fn timestep_embedding(t: f32) -> Vec<f32> {
    let half = 128;
    let mut e = vec![0f32; 256];
    for j in 0..half {
        let freq = (-(10000f32).ln() * j as f32 / half as f32).exp();
        let arg = t * freq;
        e[j] = arg.cos();
        e[half + j] = arg.sin();
    }
    e
}

/// Rotate-half RoPE on `x: [n_heads * hd, T]` with a `[hd/2, T]` table shared by all heads.
fn rope(g: &Ctx, x: Tensor, cos: Tensor, sin: Tensor, n_heads: i64) -> Tensor {
    let [dim, t, ..] = ffi::shape(x);
    let hd = dim / n_heads;
    let x4 = g.reshape_4d(g.cont(x), hd / 2, 2, n_heads, t);
    let nb = ffi::strides(x4);
    let x1 = g.cont(g.view_4d(x4, hd / 2, 1, n_heads, t, nb[1], nb[2], nb[3], 0));
    let x2 = g.cont(g.view_4d(x4, hd / 2, 1, n_heads, t, nb[1], nb[2], nb[3], nb[1]));
    let c = g.reshape_4d(cos, hd / 2, 1, 1, t);
    let s = g.reshape_4d(sin, hd / 2, 1, 1, t);
    let o1 = g.sub(g.mul(x1, c), g.mul(x2, s));
    let o2 = g.add(g.mul(x1, s), g.mul(x2, c));
    g.reshape_2d(g.concat(o1, o2, 1), dim, t)
}

/// RMSNorm over each head of `x: [n_heads * head_dim, T]` with a `[head_dim]` weight.
fn head_rms(g: &Ctx, x: Tensor, hd: i64, eps: f32, w: Option<Tensor>) -> Tensor {
    let [dim, t, ..] = ffi::shape(x);
    let x3 = g.reshape_3d(g.cont(x), hd, dim / hd, t);
    let y = rms(g, x3, eps, w);
    g.reshape_2d(y, dim, t)
}

/// Grouped-query attention: q `[n_heads*hd, Tq]`, k/v `[n_kv*hd, Tk]` -> `[n_heads*hd, Tq]`.
fn sdpa_gqa(
    g: &Ctx,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    n_heads: i64,
    n_kv: i64,
    mask: Option<Tensor>,
    flash: bool,
) -> Tensor {
    let tq = ffi::shape(q)[1];
    let tk = ffi::shape(k)[1];
    let hd = ffi::shape(q)[0] / n_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let q3 = g.permute(g.reshape_3d(g.cont(q), hd, n_heads, tq), 0, 2, 1, 3);
    let k3 = g.permute(g.reshape_3d(g.cont(k), hd, n_kv, tk), 0, 2, 1, 3);
    if flash && mask.is_none() {
        let v3 = g.permute(g.reshape_3d(g.cont(v), hd, n_kv, tk), 0, 2, 1, 3);
        let k3 = g.cast(k3, ffi::ty::F16);
        let v3 = g.cast(v3, ffi::ty::F16);
        let cur = g.flash_attn_ext(q3, k3, v3, std::ptr::null_mut(), scale, 0.0, 0.0);
        unsafe { (g.api.ggml_flash_attn_ext_set_prec)(cur, ffi::GGML_PREC_F32) };
        g.reshape_2d(cur, hd * n_heads, tq)
    } else {
        let v3 = g.cont(g.permute(g.reshape_3d(g.cont(v), hd, n_kv, tk), 1, 2, 0, 3));
        let kq = g.mul_mat(k3, q3);
        let kq = g.soft_max_ext(kq, mask.unwrap_or(std::ptr::null_mut()), scale, 0.0);
        let kqv = g.mul_mat(v3, kq);
        g.cont_2d(g.permute(kqv, 0, 2, 1, 3), hd * n_heads, tq)
    }
}

/// `Cosmos3VLTextMLP`: `down(silu(gate) * up)` or `down(relu(up)^2)`.
fn mlp(g: &Ctx, w: &Weights, prefix: &str, x: Tensor, relu2: bool) -> Result<Tensor> {
    let up = g.linear(x, w.get(&format!("{prefix}.up_proj.weight"))?, None);
    let h = if relu2 {
        g.sqr(g.clamp(up, 0.0, f32::INFINITY))
    } else {
        let gate = g.linear(x, w.get(&format!("{prefix}.gate_proj.weight"))?, None);
        g.mul(g.silu(gate), up)
    };
    Ok(g.linear(h, w.get(&format!("{prefix}.down_proj.weight"))?, None))
}

/// The understanding stream over the prompt tokens.
pub fn und_forward(hp: &Hparams, w: &Weights, be: &Backend, ids: &[u32]) -> Result<UndCache> {
    let n = ids.len() as i64;
    if n == 0 || n > MAX_PROMPT_TOKENS {
        bail!("the prompt must be 1 to {MAX_PROMPT_TOKENS} tokens, got {n}");
    }
    let mut gr = Graph::new(be.api, 4 * 1024 + hp.n_layers as usize * 96)?;
    let ids_i32: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
    let ids_t = gr.input_i32(&[n], &ids_i32, "ids");
    let (cos, sin) = mrope_tables(hp, &text_positions(n));
    let half = hp.head_dim / 2;
    let cos_t = gr.input_f32(&[half, n], &cos, "cos");
    let sin_t = gr.input_f32(&[half, n], &sin, "sin");
    // causal mask [Tk, Tq]: -inf where the key follows the query
    let mut mask = vec![0f32; (n * n) as usize];
    for q in 0..n as usize {
        for k in q + 1..n as usize {
            mask[q * n as usize + k] = f32::NEG_INFINITY;
        }
    }
    let mask_t = gr.input_f32(&[n, n], &mask, "mask");

    let g = &gr.ctx;
    let eps = hp.rms_eps;
    let hd = hp.head_dim;
    let mut x = g.get_rows(w.get("embed_tokens.weight")?, ids_t);
    let mut ks = Vec::with_capacity(hp.n_layers as usize);
    let mut vs = Vec::with_capacity(hp.n_layers as usize);
    for l in 0..hp.n_layers {
        let p = format!("layers.{l}");
        let h = rms(
            g,
            x,
            eps,
            Some(w.get(&format!("{p}.input_layernorm.weight"))?),
        );
        let mut q = g.linear(h, w.get(&format!("{p}.self_attn.to_q.weight"))?, None);
        let mut k = g.linear(h, w.get(&format!("{p}.self_attn.to_k.weight"))?, None);
        let v = g.linear(h, w.get(&format!("{p}.self_attn.to_v.weight"))?, None);
        if hp.qk_norm_for_text {
            q = head_rms(
                g,
                q,
                hd,
                eps,
                Some(w.get(&format!("{p}.self_attn.norm_q.weight"))?),
            );
            k = head_rms(
                g,
                k,
                hd,
                eps,
                Some(w.get(&format!("{p}.self_attn.norm_k.weight"))?),
            );
        }
        let mut k_gen = match w.opt(&format!("{p}.self_attn.k_norm_und_for_gen.weight")) {
            Some(nw) => head_rms(g, k, hd, eps, Some(nw)),
            None => k,
        };
        q = rope(g, q, cos_t, sin_t, hp.n_heads);
        k = rope(g, k, cos_t, sin_t, hp.n_kv_heads);
        k_gen = rope(g, k_gen, cos_t, sin_t, hp.n_kv_heads);
        let attn = sdpa_gqa(g, q, k, v, hp.n_heads, hp.n_kv_heads, Some(mask_t), false);
        x = g.add(
            x,
            g.linear(attn, w.get(&format!("{p}.self_attn.to_out.weight"))?, None),
        );
        let h2 = rms(
            g,
            x,
            eps,
            Some(w.get(&format!("{p}.post_attention_layernorm.weight"))?),
        );
        x = g.add(x, mlp(g, w, &format!("{p}.mlp"), h2, hp.relu2)?);
        let k_out = g.cont(k_gen);
        let v_out = g.cont(v);
        gr.mark_output(k_out);
        gr.mark_output(v_out);
        ks.push(k_out);
        vs.push(v_out);
    }
    gr.compute(be)?;
    Ok(UndCache {
        n_tokens: n,
        k: ks.iter().map(|&t| gr.output_f32(t)).collect(),
        v: vs.iter().map(|&t| gr.output_f32(t)).collect(),
    })
}

/// One modality's tokens of the generation stream for a denoising step.
pub struct Segment<'a> {
    /// `[n][dim]`: patchified latents (vision), sound latents, or action vectors.
    pub tokens: &'a [f32],
    pub n: i64,
    /// mRoPE tables `[n][head_dim/2]`.
    pub cos: &'a [f32],
    pub sin: &'a [f32],
    /// `1` for noisy tokens, `0` for conditioning tokens; `None` when all are noisy.
    pub noisy: Option<&'a [f32]>,
}

/// One denoising evaluation of the generation stream.
pub struct GenInputs<'a> {
    pub vision: Segment<'a>,
    pub sound: Option<Segment<'a>>,
    /// Action tokens and the embodiment domain selecting the projection weights.
    pub action: Option<(Segment<'a>, i64)>,
    /// Already scaled by `timestep_scale` (0.995 for t = 995).
    pub timestep: f32,
    pub und: &'a UndCache,
    pub flash: bool,
}

/// Velocity predictions per modality, `[n][dim]` each (empty when absent).
pub struct GenOutputs {
    pub vision: Vec<f32>,
    pub sound: Vec<f32>,
    pub action: Vec<f32>,
}

/// `DomainAwareLinear`: one domain's weight `[in, out]` and bias `[out]` from the embedding tables.
fn domain_linear(
    g: &Ctx,
    w: &Weights,
    prefix: &str,
    domain: Tensor,
    n_in: i64,
    n_out: i64,
) -> Result<(Tensor, Tensor)> {
    let row = g.get_rows(w.get(&format!("{prefix}.fc.weight"))?, domain); // [in*out, 1]
                                                                          // the row is `[in][out]` (out fastest); mul_mat wants `[in, out]` in ggml order
    let wt = g.cont(g.transpose(g.reshape_2d(row, n_out, n_in)));
    let b = g.get_rows(w.get(&format!("{prefix}.bias.weight"))?, domain); // [out, 1]
    Ok((wt, g.reshape_2d(b, n_out, 1)))
}

/// The velocity predictions for every generation token.
pub fn gen_forward(hp: &Hparams, w: &Weights, be: &Backend, inp: &GenInputs) -> Result<GenOutputs> {
    let nt = inp.und.n_tokens;
    let nv = inp.vision.n;
    let ns = inp.sound.as_ref().map_or(0, |s| s.n);
    let na = inp.action.as_ref().map_or(0, |(s, _)| s.n);
    let n = nv + ns + na;
    let mut gr = Graph::new(be.api, 4 * 1024 + hp.n_layers as usize * 128)?;
    let half = hp.head_dim / 2;
    let v_tok = gr.input_f32(&[hp.patch_dim, nv], inp.vision.tokens, "vision");
    let s_tok = inp
        .sound
        .as_ref()
        .map(|s| gr.input_f32(&[hp.sound_dim, s.n], s.tokens, "sound"));
    let a_tok = inp
        .action
        .as_ref()
        .map(|(s, _)| gr.input_f32(&[hp.action_dim, s.n], s.tokens, "action"));
    let domain = inp
        .action
        .as_ref()
        .map(|(_, d)| gr.input_i32(&[1], &[*d as i32], "domain"));
    // rope tables and the noisy mask over the concatenated tokens
    let mut cos = inp.vision.cos.to_vec();
    let mut sin = inp.vision.sin.to_vec();
    let mut noisy: Vec<f32> = match inp.vision.noisy {
        Some(m) => m.to_vec(),
        None => vec![1.0; nv as usize],
    };
    let mut any_cond = inp.vision.noisy.is_some();
    for seg in inp.sound.iter().chain(inp.action.iter().map(|(s, _)| s)) {
        cos.extend_from_slice(seg.cos);
        sin.extend_from_slice(seg.sin);
        match seg.noisy {
            Some(m) => {
                noisy.extend_from_slice(m);
                any_cond = true;
            }
            None => noisy.extend(std::iter::repeat_n(1.0, seg.n as usize)),
        }
    }
    let cos_t = gr.input_f32(&[half, n], &cos, "cos");
    let sin_t = gr.input_f32(&[half, n], &sin, "sin");
    let noisy_t = any_cond.then(|| gr.input_f32(&[1, n], &noisy, "noisy"));
    let temb_in = gr.input_f32(&[256], &timestep_embedding(inp.timestep), "temb");
    let kv_dim = hp.head_dim * hp.n_kv_heads;
    let und_k: Vec<Tensor> = (0..hp.n_layers as usize)
        .map(|l| gr.input_f32(&[kv_dim, nt], &inp.und.k[l], &format!("und_k{l}")))
        .collect();
    let und_v: Vec<Tensor> = (0..hp.n_layers as usize)
        .map(|l| gr.input_f32(&[kv_dim, nt], &inp.und.v[l], &format!("und_v{l}")))
        .collect();

    let g = &gr.ctx;
    let eps = hp.rms_eps;
    let hd = hp.head_dim;
    // embed every modality, then pack them in order: vision, sound, action
    let mut x = g.linear(
        v_tok,
        w.get("proj_in.weight")?,
        Some(w.get("proj_in.bias")?),
    );
    if let Some(st) = s_tok {
        let e = g.linear(
            st,
            w.get("audio_proj_in.weight")?,
            Some(w.get("audio_proj_in.bias")?),
        );
        let e = g.add(
            e,
            g.reshape_2d(w.get("audio_modality_embed")?, hp.hidden, 1),
        );
        x = g.concat(x, e, 1);
    }
    let mut action_out_w = None;
    if let (Some(at), Some(d)) = (a_tok, domain) {
        let (wi, bi) = domain_linear(g, w, "action_proj_in", d, hp.action_dim, hp.hidden)?;
        let e = g.add(g.mul_mat(wi, at), bi);
        let e = g.add(
            e,
            g.reshape_2d(w.get("action_modality_embed")?, hp.hidden, 1),
        );
        x = g.concat(x, e, 1);
        action_out_w = Some(domain_linear(
            g,
            w,
            "action_proj_out",
            d,
            hp.hidden,
            hp.action_dim,
        )?);
    }
    // timestep embedding, added to the noisy tokens
    let te = g.linear(
        g.reshape_2d(temb_in, 256, 1),
        w.get("time_embedder.linear_1.weight")?,
        Some(w.get("time_embedder.linear_1.bias")?),
    );
    let te = g.linear(
        g.silu(te),
        w.get("time_embedder.linear_2.weight")?,
        Some(w.get("time_embedder.linear_2.bias")?),
    );
    x = match noisy_t {
        None => g.add(x, te),
        Some(m) => g.add(x, g.mul(g.repeat(te, x), m)),
    };

    for l in 0..hp.n_layers {
        let p = format!("layers.{l}");
        let h = rms(
            g,
            x,
            eps,
            Some(w.get(&format!("{p}.input_layernorm_moe_gen.weight"))?),
        );
        let q = g.linear(h, w.get(&format!("{p}.self_attn.add_q_proj.weight"))?, None);
        let k = g.linear(h, w.get(&format!("{p}.self_attn.add_k_proj.weight"))?, None);
        let v = g.linear(h, w.get(&format!("{p}.self_attn.add_v_proj.weight"))?, None);
        let q = head_rms(
            g,
            q,
            hd,
            eps,
            Some(w.get(&format!("{p}.self_attn.norm_added_q.weight"))?),
        );
        let k = head_rms(
            g,
            k,
            hd,
            eps,
            Some(w.get(&format!("{p}.self_attn.norm_added_k.weight"))?),
        );
        let q = rope(g, q, cos_t, sin_t, hp.n_heads);
        let k = rope(g, k, cos_t, sin_t, hp.n_kv_heads);
        let kk = g.concat(und_k[l as usize], g.cont(k), 1);
        let vv = g.concat(und_v[l as usize], g.cont(v), 1);
        let attn = sdpa_gqa(g, q, kk, vv, hp.n_heads, hp.n_kv_heads, None, inp.flash);
        x = g.add(
            x,
            g.linear(
                attn,
                w.get(&format!("{p}.self_attn.to_add_out.weight"))?,
                None,
            ),
        );
        let h2 = rms(
            g,
            x,
            eps,
            Some(w.get(&format!("{p}.post_attention_layernorm_moe_gen.weight"))?),
        );
        x = g.add(x, mlp(g, w, &format!("{p}.mlp_moe_gen"), h2, hp.relu2)?);
    }
    let out = g.cont(rms(g, x, eps, Some(w.get("norm_moe_gen.weight")?)));
    let nb = ffi::strides(out);
    let part = |off: i64, len: i64| g.view_2d(out, hp.hidden, len, nb[1], (off as usize) * nb[1]);
    let v_out = g.cont(g.linear(
        part(0, nv),
        w.get("proj_out.weight")?,
        Some(w.get("proj_out.bias")?),
    ));
    gr.mark_output(v_out);
    let s_out = (ns > 0)
        .then(|| -> Result<Tensor> {
            let t = g.cont(g.linear(
                part(nv, ns),
                w.get("audio_proj_out.weight")?,
                Some(w.get("audio_proj_out.bias")?),
            ));
            gr.mark_output(t);
            Ok(t)
        })
        .transpose()?;
    let a_out = match action_out_w {
        Some((wo, bo)) if na > 0 => {
            let t = g.cont(g.add(g.mul_mat(wo, part(nv + ns, na)), bo));
            gr.mark_output(t);
            Some(t)
        }
        _ => None,
    };
    gr.compute(be)?;
    Ok(GenOutputs {
        vision: gr.output_f32(v_out),
        sound: s_out.map(|t| gr.output_f32(t)).unwrap_or_default(),
        action: a_out.map(|t| gr.output_f32(t)).unwrap_or_default(),
    })
}

/// Latents `[C][T][H][W]` -> tokens `[t][h][w][(p*P + q)*C + c]` (`_patchify_and_pack_latents`).
pub fn patchify(hp: &Hparams, lat: &[f32], t: i64, h: i64, w: i64) -> Vec<f32> {
    let (c, p) = (hp.latent_channels, hp.patch);
    // odd latent sizes are zero-padded to whole patches
    let (hp_, wp) = ((h + p - 1) / p, (w + p - 1) / p);
    let mut out = vec![0f32; (t * hp_ * wp * hp.patch_dim) as usize];
    let idx = |ci: i64, ti: i64, hi: i64, wi: i64| (((ci * t + ti) * h + hi) * w + wi) as usize;
    let mut n = 0usize;
    for ti in 0..t {
        for hh in 0..hp_ {
            for ww in 0..wp {
                let base = n * hp.patch_dim as usize;
                for pp in 0..p {
                    for q in 0..p {
                        let (hi, wi) = (hh * p + pp, ww * p + q);
                        if hi >= h || wi >= w {
                            continue;
                        }
                        for ci in 0..c {
                            out[base + ((pp * p + q) * c + ci) as usize] = lat[idx(ci, ti, hi, wi)];
                        }
                    }
                }
                n += 1;
            }
        }
    }
    out
}

/// The inverse of [`patchify`].
pub fn unpatchify(hp: &Hparams, tokens: &[f32], t: i64, h: i64, w: i64) -> Vec<f32> {
    let (c, p) = (hp.latent_channels, hp.patch);
    let (hp_, wp) = ((h + p - 1) / p, (w + p - 1) / p);
    let mut out = vec![0f32; (c * t * h * w) as usize];
    let idx = |ci: i64, ti: i64, hi: i64, wi: i64| (((ci * t + ti) * h + hi) * w + wi) as usize;
    let mut n = 0usize;
    for ti in 0..t {
        for hh in 0..hp_ {
            for ww in 0..wp {
                let base = n * hp.patch_dim as usize;
                for pp in 0..p {
                    for q in 0..p {
                        let (hi, wi) = (hh * p + pp, ww * p + q);
                        if hi >= h || wi >= w {
                            continue;
                        }
                        for ci in 0..c {
                            out[idx(ci, ti, hi, wi)] =
                                tokens[base + ((pp * p + q) * c + ci) as usize];
                        }
                    }
                }
                n += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patchify_roundtrip() {
        let hp = Hparams::test_default();
        let (t, h, w) = (2, 4, 6);
        let lat: Vec<f32> = (0..(hp.latent_channels * t * h * w))
            .map(|i| i as f32)
            .collect();
        let tok = patchify(&hp, &lat, t, h, w);
        assert_eq!(tok.len(), (t * (h / 2) * (w / 2) * hp.patch_dim) as usize);
        // first token: (t=0,h=0..2,w=0..2), entries ordered p, q, c
        assert_eq!(tok[0], lat[0]); // c0, p0, q0
        assert_eq!(tok[hp.latent_channels as usize], lat[1]); // p0, q1 -> w=1
        assert_eq!(tok[2 * hp.latent_channels as usize], lat[w as usize]); // p1 -> h=1
        assert_eq!(unpatchify(&hp, &tok, t, h, w), lat);
        // odd sizes pad to whole patches and crop back
        let (h, w) = (3, 5);
        let lat: Vec<f32> = (0..(hp.latent_channels * t * h * w))
            .map(|i| i as f32)
            .collect();
        let tok = patchify(&hp, &lat, t, h, w);
        assert_eq!(tok.len(), (t * 2 * 3 * hp.patch_dim) as usize);
        assert_eq!(unpatchify(&hp, &tok, t, h, w), lat);
    }

    #[test]
    fn mrope_axes() {
        let hp = Hparams::test_default();
        // only the H axis differs: frequencies 1,4,7,..,58 change (sin: cos of a
        // tiny angle rounds to 1 in f32)
        let (_, s0) = mrope_tables(&hp, &[[3.0, 0.0, 0.0]]);
        let (_, s1) = mrope_tables(&hp, &[[3.0, 5.0, 0.0]]);
        for j in 0..64 {
            let h_axis = j < 60 && j % 3 == 1;
            assert_eq!(s0[j] != s1[j], h_axis, "j={j}");
        }
        let (_, s2) = mrope_tables(&hp, &[[3.0, 0.0, 5.0]]);
        for j in 0..64 {
            let w_axis = j < 60 && j % 3 == 2;
            assert_eq!(s0[j] != s2[j], w_axis, "j={j}");
        }
    }

    #[test]
    fn timestep_embedding_layout() {
        let e = timestep_embedding(0.995);
        assert_eq!(e.len(), 256);
        assert!((e[0] - 0.995f32.cos()).abs() < 1e-6);
        assert!((e[128] - 0.995f32.sin()).abs() < 1e-6);
        // last frequency 10000^(-127/128)
        let f = (-(10000f32).ln() * 127.0 / 128.0).exp();
        assert!((e[127] - (0.995 * f).cos()).abs() < 1e-6);
    }

    #[test]
    fn vision_positions_layout() {
        let hp = Hparams::test_default();
        let pos = vision_positions(&hp, 10, [3, 2, 2], 24.0);
        assert_eq!(pos.len(), 12);
        assert_eq!(pos[0], [15010.0, 0.0, 0.0]);
        assert_eq!(pos[3], [15010.0, 1.0, 1.0]);
        assert_eq!(pos[4], [15011.0, 0.0, 0.0]);
        // 12 fps: latent frames are 2 base-fps latent frames apart
        let pos12 = vision_positions(&hp, 10, [3, 2, 2], 12.0);
        assert_eq!(pos12[4][0], 15012.0);
        // a single frame is not fps-modulated
        assert_eq!(vision_positions(&hp, 10, [1, 2, 2], 12.0)[0][0], 15010.0);
    }
}

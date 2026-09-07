//! The audio-video diffusion transformer: one denoising step, and the sigma schedules.

use anyhow::Result;

use super::super::backend::{Backend, Graph};
use super::super::ffi::Tensor;
use super::{
    ada, adaln_single, attention, feed_forward, rms, rope_tables, table_row, AttnWeights,
    AudioLatent, Hparams, Model, Rope, TextCond, VideoLatent,
};

/// Fractional (t, h, w) positions of the video tokens (middle of each latent cell).
fn video_positions(hp: &Hparams, lat: &VideoLatent, fps: f32) -> Vec<f64> {
    let (st, ss) = (hp.vae_scale_t as f64, hp.vae_scale_s as f64);
    let mut pos = Vec::with_capacity((lat.n_tokens() * 3) as usize);
    for f in 0..lat.n_frames {
        let (mut t0, mut t1) = (f as f64 * st, (f + 1) as f64 * st);
        if hp.causal_temporal_positioning {
            t0 = (t0 + 1.0 - st).max(0.0);
            t1 = (t1 + 1.0 - st).max(0.0);
        }
        let tp = 0.5 * (t0 + t1) / fps as f64 / hp.max_pos[0];
        for h in 0..lat.height {
            for w in 0..lat.width {
                pos.push(tp);
                pos.push(0.5 * (h as f64 * ss + (h + 1) as f64 * ss) / hp.max_pos[1]);
                pos.push(0.5 * (w as f64 * ss + (w + 1) as f64 * ss) / hp.max_pos[2]);
            }
        }
    }
    pos
}

fn audio_latent_time(hp: &Hparams, i: i64) -> f64 {
    let mel = (i * hp.audio_latent_downsample) as f64;
    let mel = (mel + 1.0 - hp.audio_latent_downsample as f64).max(0.0);
    mel * hp.audio_hop_length as f64 / hp.audio_sample_rate as f64
}

fn audio_positions(hp: &Hparams, n: i64, max_pos: f64) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (audio_latent_time(hp, i) + audio_latent_time(hp, i + 1)) / max_pos)
        .collect()
}

/// Temporal positions of the video tokens used by the audio-video cross attention.
fn video_time_positions(hp: &Hparams, lat: &VideoLatent, fps: f32, max_pos: f64) -> Vec<f64> {
    let st = hp.vae_scale_t as f64;
    let mut pos = Vec::with_capacity(lat.n_tokens() as usize);
    for f in 0..lat.n_frames {
        let (mut t0, mut t1) = (f as f64 * st, (f + 1) as f64 * st);
        if hp.causal_temporal_positioning {
            t0 = (t0 + 1.0 - st).max(0.0);
            t1 = (t1 + 1.0 - st).max(0.0);
        }
        let p = 0.5 * (t0 / fps as f64 + t1 / fps as f64) / max_pos;
        pos.extend(std::iter::repeat_n(p, (lat.height * lat.width) as usize));
    }
    pos
}

pub struct Inputs<'a> {
    pub video: &'a VideoLatent,
    pub audio: Option<&'a AudioLatent>,
    pub cond: &'a TextCond,
    pub sigma: f32,
    pub fps: f32,
    pub flash: bool,
}

fn make_rope(
    g: &mut Graph,
    pos: &[f64],
    n: i64,
    n_axes: i64,
    dim: i64,
    theta: f64,
    name: &str,
) -> Rope {
    let (c, s) = rope_tables(pos, n, n_axes, dim, theta);
    Rope {
        cos: g.input_f32(&[dim / 2, n], &c, &format!("{name}_cos")),
        sin: g.input_f32(&[dim / 2, n], &s, &format!("{name}_sin")),
    }
}

/// One forward pass: the predicted velocity of the video (and audio) tokens.
pub fn forward(model: &Model, be: &Backend, inp: &Inputs) -> Result<(Vec<f32>, Vec<f32>)> {
    let hp = &model.hp;
    let w = &model.dit;
    let has_audio = inp.audio.is_some_and(|a| a.n_frames > 0);
    let tv = inp.video.n_tokens();
    let ta = inp.audio.map_or(0, |a| a.n_frames);
    let tc = inp.cond.n_tokens;
    let (vd, ad) = (hp.v_dim, hp.a_dim);

    let mut g = Graph::new(be.api, 24 * 1024)?;
    let mut vx = g.input_f32(&[hp.in_channels, tv], &inp.video.x, "video_latent");
    let mut ax = inp.audio.filter(|_| has_audio).map(|a| {
        g.input_f32(
            &[hp.audio_channels * hp.audio_freq_bins, ta],
            &a.x,
            "audio_latent",
        )
    });
    let v_ctx = g.input_f32(&[vd, tc], &inp.cond.video, "v_context");
    let a_ctx = has_audio.then(|| g.input_f32(&[ad, tc], &inp.cond.audio, "a_context"));
    let t = g.input_f32(&[1], &[inp.sigma * hp.timestep_scale], "timestep");
    let t_av =
        has_audio.then(|| g.input_f32(&[1], &[inp.sigma * hp.av_ca_timestep_scale], "timestep_av"));

    let pos = video_positions(hp, inp.video, inp.fps);
    let v_pe = make_rope(&mut g, &pos, tv, 3, vd, hp.rope_theta, "v_pe");
    let mut av_pe = None;
    if has_audio {
        let max_pos = hp.max_pos[0].max(hp.audio_max_pos);
        let a_pe = make_rope(
            &mut g,
            &audio_positions(hp, ta, hp.audio_max_pos),
            ta,
            1,
            ad,
            hp.rope_theta,
            "a_pe",
        );
        let v_cross = make_rope(
            &mut g,
            &video_time_positions(hp, inp.video, inp.fps, max_pos),
            tv,
            1,
            ad,
            hp.rope_theta,
            "v_cross",
        );
        let a_cross = make_rope(
            &mut g,
            &audio_positions(hp, ta, max_pos),
            ta,
            1,
            ad,
            hp.rope_theta,
            "a_cross",
        );
        av_pe = Some((a_pe, v_cross, a_cross));
    }

    let ctx = &g.ctx;
    vx = ctx.linear(
        vx,
        w.get("patchify_proj.weight")?,
        Some(w.get("patchify_proj.bias")?),
    );
    if let Some(a) = ax {
        ax = Some(ctx.linear(
            a,
            w.get("audio_patchify_proj.weight")?,
            Some(w.get("audio_patchify_proj.bias")?),
        ));
    }

    let (v_mod, v_emb) = adaln_single(ctx, w, "adaln_single", t)?;
    let (vp_mod, _) = adaln_single(ctx, w, "prompt_adaln_single", t)?;
    let mut audio_mods = None;
    if has_audio {
        let t_av = t_av.unwrap();
        let (a_mod, a_emb) = adaln_single(ctx, w, "audio_adaln_single", t)?;
        let (ap_mod, _) = adaln_single(ctx, w, "audio_prompt_adaln_single", t)?;
        let (ca_a_ss, _) = adaln_single(ctx, w, "av_ca_audio_scale_shift_adaln_single", t)?;
        let (ca_v_ss, _) = adaln_single(ctx, w, "av_ca_video_scale_shift_adaln_single", t)?;
        let (ca_a2v, _) = adaln_single(ctx, w, "av_ca_a2v_gate_adaln_single", t_av)?;
        let (ca_v2a, _) = adaln_single(ctx, w, "av_ca_v2a_gate_adaln_single", t_av)?;
        audio_mods = Some((a_mod, a_emb, ap_mod, ca_a_ss, ca_v_ss, ca_a2v, ca_v2a));
    }

    for il in 0..hp.n_layers {
        let bp = format!("transformer_blocks.{il}");
        let tbl = w.get(&format!("{bp}.scale_shift_table"))?;
        let ptbl = w.get(&format!("{bp}.prompt_scale_shift_table"))?;
        let m = |i| ada(ctx, tbl, v_mod, i, vd);
        let (shift_msa, scale_msa, gate_msa) = (m(0), m(1), m(2));
        let (shift_mlp, scale_mlp, gate_mlp) = (m(3), m(4), m(5));
        let (shift_q, scale_q, gate_ca) = (m(6), m(7), m(8));
        let shift_kv = ada(ctx, ptbl, vp_mod, 0, vd);
        let scale_kv = ada(ctx, ptbl, vp_mod, 1, vd);

        // video self attention
        let attn1 = AttnWeights::load(w, &format!("{bp}.attn1"))?;
        let nx = ctx.modulate(rms(ctx, vx, 1e-6, None), scale_msa, Some(shift_msa));
        let a = attention(
            ctx,
            &attn1,
            nx,
            None,
            Some(v_pe),
            Some(v_pe),
            hp.v_heads,
            inp.flash,
        );
        vx = ctx.add(vx, ctx.mul(a, gate_msa));
        // text cross attention
        let attn2 = AttnWeights::load(w, &format!("{bp}.attn2"))?;
        let qx = ctx.modulate(rms(ctx, vx, 1e-6, None), scale_q, Some(shift_q));
        let cx = ctx.modulate(v_ctx, scale_kv, Some(shift_kv));
        let c = attention(ctx, &attn2, qx, Some(cx), None, None, hp.v_heads, inp.flash);
        vx = ctx.add(vx, ctx.mul(c, gate_ca));

        if let (
            Some(mut a),
            Some((a_mod, _, ap_mod, ca_a_ss, ca_v_ss, ca_a2v, ca_v2a)),
            Some((a_pe, v_cross, a_cross)),
        ) = (ax, audio_mods, av_pe)
        {
            let atbl = w.get(&format!("{bp}.audio_scale_shift_table"))?;
            let aptbl = w.get(&format!("{bp}.audio_prompt_scale_shift_table"))?;
            let am = |i| ada(ctx, atbl, a_mod, i, ad);
            let (a_shift_msa, a_scale_msa, a_gate_msa) = (am(0), am(1), am(2));
            let (a_shift_mlp, a_scale_mlp, a_gate_mlp) = (am(3), am(4), am(5));
            let (a_shift_q, a_scale_q, a_gate_ca) = (am(6), am(7), am(8));
            let a_shift_kv = ada(ctx, aptbl, ap_mod, 0, ad);
            let a_scale_kv = ada(ctx, aptbl, ap_mod, 1, ad);

            let attn1 = AttnWeights::load(w, &format!("{bp}.audio_attn1"))?;
            let nx = ctx.modulate(rms(ctx, a, 1e-6, None), a_scale_msa, Some(a_shift_msa));
            let o = attention(
                ctx,
                &attn1,
                nx,
                None,
                Some(a_pe),
                Some(a_pe),
                hp.a_heads,
                inp.flash,
            );
            a = ctx.add(a, ctx.mul(o, a_gate_msa));
            let attn2 = AttnWeights::load(w, &format!("{bp}.audio_attn2"))?;
            let qx = ctx.modulate(rms(ctx, a, 1e-6, None), a_scale_q, Some(a_shift_q));
            let cx = ctx.modulate(a_ctx.unwrap(), a_scale_kv, Some(a_shift_kv));
            let o = attention(ctx, &attn2, qx, Some(cx), None, None, hp.a_heads, inp.flash);
            a = ctx.add(a, ctx.mul(o, a_gate_ca));

            // audio <-> video cross attention; rows 0..3 scale/shift adaln, row 4 gate adaln
            let a_tbl = w.get(&format!("{bp}.scale_shift_table_a2v_ca_audio"))?;
            let v_tbl = w.get(&format!("{bp}.scale_shift_table_a2v_ca_video"))?;
            let a_scale_a2v = ada(ctx, a_tbl, ca_a_ss, 0, ad);
            let a_shift_a2v = ada(ctx, a_tbl, ca_a_ss, 1, ad);
            let a_scale_v2a = ada(ctx, a_tbl, ca_a_ss, 2, ad);
            let a_shift_v2a = ada(ctx, a_tbl, ca_a_ss, 3, ad);
            let v_scale_a2v = ada(ctx, v_tbl, ca_v_ss, 0, vd);
            let v_shift_a2v = ada(ctx, v_tbl, ca_v_ss, 1, vd);
            let v_scale_v2a = ada(ctx, v_tbl, ca_v_ss, 2, vd);
            let v_shift_v2a = ada(ctx, v_tbl, ca_v_ss, 3, vd);
            let gate_a2v = ctx.add(table_row(ctx, v_tbl, 4), ca_a2v);
            let gate_v2a = ctx.add(table_row(ctx, a_tbl, 4), ca_v2a);
            let a_norm = rms(ctx, a, 1e-6, None);

            let a2v = AttnWeights::load(w, &format!("{bp}.audio_to_video_attn"))?;
            let vq = ctx.modulate(rms(ctx, vx, 1e-6, None), v_scale_a2v, Some(v_shift_a2v));
            let ak = ctx.modulate(a_norm, a_scale_a2v, Some(a_shift_a2v));
            let o = attention(
                ctx,
                &a2v,
                vq,
                Some(ak),
                Some(v_cross),
                Some(a_cross),
                hp.a_heads,
                inp.flash,
            );
            vx = ctx.add(vx, ctx.mul(o, gate_a2v));

            let v2a = AttnWeights::load(w, &format!("{bp}.video_to_audio_attn"))?;
            let aq = ctx.modulate(a_norm, a_scale_v2a, Some(a_shift_v2a));
            let vk = ctx.modulate(rms(ctx, vx, 1e-6, None), v_scale_v2a, Some(v_shift_v2a));
            let o = attention(
                ctx,
                &v2a,
                aq,
                Some(vk),
                Some(a_cross),
                Some(v_cross),
                hp.a_heads,
                inp.flash,
            );
            a = ctx.add(a, ctx.mul(o, gate_v2a));

            // feed forwards
            let y = ctx.modulate(rms(ctx, vx, 1e-6, None), scale_mlp, Some(shift_mlp));
            vx = ctx.add(
                vx,
                ctx.mul(feed_forward(ctx, w, &format!("{bp}.ff"), y)?, gate_mlp),
            );
            let y = ctx.modulate(rms(ctx, a, 1e-6, None), a_scale_mlp, Some(a_shift_mlp));
            a = ctx.add(
                a,
                ctx.mul(
                    feed_forward(ctx, w, &format!("{bp}.audio_ff"), y)?,
                    a_gate_mlp,
                ),
            );
            ax = Some(a);
        } else {
            let y = ctx.modulate(rms(ctx, vx, 1e-6, None), scale_mlp, Some(shift_mlp));
            vx = ctx.add(
                vx,
                ctx.mul(feed_forward(ctx, w, &format!("{bp}.ff"), y)?, gate_mlp),
            );
        }
    }

    // outputs
    let tbl = w.get("scale_shift_table")?;
    let shift = ctx.add(table_row(ctx, tbl, 0), v_emb);
    let scale = ctx.add(table_row(ctx, tbl, 1), v_emb);
    let y = ctx.modulate(ctx.norm(vx, 1e-6), scale, Some(shift));
    let v_out = ctx.linear(y, w.get("proj_out.weight")?, Some(w.get("proj_out.bias")?));
    g.mark_output(v_out);
    let mut a_out: Option<Tensor> = None;
    if let (Some(a), Some((_, a_emb, ..))) = (ax, audio_mods) {
        let tbl = w.get("audio_scale_shift_table")?;
        let shift = ctx.add(table_row(ctx, tbl, 0), a_emb);
        let scale = ctx.add(table_row(ctx, tbl, 1), a_emb);
        let y = ctx.modulate(ctx.norm(a, 1e-6), scale, Some(shift));
        let o = ctx.linear(
            y,
            w.get("audio_proj_out.weight")?,
            Some(w.get("audio_proj_out.bias")?),
        );
        g.mark_output(o);
        a_out = Some(o);
    }
    g.compute(be)?;
    Ok((
        g.output_f32(v_out),
        a_out.map(|o| g.output_f32(o)).unwrap_or_default(),
    ))
}

/// Distilled checkpoints use a fixed schedule, others the LTX2 shifted one.
pub fn sigmas(n_steps: i32, n_tokens: i64, distilled: bool) -> Vec<f32> {
    if distilled {
        const S: [f32; 8] = [
            1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875,
        ];
        let mut s: Vec<f32> = if n_steps <= 0 || n_steps as usize >= S.len() {
            S.to_vec()
        } else {
            (0..n_steps)
                .map(|i| {
                    S[((i as f64) * (S.len() - 1) as f64 / (n_steps - 1).max(1) as f64).round()
                        as usize]
                })
                .collect()
        };
        s.push(0.0);
        return s;
    }
    let n = if n_steps <= 0 { 20 } else { n_steps.max(2) } as usize;
    let (max_shift, base_shift, terminal) = (2.05f32, 0.95f32, 0.1f32);
    let m = (max_shift - base_shift) / (4096.0 - 1024.0);
    let b = base_shift - m * 1024.0;
    let shift = (n_tokens as f32 * m + b).exp();
    let mut s: Vec<f32> = (0..=n)
        .map(|i| {
            let sigma = 1.0 - i as f32 / n as f32;
            if sigma != 0.0 {
                shift / (shift + (1.0 / sigma - 1.0))
            } else {
                0.0
            }
        })
        .collect();
    let scale = (1.0 - s[n - 1]) / (1.0 - terminal);
    if scale > 0.0 {
        for v in s.iter_mut().take(n) {
            *v = 1.0 - (1.0 - *v) / scale;
        }
    }
    s[n] = 0.0;
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distilled_schedule_ends_at_zero() {
        let s = sigmas(0, 256, true);
        assert_eq!(s.len(), 9);
        assert_eq!(s[0], 1.0);
        assert_eq!(*s.last().unwrap(), 0.0);
    }

    #[test]
    fn shifted_schedule_is_monotonic_and_finite() {
        for n in [1, 2, 8, 20] {
            let s = sigmas(n, 1024, false);
            assert!(s.iter().all(|v| v.is_finite()), "{s:?}");
            assert!(s.windows(2).all(|w| w[0] > w[1]), "{s:?}");
            assert_eq!(*s.last().unwrap(), 0.0);
        }
    }
}

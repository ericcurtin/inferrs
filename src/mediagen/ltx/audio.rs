//! LTX-2 audio: latents -> causal 2-D VAE -> mel -> BigVGAN style vocoder (16 kHz) -> bandwidth extension (48 kHz).
//! Layouts: 2-D activations `[freq, time, C, 1]`, 1-D signals `[T, C, 1]`.

use anyhow::{anyhow, bail, Result};

use super::super::backend::{Backend, Ctx, Graph};
use super::super::ffi::{self, Tensor};
use super::super::weights::Weights;
use super::{AudioLatent, Model, PI};

/// 3x3 conv, zero padded: causal along time (2 before), symmetric along frequency.
fn avae_conv(g: &Ctx, w: &Weights, prefix: &str, mut x: Tensor, k: i32) -> Result<Tensor> {
    let kw = w.get(&format!("{prefix}.weight"))?;
    if k == 3 {
        x = g.pad_ext(x, 1, 1, 2, 0, 0, 0, 0, 0);
    }
    let mut y = g.conv_2d(kw, x, 1, 1, 0, 0, 1, 1);
    if let Some(b) = w.opt(&format!("{prefix}.bias")) {
        y = g.add(y, g.reshape_4d(b, 1, 1, ffi::shape(b)[0], 1));
    }
    Ok(y)
}

fn avae_pixel_norm(g: &Ctx, x: Tensor) -> Tensor {
    let p = g.cont(g.permute(x, 1, 2, 0, 3)); // [C, F, T, 1]
    let p = g.rms_norm(p, 1e-6);
    g.cont(g.permute(p, 2, 0, 1, 3))
}

fn avae_resblock(g: &Ctx, w: &Weights, prefix: &str, mut x: Tensor) -> Result<Tensor> {
    let h = g.silu(avae_pixel_norm(g, x));
    let h = avae_conv(g, w, &format!("{prefix}.conv1.conv"), h, 3)?;
    let h = g.silu(avae_pixel_norm(g, h));
    let h = avae_conv(g, w, &format!("{prefix}.conv2.conv"), h, 3)?;
    if w.has(&format!("{prefix}.nin_shortcut.conv.weight")) {
        x = avae_conv(g, w, &format!("{prefix}.nin_shortcut.conv"), x, 1)?;
    }
    Ok(g.add(x, h))
}

/// Nearest x2 upsample, conv, drop the first time row.
fn avae_upsample(g: &Ctx, w: &Weights, prefix: &str, mut x: Tensor) -> Result<Tensor> {
    let s = ffi::shape(x);
    x = g.interpolate(
        x,
        s[0] * 2,
        s[1] * 2,
        s[2],
        s[3],
        ffi::GGML_SCALE_MODE_NEAREST,
    );
    x = avae_conv(g, w, &format!("{prefix}.conv.conv"), x, 3)?;
    let s = ffi::shape(x);
    let nb = ffi::strides(x);
    Ok(g.cont(g.view_4d(x, s[0], s[1] - 1, s[2], s[3], nb[1], nb[2], nb[3], nb[1])))
}

/// Latents `[freq, time, 8, 1]` -> mel `[64, time, 2, 1]`.
fn avae_decode(
    g: &Ctx,
    w: &Weights,
    z: Tensor,
    n_levels: i32,
    n_res_blocks: i32,
) -> Result<Tensor> {
    let pre = "audio_vae.decoder";
    let mut h = avae_conv(g, w, &format!("{pre}.conv_in.conv"), z, 3)?;
    h = avae_resblock(g, w, &format!("{pre}.mid.block_1"), h)?;
    h = avae_resblock(g, w, &format!("{pre}.mid.block_2"), h)?;
    for level in (0..n_levels).rev() {
        for i in 0..=n_res_blocks {
            h = avae_resblock(g, w, &format!("{pre}.up.{level}.block.{i}"), h)?;
        }
        if level != 0 {
            h = avae_upsample(g, w, &format!("{pre}.up.{level}.upsample"), h)?;
        }
    }
    h = g.silu(avae_pixel_norm(g, h));
    avae_conv(g, w, &format!("{pre}.conv_out.conv"), h, 3)
}

// vocoder building blocks (1-D, [T, C, 1])

fn conv1d(g: &Ctx, w: &Weights, prefix: &str, x: Tensor, pad: i32, dil: i32) -> Result<Tensor> {
    let kw = w.get(&format!("{prefix}.weight"))?;
    let mut y = g.conv_1d(kw, x, 1, pad, dil);
    if let Some(b) = w.opt(&format!("{prefix}.bias")) {
        y = g.add(y, g.reshape_3d(b, 1, ffi::shape(b)[0], 1));
    }
    Ok(y)
}

/// PyTorch ConvTranspose1d(k, stride, padding = (k - stride) / 2).
fn conv_transpose1d(g: &Ctx, w: &Weights, prefix: &str, x: Tensor, stride: i32) -> Result<Tensor> {
    let kw = w.get(&format!("{prefix}.weight"))?; // [K, OC, IC]
    let k = ffi::shape(kw)[0] as i32;
    let pad = (k - stride) / 2;
    let s = ffi::shape(x);
    let x2 = g.reshape_2d(x, s[0], s[1]);
    let mut y = g.conv_transpose_1d(kw, x2, stride, 0, 1); // [T', OC]
    if pad > 0 {
        let s = ffi::shape(y);
        let nb = ffi::strides(y);
        y = g.cont(g.view_2d(y, s[0] - 2 * pad as i64, s[1], nb[1], pad as usize * nb[0]));
    }
    let s = ffi::shape(y);
    y = g.reshape_3d(y, s[0], s[1], 1);
    if let Some(b) = w.opt(&format!("{prefix}.bias")) {
        y = g.add(y, g.reshape_3d(b, 1, ffi::shape(b)[0], 1));
    }
    Ok(y)
}

/// Replicate padding along time.
fn replicate_pad1d(g: &Ctx, x: Tensor, left: i64, right: i64) -> Tensor {
    let [t, c, ..] = ffi::shape(x);
    let nb = ffi::strides(x);
    let mut out = x;
    if left > 0 {
        let first = g.view_3d(x, 1, c, 1, nb[1], nb[2], 0);
        let rep = g.repeat(first, g.new_tensor(ffi::ty::F32, &[left, c, 1]));
        out = g.concat(rep, out, 0);
    }
    if right > 0 {
        let last = g.view_3d(x, 1, c, 1, nb[1], nb[2], (t - 1) as usize * nb[0]);
        let rep = g.repeat(last, g.new_tensor(ffi::ty::F32, &[right, c, 1]));
        out = g.concat(out, rep, 0);
    }
    out
}

/// One filter for every channel: channels become the batch of a 1-channel conv; x: [T, C, 1], filter: [K, 1, 1].
fn filter_conv1d(g: &Ctx, filter: Tensor, x: Tensor, stride: i32, pad: i32) -> Tensor {
    let [t, c, ..] = ffi::shape(x);
    let xb = g.reshape_3d(x, t, 1, c);
    let y = g.conv_1d(filter, xb, stride, pad, 1); // [T', 1, C]
    g.reshape_3d(y, ffi::shape(y)[0], c, 1)
}

/// Zero insertion: [T, C] -> [(T-1)*ratio+1, C].
/// Concat rather than pad: the CUDA pad kernel maps ne1 to gridDim.y (max 65535).
fn zero_stuff(g: &Ctx, x: Tensor, ratio: i64) -> Tensor {
    let [t, c, ..] = ffi::shape(x);
    let mut v = g.reshape_3d(g.cont(x), 1, t, c);
    let zeros = g.scale(v, 0.0);
    for _ in 1..ratio {
        v = g.concat(v, zeros, 0);
    }
    v = g.reshape_3d(v, ratio * t, c, 1);
    let nb = ffi::strides(v);
    g.cont(g.view_3d(v, (t - 1) * ratio + 1, c, 1, nb[1], nb[2], 0))
}

/// BigVGAN UpSample1d; the filter is symmetric so the transposed conv is zero stuffing + full conv.
fn aa_upsample(
    g: &Ctx,
    filter: Tensor,
    mut x: Tensor,
    ratio: i64,
    pad: i64,
    pad_left: i64,
    pad_right: i64,
) -> Tensor {
    let k = ffi::shape(filter)[0] as i32;
    x = replicate_pad1d(g, x, pad, pad);
    x = zero_stuff(g, x, ratio);
    let y = g.scale(filter_conv1d(g, filter, x, 1, k - 1), ratio as f32);
    let [t, c, ..] = ffi::shape(y);
    let nb = ffi::strides(y);
    g.cont(g.view_3d(
        y,
        t - pad_left - pad_right,
        c,
        1,
        nb[1],
        nb[2],
        pad_left as usize * nb[0],
    ))
}

/// BigVGAN DownSample1d: replicate pad then strided lowpass.
fn aa_downsample(g: &Ctx, filter: Tensor, x: Tensor, ratio: i32) -> Tensor {
    let k = ffi::shape(filter)[0];
    let x = replicate_pad1d(g, x, k / 2 - 1, k / 2);
    filter_conv1d(g, filter, x, ratio, 0)
}

/// Snake beta: x + inv_beta * sin(alpha * x)^2, parameters precomputed by `transform`.
fn snake_beta(g: &Ctx, w: &Weights, prefix: &str, x: Tensor) -> Result<Tensor> {
    let a = w.get(&format!("{prefix}.alpha"))?;
    let ib = w.get(&format!("{prefix}.beta"))?;
    let s = g.sin(g.mul(x, g.reshape_3d(a, 1, ffi::shape(a)[0], 1)));
    let s = g.mul(g.sqr(s), g.reshape_3d(ib, 1, ffi::shape(ib)[0], 1));
    Ok(g.add(x, s))
}

/// Activation1d: anti-aliased snake (2x up, act, 2x down), kernel 12.
fn activation1d(g: &Ctx, w: &Weights, prefix: &str, mut x: Tensor) -> Result<Tensor> {
    let up = w.get(&format!("{prefix}.upsample.filter"))?;
    let down = w.get(&format!("{prefix}.downsample.lowpass.filter"))?;
    let k = ffi::shape(up)[0];
    let ratio = 2;
    let pad = k / ratio - 1;
    let pad_left = pad * ratio + (k - ratio) / 2;
    let pad_right = pad * ratio + (k - ratio + 1) / 2;
    x = aa_upsample(g, up, x, ratio, pad, pad_left, pad_right);
    x = snake_beta(g, w, &format!("{prefix}.act"), x)?;
    Ok(aa_downsample(g, down, x, ratio as i32))
}

fn amp_block(
    g: &Ctx,
    w: &Weights,
    prefix: &str,
    mut x: Tensor,
    kernel: i32,
    dil: [i32; 3],
) -> Result<Tensor> {
    for (i, d) in dil.iter().enumerate() {
        let xt = activation1d(g, w, &format!("{prefix}.acts1.{i}"), x)?;
        let xt = conv1d(
            g,
            w,
            &format!("{prefix}.convs1.{i}"),
            xt,
            (kernel * d - d) / 2,
            *d,
        )?;
        let xt = activation1d(g, w, &format!("{prefix}.acts2.{i}"), xt)?;
        let xt = conv1d(
            g,
            w,
            &format!("{prefix}.convs2.{i}"),
            xt,
            (kernel - 1) / 2,
            1,
        )?;
        x = g.add(x, xt);
    }
    Ok(x)
}

/// x: [T, 128, 1] stacked stereo mels -> [T * prod(up_rates), 2, 1]
fn vocoder(
    g: &Ctx,
    w: &Weights,
    prefix: &str,
    mut x: Tensor,
    up_rates: &[i32],
    clamp_output: bool,
) -> Result<Tensor> {
    const KERNELS: [i32; 3] = [3, 7, 11];
    const DIL: [i32; 3] = [1, 3, 5];
    x = conv1d(g, w, &format!("{prefix}.conv_pre"), x, 3, 1)?;
    let nk = KERNELS.len();
    for (i, r) in up_rates.iter().enumerate() {
        x = conv_transpose1d(g, w, &format!("{prefix}.ups.{i}"), x, *r)?;
        let mut xs: Option<Tensor> = None;
        for (j, k) in KERNELS.iter().enumerate() {
            let r = amp_block(
                g,
                w,
                &format!("{prefix}.resblocks.{}", i * nk + j),
                x,
                *k,
                DIL,
            )?;
            xs = Some(match xs {
                Some(acc) => g.add(acc, r),
                None => r,
            });
        }
        x = g.scale(xs.unwrap(), 1.0 / nk as f32);
    }
    x = activation1d(g, w, &format!("{prefix}.act_post"), x)?;
    x = conv1d(g, w, &format!("{prefix}.conv_post"), x, 3, 1)?;
    if clamp_output {
        x = g.clamp(x, -1.0, 1.0);
    }
    Ok(x)
}

/// Causal log-mel of one channel [T, 1, 1] -> [frames, n_mels, 1].
/// One channel at a time: ggml_conv_1d only lays out its output correctly for a batch of one.
fn mel_stft(g: &Ctx, w: &Weights, mut x: Tensor, n_fft: i32, hop: i32) -> Result<Tensor> {
    let basis = w.get("vocoder.mel_stft.stft_fn.forward_basis")?; // [n_fft, 1, 2 * n_freqs]
    let melb = w.get("vocoder.mel_stft.mel_basis")?; // [n_freqs, n_mels]
    x = g.pad_ext(x, n_fft - hop, 0, 0, 0, 0, 0, 0, 0);
    let spec = g.conv_1d(basis, x, hop, 0, 1); // [frames, 2 * n_freqs, 1]
    let [f, n2, ..] = ffi::shape(spec);
    let nf = n2 / 2;
    let nb = ffi::strides(spec);
    let re = g.view_2d(spec, f, nf, nb[1], 0);
    let im = g.view_2d(spec, f, nf, nb[1], nf as usize * nb[1]);
    let mag = g.sqrt(g.add(g.sqr(g.cont(re)), g.sqr(g.cont(im)))); // [F, nf]
    let mag = g.cont(g.transpose(mag)); // [nf, F]
    let mel = g.mul_mat(melb, mag); // [n_mels, F]
    let mel = g.log(g.clamp(mel, 1e-5, f32::INFINITY));
    let mel = g.cont(g.transpose(mel)); // [F, n_mels]
    let s = ffi::shape(mel);
    Ok(g.reshape_3d(mel, s[0], s[1], 1))
}

/// torchaudio-style hann windowed sinc resampler filter (rolloff 0.99, width 6); returns (filter, pad, pad_left, pad_right).
fn hann_resample_filter(ratio: i64) -> (Vec<f32>, i64, i64, i64) {
    let rolloff = 0.99f64;
    let lpw = 6i64;
    let width = (lpw as f64 / rolloff).ceil() as i64;
    let k = 2 * width * ratio + 1;
    let f = (0..k)
        .map(|i| {
            let t = (i as f64 / ratio as f64 - width as f64) * rolloff;
            let tc = t.clamp(-(lpw as f64), lpw as f64);
            let win = (tc * PI / lpw as f64 / 2.0).cos().powi(2);
            let sinc = if t == 0.0 {
                1.0
            } else {
                (PI * t).sin() / (PI * t)
            };
            (sinc * win * rolloff / ratio as f64) as f32
        })
        .collect();
    (f, width, 2 * width * ratio, k - ratio)
}

/// Decodes audio latents to interleaved PCM; returns (sample_rate, n_channels, pcm).
pub fn decode(model: &Model, be: &Backend, lat: &AudioLatent) -> Result<(i32, i32, Vec<f32>)> {
    let hp = &model.hp;
    let w = model
        .audio_vae
        .as_ref()
        .ok_or_else(|| anyhow!("no audio VAE loaded"))?;
    let (t, c, fq) = (lat.n_frames, lat.channels, lat.freq);

    let mean = w.read_f32("audio_vae.per_channel_statistics.mean-of-means")?;
    let stdv = w.read_f32("audio_vae.per_channel_statistics.std-of-means")?;
    if mean.len() as i64 != c * fq || stdv.len() as i64 != c * fq {
        bail!("unexpected audio latent statistics");
    }
    // denormalize, lay out as [freq, time, channels]
    let mut z = vec![0f32; (fq * t * c) as usize];
    for ti in 0..t {
        for ci in 0..c {
            for f in 0..fq {
                let k = (ci * fq + f) as usize;
                z[(f + fq * (ti + t * ci)) as usize] =
                    lat.x[(ti * c * fq) as usize + k] * stdv[k] + mean[k];
            }
        }
    }

    let mut g = Graph::new(be.api, 64 * 1024)?;
    let zt = g.input_f32(&[fq, t, c, 1], &z, "audio_latent");
    let (hf, pad, pad_left, pad_right) = hann_resample_filter(3);
    let hfilt = g.input_f32(&[hf.len() as i64, 1, 1], &hf, "resample_filter");
    let ctx = &g.ctx;

    // vae decoder: -> [64, T_mel, 2, 1]
    let mut mel = avae_decode(ctx, w, zt, 3, 2)?;
    let t_mel = t * hp.audio_latent_downsample - (hp.audio_latent_downsample - 1);
    let ms = ffi::shape(mel);
    if ms[1] < t_mel || ms[0] * ms[2] != 128 {
        bail!("unexpected mel shape [{}, {}, {}]", ms[0], ms[1], ms[2]);
    }
    if ms[1] != t_mel {
        let nb = ffi::strides(mel);
        mel = ctx.cont(ctx.view_4d(mel, ms[0], t_mel, ms[2], 1, nb[1], nb[2], nb[3], 0));
    }
    // stereo mels stacked along channels: [T_mel, 128, 1]
    let vin = ctx.cont(ctx.permute(mel, 1, 0, 2, 3)); // [T_mel, 64, 2]
    let vs = ffi::shape(vin);
    let vin = ctx.reshape_3d(vin, vs[0], vs[1] * vs[2], 1);

    // vocoder to 16 kHz stereo: [T16, 2, 1]
    let x16 = vocoder(ctx, w, "vocoder.vocoder", vin, &[5, 2, 2, 2, 2, 2], true)?;

    let mut out = x16;
    let mut out_rate = hp.audio_sample_rate as i32;
    if w.has("vocoder.bwe_generator.conv_pre.weight") {
        let (in_rate, out_rate_bwe, hop, n_fft) = (16000i64, 48000i32, 80i64, 512i32);
        let ratio = out_rate_bwe as i64 / in_rate;
        let t_low = ffi::shape(x16)[0];
        let t_out = t_low * ratio;
        let mut xpad = x16;
        if t_low % hop != 0 {
            xpad = ctx.pad(x16, (hop - t_low % hop) as i32, 0, 0, 0);
        }
        // mel of each low rate channel, stacked along channels: [frames, 128, 1]
        let ps = ffi::shape(xpad);
        let pnb = ffi::strides(xpad);
        let mut m: Option<Tensor> = None;
        for ch in 0..ps[1] {
            let xc = ctx.view_3d(xpad, ps[0], 1, 1, pnb[1], pnb[2], ch as usize * pnb[1]);
            let mc = mel_stft(ctx, w, ctx.cont(xc), n_fft, hop as i32)?;
            m = Some(match m {
                Some(acc) => ctx.concat(acc, mc, 1),
                None => mc,
            });
        }
        let residual = vocoder(
            ctx,
            w,
            "vocoder.bwe_generator",
            m.unwrap(),
            &[6, 5, 2, 2, 2],
            false,
        )?; // [frames * 480, 2]
        let skip = aa_upsample(ctx, hfilt, xpad, ratio, pad, pad_left, pad_right);
        if ffi::shape(skip)[0] != ffi::shape(residual)[0] {
            bail!("bandwidth extension length mismatch");
        }
        out = ctx.clamp(ctx.add(residual, skip), -1.0, 1.0);
        let os = ffi::shape(out);
        let onb = ffi::strides(out);
        out = ctx.cont(ctx.view_3d(out, t_out, os[1], 1, onb[1], onb[2], 0));
        out_rate = out_rate_bwe;
    }

    g.mark_output(out);
    g.compute(be)?;
    let y = g.output_f32(out);
    let [n, nch, ..] = ffi::shape(out);
    let mut pcm = vec![0f32; (n * nch) as usize];
    for ti in 0..n as usize {
        for ci in 0..nch as usize {
            pcm[ti * nch as usize + ci] = y[ti + n as usize * ci];
        }
    }
    Ok((out_rate, nch as i32, pcm))
}

/// Snake parameters: alpha -> exp(alpha), beta -> 1 / (exp(beta) + eps).
pub fn transform(name: &str, data: &mut [f32]) {
    if name.ends_with(".act.alpha") {
        for v in data.iter_mut() {
            *v = v.exp();
        }
    } else if name.ends_with(".act.beta") {
        for v in data.iter_mut() {
            *v = 1.0 / (v.exp() + 1e-9);
        }
    }
}

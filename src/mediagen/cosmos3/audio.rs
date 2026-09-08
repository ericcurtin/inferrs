//! The sound tokenizer's Oobleck decoder (diffusers `Cosmos3AVAEAudioTokenizer`):
//! 25 latent frames/s to 48 kHz stereo. Activations are `[T, C, 1]`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::mediagen::backend::{Backend, Ctx, Graph};
use crate::mediagen::ffi::{self, Tensor};
use crate::mediagen::weights::{read_safetensors_raw, write_safetensors_raw, RawTensor, Weights};

pub const SAMPLE_RATE: i32 = 48000;
pub const HOP: i64 = 1920;
pub const LATENT_DIM: i64 = 64;
/// `dec_strides` reversed: the decoder's upsampling ratios.
const STRIDES: [i64; 5] = [8, 6, 5, 4, 2];

/// Rewrites `src` as `dst` with every `X.weight_g`/`X.weight_v` pair folded into
/// `X.weight = g * v / ||v||` (norm over all but the first dimension, as
/// `weight_norm(dim=0)` does).
pub fn fold_weight_norm(src: &Path, dst: &Path) -> Result<()> {
    let tensors = read_safetensors_raw(src)?;
    let by_name: HashMap<&str, &RawTensor> = tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut out = Vec::new();
    for t in &tensors {
        if t.name.ends_with(".weight_g") {
            continue;
        }
        let Some(stem) = t.name.strip_suffix(".weight_v") else {
            out.push(RawTensor {
                name: t.name.clone(),
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
                bytes: t.bytes.clone(),
            });
            continue;
        };
        let g_name = format!("{stem}.weight_g");
        let g = by_name
            .get(g_name.as_str())
            .with_context(|| format!("{} without {g_name}", t.name))?
            .f32s()?;
        let mut v = t.f32s()?;
        let d0 = *t
            .shape
            .first()
            .filter(|&&d| d > 0)
            .with_context(|| format!("{}: bad shape", t.name))? as usize;
        if g.len() != d0 || !v.len().is_multiple_of(d0) {
            bail!("{}: {} norms for {d0} rows of {}", g_name, g.len(), v.len());
        }
        let per = v.len() / d0;
        for (i, row) in v.chunks_exact_mut(per).enumerate() {
            let norm = row.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt() as f32;
            let s = g[i] / norm;
            row.iter_mut().for_each(|x| *x *= s);
        }
        out.push(RawTensor {
            name: format!("{stem}.weight"),
            dtype: "F32".into(),
            shape: t.shape.clone(),
            bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    write_safetensors_raw(dst, &out, "weight_norm folded")
}

/// Snake parameters as the graph uses them: alpha -> exp(alpha), beta -> 1 / (exp(beta) + eps).
pub fn transform(name: &str, data: &mut [f32]) {
    if name.ends_with(".alpha") {
        for v in data.iter_mut() {
            *v = v.exp();
        }
    } else if name.ends_with(".beta") {
        for v in data.iter_mut() {
            *v = 1.0 / (v.exp() + 1e-9);
        }
    }
}

fn param(g: &Ctx, t: Tensor) -> Tensor {
    let n: i64 = ffi::shape(t).iter().product();
    g.reshape_3d(t, 1, n, 1)
}

fn snake(g: &Ctx, w: &Weights, prefix: &str, x: Tensor) -> Result<Tensor> {
    let a = param(g, w.get(&format!("{prefix}.alpha"))?);
    let ib = param(g, w.get(&format!("{prefix}.beta"))?);
    let s = g.sin(g.mul(x, a));
    Ok(g.add(x, g.mul(g.sqr(s), ib)))
}

fn conv1d(g: &Ctx, w: &Weights, prefix: &str, x: Tensor, pad: i32, dil: i32) -> Result<Tensor> {
    let k = w.get(&format!("{prefix}.weight"))?;
    let mut y = g.conv_1d(k, x, 1, pad, dil);
    if let Some(b) = w.opt(&format!("{prefix}.bias")) {
        y = g.add(y, param(g, b));
    }
    Ok(y)
}

/// `ConvTranspose1d(k = 2s, stride s, padding ceil(s/2), output_padding s % 2)`.
fn conv_transpose(g: &Ctx, w: &Weights, prefix: &str, x: Tensor, stride: i64) -> Result<Tensor> {
    let k = w.get(&format!("{prefix}.weight"))?; // [K, OC, IC]
    let [t, c, ..] = ffi::shape(x);
    let y = g.conv_transpose_1d(k, g.reshape_2d(x, t, c), stride as i32, 0, 1); // [(t-1)s + K, OC]
    let pad = (stride + 1) / 2;
    let out_pad = stride % 2;
    let [ty, oc, ..] = ffi::shape(y);
    let keep = ty - pad - (pad - out_pad);
    let nb = ffi::strides(y);
    let mut y = g.cont(g.view_2d(y, keep, oc, nb[1], (pad as usize) * nb[0]));
    y = g.reshape_3d(y, keep, oc, 1);
    if let Some(b) = w.opt(&format!("{prefix}.bias")) {
        y = g.add(y, param(g, b));
    }
    Ok(y)
}

fn res_unit(g: &Ctx, w: &Weights, prefix: &str, x: Tensor, dil: i32) -> Result<Tensor> {
    let h = snake(g, w, &format!("{prefix}.snake1"), x)?;
    let h = conv1d(g, w, &format!("{prefix}.conv1"), h, 3 * dil, dil)?;
    let h = snake(g, w, &format!("{prefix}.snake2"), h)?;
    let h = conv1d(g, w, &format!("{prefix}.conv2"), h, 0, 1)?;
    Ok(g.add(x, h))
}

/// Decodes sound latents `[T][64]` to interleaved stereo PCM at 48 kHz.
pub fn decode(w: &Weights, be: &Backend, latents: &[f32], n_frames: i64) -> Result<Vec<f32>> {
    if latents.len() as i64 != n_frames * LATENT_DIM {
        bail!("sound latents do not match {n_frames} x {LATENT_DIM}");
    }
    // [T][C] -> ggml [T, C, 1]
    let mut x = vec![0f32; latents.len()];
    for t in 0..n_frames as usize {
        for c in 0..LATENT_DIM as usize {
            x[t + n_frames as usize * c] = latents[t * LATENT_DIM as usize + c];
        }
    }
    let mut gr = Graph::new(be.api, 4 * 1024)?;
    let z = gr.input_f32(&[n_frames, LATENT_DIM, 1], &x, "sound");
    let g = &gr.ctx;
    let mut h = conv1d(g, w, "decoder.conv1", z, 3, 1)?;
    for (i, &stride) in STRIDES.iter().enumerate() {
        let p = format!("decoder.block.{i}");
        h = snake(g, w, &format!("{p}.snake1"), h)?;
        h = conv_transpose(g, w, &format!("{p}.conv_t1"), h, stride)?;
        h = res_unit(g, w, &format!("{p}.res_unit1"), h, 1)?;
        h = res_unit(g, w, &format!("{p}.res_unit2"), h, 3)?;
        h = res_unit(g, w, &format!("{p}.res_unit3"), h, 9)?;
    }
    h = snake(g, w, "decoder.snake1", h)?;
    h = conv1d(g, w, "decoder.conv2", h, 3, 1)?;
    let out = g.cont(g.clamp(h, -1.0, 1.0)); // [N, 2, 1]
    gr.mark_output(out);
    gr.compute(be)?;
    let [n, ch, ..] = ffi::shape(out);
    let y = gr.output_f32(out);
    let (n, ch) = (n as usize, ch as usize);
    let mut pcm = vec![0f32; n * ch];
    for t in 0..n {
        for c in 0..ch {
            pcm[t * ch + c] = y[t + n * c];
        }
    }
    Ok(pcm)
}

/// The folded-weights file for `src`, created in `cache_dir` when missing.
pub fn folded_path(src: &Path, cache_dir: &Path) -> Result<std::path::PathBuf> {
    let name = src
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("sound_tokenizer");
    let dst = cache_dir.join(format!("{name}.folded.safetensors"));
    if !dst.exists() {
        std::fs::create_dir_all(cache_dir)?;
        fold_weight_norm(src, &dst).with_context(|| format!("folding {}", src.display()))?;
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_writes_a_readable_file() {
        let dir = std::env::temp_dir().join(format!("llmman-fold-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let raw = |name: &str, shape: &[i64], data: &[f32]| RawTensor {
            name: name.into(),
            dtype: "F32".into(),
            shape: shape.to_vec(),
            bytes: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        // w[i] = g[i] * v[i] / ||v[i]||
        let src = dir.join("in.safetensors");
        write_safetensors_raw(
            &src,
            &[
                raw("a.weight_v", &[2, 3], &[3.0, 4.0, 0.0, 0.0, 0.0, 2.0]),
                raw("a.weight_g", &[2, 1], &[10.0, 1.0]),
                raw("b.alpha", &[1, 2, 1], &[0.5, 0.25]),
            ],
            "test",
        )
        .unwrap();
        let dst = dir.join("out.safetensors");
        fold_weight_norm(&src, &dst).unwrap();
        let out = read_safetensors_raw(&dst).unwrap();
        let names: Vec<&str> = out.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["a.weight", "b.alpha"]);
        assert_eq!(out[0].f32s().unwrap(), [6.0, 8.0, 0.0, 0.0, 0.0, 1.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

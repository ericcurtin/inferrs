//! The causal video VAE decoder. Activations are `[W, H, C, T]`.

use anyhow::{bail, Result};

use super::super::backend::{Backend, Ctx, Graph};
use super::super::ffi::{self, Tensor};
use super::super::weights::Weights;
use super::{Model, VideoLatent};

/// 3x3x3 conv, replicate padded in time, as the sum of three 2-D convs over the temporal taps.
fn conv3d(g: &Ctx, w: &Weights, prefix: &str, x: Tensor, causal: bool) -> Result<Tensor> {
    let [wd, h, c, t] = ffi::shape(x);
    let nb = ffi::strides(x);
    let ks = [
        w.get(&format!("{prefix}.weight.t0"))?,
        w.get(&format!("{prefix}.weight.t1"))?,
        w.get(&format!("{prefix}.weight.t2"))?,
    ];
    let mut y = if t == 1 {
        // every tap sees the same frame
        let y0 = g.conv_2d(ks[0], x, 1, 1, 1, 1, 1, 1);
        let y1 = g.conv_2d(ks[1], x, 1, 1, 1, 1, 1, 1);
        let y2 = g.conv_2d(ks[2], x, 1, 1, 1, 1, 1, 1);
        g.add(g.add(y0, y1), y2)
    } else {
        let first = g.view_4d(x, wd, h, c, 1, nb[1], nb[2], nb[3], 0);
        let last = g.view_4d(
            x,
            wd,
            h,
            c,
            1,
            nb[1],
            nb[2],
            nb[3],
            (t - 1) as usize * nb[3],
        );
        let xp = if causal {
            g.concat(g.concat(first, first, 3), x, 3)
        } else {
            g.concat(g.concat(first, x, 3), last, 3)
        };
        let pnb = ffi::strides(xp);
        let mut y: Option<Tensor> = None;
        for (k, kt) in ks.iter().enumerate() {
            let xv = g.view_4d(xp, wd, h, c, t, pnb[1], pnb[2], pnb[3], k * pnb[3]);
            let yk = g.conv_2d(*kt, xv, 1, 1, 1, 1, 1, 1);
            y = Some(match y {
                Some(acc) => g.add(acc, yk),
                None => yk,
            });
        }
        y.unwrap()
    };
    if let Some(b) = w.opt(&format!("{prefix}.bias")) {
        y = g.add(y, g.reshape_4d(b, 1, 1, ffi::shape(b)[0], 1));
    }
    Ok(y)
}

/// Normalize over the channel dim.
fn pixel_norm(g: &Ctx, x: Tensor) -> Tensor {
    let p = g.cont(g.permute(x, 1, 2, 0, 3)); // [C, W, H, T]
    let p = g.rms_norm(p, 1e-8);
    g.cont(g.permute(p, 2, 0, 1, 3))
}

fn resblock(g: &Ctx, w: &Weights, prefix: &str, x: Tensor, causal: bool) -> Result<Tensor> {
    let h = g.silu(pixel_norm(g, x));
    let h = conv3d(g, w, &format!("{prefix}.conv1.conv"), h, causal)?;
    let h = g.silu(pixel_norm(g, h));
    let h = conv3d(g, w, &format!("{prefix}.conv2.conv"), h, causal)?;
    Ok(g.add(x, h))
}

/// Innermost channel factor `p` into the width: [W, H, p*R, T] -> [p*W, H, R, T]
fn shuffle_w(g: &Ctx, y: Tensor, p: i64) -> Tensor {
    let [w, h, c, t] = ffi::shape(y);
    let v = g.reshape_4d(y, w * h, p, c / p, t);
    let v = g.cont(g.permute(v, 1, 0, 2, 3));
    g.reshape_4d(v, p * w, h, c / p, t)
}

/// Innermost channel factor `p` into the height: [W, H, p*R, T] -> [W, p*H, R, T]
fn shuffle_h(g: &Ctx, y: Tensor, p: i64) -> Tensor {
    let [w, h, c, t] = ffi::shape(y);
    let v = g.reshape_4d(y, w, h, p, (c / p) * t);
    let v = g.cont(g.permute(v, 0, 2, 1, 3));
    g.reshape_4d(v, w, p * h, c / p, t)
}

/// Innermost channel factor `p` into time: [W, H, p*R, T] -> [W, H, R, p*T]
fn shuffle_t(g: &Ctx, y: Tensor, p: i64) -> Tensor {
    let [w, h, c, t] = ffi::shape(y);
    let v = g.reshape_4d(y, w * h, p, c / p, t);
    let v = g.cont(g.permute(v, 0, 2, 1, 3));
    g.reshape_4d(v, w, h, c / p, p * t)
}

/// DepthToSpaceUpsample: conv, pixel shuffle (pt, ps, ps), drop the first frame when pt == 2.
fn upsample(
    g: &Ctx,
    w: &Weights,
    prefix: &str,
    x: Tensor,
    pt: i64,
    ps: i64,
    causal: bool,
) -> Result<Tensor> {
    let mut y = conv3d(g, w, &format!("{prefix}.conv.conv"), x, causal)?;
    // channel index = c*(pt*ps*ps) + t*(ps*ps) + h*ps + w
    if ps > 1 {
        y = shuffle_w(g, y, ps);
        y = shuffle_h(g, y, ps);
    }
    if pt > 1 {
        y = shuffle_t(g, y, pt);
        let [wd, h, c, t] = ffi::shape(y);
        let nb = ffi::strides(y);
        y = g.cont(g.view_4d(y, wd, h, c, t - 1, nb[1], nb[2], nb[3], nb[3]));
    }
    Ok(y)
}

enum Block {
    Res(i64),
    CompressAll,
    CompressTime,
    CompressSpace,
}

/// LTX-2.x decoder blocks (the config lists them in encoder order).
const BLOCKS: [Block; 9] = [
    Block::Res(2),
    Block::CompressAll,
    Block::Res(2),
    Block::CompressAll,
    Block::Res(4),
    Block::CompressTime,
    Block::Res(6),
    Block::CompressSpace,
    Block::Res(4),
];

/// Decodes latents to RGB floats in [-1, 1], frame-major `[f][h][w][3]`; returns (frames, height, width, rgb).
pub fn decode(model: &Model, be: &Backend, lat: &VideoLatent) -> Result<(i64, i64, i64, Vec<f32>)> {
    let hp = &model.hp;
    let w = model
        .vae
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no video VAE loaded"))?;
    let (f, hl, wl, c) = (lat.n_frames, lat.height, lat.width, lat.channels);
    let causal = false;

    let mean = w.read_f32("per_channel_statistics.mean-of-means")?;
    let std = w.read_f32("per_channel_statistics.std-of-means")?;
    if mean.len() < c as usize || std.len() < c as usize {
        bail!("latent statistics have fewer than {c} channels");
    }
    // denormalize, lay out as [W, H, C, T]
    let mut x = vec![0f32; (wl * hl * c * f) as usize];
    for fi in 0..f {
        for h in 0..hl {
            for wi in 0..wl {
                let src = &lat.x[((fi * hl * wl + h * wl + wi) * c) as usize..][..c as usize];
                for ci in 0..c {
                    x[(wi + wl * (h + hl * (ci + c * fi))) as usize] =
                        src[ci as usize] * std[ci as usize] + mean[ci as usize];
                }
            }
        }
    }

    let mut g = Graph::new(be.api, 8192)?;
    let mut cur = g.input_f32(&[wl, hl, c, f], &x, "latent");
    let ctx = &g.ctx;
    cur = conv3d(ctx, w, "decoder.conv_in.conv", cur, causal)?;
    for (i, b) in BLOCKS.iter().enumerate() {
        let bp = format!("decoder.up_blocks.{i}");
        cur = match b {
            Block::Res(n) => {
                for l in 0..*n {
                    cur = resblock(ctx, w, &format!("{bp}.res_blocks.{l}"), cur, causal)?;
                }
                cur
            }
            Block::CompressAll => upsample(ctx, w, &bp, cur, 2, 2, causal)?,
            Block::CompressTime => upsample(ctx, w, &bp, cur, 2, 1, causal)?,
            Block::CompressSpace => upsample(ctx, w, &bp, cur, 1, 2, causal)?,
        };
    }
    cur = ctx.silu(pixel_norm(ctx, cur));
    cur = conv3d(ctx, w, "decoder.conv_out.conv", cur, causal)?;
    // unpatchify: channel = c*16 + r*4 + q, q -> height, r -> width
    let p = hp.vae_patch_size;
    cur = shuffle_h(ctx, cur, p);
    cur = shuffle_w(ctx, cur, p);
    g.mark_output(cur);
    g.compute(be)?;

    let [ow, oh, oc, of] = ffi::shape(cur);
    if oc != 3 {
        bail!("decoder produced {oc} channels");
    }
    let y = g.output_f32(cur);
    let (wu, hu) = (ow as usize, oh as usize);
    let mut rgb = vec![0f32; y.len()];
    for fi in 0..of as usize {
        for h in 0..hu {
            for wi in 0..wu {
                for ci in 0..3 {
                    rgb[((fi * hu + h) * wu + wi) * 3 + ci] = y[wi + wu * (h + hu * (ci + 3 * fi))];
                }
            }
        }
    }
    Ok((of, oh, ow, rgb))
}

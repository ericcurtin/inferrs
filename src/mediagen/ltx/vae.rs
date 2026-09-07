//! The causal video VAE decoder. Activations are `[W, H, C, T]`.
//!
//! Long clips are decoded in overlapping tiles blended with linear ramps,
//! after `ltx_core.tiling` and `ConvVideoDecoder.tiled_decode`, so memory
//! scales with the tile rather than the clip.

use anyhow::{bail, Result};

use super::super::backend::{Backend, Ctx, Graph};
use super::super::ffi::{self, Tensor};
use super::super::weights::Weights;
use super::{Hparams, Model, VideoLatent};

/// Bound on the im2col scratch of one 2-D conv (`[9 * C, W, H, frames]`
/// halves); frames are convolved in chunks that stay under it.
const IM2COL_BUDGET: usize = 256 << 20;

/// Frames per conv chunk for an activation of this shape.
fn conv_chunk(wd: i64, h: i64, c: i64, t: i64) -> i64 {
    let per_frame = (wd * h * 9 * c) as usize * 2;
    (IM2COL_BUDGET / per_frame.max(1)).clamp(1, t.max(1) as usize) as i64
}

/// Output frame ranges `[start, end)` convolved at once: interior frames in
/// groups of `chunk`, the frames reading a replicated edge frame alone.
fn conv_chunks(t: i64, causal: bool, chunk: i64) -> Vec<(i64, i64)> {
    // output frame o, tap k reads input frame clamp(o + k - off, 0, t - 1)
    let off = if causal { 2 } else { 1 };
    let (lo, hi) = (off, t - 2 + off);
    let mut ranges = Vec::new();
    for o in 0..lo.min(t) {
        ranges.push((o, o + 1));
    }
    let mut s = lo;
    while s < hi {
        let e = (s + chunk).min(hi);
        ranges.push((s, e));
        s = e;
    }
    for o in hi.max(lo)..t {
        ranges.push((o, o + 1));
    }
    ranges
}

/// Graph nodes one `conv3d` emits: per chunk 3 taps of a view and 7
/// `ggml_conv_2d` nodes, 2 adds and a concat (none for the first), then the
/// bias' reshape and add.
fn conv3d_nodes(wd: i64, h: i64, c: i64, t: i64) -> usize {
    27 * conv_chunks(t, false, conv_chunk(wd, h, c, t)).len() + 1
}

/// 3x3x3 conv, replicate padded in time, as the sum of three 2-D convs over the temporal taps.
fn conv3d(g: &Ctx, w: &Weights, prefix: &str, x: Tensor, causal: bool) -> Result<Tensor> {
    let [wd, h, c, t] = ffi::shape(x);
    let nb = ffi::strides(x);
    let ks = [
        w.get(&format!("{prefix}.weight.t0"))?,
        w.get(&format!("{prefix}.weight.t1"))?,
        w.get(&format!("{prefix}.weight.t2"))?,
    ];
    let off = if causal { 2 } else { 1 };
    let mut y: Option<Tensor> = None;
    for (a, b) in conv_chunks(t, causal, conv_chunk(wd, h, c, t)) {
        let mut yc: Option<Tensor> = None;
        for (k, kt) in ks.iter().enumerate() {
            // in range by construction except for the single edge frames
            let f0 = (a + k as i64 - off).clamp(0, t - 1);
            let xv = g.view_4d(x, wd, h, c, b - a, nb[1], nb[2], nb[3], f0 as usize * nb[3]);
            let yk = g.conv_2d(*kt, xv, 1, 1, 1, 1, 1, 1);
            yc = Some(yc.map_or(yk, |acc| g.add(acc, yk)));
        }
        let yc = yc.unwrap();
        y = Some(y.map_or(yc, |acc| g.concat(acc, yc, 3)));
    }
    let mut y = y.unwrap();
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
    /// Upsample by (time, space).
    Up(i64, i64),
}

/// LTX-2.x decoder blocks (the config lists them in encoder order).
const BLOCKS: [Block; 9] = [
    Block::Res(2),
    Block::Up(2, 2),
    Block::Res(2),
    Block::Up(2, 2),
    Block::Res(4),
    Block::Up(2, 1),
    Block::Res(6),
    Block::Up(1, 2),
    Block::Res(4),
];

/// Nodes `build` emits for this latent, to size its graph (which cannot
/// grow): the same walk over the blocks, with each conv's output channels
/// read off its weight.
fn count_nodes(w: &Weights, lat: &VideoLatent) -> Result<usize> {
    let (mut wd, mut h, mut c, mut t) = (lat.width, lat.height, lat.channels, lat.n_frames);
    let mut n = 1; // the input
    let conv = |n: &mut usize, prefix: &str, wd, h, c, t| -> Result<i64> {
        *n += conv3d_nodes(wd, h, c, t);
        Ok(ffi::shape(w.get(&format!("{prefix}.weight.t0"))?)[3])
    };
    c = conv(&mut n, "decoder.conv_in.conv", wd, h, c, t)?;
    for (i, b) in BLOCKS.iter().enumerate() {
        let bp = format!("decoder.up_blocks.{i}");
        match b {
            Block::Res(nb) => {
                for l in 0..*nb {
                    n += 13; // norms, silus, residual add
                    let rp = format!("{bp}.res_blocks.{l}");
                    c = conv(&mut n, &format!("{rp}.conv1.conv"), wd, h, c, t)?;
                    c = conv(&mut n, &format!("{rp}.conv2.conv"), wd, h, c, t)?;
                }
            }
            Block::Up(pt, ps) => {
                c = conv(&mut n, &format!("{bp}.conv.conv"), wd, h, c, t)? / (pt * ps * ps);
                n += 14; // shuffles and the frame drop
                (wd, h) = (wd * ps, h * ps);
                if *pt == 2 {
                    t = 2 * t - 1;
                }
            }
        }
    }
    n += 6; // norm, silu
    conv(&mut n, "decoder.conv_out.conv", wd, h, c, t)?;
    Ok(n + 8) // unpatchify
}

/// Builds the decoder graph for one (tile of) latent, returning its output
/// tensor `[W, H, 3, T]`.
fn build(model: &Model, be: &Backend, lat: &VideoLatent) -> Result<(Graph, Tensor)> {
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

    let nodes = count_nodes(w, lat)?;
    if nodes + 1024 > be.max_nodes() {
        bail!("the decoder graph would have {nodes} nodes, more than the scheduler can take");
    }
    let mut g = Graph::new(be.api, nodes + 1024)?;
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
            Block::Up(pt, ps) => upsample(ctx, w, &bp, cur, *pt, *ps, causal)?,
        };
    }
    cur = ctx.silu(pixel_norm(ctx, cur));
    cur = conv3d(ctx, w, "decoder.conv_out.conv", cur, causal)?;
    // unpatchify: channel = c*16 + r*4 + q, q -> height, r -> width
    let p = hp.vae_patch_size;
    cur = shuffle_h(ctx, cur, p);
    cur = shuffle_w(ctx, cur, p);
    g.mark_output(cur);
    Ok((g, cur))
}

/// Decodes one (tile of) latent to RGB floats in [-1, 1], frame-major `[f][h][w][3]`.
fn decode_tile(model: &Model, be: &Backend, lat: &VideoLatent) -> Result<Vec<f32>> {
    let (g, cur) = build(model, be, lat)?;
    g.compute(be)?;
    let [ow, oh, oc, of] = ffi::shape(cur);
    let (want_f, want_h, want_w) = output_dims(&model.hp, lat);
    if (of, oh, ow, oc) != (want_f, want_h, want_w, 3) {
        bail!("decoder produced {of} frames of {ow}x{oh}x{oc}, expected {want_f} of {want_w}x{want_h}x3");
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
    Ok(rgb)
}

/// (frames, height, width) this latent decodes to.
fn output_dims(hp: &Hparams, lat: &VideoLatent) -> (i64, i64, i64) {
    (
        (lat.n_frames - 1) * hp.vae_scale_t + 1,
        lat.height * hp.vae_scale_s,
        lat.width * hp.vae_scale_s,
    )
}

// ---------------------------------------------------------------------------
// Tiling
// ---------------------------------------------------------------------------

/// Latent frames shared by consecutive temporal tiles (24 frames).
const OVERLAP_T: i64 = 3;
/// Latent rows/columns shared by consecutive spatial tiles (64 px).
const OVERLAP_S: i64 = 2;
/// Temporal tile sizes to try, largest first (the reference default is 80 frames).
const TILE_T: [i64; 3] = [10, 8, 6];
/// Spatial tile sizes to try, largest first (the reference default is 768 px).
const TILE_S: [i64; 3] = [24, 16, 8];

/// One tile along one latent axis and the ramps it blends over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Interval {
    start: i64,
    end: i64,
    left_ramp: i64,
    right_ramp: i64,
}

/// `ltx_core.tiling.split_by_size`: tiles of `size` sharing `overlap`, the
/// last one shorter but longer than the overlap.
fn split_by_size(dim: i64, size: i64, overlap: i64) -> Vec<Interval> {
    if dim <= size {
        return vec![Interval {
            start: 0,
            end: dim,
            left_ramp: 0,
            right_ramp: 0,
        }];
    }
    let stride = size - overlap;
    let amount = (dim + size - 2 * overlap - 1) / stride;
    (0..amount)
        .map(|i| Interval {
            start: i * stride,
            end: if i + 1 == amount {
                dim
            } else {
                i * stride + size
            },
            left_ramp: if i == 0 { 0 } else { overlap },
            right_ramp: if i + 1 == amount { 0 } else { overlap },
        })
        .collect()
}

/// `split_temporal_causal`: tiles after the first start a latent frame
/// earlier. The decoder treats a tile's first latent frame as a clip start
/// (one output frame instead of eight); that frame is ramped away.
fn split_temporal_causal(dim: i64, size: i64, overlap: i64) -> Vec<Interval> {
    let mut ivs = split_by_size(dim, size, overlap);
    for iv in ivs.iter_mut().skip(1) {
        iv.start -= 1;
        iv.left_ramp += 1;
    }
    ivs
}

/// `compute_trapezoidal_mask_1d`: ones fading in over `left` and out over
/// `right`; the fade-in starts at 0 when `left_from_zero`, else just above.
fn trapezoid(len: i64, left: i64, right: i64, left_from_zero: bool) -> Vec<f32> {
    let (len, left, right) = (
        len as usize,
        left.clamp(0, len) as usize,
        right.clamp(0, len) as usize,
    );
    let mut m = vec![1f32; len];
    let (num0, den) = if left_from_zero {
        (0, left)
    } else {
        (1, left + 1)
    };
    for (i, v) in m.iter_mut().take(left).enumerate() {
        *v *= (i + num0) as f32 / den as f32;
    }
    for (i, v) in m.iter_mut().skip(len - right).enumerate() {
        *v *= 1.0 - (i + 1) as f32 / (right + 1) as f32;
    }
    m
}

/// `map_temporal_slice`: output start frame and blend mask of a temporal tile.
fn map_temporal(iv: &Interval, scale: i64) -> (i64, Vec<f32>) {
    let (start, stop) = (iv.start * scale, 1 + (iv.end - 1) * scale);
    let left = if iv.left_ramp == 0 {
        0
    } else {
        1 + (iv.left_ramp - 1) * scale
    };
    (
        start,
        trapezoid(stop - start, left, iv.right_ramp * scale, true),
    )
}

/// `map_spatial_slice`: output start pixel and blend mask of a spatial tile.
fn map_spatial(iv: &Interval, scale: i64) -> (i64, Vec<f32>) {
    let (start, stop) = (iv.start * scale, iv.end * scale);
    let mask = trapezoid(
        stop - start,
        iv.left_ramp * scale,
        iv.right_ramp * scale,
        false,
    );
    (start, mask)
}

/// Tile sizes in latent units; an axis whose size covers the latent is untiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Plan {
    t: i64,
    s: i64,
}

impl Plan {
    fn intervals(&self, lat: &VideoLatent) -> [Vec<Interval>; 3] {
        [
            split_temporal_causal(lat.n_frames, self.t, OVERLAP_T),
            split_by_size(lat.height, self.s, OVERLAP_S),
            split_by_size(lat.width, self.s, OVERLAP_S),
        ]
    }

    /// The biggest tile of this plan, for measuring.
    fn largest_tile(&self, lat: &VideoLatent) -> VideoLatent {
        let [t, h, w] = self
            .intervals(lat)
            .map(|ivs| ivs.iter().map(|iv| iv.end - iv.start).max().unwrap_or(0));
        VideoLatent {
            n_frames: t,
            height: h,
            width: w,
            channels: lat.channels,
            x: vec![0.0; (t * h * w * lat.channels) as usize],
        }
    }
}

/// Plans from the whole clip in one tile down to the smallest tiles, by
/// decreasing decoded tile size so a retry always needs less.
fn plans(hp: &Hparams, lat: &VideoLatent) -> Vec<Plan> {
    let full_s = lat.height.max(lat.width);
    let ts = std::iter::once(lat.n_frames).chain(TILE_T.into_iter().filter(|&t| t < lat.n_frames));
    let ss: Vec<i64> = std::iter::once(full_s)
        .chain(TILE_S.into_iter().filter(|&s| s < full_s))
        .collect();
    let mut out: Vec<Plan> = ts
        .flat_map(|t| ss.iter().map(move |&s| Plan { t, s }))
        .collect();
    let volume = |p: &Plan| {
        let (f, h, w) = output_dims(hp, &p.largest_tile(lat));
        f * h * w
    };
    out.sort_by_key(|p| std::cmp::Reverse(volume(p)));
    out
}

/// `MEDIAGEN_VAE_TILE=<frames>,<size>` (latent units) forces a plan, to
/// exercise tiling on a GPU that would not need it.
fn forced_plan() -> Result<Option<Plan>> {
    let Ok(v) = std::env::var("MEDIAGEN_VAE_TILE") else {
        return Ok(None);
    };
    let plan = v
        .split_once(',')
        .and_then(|(t, s)| {
            Some(Plan {
                t: t.trim().parse().ok()?,
                s: s.trim().parse().ok()?,
            })
        })
        .filter(|p| p.t > OVERLAP_T && p.s > OVERLAP_S);
    match plan {
        Some(p) => Ok(Some(p)),
        None => bail!(
            "MEDIAGEN_VAE_TILE={v:?}: want <frames>,<size> in latent units, above the overlaps {OVERLAP_T} and {OVERLAP_S}"
        ),
    }
}

fn sub_latent(lat: &VideoLatent, t: &Interval, h: &Interval, w: &Interval) -> VideoLatent {
    let c = lat.channels;
    let (nt, nh, nw) = (t.end - t.start, h.end - h.start, w.end - w.start);
    let mut x = Vec::with_capacity((nt * nh * nw * c) as usize);
    for fi in t.start..t.end {
        for hi in h.start..h.end {
            let row = ((fi * lat.height + hi) * lat.width + w.start) * c;
            x.extend_from_slice(&lat.x[row as usize..(row + nw * c) as usize]);
        }
    }
    VideoLatent {
        n_frames: nt,
        height: nh,
        width: nw,
        channels: c,
        x,
    }
}

/// Decodes the latent tile by tile, blending the overlaps.
fn decode_tiled(model: &Model, be: &Backend, lat: &VideoLatent, plan: &Plan) -> Result<Vec<f32>> {
    let hp = &model.hp;
    let (of, oh, ow) = output_dims(hp, lat);
    let [ts, hs, ws] = plan.intervals(lat);
    if ts.len() * hs.len() * ws.len() == 1 {
        return decode_tile(model, be, lat);
    }
    let (ohu, owu) = (oh as usize, ow as usize);
    let mut acc = vec![0f32; (of * oh * ow * 3) as usize];
    let mut wsum = vec![0f32; (of * oh * ow) as usize];
    for t in &ts {
        let (t0, mt) = map_temporal(t, hp.vae_scale_t);
        for h in &hs {
            let (h0, mh) = map_spatial(h, hp.vae_scale_s);
            for w in &ws {
                let (w0, mw) = map_spatial(w, hp.vae_scale_s);
                let rgb = decode_tile(model, be, &sub_latent(lat, t, h, w))?;
                let (nhu, nwu) = (mh.len(), mw.len());
                for (fi, &wt) in mt.iter().enumerate() {
                    for (hi, &wh) in mh.iter().enumerate() {
                        let src = &rgb[((fi * nhu + hi) * nwu) * 3..][..nwu * 3];
                        let o = ((t0 as usize + fi) * ohu + h0 as usize + hi) * owu + w0 as usize;
                        let dst = &mut acc[o * 3..][..nwu * 3];
                        let dw = &mut wsum[o..][..nwu];
                        for (xi, &ww) in mw.iter().enumerate() {
                            let m = wt * wh * ww;
                            dw[xi] += m;
                            for ci in 0..3 {
                                dst[xi * 3 + ci] += src[xi * 3 + ci] * m;
                            }
                        }
                    }
                }
            }
        }
    }
    for (i, v) in acc.iter_mut().enumerate() {
        *v /= wsum[i / 3].max(1e-8);
    }
    Ok(acc)
}

/// Decodes latents to RGB floats in [-1, 1], frame-major `[f][h][w][3]`;
/// returns (frames, height, width, rgb).
///
/// Tiles the decode when the whole clip's graph would not fit the GPU's
/// free memory or the scheduler; a tile that still fails is retried with
/// the next smaller plan.
pub fn decode(
    model: &Model,
    be: &mut Backend,
    lat: &VideoLatent,
) -> Result<(i64, i64, i64, Vec<f32>)> {
    let (of, oh, ow) = output_dims(&model.hp, lat);
    let mut plans = plans(&model.hp, lat);
    let mut i = 0;
    let mut note = String::new();
    if let Some(p) = forced_plan()? {
        plans = vec![p];
    } else if let Some((free, total)) = be.gpu_memory() {
        // slack for the scheduler's copies and the backend's own buffers
        let budget = free.saturating_sub(total / 20 + (128 << 20));
        for (j, plan) in plans.iter().enumerate() {
            i = j;
            // too many nodes: try the next plan, the last one reports the error
            let Ok((g, _)) = build(model, be, &plan.largest_tile(lat)) else {
                continue;
            };
            let need = g.measure(be)?;
            note = format!(" ({} MiB of {} MiB free)", need >> 20, free >> 20);
            if need <= budget {
                break;
            }
        }
    }
    let n_tiles: usize = plans[i].intervals(lat).iter().map(Vec::len).product();
    eprintln!("[llmman] mediagen: decoding {of} frame(s) of {ow}x{oh} in {n_tiles} tile(s){note}");
    loop {
        match decode_tiled(model, be, lat, &plans[i]) {
            Ok(rgb) => return Ok((of, oh, ow, rgb)),
            Err(e) if i + 1 < plans.len() => {
                eprintln!("[llmman] mediagen: decoding failed ({e}), retrying with smaller tiles");
                be.release_compute()?;
                i += 1;
            }
            Err(e) => return Err(e),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Every output frame exactly once, in order, and every tap of a
    /// multi-frame chunk stays inside the clip.
    fn check(t: i64, causal: bool, chunk: i64) -> Vec<(i64, i64)> {
        let ranges = conv_chunks(t, causal, chunk);
        let off = if causal { 2 } else { 1 };
        let mut next = 0;
        for &(a, b) in &ranges {
            assert_eq!(a, next, "t={t} causal={causal} chunk={chunk}: {ranges:?}");
            assert!(b > a && b - a <= chunk);
            if b - a > 1 {
                // first tap of the first frame, last tap of the last frame
                assert!(a - off >= 0 && b + 1 - off < t, "{ranges:?}");
            }
            next = b;
        }
        assert_eq!(next, t, "t={t} causal={causal} chunk={chunk}: {ranges:?}");
        ranges
    }

    #[test]
    fn chunks_cover_the_clip_exactly() {
        for t in 1..=45 {
            for chunk in 1..=12 {
                check(t, false, chunk);
                check(t, true, chunk);
            }
        }
    }

    #[test]
    fn edge_frames_go_alone() {
        assert_eq!(check(1, false, 8), vec![(0, 1)]);
        assert_eq!(check(2, false, 8), vec![(0, 1), (1, 2)]);
        assert_eq!(check(3, false, 8), vec![(0, 1), (1, 2), (2, 3)]);
        assert_eq!(
            check(11, false, 4),
            vec![(0, 1), (1, 5), (5, 9), (9, 10), (10, 11)]
        );
        // causal: the first two frames read the replicated first frame
        assert_eq!(check(6, true, 8), vec![(0, 1), (1, 2), (2, 6)]);
    }

    /// Blend weights of tiles along one axis, mapped to output units.
    fn summed(ivs: &[Interval], scale: i64, temporal: bool) -> Vec<f32> {
        let len = if temporal {
            (ivs.last().unwrap().end - 1) * scale + 1
        } else {
            ivs.last().unwrap().end * scale
        };
        let mut w = vec![0f32; len as usize];
        for iv in ivs {
            let (start, m) = if temporal {
                map_temporal(iv, scale)
            } else {
                map_spatial(iv, scale)
            };
            for (i, v) in m.iter().enumerate() {
                w[start as usize + i] += v;
            }
        }
        w
    }

    #[test]
    fn untiled_when_the_tile_covers_the_axis() {
        assert_eq!(
            split_by_size(6, 10, 3),
            vec![Interval {
                start: 0,
                end: 6,
                left_ramp: 0,
                right_ramp: 0
            }]
        );
        assert_eq!(split_temporal_causal(10, 10, 3).len(), 1);
        assert_eq!(
            map_temporal(&split_temporal_causal(6, 10, 3)[0], 8).1,
            vec![1.0; 41]
        );
    }

    #[test]
    fn split_by_size_matches_the_reference() {
        // ltx_core.tiling.split_by_size(10, 3)(16)
        assert_eq!(
            split_by_size(16, 10, 3),
            vec![
                Interval {
                    start: 0,
                    end: 10,
                    left_ramp: 0,
                    right_ramp: 3
                },
                Interval {
                    start: 7,
                    end: 16,
                    left_ramp: 3,
                    right_ramp: 0
                },
            ]
        );
        // the last tile is always longer than the overlap
        for dim in 11..80 {
            for size in 6..=12 {
                let ivs = split_by_size(dim, size, 3);
                assert_eq!(ivs[0].start, 0);
                assert_eq!(ivs.last().unwrap().end, dim);
                for (a, b) in ivs.iter().zip(ivs.iter().skip(1)) {
                    assert_eq!(a.end - b.start, 3, "{ivs:?}");
                    assert!(b.end - b.start > 3, "{ivs:?}");
                }
            }
        }
    }

    #[test]
    fn causal_temporal_tiles_shift_back_one_frame() {
        assert_eq!(
            split_temporal_causal(16, 10, 3),
            vec![
                Interval {
                    start: 0,
                    end: 10,
                    left_ramp: 0,
                    right_ramp: 3
                },
                Interval {
                    start: 6,
                    end: 16,
                    left_ramp: 4,
                    right_ramp: 0
                },
            ]
        );
        // the shifted tile's first output frame is blended away entirely
        let (start, m) = map_temporal(&split_temporal_causal(16, 10, 3)[1], 8);
        assert_eq!(start, 48);
        assert_eq!(m.len(), 73);
        assert_eq!(m[0], 0.0);
        assert!(m[24] < 1.0 && m[25] == 1.0);
    }

    #[test]
    fn trapezoid_matches_the_reference_masks() {
        // compute_trapezoidal_mask_1d(6, 2, 2, False)
        let m = trapezoid(6, 2, 2, false);
        let want = [1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 2.0 / 3.0, 1.0 / 3.0];
        for (a, b) in m.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{m:?}");
        }
        // compute_trapezoidal_mask_1d(5, 2, 0, True)
        assert_eq!(trapezoid(5, 2, 0, true), vec![0.0, 0.5, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn blend_weights_sum_to_one_everywhere() {
        for dim in [7, 11, 16, 23, 31, 64] {
            for tile in TILE_T {
                let w = summed(&split_temporal_causal(dim, tile, OVERLAP_T), 8, true);
                assert_eq!(w.len(), ((dim - 1) * 8 + 1) as usize);
                assert!(
                    w.iter().all(|v| (v - 1.0).abs() < 1e-5),
                    "t dim={dim} tile={tile}: {w:?}"
                );
            }
            for tile in TILE_S {
                let w = summed(&split_by_size(dim, tile, OVERLAP_S), 32, false);
                assert_eq!(w.len(), (dim * 32) as usize);
                assert!(
                    w.iter().all(|v| (v - 1.0).abs() < 1e-5),
                    "s dim={dim} tile={tile}: {w:?}"
                );
            }
        }
    }

    #[test]
    fn plans_go_from_one_tile_to_the_smallest() {
        let lat = VideoLatent {
            n_frames: 16,
            height: 16,
            width: 24,
            channels: 1,
            x: vec![0.0; 16 * 16 * 24],
        };
        let hp = Hparams::default();
        let ps = plans(&hp, &lat);
        assert_eq!(ps[0], Plan { t: 16, s: 24 });
        assert_eq!(*ps.last().unwrap(), Plan { t: 6, s: 8 });
        assert_eq!(ps.len(), 4 * 3);
        // every retry decodes fewer pixels than the attempt before it
        let volumes: Vec<i64> = ps
            .iter()
            .map(|p| {
                let (f, h, w) = output_dims(&hp, &p.largest_tile(&lat));
                f * h * w
            })
            .collect();
        assert!(volumes.windows(2).all(|w| w[0] >= w[1]), "{volumes:?}");
        let tile = Plan { t: 10, s: 24 }.largest_tile(&lat);
        assert_eq!((tile.n_frames, tile.height, tile.width), (10, 16, 24));
    }

    #[test]
    fn sub_latent_cuts_the_block() {
        let (f, h, w, c) = (3, 2, 4, 2);
        let x: Vec<f32> = (0..f * h * w * c).map(|v| v as f32).collect();
        let lat = VideoLatent {
            n_frames: f,
            height: h,
            width: w,
            channels: c,
            x,
        };
        let iv = |start, end| Interval {
            start,
            end,
            left_ramp: 0,
            right_ramp: 0,
        };
        let sub = sub_latent(&lat, &iv(1, 3), &iv(1, 2), &iv(2, 4));
        assert_eq!((sub.n_frames, sub.height, sub.width), (2, 1, 2));
        let at = |fi, hi, wi, ci| ((fi * h + hi) * w + wi) as f32 * c as f32 + ci as f32;
        assert_eq!(
            sub.x,
            vec![
                at(1, 1, 2, 0),
                at(1, 1, 2, 1),
                at(1, 1, 3, 0),
                at(1, 1, 3, 1),
                at(2, 1, 2, 0),
                at(2, 1, 2, 1),
                at(2, 1, 3, 0),
                at(2, 1, 3, 1)
            ]
        );
    }
}

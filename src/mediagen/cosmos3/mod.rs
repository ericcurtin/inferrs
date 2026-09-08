//! NVIDIA Cosmos3 (Edge / Nano / Super) text-to-image and text-to-video on
//! ggml: the omni transformer GGUF published as `ai/cosmos3-*` plus its Wan
//! 2.2 VAE and tokenizer sidecars. See `dit` for the model, `vae` for the
//! decoder, `sched` for the UniPC sampler and `tokenizer` for the prompt.

pub mod audio;
pub mod dit;
pub mod sched;
pub mod tokenizer;
pub mod vae;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context as _, Result};
use serde_json::Value;

use super::backend::Backend;
use super::ffi::Api;
use super::weights::{Index, LoadOpts, Weights};
use super::{dump, ActionMode, Frames, GenParams, Mt19937, Output};
use dit::UndCache;
use tokenizer::{PromptSpec, Template, Tokenizer};

/// `Cosmos3OmniPipeline.__call__` defaults.
pub const DEFAULT_STEPS: i32 = 35;
pub const DEFAULT_CFG: f32 = 6.0;

/// `Cosmos3OmniTransformer` config, from the GGUF's `cosmos3.*` keys.
#[derive(Clone, Debug)]
pub struct Hparams {
    pub hidden: i64,
    pub n_layers: i64,
    pub n_heads: i64,
    pub n_kv_heads: i64,
    pub head_dim: i64,
    pub rms_eps: f32,
    pub rope_theta: f64,
    pub mrope_section: [i64; 3],
    /// `hidden_act == "relu2"` (Nemotron backbone) instead of SwiGLU.
    pub relu2: bool,
    pub qk_norm_for_text: bool,
    pub latent_channels: i64,
    pub patch: i64,
    pub patch_dim: i64,
    pub base_fps: f32,
    pub enable_fps_modulation: bool,
    pub temporal_margin: i64,
    pub timestep_scale: f32,
    pub sound_gen: bool,
    pub sound_dim: i64,
    pub sound_latent_fps: f32,
    pub action_gen: bool,
    pub action_dim: i64,
    pub vae_scale_t: i64,
    pub vae_scale_s: i64,
}

impl Hparams {
    pub fn from_metadata(meta: &std::collections::HashMap<String, String>) -> Result<Hparams> {
        let cfg: Value = serde_json::from_str(
            meta.get("cosmos3.transformer_config_json")
                .ok_or_else(|| anyhow!("GGUF has no cosmos3.transformer_config_json"))?,
        )
        .context("cosmos3.transformer_config_json")?;
        let i = |k: &str, d: i64| cfg.get(k).and_then(Value::as_i64).unwrap_or(d);
        let f = |k: &str, d: f64| cfg.get(k).and_then(Value::as_f64).unwrap_or(d);
        let b = |k: &str, d: bool| cfg.get(k).and_then(Value::as_bool).unwrap_or(d);
        let sect: Vec<i64> = cfg
            .get("rope_axes_dim")
            .or_else(|| cfg.get("rope_scaling").and_then(|r| r.get("mrope_section")))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_i64).collect())
            .unwrap_or_else(|| vec![24, 20, 20]);
        let head_dim = i("head_dim", 128);
        if sect.len() != 3 || sect.iter().sum::<i64>() != head_dim / 2 {
            bail!("mrope_section {sect:?} does not cover head_dim {head_dim} / 2");
        }
        let patch = i("latent_patch_size", 2);
        let channels = i("latent_channel", 48);
        Ok(Hparams {
            hidden: i("hidden_size", 4096),
            n_layers: i("num_hidden_layers", 36),
            n_heads: i("num_attention_heads", 32),
            n_kv_heads: i("num_key_value_heads", 8),
            head_dim,
            rms_eps: f("rms_norm_eps", 1e-6) as f32,
            rope_theta: f("rope_theta", 5_000_000.0),
            mrope_section: [sect[0], sect[1], sect[2]],
            relu2: cfg.get("hidden_act").and_then(Value::as_str) == Some("relu2"),
            qk_norm_for_text: b("qk_norm_for_text", true),
            latent_channels: channels,
            patch,
            patch_dim: i("patch_latent_dim", patch * patch * channels),
            base_fps: f("base_fps", 24.0) as f32,
            enable_fps_modulation: b("enable_fps_modulation", true),
            temporal_margin: i("unified_3d_mrope_temporal_modality_margin", 15000),
            timestep_scale: f("timestep_scale", 0.001) as f32,
            sound_gen: b("sound_gen", false),
            sound_dim: i("sound_dim", 64),
            sound_latent_fps: f("sound_latent_fps", 25.0) as f32,
            action_gen: b("action_gen", false),
            action_dim: i("action_dim", 64),
            vae_scale_t: 4,
            vae_scale_s: 16,
        })
    }

    #[cfg(test)]
    pub fn test_default() -> Hparams {
        Hparams {
            hidden: 4096,
            n_layers: 36,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            rms_eps: 1e-6,
            rope_theta: 5_000_000.0,
            mrope_section: [24, 20, 20],
            relu2: false,
            qk_norm_for_text: true,
            latent_channels: 48,
            patch: 2,
            patch_dim: 192,
            base_fps: 24.0,
            enable_fps_modulation: true,
            temporal_margin: 15000,
            timestep_scale: 0.001,
            sound_gen: false,
            sound_dim: 64,
            sound_latent_fps: 25.0,
            action_gen: false,
            action_dim: 64,
            vae_scale_t: 4,
            vae_scale_s: 16,
        }
    }
}

pub fn is_cosmos3(arch: &str) -> bool {
    arch == "cosmos3"
}

/// Everything loaded for one Cosmos3 model.
pub struct Model {
    pub hp: Hparams,
    pub dit: Weights,
    pub vae: Weights,
    /// The sound tokenizer's decoder (Nano, Super).
    pub sound: Option<Weights>,
    tok: Tokenizer,
    /// `use_native_flow_schedule` of the pipeline (Edge).
    native_flow_schedule: bool,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    flash: bool,
    /// Understanding-stream caches of the last positive / negative prompt.
    cond: Option<(Vec<u32>, UndCache)>,
    ncond: Option<(Vec<u32>, UndCache)>,
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&std::fs::read(path)?).with_context(|| path.display().to_string())
}

impl Model {
    pub fn load(
        api: &'static Api,
        be: &Backend,
        idx: &Index,
        model: &Path,
        vae: Option<&Path>,
        files: &BTreeMap<String, PathBuf>,
        flash: bool,
    ) -> Result<Model> {
        let mut hp = Hparams::from_metadata(&idx.metadata)?;
        let vae_path = vae.ok_or_else(|| anyhow!("the model pack has no VAE (role \"vae\")"))?;
        let tok_path = files
            .get("text_tokenizer/tokenizer.json")
            .or_else(|| files.get("tokenizer.json"))
            .ok_or_else(|| anyhow!("the model pack has no tokenizer.json"))?;
        let template = if hp.relu2 {
            Template::Nemotron
        } else {
            Template::Qwen
        };
        let model_index = files
            .get("model_index.json")
            .and_then(|p| read_json(p).ok());
        let use_system_prompt = model_index
            .as_ref()
            .and_then(|v| v.get("default_use_system_prompt")?.as_bool())
            .unwrap_or(true);
        let native_flow_schedule = model_index
            .as_ref()
            .and_then(|v| v.get("use_native_flow_schedule")?.as_bool())
            .unwrap_or(false);
        let tok = Tokenizer::load(tok_path, template, use_system_prompt)?;

        let (mut latents_mean, mut latents_std) = (vec![0f32; 48], vec![1f32; 48]);
        match files.get("vae/config.json").map(|p| read_json(p)) {
            Some(Ok(v)) => {
                let arr = |k: &str| -> Option<Vec<f32>> {
                    Some(
                        v.get(k)?
                            .as_array()?
                            .iter()
                            .filter_map(|x| x.as_f64().map(|x| x as f32))
                            .collect(),
                    )
                };
                if let (Some(m), Some(s)) = (arr("latents_mean"), arr("latents_std")) {
                    latents_mean = m;
                    latents_std = s;
                }
                if let Some(t) = v.get("scale_factor_temporal").and_then(Value::as_i64) {
                    hp.vae_scale_t = t;
                }
                if let Some(s) = v.get("scale_factor_spatial").and_then(Value::as_i64) {
                    hp.vae_scale_s = s;
                }
            }
            Some(Err(e)) => eprintln!("[llmman] mediagen: vae/config.json unreadable ({e}), using defaults"),
            None => eprintln!("[llmman] mediagen: no vae/config.json in the model pack, using default latent statistics"),
        }
        let nch = hp.latent_channels as usize;
        if latents_mean.len() != nch
            || latents_std.len() != nch
            || latents_std.iter().any(|&s| s <= 0.0)
        {
            bail!("vae/config.json: latents_mean/latents_std must each hold {nch} positive-std entries");
        }
        // the decoder graph is fixed to Wan 2.2's 4x temporal, 16x spatial factors
        if (hp.vae_scale_t, hp.vae_scale_s) != (4, 16) {
            bail!(
                "unsupported VAE scale factors ({}, {}), expected (4, 16)",
                hp.vae_scale_t,
                hp.vae_scale_s
            );
        }

        let buft = be.weight_buft();
        let t0 = Instant::now();
        let dit = Weights::load(api, model, buft, &LoadOpts::default())?;
        let vae_merged = vae::merged_path(
            vae_path,
            &vae_path.parent().map(Path::to_path_buf).unwrap_or_default(),
        )?;
        let vae = Weights::load(
            api,
            &vae_merged,
            buft,
            &LoadOpts {
                rename: Some(Box::new(|n| {
                    (n.starts_with("decoder.")
                        || n.starts_with("post_quant_conv.")
                        || n.starts_with("encoder.")
                        || n.starts_with("quant_conv."))
                    .then(|| n.to_string())
                })),
                transform: None,
            },
        )?;
        if !vae.has("decoder.conv_in.weight") {
            bail!("the VAE is not a Wan decoder (decoder.conv_in.weight missing)");
        }
        // the sound tokenizer: weight-norm folded once, next to the VAE's link
        let sound = match (
            hp.sound_gen,
            files.get("sound_tokenizer/diffusion_pytorch_model.safetensors"),
        ) {
            (true, Some(path)) => {
                let cache_dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
                let folded = audio::folded_path(path, &cache_dir)?;
                Some(Weights::load(
                    api,
                    &folded,
                    buft,
                    &LoadOpts {
                        rename: Some(Box::new(|n| {
                            n.starts_with("decoder.").then(|| n.to_string())
                        })),
                        transform: Some(Box::new(audio::transform)),
                    },
                )?)
            }
            _ => None,
        };
        eprintln!(
            "[llmman] mediagen: cosmos3 model loaded in {:.1} s ({} layers, hidden {}, {} template, system prompt {use_system_prompt}, sound {}, actions {})",
            t0.elapsed().as_secs_f64(),
            hp.n_layers,
            hp.hidden,
            match template {
                Template::Qwen => "qwen",
                Template::Nemotron => "nemotron",
            },
            sound.is_some(),
            hp.action_gen
        );
        Ok(Model {
            hp,
            dit,
            vae,
            sound,
            tok,
            native_flow_schedule,
            latents_mean,
            latents_std,
            flash,
            cond: None,
            ncond: None,
        })
    }

    fn und(&mut self, be: &Backend, ids: Vec<u32>, negative: bool) -> Result<()> {
        let slot = if negative {
            &mut self.ncond
        } else {
            &mut self.cond
        };
        if matches!(slot, Some((cached, _)) if *cached == ids) {
            return Ok(());
        }
        let t0 = Instant::now();
        let cache = dit::und_forward(&self.hp, &self.dit, be, &ids)?;
        eprintln!(
            "[llmman] mediagen: encoded {} prompt ({} tokens) in {:.2} s",
            if negative { "negative" } else { "positive" },
            ids.len(),
            t0.elapsed().as_secs_f64()
        );
        *slot = Some((ids, cache));
        Ok(())
    }

    /// Runs one generation; `progress(step, n_steps)` returning `false` cancels.
    pub fn generate(
        &mut self,
        be: &mut Backend,
        p: &GenParams,
        mut progress: impl FnMut(i32, i32) -> bool,
    ) -> Result<Output> {
        let hp = self.hp.clone();
        let unit = hp.vae_scale_s * hp.patch;
        let action = p.action.as_ref();
        if let Some(a) = action {
            if !hp.action_gen {
                bail!("this model has no action generation");
            }
            if a.chunk_size < 1 {
                bail!("action chunk_size must be at least 1");
            }
        }
        let gen_audio = p.gen_audio && self.sound.is_some() && p.n_frames > 1;

        // ── the conditioning frames, the canvas and the frame count ──
        let mut n_frames;
        let (w, h);
        let mut cond_rgb: Option<(Vec<f32>, i64)> = None; // [f][h][w][3] in [-1, 1], frame count
        let mut cond_latent_frames: Vec<i64> = Vec::new();
        let mut content = None; // action canvas: (content_h, content_w)
        if let Some(a) = action {
            n_frames = a.chunk_size + 1;
            let src = p
                .image
                .as_ref()
                .or(p.video.as_ref())
                .ok_or_else(|| anyhow!("an action run needs a conditioning image or video"))?;
            let (frames, ch, cw, th, tw) = action_frames(src, a.resolution_tier, n_frames)?;
            (h, w) = (th, tw);
            content = Some((ch, cw));
            cond_rgb = Some((frames, n_frames));
            let lt = (n_frames - 1) / hp.vae_scale_t + 1;
            cond_latent_frames = match a.mode {
                ActionMode::InverseDynamics => (0..lt).collect(),
                _ => vec![0],
            };
        } else {
            (w, h) = (p.width, p.height);
            if w <= 0 || h <= 0 || w % unit != 0 || h % unit != 0 {
                bail!("width and height must be positive multiples of {unit}");
            }
            n_frames = p.n_frames.max(1);
            if (n_frames - 1) % hp.vae_scale_t != 0 {
                n_frames = (n_frames - 1) / hp.vae_scale_t * hp.vae_scale_t + 1;
                eprintln!(
                    "[llmman] mediagen: n_frames must be {}k+1, using {n_frames}",
                    hp.vae_scale_t
                );
            }
            let lt = (n_frames - 1) / hp.vae_scale_t + 1;
            if let (Some(vid), true) = (&p.video, n_frames > 1) {
                // video-to-video: the clip's leading frames stay clean at the given latent
                // indexes; the causal encoder needs the pixel frames up to the last of them
                let mut idx: Vec<i64> = p
                    .condition_frames
                    .iter()
                    .copied()
                    .filter(|&i| i >= 0 && i < lt)
                    .collect();
                idx.sort_unstable();
                idx.dedup();
                if idx.is_empty() {
                    bail!("condition_frames has no latent frame below {lt}");
                }
                let n_px = idx[idx.len() - 1] * hp.vae_scale_t + 1;
                cond_rgb = Some((clip_frames(vid, w, h, n_px, p.condition_keep_last)?, n_px));
                cond_latent_frames = idx;
            } else if let (Some(img), true) = (&p.image, n_frames > 1) {
                // image-to-video: frame 0 is the image, the rest is generated; the
                // causal encoder's first latent frame only sees the first pixel frame
                cond_rgb = Some((cover_crop(img, w, h)?, 1));
                cond_latent_frames = vec![0];
            }
        }
        if (n_frames - 1) % hp.vae_scale_t != 0 {
            bail!(
                "the clip must have {}k+1 frames, got {n_frames}",
                hp.vae_scale_t
            );
        }
        let n_steps = if p.n_steps > 0 {
            p.n_steps
        } else {
            DEFAULT_STEPS
        };
        let cfg = if p.cfg_scale > 0.0 {
            p.cfg_scale
        } else if action.is_some() {
            1.0
        } else {
            DEFAULT_CFG
        };
        let use_cfg = cfg != 1.0;
        if !(p.fps.is_finite() && p.fps > 0.0) {
            bail!("fps must be positive");
        }
        let flow_shift = p.flow_shift.or(action.map(|_| ACTION_FLOW_SHIFT));
        if flow_shift.is_some_and(|f| !(f.is_finite() && f > 0.0)) {
            bail!("flow_shift must be positive");
        }

        // ── prompts ──
        let (cond_ids, uncond_ids) = match action {
            Some(a) => {
                let text = action_prompt(&p.prompt, &a.view_point, n_frames, p.fps, h, w);
                (
                    self.tok.encode_chat(&text, None)?,
                    self.tok.encode_chat(&p.negative_prompt, None)?,
                )
            }
            None => {
                let spec = |prompt: &str, negative: bool| PromptSpec {
                    prompt: prompt.to_string(),
                    negative,
                    n_frames,
                    fps: p.fps,
                    width: w,
                    height: h,
                };
                (
                    self.tok.encode(&spec(&p.prompt, false))?,
                    self.tok.encode(&spec(&p.negative_prompt, true))?,
                )
            }
        };
        self.und(be, cond_ids, false)?;
        if use_cfg {
            self.und(be, uncond_ids, true)?;
        }

        // ── latents ──
        let lt = (n_frames - 1) / hp.vae_scale_t + 1;
        let (mut lh, mut lw) = (h / hp.vae_scale_s, w / hp.vae_scale_s);
        let c = hp.latent_channels as usize;
        let seed = p.seed.unwrap_or_else(super::rand_seed);
        let mut rng = Mt19937::new(seed);
        // conditioning latents: the encoded frames, normalized
        let x0 = match &cond_rgb {
            Some((rgb, nf)) => {
                let t0 = Instant::now();
                let frames = vae::encode(&self.vae, be, rgb, *nf, h, w)?;
                be.release_compute()?;
                eprintln!(
                    "[llmman] mediagen: encoded {nf} conditioning frame(s) in {:.2} s",
                    t0.elapsed().as_secs_f64()
                );
                Some(frames)
            }
            None => None,
        };
        if let Some((ch, cw)) = content {
            // the action canvas is padded; the model works on the content region
            lh = (ch / hp.vae_scale_s).max(1);
            lw = (cw / hp.vae_scale_s).max(1);
        }
        let n_lat = c * (lt * lh * lw) as usize;
        let mut x = vec![0f32; n_lat];
        if !noise_from_env("MEDIAGEN_NOISE", &mut x)? {
            for v in x.iter_mut() {
                *v = rng.normal();
            }
        }
        dump("noise_video", &x);
        let hw = (lh * lw) as usize;
        let lat_idx = |ci: usize, t: usize, hwi: usize| (ci * lt as usize + t) * hw + hwi;
        if let Some(frames) = &x0 {
            // frames[t] is [C][H_full][W_full] in VAE space; crop to the content and normalize
            let (fh, fw) = ((h / hp.vae_scale_s) as usize, (w / hp.vae_scale_s) as usize);
            for &t in &cond_latent_frames {
                let f = &frames[t as usize];
                for ci in 0..c {
                    let (m, sd) = (self.latents_mean[ci], self.latents_std[ci]);
                    for y in 0..lh as usize {
                        for xx in 0..lw as usize {
                            x[lat_idx(ci, t as usize, y * lw as usize + xx)] =
                                (f[(ci * fh + y) * fw + xx] - m) / sd;
                        }
                    }
                }
            }
            dump("cond_latent", &x);
        }
        let grid = [
            lt,
            (lh + hp.patch - 1) / hp.patch,
            (lw + hp.patch - 1) / hp.patch,
        ];
        let n_vis = grid.iter().product::<i64>();
        let frame_tokens = (grid[1] * grid[2]) as usize;
        let vis_noisy: Option<Vec<f32>> = (!cond_latent_frames.is_empty()).then(|| {
            let mut m = vec![1f32; n_vis as usize];
            for &t in &cond_latent_frames {
                m[t as usize * frame_tokens..(t as usize + 1) * frame_tokens].fill(0.0);
            }
            m
        });

        // sound latents: all noisy
        let n_sound = if gen_audio {
            let samples = (n_frames as f64 / p.fps as f64 * audio::SAMPLE_RATE as f64) as i64;
            (samples + audio::HOP - 1) / audio::HOP
        } else {
            0
        };
        let mut sound = vec![0f32; (n_sound * hp.sound_dim) as usize];
        if !noise_from_env("MEDIAGEN_NOISE_SOUND", &mut sound)? {
            for v in sound.iter_mut() {
                *v = rng.normal();
            }
        }

        // action latents: forward dynamics conditions on the given actions, the
        // other modes denoise them; channels past the domain's width stay 0
        let (domain_id, raw_dim) = match action {
            Some(a) => (
                domain_id(&a.domain)
                    .ok_or_else(|| anyhow!("unknown action domain {:?}", a.domain))?,
                raw_action_dim(&a.domain).ok_or_else(|| {
                    anyhow!("action domain {:?} has no canonical action width", a.domain)
                })?,
            ),
            None => (0, 0),
        };
        let n_act = action.map_or(0, |a| a.chunk_size);
        if action.is_some() && hp.action_dim < raw_dim {
            bail!(
                "the model's action_dim {} is below the domain's width {raw_dim}",
                hp.action_dim
            );
        }
        let ad = hp.action_dim as usize;
        let mut act = vec![0f32; (n_act as usize) * ad];
        let act_noisy: Option<Vec<f32>> = match action {
            Some(a) if a.mode == ActionMode::ForwardDynamics => {
                if a.actions.is_empty() {
                    bail!("forward dynamics needs the actions");
                }
                for t in 0..n_act as usize {
                    let row = &a.actions[t.min(a.actions.len() - 1)];
                    if row.len() != raw_dim as usize {
                        bail!(
                            "action {t} has {} values, domain {:?} takes {raw_dim}",
                            row.len(),
                            a.domain
                        );
                    }
                    act[t * ad..t * ad + row.len()].copy_from_slice(row);
                }
                Some(vec![0.0; n_act as usize])
            }
            Some(_) => {
                if noise_from_env("MEDIAGEN_NOISE_ACTION", &mut act)? {
                    for t in 0..n_act as usize {
                        act[t * ad + raw_dim as usize..(t + 1) * ad].fill(0.0);
                    }
                } else {
                    for t in 0..n_act as usize {
                        for d in 0..raw_dim as usize {
                            act[t * ad + d] = rng.normal();
                        }
                    }
                }
                None
            }
            None => None,
        };

        // ── positions ──
        let n_text = |negative: bool| -> i64 {
            let slot = if negative { &self.ncond } else { &self.cond };
            slot.as_ref().map_or(0, |(ids, _)| ids.len() as i64)
        };
        let tables = |negative: bool| {
            let nt = n_text(negative);
            let vis = dit::mrope_tables(&hp, &dit::vision_positions(&hp, nt, grid, p.fps));
            let snd = dit::mrope_tables(
                &hp,
                &dit::sound_positions(&hp, nt, n_sound, hp.sound_latent_fps),
            );
            let act = dit::mrope_tables(&hp, &dit::action_positions(&hp, nt, n_act, p.fps));
            (vis, snd, act)
        };
        let cond_tables = tables(false);
        let uncond_tables = use_cfg.then(|| tables(true));

        // ── sampling ──
        let (sigmas, timesteps) = match flow_shift {
            Some(shift) => sched::schedule_flow(n_steps as usize, shift, self.native_flow_schedule),
            None => sched::schedule(n_steps as usize),
        };
        let mut solver_v = sched::UniPC::new(sigmas.clone());
        let mut solver_s = sched::UniPC::new(sigmas.clone());
        let mut solver_a = sched::UniPC::new(sigmas);
        eprintln!(
            "[llmman] mediagen: generating {}x{}, {n_frames} frame(s), {n_vis} vision tokens{}{}, {n_steps} steps, cfg {cfg:.1}, seed {seed}{}",
            lw * hp.vae_scale_s,
            lh * hp.vae_scale_s,
            if n_sound > 0 { format!(", {n_sound} sound tokens") } else { String::new() },
            if n_act > 0 { format!(", {n_act} action tokens ({})", action.unwrap().mode.name()) } else { String::new() },
            flow_shift.map_or(String::new(), |s| format!(", flow shift {s}"))
        );
        let cond = &self.cond.as_ref().expect("positive prompt encoded").1;
        let ncond = use_cfg.then(|| &self.ncond.as_ref().expect("negative prompt encoded").1);
        let action_noisy = act_noisy.as_deref();
        let has_noisy_action = n_act > 0 && action_noisy.is_none();
        for (i, &t) in timesteps.iter().enumerate() {
            let t0 = Instant::now();
            let tokens = dit::patchify(&hp, &x, lt, lh, lw);
            let timestep = t as f32 * hp.timestep_scale;
            let run = |und: &UndCache, tb: &(dit::Tables, dit::Tables, dit::Tables)| {
                let inp = dit::GenInputs {
                    vision: dit::Segment {
                        tokens: &tokens,
                        n: n_vis,
                        cos: &tb.0 .0,
                        sin: &tb.0 .1,
                        noisy: vis_noisy.as_deref(),
                    },
                    sound: (n_sound > 0).then(|| dit::Segment {
                        tokens: &sound,
                        n: n_sound,
                        cos: &tb.1 .0,
                        sin: &tb.1 .1,
                        noisy: None,
                    }),
                    action: (n_act > 0).then(|| {
                        (
                            dit::Segment {
                                tokens: &act,
                                n: n_act,
                                cos: &tb.2 .0,
                                sin: &tb.2 .1,
                                noisy: action_noisy,
                            },
                            domain_id,
                        )
                    }),
                    timestep,
                    und,
                    flash: self.flash,
                };
                dit::gen_forward(&hp, &self.dit, be, &inp)
            };
            let mut out = run(cond, &cond_tables)?;
            if let (Some(nc), Some(tb)) = (ncond, &uncond_tables) {
                let u = run(nc, tb)?;
                for (o, u) in out.vision.iter_mut().zip(&u.vision) {
                    *o = u + cfg * (*o - u);
                }
                for (o, u) in out.sound.iter_mut().zip(&u.sound) {
                    *o = u + cfg * (*o - u);
                }
                for (o, u) in out.action.iter_mut().zip(&u.action) {
                    *o = u + cfg * (*o - u);
                }
            }
            let mut v = dit::unpatchify(&hp, &out.vision, lt, lh, lw);
            // conditioning frames keep their velocity at zero, so the solver leaves them
            for &tf in &cond_latent_frames {
                for ci in 0..c {
                    v[lat_idx(ci, tf as usize, 0)..lat_idx(ci, tf as usize, 0) + hw].fill(0.0);
                }
            }
            if i == 0 {
                dump("pred0_video", &v);
                dump("pred0_sound", &out.sound);
                dump("pred0_action", &out.action);
            }
            x = solver_v.step(&v, &x);
            if n_sound > 0 {
                sound = solver_s.step(&out.sound, &sound);
            }
            if has_noisy_action {
                let mut va = out.action;
                for t in 0..n_act as usize {
                    va[t * ad + raw_dim as usize..(t + 1) * ad].fill(0.0);
                }
                act = solver_a.step(&va, &act);
                for t in 0..n_act as usize {
                    act[t * ad + raw_dim as usize..(t + 1) * ad].fill(0.0);
                }
            }
            eprintln!(
                "[llmman] mediagen: step {}/{n_steps} t {t} sigma {:.4} -> {:.4} ({:.2} s)",
                i + 1,
                solver_v.sigma(i),
                solver_v.sigma(i + 1),
                t0.elapsed().as_secs_f64()
            );
            if !progress(i as i32 + 1, n_steps) {
                bail!("cancelled");
            }
        }
        dump("latents", &x);
        // debugging aids: decode given latents instead of the sampled ones
        noise_from_env("MEDIAGEN_LATENTS", &mut x)?;
        noise_from_env("MEDIAGEN_LATENTS_SOUND", &mut sound)?;
        dump("latents_sound", &sound);
        be.release_compute()?;

        // ── decode ──
        let t0 = Instant::now();
        let frames: Vec<Vec<f32>> = (0..lt as usize)
            .map(|t| {
                let mut f = vec![0f32; c * hw];
                for ci in 0..c {
                    let (m, sd) = (self.latents_mean[ci], self.latents_std[ci]);
                    for hwi in 0..hw {
                        f[ci * hw + hwi] = x[lat_idx(ci, t, hwi)] * sd + m;
                    }
                }
                f
            })
            .collect();
        let dec = vae::decode(&self.vae, be, &frames, hp.latent_channels, lh, lw);
        be.release_compute()?;
        let dec = dec?;
        eprintln!(
            "[llmman] mediagen: decoded {} frame(s) of {}x{} in {:.2} s",
            dec.frames,
            dec.width,
            dec.height,
            t0.elapsed().as_secs_f64()
        );
        let nf = dec.frames.min(n_frames);
        let mut out = Output {
            width: dec.width,
            height: dec.height,
            n_frames: nf,
            fps: p.fps,
            rgb: dec.rgb[..(dec.width * dec.height * 3 * nf) as usize]
                .iter()
                .map(|v| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect(),
            revised_prompt: p.prompt.clone(),
            seed,
            ..Default::default()
        };
        if n_sound > 0 {
            let t0 = Instant::now();
            let snd = self.sound.as_ref().expect("sound tokenizer");
            let pcm = audio::decode(snd, be, &sound, n_sound);
            be.release_compute()?;
            match pcm {
                Ok(pcm) => {
                    let want = (n_frames as f64 / p.fps as f64 * audio::SAMPLE_RATE as f64).round()
                        as usize;
                    let n = (pcm.len() / 2).min(want);
                    out.sample_rate = audio::SAMPLE_RATE;
                    out.n_channels = 2;
                    out.pcm = pcm[..n * 2].to_vec();
                    eprintln!(
                        "[llmman] mediagen: decoded {:.2} s of audio in {:.2} s",
                        n as f64 / audio::SAMPLE_RATE as f64,
                        t0.elapsed().as_secs_f64()
                    );
                }
                Err(e) => eprintln!(
                    "[llmman] mediagen: sound decoding failed ({e}), returning video only"
                ),
            }
        }
        if has_noisy_action {
            out.actions = (0..n_act as usize)
                .map(|t| act[t * ad..t * ad + raw_dim as usize].to_vec())
                .collect();
        }
        Ok(out)
    }
}

/// Debugging aid: fills `out` from the raw f32 file named by `var` (shared noise with a reference run).
fn noise_from_env(var: &str, out: &mut [f32]) -> Result<bool> {
    let Ok(path) = std::env::var(var) else {
        return Ok(false);
    };
    let bytes = std::fs::read(&path).with_context(|| path.clone())?;
    if bytes.len() != out.len() * 4 {
        bail!("{var} must hold {} little-endian f32 values", out.len());
    }
    for (v, c) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *v = f32::from_le_bytes(*c);
    }
    eprintln!("[llmman] mediagen: using the noise from {var}");
    Ok(true)
}

/// The scheduler shift the reference action examples run with (`--flow-shift 10`).
pub const ACTION_FLOW_SHIFT: f32 = 10.0;

// ── embodiment tables (`_EMBODIMENT_TO_DOMAIN_ID`, `_EMBODIMENT_TO_RAW_ACTION_DIM`) ──

pub fn domain_id(name: &str) -> Option<i64> {
    Some(match name {
        "no_action" => 0,
        "av" => 1,
        "camera_pose" => 2,
        "hand_pose" => 3,
        "pusht" => 4,
        "libero" => 5,
        "umi" => 6,
        "bridge_orig_lerobot" => 7,
        "droid_lerobot" | "robomind-franka" => 8,
        "galbot" => 9,
        "robomind-franka-dual" => 12,
        "robomind-ur" => 13,
        "agibotworld" | "agibot_gear_gripper" | "agibot_gear_gripper_ext" => 15,
        "fractal" => 20,
        _ => return None,
    })
}

pub fn raw_action_dim(name: &str) -> Option<i64> {
    Some(match name {
        "av" | "camera_pose" => 9,
        "pusht" => 2,
        "umi"
        | "bridge_orig_lerobot"
        | "droid_lerobot"
        | "robomind-franka"
        | "robomind-ur"
        | "fractal" => 10,
        "robomind-franka-dual" => 20,
        "galbot" => 30,
        "agibotworld" | "agibot_gear_gripper" | "agibot_gear_gripper_ext" => 29,
        "hand_pose" => 57,
        _ => return None,
    })
}

/// `_ACTION_RESOLUTION_BINS[tier]`: (aspect h/w, (height, width)).
fn resolution_bins(tier: i64) -> Option<&'static [(f64, i64, i64)]> {
    Some(match tier {
        256 => &[
            (1.0, 256, 256),
            (0.8, 256, 320),
            (1.25, 320, 256),
            (0.6, 192, 320),
            (1.6666666666666667, 320, 192),
        ],
        480 => &[
            (1.0, 640, 640),
            (0.7391304347826086, 544, 736),
            (1.3529411764705883, 736, 544),
            (0.5769230769230769, 480, 832),
            (1.7333333333333334, 832, 480),
        ],
        704 => &[
            (1.0, 960, 960),
            (0.7647058823529411, 832, 1088),
            (1.3076923076923077, 1088, 832),
            (0.55, 704, 1280),
            (1.8181818181818181, 1280, 704),
        ],
        720 => &[
            (1.0, 960, 960),
            (0.7536231884057971, 832, 1104),
            (1.3269230769230769, 1104, 832),
            (0.5625, 720, 1280),
            (1.7777777777777777, 1280, 720),
        ],
        _ => return None,
    })
}

/// `_build_action_json_prompt`
pub fn action_prompt(
    description: &str,
    view_point: &str,
    n_frames: i64,
    fps: f32,
    height: i64,
    width: i64,
) -> String {
    let secs = if fps > 0.0 {
        n_frames as f64 / fps as f64
    } else {
        0.0
    };
    let duration = secs as i64;
    let end = secs.round() as i64;
    let (minutes, seconds) = (end / 60, end % 60);
    let mut desc = description.trim().to_string();
    if !desc.is_empty() && !desc.ends_with(['.', '!', '?']) {
        desc.push('.');
    }
    let framing = match view_point {
        "ego_view" => Some("This video is captured from a first-person perspective looking at the scene."),
        "third_person_view" => Some("This video is captured from a third-person perspective looking towards the agent from the front."),
        "wrist_view" => Some("This video is captured from a wrist-mounted camera."),
        "concat_view" => Some("This video contains concatenated views from multiple camera perspectives."),
        _ => None,
    };
    let ratio = if height > 0 {
        width as f64 / height as f64
    } else {
        1.0
    };
    let aspect = ["1,1", "4,3", "3,4", "16,9", "9,16"]
        .into_iter()
        .min_by(|a, b| {
            let r = |s: &str| {
                let (x, y) = s.split_once(',').unwrap();
                x.parse::<f64>().unwrap() / y.parse::<f64>().unwrap()
            };
            (r(a) - ratio).abs().total_cmp(&(r(b) - ratio).abs())
        })
        .unwrap();
    // json.dumps key order: cinematography (if any), actions, duration, fps, resolution, aspect_ratio
    let mut obj = serde_json::Map::new();
    if let Some(f) = framing {
        obj.insert("cinematography".into(), serde_json::json!({"framing": f}));
    }
    obj.insert(
        "actions".into(),
        serde_json::json!([{"time": format!("0:00-{minutes}:{seconds:02}"), "description": desc}]),
    );
    obj.insert("duration".into(), Value::String(format!("{duration}s")));
    obj.insert("fps".into(), Value::from(fps as f64));
    obj.insert(
        "resolution".into(),
        serde_json::json!({"H": height, "W": width}),
    );
    obj.insert("aspect_ratio".into(), Value::String(aspect.into()));
    python_json(&Value::Object(obj))
}

/// `json.dumps` with its default separators (`", "` and `": "`) and float formatting.
fn python_json(v: &Value) -> String {
    match v {
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!("{}: {}", Value::String(k.clone()), python_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(python_json).collect::<Vec<_>>().join(", ")
        ),
        Value::Number(n) if n.is_f64() => {
            let f = n.as_f64().unwrap();
            if f.fract() == 0.0 && f.abs() < 1e16 {
                format!("{f:.1}")
            } else {
                format!("{f}")
            }
        }
        other => other.to_string(),
    }
}

/// `_preprocess_conditioning_image`: cover-scale, center-crop to `w x h`, `[-1, 1]`.
fn cover_crop(img: &Frames, w: i64, h: i64) -> Result<Vec<f32>> {
    if img.n_frames < 1 {
        bail!("no conditioning image");
    }
    let (sw, sh) = (img.width as f64, img.height as f64);
    let scale = (w as f64 / sw).max(h as f64 / sh);
    let (rw, rh) = ((scale * sw).ceil() as i64, (scale * sh).ceil() as i64);
    let resized = resize_rgb(
        &img.rgb[..(img.width * img.height * 3) as usize],
        img.width,
        img.height,
        rw,
        rh,
        false,
    );
    let top = ((rh - h) as f64 / 2.0).round() as i64;
    let left = ((rw - w) as f64 / 2.0).round() as i64;
    let mut out = vec![0f32; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let v = resized[(((y + top) * rw + x + left) * 3 + ch) as usize];
                out[((y * w + x) * 3 + ch) as usize] = v.round().clamp(0.0, 255.0) / 127.5 - 1.0;
            }
        }
    }
    Ok(out)
}

/// `VideoProcessor.preprocess_video`: the clip's first (or last) `max_frames`
/// frames resized straight to `w x h`, the last repeated when it is shorter.
fn clip_frames(vid: &Frames, w: i64, h: i64, max_frames: i64, keep_last: bool) -> Result<Vec<f32>> {
    if vid.n_frames < 1 {
        bail!("no conditioning frames");
    }
    let n = vid.n_frames.min(max_frames);
    let first = if keep_last { vid.n_frames - n } else { 0 };
    let frame_px = (vid.width * vid.height * 3) as usize;
    let mut out = Vec::with_capacity((max_frames * w * h * 3) as usize);
    let mut last = Vec::new();
    for f in 0..max_frames {
        if f < n {
            let fi = (first + f) as usize;
            let frame = &vid.rgb[fi * frame_px..(fi + 1) * frame_px];
            let resized = if (vid.width, vid.height) != (w, h) {
                resize_rgb(frame, vid.width, vid.height, w, h, false)
            } else {
                frame.iter().map(|&v| v as f32).collect()
            };
            last = resized
                .iter()
                .map(|v| v.round().clamp(0.0, 255.0) / 127.5 - 1.0)
                .collect();
        }
        out.extend_from_slice(&last);
    }
    Ok(out)
}

/// `_prepare_action_video_conditioning`: the clip downscaled into the tier's
/// canvas and padded right/bottom; returns frames, content size, canvas size.
fn action_frames(src: &Frames, tier: i64, n_frames: i64) -> Result<(Vec<f32>, i64, i64, i64, i64)> {
    let bins = resolution_bins(tier)
        .ok_or_else(|| anyhow!("action resolution tier must be 256, 480, 704 or 720"))?;
    if src.n_frames < 1 {
        bail!("no conditioning frames");
    }
    let (fw, fh) = (src.width, src.height);
    let ar = fh as f64 / fw as f64;
    let &(_, th, tw) = bins
        .iter()
        .min_by(|a, b| (a.0 - ar).abs().total_cmp(&(b.0 - ar).abs()))
        .unwrap();
    let scale = (tw as f64 / fw as f64).min(th as f64 / fh as f64).min(1.0);
    let ch = ((scale * fh as f64 + 0.5) as i64).max(1);
    let cw = ((scale * fw as f64 + 0.5) as i64).max(1);
    let (pad_r, pad_b) = (tw - cw, th - ch);
    let reflect = !(pad_r >= cw || pad_b >= ch);
    let frame_px = (fw * fh * 3) as usize;
    let mut out = Vec::with_capacity((n_frames * th * tw * 3) as usize);
    for f in 0..n_frames {
        // shorter clips repeat their last frame
        let fi = f.min(src.n_frames - 1) as usize;
        let frame = &src.rgb[fi * frame_px..(fi + 1) * frame_px];
        let content = if (ch, cw) != (fh, fw) {
            resize_rgb(frame, fw, fh, cw, ch, true)
        } else {
            frame.iter().map(|&v| v as f32).collect()
        };
        for y in 0..th {
            let sy = pad_index(y, ch, reflect);
            for x in 0..tw {
                let sx = pad_index(x, cw, reflect);
                for c in 0..3 {
                    let v = content[((sy * cw + sx) * 3 + c) as usize];
                    out.push(v.round().clamp(0.0, 255.0) / 127.5 - 1.0);
                }
            }
        }
    }
    Ok((out, ch, cw, th, tw))
}

fn pad_index(i: i64, n: i64, reflect: bool) -> i64 {
    if i < n {
        i
    } else if reflect {
        2 * n - 2 - i
    } else {
        n - 1
    }
}

/// Antialiased resize of RGB8 `[h][w][3]` to f32 `[nh][nw][3]` (triangle or Catmull-Rom).
fn resize_rgb(src: &[u8], w: i64, h: i64, nw: i64, nh: i64, bicubic: bool) -> Vec<f32> {
    use image::imageops::FilterType;
    let img = image::RgbImage::from_raw(w as u32, h as u32, src.to_vec()).expect("rgb buffer");
    let filter = if bicubic {
        FilterType::CatmullRom
    } else {
        FilterType::Triangle
    };
    let out = image::imageops::resize(&img, nw as u32, nh as u32, filter);
    out.into_raw().into_iter().map(|v| v as f32).collect()
}

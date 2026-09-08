//! Image, video and audio generation with diffusion models on top of the
//! ggml/llama shared libraries of a llama.cpp release, loaded at runtime.

// ggml tensors are raw handles owned by their context; the graph builders
// mirror the C API signatures one to one.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

pub mod backend;
pub mod cosmos3;
pub mod encode;
pub mod ffi;
pub mod ltx;
pub mod server;
pub mod weights;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Result};

use backend::Backend;
use ffi::Api;
use ltx::{text::TextEncoder, AudioLatent, Model, TextCond, VideoLatent};
use weights::{LoadOpts, Weights};

/// Model files of one diffusion model.
pub struct ContextParams {
    pub model: PathBuf,
    pub vae: Option<PathBuf>,
    pub audio_vae: Option<PathBuf>,
    pub text_proj: Option<PathBuf>,
    /// Text encoder GGUF; LTX needs one, Cosmos3 has none.
    pub text_model: Option<PathBuf>,
    /// Every named file of the model pack (`vae/config.json`, `tokenizer.json`, ...).
    pub files: BTreeMap<String, PathBuf>,
    pub use_gpu: bool,
    /// GPU layers of the text encoder.
    pub text_gpu_layers: i32,
    pub n_threads: i32,
    pub flash_attn: bool,
}

#[derive(Clone, Debug)]
pub struct GenParams {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: i64,
    pub height: i64,
    pub n_frames: i64,
    pub fps: f32,
    /// `0` for the model's default.
    pub n_steps: i32,
    /// `0` for the model's default (LTX: 1, Cosmos3: 6).
    pub cfg_scale: f32,
    /// `None` picks a random seed.
    pub seed: Option<u32>,
    pub gen_audio: bool,
    pub enhance_prompt: bool,
    /// Conditioning image: image-to-video (and the first frame of an action run).
    pub image: Option<Frames>,
    /// Conditioning clip: an action run's observed video, or the leading frames
    /// of a video-to-video run.
    pub video: Option<Frames>,
    /// Video-to-video: the latent frames kept from the clip (`(0, 1)` in the reference).
    pub condition_frames: Vec<i64>,
    /// Video-to-video: take the conditioning frames from the end of the clip.
    pub condition_keep_last: bool,
    /// An action-conditioned run (Cosmos3).
    pub action: Option<ActionParams>,
    /// Scheduler flow shift override (Cosmos3: Karras sigmas when `None`).
    pub flow_shift: Option<f32>,
}

/// Decoded RGB frames, `[f][h][w][3]` bytes.
#[derive(Clone, Debug, Default)]
pub struct Frames {
    pub width: i64,
    pub height: i64,
    pub n_frames: i64,
    pub rgb: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionMode {
    /// Video from the first frame and the given actions.
    ForwardDynamics,
    /// Actions connecting the frames of the given video.
    InverseDynamics,
    /// Video and actions from the first frame.
    Policy,
}

impl ActionMode {
    pub fn parse(s: &str) -> Option<ActionMode> {
        match s {
            "forward_dynamics" | "fd" => Some(ActionMode::ForwardDynamics),
            "inverse_dynamics" | "id" => Some(ActionMode::InverseDynamics),
            "policy" => Some(ActionMode::Policy),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ActionMode::ForwardDynamics => "forward_dynamics",
            ActionMode::InverseDynamics => "inverse_dynamics",
            ActionMode::Policy => "policy",
        }
    }
}

/// `CosmosActionCondition`.
#[derive(Clone, Debug)]
pub struct ActionParams {
    pub mode: ActionMode,
    /// Action transitions in the chunk; the clip has one more frame.
    pub chunk_size: i64,
    /// Embodiment domain (`bridge_orig_lerobot`, `av`, ...).
    pub domain: String,
    /// Conditioning canvas tier: 256, 480, 704 or 720.
    pub resolution_tier: i64,
    /// `[T][raw_action_dim]` for forward dynamics.
    pub actions: Vec<Vec<f32>>,
    pub view_point: String,
}

impl Default for GenParams {
    fn default() -> Self {
        GenParams {
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 768,
            height: 512,
            n_frames: 1,
            fps: 24.0,
            n_steps: 0,
            cfg_scale: 0.0,
            seed: None,
            gen_audio: false,
            enhance_prompt: true,
            image: None,
            video: None,
            condition_frames: vec![0, 1],
            condition_keep_last: false,
            action: None,
            flow_shift: None,
        }
    }
}

/// Decoded frames (`[f][h][w][3]` bytes) and optional interleaved PCM.
#[derive(Default, Clone)]
pub struct Output {
    pub width: i64,
    pub height: i64,
    pub n_frames: i64,
    pub fps: f32,
    pub rgb: Vec<u8>,
    pub sample_rate: i32,
    pub n_channels: i32,
    pub pcm: Vec<f32>,
    pub revised_prompt: String,
    pub seed: u32,
    /// Predicted actions `[T][raw_action_dim]` of an action run.
    pub actions: Vec<Vec<f32>>,
}

impl Output {
    pub fn frame(&self, i: usize) -> &[u8] {
        let n = (self.width * self.height * 3) as usize;
        &self.rgb[i * n..(i + 1) * n]
    }
}

/// The loaded model of one of the supported architectures.
enum Inner {
    Ltx(Box<LtxCtx>),
    Cosmos3(Box<cosmos3::Model>),
}

pub struct Context {
    // dropped in this order: weight buffers before their backend
    inner: Inner,
    be: Backend,
}

/// An LTX-2 model with its Gemma text encoder and prompt caches.
struct LtxCtx {
    ltx: Model,
    text: Box<TextEncoder>,
    distilled: bool,
    flash_attn: bool,
    has_video_vae: bool,
    has_audio_vae: bool,
    enhanced: Option<(String, String)>,
    cond: Option<(String, TextCond)>,
    ncond: Option<(String, TextCond)>,
}

fn is_ltx(arch: &str) -> bool {
    matches!(arch, "ltxv" | "ltx-video" | "ltxav")
}

/// Is this GGUF a diffusion transformer we can run?
pub fn is_diffusion_model(path: &Path) -> bool {
    weights::read_index(path)
        .ok()
        .and_then(|i| i.metadata.get("general.architecture").cloned())
        .is_some_and(|a| is_ltx(&a) || cosmos3::is_cosmos3(&a))
}

/// Does this architecture run without a separate text encoder GGUF?
pub fn needs_text_encoder(path: &Path) -> bool {
    weights::read_index(path)
        .ok()
        .and_then(|i| i.metadata.get("general.architecture").cloned())
        .is_none_or(|a| !cosmos3::is_cosmos3(&a))
}

/// Audio latent frames covering the whole clip; the decoded audio is trimmed to the video.
pub fn audio_latent_frames(hp: &ltx::Hparams, n_video_frames: i64, fps: f32) -> i64 {
    let duration = n_video_frames as f64 / fps as f64;
    let per_sec = hp.audio_sample_rate as f64
        / hp.audio_hop_length as f64
        / hp.audio_latent_downsample as f64;
    ((duration * per_sec).ceil() as i64 + 1).max(1)
}

/// mt19937 + `std::normal_distribution<float>` as libc++ implements it (Marsaglia polar in
/// `float`), for the same noise as the C++ implementation given a seed.
pub(super) struct Mt19937 {
    mt: [u32; 624],
    i: usize,
    saved: Option<f32>,
}

impl Mt19937 {
    pub(super) fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 {
            mt,
            i: 624,
            saved: None,
        }
    }

    fn next_u32(&mut self) -> u32 {
        if self.i >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(k + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[k] = v;
            }
            self.i = 0;
        }
        let mut y = self.mt[self.i];
        self.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `generate_canonical<float, 24>`: one 32-bit draw.
    fn canonical_f32(&mut self) -> f32 {
        self.next_u32() as f32 / 4294967296.0
    }

    pub(super) fn normal(&mut self) -> f32 {
        if let Some(v) = self.saved.take() {
            return v;
        }
        loop {
            let x = 2.0 * self.canonical_f32() - 1.0;
            let y = 2.0 * self.canonical_f32() - 1.0;
            // clang contracts `x*x + y*y` into an fma
            let r2 = x.mul_add(x, y * y);
            if r2 <= 1.0 && r2 != 0.0 {
                let m = (-2.0 * r2.ln() / r2).sqrt();
                self.saved = Some(y * m);
                return x * m;
            }
        }
    }
}

impl Context {
    pub fn init(api: &'static Api, p: &ContextParams) -> Result<Context> {
        let idx = weights::read_index(&p.model)?;
        let arch = idx
            .metadata
            .get("general.architecture")
            .cloned()
            .unwrap_or_default();
        let mut be = Backend::new(api, p.use_gpu, p.n_threads, 64 * 1024)?;
        let inner = if is_ltx(&arch) {
            Inner::Ltx(Box::new(LtxCtx::load(api, &mut be, p, &idx)?))
        } else if cosmos3::is_cosmos3(&arch) {
            let m = cosmos3::Model::load(
                api,
                &be,
                &idx,
                &p.model,
                p.vae.as_deref(),
                &p.files,
                p.flash_attn,
            )?;
            be.release_compute()?;
            Inner::Cosmos3(Box::new(m))
        } else {
            bail!("unsupported diffusion architecture {arch:?}");
        };
        Ok(Context { inner, be })
    }

    pub fn supports_video(&self) -> bool {
        match &self.inner {
            Inner::Ltx(l) => l.has_video_vae,
            Inner::Cosmos3(_) => true,
        }
    }

    pub fn supports_audio(&self) -> bool {
        match &self.inner {
            Inner::Ltx(l) => l.ltx.hp.has_audio && l.has_audio_vae,
            Inner::Cosmos3(m) => m.sound.is_some(),
        }
    }

    pub fn family(&self) -> &'static str {
        match &self.inner {
            Inner::Ltx(_) => "ltx",
            Inner::Cosmos3(_) => "cosmos3",
        }
    }

    /// Denoising steps when a request does not say (0: the schedule decides).
    pub fn default_steps(&self) -> i32 {
        match &self.inner {
            Inner::Ltx(_) => 0,
            Inner::Cosmos3(_) => cosmos3::DEFAULT_STEPS,
        }
    }

    pub fn default_cfg(&self) -> f32 {
        match &self.inner {
            Inner::Ltx(_) => 1.0,
            Inner::Cosmos3(_) => cosmos3::DEFAULT_CFG,
        }
    }

    /// Frames come in `stride * k + 1`.
    pub fn temporal_stride(&self) -> i64 {
        match &self.inner {
            Inner::Ltx(l) => l.ltx.hp.vae_scale_t,
            Inner::Cosmos3(m) => m.hp.vae_scale_t,
        }
    }

    /// Runs the whole pipeline; `progress(step, n_steps)` returning `false` cancels.
    pub fn generate(
        &mut self,
        p: &GenParams,
        progress: impl FnMut(i32, i32) -> bool,
    ) -> Result<Output> {
        let out = match &mut self.inner {
            Inner::Ltx(l) => l.generate_inner(&mut self.be, p, progress),
            Inner::Cosmos3(m) => m.generate(&mut self.be, p, progress),
        };
        // frees the compute buffers; recreates the GPU backend after a failure
        self.be.release_compute()?;
        out
    }
}

impl LtxCtx {
    fn load(
        api: &'static Api,
        be: &mut Backend,
        p: &ContextParams,
        idx: &weights::Index,
    ) -> Result<LtxCtx> {
        let mut hp = match idx.metadata.get("config") {
            Some(c) => ltx::Hparams::from_config(c)?,
            None => ltx::Hparams::default(),
        };
        hp.has_audio = idx
            .tensors
            .iter()
            .any(|d| d.name == "audio_adaln_single.linear.weight");
        let distilled = p
            .model
            .to_string_lossy()
            .to_lowercase()
            .contains("distilled");

        let buft = be.weight_buft();
        let t0 = Instant::now();
        let dit = Weights::load(api, &p.model, buft, &LoadOpts::default())?;
        let text_proj = match &p.text_proj {
            Some(path) => Some(Weights::load(
                api,
                path,
                buft,
                &LoadOpts {
                    rename: Some(Box::new(|n| {
                        n.starts_with("text_embedding_projection.")
                            .then(|| n.to_string())
                    })),
                    transform: None,
                },
            )?),
            None => None,
        };
        let vae = match &p.vae {
            Some(path) => Some(Weights::load(
                api,
                path,
                buft,
                &LoadOpts {
                    rename: Some(Box::new(|n| {
                        let s = n.strip_prefix("vae.").unwrap_or(n);
                        // decoder and statistics only
                        (!s.starts_with("encoder.")).then(|| s.to_string())
                    })),
                    transform: None,
                },
            )?),
            None => None,
        };
        let audio_vae = match &p.audio_vae {
            Some(path) => Some(Weights::load(
                api,
                path,
                buft,
                &LoadOpts {
                    rename: Some(Box::new(|n| {
                        (!n.starts_with("audio_vae.encoder.")).then(|| n.to_string())
                    })),
                    transform: Some(Box::new(ltx::audio::transform)),
                },
            )?),
            None => None,
        };
        let ltx = Model {
            hp,
            dit,
            text_proj,
            vae,
            audio_vae,
        };
        if ltx
            .tp_opt("text_embedding_projection.video_aggregate_embed.weight")
            .is_none()
        {
            bail!(
                "text embedding projection weights not found (pass the embeddings connectors file)"
            );
        }
        let has_video_vae = ltx
            .vae
            .as_ref()
            .is_some_and(|w| w.has("decoder.conv_in.conv.weight.t0"));
        let has_audio_vae = ltx
            .audio_vae
            .as_ref()
            .is_some_and(|w| w.has("audio_vae.decoder.conv_in.conv.weight"));

        // text encoder, with room for prompt enhancement
        let n_ctx = ltx.hp.text_max_tokens.max(2048) as u32;
        let text_model = p.text_model.as_ref().ok_or_else(|| {
            anyhow::anyhow!("LTX needs a text encoder GGUF (role \"text_encoder\")")
        })?;
        let text = TextEncoder::load(
            api,
            &text_model.to_string_lossy(),
            p.text_gpu_layers,
            n_ctx,
            p.n_threads,
            p.flash_attn,
        )?;
        be.release_compute()?;
        eprintln!(
            "[llmman] mediagen: ltxv model loaded in {:.1} s (distilled={distilled}, audio={}, video_vae={has_video_vae}, audio_vae={has_audio_vae})",
            t0.elapsed().as_secs_f64(),
            ltx.hp.has_audio
        );
        Ok(LtxCtx {
            ltx,
            text,
            distilled,
            flash_attn: p.flash_attn,
            has_video_vae,
            has_audio_vae,
            enhanced: None,
            cond: None,
            ncond: None,
        })
    }

    fn supports_audio(&self) -> bool {
        self.ltx.hp.has_audio && self.has_audio_vae
    }

    fn get_cond(&mut self, be: &Backend, prompt: &str, negative: bool) -> Result<TextCond> {
        let slot = if negative { &self.ncond } else { &self.cond };
        if let Some((p, c)) = slot {
            if p == prompt {
                return Ok(c.clone());
            }
        }
        let t0 = Instant::now();
        let hp = &self.ltx.hp;
        let (packed, n_tokens, n_hidden, n_states) =
            self.text.encode(prompt, hp.text_max_tokens)?;
        if n_hidden != hp.text_hidden_size || n_states != hp.text_n_states {
            bail!(
                "text encoder produced {n_states} states of size {n_hidden}, expected {} x {}",
                hp.text_n_states,
                hp.text_hidden_size
            );
        }
        let cond = ltx::text::build_text_cond(
            &self.ltx,
            be,
            &packed,
            n_tokens,
            n_hidden,
            n_states,
            self.flash_attn,
        )?;
        dump(
            if negative {
                "ntext_packed"
            } else {
                "text_packed"
            },
            &packed,
        );
        dump(
            if negative {
                "ncond_video"
            } else {
                "cond_video"
            },
            &cond.video,
        );
        dump(
            if negative {
                "ncond_audio"
            } else {
                "cond_audio"
            },
            &cond.audio,
        );
        eprintln!(
            "[llmman] mediagen: encoded {} prompt ({n_tokens} tokens) in {:.2} s",
            if negative { "negative" } else { "positive" },
            t0.elapsed().as_secs_f64()
        );
        let slot = if negative {
            &mut self.ncond
        } else {
            &mut self.cond
        };
        *slot = Some((prompt.to_string(), cond.clone()));
        Ok(cond)
    }

    fn generate_inner(
        &mut self,
        be: &mut Backend,
        p: &GenParams,
        mut progress: impl FnMut(i32, i32) -> bool,
    ) -> Result<Output> {
        if !self.has_video_vae {
            bail!("no video VAE loaded");
        }
        let hp = self.ltx.hp.clone();
        let (w, h) = (p.width, p.height);
        if w <= 0 || h <= 0 || w % hp.vae_scale_s != 0 || h % hp.vae_scale_s != 0 {
            bail!(
                "width and height must be positive multiples of {}",
                hp.vae_scale_s
            );
        }
        let mut n_frames = p.n_frames.max(1);
        if (n_frames - 1) % hp.vae_scale_t != 0 {
            n_frames = (n_frames - 1) / hp.vae_scale_t * hp.vae_scale_t + 1;
            eprintln!("[llmman] mediagen: n_frames must be 8k+1, using {n_frames}");
        }
        let gen_audio = p.gen_audio && n_frames > 1 && self.supports_audio();
        let cfg = if p.cfg_scale > 0.0 { p.cfg_scale } else { 1.0 };
        let use_cfg = cfg > 1.0;

        // prompt enhancement, fixed seed like the reference pipelines
        let mut prompt = p.prompt.clone();
        if p.enhance_prompt && !prompt.is_empty() {
            match &self.enhanced {
                Some((from, to)) if *from == prompt => prompt = to.clone(),
                _ => {
                    let t0 = Instant::now();
                    match self.text.enhance(&prompt, 10, 512) {
                        Ok(e) => {
                            eprintln!("[llmman] mediagen: enhanced prompt in {:.1} s: {e}", t0.elapsed().as_secs_f64());
                            self.enhanced = Some((prompt.clone(), e.clone()));
                            prompt = e;
                        }
                        Err(e) => eprintln!("[llmman] mediagen: prompt enhancement failed ({e}), using the prompt as is"),
                    }
                }
            }
        }

        let cond = self.get_cond(be, &prompt, false)?;
        let ncond = if use_cfg {
            Some(self.get_cond(be, &p.negative_prompt, true)?)
        } else {
            None
        };
        // free its buffers for the transformer and the VAE
        self.text.release_context();

        let mut lat = VideoLatent {
            n_frames: (n_frames - 1) / hp.vae_scale_t + 1,
            height: h / hp.vae_scale_s,
            width: w / hp.vae_scale_s,
            channels: hp.latent_channels,
            x: Vec::new(),
        };
        lat.x = vec![0.0; (lat.n_tokens() * lat.channels) as usize];
        let mut alat = AudioLatent::default();
        if gen_audio {
            alat.n_frames = audio_latent_frames(&hp, n_frames, p.fps);
            alat.channels = hp.audio_channels;
            alat.freq = hp.audio_freq_bins;
            alat.x = vec![0.0; (alat.n_frames * alat.channels * alat.freq) as usize];
        }
        let seed = p.seed.unwrap_or_else(rand_seed);
        let mut rng = Mt19937::new(seed);
        for v in lat.x.iter_mut().chain(alat.x.iter_mut()) {
            *v = rng.normal();
        }

        dump("noise_video", &lat.x);
        dump("noise_audio", &alat.x);
        let sigmas = ltx::dit::sigmas(p.n_steps, lat.n_tokens(), self.distilled);
        let n_steps = sigmas.len() as i32 - 1;
        eprintln!(
            "[llmman] mediagen: generating {w}x{h}, {n_frames} frame(s), {} latent tokens, {n_steps} steps, cfg {cfg:.1}, seed {seed}{}",
            lat.n_tokens(),
            if gen_audio { ", with audio" } else { "" }
        );

        for i in 0..n_steps as usize {
            let t0 = Instant::now();
            let inp = ltx::dit::Inputs {
                video: &lat,
                audio: gen_audio.then_some(&alat),
                cond: &cond,
                sigma: sigmas[i],
                fps: p.fps,
                flash: self.flash_attn,
            };
            let (mut v_pred, mut a_pred) = ltx::dit::forward(&self.ltx, be, &inp)?;
            if v_pred.len() != lat.x.len() || a_pred.len() != alat.x.len() {
                bail!("transformer output does not match the latents");
            }
            if i == 0 {
                dump("pred0_video", &v_pred);
                dump("pred0_audio", &a_pred);
            }
            if let Some(nc) = &ncond {
                let inp = ltx::dit::Inputs { cond: nc, ..inp };
                let (v_unc, a_unc) = ltx::dit::forward(&self.ltx, be, &inp)?;
                for (p, u) in v_pred.iter_mut().zip(&v_unc) {
                    *p = u + cfg * (*p - u);
                }
                for (p, u) in a_pred.iter_mut().zip(&a_unc) {
                    *p = u + cfg * (*p - u);
                }
            }
            // euler step on the flow: x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v
            let dt = sigmas[i + 1] - sigmas[i];
            for (x, v) in lat.x.iter_mut().zip(&v_pred) {
                *x += dt * v;
            }
            for (x, v) in alat.x.iter_mut().zip(&a_pred) {
                *x += dt * v;
            }
            eprintln!(
                "[llmman] mediagen: step {}/{n_steps} sigma {:.4} -> {:.4} ({:.2} s)",
                i + 1,
                sigmas[i],
                sigmas[i + 1],
                t0.elapsed().as_secs_f64()
            );
            if !progress(i as i32 + 1, n_steps) {
                bail!("cancelled");
            }
        }

        // the transformer's compute buffers are of no use to the VAE
        be.release_compute()?;
        let mut out = Output {
            fps: p.fps,
            revised_prompt: prompt,
            seed,
            ..Default::default()
        };
        {
            let t0 = Instant::now();
            let dec = ltx::vae::decode(&self.ltx, be, &lat);
            be.release_compute()?;
            let (of, oh, ow, rgb) = dec?;
            eprintln!(
                "[llmman] mediagen: decoded {of} frame(s) of {ow}x{oh} in {:.2} s",
                t0.elapsed().as_secs_f64()
            );
            let nf = of.min(n_frames);
            out.width = ow;
            out.height = oh;
            out.n_frames = nf;
            out.rgb = rgb[..(ow * oh * 3 * nf) as usize]
                .iter()
                .map(|v| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect();
        }
        if gen_audio {
            let t0 = Instant::now();
            match ltx::audio::decode(&self.ltx, be, &alat) {
                Err(e) => eprintln!(
                    "[llmman] mediagen: audio decoding failed ({e}), returning video only"
                ),
                Ok((sr, nch, pcm)) => {
                    // trim to the video duration
                    let want = (n_frames as f64 / p.fps as f64 * sr as f64).round() as usize;
                    let n_samples = (pcm.len() / nch as usize).min(want);
                    out.sample_rate = sr;
                    out.n_channels = nch;
                    out.pcm = pcm[..n_samples * nch as usize].to_vec();
                    eprintln!(
                        "[llmman] mediagen: decoded {:.2} s of audio in {:.2} s",
                        n_samples as f64 / sr as f64,
                        t0.elapsed().as_secs_f64()
                    );
                }
            }
        }
        Ok(out)
    }
}

/// `MEDIAGEN_DUMP=<prefix>` writes intermediates as raw f32 files, like the C++ implementation.
pub(super) fn dump(name: &str, data: &[f32]) {
    if let Ok(prefix) = std::env::var("MEDIAGEN_DUMP") {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{prefix}.{name}.bin"), bytes);
    }
}

pub(super) fn rand_seed() -> u32 {
    let mut b = [0u8; 4];
    if std::fs::File::open("/dev/urandom")
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut b))
        .is_ok()
    {
        return u32::from_le_bytes(b);
    }
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() ^ d.as_secs() as u32)
        .unwrap_or(42)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_reference() {
        // std::mt19937 with the default seed 5489: the 10000th output is 4123659995
        let mut r = Mt19937::new(5489);
        let mut v = 0;
        for _ in 0..10000 {
            v = r.next_u32();
        }
        assert_eq!(v, 4123659995);
    }

    #[test]
    fn normal_matches_libcxx() {
        // std::mt19937 rng(42); std::normal_distribution<float> nd; Apple clang -O2
        let mut r = Mt19937::new(42);
        let want = [
            -0.5169641375541687,
            1.2219213247299194,
            0.7213326692581177,
            0.8696359395980835,
            1.61821711063385,
            1.5885562896728516,
        ];
        for w in want {
            assert_eq!(r.normal() as f64, w);
        }
    }

    #[test]
    fn normal_matches_reference_dump() {
        let Ok(path) = std::env::var("MT_REF") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let mut r = Mt19937::new(42);
        for (i, c) in bytes.as_chunks::<4>().0.iter().enumerate() {
            let want = f32::from_le_bytes(*c);
            assert_eq!(r.normal().to_bits(), want.to_bits(), "sample {i}");
        }
    }

    #[test]
    fn normal_is_roughly_standard() {
        let mut r = Mt19937::new(1);
        let n = 100000;
        let xs: Vec<f64> = (0..n).map(|_| r.normal() as f64).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.02, "{mean}");
        assert!((var - 1.0).abs() < 0.05, "{var}");
    }

    #[test]
    fn audio_frames_cover_clip() {
        let hp = ltx::Hparams::default();
        assert_eq!(audio_latent_frames(&hp, 1, 24.0), 3);
        assert_eq!(audio_latent_frames(&hp, 121, 24.0), 128);
    }
}

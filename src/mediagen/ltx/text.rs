//! Text conditioning: every hidden state of the llama text encoder, packed
//! and normalized per token, then projected per modality and refined by the
//! connectors. Also the prompt enhancement that expands short prompts with
//! the same model.

use std::ffi::{c_void, CString};
use std::ptr;

use anyhow::{bail, Result};

use super::super::backend::{Backend, Graph};
use super::super::ffi::{self, Api, LlamaContextParams, LlamaModel, LlamaModelParams, Tensor};
use super::{attention, feed_forward, rms, rope_tables, AttnWeights, Model, Rope, TextCond};

/// The LTX-2 system prompt for prompt enhancement (Lightricks, Apache-2.0):
/// <https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_t2v_system_prompt.txt>
const T2V_SYSTEM_PROMPT: &str = include_str!("t2v_system_prompt.txt");

struct Capture {
    n_layers: i32,
    n_tokens: i64,
    n_hidden: i64,
    states: Vec<Vec<f32>>,
    got: Vec<bool>,
}

impl Capture {
    fn index_of(&self, name: &str) -> Option<usize> {
        if name == "inp_scaled" {
            return Some(0);
        }
        if let Some(il) = name.strip_prefix("l_out-") {
            let il: i32 = il.parse().ok()?;
            // the last state is the final norm output, read from the embeddings
            return (il + 1 < self.n_layers).then_some((il + 1) as usize);
        }
        None
    }
}

/// The text model, its context and the ggml API it was loaded with. The
/// context (gigabytes of KV cache and compute buffers) is created on demand
/// and released with [`TextEncoder::release_context`] between prompts.
pub struct TextEncoder {
    api: &'static Api,
    model: *mut LlamaModel,
    ctx: *mut ffi::LlamaContextT,
    n_ctx: u32,
    n_threads: i32,
    flash: bool,
    capture: *mut Capture,
}

unsafe impl Send for TextEncoder {}

unsafe extern "C" fn cb_eval(t: Tensor, ask: bool, user_data: *mut c_void) -> bool {
    let enc = &*(user_data as *const TextEncoder);
    if enc.capture.is_null() {
        return false;
    }
    let cap = &mut *enc.capture;
    let name = (*t).name();
    let Some(idx) = cap.index_of(&name) else {
        return false;
    };
    if ask {
        return true;
    }
    let ne = (*t).ne;
    if (*t).type_ != ffi::ty::F32 || ne[1] != cap.n_tokens {
        return true;
    }
    if cap.n_hidden == 0 {
        cap.n_hidden = ne[0];
    }
    let dst = &mut cap.states[idx];
    dst.resize((cap.n_tokens * cap.n_hidden) as usize, 0.0);
    if (enc.api.ggml_is_contiguous)(t) {
        (enc.api.ggml_backend_tensor_get)(t, dst.as_mut_ptr() as *mut _, 0, dst.len() * 4);
    } else {
        let nb1 = (*t).nb[1];
        for i in 0..ne[1] as usize {
            (enc.api.ggml_backend_tensor_get)(
                t,
                dst[i * ne[0] as usize..].as_mut_ptr() as *mut _,
                i * nb1,
                ne[0] as usize * 4,
            );
        }
    }
    cap.got[idx] = true;
    true
}

impl Drop for TextEncoder {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                (self.api.llama_free)(self.ctx);
            }
            if !self.model.is_null() {
                (self.api.llama_model_free)(self.model);
            }
        }
    }
}

impl TextEncoder {
    pub fn load(
        api: &'static Api,
        path: &str,
        n_gpu_layers: i32,
        n_ctx: u32,
        n_threads: i32,
        flash: bool,
    ) -> Result<Box<TextEncoder>> {
        unsafe {
            let mut mp = (api.llama_model_default_params)();
            {
                let p = mp.view_mut::<LlamaModelParams>();
                // defaults at the mirrored offsets, or the library's struct layout moved
                if p.split_mode != 1
                    || p.main_gpu != 0
                    || p.vocab_only
                    || p.check_tensors
                    || !p.use_extra_bufts
                {
                    bail!("llama_model_params layout does not match this libllama");
                }
                p.n_gpu_layers = n_gpu_layers;
            }
            let cpath = CString::new(path)?;
            let model = (api.llama_model_load_from_file)(cpath.as_ptr(), mp);
            if model.is_null() {
                bail!("failed to load the text encoder {path}");
            }
            let mut enc = Box::new(TextEncoder {
                api,
                model,
                ctx: ptr::null_mut(),
                n_ctx,
                n_threads,
                flash,
                capture: ptr::null_mut(),
            });
            enc.ensure_context()?;
            Ok(enc)
        }
    }

    /// Creates the llama context if there is none.
    fn ensure_context(&mut self) -> Result<()> {
        if !self.ctx.is_null() {
            return Ok(());
        }
        let api = self.api;
        unsafe {
            let mut cp = (api.llama_context_default_params)();
            {
                let p = cp.view_mut::<LlamaContextParams>();
                if p.n_batch != 2048
                    || p.n_ubatch != 512
                    || p.flash_attn_type != ffi::LLAMA_FLASH_ATTN_TYPE_AUTO
                    || p.type_k != ffi::ty::F16
                    || !p.offload_kqv
                    || !p.op_offload
                    || !p.swa_full
                    || p.kv_unified
                {
                    bail!("llama_context_params layout does not match this libllama");
                }
                p.n_ctx = self.n_ctx;
                p.n_batch = self.n_ctx;
                p.n_ubatch = self.n_ctx;
                p.n_seq_max = 1;
                p.n_threads = self.n_threads;
                p.n_threads_batch = self.n_threads;
                // embeddings mode, every token an output: final norm on all tokens, no lm_head
                p.embeddings = true;
                p.pooling_type = ffi::LLAMA_POOLING_TYPE_NONE;
                p.no_perf = true;
                p.flash_attn_type = if self.flash {
                    ffi::LLAMA_FLASH_ATTN_TYPE_AUTO
                } else {
                    ffi::LLAMA_FLASH_ATTN_TYPE_DISABLED
                };
                p.cb_eval = Some(cb_eval);
                // stable: the encoder is boxed
                p.cb_eval_user_data = self as *const TextEncoder as *mut c_void;
            }
            let ctx = (api.llama_init_from_model)(self.model, cp);
            if ctx.is_null() {
                bail!("failed to create the text encoder context");
            }
            self.ctx = ctx;
        }
        Ok(())
    }

    /// Frees the context; the next `encode` or `enhance` recreates it.
    pub fn release_context(&mut self) {
        if !self.ctx.is_null() {
            unsafe { (self.api.llama_free)(self.ctx) };
            self.ctx = ptr::null_mut();
        }
    }

    fn vocab(&self) -> *const ffi::LlamaVocab {
        unsafe { (self.api.llama_model_get_vocab)(self.model) }
    }

    fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
        let ctext = CString::new(text)?;
        let mut tokens = vec![0i32; text.len() + 16];
        let mut n = unsafe {
            (self.api.llama_tokenize)(
                self.vocab(),
                ctext.as_ptr(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                add_special,
                parse_special,
            )
        };
        if n < 0 {
            tokens.resize((-n) as usize, 0);
            n = unsafe {
                (self.api.llama_tokenize)(
                    self.vocab(),
                    ctext.as_ptr(),
                    text.len() as i32,
                    tokens.as_mut_ptr(),
                    tokens.len() as i32,
                    add_special,
                    parse_special,
                )
            };
        }
        if n < 0 {
            bail!("failed to tokenize");
        }
        tokens.truncate(n as usize);
        Ok(tokens)
    }

    /// Runs the model and returns the packed, per-token RMS-normalized hidden
    /// states `out[t * (H * L) + h * L + l]`, with `(n_tokens, n_hidden, n_states)`.
    pub fn encode(&mut self, prompt: &str, max_tokens: i64) -> Result<(Vec<f32>, i64, i64, i64)> {
        self.ensure_context()?;
        let mut tokens = self.tokenize(prompt.trim(), true, false)?;
        if tokens.is_empty() {
            bail!("empty prompt");
        }
        if tokens.len() as i64 > max_tokens {
            eprintln!(
                "[llmman] mediagen: prompt has {} tokens, truncating to {max_tokens}",
                tokens.len()
            );
            tokens.truncate(max_tokens as usize);
        }
        let n = tokens.len();
        let n_layers = unsafe { (self.api.llama_model_n_layer)(self.model) };
        let mut cap = Box::new(Capture {
            n_layers,
            n_tokens: n as i64,
            n_hidden: 0,
            states: vec![Vec::new(); n_layers as usize + 1],
            got: vec![false; n_layers as usize + 1],
        });
        unsafe {
            (self.api.llama_memory_clear)((self.api.llama_get_memory)(self.ctx), true);
            self.capture = &mut *cap;
            let batch = (self.api.llama_batch_init)(n as i32, 0, 1);
            for (i, &tok) in tokens.iter().enumerate() {
                *batch.token.add(i) = tok;
                *batch.pos.add(i) = i as i32;
                *batch.n_seq_id.add(i) = 1;
                **batch.seq_id.add(i) = 0;
                *batch.logits.add(i) = 1;
            }
            let mut batch = batch;
            batch.n_tokens = n as i32;
            let ret = (self.api.llama_decode)(self.ctx, batch);
            self.capture = ptr::null_mut();
            (self.api.llama_batch_free)(batch);
            if ret != 0 {
                bail!("llama_decode failed with {ret}");
            }
            // last hidden state: the final norm output, i.e. the per-token embeddings
            let h = if cap.n_hidden > 0 {
                cap.n_hidden
            } else {
                (self.api.llama_model_n_embd)(self.model) as i64
            };
            let last = cap.states.last_mut().unwrap();
            last.resize(n * h as usize, 0.0);
            for i in 0..n {
                let e = (self.api.llama_get_embeddings_ith)(self.ctx, i as i32);
                if e.is_null() {
                    bail!("no embeddings for token {i}");
                }
                last[i * h as usize..(i + 1) * h as usize]
                    .copy_from_slice(std::slice::from_raw_parts(e, h as usize));
            }
            cap.n_hidden = h;
            *cap.got.last_mut().unwrap() = true;
        }
        if let Some(i) = cap.got.iter().position(|g| !g) {
            bail!("hidden state {i} was not captured");
        }
        let (h, l) = (cap.n_hidden as usize, n_layers as usize + 1);
        let mut packed = vec![0f32; n * h * l];
        for t in 0..n {
            let dst = &mut packed[t * h * l..(t + 1) * h * l];
            for (li, state) in cap.states.iter().enumerate() {
                let src = &state[t * h..(t + 1) * h];
                let ss: f64 = src.iter().map(|&v| (v as f64) * (v as f64)).sum();
                let r = (1.0 / (ss / h as f64 + 1e-6).sqrt()) as f32;
                for (hi, &v) in src.iter().enumerate() {
                    dst[hi * l + li] = v * r;
                }
            }
        }
        Ok((packed, n as i64, h as i64, l as i64))
    }

    /// Expands a short prompt into a detailed caption with the text model.
    pub fn enhance(&mut self, prompt: &str, seed: u32, max_new_tokens: usize) -> Result<String> {
        self.ensure_context()?;
        let api = self.api;
        let user = CString::new(format!("user prompt: {prompt}"))?;
        let system = CString::new(T2V_SYSTEM_PROMPT)?;
        let msgs = [
            ffi::LlamaChatMessage {
                role: c"system".as_ptr(),
                content: system.as_ptr(),
            },
            ffi::LlamaChatMessage {
                role: c"user".as_ptr(),
                content: user.as_ptr(),
            },
        ];
        let text = unsafe {
            let tmpl = (api.llama_model_chat_template)(self.model, ptr::null());
            let mut buf = vec![0u8; T2V_SYSTEM_PROMPT.len() + prompt.len() + 512];
            let mut n = (api.llama_chat_apply_template)(
                tmpl,
                msgs.as_ptr(),
                2,
                true,
                buf.as_mut_ptr() as *mut _,
                buf.len() as i32,
            );
            if n < 0 {
                bail!("failed to apply the chat template");
            }
            if n as usize > buf.len() {
                buf.resize(n as usize, 0);
                n = (api.llama_chat_apply_template)(
                    tmpl,
                    msgs.as_ptr(),
                    2,
                    true,
                    buf.as_mut_ptr() as *mut _,
                    buf.len() as i32,
                );
            }
            String::from_utf8_lossy(&buf[..n as usize]).into_owned()
        };
        let tokens = self.tokenize(&text, true, true)?;
        if tokens.len() + max_new_tokens > self.n_ctx as usize {
            bail!("prompt too long for enhancement ({} tokens)", tokens.len());
        }
        let vocab = self.vocab();
        let mut gen = Vec::new();
        unsafe {
            // same context as encode(), in logits mode
            (api.llama_set_embeddings)(self.ctx, false);
            (api.llama_memory_clear)((api.llama_get_memory)(self.ctx), true);
            let mut batch = (api.llama_batch_init)(self.n_ctx as i32, 0, 1);
            for (i, &tok) in tokens.iter().enumerate() {
                *batch.token.add(i) = tok;
                *batch.pos.add(i) = i as i32;
                *batch.n_seq_id.add(i) = 1;
                **batch.seq_id.add(i) = 0;
                *batch.logits.add(i) = (i + 1 == tokens.len()) as i8;
            }
            batch.n_tokens = tokens.len() as i32;
            let mut ok = (api.llama_decode)(self.ctx, batch) == 0;
            let smpl = (api.llama_sampler_chain_init)((api.llama_sampler_chain_default_params)());
            (api.llama_sampler_chain_add)(smpl, (api.llama_sampler_init_temp)(0.7));
            (api.llama_sampler_chain_add)(smpl, (api.llama_sampler_init_dist)(seed));
            let mut piece = [0u8; 256];
            for pos in (tokens.len() as i32..).take(max_new_tokens) {
                if !ok {
                    break;
                }
                let id = (api.llama_sampler_sample)(smpl, self.ctx, -1);
                if (api.llama_vocab_is_eog)(vocab, id) {
                    break;
                }
                let np = (api.llama_token_to_piece)(
                    vocab,
                    id,
                    piece.as_mut_ptr() as *mut _,
                    piece.len() as i32,
                    0,
                    true,
                );
                if np > 0 {
                    gen.extend_from_slice(&piece[..np as usize]);
                }
                *batch.token = id;
                *batch.pos = pos;
                *batch.n_seq_id = 1;
                **batch.seq_id = 0;
                *batch.logits = 1;
                batch.n_tokens = 1;
                ok = (api.llama_decode)(self.ctx, batch) == 0;
            }
            (api.llama_sampler_free)(smpl);
            (api.llama_batch_free)(batch);
            // back to embeddings mode
            (api.llama_set_embeddings)(self.ctx, true);
            (api.llama_memory_clear)((api.llama_get_memory)(self.ctx), true);
            if !ok {
                bail!("llama_decode failed during prompt enhancement");
            }
        }
        let out = clean_response(&String::from_utf8_lossy(&gen));
        Ok(if out.is_empty() {
            prompt.to_string()
        } else {
            out
        })
    }
}

/// ltx-pipelines `clean_response`: ascii quotes and dashes, drop leading non-letters.
fn clean_response(s: &str) -> String {
    let s: String = s
        .chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{2032}' => '\'',
            '\u{201c}' | '\u{201d}' => '"',
            '\u{2014}' | '\u{2013}' | '\u{2212}' => '-',
            '\u{a0}' => ' ',
            c => c,
        })
        .collect();
    let start = s.find(|c: char| c.is_alphabetic()).unwrap_or(0);
    s[start..].trim_end().to_string()
}

fn connector(
    g: &Graph,
    model: &Model,
    prefix: &str,
    mut x: Tensor,
    reg_idx: Option<Tensor>,
    rope: Rope,
    n_heads: i64,
    flash: bool,
) -> Result<Tensor> {
    let hp = &model.hp;
    let w = &model.dit;
    let ctx = &g.ctx;
    if let Some(idx) = reg_idx {
        // learnable registers fill the padded tail
        let regs = ctx.get_rows(w.get(&format!("{prefix}.learnable_registers"))?, idx);
        x = ctx.concat(x, regs, 1);
    }
    for il in 0..hp.connector_layers {
        let bp = format!("{prefix}.transformer_1d_blocks.{il}");
        let aw = AttnWeights::load(w, &format!("{bp}.attn1"))?;
        let h = rms(ctx, x, 1e-6, None);
        let a = attention(ctx, &aw, h, None, Some(rope), Some(rope), n_heads, flash);
        x = ctx.add(x, a);
        let h = rms(ctx, x, 1e-6, None);
        x = ctx.add(x, feed_forward(ctx, w, &format!("{bp}.ff"), h)?);
    }
    Ok(rms(ctx, x, 1e-6, None))
}

/// Text projection and connectors for both modalities.
pub fn build_text_cond(
    model: &Model,
    be: &Backend,
    packed: &[f32],
    n_tokens: i64,
    n_hidden: i64,
    n_states: i64,
    flash: bool,
) -> Result<TextCond> {
    let hp = &model.hp;
    let n_tot = hp.text_max_tokens;
    if n_tokens > n_tot {
        bail!("{n_tokens} tokens exceed the maximum of {n_tot}");
    }
    let mut g = Graph::new(be.api, 4096)?;
    let inp = g.input_f32(&[n_hidden * n_states, n_tokens], packed, "text_hidden");
    let reg_idx: Vec<i32> = (n_tokens..n_tot)
        .map(|p| (p % hp.connector_regs) as i32)
        .collect();
    let t_reg =
        (!reg_idx.is_empty()).then(|| g.input_i32(&[reg_idx.len() as i64], &reg_idx, "reg_idx"));
    let pos: Vec<f64> = (0..n_tot)
        .map(|p| p as f64 / hp.connector_max_pos)
        .collect();
    let (vc, vs) = rope_tables(&pos, n_tot, 1, hp.v_dim, hp.rope_theta);
    let (ac, as_) = rope_tables(&pos, n_tot, 1, hp.a_dim, hp.rope_theta);
    let v_rope = Rope {
        cos: g.input_f32(&[hp.v_dim / 2, n_tot], &vc, "v_cos"),
        sin: g.input_f32(&[hp.v_dim / 2, n_tot], &vs, "v_sin"),
    };
    let a_rope = Rope {
        cos: g.input_f32(&[hp.a_dim / 2, n_tot], &ac, "a_cos"),
        sin: g.input_f32(&[hp.a_dim / 2, n_tot], &as_, "a_sin"),
    };
    let ctx = &g.ctx;
    let v_in = ctx.scale(inp, (hp.v_dim as f32 / hp.text_hidden_size as f32).sqrt());
    let a_in = ctx.scale(inp, (hp.a_dim as f32 / hp.text_hidden_size as f32).sqrt());
    let v = ctx.linear(
        v_in,
        model.tp("text_embedding_projection.video_aggregate_embed.weight")?,
        model.tp_opt("text_embedding_projection.video_aggregate_embed.bias"),
    );
    let a = ctx.linear(
        a_in,
        model.tp("text_embedding_projection.audio_aggregate_embed.weight")?,
        model.tp_opt("text_embedding_projection.audio_aggregate_embed.bias"),
    );
    let v = connector(
        &g,
        model,
        "video_embeddings_connector",
        v,
        t_reg,
        v_rope,
        hp.v_heads,
        flash,
    )?;
    let a = connector(
        &g,
        model,
        "audio_embeddings_connector",
        a,
        t_reg,
        a_rope,
        hp.a_heads,
        flash,
    )?;
    g.mark_output(v);
    g.mark_output(a);
    g.compute(be)?;
    Ok(TextCond {
        n_tokens: n_tot,
        video: g.output_f32(v),
        audio: g.output_f32(a),
    })
}

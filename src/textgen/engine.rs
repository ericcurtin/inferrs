//! Text generation over libllama: one model, one context, requests served
//! from a dedicated thread. Context size and GPU offload are fitted to the
//! memory the devices report, so a load cannot run the machine out of memory.

use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use tokio::sync::mpsc as tmpsc;

use crate::mediagen::ffi::{self, Api, LlamaBatch, LlamaContextParams, LlamaContextT, LlamaModel, LlamaModelParams, LlamaVocab};

/// Sampling parameters; llama-server's defaults.
#[derive(Clone, Debug)]
pub struct Sampling {
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: Option<u32>,
    pub stop: Vec<String>,
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling {
            max_tokens: -1,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
            stop: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    Stop,
    Length,
}

#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    /// Prompt tokens reused from the previous request's cache.
    pub cached_n: usize,
}

pub enum Event {
    Token { id: i32, text: String },
    Done { finish: Finish, timings: Timings },
    Error(String),
}

pub struct Request {
    pub tokens: Vec<i32>,
    pub sampling: Sampling,
    pub tx: tmpsc::UnboundedSender<Event>,
    pub cancel: Arc<AtomicBool>,
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub path: String,
    pub desc: String,
    pub n_params: u64,
    pub size: u64,
    pub n_ctx: u32,
    pub n_ctx_train: i32,
    pub n_gpu_layers: i32,
    pub chat_template: Option<String>,
    pub bos: String,
    pub eos: String,
    pub add_bos: bool,
    pub bos_id: i32,
}

/// Handle to the engine thread.
#[derive(Clone)]
pub struct Handle {
    tx: mpsc::Sender<Request>,
    pub info: Arc<ModelInfo>,
    api: &'static Api,
    vocab: *const LlamaVocab,
}

unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

impl Handle {
    pub fn generate(&self, tokens: Vec<i32>, sampling: Sampling) -> (tmpsc::UnboundedReceiver<Event>, Arc<AtomicBool>) {
        let (tx, rx) = tmpsc::unbounded_channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let req = Request { tokens, sampling, tx: tx.clone(), cancel: cancel.clone() };
        if self.tx.send(req).is_err() {
            let _ = tx.send(Event::Error("engine stopped".into()));
        }
        (rx, cancel)
    }

    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
        tokenize(self.api, self.vocab, text, add_special, parse_special)
    }

    pub fn detokenize(&self, tokens: &[i32], special: bool) -> String {
        let mut buf = vec![0u8; tokens.len() * 8 + 16];
        let mut n = unsafe {
            (self.api.llama_detokenize)(self.vocab, tokens.as_ptr(), tokens.len() as i32, buf.as_mut_ptr() as *mut _, buf.len() as i32, false, special)
        };
        if n < 0 {
            buf.resize((-n) as usize, 0);
            n = unsafe {
                (self.api.llama_detokenize)(self.vocab, tokens.as_ptr(), tokens.len() as i32, buf.as_mut_ptr() as *mut _, buf.len() as i32, false, special)
            };
        }
        String::from_utf8_lossy(&buf[..n.max(0) as usize]).into_owned()
    }
}

fn tokenize(api: &Api, vocab: *const LlamaVocab, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
    let ctext = CString::new(text)?;
    let mut tokens = vec![0i32; text.len() + 16];
    let mut n = unsafe {
        (api.llama_tokenize)(vocab, ctext.as_ptr(), text.len() as i32, tokens.as_mut_ptr(), tokens.len() as i32, add_special, parse_special)
    };
    if n < 0 {
        tokens.resize((-n) as usize, 0);
        n = unsafe {
            (api.llama_tokenize)(vocab, ctext.as_ptr(), text.len() as i32, tokens.as_mut_ptr(), tokens.len() as i32, add_special, parse_special)
        };
    }
    if n < 0 {
        bail!("failed to tokenize");
    }
    tokens.truncate(n as usize);
    Ok(tokens)
}

fn meta(api: &Api, model: *const LlamaModel, key: &str) -> Option<String> {
    let ckey = CString::new(key).ok()?;
    let mut buf = vec![0u8; 4096];
    let n = unsafe { (api.llama_model_meta_val_str)(model, ckey.as_ptr(), buf.as_mut_ptr() as *mut _, buf.len()) };
    (n >= 0).then(|| String::from_utf8_lossy(&buf[..n as usize]).into_owned())
}

/// Free and total bytes of the first GPU device, if any.
fn gpu_memory(api: &Api) -> Option<(usize, usize)> {
    unsafe {
        for i in 0..(api.ggml_backend_dev_count)() {
            let dev = (api.ggml_backend_dev_get)(i);
            let ty = (api.ggml_backend_dev_type)(dev);
            if ty == ffi::GGML_BACKEND_DEVICE_TYPE_GPU || ty == ffi::GGML_BACKEND_DEVICE_TYPE_IGPU {
                let (mut free, mut total) = (0usize, 0usize);
                (api.ggml_backend_dev_memory)(dev, &mut free, &mut total);
                return Some((free, total));
            }
        }
    }
    None
}

/// The vocab outlives the model it belongs to only as long as the engine
/// thread does, which owns the model.
struct VocabPtr(*const LlamaVocab);
unsafe impl Send for VocabPtr {}

/// The engine thread's state.
struct Engine {
    api: &'static Api,
    model: *mut LlamaModel,
    ctx: *mut LlamaContextT,
    vocab: *const LlamaVocab,
    n_ctx: u32,
    n_batch: u32,
    batch: LlamaBatch,
    /// Tokens currently in the KV cache of sequence 0.
    cache: Vec<i32>,
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            (self.api.llama_batch_free)(self.batch);
            (self.api.llama_free)(self.ctx);
            (self.api.llama_model_free)(self.model);
        }
    }
}

pub struct LoadOpts {
    /// Requested context; `0` means the model's training context.
    pub n_ctx: u32,
    pub n_threads: i32,
    pub flash_attn: bool,
}

/// Loads the model and starts the engine thread.
pub fn spawn(api: &'static Api, path: &str, opts: LoadOpts) -> Result<Handle> {
    let (tx, rx) = mpsc::channel::<Request>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(Arc<ModelInfo>, VocabPtr)>>();
    let path = path.to_string();
    std::thread::Builder::new()
        .name("textgen".into())
        .spawn(move || match Engine::load(api, &path, &opts) {
            Ok((mut engine, info)) => {
                let _ = ready_tx.send(Ok((Arc::new(info), VocabPtr(engine.vocab))));
                while let Ok(req) = rx.recv() {
                    engine.serve(req);
                }
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        })?;
    let (info, vocab) = ready_rx.recv().map_err(|_| anyhow!("engine thread died"))??;
    Ok(Handle { tx, info, api, vocab: vocab.0 })
}

impl Engine {
    fn load(api: &'static Api, path: &str, opts: &LoadOpts) -> Result<(Engine, ModelInfo)> {
        let cpath = CString::new(path)?;
        let gpu = gpu_memory(api);
        // 10% of the device, or 512 MiB, whichever is larger, stays free
        let budget = gpu.map(|(free, total)| free.saturating_sub((total / 10).max(512 << 20)));
        let mut n_gpu_layers = 999;
        let mut model = ptr::null_mut();
        for attempt in 0..4 {
            let mut mp = unsafe { (api.llama_model_default_params)() };
            {
                let p = mp.view_mut::<LlamaModelParams>();
                if p.split_mode != 1 || p.main_gpu != 0 || p.vocab_only || p.check_tensors || !p.use_extra_bufts {
                    bail!("llama_model_params layout does not match this libllama");
                }
                p.n_gpu_layers = n_gpu_layers;
            }
            model = unsafe { (api.llama_model_load_from_file)(cpath.as_ptr(), mp) };
            if !model.is_null() {
                break;
            }
            if attempt == 0 && budget.is_some() {
                // probe the size with a CPU-only load next time round
                n_gpu_layers = 0;
            } else {
                bail!("failed to load {path}");
            }
        }
        let n_layer = unsafe { (api.llama_model_n_layer)(model) };
        let size = unsafe { (api.llama_model_size)(model) };
        if n_gpu_layers == 0 {
            if let Some(b) = budget {
                // as many layers as fit, leaving room for the context
                let fit = ((b as f64 * 0.8 / size as f64) * n_layer as f64) as i32;
                n_gpu_layers = fit.clamp(0, n_layer);
                unsafe { (api.llama_model_free)(model) };
                let mut mp = unsafe { (api.llama_model_default_params)() };
                mp.view_mut::<LlamaModelParams>().n_gpu_layers = n_gpu_layers;
                model = unsafe { (api.llama_model_load_from_file)(cpath.as_ptr(), mp) };
                if model.is_null() {
                    bail!("failed to load {path} with {n_gpu_layers} GPU layers");
                }
            }
        }
        let vocab = unsafe { (api.llama_model_get_vocab)(model) };
        let n_ctx_train = unsafe { (api.llama_model_n_ctx_train)(model) };
        let n_head_kv = unsafe { (api.llama_model_n_head_kv)(model) }.max(1);
        let n_embd = unsafe { (api.llama_model_n_embd)(model) };
        let arch = meta(api, model, "general.architecture").unwrap_or_default();
        let head_dim = meta(api, model, &format!("{arch}.attention.key_length"))
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or_else(|| {
                let n_head = meta(api, model, &format!("{arch}.attention.head_count")).and_then(|v| v.parse::<i32>().ok()).unwrap_or(1);
                n_embd / n_head.max(1)
            });
        // f16 K and V per token, every layer; an overestimate for SWA and hybrid models
        let kv_per_token = n_layer as u64 * 2 * n_head_kv as u64 * head_dim as u64 * 2;

        let mut n_ctx = if opts.n_ctx == 0 { n_ctx_train as u32 } else { opts.n_ctx.min(n_ctx_train as u32) };
        if let Some(b) = budget {
            let on_gpu = size * n_gpu_layers.min(n_layer) as u64 / n_layer.max(1) as u64;
            let compute = (1u64 << 30).max(size / 10);
            let for_kv = (b as u64).saturating_sub(on_gpu + compute);
            let max_ctx = (for_kv / kv_per_token.max(1)) as u32 / 256 * 256;
            if max_ctx < n_ctx {
                eprintln!("[llmman] textgen: context {n_ctx} does not fit next to the weights, using {}", max_ctx.max(4096));
                n_ctx = max_ctx.max(4096);
            }
        }
        let n_batch = 2048u32.min(n_ctx);

        let mut ctx: *mut LlamaContextT = ptr::null_mut();
        while ctx.is_null() {
            let mut cp = unsafe { (api.llama_context_default_params)() };
            {
                let p = cp.view_mut::<LlamaContextParams>();
                if p.n_batch != 2048 || p.n_ubatch != 512 || p.flash_attn_type != ffi::LLAMA_FLASH_ATTN_TYPE_AUTO || p.type_k != ffi::ty::F16 || !p.offload_kqv || !p.op_offload || !p.swa_full || p.kv_unified {
                    bail!("llama_context_params layout does not match this libllama");
                }
                p.n_ctx = n_ctx;
                p.n_batch = n_batch;
                p.n_ubatch = 512.min(n_batch);
                p.n_seq_max = 1;
                p.n_threads = opts.n_threads;
                p.n_threads_batch = opts.n_threads;
                p.no_perf = true;
                p.flash_attn_type = if opts.flash_attn { ffi::LLAMA_FLASH_ATTN_TYPE_AUTO } else { ffi::LLAMA_FLASH_ATTN_TYPE_DISABLED };
            }
            ctx = unsafe { (api.llama_init_from_model)(model, cp) };
            if ctx.is_null() {
                if n_ctx <= 4096 {
                    unsafe { (api.llama_model_free)(model) };
                    bail!("failed to create a context of {n_ctx} tokens for {path}");
                }
                n_ctx = (n_ctx / 2).max(4096);
                eprintln!("[llmman] textgen: context allocation failed, retrying with {n_ctx}");
            }
        }
        let n_ctx = unsafe { (api.llama_n_ctx)(ctx) };
        let batch = unsafe { (api.llama_batch_init)(n_batch as i32, 0, 1) };

        let mut desc = vec![0u8; 256];
        let n = unsafe { (api.llama_model_desc)(model, desc.as_mut_ptr() as *mut _, desc.len()) };
        let desc = String::from_utf8_lossy(&desc[..n.max(0) as usize]).into_owned();
        let tmpl = unsafe { (api.llama_model_chat_template)(model, ptr::null()) };
        let chat_template = (!tmpl.is_null()).then(|| unsafe { CStr::from_ptr(tmpl) }.to_string_lossy().into_owned());
        let piece = |tok: i32| -> String {
            let mut buf = [0u8; 64];
            let n = unsafe { (api.llama_token_to_piece)(vocab, tok, buf.as_mut_ptr() as *mut _, buf.len() as i32, 0, true) };
            String::from_utf8_lossy(&buf[..n.max(0) as usize]).into_owned()
        };
        let bos_id = unsafe { (api.llama_vocab_bos)(vocab) };
        let eos_id = unsafe { (api.llama_vocab_eos)(vocab) };
        let info = ModelInfo {
            path: path.to_string(),
            desc,
            n_params: unsafe { (api.llama_model_n_params)(model) },
            size,
            n_ctx,
            n_ctx_train,
            n_gpu_layers,
            chat_template,
            bos: if bos_id >= 0 { piece(bos_id) } else { String::new() },
            eos: if eos_id >= 0 { piece(eos_id) } else { String::new() },
            add_bos: unsafe { (api.llama_vocab_get_add_bos)(vocab) },
            bos_id,
        };
        eprintln!(
            "[llmman] textgen: {} loaded, n_ctx {n_ctx}, {n_gpu_layers} of {n_layer} layers on the GPU",
            info.desc
        );
        Ok((Engine { api, model, ctx, vocab, n_ctx, n_batch, batch, cache: Vec::new() }, info))
    }

    fn decode(&mut self, tokens: &[i32], pos0: i32, logits_last: bool) -> Result<()> {
        let api = self.api;
        for (ci, chunk) in tokens.chunks(self.n_batch as usize).enumerate() {
            let last_chunk = (ci + 1) * self.n_batch as usize >= tokens.len();
            unsafe {
                for (i, &tok) in chunk.iter().enumerate() {
                    *self.batch.token.add(i) = tok;
                    *self.batch.pos.add(i) = pos0 + (ci * self.n_batch as usize + i) as i32;
                    *self.batch.n_seq_id.add(i) = 1;
                    **self.batch.seq_id.add(i) = 0;
                    *self.batch.logits.add(i) = (logits_last && last_chunk && i + 1 == chunk.len()) as i8;
                }
                self.batch.n_tokens = chunk.len() as i32;
                let ret = (api.llama_decode)(self.ctx, self.batch);
                if ret != 0 {
                    bail!("llama_decode failed with {ret}");
                }
            }
        }
        Ok(())
    }

    fn sampler(&self, s: &Sampling) -> *mut ffi::LlamaSampler {
        let api = self.api;
        unsafe {
            let chain = (api.llama_sampler_chain_init)((api.llama_sampler_chain_default_params)());
            let n_vocab = (api.llama_vocab_n_tokens)(self.vocab);
            if s.repeat_penalty != 1.0 || s.frequency_penalty != 0.0 || s.presence_penalty != 0.0 {
                (api.llama_sampler_chain_add)(chain, (api.llama_sampler_init_penalties)(n_vocab, s.repeat_last_n, s.repeat_penalty, s.frequency_penalty, s.presence_penalty));
            }
            if s.temperature <= 0.0 {
                (api.llama_sampler_chain_add)(chain, (api.llama_sampler_init_greedy)());
            } else {
                if s.top_k > 0 {
                    (api.llama_sampler_chain_add)(chain, (api.llama_sampler_init_top_k)(s.top_k));
                }
                (api.llama_sampler_chain_add)(chain, (api.llama_sampler_init_top_p)(s.top_p, 1));
                (api.llama_sampler_chain_add)(chain, (api.llama_sampler_init_min_p)(s.min_p, 1));
                (api.llama_sampler_chain_add)(chain, (api.llama_sampler_init_temp)(s.temperature));
                (api.llama_sampler_chain_add)(chain, (api.llama_sampler_init_dist)(s.seed.unwrap_or_else(crate::mediagen::rand_seed)));
            }
            chain
        }
    }

    fn serve(&mut self, req: Request) {
        let tx = req.tx.clone();
        if let Err(e) = self.run(req) {
            let _ = tx.send(Event::Error(e.to_string()));
        }
    }

    fn run(&mut self, req: Request) -> Result<()> {
        let api = self.api;
        let mut prompt = req.tokens;
        if prompt.is_empty() {
            prompt.push(unsafe { (api.llama_vocab_bos)(self.vocab) });
        }
        let max_prompt = self.n_ctx as usize - 4;
        if prompt.len() > max_prompt {
            bail!("the request ({} tokens) exceeds the available context size ({} tokens)", prompt.len(), self.n_ctx);
        }
        // reuse the cached prefix; the last prompt token is always decoded, for its logits
        let mut common = prompt.iter().zip(&self.cache).take_while(|(a, b)| a == b).count();
        if common == prompt.len() {
            common -= 1;
        }
        unsafe {
            (api.llama_memory_seq_rm)((api.llama_get_memory)(self.ctx), 0, common as i32, -1);
        }
        self.cache.truncate(common);
        let t0 = Instant::now();
        self.decode(&prompt[common..], common as i32, true)?;
        self.cache.extend_from_slice(&prompt[common..]);
        // the backend may still be computing: time the whole prefill
        unsafe { (api.llama_synchronize)(self.ctx) };
        let prompt_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let smpl = self.sampler(&req.sampling);
        let max_tokens = if req.sampling.max_tokens < 0 { i32::MAX } else { req.sampling.max_tokens };
        let mut pending = Vec::new(); // bytes not yet a whole UTF-8 sequence
        let mut text = String::new(); // everything emitted, for stop strings
        let mut finish = Finish::Length;
        let mut n_gen = 0;
        let t1 = Instant::now();
        let stop_max = req.sampling.stop.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut result = Ok(());
        while n_gen < max_tokens && !req.cancel.load(Ordering::Relaxed) {
            if self.cache.len() >= self.n_ctx as usize {
                break;
            }
            let id = unsafe { (api.llama_sampler_sample)(smpl, self.ctx, -1) };
            n_gen += 1;
            if unsafe { (api.llama_vocab_is_eog)(self.vocab, id) } {
                finish = Finish::Stop;
                break;
            }
            let mut buf = [0u8; 256];
            let n = unsafe { (api.llama_token_to_piece)(self.vocab, id, buf.as_mut_ptr() as *mut _, buf.len() as i32, 0, true) };
            pending.extend_from_slice(&buf[..n.max(0) as usize]);
            let piece = take_utf8(&mut pending);
            let mut emit = piece;
            let mut stopped = false;
            if stop_max > 0 {
                let already = text.len();
                text.push_str(&emit);
                if let Some(pos) = req.sampling.stop.iter().filter_map(|s| text.find(s.as_str())).min() {
                    // up to the stop string; nothing if it began in text already sent
                    emit = if pos > already { text[already..pos].to_string() } else { String::new() };
                    stopped = true;
                } else if text.len() > 4 * stop_max {
                    let mut cut = text.len() - 2 * stop_max;
                    while !text.is_char_boundary(cut) {
                        cut -= 1;
                    }
                    text.drain(..cut);
                }
            }
            if !emit.is_empty() && tx_send(&req.tx, Event::Token { id, text: emit }).is_err() {
                break;
            }
            if stopped {
                finish = Finish::Stop;
                break;
            }
            if let Err(e) = self.decode(&[id], self.cache.len() as i32, true) {
                result = Err(e);
                break;
            }
            self.cache.push(id);
        }
        unsafe { (api.llama_sampler_free)(smpl) };
        result?;
        let timings = Timings {
            prompt_n: prompt.len(),
            prompt_ms,
            predicted_n: n_gen as usize,
            predicted_ms: t1.elapsed().as_secs_f64() * 1000.0,
            cached_n: common,
        };
        let _ = tx_send(&req.tx, Event::Done { finish, timings });
        Ok(())
    }
}

fn tx_send(tx: &tmpsc::UnboundedSender<Event>, ev: Event) -> Result<(), ()> {
    tx.send(ev).map_err(|_| ())
}

/// Splits off the longest valid UTF-8 prefix.
fn take_utf8(pending: &mut Vec<u8>) -> String {
    match std::str::from_utf8(pending) {
        Ok(s) => {
            let s = s.to_string();
            pending.clear();
            s
        }
        Err(e) => {
            let valid = e.valid_up_to();
            if e.error_len().is_some() {
                // not a partial sequence: drop the bad byte
                let s = String::from_utf8_lossy(&pending[..valid + 1]).into_owned();
                pending.drain(..valid + 1);
                return s;
            }
            let s = String::from_utf8_lossy(&pending[..valid]).into_owned();
            pending.drain(..valid);
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_is_emitted_only_when_complete() {
        let mut p = "héllo".as_bytes()[..2].to_vec(); // 'h' + first byte of 'é'
        assert_eq!(take_utf8(&mut p), "h");
        p.extend_from_slice(&"héllo".as_bytes()[2..]);
        assert_eq!(take_utf8(&mut p), "éllo");
        assert!(p.is_empty());
    }
}

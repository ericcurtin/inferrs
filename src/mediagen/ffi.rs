//! `dlopen` bindings to the ggml and llama libraries of a llama.cpp
//! release; the GPU backend module is picked at runtime like llama-server
//! does. Only what the diffusion pipeline needs is bound.

#![allow(non_camel_case_types, dead_code)]

use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::Path;

use anyhow::{Context, Result};
use libloading::Library;

pub type GgmlContext = c_void;
pub type GgmlCgraph = c_void;
pub type GgmlBackend = *mut c_void;
pub type GgmlBackendBuft = *mut c_void;
pub type GgmlBackendBuffer = *mut c_void;
pub type GgmlBackendSched = *mut c_void;
pub type GgmlBackendDev = *mut c_void;
pub type GgmlBackendReg = *mut c_void;
pub type GgmlGallocr = *mut c_void;
pub type LlamaModel = c_void;
pub type LlamaContextT = c_void;
pub type LlamaVocab = c_void;
pub type LlamaMemory = *mut c_void;
pub type LlamaSampler = c_void;
pub type LlamaToken = i32;

pub const GGML_MAX_DIMS: usize = 4;
pub const GGML_MAX_NAME: usize = 64;

/// `enum ggml_type` values used here.
pub mod ty {
    pub const F32: u32 = 0;
    pub const F16: u32 = 1;
    pub const I32: u32 = 26;
    pub const BF16: u32 = 30;
    pub const COUNT: u32 = 44;
}

pub const GGML_BACKEND_DEVICE_TYPE_CPU: c_int = 0;
pub const GGML_BACKEND_DEVICE_TYPE_GPU: c_int = 1;
pub const GGML_BACKEND_DEVICE_TYPE_IGPU: c_int = 2;
pub const GGML_BACKEND_BUFFER_USAGE_WEIGHTS: c_int = 1;
pub const GGML_PREC_F32: c_int = 10;
pub const GGML_SCALE_MODE_NEAREST: u32 = 0;
pub const GGML_STATUS_SUCCESS: c_int = 0;
pub const LLAMA_POOLING_TYPE_NONE: c_int = 0;
pub const LLAMA_FLASH_ATTN_TYPE_AUTO: c_int = -1;
pub const LLAMA_FLASH_ATTN_TYPE_DISABLED: c_int = 0;

/// `struct ggml_tensor`, read only (shape, type, strides, name).
#[repr(C)]
pub struct GgmlTensor {
    pub type_: u32,
    pub buffer: *mut c_void,
    pub ne: [i64; GGML_MAX_DIMS],
    pub nb: [usize; GGML_MAX_DIMS],
    pub op: u32,
    pub op_params: [i32; 16],
    pub flags: i32,
    pub src: [*mut GgmlTensor; 10],
    pub view_src: *mut GgmlTensor,
    pub view_offs: usize,
    pub data: *mut c_void,
    pub name: [c_char; GGML_MAX_NAME],
    pub extra: *mut c_void,
    pub padding: [c_char; 8],
}

pub type Tensor = *mut GgmlTensor;

impl GgmlTensor {
    pub fn name(&self) -> String {
        unsafe { CStr::from_ptr(self.name.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }
}

#[repr(C)]
pub struct GgmlInitParams {
    pub mem_size: usize,
    pub mem_buffer: *mut c_void,
    pub no_alloc: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlamaBatch {
    pub n_tokens: i32,
    pub token: *mut LlamaToken,
    pub embd: *mut f32,
    pub pos: *mut i32,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut i32,
    pub logits: *mut i8,
}

#[repr(C)]
pub struct LlamaChatMessage {
    pub role: *const c_char,
    pub content: *const c_char,
}

#[repr(C)]
pub struct LlamaSamplerChainParams {
    pub no_perf: bool,
}

/// By-value parameter structs travel as this oversized blob (aggregates
/// this big go through memory on every ABI we target); known fields are
/// patched through the prefix mirrors below.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct ParamBlob(pub [u8; 1024]);

/// Prefix of `struct llama_model_params`.
#[repr(C)]
pub struct LlamaModelParams {
    pub devices: *mut c_void,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: c_int,
    pub load_mode: c_int,
    pub lazy_mode: c_int,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: *const c_void,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
}

pub type EvalCallback = unsafe extern "C" fn(t: Tensor, ask: bool, user_data: *mut c_void) -> bool;

/// Prefix of `struct llama_context_params`.
#[repr(C)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: c_int,
    pub rope_scaling_type: c_int,
    pub pooling_type: c_int,
    pub attention_type: c_int,
    pub flash_attn_type: c_int,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: Option<EvalCallback>,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: u32,
    pub type_v: u32,
    pub abort_callback: *const c_void,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
}

impl ParamBlob {
    pub fn view_mut<T>(&mut self) -> &mut T {
        assert!(std::mem::size_of::<T>() <= 1024);
        unsafe { &mut *(self.0.as_mut_ptr() as *mut T) }
    }
}

pub type LogCallback =
    unsafe extern "C" fn(level: c_int, text: *const c_char, user_data: *mut c_void);

macro_rules! api {
    ($(fn $name:ident($($arg:ident: $ty:ty),*) $(-> $ret:ty)?;)*
     optional: $(fn $oname:ident($($oarg:ident: $oty:ty),*) $(-> $oret:ty)?;)*) => {
        /// The bound function pointers; `optional` ones may be absent from older releases.
        pub struct Api {
            _libs: Vec<Library>,
            $(pub $name: unsafe extern "C" fn($($arg: $ty),*) $(-> $ret)?,)*
            $(pub $oname: Option<unsafe extern "C" fn($($oarg: $oty),*) $(-> $oret)?>,)*
        }
        impl Api {
            fn bind(libs: Vec<Library>) -> Result<Self> {
                $(let $name = lookup(&libs, stringify!($name))?;)*
                $(let $oname = lookup(&libs, stringify!($oname)).ok();)*
                Ok(Self { _libs: libs, $($name,)* $($oname,)* })
            }
        }
    };
}

api! {
    // ggml-base
    fn ggml_init(params: GgmlInitParams) -> *mut GgmlContext;
    fn ggml_free(ctx: *mut GgmlContext);
    fn ggml_tensor_overhead() -> usize;
    fn ggml_graph_overhead_custom(size: usize, grads: bool) -> usize;
    fn ggml_new_tensor(ctx: *mut GgmlContext, type_: u32, n_dims: c_int, ne: *const i64) -> Tensor;
    fn ggml_set_name(t: Tensor, name: *const c_char) -> Tensor;
    fn ggml_set_input(t: Tensor);
    fn ggml_set_output(t: Tensor);
    fn ggml_nbytes(t: *const GgmlTensor) -> usize;
    fn ggml_nelements(t: *const GgmlTensor) -> i64;
    fn ggml_type_size(type_: u32) -> usize;
    fn ggml_blck_size(type_: u32) -> i64;
    fn ggml_row_size(type_: u32, ne: i64) -> usize;
    fn ggml_is_contiguous(t: *const GgmlTensor) -> bool;
    fn ggml_new_graph_custom(ctx: *mut GgmlContext, size: usize, grads: bool) -> *mut GgmlCgraph;
    fn ggml_build_forward_expand(gf: *mut GgmlCgraph, t: Tensor);
    fn ggml_graph_n_nodes(gf: *mut GgmlCgraph) -> c_int;
    fn ggml_fp32_to_fp16_row(x: *const f32, y: *mut u16, n: i64);
    fn ggml_fp16_to_fp32_row(x: *const u16, y: *mut f32, n: i64);
    fn ggml_bf16_to_fp32_row(x: *const u16, y: *mut f32, n: i64);
    // ops
    fn ggml_add(ctx: *mut GgmlContext, a: Tensor, b: Tensor) -> Tensor;
    fn ggml_sub(ctx: *mut GgmlContext, a: Tensor, b: Tensor) -> Tensor;
    fn ggml_mul(ctx: *mut GgmlContext, a: Tensor, b: Tensor) -> Tensor;
    fn ggml_scale(ctx: *mut GgmlContext, a: Tensor, s: f32) -> Tensor;
    fn ggml_mul_mat(ctx: *mut GgmlContext, a: Tensor, b: Tensor) -> Tensor;
    fn ggml_rms_norm(ctx: *mut GgmlContext, a: Tensor, eps: f32) -> Tensor;
    fn ggml_norm(ctx: *mut GgmlContext, a: Tensor, eps: f32) -> Tensor;
    fn ggml_silu(ctx: *mut GgmlContext, a: Tensor) -> Tensor;
    fn ggml_gelu(ctx: *mut GgmlContext, a: Tensor) -> Tensor;
    fn ggml_sigmoid(ctx: *mut GgmlContext, a: Tensor) -> Tensor;
    fn ggml_sin(ctx: *mut GgmlContext, a: Tensor) -> Tensor;
    fn ggml_sqr(ctx: *mut GgmlContext, a: Tensor) -> Tensor;
    fn ggml_sqrt(ctx: *mut GgmlContext, a: Tensor) -> Tensor;
    fn ggml_log(ctx: *mut GgmlContext, a: Tensor) -> Tensor;
    fn ggml_clamp(ctx: *mut GgmlContext, a: Tensor, min: f32, max: f32) -> Tensor;
    fn ggml_cpy(ctx: *mut GgmlContext, a: Tensor, b: Tensor) -> Tensor;
    fn ggml_reshape_2d(ctx: *mut GgmlContext, a: Tensor, ne0: i64, ne1: i64) -> Tensor;
    fn ggml_reshape_3d(ctx: *mut GgmlContext, a: Tensor, ne0: i64, ne1: i64, ne2: i64) -> Tensor;
    fn ggml_reshape_4d(ctx: *mut GgmlContext, a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64) -> Tensor;
    fn ggml_view_2d(ctx: *mut GgmlContext, a: Tensor, ne0: i64, ne1: i64, nb1: usize, offset: usize) -> Tensor;
    fn ggml_view_3d(ctx: *mut GgmlContext, a: Tensor, ne0: i64, ne1: i64, ne2: i64, nb1: usize, nb2: usize, offset: usize) -> Tensor;
    fn ggml_view_4d(ctx: *mut GgmlContext, a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64, nb1: usize, nb2: usize, nb3: usize, offset: usize) -> Tensor;
    fn ggml_permute(ctx: *mut GgmlContext, a: Tensor, a0: c_int, a1: c_int, a2: c_int, a3: c_int) -> Tensor;
    fn ggml_transpose(ctx: *mut GgmlContext, a: Tensor) -> Tensor;
    fn ggml_cont(ctx: *mut GgmlContext, a: Tensor) -> Tensor;
    fn ggml_cont_2d(ctx: *mut GgmlContext, a: Tensor, ne0: i64, ne1: i64) -> Tensor;
    fn ggml_concat(ctx: *mut GgmlContext, a: Tensor, b: Tensor, dim: c_int) -> Tensor;
    fn ggml_cast(ctx: *mut GgmlContext, a: Tensor, type_: u32) -> Tensor;
    fn ggml_get_rows(ctx: *mut GgmlContext, a: Tensor, b: Tensor) -> Tensor;
    fn ggml_repeat(ctx: *mut GgmlContext, a: Tensor, b: Tensor) -> Tensor;
    fn ggml_pad(ctx: *mut GgmlContext, a: Tensor, p0: c_int, p1: c_int, p2: c_int, p3: c_int) -> Tensor;
    fn ggml_pad_ext(ctx: *mut GgmlContext, a: Tensor, lp0: c_int, rp0: c_int, lp1: c_int, rp1: c_int, lp2: c_int, rp2: c_int, lp3: c_int, rp3: c_int) -> Tensor;
    fn ggml_conv_1d(ctx: *mut GgmlContext, a: Tensor, b: Tensor, s0: c_int, p0: c_int, d0: c_int) -> Tensor;
    fn ggml_conv_2d(ctx: *mut GgmlContext, a: Tensor, b: Tensor, s0: c_int, s1: c_int, p0: c_int, p1: c_int, d0: c_int, d1: c_int) -> Tensor;
    fn ggml_conv_transpose_1d(ctx: *mut GgmlContext, a: Tensor, b: Tensor, s0: c_int, p0: c_int, d0: c_int) -> Tensor;
    fn ggml_interpolate(ctx: *mut GgmlContext, a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64, mode: u32) -> Tensor;
    fn ggml_timestep_embedding(ctx: *mut GgmlContext, timesteps: Tensor, dim: c_int, max_period: c_int) -> Tensor;
    fn ggml_flash_attn_ext(ctx: *mut GgmlContext, q: Tensor, k: Tensor, v: Tensor, mask: Tensor, scale: f32, max_bias: f32, logit_softcap: f32) -> Tensor;
    fn ggml_flash_attn_ext_set_prec(a: Tensor, prec: c_int);
    fn ggml_soft_max_ext(ctx: *mut GgmlContext, a: Tensor, mask: Tensor, scale: f32, max_bias: f32) -> Tensor;
    // backend
    fn ggml_backend_load_all_from_path(dir: *const c_char);
    fn ggml_backend_init_by_type(type_: c_int, params: *const c_char) -> GgmlBackend;
    fn ggml_backend_free(backend: GgmlBackend);
    fn ggml_backend_name(backend: GgmlBackend) -> *const c_char;
    fn ggml_backend_get_default_buffer_type(backend: GgmlBackend) -> GgmlBackendBuft;
    fn ggml_backend_graph_compute(backend: GgmlBackend, gf: *mut GgmlCgraph) -> c_int;
    fn ggml_backend_get_device(backend: GgmlBackend) -> GgmlBackendDev;
    fn ggml_backend_dev_memory(dev: GgmlBackendDev, free: *mut usize, total: *mut usize);
    fn ggml_gallocr_new(buft: GgmlBackendBuft) -> GgmlGallocr;
    fn ggml_gallocr_free(galloc: GgmlGallocr);
    fn ggml_gallocr_reserve(galloc: GgmlGallocr, gf: *mut GgmlCgraph) -> bool;
    fn ggml_gallocr_get_buffer_size(galloc: GgmlGallocr, buffer_id: c_int) -> usize;
    fn ggml_backend_dev_backend_reg(dev: GgmlBackendDev) -> GgmlBackendReg;
    fn ggml_backend_reg_get_proc_address(reg: GgmlBackendReg, name: *const c_char) -> *mut c_void;
    fn ggml_backend_alloc_ctx_tensors_from_buft(ctx: *mut GgmlContext, buft: GgmlBackendBuft) -> GgmlBackendBuffer;
    fn ggml_backend_buffer_set_usage(buf: GgmlBackendBuffer, usage: c_int);
    fn ggml_backend_buffer_free(buf: GgmlBackendBuffer);
    fn ggml_backend_buffer_get_size(buf: GgmlBackendBuffer) -> usize;
    fn ggml_backend_tensor_set(t: Tensor, data: *const c_void, offset: usize, size: usize);
    fn ggml_backend_tensor_get(t: *const GgmlTensor, data: *mut c_void, offset: usize, size: usize);
    fn ggml_backend_sched_new(backends: *mut GgmlBackend, bufts: *mut GgmlBackendBuft, n: c_int, graph_size: usize, parallel: bool, op_offload: bool) -> GgmlBackendSched;
    fn ggml_backend_sched_free(sched: GgmlBackendSched);
    fn ggml_backend_sched_reset(sched: GgmlBackendSched);
    fn ggml_backend_sched_alloc_graph(sched: GgmlBackendSched, gf: *mut GgmlCgraph) -> bool;
    fn ggml_backend_sched_graph_compute(sched: GgmlBackendSched, gf: *mut GgmlCgraph) -> c_int;
    // llama
    fn llama_backend_init();
    fn llama_log_set(cb: Option<LogCallback>, user_data: *mut c_void);
    fn llama_model_default_params() -> ParamBlob;
    fn llama_model_load_from_file(path: *const c_char, params: ParamBlob) -> *mut LlamaModel;
    fn llama_model_free(model: *mut LlamaModel);
    fn llama_model_get_vocab(model: *const LlamaModel) -> *const LlamaVocab;
    fn llama_model_n_layer(model: *const LlamaModel) -> i32;
    fn llama_model_n_embd(model: *const LlamaModel) -> i32;
    fn llama_model_chat_template(model: *const LlamaModel, name: *const c_char) -> *const c_char;
    fn llama_context_default_params() -> ParamBlob;
    fn llama_init_from_model(model: *mut LlamaModel, params: ParamBlob) -> *mut LlamaContextT;
    fn llama_free(ctx: *mut LlamaContextT);
    fn llama_get_memory(ctx: *const LlamaContextT) -> LlamaMemory;
    fn llama_memory_clear(mem: LlamaMemory, data: bool);
    fn llama_set_embeddings(ctx: *mut LlamaContextT, embeddings: bool);
    fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> LlamaBatch;
    fn llama_batch_free(batch: LlamaBatch);
    fn llama_decode(ctx: *mut LlamaContextT, batch: LlamaBatch) -> i32;
    fn llama_get_embeddings_ith(ctx: *mut LlamaContextT, i: i32) -> *mut f32;
    fn llama_tokenize(vocab: *const LlamaVocab, text: *const c_char, text_len: i32, tokens: *mut LlamaToken, n_max: i32, add_special: bool, parse_special: bool) -> i32;
    fn llama_token_to_piece(vocab: *const LlamaVocab, token: LlamaToken, buf: *mut c_char, len: i32, lstrip: i32, special: bool) -> i32;
    fn llama_vocab_is_eog(vocab: *const LlamaVocab, token: LlamaToken) -> bool;
    fn llama_chat_apply_template(tmpl: *const c_char, chat: *const LlamaChatMessage, n: usize, add_ass: bool, buf: *mut c_char, len: i32) -> i32;
    fn llama_sampler_chain_default_params() -> LlamaSamplerChainParams;
    fn llama_sampler_chain_init(params: LlamaSamplerChainParams) -> *mut LlamaSampler;
    fn llama_sampler_chain_add(chain: *mut LlamaSampler, smpl: *mut LlamaSampler);
    fn llama_sampler_init_temp(t: f32) -> *mut LlamaSampler;
    fn llama_sampler_init_dist(seed: u32) -> *mut LlamaSampler;
    fn llama_sampler_sample(smpl: *mut LlamaSampler, ctx: *mut LlamaContextT, idx: i32) -> LlamaToken;
    fn llama_sampler_free(smpl: *mut LlamaSampler);

    optional:
    // direct 2-D convolution (no im2col), ggml >= mid-2025
    fn ggml_conv_2d_direct(ctx: *mut GgmlContext, a: Tensor, b: Tensor, s0: c_int, s1: c_int, p0: c_int, p1: c_int, d0: c_int, d1: c_int) -> Tensor;
    fn ggml_backend_supports_op(backend: GgmlBackend, op: Tensor) -> bool;
}

fn lookup<T: Copy>(libs: &[Library], name: &str) -> Result<T> {
    let cname = format!("{name}\0");
    for lib in libs {
        if let Ok(sym) = unsafe { lib.get::<T>(cname.as_bytes()) } {
            return Ok(*sym);
        }
    }
    anyhow::bail!("symbol {name} not found in the ggml/llama libraries")
}

fn lib_name(stem: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else if cfg!(target_os = "windows") {
        format!("{stem}.dll")
    } else {
        format!("lib{stem}.so")
    }
}

/// Opens the library with the symbols exported globally, so the modules
/// dlopened next (`libggml`, `libllama`, the backends) resolve their
/// `ggml-base` imports even when their runpath does not point here.
fn open(path: &Path) -> Result<Library> {
    #[cfg(unix)]
    {
        use libloading::os::unix::Library as UnixLibrary;
        let lib = unsafe { UnixLibrary::open(Some(path), libc::RTLD_NOW | libc::RTLD_GLOBAL) }
            .with_context(|| format!("dlopen {}", path.display()))?;
        Ok(Library::from(lib))
    }
    #[cfg(not(unix))]
    {
        unsafe { Library::new(path) }.with_context(|| format!("load {}", path.display()))
    }
}

/// True if `dir` holds the ggml and llama libraries.
pub fn has_libs(dir: &Path) -> bool {
    ["ggml-base", "ggml", "llama"]
        .iter()
        .all(|stem| dir.join(lib_name(stem)).exists())
}

impl Api {
    /// Loads `libggml-base`, `libggml` and `libllama` from `dir`, then the
    /// backend modules found there (`libggml-cuda`, `libggml-metal`, …).
    pub fn load(dir: &Path) -> Result<Api> {
        let mut libs = Vec::new();
        for stem in ["ggml-base", "ggml", "llama"] {
            libs.push(open(&dir.join(lib_name(stem)))?);
        }
        let api = Api::bind(libs)?;
        let cdir = std::ffi::CString::new(dir.to_string_lossy().as_bytes())
            .context("library directory path")?;
        unsafe {
            (api.ggml_backend_load_all_from_path)(cdir.as_ptr());
            (api.llama_backend_init)();
        }
        Ok(api)
    }

    pub fn set_n_threads(&self, backend: GgmlBackend, n: i32) {
        unsafe {
            let reg = (self.ggml_backend_dev_backend_reg)((self.ggml_backend_get_device)(backend));
            let p = (self.ggml_backend_reg_get_proc_address)(
                reg,
                c"ggml_backend_set_n_threads".as_ptr(),
            );
            if !p.is_null() {
                let f: unsafe extern "C" fn(GgmlBackend, c_int) = std::mem::transmute(p);
                f(backend, n);
            }
        }
    }
}

/// Shape of a tensor in ggml order.
pub fn shape(t: Tensor) -> [i64; 4] {
    unsafe { (*t).ne }
}

pub fn strides(t: Tensor) -> [usize; 4] {
    unsafe { (*t).nb }
}

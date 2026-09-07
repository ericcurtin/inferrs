//! Backend, scheduler and graph building on top of [`ffi::Api`].

use std::ffi::{c_int, CString};
use std::ptr;

use anyhow::{anyhow, Result};

use super::ffi::{self, Api, GgmlBackend, GgmlBackendBuft, GgmlBackendSched, GgmlContext, Tensor};

/// The compute backends: the GPU when there is one, always the CPU.
pub struct Backend {
    pub api: &'static Api,
    pub gpu: GgmlBackend,
    pub cpu: GgmlBackend,
    ptrs: Vec<GgmlBackend>,
    bufts: Vec<GgmlBackendBuft>,
    sched: GgmlBackendSched,
    max_nodes: usize,
}

unsafe impl Send for Backend {}

impl Backend {
    pub fn new(
        api: &'static Api,
        use_gpu: bool,
        n_threads: i32,
        max_nodes: usize,
    ) -> Result<Backend> {
        unsafe {
            let cpu =
                (api.ggml_backend_init_by_type)(ffi::GGML_BACKEND_DEVICE_TYPE_CPU, ptr::null());
            if cpu.is_null() {
                return Err(anyhow!("failed to initialize the CPU backend"));
            }
            let mut gpu = ptr::null_mut();
            if use_gpu {
                gpu =
                    (api.ggml_backend_init_by_type)(ffi::GGML_BACKEND_DEVICE_TYPE_GPU, ptr::null());
                if gpu.is_null() {
                    gpu = (api.ggml_backend_init_by_type)(
                        ffi::GGML_BACKEND_DEVICE_TYPE_IGPU,
                        ptr::null(),
                    );
                }
            }
            let mut ptrs = Vec::new();
            let mut bufts = Vec::new();
            if !gpu.is_null() {
                let name = std::ffi::CStr::from_ptr((api.ggml_backend_name)(gpu)).to_string_lossy();
                eprintln!("[llmman] mediagen: using {name} backend");
                ptrs.push(gpu);
                bufts.push((api.ggml_backend_get_default_buffer_type)(gpu));
            } else {
                eprintln!("[llmman] mediagen: using CPU backend");
            }
            ptrs.push(cpu);
            bufts.push((api.ggml_backend_get_default_buffer_type)(cpu));
            api.set_n_threads(cpu, n_threads);
            let mut be = Backend {
                api,
                gpu,
                cpu,
                ptrs,
                bufts,
                sched: ptr::null_mut(),
                max_nodes,
            };
            be.release_compute()?;
            Ok(be)
        }
    }

    /// Buffer type the weights are allocated on.
    pub fn weight_buft(&self) -> GgmlBackendBuft {
        self.bufts[0]
    }

    /// Recreates the scheduler, freeing its compute buffers.
    pub fn release_compute(&mut self) -> Result<()> {
        unsafe {
            if !self.sched.is_null() {
                (self.api.ggml_backend_sched_free)(self.sched);
            }
            self.sched = (self.api.ggml_backend_sched_new)(
                self.ptrs.as_mut_ptr(),
                self.bufts.as_mut_ptr(),
                self.ptrs.len() as c_int,
                self.max_nodes,
                false,
                true,
            );
        }
        if self.sched.is_null() {
            return Err(anyhow!("ggml_backend_sched_new failed"));
        }
        Ok(())
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe {
            if !self.sched.is_null() {
                (self.api.ggml_backend_sched_free)(self.sched);
            }
            if !self.gpu.is_null() {
                (self.api.ggml_backend_free)(self.gpu);
            }
            (self.api.ggml_backend_free)(self.cpu);
        }
    }
}

/// A ggml context with the op builders; graphs and weight contexts alike.
pub struct Ctx {
    pub api: &'static Api,
    pub raw: *mut GgmlContext,
}

impl Drop for Ctx {
    fn drop(&mut self) {
        unsafe { (self.api.ggml_free)(self.raw) }
    }
}

macro_rules! ops {
    ($($name:ident => $sym:ident($($arg:ident: $ty:ty),*);)*) => {
        impl Ctx {
            $(pub fn $name(&self, $($arg: $ty),*) -> Tensor {
                unsafe { (self.api.$sym)(self.raw, $($arg),*) }
            })*
        }
    };
}

ops! {
    add => ggml_add(a: Tensor, b: Tensor);
    sub => ggml_sub(a: Tensor, b: Tensor);
    mul => ggml_mul(a: Tensor, b: Tensor);
    scale => ggml_scale(a: Tensor, s: f32);
    mul_mat => ggml_mul_mat(a: Tensor, b: Tensor);
    rms_norm => ggml_rms_norm(a: Tensor, eps: f32);
    norm => ggml_norm(a: Tensor, eps: f32);
    silu => ggml_silu(a: Tensor);
    gelu => ggml_gelu(a: Tensor);
    sigmoid => ggml_sigmoid(a: Tensor);
    sin => ggml_sin(a: Tensor);
    sqr => ggml_sqr(a: Tensor);
    sqrt => ggml_sqrt(a: Tensor);
    log => ggml_log(a: Tensor);
    clamp => ggml_clamp(a: Tensor, min: f32, max: f32);
    reshape_2d => ggml_reshape_2d(a: Tensor, ne0: i64, ne1: i64);
    reshape_3d => ggml_reshape_3d(a: Tensor, ne0: i64, ne1: i64, ne2: i64);
    reshape_4d => ggml_reshape_4d(a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64);
    view_2d => ggml_view_2d(a: Tensor, ne0: i64, ne1: i64, nb1: usize, offset: usize);
    view_3d => ggml_view_3d(a: Tensor, ne0: i64, ne1: i64, ne2: i64, nb1: usize, nb2: usize, offset: usize);
    view_4d => ggml_view_4d(a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64, nb1: usize, nb2: usize, nb3: usize, offset: usize);
    permute => ggml_permute(a: Tensor, a0: c_int, a1: c_int, a2: c_int, a3: c_int);
    transpose => ggml_transpose(a: Tensor);
    cont => ggml_cont(a: Tensor);
    cont_2d => ggml_cont_2d(a: Tensor, ne0: i64, ne1: i64);
    concat => ggml_concat(a: Tensor, b: Tensor, dim: c_int);
    cast => ggml_cast(a: Tensor, ty: u32);
    get_rows => ggml_get_rows(a: Tensor, b: Tensor);
    repeat => ggml_repeat(a: Tensor, b: Tensor);
    pad => ggml_pad(a: Tensor, p0: c_int, p1: c_int, p2: c_int, p3: c_int);
    pad_ext => ggml_pad_ext(a: Tensor, lp0: c_int, rp0: c_int, lp1: c_int, rp1: c_int, lp2: c_int, rp2: c_int, lp3: c_int, rp3: c_int);
    conv_1d => ggml_conv_1d(a: Tensor, b: Tensor, s0: c_int, p0: c_int, d0: c_int);
    conv_2d => ggml_conv_2d(a: Tensor, b: Tensor, s0: c_int, s1: c_int, p0: c_int, p1: c_int, d0: c_int, d1: c_int);
    conv_transpose_1d => ggml_conv_transpose_1d(a: Tensor, b: Tensor, s0: c_int, p0: c_int, d0: c_int);
    interpolate => ggml_interpolate(a: Tensor, ne0: i64, ne1: i64, ne2: i64, ne3: i64, mode: u32);
    timestep_embedding => ggml_timestep_embedding(t: Tensor, dim: c_int, max_period: c_int);
    flash_attn_ext => ggml_flash_attn_ext(q: Tensor, k: Tensor, v: Tensor, mask: Tensor, scale: f32, max_bias: f32, logit_softcap: f32);
    soft_max_ext => ggml_soft_max_ext(a: Tensor, mask: Tensor, scale: f32, max_bias: f32);
}

impl Ctx {
    pub fn new(api: &'static Api, mem_size: usize, no_alloc: bool) -> Result<Ctx> {
        let raw = unsafe {
            (api.ggml_init)(ffi::GgmlInitParams {
                mem_size,
                mem_buffer: ptr::null_mut(),
                no_alloc,
            })
        };
        if raw.is_null() {
            return Err(anyhow!("ggml_init failed"));
        }
        Ok(Ctx { api, raw })
    }

    pub fn new_tensor(&self, ty: u32, ne: &[i64]) -> Tensor {
        unsafe { (self.api.ggml_new_tensor)(self.raw, ty, ne.len() as c_int, ne.as_ptr()) }
    }

    pub fn set_name(&self, t: Tensor, name: &str) {
        let name = CString::new(name).unwrap_or_default();
        unsafe { (self.api.ggml_set_name)(t, name.as_ptr()) };
    }

    /// x * (1 + scale) + shift
    pub fn modulate(&self, x: Tensor, scale: Tensor, shift: Option<Tensor>) -> Tensor {
        let y = self.add(self.mul(x, scale), x);
        match shift {
            Some(s) => self.add(y, s),
            None => y,
        }
    }

    pub fn linear(&self, x: Tensor, w: Tensor, b: Option<Tensor>) -> Tensor {
        let y = self.mul_mat(w, x);
        match b {
            Some(b) => self.add(y, b),
            None => y,
        }
    }
}

/// One graph: inputs uploaded after allocation, outputs read after compute.
pub struct Graph {
    pub ctx: Ctx,
    gf: *mut ffi::GgmlCgraph,
    inputs: Vec<(Tensor, Vec<u8>)>,
}

impl Graph {
    pub fn new(api: &'static Api, max_nodes: usize) -> Result<Graph> {
        let mem = unsafe {
            (api.ggml_tensor_overhead)() * max_nodes
                + (api.ggml_graph_overhead_custom)(max_nodes, false)
        };
        let ctx = Ctx::new(api, mem, true)?;
        let gf = unsafe { (api.ggml_new_graph_custom)(ctx.raw, max_nodes, false) };
        if gf.is_null() {
            return Err(anyhow!("ggml_new_graph_custom failed"));
        }
        Ok(Graph {
            ctx,
            gf,
            inputs: Vec::new(),
        })
    }

    pub fn input_f32(&mut self, ne: &[i64], data: &[f32], name: &str) -> Tensor {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.input(ffi::ty::F32, ne, bytes, name)
    }

    pub fn input_i32(&mut self, ne: &[i64], data: &[i32], name: &str) -> Tensor {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.input(ffi::ty::I32, ne, bytes, name)
    }

    fn input(&mut self, ty: u32, ne: &[i64], bytes: Vec<u8>, name: &str) -> Tensor {
        let t = self.ctx.new_tensor(ty, ne);
        self.ctx.set_name(t, name);
        unsafe { (self.ctx.api.ggml_set_input)(t) };
        debug_assert_eq!(unsafe { (self.ctx.api.ggml_nbytes)(t) }, bytes.len());
        self.inputs.push((t, bytes));
        t
    }

    pub fn mark_output(&self, t: Tensor) {
        unsafe {
            (self.ctx.api.ggml_set_output)(t);
            (self.ctx.api.ggml_build_forward_expand)(self.gf, t);
        }
    }

    pub fn compute(&self, be: &Backend) -> Result<()> {
        let api = self.ctx.api;
        unsafe {
            (api.ggml_backend_sched_reset)(be.sched);
            if !(api.ggml_backend_sched_alloc_graph)(be.sched, self.gf) {
                return Err(anyhow!(
                    "failed to allocate a graph of {} nodes",
                    (api.ggml_graph_n_nodes)(self.gf)
                ));
            }
            for (t, bytes) in &self.inputs {
                (api.ggml_backend_tensor_set)(*t, bytes.as_ptr() as *const _, 0, bytes.len());
            }
            let status = (api.ggml_backend_sched_graph_compute)(be.sched, self.gf);
            if status != ffi::GGML_STATUS_SUCCESS {
                return Err(anyhow!("graph compute failed with status {status}"));
            }
        }
        Ok(())
    }

    pub fn output_f32(&self, t: Tensor) -> Vec<f32> {
        let api = self.ctx.api;
        let n = unsafe { (api.ggml_nelements)(t) } as usize;
        let mut out = vec![0f32; n];
        unsafe { (api.ggml_backend_tensor_get)(t, out.as_mut_ptr() as *mut _, 0, n * 4) };
        out
    }
}

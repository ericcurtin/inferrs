//! Model weights: safetensors and GGUF tensor indices, loaded into one
//! backend buffer. bf16 becomes f16 (f32 for 1-D tensors), 5-D conv
//! kernels are split into per-tap 2-D kernels (see the VAE).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

use super::backend::Ctx;
use super::ffi::{self, Api, GgmlBackendBuffer, GgmlBackendBuft, Tensor};

/// A tensor as stored in a model file.
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub name: String,
    pub ty: u32,
    /// ggml order: `ne[0]` is the fastest dimension.
    pub ne: Vec<i64>,
    pub offset: u64,
    pub nbytes: u64,
}

/// Header of a safetensors or GGUF file.
#[derive(Debug, Default)]
pub struct Index {
    pub tensors: Vec<TensorDesc>,
    /// String metadata: GGUF string KVs, or the safetensors `__metadata__`.
    pub metadata: HashMap<String, String>,
}

pub fn read_index(path: &Path) -> Result<Index> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    f.seek(SeekFrom::Start(0))?;
    if &magic == b"GGUF" {
        read_gguf(BufReader::new(f), path)
    } else {
        read_safetensors(f, path)
    }
}

fn read_safetensors(mut f: File, path: &Path) -> Result<Index> {
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let n = u64::from_le_bytes(len);
    if n == 0 || n > 1 << 30 {
        bail!("invalid safetensors header in {}", path.display());
    }
    let mut header = vec![0u8; n as usize];
    f.read_exact(&mut header)?;
    let json: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
        .with_context(|| format!("safetensors header of {}", path.display()))?;
    let base = 8 + n;
    let mut idx = Index::default();
    for (name, v) in json {
        if name == "__metadata__" {
            if let Some(m) = v.as_object() {
                for (k, v) in m {
                    idx.metadata.insert(
                        k.clone(),
                        v.as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| v.to_string()),
                    );
                }
            }
            continue;
        }
        let dtype = v["dtype"].as_str().unwrap_or("");
        let ty = match dtype {
            "BF16" => ffi::ty::BF16,
            "F16" => ffi::ty::F16,
            "F32" => ffi::ty::F32,
            other => bail!("unsupported dtype {other:?} for tensor {name}"),
        };
        let shape: Vec<i64> = v["shape"]
            .as_array()
            .ok_or_else(|| anyhow!("tensor {name}: missing shape"))?
            .iter()
            .map(|d| {
                d.as_i64()
                    .filter(|&d| d > 0)
                    .ok_or_else(|| anyhow!("tensor {name}: invalid shape"))
            })
            .collect::<Result<_>>()?;
        let offs = v["data_offsets"]
            .as_array()
            .filter(|a| a.len() == 2)
            .ok_or_else(|| anyhow!("tensor {name}: missing data_offsets"))?;
        let (start, end) = (offs[0].as_u64().unwrap_or(0), offs[1].as_u64().unwrap_or(0));
        if end < start {
            bail!("tensor {name}: invalid data_offsets");
        }
        let mut ne: Vec<i64> = shape.into_iter().rev().collect();
        if ne.is_empty() {
            ne.push(1);
        }
        idx.tensors.push(TensorDesc {
            name,
            ty,
            ne,
            offset: base + start,
            nbytes: end - start,
        });
    }
    Ok(idx)
}

struct Counting<R> {
    inner: R,
    pos: u64,
}

impl<R: Read> Counting<R> {
    fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.inner.read_exact(&mut b)?;
        self.pos += 4;
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.inner.read_exact(&mut b)?;
        self.pos += 8;
        Ok(u64::from_le_bytes(b))
    }
    fn skip(&mut self, n: u64) -> Result<()> {
        if std::io::copy(&mut (&mut self.inner).take(n), &mut std::io::sink())? != n {
            bail!("unexpected end of file");
        }
        self.pos += n;
        Ok(())
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        if n > 1 << 24 {
            bail!("implausible string length {n}");
        }
        let mut b = vec![0u8; n as usize];
        self.inner.read_exact(&mut b)?;
        self.pos += n;
        Ok(String::from_utf8_lossy(&b).into_owned())
    }
    /// Reads a metadata value, returning it when it is a string.
    fn value(&mut self, ty: u32) -> Result<Option<String>> {
        match ty {
            0 | 1 | 7 => self.skip(1)?,
            2 | 3 => self.skip(2)?,
            4..=6 => self.skip(4)?,
            10..=12 => self.skip(8)?,
            8 => return Ok(Some(self.string()?)),
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                if n > 1 << 24 {
                    bail!("implausible array length {n}");
                }
                for _ in 0..n {
                    self.value(et)?;
                }
            }
            other => bail!("unknown GGUF value type {other}"),
        }
        Ok(None)
    }
}

/// GGUF tensor index with absolute data offsets. A dedicated reader: these
/// files have tensor names longer than ggml's own limit.
fn read_gguf<R: Read>(inner: R, path: &Path) -> Result<Index> {
    let mut r = Counting { inner, pos: 0 };
    let mut magic = [0u8; 4];
    r.inner.read_exact(&mut magic)?;
    r.pos += 4;
    let version = r.u32()?;
    if version < 2 {
        bail!("unsupported GGUF version {version}");
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    if n_tensors > 1 << 24 || n_kv > 1 << 24 {
        bail!("invalid GGUF header in {}", path.display());
    }
    let mut idx = Index::default();
    let mut alignment = 32u64;
    for _ in 0..n_kv {
        let key = r.string()?;
        let ty = r.u32()?;
        if key == "general.alignment" && ty == 4 {
            alignment = r.u32()? as u64;
            if alignment == 0 || !alignment.is_power_of_two() {
                bail!("invalid general.alignment in {}", path.display());
            }
            continue;
        }
        if let Some(s) = r.value(ty)? {
            idx.metadata.insert(key, s);
        }
    }
    for _ in 0..n_tensors {
        let name = r.string()?;
        let n_dims = r.u32()?;
        if n_dims == 0 || n_dims > 8 {
            bail!("tensor {name}: invalid rank {n_dims}");
        }
        let mut ne = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let d = r.u64()?;
            if d == 0 || d > i32::MAX as u64 {
                bail!("tensor {name}: invalid dimension {d}");
            }
            ne.push(d as i64);
        }
        let ty = r.u32()?;
        let offset = r.u64()?;
        idx.tensors.push(TensorDesc {
            name,
            ty,
            ne,
            offset,
            nbytes: 0,
        });
    }
    let data_start = r.pos.div_ceil(alignment) * alignment;
    for t in &mut idx.tensors {
        t.offset += data_start;
    }
    Ok(idx)
}

/// Renames a tensor; `None` drops it.
pub type Rename = Box<dyn Fn(&str) -> Option<String>>;
/// Applied to f32 tensors while loading.
pub type Transform = Box<dyn Fn(&str, &mut [f32])>;

/// Loading options.
#[derive(Default)]
pub struct LoadOpts {
    pub rename: Option<Rename>,
    pub transform: Option<Transform>,
}

/// Tensors of one file living on one backend buffer.
pub struct Weights {
    api: &'static Api,
    _ctx: Ctx,
    buf: GgmlBackendBuffer,
    tensors: HashMap<String, Tensor>,
}

unsafe impl Send for Weights {}

impl Drop for Weights {
    fn drop(&mut self) {
        if !self.buf.is_null() {
            unsafe { (self.api.ggml_backend_buffer_free)(self.buf) };
        }
    }
}

impl Weights {
    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.tensors
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("missing tensor {name:?}"))
    }

    pub fn opt(&self, name: &str) -> Option<Tensor> {
        self.tensors.get(name).copied()
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn size(&self) -> usize {
        unsafe { (self.api.ggml_backend_buffer_get_size)(self.buf) }
    }

    /// Reads a small f32 / f16 tensor back to the host.
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.get(name)?;
        let n = unsafe { (self.api.ggml_nelements)(t) } as usize;
        let ty = unsafe { (*t).type_ };
        match ty {
            ffi::ty::F32 => {
                let mut out = vec![0f32; n];
                unsafe {
                    (self.api.ggml_backend_tensor_get)(t, out.as_mut_ptr() as *mut _, 0, n * 4)
                };
                Ok(out)
            }
            ffi::ty::F16 => {
                let mut h = vec![0u16; n];
                unsafe {
                    (self.api.ggml_backend_tensor_get)(t, h.as_mut_ptr() as *mut _, 0, n * 2)
                };
                let mut out = vec![0f32; n];
                unsafe { (self.api.ggml_fp16_to_fp32_row)(h.as_ptr(), out.as_mut_ptr(), n as i64) };
                Ok(out)
            }
            _ => bail!("tensor {name} is not f32/f16"),
        }
    }

    pub fn load(
        api: &'static Api,
        path: &Path,
        buft: GgmlBackendBuft,
        opts: &LoadOpts,
    ) -> Result<Weights> {
        let idx = read_index(path)?;
        struct Item {
            dst: Tensor,
            src: TensorDesc,
            tap: i32,
            name: String,
        }
        let mut plan: Vec<Item> = Vec::new();
        let n_planned: usize = idx
            .tensors
            .iter()
            .map(|d| {
                if d.ne.len() == 5 {
                    d.ne[2].max(1) as usize
                } else {
                    1
                }
            })
            .sum();
        let ctx = Ctx::new(
            api,
            unsafe { (api.ggml_tensor_overhead)() } * (n_planned + 16),
            true,
        )?;
        let mut tensors = HashMap::new();
        for d in &idx.tensors {
            let name = match &opts.rename {
                Some(f) => match f(&d.name) {
                    Some(n) => n,
                    None => continue,
                },
                None => d.name.clone(),
            };
            let mut ty = d.ty;
            if ty == ffi::ty::BF16 {
                ty = if d.ne.len() == 1 {
                    ffi::ty::F32
                } else {
                    ffi::ty::F16
                };
            }
            if d.ne.len() == 5 {
                // [KW, KH, KT, IC, OC] -> KT x [KW, KH, IC, OC]
                for k in 0..d.ne[2] {
                    let t = ctx.new_tensor(ty, &[d.ne[0], d.ne[1], d.ne[3], d.ne[4]]);
                    let tname = format!("{name}.t{k}");
                    ctx.set_name(t, &tname);
                    tensors.insert(tname.clone(), t);
                    plan.push(Item {
                        dst: t,
                        src: d.clone(),
                        tap: k as i32,
                        name: tname,
                    });
                }
                continue;
            }
            if d.ne.len() > 4 {
                bail!("tensor {} has {} dims", d.name, d.ne.len());
            }
            let t = ctx.new_tensor(ty, &d.ne);
            ctx.set_name(t, &name);
            tensors.insert(name.clone(), t);
            plan.push(Item {
                dst: t,
                src: d.clone(),
                tap: -1,
                name,
            });
        }
        let buf = unsafe { (api.ggml_backend_alloc_ctx_tensors_from_buft)(ctx.raw, buft) };
        if buf.is_null() {
            bail!("failed to allocate a buffer for {}", path.display());
        }
        unsafe { (api.ggml_backend_buffer_set_usage)(buf, ffi::GGML_BACKEND_BUFFER_USAGE_WEIGHTS) };
        let w = Weights {
            api,
            _ctx: ctx,
            buf,
            tensors,
        };

        let mut f = File::open(path)?;
        let file_len = f.metadata()?.len();
        let mut read_buf = Vec::new();
        let mut conv_buf = Vec::new();
        for item in &plan {
            let d = &item.src;
            let elems =
                d.ne.iter()
                    .try_fold(1i64, |a, &b| a.checked_mul(b))
                    .ok_or_else(|| anyhow!("tensor {}: shape overflows", d.name))?;
            let src_bytes = unsafe { (api.ggml_row_size)(d.ty, elems) };
            if d.nbytes != 0 && d.nbytes != src_bytes as u64 {
                bail!(
                    "tensor {}: {} bytes in file, expected {}",
                    d.name,
                    d.nbytes,
                    src_bytes
                );
            }
            if d.offset
                .checked_add(src_bytes as u64)
                .is_none_or(|end| end > file_len)
            {
                bail!("tensor {}: data past the end of the file", d.name);
            }
            let dst_bytes = unsafe { (api.ggml_nbytes)(item.dst) };
            let dst_ty = unsafe { (*item.dst).type_ };
            read_buf.resize(src_bytes, 0);
            f.seek(SeekFrom::Start(d.offset))?;
            f.read_exact(&mut read_buf)
                .with_context(|| format!("read tensor {}", d.name))?;
            let src: &[u8] = if item.tap < 0 {
                &read_buf
            } else {
                // one temporal tap of [KW, KH, KT, IC, OC]
                let (kw, kh, kt, ic, oc) = (d.ne[0], d.ne[1], d.ne[2], d.ne[3], d.ne[4]);
                let ts = unsafe { (api.ggml_type_size)(d.ty) };
                let blk = (kw * kh) as usize * ts;
                conv_buf.clear();
                conv_buf.reserve(blk * (ic * oc) as usize);
                for o in 0..oc {
                    for i in 0..ic {
                        let off = (((o * ic + i) * kt + item.tap as i64) as usize) * blk;
                        conv_buf.extend_from_slice(&read_buf[off..off + blk]);
                    }
                }
                &conv_buf
            };
            let n = unsafe { (api.ggml_nelements)(item.dst) };
            let converted = convert(api, d.ty, dst_ty, src, n)?;
            let mut data = converted.unwrap_or_else(|| src.to_vec());
            if dst_ty == ffi::ty::F32 {
                if let Some(tf) = &opts.transform {
                    let mut floats: Vec<f32> = data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect();
                    tf(&item.name, &mut floats);
                    data = floats.iter().flat_map(|v| v.to_le_bytes()).collect();
                }
            }
            if data.len() != dst_bytes {
                bail!(
                    "tensor {}: {} bytes, expected {}",
                    item.name,
                    data.len(),
                    dst_bytes
                );
            }
            unsafe {
                (api.ggml_backend_tensor_set)(item.dst, data.as_ptr() as *const _, 0, dst_bytes)
            };
        }
        eprintln!(
            "[llmman] mediagen: loaded {} tensors ({:.1} MiB) from {}",
            plan.len(),
            w.size() as f64 / 1024.0 / 1024.0,
            path.display()
        );
        Ok(w)
    }
}

/// Converts `src` of type `from` to `to`; `None` when no conversion is needed.
fn convert(api: &Api, from: u32, to: u32, src: &[u8], n: i64) -> Result<Option<Vec<u8>>> {
    if from == to {
        return Ok(None);
    }
    if src.len() != unsafe { (api.ggml_row_size)(from, n) } {
        bail!("source size does not match the tensor");
    }
    let mut f32s = vec![0f32; n as usize];
    // aligned copies: the byte buffer has no u16/f32 alignment
    let halves = || -> Vec<u16> {
        src.as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect()
    };
    unsafe {
        match from {
            ffi::ty::BF16 => (api.ggml_bf16_to_fp32_row)(halves().as_ptr(), f32s.as_mut_ptr(), n),
            ffi::ty::F16 => (api.ggml_fp16_to_fp32_row)(halves().as_ptr(), f32s.as_mut_ptr(), n),
            ffi::ty::F32 => {
                for (v, b) in f32s.iter_mut().zip(src.as_chunks::<4>().0) {
                    *v = f32::from_le_bytes(*b);
                }
            }
            other => bail!("cannot convert from ggml type {other}"),
        }
    }
    match to {
        ffi::ty::F32 => Ok(Some(f32s.iter().flat_map(|v| v.to_le_bytes()).collect())),
        ffi::ty::F16 => {
            let mut h = vec![0u16; n as usize];
            unsafe { (api.ggml_fp32_to_fp16_row)(f32s.as_ptr(), h.as_mut_ptr(), n) };
            Ok(Some(h.iter().flat_map(|v| v.to_le_bytes()).collect()))
        }
        other => bail!("cannot convert to ggml type {other}"),
    }
}

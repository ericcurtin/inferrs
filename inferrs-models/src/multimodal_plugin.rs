//! Runtime loader for the `inferrs-multimodal` cdylib plugin.
//!
//! Dlopens `libinferrs_multimodal.{dylib,so,dll}` from the same directory as
//! the running binary and wraps every `inferrs_mm_*` C ABI symbol behind safe
//! Rust types that look identical to the old `AudioEncoder` / `VisionEncoder`
//! structs that used to be statically linked.
//!
//! If the plugin is not found the server still starts; it just won't be able
//! to serve audio/vision requests.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

// ---------------------------------------------------------------------------
// Raw function-pointer types matching the C ABI in inferrs-multimodal/src/ffi.rs
// ---------------------------------------------------------------------------

type FnLastError = unsafe extern "C" fn() -> *const std::os::raw::c_char;
type FnNMel = extern "C" fn() -> usize;
type FnMaxMelFrames = extern "C" fn() -> usize;
type FnDecodeAudio = unsafe extern "C" fn(
    data: *const u8,
    data_len: usize,
    format: *const std::os::raw::c_char,
    out_samples: *mut *mut f32,
    out_len: *mut usize,
) -> i32;
type FnComputeLogMel = unsafe extern "C" fn(
    samples: *const f32,
    n_samples: usize,
    out_mel: *mut *mut f32,
    out_n_frames: *mut usize,
) -> i32;
type FnFreeF32 = unsafe extern "C" fn(ptr: *mut f32, len: usize);
type FnAudioEncoderLoad = unsafe extern "C" fn(
    paths: *const *const std::os::raw::c_char,
    n_paths: usize,
    audio_cfg_json: *const std::os::raw::c_char,
    lm_hidden_size: usize,
    dtype_tag: u8,
    device_tag: u8,
) -> *mut std::os::raw::c_void;
type FnAudioEncoderEncode = unsafe extern "C" fn(
    handle: *mut std::os::raw::c_void,
    mel: *const f32,
    n_frames: usize,
    device_tag: u8,
    out_data: *mut *mut f32,
    out_rows: *mut usize,
    out_cols: *mut usize,
) -> i32;
type FnAudioEncoderFree = unsafe extern "C" fn(handle: *mut std::os::raw::c_void);
type FnVisionEncoderLoad = unsafe extern "C" fn(
    paths: *const *const std::os::raw::c_char,
    n_paths: usize,
    vision_cfg_json: *const std::os::raw::c_char,
    lm_hidden_size: usize,
    dtype_tag: u8,
    device_tag: u8,
) -> *mut std::os::raw::c_void;
type FnVisionEncoderEncode = unsafe extern "C" fn(
    handle: *mut std::os::raw::c_void,
    pixel_values: *const f32,
    pv_rows: usize,
    pv_cols: usize,
    position_ids: *const i64,
    n_soft_tokens: usize,
    device_tag: u8,
    out_data: *mut *mut f32,
    out_rows: *mut usize,
    out_cols: *mut usize,
) -> i32;
type FnVisionEncoderFree = unsafe extern "C" fn(handle: *mut std::os::raw::c_void);

// ---------------------------------------------------------------------------
// Plugin vtable — owns the libloading::Library so symbols remain valid
// ---------------------------------------------------------------------------

struct PluginVtable {
    // Keep the library loaded as long as any vtable reference exists.
    _lib: libloading::Library,

    last_error: FnLastError,
    n_mel: FnNMel,
    max_mel_frames: FnMaxMelFrames,
    decode_audio: FnDecodeAudio,
    compute_log_mel: FnComputeLogMel,
    free_f32: FnFreeF32,
    audio_encoder_load: FnAudioEncoderLoad,
    audio_encoder_encode: FnAudioEncoderEncode,
    audio_encoder_free: FnAudioEncoderFree,
    vision_encoder_load: FnVisionEncoderLoad,
    vision_encoder_encode: FnVisionEncoderEncode,
    vision_encoder_free: FnVisionEncoderFree,
}

// SAFETY: the function pointers loaded from the dylib are statically compiled
// and do not capture any non-Send state; the Library is protected by Arc.
unsafe impl Send for PluginVtable {}
unsafe impl Sync for PluginVtable {}

macro_rules! load_sym {
    ($lib:expr, $name:expr, $ty:ty) => {{
        let sym: libloading::Symbol<$ty> = unsafe { $lib.get($name) }.with_context(|| {
            format!(
                "symbol not found in plugin: {}",
                String::from_utf8_lossy($name)
            )
        })?;
        *sym
    }};
}

impl PluginVtable {
    fn load(lib_path: &Path) -> Result<Self> {
        let lib = unsafe { libloading::Library::new(lib_path) }
            .with_context(|| format!("failed to dlopen {}", lib_path.display()))?;

        let vtable = Self {
            last_error: load_sym!(lib, b"inferrs_mm_last_error\0", FnLastError),
            n_mel: load_sym!(lib, b"inferrs_mm_n_mel\0", FnNMel),
            max_mel_frames: load_sym!(lib, b"inferrs_mm_max_mel_frames\0", FnMaxMelFrames),
            decode_audio: load_sym!(lib, b"inferrs_mm_decode_audio\0", FnDecodeAudio),
            compute_log_mel: load_sym!(lib, b"inferrs_mm_compute_log_mel\0", FnComputeLogMel),
            free_f32: load_sym!(lib, b"inferrs_mm_free_f32\0", FnFreeF32),
            audio_encoder_load: load_sym!(
                lib,
                b"inferrs_mm_audio_encoder_load\0",
                FnAudioEncoderLoad
            ),
            audio_encoder_encode: load_sym!(
                lib,
                b"inferrs_mm_audio_encoder_encode\0",
                FnAudioEncoderEncode
            ),
            audio_encoder_free: load_sym!(
                lib,
                b"inferrs_mm_audio_encoder_free\0",
                FnAudioEncoderFree
            ),
            vision_encoder_load: load_sym!(
                lib,
                b"inferrs_mm_vision_encoder_load\0",
                FnVisionEncoderLoad
            ),
            vision_encoder_encode: load_sym!(
                lib,
                b"inferrs_mm_vision_encoder_encode\0",
                FnVisionEncoderEncode
            ),
            vision_encoder_free: load_sym!(
                lib,
                b"inferrs_mm_vision_encoder_free\0",
                FnVisionEncoderFree
            ),
            _lib: lib,
        };
        Ok(vtable)
    }

    fn last_error_string(&self) -> String {
        let ptr = unsafe { (self.last_error)() };
        if ptr.is_null() {
            "(no error)".to_string()
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        }
    }
}

// ---------------------------------------------------------------------------
// Public handle: MultimodalPlugin
// ---------------------------------------------------------------------------

/// A loaded instance of the `inferrs-multimodal` plugin.
///
/// Clone-cheap via `Arc`; the underlying library is unloaded when the last
/// clone is dropped.
#[derive(Clone)]
#[allow(dead_code)]
pub struct MultimodalPlugin {
    vt: Arc<PluginVtable>,
}

#[allow(dead_code)]
impl MultimodalPlugin {
    /// Attempt to load the plugin from the same directory as the running binary.
    pub fn load() -> Result<Self> {
        let exe = std::env::current_exe().context("cannot determine executable path")?;
        let dir = exe.parent().unwrap_or(Path::new("."));

        #[cfg(target_os = "macos")]
        let name = "libinferrs_multimodal.dylib";
        #[cfg(target_os = "linux")]
        let name = "libinferrs_multimodal.so";
        #[cfg(target_os = "windows")]
        let name = "inferrs_multimodal.dll";
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        let name = "libinferrs_multimodal.so";

        let path = dir.join(name);
        let vt = PluginVtable::load(&path)?;
        Ok(Self { vt: Arc::new(vt) })
    }

    // ── Constant accessors ───────────────────────────────────────────────────

    pub fn n_mel(&self) -> usize {
        (self.vt.n_mel)()
    }

    pub fn max_mel_frames(&self) -> usize {
        (self.vt.max_mel_frames)()
    }

    // ── Audio preprocessing ──────────────────────────────────────────────────

    /// Decode audio bytes (WAV or raw f32 PCM) to mono 16 kHz f32 samples.
    pub fn decode_audio(&self, data: &[u8], format: &str) -> Result<Vec<f32>> {
        let fmt = CString::new(format).context("format contains null byte")?;
        let mut out_ptr: *mut f32 = std::ptr::null_mut();
        let mut out_len: usize = 0;
        let rc = unsafe {
            (self.vt.decode_audio)(
                data.as_ptr(),
                data.len(),
                fmt.as_ptr(),
                &mut out_ptr,
                &mut out_len,
            )
        };
        if rc != 0 {
            anyhow::bail!("decode_audio failed: {}", self.vt.last_error_string());
        }
        let samples = unsafe { std::slice::from_raw_parts(out_ptr, out_len) }.to_vec();
        unsafe { (self.vt.free_f32)(out_ptr, out_len) };
        Ok(samples)
    }

    /// Compute log-mel spectrogram. Returns `(data, n_frames)` where data is
    /// row-major `[n_frames, N_MEL]`.
    pub fn compute_log_mel(&self, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        let mut out_ptr: *mut f32 = std::ptr::null_mut();
        let mut out_frames: usize = 0;
        let rc = unsafe {
            (self.vt.compute_log_mel)(
                samples.as_ptr(),
                samples.len(),
                &mut out_ptr,
                &mut out_frames,
            )
        };
        if rc != 0 {
            anyhow::bail!("compute_log_mel failed: {}", self.vt.last_error_string());
        }
        let len = out_frames * self.n_mel();
        let mel = unsafe { std::slice::from_raw_parts(out_ptr, len) }.to_vec();
        unsafe { (self.vt.free_f32)(out_ptr, len) };
        Ok((mel, out_frames))
    }

    // ── Encoder construction ─────────────────────────────────────────────────

    /// Load a Gemma4 audio encoder from safetensors weight files.
    pub fn load_audio_encoder(
        &self,
        weight_paths: &[&Path],
        audio_cfg_json: &str,
        lm_hidden_size: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<AudioEncoderHandle> {
        let c_paths: Vec<CString> = weight_paths
            .iter()
            .map(|p| CString::new(p.to_str().context("path not UTF-8")?).context("null in path"))
            .collect::<Result<_>>()?;
        let ptr_array: Vec<*const std::os::raw::c_char> =
            c_paths.iter().map(|s| s.as_ptr()).collect();
        let cfg_c = CString::new(audio_cfg_json).context("null in cfg json")?;

        let handle = unsafe {
            (self.vt.audio_encoder_load)(
                ptr_array.as_ptr(),
                ptr_array.len(),
                cfg_c.as_ptr(),
                lm_hidden_size,
                dtype_tag(dtype),
                device_tag(device)?,
            )
        };
        if handle.is_null() {
            anyhow::bail!("audio encoder load failed: {}", self.vt.last_error_string());
        }
        Ok(AudioEncoderHandle {
            handle,
            vt: self.vt.clone(),
            device_tag: device_tag(device)?,
        })
    }

    /// Load a Gemma4 vision encoder from safetensors weight files.
    pub fn load_vision_encoder(
        &self,
        weight_paths: &[&Path],
        vision_cfg_json: &str,
        lm_hidden_size: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<VisionEncoderHandle> {
        let c_paths: Vec<CString> = weight_paths
            .iter()
            .map(|p| CString::new(p.to_str().context("path not UTF-8")?).context("null in path"))
            .collect::<Result<_>>()?;
        let ptr_array: Vec<*const std::os::raw::c_char> =
            c_paths.iter().map(|s| s.as_ptr()).collect();
        let cfg_c = CString::new(vision_cfg_json).context("null in cfg json")?;

        let handle = unsafe {
            (self.vt.vision_encoder_load)(
                ptr_array.as_ptr(),
                ptr_array.len(),
                cfg_c.as_ptr(),
                lm_hidden_size,
                dtype_tag(dtype),
                device_tag(device)?,
            )
        };
        if handle.is_null() {
            anyhow::bail!(
                "vision encoder load failed: {}",
                self.vt.last_error_string()
            );
        }
        Ok(VisionEncoderHandle {
            handle,
            vt: self.vt.clone(),
            device_tag: device_tag(device)?,
        })
    }
}

// ---------------------------------------------------------------------------
// AudioEncoderHandle — RAII wrapper around the opaque C handle
// ---------------------------------------------------------------------------

/// A loaded audio encoder living inside the plugin.
pub struct AudioEncoderHandle {
    handle: *mut std::os::raw::c_void,
    vt: Arc<PluginVtable>,
    device_tag: u8,
}

// SAFETY: the handle is an owned Box inside the plugin; it is not shared.
unsafe impl Send for AudioEncoderHandle {}
unsafe impl Sync for AudioEncoderHandle {}

impl AudioEncoderHandle {
    /// Maximum mel frames the encoder will process before truncating.
    pub const MAX_MEL_FRAMES: usize = 1500;

    /// Encode a log-mel spectrogram tensor `[1, T, N_MEL]` f32 → `[T/4, lm_hidden]`.
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        let (_, n_frames, _n_mel) = mel.dims3()?;
        let mel_f32 = mel.to_dtype(DType::F32)?.flatten_all()?;
        let mel_data: Vec<f32> = mel_f32.to_vec1()?;

        let mut out_ptr: *mut f32 = std::ptr::null_mut();
        let mut out_rows: usize = 0;
        let mut out_cols: usize = 0;

        let rc = unsafe {
            (self.vt.audio_encoder_encode)(
                self.handle,
                mel_data.as_ptr(),
                n_frames,
                self.device_tag,
                &mut out_ptr,
                &mut out_rows,
                &mut out_cols,
            )
        };
        if rc != 0 {
            anyhow::bail!("audio encode failed: {}", self.vt.last_error_string());
        }
        let len = out_rows * out_cols;
        let data = unsafe { std::slice::from_raw_parts(out_ptr, len) }.to_vec();
        unsafe { (self.vt.free_f32)(out_ptr, len) };

        // Reconstruct on the same device the mel came from.
        let out = Tensor::from_vec(data, (out_rows, out_cols), mel.device())?;
        Ok(out)
    }
}

impl Drop for AudioEncoderHandle {
    fn drop(&mut self) {
        unsafe { (self.vt.audio_encoder_free)(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// VisionEncoderHandle
// ---------------------------------------------------------------------------

/// A loaded vision encoder living inside the plugin.
pub struct VisionEncoderHandle {
    handle: *mut std::os::raw::c_void,
    vt: Arc<PluginVtable>,
    device_tag: u8,
}

unsafe impl Send for VisionEncoderHandle {}
unsafe impl Sync for VisionEncoderHandle {}

impl VisionEncoderHandle {
    /// Encode pixel values `[N_patches, patch_pixels]` + position ids `[N_patches, 2]`
    /// → `[n_soft_tokens, lm_hidden]`.
    pub fn encode(
        &self,
        pixel_values: &Tensor,
        position_ids: &Tensor,
        n_soft_tokens: usize,
    ) -> Result<Tensor> {
        let (pv_rows, pv_cols) = pixel_values.dims2()?;
        let pv_data: Vec<f32> = pixel_values
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let pos_data: Vec<i64> = position_ids.flatten_all()?.to_vec1()?;

        let mut out_ptr: *mut f32 = std::ptr::null_mut();
        let mut out_rows: usize = 0;
        let mut out_cols: usize = 0;

        let rc = unsafe {
            (self.vt.vision_encoder_encode)(
                self.handle,
                pv_data.as_ptr(),
                pv_rows,
                pv_cols,
                pos_data.as_ptr(),
                n_soft_tokens,
                self.device_tag,
                &mut out_ptr,
                &mut out_rows,
                &mut out_cols,
            )
        };
        if rc != 0 {
            anyhow::bail!("vision encode failed: {}", self.vt.last_error_string());
        }
        let len = out_rows * out_cols;
        let data = unsafe { std::slice::from_raw_parts(out_ptr, len) }.to_vec();
        unsafe { (self.vt.free_f32)(out_ptr, len) };

        let out = Tensor::from_vec(data, (out_rows, out_cols), pixel_values.device())?;
        Ok(out)
    }
}

impl Drop for VisionEncoderHandle {
    fn drop(&mut self) {
        unsafe { (self.vt.vision_encoder_free)(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// Helpers: candle DType / Device → u8 tags
// ---------------------------------------------------------------------------

fn dtype_tag(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        _ => 0, // fall back to F32
    }
}

fn device_tag(device: &Device) -> Result<u8> {
    match device {
        Device::Cpu => Ok(0),
        Device::Metal(_) => Ok(1),
        Device::Cuda(_) => Ok(2),
    }
}

//! C ABI exports for the `inferrs-multimodal` cdylib.
//!
//! # Design
//!
//! `candle_core::Tensor` and `VarBuilder` are not FFI-safe, so the boundary is:
//!
//! - Encoder objects are heap-boxed and returned as opaque `*mut c_void` handles.
//! - Tensor data crosses as raw `f32`/`i64` pointers + shape slices.
//! - Weight sources are passed as null-terminated C-string paths (safetensors).
//! - Device is encoded as a `u8`: 0 = CPU, 1 = Metal, 2 = CUDA.
//! - DType is encoded as a `u8`: 0 = F32, 1 = F16, 2 = BF16.
//! - Errors are returned as a heap-allocated C string via `inferrs_mm_last_error()`
//!   and a non-zero return code.
//!
//! All exported symbols are prefixed `inferrs_mm_` (inferrs multimodal).

use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::Mutex;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::audio_encoder::AudioEncoder;
use crate::config::{AudioConfig, Gemma4VisionConfig};
use crate::vision_encoder::VisionEncoder;

// ---------------------------------------------------------------------------
// Thread-local last-error storage
// ---------------------------------------------------------------------------

std::thread_local! {
    static LAST_ERROR: Mutex<Option<CString>> = const { Mutex::new(None) };
}

fn set_last_error(e: impl std::fmt::Display) {
    let msg = CString::new(format!("{e:#}"))
        .unwrap_or_else(|_| CString::new("error message contained a null byte").unwrap());
    LAST_ERROR.with(|cell| {
        *cell.lock().unwrap() = Some(msg);
    });
}

/// Return the last error as a null-terminated C string, or null if none.
/// The returned pointer is valid until the next plugin call on this thread.
///
/// # Safety
/// The returned pointer must not be used after any subsequent call to another
/// `inferrs_mm_*` function on the same thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferrs_mm_last_error() -> *const std::os::raw::c_char {
    LAST_ERROR.with(|cell| {
        cell.lock()
            .unwrap()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null())
    })
}

// ---------------------------------------------------------------------------
// Helpers: decode device / dtype from u8 tags
// ---------------------------------------------------------------------------

fn decode_device(tag: u8) -> Result<Device> {
    match tag {
        0 => Ok(Device::Cpu),
        #[cfg(any(feature = "metal", target_os = "macos"))]
        1 => Ok(Device::new_metal(0)?),
        #[cfg(any(feature = "cuda", target_os = "linux", target_os = "windows"))]
        2 => Ok(Device::new_cuda(0)?),
        other => anyhow::bail!("unknown device tag {other}"),
    }
}

fn decode_dtype(tag: u8) -> Result<DType> {
    match tag {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        2 => Ok(DType::BF16),
        other => anyhow::bail!("unknown dtype tag {other}"),
    }
}

/// Build a `VarBuilder` from a list of null-terminated safetensors paths.
///
/// Returns `(vb, device)` so the caller can reuse the *exact same* `Device`
/// instance that was used to load the weights.  On Metal, `same_device` checks
/// pointer identity (via `DeviceId`), so a second call to `decode_device` would
/// produce a different ID and trigger spurious "device mismatch" errors when
/// input tensors are created later.
///
/// # Safety
/// `paths` must point to a valid array of `n_paths` non-null, null-terminated
/// UTF-8 strings, all of which remain valid for the duration of the call.
unsafe fn build_var_builder(
    paths: *const *const std::os::raw::c_char,
    n_paths: usize,
    dtype_tag: u8,
    device_tag: u8,
) -> Result<(VarBuilder<'static>, Device)> {
    let device = decode_device(device_tag)?;
    let dtype = decode_dtype(dtype_tag)?;

    let rust_paths: Vec<&Path> = (0..n_paths)
        .map(|i| {
            let ptr = unsafe { *paths.add(i) };
            let s = unsafe { CStr::from_ptr(ptr) }
                .to_str()
                .expect("path not UTF-8");
            Path::new(s)
        })
        .collect();

    // SAFETY: the VarBuilder's mmap lifetime is tied to the encoder object
    // (BoxedEncoder), which is heap-allocated and lives until the caller frees it.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&rust_paths, dtype, &device)? };
    Ok((vb, device))
}

// ---------------------------------------------------------------------------
// Audio preprocessing
// ---------------------------------------------------------------------------

/// Mel constant: number of mel bins (128).
#[unsafe(no_mangle)]
pub extern "C" fn inferrs_mm_n_mel() -> usize {
    crate::audio::N_MEL
}

/// Max mel frames the audio encoder will process (1500).
#[unsafe(no_mangle)]
pub extern "C" fn inferrs_mm_max_mel_frames() -> usize {
    AudioEncoder::MAX_MEL_FRAMES
}

/// Decode audio bytes to mono f32 PCM samples.
///
/// `data`/`data_len` – input bytes.
/// `format`          – null-terminated format string ("wav" or "pcm_f32").
/// `out_samples`     – on success, written with a heap-allocated `f32` array.
/// `out_len`         – on success, written with the number of samples.
///
/// Returns 0 on success, non-zero on error (call `inferrs_mm_last_error()`).
/// The caller must free `*out_samples` via `inferrs_mm_free_f32`.
///
/// # Safety
/// `data` must point to `data_len` valid bytes. `format` must be a valid
/// null-terminated C string. `out_samples` and `out_len` must be non-null
/// writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferrs_mm_decode_audio(
    data: *const u8,
    data_len: usize,
    format: *const std::os::raw::c_char,
    out_samples: *mut *mut f32,
    out_len: *mut usize,
) -> i32 {
    let bytes = unsafe { std::slice::from_raw_parts(data, data_len) };
    let fmt = unsafe { CStr::from_ptr(format) }.to_str().unwrap_or("wav");

    match crate::audio::decode_audio(bytes, fmt) {
        Ok(samples) => {
            let len = samples.len();
            let mut boxed = samples.into_boxed_slice();
            unsafe {
                *out_samples = boxed.as_mut_ptr();
                *out_len = len;
            }
            std::mem::forget(boxed);
            0
        }
        Err(e) => {
            set_last_error(e);
            -1
        }
    }
}

/// Compute log-mel spectrogram from mono f32 PCM samples.
///
/// `samples`/`n_samples` – input PCM data.
/// `out_mel`             – on success, written with a heap-allocated row-major
///                         `[n_frames, N_MEL]` f32 array.
/// `out_n_frames`        – on success, written with the number of frames.
///
/// Returns 0 on success. Free `*out_mel` via `inferrs_mm_free_f32`.
///
/// # Safety
/// `samples` must point to `n_samples` valid `f32` values. `out_mel` and
/// `out_n_frames` must be non-null writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferrs_mm_compute_log_mel(
    samples: *const f32,
    n_samples: usize,
    out_mel: *mut *mut f32,
    out_n_frames: *mut usize,
) -> i32 {
    let slice = unsafe { std::slice::from_raw_parts(samples, n_samples) };
    match crate::audio::compute_log_mel(slice) {
        Ok((data, n_frames)) => {
            let mut boxed = data.into_boxed_slice();
            unsafe {
                *out_mel = boxed.as_mut_ptr();
                *out_n_frames = n_frames;
            }
            std::mem::forget(boxed);
            0
        }
        Err(e) => {
            set_last_error(e);
            -1
        }
    }
}

/// Free an `f32` array previously returned by this plugin.
///
/// # Safety
/// `ptr` must have been returned by a prior `inferrs_mm_*` call with the same
/// `len`, and must not be used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferrs_mm_free_f32(ptr: *mut f32, len: usize) {
    if !ptr.is_null() {
        unsafe {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)));
        }
    }
}

// ---------------------------------------------------------------------------
// Opaque encoder handle
// ---------------------------------------------------------------------------

/// Opaque handle wrapping either an `AudioEncoder` or `VisionEncoder`.
///
/// The `Device` is stored alongside the encoder so that `encode` calls reuse
/// the exact same `MetalDevice` instance that was used during weight loading.
/// On Metal, `same_device` checks pointer identity (not just the registry ID),
/// so creating a new `Device::new_metal(0)` on every encode call produces a
/// different identity and triggers a spurious "device mismatch" error.
enum BoxedEncoder {
    Audio(Box<AudioEncoder>, Device),
    Vision(Box<VisionEncoder>, Device),
}

// ---------------------------------------------------------------------------
// AudioEncoder
// ---------------------------------------------------------------------------

/// Load a Gemma4 audio encoder from safetensors weight files.
///
/// `paths`/`n_paths`        – array of null-terminated UTF-8 safetensors paths.
/// `audio_cfg_json`         – null-terminated JSON of `AudioConfig`.
/// `lm_hidden_size`         – language model hidden dimension.
/// `dtype_tag`/`device_tag` – 0-based dtype/device selectors (see module docs).
///
/// Returns a non-null opaque handle on success, or null on error.
/// Free with `inferrs_mm_audio_encoder_free`.
///
/// # Safety
/// `paths` must be a valid array of `n_paths` null-terminated UTF-8 C strings.
/// `audio_cfg_json` must be a valid null-terminated UTF-8 JSON string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferrs_mm_audio_encoder_load(
    paths: *const *const std::os::raw::c_char,
    n_paths: usize,
    audio_cfg_json: *const std::os::raw::c_char,
    lm_hidden_size: usize,
    dtype_tag: u8,
    device_tag: u8,
) -> *mut std::os::raw::c_void {
    let result = (|| -> Result<*mut std::os::raw::c_void> {
        let (vb, device) = unsafe { build_var_builder(paths, n_paths, dtype_tag, device_tag)? };
        let dtype = decode_dtype(dtype_tag)?;
        let cfg_str = unsafe { CStr::from_ptr(audio_cfg_json) }
            .to_str()
            .map_err(|e| anyhow::anyhow!("audio_cfg_json not UTF-8: {e}"))?;
        let cfg: AudioConfig = serde_json::from_str(cfg_str)?;
        let enc = AudioEncoder::load(vb.pp("model"), &cfg, lm_hidden_size, &device, dtype)?;
        let boxed = Box::new(BoxedEncoder::Audio(Box::new(enc), device));
        Ok(Box::into_raw(boxed) as *mut std::os::raw::c_void)
    })();
    match result {
        Ok(ptr) => ptr,
        Err(e) => {
            set_last_error(e);
            std::ptr::null_mut()
        }
    }
}

/// Encode a log-mel spectrogram to LM-space embeddings.
///
/// `handle`    – audio encoder handle from `inferrs_mm_audio_encoder_load`.
/// `mel`       – row-major f32 data of shape `[1, n_frames, N_MEL]`.
/// `n_frames`  – number of mel frames.
/// `device_tag`– device the encoder lives on.
/// `out_data`  – on success, written with heap-allocated f32 embedding data,
///               row-major `[n_out_tokens, lm_hidden_size]`.
/// `out_rows`  – number of output tokens.
/// `out_cols`  – lm_hidden_size.
///
/// Returns 0 on success. Free `*out_data` via `inferrs_mm_free_f32`.
///
/// # Safety
/// `handle` must be a valid handle from `inferrs_mm_audio_encoder_load`.
/// `mel` must point to `n_frames * N_MEL` valid `f32` values.
/// `out_data`, `out_rows`, `out_cols` must be non-null writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferrs_mm_audio_encoder_encode(
    handle: *mut std::os::raw::c_void,
    mel: *const f32,
    n_frames: usize,
    _device_tag: u8, // ignored: encoder reuses the device from load time
    out_data: *mut *mut f32,
    out_rows: *mut usize,
    out_cols: *mut usize,
) -> i32 {
    let result = (|| -> Result<()> {
        let enc = unsafe { &*(handle as *const BoxedEncoder) };
        let BoxedEncoder::Audio(audio_enc, device) = enc else {
            anyhow::bail!("handle is not an audio encoder");
        };

        let n_mel = crate::audio::N_MEL;
        let mel_slice = unsafe { std::slice::from_raw_parts(mel, n_frames * n_mel) };
        let mel_tensor = Tensor::from_slice(mel_slice, (1usize, n_frames, n_mel), device)?
            .to_dtype(DType::F32)?;

        let embeds = audio_enc.encode(&mel_tensor)?;
        let (rows, cols) = embeds.dims2()?;
        let data: Vec<f32> = embeds.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let mut boxed = data.into_boxed_slice();
        unsafe {
            *out_data = boxed.as_mut_ptr();
            *out_rows = rows;
            *out_cols = cols;
        }
        std::mem::forget(boxed);
        Ok(())
    })();
    match result {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(e);
            -1
        }
    }
}

/// Free an audio encoder handle.
///
/// # Safety
/// `handle` must be a valid handle from `inferrs_mm_audio_encoder_load` and
/// must not be used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferrs_mm_audio_encoder_free(handle: *mut std::os::raw::c_void) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle as *mut BoxedEncoder));
        }
    }
}

// ---------------------------------------------------------------------------
// VisionEncoder
// ---------------------------------------------------------------------------

/// Load a Gemma4 vision encoder from safetensors weight files.
///
/// `vision_cfg_json` – null-terminated JSON of `Gemma4VisionConfig`.
/// All other parameters mirror `inferrs_mm_audio_encoder_load`.
///
/// Returns a non-null opaque handle on success, or null on error.
/// Free with `inferrs_mm_vision_encoder_free`.
///
/// # Safety
/// `paths` must be a valid array of `n_paths` null-terminated UTF-8 C strings.
/// `vision_cfg_json` must be a valid null-terminated UTF-8 JSON string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferrs_mm_vision_encoder_load(
    paths: *const *const std::os::raw::c_char,
    n_paths: usize,
    vision_cfg_json: *const std::os::raw::c_char,
    lm_hidden_size: usize,
    dtype_tag: u8,
    device_tag: u8,
) -> *mut std::os::raw::c_void {
    let result = (|| -> Result<*mut std::os::raw::c_void> {
        let (vb, device) = unsafe { build_var_builder(paths, n_paths, dtype_tag, device_tag)? };
        let dtype = decode_dtype(dtype_tag)?;
        let cfg_str = unsafe { CStr::from_ptr(vision_cfg_json) }
            .to_str()
            .map_err(|e| anyhow::anyhow!("vision_cfg_json not UTF-8: {e}"))?;
        let cfg: Gemma4VisionConfig = serde_json::from_str(cfg_str)?;
        let enc = VisionEncoder::load(vb.pp("model"), &cfg, lm_hidden_size, &device, dtype)?;
        let boxed = Box::new(BoxedEncoder::Vision(Box::new(enc), device));
        Ok(Box::into_raw(boxed) as *mut std::os::raw::c_void)
    })();
    match result {
        Ok(ptr) => ptr,
        Err(e) => {
            set_last_error(e);
            std::ptr::null_mut()
        }
    }
}

/// Encode pre-patchified pixel values to LM-space embeddings.
///
/// `handle`         – vision encoder handle from `inferrs_mm_vision_encoder_load`.
/// `pixel_values`   – f32 row-major `[n_patches, patch_pixels]` in [0,1].
/// `pv_rows`        – n_patches.
/// `pv_cols`        – patch_pixels (patch_size² × 3).
/// `position_ids`   – i64 row-major `[n_patches, 2]` (x, y) coordinates.
/// `n_soft_tokens`  – requested output soft-token count.
/// `device_tag`     – device the encoder lives on.
/// `out_data`/`out_rows`/`out_cols` – as in `inferrs_mm_audio_encoder_encode`.
///
/// Returns 0 on success. Free `*out_data` via `inferrs_mm_free_f32`.
///
/// # Safety
/// `handle` must be a valid handle from `inferrs_mm_vision_encoder_load`.
/// `pixel_values` must point to `pv_rows * pv_cols` valid `f32` values.
/// `position_ids` must point to `pv_rows * 2` valid `i64` values.
/// `out_data`, `out_rows`, `out_cols` must be non-null writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferrs_mm_vision_encoder_encode(
    handle: *mut std::os::raw::c_void,
    pixel_values: *const f32,
    pv_rows: usize,
    pv_cols: usize,
    position_ids: *const i64,
    n_soft_tokens: usize,
    _device_tag: u8, // ignored: encoder reuses the device from load time
    out_data: *mut *mut f32,
    out_rows: *mut usize,
    out_cols: *mut usize,
) -> i32 {
    let result = (|| -> Result<()> {
        let enc = unsafe { &*(handle as *const BoxedEncoder) };
        let BoxedEncoder::Vision(vision_enc, device) = enc else {
            anyhow::bail!("handle is not a vision encoder");
        };

        let pv_slice = unsafe { std::slice::from_raw_parts(pixel_values, pv_rows * pv_cols) };
        let pixel_tensor = Tensor::from_slice(pv_slice, (pv_rows, pv_cols), device)?;

        let pos_slice = unsafe { std::slice::from_raw_parts(position_ids, pv_rows * 2) };
        let pos_tensor = Tensor::from_slice(pos_slice, (pv_rows, 2usize), device)?;

        let embeds = vision_enc.encode(&pixel_tensor, &pos_tensor, Some(n_soft_tokens))?;
        let (rows, cols) = embeds.dims2()?;
        let data: Vec<f32> = embeds.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let mut boxed = data.into_boxed_slice();
        unsafe {
            *out_data = boxed.as_mut_ptr();
            *out_rows = rows;
            *out_cols = cols;
        }
        std::mem::forget(boxed);
        Ok(())
    })();
    match result {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(e);
            -1
        }
    }
}

/// Free a vision encoder handle.
///
/// # Safety
/// `handle` must be a valid handle from `inferrs_mm_vision_encoder_load` and
/// must not be used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferrs_mm_vision_encoder_free(handle: *mut std::os::raw::c_void) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle as *mut BoxedEncoder));
        }
    }
}

//! Qualcomm FastRPC / Hexagon HTP driver loader.
//!
//! This module dynamically loads `libcdsprpc.so` (Linux / Android) or
//! `libcdsprpc.dll` (Windows ARM64) and resolves all required symbols via
//! `dlopen` / `LoadLibraryW`.  No link-time dependency on the Qualcomm SDK
//! is introduced — the backend plugin `.so` / `.dll` compiles and loads on
//! any system; only the probe call returns non-zero when the Hexagon NPU is
//! absent.
//!
//! The design mirrors `ggml/src/ggml-hexagon/htp-drv.cpp` from the llama.cpp
//! project.

use std::ffi::c_void;
use std::sync::OnceLock;

use crate::libdl::sys::{dl_error, dl_open, dl_sym, DlHandle};

// ── FastRPC opaque types (forward-declared as integer handles) ────────────────

/// Opaque 64-bit handle to an open remote FastRPC session.
pub type RemoteHandle64 = u64;

/// Opaque handle to a DSP async command queue.
pub type DspQueue = *mut c_void;

// ── FastRPC DSP capability query constants ────────────────────────────────────

/// Request code for `remote_handle_control` to query DSP information.
pub const DSPRPC_GET_DSP_INFO: u32 = 6;
/// Attribute ID for the Hexagon architecture version.
pub const ARCH_VER: u32 = 0;

/// FastRPC latency control request code (enable low-latency mode).
/// Used when opening a session to reduce round-trip latency for inference.
#[allow(dead_code)]
pub const DSPRPC_CONTROL_LATENCY: u32 = 1;

/// `remote_session_control` request: reserve a new DSP session slot.
/// Required for multi-session configurations (GGML_HEXAGON_NDEV > 1).
#[allow(dead_code)]
pub const FASTRPC_RESERVE_NEW_SESSION: u32 = 1;

/// `remote_session_control` request: retrieve the session URI.
/// Used to build the `file:///libggml-htp-v{arch}.so?...` URI for
/// `remote_handle64_open`.
#[allow(dead_code)]
pub const FASTRPC_GET_URI: u32 = 2;

/// `remote_session_control` request: enable unsigned DSP PD.
/// Required to load unsigned DSP skeleton libraries during development.
#[allow(dead_code)]
pub const DSPRPC_CONTROL_UNSIGNED_MODULE: u32 = 3;

/// Raw DSP capability structure passed to `DSPRPC_GET_DSP_INFO`.
#[repr(C)]
pub struct RemoteDspCapability {
    pub domain: u32,
    pub attribute_id: u32,
    pub capability: u32,
}

// ── Function pointer types ────────────────────────────────────────────────────

type RpcmemAllocFn = unsafe extern "C" fn(heapid: i32, flags: u32, size: i32) -> *mut c_void;
type RpcmemAlloc2Fn = unsafe extern "C" fn(heapid: i32, flags: u32, size: usize) -> *mut c_void;
type RpcmemFreeFn = unsafe extern "C" fn(po: *mut c_void);
type RpcmemToFdFn = unsafe extern "C" fn(po: *mut c_void) -> i32;

type FastrpcMmapFn = unsafe extern "C" fn(
    domain: i32,
    fd: i32,
    addr: *mut c_void,
    offset: i32,
    length: usize,
    flags: u32,
) -> i32;
type FastrpcMunmapFn =
    unsafe extern "C" fn(domain: i32, fd: i32, addr: *mut c_void, length: usize) -> i32;

type DspqueueCreateFn = unsafe extern "C" fn(
    domain: i32,
    flags: u32,
    req_queue_size: u32,
    resp_queue_size: u32,
    packet_callback: *mut c_void,
    error_callback: *mut c_void,
    callback_context: *mut c_void,
    queue: *mut DspQueue,
) -> i32;
type DspqueueCloseFn = unsafe extern "C" fn(queue: DspQueue) -> i32;
type DspqueueExportFn = unsafe extern "C" fn(queue: DspQueue, queue_id: *mut u64) -> i32;
type DspqueueWriteFn = unsafe extern "C" fn(
    queue: DspQueue,
    flags: u32,
    num_buffers: u32,
    buffers: *mut c_void,
    message_length: u32,
    message: *const u8,
    timeout_us: u32,
) -> i32;
type DspqueueReadFn = unsafe extern "C" fn(
    queue: DspQueue,
    flags: *mut u32,
    max_buffers: u32,
    num_buffers: *mut u32,
    buffers: *mut c_void,
    max_message_length: u32,
    message_length: *mut u32,
    message: *mut u8,
    timeout_us: u32,
) -> i32;

type RemoteHandle64OpenFn =
    unsafe extern "C" fn(name: *const libc::c_char, ph: *mut RemoteHandle64) -> i32;
type RemoteHandle64InvokeFn =
    unsafe extern "C" fn(h: RemoteHandle64, dw_scalars: u32, pra: *mut c_void) -> i32;
type RemoteHandle64CloseFn = unsafe extern "C" fn(h: RemoteHandle64) -> i32;
type RemoteHandleControlFn = unsafe extern "C" fn(req: u32, data: *mut c_void, datalen: u32) -> i32;
type RemoteHandle64ControlFn =
    unsafe extern "C" fn(h: RemoteHandle64, req: u32, data: *mut c_void, datalen: u32) -> i32;
type RemoteSessionControlFn =
    unsafe extern "C" fn(req: u32, data: *mut c_void, datalen: u32) -> i32;

// ── Loaded driver state ───────────────────────────────────────────────────────

/// All required FastRPC function pointers, populated after `htpdrv_init()`.
///
/// Fields beyond `remote_handle_control` (which is used in the probe) are
/// resolved eagerly so that any library ABI mismatch is caught at load time
/// rather than at first use.  They will be exercised once candle grows a
/// Hexagon device variant.
#[allow(dead_code)]
struct FastRpcDriver {
    // Keep the handle alive so the symbols remain valid.
    _lib: DlHandle,

    pub rpcmem_alloc: RpcmemAllocFn,
    pub rpcmem_alloc2: Option<RpcmemAlloc2Fn>, // optional — only on newer SDKs
    pub rpcmem_free: RpcmemFreeFn,
    pub rpcmem_to_fd: RpcmemToFdFn,

    pub fastrpc_mmap: FastrpcMmapFn,
    pub fastrpc_munmap: FastrpcMunmapFn,

    pub dspqueue_create: DspqueueCreateFn,
    pub dspqueue_close: DspqueueCloseFn,
    pub dspqueue_export: DspqueueExportFn,
    pub dspqueue_write: DspqueueWriteFn,
    pub dspqueue_read: DspqueueReadFn,

    pub remote_handle64_open: RemoteHandle64OpenFn,
    pub remote_handle64_invoke: RemoteHandle64InvokeFn,
    pub remote_handle64_close: RemoteHandle64CloseFn,
    pub remote_handle_control: RemoteHandleControlFn,
    pub remote_handle64_control: RemoteHandle64ControlFn,
    pub remote_session_control: RemoteSessionControlFn,
}

// SAFETY: All function pointers are valid for the lifetime of the loaded
// library, and FastRpcDriver is only constructed once (via OnceLock).
unsafe impl Send for FastRpcDriver {}
unsafe impl Sync for FastRpcDriver {}

/// Global singleton — initialised at most once.
static DRIVER: OnceLock<FastRpcDriver> = OnceLock::new();

// ── Platform-specific library path resolution ─────────────────────────────────

/// Return the path to `libcdsprpc` for the current platform.
///
/// - **Linux / Android**: `"libcdsprpc.so"` — resolved by the dynamic linker
///   search path, which on Android includes `/vendor/lib64`.
/// - **macOS**: Hexagon is Qualcomm-only silicon; macOS is x86_64 or Apple
///   Silicon (ARM64).  Neither ships Hexagon DSP hardware, so we return `None`
///   immediately to signal that the probe should fail gracefully.
/// - **Windows ARM64**: query the Service Control Manager for the
///   `qcnspmcdm` service binary path to locate the driver store directory
///   where `libcdsprpc.dll` resides.
/// - **Windows x86_64**: no Hexagon hardware exists; return `None`.
fn cdsprpc_lib_path() -> Option<String> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        Some("libcdsprpc.so".to_owned())
    }

    // macOS (x86_64 or aarch64 / Apple Silicon): no Hexagon hardware.
    #[cfg(target_os = "macos")]
    {
        None
    }

    // Windows ARM64 only: locate libcdsprpc.dll via SCM.
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        windows_driver_path()
    }

    // Windows x86_64: no Hexagon hardware.
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        None
    }
}

/// On Windows ARM64, query the `qcnspmcdm` SCM service to find the driver
/// store directory containing `libcdsprpc.dll`.
///
/// The service binary path looks like:
/// `\SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_...\...`
/// We strip `\SystemRoot`, substitute the actual `%windir%` value, take the
/// parent directory, and append `\libcdsprpc.dll`.
#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
fn windows_driver_path() -> Option<String> {
    use std::ffi::OsString;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use windows_sys::Win32::Foundation::{ERROR_INSUFFICIENT_BUFFER, NO_ERROR};
    use windows_sys::Win32::System::Services::{
        CloseServiceHandle, OpenSCManagerW, OpenServiceW, QueryServiceConfigW,
        QUERY_SERVICE_CONFIGW, SC_MANAGER_CONNECT, SERVICE_QUERY_CONFIG,
    };

    const SERVICE_NAME: &[u16] = &[
        b'q' as u16,
        b'c' as u16,
        b'n' as u16,
        b's' as u16,
        b'p' as u16,
        b'm' as u16,
        b'c' as u16,
        b'd' as u16,
        b'm' as u16,
        0u16,
    ];

    // SAFETY: Windows SCM API calls with valid parameters.
    unsafe {
        let scm = OpenSCManagerW(
            std::ptr::null(),
            std::ptr::null(),
            SC_MANAGER_CONNECT as u32,
        );
        if scm.is_null() {
            return None;
        }

        let svc = OpenServiceW(scm, SERVICE_NAME.as_ptr(), SERVICE_QUERY_CONFIG as u32);
        if svc.is_null() {
            CloseServiceHandle(scm);
            return None;
        }

        // First call to get the required buffer size.
        let mut buf_size: u32 = 0;
        QueryServiceConfigW(svc, std::ptr::null_mut(), 0, &mut buf_size);
        if buf_size == 0 {
            CloseServiceHandle(svc);
            CloseServiceHandle(scm);
            return None;
        }

        // Use Vec<u64> to guarantee 8-byte alignment. QUERY_SERVICE_CONFIGW
        // contains pointers and requires pointer-sized alignment (8 bytes on
        // aarch64). Vec<u8> only guarantees 1-byte alignment which would cause
        // UB when casting to *const QUERY_SERVICE_CONFIGW.
        let mut buf: Vec<u64> = vec![0u64; (buf_size as usize + 7) / 8];
        if QueryServiceConfigW(
            svc,
            buf.as_mut_ptr() as *mut QUERY_SERVICE_CONFIGW,
            buf_size,
            &mut buf_size,
        ) == 0
        {
            CloseServiceHandle(svc);
            CloseServiceHandle(scm);
            return None;
        }

        CloseServiceHandle(svc);
        CloseServiceHandle(scm);

        // SAFETY: buf is 8-byte aligned (Vec<u64>) and QueryServiceConfigW
        // has written a valid QUERY_SERVICE_CONFIGW into it.
        let cfg = &*(buf.as_ptr() as *const QUERY_SERVICE_CONFIGW);
        if cfg.lpBinaryPathName.is_null() {
            return None;
        }

        // Convert the wide-string path to a Rust OsString.
        let mut len = 0usize;
        while *cfg.lpBinaryPathName.add(len) != 0 {
            len += 1;
        }
        let wide_slice = std::slice::from_raw_parts(cfg.lpBinaryPathName, len);
        let path_os = OsString::from_wide(wide_slice);
        let path_str = path_os.to_string_lossy().into_owned();

        // Strip parent filename to get directory, replace \SystemRoot.
        let dir = path_str.rfind('\\').map(|i| &path_str[..i])?;
        let system_root_prefix = r"\SystemRoot";
        if !dir.starts_with(system_root_prefix) {
            return None;
        }

        let windir = std::env::var("windir").ok()?;
        let resolved = format!("{}{}", windir, &dir[system_root_prefix.len()..]);
        Some(format!(r"{}\libcdsprpc.dll", resolved))
    }
}

// ── Symbol resolution helper macro ───────────────────────────────────────────

/// Resolve a required symbol from the loaded library handle.
///
/// Returns `Err` if the symbol is missing.
macro_rules! require_sym {
    ($handle:expr, $ty:ty, $name:literal) => {{
        // SAFETY: We cast the raw pointer to the declared function pointer type.
        // The caller is responsible for declaring the correct type.
        match unsafe { dl_sym($handle, $name) } {
            Some(ptr) => Ok(unsafe { std::mem::transmute::<*mut c_void, $ty>(ptr) }),
            None => Err(format!("symbol '{}' not found: {}", $name, dl_error())),
        }
    }};
}

/// Resolve an optional symbol; returns `None` on failure instead of an error.
macro_rules! optional_sym {
    ($handle:expr, $ty:ty, $name:literal) => {{
        unsafe { dl_sym($handle, $name) }
            .map(|ptr| unsafe { std::mem::transmute::<*mut c_void, $ty>(ptr) })
    }};
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Initialise the FastRPC driver.
///
/// Opens `libcdsprpc.so` / `libcdsprpc.dll`, resolves all required symbols,
/// and stores the driver in a global `OnceLock`.
///
/// Returns `Ok(())` on success, or an error string describing the first
/// failure.  Repeated calls are no-ops (returns `Ok(())` if already loaded).
pub fn htpdrv_init() -> Result<(), String> {
    if DRIVER.get().is_some() {
        return Ok(());
    }

    let path = cdsprpc_lib_path()
        .ok_or_else(|| "Hexagon HTP is not supported on this platform".to_owned())?;

    let lib = dl_open(&path).ok_or_else(|| format!("failed to open {}: {}", path, dl_error()))?;

    let drv = FastRpcDriver {
        rpcmem_alloc: require_sym!(&lib, RpcmemAllocFn, "rpcmem_alloc")?,
        rpcmem_alloc2: optional_sym!(&lib, RpcmemAlloc2Fn, "rpcmem_alloc2"),
        rpcmem_free: require_sym!(&lib, RpcmemFreeFn, "rpcmem_free")?,
        rpcmem_to_fd: require_sym!(&lib, RpcmemToFdFn, "rpcmem_to_fd")?,

        fastrpc_mmap: require_sym!(&lib, FastrpcMmapFn, "fastrpc_mmap")?,
        fastrpc_munmap: require_sym!(&lib, FastrpcMunmapFn, "fastrpc_munmap")?,

        dspqueue_create: require_sym!(&lib, DspqueueCreateFn, "dspqueue_create")?,
        dspqueue_close: require_sym!(&lib, DspqueueCloseFn, "dspqueue_close")?,
        dspqueue_export: require_sym!(&lib, DspqueueExportFn, "dspqueue_export")?,
        dspqueue_write: require_sym!(&lib, DspqueueWriteFn, "dspqueue_write")?,
        dspqueue_read: require_sym!(&lib, DspqueueReadFn, "dspqueue_read")?,

        remote_handle64_open: require_sym!(&lib, RemoteHandle64OpenFn, "remote_handle64_open")?,
        remote_handle64_invoke: require_sym!(
            &lib,
            RemoteHandle64InvokeFn,
            "remote_handle64_invoke"
        )?,
        remote_handle64_close: require_sym!(&lib, RemoteHandle64CloseFn, "remote_handle64_close")?,
        remote_handle_control: require_sym!(&lib, RemoteHandleControlFn, "remote_handle_control")?,
        remote_handle64_control: require_sym!(
            &lib,
            RemoteHandle64ControlFn,
            "remote_handle64_control"
        )?,
        remote_session_control: require_sym!(
            &lib,
            RemoteSessionControlFn,
            "remote_session_control"
        )?,

        _lib: lib,
    };

    // If another thread raced us to initialise, drop our copy and return.
    let _ = DRIVER.set(drv);
    Ok(())
}

/// Query the Hexagon DSP architecture version from the currently loaded driver.
///
/// Uses `remote_handle_control(DSPRPC_GET_DSP_INFO, …)` to query the Hexagon
/// compute domain on `domain_id` (typically `3` for the compute DSP).
///
/// Returns the integer architecture version (`68`, `69`, `73`, `75`, `79`, or
/// `81`) on success, or an error string if the query fails or the version is
/// unrecognised.
///
/// # Precondition
/// `htpdrv_init()` must have been called and succeeded.
pub fn get_hex_arch_ver(domain_id: u32) -> Result<u32, String> {
    let drv = DRIVER
        .get()
        .ok_or_else(|| "FastRPC driver not initialised".to_owned())?;

    let mut cap = RemoteDspCapability {
        domain: domain_id,
        attribute_id: ARCH_VER,
        capability: 0,
    };

    // SAFETY: `remote_handle_control` is a valid resolved function pointer.
    // The capability struct is correctly sized and aligned.
    let err = unsafe {
        (drv.remote_handle_control)(
            DSPRPC_GET_DSP_INFO,
            &mut cap as *mut RemoteDspCapability as *mut c_void,
            std::mem::size_of::<RemoteDspCapability>() as u32,
        )
    };

    if err != 0 {
        return Err(format!("DSPRPC_GET_DSP_INFO failed with error {err:#x}"));
    }

    // Map the raw capability byte to the human-readable version number.
    match cap.capability & 0xff {
        0x68 => Ok(68),
        0x69 => Ok(69),
        0x73 => Ok(73),
        0x75 => Ok(75),
        0x79 => Ok(79),
        0x81 => Ok(81),
        other => Err(format!(
            "unrecognised Hexagon arch capability byte: {other:#x}"
        )),
    }
}

/// Allocate a physically-contiguous, CPU/DSP-shared buffer via `rpcmem_alloc2`
/// (if available) or `rpcmem_alloc` as a fallback.
///
/// Returns a raw pointer to the allocation on success, or `null` on failure.
///
/// # Safety
/// The returned memory is owned by the caller and must be freed with
/// [`rpcmem_free`].
#[allow(dead_code)]
pub unsafe fn rpcmem_alloc(heapid: i32, flags: u32, size: usize) -> *mut c_void {
    let drv = match DRIVER.get() {
        Some(d) => d,
        None => return std::ptr::null_mut(),
    };

    if let Some(alloc2) = drv.rpcmem_alloc2 {
        // SAFETY: alloc2 is a valid resolved function pointer.
        unsafe { alloc2(heapid, flags, size) }
    } else {
        // The legacy rpcmem_alloc takes an i32 size. Reject allocations that
        // exceed i32::MAX rather than silently truncating, which could cause
        // the caller to receive a far-smaller buffer than requested and corrupt
        // heap memory on first use.
        let size_i32 = match i32::try_from(size) {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };
        // SAFETY: rpcmem_alloc is a valid resolved function pointer.
        unsafe { (drv.rpcmem_alloc)(heapid, flags, size_i32) }
    }
}

/// Free a buffer previously allocated by [`rpcmem_alloc`].
///
/// # Safety
/// `ptr` must be a pointer returned by `rpcmem_alloc` that has not yet been
/// freed.
#[allow(dead_code)]
pub unsafe fn rpcmem_free(ptr: *mut c_void) {
    if let Some(drv) = DRIVER.get() {
        // SAFETY: ptr is a valid rpcmem allocation.
        unsafe { (drv.rpcmem_free)(ptr) };
    }
}

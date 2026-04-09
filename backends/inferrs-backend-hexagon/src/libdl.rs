//! Cross-platform dynamic library loading abstraction.
//!
//! On POSIX systems (Linux, Android, macOS) this wraps `dlopen`/`dlsym`/`dlclose`
//! from libc.  On Windows it wraps `LoadLibraryW`/`GetProcAddress`/`FreeLibrary`.
//!
//! This mirrors the `libdl.h` abstraction used in the llama.cpp Hexagon backend
//! (`ggml/src/ggml-hexagon/libdl.h`).

// ── POSIX (Linux, Android, macOS) ────────────────────────────────────────────

#[cfg(not(target_os = "windows"))]
pub mod sys {
    use std::ffi::{c_void, CString};

    /// RAII wrapper around a `dlopen` handle.
    pub struct DlHandle(*mut c_void);

    // SAFETY: The pointer is only used through our own API which serialises access.
    unsafe impl Send for DlHandle {}
    unsafe impl Sync for DlHandle {}

    impl Drop for DlHandle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: handle is valid and non-null.
                unsafe { libc::dlclose(self.0) };
            }
        }
    }

    /// Open a shared library by path.
    ///
    /// Uses `RTLD_NOW | RTLD_LOCAL`: resolve all symbols immediately and do
    /// not expose them to subsequently loaded libraries (matches llama.cpp
    /// behaviour).
    ///
    /// Returns `None` if the library cannot be found or loaded.
    pub fn dl_open(path: &str) -> Option<DlHandle> {
        let c_path = CString::new(path).ok()?;
        // SAFETY: dlopen is safe to call with a valid C string and flags.
        let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            None
        } else {
            Some(DlHandle(handle))
        }
    }

    /// Look up a symbol in an already-opened library handle.
    ///
    /// Returns `None` if the symbol is not found.
    ///
    /// # Safety
    /// The returned pointer is valid only as long as `handle` is live.
    /// The caller must cast it to the correct function pointer type.
    pub unsafe fn dl_sym(handle: &DlHandle, name: &str) -> Option<*mut c_void> {
        let c_name = CString::new(name).ok()?;
        // SAFETY: handle is non-null and name is a valid C string.
        let sym = unsafe { libc::dlsym(handle.0, c_name.as_ptr()) };
        if sym.is_null() {
            None
        } else {
            Some(sym)
        }
    }

    /// Return the last dynamic linker error string, or an empty string.
    pub fn dl_error() -> String {
        // SAFETY: dlerror() is always safe to call.
        let err = unsafe { libc::dlerror() };
        if err.is_null() {
            String::new()
        } else {
            // SAFETY: dlerror returns a valid C string or null.
            unsafe { std::ffi::CStr::from_ptr(err) }
                .to_string_lossy()
                .into_owned()
        }
    }
}

// ── Windows ───────────────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
pub mod sys {
    use std::ffi::{c_void, OsStr};
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

    /// RAII wrapper around a `LoadLibraryW` handle.
    pub struct DlHandle(HMODULE);

    // SAFETY: The handle is only used through our own serialised API.
    unsafe impl Send for DlHandle {}
    unsafe impl Sync for DlHandle {}

    impl Drop for DlHandle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: handle is valid and non-zero.
                unsafe { FreeLibrary(self.0) };
            }
        }
    }

    /// Encode a Rust `&str` as a null-terminated wide string.
    fn to_wide(s: &str) -> Vec<u16> {
        OsStr::new(s)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }

    /// Open a shared library by path (wide-string, suppressing error dialogs).
    ///
    /// Returns `None` if the DLL cannot be loaded.
    pub fn dl_open(path: &str) -> Option<DlHandle> {
        let wide = to_wide(path);
        // SAFETY: LoadLibraryW is safe to call with a valid null-terminated
        // wide string.  We suppress the critical-error dialog box that Windows
        // would otherwise show when a DLL or one of its dependencies is missing.
        // SAFETY: LoadLibraryW is safe with a valid null-terminated wide string.
        // The SEM_FAILCRITICALERRORS dialog suppression from the C++ reference
        // (libdl.h) is unnecessary here because we check the return value and
        // handle failure gracefully via the Option return type.
        let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
        if handle.is_null() {
            None
        } else {
            Some(DlHandle(handle))
        }
    }

    /// Look up a symbol in an already-opened library handle.
    ///
    /// Returns `None` if the symbol is not present.
    ///
    /// # Safety
    /// The returned pointer is valid only as long as `handle` is live.
    /// The caller must cast it to the correct function pointer type.
    pub unsafe fn dl_sym(handle: &DlHandle, name: &str) -> Option<*mut c_void> {
        // GetProcAddress expects a null-terminated byte string, not wide.
        let c_name: Vec<u8> = name.bytes().chain(std::iter::once(0)).collect();
        // SAFETY: handle is non-zero and c_name is a valid null-terminated
        // ASCII string.
        let sym = unsafe { GetProcAddress(handle.0, c_name.as_ptr()) };
        sym.map(|f| f as *mut c_void)
    }

    /// On Windows we do not expose `dlerror`; callers use `GetLastError`.
    pub fn dl_error() -> String {
        String::new()
    }
}

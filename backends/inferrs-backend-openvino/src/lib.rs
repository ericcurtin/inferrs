//! Probe whether an OpenVINO runtime is available on this system.
//!
//! This plugin is a pure probe — it does **not** link against OpenVINO at
//! compile time.  Instead it uses `dlopen` (Linux/Android/macOS) or
//! `LoadLibraryW` (Windows) to discover the OpenVINO runtime library at
//! runtime, mirroring the pattern used by `inferrs-backend-vulkan`.
//!
//! # Library names probed per platform
//!
//! | Platform          | Library name(s)                                        |
//! |-------------------|--------------------------------------------------------|
//! | Linux x86_64      | `libopenvino.so.2025.2`, `libopenvino.so.2024.6`, `libopenvino.so` |
//! | Linux aarch64     | same as x86_64                                         |
//! | Android aarch64   | `libopenvino.so`                                       |
//! | macOS x86_64      | `libopenvino.dylib`, `libopenvino.2025.2.dylib`        |
//! | macOS aarch64     | same as macOS x86_64                                   |
//! | Windows x86_64    | `openvino.dll`                                         |
//! | Windows aarch64   | `openvino.dll`                                         |
//!
//! OpenVINO supports x86_64 and aarch64; no other CPU architectures are
//! targeted by Intel.  The probe returns non-zero (unavailable) on any other
//! architecture or OS.
//!
//! # Return value
//!
//! `0`  — OpenVINO runtime library was found and loaded successfully.
//! `1`  — Library not found or failed to load.

// ── Linux ─────────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
mod linux_probe {
    use std::ffi::CString;

    /// Versioned and unversioned candidate library names for Linux x86_64/aarch64.
    ///
    /// Versioned names are tried first so that the probe binds to a specific,
    /// known-good release rather than whatever symlink `libopenvino.so`
    /// happens to point to.  The list matches the versions OpenVINO ships in
    /// its official installers and pip wheels as of 2025.
    ///
    /// The naming convention is identical on x86_64 and aarch64 Linux.
    const CANDIDATES: &[&str] = &[
        "libopenvino.so.2025.2",
        "libopenvino.so.2025.1",
        "libopenvino.so.2024.6",
        "libopenvino.so.2024.5",
        "libopenvino.so",
    ];

    /// Try to open any of the candidate library names via `dlopen`.
    /// Returns `true` if at least one name succeeded.
    pub fn probe() -> bool {
        for name in CANDIDATES {
            let Ok(cname) = CString::new(*name) else {
                continue;
            };
            // SAFETY: `dlopen` is safe to call with a valid C string and
            // standard flags.  We immediately `dlclose` the handle — we only
            // need to know whether the library can be found.
            let handle =
                unsafe { libc::dlopen(cname.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
            if !handle.is_null() {
                unsafe { libc::dlclose(handle) };
                return true;
            }
        }
        false
    }
}

// ── Android ───────────────────────────────────────────────────────────────────
//
// OpenVINO for Android targets aarch64 devices only.  The library is packaged
// as an unversioned `.so` — the versioned symlink convention used on Linux
// desktop distributions does not apply in the Android ecosystem.

#[cfg(target_os = "android")]
mod android_probe {
    use std::ffi::CString;

    /// Android packages OpenVINO as a plain unversioned shared library.
    const CANDIDATES: &[&str] = &["libopenvino.so"];

    pub fn probe() -> bool {
        for name in CANDIDATES {
            let Ok(cname) = CString::new(*name) else {
                continue;
            };
            // SAFETY: same as the Linux case above.
            let handle =
                unsafe { libc::dlopen(cname.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
            if !handle.is_null() {
                unsafe { libc::dlclose(handle) };
                return true;
            }
        }
        false
    }
}

// ── macOS ─────────────────────────────────────────────────────────────────────
//
// OpenVINO provides a macOS distribution for both x86_64 and Apple Silicon
// (aarch64).  The official installer places the runtime as a `.dylib` in
// the chosen prefix, e.g. `/opt/intel/openvino/lib/libopenvino.dylib`.

#[cfg(target_os = "macos")]
mod macos_probe {
    use std::ffi::CString;

    /// macOS ships both versioned and unversioned dylibs.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    const CANDIDATES: &[&str] = &[
        "libopenvino.2025.2.dylib",
        "libopenvino.2025.1.dylib",
        "libopenvino.2024.6.dylib",
        "libopenvino.2024.5.dylib",
        "libopenvino.dylib",
    ];

    pub fn probe() -> bool {
        for name in CANDIDATES {
            let Ok(cname) = CString::new(*name) else {
                continue;
            };
            // SAFETY: same as the Linux case above.
            let handle =
                unsafe { libc::dlopen(cname.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
            if !handle.is_null() {
                unsafe { libc::dlclose(handle) };
                return true;
            }
        }
        false
    }
}

// ── Windows ───────────────────────────────────────────────────────────────────
//
// On Windows OpenVINO is distributed as `openvino.dll` (no `lib` prefix, no
// version suffix in the DLL name itself — the version lives in the installer
// path).  Both x86_64 and ARM64 Windows are supported by Intel since 2024.1.
//
// We use the raw Win32 `LoadLibraryW` / `FreeLibrary` API via `windows-sys`
// rather than `libloading` so that this plugin carries no heavyweight
// dependency that would complicate cross-compilation.

#[cfg(target_os = "windows")]
mod windows_probe {
    use std::ffi::OsStr;
    use std::iter::once;
    use std::os::windows::ffi::OsStrExt;

    use windows_sys::Win32::Foundation::HMODULE;
    use windows_sys::Win32::System::LibraryLoader::{FreeLibrary, LoadLibraryW};

    /// Candidate DLL names on Windows.
    const CANDIDATES: &[&str] = &[
        "openvino.dll",
        // The redistributable installer may also place a versioned copy.
        "openvino_2025_2.dll",
        "openvino_2024_6.dll",
    ];

    /// Encode a `&str` as a null-terminated wide string for `LoadLibraryW`.
    fn to_wide_null(s: &str) -> Vec<u16> {
        OsStr::new(s).encode_wide().chain(once(0u16)).collect()
    }

    pub fn probe() -> bool {
        for name in CANDIDATES {
            let wide = to_wide_null(name);
            // SAFETY: `LoadLibraryW` is safe to call with a valid pointer to a
            // null-terminated wide string.  We immediately `FreeLibrary` the
            // handle — we only need to know whether the DLL can be found.
            let handle: HMODULE = unsafe { LoadLibraryW(wide.as_ptr()) };
            if handle != 0 {
                unsafe { FreeLibrary(handle) };
                return true;
            }
        }
        false
    }
}

// ── Public probe entry point ──────────────────────────────────────────────────

/// Probe whether OpenVINO is available on this system.
///
/// Called by the `inferrs` binary after `dlopen`ing this plugin.
/// Returns `0` if the OpenVINO runtime library was found, `1` otherwise.
///
/// Supported configurations:
/// - Linux x86_64 and aarch64 (including Raspberry Pi 5, Jetson)
/// - Android aarch64
/// - macOS x86_64 and aarch64 (Apple Silicon)
/// - Windows x86_64 and aarch64
///
/// On any other platform the probe always returns `1` (unavailable).
#[no_mangle]
pub extern "C" fn inferrs_backend_probe() -> i32 {
    #[cfg(target_os = "linux")]
    {
        // OpenVINO official Linux builds target x86_64 and aarch64 only.
        // Return unavailable on other arches (riscv64, mips, etc.) without probing.
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            return 1;
        }

        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        if linux_probe::probe() {
            0
        } else {
            1
        }
    }

    #[cfg(target_os = "android")]
    {
        // OpenVINO for Android targets aarch64 devices only; no x86_64 build exists.
        #[cfg(not(target_arch = "aarch64"))]
        {
            return 1;
        }

        #[cfg(target_arch = "aarch64")]
        if android_probe::probe() {
            0
        } else {
            1
        }
    }

    #[cfg(target_os = "macos")]
    {
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            return 1;
        }

        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        if macos_probe::probe() {
            0
        } else {
            1
        }
    }

    #[cfg(target_os = "windows")]
    {
        if windows_probe::probe() {
            0
        } else {
            1
        }
    }

    // Any other OS (BSDs, Fuchsia, etc.) — OpenVINO is not available.
    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "windows",
    )))]
    {
        1
    }
}

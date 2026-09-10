//! Safe wrappers around the CGO-exported Go shim functions.
//!
//! Every Go function returns a JSON-encoded `{"ok":bool,"data":"...","error":"..."}`.
//! The wrappers decode this envelope and surface Rust `Result`s.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use anyhow::{anyhow, Context};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Raw FFI declarations — symbols produced by the Go shim static archive
// ---------------------------------------------------------------------------
extern "C" {
    fn llmman_free(s: *mut c_char);
    fn llmman_login(
        server: *const c_char,
        username: *const c_char,
        password: *const c_char,
    ) -> *mut c_char;
    fn llmman_logout(server: *const c_char) -> *mut c_char;
    fn llmman_push(layout_dir: *const c_char, reference: *const c_char) -> *mut c_char;
    fn llmman_pull(reference: *const c_char, layout_dir: *const c_char) -> *mut c_char;
    fn llmman_inspect(reference: *const c_char) -> *mut c_char;
    fn llmman_transfer(source: *const c_char, destination: *const c_char) -> *mut c_char;
    fn llmman_progress(key: *const c_char) -> *mut c_char;
    fn llmman_verify(
        reference: *const c_char,
        digest: *const c_char,
        keys_json: *const c_char,
    ) -> *mut c_char;
    fn llmman_sign(
        reference: *const c_char,
        digest: *const c_char,
        key_path: *const c_char,
        password: *const c_char,
    ) -> *mut c_char;
    fn llmman_resolve_digest(reference: *const c_char) -> *mut c_char;
    fn llmman_set_registry_mirrors(mirrors_json: *const c_char) -> *mut c_char;
}

// go-shim/push_stream.go is `!podman`-only: containerd's docker.Resolver
// has a per-blob streaming push primitive, podman's copy.Image doesn't
// (see `crate::hf::transfer`'s doc comment for its podman fallback).
#[cfg(feature = "docker")]
extern "C" {
    fn llmman_push_session_open(destination: *const c_char) -> *mut c_char;
    fn llmman_push_session_close(session: i64) -> *mut c_char;
    fn llmman_push_stream_open(
        session: i64,
        media_type: *const c_char,
        digest: *const c_char,
        size: i64,
        annotations_json: *const c_char,
    ) -> *mut c_char;
    fn llmman_push_stream_wait(handle: i64) -> *mut c_char;
}

// ---------------------------------------------------------------------------
// Windows-only: manual Go runtime bootstrap
// ---------------------------------------------------------------------------
//
// On Linux (ELF) and macOS (Mach-O), `go build -buildmode=c-archive`'s
// runtime-init entry point (`_rt0_<arch>_lib`) gets registered as a real
// global constructor that the platform's own C runtime startup invokes
// automatically before `main()` ever runs (`.init_array`/`__mod_init_
// func`), exactly as cgo's docs promise — every FFI call above just works.
//
// On windows-gnu (MinGW-w64's own ld + CRT startup) this mechanism is
// emitted differently but still works: `cmd/link` (Go's own linker —
// this is unconditional, regardless of which C compiler compiled the
// shim's cgo object files) always emits an old-style GNU `.ctors` COFF
// section for this (see $GOROOT/src/cmd/link/internal/ld/pe.go's
// addInitArray), and MinGW-w64's CRT startup scans exactly that section
// for constructors to run before `main()` — confirmed locally with a
// minimal repro. But that is *not* a convention the MSVC/UCRT startup
// code that Rust's own *-pc-windows-msvc targets link against (via
// lld-link/link.exe, see build.rs) understands at all; it looks for
// `.CRT$XCU` instead. The net effect — confirmed by reproducing it
// locally against the exact same clang+lld-link toolchain this
// project's Windows CI build uses, with this crate's own real go-shim
// archive, and by reading $GOROOT's own runtime/cgo and cmd/link source
// — is that the constructor meant to kick off the Go runtime's
// background init thread (the one that unblocks every cgo call's own
// `_cgo_wait_runtime_init_done()`, which cmd/cgo generates into the
// front of every `//export`ed function — see backend_docker.go's
// llmman_pull) never runs on an MSVC target, and the very first call
// into this shim from anywhere in the process hangs forever waiting on
// an event nothing will ever signal. This was the cause of the
// Windows-only `llmman_pull` hang tracked across this repo's earlier
// trace-logging commits, which had already narrowed it down to
// "somewhere before llmman_pull's own first line ever runs" — i.e.
// squarely in this gap.
//
// The fix: call the Go runtime's own entry point ourselves. It's an
// ordinary exported symbol (`_rt0_<arch>_windows_lib`, see
// $GOROOT/src/runtime/rt0_windows_{amd64,arm64}.s) that's always present
// in the archive whether or not anything actually invokes it as a
// constructor. Calling it directly, exactly once, before any other call
// into this shim, starts the same background init thread the (non-
// functional, on this target) automatic constructor was always meant to
// start; every generated `_cgo_wait_runtime_init_done()` call proceeds
// normally after that.
//
// Scoped to `target_env = "msvc"` specifically (not all of `windows`):
// on windows-gnu the `.ctors` constructor already runs automatically as
// described above, and `_rt0_<arch>_lib`'s own body is *not* idempotent
// (it unconditionally spins up a fresh runtime-init thread every time
// it's called, with no once-guard of its own — confirmed by reading
// $GOROOT/src/runtime/asm_amd64.s) — calling it a second time here would
// race a second Go runtime bootstrap against the one already running.
// llmman doesn't currently ship a windows-gnu target (see ci.yml's
// build matrix), but scoping this to the target that actually needs it
// costs nothing and avoids relying on that.
#[cfg(all(target_os = "windows", target_env = "msvc", target_arch = "x86_64"))]
extern "C" {
    fn _rt0_amd64_windows_lib(argc: i32, argv: *mut *mut u8);
}

#[cfg(all(target_os = "windows", target_env = "msvc", target_arch = "aarch64"))]
extern "C" {
    fn _rt0_arm64_windows_lib();
}

/// Must be called once, before the first call into any other function in
/// this module — see the doc comment above. A no-op on every target
/// except windows-msvc, where the platform's own C runtime already runs
/// this automatically as a real constructor. Idempotent and cheap to
/// call more than once (guarded by a `Once`), so callers can simply
/// invoke it defensively rather than needing to prove they're first.
pub fn ensure_runtime_init() {
    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| unsafe {
            #[cfg(target_arch = "x86_64")]
            _rt0_amd64_windows_lib(0, std::ptr::null_mut());
            #[cfg(target_arch = "aarch64")]
            _rt0_arm64_windows_lib();
        });
    }
}

// ---------------------------------------------------------------------------
// Response envelope (mirrors the Go `response` struct)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GoResponse {
    ok: bool,
    #[serde(default)]
    data: String,
    #[serde(default)]
    error: String,
}

/// Consume a raw C string returned by the Go shim and decode the JSON envelope.
/// Returns `Ok(data)` on success, `Err(error)` on failure.
fn consume(raw: *mut c_char) -> anyhow::Result<String> {
    assert!(!raw.is_null(), "Go shim returned a null pointer");
    let json_str = unsafe { CStr::from_ptr(raw).to_string_lossy().into_owned() };
    unsafe { llmman_free(raw) };
    let resp: GoResponse =
        serde_json::from_str(&json_str).context("failed to decode Go shim response")?;
    if resp.ok {
        Ok(resp.data)
    } else {
        Err(anyhow!("{}", resp.error))
    }
}

fn cstr(s: &str) -> anyhow::Result<CString> {
    CString::new(s).context("string contains interior NUL byte")
}

// ---------------------------------------------------------------------------
// Registry configuration
// ---------------------------------------------------------------------------

/// Tells the Go shim the mirrors `llmman.conf` configures, once per
/// process. Nothing but these arguments crosses the boundary, so it has
/// to be told; a refused mirror is reported and dropped, since a pull
/// straight from the registry is the right fallback.
fn ensure_registries_configured() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mirrors = crate::config::registry_mirrors();
        if mirrors.is_empty() {
            return;
        }
        let result = serde_json::to_string(&mirrors)
            .context("encode registry mirrors")
            .and_then(|json| {
                let j = cstr(&json)?;
                consume(unsafe { llmman_set_registry_mirrors(j.as_ptr()) })
            });
        if let Err(e) = result {
            eprintln!("[llmman] warning: ignoring registry mirrors: {e}");
        }
    });
}

/// [`consume`] for a call that reaches a registry: configures mirrors
/// first, so no wrapper can forget to.
fn registry_call(call: impl FnOnce() -> *mut c_char) -> anyhow::Result<String> {
    ensure_registries_configured();
    consume(call())
}

// ---------------------------------------------------------------------------
// Safe public API
// ---------------------------------------------------------------------------

/// Store registry credentials.
pub fn login(server: &str, username: &str, password: &str) -> anyhow::Result<()> {
    let s = cstr(server)?;
    let u = cstr(username)?;
    let p = cstr(password)?;
    consume(unsafe { llmman_login(s.as_ptr(), u.as_ptr(), p.as_ptr()) }).map(|_| ())
}

/// Remove stored registry credentials.
pub fn logout(server: &str) -> anyhow::Result<()> {
    let s = cstr(server)?;
    consume(unsafe { llmman_logout(s.as_ptr()) }).map(|_| ())
}

/// Push the image tagged `reference` from `layout_dir` (OCI layout) to a registry.
pub fn push(layout_dir: &str, reference: &str) -> anyhow::Result<()> {
    let l = cstr(layout_dir)?;
    let r = cstr(reference)?;
    registry_call(|| unsafe { llmman_push(l.as_ptr(), r.as_ptr()) }).map(|_| ())
}

/// Pull an image from a registry into `layout_dir` (OCI layout).
pub fn pull(reference: &str, layout_dir: &str) -> anyhow::Result<()> {
    let r = cstr(reference)?;
    let l = cstr(layout_dir)?;
    registry_call(|| unsafe { llmman_pull(r.as_ptr(), l.as_ptr()) }).map(|_| ())
}

/// Fetch and return the raw manifest JSON for a remote registry reference.
pub fn inspect_remote(reference: &str) -> anyhow::Result<String> {
    let r = cstr(reference)?;
    registry_call(|| unsafe { llmman_inspect(r.as_ptr()) })
}

/// The result of one completed transfer, whichever of the three
/// implementations (`ffi::transfer`, `crate::hf::transfer`,
/// `crate::sources::transfer`) produced it.
#[derive(Debug, Clone, Deserialize)]
pub struct TransferOutcome {
    /// Whether anything was actually pushed. Everything is
    /// content-addressed, so re-running a transfer for an unchanged
    /// source pushes nothing — this lets `cmd::transfer` say so instead
    /// of always printing "Transferred ...".
    pub changed: bool,
    /// The manifest digest now at the destination. `--sign-key` signs
    /// exactly this rather than re-resolving the tag afterwards, so
    /// nothing can slip in between the push and the signature.
    #[serde(default)]
    pub digest: Option<String>,
}

impl TransferOutcome {
    /// For the transfer paths that push a manifest they built or staged
    /// themselves, and so already know its digest.
    pub fn new(changed: bool, digest: impl Into<String>) -> Self {
        Self {
            changed,
            digest: Some(digest.into()),
        }
    }
}

/// Transfer an image directly from `source` to `destination` without
/// going through the local store. See go-shim/transfer_docker.go /
/// transfer_podman.go for how each backend implements this (streamed
/// blob-for-blob where possible, falling back to a throwaway local
/// staging directory only for source kinds that transport has no way to
/// stream).
pub fn transfer(source: &str, destination: &str) -> anyhow::Result<TransferOutcome> {
    let s = cstr(source)?;
    let d = cstr(destination)?;
    let data = registry_call(|| unsafe { llmman_transfer(s.as_ptr(), d.as_ptr()) })?;
    serde_json::from_str(&data).context("decode transfer outcome")
}

/// One signature that verified, and the trusted key that accepted it.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SignatureMatch {
    pub key_path: String,
    /// The `docker-reference` the signer claimed to be signing.
    pub identity: String,
}

/// The outcome of checking a manifest's signatures — see
/// go-shim/sigstore.go's `verifyReport`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerifyReport {
    pub reference: String,
    pub digest: String,
    pub verified: bool,
    /// Signature layers present on the registry, whether or not any
    /// verified — "nobody signed this" and "someone signed this, but not
    /// with a key you trust" are very different things to report.
    pub signatures_found: u32,
    pub matches: Vec<SignatureMatch>,
    /// Why `verified` is false. Empty when it's true.
    pub reason: String,
}

/// Check the signatures published for `reference` against the PEM public
/// keys at `keys`. `digest` may be `""` to resolve `reference` first.
///
/// "Not verified" comes back as `Ok` with `verified` false; `Err` means
/// no answer could be reached. `crate::verify` decides what each means.
pub fn verify(reference: &str, digest: &str, keys: &[String]) -> anyhow::Result<VerifyReport> {
    let r = cstr(reference)?;
    let d = cstr(digest)?;
    let k = cstr(&serde_json::to_string(keys).context("encode trusted key list")?)?;
    let data = registry_call(|| unsafe { llmman_verify(r.as_ptr(), d.as_ptr(), k.as_ptr()) })?;
    serde_json::from_str(&data).context("decode verification report")
}

/// Sign `reference` at `digest` with the PEM private key at `key_path`
/// and publish a cosign-format signature beside it in the same
/// repository. Returns the digest that was signed.
///
/// `digest` may be `""` to resolve `reference` remotely first.
/// `password` is ignored for an unencrypted key.
pub fn sign(
    reference: &str,
    digest: &str,
    key_path: &str,
    password: &str,
) -> anyhow::Result<String> {
    let r = cstr(reference)?;
    let d = cstr(digest)?;
    let k = cstr(key_path)?;
    let p = cstr(password)?;
    registry_call(|| unsafe { llmman_sign(r.as_ptr(), d.as_ptr(), k.as_ptr(), p.as_ptr()) })
}

/// The manifest digest the registry currently serves for `reference`,
/// without fetching any layer. `crate::verify::PullGuard` uses it to
/// check a signature before starting a multi-gigabyte download.
pub fn resolved_digest_of(reference: &str) -> anyhow::Result<String> {
    let r = cstr(reference)?;
    registry_call(|| unsafe { llmman_resolve_digest(r.as_ptr()) })
}

/// A registry destination resolved once via [`push_session_open`] and
/// shared by every [`push_stream_open`] call within one transfer —
/// resolving a destination negotiates auth (and, for a non-HTTPS-only
/// host, containerd's HTTPS-then-HTTP-fallback probe), neither of which
/// is free to redo per blob. Releases itself (best-effort) on `Drop`, so
/// an early `?` return partway through a transfer can't leak it.
#[cfg(feature = "docker")]
pub struct PushSession(i64);

#[cfg(feature = "docker")]
pub fn push_session_open(destination: &str) -> anyhow::Result<PushSession> {
    #[derive(Deserialize)]
    struct Result_ {
        session: i64,
    }
    let d = cstr(destination)?;
    let data = registry_call(|| unsafe { llmman_push_session_open(d.as_ptr()) })?;
    let result: Result_ =
        serde_json::from_str(&data).context("decode push_session_open response")?;
    Ok(PushSession(result.session))
}

#[cfg(feature = "docker")]
impl Drop for PushSession {
    fn drop(&mut self) {
        if let Err(e) = consume(unsafe { llmman_push_session_close(self.0) }) {
            eprintln!("[llmman] close push session: {e:#}");
        }
    }
}

/// The write end of a pipe Go created for one streamed blob/manifest
/// push (see go-shim/push_stream.go), plus the handle to collect its
/// outcome with [`push_stream_wait`]. `fd` is a raw POSIX fd on Unix or
/// a raw Windows `HANDLE` widened to `u64` — Go, not this side, creates
/// the pipe (see `crate::hf::transfer`'s doc comment for why). Once
/// wrapped in a `File` (`from_raw_fd`/`from_raw_handle`), `fd` itself is
/// a stale value — don't read it again.
#[cfg(feature = "docker")]
#[must_use = "must be passed to push_stream_wait, or the Go-side handle and its goroutine leak"]
pub struct PushStream {
    pub fd: u64,
    handle: i64,
}

#[cfg(feature = "docker")]
#[derive(Deserialize)]
struct PushStreamOpenResult {
    fd: u64,
    handle: i64,
}

/// Starts pushing one blob (or the final manifest — the registry tells
/// them apart by `media_type`, not anything passed here) within
/// `session` (see [`push_session_open`]). `digest` is `"sha256:<hex>"`.
/// `annotations_json` is an optional JSON object (`""` for none).
/// Returns immediately with the write end of a pipe to stream the
/// content into; see [`push_stream_wait`].
#[cfg(feature = "docker")]
pub fn push_stream_open(
    session: &PushSession,
    media_type: &str,
    digest: &str,
    size: i64,
    annotations_json: &str,
) -> anyhow::Result<PushStream> {
    let m = cstr(media_type)?;
    let g = cstr(digest)?;
    let a = cstr(annotations_json)?;
    let data = consume(unsafe {
        llmman_push_stream_open(session.0, m.as_ptr(), g.as_ptr(), size, a.as_ptr())
    })?;
    let result: PushStreamOpenResult =
        serde_json::from_str(&data).context("decode push_stream_open response")?;
    Ok(PushStream {
        fd: result.fd,
        handle: result.handle,
    })
}

/// Blocks until the streamed push started by [`push_stream_open`]
/// finishes — the caller must have already finished writing and closed
/// its own end of `stream.fd` before calling this, or it hangs waiting
/// for an EOF that will never come. Returns whether the destination
/// didn't already have this exact blob (by digest).
#[cfg(feature = "docker")]
pub fn push_stream_wait(stream: PushStream) -> anyhow::Result<bool> {
    #[derive(Deserialize)]
    struct WaitResult {
        changed: bool,
    }
    let data = consume(unsafe { llmman_push_stream_wait(stream.handle) })?;
    let result: WaitResult =
        serde_json::from_str(&data).context("decode push_stream_wait response")?;
    Ok(result.changed)
}

/// A byte-level snapshot of one particular pull/push (identified by
/// `key`, the exact model reference it was started with), as tracked by
/// the Go shim's `progressState` (see go-shim/progress_state.go).
/// `total`/`completed` are 0 until the shim learns a blob's size and
/// starts transferring it, or if `key` has no tracked entry (not started
/// yet, or already finished and cleaned up).
#[derive(Deserialize)]
pub struct ProgressSnapshot {
    pub status: String,
    pub total: i64,
    pub completed: i64,
}

/// Polls the Go shim's pull/push progress snapshot for `key` (the same
/// model reference passed to `pull`/`push` above). Called by `cmd::serve`
/// every ~200ms while a `/api/pull` or `/api/push` task for that same key
/// is in flight, to relay real byte counts over its NDJSON stream instead
/// of just a coarse "pulling <model>" heartbeat — see llmman_progress's
/// own doc comment for why the daemon can't just let the shim's own mpb
/// bars (go-shim/shared_oci.go) reach an interactive terminal directly.
/// Keying by reference (rather than one process-wide snapshot, as this
/// used to be) is what lets two different models' pulls/pushes run
/// concurrently in the same daemon without their progress numbers
/// interleaving — see serve.rs's per-model lock registry.
pub fn progress(key: &str) -> anyhow::Result<ProgressSnapshot> {
    let k = cstr(key)?;
    let data = consume(unsafe { llmman_progress(k.as_ptr()) })?;
    serde_json::from_str(&data).context("failed to decode progress snapshot")
}

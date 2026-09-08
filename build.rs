use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use flate2::write::GzEncoder;
use flate2::Compression;
use sha2::{Digest, Sha256};

#[path = "src/webui/content_type.rs"]
mod content_type;

/// Strip Go's SEH unwind sections from `go.o`, so MSVC `link.exe` accepts
/// the archive instead of failing with "LNK1223: ... invalid .pdata
/// contributions".
///
/// `go.o` comes from the Go linker, not the clang wrapper below, so no
/// compiler flag can affect it. `lld-link` tolerates it -- which is what CI
/// used until now -- but that is not available to `cargo install llmman`:
/// Cargo reads no config from a downloaded package and a build script
/// cannot set `-C linker`. The archive is the only lever left.
///
/// Cost: Windows can no longer unwind through Go frames, losing Go stack
/// fidelity in debuggers and crash dumps. Nothing llmman needs -- Go's
/// panic/recover uses its own stack maps, faults go to the runtime's
/// vectored handler, and no C++ exception crosses Go. x86_64 only; aarch64
/// links with this data intact, so it keeps it.
fn strip_go_unwind_sections(objs: &[PathBuf]) {
    let mut found = false;
    for obj in objs {
        if obj.file_name().and_then(|s| s.to_str()) != Some("go.o") {
            continue;
        }
        found = true;
        // .xdata too: .pdata entries point into it, so it would otherwise
        // just be unreferenced bytes.
        let ok = Command::new("llvm-objcopy")
            .args(["--remove-section=.pdata", "--remove-section=.xdata"])
            .arg(obj)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            println!("cargo:warning=Stripped SEH unwind sections from go.o");
        } else {
            println!(
                "cargo:warning=llvm-objcopy failed on go.o; the link will fail with \
                 LNK1223. Install LLVM, or use RUSTFLAGS=\"-C linker=lld-link\"."
            );
        }
    }
    // Guards against a future Go release naming this object something else,
    // which would otherwise silently stop stripping and resurface as a bare
    // LNK1223.
    if !found {
        println!(
            "cargo:warning=no go.o in the Go archive, so nothing was stripped; \
             expect LNK1223 unless linking with lld-link."
        );
    }
}

/// Extract every object from a (possibly GNU ar) static archive and repack
/// it as an MSVC-format LIB using lib.exe. Go uses GNU 'ar' when it can't
/// identify the C compiler as cl.exe, and MSVC link.exe rejects that format
/// with LNK4003.
///
/// `strip_unwind` also drops Go's SEH sections -- see
/// strip_go_unwind_sections.
fn repack_as_msvc_lib(lib_path: &Path, out_dir: &Path, strip_unwind: bool) {
    let extract_dir = out_dir.join("ar_extract");
    let _ = fs::remove_dir_all(&extract_dir);
    if fs::create_dir_all(&extract_dir).is_err() {
        return;
    }

    // llvm-ar can read both GNU ar and MSVC LIB archives
    let ok = Command::new("llvm-ar")
        .args(["x", lib_path.to_str().unwrap_or("")])
        .current_dir(&extract_dir)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("cargo:warning=llvm-ar extraction failed; keeping original archive");
        return;
    }

    let objs: Vec<PathBuf> = match fs::read_dir(&extract_dir) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
        Err(_) => return,
    };
    if objs.is_empty() {
        return;
    }

    if strip_unwind {
        strip_go_unwind_sections(&objs);
    }

    // lib.exe is the MSVC archiver; it infers machine type from objects
    let tmp = out_dir.join("_shim_repack.lib");
    let mut cmd = Command::new("lib.exe");
    cmd.arg("/nologo").arg(format!("/out:{}", tmp.display()));
    for obj in &objs {
        if let Some(s) = obj.to_str() {
            cmd.arg(s);
        }
    }

    if cmd.status().map(|s| s.success()).unwrap_or(false) {
        let _ = fs::rename(&tmp, lib_path);
        println!("cargo:warning=Repacked Go archive as MSVC LIB");
    } else {
        eprintln!("cargo:warning=lib.exe repack failed; keeping original archive");
    }
    let _ = fs::remove_dir_all(&extract_dir);
}

/// True if `tool` is on `PATH`. `--version`, not a bare spawn: a stale PATH
/// entry can leave a name resolvable but not executable.
fn on_path(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Name missing Windows prerequisites up front, rather than letting them
/// surface minutes later as a cgo compiler error or a bare LNK1223. Both
/// ship with LLVM, already required here, so this adds no dependency:
/// `clang` backs the CC wrapper below, `llvm-objcopy` does the strip.
fn warn_missing_msvc_tools(target_arch: &str) {
    if !on_path("clang") {
        println!(
            "cargo:warning=clang was not found on PATH. Building llmman for \
             *-pc-windows-msvc needs it as cgo's C compiler; install LLVM \
             (https://releases.llvm.org) or `winget install LLVM.LLVM`."
        );
    }

    if target_arch == "x86_64" && !on_path("llvm-objcopy") {
        println!(
            "cargo:warning=llvm-objcopy was not found on PATH. It is needed to strip \
             Go's SEH unwind sections for x86_64-pc-windows-msvc, without which \
             linking fails with LNK1223; it ships with LLVM."
        );
    }
}

/// Emits `LLMMAN_VERSION`: the Cargo package version plus, when built from
/// a git checkout, the commit it was built from (e.g. "0.1.0 (a1b2c3d)").
/// Every nightly/CI build shares the same Cargo.toml version, so without
/// the commit suffix two different builds are indistinguishable to
/// `llmman --version` and the daemon's /api/version.
///
/// `LLMMAN_GIT_DESCRIBE`, if set, replaces the `git describe` output. CI
/// sets it to the bare commit hash after `packaging/version.sh --apply`
/// edits Cargo.toml/Cargo.lock, which would otherwise make every release
/// binary report `-dirty`.
fn emit_version() {
    let pkg = env::var("CARGO_PKG_VERSION").unwrap_or_default();
    println!("cargo:rerun-if-env-changed=LLMMAN_GIT_DESCRIBE");
    let describe = env::var("LLMMAN_GIT_DESCRIBE")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            Command::new("git")
                .args(["describe", "--always", "--dirty"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        });
    let version = match describe {
        Some(desc) => format!("{pkg} ({desc})"),
        None => pkg,
    };
    println!("cargo:rustc-env=LLMMAN_VERSION={version}");
    // `HEAD` only changes on a checkout/branch switch; an ordinary `git
    // commit` on the current branch instead updates `logs/HEAD` (the
    // reflog) — without watching that too, `--dirty`'s output (and so
    // LLMMAN_VERSION) can go stale across a commit, defeating
    // stale_daemon's version comparison in daemon.rs. Resolved via
    // `git rev-parse --git-path` rather than a hardcoded `.git/...` path
    // so this also reruns correctly from a linked worktree, where the
    // real HEAD/logs live under `.git/worktrees/<name>/` instead.
    for path in ["HEAD", "logs/HEAD"] {
        if let Some(resolved) = git_path(path) {
            println!("cargo:rerun-if-changed={}", resolved.display());
        }
    }
}

/// Resolves a path relative to the repo's git dir (e.g. `"HEAD"` or
/// `"logs/HEAD"`) via `git rev-parse --git-path`, so it's correct both for
/// a plain `.git` directory and a linked worktree's own `.git` file.
fn git_path(path: &str) -> Option<PathBuf> {
    Command::new("git")
        .args(["rev-parse", "--git-path", path])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| PathBuf::from(s.trim()))
}

/// docs.rs sets `DOCS_RS`. Its sandbox has no Go and no network, so the Go
/// shim cannot be built there; skipping it is safe because rustdoc never
/// links, so the `rustc-link-lib=static=` directive is never acted on. A
/// real `cargo build` still hard-fails without Go.
fn is_docs_rs() -> bool {
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    env::var_os("DOCS_RS").is_some()
}

fn main() {
    emit_version();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shim_dir = manifest_dir.join("go-shim");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    // src/webui.rs include_bytes!s the gzipped webui, so that part still
    // runs; only the Go half is skipped.
    if is_docs_rs() {
        println!("cargo:warning=DOCS_RS set: skipping the Go shim build (rustdoc does not link)");
        gzip_webui(&manifest_dir, &out_dir);
        return;
    }

    // Determine backend build tags from Cargo features.
    // For the podman backend we add two extra tags to avoid pulling in C library
    // dependencies (libgpgme, libbtrfs, libdevmapper) that are not present on CI
    // runners and are not needed for the docker:// and oci: transports we use:
    //   containers_image_openpgp  — use Go's native OpenPGP instead of libgpgme
    //   exclude_graphdriver_btrfs — omit the btrfs storage driver (needs libbtrfs-dev)
    //   exclude_graphdriver_devicemapper — omit the device-mapper driver
    let go_tags = if env::var("CARGO_FEATURE_PODMAN").is_ok() {
        "podman,containers_image_openpgp,exclude_graphdriver_btrfs,exclude_graphdriver_devicemapper"
    } else {
        "docker"
    };

    // Always build natively — each platform's CI runner builds its own binary.
    // CGO requires the host C toolchain, which is always present on native runners.
    //
    // Only *-pc-windows-msvc uses the MSVC "NAME.lib" naming; *-pc-windows-gnu
    // is a GNU-ABI target like Linux/macOS and needs "libNAME.a". The final
    // link step (mingw's ld) tolerates either name on a GNU target, but the
    // `llmman` lib crate's own compilation checks a `#[link(kind =
    // "static")]` dependency's presence itself first, by each target's
    // canonical name, with no such leniency.
    let lib_name = if target_os == "windows" && target_env == "msvc" {
        "llmman_shim.lib"
    } else {
        "libllmman_shim.a"
    };
    let lib_path = out_dir.join(lib_name);

    // Ensure Go module dependencies are present
    let _ = Command::new("go")
        .current_dir(&shim_dir)
        .args(["mod", "download"])
        .status();

    let mut cmd = Command::new("go");
    cmd.current_dir(&shim_dir)
        .env("CGO_ENABLED", "1")
        .arg("build")
        .arg(format!("-tags={}", go_tags))
        .arg("-buildmode=c-archive")
        .arg("-o")
        .arg(&lib_path)
        .arg(".");

    // On *-pc-windows-msvc targets the Rust linker (lld-link, set via
    // RUSTFLAGS in CI) requires MSVC-ABI COFF objects from Go's CGO.
    //
    // clang in GCC-driver mode is the only C compiler that:
    //   • accepts all GCC-style flags Go passes (-Werror, -dM, -fno-stack-protector)
    //   • produces MSVC-compatible COFF when given the right --target
    //
    // We cannot rely on CGO_CFLAGS to pass --target because Go's CGO security
    // filter may strip unrecognised flags before they reach clang.  Instead we
    // write a tiny .cmd wrapper that hard-codes --target as part of the CC
    // command itself; this is unconditional and cannot be filtered out.
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
        warn_missing_msvc_tools(&arch);
        let msvc_triple = match arch.as_str() {
            "x86_64" => "x86_64-pc-windows-msvc",
            "aarch64" => "aarch64-pc-windows-msvc",
            other => panic!("unsupported Windows MSVC arch: {}", other),
        };
        // The wrapper calls clang with a fixed --target then forwards all other
        // args (%*).  Go treats this .cmd as the C compiler.
        let wrapper = out_dir.join("cgo_cc.cmd");
        fs::write(
            &wrapper,
            format!("@echo off\r\nclang --target={} %*\r\n", msvc_triple),
        )
        .expect("write CGO CC wrapper");
        cmd.env("CC", &wrapper);
    }

    // Align the Go shim's minimum macOS version with Rust's aarch64-apple-darwin
    // deployment target (11.0).  Without this Go defaults to the SDK version
    // (15.x on macos-15 runners), producing objects that emit "built for newer
    // macOS" warnings and may reference symbols gated behind the newer version.
    if target_os == "macos" {
        cmd.env("MACOSX_DEPLOYMENT_TARGET", "11.0");
    }

    let status = cmd
        .status()
        .expect("Failed to invoke `go build` — is Go (1.22+) installed and on PATH?");

    if !status.success() {
        panic!("Go shim build failed for tags={}", go_tags);
    }

    // Go archives the CGO objects with GNU 'ar' on every *-pc-windows-msvc
    // target (it identifies MSVC by looking for "cl.exe" in the compiler
    // name, which our clang wrapper is not), and MSVC link.exe rejects that
    // with LNK4003. Extract with llvm-ar, which reads both formats, and
    // repack with lib.exe; it infers machine type from the objects.
    //
    // Previously aarch64-only, because x86_64 got lld-link via RUSTFLAGS in
    // CI and lld-link accepts GNU ar archives. Now that x86_64 links with
    // the default toolchain it needs the MSVC-format archive too, plus the
    // strip -- see strip_go_unwind_sections.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        let strip_unwind = env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64");
        repack_as_msvc_lib(&lib_path, &out_dir, strip_unwind);
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=llmman_shim");

    // Platform-specific link dependencies required by Go runtime and shim libraries
    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=dl");
        }
        "macos" => {
            println!("cargo:rustc-link-lib=framework=CoreFoundation");
            println!("cargo:rustc-link-lib=framework=Security");
            println!("cargo:rustc-link-lib=framework=SystemConfiguration");
            // The podman backend (and Go's CGO net resolver in general) references
            // res_9_ninit / res_9_nclose / res_9_nsearch from libresolv.
            println!("cargo:rustc-link-lib=resolv");
        }
        "windows" => {
            println!("cargo:rustc-link-lib=bcrypt");
            println!("cargo:rustc-link-lib=ws2_32");
            println!("cargo:rustc-link-lib=userenv");
            // With CC=cl the CGO objects are compiled by MSVC which links the CRT
            // automatically; legacy_stdio_definitions is not needed and causes
            // LNK4078 / LNK1223 when mixed with MSVC-format objects.
        }
        _ => {}
    }

    println!("cargo:rerun-if-changed=go-shim/");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PODMAN");

    gzip_webui(&manifest_dir, &out_dir);
}

/// Gzip every file under `webui/` into `$OUT_DIR/webui_gz/` and write
/// `$OUT_DIR/webui_assets.rs`: a sorted `(path, content-type, etag, bytes)`
/// table src/webui/mod.rs `include!`s. Paths are relative to `webui/` with
/// `/` separators. Separate so the docs.rs early return can still run it.
fn gzip_webui(manifest_dir: &Path, out_dir: &Path) {
    let webui_src = manifest_dir.join("webui");
    let webui_out = out_dir.join("webui_gz");
    fs::create_dir_all(&webui_out).expect("create webui_gz dir");

    let mut files = Vec::new();
    collect_webui_files(&webui_src, &webui_src, &mut files);
    assert!(
        files.iter().any(|f| f == "index.html"),
        "webui/index.html is missing"
    );
    // Binary assets are not kept in the repository; see FETCHED_ASSETS.
    let fetched_dir = out_dir.join("webui_fetched");
    for asset in FETCHED_ASSETS {
        fetch_asset(asset, &fetched_dir);
        files.push(asset.path.to_string());
    }
    files.sort();
    files.dedup();

    let mut table = String::from(
        "/// Generated by build.rs: every file under `webui/`, gzipped:\n\
         /// `(path, content-type, etag, bytes)`.\n\
         pub static ASSETS: &[(&str, &str, &str, &[u8])] = &[\n",
    );
    for rel in &files {
        let content_type = content_type::for_path(rel)
            .unwrap_or_else(|| panic!("webui/{rel}: no content type for its extension; add one to src/webui/content_type.rs"));
        let src = if FETCHED_ASSETS.iter().any(|a| a.path == rel) {
            fetched_dir.join(rel)
        } else {
            webui_src.join(rel)
        };
        let dst = webui_out.join(format!("{}.gz", rel.replace('/', "__")));
        let data = fs::read(&src).unwrap_or_else(|e| panic!("read webui/{rel}: {e}"));
        let mut enc = GzEncoder::new(Vec::new(), Compression::best());
        enc.write_all(&data).expect("gzip write");
        let compressed = enc.finish().expect("gzip finish");
        fs::write(&dst, &compressed).unwrap_or_else(|e| panic!("write {}: {e}", dst.display()));
        table.push_str(&format!(
            "    ({rel:?}, {content_type:?}, {:?}, include_bytes!({:?})),\n",
            format!("\"{:016x}\"", fnv1a64(&compressed)),
            dst.to_str().expect("OUT_DIR is UTF-8"),
        ));
    }
    table.push_str("];\n");
    fs::write(out_dir.join("webui_assets.rs"), table).expect("write webui_assets.rs");

    println!("cargo:rerun-if-changed=webui/");
}

/// A web UI file kept out of the repository (no binaries in git) and
/// pulled from the `docs-assets` GitHub release at build time instead,
/// pinned by SHA-256.
struct FetchedAsset {
    /// Path under `webui/` it is served as.
    path: &'static str,
    url: &'static str,
    sha256: &'static str,
}

const FETCHED_ASSETS: &[FetchedAsset] = &[FetchedAsset {
    path: "manatee.png",
    url: "https://github.com/llmmanorg/llmman/releases/download/docs-assets/llmman-mark.png",
    sha256: "c6036711a349dc8f8b2b4405be90b342b477dbcf4a7d95eb7d05b42702a241d5",
}];

/// A 1x1 transparent PNG (Pillow output, verified to decode), served when
/// the download is impossible (docs.rs, offline builds).
const BLANK_PNG: &[u8] = &[
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4,
    0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0x60, 0x60, 0x60, 0x60,
    0x00, 0x00, 0x00, 0x05, 0x00, 0x01, 0x7a, 0xa8, 0x57, 0x50, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
    0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
];

/// Downloads `asset` into `dir` unless a copy with the right hash is
/// already there. `curl` rather than an HTTP crate: it is on every target
/// this builds on, and a build dependency would cost more compile time
/// than the file is worth. A failed download warns and writes the blank
/// PNG; a wrong hash is an error, since that is a changed upstream file.
fn fetch_asset(asset: &FetchedAsset, dir: &Path) {
    fs::create_dir_all(dir).expect("create webui_fetched dir");
    let dst = dir.join(asset.path);
    let hash_ok = |bytes: &[u8]| hex::encode(Sha256::digest(bytes)) == asset.sha256;
    if fs::read(&dst).is_ok_and(|b| hash_ok(&b)) {
        return;
    }
    let tmp = dst.with_extension("part");
    let fetched = Command::new("curl")
        .args(["-fsSL", "--retry", "3", "--max-time", "60", "-o"])
        .arg(&tmp)
        .arg(asset.url)
        .status()
        .is_ok_and(|s| s.success());
    let bytes = fetched.then(|| fs::read(&tmp).ok()).flatten();
    let _ = fs::remove_file(&tmp);
    match bytes {
        Some(bytes) if hash_ok(&bytes) => fs::write(&dst, bytes).expect("write fetched asset"),
        Some(_) => panic!(
            "{} does not match the sha256 pinned in build.rs ({}); update the pin if it changed on purpose",
            asset.url, asset.sha256
        ),
        None => {
            println!(
                "cargo:warning=could not download {} (offline?); the web UI will show no mark",
                asset.url
            );
            fs::write(&dst, BLANK_PNG).expect("write placeholder asset");
        }
    }
}

/// Regular files under `dir` as `/`-separated paths relative to `root`;
/// dotfiles (editor swap files, `.DS_Store`) are skipped.
fn collect_webui_files(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.expect("read webui dir entry");
        let path = entry.path();
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        if path.is_dir() {
            collect_webui_files(root, &path, out);
        } else if path.is_file() {
            let rel = path
                .strip_prefix(root)
                .expect("under webui/")
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            out.push(rel);
        }
    }
}

/// FNV-1a over the gzipped bytes, for the `ETag` header: no dependency,
/// and only "same file, same tag" matters.
fn fnv1a64(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

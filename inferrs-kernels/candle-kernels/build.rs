use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo::rerun-if-changed=src/iq_dequant.cu");
    println!("cargo::rerun-if-changed=src/ggml_iq_tables_device.inc");
    println!("cargo::rerun-if-changed=src/linear_attn.cu");
    println!("cargo::rerun-if-changed=src/linear_attn_scan.cu");

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");
    let bindings = builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();

    // Remove unwanted MOE PTX constants from ptx.rs
    remove_lines(&ptx_path, &["MOE_GGUF", "MOE_WMMA", "MOE_WMMA_GGUF"]);

    let mut moe_builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Build for FFI binding (must use custom bindgen_cuda, which supports simutanously build PTX and lib)
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    let moe_builder = moe_builder.kernel_paths(vec![
        "src/moe/moe_gguf.cu",
        "src/moe/moe_wmma.cu",
        "src/moe/moe_wmma_gguf.cu",
    ]);
    moe_builder.build_lib(out_dir.join("libmoe.a"));
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");

    // Statically link the CUDA runtime instead of hard-linking libcudart.so.
    // libcudart_static.a resolves the CUDA driver API (libcuda.so) via
    // dlopen/dlsym internally, so the final binary ends up with NO DT_NEEDED
    // entries for libcudart / libcuda / libcublas / libcurand — matching the
    // behaviour already achieved for cudarc via `fallback-dynamic-loading`.
    // This is what makes "brew install inferrs" viable as a single binary
    // that dlopens whatever CUDA libs are present at runtime (12.x, 13.x, …)
    // and falls back cleanly on systems without CUDA at all.
    //
    // `rustc-link-lib=static=` propagates from lib build scripts to downstream
    // binaries (unlike `rustc-link-arg`, which only applies to the current
    // compilation unit — a no-op for a lib crate).  We also need to be robust
    // to the various CUDA toolkit layouts:
    //
    //   - Debian/x86_64:  /usr/local/cuda/lib64/libcudart_static.a
    //   - Debian/sbsa:    /usr/local/cuda/targets/sbsa-linux/lib/libcudart_static.a
    //   - Conda etc.:     $CUDA_PATH/lib/libcudart_static.a
    //   - Windows MSVC:   $CUDA_PATH/lib/x64/cudart_static.lib
    //
    // We probe every plausible directory and add those that exist as native
    // search paths, so rustc can resolve `-l static=cudart_static` regardless
    // of which layout the host toolkit ships.
    //
    // When `CUDA_PATH` is not set, we try a platform-specific list of default
    // install roots: the conventional `/usr/local/cuda` on Unix, and the
    // `NVIDIA GPU Computing Toolkit\CUDA\vXX.Y` directories on Windows (newest
    // first).  Setting `CUDA_PATH` explicitly is still recommended.
    let cuda_path_env = std::env::var("CUDA_PATH").ok();
    let cuda_roots: Vec<String> = match cuda_path_env.as_deref() {
        Some(p) => vec![p.to_string()],
        None if is_target_msvc => {
            let base = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA";
            // Newest first — the linker will use whichever exists.
            [
                "v13.2", "v13.1", "v13.0", "v12.9", "v12.8", "v12.6", "v12.0",
            ]
            .iter()
            .map(|v| format!("{base}/{v}"))
            .collect()
        }
        None => vec!["/usr/local/cuda".to_string()],
    };

    let lib_filename = if is_target_msvc {
        "cudart_static.lib"
    } else {
        "libcudart_static.a"
    };

    let layout_subdirs: &[&str] = if is_target_msvc {
        &["lib/x64"]
    } else {
        &[
            "lib64",
            "lib",
            "targets/x86_64-linux/lib",
            "targets/sbsa-linux/lib",
            "targets/aarch64-linux/lib",
        ]
    };

    let mut search_dirs: Vec<String> = Vec::new();
    for root in &cuda_roots {
        for sub in layout_subdirs {
            search_dirs.push(format!("{root}/{sub}"));
        }
    }

    let mut resolved = false;
    for dir in &search_dirs {
        let candidate = format!("{dir}/{lib_filename}");
        if std::path::Path::new(&candidate).exists() {
            println!("cargo:warning=candle-kernels: found {candidate}");
            println!("cargo:rustc-link-search=native={dir}");
            resolved = true;
        }
    }
    if !resolved {
        let hint = if cuda_path_env.is_none() {
            " — set CUDA_PATH to your CUDA toolkit root (e.g. /usr/local/cuda \
             or C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6)"
        } else {
            ""
        };
        println!(
            "cargo:warning=candle-kernels: could not locate {lib_filename} under \
             any of {search_dirs:?}{hint}; the final link step will fail with \
             undefined cudart symbols."
        );
        // Still add the probed paths as a last-ditch effort — the linker may
        // find the file even if our probe missed it (e.g. via a symlink chain
        // we didn't walk).
        for dir in &search_dirs {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }

    // Modifier breakdown:
    //
    //   -bundle            : do NOT extract libcudart_static.a's objects into
    //                        the candle_kernels rlib.  Bundling was silently
    //                        failing on ubuntu-24.04-arm CI (the sbsa toolkit
    //                        layout confused rustc's internal native-lib
    //                        lookup, so the directive was dropped entirely
    //                        and the downstream linker never saw
    //                        -lcudart_static).  `-bundle` forces rustc to
    //                        pass the `-l` through verbatim and let the
    //                        final linker resolve it via the `-L` paths we
    //                        emitted above.
    //
    //   +whole-archive     : wrap the linker inclusion in
    //                        `-Wl,--whole-archive ... -Wl,--no-whole-archive`
    //                        so every object in libcudart_static.a is pulled
    //                        in regardless of ordering with libmoe.a.  Without
    //                        this, a link-order quirk where cudart is seen
    //                        before libmoe.a's references can cause the
    //                        linker to skip cudart objects and then complain
    //                        about undefined references when it gets to
    //                        libmoe.a.
    println!("cargo:rustc-link-lib=static:-bundle,+whole-archive=cudart_static");
    if !is_target_msvc {
        // cudart_static uses dlopen and POSIX realtime clocks internally.
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=rt");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=stdc++");
    }
}

fn remove_lines<P: AsRef<std::path::Path>>(file: P, patterns: &[&str]) {
    let content = std::fs::read_to_string(&file).unwrap();
    let filtered = content
        .lines()
        .filter(|line| !patterns.iter().any(|p| line.contains(p)))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(file, filtered).unwrap();
}

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

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
    let cuda_path = std::env::var("CUDA_PATH").unwrap_or_else(|_| {
        if is_target_msvc {
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6".to_string()
        } else {
            "/usr/local/cuda".to_string()
        }
    });

    let search_dirs: Vec<String> = if is_target_msvc {
        vec![format!("{cuda_path}/lib/x64")]
    } else {
        vec![
            format!("{cuda_path}/lib64"),
            format!("{cuda_path}/lib"),
            format!("{cuda_path}/targets/x86_64-linux/lib"),
            format!("{cuda_path}/targets/sbsa-linux/lib"),
            format!("{cuda_path}/targets/aarch64-linux/lib"),
        ]
    };

    let lib_filename = if is_target_msvc {
        "cudart_static.lib"
    } else {
        "libcudart_static.a"
    };

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
        println!(
            "cargo:warning=candle-kernels: could not locate {lib_filename} under \
             CUDA_PATH={cuda_path}; the final link step is likely to fail with \
             undefined cudart symbols."
        );
        // Still add the canonical lib64 path as a last-ditch effort — the
        // linker may find the file even if our probe missed it (e.g. via a
        // symlink chain we didn't walk).
        for dir in &search_dirs {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }

    // Statically link the CUDA runtime instead of hard-linking libcudart.so.
    // libcudart_static.a resolves the CUDA driver API (libcuda.so) via
    // dlopen/dlsym internally, so the final binary ends up with NO DT_NEEDED
    // entries for libcudart / libcuda / libcublas / libcurand — matching the
    // behaviour already achieved for cudarc via `fallback-dynamic-loading`.
    // This is what makes "brew install inferrs" viable as a single binary
    // that dlopens whatever CUDA libs are present at runtime (12.x, 13.x, …)
    // and falls back cleanly on systems without CUDA at all.
    println!("cargo:rustc-link-lib=static=cudart_static");
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

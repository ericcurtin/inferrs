.DEFAULT_GOAL := build

# Detect OS and architecture for conditional package inclusion.
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Shared library extension and install_name flag vary by OS.
# On macOS, -install_name @rpath/... is needed so dyld uses the binary's
# rpath (set by build.rs to @executable_path) to find the library at runtime.
ifeq ($(UNAME_S),Darwin)
  LIB_EXT := dylib
  GO_INSTALL_NAME = -ldflags="-extldflags '-Wl,-install_name,@rpath/libocimodels.$(LIB_EXT)'"
  GO_INSTALL_NAME_RELEASE = -ldflags="-extldflags '-Wl,-install_name,@rpath/libocimodels.$(LIB_EXT)' -s -w"
else
  LIB_EXT := so
  GO_INSTALL_NAME =
  GO_INSTALL_NAME_RELEASE = -ldflags='-s -w'
endif

# inferrs-backend-cuda is only built on Linux or Windows x86_64 (not macOS,
# not Windows ARM64).  When it is included, the `cuda` feature of the main
# binary / multimodal / kernels crates is also enabled so the direct-linked
# CUDA execution path compiles alongside the plugin.  Hosts without a CUDA
# toolkit should build without these make targets (e.g. call `cargo build`
# directly with an explicit `-p ...` list and no `--features cuda`) to get
# a CPU-only binary that still probes the runtime backends via dlopen.
#
# Architecture gate: on Windows ARM64 running Git Bash / MSYS2, `uname -s`
# returns something like MINGW64_NT-… (not `Darwin`), so the arch check
# below is what keeps the CUDA flags off that host.  Linux aarch64 (Jetson /
# Grace) is intentionally allowed because NVIDIA ships a CUDA toolkit for it.
ifeq ($(UNAME_S),Darwin)
  CUDA_PKG :=
  CUDA_FEATURES :=
else ifeq ($(UNAME_S),Linux)
  CUDA_PKG := -p inferrs-backend-cuda
  CUDA_FEATURES := --features inferrs/cuda --features inferrs-multimodal/cuda --features inferrs-kernels/cuda
else ifeq ($(UNAME_M),x86_64)
  CUDA_PKG := -p inferrs-backend-cuda
  CUDA_FEATURES := --features inferrs/cuda --features inferrs-multimodal/cuda --features inferrs-kernels/cuda
else
  # Non-x86_64 non-Linux non-Darwin host (e.g. Windows ARM64): no CUDA.
  CUDA_PKG :=
  CUDA_FEATURES :=
endif

# inferrs-backend-metal is only built on macOS (standard Apple SDK, no exotic
# toolchain required).
ifeq ($(UNAME_S),Darwin)
  METAL_PKG := -p inferrs-backend-metal
else
  METAL_PKG :=
endif

# inferrs-backend-hexagon compiles cleanly on all host platforms (it fast-fails
# at runtime on non-Snapdragon hardware).  Include it unconditionally.
HEXAGON_PKG := -p inferrs-backend-hexagon

# Packages that can be built/tested without GPU toolchains (CUDA, ROCm).
# Both the Hexagon and OpenVINO backends have no exotic toolchain requirement
# and can be built anywhere (they probe at runtime via dlopen/LoadLibrary).
NO_GPU_PKGS := -p inferrs -p inferrs-benchmark -p inferrs-multimodal -p inferrs-kernels -p inferrs-backend-vulkan $(HEXAGON_PKG) -p inferrs-backend-openvino $(CUDA_PKG) $(METAL_PKG)

.PHONY: all build release lint test ui oci-lib oci-lib-release oci-models oci-models-release

all: lint test build

# Convenience target: rebuild just the inferrs binary (which also re-runs
# build.rs to recompress the web UI if inferrs/ui/index.html changed).
ui:
	cargo build -p inferrs

build: oci-lib
	cargo build $(NO_GPU_PKGS) $(CUDA_FEATURES)

release: oci-lib-release
	cargo build --release $(NO_GPU_PKGS) $(CUDA_FEATURES)

lint:
	cargo fmt --check $(NO_GPU_PKGS)
	cargo clippy $(NO_GPU_PKGS) $(CUDA_FEATURES) -- -D warnings

test:
	cargo test $(NO_GPU_PKGS) $(CUDA_FEATURES)

# Go C shared library for OCI model operations (called via FFI from Rust).
oci-lib:
	mkdir -p target/debug
	cd oci-models && CGO_ENABLED=1 go build -buildmode=c-shared -tags cshared \
		$(GO_INSTALL_NAME) \
		-o ../target/debug/libocimodels.$(LIB_EXT) .

oci-lib-release:
	mkdir -p target/release
	cd oci-models && CGO_ENABLED=1 go build -buildmode=c-shared -tags cshared \
		-trimpath $(GO_INSTALL_NAME_RELEASE) \
		-o ../target/release/libocimodels.$(LIB_EXT) .

# Standalone CLI binary (optional, useful for debugging).
oci-models:
	mkdir -p target/debug
	cd oci-models && go build -o ../target/debug/inferrs-oci-models .

oci-models-release:
	mkdir -p target/release
	cd oci-models && go build -trimpath -ldflags='-s -w' -o ../target/release/inferrs-oci-models .

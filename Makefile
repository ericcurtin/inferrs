.DEFAULT_GOAL := build

# Detect OS and architecture for conditional package inclusion.
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Shared library extension and install_name flag vary by OS.
# On macOS, -install_name @rpath/... is needed so dyld uses the binary's
# rpath (set by build.rs to @executable_path) to find the library at runtime.
ifeq ($(UNAME_S),Darwin)
  LIB_EXT := dylib
  GO_INSTALL_NAME = -ldflags="-extldflags '-Wl,-install_name,@rpath/libocipull.$(LIB_EXT)'"
  GO_INSTALL_NAME_RELEASE = -ldflags="-extldflags '-Wl,-install_name,@rpath/libocipull.$(LIB_EXT)' -s -w"
else
  LIB_EXT := so
  GO_INSTALL_NAME =
  GO_INSTALL_NAME_RELEASE = -ldflags='-s -w'
endif

# inferrs-backend-cuda is only built on Linux or Windows x86_64 (not macOS,
# not Windows ARM64).
ifeq ($(UNAME_S),Darwin)
  CUDA_PKG :=
else
  CUDA_PKG := -p inferrs-backend-cuda
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

.PHONY: all build release lint test ui oci-lib oci-lib-release oci-pull oci-pull-release

all: lint test build

# Convenience target: rebuild just the inferrs binary (which also re-runs
# build.rs to recompress the web UI if inferrs/ui/index.html changed).
ui:
	cargo build -p inferrs

build: oci-lib
	cargo build $(NO_GPU_PKGS)

release: oci-lib-release
	cargo build --release $(NO_GPU_PKGS)

lint:
	cargo fmt --check $(NO_GPU_PKGS)
	cargo clippy $(NO_GPU_PKGS) -- -D warnings

test:
	cargo test $(NO_GPU_PKGS)

# Go C shared library for OCI registry pulls (called via FFI from Rust).
oci-lib:
	mkdir -p target/debug
	cd oci-pull && CGO_ENABLED=1 go build -buildmode=c-shared -tags cshared \
		$(GO_INSTALL_NAME) \
		-o ../target/debug/libocipull.$(LIB_EXT) .

oci-lib-release:
	mkdir -p target/release
	cd oci-pull && CGO_ENABLED=1 go build -buildmode=c-shared -tags cshared \
		-trimpath $(GO_INSTALL_NAME_RELEASE) \
		-o ../target/release/libocipull.$(LIB_EXT) .

# Standalone CLI binary (optional, useful for debugging).
oci-pull:
	mkdir -p target/debug
	cd oci-pull && go build -o ../target/debug/inferrs-oci-pull .

oci-pull-release:
	mkdir -p target/release
	cd oci-pull && go build -trimpath -ldflags='-s -w' -o ../target/release/inferrs-oci-pull .

# Fuzzing

This crate holds cargo-fuzz targets for src/shortnames.rs's model-reference
parsing: parse_registry_ref, validate_reference, resolve_ollama_api. Each
target's oracle lives next to the function it fuzzes in src/shortnames.rs,
behind the non-default fuzzing Cargo feature (fuzz_check_parse_registry_ref,
fuzz_check_validate_reference, fuzz_check_resolve_ollama_api). A signature
change that breaks a target shows up as a compile error, not a silent gap.

fuzzing only widens visibility of those wrapper functions. It does not
change parsing behavior. Root CI compiles this crate
(cargo check --manifest-path fuzz/Cargo.toml) and the fuzzing feature
(cargo clippy --features fuzzing --lib) on every push, but does not run the
fuzzer: that needs a nightly toolchain. .github/workflows/fuzz-nightly.yml
installs one and runs the fuzzer on a schedule, separate from main CI.

## Running locally

Needs a nightly toolchain:

```
rustup install nightly
cargo install cargo-fuzz
cargo +nightly fuzz run parse_registry_ref -- -max_total_time=300
cargo +nightly fuzz run validate_reference -- -max_total_time=300
cargo +nightly fuzz run resolve_ollama_api -- -max_total_time=300
```

Each target reads seeds from fuzz/corpus/<target>/. A unit test replays
those same seeds on every cargo test (see e.g.
parse_registry_ref_oracle_holds_on_the_seed_corpus in src/shortnames.rs),
so a regression pinned by a seed fails in normal CI too, with no fuzzer run
needed.

A crash writes a file under fuzz/artifacts/<target>/. Minimize it and copy
it into fuzz/corpus/<target>/ under a descriptive name before fixing the
bug, so the regression stays pinned:

```
cargo +nightly fuzz tmin parse_registry_ref fuzz/artifacts/parse_registry_ref/<crash-file>
cp <minimized-file> fuzz/corpus/parse_registry_ref/<descriptive-name>
```

fuzz/.gitignore tracks only the hand-curated corpus directories: corpus/*
is ignored by default, with one `!corpus/<target>/` exception per target.
A new target's corpus directory needs its own exception line added there,
or git add silently skips its seeds.

## Lockfile drift

fuzz/Cargo.lock is a separate lockfile from the root Cargo.lock: cargo-fuzz
makes the fuzz crate a standalone workspace member on purpose, so a change
to its dependency tree can't affect the root build. This repo has no
Dependabot config at all (no .github/dependabot.yml), so a Dependabot entry
scoped only to fuzz/ would be inconsistent, not a fix. Until this repo
adopts Dependabot generally, bump both lockfiles by hand together: after
`cargo update` at the root, also run
`cargo update --manifest-path fuzz/Cargo.toml`, and land both diffs in the
same commit.

//! Short-name alias resolution — from the `[aliases]` table of
//! `llmman.conf`, at runtime. Nothing is compiled into the binary.
//!
//! Mirrors podman's approach: files are merged in priority order, later
//! entries winning. See [`crate::config`] for the locations.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Merge the `[aliases]` table of every `llmman.conf`, later files
/// winning.
///
/// A file that will not parse is reported once by [`crate::config`] and
/// leaves no aliases, degrading to "unknown reference" rather than
/// resolving a short name somewhere unintended — an alias pointing at
/// the wrong registry is how `gemma4` becomes someone else's model.
fn load_aliases() -> HashMap<String, String> {
    let mut merged: HashMap<String, String> = HashMap::new();
    for file in crate::config::files().unwrap_or_default() {
        merged.extend(
            file.conf
                .aliases
                .iter()
                .map(|(k, v)| (k.clone(), v.clone())),
        );
    }
    merged
}

fn aliases() -> &'static HashMap<String, String> {
    static CACHE: OnceLock<HashMap<String, String>> = OnceLock::new();
    CACHE.get_or_init(load_aliases)
}

/// Registry/HF URI schemes [`parse_registry_ref`] strips before parsing the
/// remainder. Any other `scheme://` prefix is rejected up front (the
/// object-store schemes in [`PASSTHROUGH_SCHEMES`] are handled separately by
/// [`validate_reference`] before the parser runs).
const REGISTRY_SCHEMES: &[&str] = &["hf", "huggingface", "ms", "modelscope"];

/// Object-store URI scheme prefixes that [`resolve_inner`] forwards verbatim
/// to the Go shim, so [`validate_reference`] does not apply the registry
/// per-part grammar of [`parse_registry_ref`] to their remainder. This is a
/// validation choice, not a claim that the remainder is
/// unstructured: the Go handlers do parse it (`pullNGC` expects 2-3
/// components, `pullGCS`/`pullS3` split bucket and key), and a shape they
/// reject still fails downstream. Shared by both functions so validation and
/// resolution cannot drift.
pub(crate) const PASSTHROUGH_SCHEMES: &[&str] = &["ngc://", "s3://", "gs://"];

/// A model reference that [`validate_reference`] rejected. Carries the full
/// human-readable message (already includes the offending reference and the
/// reason) so callers can surface it directly.
#[derive(Debug)]
pub struct InvalidReference(String);

impl std::fmt::Display for InvalidReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InvalidReference {}

/// A registry/HF reference decomposed into its parts by
/// [`parse_registry_ref`]. Every field has already passed its per-part byte
/// allowlist and length bound, so a `ParsedRef` cannot hold a malformed part.
struct ParsedRef<'a> {
    /// Known URI scheme ("hf", "huggingface", "ms", "modelscope"), if any.
    scheme: Option<&'a str>,
    /// Registry host, optionally with a ":port" suffix.
    host: Option<&'a str>,
    /// Path components between the host (if any) and the model.
    namespace: Vec<&'a str>,
    model: &'a str,
    tag: Option<&'a str>,
    digest: Option<&'a str>,
}

/// Maximum byte length for a registry host (including an optional ":port").
const MAX_HOST_LEN: usize = 350;
/// Maximum byte length for every part other than the host (namespace
/// component, model, tag, digest).
/// 128 rather than 80 because real model names exceed 80 bytes: HuggingFace
/// caps repo names at 96 characters, and several of the top-500 GGUF repos
/// by downloads run close to that cap.
const MAX_PART_LEN: usize = 128;

/// True for the bytes allowed as the first byte of a name part.
fn is_part_first_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// True for the bytes allowed after the first byte of a name part.
fn is_part_byte(b: u8) -> bool {
    is_part_first_byte(b) || b == b'.' || b == b'-'
}

/// Validate one name part against the byte allowlist: first byte
/// `[A-Za-z0-9_]`, remaining bytes `[A-Za-z0-9_.-]` (minus '.' when
/// `allow_dot` is false), at most `max_len` bytes. Returns the failure
/// detail; the caller prefixes the full reference.
fn check_part(part: &str, kind: &str, allow_dot: bool, max_len: usize) -> Result<(), String> {
    if part.is_empty() {
        return Err(format!("{kind} is empty"));
    }
    if part.len() > max_len {
        return Err(format!("{kind} exceeds {max_len} bytes"));
    }
    let bytes = part.as_bytes();
    if !is_part_first_byte(bytes[0]) {
        return Err(format!(
            "{kind} {part:?} starts with invalid character {:?}",
            bytes[0] as char
        ));
    }
    for &b in &bytes[1..] {
        if !is_part_byte(b) || (b == b'.' && !allow_dot) {
            return Err(format!(
                "{kind} {part:?} contains invalid character {:?}",
                b as char
            ));
        }
    }
    Ok(())
}

/// Validate a registry host: a DNS-name allowlist ('.' permitted) plus at
/// most one ":port" suffix of 1-5 digits, MAX_HOST_LEN bytes total.
fn check_host(host: &str) -> Result<(), String> {
    if host.len() > MAX_HOST_LEN {
        return Err(format!("host exceeds {MAX_HOST_LEN} bytes"));
    }
    let (name, port) = match host.split_once(':') {
        Some((name, port)) => (name, Some(port)),
        None => (host, None),
    };
    check_part(name, "host", true, MAX_HOST_LEN)?;
    if let Some(port) = port {
        if port.is_empty() || port.len() > 5 || !port.bytes().all(|b| b.is_ascii_digit()) {
            return Err(format!("host port {port:?} is not 1-5 digits"));
        }
    }
    Ok(())
}

/// Validate an "@digest" suffix: `<algo>:<hex>` where algo is `[a-z0-9]+`
/// and hex is non-empty `[0-9a-fA-F]+`, MAX_PART_LEN bytes total.
fn check_digest(digest: &str) -> Result<(), String> {
    if digest.len() > MAX_PART_LEN {
        return Err(format!("digest exceeds {MAX_PART_LEN} bytes"));
    }
    let Some((algo, hex)) = digest.split_once(':') else {
        return Err(format!(
            "digest {digest:?} is missing the algorithm separator ':'"
        ));
    };
    if algo.is_empty()
        || !algo
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
    {
        return Err(format!("digest algorithm {algo:?} is not [a-z0-9]+"));
    }
    if hex.is_empty() || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("digest value {hex:?} is not hexadecimal"));
    }
    Ok(())
}

/// True if a leading path component names a registry host rather than a
/// namespace: it contains a dot or colon (a "host:port" form, e.g.
/// "localhost:5000"; a real repository name never contains a colon itself,
/// matching docker/distribution's own reference grammar) or equals
/// "localhost" outright.
fn is_host_component(first: &str) -> bool {
    first.contains('.') || first.contains(':') || first.eq_ignore_ascii_case("localhost")
}

/// Split a registry/HF reference into scheme / host / namespace / model /
/// tag / digest and validate each part against its byte allowlist and
/// length bound. Malformed input fails by construction: any byte outside a
/// part's allowlist (whitespace, control chars, '%', '?', '#', a stray
/// ':' or '/'), an empty component, a ".." component, or an over-length
/// part has no valid decomposition, so no per-shape reject rules exist.
///
/// Grammar (matching the shapes [`resolve_inner`] resolves):
///   - Optional known scheme prefix ([`REGISTRY_SCHEMES`] only; any other
///     "scheme://" is malformed). The remainder parses with the same rules,
///     so "hf://owner/repo" has no host and "hf://hf.co/owner/repo" does.
///   - "@digest" (split at the first '@'): `<algo>:<hex>`.
///   - ":tag" (a ':' after the last '/'): a name part. A ':' before the
///     last '/' belongs to the host:port instead.
///   - The rest splits on '/': the first of two or more components is the
///     host when it contains '.' or ':' or equals "localhost" (the same
///     rule [`has_host`] exposes); everything between host and model is
///     namespace components.
///   - Host: `[A-Za-z0-9_.-]` plus one optional ":port" of 1-5 digits,
///     at most [`MAX_HOST_LEN`] bytes.
///   - Namespace components, model, tag: first byte `[A-Za-z0-9_]`, rest
///     `[A-Za-z0-9_.-]`, at most [`MAX_PART_LEN`] bytes each. '.' is
///     rejected in namespace components (it only belongs in a host, a
///     model name, or a tag).
fn parse_registry_ref(reference: &str) -> Result<ParsedRef<'_>, InvalidReference> {
    let fail = |detail: String| {
        InvalidReference(format!("invalid model reference {reference:?}: {detail}"))
    };

    if reference.trim().is_empty() {
        return Err(fail("empty".to_owned()));
    }

    let (scheme, rest) = match reference.split_once("://") {
        Some((scheme, rest)) if REGISTRY_SCHEMES.contains(&scheme) => (Some(scheme), rest),
        Some((scheme, _)) => {
            return Err(fail(format!("unsupported or malformed scheme {scheme:?}")));
        }
        None => (None, reference),
    };
    if rest.is_empty() {
        return Err(fail("empty reference after scheme".to_owned()));
    }

    let (rest, digest) = match rest.split_once('@') {
        Some((rest, digest)) => {
            check_digest(digest).map_err(&fail)?;
            (rest, Some(digest))
        }
        None => (rest, None),
    };

    // A ':' after the last '/' starts the tag; a ':' before it can only be
    // a host:port and is validated as part of the host below.
    let last_component_start = rest.rfind('/').map_or(0, |i| i + 1);
    let (name, tag) = match rest[last_component_start..].find(':') {
        Some(offset) => {
            let i = last_component_start + offset;
            (&rest[..i], Some(&rest[i + 1..]))
        }
        None => (rest, None),
    };
    if let Some(tag) = tag {
        check_part(tag, "tag", true, MAX_PART_LEN).map_err(&fail)?;
    }

    let components: Vec<&str> = name.split('/').collect();
    let (host, path) = match components.split_first() {
        Some((first, path)) if !path.is_empty() && is_host_component(first) => (Some(*first), path),
        _ => (None, components.as_slice()),
    };
    if let Some(host) = host {
        check_host(host).map_err(&fail)?;
    }
    // `path` is non-empty: `components` has at least one element (split of a
    // non-empty string), and the host branch only fires when more follow it.
    let (model, namespace) = path.split_last().expect("path has at least the model");
    for component in namespace {
        check_part(component, "namespace component", false, MAX_PART_LEN).map_err(&fail)?;
    }
    check_part(model, "model", true, MAX_PART_LEN).map_err(&fail)?;

    let parsed = ParsedRef {
        scheme,
        host,
        namespace: namespace.to_vec(),
        model,
        tag,
        digest,
    };
    // The decomposition must be lossless: rejoining the parts reproduces
    // the input byte for byte, so no byte of an accepted reference can
    // hide between parts (a stray delimiter fails a check above instead).
    // Enforced on every parse in debug builds; the rejoin test pins the
    // same property against a fixed corpus.
    debug_assert_eq!(rejoin(&parsed), reference);
    Ok(parsed)
}

/// Reserialize a [`ParsedRef`] back into reference syntax: the inverse of
/// [`parse_registry_ref`], used by its lossless-decomposition assertion.
fn rejoin(parsed: &ParsedRef<'_>) -> String {
    let mut out = String::new();
    if let Some(scheme) = parsed.scheme {
        out.push_str(scheme);
        out.push_str("://");
    }
    let mut components: Vec<&str> = Vec::new();
    components.extend(parsed.host);
    components.extend(parsed.namespace.iter().copied());
    components.push(parsed.model);
    out.push_str(&components.join("/"));
    if let Some(tag) = parsed.tag {
        out.push(':');
        out.push_str(tag);
    }
    if let Some(digest) = parsed.digest {
        out.push('@');
        out.push_str(digest);
    }
    out
}

/// Panics unless every field of a successful parse holds its own grammar:
/// name parts (model, tag, namespace components, host name) against the
/// per-part byte allowlist, the host port against the 1-5 digit rule, the
/// digest against the `<algo>:<hex>` shape, and the scheme against
/// [`REGISTRY_SCHEMES`]. Shared oracle for the fuzz target and the unit
/// tests below.
///
/// Deliberately independent of `check_part` / `check_host` /
/// `check_digest`: a regression in those functions is exactly what this
/// oracle must catch, so their rules are restated here byte by byte
/// (following the deliberate-copy philosophy the rejoin test documents).
/// Only the two one-line predicates [`is_part_first_byte`] and
/// [`is_part_byte`] are reused; everything else is inlined.
#[cfg(any(test, feature = "fuzzing"))]
fn assert_parsed_ref_grammar(parsed: &ParsedRef<'_>, reference: &str) {
    // First byte [A-Za-z0-9_], rest [A-Za-z0-9_.-] ('.' only when
    // allow_dot), non-empty, at most max_len bytes. ':' is outside the
    // allowlist, so no name part can carry one.
    let assert_name_part = |kind: &str, part: &str, allow_dot: bool, max_len: usize| {
        assert_ne!(part, "..", "{reference:?} parsed with a \"..\" {kind}");
        assert!(!part.is_empty(), "{reference:?} parsed an empty {kind}");
        assert!(
            part.len() <= max_len,
            "{reference:?} parsed an over-length {kind}: {part:?}"
        );
        let bytes = part.as_bytes();
        assert!(
            is_part_first_byte(bytes[0]),
            "{reference:?} parsed a {kind} with an invalid first byte: {part:?}"
        );
        for &b in &bytes[1..] {
            assert!(
                is_part_byte(b) && (allow_dot || b != b'.'),
                "{reference:?} parsed a {kind} with a byte outside its allowlist: {part:?}"
            );
        }
    };

    if let Some(scheme) = parsed.scheme {
        assert!(
            REGISTRY_SCHEMES.contains(&scheme),
            "{reference:?} parsed with an unknown scheme: {scheme:?}"
        );
    }
    if let Some(host) = parsed.host {
        assert!(
            host.len() <= MAX_HOST_LEN,
            "{reference:?} parsed an over-length host: {host:?}"
        );
        // At most one ':', splitting name and port.
        let (name, port) = match host.split_once(':') {
            Some((name, port)) => (name, Some(port)),
            None => (host, None),
        };
        assert_name_part("host name", name, true, MAX_HOST_LEN);
        if let Some(port) = port {
            assert!(
                !port.is_empty() && port.len() <= 5 && port.bytes().all(|b| b.is_ascii_digit()),
                "{reference:?} parsed a host port that is not 1-5 digits: {host:?}"
            );
        }
    }
    for ns in &parsed.namespace {
        assert_name_part("namespace component", ns, false, MAX_PART_LEN);
    }
    assert_name_part("model", parsed.model, true, MAX_PART_LEN);
    if let Some(tag) = parsed.tag {
        assert_name_part("tag", tag, true, MAX_PART_LEN);
    }
    if let Some(digest) = parsed.digest {
        assert_ne!(digest, "..", "{reference:?} parsed with a \"..\" digest");
        assert!(
            digest.len() <= MAX_PART_LEN,
            "{reference:?} parsed an over-length digest: {digest:?}"
        );
        let Some((algo, hex)) = digest.split_once(':') else {
            panic!("{reference:?} parsed a digest without an algo separator: {digest:?}");
        };
        assert!(
            !algo.is_empty()
                && algo
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit()),
            "{reference:?} parsed a digest algorithm that is not [a-z0-9]+: {digest:?}"
        );
        // No per-algorithm hex-length rule: the parser accepts short
        // payloads ("docker.io/ai/gemma4@sha256:abc" is pinned valid), so
        // asserting sha256=64 / sha512=128 would false-positive on
        // accepted parses. The hexdigit check still catches e.g.
        // "sha256:.." ('.' is not a hexdigit).
        assert!(
            !hex.is_empty() && hex.bytes().all(|b| b.is_ascii_hexdigit()),
            "{reference:?} parsed a digest with a non-hex value: {digest:?}"
        );
    }
}

/// Fuzz-target entry point (see `fuzz/fuzz_targets/parse_registry_ref.rs`).
/// Feature-gated so it is absent from every normal build; the `fuzzing`
/// feature only widens visibility, it does not change parsing behavior.
/// Panics whenever a successful parse violates a field's own grammar (see
/// [`assert_parsed_ref_grammar`]): a ".." or over-length part, a byte
/// outside a name part's allowlist, a malformed host port, a non-`algo:hex`
/// digest, or an unknown scheme. The unit tests pin the same oracle against
/// fixed inputs and the seed corpus. Kept here rather than exporting
/// [`ParsedRef`] so the fuzz crate only calls one function and gets a panic
/// (or not) back.
#[cfg(feature = "fuzzing")]
#[doc(hidden)]
pub fn fuzz_check_parse_registry_ref(reference: &str) {
    if let Ok(parsed) = parse_registry_ref(reference) {
        assert_parsed_ref_grammar(&parsed, reference);
    }
}

/// Fuzz-target entry point (see `fuzz/fuzz_targets/validate_reference.rs`).
/// Checks that [`validate_reference`]'s branch (c), everything that is
/// neither an absolute path nor a [`PASSTHROUGH_SCHEMES`] URI, can never
/// drift from [`parse_registry_ref`], since branch (c) is defined as calling
/// it: if `validate_reference(reference)` accepts, `parse_registry_ref`
/// must accept too, and the parse it produces must hold
/// [`assert_parsed_ref_grammar`]. Branches (a) and (b) are exempt from this
/// cross-check (they never call `parse_registry_ref`), so this only asserts
/// they do not panic.
///
/// Reuses `assert_parsed_ref_grammar` directly instead of re-deriving its
/// checks here, unlike this file's usual deliberate-copy philosophy (see
/// the rejoin test and `assert_parsed_ref_grammar`'s own doc comment). That
/// philosophy guards against a bug shared between an implementation and its
/// own oracle. This oracle instead checks agreement between two different
/// functions, so there is no shared-bug risk from reuse here: a regression
/// in the grammar `assert_parsed_ref_grammar` enforces is already caught by
/// its own test coverage, not by this cross-check.
#[cfg(feature = "fuzzing")]
#[doc(hidden)]
pub fn fuzz_check_validate_reference(reference: &str) {
    if reference.starts_with('/') || PASSTHROUGH_SCHEMES.iter().any(|s| reference.starts_with(s)) {
        let _ = validate_reference(reference);
        return;
    }
    if validate_reference(reference).is_ok() {
        match parse_registry_ref(reference) {
            Ok(parsed) => assert_parsed_ref_grammar(&parsed, reference),
            Err(e) => panic!(
                "{reference:?}: validate_reference() accepted but parse_registry_ref() \
                 rejected it: {e}"
            ),
        }
    }
}

/// Fuzz-target entry point (see `fuzz/fuzz_targets/resolve_ollama_api.rs`).
/// Checks that [`resolve_ollama_api`] and [`validate_reference`] never
/// disagree on whether a reference is valid: both start by calling the same
/// [`validate_reference_parsed`] internally, so an Ok/Err split between them
/// would mean one of the two stopped sharing that validation. Panics
/// showing both results when they disagree.
#[cfg(feature = "fuzzing")]
#[doc(hidden)]
pub fn fuzz_check_resolve_ollama_api(reference: &str) {
    let resolved = resolve_ollama_api(reference);
    let validated = validate_reference(reference);
    assert_eq!(
        resolved.is_ok(),
        validated.is_ok(),
        "{reference:?}: resolve_ollama_api() and validate_reference() disagree: \
         resolve_ollama_api={resolved:?}, validate_reference={validated:?}"
    );
}

/// Rejects a raw user-supplied model reference that can never resolve to a
/// real source, before any resolution or network I/O runs.
///
/// Validation branches by the same categories [`resolve_inner`] resolves
/// by, so the two cannot drift:
///   (a) local absolute path ("/..."): checked only for blank and control
///       chars; it is forwarded verbatim to the local-directory importer,
///       so the registry grammar does not apply.
///   (b) passthrough object-store URI ([`PASSTHROUGH_SCHEMES`]): the
///       remainder is an opaque object key, so only a non-empty remainder
///       and no control chars are required.
///   (c) everything else (bare names, owner/repo, host/owner/repo, and the
///       hf/huggingface/ms/modelscope schemes): parsed into host /
///       namespace / model / tag / digest by [`parse_registry_ref`], each
///       part checked against a byte allowlist with length bounds, so
///       malformed input fails by construction rather than by enumerating
///       bad shapes.
pub fn validate_reference(reference: &str) -> Result<(), InvalidReference> {
    validate_reference_parsed(reference).map(|_| ())
}

/// The body of [`validate_reference`], additionally returning the
/// [`ParsedRef`] produced by branch (c) so callers that go on to resolve the
/// reference (`resolve`, `resolve_ollama_api`) don't have to parse it again.
/// `None` for the (a) absolute-path and (b) passthrough-scheme branches,
/// neither of which calls [`parse_registry_ref`].
fn validate_reference_parsed(reference: &str) -> Result<Option<ParsedRef<'_>>, InvalidReference> {
    let reject = |detail: &str| {
        Err(InvalidReference(format!(
            "invalid model reference {reference:?}: {detail}"
        )))
    };

    if reference.trim().is_empty() {
        return reject("empty");
    }

    // (a) Absolute paths go to the local-directory importer. A space is not a
    // control character, so a path like /models/My Model/x.gguf stays valid.
    if reference.starts_with('/') {
        if reference.chars().any(|c| c.is_ascii_control()) {
            return reject("contains a control character");
        }
        return Ok(None);
    }

    // (b) Object-store passthrough URIs carry an opaque key after the bucket,
    // so a trailing slash or other registry-illegal character is legitimate.
    for scheme in PASSTHROUGH_SCHEMES {
        if let Some(rest) = reference.strip_prefix(scheme) {
            if rest.is_empty() {
                return reject(&format!("empty reference after {scheme:?} scheme"));
            }
            if reference.chars().any(|c| c.is_ascii_control()) {
                return reject("contains a control character");
            }
            return Ok(None);
        }
    }

    // (c) Registry/HF references: parse-by-construction.
    parse_registry_ref(reference).map(Some)
}

/// Resolve `reference` through the short-name alias table, then default the
/// registry to `hf.co` when no host is present.
///
/// URI scheme handling (processed before alias lookup):
///   hf:// huggingface://  → strip scheme, continue as bare owner/repo
///   ms:// modelscope://   → normalise to ms:// (crate::sources routes to ModelScope)
///   ngc:// s3:// gs://    → pass through verbatim (crate::sources handles natively)
///   /absolute/path        → pass through verbatim (local directory import)
///
/// Resolution order for everything else:
///   1. Exact alias match  → return the mapped value
///   2. Has a registry host → return as-is
///   3. No host            → prepend `hf.co/`
pub fn resolve(reference: &str) -> Result<String, InvalidReference> {
    let parsed = validate_reference_parsed(reference)?;
    Ok(resolve_inner(reference, parsed))
}

/// The resolution body of [`resolve`] without validation. Callers that have
/// already validated the reference (or that validate a bare shortcut first,
/// like [`resolve_ollama_api`]) use this to avoid validating twice. `parsed`
/// is the [`ParsedRef`] [`validate_reference_parsed`] already produced for
/// `reference` (registry-grammar branch only; `None` for the passthrough and
/// absolute-path branches, which return before it would be consulted) so the
/// host check below doesn't call [`parse_registry_ref`] a second time.
fn resolve_inner(reference: &str, parsed: Option<ParsedRef<'_>>) -> String {
    // ── URI schemes that bypass alias lookup and hf.co defaulting ─────────
    // Local absolute paths and object-store URIs are forwarded as-is to
    // crate::sources, which dispatches them to the appropriate source
    // handler.
    for passthrough in PASSTHROUGH_SCHEMES {
        if reference.starts_with(passthrough) {
            return reference.to_owned();
        }
    }
    if reference.starts_with('/') {
        return reference.to_owned();
    }

    // ── Normalise well-known URI schemes to canonical form ─────────────────
    // hf:// and huggingface:// are stripped; the remainder is treated as a
    // bare HuggingFace owner/repo reference through the normal path below.
    let reference = if let Some(r) = reference
        .strip_prefix("hf://")
        .or_else(|| reference.strip_prefix("huggingface://"))
    {
        r
    }
    // ms:// and modelscope:// are normalised to ms:// so crate::sources can
    // detect the scheme and route to the ModelScope download path.
    else if let Some(r) = reference.strip_prefix("modelscope://") {
        return format!("ms://{r}");
    } else if reference.starts_with("ms://") {
        return reference.to_owned();
    } else {
        reference
    };

    // ── Alias lookup → hf.co default ──────────────────────────────────────
    if let Some(mapped) = aliases().get(reference) {
        return mapped.clone();
    }
    if parsed.is_some_and(|p| p.host.is_some()) {
        return reference.to_owned();
    }
    format!("hf.co/{reference}")
}

/// Returns true if `reference` is bare: no "/" at all, i.e. no owner/repo
/// or registry-host structure — just a single path component, optionally
/// with a ":tag". Dots are deliberately *not* checked here (unlike a
/// stricter earlier version of this function): a dotted version number
/// such as "3.5" in "qwen3.5:0.8B" is just part of the name/tag, not a
/// registry host, and real ollama makes the same distinction purely on
/// "/" — it never treats embedded dots specially. Since a single bare
/// component (with or without dots) can never satisfy HuggingFace's
/// required host/owner/repo shape anyway, sending it to `resolve`'s
/// hf.co default would be a guaranteed dead end; docker.io/ai/<reference>
/// below is the only default that's ever actually resolvable for it.
fn is_bare(reference: &str) -> bool {
    !reference.contains('/')
}

/// Resolve `reference` the way every Ollama-API-facing path in `cmd::serve`
/// does (handle_pull, handle_show, handle_delete, ensure_model, and the
/// `--model` preload in serve_async): identical to `resolve`, except a
/// *bare* reference — no "/" anywhere, e.g. "gemma4" or "qwen3.5:0.8B" —
/// defaults to Docker's official curated-model namespace on Docker Hub,
/// `docker.io/ai/<reference>` (e.g. "gemma4" -> "docker.io/ai/gemma4",
/// "qwen3.5:0.8B" -> "docker.io/ai/qwen3.5:0.8B"), instead of `resolve`'s
/// general `hf.co/<reference>` default. Dots in the name or tag don't
/// disqualify this — only a "/" does — since `resolve`'s hf.co default
/// requires a host/owner/repo shape that a single bare component can never
/// satisfy anyway. Any reference with a "/" (an owner/repo path, a URI
/// scheme, an explicit host) is left to `resolve`'s normal rules unchanged.
///
/// CLI subcommands that talk to a local server over the Ollama API (pull,
/// push) go through this same resolution server-side, so the docker.io/ai/
/// default is consistent regardless of whether a bare name reaches llmman
/// via the CLI or directly over HTTP.
pub fn resolve_ollama_api(reference: &str) -> Result<String, InvalidReference> {
    // Validate up front: the bare-name branch below returns without calling
    // resolve, so validation cannot be deferred to it.
    let parsed = validate_reference_parsed(reference)?;
    if is_bare(reference) {
        if let Some(mapped) = aliases().get(reference) {
            return Ok(mapped.clone());
        }
        return Ok(format!("docker.io/ai/{reference}"));
    }
    // Already validated: use the non-validating body so we don't validate twice.
    Ok(resolve_inner(reference, parsed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_reference_accepts_valid_shapes() {
        for r in [
            "gemma4",
            "qwen3.5:0.8B",
            "unsloth/Qwen3.5-0.8B-GGUF",
            "hf.co/foo/bar",
            "localhost:5000/foo/bar:tag",
            "docker.io/ai/gemma4@sha256:abc",
            "hf://owner/repo",
            "ms://x",
            "ngc://x",
            "s3://b/k",
            "gs://b/k",
            // Passthrough object-store keys are opaque: a trailing slash and
            // deeper key paths are legitimate, unlike registry/HF forms.
            "s3://bucket/models/qwen/",
            "gs://bucket/a/b/",
            "ngc://org/team/model",
            "/abs/path/model.gguf",
            "/models/My Model/x.gguf",
        ] {
            assert!(validate_reference(r).is_ok(), "{r:?} should be accepted");
        }
    }

    #[test]
    fn validate_reference_rejects_bad_shapes() {
        for r in [
            "",
            "   ",
            "oci://",
            "http://x",
            "hf://",
            "a\tb",
            "/abs/pa\tth",
            "s3://",
            "s3://bucket/k\tey",
            "../../../etc/passwd",
            "a/../b",
            "docker.io/ai/%2e%2e",
            "hf:///foo",
            "ms:///x",
            "hf.co//foo",
            "owner//repo",
            "trailing/slash/",
            "hf.co/owner/repo?expand[]=siblings",
            "hf.co/owner/repo#frag",
            // Newly rejected by the per-part parser; pinned so a grammar
            // relaxation cannot silently re-accept them.
            "a b",
            "model:",
            "m:t:x",
            "a@b/c",
            "m@sha256:abc:tag",
        ] {
            assert!(validate_reference(r).is_err(), "{r:?} should be rejected");
        }
    }

    #[test]
    fn validate_reference_rejects_over_length_parts() {
        // Host over 350 bytes.
        assert!(validate_reference(&format!("{}.com/ns/model", "a".repeat(350))).is_err());
        // Host at the bound stays valid ("a"*346 + ".com" = 350 bytes).
        assert!(validate_reference(&format!("{}.com/ns/model", "a".repeat(346))).is_ok());
        // Model over 128 bytes; at the bound stays valid.
        assert!(validate_reference(&"a".repeat(129)).is_err());
        assert!(validate_reference(&"a".repeat(128)).is_ok());
        // Tag and namespace component over 128 bytes.
        assert!(validate_reference(&format!("model:{}", "t".repeat(129))).is_err());
        assert!(validate_reference(&format!("hf.co/{}/model", "n".repeat(129))).is_err());
    }

    /// Real repos exceed 80 bytes in the model part: 4 of the top 500 GGUF
    /// repos on HuggingFace by downloads do, including these two. They are
    /// why [`MAX_PART_LEN`] is 128 rather than 80.
    #[test]
    fn validate_reference_accepts_long_real_model_names() {
        for r in [
            "hf.co/DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF",
            "hf.co/mradermacher/Llama3.3-8B-Instruct-Thinking-Heretic-Uncensored-Claude-4.5-Opus-High-Reasoning-i1-GGUF",
        ] {
            assert!(validate_reference(r).is_ok(), "{r:?} should be accepted");
        }
    }

    #[test]
    fn validate_reference_rejects_dotted_namespace_components() {
        // '.' belongs in hosts, models, and tags, not namespace components.
        assert!(validate_reference("hf.co/own.er/repo").is_err());
        assert!(validate_reference("docker.io/a.i/gemma4").is_err());
        // A dotted model and a dotted tag stay valid.
        assert!(validate_reference("hf.co/owner/re.po").is_ok());
        assert!(validate_reference("qwen3.5:0.8B").is_ok());
    }

    #[test]
    fn validate_reference_rejects_malformed_digests() {
        for r in [
            "docker.io/ai/gemma4@sha256",
            "docker.io/ai/gemma4@sha256:",
            "docker.io/ai/gemma4@:abc",
            "docker.io/ai/gemma4@sha256:xyz-nonhex",
            "docker.io/ai/gemma4@SHA256:abc",
            "docker.io/ai/gemma4@",
        ] {
            assert!(validate_reference(r).is_err(), "{r:?} should be rejected");
        }
        // Uppercase hex and a tag before the digest stay valid.
        assert!(validate_reference("docker.io/ai/gemma4@sha256:ABCDEF").is_ok());
        assert!(validate_reference("docker.io/ai/gemma4:tag@sha256:abc").is_ok());
    }

    #[test]
    fn validate_reference_applies_the_port_rule() {
        assert!(validate_reference("localhost:5000/x").is_ok());
        assert!(validate_reference("host.example:99999/x").is_ok());
        // Six digits, empty, and non-numeric ports are rejected.
        assert!(validate_reference("host.example:999999/x").is_err());
        assert!(validate_reference("localhost:/x").is_err());
        assert!(validate_reference("host.example:8o80/x").is_err());
    }

    /// For every valid reference, splitting into parts and rejoining them
    /// must reproduce the input byte for byte: the parser loses nothing.
    #[test]
    fn parse_registry_ref_parts_rejoin_to_the_input() {
        for r in [
            "gemma4",
            "qwen3.5:0.8B",
            "unsloth/Qwen3.5-0.8B-GGUF",
            "hf.co/foo/bar",
            "localhost:5000/foo/bar:tag",
            "docker.io/ai/gemma4@sha256:abc",
            "docker.io/ai/gemma4:tag@sha256:ABCDEF",
            "hf://owner/repo",
            "huggingface://owner/repo",
            "hf://hf.co/owner/repo",
            "ms://owner/repo",
            "modelscope://owner/repo",
            "example.com:5000/ns/model:tag",
            "localhost/ns/model",
        ] {
            let parsed = parse_registry_ref(r).unwrap_or_else(|e| panic!("{e}"));
            // Deliberate copy of rejoin(): an independent oracle, so a bug
            // in rejoin() itself cannot self-confirm through the
            // debug_assert. Update both when the grammar grows a field.
            let mut rejoined = String::new();
            if let Some(scheme) = parsed.scheme {
                rejoined.push_str(scheme);
                rejoined.push_str("://");
            }
            let mut components: Vec<&str> = Vec::new();
            components.extend(parsed.host);
            components.extend(&parsed.namespace);
            components.push(parsed.model);
            rejoined.push_str(&components.join("/"));
            if let Some(tag) = parsed.tag {
                rejoined.push(':');
                rejoined.push_str(tag);
            }
            if let Some(digest) = parsed.digest {
                rejoined.push('@');
                rejoined.push_str(digest);
            }
            assert_eq!(rejoined, r);
        }
    }

    /// Fuzz-shaped invariant: every input that parses successfully must
    /// decompose into fields that hold their own grammar (the shared
    /// [`assert_parsed_ref_grammar`] oracle). Nasty inputs must either fail
    /// to parse or pass the per-field checks.
    #[test]
    fn parse_registry_ref_holds_the_part_invariants() {
        let long = "a".repeat(400);
        let nasty = [
            "../../../etc/passwd".to_owned(),
            "a/../b".to_owned(),
            "hf.co/../x".to_owned(),
            "hf://../x".to_owned(),
            "%2e%2e/x".to_owned(),
            "a?b".to_owned(),
            "a#b".to_owned(),
            "a b".to_owned(),
            "a\tb".to_owned(),
            "a\0b".to_owned(),
            ":/x".to_owned(),
            "://x".to_owned(),
            "x@sha256:..".to_owned(),
            "x:..".to_owned(),
            long.clone(),
            format!("{long}/m"),
            format!("{long}.com/m"),
            format!("m:{long}"),
            format!("m@sha256:{long}"),
            format!("hf.co/{long}/m"),
            "hf.co/owner/repo".to_owned(),
            "qwen3.5:0.8B".to_owned(),
        ];
        for r in &nasty {
            let Ok(parsed) = parse_registry_ref(r) else {
                continue;
            };
            assert_parsed_ref_grammar(&parsed, r);
        }
    }

    /// Sanity-runs the fuzz-harness logic against the checked-in seed
    /// corpus on every `cargo test`, so a seeded regression (e.g. the
    /// digest-dotdot seed) fails here even when no fuzzer runs.
    #[test]
    fn parse_registry_ref_oracle_holds_on_the_seed_corpus() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/corpus/parse_registry_ref");
        let mut seeds = 0usize;
        for entry in std::fs::read_dir(&dir).expect("read the seed corpus directory") {
            let path = entry.expect("read a corpus directory entry").path();
            let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
            seeds += 1;
            // Mirror the fuzz target byte for byte: skip non-UTF-8 input
            // (the nul-byte seed is valid UTF-8 and goes through), ignore
            // parse errors, run the oracle on every successful parse.
            let Ok(s) = std::str::from_utf8(&data) else {
                continue;
            };
            if let Ok(parsed) = parse_registry_ref(s) {
                assert_parsed_ref_grammar(&parsed, s);
            }
        }
        // A wrong path must fail loudly, not pass over zero files.
        assert!(seeds > 0, "seed corpus at {dir:?} is empty");
    }

    /// Sanity-runs the `fuzz_check_validate_reference` oracle against the
    /// checked-in seed corpus on every `cargo test`. Mirrors the fuzz
    /// target's logic inline (rather than calling the feature-gated
    /// `fuzz_check_validate_reference` itself) so this runs in a plain
    /// `cargo test --lib`, with no `fuzzing` feature required.
    #[test]
    fn validate_reference_oracle_holds_on_the_seed_corpus() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/corpus/validate_reference");
        let mut seeds = 0usize;
        for entry in std::fs::read_dir(&dir).expect("read the seed corpus directory") {
            let path = entry.expect("read a corpus directory entry").path();
            let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
            seeds += 1;
            let Ok(s) = std::str::from_utf8(&data) else {
                continue;
            };
            if s.starts_with('/') || PASSTHROUGH_SCHEMES.iter().any(|p| s.starts_with(p)) {
                continue;
            }
            if validate_reference(s).is_ok() {
                let parsed = parse_registry_ref(s).unwrap_or_else(|e| {
                    panic!("{s:?}: validate_reference() accepted but parse_registry_ref() rejected it: {e}")
                });
                assert_parsed_ref_grammar(&parsed, s);
            }
        }
        assert!(seeds > 0, "seed corpus at {dir:?} is empty");
    }

    /// Sanity-runs the `fuzz_check_resolve_ollama_api` oracle against the
    /// checked-in seed corpus on every `cargo test`. Mirrors the fuzz
    /// target's logic inline, same reason as the sibling test above.
    #[test]
    fn resolve_ollama_api_oracle_holds_on_the_seed_corpus() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/corpus/resolve_ollama_api");
        let mut seeds = 0usize;
        for entry in std::fs::read_dir(&dir).expect("read the seed corpus directory") {
            let path = entry.expect("read a corpus directory entry").path();
            let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
            seeds += 1;
            let Ok(s) = std::str::from_utf8(&data) else {
                continue;
            };
            assert_eq!(
                resolve_ollama_api(s).is_ok(),
                validate_reference(s).is_ok(),
                "{s:?}: resolve_ollama_api() and validate_reference() disagree"
            );
        }
        assert!(seeds > 0, "seed corpus at {dir:?} is empty");
    }

    #[test]
    fn resolve_ollama_api_defaults_bare_names_to_docker_ai() {
        assert_eq!(resolve_ollama_api("gemma4").unwrap(), "docker.io/ai/gemma4");
        // A tag with no dot is still "bare" by this rule (only "/"
        // disqualifies it): matches the ai/<name>:<tag> shape on Docker Hub.
        assert_eq!(
            resolve_ollama_api("gemma4:e4b").unwrap(),
            "docker.io/ai/gemma4:e4b"
        );
        // Dots in the name and/or tag don't disqualify "bare" either — only
        // a "/" does. Regression test: this used to fall through unchanged
        // (neither hf.co- nor docker.io/ai-prefixed) because has_host()
        // mistook the dotted version number for an explicit registry host,
        // and dead-ended in the Go shim's HF-only parser with a misleading
        // "invalid HuggingFace reference" error instead.
        assert_eq!(
            resolve_ollama_api("qwen3.5").unwrap(),
            "docker.io/ai/qwen3.5"
        );
        assert_eq!(
            resolve_ollama_api("qwen3.5:0.8B").unwrap(),
            "docker.io/ai/qwen3.5:0.8B"
        );
    }

    #[test]
    fn resolve_ollama_api_leaves_structured_references_to_resolve() {
        // Owner/repo (has a "/") falls back to resolve()'s hf.co default.
        assert_eq!(
            resolve_ollama_api("unsloth/Qwen3.5-0.8B-GGUF").unwrap(),
            resolve("unsloth/Qwen3.5-0.8B-GGUF").unwrap()
        );
        // Already has an explicit host.
        assert_eq!(
            resolve_ollama_api("hf.co/foo/bar").unwrap(),
            "hf.co/foo/bar"
        );
        assert_eq!(
            resolve_ollama_api("docker.io/ai/gemma4").unwrap(),
            "docker.io/ai/gemma4"
        );
    }

    #[test]
    fn resolve_ollama_api_matches_resolve_for_uri_schemes_and_paths() {
        assert_eq!(
            resolve_ollama_api("hf://unsloth/Qwen3.5-0.8B-GGUF").unwrap(),
            resolve("hf://unsloth/Qwen3.5-0.8B-GGUF").unwrap()
        );
        assert_eq!(
            resolve_ollama_api("/abs/path/model.gguf").unwrap(),
            "/abs/path/model.gguf"
        );
    }

    #[test]
    fn is_bare_rejects_only_slashes() {
        assert!(is_bare("gemma4"));
        assert!(is_bare("gemma4:e4b"));
        assert!(!is_bare("unsloth/gemma4"));
        // Dots alone (no "/") no longer disqualify bareness — see
        // resolve_requires_a_slash_to_treat_a_component_as_a_host below for
        // the corresponding fix.
        assert!(is_bare("qwen3.5"));
        assert!(is_bare("qwen3.5:0.8B"));
        assert!(!is_bare("hf.co/gemma4"));
    }

    /// Ported from ollama's types/model/name_test.go (TestParseNameParts /
    /// TestNameparseNameDefault): ollama fills an unqualified name out to
    /// registry.ollama.ai/library/<model>:latest; llmman's equivalents are
    /// resolve_ollama_api's docker.io/ai/<name> default for bare names and
    /// resolve's hf.co/<owner>/<repo> default for host-less paths, while
    /// anything already carrying a host passes through untouched.
    #[test]
    fn resolve_fills_in_default_registry_like_ollama_parse_name() {
        // Bare model name (ollama: "model" -> registry.ollama.ai/library/model:latest).
        assert_eq!(
            resolve_ollama_api("mistral").unwrap(),
            "docker.io/ai/mistral"
        );
        assert_eq!(
            resolve_ollama_api("mistral:7b").unwrap(),
            "docker.io/ai/mistral:7b"
        );
        // namespace/model (ollama: -> registry.ollama.ai/namespace/model).
        assert_eq!(resolve("namespace/model").unwrap(), "hf.co/namespace/model");
        // Fully-qualified references pass through untouched...
        assert_eq!(
            resolve("example.com/ns/model:tag").unwrap(),
            "example.com/ns/model:tag"
        );
        // ...including a host:port first component (ollama's
        // "host:port/namespace/model:tag" case) and localhost.
        assert_eq!(
            resolve("example.com:5000/ns/model:tag").unwrap(),
            "example.com:5000/ns/model:tag"
        );
        assert_eq!(resolve("localhost/ns/model").unwrap(), "localhost/ns/model");
        // A dot-free, colon-free first component is NOT a host, even in a
        // 3-component reference: the hf.co default applies. This pins the
        // on-disk store key for such references; changing the host-detection
        // rule (e.g. "3+ components always have a host") is a breaking change.
        assert_eq!(resolve("a/b/c").unwrap(), "hf.co/a/b/c");
        assert_eq!(resolve("hf://a/b/c").unwrap(), "hf.co/a/b/c");
    }

    /// Ported from ollama's types/model/name_test.go scheme cases
    /// ("scheme://host/namespace/model:tag" parses with the scheme split
    /// off): llmman likewise never treats a URI scheme as part of the
    /// reference — hf:// and huggingface:// are stripped before the normal
    /// defaulting rules run, modelscope:// is normalised to ms://, and
    /// object-store schemes pass through verbatim.
    #[test]
    fn resolve_splits_uri_schemes_like_ollama_parse_name() {
        assert_eq!(resolve("hf://owner/repo").unwrap(), "hf.co/owner/repo");
        assert_eq!(
            resolve("huggingface://owner/repo").unwrap(),
            "hf.co/owner/repo"
        );
        assert_eq!(
            resolve("hf://hf.co/owner/repo").unwrap(),
            "hf.co/owner/repo"
        );
        assert_eq!(
            resolve("modelscope://owner/repo").unwrap(),
            "ms://owner/repo"
        );
        assert_eq!(resolve("ms://owner/repo").unwrap(), "ms://owner/repo");
        assert_eq!(resolve("s3://bucket/key").unwrap(), "s3://bucket/key");
        assert_eq!(resolve("gs://bucket/key").unwrap(), "gs://bucket/key");
        assert_eq!(resolve("ngc://org/model").unwrap(), "ngc://org/model");
    }

    #[test]
    fn resolve_requires_a_slash_to_treat_a_component_as_a_host() {
        // No "/" at all: a dotted version number must not be mistaken for
        // an explicit host, no matter how host-like the dot looks. So
        // resolve() prepends the hf.co default instead of leaving it as-is.
        assert_eq!(resolve("qwen3.5:0.8B").unwrap(), "hf.co/qwen3.5:0.8B");
        assert_eq!(resolve("qwen3.5").unwrap(), "hf.co/qwen3.5");
        // With a "/", the first component is genuinely checked for a host.
        // A host-shaped first component passes through untouched.
        assert_eq!(resolve("hf.co/foo/bar").unwrap(), "hf.co/foo/bar");
        assert_eq!(resolve("localhost/foo").unwrap(), "localhost/foo");
        assert_eq!(
            resolve("unsloth/Qwen3.5-0.8B-GGUF").unwrap(),
            "hf.co/unsloth/Qwen3.5-0.8B-GGUF"
        );
    }

    #[test]
    // Regression: "localhost:PORT/..." (a local test registry) was mistaken
    // for a host-less reference, since neither the dot check nor the exact
    // "localhost" match recognized it. "resolve" then wrongly prepended
    // "hf.co/", producing "hf.co/localhost:PORT/...".
    fn resolve_recognizes_an_explicit_port_as_a_host() {
        assert_eq!(
            resolve("localhost:5000/foo/bar").unwrap(),
            "localhost:5000/foo/bar"
        );
        assert_eq!(
            resolve("registry.example.com:5000/foo").unwrap(),
            "registry.example.com:5000/foo"
        );
        assert_eq!(
            resolve("localhost:5000/foo/bar:tag").unwrap(),
            "localhost:5000/foo/bar:tag"
        );
    }

    #[test]
    fn resolve_and_resolve_ollama_api_reject_invalid_references() {
        assert!(resolve("hf:///foo").is_err());
        assert!(resolve("hf.co//foo").is_err());
        assert!(resolve_ollama_api("hf.co//foo").is_err());
        assert!(resolve_ollama_api("ms:///x").is_err());
        // Valid references still resolve to the expected value.
        assert_eq!(resolve("hf://owner/repo").unwrap(), "hf.co/owner/repo");
        assert_eq!(resolve_ollama_api("gemma4").unwrap(), "docker.io/ai/gemma4");
    }

    #[test]
    fn validate_reference_rejects_malformed_scheme_prefixes() {
        // Single-slash "scheme:/" must not slip past the scheme gate and be
        // read by has_host as a host:port (regression: "http:" was probed).
        assert!(validate_reference("http:/evil.test/x").is_err());
        assert!(validate_reference("oci:/registry/ns/model").is_err());
        assert!(validate_reference("hf:/owner/repo").is_err());
        // Unknown double-slash schemes stay rejected.
        assert!(validate_reference("http://evil.test/x").is_err());
        assert!(validate_reference("oci://registry/ns/model").is_err());
        // Prefixes outside the URI scheme grammar are rejected too. There
        // is no ":/" gate: "scheme://" forms hit the unknown-scheme arm of
        // the "://" split, and single-slash "x:/..." forms parse "x:" as a
        // host whose empty port check_host rejects, so nothing
        // colon-slash-shaped reaches has_host.
        assert!(validate_reference("1a://evil.test/x").is_err());
        assert!(validate_reference("my_scheme://x").is_err());
        assert!(validate_reference("://x").is_err());
        assert!(validate_reference("1a:/evil.test/x").is_err());
        assert!(validate_reference(":/x").is_err());
        assert!(validate_reference("localhost:/foo").is_err());
        // Known scheme://, host:port, tag and digest colons stay valid.
        assert!(validate_reference("hf://owner/repo").is_ok());
        assert!(validate_reference("s3://bucket/models/qwen/").is_ok());
        assert!(validate_reference("localhost:5000/foo/bar:tag").is_ok());
        assert!(validate_reference("docker.io/ai/gemma4@sha256:abc").is_ok());
        assert!(validate_reference("qwen3.5:0.8B").is_ok());
    }
}

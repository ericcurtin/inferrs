//! What the daemon passes to the backends it spawns: the `LLMMAN_*` env
//! vars (context length, flash attention, KV-cache type, scheduling,
//! parallelism, threads, queue and load limits, metrics) and the
//! `--ctx-size` arithmetic around them (trained-context cap, `--parallel`
//! scaling, the OOM shrink retry). Each `*_from_env` reads the process
//! environment; its `parse_*` half is what the tests exercise.

/// Context tokens requested for every backend this daemon spawns — read
/// from `LLMMAN_CONTEXT_LENGTH` (an env var, not a `llmman serve` flag).
/// Forwarded to llama-server as-is for generation models (0 meaning
/// "trained context"); embedding models are always capped to their
/// trained context. See [`initial_ctx_size`]. Not forwarded to vLLM
/// when 0.
///
/// Unset or unparseable, this falls back to [`DEFAULT_CTX_SIZE`].
pub(super) fn context_length_from_env() -> Option<u32> {
    parse_context_length(std::env::var("LLMMAN_CONTEXT_LENGTH").ok().as_deref())
}

/// [`context_length_from_env`]'s parsing, split out so it's testable
/// without mutating the real process environment.
fn parse_context_length(value: Option<&str>) -> Option<u32> {
    value?.trim().parse().ok()
}

/// `--ctx-size` when `LLMMAN_CONTEXT_LENGTH` is unset: 256k regardless
/// of VRAM, capped per model to its trained context (see
/// [`initial_ctx_size`]). A load that then OOMs is retried halved (see
/// [`next_ctx_size_after_oom`]) instead of guessing from memory up front.
pub const DEFAULT_CTX_SIZE: u32 = 262144;

/// Flash Attention mode requested for every `llama-server` this daemon
/// spawns — read from `LLMMAN_FLASH_ATTENTION` (an env var, not a
/// `llmman serve` flag, mirroring [`context_length_from_env`]). Forwarded
/// verbatim as `--flash-attn <mode>`; unset leaves it off llama-server's
/// own command line entirely, falling back to its own default (`auto`,
/// which already enables it whenever the backend/model support it).
pub(super) fn flash_attention_from_env() -> Option<String> {
    parse_flash_attention(std::env::var("LLMMAN_FLASH_ATTENTION").ok().as_deref())
}

/// [`flash_attention_from_env`]'s parsing, split out so it's testable
/// without mutating the real process environment. Accepts llama-server's
/// own vocabulary (`on`/`off`/`auto`) as well as the boolean spelling
/// (`1`/`0`, `true`/`false`) Ollama documents for `OLLAMA_FLASH_ATTENTION`,
/// since users porting a config from there would otherwise silently get
/// llama-server's default instead of what they asked for.
fn parse_flash_attention(value: Option<&str>) -> Option<String> {
    let v = value?.trim();
    if v.is_empty() {
        return None;
    }
    Some(match v.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => "on".to_string(),
        "0" | "false" | "no" => "off".to_string(),
        other => other.to_string(),
    })
}

/// KV-cache quantization type requested for every `llama-server` this
/// daemon spawns — read from `LLMMAN_KV_CACHE_TYPE` (an env var, not a
/// `llmman serve` flag, mirroring [`context_length_from_env`]). Forwarded
/// as both `--cache-type-k` and `--cache-type-v`: llama-server takes
/// those separately, but Ollama's `OLLAMA_KV_CACHE_TYPE` (the convention
/// this mirrors) documents a single value applied to both, and there's no
/// use case yet for setting K and V independently through this daemon.
///
/// One of `f16` (llama-server's own default), `q8_0`, or `q4_0` — the
/// same set Ollama documents — trades output quality for a smaller
/// KV-cache footprint at long context lengths. Not validated here;
/// llama-server rejects an unsupported value itself, surfaced via
/// `wait_for_ready`'s stderr-tail capture same as any other startup
/// failure.
pub(super) fn kv_cache_type_from_env() -> Option<String> {
    parse_kv_cache_type(std::env::var("LLMMAN_KV_CACHE_TYPE").ok().as_deref())
}

/// [`kv_cache_type_from_env`]'s parsing, split out so it's testable
/// without mutating the real process environment.
fn parse_kv_cache_type(value: Option<&str>) -> Option<String> {
    let v = value?.trim();
    (!v.is_empty()).then(|| v.to_string())
}

/// `--split-mode` value requested for every `llama-server` spawn — read
/// from `LLMMAN_SCHED_SPREAD`, llmman's equivalent of Ollama's
/// `OLLAMA_SCHED_SPREAD`. Truthy forwards `--split-mode layer` (spread
/// across every GPU — already llama-server's own default, now explicit);
/// falsey forwards `--split-mode none` (restrict to one GPU). Unset
/// leaves llama-server's own default untouched.
pub(super) fn sched_spread_from_env() -> Option<&'static str> {
    parse_sched_spread(std::env::var("LLMMAN_SCHED_SPREAD").ok().as_deref())
}

/// [`sched_spread_from_env`]'s parsing, split out so it's testable
/// without mutating the real process environment.
fn parse_sched_spread(value: Option<&str>) -> Option<&'static str> {
    let v = value?.trim();
    if v.is_empty() {
        return None;
    }
    match v.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" | "layer" => Some("layer"),
        "0" | "false" | "no" | "off" | "none" => Some("none"),
        _ => None,
    }
}

/// `--parallel <n>` for every `llama-server` this daemon spawns (GGUF
/// models only — vllm/mlx_lm.server handle concurrency their own way,
/// with no equivalent flag), from `LLMMAN_NUM_PARALLEL`. Mirrors
/// Ollama's `OLLAMA_NUM_PARALLEL`; unset leaves llama-server's own
/// default of 1 untouched.
pub(super) fn num_parallel_from_env() -> Option<u32> {
    parse_num_parallel(std::env::var("LLMMAN_NUM_PARALLEL").ok().as_deref())
}

/// `0` is rejected, same as llama-server's own `--parallel` validation.
fn parse_num_parallel(value: Option<&str>) -> Option<u32> {
    let n: u32 = value?.trim().parse().ok()?;
    (n != 0).then_some(n)
}

/// `--threads <n>` for local `llama-server` spawns, `Some` only when a
/// CPU limit binds. llama-server's own autodetection
/// (`cpu_get_num_math()`) already picks the physical/math cores, so an
/// unconstrained host passes nothing and leaves that choice alone. The
/// derived value only corrects the case autodetection cannot see: a
/// cgroup CPU quota, or a narrowed affinity mask, both carried by
/// `std::thread::available_parallelism` (std walks /proc/self/cgroup
/// and the ancestor chain itself, v1 and v2). A limit binds when
/// `available_parallelism` is below the online CPU count; then that
/// smaller value is passed. Accepted tradeoff: a quota between the
/// physical-core and SMT-thread counts (e.g. --cpus=12 on an
/// 8-core/16-thread host) passes 12 where autodetection would pick 8.
/// Any read or parse failure returns `None`: fail closed to
/// autodetection. `LLAMA_ARG_THREADS` set in the environment wins:
/// llama-server reads it itself via plain env inheritance, so `None`
/// here keeps that explicit choice untouched.
pub(super) fn threads_from_env_or_host() -> Option<u32> {
    if std::env::var_os("LLAMA_ARG_THREADS").is_some() {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let allowed = std::thread::available_parallelism().ok()?.get() as u32;
        let online = online_cpu_count()?;
        (allowed < online).then_some(allowed)
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// Online CPUs from /sys/devices/system/cpu/online, the baseline
/// `available_parallelism` is compared against to decide whether a
/// quota or affinity limit binds. `None` when the file is unreadable
/// or malformed.
#[cfg(target_os = "linux")]
fn online_cpu_count() -> Option<u32> {
    cpu_list_count(&std::fs::read_to_string("/sys/devices/system/cpu/online").ok()?)
}

/// CPU count from a kernel CPU list such as /sys/devices/system/cpu/online:
/// comma-separated single IDs or inclusive ranges (`0-15`, `0,4-7`).
/// `None` on empty or malformed content.
#[cfg(target_os = "linux")]
fn cpu_list_count(list: &str) -> Option<u32> {
    let mut count: u32 = 0;
    for part in list.trim().split(',') {
        match part.split_once('-') {
            Some((lo, hi)) => {
                let (lo, hi): (u32, u32) = (lo.trim().parse().ok()?, hi.trim().parse().ok()?);
                if lo > hi {
                    return None;
                }
                count = count.checked_add(hi.checked_sub(lo)?.checked_add(1)?)?;
            }
            None => {
                let _: u32 = part.trim().parse().ok()?;
                count = count.checked_add(1)?;
            }
        }
    }
    (count > 0).then_some(count)
}

/// The `--ctx-size` value to actually forward to llama-server: `ctx_size`
/// (the per-request context every other computation — retries, error
/// messages, `LLMMAN_CONTEXT_LENGTH` itself — is expressed in) times
/// `num_parallel`. llama-server splits one `--ctx-size` evenly across
/// every `--parallel` slot rather than giving each its own full amount,
/// so forwarding `ctx_size` unscaled would silently divide a request's
/// real context by `num_parallel`; Ollama avoids exactly this by
/// launching with `NumCtx * numParallel` (`llm/llama_server.go`).
/// Callers should only ever pass a non-`None` `num_parallel` alongside
/// a `Some` `ctx_size` — see `ensure_model`'s own `num_parallel`
/// fallback, which drops it to `None` otherwise (nothing safe to scale
/// against).
pub(super) fn backend_ctx_size(ctx_size: Option<u32>, num_parallel: Option<u32>) -> Option<u32> {
    ctx_size.map(|c| c.saturating_mul(num_parallel.unwrap_or(1)))
}

/// `num_parallel` unless `ctx_size` is `None` (a high-VRAM host
/// deferring to the model's own trained context, nothing safe to scale
/// against — see `backend_ctx_size`'s doc comment), in which case
/// `None`: forwarding `--parallel` unscaled would silently divide that
/// trained context across slots instead.
pub(super) fn effective_num_parallel(
    ctx_size: Option<u32>,
    num_parallel: Option<u32>,
) -> Option<u32> {
    ctx_size.and(num_parallel)
}

/// Matches Ollama's own default for `OLLAMA_MAX_QUEUE`.
const DEFAULT_MAX_QUEUE: usize = 512;

/// Maximum number of requests [`ensure_model`] admits at once before
/// rejecting with a 503, from `LLMMAN_MAX_QUEUE` (mirrors Ollama's
/// `OLLAMA_MAX_QUEUE`). See [`try_admit`].
pub(super) fn max_queue_from_env() -> usize {
    parse_max_queue(std::env::var("LLMMAN_MAX_QUEUE").ok().as_deref())
}

/// Unlike most other `parse_*` functions here, `0` is a real value (see
/// `try_admit_against`'s doc comment), not "unset".
fn parse_max_queue(value: Option<&str>) -> usize {
    match value.map(str::trim) {
        Some(v) if !v.is_empty() => v.parse().unwrap_or(DEFAULT_MAX_QUEUE),
        _ => DEFAULT_MAX_QUEUE,
    }
}

/// Whether to serve `GET /metrics` at all, from `LLMMAN_METRICS`. Off
/// unless the operator asked for it: the router has no authentication,
/// `LLMMAN_HOST` can bind it beyond loopback, and a scrape reports
/// version, route mix, model names and model churn. None of that should
/// start answering because llmman was upgraded.
pub(super) fn metrics_enabled_from_env() -> bool {
    parse_metrics_enabled(std::env::var("LLMMAN_METRICS").ok().as_deref())
}

/// [`metrics_enabled_from_env`]'s parsing, split out so it's testable
/// without mutating the real process environment. `1` is what the README
/// documents; the rest of the truthy set is here because every other
/// boolean env var in this file accepts it, and a `LLMMAN_METRICS=true`
/// that silently did nothing would be worse than the extra branch.
fn parse_metrics_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// Maximum number of models [`ensure_model`] keeps loaded at once, from
/// `LLMMAN_MAX_LOADED_MODELS` (mirrors Ollama's `OLLAMA_MAX_LOADED_MODELS`,
/// but as one flat daemon-wide total, not per-GPU — llmman has no
/// per-model memory estimate to size a per-GPU figure against). `0` =
/// unbounded. See [`enforce_max_loaded_models`].
pub(super) fn max_loaded_models_from_env() -> usize {
    parse_max_loaded_models(std::env::var("LLMMAN_MAX_LOADED_MODELS").ok().as_deref())
}

fn parse_max_loaded_models(value: Option<&str>) -> usize {
    value
        .map(str::trim)
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Whether `model_ref` gets `--context-shift`. Enabled except for
/// DeepSeek-family ("deepseek2" architecture) models, mirroring Ollama's
/// own `supportsContextShift` (`server/sched.go`) — their MLA-compressed
/// KV cache can't be shifted the way llama-server expects. Ollama reads
/// the GGUF architecture; this is a coarser name-based heuristic.
pub(super) fn supports_context_shift(model_ref: &str) -> bool {
    !model_ref.to_ascii_lowercase().contains("deepseek")
}

/// A GGUF's `{arch}.context_length` (`n_ctx_train`), if present and
/// non-zero.
pub(super) fn gguf_trained_ctx(info: &crate::gguf::Info) -> Option<u32> {
    info.context_length()
        .and_then(|n| u32::try_from(n).ok())
        .filter(|n| *n > 0)
}

/// `Some(trained context)` if the GGUF is an embedding model — one with
/// an `{arch}.pooling_type` key, ollama's own test (`llm/llama_server.go`)
/// — else `None`.
pub(super) fn embedding_model_ctx(info: &crate::gguf::Info) -> Option<Option<u32>> {
    let arch = info.architecture()?;
    info.u64(&format!("{arch}.pooling_type"))?;
    Some(gguf_trained_ctx(info))
}

/// The `--ctx-size` a load starts from: `configured` capped to `trained`
/// unless it's an `explicit` user value for a non-`embedding` model.
/// llama-server allocates the KV cache at `--ctx-size` and only caps
/// per-slot use to `n_ctx_train`, so an unclamped [`DEFAULT_CTX_SIZE`]
/// would reserve 256k of KV for a 32k model. 0/`None` mean `trained`.
pub(super) fn initial_ctx_size(
    configured: Option<u32>,
    explicit: bool,
    trained: Option<u32>,
    embedding: bool,
) -> Option<u32> {
    let Some(trained) = trained else {
        return configured;
    };
    if explicit && !embedding {
        return configured;
    }
    Some(
        configured
            .filter(|n| *n > 0)
            .map_or(trained, |n| n.min(trained)),
    )
}

// ---------------------------------------------------------------------------
// Out-of-memory auto-shrink retry (ensure_model) — mirrors Ollama's
// reduceAutoNumCtxForLoadOOM: a chosen --ctx-size can still be too big
// for actual free VRAM, so retry with it halved a few times instead of
// failing the load outright.
// ---------------------------------------------------------------------------

/// Max halving retries for an OOM-looking llama-server load.
pub(super) const MAX_CTX_SHRINK_ATTEMPTS: u32 = 4;

/// Floor below which a still-failing load is a hard failure, not
/// something to keep shrinking.
const MIN_CTX_SIZE_FOR_RETRY: u32 = 16384;

/// Next `--ctx-size` to retry an OOM'd load with, or `None` if shrinking
/// further wouldn't help (at/under the floor already).
pub(super) fn next_ctx_size_after_oom(current: u32) -> Option<u32> {
    let next = (current / 2).max(MIN_CTX_SIZE_FOR_RETRY);
    // `next < current`, not `!=`: below the floor, halving+max would
    // otherwise suggest a *larger* ctx-size after an OOM.
    (next < current).then_some(next)
}

/// True if `detail` (a failed load's stderr tail, or an error message)
/// looks like a memory-allocation failure rather than some other startup
/// error. Matched against known ggml/llama.cpp allocator log phrasings —
/// deliberately specific rather than one broad substring, since
/// misclassifying an unrelated failure as OOM would burn several slow
/// retries before surfacing the real error.
pub(super) fn looks_like_oom(detail: &str) -> bool {
    let d = detail.to_ascii_lowercase();
    [
        "failed to allocate",
        "out of memory",
        "not enough memory",
        "insufficient memory",
        "cudamalloc failed",
        "std::bad_alloc",
    ]
    .iter()
    .any(|needle| d.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_max_queue_defaults_to_ollamas_own_512_on_anything_unparseable() {
        assert_eq!(parse_max_queue(None), 512);
        assert_eq!(parse_max_queue(Some("")), 512);
        assert_eq!(parse_max_queue(Some("garbage")), 512);
        assert_eq!(parse_max_queue(Some("10")), 10);
        // Unlike most other LLMMAN_*/parse_* pairs, an explicit "0" is a
        // real value (disables the bound), not treated as unset.
        assert_eq!(parse_max_queue(Some("0")), 0);
    }

    #[test]
    fn parse_max_loaded_models_defaults_to_unbounded_on_anything_unparseable() {
        assert_eq!(parse_max_loaded_models(None), 0);
        assert_eq!(parse_max_loaded_models(Some("")), 0);
        assert_eq!(parse_max_loaded_models(Some("garbage")), 0);
        assert_eq!(parse_max_loaded_models(Some("3")), 3);
    }

    #[test]
    fn parse_metrics_enabled_accepts_the_truthy_spellings_and_nothing_else() {
        for on in ["1", "true", "TRUE", "yes", "on", " 1 "] {
            assert!(parse_metrics_enabled(Some(on)), "{on:?} should enable");
        }
        for off in ["0", "false", "no", "off", "", "  ", "2", "enabled"] {
            assert!(!parse_metrics_enabled(Some(off)), "{off:?} should not");
        }
        assert!(
            !parse_metrics_enabled(None),
            "unset is the default, and the default is off"
        );
    }

    #[test]
    fn parse_context_length_accepts_a_plain_number_and_rejects_everything_else() {
        assert_eq!(parse_context_length(Some("32768")), Some(32768));
        assert_eq!(parse_context_length(Some(" 32768 \n")), Some(32768));
        assert_eq!(parse_context_length(Some("0")), Some(0));
        assert_eq!(parse_context_length(None), None);
        assert_eq!(parse_context_length(Some("")), None);
        assert_eq!(parse_context_length(Some("not-a-number")), None);
        assert_eq!(parse_context_length(Some("-1")), None);
    }

    #[test]
    fn parse_flash_attention_accepts_llama_server_and_ollama_spellings() {
        // llama-server's own vocabulary passes straight through.
        assert_eq!(parse_flash_attention(Some("on")), Some("on".into()));
        assert_eq!(parse_flash_attention(Some("off")), Some("off".into()));
        assert_eq!(parse_flash_attention(Some("auto")), Some("auto".into()));
        // Ollama's OLLAMA_FLASH_ATTENTION boolean spelling is translated.
        assert_eq!(parse_flash_attention(Some("1")), Some("on".into()));
        assert_eq!(parse_flash_attention(Some("true")), Some("on".into()));
        assert_eq!(parse_flash_attention(Some("0")), Some("off".into()));
        assert_eq!(parse_flash_attention(Some("false")), Some("off".into()));
        // Case-insensitive, whitespace-tolerant.
        assert_eq!(parse_flash_attention(Some(" ON \n")), Some("on".into()));
        // Unset/empty leaves llama-server's own default untouched.
        assert_eq!(parse_flash_attention(None), None);
        assert_eq!(parse_flash_attention(Some("")), None);
        assert_eq!(parse_flash_attention(Some("   ")), None);
    }

    #[test]
    fn parse_kv_cache_type_trims_whitespace_and_treats_empty_as_unset() {
        assert_eq!(parse_kv_cache_type(Some("q8_0")), Some("q8_0".into()));
        assert_eq!(parse_kv_cache_type(Some(" q4_0 \n")), Some("q4_0".into()));
        assert_eq!(parse_kv_cache_type(None), None);
        assert_eq!(parse_kv_cache_type(Some("")), None);
        assert_eq!(parse_kv_cache_type(Some("   ")), None);
    }

    #[test]
    fn parse_sched_spread_maps_truthy_and_falsey_spellings_to_split_mode() {
        for truthy in ["1", "true", "yes", "on", "layer", " ON \n"] {
            assert_eq!(
                parse_sched_spread(Some(truthy)),
                Some("layer"),
                "input {truthy:?}"
            );
        }
        for falsey in ["0", "false", "no", "off", "none", " OFF \n"] {
            assert_eq!(
                parse_sched_spread(Some(falsey)),
                Some("none"),
                "input {falsey:?}"
            );
        }
    }

    #[test]
    fn parse_sched_spread_leaves_llama_servers_own_default_untouched_when_unset_or_unparseable() {
        assert_eq!(parse_sched_spread(None), None);
        assert_eq!(parse_sched_spread(Some("")), None);
        assert_eq!(parse_sched_spread(Some("   ")), None);
        assert_eq!(parse_sched_spread(Some("garbage")), None);
    }

    #[test]
    fn parse_num_parallel_accepts_a_positive_integer() {
        assert_eq!(parse_num_parallel(Some("4")), Some(4));
        assert_eq!(parse_num_parallel(Some(" 1 ")), Some(1));
    }

    #[test]
    fn parse_num_parallel_rejects_zero_and_unparseable_values() {
        assert_eq!(parse_num_parallel(Some("0")), None);
        assert_eq!(parse_num_parallel(None), None);
        assert_eq!(parse_num_parallel(Some("")), None);
        assert_eq!(parse_num_parallel(Some("-1")), None);
        assert_eq!(parse_num_parallel(Some("garbage")), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_list_count_counts_ids_and_inclusive_ranges() {
        // (kernel CPU list, expected count)
        let cases = [
            ("0-15\n", Some(16)),
            ("0", Some(1)),
            ("0,4-7\n", Some(5)),
            ("0-3,8-11\n", Some(8)),
            // Malformed or empty content fails closed: the caller then
            // passes no --threads and llama-server autodetects.
            ("", None),
            ("\n", None),
            ("3-1", None),
            ("0-x", None),
            ("a", None),
            ("0,,2", None),
            // Range length would overflow u32; no kernel emits this.
            ("0-4294967295", None),
            ("0-4294967295,4", None),
        ];
        for (list, expected) in &cases {
            assert_eq!(&cpu_list_count(list), expected, "list={list:?}");
        }
    }

    #[test]
    fn backend_ctx_size_scales_by_num_parallel_so_each_slot_keeps_the_full_context() {
        // llama-server splits one --ctx-size evenly across every
        // --parallel slot, so this must scale up to compensate —
        // matching Ollama's own NumCtx * numParallel.
        assert_eq!(backend_ctx_size(Some(4096), Some(4)), Some(16384));
        assert_eq!(backend_ctx_size(Some(4096), Some(1)), Some(4096));
        assert_eq!(backend_ctx_size(Some(4096), None), Some(4096));
        assert_eq!(backend_ctx_size(None, Some(4)), None);
        assert_eq!(backend_ctx_size(None, None), None);
    }

    #[test]
    fn backend_ctx_size_saturates_instead_of_overflowing() {
        assert_eq!(backend_ctx_size(Some(u32::MAX), Some(2)), Some(u32::MAX));
    }

    #[test]
    fn effective_num_parallel_drops_to_none_without_a_ctx_size_to_scale() {
        // Nothing to scale --parallel against, so don't forward it at
        // all rather than let llama-server silently divide the model's
        // own trained context across slots.
        assert_eq!(effective_num_parallel(None, Some(4)), None);
        assert_eq!(effective_num_parallel(Some(4096), Some(4)), Some(4));
        assert_eq!(effective_num_parallel(Some(4096), None), None);
        assert_eq!(effective_num_parallel(None, None), None);
    }

    /// The GGUF test (ollama's own): a `{arch}.pooling_type` key marks an
    /// embedding model; both kinds report their trained context.
    #[test]
    fn embedding_model_ctx_keys_off_pooling_type_and_reports_the_trained_context() {
        let chat = crate::gguf::write_test_gguf_with(&[]);
        let embed = crate::gguf::write_test_gguf_with(&[("llama.pooling_type", 1)]);
        let chat_info = crate::gguf::read_info(&chat).unwrap();
        let embed_info = crate::gguf::read_info(&embed).unwrap();
        std::fs::remove_file(&chat).ok();
        std::fs::remove_file(&embed).ok();
        assert_eq!(embedding_model_ctx(&chat_info), None);
        assert_eq!(embedding_model_ctx(&embed_info), Some(Some(4096)));
        assert_eq!(gguf_trained_ctx(&chat_info), Some(4096));
        assert_eq!(gguf_trained_ctx(&embed_info), Some(4096));
    }

    #[test]
    fn initial_ctx_size_caps_the_auto_default_at_the_trained_context() {
        let auto = Some(DEFAULT_CTX_SIZE);
        // Smaller trained context wins; larger leaves the default alone.
        assert_eq!(
            initial_ctx_size(auto, false, Some(32768), false),
            Some(32768)
        );
        assert_eq!(initial_ctx_size(auto, false, Some(1 << 20), false), auto);
        // No readable header: nothing to clamp against.
        assert_eq!(initial_ctx_size(auto, false, None, false), auto);
    }

    #[test]
    fn initial_ctx_size_forwards_an_explicit_value_unless_embedding() {
        // A user's LLMMAN_CONTEXT_LENGTH is theirs to get wrong; 0 stays 0.
        assert_eq!(
            initial_ctx_size(Some(65536), true, Some(4096), false),
            Some(65536)
        );
        assert_eq!(initial_ctx_size(Some(0), true, Some(4096), false), Some(0));
        // Embedding models are capped regardless, and 0/None mean trained.
        assert_eq!(
            initial_ctx_size(Some(65536), true, Some(4096), true),
            Some(4096)
        );
        assert_eq!(
            initial_ctx_size(Some(2048), true, Some(4096), true),
            Some(2048)
        );
        assert_eq!(
            initial_ctx_size(Some(0), false, Some(4096), true),
            Some(4096)
        );
        assert_eq!(initial_ctx_size(None, false, Some(4096), true), Some(4096));
    }

    #[test]
    fn supports_context_shift_disables_only_for_deepseek_family_models() {
        assert!(!supports_context_shift("deepseek-v3:latest"));
        assert!(!supports_context_shift("deepseek-r1:70b"));
        assert!(!supports_context_shift("DeepSeek-V2.5:latest")); // case-insensitive
        assert!(supports_context_shift("qwen3.5:latest"));
        assert!(supports_context_shift("gpt-oss:20b"));
    }

    #[test]
    fn next_ctx_size_after_oom_halves_from_the_default_down_to_the_floor() {
        assert_eq!(next_ctx_size_after_oom(DEFAULT_CTX_SIZE), Some(131072));
        assert_eq!(next_ctx_size_after_oom(131072), Some(65536));
        assert_eq!(next_ctx_size_after_oom(65536), Some(32768));
        assert_eq!(next_ctx_size_after_oom(32768), Some(16384));
        // At (or below) the floor, no further shrink is offered.
        assert_eq!(next_ctx_size_after_oom(16384), None);
        assert_eq!(next_ctx_size_after_oom(8192), None);
    }

    #[test]
    fn default_ctx_size_reaches_the_floor_within_the_shrink_budget() {
        // 262144 -> 131072 -> 65536 -> 32768 -> 16384: exactly
        // MAX_CTX_SHRINK_ATTEMPTS halvings, so the floor is reachable.
        let mut ctx = DEFAULT_CTX_SIZE;
        let mut attempts = 0;
        while let Some(next) = next_ctx_size_after_oom(ctx) {
            ctx = next;
            attempts += 1;
        }
        assert_eq!(ctx, MIN_CTX_SIZE_FOR_RETRY);
        assert!(attempts <= MAX_CTX_SHRINK_ATTEMPTS);
    }

    #[test]
    fn looks_like_oom_matches_known_allocation_failure_phrasings() {
        for msg in [
            "ggml_backend_alloc_ctx_tensors_from_buft: failed to allocate CUDA0 buffer of size 123",
            "llama_kv_cache: failed to allocate buffer for kv cache",
            "CUDA error: out of memory",
            "terminate called after throwing an instance of 'std::bad_alloc'",
            "cudaMalloc failed: out of memory",
        ] {
            assert!(looks_like_oom(msg), "expected OOM match for {msg:?}");
        }
    }

    #[test]
    fn looks_like_oom_does_not_flag_unrelated_startup_failures() {
        for msg in [
            "error while loading shared libraries: libcuda.so.1: cannot open shared object file",
            "error loading model: unknown architecture 'not-a-real-arch'",
            "error: unknown argument: --not-a-real-flag",
        ] {
            assert!(!looks_like_oom(msg), "unexpected OOM match for {msg:?}");
        }
    }
}

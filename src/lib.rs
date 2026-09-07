#![recursion_limit = "256"]

pub mod cmd;
pub mod config;
pub mod container;
pub mod daemon;
pub mod ffi;
pub mod fmt;
pub mod fsutil;
pub mod gguf;
pub mod harmony;
pub mod hf;
pub mod hostgpu;
pub mod hybrid;
pub mod imagegen;
pub mod llama_release;
pub mod mediagen;
pub mod metrics;
pub mod modelpack;
pub mod oauth;
pub mod promptlog;
pub mod providers;
pub mod shortnames;
pub mod sources;
pub mod storage;
pub mod thinking;
pub mod verify;
pub mod webui;
pub mod xet_fetch;

use std::path::PathBuf;

/// Path to the local OCI store, from `LLMMAN_MODELS` (mirrors Ollama's
/// `OLLAMA_MODELS`) or else `~/.local/share/llmman/store`
/// (`%LOCALAPPDATA%\llmman\store` on Windows).
pub fn default_store() -> anyhow::Result<PathBuf> {
    if let Some(dir) = models_dir_from_env() {
        return Ok(dir);
    }
    #[cfg(not(target_os = "windows"))]
    let base = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("could not determine home directory"))?
        .join(".local")
        .join("share");
    #[cfg(target_os = "windows")]
    let base = dirs::data_local_dir()
        .ok_or_else(|| anyhow::anyhow!("could not determine local data directory"))?;
    Ok(base.join("llmman").join("store"))
}

fn models_dir_from_env() -> Option<PathBuf> {
    parse_env_path(std::env::var("LLMMAN_MODELS").ok().as_deref())
}

/// Path to the extracted-model cache — a sibling of the store
/// (`<store>/../cache`), matching what `llmman serve` (the long-running
/// server that extracts most models) uses. The single canonical answer for
/// "where is the cache", so extraction and the GC sweep agree on it rather
/// than one command writing to `<store>/cache` and another cleaning
/// `<store>/../cache`.
pub fn default_cache() -> anyhow::Result<PathBuf> {
    let store = default_store()?;
    Ok(store.parent().unwrap_or(&store).join("cache"))
}

/// Whether verbose diagnostic logging is enabled, from `LLMMAN_DEBUG`
/// (mirrors Ollama's `OLLAMA_DEBUG`). llmman has only one verbosity
/// tier, so any truthy value or non-zero integer enables it (Ollama's
/// `OLLAMA_DEBUG=2` TRACE spelling also works, just maps to the same
/// tier).
pub fn debug_enabled() -> bool {
    parse_debug_enabled(std::env::var("LLMMAN_DEBUG").ok().as_deref())
}

fn parse_debug_enabled(value: Option<&str>) -> bool {
    let v = match value.map(str::trim) {
        Some(v) if !v.is_empty() => v,
        _ => return false,
    };
    match v.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        other => other.parse::<i64>().map(|n| n != 0).unwrap_or(false),
    }
}

/// Prints a `[llmman] [debug] ...` line to stderr, only when
/// [`debug_enabled`]. A macro (not a plain function) so the `format!`
/// args aren't even evaluated when debug logging is off.
#[macro_export]
macro_rules! debug_log {
    ($($arg:tt)*) => {
        if $crate::debug_enabled() {
            eprintln!("[llmman] [debug] {}", format!($($arg)*));
        }
    };
}

/// Split out from [`models_dir_from_env`] for testing without touching
/// the real environment. Blank values count as unset.
fn parse_env_path(value: Option<&str>) -> Option<PathBuf> {
    let trimmed = value?.trim();
    (!trimmed.is_empty()).then(|| PathBuf::from(trimmed))
}

/// Whether an opt-out variable (`LLMMAN_NOPRUNE`, `LLMMAN_NOHISTORY`) is
/// set: anything but unset, blank, or `0`/`false`/`no`/`off`.
pub fn env_flag_set(name: &str) -> bool {
    parse_env_flag(std::env::var(name).ok().as_deref())
}

fn parse_env_flag(value: Option<&str>) -> bool {
    let Some(v) = value else { return false };
    let v = v.trim();
    if v.is_empty() {
        return false;
    }
    !matches!(
        v.to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_env_path_returns_none_when_unset_or_blank() {
        assert_eq!(parse_env_path(None), None);
        assert_eq!(parse_env_path(Some("")), None);
        assert_eq!(parse_env_path(Some("   ")), None);
    }

    #[test]
    fn parse_env_path_trims_and_returns_the_given_path() {
        assert_eq!(
            parse_env_path(Some("  /custom/store  ")),
            Some(PathBuf::from("/custom/store"))
        );
    }

    #[test]
    fn parse_debug_enabled_recognizes_boolean_spellings() {
        assert!(!parse_debug_enabled(None));
        assert!(!parse_debug_enabled(Some("")));
        assert!(!parse_debug_enabled(Some("0")));
        assert!(!parse_debug_enabled(Some("false")));
        assert!(!parse_debug_enabled(Some("no")));
        assert!(!parse_debug_enabled(Some("off")));
        assert!(parse_debug_enabled(Some("1")));
        assert!(parse_debug_enabled(Some("true")));
        assert!(parse_debug_enabled(Some("YES")));
        assert!(parse_debug_enabled(Some("on")));
    }

    #[test]
    fn parse_debug_enabled_treats_any_nonzero_integer_as_enabled() {
        assert!(parse_debug_enabled(Some("2")));
        assert!(parse_debug_enabled(Some("-1")));
        assert!(!parse_debug_enabled(Some("not-a-number")));
    }

    #[test]
    fn parse_env_flag_is_false_when_unset_blank_or_explicitly_falsy() {
        assert!(!parse_env_flag(None));
        assert!(!parse_env_flag(Some("")));
        assert!(!parse_env_flag(Some("   ")));
        for falsy in ["0", "false", "no", "off", "FALSE", "Off", "  no  "] {
            assert!(!parse_env_flag(Some(falsy)), "{falsy:?} should be falsy");
        }
    }

    #[test]
    fn parse_env_flag_is_true_for_any_other_value() {
        for truthy in ["1", "true", "yes", "on", "anything"] {
            assert!(parse_env_flag(Some(truthy)), "{truthy:?} should be truthy");
        }
    }
}

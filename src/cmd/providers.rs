//! `llmman providers` — the providers `--provider` accepts.
//!
//! Answered by `llmman serve` (`GET /llmman/providers`), not by fetching
//! models.dev here: the daemon forwards the request, so its catalog
//! decides whether `--provider x` works and its environment holds the key
//! that gets spent (see `resolve_remote_target` in cmd::serve).

use clap::Args;

use crate::daemon::{self, ProviderSummary};

#[derive(Args, Debug)]
pub struct ProvidersArgs {
    /// Only show providers whose id or name contains this substring
    #[arg(value_name = "FILTER")]
    pub filter: Option<String>,
}

pub fn run(args: &ProvidersArgs) -> anyhow::Result<()> {
    // Same contract as `run`/`pull`/`launch`: start the daemon rather
    // than tell the user to. It owns the catalog, and whatever runs next
    // needs it anyway.
    daemon::ensure_server("")?;

    let all = daemon::providers()?;
    let total = all.len();
    let filter = args
        .filter
        .as_deref()
        .map(str::trim)
        .filter(|f| !f.is_empty())
        .map(str::to_lowercase);
    let shown: Vec<&ProviderSummary> = all
        .iter()
        .filter(|p| matches(p, filter.as_deref()))
        .collect();

    if shown.is_empty() {
        // A header with no rows would read as "there are none".
        anyhow::bail!(
            "no provider matches {:?} — run 'llmman providers' for all {total} of them",
            args.filter.as_deref().unwrap_or_default()
        );
    }

    let id_w = shown.iter().map(|p| p.id.len()).max().unwrap_or(8).max(8);
    let name_w = shown.iter().map(|p| p.name.len()).max().unwrap_or(4).max(4);
    let key_w = shown
        .iter()
        .map(|p| key_env(p).len())
        .max()
        .unwrap_or(7)
        .max(7);

    println!(
        "{:<id_w$}    {:<name_w$}    {:<key_w$}    {:<14}    MODELS",
        "PROVIDER",
        "NAME",
        "API KEY",
        "KEY",
        id_w = id_w,
        name_w = name_w,
        key_w = key_w,
    );
    for p in &shown {
        println!(
            "{:<id_w$}    {:<name_w$}    {:<key_w$}    {:<14}    {}",
            p.id,
            p.name,
            key_env(p),
            key_status(p),
            model_count(p),
            id_w = id_w,
            name_w = name_w,
            key_w = key_w,
        );
    }

    // Nothing after the last row: a trailing count and usage block is
    // something to skip past every time, and something a pipe into
    // `grep`/`awk` has to filter out. `--help` is where usage belongs.
    Ok(())
}

fn matches(provider: &ProviderSummary, needle: Option<&str>) -> bool {
    let Some(needle) = needle else { return true };
    provider.id.to_lowercase().contains(needle) || provider.name.to_lowercase().contains(needle)
}

/// The variable column: `-` for a configured provider that names none.
fn key_env(provider: &ProviderSummary) -> &str {
    provider.key_env.as_deref().unwrap_or("-")
}

/// The models column: `-` for a configured provider, whose endpoint is
/// only asked by `llmman list --provider <id>`; `0` would read as
/// "serves nothing".
fn model_count(provider: &ProviderSummary) -> String {
    if provider.key_optional && provider.models == 0 {
        "-".to_string()
    } else {
        provider.models.to_string()
    }
}

/// Where a *usable* key is — the one thing to act on before `--provider`
/// works.
///
/// "client" is a key only this process has, which travels per request —
/// from its environment or its own `llmman.conf`, which is why the word
/// is not "shell". "withheld" is one the daemon has but will not spend,
/// being bound where others could reach it (see `resolve_remote_target`
/// in cmd::serve) — a state of its own, since what needs fixing there is
/// the bind, not the key. "none needed" is a keyless provider: nothing
/// to act on.
fn key_status(provider: &ProviderSummary) -> &'static str {
    match (provider.key_usable, provider.key_here(), provider.key_set) {
        (true, _, _) => "set",
        (false, true, _) => "set (client)",
        (false, false, true) => "set (withheld)",
        (false, false, false) if provider.key_optional => "none needed",
        (false, false, false) => "unset",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(id: &str, name: &str) -> ProviderSummary {
        ProviderSummary {
            id: id.to_string(),
            name: name.to_string(),
            key_env: Some("LLMMAN_TEST_PROVIDER_KEY_UNSET".to_string()),
            key_set: false,
            key_usable: false,
            key_optional: false,
            models: 0,
        }
    }

    /// Findable by the name a user knows, not only by its id.
    #[test]
    fn filter_matches_id_or_name_case_insensitively() {
        let p = summary("togetherai", "Together AI");
        assert!(matches(&p, None));
        assert!(matches(&p, Some("together")));
        assert!(matches(&p, Some("ai")));
        assert!(!matches(&p, Some("groq")));

        // The needle is lowercased by `run` before it gets here; the
        // provider's own casing must not matter either way.
        let p = summary("openai", "OpenAI");
        assert!(matches(&p, Some("openai")));
    }

    /// Each way a key can be present is a different thing to do about
    /// it, so none may collapse into another.
    #[test]
    fn key_status_reports_the_key_that_would_actually_be_used() {
        let mut p = summary("openai", "OpenAI");
        assert_eq!(key_status(&p), "unset");
        // Held by the daemon but withheld: the bind needs fixing, not
        // the variable.
        p.key_set = true;
        assert_eq!(key_status(&p), "set (withheld)");
        p.key_usable = true;
        assert_eq!(key_status(&p), "set");

        // A provider that takes no key is not "unset": there is nothing
        // to set. One that has a key anyway reports it as any other.
        let mut p = summary("gpubox", "GPU box");
        p.key_env = None;
        p.key_optional = true;
        assert_eq!(key_status(&p), "none needed");
        assert_eq!(key_env(&p), "-");
        assert_eq!(model_count(&p), "-");
        assert_eq!(model_count(&summary("openai", "OpenAI")), "0");
        p.key_set = true;
        p.key_usable = true;
        assert_eq!(key_status(&p), "set");
    }
}

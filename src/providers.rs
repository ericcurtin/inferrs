//! The models.dev provider catalog, fetched at runtime rather than baked
//! into this binary, so a provider models.dev gains tomorrow works
//! without an llmman release.
//!
//! opencode resolves the same catalog the same way, through its own
//! mirror of it (`packages/core/src/models-dev.ts` GETs
//! `models.opencode.ai/api.json`, byte-identical to `models.dev`'s own
//! today). llmman reads the upstream directly: a mirror is one more
//! thing that can disagree, and llmman has no say in what opencode's
//! serves.
//!
//! Only a *subset* is exposed as [`Provider`]s, because `llmman serve`
//! reaches an upstream exactly two ways, one per [`Wire`]: an HTTPS POST
//! of an OpenAI Chat Completions body with `Authorization: Bearer <key>`,
//! or an HTTPS POST of an Anthropic Messages body with `x-api-key: <key>`
//! (see `Target::Remote` in `cmd::serve`). An entry is offered only if
//! llmman can be *sure* it speaks one of those:
//!
//! * its `npm` driver is one of [`OPENAI_COMPATIBLE_NPM`], or it is a
//!   [`BUILTIN_ENDPOINTS`] entry vetted by hand (the one way onto the
//!   Anthropic wire, since `@ai-sdk/anthropic` names a body format, not
//!   an auth scheme), as opposed to `@ai-sdk/amazon-bedrock` (SigV4
//!   signing), `@ai-sdk/google-vertex` (GCP credentials), and the rest;
//! * it has one concrete `https` base URL. models.dev leaves `api` unset
//!   where the SDK hardcodes the endpoint — recovered from
//!   [`BUILTIN_ENDPOINTS`] — and templated (`${VAR}`) where it is
//!   per-account, which llmman has nothing to interpolate from;
//! * exactly one of its `env` entries is an API key. Multi-variable auth
//!   (`CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_KEY`, ...) is not "one
//!   API key".
//!
//! Everything filtered out is deliberately *absent* rather than
//! half-supported: a provider llmman offers is one it can actually reach.
//!
//! The one way past the filter is `llmman.conf`: a `[providers.<id>]`
//! with a `base_url` defines a provider by hand — an inference server on
//! some host models.dev has never heard of — and is added to (or shadows)
//! the catalog in [`catalog`]. Those rules exist to vet a list fetched
//! from the network; a URL the user wrote into their own file needs no
//! vetting beyond parsing, so a defined provider may be plain `http`,
//! and may take no key at all. See [`Provider::key_optional`].

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use anyhow::Context;
use serde::Deserialize;

/// Where the catalog is fetched from.
const CATALOG_URL: &str = "https://models.dev/api.json";

/// Cache filename under [`crate::default_cache`].
const CACHE_FILE: &str = "models-dev.json";

/// How long a cached catalog is served without re-fetching. Long,
/// deliberately: the provider list moves in days, while a fetch on every
/// `llmman launch` would put a network round-trip in front of an
/// otherwise instant command.
const CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// Bounded so a hung models.dev can't wedge `llmman launch`; a failed
/// fetch falls back to any cached copy, however stale.
const FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// The AI SDK driver packages that speak the OpenAI wire format, which
/// [`Wire::OpenAi`] drives. See this module's own doc comment.
const OPENAI_COMPATIBLE_NPM: &[&str] = &[
    "@ai-sdk/openai-compatible",
    "@ai-sdk/openai",
    "@openrouter/ai-sdk-provider",
];

/// The wire format `llmman serve` speaks to a provider: the route, the
/// credential header, and whether a request is translated on the way.
///
/// `Deserialize` for the `wire = "openai"` field of a provider defined in
/// `llmman.conf` (see [`crate::config`]), spelled as [`Wire::as_str`]
/// reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Wire {
    /// OpenAI Chat Completions, `Authorization: Bearer <key>`.
    OpenAi,
    /// Anthropic Messages, `x-api-key: <key>` + `anthropic-version`.
    Anthropic,
}

impl Wire {
    /// The lowercase name the daemon's own API reports.
    pub fn as_str(self) -> &'static str {
        match self {
            Wire::OpenAi => "openai",
            Wire::Anthropic => "anthropic",
        }
    }
}

/// A provider whose endpoint models.dev leaves unset because its AI SDK
/// package hardcodes it, plus the one `env` entry that is its API key.
struct Builtin {
    /// models.dev provider id.
    id: &'static str,
    /// Base URL that the wire's route (`/chat/completions`, `/messages`)
    /// is appended to.
    base_url: &'static str,
    /// The API-key variable, picked out of the provider's `env` list.
    key_env: &'static str,
    /// What is spoken at `base_url`.
    wire: Wire,
}

/// Endpoints for providers models.dev has no `api` for, restricted to
/// ones with a published, stable endpoint that was checked by hand to
/// answer its wire's route at the URL below.
///
/// Not a second registry: an entry here must still be present in the
/// fetched catalog to be offered, and supplies only the one field
/// models.dev omits. The rest (`amazon-bedrock`, `azure`,
/// `google-vertex`, `watsonx`, ...) are left out because their auth is
/// not an API key, their endpoint is per-account, or their wire format
/// is neither of [`Wire`]'s. `v0` is left out for a third reason: nothing
/// answers at its documented `https://api.v0.dev/v1`, so llmman has no
/// endpoint it can vouch for.
const BUILTIN_ENDPOINTS: &[Builtin] = &[
    Builtin {
        id: "openai",
        base_url: "https://api.openai.com/v1",
        key_env: "OPENAI_API_KEY",
        wire: Wire::OpenAi,
    },
    // Its own Messages API, never the OpenAI-compatibility shim, which
    // has no thinking, cache control or beta headers.
    Builtin {
        id: "anthropic",
        base_url: "https://api.anthropic.com/v1",
        key_env: "ANTHROPIC_API_KEY",
        wire: Wire::Anthropic,
    },
    // Gemini's OpenAI-compatibility endpoint, not the native
    // generateContent one. GEMINI_API_KEY of the three variables
    // models.dev lists, matching what the Gemini docs use.
    Builtin {
        id: "google",
        base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
        key_env: "GEMINI_API_KEY",
        wire: Wire::OpenAi,
    },
    Builtin {
        id: "groq",
        base_url: "https://api.groq.com/openai/v1",
        key_env: "GROQ_API_KEY",
        wire: Wire::OpenAi,
    },
    Builtin {
        id: "mistral",
        base_url: "https://api.mistral.ai/v1",
        key_env: "MISTRAL_API_KEY",
        wire: Wire::OpenAi,
    },
    Builtin {
        id: "xai",
        base_url: "https://api.x.ai/v1",
        key_env: "XAI_API_KEY",
        wire: Wire::OpenAi,
    },
    Builtin {
        id: "cerebras",
        base_url: "https://api.cerebras.ai/v1",
        key_env: "CEREBRAS_API_KEY",
        wire: Wire::OpenAi,
    },
    Builtin {
        id: "togetherai",
        base_url: "https://api.together.xyz/v1",
        key_env: "TOGETHER_API_KEY",
        wire: Wire::OpenAi,
    },
    Builtin {
        id: "deepinfra",
        base_url: "https://api.deepinfra.com/v1/openai",
        key_env: "DEEPINFRA_API_KEY",
        wire: Wire::OpenAi,
    },
    // No `/v1`: Perplexity serves /chat/completions off the bare host.
    Builtin {
        id: "perplexity",
        base_url: "https://api.perplexity.ai",
        key_env: "PERPLEXITY_API_KEY",
        wire: Wire::OpenAi,
    },
    Builtin {
        id: "cohere",
        base_url: "https://api.cohere.ai/compatibility/v1",
        key_env: "COHERE_API_KEY",
        wire: Wire::OpenAi,
    },
    Builtin {
        id: "venice",
        base_url: "https://api.venice.ai/api/v1",
        key_env: "VENICE_API_KEY",
        wire: Wire::OpenAi,
    },
    Builtin {
        id: "vercel",
        base_url: "https://ai-gateway.vercel.sh/v1",
        key_env: "AI_GATEWAY_API_KEY",
        wire: Wire::OpenAi,
    },
];

// ---------------------------------------------------------------------------
// Remote model references
// ---------------------------------------------------------------------------

/// Namespace every provider-routed model reference is carried under.
///
/// Such a request reaches `llmman serve` through the very same `"model"`
/// field a local one does — the only field the Ollama, OpenAI, and
/// Anthropic surfaces have in common, and one the launched integrations
/// pass through opaquely.
///
/// The bare `<provider>/<model>` spelling opencode uses would be
/// ambiguous here in a way it never is there: llmman also resolves
/// two-segment names as HuggingFace repositories (see
/// `shortnames::resolve`), so routing on a bare prefix would silently
/// capture `google/gemma-3`, `openai/gpt-oss-120b` and every other real
/// `hf.co/<org>/<repo>` whose org shares a name with a provider. This
/// prefix cannot collide with one: `shortnames` only treats a first
/// segment as a registry host when it contains a dot.
pub const REMOTE_PREFIX: &str = "llmman.provider/";

/// The stand-in `cmd::launch` gives integrations pointed at a
/// locally-served model, where an API key is meaningless but omitting it
/// makes several of them refuse to start. It is not a credential, so
/// `cmd::serve` must never forward it to a real provider.
pub const PLACEHOLDER_API_KEY: &str = "llmman";

/// Encodes `provider` + `model` into the single reference that travels in
/// a request's `"model"` field. Inverse of [`split_remote_ref`].
pub fn format_remote_ref(provider: &str, model: &str) -> String {
    format!("{REMOTE_PREFIX}{provider}/{model}")
}

/// Splits a [`REMOTE_PREFIX`]-namespaced reference back into its provider
/// id and upstream model id, or `None` for any ordinary local reference.
///
/// Purely syntactic — it does not consult the catalog, so callers still
/// have to look the provider id up (and report an unknown one) themselves.
/// The model id keeps any further slashes: `openrouter`'s ids are
/// themselves `<vendor>/<model>`.
pub fn split_remote_ref(reference: &str) -> Option<(&str, &str)> {
    let rest = reference.strip_prefix(REMOTE_PREFIX)?;
    let (provider, model) = rest.split_once('/')?;
    (!provider.is_empty() && !model.is_empty()).then_some((provider, model))
}

/// Whether `reference` names a provider-routed model rather than a local
/// one. Cheaper than [`split_remote_ref`] when the parts aren't needed.
pub fn is_remote_ref(reference: &str) -> bool {
    split_remote_ref(reference).is_some()
}

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

/// One reachable provider: everything `llmman serve` needs to forward a
/// request upstream, and everything `llmman launch` needs to explain how
/// to authenticate to it.
#[derive(Clone, Debug, PartialEq)]
pub struct Provider {
    /// models.dev provider id, as passed to `--provider`.
    pub id: String,
    /// Human-readable name, for listings and error messages.
    pub name: String,
    /// Base URL that the wire's route (`/chat/completions`, `/messages`,
    /// ...) is appended to. Never has a trailing slash.
    pub base_url: String,
    /// Environment variable holding this provider's API key. Always set
    /// for a catalog provider; a configured one may name none, and then
    /// only `llmman.conf` can hold its key.
    pub key_env: Option<String>,
    /// What is spoken at `base_url`.
    pub wire: Wire,
    /// Whether a request may go upstream with no credential at all.
    /// True for a provider defined in `llmman.conf` — a vLLM or
    /// llama-server on the LAN usually takes none — and never for a
    /// catalog one, where a keyless request is a certain 401 better
    /// reported before the handoff than inside an integration.
    pub key_optional: bool,
    /// Models this provider serves, sorted by id. Used to validate
    /// `--model`, to suggest values, and to answer `llmman list
    /// --provider`; the id is never sent upstream verbatim. Empty for a
    /// configured provider, whose endpoint is asked instead (see
    /// `cmd::serve`).
    pub models: Vec<Model>,
}

/// One model a provider serves.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// The model id as the *provider* knows it.
    pub id: String,
    /// `None` where models.dev publishes no price, which is not the same
    /// as free and must not be rendered as one.
    pub cost: Option<Cost>,
    /// models.dev's `limit.output`, the default `max_tokens` for a wire
    /// that requires one. `None` where the catalog has none.
    pub max_output: Option<u32>,
}

/// US dollars per million tokens — models.dev's own unit, unconverted so
/// a printed figure matches the provider's pricing page.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    pub input: f64,
    pub output: f64,
}

impl Provider {
    /// This provider's API key, or `None` when neither the environment
    /// nor `llmman.conf` has one. See [`key_for`].
    pub fn api_key(&self) -> Option<String> {
        key_for(&self.id, self.key_env.as_deref())
    }

    /// Appends an OpenAI route to this provider's base URL. See
    /// [`rebase_url`].
    pub fn url(&self, route: &str) -> String {
        rebase_url(&self.base_url, route)
    }

    /// A provider as `llmman.conf` defines it. No models: the file names
    /// an endpoint, not what it serves, and `cmd::serve` asks the
    /// endpoint itself when a caller wants the list.
    pub fn from_configured(conf: &crate::config::ConfiguredProvider) -> Self {
        Self {
            id: conf.id.clone(),
            name: conf.name.clone(),
            base_url: conf.base_url.clone(),
            key_env: conf.key_env.clone(),
            wire: conf.wire,
            key_optional: true,
            models: Vec::new(),
        }
    }
}

/// The API key for provider `id`: `var` from the environment, else the
/// `[providers.<id>]` entry in `llmman.conf` (see [`crate::config`]).
/// `None` when neither has one — or when there is no variable to read
/// and the file has nothing, the case for most configured providers.
///
/// The environment wins, as it does for `aws` and `gh`: the file is the
/// standing answer, an `export` the deliberate this-session-only
/// override. The reverse would make `OPENAI_API_KEY=... llmman run`
/// silently spend the wrong key.
///
/// Free-standing because a `/llmman/providers` client learns only the id
/// and the variable's *name* from the daemon, never a [`Provider`].
pub fn key_for(id: &str, var: Option<&str>) -> Option<String> {
    resolve_key(
        var.and_then(key_from_env),
        crate::config::provider_api_key(id),
    )
}

/// Split out so the precedence is testable without touching the process
/// environment or the filesystem.
fn resolve_key(from_env: Option<String>, from_conf: Option<&str>) -> Option<String> {
    from_env.or_else(|| {
        from_conf
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty())
    })
}

/// An API key read out of `var`, or `None` when it is unset or blank.
fn key_from_env(var: &str) -> Option<String> {
    std::env::var(var)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Both places llmman looks, named, for an error that fires because it
/// found neither. One string so every such message agrees: a user told
/// only about the variable would never learn the file exists.
pub fn key_hint(id: &str, var: Option<&str>) -> String {
    match var {
        Some(var) => format!(
            "set {var} in the environment, or add a [providers.{}] api_key to {}",
            toml_key(id),
            crate::config::user_path_display()
        ),
        // A configured provider that names no variable: the file is the
        // one place, so there is no "or".
        None => format!(
            "add an api_key to [providers.{}] in {}",
            toml_key(id),
            crate::config::user_path_display()
        ),
    }
}

/// `id` as a TOML key, quoted when it is not a bare one.
///
/// models.dev has ids with a dot in them (`wafer.ai`), and
/// `[providers.wafer.ai]` is two nested tables, not the key `wafer.ai` —
/// so an unquoted hint tells the user to write something that silently
/// does not configure the provider they asked about.
fn toml_key(id: &str) -> String {
    let bare = !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if bare {
        id.to_string()
    } else {
        format!("{id:?}")
    }
}

/// The provider id `--provider` names: `None` when the flag is absent,
/// an error when it is present but blank.
///
/// Blank is not absent. `--provider "$PROVIDER"` with the variable unset
/// would otherwise silently run, list or launch *locally* — pulling
/// weights, or reading the local store — when what was asked for was a
/// hosted model.
pub fn provider_flag(value: Option<&str>) -> anyhow::Result<Option<&str>> {
    match value.map(str::trim) {
        Some("") => anyhow::bail!(
            "--provider needs a provider id — run 'llmman providers' for the ones llmman \
             can route to"
        ),
        other => Ok(other),
    }
}

/// A few real model ids, to make "which models?" answerable without
/// leaving the error message. Ids, not a [`Provider`]: callers hold a
/// wire shape from the daemon (see `cmd::providers`).
pub fn example_models(name: &str, ids: &[&str]) -> String {
    if ids.is_empty() {
        return format!("{name} lists no models");
    }
    let shown: Vec<&str> = ids.iter().take(5).copied().collect();
    let more = ids.len().saturating_sub(shown.len());
    let suffix = if more > 0 {
        format!(", … ({more} more)")
    } else {
        String::new()
    };
    format!("{name} models include: {}{suffix}", shown.join(", "))
}

/// Suggests near-matches before falling back to `llmman providers`, so a
/// typo is one line from the right answer rather than a 180-entry
/// listing. Next to the catalog because `llmman serve` owns it, and so is
/// what reports an unknown id — to `/llmman/providers/:id` callers and to
/// a request routed at a provider that does not exist alike.
pub fn unknown_provider_error(provider: &str, catalog: &Catalog) -> anyhow::Error {
    let close = suggestions(provider, catalog);
    if close.is_empty() {
        anyhow::anyhow!(
            "unknown provider {provider:?}\nRun 'llmman providers' for the {} providers \
             llmman can route to.",
            catalog.len()
        )
    } else {
        anyhow::anyhow!(
            "unknown provider {provider:?}\nDid you mean: {}?\nRun 'llmman providers' for \
             all {} of them.",
            close.join(", "),
            catalog.len()
        )
    }
}

/// Catalog ids closest to `provider`, or nothing when it is too long to
/// be one: an id can arrive in a request body (see
/// `resolve_remote_target` in cmd::serve), and both passes below scan the
/// whole catalog against it — the second at O(needle × id) a candidate.
/// Neither could match a needle that long anyway, so the guard costs no
/// suggestion anyone would have wanted.
fn suggestions<'a>(provider: &str, catalog: &'a Catalog) -> Vec<&'a str> {
    let needle = provider.to_lowercase();
    if needle.len() > MAX_SUGGESTION_LEN {
        return Vec::new();
    }
    // A shortened or padded id first ("together" for `togetherai`), then
    // — no substring match catches a slip like "togethr" — the nearest
    // ids, allowing about one edit per three characters.
    let close: Vec<&str> = catalog
        .ids()
        .filter(|id| id.contains(&needle) || needle.contains(*id))
        .take(10)
        .collect();
    if !close.is_empty() {
        return close;
    }
    let mut ranked: Vec<(usize, &str)> = catalog
        .ids()
        .map(|id| (edit_distance(&needle, id), id))
        .filter(|(d, id)| d * 3 <= needle.len().max(id.len()))
        .collect();
    ranked.sort_unstable();
    ranked.into_iter().take(5).map(|(_, id)| id).collect()
}

/// Longest id [`suggestions`] will look for a near-match to. A real one
/// is ~30 characters; a request body allows megabytes.
const MAX_SUGGESTION_LEN: usize = 64;

/// Levenshtein distance, two rows at a time. Run over ~180 short ids,
/// and only once no substring matched.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != *cb))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Re-bases one of llmman's own internal routes (`/v1/chat/completions`)
/// onto a provider base URL.
///
/// The `/v1` is dropped because a models.dev `api` already includes
/// whatever version segment the provider uses, which is not always `v1`
/// (`.../v1beta/openai` for Gemini, `.../compatibility/v1` for Cohere,
/// none at all for Perplexity).
pub fn rebase_url(base_url: &str, route: &str) -> String {
    let suffix = route.trim_start_matches('/');
    let suffix = suffix.strip_prefix("v1/").unwrap_or(suffix);
    format!("{base_url}/{suffix}")
}

/// Every provider llmman can route to, keyed by id.
#[derive(Debug, Default)]
pub struct Catalog {
    providers: BTreeMap<String, Provider>,
}

impl Catalog {
    /// The provider `id` names, or `None` if models.dev has no such
    /// provider or llmman cannot reach it (see the module doc comment).
    pub fn get(&self, id: &str) -> Option<&Provider> {
        self.providers.get(id)
    }

    /// Every routable provider, ordered by id.
    pub fn iter(&self) -> impl Iterator<Item = &Provider> {
        self.providers.values()
    }

    /// Every routable provider id, ordered.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.providers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Parses a models.dev `api.json` body, keeping only the entries this
    /// module's doc comment describes as reachable.
    ///
    /// Each entry is deserialized on its own, so one upstream record that
    /// grows an unexpected shape costs that provider rather than the
    /// whole catalog — and `--provider` keeps working until llmman
    /// catches up. Split out from the fetch/cache plumbing so the
    /// filtering rules are unit-testable with no network or disk at all.
    pub fn from_json(raw: &[u8]) -> anyhow::Result<Self> {
        let raw: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(raw).context("parse models.dev api.json")?;
        let providers: BTreeMap<_, _> = raw
            .into_iter()
            .filter_map(|(id, provider)| {
                let provider = serde_json::from_value(provider).ok()?;
                Some((id.clone(), routable(&id, provider)?))
            })
            .collect();
        // An empty result is a failed load, not a catalog with nothing in
        // it: `{"error":"temporarily unavailable"}` is valid JSON and a
        // valid map, and every entry drops out of the filter. Treating it
        // as success would overwrite a good cache with nothing and
        // memoize it, so `--provider` would stay broken for a day.
        anyhow::ensure!(
            !providers.is_empty(),
            "models.dev returned no provider llmman can route to"
        );
        Ok(Self { providers })
    }

    /// The providers `llmman.conf` defines, and nothing else — what a
    /// machine that cannot reach models.dev still has (see [`catalog`]).
    pub fn from_configured(configured: &[crate::config::ConfiguredProvider]) -> Self {
        Self::default().with_configured(configured)
    }

    /// Adds the providers `llmman.conf` defines. One with a catalog id
    /// shadows the catalog entry: the file is the user's deliberate
    /// answer, and is also the only way to correct a catalog URL for a
    /// network where it is wrong (a proxy in front of `api.openai.com`).
    ///
    /// A plaintext `base_url` with a key behind it is warned about here,
    /// once per load, rather than refused as a catalog entry's would be:
    /// the URL is the user's own, and `http://gpubox:8000` on a LAN is
    /// the whole point — but a key on that wire is still a key on that
    /// wire, and the file cannot say which the user weighed.
    pub fn with_configured(mut self, configured: &[crate::config::ConfiguredProvider]) -> Self {
        for conf in configured {
            let provider = Provider::from_configured(conf);
            if provider.base_url.starts_with("http://") && provider.api_key().is_some() {
                eprintln!(
                    "[llmman] warning: provider {} has an API key and a plain-http base_url \
                     ({}); the key will cross the network in cleartext",
                    provider.id, provider.base_url
                );
            }
            if self
                .providers
                .insert(provider.id.clone(), provider)
                .is_some()
            {
                crate::debug_log!(
                    "provider catalog: {} from llmman.conf shadows the models.dev entry",
                    conf.id
                );
            }
        }
        self
    }
}

/// A models.dev catalog entry, narrowed to the fields llmman reads —
/// `api.json` is megabytes of per-model limit/modality metadata nothing
/// here reads.
#[derive(Debug, Deserialize)]
struct RawProvider {
    #[serde(default)]
    name: String,
    #[serde(default)]
    api: Option<String>,
    #[serde(default)]
    npm: Option<String>,
    #[serde(default)]
    env: Vec<String>,
    #[serde(default)]
    models: BTreeMap<String, RawModel>,
}

/// A models.dev model entry — the id is the map key, so only the price
/// and the output ceiling are read out of the value.
#[derive(Debug, Deserialize)]
struct RawModel {
    /// Untyped on purpose: a typed `{input, output}` would let an
    /// upstream `cost` that grew a new shape drop the whole *provider*,
    /// and a listing column is not worth `--provider x` breaking over.
    #[serde(default)]
    cost: Option<serde_json::Value>,
    /// Untyped for the same reason; only `output` is read.
    #[serde(default)]
    limit: Option<serde_json::Value>,
}

/// A models.dev `limit.output` as a token count, or `None` unless it is
/// a positive whole number that fits.
fn max_output_of(raw: &serde_json::Value) -> Option<u32> {
    let n = raw.get("output")?.as_u64()?;
    u32::try_from(n).ok().filter(|&n| n > 0)
}

/// A models.dev `cost` object as a [`Cost`], or `None` unless *both*
/// figures are there and sane — half a price misleads worse than none.
fn cost_of(raw: &serde_json::Value) -> Option<Cost> {
    let field = |name: &str| -> Option<f64> {
        raw.get(name)?
            .as_f64()
            .filter(|v| v.is_finite() && *v >= 0.0)
    };
    Some(Cost {
        input: field("input")?,
        output: field("output")?,
    })
}

/// Providers that pass every rule below but still cannot be driven with
/// the variable models.dev names, because the real credential is minted
/// by an OAuth exchange rather than exported by hand.
///
/// `github-copilot` is the case: it advertises `@ai-sdk/openai-compatible`
/// and `GITHUB_TOKEN`, but `api.githubcopilot.com` 401s a `GITHUB_TOKEN`
/// bearer — opencode has a whole plugin that trades it through
/// `login/oauth/access_token` first, and adds `X-GitHub-Api-Version` and
/// friends. Offering it would mean a user exports the documented variable
/// and gets an unexplained 401.
const OAUTH_ONLY: &[&str] = &["github-copilot"];

/// Applies this module's reachability rules to one catalog entry,
/// returning `None` for anything llmman cannot talk to.
fn routable(id: &str, raw: RawProvider) -> Option<Provider> {
    if OAUTH_ONLY.contains(&id) {
        return None;
    }
    let builtin = BUILTIN_ENDPOINTS.iter().find(|b| b.id == id);

    // A builtin's whole purpose is to supply an endpoint models.dev has
    // no `api` for, so it is trusted over a templated, absent or
    // plaintext one — but never over a concrete https `api`, which is the
    // newer of the two and the one that tracks a provider moving hosts.
    //
    // https only: this URL is where an API key is sent, and the catalog
    // is fetched at runtime, so an `http://` entry upstream would be
    // enough to put a user's key on the wire in the clear. Today that
    // rule drops exactly the loopback entries (`lmstudio`, `lynkr`,
    // `privatemode-ai`, `atomic-chat`) — not a loss, since `--provider`
    // is for models llmman does not serve itself, and a local one is
    // llmman's own job.
    let base_url = match raw.api.as_deref().map(str::trim) {
        Some(api) if api.starts_with("https://") && !api.contains("${") => api,
        _ => builtin?.base_url,
    };

    // `npm` is the wire-format signal. A builtin has already been vetted
    // by hand, so its own `wire` stands in for a driver package this
    // doesn't know (`google`'s entry names `@ai-sdk/google` even though
    // the endpoint used here is its OpenAI-compatible one).
    let wire = match (builtin, raw.npm.as_deref()) {
        (Some(b), _) => b.wire,
        (None, Some(npm)) if OPENAI_COMPATIBLE_NPM.contains(&npm) => Wire::OpenAi,
        _ => return None,
    };

    // One API key, and llmman must know which variable holds it. A
    // builtin names it explicitly (`google` lists three aliases); for
    // everything else a single-entry `env` is the only unambiguous case.
    let key_env = match builtin {
        Some(b) => b.key_env.to_string(),
        None => match raw.env.as_slice() {
            [only] if !only.trim().is_empty() => only.trim().to_string(),
            _ => return None,
        },
    };

    Some(Provider {
        id: id.to_string(),
        name: raw.name,
        base_url: base_url_of(base_url),
        key_env: Some(key_env),
        wire,
        key_optional: false,
        // Sorted by id, since `models` came out of a BTreeMap.
        models: raw
            .models
            .into_iter()
            .map(|(id, model)| Model {
                id,
                cost: model.cost.as_ref().and_then(cost_of),
                max_output: model.limit.as_ref().and_then(max_output_of),
            })
            .collect(),
    })
}

/// Normalizes a models.dev `api` into a base URL that a route can be
/// appended to.
///
/// Some entries give the full route rather than the base it hangs off
/// (`bailing` is `.../v1/chat/completions` today). Appending to that
/// yields `.../chat/completions/chat/completions`, which fails as a
/// confusing 404 from the provider rather than anything llmman reports.
/// Trailing slashes go for the same reason — `url` adds its own.
fn base_url_of(api: &str) -> String {
    let api = api.trim_end_matches('/');
    api.strip_suffix("/chat/completions")
        .or_else(|| api.strip_suffix("/messages"))
        .unwrap_or(api)
        .trim_end_matches('/')
        .to_string()
}

// ---------------------------------------------------------------------------
// Fetch + cache
// ---------------------------------------------------------------------------

/// How long a failed load is remembered before the next call retries.
/// Short, because `llmman serve` outlives the transient failure that
/// produced it — but not zero, or every request on a machine that cannot
/// reach models.dev would pay [`FETCH_TIMEOUT`] again.
const RETRY_COOLDOWN: Duration = Duration::from_secs(60);

/// The last [`load`], when it happened, and how long it is good for. The
/// error is kept as a message rather than an `anyhow::Error`, which is
/// not cloneable.
type Cached = (Instant, Duration, Result<Arc<Catalog>, String>);
static CATALOG: Mutex<Option<Cached>> = Mutex::new(None);

/// The routable provider catalog, fetched from models.dev on first use.
///
/// Memoized, but not for the life of the process: `llmman serve` runs for
/// days, so a success is re-checked after [`CACHE_TTL`] and a failure
/// after [`RETRY_COOLDOWN`] — otherwise one blip at startup would leave a
/// daemon with no providers until someone restarted it.
///
/// The lock is held across the fetch, so concurrent callers wait for one
/// load rather than racing several. Blocking: callers on an async runtime
/// (`cmd::serve`) must go through `spawn_blocking`.
///
/// The providers `llmman.conf` defines are merged in on top (see
/// [`Catalog::with_configured`]), and stand alone when models.dev cannot
/// be loaded at all: a machine with no route out and a vLLM on the LAN
/// is exactly where a defined provider is wanted, and must not be held
/// hostage to a catalog it has no use for.
pub fn catalog() -> anyhow::Result<Arc<Catalog>> {
    let mut cached = CATALOG.lock().unwrap_or_else(|e| e.into_inner());
    let live = cached
        .as_ref()
        .is_some_and(|(at, good_for, _)| at.elapsed() < *good_for);
    if !live {
        let configured = crate::config::configured_providers();
        // A stale catalog is held only as long as a failure: what
        // produced it was a failed refresh, whatever it managed to
        // return. See Loaded.
        *cached = Some(match load() {
            Ok(loaded) => (
                Instant::now(),
                loaded.good_for(),
                Ok(Arc::new(loaded.into_catalog().with_configured(configured))),
            ),
            Err(e) if !configured.is_empty() => {
                eprintln!(
                    "[llmman] using only the {} provider(s) defined in llmman.conf ({e:#})",
                    configured.len()
                );
                (
                    Instant::now(),
                    RETRY_COOLDOWN,
                    Ok(Arc::new(Catalog::from_configured(configured))),
                )
            }
            Err(e) => (Instant::now(), RETRY_COOLDOWN, Err(format!("{e:#}"))),
        });
    }
    match &cached.as_ref().expect("just populated").2 {
        Ok(catalog) => Ok(catalog.clone()),
        Err(e) => Err(anyhow::anyhow!(e.clone())),
    }
}

fn cache_path() -> Option<PathBuf> {
    crate::default_cache().ok().map(|d| d.join(CACHE_FILE))
}

/// Whether `path` was written within [`CACHE_TTL`]. An mtime that can't
/// be read, or is in the future on a machine whose clock moved, counts as
/// stale: re-fetching is cheap, serving something unbounded is not.
fn is_fresh(path: &std::path::Path) -> bool {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|m| SystemTime::now().duration_since(m).ok())
        .is_some_and(|age| age < CACHE_TTL)
}

/// A loaded catalog, and whether it is the result llmman wanted.
///
/// [`Loaded::Stale`] is a success the caller should not settle into: the
/// refresh behind it failed, so [`catalog`] re-tries it on
/// [`RETRY_COOLDOWN`] rather than the full [`CACHE_TTL`] it gives a real
/// one. Without the distinction, going offline once for a second would
/// pin an out-of-date provider list for a day.
enum Loaded {
    Fresh(Catalog),
    Stale(Catalog),
}

impl Loaded {
    /// How long [`catalog`] may serve this without loading again.
    fn good_for(&self) -> Duration {
        match self {
            Self::Fresh(_) => CACHE_TTL,
            Self::Stale(_) => RETRY_COOLDOWN,
        }
    }

    fn into_catalog(self) -> Catalog {
        match self {
            Self::Fresh(c) | Self::Stale(c) => c,
        }
    }
}

/// Cache-first catalog load: a fresh cache is used as-is, otherwise
/// models.dev is fetched and the result cached. If that refresh fails,
/// falls back to a stale cache — an out-of-date provider list beats no
/// provider list on a machine that is merely offline.
fn load() -> anyhow::Result<Loaded> {
    let path = cache_path();

    if let Some(path) = path.as_deref().filter(|p| is_fresh(p)) {
        if let Ok(raw) = std::fs::read(path) {
            if let Ok(catalog) = Catalog::from_json(&raw) {
                crate::debug_log!("provider catalog: {} cached", path.display());
                return Ok(Loaded::Fresh(catalog));
            }
        }
    }

    crate::debug_log!("provider catalog: fetching {CATALOG_URL}");
    // Parse is part of the refresh, not a separate step after it: a 200
    // carrying a truncated body or a captive portal's HTML is a failed
    // refresh, and must reach the stale cache below like any other.
    let refreshed = fetch(CATALOG_URL).and_then(|raw| {
        let catalog = Catalog::from_json(&raw)?;
        Ok((catalog, raw))
    });

    match refreshed {
        Ok((catalog, raw)) => {
            // Best-effort: an unwritable cache costs a fetch next time,
            // nothing more, so it must not fail the load.
            if let Some(path) = &path {
                let _ = write_cache(path, &raw);
            }
            Ok(Loaded::Fresh(catalog))
        }
        Err(err) => {
            let stale = path
                .as_deref()
                .and_then(|p| std::fs::read(p).ok())
                .and_then(|raw| Catalog::from_json(&raw).ok());
            match stale {
                Some(catalog) => {
                    eprintln!("[llmman] using cached provider list ({err:#})");
                    Ok(Loaded::Stale(catalog))
                }
                None => Err(err.context(format!(
                    "could not fetch the provider list from {CATALOG_URL}, and no cached \
                     copy is available"
                ))),
            }
        }
    }
}

/// Writes the cache atomically (`fsutil::write_atomic`), as opencode does
/// for its own copy: `llmman launch` and `llmman serve` refresh it
/// independently and do overlap, and a plain write leaves a window in
/// which the other reads a half-written file.
fn write_cache(path: &std::path::Path, raw: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    crate::fsutil::write_atomic(path, raw)
}

fn fetch(url: &str) -> anyhow::Result<Vec<u8>> {
    let resp = reqwest::blocking::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .build()
        .context("build HTTP client")?
        .get(url)
        .send()
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    anyhow::ensure!(status.is_success(), "GET {url} returned {status}");
    Ok(resp.bytes().context("read models.dev response")?.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every builtin must name a plausible HTTPS endpoint and a key
    /// variable — a typo here would surface only as a confusing runtime
    /// failure against a real provider.
    #[test]
    fn builtin_endpoints_are_well_formed_and_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for b in BUILTIN_ENDPOINTS {
            assert!(seen.insert(b.id), "duplicate builtin {}", b.id);
            assert!(
                b.base_url.starts_with("https://"),
                "{} base_url is not https",
                b.id
            );
            assert!(
                !b.base_url.ends_with('/'),
                "{} base_url has a trailing slash",
                b.id
            );
            assert!(!b.key_env.is_empty(), "{} has no key_env", b.id);
        }
    }

    #[test]
    fn format_and_split_remote_ref_round_trip() {
        let reference = format_remote_ref("openrouter", "qwen/qwen3-coder");
        assert_eq!(reference, "llmman.provider/openrouter/qwen/qwen3-coder");
        assert_eq!(
            split_remote_ref(&reference),
            Some(("openrouter", "qwen/qwen3-coder"))
        );
        assert_eq!(
            split_remote_ref("llmman.provider/groq/llama-3.3-70b"),
            Some(("groq", "llama-3.3-70b"))
        );
    }

    /// The whole reason [`REMOTE_PREFIX`] exists: a bare `<org>/<repo>`
    /// is a HuggingFace reference (`shortnames::resolve` turns it into
    /// `hf.co/<org>/<repo>`), and must never be mistaken for a
    /// provider-routed one just because the org shares a provider's name.
    #[test]
    fn split_remote_ref_ignores_local_and_huggingface_references() {
        for local in [
            "google/gemma-3",
            "openai/gpt-oss-120b",
            "qwen3.5:0.8b",
            "docker.io/ai/gemma4",
            "hf.co/unsloth/Qwen3.5-0.8B-GGUF",
            "",
        ] {
            assert_eq!(split_remote_ref(local), None, "{local} routed as remote");
            assert!(!is_remote_ref(local));
        }
    }

    /// A truncated reference is not a remote one — better to fall through
    /// to the normal local path (which reports "model not found") than to
    /// route to an empty provider or an empty model.
    #[test]
    fn split_remote_ref_rejects_incomplete_references() {
        assert_eq!(split_remote_ref("llmman.provider/"), None);
        assert_eq!(split_remote_ref("llmman.provider/openrouter"), None);
        assert_eq!(split_remote_ref("llmman.provider/openrouter/"), None);
        assert_eq!(split_remote_ref("llmman.provider//gpt-5"), None);
    }

    fn catalog_from(json: &str) -> Catalog {
        Catalog::from_json(json.as_bytes()).expect("fixture parses")
    }

    /// The base case: an `@ai-sdk/openai-compatible` provider with one
    /// concrete URL and one key variable is taken straight from
    /// models.dev, builtins uninvolved.
    #[test]
    fn openai_compatible_providers_come_from_the_catalog() {
        let catalog = catalog_from(
            r#"{
                "openrouter": {
                    "id": "openrouter",
                    "name": "OpenRouter",
                    "api": "https://openrouter.ai/api/v1",
                    "npm": "@openrouter/ai-sdk-provider",
                    "env": ["OPENROUTER_API_KEY"],
                    "models": {
                        "z-model": { "cost": { "input": 2.5, "output": 10 }, "limit": { "context": 200000, "output": 32000 } },
                        "a-model": {}
                    }
                }
            }"#,
        );
        let p = catalog.get("openrouter").expect("openrouter is routable");
        assert_eq!(p.name, "OpenRouter");
        assert_eq!(p.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(p.key_env.as_deref(), Some("OPENROUTER_API_KEY"));
        assert_eq!(p.wire, Wire::OpenAi);
        // Sorted by id; an unpriced model keeps `None`, not a zero.
        assert_eq!(
            p.models,
            vec![
                Model {
                    id: "a-model".into(),
                    cost: None,
                    max_output: None,
                },
                Model {
                    id: "z-model".into(),
                    cost: Some(Cost {
                        input: 2.5,
                        output: 10.0
                    }),
                    max_output: Some(32000),
                },
            ]
        );
    }

    /// models.dev leaves `api` unset for providers whose SDK hardcodes
    /// the endpoint; those are recovered from `BUILTIN_ENDPOINTS`, each
    /// with the wire format its builtin was vetted for.
    #[test]
    fn builtins_supply_endpoints_the_catalog_omits() {
        let catalog = catalog_from(
            r#"{
                "anthropic": {
                    "id": "anthropic", "name": "Anthropic",
                    "npm": "@ai-sdk/anthropic", "env": ["ANTHROPIC_API_KEY"],
                    "models": { "claude-sonnet-4": {} }
                },
                "google": {
                    "id": "google", "name": "Google",
                    "npm": "@ai-sdk/google",
                    "env": ["GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY"],
                    "models": { "gemini-2.5-pro": {} }
                }
            }"#,
        );
        let anthropic = catalog.get("anthropic").expect("anthropic is routable");
        assert_eq!(anthropic.base_url, "https://api.anthropic.com/v1");
        assert_eq!(anthropic.key_env.as_deref(), Some("ANTHROPIC_API_KEY"));
        // Its own Messages API, never the OpenAI-compatibility shim.
        assert_eq!(anthropic.wire, Wire::Anthropic);

        // Three candidate variables in the catalog, one unambiguous
        // choice from the builtin.
        let google = catalog.get("google").expect("google is routable");
        assert_eq!(google.key_env.as_deref(), Some("GEMINI_API_KEY"));
        assert_eq!(google.wire, Wire::OpenAi);
    }

    /// A full `/messages` route is trimmed to its base like
    /// `/chat/completions` is.
    #[test]
    fn a_messages_route_is_trimmed_to_its_base() {
        assert_eq!(
            base_url_of("https://api.anthropic.com/v1/messages/"),
            "https://api.anthropic.com/v1"
        );
    }

    /// A concrete `api` tracks a provider moving hosts, so it wins over
    /// the hand-written fallback rather than the other way round.
    #[test]
    fn a_concrete_api_overrides_the_builtin_endpoint() {
        let catalog = catalog_from(
            r#"{
                "openai": {
                    "id": "openai", "name": "OpenAI",
                    "api": "https://api.openai.example/v2/",
                    "npm": "@ai-sdk/openai", "env": ["OPENAI_API_KEY"],
                    "models": { "gpt-5": {} }
                }
            }"#,
        );
        let p = catalog.get("openai").expect("openai is routable");
        // Trailing slash normalized away so `url()` can't double it.
        assert_eq!(p.base_url, "https://api.openai.example/v2");
    }

    /// Everything llmman cannot reach with an API-key POST in one of its
    /// two wire formats must be absent rather than half-supported.
    #[test]
    fn unreachable_providers_are_dropped() {
        let catalog = catalog_from(
            r#"{
                "minimax": {
                    "id": "minimax", "name": "MiniMax",
                    "api": "https://api.minimax.io/anthropic/v1",
                    "npm": "@ai-sdk/anthropic", "env": ["MINIMAX_API_KEY"],
                    "models": { "minimax-m2": {} }
                },
                "amazon-bedrock": {
                    "id": "amazon-bedrock", "name": "Amazon Bedrock",
                    "npm": "@ai-sdk/amazon-bedrock",
                    "env": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
                    "models": { "claude": {} }
                },
                "snowflake-cortex": {
                    "id": "snowflake-cortex", "name": "Snowflake Cortex",
                    "api": "https://${SNOWFLAKE_ACCOUNT}.snowflakecomputing.com/api/v2/cortex/v1",
                    "npm": "@ai-sdk/openai-compatible",
                    "env": ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_CORTEX_PAT"],
                    "models": { "claude": {} }
                },
                "cloudflare-workers-ai": {
                    "id": "cloudflare-workers-ai", "name": "Cloudflare Workers AI",
                    "api": "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/v1",
                    "npm": "@ai-sdk/openai-compatible",
                    "env": ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_KEY"],
                    "models": { "llama": {} }
                },
                "github-copilot": {
                    "id": "github-copilot", "name": "GitHub Copilot",
                    "api": "https://api.githubcopilot.com",
                    "npm": "@ai-sdk/openai-compatible", "env": ["GITHUB_TOKEN"],
                    "models": { "gpt-5": {} }
                },
                "openrouter": {
                    "id": "openrouter", "name": "OpenRouter",
                    "api": "https://openrouter.ai/api/v1",
                    "npm": "@openrouter/ai-sdk-provider", "env": ["OPENROUTER_API_KEY"],
                    "models": { "gpt-5": {} }
                }
            }"#,
        );
        // The Messages body format, but no vetted auth scheme: only the
        // `anthropic` builtin rides that wire.
        assert!(catalog.get("minimax").is_none());
        // SigV4 signing, and no endpoint at all.
        assert!(catalog.get("amazon-bedrock").is_none());
        // Per-account templated endpoint, two-variable auth.
        assert!(catalog.get("snowflake-cortex").is_none());
        assert!(catalog.get("cloudflare-workers-ai").is_none());
        // Advertises a plain bearer variable, but the real credential is
        // minted by an OAuth exchange. See OAUTH_ONLY.
        assert!(catalog.get("github-copilot").is_none());
        // The one entry llmman can actually drive, so the filter is doing
        // the dropping rather than the catalog being unusable outright.
        assert_eq!(catalog.len(), 1);
        assert!(catalog.get("openrouter").is_some());
    }

    /// A plaintext `api` is where an API key would be sent, so it is no
    /// more usable than a templated one: a builtin's own https endpoint
    /// stands in where there is one, and the provider is dropped where
    /// there isn't. In today's catalog that second case is the loopback
    /// entries, which `--provider` has no business routing to anyway.
    #[test]
    fn plaintext_endpoints_are_never_routed_to() {
        let catalog = catalog_from(
            r#"{
                "openai": {
                    "id": "openai", "name": "OpenAI",
                    "api": "http://api.openai.example/v1",
                    "npm": "@ai-sdk/openai", "env": ["OPENAI_API_KEY"],
                    "models": { "gpt-5": {} }
                },
                "lmstudio": {
                    "id": "lmstudio", "name": "LM Studio",
                    "api": "http://127.0.0.1:1234/v1",
                    "npm": "@ai-sdk/openai-compatible", "env": ["LMSTUDIO_API_KEY"],
                    "models": { "some-model": {} }
                }
            }"#,
        );
        assert_eq!(
            catalog.get("openai").expect("openai is routable").base_url,
            "https://api.openai.com/v1"
        );
        assert!(catalog.get("lmstudio").is_none());
    }

    /// A real models.dev response, trimmed to the fields llmman reads
    /// (the per-model metadata is megabytes and none of it is used).
    /// Hand-written fixtures above pin one rule each; this one is what
    /// says the rules together produce a usable list from real data.
    const REAL_CATALOG: &[u8] = include_bytes!("../tests/fixtures/models-dev-providers.json");

    /// The filter has to be selective without being empty: a rule that
    /// accidentally excluded everything, or admitted a provider llmman
    /// cannot authenticate to, would pass every fixture above.
    #[test]
    fn the_real_catalog_yields_a_usable_provider_list() {
        let catalog = Catalog::from_json(REAL_CATALOG).expect("real catalog parses");
        assert!(
            (150..207).contains(&catalog.len()),
            "{} of 207 routable — the filter moved a lot",
            catalog.len()
        );
        for p in catalog.iter() {
            assert!(
                p.base_url.starts_with("https://"),
                "{}: {}",
                p.id,
                p.base_url
            );
            assert!(!p.base_url.ends_with('/'), "{}: {}", p.id, p.base_url);
            assert!(!p.base_url.contains("${"), "{}: {}", p.id, p.base_url);
            assert!(
                p.key_env.as_deref().is_some_and(|v| !v.is_empty()),
                "{} has no key variable",
                p.id
            );
        }

        // The providers someone actually reaches for, at the endpoints
        // each was verified to answer `POST /chat/completions` on.
        for (id, base_url) in [
            ("openai", "https://api.openai.com/v1"),
            ("anthropic", "https://api.anthropic.com/v1"),
            (
                "google",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ),
            ("groq", "https://api.groq.com/openai/v1"),
            ("openrouter", "https://openrouter.ai/api/v1"),
            ("perplexity", "https://api.perplexity.ai"),
        ] {
            let p = catalog.get(id).unwrap_or_else(|| panic!("{id} is dropped"));
            assert_eq!(p.base_url, base_url, "{id}");
        }

        // And the ones it must not offer, each for a different reason.
        for id in [
            "amazon-bedrock", // SigV4 signing
            "google-vertex",  // GCP service-account credentials
            "azure",          // per-account endpoint
            "lmstudio",       // loopback, and llmman's own job anyway
        ] {
            assert!(catalog.get(id).is_none(), "{id} is offered");
        }
    }

    /// Some entries give the full chat route where a base URL belongs
    /// (`bailing` does today). Appending to that yields
    /// `.../chat/completions/chat/completions`, which reaches the user as
    /// a bare 404 from the provider rather than anything llmman said.
    #[test]
    fn a_route_specific_api_is_normalized_back_to_its_base() {
        let catalog = catalog_from(
            r#"{
                "bailing": {
                    "id": "bailing", "name": "Bailing",
                    "api": "https://api.tbox.cn/api/llm/v1/chat/completions",
                    "npm": "@ai-sdk/openai-compatible", "env": ["BAILING_API_KEY"],
                    "models": { "ling-1t": {} }
                }
            }"#,
        );
        let p = catalog.get("bailing").expect("bailing is routable");
        assert_eq!(p.base_url, "https://api.tbox.cn/api/llm/v1");
        assert_eq!(
            p.url("/v1/chat/completions"),
            "https://api.tbox.cn/api/llm/v1/chat/completions"
        );

        // A trailing slash on the route form too, and a base that merely
        // ends in something similar must be left alone.
        assert_eq!(
            base_url_of("https://x.example/v1/chat/completions/"),
            "https://x.example/v1"
        );
        assert_eq!(
            base_url_of("https://x.example/completions"),
            "https://x.example/completions"
        );
    }

    /// No provider may produce a doubled route, whatever shape its `api`
    /// arrives in — the fixture is where such an entry actually shows up.
    #[test]
    fn no_real_provider_builds_a_doubled_route() {
        for p in Catalog::from_json(REAL_CATALOG).unwrap().iter() {
            let url = p.url("/v1/chat/completions");
            assert_eq!(
                url.matches("/chat/completions").count(),
                1,
                "{}: {url}",
                p.id
            );
        }
    }

    /// A body that parses but yields nothing routable is a failed
    /// refresh, not an empty catalog. `{"error":"..."}` is the shape that
    /// matters: accepting it would overwrite a good cache with nothing
    /// and memoize that, breaking `--provider` until the TTL expired.
    #[test]
    fn a_body_with_nothing_routable_is_an_error_not_an_empty_catalog() {
        for body in [
            r#"{"error":"temporarily unavailable"}"#,
            r#"{"message":"rate limited","retry_after":30}"#,
            "{}",
        ] {
            assert!(
                Catalog::from_json(body.as_bytes()).is_err(),
                "{body} was accepted as a catalog"
            );
        }
    }

    /// A stale catalog is a *failed* refresh that happened to have
    /// something to fall back on, so it must expire on the retry cooldown
    /// like any other failure. Holding it for the full TTL is how one
    /// second offline pins an out-of-date provider list for a day.
    #[test]
    fn a_stale_catalog_is_held_no_longer_than_a_failure() {
        assert_eq!(Loaded::Fresh(Catalog::default()).good_for(), CACHE_TTL);
        assert_eq!(Loaded::Stale(Catalog::default()).good_for(), RETRY_COOLDOWN);
        assert!(
            RETRY_COOLDOWN < CACHE_TTL,
            "a failed refresh outlives a good one"
        );
    }

    /// A builtin only ever supplies the endpoint models.dev omits, so one
    /// naming a provider that already has a concrete `api` upstream is
    /// dead weight nobody would notice — the entry is silently unused.
    #[test]
    fn every_builtin_is_for_a_provider_the_catalog_leaves_without_an_api() {
        let raw: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(REAL_CATALOG).expect("fixture parses");
        for b in BUILTIN_ENDPOINTS {
            let entry = raw
                .get(b.id)
                .unwrap_or_else(|| panic!("builtin {} is not in the catalog at all", b.id));
            let api = entry.get("api").and_then(|v| v.as_str()).unwrap_or("");
            assert!(
                !api.starts_with("https://"),
                "builtin {} is unused: the catalog already gives it {api}",
                b.id
            );
            let env: Vec<&str> = entry["env"]
                .as_array()
                .map(|e| e.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();
            assert!(
                env.contains(&b.key_env),
                "builtin {} names {}, which is not one of the catalog's {env:?}",
                b.id,
                b.key_env
            );
        }
    }

    /// One upstream entry that grows an unexpected shape must cost that
    /// provider, not the catalog — `--provider` keeps working on today's
    /// llmman against tomorrow's api.json.
    #[test]
    fn a_malformed_entry_does_not_reject_the_whole_catalog() {
        let catalog = catalog_from(
            r#"{
                "broken": { "id": "broken", "name": ["not", "a", "string"] },
                "openrouter": {
                    "id": "openrouter", "name": "OpenRouter",
                    "api": "https://openrouter.ai/api/v1",
                    "npm": "@openrouter/ai-sdk-provider",
                    "env": ["OPENROUTER_API_KEY"],
                    "models": { "gpt-5": {} }
                }
            }"#,
        );
        assert!(catalog.get("broken").is_none());
        assert!(catalog.get("openrouter").is_some());
    }

    /// An OpenAI-compatible provider is still dropped when llmman can't
    /// tell which of its variables is the API key.
    #[test]
    fn multi_variable_auth_is_dropped_without_a_builtin() {
        let catalog = catalog_from(
            r#"{
                "databricks": {
                    "id": "databricks", "name": "Databricks",
                    "api": "https://example.cloud.databricks.com/ai-gateway/mlflow/v1",
                    "npm": "@ai-sdk/openai-compatible",
                    "env": ["DATABRICKS_HOST", "DATABRICKS_TOKEN"],
                    "models": { "dbrx": {} }
                },
                "openrouter": {
                    "id": "openrouter", "name": "OpenRouter",
                    "api": "https://openrouter.ai/api/v1",
                    "npm": "@openrouter/ai-sdk-provider", "env": ["OPENROUTER_API_KEY"],
                    "models": { "gpt-5": {} }
                }
            }"#,
        );
        assert!(catalog.get("databricks").is_none());
        assert!(catalog.get("openrouter").is_some());
    }

    /// `url()` has to append to whatever version segment the provider
    /// already published, since that is not always `/v1`.
    #[test]
    fn url_appends_routes_to_the_published_base() {
        let p = Provider {
            id: "groq".into(),
            name: "Groq".into(),
            base_url: "https://api.groq.com/openai/v1".into(),
            key_env: Some("GROQ_API_KEY".into()),
            key_optional: false,
            wire: Wire::OpenAi,
            models: vec![],
        };
        assert_eq!(
            p.url("/v1/chat/completions"),
            "https://api.groq.com/openai/v1/chat/completions"
        );
        assert_eq!(
            p.url("/v1/embeddings"),
            "https://api.groq.com/openai/v1/embeddings"
        );

        // A base with a non-`v1` version segment, and one with none at
        // all, must both come out with exactly one `/chat/completions`.
        let gemini = Provider {
            base_url: "https://generativelanguage.googleapis.com/v1beta/openai".into(),
            ..p.clone()
        };
        assert_eq!(
            gemini.url("/v1/chat/completions"),
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        );
        let perplexity = Provider {
            base_url: "https://api.perplexity.ai".into(),
            ..p
        };
        assert_eq!(
            perplexity.url("/v1/chat/completions"),
            "https://api.perplexity.ai/chat/completions"
        );
    }

    #[test]
    fn api_key_reads_the_providers_own_variable() {
        let p = Provider {
            id: "test".into(),
            name: "Test".into(),
            base_url: "https://example.invalid/v1".into(),
            key_env: Some("LLMMAN_TEST_PROVIDER_KEY_UNSET".into()),
            key_optional: false,
            wire: Wire::OpenAi,
            models: vec![],
        };
        assert_eq!(p.api_key(), None);
    }

    /// An `export` overrides the standing answer in `llmman.conf`. The
    /// other way round would quietly spend the wrong key.
    #[test]
    fn the_environment_beats_the_config_file() {
        assert_eq!(
            resolve_key(Some("from-env".into()), Some("from-conf")),
            Some("from-env".to_string())
        );
        assert_eq!(
            resolve_key(None, Some("from-conf")),
            Some("from-conf".to_string())
        );
        assert_eq!(
            resolve_key(Some("from-env".into()), None),
            Some("from-env".to_string())
        );
        assert_eq!(resolve_key(None, None), None);
    }

    /// What is only whitespace is no key — as in the environment.
    #[test]
    fn a_config_file_key_is_trimmed_and_a_blank_one_ignored() {
        assert_eq!(
            resolve_key(None, Some("  padded  ")),
            Some("padded".to_string())
        );
        assert_eq!(resolve_key(None, Some("   ")), None);
        assert_eq!(resolve_key(None, Some("")), None);
    }

    /// Both places, in one message.
    #[test]
    fn key_hint_names_the_variable_and_the_config_file() {
        let hint = key_hint("openrouter", Some("OPENROUTER_API_KEY"));
        assert!(hint.contains("OPENROUTER_API_KEY"), "{hint}");
        assert!(hint.contains("[providers.openrouter]"), "{hint}");
        assert!(hint.contains("llmman.conf"), "{hint}");

        // No variable to name: the file alone, and no dangling "or".
        let hint = key_hint("gpubox", None);
        assert!(hint.contains("[providers.gpubox]"), "{hint}");
        assert!(hint.contains("llmman.conf"), "{hint}");
        assert!(!hint.contains("environment"), "{hint}");
    }

    fn configured(id: &str, base_url: &str, wire: Wire) -> crate::config::ConfiguredProvider {
        crate::config::ConfiguredProvider {
            id: id.into(),
            name: id.into(),
            base_url: base_url.into(),
            wire,
            key_env: None,
        }
    }

    /// A provider `llmman.conf` defines joins the catalog as a full
    /// [`Provider`]: no models, no key demanded, plain `http` allowed —
    /// none of which a catalog entry could get away with.
    #[test]
    fn configured_providers_join_the_catalog_with_key_optional() {
        let catalog = catalog_from(
            r#"{
                "openrouter": {
                    "id": "openrouter", "name": "OpenRouter",
                    "api": "https://openrouter.ai/api/v1",
                    "npm": "@openrouter/ai-sdk-provider",
                    "env": ["OPENROUTER_API_KEY"],
                    "models": { "m": {} }
                }
            }"#,
        )
        .with_configured(&[
            configured("gpubox", "http://gpubox:8000/v1", Wire::OpenAi),
            crate::config::ConfiguredProvider {
                id: "relay".into(),
                name: "Claude relay".into(),
                base_url: "https://relay.example/v1".into(),
                wire: Wire::Anthropic,
                key_env: Some("RELAY_KEY".into()),
            },
        ]);

        assert_eq!(catalog.len(), 3);
        assert_eq!(
            catalog.ids().collect::<Vec<_>>(),
            ["gpubox", "openrouter", "relay"]
        );

        let gpubox = catalog.get("gpubox").unwrap();
        assert_eq!(gpubox.base_url, "http://gpubox:8000/v1");
        assert_eq!(gpubox.wire, Wire::OpenAi);
        assert!(gpubox.key_optional);
        assert_eq!(gpubox.key_env, None);
        assert!(gpubox.models.is_empty());
        assert_eq!(gpubox.api_key(), None);
        assert_eq!(
            gpubox.url("/v1/chat/completions"),
            "http://gpubox:8000/v1/chat/completions"
        );

        let relay = catalog.get("relay").unwrap();
        assert_eq!(relay.name, "Claude relay");
        assert_eq!(relay.wire, Wire::Anthropic);
        assert_eq!(relay.key_env.as_deref(), Some("RELAY_KEY"));
        assert!(relay.key_optional);

        // The catalog entry is as it was.
        let openrouter = catalog.get("openrouter").unwrap();
        assert!(!openrouter.key_optional);
        assert_eq!(openrouter.models.len(), 1);
    }

    /// The file is the user's deliberate word, so it wins over the
    /// catalog for the same id — the one way to point `openai` at a
    /// proxy without renaming every integration's model.
    #[test]
    fn a_configured_provider_shadows_the_catalog_entry_with_its_id() {
        let catalog = catalog_from(
            r#"{
                "openai": {
                    "id": "openai", "name": "OpenAI",
                    "npm": "@ai-sdk/openai", "env": ["OPENAI_API_KEY"],
                    "models": { "gpt-5": {} }
                }
            }"#,
        )
        .with_configured(&[configured(
            "openai",
            "http://proxy.corp:4000/v1",
            Wire::OpenAi,
        )]);
        assert_eq!(catalog.len(), 1);
        let p = catalog.get("openai").unwrap();
        assert_eq!(p.base_url, "http://proxy.corp:4000/v1");
        assert!(p.key_optional);
        assert!(
            p.models.is_empty(),
            "the catalog's models are not the proxy's"
        );
    }

    /// With no models.dev at all — offline, or a first run with no cache
    /// — the configured providers are still a catalog. That is the case
    /// they exist for.
    #[test]
    fn configured_providers_stand_alone_without_the_catalog() {
        let catalog = Catalog::from_configured(&[configured(
            "gpubox",
            "http://gpubox:8000/v1",
            Wire::OpenAi,
        )]);
        assert_eq!(catalog.len(), 1);
        assert!(catalog.get("gpubox").is_some());
        assert!(Catalog::from_configured(&[]).is_empty());
    }

    /// `wire = "openai"` in the file is the same spelling the daemon's
    /// API reports; anything else is a parse error, not a default.
    #[test]
    fn wire_deserializes_from_its_reported_name() {
        #[derive(Deserialize)]
        struct W {
            wire: Wire,
        }
        let parse = |s: &str| toml::from_str::<W>(&format!("wire = {s:?}")).map(|w| w.wire);
        assert_eq!(parse("openai").unwrap(), Wire::OpenAi);
        assert_eq!(parse("anthropic").unwrap(), Wire::Anthropic);
        assert!(parse("OpenAI").is_err());
        assert!(parse("ollama").is_err());
        for wire in [Wire::OpenAi, Wire::Anthropic] {
            assert_eq!(parse(wire.as_str()).unwrap(), wire);
        }
    }

    /// models.dev ships `wafer.ai`, and `[providers.wafer.ai]` is two
    /// nested tables rather than that key — a hint saying so would not
    /// configure the provider it names.
    #[test]
    fn key_hint_quotes_a_provider_id_that_is_not_a_bare_toml_key() {
        assert!(
            key_hint("wafer.ai", Some("WAFER_API_KEY")).contains(r#"[providers."wafer.ai"]"#),
            "{}",
            key_hint("wafer.ai", Some("WAFER_API_KEY"))
        );
        assert_eq!(toml_key("openrouter"), "openrouter");
        assert_eq!(toml_key("z-ai"), "z-ai");
        assert_eq!(toml_key("wafer.ai"), r#""wafer.ai""#);
    }

    /// The listing has to stay short enough to read in an error message
    /// while still saying how much was elided.
    #[test]
    fn example_models_lists_a_few_and_counts_the_rest() {
        assert_eq!(
            example_models("Groq", &["a", "b"]),
            "Groq models include: a, b"
        );
        assert_eq!(
            example_models("Groq", &["a", "b", "c", "d", "e", "f", "g"]),
            "Groq models include: a, b, c, d, e, … (2 more)"
        );
        assert_eq!(example_models("Groq", &[]), "Groq lists no models");
    }

    /// An explicitly blank `--provider` asked for a hosted model and gave
    /// no id; running locally instead would pull weights nobody asked
    /// for.
    #[test]
    fn a_blank_provider_flag_is_an_error_not_an_absent_one() {
        assert_eq!(provider_flag(None).unwrap(), None);
        assert_eq!(provider_flag(Some("  openai ")).unwrap(), Some("openai"));
        assert!(provider_flag(Some("")).is_err());
        assert!(provider_flag(Some("   ")).is_err());
    }

    /// A price is carried only when llmman is sure of it; anything else
    /// is `None`, which `list --provider` prints as unknown, not free.
    #[test]
    fn cost_of_takes_both_figures_or_neither() {
        let cost = |json: &str| cost_of(&serde_json::from_str(json).unwrap());
        assert_eq!(
            cost(r#"{"input": 2.5, "output": 10}"#),
            Some(Cost {
                input: 2.5,
                output: 10.0
            })
        );
        // A genuinely free model is a price, not a missing one.
        assert_eq!(
            cost(r#"{"input": 0, "output": 0, "cache_read": 1}"#),
            Some(Cost {
                input: 0.0,
                output: 0.0
            })
        );
        // Half a price, no price, and a shape llmman doesn't recognize.
        assert_eq!(cost(r#"{"input": 2.5}"#), None);
        assert_eq!(cost(r#"{}"#), None);
        assert_eq!(cost(r#"{"input": "2.5", "output": "10"}"#), None);
        assert_eq!(cost(r#"{"input": -1, "output": 10}"#), None);
        assert_eq!(cost(r#"[]"#), None);
    }

    /// The reason `cost` is untyped: a shape llmman doesn't know costs
    /// that model its price, not the provider its routability.
    #[test]
    fn an_unrecognized_cost_shape_keeps_the_provider() {
        let catalog = catalog_from(
            r#"{
                "openrouter": {
                    "id": "openrouter", "name": "OpenRouter",
                    "api": "https://openrouter.ai/api/v1",
                    "npm": "@openrouter/ai-sdk-provider", "env": ["OPENROUTER_API_KEY"],
                    "models": {
                        "weird": { "cost": "free, actually" },
                        "fine": { "cost": { "input": 1, "output": 2 } }
                    }
                }
            }"#,
        );
        let p = catalog.get("openrouter").expect("openrouter is routable");
        assert_eq!(p.models.len(), 2);
        assert_eq!(p.models[0].id, "fine");
        assert!(p.models[1].cost.is_none());
    }

    /// A half-remembered or fat-fingered id has to come back with the
    /// real one, or a 180-entry listing has to be re-read.
    #[test]
    fn unknown_provider_error_suggests_near_matches() {
        let catalog = catalog_from(
            r#"{
                "togetherai": {
                    "id": "togetherai", "name": "Together",
                    "api": "https://api.together.xyz/v1",
                    "npm": "@ai-sdk/openai-compatible", "env": ["TOGETHER_API_KEY"],
                    "models": { "gpt-5": {} }
                }
            }"#,
        );
        // A shortened id, caught by the substring pass.
        let short = unknown_provider_error("together", &catalog).to_string();
        assert!(short.contains("unknown provider \"together\""), "{short}");
        assert!(short.contains("Did you mean: togetherai?"), "{short}");
        assert!(short.contains("llmman providers"), "{short}");

        // A dropped character, which only the distance pass catches.
        let typo = unknown_provider_error("togethr", &catalog).to_string();
        assert!(typo.contains("Did you mean: togetherai?"), "{typo}");

        // Nothing close: still name the command that lists them all,
        // without inventing a suggestion.
        let far = unknown_provider_error("nope", &catalog).to_string();
        assert!(!far.contains("Did you mean"), "{far}");
        assert!(far.contains("llmman providers"), "{far}");
    }

    /// An id from a request body can be megabytes long (see
    /// `resolve_remote_target` in cmd::serve). Scanning the catalog
    /// against it — let alone a distance matrix per entry — buys nothing:
    /// it cannot match either pass.
    #[test]
    fn an_absurdly_long_id_is_reported_without_a_suggestion_pass() {
        let catalog = catalog_from(
            r#"{
                "togetherai": {
                    "id": "togetherai", "name": "Together",
                    "api": "https://api.together.xyz/v1",
                    "npm": "@ai-sdk/openai-compatible", "env": ["TOGETHER_API_KEY"],
                    "models": { "gpt-5": {} }
                }
            }"#,
        );
        // Absurdly long, and *padding a real id* so the assertion below
        // actually pins the length guard: `needle.contains(id)` would
        // match without it, so "Did you mean" appearing is exactly the
        // signal that the search was not skipped. A plain run of "z"
        // returns no suggestion either way and would pass regardless.
        //
        // No wall-clock budget: the previous one measured this test's own
        // multi-megabyte `{:?}` formatting and `contains` scans on a
        // shared CI runner more than anything in `suggestions`.
        let huge = format!("togetherai{}", "z".repeat(64 * 1024));
        let error = unknown_provider_error(&huge, &catalog).to_string();
        assert!(!error.contains("Did you mean"), "suggested for a huge id");
        assert!(error.contains("llmman providers"));
    }

    /// The exact threshold, pinned deterministically — what the test
    /// above used to approximate with a timer. `suggestions` short-
    /// circuits on length because `edit_distance` over every catalog id
    /// is superlinear in a caller-supplied one.
    #[test]
    fn suggestions_are_skipped_just_above_the_length_limit() {
        let catalog = catalog_from(
            r#"{
                "togetherai": {
                    "id": "togetherai", "name": "Together",
                    "api": "https://api.together.xyz/v1",
                    "npm": "@ai-sdk/openai-compatible", "env": ["TOGETHER_API_KEY"],
                    "models": { "gpt-5": {} }
                }
            }"#,
        );
        // Both of these contain a real id, so both would be suggested if
        // the guard were gone; only the length distinguishes them.
        let pad = |len: usize| format!("togetherai{}", "z".repeat(len - "togetherai".len()));

        assert_eq!(
            suggestions(&pad(MAX_SUGGESTION_LEN), &catalog),
            vec!["togetherai"],
            "an id at exactly the limit was not searched"
        );
        assert!(
            suggestions(&pad(MAX_SUGGESTION_LEN + 1), &catalog).is_empty(),
            "one byte over the limit was still searched"
        );
    }
}

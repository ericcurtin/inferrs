//! `llmman.conf` — one config file, read once, shared by every consumer.
//!
//! Two locations, later overriding earlier:
//!
//!   1. `/etc/llmman/llmman.conf`        system-wide
//!   2. `~/.config/llmman/llmman.conf`   per-user
//!
//! The same paths on every platform, with no llmman-specific variable
//! to move them: one documented answer to "where does this go". (`~` is
//! `$HOME` as usual.)
//!
//! ```toml
//! [aliases]                            # crate::shortnames
//! gemma4 = "docker.io/ai/gemma4"
//!
//! [providers.openrouter]               # crate::providers
//! api_key = "sk-or-..."
//!
//! [providers.gpubox]                   # a provider models.dev does not list
//! base_url = "http://gpubox:8000/v1"
//! wire     = "openai"                  # default; or "anthropic"
//!
//! [verify]                             # crate::verify
//! default = "off"
//!
//! [[verify.trust]]
//! pattern = "docker.io/myorg/**"
//! keys    = ["keys/myorg.pub"]         # relative to this file's directory
//! mode    = "enforce"
//!
//! [aggregation]                        # cmd::serve::aggregation
//! peers = "asahi,spark:17434"
//!
//! [registries."docker.io"]             # crate::ffi, go-shim
//! mirrors = "https://mirror.gcr.io,registry-mirror.corp:5000"
//! ```
//!
//! Parsing happens once, in [`files`]; what a *failed* parse means is
//! left to each consumer, because it differs. [`crate::verify`] refuses
//! to run, since a trust policy it cannot read must not quietly become
//! `off`; aliases and keys degrade to none. The error is reported once,
//! centrally, so one typo is not announced three times.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Deserialize;

const FILE: &str = "llmman.conf";

/// Everything `llmman.conf` configures.
///
/// `deny_unknown_fields` throughout: a misspelling that parsed happily
/// would be a policy, alias or credential that silently never takes
/// effect, surfacing somewhere else entirely — an unsigned pull that
/// passes, a 401 inside someone else's TUI.
#[derive(Deserialize, Default, Debug)]
#[serde(deny_unknown_fields)]
pub struct Conf {
    /// Short-name aliases — see [`crate::shortnames`].
    #[serde(default)]
    pub aliases: HashMap<String, String>,
    /// Private: a key leaves this module only through
    /// [`provider_api_key`], which gates it on the file's mode.
    #[serde(default)]
    providers: HashMap<String, ProviderConf>,
    /// Signature trust policy — see [`crate::verify`].
    #[serde(default)]
    pub verify: VerifyConf,
    /// Peer daemons — see `cmd::serve::aggregation`.
    #[serde(default)]
    pub aggregation: AggregationConf,
    /// `[registries."<host>"]` tables, keyed by the registry host a
    /// reference names — see [`registry_mirrors`].
    #[serde(default)]
    pub registries: HashMap<String, RegistryConf>,
}

/// The `[aggregation]` section.
#[derive(Deserialize, Default, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct AggregationConf {
    /// Comma-separated `[scheme://]host[:port]`, as `LLMMAN_PEERS` — a
    /// string so `llmman config set aggregation.peers a,b` can write it.
    #[serde(default)]
    pub peers: Option<String>,
}

/// One `[registries."<host>"]` table.
#[derive(Deserialize, Default, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct RegistryConf {
    /// Comma-separated `[scheme://]host[:port][/path]` mirrors, tried in
    /// order before the registry for pulls. A string, like `peers`, so
    /// `llmman config set` can write it.
    #[serde(default)]
    pub mirrors: Option<String>,
}

/// One `[providers.<id>]` table. Without `base_url` it keys a catalog
/// provider; with it, it *defines* one — an inference server at a host
/// models.dev does not list — and shadows any catalog entry with its id.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderConf {
    #[serde(default)]
    api_key: Option<String>,
    /// Base URL the wire's route is appended to. Plain `http://` is
    /// allowed: the user typed it into their own owner-only file.
    #[serde(default)]
    base_url: Option<String>,
    /// What is spoken at `base_url`; `openai` by default.
    #[serde(default)]
    wire: Option<crate::providers::Wire>,
    /// Variable holding the key. With neither this nor `api_key`, no
    /// credential is sent — most local servers take none.
    #[serde(default)]
    api_key_env: Option<String>,
    /// Display name for listings; the id when absent.
    #[serde(default)]
    name: Option<String>,
}

/// Hand-written so the key cannot reach a log through the derived
/// `Debug` of the public [`Conf`] and [`File`]. Same reasoning as
/// `RemoteTarget` in `cmd::serve`.
impl std::fmt::Debug for ProviderConf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderConf")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("base_url", &self.base_url)
            .field("wire", &self.wire)
            .field("api_key_env", &self.api_key_env)
            .field("name", &self.name)
            .finish()
    }
}

impl ProviderConf {
    /// Whether this table defines a provider rather than only keying one.
    fn defines(&self) -> bool {
        self.base_url.is_some()
    }

    /// Fields that only mean something on a definition (see [`validate`]).
    fn definition_only_fields(&self) -> Vec<&'static str> {
        let mut set = Vec::new();
        if self.wire.is_some() {
            set.push("wire");
        }
        if self.api_key_env.is_some() {
            set.push("api_key_env");
        }
        if self.name.is_some() {
            set.push("name");
        }
        set
    }
}

/// A provider `llmman.conf` defines, merged across every file that
/// mentions it. What `crate::providers` turns into a `Provider`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfiguredProvider {
    pub id: String,
    /// `name` from the file, else the id.
    pub name: String,
    /// Without its trailing slash, as `Provider::base_url` is kept.
    pub base_url: String,
    pub wire: crate::providers::Wire,
    /// `api_key_env`, when set. The `api_key` itself is not here: it is
    /// reached through [`provider_api_key`], behind the file-mode gate.
    pub key_env: Option<String>,
}

/// The `[verify]` section. [`crate::verify`] owns what these strings
/// mean; this module only spells out their shape.
#[derive(Deserialize, Default, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct VerifyConf {
    /// Mode for references no rule matches.
    #[serde(default)]
    pub default: Option<String>,
    /// `[[verify.trust]]` entries, in the order they appear.
    #[serde(default)]
    pub trust: Vec<TrustConf>,
}

/// One `[[verify.trust]]` entry.
#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct TrustConf {
    pub pattern: String,
    #[serde(default)]
    pub keys: Vec<String>,
    #[serde(default)]
    pub mode: Option<String>,
}

impl Conf {
    /// Every entry that sets `api_key` at all, trimmed, by provider id.
    ///
    /// A blank value is kept here rather than dropped, so a user file can
    /// blank out a key `/etc` set — otherwise there is no way to opt out
    /// of a system-wide credential. [`load_provider_keys`] drops blanks
    /// after merging, since sending `Authorization: Bearer ` upstream
    /// would turn a clear "no API key" into someone else's 401.
    fn provider_keys(&self) -> HashMap<String, String> {
        self.providers
            .iter()
            .filter_map(|(id, p)| Some((id.clone(), p.api_key.as_deref()?.trim().to_string())))
            .collect()
    }
}

/// One `llmman.conf` that exists on disk.
#[derive(Debug)]
pub struct File {
    /// Where it was read from.
    pub path: PathBuf,
    /// Its own directory — what relative paths inside it resolve
    /// against, so a trust policy can ship alongside its keys.
    pub dir: PathBuf,
    pub conf: Conf,
}

// ---------------------------------------------------------------------------
// Search paths
// ---------------------------------------------------------------------------

/// Every place [`FILE`] may live, in ascending priority order.
///
/// Not the `/usr/share/llmman/`, `<binary>/../share/llmman/` and
/// `<binary-dir>/` tiers earlier versions searched: nothing ever shipped
/// a file to them, and they are package-managed and world-readable,
/// which `llmman.conf` cannot be.
pub fn search_paths() -> Vec<PathBuf> {
    let mut paths = vec![system_dir().join(FILE)];
    paths.extend(user_dir().map(|d| d.join(FILE)));
    paths
}

/// `/etc/llmman`.
#[cfg(not(windows))]
fn system_dir() -> PathBuf {
    PathBuf::from("/etc/llmman")
}

/// `/etc/llmman` anchored to the system drive.
///
/// A leading `/` is drive-*relative* on Windows, so a bare `/etc/llmman`
/// would mean `D:\etc\llmman` for a process launched from `D:` — the
/// system-wide location would move with the working directory.
#[cfg(windows)]
fn system_dir() -> PathBuf {
    let drive = std::env::var("SystemDrive").unwrap_or_else(|_| "C:".to_string());
    PathBuf::from(format!("{drive}\\etc\\llmman"))
}

/// `~/.config/llmman`, on every platform. No llmman-specific override;
/// `~` is `$HOME` as usual.
fn user_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".config").join("llmman"))
}

/// Where a user's own config belongs, for a message that has to name
/// it or a [`crate::cmd::config`] that has to write it. `None` only when
/// there is no home directory.
pub fn user_path() -> Option<PathBuf> {
    user_dir().map(|d| d.join(FILE))
}

/// Where the system-wide config belongs. Always known, unlike
/// [`user_path`]: it does not depend on there being a home directory.
pub fn system_path() -> PathBuf {
    system_dir().join(FILE)
}

/// [`user_path`] for printing, falling back to the bare file name.
pub fn user_path_display() -> String {
    user_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| FILE.to_string())
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Every `llmman.conf` that exists, lowest priority first, parsed once
/// for the process.
///
/// `Err` when one is there but unreadable or malformed — see the module
/// docs for why what to do about that is the caller's call.
pub fn files() -> Result<&'static [File], &'static str> {
    cache().as_deref().map_err(String::as_str)
}

fn cache() -> &'static Result<Vec<File>, String> {
    static CACHE: OnceLock<Result<Vec<File>, String>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let loaded = load();
        // Once, here, rather than by each of the three consumers.
        if let Err(e) = &loaded {
            eprintln!("[llmman] warning: {e}");
        }
        loaded
    })
}

fn load() -> Result<Vec<File>, String> {
    // No unit test may depend on whoever runs it having an llmman.conf:
    // their aliases would redirect a fixture reference, and a provider
    // called `openai` would resolve their real key, flipping the
    // assertions in `cmd::serve` and `cmd::providers` that prove no key
    // leaks. `parse`, `provider_keys` and `Policy::parse` are covered
    // directly instead. The binary the e2e tests spawn is built without
    // `cfg(test)`, so it reads the real files.
    if cfg!(test) {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    for path in search_paths() {
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            // Absence is the common case: most machines never have one.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(format!("ignoring {}: {e}", path.display())),
        };
        let conf = parse(&text).map_err(|e| format!("ignoring {}: {e}", path.display()))?;
        let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        files.push(File { path, dir, conf });
    }
    Ok(files)
}

/// Parse one `llmman.conf`. Split out from [`load`] so the format is
/// testable without a file, a home directory, or a particular umask.
pub(crate) fn parse(text: &str) -> Result<Conf, String> {
    let conf: Conf = toml::from_str(text).map_err(|e| e.to_string())?;
    validate(&conf)?;
    Ok(conf)
}

/// What the TOML shape alone cannot say, checked at parse time so
/// `llmman config set` refuses it on the spot.
fn validate(conf: &Conf) -> Result<(), String> {
    let mut seen: HashMap<String, &str> = HashMap::new();
    for (host, r) in &conf.registries {
        let canonical = canonical_registry_host(host).ok_or_else(|| {
            format!(
                "[registries.{host:?}]: expected a host[:port], as in [registries.\"docker.io\"]"
            )
        })?;
        if let Some(other) = seen.insert(canonical, host) {
            return Err(format!(
                "[registries.{host:?}] and [registries.{other:?}] name the same registry"
            ));
        }
        for mirror in split_list(r.mirrors.as_deref()) {
            mirror_url_of(&mirror).map_err(|e| format!("[registries.{host:?}] mirrors: {e}"))?;
        }
    }
    for (id, p) in &conf.providers {
        // `<provider>/<model>` is how a reference travels (see
        // `crate::providers::split_remote_ref`), so a slash in the id
        // could never be routed.
        if id.is_empty() || id.contains('/') {
            return Err(format!(
                "[providers.{id:?}]: a provider id cannot be empty or contain '/'"
            ));
        }
        match &p.base_url {
            Some(url) => {
                base_url_of(url).map_err(|e| format!("[providers.{id}] base_url: {e}"))?;
            }
            None => {
                let only_on_definition = p.definition_only_fields();
                if !only_on_definition.is_empty() {
                    return Err(format!(
                        "[providers.{id}] sets {} but no base_url, so it only keys the catalog \
                         provider {id:?}",
                        only_on_definition.join(", ")
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Parses an `http`/`https` URL with a host and no userinfo, query or
/// fragment. Userinfo is refused because these URLs are reported and
/// logged; `credential` names where a credential goes instead.
fn plain_http_url(url: &str, credential: &str) -> Result<reqwest::Url, String> {
    let parsed = reqwest::Url::parse(url).map_err(|e| format!("{url:?}: {e}"))?;
    match parsed.scheme() {
        "http" | "https" => {}
        other => {
            return Err(format!(
                "{url:?}: scheme must be http or https, not {other}"
            ))
        }
    }
    if parsed.host_str().is_none() {
        return Err(format!("{url:?}: no host"));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(format!(
            "{url:?}: cannot carry a username or password; use {credential}"
        ));
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(format!("{url:?}: no query or fragment allowed"));
    }
    Ok(parsed)
}

/// Normalizes a configured `base_url`: lowercased scheme and host, no
/// trailing slash.
fn base_url_of(url: &str) -> Result<String, String> {
    let parsed = plain_http_url(url.trim(), "api_key")?;
    Ok(parsed.as_str().trim_end_matches('/').to_string())
}

/// Normalizes a mirror, `[scheme://]host[:port][/path]`, `https` when no
/// scheme is given. A path is kept (a mirror served under a prefix);
/// `llmman login <mirror-host>` is where a mirror's credential goes.
fn mirror_url_of(mirror: &str) -> Result<String, String> {
    let mirror = mirror.trim();
    if mirror.is_empty() {
        return Err("an empty entry".to_string());
    }
    let with_scheme = if mirror.contains("://") {
        mirror.to_string()
    } else {
        format!("https://{mirror}")
    };
    let parsed = plain_http_url(&with_scheme, "`llmman login`")?;
    Ok(parsed.as_str().trim_end_matches('/').to_string())
}

/// The `host[:port]` a `[registries."<host>"]` key names, as the Go shim
/// looks it up: lowercased, Docker Hub's several names folded onto
/// `docker.io`. `None` for anything that is not a bare authority.
fn canonical_registry_host(host: &str) -> Option<String> {
    let host = host.trim();
    // `*` is a wildcard in registries.conf, which the url crate lets through.
    if host.is_empty() || host.contains("://") || host.contains(['/', '*']) {
        return None;
    }
    let parsed = plain_http_url(&format!("https://{host}"), "").ok()?;
    if parsed.path() != "/" {
        return None;
    }
    let mut canonical = parsed.host_str()?.to_string();
    if let Some(port) = parsed.port() {
        canonical = format!("{canonical}:{port}");
    }
    Some(match canonical.as_str() {
        "index.docker.io" | "registry-1.docker.io" => "docker.io".to_string(),
        _ => canonical,
    })
}

/// Splits a comma-separated setting into its trimmed, non-empty entries.
fn split_list(value: Option<&str>) -> Vec<String> {
    value
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

// ---------------------------------------------------------------------------
// Registry mirrors
// ---------------------------------------------------------------------------

/// Mirrors per registry host, normalized and in order: for Docker Hub
/// `LLMMAN_REGISTRY_MIRRORS` when set (even empty; Hub-only, like
/// dockerd's `--registry-mirror`), else the last file that sets the
/// host's `mirrors`. Replaced per host, not merged, so `mirrors = ""`
/// opts a host out. What a mirror does is the Go shim's business (see
/// go-shim/registry_mirrors.go).
pub fn registry_mirrors() -> BTreeMap<String, Vec<String>> {
    registry_mirrors_from(
        std::env::var("LLMMAN_REGISTRY_MIRRORS").ok().as_deref(),
        files().unwrap_or_default(),
    )
}

/// Whether `host`, as a reference spells it, has mirrors configured.
/// Cached for the process; `crate::hf` asks per reference classified.
pub fn has_registry_mirrors(host: &str) -> bool {
    static CACHE: OnceLock<BTreeMap<String, Vec<String>>> = OnceLock::new();
    canonical_registry_host(host)
        .is_some_and(|host| CACHE.get_or_init(registry_mirrors).contains_key(&host))
}

fn registry_mirrors_from(env: Option<&str>, files: &[File]) -> BTreeMap<String, Vec<String>> {
    let mut by_host: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for file in files {
        for (host, r) in &file.conf.registries {
            let Some(mirrors) = r.mirrors.as_deref() else {
                continue;
            };
            let host = canonical_registry_host(host).expect("validated at parse");
            let mirrors = split_list(Some(mirrors))
                .iter()
                .map(|m| mirror_url_of(m).expect("validated at parse"))
                .collect();
            by_host.insert(host, mirrors);
        }
    }
    if let Some(env) = env {
        let mut mirrors = Vec::new();
        for m in split_list(Some(env)) {
            match mirror_url_of(&m) {
                Ok(m) => mirrors.push(m),
                Err(e) => eprintln!("[llmman] warning: ignoring LLMMAN_REGISTRY_MIRRORS entry {e}"),
            }
        }
        by_host.insert("docker.io".to_string(), mirrors);
    }
    by_host.retain(|_, mirrors| !mirrors.is_empty());
    by_host
}

// ---------------------------------------------------------------------------
// Configured providers
// ---------------------------------------------------------------------------

/// Every provider `llmman.conf` defines, sorted by id. Read once for the
/// process, like the keys.
pub fn configured_providers() -> &'static [ConfiguredProvider] {
    static CACHE: OnceLock<Vec<ConfiguredProvider>> = OnceLock::new();
    CACHE.get_or_init(|| configured_from(files().unwrap_or_default()))
}

/// Merged field by field, later files overriding earlier, as keys do.
/// Split out to be testable without a filesystem.
fn configured_from(files: &[File]) -> Vec<ConfiguredProvider> {
    let mut merged: HashMap<&str, ConfiguredProvider> = HashMap::new();
    for file in files {
        for (id, p) in file.conf.providers.iter().filter(|(_, p)| p.defines()) {
            let base_url = p
                .base_url
                .as_deref()
                .and_then(|u| base_url_of(u).ok())
                .expect("validated at parse");
            let entry = merged.entry(id).or_insert_with(|| ConfiguredProvider {
                id: id.clone(),
                name: id.clone(),
                base_url: String::new(),
                wire: crate::providers::Wire::OpenAi,
                key_env: None,
            });
            entry.base_url = base_url;
            if let Some(wire) = p.wire {
                entry.wire = wire;
            }
            // Blank clears a variable an earlier file named, as `api_key = ""`
            // clears a key.
            if let Some(var) = p.api_key_env.as_deref().map(str::trim) {
                entry.key_env = (!var.is_empty()).then(|| var.to_string());
            }
            if let Some(name) = p.name.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
                entry.name = name.to_string();
            }
        }
    }
    let mut out: Vec<ConfiguredProvider> = merged.into_values().collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

// ---------------------------------------------------------------------------
// Provider API keys
// ---------------------------------------------------------------------------

/// The key `llmman.conf` configures for `provider_id`, or `None`. Keyed
/// by the id `llmman providers` prints, not by the environment variable,
/// which is models.dev's naming rather than llmman's.
///
/// Borrowed from a process-lifetime cache, so a daemon serving requests
/// is not stat'ing `/etc` per token.
pub fn provider_api_key(provider_id: &str) -> Option<&'static str> {
    static CACHE: OnceLock<HashMap<String, String>> = OnceLock::new();
    CACHE
        .get_or_init(load_provider_keys)
        .get(provider_id)
        .map(String::as_str)
}

fn load_provider_keys() -> HashMap<String, String> {
    let mut accepted = Vec::new();
    for file in files().unwrap_or_default() {
        let keys = file.conf.provider_keys();
        // The mode gate is for a file that holds a secret. One that only
        // blanks a key out does not.
        if !keys.values().all(String::is_empty) {
            if let Err(e) = owner_readable_only(&file.path) {
                eprintln!(
                    "[llmman] warning: ignoring the API keys in {}: {e}",
                    file.path.display()
                );
                continue;
            }
        }
        accepted.push(keys);
    }
    merge_keys(accepted)
}

/// Merge per-file key tables, lowest priority first, then drop blanks.
///
/// Blanks survive the merge so a user file can shadow a key `/etc` set,
/// and are dropped only at the end so none is ever spent. Split out to
/// be testable without a filesystem.
fn merge_keys(per_file: Vec<HashMap<String, String>>) -> HashMap<String, String> {
    let mut merged: HashMap<String, String> = HashMap::new();
    for keys in per_file {
        merged.extend(keys);
    }
    merged.retain(|_, key| !key.is_empty());
    merged
}

// ---------------------------------------------------------------------------
// Aggregation peers
// ---------------------------------------------------------------------------

/// Peer daemons, as written: `LLMMAN_PEERS` when set (even empty), else
/// the last file that sets `aggregation.peers`. Replaced, not merged —
/// half of two aggregations is no one's intent — so `peers = ""` opts out.
pub fn peers() -> Vec<String> {
    peers_from(
        std::env::var("LLMMAN_PEERS").ok().as_deref(),
        files().unwrap_or_default(),
    )
}

fn peers_from(env: Option<&str>, files: &[File]) -> Vec<String> {
    env.or_else(|| {
        files
            .iter()
            .rev()
            .find_map(|f| f.conf.aggregation.peers.as_deref())
    })
    .map(|peers| split_list(Some(peers)))
    .unwrap_or_default()
}

/// Refuses a file that group or other can read, the way `ssh` refuses a
/// loose private key.
///
/// Gates the keys, not the read: `/etc/llmman/llmman.conf` is
/// legitimately world-readable for the aliases and trust policy it also
/// carries, so a loose file still supplies everything but the key.
#[cfg(unix)]
pub(crate) fn owner_readable_only(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    let mode = std::fs::metadata(path)
        .map_err(|e| e.to_string())?
        .permissions()
        .mode();
    if mode & 0o077 != 0 {
        return Err(format!(
            "mode {:04o} lets other users read it. Run: chmod 600 {}",
            mode & 0o7777,
            path.display()
        ));
    }
    Ok(())
}

/// Unchecked on Windows: there are no mode bits, and reading an ACL
/// needs a `windows` dependency this crate does not have. Defaults are
/// owner-only under the user's profile, but a file with a widened ACL,
/// or one under the system directory, is accepted as-is.
#[cfg(not(unix))]
pub(crate) fn owner_readable_only(_path: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conf(text: &str) -> Conf {
        parse(text).expect("valid conf")
    }

    /// A key must not reach a log through the derived `Debug` of the
    /// public `Conf`/`File`.
    #[test]
    fn debug_output_never_carries_a_key() {
        let c = conf("[providers.openai]\napi_key = \"sk-secret-value\"");
        let rendered = format!("{c:?}");
        assert!(!rendered.contains("sk-secret-value"), "{rendered}");
        assert!(rendered.contains("redacted"), "{rendered}");
    }

    /// The documented shape, all three sections in one file.
    #[test]
    fn parses_every_section_of_one_file() {
        let c = conf(
            r#"
            [aliases]
            gemma4 = "docker.io/ai/gemma4"

            [providers.openrouter]
            api_key = "sk-or-abc"

            [verify]
            default = "warn"

            [[verify.trust]]
            pattern = "docker.io/myorg/**"
            keys    = ["keys/myorg.pub"]
            mode    = "enforce"

            [aggregation]
            peers = "asahi,spark:17434"

            [registries."docker.io"]
            mirrors = "https://mirror.gcr.io,registry-mirror.corp:5000"
            "#,
        );
        assert_eq!(
            c.aliases.get("gemma4").map(String::as_str),
            Some("docker.io/ai/gemma4")
        );
        assert_eq!(c.aggregation.peers.as_deref(), Some("asahi,spark:17434"));
        assert_eq!(
            c.registries["docker.io"].mirrors.as_deref(),
            Some("https://mirror.gcr.io,registry-mirror.corp:5000")
        );
        assert_eq!(
            c.provider_keys().get("openrouter").map(String::as_str),
            Some("sk-or-abc")
        );
        assert_eq!(c.verify.default.as_deref(), Some("warn"));
        assert_eq!(c.verify.trust.len(), 1);
        assert_eq!(c.verify.trust[0].mode.as_deref(), Some("enforce"));
    }

    /// A file that sets one section must not need the other two.
    #[test]
    fn every_section_is_optional() {
        let c = conf("");
        assert!(c.aliases.is_empty());
        assert!(c.provider_keys().is_empty());
        assert!(c.verify.default.is_none());
        assert!(c.verify.trust.is_empty());

        assert!(!conf("[aliases]\nx = \"y\"").aliases.is_empty());
        assert!(conf("[verify]\ndefault = \"off\"").aliases.is_empty());
    }

    /// A key is trimmed; one that sets no `api_key` at all is absent.
    /// A blank *is* carried this far — see [`merge_keys`].
    #[test]
    fn a_key_is_trimmed_and_an_unset_one_is_absent() {
        let keys = conf(
            r#"
            [providers.openai]
            api_key = "  sk-padded  "

            [providers.groq]
            api_key = "   "

            [providers.mistral]
            "#,
        )
        .provider_keys();
        assert_eq!(keys.get("openai").map(String::as_str), Some("sk-padded"));
        assert_eq!(keys.get("groq").map(String::as_str), Some(""));
        assert_eq!(keys.get("mistral"), None);
    }

    /// A user file must be able to blank out a key `/etc` set, or there
    /// is no way to opt out of a system-wide credential. A blank never
    /// survives as an empty bearer token either.
    #[test]
    fn a_later_blank_shadows_an_earlier_key_and_is_never_spent() {
        let system = HashMap::from([
            ("openai".to_string(), "sk-system".to_string()),
            ("groq".to_string(), "sk-groq".to_string()),
        ]);
        let user = HashMap::from([("openai".to_string(), String::new())]);

        let merged = merge_keys(vec![system, user]);
        assert_eq!(merged.get("openai"), None, "blanked out by the user file");
        assert_eq!(merged.get("groq").map(String::as_str), Some("sk-groq"));

        // And the ordinary case: a later real key replaces an earlier one.
        let merged = merge_keys(vec![
            HashMap::from([("openai".to_string(), "sk-system".to_string())]),
            HashMap::from([("openai".to_string(), "sk-user".to_string())]),
        ]);
        assert_eq!(merged.get("openai").map(String::as_str), Some("sk-user"));
    }

    /// A misspelling that parsed happily would silently never take
    /// effect. Every section rejects one.
    #[test]
    fn a_misspelled_section_or_field_is_rejected_rather_than_ignored() {
        assert!(parse("[provider.openai]\napi_key = \"x\"").is_err());
        assert!(parse("[providers.openai]\napi_kye = \"x\"").is_err());
        assert!(parse("[alias]\nx = \"y\"").is_err());
        assert!(parse("[verify]\ndefualt = \"off\"").is_err());
        assert!(parse("[[verify.trust]]\npattern = \"a/b\"\nkyes = []").is_err());
        assert!(parse("[registry.\"docker.io\"]\nmirrors = \"m\"").is_err());
        assert!(parse("[registries.\"docker.io\"]\nmirror = \"m\"").is_err());
        assert!(parse("api_key = \"x\"").is_err());
    }

    /// A mangled file is reported, not treated as empty.
    #[test]
    fn malformed_toml_is_an_error() {
        assert!(parse("[providers.openai").is_err());
    }

    // -- configured providers ------------------------------------------------

    /// A `[providers.<id>]` with a `base_url` defines a provider; every
    /// other field has a default. One without is only a key, as before.
    #[test]
    fn a_base_url_turns_a_provider_table_into_a_definition() {
        let files = [file(
            r#"
            [providers.gpubox]
            base_url = "http://gpubox:8000/v1/"

            [providers.relay]
            base_url    = "https://relay.example/v1"
            wire        = "anthropic"
            api_key_env = "RELAY_KEY"
            name        = "Claude relay"
            api_key     = "sk-relay"

            [providers.openrouter]
            api_key = "sk-or"
            "#,
        )];
        let configured = configured_from(&files);
        assert_eq!(
            configured,
            vec![
                ConfiguredProvider {
                    id: "gpubox".into(),
                    name: "gpubox".into(),
                    // Trailing slash gone: a route appends its own.
                    base_url: "http://gpubox:8000/v1".into(),
                    wire: crate::providers::Wire::OpenAi,
                    key_env: None,
                },
                ConfiguredProvider {
                    id: "relay".into(),
                    name: "Claude relay".into(),
                    base_url: "https://relay.example/v1".into(),
                    wire: crate::providers::Wire::Anthropic,
                    key_env: Some("RELAY_KEY".into()),
                },
            ]
        );
        // The key of a defined provider is reached the same way as any.
        assert_eq!(
            files[0]
                .conf
                .provider_keys()
                .get("relay")
                .map(String::as_str),
            Some("sk-relay")
        );
        assert!(configured_from(&[]).is_empty());
    }

    /// A later file re-defining an id replaces the fields it sets and
    /// keeps the rest, as keys merge.
    #[test]
    fn a_later_definition_overrides_field_by_field() {
        let system = file(
            "[providers.gpubox]\nbase_url = \"http://gpubox:8000/v1\"\nname = \"Shared box\"\n\
             api_key_env = \"SHARED_KEY\"",
        );
        let user = file(
            "[providers.gpubox]\nbase_url = \"http://10.0.0.5:8000/v1\"\nwire = \"anthropic\"\n\
             api_key_env = \"\"",
        );
        let merged = configured_from(&[system, user]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].base_url, "http://10.0.0.5:8000/v1");
        assert_eq!(merged[0].name, "Shared box");
        assert_eq!(merged[0].wire, crate::providers::Wire::Anthropic);
        // A blank clears the inherited variable, so /etc's key does not
        // follow the user to another endpoint.
        assert_eq!(merged[0].key_env, None);
    }

    /// What `config set` has to refuse on the spot: a `base_url` that is
    /// not an absolute http(s) URL, a definition field on a table that
    /// defines nothing, a `wire` llmman does not speak.
    #[test]
    fn a_bad_provider_definition_is_a_parse_error() {
        let bad = |body: &str| parse(&format!("[providers.gpubox]\n{body}")).expect_err(body);

        assert!(bad("base_url = \"gpubox:8000\"").contains("base_url"));
        assert!(bad("base_url = \"gpubox:8000/v1\"").contains("base_url"));
        assert!(bad("base_url = \"ftp://gpubox/v1\"").contains("http or https"));
        assert!(bad("base_url = \"http://\"").contains("base_url"));
        assert!(bad("base_url = \"http://gpubox/v1?x=1\"").contains("query"));
        assert!(bad("base_url = \"\"").contains("base_url"));
        // The URL is reported by the daemon's API; a secret does not go in it.
        assert!(bad("base_url = \"https://user:pw@gpubox/v1\"").contains("api_key"));
        assert!(bad("base_url = \"https://user@gpubox/v1\"").contains("api_key"));
        // A slash in the id would never survive `split_remote_ref`.
        assert!(
            parse("[providers.\"team/gpu\"]\nbase_url = \"http://g/v1\"")
                .expect_err("slash id")
                .contains("'/'")
        );
        assert!(parse("[providers.\"\"]\napi_key = \"x\"").is_err());

        let err = bad("wire = \"anthropic\"");
        assert!(err.contains("no base_url"), "{err}");
        assert!(err.contains("wire"), "{err}");
        let err = bad("api_key_env = \"X\"\nname = \"n\"");
        assert!(err.contains("api_key_env, name"), "{err}");

        assert!(bad("base_url = \"http://g/v1\"\nwire = \"ollama\"").contains("wire"));
        assert!(bad("base_url = \"http://g/v1\"\nbase_ulr = \"x\"").contains("base_ulr"));

        // The good shapes parse, and come out normalized: lowercase
        // scheme and host, no trailing slash.
        for (url, want) in [
            ("http://gpubox:8000/v1", "http://gpubox:8000/v1"),
            ("http://gpubox", "http://gpubox"),
            ("HTTP://GPUBox:8000/v1/", "http://gpubox:8000/v1"),
            ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/v1"),
            ("http://[::1]:8000/v1", "http://[::1]:8000/v1"),
            (
                "https://relay.example/api/v1",
                "https://relay.example/api/v1",
            ),
        ] {
            assert_eq!(base_url_of(url).expect(url), want);
            parse(&format!("[providers.gpubox]\nbase_url = {url:?}")).expect(url);
        }
    }

    /// The definition fields show in `Debug`; the key still does not.
    #[test]
    fn debug_output_shows_a_definition_but_never_a_key() {
        let c = conf(
            "[providers.gpubox]\nbase_url = \"http://gpubox:8000/v1\"\napi_key = \"sk-secret-value\"",
        );
        let rendered = format!("{c:?}");
        assert!(rendered.contains("gpubox:8000"), "{rendered}");
        assert!(!rendered.contains("sk-secret-value"), "{rendered}");
    }

    // -- aggregation peers ---------------------------------------------------

    fn file(text: &str) -> File {
        File {
            path: PathBuf::from("llmman.conf"),
            dir: PathBuf::from("."),
            conf: conf(text),
        }
    }

    #[test]
    fn peers_come_from_the_environment_then_the_last_file_that_sets_them() {
        let system = || file("[aggregation]\npeers = \"a, b\"");
        let user = file("[aggregation]\npeers = \"c\"");
        let silent = file("");
        let opted_out = file("[aggregation]\npeers = \"\"");

        assert_eq!(
            peers_from(Some(" x , ,y:1 "), &[]),
            vec!["x".to_string(), "y:1".to_string()]
        );
        assert!(peers_from(Some(""), &[system()]).is_empty());
        assert_eq!(
            peers_from(None, &[system(), silent]),
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(peers_from(None, &[system(), user]), vec!["c".to_string()]);
        assert!(peers_from(None, &[system(), opted_out]).is_empty());
        assert!(peers_from(None, &[]).is_empty());
        assert!(parse("[aggregation]\npeer = \"a\"").is_err());
    }

    // -- registry mirrors ----------------------------------------------------

    /// A mirror is spelled like a URL with the scheme optional, and
    /// comes out normalized the way the Go shim wants it.
    #[test]
    fn a_mirror_is_normalized_and_defaults_to_https() {
        assert_eq!(
            mirror_url_of("mirror.gcr.io").as_deref(),
            Ok("https://mirror.gcr.io")
        );
        assert_eq!(
            mirror_url_of(" HTTP://Mirror.Corp:5000/ ").as_deref(),
            Ok("http://mirror.corp:5000")
        );
        assert_eq!(
            mirror_url_of("https://proxy.corp/registry/").as_deref(),
            Ok("https://proxy.corp/registry")
        );
        for bad in [
            "",
            "ftp://mirror",
            "https://",
            "https://user:pw@mirror",
            "https://mirror?x=1",
            "https://mirror#frag",
        ] {
            assert!(mirror_url_of(bad).is_err(), "{bad:?}");
        }
    }

    /// The table key is a host. Hub's aliases fold onto `docker.io`,
    /// which is what a reference's `docker.io/ai/x` names and what the
    /// Go shim is asked for.
    #[test]
    fn a_registry_key_is_a_host_and_hub_has_one_name() {
        assert_eq!(
            canonical_registry_host("Index.Docker.io").as_deref(),
            Some("docker.io")
        );
        assert_eq!(
            canonical_registry_host("registry-1.docker.io").as_deref(),
            Some("docker.io")
        );
        assert_eq!(
            canonical_registry_host("localhost:5000").as_deref(),
            Some("localhost:5000")
        );
        assert_eq!(
            canonical_registry_host("ghcr.io").as_deref(),
            Some("ghcr.io")
        );
        assert_eq!(
            canonical_registry_host("[::1]:5000").as_deref(),
            Some("[::1]:5000")
        );
        assert_eq!(
            canonical_registry_host("ghcr.io:443").as_deref(),
            Some("ghcr.io")
        );
        for bad in [
            "",
            "https://ghcr.io",
            "ghcr.io/org",
            "*.example.com",
            "host:port",
            ":5000",
            "user@registry",
            "registry?q=1",
            "registry#f",
            "[not-an-ip]:5000",
            "a b",
        ] {
            assert_eq!(canonical_registry_host(bad), None, "{bad:?}");
        }
        assert!(parse("[registries.\"https://ghcr.io\"]\nmirrors = \"m\"").is_err());
        assert!(parse("[registries.\"ghcr.io\"]\nmirrors = \"ftp://m\"").is_err());
        assert!(parse("[registries.\"ghcr.io\"]\nmirrors = \"m\"").is_ok());
        // Two keys for one registry would leave which wins to HashMap order.
        assert!(parse(
            "[registries.\"docker.io\"]\nmirrors = \"a\"\n[registries.\"index.docker.io\"]\nmirrors = \"b\""
        )
        .is_err());
        assert!(parse(
            "[registries.\"ghcr.io\"]\nmirrors = \"a\"\n[registries.\"GHCR.io\"]\nmirrors = \"b\""
        )
        .is_err());
    }

    /// Per host, the environment (Hub only), then the last file that
    /// sets it; a blank opts a host out; hosts sort, mirrors keep order.
    #[test]
    fn mirrors_come_from_the_environment_then_the_last_file_that_sets_them() {
        let system = || {
            file(
                r#"
                [registries."docker.io"]
                mirrors = "https://a, b:5000"

                [registries."ghcr.io"]
                mirrors = "http://g"
                "#,
            )
        };
        let user = file("[registries.\"index.docker.io\"]\nmirrors = \"c\"");
        let opted_out = file("[registries.\"docker.io\"]\nmirrors = \"\"");
        let silent = file("[registries.\"docker.io\"]");

        let got = registry_mirrors_from(None, &[system(), silent]);
        assert_eq!(
            got.get("docker.io").map(Vec::as_slice),
            Some(&["https://a".to_string(), "https://b:5000".to_string()][..])
        );
        assert_eq!(
            got.get("ghcr.io").map(Vec::as_slice),
            Some(&["http://g".to_string()][..])
        );

        let got = registry_mirrors_from(None, &[system(), user]);
        assert_eq!(
            got.get("docker.io").map(Vec::as_slice),
            Some(&["https://c".to_string()][..]),
            "a Hub alias replaces docker.io's own entry"
        );

        let got = registry_mirrors_from(None, &[system(), opted_out]);
        assert!(!got.contains_key("docker.io"));
        assert!(
            got.contains_key("ghcr.io"),
            "opting Hub out leaves ghcr.io alone"
        );

        let got = registry_mirrors_from(Some(" x , ,http://y:1/ "), &[system()]);
        assert_eq!(
            got.get("docker.io").map(Vec::as_slice),
            Some(&["https://x".to_string(), "http://y:1".to_string()][..])
        );
        assert!(!registry_mirrors_from(Some(""), &[system()]).contains_key("docker.io"));
        assert!(registry_mirrors_from(Some("ftp://nope"), &[]).is_empty());
        assert!(registry_mirrors_from(None, &[]).is_empty());
    }

    // -- search paths --------------------------------------------------------

    /// Two locations, system first so the user's file wins — and none
    /// of the package-managed ones earlier versions searched.
    #[test]
    fn the_search_path_is_etc_then_the_user_directory() {
        let paths = search_paths();
        assert_eq!(
            paths.first().expect("a system path"),
            &system_dir().join(FILE)
        );
        assert!(system_dir().ends_with("etc/llmman"), "{:?}", system_dir());
        // Rooted, so it cannot resolve against the working directory.
        assert!(system_dir().is_absolute(), "{:?}", system_dir());
        assert!(
            !paths.iter().any(|p| p.starts_with("/usr/share")),
            "{paths:?}"
        );
        if let Some(user) = user_path() {
            assert_eq!(paths.last(), Some(&user));
            assert!(user.ends_with("llmman/llmman.conf"), "{}", user.display());
        }
    }

    /// One per-user location on every platform, so there is exactly one
    /// place `user_path` can name in an error.
    #[test]
    fn there_is_one_user_directory_on_every_platform() {
        assert_eq!(search_paths().len(), user_dir().iter().count() + 1);
        if let Some(dir) = user_dir() {
            assert!(dir.ends_with(".config/llmman"), "{}", dir.display());
        }
    }

    /// A key any account on the box can read is not one worth spending.
    ///
    /// Everything here is inlined rather than sharing a helper: a helper
    /// used only by a `cfg(unix)` test is dead code on Windows, which
    /// `clippy --all-targets -D warnings` rejects.
    #[cfg(unix)]
    #[test]
    fn a_group_or_world_readable_file_is_refused() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!("llmman-conf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(FILE);
        std::fs::write(&path, "[providers.openai]\napi_key = \"sk-x\"\n").expect("write");

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).expect("chmod");
        let err = owner_readable_only(&path).expect_err("0644 is refused");
        assert!(err.contains("chmod 600"), "{err}");

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
        assert!(owner_readable_only(&path).is_ok());

        std::fs::remove_dir_all(&dir).ok();
    }
}

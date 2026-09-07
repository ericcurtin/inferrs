//! Signature trust policy: which references must be signed, and by whom.
//!
//! The cryptography lives in the Go shim (go-shim/sigstore.go, reached
//! through [`crate::ffi::verify`]); this module decides *when* to invoke
//! it and *what a negative answer means*, which is a policy question.
//!
//! Policy comes from the `[verify]` section of `llmman.conf` (see
//! [`crate::config`] for the locations), merged with later files
//! winning:
//!
//! ```toml
//! [verify]
//! default = "off"                  # for references no rule matches
//!
//! [[verify.trust]]
//! pattern = "docker.io/myorg/**"
//! keys    = ["keys/myorg.pub"]     # relative to this file's directory
//! mode    = "enforce"              # off | warn | enforce
//! ```
//!
//! The default is `off`, not `warn`, on purpose: with no configured trust
//! roots there is nothing any model could be checked against, so `warn`
//! would print an unsigned-model warning on every pull that the user can
//! do nothing about — a reliable way to teach people to ignore the one
//! warning that eventually matters.
//!
//! `LLMMAN_VERIFY=off|warn|enforce` overrides the mode (not the keys) for
//! every reference, which is the useful knob in CI.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{anyhow, bail, Context};

use crate::config::VerifyConf;

/// Longest accepted `pattern`. Nothing real comes close; the bound just
/// keeps a pathological one uninteresting.
const MAX_PATTERN_LEN: usize = 256;

/// What a reference that does not verify should cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Don't check at all.
    #[default]
    Off,
    /// Check, report the outcome, proceed either way.
    Warn,
    /// Check, and refuse to proceed unless a trusted key signed it.
    Enforce,
}

impl Mode {
    fn parse(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "false" | "0" | "no" => Ok(Mode::Off),
            "warn" | "warning" => Ok(Mode::Warn),
            "enforce" | "require" | "required" | "true" | "1" | "yes" => Ok(Mode::Enforce),
            other => Err(anyhow!(
                "unknown verification mode {other:?} (expected \"off\", \"warn\", or \"enforce\")"
            )),
        }
    }
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Mode::Off => "off",
            Mode::Warn => "warn",
            Mode::Enforce => "enforce",
        })
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Rule {
    pattern: String,
    keys: Vec<PathBuf>,
    mode: Mode,
}

#[derive(Debug, Default)]
pub struct Policy {
    /// `None` where no file set `default`. Kept optional so an explicit
    /// `default = "off"` in a higher-priority file can override a lower
    /// one's `warn`, which "later files win" requires and a plain
    /// `Mode::Off` could not express.
    default_mode: Option<Mode>,
    /// `LLMMAN_VERIFY`, overriding every rule's mode. Resolved once at
    /// load, so an unparsable value fails there rather than being
    /// re-reported (or worse, ignored) on every decision.
    forced_mode: Option<Mode>,
    /// Ascending precedence: the *last* matching rule wins.
    rules: Vec<Rule>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    pub mode: Mode,
    pub keys: Vec<PathBuf>,
}

impl Policy {
    /// One file's `[verify]` section. `dir` is that file's own
    /// directory, which relative key paths resolve against, so a policy
    /// can ship with its keys.
    fn from_conf(conf: &VerifyConf, dir: &Path) -> anyhow::Result<Self> {
        let default_mode = conf
            .default
            .as_deref()
            .map(|s| Mode::parse(s).context("`default`"))
            .transpose()?;

        let mut rules = Vec::with_capacity(conf.trust.len());
        for entry in &conf.trust {
            validate_pattern(&entry.pattern)?;
            let mode = match entry.mode.as_deref() {
                Some(s) => Mode::parse(s)
                    .with_context(|| format!("[[verify.trust]] pattern {:?}", entry.pattern))?,
                // Keys but no mode means "check this, tell me, don't
                // break my build".
                None => Mode::Warn,
            };
            rules.push(Rule {
                pattern: entry.pattern.clone(),
                keys: entry
                    .keys
                    .iter()
                    .map(|k| resolve_key_path(k, dir))
                    .collect(),
                mode,
            });
        }
        Ok(Policy {
            default_mode,
            forced_mode: None,
            rules,
        })
    }

    /// A `[verify]` section on its own, for tests that care about the
    /// policy and not the file it came from.
    #[cfg(test)]
    fn parse(text: &str, dir: &Path) -> anyhow::Result<Self> {
        Self::from_conf(&toml::from_str(text).context("parse TOML")?, dir)
    }

    /// The mode and trusted keys that apply to `reference`.
    ///
    /// Only the winning rule's keys are used; they are not accumulated
    /// across matching rules. A narrower rule must be able to *replace*
    /// a broad one's key set, or "trust the org key everywhere except
    /// here" would be inexpressible.
    pub fn decide(&self, reference: &str) -> Decision {
        let repo = canonical_repository(repository_of(reference));
        let winner = self
            .rules
            .iter()
            .rev()
            .find(|rule| glob_match(&rule.pattern, &repo));
        let mut decision = match winner {
            Some(rule) => Decision {
                mode: rule.mode,
                keys: rule.keys.clone(),
            },
            None => Decision {
                mode: self.default_mode.unwrap_or_default(),
                keys: Vec::new(),
            },
        };
        if let Some(forced) = self.forced_mode {
            decision.mode = forced;
        }
        decision
    }

    /// Load and merge every config file present, cached for the process.
    ///
    /// A file that cannot be read or parsed is fatal here, not skipped:
    /// it could be the one demanding enforcement, and treating it like
    /// an absent file would silently downgrade to `off`. This is why
    /// [`crate::config::files`] reports a parse failure rather than
    /// deciding what it means — the other two readers of that file can
    /// carry on without it, and this one cannot.
    pub fn load() -> anyhow::Result<&'static Policy> {
        static CACHE: OnceLock<Result<Policy, String>> = OnceLock::new();
        CACHE
            .get_or_init(|| load_uncached().map_err(|e| format!("{e:#}")))
            .as_ref()
            .map_err(|e| anyhow!("{e}"))
    }
}

fn load_uncached() -> anyhow::Result<Policy> {
    let mut merged = Policy::default();
    for file in crate::config::files().map_err(|e| anyhow!("{e}"))? {
        let policy = Policy::from_conf(&file.conf.verify, &file.dir)
            .with_context(|| format!("read {}", file.path.display()))?;
        if let Some(mode) = policy.default_mode {
            merged.default_mode = Some(mode);
        }
        merged.rules.extend(policy.rules);
    }
    merged.forced_mode = mode_override()?;
    Ok(merged)
}

/// Rejects a pattern that cannot match anything. Patterns are compared
/// against the repository, so one carrying a tag or digest would simply
/// never fire — silently leaving a rule the author believes is enforcing
/// something. Same reasoning as rejecting an unknown `mode`.
fn validate_pattern(pattern: &str) -> anyhow::Result<()> {
    if pattern.is_empty() {
        bail!("a [[verify.trust]] entry has an empty `pattern`");
    }
    if pattern.len() > MAX_PATTERN_LEN {
        bail!("[[verify.trust]] pattern is longer than {MAX_PATTERN_LEN} bytes: {pattern:?}");
    }
    if pattern.contains('@') {
        bail!(
            "[[verify.trust]] pattern {pattern:?} contains a digest; patterns match the \
             repository only"
        );
    }
    let last = pattern.rsplit('/').next().unwrap_or(pattern);
    if last.contains(':') {
        bail!(
            "[[verify.trust]] pattern {pattern:?} contains a tag; patterns match the \
             repository only"
        );
    }
    Ok(())
}

/// `LLMMAN_VERIFY`, if set.
///
/// An unparsable value is an error, not a warning. Most of the time this
/// is read inside the daemon, whose stderr is a log file — so warning
/// and carrying on would turn `LLMMAN_VERIFY=enfroce` into a silent
/// downgrade to whatever the config files happened to say, which is the
/// exact failure this whole module refuses elsewhere.
fn mode_override() -> anyhow::Result<Option<Mode>> {
    parse_mode_override(std::env::var("LLMMAN_VERIFY").ok().as_deref())
}

/// Split out so it is testable without touching the process environment.
fn parse_mode_override(raw: Option<&str>) -> anyhow::Result<Option<Mode>> {
    match raw.map(str::trim) {
        None | Some("") => Ok(None),
        Some(raw) => Mode::parse(raw).map(Some).context("LLMMAN_VERIFY"),
    }
}

/// Expand `~` and make a relative key path absolute against the config
/// file's own directory.
fn resolve_key_path(raw: &str, dir: &Path) -> PathBuf {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    let path = PathBuf::from(trimmed);
    if path.is_absolute() {
        path
    } else {
        dir.join(path)
    }
}

// ---------------------------------------------------------------------------
// Reference matching
// ---------------------------------------------------------------------------

/// The repository part of a reference: host/namespace/name, tag and
/// digest removed. Patterns match this, because a trust rule is about
/// who publishes a model and tags move.
///
/// Mirrors `repositoryOf` in go-shim/sigstore.go, including the rule that
/// a ":" only starts a tag if it comes after the last "/".
fn repository_of(reference: &str) -> &str {
    let mut r = reference;
    if let Some(i) = r.rfind('@') {
        r = &r[..i];
    }
    let last_slash = r.rfind('/').map(|i| i as isize).unwrap_or(-1);
    if let Some(i) = r.rfind(':') {
        if (i as isize) > last_slash {
            r = &r[..i];
        }
    }
    r
}

/// Normalizes a repository to its one fully-qualified spelling, so every
/// way of writing a Docker Hub reference matches the same pattern:
/// `gemma4`, `ai/gemma4`, `docker.io/ai/gemma4` and
/// `index.docker.io/ai/gemma4` all become `docker.io/...`.
///
/// Applied to the *reference* only, never to a pattern. Normalizing a
/// pattern would rewrite what the author wrote: `docker.io/**` would lose
/// its host and become `**`, quietly widening a Docker Hub rule to every
/// registry — so a `mode = "off"` rule for Hub would disable verification
/// everywhere, and an `enforce` one would demand the Hub key from quay.io.
///
/// This normalizes *up* where go-shim/sigstore.go's `canonicalRepository`
/// folds *down*. They are doing different jobs — this one produces a
/// string to glob against, that one compares two references for equality
/// — and each is internally consistent, which is all either needs. What
/// they must agree on is which spellings are the same repository, and
/// `canonical_repository_agrees_with_the_go_side_on_what_is_equal` pins
/// that.
fn canonical_repository(repo: &str) -> String {
    for alias in ["index.docker.io", "registry-1.docker.io"] {
        if repo == alias {
            return "docker.io".to_string();
        }
        if let Some(rest) = repo.strip_prefix(&format!("{alias}/")) {
            return canonical_repository(&format!("docker.io/{rest}"));
        }
    }
    let (first, rest) = match repo.split_once('/') {
        Some((first, rest)) => (first, Some(rest)),
        None => (repo, None),
    };
    // Docker's own rule for telling a registry host from a namespace.
    let has_host = first.contains('.') || first.contains(':') || first == "localhost";
    if !has_host {
        return match rest {
            Some(rest) => format!("docker.io/{first}/{rest}"),
            None => format!("docker.io/library/{first}"),
        };
    }
    match repo.strip_prefix("docker.io/") {
        // An official image: "docker.io/ubuntu" is "docker.io/library/ubuntu".
        Some(path) if !path.contains('/') => format!("docker.io/library/{path}"),
        _ => repo.to_string(),
    }
}

/// Glob match where `*` stays within one path segment and `**` crosses
/// them — the semantics of `.gitignore` and shell globstar. So
/// `docker.io/org/*` matches `docker.io/org/model` but not
/// `docker.io/org/team/model`, and `docker.io/org/**` matches both.
///
/// Byte-wise and case-sensitive; registry references are lowercase by
/// grammar (`shortnames::parse_registry_ref` enforces it).
fn glob_match(pattern: &str, text: &str) -> bool {
    // Memoized on (pattern index, text index). Without it, a pattern of
    // repeated `*a` branches is exponential in its length, so a valid
    // config file could stall every policy decision.
    let mut seen = HashSet::new();
    glob_bytes(pattern.as_bytes(), text.as_bytes(), &mut seen)
}

fn glob_bytes(p: &[u8], t: &[u8], seen: &mut HashSet<(usize, usize)>) -> bool {
    if !seen.insert((p.len(), t.len())) {
        return false; // this state already failed
    }
    match p.first() {
        None => t.is_empty(),
        Some(b'*') => {
            if p.get(1) == Some(&b'*') {
                let rest = &p[2..];
                (0..=t.len()).any(|i| glob_bytes(rest, &t[i..], seen))
            } else {
                let rest = &p[1..];
                for i in 0..=t.len() {
                    if glob_bytes(rest, &t[i..], seen) {
                        return true;
                    }
                    // Having tried leaving the separator to the rest of
                    // the pattern, a single `*` may not swallow it.
                    if t.get(i) == Some(&b'/') {
                        break;
                    }
                }
                false
            }
        }
        Some(&c) => match t.first() {
            Some(&d) if c == d => glob_bytes(&p[1..], &t[1..], seen),
            _ => false,
        },
    }
}

// ---------------------------------------------------------------------------
// Applying the policy
// ---------------------------------------------------------------------------

/// What actually happened when a reference was checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Policy said `off`; nothing was checked.
    Skipped,
    /// A trusted key signed this exact manifest.
    Verified,
    /// Checked and not verified, under `warn`. The caller proceeds and
    /// must surface the accompanying notice.
    Unverified,
}

/// A completed check, plus what the user needs to be told about it.
///
/// Notices are returned rather than printed: `cmd::transfer` can print,
/// but a pull runs inside the daemon, whose stderr is a log file nobody
/// watches, so its notices are relayed over the progress stream instead.
#[derive(Debug, Clone)]
#[must_use = "a verification notice that is never surfaced is not a warning"]
pub struct Verdict {
    pub outcome: Outcome,
    pub notices: Vec<String>,
    /// The mode this verdict was reached under, so [`Verdict::escalate`]
    /// can treat a caller's own follow-up failure the way the policy
    /// would. Private: a caller that could set it to `Off` could silence
    /// its own escalation.
    mode: Mode,
    /// The digest a trusted key actually signed. `Some` only when
    /// `outcome` is `Verified`, and it comes from the verifier's own
    /// report rather than from a separate resolve — so a caller
    /// comparing against it is comparing against what was checked.
    pub digest: Option<String>,
}

impl Verdict {
    fn new(outcome: Outcome, mode: Mode) -> Self {
        Self {
            outcome,
            notices: Vec::new(),
            mode,
            digest: None,
        }
    }

    fn with(outcome: Outcome, mode: Mode, notice: String) -> Self {
        Self {
            outcome,
            notices: vec![notice],
            mode,
            digest: None,
        }
    }

    /// Reports `msg` the way `mode` says to: fatal under `enforce`,
    /// a notice under `warn`, silent when nothing was being checked.
    pub fn escalate(&mut self, msg: String) -> anyhow::Result<()> {
        match self.mode {
            Mode::Enforce => bail!("{msg}"),
            Mode::Warn => {
                self.notices.push(format!("warning: {msg}"));
                Ok(())
            }
            Mode::Off => Ok(()),
        }
    }

    /// Prints every notice, for a caller that is the foreground process.
    pub fn report(&self) {
        for notice in &self.notices {
            eprintln!("[llmman] {notice}");
        }
    }
}

/// A refusal that says nothing about the model itself — the registry was
/// unreachable, a key unreadable, no keys configured. Under `enforce` it
/// still refuses, but a caller holding a local copy must not treat it as
/// a verdict *against* that copy.
///
/// Wraps the real cause and displays it verbatim, so tagging an error
/// this way adds a marker without adding words to what the user reads.
#[derive(Debug)]
pub struct Indeterminate(anyhow::Error);

impl std::fmt::Display for Indeterminate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#}", self.0)
    }
}

impl std::error::Error for Indeterminate {}

/// Whether `err` from [`check`] means "could not find out" rather than
/// "definitely not signed by a trusted key".
pub fn is_indeterminate(err: &anyhow::Error) -> bool {
    err.chain().any(|e| e.is::<Indeterminate>())
}

/// Runs the policy for `reference` at `digest`.
///
/// `digest` is the manifest digest to check, or `None` to let the shim
/// resolve it. Pass it when known: it saves a round trip and closes the
/// window in which a tag could move between verification and use.
///
/// Returns `Err` only when the policy says the caller must not proceed.
/// "Unsigned" and "signed by a stranger" are ordinary outcomes under
/// `warn`, reported rather than raised.
pub fn check(reference: &str, digest: Option<&str>) -> anyhow::Result<Verdict> {
    // A policy that won't load is indeterminate, not a verdict: a
    // malformed config file says nothing about the model, and a caller
    // holding a local copy must not delete it over one.
    let policy = Policy::load().map_err(|e| anyhow!(Indeterminate(e)))?;
    check_with(reference, digest, &policy.decide(reference))
}

fn check_with(
    reference: &str,
    digest: Option<&str>,
    decision: &Decision,
) -> anyhow::Result<Verdict> {
    if decision.mode == Mode::Off {
        return Ok(Verdict::new(Outcome::Skipped, decision.mode));
    }
    if decision.keys.is_empty() {
        // `LLMMAN_VERIFY=warn|enforce` with no policy file, or a rule
        // naming no keys. Nothing to check against.
        let msg = format!(
            "verification is set to {} for {reference}, but no trusted public keys are configured for it \
             (add a [[verify.trust]] rule with `keys` to llmman.conf)",
            decision.mode
        );
        if decision.mode == Mode::Enforce {
            return Err(anyhow!(Indeterminate(anyhow!(msg))));
        }
        return Ok(Verdict::with(
            Outcome::Unverified,
            decision.mode,
            format!("warning: {msg}"),
        ));
    }

    let keys: Vec<String> = decision
        .keys
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();

    let report = match crate::ffi::verify(reference, digest.unwrap_or(""), &keys) {
        Ok(report) => report,
        // No answer could be reached (unreadable key, unreachable
        // registry). Under `enforce` that is indistinguishable from
        // "not verified" and must fail closed.
        Err(e) => {
            if decision.mode == Mode::Enforce {
                return Err(anyhow!(Indeterminate(e)).context(format!("cannot verify {reference}")));
            }
            return Ok(Verdict::with(
                Outcome::Unverified,
                decision.mode,
                format!("warning: cannot verify {reference}: {e:#}"),
            ));
        }
    };

    if report.verified {
        let signers: Vec<&str> = report.matches.iter().map(|m| m.key_path.as_str()).collect();
        let mut verdict = Verdict::with(
            Outcome::Verified,
            decision.mode,
            format!(
                "verified signature on {reference} ({}) with {}",
                short_digest(&report.digest),
                signers.join(", ")
            ),
        );
        verdict.digest = Some(report.digest);
        return Ok(verdict);
    }

    let detail = if report.reason.is_empty() {
        "no trusted key accepted a signature for it".to_string()
    } else {
        report.reason.clone()
    };
    let msg = format!(
        "{reference} ({}) is not signed by a trusted key: {detail}",
        short_digest(&report.digest)
    );
    if decision.mode == Mode::Enforce {
        bail!("{msg}");
    }
    Ok(Verdict::with(
        Outcome::Unverified,
        decision.mode,
        format!("warning: {msg}"),
    ))
}

/// `reference` rewritten to name `digest` explicitly, so a subsequent
/// resolve cannot land on anything else.
///
/// Verification resolves a tag to decide whether to trust it; whatever
/// acts on that decision has to be pinned to the same manifest, or the
/// tag can move in between and the check applies to bytes nobody used.
pub fn pin_to_digest(reference: &str, digest: &str) -> String {
    format!("{}@{digest}", repository_of(reference))
}

/// Whether `reference` names an OCI registry, decided without touching
/// the network.
///
/// `crate::hf::classify` answers this better — but only by probing
/// `/v2/`, and a probe that *fails* classifies a real registry reference
/// as HuggingFace. Deciding policy on that would take a stored model out
/// of the policy's reach exactly when the network is being interfered
/// with, which is fail-open. This errs the other way: anything not
/// recognizably one of the non-registry forms is treated as a registry,
/// so an unknown host gets checked rather than waved through.
pub fn is_registry_reference(reference: &str) -> bool {
    if crate::sources::handles(reference) {
        return false;
    }
    if ["hf://", "huggingface://"]
        .iter()
        .any(|p| reference.starts_with(p))
    {
        return false;
    }
    let host = reference.split('/').next().unwrap_or(reference);
    !crate::hf::is_known_hf_host(host)
}

/// Whether any checking would happen for `reference` at all. Lets a
/// caller skip work on the overwhelmingly common no-policy path without
/// duplicating the decision.
pub fn is_enabled_for(reference: &str) -> anyhow::Result<bool> {
    Ok(Policy::load()?.decide(reference).mode != Mode::Off)
}

/// First 12 hex characters of a digest, matching the Go shim's output.
fn short_digest(digest: &str) -> String {
    let hex = digest.split_once(':').map(|(_, h)| h).unwrap_or(digest);
    hex.chars().take(12).collect()
}

/// Whether two digest strings name the same content. Case-insensitive,
/// matching `OciStore::find`, so a spelling difference between the
/// registry's answer and the store's cannot look like a moved tag.
fn digests_match(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

// ---------------------------------------------------------------------------
// Pull-time enforcement
// ---------------------------------------------------------------------------

/// Enforces the policy across a pull, in two halves: [`check`] before any
/// layer is fetched, so a rejected model costs one manifest lookup rather
/// than a multi-gigabyte download, and [`confirm`] afterwards, so a tag
/// repointed in between cannot slip past. Neither is useful alone.
///
/// [`check`]: PullGuard::check
/// [`confirm`]: PullGuard::confirm
#[must_use = "a checked pull must be confirmed against the digest it actually stored"]
pub struct PullGuard {
    reference: String,
    mode: Mode,
    /// The digest a trusted key signed, straight from the verifier's
    /// report. `None` means nothing was verified, and `confirm` has
    /// nothing to compare against.
    verified_digest: Option<String>,
    notices: Vec<String>,
}

impl PullGuard {
    /// Runs the policy before the pull starts. `Err` if it forbids
    /// proceeding.
    pub fn check(reference: &str) -> anyhow::Result<Self> {
        let decision = Policy::load()?.decide(reference);
        let mode = decision.mode;
        if mode == Mode::Off {
            return Ok(Self {
                reference: reference.to_owned(),
                mode,
                verified_digest: None,
                notices: Vec::new(),
            });
        }
        // Resolving here only saves the shim a round trip; its failure
        // is not fatal, because the digest that matters comes back in
        // the verdict either way.
        let resolved = crate::ffi::resolved_digest_of(reference).ok();
        let verdict = check_with(reference, resolved.as_deref(), &decision)?;

        // Under `enforce`, a pass with nothing to confirm against would
        // silently drop the second half of the guard.
        if mode == Mode::Enforce && verdict.outcome == Outcome::Verified && verdict.digest.is_none()
        {
            bail!("{reference} verified but reported no digest to confirm against");
        }
        Ok(Self {
            reference: reference.to_owned(),
            mode,
            verified_digest: verdict.digest,
            notices: verdict.notices,
        })
    }

    /// Confirms `stored` — the digest the pull landed on — is the digest
    /// that was verified, and returns the notices from both halves.
    ///
    /// A mismatch means the tag moved mid-pull: under `enforce` the
    /// bytes now in the store were never verified, whatever was verified
    /// a moment ago.
    pub fn confirm(mut self, stored: &str) -> anyhow::Result<Vec<String>> {
        let Some(verified) = self.verified_digest.take() else {
            return Ok(self.notices);
        };
        if digests_match(&verified, stored) {
            return Ok(self.notices);
        }
        let msg = format!(
            "{} changed while being pulled: verified {}, stored {}",
            self.reference,
            short_digest(&verified),
            short_digest(stored)
        );
        if self.mode == Mode::Enforce {
            bail!("{msg}");
        }
        self.notices.push(format!("warning: {msg}"));
        Ok(self.notices)
    }
}

// ---------------------------------------------------------------------------
// Signing keys
// ---------------------------------------------------------------------------

/// Passphrase for a `--sign-key`, from `LLMMAN_SIGN_PASSWORD` or
/// cosign's own `COSIGN_PASSWORD`. Ignored for an unencrypted PEM.
pub fn signing_password() -> String {
    std::env::var("LLMMAN_SIGN_PASSWORD")
        .or_else(|_| std::env::var("COSIGN_PASSWORD"))
        .unwrap_or_default()
}

/// Checks a `--sign-key` is readable and returns its absolute path.
///
/// Opened, not stat-ed, so a directory or a bad mode fails here rather
/// than after a multi-gigabyte upload. Absolute because the daemon
/// resolves it in its own working directory, not the client's.
pub fn check_signing_key(path: &str) -> anyhow::Result<PathBuf> {
    let meta =
        std::fs::metadata(path).with_context(|| format!("cannot read signing key {path}"))?;
    if !meta.is_file() {
        bail!("signing key {path} is not a file");
    }
    std::fs::File::open(path).with_context(|| format!("cannot read signing key {path}"))?;
    dunce::canonicalize(path).with_context(|| format!("resolve signing key {path}"))
}

/// Every distinct trusted key the policy names, for `llmman verify` to
/// fall back on when no `--key` was given and no rule matched.
pub fn all_configured_keys() -> anyhow::Result<Vec<PathBuf>> {
    let mut seen = HashSet::new();
    let mut keys: Vec<PathBuf> = Policy::load()?
        .rules
        .iter()
        .flat_map(|r| r.keys.iter())
        .filter(|k| seen.insert((*k).clone()))
        .cloned()
        .collect();
    keys.sort();
    Ok(keys)
}

/// Exposed for `Policy::decide` callers that already hold a reference.
pub fn decide(reference: &str) -> anyhow::Result<Decision> {
    Ok(Policy::load()?.decide(reference))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(text: &str) -> Policy {
        Policy::parse(text, Path::new("/etc/llmman")).expect("parse policy")
    }

    /// The other tests hand `Policy` a `[verify]` section on its own.
    /// This one goes the way the daemon does, through a whole
    /// `llmman.conf`, so the two cannot drift into a policy that parses
    /// in tests and is silently absent in production.
    #[test]
    fn a_nested_verify_section_of_a_whole_llmman_conf_becomes_a_policy() {
        let conf = crate::config::parse(
            r#"
            [aliases]
            gemma4 = "docker.io/ai/gemma4"

            [providers.openai]
            api_key = "sk-x"

            [verify]
            default = "warn"

            [[verify.trust]]
            pattern = "docker.io/myorg/**"
            keys    = ["keys/myorg.pub"]
            mode    = "enforce"
            "#,
        )
        .expect("valid llmman.conf");

        let policy =
            Policy::from_conf(&conf.verify, Path::new("/etc/llmman")).expect("valid policy");

        // The rule applies where it matches, keys resolved against the
        // file's own directory...
        let decision = policy.decide("docker.io/myorg/model:v1");
        assert_eq!(decision.mode, Mode::Enforce);
        assert_eq!(
            decision.keys,
            vec![PathBuf::from("/etc/llmman/keys/myorg.pub")]
        );
        // ...and `default` covers everything it does not.
        assert_eq!(policy.decide("docker.io/other/model").mode, Mode::Warn);
    }

    #[test]
    fn repository_of_strips_tags_and_digests() {
        assert_eq!(
            repository_of("docker.io/org/model:v1"),
            "docker.io/org/model"
        );
        assert_eq!(repository_of("docker.io/org/model"), "docker.io/org/model");
        assert_eq!(
            repository_of("docker.io/org/model@sha256:abc"),
            "docker.io/org/model"
        );
        assert_eq!(
            repository_of("docker.io/org/model:v1@sha256:abc"),
            "docker.io/org/model"
        );
    }

    #[test]
    fn repository_of_keeps_a_registry_port() {
        // The colon precedes the last slash, so it is a port, not a tag.
        assert_eq!(
            repository_of("registry.example.com:5000/org/model:v1"),
            "registry.example.com:5000/org/model"
        );
    }

    #[test]
    fn canonical_repository_agrees_with_the_go_side_on_what_is_equal() {
        // Different normal forms (this one qualifies, Go's strips), but
        // they must fold the same spellings together.
        let want = canonical_repository("docker.io/ai/gemma4");
        assert_eq!(want, "docker.io/ai/gemma4");
        for spelling in [
            "index.docker.io/ai/gemma4",
            "registry-1.docker.io/ai/gemma4",
            "ai/gemma4",
        ] {
            assert_eq!(canonical_repository(spelling), want, "{spelling}");
        }
        // Official images, both spellings.
        assert_eq!(
            canonical_repository("ubuntu"),
            canonical_repository("docker.io/ubuntu")
        );
        assert_eq!(
            canonical_repository("ubuntu"),
            canonical_repository("docker.io/library/ubuntu")
        );
        // A bare host with no path still folds — the alias check must
        // not require a trailing "/" to fire.
        assert_eq!(
            canonical_repository("index.docker.io"),
            canonical_repository("docker.io")
        );
        // Other hosts are left exactly alone.
        assert_eq!(canonical_repository("quay.io/org/m"), "quay.io/org/m");
        assert_eq!(
            canonical_repository("registry.example.com:5000/org/m"),
            "registry.example.com:5000/org/m"
        );
    }

    #[test]
    fn a_docker_hub_rule_does_not_leak_onto_other_registries() {
        // Regression: canonicalizing the *pattern* stripped its host, so
        // `docker.io/**` became `**` and matched every registry — an
        // `off` rule for Hub would have disabled verification globally.
        let p = policy(
            r#"
            [[trust]]
            pattern = "docker.io/**"
            keys = ["/hub.pub"]
            mode = "enforce"
        "#,
        );
        assert_eq!(p.decide("docker.io/ai/gemma4:latest").mode, Mode::Enforce);
        for elsewhere in [
            "quay.io/other/model:v1",
            "registry.internal.example.com/team/secret:v1",
            "ghcr.io/a/b",
        ] {
            assert_eq!(p.decide(elsewhere).mode, Mode::Off, "{elsewhere}");
        }
    }

    #[test]
    fn a_docker_hub_rule_matches_every_spelling() {
        let p = policy(
            r#"
            [[trust]]
            pattern = "docker.io/ai/*"
            keys = ["/k.pub"]
            mode = "enforce"
        "#,
        );
        for reference in [
            "docker.io/ai/gemma4:latest",
            "index.docker.io/ai/gemma4:latest",
            "ai/gemma4",
        ] {
            assert_eq!(p.decide(reference).mode, Mode::Enforce, "{reference}");
        }
    }

    // -- glob ------------------------------------------------------------

    #[test]
    fn glob_matches_literals_exactly() {
        assert!(glob_match("docker.io/org/model", "docker.io/org/model"));
        assert!(!glob_match("docker.io/org/model", "docker.io/org/model2"));
    }

    #[test]
    fn single_star_stays_within_one_path_segment() {
        assert!(glob_match("docker.io/org/*", "docker.io/org/model"));
        assert!(glob_match("docker.io/*/model", "docker.io/org/model"));
        assert!(!glob_match("docker.io/org/*", "docker.io/org/team/model"));
    }

    #[test]
    fn double_star_crosses_path_segments() {
        assert!(glob_match("docker.io/**", "docker.io/org/model"));
        assert!(glob_match("docker.io/**", "docker.io/org/team/model"));
        assert!(glob_match("docker.io/**/model", "docker.io/org/team/model"));
        assert!(!glob_match("quay.io/**", "docker.io/org/model"));
    }

    #[test]
    fn glob_is_anchored_at_both_ends() {
        assert!(glob_match("docker.io/org/model*", "docker.io/org/model"));
        assert!(!glob_match("docker.io", "docker.io/org/model"));
    }

    #[test]
    fn glob_does_not_blow_up_on_a_pathological_pattern() {
        // The trailing "b" is what makes this the bad case: every one of
        // the 24 `*a` branches has to be explored before the match can
        // be ruled out. Unmemoized that is exponential and this test
        // does not finish; the assertion is really just the clock.
        let pattern = "*a".repeat(24) + "b";
        let text = "a".repeat(64);
        assert!(!glob_match(&pattern, &text));
        // Still correct on the matching side.
        assert!(glob_match(&("*a".repeat(24)), &text));
    }

    // -- rule selection --------------------------------------------------

    #[test]
    fn no_config_means_off() {
        assert_eq!(
            Policy::default().decide("docker.io/org/model:v1").mode,
            Mode::Off
        );
    }

    #[test]
    fn unmatched_reference_falls_back_to_the_default_mode() {
        let p = policy(
            r#"
            default = "warn"
            [[trust]]
            pattern = "docker.io/org/**"
            keys = ["/k.pub"]
            mode = "enforce"
        "#,
        );
        assert_eq!(p.decide("quay.io/other/model:v1").mode, Mode::Warn);
        assert!(p.decide("quay.io/other/model:v1").keys.is_empty());
    }

    #[test]
    fn the_last_matching_rule_wins_and_replaces_its_keys() {
        let p = policy(
            r#"
            [[trust]]
            pattern = "docker.io/**"
            keys = ["/broad.pub"]
            mode = "enforce"

            [[trust]]
            pattern = "docker.io/org/experimental"
            keys = ["/narrow.pub"]
            mode = "warn"
        "#,
        );
        let broad = p.decide("docker.io/org/model:v1");
        assert_eq!(broad.mode, Mode::Enforce);
        assert_eq!(broad.keys, vec![PathBuf::from("/broad.pub")]);

        let narrow = p.decide("docker.io/org/experimental:v1");
        assert_eq!(narrow.mode, Mode::Warn);
        // Replaced, not added to, so "trust the org key everywhere
        // except here" is expressible.
        assert_eq!(narrow.keys, vec![PathBuf::from("/narrow.pub")]);
    }

    #[test]
    fn a_rule_matches_regardless_of_tag() {
        let p = policy(
            r#"
            [[trust]]
            pattern = "docker.io/org/model"
            keys = ["/k.pub"]
            mode = "enforce"
        "#,
        );
        for reference in [
            "docker.io/org/model",
            "docker.io/org/model:v1",
            "docker.io/org/model@sha256:abc",
        ] {
            assert_eq!(p.decide(reference).mode, Mode::Enforce, "{reference}");
        }
    }

    // -- parsing ---------------------------------------------------------

    #[test]
    fn a_rule_without_an_explicit_mode_warns() {
        let p = policy("[[trust]]\npattern = \"docker.io/org/**\"\nkeys = [\"/k.pub\"]\n");
        assert_eq!(p.decide("docker.io/org/model").mode, Mode::Warn);
    }

    #[test]
    fn an_explicit_default_off_overrides_a_lower_priority_file() {
        // "later files win" has to include downgrades, or a per-user
        // file could never relax a system one.
        let lower = policy("default = \"warn\"\n");
        let upper = policy("default = \"off\"\n");
        assert_eq!(lower.default_mode, Some(Mode::Warn));
        assert_eq!(upper.default_mode, Some(Mode::Off));
        // A file that says nothing must not override either.
        assert_eq!(policy("[[trust]]\npattern=\"a/b\"\n").default_mode, None);
    }

    #[test]
    fn mode_spellings_are_accepted_case_insensitively() {
        assert_eq!(Mode::parse("Off").unwrap(), Mode::Off);
        assert_eq!(Mode::parse("  WARN ").unwrap(), Mode::Warn);
        assert_eq!(Mode::parse("require").unwrap(), Mode::Enforce);
        assert!(Mode::parse("maybe").is_err());
    }

    #[test]
    fn relative_key_paths_resolve_against_the_config_file() {
        let p = Policy::parse(
            "[[trust]]\npattern = \"docker.io/org/**\"\nkeys = [\"keys/org.pub\", \"/abs/other.pub\"]\n",
            Path::new("/etc/llmman"),
        )
        .unwrap();
        assert_eq!(
            p.decide("docker.io/org/model").keys,
            vec![
                PathBuf::from("/etc/llmman/keys/org.pub"),
                PathBuf::from("/abs/other.pub"),
            ]
        );
    }

    #[test]
    fn an_unusable_pattern_is_rejected_rather_than_silently_never_matching() {
        for pattern in [
            "",
            // A tag or digest can never match, since patterns are
            // compared against the repository.
            "docker.io/org/model:v1",
            "docker.io/org/model@sha256:abc",
        ] {
            assert!(validate_pattern(pattern).is_err(), "accepted {pattern:?}");
        }
        assert!(validate_pattern(&"a".repeat(MAX_PATTERN_LEN + 1)).is_err());
        // A port is not a tag.
        assert!(validate_pattern("registry.example.com:5000/org/*").is_ok());
    }

    #[test]
    fn an_unknown_mode_is_rejected_rather_than_defaulted() {
        assert!(Policy::parse(
            "[[trust]]\npattern=\"a/b\"\nmode=\"enfroce\"\n",
            Path::new("/x")
        )
        .is_err());
        assert!(Policy::parse("default = \"nope\"\n", Path::new("/x")).is_err());
    }

    // -- check_with ------------------------------------------------------

    #[test]
    fn off_skips_without_calling_the_verifier() {
        let decision = Decision {
            mode: Mode::Off,
            keys: vec![PathBuf::from("/k.pub")],
        };
        let verdict = check_with("docker.io/org/model:v1", None, &decision).unwrap();
        assert_eq!(verdict.outcome, Outcome::Skipped);
        assert!(verdict.notices.is_empty());
    }

    #[test]
    fn enforce_without_keys_fails_closed() {
        let decision = Decision {
            mode: Mode::Enforce,
            keys: vec![],
        };
        let err = check_with("docker.io/org/model:v1", None, &decision).unwrap_err();
        assert!(
            err.to_string().contains("no trusted public keys"),
            "{err:#}"
        );
    }

    #[test]
    fn warn_without_keys_reports_and_continues() {
        let decision = Decision {
            mode: Mode::Warn,
            keys: vec![],
        };
        let verdict = check_with("docker.io/org/model:v1", None, &decision).unwrap();
        assert_eq!(verdict.outcome, Outcome::Unverified);
        // `warn` must produce something the caller can actually show.
        assert_eq!(verdict.notices.len(), 1);
        assert!(verdict.notices[0].contains("no trusted public keys"));
    }

    // -- PullGuard -------------------------------------------------------

    fn guard(mode: Mode, verified: Option<&str>) -> PullGuard {
        PullGuard {
            reference: "docker.io/org/model:v1".into(),
            mode,
            verified_digest: verified.map(str::to_owned),
            notices: Vec::new(),
        }
    }

    #[test]
    fn confirm_is_a_no_op_when_nothing_was_verified() {
        let mut g = guard(Mode::Warn, None);
        g.notices.push("warning: something".into());
        assert_eq!(g.confirm("sha256:whatever").unwrap().len(), 1);
    }

    #[test]
    fn confirm_accepts_the_digest_that_was_verified() {
        assert!(guard(Mode::Enforce, Some("sha256:aaa"))
            .confirm("sha256:aaa")
            .is_ok());
    }

    #[test]
    fn confirm_ignores_digest_case() {
        // OciStore::find compares case-insensitively; a spelling
        // difference must not look like a moved tag.
        assert!(guard(Mode::Enforce, Some("sha256:ABC"))
            .confirm("sha256:abc")
            .is_ok());
    }

    #[test]
    fn confirm_rejects_a_tag_that_moved_mid_pull_under_enforce() {
        let err = guard(Mode::Enforce, Some("sha256:aaa"))
            .confirm("sha256:bbb")
            .unwrap_err();
        assert!(
            err.to_string().contains("changed while being pulled"),
            "{err:#}"
        );
    }

    #[test]
    fn confirm_only_warns_about_a_moved_tag_under_warn() {
        let notices = guard(Mode::Warn, Some("sha256:aaa"))
            .confirm("sha256:bbb")
            .unwrap();
        assert_eq!(notices.len(), 1);
        assert!(notices[0].contains("changed while being pulled"));
    }

    #[test]
    fn an_invalid_llmman_verify_is_an_error_not_a_silent_downgrade() {
        // Read inside the daemon, whose stderr nobody watches, so
        // warning and carrying on would turn a typo into a downgrade.
        let err = parse_mode_override(Some("enfroce"))
            .expect_err("a typo'd LLMMAN_VERIFY must not be ignored");
        assert!(err.to_string().contains("LLMMAN_VERIFY"), "{err:#}");

        assert_eq!(parse_mode_override(None).unwrap(), None);
        assert_eq!(parse_mode_override(Some("  ")).unwrap(), None);
        assert_eq!(
            parse_mode_override(Some("enforce")).unwrap(),
            Some(Mode::Enforce)
        );
    }

    #[test]
    fn escalate_follows_the_mode_it_was_decided_under() {
        // `cmd::transfer` confirms the digest it actually transferred
        // and must treat a mismatch the way the policy would.
        let mut warn = Verdict::new(Outcome::Verified, Mode::Warn);
        assert!(warn.escalate("moved".into()).is_ok());
        assert_eq!(warn.notices, vec!["warning: moved".to_string()]);

        let mut enforce = Verdict::new(Outcome::Verified, Mode::Enforce);
        assert!(enforce.escalate("moved".into()).is_err());

        let mut off = Verdict::new(Outcome::Skipped, Mode::Off);
        assert!(off.escalate("moved".into()).is_ok());
        assert!(off.notices.is_empty());
    }

    #[test]
    fn check_signing_key_rejects_what_cannot_be_signed_with() {
        let dir = std::env::temp_dir().join(format!("llmman-key-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        assert!(
            check_signing_key(dir.to_str().unwrap()).is_err(),
            "a directory"
        );
        assert!(
            check_signing_key(&dir.join("absent").to_string_lossy()).is_err(),
            "a missing file"
        );

        let key = dir.join("k.pem");
        std::fs::write(&key, b"not really a key").unwrap();
        // Returns an absolute path: the daemon resolves it in its own
        // working directory, not the client's.
        let resolved = check_signing_key(key.to_str().unwrap()).unwrap();
        assert!(resolved.is_absolute());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pin_to_digest_replaces_a_tag_rather_than_appending_to_it() {
        let d = "sha256:abc";
        assert_eq!(
            pin_to_digest("docker.io/org/m:v1", d),
            "docker.io/org/m@sha256:abc"
        );
        assert_eq!(
            pin_to_digest("docker.io/org/m", d),
            "docker.io/org/m@sha256:abc"
        );
        // An already-pinned reference is repinned, not doubled up.
        assert_eq!(
            pin_to_digest("docker.io/org/m@sha256:old", d),
            "docker.io/org/m@sha256:abc"
        );
        // A registry port survives.
        assert_eq!(
            pin_to_digest("registry.example.com:5000/org/m:v1", d),
            "registry.example.com:5000/org/m@sha256:abc"
        );
    }

    #[test]
    fn an_unknown_config_key_is_rejected_rather_than_ignored() {
        // `defualt` or `[[turst]]` would otherwise load as an off
        // policy — the same silent downgrade a typo'd mode would be.
        for text in [
            "defualt = \"enforce\"\n",
            "[[turst]]\npattern = \"docker.io/**\"\n",
            "[[trust]]\npattern = \"docker.io/**\"\nkyes = [\"/k.pub\"]\n",
        ] {
            assert!(
                Policy::parse(text, Path::new("/etc/llmman")).is_err(),
                "accepted {text:?}"
            );
        }
    }

    #[test]
    fn is_registry_reference_fails_closed_on_an_unknown_host() {
        // The whole point: an unreachable or hostile network must not be
        // able to reclassify a registry reference out of the policy's
        // reach, so anything not recognizably non-registry counts.
        for reference in [
            "docker.io/org/model:v1",
            "quay.io/org/model",
            "registry.internal.example.com:5000/team/m",
            // Not a registry at all, but unknowable without a probe —
            // treated as one, so it gets checked and refused rather
            // than silently skipped.
            "some-unknown-host.example/org/m",
        ] {
            assert!(is_registry_reference(reference), "{reference}");
        }
        for reference in [
            "hf.co/org/model",
            "huggingface.co/org/model",
            "modelscope.cn/org/model",
            "hf://org/model",
            "huggingface://org/model",
            "ms://org/model",
            "ngc://org/model",
            "s3://bucket/key",
            "gs://bucket/key",
            "/local/path/to/model",
        ] {
            assert!(!is_registry_reference(reference), "{reference}");
        }
    }

    #[test]
    fn short_digest_abbreviates_like_the_shim_does() {
        assert_eq!(short_digest("sha256:0123456789abcdef0123"), "0123456789ab");
        assert_eq!(short_digest("abc"), "abc");
    }
}

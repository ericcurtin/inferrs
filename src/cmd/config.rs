//! `llmman config` — read and write `llmman.conf` without an editor.
//!
//! `git config` verb for verb, with `--system`/`--user`/`--file`
//! picking one of [`crate::config`]'s locations. Reads see every file
//! merged, later overriding earlier; writes go to the per-user file.
//!
//! Keys are TOML dotted keys, quoted as the file quotes them:
//! `providers."wafer.ai".api_key`. Values are always written as
//! strings, which is every field the format has, and `set`/`unset`
//! cannot reach inside `[[verify.trust]]` — that is what `edit` is for.
//!
//! Every write goes through [`crate::config::parse`] first. That is the
//! point of the command: the format is `deny_unknown_fields`, so a
//! hand-edited `api_kye` is a credential that never takes effect and
//! surfaces later as someone else's 401.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Subcommand};
use toml_edit::{DocumentMut, Item, Key, Table, TableLike, Value};

#[derive(Args, Debug)]
pub struct ConfigArgs {
    #[command(subcommand)]
    pub command: ConfigCommand,

    /// Act on the system-wide file (/etc/llmman/llmman.conf)
    #[arg(long, global = true, conflicts_with_all = ["user", "file"])]
    pub system: bool,

    /// Act on the per-user file (~/.config/llmman/llmman.conf), which is
    /// already where `set`, `unset` and `edit` write
    #[arg(long, global = true, conflicts_with = "file")]
    pub user: bool,

    /// Act on this file instead of the configured locations
    #[arg(long, global = true, value_name = "PATH")]
    pub file: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum ConfigCommand {
    /// Print every key that is set, one `key=value` per line
    List(ListArgs),
    /// Print the value of one key, or exit non-zero if it is not set
    Get(GetArgs),
    /// Set one key to a string value
    Set(SetArgs),
    /// Remove one key
    Unset(UnsetArgs),
    /// Open the file in $VISUAL or $EDITOR, then check that it parses
    Edit,
}

#[derive(Args, Debug)]
pub struct ListArgs {
    /// Prefix each line with the file the key came from
    #[arg(long)]
    pub show_origin: bool,
}

#[derive(Args, Debug)]
pub struct GetArgs {
    /// A dotted key, exactly as `llmman config list` prints it
    #[arg(value_name = "KEY")]
    pub key: String,
}

#[derive(Args, Debug)]
pub struct SetArgs {
    /// A dotted key, e.g. `aliases.gemma4` or `providers.openai.api_key`
    #[arg(value_name = "KEY")]
    pub key: String,
    /// The value to store, always written as a TOML string
    #[arg(value_name = "VALUE")]
    pub value: String,
}

#[derive(Args, Debug)]
pub struct UnsetArgs {
    /// A dotted key, e.g. `providers.openai.api_key`
    #[arg(value_name = "KEY")]
    pub key: String,
}

pub fn run(args: &ConfigArgs) -> Result<()> {
    match &args.command {
        ConfigCommand::List(a) => list(args, a),
        ConfigCommand::Get(a) => get(args, a),
        ConfigCommand::Set(a) => set(args, a),
        ConfigCommand::Unset(a) => unset(args, a),
        ConfigCommand::Edit => edit(args),
    }
}

// ---------------------------------------------------------------------------
// Which file
// ---------------------------------------------------------------------------

impl ConfigArgs {
    /// The one file the flags name, or `None` for "all of them".
    fn selected(&self) -> Result<Option<PathBuf>> {
        if let Some(file) = &self.file {
            return Ok(Some(file.clone()));
        }
        if self.system {
            return Ok(Some(crate::config::system_path()));
        }
        if self.user {
            return Ok(Some(user_path()?));
        }
        Ok(None)
    }

    /// Ascending priority, so the last value printed wins.
    fn read_paths(&self) -> Result<Vec<PathBuf>> {
        Ok(match self.selected()? {
            Some(path) => vec![path],
            None => crate::config::search_paths(),
        })
    }

    /// The per-user file by default: `/etc` is admin-managed and needs
    /// an explicit `--system`.
    fn write_path(&self) -> Result<PathBuf> {
        match self.selected()? {
            Some(path) => Ok(path),
            None => user_path(),
        }
    }
}

fn user_path() -> Result<PathBuf> {
    crate::config::user_path()
        .ok_or_else(|| anyhow!("no home directory to hold llmman.conf; use --file"))
}

// ---------------------------------------------------------------------------
// The key space
// ---------------------------------------------------------------------------

/// Every scalar in `doc`, as the dotted key naming it and its value.
///
/// The whole key space: `get` looks its argument up in what `list`
/// prints, so the two cannot disagree about what a key is called.
/// Indices name array elements, making `[[verify.trust]]` readable
/// without inventing a notation.
fn entries(doc: &DocumentMut) -> Vec<(String, String)> {
    let mut out = Vec::new();
    walk_item(doc.as_item(), "", &mut out);
    out
}

fn walk_item(item: &Item, prefix: &str, out: &mut Vec<(String, String)>) {
    match item {
        Item::Table(table) => walk_table(table, prefix, out),
        Item::ArrayOfTables(tables) => {
            for (i, table) in tables.iter().enumerate() {
                walk_table(table, &format!("{prefix}[{i}]"), out);
            }
        }
        Item::Value(value) => walk_value(value, prefix, out),
        Item::None => {}
    }
}

fn walk_table(table: &Table, prefix: &str, out: &mut Vec<(String, String)>) {
    for (key, item) in table.iter() {
        walk_item(item, &join(prefix, key), out);
    }
}

fn walk_value(value: &Value, prefix: &str, out: &mut Vec<(String, String)>) {
    match value {
        Value::InlineTable(table) => {
            for (key, value) in table.iter() {
                walk_value(value, &join(prefix, key), out);
            }
        }
        Value::Array(array) => {
            for (i, value) in array.iter().enumerate() {
                walk_value(value, &format!("{prefix}[{i}]"), out);
            }
        }
        // Unquoted: the quotes are TOML's, not part of the value.
        Value::String(s) => out.push((prefix.to_string(), s.value().clone())),
        other => out.push((prefix.to_string(), other.to_string().trim().to_string())),
    }
}

/// Append one key to a dotted path, quoted if TOML would quote it.
fn join(prefix: &str, key: &str) -> String {
    let key = Key::new(key).display_repr().into_owned();
    if prefix.is_empty() {
        key
    } else {
        format!("{prefix}.{key}")
    }
}

/// Split by TOML's rules, so `providers."wafer.ai"` is the two keys the
/// file spells that way rather than three. Rebuilt from the decoded
/// text: a parsed key carries its own whitespace, and `aliases.qwen`
/// would land as `qwen="…"` among lines reading `gemma4 = "…"`.
fn parse_key(name: &str) -> Result<Vec<Key>> {
    let keys = Key::parse(name).map_err(|e| anyhow!("invalid key `{name}`: {e}"))?;
    if keys.is_empty() {
        bail!("invalid key `{name}`: a key cannot be empty");
    }
    Ok(keys.iter().map(|key| Key::new(key.get())).collect())
}

fn is_credential(key: &str) -> bool {
    key == "api_key" || key.ends_with(".api_key") || key == "auth.api_keys"
}

/// Whether the document holds a secret — see [`write_atomic`].
fn carries_key(doc: &DocumentMut) -> bool {
    entries(doc)
        .iter()
        .any(|(key, value)| is_credential(key) && !value.trim().is_empty())
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Empty when the file is absent — the common case.
fn read(path: &Path) -> Result<DocumentMut> {
    match std::fs::read_to_string(path) {
        Ok(text) => text
            .parse::<DocumentMut>()
            .with_context(|| format!("parsing {}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(DocumentMut::new()),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// [`read`], warning where the file's keys would be dropped at load
/// time, so a read cannot report a value that will never be spent.
fn read_effective(path: &Path) -> Result<DocumentMut> {
    let doc = read(path)?;
    if carries_key(&doc) {
        if let Err(e) = crate::config::owner_readable_only(path) {
            eprintln!(
                "[llmman] warning: the API keys in {} are ignored: {e}",
                path.display()
            );
        }
    }
    Ok(doc)
}

/// A credential prints as `<redacted>`: this is the command whose
/// output ends up in a bug report. `get` names one key and prints it.
fn list(args: &ConfigArgs, opts: &ListArgs) -> Result<()> {
    for path in args.read_paths()? {
        for (key, value) in entries(&read_effective(&path)?) {
            let value = if is_credential(&key) {
                "<redacted>"
            } else {
                &value
            };
            if opts.show_origin {
                println!("{}\t{key}={value}", path.display());
            } else {
                println!("{key}={value}");
            }
        }
    }
    Ok(())
}

/// The last value set, across files and within a file — what llmman
/// would use. Except an array index: `[[verify.trust]]` is appended
/// across files rather than replaced, so `verify.trust[0]` names one
/// per file and `list --show-origin` tells them apart.
fn get(args: &ConfigArgs, opts: &GetArgs) -> Result<()> {
    let mut found = None;
    for path in args.read_paths()? {
        if let Some((_, value)) = entries(&read_effective(&path)?)
            .into_iter()
            .rev()
            .find(|(key, _)| *key == opts.key)
        {
            found = Some(value);
        }
    }
    match found {
        Some(value) => {
            println!("{value}");
            Ok(())
        }
        None => bail!("{}: not set", opts.key),
    }
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

fn set(args: &ConfigArgs, opts: &SetArgs) -> Result<()> {
    let path = args.write_path()?;
    let mut doc = read(&path)?;
    set_in(&mut doc, &opts.key, &opts.value)?;
    save(&path, &doc)
}

fn unset(args: &ConfigArgs, opts: &UnsetArgs) -> Result<()> {
    let path = args.write_path()?;
    let mut doc = read(&path)?;
    unset_in(&mut doc, &opts.key)?;
    save(&path, &doc)
}

/// Split out from [`set`] to be testable without a file, a home
/// directory, or a particular umask.
fn set_in(doc: &mut DocumentMut, name: &str, value: &str) -> Result<()> {
    let keys = parse_key(name)?;
    let (leaf, parents) = keys.split_last().expect("parse_key rejects an empty key");

    // `as_table_like_mut`: `aliases = { gemma4 = "…" }` is valid, and
    // a key `list` prints has to be one `set` can reach.
    let mut table: &mut dyn TableLike = doc.as_table_mut();
    for key in parents {
        table = table
            .entry_format(key)
            .or_insert_with(|| {
                let mut new = Table::new();
                // One `[providers.openai]` header, not an empty
                // `[providers]` above it.
                new.set_implicit(true);
                Item::Table(new)
            })
            .as_table_like_mut()
            .ok_or_else(|| anyhow!("{name}: {} is not a table", key.display_repr()))?;
    }

    let item = table.entry_format(leaf).or_insert(Item::None);
    let mut replacement = toml_edit::value(value);
    // Keep the old line's spacing and trailing comment.
    if let (Some(old), Some(new)) = (item.as_value(), replacement.as_value_mut()) {
        *new.decor_mut() = old.decor().clone();
    }
    *item = replacement;
    Ok(())
}

/// Split out from [`unset`] for the same reason as [`set_in`].
fn unset_in(doc: &mut DocumentMut, name: &str) -> Result<()> {
    let keys = parse_key(name)?;
    let (leaf, parents) = keys.split_last().expect("parse_key rejects an empty key");

    let mut table: &mut dyn TableLike = doc.as_table_mut();
    for key in parents {
        table = table
            .get_mut(key.get())
            .and_then(Item::as_table_like_mut)
            .ok_or_else(|| anyhow!("{name}: not set"))?;
    }
    if table.remove(leaf.get()).is_none() {
        bail!("{name}: not set");
    }
    Ok(())
}

/// Check, then replace atomically. Checked before the rename, never
/// after: a rejected edit has to leave the file as it was.
fn save(path: &Path, doc: &DocumentMut) -> Result<()> {
    let text = doc.to_string();
    crate::config::parse(&text)
        .map_err(|e| anyhow!("refusing to write {}: {e}", path.display()))?;
    write_atomic(path, &text, carries_key(doc))
}

/// Via a temporary file in the same directory, so an interrupted write
/// cannot leave half a config behind.
///
/// The mode is the existing file's, or 0644 for a new one:
/// `/etc/llmman/llmman.conf` has to stay readable for the aliases and
/// trust policy it carries. A file holding a key is tightened, since
/// [`crate::config`] would otherwise ignore the key just written.
fn write_atomic(path: &Path, text: &str, carries_key: bool) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));

    #[cfg(unix)]
    {
        use std::io::Write as _;
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

        let existing = std::fs::metadata(path)
            .ok()
            .map(|m| m.permissions().mode() & 0o7777);
        let mut mode = existing.unwrap_or(0o644);
        if carries_key && mode & 0o077 != 0 {
            // Silent for a new file: nobody chose that mode.
            if existing.is_some() {
                eprintln!(
                    "[llmman] tightening {} to 0600: it now holds an API key",
                    path.display()
                );
            }
            mode = 0o600;
        }
        // `mode` on create is masked by the umask, so set it again.
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(mode)
            .open(&tmp)
            .and_then(|mut f| {
                f.set_permissions(std::fs::Permissions::from_mode(mode))?;
                f.write_all(text.as_bytes())
            })
            .with_context(|| format!("writing {}", tmp.display()))?;
    }
    #[cfg(not(unix))]
    {
        let _ = carries_key;
        std::fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
    }

    std::fs::rename(&tmp, path)
        .inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp);
        })
        .with_context(|| format!("replacing {}", path.display()))
}

// ---------------------------------------------------------------------------
// Editing
// ---------------------------------------------------------------------------

fn edit(args: &ConfigArgs) -> Result<()> {
    let path = args.write_path()?;
    prepare(&path)?;

    let editor = editor();
    let status = editor_command(&editor, &path)?
        .status()
        .with_context(|| format!("running {editor}"))?;
    if !status.success() {
        bail!("{editor} exited with {status}");
    }
    if !path.exists() {
        return Ok(()); // quit without saving
    }

    // Rewritten, not just checked, so a key typed into the editor gets
    // the owner-only mode a `set` would give it. Bytes unchanged. One
    // that does not parse is reported, not reverted: a file the user
    // can go back and fix beats one llmman threw away.
    let doc = read(&path)?;
    let text = doc.to_string();
    crate::config::parse(&text).map_err(|e| anyhow!("{} is not valid: {e}", path.display()))?;
    write_atomic(&path, &text, carries_key(&doc))
}

/// The directory only: an editor handed a path under a missing one
/// takes the edit and then cannot save it. Not the file, so an
/// abandoned edit leaves nothing behind.
fn prepare(path: &Path) -> Result<()> {
    match path.parent() {
        Some(dir) => {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))
        }
        None => Ok(()),
    }
}

/// Through `sh`, as git does: `$EDITOR` is a command line, not a
/// program name, so `"/Applications/Sublime Text.app/…/subl" -w` has to
/// work. Windows has no such shell to assume.
fn editor_command(editor: &str, path: &Path) -> Result<std::process::Command> {
    if editor.trim().is_empty() {
        bail!("$VISUAL/$EDITOR is set but empty");
    }
    #[cfg(unix)]
    {
        let mut cmd = std::process::Command::new("sh");
        cmd.arg("-c")
            .arg(format!("{editor} \"$1\""))
            .arg("sh")
            .arg(path);
        Ok(cmd)
    }
    #[cfg(not(unix))]
    {
        let mut words = editor.split_whitespace();
        let program = words.next().expect("a non-empty editor has a program");
        let mut cmd = std::process::Command::new(program);
        cmd.args(words).arg(path);
        Ok(cmd)
    }
}

/// `$VISUAL`, then `$EDITOR`, then the platform fallback. Not a
/// `llmman.conf` setting: the editor belongs to the terminal, not to a
/// file shared across machines.
fn editor() -> String {
    for var in ["VISUAL", "EDITOR"] {
        match std::env::var(var) {
            Ok(value) if !value.trim().is_empty() => return value,
            _ => {}
        }
    }
    if cfg!(windows) {
        "notepad".to_string()
    } else {
        "vi".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(text: &str) -> DocumentMut {
        text.parse::<DocumentMut>().expect("valid toml")
    }

    fn keys(text: &str) -> Vec<String> {
        entries(&doc(text))
            .into_iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect()
    }

    fn scratch(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("llmman-{label}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// A trust policy `list` cannot show is one nobody can audit.
    #[test]
    fn every_scalar_is_addressable_including_one_in_an_array_of_tables() {
        let listed = keys(
            r#"
            [aliases]
            gemma4 = "docker.io/ai/gemma4"

            [verify]
            default = "warn"

            [[verify.trust]]
            pattern = "docker.io/myorg/**"
            keys    = ["keys/myorg.pub", "keys/old.pub"]
            "#,
        );
        assert_eq!(
            listed,
            vec![
                "aliases.gemma4=docker.io/ai/gemma4",
                "verify.default=warn",
                "verify.trust[0].pattern=docker.io/myorg/**",
                "verify.trust[0].keys[0]=keys/myorg.pub",
                "verify.trust[0].keys[1]=keys/old.pub",
            ],
            "{listed:?}"
        );
    }

    /// An id that is not a bare TOML key is one table, not two nested
    /// ones, and the key `list` prints has to be one `set` accepts.
    #[test]
    fn a_key_needing_quotes_is_printed_and_parsed_the_way_toml_spells_it() {
        let listed = keys("[providers.\"wafer.ai\"]\napi_key = \"sk-x\"");
        assert_eq!(
            listed,
            vec![r#"providers."wafer.ai".api_key=sk-x"#],
            "{listed:?}"
        );

        let mut d = doc("");
        set_in(&mut d, r#"providers."wafer.ai".api_key"#, "sk-y").expect("set");
        assert_eq!(
            entries(&d),
            vec![(
                r#"providers."wafer.ai".api_key"#.to_string(),
                "sk-y".to_string()
            )],
            "{d}"
        );
    }

    /// Comments and unrelated sections are why anyone hand-edits this
    /// file at all.
    #[test]
    fn set_preserves_the_comments_and_layout_of_everything_it_did_not_touch() {
        let mut d = doc("# my models\n[aliases]\ngemma4 = \"docker.io/ai/gemma4\"  # local\n");
        set_in(&mut d, "aliases.qwen", "docker.io/ai/qwen").expect("set");
        set_in(&mut d, "providers.openai.api_key", "sk-new").expect("set");

        let rendered = d.to_string();
        assert!(
            rendered.starts_with("# my models\n[aliases]\n"),
            "{rendered}"
        );
        assert!(
            rendered.contains("gemma4 = \"docker.io/ai/gemma4\"  # local"),
            "{rendered}"
        );
        assert!(
            rendered.contains("qwen = \"docker.io/ai/qwen\""),
            "{rendered}"
        );
        assert!(rendered.contains("[providers.openai]"), "{rendered}");
        assert!(!rendered.contains("[providers]\n"), "{rendered}");
    }

    /// TOML would refuse a duplicate key, and the comment on the line
    /// is not the thing that changed.
    #[test]
    fn set_replaces_a_value_in_place_keeping_its_comment() {
        let mut d = doc("[verify]\ndefault = \"off\"\n");
        set_in(&mut d, "verify.default", "enforce").expect("set");
        assert_eq!(
            entries(&d),
            vec![("verify.default".to_string(), "enforce".to_string())],
            "{d}"
        );
        crate::config::parse(&d.to_string()).expect("still valid");

        let mut d = doc("[aliases]\ngemma4 = \"old\"  # what I usually pull\n");
        set_in(&mut d, "aliases.gemma4", "new").expect("set");
        assert_eq!(
            d.to_string(),
            "[aliases]\ngemma4 = \"new\"  # what I usually pull\n",
            "{d}"
        );
    }

    /// An inline table is valid `llmman.conf`, so a key `list` prints
    /// from one has to be a key `set` and `unset` can reach.
    #[test]
    fn a_section_written_as_an_inline_table_is_still_writable() {
        let mut d = doc("aliases = { gemma4 = \"docker.io/ai/gemma4\" }\n");
        assert_eq!(
            keys("aliases = { gemma4 = \"docker.io/ai/gemma4\" }"),
            vec!["aliases.gemma4=docker.io/ai/gemma4"]
        );

        set_in(&mut d, "aliases.gemma4", "docker.io/ai/gemma5").expect("set");
        set_in(&mut d, "aliases.qwen", "docker.io/ai/qwen").expect("set");
        assert_eq!(
            entries(&d),
            vec![
                (
                    "aliases.gemma4".to_string(),
                    "docker.io/ai/gemma5".to_string()
                ),
                ("aliases.qwen".to_string(), "docker.io/ai/qwen".to_string()),
            ],
            "{d}"
        );

        unset_in(&mut d, "aliases.qwen").expect("unset");
        assert_eq!(entries(&d).len(), 1, "{d}");
        crate::config::parse(&d.to_string()).expect("still valid");
    }

    /// The whole reason the command exists: a misspelling TOML accepts
    /// is a setting that silently never takes effect.
    #[test]
    fn a_key_no_section_of_the_format_has_is_rejected_before_it_is_written() {
        let mut d = doc("");
        set_in(&mut d, "providers.openai.api_kye", "sk-x").expect("toml accepts it");
        let err = crate::config::parse(&d.to_string()).expect_err("llmman.conf must not");
        assert!(err.contains("api_kye"), "{err}");

        let mut d = doc("");
        set_in(&mut d, "alias.gemma4", "docker.io/ai/gemma4").expect("toml accepts it");
        assert!(crate::config::parse(&d.to_string()).is_err(), "{d}");
    }

    /// A success that quietly did nothing reports a credential removed
    /// that is still there.
    #[test]
    fn unset_reports_a_key_that_was_never_set() {
        let mut d = doc("[providers.openai]\napi_key = \"sk-x\"\n");
        let err = unset_in(&mut d, "providers.groq.api_key").expect_err("absent section");
        assert!(err.to_string().contains("not set"), "{err}");
        let err = unset_in(&mut d, "providers.openai.token").expect_err("absent key");
        assert!(err.to_string().contains("not set"), "{err}");

        unset_in(&mut d, "providers.openai.api_key").expect("set, so removable");
        assert!(entries(&d).is_empty(), "{d}");
    }

    /// A key is a value asked for by name, not one to spray across a
    /// terminal that ends up in an issue.
    #[test]
    fn only_a_credential_is_treated_as_one() {
        assert!(carries_key(&doc("[providers.openai]\napi_key = \"sk-x\"")));
        assert!(!carries_key(&doc("[providers.openai]\napi_key = \"\"")));
        assert!(!carries_key(&doc("[aliases]\napi_keyring = \"x\"")));
        assert!(!carries_key(&doc("[verify]\ndefault = \"off\"")));
    }

    /// `config::owner_readable_only` ignores the keys in a file group
    /// or other can read, so a key llmman writes must be one it reads
    /// back. A config holding no key stays readable, since
    /// `/etc/llmman/llmman.conf` has to be.
    #[cfg(unix)]
    #[test]
    fn only_a_file_holding_a_key_is_made_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch("mode");
        let path = dir.join("llmman.conf");
        let mode = |p: &Path| std::fs::metadata(p).expect("metadata").permissions().mode() & 0o777;

        let mut d = doc("");
        set_in(&mut d, "aliases.gemma4", "docker.io/ai/gemma4").expect("set");
        save(&path, &d).expect("save");
        assert_eq!(mode(&path), 0o644, "a new file with no key in it");

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).expect("chmod");
        let mut d = read(&path).expect("read");
        set_in(&mut d, "aliases.qwen", "docker.io/ai/qwen").expect("set");
        save(&path, &d).expect("save");
        assert_eq!(mode(&path), 0o640, "a mode an admin chose is kept");

        let mut d = read(&path).expect("read");
        set_in(&mut d, "providers.openai.api_key", "sk-x").expect("set");
        save(&path, &d).expect("save");
        assert_eq!(mode(&path), 0o600, "tightened once it holds a key");

        // A key is never on disk at a loose mode, not even briefly.
        std::fs::remove_file(&path).expect("rm");
        let mut d = doc("");
        set_in(&mut d, "providers.openai.api_key", "sk-x").expect("set");
        save(&path, &d).expect("save");
        assert_eq!(mode(&path), 0o600, "a new file holding a key");

        // The daemon's own keys and the peer key are credentials too.
        for key in ["auth.api_keys", "aggregation.api_key"] {
            std::fs::remove_file(&path).expect("rm");
            let mut d = doc("");
            set_in(&mut d, key, "k").expect("set");
            save(&path, &d).expect("save");
            assert_eq!(mode(&path), 0o600, "{key}");
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Most machines never have one, and `set` has to create the first.
    #[test]
    fn an_absent_file_reads_as_empty() {
        let path = std::env::temp_dir().join(format!("llmman-absent-{}.conf", std::process::id()));
        std::fs::remove_file(&path).ok();
        assert!(entries(&read(&path).expect("absent is empty")).is_empty());
    }

    /// Every `set` after the first renames over a file already there.
    /// Not `cfg(unix)`: catches a platform whose rename will not
    /// replace.
    #[test]
    fn a_second_write_replaces_the_file_in_place() {
        let dir = scratch("rewrite");
        let path = dir.join("llmman.conf");

        let mut d = doc("");
        set_in(&mut d, "aliases.gemma4", "docker.io/ai/gemma4").expect("set");
        save(&path, &d).expect("first save");

        let mut d = read(&path).expect("read back");
        set_in(&mut d, "aliases.gemma4", "docker.io/ai/gemma5").expect("set");
        save(&path, &d).expect("second save must replace, not fail");

        assert_eq!(
            entries(&read(&path).expect("read back")),
            vec![(
                "aliases.gemma4".to_string(),
                "docker.io/ai/gemma5".to_string()
            )]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `~/.config/llmman` may not exist yet, and an editor that cannot
    /// save takes the edit anyway — llmman would then report success
    /// over an empty file. An abandoned edit still leaves nothing.
    #[test]
    fn edit_prepares_the_directory_but_not_the_file() {
        let root = scratch("edit");
        let path = root.join("never").join("existed").join("llmman.conf");

        prepare(&path).expect("prepare");
        assert!(path.parent().expect("a parent").is_dir());
        assert!(!path.exists(), "{}", path.display());

        std::fs::remove_dir_all(&root).ok();
    }

    /// `$EDITOR` is a command line, not a program name: a path with a
    /// space in it and a trailing flag both have to survive.
    #[cfg(unix)]
    #[test]
    fn an_editor_with_spaces_and_flags_reaches_the_file() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch("editor");
        let script = dir.join("my editor.sh");
        std::fs::write(&script, "#!/bin/sh\nprintf '%s\\n' \"$1\" > \"$2\"\n").expect("write");
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).expect("chmod");
        let out = dir.join("out");

        let status = editor_command(&format!("\"{}\" --flag", script.display()), &out)
            .expect("a command")
            .status()
            .expect("run");
        assert!(status.success(), "{status}");
        assert_eq!(
            std::fs::read_to_string(&out).expect("read"),
            "--flag\n",
            "the flag and the path both arrived"
        );

        assert!(editor_command("   ", &out).is_err(), "an empty editor");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A malformed key is refused rather than treated as a literal name
    /// with a dot in it.
    #[test]
    fn a_malformed_key_is_refused() {
        assert!(parse_key("").is_err());
        assert!(parse_key("providers..api_key").is_err());
        assert!(parse_key("providers.\"unterminated").is_err());
        assert_eq!(parse_key("aliases.gemma4").expect("valid").len(), 2);
    }
}

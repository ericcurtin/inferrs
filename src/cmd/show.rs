//! `llmman show MODEL` — display a locally stored model's metadata:
//! architecture, parameter count, context length, quantization, license,
//! and chat template — the llmman equivalent of `ollama show`.
//!
//! Unlike `ollama show` (which always goes through a running server),
//! this runs entirely against the local `OciStore`, same as `inspect`/
//! `tag`/`rm` — no daemon required. Real GGUF/safetensors metadata is
//! read directly (see `crate::gguf`) rather than relying on a stored
//! Modelfile — llmman has no such document, model config lives in the
//! model file itself (GGUF metadata) or its `config.json`/
//! `tokenizer_config.json` (safetensors).

use std::io::Read;
use std::path::Path;

use clap::Args;

use crate::modelpack::{resolve_model, ModelPath};
use crate::storage::oci::Descriptor;
use crate::storage::OciStore;

#[derive(Args, Debug)]
pub struct ShowArgs {
    /// Model to show
    #[arg(value_name = "MODEL")]
    pub model: String,

    /// Show only the model's license text
    #[arg(long)]
    pub license: bool,

    /// Show only the model's chat template
    #[arg(long)]
    pub template: bool,

    /// Show only the model's architecture/parameters/quantization block
    #[arg(long)]
    pub parameters: bool,

    /// Show every parsed GGUF metadata key/value pair
    #[arg(short, long)]
    pub verbose: bool,
}

pub fn run(args: &ShowArgs) -> anyhow::Result<()> {
    // Resolve/validate before opening the store so a bad ref never creates
    // the store tree, matching cp.rs/build.rs.
    let reference = crate::shortnames::resolve_ollama_api(&args.model)?;
    let store_path = crate::default_store()?;
    let store = OciStore::open(&store_path)?;
    let desc = store.find(&reference)?;
    let manifest = store.read_manifest(&desc.digest)?;

    let cache_path = crate::default_cache()?;
    std::fs::create_dir_all(&cache_path).ok();
    let resolved = resolve_model(&store_path, &cache_path, &reference)?;

    // A read failure here (a corrupt/truncated GGUF header) downgrades to
    // `None` rather than aborting the whole command with `?`: the
    // digest/size lines, `--license` (manifest-layer case), and any
    // Labels section below don't depend on GGUF metadata at all, and
    // still deserve to print even when it can't be read.
    let gguf_info = match &resolved {
        ModelPath::Gguf(path, _) => match crate::gguf::read_info(path) {
            Ok(info) => Some(info),
            Err(e) => {
                eprintln!("[llmman] reading GGUF metadata for {reference}: {e:#}");
                None
            }
        },
        ModelPath::SafeTensors(_) | ModelPath::Diffusion(_) | ModelPath::Omni(_) => None,
    };
    let safetensors_config = match &resolved {
        ModelPath::SafeTensors(dir) | ModelPath::Omni(dir) => {
            read_json_file(&dir.join("config.json"))
        }
        ModelPath::Gguf(..) | ModelPath::Diffusion(_) => None,
    };
    let tokenizer_config = match &resolved {
        ModelPath::SafeTensors(dir) | ModelPath::Omni(dir) => {
            read_json_file(&dir.join("tokenizer_config.json"))
        }
        ModelPath::Gguf(..) | ModelPath::Diffusion(_) => None,
    };

    // Any single-focus flag suppresses the full summary and prints just
    // that one thing, matching `ollama show --license`/`--template`/
    // `--parameters`'s own behavior.
    if args.license {
        match find_license(&store, &manifest, gguf_info.as_ref()) {
            Some(text) => println!("{}", text.trim_end()),
            None => println!("(no license found)"),
        }
        return Ok(());
    }
    if args.template {
        match chat_template(gguf_info.as_ref(), tokenizer_config.as_ref()) {
            Some(text) => println!("{}", text.trim_end()),
            None => println!("(no chat template found)"),
        }
        return Ok(());
    }
    if args.parameters {
        print_parameters_block(
            gguf_info.as_ref(),
            safetensors_config.as_ref(),
            resolved.format(),
        );
        return Ok(());
    }
    if args.verbose {
        if let Some(info) = &gguf_info {
            let mut keys: Vec<_> = info.metadata.keys().collect();
            keys.sort();
            for key in keys {
                println!("{key} = {}", format_value(&info.metadata[key]));
            }
        } else if let Some(cfg) = &safetensors_config {
            println!("{}", serde_json::to_string_pretty(cfg)?);
        } else {
            println!("(no metadata found)");
        }
        return Ok(());
    }

    // Full summary.
    println!("  {}", reference);
    println!();
    println!("  Digest       {}", desc.digest);
    println!(
        "  Size         {}",
        crate::fmt::human_size(store.total_size(&desc))
    );
    println!();
    println!("Model");
    print_parameters_block(
        gguf_info.as_ref(),
        safetensors_config.as_ref(),
        resolved.format(),
    );

    if let Some(text) = find_license(&store, &manifest, gguf_info.as_ref()) {
        println!();
        println!("License");
        for line in text.lines().take(3) {
            println!("    {line}");
        }
        if text.lines().count() > 3 {
            println!("    ...");
        }
    }

    if chat_template(gguf_info.as_ref(), tokenizer_config.as_ref()).is_some() {
        println!();
        println!("Template");
        println!(
            "    (present — use `llmman show --template {}` to view it)",
            args.model
        );
    }

    if let Some(annotations) = &manifest.annotations {
        if !annotations.is_empty() {
            println!();
            println!("Labels");
            let mut keys: Vec<_> = annotations.keys().collect();
            keys.sort();
            for key in keys {
                println!("    {key} = {}", annotations[key]);
            }
        }
    }

    Ok(())
}

fn print_parameters_block(
    gguf_info: Option<&crate::gguf::Info>,
    safetensors_config: Option<&serde_json::Value>,
    format: &str,
) {
    println!("    format               {format}");
    if let Some(info) = gguf_info {
        if let Some(arch) = info.architecture() {
            println!("    architecture         {arch}");
        }
        if info.parameter_count > 0 {
            println!(
                "    parameters           {}",
                crate::fmt::human_count(info.parameter_count)
            );
        }
        if let Some(ctx) = info.context_length() {
            println!("    context length       {ctx}");
        }
        if let Some(emb) = info.embedding_length() {
            println!("    embedding length     {emb}");
        }
        if let Some(blocks) = info.block_count() {
            println!("    block count          {blocks}");
        }
        if let Some(q) = &info.quantization {
            println!("    quantization         {q}");
        }
    } else if let Some(cfg) = safetensors_config {
        if let Some(arch) = cfg
            .get("model_type")
            .or_else(|| cfg.get("architectures").and_then(|a| a.get(0)))
            .and_then(|v| v.as_str())
        {
            println!("    architecture         {arch}");
        }
        for (label, key) in [
            ("hidden size", "hidden_size"),
            ("layers", "num_hidden_layers"),
            ("attention heads", "num_attention_heads"),
            ("context length", "max_position_embeddings"),
            ("vocab size", "vocab_size"),
        ] {
            if let Some(v) = cfg.get(key).and_then(|v| v.as_u64()) {
                println!("    {label:<20} {v}");
            }
        }
        if let Some(dtype) = cfg.get("torch_dtype").and_then(|v| v.as_str()) {
            println!("    quantization         {dtype}");
        }
    }
}

fn format_value(v: &crate::gguf::Value) -> String {
    use crate::gguf::Value;
    match v {
        // `&s[..200]` truncates on a *byte* offset — GGUF metadata
        // strings (a chat template, `general.description`, ...) are
        // routinely non-ASCII, and slicing mid-character panics with
        // "byte index 200 is not a char boundary". Walking char_indices
        // to the last boundary at or before 200 keeps the same
        // ~200-byte-ish truncation length without ever landing inside a
        // multi-byte character.
        Value::String(s) if s.len() > 200 => {
            let end = s
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|&i| i <= 200)
                .last()
                .unwrap_or(0);
            format!("{}... ({} bytes)", &s[..end], s.len())
        }
        Value::String(s) => s.clone(),
        Value::Array(items) => format!("[{} items]", items.len()),
        Value::Bool(b) => b.to_string(),
        Value::F32(f) => f.to_string(),
        Value::F64(f) => f.to_string(),
        other => other
            .as_u64()
            .map(|n| n.to_string())
            .unwrap_or_else(|| format!("{other:?}")),
    }
}

/// The model's chat template, if one can be found — GGUF's
/// `tokenizer.chat_template` key, or (for safetensors) a
/// `tokenizer_config.json`'s own `chat_template` field.
fn chat_template(
    gguf_info: Option<&crate::gguf::Info>,
    tokenizer_config: Option<&serde_json::Value>,
) -> Option<String> {
    if let Some(info) = gguf_info {
        if let Some(t) = info.str("tokenizer.chat_template") {
            return Some(t.to_string());
        }
    }
    tokenizer_config?
        .get("chat_template")?
        .as_str()
        .map(str::to_string)
}

/// Finds and reads a manifest layer that looks like a license file
/// (`LICENSE`, `LICENSE.txt`, `LICENSE.md`, case-insensitive) — or, for a
/// GGUF file, falls back to any `general.license*` metadata string.
fn find_license(
    store: &OciStore,
    manifest: &crate::storage::oci::Manifest,
    gguf_info: Option<&crate::gguf::Info>,
) -> Option<String> {
    for layer in &manifest.layers {
        let Some(filepath) = layer.annotations.as_ref().and_then(|a| {
            a.get("org.cncf.model.filepath")
                .or_else(|| a.get("org.opencontainers.image.title"))
        }) else {
            continue;
        };
        let base = Path::new(filepath)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or(filepath)
            .to_lowercase();
        if base == "license" || base.starts_with("license.") {
            if let Ok(text) = read_layer_text(store, layer) {
                return Some(text);
            }
        }
    }
    // No LICENSE-ish layer found — fall back to whichever
    // `general.license*` metadata key (if any) the GGUF file itself
    // carries, matching this function's own doc comment.
    let info = gguf_info?;
    info.str("general.license")
        .or_else(|| info.str("general.license.name"))
        .map(str::to_string)
}

/// Reads a manifest layer's text content — transparently un-tarring it if
/// it's a single-file tar (as every layer `llmman build` produces is —
/// see `storage::oci::classify_model_layer`'s own doc comment), or else
/// treating the whole blob as raw text (as HuggingFace/cloud-source pulls
/// store un-archived doc/config blobs — see
/// [`crate::sources::classify_file`]).
fn read_layer_text(store: &OciStore, layer: &Descriptor) -> anyhow::Result<String> {
    let blob = store.read_blob(&layer.digest)?;
    if blob.len() >= 512 {
        let mut archive = tar::Archive::new(std::io::Cursor::new(&blob));
        if let Ok(entries) = archive.entries() {
            for entry in entries.flatten() {
                let mut entry = entry;
                let mut s = String::new();
                if entry.read_to_string(&mut s).is_ok() && !s.is_empty() {
                    return Ok(s);
                }
            }
        }
    }
    Ok(String::from_utf8_lossy(&blob).into_owned())
}

fn read_json_file(path: &Path) -> Option<serde_json::Value> {
    let data = std::fs::read(path).ok()?;
    serde_json::from_slice(&data).ok()
}

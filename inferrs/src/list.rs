//! `inferrs list` — show locally cached HuggingFace models.

use anyhow::Result;
use clap::Parser;

use crate::util::{cache_root, dir_size, format_bytes};

#[derive(Parser, Clone)]
pub struct ListArgs {}

pub fn run(_args: ListArgs) -> Result<()> {
    let cache_dir = cache_root();

    if !cache_dir.exists() {
        println!(
            "No models cached (cache directory does not exist: {})",
            cache_dir.display()
        );
        return Ok(());
    }

    let mut entries: Vec<(String, u64)> = std::fs::read_dir(&cache_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().map(|t| t.is_dir()).unwrap_or(false)
                && e.file_name().to_string_lossy().starts_with("models--")
        })
        .map(|e| {
            let folder = e.file_name().to_string_lossy().into_owned();
            let model_id = folder_to_model_id(&folder);
            let size = dir_size(&e.path()).unwrap_or(0);
            (model_id, size)
        })
        .collect();

    if entries.is_empty() {
        println!("No models cached.");
        return Ok(());
    }

    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let name_width = entries.iter().map(|(id, _)| id.len()).max().unwrap_or(0);

    for (model_id, size) in &entries {
        println!(
            "{:<width$}  {}",
            model_id,
            format_bytes(*size),
            width = name_width
        );
    }

    Ok(())
}

/// Convert "models--Org--Name" back to "Org/Name".
fn folder_to_model_id(folder: &str) -> String {
    folder
        .strip_prefix("models--")
        .unwrap_or(folder)
        .replace("--", "/")
}

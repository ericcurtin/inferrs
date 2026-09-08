//! The prompt as `Cosmos3OmniPipeline.tokenize_prompt` builds it: chat template,
//! metadata sentences, then `<|im_end|><|vision_start|>`.

use std::path::Path;

use anyhow::{anyhow, Context, Result};

const SYSTEM_IMAGE: &str =
    "You are a helpful assistant who will generate images from a give prompt.";
const SYSTEM_VIDEO: &str =
    "You are a helpful assistant who will generate videos from a give prompt.";

/// Which chat template the text backbone was trained with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Template {
    /// Qwen3-VL (Nano, Super): ChatML, no default system block.
    Qwen,
    /// Nemotron (Edge): ChatML with a leading newline, an always-present
    /// system block and a `<think>\n` generation prompt.
    Nemotron,
}

pub struct Tokenizer {
    tok: tokenizers::Tokenizer,
    pub template: Template,
    pub use_system_prompt: bool,
    pub eos: u32,
    pub vision_start: u32,
}

/// The prompt-side description of what is being generated.
pub struct PromptSpec {
    pub prompt: String,
    pub negative: bool,
    pub n_frames: i64,
    pub fps: f32,
    pub width: i64,
    pub height: i64,
}

impl Tokenizer {
    pub fn load(path: &Path, template: Template, use_system_prompt: bool) -> Result<Tokenizer> {
        let tok = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow!("{}: {e}", path.display()))?;
        let id = |s: &str| {
            tok.token_to_id(s)
                .ok_or_else(|| anyhow!("{}: no {s} token", path.display()))
        };
        Ok(Tokenizer {
            eos: id("<|im_end|>")?,
            vision_start: id("<|vision_start|>")?,
            tok,
            template,
            use_system_prompt,
        })
    }

    /// The augmented user text: templates appended like the pipeline does
    /// (`_apply_templates`), inverse forms for the negative prompt.
    pub fn user_text(spec: &PromptSpec) -> String {
        let is_image = spec.n_frames == 1;
        let mut text = spec.prompt.clone();
        let mut append = |s: String| {
            let base = text.trim_end_matches('.');
            text = if base.is_empty() {
                s
            } else {
                format!("{base}. {s}")
            };
        };
        if !is_image {
            let duration = spec.n_frames as f64 / spec.fps as f64;
            append(if spec.negative {
                format!(
                    "The video is not {duration:.1} seconds long and is not of {:.0} FPS.",
                    spec.fps
                )
            } else {
                format!(
                    "The video is {duration:.1} seconds long and is of {:.0} FPS.",
                    spec.fps
                )
            });
        }
        let (h, w) = (spec.height, spec.width);
        append(match (is_image, spec.negative) {
            (true, false) => format!("This image is of {h}x{w} resolution."),
            (true, true) => format!("This image is not of {h}x{w} resolution."),
            (false, false) => format!("This video is of {h}x{w} resolution."),
            (false, true) => format!("This video is not of {h}x{w} resolution."),
        });
        text
    }

    /// Token ids of the chat plus `[eos, <|vision_start|>]`.
    pub fn encode(&self, spec: &PromptSpec) -> Result<Vec<u32>> {
        let text = render(self.template, self.use_system_prompt, spec);
        self.encode_text(&text)
    }

    /// The chat around a verbatim user message (an action run's JSON caption
    /// or its raw negative prompt), with the given system message if any.
    pub fn encode_chat(&self, user: &str, system: Option<&str>) -> Result<Vec<u32>> {
        let text = render_chat(self.template, system, user);
        self.encode_text(&text)
    }

    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(|e| anyhow!("tokenizing the prompt: {e}"))
            .context("cosmos3 tokenizer")?;
        let mut ids = enc.get_ids().to_vec();
        ids.push(self.eos);
        ids.push(self.vision_start);
        Ok(ids)
    }
}

/// The chat as `apply_chat_template(..., add_generation_prompt=True)` renders it
/// for each checkpoint's template, before tokenization.
pub fn render(template: Template, use_system_prompt: bool, spec: &PromptSpec) -> String {
    let user = Tokenizer::user_text(spec);
    let system = use_system_prompt.then_some(if spec.n_frames == 1 {
        SYSTEM_IMAGE
    } else {
        SYSTEM_VIDEO
    });
    render_chat(template, system, &user)
}

fn render_chat(template: Template, system: Option<&str>, user: &str) -> String {
    match template {
        Template::Qwen => {
            let mut s = String::new();
            if let Some(sys) = system {
                s.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
            }
            s.push_str(&format!(
                "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
            ));
            s
        }
        // Nemotron's template starts with a blank line, always emits the
        // system block and opens a thinking block in the generation prompt.
        Template::Nemotron => format!(
            "\n<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n",
            system.unwrap_or("")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_text_templates() {
        let spec = PromptSpec {
            prompt: "A cat sits on a mat.".into(),
            negative: false,
            n_frames: 49,
            fps: 24.0,
            width: 832,
            height: 480,
        };
        assert_eq!(
            Tokenizer::user_text(&spec),
            "A cat sits on a mat. The video is 2.0 seconds long and is of 24 FPS. This video is of 480x832 resolution."
        );
        let img = PromptSpec {
            n_frames: 1,
            negative: true,
            prompt: String::new(),
            ..spec
        };
        assert_eq!(
            Tokenizer::user_text(&img),
            "This image is not of 480x832 resolution."
        );
    }

    #[test]
    fn chat_templates() {
        // rendered exactly as transformers' apply_chat_template does for both checkpoints
        let spec = PromptSpec {
            prompt: "A cat.".into(),
            negative: false,
            n_frames: 1,
            fps: 24.0,
            width: 256,
            height: 256,
        };
        let user = "A cat. This image is of 256x256 resolution.";
        assert_eq!(
            render(Template::Qwen, true, &spec),
            format!("<|im_start|>system\n{SYSTEM_IMAGE}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
        );
        assert_eq!(
            render(Template::Qwen, false, &spec),
            format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
        );
        assert_eq!(
            render(Template::Nemotron, false, &spec),
            format!("\n<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n")
        );
    }
}

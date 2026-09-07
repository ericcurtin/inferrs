//! `llmman run` for media generation models, after ollama's (since removed)
//! `x/imagegen/cli.go`: one prompt in, one file out (png, or mp4 / wav
//! with `--video` / `--audio`), saved next to the shell and shown inline
//! where the terminal can. Progress comes as server-sent events.

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use base64::Engine as _;
use indicatif::{ProgressBar, ProgressStyle};

/// What to generate for each prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Media {
    #[default]
    Image,
    Video,
    Audio,
}

/// Per-request knobs, `ollama run`'s hidden `--width/--height/--steps/
/// --seed/--negative` flags. Zero / empty means "model default".
#[derive(Debug, Clone, Default)]
pub struct ImageOptions {
    pub media: Media,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub seed: Option<u32>,
    /// Guidance scale; the negative prompt only counts above 1.
    pub cfg_scale: f32,
    pub negative: String,
    /// `--video` / `--audio` clip length; 0 = server default.
    pub seconds: f32,
}

impl ImageOptions {
    fn request(&self, model: &str, prompt: &str) -> serde_json::Value {
        let mut req = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": true,
            "response_format": "b64_json",
        });
        if self.width > 0 {
            req["width"] = self.width.into();
        }
        if self.height > 0 {
            req["height"] = self.height.into();
        }
        if self.steps > 0 {
            req["steps"] = self.steps.into();
        }
        if let Some(seed) = self.seed {
            req["seed"] = seed.into();
        }
        self.guidance(&mut req);
        req
    }

    fn guidance(&self, req: &mut serde_json::Value) {
        if self.cfg_scale > 0.0 {
            req["cfg_scale"] = self.cfg_scale.into();
        }
        if !self.negative.is_empty() {
            req["negative_prompt"] = serde_json::Value::String(self.negative.clone());
        }
    }

    /// The `/v1/videos` request: no streaming, the mp4 is fetched from
    /// the job's `content_url` afterwards.
    fn video_request(&self, model: &str, prompt: &str) -> serde_json::Value {
        let mut req = self.request(model, prompt);
        req["stream"] = false.into();
        req.as_object_mut().map(|o| o.remove("response_format"));
        if self.seconds > 0.0 {
            req["seconds"] = self.seconds.into();
        }
        req
    }

    /// The `/v1/audio/speech` request: OpenAI's field is `input`.
    fn audio_request(&self, model: &str, prompt: &str) -> serde_json::Value {
        let mut req = serde_json::json!({
            "model": model,
            "input": prompt,
            "response_format": "wav",
        });
        if self.steps > 0 {
            req["steps"] = self.steps.into();
        }
        if let Some(seed) = self.seed {
            req["seed"] = seed.into();
        }
        self.guidance(&mut req);
        if self.seconds > 0.0 {
            req["seconds"] = self.seconds.into();
        }
        req
    }
}

/// `llmman run <image model> ["prompt"]`: one-shot when a prompt was
/// given (or piped), otherwise ollama's `>>> ` loop where every line is
/// a prompt and `/set` adjusts the options.
pub fn run(model: &str, prompt: &str, interactive: bool, mut opts: ImageOptions) -> Result<()> {
    if !interactive {
        let p = if prompt.is_empty() {
            let mut s = String::new();
            io::stdin().read_line(&mut s)?;
            s.trim().to_string()
        } else {
            prompt.to_string()
        };
        if p.is_empty() {
            return Ok(());
        }
        return generate(model, &p, &opts).map(|_| ());
    }

    eprintln!(
        "Send a prompt to generate {} (/? for help)",
        match opts.media {
            Media::Image => "an image",
            Media::Video => "a video",
            Media::Audio => "audio",
        }
    );
    let stdin = io::stdin();
    loop {
        eprint!(">>> ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            eprintln!();
            return Ok(());
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(cmd) = line.strip_prefix('/') {
            match handle_command(cmd, &mut opts) {
                Command::Continue => continue,
                Command::Quit => return Ok(()),
            }
        }
        if let Err(e) = generate(model, line, &opts) {
            eprintln!("Error: {e:#}");
        }
    }
}

enum Command {
    Continue,
    Quit,
}

/// ollama's image REPL slash commands.
fn handle_command(cmd: &str, opts: &mut ImageOptions) -> Command {
    let mut words = cmd.split_whitespace();
    match words.next().unwrap_or("") {
        "bye" | "exit" | "quit" => return Command::Quit,
        "?" | "help" => {
            eprintln!("Available Commands:");
            eprintln!("  /set width <n>     Image width in pixels");
            eprintln!("  /set height <n>    Image height in pixels");
            eprintln!("  /set steps <n>     Denoising steps");
            eprintln!("  /set seed <n>      Random seed");
            eprintln!("  /set cfg <n>       Guidance scale (>1 enables the negative prompt)");
            eprintln!("  /set negative <t>  Negative prompt");
            eprintln!("  /set seconds <n>   Video / audio length");
            eprintln!("  /set media <m>     image, video or audio");
            eprintln!("  /show              Show current settings");
            eprintln!("  /bye               Exit");
        }
        "show" => {
            eprintln!(
                "width={} height={} steps={} seed={} cfg={} negative={:?}",
                opts.width,
                opts.height,
                opts.steps,
                opts.seed.map_or("random".to_string(), |s| s.to_string()),
                opts.cfg_scale,
                opts.negative
            );
        }
        "set" => {
            let key = words.next().unwrap_or("");
            let rest: Vec<&str> = words.collect();
            let value = rest.join(" ");
            let num = value.parse::<u32>();
            match (key, num) {
                ("width", Ok(v)) => opts.width = v,
                ("height", Ok(v)) => opts.height = v,
                ("steps", Ok(v)) => opts.steps = v,
                ("seed", Ok(v)) => opts.seed = Some(v),
                ("negative", _) => opts.negative = value,
                ("seconds", _) | ("cfg", _) => match value.parse::<f32>() {
                    Ok(v) if key == "cfg" => opts.cfg_scale = v,
                    Ok(v) => opts.seconds = v,
                    Err(_) => eprintln!("Invalid value for /set {key}: {value}"),
                },
                ("media", _) => {
                    opts.media = match value.as_str() {
                        "image" => Media::Image,
                        "video" => Media::Video,
                        "audio" => Media::Audio,
                        other => {
                            eprintln!("Unknown media type: {other} (image, video or audio)");
                            opts.media
                        }
                    }
                }
                (k, _) => eprintln!("Unknown or invalid setting: /set {k} {value}"),
            }
        }
        other => eprintln!("Unknown command: /{other}. Type /? for help"),
    }
    Command::Continue
}

/// One generation: an image (streamed, with a step bar), or a video / an
/// audio clip (one blocking request behind a spinner). Returns the saved
/// file's path.
pub fn generate(model: &str, prompt: &str, opts: &ImageOptions) -> Result<PathBuf> {
    match opts.media {
        Media::Image => generate_image(model, prompt, opts),
        Media::Video => generate_video(model, prompt, opts),
        Media::Audio => generate_audio(model, prompt, opts),
    }
}

fn client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .context("build http client")
}

fn spinner(msg: &str) -> Option<ProgressBar> {
    io::stderr().is_terminal().then(|| {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("{spinner} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner()),
        );
        pb.set_message(msg.to_string());
        pb.enable_steady_tick(Duration::from_millis(100));
        pb
    })
}

/// Clears a spinner on every exit path, like ollama's `defer p.StopAndClear()`.
struct Spinner(Option<ProgressBar>);

impl Drop for Spinner {
    fn drop(&mut self) {
        if let Some(pb) = self.0.take() {
            pb.finish_and_clear();
        }
    }
}

fn fail(resp: reqwest::blocking::Response) -> anyhow::Error {
    let body = resp.text().unwrap_or_default();
    anyhow::anyhow!("{}", crate::daemon::api_error(&body).unwrap_or(body))
}

/// `/v1/videos` then `GET <content_url>` for the mp4 (llama-server muxes
/// it with ffmpeg; without ffmpeg the job carries frames only, which is
/// reported as an error here rather than written as hundreds of PNGs).
fn generate_video(model: &str, prompt: &str, opts: &ImageOptions) -> Result<PathBuf> {
    let client = client()?;
    let _pb = Spinner(spinner("Generating video..."));
    let resp = client
        .post(format!("{}/v1/videos", crate::daemon::server()))
        .json(&opts.video_request(model, prompt))
        .send()
        .context("request /v1/videos")?;
    if !resp.status().is_success() {
        return Err(fail(resp));
    }
    let job: serde_json::Value = resp.json().context("decode /v1/videos response")?;
    let Some(content_url) = job["content_url"].as_str() else {
        anyhow::bail!(
            "the server generated {} frames but has no ffmpeg to mux an mp4; install ffmpeg next to llama-server",
            job["n_frames"].as_u64().unwrap_or(0)
        );
    };
    let mp4 = client
        .get(format!("{}{}", crate::daemon::server(), content_url))
        .send()
        .context("fetch video content")?;
    if !mp4.status().is_success() {
        return Err(fail(mp4));
    }
    let bytes = mp4.bytes().context("read video content")?;
    drop(_pb);
    let path = output_path(prompt, "mp4");
    std::fs::write(&path, &bytes).with_context(|| format!("write {}", path.display()))?;
    println!("Video saved to: {}", path.display());
    Ok(path)
}

/// `/v1/audio/speech`, saved as a wav.
fn generate_audio(model: &str, prompt: &str, opts: &ImageOptions) -> Result<PathBuf> {
    let client = client()?;
    let _pb = Spinner(spinner("Generating audio..."));
    let resp = client
        .post(format!("{}/v1/audio/speech", crate::daemon::server()))
        .json(&opts.audio_request(model, prompt))
        .send()
        .context("request /v1/audio/speech")?;
    if !resp.status().is_success() {
        return Err(fail(resp));
    }
    let bytes = resp.bytes().context("read audio")?;
    drop(_pb);
    let path = output_path(prompt, "wav");
    std::fs::write(&path, &bytes).with_context(|| format!("write {}", path.display()))?;
    println!("Audio saved to: {}", path.display());
    Ok(path)
}

/// One streamed `/v1/images/generations` request: spinner, then a step
/// bar once the first progress event arrives, then the saved PNG's path.
fn generate_image(model: &str, prompt: &str, opts: &ImageOptions) -> Result<PathBuf> {
    let client = client()?;
    let resp = client
        .post(format!("{}/v1/images/generations", crate::daemon::server()))
        .json(&opts.request(model, prompt))
        .send()
        .context("request /v1/images/generations")?;
    if !resp.status().is_success() {
        let body = resp.text().unwrap_or_default();
        anyhow::bail!("{}", crate::daemon::api_error(&body).unwrap_or(body));
    }

    let mut bar: Option<ProgressBar> = spinner("Loading model...");

    let mut png: Option<Vec<u8>> = None;
    let reader = io::BufReader::new(resp);
    for line in reader.lines() {
        let line = line.context("read image generation stream")?;
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let ev: serde_json::Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        match ev["type"].as_str().unwrap_or("") {
            "image_generation.progress" => {
                let step = ev["step"].as_u64().unwrap_or(0);
                let total = ev["total"].as_u64().unwrap_or(0);
                if let Some(pb) = &mut bar {
                    if pb.length().is_none() && total > 0 {
                        pb.finish_and_clear();
                        let steps = ProgressBar::new(total);
                        steps.set_style(
                            ProgressStyle::with_template("Generating {bar:30} {pos}/{len}")
                                .unwrap_or_else(|_| ProgressStyle::default_bar()),
                        );
                        *pb = steps;
                    }
                    pb.set_position(step);
                }
            }
            "image_generation.completed" => {
                let b64 = ev["b64_json"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .context("completed event carried no image")?;
                png = Some(
                    base64::engine::general_purpose::STANDARD
                        .decode(b64)
                        .context("decode image")?,
                );
            }
            "error" => {
                if let Some(pb) = bar.take() {
                    pb.finish_and_clear();
                }
                let msg = ev["error"]["message"]
                    .as_str()
                    .unwrap_or("image generation failed");
                anyhow::bail!("{msg}");
            }
            _ => {}
        }
    }
    if let Some(pb) = bar.take() {
        pb.finish_and_clear();
    }
    let png = png.ok_or_else(|| anyhow::anyhow!("stream ended without an image"))?;

    let path = output_path(prompt, "png");
    std::fs::write(&path, &png).with_context(|| format!("write {}", path.display()))?;
    if io::stdout().is_terminal() {
        display_inline(&png);
    }
    println!("Image saved to: {}", path.display());
    Ok(path)
}

/// `<prompt slug>-<YYYYMMDD-HHMMSS>.<ext>` in the working directory, as
/// ollama named its images.
fn output_path(prompt: &str, ext: &str) -> PathBuf {
    let mut slug: String = prompt
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    let slug = slug.trim_matches('-');
    let slug = if slug.is_empty() {
        "image"
    } else {
        &slug[..slug.len().min(50)]
    };
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    Path::new(&format!("{slug}-{stamp}.{ext}")).to_path_buf()
}

/// Shows the PNG in terminals that can: iTerm2 / WezTerm (OSC 1337
/// inline images) and kitty / Ghostty (kitty graphics protocol). A no-op
/// elsewhere — the file path is printed regardless.
fn display_inline(png: &[u8]) {
    let term_program = std::env::var("TERM_PROGRAM").unwrap_or_default();
    let term = std::env::var("TERM").unwrap_or_default();
    let b64 = base64::engine::general_purpose::STANDARD.encode(png);
    let mut out = io::stdout().lock();
    if term_program == "iTerm.app"
        || term_program == "WezTerm"
        || std::env::var("LC_TERMINAL").as_deref() == Ok("iTerm2")
    {
        let _ = writeln!(
            out,
            "\x1b]1337;File=inline=1;size={}:{}\x07",
            png.len(),
            b64
        );
    } else if term.contains("kitty")
        || term_program == "ghostty"
        || std::env::var("KITTY_WINDOW_ID").is_ok()
    {
        // kitty graphics protocol: PNG payload in 4096-byte base64 chunks
        let bytes = b64.as_bytes();
        let mut first = true;
        let mut i = 0;
        while i < bytes.len() {
            let end = (i + 4096).min(bytes.len());
            let more = if end < bytes.len() { 1 } else { 0 };
            let chunk = std::str::from_utf8(&bytes[i..end]).unwrap_or("");
            if first {
                let _ = write!(out, "\x1b_Ga=T,f=100,m={more};{chunk}\x1b\\");
                first = false;
            } else {
                let _ = write!(out, "\x1b_Gm={more};{chunk}\x1b\\");
            }
            i = end;
        }
        let _ = writeln!(out);
    }
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_includes_only_set_options() {
        let opts = ImageOptions {
            width: 512,
            seed: Some(7),
            ..Default::default()
        };
        let req = opts.request("m", "a cat");
        assert_eq!(req["width"], 512);
        assert!(req.get("height").is_none());
        assert_eq!(req["seed"], 7);
        assert_eq!(req["stream"], true);
        assert!(req.get("steps").is_none());
        assert!(req.get("negative_prompt").is_none());
    }

    #[test]
    fn output_path_slugifies_prompt() {
        let p = output_path("Draw a cat!", "png").display().to_string();
        assert!(p.starts_with("draw-a-cat-"), "{p}");
        assert!(p.ends_with(".png"));
        assert!(output_path("x", "mp4")
            .display()
            .to_string()
            .ends_with(".mp4"));
    }

    #[test]
    fn video_and_audio_requests_use_their_own_fields() {
        let opts = ImageOptions {
            media: Media::Video,
            seconds: 2.0,
            seed: Some(1),
            ..Default::default()
        };
        let v = opts.video_request("m", "waves");
        assert_eq!(v["stream"], false);
        assert_eq!(v["seconds"], 2.0);
        assert!(v.get("response_format").is_none());
        let a = opts.audio_request("m", "rain");
        assert_eq!(a["input"], "rain");
        assert_eq!(a["response_format"], "wav");
        assert_eq!(a["seed"], 1);
    }

    #[test]
    fn set_commands_update_options() {
        let mut opts = ImageOptions::default();
        assert!(matches!(
            handle_command("set width 768", &mut opts),
            Command::Continue
        ));
        assert!(matches!(
            handle_command("set negative blurry, dark", &mut opts),
            Command::Continue
        ));
        assert_eq!(opts.width, 768);
        assert_eq!(opts.negative, "blurry, dark");
        assert!(matches!(handle_command("bye", &mut opts), Command::Quit));
    }
}

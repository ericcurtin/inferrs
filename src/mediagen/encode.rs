//! PNG, WAV and (via ffmpeg) MP4 encoders for the generated media.

use std::io::Write;
use std::process::Command;

use anyhow::{bail, Context as _, Result};

use super::Output;

fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xffff_ffffu32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0xedb8_8320
            } else {
                c >> 1
            };
        }
    }
    !c
}

fn png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// 8-bit RGB PNG, unfiltered scanlines.
pub fn png(rgb: &[u8], width: i64, height: i64) -> Result<Vec<u8>> {
    let (w, h) = (width as usize, height as usize);
    if rgb.len() != w * h * 3 {
        bail!("rgb buffer of {} bytes does not match {w}x{h}", rgb.len());
    }
    let mut raw = Vec::with_capacity((w * 3 + 1) * h);
    for row in rgb.chunks(w * 3) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    enc.write_all(&raw)?;
    let idat = enc.finish()?;
    let mut out = vec![0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    png_chunk(&mut out, b"IHDR", &ihdr);
    png_chunk(&mut out, b"IDAT", &idat);
    png_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

/// 16-bit PCM WAV.
pub fn wav(pcm: &[f32], n_channels: i32, sample_rate: i32) -> Vec<u8> {
    let data_size = (pcm.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_size).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(n_channels as u16).to_le_bytes());
    out.extend_from_slice(&(sample_rate as u32).to_le_bytes());
    out.extend_from_slice(&(sample_rate as u32 * n_channels as u32 * 2).to_le_bytes());
    out.extend_from_slice(&(n_channels as u16 * 2).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    for &v in pcm {
        out.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    out
}

pub fn has_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .args(["-version", "-loglevel", "quiet"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// H.264/AAC MP4 through an ffmpeg subprocess.
pub fn mp4(res: &Output) -> Result<Vec<u8>> {
    if res.n_frames <= 0 || res.rgb.is_empty() {
        bail!("nothing to encode");
    }
    if !has_ffmpeg() {
        bail!("ffmpeg not found");
    }
    let dir = std::env::temp_dir().join(format!(
        "llmman-mediagen-{}-{:08x}",
        std::process::id(),
        super::rand_seed()
    ));
    std::fs::create_dir(&dir).context("creating a temporary directory")?;
    let run = || -> Result<Vec<u8>> {
        let rgb_path = dir.join("frames.rgb");
        let wav_path = dir.join("audio.wav");
        let mp4_path = dir.join("out.mp4");
        std::fs::write(&rgb_path, &res.rgb)?;
        let has_audio = !res.pcm.is_empty();
        if has_audio {
            std::fs::write(&wav_path, wav(&res.pcm, res.n_channels, res.sample_rate))?;
        }
        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
        ])
        .arg(format!("{}x{}", res.width, res.height))
        .arg("-r")
        .arg(format!("{}", res.fps))
        .arg("-i")
        .arg(&rgb_path);
        if has_audio {
            cmd.args(["-f", "wav", "-i"]).arg(&wav_path);
        }
        cmd.args([
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
        ]);
        if has_audio {
            cmd.args(["-c:a", "aac", "-b:a", "192k", "-af", "apad"]);
        }
        cmd.arg("-t")
            .arg(format!("{:.6}", res.n_frames as f64 / res.fps as f64))
            .args(["-f", "mp4"])
            .arg(&mp4_path);
        let status = cmd.status().context("running ffmpeg")?;
        if !status.success() {
            bail!("ffmpeg failed with {status}");
        }
        Ok(std::fs::read(&mp4_path)?)
    };
    let out = run();
    let _ = std::fs::remove_dir_all(&dir);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_has_valid_signature_and_chunks() {
        let p = png(&[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255], 2, 2).unwrap();
        assert_eq!(
            &p[..8],
            &[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n']
        );
        assert_eq!(&p[12..16], b"IHDR");
        assert_eq!(&p[p.len() - 8..p.len() - 4], b"IEND");
        assert_eq!(crc32(b"IEND"), 0xae42_6082);
    }

    #[test]
    fn wav_header_is_44_bytes() {
        let w = wav(&[0.0, 0.5, -0.5, 1.0], 2, 48000);
        assert_eq!(w.len(), 44 + 8);
        assert_eq!(&w[..4], b"RIFF");
        assert_eq!(i16::from_le_bytes([w[50], w[51]]), 32767);
    }
}

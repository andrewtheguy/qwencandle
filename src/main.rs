use anyhow::{bail, Context, Result};
use qwencandle::{
    best_device, parse_device, quantize_to_gguf, Quantization, QwenAsr, DEFAULT_MODEL_ID,
    DEFAULT_QUANTIZATION, SUPPORTED_LANGUAGES,
};
use std::io::Read;
use std::path::PathBuf;

fn print_usage() {
    eprintln!("Usage:");
    eprintln!(
        "  ffmpeg -i INPUT -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | qwencandle [options]"
    );
    eprintln!("  qwencandle quantize --src <id-or-path> --dst <dir> [--dtype q8_0]");
    eprintln!();
    eprintln!("Options:");
    eprintln!(
        "  --model <id>       HuggingFace model ID or local path (default: {DEFAULT_MODEL_ID})"
    );
    eprintln!("  --device <dev>     Device: cpu, metal, cuda (default: auto-detect)");
    eprintln!("  --language <lang>  Force output language (e.g. English, Chinese, Japanese)");
    eprintln!("  --context <text>   Condition on previous text (system prompt for consistency)");
    eprintln!();
    eprintln!("Supported languages:");
    eprintln!("  {}", SUPPORTED_LANGUAGES.join(", "));
    eprintln!();
    eprintln!("The model is automatically downloaded from HuggingFace on first use.");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | qwencandle");
    eprintln!("  ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | qwencandle --device metal");
    eprintln!(
        "  ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | qwencandle -l Japanese"
    );
    eprintln!("  ffmpeg -i audio.mp3 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | qwencandle --context \"Previous sentence.\"");
    eprintln!(
        "  qwencandle quantize --src Qwen/Qwen3-ASR-0.6B --dst ./qwen3-asr-q8_0 --dtype q8_0"
    );
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|arg| arg == "quantize") {
        return run_quantize(&args[2..]);
    }

    let mut model_id: Option<String> = None;
    let mut language: Option<String> = None;
    let mut context: Option<String> = None;
    let mut device_str: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--language" | "-l" => {
                i += 1;
                if i >= args.len() {
                    bail!("--language requires a value");
                }
                language = Some(args[i].clone());
            }
            "--model" | "-m" => {
                i += 1;
                if i >= args.len() {
                    bail!("--model requires a value");
                }
                model_id = Some(args[i].clone());
            }
            "--context" | "-c" => {
                i += 1;
                if i >= args.len() {
                    bail!("--context requires a value");
                }
                context = Some(args[i].clone());
            }
            "--device" | "-d" => {
                i += 1;
                if i >= args.len() {
                    bail!("--device requires a value");
                }
                device_str = Some(args[i].clone());
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => bail!("Unknown argument: {}. Use --help for usage.", args[i]),
        }
        i += 1;
    }

    let model_id = model_id.unwrap_or_else(|| DEFAULT_MODEL_ID.to_string());
    let device = match &device_str {
        Some(d) => parse_device(d)?,
        None => best_device()?,
    };

    let samples = read_wav_stdin()?;
    eprintln!(
        "Audio: {} samples ({:.1}s)",
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    eprintln!("Loading model on {:?}...", device);
    let mut model = QwenAsr::load_on(&model_id, &device)?;

    let text = model.transcribe(
        &samples,
        language.as_deref(),
        context.as_deref(),
        None,
        None,
    )?;
    println!("{text}");

    Ok(())
}

fn run_quantize(args: &[String]) -> Result<()> {
    let mut src: Option<String> = None;
    let mut dst: Option<PathBuf> = None;
    let mut dtype = DEFAULT_QUANTIZATION;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--src" | "-s" => {
                i += 1;
                if i >= args.len() {
                    bail!("--src requires a value");
                }
                src = Some(args[i].clone());
            }
            "--dst" | "-o" => {
                i += 1;
                if i >= args.len() {
                    bail!("--dst requires a value");
                }
                dst = Some(PathBuf::from(&args[i]));
            }
            "--dtype" | "-t" => {
                i += 1;
                if i >= args.len() {
                    bail!("--dtype requires a value");
                }
                dtype = args[i].parse::<Quantization>()?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => bail!(
                "Unknown quantize argument: {}. Use --help for usage.",
                args[i]
            ),
        }
        i += 1;
    }

    let src = src.context("--src is required")?;
    let dst = dst.context("--dst is required")?;

    eprintln!("Quantizing {} to {:?} with {}...", src, dst, dtype.as_str(),);
    let gguf_path = quantize_to_gguf(&src, &dst, dtype)?;
    eprintln!("Wrote {:?}", gguf_path);
    Ok(())
}

/// Read WAV float32 16kHz mono from stdin. Errors on wrong format.
fn read_wav_stdin() -> Result<Vec<f32>> {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf)?;

    if buf.len() < 44 {
        bail!("Input too short for WAV");
    }
    if &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        bail!("Not a WAV file");
    }

    let mut pos = 12;
    let mut fmt_found = false;
    let mut audio_format: u16 = 0;
    let mut num_channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;

    loop {
        if pos + 8 > buf.len() {
            bail!("Could not find data chunk");
        }
        let chunk_id = &buf[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into()?) as usize;

        if chunk_id == b"fmt " && chunk_size >= 16 {
            let d = pos + 8;
            audio_format = u16::from_le_bytes(buf[d..d + 2].try_into()?);
            num_channels = u16::from_le_bytes(buf[d + 2..d + 4].try_into()?);
            sample_rate = u32::from_le_bytes(buf[d + 4..d + 8].try_into()?);
            bits_per_sample = u16::from_le_bytes(buf[d + 14..d + 16].try_into()?);
            fmt_found = true;
        }

        if chunk_id == b"data" {
            if !fmt_found {
                bail!("WAV missing fmt chunk before data");
            }
            if audio_format != 3 || bits_per_sample != 32 {
                bail!(
                    "Wrong WAV format: audio_format={} bits={} (expected float32).\n\
                     Convert with: ffmpeg -i INPUT -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | qwencandle",
                    audio_format, bits_per_sample,
                );
            }
            if num_channels != 1 {
                bail!(
                    "Wrong WAV channels: {} (expected mono).\n\
                     Convert with: ffmpeg -i INPUT -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | qwencandle",
                    num_channels,
                );
            }
            if sample_rate != 16000 {
                bail!(
                    "Wrong WAV sample rate: {} (expected 16000).\n\
                     Convert with: ffmpeg -i INPUT -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | qwencandle",
                    sample_rate,
                );
            }

            let data_start = pos + 8;
            let data_end = std::cmp::min(data_start + chunk_size, buf.len());
            let data = &buf[data_start..data_end];
            let n_samples = data.len() / 4;
            let mut samples = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let bytes: [u8; 4] = data[i * 4..(i + 1) * 4].try_into()?;
                samples.push(f32::from_le_bytes(bytes));
            }
            return Ok(samples);
        }

        pos += 8 + chunk_size;
        if !chunk_size.is_multiple_of(2) {
            pos += 1;
        }
    }
}

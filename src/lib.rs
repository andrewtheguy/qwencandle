pub mod audio;
mod decoder;
mod encoder;
mod gguf;
mod layers;
pub mod tokenizer;

#[cfg(feature = "python")]
mod python;

// Candle's MKL backend references hgemm_ (half-precision GEMM) but intel-mkl-src
// doesn't provide it. This model uses F32 only, so provide a stub to satisfy the linker.
#[cfg(feature = "mkl")]
#[no_mangle]
pub extern "C" fn hgemm_(
    _transa: *const i8,
    _transb: *const i8,
    _m: *const i32,
    _n: *const i32,
    _k: *const i32,
    _alpha: *const u16,
    _a: *const u16,
    _lda: *const i32,
    _b: *const u16,
    _ldb: *const i32,
    _beta: *const u16,
    _c: *mut u16,
    _ldc: *const i32,
) {
    panic!("hgemm_ (f16 matmul) is not supported with MKL — model should use F32");
}

use anyhow::{bail, Context, Result};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::{Api, ApiRepo};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

pub use candle_core::Device;
pub use gguf::{quantize_to_gguf, Quantization, DEFAULT_QUANTIZATION};

/// Returns true if CUDA support was compiled in and a CUDA device can be created.
/// Analogous to `torch.cuda.is_available()`.
pub fn is_cuda_available() -> bool {
    candle_core::utils::cuda_is_available() && Device::new_cuda(0).is_ok()
}

/// Returns true if Metal/MPS support was compiled in and a Metal device can be created.
/// Analogous to `torch.backends.mps.is_available()`.
pub fn is_metal_available() -> bool {
    candle_core::utils::metal_is_available() && Device::new_metal(0).is_ok()
}

/// Select the best available device: CUDA > Metal > CPU.
pub fn best_device() -> Result<Device> {
    if is_cuda_available() {
        return Ok(Device::new_cuda(0)?);
    }
    if is_metal_available() {
        return Ok(Device::new_metal(0)?);
    }
    Ok(Device::Cpu)
}

/// Parse a device name string ("cpu", "metal"/"mps", "cuda") into a Device.
pub fn parse_device(s: &str) -> Result<Device> {
    match s.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" | "mps" => {
            if !candle_core::utils::metal_is_available() {
                bail!("Metal support not compiled. Rebuild with --features metal");
            }
            Ok(Device::new_metal(0)?)
        }
        "cuda" => {
            if !candle_core::utils::cuda_is_available() {
                bail!("CUDA support not compiled. Rebuild with --features cuda");
            }
            Ok(Device::new_cuda(0)?)
        }
        _ => bail!("Unknown device: {}. Supported: cpu, metal, cuda", s),
    }
}

pub const DEFAULT_MODEL_ID: &str = "Qwen/Qwen3-ASR-0.6B";

// Special token IDs
const TOKEN_IM_START: u32 = 151644;
const TOKEN_IM_END: u32 = 151645;
const TOKEN_AUDIO_START: u32 = 151669;
const TOKEN_AUDIO_END: u32 = 151670;
const TOKEN_AUDIO_PAD: u32 = 151676;
const TOKEN_ENDOFTEXT: u32 = 151643;
const TOKEN_ASR_TEXT: u32 = 151704;

const PROMPT_SUFFIX: &[u32] = &[
    TOKEN_AUDIO_END,
    TOKEN_IM_END,
    198,
    TOKEN_IM_START,
    77091,
    198,
];

pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
];

pub struct QwenAsr {
    encoder: encoder::AudioEncoder,
    decoder: decoder::Decoder,
    tokenizer: tokenizer::Tokenizer,
}

impl QwenAsr {
    /// Load model on CPU from a HuggingFace model ID or local directory path.
    pub fn load(model_id: &str) -> Result<Self> {
        Self::load_on(model_id, &Device::Cpu)
    }

    /// Load model on a specific device from a HuggingFace model ID or local directory path.
    pub fn load_on(model_id: &str, device: &Device) -> Result<Self> {
        match resolve_model(model_id)? {
            ModelSource::Safetensors {
                safetensors_paths,
                model_dir,
            } => {
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(&safetensors_paths, DType::F32, device)?
                };

                let encoder = encoder::AudioEncoder::load(vb.pp("thinker.audio_tower"), device)?;
                let tokenizer = tokenizer::Tokenizer::load(&model_dir)?;
                let decoder = decoder::Decoder::load(vb.pp("thinker"), device)?;

                Ok(Self {
                    encoder,
                    decoder,
                    tokenizer,
                })
            }
            ModelSource::Gguf {
                gguf_path,
                model_dir,
            } => {
                let mut loader = gguf::GgufLoader::open(&gguf_path, device)?;
                let encoder = encoder::AudioEncoder::load_gguf(&mut loader, device)?;
                let tokenizer = tokenizer::Tokenizer::load(&model_dir)?;
                let decoder = decoder::Decoder::load_gguf(&mut loader, device)?;
                Ok(Self {
                    encoder,
                    decoder,
                    tokenizer,
                })
            }
        }
    }

    /// Transcribe f32 PCM audio samples (16kHz mono).
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        context: Option<&str>,
    ) -> Result<String> {
        // Validate language
        if let Some(lang) = language {
            if !SUPPORTED_LANGUAGES
                .iter()
                .any(|&l| l.eq_ignore_ascii_case(lang))
            {
                bail!(
                    "Unsupported language: {}\nSupported: {}",
                    lang,
                    SUPPORTED_LANGUAGES.join(", ")
                );
            }
        }

        // Mel spectrogram
        let n_frames = audio::mel_frames(samples.len());
        let mel = audio::compute_mel_spectrogram(samples);

        // Encoder
        let audio_embeds = self.encoder.forward(&mel, n_frames)?;
        let n_audio = audio_embeds.dim(0)?;

        // Tokenize context and language
        let context_tokens = match context {
            Some(ctx) if !ctx.is_empty() => self.tokenizer.encode(ctx)?,
            _ => Vec::new(),
        };

        let lang_tokens = match language {
            Some(lang) => {
                let mut toks = self.tokenizer.encode(&format!("language {}", lang))?;
                toks.push(TOKEN_ASR_TEXT);
                toks
            }
            None => Vec::new(),
        };

        // Build input_ids:
        // <|im_start|>system\n [context] <|im_end|>\n<|im_start|>user\n<|audio_start|> [audio_pads] <|audio_end|><|im_end|>\n<|im_start|>assistant\n [lang_tokens]
        let mut input_ids: Vec<u32> = Vec::new();
        input_ids.push(TOKEN_IM_START);
        input_ids.push(8948); // "system"
        input_ids.push(198); // "\n"
        input_ids.extend_from_slice(&context_tokens);
        input_ids.push(TOKEN_IM_END);
        input_ids.push(198); // "\n"
        input_ids.push(TOKEN_IM_START);
        input_ids.push(872); // "user"
        input_ids.push(198); // "\n"
        input_ids.push(TOKEN_AUDIO_START);
        let prefix_len = input_ids.len();
        input_ids.extend(std::iter::repeat_n(TOKEN_AUDIO_PAD, n_audio));
        input_ids.extend_from_slice(PROMPT_SUFFIX);
        input_ids.extend_from_slice(&lang_tokens);
        let prompt_len = input_ids.len();

        // Embed tokens and replace audio positions
        let input_embeds = self.decoder.embed_tokens(&input_ids)?;
        let before = input_embeds.narrow(0, 0, prefix_len)?;
        let after =
            input_embeds.narrow(0, prefix_len + n_audio, prompt_len - prefix_len - n_audio)?;
        let input_embeds = Tensor::cat(&[&before, &audio_embeds, &after], 0)?;

        // Reset KV cache for fresh transcription
        self.decoder.clear_kv_cache();

        // Prefill
        let prefill_embeds = input_embeds.narrow(0, 0, prompt_len - 1)?;
        self.decoder.forward_embed(&prefill_embeds, 0)?;

        // First token
        let last_embed = input_embeds.narrow(0, prompt_len - 1, 1)?;
        let logits = self.decoder.forward_embed(&last_embed, prompt_len - 1)?;
        let mut token = logits.argmax(1)?.to_vec1::<u32>()?[0];
        let mut generated = vec![token];

        // Autoregressive decode
        let max_new_tokens = 1024;
        for step in 0..max_new_tokens - 1 {
            if token == TOKEN_ENDOFTEXT || token == TOKEN_IM_END {
                break;
            }
            let pos = prompt_len + step;
            let logits = self.decoder.forward_token(token, pos)?;
            token = logits.argmax(1)?.to_vec1::<u32>()?[0];
            generated.push(token);
        }

        // Remove trailing EOS
        while let Some(&last) = generated.last() {
            if last == TOKEN_ENDOFTEXT || last == TOKEN_IM_END {
                generated.pop();
            } else {
                break;
            }
        }

        Ok(self.tokenizer.decode(&generated))
    }
}

enum ModelSource {
    Safetensors {
        safetensors_paths: Vec<PathBuf>,
        model_dir: PathBuf,
    },
    Gguf {
        gguf_path: PathBuf,
        model_dir: PathBuf,
    },
}

/// Resolve model: local GGUF, local safetensors directory, or HuggingFace hub download.
fn resolve_model(model_id: &str) -> Result<ModelSource> {
    let local = PathBuf::from(model_id);
    if is_gguf_file(&local) {
        let model_dir = local
            .parent()
            .map(|parent| {
                if parent.as_os_str().is_empty() {
                    PathBuf::from(".")
                } else {
                    parent.to_path_buf()
                }
            })
            .unwrap_or_else(|| PathBuf::from("."));
        return Ok(ModelSource::Gguf {
            gguf_path: local,
            model_dir,
        });
    }

    if local.is_dir() {
        let gguf_path = local.join("model.gguf");
        if gguf_path.exists() {
            return Ok(ModelSource::Gguf {
                gguf_path,
                model_dir: local,
            });
        }

        let safetensors = find_safetensors(&local)?;
        return Ok(ModelSource::Safetensors {
            safetensors_paths: safetensors,
            model_dir: local,
        });
    }

    let (safetensors_paths, model_dir) = resolve_safetensors_model(model_id)?;
    Ok(ModelSource::Safetensors {
        safetensors_paths,
        model_dir,
    })
}

pub(crate) fn resolve_safetensors_model(model_id: &str) -> Result<(Vec<PathBuf>, PathBuf)> {
    let local = PathBuf::from(model_id);
    if local.is_dir() {
        let safetensors = find_safetensors(&local)?;
        return Ok((safetensors, local));
    }

    if is_gguf_file(&local) {
        bail!(
            "Expected a safetensors source model for quantization, got {:?}",
            local
        );
    }

    let api = Api::new()?;
    let repo = api.model(model_id.to_string());
    let remote_files = repo
        .info()
        .with_context(|| format!("Failed to inspect Hugging Face repo {model_id}"))?
        .siblings
        .into_iter()
        .map(|sibling| sibling.rfilename)
        .collect::<BTreeSet<_>>();

    let (safetensors_paths, model_dir) = download_remote_safetensors(&repo, &remote_files)?;
    download_remote_tokenizer_files(&repo, &remote_files)?;
    Ok((safetensors_paths, model_dir))
}

fn download_remote_safetensors(
    repo: &ApiRepo,
    remote_files: &BTreeSet<String>,
) -> Result<(Vec<PathBuf>, PathBuf)> {
    match pick_remote_checkpoint(remote_files)? {
        RemoteCheckpoint::Single(filename) => {
            let safetensors_path = repo
                .get(&filename)
                .with_context(|| format!("Failed to download {filename}"))?;
            let model_dir = safetensors_path
                .parent()
                .context("Downloaded safetensors path has no parent directory")?
                .to_path_buf();
            Ok((vec![safetensors_path], model_dir))
        }
        RemoteCheckpoint::Indexed(index_filename) => {
            let index_path = repo
                .get(&index_filename)
                .with_context(|| format!("Failed to download {index_filename}"))?;
            let model_dir = index_path
                .parent()
                .context("Downloaded safetensors index path has no parent directory")?
                .to_path_buf();
            let shard_paths = safetensors_from_index(&index_path)?;
            for shard_path in &shard_paths {
                let repo_path = repo_relative_path(shard_path, &model_dir)?;
                repo.get(&repo_path)
                    .with_context(|| format!("Failed to download {repo_path}"))?;
            }
            Ok((shard_paths, model_dir))
        }
    }
}

fn download_remote_tokenizer_files(repo: &ApiRepo, remote_files: &BTreeSet<String>) -> Result<()> {
    let has_tokenizer_json = remote_files.contains("tokenizer.json");
    let has_vocab_json = remote_files.contains("vocab.json");
    let has_merges_txt = remote_files.contains("merges.txt");

    if has_tokenizer_json {
        repo.get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;
    }
    if remote_files.contains("tokenizer_config.json") {
        let _ = repo.get("tokenizer_config.json");
    }
    if has_vocab_json && has_merges_txt {
        repo.get("vocab.json")
            .context("Failed to download vocab.json")?;
        repo.get("merges.txt")
            .context("Failed to download merges.txt")?;
    }

    if has_tokenizer_json || (has_vocab_json && has_merges_txt) {
        return Ok(());
    }

    bail!(
        "Remote model is missing tokenizer.json and vocab.json + merges.txt, so the tokenizer cannot be loaded"
    );
}

fn pick_remote_checkpoint(remote_files: &BTreeSet<String>) -> Result<RemoteCheckpoint> {
    if let Some(index_file) = pick_unique_remote_file(
        remote_files,
        "model.safetensors.index.json",
        ".safetensors.index.json",
    )? {
        return Ok(RemoteCheckpoint::Indexed(index_file));
    }

    if let Some(weight_file) =
        pick_unique_remote_file(remote_files, "model.safetensors", ".safetensors")?
    {
        return Ok(RemoteCheckpoint::Single(weight_file));
    }

    bail!("Remote model does not expose a safetensors checkpoint");
}

fn pick_unique_remote_file(
    remote_files: &BTreeSet<String>,
    preferred_name: &str,
    suffix: &str,
) -> Result<Option<String>> {
    if remote_files.contains(preferred_name) {
        return Ok(Some(preferred_name.to_string()));
    }

    let matches = remote_files
        .iter()
        .filter(|filename| filename.ends_with(suffix))
        .cloned()
        .collect::<Vec<_>>();

    match matches.as_slice() {
        [] => Ok(None),
        [filename] => Ok(Some(filename.clone())),
        _ => bail!(
            "Multiple remote files matched {suffix} but none used the standard name {preferred_name}: {:?}",
            matches
        ),
    }
}

fn repo_relative_path(path: &Path, repo_root: &Path) -> Result<String> {
    let relative = path
        .strip_prefix(repo_root)
        .with_context(|| format!("Path {:?} is not inside {:?}", path, repo_root))?
        .to_str()
        .context("Non-UTF-8 repo path")?;
    Ok(relative.to_string())
}

fn find_safetensors(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        return safetensors_from_index(&index_path);
    }

    let mut indexed = Vec::new();
    let mut singles = Vec::new();
    for entry in std::fs::read_dir(model_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };

        if file_name.ends_with(".safetensors.index.json") {
            indexed.push(path);
        } else if file_name.ends_with(".safetensors") {
            singles.push(path);
        }
    }
    indexed.sort();
    singles.sort();

    match indexed.as_slice() {
        [] => {}
        [index_path] => return safetensors_from_index(index_path),
        _ => bail!(
            "Multiple safetensors index files found in {:?}: {:?}",
            model_dir,
            indexed
        ),
    }

    match singles.as_slice() {
        [] => {}
        [single] => return Ok(vec![single.clone()]),
        _ => bail!(
            "Multiple .safetensors files found in {:?} without an index: {:?}",
            model_dir,
            singles
        ),
    }

    bail!("No safetensors checkpoint found in {:?}", model_dir);
}

fn safetensors_from_index(index_path: &Path) -> Result<Vec<PathBuf>> {
    let index_str = std::fs::read_to_string(index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_str)?;
    let weight_map = index
        .get("weight_map")
        .context("Missing weight_map in index")?
        .as_object()
        .context("weight_map not object")?;

    let model_dir = index_path
        .parent()
        .context("Safetensors index path has no parent directory")?;
    let mut shards = BTreeSet::new();
    for shard in weight_map.values() {
        if let Some(filename) = shard.as_str() {
            shards.insert(filename.to_string());
        }
    }

    Ok(shards
        .iter()
        .map(|filename| model_dir.join(filename))
        .collect())
}

fn is_gguf_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RemoteCheckpoint {
    Single(String),
    Indexed(String),
}

#[cfg(test)]
mod tests {
    use super::{find_safetensors, pick_remote_checkpoint, RemoteCheckpoint};
    use anyhow::Result;
    use std::{
        collections::BTreeSet,
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new() -> Result<Self> {
            let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
            let path = std::env::temp_dir().join(format!(
                "qwencandle-test-{}-{}",
                std::process::id(),
                unique
            ));
            fs::create_dir_all(&path)?;
            Ok(Self { path })
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn find_safetensors_reads_sharded_index() -> Result<()> {
        let tempdir = TempDir::new()?;
        fs::write(
            tempdir.path().join("model.safetensors.index.json"),
            r#"{
                "weight_map": {
                    "layer1": "model-00002-of-00002.safetensors",
                    "layer0": "model-00001-of-00002.safetensors",
                    "layer2": "model-00001-of-00002.safetensors"
                }
            }"#,
        )?;

        let shards = find_safetensors(tempdir.path())?;

        assert_eq!(
            shards,
            vec![
                tempdir.path().join("model-00001-of-00002.safetensors"),
                tempdir.path().join("model-00002-of-00002.safetensors"),
            ]
        );
        Ok(())
    }

    #[test]
    fn find_safetensors_accepts_single_nonstandard_file() -> Result<()> {
        let tempdir = TempDir::new()?;
        let weights = tempdir.path().join("weights.safetensors");
        fs::write(&weights, b"")?;

        let shards = find_safetensors(tempdir.path())?;

        assert_eq!(shards, vec![weights]);
        Ok(())
    }

    #[test]
    fn pick_remote_checkpoint_prefers_index() -> Result<()> {
        let remote_files = BTreeSet::from([
            "model-00001-of-00002.safetensors".to_string(),
            "model-00002-of-00002.safetensors".to_string(),
            "model.safetensors.index.json".to_string(),
        ]);

        let checkpoint = pick_remote_checkpoint(&remote_files)?;

        assert_eq!(
            checkpoint,
            RemoteCheckpoint::Indexed("model.safetensors.index.json".to_string())
        );
        Ok(())
    }
}

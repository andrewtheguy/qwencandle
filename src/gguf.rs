use anyhow::{bail, Context, Result};
use candle_core::{
    quantized::{gguf_file, GgmlDType, QMatMul, QTensor},
    safetensors::MmapedSafetensors,
    DType, Device, Tensor,
};
use std::{
    fs,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    str::FromStr,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Quantization {
    F16,
    BF16,
    Q4_0,
    Q5_0,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

pub const DEFAULT_QUANTIZATION: Quantization = Quantization::Q8_0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LmHeadPolicy {
    Quantized,
    Tied,
}

pub const DEFAULT_LM_HEAD_POLICY: LmHeadPolicy = LmHeadPolicy::Quantized;

impl Quantization {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::Q4_0 => "q4_0",
            Self::Q5_0 => "q5_0",
            Self::Q8_0 => "q8_0",
            Self::Q4K => "q4k",
            Self::Q5K => "q5k",
            Self::Q6K => "q6k",
            Self::Q8K => "q8k",
        }
    }

    pub fn ggml_dtype(self) -> GgmlDType {
        match self {
            Self::F16 => GgmlDType::F16,
            Self::BF16 => GgmlDType::BF16,
            Self::Q4_0 => GgmlDType::Q4_0,
            Self::Q5_0 => GgmlDType::Q5_0,
            Self::Q8_0 => GgmlDType::Q8_0,
            Self::Q4K => GgmlDType::Q4K,
            Self::Q5K => GgmlDType::Q5K,
            Self::Q6K => GgmlDType::Q6K,
            Self::Q8K => GgmlDType::Q8K,
        }
    }
}

impl FromStr for Quantization {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::BF16),
            "q4_0" => Ok(Self::Q4_0),
            "q5_0" => Ok(Self::Q5_0),
            "q8_0" => Ok(Self::Q8_0),
            "q4k" => Ok(Self::Q4K),
            "q5k" => Ok(Self::Q5K),
            "q6k" => Ok(Self::Q6K),
            "q8k" => Ok(Self::Q8K),
            _ => {
                bail!("Unknown quantization: {s}. Supported: f16, bf16, q4_0, q5_0, q8_0, q4k, q5k, q6k, q8k")
            }
        }
    }
}

impl LmHeadPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Quantized => "quantized",
            Self::Tied => "tied",
        }
    }
}

impl FromStr for LmHeadPolicy {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "quantized" => Ok(Self::Quantized),
            "tied" => Ok(Self::Tied),
            _ => bail!("Unknown lm_head policy: {s}. Supported: quantized, tied"),
        }
    }
}

pub struct GgufLoader {
    reader: BufReader<File>,
    content: gguf_file::Content,
    device: Device,
}

impl GgufLoader {
    pub fn open(path: &Path, device: &Device) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("Failed to open {:?}", path))?;
        let mut reader = BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader)
            .with_context(|| format!("Failed to read GGUF header from {:?}", path))?;
        Ok(Self {
            reader,
            content,
            device: device.clone(),
        })
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }

    pub fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let qtensor = self.qtensor(name)?;
        QMatMul::from_qtensor(qtensor).with_context(|| format!("Failed to load QMatMul {name}"))
    }

    pub fn tensor(&mut self, name: &str) -> Result<Tensor> {
        self.qtensor(name)?
            .dequantize(&self.device)
            .with_context(|| format!("Failed to dequantize tensor {name}"))
    }

    pub fn tensor_f32(&mut self, name: &str) -> Result<Tensor> {
        let tensor = self.tensor(name)?;
        if tensor.dtype() == DType::F32 {
            Ok(tensor)
        } else {
            tensor
                .to_dtype(DType::F32)
                .with_context(|| format!("Failed to cast tensor {name} to f32"))
        }
    }

    pub fn embedding_tensor(&mut self, name: &str) -> Result<Tensor> {
        let tensor = self.tensor(name)?;
        if tensor.dtype() == DType::F16 {
            Ok(tensor)
        } else {
            tensor
                .to_dtype(DType::F16)
                .with_context(|| format!("Failed to cast embedding {name} to f16"))
        }
    }

    fn qtensor(&mut self, name: &str) -> Result<QTensor> {
        self.content
            .tensor(&mut self.reader, name, &self.device)
            .with_context(|| format!("Failed to load tensor {name} from GGUF"))
    }
}

pub fn quantize_to_gguf(
    src_model_id: &str,
    dst_dir: &Path,
    quantization: Quantization,
    lm_head_policy: LmHeadPolicy,
) -> Result<PathBuf> {
    let (safetensors_paths, model_dir) = crate::resolve_safetensors_model(src_model_id)?;
    let tensors = unsafe { MmapedSafetensors::multi(&safetensors_paths)? };

    fs::create_dir_all(dst_dir).with_context(|| format!("Failed to create {:?}", dst_dir))?;

    let mut names: Vec<String> = tensors
        .tensors()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    names.sort();

    let mut qtensors = Vec::with_capacity(names.len() + 1);
    for name in &names {
        if is_runtime_lm_head_weight(name) {
            continue;
        }
        let tensor = tensors
            .load(name, &Device::Cpu)
            .with_context(|| format!("Failed to load source tensor {name}"))?;
        validate_quantization_target(name, &tensor, quantization)?;
        let dtype = export_dtype_for(name, quantization);
        let qtensor = QTensor::quantize(&tensor, dtype)
            .with_context(|| format!("Failed to quantize {name} as {:?}", dtype))?;
        qtensors.push((name.clone(), qtensor));
    }

    if lm_head_policy == LmHeadPolicy::Quantized {
        let embed_name = "thinker.model.embed_tokens.weight";
        let embed_tensor = tensors
            .load(embed_name, &Device::Cpu)
            .with_context(|| format!("Failed to load source tensor {embed_name}"))?;
        validate_quantization_target(RUNTIME_LM_HEAD_WEIGHT, &embed_tensor, quantization)?;
        let lm_head =
            QTensor::quantize(&embed_tensor, quantization.ggml_dtype()).with_context(|| {
                format!(
                    "Failed to quantize {embed_name} as {:?} for lm_head",
                    quantization.ggml_dtype()
                )
            })?;
        qtensors.push((RUNTIME_LM_HEAD_WEIGHT.to_string(), lm_head));
    }

    let alignment = gguf_file::Value::U32(gguf_file::DEFAULT_ALIGNMENT as u32);
    let architecture = gguf_file::Value::String("qwencandle".to_string());
    let model_id = gguf_file::Value::String(src_model_id.to_string());
    let quantized = gguf_file::Value::String(quantization.as_str().to_string());
    let lm_head_policy_value = gguf_file::Value::String(lm_head_policy.as_str().to_string());
    let metadata = vec![
        ("general.alignment", &alignment),
        ("general.architecture", &architecture),
        ("qwencandle.source_model", &model_id),
        ("qwencandle.quantization", &quantized),
        ("qwencandle.lm_head_policy", &lm_head_policy_value),
    ];

    let tensor_refs: Vec<(&str, &QTensor)> = qtensors
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();

    let gguf_path = dst_dir.join("model.gguf");
    let mut file =
        File::create(&gguf_path).with_context(|| format!("Failed to create {:?}", gguf_path))?;
    gguf_file::write(&mut file, &metadata, &tensor_refs)
        .with_context(|| format!("Failed to write {:?}", gguf_path))?;

    copy_tokenizer_files(&model_dir, dst_dir)?;

    Ok(gguf_path)
}

const RUNTIME_LM_HEAD_WEIGHT: &str = "thinker.model.lm_head.weight";

fn is_runtime_lm_head_weight(name: &str) -> bool {
    name == RUNTIME_LM_HEAD_WEIGHT
}

fn export_dtype_for(name: &str, quantization: Quantization) -> GgmlDType {
    if is_quantized_linear_weight(name) {
        quantization.ggml_dtype()
    } else if is_f32_tensor(name) {
        GgmlDType::F32
    } else {
        GgmlDType::F16
    }
}

fn validate_quantization_target(
    name: &str,
    tensor: &Tensor,
    quantization: Quantization,
) -> Result<()> {
    if !is_quantized_linear_weight(name) && !is_runtime_lm_head_weight(name) {
        return Ok(());
    }

    let block_size = quantization.ggml_dtype().block_size();
    let last_dim = *tensor
        .dims()
        .last()
        .with_context(|| format!("Tensor {name} has no dimensions"))?;

    if last_dim.is_multiple_of(block_size) {
        return Ok(());
    }

    bail!(
        "Cannot quantize {name} with {}: last dim {} is not divisible by block size {}. \
Use a compatible dtype such as q8_0, q5_0, or q4_0 instead.",
        quantization.as_str(),
        last_dim,
        block_size,
    );
}

fn is_quantized_linear_weight(name: &str) -> bool {
    matches!(
        name,
        "thinker.audio_tower.conv_out.weight"
            | "thinker.audio_tower.proj1.weight"
            | "thinker.audio_tower.proj2.weight"
    ) || name.ends_with(".self_attn.q_proj.weight")
        || name.ends_with(".self_attn.k_proj.weight")
        || name.ends_with(".self_attn.v_proj.weight")
        || name.ends_with(".self_attn.out_proj.weight")
        || name.ends_with(".mlp.gate_proj.weight")
        || name.ends_with(".mlp.up_proj.weight")
        || name.ends_with(".mlp.down_proj.weight")
        || name.ends_with(".fc1.weight")
        || name.ends_with(".fc2.weight")
}

fn is_f32_tensor(name: &str) -> bool {
    name.ends_with(".bias")
        || name.ends_with(".q_norm.weight")
        || name.ends_with(".k_norm.weight")
        || name.ends_with(".norm.weight")
        || name.ends_with(".ln_post.weight")
        || name.contains("layer_norm")
        || name.contains("layernorm")
}

fn copy_tokenizer_files(src_dir: &Path, dst_dir: &Path) -> Result<()> {
    for file_name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "config.json",
    ] {
        let src = src_dir.join(file_name);
        if src.exists() {
            let dst = dst_dir.join(file_name);
            fs::copy(&src, &dst)
                .with_context(|| format!("Failed to copy {:?} to {:?}", src, dst))?;
        }
    }
    Ok(())
}

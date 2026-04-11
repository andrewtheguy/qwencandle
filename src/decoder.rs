use crate::{
    gguf::GgufLoader,
    layers::{LinearLayer, OutputProjection},
};
/// Qwen3 LLM Decoder for ASR.
/// GQA with Q/K RMSNorm, NeoX RoPE, KV cache, SwiGLU Mlp, tied embeddings.
use anyhow::Result as AnyResult;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{kv_cache::ConcatKvCache, linear_no_bias, Embedding, RmsNorm, VarBuilder};

#[derive(Clone)]
pub struct DecoderConfig {
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub rms_eps: f64,
    pub max_seq_len: usize,
}

// ── RoPE ────────────────────────────────────────────────────────────────────

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, device: &Device, cfg: &DecoderConfig) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..cfg.head_dim)
            .step_by(2)
            .map(|i| 1.0 / cfg.rope_theta.powf(i as f64 / cfg.head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, cfg.max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE. q,k shape: [B, H, L, D]
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ── Attention ───────────────────────────────────────────────────────────────

struct Attention {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    kv_cache: ConcatKvCache,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(vb: VarBuilder, cfg: &DecoderConfig) -> Result<Self> {
        Ok(Self {
            q_proj: LinearLayer::from_linear(linear_no_bias(
                cfg.hidden_size,
                cfg.n_heads * cfg.head_dim,
                vb.pp("q_proj"),
            )?),
            k_proj: LinearLayer::from_linear(linear_no_bias(
                cfg.hidden_size,
                cfg.n_kv_heads * cfg.head_dim,
                vb.pp("k_proj"),
            )?),
            v_proj: LinearLayer::from_linear(linear_no_bias(
                cfg.hidden_size,
                cfg.n_kv_heads * cfg.head_dim,
                vb.pp("v_proj"),
            )?),
            o_proj: LinearLayer::from_linear(linear_no_bias(
                cfg.n_heads * cfg.head_dim,
                cfg.hidden_size,
                vb.pp("o_proj"),
            )?),
            q_norm: candle_nn::rms_norm(cfg.head_dim, cfg.rms_eps, vb.pp("q_norm"))?,
            k_norm: candle_nn::rms_norm(cfg.head_dim, cfg.rms_eps, vb.pp("k_norm"))?,
            kv_cache: ConcatKvCache::new(2), // dim=2 for [B,H,S,D]
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn load_gguf(loader: &mut GgufLoader, prefix: &str, cfg: &DecoderConfig) -> AnyResult<Self> {
        Ok(Self {
            q_proj: load_quantized_linear(loader, &format!("{prefix}.q_proj"))?,
            k_proj: load_quantized_linear(loader, &format!("{prefix}.k_proj"))?,
            v_proj: load_quantized_linear(loader, &format!("{prefix}.v_proj"))?,
            o_proj: load_quantized_linear(loader, &format!("{prefix}.o_proj"))?,
            q_norm: load_rms_norm(loader, &format!("{prefix}.q_norm"), cfg.rms_eps)?,
            k_norm: load_rms_norm(loader, &format!("{prefix}.k_norm"), cfg.rms_eps)?,
            kv_cache: ConcatKvCache::new(2),
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rotary: &RotaryEmbedding,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape: [B, L, H, D] → [B, H, L, D]
        let q = q
            .reshape((b, l, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head RMSNorm on Q and K
        let q_flat = q.flatten(0, 2)?; // [B*H*L, D]
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.n_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.n_kv_heads, l, self.head_dim))?;

        // RoPE
        let (q, k) = rotary.apply(&q, &k, offset)?;

        // KV cache
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // GQA: repeat KV heads
        let n_groups = self.n_heads / self.n_kv_heads;
        let k = if n_groups > 1 {
            let (b_sz, n_kv, seq, hd) = k.dims4()?;
            Tensor::cat(&vec![&k; n_groups], 2)?
                .reshape((b_sz, n_kv * n_groups, seq, hd))?
                .contiguous()?
        } else {
            k.contiguous()?
        };
        let v = if n_groups > 1 {
            let (b_sz, n_kv, seq, hd) = v.dims4()?;
            Tensor::cat(&vec![&v; n_groups], 2)?
                .reshape((b_sz, n_kv * n_groups, seq, hd))?
                .contiguous()?
        } else {
            v.contiguous()?
        };

        // Attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        // Merge heads
        ctx.transpose(1, 2)?
            .reshape((b, l, self.n_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

// ── Mlp ─────────────────────────────────────────────────────────────────────

struct Mlp {
    gate_proj: LinearLayer,
    up_proj: LinearLayer,
    down_proj: LinearLayer,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &DecoderConfig) -> Result<Self> {
        Ok(Self {
            gate_proj: LinearLayer::from_linear(linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate,
                vb.pp("gate_proj"),
            )?),
            up_proj: LinearLayer::from_linear(linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate,
                vb.pp("up_proj"),
            )?),
            down_proj: LinearLayer::from_linear(linear_no_bias(
                cfg.intermediate,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?),
        })
    }

    fn load_gguf(loader: &mut GgufLoader, prefix: &str) -> AnyResult<Self> {
        Ok(Self {
            gate_proj: load_quantized_linear(loader, &format!("{prefix}.gate_proj"))?,
            up_proj: load_quantized_linear(loader, &format!("{prefix}.up_proj"))?,
            down_proj: load_quantized_linear(loader, &format!("{prefix}.down_proj"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

// ── Decoder Layer ───────────────────────────────────────────────────────────

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn load(vb: VarBuilder, cfg: &DecoderConfig) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::load(vb.pp("self_attn"), cfg)?,
            mlp: Mlp::load(vb.pp("mlp"), cfg)?,
            input_layernorm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn load_gguf(loader: &mut GgufLoader, prefix: &str, cfg: &DecoderConfig) -> AnyResult<Self> {
        Ok(Self {
            self_attn: Attention::load_gguf(loader, &format!("{prefix}.self_attn"), cfg)?,
            mlp: Mlp::load_gguf(loader, &format!("{prefix}.mlp"))?,
            input_layernorm: load_rms_norm(
                loader,
                &format!("{prefix}.input_layernorm"),
                cfg.rms_eps,
            )?,
            post_attention_layernorm: load_rms_norm(
                loader,
                &format!("{prefix}.post_attention_layernorm"),
                cfg.rms_eps,
            )?,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rotary: &RotaryEmbedding,
        offset: usize,
    ) -> Result<Tensor> {
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, mask, rotary, offset)?;
        let x = (x + h)?;
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = h.apply(&self.mlp)?;
        x + h
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ── Full Decoder ────────────────────────────────────────────────────────────

pub struct Decoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: OutputProjection,
    rotary: RotaryEmbedding,
    device: Device,
    dtype: DType,
}

impl Decoder {
    pub fn load(vb: VarBuilder, device: &Device, cfg: &DecoderConfig) -> Result<Self> {
        let vb_model = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        let vb_l = vb_model.pp("layers");
        for i in 0..cfg.n_layers {
            layers.push(DecoderLayer::load(vb_l.pp(i), cfg)?);
        }

        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_eps, vb_model.pp("norm"))?;

        // Tied embeddings: lm_head shares weights with embed_tokens
        let lm_head = OutputProjection::Tensor(embed_tokens.embeddings().clone());

        let rotary = RotaryEmbedding::new(vb.dtype(), device, cfg)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
            device: device.clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn load_gguf(
        loader: &mut GgufLoader,
        device: &Device,
        cfg: &DecoderConfig,
    ) -> AnyResult<Self> {
        let embed_tokens = Embedding::new(
            loader.embedding_tensor("thinker.model.embed_tokens.weight")?,
            cfg.hidden_size,
        );

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(DecoderLayer::load_gguf(
                loader,
                &format!("thinker.model.layers.{i}"),
                cfg,
            )?);
        }

        let norm = load_rms_norm(loader, "thinker.model.norm", cfg.rms_eps)?;
        let lm_head = if loader.has_tensor("thinker.model.lm_head.weight") {
            OutputProjection::Quantized(loader.qmatmul("thinker.model.lm_head.weight")?)
        } else {
            OutputProjection::Tensor(embed_tokens.embeddings().clone())
        };

        let rotary = RotaryEmbedding::new(DType::F32, device, cfg)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
            device: device.clone(),
            dtype: DType::F32,
        })
    }

    /// Embed a single token ID.
    pub fn embed_token(&self, token_id: u32) -> Result<Tensor> {
        let ids = Tensor::from_vec(vec![token_id], (1,), &self.device)?;
        let embeds = self.embed_tokens.forward(&ids)?;
        if embeds.dtype() == DType::F32 {
            Ok(embeds)
        } else {
            embeds.to_dtype(DType::F32)
        }
    }

    /// Embed multiple token IDs. Returns [seq, hidden].
    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::from_vec(token_ids.to_vec(), (token_ids.len(),), &self.device)?;
        let embeds = self.embed_tokens.forward(&ids)?;
        if embeds.dtype() == DType::F32 {
            Ok(embeds)
        } else {
            embeds.to_dtype(DType::F32)
        }
    }

    fn causal_mask(&self, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<f32> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| if j <= i + offset { 0.0 } else { minf })
            })
            .collect();
        Tensor::from_slice(&mask, (1, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    /// Forward pass on embeddings. Returns hidden states [B, L, hidden].
    fn forward_hidden(&mut self, h: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l, _) = h.dims3()?;
        let mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(l, offset)?)
        };

        let mut h = h.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h, mask.as_ref(), &self.rotary, offset)?;
        }
        self.norm.forward(&h)
    }

    /// Forward pass with pre-built embeddings (for prefill with audio embeddings).
    /// embeds: [seq, hidden] → returns logits [vocab]
    pub fn forward_embed(&mut self, embeds: &Tensor, offset: usize) -> Result<Tensor> {
        let embeds_3d = embeds.unsqueeze(0)?; // [1, seq, hidden]
        let h = self.forward_hidden(&embeds_3d, offset)?;
        // Take last position
        let seq_len = h.dim(1)?;
        let last = h.narrow(1, seq_len - 1, 1)?.squeeze(1)?; // [1, hidden]
        self.lm_head.forward(&last)
    }

    /// Forward pass for a single token during autoregressive generation.
    /// Returns logits [1, vocab].
    pub fn forward_token(&mut self, token_id: u32, offset: usize) -> Result<Tensor> {
        let embed = self.embed_token(token_id)?; // [1, hidden]
        let embed_3d = embed.unsqueeze(0)?; // [1, 1, hidden]
        let h = self.forward_hidden(&embed_3d, offset)?;
        let h = h.squeeze(1)?; // [1, hidden]
        self.lm_head.forward(&h)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

fn load_quantized_linear(loader: &mut GgufLoader, prefix: &str) -> AnyResult<LinearLayer> {
    Ok(LinearLayer::from_quantized(
        loader.qmatmul(&format!("{prefix}.weight"))?,
        None,
    ))
}

fn load_rms_norm(loader: &mut GgufLoader, prefix: &str, eps: f64) -> AnyResult<RmsNorm> {
    Ok(RmsNorm::new(
        loader.tensor_f32(&format!("{prefix}.weight"))?,
        eps,
    ))
}

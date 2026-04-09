/// Minimal decode-only BPE tokenizer for Qwen3-ASR.
/// Loads vocab.json, decodes token IDs to text via GPT-2 byte mapping.
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

const TOKEN_ASR_TEXT: u32 = 151704;

pub struct Tokenizer {
    id_to_token: HashMap<u32, String>,
    byte_decoder: HashMap<char, u8>,
    special_tokens: std::collections::HashSet<u32>,
}

impl Tokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let vocab_path = model_dir.join("vocab.json");
        let vocab_str = std::fs::read_to_string(&vocab_path)?;
        let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_str)?;
        let id_to_token: HashMap<u32, String> = vocab.into_iter().map(|(k, v)| (v, k)).collect();

        // Load special token IDs from tokenizer_config.json
        let mut special_tokens = std::collections::HashSet::new();
        let tc_path = model_dir.join("tokenizer_config.json");
        if tc_path.exists() {
            let tc_str = std::fs::read_to_string(&tc_path)?;
            let tc: serde_json::Value = serde_json::from_str(&tc_str)?;
            if let Some(added) = tc.get("added_tokens_decoder").and_then(|v| v.as_object()) {
                for tid_str in added.keys() {
                    if let Ok(tid) = tid_str.parse::<u32>() {
                        special_tokens.insert(tid);
                    }
                }
            }
        }

        let byte_encoder = bytes_to_unicode();
        let byte_decoder: HashMap<char, u8> = byte_encoder.into_iter().map(|(b, c)| (c, b)).collect();

        Ok(Self {
            id_to_token,
            byte_decoder,
            special_tokens,
        })
    }

    /// Decode token IDs to text, then parse ASR output (strip "language X<asr_text>" prefix).
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut pieces = Vec::new();
        for &tid in token_ids {
            if self.special_tokens.contains(&tid) {
                if tid == TOKEN_ASR_TEXT {
                    pieces.push("<asr_text>".to_string());
                }
                continue;
            }
            if let Some(tok) = self.id_to_token.get(&tid) {
                pieces.push(tok.clone());
            }
        }
        let text = pieces.join("");
        // GPT-2 byte-level decode
        let bytes: Vec<u8> = text
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect();
        let decoded = String::from_utf8_lossy(&bytes).to_string();

        // Parse ASR output: split on <asr_text>, take text after
        if let Some(pos) = decoded.find("<asr_text>") {
            decoded[pos + "<asr_text>".len()..].trim().to_string()
        } else {
            decoded.trim().to_string()
        }
    }
}

/// GPT-2 style byte-to-unicode mapping used by Qwen2 tokenizer.
fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = Vec::new();
    // printable ASCII range
    bs.extend(b'!'..=b'~');
    // Latin-1 supplement
    bs.extend(0xa1u8..=0xac);
    bs.extend(0xaeu8..=0xff);

    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n: u32 = 0;
    for b in 0u16..=255 {
        let b = b as u8;
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    bs.into_iter()
        .zip(cs.into_iter())
        .map(|(b, c)| (b, char::from_u32(c).unwrap()))
        .collect()
}

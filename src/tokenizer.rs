/// Tokenizer wrapper around HuggingFace tokenizers crate.
/// Loads from tokenizer.json if available, otherwise builds from vocab.json + merges.txt.
use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::models::bpe::BPE;

const TOKEN_ASR_TEXT: u32 = 151704;

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let json_path = model_dir.join("tokenizer.json");
        let inner = if json_path.exists() {
            tokenizers::Tokenizer::from_file(&json_path)
                .map_err(|e| anyhow::anyhow!("{}", e))
                .with_context(|| format!("Failed to load {:?}", json_path))?
        } else {
            // Build from vocab.json + merges.txt
            let vocab_path = model_dir.join("vocab.json");
            let merges_path = model_dir.join("merges.txt");
            let bpe = BPE::from_file(
                vocab_path.to_str().context("non-UTF-8 vocab path")?,
                merges_path.to_str().context("non-UTF-8 merges path")?,
            )
            .build()
            .map_err(|e| anyhow::anyhow!("{}", e))?;

            let mut tok = tokenizers::Tokenizer::new(bpe);
            // Qwen2 uses byte-level pre-tokenizer
            tok.with_pre_tokenizer(Some(
                tokenizers::pre_tokenizers::byte_level::ByteLevel::default(),
            ));
            tok.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
            tok
        };
        Ok(Self { inner })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode token IDs to text, then parse ASR output (strip "language X<asr_text>" prefix).
    pub fn decode(&self, token_ids: &[u32]) -> String {
        // If <asr_text> token present, decode only tokens after it
        if let Some(pos) = token_ids.iter().position(|&t| t == TOKEN_ASR_TEXT) {
            let after = &token_ids[pos + 1..];
            self.inner
                .decode(after, true)
                .unwrap_or_default()
                .trim()
                .to_string()
        } else {
            self.inner
                .decode(token_ids, true)
                .unwrap_or_default()
                .trim()
                .to_string()
        }
    }
}

/// Collapse character runs longer than `threshold` to a single occurrence.
/// Ported from Qwen3-ASR reference: `detect_and_fix_repetitions`.
fn fix_char_repeats(s: &str, threshold: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut result = String::with_capacity(n);
    let mut i = 0;
    while i < n {
        let ch = chars[i];
        let mut count = 1;
        while i + count < n && chars[i + count] == ch {
            count += 1;
        }
        if count > threshold {
            result.push(ch);
        } else {
            for _ in 0..count {
                result.push(ch);
            }
        }
        i += count;
    }
    result
}

/// Collapse repeated patterns longer than `threshold` repetitions to a single occurrence.
/// Ported from Qwen3-ASR reference: `detect_and_fix_repetitions`.
fn fix_pattern_repeats(s: &str, threshold: usize, max_pattern_len: usize) -> String {
    let mut chars: Vec<char> = s.chars().collect();
    let min_repeat_chars = threshold * 2;
    let mut result = String::with_capacity(chars.len());

    // Outer loop replaces the recursion: after collapsing a repeated pattern we
    // restart scanning on the remainder instead of recursing.
    loop {
        let n = chars.len();
        if n < min_repeat_chars {
            for &ch in &chars {
                result.push(ch);
            }
            break;
        }

        let mut i = 0;
        let mut found = false;

        while i <= n.saturating_sub(min_repeat_chars) {
            for k in 1..=max_pattern_len {
                if i + k * threshold > n {
                    break;
                }
                let pattern = &chars[i..i + k];
                let mut valid = true;
                for rep in 1..threshold {
                    let start_idx = i + rep * k;
                    if start_idx + k > n || chars[start_idx..start_idx + k] != *pattern {
                        valid = false;
                        break;
                    }
                }
                if valid {
                    // Count total repetitions beyond threshold
                    let mut end_index = i + threshold * k;
                    while end_index + k <= n && chars[end_index..end_index + k] == *pattern {
                        end_index += k;
                    }
                    // Emit everything before the match + one occurrence of the pattern
                    for &ch in &chars[..i] {
                        result.push(ch);
                    }
                    for &ch in pattern {
                        result.push(ch);
                    }
                    // Continue scanning the remainder
                    chars = chars[end_index..].to_vec();
                    found = true;
                    break;
                }
            }
            if found {
                break;
            }
            i += 1;
        }

        if !found {
            // No repetition found in remaining chars — emit them all
            for &ch in &chars {
                result.push(ch);
            }
            break;
        }
    }

    result
}

/// Detect and fix repetitive output from the ASR model.
/// Matches the reference implementation's `detect_and_fix_repetitions`.
pub fn detect_and_fix_repetitions(text: &str, threshold: usize) -> String {
    let text = fix_char_repeats(text, threshold);
    fix_pattern_repeats(&text, threshold, 20)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_repeats_collapsed() {
        // 25 'a's should collapse to 1 (threshold=20)
        let input = "a".repeat(25);
        assert_eq!(fix_char_repeats(&input, 20), "a");
    }

    #[test]
    fn char_repeats_below_threshold_kept() {
        let input = "a".repeat(19);
        assert_eq!(fix_char_repeats(&input, 20), input);
    }

    #[test]
    fn pattern_repeats_collapsed() {
        // "hello" repeated 25 times should collapse to 1
        let input = "hello".repeat(25);
        assert_eq!(fix_pattern_repeats(&input, 20, 20), "hello");
    }

    #[test]
    fn pattern_repeats_below_threshold_kept() {
        let input = "hello".repeat(5);
        assert_eq!(fix_pattern_repeats(&input, 20, 20), input);
    }

    #[test]
    fn mixed_repetitions() {
        let input = format!("Good morning. {}", "Hello world. ".repeat(25));
        let result = detect_and_fix_repetitions(&input, 20);
        assert_eq!(result, "Good morning. Hello world. ");
    }

    #[test]
    fn no_repetitions_unchanged() {
        let input = "And so my fellow Americans, ask not what your country can do for you.";
        assert_eq!(detect_and_fix_repetitions(input, 20), input);
    }

    #[test]
    fn char_repeats_cjk() {
        let input = "的".repeat(25);
        assert_eq!(fix_char_repeats(&input, 20), "的");
        // Below threshold — kept
        let input = "的".repeat(19);
        assert_eq!(fix_char_repeats(&input, 20), input);
    }

    #[test]
    fn char_repeats_emoji() {
        let input = "😊".repeat(25);
        assert_eq!(fix_char_repeats(&input, 20), "😊");
        let input = "😊".repeat(10);
        assert_eq!(fix_char_repeats(&input, 20), input);
    }

    #[test]
    fn pattern_repeats_cjk() {
        let pattern = "你好世界";
        let input = pattern.repeat(25);
        assert_eq!(fix_pattern_repeats(&input, 20, 20), pattern);
        // Below threshold — kept
        let input = pattern.repeat(5);
        assert_eq!(fix_pattern_repeats(&input, 20, 20), input);
    }

    #[test]
    fn pattern_repeats_emoji() {
        let pattern = "🎵🎶";
        let input = pattern.repeat(25);
        assert_eq!(fix_pattern_repeats(&input, 20, 20), pattern);
        let input = pattern.repeat(3);
        assert_eq!(fix_pattern_repeats(&input, 20, 20), input);
    }

    #[test]
    fn mixed_unicode_repetitions() {
        let input = format!("早上好。{}", "谢谢。".repeat(25));
        let result = detect_and_fix_repetitions(&input, 20);
        assert_eq!(result, "早上好。谢谢。");
    }
}

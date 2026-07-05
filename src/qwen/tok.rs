//! Tokenizer + chat template for the Qwen3 family (both target and draft share it).
//!
//! The engine works on raw token ids; this is the text boundary. M0 hardcodes the simple
//! Qwen3 ChatML-style template (single-turn, thinking disabled) instead of parsing the
//! repo's jinja template.

use anyhow::{anyhow, Context, Result};

pub struct ChatTokenizer {
    inner: tokenizers::Tokenizer,
}

impl ChatTokenizer {
    /// Download (HF cache) + load the repo's `tokenizer.json`.
    pub fn load(repo: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().context("hf-hub api")?;
        let path = api
            .model(repo.to_string())
            .get("tokenizer.json")
            .with_context(|| format!("download tokenizer.json for {repo}"))?;
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| anyhow!("load tokenizer: {e}"))?;
        Ok(Self { inner })
    }

    /// Encode a single-turn user message with the Qwen3 chat template (thinking disabled —
    /// the empty think block is what the official template emits with enable_thinking=false).
    pub fn encode_chat(&self, user: &str) -> Result<Vec<i32>> {
        let text = format!(
            "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let enc = self.inner.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().iter().map(|&u| u as i32).collect())
    }

    /// Decode generated ids to text (skipping special tokens).
    pub fn decode(&self, ids: &[i32]) -> Result<String> {
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.inner.decode(&ids, true).map_err(|e| anyhow!("decode: {e}"))
    }
}

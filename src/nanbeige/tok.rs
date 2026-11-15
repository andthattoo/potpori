//! Tokenizer and chat template.
//!
//! The engine works on token ids; this is the text boundary. The template is the single-turn,
//! no-tools path of the one published in the repository's `tokenizer_config.json`, written
//! out here rather than interpreted.

use anyhow::{anyhow, Context, Result};

use crate::engine::TokenId;

/// The system prompt the published template inserts when the conversation has none.
const DEFAULT_SYSTEM: &str = "你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。";

pub struct ChatTokenizer {
    inner: tokenizers::Tokenizer,
}

impl ChatTokenizer {
    /// Download (Hugging Face cache) and load the repository's `tokenizer.json`.
    pub fn load(repo: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().context("hf-hub api")?;
        let path = api
            .model(repo.to_string())
            .get("tokenizer.json")
            .with_context(|| format!("download tokenizer.json for {repo}"))?;
        let inner =
            tokenizers::Tokenizer::from_file(&path).map_err(|e| anyhow!("load tokenizer: {e}"))?;
        Ok(Self { inner })
    }

    /// The prompt text for one user message. With `think` the model reasons inside a
    /// `<think>` block before answering; without, the template closes the block empty.
    pub fn chat_text(user: &str, think: bool) -> String {
        let open = if think {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        };
        format!(
            "<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n\
             <|im_start|>assistant\n{open}"
        )
    }

    pub fn encode_chat(&self, user: &str, think: bool) -> Result<Vec<TokenId>> {
        let enc = self
            .inner
            .encode(Self::chat_text(user, think), false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().iter().map(|&u| u as TokenId).collect())
    }

    /// Tokens that end a generation: `<|im_end|>` and `<|endoftext|>`.
    pub fn eos_tokens(&self) -> Vec<TokenId> {
        ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| self.inner.token_to_id(t))
            .map(|id| id as TokenId)
            .collect()
    }

    /// Decode generated ids to text (skipping special tokens).
    pub fn decode(&self, ids: &[TokenId]) -> Result<String> {
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.inner
            .decode(&ids, true)
            .map_err(|e| anyhow!("decode: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_matches_the_published_one() {
        assert_eq!(
            ChatTokenizer::chat_text("hi", false),
            format!(
                "<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n\
                 <|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        );
        assert!(ChatTokenizer::chat_text("hi", true).ends_with("assistant\n<think>\n"));
    }

    #[test]
    #[ignore = "downloads a tokenizer from Hugging Face"]
    fn prompt_ids_are_in_range_and_eos_is_known() {
        let tok = ChatTokenizer::load(crate::nanbeige::TARGET_REPO).unwrap();
        let ids = tok
            .encode_chat("Write a short poem about sharing work.", false)
            .unwrap();
        eprintln!("{} tokens: {ids:?}", ids.len());
        assert!(ids.iter().all(|&i| (0..166912).contains(&i)));
        assert!(tok.eos_tokens().contains(&crate::nanbeige::EOS_TOKEN));
    }
}

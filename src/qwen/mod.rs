//! Qwen3 on Luminal: the model family the chain runs today (0.6B draft; 1.7B and 4B targets).
//!
//! - [`model`] builds the graph for a contiguous layer range; [`runtime`] holds the CUDA and
//!   Metal adapters that compile and execute it.
//! - [`hf`] downloads and prepares weights; [`tok`] is the tokenizer and chat template.
//! - [`shard`] runs one layer range and plugs into the chain protocol ([`crate::node`]).
//! - [`target`] runs a whole model in one process: the draft model, and the single-process
//!   baseline in tests. [`draft`] wraps it as the head's incremental draft engine.

pub mod draft;
pub mod dspark;
pub mod hf;
pub mod model;
pub mod runtime;
pub mod shard;
pub mod target;
pub mod tok;

// GPU integration tests (`--ignored`): CUDA on a tiny generated model and on real weights,
// and the chain protocol over real Metal shards.
#[cfg(all(test, feature = "candle"))]
mod candle_check;
#[cfg(all(test, feature = "metal", target_vendor = "apple"))]
mod chain_tests;
#[cfg(all(test, feature = "reference"))]
mod cpu_check;
#[cfg(all(test, feature = "cuda"))]
mod cuda_check;
#[cfg(all(
    test,
    any(feature = "cuda", all(feature = "metal", target_vendor = "apple"))
))]
mod dspark_check;
#[cfg(all(test, feature = "metal", target_vendor = "apple"))]
mod metal_profile;
#[cfg(all(test, any(feature = "cuda", feature = "reference", feature = "candle")))]
mod tiny;
// Prototype candle backend; only the tests call it so far.
#[cfg(feature = "candle")]
#[allow(dead_code)]
pub mod candle_shard;

use crate::engine::TokenId;
use model::QwenConfig;

/// Tokens that end a Qwen3 generation: `<|im_end|>` and `<|endoftext|>`.
pub const EOS_TOKENS: [TokenId; 2] = [151645, 151643];

pub fn is_eos(token: TokenId) -> bool {
    EOS_TOKENS.contains(&token)
}

/// Map an HF repo name to a known Qwen3 config. Exact match only.
pub fn config_for(repo: &str) -> Option<QwenConfig> {
    match repo {
        "Qwen/Qwen3-0.6B" => Some(QwenConfig::QWEN3_0_6B),
        "Qwen/Qwen3-1.7B" => Some(QwenConfig::QWEN3_1_7B),
        "Qwen/Qwen3-4B" => Some(QwenConfig::QWEN3_4B),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_table_agrees_with_the_chain_config() {
        for repo in ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"] {
            let cfg = config_for(repo).unwrap();
            assert_eq!(Some(cfg.layers), crate::chain::model_layers(repo), "{repo}");
        }
        assert!(config_for("Qwen/Qwen3-14B").is_none());
    }
}

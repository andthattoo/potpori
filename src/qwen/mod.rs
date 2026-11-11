//! Qwen3 on Luminal — the bring-up model family for the sharded spec-decode pipeline (0.6B
//! draft / 1.7B / 4B targets; the GLM-5.2 port follows the same shape with its own config).
//! `model` is the layer-loop forward + `QwenShard` (the pipeline cut); `hf` downloads/prepares
//! weights; `target` runs a whole model in one process (the draft, and the single-process
//! baseline in tests); `shard` runs one layer block and plugs into the chain protocol in
//! [`crate::node`].

pub mod hf;
pub mod model;

pub mod target;

pub mod shard;

// Real-model chain tests (Metal, `--ignored`): the protocol in `crate::node` over actual shards.
#[cfg(all(test, feature = "metal", target_vendor = "apple"))]
mod chain_tests;

pub mod draft;
pub mod tok;

// CUDA substrate diagnostic — only under `--features cuda` (run on an NVIDIA box).
#[cfg(all(test, feature = "cuda"))]
mod cuda_check;

use model::QwenConfig;

use crate::engine::TokenId;

/// Tokens that end a Qwen3 generation: `<|im_end|>` and `<|endoftext|>`.
pub const EOS_TOKENS: [TokenId; 2] = [151645, 151643];

pub fn is_eos(token: TokenId) -> bool {
    EOS_TOKENS.contains(&token)
}

/// The HF repos this build has a model config for.
pub const KNOWN_MODELS: [&str; 3] = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"];

/// Map an HF repo name to a known Qwen3 config. Exact match only: a substring match would
/// hand e.g. `Qwen/Qwen3-14B` the 4B config and fail much later, inside the weight load.
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
    fn config_for_is_exact() {
        for repo in KNOWN_MODELS {
            assert!(config_for(repo).is_some(), "{repo}");
        }
        assert_eq!(config_for("Qwen/Qwen3-4B").unwrap().layers, 36);
        assert_eq!(config_for("Qwen/Qwen3-0.6B").unwrap().hidden, 1024);
        for repo in ["Qwen/Qwen3-14B", "Qwen/Qwen3-4B-Instruct", "qwen/qwen3-4b", "4B", ""] {
            assert!(config_for(repo).is_none(), "{repo:?} must not match");
        }
    }
}

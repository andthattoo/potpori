//! Qwen3 on Luminal — the bring-up model family for the sharded spec-decode pipeline (0.6B
//! draft / 1.7B / 4B targets; the GLM-5.2 port follows the same shape with its own config).
//! `model` is the layer-loop forward + `QwenShard` (the pipeline cut); `hf` downloads/prepares
//! weights; `target`/`shard`/`distributed` drive it locally, in-process-sharded, and over QUIC.
#![allow(dead_code)]

pub mod hf;
pub mod model;

pub mod target;

pub mod shard;

pub mod distributed;

pub mod draft;
pub mod tok;

// CUDA substrate diagnostic — only under `--features cuda` (run on an NVIDIA box).
#[cfg(all(test, feature = "cuda"))]
mod cuda_check;

use model::QwenConfig;

/// Map an HF repo name to a known Qwen3 config.
pub fn config_for(repo: &str) -> Option<QwenConfig> {
    if repo == "Qwen/Qwen3-0.6B" {
        Some(QwenConfig::QWEN3_0_6B)
    } else if repo == "Qwen/Qwen3-1.7B" {
        Some(QwenConfig::QWEN3_1_7B)
    } else if repo == "Qwen/Qwen3-4B" {
        Some(QwenConfig::QWEN3_4B)
    } else {
        None
    }
}

pub mod runtime;

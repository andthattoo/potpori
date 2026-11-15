//! Nanbeige4.3-33B, the one model this program runs, and its DFlash2 drafter.
//!
//! - [`config`] is the geometry of both; [`weights`] reads the published checkpoints;
//!   [`tok`] is the tokenizer and chat template.
//! - [`ops`] launches the CUDA kernels; [`shard`] runs a contiguous range of target layers and
//!   plugs into the chain protocol ([`crate::node`]); [`dflash`] is the drafter, which runs
//!   on the head and reads the target's hidden states.

pub mod config;
pub mod dflash;
pub mod ops;
pub mod shard;
pub mod stream;
pub mod tok;
pub mod weights;

#[cfg(test)]
mod reference;
#[cfg(test)]
mod tests;

use crate::engine::TokenId;

/// The target's Hugging Face repository: the original bf16 weights.
pub const TARGET_REPO: &str = "Nanbeige/Nanbeige4.3-33B";
/// The same model with its layer matrices rounded to 4 bits (NVFP4); embedding and output
/// head stay bf16.
pub const TARGET_REPO_FP4: &str = "ProCreations/Nanbeige4.3-33B-NVFP4-A16";
/// The drafter trained for it.
pub const DRAFTER_REPO: &str = "ProCreations/Nanbeige4.3-33B-DFlash2";
/// What two nodes compare in the handshake to know they serve the same model: the 4-bit
/// weights are a different model from the originals, and a chain must not mix them.
pub fn model_id(fp4: bool) -> &'static str {
    if fp4 {
        "nanbeige4.3-33b-nvfp4"
    } else {
        "nanbeige4.3-33b"
    }
}

/// `<|im_end|>`: the token that ends a turn.
pub const EOS_TOKEN: TokenId = 166101;

pub fn is_eos(token: TokenId) -> bool {
    token == EOS_TOKEN
}

/// Greedy pick over one row of logits. Panics on empty or non-finite logits: in a local model
/// that is a broken kernel, and continuing would only produce garbage tokens.
pub(crate) fn argmax(row: &[f32]) -> TokenId {
    crate::engine::argmax(row).expect("non-finite or empty logits")
}

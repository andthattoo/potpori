//! Model-agnostic traits the speculative-decode drivers are written against.
//!
//! The drivers ([`crate::specdecode`], [`crate::specpipe`]) only see these traits, so their
//! accept/reject logic is tested with synthetic engines — no Luminal, GPU or model needed.
//! The real implementations are a small Qwen model on the head
//! ([`crate::qwen::draft::IncrementalDraft`]) and the sharded target behind
//! [`crate::node::ChainLink`].

/// A vocabulary token id. Matches Luminal/HF conventions (i32).
pub type TokenId = i32;

/// Greedy pick over one row of logits; `None` if the row is empty or has a non-finite value
/// (a miscompiled kernel or an overflow — never a token to trust).
pub fn argmax(row: &[f32]) -> Option<TokenId> {
    if row.is_empty() || row.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let best = row
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i, v)
            } else {
                best
            }
        });
    Some(best.0 as TokenId)
}

/// The draft model. Runs entirely on the head, off the network, and proposes the next
/// `gamma` tokens greedily.
pub trait DraftEngine {
    /// KV positions available to the draft model.
    fn capacity(&self) -> usize {
        usize::MAX
    }
    /// Greedily draft `gamma` tokens continuing `context` (`prompt ++ generated ++ [pending]`).
    /// Must return exactly `gamma` tokens.
    fn draft(&mut self, context: &[TokenId], gamma: usize) -> Vec<TokenId>;
}

/// A target model driven one chunk at a time. This is the simple reference interface: the
/// synchronous driver, the draft model and the single-process baselines in tests use it. The
/// CLI drives the remote chain through [`crate::specpipe::PipelinedChain`] instead.
///
/// There is no rollback call: implementations write their KV at absolute positions, so
/// re-verifying from an earlier position overwrites a rejected speculative tail.
pub trait TargetChain {
    /// KV positions available.
    fn capacity(&self) -> usize {
        usize::MAX
    }
    /// Feed `prompt` at positions `[0, prompt.len())` and return the target's greedy token for
    /// the first generated position.
    fn prefill(&mut self, prompt: &[TokenId]) -> TokenId;

    /// Feed the chunk `[cur] + drafts`, whose first token sits at absolute position
    /// `start_pos`, and return the target's greedy token after each fed token (length
    /// `drafts.len() + 1`).
    fn verify(&mut self, start_pos: u32, cur: TokenId, drafts: &[TokenId]) -> Vec<TokenId>;

    /// Whether `token` terminates generation.
    fn is_eos(&self, token: TokenId) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_rejects_non_finite_rows() {
        assert_eq!(argmax(&[0.1, 2.0, -3.0]), Some(1));
        assert_eq!(argmax(&[-1.0, -2.0]), Some(0));
        assert_eq!(argmax(&[]), None);
        assert_eq!(argmax(&[0.0, f32::NAN]), None);
        assert_eq!(argmax(&[f32::INFINITY, 0.0]), None);
    }
}

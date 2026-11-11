//! Substrate-agnostic engine traits.
//!
//! The speculative-decode **drivers** ([`crate::specdecode`], [`crate::specpipe`]) are written
//! purely against these traits — so their accept/reject logic is unit-testable with synthetic
//! engines, with zero dependency on Luminal, a GPU, or a real model. The production
//! implementations are:
//!   - [`DraftEngine`] → a small model on the head ([`crate::qwen::draft::IncrementalDraft`]).
//!   - the sharded target → [`crate::node::ChainLink`] (the head's shard + the remote chain
//!     over QUIC), driven through [`crate::specpipe::PipelinedChain`].
//!
//! See `docs/pipeline-specdecode-design.md` and `docs/substrate-decision.md`.

/// A vocabulary token id. Matches llama/Luminal/HF conventions (i32).
pub type TokenId = i32;

/// Greedy pick over one row of logits. NaN never wins (an all-NaN row yields token 0, which
/// the head's draft sanity check treats as the "miscompiled kernel" signal).
pub fn argmax(row: &[f32]) -> TokenId {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as TokenId
}

/// The local draft model. Runs entirely on the head, off the WAN, and
/// proposes the next `gamma` tokens greedily given the committed context.
pub trait DraftEngine {
    /// Greedily draft `gamma` tokens continuing `context` (the committed token ids,
    /// i.e. `prompt ++ generated ++ [pending]`). Must return exactly `gamma` tokens.
    fn draft(&mut self, context: &[TokenId], gamma: usize) -> Vec<TokenId>;
}

/// A target model exposed to the synchronous reference driver as a black box (one chunk in
/// flight). The CLI path uses the pipelined [`crate::specpipe::PipelinedChain`] instead; this
/// stays as the simple correctness reference the pipelined driver is tested against.
#[allow(dead_code)] // reference driver: consumed by tests + single-process diagnostics
pub trait TargetChain {
    /// Prefill `prompt` into the chain's (per-shard) KV — leaving each shard's cache
    /// holding positions `[0, prompt.len())` — and return the target's greedy argmax for
    /// the first generated position (the `cur` token pending at position `prompt.len()`).
    fn prefill(&mut self, prompt: &[TokenId]) -> TokenId;

    /// Verify the candidate chunk `[cur] + drafts`, whose first token (`cur`) sits at
    /// absolute KV position `start_pos`. Advances each shard's KV by the full chunk
    /// (`drafts.len() + 1` positions) and returns the target's greedy argmax at **each**
    /// of those positions — i.e. `out[j]` is what the target would produce after the
    /// `j`-th fed token. The returned vector has length `drafts.len() + 1`.
    ///
    /// The driver compares `out[j]` against `drafts[j]` to find the accepted prefix, then
    /// calls [`TargetChain::truncate`] to discard the rejected speculative tail.
    fn verify(&mut self, start_pos: u32, cur: TokenId, drafts: &[TokenId]) -> Vec<TokenId>;

    /// Roll back every shard's KV to keep only positions `[0, keep_to_pos)` — the
    /// `KvTruncate` primitive. This is trivial on Luminal (the host owns the cache, so a
    /// truncate is dropping page-table slots), and is the exact operation llama.cpp's
    /// `seq_rm` *refused* on SWA/hybrid caches — the M1 finding that drove the substrate
    /// decision (see `docs/substrate-decision.md`).
    fn truncate(&mut self, keep_to_pos: u32);

    /// Whether `token` terminates generation (EOS/EOG).
    fn is_eos(&self, token: TokenId) -> bool;
}

//! The production draft engine: a small Qwen target kept incrementally in sync on the head.
//!
//! Unlike the test-only `QwenDraft` (which re-prefilled the whole context every call — O(ctx)
//! per round), this keeps its own KV cache warm: each `draft()` call feeds only the tokens
//! committed since the last call (single-token steps, so it works with both the generic and
//! the chunk-specialized kernel buckets), then greedily speculates γ tokens.
//!
//! Rollback safety is the same absolute-position-overwrite property as the target chain: the
//! speculative feeds (positions ≥ n) are never counted as synced, so the next catch-up
//! overwrites them; on a driver divergence-reset (context shrinks) we re-feed the last
//! committed token to refresh the prediction.

use crate::engine::{DraftEngine, TargetChain, TokenId};

use super::target::{QwenRuntime, QwenTarget};

pub struct IncrementalDraft<R: QwenRuntime + 'static> {
    model: QwenTarget<R>,
    /// How many leading tokens of the caller's `context` are already in our KV.
    synced: usize,
}

impl<R: QwenRuntime + 'static> IncrementalDraft<R> {
    pub fn new(model: QwenTarget<R>) -> Self {
        Self { model, synced: 0 }
    }
}

impl<R: QwenRuntime + 'static> DraftEngine for IncrementalDraft<R> {
    fn draft(&mut self, context: &[TokenId], gamma: usize) -> Vec<TokenId> {
        let n = context.len();
        debug_assert!(n > 0, "draft called with empty context");
        // Divergence reset: the driver rolled `context` back below what we've fed. Positions
        // are absolute, so stale KV beyond n is harmless — but we must re-feed the last
        // committed token to get a fresh prediction for position n.
        if self.synced >= n {
            self.synced = n - 1;
        }

        // Catch up on tokens committed since last call; the last feed's argmax is draft[0].
        let mut cur = if self.synced == 0 {
            let first = self.model.prefill(context);
            self.synced = n;
            first
        } else {
            let mut last = 0;
            for i in self.synced..n {
                last = self.model.verify(i as u32, context[i], &[])[0];
            }
            self.synced = n;
            last
        };

        // Greedy speculation: γ tokens, single-token steps at positions n, n+1, …
        // (not counted as synced — the next catch-up overwrites these KV slots).
        let mut out = Vec::with_capacity(gamma);
        out.push(cur);
        let mut pos = n as u32;
        for _ in 1..gamma {
            cur = self.model.verify(pos, cur, &[])[0];
            out.push(cur);
            pos += 1;
        }
        out
    }
}

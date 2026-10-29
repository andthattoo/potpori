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
    /// Catch-up feeds pieces of exactly this many tokens (+ single-token remainder), matching
    /// the model's compiled point buckets {1, chunk}. 0/1 = single-token feeds only.
    chunk: usize,
    /// The exact tokens we have fed at absolute positions `0..fed.len()` (committed AND
    /// speculative). Catch-up only re-feeds positions where the caller's context differs —
    /// so our own speculative feeds (which the pipelined driver echoes back in `context`)
    /// are not paid for twice.
    fed: Vec<TokenId>,
}

impl<R: QwenRuntime + 'static> IncrementalDraft<R> {
    pub fn new(model: QwenTarget<R>, chunk: usize) -> Self {
        Self {
            model,
            chunk,
            fed: Vec::new(),
        }
    }

    /// Feed `ctx[from..to]` at absolute positions `from..to` in point-bucket-sized pieces;
    /// return the model's prediction for position `to`.
    fn feed_range(&mut self, ctx: &[TokenId], from: usize, to: usize) -> TokenId {
        let mut pos = from;
        let mut last = 0;
        while pos < to {
            let take = if self.chunk > 1 && to - pos >= self.chunk {
                self.chunk
            } else {
                1
            };
            let piece = &ctx[pos..pos + take];
            last = *self
                .model
                .verify(pos as u32, piece[0], &piece[1..])
                .last()
                .unwrap();
            pos += take;
        }
        last
    }
}

impl<R: QwenRuntime + 'static> DraftEngine for IncrementalDraft<R> {
    fn capacity(&self) -> usize {
        self.model.capacity()
    }
    fn draft(&mut self, context: &[TokenId], gamma: usize) -> Vec<TokenId> {
        let n = context.len();
        debug_assert!(n > 0, "draft called with empty context");
        // First position where what we've fed disagrees with the caller's context (a driver
        // divergence-reset rewrites the tail; our own echoed speculation matches and is free).
        let agree = self
            .fed
            .iter()
            .zip(context)
            .take_while(|(a, b)| a == b)
            .count();
        // If everything up to n matches we still need one re-feed to refresh the prediction
        // for position n (absolute-position overwrite makes the re-feed harmless).
        let start = agree.min(n.saturating_sub(1));

        if std::env::var("POTPORI_DEBUG").is_ok() && n < 120 {
            eprintln!(
                "[dbg-draft] n={n} agree={agree} start={start} tail_ctx={:?}",
                &context[n.saturating_sub(3)..]
            );
        }
        let mut cur = self.feed_range(context, start, n);
        self.fed.truncate(start);
        self.fed.extend_from_slice(&context[start..n]);

        // Greedy speculation: γ tokens, single-token steps at positions n, n+1, … Each step
        // feeds the PREVIOUS draft token, so all but the last land in the KV → record them.
        let mut out = Vec::with_capacity(gamma);
        out.push(cur);
        for (pos, _) in (n as u32..).zip(1..gamma) {
            self.fed.push(cur); // `cur` is fed at `pos` by this verify call
            cur = self.model.verify(pos, cur, &[])[0];
            out.push(cur);
        }
        out
    }
}

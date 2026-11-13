//! The head's draft engine: a small Qwen model whose KV cache is kept in step with the
//! generation, so each call only pays for tokens it has not seen.
//!
//! Each `draft()` call first feeds whatever part of the context differs from what the model
//! was last fed (in pieces the model accepts), then greedily speculates γ tokens.
//! The speculative tokens stay in the KV: when the driver echoes them back as context they
//! cost nothing, and when the driver rejects them the next call overwrites them from the
//! first position that differs — the same absolute-position overwrite the target chain uses.

use crate::engine::{DraftEngine, TargetChain, TokenId};

/// `M` is a whole model in one process, on any backend.
pub struct IncrementalDraft<M: TargetChain> {
    model: M,
    /// Catch-up feeds pieces of exactly this many tokens (+ single-token remainder), matching
    /// a model with kernels for {1, chunk} tokens. 1 = single-token feeds only; 0 = the model
    /// takes any number of tokens, so a catch-up is one feed.
    chunk: usize,
    /// The exact tokens we have fed at absolute positions `0..fed.len()` (committed AND
    /// speculative). Catch-up only re-feeds positions where the caller's context differs —
    /// so our own speculative feeds (which the pipelined driver echoes back in `context`)
    /// are not paid for twice.
    fed: Vec<TokenId>,
}

impl<M: TargetChain> IncrementalDraft<M> {
    pub fn new(model: M, chunk: usize) -> Self {
        Self {
            model,
            chunk,
            fed: Vec::new(),
        }
    }

    /// Feed `ctx[from..to]` at absolute positions `from..to` in pieces the model accepts;
    /// return the model's prediction for position `to`.
    fn feed_range(&mut self, ctx: &[TokenId], from: usize, to: usize) -> TokenId {
        let mut pos = from;
        let mut last = 0;
        while pos < to {
            let take = if self.chunk == 0 {
                to - pos
            } else if self.chunk > 1 && to - pos >= self.chunk {
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

impl<M: TargetChain> DraftEngine for IncrementalDraft<M> {
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

        tracing::trace!(n, agree, start, "draft catch-up");
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

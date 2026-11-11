//! Depth-N **pipelined** speculative-decode driver — the one `potpori chat` runs.
//!
//! The synchronous driver ([`crate::specdecode::run_spec_decode`]) pays one full chain
//! traversal per committed chunk. This driver keeps up to `depth` verification chunks in
//! flight, chaining each on the previous chunk's *draft* output, so the chain's stages work on
//! different chunks at once. When a chunk diverges from the target, every chunk still in
//! flight was drafted on a wrong prefix: their results are drained and discarded, and
//! drafting restarts from the corrected prefix.
//!
//! Only tokens equal to the target's own greedy pick are committed, so the output equals the
//! synchronous driver's (and plain greedy decode of the same chain) for any draft.

use std::collections::VecDeque;

use anyhow::{bail, Result};

use crate::engine::{DraftEngine, TokenId};
use crate::specdecode::{SpecConfig, SpecStats};

/// A target chain that supports pipelining: several verification chunks in flight at once,
/// completing in submission (FIFO) order.
///
/// There is no rollback call. Every shard writes its KV at absolute positions and attends
/// only to positions below the chunk it is running, so after a divergence the driver simply
/// re-submits from the committed boundary and the stale speculative tail is overwritten.
///
/// The chain is other machines over a network, so every call can fail; the driver stops at
/// the first error.
pub trait PipelinedChain {
    /// KV positions available on the smallest shard.
    fn capacity(&self) -> usize {
        usize::MAX
    }
    /// Prefill `prompt`; return the first pending token (the chain's argmax after the prompt).
    fn prefill(&mut self, prompt: &[TokenId]) -> Result<TokenId>;
    /// Submit a verification chunk `[cur] + drafts` whose first token sits at absolute KV
    /// position `start_pos`. Does not wait for the result.
    fn submit(&mut self, start_pos: u32, cur: TokenId, drafts: &[TokenId]) -> Result<()>;
    /// Block for the oldest in-flight chunk's per-position argmaxes (length `drafts.len() + 1`).
    fn poll(&mut self) -> Result<Vec<TokenId>>;
    /// Whether `token` terminates generation.
    fn is_eos(&self, token: TokenId) -> bool;
}

/// Run pipelined greedy speculative decode with up to `depth` chunks in flight. Returns the
/// generated tokens (without the prompt) and telemetry.
pub fn run_spec_decode_pipelined<D: DraftEngine, C: PipelinedChain>(
    draft: &mut D,
    chain: &mut C,
    prompt: &[TokenId],
    cfg: &SpecConfig,
    depth: usize,
) -> Result<(Vec<TokenId>, SpecStats)> {
    assert!(cfg.gamma > 0, "gamma must be positive");
    if cfg.max_new == 0 {
        return Ok((Vec::new(), SpecStats::default()));
    }
    let capacity = chain.capacity().min(draft.capacity());
    assert!(
        !prompt.is_empty() && prompt.len() < capacity,
        "prompt must leave room in the KV cache"
    );
    let max_new = cfg.max_new.min(capacity - prompt.len());
    let gamma = cfg.gamma;
    let depth = depth.max(1);
    let prompt_len = prompt.len() as u32;

    let cur0 = chain.prefill(prompt)?;
    let mut out: Vec<TokenId> = Vec::new();
    let mut done = chain.is_eos(cur0);
    if !done {
        out.push(cur0); // out is seeded with the first pending token (shard: `out = [cur]`)
    }
    done |= out.len() >= max_new;

    // `dprefix` is the speculative draft chain; its last element is the next chunk's `cur`.
    let mut dprefix: Vec<TokenId> = prompt.to_vec();
    dprefix.push(cur0);
    // Consecutive chunks overlap by 1 token (their shared `cur`), so send_pos advances by
    // `gamma`, not `gamma + 1` (the re-fed `cur` lands on the previous chunk's last position).
    let mut send_pos = prompt_len;

    let mut inflight: VecDeque<Vec<TokenId>> = VecDeque::new(); // drafts per in-flight chunk
    let mut discard = 0usize; // stale post-divergence results still to drain
    let mut traversals = 0usize;

    loop {
        // FILL: keep `depth` chunks marching, chaining speculatively on `dprefix`.
        while inflight.len() < depth && !done && discard == 0 {
            let cur = *dprefix.last().unwrap();
            if chain.is_eos(cur) {
                // Only a verified EOS terminates output. Pause speculative filling.
                break;
            }
            let remaining = capacity.saturating_sub(send_pos as usize);
            if remaining == 0 {
                break;
            }
            // Point buckets support full verify chunks or one token. Near capacity,
            // drain speculative work and advance using target-only single-token steps.
            let single = remaining <= gamma;
            if single && !inflight.is_empty() {
                break;
            }
            let ds = if single {
                Vec::new()
            } else {
                draft.draft(&dprefix, gamma)
            };
            assert!(
                single || ds.len() == gamma,
                "draft returned the wrong token count"
            );
            chain.submit(send_pos, cur, &ds)?;
            dprefix.extend_from_slice(&ds);
            inflight.push_back(ds);
            send_pos += gamma as u32;
            if single {
                break;
            }
        }
        if inflight.is_empty() {
            break;
        }

        // READ one completed chunk (FIFO).
        let r = chain.poll()?;
        let ds = inflight.pop_front().unwrap();
        if r.len() != ds.len() + 1 {
            // `r[n]` below relies on this; a short answer from a remote must not panic the head.
            bail!(
                "chain returned {} tokens for a {}-token chunk",
                r.len(),
                ds.len() + 1
            );
        }
        traversals += 1;
        tracing::trace!(traversals, discard, ?ds, ?r, "chunk verified");
        if discard > 0 {
            discard -= 1; // stale (post-divergence or post-terminal) → skip
            continue;
        }

        // Accept the longest prefix matching the target's argmax.
        let mut n = 0usize;
        while n < ds.len() && n < r.len() && ds[n] == r[n] {
            n += 1;
        }
        let diverged = ds.is_empty() || n != ds.len();
        let committed: Vec<TokenId> = if diverged {
            let mut c = ds[..n].to_vec();
            c.push(r[n]); // accepted prefix + correction
            c
        } else {
            ds // all accepted
        };

        for t in committed {
            if chain.is_eos(t) {
                done = true;
                break;
            }
            out.push(t);
            if out.len() >= max_new {
                done = true;
                break;
            }
        }

        if done {
            // Terminal: stop filling; drain the chunks still in flight.
            discard = inflight.len();
        } else if diverged {
            // Divergence: every still-in-flight chunk was drafted on a now-wrong prefix.
            discard = inflight.len();
            // `out` includes the seed `cur0`, so the committed KV boundary is
            // prompt_len + out.len() - 1 (shard's `pos`). The next chunk re-feeds the
            // correction at this position, overwriting the stale tail.
            let keep = prompt_len + out.len() as u32 - 1;
            dprefix = prompt.to_vec();
            dprefix.extend_from_slice(&out);
            send_pos = keep;
        }
    }

    let committed = out.len();
    Ok((
        out,
        SpecStats {
            committed,
            traversals,
        },
    ))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::engine::TargetChain;

    #[test]
    fn unverified_eos_does_not_stop_output() {
        struct EarlyEos;
        impl DraftEngine for EarlyEos {
            fn draft(&mut self, _: &[TokenId], gamma: usize) -> Vec<TokenId> {
                let mut d = vec![-100; gamma];
                d[gamma - 1] = 9999;
                d
            }
        }
        let truth: Vec<_> = (1..=30).collect();
        for depth in [1, 2, 4, 8] {
            let mut chain = SyntheticPipe::new(truth.clone(), 9999);
            let (out, _) = run_spec_decode_pipelined(
                &mut EarlyEos,
                &mut chain,
                &[100],
                &SpecConfig {
                    gamma: 2,
                    max_new: 10,
                },
                depth,
            )
            .unwrap();
            assert_eq!(out, truth[..10], "depth={depth}");
        }
    }
    #[test]
    fn respects_zero_and_one_token_caps_and_verified_eos() {
        let truth = vec![1, 2, 3, 9999];
        for max_new in [0, 1, 2, 10] {
            for depth in [1, 4] {
                let mut chain = SyntheticPipe::new(truth.clone(), 9999);
                let mut draft = PerfectDraft {
                    truth: truth.clone(),
                    prompt_len: 1,
                };
                let (out, _) = run_spec_decode_pipelined(
                    &mut draft,
                    &mut chain,
                    &[100],
                    &SpecConfig { gamma: 2, max_new },
                    depth,
                )
                .unwrap();
                assert_eq!(out, truth[..max_new.min(3)]);
                assert!(chain.q.is_empty(), "all in-flight requests must drain");
            }
        }
    }
    #[test]
    fn lookahead_stays_within_capacity() {
        struct Bounded {
            inner: SyntheticPipe,
            limit: usize,
            single: usize,
        }
        impl PipelinedChain for Bounded {
            fn capacity(&self) -> usize {
                self.limit
            }
            fn prefill(&mut self, p: &[TokenId]) -> Result<TokenId> {
                self.inner.prefill(p)
            }
            fn submit(&mut self, pos: u32, cur: TokenId, ds: &[TokenId]) -> Result<()> {
                assert!(pos as usize + ds.len() < self.limit);
                assert!(
                    ds.is_empty() || ds.len() == 7,
                    "must use compiled point buckets"
                );
                if ds.is_empty() {
                    self.single += 1;
                }
                self.inner.submit(pos, cur, ds)
            }
            fn poll(&mut self) -> Result<Vec<TokenId>> {
                self.inner.poll()
            }
            fn is_eos(&self, t: TokenId) -> bool {
                self.inner.is_eos(t)
            }
        }
        let truth: Vec<_> = (1..100).collect();
        for depth in [1, 4, 8] {
            let mut c = Bounded {
                inner: SyntheticPipe::new(truth.clone(), 9999),
                limit: 768,
                single: 0,
            };
            let mut d = PerfectDraft {
                truth: truth.clone(),
                prompt_len: 740,
            };
            let (out, _) = run_spec_decode_pipelined(
                &mut d,
                &mut c,
                &vec![100; 740],
                &SpecConfig {
                    gamma: 7,
                    max_new: 28,
                },
                depth,
            )
            .unwrap();
            assert_eq!(out, truth[..28]);
            assert!(c.single > 0);
        }
    }

    /// Pipelined synthetic chain: its greedy continuation of any prompt is a fixed `truth`.
    /// FIFO queue of submitted chunks' `(start_pos, n_drafts)`; `poll` computes argmaxes
    /// purely from `start_pos` (stateless). The driver must reproduce `truth` exactly, at
    /// every `depth`, with any draft.
    struct SyntheticPipe {
        truth: Vec<TokenId>,
        prompt_len: u32,
        eos: TokenId,
        q: VecDeque<(u32, usize)>,
    }
    impl SyntheticPipe {
        fn new(truth: Vec<TokenId>, eos: TokenId) -> Self {
            Self {
                truth,
                prompt_len: 0,
                eos,
                q: VecDeque::new(),
            }
        }
    }
    impl PipelinedChain for SyntheticPipe {
        fn prefill(&mut self, prompt: &[TokenId]) -> Result<TokenId> {
            self.prompt_len = prompt.len() as u32;
            Ok(*self.truth.first().unwrap_or(&self.eos))
        }
        fn submit(&mut self, start_pos: u32, _cur: TokenId, drafts: &[TokenId]) -> Result<()> {
            self.q.push_back((start_pos, drafts.len()));
            Ok(())
        }
        fn poll(&mut self) -> Result<Vec<TokenId>> {
            let (start_pos, len) = self.q.pop_front().expect("poll with no in-flight chunk");
            let l = (start_pos - self.prompt_len) as usize; // gen index of `cur`
            Ok((0..=len)
                .map(|j| *self.truth.get(l + 1 + j).unwrap_or(&self.eos))
                .collect())
        }
        fn is_eos(&self, token: TokenId) -> bool {
            token == self.eos
        }
    }

    struct PerfectDraft {
        truth: Vec<TokenId>,
        prompt_len: usize,
    }
    impl DraftEngine for PerfectDraft {
        fn draft(&mut self, context: &[TokenId], gamma: usize) -> Vec<TokenId> {
            let l = context.len() - self.prompt_len;
            (0..gamma)
                .map(|j| *self.truth.get(l + j).unwrap_or(&-1))
                .collect()
        }
    }
    struct GarbageDraft;
    impl DraftEngine for GarbageDraft {
        fn draft(&mut self, _context: &[TokenId], gamma: usize) -> Vec<TokenId> {
            vec![i32::MIN; gamma]
        }
    }

    #[test]
    fn pipelined_is_lossless_across_depths_and_draft_quality() {
        let truth: Vec<TokenId> = (1..=80).collect();
        let eos = 9999;
        let prompt = vec![5, 6, 7];
        let cfg = SpecConfig {
            gamma: 4,
            max_new: 50,
        };
        let expected: Vec<TokenId> = truth[..cfg.max_new].to_vec();

        for depth in [1usize, 2, 4, 8] {
            // Perfect draft: pipeline stays full, no flushes.
            let mut c = SyntheticPipe::new(truth.clone(), eos);
            let mut d = PerfectDraft {
                truth: truth.clone(),
                prompt_len: prompt.len(),
            };
            let (out_p, _) =
                run_spec_decode_pipelined(&mut d, &mut c, &prompt, &cfg, depth).unwrap();
            assert_eq!(out_p, expected, "perfect draft, depth={depth}");

            // Garbage draft: every chunk diverges ⇒ exercises the flush/discard path hard.
            let mut c2 = SyntheticPipe::new(truth.clone(), eos);
            let mut g = GarbageDraft;
            let (out_g, _) =
                run_spec_decode_pipelined(&mut g, &mut c2, &prompt, &cfg, depth).unwrap();
            assert_eq!(out_g, expected, "garbage draft, depth={depth}");
        }
    }

    #[test]
    fn pipelined_matches_synchronous_driver() {
        // The pipelined and synchronous drivers must produce identical greedy output.
        let truth: Vec<TokenId> = (100..200).collect();
        let eos = -7;
        let prompt = vec![1, 2];
        let cfg = SpecConfig {
            gamma: 6,
            max_new: 60,
        };

        // synchronous reference (reuse the sync driver's synthetic target via a thin adapter)
        struct SyncTarget {
            truth: Vec<TokenId>,
            prompt_len: u32,
            eos: TokenId,
        }
        impl TargetChain for SyncTarget {
            fn prefill(&mut self, prompt: &[TokenId]) -> TokenId {
                self.prompt_len = prompt.len() as u32;
                *self.truth.first().unwrap_or(&self.eos)
            }
            fn verify(
                &mut self,
                start_pos: u32,
                _cur: TokenId,
                drafts: &[TokenId],
            ) -> Vec<TokenId> {
                let l = (start_pos - self.prompt_len) as usize;
                (0..=drafts.len())
                    .map(|j| *self.truth.get(l + 1 + j).unwrap_or(&self.eos))
                    .collect()
            }
            fn is_eos(&self, t: TokenId) -> bool {
                t == self.eos
            }
        }

        let mut st = SyncTarget {
            truth: truth.clone(),
            prompt_len: 0,
            eos,
        };
        let mut sd = PerfectDraft {
            truth: truth.clone(),
            prompt_len: prompt.len(),
        };
        let (sync_out, _) = crate::specdecode::run_spec_decode(&mut sd, &mut st, &prompt, &cfg);

        let mut pc = SyntheticPipe::new(truth.clone(), eos);
        let mut pd = PerfectDraft {
            truth: truth.clone(),
            prompt_len: prompt.len(),
        };
        let (pipe_out, _) = run_spec_decode_pipelined(&mut pd, &mut pc, &prompt, &cfg, 4).unwrap();

        assert_eq!(
            sync_out, pipe_out,
            "pipelined output must equal synchronous output"
        );
    }

    // ---- Stateful synthetic chain: catches position bookkeeping, not just accept logic ----

    /// Deterministic "model": the next token is a hash of the whole prefix, with EOS forced
    /// once `eos_at` tokens have been generated. Unlike `SyntheticPipe`, the answer depends on
    /// the tokens actually fed, so a driver that submits wrong tokens or positions is caught.
    #[derive(Clone, Copy)]
    pub(crate) struct Oracle {
        pub prompt_len: usize,
        pub eos_at: Option<usize>,
        pub eos: TokenId,
    }
    impl Oracle {
        pub fn next(&self, prefix: &[TokenId]) -> TokenId {
            let generated = prefix.len().saturating_sub(self.prompt_len);
            if self.eos_at.is_some_and(|e| generated >= e) {
                return self.eos;
            }
            let h = prefix.iter().fold(0x9E37_79B9u32, |h, &t| {
                h.rotate_left(5).wrapping_mul(0x85EB_CA6B) ^ t as u32
            });
            (h % 50) as TokenId + 1
        }
        /// Plain greedy decode — the ground truth every driver must reproduce.
        pub fn greedy(&self, prompt: &[TokenId], max_new: usize) -> Vec<TokenId> {
            let mut ctx = prompt.to_vec();
            let mut out = Vec::new();
            while out.len() < max_new {
                let t = self.next(&ctx);
                if t == self.eos {
                    break;
                }
                out.push(t);
                ctx.push(t);
            }
            out
        }
    }

    /// A chain with a position-indexed "KV" (the fed tokens by absolute position). Results are
    /// computed at submit time from the KV as it stands, like the real FIFO chain, so stale
    /// post-divergence chunks see stale state and the overwrite-rollback property is exercised.
    struct KvPipe {
        oracle: Oracle,
        kv: Vec<TokenId>,
        /// One past the highest absolute position ever fed.
        positions_used: usize,
        q: VecDeque<Vec<TokenId>>,
    }
    impl KvPipe {
        fn new(oracle: Oracle) -> Self {
            Self {
                oracle,
                kv: Vec::new(),
                positions_used: 0,
                q: VecDeque::new(),
            }
        }
        fn feed(&mut self, start_pos: u32, tokens: &[TokenId]) -> Vec<TokenId> {
            let start = start_pos as usize;
            assert!(start <= self.kv.len(), "chunk at {start} leaves a KV gap");
            self.kv.truncate(start);
            let mut out = Vec::with_capacity(tokens.len());
            for &t in tokens {
                self.kv.push(t);
                out.push(self.oracle.next(&self.kv));
            }
            self.positions_used = self.positions_used.max(self.kv.len());
            out
        }
        fn chunk(cur: TokenId, drafts: &[TokenId]) -> Vec<TokenId> {
            std::iter::once(cur).chain(drafts.iter().copied()).collect()
        }
    }
    impl PipelinedChain for KvPipe {
        fn prefill(&mut self, prompt: &[TokenId]) -> Result<TokenId> {
            self.q.clear();
            Ok(*self.feed(0, prompt).last().unwrap())
        }
        fn submit(&mut self, start_pos: u32, cur: TokenId, drafts: &[TokenId]) -> Result<()> {
            let r = self.feed(start_pos, &Self::chunk(cur, drafts));
            self.q.push_back(r);
            Ok(())
        }
        fn poll(&mut self) -> Result<Vec<TokenId>> {
            Ok(self.q.pop_front().expect("poll with no in-flight chunk"))
        }
        fn is_eos(&self, token: TokenId) -> bool {
            token == self.oracle.eos
        }
    }
    impl TargetChain for KvPipe {
        fn prefill(&mut self, prompt: &[TokenId]) -> TokenId {
            *self.feed(0, prompt).last().unwrap()
        }
        fn verify(&mut self, start_pos: u32, cur: TokenId, drafts: &[TokenId]) -> Vec<TokenId> {
            self.feed(start_pos, &Self::chunk(cur, drafts))
        }
        fn is_eos(&self, token: TokenId) -> bool {
            token == self.oracle.eos
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Quality {
        /// The target's own greedy continuation (keeps emitting EOS after the end).
        Perfect,
        /// Never right.
        Garbage,
        /// Right, except one wrong token on every other call.
        Flaky,
        /// Right, except the chunk's last token is always EOS.
        EosAtChunkEnd,
        /// Every token is EOS.
        AllEos,
    }
    pub(crate) const QUALITIES: [Quality; 5] = [
        Quality::Perfect,
        Quality::Garbage,
        Quality::Flaky,
        Quality::EosAtChunkEnd,
        Quality::AllEos,
    ];
    pub(crate) struct OracleDraft {
        pub oracle: Oracle,
        pub quality: Quality,
        pub calls: usize,
    }
    impl DraftEngine for OracleDraft {
        fn draft(&mut self, context: &[TokenId], gamma: usize) -> Vec<TokenId> {
            self.calls += 1;
            let mut ctx = context.to_vec();
            let mut out = Vec::with_capacity(gamma);
            for _ in 0..gamma {
                let t = self.oracle.next(&ctx);
                out.push(t);
                ctx.push(t);
            }
            match self.quality {
                Quality::Perfect => {}
                Quality::Garbage => out.fill(i32::MIN),
                Quality::Flaky if self.calls.is_multiple_of(2) => out[self.calls % gamma] = 77,
                Quality::Flaky => {}
                Quality::EosAtChunkEnd => *out.last_mut().unwrap() = self.oracle.eos,
                Quality::AllEos => out.fill(self.oracle.eos),
            }
            out
        }
    }

    /// The whole grid: gamma x depth x answer length x draft quality. Both drivers must equal
    /// plain greedy decode, whether generation ends on EOS or on `max_new`, and must leave
    /// nothing in flight.
    #[test]
    fn drivers_equal_greedy_across_the_grid() {
        let eos = 9999;
        let prompt = vec![11, 12, 13];
        let max_new = 17;
        for eos_at in (0..=22usize).map(Some).chain([None]) {
            let oracle = Oracle {
                prompt_len: prompt.len(),
                eos_at,
                eos,
            };
            let expected = oracle.greedy(&prompt, max_new);
            for gamma in 1..=5usize {
                let cfg = SpecConfig { gamma, max_new };
                for quality in QUALITIES {
                    let case = format!("eos_at={eos_at:?} gamma={gamma} {quality:?}");
                    let mut draft = OracleDraft {
                        oracle,
                        quality,
                        calls: 0,
                    };
                    let mut target = KvPipe::new(oracle);
                    let (out, _) =
                        crate::specdecode::run_spec_decode(&mut draft, &mut target, &prompt, &cfg);
                    assert_eq!(out, expected, "sync {case}");

                    for depth in 1..=6usize {
                        let mut draft = OracleDraft {
                            oracle,
                            quality,
                            calls: 0,
                        };
                        let mut chain = KvPipe::new(oracle);
                        let (out, stats) =
                            run_spec_decode_pipelined(&mut draft, &mut chain, &prompt, &cfg, depth)
                                .unwrap();
                        assert_eq!(out, expected, "pipelined depth={depth} {case}");
                        assert_eq!(stats.committed, out.len());
                        assert!(chain.q.is_empty(), "driver left chunks in flight: {case}");
                        assert!(
                            chain.positions_used <= prompt.len() + max_new + depth * gamma,
                            "speculation ran past the documented bound: {case}"
                        );
                    }
                }
            }
        }
    }
}

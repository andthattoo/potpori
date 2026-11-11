//! Async depth-N **pipelined** speculative-decode driver — shard's `coordinate_pipe` in Rust.
//!
//! The synchronous driver ([`crate::specdecode::run_spec_decode`]) pays one full chain
//! traversal per committed chunk (the ~3 tok/s regime over WAN). This driver keeps `depth`
//! verify chunks **in flight at once** over a direct-return ring, speculatively chaining
//! each chunk on the previous chunk's *draft* output — so throughput approaches the ring's
//! per-stage throughput, not its full latency (shard measured 2.94 → 16.6 tok/s). On a
//! divergence, every still-in-flight chunk is stale: they're drained-and-discarded and
//! drafting re-seeds from the corrected prefix. Greedy id-equality keeps the committed
//! output **bit-identical to the synchronous driver** (and to plain greedy decode).
//!
//! See `docs/shard-adoption-brief.md` (technique #1) and `docs/pipeline-specdecode-design.md`.

use std::collections::VecDeque;

use anyhow::{bail, Result};

use crate::engine::{DraftEngine, TokenId};
use crate::specdecode::{SpecConfig, SpecStats};

/// A target chain that supports depth-N pipelining: many verify chunks in flight at once,
/// completing in FIFO (submission) order.
///
/// There is no rollback op: every shard writes its KV at absolute positions and attends only
/// to positions below the chunk it is running, so after a divergence the driver simply
/// re-submits from the committed boundary and the stale speculative tail is overwritten.
///
/// Every call can fail (the chain is other people's machines over a network); the driver
/// stops at the first error.
pub trait PipelinedChain {
    /// Prefill `prompt`; return the first pending token (the chain's argmax after the prompt).
    fn prefill(&mut self, prompt: &[TokenId]) -> Result<TokenId>;
    /// Submit a verify chunk `[cur] + drafts` whose first token sits at absolute KV position
    /// `start_pos`. Non-blocking — the chunk enters the pipeline.
    fn submit(&mut self, start_pos: u32, cur: TokenId, drafts: &[TokenId]) -> Result<()>;
    /// Block for the next completed chunk's per-position argmaxes (FIFO; length `drafts.len()+1`).
    fn poll(&mut self) -> Result<Vec<TokenId>>;
    /// Whether `token` terminates generation.
    fn is_eos(&self, token: TokenId) -> bool;
}

/// Run depth-N pipelined greedy speculative decode. `depth` chunks are kept in flight.
/// Output is identical to [`crate::specdecode::run_spec_decode`] (and to plain greedy).
pub fn run_spec_decode_pipelined<D: DraftEngine, C: PipelinedChain>(
    draft: &mut D,
    chain: &mut C,
    prompt: &[TokenId],
    cfg: &SpecConfig,
    depth: usize,
) -> Result<(Vec<TokenId>, SpecStats)> {
    let gamma = cfg.gamma.max(1);
    let depth = depth.max(1);
    let prompt_len = prompt.len() as u32;

    let cur0 = chain.prefill(prompt)?;
    let mut out: Vec<TokenId> = Vec::new();
    let mut done = chain.is_eos(cur0) || cfg.max_new == 0;
    if !done {
        out.push(cur0); // out is seeded with the first pending token (shard: `out = [cur]`)
    }

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
        while inflight.len() < depth && !done {
            let cur = *dprefix.last().unwrap();
            if chain.is_eos(cur) {
                // The draft *speculates* EOS here. Stop extending the pipeline, but do NOT set
                // `done`: only a verified commit may end generation. Setting it here discarded
                // the still-in-flight (valid) chunks and truncated the output at depth >= 3.
                // If the EOS guess was wrong, the divergence path resets `dprefix` and filling
                // resumes.
                break;
            }
            let ds = draft.draft(&dprefix, gamma);
            chain.submit(send_pos, cur, &ds)?;
            dprefix.extend_from_slice(&ds);
            inflight.push_back(ds);
            send_pos += gamma as u32;
        }
        if inflight.is_empty() {
            break;
        }

        // READ one completed chunk (FIFO).
        let r = chain.poll()?;
        let ds = inflight.pop_front().unwrap();
        traversals += 1;
        if r.len() != ds.len() + 1 {
            // Indexing `r[n]` below trusts this; a short answer from a remote must not panic us.
            bail!("chain returned {} argmaxes for a {}-token chunk", r.len(), ds.len() + 1);
        }
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
        let diverged = n != ds.len();
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
            if out.len() >= cfg.max_new {
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
    Ok((out, SpecStats { committed, traversals }))
}

/// Drive a [`PipelinedChain`] through the synchronous [`TargetChain`] interface (one chunk in
/// flight) — lets the reference driver and plain greedy decode run over the real chain in tests.
#[cfg(test)]
pub struct Blocking<C: PipelinedChain>(pub C);

#[cfg(test)]
impl<C: PipelinedChain> crate::engine::TargetChain for Blocking<C> {
    fn prefill(&mut self, prompt: &[TokenId]) -> TokenId {
        self.0.prefill(prompt).expect("prefill")
    }
    fn verify(&mut self, start_pos: u32, cur: TokenId, drafts: &[TokenId]) -> Vec<TokenId> {
        self.0.submit(start_pos, cur, drafts).expect("submit");
        self.0.poll().expect("poll")
    }
    fn truncate(&mut self, _keep_to_pos: u32) {}
    fn is_eos(&self, token: TokenId) -> bool {
        self.0.is_eos(token)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::engine::TargetChain;
    use super::*;

    /// Pipelined synthetic chain: its greedy continuation of any prompt is a fixed `truth`.
    /// FIFO queue of submitted chunks' `(start_pos, n_drafts)`; `poll` computes argmaxes
    /// purely from `start_pos` (stateless ⇒ `truncate` is a no-op). The driver must
    /// reproduce `truth` exactly, at every `depth`, with any draft.
    struct SyntheticPipe {
        truth: Vec<TokenId>,
        prompt_len: u32,
        eos: TokenId,
        q: VecDeque<(u32, usize)>,
    }
    impl SyntheticPipe {
        fn new(truth: Vec<TokenId>, eos: TokenId) -> Self {
            Self { truth, prompt_len: 0, eos, q: VecDeque::new() }
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
            Ok((0..=len).map(|j| *self.truth.get(l + 1 + j).unwrap_or(&self.eos)).collect())
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
            (0..gamma).map(|j| *self.truth.get(l + j).unwrap_or(&-1)).collect()
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
        let cfg = SpecConfig { gamma: 4, max_new: 50 };
        let expected: Vec<TokenId> = truth[..cfg.max_new].to_vec();

        for depth in [1usize, 2, 4, 8] {
            // Perfect draft: pipeline stays full, no flushes.
            let mut c = SyntheticPipe::new(truth.clone(), eos);
            let mut d = PerfectDraft { truth: truth.clone(), prompt_len: prompt.len() };
            let (out_p, _) = run_spec_decode_pipelined(&mut d, &mut c, &prompt, &cfg, depth).unwrap();
            assert_eq!(out_p, expected, "perfect draft, depth={depth}");

            // Garbage draft: every chunk diverges ⇒ exercises the flush/discard path hard.
            let mut c2 = SyntheticPipe::new(truth.clone(), eos);
            let mut g = GarbageDraft;
            let (out_g, _) = run_spec_decode_pipelined(&mut g, &mut c2, &prompt, &cfg, depth).unwrap();
            assert_eq!(out_g, expected, "garbage draft, depth={depth}");
        }
    }

    #[test]
    fn pipelined_matches_synchronous_driver() {
        // The pipelined and synchronous drivers must produce identical greedy output.
        let truth: Vec<TokenId> = (100..200).collect();
        let eos = -7;
        let prompt = vec![1, 2];
        let cfg = SpecConfig { gamma: 6, max_new: 60 };

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
            fn verify(&mut self, start_pos: u32, _cur: TokenId, drafts: &[TokenId]) -> Vec<TokenId> {
                let l = (start_pos - self.prompt_len) as usize;
                (0..=drafts.len()).map(|j| *self.truth.get(l + 1 + j).unwrap_or(&self.eos)).collect()
            }
            fn truncate(&mut self, _: u32) {}
            fn is_eos(&self, t: TokenId) -> bool {
                t == self.eos
            }
        }

        let mut st = SyncTarget { truth: truth.clone(), prompt_len: 0, eos };
        let mut sd = PerfectDraft { truth: truth.clone(), prompt_len: prompt.len() };
        let (sync_out, _) = crate::specdecode::run_spec_decode(&mut sd, &mut st, &prompt, &cfg);

        let mut pc = SyntheticPipe::new(truth.clone(), eos);
        let mut pd = PerfectDraft { truth: truth.clone(), prompt_len: prompt.len() };
        let (pipe_out, _) = run_spec_decode_pipelined(&mut pd, &mut pc, &prompt, &cfg, 4).unwrap();

        assert_eq!(sync_out, pipe_out, "pipelined output must equal synchronous output");
    }

    // ---- Stateful synthetic chain: catches position-bookkeeping bugs, not just accept logic ----

    /// Deterministic "model": the next token is a hash of the whole prefix, with EOS forced once
    /// `eos_at` tokens have been generated. Unlike `SyntheticPipe`, the answer depends on the
    /// tokens actually fed, so a driver that submits the wrong tokens/positions gets caught.
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
            let h = prefix
                .iter()
                .fold(0x9E37_79B9u32, |h, &t| h.rotate_left(5).wrapping_mul(0x85EB_CA6B) ^ t as u32);
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
    /// computed at submit time from the KV as it stands, like the real FIFO chain — so stale
    /// post-divergence chunks see stale state and the overwrite-rollback property is exercised.
    struct KvPipe {
        oracle: Oracle,
        kv: Vec<TokenId>,
        /// High-water mark: one past the highest absolute position ever fed.
        positions_used: usize,
        q: VecDeque<Vec<TokenId>>,
    }
    impl KvPipe {
        fn new(oracle: Oracle) -> Self {
            Self { oracle, kv: Vec::new(), positions_used: 0, q: VecDeque::new() }
        }
        fn feed(&mut self, start_pos: u32, tokens: &[TokenId]) -> Vec<TokenId> {
            let start = start_pos as usize;
            assert!(start <= self.kv.len(), "chunk at {start} leaves a KV gap (len {})", self.kv.len());
            self.kv.truncate(start);
            let mut out = Vec::with_capacity(tokens.len());
            for &t in tokens {
                self.kv.push(t);
                out.push(self.oracle.next(&self.kv));
            }
            self.positions_used = self.positions_used.max(self.kv.len());
            out
        }
    }
    impl PipelinedChain for KvPipe {
        fn prefill(&mut self, prompt: &[TokenId]) -> Result<TokenId> {
            self.q.clear();
            Ok(*self.feed(0, prompt).last().unwrap())
        }
        fn submit(&mut self, start_pos: u32, cur: TokenId, drafts: &[TokenId]) -> Result<()> {
            let mut chunk = vec![cur];
            chunk.extend_from_slice(drafts);
            let r = self.feed(start_pos, &chunk);
            self.q.push_back(r);
            Ok(())
        }
        fn poll(&mut self) -> Result<Vec<TokenId>> {
            self.q.pop_front().ok_or_else(|| anyhow::anyhow!("poll with no in-flight chunk"))
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
            let mut chunk = vec![cur];
            chunk.extend_from_slice(drafts);
            self.feed(start_pos, &chunk)
        }
        fn truncate(&mut self, _keep_to_pos: u32) {}
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
        /// Right, except the chunk's LAST token is always EOS (the truncation-bug trigger,
        /// in its wrong-guess form).
        EosAtChunkEnd,
        /// Every token is EOS.
        AllEos,
    }
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
                Quality::Flaky => {
                    if self.calls % 2 == 0 {
                        out[self.calls % gamma] = 77; // outside the oracle's 1..=50 range
                    }
                }
                Quality::EosAtChunkEnd => *out.last_mut().unwrap() = self.oracle.eos,
                Quality::AllEos => out.fill(self.oracle.eos),
            }
            out
        }
    }

    pub(crate) const QUALITIES: [Quality; 5] =
        [Quality::Perfect, Quality::Garbage, Quality::Flaky, Quality::EosAtChunkEnd, Quality::AllEos];

    /// Regression: a draft that (correctly) ends a chunk on EOS must not cost the still-in-flight
    /// chunks. Before the fix, depth >= 3 dropped up to `depth - 1` chunks of valid output.
    #[test]
    fn eos_on_a_chunk_boundary_does_not_truncate() {
        let eos = 9999;
        let prompt = vec![7, 8];
        let gamma = 4;
        for eos_at in 0..=24usize {
            let oracle = Oracle { prompt_len: prompt.len(), eos_at: Some(eos_at), eos };
            let expected = oracle.greedy(&prompt, 100);
            assert_eq!(expected.len(), eos_at);
            for depth in 1..=8usize {
                let mut c = KvPipe::new(oracle);
                let mut d = OracleDraft { oracle, quality: Quality::Perfect, calls: 0 };
                let cfg = SpecConfig { gamma, max_new: 100 };
                let (out, _) = run_spec_decode_pipelined(&mut d, &mut c, &prompt, &cfg, depth).unwrap();
                assert_eq!(out, expected, "eos_at={eos_at} depth={depth}");
            }
        }
    }

    /// Regression: a draft that WRONGLY predicts EOS (target keeps going) must neither end
    /// generation nor stall it.
    #[test]
    fn wrong_eos_speculation_does_not_end_generation() {
        let eos = 9999;
        let prompt = vec![3];
        let oracle = Oracle { prompt_len: prompt.len(), eos_at: None, eos };
        let cfg = SpecConfig { gamma: 4, max_new: 30 };
        let expected = oracle.greedy(&prompt, cfg.max_new);
        assert_eq!(expected.len(), 30);
        for quality in [Quality::EosAtChunkEnd, Quality::AllEos] {
            for depth in 1..=6usize {
                let mut c = KvPipe::new(oracle);
                let mut d = OracleDraft { oracle, quality, calls: 0 };
                let (out, _) = run_spec_decode_pipelined(&mut d, &mut c, &prompt, &cfg, depth).unwrap();
                assert_eq!(out, expected, "{quality:?} depth={depth}");
            }
        }
    }

    /// The whole grid: gamma x depth x answer length x draft quality. Both drivers must equal
    /// plain greedy decode, whether generation ends on EOS or on `max_new`.
    #[test]
    fn drivers_equal_greedy_across_the_grid() {
        let eos = 9999;
        let prompt = vec![11, 12, 13];
        let max_new = 17;
        let mut cases = 0usize;
        for eos_at in (0..=22usize).map(Some).chain([None]) {
            let oracle = Oracle { prompt_len: prompt.len(), eos_at, eos };
            let expected = oracle.greedy(&prompt, max_new);
            for gamma in 1..=5usize {
                let cfg = SpecConfig { gamma, max_new };
                for quality in QUALITIES {
                    let mut t = KvPipe::new(oracle);
                    let mut d = OracleDraft { oracle, quality, calls: 0 };
                    let (sync_out, _) = crate::specdecode::run_spec_decode(&mut d, &mut t, &prompt, &cfg);
                    assert_eq!(sync_out, expected, "sync eos_at={eos_at:?} gamma={gamma} {quality:?}");

                    for depth in 1..=6usize {
                        let mut c = KvPipe::new(oracle);
                        let mut d = OracleDraft { oracle, quality, calls: 0 };
                        let (out, stats) = run_spec_decode_pipelined(&mut d, &mut c, &prompt, &cfg, depth).unwrap();
                        assert_eq!(
                            out, expected,
                            "pipelined eos_at={eos_at:?} gamma={gamma} depth={depth} {quality:?}"
                        );
                        assert_eq!(stats.committed, out.len());
                        assert!(c.q.is_empty(), "driver left chunks in flight");
                        // The bound `ChainConfig::positions_needed` relies on.
                        assert!(
                            c.positions_used <= prompt.len() + max_new + depth * gamma,
                            "fed {} positions > bound (gamma={gamma} depth={depth})",
                            c.positions_used
                        );
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 24 * 5 * 5 * 6);
    }
}

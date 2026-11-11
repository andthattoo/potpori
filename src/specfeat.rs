//! Greedy speculative decoding with a draft model that reads the target's hidden states.
//!
//! A feature-conditioned draft (DSpark) proposes a whole block of tokens in one pass, but it
//! needs the target's tapped-layer outputs for every position before the block. Those only
//! exist once the previous block has been through the whole chain, so this driver keeps one
//! block in flight:
//!
//!   1. the draft proposes a block after the pending token, from the features so far,
//!   2. the chain verifies `[cur] + block` in one trip and returns each position's features,
//!   3. the matching prefix is committed, and the features of the positions that are now
//!      settled — the pending token and the accepted drafts — go to the draft.
//!
//! As in [`crate::specdecode`], only the target's own picks are committed: the output equals
//! plain greedy decoding whatever the draft proposes.

use anyhow::{ensure, Result};

use crate::engine::TokenId;
use crate::specdecode::SpecStats;

/// A target that returns, with each verification, the features of the positions it ran.
/// Implemented by [`crate::node::ChainLink`] for a session with tapped layers.
pub trait FeatureChain {
    /// KV positions available.
    fn capacity(&self) -> usize;
    /// Feed `prompt` at positions `[0, prompt.len())`. Returns the target's greedy token for
    /// the first generated position, and one feature row per prompt position, flat.
    fn prefill_features(&mut self, prompt: &[TokenId]) -> Result<(TokenId, Vec<f32>)>;
    /// Feed `[cur] + drafts` from absolute position `start_pos`. Returns the target's greedy
    /// token after each fed token, and one feature row per fed position, flat.
    fn verify_features(
        &mut self,
        start_pos: u32,
        cur: TokenId,
        drafts: &[TokenId],
    ) -> Result<(Vec<TokenId>, Vec<f32>)>;
    /// Whether `token` terminates generation.
    fn is_eos(&self, token: TokenId) -> bool;
}

/// A draft model conditioned on target features.
pub trait FeatureDraft {
    /// Tokens proposed per call.
    fn block(&self) -> usize;
    /// Positions the draft can address.
    fn capacity(&self) -> usize {
        usize::MAX
    }
    /// Target features for settled positions `start_pos, start_pos + 1, …`, one row each.
    /// Anything previously observed from `start_pos` on is replaced.
    fn observe(&mut self, start_pos: usize, features: &[f32]) -> Result<()>;
    /// Propose `block()` tokens following `anchor`, the token at absolute position `start`.
    /// Features for every position before `start` have been observed.
    fn draft(&mut self, anchor: TokenId, start: usize) -> Result<Vec<TokenId>>;
}

/// Run greedy speculative decode, one block per trip. Returns the generated tokens (without
/// the prompt) and telemetry.
pub fn run_feature_decode<D: FeatureDraft, C: FeatureChain>(
    draft: &mut D,
    chain: &mut C,
    prompt: &[TokenId],
    max_new: usize,
) -> Result<(Vec<TokenId>, SpecStats)> {
    if max_new == 0 {
        return Ok((Vec::new(), SpecStats::default()));
    }
    let capacity = chain.capacity().min(draft.capacity());
    ensure!(
        !prompt.is_empty() && prompt.len() < capacity,
        "prompt must leave room in the KV cache"
    );
    let max_new = max_new.min(capacity - prompt.len());
    let block = draft.block();
    ensure!(block > 0, "the draft must propose at least one token");

    let (mut cur, features) = chain.prefill_features(prompt)?;
    ensure!(
        !features.is_empty() && features.len() % prompt.len() == 0,
        "the chain returned no features — is this a session with tapped layers?"
    );
    let width = features.len() / prompt.len();
    draft.observe(0, &features)?;
    let mut base = prompt.len(); // absolute position where `cur` is fed
    let mut out: Vec<TokenId> = Vec::new();
    let mut traversals = 0usize;

    'gen: while !chain.is_eos(cur) && out.len() < max_new {
        // Near the end of the cache only a single token still fits.
        let drafts = if capacity - base <= block {
            Vec::new()
        } else {
            let drafts = draft.draft(cur, base)?;
            ensure!(drafts.len() == block, "the draft returned a short block");
            drafts
        };
        let (r, features) = chain.verify_features(base as u32, cur, &drafts)?;
        ensure!(
            r.len() == drafts.len() + 1 && features.len() == r.len() * width,
            "the chain returned a result of the wrong size"
        );
        traversals += 1;

        let n = drafts.iter().zip(&r).take_while(|(d, t)| d == t).count();
        for t in std::iter::once(cur).chain(drafts[..n].iter().copied()) {
            if chain.is_eos(t) {
                break 'gen;
            }
            out.push(t);
            if out.len() >= max_new {
                break 'gen;
            }
        }
        // Positions base..=base+n now hold settled tokens; the rest of the chunk is a
        // rejected tail that the next chunk overwrites.
        draft.observe(base, &features[..(n + 1) * width])?;
        cur = r[n];
        base += n + 1;
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
    use crate::engine::DraftEngine;
    use crate::specpipe::tests::{Oracle, OracleDraft, Quality, QUALITIES};

    /// The feature of a position in these tests: the oracle's next token after it, twice.
    /// A real feature is the target's hidden state, which determines that token too.
    const WIDTH: usize = 2;

    /// A chain with a position-indexed "KV", like the real one: a chunk at an earlier
    /// position overwrites what followed.
    struct FeatureKv {
        oracle: Oracle,
        kv: Vec<TokenId>,
        capacity: usize,
        chunks: Vec<usize>,
    }
    impl FeatureKv {
        fn feed(&mut self, start: usize, tokens: &[TokenId]) -> (Vec<TokenId>, Vec<f32>) {
            assert!(start <= self.kv.len(), "chunk at {start} leaves a KV gap");
            assert!(start + tokens.len() <= self.capacity, "past the cache");
            self.kv.truncate(start);
            let mut next = Vec::new();
            for &t in tokens {
                self.kv.push(t);
                next.push(self.oracle.next(&self.kv));
            }
            let features = next.iter().flat_map(|&t| [t as f32; WIDTH]).collect();
            (next, features)
        }
    }
    impl FeatureChain for FeatureKv {
        fn capacity(&self) -> usize {
            self.capacity
        }
        fn prefill_features(&mut self, prompt: &[TokenId]) -> Result<(TokenId, Vec<f32>)> {
            let (next, features) = self.feed(0, prompt);
            Ok((*next.last().unwrap(), features))
        }
        fn verify_features(
            &mut self,
            start_pos: u32,
            cur: TokenId,
            drafts: &[TokenId],
        ) -> Result<(Vec<TokenId>, Vec<f32>)> {
            self.chunks.push(drafts.len() + 1);
            let chunk: Vec<_> = std::iter::once(cur).chain(drafts.iter().copied()).collect();
            Ok(self.feed(start_pos as usize, &chunk))
        }
        fn is_eos(&self, token: TokenId) -> bool {
            token == self.oracle.eos
        }
    }

    /// A draft that knows the prompt and otherwise only what the features tell it: it reads
    /// the generated tokens out of them (feature of position p = token at p + 1) and checks
    /// the driver's bookkeeping on every call. `decode` reads one feature row.
    pub(crate) struct FeatureOracleDraft<F: Fn(&[f32]) -> TokenId> {
        pub inner: OracleDraft,
        pub prompt: Vec<TokenId>,
        pub block: usize,
        pub width: usize,
        pub decode: F,
        /// The target's token after each settled position, by feature.
        pub next: Vec<TokenId>,
    }
    impl<F: Fn(&[f32]) -> TokenId> FeatureDraft for FeatureOracleDraft<F> {
        fn block(&self) -> usize {
            self.block
        }
        fn observe(&mut self, start_pos: usize, features: &[f32]) -> Result<()> {
            assert!(features.len().is_multiple_of(self.width));
            assert!(start_pos <= self.next.len(), "features leave a gap");
            self.next.truncate(start_pos);
            self.next
                .extend(features.chunks_exact(self.width).map(&self.decode));
            Ok(())
        }
        fn draft(&mut self, anchor: TokenId, start: usize) -> Result<Vec<TokenId>> {
            assert_eq!(
                self.next.len(),
                start,
                "features for every earlier position"
            );
            assert_eq!(
                self.next[start - 1],
                anchor,
                "the anchor follows the features"
            );
            let mut context = self.prompt.clone();
            context.extend_from_slice(&self.next[self.prompt.len() - 1..]);
            Ok(self.inner.draft(&context, self.block))
        }
    }

    pub(crate) fn feature_draft<F: Fn(&[f32]) -> TokenId>(
        oracle: Oracle,
        quality: Quality,
        prompt: &[TokenId],
        block: usize,
        width: usize,
        decode: F,
    ) -> FeatureOracleDraft<F> {
        FeatureOracleDraft {
            inner: OracleDraft {
                oracle,
                quality,
                calls: 0,
            },
            prompt: prompt.to_vec(),
            block,
            width,
            decode,
            next: Vec::new(),
        }
    }

    #[test]
    fn equals_greedy_for_every_draft_quality() {
        let prompt: Vec<TokenId> = vec![11, 12, 13, 14, 15];
        for eos_at in [None, Some(0), Some(1), Some(9), Some(30)] {
            let oracle = Oracle {
                prompt_len: prompt.len(),
                eos_at,
                eos: 60,
            };
            for quality in QUALITIES {
                for block in [1usize, 3, 7] {
                    for max_new in [0usize, 1, 2, 25] {
                        let mut chain = FeatureKv {
                            oracle,
                            kv: Vec::new(),
                            capacity: 256,
                            chunks: Vec::new(),
                        };
                        let mut draft =
                            feature_draft(oracle, quality, &prompt, block, WIDTH, |row| {
                                row[0] as TokenId
                            });
                        let (out, stats) =
                            run_feature_decode(&mut draft, &mut chain, &prompt, max_new).unwrap();
                        let case = format!("{eos_at:?} {quality:?} block={block} max={max_new}");
                        assert_eq!(out, oracle.greedy(&prompt, max_new), "{case}");
                        assert_eq!(stats.traversals, chain.chunks.len(), "{case}");
                        assert!(chain.chunks.iter().all(|&c| c == block + 1), "{case}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_perfect_draft_commits_a_whole_chunk_per_trip() {
        let prompt: Vec<TokenId> = vec![3, 4, 5];
        let oracle = Oracle {
            prompt_len: prompt.len(),
            eos_at: None,
            eos: 60,
        };
        let mut chain = FeatureKv {
            oracle,
            kv: Vec::new(),
            capacity: 256,
            chunks: Vec::new(),
        };
        let mut draft = feature_draft(oracle, Quality::Perfect, &prompt, 7, WIDTH, |row| {
            row[0] as TokenId
        });
        let (out, stats) = run_feature_decode(&mut draft, &mut chain, &prompt, 64).unwrap();
        assert_eq!(out.len(), 64);
        assert_eq!(stats.traversals, 8, "eight tokens per trip");
    }

    #[test]
    fn the_end_of_the_cache_is_reached_with_single_tokens() {
        let prompt: Vec<TokenId> = vec![7; 20];
        let oracle = Oracle {
            prompt_len: prompt.len(),
            eos_at: None,
            eos: 60,
        };
        let mut chain = FeatureKv {
            oracle,
            kv: Vec::new(),
            capacity: 40,
            chunks: Vec::new(),
        };
        let mut draft = feature_draft(oracle, Quality::Perfect, &prompt, 7, WIDTH, |row| {
            row[0] as TokenId
        });
        let (out, _) = run_feature_decode(&mut draft, &mut chain, &prompt, 1000).unwrap();
        assert_eq!(out, oracle.greedy(&prompt, 20), "fills the cache exactly");
        assert!(chain.chunks.iter().all(|&c| c == 8 || c == 1));
        assert!(chain.chunks.contains(&1));
    }

    #[test]
    fn a_chain_without_features_is_refused() {
        struct Plain;
        impl FeatureChain for Plain {
            fn capacity(&self) -> usize {
                64
            }
            fn prefill_features(&mut self, _: &[TokenId]) -> Result<(TokenId, Vec<f32>)> {
                Ok((1, Vec::new()))
            }
            fn verify_features(
                &mut self,
                _: u32,
                _: TokenId,
                _: &[TokenId],
            ) -> Result<(Vec<TokenId>, Vec<f32>)> {
                unreachable!()
            }
            fn is_eos(&self, _: TokenId) -> bool {
                false
            }
        }
        let oracle = Oracle {
            prompt_len: 1,
            eos_at: None,
            eos: 60,
        };
        let mut draft = feature_draft(oracle, Quality::Perfect, &[1], 7, WIDTH, |row| {
            row[0] as TokenId
        });
        let err = run_feature_decode(&mut draft, &mut Plain, &[1], 4).unwrap_err();
        assert!(err.to_string().contains("tapped layers"), "{err}");
    }
}

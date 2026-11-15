//! The whole model on one GPU that cannot hold it: `potpori local`.
//!
//! [`Streamed`] keeps every layer's KV cache on the device and loads the weights a few layers
//! at a time, for every chunk it runs. That reads the whole checkpoint from disk per chunk, so
//! it is a way to check the real weights end to end on a single card, not a way to chat.
//! [`LocalChain`] turns any shard into the target the decoding driver expects, with no network.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;

use crate::engine::TokenId;
use crate::gpu::Buf;
use crate::node::{tap_ranges, ShardEngine};
use crate::specfeat::FeatureChain;
use crate::transport::SessionHello;

use super::config::Config;
use super::shard::{Kept, Shard};
use super::weights::Checkpoint;
use super::{argmax, model_id};

pub struct Streamed {
    cfg: Config,
    ck: Arc<Checkpoint>,
    max_seq: usize,
    /// Layer groups in order; a tapped layer is always the last of its group.
    groups: Vec<(usize, usize)>,
    taps: Vec<usize>,
    embed: Arc<Buf<u16>>,
    lm_head: Arc<Buf<u16>>,
    /// Per group, one `(keys, values)` per layer; empty until the group first runs.
    caches: Vec<Vec<(Buf<f32>, Buf<f32>)>>,
    last_taps: Vec<Vec<f32>>,
}

impl Streamed {
    /// `group` is the most layers held on the device at once.
    pub fn new(
        cfg: Config,
        ck: Arc<Checkpoint>,
        max_seq: usize,
        mut taps: Vec<usize>,
        group: usize,
        embed: Arc<Buf<u16>>,
        lm_head: Arc<Buf<u16>>,
    ) -> Self {
        assert!(group > 0, "a group holds at least one layer");
        taps.sort_unstable();
        taps.dedup();
        let mut groups = Vec::new();
        for (start, end) in tap_ranges((0, cfg.layers), &taps) {
            let mut a = start;
            while a < end {
                let b = (a + group).min(end);
                groups.push((a, b));
                a = b;
            }
        }
        Self {
            caches: groups.iter().map(|_| Vec::new()).collect(),
            cfg,
            ck,
            max_seq,
            groups,
            taps,
            embed,
            lm_head,
            last_taps: Vec::new(),
        }
    }

    fn run(&mut self, pos: u32, tokens: Option<&[TokenId]>, s: usize, mut x: Vec<f32>) -> Vec<f32> {
        self.last_taps.clear();
        for (i, &(start, end)) in self.groups.clone().iter().enumerate() {
            let kept = Kept {
                embed: (start == 0).then(|| self.embed.clone()),
                lm_head: (end == self.cfg.layers).then(|| self.lm_head.clone()),
                caches: std::mem::take(&mut self.caches[i]),
            };
            let mut shard = Shard::load_with(self.cfg, (start, end), &self.ck, self.max_seq, kept)
                .unwrap_or_else(|e| panic!("load layers [{start}, {end}): {e}"));
            x = match tokens {
                Some(tokens) if i == 0 => shard.forward_tokens(pos, tokens),
                _ => shard.forward_hidden(pos, s, x),
            };
            if self.taps.contains(&(end - 1)) {
                self.last_taps.push(x.clone());
            }
            self.caches[i] = shard.into_caches();
        }
        x
    }
}

impl ShardEngine for Streamed {
    fn hello(&self, next_layer: usize) -> SessionHello {
        SessionHello {
            model: model_id(self.ck.is_fp4("model.layers.0.self_attn.q_proj.weight")).into(),
            next_layer,
            max_seq: self.max_seq,
            chunk: 0,
            taps: self.taps.clone(),
        }
    }
    fn range(&self) -> (usize, usize) {
        (0, self.cfg.layers)
    }
    fn model_layers(&self) -> usize {
        self.cfg.layers
    }
    fn hidden_dim(&self) -> usize {
        self.cfg.hidden
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab
    }
    fn capacity(&self) -> usize {
        self.max_seq
    }
    fn chunk_size(&self) -> usize {
        0
    }
    fn forward_tokens(&mut self, pos: u32, tokens: &[TokenId]) -> Vec<f32> {
        self.run(pos, Some(tokens), tokens.len(), Vec::new())
    }
    fn forward_hidden(&mut self, pos: u32, s: usize, hidden: Vec<f32>) -> Vec<f32> {
        self.run(pos, None, s, hidden)
    }
    fn reset(&mut self) {}
    fn tap_layers(&self) -> Vec<usize> {
        self.taps.clone()
    }
    fn take_taps(&mut self) -> Vec<Vec<f32>> {
        std::mem::take(&mut self.last_taps)
    }
}

/// A whole-model shard as the driver's target: every trip is a local call.
pub struct LocalChain<S: ShardEngine> {
    shard: S,
    eos: Vec<TokenId>,
    /// Print one line per trip (the streamed model takes a while for each).
    pub verbose: bool,
    trips: usize,
}

impl<S: ShardEngine> LocalChain<S> {
    pub fn new(shard: S, eos: &[TokenId]) -> Self {
        assert_eq!(
            shard.range(),
            (0, shard.model_layers()),
            "a local chain needs the whole model"
        );
        Self {
            shard,
            eos: eos.to_vec(),
            verbose: false,
            trips: 0,
        }
    }

    fn run(&mut self, pos: u32, tokens: &[TokenId], drafted: bool) -> (Vec<TokenId>, Vec<f32>) {
        let t0 = Instant::now();
        let logits = self.shard.forward_tokens(pos, tokens);
        let taps = self.shard.take_taps();
        let (hidden, vocab) = (self.shard.hidden_dim(), self.shard.vocab_size());
        // One row per position: the tapped layers side by side, ascending.
        let mut features = Vec::with_capacity(tokens.len() * taps.len() * hidden);
        for row in 0..tokens.len() {
            for tap in &taps {
                features.extend_from_slice(&tap[row * hidden..(row + 1) * hidden]);
            }
        }
        let picks: Vec<TokenId> = logits.chunks(vocab).map(argmax).collect();
        self.trips += 1;
        if self.verbose && !drafted {
            eprintln!(
                "[prefill: {} tokens in {:.1}s]",
                tokens.len(),
                t0.elapsed().as_secs_f64()
            );
        } else if self.verbose {
            let agree = tokens[1..]
                .iter()
                .zip(&picks)
                .take_while(|(d, t)| d == t)
                .count();
            eprintln!(
                "[trip {}: {} tokens at position {pos} in {:.1}s — {agree} of {} drafts accepted]",
                self.trips,
                tokens.len(),
                t0.elapsed().as_secs_f64(),
                tokens.len() - 1,
            );
        }
        (picks, features)
    }
}

impl<S: ShardEngine> FeatureChain for LocalChain<S> {
    fn capacity(&self) -> usize {
        self.shard.capacity()
    }
    fn prefill_features(&mut self, prompt: &[TokenId]) -> Result<(TokenId, Vec<f32>)> {
        let (picks, features) = self.run(0, prompt, false);
        Ok((*picks.last().expect("non-empty prompt"), features))
    }
    fn verify_features(
        &mut self,
        start_pos: u32,
        cur: TokenId,
        drafts: &[TokenId],
    ) -> Result<(Vec<TokenId>, Vec<f32>)> {
        let chunk: Vec<TokenId> = std::iter::once(cur).chain(drafts.iter().copied()).collect();
        Ok(self.run(start_pos, &chunk, true))
    }
    fn is_eos(&self, token: TokenId) -> bool {
        self.eos.contains(&token)
    }
}

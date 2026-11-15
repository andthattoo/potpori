//! The DFlash2 drafter: one pass proposes a whole block of tokens.
//!
//! It never sees tokens before the block. Its context is the target's own hidden states: for
//! every settled position, the outputs of a few target layers are projected to one vector and
//! turned directly into this model's keys and values ([`FeatureDraft::observe`]). A block is
//! the last verified token followed by mask tokens; a few bidirectional layers run over it
//! once, the target's output head turns each masked slot into candidate tokens, and a small
//! selector picks one path through the candidates ([`FeatureDraft::draft`]).
//!
//! The layer stack and both output projections run on the GPU. The selector is a few thousand
//! multiply-adds and runs on the host.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Result};
use half::bf16;

use crate::engine::TokenId;
use crate::gpu::Buf;
use crate::specfeat::FeatureDraft;

use super::config::{rope_table, DraftConfig, HEAD_DIM};
use super::ops::{self, Span};
use super::shard::PIECE;
use super::weights::Checkpoint;

struct Layer {
    ln1: Buf<f32>,
    wq: Buf<u16>,
    wk: Buf<u16>,
    wv: Buf<u16>,
    wo: Buf<u16>,
    q_norm: Buf<f32>,
    k_norm: Buf<f32>,
    ln2: Buf<f32>,
    /// Gate and up matrices stacked: `[2 * intermediate, hidden]`.
    wgu: Buf<u16>,
    wd: Buf<u16>,
    /// `[2 sides, 2 taps, hidden]` and the projection that predicts per-token adjustments.
    attn_base: Buf<f32>,
    attn_proj: Buf<u16>,
    mlp_base: Buf<f32>,
    mlp_proj: Buf<u16>,
    keys: Buf<f32>,
    values: Buf<f32>,
}

/// Where a draft's time went.
#[derive(Clone, Copy, Debug, Default)]
pub struct DraftStats {
    pub drafts: usize,
    /// Layers and output projections, on the GPU.
    pub gpu: Duration,
    /// Candidate ranking and the selector walk, on the host.
    pub host: Duration,
}

pub struct Dflash {
    cfg: DraftConfig,
    max_seq: usize,
    /// The target's embedding table and output head.
    embed: Arc<Buf<u16>>,
    lm_head: Arc<Buf<u16>>,
    fc: Buf<u16>,
    hidden_norm: Buf<f32>,
    norm: Buf<f32>,
    layers: Vec<Layer>,
    rope: Buf<f32>,
    sel_proj: Buf<u16>,
    selector: Selector,
    // Work buffers.
    feat: Buf<f32>,
    ctx: Buf<f32>,
    ids: Buf<i32>,
    h: Buf<f32>,
    xn: Buf<f32>,
    xc: Buf<f32>,
    coef: Buf<f32>,
    q: Buf<f32>,
    a: Buf<f32>,
    y: Buf<f32>,
    gu: Buf<f32>,
    m: Buf<f32>,
    scratch: Buf<f32>,
    logits: Buf<f32>,
    hp: Buf<f32>,
    /// The `selector_top_k` most likely tokens of each masked slot, and their logits.
    cand_ids: Buf<i32>,
    cand_vals: Buf<f32>,
    stats: DraftStats,
}

impl Dflash {
    /// Load the drafter from `ck`. `embed` and `lm_head` are the target's `[vocab, hidden]`
    /// embedding table and output head, already on this GPU.
    pub fn load(
        cfg: DraftConfig,
        ck: &Checkpoint,
        embed: Arc<Buf<u16>>,
        lm_head: Arc<Buf<u16>>,
        max_seq: usize,
    ) -> Result<Self> {
        super::weights::check(ck, &cfg.parameters())?;
        ensure!(
            cfg.block >= 2,
            "a block is an anchor plus at least one draft"
        );
        ensure!(
            cfg.hidden.is_multiple_of(cfg.conv_group),
            "conv_group must divide hidden"
        );
        ensure!(
            embed.len() == cfg.vocab * cfg.hidden && lm_head.len() == cfg.vocab * cfg.hidden,
            "the target's embedding and head do not match the drafter's width"
        );
        ensure!(
            (0..cfg.vocab as i32).contains(&cfg.mask_token),
            "mask token outside the vocabulary"
        );
        let (hidden, inter) = (cfg.hidden, cfg.intermediate);
        let (q_dim, kv_dim) = (cfg.q_dim(), cfg.kv_dim());
        let coef = 4 * cfg.conv_groups();
        let width = cfg.taps.len() * hidden;
        let mat = |name: &str, shape: [usize; 2]| -> Result<Buf<u16>> {
            Ok(Buf::from_slice(&ck.bf16(name, &shape)?))
        };
        let vec = |name: &str, shape: &[usize]| -> Result<Buf<f32>> {
            Ok(Buf::from_slice(&ck.f32(name, shape)?))
        };
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = |name: &str| format!("layers.{l}.{name}");
            layers.push(Layer {
                ln1: vec(&p("input_layernorm.weight"), &[hidden])?,
                wq: mat(&p("self_attn.q_proj.weight"), [q_dim, hidden])?,
                wk: mat(&p("self_attn.k_proj.weight"), [kv_dim, hidden])?,
                wv: mat(&p("self_attn.v_proj.weight"), [kv_dim, hidden])?,
                wo: mat(&p("self_attn.o_proj.weight"), [hidden, q_dim])?,
                q_norm: vec(&p("self_attn.q_norm.weight"), &[HEAD_DIM])?,
                k_norm: vec(&p("self_attn.k_norm.weight"), &[HEAD_DIM])?,
                ln2: vec(&p("post_attention_layernorm.weight"), &[hidden])?,
                wgu: Buf::from_slice(&ck.bf16_stacked(
                    &p("mlp.gate_proj.weight"),
                    &p("mlp.up_proj.weight"),
                    &[inter, hidden],
                )?),
                wd: mat(&p("mlp.down_proj.weight"), [hidden, inter])?,
                attn_base: vec(&p("attention_conv.base_kernel"), &[2, 2, hidden])?,
                attn_proj: mat(
                    &p("attention_conv.kernel_projection.weight"),
                    [coef, hidden],
                )?,
                mlp_base: vec(&p("mlp_conv.base_kernel"), &[2, 2, hidden])?,
                mlp_proj: mat(&p("mlp_conv.kernel_projection.weight"), [coef, hidden])?,
                keys: Buf::zeros(max_seq * kv_dim),
                values: Buf::zeros(max_seq * kv_dim),
            });
        }
        let rank = cfg.selector_rank;
        let selector = Selector {
            rank,
            top_k: cfg.selector_top_k,
            vocab: cfg.vocab,
            predecessor: ck.bf16(
                "candidate_selector.predecessor_codebook",
                &[cfg.vocab, rank],
            )?,
            successor: ck.bf16("candidate_selector.successor_codebook", &[cfg.vocab, rank])?,
        };
        ensure!(
            selector.top_k >= 1 && selector.top_k <= cfg.vocab,
            "selector_top_k out of range"
        );
        let block = cfg.block;
        Ok(Self {
            max_seq,
            embed,
            lm_head,
            fc: mat("fc.weight", [hidden, width])?,
            hidden_norm: vec("hidden_norm.weight", &[hidden])?,
            norm: vec("norm.weight", &[hidden])?,
            layers,
            rope: Buf::from_slice(&rope_table(cfg.rope_theta, max_seq)),
            sel_proj: mat(
                "candidate_selector.hidden_projection.weight",
                [rank, hidden],
            )?,
            selector,
            feat: Buf::zeros(PIECE * width),
            ctx: Buf::zeros(PIECE * hidden),
            ids: Buf::zeros(block),
            h: Buf::zeros(block * hidden),
            xn: Buf::zeros(block * hidden),
            xc: Buf::zeros(block * hidden),
            coef: Buf::zeros(block * coef),
            q: Buf::zeros(block * q_dim),
            a: Buf::zeros(block * q_dim),
            y: Buf::zeros(block * hidden),
            gu: Buf::zeros(block * 2 * inter),
            m: Buf::zeros(block * inter),
            scratch: Buf::zeros(block * cfg.n_heads * max_seq),
            logits: Buf::zeros((block - 1) * cfg.vocab),
            hp: Buf::zeros((block - 1) * rank),
            cand_ids: Buf::zeros((block - 1) * cfg.selector_top_k),
            cand_vals: Buf::zeros((block - 1) * cfg.selector_top_k),
            stats: DraftStats::default(),
            cfg,
        })
    }

    pub fn stats(&self) -> DraftStats {
        self.stats
    }

    /// Run the layers over one block starting at absolute position `start`; the final hidden
    /// states are left in `self.xn`, `[block, hidden]`.
    fn run_block(&mut self, anchor: TokenId, start: usize) {
        let c = &self.cfg;
        let (hidden, inter, block) = (c.hidden, c.intermediate, c.block);
        let (q_dim, kv_dim, group) = (c.q_dim(), c.kv_dim(), c.conv_group);
        let coef_n = 4 * c.conv_groups();
        let eps = c.rms_norm_eps;
        let mut ids = vec![c.mask_token; block];
        ids[0] = anchor;
        self.ids.write(0, &ids);
        let (h, xn, xc, coef) = (self.h.ptr(), self.xn.ptr(), self.xc.ptr(), self.coef.ptr());
        let (q, a, y, m) = (self.q.ptr(), self.a.ptr(), self.y.ptr(), self.m.ptr());
        let gu = self.gu.ptr();
        ops::embed(h, &self.embed, &self.ids, hidden, block);
        for l in &self.layers {
            let (k, v) = (l.keys.at(start * kv_dim), l.values.at(start * kv_dim));
            // Attention, with a convolution on its input and one on its output.
            ops::rmsnorm(xn, h, &l.ln1, eps, hidden, block);
            ops::gemm(coef, xn, &l.attn_proj, coef_n, hidden, block, coef_n, false);
            ops::conv2(xc, xn, coef, &l.attn_base, hidden, group, 0, false, block);
            ops::gemm(q, xc, &l.wq, q_dim, hidden, block, q_dim, false);
            ops::gemm(k, xc, &l.wk, kv_dim, hidden, block, kv_dim, false);
            ops::gemm(v, xc, &l.wv, kv_dim, hidden, block, kv_dim, false);
            ops::rmsnorm(q, q, &l.q_norm, eps, HEAD_DIM, block * c.n_heads);
            ops::rmsnorm(k, k, &l.k_norm, eps, HEAD_DIM, block * c.n_kv_heads);
            ops::rope(q, &self.rope, c.n_heads, start, block);
            ops::rope(k, &self.rope, c.n_kv_heads, start, block);
            ops::attn(
                a,
                q,
                &l.keys,
                &l.values,
                &self.scratch,
                c.n_heads,
                c.n_kv_heads,
                Span::All { len: start + block },
                self.max_seq,
                block,
            );
            ops::gemm(y, a, &l.wo, hidden, q_dim, block, hidden, false);
            ops::conv2(h, y, coef, &l.attn_base, hidden, group, 1, true, block);
            // The MLP, wrapped the same way.
            ops::rmsnorm(xn, h, &l.ln2, eps, hidden, block);
            ops::gemm(coef, xn, &l.mlp_proj, coef_n, hidden, block, coef_n, false);
            ops::conv2(xc, xn, coef, &l.mlp_base, hidden, group, 0, false, block);
            ops::gemm(gu, xc, &l.wgu, 2 * inter, hidden, block, 2 * inter, false);
            ops::silu_mul(m, gu, inter, block);
            ops::gemm(y, m, &l.wd, hidden, inter, block, hidden, false);
            ops::conv2(h, y, coef, &l.mlp_base, hidden, group, 1, true, block);
        }
        ops::rmsnorm(xn, h, &self.norm, eps, hidden, block);
    }

    /// The block's final hidden states, `[block, hidden]`: for tests.
    #[cfg(test)]
    pub fn block_hidden(&mut self, anchor: TokenId, start: usize) -> Vec<f32> {
        self.run_block(anchor, start);
        self.xn.read(0, self.cfg.block * self.cfg.hidden)
    }
}

impl FeatureDraft for Dflash {
    fn block(&self) -> usize {
        self.cfg.block - 1
    }

    fn capacity(&self) -> usize {
        self.max_seq
    }

    fn observe(&mut self, start_pos: usize, features: &[f32]) -> Result<()> {
        let c = &self.cfg;
        let (hidden, kv_dim) = (c.hidden, c.kv_dim());
        let width = c.taps.len() * hidden;
        ensure!(
            features.len().is_multiple_of(width),
            "features are not rows of {} layers x {hidden}",
            c.taps.len()
        );
        let n = features.len() / width;
        ensure!(
            start_pos + n <= self.max_seq,
            "features past the drafter's cache"
        );
        let ctx = self.ctx.ptr();
        for (i, piece) in features.chunks(PIECE * width).enumerate() {
            let (pos, s) = (start_pos + i * PIECE, piece.len() / width);
            self.feat.write(0, piece);
            ops::gemm(
                ctx,
                self.feat.ptr(),
                &self.fc,
                hidden,
                width,
                s,
                hidden,
                false,
            );
            ops::rmsnorm(ctx, ctx, &self.hidden_norm, c.rms_norm_eps, hidden, s);
            for l in &self.layers {
                let (k, v) = (l.keys.at(pos * kv_dim), l.values.at(pos * kv_dim));
                ops::gemm(k, ctx, &l.wk, kv_dim, hidden, s, kv_dim, false);
                ops::gemm(v, ctx, &l.wv, kv_dim, hidden, s, kv_dim, false);
                ops::rmsnorm(k, k, &l.k_norm, c.rms_norm_eps, HEAD_DIM, s * c.n_kv_heads);
                ops::rope(k, &self.rope, c.n_kv_heads, pos, s);
            }
        }
        Ok(())
    }

    fn draft(&mut self, anchor: TokenId, start: usize) -> Result<Vec<TokenId>> {
        let (hidden, vocab, block) = (self.cfg.hidden, self.cfg.vocab, self.cfg.block);
        let rank = self.cfg.selector_rank;
        ensure!(
            start + block <= self.max_seq,
            "block past the drafter's cache"
        );
        ensure!(
            (0..vocab as i32).contains(&anchor),
            "anchor token outside the vocabulary"
        );
        let t0 = Instant::now();
        self.run_block(anchor, start);
        // The masked slots only: row 0 is the anchor.
        let slots = self.xn.at(hidden);
        let n = block - 1;
        ops::gemm(
            self.logits.ptr(),
            slots,
            &self.lm_head,
            vocab,
            hidden,
            n,
            vocab,
            false,
        );
        ops::gemm(
            self.hp.ptr(),
            slots,
            &self.sel_proj,
            rank,
            hidden,
            n,
            rank,
            false,
        );
        let k = self.selector.top_k;
        ops::topk(
            &self.cand_ids,
            &self.cand_vals,
            self.logits.ptr(),
            vocab,
            k,
            n,
        );
        let ids = self.cand_ids.read(0, n * k);
        let vals = self.cand_vals.read(0, n * k);
        let hp = self.hp.read(0, n * rank);
        let t1 = Instant::now();
        let tokens = self.selector.pick(anchor, &ids, &vals, &hp)?;
        self.stats.drafts += 1;
        self.stats.gpu += t1 - t0;
        self.stats.host += t1.elapsed();
        Ok(tokens)
    }
}

/// The candidate selector. Slot by slot it scores the `top_k` most likely tokens as
/// `logit + <predecessor[previous token] * projected hidden, successor[candidate]>` and keeps
/// the best, so each choice depends on the one before it.
pub struct Selector {
    pub rank: usize,
    pub top_k: usize,
    pub vocab: usize,
    /// `[vocab, rank]` bf16 bits.
    pub predecessor: Vec<u16>,
    pub successor: Vec<u16>,
}

impl Selector {
    /// One token per slot. `ids` and `vals` are `[slots, top_k]`: each slot's most likely
    /// tokens, most likely first, and their logits, as [`ops::topk`] returns them; `hp` is
    /// `[slots, rank]`.
    pub fn pick(
        &self,
        anchor: TokenId,
        ids: &[i32],
        vals: &[f32],
        hp: &[f32],
    ) -> Result<Vec<TokenId>> {
        let (rank, k) = (self.rank, self.top_k);
        let row = |table: &[u16], token: usize| -> Vec<f32> {
            table[token * rank..(token + 1) * rank]
                .iter()
                .map(|&b| bf16::from_bits(b).to_f32())
                .collect()
        };
        let mut previous = anchor as usize;
        let mut out = Vec::with_capacity(ids.len() / k);
        for ((ids, vals), h) in ids.chunks(k).zip(vals.chunks(k)).zip(hp.chunks(rank)) {
            if ids[0] < 0 || ids.iter().any(|&t| t as usize >= self.vocab) {
                bail!("the drafter produced non-finite logits");
            }
            // The transition weights for this slot: predecessor row scaled by the hidden state.
            let pred: Vec<f32> = row(&self.predecessor, previous)
                .iter()
                .zip(h)
                .map(|(a, b)| a * b)
                .collect();
            let mut best = (f32::NEG_INFINITY, 0usize);
            for (&token, &logit) in ids.iter().zip(vals) {
                let token = token as usize;
                let pair: f32 = row(&self.successor, token)
                    .iter()
                    .zip(&pred)
                    .map(|(s, p)| s * p)
                    .sum();
                let score = logit + pair;
                if score > best.0 {
                    best = (score, token);
                }
            }
            out.push(best.1 as TokenId);
            previous = best.1;
        }
        Ok(out)
    }
}

/// What [`ops::topk`] computes, on the host: each row's `k` most likely tokens and their
/// logits, with `-1` first for a row that holds a NaN or an infinity.
#[cfg(test)]
pub fn rank(logits: &[f32], vocab: usize, k: usize) -> (Vec<i32>, Vec<f32>) {
    let (mut ids, mut vals) = (Vec::new(), Vec::new());
    for row in logits.chunks(vocab) {
        let top = top_k(row, k);
        let bad = row.iter().any(|x| !x.is_finite());
        ids.extend(
            top.iter()
                .enumerate()
                .map(|(i, &t)| if bad && i == 0 { -1 } else { t as i32 }),
        );
        vals.extend(top.iter().map(|&t| row[t]));
    }
    (ids, vals)
}

/// Indices of the `k` largest values, largest first; ties go to the lower index.
#[cfg(test)]
fn top_k(values: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    let order = |a: &usize, b: &usize| values[*b].total_cmp(&values[*a]).then(a.cmp(b));
    if k < idx.len() {
        idx.select_nth_unstable_by(k - 1, order);
        idx.truncate(k);
    }
    idx.sort_unstable_by(order);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(xs: &[f32]) -> Vec<u16> {
        xs.iter().map(|&x| bf16::from_f32(x).to_bits()).collect()
    }

    #[test]
    fn top_k_orders_by_value_then_index() {
        assert_eq!(top_k(&[0.5, 2.0, -1.0, 2.0, 1.0], 3), vec![1, 3, 4]);
        assert_eq!(top_k(&[3.0, 1.0], 5), vec![0, 1]);
    }

    #[test]
    fn selector_conditions_each_slot_on_the_previous_choice() {
        // Vocabulary of 3, rank 1. After token 0 the pair term favours token 2; after token 2
        // it favours token 1. Without the pair term both slots would pick token 0.
        let sel = Selector {
            rank: 1,
            top_k: 3,
            vocab: 3,
            predecessor: bits(&[1.0, 0.0, -1.0]),
            successor: bits(&[0.0, -4.0, 4.0]),
        };
        let logits = [1.0, 0.0, 0.5, 1.0, 0.0, 0.5];
        let hp = [1.0, 1.0];
        let pick = |sel: &Selector, logits: &[f32], hp: &[f32]| {
            let (ids, vals) = rank(logits, 3, sel.top_k);
            sel.pick(0, &ids, &vals, hp)
        };
        assert_eq!(pick(&sel, &logits, &hp).unwrap(), vec![2, 1]);
        // With top_k = 1 only the most likely token is a candidate.
        let greedy = Selector { top_k: 1, ..sel };
        assert_eq!(pick(&greedy, &logits, &hp).unwrap(), vec![0, 0]);
        assert!(pick(&greedy, &[f32::NAN, 0.0, 0.0], &[1.0]).is_err());
    }
}

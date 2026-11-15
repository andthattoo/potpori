//! A contiguous range of Nanbeige layers on one GPU: the unit a node serves.
//!
//! The first shard owns the embedding table, the last one the final norm and the output head.
//! Keys and values are written at absolute positions, so feeding a chunk again from an earlier
//! position overwrites whatever a rejected draft left there.

use std::sync::Arc;

use anyhow::{ensure, Result};

use crate::engine::{TargetChain, TokenId};
use crate::gpu::Buf;
use crate::node::ShardEngine;
use crate::transport::SessionHello;

use super::config::{rope_table, Config};
use super::ops::{self, Span};
use super::weights::Checkpoint;
use super::{argmax, is_eos, model_id};

/// Tokens run per launch. Longer feeds (a prompt) are cut into pieces of this size; the
/// result does not depend on where the cuts fall.
pub const PIECE: usize = 32;

/// A weight matrix on the device, in whichever form the checkpoint stores it.
enum Mat {
    Bf16(Buf<u16>),
    /// NVFP4: packed 4-bit weights, an 8-bit scale per 16 of them, one f32 scale.
    Fp4 {
        w: Buf<u8>,
        scale: Buf<u8>,
        global: f32,
    },
}

impl Mat {
    fn load(ck: &Checkpoint, name: &str, shape: [usize; 2]) -> Result<Self> {
        if ck.is_fp4(name) {
            ck.fp4(name, shape, |w, scale, global| Mat::Fp4 {
                w: Buf::from_slice(w),
                scale: Buf::from_slice(scale),
                global,
            })
        } else {
            Ok(Mat::Bf16(Buf::from_slice(&ck.bf16(name, &shape)?)))
        }
    }

    /// `out[t, :n_out] (+)= self x[t, :]` for `s` tokens; see [`ops::gemm`].
    #[allow(clippy::too_many_arguments)]
    fn mul(&self, out: u64, x: u64, n_out: usize, n_in: usize, s: usize, stride: usize, add: bool) {
        match self {
            Mat::Bf16(w) => ops::gemm(out, x, w, n_out, n_in, s, stride, add),
            Mat::Fp4 { w, scale, global } => {
                ops::gemm4(out, x, w, scale, *global, n_out, n_in, s, stride, add)
            }
        }
    }
}

struct Layer {
    ln1: Buf<f32>,
    wq: Mat,
    wk: Mat,
    wv: Mat,
    wo: Mat,
    ln2: Buf<f32>,
    wg: Mat,
    wu: Mat,
    wd: Mat,
    /// `[max_seq, kv_heads, HEAD_DIM]`, keys already rotated.
    keys: Buf<f32>,
    values: Buf<f32>,
}

pub struct Shard {
    cfg: Config,
    /// Whether any layer matrix is 4-bit: part of what nodes compare in the handshake.
    fp4: bool,
    range: (usize, usize),
    max_seq: usize,
    embed: Option<Arc<Buf<u16>>>,
    layers: Vec<Layer>,
    /// Final norm weight and output head.
    head: Option<(Buf<f32>, Arc<Buf<u16>>)>,
    rope: Buf<f32>,
    // Work buffers for one piece.
    ids: Buf<i32>,
    h: Buf<f32>,
    xn: Buf<f32>,
    q: Buf<f32>,
    a: Buf<f32>,
    gu: Buf<f32>,
    m: Buf<f32>,
    scratch: Buf<f32>,
    logits: Buf<f32>,
}

/// Device state that outlives one [`Shard`]: what a node that swaps layer groups in and out
/// of a small GPU keeps resident between them.
#[derive(Default)]
pub struct Kept {
    /// The embedding table, for a range that starts at layer 0.
    pub embed: Option<Arc<Buf<u16>>>,
    /// The output head, for a range that ends at the last layer.
    pub lm_head: Option<Arc<Buf<u16>>>,
    /// One `(keys, values)` cache per layer of the range, or empty for fresh ones.
    pub caches: Vec<(Buf<f32>, Buf<f32>)>,
}

impl Shard {
    /// Load layers `range` from `ck`, with a KV cache for `max_seq` positions.
    pub fn load(
        cfg: Config,
        range: (usize, usize),
        ck: &Checkpoint,
        max_seq: usize,
    ) -> Result<Self> {
        Self::load_with(cfg, range, ck, max_seq, Kept::default())
    }

    /// As [`Self::load`], reusing whatever `kept` already holds on the device.
    pub fn load_with(
        cfg: Config,
        range: (usize, usize),
        ck: &Checkpoint,
        max_seq: usize,
        kept: Kept,
    ) -> Result<Self> {
        let (start, end) = range;
        ensure!(
            start < end && end <= cfg.layers,
            "invalid layer range {range:?}"
        );
        ensure!(max_seq > 0, "max_seq must be positive");
        ensure!(
            kept.caches.is_empty() || kept.caches.len() == end - start,
            "one kept cache per layer"
        );
        let (hidden, inter) = (cfg.hidden, cfg.intermediate);
        let (q_dim, kv_dim) = (cfg.q_dim(), cfg.kv_dim());
        let table = |name: &str| -> Result<Buf<u16>> {
            Ok(Buf::from_slice(&ck.bf16(name, &[cfg.vocab, hidden])?))
        };
        let mat = |name: &str, shape: [usize; 2]| Mat::load(ck, name, shape);
        let vec =
            |name: &str| -> Result<Buf<f32>> { Ok(Buf::from_slice(&ck.f32(name, &[hidden])?)) };

        let embed = match (start == 0, kept.embed) {
            (false, _) => None,
            (true, Some(embed)) => Some(embed),
            (true, None) => Some(Arc::new(table("model.embed_tokens.weight")?)),
        };
        let mut caches = kept.caches.into_iter();
        let mut layers = Vec::with_capacity(end - start);
        for l in start..end {
            let p = |name: &str| format!("model.layers.{l}.{name}.weight");
            let (keys, values) = caches
                .next()
                .unwrap_or_else(|| (Buf::zeros(max_seq * kv_dim), Buf::zeros(max_seq * kv_dim)));
            ensure!(
                keys.len() == max_seq * kv_dim && values.len() == max_seq * kv_dim,
                "kept cache of the wrong size"
            );
            layers.push(Layer {
                ln1: vec(&p("input_layernorm"))?,
                wq: mat(&p("self_attn.q_proj"), [q_dim, hidden])?,
                wk: mat(&p("self_attn.k_proj"), [kv_dim, hidden])?,
                wv: mat(&p("self_attn.v_proj"), [kv_dim, hidden])?,
                wo: mat(&p("self_attn.o_proj"), [hidden, q_dim])?,
                ln2: vec(&p("post_attention_layernorm"))?,
                wg: mat(&p("mlp.gate_proj"), [inter, hidden])?,
                wu: mat(&p("mlp.up_proj"), [inter, hidden])?,
                wd: mat(&p("mlp.down_proj"), [hidden, inter])?,
                keys,
                values,
            });
        }
        let head = match (end == cfg.layers, kept.lm_head) {
            (false, _) => None,
            (true, Some(lm_head)) => Some((vec("model.norm.weight")?, lm_head)),
            (true, None) => Some((
                vec("model.norm.weight")?,
                Arc::new(table("lm_head.weight")?),
            )),
        };
        let logits = Buf::zeros(if head.is_some() { PIECE * cfg.vocab } else { 0 });
        let fp4 = layers.iter().any(|l| matches!(l.wq, Mat::Fp4 { .. }));
        Ok(Self {
            cfg,
            fp4,
            range,
            max_seq,
            embed,
            layers,
            head,
            rope: Buf::from_slice(&rope_table(cfg.rope_theta, max_seq)),
            ids: Buf::zeros(PIECE),
            h: Buf::zeros(PIECE * hidden),
            xn: Buf::zeros(PIECE * hidden),
            q: Buf::zeros(PIECE * q_dim),
            a: Buf::zeros(PIECE * q_dim),
            gu: Buf::zeros(PIECE * 2 * inter),
            m: Buf::zeros(PIECE * inter),
            scratch: Buf::zeros(PIECE * cfg.n_heads * max_seq),
            logits,
        })
    }

    /// The embedding table, for a drafter on the same GPU. First shard only.
    pub fn embedding(&self) -> Option<Arc<Buf<u16>>> {
        self.embed.clone()
    }

    /// Give up the weights and keep the KV caches, in layer order.
    pub fn into_caches(self) -> Vec<(Buf<f32>, Buf<f32>)> {
        self.layers
            .into_iter()
            .map(|l| (l.keys, l.values))
            .collect()
    }

    /// Run the layers on the `s` rows in `self.h`, tokens at positions `pos..pos + s`, and
    /// return this shard's output for them: the residual stream, or logits from the last one.
    fn run(&mut self, pos: usize, s: usize) -> Vec<f32> {
        let c = self.cfg;
        let (hidden, inter) = (c.hidden, c.intermediate);
        let (q_dim, kv_dim) = (c.q_dim(), c.kv_dim());
        let (gu, up) = (self.gu.ptr(), self.gu.at(inter));
        let (h, xn, q, a, m) = (
            self.h.ptr(),
            self.xn.ptr(),
            self.q.ptr(),
            self.a.ptr(),
            self.m.ptr(),
        );
        for l in &self.layers {
            let (k, v) = (l.keys.at(pos * kv_dim), l.values.at(pos * kv_dim));
            ops::rmsnorm(xn, h, &l.ln1, c.rms_norm_eps, hidden, s);
            l.wq.mul(q, xn, q_dim, hidden, s, q_dim, false);
            l.wk.mul(k, xn, kv_dim, hidden, s, kv_dim, false);
            l.wv.mul(v, xn, kv_dim, hidden, s, kv_dim, false);
            ops::rope(q, &self.rope, c.n_heads, pos, s);
            ops::rope(k, &self.rope, c.n_kv_heads, pos, s);
            ops::attn(
                a,
                q,
                &l.keys,
                &l.values,
                &self.scratch,
                c.n_heads,
                c.n_kv_heads,
                Span::Causal { pos0: pos },
                self.max_seq,
                s,
            );
            l.wo.mul(h, a, hidden, q_dim, s, hidden, true);
            ops::rmsnorm(xn, h, &l.ln2, c.rms_norm_eps, hidden, s);
            // Gate into the first half of each row of `gu`, up into the second.
            l.wg.mul(gu, xn, inter, hidden, s, 2 * inter, false);
            l.wu.mul(up, xn, inter, hidden, s, 2 * inter, false);
            ops::silu_mul(m, gu, inter, s);
            l.wd.mul(h, m, hidden, inter, s, hidden, true);
        }
        match &self.head {
            None => self.h.read(0, s * hidden),
            Some((norm, lm_head)) => {
                ops::rmsnorm(xn, h, norm, c.rms_norm_eps, hidden, s);
                ops::gemm(
                    self.logits.ptr(),
                    xn,
                    lm_head,
                    c.vocab,
                    hidden,
                    s,
                    c.vocab,
                    false,
                );
                self.logits.read(0, s * c.vocab)
            }
        }
    }

    fn check_span(&self, pos: u32, s: usize) {
        assert!(
            s > 0 && pos as usize + s <= self.max_seq,
            "{s} tokens at position {pos} do not fit a cache of {}",
            self.max_seq
        );
    }
}

impl ShardEngine for Shard {
    fn hello(&self, next_layer: usize) -> SessionHello {
        SessionHello {
            model: model_id(self.fp4).into(),
            next_layer,
            max_seq: self.max_seq,
            chunk: 0,
            taps: Vec::new(),
        }
    }
    fn range(&self) -> (usize, usize) {
        self.range
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
        self.check_span(pos, tokens.len());
        let embed = self
            .embed
            .clone()
            .expect("only the first shard takes tokens");
        assert!(
            tokens
                .iter()
                .all(|&t| (0..self.cfg.vocab as i32).contains(&t)),
            "token id outside the vocabulary"
        );
        let mut out = Vec::new();
        for (i, piece) in tokens.chunks(PIECE).enumerate() {
            self.ids.write(0, piece);
            ops::embed(
                self.h.ptr(),
                &embed,
                &self.ids,
                self.cfg.hidden,
                piece.len(),
            );
            out.extend(self.run(pos as usize + i * PIECE, piece.len()));
        }
        out
    }

    fn forward_hidden(&mut self, pos: u32, s: usize, hidden: Vec<f32>) -> Vec<f32> {
        self.check_span(pos, s);
        assert_eq!(
            hidden.len(),
            s * self.cfg.hidden,
            "hidden state of the wrong size"
        );
        let mut out = Vec::new();
        for (i, piece) in hidden.chunks(PIECE * self.cfg.hidden).enumerate() {
            self.h.write(0, piece);
            out.extend(self.run(pos as usize + i * PIECE, piece.len() / self.cfg.hidden));
        }
        out
    }

    /// Nothing to clear: a new session writes its keys and values from position 0, and no
    /// token ever attends past its own position.
    fn reset(&mut self) {}
}

/// A shard that holds the whole model is a complete target: the single-process baseline.
impl TargetChain for Shard {
    fn capacity(&self) -> usize {
        self.max_seq
    }

    fn prefill(&mut self, prompt: &[TokenId]) -> TokenId {
        let vocab = self.cfg.vocab;
        let logits = self.forward_tokens(0, prompt);
        argmax(&logits[logits.len() - vocab..])
    }

    fn verify(&mut self, start_pos: u32, cur: TokenId, drafts: &[TokenId]) -> Vec<TokenId> {
        let chunk: Vec<TokenId> = std::iter::once(cur).chain(drafts.iter().copied()).collect();
        self.forward_tokens(start_pos, &chunk)
            .chunks(self.cfg.vocab)
            .map(argmax)
            .collect()
    }

    fn is_eos(&self, token: TokenId) -> bool {
        is_eos(token)
    }
}

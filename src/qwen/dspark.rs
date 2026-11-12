//! A DSpark drafter for Qwen3 on Luminal (see docs/dspark.md).
//!
//! The drafter is a few Qwen3-shaped layers that read the *target's* hidden states. For every
//! settled position it gets the outputs of a fixed set of target layers (the taps), projected
//! to one vector `P`. One pass then fills a block of tokens after the last settled token (the
//! anchor): the block's rows attend to `P` of every earlier position and to each other, with
//! no causal mask. A small Markov head adds a previous-token correction to the logits, token
//! by token.
//!
//! The graph runs a fixed number of rows per call, at consecutive absolute positions. Each
//! row is either a *context* row (its keys and values come from `P`, written to the cache at
//! its position) or a block row (keys and values from its own stream). A draft call rewrites
//! the last few context positions — at most `block + 1` are new after a verification — and
//! runs the block behind them; older positions are already in the cache. Rewriting a
//! position is an overwrite, as in the target's KV.
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use luminal::prelude::*;
use luminal_nn::{AttentionGeometry, KvCache};
use serde::Deserialize;

use super::model::{
    linear, linear_columns, rows_from_columns, weight, Inputs, QwenConfig, QwenGraph, StateBinding,
    TensorData,
};
use super::target::QwenRuntime;
use crate::engine::TokenId;
use crate::specfeat::FeatureDraft;

/// What a drafter checkpoint's `config.json` says.
#[derive(Clone, Debug)]
pub struct DsparkConfig {
    /// Geometry of the draft layers (`layers` = how many of them).
    pub model: QwenConfig,
    /// Tokens drafted per pass.
    pub block: usize,
    /// Layers of the target, and which of them the drafter reads (ascending).
    pub target_layers: usize,
    pub taps: Vec<usize>,
    /// Token that fills the block's unknown slots.
    pub mask_token: TokenId,
    /// Width of the Markov head; 0 = none.
    pub markov_rank: usize,
}

impl DsparkConfig {
    pub fn from_json(text: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct Raw {
            block_size: usize,
            num_hidden_layers: usize,
            hidden_size: usize,
            intermediate_size: usize,
            head_dim: usize,
            num_attention_heads: usize,
            num_key_value_heads: usize,
            vocab_size: usize,
            rms_norm_eps: f32,
            // transformers 5 nests the RoPE base; older configs have it at the top level.
            rope_theta: Option<f32>,
            rope_parameters: Option<Rope>,
            num_target_layers: usize,
            target_layer_ids: Vec<usize>,
            mask_token_id: TokenId,
            #[serde(default)]
            markov_rank: usize,
            #[serde(default)]
            markov_head_type: Option<String>,
        }
        #[derive(Deserialize)]
        struct Rope {
            rope_theta: f32,
        }
        let raw: Raw = serde_json::from_str(text).context("drafter config.json")?;
        let rope_theta = raw
            .rope_theta
            .or(raw.rope_parameters.map(|r| r.rope_theta))
            .context("drafter config.json has no rope_theta")?;
        ensure!(
            raw.markov_rank == 0 || raw.markov_head_type.as_deref() == Some("vanilla"),
            "only the \"vanilla\" Markov head is supported, not {:?}",
            raw.markov_head_type
        );
        let mut taps = raw.target_layer_ids;
        ensure!(
            !taps.is_empty() && taps.is_sorted() && taps.iter().all(|&l| l < raw.num_target_layers),
            "invalid target_layer_ids {taps:?}"
        );
        taps.dedup();
        ensure!(
            raw.block_size > 0 && (raw.mask_token_id as usize) < raw.vocab_size,
            "invalid block size or mask token"
        );
        Ok(Self {
            model: QwenConfig {
                layers: raw.num_hidden_layers,
                hidden: raw.hidden_size,
                intermediate: raw.intermediate_size,
                head_dim: raw.head_dim,
                n_heads: raw.num_attention_heads,
                n_kv_heads: raw.num_key_value_heads,
                vocab_size: raw.vocab_size,
                rms_norm_eps: raw.rms_norm_eps,
                rope_theta,
            },
            block: raw.block_size,
            target_layers: raw.num_target_layers,
            taps,
            mask_token: raw.mask_token_id,
            markov_rank: raw.markov_rank,
        })
    }

    /// Whether this drafter was trained for `target`.
    pub fn check_target(&self, target: &QwenConfig) -> Result<()> {
        ensure!(
            self.target_layers == target.layers
                && self.model.hidden == target.hidden
                && self.model.vocab_size == target.vocab_size,
            "this drafter is for a target with {} layers of width {}, not {} of width {}",
            self.target_layers,
            self.model.hidden,
            target.layers,
            target.hidden
        );
        Ok(())
    }

    /// Context rows per call: enough for everything one verification can settle.
    pub fn context_rows(&self) -> usize {
        self.block + 1
    }
    /// Rows per call.
    pub fn rows(&self) -> usize {
        self.context_rows() + self.block
    }
    /// Numbers per position the drafter reads: one hidden state per tapped layer.
    pub fn feature_width(&self) -> usize {
        self.taps.len() * self.model.hidden
    }
}

/// The drafter graph's step inputs beyond a shard's.
#[derive(Clone, Copy)]
pub struct DrafterPorts {
    /// `[rows, taps * hidden]`: target features of the context rows (zeros elsewhere).
    features: NodeIndex,
    /// `[rows]`: 1.0 for a context row, 0.0 for a block row.
    from_target: NodeIndex,
    /// `[block]`: which rows are the block.
    block_rows: NodeIndex,
    block: usize,
    width: usize,
}

impl DrafterPorts {
    pub fn ids(&self) -> [NodeIndex; 3] {
        [self.features, self.from_target, self.block_rows]
    }
}

/// Build the drafter graph. `tied` = the output head is the embedding table (one tensor).
pub fn build_graph(cfg: &DsparkConfig, capacity: usize, tied: bool) -> Result<QwenGraph> {
    let m = cfg.model;
    let rows = cfg.rows();
    ensure!(
        rows <= capacity && capacity <= i32::MAX as usize,
        "the drafter needs a cache of at least {rows} positions"
    );
    let mut cx = Graph::new();
    let tokens = cx.tensor(vec!['q'], DType::Int);
    let positions = cx.tensor(vec!['q'], DType::Int);
    let gather = cx.tensor(vec!['c'], DType::Int);
    let cos = cx.tensor(vec![IntExpr::from('q'), m.head_dim.into()], DType::F32);
    let sin = cx.tensor(vec![IntExpr::from('q'), m.head_dim.into()], DType::F32);
    let rotation = cx.tensor(vec![m.head_dim, m.head_dim], DType::F32);
    let features = cx.tensor(
        vec![IntExpr::from('q'), cfg.feature_width().into()],
        DType::F32,
    );
    let from_target = cx.tensor(vec!['q'], DType::F32);
    let block_rows = cx.tensor(vec![cfg.block], DType::Int);

    let caches: Vec<_> = (0..m.layers)
        .map(|_| {
            let mut slots = || cx.tensor(vec![capacity, m.kv_dim()], DType::F32);
            (slots(), slots())
        })
        .collect();

    let mut parameters = Vec::new();
    let mut w = |name: String, shape| weight(&mut cx, &mut parameters, name, shape);
    let embedding = w("embed_tokens.weight".into(), vec![m.vocab_size, m.hidden]);
    let head = if tied {
        embedding
    } else {
        w("lm_head.weight".into(), vec![m.vocab_size, m.hidden])
    };
    let fc = w("fc.weight".into(), vec![m.hidden, cfg.feature_width()]);
    let hidden_norm = w("hidden_norm.weight".into(), vec![m.hidden]);

    // P: the target's view of each context row. It feeds every layer unchanged.
    let p = luminal_nn::rms_norm(
        linear(features, fc),
        Some(hidden_norm),
        None,
        m.rms_norm_eps,
    );
    let pick = from_target.expand_dim(1, m.hidden);
    let geometry = AttentionGeometry::new(m.n_heads, m.n_kv_heads, m.head_dim);
    let mut x = luminal_nn::embedding(tokens, embedding);
    let mut state = Vec::new();
    for (l, &(k_cache, v_cache)) in caches.iter().enumerate() {
        let mut w = |suffix: &str, shape| w(format!("layers.{l}.{suffix}.weight"), shape);
        let attn_norm = w("input_layernorm", vec![m.hidden]);
        let wq = w("self_attn.q_proj", vec![m.q_dim(), m.hidden]);
        let wk = w("self_attn.k_proj", vec![m.kv_dim(), m.hidden]);
        let wv = w("self_attn.v_proj", vec![m.kv_dim(), m.hidden]);
        let wo = w("self_attn.o_proj", vec![m.hidden, m.q_dim()]);
        let qn = w("self_attn.q_norm", vec![m.head_dim]);
        let kn = w("self_attn.k_norm", vec![m.head_dim]);
        let ffn_norm = w("post_attention_layernorm", vec![m.hidden]);
        let gate = w("mlp.gate_proj", vec![m.intermediate, m.hidden]);
        let up = w("mlp.up_proj", vec![m.intermediate, m.hidden]);
        let down = w("mlp.down_proj", vec![m.hidden, m.intermediate]);

        let normed = luminal_nn::rms_norm(x, Some(attn_norm), None, m.rms_norm_eps);
        // Keys and values come from P on context rows and from the row's own stream on
        // block rows; queries always come from the row's own stream.
        let source = normed + (p - normed) * pick;
        let rotary = |t: GraphTensor, norm| {
            luminal_nn::rotary_apply(
                luminal_nn::rms_norm_heads(t, m.head_dim, norm, m.rms_norm_eps),
                m.head_dim,
                cos,
                sin,
                rotation,
            )
        };
        let q = rotary(linear(normed, wq), qn);
        let k = rotary(linear(source, wk), kn);
        let cache = KvCache::new(k_cache, v_cache).write(positions, k, linear(source, wv));
        let seen = cache.read(gather);
        // No bias: every row sees every gathered position, earlier or later.
        let weights = luminal_nn::grouped_query_scores(q, seen.keys, geometry).softmax(vec![3]);
        let attention = luminal_nn::grouped_query_apply(weights, seen.values, geometry);
        for (input, output) in [(k_cache, cache.keys), (v_cache, cache.values)] {
            state.push(StateBinding {
                input: input.id,
                output: output.id,
                elements: capacity * m.kv_dim(),
                dtype: DType::F32,
            });
        }
        x += linear(attention, wo);
        let normed = luminal_nn::rms_norm(x, Some(ffn_norm), None, m.rms_norm_eps);
        x += linear(linear(normed, gate).silu() * linear(normed, up), down);
    }
    let norm = w("norm.weight".into(), vec![m.hidden]);
    let block = luminal_nn::gather_rows(x, block_rows);
    // `[vocab, block]`; the host turns it.
    let logits = linear_columns(
        luminal_nn::rms_norm(block, Some(norm), None, m.rms_norm_eps),
        head,
    );
    Ok(QwenGraph {
        graph: cx,
        parameters,
        state,
        logits: logits.id,
        capacity,
        chunk_size: rows,
        cfg: m,
        first: true,
        input: tokens.id,
        positions: positions.id,
        gather: gather.id,
        cos: cos.id,
        sin: sin.id,
        rotation: rotation.id,
        drafter: Some(DrafterPorts {
            features: features.id,
            from_target: from_target.id,
            block_rows: block_rows.id,
            block: cfg.block,
            width: cfg.feature_width(),
        }),
    })
}

/// One execution of the drafter graph: `rows` consecutive positions from `first_pos`.
struct Call<'a> {
    first_pos: usize,
    tokens: Vec<i32>,
    /// Per row: `Some(features)` for a context row.
    features: Vec<Option<&'a [f32]>>,
    /// Cache positions `[0, visible)` are attended to.
    visible: usize,
    /// Row index of the block's first row.
    block_at: usize,
}

fn call_inputs(graph: &QwenGraph, call: Call) -> Result<Inputs> {
    let ports = graph.drafter.as_ref().expect("a drafter graph");
    let rows = graph.chunk_size;
    ensure!(
        call.tokens.len() == rows
            && call.features.len() == rows
            && call.first_pos + rows <= graph.capacity
            && call.visible >= 1
            && call.visible <= graph.capacity
            && call.block_at + ports.block <= rows,
        "invalid drafter call"
    );
    let positions: Vec<i32> = (call.first_pos..call.first_pos + rows)
        .map(|p| p as i32)
        .collect();
    let (cos, sin) = luminal_nn::rope_tables_split_half(
        &positions.iter().map(|&p| p as f32).collect::<Vec<_>>(),
        graph.cfg.head_dim,
        graph.cfg.rope_theta,
        1.,
    );
    let mut features = vec![0f32; rows * ports.width];
    let mut from_target = vec![0f32; rows];
    for (row, values) in call.features.iter().enumerate() {
        if let Some(values) = values {
            ensure!(
                values.len() == ports.width,
                "feature row of the wrong width"
            );
            features[row * ports.width..(row + 1) * ports.width].copy_from_slice(values);
            from_target[row] = 1.;
        }
    }
    let block_rows = (call.block_at..call.block_at + ports.block)
        .map(|r| r as i32)
        .collect();
    Ok([
        (graph.input, TensorData::I32(call.tokens)),
        (graph.positions, TensorData::I32(positions)),
        (
            graph.gather,
            TensorData::I32((0..call.visible as i32).collect()),
        ),
        (graph.cos, TensorData::F32(cos)),
        (graph.sin, TensorData::F32(sin)),
        (ports.features, TensorData::F32(features)),
        (ports.from_target, TensorData::F32(from_target)),
        (ports.block_rows, TensorData::I32(block_rows)),
    ]
    .into_iter()
    .collect())
}

/// Representative inputs for kernel search: a draft call with `c` positions visible.
pub(super) fn profile_inputs(graph: &QwenGraph, c: usize) -> Result<Inputs> {
    let ports = graph.drafter.as_ref().expect("a drafter graph");
    let rows = graph.chunk_size;
    let context = rows - ports.block;
    let row = vec![0.02f32; ports.width];
    call_inputs(
        graph,
        Call {
            first_pos: c - rows,
            tokens: vec![1; rows],
            features: (0..rows)
                .map(|r| (r < context).then_some(&row[..]))
                .collect(),
            visible: c,
            block_at: context,
        },
    )
}

/// The previous-token correction: `logits += W2 · W1[previous token]`.
struct Markov {
    rank: usize,
    /// `[vocab, rank]` each, row-major.
    w1: Vec<f32>,
    w2: Vec<f32>,
}

impl Markov {
    /// Greedy pick over `base + W2 · W1[prev]`, without materializing the corrected logits.
    fn pick(&self, base: &[f32], prev: TokenId) -> Option<TokenId> {
        let e = &self.w1[prev as usize * self.rank..(prev as usize + 1) * self.rank];
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
        let per = base.len().div_ceil(threads);
        let best = std::thread::scope(|scope| {
            let jobs: Vec<_> = base
                .chunks(per)
                .zip(self.w2.chunks(per * self.rank))
                .enumerate()
                .map(|(i, (base, w2))| {
                    scope.spawn(move || {
                        let mut best = (f32::NEG_INFINITY, 0usize);
                        for (j, (&b, row)) in
                            base.iter().zip(w2.chunks_exact(self.rank)).enumerate()
                        {
                            let score = b + dot(row, e);
                            if !score.is_finite() {
                                return None;
                            }
                            if score > best.0 {
                                best = (score, i * per + j);
                            }
                        }
                        Some(best)
                    })
                })
                .collect();
            let mut best = (f32::NEG_INFINITY, 0usize);
            for job in jobs {
                let candidate = job.join().expect("markov worker")?;
                // Strictly greater: the lowest index wins a tie, as in a single pass.
                if candidate.0 > best.0 {
                    best = candidate;
                }
            }
            Some(best)
        })?;
        Some(best.1 as TokenId)
    }
}

/// Dot product with independent partial sums, so the compiler can vectorize it.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (a8, a_rest) = a.as_chunks::<8>();
    let (b8, b_rest) = b.as_chunks::<8>();
    for (x, y) in a8.iter().zip(b8) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let tail: f32 = a_rest.iter().zip(b_rest).map(|(x, y)| x * y).sum();
    acc.iter().sum::<f32>() + tail
}

/// Where a drafter's time went.
#[derive(Clone, Copy, Debug, Default)]
pub struct DraftStats {
    pub drafts: u64,
    /// Graph executions, including the ones that only load context.
    pub graph: Duration,
    pub graph_calls: u64,
    pub markov: Duration,
}

/// A DSpark drafter on a Luminal runtime: implements [`FeatureDraft`] for the head.
pub struct DsparkDraft<R: QwenRuntime> {
    cfg: DsparkConfig,
    graph: QwenGraph,
    runtime: R,
    markov: Option<Markov>,
    /// Target features of every settled position, `[settled, width]`.
    features: Vec<f32>,
    /// Positions `[0, written)` have their context keys and values in the cache.
    written: usize,
    stats: DraftStats,
}

impl<R: QwenRuntime> DsparkDraft<R> {
    /// Load the drafter checkpoint at `weights` (a `model.safetensors` as published) and
    /// compile its graph with a cache of `capacity` positions.
    pub fn load(
        mut runtime: R,
        cfg: DsparkConfig,
        weights: &Path,
        capacity: usize,
        search: usize,
    ) -> Result<Self> {
        let tied = super::hf::same_tensor(weights, "embed_tokens.weight", "lm_head.weight")?;
        let graph = build_graph(&cfg, capacity, tied)?;
        let inputs = super::hf::load_inputs(&graph, weights).context("load drafter weights")?;
        let markov = if cfg.markov_rank > 0 {
            let shape = [cfg.model.vocab_size, cfg.markov_rank];
            let read = |name| super::hf::read_f32(weights, name, &shape);
            Some(Markov {
                rank: cfg.markov_rank,
                w1: read("markov_head.markov_w1.weight")?,
                w2: read("markov_head.markov_w2.weight")?,
            })
        } else {
            None
        };
        runtime
            .compile(&graph, inputs, search)
            .context("compile the drafter")?;
        Ok(Self {
            cfg,
            graph,
            runtime,
            markov,
            features: Vec::new(),
            written: 0,
            stats: DraftStats::default(),
        })
    }

    pub fn stats(&self) -> DraftStats {
        self.stats
    }

    fn settled(&self) -> usize {
        self.features.len() / self.cfg.feature_width()
    }

    /// Run the graph on `rows` positions from `first_pos`, of which those before `start` are
    /// context. With an `anchor` (the token at `start`) the block follows it and its logits
    /// are returned; without, the call only loads context.
    fn run(&mut self, first_pos: usize, start: usize, anchor: Option<TokenId>) -> Result<Vec<f32>> {
        let (rows, width) = (self.cfg.rows(), self.cfg.feature_width());
        let mut tokens = vec![self.cfg.mask_token; rows];
        let mut features = vec![None; rows];
        for (row, pos) in (first_pos..first_pos + rows).enumerate() {
            if pos < start {
                features[row] = Some(&self.features[pos * width..(pos + 1) * width]);
            } else if pos == start {
                tokens[row] = anchor.unwrap_or(self.cfg.mask_token);
            }
        }
        let (visible, block_at) = match anchor {
            Some(_) => (start + self.cfg.block, start - first_pos),
            None => (start.min(first_pos + rows), 0),
        };
        let inputs = call_inputs(
            &self.graph,
            Call {
                first_pos,
                tokens,
                features,
                visible,
                block_at,
            },
        )?;
        let t0 = Instant::now();
        let logits = rows_from_columns(&self.runtime.step(inputs, rows, visible)?, self.cfg.block);
        self.stats.graph += t0.elapsed();
        self.stats.graph_calls += 1;
        Ok(logits)
    }

    /// The block's logits before the Markov correction, `[block, vocab]`: what the reference
    /// implementation calls the base logits.
    pub(super) fn block_logits(&mut self, anchor: TokenId, start: usize) -> Result<Vec<f32>> {
        let (block, context) = (self.cfg.block, self.cfg.context_rows());
        let vocab = self.cfg.model.vocab_size;
        ensure!(
            start >= 1 && self.settled() == start,
            "drafting at position {start} needs features for exactly the {start} before it, \
             have {}",
            self.settled()
        );
        ensure!(
            start + block < self.graph.capacity && (0..vocab as TokenId).contains(&anchor),
            "no room to draft at position {start}, or an invalid anchor"
        );
        // Load context the draft call below will not reach (after a prompt, mostly).
        while self.written + context < start {
            let first = self.written;
            self.run(first, start, None)?;
            self.written = start.min(first + self.cfg.rows());
        }
        let logits = self.run(start.saturating_sub(context), start, Some(anchor))?;
        self.written = start;
        ensure!(
            logits.len() == block * vocab,
            "drafter output of the wrong size"
        );
        Ok(logits)
    }
}

impl<R: QwenRuntime> FeatureDraft for DsparkDraft<R> {
    fn block(&self) -> usize {
        self.cfg.block
    }

    fn capacity(&self) -> usize {
        self.graph.capacity
    }

    fn observe(&mut self, start_pos: usize, features: &[f32]) -> Result<()> {
        let width = self.cfg.feature_width();
        ensure!(
            features.len().is_multiple_of(width) && start_pos <= self.settled(),
            "features for position {start_pos} do not continue the {} observed",
            self.settled()
        );
        ensure!(
            features.iter().all(|x| x.is_finite()),
            "non-finite target features"
        );
        self.features.truncate(start_pos * width);
        self.features.extend_from_slice(features);
        self.written = self.written.min(start_pos);
        Ok(())
    }

    fn draft(&mut self, anchor: TokenId, start: usize) -> Result<Vec<TokenId>> {
        let (block, vocab) = (self.cfg.block, self.cfg.model.vocab_size);
        let logits = self.block_logits(anchor, start)?;

        let t0 = Instant::now();
        let mut prev = anchor;
        let mut out = Vec::with_capacity(block);
        for base in logits.chunks_exact(vocab) {
            let token = match &self.markov {
                Some(markov) => markov.pick(base, prev),
                None => crate::engine::argmax(base),
            };
            let Some(token) = token else {
                bail!("the drafter produced non-finite logits");
            };
            out.push(token);
            prev = token;
        }
        self.stats.markov += t0.elapsed();
        self.stats.drafts += 1;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{
        "architectures": ["Qwen3DSparkModel"], "block_size": 7, "num_hidden_layers": 5,
        "hidden_size": 2560, "intermediate_size": 9728, "head_dim": 128,
        "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 151936,
        "rms_norm_eps": 1e-06, "rope_theta": 1000000, "num_target_layers": 36,
        "target_layer_ids": [1, 9, 17, 25, 33], "mask_token_id": 151669,
        "markov_rank": 256, "markov_head_type": "vanilla", "enable_confidence_head": true
    }"#;

    #[test]
    fn reads_the_published_qwen3_4b_drafter_config() {
        let cfg = DsparkConfig::from_json(CONFIG).unwrap();
        assert_eq!((cfg.block, cfg.rows(), cfg.context_rows()), (7, 15, 8));
        assert_eq!(cfg.taps, [1, 9, 17, 25, 33]);
        assert_eq!(cfg.feature_width(), 12800);
        assert_eq!((cfg.model.layers, cfg.mask_token), (5, 151669));
        cfg.check_target(&QwenConfig::QWEN3_4B).unwrap();
        assert!(cfg.check_target(&QwenConfig::QWEN3_0_6B).is_err());
        // The chain is cut after each tapped layer.
        assert_eq!(
            crate::node::tap_ranges((0, 18), &cfg.taps),
            [(0, 2), (2, 10), (10, 18)]
        );
    }

    #[test]
    fn refuses_configs_it_cannot_run() {
        let rnn = CONFIG.replace("vanilla", "rnn");
        assert!(DsparkConfig::from_json(&rnn).is_err());
        let bad_tap = CONFIG.replace("[1, 9, 17, 25, 33]", "[1, 9, 36]");
        assert!(DsparkConfig::from_json(&bad_tap).is_err());
    }

    #[test]
    fn the_markov_pick_matches_a_plain_argmax() {
        let (vocab, rank) = (1013, 12);
        let wave = |seed: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * 31 + seed * 17) as f32 * 0.37).sin())
                .collect()
        };
        let markov = Markov {
            rank,
            w1: wave(1, vocab * rank),
            w2: wave(2, vocab * rank),
        };
        let base = wave(3, vocab);
        for prev in [0, 5, 1012] {
            let e = &markov.w1[prev * rank..(prev + 1) * rank];
            let corrected: Vec<f32> = (0..vocab)
                .map(|v| base[v] + dot(&markov.w2[v * rank..(v + 1) * rank], e))
                .collect();
            assert_eq!(
                markov.pick(&base, prev as TokenId),
                crate::engine::argmax(&corrected)
            );
        }
        let mut broken = base.clone();
        broken[700] = f32::NAN;
        assert_eq!(markov.pick(&broken, 0), None);
    }
}

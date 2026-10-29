//! Contiguous Qwen layer shards on the native Luminal runtime.
#![allow(dead_code)]
use super::model::{QwenConfig, QwenGraph, TensorData};
use super::target::{argmax, QwenRuntime};
use crate::engine::{TargetChain, TokenId};
const EOS_TOKEN: i32 = 151645;
const STOP_TOKEN: i32 = 151643;
enum ShardInput {
    Tokens(Vec<i32>),
    Hidden(Vec<f32>),
}
pub struct Shard<R: QwenRuntime> {
    graph: QwenGraph,
    runtime: R,
    cfg: QwenConfig,
    is_first: bool,
    is_last: bool,
    range: (usize, usize),
}
impl<R: QwenRuntime + 'static> Shard<R> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        runtime: R,
        cfg: QwenConfig,
        range: (usize, usize),
        first: bool,
        last: bool,
        path: &str,
        capacity: usize,
        search: usize,
    ) -> Self {
        Self::load_chunked(runtime, cfg, range, first, last, path, capacity, search, 0)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn load_chunked(
        mut runtime: R,
        cfg: QwenConfig,
        range: (usize, usize),
        first: bool,
        last: bool,
        path: &str,
        capacity: usize,
        search: usize,
        chunk: usize,
    ) -> Self {
        let graph = QwenGraph::build(cfg, range, first, last, capacity, chunk)
            .expect("invalid shard configuration");
        let weights =
            super::hf::load_inputs(&graph, std::path::Path::new(path)).expect("load shard weights");
        runtime
            .compile(&graph, weights, search)
            .expect("compile shard");
        Self {
            graph,
            runtime,
            cfg,
            is_first: first,
            is_last: last,
            range,
        }
    }
    fn run_chunk(&mut self, start_pos: u32, s: usize, input: ShardInput) -> Vec<f32> {
        let data = match input {
            ShardInput::Tokens(t) => TensorData::I32(t),
            ShardInput::Hidden(h) => TensorData::F32(h),
        };
        let inputs = self
            .graph
            .step_inputs(data, s, start_pos as usize)
            .expect("invalid shard input");
        self.runtime
            .step(inputs, s, start_pos as usize + s)
            .expect("execute shard")
    }
    pub fn reset(&mut self) {
        self.runtime.reset().expect("reset KV");
    }
    pub fn forward_tokens(&mut self, pos: u32, tokens: &[TokenId]) -> Vec<f32> {
        self.run_chunk(pos, tokens.len(), ShardInput::Tokens(tokens.to_vec()))
    }
    pub fn forward_hidden(&mut self, pos: u32, s: usize, hidden: Vec<f32>) -> Vec<f32> {
        self.run_chunk(pos, s, ShardInput::Hidden(hidden))
    }
    pub fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    pub fn hidden_dim(&self) -> usize {
        self.cfg.hidden
    }
    pub fn is_last(&self) -> bool {
        self.is_last
    }
    pub fn capacity(&self) -> usize {
        self.graph.capacity
    }
    pub fn chunk_size(&self) -> usize {
        self.graph.chunk_size
    }
    pub fn range(&self) -> (usize, usize) {
        self.range
    }
    pub fn model_layers(&self) -> usize {
        self.cfg.layers
    }
    pub fn hello(&self, next_layer: usize) -> crate::transport::SessionHello {
        crate::transport::SessionHello {
            model: format!(
                "qwen3:{}:{}:{}:{}:{}:{}",
                self.cfg.layers,
                self.cfg.hidden,
                self.cfg.intermediate,
                self.cfg.head_dim,
                self.cfg.n_heads,
                self.cfg.n_kv_heads
            ),
            next_layer,
            max_seq: self.capacity(),
            chunk: self.chunk_size(),
        }
    }
}
/// A node chain: shards run in sequence, each consuming the previous shard's output. Implements
/// [`TargetChain`] so the spec-decode driver drives the sharded model unchanged. In-process
/// hidden-state hand-off (the QUIC transport replaces the shard→shard hop).
pub struct ShardedChain<R: QwenRuntime> {
    shards: Vec<Shard<R>>,
    vocab_size: usize,
}

impl<R: QwenRuntime + 'static> ShardedChain<R> {
    pub fn new(shards: Vec<Shard<R>>) -> Self {
        assert!(!shards.is_empty(), "a chain needs at least one shard");
        assert!(shards[0].is_first, "first shard must embed tokens");
        assert!(
            shards.last().unwrap().is_last,
            "last shard must emit logits"
        );
        let vocab_size = shards[0].cfg.vocab_size;
        Self { shards, vocab_size }
    }

    /// Run `tokens` (absolute positions from `start_pos`) through every shard; return the last
    /// shard's logits as `[s, vocab]` flat.
    fn run(&mut self, start_pos: u32, tokens: &[TokenId]) -> Vec<f32> {
        let s = tokens.len();
        let last = self.shards.len() - 1;
        let mut carry: Option<ShardInput> = Some(ShardInput::Tokens(tokens.to_vec()));
        let mut logits = Vec::new();
        for (idx, shard) in self.shards.iter_mut().enumerate() {
            let input = carry.take().expect("carry is set for every shard");
            let out = shard.run_chunk(start_pos, s, input);
            if idx == last {
                logits = out;
            } else {
                carry = Some(ShardInput::Hidden(out));
            }
        }
        logits
    }

    pub fn reset(&mut self) {
        for shard in &mut self.shards {
            shard.reset();
        }
    }
}

impl<R: QwenRuntime + 'static> TargetChain for ShardedChain<R> {
    fn capacity(&self) -> usize {
        self.shards.iter().map(Shard::capacity).min().unwrap()
    }
    fn prefill(&mut self, prompt: &[TokenId]) -> TokenId {
        let logits = self.run(0, prompt);
        let v = self.vocab_size;
        let last = (prompt.len() - 1) * v;
        argmax(&logits[last..last + v])
    }

    fn verify(&mut self, start_pos: u32, cur: TokenId, drafts: &[TokenId]) -> Vec<TokenId> {
        let mut chunk = Vec::with_capacity(drafts.len() + 1);
        chunk.push(cur);
        chunk.extend_from_slice(drafts);
        let logits = self.run(start_pos, &chunk);
        let v = self.vocab_size;
        (0..chunk.len())
            .map(|j| argmax(&logits[j * v..(j + 1) * v]))
            .collect()
    }

    fn truncate(&mut self, _keep_to_pos: u32) {
        // No-op: each shard's KV is position-indexed; the next chunk overwrites the rejected tail.
    }

    fn is_eos(&self, token: TokenId) -> bool {
        token == EOS_TOKEN || token == STOP_TOKEN
    }
}

#[cfg(all(test, feature = "metal", target_vendor = "apple"))]
mod tests {
    use super::*;
    use crate::qwen::hf::prepare_hf_model;
    use crate::qwen::target::QwenTarget;

    fn greedy<T: TargetChain>(t: &mut T, prompt: &[TokenId], max_new: usize) -> Vec<TokenId> {
        let mut cur = t.prefill(prompt);
        let mut out = Vec::new();
        let mut base = prompt.len() as u32;
        while !t.is_eos(cur) && out.len() < max_new {
            out.push(cur);
            cur = t.verify(base, cur, &[])[0];
            base += 1;
        }
        out
    }

    // THE CUT, on the real model: a 2-shard pipeline (hidden state round-tripped at the layer
    // boundary) must reproduce the single-process forward bit-for-bit.
    // cargo test -p dria-node --features metal qwen_cut_2shard -- --ignored --nocapture
    #[test]
    #[ignore]
    fn qwen_cut_2shard() {
        let cfg = QwenConfig::QWEN3_0_6B;
        let dir = prepare_hf_model("Qwen/Qwen3-0.6B").expect("prepare 0.6B");
        let w = dir.join("model_combined_bf16_v1.safetensors");
        let wp = w.to_str().unwrap();

        // Single-process baseline.
        let mut full = QwenTarget::load(
            crate::qwen::runtime::MetalRuntime::initialize(()),
            cfg,
            wp,
            512,
            200,
        );

        // 2-shard chain: layers [0,k) (first) -> [k,L) (last).
        let k = cfg.layers / 2;
        let s0 = Shard::load(
            crate::qwen::runtime::MetalRuntime::initialize(()),
            cfg,
            (0, k),
            true,
            false,
            wp,
            512,
            200,
        );
        let s1 = Shard::load(
            crate::qwen::runtime::MetalRuntime::initialize(()),
            cfg,
            (k, cfg.layers),
            false,
            true,
            wp,
            512,
            200,
        );
        let mut chain = ShardedChain::new(vec![s0, s1]);

        let prompt: Vec<TokenId> = vec![40, 1079, 264, 4108, 4108];
        let max_new = 24;

        full.reset();
        let base = greedy(&mut full, &prompt, max_new);
        chain.reset();
        let sharded = greedy(&mut chain, &prompt, max_new);

        println!("\nfull (1 proc)   = {base:?}");
        println!("cut  (2 shards) = {sharded:?}");
        match (0..base.len().min(sharded.len())).find(|&i| base[i] != sharded[i]) {
            None if base.len() == sharded.len() => {
                println!("LOSSLESS CUT: 2-shard pipeline == single-process forward (0.6B, Metal)")
            }
            Some(i) => println!(
                "diverge @{i}: full={} cut={} (bf16 near-tie at the boundary?)",
                base[i], sharded[i]
            ),
            None => println!("length differ: full={} cut={}", base.len(), sharded.len()),
        }
        assert_eq!(base, sharded, "the cut must be lossless");
    }
}

//! One contiguous range of Qwen layers on a Luminal runtime, with its own KV cache.
//!
//! The first shard embeds token ids; later shards take the residual stream. The last shard
//! applies the final norm and the tied output projection and returns logits; the others
//! return the residual stream for the next shard. A [`Shard`] implements [`ShardEngine`], so
//! [`crate::node`] can serve it or use it as the head.
use super::model::{QwenConfig, QwenGraph, TensorData};
use super::target::QwenRuntime;
use crate::engine::TokenId;
use crate::node::ShardEngine;
use crate::transport::SessionHello;

enum ShardInput {
    Tokens(Vec<i32>),
    Hidden(Vec<f32>),
}
pub struct Shard<R: QwenRuntime> {
    graph: QwenGraph,
    runtime: R,
    cfg: QwenConfig,
    range: (usize, usize),
}
impl<R: QwenRuntime + 'static> Shard<R> {
    /// Build the graph for layers `range`, load its weights from the prepared file at `path`
    /// and compile it. `first` embeds tokens; `last` emits logits. `chunk > 0` compiles
    /// kernels for exactly 1 and `chunk` tokens (the verification chunk, gamma + 1).
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
        let output = self
            .runtime
            .step(inputs, s, start_pos as usize + s)
            .expect("execute shard");
        if self.range.1 == self.cfg.layers {
            // The output head returns `[vocab, s]` (see `model::linear_columns`).
            super::model::rows_from_columns(&output, s)
        } else {
            output
        }
    }
}

impl<R: QwenRuntime> Shard<R> {
    /// Where this shard's device time went, if its backend measures it.
    #[allow(dead_code)] // called by the Metal profiling test
    pub fn profile(&self) -> Option<super::runtime::StepProfile> {
        self.runtime.profile()
    }
    /// A summary of the compiled programs, if the backend provides one.
    #[allow(dead_code)] // called by the Metal profiling test
    pub fn describe(&self) -> Option<String> {
        self.runtime.describe()
    }
}

impl<R: QwenRuntime + 'static> ShardEngine for Shard<R> {
    fn hello(&self, next_layer: usize) -> SessionHello {
        SessionHello {
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
        self.cfg.vocab_size
    }
    fn capacity(&self) -> usize {
        self.graph.capacity
    }
    fn chunk_size(&self) -> usize {
        self.graph.chunk_size
    }
    fn forward_tokens(&mut self, pos: u32, tokens: &[TokenId]) -> Vec<f32> {
        self.run_chunk(pos, tokens.len(), ShardInput::Tokens(tokens.to_vec()))
    }
    fn forward_hidden(&mut self, pos: u32, s: usize, hidden: Vec<f32>) -> Vec<f32> {
        self.run_chunk(pos, s, ShardInput::Hidden(hidden))
    }
    fn reset(&mut self) {
        self.runtime.reset().expect("reset KV");
    }
}

#[cfg(all(test, feature = "metal", target_vendor = "apple"))]
mod tests {
    use super::*;
    use crate::engine::TargetChain;
    use crate::qwen::hf::{prepare_hf_model, COMBINED_WEIGHTS_FILE};
    use crate::qwen::runtime::MetalRuntime;
    use crate::qwen::target::{greedy, QwenTarget};

    /// Two shards in one process, the residual stream handed over as a plain `Vec<f32>`.
    struct TwoShards(Shard<MetalRuntime>, Shard<MetalRuntime>);
    impl TargetChain for TwoShards {
        fn prefill(&mut self, prompt: &[TokenId]) -> TokenId {
            // One token at a time: the kernels are specialized to 1 or `chunk` tokens.
            let mut last = 0;
            for (pos, &token) in prompt.iter().enumerate() {
                last = self.verify(pos as u32, token, &[])[0];
            }
            last
        }
        fn verify(&mut self, pos: u32, cur: TokenId, drafts: &[TokenId]) -> Vec<TokenId> {
            let chunk: Vec<_> = std::iter::once(cur).chain(drafts.iter().copied()).collect();
            let hidden = self.0.forward_tokens(pos, &chunk);
            let logits = self.1.forward_hidden(pos, chunk.len(), hidden);
            logits
                .chunks_exact(self.1.vocab_size())
                .map(crate::qwen::target::argmax)
                .collect()
        }
        fn is_eos(&self, token: TokenId) -> bool {
            crate::qwen::is_eos(token)
        }
    }

    // THE CUT, on the real model: two shards with the hidden state handed over at the layer
    // boundary must reproduce the single-process forward token for token.
    // cargo test --release --features metal qwen_cut_2shard -- --ignored --nocapture
    #[test]
    #[ignore = "needs Metal and downloads Qwen3-0.6B"]
    fn qwen_cut_2shard() {
        let cfg = QwenConfig::QWEN3_0_6B;
        let dir = prepare_hf_model("Qwen/Qwen3-0.6B").expect("prepare 0.6B");
        let w = dir.join(COMBINED_WEIGHTS_FILE);
        let wp = w.to_str().unwrap();
        let k = cfg.layers / 2;
        let rt = || MetalRuntime::default();

        let mut full = QwenTarget::load_chunked(rt(), cfg, wp, 512, 200, 8);
        let mut cut = TwoShards(
            Shard::load_chunked(rt(), cfg, (0, k), true, false, wp, 512, 200, 8),
            Shard::load_chunked(rt(), cfg, (k, cfg.layers), false, true, wp, 512, 200, 8),
        );

        let prompt: Vec<TokenId> = vec![40, 1079, 264, 4108, 4108];
        let base = greedy(&mut full, &prompt, 24);
        let sharded = greedy(&mut cut, &prompt, 24);
        println!("\nfull (1 proc)   = {base:?}\ncut  (2 shards) = {sharded:?}");
        assert_eq!(base, sharded, "the cut must not change the output");
    }
}

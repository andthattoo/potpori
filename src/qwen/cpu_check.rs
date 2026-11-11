//! The real graphs on Luminal's CPU reference executor, with tiny models: no GPU, no download.
//!   cargo test --release --features reference
//!
//! `tests/fixtures/dspark_tiny` holds a tiny random Qwen3 target, a tiny random DSpark drafter
//! and what transformers and DeepSpec compute with them (`tools/make_dspark_fixture.py`).
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Deserialize;

use super::dspark::{DsparkConfig, DsparkDraft};
use super::hf::read_f32;
use super::model::QwenConfig;
use super::runtime::CpuRuntime;
use super::shard::Shard;
use crate::engine::TokenId;
use crate::node::{tap_ranges, ShardEngine, TappedShards};
use crate::specfeat::{run_feature_decode, FeatureChain, FeatureDraft};

#[test]
fn tiny_shards_and_rollback_on_cpu() {
    super::tiny::shards_and_rollback::<CpuRuntime>();
}

const CAPACITY: usize = 64;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/dspark_tiny")
        .join(name)
}

#[derive(Deserialize)]
struct Expected {
    target: Target,
    prompt: Vec<TokenId>,
    /// The prompt followed by the target's greedy continuation.
    sequence: Vec<TokenId>,
    rounds: Vec<Round>,
}
#[derive(Deserialize)]
struct Target {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
}
/// One draft call of the reference run: the block proposed after `anchor` at `start`.
#[derive(Deserialize)]
struct Round {
    start: usize,
    anchor: TokenId,
    drafts: Vec<TokenId>,
}

fn expected() -> Expected {
    serde_json::from_str(&std::fs::read_to_string(fixture("expected.json")).unwrap()).unwrap()
}

fn drafter_config() -> DsparkConfig {
    DsparkConfig::from_json(&std::fs::read_to_string(fixture("config.json")).unwrap()).unwrap()
}

fn load_drafter() -> DsparkDraft<CpuRuntime> {
    let weights = fixture("drafter.safetensors");
    DsparkDraft::load(
        CpuRuntime::default(),
        drafter_config(),
        &weights,
        CAPACITY,
        8,
    )
    .unwrap()
}

fn assert_close(got: &[f32], want: &[f32], tolerance: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < tolerance && got.iter().all(|x| x.is_finite()),
        "{what}: differs from the reference by up to {worst}"
    );
}

// The drafter graph against DeepSpec, given the reference's own target features: the same
// logits and the same drafted tokens, call after call, as the settled context grows by
// zero to eight positions — and again after going back to an earlier position.
#[test]
fn tiny_drafter_matches_deepspec() {
    let cfg = drafter_config();
    let expected = expected();
    let (width, vocab) = (cfg.feature_width(), cfg.model.vocab_size);
    let positions = expected.sequence.len();
    let features = read_f32(
        &fixture("expected.safetensors"),
        "features",
        &[positions, width],
    )
    .unwrap();
    let rounds = expected.rounds.len();
    let logits = read_f32(
        &fixture("expected.safetensors"),
        "base_logits",
        &[rounds, cfg.block, vocab],
    )
    .unwrap();

    let mut drafter = load_drafter();
    let mut seen = 0;
    let check = |drafter: &mut DsparkDraft<CpuRuntime>, seen: usize, i: usize| {
        let round = &expected.rounds[i];
        assert_eq!(expected.sequence[round.start], round.anchor);
        drafter
            .observe(seen, &features[seen * width..round.start * width])
            .unwrap();
        let want = &logits[i * cfg.block * vocab..(i + 1) * cfg.block * vocab];
        let got = drafter.block_logits(round.anchor, round.start).unwrap();
        assert_close(&got, want, 2e-3, &format!("round {i} logits"));
        // Drafting again from the same state changes nothing, and gives the tokens.
        let drafts = drafter.draft(round.anchor, round.start).unwrap();
        assert_eq!(drafts, round.drafts, "round {i}");
    };
    for i in 0..rounds {
        check(&mut drafter, seen, i);
        seen = expected.rounds[i].start;
    }
    // A rejected tail: go back to an earlier round's position and continue from there.
    let back = expected.rounds[3].start;
    check(&mut drafter, back - 2, 3);
    check(&mut drafter, back, 4);
    let stats = drafter.stats();
    assert!(stats.drafts as usize == rounds + 2 && stats.graph_calls > stats.drafts);
}

/// Head and tail of the tiny target in one process, each cut at the tapped layers.
struct LocalChain {
    head: TappedShards<Shard<CpuRuntime>>,
    tail: TappedShards<Shard<CpuRuntime>>,
    eos: TokenId,
    logits: Vec<f32>,
}

impl LocalChain {
    fn load(cfg: QwenConfig, taps: &[usize], cut: usize, chunk: usize) -> Self {
        let weights = fixture("target.safetensors");
        let node = |range| {
            let parts = tap_ranges(range, taps)
                .into_iter()
                .map(|(a, b)| {
                    Shard::load_chunked(
                        CpuRuntime::default(),
                        cfg,
                        (a, b),
                        a == 0,
                        b == cfg.layers,
                        weights.to_str().unwrap(),
                        CAPACITY,
                        8,
                        chunk,
                    )
                })
                .collect();
            TappedShards::new(parts, taps.to_vec()).unwrap()
        };
        Self {
            head: node((0, cut)),
            tail: node((cut, cfg.layers)),
            eos: -1,
            logits: Vec::new(),
        }
    }

    /// Run a chunk through both nodes: the greedy token and the features of each position.
    fn feed(&mut self, pos: u32, tokens: &[TokenId]) -> (Vec<TokenId>, Vec<f32>) {
        let hidden = self.head.forward_tokens(pos, tokens);
        let logits = self.tail.forward_hidden(pos, tokens.len(), hidden);
        let mut taps = self.head.take_taps();
        taps.extend(self.tail.take_taps());
        let width = self.head.hidden_dim();
        let mut features = Vec::new();
        for row in 0..tokens.len() {
            for tap in &taps {
                features.extend_from_slice(&tap[row * width..(row + 1) * width]);
            }
        }
        let next = logits
            .chunks_exact(self.head.vocab_size())
            .map(|row| crate::engine::argmax(row).unwrap())
            .collect();
        self.logits.extend(logits);
        (next, features)
    }
}

impl FeatureChain for LocalChain {
    fn capacity(&self) -> usize {
        CAPACITY
    }
    fn prefill_features(&mut self, prompt: &[TokenId]) -> Result<(TokenId, Vec<f32>)> {
        let mut features = Vec::new();
        let mut last = 0;
        let chunk = self.head.chunk_size();
        let mut pos = 0;
        while pos < prompt.len() {
            let take = if prompt.len() - pos >= chunk {
                chunk
            } else {
                1
            };
            let (next, rows) = self.feed(pos as u32, &prompt[pos..pos + take]);
            last = *next.last().unwrap();
            features.extend(rows);
            pos += take;
        }
        Ok((last, features))
    }
    fn verify_features(
        &mut self,
        start_pos: u32,
        cur: TokenId,
        drafts: &[TokenId],
    ) -> Result<(Vec<TokenId>, Vec<f32>)> {
        let chunk: Vec<_> = std::iter::once(cur).chain(drafts.iter().copied()).collect();
        Ok(self.feed(start_pos, &chunk))
    }
    fn is_eos(&self, token: TokenId) -> bool {
        token == self.eos
    }
}

// The target cut into four shards at the tapped layers against transformers: the same
// logits, and tapped outputs equal to `hidden_states` of the layers the drafter was
// configured with. Then the whole loop — tapped target, drafter, driver — reproduces the
// target's greedy continuation.
#[test]
fn tiny_tapped_target_matches_transformers_and_drives_the_drafter() {
    let expected = expected();
    let t = &expected.target;
    let cfg = QwenConfig {
        layers: t.num_hidden_layers,
        hidden: t.hidden_size,
        intermediate: t.intermediate_size,
        head_dim: t.head_dim,
        n_heads: t.num_attention_heads,
        n_kv_heads: t.num_key_value_heads,
        vocab_size: t.vocab_size,
        rms_norm_eps: t.rms_norm_eps,
        rope_theta: t.rope_theta,
    };
    let draft_cfg = drafter_config();
    draft_cfg.check_target(&cfg).unwrap();
    let chunk = draft_cfg.block + 1;
    let mut chain = LocalChain::load(cfg, &draft_cfg.taps, cfg.layers / 2, chunk);
    assert_eq!(chain.head.tap_layers(), [0, 2]);
    assert_eq!(chain.tail.tap_layers(), [4]);

    // The whole reference sequence, in the pieces the kernels take.
    let sequence = &expected.sequence;
    let (next, features) = chain.prefill_features(sequence).unwrap();
    let positions = sequence.len();
    let file = fixture("expected.safetensors");
    let want = read_f32(&file, "target_logits", &[positions, cfg.vocab_size]).unwrap();
    assert_close(&chain.logits, &want, 2e-3, "target logits");
    let want = read_f32(&file, "features", &[positions, draft_cfg.feature_width()]).unwrap();
    assert_close(&features, &want, 2e-3, "tapped features");
    let _ = next;

    // Generation: only the target's own picks are committed, whatever the drafter says.
    let mut drafter = load_drafter();
    let prompt = &expected.prompt;
    let max_new = 36;
    let (out, stats) = run_feature_decode(&mut drafter, &mut chain, prompt, max_new).unwrap();
    assert_eq!(out, sequence[prompt.len()..prompt.len() + max_new]);
    assert!(stats.traversals <= max_new && drafter.stats().drafts > 0);
}

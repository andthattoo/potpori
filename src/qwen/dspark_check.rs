//! The DSpark drafter on real weights and a real GPU: Qwen3-4B and its published drafter.
//! `#[ignore]` — it downloads about 11 GB and runs seven kernel searches.
//!
//!   cargo test --release --features metal dspark_on_real_weights -- --ignored --nocapture
//!   cargo test --release --features cuda  dspark_on_real_weights -- --ignored --nocapture
//!
//! It prints what the drafter buys — tokens per trip through the target — and fails if the
//! drafter is clearly not reading the target (acceptance near zero).
//!
//! The target and the drafter never share the GPU, so this fits where both at once would
//! not (about 21 GB with 32-bit weights). First the target, cut at the tapped layers as on
//! a real chain, decodes each prompt greedily and its features are kept. Then the target is
//! dropped and the drafter replays the generation: a position's features do not depend on
//! later tokens, so the features of the greedy sequence are exactly what each verification
//! would have returned, and the accepted lengths are the ones a live chain would see.

use std::time::Instant;

use super::dspark::{DsparkConfig, DsparkDraft};
use super::hf::{download_drafter, prepare_hf_model, COMBINED_WEIGHTS_FILE};
use super::model::QwenConfig;
use super::shard::Shard;
use super::tok::ChatTokenizer;
use super::EOS_TOKENS;
use crate::engine::TokenId;
use crate::node::{tap_ranges, ShardEngine, TappedShards};
use crate::specfeat::FeatureDraft;

#[cfg(feature = "cuda")]
type Backend = super::runtime::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type Backend = super::runtime::MetalRuntime;

const TARGET: &str = "Qwen/Qwen3-4B";
const DRAFTER: &str = "deepseek-ai/dspark_qwen3_4b_block7";
const CFG: QwenConfig = QwenConfig::QWEN3_4B;
const MAX_SEQ: usize = 512;
/// The target only has to be right here, not fast: a short kernel search.
const TARGET_SEARCH: usize = 16;
const DRAFTER_SEARCH: usize = 200;
const MAX_NEW: usize = 96;
const PROMPTS: [&str; 3] = [
    "Explain speculative decoding in one paragraph.",
    "Write a Python function that checks whether a string is a palindrome.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the trip? Show the steps.",
];

fn load_node(range: (usize, usize), taps: &[usize], chunk: usize) -> TappedShards<Shard<Backend>> {
    let dir = prepare_hf_model(TARGET).expect("prepare the target");
    let weights = dir.join(COMBINED_WEIGHTS_FILE);
    let parts = tap_ranges(range, taps)
        .into_iter()
        .map(|(a, b)| {
            Shard::load_chunked(
                Backend::default(),
                CFG,
                (a, b),
                a == 0,
                b == CFG.layers,
                weights.to_str().unwrap(),
                MAX_SEQ,
                TARGET_SEARCH,
                chunk,
            )
        })
        .collect();
    TappedShards::new(parts, taps.to_vec()).unwrap()
}

/// A prompt, the target's greedy continuation, and the target's features of every position.
struct Decoded {
    prompt: usize,
    tokens: Vec<TokenId>,
    features: Vec<f32>,
}

/// Greedy decoding through both halves of the target, keeping each position's features.
fn decode(
    head: &mut TappedShards<Shard<Backend>>,
    tail: &mut TappedShards<Shard<Backend>>,
    prompt: &[TokenId],
    chunk: usize,
) -> Decoded {
    let (hidden, vocab) = (CFG.hidden, CFG.vocab_size);
    let mut tokens = prompt.to_vec();
    let mut features = Vec::new();
    let mut feed = |pos: usize, chunk: &[TokenId]| -> TokenId {
        let stream = head.forward_tokens(pos as u32, chunk);
        let logits = tail.forward_hidden(pos as u32, chunk.len(), stream);
        let mut taps = head.take_taps();
        taps.extend(tail.take_taps());
        for row in 0..chunk.len() {
            for tap in &taps {
                features.extend_from_slice(&tap[row * hidden..(row + 1) * hidden]);
            }
        }
        crate::engine::argmax(&logits[logits.len() - vocab..]).expect("finite logits")
    };
    // The prompt in the sizes the kernels take, then one token at a time.
    let mut pos = 0;
    let mut next = 0;
    while pos < prompt.len() {
        let take = if prompt.len() - pos >= chunk {
            chunk
        } else {
            1
        };
        next = feed(pos, &prompt[pos..pos + take]);
        pos += take;
    }
    while !EOS_TOKENS.contains(&next) && tokens.len() - prompt.len() < MAX_NEW {
        tokens.push(next);
        next = feed(tokens.len() - 1, &[next]);
    }
    Decoded {
        prompt: prompt.len(),
        tokens,
        features,
    }
}

#[test]
#[ignore = "needs a GPU and downloads Qwen3-4B and its DSpark drafter"]
fn dspark_on_real_weights() {
    let (config, weights) = download_drafter(DRAFTER).expect("download the drafter");
    let dcfg = DsparkConfig::from_json(&std::fs::read_to_string(config).unwrap()).unwrap();
    dcfg.check_target(&CFG).unwrap();
    let (block, width, chunk) = (dcfg.block, dcfg.feature_width(), dcfg.block + 1);
    let cut = CFG.layers / 2;
    let tok = ChatTokenizer::load(TARGET).unwrap();

    // 1. The target alone.
    let decoded: Vec<Decoded> = {
        let mut head = load_node((0, cut), &dcfg.taps, chunk);
        let mut tail = load_node((cut, CFG.layers), &dcfg.taps, chunk);
        PROMPTS
            .iter()
            .map(|prompt| {
                let ids = tok.encode_chat(prompt).unwrap();
                let t0 = Instant::now();
                let run = decode(&mut head, &mut tail, &ids, chunk);
                println!(
                    "target: {} prompt + {} generated tokens in {:.1}s",
                    run.prompt,
                    run.tokens.len() - run.prompt,
                    t0.elapsed().as_secs_f64()
                );
                assert_eq!(run.features.len(), run.tokens.len() * width);
                run
            })
            .collect()
    }; // the target's shards, and their GPU memory, are dropped here

    // 2. The drafter alone, replaying each generation.
    let mut drafter =
        DsparkDraft::load(Backend::default(), dcfg, &weights, MAX_SEQ, DRAFTER_SEARCH).unwrap();
    let mut accepted_at = vec![0usize; block];
    let (mut committed, mut trips) = (0usize, 0usize);
    for (prompt, run) in PROMPTS.iter().zip(&decoded) {
        let features = |from: usize, to: usize| &run.features[from * width..to * width];
        drafter.observe(0, features(0, run.prompt)).unwrap();
        let mut start = run.prompt; // position of the pending token
        let (mut tokens, mut rounds) = (0usize, 0usize);
        // Stop where a full block no longer has known tokens to be checked against.
        while start + block < run.tokens.len() {
            let drafts = drafter.draft(run.tokens[start], start).unwrap();
            let truth = &run.tokens[start + 1..];
            let n = drafts.iter().zip(truth).take_while(|(d, t)| d == t).count();
            (0..n).for_each(|i| accepted_at[i] += 1);
            // The pending token and the accepted drafts are settled; their features are
            // what the verification would have returned.
            drafter
                .observe(start, features(start, start + n + 1))
                .unwrap();
            start += n + 1;
            tokens += n + 1;
            rounds += 1;
        }
        println!(
            "\n{prompt}\n  {tokens} tokens in {rounds} trips = {:.2} tok/trip",
            tokens as f64 / rounds.max(1) as f64
        );
        println!(
            "  {}",
            tok.decode(&run.tokens[run.prompt..])
                .unwrap()
                .replace('\n', " ")
        );
        committed += tokens;
        trips += rounds;
    }
    let d = drafter.stats();
    let rate = committed as f64 / trips.max(1) as f64;
    println!(
        "\noverall {rate:.2} tok/trip over {trips} trips | accepted at draft position 1..{block}: \
         {} | per block: drafter graph {:.1} ms, markov {:.1} ms",
        accepted_at
            .iter()
            .map(|&n| format!("{:.2}", n as f64 / trips.max(1) as f64))
            .collect::<Vec<_>>()
            .join(" "),
        d.graph.as_secs_f64() * 1e3 / d.drafts.max(1) as f64,
        d.markov.as_secs_f64() * 1e3 / d.drafts.max(1) as f64,
    );
    assert!(
        rate > 2.0,
        "{rate:.2} tokens per trip: the drafter is not reading the target's features correctly"
    );
}

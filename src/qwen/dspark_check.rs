//! The DSpark path on real weights and a real GPU, in one process: Qwen3-4B cut in half
//! (each half split at the tapped layers), the published drafter, real QUIC over loopback.
//! `#[ignore]` — it downloads about 11 GB and runs seven kernel searches.
//!
//!   cargo test --release --features metal dspark_loopback -- --ignored --nocapture
//!   cargo test --release --features cuda  dspark_loopback -- --ignored --nocapture
//!
//! It checks that the output equals plain greedy decoding through the same chain, and prints
//! what the drafter buys: tokens per trip and where the time goes.

use std::time::{Duration, Instant};

use super::dspark::{DsparkConfig, DsparkDraft};
use super::hf::{download_drafter, prepare_hf_model, COMBINED_WEIGHTS_FILE};
use super::model::QwenConfig;
use super::shard::Shard;
use super::tok::ChatTokenizer;
use super::EOS_TOKENS;
use crate::engine::TokenId;
use crate::node::{serve_tail, tap_ranges, ChainLink, TappedShards};
use crate::specfeat::{run_feature_decode, FeatureChain};
use crate::transport::{listen_endpoint, TlsIdentity};

#[cfg(feature = "cuda")]
type Backend = super::runtime::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type Backend = super::runtime::MetalRuntime;

const TARGET: &str = "Qwen/Qwen3-4B";
const DRAFTER: &str = "deepseek-ai/dspark_qwen3_4b_block7";
const CFG: QwenConfig = QwenConfig::QWEN3_4B;
const MAX_SEQ: usize = 512;
const SEARCH_GRAPHS: usize = 200;
const MAX_NEW: usize = 64;
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
                SEARCH_GRAPHS,
                chunk,
            )
        })
        .collect();
    TappedShards::new(parts, taps.to_vec()).unwrap()
}

/// Plain greedy decoding through the chain, one token per trip.
fn greedy(chain: &mut impl FeatureChain, prompt: &[TokenId], max_new: usize) -> Vec<TokenId> {
    let (mut cur, _) = chain.prefill_features(prompt).unwrap();
    let mut out = Vec::new();
    while !chain.is_eos(cur) && out.len() < max_new {
        out.push(cur);
        let pos = (prompt.len() + out.len() - 1) as u32;
        cur = chain.verify_features(pos, cur, &[]).unwrap().0[0];
    }
    out
}

#[test]
#[ignore = "needs a GPU and downloads Qwen3-4B and its DSpark drafter"]
fn dspark_loopback() {
    let (config, weights) = download_drafter(DRAFTER).expect("download the drafter");
    let dcfg = DsparkConfig::from_json(&std::fs::read_to_string(config).unwrap()).unwrap();
    dcfg.check_target(&CFG).unwrap();
    let chunk = dcfg.block + 1;
    let cut = CFG.layers / 2;

    // The tail on its own thread, as `potpori serve --taps …` runs it: load, then listen.
    // The head loads only after the tail is up, so the two kernel searches do not share the
    // GPU and skew each other's timings.
    let (addr_tx, addr_rx) = std::sync::mpsc::channel();
    let taps = dcfg.taps.clone();
    let tail = std::thread::spawn(move || {
        let mut shard = load_node((cut, CFG.layers), &taps, chunk);
        let tls = TlsIdentity::for_tests();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let ep = {
            let _g = rt.enter();
            listen_endpoint("127.0.0.1:0".parse().unwrap(), &tls).unwrap()
        };
        addr_tx.send(ep.local_addr().unwrap()).unwrap();
        serve_tail(&mut shard, &ep, &rt).expect("tail serve");
    });
    let tail_addr = addr_rx.recv().expect("the tail loads");

    let head = load_node((0, cut), &dcfg.taps, chunk);
    let mut drafter =
        DsparkDraft::load(Backend::default(), dcfg, &weights, MAX_SEQ, SEARCH_GRAPHS).unwrap();
    let mut chain = ChainLink::connect(
        head,
        tail_addr,
        &TlsIdentity::for_tests(),
        &EOS_TOKENS,
        Duration::from_secs(1800),
    )
    .expect("connect");

    let tok = ChatTokenizer::load(TARGET).unwrap();
    let (mut tokens, mut trips) = (0, 0);
    for prompt in PROMPTS {
        let ids = tok.encode_chat(prompt).unwrap();
        let t0 = Instant::now();
        let reference = greedy(&mut chain, &ids, MAX_NEW);
        let greedy_secs = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let (out, stats) = run_feature_decode(&mut drafter, &mut chain, &ids, MAX_NEW).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "\n{prompt}\n  {} tokens: {:.2} tok/trip over {} trips | {:.1} tok/s with the \
             drafter, {:.1} tok/s one token per trip",
            out.len(),
            stats.toks_per_traversal(),
            stats.traversals,
            out.len() as f64 / secs,
            reference.len() as f64 / greedy_secs,
        );
        println!("  {}", tok.decode(&out).unwrap().replace('\n', " "));
        assert_eq!(out, reference, "the drafter must not change the output");
        tokens += out.len();
        trips += stats.traversals;
    }
    let d = drafter.stats();
    let link = chain.stats();
    println!(
        "\noverall {:.2} tok/trip | per block: drafter graph {:.1} ms, markov {:.1} ms | {:.1} \
         KiB of tapped layers per result",
        tokens as f64 / trips.max(1) as f64,
        d.graph.as_secs_f64() * 1e3 / d.drafts.max(1) as f64,
        d.markov.as_secs_f64() * 1e3 / d.drafts.max(1) as f64,
        link.tap_bytes as f64 / link.frames.max(1) as f64 / 1024.0,
    );
    chain.shutdown();
    tail.join().expect("tail thread");
}

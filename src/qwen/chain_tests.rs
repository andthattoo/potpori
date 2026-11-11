//! Real-model chain tests: the protocol in [`crate::node`] over actual Luminal shards of
//! Qwen3-0.6B on Metal, each node on its own thread with real QUIC in between. All `#[ignore]`
//! (they download weights and run a kernel search); run them with
//!
//!   cargo test --release --features metal -- --ignored <name> --nocapture
//!
//! The same protocol is covered GPU-free, with a synthetic shard, in `crate::node::tests`.

use std::net::SocketAddr;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use luminal::prelude::Runtime as _; // MetalRuntime::initialize is a Runtime-trait method
use luminal_metal::MetalRuntime;

use super::hf::{prepare_hf_model, COMBINED_WEIGHTS_FILE};
use super::model::QwenConfig;
use super::shard::Shard;
use super::target::QwenTarget;
use super::EOS_TOKENS;
use crate::engine::{DraftEngine, TargetChain, TokenId};
use crate::node::{serve_middle, serve_tail, ChainLink, NodeSpec};
use crate::specdecode::SpecConfig;
use crate::specpipe::{run_spec_decode_pipelined, Blocking};
use crate::transport::{listen_endpoint, Hello, PROTO_VERSION};

const MODEL: &str = "Qwen/Qwen3-0.6B";
const CFG: QwenConfig = QwenConfig::QWEN3_0_6B;
const SEARCH_GRAPHS: usize = 200;
const PROMPT: [TokenId; 5] = [40, 1079, 264, 4108, 4108]; // raw token ids (no tokenizer)

fn weights() -> String {
    let dir = prepare_hf_model(MODEL).expect("prepare 0.6B");
    dir.join(COMBINED_WEIGHTS_FILE).to_str().unwrap().to_string()
}

/// Load the shard for `layers`. `chunk` 0 = generic kernels, else the point buckets {1, chunk}.
fn load_shard(layers: (usize, usize), chunk: usize, max_seq: usize) -> Shard<MetalRuntime> {
    let (first, last) = (layers.0 == 0, layers.1 == CFG.layers);
    Shard::load_chunked(MetalRuntime::initialize(()), CFG, layers, first, last, &weights(), max_seq, SEARCH_GRAPHS, chunk)
}

fn load_full(max_seq: usize) -> QwenTarget<MetalRuntime> {
    QwenTarget::load(MetalRuntime::initialize(()), CFG, &weights(), max_seq, SEARCH_GRAPHS)
}

/// Run a remote node (tail if `downstream` is `None`) on its own thread for one session. The
/// address is reported before the slow shard load, so loads overlap.
fn spawn_remote(
    layers: (usize, usize),
    downstream: Option<SocketAddr>,
    chunk: usize,
    max_seq: usize,
) -> (SocketAddr, JoinHandle<()>) {
    let (addr_tx, addr_rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        let ep = {
            let _g = rt.enter();
            listen_endpoint("127.0.0.1:0".parse().unwrap()).unwrap()
        };
        addr_tx.send(ep.local_addr().unwrap()).unwrap();
        let spec = NodeSpec {
            model: MODEL.into(),
            layers,
            is_tail: downstream.is_none(),
            chunk,
            max_seq,
            token: None,
            backend: "metal".into(),
        };
        let mut shard = load_shard(layers, chunk, max_seq);
        match downstream {
            None => serve_tail(&mut shard, &spec, &ep, &rt).expect("tail serve"),
            Some(next) => serve_middle(&mut shard, &spec, &ep, next, &rt).expect("middle serve"),
        }
    });
    (addr_rx.recv().expect("node addr"), handle)
}

/// Bring up the whole chain for `plan` (remote nodes on threads, tail first) and connect the
/// head to it.
fn chain(plan: &[(usize, usize)], chunk: usize, max_seq: usize) -> (ChainLink<Shard<MetalRuntime>>, Vec<JoinHandle<()>>) {
    let mut handles = Vec::new();
    let mut next: Option<SocketAddr> = None;
    for &layers in plan[1..].iter().rev() {
        let (addr, handle) = spawn_remote(layers, next, chunk, max_seq);
        next = Some(addr);
        handles.push(handle);
    }
    let head = load_shard(plan[0], chunk, max_seq);
    let hello = Hello {
        proto: PROTO_VERSION,
        token: None,
        model: MODEL.into(),
        chunk: chunk as u32,
        max_seq: max_seq as u32,
        hidden_dim: CFG.hidden as u32,
        plan: plan.iter().map(|&(a, b)| [a as u32, b as u32]).collect(),
        index: 1,
    };
    let link = ChainLink::connect(head, next.unwrap(), hello, &EOS_TOKENS, Duration::from_secs(900))
        .expect("connect chain");
    (link, handles)
}

fn join_all(handles: Vec<JoinHandle<()>>) {
    for h in handles {
        h.join().expect("node thread");
    }
}

/// Plain incremental greedy decode via the verify primitive (empty draft → 1 token/step).
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

fn report(what: &str, base: &[TokenId], got: &[TokenId]) {
    println!("\nreference = {base:?}\n{what} = {got:?}");
    match (0..base.len().min(got.len())).find(|&i| base[i] != got[i]) {
        None if base.len() == got.len() => println!("LOSSLESS: {what} == reference"),
        Some(i) => println!("diverge @{i}: reference={} {what}={}", base[i], got[i]),
        None => println!("length differ: reference={} {what}={}", base.len(), got.len()),
    }
}

/// Prompt-lookup n-gram draft — exercises accept AND reject (the divergence/overwrite path).
struct NgramDraft;
impl DraftEngine for NgramDraft {
    fn draft(&mut self, context: &[TokenId], gamma: usize) -> Vec<TokenId> {
        let n = context.len();
        if n >= 2 {
            let (a, b) = (context[n - 2], context[n - 1]);
            for i in (0..n - 2).rev() {
                if context[i] == a && context[i + 1] == b {
                    let start = i + 2;
                    let mut d: Vec<TokenId> = (0..gamma).filter_map(|k| context.get(start + k).copied()).collect();
                    if !d.is_empty() {
                        while d.len() < gamma {
                            d.push(*d.last().unwrap());
                        }
                        return d;
                    }
                }
            }
        }
        vec![*context.last().unwrap_or(&0); gamma]
    }
}

/// Perfect draft: returns a known continuation, so the pipeline never flushes.
struct PerfectDraft {
    truth: Vec<TokenId>,
    prompt_len: usize,
}
impl DraftEngine for PerfectDraft {
    fn draft(&mut self, context: &[TokenId], gamma: usize) -> Vec<TokenId> {
        let l = context.len() - self.prompt_len;
        (0..gamma).map(|j| *self.truth.get(l + j).unwrap_or(&0)).collect()
    }
}

// THE 2-NODE PIPELINE: head and tail on separate threads, the hidden state crossing a REAL QUIC
// link, must reproduce the single-process forward token for token.
#[test]
#[ignore]
fn qwen_pipeline_over_quic() {
    let k = CFG.layers / 2;
    let (link, handles) = chain(&[(0, k), (k, CFG.layers)], 0, 512);
    let mut full = load_full(512);
    let base = greedy(&mut full, &PROMPT, 24);

    let mut chain = Blocking(link);
    let distributed = greedy(&mut chain, &PROMPT, 24);
    chain.0.shutdown();
    join_all(handles);

    report("pipeline (2 nodes/QUIC)", &base, &distributed);
    assert_eq!(base, distributed, "the 2-node pipeline must be lossless");
}

// N-NODE CHAIN: head + middle + tail (0.6B split in thirds), the hidden state crossing TWO QUIC
// links, must equal the single-process forward.
#[test]
#[ignore]
fn qwen_pipeline_3node_over_quic() {
    let l = CFG.layers;
    let (a, b) = (l / 3, 2 * l / 3);
    let (link, handles) = chain(&[(0, a), (a, b), (b, l)], 0, 512);
    let mut full = load_full(512);
    let base = greedy(&mut full, &PROMPT, 24);

    let mut chain = Blocking(link);
    let dist = greedy(&mut chain, &PROMPT, 24);
    chain.0.shutdown();
    join_all(handles);

    report("pipeline (3 nodes/QUIC)", &base, &dist);
    assert_eq!(base, dist, "the 3-node pipeline must be lossless");
}

// PIPELINED over QUIC: the depth-N driver over the real 2-node chain must stay lossless vs
// greedy — exercises submit/poll FIFO + overwrite-rollback across divergences.
#[test]
#[ignore]
fn qwen_pipelined_over_quic() {
    let k = CFG.layers / 2;
    let (mut chain, handles) = chain(&[(0, k), (k, CFG.layers)], 0, 512);
    let mut full = load_full(512);
    let base = greedy(&mut full, &PROMPT, 24);

    let spec_cfg = SpecConfig { gamma: 4, max_new: 24 };
    let (out, stats) = run_spec_decode_pipelined(&mut NgramDraft, &mut chain, &PROMPT, &spec_cfg, 4).unwrap();
    chain.shutdown();
    join_all(handles);

    report("pipelined (depth 4, QUIC)", &base, &out);
    println!("toks/traversal = {:.2}", stats.toks_per_traversal());
    assert_eq!(base, out, "the pipelined QUIC chain must be lossless vs greedy");
}

// THROUGHPUT: depth-1 (one chunk in flight) vs depth-4 (pipelined) over the 2-node chain, with
// a perfect draft so the pipeline stays full. On loopback (RTT≈0) the win is compute overlap;
// over real WAN it additionally hides the chain RTT. Same session throughout — each run's
// prefill at position 0 overwrites the KV.
#[test]
#[ignore]
fn qwen_pipeline_throughput() {
    let k = CFG.layers / 2;
    let (mut chain, handles) = chain(&[(0, k), (k, CFG.layers)], 0, 768);
    let mut full = load_full(768);
    let truth = greedy(&mut full, &PROMPT, 500);

    let spec_cfg = SpecConfig { gamma: 6, max_new: truth.len() };
    let mut draft = PerfectDraft { truth: truth.clone(), prompt_len: PROMPT.len() };
    let mut run = |depth| {
        let t0 = Instant::now();
        let (out, st) = run_spec_decode_pipelined(&mut draft, &mut chain, &PROMPT, &spec_cfg, depth).unwrap();
        (out, st, t0.elapsed().as_secs_f64())
    };
    let _ = run(1); // warm-up (first run pays one-time per-shape costs)
    let (out1, st1, dt1) = run(1);
    let (out4, st4, dt4) = run(4);
    chain.shutdown();
    join_all(handles);

    let (tps1, tps4) = (out1.len() as f64 / dt1, out4.len() as f64 / dt4);
    println!("\ndepth-1 (no pipelining): {} tok in {dt1:.3}s = {tps1:.1} tok/s ({} traversals)", out1.len(), st1.traversals);
    println!("depth-4 (pipelined):     {} tok in {dt4:.3}s = {tps4:.1} tok/s ({} traversals)", out4.len(), st4.traversals);
    println!("PIPELINING SPEEDUP: {:.2}x (loopback ⇒ compute-overlap only; WAN adds RTT-hiding)", tps4 / tps1);
    assert_eq!(out1, truth, "depth-1 must be lossless");
    assert_eq!(out4, truth, "depth-4 must be lossless");
}

// DIAGNOSTIC: split each single-token step into head compute (+ send) vs waiting on the rest of
// the chain, at growing context. Localizes the per-traversal cost.
#[test]
#[ignore]
fn qwen_pipeline_step_breakdown() {
    use crate::specpipe::PipelinedChain as _;
    let k = CFG.layers / 2;
    let (mut chain, handles) = chain(&[(0, k), (k, CFG.layers)], 0, 768);

    let mut cur = chain.prefill(&[40]).unwrap();
    println!("\nctx -> head compute ms | waiting ms (single-token greedy, 2-node, avg of 20):");
    let mut mark = chain.stats();
    for pos in 1..=120u32 {
        chain.submit(pos, cur, &[]).unwrap();
        cur = chain.poll().unwrap()[0];
        if pos % 20 == 0 {
            let now = chain.stats();
            let head_ms = (now.head_compute - mark.head_compute).as_secs_f64() * 1000.0 / 20.0;
            let wait_ms = (now.waiting - mark.waiting).as_secs_f64() * 1000.0 / 20.0;
            println!("  ctx≈{pos:>3}: head {head_ms:>6.1} ms | waiting {wait_ms:>7.1} ms");
            mark = now;
        }
    }
    chain.shutdown();
    join_all(handles);
}

// THE METAL FIX, end to end: shards compiled with the SPECIALIZED s=8 bucket + the pipeline.
// 8-token prompt → prefill is s=8; γ=7 → verify is s=8; so every forward rides the fast
// specialized kernel instead of the generic multi-token one (~20× slower on Metal).
#[test]
#[ignore]
fn qwen_n8_pipeline_throughput() {
    let k = CFG.layers / 2;
    let chunk = 8usize; // γ+1 (γ=7)
    let (mut chain, handles) = chain(&[(0, k), (k, CFG.layers)], chunk, 768);
    let prompt: Vec<TokenId> = vec![40, 1079, 264, 4108, 4108, 11, 323, 358]; // 8 → prefill s=8
    let spec_cfg = SpecConfig { gamma: chunk - 1, max_new: 200 };

    // Truth = the chain's OWN greedy continuation (via a draft that exercises rejects). A
    // single-process truth can differ at near-ties: different kernels round differently.
    let (truth, _) = run_spec_decode_pipelined(&mut NgramDraft, &mut chain, &prompt, &spec_cfg, 1).unwrap();
    let mut draft = PerfectDraft { truth: truth.clone(), prompt_len: prompt.len() };
    let mut run = |depth| {
        let t0 = Instant::now();
        let (out, st) = run_spec_decode_pipelined(&mut draft, &mut chain, &prompt, &spec_cfg, depth).unwrap();
        (out, st, t0.elapsed().as_secs_f64())
    };
    let _ = run(1); // warm-up
    let (out1, st1, dt1) = run(1);
    let (out4, st4, dt4) = run(4);
    chain.shutdown();
    join_all(handles);

    let ms_per_trav = dt1 * 1000.0 / st1.traversals.max(1) as f64;
    let (tps1, tps4) = (out1.len() as f64 / dt1, out4.len() as f64 / dt4);
    println!("\n[SPECIALIZED s=8 kernel + matching perfect draft]");
    println!("depth-1: {} tok in {dt1:.3}s = {tps1:.1} tok/s ({} traversals, {ms_per_trav:.0} ms/traversal)", out1.len(), st1.traversals);
    println!("depth-4: {} tok in {dt4:.3}s = {tps4:.1} tok/s ({} traversals)", out4.len(), st4.traversals);
    println!("pipelining {:.2}×", tps4 / tps1);
    assert_eq!(out1, truth, "perfect-draft depth-1 must reproduce the chain's greedy");
    assert_eq!(out4, truth, "perfect-draft depth-4 must reproduce the chain's greedy");
}

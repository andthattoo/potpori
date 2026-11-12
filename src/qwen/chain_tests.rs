//! The chain protocol ([`crate::node`]) over real Luminal shards of Qwen3-0.6B on Metal: each
//! node on its own thread, real QUIC in between. All `#[ignore]` — they download weights and
//! run a kernel search. Run one with
//!
//!   cargo test --release --features metal -- --ignored <name> --nocapture
//!
//! The same protocol is covered without a GPU, using a synthetic shard, in `crate::node`.

use std::net::SocketAddr;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::hf::{prepare_hf_model, COMBINED_WEIGHTS_FILE};
use super::model::QwenConfig;
use super::runtime::MetalRuntime;
use super::shard::Shard;
use super::target::{greedy, QwenTarget};
use super::EOS_TOKENS;
use crate::engine::{DraftEngine, TargetChain, TokenId};
use crate::node::{serve_middle, serve_tail, ChainLink};
use crate::specdecode::SpecConfig;
use crate::specpipe::{run_spec_decode_pipelined, PipelinedChain};
use crate::transport::{listen_endpoint, TlsIdentity};

const CFG: QwenConfig = QwenConfig::QWEN3_0_6B;
const SEARCH_GRAPHS: usize = 200;
const PROMPT: [TokenId; 5] = [40, 1079, 264, 4108, 4108]; // raw token ids (no tokenizer)

type Link = ChainLink<Shard<MetalRuntime>>;

fn weights() -> String {
    let dir = prepare_hf_model("Qwen/Qwen3-0.6B").expect("prepare 0.6B");
    dir.join(COMBINED_WEIGHTS_FILE)
        .to_str()
        .unwrap()
        .to_string()
}

/// Load the shard for `layers`. `chunk` 0 = generic kernels, else kernels for 1 and `chunk`.
fn load_shard(layers: (usize, usize), chunk: usize, max_seq: usize) -> Shard<MetalRuntime> {
    Shard::load_chunked(
        MetalRuntime::default(),
        CFG,
        layers,
        layers.0 == 0,
        layers.1 == CFG.layers,
        &weights(),
        max_seq,
        SEARCH_GRAPHS,
        chunk,
    )
}

/// The whole model in one process, with the same chunk-specialized kernels as the chain.
/// (Generic kernels for any chunk size need a Metal arena sized for a full-cache chunk —
/// hundreds of GiB at this capacity — and are not what `chat` and `serve` run.)
fn load_full(chunk: usize, max_seq: usize) -> QwenTarget<MetalRuntime> {
    QwenTarget::load_chunked(
        MetalRuntime::default(),
        CFG,
        &weights(),
        max_seq,
        SEARCH_GRAPHS,
        chunk,
    )
}

/// Run a remote node (the tail if `downstream` is `None`) on its own thread for one session.
/// The address is reported before the slow shard load, so the nodes load concurrently.
fn spawn_remote(
    layers: (usize, usize),
    downstream: Option<SocketAddr>,
    chunk: usize,
    max_seq: usize,
) -> (SocketAddr, JoinHandle<()>) {
    let (addr_tx, addr_rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
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
        let mut shard = load_shard(layers, chunk, max_seq);
        match downstream {
            None => serve_tail(&mut shard, &ep, &rt).expect("tail serve"),
            Some(next) => serve_middle(&mut shard, &ep, next, &rt, &tls).expect("middle serve"),
        }
    });
    (addr_rx.recv().expect("node addr"), handle)
}

/// Bring up the remote nodes of `plan` (tail first) and connect the head to them.
fn chain(plan: &[(usize, usize)], chunk: usize, max_seq: usize) -> (Link, Vec<JoinHandle<()>>) {
    let mut handles = Vec::new();
    let mut next: Option<SocketAddr> = None;
    for &layers in plan[1..].iter().rev() {
        let (addr, handle) = spawn_remote(layers, next, chunk, max_seq);
        next = Some(addr);
        handles.push(handle);
    }
    let head = load_shard(plan[0], chunk, max_seq);
    let wait = Duration::from_secs(900);
    let link = ChainLink::connect(
        head,
        next.unwrap(),
        &TlsIdentity::for_tests(),
        &EOS_TOKENS,
        wait,
    )
    .expect("connect chain");
    (link, handles)
}

fn finish(mut link: Link, handles: Vec<JoinHandle<()>>) {
    link.shutdown();
    for h in handles {
        h.join().expect("node thread");
    }
}

/// Drive the chain one chunk at a time, so plain greedy decode can run over it.
struct Blocking<'a>(&'a mut Link);
impl TargetChain for Blocking<'_> {
    fn prefill(&mut self, prompt: &[TokenId]) -> TokenId {
        self.0.prefill(prompt).expect("prefill")
    }
    fn verify(&mut self, start_pos: u32, cur: TokenId, drafts: &[TokenId]) -> Vec<TokenId> {
        self.0.submit(start_pos, cur, drafts).expect("submit");
        self.0.poll().expect("poll")
    }
    fn is_eos(&self, token: TokenId) -> bool {
        self.0.is_eos(token)
    }
}

/// Prompt-lookup n-gram draft — produces both accepted and rejected drafts.
struct NgramDraft;
impl DraftEngine for NgramDraft {
    fn draft(&mut self, context: &[TokenId], gamma: usize) -> Vec<TokenId> {
        let n = context.len();
        if n >= 2 {
            let (a, b) = (context[n - 2], context[n - 1]);
            for i in (0..n - 2).rev() {
                if context[i] == a && context[i + 1] == b {
                    let start = i + 2;
                    let mut d: Vec<TokenId> = (0..gamma)
                        .filter_map(|k| context.get(start + k).copied())
                        .collect();
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

/// Returns a known continuation, so the pipeline never has to discard work.
struct PerfectDraft {
    truth: Vec<TokenId>,
    prompt_len: usize,
}
impl DraftEngine for PerfectDraft {
    fn draft(&mut self, context: &[TokenId], gamma: usize) -> Vec<TokenId> {
        let l = context.len() - self.prompt_len;
        (0..gamma)
            .map(|j| *self.truth.get(l + j).unwrap_or(&0))
            .collect()
    }
}

// Two and three nodes, the hidden state crossing real QUIC links: single-token greedy decode
// over the chain must equal the single-process model token for token.
#[test]
#[ignore = "needs Metal and downloads Qwen3-0.6B"]
fn qwen_chain_equals_single_process() {
    let l = CFG.layers;
    let base = greedy(&mut load_full(8, 512), &PROMPT, 24);
    for plan in [
        vec![(0, l / 2), (l / 2, l)],
        vec![(0, l / 3), (l / 3, 2 * l / 3), (2 * l / 3, l)],
    ] {
        let (mut link, handles) = chain(&plan, 8, 512);
        let out = greedy(&mut Blocking(&mut link), &PROMPT, 24);
        finish(link, handles);
        println!(
            "\nsingle process = {base:?}\n{}-node chain   = {out:?}",
            plan.len()
        );
        assert_eq!(base, out, "{}-node chain differs", plan.len());
    }
}

// The pipelined driver over a real 2-node chain, with a draft that is often wrong: output must
// still equal greedy decode (submit/poll ordering and overwrite-rollback on real KV caches).
#[test]
#[ignore = "needs Metal and downloads Qwen3-0.6B"]
fn qwen_pipelined_chain_equals_greedy() {
    let k = CFG.layers / 2;
    let base = greedy(&mut load_full(5, 512), &PROMPT, 24);
    let (mut link, handles) = chain(&[(0, k), (k, CFG.layers)], 5, 512); // chunk = gamma + 1
    let cfg = SpecConfig {
        gamma: 4,
        max_new: 24,
    };
    let (out, stats) = run_spec_decode_pipelined(&mut NgramDraft, &mut link, &PROMPT, &cfg, 4)
        .expect("generation");
    finish(link, handles);
    println!("\ngreedy    = {base:?}\npipelined = {out:?}");
    println!("toks/traversal = {:.2}", stats.toks_per_traversal());
    assert_eq!(base, out, "pipelined output differs from greedy");
}

// Throughput with chunk-specialized kernels (the configuration `potpori chat` uses): depth 1
// vs depth 4 on a 2-node loopback chain, with a perfect draft so the pipeline stays full. On
// loopback the gain is compute overlap only; over a real link it also hides round-trip time.
#[test]
#[ignore = "needs Metal and downloads Qwen3-0.6B"]
fn qwen_pipeline_throughput() {
    let k = CFG.layers / 2;
    let chunk = 8usize; // gamma + 1
    let (mut link, handles) = chain(&[(0, k), (k, CFG.layers)], chunk, 768);
    let prompt: Vec<TokenId> = vec![40, 1079, 264, 4108, 4108, 11, 323, 358]; // one full chunk
    let cfg = SpecConfig {
        gamma: chunk - 1,
        max_new: 200,
    };

    // Truth = the chain's own greedy continuation, found with a draft that is often wrong.
    // (A single-process truth can differ at numerical near-ties between kernels.)
    let (truth, _) = run_spec_decode_pipelined(&mut NgramDraft, &mut link, &prompt, &cfg, 1)
        .expect("reference generation");
    let mut draft = PerfectDraft {
        truth: truth.clone(),
        prompt_len: prompt.len(),
    };
    let mut run = |depth| {
        let before = link.stats();
        let t0 = Instant::now();
        let (out, st) = run_spec_decode_pipelined(&mut draft, &mut link, &prompt, &cfg, depth)
            .expect("generation");
        let after = link.stats();
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "depth {depth}: {} tok in {secs:.2}s = {:.1} tok/s | {} traversals | head {:.2}s, waiting {:.2}s",
            out.len(),
            out.len() as f64 / secs,
            st.traversals,
            (after.head_compute - before.head_compute).as_secs_f64(),
            (after.waiting - before.waiting).as_secs_f64(),
        );
        (out, secs)
    };
    let _ = run(1); // warm-up
    let (out1, secs1) = run(1);
    let (out4, secs4) = run(4);
    finish(link, handles);
    println!("pipelining speedup: {:.2}x", secs1 / secs4);
    assert_eq!(
        out1, truth,
        "depth 1 must reproduce the chain's greedy output"
    );
    assert_eq!(
        out4, truth,
        "depth 4 must reproduce the chain's greedy output"
    );
}

//! Checks for the candle shard ([`super::candle_shard`]).
//!
//! `candle_tiny_shards_and_rollback` runs on the CPU with a generated model, against the same
//! scalar reference as the Luminal backends (`cargo test --features candle candle_tiny`).
//! The rest run on the GPU with `candle-cuda` or `candle-metal` (CPU with plain `candle`) and,
//! but for `candle_tiny_gpu_shards_and_rollback`, load Qwen3-0.6B — all `#[ignore]`:
//!
//!   cargo test --release --features candle-cuda candle_ -- --ignored --nocapture --test-threads 1
//!
//! `POTPORI_CANDLE_DTYPE=bf16` runs them in BF16 instead of F32.
//!
//! `candle_chain_under_latency` puts a delaying UDP relay on every hop, so the chain runs with
//! a real QUIC round trip between nodes on one machine.
use std::net::{SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use candle_core::{DType, Device};

use super::candle_shard::{default_device, default_dtype, CandleShard};
use super::config::QwenConfig;
use super::draft::IncrementalDraft;
use super::greedy;
use super::hf::{prepare_hf_model, COMBINED_WEIGHTS_FILE};
use super::tok::ChatTokenizer;
use super::EOS_TOKENS;
use crate::engine::{DraftEngine, TargetChain, TokenId};
use crate::node::{serve_middle, serve_tail, ChainLink};
use crate::specdecode::SpecConfig;
use crate::specpipe::{run_spec_decode_pipelined, PipelinedChain};
use crate::transport::{listen_endpoint, TlsIdentity};

const CFG: QwenConfig = QwenConfig::QWEN3_0_6B;
const PROMPT: [TokenId; 5] = [40, 1079, 264, 4108, 4108]; // raw token ids (no tokenizer)
/// Luminal CUDA's greedy continuation of PROMPT (`cuda_substrate_check` on an RTX 5080,
/// 2026-11-12): " assistant, and I need to write a response to a user's query. The user".
const LUMINAL_GREEDY: [TokenId; 17] = [
    17847, 11, 323, 358, 1184, 311, 3270, 264, 2033, 311, 264, 1196, 594, 3239, 13, 576, 1196,
];

type Link = ChainLink<CandleShard>;

#[test]
fn candle_tiny_shards_and_rollback() {
    super::tiny::check_shards_and_rollback(|cfg, range, path| {
        CandleShard::load(cfg, range, path, 8, 2, &Device::Cpu, DType::F32).unwrap()
    });
}

// The same tiny-model check on the build's GPU: every operation the shard uses, in seconds
// and with no download. The first thing to run on a new backend.
#[test]
#[ignore = "a GPU with candle-cuda or candle-metal"]
fn candle_tiny_gpu_shards_and_rollback() {
    let device = default_device();
    println!("[candle] tiny model on {:?}", device.location());
    super::tiny::check_shards_and_rollback(|cfg, range, path| {
        CandleShard::load(cfg, range, path, 8, 2, &device, DType::F32).unwrap()
    });
}

/// A model the tests load: its HF repo and geometry.
type Model = (&'static str, QwenConfig);
const SMALL: Model = ("Qwen/Qwen3-0.6B", CFG);

fn load_of(
    (repo, cfg): Model,
    layers: (usize, usize),
    chunk: usize,
    max_seq: usize,
) -> CandleShard {
    let dir = prepare_hf_model(repo).expect("prepare weights");
    let weights = dir.join(COMBINED_WEIGHTS_FILE);
    let t0 = Instant::now();
    let shard = CandleShard::load(
        cfg,
        layers,
        weights.to_str().unwrap(),
        max_seq,
        chunk,
        &default_device(),
        default_dtype(),
    )
    .expect("load candle shard");
    println!(
        "[candle] {repo} layers {layers:?} loaded in {:.2} s",
        t0.elapsed().as_secs_f64()
    );
    shard
}

fn load(layers: (usize, usize), chunk: usize, max_seq: usize) -> CandleShard {
    load_of(SMALL, layers, chunk, max_seq)
}

/// A remote node on its own thread, loading then listening like `potpori serve`.
fn spawn_remote(
    model: Model,
    layers: (usize, usize),
    downstream: Option<Receiver<SocketAddr>>,
    chunk: usize,
    max_seq: usize,
    delay: Option<Arc<AtomicU64>>,
) -> (Receiver<SocketAddr>, JoinHandle<()>) {
    let (addr_tx, addr_rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let mut shard = load_of(model, layers, chunk, max_seq);
        let downstream = downstream.map(|ready| ready.recv().expect("downstream addr"));
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
        let addr = ep.local_addr().unwrap();
        addr_tx
            .send(match delay {
                Some(micros) => delayed(addr, micros),
                None => addr,
            })
            .unwrap();
        match downstream {
            None => serve_tail(&mut shard, &ep, &rt).expect("tail serve"),
            Some(next) => serve_middle(&mut shard, &ep, next, &rt, &tls).expect("middle serve"),
        }
    });
    (addr_rx, handle)
}

/// A UDP relay in front of `target` that holds every datagram, in each direction, for the
/// microseconds in `micros` (read per datagram, so it can change while a session runs).
/// Returns the address to dial instead of `target`. QUIC sees an ordinary slow path: its
/// round trip grows by twice the delay. The relay threads live until the process exits.
fn delayed(target: SocketAddr, micros: Arc<AtomicU64>) -> SocketAddr {
    let front = Arc::new(UdpSocket::bind("127.0.0.1:0").unwrap());
    let back = Arc::new(UdpSocket::bind("127.0.0.1:0").unwrap());
    let addr = front.local_addr().unwrap();
    let client = Arc::new(Mutex::new(None::<SocketAddr>));
    // One direction: a thread stamps each datagram with its release time, a second sends it
    // then. One sender in arrival order keeps the datagrams in order.
    let pump = |from: Arc<UdpSocket>, to: Arc<UdpSocket>, upstream: bool| {
        let (tx, rx) = std::sync::mpsc::channel::<(Instant, Vec<u8>)>();
        let (micros, seen) = (micros.clone(), client.clone());
        std::thread::spawn(move || {
            let mut buf = vec![0u8; 65536];
            loop {
                // Windows reports a peer's closed port as an error on this socket; skip it.
                let Ok((n, src)) = from.recv_from(&mut buf) else {
                    continue;
                };
                if upstream {
                    *seen.lock().unwrap() = Some(src);
                }
                let due = Instant::now() + Duration::from_micros(micros.load(Ordering::Relaxed));
                if tx.send((due, buf[..n].to_vec())).is_err() {
                    return;
                }
            }
        });
        let seen = client.clone();
        std::thread::spawn(move || {
            for (due, datagram) in rx {
                std::thread::sleep(due.saturating_duration_since(Instant::now()));
                let dest = if upstream {
                    Some(target)
                } else {
                    *seen.lock().unwrap()
                };
                if let Some(dest) = dest {
                    let _ = to.send_to(&datagram, dest);
                }
            }
        });
    };
    pump(front.clone(), back.clone(), true);
    pump(back, front, false);
    addr
}

/// Load every node of `plan` and connect the head to the chain over loopback QUIC, through a
/// [`delayed`] relay on every hop when `delay` is given.
fn chain(
    model: Model,
    plan: &[(usize, usize)],
    chunk: usize,
    max_seq: usize,
    delay: Option<Arc<AtomicU64>>,
) -> (Link, Vec<JoinHandle<()>>) {
    let mut handles = Vec::new();
    let mut next: Option<Receiver<SocketAddr>> = None;
    for &layers in plan[1..].iter().rev() {
        let (ready, handle) =
            spawn_remote(model, layers, next.take(), chunk, max_seq, delay.clone());
        next = Some(ready);
        handles.push(handle);
    }
    let head = load_of(model, plan[0], chunk, max_seq);
    let addr = next.unwrap().recv().expect("node addr");
    let link = ChainLink::connect(
        head,
        addr,
        &TlsIdentity::for_tests(),
        &EOS_TOKENS,
        Duration::from_secs(120),
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

/// The known continuation with a fixed, scattered `100 - accuracy` percent of its positions
/// drafted wrong: a draft model of that per-token accuracy, the same at every depth.
struct NoisyDraft {
    truth: Vec<TokenId>,
    prompt_len: usize,
    accuracy: u64,
}
impl DraftEngine for NoisyDraft {
    fn draft(&mut self, context: &[TokenId], gamma: usize) -> Vec<TokenId> {
        let l = context.len() - self.prompt_len;
        (l..l + gamma)
            .map(|i| {
                let right = *self.truth.get(i).unwrap_or(&0);
                // splitmix64: independent-looking rolls, so wrong positions clump and
                // spread as a real draft's do, instead of landing at even intervals.
                let mut roll = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                roll = (roll ^ (roll >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                roll = (roll ^ (roll >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                roll ^= roll >> 31;
                if roll % 100 < self.accuracy {
                    right
                } else {
                    right + 1
                }
            })
            .collect()
    }
}

// The whole model on candle reproduces Luminal CUDA's greedy output token for token.
#[test]
#[ignore = "downloads Qwen3-0.6B; a GPU with candle-cuda"]
fn candle_qwen_greedy_matches_luminal() {
    let mut model = load((0, CFG.layers), 0, 512);
    let out = greedy(&mut model, &PROMPT, LUMINAL_GREEDY.len());
    println!("\ncandle  = {out:?}\nluminal = {LUMINAL_GREEDY:?}");
    assert_eq!(out, LUMINAL_GREEDY, "candle differs from Luminal");
}

// Two and three candle nodes over real QUIC: greedy decode over the chain equals the
// single-process model token for token.
#[test]
#[ignore = "downloads Qwen3-0.6B; a GPU with candle-cuda"]
fn candle_chain_equals_single_process() {
    let l = CFG.layers;
    let base = greedy(&mut load((0, l), 8, 512), &PROMPT, 24);
    for plan in [
        vec![(0, l / 2), (l / 2, l)],
        vec![(0, l / 3), (l / 3, 2 * l / 3), (2 * l / 3, l)],
    ] {
        let (mut link, handles) = chain(SMALL, &plan, 8, 512, None);
        let out = greedy(&mut Blocking(&mut link), &PROMPT, 24);
        finish(link, handles);
        println!(
            "\nsingle process = {base:?}\n{}-node chain   = {out:?}",
            plan.len()
        );
        assert_eq!(base, out, "{}-node chain differs", plan.len());
    }
}

// The pipelined driver over a 2-node candle chain with a draft that is often wrong: output
// still equals greedy decode (submit/poll ordering and overwrite-rollback on real caches).
#[test]
#[ignore = "downloads Qwen3-0.6B; a GPU with candle-cuda"]
fn candle_pipelined_chain_equals_greedy() {
    let k = CFG.layers / 2;
    let base = greedy(&mut load((0, CFG.layers), 5, 512), &PROMPT, 24);
    let (mut link, handles) = chain(SMALL, &[(0, k), (k, CFG.layers)], 5, 512, None); // chunk = gamma + 1
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

// Throughput of a 2-node candle chain over loopback with a perfect draft: depth 1 vs depth 4,
// next to plain single-process greedy decode.
#[test]
#[ignore = "downloads Qwen3-0.6B; a GPU with candle-cuda"]
fn candle_pipeline_throughput() {
    let k = CFG.layers / 2;
    let chunk = 8usize; // gamma + 1
    let prompt: Vec<TokenId> = vec![40, 1079, 264, 4108, 4108, 11, 323, 358];

    let mut single = load((0, CFG.layers), chunk, 768);
    let _ = greedy(&mut single, &prompt, 8); // warm-up
    let t0 = Instant::now();
    let plain = greedy(&mut single, &prompt, 200);
    let plain_secs = t0.elapsed().as_secs_f64();
    println!(
        "single process, plain greedy: {} tok in {plain_secs:.2}s = {:.1} tok/s",
        plain.len(),
        plain.len() as f64 / plain_secs
    );
    drop(single);

    let (mut link, handles) = chain(SMALL, &[(0, k), (k, CFG.layers)], chunk, 768, None);
    let cfg = SpecConfig {
        gamma: chunk - 1,
        max_new: 200,
    };
    // Truth = the chain's own greedy continuation, found with a draft that is often wrong.
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

// The thesis under a real round trip: 2- and 3-node chains with every hop held for 0, 10 and
// 50 ms each way. Plain greedy pays the whole round trip per token; speculation pays it per
// chunk; pipelining should stop paying it. Drafts of 100%, 95% and 80% per-token accuracy, depth
// 1 to 8. Every run must still equal the chain's greedy output. `POTPORI_GAMMA` sets the
// drafted tokens per chunk (default 7), `POTPORI_ONE_WAY_MS` the delays (default `0,10,50`).
#[test]
#[ignore = "downloads Qwen3-0.6B; a GPU with candle-cuda"]
fn candle_chain_under_latency() {
    let l = CFG.layers;
    let gamma: usize = std::env::var("POTPORI_GAMMA")
        .ok()
        .and_then(|g| g.parse().ok())
        .unwrap_or(7);
    let chunk = gamma + 1;
    println!("gamma {gamma}: chunks of {chunk} tokens");
    let delays: Vec<u64> = std::env::var("POTPORI_ONE_WAY_MS")
        .unwrap_or_else(|_| "0,10,50".to_string())
        .split(',')
        .map(|ms| ms.trim().parse().expect("POTPORI_ONE_WAY_MS: milliseconds"))
        .collect();
    let prompt: Vec<TokenId> = vec![40, 1079, 264, 4108, 4108, 11, 323, 358];
    let cfg = SpecConfig {
        gamma: chunk - 1,
        max_new: 200,
    };
    for plan in [
        vec![(0, l / 2), (l / 2, l)],
        vec![(0, l / 3), (l / 3, 2 * l / 3), (2 * l / 3, l)],
    ] {
        let delay = Arc::new(AtomicU64::new(0));
        let (mut link, handles) = chain(SMALL, &plan, chunk, 768, Some(delay.clone()));
        let (truth, _) = run_spec_decode_pipelined(&mut NgramDraft, &mut link, &prompt, &cfg, 1)
            .expect("reference generation");
        println!("\n{}-node chain", plan.len());
        for &one_way_ms in &delays {
            delay.store(one_way_ms * 1000, Ordering::Relaxed);
            // Plain greedy first: it also lets QUIC's round-trip estimate settle.
            let t0 = Instant::now();
            let plain = greedy(&mut Blocking(&mut link), &prompt, 32);
            let plain_rate = plain.len() as f64 / t0.elapsed().as_secs_f64();
            assert!(truth.starts_with(&plain), "plain greedy differs");
            println!(
                "{one_way_ms} ms each way per hop (QUIC rtt to the next node {:.1} ms): plain greedy {plain_rate:.1} tok/s",
                link.rtt().as_secs_f64() * 1e3
            );
            for accuracy in [100, 95, 80] {
                let mut draft = NoisyDraft {
                    truth: truth.clone(),
                    prompt_len: prompt.len(),
                    accuracy,
                };
                let mut run = |depth| {
                    let t0 = Instant::now();
                    let (out, st) =
                        run_spec_decode_pipelined(&mut draft, &mut link, &prompt, &cfg, depth)
                            .expect("generation");
                    assert_eq!(out, truth, "depth {depth} differs from greedy");
                    (out.len() as f64 / t0.elapsed().as_secs_f64(), st.traversals)
                };
                let _ = run(1); // warm-up: opens QUIC's congestion window for full frames
                let mut line = format!("  draft {accuracy:>3}% right:");
                for depth in [1, 2, 4, 8] {
                    let (rate, traversals) = run(depth);
                    line += &format!("  depth {depth} {rate:6.1} tok/s ({traversals} chunks)");
                }
                println!("{line}");
            }
        }
        finish(link, handles);
    }
}

// The real pairing under a round trip: Qwen3-0.6B drafts on the head for a 2-node Qwen3-1.7B
// chain, as `potpori chat` runs it, with every hop held for 0, 10 and 50 ms each way. Three
// chat prompts, depth 1 to 4. Tokens per chunk is the draft's real acceptance; it does not
// depend on timing. Every run must equal plain greedy decode over the same chain.
// `POTPORI_GAMMA` and `POTPORI_ONE_WAY_MS` as in `candle_chain_under_latency`.
#[test]
#[ignore = "downloads Qwen3-0.6B and Qwen3-1.7B; a GPU with candle-cuda"]
fn candle_real_draft_under_latency() {
    const TARGET: Model = ("Qwen/Qwen3-1.7B", QwenConfig::QWEN3_1_7B);
    let gamma: usize = std::env::var("POTPORI_GAMMA")
        .ok()
        .and_then(|g| g.parse().ok())
        .unwrap_or(7);
    let chunk = gamma + 1;
    let delays: Vec<u64> = std::env::var("POTPORI_ONE_WAY_MS")
        .unwrap_or_else(|_| "0,10,50".to_string())
        .split(',')
        .map(|ms| ms.trim().parse().expect("POTPORI_ONE_WAY_MS: milliseconds"))
        .collect();
    let max_new = 96;
    let cfg = SpecConfig { gamma, max_new };
    let tok = ChatTokenizer::load(TARGET.0).expect("tokenizer");
    let prompts: Vec<Vec<TokenId>> = [
        "Explain in a few sentences why the sky is blue.",
        "Write a Python function that returns the n-th Fibonacci number.",
        "Give me three tips for a first trip to Istanbul.",
    ]
    .iter()
    .map(|p| tok.encode_chat(p).expect("encode"))
    .collect();

    let l = TARGET.1.layers;
    let delay = Arc::new(AtomicU64::new(0));
    let (mut link, handles) = chain(
        TARGET,
        &[(0, l / 2), (l / 2, l)],
        chunk,
        768,
        Some(delay.clone()),
    );
    let mut draft = IncrementalDraft::new(load((0, CFG.layers), 0, 768), 0);
    let _ = draft.draft(&prompts[0], gamma); // warm-up
    let truth: Vec<Vec<TokenId>> = prompts
        .iter()
        .map(|p| greedy(&mut Blocking(&mut link), p, max_new))
        .collect();
    println!(
        "\ngamma {gamma}: {} drafts for a 2-node {} chain, {} tokens over {} prompts",
        SMALL.0,
        TARGET.0,
        truth.iter().map(Vec::len).sum::<usize>(),
        prompts.len()
    );
    for &one_way_ms in &delays {
        delay.store(one_way_ms * 1000, Ordering::Relaxed);
        // Plain greedy first: it also lets QUIC's round-trip estimate settle.
        let t0 = Instant::now();
        let plain = greedy(&mut Blocking(&mut link), &prompts[0], 24);
        let plain_rate = plain.len() as f64 / t0.elapsed().as_secs_f64();
        println!(
            "{one_way_ms} ms each way (QUIC rtt {:.1} ms): plain greedy {plain_rate:.1} tok/s",
            link.rtt().as_secs_f64() * 1e3
        );
        for depth in [1, 2, 4] {
            let (mut tokens, mut chunks, mut secs) = (0, 0, 0.);
            for (prompt, want) in prompts.iter().zip(&truth) {
                let t0 = Instant::now();
                let (out, st) =
                    run_spec_decode_pipelined(&mut draft, &mut link, prompt, &cfg, depth)
                        .expect("generation");
                secs += t0.elapsed().as_secs_f64();
                let same = out.iter().zip(want).take_while(|(a, b)| a == b).count();
                assert!(
                    &out == want,
                    "depth {depth} differs from greedy at token {same} of {}",
                    want.len()
                );
                tokens += out.len();
                chunks += st.traversals;
            }
            println!(
                "  depth {depth}: {:6.1} tok/s | {chunks} chunks, {:.2} tokens per chunk",
                tokens as f64 / secs,
                tokens as f64 / chunks as f64
            );
        }
    }
    finish(link, handles);
}

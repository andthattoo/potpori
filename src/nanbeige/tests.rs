//! GPU checks on generated models: seconds to run, no download. All `#[ignore]` because they
//! need an NVIDIA GPU:
//!
//!   cargo test --release -- --ignored --test-threads 1
//!
//! The kernels are compared with the plain-Rust models in [`super::reference`], and with
//! themselves: a token must come out bit-identical whether it is fed alone, inside a chunk,
//! over a rejected draft, or through shards on different nodes.

use std::net::SocketAddr;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::config::{Config, DraftConfig};
use super::dflash::Dflash;
use super::reference::{RefDraft, RefTarget};
use super::shard::Shard;
use super::stream::{LocalChain, Streamed};
use super::weights::{fp4_tensors, write_bf16, write_fp4, write_tensors, Checkpoint, RawTensor};
use super::{argmax, EOS_TOKEN};
use crate::engine::{TargetChain, TokenId};
use crate::gpu::{gpu, Buf};
use crate::node::{serve_middle, serve_tail, tap_ranges, ChainLink, ShardEngine, TappedShards};
use crate::specfeat::{run_feature_decode, FeatureDraft};
use crate::transport::{listen_endpoint, TlsIdentity};

const TINY: Config = Config {
    layers: 4,
    hidden: 256,
    intermediate: 384,
    n_heads: 4,
    n_kv_heads: 2,
    vocab: 320,
    rms_norm_eps: 1e-5,
    rope_theta: 10_000.0,
};
const MAX_SEQ: usize = 96;

fn tiny_draft() -> DraftConfig {
    DraftConfig {
        layers: 2,
        hidden: TINY.hidden,
        intermediate: 320,
        n_heads: 2,
        n_kv_heads: 1,
        vocab: TINY.vocab,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        block: 8,
        taps: vec![0, 2],
        conv_group: 16,
        selector_rank: 8,
        selector_top_k: 4,
        mask_token: 0,
    }
}

/// Deterministic values in `[-1, 1)`.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// Random weights of a plausible scale for each kind of tensor.
fn generate(parameters: &[(String, Vec<usize>)], seed: u64) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut rng = Rng(seed);
    parameters
        .iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let fan_in = *shape.last().unwrap() as f32;
            let values: Vec<f32> = if name.contains("base_kernel") {
                // [side, tap, hidden]: near the identity on tap 0, a smaller tap 1.
                let hidden = shape[2];
                (0..n)
                    .map(|i| {
                        if (i / hidden) % 2 == 0 {
                            1.0 + 0.2 * rng.next()
                        } else {
                            0.4 * rng.next()
                        }
                    })
                    .collect()
            } else if name.contains("norm") {
                (0..n).map(|_| 1.0 + 0.3 * rng.next()).collect()
            } else if name.contains("embed") || name.contains("codebook") {
                (0..n).map(|_| rng.next()).collect()
            } else {
                (0..n).map(|_| 2.0 * rng.next() / fan_in.sqrt()).collect()
            };
            (name.clone(), shape.clone(), values)
        })
        .collect()
}

/// Generated checkpoints for the tiny target and its drafter. Keep the directory alive.
fn checkpoints() -> (tempfile::TempDir, Checkpoint, Checkpoint) {
    checkpoints_in(false)
}

/// As [`checkpoints`]; with `fp4` the target's layer matrices are rounded to 4 bits and stored
/// the way the published 4-bit checkpoint stores them.
fn checkpoints_in(fp4: bool) -> (tempfile::TempDir, Checkpoint, Checkpoint) {
    let dir = tempfile::tempdir().unwrap();
    let (t, d) = (
        dir.path().join("target.safetensors"),
        dir.path().join("draft.safetensors"),
    );
    let target = generate(&TINY.parameters((0, TINY.layers)), 1);
    write_fp4(&t, &target, |name| fp4 && name.ends_with("_proj.weight")).unwrap();
    write_bf16(&d, &generate(&tiny_draft().parameters(), 2)).unwrap();
    let (t, d) = (
        Checkpoint::open(&[t]).unwrap(),
        Checkpoint::open(&[d]).unwrap(),
    );
    (dir, t, d)
}

fn prompt(n: usize) -> Vec<TokenId> {
    let mut rng = Rng(7);
    (0..n)
        .map(|_| ((rng.next() + 1.0) * 0.5 * (TINY.vocab - 1) as f32) as TokenId + 1)
        .map(|t| t.min(TINY.vocab as TokenId - 1))
        .collect()
}

/// Largest difference between `got` and `want`, relative to the largest wanted magnitude.
fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(1e-6f32, |m, x| m.max(x.abs()));
    got.iter()
        .zip(want)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale
}

/// The whole tiny target cut at its tapped layers, on this GPU.
fn tapped(ck: &Checkpoint, range: (usize, usize), taps: &[usize]) -> TappedShards<Shard> {
    let parts = tap_ranges(range, taps)
        .into_iter()
        .map(|r| Shard::load(TINY, r, ck, MAX_SEQ).unwrap())
        .collect();
    TappedShards::new(parts, taps.to_vec()).unwrap()
}

fn greedy<T: TargetChain>(t: &mut T, prompt: &[TokenId], max_new: usize) -> Vec<TokenId> {
    let mut cur = t.prefill(prompt);
    let mut out = Vec::new();
    while out.len() < max_new {
        out.push(cur);
        cur = t.verify((prompt.len() + out.len() - 1) as u32, cur, &[])[0];
    }
    out
}

fn load_draft(dck: &Checkpoint, tck: &Checkpoint, embed: Arc<Buf<u16>>) -> Dflash {
    let lm_head = Arc::new(Buf::from_slice(
        &tck.bf16("lm_head.weight", &[TINY.vocab, TINY.hidden])
            .unwrap(),
    ));
    Dflash::load(tiny_draft(), dck, embed, lm_head, MAX_SEQ).unwrap()
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn target_matches_the_reference_and_is_exact_under_chunking_and_rollback() {
    let (_dir, ck, _) = checkpoints();
    println!("[gpu] {}", gpu().name());
    check_target(&ck);
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn four_bit_target_matches_the_reference_and_is_exact_under_chunking_and_rollback() {
    let (_dir, ck, _) = checkpoints_in(true);
    assert!(ck.is_fp4("model.layers.0.mlp.down_proj.weight"));
    assert_eq!(
        Shard::load(TINY, (1, 2), &ck, MAX_SEQ)
            .unwrap()
            .hello(2)
            .model,
        super::model_id(true)
    );
    check_target(&ck);
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn four_bit_target_cut_into_shards_and_drafted_equals_plain_greedy() {
    let (_dir, tck, dck) = checkpoints_in(true);
    let tokens = prompt(9);
    let mut plain = Shard::load(TINY, (0, TINY.layers), &tck, MAX_SEQ).unwrap();
    let want_logits = plain.forward_tokens(0, &tokens);
    let want = greedy(&mut plain, &tokens, 48);

    let taps = tiny_draft().taps;
    let mut cut = tapped(&tck, (0, TINY.layers), &taps);
    assert!(
        cut.forward_tokens(0, &tokens) == want_logits,
        "cutting the model changed its output"
    );
    let embed = Shard::load(TINY, (0, 1), &tck, MAX_SEQ)
        .unwrap()
        .embedding()
        .unwrap();
    let mut draft = load_draft(&dck, &tck, embed);
    let mut chain = LocalChain::new(cut, &[EOS_TOKEN]);
    let (got, stats) = run_feature_decode(&mut draft, &mut chain, &tokens, 48).unwrap();
    println!(
        "[decode, 4-bit] {} tokens in {} trips",
        got.len(),
        stats.traversals
    );
    assert_eq!(got, want);
}

fn check_target(ck: &Checkpoint) {
    let tokens = prompt(40); // crosses a piece boundary
    let mut whole = Shard::load(TINY, (0, TINY.layers), ck, MAX_SEQ).unwrap();
    let logits = whole.forward_tokens(0, &tokens);
    assert_eq!(logits.len(), tokens.len() * TINY.vocab);

    let mut reference = RefTarget::new(TINY, ck, MAX_SEQ);
    let want: Vec<f32> = tokens
        .iter()
        .enumerate()
        .flat_map(|(pos, &t)| reference.feed(pos, t).0)
        .collect();
    let err = rel_err(&logits, &want);
    println!("[target] logits vs reference: relative error {err:.2e}");
    assert!(err < 1e-4, "logits differ from the reference by {err}");
    for (got, want) in logits.chunks(TINY.vocab).zip(want.chunks(TINY.vocab)) {
        assert_eq!(argmax(got), argmax(want));
    }

    // One token at a time: the same bits.
    let mut single = Shard::load(TINY, (0, TINY.layers), ck, MAX_SEQ).unwrap();
    let one_by_one: Vec<f32> = tokens
        .iter()
        .enumerate()
        .flat_map(|(pos, &t)| single.forward_tokens(pos as u32, &[t]))
        .collect();
    assert!(
        one_by_one == logits,
        "chunked and single-token feeds differ"
    );

    // A rejected draft at positions 20..28, then the real tokens over it: the same bits.
    single.forward_tokens(20, &[3, 1, 4, 1, 5, 9, 2, 6]);
    let again = single.forward_tokens(20, &tokens[20..28]);
    assert!(
        again == logits[20 * TINY.vocab..28 * TINY.vocab],
        "rollback left a trace"
    );
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn shards_cut_at_tapped_layers_equal_the_whole_model() {
    let (_dir, ck, _) = checkpoints();
    let tokens = prompt(12);
    let mut whole = Shard::load(TINY, (0, TINY.layers), &ck, MAX_SEQ).unwrap();
    let want = whole.forward_tokens(0, &tokens);

    let taps = tiny_draft().taps;
    let mut cut = tapped(&ck, (0, TINY.layers), &taps);
    let got = cut.forward_tokens(0, &tokens);
    assert!(got == want, "cutting the model changed its output");

    // The tapped outputs are the residual stream after those layers.
    let got = cut.take_taps();
    let mut reference = RefTarget::new(TINY, &ck, MAX_SEQ);
    let after: Vec<Vec<Vec<f32>>> = tokens
        .iter()
        .enumerate()
        .map(|(pos, &t)| reference.feed(pos, t).1)
        .collect();
    for (tap, layer) in got.iter().zip(&taps) {
        let want: Vec<f32> = after.iter().flat_map(|a| a[*layer].clone()).collect();
        let err = rel_err(tap, &want);
        assert!(
            err < 1e-4,
            "layer {layer} tap differs from the reference by {err}"
        );
    }
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn drafter_matches_the_reference() {
    let (_dir, tck, dck) = checkpoints();
    let dcfg = tiny_draft();
    let (hidden, width) = (TINY.hidden, dcfg.taps.len() * TINY.hidden);
    let tokens = prompt(40);
    let start = tokens.len() - 1; // the last prompt token stands in for the anchor's position

    // Features for every position before the block, from the reference target.
    let mut target = RefTarget::new(TINY, &tck, MAX_SEQ);
    let mut features = Vec::new();
    for (pos, &t) in tokens[..start].iter().enumerate() {
        let after = target.feed(pos, t).1;
        for &layer in &dcfg.taps {
            features.extend_from_slice(&after[layer]);
        }
    }

    let embed_f32 = tck
        .f32("model.embed_tokens.weight", &[TINY.vocab, hidden])
        .unwrap();
    let mut reference = RefDraft::new(dcfg.clone(), &dck, embed_f32, MAX_SEQ);
    for (pos, row) in features.chunks(width).enumerate() {
        reference.observe(pos, row);
    }
    let anchor = tokens[start];
    let want_rows = reference.block(anchor, start);
    let want: Vec<f32> = want_rows.iter().flatten().copied().collect();

    let shard = Shard::load(TINY, (0, 1), &tck, MAX_SEQ).unwrap();
    let mut draft = load_draft(&dck, &tck, shard.embedding().unwrap());
    // Observed in two calls, the second overwriting stale rows first.
    draft.observe(0, &features[..10 * width]).unwrap();
    draft.observe(10, &vec![0.25; 5 * width]).unwrap();
    draft.observe(10, &features[10 * width..]).unwrap();
    let got = draft.block_hidden(anchor, start);
    let err = rel_err(&got, &want);
    println!("[drafter] block hidden states vs reference: relative error {err:.2e}");
    assert!(err < 1e-4, "drafter differs from the reference by {err}");

    // The same block again gives the same bits, and the drafted tokens are what the
    // selector picks from the reference's hidden states.
    assert!(draft.block_hidden(anchor, start) == got);
    let lm_head = tck.f32("lm_head.weight", &[TINY.vocab, hidden]).unwrap();
    let (logits, hp) = reference.slot_outputs(&want_rows, &lm_head);
    let selector = super::dflash::Selector {
        rank: dcfg.selector_rank,
        top_k: dcfg.selector_top_k,
        vocab: dcfg.vocab,
        predecessor: dck
            .bf16(
                "candidate_selector.predecessor_codebook",
                &[dcfg.vocab, dcfg.selector_rank],
            )
            .unwrap(),
        successor: dck
            .bf16(
                "candidate_selector.successor_codebook",
                &[dcfg.vocab, dcfg.selector_rank],
            )
            .unwrap(),
    };
    let want_tokens = selector.pick(anchor, &logits, &hp).unwrap();
    let got_tokens = draft.draft(anchor, start).unwrap();
    println!("[drafter] drafted {got_tokens:?}");
    assert_eq!(got_tokens.len(), dcfg.block - 1);
    assert_eq!(got_tokens, want_tokens);
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn drafted_decoding_equals_plain_greedy() {
    let (_dir, tck, dck) = checkpoints();
    let tokens = prompt(9);
    let mut plain = Shard::load(TINY, (0, TINY.layers), &tck, MAX_SEQ).unwrap();
    let want = greedy(&mut plain, &tokens, 48);

    let taps = tiny_draft().taps;
    let embed = Shard::load(TINY, (0, 1), &tck, MAX_SEQ)
        .unwrap()
        .embedding()
        .unwrap();
    let mut draft = load_draft(&dck, &tck, embed);
    let mut chain = LocalChain::new(tapped(&tck, (0, TINY.layers), &taps), &[EOS_TOKEN]);
    let (got, stats) = run_feature_decode(&mut draft, &mut chain, &tokens, 48).unwrap();
    println!(
        "[decode] {} tokens in {} trips ({:.2} per trip)",
        got.len(),
        stats.traversals,
        stats.toks_per_traversal()
    );
    assert_eq!(got, want);
}

/// A remote node on its own thread, loading then listening like `potpori serve`.
fn spawn_remote(
    ck: Arc<Checkpoint>,
    range: (usize, usize),
    taps: Vec<usize>,
    downstream: Option<Receiver<SocketAddr>>,
) -> Receiver<SocketAddr> {
    let (addr_tx, addr_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut shard = tapped(&ck, range, &taps);
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
        addr_tx.send(ep.local_addr().unwrap()).unwrap();
        match downstream {
            None => serve_tail(&mut shard, &ep, &rt).expect("tail serve"),
            Some(next) => serve_middle(&mut shard, &ep, next, &rt, &tls).expect("middle serve"),
        }
    });
    addr_rx
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn three_node_chain_with_the_drafter_equals_plain_greedy() {
    let (_dir, tck, dck) = checkpoints();
    let tck = Arc::new(tck);
    let tokens = prompt(9);
    let mut plain = Shard::load(TINY, (0, TINY.layers), &tck, MAX_SEQ).unwrap();
    let want = greedy(&mut plain, &tokens, 48);

    // Head [0,1), middle [1,3), tail [3,4); layers 0 and 2 are tapped.
    let taps = tiny_draft().taps;
    let tail = spawn_remote(tck.clone(), (3, 4), taps.clone(), None);
    let middle = spawn_remote(tck.clone(), (1, 3), taps.clone(), Some(tail));
    let head = tapped(&tck, (0, 1), &taps);
    let embed = Shard::load(TINY, (0, 1), &tck, MAX_SEQ)
        .unwrap()
        .embedding()
        .unwrap();
    let mut draft = load_draft(&dck, &tck, embed);
    let mut chain = ChainLink::connect(
        head,
        middle.recv().unwrap(),
        &TlsIdentity::for_tests(),
        &[EOS_TOKEN],
        Duration::from_secs(30),
    )
    .unwrap();
    let (got, stats) = run_feature_decode(&mut draft, &mut chain, &tokens, 48).unwrap();
    chain.shutdown();
    println!(
        "[chain] {} tokens in {} trips over 3 nodes",
        got.len(),
        stats.traversals
    );
    // Remote taps travel as bf16, so the drafter sees slightly different features than in
    // one process; the committed tokens are the target's own either way.
    assert_eq!(got, want);
}

/// Speed of the real model's layers on this GPU, with random weights: no download.
/// `POTPORI_BENCH_LAYERS` sets how many target layers to load (default 2, about 2.4 GB).
#[test]
#[ignore = "needs an NVIDIA GPU with a few GB free; prints timings"]
fn bench_real_size_layers() {
    let cfg = Config::NANBEIGE_4_3_33B;
    let layers: usize = std::env::var("POTPORI_BENCH_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    // A middle shard: no embedding or head.
    let range = (1, 1 + layers);
    for fp4 in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("layers.safetensors");
        if fp4 {
            write_tensors(&path, &random_fp4(&cfg.parameters(range), 3)).unwrap();
        } else {
            write_bf16(&path, &generate(&cfg.parameters(range), 3)).unwrap();
        }
        let ck = Checkpoint::open(&[path]).unwrap();
        let label = if fp4 { "4-bit" } else { "bf16" };
        let (before, total) = gpu().memory();
        let t0 = Instant::now();
        let mut shard = Shard::load(cfg, range, &ck, 1024).unwrap();
        gpu().sync();
        let (free, _) = gpu().memory();
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        println!(
            "[bench {label}] {} | {layers} layers loaded in {:.1} s, {:.2} GiB each with a \
             1024-token cache | {:.1} of {:.1} GiB free",
            gpu().name(),
            t0.elapsed().as_secs_f64(),
            gib(before - free) / layers as f64,
            gib(free),
            gib(total)
        );
        let mut rng = Rng(11);
        for s in [1usize, 8, 32] {
            let hidden: Vec<f32> = (0..s * cfg.hidden).map(|_| rng.next()).collect();
            shard.forward_hidden(0, s, hidden.clone()); // warm up
            let runs = 5;
            let t0 = Instant::now();
            for i in 0..runs {
                shard.forward_hidden((64 + i * s) as u32, s, hidden.clone());
            }
            let per_layer = t0.elapsed().as_secs_f64() * 1e3 / (runs * layers) as f64;
            println!(
                "[bench {label}] {s:>2} tokens: {per_layer:6.2} ms per layer -> {:6.1} ms for \
                 all {} layers",
                per_layer * cfg.layers as f64,
                cfg.layers
            );
        }
    }
}

/// Random 4-bit matrices in the published layout (random nibbles, scales in the range the
/// real checkpoint uses), bf16 for the rest: for timing, without rounding 600M weights.
fn random_fp4(parameters: &[(String, Vec<usize>)], seed: u64) -> Vec<RawTensor> {
    let mut rng = Rng(seed);
    let mut byte = move || (rng.next().to_bits() >> 8) as u8;
    let mut out = Vec::new();
    for (name, shape, values) in generate(
        &parameters
            .iter()
            .filter(|(name, _)| !name.ends_with("_proj.weight"))
            .cloned()
            .collect::<Vec<_>>(),
        seed,
    ) {
        let bytes = values
            .iter()
            .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
            .collect();
        out.push((name, safetensors::Dtype::BF16, shape, bytes));
    }
    for (name, shape) in parameters {
        if name.ends_with("_proj.weight") {
            let (n_out, n_in) = (shape[0], shape[1]);
            let packed = (0..n_out * n_in / 2).map(|_| byte()).collect();
            let scales = (0..n_out * n_in / 16).map(|_| 76 + byte() % 51).collect();
            // A matrix scale that keeps the product near a trained layer's size.
            let global = 2.0 / (n_in as f32).sqrt() / 6.0 / 16.0;
            out.extend(fp4_tensors(name, [n_out, n_in], packed, scales, global));
        }
    }
    out
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn streaming_layer_groups_through_the_gpu_equals_the_whole_model() {
    let (_dir, tck, dck) = checkpoints();
    let tck = Arc::new(tck);
    let tokens = prompt(9);
    let mut plain = Shard::load(TINY, (0, TINY.layers), &tck, MAX_SEQ).unwrap();
    let want_logits = plain.forward_tokens(0, &tokens);
    let want = greedy(&mut plain, &tokens, 24);

    let taps = tiny_draft().taps;
    let embed = Shard::load(TINY, (0, 1), &tck, MAX_SEQ)
        .unwrap()
        .embedding()
        .unwrap();
    let lm_head = Arc::new(Buf::from_slice(
        &tck.bf16("lm_head.weight", &[TINY.vocab, TINY.hidden])
            .unwrap(),
    ));
    // One layer on the device at a time; the caches carry the context between chunks.
    let mut streamed = Streamed::new(TINY, tck.clone(), MAX_SEQ, taps, 1, embed.clone(), lm_head);
    assert!(streamed.forward_tokens(0, &tokens) == want_logits);
    assert_eq!(streamed.take_taps().len(), 2);

    let mut draft = load_draft(&dck, &tck, embed);
    let mut chain = LocalChain::new(streamed, &[EOS_TOKEN]);
    let (got, _) = run_feature_decode(&mut draft, &mut chain, &tokens, 24).unwrap();
    assert_eq!(got, want);
}

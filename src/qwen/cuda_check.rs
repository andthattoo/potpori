//! CUDA integration checks: tiny deterministic sharding/rollback regression and an optional
//! full Qwen benchmark. Native compilation requires NVRTC and CUDA toolkit headers.
//!
//! This module only compiles under `--features cuda` (see mod.rs gate).
//!   cargo test --features cuda cuda_substrate_check -- --ignored --nocapture
//!
//! Checkpoint matrices use BF16 storage with explicit conversion to F32 arithmetic.

use std::time::Instant;

use crate::qwen::runtime::CudaRuntime;

use super::hf::prepare_hf_model;
use super::model::QwenConfig;
use super::target::QwenTarget;
use crate::engine::{TargetChain, TokenId};

/// No download: exercise native compilation, explicit checkpoint casts, shared KV across
/// buckets, shard boundaries, and absolute-position rollback on a tiny deterministic model.
#[test]
#[ignore = "requires a CUDA GPU, NVRTC and cuBLAS"]
fn native_cuda_tiny_shards_and_rollback() {
    super::tiny::shards_and_rollback::<CudaRuntime>();
}

/// Median-ish ms per `verify` of chunk `[cur] + drafts` at `pos`, after a warm-up (which also
/// triggers the per-bucket kernel compile so it's not in the measured window).
fn verify_ms(
    t: &mut QwenTarget<CudaRuntime>,
    pos: u32,
    cur: TokenId,
    drafts: &[TokenId],
    reps: usize,
) -> f64 {
    let _ = t.verify(pos, cur, drafts); // warm-up / NVRTC compile
    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = t.verify(pos, cur, drafts);
    }
    t0.elapsed().as_secs_f64() * 1000.0 / reps as f64
}

#[test]
#[ignore]
fn cuda_substrate_check() {
    let dir = prepare_hf_model("Qwen/Qwen3-0.6B").expect("download/prepare Qwen3-0.6B");
    let weights = dir.join(super::hf::COMBINED_WEIGHTS_FILE);
    // (1) Build + run: if we get past load(), Luminal CUDA + NVRTC works.
    let rt = CudaRuntime::default();
    let mut t = QwenTarget::load(
        rt,
        QwenConfig::QWEN3_0_6B,
        weights.to_str().unwrap(),
        768,
        200,
    );
    println!("\n[CUDA] Luminal CUDA loaded a 0.6B model (NVRTC ok).");

    // (2) Correctness: greedy 16 tokens; eyeball that they're plausible (not all-zero / garbage).
    t.reset();
    let prompt: Vec<TokenId> = vec![40, 1079, 264, 4108, 4108];
    let mut cur = t.prefill(&prompt);
    let mut out = vec![cur];
    for base in (prompt.len() as u32..).take(16) {
        if t.is_eos(cur) {
            break;
        }
        cur = t.verify(base, cur, &[])[0];
        out.push(cur);
    }
    println!("[CUDA] greedy 0.6B: {out:?}");

    // (3) Kernel cliff: build ~100 tokens of context, then time a single-token (s=1) vs an
    // 8-token (s=8) forward to measure the current backend's bucket behavior.
    t.reset();
    let mut c = t.prefill(&[40]);
    let mut pos = 1u32;
    for _ in 0..100 {
        c = t.verify(pos, c, &[])[0];
        pos += 1;
    }
    let ms1 = verify_ms(&mut t, pos, c, &[], 5);
    let drafts = vec![c; 7];
    let ms8 = verify_ms(&mut t, pos, c, &drafts, 5);
    let ratio = ms8 / ms1.max(1e-6);
    println!("[CUDA] generic-bucket forward (ctx~{pos}): s=1 {ms1:.1} ms | s=8 {ms8:.1} ms | ratio {ratio:.1}×");
    if ratio > 3.0 {
        println!("[CUDA] => the generic multi-token kernel is slow here; chunk-specialized kernels (--chunk) matter.");
    } else {
        println!("[CUDA] => the generic multi-token kernel scales normally here.");
    }
}

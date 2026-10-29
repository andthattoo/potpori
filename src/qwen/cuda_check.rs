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

/// Small, independent scalar reference: no Luminal operations or runtime state.
fn reference_logits(
    cfg: QwenConfig,
    weights: &std::collections::HashMap<String, Vec<f32>>,
    tokens: &[TokenId],
) -> Vec<f32> {
    fn project(matrix: &[f32], input: &[f32]) -> Vec<f32> {
        matrix
            .chunks_exact(input.len())
            .map(|row| row.iter().zip(input).map(|(a, b)| a * b).sum())
            .collect()
    }
    fn norm(input: &[f32], scale: &[f32], eps: f32) -> Vec<f32> {
        let rms = (input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32 + eps).sqrt();
        input.iter().zip(scale).map(|(x, w)| x / rms * w).collect()
    }
    fn rotary_heads(mut input: Vec<f32>, scale: &[f32], pos: usize, cfg: QwenConfig) -> Vec<f32> {
        for head in input.chunks_exact_mut(cfg.head_dim) {
            head.copy_from_slice(&norm(head, scale, cfg.rms_norm_eps));
            let half = cfg.head_dim / 2;
            for i in 0..half {
                let angle = pos as f32 / cfg.rope_theta.powf(2. * i as f32 / cfg.head_dim as f32);
                let (sin, cos) = angle.sin_cos();
                let (a, b) = (head[i], head[i + half]);
                head[i] = a * cos - b * sin;
                head[i + half] = b * cos + a * sin;
            }
        }
        input
    }
    let embedding = &weights["model.embed_tokens.weight"];
    let mut x: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| embedding[t as usize * cfg.hidden..(t as usize + 1) * cfg.hidden].to_vec())
        .collect();
    for layer in 0..cfg.layers {
        let w = |name: &str| weights[&format!("model.layers.{layer}.{name}.weight")].as_slice();
        let normalized: Vec<_> = x
            .iter()
            .map(|row| norm(row, w("input_layernorm"), cfg.rms_norm_eps))
            .collect();
        let queries: Vec<_> = normalized
            .iter()
            .enumerate()
            .map(|(p, row)| {
                rotary_heads(
                    project(w("self_attn.q_proj"), row),
                    w("self_attn.q_norm"),
                    p,
                    cfg,
                )
            })
            .collect();
        let keys: Vec<_> = normalized
            .iter()
            .enumerate()
            .map(|(p, row)| {
                rotary_heads(
                    project(w("self_attn.k_proj"), row),
                    w("self_attn.k_norm"),
                    p,
                    cfg,
                )
            })
            .collect();
        let values: Vec<_> = normalized
            .iter()
            .map(|row| project(w("self_attn.v_proj"), row))
            .collect();
        for (pos, residual) in x.iter_mut().enumerate() {
            let mut attention = vec![0.; cfg.q_dim()];
            for head in 0..cfg.n_heads {
                let q_start = head * cfg.head_dim;
                let kv_start = (head / cfg.kv_groups()) * cfg.head_dim;
                let q = &queries[pos][q_start..q_start + cfg.head_dim];
                let scores: Vec<f32> = keys[..=pos]
                    .iter()
                    .map(|k| {
                        q.iter()
                            .zip(&k[kv_start..kv_start + cfg.head_dim])
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                            / (cfg.head_dim as f32).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let probabilities: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f32 = probabilities.iter().sum();
                for (p, probability) in probabilities.iter().enumerate() {
                    for i in 0..cfg.head_dim {
                        attention[q_start + i] += probability / total * values[p][kv_start + i];
                    }
                }
            }
            for (x, a) in residual
                .iter_mut()
                .zip(project(w("self_attn.o_proj"), &attention))
            {
                *x += a;
            }
            let normalized = norm(residual, w("post_attention_layernorm"), cfg.rms_norm_eps);
            let gate = project(w("mlp.gate_proj"), &normalized);
            let up = project(w("mlp.up_proj"), &normalized);
            let activated: Vec<_> = gate
                .iter()
                .zip(up)
                .map(|(g, u)| g / (1. + (-g).exp()) * u)
                .collect();
            for (x, a) in residual
                .iter_mut()
                .zip(project(w("mlp.down_proj"), &activated))
            {
                *x += a;
            }
        }
    }
    x.iter()
        .flat_map(|row| {
            project(
                embedding,
                &norm(row, &weights["model.norm.weight"], cfg.rms_norm_eps),
            )
        })
        .collect()
}

/// No download: exercise native compilation, explicit checkpoint casts, shared KV across
/// buckets, shard boundaries, and absolute-position rollback on a tiny deterministic model.
#[test]
#[ignore = "requires a CUDA GPU, NVRTC and cuBLAS"]
fn native_cuda_tiny_shards_and_rollback() {
    use super::{model::QwenGraph, shard::Shard};
    use safetensors::{tensor::TensorView, Dtype};
    let cfg = QwenConfig {
        layers: 2,
        hidden: 8,
        intermediate: 12,
        head_dim: 4,
        n_heads: 2,
        n_kv_heads: 1,
        vocab_size: 16,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.,
    };
    let graph = QwenGraph::build(cfg, (0, 2), true, true, 8, 2).unwrap();
    let tensors: Vec<_> = graph
        .parameters
        .iter()
        .enumerate()
        .map(|(seed, p)| {
            let values = (0..p.shape.iter().product::<usize>()).map(|i| {
                if p.name.contains("norm") {
                    1.
                } else {
                    ((i + seed * 13) as f32 * 0.37).sin() * 0.1
                }
            });
            let (dtype, bytes) = if p.name.contains("norm") {
                (
                    Dtype::F32,
                    values.flat_map(f32::to_le_bytes).collect::<Vec<_>>(),
                )
            } else {
                (
                    Dtype::BF16,
                    values
                        .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
                        .collect(),
                )
            };
            (p.name.clone(), p.shape.clone(), dtype, bytes)
        })
        .collect();
    let reference_weights = tensors
        .iter()
        .map(|(name, _, dtype, data)| {
            let values = match dtype {
                Dtype::F32 => data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
                    .collect(),
                Dtype::BF16 => data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|v| half::bf16::from_le_bytes(*v).to_f32())
                    .collect(),
                _ => unreachable!(),
            };
            (name.clone(), values)
        })
        .collect();
    let views = tensors.iter().map(|(name, shape, dtype, data)| {
        (
            name.as_str(),
            TensorView::new(*dtype, shape.clone(), data).unwrap(),
        )
    });
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.safetensors");
    std::fs::write(&path, safetensors::serialize(views, None).unwrap()).unwrap();
    let wp = path.to_str().unwrap();
    let mut full = Shard::load_chunked(
        CudaRuntime::new().unwrap(),
        cfg,
        (0, 2),
        true,
        true,
        wp,
        8,
        8,
        2,
    );
    let mut head = Shard::load_chunked(
        CudaRuntime::new().unwrap(),
        cfg,
        (0, 1),
        true,
        false,
        wp,
        8,
        8,
        2,
    );
    let mut tail = Shard::load_chunked(
        CudaRuntime::new().unwrap(),
        cfg,
        (1, 2),
        false,
        true,
        wp,
        8,
        8,
        2,
    );
    let mut rollback = None;
    let mut prefix = Vec::new();
    for (pos, tokens) in [(0, vec![1]), (1, vec![2, 3]), (1, vec![4, 5]), (3, vec![6])] {
        let expected = full.forward_tokens(pos, &tokens);
        prefix.truncate(pos as usize);
        prefix.extend_from_slice(&tokens);
        let reference = reference_logits(cfg, &reference_weights, &prefix);
        let reference = &reference[pos as usize * cfg.vocab_size..];
        assert_eq!(expected.len(), reference.len());
        for (a, b) in expected.iter().zip(reference) {
            assert!(
                a.is_finite() && (a - b).abs() < 2e-4,
                "CPU reference mismatch {a} vs {b}"
            );
        }
        let hidden = head.forward_tokens(pos, &tokens);
        let actual = tail.forward_hidden(pos, tokens.len(), hidden);
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(&expected) {
            assert!(
                a.is_finite() && b.is_finite() && (a - b).abs() < 2e-4,
                "shard mismatch {a} vs {b}"
            );
        }
        if pos == 3 {
            rollback = Some(expected);
        }
    }
    full.reset();
    full.forward_tokens(0, &[1]);
    full.forward_tokens(1, &[4, 5]);
    let clean = full.forward_tokens(3, &[6]);
    for (a, b) in rollback.unwrap().iter().zip(clean) {
        assert!((a - b).abs() < 2e-4, "rollback differs from fresh cache");
    }
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
    let weights = dir.join("model_combined_bf16_v1.safetensors");
    // (1) Build + run: if we get past load(), Luminal CUDA + NVRTC works.
    let rt = CudaRuntime::new().expect("init CUDA runtime (toolkit + driver present?)");
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
        println!("[CUDA] => CLIFF EXISTS on CUDA too → Shard::load_chunked (specialized bucket) is needed in production (already implemented).");
    } else {
        println!("[CUDA] => NO cliff on CUDA → the generic multi-token kernel is fine; the s≥2 slowdown is Metal-only. load_chunked is not needed for the CUDA lane.");
    }
}

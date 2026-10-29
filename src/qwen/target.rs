//! Whole-model target implemented through the same native shard adapter.
pub use super::runtime::QwenRuntime;
use super::{model::QwenConfig, shard::Shard};
use crate::engine::{TargetChain, TokenId};
pub(crate) fn argmax(row: &[f32]) -> TokenId {
    assert!(
        !row.is_empty() && row.iter().all(|x| x.is_finite()),
        "non-finite or empty logits"
    );
    row.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0 as TokenId
}
pub struct QwenTarget<R: QwenRuntime> {
    shard: Shard<R>,
}
impl<R: QwenRuntime + 'static> QwenTarget<R> {
    pub fn load(runtime: R, cfg: QwenConfig, path: &str, capacity: usize, search: usize) -> Self {
        Self::load_chunked(runtime, cfg, path, capacity, search, 0)
    }
    pub fn load_chunked(
        runtime: R,
        cfg: QwenConfig,
        path: &str,
        capacity: usize,
        search: usize,
        chunk: usize,
    ) -> Self {
        Self {
            shard: Shard::load_chunked(
                runtime,
                cfg,
                (0, cfg.layers),
                true,
                true,
                path,
                capacity,
                search,
                chunk,
            ),
        }
    }
    pub fn reset(&mut self) {
        self.shard.reset();
    }
}
impl<R: QwenRuntime + 'static> TargetChain for QwenTarget<R> {
    fn capacity(&self) -> usize {
        self.shard.capacity()
    }
    fn prefill(&mut self, prompt: &[TokenId]) -> TokenId {
        let logits = self.shard.forward_tokens(0, prompt);
        let v = self.shard.vocab_size();
        argmax(&logits[logits.len() - v..])
    }
    fn verify(&mut self, pos: u32, cur: TokenId, drafts: &[TokenId]) -> Vec<TokenId> {
        let chunk: Vec<_> = std::iter::once(cur).chain(drafts.iter().copied()).collect();
        let logits = self.shard.forward_tokens(pos, &chunk);
        logits
            .chunks_exact(self.shard.vocab_size())
            .map(argmax)
            .collect()
    }
    fn truncate(&mut self, _: u32) {}
    fn is_eos(&self, t: TokenId) -> bool {
        t == 151645 || t == 151643
    }
}
#[cfg(all(test, feature = "metal", target_vendor = "apple"))]
mod tests {
    use super::*;
    use crate::engine::DraftEngine;
    use crate::specdecode::{run_spec_decode, SpecConfig};

    /// Prompt-lookup n-gram drafter (no second model) — exercises accept AND reject.
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

    /// Plain incremental greedy decode via the verify primitive (empty draft → 1 token/step).
    fn greedy_baseline(
        t: &mut impl TargetChain,
        prompt: &[TokenId],
        max_new: usize,
    ) -> Vec<TokenId> {
        let mut cur = t.prefill(prompt);
        let mut out = Vec::new();
        let mut base = prompt.len() as u32;
        loop {
            if t.is_eos(cur) || out.len() >= max_new {
                break;
            }
            out.push(cur);
            cur = t.verify(base, cur, &[])[0];
            base += 1;
        }
        out
    }

    // cargo test -p dria-node --features metal qwen_target_spec_decode -- --ignored --nocapture
    #[test]
    #[ignore]
    fn qwen_target_spec_decode() {
        let model_dir = crate::qwen::hf::prepare_hf_model("Qwen/Qwen3-0.6B")
            .expect("download/prepare Qwen3-0.6B");
        let weights = model_dir.join("model_combined_bf16_v1.safetensors");
        let rt = crate::qwen::runtime::MetalRuntime::initialize(());
        let mut target = QwenTarget::load(
            rt,
            QwenConfig::QWEN3_0_6B,
            weights.to_str().unwrap(),
            512,
            200,
        );

        let prompt: Vec<TokenId> = vec![40, 1079, 264, 4108, 4108]; // raw token ids (no tokenizer)
        let max_new = 32;
        let cfg = SpecConfig { gamma: 4, max_new };

        target.reset();
        let baseline = greedy_baseline(&mut target, &prompt, max_new);

        target.reset();
        let mut draft = NgramDraft;
        let (spec, stats) = run_spec_decode(&mut draft, &mut target, &prompt, &cfg);

        println!("\nbaseline = {baseline:?}");
        println!("spec     = {spec:?}");
        println!("toks/traversal = {:.2}", stats.toks_per_traversal());
        match (0..baseline.len().min(spec.len())).find(|&i| baseline[i] != spec[i]) {
            None if baseline.len() == spec.len() => {
                println!("LOSSLESS: spec-decode == greedy baseline on Qwen3-0.6B (Luminal/Metal)")
            }
            Some(i) => println!(
                "DIVERGE @{i}: baseline={} spec={} (likely a bf16 near-tie — cf. M1 substrate finding)",
                baseline[i], spec[i]
            ),
            None => println!("length differ: baseline={} spec={}", baseline.len(), spec.len()),
        }
        assert!(!baseline.is_empty(), "model must generate something");
    }

    /// A real draft model: greedily generate `gamma` tokens from a smaller QwenTarget.
    /// Re-prefills the whole context on each call (O(context) — fine for a g measurement;
    /// the production draft keeps an incremental KV).
    struct QwenDraft<'a, R: QwenRuntime + 'static> {
        model: &'a mut QwenTarget<R>,
    }
    impl<R: QwenRuntime + 'static> DraftEngine for QwenDraft<'_, R> {
        fn draft(&mut self, context: &[TokenId], gamma: usize) -> Vec<TokenId> {
            self.model.reset();
            let mut cur = self.model.prefill(context);
            let mut out = Vec::with_capacity(gamma);
            out.push(cur);
            let mut base = context.len() as u32;
            for _ in 1..gamma {
                cur = self.model.verify(base, cur, &[])[0];
                out.push(cur);
                base += 1;
            }
            out
        }
    }

    // THE REAL ACCEPTANCE NUMBER: Qwen3-0.6B drafts Qwen3-4B, greedy, on Luminal/Metal.
    // toks/traversal = E[accepted]+1 is the speedup proxy the throughput thesis hinges on.
    // cargo test -p dria-node --features metal qwen_real_g_4b -- --ignored --nocapture
    #[test]
    #[ignore]
    fn qwen_real_g_4b() {
        let dir4b = crate::qwen::hf::prepare_hf_model("Qwen/Qwen3-4B").expect("prepare 4B");
        let dir06b = crate::qwen::hf::prepare_hf_model("Qwen/Qwen3-0.6B").expect("prepare 0.6B");
        let w4b = dir4b.join("model_combined_bf16_v1.safetensors");
        let w06b = dir06b.join("model_combined_bf16_v1.safetensors");

        let mut target = QwenTarget::load(
            crate::qwen::runtime::MetalRuntime::initialize(()),
            QwenConfig::QWEN3_4B,
            w4b.to_str().unwrap(),
            512,
            200,
        );
        let mut draft_model = QwenTarget::load(
            crate::qwen::runtime::MetalRuntime::initialize(()),
            QwenConfig::QWEN3_0_6B,
            w06b.to_str().unwrap(),
            512,
            200,
        );

        let prompt: Vec<TokenId> = vec![40, 1079, 264, 4108, 4108]; // raw token ids
        let max_new = 48;
        let spec_cfg = SpecConfig { gamma: 8, max_new };

        target.reset();
        let baseline = greedy_baseline(&mut target, &prompt, max_new);

        target.reset();
        draft_model.reset();
        let mut draft = QwenDraft {
            model: &mut draft_model,
        };
        let (spec, stats) = run_spec_decode(&mut draft, &mut target, &prompt, &spec_cfg);

        let tpt = stats.toks_per_traversal();
        println!("\n4B baseline           = {baseline:?}");
        println!("spec (0.6B drafts 4B) = {spec:?}");
        println!(
            "toks/traversal = {tpt:.2}  (= E[accepted]+1, gamma={})  => mean accepted/round = {:.2}",
            spec_cfg.gamma,
            tpt - 1.0
        );
        match (0..baseline.len().min(spec.len())).find(|&i| baseline[i] != spec[i]) {
            None if baseline.len() == spec.len() => println!("LOSSLESS vs greedy-4B"),
            Some(i) => println!(
                "diverge @{i}: baseline={} spec={} (bf16 near-tie? cf. M1)",
                baseline[i], spec[i]
            ),
            None => println!(
                "length differ: baseline={} spec={}",
                baseline.len(),
                spec.len()
            ),
        }
        assert!(!spec.is_empty(), "spec-decode must produce output");
    }

    // DIAGNOSTIC: per-step decode time vs context, single-process (no QUIC/sharding). For a 0.6B
    // model the MLP/projections dominate and are CONSTANT per token; attention is sub-dominant, so
    // ms/step should be nearly flat. If it grows steeply with context, the model forward itself is
    // the culprit (re-prefill? naive O(context) attention/GQA kernel?), not the pipeline.
    // cargo test -p dria-node --features metal qwen_forward_scaling -- --ignored --nocapture
    #[test]
    #[ignore]
    fn qwen_forward_scaling() {
        use std::time::Instant;
        let dir = crate::qwen::hf::prepare_hf_model("Qwen/Qwen3-0.6B").expect("0.6B");
        let weights = dir.join("model_combined_bf16_v1.safetensors");
        let rt = crate::qwen::runtime::MetalRuntime::initialize(());
        let mut target = QwenTarget::load(
            rt,
            QwenConfig::QWEN3_0_6B,
            weights.to_str().unwrap(),
            768,
            200,
        );

        target.reset();
        let mut cur = target.prefill(&[40]); // 1-token prompt
        let mut pos = 1u32;
        let window = 50u32;
        println!(
            "\ncontext -> ms per single-token decode (avg over {window}), single-process 0.6B:"
        );
        let mut t = Instant::now();
        for step in 1..=600u32 {
            cur = target.verify(pos, cur, &[])[0];
            pos += 1;
            if step % window == 0 {
                println!(
                    "  ctx≈{:>3}: {:>7.1} ms/step",
                    pos,
                    t.elapsed().as_secs_f64() * 1000.0 / window as f64
                );
                t = Instant::now();
            }
        }
    }

    // DIAGNOSTIC: forward time vs CHUNK SIZE s (tokens fed at once) at fixed context, single-
    // process. Spec-decode uses s=γ+1 (multi-token); single-token decode uses s=1. If s≥2 is far
    // slower than s=1, the multi-token kernel (dim bucket [2,max_seq], generic) is the bottleneck
    // vs the specialized [1,1] kernel. cargo test -p dria-node --features metal
    // qwen_forward_vs_chunk_size -- --ignored --nocapture
    #[test]
    #[ignore]
    fn qwen_forward_vs_chunk_size() {
        use std::time::Instant;
        let dir = crate::qwen::hf::prepare_hf_model("Qwen/Qwen3-0.6B").expect("0.6B");
        let weights = dir.join("model_combined_bf16_v1.safetensors");
        let rt = crate::qwen::runtime::MetalRuntime::initialize(());
        let mut target = QwenTarget::load(
            rt,
            QwenConfig::QWEN3_0_6B,
            weights.to_str().unwrap(),
            768,
            200,
        );

        // Build ~100 tokens of context with single-token decodes.
        target.reset();
        let mut cur = target.prefill(&[40]);
        let mut pos = 1u32;
        for _ in 0..100 {
            cur = target.verify(pos, cur, &[])[0];
            pos += 1;
        }
        let base = pos; // fixed past context for all chunk sizes

        println!("\nchunk size s -> ms per forward (fixed ctx ~{base}, single-process 0.6B):");
        for &d in &[0usize, 1, 3, 6, 7, 15] {
            let drafts = vec![cur; d]; // dummy tokens; we only care about timing
            let _ = target.verify(base, cur, &drafts); // warm-up (compile the bucket kernel)
            let reps = 5;
            let t = Instant::now();
            for _ in 0..reps {
                let _ = target.verify(base, cur, &drafts);
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;
            println!("  s={:>2}: {:>8.1} ms/forward", d + 1, ms);
        }
    }
}

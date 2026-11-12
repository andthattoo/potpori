//! One contiguous range of Qwen3 layers on candle — a [`ShardEngine`] beside the Luminal
//! [`super::shard::Shard`], speaking the same wire contract.
//!
//! Prototype. The model is written out here as plain tensor operations: candle's own Qwen3
//! keeps its layers private and its KV cache append-only, and a shard needs a layer range and
//! a cache it can overwrite. There is no compile step and no kernel search: weights load and
//! the shard runs.
//!
//! The KV cache is preallocated for `capacity` positions and written at absolute positions, so
//! a chunk at `pos` overwrites whatever a rejected speculation left there and attends only to
//! `[0, pos + s)` — rollback is the same overwrite Luminal's shard does. Any chunk size runs;
//! `chunk` is only what the shard announces in the handshake, so it can sit in a chain with
//! Luminal nodes compiled for that chunk.
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};

use super::model::QwenConfig;
use crate::engine::{TargetChain, TokenId};
use crate::node::ShardEngine;
use crate::transport::SessionHello;

struct Layer {
    attn_norm: Tensor,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    ffn_norm: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
    /// `[1, n_kv_heads, capacity, head_dim]`, written in place at absolute positions.
    k_cache: Tensor,
    v_cache: Tensor,
}

pub struct CandleShard {
    cfg: QwenConfig,
    range: (usize, usize),
    capacity: usize,
    chunk: usize,
    device: Device,
    dtype: DType,
    embedding: Option<Tensor>,
    layers: Vec<Layer>,
    /// Final norm and the tied output head (the embedding table), on the last shard.
    head: Option<(Tensor, Linear)>,
    /// RoPE tables `[capacity, head_dim / 2]`.
    cos: Tensor,
    sin: Tensor,
}

/// The device a candle shard runs on: the first CUDA GPU when built with `candle-cuda` and
/// one is present, else the CPU.
pub fn default_device() -> Device {
    Device::cuda_if_available(0).unwrap_or(Device::Cpu)
}

impl CandleShard {
    /// Load layers `range` of `cfg` from the safetensors file at `path` (the prepared file or
    /// the checkpoint itself; tensor names are the checkpoint's). `dtype` is the arithmetic
    /// and storage type on `device`: F32 matches the Luminal shards, BF16 halves memory.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        cfg: QwenConfig,
        range: (usize, usize),
        path: &str,
        capacity: usize,
        chunk: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        ensure!(
            range.0 < range.1 && range.1 <= cfg.layers,
            "invalid layer range {range:?} for {} layers",
            cfg.layers
        );
        ensure!(capacity > 0, "capacity must be positive");
        // SAFETY: the weight file is not modified while it is mapped.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device) }
            .with_context(|| format!("map {path}"))?;
        let first = range.0 == 0;
        let last = range.1 == cfg.layers;
        let table = || vb.get((cfg.vocab_size, cfg.hidden), "model.embed_tokens.weight");
        let embedding = if first { Some(table()?) } else { None };
        let head = if last {
            let table = match &embedding {
                Some(t) => t.clone(),
                None => table()?,
            };
            let norm = vb.get(cfg.hidden, "model.norm.weight")?;
            Some((norm, Linear::new(table, None)))
        } else {
            None
        };
        let mut layers = Vec::with_capacity(range.1 - range.0);
        for l in range.0..range.1 {
            let p = vb.pp(format!("model.layers.{l}"));
            let lin = |name: &str, out: usize, inp: usize| -> Result<Linear> {
                Ok(Linear::new(
                    p.get((out, inp), &format!("{name}.weight"))?,
                    None,
                ))
            };
            let vec = |name: &str, n: usize| p.get(n, &format!("{name}.weight"));
            let cache =
                || Tensor::zeros((1, cfg.n_kv_heads, capacity, cfg.head_dim), dtype, device);
            layers.push(Layer {
                attn_norm: vec("input_layernorm", cfg.hidden)?,
                q: lin("self_attn.q_proj", cfg.q_dim(), cfg.hidden)?,
                k: lin("self_attn.k_proj", cfg.kv_dim(), cfg.hidden)?,
                v: lin("self_attn.v_proj", cfg.kv_dim(), cfg.hidden)?,
                o: lin("self_attn.o_proj", cfg.hidden, cfg.q_dim())?,
                q_norm: vec("self_attn.q_norm", cfg.head_dim)?,
                k_norm: vec("self_attn.k_norm", cfg.head_dim)?,
                ffn_norm: vec("post_attention_layernorm", cfg.hidden)?,
                gate: lin("mlp.gate_proj", cfg.intermediate, cfg.hidden)?,
                up: lin("mlp.up_proj", cfg.intermediate, cfg.hidden)?,
                down: lin("mlp.down_proj", cfg.hidden, cfg.intermediate)?,
                k_cache: cache()?,
                v_cache: cache()?,
            });
        }
        let (cos, sin) = rope_tables(cfg, capacity, device, dtype)?;
        Ok(Self {
            cfg,
            range,
            capacity,
            chunk,
            device: device.clone(),
            dtype,
            embedding,
            layers,
            head,
            cos,
            sin,
        })
    }

    /// Run the block on residual rows `x` (`[1, s, hidden]`) at absolute position `pos`;
    /// returns logits on the last shard, else the residual stream, as flat F32.
    fn run(&mut self, pos: usize, x: Tensor) -> Result<Vec<f32>> {
        let cfg = self.cfg;
        let s = x.dim(1)?;
        ensure!(
            pos + s <= self.capacity,
            "positions {pos}..{} exceed capacity {}",
            pos + s,
            self.capacity
        );
        let eps = cfg.rms_norm_eps;
        let groups = cfg.n_heads / cfg.n_kv_heads;
        let total = pos + s;
        let cos = self.cos.narrow(0, pos, s)?;
        let sin = self.sin.narrow(0, pos, s)?;
        // Additive causal mask over the visible cache, `[s, total]`; a lone row sees all.
        let mask = if s > 1 {
            let rows: Vec<f32> = (0..s)
                .flat_map(|i| {
                    (0..total).map(move |j| if j > pos + i { f32::NEG_INFINITY } else { 0. })
                })
                .collect();
            Some(Tensor::from_vec(rows, (s, total), &self.device)?.to_dtype(self.dtype)?)
        } else {
            None
        };
        let scale = 1. / (cfg.head_dim as f64).sqrt();
        let mut x = x;
        for layer in &mut self.layers {
            let h = candle_nn::ops::rms_norm(&x, &layer.attn_norm, eps)?;
            let heads = |t: Tensor, n: usize, norm: &Tensor| -> Result<Tensor> {
                let t = t
                    .reshape((1, s, n, cfg.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
                let t = candle_nn::ops::rms_norm(&t, norm, eps)?;
                Ok(candle_nn::rotary_emb::rope(&t, &cos, &sin)?)
            };
            let q = heads(layer.q.forward(&h)?, cfg.n_heads, &layer.q_norm)?;
            let k = heads(layer.k.forward(&h)?, cfg.n_kv_heads, &layer.k_norm)?;
            let v = layer
                .v
                .forward(&h)?
                .reshape((1, s, cfg.n_kv_heads, cfg.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            layer.k_cache.slice_set(&k, 2, pos)?;
            layer.v_cache.slice_set(&v, 2, pos)?;
            let keys = layer.k_cache.narrow(2, 0, total)?;
            let values = layer.v_cache.narrow(2, 0, total)?;
            // Grouped-query attention without copying KV heads: the query heads that share a
            // KV head become extra rows against it.
            let q = q.reshape((1, cfg.n_kv_heads, groups * s, cfg.head_dim))?;
            let scores = (q.matmul(&keys.t()?)? * scale)?;
            let scores = match &mask {
                Some(m) => scores
                    .reshape((1, cfg.n_kv_heads, groups, s, total))?
                    .broadcast_add(m)?
                    .reshape((1, cfg.n_kv_heads, groups * s, total))?,
                None => scores,
            };
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let attn = probs
                .matmul(&values)?
                .reshape((1, cfg.n_heads, s, cfg.head_dim))?
                .transpose(1, 2)?
                .reshape((1, s, cfg.q_dim()))?;
            x = (x + layer.o.forward(&attn)?)?;
            let h = candle_nn::ops::rms_norm(&x, &layer.ffn_norm, eps)?;
            let mlp = (candle_nn::ops::silu(&layer.gate.forward(&h)?)? * layer.up.forward(&h)?)?;
            x = (x + layer.down.forward(&mlp)?)?;
        }
        let out = match &self.head {
            Some((norm, head)) => head.forward(&candle_nn::ops::rms_norm(&x, norm, eps)?)?,
            None => x,
        };
        Ok(out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
    }

    fn embed(&self, tokens: &[TokenId]) -> Result<Tensor> {
        let table = self.embedding.as_ref().context("not the first shard")?;
        let ids: Vec<u32> = tokens
            .iter()
            .map(|&t| u32::try_from(t).context("negative token id"))
            .collect::<Result<_>>()?;
        let ids = Tensor::new(ids.as_slice(), &self.device)?;
        Ok(table.index_select(&ids, 0)?.unsqueeze(0)?)
    }
}

/// Qwen's split-half RoPE: angle `pos / theta^(2i / head_dim)` for `i < head_dim / 2`.
fn rope_tables(
    cfg: QwenConfig,
    capacity: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half = cfg.head_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1. / cfg.rope_theta.powf(2. * i as f32 / cfg.head_dim as f32))
        .collect();
    let inv = Tensor::from_vec(inv, (1, half), device)?;
    let positions = Tensor::arange(0u32, capacity as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((capacity, 1))?;
    let angles = positions.broadcast_mul(&inv)?;
    Ok((
        angles.cos()?.to_dtype(dtype)?,
        angles.sin()?.to_dtype(dtype)?,
    ))
}

impl ShardEngine for CandleShard {
    fn hello(&self, next_layer: usize) -> SessionHello {
        let c = self.cfg;
        SessionHello {
            model: format!(
                "qwen3:{}:{}:{}:{}:{}:{}",
                c.layers, c.hidden, c.intermediate, c.head_dim, c.n_heads, c.n_kv_heads
            ),
            next_layer,
            max_seq: self.capacity,
            chunk: self.chunk,
            taps: Vec::new(),
        }
    }
    fn range(&self) -> (usize, usize) {
        self.range
    }
    fn model_layers(&self) -> usize {
        self.cfg.layers
    }
    fn hidden_dim(&self) -> usize {
        self.cfg.hidden
    }
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn chunk_size(&self) -> usize {
        self.chunk
    }
    fn forward_tokens(&mut self, pos: u32, tokens: &[TokenId]) -> Vec<f32> {
        let x = self.embed(tokens).expect("embed tokens");
        self.run(pos as usize, x).expect("run candle shard")
    }
    fn forward_hidden(&mut self, pos: u32, s: usize, hidden: Vec<f32>) -> Vec<f32> {
        let x = Tensor::from_vec(hidden, (1, s, self.cfg.hidden), &self.device)
            .and_then(|t| t.to_dtype(self.dtype))
            .expect("residual stream");
        self.run(pos as usize, x).expect("run candle shard")
    }
    fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.k_cache = layer.k_cache.zeros_like().expect("reset KV");
            layer.v_cache = layer.v_cache.zeros_like().expect("reset KV");
        }
    }
}

/// The whole model in one candle shard, as a [`TargetChain`]: the single-process baseline.
/// Any chunk size runs, so the prompt goes in one forward.
impl TargetChain for CandleShard {
    fn capacity(&self) -> usize {
        self.capacity
    }
    fn prefill(&mut self, prompt: &[TokenId]) -> TokenId {
        assert!(
            self.embedding.is_some() && self.head.is_some(),
            "prefill needs the whole model"
        );
        let logits = self.forward_tokens(0, prompt);
        super::target::argmax(&logits[logits.len() - self.cfg.vocab_size..])
    }
    fn verify(&mut self, pos: u32, cur: TokenId, drafts: &[TokenId]) -> Vec<TokenId> {
        let chunk: Vec<_> = std::iter::once(cur).chain(drafts.iter().copied()).collect();
        self.forward_tokens(pos, &chunk)
            .chunks_exact(self.cfg.vocab_size)
            .map(super::target::argmax)
            .collect()
    }
    fn is_eos(&self, token: TokenId) -> bool {
        super::is_eos(token)
    }
}

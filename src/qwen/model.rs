//! Qwen3 graph on Luminal's native logical IR. Storage and execution are backend-owned.
use anyhow::{ensure, Result};
use luminal::bucketing::BucketSpec;
use luminal::prelude::*;
use luminal::shape::SymbolBounds;
use luminal_nn::{AttentionGeometry, CacheAccess, KvCache};

/// Qwen3 model configuration. Parameterized so the small draft (0.6B) and the bigger targets
/// (1.7B, 4B) load from this one module.
#[derive(Clone, Copy, Debug)]
pub struct QwenConfig {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl QwenConfig {
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn kv_groups(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    pub const QWEN3_0_6B: Self = Self {
        layers: 28,
        hidden: 1024,
        intermediate: 3072,
        head_dim: 128,
        n_heads: 16,
        n_kv_heads: 8,
        vocab_size: 151936,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
    pub const QWEN3_1_7B: Self = Self {
        layers: 28,
        hidden: 2048,
        intermediate: 6144,
        head_dim: 128,
        n_heads: 16,
        n_kv_heads: 8,
        vocab_size: 151936,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
    pub const QWEN3_4B: Self = Self {
        layers: 36,
        hidden: 2560,
        intermediate: 9728,
        head_dim: 128,
        n_heads: 32,
        n_kv_heads: 8,
        vocab_size: 151936,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
    };
}

#[derive(Clone, Debug)]
pub enum TensorData {
    F32(Vec<f32>),
    BF16(Vec<u16>),
    F16(Vec<u16>),
    I32(Vec<i32>),
}
impl TensorData {
    pub fn zeros(dtype: DType, n: usize) -> Result<Self> {
        ensure!(dtype == DType::F32, "KV arithmetic must be F32");
        Ok(Self::F32(vec![0.; n]))
    }
}
pub type Inputs = FxHashMap<NodeIndex, TensorData>;
pub struct Parameter {
    pub input: NodeIndex,
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
}
#[derive(Clone)]
pub struct StateBinding {
    pub input: NodeIndex,
    pub output: NodeIndex,
    pub elements: usize,
    pub dtype: DType,
}

pub struct QwenGraph {
    pub graph: Graph,
    pub parameters: Vec<Parameter>,
    pub state: Vec<StateBinding>,
    pub logits: NodeIndex,
    pub capacity: usize,
    pub chunk_size: usize,
    pub cfg: QwenConfig,
    pub first: bool,
    pub(super) input: NodeIndex,
    pub(super) positions: NodeIndex,
    pub(super) gather: NodeIndex,
    pub(super) cos: NodeIndex,
    pub(super) sin: NodeIndex,
    pub(super) rotation: NodeIndex,
    /// Set for a DSpark drafter graph ([`super::dspark`]), which has more step inputs and
    /// runs a fixed number of rows.
    pub(super) drafter: Option<super::dspark::DrafterPorts>,
}

/// How checkpoint matrices are stored on the device. Arithmetic is F32 either way; a
/// narrower type halves the memory but costs a cast of every matrix on every call.
///
/// CUDA keeps the checkpoint's bf16. Metal has no bf16 buffers, and there the per-call cast
/// dominates a shard's time, so Metal stores f32. `POTPORI_WEIGHT_DTYPE` = `f32`, `f16` or
/// `bf16` overrides the choice (f16 is exact for bf16 weights except below about 6e-5).
pub fn matrix_dtype() -> DType {
    match std::env::var("POTPORI_WEIGHT_DTYPE").as_deref() {
        Ok("f32") => return DType::F32,
        Ok("f16") => return DType::F16,
        Ok("bf16") => return DType::Bf16,
        _ => {}
    }
    if cfg!(all(
        feature = "metal",
        target_os = "macos",
        not(feature = "cuda")
    )) {
        DType::F32
    } else {
        DType::Bf16
    }
}

/// A checkpoint tensor, in the checkpoint's orientation.
pub(super) fn weight(
    cx: &mut Graph,
    params: &mut Vec<Parameter>,
    name: String,
    shape: Vec<usize>,
) -> GraphTensor {
    // F32 arithmetic, with the matrix's device storage declared explicitly.
    let port = cx.named_tensor(&name, shape.clone(), DType::F32);
    let dtype = if name.contains("norm") {
        DType::F32
    } else {
        matrix_dtype()
    };
    let source = cx.input_source(port.id, dtype);
    params.push(Parameter {
        input: source.id,
        name,
        shape,
        dtype,
    });
    port
}

/// A linear layer `x · Wᵀ` for rows `x` and a checkpoint matrix `W` of `[out, in]` — also the
/// tied output head, with the embedding table as `W`.
///
/// Written as `(W · xᵀ)ᵀ` on purpose. Luminal's kernels run one thread per output element;
/// in this form a thread reads one contiguous row of `W`, and the threads for the rows of a
/// chunk sit next to each other and read the same one. The textbook `x.matmul(W.t())` makes
/// every row of the chunk walk the whole matrix again: on Metal the output head of a
/// chunk of eight took six times as long.
pub(super) fn linear(x: GraphTensor, w: GraphTensor) -> GraphTensor {
    linear_columns(x, w).t()
}

/// [`linear`] without the final turn: `[out, rows]`. A graph's output has to be a buffer of
/// its own, and turning it there would cost a copy kernel, so graphs that end in a linear
/// layer (the output head) return this and the host turns it with [`rows_from_columns`].
pub(super) fn linear_columns(x: GraphTensor, w: GraphTensor) -> GraphTensor {
    w.matmul(x.t())
}

/// A `[columns, rows]` row-major matrix as `[rows, columns]`.
pub fn rows_from_columns(data: &[f32], rows: usize) -> Vec<f32> {
    let columns = data.len() / rows;
    let mut out = vec![0f32; data.len()];
    for (c, column) in data.chunks_exact(rows).enumerate() {
        for (r, &value) in column.iter().enumerate() {
            out[r * columns + c] = value;
        }
    }
    out
}

impl QwenGraph {
    pub fn build(
        cfg: QwenConfig,
        range: (usize, usize),
        first: bool,
        last: bool,
        capacity: usize,
        chunk_size: usize,
    ) -> Result<Self> {
        ensure!(
            range.0 < range.1 && range.1 <= cfg.layers,
            "invalid shard layer range"
        );
        ensure!(
            first == (range.0 == 0) && last == (range.1 == cfg.layers),
            "shard role does not match layer range"
        );
        ensure!(
            capacity > 0 && capacity <= i32::MAX as usize && chunk_size <= capacity,
            "invalid cache/chunk capacity"
        );
        let mut cx = Graph::new();
        let input = if first {
            cx.tensor(vec!['q'], DType::Int)
        } else {
            cx.tensor(vec![IntExpr::from('q'), cfg.hidden.into()], DType::F32)
        };
        let positions = cx.tensor(vec!['q'], DType::Int);
        let gather = cx.tensor(vec!['c'], DType::Int);
        let cos = cx.tensor(vec![IntExpr::from('q'), cfg.head_dim.into()], DType::F32);
        let sin = cx.tensor(vec![IntExpr::from('q'), cfg.head_dim.into()], DType::F32);
        let rotation = cx.tensor(vec![cfg.head_dim, cfg.head_dim], DType::F32);
        let mut parameters = Vec::new();
        // The embedding table is also the output head.
        let embedding = (first || last).then(|| {
            weight(
                &mut cx,
                &mut parameters,
                "model.embed_tokens.weight".into(),
                vec![cfg.vocab_size, cfg.hidden],
            )
        });
        let mut x = if first {
            luminal_nn::embedding(input, embedding.unwrap())
        } else {
            input
        };
        let mut state = Vec::new();
        for l in range.0..range.1 {
            let mut w = |suffix: &str, shape| {
                weight(
                    &mut cx,
                    &mut parameters,
                    format!("model.layers.{l}.{suffix}.weight"),
                    shape,
                )
            };
            let attn_norm = w("input_layernorm", vec![cfg.hidden]);
            let wq = w("self_attn.q_proj", vec![cfg.q_dim(), cfg.hidden]);
            let wk = w("self_attn.k_proj", vec![cfg.kv_dim(), cfg.hidden]);
            let wv = w("self_attn.v_proj", vec![cfg.kv_dim(), cfg.hidden]);
            let wo = w("self_attn.o_proj", vec![cfg.hidden, cfg.q_dim()]);
            let qn = w("self_attn.q_norm", vec![cfg.head_dim]);
            let kn = w("self_attn.k_norm", vec![cfg.head_dim]);
            let ffn_norm = w("post_attention_layernorm", vec![cfg.hidden]);
            let gate = w("mlp.gate_proj", vec![cfg.intermediate, cfg.hidden]);
            let up = w("mlp.up_proj", vec![cfg.intermediate, cfg.hidden]);
            let down = w("mlp.down_proj", vec![cfg.hidden, cfg.intermediate]);
            let k_cache = cx.tensor(vec![capacity, cfg.kv_dim()], DType::F32);
            let v_cache = cx.tensor(vec![capacity, cfg.kv_dim()], DType::F32);
            let normed = luminal_nn::rms_norm(x, Some(attn_norm), None, cfg.rms_norm_eps);
            let q = luminal_nn::rotary_apply(
                luminal_nn::rms_norm_heads(linear(normed, wq), cfg.head_dim, qn, cfg.rms_norm_eps),
                cfg.head_dim,
                cos,
                sin,
                rotation,
            );
            let k = luminal_nn::rotary_apply(
                luminal_nn::rms_norm_heads(linear(normed, wk), cfg.head_dim, kn, cfg.rms_norm_eps),
                cfg.head_dim,
                cos,
                sin,
                rotation,
            );
            let attention = luminal_nn::paged_attention(
                q,
                k,
                linear(normed, wv),
                KvCache::new(k_cache, v_cache),
                CacheAccess::new(positions, gather),
                luminal_nn::causal_bias(positions, gather),
                AttentionGeometry::new(cfg.n_heads, cfg.n_kv_heads, cfg.head_dim),
            );
            state.push(StateBinding {
                input: k_cache.id,
                output: attention.cache.keys.id,
                elements: capacity * cfg.kv_dim(),
                dtype: DType::F32,
            });
            state.push(StateBinding {
                input: v_cache.id,
                output: attention.cache.values.id,
                elements: capacity * cfg.kv_dim(),
                dtype: DType::F32,
            });
            x += linear(attention.output, wo);
            let normed = luminal_nn::rms_norm(x, Some(ffn_norm), None, cfg.rms_norm_eps);
            x += linear(linear(normed, gate).silu() * linear(normed, up), down);
        }
        if last {
            let norm = weight(
                &mut cx,
                &mut parameters,
                "model.norm.weight".into(),
                vec![cfg.hidden],
            );
            // Logits leave the graph as `[vocab, q]`; the shard turns them.
            x = linear_columns(
                luminal_nn::rms_norm(x, Some(norm), None, cfg.rms_norm_eps),
                embedding.unwrap(),
            );
        }
        Ok(Self {
            graph: cx,
            parameters,
            state,
            logits: x.id,
            capacity,
            chunk_size,
            cfg,
            first,
            input: input.id,
            positions: positions.id,
            gather: gather.id,
            cos: cos.id,
            sin: sin.id,
            rotation: rotation.id,
            drafter: None,
        })
    }
    pub fn rope_matrices(&self) -> Vec<NodeIndex> {
        vec![self.rotation]
    }
    pub fn step_input_ids(&self) -> Vec<NodeIndex> {
        let mut ids = vec![self.input, self.positions, self.gather, self.cos, self.sin];
        if let Some(ports) = &self.drafter {
            ids.extend(ports.ids());
        }
        ids
    }
    /// Representative step inputs for kernel search: `q` rows with `c` positions visible.
    pub fn profile_inputs(&self, q: usize, c: usize) -> Result<Inputs> {
        match &self.drafter {
            Some(_) => super::dspark::profile_inputs(self, c),
            None => self.step_inputs(self.profile_input(q), q, c - q),
        }
    }
    pub fn initial_inputs(&self) -> Inputs {
        let mut data = Inputs::default();
        data.insert(
            self.rotation,
            TensorData::F32(luminal_nn::rope_pairing_matrix(self.cfg.head_dim, false)),
        );
        for state in &self.state {
            data.insert(state.input, TensorData::F32(vec![0.; state.elements]));
        }
        data
    }
    pub fn profile_input(&self, q: usize) -> TensorData {
        if self.first {
            TensorData::I32(vec![1; q])
        } else {
            TensorData::F32(vec![0.02; q * self.cfg.hidden])
        }
    }
    pub fn step_inputs(&self, input: TensorData, q: usize, pos: usize) -> Result<Inputs> {
        let end = pos
            .checked_add(q)
            .ok_or_else(|| anyhow::anyhow!("position overflow"))?;
        ensure!(q > 0 && end <= self.capacity, "chunk exceeds KV capacity");
        ensure!(
            self.chunk_size == 0 || q == 1 || q == self.chunk_size,
            "unsupported verify chunk size"
        );
        match &input {
            TensorData::I32(v) => ensure!(
                self.first
                    && v.len() == q
                    && v.iter()
                        .all(|&t| t >= 0 && (t as usize) < self.cfg.vocab_size),
                "invalid token input"
            ),
            TensorData::F32(v) => ensure!(
                !self.first && v.len() == q * self.cfg.hidden && v.iter().all(|t| t.is_finite()),
                "invalid hidden input"
            ),
            _ => anyhow::bail!("invalid step input dtype"),
        }
        let positions: Vec<i32> = (pos..end).map(|p| p as i32).collect();
        let (cos, sin) = luminal_nn::rope_tables_split_half(
            &positions.iter().map(|&p| p as f32).collect::<Vec<_>>(),
            self.cfg.head_dim,
            self.cfg.rope_theta,
            1.,
        );
        Ok([
            (self.input, input),
            (self.positions, TensorData::I32(positions)),
            (self.gather, TensorData::I32((0..end as i32).collect())),
            (self.cos, TensorData::F32(cos)),
            (self.sin, TensorData::F32(sin)),
        ]
        .into_iter()
        .collect())
    }
}

/// What the kernel search chose for one compiled bucket (kept by the adapters as a report).
#[allow(dead_code)]
pub struct BucketPlan<P, O> {
    pub ranges: std::collections::BTreeMap<luminal::shape::Symbol, (usize, usize)>,
    pub representative: luminal::shape::DynMap,
    pub plan: P,
    pub outcome: O,
    pub finalist_rank: usize,
    pub slab_bytes: usize,
}
pub fn buckets(graph: &QwenGraph) -> Result<Vec<BucketSpec>> {
    let mut qs = vec![(1, 1, 1)];
    if graph.drafter.is_some() {
        // A drafter always runs the same number of rows.
        qs = vec![(graph.chunk_size, graph.chunk_size, graph.chunk_size)];
    } else if graph.chunk_size == 0 && graph.capacity > 1 {
        qs.push((2, graph.capacity, graph.capacity.min(8)));
    } else if graph.chunk_size > 1 {
        qs.push((graph.chunk_size, graph.chunk_size, graph.chunk_size));
    }
    qs.into_iter()
        .map(|(lo, hi, q)| {
            BucketSpec::new(
                SymbolBounds::from_ranges([
                    ('q'.into(), (lo, hi)),
                    ('c'.into(), (1, graph.capacity)),
                ])?,
                [
                    ('q'.into(), q),
                    ('c'.into(), graph.capacity.min(128).max(q)),
                ]
                .into_iter()
                .collect(),
            )
        })
        .collect()
}

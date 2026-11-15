//! Model geometry: Nanbeige4.3-33B and its DFlash2 drafter.

/// The kernels are written for this attention head size.
pub const HEAD_DIM: usize = 128;

/// A Llama-style decoder: pre-norm RMSNorm, grouped-query attention with rotary positions,
/// SwiGLU MLP, untied embedding and output head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
}

/// Which published weights of the target to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Weights {
    /// Layer matrices in 4 bits (NVFP4): 22 GB, about 0.33 GB per layer on the GPU.
    Nvfp4,
    /// The original bf16: 66 GB, about 1.2 GB per layer.
    Bf16,
}

impl Weights {
    pub fn repo(self) -> &'static str {
        match self {
            Weights::Nvfp4 => super::TARGET_REPO_FP4,
            Weights::Bf16 => super::TARGET_REPO,
        }
    }
}

impl Config {
    pub const NANBEIGE_4_3_33B: Self = Self {
        layers: 52,
        hidden: 7168,
        intermediate: 22016,
        n_heads: 56,
        n_kv_heads: 8,
        vocab: 166912,
        rms_norm_eps: 1e-5,
        rope_theta: 120_000_000.0,
    };

    pub fn q_dim(&self) -> usize {
        self.n_heads * HEAD_DIM
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * HEAD_DIM
    }

    /// The names of the checkpoint tensors layers `[start, end)` need in `weights`: those of
    /// [`Self::parameters`], plus the two scales of every 4-bit matrix.
    pub fn tensor_names(&self, range: (usize, usize), weights: Weights) -> Vec<String> {
        let mut out = Vec::new();
        for (name, _) in self.parameters(range) {
            if weights == Weights::Nvfp4 && name.ends_with("_proj.weight") {
                out.push(format!("{name}_scale"));
                out.push(format!("{name}_scale_2"));
            }
            out.push(name);
        }
        out
    }

    /// The checkpoint tensors layers `[start, end)` need, as `(name, shape)`; with the
    /// embedding table when the range starts at layer 0, and the final norm and output head
    /// when it ends at the last layer.
    pub fn parameters(&self, (start, end): (usize, usize)) -> Vec<(String, Vec<usize>)> {
        let c = self;
        let mut out = Vec::new();
        if start == 0 {
            out.push(("model.embed_tokens.weight".into(), vec![c.vocab, c.hidden]));
        }
        for l in start..end {
            for (name, shape) in [
                ("input_layernorm", vec![c.hidden]),
                ("self_attn.q_proj", vec![c.q_dim(), c.hidden]),
                ("self_attn.k_proj", vec![c.kv_dim(), c.hidden]),
                ("self_attn.v_proj", vec![c.kv_dim(), c.hidden]),
                ("self_attn.o_proj", vec![c.hidden, c.q_dim()]),
                ("post_attention_layernorm", vec![c.hidden]),
                ("mlp.gate_proj", vec![c.intermediate, c.hidden]),
                ("mlp.up_proj", vec![c.intermediate, c.hidden]),
                ("mlp.down_proj", vec![c.hidden, c.intermediate]),
            ] {
                out.push((format!("model.layers.{l}.{name}.weight"), shape));
            }
        }
        if end == c.layers {
            out.push(("model.norm.weight".into(), vec![c.hidden]));
            out.push(("lm_head.weight".into(), vec![c.vocab, c.hidden]));
        }
        out
    }
}

/// The DFlash2 drafter: a few bidirectional layers over one block of tokens, conditioned on
/// the target's hidden states, plus a candidate selector. It borrows the target's embedding
/// table and output head.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftConfig {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    /// Tokens per block, the verified anchor included: it drafts `block - 1`.
    pub block: usize,
    /// Target layers whose outputs it reads, ascending.
    pub taps: Vec<usize>,
    /// Channels per convolution group.
    pub conv_group: usize,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub mask_token: i32,
}

impl DraftConfig {
    pub fn nanbeige_4_3_33b() -> Self {
        Self {
            layers: 3,
            hidden: 7168,
            intermediate: 8192,
            n_heads: 28,
            n_kv_heads: 4,
            vocab: 166912,
            rms_norm_eps: 1e-5,
            rope_theta: 120_000_000.0,
            block: 8,
            taps: vec![1, 25, 49],
            conv_group: 16,
            selector_rank: 128,
            selector_top_k: 32,
            mask_token: 0,
        }
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * HEAD_DIM
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * HEAD_DIM
    }
    pub fn conv_groups(&self) -> usize {
        self.hidden / self.conv_group
    }

    /// Every tensor of the drafter checkpoint, as `(name, shape)`.
    pub fn parameters(&self) -> Vec<(String, Vec<usize>)> {
        let c = self;
        let coef = 4 * c.conv_groups(); // 2 sides x 2 taps
        let mut out = vec![
            (
                "fc.weight".to_string(),
                vec![c.hidden, c.taps.len() * c.hidden],
            ),
            ("hidden_norm.weight".into(), vec![c.hidden]),
            ("norm.weight".into(), vec![c.hidden]),
            (
                "candidate_selector.hidden_projection.weight".into(),
                vec![c.selector_rank, c.hidden],
            ),
            (
                "candidate_selector.predecessor_codebook".into(),
                vec![c.vocab, c.selector_rank],
            ),
            (
                "candidate_selector.successor_codebook".into(),
                vec![c.vocab, c.selector_rank],
            ),
        ];
        for l in 0..c.layers {
            for (name, shape) in [
                ("input_layernorm.weight", vec![c.hidden]),
                ("self_attn.q_proj.weight", vec![c.q_dim(), c.hidden]),
                ("self_attn.k_proj.weight", vec![c.kv_dim(), c.hidden]),
                ("self_attn.v_proj.weight", vec![c.kv_dim(), c.hidden]),
                ("self_attn.o_proj.weight", vec![c.hidden, c.q_dim()]),
                ("self_attn.q_norm.weight", vec![HEAD_DIM]),
                ("self_attn.k_norm.weight", vec![HEAD_DIM]),
                ("post_attention_layernorm.weight", vec![c.hidden]),
                ("mlp.gate_proj.weight", vec![c.intermediate, c.hidden]),
                ("mlp.up_proj.weight", vec![c.intermediate, c.hidden]),
                ("mlp.down_proj.weight", vec![c.hidden, c.intermediate]),
                ("attention_conv.base_kernel", vec![2, 2, c.hidden]),
                (
                    "attention_conv.kernel_projection.weight",
                    vec![coef, c.hidden],
                ),
                ("mlp_conv.base_kernel", vec![2, 2, c.hidden]),
                ("mlp_conv.kernel_projection.weight", vec![coef, c.hidden]),
            ] {
                out.push((format!("layers.{l}.{name}"), shape));
            }
        }
        out
    }
}

/// Cosines and sines of the rotary angles for positions `[0, positions)`:
/// `[positions, 2, HEAD_DIM / 2]`, angle = position * theta^(-2i / HEAD_DIM).
pub fn rope_table(theta: f64, positions: usize) -> Vec<f32> {
    let half = HEAD_DIM / 2;
    let mut out = vec![0f32; positions * HEAD_DIM];
    for p in 0..positions {
        for i in 0..half {
            // The angle is formed in f32, as the published implementations do.
            let inv_freq = (1.0 / theta.powf(2.0 * i as f64 / HEAD_DIM as f64)) as f32;
            let angle = p as f32 * inv_freq;
            out[p * HEAD_DIM + i] = angle.cos();
            out[p * HEAD_DIM + half + i] = angle.sin();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_counts_match_the_published_models() {
        let c = Config::NANBEIGE_4_3_33B;
        let count = |ps: Vec<(String, Vec<usize>)>| -> usize {
            ps.iter().map(|(_, s)| s.iter().product::<usize>()).sum()
        };
        // 66,237,954,048 bytes of bf16 in the repository index.
        assert_eq!(count(c.parameters((0, c.layers))) * 2, 66_237_954_048);
        // The 4-bit repository: half a byte per layer-matrix weight, a scale byte per 16 of
        // them and four bytes per matrix; everything else bf16. 22,070,230,448 bytes.
        let fp4: usize = c
            .parameters((0, c.layers))
            .iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                if name.ends_with("_proj.weight") {
                    n / 2 + n / 16 + 4
                } else {
                    n * 2
                }
            })
            .sum();
        assert_eq!(fp4, 22_070_230_448);
        assert_eq!(
            c.tensor_names((0, c.layers), Weights::Nvfp4).len(),
            c.parameters((0, c.layers)).len() + 2 * 7 * c.layers
        );
        // The drafter card: 979,731,200 parameters.
        assert_eq!(
            count(DraftConfig::nanbeige_4_3_33b().parameters()),
            979_731_200
        );
    }

    #[test]
    fn rope_table_starts_at_identity() {
        let t = rope_table(1e4, 2);
        assert!(t[..64].iter().all(|&c| c == 1.0) && t[64..128].iter().all(|&s| s == 0.0));
        assert!((t[128] - 1f32.cos()).abs() < 1e-6 && (t[192] - 1f32.sin()).abs() < 1e-6);
    }
}

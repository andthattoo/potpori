//! Qwen3 model geometry, shared by every backend.

/// Qwen3 model configuration. Parameterized so the small draft (0.6B) and the bigger targets
/// (1.7B, 4B) load from one description.
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

    /// Every checkpoint tensor of the model, as `(name, shape)`: the embedding table, each
    /// layer's norms and matrices, and the final norm. The output head is the embedding.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn parameters(&self) -> Vec<(String, Vec<usize>)> {
        let c = self;
        let mut out = vec![(
            "model.embed_tokens.weight".to_string(),
            vec![c.vocab_size, c.hidden],
        )];
        for l in 0..c.layers {
            for (name, shape) in [
                ("input_layernorm", vec![c.hidden]),
                ("self_attn.q_proj", vec![c.q_dim(), c.hidden]),
                ("self_attn.k_proj", vec![c.kv_dim(), c.hidden]),
                ("self_attn.v_proj", vec![c.kv_dim(), c.hidden]),
                ("self_attn.o_proj", vec![c.hidden, c.q_dim()]),
                ("self_attn.q_norm", vec![c.head_dim]),
                ("self_attn.k_norm", vec![c.head_dim]),
                ("post_attention_layernorm", vec![c.hidden]),
                ("mlp.gate_proj", vec![c.intermediate, c.hidden]),
                ("mlp.up_proj", vec![c.intermediate, c.hidden]),
                ("mlp.down_proj", vec![c.hidden, c.intermediate]),
            ] {
                out.push((format!("model.layers.{l}.{name}.weight"), shape));
            }
        }
        out.push(("model.norm.weight".to_string(), vec![c.hidden]));
        out
    }
}

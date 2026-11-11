//! Chain configuration — the static "who serves which layers" file for a friend-group chain.
//!
//! M0 keeps this deliberately dumb: a TOML file everyone shares. The first node is the HEAD
//! (runs locally in `potpori chat`: embeddings + first layer block + the draft model). The
//! remaining nodes are remote, in pipeline order; the last one is the TAIL (final norm+lm_head).
//! Each remote friend runs `potpori serve` with their layer range; middles are told their
//! downstream address. Discovery/NAT traversal/re-planning come later (see
//! `docs/chain-formation-design.md`).

use std::net::SocketAddr;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ChainConfig {
    /// Target model HF repo, e.g. "Qwen/Qwen3-4B".
    pub model: String,
    /// Draft model HF repo (runs entirely on the head). Small + same tokenizer family.
    #[serde(default = "default_draft")]
    pub draft_model: String,
    /// Draft length γ per round; the verify chunk is γ+1 tokens.
    #[serde(default = "default_gamma")]
    pub gamma: usize,
    /// Max sequence length — sizes every shard's KV cache.
    #[serde(default = "default_max_seq")]
    pub max_seq: usize,
    /// How many in-flight verify chunks the pipelined driver keeps (depth-N pipelining).
    #[serde(default = "default_depth")]
    pub depth: usize,
    /// Egglog search budget per shard load (higher = better kernels, slower startup).
    #[serde(default = "default_search_graphs")]
    pub search_graphs: usize,
    /// Pipeline order: nodes[0] = head (local), nodes[1..] remote, last = tail.
    pub nodes: Vec<ChainNode>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChainNode {
    /// "host:port" the node listens on. Optional (ignored) for the head.
    pub addr: Option<String>,
    /// Global layer range [start, end) this node serves.
    pub layers: [usize; 2],
}

fn default_draft() -> String {
    "Qwen/Qwen3-0.6B".to_string()
}
fn default_gamma() -> usize {
    7
}
fn default_max_seq() -> usize {
    768 // the validated point; `potpori serve --max-seq` defaults to the same value
}
fn default_depth() -> usize {
    4
}
fn default_search_graphs() -> usize {
    200
}

impl ChainConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read chain config {}", path.display()))?;
        let cfg: ChainConfig = toml::from_str(&text).context("parse chain config TOML")?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// KV positions a generation can touch: the prompt, up to `max_new` committed tokens, and
    /// up to `depth` speculative chunks of `gamma` tokens in flight past the committed boundary.
    /// (The driver grid test asserts the drivers never feed a position beyond this.)
    pub fn positions_needed(&self, prompt_tokens: usize, max_new: usize) -> usize {
        prompt_tokens + max_new + self.depth.max(1) * self.gamma.max(1)
    }

    /// Fail before the slow loads if this generation cannot fit in every shard's KV cache —
    /// past `max_seq` the KV scatter would write out of range on the GPU.
    pub fn check_context(&self, prompt_tokens: usize, max_new: usize) -> Result<()> {
        let need = self.positions_needed(prompt_tokens, max_new);
        if need > self.max_seq {
            let spec = self.depth.max(1) * self.gamma.max(1);
            bail!(
                "context does not fit: {prompt_tokens} prompt tokens + {max_new} max_new + {spec} \
                 speculative (depth x gamma) = {need} positions, but max_seq is {}. Shorten the \
                 prompt, lower --max-new (at most {} here), or raise max_seq on every node.",
                self.max_seq,
                self.max_seq.saturating_sub(prompt_tokens + spec),
            );
        }
        Ok(())
    }

    /// The verify chunk size every shard must be compiled for (`potpori serve --chunk`).
    pub fn chunk(&self) -> usize {
        self.gamma + 1
    }

    /// Check the plan against the target model's layer count (known only once the model is
    /// resolved, so this is separate from `validate`).
    pub fn check_layers(&self, model_layers: usize) -> Result<()> {
        let end = self.nodes.last().expect("validated: non-empty").layers[1];
        if end != model_layers {
            bail!(
                "the last node must end at layer {model_layers} (the tail owns the lm_head of {}), \
                 but the chain ends at {end}",
                self.model
            );
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        if self.nodes.len() < 2 {
            bail!("a chain needs at least 2 nodes (a local head + a remote tail)");
        }
        if self.gamma == 0 {
            bail!("gamma must be at least 1");
        }
        if self.depth == 0 {
            bail!("depth must be at least 1");
        }
        if self.max_seq <= self.chunk() {
            bail!("max_seq ({}) must be larger than the verify chunk gamma+1 ({})", self.max_seq, self.chunk());
        }
        if self.nodes[0].layers[0] != 0 {
            bail!("the first node is the head and must start at layer 0");
        }
        for w in self.nodes.windows(2) {
            if w[0].layers[1] != w[1].layers[0] {
                bail!(
                    "layer ranges must be contiguous: [{}, {}) is followed by [{}, {})",
                    w[0].layers[0],
                    w[0].layers[1],
                    w[1].layers[0],
                    w[1].layers[1]
                );
            }
        }
        for (i, n) in self.nodes.iter().enumerate() {
            if n.layers[0] >= n.layers[1] {
                bail!("node {i} has an empty layer range [{}, {})", n.layers[0], n.layers[1]);
            }
        }
        for i in 1..self.nodes.len() {
            self.remote_addr(i)?;
        }
        Ok(())
    }

    /// The socket address of remote node `i` (i >= 1).
    pub fn remote_addr(&self, i: usize) -> Result<SocketAddr> {
        let Some(addr) = self.nodes[i].addr.as_deref() else {
            bail!("remote node {i} needs an addr");
        };
        addr.parse()
            .with_context(|| format!("node {i}: addr {addr:?} must be IP:port, e.g. 100.64.0.2:4010"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_validates() {
        let cfg: ChainConfig = toml::from_str(
            r#"
            model = "Qwen/Qwen3-4B"
            [[nodes]]
            layers = [0, 12]
            [[nodes]]
            addr = "100.64.0.2:4010"
            layers = [12, 24]
            [[nodes]]
            addr = "100.64.0.3:4010"
            layers = [24, 36]
            "#,
        )
        .unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.gamma, 7);
        assert_eq!(cfg.chunk(), 8);
        assert_eq!(cfg.max_seq, 768);
        assert_eq!(cfg.draft_model, "Qwen/Qwen3-0.6B");
        assert_eq!(cfg.nodes.len(), 3);
        assert_eq!(cfg.remote_addr(2).unwrap(), "100.64.0.3:4010".parse().unwrap());
        cfg.check_layers(36).unwrap();
        assert!(cfg.check_layers(28).is_err());
    }

    fn two_node(extra: &str, head: &str, tail_addr: &str, tail: &str) -> Result<ChainConfig> {
        let cfg: ChainConfig = toml::from_str(&format!(
            "model = \"Qwen/Qwen3-4B\"\n{extra}\n[[nodes]]\nlayers = {head}\n[[nodes]]\n{tail_addr}\nlayers = {tail}\n"
        ))?;
        cfg.validate()?;
        Ok(cfg)
    }

    #[test]
    fn rejects_bad_plans() {
        const ADDR: &str = "addr = \"1.2.3.4:4010\"";
        two_node("", "[0, 18]", ADDR, "[18, 36]").unwrap();
        for (what, r) in [
            ("gamma 0", two_node("gamma = 0", "[0, 18]", ADDR, "[18, 36]")),
            ("depth 0", two_node("depth = 0", "[0, 18]", ADDR, "[18, 36]")),
            ("tiny max_seq", two_node("max_seq = 8", "[0, 18]", ADDR, "[18, 36]")),
            ("empty head", two_node("", "[0, 0]", ADDR, "[0, 36]")),
            ("empty tail", two_node("", "[0, 36]", ADDR, "[36, 36]")),
            ("head not at 0", two_node("", "[1, 18]", ADDR, "[18, 36]")),
            ("missing addr", two_node("", "[0, 18]", "", "[18, 36]")),
            ("placeholder addr", two_node("", "[0, 18]", "addr = \"MAC2_IP:4010\"", "[18, 36]")),
            ("addr without port", two_node("", "[0, 18]", "addr = \"1.2.3.4\"", "[18, 36]")),
        ] {
            assert!(r.is_err(), "{what} should be rejected");
        }
    }

    #[test]
    fn context_bound() {
        let cfg: ChainConfig = toml::from_str(
            r#"
            model = "Qwen/Qwen3-4B"
            max_seq = 768
            gamma = 7
            depth = 4
            [[nodes]]
            layers = [0, 18]
            [[nodes]]
            addr = "1.2.3.4:4010"
            layers = [18, 36]
            "#,
        )
        .unwrap();
        assert_eq!(cfg.positions_needed(100, 256), 100 + 256 + 28);
        cfg.check_context(100, 256).unwrap();
        cfg.check_context(100, 640).unwrap(); // exactly 768
        let err = cfg.check_context(100, 641).unwrap_err().to_string();
        assert!(err.contains("max_seq is 768") && err.contains("at most 640"), "{err}");
        assert!(cfg.check_context(800, 1).is_err());
    }

    #[test]
    fn rejects_gaps() {
        let cfg: ChainConfig = toml::from_str(
            r#"
            model = "Qwen/Qwen3-4B"
            [[nodes]]
            layers = [0, 12]
            [[nodes]]
            addr = "1.2.3.4:4010"
            layers = [13, 36]
            "#,
        )
        .unwrap();
        assert!(cfg.validate().is_err());
    }
}

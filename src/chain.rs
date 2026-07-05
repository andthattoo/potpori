//! Chain configuration — the static "who serves which layers" file for a friend-group chain.
//!
//! M0 keeps this deliberately dumb: a TOML file everyone shares. The first node is the HEAD
//! (runs locally in `potpori chat`: embeddings + first layer block + the draft model). The
//! remaining nodes are remote, in pipeline order; the last one is the TAIL (final norm+lm_head).
//! Each remote friend runs `potpori serve` with their layer range; middles are told their
//! downstream address. Discovery/NAT traversal/re-planning come later (see
//! `docs/chain-formation-design.md`).

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
    2048
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

    fn validate(&self) -> Result<()> {
        if self.nodes.len() < 2 {
            bail!("a chain needs at least 2 nodes (a local head + a remote tail)");
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
        for (i, n) in self.nodes.iter().enumerate().skip(1) {
            if n.addr.is_none() {
                bail!("remote node {i} needs an addr");
            }
            if n.layers[0] >= n.layers[1] {
                bail!("node {i} has an empty layer range");
            }
        }
        Ok(())
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
        assert_eq!(cfg.draft_model, "Qwen/Qwen3-0.6B");
        assert_eq!(cfg.nodes.len(), 3);
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

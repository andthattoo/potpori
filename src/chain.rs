//! The chain config: a TOML file on the head that says which node serves which layers.
//!
//! The first node is the head (it runs inside `potpori chat`: tokenizer, first layer block and
//! the draft model). The remaining nodes are remote, in pipeline order; the last one is the
//! tail (final norm and output projection). Each remote friend runs `potpori serve` with the
//! matching layer range, and middles are told their downstream address.

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
    /// How many verification chunks the driver keeps in flight (1 = no pipelining).
    #[serde(default = "default_depth")]
    pub depth: usize,
    /// Kernel search budget per shard load (higher = better kernels, slower startup).
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
    768
}
fn default_depth() -> usize {
    4
}
fn default_search_graphs() -> usize {
    200
}

/// Layer count of each supported model. Kept here, not with the model code, so the config is
/// validated in builds without the inference engine (a test checks the two tables agree).
pub fn model_layers(repo: &str) -> Option<usize> {
    match repo {
        "Qwen/Qwen3-0.6B" | "Qwen/Qwen3-1.7B" => Some(28),
        "Qwen/Qwen3-4B" => Some(36),
        _ => None,
    }
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
        let layers = model_layers(&self.model)
            .ok_or_else(|| anyhow::anyhow!("unsupported target model {}", self.model))?;
        if model_layers(&self.draft_model).is_none() {
            bail!("unsupported draft model {}", self.draft_model);
        }
        if self.gamma == 0 || self.gamma.checked_add(1).is_none_or(|c| c > self.max_seq) {
            bail!("require 1 <= gamma < max_seq");
        }
        if self.max_seq > i32::MAX as usize
            || self.depth == 0
            || self.depth > 64
            || self.search_graphs == 0
        {
            bail!("invalid max_seq, depth (1..64), or search_graphs");
        }
        if self.nodes.len() < 2 {
            bail!("a chain needs at least 2 nodes (a local head + a remote tail)");
        }
        if self.nodes[0].layers[0] != 0 {
            bail!("the first node is the head and must start at layer 0");
        }
        if self.nodes.last().unwrap().layers[1] != layers {
            bail!("chain must cover all {layers} model layers");
        }
        for (i, n) in self.nodes.iter().enumerate() {
            if n.layers[0] >= n.layers[1] || n.layers[1] > layers {
                bail!("invalid layer range on node {i}");
            }
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
            n.addr
                .as_ref()
                .unwrap()
                .parse::<std::net::SocketAddr>()
                .with_context(|| format!("node {i}: addr must be IP:port"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_settings_before_loading_models() {
        let valid: ChainConfig = toml::from_str(include_str!("../chain.example.toml")).unwrap();
        valid.validate().unwrap();
        for change in [0, 1, 2, 3, 4, 5, 6] {
            let mut c = valid.clone();
            match change {
                0 => c.gamma = 0,
                1 => c.gamma = usize::MAX,
                2 => c.max_seq = 0,
                3 => c.nodes[0].layers = [0, 0],
                4 => c.nodes.last_mut().unwrap().layers[1] = 35,
                5 => c.depth = 0,
                _ => c.model = "Qwen/Qwen3-14B".into(),
            }
            assert!(c.validate().is_err(), "case {change}");
        }
    }

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

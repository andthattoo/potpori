//! The chain config: a TOML file on the head that says which node serves which layers.
//!
//! The first node is the head (it runs inside `potpori chat`: tokenizer, first layer block and
//! the drafter). The remaining nodes are remote, in pipeline order; the last one is the tail
//! (final norm and output head). Each remote friend runs `potpori serve` with the matching
//! layer range, and middles are told their downstream address.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::nanbeige::config::Config;

pub const DEFAULT_MAX_SEQ: usize = 4096;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChainConfig {
    /// Max sequence length — sizes every shard's KV cache.
    #[serde(default = "default_max_seq")]
    pub max_seq: usize,
    /// Pipeline order: nodes[0] = head (local), nodes[1..] remote, last = tail.
    pub nodes: Vec<ChainNode>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChainNode {
    /// "host:port" the node listens on. Optional (ignored) for the head.
    pub addr: Option<String>,
    /// Global layer range [start, end) this node serves.
    pub layers: [usize; 2],
}

fn default_max_seq() -> usize {
    DEFAULT_MAX_SEQ
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
        let layers = Config::NANBEIGE_4_3_33B.layers;
        if self.max_seq < 16 || self.max_seq > i32::MAX as usize {
            bail!("invalid max_seq");
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
        for change in 0..4 {
            let mut c = valid.clone();
            match change {
                0 => c.max_seq = 0,
                1 => c.nodes[0].layers = [0, 0],
                2 => c.nodes.last_mut().unwrap().layers[1] = 51,
                _ => c.nodes.truncate(1),
            }
            assert!(c.validate().is_err(), "case {change}");
        }
    }

    #[test]
    fn parses_and_validates() {
        let cfg: ChainConfig = toml::from_str(
            r#"
            [[nodes]]
            layers = [0, 8]
            [[nodes]]
            addr = "100.64.0.2:4010"
            layers = [8, 30]
            [[nodes]]
            addr = "100.64.0.3:4010"
            layers = [30, 52]
            "#,
        )
        .unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.max_seq, DEFAULT_MAX_SEQ);
        assert_eq!(cfg.nodes.len(), 3);
    }

    #[test]
    fn rejects_gaps_and_settings_from_older_versions() {
        let cfg: ChainConfig = toml::from_str(
            r#"
            [[nodes]]
            layers = [0, 12]
            [[nodes]]
            addr = "1.2.3.4:4010"
            layers = [13, 52]
            "#,
        )
        .unwrap();
        assert!(cfg.validate().is_err());
        let old = "gamma = 7\n[[nodes]]\nlayers = [0, 52]\n";
        assert!(toml::from_str::<ChainConfig>(old).is_err());
    }
}

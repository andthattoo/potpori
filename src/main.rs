//! potpori — sharded LLM inference with your friends over WAN.
//!
//! A model is cut into contiguous layer shards; each node serves one shard; the residual-stream
//! hidden state hops node→node over QUIC. Generation is driven by lossless greedy speculative
//! decoding (a draft model on the head proposes γ tokens; the chain verifies them in ONE
//! traversal) with depth-N pipelining to hide WAN RTT. See `docs/` for the design.
//!
//! Roles:
//!   - `potpori serve` — run a middle/tail shard on each friend's box.
//!   - `potpori chat`  — the head: embeds + first shard + draft model + tokenizer, locally.

// Without a backend feature these have no consumer (`run` is compiled out) — don't warn.
#[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
mod chain;
#[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
mod engine;
#[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
mod node;
#[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
mod specdecode;
#[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
mod specpipe;
#[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
mod transport;

#[cfg(feature = "engine")]
#[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
mod qwen;

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "potpori",
    version,
    about = "Sharded LLM serving with your friends over WAN — pipeline-parallel + lossless speculative decoding on Luminal"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve a shard of the chain (middle or tail) — run this on each friend's box.
    Serve {
        /// Address to listen on, e.g. 0.0.0.0:4010
        #[arg(long)]
        listen: SocketAddr,
        /// Target model HF repo (must match the chain config), e.g. Qwen/Qwen3-4B
        #[arg(long)]
        model: String,
        /// Global layer range to serve as start:end, e.g. 12:24
        #[arg(long, value_parser = parse_layers)]
        layers: (usize, usize),
        /// Role in the chain: tail (last shard, owns the lm_head) or middle
        #[arg(long, value_enum)]
        role: Role,
        /// Downstream node address (required for middles)
        #[arg(long)]
        downstream: Option<SocketAddr>,
        /// Max sequence length — sizes the KV cache (must match the chain config's max_seq;
        /// 768 is the validated point and the chain default)
        #[arg(long, default_value_t = 768)]
        max_seq: usize,
        /// Verify chunk size = γ+1. Compiles chunk-specialized kernels; 0 = generic dynamic
        /// kernel. Metal NEEDS the specialized kernel (the generic multi-token kernel is ~20×
        /// slower there); on CUDA 0 is fine. Must equal the chain's gamma+1.
        #[arg(long, default_value_t = 8)]
        chunk: usize,
        /// Egglog search budget per shard load (kernel quality vs startup time)
        #[arg(long, default_value_t = 200)]
        search_graphs: usize,
        /// Shared secret a head must present (its chain config's `token`). Without one, anyone
        /// who can reach the port can use this shard. Also read from POTPORI_TOKEN.
        #[arg(long)]
        token: Option<String>,
    },
    /// Run the head: prompt in → sharded, spec-decoded generation out.
    Chat {
        /// Path to the chain TOML (see chain.example.toml)
        #[arg(long, default_value = "chain.toml")]
        chain: PathBuf,
        /// The user message
        #[arg(long)]
        prompt: String,
        /// Max new tokens
        #[arg(long, default_value_t = 256)]
        max_new: usize,
        /// Override the chain config's `depth` (in-flight verify chunks) for this run —
        /// e.g. `--depth 1` vs `--depth 4` to measure what pipelining buys over a given link.
        #[arg(long)]
        depth: Option<usize>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Role {
    Middle,
    Tail,
}

impl Role {
    #[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
    fn name(self) -> &'static str {
        match self {
            Role::Middle => "middle",
            Role::Tail => "tail",
        }
    }
}

/// Parse `--layers start:end` into a non-empty global layer range.
fn parse_layers(s: &str) -> Result<(usize, usize), String> {
    let (a, b) = s
        .split_once(':')
        .ok_or_else(|| "must be start:end, e.g. 12:24".to_string())?;
    let a: usize = a.trim().parse().map_err(|e| format!("start {a:?}: {e}"))?;
    let b: usize = b.trim().parse().map_err(|e| format!("end {b:?}: {e}"))?;
    if a >= b {
        return Err(format!("empty layer range {a}:{b} (start must be below end)"));
    }
    Ok((a, b))
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let cli = Cli::parse();

    #[cfg(any(feature = "metal", feature = "cuda"))]
    {
        run::dispatch(cli.cmd)
    }
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        let _ = cli.cmd;
        anyhow::bail!(
            "potpori was built without a GPU backend — rebuild with `--features metal` (Apple) \
             or `--features cuda` (NVIDIA)"
        )
    }
}

#[cfg(any(feature = "metal", feature = "cuda"))]
mod run {
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use anyhow::{anyhow, bail, Result};

    use crate::chain::ChainConfig;
    use crate::node::{serve_middle, serve_tail, ChainLink, NodeSpec, ShardEngine as _};
    use crate::qwen::draft::IncrementalDraft;
    use crate::qwen::hf::{prepare_hf_model, COMBINED_WEIGHTS_FILE};
    use crate::qwen::model::{main_dtype, QwenConfig};
    use crate::qwen::shard::Shard;
    use crate::qwen::target::QwenTarget;
    use crate::qwen::tok::ChatTokenizer;
    use crate::qwen::{config_for, EOS_TOKENS, KNOWN_MODELS};
    use crate::specdecode::SpecConfig;
    use crate::specpipe::run_spec_decode_pipelined;
    use crate::transport::listen_endpoint;
    use crate::{Cmd, Role};

    // Backend selection: CUDA wins if both features are enabled.
    #[cfg(feature = "cuda")]
    type Backend = luminal_cuda_lite::runtime::CudaRuntime;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    type Backend = luminal_metal::MetalRuntime;

    fn new_runtime() -> Backend {
        #[cfg(feature = "cuda")]
        {
            luminal_cuda_lite::runtime::CudaRuntime::new().expect("init CUDA runtime")
        }
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        {
            use luminal::prelude::Runtime as _;
            luminal_metal::MetalRuntime::initialize(())
        }
    }

    /// e.g. "cuda/Bf16" — what this build computes with (reported in the handshake).
    fn backend_name() -> String {
        let backend = if cfg!(feature = "cuda") { "cuda" } else { "metal" };
        format!("{backend}/{:?}", main_dtype())
    }

    /// How long the head keeps retrying a node that is not reachable yet (still loading).
    const CONNECT_WAIT: Duration = Duration::from_secs(15 * 60);

    pub fn dispatch(cmd: Cmd) -> Result<()> {
        match cmd {
            Cmd::Serve { listen, model, layers, role, downstream, max_seq, chunk, search_graphs, token } => {
                serve(listen, model, layers, role, downstream, max_seq, chunk, search_graphs, token)
            }
            Cmd::Chat { chain, prompt, max_new, depth } => chat(chain, prompt, max_new, depth),
        }
    }

    /// Resolve a model config without touching the network (cheap; for early validation).
    fn known_config(repo: &str) -> Result<QwenConfig> {
        config_for(repo)
            .ok_or_else(|| anyhow!("unknown model {repo:?} — known: {}", KNOWN_MODELS.join(", ")))
    }

    /// Resolve a known Qwen config + prepared (downloaded/combined) weights path.
    fn weights_for(repo: &str) -> Result<(QwenConfig, String)> {
        let cfg = known_config(repo)?;
        tracing::info!("preparing weights for {repo} (downloads + combines on first run)…");
        let dir = prepare_hf_model(repo).map_err(|e| anyhow!("prepare {repo}: {e}"))?;
        Ok((cfg, dir.join(COMBINED_WEIGHTS_FILE).to_string_lossy().into_owned()))
    }

    /// `--token` / config value, overridden by the POTPORI_TOKEN environment variable.
    fn resolve_token(configured: Option<String>) -> Option<String> {
        std::env::var("POTPORI_TOKEN").ok().or(configured).filter(|t| !t.is_empty())
    }

    #[allow(clippy::too_many_arguments)]
    fn serve(
        listen: SocketAddr,
        model: String,
        (a, b): (usize, usize),
        role: Role,
        downstream: Option<SocketAddr>,
        max_seq: usize,
        chunk: usize,
        search_graphs: usize,
        token: Option<String>,
    ) -> Result<()> {
        let is_last = role == Role::Tail;
        let role = role.name();
        // Validate everything cheap before the download + kernel search.
        let cfg = known_config(&model)?;
        if a == 0 {
            bail!("layer 0 belongs to the head — run `potpori chat` on the head instead");
        }
        if is_last && b != cfg.layers {
            bail!("tail must end at the last layer ({}), got {b}", cfg.layers);
        }
        if !is_last && b >= cfg.layers {
            bail!(
                "a middle must end before the last layer ({}), got {b} — the shard that ends \
                 there is the tail (--role tail)",
                cfg.layers
            );
        }
        if !is_last && downstream.is_none() {
            bail!("--downstream is required for middles");
        }
        if is_last && downstream.is_some() {
            bail!("--downstream is only for middles (the tail is the end of the chain)");
        }
        if chunk == 1 {
            bail!("--chunk must be 0 (generic kernel) or gamma+1 >= 2");
        }
        if chunk >= max_seq {
            bail!("--chunk ({chunk}) must be smaller than --max-seq ({max_seq})");
        }
        let token = resolve_token(token);
        if token.is_none() {
            tracing::warn!(
                "no --token set: anyone who can reach {listen} can use this shard. Fine on a \
                 private network (Tailscale); otherwise set --token here and `token` in the chain config."
            );
        }
        let spec = NodeSpec {
            model: model.clone(),
            layers: (a, b),
            is_tail: is_last,
            chunk,
            max_seq,
            token,
            backend: backend_name(),
        };

        let (cfg, wp) = weights_for(&model)?;
        tracing::info!(
            "loading shard [{a},{b}) of {model} on {} — egglog kernel search, takes minutes…",
            spec.backend
        );
        let mut shard = Shard::load_chunked(
            new_runtime(),
            cfg,
            (a, b),
            false,
            is_last,
            &wp,
            max_seq,
            search_graphs,
            chunk,
        );
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        let ep = {
            let _g = rt.enter();
            listen_endpoint(listen).map_err(|e| anyhow!("listen on {listen}: {e}"))?
        };
        tracing::info!("shard ready — serving [{a},{b}) as {role} on {listen}");
        loop {
            shard.reset(); // fresh KV per session
            let res = match downstream {
                None => serve_tail(&mut shard, &spec, &ep, &rt),
                Some(next) => serve_middle(&mut shard, &spec, &ep, next, &rt),
            };
            match res {
                Ok(()) => tracing::info!("session ended cleanly — waiting for the next head"),
                Err(e) => tracing::warn!("session ended: {e} — waiting for the next head"),
            }
        }
    }

    fn chat(chain_path: PathBuf, prompt: String, max_new: usize, depth: Option<usize>) -> Result<()> {
        let cc = ChainConfig::load(&chain_path)?;
        let chunk = cc.chunk();
        let depth = depth.unwrap_or(cc.depth).max(1);
        // Cheap plan checks first, so config errors surface before the slow loads.
        let tcfg = known_config(&cc.model)?;
        cc.check_layers(tcfg.layers)?;
        known_config(&cc.draft_model)?;
        let next_addr = cc.remote_addr(1)?;

        // Tokenize first (fast) so a too-long prompt also surfaces before the slow loads.
        let tok = ChatTokenizer::load(&cc.model)?;
        let ids = tok.encode_chat(&prompt)?;
        ChainConfig { depth, ..cc.clone() }.check_context(ids.len(), max_new)?;

        let (dcfg, dwp) = weights_for(&cc.draft_model)?;
        tracing::info!("loading draft model {}…", cc.draft_model);
        let draft_target = QwenTarget::load_chunked(
            new_runtime(),
            dcfg,
            &dwp,
            cc.max_seq,
            cc.search_graphs,
            chunk, // point buckets {1, γ+1} — the generic dynamic bucket miscompiles on Metal
        );

        let (tcfg, twp) = weights_for(&cc.model)?;
        let head_range = (cc.nodes[0].layers[0], cc.nodes[0].layers[1]);
        tracing::info!(
            "loading head shard [{},{}) of {} on {} — egglog kernel search, takes minutes…",
            head_range.0,
            head_range.1,
            cc.model,
            backend_name()
        );
        let head = Shard::load_chunked(
            new_runtime(),
            tcfg,
            head_range,
            true,
            false,
            &twp,
            cc.max_seq,
            cc.search_graphs,
            chunk,
        );
        let mut draft = IncrementalDraft::new(draft_target, chunk);
        {
            // Sanity: all-NaN logits argmax to token 0. Probe with a 1-token draft (this also
            // warm-feeds the prompt into the draft's KV — not wasted work).
            use crate::engine::DraftEngine as _;
            let probe = draft.draft(&ids, 1);
            if probe[0] == 0 {
                bail!(
                    "draft sanity check failed: first prediction is token 0 (all-NaN logits?). \
                     A Luminal kernel for this config miscompiled — try a different max_seq."
                );
            }
            tracing::info!("draft sanity check ok (first prediction = {})", probe[0]);
        }

        tracing::info!("prompt = {} tokens; connecting to {next_addr}…", ids.len());
        let hello = cc.hello(head.hidden_dim(), resolve_token(cc.token.clone()));
        let mut chain = ChainLink::connect(head, next_addr, hello, &EOS_TOKENS, CONNECT_WAIT)?;
        for (i, node) in chain.nodes().iter().enumerate() {
            tracing::info!(
                "chain node {}: layers [{}, {}) on {} (potpori {})",
                i + 1,
                node.layers[0],
                node.layers[1],
                node.backend,
                node.version
            );
        }
        tracing::info!("chain up — RTT to next node {:.1} ms", chain.rtt().as_secs_f64() * 1e3);

        let t0 = Instant::now();
        let generated = run_spec_decode_pipelined(
            &mut draft,
            &mut chain,
            &ids,
            &SpecConfig { gamma: cc.gamma, max_new },
            depth,
        );
        let dt = t0.elapsed().as_secs_f64();
        chain.shutdown();
        let (out, stats) = generated?;

        println!("\n{}", tok.decode(&out)?);
        let link = chain.stats();
        let secs = |d: Duration| d.as_secs_f64();
        let gen = (dt - secs(link.prefill)).max(1e-9);
        eprintln!(
            "\n[{} tokens in {dt:.1}s = {:.1} tok/s overall, {:.1} tok/s after prefill | {:.2} tok/traversal \
             over {} traversals | γ={} depth={depth} chain={} nodes]",
            out.len(),
            out.len() as f64 / dt,
            out.len() as f64 / gen,
            stats.toks_per_traversal(),
            stats.traversals,
            cc.gamma,
            cc.nodes.len(),
        );
        eprintln!(
            "[time: prefill {:.2}s ({} prompt tokens) | head shard {:.2}s | waiting on chain {:.2}s | \
             draft + rest {:.2}s]",
            secs(link.prefill),
            ids.len(),
            secs(link.head_compute),
            secs(link.waiting),
            (gen - secs(link.head_compute) - secs(link.waiting)).max(0.0),
        );
        eprintln!(
            "[link: RTT {:.1} ms at connect, {:.1} ms now | {} frames, {:.1} KiB/frame | head {} → {}]",
            secs(link.rtt) * 1e3,
            secs(chain.rtt()) * 1e3,
            link.frames,
            link.payload_bytes as f64 / link.frames.max(1) as f64 / 1024.0,
            backend_name(),
            chain.nodes().iter().map(|n| n.backend.as_str()).collect::<Vec<_>>().join(" → "),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_is_well_formed() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    #[test]
    fn layers_parse() {
        assert_eq!(parse_layers("12:24"), Ok((12, 24)));
        assert_eq!(parse_layers(" 0 : 1 "), Ok((0, 1)));
        for bad in ["12", "12-24", "24:12", "5:5", "a:3", "3:", ":"] {
            assert!(parse_layers(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn serve_args() {
        let serve = |extra: &[&str]| {
            let mut args = vec!["potpori", "serve", "--listen", "0.0.0.0:4010", "--model", "Qwen/Qwen3-4B"];
            args.extend_from_slice(extra);
            Cli::try_parse_from(args)
        };
        match serve(&["--layers", "24:36", "--role", "tail"]).unwrap().cmd {
            Cmd::Serve { layers, role, max_seq, chunk, downstream, token, .. } => {
                assert!(token.is_none());
                assert_eq!(layers, (24, 36));
                assert_eq!(role, Role::Tail);
                assert_eq!(max_seq, 768, "must match ChainConfig's default max_seq");
                assert_eq!(chunk, 8, "must match ChainConfig's default gamma + 1");
                assert!(downstream.is_none());
            }
            _ => panic!("expected serve"),
        }
        assert!(serve(&["--layers", "24:36", "--role", "head"]).is_err());
        assert!(serve(&["--layers", "36:24", "--role", "tail"]).is_err());
    }
}

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
mod specdecode;
#[cfg_attr(not(any(feature = "metal", feature = "cuda")), allow(dead_code))]
mod specpipe;
mod transport;

#[cfg(feature = "engine")]
mod qwen;

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

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
        #[arg(long)]
        layers: String,
        /// Role in the chain: "tail" (last shard, owns the lm_head) or "middle"
        #[arg(long)]
        role: String,
        /// Downstream node address (required for middles)
        #[arg(long)]
        downstream: Option<SocketAddr>,
        /// Max sequence length — sizes the KV cache (must match the chain config)
        #[arg(long, default_value_t = 2048)]
        max_seq: usize,
        /// Verify chunk size = γ+1. Compiles chunk-specialized kernels; 0 = generic dynamic
        /// kernel. Metal NEEDS the specialized kernel (the generic multi-token kernel is ~20×
        /// slower there); on CUDA 0 is fine. Must equal the chain's gamma+1.
        #[arg(long, default_value_t = 8)]
        chunk: usize,
        /// Egglog search budget per shard load (kernel quality vs startup time)
        #[arg(long, default_value_t = 200)]
        search_graphs: usize,
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
    },
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

    use anyhow::{anyhow, bail, Result};

    use crate::chain::ChainConfig;
    use crate::qwen::distributed::{serve_middle, serve_tail, PipelinedQuicChain};
    use crate::qwen::draft::IncrementalDraft;
    use crate::qwen::hf::prepare_hf_model;
    use crate::qwen::model::QwenConfig;
    use crate::qwen::shard::Shard;
    use crate::qwen::target::QwenTarget;
    use crate::qwen::tok::ChatTokenizer;
    use crate::qwen::config_for;
    use crate::specdecode::SpecConfig;
    use crate::specpipe::run_spec_decode_pipelined;
    use crate::transport::listen_endpoint;
    use crate::Cmd;

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

    pub fn dispatch(cmd: Cmd) -> Result<()> {
        match cmd {
            Cmd::Serve { listen, model, layers, role, downstream, max_seq, chunk, search_graphs } => {
                serve(listen, model, layers, role, downstream, max_seq, chunk, search_graphs)
            }
            Cmd::Chat { chain, prompt, max_new } => chat(chain, prompt, max_new),
        }
    }

    /// Resolve a known Qwen config + prepared (downloaded/combined) weights path.
    fn weights_for(repo: &str) -> Result<(QwenConfig, String)> {
        let cfg = config_for(repo)
            .ok_or_else(|| anyhow!("unknown model {repo} — known: Qwen/Qwen3-{{0.6B,1.7B,4B}}"))?;
        tracing::info!("preparing weights for {repo} (downloads + combines on first run)…");
        let dir = prepare_hf_model(repo).map_err(|e| anyhow!("prepare {repo}: {e}"))?;
        Ok((cfg, dir.join("model_combined_bf16_v1.safetensors").to_string_lossy().into_owned()))
    }

    fn parse_layers(s: &str) -> Result<(usize, usize)> {
        let (a, b) = s
            .split_once(':')
            .ok_or_else(|| anyhow!("--layers must be start:end, e.g. 12:24"))?;
        Ok((a.trim().parse()?, b.trim().parse()?))
    }

    #[allow(clippy::too_many_arguments)]
    fn serve(
        listen: SocketAddr,
        model: String,
        layers: String,
        role: String,
        downstream: Option<SocketAddr>,
        max_seq: usize,
        chunk: usize,
        search_graphs: usize,
    ) -> Result<()> {
        let (a, b) = parse_layers(&layers)?;
        let is_last = match role.as_str() {
            "tail" => true,
            "middle" => false,
            r => bail!("--role must be tail or middle, got {r}"),
        };
        let (cfg, wp) = weights_for(&model)?;
        if a == 0 {
            bail!("layer 0 belongs to the head — run `potpori chat` on the head instead");
        }
        if is_last && b != cfg.layers {
            bail!("tail must end at the last layer ({}), got {b}", cfg.layers);
        }
        if !is_last && downstream.is_none() {
            bail!("--downstream is required for middles");
        }

        tracing::info!("loading shard [{a},{b}) of {model} — egglog kernel search, takes minutes…");
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
            shard.reset(); // fresh KV per session (absolute-position writes make this belt-and-braces)
            let res = if is_last {
                serve_tail(&mut shard, &ep, &rt)
            } else {
                serve_middle(&mut shard, &ep, downstream.unwrap(), &rt)
            };
            match res {
                Ok(()) => tracing::info!("session ended cleanly — waiting for the next head"),
                Err(e) => tracing::warn!("session error: {e} — waiting for the next head"),
            }
        }
    }

    fn chat(chain_path: PathBuf, prompt: String, max_new: usize) -> Result<()> {
        let cc = ChainConfig::load(&chain_path)?;
        let chunk = cc.gamma + 1;

        // Tokenize first (fast) so config errors surface before the slow loads.
        let tok = ChatTokenizer::load(&cc.model)?;
        let ids = tok.encode_chat(&prompt)?;

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
            "loading head shard [{},{}) of {} — egglog kernel search, takes minutes…",
            head_range.0,
            head_range.1,
            cc.model
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

        let next_addr: SocketAddr = cc.nodes[1]
            .addr
            .as_ref()
            .expect("validated: remote nodes have addrs")
            .parse()?;
        tracing::info!("prompt = {} tokens; dialing {next_addr}…", ids.len());
        let mut chain = PipelinedQuicChain::connect(vec![head], next_addr)
            .map_err(|e| anyhow!("dial {next_addr}: {e}"))?;
        chain.set_prefill_chunk(chunk);

        let t0 = std::time::Instant::now();
        let (out, stats) = run_spec_decode_pipelined(
            &mut draft,
            &mut chain,
            &ids,
            &SpecConfig { gamma: cc.gamma, max_new },
            cc.depth,
        );
        let dt = t0.elapsed().as_secs_f64();
        chain.shutdown();

        println!("\n{}", tok.decode(&out)?);
        eprintln!(
            "\n[{} tokens in {:.1}s = {:.1} tok/s | {:.2} tok/traversal | γ={} depth={} chain={} nodes]",
            out.len(),
            dt,
            out.len() as f64 / dt,
            stats.toks_per_traversal(),
            cc.gamma,
            cc.depth,
            cc.nodes.len(),
        );
        Ok(())
    }
}

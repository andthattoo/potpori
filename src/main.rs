//! potpori — sharded LLM inference with your friends over WAN.
//!
//! A model is cut into contiguous layer shards; each node serves one shard; the residual-stream
//! hidden state hops node→node over QUIC. Generation is driven by greedy speculative decoding
//! (a draft model on the head proposes γ tokens; the chain verifies them in one traversal),
//! with several verification chunks in flight to hide network latency.
//!
//! Commands:
//!   - `potpori identity` — create this node's TLS identity.
//!   - `potpori serve`    — run a middle or tail shard on each friend's machine.
//!   - `potpori chat`     — the head: tokenizer, first shard and draft model, locally.
//!
//! Layout: [`chain`] is the config file, [`transport`] the wire, [`node`] the three roles,
//! [`specpipe`] / [`specdecode`] the decoding drivers, and `qwen` the model on Luminal.

// Without a GPU backend only `identity` runs, so most of the crate has no caller — don't warn.
#[cfg_attr(
    not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))),
    allow(dead_code)
)]
mod chain;
#[cfg_attr(
    not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))),
    allow(dead_code)
)]
mod engine;
#[cfg_attr(
    not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))),
    allow(dead_code)
)]
mod node;
#[cfg_attr(
    not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))),
    allow(dead_code)
)]
mod specdecode;
#[cfg_attr(
    not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))),
    allow(dead_code)
)]
mod specfeat;
#[cfg_attr(
    not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))),
    allow(dead_code)
)]
mod specpipe;
#[cfg_attr(
    not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))),
    allow(dead_code)
)]
mod transport;

#[cfg(feature = "engine")]
#[cfg_attr(
    not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))),
    allow(dead_code)
)]
mod qwen;

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(clap::Args)]
struct TlsArgs {
    /// This node's certificate (share it with adjacent peers).
    #[arg(long, default_value = "identity/cert.pem")]
    cert: PathBuf,
    /// This node's private key (keep it private).
    #[arg(long, default_value = "identity/key.pem")]
    key: PathBuf,
    /// PEM bundle containing the certificates of trusted peers.
    #[arg(long, default_value = "identity/peers.pem")]
    trust: PathBuf,
}

impl TlsArgs {
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    fn load(&self) -> Result<transport::TlsIdentity> {
        transport::TlsIdentity::load(&self.cert, &self.key, &self.trust)
    }
}

#[derive(Parser)]
#[command(
    name = "potpori",
    version,
    about = "Sharded LLM inference with your friends over WAN — pipelined speculative decoding on Luminal"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate this node's identity. Share cert.pem with peers; keep key.pem private.
    Identity {
        #[arg(long, default_value = "identity")]
        out: PathBuf,
    },
    /// Serve a shard of the chain (middle or tail) — run this on each friend's machine.
    Serve {
        #[command(flatten)]
        tls: TlsArgs,
        /// Address to listen on, e.g. 0.0.0.0:4010
        #[arg(long)]
        listen: SocketAddr,
        /// Target model HF repo (must match the chain config), e.g. Qwen/Qwen3-4B
        #[arg(long)]
        model: String,
        /// Global layer range to serve as start:end, e.g. 12:24
        #[arg(long, value_parser = parse_layers)]
        layers: (usize, usize),
        /// Role in the chain: tail (last shard, owns the output projection) or middle
        #[arg(long, value_enum)]
        role: Role,
        /// Downstream node address (required for middles)
        #[arg(long)]
        downstream: Option<SocketAddr>,
        /// Max sequence length — sizes the KV cache (must equal the chain config's max_seq)
        #[arg(long, default_value_t = 768)]
        max_seq: usize,
        /// Verification chunk size; must equal the chain's gamma+1. Kernels are compiled for
        /// exactly 1 and this many tokens. 0 compiles generic kernels (diagnostics only).
        #[arg(long, default_value_t = 8)]
        chunk: usize,
        /// Kernel search budget per shard load (kernel quality vs startup time)
        #[arg(long, default_value_t = 200)]
        search_graphs: usize,
        /// Layers whose outputs go back to the head with each result, e.g. 1,9,17,25,33 —
        /// for a chain whose config names a `drafter`. `potpori chat` prints the list. The
        /// node runs one shard per stretch between tapped layers.
        #[arg(long, value_delimiter = ',')]
        taps: Vec<usize>,
    },
    /// Run the head: prompt in → sharded, speculatively decoded generation out.
    Chat {
        #[command(flatten)]
        tls: TlsArgs,
        /// Path to the chain TOML (see chain.example.toml)
        #[arg(long, default_value = "chain.toml")]
        chain: PathBuf,
        /// The user message
        #[arg(long)]
        prompt: String,
        /// Max new tokens
        #[arg(long, default_value_t = 256)]
        max_new: usize,
        /// Override the chain config's `depth` (verification chunks in flight) for this run,
        /// e.g. `--depth 1` vs `--depth 4` to measure what pipelining buys over a link.
        #[arg(long)]
        depth: Option<usize>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Role {
    Middle,
    Tail,
}

/// Parse `--layers start:end` into a non-empty global layer range.
fn parse_layers(s: &str) -> Result<(usize, usize), String> {
    let (a, b) = s
        .split_once(':')
        .ok_or_else(|| "must be start:end, e.g. 12:24".to_string())?;
    let a: usize = a.trim().parse().map_err(|e| format!("start {a:?}: {e}"))?;
    let b: usize = b.trim().parse().map_err(|e| format!("end {b:?}: {e}"))?;
    if a >= b {
        return Err(format!("empty layer range {a}:{b}"));
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
    if let Cmd::Identity { out } = &cli.cmd {
        transport::TlsIdentity::generate(out)?;
        println!(
            "Identity created in {}. Exchange cert.pem with adjacent peers and collect their \
             certificates in peers.pem.",
            out.display()
        );
        return Ok(());
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    {
        run::dispatch(cli.cmd)
    }
    #[cfg(not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))))]
    {
        let _ = cli.cmd;
        anyhow::bail!(
            "potpori was built without a GPU backend — rebuild with `--features metal` (Apple) \
             or `--features cuda` (NVIDIA)"
        )
    }
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
mod run {
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use anyhow::{anyhow, bail, Result};

    use crate::chain::ChainConfig;
    use crate::node::{
        serve_middle, serve_tail, tap_ranges, ChainLink, ShardEngine as _, TappedShards,
    };
    use crate::qwen::draft::IncrementalDraft;
    use crate::qwen::dspark::{DsparkConfig, DsparkDraft};
    use crate::qwen::hf::{download_drafter, prepare_hf_model, COMBINED_WEIGHTS_FILE};
    use crate::qwen::model::QwenConfig;
    use crate::qwen::shard::Shard;
    use crate::qwen::target::QwenTarget;
    use crate::qwen::tok::ChatTokenizer;
    use crate::qwen::{config_for, EOS_TOKENS};
    use crate::specdecode::{SpecConfig, SpecStats};
    use crate::specfeat::run_feature_decode;
    use crate::specpipe::run_spec_decode_pipelined;
    use crate::transport::{listen_endpoint, TlsIdentity};
    use crate::{Cmd, Role};

    // Backend selection: CUDA wins if both features are enabled.
    #[cfg(feature = "cuda")]
    type Backend = crate::qwen::runtime::CudaRuntime;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    type Backend = crate::qwen::runtime::MetalRuntime;

    const BACKEND: &str = if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "metal"
    };

    /// How long the head keeps retrying a node that does not answer yet (still loading).
    const CONNECT_WAIT: Duration = Duration::from_secs(15 * 60);

    fn new_runtime() -> Backend {
        Backend::default()
    }

    pub fn dispatch(cmd: Cmd) -> Result<()> {
        match cmd {
            Cmd::Serve {
                listen,
                model,
                layers,
                role,
                downstream,
                max_seq,
                chunk,
                search_graphs,
                taps,
                tls,
            } => serve(
                listen,
                model,
                layers,
                role,
                downstream,
                max_seq,
                chunk,
                search_graphs,
                taps,
                tls.load()?,
            ),
            Cmd::Chat {
                chain,
                prompt,
                max_new,
                depth,
                tls,
            } => chat(chain, prompt, max_new, depth, tls.load()?),
            Cmd::Identity { .. } => unreachable!("handled before backend dispatch"),
        }
    }

    fn known_config(repo: &str) -> Result<QwenConfig> {
        config_for(repo)
            .ok_or_else(|| anyhow!("unknown model {repo} — known: Qwen/Qwen3-{{0.6B,1.7B,4B}}"))
    }

    /// Download and prepare `repo`'s weights (first run only); return the prepared file's path.
    fn weights_for(repo: &str) -> Result<String> {
        tracing::info!("preparing weights for {repo} (downloads + combines on first run)…");
        let dir = prepare_hf_model(repo).map_err(|e| anyhow!("prepare {repo}: {e}"))?;
        Ok(dir
            .join(COMBINED_WEIGHTS_FILE)
            .to_string_lossy()
            .into_owned())
    }

    /// Load layers `range` of `model` as one shard per stretch between tapped layers (a
    /// single shard when nothing is tapped).
    fn load_node(
        model: &str,
        range: (usize, usize),
        taps: &[usize],
        max_seq: usize,
        search_graphs: usize,
        chunk: usize,
    ) -> Result<TappedShards<Shard<Backend>>> {
        let cfg = known_config(model)?;
        if let Some(bad) = taps.iter().find(|&&layer| layer + 1 >= cfg.layers) {
            bail!(
                "layer {bad} cannot be tapped: taps must be below the last layer ({})",
                cfg.layers - 1
            );
        }
        let weights = weights_for(model)?;
        let ranges = tap_ranges(range, taps);
        let parts = ranges
            .iter()
            .map(|&(a, b)| {
                tracing::info!(
                    "loading shard [{a},{b}) of {model} on {BACKEND} — kernel search, takes \
                     minutes…"
                );
                Shard::load_chunked(
                    new_runtime(),
                    cfg,
                    (a, b),
                    a == 0,
                    b == cfg.layers,
                    &weights,
                    max_seq,
                    search_graphs,
                    chunk,
                )
            })
            .collect();
        TappedShards::new(parts, taps.to_vec())
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
        taps: Vec<usize>,
        tls: TlsIdentity,
    ) -> Result<()> {
        // Validate everything cheap before the download and kernel search.
        let cfg = known_config(&model)?;
        if a == 0 {
            bail!("layer 0 belongs to the head — run `potpori chat` on the head instead");
        }
        if b > cfg.layers {
            bail!("{model} has {} layers, got --layers {a}:{b}", cfg.layers);
        }
        if max_seq == 0 || max_seq > i32::MAX as usize || chunk > max_seq || search_graphs == 0 {
            bail!("invalid --max-seq, --chunk or --search-graphs");
        }
        match (role, downstream) {
            (Role::Tail, None) if b == cfg.layers => {}
            (Role::Tail, None) => bail!("the tail must end at the last layer ({})", cfg.layers),
            (Role::Tail, Some(_)) => bail!("--downstream is only for middles"),
            (Role::Middle, Some(_)) if b < cfg.layers => {}
            (Role::Middle, Some(_)) => {
                bail!("a middle cannot own the final layer — use --role tail")
            }
            (Role::Middle, None) => bail!("--downstream is required for middles"),
        }
        let mut shard = load_node(&model, (a, b), &taps, max_seq, search_graphs, chunk)?;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        let ep = {
            let _g = rt.enter();
            listen_endpoint(listen, &tls).map_err(|e| anyhow!("listen on {listen}: {e}"))?
        };
        tracing::info!(
            "shard ready — serving [{a},{b}) as {role:?} on {listen}, reporting layers {:?}",
            shard.tap_layers()
        );
        loop {
            shard.reset(); // fresh KV per session
            let res = match downstream {
                None => serve_tail(&mut shard, &ep, &rt),
                Some(next) => serve_middle(&mut shard, &ep, next, &rt, &tls),
            };
            match res {
                Ok(()) => tracing::info!("session ended cleanly — waiting for the next head"),
                Err(e) => tracing::warn!("session ended: {e} — waiting for the next head"),
            }
        }
    }

    fn chat(
        chain_path: PathBuf,
        prompt: String,
        max_new: usize,
        depth: Option<usize>,
        tls: TlsIdentity,
    ) -> Result<()> {
        let cc = ChainConfig::load(&chain_path)?;
        let chunk = cc.gamma + 1;
        let depth = depth.unwrap_or(cc.depth);
        if depth == 0 || depth > 64 {
            bail!("--depth must be between 1 and 64");
        }
        if max_new == 0 {
            return Ok(());
        }

        // Tokenize first (fast) so a request that cannot fit fails before the slow loads.
        let tok = ChatTokenizer::load(&cc.model)?;
        let ids = tok.encode_chat(&prompt)?;
        if ids
            .len()
            .checked_add(max_new)
            .is_none_or(|n| n > cc.max_seq)
        {
            bail!(
                "prompt ({} tokens) plus --max-new {max_new} exceeds max_seq {}; shorten the \
                 request or raise max_seq on every node",
                ids.len(),
                cc.max_seq
            );
        }

        let head_range = (cc.nodes[0].layers[0], cc.nodes[0].layers[1]);
        let next_addr: SocketAddr = cc.nodes[1]
            .addr
            .as_ref()
            .expect("validated: remote nodes have addrs")
            .parse()?;
        let connect = |head, prompt_tokens: usize| -> Result<ChainLink<_>> {
            tracing::info!("prompt = {prompt_tokens} tokens; connecting to {next_addr}…");
            let chain = ChainLink::connect(head, next_addr, &tls, &EOS_TOKENS, CONNECT_WAIT)?;
            tracing::info!(
                "chain ready — round trip to the next node {:.1} ms",
                chain.rtt().as_secs_f64() * 1e3
            );
            Ok(chain)
        };

        let (out, stats, dt, link, rtts, how): (Vec<_>, SpecStats, f64, _, _, String);
        if let Some(repo) = &cc.drafter {
            // A drafter that reads the target's hidden states: one block per trip.
            let (config, weights) =
                download_drafter(repo).map_err(|e| anyhow!("download {repo}: {e}"))?;
            let dcfg = DsparkConfig::from_json(&std::fs::read_to_string(config)?)?;
            dcfg.check_target(&known_config(&cc.model)?)?;
            if dcfg.block != cc.gamma {
                bail!(
                    "{repo} drafts {} tokens per block: set gamma = {} in the chain config \
                     (and --chunk {} on every node)",
                    dcfg.block,
                    dcfg.block,
                    dcfg.block + 1
                );
            }
            let taps: Vec<String> = dcfg.taps.iter().map(|l| l.to_string()).collect();
            tracing::info!(
                "{repo} reads target layers {:?}: every remote node must be started with \
                 --taps {}",
                dcfg.taps,
                taps.join(",")
            );
            let head = load_node(
                &cc.model,
                head_range,
                &dcfg.taps,
                cc.max_seq,
                cc.search_graphs,
                chunk,
            )?;
            tracing::info!("loading drafter {repo} on {BACKEND}…");
            let mut draft =
                DsparkDraft::load(new_runtime(), dcfg, &weights, cc.max_seq, cc.search_graphs)?;
            let mut chain = connect(head, ids.len())?;
            let rtt_at_connect = chain.rtt();
            let t0 = Instant::now();
            let generated = run_feature_decode(&mut draft, &mut chain, &ids, max_new);
            dt = t0.elapsed().as_secs_f64();
            chain.shutdown();
            (out, stats) = generated?;
            link = chain.stats();
            rtts = (rtt_at_connect, chain.rtt());
            let d = draft.stats();
            how = format!(
                "drafter {repo}: {} blocks, {:.1} ms graph + {:.1} ms markov each | {:.1} KiB \
                 of tapped layers per result",
                d.drafts,
                d.graph.as_secs_f64() * 1e3 / d.drafts.max(1) as f64,
                d.markov.as_secs_f64() * 1e3 / d.drafts.max(1) as f64,
                link.tap_bytes as f64 / link.frames.max(1) as f64 / 1024.0,
            );
        } else {
            tracing::info!("loading draft model {} on {BACKEND}…", cc.draft_model);
            let draft_target = QwenTarget::load_chunked(
                new_runtime(),
                known_config(&cc.draft_model)?,
                &weights_for(&cc.draft_model)?,
                cc.max_seq,
                cc.search_graphs,
                chunk, // kernels for {1, γ+1} tokens, as on the head and the remote shards
            );
            let mut draft = IncrementalDraft::new(draft_target, chunk);
            let head = load_node(
                &cc.model,
                head_range,
                &[],
                cc.max_seq,
                cc.search_graphs,
                chunk,
            )?;
            {
                // Feeds the prompt into the draft's KV (not wasted work) and fails here,
                // rather than mid-generation, if the draft's kernels produce non-finite
                // logits.
                use crate::engine::DraftEngine as _;
                let probe = draft.draft(&ids, 1);
                tracing::info!("draft model ready (first prediction = {})", probe[0]);
            }
            let mut chain = connect(head, ids.len())?;
            let rtt_at_connect = chain.rtt();
            let t0 = Instant::now();
            let generated = run_spec_decode_pipelined(
                &mut draft,
                &mut chain,
                &ids,
                &SpecConfig {
                    gamma: cc.gamma,
                    max_new,
                },
                depth,
            );
            dt = t0.elapsed().as_secs_f64();
            chain.shutdown();
            (out, stats) = generated?;
            link = chain.stats();
            rtts = (rtt_at_connect, chain.rtt());
            how = format!("draft model {}, depth={depth}", cc.draft_model);
        }

        println!("\n{}", tok.decode(&out)?);
        let secs = |d: Duration| d.as_secs_f64();
        let after_prefill = (dt - secs(link.prefill)).max(1e-9);
        eprintln!(
            "\n[{} tokens in {dt:.1}s = {:.1} tok/s overall, {:.1} tok/s after prefill | {:.2} \
             tok/traversal over {} traversals | γ={} chain={} nodes]",
            out.len(),
            out.len() as f64 / dt,
            out.len() as f64 / after_prefill,
            stats.toks_per_traversal(),
            stats.traversals,
            cc.gamma,
            cc.nodes.len(),
        );
        eprintln!(
            "[time: prefill {:.2}s ({} prompt tokens) | head shard {:.2}s | waiting on chain \
             {:.2}s | draft + rest {:.2}s]",
            secs(link.prefill),
            ids.len(),
            secs(link.head_compute),
            secs(link.waiting),
            (after_prefill - secs(link.head_compute) - secs(link.waiting)).max(0.0),
        );
        eprintln!(
            "[link: round trip {:.1} ms at connect, {:.1} ms now | {} frames, {:.1} KiB each | \
             head on {BACKEND}]",
            secs(rtts.0) * 1e3,
            secs(rtts.1) * 1e3,
            link.frames,
            link.payload_bytes as f64 / link.frames.max(1) as f64 / 1024.0,
        );
        eprintln!("[{how}]");
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
    fn serve_defaults_match_the_chain_config_defaults() {
        let serve = |extra: &[&str]| {
            let mut args = vec![
                "potpori",
                "serve",
                "--listen",
                "0.0.0.0:4010",
                "--model",
                "Qwen/Qwen3-4B",
            ];
            args.extend_from_slice(extra);
            Cli::try_parse_from(args)
        };
        let chain: chain::ChainConfig =
            toml::from_str(include_str!("../chain.example.toml")).unwrap();
        match serve(&["--layers", "24:36", "--role", "tail"]).unwrap().cmd {
            Cmd::Serve {
                layers,
                role,
                max_seq,
                chunk,
                ..
            } => {
                assert_eq!((layers, role), ((24, 36), Role::Tail));
                assert_eq!(max_seq, chain.max_seq);
                assert_eq!(chunk, chain.gamma + 1);
            }
            _ => panic!("expected serve"),
        }
        assert!(serve(&["--layers", "24:36", "--role", "head"]).is_err());
        assert!(serve(&["--layers", "36:24", "--role", "tail"]).is_err());
    }
}

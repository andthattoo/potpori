//! potpori — Nanbeige4.3-33B across your friends' NVIDIA GPUs.
//!
//! The model is cut into contiguous layer shards; each node serves one shard; the residual
//! stream hops node to node over QUIC. Generation is greedy speculative decoding: a DFlash2
//! drafter on the head proposes a block of tokens from the target's own hidden states, and
//! the chain verifies the block in one trip.
//!
//! Commands:
//!   - `potpori identity` — create this node's TLS identity.
//!   - `potpori serve`    — run a middle or tail shard on each friend's machine.
//!   - `potpori chat`     — the head: tokenizer, first shard and drafter, locally.
//!   - `potpori local`    — the whole model on one GPU, a few layers at a time (a check).
//!
//! Layout: [`chain`] is the config file, [`transport`] the wire, [`node`] the three roles,
//! [`specfeat`] the decoding driver, [`gpu`] the CUDA device and `cu/` its kernels, and
//! [`nanbeige`] the model.

mod chain;
mod engine;
mod gpu;
mod nanbeige;
mod node;
// The token-drafting drivers: generic, tested with synthetic engines, and not what `chat`
// uses with a drafter that reads hidden states.
#[allow(dead_code)]
mod specdecode;
mod specfeat;
#[allow(dead_code)]
mod specpipe;
mod transport;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use clap::{Parser, Subcommand, ValueEnum};

use chain::ChainConfig;
use gpu::{gpu, Buf};
use nanbeige::config::{Config, DraftConfig, Weights};
use nanbeige::dflash::Dflash;
use nanbeige::shard::Shard;
use nanbeige::stream::{LocalChain, Streamed};
use nanbeige::tok::ChatTokenizer;
use nanbeige::weights::Checkpoint;
use nanbeige::DRAFTER_REPO;
use node::{serve_middle, serve_tail, tap_ranges, ChainLink, ShardEngine as _, TappedShards};
use specfeat::run_feature_decode;
use transport::{listen_endpoint, TlsIdentity};

const CFG: Config = Config::NANBEIGE_4_3_33B;

/// How long the head keeps retrying a node that does not answer yet (still loading).
const CONNECT_WAIT: Duration = Duration::from_secs(30 * 60);

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
    fn load(&self) -> Result<TlsIdentity> {
        TlsIdentity::load(&self.cert, &self.key, &self.trust)
    }
}

#[derive(Parser)]
#[command(
    name = "potpori",
    version,
    about = "Nanbeige4.3-33B sharded across friends' NVIDIA GPUs over WAN"
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
        /// Global layer range to serve as start:end, e.g. 12:24
        #[arg(long, value_parser = parse_layers)]
        layers: (usize, usize),
        /// Role in the chain: tail (last shard, owns the output head) or middle
        #[arg(long, value_enum)]
        role: Role,
        /// Downstream node address (required for middles)
        #[arg(long)]
        downstream: Option<SocketAddr>,
        /// Max sequence length — sizes the KV cache (must equal the chain config's max_seq)
        #[arg(long, default_value_t = chain::DEFAULT_MAX_SEQ)]
        max_seq: usize,
        /// Layers whose outputs go back to the head with each result: the ones the drafter
        /// reads. Must be the same on every node.
        #[arg(long, value_delimiter = ',', default_values_t = DraftConfig::nanbeige_4_3_33b().taps)]
        taps: Vec<usize>,
        /// Which weights to serve; every node of a chain must use the same.
        #[arg(long, value_enum, default_value_t = Weights::Nvfp4)]
        weights: Weights,
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
        /// Let the model reason in a <think> block before it answers.
        #[arg(long)]
        think: bool,
        /// Which weights to run; every node of the chain must use the same.
        #[arg(long, value_enum, default_value_t = Weights::Nvfp4)]
        weights: Weights,
    },
    /// Run the whole model on this one GPU by loading a few layers at a time: slow (the
    /// checkpoint is read from disk for every chunk), but it checks the real weights and the
    /// drafter end to end without friends.
    Local {
        /// The user message
        #[arg(long)]
        prompt: String,
        /// Max new tokens
        #[arg(long, default_value_t = 32)]
        max_new: usize,
        /// Max sequence length — sizes the KV caches
        #[arg(long, default_value_t = 1024)]
        max_seq: usize,
        /// Layers held on the GPU at once, next to 7 GB that stays loaded: 0.33 GB each in
        /// 4 bits (default 16), 1.2 GB each in bf16 (default 4)
        #[arg(long)]
        group: Option<usize>,
        /// Let the model reason in a <think> block before it answers.
        #[arg(long)]
        think: bool,
        /// Which weights to run.
        #[arg(long, value_enum, default_value_t = Weights::Nvfp4)]
        weights: Weights,
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
    match Cli::parse().cmd {
        Cmd::Identity { out } => {
            TlsIdentity::generate(&out)?;
            println!(
                "Identity created in {}. Exchange cert.pem with adjacent peers and collect \
                 their certificates in peers.pem.",
                out.display()
            );
            Ok(())
        }
        Cmd::Serve {
            listen,
            layers,
            role,
            downstream,
            max_seq,
            taps,
            weights,
            tls,
        } => serve(
            listen,
            layers,
            role,
            downstream,
            max_seq,
            taps,
            weights,
            tls.load()?,
        ),
        Cmd::Chat {
            chain,
            prompt,
            max_new,
            think,
            weights,
            tls,
        } => chat(chain, prompt, max_new, think, weights, tls.load()?),
        Cmd::Local {
            prompt,
            max_new,
            max_seq,
            group,
            think,
            weights,
        } => local(prompt, max_new, max_seq, group, think, weights),
    }
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// Download what layers `range` need (first run only) and load them as one shard per stretch
/// between tapped layers. `extra` names further target tensors to fetch with them.
fn load_node(
    range: (usize, usize),
    taps: &[usize],
    max_seq: usize,
    extra: &[&str],
    weights: Weights,
) -> Result<(Vec<Shard>, Checkpoint)> {
    let repo = weights.repo();
    if let Some(bad) = taps.iter().find(|&&layer| layer + 1 >= CFG.layers) {
        bail!(
            "layer {bad} cannot be tapped: taps must be below the last layer ({})",
            CFG.layers - 1
        );
    }
    let mut names = CFG.tensor_names(range, weights);
    names.extend(extra.iter().map(|n| n.to_string()));
    tracing::info!(
        "fetching layers [{}, {}) of {repo} (downloads on first run)…",
        range.0,
        range.1
    );
    let ck = Checkpoint::download(repo, &names)?;
    let t0 = Instant::now();
    let parts = tap_ranges(range, taps)
        .into_iter()
        .map(|r| Shard::load(CFG, r, &ck, max_seq))
        .collect::<Result<Vec<_>>>()?;
    let (free, total) = gpu().memory();
    tracing::info!(
        "layers [{}, {}) on {} in {:.1} s — {:.1} of {:.1} GiB free",
        range.0,
        range.1,
        gpu().name(),
        t0.elapsed().as_secs_f64(),
        gib(free),
        gib(total)
    );
    Ok((parts, ck))
}

#[allow(clippy::too_many_arguments)]
fn serve(
    listen: SocketAddr,
    (a, b): (usize, usize),
    role: Role,
    downstream: Option<SocketAddr>,
    max_seq: usize,
    taps: Vec<usize>,
    weights: Weights,
    tls: TlsIdentity,
) -> Result<()> {
    // Validate everything cheap before the download.
    if a == 0 {
        bail!("layer 0 belongs to the head — run `potpori chat` on the head instead");
    }
    if b > CFG.layers {
        bail!("the model has {} layers, got --layers {a}:{b}", CFG.layers);
    }
    if max_seq == 0 || max_seq > i32::MAX as usize {
        bail!("invalid --max-seq");
    }
    match (role, downstream) {
        (Role::Tail, None) if b == CFG.layers => {}
        (Role::Tail, None) => bail!("the tail must end at the last layer ({})", CFG.layers),
        (Role::Tail, Some(_)) => bail!("--downstream is only for middles"),
        (Role::Middle, Some(_)) if b < CFG.layers => {}
        (Role::Middle, Some(_)) => bail!("a middle cannot own the final layer — use --role tail"),
        (Role::Middle, None) => bail!("--downstream is required for middles"),
    }
    let (parts, _) = load_node((a, b), &taps, max_seq, &[], weights)?;
    let mut shard = TappedShards::new(parts, taps)?;
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
        shard.reset();
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
    think: bool,
    weights: Weights,
    tls: TlsIdentity,
) -> Result<()> {
    let cc = ChainConfig::load(&chain_path)?;
    if max_new == 0 {
        return Ok(());
    }
    let dcfg = DraftConfig::nanbeige_4_3_33b();

    // Tokenize first (fast) so a request that cannot fit fails before the slow loads.
    let tok = ChatTokenizer::load(weights.repo())?;
    let ids = tok.encode_chat(&prompt, think)?;
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

    // The head: its layers, and the drafter with the target's embedding and output head.
    let (parts, target) = load_node(
        head_range,
        &dcfg.taps,
        cc.max_seq,
        &["lm_head.weight"],
        weights,
    )?;
    let embed = parts[0].embedding().expect("the head owns layer 0");
    let lm_head = Arc::new(Buf::from_slice(
        &target.bf16("lm_head.weight", &[CFG.vocab, CFG.hidden])?,
    ));
    tracing::info!("loading drafter {DRAFTER_REPO}…");
    let names: Vec<String> = dcfg.parameters().into_iter().map(|(n, _)| n).collect();
    let mut draft = Dflash::load(
        dcfg.clone(),
        &Checkpoint::download(DRAFTER_REPO, &names)?,
        embed,
        lm_head,
        cc.max_seq,
    )?;
    let head = TappedShards::new(parts, dcfg.taps.clone())?;

    tracing::info!("prompt = {} tokens; connecting to {next_addr}…", ids.len());
    let mut chain = ChainLink::connect(head, next_addr, &tls, &tok.eos_tokens(), CONNECT_WAIT)?;
    let rtt_at_connect = chain.rtt();
    tracing::info!(
        "chain ready — round trip to the next node {:.1} ms",
        rtt_at_connect.as_secs_f64() * 1e3
    );

    let t0 = Instant::now();
    let generated = run_feature_decode(&mut draft, &mut chain, &ids, max_new);
    let dt = t0.elapsed().as_secs_f64();
    chain.shutdown();
    let (out, stats) = generated?;
    let link = chain.stats();
    let d = draft.stats();

    println!("\n{}", tok.decode(&out)?);
    let secs = |d: Duration| d.as_secs_f64();
    let after_prefill = (dt - secs(link.prefill)).max(1e-9);
    eprintln!(
        "\n[{} tokens in {dt:.1}s = {:.1} tok/s overall, {:.1} tok/s after prefill | {:.2} \
         tokens per trip over {} trips | chain of {} nodes]",
        out.len(),
        out.len() as f64 / dt,
        out.len() as f64 / after_prefill,
        stats.toks_per_traversal(),
        stats.traversals,
        cc.nodes.len(),
    );
    eprintln!(
        "[time: prefill {:.2}s ({} prompt tokens) | head shard {:.2}s | waiting on chain \
         {:.2}s | drafter {:.2}s]",
        secs(link.prefill),
        ids.len(),
        secs(link.head_compute),
        secs(link.waiting),
        secs(d.gpu + d.host),
    );
    eprintln!(
        "[link: round trip {:.1} ms at connect, {:.1} ms now | {} frames, {:.1} KiB each, \
         {:.1} KiB of tapped layers per result | head on {}]",
        secs(rtt_at_connect) * 1e3,
        secs(chain.rtt()) * 1e3,
        link.frames,
        link.payload_bytes as f64 / link.frames.max(1) as f64 / 1024.0,
        link.tap_bytes as f64 / link.frames.max(1) as f64 / 1024.0,
        gpu().name(),
    );
    eprintln!(
        "[drafter: {} blocks, {:.1} ms on the GPU + {:.1} ms selecting each]",
        d.drafts,
        secs(d.gpu) * 1e3 / d.drafts.max(1) as f64,
        secs(d.host) * 1e3 / d.drafts.max(1) as f64,
    );
    Ok(())
}

fn local(
    prompt: String,
    max_new: usize,
    max_seq: usize,
    group: Option<usize>,
    think: bool,
    weights: Weights,
) -> Result<()> {
    let group = group.unwrap_or(match weights {
        Weights::Nvfp4 => 16,
        Weights::Bf16 => 4,
    });
    let repo = weights.repo();
    if max_new == 0 || group == 0 {
        bail!("--max-new and --group must be positive");
    }
    let dcfg = DraftConfig::nanbeige_4_3_33b();
    let tok = ChatTokenizer::load(weights.repo())?;
    let ids = tok.encode_chat(&prompt, think)?;
    if ids.len() + max_new > max_seq {
        bail!(
            "prompt ({} tokens) plus --max-new {max_new} exceeds --max-seq {max_seq}",
            ids.len()
        );
    }
    tracing::info!("fetching {repo} (the whole checkpoint on first run)…");
    let all = CFG.tensor_names((0, CFG.layers), weights);
    let target = Arc::new(Checkpoint::download(repo, &all)?);
    let table = |name: &str| -> Result<Arc<Buf<u16>>> {
        Ok(Arc::new(Buf::from_slice(
            &target.bf16(name, &[CFG.vocab, CFG.hidden])?,
        )))
    };
    let (embed, lm_head) = (
        table("model.embed_tokens.weight")?,
        table("lm_head.weight")?,
    );
    tracing::info!("loading drafter {DRAFTER_REPO}…");
    let names: Vec<String> = dcfg.parameters().into_iter().map(|(n, _)| n).collect();
    let mut draft = Dflash::load(
        dcfg.clone(),
        &Checkpoint::download(DRAFTER_REPO, &names)?,
        embed.clone(),
        lm_head.clone(),
        max_seq,
    )?;
    let (free, total) = gpu().memory();
    tracing::info!(
        "embedding, output head and drafter on {} — {:.1} of {:.1} GiB free for {group} layers \
         at a time",
        gpu().name(),
        gib(free),
        gib(total)
    );
    let streamed = Streamed::new(CFG, target, max_seq, dcfg.taps, group, embed, lm_head);
    let mut chain = LocalChain::new(streamed, &tok.eos_tokens());
    chain.verbose = true;

    let t0 = Instant::now();
    let (out, stats) = run_feature_decode(&mut draft, &mut chain, &ids, max_new)?;
    let dt = t0.elapsed().as_secs_f64();
    let d = draft.stats();
    println!("\n{}", tok.decode(&out)?);
    eprintln!(
        "\n[{} tokens in {dt:.0}s | {:.2} tokens per trip over {} trips | drafter: {:.1} ms on \
         the GPU + {:.1} ms selecting per block]",
        out.len(),
        stats.toks_per_traversal(),
        stats.traversals,
        d.gpu.as_secs_f64() * 1e3 / d.drafts.max(1) as f64,
        d.host.as_secs_f64() * 1e3 / d.drafts.max(1) as f64,
    );
    eprintln!("[token ids: {out:?}]");
    Ok(())
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
    fn serve_defaults_match_the_chain_config_and_the_drafter() {
        let serve = |extra: &[&str]| {
            let mut args = vec!["potpori", "serve", "--listen", "0.0.0.0:4010"];
            args.extend_from_slice(extra);
            Cli::try_parse_from(args)
        };
        let chain: ChainConfig = toml::from_str(include_str!("../chain.example.toml")).unwrap();
        match serve(&["--layers", "40:52", "--role", "tail"]).unwrap().cmd {
            Cmd::Serve {
                layers,
                role,
                max_seq,
                taps,
                ..
            } => {
                assert_eq!((layers, role), ((40, 52), Role::Tail));
                assert_eq!(max_seq, chain.max_seq);
                assert_eq!(taps, DraftConfig::nanbeige_4_3_33b().taps);
            }
            _ => panic!("expected serve"),
        }
        assert!(serve(&["--layers", "40:52", "--role", "head"]).is_err());
        assert!(serve(&["--layers", "52:40", "--role", "tail"]).is_err());
    }
}

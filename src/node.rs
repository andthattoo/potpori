//! The chain protocol: what a node does with the wire in [`crate::transport`].
//!
//!   - **Head** ([`ChainLink`]): runs the first shard locally, ships the residual stream to the
//!     next node, and reads the tail's per-position argmax. It implements
//!     [`PipelinedChain`], so the spec-decode driver drives the distributed model directly.
//!   - **Middle** ([`serve_middle`]): runs its shard and forwards downstream, relaying results
//!     (and errors) back up.
//!   - **Tail** ([`serve_tail`]): runs the last shard and answers with the argmax of its logits.
//!
//! Every session starts with a handshake ([`Hello`]): the head sends the whole plan, each node
//! checks its own seat against how it was started and forwards it downstream. A misconfigured
//! chain therefore fails at connect, naming the node and the reason. After that, any failure
//! travels upstream as [`PeerMessage::Error`] and surfaces on the head as an `Err`.
//!
//! Everything here is generic over [`ShardEngine`], so the whole protocol is tested over real
//! QUIC with a synthetic shard — no GPU, no model (see the tests at the bottom).
//!
//! Threading: the Luminal runtime is `!Send` and the spec-decode driver is synchronous, while
//! QUIC is async. Each node therefore owns its shard on one OS thread and `block_on`s the async
//! hops there; the QUIC I/O driver runs on the tokio runtime's worker thread(s), so the
//! connection keeps moving (sends flush, keepalives fire) while the shard computes.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use quinn::Endpoint;
use tokio::runtime::Runtime;

use crate::engine::{argmax, TokenId};
use crate::specpipe::PipelinedChain;
use crate::transport::{
    accept, dial, dial_endpoint, Hello, HiddenStateFrame, NodeInfo, PeerConnection, PeerMessage,
    VerifyResult, PROTO_VERSION,
};

/// How long a node waits for the first message of a session, and for a downstream handshake.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
/// How long one dial attempt may take before it counts as "not reachable (yet)".
const DIAL_TIMEOUT: Duration = Duration::from_secs(10);
/// How long to wait for the peer to close after we said goodbye.
const CLOSE_TIMEOUT: Duration = Duration::from_secs(3);

/// The compute a node contributes: a contiguous block of layers with its own KV cache.
/// Implemented by the real Luminal shard ([`crate::qwen::shard::Shard`]) and by the synthetic
/// shard in this module's tests.
pub trait ShardEngine {
    fn hidden_dim(&self) -> usize;
    fn vocab_size(&self) -> usize;
    /// Whether a chunk of `s` positions starting at absolute `start_pos` can run here (a kernel
    /// exists for `s`; every position fits the KV cache). Checked on every inbound frame.
    fn check_chunk(&self, start_pos: u32, s: usize) -> Result<(), String>;
    /// First shard: embed `tokens` at absolute positions from `start_pos`, run the block, return
    /// the residual stream `[s, hidden]` flat.
    fn forward_tokens(&mut self, start_pos: u32, tokens: &[TokenId]) -> Vec<f32>;
    /// Non-first shard: consume `[s, hidden]`, run the block, return `[s, hidden]` — or
    /// `[s, vocab]` logits if this is the last shard.
    fn forward_hidden(&mut self, start_pos: u32, s: usize, hidden: Vec<f32>) -> Vec<f32>;
    /// Clear the KV cache (between sessions).
    fn reset(&mut self);
}

/// How a remote node was started — what it checks an incoming [`Hello`] against.
#[derive(Clone, Debug)]
pub struct NodeSpec {
    pub model: String,
    /// Global layer range `[start, end)` this node serves.
    pub layers: (usize, usize),
    pub is_tail: bool,
    /// Kernel chunk size: 0 = generic kernel (any chunk size), else frames must carry 1 or
    /// `chunk` tokens.
    pub chunk: usize,
    pub max_seq: usize,
    /// Shared secret a head must present; `None` = open to anyone who can reach the port.
    pub token: Option<String>,
    /// e.g. "cuda/Bf16" — reported to the head, for the record.
    pub backend: String,
}

impl NodeSpec {
    fn label(&self) -> String {
        let role = if self.is_tail { "tail" } else { "middle" };
        format!("{role} [{}, {})", self.layers.0, self.layers.1)
    }

    fn info(&self) -> NodeInfo {
        NodeInfo {
            layers: [self.layers.0 as u32, self.layers.1 as u32],
            backend: self.backend.clone(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Check an upstream [`Hello`] against this node. The error is sent back to the head.
    pub fn check_hello(&self, hidden_dim: usize, hello: &Hello) -> Result<(), String> {
        if hello.proto != PROTO_VERSION {
            return Err(format!(
                "protocol version mismatch: head speaks v{}, this node v{PROTO_VERSION} — \
                 rebuild both from the same commit",
                hello.proto
            ));
        }
        if let Some(expected) = &self.token {
            if !token_matches(expected, hello.token.as_deref()) {
                return Err("bad or missing token (this node was started with --token)".into());
            }
        }
        if hello.model != self.model {
            return Err(format!("serving model {:?}, but the chain wants {:?}", self.model, hello.model));
        }
        let seat = hello.index as usize;
        if seat == 0 || seat >= hello.plan.len() {
            return Err(format!("seat {seat} is not a remote seat in a {}-node plan", hello.plan.len()));
        }
        let [a, b] = hello.plan[seat];
        if (a as usize, b as usize) != self.layers {
            return Err(format!(
                "serving layers {}:{}, but the chain config puts {a}:{b} at this address",
                self.layers.0, self.layers.1
            ));
        }
        let last = seat == hello.plan.len() - 1;
        if last != self.is_tail {
            return Err(if last {
                "started with --role middle, but the chain config has this node last (use --role tail)".into()
            } else {
                "started with --role tail, but the chain config has more nodes after this one".to_string()
            });
        }
        if self.chunk != 0 && hello.chunk as usize != self.chunk {
            return Err(format!(
                "started with --chunk {}, but the chain's gamma+1 is {}",
                self.chunk, hello.chunk
            ));
        }
        if hello.max_seq as usize > self.max_seq {
            return Err(format!(
                "started with --max-seq {}, but the chain needs {}",
                self.max_seq, hello.max_seq
            ));
        }
        if hello.hidden_dim as usize != hidden_dim {
            return Err(format!("hidden size {hidden_dim}, but the head sends {}", hello.hidden_dim));
        }
        Ok(())
    }
}

/// Constant-time-ish comparison (no early exit on the first differing byte).
fn token_matches(expected: &str, got: Option<&str>) -> bool {
    let got = got.unwrap_or("");
    let mut diff = expected.len() ^ got.len();
    for (a, b) in expected.bytes().zip(got.bytes()) {
        diff |= (a ^ b) as usize;
    }
    diff == 0
}

fn kind(msg: &PeerMessage) -> &'static str {
    match msg {
        PeerMessage::Hello(_) => "Hello",
        PeerMessage::HelloAck(_) => "HelloAck",
        PeerMessage::Hidden(_) => "Hidden",
        PeerMessage::Result(_) => "Result",
        PeerMessage::Error(_) => "Error",
    }
}

/// Validate an inbound (untrusted) hidden-state frame against this shard before it reaches the
/// GPU: codec, shape, payload length, kernel bucket and KV bounds.
fn check_frame<S: ShardEngine>(shard: &S, frame: &HiddenStateFrame) -> Result<(), String> {
    if frame.dtype != "f32" {
        return Err(format!("bad frame: unsupported codec {:?} (only f32)", frame.dtype));
    }
    let hidden = shard.hidden_dim();
    if frame.hidden_dim as usize != hidden {
        return Err(format!("bad frame: hidden_dim {} != this shard's {hidden}", frame.hidden_dim));
    }
    let s = frame.n_tokens as usize;
    shard.check_chunk(frame.start_pos, s).map_err(|e| format!("bad frame: {e}"))?;
    let want = s * hidden * 4;
    if frame.data.len() != want {
        return Err(format!(
            "bad frame: payload is {} bytes, expected {want} for {s} tokens",
            frame.data.len()
        ));
    }
    Ok(())
}

/// Tell upstream why the session is over, give it a moment to read that, and return the same
/// reason as this side's error.
async fn reject(up: &mut PeerConnection, why: String) -> anyhow::Error {
    let _ = up.send(&PeerMessage::Error(why.clone())).await;
    up.finish();
    let _ = tokio::time::timeout(CLOSE_TIMEOUT, up.wait_closed()).await;
    anyhow!(why)
}

/// Read and check the session's first message.
async fn read_hello(up: &mut PeerConnection, spec: &NodeSpec, hidden_dim: usize) -> Result<Hello, String> {
    let first = tokio::time::timeout(HANDSHAKE_TIMEOUT, up.recv())
        .await
        .map_err(|_| "no Hello within the handshake timeout".to_string())?;
    match first {
        Ok(Some(PeerMessage::Hello(hello))) => spec.check_hello(hidden_dim, &hello).map(|()| hello),
        Ok(Some(other)) => Err(format!("expected Hello first, got {}", kind(&other))),
        Ok(None) => Err("upstream closed before saying Hello".into()),
        Err(e) => Err(format!("unreadable first message (different potpori version?): {e}")),
    }
}

enum HandshakeError {
    /// Could not connect at all — the node may still be loading; worth retrying.
    Unreachable(String),
    /// Connected, and the chain said no. Retrying will not help.
    Rejected(String),
}

/// Dial `addr`, send `hello`, and wait for the chain below to accept it.
async fn handshake(
    ep: &Endpoint,
    addr: SocketAddr,
    hello: Hello,
) -> Result<(PeerConnection, Vec<NodeInfo>), HandshakeError> {
    let mut peer = match tokio::time::timeout(DIAL_TIMEOUT, dial(ep, addr)).await {
        Ok(Ok(peer)) => peer,
        Ok(Err(e)) => return Err(HandshakeError::Unreachable(e.to_string())),
        Err(_) => return Err(HandshakeError::Unreachable("no answer".into())),
    };
    let lost = |what: String| HandshakeError::Rejected(format!("handshake with {addr} failed: {what}"));
    peer.send(&PeerMessage::Hello(hello)).await.map_err(|e| lost(e.to_string()))?;
    // A middle only answers once everything below it has answered, so allow a few hops.
    match tokio::time::timeout(HANDSHAKE_TIMEOUT * 3, peer.recv()).await {
        Ok(Ok(Some(PeerMessage::HelloAck(nodes)))) => Ok((peer, nodes)),
        Ok(Ok(Some(PeerMessage::Error(why)))) => Err(HandshakeError::Rejected(why)),
        Ok(Ok(Some(other))) => Err(lost(format!("unexpected {}", kind(&other)))),
        Ok(Ok(None)) => Err(lost("closed without answering (different potpori version?)".into())),
        Ok(Err(e)) => Err(lost(e.to_string())),
        Err(_) => Err(lost("timed out".into())),
    }
}

/// Serve one session as the tail: accept a peer, handshake, then answer each hidden-state frame
/// with the argmax of this shard's logits until the head closes. `Ok(())` = clean end.
pub fn serve_tail<S: ShardEngine>(
    shard: &mut S,
    spec: &NodeSpec,
    endpoint: &Endpoint,
    rt: &Runtime,
) -> Result<()> {
    assert!(spec.is_tail, "serve_tail needs a tail spec");
    let vocab = shard.vocab_size();
    let label = spec.label();
    rt.block_on(async {
        let mut up = accept(endpoint).await?;
        let hello = match read_hello(&mut up, spec, shard.hidden_dim()).await {
            Ok(hello) => hello,
            Err(why) => return Err(reject(&mut up, format!("{label}: {why}")).await),
        };
        up.send(&PeerMessage::HelloAck(vec![spec.info()])).await?;
        tracing::info!("session started ({}-node chain)", hello.plan.len());

        loop {
            let frame = match up.recv().await? {
                None => return Ok(()), // clean EOF from the head; dropping `up` closes back
                Some(PeerMessage::Hidden(frame)) => frame,
                Some(other) => {
                    let why = format!("{label}: unexpected {} mid-session", kind(&other));
                    return Err(reject(&mut up, why).await);
                }
            };
            if let Err(why) = check_frame(shard, &frame) {
                return Err(reject(&mut up, format!("{label}: {why}")).await);
            }
            let s = frame.n_tokens as usize;
            let logits = shard.forward_hidden(frame.start_pos, s, frame.to_f32()); // sync GPU
            if logits.len() != s * vocab {
                let why = format!("{label}: shard produced {} logits for {s} tokens", logits.len());
                return Err(reject(&mut up, why).await);
            }
            let argmax = (0..s).map(|j| argmax(&logits[j * vocab..(j + 1) * vocab])).collect();
            up.send(&PeerMessage::Result(VerifyResult { step: frame.step, argmax })).await?;
        }
    })
}

/// Serve one session as a middle: accept the upstream peer, handshake, bring up the downstream
/// link, then for each frame run the local shard, forward downstream and relay the result back
/// up. A chain of any length is head → middle → … → middle → tail.
pub fn serve_middle<S: ShardEngine>(
    shard: &mut S,
    spec: &NodeSpec,
    upstream: &Endpoint,
    downstream_addr: SocketAddr,
    rt: &Runtime,
) -> Result<()> {
    assert!(!spec.is_tail, "serve_middle needs a middle spec");
    let hidden_dim = shard.hidden_dim();
    let label = spec.label();
    rt.block_on(async {
        let mut up = accept(upstream).await?;
        let hello = match read_hello(&mut up, spec, hidden_dim).await {
            Ok(hello) => hello,
            Err(why) => return Err(reject(&mut up, format!("{label}: {why}")).await),
        };

        // Bring up the rest of the chain before accepting the plan ourselves, so the head hears
        // about a problem anywhere below us at connect time.
        let down_ep = dial_endpoint()?;
        let next = Hello { index: hello.index + 1, ..hello.clone() };
        let (mut down, below) = match handshake(&down_ep, downstream_addr, next).await {
            Ok(ok) => ok,
            Err(HandshakeError::Unreachable(e)) => {
                let why = format!("{label}: cannot reach its downstream {downstream_addr}: {e}");
                return Err(reject(&mut up, why).await);
            }
            // Already labelled by the node that said no.
            Err(HandshakeError::Rejected(why)) => return Err(reject(&mut up, why).await),
        };
        let mut nodes = vec![spec.info()];
        nodes.extend(below);
        up.send(&PeerMessage::HelloAck(nodes)).await?;
        tracing::info!("session started ({}-node chain)", hello.plan.len());

        loop {
            let frame = match up.recv().await? {
                None => break,
                Some(PeerMessage::Hidden(frame)) => frame,
                Some(other) => {
                    let why = format!("{label}: unexpected {} mid-session", kind(&other));
                    return Err(reject(&mut up, why).await);
                }
            };
            if let Err(why) = check_frame(shard, &frame) {
                return Err(reject(&mut up, format!("{label}: {why}")).await);
            }
            let s = frame.n_tokens as usize;
            let out = shard.forward_hidden(frame.start_pos, s, frame.to_f32()); // sync GPU
            let fwd = HiddenStateFrame::from_f32(frame.step, frame.start_pos, s as u32, hidden_dim as u32, &out);
            if let Err(e) = down.send(&PeerMessage::Hidden(fwd)).await {
                let why = format!("{label}: lost its downstream {downstream_addr}: {e}");
                return Err(reject(&mut up, why).await);
            }
            let failure = match down.recv().await {
                Ok(Some(PeerMessage::Result(result))) => {
                    up.send(&PeerMessage::Result(result)).await?;
                    continue;
                }
                Ok(Some(PeerMessage::Error(why))) => why, // labelled by its origin
                Ok(Some(other)) => format!("{label}: downstream sent unexpected {}", kind(&other)),
                Ok(None) => format!("{label}: downstream {downstream_addr} closed mid-session"),
                Err(e) => format!("{label}: lost its downstream {downstream_addr}: {e}"),
            };
            return Err(reject(&mut up, failure).await);
        }
        // Clean EOF from upstream → propagate the FIN downstream, then return.
        down.finish();
        let _ = tokio::time::timeout(CLOSE_TIMEOUT, down.wait_closed()).await;
        Ok(())
    })
}

/// Where a generation's wall-clock went, as seen from the head.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinkStats {
    /// QUIC's round-trip estimate to the next node right after the handshake.
    pub rtt: Duration,
    /// Prompt prefill: head compute + every prefill round trip.
    pub prefill: Duration,
    /// Head shard compute for verify chunks.
    pub head_compute: Duration,
    /// Time blocked waiting for verify results (the chain's latency, net of overlap).
    pub waiting: Duration,
    /// Hidden-state frames sent, and their payload bytes.
    pub frames: u64,
    pub payload_bytes: u64,
}

/// The head's link to the chain: the local first shard plus one persistent QUIC bi-stream to
/// the next node. `submit` fires a hidden-state frame downstream without waiting; `poll` reads
/// results in FIFO order (one bi-stream, processed in order by every node), so the head's
/// compute for chunk k+1 overlaps the rest of the chain's work on chunk k.
pub struct ChainLink<S: ShardEngine> {
    head: S,
    hidden_dim: usize,
    rt: Runtime,
    // Held to keep the client side driving I/O; dropping it would stall the connection.
    _endpoint: Endpoint,
    peer: PeerConnection,
    next_step: u64,
    /// `(step, n_tokens)` of every frame sent and not yet answered, oldest first.
    awaiting: VecDeque<(u64, usize)>,
    /// 0 = send the whole prompt as one frame (generic kernels); >0 = prefill in frames of
    /// this many tokens plus a single-token remainder (chunk-specialized kernels).
    prefill_chunk: usize,
    eos: Vec<TokenId>,
    nodes: Vec<NodeInfo>,
    stats: LinkStats,
}

impl<S: ShardEngine> ChainLink<S> {
    /// Dial the next node and run the handshake. `hello.index` must be 1 (the first remote).
    /// Remote nodes come up at different times (their kernel search takes minutes), so an
    /// unreachable node is retried for up to `wait`; a node that *answers* with a rejection
    /// fails immediately.
    pub fn connect(head: S, addr: SocketAddr, hello: Hello, eos: &[TokenId], wait: Duration) -> Result<Self> {
        let hidden_dim = head.hidden_dim();
        let prefill_chunk = hello.chunk as usize;
        // One worker thread drives QUIC while this thread is busy in the shard: frames leave
        // the moment they are submitted and keepalives keep firing during long computes. (A
        // current-thread runtime only makes progress inside `block_on`.)
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build()?;
        let (endpoint, peer, nodes) = rt.block_on(async {
            let ep = dial_endpoint()?;
            let deadline = Instant::now() + wait;
            let mut attempt = 0u32;
            loop {
                match handshake(&ep, addr, hello.clone()).await {
                    Ok((peer, nodes)) => return Ok((ep, peer, nodes)),
                    Err(HandshakeError::Rejected(why)) => bail!("the chain refused the session — {why}"),
                    Err(HandshakeError::Unreachable(e)) => {
                        if Instant::now() >= deadline {
                            bail!("could not reach {addr} within {}s: {e}", wait.as_secs());
                        }
                        if attempt % 6 == 0 {
                            tracing::info!("waiting for {addr} (still loading?): {e}");
                        }
                        attempt += 1;
                        tokio::time::sleep(Duration::from_secs(2).min(wait)).await;
                    }
                }
            }
        })?;
        let stats = LinkStats { rtt: peer.rtt(), ..LinkStats::default() };
        Ok(Self {
            head,
            hidden_dim,
            rt,
            _endpoint: endpoint,
            peer,
            next_step: 0,
            awaiting: VecDeque::new(),
            prefill_chunk,
            eos: eos.to_vec(),
            nodes,
            stats,
        })
    }

    /// The remote nodes, in pipeline order, as they described themselves in the handshake.
    pub fn nodes(&self) -> &[NodeInfo] {
        &self.nodes
    }

    pub fn stats(&self) -> LinkStats {
        self.stats
    }

    /// QUIC's current round-trip estimate to the next node.
    pub fn rtt(&self) -> Duration {
        self.peer.rtt()
    }

    /// Run `tokens` through the head shard and send the residual stream downstream.
    fn send_chunk(&mut self, start_pos: u32, tokens: &[TokenId]) -> Result<()> {
        let s = tokens.len();
        self.head.check_chunk(start_pos, s).map_err(|e| anyhow!("head shard: {e}"))?;
        let hidden = self.head.forward_tokens(start_pos, tokens); // sync GPU
        let step = self.next_step;
        self.next_step += 1;
        let frame = HiddenStateFrame::from_f32(step, start_pos, s as u32, self.hidden_dim as u32, &hidden);
        self.stats.frames += 1;
        self.stats.payload_bytes += frame.data.len() as u64;
        let peer = &mut self.peer;
        self.rt
            .block_on(peer.send(&PeerMessage::Hidden(frame)))
            .map_err(|e| anyhow!("lost the chain while sending: {e}"))?;
        self.awaiting.push_back((step, s));
        Ok(())
    }

    /// Block for the oldest outstanding frame's result.
    fn recv_result(&mut self) -> Result<Vec<TokenId>> {
        let Some((step, s)) = self.awaiting.pop_front() else {
            bail!("poll with no chunk in flight");
        };
        let peer = &mut self.peer;
        let result = match self.rt.block_on(peer.recv()) {
            Ok(Some(PeerMessage::Result(result))) => result,
            Ok(Some(PeerMessage::Error(why))) => bail!("the chain failed — {why}"),
            Ok(Some(other)) => bail!("the chain sent an unexpected {}", kind(&other)),
            Ok(None) => bail!("the chain closed the session mid-generation"),
            Err(e) => bail!("lost the chain: {e}"),
        };
        if result.step != step || result.argmax.len() != s {
            bail!(
                "the chain answered out of order or short (step {} with {} tokens; expected step \
                 {step} with {s})",
                result.step,
                result.argmax.len()
            );
        }
        Ok(result.argmax)
    }

    /// End the session cleanly: finish the send half (every node sees EOF and returns), then
    /// wait briefly for the chain to close back.
    pub fn shutdown(&mut self) {
        let peer = &mut self.peer;
        self.rt.block_on(async {
            peer.finish();
            let _ = tokio::time::timeout(CLOSE_TIMEOUT, peer.wait_closed()).await;
        });
    }
}

impl<S: ShardEngine> PipelinedChain for ChainLink<S> {
    fn prefill(&mut self, prompt: &[TokenId]) -> Result<TokenId> {
        if prompt.is_empty() {
            bail!("empty prompt");
        }
        let t0 = Instant::now();
        let chunk = if self.prefill_chunk == 0 { prompt.len() } else { self.prefill_chunk };
        let mut pos = 0usize;
        let mut last = 0;
        while pos < prompt.len() {
            // Full chunks first, then single tokens — every frame is s ∈ {chunk, 1}.
            let take = if prompt.len() - pos >= chunk { chunk } else { 1 };
            self.send_chunk(pos as u32, &prompt[pos..pos + take])?;
            last = *self.recv_result()?.last().expect("checked: one argmax per token");
            pos += take;
        }
        self.stats.prefill += t0.elapsed();
        Ok(last)
    }

    fn submit(&mut self, start_pos: u32, cur: TokenId, drafts: &[TokenId]) -> Result<()> {
        let t0 = Instant::now();
        let mut chunk = Vec::with_capacity(drafts.len() + 1);
        chunk.push(cur);
        chunk.extend_from_slice(drafts);
        self.send_chunk(start_pos, &chunk)?;
        self.stats.head_compute += t0.elapsed();
        Ok(())
    }

    fn poll(&mut self) -> Result<Vec<TokenId>> {
        let t0 = Instant::now();
        let result = self.recv_result();
        self.stats.waiting += t0.elapsed();
        result
    }

    fn is_eos(&self, token: TokenId) -> bool {
        self.eos.contains(&token)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc};

    use super::*;
    use crate::specdecode::SpecConfig;
    use crate::specpipe::run_spec_decode_pipelined;
    use crate::specpipe::tests::{Oracle, OracleDraft, Quality, QUALITIES};
    use crate::transport::listen_endpoint;

    const MODEL: &str = "fake/model";
    const TOTAL_LAYERS: usize = 8;
    const HIDDEN: usize = 5; // 4 bytes of the "next token" + a per-layer checksum
    const VOCAB: usize = 64;
    const EOS: TokenId = 60;
    const GAMMA: usize = 3;
    const CHUNK: usize = GAMMA + 1;
    const MAX_SEQ: usize = 128;

    /// A synthetic shard. The first shard is the whole "model" (an [`Oracle`] over the prefix
    /// held in its position-indexed KV) and encodes the next token into the residual stream;
    /// every shard adds a checksum term per layer it owns; the last shard only decodes the
    /// token if the checksum shows each layer ran exactly once — so a chain that skips,
    /// repeats or mis-wires a block produces token 0, not the right answer by accident.
    struct FakeShard {
        layers: (usize, usize),
        oracle: Oracle,
        kv: Vec<TokenId>,
        chunk: usize,
        max_seq: usize,
        /// Frames this shard has processed (shared with the test).
        seen: Arc<AtomicUsize>,
    }

    impl FakeShard {
        fn new(layers: (usize, usize), oracle: Oracle) -> Self {
            Self { layers, oracle, kv: Vec::new(), chunk: CHUNK, max_seq: MAX_SEQ, seen: Arc::default() }
        }
        fn checksum(&self) -> f32 {
            (self.layers.0..self.layers.1).map(|l| (l + 1) as f32).sum()
        }
    }

    impl ShardEngine for FakeShard {
        fn hidden_dim(&self) -> usize {
            HIDDEN
        }
        fn vocab_size(&self) -> usize {
            VOCAB
        }
        fn check_chunk(&self, start_pos: u32, s: usize) -> Result<(), String> {
            if s == 0 {
                return Err("empty chunk (0 tokens)".into());
            }
            if self.chunk > 0 && s != 1 && s != self.chunk {
                return Err(format!("no kernel for a {s}-token chunk"));
            }
            if start_pos as usize + s > self.max_seq {
                return Err(format!("chunk exceeds max_seq {}", self.max_seq));
            }
            Ok(())
        }
        fn forward_tokens(&mut self, start_pos: u32, tokens: &[TokenId]) -> Vec<f32> {
            assert_eq!(self.layers.0, 0, "only the first shard embeds tokens");
            assert!(start_pos as usize <= self.kv.len(), "chunk leaves a KV gap");
            self.kv.truncate(start_pos as usize);
            self.seen.fetch_add(1, Ordering::SeqCst);
            let mut out = Vec::with_capacity(tokens.len() * HIDDEN);
            for &t in tokens {
                self.kv.push(t);
                let next = self.oracle.next(&self.kv) as u32;
                out.extend(next.to_le_bytes().map(f32::from));
                out.push(self.checksum());
            }
            out
        }
        fn forward_hidden(&mut self, _start_pos: u32, s: usize, mut hidden: Vec<f32>) -> Vec<f32> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            for row in hidden.chunks_exact_mut(HIDDEN) {
                row[HIDDEN - 1] += self.checksum();
            }
            if self.layers.1 != TOTAL_LAYERS {
                return hidden;
            }
            let want: f32 = (1..=TOTAL_LAYERS).map(|l| l as f32).sum();
            let mut logits = vec![0.0f32; s * VOCAB];
            for (j, row) in hidden.chunks_exact(HIDDEN).enumerate() {
                let bytes = [row[0] as u8, row[1] as u8, row[2] as u8, row[3] as u8];
                let token = if row[HIDDEN - 1] == want { u32::from_le_bytes(bytes) as usize } else { 0 };
                logits[j * VOCAB + token.min(VOCAB - 1)] = 1.0;
            }
            logits
        }
        fn reset(&mut self) {
            self.kv.clear();
        }
    }

    fn oracle(prompt_len: usize, eos_at: Option<usize>) -> Oracle {
        Oracle { prompt_len, eos_at, eos: EOS }
    }

    fn spec(layers: (usize, usize)) -> NodeSpec {
        NodeSpec {
            model: MODEL.into(),
            layers,
            is_tail: layers.1 == TOTAL_LAYERS,
            chunk: CHUNK,
            max_seq: MAX_SEQ,
            token: Some("s3cret".into()),
            backend: "fake".into(),
        }
    }

    fn hello(plan: &[(usize, usize)]) -> Hello {
        Hello {
            proto: PROTO_VERSION,
            token: Some("s3cret".into()),
            model: MODEL.into(),
            chunk: CHUNK as u32,
            max_seq: MAX_SEQ as u32,
            hidden_dim: HIDDEN as u32,
            plan: plan.iter().map(|&(a, b)| [a as u32, b as u32]).collect(),
            index: 1,
        }
    }

    fn server_rt() -> Runtime {
        tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap()
    }

    struct Remote {
        addr: SocketAddr,
        seen: Arc<AtomicUsize>,
        /// One entry per finished session: `Ok` or the session's error text.
        sessions: mpsc::Receiver<Result<(), String>>,
    }

    /// Run a remote node on its own thread, serving `sessions` sessions back to back (the same
    /// loop `potpori serve` runs).
    fn spawn_node(spec: NodeSpec, downstream: Option<SocketAddr>, sessions: usize) -> Remote {
        let (addr_tx, addr_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let mut shard = FakeShard::new(spec.layers, oracle(0, None));
        let seen = shard.seen.clone();
        std::thread::spawn(move || {
            let rt = server_rt();
            let ep = {
                let _g = rt.enter();
                listen_endpoint("127.0.0.1:0".parse().unwrap()).unwrap()
            };
            addr_tx.send(ep.local_addr().unwrap()).unwrap();
            for _ in 0..sessions {
                shard.reset();
                let res = match downstream {
                    None => serve_tail(&mut shard, &spec, &ep, &rt),
                    Some(next) => serve_middle(&mut shard, &spec, &ep, next, &rt),
                };
                let _ = done_tx.send(res.map_err(|e| e.to_string()));
            }
        });
        Remote { addr: addr_rx.recv().unwrap(), seen, sessions: done_rx }
    }

    /// Spawn the remote nodes of `plan` (tail first, so each middle knows its downstream) and
    /// return them in pipeline order.
    fn spawn_chain(plan: &[(usize, usize)], sessions: usize) -> Vec<Remote> {
        let mut remotes: Vec<Remote> = Vec::new();
        for &layers in plan[1..].iter().rev() {
            let downstream = remotes.last().map(|r| r.addr);
            remotes.push(spawn_node(spec(layers), downstream, sessions));
        }
        remotes.reverse();
        remotes
    }

    fn connect(plan: &[(usize, usize)], to: SocketAddr, oracle: Oracle) -> Result<ChainLink<FakeShard>> {
        let head = FakeShard::new(plan[0], oracle);
        ChainLink::connect(head, to, hello(plan), &[EOS], Duration::from_secs(5))
    }

    const WAIT: Duration = Duration::from_secs(10);

    // The real protocol over real QUIC, 2 and 3 nodes: handshake, chunked prefill, depth-N
    // pipelining, divergence/overwrite, EOS — every draft quality must reproduce plain greedy.
    #[test]
    fn chain_over_quic_equals_greedy() {
        let prompt: Vec<TokenId> = vec![11, 12, 13, 14, 15, 16]; // prefill = one chunk + 2 singles
        for plan in [vec![(0, 4), (4, 8)], vec![(0, 3), (3, 5), (5, 8)]] {
            let remotes = spawn_chain(&plan, 1);
            let cfg = SpecConfig { gamma: GAMMA, max_new: 21 };
            // One session, many generations: each prefill at position 0 overwrites the KV.
            let mut chain = connect(&plan, remotes[0].addr, oracle(prompt.len(), None)).unwrap();
            assert_eq!(chain.nodes().len(), plan.len() - 1);
            assert_eq!(chain.nodes().last().unwrap().layers, [plan.last().unwrap().0 as u32, 8]);
            for eos_at in [None, Some(0), Some(9), Some(20)] {
                let o = oracle(prompt.len(), eos_at);
                chain.head.oracle = o;
                let expected = o.greedy(&prompt, cfg.max_new);
                for quality in QUALITIES {
                    for depth in [1usize, 2, 4] {
                        let mut draft = OracleDraft { oracle: o, quality, calls: 0 };
                        let (out, stats) =
                            run_spec_decode_pipelined(&mut draft, &mut chain, &prompt, &cfg, depth).unwrap();
                        assert_eq!(out, expected, "{plan:?} eos_at={eos_at:?} {quality:?} depth={depth}");
                        assert_eq!(stats.committed, out.len());
                        assert!(chain.awaiting.is_empty());
                    }
                }
            }
            let stats = chain.stats();
            assert!(stats.frames > 0 && stats.payload_bytes % (HIDDEN as u64 * 4) == 0);
            assert!(stats.payload_bytes >= stats.frames * (HIDDEN as u64 * 4));
            chain.shutdown();
            for r in &remotes {
                assert_eq!(r.sessions.recv_timeout(WAIT).unwrap(), Ok(()), "clean session end");
            }
        }
    }

    // A misconfigured chain must fail at connect with the reason, and the node must stay up
    // for the next (correct) head.
    #[test]
    fn handshake_rejects_mismatches_and_node_survives() {
        let plan = [(0, 4), (4, 8)];
        let bad: Vec<(&str, Box<dyn Fn(&mut Hello)>)> = vec![
            ("bad or missing token", Box::new(|h| h.token = Some("nope".into()))),
            ("bad or missing token", Box::new(|h| h.token = None)),
            ("serving model", Box::new(|h| h.model = "other/model".into())),
            ("serving layers 4:8", Box::new(|h| h.plan = vec![[0, 5], [5, 8]])),
            ("--role tail", Box::new(|h| h.plan = vec![[0, 4], [4, 8], [8, 9]])),
            ("--chunk 4", Box::new(|h| h.chunk = 8)),
            ("--max-seq 128", Box::new(|h| h.max_seq = 4096)),
            ("protocol version", Box::new(|h| h.proto = PROTO_VERSION + 1)),
            ("hidden size", Box::new(|h| h.hidden_dim = 7)),
            ("not a remote seat", Box::new(|h| h.index = 0)),
        ];
        let tail = spawn_node(spec((4, 8)), None, bad.len() + 1);
        for (want, mutate) in &bad {
            let mut h = hello(&plan);
            mutate(&mut h);
            let head = FakeShard::new((0, 4), oracle(1, None));
            let err = ChainLink::connect(head, tail.addr, h, &[EOS], Duration::from_secs(5))
                .err()
                .unwrap_or_else(|| panic!("{want}: should have been refused"))
                .to_string();
            assert!(err.contains(want) && err.contains("tail [4, 8)"), "{want}: got {err:?}");
            assert!(!err.contains("s3cret"), "the token must never be echoed: {err:?}");
            assert!(tail.sessions.recv_timeout(WAIT).unwrap().is_err());
        }
        // The node is still serving: a correct head gets a working session.
        let prompt = vec![5];
        let o = oracle(prompt.len(), Some(6));
        let mut chain = connect(&plan, tail.addr, o).unwrap();
        let mut draft = OracleDraft { oracle: o, quality: Quality::Perfect, calls: 0 };
        let cfg = SpecConfig { gamma: GAMMA, max_new: 50 };
        let (out, _) = run_spec_decode_pipelined(&mut draft, &mut chain, &prompt, &cfg, 2).unwrap();
        assert_eq!(out, o.greedy(&prompt, 50));
        chain.shutdown();
        assert_eq!(tail.sessions.recv_timeout(WAIT).unwrap(), Ok(()));
    }

    // A problem two hops away must reach the head at connect, naming the node at fault.
    #[test]
    fn rejection_below_a_middle_reaches_the_head() {
        // The tail really serves 6:8, but the plan (and the middle) say the tail is 5:8.
        let tail = spawn_node(spec((6, 8)), None, 1);
        let middle = spawn_node(spec((3, 5)), Some(tail.addr), 2);
        let plan = [(0, 3), (3, 5), (5, 8)];
        let err = connect(&plan, middle.addr, oracle(1, None)).err().unwrap().to_string();
        assert!(err.contains("tail [6, 8)") && err.contains("serving layers 6:8"), "{err:?}");

        // And a middle whose downstream is not there at all says so.
        let dead: SocketAddr = "127.0.0.1:9".parse().unwrap();
        let lonely = spawn_node(spec((3, 5)), Some(dead), 1);
        let head = FakeShard::new((0, 3), oracle(1, None));
        let err = ChainLink::connect(head, lonely.addr, hello(&plan), &[EOS], Duration::from_secs(30))
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("middle [3, 5)") && err.contains("cannot reach its downstream"), "{err:?}");
    }

    #[test]
    fn unreachable_node_times_out_with_a_clear_error() {
        let dead: SocketAddr = "127.0.0.1:9".parse().unwrap();
        let head = FakeShard::new((0, 4), oracle(1, None));
        let t0 = Instant::now();
        let err = ChainLink::connect(head, dead, hello(&[(0, 4), (4, 8)]), &[EOS], Duration::from_secs(1))
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("could not reach 127.0.0.1:9"), "{err:?}");
        assert!(t0.elapsed() < Duration::from_secs(25), "took {:?}", t0.elapsed());
    }

    // Raw-wire abuse: a peer that passes the handshake and then sends a malformed frame gets an
    // Error back; the node ends that session and serves the next one.
    #[test]
    fn malformed_frames_end_the_session_not_the_node() {
        let plan = [(0, 4), (4, 8)];
        let frames: Vec<(&str, HiddenStateFrame)> = vec![
            ("payload is 4 bytes", HiddenStateFrame { step: 0, start_pos: 0, n_tokens: 1, hidden_dim: 5, dtype: "f32".into(), data: vec![0; 4] }),
            ("unsupported codec", HiddenStateFrame { step: 0, start_pos: 0, n_tokens: 1, hidden_dim: 5, dtype: "fp8".into(), data: vec![0; 20] }),
            ("hidden_dim 9", HiddenStateFrame { step: 0, start_pos: 0, n_tokens: 1, hidden_dim: 9, dtype: "f32".into(), data: vec![0; 36] }),
            ("no kernel for a 3-token chunk", HiddenStateFrame::from_f32(0, 0, 3, 5, &[0.0; 15])),
            ("exceeds max_seq", HiddenStateFrame::from_f32(0, MAX_SEQ as u32, 1, 5, &[0.0; 5])),
            ("empty chunk", HiddenStateFrame::from_f32(0, 0, 0, 5, &[])),
        ];
        let tail = spawn_node(spec((4, 8)), None, frames.len() + 1);
        let rt = server_rt();
        for (want, frame) in frames {
            let got = rt.block_on(async {
                let ep = dial_endpoint().unwrap();
                let (mut peer, _) = handshake(&ep, tail.addr, hello(&plan)).await.ok().expect("handshake");
                peer.send(&PeerMessage::Hidden(frame)).await.unwrap();
                match peer.recv().await.unwrap() {
                    Some(PeerMessage::Error(why)) => why,
                    other => panic!("{want}: expected Error, got {other:?}"),
                }
            });
            assert!(got.contains(want), "{want}: got {got:?}");
            assert!(tail.sessions.recv_timeout(WAIT).unwrap().is_err());
            assert_eq!(tail.seen.load(Ordering::SeqCst), 0, "a bad frame must never reach the shard");
        }
        let mut chain = connect(&plan, tail.addr, oracle(1, None)).unwrap();
        assert_eq!(chain.prefill(&[7]).unwrap(), oracle(1, None).next(&[7]));
        chain.shutdown();
    }

    // A remote that lies (short answer) or vanishes mid-generation must surface as an Err on
    // the head — never a panic, never a hang.
    #[test]
    fn misbehaving_tail_is_an_error_on_the_head() {
        let plan = [(0, 4), (4, 8)];
        for vanish in [false, true] {
            let (addr_tx, addr_rx) = mpsc::channel();
            std::thread::spawn(move || {
                let rt = server_rt();
                let ep = {
                    let _g = rt.enter();
                    listen_endpoint("127.0.0.1:0".parse().unwrap()).unwrap()
                };
                addr_tx.send(ep.local_addr().unwrap()).unwrap();
                rt.block_on(async {
                    let mut up = accept(&ep).await.unwrap();
                    let Some(PeerMessage::Hello(_)) = up.recv().await.unwrap() else { panic!("no hello") };
                    up.send(&PeerMessage::HelloAck(vec![spec((4, 8)).info()])).await.unwrap();
                    let Some(PeerMessage::Hidden(frame)) = up.recv().await.unwrap() else { panic!("no frame") };
                    if vanish {
                        up.close();
                        ep.wait_idle().await; // let the close actually leave before the runtime dies
                    } else {
                        // One argmax short.
                        let argmax = vec![1; frame.n_tokens as usize - 1];
                        up.send(&PeerMessage::Result(VerifyResult { step: frame.step, argmax })).await.unwrap();
                        up.wait_closed().await;
                    }
                });
            });
            let mut chain = connect(&plan, addr_rx.recv().unwrap(), oracle(4, None)).unwrap();
            let err = chain.prefill(&[1, 2, 3, 4]).unwrap_err().to_string();
            let want = if vanish { "lost the chain" } else { "out of order or short" };
            assert!(err.contains(want), "vanish={vanish}: {err:?}");
        }
    }

    // Pipelining depends on a submitted frame leaving NOW, not at the head's next blocking
    // call: the tail must see it while the head thread is busy elsewhere.
    #[test]
    fn submitted_frames_flush_while_the_head_is_busy() {
        let plan = [(0, 4), (4, 8)];
        let tail = spawn_node(spec((4, 8)), None, 1);
        let mut chain = connect(&plan, tail.addr, oracle(1, None)).unwrap();
        chain.prefill(&[9]).unwrap();
        let before = tail.seen.load(Ordering::SeqCst);
        chain.submit(1, 3, &[4, 5, 6]).unwrap();
        // The head "computes" (no runtime calls); the frame must reach the tail regardless.
        let t0 = Instant::now();
        while tail.seen.load(Ordering::SeqCst) == before && t0.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(tail.seen.load(Ordering::SeqCst), before + 1, "frame was not flushed");
        assert_eq!(chain.poll().unwrap().len(), 4);
        chain.shutdown();
    }

    #[test]
    fn token_comparison() {
        assert!(token_matches("abc", Some("abc")));
        for wrong in [None, Some(""), Some("ab"), Some("abd"), Some("abcd")] {
            assert!(!token_matches("abc", wrong), "{wrong:?}");
        }
    }
}

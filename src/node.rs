//! The three roles in a chain, on top of the wire in [`crate::transport`].
//!
//!   - **Head** ([`ChainLink`]): runs the first shard locally, ships the residual stream to
//!     the next node, and reads back the tail's greedy tokens. It implements
//!     [`PipelinedChain`], so the speculative-decode driver drives the whole chain through it.
//!   - **Middle** ([`serve_middle`]): runs its shard and forwards downstream; results and
//!     errors are relayed back up independently of the work flowing down.
//!   - **Tail** ([`serve_tail`]): runs the last shard and answers each frame with the argmax
//!     of its logits.
//!
//! A session starts with a hop-by-hop handshake: each node tells the next which model, layer
//! boundary, cache size and chunk size it expects, and only reports ready once every node
//! below it has. A misconfigured chain fails at connect, and the head is told which node
//! refused and which setting differs. After that, any failure travels upstream as
//! [`PeerMessage::Error`] and surfaces on the head as an `Err`.
//!
//! A session may *tap* layers: every node then reports the outputs of the tapped layers it
//! owns with each result, and the head assembles them into per-position features for a draft
//! model that reads the target's hidden states ([`crate::specfeat`]). A node exposes a tapped
//! layer by running its range as several shards that end there ([`TappedShards`]).
//!
//! Everything here is generic over [`ShardEngine`], so the protocol is tested over real QUIC
//! with a synthetic shard — no GPU and no model (see the tests below).
//!
//! Threading: a GPU runtime stays on one thread and the driver is synchronous, while QUIC is
//! async. Each node therefore owns its shard on one OS thread and `block_on`s network calls
//! there; QUIC I/O runs on the tokio runtime's worker threads, so frames leave and keepalives
//! fire while the shard computes.

use std::collections::VecDeque;
use std::io;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, ensure, Result};
use quinn::Endpoint;
use tokio::runtime::Runtime;

use crate::engine::{argmax, TokenId};
use crate::specfeat::FeatureChain;
use crate::specpipe::PipelinedChain;
use crate::transport::{
    accept, dial, dial_endpoint, negotiate, receive_hello, relay_middle, HiddenStateFrame,
    LayerTap, PeerConnection, PeerMessage, SessionHello, TlsIdentity, VerifyResult, CLOSE_TIMEOUT,
};

/// How long one dial attempt may take before it counts as "not reachable yet".
const DIAL_TIMEOUT: Duration = Duration::from_secs(10);
/// Pause between dial attempts while a node is still loading.
const DIAL_RETRY: Duration = Duration::from_secs(5);

/// The compute a node contributes: a contiguous block of layers with its own KV cache.
/// Implemented by the Luminal shard ([`crate::qwen::shard::Shard`]) and by the synthetic shard
/// in this module's tests.
pub trait ShardEngine {
    /// What an adjacent node must agree on, for the boundary at `next_layer`.
    fn hello(&self, next_layer: usize) -> SessionHello;
    /// Global layer range `[start, end)` this shard runs.
    fn range(&self) -> (usize, usize);
    /// Number of layers in the whole model.
    fn model_layers(&self) -> usize;
    fn hidden_dim(&self) -> usize;
    fn vocab_size(&self) -> usize;
    /// KV positions available.
    fn capacity(&self) -> usize;
    /// Kernel chunk size: 0 = any chunk size, else chunks carry 1 or this many tokens.
    fn chunk_size(&self) -> usize;
    /// First shard: embed `tokens` at absolute positions from `pos`, run the block, return the
    /// residual stream `[s, hidden]` flat.
    fn forward_tokens(&mut self, pos: u32, tokens: &[TokenId]) -> Vec<f32>;
    /// Non-first shard: consume `[s, hidden]`, run the block, return `[s, hidden]` — or
    /// `[s, vocab]` logits if this is the last shard.
    fn forward_hidden(&mut self, pos: u32, s: usize, hidden: Vec<f32>) -> Vec<f32>;
    /// Clear the KV cache (between sessions).
    fn reset(&mut self);
    /// Layers of this shard whose outputs it reports with every result, ascending. Their
    /// union over the chain is the session's `taps` in [`SessionHello`].
    fn tap_layers(&self) -> Vec<usize> {
        Vec::new()
    }
    /// The tapped outputs of the most recent forward: one `[s, hidden]` block per entry of
    /// [`Self::tap_layers`], in the same order.
    fn take_taps(&mut self) -> Vec<Vec<f32>> {
        Vec::new()
    }
}

/// The ranges a node covering `range` must run as separate shards so that every tapped layer
/// it owns is the last layer of one of them.
pub fn tap_ranges(range: (usize, usize), taps: &[usize]) -> Vec<(usize, usize)> {
    let mut cuts: Vec<usize> = taps
        .iter()
        .map(|&layer| layer + 1)
        .filter(|&cut| cut > range.0 && cut < range.1)
        .collect();
    cuts.sort_unstable();
    cuts.dedup();
    cuts.push(range.1);
    let mut start = range.0;
    cuts.into_iter()
        .map(|end| (std::mem::replace(&mut start, end), end))
        .collect()
}

/// One node's layers as consecutive shards, cut after each tapped layer, behaving as a single
/// [`ShardEngine`] that also reports the tapped outputs. With no taps it is one shard and
/// adds nothing.
pub struct TappedShards<S: ShardEngine> {
    parts: Vec<S>,
    /// Tapped layers of the whole model, ascending.
    taps: Vec<usize>,
    last: Vec<Vec<f32>>,
}

impl<S: ShardEngine> TappedShards<S> {
    /// `parts` must be the shards for [`tap_ranges`] of the node's range, in order. The
    /// model's last layer cannot be tapped: the shard that owns it returns logits.
    pub fn new(parts: Vec<S>, mut taps: Vec<usize>) -> Result<Self> {
        taps.sort_unstable();
        taps.dedup();
        ensure!(!parts.is_empty(), "a node needs at least one shard");
        let layers = parts[0].model_layers();
        ensure!(
            taps.iter().all(|&layer| layer + 1 < layers),
            "tapped layers {taps:?} must be below the model's last layer ({})",
            layers - 1
        );
        let range = (parts[0].range().0, parts[parts.len() - 1].range().1);
        let got: Vec<_> = parts.iter().map(|p| p.range()).collect();
        let want = tap_ranges(range, &taps);
        ensure!(
            got == want,
            "shards {got:?} do not expose the tapped layers {taps:?} (expected {want:?})"
        );
        Ok(Self {
            parts,
            taps,
            last: Vec::new(),
        })
    }

    fn run(&mut self, pos: u32, s: usize, mut x: Vec<f32>, first: usize) -> Vec<f32> {
        for i in first..self.parts.len() {
            x = self.parts[i].forward_hidden(pos, s, x);
            self.keep(i, &x);
        }
        x
    }

    fn keep(&mut self, part: usize, output: &[f32]) {
        let layer = self.parts[part].range().1 - 1;
        if self.taps.contains(&layer) {
            self.last.push(output.to_vec());
        }
    }
}

impl<S: ShardEngine> ShardEngine for TappedShards<S> {
    fn hello(&self, next_layer: usize) -> SessionHello {
        SessionHello {
            taps: self.taps.clone(),
            ..self.parts[0].hello(next_layer)
        }
    }
    fn range(&self) -> (usize, usize) {
        (
            self.parts[0].range().0,
            self.parts[self.parts.len() - 1].range().1,
        )
    }
    fn model_layers(&self) -> usize {
        self.parts[0].model_layers()
    }
    fn hidden_dim(&self) -> usize {
        self.parts[0].hidden_dim()
    }
    fn vocab_size(&self) -> usize {
        self.parts[0].vocab_size()
    }
    fn capacity(&self) -> usize {
        self.parts[0].capacity()
    }
    fn chunk_size(&self) -> usize {
        self.parts[0].chunk_size()
    }
    fn forward_tokens(&mut self, pos: u32, tokens: &[TokenId]) -> Vec<f32> {
        self.last.clear();
        let x = self.parts[0].forward_tokens(pos, tokens);
        self.keep(0, &x);
        self.run(pos, tokens.len(), x, 1)
    }
    fn forward_hidden(&mut self, pos: u32, s: usize, hidden: Vec<f32>) -> Vec<f32> {
        self.last.clear();
        self.run(pos, s, hidden, 0)
    }
    fn reset(&mut self) {
        self.parts.iter_mut().for_each(S::reset);
        self.last.clear();
    }
    fn tap_layers(&self) -> Vec<usize> {
        let (start, end) = self.range();
        self.taps
            .iter()
            .copied()
            .filter(|&layer| layer >= start && layer < end)
            .collect()
    }
    fn take_taps(&mut self) -> Vec<Vec<f32>> {
        std::mem::take(&mut self.last)
    }
}

/// This node's tapped outputs for the frame it just ran, ready for the wire.
fn wire_taps<S: ShardEngine>(shard: &mut S) -> Vec<LayerTap> {
    let layers = shard.tap_layers();
    let outputs = shard.take_taps();
    assert_eq!(layers.len(), outputs.len(), "one output per tapped layer");
    layers
        .into_iter()
        .zip(&outputs)
        .map(|(layer, values)| LayerTap::from_f32(layer, values))
        .collect()
}

/// How a node names itself in errors that travel to the head.
fn label<S: ShardEngine>(shard: &S) -> String {
    let (start, end) = shard.range();
    format!("node [{start}, {end})")
}

fn validate_frame<S: ShardEngine>(shard: &S, frame: &HiddenStateFrame) -> io::Result<()> {
    frame.validate(
        shard.hidden_dim(),
        shard.range().0,
        shard.capacity(),
        shard.chunk_size(),
    )
}

/// Serve one session as the tail: accept a peer, handshake, then answer each hidden-state
/// frame with the argmax of this shard's logits until the head closes. `Ok(())` = clean end.
pub fn serve_tail<S: ShardEngine>(shard: &mut S, endpoint: &Endpoint, rt: &Runtime) -> Result<()> {
    let (start, end) = shard.range();
    assert_eq!(end, shard.model_layers(), "the tail owns the last layer");
    let vocab = shard.vocab_size();
    let who = label(shard);
    rt.block_on(async {
        let mut up = accept(endpoint).await?;
        receive_hello(&mut up, &shard.hello(start), &who).await?;
        up.send(&PeerMessage::Ready { final_layer: end }).await?;
        tracing::info!("session started");
        loop {
            let frame = match up.recv().await? {
                None => return Ok(()), // clean EOF from the head; dropping `up` closes back
                Some(PeerMessage::Hidden(frame)) => frame,
                Some(_) => {
                    let why = format!("{who}: unexpected message during the session");
                    return Err(up.fail(why).await.into());
                }
            };
            if let Err(e) = validate_frame(shard, &frame) {
                return Err(up.fail(format!("{who}: {e}")).await.into());
            }
            let s = frame.n_tokens as usize;
            let logits = shard.forward_hidden(frame.start_pos, s, frame.to_f32()); // sync GPU
            let tokens: Option<Vec<TokenId>> = (logits.len() == s * vocab)
                .then(|| logits.chunks_exact(vocab).map(argmax).collect())
                .flatten();
            let Some(argmax) = tokens else {
                let why = format!("{who}: the shard produced missing or non-finite logits");
                return Err(up.fail(why).await.into());
            };
            let result = VerifyResult {
                step: frame.step,
                argmax,
                taps: wire_taps(shard),
            };
            up.send(&PeerMessage::Result(result)).await?;
        }
    })
}

/// Serve one session as a middle: accept the upstream peer, handshake, bring up the chain
/// below, then run [`relay_middle`] with this shard. A chain of any length is
/// head → middle → … → middle → tail.
pub fn serve_middle<S: ShardEngine>(
    shard: &mut S,
    upstream: &Endpoint,
    downstream_addr: SocketAddr,
    rt: &Runtime,
    tls: &TlsIdentity,
) -> Result<()> {
    let (start, end) = shard.range();
    let who = label(shard);
    rt.block_on(async {
        let mut up = accept(upstream).await?;
        receive_hello(&mut up, &shard.hello(start), &who).await?;

        // Bring up the rest of the chain before reporting ready, so the head hears about a
        // problem anywhere below this node at connect time.
        let down_ep = dial_endpoint(tls)?;
        let mut down = match tokio::time::timeout(DIAL_TIMEOUT, dial(&down_ep, downstream_addr))
            .await
            .unwrap_or_else(|_| Err(io::Error::other("no answer")))
        {
            Ok(down) => down,
            Err(e) => {
                let why = format!("{who}: cannot reach its downstream {downstream_addr}: {e}");
                return Err(up.fail(why).await.into());
            }
        };
        if let Err(e) = negotiate(&mut down, shard.hello(end), shard.model_layers()).await {
            // A refusal is already prefixed by the node that refused.
            return Err(up.fail(e.to_string()).await.into());
        }
        up.send(&PeerMessage::Ready {
            final_layer: shard.model_layers(),
        })
        .await?;
        tracing::info!("session started");

        let outcome = relay_middle(up, down, |frame| {
            validate_frame(shard, &frame).map_err(|e| io::Error::other(format!("{who}: {e}")))?;
            let s = frame.n_tokens as usize;
            let hidden = shard.forward_hidden(frame.start_pos, s, frame.to_f32()); // sync GPU
            let out = HiddenStateFrame::from_f32(
                frame.step,
                frame.start_pos,
                end as u32,
                s as u32,
                shard.hidden_dim() as u32,
                &hidden,
            );
            Ok((out, wire_taps(shard)))
        })
        .await;
        // Let the close reach the node below before this endpoint is dropped; otherwise it
        // waits out the idle timeout and cannot serve the next head.
        let _ = tokio::time::timeout(CLOSE_TIMEOUT, down_ep.wait_idle()).await;
        Ok(outcome?)
    })
}

/// Where a generation's wall-clock went, as seen from the head.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinkStats {
    /// Prompt prefill: head compute plus every prefill round trip.
    pub prefill: Duration,
    /// Head shard compute (and send) for verification chunks.
    pub head_compute: Duration,
    /// Time blocked waiting for verification results.
    pub waiting: Duration,
    /// Hidden-state frames sent, and their payload bytes.
    pub frames: u64,
    pub payload_bytes: u64,
    /// Tapped-layer bytes received with results.
    pub tap_bytes: u64,
}

/// A frame sent and not yet answered.
struct Pending {
    step: u64,
    n_tokens: usize,
    /// The head's own tapped outputs for the frame, one block per layer of `head_taps`.
    taps: Vec<Vec<f32>>,
}

/// The head's link to the chain: the local first shard plus one persistent QUIC bi-stream to
/// the next node. `submit` sends a hidden-state frame without waiting; `poll` reads results
/// in order. The head's compute for chunk k+1 therefore overlaps the rest of the chain's work
/// on chunk k.
pub struct ChainLink<S: ShardEngine> {
    head: S,
    rt: Runtime,
    // Held to keep the client side driving I/O; dropping it would stall the connection.
    _endpoint: Endpoint,
    peer: PeerConnection,
    next_step: u64,
    /// Every frame sent and not yet answered, oldest first.
    awaiting: VecDeque<Pending>,
    /// Tapped layers the head reports itself, and those it expects back from the chain.
    head_taps: Vec<usize>,
    remote_taps: Vec<usize>,
    eos: Vec<TokenId>,
    stats: LinkStats,
}

impl<S: ShardEngine> ChainLink<S> {
    /// Dial the next node and run the handshake. Remote nodes come up at different times
    /// (their kernel search takes minutes), so a node that does not answer is retried for up
    /// to `wait`. A node that answers and refuses — an untrusted certificate, a different
    /// build, or settings that do not match — fails immediately.
    pub fn connect(
        head: S,
        addr: SocketAddr,
        tls: &TlsIdentity,
        eos: &[TokenId],
        wait: Duration,
    ) -> Result<Self> {
        assert_eq!(head.range().0, 0, "the head owns layer 0");
        let hello = head.hello(head.range().1);
        let head_taps = head.tap_layers();
        let remote_taps = hello
            .taps
            .iter()
            .copied()
            .filter(|layer| !head_taps.contains(layer))
            .collect();
        let final_layer = head.model_layers();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        let (endpoint, peer) = rt.block_on(async {
            let ep = dial_endpoint(tls)?;
            let deadline = Instant::now() + wait;
            let mut attempt = 0u32;
            let mut peer = loop {
                let why = match tokio::time::timeout(DIAL_TIMEOUT, dial(&ep, addr)).await {
                    Ok(Ok(peer)) => break peer,
                    Ok(Err(e)) if e.to_string().contains("cryptographic handshake") => bail!(
                        "{addr} is up but the TLS handshake failed ({e}) — each side needs the \
                         other's cert.pem in its peers.pem, and both must run the same build"
                    ),
                    Ok(Err(e)) => e.to_string(),
                    Err(_) => "no answer".to_string(),
                };
                if Instant::now() >= deadline {
                    bail!("could not reach {addr} within {}s: {why}", wait.as_secs());
                }
                if attempt.is_multiple_of(12) {
                    tracing::info!("waiting for {addr} (still loading?): {why}");
                }
                attempt += 1;
                tokio::time::sleep(DIAL_RETRY.min(wait)).await;
            };
            negotiate(&mut peer, hello, final_layer)
                .await
                .map_err(|e| anyhow!("the chain refused the session — {e}"))?;
            Ok((ep, peer))
        })?;
        Ok(Self {
            head,
            rt,
            _endpoint: endpoint,
            peer,
            next_step: 0,
            awaiting: VecDeque::new(),
            head_taps,
            remote_taps,
            eos: eos.to_vec(),
            stats: LinkStats::default(),
        })
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
        let (chunk, capacity) = (self.head.chunk_size(), self.head.capacity());
        if s == 0 || (chunk != 0 && s != 1 && s != chunk) || start_pos as usize + s > capacity {
            bail!("no kernel or cache room for {s} tokens at position {start_pos}");
        }
        let hidden = self.head.forward_tokens(start_pos, tokens); // sync GPU
        let taps = self.head.take_taps();
        let step = self.next_step;
        self.next_step += 1;
        let frame = HiddenStateFrame::from_f32(
            step,
            start_pos,
            self.head.range().1 as u32,
            s as u32,
            self.head.hidden_dim() as u32,
            &hidden,
        );
        self.stats.frames += 1;
        self.stats.payload_bytes += frame.data.len() as u64;
        let peer = &mut self.peer;
        self.rt
            .block_on(peer.send(&PeerMessage::Hidden(frame)))
            .map_err(|e| anyhow!("lost the chain while sending: {e}"))?;
        self.awaiting.push_back(Pending {
            step,
            n_tokens: s,
            taps,
        });
        Ok(())
    }

    /// Block for the oldest outstanding frame's result: the target's greedy token after each
    /// of its positions, and (in a tapped session) each position's features — the tapped
    /// layers' outputs in ascending layer order, `[n_tokens, taps * hidden]` flat.
    fn recv_result(&mut self) -> Result<(Vec<TokenId>, Vec<f32>)> {
        let Some(Pending {
            step,
            n_tokens: s,
            taps: own,
        }) = self.awaiting.pop_front()
        else {
            bail!("poll with no chunk in flight");
        };
        let peer = &mut self.peer;
        let result = match self.rt.block_on(peer.recv()) {
            Ok(Some(PeerMessage::Result(result))) => result,
            Ok(Some(PeerMessage::Error(why))) => bail!("the chain failed — {why}"),
            Ok(Some(_)) => bail!("the chain sent an unexpected message"),
            Ok(None) => bail!("the chain closed the session mid-generation"),
            Err(e) => bail!("lost the chain: {e}"),
        };
        let hidden = self.head.hidden_dim();
        result
            .validate(step, s, self.head.vocab_size(), &self.remote_taps, hidden)
            .map_err(|e| anyhow!("the chain answered wrongly: {e}"))?;
        self.stats.tap_bytes += result.taps.iter().map(|t| t.data.len() as u64).sum::<u64>();

        // Interleave the head's and the chain's layers into one row per position.
        let mut layers: Vec<(usize, Vec<f32>)> = self.head_taps.iter().copied().zip(own).collect();
        layers.extend(result.taps.iter().map(|t| (t.layer as usize, t.to_f32())));
        layers.sort_unstable_by_key(|(layer, _)| *layer);
        let mut features = Vec::with_capacity(s * layers.len() * hidden);
        for row in 0..s {
            for (_, values) in &layers {
                features.extend_from_slice(&values[row * hidden..(row + 1) * hidden]);
            }
        }
        Ok((result.argmax, features))
    }

    /// Feed the prompt in pieces the kernels accept; the token after it and its features.
    fn prefill_prompt(&mut self, prompt: &[TokenId]) -> Result<(TokenId, Vec<f32>)> {
        if prompt.is_empty() {
            bail!("empty prompt");
        }
        let t0 = Instant::now();
        // With chunk-specialized kernels only frames of 1 or `chunk` tokens can run: send
        // full chunks first, then single tokens. Generic kernels take the prompt whole.
        let chunk = match self.head.chunk_size() {
            0 => prompt.len(),
            chunk => chunk,
        };
        let mut pos = 0usize;
        let mut last = 0;
        let mut features = Vec::new();
        while pos < prompt.len() {
            let take = if prompt.len() - pos >= chunk {
                chunk
            } else {
                1
            };
            self.send_chunk(pos as u32, &prompt[pos..pos + take])?;
            let (tokens, rows) = self.recv_result()?;
            last = *tokens.last().expect("validated: one per token");
            features.extend(rows);
            pos += take;
        }
        self.stats.prefill += t0.elapsed();
        Ok((last, features))
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
    fn capacity(&self) -> usize {
        self.head.capacity()
    }

    fn prefill(&mut self, prompt: &[TokenId]) -> Result<TokenId> {
        Ok(self.prefill_prompt(prompt)?.0)
    }

    fn submit(&mut self, start_pos: u32, cur: TokenId, drafts: &[TokenId]) -> Result<()> {
        let t0 = Instant::now();
        let chunk: Vec<TokenId> = std::iter::once(cur).chain(drafts.iter().copied()).collect();
        self.send_chunk(start_pos, &chunk)?;
        self.stats.head_compute += t0.elapsed();
        Ok(())
    }

    fn poll(&mut self) -> Result<Vec<TokenId>> {
        let t0 = Instant::now();
        let result = self.recv_result();
        self.stats.waiting += t0.elapsed();
        Ok(result?.0)
    }

    fn is_eos(&self, token: TokenId) -> bool {
        self.eos.contains(&token)
    }
}

impl<S: ShardEngine> FeatureChain for ChainLink<S> {
    fn capacity(&self) -> usize {
        self.head.capacity()
    }

    fn prefill_features(&mut self, prompt: &[TokenId]) -> Result<(TokenId, Vec<f32>)> {
        self.prefill_prompt(prompt)
    }

    fn verify_features(
        &mut self,
        start_pos: u32,
        cur: TokenId,
        drafts: &[TokenId],
    ) -> Result<(Vec<TokenId>, Vec<f32>)> {
        PipelinedChain::submit(self, start_pos, cur, drafts)?;
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
pub(crate) mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc};

    use super::*;
    use crate::specdecode::SpecConfig;
    use crate::specpipe::run_spec_decode_pipelined;
    use crate::specpipe::tests::{Oracle, OracleDraft, Quality, QUALITIES};
    use crate::transport::listen_endpoint;

    const TOTAL_LAYERS: usize = 8;
    const HIDDEN: usize = 5; // 4 bytes of the "next token" + a per-layer checksum
    const VOCAB: usize = 64;
    const EOS: TokenId = 60;
    const GAMMA: usize = 3;
    const CHUNK: usize = GAMMA + 1;
    const MAX_SEQ: usize = 128;
    const WAIT: Duration = Duration::from_secs(10);

    /// A synthetic shard. The first shard is the whole "model" (an [`Oracle`] over the prefix
    /// held in its position-indexed KV) and encodes the next token into the residual stream.
    /// Every shard adds a checksum term per layer it owns, and the last shard only decodes
    /// the token if the checksum shows each layer ran exactly once — so a chain that skips,
    /// repeats or mis-wires a block yields token 0 rather than the right answer by accident.
    struct FakeShard {
        layers: (usize, usize),
        oracle: Oracle,
        kv: Vec<TokenId>,
        model: &'static str,
        chunk: usize,
        /// Advertised in the handshake.
        max_seq: usize,
        /// Actually enforced; differs from `max_seq` only to provoke a mid-session failure.
        capacity: usize,
        /// Emit NaN logits (a miscompiled kernel).
        nan: bool,
        /// Frames this shard has started on (shared with the test).
        seen: Arc<AtomicUsize>,
    }

    impl FakeShard {
        fn new(layers: (usize, usize), oracle: Oracle) -> Self {
            Self {
                layers,
                oracle,
                kv: Vec::new(),
                model: "fake",
                chunk: CHUNK,
                max_seq: MAX_SEQ,
                capacity: MAX_SEQ,
                nan: false,
                seen: Arc::default(),
            }
        }
        fn checksum(&self) -> f32 {
            (self.layers.0..self.layers.1).map(|l| (l + 1) as f32).sum()
        }
    }

    impl ShardEngine for FakeShard {
        fn hello(&self, next_layer: usize) -> SessionHello {
            SessionHello {
                model: self.model.to_string(),
                next_layer,
                max_seq: self.max_seq,
                chunk: self.chunk,
                taps: Vec::new(),
            }
        }
        fn range(&self) -> (usize, usize) {
            self.layers
        }
        fn model_layers(&self) -> usize {
            TOTAL_LAYERS
        }
        fn hidden_dim(&self) -> usize {
            HIDDEN
        }
        fn vocab_size(&self) -> usize {
            VOCAB
        }
        fn capacity(&self) -> usize {
            self.capacity
        }
        fn chunk_size(&self) -> usize {
            self.chunk
        }
        fn forward_tokens(&mut self, pos: u32, tokens: &[TokenId]) -> Vec<f32> {
            assert!(pos as usize <= self.kv.len(), "chunk leaves a KV gap");
            self.kv.truncate(pos as usize);
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
        fn forward_hidden(&mut self, _pos: u32, s: usize, mut hidden: Vec<f32>) -> Vec<f32> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            for row in hidden.as_chunks_mut::<HIDDEN>().0 {
                row[HIDDEN - 1] += self.checksum();
            }
            if self.layers.1 != TOTAL_LAYERS {
                return hidden;
            }
            let want: f32 = (1..=TOTAL_LAYERS).map(|l| l as f32).sum();
            let mut logits = vec![if self.nan { f32::NAN } else { 0.0 }; s * VOCAB];
            for (j, row) in hidden.as_chunks::<HIDDEN>().0.iter().enumerate() {
                let bytes = [row[0] as u8, row[1] as u8, row[2] as u8, row[3] as u8];
                let token = if row[HIDDEN - 1] == want {
                    u32::from_le_bytes(bytes) as usize
                } else {
                    0
                };
                logits[j * VOCAB + token.min(VOCAB - 1)] = 1.0;
            }
            logits
        }
        fn reset(&mut self) {
            self.kv.clear();
        }
    }

    fn oracle(prompt_len: usize, eos_at: Option<usize>) -> Oracle {
        Oracle {
            prompt_len,
            eos_at,
            eos: EOS,
        }
    }

    fn server_rt() -> Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
    }

    struct Remote {
        addr: SocketAddr,
        seen: Arc<AtomicUsize>,
        /// One entry per finished session: `Ok` or the session's error text.
        sessions: mpsc::Receiver<Result<(), String>>,
    }

    /// Run a remote node on its own thread, serving `sessions` sessions back to back (the
    /// loop `potpori serve` runs). `tweak` makes the node's shard odd in some way.
    fn spawn_node(
        layers: (usize, usize),
        downstream: Option<SocketAddr>,
        sessions: usize,
        tls: TlsIdentity,
        tweak: impl FnOnce(&mut FakeShard),
    ) -> Remote {
        let (addr_tx, addr_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let mut shard = FakeShard::new(layers, oracle(0, None));
        tweak(&mut shard);
        let seen = shard.seen.clone();
        std::thread::spawn(move || {
            let rt = server_rt();
            let ep = {
                let _g = rt.enter();
                listen_endpoint("127.0.0.1:0".parse().unwrap(), &tls).unwrap()
            };
            addr_tx.send(ep.local_addr().unwrap()).unwrap();
            for _ in 0..sessions {
                shard.reset();
                let res = match downstream {
                    None => serve_tail(&mut shard, &ep, &rt),
                    Some(next) => serve_middle(&mut shard, &ep, next, &rt, &tls),
                };
                let _ = done_tx.send(res.map_err(|e| e.to_string()));
            }
        });
        Remote {
            addr: addr_rx.recv().unwrap(),
            seen,
            sessions: done_rx,
        }
    }

    /// Spawn the remote nodes of `plan` (tail first, so each middle knows its downstream) and
    /// return them in pipeline order.
    fn spawn_chain(plan: &[(usize, usize)], sessions: usize) -> Vec<Remote> {
        let mut remotes: Vec<Remote> = Vec::new();
        for &layers in plan[1..].iter().rev() {
            let downstream = remotes.last().map(|r| r.addr);
            let tls = TlsIdentity::for_tests();
            remotes.push(spawn_node(layers, downstream, sessions, tls, |_| {}));
        }
        remotes.reverse();
        remotes
    }

    fn connect(head: FakeShard, to: SocketAddr) -> Result<ChainLink<FakeShard>> {
        ChainLink::connect(head, to, &TlsIdentity::for_tests(), &[EOS], WAIT)
    }

    // The real protocol over real QUIC, 2 and 3 nodes: handshake, chunked prefill, pipelining,
    // divergence and overwrite, EOS — every draft quality must reproduce plain greedy decode.
    #[test]
    fn chain_over_quic_equals_greedy() {
        let prompt: Vec<TokenId> = vec![11, 12, 13, 14, 15, 16]; // one chunk + two singles
        for plan in [vec![(0, 4), (4, 8)], vec![(0, 3), (3, 5), (5, 8)]] {
            let remotes = spawn_chain(&plan, 1);
            let cfg = SpecConfig {
                gamma: GAMMA,
                max_new: 21,
            };
            // One session, many generations: each prefill at position 0 overwrites the KV.
            let head = FakeShard::new(plan[0], oracle(prompt.len(), None));
            let mut chain = connect(head, remotes[0].addr).unwrap();
            for eos_at in [None, Some(0), Some(9), Some(20)] {
                let o = oracle(prompt.len(), eos_at);
                chain.head.oracle = o;
                let expected = o.greedy(&prompt, cfg.max_new);
                for quality in QUALITIES {
                    for depth in [1usize, 2, 4] {
                        let mut draft = OracleDraft {
                            oracle: o,
                            quality,
                            calls: 0,
                        };
                        let (out, stats) =
                            run_spec_decode_pipelined(&mut draft, &mut chain, &prompt, &cfg, depth)
                                .unwrap();
                        let case = format!("{plan:?} eos_at={eos_at:?} {quality:?} depth={depth}");
                        assert_eq!(out, expected, "{case}");
                        assert_eq!(stats.committed, out.len(), "{case}");
                        assert!(chain.awaiting.is_empty(), "{case}");
                    }
                }
            }
            let stats = chain.stats();
            assert!(stats.frames > 0 && stats.payload_bytes >= stats.frames * HIDDEN as u64 * 4);
            chain.shutdown();
            for r in &remotes {
                assert_eq!(r.sessions.recv_timeout(WAIT).unwrap(), Ok(()));
            }
        }
    }

    // A misconfigured chain must fail at connect, telling the head which node refused and
    // which setting differs — and the node must stay up for the next, correct head.
    #[test]
    fn handshake_names_the_node_and_the_setting() {
        type Tweak = fn(&mut FakeShard);
        let bad: [(&str, (usize, usize), Tweak); 4] = [
            ("max_seq 64 (this node has 128)", (0, 4), |s| s.max_seq = 64),
            ("chunk 8 (this node has 4)", (0, 4), |s| s.chunk = 8),
            ("model \"other\"", (0, 4), |s| s.model = "other"),
            ("next layer 3 (this node starts at 4)", (0, 3), |_| {}),
        ];
        let id = TlsIdentity::for_tests();
        let tail = spawn_node((4, 8), None, bad.len() + 1, id, |_| {});
        for (want, layers, tweak) in bad {
            let mut head = FakeShard::new(layers, oracle(1, None));
            tweak(&mut head);
            let err = connect(head, tail.addr).err().expect(want).to_string();
            assert!(
                err.contains(want) && err.contains("node [4, 8)"),
                "{want}: got {err:?}"
            );
            assert!(tail.sessions.recv_timeout(WAIT).unwrap().is_err());
        }
        // Still serving: a correct head gets a working session.
        let prompt = vec![5];
        let o = oracle(prompt.len(), Some(6));
        let mut chain = connect(FakeShard::new((0, 4), o), tail.addr).unwrap();
        let mut draft = OracleDraft {
            oracle: o,
            quality: Quality::Perfect,
            calls: 0,
        };
        let cfg = SpecConfig {
            gamma: GAMMA,
            max_new: 50,
        };
        let (out, _) = run_spec_decode_pipelined(&mut draft, &mut chain, &prompt, &cfg, 2).unwrap();
        assert_eq!(out, o.greedy(&prompt, 50));
        chain.shutdown();
        assert_eq!(tail.sessions.recv_timeout(WAIT).unwrap(), Ok(()));
    }

    // A problem two hops away must reach the head at connect, naming the node at fault.
    #[test]
    fn problems_below_a_middle_reach_the_head_at_connect() {
        let id = TlsIdentity::for_tests;
        // The tail was started with a different cache size than the rest of the chain.
        let tail = spawn_node((5, 8), None, 1, id(), |s| s.max_seq = 64);
        let middle = spawn_node((3, 5), Some(tail.addr), 1, id(), |_| {});
        let head = FakeShard::new((0, 3), oracle(1, None));
        let err = connect(head, middle.addr).err().unwrap().to_string();
        assert!(
            err.contains("node [5, 8)") && err.contains("max_seq 128 (this node has 64)"),
            "{err:?}"
        );

        // A chain that stops short of the last layer (a "tail" configured as a middle's peer).
        let dead: SocketAddr = "127.0.0.1:9".parse().unwrap();
        let lonely = spawn_node((3, 5), Some(dead), 1, id(), |_| {});
        let head = FakeShard::new((0, 3), oracle(1, None));
        let err = ChainLink::connect(head, lonely.addr, &id(), &[EOS], Duration::from_secs(60))
            .err()
            .unwrap()
            .to_string();
        assert!(
            err.contains("node [3, 5)") && err.contains("cannot reach its downstream"),
            "{err:?}"
        );
    }

    #[test]
    fn unreachable_or_untrusted_nodes_fail_with_a_clear_error() {
        // Nothing listening: retried until `wait`, then a plain error.
        let dead: SocketAddr = "127.0.0.1:9".parse().unwrap();
        let head = FakeShard::new((0, 4), oracle(1, None));
        let wait = Duration::from_secs(1);
        let err = ChainLink::connect(head, dead, &TlsIdentity::for_tests(), &[EOS], wait)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("could not reach 127.0.0.1:9"), "{err:?}");

        // A node that is up but whose certificate the head does not trust: no point retrying.
        let tail = spawn_node((4, 8), None, 1, TlsIdentity::test_identity(), |_| {});
        let head = FakeShard::new((0, 4), oracle(1, None));
        let stranger = TlsIdentity::test_identity();
        let t0 = Instant::now();
        let err = ChainLink::connect(head, tail.addr, &stranger, &[EOS], Duration::from_secs(120))
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("peers.pem"), "{err:?}");
        assert!(
            t0.elapsed() < Duration::from_secs(30),
            "retried a TLS failure"
        );
    }

    // A peer that passes the handshake and then sends a malformed frame gets an Error back;
    // the node ends that session, never runs the frame, and serves the next head.
    #[test]
    fn malformed_frames_end_the_session_not_the_node() {
        let hello = FakeShard::new((0, 4), oracle(1, None)).hello(4);
        let good = HiddenStateFrame::from_f32(0, 0, 4, 1, HIDDEN as u32, &[0.0; HIDDEN]);
        let frames: Vec<(&str, HiddenStateFrame)> = vec![
            ("short payload", {
                let mut f = good.clone();
                f.data.truncate(4);
                f
            }),
            ("wrong boundary", {
                let mut f = good.clone();
                f.layer_boundary = 3;
                f
            }),
            ("no kernel for 3 tokens", {
                HiddenStateFrame::from_f32(0, 0, 4, 3, HIDDEN as u32, &[0.0; 3 * HIDDEN])
            }),
            ("past the cache", {
                let mut f = good.clone();
                f.start_pos = MAX_SEQ as u32;
                f
            }),
            ("NaN", {
                let mut f = good.clone();
                f.data[..4].copy_from_slice(&f32::NAN.to_le_bytes());
                f
            }),
        ];
        let id = TlsIdentity::for_tests();
        let tail = spawn_node((4, 8), None, frames.len() + 1, id.clone(), |_| {});
        let rt = server_rt();
        for (what, frame) in frames {
            let got = rt.block_on(async {
                let ep = dial_endpoint(&id).unwrap();
                let mut peer = dial(&ep, tail.addr).await.unwrap();
                negotiate(&mut peer, hello.clone(), TOTAL_LAYERS)
                    .await
                    .unwrap();
                peer.send(&PeerMessage::Hidden(frame)).await.unwrap();
                match peer.recv().await.unwrap() {
                    Some(PeerMessage::Error(why)) => why,
                    other => panic!("{what}: expected Error, got {other:?}"),
                }
            });
            assert!(
                got.contains("node [4, 8)") && got.contains("invalid hidden-state"),
                "{what}: got {got:?}"
            );
            assert!(tail.sessions.recv_timeout(WAIT).unwrap().is_err());
            assert_eq!(
                tail.seen.load(Ordering::SeqCst),
                0,
                "{what} reached the shard"
            );
        }
        let mut chain = connect(FakeShard::new((0, 4), oracle(1, None)), tail.addr).unwrap();
        assert_eq!(chain.prefill(&[7]).unwrap(), oracle(1, None).next(&[7]));
        chain.shutdown();
    }

    // Failures in the middle of a generation, on any node, must surface on the head as an
    // `Err` carrying the reason — never a panic or a hang — and end the session everywhere.
    #[test]
    fn mid_session_failures_reach_the_head() {
        let id = TlsIdentity::for_tests;
        type Tweak = fn(&mut FakeShard);
        // (what the head must be told, which node fails, how)
        let cases: [(&str, bool, Tweak); 3] = [
            ("node [5, 8): invalid hidden-state", true, |s| {
                s.capacity = 8
            }),
            ("node [3, 5): invalid hidden-state", false, |s| {
                s.capacity = 8
            }),
            ("node [5, 8): the shard produced", true, |s| s.nan = true),
        ];
        for (want, tail_fails, tweak) in cases {
            let ok: Tweak = |_| {};
            let (tail_tweak, middle_tweak) = if tail_fails { (tweak, ok) } else { (ok, tweak) };
            let tail = spawn_node((5, 8), None, 1, id(), tail_tweak);
            let middle = spawn_node((3, 5), Some(tail.addr), 1, id(), middle_tweak);
            let mut chain = connect(FakeShard::new((0, 3), oracle(4, None)), middle.addr).unwrap();
            // The prefill fits the small cache (or meets the NaN straight away); the second
            // chunk runs past position 8.
            let result = chain.prefill(&[1, 2, 3, 4]).and_then(|_| {
                chain.submit(4, 5, &[6, 7, 8])?;
                chain.poll()?;
                chain.submit(7, 8, &[9, 10, 11])?;
                chain.poll()
            });
            let err = result.unwrap_err().to_string();
            assert!(err.contains(want), "{want}: got {err:?}");
            drop(chain);
            assert!(
                middle.sessions.recv_timeout(WAIT).unwrap().is_err(),
                "{want}"
            );
            assert!(tail.sessions.recv_timeout(WAIT).unwrap().is_err(), "{want}");
        }
    }

    // A tail that answers with the wrong number of tokens, or disappears, is an error too.
    #[test]
    fn a_lying_or_vanishing_tail_is_an_error() {
        for vanish in [false, true] {
            let (addr_tx, addr_rx) = mpsc::channel();
            std::thread::spawn(move || {
                let rt = server_rt();
                let ep = {
                    let _g = rt.enter();
                    listen_endpoint("127.0.0.1:0".parse().unwrap(), &TlsIdentity::for_tests())
                        .unwrap()
                };
                addr_tx.send(ep.local_addr().unwrap()).unwrap();
                rt.block_on(async {
                    let mut up = accept(&ep).await.unwrap();
                    assert!(matches!(
                        up.recv().await.unwrap(),
                        Some(PeerMessage::Hello(_))
                    ));
                    let ready = PeerMessage::Ready {
                        final_layer: TOTAL_LAYERS,
                    };
                    up.send(&ready).await.unwrap();
                    let Some(PeerMessage::Hidden(frame)) = up.recv().await.unwrap() else {
                        panic!("expected a frame");
                    };
                    if vanish {
                        up.close();
                        ep.wait_idle().await; // let the close leave before the runtime dies
                    } else {
                        let result = VerifyResult {
                            step: frame.step,
                            argmax: vec![1; frame.n_tokens as usize - 1], // one short
                            taps: Vec::new(),
                        };
                        up.send(&PeerMessage::Result(result)).await.unwrap();
                        up.wait_closed().await;
                    }
                });
            });
            let head = FakeShard::new((0, 4), oracle(4, None));
            let mut chain = connect(head, addr_rx.recv().unwrap()).unwrap();
            let err = chain.prefill(&[1, 2, 3, 4]).unwrap_err().to_string();
            let want = if vanish {
                "lost the chain"
            } else {
                "answered wrongly"
            };
            assert!(err.contains(want), "vanish={vanish}: {err:?}");
        }
    }

    // Pipelining depends on a submitted frame leaving at once, not at the head's next
    // blocking call: the tail must see it while the head thread is busy elsewhere.
    #[test]
    fn submitted_frames_leave_while_the_head_is_busy() {
        let tail = spawn_node((4, 8), None, 1, TlsIdentity::for_tests(), |_| {});
        let mut chain = connect(FakeShard::new((0, 4), oracle(1, None)), tail.addr).unwrap();
        chain.prefill(&[9]).unwrap();
        let before = tail.seen.load(Ordering::SeqCst);
        chain.submit(1, 3, &[4, 5, 6]).unwrap();
        // The head "computes" (no runtime calls); the frame must reach the tail regardless.
        let t0 = Instant::now();
        while tail.seen.load(Ordering::SeqCst) == before && t0.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(tail.seen.load(Ordering::SeqCst), before + 1);
        assert_eq!(chain.poll().unwrap().len(), 4);
        chain.shutdown();
    }

    // ---- tapped sessions -----------------------------------------------------------------

    fn tapped(layers: (usize, usize), taps: &[usize], o: Oracle) -> TappedShards<FakeShard> {
        let parts = tap_ranges(layers, taps)
            .into_iter()
            .map(|range| FakeShard::new(range, o))
            .collect();
        TappedShards::new(parts, taps.to_vec()).unwrap()
    }

    /// Like [`spawn_chain`], with every node split at the tapped layers.
    fn spawn_tapped_chain(plan: &[(usize, usize)], taps: &[usize]) -> Vec<Remote> {
        let mut remotes: Vec<Remote> = Vec::new();
        for &layers in plan[1..].iter().rev() {
            let downstream = remotes.last().map(|r| r.addr);
            let (addr_tx, addr_rx) = mpsc::channel();
            let (done_tx, done_rx) = mpsc::channel();
            let mut shard = tapped(layers, taps, oracle(0, None));
            std::thread::spawn(move || {
                let rt = server_rt();
                let tls = TlsIdentity::for_tests();
                let ep = {
                    let _g = rt.enter();
                    listen_endpoint("127.0.0.1:0".parse().unwrap(), &tls).unwrap()
                };
                addr_tx.send(ep.local_addr().unwrap()).unwrap();
                let res = match downstream {
                    None => serve_tail(&mut shard, &ep, &rt),
                    Some(next) => serve_middle(&mut shard, &ep, next, &rt, &tls),
                };
                let _ = done_tx.send(res.map_err(|e| e.to_string()));
            });
            remotes.push(Remote {
                addr: addr_rx.recv().unwrap(),
                seen: Arc::default(),
                sessions: done_rx,
            });
        }
        remotes.reverse();
        remotes
    }

    #[test]
    fn nodes_are_cut_after_each_tapped_layer() {
        assert_eq!(
            tap_ranges((0, 18), &[1, 9, 17, 25, 33]),
            [(0, 2), (2, 10), (10, 18)]
        );
        assert_eq!(
            tap_ranges((18, 36), &[1, 9, 17, 25, 33]),
            [(18, 26), (26, 34), (34, 36)]
        );
        assert_eq!(tap_ranges((0, 8), &[]), [(0, 8)]);
        assert_eq!(
            tap_ranges((3, 5), &[4, 6]),
            [(3, 5)],
            "a tap on the node's last layer"
        );

        let o = oracle(0, None);
        let stack = tapped((0, 4), &[1, 2, 6], o);
        assert_eq!(stack.range(), (0, 4));
        assert_eq!(stack.tap_layers(), [1, 2]);
        assert_eq!(stack.hello(4).taps, [1, 2, 6]);
        // The last layer returns logits, not a hidden state.
        let whole = vec![FakeShard::new((0, TOTAL_LAYERS), o)];
        let err = TappedShards::new(whole, vec![TOTAL_LAYERS - 1])
            .err()
            .unwrap();
        assert!(err.to_string().contains("last layer"), "{err}");
        // Shards that do not end at the tapped layer cannot report it.
        let err = TappedShards::new(vec![FakeShard::new((0, 4), o)], vec![1])
            .err()
            .unwrap();
        assert!(err.to_string().contains("do not expose"), "{err}");
    }

    // A tapped session end to end: every node reports its tapped layers, the head assembles
    // them per position in layer order, and a draft that sees only those features drives the
    // generation — over two and three nodes, with taps on node boundaries and inside nodes.
    #[test]
    fn tapped_layers_reach_the_head_as_features() {
        use crate::specfeat::run_feature_decode;
        use crate::specfeat::tests::feature_draft;

        const TAPS: [usize; 4] = [1, 2, 4, 6];
        // One feature row: per tapped layer, the residual stream after it.
        let decode = |row: &[f32]| {
            let (blocks, rest) = row.as_chunks::<HIDDEN>();
            assert!(rest.is_empty() && blocks.len() == TAPS.len());
            for (block, layer) in blocks.iter().zip(TAPS) {
                let ran: f32 = (1..=layer + 1).map(|l| l as f32).sum();
                assert_eq!(block[HIDDEN - 1], ran, "layer {layer}: wrong layers ran");
                assert_eq!(
                    block[..4],
                    blocks[0][..4],
                    "layer {layer}: another position"
                );
            }
            u32::from_le_bytes(std::array::from_fn(|i| blocks[0][i] as u8)) as TokenId
        };
        let prompt: Vec<TokenId> = vec![11, 12, 13, 14, 15, 16]; // one chunk + two singles
        for plan in [vec![(0, 4), (4, 8)], vec![(0, 3), (3, 5), (5, 8)]] {
            let remotes = spawn_tapped_chain(&plan, &TAPS);
            let o = oracle(prompt.len(), Some(40));
            let head = tapped(plan[0], &TAPS, o);
            let mut chain = ChainLink::connect(
                head,
                remotes[0].addr,
                &TlsIdentity::for_tests(),
                &[EOS],
                WAIT,
            )
            .unwrap();
            let expected = o.greedy(&prompt, 64);
            assert_eq!(expected.len(), 40);
            for quality in QUALITIES {
                let mut draft =
                    feature_draft(o, quality, &prompt, GAMMA, TAPS.len() * HIDDEN, decode);
                let (out, stats) = run_feature_decode(&mut draft, &mut chain, &prompt, 64).unwrap();
                assert_eq!(out, expected, "{plan:?} {quality:?}");
                if matches!(quality, Quality::Perfect) {
                    assert_eq!(stats.traversals, 40 / CHUNK, "{plan:?}");
                }
                assert!(chain.awaiting.is_empty());
            }
            // The plain pipelined driver still works on a tapped session; it ignores features.
            let mut draft = OracleDraft {
                oracle: o,
                quality: Quality::Perfect,
                calls: 0,
            };
            let cfg = SpecConfig {
                gamma: GAMMA,
                max_new: 64,
            };
            let (out, _) =
                run_spec_decode_pipelined(&mut draft, &mut chain, &prompt, &cfg, 3).unwrap();
            assert_eq!(out, expected);
            assert!(chain.stats().tap_bytes > 0);
            chain.shutdown();
            for remote in &remotes {
                assert_eq!(remote.sessions.recv_timeout(WAIT).unwrap(), Ok(()));
            }
        }
    }

    #[test]
    fn nodes_must_agree_on_the_tapped_layers() {
        // The tail taps nothing; the head asks for layers 1 and 5.
        let tail = spawn_node((4, 8), None, 1, TlsIdentity::for_tests(), |_| {});
        let head = tapped((0, 4), &[1, 5], oracle(1, None));
        let err = ChainLink::connect(head, tail.addr, &TlsIdentity::for_tests(), &[EOS], WAIT)
            .err()
            .unwrap()
            .to_string();
        assert!(
            err.contains("node [4, 8)") && err.contains("tapped layers [1, 5]"),
            "{err}"
        );
    }
}

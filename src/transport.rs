//! Node-to-node hidden-state transport for the shard chain.
//!
//! The residual-stream hidden state crosses each shard boundary here, over a quinn bi-stream
//! with length-prefixed msgpack framing ([`read_framed`]/[`write_framed`]).
//!
//! This module is the WIRE. [`crate::qwen::shard::ShardedChain`] does the in-process hand-off; a
//! remote shard replaces that one `Vec<f32>` hop with [`PeerConnection::send`] of a
//! [`HiddenStateFrame`]. The hidden round-trip is lossless (f32 bytes → msgpack bin → f32).
//!
//! NAT traversal / brokered path formation and the activation codec are later (see
//! `docs/chain-formation-design.md`, `docs/pipeline-specdecode-design.md`).
#![allow(dead_code)]

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use quinn::{ClientConfig, Endpoint, RecvStream, SendStream, ServerConfig, TransportConfig};
use rustls::pki_types::CertificateDer;
use serde::{Deserialize, Serialize};

use crate::engine::TokenId;

/// ALPN for node↔node shard streams.
const ALPN: &[u8] = b"potpori-shard-v2";

// ---------------------------------------------------------------------------
// Length-prefixed msgpack framing (inlined from dkn-protocol's framing.rs)
// Wire format: [4-byte BE length][msgpack payload]
// ---------------------------------------------------------------------------

/// Maximum allowed message size (10 MB — a ~2 MB f32 hidden state fits comfortably).
pub const MAX_MESSAGE_SIZE: u32 = 10 * 1024 * 1024;

/// Write a length-prefixed MessagePack message to a QUIC send stream.
pub async fn write_framed<T: Serialize>(send: &mut SendStream, msg: &T) -> io::Result<()> {
    let payload = rmp_serde::to_vec(msg).map_err(io_other)?;
    let len = payload.len() as u32;
    if len > MAX_MESSAGE_SIZE {
        return Err(io_other(format!(
            "message too large: {len} bytes (max {MAX_MESSAGE_SIZE})"
        )));
    }
    send.write_all(&len.to_be_bytes()).await.map_err(io_other)?;
    send.write_all(&payload).await.map_err(io_other)?;
    Ok(())
}

/// Read a length-prefixed MessagePack message; `Ok(None)` on clean EOF (stream finished).
pub async fn read_framed<T: serde::de::DeserializeOwned>(
    recv: &mut RecvStream,
) -> io::Result<Option<T>> {
    let mut len_buf = [0u8; 4];
    match recv.read_exact(&mut len_buf).await {
        Ok(()) => {}
        Err(quinn::ReadExactError::FinishedEarly(0)) => return Ok(None),
        Err(e) => return Err(io_other(format!("read length: {e}"))),
    }
    let len = u32::from_be_bytes(len_buf);
    if len > MAX_MESSAGE_SIZE {
        return Err(io_other(format!(
            "message too large: {len} bytes (max {MAX_MESSAGE_SIZE})"
        )));
    }
    let mut payload = vec![0u8; len as usize];
    recv.read_exact(&mut payload).await.map_err(io_other)?;
    rmp_serde::from_slice(&payload).map(Some).map_err(io_other)
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// A residual-stream hidden state crossing a shard boundary: logically `[n_tokens, hidden]`.
/// Bandwidth is small (~12 KiB/token at H=6144, bf16) — the wire is RTT-bound, not byte-bound —
/// so the activation codec (`dtype`) is a second-order optimization for thin home uplinks, not
/// the latency lever. The payload is the row-major hidden state in little-endian `dtype`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HiddenStateFrame {
    pub request_id: u64,
    pub chain_id: u64,
    /// Reserved for future chain re-planning; current sessions use epoch zero.
    pub epoch: u64,
    pub step: u64,
    /// Absolute sequence position of the first token in this chunk (the spec-decode `start_pos`);
    /// every shard writes its KV at the same absolute positions, which makes rollback a no-op.
    pub start_pos: u32,
    /// Which layer-block boundary this hidden state sits at (the cut index).
    pub layer_boundary: u32,
    pub n_tokens: u32,
    pub hidden_dim: u32,
    /// Activation codec on the wire, e.g. "f32" | "bf16" | "fp8".
    pub dtype: String,
    pub data: Vec<u8>,
}

impl HiddenStateFrame {
    /// Build an f32 hidden-state frame from a flat `[n_tokens * hidden_dim]` residual stream.
    #[allow(clippy::too_many_arguments)]
    pub fn from_f32(
        request_id: u64,
        epoch: u64,
        step: u64,
        start_pos: u32,
        layer_boundary: u32,
        n_tokens: u32,
        hidden_dim: u32,
        hidden: &[f32],
    ) -> Self {
        Self {
            request_id,
            chain_id: 0,
            epoch,
            step,
            start_pos,
            layer_boundary,
            n_tokens,
            hidden_dim,
            dtype: "f32".to_string(),
            data: f32_to_le_bytes(hidden),
        }
    }

    /// Decode the payload back to a flat f32 residual stream (only `dtype == "f32"` for now).
    pub fn to_f32(&self) -> Vec<f32> {
        assert_eq!(self.dtype, "f32", "only the f32 codec is implemented");
        le_bytes_to_f32(&self.data)
    }
    pub fn validate(
        &self,
        hidden: usize,
        boundary: usize,
        capacity: usize,
        chunk: usize,
    ) -> io::Result<()> {
        let s = self.n_tokens as usize;
        let bytes = s.checked_mul(hidden).and_then(|n| n.checked_mul(4));
        if self.dtype != "f32"
            || self.hidden_dim as usize != hidden
            || self.layer_boundary as usize != boundary
            || s == 0
            || bytes != Some(self.data.len())
            || (chunk != 0 && s != 1 && s != chunk)
            || (self.start_pos as usize)
                .checked_add(s)
                .is_none_or(|end| end > capacity)
            || self.to_f32().iter().any(|x| !x.is_finite())
        {
            return Err(io_other(
                "invalid hidden-state shape, dtype, boundary, values, or position",
            ));
        }
        Ok(())
    }
}

/// The tail owns `lm_head` and returns the target's greedy argmax at each verify position.
/// Intermediate shards relay the result upstream to the head for acceptance checking.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyResult {
    pub request_id: u64,
    pub step: u64,
    pub argmax: Vec<TokenId>,
}
impl VerifyResult {
    pub fn validate(
        &self,
        request_id: u64,
        step: u64,
        count: usize,
        vocab: usize,
    ) -> io::Result<()> {
        if self.request_id != request_id
            || self.step != step
            || self.argmax.len() != count
            || self.argmax.iter().any(|&t| t < 0 || t as usize >= vocab)
        {
            return Err(io_other(
                "invalid verification result correlation, length, or token",
            ));
        }
        Ok(())
    }
}

/// Roll back every shard's KV to the committed prefix. With absolute-position KV writes this is
/// advisory (the next chunk overwrites the rejected tail); maps to [`crate::engine::TargetChain::truncate`].
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct KvTruncate {
    pub request_id: u64,
    pub epoch: u64,
    pub keep_to_pos: u32,
}

/// One framed message on a shard bi-stream.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PeerMessage {
    Hello(SessionHello),
    Ready { final_layer: usize },
    Hidden(HiddenStateFrame),
    Result(VerifyResult),
    Truncate(KvTruncate),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionHello {
    pub model: String,
    pub next_layer: usize,
    pub max_seq: usize,
    pub chunk: usize,
}
pub async fn negotiate(
    peer: &mut PeerConnection,
    hello: SessionHello,
    final_layer: usize,
) -> io::Result<()> {
    peer.send(&PeerMessage::Hello(hello)).await?;
    match tokio::time::timeout(Duration::from_secs(30), peer.recv())
        .await
        .map_err(io_other)??
    {
        Some(PeerMessage::Ready { final_layer: got }) if got == final_layer => Ok(()),
        _ => Err(io_other(
            "peer did not confirm the complete compatible model chain",
        )),
    }
}
pub async fn receive_hello(peer: &mut PeerConnection, expected: &SessionHello) -> io::Result<()> {
    match tokio::time::timeout(Duration::from_secs(30), peer.recv())
        .await
        .map_err(io_other)??
    {
        Some(PeerMessage::Hello(got)) if got == *expected => Ok(()),
        _ => Err(io_other(
            "peer model, layer boundary, cache capacity, or chunk differs",
        )),
    }
}

fn f32_to_le_bytes(xs: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn le_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Duplex link to one adjacent peer
// ---------------------------------------------------------------------------

/// A duplex link to an adjacent shard, supporting ordered pipelined requests and results.
pub struct PeerConnection {
    // Held to keep the QUIC connection open: dropping the last Connection handle closes it,
    // even while the stream halves are still alive.
    conn: quinn::Connection,
    send: SendStream,
    recv: RecvStream,
}

impl PeerConnection {
    pub fn new(conn: quinn::Connection, send: SendStream, recv: RecvStream) -> Self {
        Self { conn, send, recv }
    }

    pub async fn send(&mut self, msg: &PeerMessage) -> io::Result<()> {
        write_framed(&mut self.send, msg).await.map_err(io_other)
    }

    /// Read the next message; `Ok(None)` on clean stream EOF.
    pub async fn recv(&mut self) -> io::Result<Option<PeerMessage>> {
        read_framed(&mut self.recv).await.map_err(io_other)
    }

    /// Close this side of the connection (call after the final frame is read). Dropping the
    /// connection sends an immediate CONNECTION_CLOSE that can discard un-read data, so the
    /// reader closes explicitly and the writer awaits [`wait_closed`] before tearing down.
    pub fn close(&self) {
        self.conn.close(0u32.into(), b"bye");
    }

    /// Await the peer closing the connection — the writer of the last frame awaits this so it
    /// does not drop (and truncate) before the reader has the frame.
    pub async fn wait_closed(&self) {
        self.conn.closed().await;
    }

    /// Finish the send half cleanly (FIN). The peer's next `recv` returns `Ok(None)` (clean EOF)
    /// rather than a "connection lost" error — used to end a session gracefully.
    pub fn finish(&mut self) {
        let _ = self.send.finish();
    }
    fn into_parts(self) -> (quinn::Connection, SendStream, RecvStream) {
        (self.conn, self.send, self.recv)
    }
}

/// Bounded duplex relay: GPU computation and forwarding proceed independently of result reads.
/// One closure invocation owns the shard at a time; request/result order remains FIFO.
pub async fn relay_middle<F>(
    up: PeerConnection,
    down: PeerConnection,
    mut forward: F,
) -> io::Result<()>
where
    F: FnMut(HiddenStateFrame) -> io::Result<HiddenStateFrame>,
{
    let (_up_conn, mut up_send, mut up_recv) = up.into_parts();
    let (down_conn, mut down_send, mut down_recv) = down.into_parts();
    let (pending_tx, mut pending_rx) = tokio::sync::mpsc::channel::<(u64, u64, usize)>(64);
    let transmit = async {
        while let Some(msg) = read_framed::<PeerMessage>(&mut up_recv).await? {
            match msg {
                PeerMessage::Hidden(frame) => {
                    let expected = (frame.request_id, frame.step, frame.n_tokens as usize);
                    let out = forward(frame)?;
                    pending_tx.send(expected).await.map_err(io_other)?;
                    write_framed(&mut down_send, &PeerMessage::Hidden(out)).await?;
                }
                PeerMessage::Truncate(_) => {}
                _ => return Err(io_other("unexpected message during shard session")),
            }
        }
        down_send.finish().map_err(io_other)?;
        drop(pending_tx);
        io::Result::Ok(())
    };
    let receive = async {
        while let Some((request_id, step, n)) = pending_rx.recv().await {
            let result = match read_framed::<PeerMessage>(&mut down_recv).await? {
                Some(PeerMessage::Result(r))
                    if r.request_id == request_id && r.step == step && r.argmax.len() == n =>
                {
                    r
                }
                _ => return Err(io_other("invalid or missing downstream result")),
            };
            write_framed(&mut up_send, &PeerMessage::Result(result)).await?;
        }
        up_send.finish().map_err(io_other)?;
        io::Result::Ok(())
    };
    tokio::try_join!(transmit, receive)?;
    down_conn.closed().await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Endpoints authenticate both peers against the configured certificate bundle.
// ---------------------------------------------------------------------------

/// Each node presents its own certificate and trusts only explicitly supplied peer certificates.
pub struct TlsIdentity {
    certs: Vec<CertificateDer<'static>>,
    key: rustls::pki_types::PrivateKeyDer<'static>,
    roots: rustls::RootCertStore,
}
impl Clone for TlsIdentity {
    fn clone(&self) -> Self {
        Self {
            certs: self.certs.clone(),
            key: self.key.clone_key(),
            roots: self.roots.clone(),
        }
    }
}
impl TlsIdentity {
    pub fn load(
        cert: &std::path::Path,
        key: &std::path::Path,
        trust: &std::path::Path,
    ) -> anyhow::Result<Self> {
        let certs = rustls_pemfile::certs(&mut std::io::BufReader::new(std::fs::File::open(cert)?))
            .collect::<Result<Vec<_>, _>>()?;
        anyhow::ensure!(!certs.is_empty(), "identity certificate file is empty");
        let key =
            rustls_pemfile::private_key(&mut std::io::BufReader::new(std::fs::File::open(key)?))?
                .ok_or_else(|| anyhow::anyhow!("private key missing"))?;
        let mut roots = rustls::RootCertStore::empty();
        for cert in rustls_pemfile::certs(&mut std::io::BufReader::new(std::fs::File::open(trust)?))
        {
            roots.add(cert?)?;
        }
        anyhow::ensure!(!roots.is_empty(), "trusted peer certificate file is empty");
        Ok(Self { certs, key, roots })
    }
    pub fn generate(directory: &std::path::Path) -> anyhow::Result<()> {
        use std::io::Write;
        std::fs::create_dir_all(directory)?;
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()])?;
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut key = options.open(directory.join("key.pem"))?;
        key.write_all(cert.key_pair.serialize_pem().as_bytes())?;
        let mut public = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("cert.pem"))?;
        public.write_all(cert.cert.pem().as_bytes())?;
        Ok(())
    }
    #[cfg(test)]
    pub fn for_tests() -> Self {
        static ID: std::sync::OnceLock<TlsIdentity> = std::sync::OnceLock::new();
        ID.get_or_init(Self::test_identity).clone()
    }
    #[cfg(test)]
    fn test_identity() -> Self {
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let cert_der = CertificateDer::from(cert.cert);
        let mut roots = rustls::RootCertStore::empty();
        roots.add(cert_der.clone()).unwrap();
        Self {
            certs: vec![cert_der],
            key: rustls::pki_types::PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der()).into(),
            roots,
        }
    }
}

pub fn dial_endpoint(identity: &TlsIdentity) -> io::Result<Endpoint> {
    let mut crypto = rustls::ClientConfig::builder()
        .with_root_certificates(identity.roots.clone())
        .with_client_auth_cert(identity.certs.clone(), identity.key.clone_key())
        .map_err(io_other)?;
    crypto.alpn_protocols = vec![ALPN.to_vec()];
    let mut config = ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto).map_err(io_other)?,
    ));
    config.transport_config(Arc::new(keepalive_transport()));
    let mut ep = Endpoint::client("0.0.0.0:0".parse().unwrap()).map_err(io_other)?;
    ep.set_default_client_config(config);
    Ok(ep)
}

pub fn listen_endpoint(addr: SocketAddr, identity: &TlsIdentity) -> io::Result<Endpoint> {
    let verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(identity.roots.clone()))
        .build()
        .map_err(io_other)?;
    let mut crypto = rustls::ServerConfig::builder()
        .with_client_cert_verifier(verifier)
        .with_single_cert(identity.certs.clone(), identity.key.clone_key())
        .map_err(io_other)?;
    crypto.alpn_protocols = vec![ALPN.to_vec()];
    let mut config = ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(crypto).map_err(io_other)?,
    ));
    let mut transport = keepalive_transport();
    transport.max_concurrent_bidi_streams(1u32.into());
    config.transport_config(Arc::new(transport));
    Endpoint::server(config, addr).map_err(io_other)
}

/// Transport config with keepalives so a persistent shard link survives long idle gaps (e.g. a
/// multi-minute model load before the first hop) without hitting the idle timeout.
fn keepalive_transport() -> TransportConfig {
    let mut transport = TransportConfig::default();
    transport.keep_alive_interval(Some(Duration::from_secs(5)));
    transport.max_idle_timeout(Some(
        Duration::from_secs(30)
            .try_into()
            .expect("valid idle timeout"),
    ));
    transport
}

/// Dial `addr` and open one bi-stream for a hop (upstream node → this peer).
pub async fn dial(ep: &Endpoint, addr: SocketAddr) -> io::Result<PeerConnection> {
    let conn = ep
        .connect(addr, "localhost")
        .map_err(io_other)?
        .await
        .map_err(io_other)?;
    let (send, recv) = conn.open_bi().await.map_err(io_other)?;
    Ok(PeerConnection::new(conn, send, recv))
}

/// Accept one inbound connection + its first bi-stream (resolves once the dialer sends).
pub async fn accept(ep: &Endpoint) -> io::Result<PeerConnection> {
    let incoming = ep
        .accept()
        .await
        .ok_or_else(|| io_other("endpoint closed"))?;
    let conn = incoming.await.map_err(io_other)?;
    let (send, recv) = conn.accept_bi().await.map_err(io_other)?;
    Ok(PeerConnection::new(conn, send, recv))
}

fn io_other<E: ToString>(e: E) -> io::Error {
    io::Error::other(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_verification_results() {
        let valid = VerifyResult {
            request_id: 5,
            step: 7,
            argmax: vec![0, 15],
        };
        valid.validate(5, 7, 2, 16).unwrap();
        assert!(valid.validate(6, 7, 2, 16).is_err());
        assert!(valid.validate(5, 8, 2, 16).is_err());
        assert!(valid.validate(5, 7, 1, 16).is_err());
        assert!(valid.validate(5, 7, 2, 15).is_err());
        let invalid = VerifyResult {
            argmax: vec![-1, 1],
            ..valid
        };
        assert!(invalid.validate(5, 7, 2, 16).is_err());
    }

    #[test]
    fn rejects_malformed_frames() {
        let valid = HiddenStateFrame::from_f32(1, 0, 1, 6, 12, 2, 4, &[0.; 8]);
        valid.validate(4, 12, 8, 2).unwrap();
        for case in 0..7 {
            let mut f = valid.clone();
            match case {
                0 => f.data.pop().map(|_| ()).unwrap(),
                1 => f.dtype = "bf16".into(),
                2 => f.start_pos = 7,
                3 => f.hidden_dim = 3,
                4 => f.layer_boundary = 11,
                5 => f.n_tokens = 0,
                _ => f.data[..4].copy_from_slice(&f32::NAN.to_le_bytes()),
            }
            assert!(f.validate(4, 12, 8, 2).is_err(), "case {case}");
        }
    }

    #[tokio::test]
    async fn untrusted_server_and_client_are_rejected() {
        let a = TlsIdentity::test_identity();
        let mut b = TlsIdentity::test_identity();
        for trusts_server in [false, true] {
            if trusts_server {
                b.roots.add(a.certs[0].clone()).unwrap();
            }
            let server = listen_endpoint("127.0.0.1:0".parse().unwrap(), &a).unwrap();
            let client = dial_endpoint(&b).unwrap();
            let addr = server.local_addr().unwrap();
            let accept_fut = async {
                assert!(accept(&server).await.is_err());
            };
            let dial_fut = async {
                if let Ok(mut peer) = dial(&client, addr).await {
                    let _ = peer.send(&PeerMessage::Ready { final_layer: 1 }).await;
                    assert!(peer.recv().await.is_err());
                }
            };
            tokio::time::timeout(Duration::from_secs(5), async {
                tokio::join!(accept_fut, dial_fut);
            })
            .await
            .unwrap();
        }
    }

    #[tokio::test]
    async fn rejects_mismatched_session_settings() {
        let id = TlsIdentity::for_tests();
        let server = listen_endpoint("127.0.0.1:0".parse().unwrap(), &id).unwrap();
        let client = dial_endpoint(&id).unwrap();
        let addr = server.local_addr().unwrap();
        let expected = SessionHello {
            model: "qwen".into(),
            next_layer: 12,
            max_seq: 768,
            chunk: 8,
        };
        let mut wrong = expected.clone();
        wrong.max_seq = 512;
        let server_fut = async {
            let mut p = accept(&server).await.unwrap();
            assert!(receive_hello(&mut p, &expected).await.is_err());
        };
        let client_fut = async {
            let mut p = dial(&client, addr).await.unwrap();
            assert!(negotiate(&mut p, wrong, 36).await.is_err());
        };
        tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(server_fut, client_fut);
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn middle_forwards_next_chunk_before_first_reply() {
        let id = TlsIdentity::for_tests();
        let tail = listen_endpoint("127.0.0.1:0".parse().unwrap(), &id).unwrap();
        let middle = listen_endpoint("127.0.0.1:0".parse().unwrap(), &id).unwrap();
        let head = dial_endpoint(&id).unwrap();
        let middle_addr = middle.local_addr().unwrap();
        let tail_addr = tail.local_addr().unwrap();
        let tail_fut = async {
            let mut p = accept(&tail).await.unwrap();
            let mut frames = Vec::new();
            // Do not reply until two chunks arrive: a serial middle would deadlock here.
            for _ in 0..2 {
                if let Some(PeerMessage::Hidden(f)) = p.recv().await.unwrap() {
                    frames.push(f);
                } else {
                    panic!("expected hidden");
                }
            }
            for f in frames {
                p.send(&PeerMessage::Result(VerifyResult {
                    request_id: f.request_id,
                    step: f.step,
                    argmax: vec![f.step as i32],
                }))
                .await
                .unwrap();
            }
            assert!(p.recv().await.unwrap().is_none());
        };
        let middle_fut = async {
            let up = accept(&middle).await.unwrap();
            let ep = dial_endpoint(&id).unwrap();
            let down = dial(&ep, tail_addr).await.unwrap();
            relay_middle(up, down, Ok).await.unwrap();
        };
        let head_fut = async {
            let mut p = dial(&head, middle_addr).await.unwrap();
            for step in 1..=2 {
                p.send(&PeerMessage::Hidden(HiddenStateFrame::from_f32(
                    step,
                    0,
                    step,
                    step as u32,
                    1,
                    1,
                    1,
                    &[0.],
                )))
                .await
                .unwrap();
            }
            for step in 1..=2 {
                match p.recv().await.unwrap() {
                    Some(PeerMessage::Result(r)) => assert_eq!(r.argmax, vec![step]),
                    _ => panic!("expected result"),
                }
            }
            p.finish();
            p.wait_closed().await;
        };
        tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(tail_fut, middle_fut, head_fut);
        })
        .await
        .unwrap();
    }

    // The node→node hop: a hidden state crosses a real QUIC link and the tail returns a result.
    // Proves the wire (serialization + framing + bi-stream) end to end, GPU-free.
    #[tokio::test]
    async fn hidden_state_roundtrips_over_quic() {
        let server =
            listen_endpoint("127.0.0.1:0".parse().unwrap(), &TlsIdentity::for_tests()).unwrap();
        let addr = server.local_addr().unwrap();

        // A residual stream with negatives + fractions to catch any byte/codec corruption.
        let hidden: Vec<f32> = (0..32).map(|i| (i as f32) * 0.37 - 5.5).collect();
        let hidden_expect = hidden.clone();

        // Tail peer: receive the hidden state, "run its shard", return a VerifyResult.
        let server_fut = async move {
            let mut peer = accept(&server).await?;
            let frame = match peer.recv().await?.expect("clean EOF before a frame") {
                PeerMessage::Hidden(f) => f,
                other => panic!("expected Hidden, got {other:?}"),
            };
            let got = frame.to_f32();
            // Stand-in for the last shard's argmax: report what we received so the client can check.
            let argmax = vec![
                got.len() as TokenId,
                frame.n_tokens as TokenId,
                frame.start_pos as TokenId,
            ];
            peer.send(&PeerMessage::Result(VerifyResult {
                request_id: frame.request_id,
                step: frame.step,
                argmax,
            }))
            .await?;
            peer.wait_closed().await; // hold the connection open until the client has the result
            io::Result::Ok(got)
        };

        // Owner peer (node-0): dial, send the hidden state, read the result.
        let client = dial_endpoint(&TlsIdentity::for_tests()).unwrap();
        let client_fut = async {
            let mut peer = dial(&client, addr).await?;
            let frame = HiddenStateFrame::from_f32(42, 1, 0, 7, 14, 4, 8, &hidden);
            peer.send(&PeerMessage::Hidden(frame)).await?;
            let result = match peer.recv().await?.expect("clean EOF before a result") {
                PeerMessage::Result(r) => r,
                other => panic!("expected Result, got {other:?}"),
            };
            peer.close(); // release the server's wait_closed
            io::Result::Ok(result)
        };

        let (server_got, result) = tokio::join!(server_fut, client_fut);
        let server_got = server_got.expect("server side");
        let result = result.expect("client side");
        assert_eq!(
            server_got, hidden_expect,
            "hidden state must survive the QUIC round-trip exactly"
        );
        assert_eq!(result.request_id, 42);
        assert_eq!(
            result.argmax,
            vec![32, 4, 7],
            "n_floats=32, n_tokens=4, start_pos=7"
        );
    }
}

//! Node-to-node hidden-state transport for the shard chain.
//!
//! The residual-stream hidden state crosses each shard boundary here, over a quinn bi-stream
//! with length-prefixed msgpack framing ([`read_framed`]/[`write_framed`]).
//!
//! This module is the WIRE: message types, framing and QUIC endpoints. [`crate::node`] is the
//! protocol on top (handshake, serve loops, the head's pipelined link). The hidden round-trip is
//! lossless (f32 bytes → msgpack bin → f32).
//!
//! NAT traversal / brokered path formation and the activation codec are later (see
//! `docs/chain-formation-design.md`, `docs/pipeline-specdecode-design.md`).
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use quinn::{ClientConfig, Endpoint, RecvStream, SendStream, ServerConfig, TransportConfig};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::DigitallySignedStruct;
use serde::{Deserialize, Serialize};

use crate::engine::TokenId;

/// ALPN for node↔node shard streams.
const ALPN: &[u8] = b"potpori-shard";

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
        return Err(io_other(format!("message too large: {len} bytes (max {MAX_MESSAGE_SIZE})")));
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
        Err(quinn::ReadExactError::FinishedEarly(_)) => return Ok(None),
        Err(e) => return Err(io_other(format!("read length: {e}"))),
    }
    let len = u32::from_be_bytes(len_buf);
    if len > MAX_MESSAGE_SIZE {
        return Err(io_other(format!("message too large: {len} bytes (max {MAX_MESSAGE_SIZE})")));
    }
    let mut payload = vec![0u8; len as usize];
    recv.read_exact(&mut payload).await.map_err(io_other)?;
    rmp_serde::from_slice(&payload).map(Some).map_err(io_other)
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// Wire protocol version. Bump on any incompatible change to the types below; nodes on
/// different versions refuse each other at the handshake instead of mis-parsing frames.
pub const PROTO_VERSION: u32 = 1;

/// First message on every session, sent by the upstream node. The head describes the whole
/// chain plan; each node checks its own seat against how it was started, so a mismatch (wrong
/// model, layer range, chunk size, …) fails at connect with a readable error instead of
/// producing garbage tokens or a missing-kernel panic mid-generation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hello {
    pub proto: u32,
    /// Shared secret; must equal the node's `--token` when it was started with one.
    pub token: Option<String>,
    /// Target model HF repo.
    pub model: String,
    /// Verify chunk size (gamma + 1): every hidden-state frame carries 1 or `chunk` tokens.
    pub chunk: u32,
    /// KV positions the chain will use; every node needs `--max-seq` at least this large.
    pub max_seq: u32,
    pub hidden_dim: u32,
    /// Layer range `[start, end)` of every node in pipeline order; `plan[0]` is the head.
    pub plan: Vec<[u32; 2]>,
    /// The receiving node's seat in `plan` (>= 1).
    pub index: u32,
}

/// What a remote node reports about itself in the handshake reply (logged by the head).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeInfo {
    pub layers: [u32; 2],
    /// e.g. "cuda/Bf16", "metal/F32".
    pub backend: String,
    pub version: String,
}

/// A residual-stream hidden state crossing a shard boundary: logically `[n_tokens, hidden]`,
/// row-major, little-endian `dtype`. Bandwidth is small (tens of KiB per chunk) — the wire is
/// RTT-bound, not byte-bound — so the activation codec is a second-order optimization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HiddenStateFrame {
    /// Monotonic per-session frame counter; echoed in the matching [`VerifyResult`].
    pub step: u64,
    /// Absolute sequence position of the first token in this chunk; every shard writes its KV
    /// at the same absolute positions, which makes speculative rollback a plain overwrite.
    pub start_pos: u32,
    pub n_tokens: u32,
    pub hidden_dim: u32,
    /// Activation codec on the wire. Only "f32" exists today.
    pub dtype: String,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

impl HiddenStateFrame {
    /// Build an f32 hidden-state frame from a flat `[n_tokens * hidden_dim]` residual stream.
    pub fn from_f32(step: u64, start_pos: u32, n_tokens: u32, hidden_dim: u32, hidden: &[f32]) -> Self {
        Self { step, start_pos, n_tokens, hidden_dim, dtype: "f32".to_string(), data: f32_to_le_bytes(hidden) }
    }

    /// Decode the payload back to a flat f32 residual stream (callers check `dtype == "f32"`).
    pub fn to_f32(&self) -> Vec<f32> {
        le_bytes_to_f32(&self.data)
    }
}

/// The tail's answer to one [`HiddenStateFrame`]: the target's greedy argmax at each of its
/// positions. (For true sampling later this carries logprobs instead — see design docs.)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyResult {
    pub step: u64,
    pub argmax: Vec<TokenId>,
}

/// One framed message on a shard bi-stream.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PeerMessage {
    /// Upstream → downstream, first message of a session.
    Hello(Hello),
    /// Downstream → upstream: this node and everything below it accepted the plan. Nodes are
    /// listed in pipeline order starting with the replying node.
    HelloAck(Vec<NodeInfo>),
    Hidden(HiddenStateFrame),
    Result(VerifyResult),
    /// Downstream → upstream: the session is over, and why. Relayed up to the head.
    Error(String),
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
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Duplex link to one adjacent peer
// ---------------------------------------------------------------------------

/// A duplex link to one adjacent shard (next/prev hop), or tail→owner direct return: the two
/// halves of a single quinn bi-stream, with one in-flight (request, hop). Epoch fencing and
/// app-level backpressure layer on top when the chain control-plane lands.
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
    #[cfg(test)]
    pub fn close(&self) {
        self.conn.close(0u32.into(), b"bye");
    }

    /// Await the peer closing the connection — the writer of the last frame awaits this so it
    /// does not drop (and truncate) before the reader has the frame.
    pub async fn wait_closed(&self) {
        self.conn.closed().await;
    }

    /// QUIC's current round-trip estimate to this peer.
    pub fn rtt(&self) -> Duration {
        self.conn.rtt()
    }

    /// Finish the send half cleanly (FIN). The peer's next `recv` returns `Ok(None)` (clean EOF)
    /// rather than a "connection lost" error — used to end a session gracefully.
    pub fn finish(&mut self) {
        let _ = self.send.finish();
    }
}

// ---------------------------------------------------------------------------
// Endpoints (insecure TLS — dev/test + `--insecure`; prod pins peer certs at path formation)
// ---------------------------------------------------------------------------

/// A client endpoint that dials shard peers (accepts any server cert).
pub fn dial_endpoint() -> io::Result<Endpoint> {
    let mut crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();
    crypto.alpn_protocols = vec![ALPN.to_vec()];
    let mut client_config = ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto).map_err(io_other)?,
    ));
    client_config.transport_config(Arc::new(keepalive_transport()));
    let mut ep = Endpoint::client("0.0.0.0:0".parse().unwrap()).map_err(io_other)?;
    ep.set_default_client_config(client_config);
    Ok(ep)
}

/// A listening endpoint for an inbound shard peer (self-signed cert; allows client-opened
/// bi-streams so the upstream node initiates each hop).
pub fn listen_endpoint(addr: SocketAddr) -> io::Result<Endpoint> {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).map_err(io_other)?;
    let cert_der = CertificateDer::from(cert.cert);
    let key_der = PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der());

    let mut server_crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der.into())
        .map_err(io_other)?;
    server_crypto.alpn_protocols = vec![ALPN.to_vec()];

    let mut server_config = ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto).map_err(io_other)?,
    ));
    let mut transport = keepalive_transport();
    transport.max_concurrent_bidi_streams(8u32.into());
    server_config.transport_config(Arc::new(transport));

    Endpoint::server(server_config, addr).map_err(io_other)
}

/// Transport config with keepalives so a persistent shard link survives long idle gaps (e.g. a
/// multi-minute model load before the first hop) without hitting the idle timeout.
fn keepalive_transport() -> TransportConfig {
    let mut transport = TransportConfig::default();
    transport.keep_alive_interval(Some(Duration::from_secs(5)));
    transport.max_idle_timeout(Some(Duration::from_secs(30).try_into().expect("valid idle timeout")));
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
    let incoming = ep.accept().await.ok_or_else(|| io_other("endpoint closed"))?;
    let conn = incoming.await.map_err(io_other)?;
    let (send, recv) = conn.accept_bi().await.map_err(io_other)?;
    Ok(PeerConnection::new(conn, send, recv))
}

fn io_other<E: ToString>(e: E) -> io::Error {
    io::Error::other(e.to_string())
}

/// TLS verifier that accepts any certificate (dev/test peer links; prod pins certs).
#[derive(Debug)]
struct SkipServerVerification;

impl ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The node→node hop: a hidden state crosses a real QUIC link and the tail returns a result.
    // Proves the wire (serialization + framing + bi-stream) end to end, GPU-free.
    #[tokio::test]
    async fn hidden_state_roundtrips_over_quic() {
        let server = listen_endpoint("127.0.0.1:0".parse().unwrap()).unwrap();
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
            let argmax = vec![got.len() as TokenId, frame.n_tokens as TokenId, frame.start_pos as TokenId];
            peer.send(&PeerMessage::Result(VerifyResult { step: frame.step, argmax }))
            .await?;
            peer.wait_closed().await; // hold the connection open until the client has the result
            io::Result::Ok(got)
        };

        // Head: dial, send the hidden state, read the result.
        let client = dial_endpoint().unwrap();
        let client_fut = async {
            let mut peer = dial(&client, addr).await?;
            let frame = HiddenStateFrame::from_f32(42, 7, 4, 8, &hidden);
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
        assert_eq!(server_got, hidden_expect, "hidden state must survive the QUIC round-trip exactly");
        assert_eq!(result.step, 42);
        assert_eq!(result.argmax, vec![32, 4, 7], "n_floats=32, n_tokens=4, start_pos=7");
    }

    /// The payload must travel as one msgpack `bin`, not an array of integers (which costs up
    /// to 2 bytes per byte and is what a plain `Vec<u8>` serializes to).
    #[test]
    fn hidden_payload_is_compact_on_the_wire() {
        let hidden = vec![-1.5f32; 8 * 2560]; // one gamma+1 chunk at Qwen3-4B width
        let frame = PeerMessage::Hidden(HiddenStateFrame::from_f32(1, 0, 8, 2560, &hidden));
        let bytes = rmp_serde::to_vec(&frame).unwrap();
        let payload = hidden.len() * 4;
        assert!(bytes.len() < payload + 64, "{} wire bytes for a {payload}-byte payload", bytes.len());
        match rmp_serde::from_slice(&bytes).unwrap() {
            PeerMessage::Hidden(f) => assert_eq!(f.to_f32(), hidden),
            other => panic!("expected Hidden, got {other:?}"),
        }
    }

    #[test]
    fn handshake_messages_roundtrip() {
        let hello = Hello {
            proto: PROTO_VERSION,
            token: Some("s3cret".into()),
            model: "Qwen/Qwen3-4B".into(),
            chunk: 8,
            max_seq: 768,
            hidden_dim: 2560,
            plan: vec![[0, 18], [18, 36]],
            index: 1,
        };
        let bytes = rmp_serde::to_vec(&PeerMessage::Hello(hello.clone())).unwrap();
        match rmp_serde::from_slice(&bytes).unwrap() {
            PeerMessage::Hello(h) => assert_eq!(h, hello),
            other => panic!("expected Hello, got {other:?}"),
        }
    }
}

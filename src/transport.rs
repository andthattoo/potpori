//! The wire between adjacent nodes: message types, framing, mutually authenticated QUIC
//! endpoints, the session handshake, and the middle node's relay.
//!
//! One quinn bi-stream per hop carries length-prefixed msgpack ([`read_framed`] /
//! [`write_framed`]). Hidden states travel downstream, verification results travel back up,
//! and both stay in submission order. [`crate::node`] builds the head, middle and tail roles
//! on top of this module.

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use quinn::{ClientConfig, Endpoint, RecvStream, SendStream, ServerConfig, TransportConfig};
use rustls::pki_types::CertificateDer;
use serde::{Deserialize, Serialize};

use crate::engine::TokenId;

/// Bumped on every incompatible wire change, so mismatched builds fail at the TLS handshake.
const ALPN: &[u8] = b"potpori-shard-v3";

// ---------------------------------------------------------------------------
// Length-prefixed msgpack framing
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

/// A residual-stream hidden state crossing a shard boundary: logically `[n_tokens, hidden]`,
/// row-major, little-endian `dtype`. A verification chunk at Qwen3-4B width is about 80 KiB,
/// so the link is bound by round-trip time, not bandwidth.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HiddenStateFrame {
    /// Per-session frame counter, echoed in the matching [`VerifyResult`].
    pub step: u64,
    /// Absolute sequence position of the first token in this chunk. Every shard writes its KV
    /// at the same absolute positions, so a rejected speculative tail is simply overwritten.
    pub start_pos: u32,
    /// The layer boundary this hidden state sits at: the receiving shard's first layer.
    pub layer_boundary: u32,
    pub n_tokens: u32,
    pub hidden_dim: u32,
    /// Activation codec on the wire. Only "f32" is implemented.
    pub dtype: String,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

impl HiddenStateFrame {
    /// Build an f32 hidden-state frame from a flat `[n_tokens * hidden_dim]` residual stream.
    pub fn from_f32(
        step: u64,
        start_pos: u32,
        layer_boundary: u32,
        n_tokens: u32,
        hidden_dim: u32,
        hidden: &[f32],
    ) -> Self {
        Self {
            step,
            start_pos,
            layer_boundary,
            n_tokens,
            hidden_dim,
            dtype: "f32".to_string(),
            data: f32_to_le_bytes(hidden),
        }
    }

    /// Decode the payload back to a flat f32 residual stream.
    pub fn to_f32(&self) -> Vec<f32> {
        assert_eq!(self.dtype, "f32", "only the f32 codec is implemented");
        le_bytes_to_f32(&self.data)
    }

    /// Check an inbound frame against the receiving shard before it reaches the GPU: codec,
    /// width, boundary, a compiled kernel for the chunk size (`chunk` 0 = any size), cache
    /// bounds, and finite values.
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

/// The tail's answer to one [`HiddenStateFrame`]: the target's greedy token at each of its
/// positions. Middles relay it upstream to the head, which does the acceptance check.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyResult {
    pub step: u64,
    pub argmax: Vec<TokenId>,
}

impl VerifyResult {
    pub fn validate(&self, step: u64, count: usize, vocab: usize) -> io::Result<()> {
        if self.step != step
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

/// One framed message on a shard bi-stream.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PeerMessage {
    /// Upstream → downstream, first message of a session.
    Hello(SessionHello),
    /// Downstream → upstream: this node and every node below it accepted the session.
    Ready {
        final_layer: usize,
    },
    Hidden(HiddenStateFrame),
    Result(VerifyResult),
    /// Downstream → upstream: the session is over, and why. Relayed up to the head.
    Error(String),
}

/// What two adjacent nodes must agree on before any activation is exchanged.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionHello {
    /// Model geometry (layers, widths, heads), so different checkpoints cannot be mixed.
    pub model: String,
    /// The layer boundary between the two nodes.
    pub next_layer: usize,
    pub max_seq: usize,
    pub chunk: usize,
}

impl SessionHello {
    /// Why `got` (the upstream node's hello) does not fit `self` (what this node serves).
    fn mismatch(&self, got: &SessionHello) -> String {
        let mut diffs = Vec::new();
        if got.model != self.model {
            diffs.push(format!(
                "model {:?} (this node has {:?})",
                got.model, self.model
            ));
        }
        if got.next_layer != self.next_layer {
            diffs.push(format!(
                "next layer {} (this node starts at {})",
                got.next_layer, self.next_layer
            ));
        }
        if got.max_seq != self.max_seq {
            diffs.push(format!(
                "max_seq {} (this node has {})",
                got.max_seq, self.max_seq
            ));
        }
        if got.chunk != self.chunk {
            diffs.push(format!(
                "chunk {} (this node has {})",
                got.chunk, self.chunk
            ));
        }
        format!("upstream expects {}", diffs.join(", "))
    }
}

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
/// How long to wait for a peer to read a final message and close.
pub const CLOSE_TIMEOUT: Duration = Duration::from_secs(3);

/// Open a session with the downstream peer: send `hello`, then wait for it to confirm that
/// the chain below it is complete up to `final_layer`. A peer that refuses says why.
pub async fn negotiate(
    peer: &mut PeerConnection,
    hello: SessionHello,
    final_layer: usize,
) -> io::Result<()> {
    peer.send(&PeerMessage::Hello(hello)).await?;
    // A middle only answers once everything below it has answered, so allow a few hops.
    match tokio::time::timeout(HANDSHAKE_TIMEOUT * 4, peer.recv())
        .await
        .map_err(|_| {
            io_other("timed out waiting for the downstream node to accept the session")
        })?? {
        Some(PeerMessage::Ready { final_layer: got }) if got == final_layer => Ok(()),
        Some(PeerMessage::Ready { final_layer: got }) => Err(io_other(format!(
            "the chain ends at layer {got}, but the model has {final_layer}"
        ))),
        Some(PeerMessage::Error(why)) => Err(io_other(why)),
        _ => Err(io_other("downstream node did not complete the handshake")),
    }
}

/// Accept a session from the upstream peer if its hello equals `expected`. On a mismatch the
/// peer is told which settings differ (prefixed with `who`, this node's name, since the text
/// is relayed all the way to the head), and the same text is returned as the error.
pub async fn receive_hello(
    peer: &mut PeerConnection,
    expected: &SessionHello,
    who: &str,
) -> io::Result<()> {
    let why = match tokio::time::timeout(HANDSHAKE_TIMEOUT, peer.recv())
        .await
        .map_err(|_| io_other("timed out waiting for the upstream node's hello"))??
    {
        Some(PeerMessage::Hello(got)) if got == *expected => return Ok(()),
        Some(PeerMessage::Hello(got)) => expected.mismatch(&got),
        _ => "expected a session hello first".to_string(),
    };
    Err(peer.fail(format!("{who}: {why}")).await)
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

/// A duplex link to an adjacent node, carrying ordered pipelined requests and results.
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
        write_framed(&mut self.send, msg).await
    }

    /// Read the next message; `Ok(None)` on clean stream EOF.
    pub async fn recv(&mut self) -> io::Result<Option<PeerMessage>> {
        read_framed(&mut self.recv).await
    }

    /// End the session with a reason: tell the peer, give it a moment to read that, and
    /// return the same reason as this side's error.
    pub async fn fail(&mut self, why: String) -> io::Error {
        let _ = self.send(&PeerMessage::Error(why.clone())).await;
        self.finish();
        let _ = tokio::time::timeout(CLOSE_TIMEOUT, self.wait_closed()).await;
        io_other(why)
    }

    /// Close the connection now, without a goodbye.
    #[cfg(test)]
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

    /// QUIC's current round-trip estimate to this peer.
    pub fn rtt(&self) -> Duration {
        self.conn.rtt()
    }
}

/// The middle node's session loop. Work flows downstream and results flow upstream as two
/// independent halves, so this node starts on the next chunk without waiting for the answer
/// to the previous one; request/result order stays FIFO and at most 64 results are pending.
///
/// `forward` runs this node's shard on one frame. If anything fails — a bad frame, a lost
/// link, or a [`PeerMessage::Error`] from below — the reason is sent upstream and the
/// downstream connection is closed, so no node is left waiting in a dead session.
pub async fn relay_middle<F>(
    up: PeerConnection,
    down: PeerConnection,
    mut forward: F,
) -> io::Result<()>
where
    F: FnMut(HiddenStateFrame) -> io::Result<HiddenStateFrame>,
{
    let PeerConnection {
        conn: up_conn,
        send: mut up_send,
        recv: mut up_recv,
    } = up;
    let PeerConnection {
        conn: down_conn,
        send: mut down_send,
        recv: mut down_recv,
    } = down;
    let (pending_tx, mut pending_rx) = tokio::sync::mpsc::channel::<(u64, usize)>(64);
    let transmit = async {
        while let Some(msg) = read_framed::<PeerMessage>(&mut up_recv).await? {
            let PeerMessage::Hidden(frame) = msg else {
                return Err(io_other("unexpected message during shard session"));
            };
            let expected = (frame.step, frame.n_tokens as usize);
            let out = forward(frame)?;
            pending_tx.send(expected).await.map_err(io_other)?;
            write_framed(&mut down_send, &PeerMessage::Hidden(out)).await?;
        }
        down_send.finish().map_err(io_other)?;
        drop(pending_tx);
        io::Result::Ok(())
    };
    let receive = async {
        while let Some((step, n)) = pending_rx.recv().await {
            let result = match read_framed::<PeerMessage>(&mut down_recv).await? {
                Some(PeerMessage::Result(r)) if r.step == step && r.argmax.len() == n => r,
                Some(PeerMessage::Error(why)) => return Err(io_other(why)),
                _ => return Err(io_other("invalid or missing downstream result")),
            };
            write_framed(&mut up_send, &PeerMessage::Result(result)).await?;
        }
        io::Result::Ok(())
    };
    let outcome = tokio::try_join!(transmit, receive);
    match &outcome {
        Ok(_) => {
            up_send.finish().map_err(io_other)?;
            let _ = tokio::time::timeout(CLOSE_TIMEOUT, down_conn.closed()).await;
        }
        Err(e) => {
            let _ = write_framed(&mut up_send, &PeerMessage::Error(e.to_string())).await;
            let _ = up_send.finish();
            down_conn.close(0u32.into(), b"session failed");
            let _ = tokio::time::timeout(CLOSE_TIMEOUT, up_conn.closed()).await;
        }
    }
    outcome.map(|_| ())
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
    /// A fresh identity that trusts only itself.
    #[cfg(test)]
    pub fn test_identity() -> Self {
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
            step: 7,
            argmax: vec![0, 15],
        };
        valid.validate(7, 2, 16).unwrap();
        assert!(valid.validate(8, 2, 16).is_err());
        assert!(valid.validate(7, 1, 16).is_err());
        assert!(valid.validate(7, 2, 15).is_err());
        let invalid = VerifyResult {
            argmax: vec![-1, 1],
            ..valid
        };
        assert!(invalid.validate(7, 2, 16).is_err());
    }

    #[test]
    fn rejects_malformed_frames() {
        let valid = HiddenStateFrame::from_f32(1, 6, 12, 2, 4, &[0.; 8]);
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
        // Both sides must learn WHICH setting differs, not just that something does.
        let server_fut = async {
            let mut p = accept(&server).await.unwrap();
            let err = receive_hello(&mut p, &expected, "tail")
                .await
                .unwrap_err()
                .to_string();
            assert!(err.contains("max_seq 512 (this node has 768)"), "{err}");
        };
        let client_fut = async {
            let mut p = dial(&client, addr).await.unwrap();
            let err = negotiate(&mut p, wrong, 36).await.unwrap_err().to_string();
            assert!(err.contains("tail: upstream expects max_seq 512"), "{err}");
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
                step: frame.step,
                argmax,
            }))
            .await?;
            peer.wait_closed().await; // hold the connection open until the client has the result
            io::Result::Ok(got)
        };

        // Head: dial, send the hidden state, read the result.
        let client = dial_endpoint(&TlsIdentity::for_tests()).unwrap();
        let client_fut = async {
            let mut peer = dial(&client, addr).await?;
            let frame = HiddenStateFrame::from_f32(42, 7, 14, 4, 8, &hidden);
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
        assert_eq!(result.step, 42);
        assert_eq!(
            result.argmax,
            vec![32, 4, 7],
            "n_floats=32, n_tokens=4, start_pos=7"
        );
    }

    // A failure below a middle must travel up as an Error carrying the reason, and the middle
    // must close its downstream link so the node below is not left in a dead session.
    #[tokio::test]
    async fn middle_relays_a_downstream_error_and_closes_both_sides() {
        let id = TlsIdentity::for_tests();
        let tail = listen_endpoint("127.0.0.1:0".parse().unwrap(), &id).unwrap();
        let middle = listen_endpoint("127.0.0.1:0".parse().unwrap(), &id).unwrap();
        let head = dial_endpoint(&id).unwrap();
        let (middle_addr, tail_addr) = (middle.local_addr().unwrap(), tail.local_addr().unwrap());
        let tail_fut = async {
            let mut p = accept(&tail).await.unwrap();
            assert!(matches!(
                p.recv().await.unwrap(),
                Some(PeerMessage::Hidden(_))
            ));
            let _ = p.fail("tail: out of cache".into()).await;
        };
        let middle_fut = async {
            let up = accept(&middle).await.unwrap();
            let ep = dial_endpoint(&id).unwrap();
            let down = dial(&ep, tail_addr).await.unwrap();
            let err = relay_middle(up, down, Ok).await.unwrap_err().to_string();
            assert!(err.contains("tail: out of cache"), "{err}");
        };
        let head_fut = async {
            let mut p = dial(&head, middle_addr).await.unwrap();
            let frame = HiddenStateFrame::from_f32(1, 0, 1, 1, 1, &[0.]);
            p.send(&PeerMessage::Hidden(frame)).await.unwrap();
            match p.recv().await.unwrap() {
                Some(PeerMessage::Error(why)) => assert!(why.contains("tail: out of cache")),
                other => panic!("expected Error, got {other:?}"),
            }
        };
        tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(tail_fut, middle_fut, head_fut);
        })
        .await
        .unwrap();
    }

    // The payload must travel as one msgpack `bin`. A plain `Vec<u8>` serializes as an array
    // of integers, which costs up to two bytes per byte.
    #[test]
    fn hidden_payload_is_compact_on_the_wire() {
        let hidden = vec![-1.5f32; 8 * 2560]; // one verification chunk at Qwen3-4B width
        let frame = PeerMessage::Hidden(HiddenStateFrame::from_f32(1, 0, 18, 8, 2560, &hidden));
        let bytes = rmp_serde::to_vec(&frame).unwrap();
        assert!(
            bytes.len() < hidden.len() * 4 + 64,
            "{} wire bytes",
            bytes.len()
        );
        match rmp_serde::from_slice(&bytes).unwrap() {
            PeerMessage::Hidden(f) => assert_eq!(f.to_f32(), hidden),
            other => panic!("expected Hidden, got {other:?}"),
        }
    }
}

# Luminal native-runtime migration

Checked upstream `main` on 2026-10-29 and pinned every Luminal dependency to
`1cbb9173dc53482bb6d87463aea2f1cfd94a1cf1` at `luminal-ai/luminal`.
The former `jafioti/luminal` URL redirects to this repository. The old project
revision was `497b1aad9b7af7db7efeb93b46845d9f8f8919cd`.

This is an API migration, not a revision-only bump:

- Graph tensors declare dtype at construction. Parameter storage is connected with
  `Graph::input_source`; explicit casts preserve F32 arithmetic.
- Qwen uses current functional `luminal_nn` normalization, rotary, embedding and
  paged-attention operations. HF projection orientation remains `[out, in]` and
  is transposed explicitly in the graph.
- Runtime-owned search replaces `Graph::build_search_space` / `Graph::search`.
  `BucketSpec` and `SymbolBounds` describe query/context ranges.
- Each CUDA/Metal runtime declares inputs, outputs, and read/write KV aliases.
  `SharedArenaPlan` assigns shared persistent weights/KV across compiled buckets.
  The application owns device allocation, uploads, readbacks, and lifetime.
- CUDA enables Luminal's `device` feature and uses the same cudarc revision as
  upstream. CUDA and Metal adapters are derived from the current chat examples.
- Model dependencies are optional under `engine`; coordination tests compile
  without the Luminal compiler. `search_graphs` maps to an eight-candidate
  population and `ceil(search_graphs / 8)` generations, minimum one generation.

Behavioral changes accompanying the migration:

- EOS is terminal only after verification. Zero/one-token limits are respected.
- Capacity bounds include speculative lookahead; single-token steps handle the
  last cache positions without requesting unsupported partial kernel sizes.
- Mutual TLS replaces disabled certificate verification. Wire ALPN is now
  `potpori-shard-v2`; older binaries are intentionally incompatible.
- Session negotiation checks model geometry, contiguous layer boundaries, cache
  size and chunk size. Frames and result correlation are validated.
- Middles process/forward work independently of downstream result relay with a
  bounded pending queue. The head's I/O runtime continues driving QUIC during GPU
  computation. Replies still travel through the chain; no direct-return ring is
  claimed.
- Combined weights are validated and published with an atomic rename; readers
  see a completed file, including after interrupted earlier preparations.

Local validation on Windows with Rust 1.99.1, CUDA 12.9 and an RTX 5080:

- Default coordination/transport suite: 16 passed.
- `engine` suite: 17 passed; one tokenizer download test explicitly ignored.
- Strict Clippy (`-D warnings`) passed for `engine` and `cuda`, including tests.
- CUDA CLI built successfully and its help/identity commands were checked.
- The tiny CUDA integration test passed in 148.49 seconds. Full and two-shard
  execution matched an independent scalar CPU reference within `2e-4` absolute
  logit error; overwriting speculative positions matched a fresh-cache run.

Validation commands are in the README and CI workflow. The generated tiny CUDA
test exercises actual device execution without external weights.
Full Qwen parity/performance, mixed Metal/CUDA chains and real WAN throughput
require hardware-specific integration runs. Historical A6000 and Metal timings
predate this migration and must not be carried forward as new measurements.
Metal runtime execution was not available on this Windows host. The workflow adds
macOS Metal compilation and Linux CUDA compilation; those CI jobs have not run yet.

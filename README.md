# potpori

Sharded Qwen3 inference across friends' GPUs, with greedy speculative decoding over authenticated QUIC connections.

Potpori is a Rust CLI on [Luminal](https://github.com/luminal-ai/luminal). The current dependency is pinned to upstream commit [`1cbb917`](https://github.com/luminal-ai/luminal/commit/1cbb9173dc53482bb6d87463aea2f1cfd94a1cf1), the latest `main` checked on October 29, 2026. See [the migration notes](docs/luminal-migration.md) for the API changes and validation limits.

## Architecture

```text
head: tokenizer + embeddings + first layers + draft model
  → middle: next layers → tail: final layers + output projection
  ← verification results relayed through the middle ←
```

The head proposes tokens with a small Qwen3 model. Each target shard verifies a chunk, returning the target's greedy token at every position. Accepted tokens match the target; rejected tails are overwritten at absolute KV positions. Only verified EOS tokens stop generation. Near cache capacity, the driver drains speculation and uses single-token verification.

Several chunks can remain in flight. Middle nodes forward new work independently of result relay using bounded queues. This enables stage overlap; it does not eliminate network latency. Results currently return through the chain.

Supported configurations: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, and `Qwen/Qwen3-4B`. GLM/MoE, NVFP4, sampling, NAT traversal and session reconnection remain future work. Greedy equivalence also depends on the backend's numerical behavior across batch sizes; full-model parity must be validated for the intended hardware/configuration.

## Build

Use current stable Rust (Luminal requires at least Rust 1.91) and the platform's native build tools.

```sh
cargo build --locked --release --features metal  # macOS / Apple Silicon
cargo build --locked --release --features cuda   # NVIDIA
```

CUDA needs a compatible driver, CUDA toolkit headers including CCCL, NVRTC, and cuBLAS available to the dynamic loader. CUDA 12.9 was used for the local migration checks. `CUDA_PATH` / `CUDA_HOME` can identify the toolkit; `LUMINAL_CUDA_HEADERS_DIR` can point directly to its include directory. For an installation without `nvcc`, set `CUDARC_CUDA_VERSION` to the installed version (for example `12090` for 12.9).

The native graph uses F32 arithmetic and KV state on both backends, with explicit casts from BF16 checkpoint storage. `POTPORI_DTYPE` no longer selects arithmetic precision. Previous F16/BF16 performance measurements apply to the historical backend, not this migration.

## Establish peer trust

Generate a separate identity on each machine:

```sh
potpori identity --out identity
```

Keep `identity/key.pem` private. Exchange `identity/cert.pem` through a trusted channel with your adjacent peers. Collect the peer certificates in `identity/peers.pem`, for example on the middle node:

```sh
cat head-cert.pem tail-cert.pem > identity/peers.pem
```

Both ends verify certificates. There is no insecure mode. The head needs the middle's certificate; the middle needs the head's and tail's certificates; the tail needs the middle's certificate. Custom paths are accepted by `serve` and `chat` through `--cert`, `--key`, and `--trust`.

## Run a three-machine chain

Copy `chain.example.toml` to `chain.toml` on the head and edit the IP addresses. Every server's model, layer boundaries, cache capacity, and verification chunk must agree with the chain. The session handshake checks compatibility before processing activations.

Start the tail, then the middle:

```sh
potpori serve --listen 0.0.0.0:4010 --model Qwen/Qwen3-4B \
  --layers 24:36 --role tail --chunk 8 --max-seq 768

potpori serve --listen 0.0.0.0:4010 --model Qwen/Qwen3-4B \
  --layers 12:24 --role middle --downstream 100.64.0.3:4010 \
  --chunk 8 --max-seq 768
```

Then run the head:

```sh
potpori chat --chain chain.toml --max-new 128 \
  --prompt "Explain speculative decoding in one paragraph."
```

`--chunk` must equal `gamma + 1`; `--max-seq` must equal `max_seq`. Tokenized prompt length plus `--max-new` must fit the cache. Fixed kernels cover one token and the verification chunk size. Configuration errors are rejected before loading model weights.

The first run downloads and prepares weights, then searches GPU implementations for each bucket. Preparation currently downloads the whole model on each participant and combines its tensors in host memory; only the local shard's parameters are uploaded to its runtime. Allow host RAM and disk for preparation. Prepared files are validated and published by atomic replacement.

## Checks

```sh
cargo test --locked                       # GPU-free coordination and QUIC tests
cargo test --locked --features engine     # also model/loading checks; no GPU required
cargo check --locked --features cuda --all-targets
cargo clippy --locked --features cuda --all-targets -- -D warnings
cargo test --locked --features cuda native_cuda_tiny_shards_and_rollback -- --ignored --nocapture
```

The tiny GPU test creates deterministic weights locally and checks an independent scalar CPU reference, shard equivalence, and KV rollback. Tests that download real Qwen weights or need a GPU are explicitly ignored by default. QUIC tests require local UDP access. Historical GPU tests remain available under the matching backend feature; their earlier results do not establish parity after a compiler upgrade.

## Lineage and license

Inspired by [dnet](https://github.com/firstbatchxyz/dnet) and [shard](https://github.com/leyten/shard). Historical design notes remain in `docs/`; the current implementation and migration notes take precedence over earlier proposed APIs and benchmarks.

Apache-2.0. The native runtime/memory adapters are adapted from Luminal's MIT/Apache-2.0 examples at the pinned revision; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

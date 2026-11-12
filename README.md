# potpori

Sharded Qwen3 inference across friends' GPUs, with greedy speculative decoding over
authenticated QUIC connections.

Potpori is a Rust CLI on [Luminal](https://github.com/luminal-ai/luminal), pinned to upstream
commit [`1cbb917`](https://github.com/luminal-ai/luminal/commit/1cbb9173dc53482bb6d87463aea2f1cfd94a1cf1).

## How it works

```text
head: tokenizer + embeddings + first layers + draft model
  → middle: next layers → tail: final layers + output projection
  ← verification results relayed back through the middle ←
```

A model is cut into contiguous layer ranges, one per machine. The head proposes several
tokens with a small draft model; the chain checks them all in one pass and returns the target
model's own greedy token at every position. Tokens that match are kept, the first mismatch is
replaced by the target's token, and anything speculative after it is overwritten in each
shard's KV cache. A wrong draft costs speed, never output: only the target's own picks are
committed, and only a verified end-of-sequence token stops generation.

Several verification chunks stay in flight at once (`depth`), so the machines work on
different chunks at the same time and network round trips overlap with compute. Middle nodes
forward new work without waiting for results. Results currently return through the chain.

Supported models: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B`. Not built yet:
other model families, sampling (decoding is greedy), NAT traversal, and reconnecting after a
node drops. Output equals greedy decoding *of the same chain*; whether that matches a
single-process run token for token depends on the backend's numerics and has to be checked
per hardware setup.

## Build

Current stable Rust (1.91 or newer) and the platform's native build tools.

```sh
cargo build --locked --release --features metal  # macOS / Apple Silicon
cargo build --locked --release --features cuda   # NVIDIA
```

CUDA needs a compatible driver and the CUDA toolkit (headers including CCCL, NVRTC, cuBLAS)
visible to the dynamic loader; 12.9 is what has been tested. `CUDA_PATH` / `CUDA_HOME`
identify the toolkit, and `LUMINAL_CUDA_HEADERS_DIR` can point straight at its include
directory. Without `nvcc`, set `CUDARC_CUDA_VERSION` (for example `12090` for 12.9).

Both backends compute in F32. CUDA keeps the checkpoint's BF16 on the device; Metal has no BF16
buffers, so weights are converted to F16 at load.

## Establish peer trust

Each machine has its own identity:

```sh
potpori identity --out identity
```

Keep `identity/key.pem` private. Give `identity/cert.pem` to the machines next to you in the
chain, and collect theirs in `identity/peers.pem` — for example on a middle node:

```sh
cat head-cert.pem tail-cert.pem > identity/peers.pem
```

Both ends of every connection verify certificates; there is no insecure mode. The head needs
the certificate of the first remote node, each middle needs its two neighbours', and the tail
needs its upstream neighbour's. `serve` and `chat` take `--cert`, `--key` and `--trust` for
other paths.

## Run a chain

Copy `chain.example.toml` to `chain.toml` on the head and edit the addresses. Start the tail,
then the middle:

```sh
potpori serve --listen 0.0.0.0:4010 --model Qwen/Qwen3-4B \
  --layers 24:36 --role tail --chunk 8 --max-seq 768

potpori serve --listen 0.0.0.0:4010 --model Qwen/Qwen3-4B \
  --layers 12:24 --role middle --downstream 100.64.0.3:4010 \
  --chunk 8 --max-seq 768
```

Then the head:

```sh
potpori chat --chain chain.toml --max-new 128 \
  --prompt "Explain speculative decoding in one paragraph."
```

`--chunk` must equal `gamma + 1` and `--max-seq` must equal `max_seq` from the chain config.
The prompt plus `--max-new` must fit in `max_seq`. A session starts with a handshake between
each pair of neighbours; if a node was started with a different model, layer range, cache size
or chunk size, `chat` fails at connect and says which node refused and which setting differs.
A failure during generation is reported the same way.

The first run on each machine downloads the whole model, prepares the weights (streamed to a
single file in the Hugging Face cache; allow disk for a second copy) and searches GPU kernels
for the shard, which takes minutes. The head waits up to 15 minutes for a node that is still
loading.

`chat` ends with four lines of measurements:

```text
[212 tokens in 21.4s = 9.9 tok/s overall, 11.6 tok/s after prefill | 2.71 tok/traversal over 78 traversals | γ=7 chain=2 nodes]
[time: prefill 3.10s (41 prompt tokens) | head shard 4.20s | waiting on chain 9.80s | draft + rest 4.30s]
[link: round trip 38.2 ms at connect, 36.9 ms now | 85 frames, 80.0 KiB each | head on metal]
[draft model Qwen/Qwen3-0.6B, depth=4]
```

`tok/traversal` is how many tokens each trip through the chain committed (draft acceptance).
`--depth N` overrides the config for one run; comparing `--depth 1` with `--depth 4` shows
what pipelining buys over a given link.

For a first run between two machines, follow [docs/two-machine-test.md](docs/two-machine-test.md).

### A drafter that reads the target

Instead of a small draft model, the head can run a [DSpark](docs/dspark.md) drafter: a few
extra layers that read the target's own hidden states and propose a block of seven tokens in
one pass. Its guesses are better, but it cannot draft ahead, so one block is verified per
trip and `depth` does not apply. Add `drafter = "deepseek-ai/dspark_qwen3_4b_block7"` to
`chain.toml` and `--taps 1,9,17,25,33` to every `serve` command. This path has been checked
on the CPU against the reference implementation but not yet on a GPU;
[docs/dspark.md](docs/dspark.md) has the one-machine check to run first.

## Checks

```sh
cargo test --locked                       # drivers, wire and chain protocol — no GPU
cargo test --locked --features engine     # plus model and weight-preparation checks — no GPU
cargo clippy --locked --features engine --all-targets -- -D warnings
cargo test --locked --features reference cpu_check   # the real graphs on tiny models — CPU, minutes
```

The chain protocol is tested end to end over real QUIC with a synthetic shard: two- and
three-node chains against greedy decoding, handshake refusals, malformed frames, and failing
or vanishing nodes.

The `reference` feature runs the real Luminal graphs on Luminal's CPU executor with tiny
models: the shard cut and KV rollback, and the tapped layers and DSpark drafter against
outputs of transformers and DeepSpec (`tests/fixtures/dspark_tiny`).

GPU tests are `#[ignore]`d and run by name:

```sh
# CUDA: tiny generated model, no download — shard equivalence and KV rollback
cargo test --locked --release --features cuda native_cuda_tiny_shards_and_rollback -- --ignored --nocapture
# Metal: real Qwen3-0.6B chain over loopback QUIC vs a single process
cargo test --locked --release --features metal qwen_chain_equals_single_process -- --ignored --nocapture
# Either backend: Qwen3-4B with its DSpark drafter over loopback QUIC vs greedy (11 GB download)
cargo test --locked --release --features metal dspark_loopback -- --ignored --nocapture
```

The other Metal tests (`qwen_cut_2shard`, `qwen_pipelined_chain_equals_greedy`,
`qwen_pipeline_throughput`, and the single-model tests in `src/qwen/target.rs`) run the same
way. Metal execution has not been validated since the Luminal migration; see
[the migration notes](docs/luminal-migration.md).

## Layout

| Path | What it holds |
|---|---|
| `src/chain.rs` | The chain config file. |
| `src/transport.rs` | Wire messages, framing, TLS identities, the handshake, the middle's relay. |
| `src/node.rs` | The head, middle and tail roles, generic over a shard. |
| `src/specpipe.rs`, `src/specdecode.rs` | The pipelined driver and the one-chunk-at-a-time reference driver. |
| `src/qwen/` | Qwen3 on Luminal: graph, CUDA/Metal runtimes, weights, tokenizer, shard, draft. |
| `src/specfeat.rs`, `src/qwen/dspark.rs` | The one-block-per-trip driver and the DSpark drafter it drives. |
| `docs/dspark.md` | DSpark drafters for Qwen3: how they work, how potpori runs them, what has been checked. |
| `tools/compare_drafts.py` | Measures both kinds of draft on one GPU (PyTorch). |
| `docs/luminal-migration.md` | What changed with the current Luminal revision, and what has been validated. |
| `docs/history/` | Earlier design notes, kept for the reasoning. |

## Lineage and license

Inspired by [dnet](https://github.com/firstbatchxyz/dnet) and
[shard](https://github.com/leyten/shard).

Apache-2.0. The runtime and memory adapters under `src/qwen/runtime/` are adapted from
Luminal's MIT/Apache-2.0 examples at the pinned revision; see
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

# potpori

[Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) across your friends' NVIDIA
gaming GPUs, in Rust and hand-written CUDA.

The model is 66 GB in bf16 and 22 GB in its
[4-bit form](https://huggingface.co/ProCreations/Nanbeige4.3-33B-NVFP4-A16); no single gaming
card holds either. Potpori cuts its 52 layers into contiguous ranges, one per machine, and
passes the residual stream from machine to machine over authenticated QUIC connections. With
the 4-bit weights, which are the default, two 16 GB cards are enough. A
[DFlash2 drafter](https://huggingface.co/ProCreations/Nanbeige4.3-33B-DFlash2) on the first
machine proposes seven tokens at a time, and the chain checks all of them in one trip.

There is no ML framework underneath: the kernels are in [`cu/kernels.cu`](cu/kernels.cu),
`build.rs` compiles them with `nvcc`, and the Rust side talks to the CUDA driver directly. The
program runs this one model.

## How it works

```text
head: tokenizer + embedding + first layers + drafter
  → middle: next layers → … → tail: last layers + output head
  ← verification results and tapped hidden states relayed back ←
```

The drafter never sees earlier tokens. It reads the target's own hidden states — the outputs
of layers 1, 25 and 49, which travel back to the head with every result — and from them
proposes a block of seven tokens in one pass. The chain runs the last settled token plus the
seven drafts and returns the target's own greedy token at each position. Drafts that match are
kept, the first mismatch is replaced by the target's token, and anything speculative after it
is overwritten in each shard's KV cache. A wrong draft costs speed, never output: only the
target's own picks are committed.

Decoding is greedy. Not built yet: sampling, multi-turn chat, NAT traversal, and reconnecting
after a node drops.

## What has been checked

On an RTX 5080, with generated models of a few hundred channels (`cargo test … --ignored`):

- The target, with bf16 and with 4-bit weights, and the drafter match plain-Rust
  implementations written from the published architectures, to a relative error under 1e-6.
- A token gives the same bits whether it is fed alone, inside a chunk, over a rejected draft,
  cut across shards, or with its layers streamed through the GPU a group at a time.
- Drafted decoding equals plain greedy decoding, in one process and over a three-node QUIC
  chain on loopback.

With layers of the real model's size and random weights, GPU time summed over all 52 layers,
before network and the drafter:

| Weights | One token | A chunk of eight (one drafted trip) |
|---|---|---|
| 4-bit | 0.60 ms per layer, 31 ms | 1.13 ms per layer, 59 ms |
| bf16 | 1.7 ms per layer, 90 ms | 1.9 ms per layer, 97 ms |

The 4-bit layout was checked against the published files: one matrix of layer 0, decoded as
the kernel decodes it, differs from the same matrix in the bf16 repository by 14% (relative,
root mean square), the size of error 4-bit rounding gives; any other reading of the bytes is
off by more than 100%.

With the published 4-bit weights and drafter, on one RTX 5080 through `potpori local`
(greedy, no thinking block, one prompt each):

| Prompt | Tokens | Tokens committed per trip | The drafter's card, same weights |
|---|---|---|---|
| A Python function with tests | 160 | 2.96 | 2.37 (code) |
| TCP against UDP | 160 | 1.72 | 1.53 (chat) |
| Speculative decoding in a paragraph | 48 | 1.17 | 1.53 (chat) |

The text is fluent and on topic in all three. Acceptance depends heavily on the kind of text:
the drafter's card reports 2.9 for agentic coding and 1.4 to 1.5 for chat and Chinese.

**Not yet run:** a chain of real machines with the real weights, and the bf16 weights end to
end. The 4-bit model is also a slightly different model from the original: its publisher
reports longer agentic runs and a small drop on a coding set.

## Build

Current stable Rust (1.91 or newer), an NVIDIA driver, and the CUDA toolkit for `nvcc` (12.9 is
what has been tested). No CUDA libraries are linked: only the driver is used, and it is opened
at run time.

```sh
cargo build --locked --release
```

`nvcc` is found on `PATH` or under `CUDA_PATH`; set `NVCC` to point elsewhere. On Windows,
`nvcc` needs MSVC's `cl.exe`: build from a Visual Studio developer prompt or set `NVCC_CCBIN`
to its full path. The kernels are compiled to PTX for RTX 20-series cards and newer
(`POTPORI_SM=sm_86` raises the floor). `POTPORI_GPU` picks the device when a machine has
several.

## Sizing a chain

| Piece | On the GPU |
|---|---|
| One layer, 4-bit, with a 4096-token cache | 0.34 GiB |
| One layer, bf16, with a 4096-token cache | 1.13 GiB |
| Embedding table (head) | 2.2 GiB |
| Output head (tail, and the head for the drafter) | 2.2 GiB |
| Drafter (head) | 1.7 GiB |

The embedding table, the output head and the drafter are bf16 in both cases.

**4-bit (default).** Two 16 GB cards cover the model: the head takes layers 0–19 (13.0 GiB
with the drafter) and the tail layers 20–51 (13.1 GiB), as in `chain.example.toml`. A 16 GB
card has 15.9 GiB and the desktop usually holds 1 to 2.5 GiB of it, so close what you can
before starting, or lower `max_seq`. On Windows, a card that runs out of memory may spill into
system memory and become very slow instead of failing: check the free memory each node logs
after loading. On first run the head downloads about 16 GB (the drafter included) and the tail about 14 GB.

**bf16 (`--weights bf16` on every node).** A 16 GB card serves about 12 middle layers, 10 as
the tail, or 6 as the head: five such cards, or three 24 GB cards. The repository's 14 files
mix layers freely, so expect most of the 66 GB on every machine.

A chain cannot mix the two: nodes compare which weights they hold in the handshake.

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

Copy `chain.example.toml` to `chain.toml` on the head and edit the address. The remote machine
must be reachable on its UDP port from the head: a VPN such as Tailscale or WireGuard between
the machines, or a forwarded port. Start the tail:

```sh
potpori serve --listen 0.0.0.0:4010 --layers 20:52 --role tail
```

With more than two machines, start each middle after the node it points to, last to first:

```sh
potpori serve --listen 0.0.0.0:4010 --layers 20:36 --role middle --downstream 100.64.0.3:4010
```

Then the head:

```sh
potpori chat --chain chain.toml --max-new 128 --prompt "Explain speculative decoding in one paragraph."
```

`--max-seq` on every node must equal `max_seq` from the chain config, and the prompt plus
`--max-new` must fit in it. A session starts with a handshake between each pair of
neighbours; if a node was started with a different layer range, cache size or tapped layers,
`chat` fails at connect and says which node refused and which setting differs. The head waits
up to 30 minutes for a node that is still downloading.

`chat` ends with four lines of measurements: tokens per second, tokens committed per trip
through the chain (draft acceptance), where the time went, the link's round trip and payload
sizes, and the drafter's cost per block.

## One machine: `potpori local`

```sh
potpori local --max-new 32 --prompt "Write a haiku about sharing a GPU."
```

This runs the whole model on a single GPU by loading a group of layers at a time (`--group`,
16 by default in 4 bits and 4 in bf16) for every chunk, keeping only the KV caches, the
embedding, the output head and the drafter resident. It copies every layer to the GPU again
for each trip, so it is a check of the real weights and of draft acceptance, not a way to
chat. It needs the whole checkpoint (22 GB in 4 bits, 66 GB with `--weights bf16`), the 2 GB
drafter, and about 12 GiB of free GPU memory.

## Checks

```sh
cargo test --locked                # drivers, wire, chain protocol, config, selector — no GPU
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --release -- --ignored --test-threads 1 --skip tokenizer   # GPU, seconds
```

The chain protocol is tested end to end over real QUIC with a synthetic shard: two- and
three-node chains against greedy decoding, handshake refusals, malformed frames, and failing
or vanishing nodes. The GPU tests are described under "What has been checked";
`bench_real_size_layers` prints the per-layer timings (`POTPORI_BENCH_LAYERS` sets how many
layers of random weights it loads).

## Layout

| Path | What it holds |
|---|---|
| `cu/kernels.cu` | Every kernel: embedding, RMSNorm, the bf16 and 4-bit matrix products, RoPE, attention, the drafter's convolution. |
| `build.rs` | Compiles the kernels to PTX with `nvcc`. |
| `src/gpu.rs` | The CUDA context, kernel launches and device buffers. |
| `src/nanbeige/` | The model: geometry, checkpoint reader, tokenizer, the target shard, the drafter, single-GPU streaming, and the plain-Rust reference the tests compare against. |
| `src/chain.rs` | The chain config file. |
| `src/transport.rs` | Wire messages, framing, TLS identities, the handshake, the middle's relay. |
| `src/node.rs` | The head, middle and tail roles, generic over a shard. |
| `src/specfeat.rs` | The decoding driver: one drafted block per trip. |
| `src/specpipe.rs`, `src/specdecode.rs` | Drivers for a draft model that reads tokens and can run ahead of the chain. Tested, not used by this drafter. |

## Lineage and license

Inspired by [dnet](https://github.com/firstbatchxyz/dnet),
[shard](https://github.com/leyten/shard) and Baseten's
[deepseek-ultrafast](https://github.com/basetenlabs/deepseek-ultrafast). The drafter's
architecture follows its implementation in [SGLang](https://github.com/sgl-project/sglang).

Apache-2.0.

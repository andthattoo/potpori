# potpori

**Sharded LLM inference with your friends over WAN** — cut a model into layer shards, run each
shard on a different machine, and generate with **lossless speculative decoding** pipelined over
QUIC. A medley of GPUs.

Rust, on [Luminal](https://github.com/jafioti/luminal) (Metal + CUDA). One static binary — no
Python, no Docker.

## How it works

```
   head (you)                    friend A                    friend B
┌──────────────────┐        ┌───────────────┐          ┌────────────────────┐
│ tokenizer        │ hidden │               │  hidden  │                    │
│ embed + L[0,12)  │ ─────► │   L[12,24)    │ ───────► │ L[24,36) + lm_head │
│ draft model (γ)  │  QUIC  │               │   QUIC   │                    │
│ spec-decode loop │ ◄──────┴───────────────┴──────────│  argmax per pos    │
└──────────────────┘              results              └────────────────────┘
```

- **Pipeline-parallel**: contiguous layer shards; the residual-stream hidden state
  (~2-12 KiB/token) hops node→node over QUIC bi-streams. Bandwidth is trivial — WAN **RTT** is
  the enemy.
- **Speculative decoding**: a small draft model on the head proposes γ tokens; the whole chain
  verifies them in **one** traversal. Greedy token-id acceptance makes the output **bit-identical
  to plain greedy decode** of the target — provably lossless, any draft quality.
- **Depth-N pipelining**: several verify chunks stay in flight, chained on the draft's output —
  throughput approaches per-stage compute instead of full-chain latency, hiding the RTT.
- **KV rollback for free**: every shard writes its KV at absolute positions, so rejected
  speculative tails are simply overwritten — no cache surgery, no `seq_rm`.

## Status

Early. What's here today, all verified lossless (bit-identical to single-process greedy) in the
test suite:

- ✅ sync + depth-N pipelined spec-decode drivers (GPU-free unit tests)
- ✅ 2-node and 3-node (head → middle → tail) chains over real QUIC — Qwen3-0.6B, Metal
- ✅ real acceptance measured: Qwen3-0.6B drafting Qwen3-4B → 2.67 tok/traversal
- ✅ Metal multi-token kernel cliff found + worked around (chunk-specialized kernels, ~20×)
- ✅ CUDA substrate validated (A6000): builds, correct, **no** kernel cliff there
- 🚧 `serve`/`chat` CLI (this repo's M0 — wiring the above into a usable tool)
- 🔜 GLM-5.2 (744B, NVFP4) as the flagship target — the whole point; needs the big-GPU chain
- 🔜 NAT traversal, reconnection, activation codecs, sampling (temperature) story

## Quickstart (a 3-machine chain)

Everyone builds the binary (Rust + a GPU):

```bash
# Apple Silicon                      # NVIDIA (CUDA toolkit >= 12.8 required)
cargo build --release --features metal
                                     cargo build --release --features cuda
```

Copy `chain.example.toml` → `chain.toml`, put your friends' addresses in (Tailscale IPs work
great), split the layers. Friends start their shards (weights auto-download from HF):

```bash
# friend A (middle)
potpori serve --listen 0.0.0.0:4010 --model Qwen/Qwen3-4B --layers 12:24 \
              --role middle --downstream <friend-B>:4010 --chunk 8
# friend B (tail)
potpori serve --listen 0.0.0.0:4010 --model Qwen/Qwen3-4B --layers 24:36 --role tail --chunk 8
```

Then chat from the head:

```bash
potpori chat --chain chain.toml --prompt "explain speculative decoding in one paragraph"
```

First runs are slow to start: weights download once, and Luminal's kernel search compiles each
shard (~minutes). Generation itself is the fast part.

> `--chunk` must be `gamma + 1` from the chain config. On Metal it's required (the specialized
> kernel is ~20× faster than the generic multi-token one); on CUDA you can pass `--chunk 0`.

## Design notes

The `docs/` folder carries the full design + findings history: the pipeline/spec-decode design,
chain formation, the substrate decision (why Luminal), and the CUDA runbook. Notable findings
baked into the code:

- Metal has no bf16 load path at the pinned Luminal rev → weights run F16 on Metal, Bf16 on CUDA
  (`MAIN_DTYPE`).
- Luminal/Metal's generic dynamic-length forward kernel is ~20× slower than a fixed-size one →
  shards compile **point buckets** for `s ∈ {1, γ+1}` (`Shard::load_chunked`). CUDA doesn't
  need this (measured: no cliff).
- QUIC teardown must be FIN-based (`finish()` + `wait_closed()`), or the last frame gets cut.

## Lineage

Successor-in-spirit to [dnet](https://github.com/firstbatchxyz/dnet) (Python/MLX pipeline
engine for Apple Silicon), rebuilt in Rust on Luminal with WAN + speculative decoding as the
core premise. Techniques adapted from [shard](https://github.com/leyten/shard) (async depth-N
pipelining, direct-return ring) — see `docs/shard-adoption-brief.md`.

## License

Apache-2.0

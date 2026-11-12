# A first run between two machines

One generation across two machines — for example a Mac (Metal) and an NVIDIA box (CUDA) —
with numbers. Mixed-backend chains and real network links have not been measured yet, so
this is as much a test of the code as of the setup; the table at the end maps what you might
see to its cause.

## 0. Same code, and each machine works alone

Both machines must be built from the same commit: builds with different wire versions refuse
each other during the TLS handshake.

```sh
git pull
cargo build --locked --release --features metal   # Mac
cargo build --locked --release --features cuda    # NVIDIA (see the README for toolkit setup)
```

Check each backend on its own before involving the network:

```sh
# NVIDIA: tiny generated model, no download
cargo test --locked --release --features cuda native_cuda_tiny_shards_and_rollback -- --ignored --nocapture
# Mac: Qwen3-0.6B split over loopback, compared with a single process (downloads ~1.5 GB)
cargo test --locked --release --features metal qwen_chain_equals_single_process -- --ignored --nocapture
```

Both take several minutes: every shard searches its kernels at load.

## 1. Network

The head dials the tail, so only the tail has to accept an inbound connection (UDP, port 4010
below). [Tailscale](https://tailscale.com) on both machines is the easy way to get addresses
that reach each other; `tailscale ip -4` prints a machine's address.

- Tail on Windows: allow the port in an elevated PowerShell —
  `New-NetFirewallRule -DisplayName potpori -Direction Inbound -Protocol UDP -LocalPort 4010 -Action Allow`
- Tail on macOS: allow `potpori` when the system asks about incoming connections.

On the same LAN the round trip is about 1 ms, which is a fine first run but not a WAN
measurement. Putting one machine on a phone hotspot is enough to get a real one.

## 2. Identities

On each machine:

```sh
potpori identity --out identity
```

Copy each machine's `identity/cert.pem` to the other and save it there as
`identity/peers.pem`. (With more than two machines, `peers.pem` holds the certificates of the
machine's neighbours, concatenated.) `key.pem` never leaves its machine.

## 3. Start the tail

Either machine can be the tail. For Qwen3-4B split in half:

```sh
potpori serve --listen 0.0.0.0:4010 --model Qwen/Qwen3-4B \
  --layers 18:36 --role tail --chunk 8 --max-seq 768
```

Wait for `shard ready — serving [18,36) as Tail`. The first run downloads about 8 GB,
prepares the weights and searches kernels, which takes minutes.

## 4. Run the head

On the other machine, `chain.toml`:

```toml
model = "Qwen/Qwen3-4B"
draft_model = "Qwen/Qwen3-0.6B"
gamma = 7
max_seq = 768
depth = 4

[[nodes]]                  # head — this machine
layers = [0, 18]

[[nodes]]                  # tail
addr = "100.x.y.z:4010"    # the tail's address
layers = [18, 36]
```

```sh
potpori chat --chain chain.toml --max-new 200 \
  --prompt "Explain speculative decoding in one paragraph."
```

Then the same command with `--depth 1`.

## 5. What to write down

The three bracketed lines `chat` prints at the end, for both depths:

- **tok/traversal** — draft acceptance: tokens committed per trip through the chain.
- **waiting on chain**, **head shard**, **draft + rest** — where the time went.
- **round trip** — the link's latency.
- The difference between `--depth 1` and `--depth 4` — what pipelining buys over this link.

Also note which machine was the head and which backend each side ran.

## 6. The same run with a DSpark drafter

A [DSpark drafter](dspark.md) replaces the draft model. Check it on one machine first
(`dspark_on_real_weights` in the README), then:

- restart the tail with `--taps 1,9,17,25,33` added to the command in step 3;
- add `drafter = "deepseek-ai/dspark_qwen3_4b_block7"` to `chain.toml` on the head.

Run the same prompt. `--depth` has no effect here. Compare **tok/traversal** and tok/s with
the runs above; the last line also shows the drafter's time per block and how many KiB of
tapped layers come back with each result.

## When it does not work

| You see | Meaning |
|---|---|
| `could not reach <addr> within 900s` | Nothing answered: wrong address or port, a firewall, or the tail is not running. The head retries for 15 minutes because a tail may still be loading. |
| `<addr> is up but the TLS handshake failed …` | The tail answered and the handshake was refused: one side does not have the other's certificate in its `peers.pem`, or the two machines run different builds. The text in brackets says which. |
| `the chain refused the session — node [18, 36): upstream expects …` | The tail was started with settings that differ from `chain.toml`; the message lists each one (model, layer boundary, `max_seq`, chunk). |
| `the chain failed — node [18, 36): invalid hidden-state …` | The tail rejected a frame: wrong shape, a position past its cache, or non-finite values coming out of the head's shard. |
| `the chain failed — node [18, 36): the shard produced missing or non-finite logits` | The tail's own kernels produced NaN or infinity. |
| `… upstream expects tapped layers [1, 9, 17, 25, 33] (this node has [])` | The chain config names a drafter and this node was started without `--taps` (or the other way round). |
| A panic mentioning `non-finite or empty logits` on the head | The draft model or the head shard produced NaN locally. |
| Sensible text, but `tok/traversal` close to 1.0 | The draft is rarely right. If the same prompt accepts well with both models on one machine, suspect a numerical difference between the two backends. |

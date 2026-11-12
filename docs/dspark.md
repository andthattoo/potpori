# DSpark drafters for Qwen3, and what they mean for potpori

Status: implemented. Checked on the CPU against the reference implementations with tiny
models, and run on Metal with Qwen3-4B and the published checkpoint: 4.69 tokens per trip
over three prompts (see [What has been checked](#what-has-been-checked-and-what-has-not)).
Not yet run on CUDA or across two machines.

Sources: the DSpark paper (arXiv 2607.05147), the reference code
([deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec), MIT, read at commit
`005e03b`), and the checkpoint
[`deepseek-ai/dspark_qwen3_4b_block7`](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7).

## The short version

- A DSpark drafter is not a small language model. It is five extra Qwen3-4B-shaped layers
  that read the **target's own hidden states** and fill in a block of seven tokens in one
  pass.
- It is much better at guessing than a small standalone model: the paper reports 3.3–6.1
  tokens per verification on Qwen3-4B, against the 2.67 potpori measured once with
  Qwen3-0.6B.
- It costs potpori two things. The drafter needs hidden states from layers on every node, so
  **verification results must carry activations back to the head** (about 80 KiB per chunk
  for a two-node split). And it cannot draft ahead of verification, so **pipelining
  (`depth` > 1) is gone**: one block per trip through the chain.
- Whether that trade wins depends on the link and on how often the small draft gets a whole
  chunk right. Both drafts now run in potpori, so the comparison is two `potpori chat` runs
  with the same prompt; `tools/compare_drafts.py` makes the same comparison in PyTorch.

## 1. What the drafter is

### Checkpoint

`dspark_qwen3_4b_block7`: `config.json` and one `model.safetensors` (2.79 GB, BF16, about
1.39 B parameters).

| Config field | Value | Meaning |
|---|---|---|
| `architectures` | `Qwen3DSparkModel` | |
| `block_size` | 7 | tokens drafted per pass |
| `num_hidden_layers` | 5 | draft layers, all `full_attention` |
| `hidden_size`, `intermediate_size` | 2560, 9728 | same as Qwen3-4B |
| `num_attention_heads`, `num_key_value_heads`, `head_dim` | 32, 8, 128 | same as Qwen3-4B |
| `rms_norm_eps`, `rope_theta` | 1e-6, 1 000 000 | same as Qwen3-4B |
| `num_target_layers` | 36 | the target it was trained for |
| `target_layer_ids` | `[1, 9, 17, 25, 33]` | target layers whose outputs it reads |
| `mask_token_id` | 151669 | fills the block's unknown slots |
| `markov_rank`, `markov_head_type` | 256, `vanilla` | the token-to-token correction head |
| `enable_confidence_head`, `confidence_head_with_markov` | true, true | acceptance predictor |
| `num_anchors` | 512 | training only |
| `vocab_size` | 151 936 | |

Of the 1.39 B parameters, about 0.78 B are frozen copies of the target's embedding table
and output head. What is actually new is about 0.61 B: five layers (~0.5 B), the feature
projection (33 M) and the Markov head (78 M). A node that already holds Qwen3-4B's
embedding table — potpori's head — does not need the copies.

A drafter belongs to one target. Released Qwen3 drafters: 4B, 8B and 14B (block 7). They
were trained on the target's own non-thinking-mode answers, which matches potpori's chat
template. There is none for Qwen3.5, and a Qwen3 drafter will not work with it.

### Inputs from the target

For every context position the drafter needs the **output of target layers 1, 9, 17, 25 and
33** (zero-based; the residual stream after that layer, before the final norm). The five
vectors are concatenated to 12 800 numbers per position and projected:

    P = RMSNorm(fc(features))          fc: 12 800 → 2560, no bias

`P` is the drafter's whole view of the context. The same `P` feeds all five draft layers.

### One draft pass

The block is seven rows at positions `start … start+6`, where `start` is the position of the
last committed token (the "anchor"). Row 0 is the anchor's embedding; rows 1–6 are the mask
token's embedding.

Each draft layer is a Qwen3 layer with one change in attention:

- queries come from the block rows, as usual: `q = q_norm(q_proj(norm(h)))`;
- keys and values for the **block rows** come from the block: `k_proj(norm(h))`, `v_proj(norm(h))`;
- keys and values for the **context positions** `0 … start-1` come from the target features:
  `k_proj(P)`, `v_proj(P)` — `P` goes straight in, not through the layer's input norm;
- `k_norm` and RoPE (absolute positions) apply to both; the attention is **not causal**: every
  block row sees all context positions and all seven block rows.

Then the usual `o_proj`, residual, MLP, residual. After five layers: final norm, then the
target's own output head gives base logits for the seven slots. Slot `i` predicts the token
at position `start+1+i`.

The context keys and values depend only on `P`, so the drafter keeps them in a per-layer
cache and each round only adds the newly verified positions.

### Markov head

A single parallel pass cannot make slot 3 depend on what slot 2 turned out to be. The Markov
head patches that cheaply, token by token:

    logits_k = base_logits_k + W2 · W1[previous token]     W1: vocab × 256,  W2: 256 → vocab

The previous token is the anchor for slot 0 and the token just chosen for each later slot.
Greedy drafting is seven argmaxes in sequence, each after one small matrix product.

### Confidence head

`sigmoid(Linear([hidden_k ; W1[previous token]]))` predicts whether slot `k` will be
accepted. DeepSpec uses it to cut a block short. With a threshold of 0 the whole block is
proposed. For potpori it could size the chunk, which matters once each row sent back costs
bandwidth.

### The loop

1. Prefill the prompt through the target; keep the tapped features of every prompt position.
2. Draft seven tokens after the anchor.
3. Verify `[anchor, d1 … d7]` — eight tokens, exactly potpori's `gamma = 7`, `chunk = 8`.
4. Accept the matching prefix (`n` drafts) and take the target's own next token as the new
   anchor. The verify pass has produced target features for all eight positions; the first
   `n + 1` (the old anchor and the accepted drafts) are now context. Hand them to the drafter.
5. Go to 2.

Step 4 is why this drafter cannot run ahead: the features it needs for the next block only
exist once the previous block has been through the whole target.

### Published numbers

Tokens per verification (accepted drafts plus the target's one), block 7, **temperature 1**:

| Target | Method | GSM8K | MATH | AIME25 | MBPP | HumanEval | LiveCodeBench | MT-Bench | Alpaca | Arena-Hard | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | Eagle3 | 5.14 | 4.62 | 3.92 | 3.69 | 4.16 | 3.77 | 2.39 | 2.26 | 2.55 | 3.61 |
| Qwen3-4B | DFlash | 5.40 | 4.85 | 4.15 | 4.40 | 4.74 | 4.18 | 3.07 | 2.96 | 2.83 | 4.06 |
| Qwen3-4B | **DSpark** | 6.11 | 5.70 | 4.89 | 5.13 | 5.38 | 4.86 | 3.64 | 3.54 | 3.29 | 4.73 |

Means for the larger targets: 4.81 (8B) and 4.78 (14B). The paper has no greedy numbers and
no small-model baseline, so it does not tell us how DSpark compares with what potpori does
today. Math and code accept well; open conversation is around 3.5.

## 2. What it changes in potpori

### The drafter needs layers from every node

With Qwen3-4B split `[0, 18)` / `[18, 36)`, layers 1, 9 and 17 are on the head and layers 25
and 33 are on the tail. Wherever the drafter lives, the other side's taps have to travel.

The head is the right home: it runs the driver, already holds the embedding table, and its
own three taps are free. The tail's two taps come back with each verification result:

    8 positions × 2 layers × 2560 × 2 bytes = 80 KiB per chunk

Without a drafter a result is eight token ids. The forward frame is 80 KiB too (f32, one
layer), so the link becomes symmetric. On a 20 Mbit/s uplink 80 KiB is about 32 ms. Tapped
layers travel as bf16 — the precision the drafter was trained on. Their precision only
affects how good the drafts are, never the output, so coarser encodings are worth trying.
Prefill results carry taps for every prompt position, once.

(Putting the drafter on the tail instead moves the same bytes into the forward frame. No
gain.)

### No drafting ahead

The small-model draft works from tokens alone, so the head keeps `depth` chunks in flight
and the link's latency is hidden whenever a whole chunk is accepted. DSpark gets one block
per trip, every trip paid in full:

    DSpark tok/s  ≈  tokens per trip / (draft + head shard + round trip + remote shards)

Pipelining only pays when entire chunks are accepted. At the one acceptance potpori has
measured (2.67 tokens per trip, so roughly 63 % per token) a seven-token chunk survives
whole about 4 % of the time, and depth 4 is mostly wasted work. If that holds, DSpark wins
on acceptance alone. On prompts where the small model is nearly always right — boilerplate
code, arithmetic — the answer can flip, more so on slow links.

### Cheaper drafting

Today the head runs Qwen3-0.6B seven times in sequence per chunk (28 layers each). DSpark
is one pass of five 4B-sized layers over seven rows, plus seven small matrix products.

## 3. How potpori runs it

- **Taps** (`src/node.rs`). A node exposes a tapped layer by running its range as several
  shards that end there: with taps 1, 9, 17, 25, 33 the head `[0, 18)` runs `[0,2) [2,10)
  [10,18)` and the tail `[18, 36)` runs `[18,26) [26,34) [34,36)`. No change to the GPU
  adapters; three kernel searches per node instead of one, and three calls per chunk.
- **Wire** (`potpori-shard-v4`). The handshake carries the tapped layers, so a node started
  without them is refused with a message that says so. Each result carries the outputs of the
  remote nodes' tapped layers; a middle adds its own to results it relays.
- **Driver** (`src/specfeat.rs`). One block per trip: draft, verify, hand the features of the
  settled positions (the pending token and the accepted drafts) to the drafter.
- **Drafter** (`src/qwen/dspark.rs`). One Luminal graph of 15 rows at consecutive positions:
  up to 8 context rows, whose keys and values come from `P`, then the 7 block rows. Each call
  rewrites the last context positions (an overwrite, like the target's rollback) and runs the
  block behind them; a prompt's context is loaded 15 positions per call. The Markov head and
  the seven argmaxes run on the CPU. The checkpoint is used as published; the confidence head
  is not used yet.

Memory on the head for Qwen3-4B: about 2 GB for the drafter with 16-bit weights (its
layers, plus its own copy of the embedding table), twice that on Metal, which stores 32-bit
weights; and 310 MB of RAM for the Markov head.

## 4. Running it

`chain.toml` on the head names the drafter instead of a draft model:

```toml
model = "Qwen/Qwen3-4B"
drafter = "deepseek-ai/dspark_qwen3_4b_block7"
gamma = 7
```

Every remote node adds the tapped layers to its `serve` command (`potpori chat` prints the
list for the drafter it loaded):

```sh
potpori serve --listen 0.0.0.0:4010 --model Qwen/Qwen3-4B \
  --layers 18:36 --role tail --chunk 8 --max-seq 768 --taps 1,9,17,25,33
```

`potpori chat` then prints tokens per trip as before, plus the drafter's time per block and
the size of the tapped layers per result. Remove the `drafter` line (and `--taps`) to run
the same prompt with the small draft model.

**Start on one machine.** This runs Qwen3-4B, cut at the tapped layers as on a chain, and
then the published drafter on the features it produced, and prints tokens per trip for
three prompts:

```sh
cargo test --release --features metal dspark_on_real_weights -- --ignored --nocapture   # or cuda
```

The target and the drafter run one after the other, because together they do not fit on a
32 GB Mac (about 21 GB of 32-bit weights). A position's features do not depend on later
tokens, so replaying the target's greedy output gives the accepted lengths a live chain
would see. It downloads about 11 GB and runs seven kernel searches.

### What has been checked, and what has not

Checked, on the CPU with tiny random models (`cargo test --features reference`, also in CI):

- the drafter graph gives DeepSpec's logits and DeepSpec's tokens, call after call, as the
  context grows by zero to eight positions, and after going back to an earlier position;
- the target cut at the tapped layers gives transformers' logits, and the tapped outputs
  equal transformers' `hidden_states` for the configured layers;
- the whole loop reproduces the target's greedy output;
- the protocol, over QUIC with synthetic shards: features reach the head in layer order
  through two- and three-node chains, and mismatched taps are refused at connect.

Measured, on Metal (M-series Mac, 32 GB) with Qwen3-4B and `dspark_qwen3_4b_block7`
(`dspark_on_real_weights`, greedy, 96 new tokens per prompt):

| Prompt | Tokens per trip |
|---|---|
| "Explain speculative decoding in one paragraph." | 3.21 |
| "Write a Python function that checks whether a string is a palindrome." | 6.86 |
| "A train leaves at 9:40 and arrives at 13:05. How long is the trip? Show the steps." | 5.35 |
| all three | 4.69 |

Acceptance by draft position 1 to 7: 0.93, 0.73, 0.56, 0.46, 0.42, 0.32, 0.27. The paper
reports a mean of 4.73 for this drafter (at temperature 1) and 0.93 at the first position on
math, so the port reads the target the way the reference does. The drafter took 67 ms per
block in its graph and 11 ms in the Markov head; a chunk of eight took about 110 ms in the
head's three shards and 120 ms in the tail's.

Not checked: CUDA (the drafter graph has never been through its kernel search), and a live
chain with the drafter between two machines.

### Which is better

Two `potpori chat` runs with the same prompt, with and without the `drafter` line, give the
real answer for your link. `tools/compare_drafts.py` asks the same question in PyTorch on
one GPU: it replays both drivers on real model outputs, greedily, and estimates tok/s for a
list of round-trip times (`--selftest` runs it on tiny random models).

| Result | Reading |
|---|---|
| DSpark's tokens per trip well above the small model's, small model's full chunks under ~20 % | DSpark. Pipelining was not buying much. |
| Small model's full chunks above ~50 % on the prompts you care about | Keep the small model for those; pipelining is doing real work. |
| Close | Link latency decides. |

Qwen3-8B and 14B have released drafters and the same architecture; they need config entries
and, for a split with more than two nodes, nothing else.

## 5. Later: getting pipelining back

The head computes its own taps (layers 1, 9, 17) for speculative tokens the moment it runs
its shard. Only the tail's taps force the wait. A drafter trained on head-side layers alone
could draft ahead exactly as the small model does today, with DSpark-style acceptance and no
activations on the return path. No such checkpoint exists; DeepSpec ships the training code,
and the paper shows a 2-layer DSpark already beating a 5-layer DFlash, so a weaker-featured
drafter may still be good. This is a training project, worth considering once the released
drafter has been measured.

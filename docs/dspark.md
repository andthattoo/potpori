# DSpark drafters for Qwen3, and what they mean for potpori

Status: design note. Nothing here is implemented in the Rust code yet. The one piece that
exists is `tools/compare_drafts.py`, which measures the question this note cannot answer
from reading alone: which draft is better over a real link.

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
  **verification results must carry activations back to the head** (about 160 KiB per chunk
  for a two-node split). And it cannot draft ahead of verification, so **pipelining
  (`depth` > 1) is gone**: one block per trip through the chain.
- Whether that trade wins depends on the link and on how often the small draft gets a whole
  chunk right. `tools/compare_drafts.py` measures both drafts on the same prompts and prices
  them with the same clock. Run it before porting anything.

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

    8 positions × 2 layers × 2560 × 4 bytes = 160 KiB per chunk (80 KiB as 16-bit floats)

Today a result is eight token ids. The forward frame is already 80 KiB, so the link becomes
roughly symmetric. On a 20 Mbit/s uplink 160 KiB is about 65 ms; the precision of the
features only affects how good the drafts are, never the output, so 16-bit is the obvious
default and coarser encodings are worth trying. Prefill results carry taps for every prompt
position, once.

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

## 3. Which is better: measure first

`tools/compare_drafts.py` runs the target, Qwen3-0.6B and the DSpark drafter in PyTorch on
one GPU and replays potpori's driver logic on their real outputs, greedily:

- small model at depth 1 and depth 4 (and depth 4 without the drain-before-resubmit wait
  that the current driver has);
- DSpark, one block per trip.

It checks every output against plain greedy decoding, reports tokens per trip, how often
whole chunks are accepted, acceptance by draft position, how well the confidence head
predicts acceptance, and an estimated tok/s for a list of round-trip times.

On the 5080 (about 11 GB of VRAM for all three models):

```sh
git clone https://github.com/deepseek-ai/DeepSpec
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.10.2 safetensors sentencepiece numpy tqdm PyYAML prettytable tensorboard
python tools/compare_drafts.py --deepspec DeepSpec --json drafts.json
```

Defaults: 10 prompts each from MT-Bench, HumanEval and GSM8K, 200 new tokens, round trips of
1, 20, 50, 100 and 200 ms. `--selftest` runs the whole script on tiny random models on a CPU.

The tok/s estimate uses PyTorch timings from the machine it runs on, with the target's chunk
time split evenly between head and remote. Once `potpori chat` has been run across the two
machines, pass its real times (`--head-ms`, `--remote-ms`, `--small-draft-ms`). The estimate
does not include the time to send tap features back; add it to `--remote-ms` for a link of
known bandwidth.

What would decide it:

| Result | Reading |
|---|---|
| DSpark's tokens per trip well above the small model's, small model's full chunks under ~20 % | Port DSpark. Pipelining was not buying much. |
| Small model's full chunks above ~50 % on the prompts you care about | Keep the small model for those; pipelining is doing real work. |
| Close | Link latency decides; read the tok/s table at your round trip. |

## 4. The port, if the numbers say yes

In order. Steps 1–3 need no GPU; 4–5 need iteration on real hardware.

1. **Qwen3-4B as an ordinary potpori target.** It should already work (`config_for`); it has
   not been run. The two-machine runbook uses it.
2. **Wire format** (`potpori-shard-v4`). `SessionHello` gains the list of tapped layers so
   each node knows which of its own to report. `VerifyResult` gains an optional feature
   block (layer ids, rows, dtype, bytes). A middle adds its own taps to results it relays.
   `HiddenStateFrame` is unchanged.
3. **Driver.** A feature-aware draft trait (`observe(features)`, `draft(anchor, start)`) and
   a one-block-per-trip loop: this is `specdecode.rs` plus features, and
   `run_dspark` in the comparison script is the reference. `PipelinedChain::poll` returns
   tokens and features. All of it can be tested with the existing fake shards.
4. **Taps from the shards.** Two ways:
   - *Split at the taps.* A node runs several shards in a row — head `[0,2) [2,10) [10,18)`,
     tail `[18,26) [26,34) [34,36)` — and each shard's output is a tap. No change to the
     Luminal adapters; more graphs to search and more calls per chunk.
   - *Multi-output graphs.* The graph returns the tapped layers alongside its normal output.
     Faster, but the CUDA and Metal adapters read one output today.

   Start with the first.
5. **The drafter as a Luminal graph.** Everything it needs exists in `luminal_nn`
   (`paged_attention` takes an explicit bias, so non-causal attention is a zero bias). One
   fixed-shape graph of 15 rows: 8 context rows whose keys and values come from `P`, 7 block
   rows. Each call rewrites the last 8 context positions (an overwrite, like the target's
   rollback) and reads the block. The prompt's context is written 8 rows at a time. The
   Markov head and the argmaxes run on the host. Load only the new weights (~1.2 GB BF16);
   the output head is the head node's embedding table.
6. **Check.** Same prompts, potpori against the Python script: identical drafts, identical
   tokens per trip.

Then Qwen3-8B and 14B: the same architecture and released drafters, so mostly config.

## 5. Later: getting pipelining back

The head computes its own taps (layers 1, 9, 17) for speculative tokens the moment it runs
its shard. Only the tail's taps force the wait. A drafter trained on head-side layers alone
could draft ahead exactly as the small model does today, with DSpark-style acceptance and no
activations on the return path. No such checkpoint exists; DeepSpec ships the training code,
and the paper shows a 2-layer DSpark already beating a 5-layer DFlash, so a weaker-featured
drafter may still be good. This is a training project, worth considering once the released
drafter has been measured.

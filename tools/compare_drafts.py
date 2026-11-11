#!/usr/bin/env python3
"""Which draft is better for potpori: a small standalone model, or a DSpark drafter?

Runs both against the same target, on the same prompts, greedily (temperature 0), and
reports for each:

  * tok/traversal — tokens committed per trip through the target (what `potpori chat` prints),
  * how often a whole chunk is accepted (what pipelining depends on),
  * an estimated tok/s over a range of link round-trip times.

The two drafts differ in a way acceptance alone does not capture:

  * The small model drafts from tokens only, so the head can draft several chunks ahead and
    keep `depth` of them in flight. That pays off only when whole chunks are accepted.
  * DSpark needs the target's hidden states for every token before the block it drafts, so
    it cannot draft ahead: one block per trip through the chain.

The script therefore replays potpori's own driver logic (src/specpipe.rs, src/specdecode.rs)
on real model outputs, records the order of submits and polls, and prices that order with a
simple clock: the head drafts and runs its shard (serially), a frame takes half a round trip
to reach the remote nodes, they process frames one at a time, and the answer takes the other
half back. It is an estimate — it ignores prefill and per-stage overlap inside a multi-node
remote — but both drafts are priced by the same clock.

Needs a GPU with room for the target plus both drafts (about 11 GB for Qwen3-4B), PyTorch,
transformers, and a checkout of https://github.com/deepseek-ai/DeepSpec for the DSpark
model code and the benchmark prompts:

    git clone https://github.com/deepseek-ai/DeepSpec
    pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
    pip install transformers==5.10.2 safetensors sentencepiece numpy tqdm PyYAML prettytable tensorboard
    python tools/compare_drafts.py --deepspec DeepSpec

(DeepSpec's own requirements.txt also pins triton, which has no Windows build; the list
above is what this script needs.)

`--selftest` runs the whole thing on tiny random models (CPU, no downloads) to check the
script itself.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


class Target:
    """The target model with potpori's KV semantics: a chunk is written at absolute
    positions, and running a chunk at an earlier position overwrites what came after."""

    def __init__(self, model, tap_layers):
        from transformers import DynamicCache

        self.model = model
        self.tap_layers = list(tap_layers)
        self.cache = DynamicCache()
        self.seconds = []

    def reset(self):
        from transformers import DynamicCache

        self.cache = DynamicCache()

    @torch.inference_mode()
    def feed(self, start: int, tokens: list[int]):
        """Run `tokens` at positions [start, start+len). Returns the greedy token after each
        position, and the tapped-layer hidden states for each position ([len, taps*hidden])."""
        from deepspec.modeling.dspark.common import extract_context_feature

        assert self.cache.get_seq_length() >= start, "chunk would leave a KV gap"
        self.cache.crop(start)
        device = self.model.device
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        pos = torch.arange(start, start + len(tokens), device=device).unsqueeze(0)
        t0 = time.perf_counter()
        out = self.model(
            input_ids=ids,
            position_ids=pos,
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=True,
        )
        argmax = out.logits[0].argmax(dim=-1).tolist()
        _sync(device)
        self.seconds.append(time.perf_counter() - t0)
        feats = extract_context_feature(out.hidden_states, self.tap_layers)[0]
        return argmax, feats


class SmallDraft:
    """A standalone small model kept in step with the generation — a port of potpori's
    IncrementalDraft (src/qwen/draft.rs)."""

    def __init__(self, model):
        from transformers import DynamicCache

        self.model = model
        self.cache = DynamicCache()
        self.fed: list[int] = []
        self.seconds = []

    def reset(self):
        from transformers import DynamicCache

        self.cache = DynamicCache()
        self.fed = []

    @torch.inference_mode()
    def _run(self, start: int, tokens: list[int]) -> int:
        device = self.model.device
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        pos = torch.arange(start, start + len(tokens), device=device).unsqueeze(0)
        out = self.model(input_ids=ids, position_ids=pos, past_key_values=self.cache, use_cache=True)
        return int(out.logits[0, -1].argmax().item())

    def draft(self, context: list[int], gamma: int) -> list[int]:
        t0 = time.perf_counter()
        n = len(context)
        agree = 0
        for a, b in zip(self.fed, context):
            if a != b:
                break
            agree += 1
        start = min(agree, n - 1)
        self.cache.crop(start)
        cur = self._run(start, context[start:n])
        self.fed = self.fed[:start] + context[start:n]
        out = [cur]
        for pos in range(n, n + gamma - 1):
            self.fed.append(cur)
            cur = self._run(pos, [cur])
            out.append(cur)
        _sync(self.model.device)
        self.seconds.append(time.perf_counter() - t0)
        return out


class DSparkDraft:
    """DeepSpec's DSpark drafter, driven the way its own evaluator drives it."""

    def __init__(self, model):
        from transformers import DynamicCache

        self.model = model
        self.block = int(model.block_size)
        self.cache = DynamicCache()
        self.pending = None  # target features for context positions not yet in the cache
        self.seconds = []
        self.confidence = []  # (predicted acceptance probability, accepted?) per position

    def reset(self):
        from transformers import DynamicCache

        self.cache = DynamicCache()
        self.pending = None

    def observe(self, feats: torch.Tensor):
        """Target features for newly verified context positions, in order."""
        self.pending = feats if self.pending is None else torch.cat([self.pending, feats], dim=0)

    @torch.inference_mode()
    def draft(self, anchor: int, start: int):
        """Draft `block` tokens after `anchor`, which sits at absolute position `start`.
        Needs target features for every position before `start`."""
        from deepspec.eval.dspark.draft_ops import build_dspark_proposal, forward_dspark_draft_block

        assert self.pending is not None, "no target features for the context yet"
        assert self.cache.get_seq_length() + self.pending.shape[0] == start, "context features out of step"
        t0 = time.perf_counter()
        device = self.model.device
        ids = torch.full((1, self.block), int(self.model.mask_token_id), dtype=torch.long, device=device)
        ids[0, 0] = anchor
        position_ids = torch.arange(start + self.block, device=device).unsqueeze(0)
        hidden = forward_dspark_draft_block(
            self.model,
            draft_input_ids=ids,
            position_ids=position_ids,
            past_key_values_draft=self.cache,
            target_hidden_states=self.pending.unsqueeze(0).to(self.model.dtype),
            start=start,
            block_size=self.block,
        )
        proposal = build_dspark_proposal(
            self.model,
            draft_input_ids=ids,
            block_hidden=hidden,
            block_size=self.block,
            temperature=0.0,
            confidence_threshold=0.0,
        )
        self.pending = None
        _sync(device)
        self.seconds.append(time.perf_counter() - t0)
        drafts = proposal.verify_input_ids[0, 1:].tolist()
        conf = None
        if proposal.confidence_logits is not None:
            conf = proposal.confidence_logits[0].float().sigmoid().tolist()
        return drafts, conf


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


# --------------------------------------------------------------------------------------
# Drivers (ports of src/specpipe.rs and src/specdecode.rs) and the clock that prices them
# --------------------------------------------------------------------------------------


@dataclass
class Trace:
    """What a driver did, in order: ("submit", drafted?) and ("poll",)."""

    events: list = field(default_factory=list)
    accepted: list = field(default_factory=list)  # accepted drafts per verified chunk
    chunk_sizes: list = field(default_factory=list)  # drafts offered per verified chunk
    out: list = field(default_factory=list)

    @property
    def traversals(self) -> int:
        return sum(1 for e in self.events if e[0] == "poll")


def estimate_seconds(trace: Trace, *, head: float, remote: float, rtt: float, draft: float) -> float:
    """Price a trace. The head drafts then runs its shard, serially; a frame reaches the
    remote nodes after rtt/2; they take `remote` per frame, one at a time; the answer takes
    rtt/2 back."""
    now = 0.0
    remote_free = 0.0
    results = deque()
    for event in trace.events:
        if event[0] == "submit":
            now += (draft if event[1] else 0.0) + head
            begin = max(now + rtt / 2, remote_free)
            remote_free = begin + remote
            results.append(remote_free + rtt / 2)
        else:
            now = max(now, results.popleft())
    return now


def run_pipelined(draft, target, prompt, *, gamma, max_new, depth, eos, drain_before_resubmit=True) -> Trace:
    """potpori's pipelined driver. `drain_before_resubmit` mirrors the current code, which
    waits for every stale chunk to return after a wrong draft before submitting again."""
    trace = Trace()
    target.reset()
    draft.reset()
    cur0 = target.feed(0, prompt)[0][-1]
    out = []
    done = cur0 in eos
    if not done:
        out.append(cur0)
    done = done or len(out) >= max_new
    dprefix = list(prompt) + [cur0]
    send_pos = len(prompt)
    inflight = deque()
    discard = 0
    while True:
        while len(inflight) < depth and not done and (discard == 0 or not drain_before_resubmit):
            cur = dprefix[-1]
            if cur in eos:
                break
            ds = draft.draft(dprefix, gamma)
            result = target.feed(send_pos, [cur] + ds)[0]
            trace.events.append(("submit", True))
            dprefix.extend(ds)
            inflight.append((ds, result))
            send_pos += gamma
        if not inflight:
            break
        ds, r = inflight.popleft()
        trace.events.append(("poll",))
        if discard > 0:
            discard -= 1
            continue
        n = 0
        while n < len(ds) and ds[n] == r[n]:
            n += 1
        trace.accepted.append(n)
        trace.chunk_sizes.append(len(ds))
        diverged = n != len(ds)
        committed = ds[:n] + [r[n]] if diverged else ds
        for t in committed:
            if t in eos:
                done = True
                break
            out.append(t)
            if len(out) >= max_new:
                done = True
                break
        if done:
            discard = len(inflight)
        elif diverged:
            discard = len(inflight)
            keep = len(prompt) + len(out) - 1
            dprefix = list(prompt) + out
            send_pos = keep
    trace.out = out
    return trace


def run_dspark(draft: DSparkDraft, target, prompt, *, max_new, eos) -> Trace:
    """One block per trip: draft from the target's features, verify, feed the features of
    the accepted positions back to the drafter (potpori's synchronous driver)."""
    trace = Trace()
    target.reset()
    draft.reset()
    argmax, feats = target.feed(0, prompt)
    draft.observe(feats)
    cur = argmax[-1]
    base = len(prompt)
    out = []
    while cur not in eos and len(out) < max_new:
        ds, conf = draft.draft(cur, base)
        r, feats = target.feed(base, [cur] + ds)
        trace.events.append(("submit", True))
        trace.events.append(("poll",))
        n = 0
        while n < len(ds) and ds[n] == r[n]:
            n += 1
        trace.accepted.append(n)
        trace.chunk_sizes.append(len(ds))
        if conf is not None:
            draft.confidence.extend((p, i < n) for i, p in enumerate(conf))
        stop = False
        for t in [cur] + ds[:n]:
            if t in eos:
                stop = True
                break
            out.append(t)
            if len(out) >= max_new:
                stop = True
                break
        if stop:
            break
        # Positions base..base+n now hold verified tokens: the anchor and the accepted drafts.
        draft.observe(feats[: n + 1])
        cur = r[n]
        base += n + 1
    trace.out = out
    return trace


def run_greedy(target, prompt, *, max_new, eos) -> list[int]:
    """Plain greedy decode, one token per trip — the reference output."""
    target.reset()
    cur = target.feed(0, prompt)[0][-1]
    out = []
    pos = len(prompt)
    while cur not in eos and len(out) < max_new:
        out.append(cur)
        cur = target.feed(pos, [cur])[0][0]
        pos += 1
    return out


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


@dataclass
class Tally:
    name: str
    tokens: int = 0
    traversals: int = 0
    accepted: list = field(default_factory=list)
    sizes: list = field(default_factory=list)
    traces: list = field(default_factory=list)
    mismatches: int = 0

    def add(self, trace: Trace, reference: list[int]):
        self.tokens += len(trace.out)
        self.traversals += trace.traversals
        self.accepted += trace.accepted
        self.sizes += trace.chunk_sizes
        self.traces.append(trace)
        self.mismatches += int(trace.out != reference)

    def tok_per_traversal(self) -> float:
        return self.tokens / max(self.traversals, 1)

    def full_accept_rate(self) -> float:
        full = sum(1 for n, s in zip(self.accepted, self.sizes) if n == s)
        return full / max(len(self.accepted), 1)

    def accept_by_position(self, gamma: int) -> list[float]:
        """P(position i accepted | the chunk was verified), i = 0..gamma-1."""
        total = max(len(self.accepted), 1)
        return [sum(1 for n in self.accepted if n > i) / total for i in range(gamma)]

    def tok_per_second(self, **timing) -> float:
        seconds = sum(estimate_seconds(t, **timing) for t in self.traces)
        return self.tokens / max(seconds, 1e-9)


def median_ms(samples) -> float:
    return 1000 * statistics.median(samples) if samples else float("nan")


def report(tallies, *, gamma, timings, rtts_ms, dspark: DSparkDraft | None):
    print("\n== Acceptance ==")
    print(f"{'draft':34} {'tokens':>7} {'trips':>6} {'tok/trip':>9} {'full chunks':>12} {'!= greedy':>10}")
    for t in tallies:
        print(
            f"{t.name:34} {t.tokens:7d} {t.traversals:6d} {t.tok_per_traversal():9.2f} "
            f"{100 * t.full_accept_rate():11.1f}% {t.mismatches:10d}"
        )
    print("\nacceptance by draft position (share of verified chunks where it was accepted):")
    for t in tallies:
        print(f"  {t.name:32} " + " ".join(f"{p:.2f}" for p in t.accept_by_position(gamma)))
    if dspark is not None and dspark.confidence:
        print("\nDSpark confidence head (predicted acceptance probability vs what happened):")
        for lo in (0.0, 0.25, 0.5, 0.75, 0.9):
            hi = {0.0: 0.25, 0.25: 0.5, 0.5: 0.75, 0.75: 0.9, 0.9: 1.01}[lo]
            bucket = [ok for p, ok in dspark.confidence if lo <= p < hi]
            if bucket:
                print(f"  predicted {lo:.2f}-{min(hi, 1.0):.2f}: accepted {100 * sum(bucket) / len(bucket):5.1f}%  (n={len(bucket)})")

    print("\n== Estimated tok/s by link round trip ==")
    print(
        "per chunk: head shard {head_ms:.0f} ms, remote shards {remote_ms:.0f} ms, "
        "small draft {small_ms:.0f} ms, DSpark draft {dspark_ms:.0f} ms".format(**timings)
    )
    print(f"{'draft':34} " + " ".join(f"{int(r):>5}ms" for r in rtts_ms))
    for t in tallies:
        draft_ms = timings["dspark_ms"] if t.name.startswith("dspark") else timings["small_ms"]
        row = [
            t.tok_per_second(
                head=timings["head_ms"] / 1000,
                remote=timings["remote_ms"] / 1000,
                rtt=r / 1000,
                draft=draft_ms / 1000,
            )
            for r in rtts_ms
        ]
        print(f"{t.name:34} " + " ".join(f"{v:7.1f}" for v in row))
    print(
        "\nThe estimate prices each driver's real sequence of submits and polls with one clock.\n"
        "Replace the per-chunk times with numbers from `potpori chat` (--head-ms, --remote-ms,\n"
        "--small-draft-ms, --dspark-draft-ms) once you have them; the defaults are this\n"
        "machine's PyTorch timings, with the target's chunk time split evenly between head and\n"
        "remote."
    )


# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------


def load_prompts(args, tokenizer) -> list[list[int]]:
    from deepspec.data.parser import encode_chat_messages

    prompts = []
    for name in args.datasets.split(","):
        path = Path(args.deepspec) / "eval_datasets" / f"{name}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows[: args.samples]:
            ids = encode_chat_messages(
                tokenizer,
                [{"role": "user", "content": row["turns"][0]}],
                add_generation_prompt=True,
                enable_thinking=False,  # potpori's chat template also disables thinking
            )
            prompts.append(ids[0].tolist())
    return prompts


def load_real(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from deepspec.eval.base_evaluator import resolve_stop_token_ids
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel

    device = torch.device(args.device)
    kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    target_model = AutoModelForCausalLM.from_pretrained(args.target, **kwargs).to(device).eval()
    small_model = AutoModelForCausalLM.from_pretrained(args.small, **kwargs).to(device).eval()
    dspark_model = Qwen3DSparkModel.from_pretrained(args.dspark, **kwargs).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    eos = set(resolve_stop_token_ids(target_model, tokenizer) or [])
    eos |= {151645, 151643}  # <|im_end|>, <|endoftext|> — what potpori stops on
    return target_model, small_model, dspark_model, load_prompts(args, tokenizer), eos


def load_selftest(args):
    """Tiny random models with the real architectures: checks the script, not the drafts."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel, build_draft_config

    torch.manual_seed(0)
    base = dict(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
    )
    target_model = Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=6, **base)).eval()
    small_model = Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=2, **base)).eval()
    class Args(dict):  # build_draft_config wants both `"key" in args` and `args.key`
        __getattr__ = dict.__getitem__

    draft_config = build_draft_config(
        target_model.config,
        Args(
            num_draft_layers=2,
            target_layer_ids=[0, 2, 4],
            block_size=args.gamma,
            mask_token_id=96,
            num_anchors=8,
            confidence_head_alpha=1.0,
            confidence_head_with_markov=True,
            markov_rank=8,
            markov_head_type="vanilla",
        ),
    )
    draft_config._attn_implementation = "sdpa"
    dspark_model = Qwen3DSparkModel(draft_config).eval()
    dspark_model.initialize_embeddings_and_head(
        embed_tokens=target_model.model.embed_tokens, lm_head=target_model.lm_head
    )
    generator = torch.Generator().manual_seed(1)
    prompts = [torch.randint(0, 90, (n,), generator=generator).tolist() for n in (5, 9, 14)]
    return target_model, small_model, dspark_model, prompts, {95}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deepspec", required=True, help="path to a DeepSpec checkout")
    parser.add_argument("--target", default="Qwen/Qwen3-4B")
    parser.add_argument("--small", default="Qwen/Qwen3-0.6B", help="standalone draft model")
    parser.add_argument("--dspark", default="deepseek-ai/dspark_qwen3_4b_block7")
    parser.add_argument("--datasets", default="mt-bench,humaneval,gsm8k", help="names under DeepSpec/eval_datasets")
    parser.add_argument("--samples", type=int, default=10, help="prompts per dataset")
    parser.add_argument("--max-new", type=int, default=200)
    parser.add_argument("--gamma", type=int, default=7, help="drafts per chunk; must equal the DSpark block size")
    parser.add_argument("--depth", type=int, default=4, help="chunks in flight for the small draft")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtt-ms", default="1,20,50,100,200", help="link round trips to estimate tok/s for")
    parser.add_argument("--head-ms", type=float, help="head shard time per chunk")
    parser.add_argument("--remote-ms", type=float, help="all remote shards' time per chunk")
    parser.add_argument("--small-draft-ms", type=float)
    parser.add_argument("--dspark-draft-ms", type=float)
    parser.add_argument("--selftest", action="store_true", help="tiny random models, CPU, no downloads")
    parser.add_argument("--json", help="also write the raw numbers here")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.deepspec).resolve()))

    if args.selftest:
        target_model, small_model, dspark_model, prompts, eos = load_selftest(args)
        args.max_new = min(args.max_new, 40)
    else:
        target_model, small_model, dspark_model, prompts, eos = load_real(args)
    assert int(dspark_model.block_size) == args.gamma, "--gamma must equal the DSpark block size"

    target = Target(target_model, dspark_model.target_layer_ids)
    small = SmallDraft(small_model)
    dspark = DSparkDraft(dspark_model)
    names = [
        "small model, depth 1",
        f"small model, depth {args.depth}",
        f"small model, depth {args.depth}, no drain",
        "dspark, one block per trip",
    ]
    tallies = [Tally(name) for name in names]
    for i, prompt in enumerate(prompts):
        reference = run_greedy(target, prompt, max_new=args.max_new, eos=eos)
        common = dict(gamma=args.gamma, max_new=args.max_new, eos=eos)
        tallies[0].add(run_pipelined(small, target, prompt, depth=1, **common), reference)
        tallies[1].add(run_pipelined(small, target, prompt, depth=args.depth, **common), reference)
        tallies[2].add(
            run_pipelined(small, target, prompt, depth=args.depth, drain_before_resubmit=False, **common),
            reference,
        )
        tallies[3].add(run_dspark(dspark, target, prompt, max_new=args.max_new, eos=eos), reference)
        print(f"prompt {i + 1}/{len(prompts)}: {len(reference)} tokens", flush=True)

    # Chunk-sized target passes only (single-token passes belong to the greedy reference).
    chunk_ms = median_ms(target.seconds[-200:])
    timings = dict(
        head_ms=args.head_ms if args.head_ms is not None else chunk_ms / 2,
        remote_ms=args.remote_ms if args.remote_ms is not None else chunk_ms / 2,
        small_ms=args.small_draft_ms if args.small_draft_ms is not None else median_ms(small.seconds),
        dspark_ms=args.dspark_draft_ms if args.dspark_draft_ms is not None else median_ms(dspark.seconds),
    )
    rtts = [float(x) for x in args.rtt_ms.split(",")]
    report(tallies, gamma=args.gamma, timings=timings, rtts_ms=rtts, dspark=dspark)
    if args.json:
        payload = {
            "args": vars(args),
            "timings_ms": timings,
            "drafts": [
                {
                    "name": t.name,
                    "tokens": t.tokens,
                    "traversals": t.traversals,
                    "tok_per_traversal": t.tok_per_traversal(),
                    "full_accept_rate": t.full_accept_rate(),
                    "accept_by_position": t.accept_by_position(args.gamma),
                    "outputs_differing_from_greedy": t.mismatches,
                }
                for t in tallies
            ],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

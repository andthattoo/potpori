#!/usr/bin/env python3
"""Generate tests/fixtures/dspark_tiny: a tiny random Qwen3 target, a tiny random DSpark
drafter for it, and what the reference implementations (transformers, DeepSpec) compute with
them. The Rust tests run the same weights through potpori's graphs on the CPU and compare.

    python tools/make_dspark_fixture.py --deepspec path/to/DeepSpec

The weights are random, so the drafts are nonsense; the point is that potpori reproduces
them. Only needed again if the fixture's shape has to change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

TARGET = dict(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=4,
    max_position_embeddings=128,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    tie_word_embeddings=True,
)
TAPS = [0, 2, 4]
BLOCK = 7
MASK = 63
PROMPT = [5, 9, 2, 33, 17, 8, 41, 3, 12, 60, 7]
GENERATED = 48
# Accepted drafts per round, scripted: the drafter must cope with every count 0..BLOCK.
ACCEPTED = [0, 3, 7, 1, 0, 5, 2, 6, 4]


def randomize(model, generator):
    """Random weights that are exact in bf16, so every implementation starts equal."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1 + 0.3 * torch.randn(p.shape, generator=generator))
            else:
                p.copy_(0.2 * torch.randn(p.shape, generator=generator))
            p.copy_(p.to(torch.bfloat16).to(torch.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepspec", required=True)
    parser.add_argument("--seed", type=int, default=2, help="one whose drafts are varied and far from ties")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "tests/fixtures/dspark_tiny"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.deepspec).resolve()))

    from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

    from deepspec.eval.dspark.draft_ops import build_dspark_proposal, forward_dspark_draft_block
    from deepspec.modeling.dspark.common import extract_context_feature
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel, build_draft_config

    generator = torch.Generator().manual_seed(args.seed)
    target = Qwen3ForCausalLM(Qwen3Config(**TARGET)).eval()
    randomize(target, generator)

    class Args(dict):
        __getattr__ = dict.__getitem__

    draft_config = build_draft_config(
        target.config,
        Args(
            num_draft_layers=2,
            target_layer_ids=TAPS,
            block_size=BLOCK,
            mask_token_id=MASK,
            num_anchors=8,
            confidence_head_alpha=1.0,
            confidence_head_with_markov=True,
            markov_rank=4,
            markov_head_type="vanilla",
        ),
    )
    draft_config._attn_implementation = "sdpa"
    drafter = Qwen3DSparkModel(draft_config).eval()
    randomize(drafter, generator)
    drafter.initialize_embeddings_and_head(embed_tokens=target.model.embed_tokens, lm_head=target.lm_head)

    with torch.inference_mode():
        # The target's own greedy continuation of the prompt, one token at a time.
        sequence = list(PROMPT)
        for _ in range(GENERATED):
            logits = target(input_ids=torch.tensor([sequence])).logits
            sequence.append(int(logits[0, -1].argmax()))
        out = target(input_ids=torch.tensor([sequence]), output_hidden_states=True)
        next_tokens = out.logits[0].argmax(dim=-1).tolist()
        assert next_tokens[len(PROMPT) - 1 : -1] == sequence[len(PROMPT) :]
        features = extract_context_feature(out.hidden_states, TAPS)[0]  # [positions, taps*hidden]

        def draft(cache, new_features, anchor, start):
            ids = torch.full((1, BLOCK), MASK, dtype=torch.long)
            ids[0, 0] = anchor
            hidden = forward_dspark_draft_block(
                drafter,
                draft_input_ids=ids,
                position_ids=torch.arange(start + BLOCK).unsqueeze(0),
                past_key_values_draft=cache,
                target_hidden_states=new_features.unsqueeze(0),
                start=start,
                block_size=BLOCK,
            )
            proposal = build_dspark_proposal(
                drafter,
                draft_input_ids=ids,
                block_hidden=hidden,
                block_size=BLOCK,
                temperature=0.0,
                confidence_threshold=0.0,
            )
            return proposal.verify_input_ids[0, 1:].tolist(), drafter.compute_logits(hidden[:, :BLOCK])[0]

        # Rounds as the driver runs them: the cache grows by the positions each round settles.
        rounds = []
        base_logits = []
        cache = DynamicCache()
        start, seen = len(PROMPT), 0
        for accepted in ACCEPTED:
            anchor = sequence[start]
            tokens, logits = draft(cache, features[seen:start], anchor, start)
            # The same block from an empty cache: the incremental path adds nothing of its own.
            fresh_tokens, fresh_logits = draft(DynamicCache(), features[:start], anchor, start)
            assert tokens == fresh_tokens and torch.allclose(logits, fresh_logits, atol=1e-4)
            corrected = logits + drafter.markov_head.markov_w2(
                drafter.markov_head.markov_w1(torch.tensor([anchor] + tokens[:-1]))
            )
            top2 = corrected.topk(2, dim=-1).values
            rounds.append(
                dict(start=start, anchor=anchor, drafts=tokens, margin=float((top2[:, 0] - top2[:, 1]).min()))
            )
            base_logits.append(logits)
            seen = start
            start += accepted + 1
        assert start + BLOCK < len(sequence)

    variety = len({t for r in rounds for t in r["drafts"]})
    print("distinct draft tokens", variety, "smallest draft margin", min(r["margin"] for r in rounds))
    assert min(r["margin"] for r in rounds) > 0.01 and variety >= 8, "near-tie or dull drafts: try another --seed"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bf16 = lambda t: t.detach().to(torch.bfloat16).contiguous()
    # The target as potpori prepares it: bf16 matrices, f32 norms, tied output head.
    save_file(
        {
            name: (p.detach().contiguous() if "norm" in name else bf16(p))
            for name, p in target.state_dict().items()
            if name != "lm_head.weight"
        },
        str(out_dir / "target.safetensors"),
    )
    # The drafter as DeepSpec publishes it: everything bf16.
    save_file({name: bf16(p) for name, p in drafter.state_dict().items()}, str(out_dir / "drafter.safetensors"))
    save_file(
        {
            "features": features.contiguous(),
            "target_logits": out.logits[0].contiguous(),
            "base_logits": torch.stack(base_logits).contiguous(),
        },
        str(out_dir / "expected.safetensors"),
    )
    (out_dir / "config.json").write_text(draft_config.to_json_string())
    (out_dir / "expected.json").write_text(
        json.dumps(
            dict(
                target=TARGET,
                prompt=PROMPT,
                sequence=sequence,
                next_tokens=next_tokens,
                rounds=rounds,
            ),
            indent=1,
        )
    )
    print(f"wrote {out_dir}")
    print("sequence", sequence)
    for r in rounds:
        print(r)


if __name__ == "__main__":
    main()

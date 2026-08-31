#!/usr/bin/env python3
"""Python reference for the dspark Phase 2 block-draft loop gate.

Drives the same synthetic rounds as tests/test-dspark-loop.cpp through the real
upstream reference ops (deepseek-ai/DeepSpec) and dumps the expected per-round
drafted block to ref.json, which the C++ test diffs against round for round.

    test-dspark-loop <tiny.gguf> <ref.json>

torch must run in a container on this host. See scripts/dspark/README.md for the
exact docker invocation, the fixture builder (build_tiny.py), and the derivation
of every constant duplicated below.

WINDOWING: upstream's forward_dspark_draft_block slices position_ids at ABSOLUTE
positions and calls past_key_values.crop(start). common/speculative.cpp instead
keeps drafter positions inside [0, window) and rebases them, so this driver
applies the same window/rebase rule and calls model._forward_backbone directly
rather than forward_dspark_draft_block. Everything else (embed_tokens, the
backbone, compute_logits, sample_draft_token_step) is the reference's own code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache

from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel

# mirrored verbatim in tests/test-dspark-loop.cpp -- keep both in sync
ACCEPT_SCHEDULE = [7, 3, 0, 7, 5, 1, 4]
PROMPT = [1, 2, 3, 4, 5]

U32 = 0xFFFFFFFF


def hash_u32(x: int) -> int:
    x &= U32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & U32
    x ^= x >> 15
    x = (x * 0x846CA68B) & U32
    x ^= x >> 16
    return x


def synth_feat(pos: int, d: int) -> float:
    h = hash_u32((pos * 131071 + d * 97 + 12345) & U32)
    return float(np.float32((h % 2000) - 1000) / np.float32(500.0))


def synth_bonus_token(round_idx: int, vocab_size: int, mask_token_id: int) -> int:
    h = hash_u32(((round_idx & U32) * 2654435761 + 999983) & U32)
    v = h % (vocab_size - 1)
    if v == mask_token_id:
        v = (v + 1) % vocab_size
    return v


def synth_feat_rows(pos_beg: int, n_rows: int, n_embd_cap: int) -> np.ndarray:
    return np.array(
        [[synth_feat(pos_beg + i, d) for d in range(n_embd_cap)] for i in range(n_rows)],
        dtype=np.float32,
    )


def load_model(export_dir: Path, device: torch.device) -> Qwen3DSparkModel:
    cfg = AutoConfig.from_pretrained(export_dir)
    for k, v in json.loads((export_dir / "config.json").read_text()).items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
    # eager keeps the gate deterministic; the trained config uses flex_attention
    cfg._attn_implementation = "eager"
    model = Qwen3DSparkModel(cfg)
    model.load_state_dict(load_file(export_dir / "model.safetensors"), strict=True)
    return model.to(device=device, dtype=torch.float32).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("export_dir", type=Path, help="tiny export from build_tiny.py")
    ap.add_argument("out", type=Path, help="ref.json to write")
    ap.add_argument("--rounds", type=int, default=len(ACCEPT_SCHEDULE))
    ap.add_argument("--n-batch", type=int, default=None,
                    help="drafter n_batch; defaults to test-dspark-loop.cpp's 2*block_size+9")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.export_dir, device)
    cfg = model.config

    block_size = int(cfg.block_size)
    vocab_size = int(cfg.vocab_size)
    mask_token_id = int(cfg.mask_token_id)
    n_embd_cap = len(cfg.target_layer_ids) * int(cfg.hidden_size)

    # window/w_keep exactly as common/speculative.cpp derives them
    n_batch = args.n_batch if args.n_batch is not None else 2 * block_size + 9
    window = min(int(cfg.max_position_embeddings), n_batch)
    if window <= 2 * block_size:
        raise SystemExit(f"window={window} too small for block_size={block_size}")
    w_keep = min(window // 2, window - block_size)

    start = len(PROMPT)
    base = 0
    id_last = synth_bonus_token(-1, vocab_size, mask_token_id)

    rounds = []
    n_rebases = 0
    for r in range(args.rounds):
        n_accepted = ACCEPT_SCHEDULE[r % len(ACCEPT_SCHEDULE)]

        # the feature buffer is a pure function of position: after the prefill
        # and every accept(), it covers absolute positions [0, start)
        n_buf = start
        rebase = start - base + block_size > window
        if rebase:
            base = start - min(n_buf, w_keep)
            n_rebases += 1

        # attended context is exactly [base, start): a rebase wiped everything
        # below `base` out of the drafter cache, so restaging from 0 would hand
        # the rows negative rebased positions
        ctx_pos = list(range(base, n_buf))
        feat = synth_feat_rows(base, len(ctx_pos), n_embd_cap)
        target_hidden = torch.from_numpy(feat).unsqueeze(0).to(device)

        draft_input_ids = torch.full((1, block_size), mask_token_id,
                                     dtype=torch.long, device=device)
        draft_input_ids[0, 0] = id_last

        # rebased positions, matching the C++ batch: context rows at pos-base,
        # block rows at start-base+k
        position_ids = torch.tensor(
            [[p - base for p in ctx_pos] + [start - base + k for k in range(block_size)]],
            dtype=torch.long, device=device,
        )

        with torch.no_grad():
            block_hidden = model._forward_backbone(
                target_hidden_states=target_hidden,
                noise_embedding=model.embed_tokens(draft_input_ids),
                position_ids=position_ids,
                attention_mask=None,
                past_key_values=DynamicCache(),
                use_cache=False,
                is_causal=False,
            )
            base_logits = model.compute_logits(block_hidden).float()

            # sequential Markov resample: never batched over the block
            prev = torch.tensor([id_last], dtype=torch.long, device=device)
            sampled = []
            for k in range(block_size):
                tok, _ = model.sample_draft_token_step(
                    base_logits[:, k, :], prev_token_ids=prev, temperature=0.0,
                )
                sampled.append(int(tok.item()))
                prev = tok

        bonus = synth_bonus_token(r, vocab_size, mask_token_id)
        rounds.append({
            "n_accepted": n_accepted,
            "ctx_len": n_buf,
            "start": start,
            "sampled": sampled,
            "bonus": bonus,
            "rebase": rebase,
        })
        print(f"round {r}: start={start} base={base} ctx={len(ctx_pos)} "
              f"rebase={rebase} sampled={sampled}")

        id_last = bonus
        start += n_accepted + 1

    ref = {
        "n_embd_cap": n_embd_cap,
        "block_size": block_size,
        "vocab_size": vocab_size,
        "mask_token_id": mask_token_id,
        "prefill_bonus": synth_bonus_token(-1, vocab_size, mask_token_id),
        "window": window,
        "w_keep": w_keep,
        "rounds": rounds,
    }
    args.out.write_text(json.dumps(ref, indent=2) + "\n")
    print(f"wrote {args.out} ({len(rounds)} rounds, {n_rebases} rebase(s), "
          f"window={window}, w_keep={w_keep})")
    if n_rebases == 0:
        print("WARNING: no rebase fired -- the gate would pass vacuously")


if __name__ == "__main__":
    main()

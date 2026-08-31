#!/usr/bin/env python3
"""Build the tiny synthetic dspark drafter export used by the Phase 2 gate.

Writes an HF-format export (config.json + model.safetensors). Convert it with
the repo's own converter so the GGUF and the torch model hold identical weights:

    python convert_hf_to_gguf.py <outdir> --outfile tiny.gguf --outtype f32

The result feeds tests/test-dspark-loop.cpp as argv[1], and the same export dir
feeds scripts/dspark/phase2_py_ref.py, which loads it through the upstream
DeepSpec Qwen3DSparkModel. See scripts/dspark/README.md.

Weights are drawn from a fixed seed, so the export is reproducible. They are not
the original fixture's weights (those were lost); the gate compares C++ against
Python on the SAME checkpoint, so only self-consistency matters.

No torch: numpy + safetensors only, so this runs on the host. Only the reference
driver needs a torch container.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

# max_position_embeddings becomes the GGUF context_length, which becomes the
# drafter's n_ctx_train and therefore `window` in common/speculative.cpp
# (window = min(n_ctx_train, n_batch)). test-dspark-loop.cpp runs at
# n_batch = 2*block_size + 9 = 23, so 20 keeps window at 20 with w_keep = 10 and
# forces a rebase on round 3 of the ACCEPT_SCHEDULE. Do not raise it without
# rechecking that arithmetic -- above 23 the window stops binding and the gate
# never exercises the rebase path. See scripts/dspark/README.md.
CONFIG = {
    "architectures": ["Qwen3DSparkModel"],
    "model_type": "qwen3",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "vocab_size": 64,
    "max_position_embeddings": 20,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "sliding_window": None,
    "layer_types": ["full_attention", "full_attention"],
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    # dspark-specific (read by Qwen3DSparkModel.__init__ and conversion/dspark.py)
    "num_target_layers": 4,
    "target_layer_ids": [0, 1, 2],
    "block_size": 7,
    "mask_token_id": 63,
    "num_anchors": 1,
    "enable_confidence_head": False,
    "markov_rank": 4,
    "markov_head_type": "vanilla",
}


def build_tensors(cfg: dict, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    h = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    head_dim = cfg["head_dim"]
    n_q = cfg["num_attention_heads"] * head_dim
    n_kv = cfg["num_key_value_heads"] * head_dim
    n_cap = len(cfg["target_layer_ids"])
    rank = cfg["markov_rank"]

    def lin(*shape):
        return rng.normal(0.0, 0.02, size=shape).astype(np.float32)

    def norm(n):
        # jittered rather than exactly 1.0: an ignored norm weight would be
        # invisible against an all-ones init
        return (1.0 + rng.normal(0.0, 0.02, size=(n,))).astype(np.float32)

    t = {
        "embed_tokens.weight": lin(vocab, h),
        "norm.weight": norm(h),
        "fc.weight": lin(h, n_cap * h),
        "hidden_norm.weight": norm(h),
        "lm_head.weight": lin(vocab, h),
        "markov_head.markov_w1.weight": lin(vocab, rank),
        "markov_head.markov_w2.weight": lin(vocab, rank),
    }
    for i in range(cfg["num_hidden_layers"]):
        p = f"layers.{i}."
        t[p + "self_attn.q_proj.weight"] = lin(n_q, h)
        t[p + "self_attn.k_proj.weight"] = lin(n_kv, h)
        t[p + "self_attn.v_proj.weight"] = lin(n_kv, h)
        t[p + "self_attn.o_proj.weight"] = lin(h, n_q)
        t[p + "self_attn.q_norm.weight"] = norm(head_dim)
        t[p + "self_attn.k_norm.weight"] = norm(head_dim)
        t[p + "mlp.gate_proj.weight"] = lin(inter, h)
        t[p + "mlp.up_proj.weight"] = lin(inter, h)
        t[p + "mlp.down_proj.weight"] = lin(h, inter)
        t[p + "input_layernorm.weight"] = norm(h)
        t[p + "post_attention_layernorm.weight"] = norm(h)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir", type=Path, help="export directory to create")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    tensors = build_tensors(CONFIG, args.seed)
    save_file(tensors, str(args.outdir / "model.safetensors"))
    (args.outdir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")

    n_params = sum(int(v.size) for v in tensors.values())
    print(f"wrote {len(tensors)} tensors ({n_params} params, seed={args.seed}) to {args.outdir}")
    print(f"convert with:\n  python convert_hf_to_gguf.py {args.outdir} "
          f"--outfile {args.outdir / 'tiny.gguf'} --outtype f32")


if __name__ == "__main__":
    main()

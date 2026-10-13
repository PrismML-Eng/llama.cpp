#!/usr/bin/env python3
# Generate a tiny random-weight dspark GGUF for CPU tests of the loader's
# log-SNR metadata contract (src/models/dspark.cpp load_arch_hparams):
#
#   tests/test-dspark-logsnr-meta.cpp
#
# The weights are seeded random: the test only exercises model LOADING (the
# hparam guard fires before any tensor data is touched), so weight values are
# irrelevant. Unlike gen-tiny-qwen35.py this writes no tokenizer: the real
# dspark converter calls _set_vocab_none() (the drafter ties to the TARGET
# model's vocab), so tokenizer.ggml.model = "none" both matches production
# GGUFs. Explicit vocab_size also makes token IDs valid in forward smoke tests.
#
# usage (single variant):
#   python3 tests/gen-tiny-dspark.py out.gguf --log-snr-conditioning 1 \
#       --min-log-snr -9 --max-log-snr 6
#   python3 tests/gen-tiny-dspark.py out.gguf --log-snr-conditioning 1 \
#       --min-log-snr nan --max-log-snr 6
#   python3 tests/gen-tiny-dspark.py out.gguf --log-snr-conditioning 1 --omit-min
#
# usage (full suite expected by test-dspark-logsnr-meta):
#   python3 tests/gen-tiny-dspark.py --suite <dir>

from __future__ import annotations

import logging
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))
from gguf import GGUFWriter, GGUFValueType  # noqa: E402

# smallest shape the dspark loader accepts: every tensor in
# load_arch_tensors is present, with the optional heads (markov_rank = 0,
# confidence_head off) and the tied-output fallback left to defaults.
N_LAYER       = 2
N_EMBD        = 64
N_FF          = 96
N_HEAD        = 2
N_HEAD_KV     = 1
N_CTX         = 512
N_VOCAB       = 256   # only exists as token_embd.weight's shape (vocab "none")
HEAD_DIM      = N_EMBD // N_HEAD              # 32
TARGET_LAYERS = [1, 16]                       # n_capture = 2 (target-model indices)
BLOCK_SIZE    = 4
MASK_TOKEN_ID = 255
N_FREQ        = 128   # LogSnrEmbed.NUM_FREQ_FEATURES (fixed in dspark.cpp)


logger = logging.getLogger("gen-tiny-dspark")


def write_variant(out_path: str, conditioning: bool,
                  min_log_snr: float | None, max_log_snr: float | None,
                  correction: bool = False) -> None:
    rng = np.random.default_rng(1234)

    def rand(*shape: int) -> np.ndarray:
        return rng.normal(0.0, 0.02, size=shape).astype(np.float32)

    w = GGUFWriter(out_path, "dspark")
    w.add_name("tiny-random-dspark")
    w.add_file_type(0)  # all f32

    w.add_block_count(N_LAYER)
    w.add_context_length(N_CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(10000.0)

    w.add_uint32("dspark.dspark.block_size",    BLOCK_SIZE)
    w.add_uint32("dspark.dspark.mask_token_id", MASK_TOKEN_ID)
    w.add_key_value("dspark.dspark.target_layers", TARGET_LAYERS,
                    GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT32)

    w.add_bool("dspark.dspark.log_snr_conditioning", conditioning)
    if min_log_snr is not None:
        w.add_float32("dspark.dspark.min_log_snr", min_log_snr)
    if max_log_snr is not None:
        w.add_float32("dspark.dspark.max_log_snr", max_log_snr)

    w.add_tokenizer_model("none")
    w.add_vocab_size(N_VOCAB)
    if correction:
        w.add_bool("dspark.dspark.hidden_correction", True)
        w.add_bool("dspark.dspark.markov_deploy_exposure", True)
        w.add_uint32("dspark.dspark.correction_size", N_EMBD)
        w.add_uint32("dspark.dspark.markov_rank", 8)
        w.add_tensor("dspark.correction_hidden_norm.weight", np.ones(N_EMBD, dtype=np.float32))
        w.add_tensor("dspark.correction_embed_norm.weight", np.ones(N_EMBD, dtype=np.float32))
        w.add_tensor("dspark.correction_gate.weight", rand(N_EMBD, 2 * N_EMBD))
        w.add_tensor("dspark.correction_up.weight", rand(N_EMBD, 2 * N_EMBD))
        w.add_tensor("dspark.correction_down.weight", rand(N_EMBD, N_EMBD))
        w.add_tensor("dspark.markov_head_a.weight", rand(N_VOCAB, 8))
        w.add_tensor("dspark.markov_head_b.weight", rand(N_VOCAB, 8))

    # note: numpy shapes are the reverse of the ggml ne[] order
    w.add_tensor("token_embd.weight", rand(N_VOCAB, N_EMBD))
    w.add_tensor("output_norm.weight", np.ones(N_EMBD, dtype=np.float32))
    # output.weight omitted -> duplicated from token_embd

    n_embd_cap = len(TARGET_LAYERS) * N_EMBD
    w.add_tensor("dspark.fc.weight", rand(N_EMBD, n_embd_cap))
    w.add_tensor("dspark.hidden_norm.weight", np.ones(N_EMBD, dtype=np.float32))

    # the log-SNR MLP is REQUIRED by the loader when conditioning is on, and
    # must be absent otherwise (unreferenced tensors fail the load)
    if conditioning:
        w.add_tensor("dspark.log_snr_fc1.weight", rand(N_EMBD, N_FREQ))
        w.add_tensor("dspark.log_snr_fc1.bias",   rand(N_EMBD))
        w.add_tensor("dspark.log_snr_fc2.weight", rand(N_EMBD, N_EMBD))
        w.add_tensor("dspark.log_snr_fc2.bias",   rand(N_EMBD))

    for il in range(N_LAYER):
        w.add_tensor(f"blk.{il}.attn_norm.weight", np.ones(N_EMBD, dtype=np.float32))

        w.add_tensor(f"blk.{il}.attn_q.weight", rand(HEAD_DIM * N_HEAD,    N_EMBD))
        w.add_tensor(f"blk.{il}.attn_k.weight", rand(HEAD_DIM * N_HEAD_KV, N_EMBD))
        w.add_tensor(f"blk.{il}.attn_v.weight", rand(HEAD_DIM * N_HEAD_KV, N_EMBD))
        w.add_tensor(f"blk.{il}.attn_output.weight", rand(N_EMBD, HEAD_DIM * N_HEAD))

        w.add_tensor(f"blk.{il}.attn_q_norm.weight", np.ones(HEAD_DIM, dtype=np.float32))
        w.add_tensor(f"blk.{il}.attn_k_norm.weight", np.ones(HEAD_DIM, dtype=np.float32))

        w.add_tensor(f"blk.{il}.ffn_norm.weight", np.ones(N_EMBD, dtype=np.float32))
        w.add_tensor(f"blk.{il}.ffn_gate.weight", rand(N_FF, N_EMBD))
        w.add_tensor(f"blk.{il}.ffn_down.weight", rand(N_EMBD, N_FF))
        w.add_tensor(f"blk.{il}.ffn_up.weight",   rand(N_FF, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    logger.info(f"wrote {out_path} (conditioning={int(conditioning)} "
                f"min={min_log_snr} max={max_log_snr})")


# canonical suite consumed by tests/test-dspark-logsnr-meta.cpp: keep the
# file names and settings in sync with EXPECTED_VARIANTS there
SUITE = [
    # name                  cond   min            max
    ("logsnr-valid.gguf",    True, -9.0,          6.0),
    ("logsnr-omit-min.gguf", True, None,          6.0),
    ("logsnr-omit-max.gguf", True, -9.0,          None),
    ("logsnr-min-nan.gguf",  True, float("nan"),  6.0),
    ("logsnr-max-inf.gguf",  True, -9.0,          float("inf")),
    ("logsnr-equal.gguf",    True, 6.0,           6.0),
    ("logsnr-reversed.gguf", True, 6.0,          -9.0),
    ("logsnr-cond-off.gguf", False, None,         None),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", help="output GGUF path (single-variant mode)")
    ap.add_argument("--log-snr-conditioning", type=int, choices=(0, 1), default=0)
    ap.add_argument("--min-log-snr", type=float, default=None,
                    help="min_log_snr metadata value (accepts nan/inf/-inf)")
    ap.add_argument("--max-log-snr", type=float, default=None,
                    help="max_log_snr metadata value (accepts nan/inf/-inf)")
    ap.add_argument("--omit-min", action="store_true", help="omit min_log_snr")
    ap.add_argument("--omit-max", action="store_true", help="omit max_log_snr")
    ap.add_argument("--hidden-correction", action="store_true")
    ap.add_argument("--suite", metavar="DIR",
                    help="write the full variant suite for test-dspark-logsnr-meta")
    args = ap.parse_args()

    if args.suite:
        out_dir = Path(args.suite)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, cond, lo, hi in SUITE:
            write_variant(str(out_dir / name), cond, lo, hi)
        return

    if not args.out:
        ap.error("either an output path or --suite is required")

    min_v = None if args.omit_min else args.min_log_snr
    max_v = None if args.omit_max else args.max_log_snr
    write_variant(args.out, bool(args.log_snr_conditioning), min_v, max_v, args.hidden_correction)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

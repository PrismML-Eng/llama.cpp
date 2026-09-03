#!/usr/bin/env python3
"""Generate a tiny random DFly (AngelSpec) draft GGUF for loader smoke tests.

Shapes mirror the reference AngelSlim/Qwen3-8B-DFly-Block8 topology (5 target capture
layers, per-draft-layer context fusion, swiglu predecessor correction) at toy width, with
n_layer deliberately != n_target so a transposed fusion axis cannot pass unnoticed.

    python3 tests/gen-tiny-dfly.py models/ggml-vocab-qwen2.gguf /tmp/tiny-dfly.gguf
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))
import gguf  # noqa: E402

VOCAB_KEY_PREFIXES = ("tokenizer.",)


def copy_vocab(writer: gguf.GGUFWriter, vocab_gguf: Path) -> int:
    reader = gguf.GGUFReader(vocab_gguf)
    n_vocab = 0
    for field in reader.fields.values():
        if not field.name.startswith(VOCAB_KEY_PREFIXES):
            continue
        if field.types[:1] == [gguf.GGUFValueType.ARRAY]:
            sub = field.types[1]
            if sub == gguf.GGUFValueType.STRING:
                vals = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
                writer.add_array(field.name, vals)
                if field.name == "tokenizer.ggml.tokens":
                    n_vocab = len(vals)
            else:
                vals = [field.parts[i].tolist()[0] for i in field.data]
                writer.add_array(field.name, vals)
        else:
            val = field.parts[field.data[0]]
            if field.types[0] == gguf.GGUFValueType.STRING:
                writer.add_string(field.name, bytes(val).decode("utf-8"))
            else:
                writer.add_uint32(field.name, int(val.tolist()[0]))
    if n_vocab == 0:
        raise SystemExit(f"no tokenizer.ggml.tokens found in {vocab_gguf}")
    return n_vocab


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <vocab.gguf> <out.gguf>")
    vocab_gguf, out_path = Path(sys.argv[1]), Path(sys.argv[2])

    n_embd, n_layer, n_head, n_head_kv = 64, 3, 4, 2
    n_ff, n_feat, n_ff_hc = 128, 5, 64
    head_dim = n_embd // n_head
    eps, block_size = 1e-6, 8

    writer = gguf.GGUFWriter(out_path, "dflash")
    n_vocab = copy_vocab(writer, vocab_gguf)

    writer.add_context_length(512)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_layer_norm_rms_eps(eps)
    writer.add_rope_freq_base(1000000.0)
    writer.add_rope_dimension_count(head_dim)
    writer.add_block_size(block_size)
    writer.add_sample_from_anchor(False)
    # +1: the runtime taps a layer's INPUT (same convention as the DFlash converter)
    writer.add_target_layers([i + 1 for i in (1, 9, 17, 25, 33)][:n_feat])

    rng = np.random.default_rng(7)

    def t(name: str, shape: tuple[int, ...]) -> None:
        writer.add_tensor(name, rng.standard_normal(shape).astype(np.float32) * 0.05)

    t("token_embd.weight", (n_vocab, n_embd))
    t("output_norm.weight", (n_embd,))

    # DFly encoder: shared base projection + per-draft-layer fusion, then context_norm.
    # Note there is NO enc.output_norm (hidden_norm): DFly replaces it with context_norm.
    t("fc.weight", (n_embd, n_feat * n_embd))
    t("layer_fusion", (n_layer, n_feat))
    t("context_norm.weight", (n_embd,))

    # TreeFlash predecessor correction (swiglu over [hidden ; prev-token embedding])
    t("hidden_correction.hidden_norm.weight", (n_embd,))
    t("hidden_correction.embed_norm.weight", (n_embd,))
    t("hidden_correction.gate.weight", (n_ff_hc, 2 * n_embd))
    t("hidden_correction.up.weight", (n_ff_hc, 2 * n_embd))
    t("hidden_correction.down.weight", (n_embd, n_ff_hc))

    for i in range(n_layer):
        t(f"blk.{i}.attn_norm.weight", (n_embd,))
        t(f"blk.{i}.attn_q.weight", (n_head * head_dim, n_embd))
        t(f"blk.{i}.attn_k.weight", (n_head_kv * head_dim, n_embd))
        t(f"blk.{i}.attn_v.weight", (n_head_kv * head_dim, n_embd))
        t(f"blk.{i}.attn_output.weight", (n_embd, n_head * head_dim))
        t(f"blk.{i}.attn_q_norm.weight", (head_dim,))
        t(f"blk.{i}.attn_k_norm.weight", (head_dim,))
        t(f"blk.{i}.ffn_norm.weight", (n_embd,))
        t(f"blk.{i}.ffn_gate.weight", (n_ff, n_embd))
        t(f"blk.{i}.ffn_up.weight", (n_ff, n_embd))
        t(f"blk.{i}.ffn_down.weight", (n_embd, n_ff))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {out_path} (n_vocab={n_vocab}, n_layer={n_layer}, n_feat={n_feat}, "
          f"expected n_embd_out={n_layer * n_embd})")


if __name__ == "__main__":
    main()

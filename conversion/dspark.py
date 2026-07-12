from __future__ import annotations

from typing import Any, Iterable, TYPE_CHECKING

from .base import ModelBase, TextModel, gguf, logger

if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("Qwen3DSparkModel", "DSparkForCausalLM", "DsparkSpeculator")
class DSparkModel(TextModel):
    """Converter for the dspark speculative-decoding drafter.

    dspark is an EAGLE-style block-diffusion drafter. This converter maps an
    EasyDeL dspark export onto the dspark GGUF tensor names. The drafter forward
    graph and block-diffusion draft loop are implemented (src/models/dspark.cpp,
    common/speculative.cpp), so the produced GGUF loads and runs as a draft-dspark
    speculator. See docs/dspark-scope.md for the drafter shape and the capture API.

    Tensor name mapping (HF Qwen3DSparkModel export -> gguf):
        fc                          -> dspark.fc
        hidden_norm                 -> dspark.hidden_norm
        norm                        -> output_norm
        lm_head                     -> output
        embed_tokens                -> token_embd
        markov_head.markov_w1       -> dspark.markov_head_a   (prev-token Embed [vocab, rank])
        markov_head.markov_w2       -> dspark.markov_head_b   (Linear [vocab, rank])
        confidence_head.proj.weight -> dspark.confidence_head.weight
        confidence_head.proj.bias   -> dspark.confidence_head.bias
        layers.{i}.<attn/ffn/...>   -> blk.{i}.<standard names via tensor_map>

    Older EasyDeL exports used markov_head.{down,up} and a bare confidence_head;
    those aliases are kept below so both layouts convert.
    """

    model_arch = gguf.MODEL_ARCH.DSPARK

    def set_vocab(self):
        # dspark drafter ships no tokenizer; it ties to the TARGET model's
        # vocab and always operates on token IDs the target's own tokenizer
        # already produced (never on strings). tokenizer.ggml.model=none means
        # llama.cpp loads zero real vocab entries by default, which then makes
        # every token id fail the generic "token < n_vocab" batch-validation
        # check in llama-batch.cpp -- not just detokenization, ANY decode()
        # call. add_vocab_size() tells the "none" tokenizer loader (see
        # llama_vocab::impl::load in src/llama-vocab.cpp) to fill in that many
        # placeholder/dummy entries, purely so vocab.n_tokens() reports the
        # real (target) vocab width and batch validation passes. These
        # placeholder entries carry no real strings and this GGUF cannot be
        # used standalone for text I/O -- don't try to "fix" them into
        # something meaningful.
        self._set_vocab_none()
        vocab_size = self.hparams.get("vocab_size")
        if vocab_size is not None:
            self.gguf_writer.add_vocab_size(int(vocab_size))

    # explicit head/structural remap; per-layer decoder tensors fall through to
    # the standard tensor_map (self.map_tensor_name) used by TextModel.
    _name_map = {
        "fc":                    gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DSPARK_FC],
        "hidden_norm":           gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DSPARK_HIDDEN_NORM],
        # HF Qwen3DSparkModel names: markov_w1 = prev-token Embed [vocab, rank],
        # markov_w2 = Linear [vocab, rank]; confidence_head is a proj with a bias.
        "markov_head.markov_w1": gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DSPARK_MARKOV_HEAD_A],
        "markov_head.markov_w2": gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DSPARK_MARKOV_HEAD_B],
        "confidence_head.proj":  gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DSPARK_CONFIDENCE_HEAD],
        # legacy EasyDeL aliases (kept so older exports still convert):
        "markov_head.down":      gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DSPARK_MARKOV_HEAD_A],
        "markov_head.up":        gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DSPARK_MARKOV_HEAD_B],
        "confidence_head":       gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DSPARK_CONFIDENCE_HEAD],
    }

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        hp = self.hparams

        # dspark drafter trunk is a small transformer; block_count is its depth.
        # (EasyDeL exports this under "num_hidden_layers"; TextModel already wired
        # block_count from that, so nothing extra needed here.)

        block_size = int(hp.get("block_size", 7))
        self.gguf_writer.add_dspark_block_size(block_size)

        mask_token_id = hp.get("mask_token_id")
        if mask_token_id is not None:
            self.gguf_writer.add_dspark_mask_token_id(int(mask_token_id))

        target_layers = hp.get("target_layer_ids")
        if target_layers is not None:
            self.gguf_writer.add_dspark_target_layers([int(x) for x in target_layers])

        markov_rank = hp.get("markov_rank")
        if markov_rank is not None:
            self.gguf_writer.add_dspark_markov_rank(int(markov_rank))

        enable_conf = hp.get("enable_confidence_head")
        if enable_conf is not None:
            self.gguf_writer.add_dspark_confidence_head(bool(enable_conf))

        conf_with_markov = hp.get("confidence_head_with_markov")
        if conf_with_markov is not None:
            self.gguf_writer.add_dspark_confidence_head_with_markov(bool(conf_with_markov))

        logger.info(
            "dspark: exported drafter (block_size=%d, target_layers=%s); "
            "see docs/dspark-scope.md",
            block_size, target_layers,
        )

    def modify_tensors(self, data_torch: "Tensor", name: str, bid: int | None) -> Iterable[tuple[str, "Tensor"]]:
        n = name

        # strip a leading model. / drafter. wrapper if present
        for prefix in ("model.", "drafter.", "dspark."):
            if n.startswith(prefix):
                n = n[len(prefix):]
                break

        # structural / head tensors with a direct mapping
        for src, dst in self._name_map.items():
            if n == f"{src}.weight":
                return [(dst + ".weight", data_torch)]
            if n == f"{src}.bias":
                return [(dst + ".bias", data_torch)]

        if n in ("norm.weight", "final_norm.weight"):
            return [(gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.OUTPUT_NORM] + ".weight", data_torch)]
        if n in ("lm_head.weight",):
            return [(gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.OUTPUT] + ".weight", data_torch)]
        if n in ("embed_tokens.weight", "tok_embeddings.weight"):
            return [(gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.TOKEN_EMBD] + ".weight", data_torch)]

        # per-layer decoder tensors fall through to the standard mapping. The base
        # class resolves (model.)layers.{bid}.<attn/mlp...> to blk.{bid}.<gguf name>.
        # Strip only the dspark-specific wrapper (drafter./dspark.) here -- the head
        # loop above matched against the fully stripped `n`, but this fallthrough
        # must not pass an unsupported drafter./dspark. prefix to map_tensor_name or
        # the per-layer tensor fails to map. Keep any standard model. prefix, which
        # map_tensor_name already understands.
        fallthrough_name = name
        for prefix in ("drafter.", "dspark."):
            if fallthrough_name.startswith(prefix):
                fallthrough_name = fallthrough_name[len(prefix):]
                break
        return [(self.map_tensor_name(fallthrough_name), data_torch)]

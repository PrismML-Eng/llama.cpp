from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

from .base import ModelBase, TextModel, gguf, logger

if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("Qwen3DSparkModel", "DSparkForCausalLM", "DsparkSpeculator", "DSparkModel")
class DSparkModel(TextModel):
    """Convert a DSpark checkpoint with its target vocabulary and correction weights."""

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
        "log_snr_embed.fc1": "dspark.log_snr_fc1",
        "log_snr_embed.fc2": "dspark.log_snr_fc2",
        "hidden_correction.hidden_norm": "dspark.correction_hidden_norm",
        "hidden_correction.embed_norm": "dspark.correction_embed_norm",
        "hidden_correction.gate_proj": "dspark.correction_gate",
        "hidden_correction.up_proj": "dspark.correction_up",
        "hidden_correction.down_proj": "dspark.correction_down",
        # HF Qwen3DSparkModel names: markov_w1 = prev-token Embed [vocab, rank],
        # markov_w2 = Linear [vocab, rank]; confidence_head is a proj with a bias.
        "markov_head.markov_w1": gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DSPARK_MARKOV_HEAD_A],
        "markov_head.prev_embed": gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DSPARK_MARKOV_HEAD_A],
        "markov_head.markov_w2": gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DSPARK_MARKOV_HEAD_B],
        "confidence_head.proj":  gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DSPARK_CONFIDENCE_HEAD],
        # legacy aliases (kept so older exports still convert):
        "markov_head.down":      gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DSPARK_MARKOV_HEAD_A],
        "markov_head.up":        gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DSPARK_MARKOV_HEAD_B],
        "confidence_head":       gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.DSPARK_CONFIDENCE_HEAD],
    }

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        hp = self.hparams
        correction = bool(hp.get("enable_hidden_correction", False))
        exposure = bool(hp.get("markov_deploy_exposure", False))
        if correction and (not exposure or hp.get("markov_head_type", "vanilla") != "vanilla"):
            raise ValueError("Production DSpark requires vanilla deploy-exposure Markov")
        if exposure and not correction:
            raise ValueError("Deploy exposure without hidden correction is unsupported")
        self.gguf_writer.add_bool("dspark.dspark.hidden_correction", correction)
        self.gguf_writer.add_bool("dspark.dspark.markov_deploy_exposure", exposure)
        if correction:
            self.gguf_writer.add_uint32("dspark.dspark.correction_size", int(hp.get("hidden_correction_intermediate_size") or hp["hidden_size"]))
        snr = bool(hp.get("log_snr_conditioning", False))
        self.gguf_writer.add_bool("dspark.dspark.log_snr_conditioning", snr)
        if snr:
            self.gguf_writer.add_float32("dspark.dspark.min_log_snr", float(hp["min_log_snr"]))
            self.gguf_writer.add_float32("dspark.dspark.max_log_snr", float(hp["max_log_snr"]))

        # dspark drafter trunk is a small transformer; block_count is its depth.
        # (exports carry this as "num_hidden_layers"; TextModel already wired
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

        logger.info("dspark: block_size=%d target_layers=%s hidden_correction=%s", block_size, target_layers, correction)

    def modify_tensors(self, data_torch: "Tensor", name: str, bid: int | None) -> Iterable[tuple[str, "Tensor"]]:
        n = name

        # strip a leading model. / drafter. wrapper if present
        for prefix in ("model.", "drafter.", "dspark."):
            if n.startswith(prefix):
                n = n[len(prefix):]
                break

        # structural / head tensors with a direct mapping
        if n == "hidden_correction.gate_up_proj.weight":
            width = int(self.hparams.get("hidden_correction_intermediate_size") or self.hparams["hidden_size"])
            if tuple(data_torch.shape) != (2 * width, 2 * int(self.hparams["hidden_size"])):
                raise ValueError("Unexpected fused hidden-correction projection shape")
            if int(self.hparams.get("fused_param_tp", 1)) != 1:
                raise ValueError("Unfuse TP-interleaved correction weights before GGUF conversion")
            return [("dspark.correction_gate.weight", data_torch[:width]),
                    ("dspark.correction_up.weight", data_torch[width:])]
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
        # class resolves layers.{bid}.<attn/mlp...> to blk.{bid}.<gguf name>.
        return [(self.map_tensor_name(n), data_torch)]

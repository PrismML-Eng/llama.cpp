from __future__ import annotations

import json

from typing import Iterable, TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf


@ModelBase.register("MapleForCausalLM")
@ModelBase.example("deepgrove/maple-preview")
class MapleModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MAPLE

    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens, toktypes, tokpre = super().get_vocab_base()
        with (self.dir_model / "tokenizer.json").open(encoding="utf-8") as stream:
            tokenizer = json.load(stream)
        with (self.dir_model / "tokenizer_config.json").open(encoding="utf-8") as stream:
            settings = json.load(stream)
        if tokenizer.get("normalizer") != {"type": "NFC"}:
            raise ValueError("Maple requires the canonical exact NFC normalizer")
        if tokenizer.get("model", {}).get("type") != "BPE" or tokpre != "qwen2":
            raise ValueError("Unsupported Maple tokenizer family")
        # Keep the existing BPE vocabulary/padding conventions, but do not infer
        # specialness from spelling: Maple's FIM/tool/think tokens are lexical.
        for text, token_id in tokenizer["model"]["vocab"].items():
            if type(token_id) is not int or not 0 <= token_id < len(tokens) or tokens[token_id] != text:
                raise ValueError(f"Maple canonical vocabulary mismatch at {token_id}")
            toktypes[token_id] = gguf.TokenType.NORMAL
        for added in tokenizer["added_tokens"]:
            token_id = added["id"]
            if type(token_id) is not int or not 0 <= token_id < len(tokens) or tokens[token_id] != added["content"]:
                raise ValueError(f"Maple canonical added-token mismatch at {token_id}")
            if type(added.get("special")) is not bool:
                raise ValueError(f"Missing Maple added-token special flag at {token_id}")
            other = settings.get("added_tokens_decoder", {}).get(str(token_id))
            flags = ("content", "special", "lstrip", "rstrip", "single_word", "normalized")
            if other is None or any(added.get(key) != other.get(key) for key in flags):
                raise ValueError(f"Maple added-token configurations disagree at {token_id}")
            if any(added.get(key) is not False for key in ("lstrip", "rstrip", "single_word", "normalized")):
                raise ValueError("Unsupported Maple added-token boundary transformation")
            toktypes[token_id] = gguf.TokenType.CONTROL if added["special"] else gguf.TokenType.USER_DEFINED
        return tokens, toktypes, tokpre

    def set_vocab(self):
        self._set_vocab_gpt2()
        # GenerationConfig overrides model EOS, including an explicit null/empty
        # policy. The runtime must not append stop-looking lexical tokens.
        generation_path = self.dir_model / "generation_config.json"
        generation = {}
        if generation_path.is_file():
            with generation_path.open(encoding="utf-8") as stream:
                generation = json.load(stream)
        eos = generation.get("eos_token_id", self.hparams.get("eos_token_id"))
        ids = [] if eos is None else eos if isinstance(eos, list) else [eos]
        if any(type(token_id) is not int or not 0 <= token_id < self.hparams["vocab_size"] for token_id in ids):
            raise ValueError("Invalid Maple generation EOS IDs")
        self.gguf_writer.add_eog_token_ids(list(dict.fromkeys(ids)))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        assert hparams["hidden_act"] == "silu"
        assert hparams.get("num_shared_experts", 0) == 0
        assert hparams.get("norm_topk_prob", True)
        assert hparams.get("nope_on_global_attention", False)

        head_dim = hparams.get("head_dim", hparams["hidden_size"] // hparams["num_attention_heads"])
        partial_rotary_factor = self.rope_parameters.get("partial_rotary_factor", 1.0)

        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_rope_dimension_count(int(head_dim * partial_rotary_factor))
        # FlashAttention's left span is inclusive: 512 past keys + current key.
        # GGUF STANDARD stores total causal keys; normalize only in the producer.
        left_span = hparams["sliding_window"]
        if type(left_span) is not int or not 0 < left_span < 2**32 - 1:
            raise ValueError("Invalid Maple sliding-window left span")
        self.gguf_writer.add_sliding_window(left_span + 1)
        self.gguf_writer.add_sliding_window_pattern([layer_type == "sliding_attention" for layer_type in hparams["layer_types"]])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        # the reference clamps the MoE SwiGLU gate/up at 7.0 (modeling_maple.py)
        self.gguf_writer.add_swiglu_clamp_exp([7.0] * self.block_count)

    _experts: list[dict[str, Tensor]] | None = None

    @staticmethod
    def _stack_experts(tensors: list[Tensor]) -> Tensor:
        shape = (len(tensors), *tensors[0].shape)
        dtype = tensors[0].dtype
        meta = LazyTorchTensor.meta_with_dtype_and_shape(dtype, shape)

        # tensors goes through args, not the closure, so that `func` matches
        # LazyBase's single-argument shape
        def stack(ts: list[Tensor]) -> Tensor:
            result = torch.empty(shape, dtype=dtype)
            for expert_id, tensor in enumerate(ts):
                result[expert_id].copy_(LazyTorchTensor.to_eager(tensor))
            ts.clear()
            return result

        return cast(torch.Tensor, LazyTorchTensor(meta=meta, args=(tensors,), func=stack))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "mlp.experts" in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                for weight_name in ("down_proj", "gate_proj", "up_proj"):
                    tensors = []

                    for expert_id in range(n_experts):
                        expert_name = f"model.layers.{bid}.mlp.experts.{expert_id}.{weight_name}.weight"
                        tensors.append(self._experts[bid].pop(expert_name))

                    merged_name = f"model.layers.{bid}.mlp.experts.{weight_name}.weight"
                    yield from super().modify_tensors(self._stack_experts(tensors), merged_name, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            experts = [name for layer in self._experts for name in layer]
            if experts:
                raise ValueError(f"Unprocessed experts: {experts}")

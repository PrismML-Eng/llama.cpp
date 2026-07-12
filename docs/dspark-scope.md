# dspark drafter: scope and groundwork

## Status

This document records what is built now and what is deliberately deferred for the
dspark speculative-decoding drafter. dspark is an EAGLE-style drafter that reuses
target-model features from several layers, drafts a block of tokens in parallel
with a diffusion-style masked-prediction loop, and carries two auxiliary heads.
It is not yet proven: at the time of writing the upstream author is around 600
training steps in and the measured accept rate does not yet beat MTP. The runtime
work below is gated on that result.

## Done in this branch

1. Multi-layer hidden-state tap. A reusable capture path that exposes the hidden
   state from an arbitrary set of intermediate decoder layers, concatenated per
   position into a single `[n_capture * n_embd]` row. This is the piece that both
   EAGLE3-proper and dspark need, and it is independent of any draft loop. See the
   API summary below.
2. dspark GGUF architecture and converter. The architecture enum, its tensor
   names, and a converter that maps an EasyDeL dspark export onto those names
   (`conversion/dspark.py`).
3. dspark forward graph and block-diffusion draft loop. The model build path
   (`src/models/dspark.cpp`) implements the feature-reuse projection and the
   masked block forward, and the `draft-dspark` speculative impl
   (`common/speculative.cpp`) runs the block-diffusion propose plus the
   sequential Markov resample (host BLAS/scalar, with an optional CUDA path).

## Open / deferred

The remaining question is not runtime engineering but the accept-rate gate below:
whether the trained drafter beats MTP at convergence. The `confidence_head`
adaptive commit-count policy is exported but its use for choosing how many drafted
tokens to keep is left to the driver.

## dspark drafter shape

- Feature reuse. The drafter consumes the target model's hidden states from a
  fixed set of layers, `target_layer_ids = (1, 9, 17, 25, 33)`. The five rows are
  concatenated to width `5 * target_hidden`, projected by `fc` down to
  `hidden_size`, then normalized by `hidden_norm` (RMSNorm).
- Trunk. A 5-layer transformer over the projected features: per layer RMSNorm,
  self-attention, RMSNorm, SwiGLU MLP. A final `norm` then `lm_head`.
- Aux heads. `markov_head` produces a low-rank additive bias on the logits.
  `confidence_head` predicts a per-position accept probability used to decide how
  many drafted tokens to commit.
- Block diffusion. `block_size = 7`. A block of `block_size` positions is seeded
  with `mask_token_id` and predicted in parallel (not autoregressively), optionally
  over a few refinement passes, then truncated at the first low-confidence position.

## Block-draft loop design (deferred)

Five-line summary:

1. Seed a block of `block_size` masked positions after `id_last`; build the draft
   batch from the concatenated multi-layer target features for the committed
   prefix plus learned mask embeddings for the masked tail.
2. Run the dspark trunk once over the whole block with an intra-block attention
   mask that lets masked positions attend to the prefix and to each other (full,
   non-causal within the block) so all positions are predicted in parallel.
3. Add the `markov_head` low-rank bias to `lm_head` logits, sample one token per
   masked position, and optionally re-mask the lowest-confidence positions and
   repeat for a small fixed number of refinement passes.
4. Use `confidence_head` to truncate the block at the first position whose
   predicted accept probability falls below a threshold, yielding the draft token
   run for this step.
5. On target verification, accept the longest matching prefix, then advance the
   reused target features by the number of accepted tokens (same carryover bookkeeping
   the MTP path already does) so the next block starts from the correct feature row.

### What it reuses from draft-eagle3 and draft-mtp

- The feature-reuse contract. dspark feeds the target's hidden state into the draft
  through the same staging path the MTP impl uses today: `need_embd_pre_norm()`
  drives the target context to emit the captured hidden, and the draft batch carries
  it in `batch.embd`. The only extension dspark needs over MTP is width: MTP carries
  one `n_embd` row, dspark carries the concatenated `n_capture * n_embd` row produced
  by the multi-layer tap built in this branch.
- The cross-batch carryover. The MTP impl already stashes the last target hidden row
  per sequence (`pending_h`) and, on accept, rewinds to the row matching the number of
  accepted tokens (`verify_h`, `accept()` selecting row `min(n_accepted, n_rows-1)`).
  dspark's accept logic is the same bookkeeping over a block instead of a single row.
- The plumbing in `common_speculative`: the per-sequence `draft_params`, the
  `process()` / `draft()` / `accept()` lifecycle, the priority chaining in
  `common_speculative_init`, and the stats counters. dspark is a new
  `common_speculative_impl` subclass alongside the existing ones; the harness does
  not change.

### markov and confidence head ops

- `markov_head`: a low-rank factor pair (down then up projection) producing a bias
  tensor of shape `[n_vocab, n_block_positions]`, added to the trunk logits before
  sampling. It is a small `ggml_mul_mat` then add; no new op kinds are needed.
- `confidence_head`: a small projection from the per-position trunk hidden to a
  scalar logit, sigmoid-activated to an accept probability per masked position. Also
  expressible with existing ggml ops.

## Gate

Build the block-diffusion `draft()` loop and the dspark forward graph only once
dspark beats the MTP accept rate at training convergence on the target model. Until
then this branch stops at the multi-layer tap (reusable, low risk) and the arch
scaffolding (registration only). The tap is useful on its own for EAGLE3-proper and
for any future multi-layer-feature drafter, independent of whether dspark ships.

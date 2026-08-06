# K-cache mean-centering

`--kv-mean-center` is an optional, opt-in feature that subtracts a fixed, precomputed
per-(kv-head, channel) bias from the K vector at the moment it is written into the KV cache,
in order to improve quantization fidelity for `GGML_TYPE_Q4_0` K caches.

This is currently scoped to `GGML_TYPE_Q4_0` only.

## The idea

`GGML_TYPE_Q4_0` is a symmetric (zero-point-free) block quantizer: each block is represented as
`value ~= scale * q`, where `q` is a signed low-bit integer and there is no bias/zero-point term.
If a given (kv-head, channel) position's real K activations have a nonzero mean across tokens,
symmetric quantization wastes some of its dynamic range encoding that constant bias, which
increases quantization error for that channel.

The fix: measure a per-(kv-head, channel) bias `k_bar` ahead of time (see
[Calibration](#calibration) below), and subtract it from the K vector for every token, right
before `Q4_0` quantization happens as part of writing it into the cache. Nothing else in
attention needs to change.

### Why this is safe (softmax-invariance)

For a fixed query row `q` attending over cached keys `k_0 ... k_n` (all in one layer/head), if
every cached key is centered by the same `k_bar` before being quantized and stored, the true dot
product decomposes as:

```
q . k_i = q . (k_i - k_bar) + q . k_bar = q . k_i_stored + q . k_bar
```

The `q . k_bar` term does not depend on `i` (the key's position) -- it is added identically to
every logit in that query's row. Softmax is invariant to a constant additive shift applied to
every logit in the same row (`softmax(x + c) == softmax(x)`), so the attention weights, and
therefore the rest of the model's output, are unaffected. This means the technique is exactly
correctness-preserving in infinite precision, and in practice the only observable difference is
ordinary floating point rounding (see `tests/test-kv-mean-center.cpp`, which checks this directly
against an unquantized F32 K cache). This is also why it is a zero decode-time-cost win: one
subtract at the point of cache write, nothing else changes.

The actual benefit is purely on quantization fidelity: centering the residual around zero before
`Q4_0`'s symmetric quantizer reduces per-channel quantization error for channels that have a real,
consistent activation bias. Quantifying that improvement on a production-scale model (e.g. via a
logit-KLD comparison against an uncentered `Q4_0` baseline) is a natural follow-up; this repo does
not ship a measured number for a specific trained model.

## Usage

1. Generate a bias file with `tools/kv-mean-center` (see its
   [README](../tools/kv-mean-center/README.md) for details):

   ```
   ./llama-kv-mean-center -m model.gguf -f calibration-data.txt -o kv-mean-center.gguf
   ```

2. Load it at inference time, together with a `Q4_0` K cache:

   ```
   ./llama-cli -m model.gguf -ctk q4_0 --kv-mean-center kv-mean-center.gguf -p "..."
   ```

`--kv-mean-center` requires `--cache-type-k q4_0`. If the K cache type is anything else, context
creation fails with a clear error rather than silently doing nothing, matching this codebase's
existing convention for other cache-type-gated options (e.g. quantized V cache requiring flash
attention).

## Bias file format

The bias file is a small GGUF file with one F32 1-D tensor per layer that has a bias, named
`kv_bar.blk.<il>.k`, holding `n_embd_head_k(il) * n_head_kv(il)` values laid out as
`[n_embd_head_k, n_head_kv]` (channel-fastest). This matches the in-memory layout of the K tensor
at the point it is written into the cache, so the file can be loaded directly as a small
broadcastable bias tensor per layer.

## Calibration

`tools/kv-mean-center` computes the bias by running a plain text calibration corpus through the
model and averaging the K tensor right before it would be written into the cache (via the
`k_cache_in` tag added to `llm_graph_context::build_attn()`, read through the same backend
scheduler eval-callback mechanism `llama-imatrix` uses to capture activations). See
[tools/kv-mean-center/README.md](../tools/kv-mean-center/README.md) for usage.

## Scope and limitations

- Only `GGML_TYPE_Q4_0` is supported; other K cache types are rejected. Generalizing the mechanism
  to other quantization types is future work.
- Every standard-attention KV cache layout is supported: the plain cache, the base/SWA pair of
  sliding-window models, and the attention sub-cache of hybrid (recurrent + attention) models,
  with or without SWA. Recurrent-only and MLA/DSA memory types are not.
- The calibration hook (`k_cache_in`) is currently only wired into the standard
  dense/GQA attention path (`llm_graph_context::build_attn(llm_graph_input_attn_kv *, ...)`),
  which covers the large majority of architectures. MLA and other specialized attention variants
  are not covered yet.
- The bias lives in the basis the calibration run's K cache used. With this fork's optional
  Hadamard K rotation (automatic for quantized K caches whose head dimension is a multiple of 64,
  unless `LLAMA_ATTN_ROT_DISABLE=1`), that means: calibrate with the same `--cache-type-k` and
  rotation settings you serve with, e.g. `-ctk q4_0` for the common case. Applying a bias in the
  wrong basis remains exactly safe for attention logits (the invariance argument above is
  basis-independent) but measurably degrades quantization quality instead of improving it, so the
  calibration tool records its basis in the file (`kv_mean_center.k_rot`) and the loader rejects
  a mismatch. In the matching basis the two features compose; see tools/kv-mean-center/README.md
  for measured numbers.

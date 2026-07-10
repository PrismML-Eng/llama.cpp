# llama.cpp/tools/kv-mean-center

Compute a per-layer, per-(kv-head, channel) K-cache mean-centering bias from a text calibration
corpus, for use with the `--kv-mean-center` flag (`GGML_TYPE_Q4_0` K cache only).

See [docs/kv-mean-center.md](../../docs/kv-mean-center.md) for the full description of the
feature. In short: `GGML_TYPE_Q4_0` is a symmetric quantizer, so a K channel with a real,
consistent nonzero mean across tokens wastes some of its dynamic range encoding that constant
bias. Subtracting a fixed per-(head, channel) bias before quantization removes that waste; because
the same bias is subtracted from every cached key, it shifts every attention logit in a query's row
by the same constant, which softmax is invariant to. Nothing else about attention changes.

This tool measures that bias by running calibration text through the model and averaging the K
values right before they would be written into the cache.

## Usage

```
./llama-kv-mean-center \
    -m model.gguf -f calibration-data.txt -o kv-mean-center.gguf \
    [-c 512] [--chunks N] [-ngl 99]
```

* `-m | --model` the model to calibrate against (mandatory).
* `-f | --file` a plain text calibration corpus (mandatory). A few hundred KB of representative
  text is enough to get a stable per-channel mean; the same kind of corpus used for `llama-imatrix`
  works well here too.
* `-o | --output-file` where to write the bias file (default: `kv-mean-center.gguf`).
* `-c | --ctx-size` chunk size in tokens (default: 512). Each chunk is scored independently with a
  cleared cache, matching `llama-imatrix`'s chunking.
* `--chunks` maximum number of chunks to process (default: all available).
* `-ngl | --n-gpu-layers` offload layers to GPU for faster calibration.

If you do not have a calibration corpus at hand, `make-calib-corpus.sh` generates one from the
model itself, so no external data file is needed:

```
./tools/kv-mean-center/make-calib-corpus.sh \
    -s ./llama-server -m model.gguf -o calib-corpus.txt
```

It starts a temporary `llama-server`, generates one response per built-in seed prompt (a mix of
expository, narrative, technical, code and dialogue prompts) with thinking disabled, and refuses
to write a corpus that looks degenerate (checked via gzip compression ratio). The per-channel K
mean is dominated by model-intrinsic channel structure rather than corpus content, so
self-generated text measures the same bias as an external corpus: in an A/B test, a self-generated
corpus and a standard multi-domain calibration set produced biases agreeing to within the
calibration sampling noise (cosine similarity above 0.95 on every layer).

The output is a small GGUF file with one F32 tensor per layer, named `kv_bar.blk.<il>.k`. Load it
at inference time with:

```
./llama-cli -m model.gguf -ctk q4_0 --kv-mean-center kv-mean-center.gguf -p "..."
```

`--kv-mean-center` requires `--cache-type-k q4_0`; loading fails with a clear error otherwise
(centering is currently only implemented/validated for `GGML_TYPE_Q4_0`).

Note: if the K cache also has this fork's optional Hadamard rotation feature active (automatic for
`GGML_TYPE_Q4_0` caches whose head dimension is a multiple of 64, unless `LLAMA_ATTN_ROT_DISABLE=1`),
the bias measured here is taken in the pre-rotation basis, since that is where the model's `Kcur`
tensor naturally sits before the rotation is applied. The centering subtraction still happens in
whatever basis `cpy_k()` sees (post-rotation, if active), which remains exactly safe (see
docs/kv-mean-center.md), but a mismatched basis means the calibrated bias is a less accurate
estimate of that channel's true post-rotation mean. Recalibrating directly against the
post-rotation representation is a natural follow-up.

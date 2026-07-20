# Phase 9: four-row VNNI64 decode tile

Phase 9 adds a separately opt-in single-token Q2_0 decode route:

```text
GGML_TBKERN_Q2_0_VNNI64_4R=1
```

The VNNI64 route previously rebuilt the identical 64-byte Q8 activation vector for each output row. The four-row tile constructs that vector once for each 64-value subgroup, then applies it to four independent packed-weight rows. Each output row retains the original `b=0,1` and group accumulation order, per-32 Q8 scales, and precomputed Q8-sum correction. A thread's final one to three rows use the unchanged Phase 8 route.

This is intentionally opt-in and does not change native Prism, Phase 7 VNNI, or Phase 8 VNNI64 selection. It requires AVX-512F and AVX-512VNNI. It changes no GGUF storage, repack allocation, or generic-ggml fallback behavior.

Validation gate: exact native/Phase 8/Phase 9 logits and token parity on the real 27B Q2_0 model, then matched CPU decode and PPL measurements. Retain it only if matched decode improvement exceeds run-to-run noise.
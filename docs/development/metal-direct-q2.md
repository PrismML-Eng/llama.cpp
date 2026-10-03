# Experimental Metal 4.1 direct Q2 operands

This path is disabled by default. Enable it before device initialization:

```sh
GGML_METAL_Q2_DIRECT_MAX=16 ./build/bin/llama-bench -m model.gguf -p 8,16 -n 0 -d 128 -r 3 -ngl 99 -fa 0
```

`GGML_METAL_Q2_DIRECT_MIN` defaults to 8. Unset `GGML_METAL_Q2_DIRECT_MAX`, or set it to zero, to retain existing dispatch. The path requires an embedded Metal source build, enabled TensorOps, an M5 device, macOS 27 or later, and a successful Metal 4.1 compiler probe. Unsupported configurations retain existing kernels. The experimental shaders are excluded when the gate is off.

The 16 x 32 x 16 tile decodes Q2_0/PQ2_0 blocks directly into half cooperative operands, preserving the F32/F16 activation type with relaxed TensorOps precision. Unlike the existing PQ2_0 few-row implementation, this implementation also handles Q2_0 and uses the generic matrix batch strides. It has no split-K reduction. Explicit fragment layout is validated only on M5.

## Current result

Do not select this as the default performance path. Against the current branch's existing PQ2_0 few-row kernel, three alternating paired model runs on M5 Pro measured median block times of 59.18 ms vs 95.79 ms at width 8, and 72.57 ms vs 114.53 ms at width 16 (existing vs experimental). Each run used three repetitions, prefix depth 128 and flash attention disabled. Earlier improvements against the older staged kernel do not establish an improvement over the current branch.

Focused Q2_0/PQ2_0 MUL_MAT backend checks pass with the gate off and on; unsupported reference combinations are skipped by the harness. Broader accuracy, current-branch full-model logits, other devices, and end-to-end speculative decoding remain release gates. This is an experimental comparison path, not a production recommendation.

## Follow-up experiments

Three changes were tested separately against matched current-default controls. Each passed the focused backend oracle before three alternating timing pairs; every timing run used three repetitions. Ratios below are candidate/default latency, so values above 1 are regressions.

| Candidate | Width 8 ratio | Width 16 ratio |
| --- | ---: | ---: |
| Four-way K split, original activation type | 1.331 | 1.345 |
| Four-way K split, half activation conversion | 1.531 | 1.555 |
| Four-way K split, original activation type, packed-word sharing | 1.511 | 1.462 |

K splitting reduced the candidate's latency, but none of these variants beat the current default. Half conversion and packed-word sharing were tested independently from the original-type four-way split. The follow-up variants were removed rather than added as more runtime knobs.

A separate compiler-language control kept the existing kernel selected in both arms: `GGML_METAL_Q2_DIRECT_MIN=17` with MAX=0 versus MAX=16. Current-kernel latency was 58.22/58.12 ms at width 8 and 71.36/71.28 ms at width 16 (Metal 4.0/4.1). This difference is within measurement variation and does not explain the candidate's regression.

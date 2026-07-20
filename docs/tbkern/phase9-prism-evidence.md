# Phase 9: Prism 27B minimal-PR evidence

This document records the evidence for the opt-in four-row CPU decode path.
It is a decode measurement, not a claim about prompt processing or GPU
throughput.

## Scope

The minimal branch is based on Prism's prism branch at 7529fdaaf. It keeps only
the CPU backend integration, the Q2_0 cache packer, the parity comparator, and
this evidence. The native Prism Q2_0 path remains the default unless the
selector is set:

GGML_TBKERN_Q2_0_VNNI64_4R=1

## Configuration

- Model: Ternary-Bonsai-27B-Q2_0.gguf, 7,165,121,600 bytes,
  SHA-256 868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757.
- Host: Google Cloud C3 highmem-8, Intel Xeon Platinum 8481C, 4 physical
  cores plus SMT siblings, one NUMA node, AVX-512F/VNNI/VBMI.
- Decode command: llama-bench -m MODEL -p 0 -n 16 -t 8 -ngl 0 -b 32 -ub 32.
- Baseline: unmodified Prism prism branch, same model, host, command, and
  eight ggml threads, with no TBKERN selector.

## Correctness

llama-debug with prompt Hello, deterministic generation, and the same runtime
settings produced bit-identical files relative to the native path:

- logits SHA-256:
  65d581397ae42288fe1115e7fa434589700cacd98edc67a750c0fba1765062f2
- generated-token SHA-256:
  7a7748eacf971049271242b9d921628019d6c44698574e9301da9b8c88026381
- NMSE: 0
- top-1 mismatches: 0
- one-chunk PPL: 1.0645 +/- 0.03006 for both routes

## Matched decode triplicate

The recorded three-run matched set used the exact command above:

| Route | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| Native Prism | 1.09 tok/s | 1.09 tok/s | 1.09 tok/s |
| Phase 9 four-row VNNI64 | 1.52 tok/s | 1.53 tok/s | 1.53 tok/s |

The Phase 9 route is approximately 40% faster for target decode on this host.
These are historical C3 measurements from the equivalent Phase 9 implementation
before this cleanup. This cleanup removes unused files and narrows the packing
header; it does not intentionally change the arithmetic or dispatch.

## Reproduction status

The current Windows workspace has no CMake, GCC, Clang, or llama-bench, so the
cleaned branch could not be rebuilt and rerun here. Before marking the PR ready
for review, rebuild this exact commit on the Linux C3 host and repeat native and
Phase 9 llama-bench three times, then rerun the deterministic parity check.

The route remains opt-in because the recorded perplexity workload was slower
even though the PPL value matched. Attention, norms, RoPE, softmax, GDN
recurrence/convolution, KV cache, sampling, scheduling, and GPU execution
remain in Prism's native runtime.
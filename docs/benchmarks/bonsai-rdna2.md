# Bonsai 27B on RX 6900 XT

## Installation

Prism fork baseline: `9a9394a895b96003ca842a6041cb28ac49a108f7` (`prism`).
Hardware: RX 6900 XT, gfx1030, 16368 MiB VRAM; Windows driver `32.0.21045.5002`.
Compiler: TheRock 10.0.0 / AMD clang 23, existing SDK from the reference installation.
The baseline build uses unchanged inference source, HIP, gfx1030, Release, and default HIP graph/MMQ options.

Model revision: `86e89f34c93201c3dfd5e5880fedb0022fc7e34d` from `prism-ml/Ternary-Bonsai-27B-gguf`.
`Ternary-Bonsai-27B-PQ2_0.gguf`: 7,165,121,600 bytes; SHA-256 `e4781999f1997ef97ce0c58d05750835acc999d18d83ee6489ba7ac7b14cb5f6`, verified against the published LFS hash.
Model architecture is `qwen35`, 64 blocks; tensor types are F32 and PQ2_0 (142). No MTP/nextn tensors were found.

HIP device mapping: `HIP_VISIBLE_DEVICES=1`, then `--device ROCm0`. Do not send gfx1030 kernels to the iGPU. `HSA_OVERRIDE_GFX_VERSION` is not supported on Windows.

Frozen original HIP DLL (`build-hip-original`): SHA-256 `7920de6a4b6f843da72f9f258c8e8e58aa006a99874fb267f796fb17e37bb6fc`.
Working-tree nwarps=8 rebuild (`build-hip-baseline`) is a different DLL and is not the untouched baseline.

## Launch

Target-only Pi default (port 8081, 32768 context, one slot):

```text
python scripts/bonsai/start-server.py --build build-hip-original --label serve-target
```

Pi provider `bonsai-local` / model `bonsai-27b` at `http://127.0.0.1:8081/v1`. Existing Qwen provider on 8080 is unchanged. Rollback copies: Pi `agent/backups/bonsai-20260919T033310Z/`.

## Untouched baseline

`scripts/bonsai/benchmark.py --label baseline-bench` against the frozen original HIP DLL.
Configuration: full GPU offload, ROCm0 after physical ordinal 1 filtering, FA on, batch 2048, ubatch 512, threads 8, f16 K/V cache, three repetitions after warmup.

| Workload | Mean tokens/s | Sample standard deviation |
|---|---:|---:|
| pp512 | 389.012 | 2.366 |
| tg128 | 51.077 | 0.16 |

API smokes on the same binary: deterministic short chat matched the requested phrase. A binary-search coding prompt produced 220 tokens at 50.43 tokens/s (one sample). Repeated API code/math/chat at temperature 0, seed 42, thinking off: decode about 50.6–50.8 tokens/s across nine runs (`results/api-baseline.json`).

Pi `-p` (`results/pi-baseline.json`): provider `bonsai-local`, model `bonsai-27b`, three ordinary and three controlled repeats of a short reply (`BONSAI_OK`) and a read-tool probe (`Bonsai Pi read tool check: 7C4A9E`). All passed.

## Runtime sweep

A first t/ub sweep was discarded (`results/runtime-sweep/INVALID.txt`): the baseline server was still resident. Use `results/runtime-sweep-clean/` only. Same original HIP DLL as the baseline.

| threads | ubatch | pp512 | tg128 |
|---:|---:|---:|---:|
| 4 | 256 | 382.49 | 51.16 |
| 8 | 256 | 381.86 | 51.15 |
| 4 | 1024 | 390.04 | 51.10 |
| 8 | 1024 | 388.84 | 50.96 |

No setting beat the baseline by 5%. Keep serving at threads 8, ubatch 512.

## Reference findings

The user-provided reference contains an uncommitted H-R9 change in `ggml/src/ggml-cuda/mmvq.cu` on upstream revision `7ceed87`.
It selects eight warps for a whitelist of RDNA2 quantized formats and changes launch bounds. PQ2_0 is absent from that whitelist, so the reference's results do not establish a Bonsai benefit.
Its prior experiment ledger retracts apparent force-CUBLAS gains caused by competing VRAM use. Each experiment here must run with only one model process active. Same flags on this desktop have moved about 37% across sessions; pair and interleave before ranking. Effects under 5% are inconclusive.

Rejected without retry: FORCE_CUBLAS, FORCE_MMQ, HIP graphs off, MMQ_MFMA off, unsafe-math, nwarps=16, FA-off.

## PQ2_0 nwarps=8

Working-tree `calc_nwarps` returned 8 for RDNA2 PQ2_0 with `ncols_dst == 1`. Compared that binary to `build-hip-original` with interleaved llama-bench (`results/ab-original-i{1,2,3}` vs `results/ab-8warps-i{1,2,3}`). `test-backend-ops -o MUL_MAT -p pq2_0` on the patched binary: 49/49 passed.

| Pair | original tg128 | patched tg128 | tg delta | original pp512 | patched pp512 |
|---:|---:|---:|---:|---:|---:|
| 1 | 51.50 | 49.35 | −4.18% | 386.68 | 390.31 |
| 2 | 51.15 | 49.02 | −4.16% | 386.65 | 388.31 |
| 3 | 50.48 | 49.06 | −2.81% | 384.40 | 386.44 |

Mean paired tg128: **−3.72%**. Dropped. The override was reverted from `ggml/src/ggml-cuda/mmvq.cu`. Prefill was slightly up and is informational. `__launch_bounds__(…, 2)` was not applied. Extra PQ2_0 `test-backend-ops` shapes were kept.

## Multi-token acceleration

Native MTP is infeasible from these weights. Qwen3.5 MTP requires `n_layer_nextn == 1` and `nextn.eh_proj` / `enorm` / `hnorm`. This GGUF has 64 trunk blocks and no nextn tensors. Do not use `--spec-type draft-mtp`.

The published DSpark drafter is the supported speculative path (`--spec-type draft-dspark --spec-draft-n-max 4`). Converted files:

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `Ternary-Bonsai-27B-dspark-dflash-bf16.gguf` | 2,217,233,120 | `72de8e296081ee5545c78e08852d4cf89eb34a1db86a094f70741a2bcd0753d2` |
| `Ternary-Bonsai-27B-dspark-dflash-Q4_0.gguf` | 631,712,480 | `8138ddef5ee20d460b86c8ff982782a651bdc52d9e998d3c1f038b8c5669a60a` |

Architecture `dflash`, block size 4, shared embedding/lm-head dropped. The BF16 sidecar cannot co-reside on 16 GB with the target. Speculative serving disables cross-request prompt cache.

HIP load succeeded at 16384 and 32768 on this 16 GB card with the Q4_0 sidecar. Small API prompts (temperature 0, seed 42, thinking off, `cache_prompt` false):

| Context | Task | predicted tok/s | draft_n | accepted |
|---:|---|---:|---:|---:|
| 32768 | code | 110.96 | 200 | 169 |
| 32768 | math | 108.03 | 100 | 80 |
| 32768 | chat | 67.70 | 324 | 132 |

Code decode is about **2.18×** the 50.57 tok/s target-only baseline. Ordinary Pi `-p` against that same 32768 DSpark server failed all six short and read-tool runs (empty responses, `stopReason` error, ~21 s walls). Pi default therefore stays **target-only**. DSpark is documented opt-in for small-prompt decode, not the coding-agent default.

Opt-in launch:

```text
python scripts/bonsai/start-server.py --build build-hip-original --context 32768 --draft ..\models\Ternary-Bonsai-27B-dspark-dflash-Q4_0.gguf --label serve-dspark
```

## Rollback

- Server: `python scripts/bonsai/start-server.py --build build-hip-original`
- Pi: restore `agent/backups/bonsai-20260919T033310Z/` if the dedicated provider must be removed

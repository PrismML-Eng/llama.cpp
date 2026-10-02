# Bonsai 2 27B on AMD Radeon 890M — Full Speed Recipe (14 t/s)

**Result: 27B dense-class hybrid model generating at 11.5–12 t/s through HTTP (12.4–14.1 t/s server-side), spec-decode acceptance 0.84, 5/5 quality gate — on an integrated GPU (AMD Radeon 890M, Ryzen AI 9 HX 470, 28 GB shared LPDDR5X).**

This document is the complete, reproducible recipe: build configuration, launch flags, environment variables, and the hard-won findings from a 3-day optimization campaign. Stock llama.cpp Vulkan runs the same model at 1.85 t/s — **~7.5× faster with these changes**.

---

## 1. Hardware / software baseline

| Component | Value |
|---|---|
| APU | AMD Ryzen AI 9 HX 470 (Radeon 890M iGPU, RDNA 3.5) |
| RAM | 28 GB LPDDR5X-7500 (unified memory — the bandwidth wall) |
| Driver | AMD Adrenalin 26.9, Vulkan 1.4.349 |
| OS | Windows 11 |
| Toolchain | MinGW-w64 (winlibs GCC 16.2), CMake + MinGW Makefiles |

## 2. Build configuration

```bash
cmake -S . -B build-dspark -G "MinGW Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_C_COMPILER=<winlibs>/gcc.exe \
  -DCMAKE_CXX_COMPILER=<winlibs>/g++.exe

mingw32-make -C build-dspark -j8 llama-server
```

⚠️ **Always wipe the build directory after any stash/revert/branch-switch.** Incremental rebuilds after tree changes produce stale-object mazes that silently build old code ("0 errors" that hides 10 real errors).

⚠️ **Kill any running `llama-server.exe` BEFORE relinking** — on Windows the linker fails with "Permission denied" if the exe is loaded, and you may silently benchmark a stale binary.

## 3. The model: stock GGUFs will NOT reproduce this (graft required)

The speed comes from `--spec-type draft-mtp` — but **PrismML ships all Bonsai 2 GGUFs WITHOUT the MTP head** (layers 0–63 only, nextn stripped). A stock file cannot run speculative decoding. The champion file (`Bonsai-2-27B-Q2_0-fork-MTP.gguf`) is:

1. **Base**: Q2_0-fork GGUF from [`prism-ml/Ternary-Bonsai-2-27B-gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
2. **Grafted MTP head**: the 15 `blk.64.*` (nextn) tensors range-requested from unsloth's `Qwen3.8-27B` GGUF (~340 MB, not the full 17 GB) via the community `graft_mtp.py`; metadata `block_count 64→65`, `nextn_predict_layers=1`. Verbatim copy is valid because Bonsai's RMSNorm weights match stock Qwen3.8 elementwise (cos > 0.99996) — the residual stream stays in the original basis.
3. **Hadamard-inverse patch** in the MTP graph (shipped in PR #187 commit `4b8092c3`): the draft graph's embedding lookup must apply the inverse Hadamard rotation (rotation first, sign flip second) or the server dies at context init.

⚠️ **Build from the PR #187 branch, NOT upstream main** — until it merges, upstream lacks the bf16 state pools, the fused GDN rows-mode kernel, the Q2fork matvec, and the Hadamard fix. A stock build either refuses the Q2fork format or runs it on CPU.

## 4. Production launch flags

```bash
LLAMA_SSM_BF16_STATE=1 ./llama-server \
  -m Bonsai-2-27B-Q2_0-fork-MTP.gguf \
  --spec-type draft-mtp --spec-draft-n-max 1 \
  -ngl 99 -ngld 99 -fa on \
  -t 24 -td 8 -c 16384 -np 1 \
  --jinja --host 0.0.0.0 --port 8080
```

| Flag | Why |
|---|---|
| `LLAMA_SSM_BF16_STATE=1` | BF16 SSM state pools (+22% decode). **Required** — the F32-state path has a selection hole that can crash. |
| `--spec-type draft-mtp` | MTP self-speculation using the model's native draft head |
| `--spec-draft-n-max 1` | n=1 is optimal on iGPU. n=2 collapses acceptance (0.84 → 0.44) and loses 15% speed. The draft chain costs more than it earns at shared-memory bandwidth. |
| `-ngl 99 -ngld 99` | Full offload of target AND draft. Everything must fit in VRAM — spilling to CPU caps you at 2–3 t/s. |
| `-fa on` | Flash attention |
| `-t 24 -td 8` | Target threads = all 24; draft threads = 8 (16 oversubscribes against the target). Try both head placements on new hardware: GPU head (`-ngld 99`) won here; on machines with a stronger CPU:GPU ratio, CPU head (`-ngld 0 -td 8`) can win by overlapping. |
| `-np 8` (batch mode) | For multi-user serving: ~14.3 t/s aggregate across 8 streams |

## 5. Measured results

| Configuration | Speed | Notes |
|---|---|---|
| Stock llama.cpp Vulkan | 1.85 t/s | baseline |
| + BF16 SSM state pools | +22% | commit `53608747` |
| + fused GDN rows-mode matvec | +30% cumulative | commit `96062463` |
| + PTQ1_0/Q2fork matvec v2 (4-row workgroups) | closed | commit `bb5997c5` |
| + MTP speculation n=1, acc 0.84 | **11.5–12 t/s HTTP / 14.1 peak** | production |
| n=2 speculation | 9.9 t/s | ❌ regression |
| KV cache q8_0 | no gain | KV is a small share |
| 32 threads | no gain | threads saturated at 24 |
| Leaner quant (IQ3, −21% bytes) | −27% speed | dequant-bound, not bandwidth-bound |

Effective bandwidth: ~180 GB/s sustained on LPDDR5X — this is the hardware wall for a weights-streaming workload on unified memory.

## 6. Findings that will save you days

1. **Raw-gates fusion is CPU/Metal/CUDA-only upstream — keep it that way on Vulkan.** The `qwen35.cpp` device allowlist deliberately excludes Vulkan from the raw-gates path. Enabling it produces corrupted output (`////`) in every shader variant (subgroup, nocluster, shmem). We verified this is a real incompatibility, not a stale allowlist.

2. **`GGML_SSM_BF16_STATE` is load-bearing.** Without it, F32-state + rows-mode hits a pipeline-selection gap (returns nullptr → CPU fallback or worse).

3. **Leaner quant ≠ faster on iGPU.** PTQ1_0 (5.9 GB) measured 27% *slower* than Q2fork (7.4 GB): iGPU decode is dequant-compute-bound, not bandwidth-bound, until you hit specialized kernels. The fork's custom 2-bit kernels beat generic 3-bit kernels despite moving more bytes.

4. **Test spec-mode quality with an exact-answer gate** (e.g. `17×23=391`, translation, sequence completion) before benchmarking speed. Speculative decoding bugs present as output corruption, not crashes.

5. **Debugging order that actually works:** (a) prove the GPU healthy with a known-good binary, (b) binary-search the *config* (env, flags) before the code, (c) diff allowlists/selection predicates between trees — the bug is usually a path-selection mismatch, not shader math.

6. **VRAM wall rule:** model must fit fully in VRAM. A 17 GB model on 14.4 GB usable VRAM = 2.5 t/s (CPU spillover); the same class of model at 10 GB = 5.9 t/s; specialized 7.4 GB = 14 t/s.

## 7. Related commits (PR #187)

- `905c29b6` PTQ1_0 decode: trit-table + vectorized dequantize4, MUL+FWHT fusion
- `a078ce98` review feedback
- `63854022` dedicated PTQ1_0 matvec kernel, coopmat A-load, MTP hadamard inverse
- `bb5997c5` PTQ1_0 matvec v2 (4-row workgroups)
- `53608747` bf16 SSM state pools
- `96062463` GDN rows-mode + bf16 state read (fused)
- `6e17c24d` build fix + tail-guard hoist

## 8. Reproducibility notes

- Numbers are single-stream, temperature 0.1, 300-token generations, measured server-side (`eval time`) and cross-checked over HTTP.
- Quality gate: 5/5 exact-answer checks (arithmetic, knowledge, sequence, translation, word problem) — spec-decode must not degrade answers.
- `draft acceptance` logged by llama-server should read 0.55–0.85 at n=1. If it reads 0.0 or 1.0 persistently, suspect a draft-graph misconfiguration.

*Hardware: AMD Ryzen AI 9 HX 470 / Radeon 890M. Your mileage on other iGPUs will scale with memory bandwidth.*

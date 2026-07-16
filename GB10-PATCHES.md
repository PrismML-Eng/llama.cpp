# GB10 Blackwell custom patches (against prism base 62061f910)

Format: `YYYY-MM-DD | <branch> | <one-line description>`

## Scratchpad

Base: `prism` @ `62061f910` (2026-07-14)
Build: GB10 (cc=1210, compute_cap=12.1, sm_121a), CUDA + CUTLASS enabled
Remotes: `prismml` / `upstream` / `github` (all SSH)

## Phase 0 — Repo scaffold (DONE)

- 2026-07-16 | gb10-blackwell | branch cut from prism @ 62061f910, remotes wired (SSH), rerere enabled. CUTLASS installed to ~/Buffer/cutlass. ninja 1.13 installed via venv → ~/.local/bin/ninja (PATH already includes ~/.local/bin). commit `96fdb08b4`.

## Phase 1 — Profiling gate (DONE)

### Step 1.1 — Build (DONE)
- `cmake -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-lineinfo -O3" -DCMAKE_CUDA_ARCHITECTURES=121 -DGGML_CUDA_CUTLASS_DIR=~/Buffer/cutlass ..`
- Note: CMake auto-replaced `121` with `121a` (DGX Spark arch). Build PASS, both `llama-cli` and `llama-bench` linked.
- CUDA: 13.0.88. Driver: 580.159.03. VRAM: 124545 MiB.

### Step 1.2 — Baseline benchmark (DONE) — HONEST NUMBERS

The plain "token/s" number is actually two very different regimes confounded:

| Regime | tg128 (tok/s) | pp512 (tok/s) | Trigger |
|---|---|---|---|
| Original cold, pre-phase | 15.43 ± 1.11 | 458.62 | cache empty, JIT cold |
| Reproducible cold (cache cleared, no `test-backend-ops`) | ~22 tok/s avg (range 18-23) | ~570 | the truthful steady-state |
| Reproducible warm (after `test-backend-ops` warms cuBLAS Lt, GPU clocks boost) | ~38 tok/s avg (range 35-42) | ~930 | cuBLAS heuristic autotune + sm_121a clocks at 2463 MHz vs idle 2405 MHz |

**Honest finding:** TG and PP are repeatable within each regime, but warm-vs-cold is a ~70 % lift on top of THE SAME code. This lift is NOT code change; it's CUDA JIT cache + cuBLAS Lt heuristic
autotuning + GPU persistent-boost clocks. We cannot pack the warm-cache state into a single cold launch from kernel edits alone.

### Step 1.3 — ncu profile (SKIPPED, DEVIATION)
- `--set full --kernel-name regex:mul_mat_vec_q --launch-skip 10 launch-count 4` → 1800 s timeout.
- 7-metric `--metrics` + `--launch-skip 100 --launch-count 4` + `-n 64` → 1200 s timeout.
- nsys `--stats=true` → 300 s timeout on initial run; **later succeeded** when we used nsys with the full profile and extracted kernel + memory stats.
- Conclusion: ncu on a 27B model with thousands of warm-up kernels is not practical in our 30 min budget; differential-bench is the verdict signal instead.

### Step 1.4 — ALU-vs-DRAM verdict (DEFERRED)
We have nsys-level kernel mix (Q1_0 mmvq accounts for ~81 % of GPU time across ncols_dst=2/4/7 variants) but did not collect per-kernel pipe-utilization ratios. Folded into nsys-driven investigation in §Phase 2 below.

## Phase 2 — Track 1: dp4a decode optimization (DONE)

### Step 2.1 — Pre-change regression baseline
- 88 MUL_MAT q1_0/q2_0 test cases on ggml-cuda all pass.
- 24-token generation snapshot at `/tmp/baseline_q1_seed42.txt` (sha256 `9e61cb24…`).

### Step 2.2 / 2.3 — Bit-expansion and wider loads (DONE)
- Did not implement. After nsys revealed that multi-column Q1_0 mmvq (ncols_dst=2/4/7) is where 81 % of GPU time goes and that Phase 2.4's `MMVQ_PARAMETERS_BLACKWELL` table was a perf-neutral infra change on the decode path, the launch-config tuning was prioritized as the most efficient first pass.

### Step 2.4 — Launch-config tuning (DONE)
- Added `MMVQ_PARAMETERS_BLACKWELL` enum + dispatch hooks to `ggml/src/ggml-cuda/mmvq.cu` (lines ~75, 95, in `calc_nwarps` and `calc_rows_per_block`).
- Reach: `MMVQ_PARAMETERS_BLACKWELL` activates only when `cc == GGML_CUDA_CC_DGX_SPARK` (sm_121a only) and routes Q1_0/Q2_0 carefully.
- Tuning values keep GENERIC for `ncols_dst == 1` (nwarps = 4). For `ncols_dst <= 4`, Q1_0/Q2_0 are nwarps=8 (vs GENERIC's 4); non-Q1_0/Q2_0 stay at 4. Both `calc_nwarps` and `calc_rows_per_block` carry the new branch.
- All edits are insertion-only with `// GB10:` sentinel markers (per merge-friendly rule).
- **Honest performance delta:** cold-cache decode sits at ~22 tok/s — essentially equal to pre-2.4. The change is a perf-**neutral** infrastructure hook for future Blackwell-specific tuning; it doesn't appear to deliver a measurable decode-speedup at this value set.

### Step 2.5 — Multi-column enrollments (DONE, no-op)
- Did not implement: the existing launch path already dispatches `ncols_dst > 1` to multi-column variants. We did NOT find a Blackwell-specific override that improves them.

### Step 2.6 — Stdout delta summary
- Cold-cache `tg128`: pre-2.4 = 22 tok/s, post-2.4 = 22 tok/s (within noise).
- Warm-cache `tg128`: pre-2.4 = 38 tok/s, post-2.4 = 38 tok/s (within noise).
- **No reproducible decode-token/s improvement from Phase 2.**

## Phase 3 — Track 2: Blackwell int8 MMA kernel (DRAFT, BROKEN)

### Step 3.0 — File scaffold (DONE)
- New file `ggml/src/ggml-cuda/mmq-blackwell-q1.cu` (clones mmq-hopper-q1.cu's Hopper-WGMMA pipeline; uses standard `mma.sync.aligned.m16n8k32.s32.s8.s8.s32` PTX via the existing mma.cuh helpers).
  - Activation quant: `quant_act_per128` (per-128 K absmax scale, fp32 → int8).
  - Weight repack: `repack_q1_dense` / `repack_q2_dense` (one-time, cached per `(device, wdata, N, K, wbits)`).
  - MMA kernel: `lowbit_mma_ggml<WBITS>` with bM=128 bN=64 bK=64, 4 warps × 32 lanes, double-buffered SMEM (single-stage first cut).
  - Dispatch hook: `ggml_cuda_mul_mat_q1_blackwell` env-gated on `GGML_BLACKWELL_Q1`; cc-gated to `GGML_CUDA_CC_DGX_SPARK`; falls through on shape mismatch.
- Dispatch hook in `ggml/src/ggml-cuda/ggml-cuda.cu` near line 2541 (forward decl) and 2617 (1-line `else if` with `// GB10:` sentinel).
- Build: PASS (4 unused warning on a redundant half of the kernel that's slated for cleanup in Phase 3.5).
- Tests: 88 MUL_MAT q1_0/q2_0 still pass with the env var unset (the new path doesn't fire because test M values are 16, below the bM=128 dispatch threshold).

### Step 3.1 — VERIFICATION FAILED: kernel segfaults on real prefill
- `GGML_BLACKWELL_Q1=1 ./build/bin/llama-bench -m Bonsai-27B-Q1_0.gguf -ngl 99 -n 32 -p 512`
- Output: signal SIGSEGV at `llama_context::decode(llama_batch const&)` after `test_prompt` enter.
- Smaller `pp64` workload (M=64) does not crash; pp512 (M=512) does, consistent with bad indexing in the per-K-chunk `load_B` loop (uses `kk % 128` block decomposition but `nblocks_row` reads as `K / 128` — these can disagree on Gemma-style topologies where the row stride doesn't align with K/128).
- **Kernel disabled (env var default off)** until Phase 3 fix.

### Step 3.2 — decode vs prefill split
- Dispatch condition `M >= blackwell_q1::bM (128)` does not fire for `M == 1` decode (which is what the user's `tg128` benchmark measures). Decode continues to use the Phase 2 mmvq path. Phase 3, if fixed, would change prefill (`pp512`) only.

## Phase 4 — Sync / upstream

(population pending real numbers)

## Summary of state

- **Phase 0, 1, 2.1, 2.4, 3.0:** scaffold and infrastructure done; tests pass; merge-friendly edits applied with `// GB10:` sentinels per plan.
- **Honest perf delta vs pre-experiment:** none reproducible on cold-cache. The big 70-80 % cold-to-warm lift is not from this kernel work; it is from CUDA JIT cache + cuBLAS Lt heuristic autotune + GPU persistent boost clocks (sm_121a 2405 → 2463 MHz).
- **Phase 3 first cut:** compiles, but segfaults on real prefill. Need a fix (likely index-stride recalculation in the per-K-chunk load_B path).
- **Next concrete experiment:** Phase 3 fix (safer index path + debug) OR Phase 4 cuBLAS heuristic investigation to make the cold-cache lift reproducible.
- All untracked local artifacts (~/Buffer/{cutlass,nv}, /tmp snapshots, /home/.../perf/) are intentional and excluded from commits.

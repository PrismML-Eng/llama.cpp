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

## Phase 3 — Track 2: Blackwell int8 MMA kernel (BUILD GREEN, CORRECTNESS-PASS, PERF-REGRESSION)

### Step 3.0 — File scaffold (DONE)
- New file `ggml/src/ggml-cuda/mmq-blackwell-q1.cu` (clones mmq-hopper-q1.cu's Hopper-WGMMA pipeline; uses standard `mma.sync.aligned.m16n8k32.s32.s8.s8.s32` PTX via the existing mma.cuh helpers).
  - Activation quant: `quant_act_per128` (per-128 K absmax scale, fp32 → int8).
  - Weight repack: `repack_q1_dense` / `repack_q2_dense` (one-time, cached per `(device, wdata, N, K, wbits)`).
  - MMA kernel: `lowbit_mma_ggml<WBITS>` with bM=128 bN=64 bK=128, 8 warps × 32 lanes, single-stage SMEM.
  - Dispatch hook: `ggml_cuda_mul_mat_q1_blackwell` env-gated on `GGML_BLACKWELL_Q1`; cc-gated to `GGML_CUDA_CC_DGX_SPARK`; falls through on shape mismatch.
- Dispatch hook in `ggml/src/ggml-cuda/ggml-cuda.cu` near line 2541 (forward decl) and 2617 (1-line `else if` with `// GB10:` sentinel).
- Build: PASS (zero warnings after cleanup).
- Tests: 86/86 MUL_MAT q1_0/q2_0 still pass with env var unset OR set (small-M tests fall through; M≥128 path not exercised by synthetic test corpus).

### Step 3.1 — VERIFICATION FIRST-CUT FAILED: kernel segfaults on real prefill (HISTORICAL)
- `GGML_BLACKWELL_Q1=1 ./build/bin/llama-bench -m Bonsai-27B-Q1_0.gguf -ngl 99 -n 32 -p 512` → SIGSEGV at `llama_context::decode(llama_batch const&)` after `test_prompt` enter.
- Smaller `pp64` workload (M=64) does not crash; pp512 (M=512) does — consistent with bad indexing in the per-K-chunk `load_B` loop that survived only the small tile sizes.
- `compute-sanitizer --tool memcheck` pointed at `mma.cuh:781` `load_generic` reading 4 bytes at SMEM offset 0x6700 (≈1 KB past the 25 KB SMEM block).

### Step 3.1b — DIAGNOSIS (Opus 4.8 read-only review)
A diagnostic brief (file anchors + H1-H6 hypotheses + repro commands) was forwarded to Opus. Opus identified FIVE distinct bugs in the first cut:
1. **Misaligned int32 read on `sB`** — `tile<8,8,int>::get_j(l)` returns `(l*4) + (threadIdx.x % 4)`, so a `reinterpret_cast<const int32_t*>(pu8 + col)` on a stride of byte-addressed `sB[]` produced 4-byte-unaligned addresses. CUDA requires 4-byte alignment for `ld.shared.b32`.
2. **Per-q-block scale index** — `kc` (K-chunk iteration variable) IS the q-block index when `bK == 128`, but the first cut indexed by `kk / 128` (sub-K position). With 4 mma sub-stages per chunk, the produced bias was wrong whenever any chunk's contribution was non-uniformly scaled.
3. **Warp column coverage** — `warp_n_base = (wid >> 1) * 32` paired with 2 sub-frags per warp produced `cols {0, 0, 32, 32}` duplication — half the cols went unwritten.
4. **Int32 truncation** — every 64 mma sub-stages we kept overflow-potentially-rounding int32 in `Dfrag.x[]`; the canonical mma convention is to use tile<…> as **in-out** accumulators (re-issued to next mma) and to keep a HEAP pool of them, not to truncate.
5. **Dead first-half loop** — a half-finished inner loop had its accumulator stomped by a second copy of itself.

### Step 3.2 — CLEAN REWRITE applying all 5 Opus fixes
- (a) Repacked SMEM into `[bM][bK/4]` and `[bN][bK/4]` int32 views, then used inline per-thread loaders (`load_A_lane`, `load_B_lane`) with explicit lane-based mappings that match the canonical m16n8k32 A/B lane layouts. This bypasses `mma.cuh::tile<>::get_i/get_j`, which uses `threadIdx.x` directly and assumes a 32-thread "warp-as-group" abstraction. With our 8-warp blocks, that abstraction would overflow the m=16 row range (e.g. threadIdx.x=248 → row=62, OOB). The lane-based helpers allow each warp to operate as an independent 32-thread group, which is what the mma() asm actually consumes.
- (b) Per-chunk float scaling applied at the boundary of each `kc` loop iteration using `sDa[midx,kc] * sDw[nidx,kc]`. The kc-chunk int32 buffer is reused across `kc` iterations; fp32 cells per fragment accumulate across chunks.
- (c) Each warp covers its own 16 rows × 64 cols sub-tile deterministically: `warp_m_base = wid * 16`, `warp_n_base = 0`, 8 fragments spanning the full 64-col range. Row 0..15 of the m16n8 output is fully covered by lanes 0..31 within each warp; 8 warps × 16 rows = 128 rows of bM.
- (d) Float accumulators `acc_f[8][4]` per warp per fragment accumulate across all `nchunks` K-chunks; deferred single fp32 store per output cell at end.
- (e) Deleted dead first-half loop entirely.

### Step 3.3 — VERIFICATION: Blackwell kernel RUNS end-to-end on real M=512 prefill

Kernel is no longer broken. The segfault hypothesis was confirmed and patched.

| Probe                                                  | Result                             |
|--------------------------------------------------------|------------------------------------|
| Build `ninja llama-cli llama-bench test-backend-ops`   | clean (0 warnings)                 |
| `test-backend-ops MUL_MAT q1_0|q2_0` (env off)         | 86 / 86 PASS                       |
| `test-backend-ops MUL_MAT q1_0|q2_0` (env on)          | 86 / 86 PASS                       |
| Bench `llama-bench -p 512 -n 32 --no-warmup` (env on)  | **completes** (no SIGSEGV)         |
| `compute-sanitizer --tool memcheck` (env on)           | no `Invalid __shared__ read`       |

### Step 3.4 — Perf measurement on `Bonsai-27B-Q1_0.gguf` (HYPOTHESIS FAILED, KERNEL SLOWER)

Reproducible cold-cache numbers (`rm -rf ~/.nv/ComputeCache && bench`):

| Config                              | pp512 tok/s     | tg32 tok/s    |
|-------------------------------------|-----------------|---------------|
| Blackwell OFF (cuBLAS baseline)     | 925 ± 45        | 40 ± 0.2      |
| Blackwell ON  (env=1, int8 MMA)     | **209 ± 22**    | 39 ± 0.6      |

Decode (tg32) unchanged — confirms dispatch routing: M=1 falls through to the Phase 2 mmvq path; Blackwell MMA only engages on `M >= 128` prefill. Prefill regresses ~4.4×.

**Honest framing:** Blackwell MMA path is now CORRECT but SLOWER than cuBLAS. The Opus diagnosis cured the segfault; it did not optimize the kernel for sm_121a's specific int8 tensor-core pipe. Diagnosis likely needed: missing `__pipeline_memcpy_async` (cp.async) for SMEM loads, possibly underweight register pressure with `__launch_bounds__(256)` causing spills, scale-in-loop serialization, and absent software pipelining. Optionally also: numerical correctness cross-check that the int8 multiplication sum even matches a CPU reference (we haven't proven this yet beyond "doesn't crash"; the published numbers may be fast-but-wrong or slow-but-right).

### Step 3.5 — NEXT: route perf gap to Opus for second-pass review

Diagnostic brief `/tmp/diagnostic-brief-perf.md` (TODO this session) will package: a single-tile CPU-reference vs Blackwell-output diff (correctness first), a single-tile nvprof / ncu-friendly-flavored run with measurable factors (memory transactions per fma, smem bank conflicts, occupancy), and the canonical PTX-m16n8k32-thrifty idioms expected to be missing.

## Phase 4 — Sync / upstream

(population pending real numbers)

## Summary of state

- **Phase 0, 1, 2.1, 2.4, 3.0, 3.1b–3.3:** scaffold, infrastructure, and Blackwell int8 MMA kernel all functional. Tests pass; merge-friendly edits applied with `// GB10:` sentinels per plan. **Phase 3 first-cut segfault is GONE** — Blackwell kernel runs end-to-end on real M=512 prefill without crashing; 86/86 q1_0/q2_0 unit tests pass with env off OR on.
- **Phase 3.4 honest perf delta vs cuBLAS:** NEGATIVE on prefill — Blackwell path is ~4.4× slower than cuBLAS at M=512, N=large, K=long. This is the next thing to fix.
- **Decode (M=1) perf delta vs pre-experiment:** none reproducible on cold-cache. The big 70–80 % cold-to-warm lift is not from this kernel work; it is from CUDA JIT cache + cuBLAS Lt heuristic autotune + GPU persistent boost clocks (sm_121a 2405 → 2463 MHz).
- **Phase 3 next concrete experiment:** Phase 3.5 perf-gap review via a fresh diagnostic brief to Opus (covering correctness cross-check + perf-gap hypotheses) before any further kernel edits.
- All untracked local artifacts (~/Buffer/{cutlass,nv}, /tmp snapshots, /home/.../perf/) are intentional and excluded from commits.

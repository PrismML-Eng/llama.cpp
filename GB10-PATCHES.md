# GB10 Blackwell custom patches (against prism base 62061f910)

Format: `YYYY-MM-DD | <branch> | <one-line description>`

## Scratchpad

Base: `prism` @ `62061f910` (2026-07-14)
Build: GB10 (cc=1210, compute_cap=12.1, sm_121a-replaced-by-toolchain), CUDA + CUTLASS enabled
Remotes:
- `prismml`  — PrismML-Eng/llama.cpp (read + write)
- `upstream` — ggerganov/llama.cpp (NEVER push)
- `github`   — sumergoconicio/bonsai-llama.cpp (read + write, SSH)

## Phase 0 — Repo scaffold (DONE)

- 2026-07-16 | gb10-blackwell | branch cut from prism @ 62061f910, remotes wired (SSH), rerere enabled. CUTLASS installed to ~/Buffer/cutlass. ninja 1.13 installed via venv → ~/.local/bin/ninja (PATH already includes ~/.local/bin). commit `96fdb08b4`.

## Phase 1 — Profiling gate (IN PROGRESS)

### Step 1.1 — Build (DONE)
- `cmake -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-lineinfo -O3" -DCMAKE_CUDA_ARCHITECTURES=121 -DGGML_CUDA_CUTLASS_DIR=~/Buffer/cutlass ..`
- Note: CMake auto-replaced `121` with `121a` (DGX Spark arch). Build PASS, both `llama-cli` and `llama-bench` linked.
- CUDA: 13.0.88. Driver: 580.159.03. VRAM: 124545 MiB.

### Step 1.2 — Baseline benchmark (DONE)
- `llama-bench -m Bonsai-27B-Q1_0.gguf -ngl 99 -n 128 -p 512 -o md` (commit `96fdb08b4`):
  - **pp512 = 458.62 ± 4.57 tok/s** (prefill)
  - **tg128 = 15.43 ± 1.11 tok/s** (decode/batch-1)
  - model self-id: `qwen35 27B Q1_0`, 3.53 GiB, 26.90 B params

### Step 1.3 — ncu profile (BLOCKED — DEVIATION)
- Tried `--set full --kernel-name regex:mul_mat_vec_q --launch-skip 10 launch-count 4` → timed out at 1800 s, no CSV written.
- Tried minimal `--metrics` set (7 metrics) with `--launch-skip 100 --launch-count 4` and `-n 64` → timed out at 1200 s, no CSV.
- nsys `--stats=true` → timed out at 300 s, no `.nsys-rep` written.
- Likely root cause: 27B Q1_0 model needs hundreds of warm-up launches (PP indices are ~512 activations + many Q8/K/V/RMS/softmax/etc. gates) before any MMV-Q decode kernel fires. ncu has per-process fork+attach overhead which dominates short kernel runs. Tokio-synchronous CLI also may be doing significant host work at finish.

### Step 1.4 — ALU-vs-DRAM verdict (DEFERRED)

Four options to consider (see plan-deviation ask in handoff):
1. Continue bumping `--launch-skip` (e.g. 5000+) and reduce `--launch-count` to 1–2.
2. Use a smaller model (e.g. Bonsai-3B-Q1_0 if available) for profiling, extrapolate to 27B.
3. Use the cuda event-based manual profiler in `vec_dot_q1_0_q8_1` directly (insert `cudaEventRecord` around a synthetic benchmark).
4. Skip profiling; proceed with doc's "ALU-bound hypothesis" and verify with track-1 deltas.

## Phase 2 — Track 1: dp4a decode optimization

(population pending real numbers)

## Phase 3 — Track 2: Blackwell int8 MMA kernel

(population pending real numbers)

## Phase 4 — Sync / upstream

(population pending real numbers)

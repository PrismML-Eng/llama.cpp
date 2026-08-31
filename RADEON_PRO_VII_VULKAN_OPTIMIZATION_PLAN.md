# Radeon Pro VII Vulkan optimization plan for 27B Q2 generation throughput

## Objective

Increase generation throughput for the 27B Q2 model on the AMD Radeon Pro VII under the Vulkan backend in llama.cpp, with a focus on the decode path where current throughput is bottlenecked by dispatch overhead, memory access patterns, and non-ideal kernel specialization.

Current measured baseline (verified in this repo):

- Model: Qwen 35 27B Q1_0 metadata loaded from the GGUF, consistent with the target 27B class
- GPU: AMD Radeon Pro VII (Vulkan0)
- Benchmark: `llama-bench`
- Release build: yes
- Baseline generation throughput: about 17.14 t/s at `tg128`, with similar behavior around 17 t/s for `tg256`

This is the baseline we are improving from.

## Immediate execution order

The benchmark results are now conclusive enough to stop chasing generic micro-tuning. The next sprint should follow this sequence:

1. Keep the benchmark matrix as the gate for every change.
2. Implement a dedicated short-decode Vulkan profile for Radeon Pro VII / Vega 20.
3. Reduce dispatch and synchronization overhead in the attention path for small token counts.
4. Rework the KV-cache and attention memory layout for sequential decode.
5. Fuse the most expensive QKV + attention stages where the backend currently spills to generic stages.
6. Only after the structural path is stable, re-evaluate subgroup/tile tuning as a secondary refinement.

This is the order we will use to avoid another low-yield tuning loop.

## Phase 1 results (recorded)

### Measured before the first tuning patch

Verified benchmark matrix on the target hardware:

- `tg32`: ~8.4 t/s
- `tg64`: ~8.5 t/s
- `tg128`: ~17.1 t/s
- `tg256`: ~17.1 t/s

This was reproduced across thread counts and batch/ubatch combinations; the result was stable and did not materially change with CPU thread settings.

### Measured after the first architecture-specific patch

After adding explicit Radeon Pro VII / Vega 20 subgroup and tile tuning in [ggml/src/ggml-vulkan/ggml-vulkan.cpp](ggml/src/ggml-vulkan/ggml-vulkan.cpp), the same benchmark matrix produced:

- `tg32`: ~8.4 t/s
- `tg64`: ~8.5 t/s
- `tg128`: ~17.0–17.1 t/s
- `tg256`: ~17.0–17.4 t/s

This is effectively unchanged from the baseline; the gain is within noise.

### Conclusion from Phase 1

The bottleneck is not simply the wrong 64-wavefront subgroup default or generic tile geometry. The decode path is still dominated by a deeper issue, most likely one of:

- attention / KV-cache access pattern
- dispatch and sync overhead
- generic decomposition of decode kernels into many tiny Vulkan stages
- lack of fused decode path for short generation lengths

This means the next optimization target is no longer “more native subgroup tuning”; it is the fused attention / KV-cache / dispatch reduction path.

---

## Why this GPU needs a specialized strategy

The Radeon Pro VII is strong on memory bandwidth and can be very effective for LLM generation, but the current Vulkan path in llama.cpp is still too generic for this architecture. The main problems are likely:

- decode kernels are not specialized enough for small token counts
- too many tiny Vulkan dispatches and sync points
- attention and KV-cache paths are not tuned for HBM2 access patterns
- quantized low-bit matmul kernels are not fully optimized for Vega 20 / Radeon Pro VII
- runtime scheduling still favors generic paths rather than a dedicated decode profile

The goal is not to patch one isolated shader, but to create a dedicated decode-optimized Vulkan strategy for this GPU class.

---

## Scope of work

Primary files to work in:

- [ggml/src/ggml-vulkan/ggml-vulkan.cpp](ggml/src/ggml-vulkan/ggml-vulkan.cpp)
- [ggml/src/ggml-vulkan](ggml/src/ggml-vulkan)
- [tools/llama-bench/llama-bench.cpp](tools/llama-bench/llama-bench.cpp)
- [ggml/src](ggml/src)

---

## Phase 1: establish exact bottlenecks

### Task 1.1: create a benchmark matrix for the exact target hardware

Run a command matrix focused on decode behavior:

```powershell
.\build\bin\Release\llama-bench.exe `
  -m "C:\D\llm\Ternary-Bonsai-27B-Q2_0.gguf" `
  -dev Vulkan0 `
  -p 0 `
  -n 32,64,128,256,512 `
  -b 256,512,1024,2048 `
  -ub 128,256,512 `
  -t 8,12,16 `
  -fa 0,1 `
  -r 3
```

Record:

- tg throughput by generation length
- impact of batch size
- impact of `-fa 1`
- effect of thread count
- where throughput stops scaling

### Task 1.2: identify the kinetic bottleneck

For each case, determine whether the bottleneck is:

- decode matmul bandwidth
- KV-cache access
- attention softmax / reduction
- scheduler overhead
- dispatch overhead

### Task 1.3: validate the model and metadata

Confirm that the file really is the intended 27B model and not a similar family variant with different runtime expectations.

Accept criteria:

- we know which exact workload is slow
- we know whether the bottleneck is memory, attention, or dispatch
- we have a reproducible benchmark matrix

---

## Phase 2: build a Radeon Pro VII specific decode profile

### Task 2.1: add a dedicated decode-optimized Vulkan execution path

Introduce a specialized path for short decode workloads (`n_tokens <= 256` and small batches) that prioritizes:

- lower launch overhead
- fewer barriers
- stronger locality
- higher occupancy for small workgroups

This should be selected from the Vulkan backend runtime instead of the default generic path.

### Task 2.2: choose separate prefill and decode kernel families

Split kernels into the following classes:

- decode kernels for small batch / small sequence lengths
- prefill kernels for large prompt processing
- fused kernels for attention-heavy layers

This is critical because decode and prefill behave very differently on Vega 20 / HBM2.

### Task 2.3: tune a tile strategy for 64-wide wavefront and HBM2

For Vulkan kernels, tune:

- workgroup size
- subgroup/block geometry
- shared memory usage
- vectorized loads
- memory coalescing

The goal is to match Radeon Pro VII characteristics, not generic AMD Vulkan defaults.

Accept criteria:

- decode path uses a dedicated kernel profile
- prefill path remains stable and does not regress
- kernels are tuned for the actual wavefront size and memory characteristics of the hardware

---

## Phase 3: optimize low-bit matmul kernels

### Task 3.1: prioritize Q2/K quantized matmul kernels

Focus on the hottest low-bit decode kernels, especially the ones used by the model family and quantization path in the current benchmark.

Primary targets:

- `mul_mat_vec` / `mul_mv` decode path
- quantized K-block multiply kernels
- properly packed weight layouts for the target GPU

### Task 3.2: reduce memory pressure in Q2/K decode paths

Improve:

- contiguous reads
- weight dequantization strategy
- reuse of temporary results in registers/shared memory
- occupancy tuning
- tile shape selection

### Task 3.3: avoid global-memory round trips per token

The key optimization is to keep intermediate values in registers or local shared memory as long as possible rather than materializing them back to global memory.

Accept criteria:

- measurable lift in decode throughput without regressions in correctness
- reduction in kernel launches for small token workloads

---

## Phase 4: fuse attention and layer subgraphs

### Task 4.1: fuse QKV + attention + output pattern

Combine operations into fewer pipeline stages:

- Q projection
- K/V projection
- attention score computation
- softmax
- V mix
- output projection

This reduces barrier and memory traffic significantly.

### Task 4.2: optimize KV-cache layout for sequential decode

The GPU should use a cache layout that supports:

- efficient sequential write during generation
- efficient read for attention heads
- minimal scattered access
- reduced bank conflicts

### Task 4.3: move attention reductions closer to compute stage

Keep reductions in registers/shared memory before returning to global memory as much as possible. Prefer on-chip reduction over repeated global writes.

Accept criteria:

- attention path consumes less memory bandwidth
- decode throughput improves for small generation lengths
- no accuracy regressions

---

## Phase 5: reduce dispatch and scheduler overhead

### Task 5.1: minimize tiny Vulkan launches

Current decode workloads often suffer from too many tiny dispatches. Replace a set of tiny launches with larger fused kernels or persistent dispatch patterns where possible.

### Task 5.2: add smarter backend heuristics

In the backend runtime, choose a specialized heuristics profile when:

- token count is small
- batch is small
- model is a decode-heavy llm path
- GPU family matches Radeon Pro VII / Vega 20 characteristics

### Task 5.3: prefer latency-aware scheduling over generic throughput scheduling

For generation, the optimizer should prefer:

- better overlap of memory and compute
- fewer synchronization points
- better reuse of allocated buffers

Accept criteria:

- lower launch count for decode path
- measurable reduction in latency overhead per token

---

## Phase 6: validation and release gates

### Task 6.1: create a validation suite that must pass before claiming win

Use the following required checks:

```powershell
.\build\bin\Release\llama-bench.exe -m "C:\D\llm\Ternary-Bonsai-27B-Q2_0.gguf" -dev Vulkan0 -p 0 -n 64,128,256 -b 1024 -ub 512 -fa 1 -t 12 -r 3
```

And compare against the baseline.

### Task 6.2: acceptance thresholds

A patch is considered successful when:

- `tg128` improves materially over the 17.14 t/s baseline
- `tg256` does not regress
- no correctness issues appear in inference or validation tests
- no stability regressions on the Radeon Pro VII

### Task 6.3: regression guardrail for other GPUs

Ensure that the new path does not cause regressions on other Vulkan devices or fallback paths.

---

## Priority order

### Highest priority

1. dedicated decode pipeline for Radeon Pro VII
2. fused attention + KV-cache optimization
3. low-bit matmul kernel tuning
4. dispatch reduction

### Medium priority

5. prefill-specialized path
6. backend heuristics and scheduling policy

### Lower priority

7. general-purpose Vulkan cleanup not directly tied to generation throughput

---

## Expected result

This plan is designed to improve generation behavior on Radeon Pro VII by making the Vulkan backend behave more like a specialized decode engine rather than a generic all-purpose backend.

A realistic target after these changes is not necessarily 100+ t/s immediately, but a clear lift from the current ~17 t/s baseline into a materially higher sustained decode throughput range, with the biggest gains in the small-generation decode cases that dominate interactive generation workloads.

---

## Working checklist

- [ ] Measure exact bottlenecks on current Vulkan path
- [ ] Create dedicated decode profile for Radeon Pro VII
- [ ] Tune Q2/K quantized matmul kernels for Vega 20
- [ ] Fuse attention and KV-cache path
- [ ] Reduce dispatch overhead
- [ ] Validate speed and correctness
- [ ] Re-run benchmark matrix and compare to baseline

---

## Notes

This plan is intentionally tailored to the Radeon Pro VII, not a generic AMD roadmap. The emphasis is on architecture-aware kernel design and backend scheduling, because the biggest opportunities on this GPU are in small-batch decode efficiency and memory-coalesced attention handling.

# CUDA GDN verification kernel

`GGML_CUDA_GDN_L8=1` enables an alternate gated-delta-net kernel for 8-16 token verification batches with saved recurrent states. The default is off. Unset the variable or use `GGML_CUDA_GDN_L8=0` for the existing kernel. Set the variable before starting the process.

The alternate kernel requires NVIDIA Ampere or later device code, a 32-lane warp, state size 128, scalar gates, and more than one snapshot slot. Other shapes, KDA, HIP and MUSA use the existing implementation. It supports activated and raw gates, strided q/k/v inputs, and indexed state gathering.

Eight lanes cooperate on each state column. Each lane holds the partial sums of four lanes from the existing kernel; the reduction preserves its xor 16, 8, 4, 2, 1 addition tree. All k/v/gate inputs for the verification batch are staged in shared memory before recurrence. The kernel writes the ordinary eager snapshots and uses the existing rollback layout.

## Provenance

Adapted from ArthurOstapenko's `gated_delta_net_l8v_cuda` and `gdn_l8_reduce` in the [Yukon CUDA engine](https://github.com/Layr-Labs/cudafast-bonsai2-9b-engine/tree/36c95362b23b83724503fef4e88e61d748f9355f). The source MIT notice is in [licenses/yukon-cudafast](../../licenses/yukon-cudafast).

This port contains the eager-snapshot verification kernel. It does not adopt the source's lazy state encoding, prefill kernel, cache eviction policy, matrix multiplication changes, graph fusions, DFlash selector, or benchmark harness. Source dispatch was specific to DGX Spark; the portable Ampere instructions here remain opt-in while other devices are measured.

## Validation

The backend test suite includes the dispatch boundaries 7/8/15/16/17 tokens, strided tensors, multiple sequences, indexed state reads, raw gates, partial snapshot windows, final-state-only and KDA fallbacks:

```sh
GGML_CUDA_GDN_L8=0 ./build/bin/test-backend-ops test -b CUDA0 -o GATED_DELTA_NET
GGML_CUDA_GDN_L8=1 ./build/bin/test-backend-ops test -b CUDA0 -o GATED_DELTA_NET
```

Keep performance claims specific to the measured shapes and device. The source leaderboard compares a full speculative runtime against a serial control; it does not isolate this kernel or compare against the current fork.

### H200 validation (2026-10-18)

A Release sm_90 build passed all 56 GATED_DELTA_NET backend cases with the option off and on. A direct original/candidate comparison passed 432 bytewise attention and snapshot checks with production `-use_fast_math`, three fresh seeds, token counts 8/9/15/16, 12/32/48 value heads, one/two sequences, snapshot counts 3/17/18, and activated/raw gates. A compile-only sm_75 check passed; no sm_75 execution is claimed.

All 14 pinned public Bonsai 2 target artifacts passed bytewise full-logit comparisons and Nsight confirmation of `gated_delta_net_l8v_cuda` execution:

| Size | Formats |
|---|---|
| 2B | PTQ1_0, PQ2_0, Q2_0 |
| 4B | PTQ1_0, PQ2_0, Q2_0, PQ2_0-MTP-Q8_0 |
| 9B | PTQ1_0, PQ2_0, Q2_0, PQ2_0-MTP-Q8_0 |
| 27B | PTQ1_0, PQ2_0, F16 |

Each artifact's SHA256 matched its pinned public repository revision. Each arm used 16 corpus slices of 512 accepted tokens, one untimed warmup slice, and 8/15/16-token verification batches with partial rollback. Each comparison covered 10,234,757,120 bytes of logits. MTP-bearing artifacts were exercised as targets; this does not validate the complete MTP/DFlash serving loop. DGX Spark remains untested.

Five matched timing pairs per artifact gave median paired original/candidate time ratios from 0.9900x to 1.0220x. Several pairs had substantial variability. The timer includes decode, all-logit retrieval and writes to `/dev/null`; it excludes loading, state resets and rollback API calls. These verification-workload timings do not establish a broad generation-throughput gain.

At the isolated kernel level, 32 value heads, one sequence and 18 snapshot slots gave median original/candidate time of 1.219x for activated gates and 1.214x for raw gates. At 48 heads and two sequences this fell to 0.864x and 0.875x. The multi-sequence regression, small/noisy model-level differences and unmeasured Spark behavior are reasons to keep the option off by default.

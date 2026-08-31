Most effective optimization points
The hot path is the dequant + matvec loop in mul_mat_vec.comp, with the reduction logic in mul_mat_vec_base.glsl. For Radeon Pro VII, the best wins are usually:

Reduce decode cost to almost zero
Increase memory coalescing and vector throughput
Keep the reduction stage branchless and subgroup-friendly
Match the workgroup heuristic to the short decode pattern
1) Keep the ternary decode branchless and vectorized
Right now the critical dequant function for the ternary family is:

dequant_funcs.glsl:107-121
It does bit-extraction and returns ±1.0 values. That is already the right idea. The next optimization is to keep it fully vectorized:

decode 4 or 8 lanes at once with vec4 / vec8 instead of scalar bit manipulations
compute sign masks in a single register block
avoid extra scalar conversions inside the loop
This matters because decode is repeated across all tokens and all layers.

2) Specialize the kernel for the exact input format
The dispatch layer in ggml-vulkan.cpp:6462-6568 is already choosing the shader family by type. The optimization to pursue is:

keep Q2_0 mapped to a dedicated ternary-family path
avoid falling through into generic Q1_0 / Q2_0 logic that still carries non-ternary checks
use a distinct Q2_0 or Q1_0_g128 specialized shader for the Pro VII decode pattern
In other words, do not only alias by family; generate a dedicated pipeline name and shader variant tuned for the exact layout.

3) Use the Pro VII-friendly workgroup sizing aggressively
The workgroup selection is already in ggml-vulkan.cpp:6490-6558:

DMMV_WG_SIZE_LARGE
DMMV_WG_SIZE_SUBGROUP
For short decode / small N, the large workgroup helps if the kernel stays memory-bound and the reduction overhead remains small. The optimization is:

keep the short-decode fast path active for dst->ne[1] <= 8 and similar cases
avoid oversubscription when the row count is large
tune the BLOCK_SIZE / NUM_ROWS constants to match Pro VII wavefront behavior
This is the main platform-specific lever.

4) Reduce the reduction overhead
The reduction stage in mul_mat_vec_base.glsl has three modes:

subgroupAdd
shared-memory sum across subgroups
tree reduction in shared memory
The best performance on AMD GCN is usually:

use subgroup reduction when available
pack multiple scalar sums into a vectorized accumulation before the final reduction
minimize barrier() churn
A good optimization is to keep the accumulation in registers across a few K chunks, then do one final reduction per workgroup instead of repeatedly writing to shared memory.

5) Remove unnecessary OOB and branch checks in the hot loop
In mul_mat_vec.comp, there is a per-iteration OOB check:


That’s necessary for correctness, but for fixed-length decode workloads it can be optimized away or predicated once per workgroup:

compute a compact “tail” handling path
keep the common case fully branchless
do tail handling only on the last K-block
That reduces control-flow overhead.

6) Tune memory layout to match the Pro VII cache hierarchy
The kernel is reading packed weights and then a dense activation vector. The key improvement is:

keep A blocks contiguous and aligned
avoid scattered reads across the packed layout
process multiple rows per invocation so each block fetch is reused
For Q1_0_g128/Q2_0-like formats, the pairwise access pattern is extremely simple; the best path is to stretch a block across a wavefront and let memory-level parallelism cover the fetch.

7) Better specialized shader family
The highest-value engineering change is not just a “better alias”, but a dedicated generated shader family for the real format:

mul_mat_vec_q2_0_f32_f32
mul_mat_vec_q1_0_g128_f32_f32
maybe a *_pro_vii variant
This allows:

different BLOCK_SIZE
different NUM_ROWS
different reduction mode
different register pressure
different workgroup shape
The current generic family is already close, but specific tuning is where the Pro VII gain comes from.

Practical prioritization
If I had to rank the optimization order for this exact path:

Dedicated Pro VII ternary shader family with tuned workgroup
Vectorized dequant (vec4/vec8) with branchless sign extraction
Reduce shared-memory reduction overhead
Minimize OOB / tail-path overhead
Tune NUM_ROWS and BLOCK_SIZE empirically
That is the highest-leverage way to push this kernel beyond the generic path.
# Ternary Vulkan backend implementation blueprint for Radeon Pro VII

This document is a concrete implementation guide for adding a specialized Vulkan backend path for truly ternary models, assuming the model is ternary by construction.

The goal is to replace the generic `GGML_OP_MUL_MAT` / dequantized matmul path with a direct ternary matmul kernel that uses addition/subtraction instead of multiply in the hot loop, optimized for Radeon Pro VII decode generation.

---

## 1. Scope and assumptions

Assumptions:
- the model is ternary, so no runtime detection is necessary
- all weights are stored in a packed ternary representation
- no fallback to generic Q2/Q4/Q8 path is needed for the target workload
- target backend: Vulkan on AMD Radeon Pro VII
- target operation: decode-time token generation (`MUL_MAT` in the autoregressive decode path)

Primary objective:
- remove multiplication from the hot path
- replace with direct accumulation using:
  - `+1` => add input
  - `-1` => subtract input
  - `0` => skip

Key formula:

$$
y = \sum_i x_i \cdot w_i, \quad w_i \in \{-1, 0, +1\}
$$

Equivalent computation:

$$
y = \sum_{w_i=+1} x_i - \sum_{w_i=-1} x_i
$$

This is the core optimization to implement in Vulkan.

---

## 2. Exact edit points in the current code

These are the exact functions and regions in the repository that must be extended or replaced.

### 2.1. Main Vulkan matmul dispatch
File:
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

Function:
- `static void ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx)`
- location: around lines 8490-8548

This is the first important insertion point. It decides whether the model uses:
- vectorized path
- q_f16 path
- generic matmul path

This function must be given a direct ternary branch before the generic `ggml_vk_mul_mat_q_f16(...)` fallback.

### 2.2. Generic matmul implementation
Function:
- `static void ggml_vk_mul_mat_q_f16(...)`
- location: around lines 7731-8064

This is the current generic Vulkan matmul implementation used for `GGML_OP_MUL_MAT`.

This function is the main replacement target for the ternary specialized path.

### 2.3. Real dispatch into GPU pipeline
Function:
- `static void ggml_vk_matmul(...)`
- location: around lines 7429-7489

This is the actual GPU dispatch function that calls the Vulkan pipeline and provides push constants.

This is the place where the new ternary kernel will be invoked, instead of the generic matmul pipeline.

### 2.4. Pipeline selection helpers
Relevant helpers:
- `ggml_vk_get_mul_mat_mat_pipeline(...)`
- `ggml_vk_guess_matmul_pipeline(...)`
- `ggml_vk_guess_matmul_pipeline_align(...)`
- `ggml_vk_get_mul_mat_mat_id_pipeline(...)`

These are near the earlier section of the same file, around the matmul pipeline registry.

For the ternary backend path, a new specialized pipeline family should be created alongside these rather than reusing them.

---

## 3. High-level design

### 3.1. Create a ternary-specific backend path
Do not modify the generic matmul path in place for the default path.
Instead add a dedicated path:

- `ggml_vk_mul_mat_ternary(...)`
- `ggml_vk_matmul_ternary(...)`
- `ggml_vk_dispatch_ternary_matmul(...)`

This new path must be selected explicitly in `ggml_vk_mul_mat(...)`.

### 3.2. New packed ternary weight format
The weights must be stored in packed ternary format, such as:
- 2 bit / value or custom sign+zero-mask pack
- grouped by K and N tiles for AMD wavefront-friendly access
- aligned to 128 byte / 256 byte boundaries depending on the device

The packed format should be optimized for the following pattern:
- load contiguous K-block, decode sign patterns in the shader
- do not reconstruct floats in memory
- directly add/subtract inputs

### 3.3. Hot loop operation
The shader hot loop should be:

```glsl
for (int k = 0; k < K_per_tile; ++k) {
    uint8_t code = ternary_decode(packed_block[k]);
    if (code == +1) {
        acc += x[k];
    } else if (code == -1) {
        acc -= x[k];
    } else {
        // zero weight => skip
    }
}
```

This is the central transformation.

---

## 4. Exact implementation plan

### Step 1. Add a dedicated ternary matmul branch in the Vulkan dispatch selector
Edit:
- `ggml_vk_mul_mat(...)` in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

Insert before the generic fallback:

```cpp
if (src0->type == GGML_TYPE_TERNARY_PACKED || src1->type == GGML_TYPE_TERNARY_PACKED) {
    ggml_vk_mul_mat_ternary(ctx, subctx, cgraph, node_idx);
    return;
}
```

Or, if the ternary type is guaranteed to be the weight type for the whole model, a stronger version can be used:

```cpp
if (src0->type == GGML_TYPE_TERNARY_PACKED) {
    ggml_vk_mul_mat_ternary(ctx, subctx, cgraph, node_idx);
    return;
}
```

This branch must be placed before the generic `ggml_vk_mul_mat_vec_q_f16(...)` / `ggml_vk_mul_mat_q_f16(...)` path.

### Step 2. Implement a specialized ternary matmul entry function
Add a function near the generic matmul functions, for example just before `ggml_vk_mul_mat_q_f16(...)`.

Suggested outline:

```cpp
static void ggml_vk_mul_mat_ternary(
    ggml_backend_vk_context * ctx,
    vk_context& subctx,
    const struct ggml_cgraph * cgraph,
    int node_idx)
{
    ggml_tensor * dst = cgraph->nodes[node_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    // assume src0 or src1 is packed ternary
    // validate shapes and packed layout
    // choose ternary pipeline
    // prepare buffers
    // dispatch ternary matmul
}
```

Important behavior:
- no dequantization to fp16/f32
- no `ggml_vk_get_to_fp16(...)`
- no `ggml_vk_quantize_q8_1(...)`
- no generic `ggml_vk_matmul(...)` call for the ternary path

### Step 3. Add dedicated ternary pipeline selection
Near the pipeline selection code, add a new family:

```cpp
static vk_pipeline ggml_vk_get_ternary_matmul_pipeline(...);
static vk_pipeline ggml_vk_get_ternary_matmul_vec_pipeline(...);
```

These should be selected based on:
- M dimension
- N dimension
- K dimension
- tile size for AMD Pro VII
- decode-length regime (very small M / short decode)

The important thing is not to share the existing generic pipeline tables. Ternary path should have its own pipeline family and its own tuning decisions.

### Step 4. Implement a ternary matmul helper that mirrors `ggml_vk_matmul` but without generic F16 path
Add a helper beside `ggml_vk_matmul(...)`:

```cpp
static void ggml_vk_matmul_ternary(
    ggml_backend_vk_context * ctx,
    vk_context& subctx,
    vk_pipeline& pipeline,
    vk_subbuffer&& a,
    vk_subbuffer&& b,
    vk_subbuffer&& d,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t stride_a,
    uint32_t stride_b,
    uint32_t stride_d,
    uint32_t batch_stride_a,
    uint32_t batch_stride_b,
    uint32_t batch_stride_d,
    uint32_t batch,
    uint32_t ne02,
    uint32_t ne12,
    uint32_t broadcast2,
    uint32_t broadcast3,
    uint32_t padded_n)
```

This helper should dispatch directly to the ternary shader with a very small set of push constants:
- M
- N
- K
- stride of X and W
- batch metadata
- packed block metadata
- maybe row offset metadata

### Step 5. Shader file addition
The actual shader should live in the Vulkan shader directory, next to the generic shaders.

Suggested file names:
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_ternary.comp`
- or a similar AMD-specific naming scheme

The shader should:
- read packed ternary weights
- decode sign/zero pattern
- operate on loaded input `x`
- use add/subtract accumulation
- produce the output tile

The shader must target AMD GCN semantics and use proper subgroup/wavefront behavior.

### Step 6. Fuse the decode operations around the ternary matmul
In the ternary path, do not do a separate dispatch for each surrounding op if the decode graph allows fusion.

The target fused pattern should be:
- RMSNorm / pre-norm
- scale
- ternary matmul
- residual add
- optional activation

This should be implemented in the same dispatch flow, or in a tightly-coupled fused pipeline sequence.

This is where the throughput gain becomes real in short decode generation.

### Step 7. Add AMD Pro VII tuning parameters
The new ternary kernel must be tuned specifically for `AMD_GCN` / Radeon Pro VII.

Hardcoded or selected tuning should include:
- workgroup size
- tile shape per `M, N, K`
- subgroup usage
- LDS tile size
- register budget
- memory coalescing pattern

Recommended tuning approach:
- `M` tile: small for decode, likely 16/32/64 rows
- `N` tile: 64/128/256 columns
- `K` tile: pack in contiguous blocks for efficient sign decode
- avoid large `split_k` in decode path

This is especially important because decode is short, and dispatch overhead dominates if the tile is too big or split logic is too expensive.

### Step 8. Validate decode-generation performance
Use the existing benchmark harness and `llama-bench` path.

Target validation command:
- `llama-bench` with `-dev Vulkan0`
- `-p 0`
- `-n 32,64,128,256`
- `-b 512,1024`
- `-ub 128,256,512`
- `-t 8,12,16`

Measure:
- tokens/sec
- GPU time
- memory bandwidth usage
- dispatch count
- average latency for short decode generation

This gives direct comparison between:
- generic `mul_mat_q_f16`
- ternary specialized path

---

## 5. Practical pseudo-code skeleton

### 5.1. New entry point in `ggml_vk_mul_mat`
```cpp
static void ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * dst = cgraph->nodes[node_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    // AXIOM: the target model is ternary
    if (src0->type == GGML_TYPE_TERNARY_PACKED || src1->type == GGML_TYPE_TERNARY_PACKED) {
        ggml_vk_mul_mat_ternary(ctx, subctx, cgraph, node_idx);
        return;
    }

    // existing fallback path remains unchanged for non-ternary workloads
    // ... old code
}
```

### 5.2. New ternary helper
```cpp
static void ggml_vk_mul_mat_ternary(
    ggml_backend_vk_context * ctx,
    vk_context& subctx,
    const struct ggml_cgraph * cgraph,
    int node_idx)
{
    ggml_tensor * dst = cgraph->nodes[node_idx];
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    // validate ternary shapes and metadata
    // choose packed-ternary pipeline
    // allocate / map GPU buffers
    // dispatch specialized ternary shader
}
```

### 5.3. Shader conceptual structure
```glsl
layout(local_size_x = WG_X, local_size_y = WG_Y, local_size_z = 1) in;

void main() {
    uint row = gl_GlobalInvocationID.x;
    uint col = gl_GlobalInvocationID.y;

    float acc = 0.0;

    // iterate packed K-blocks
    for (uint k = 0; k < K_BLOCKS; ++k) {
        uint code = unpack_ternary(weights, k);
        float x = load_input(row, k);

        if (code == 1u) {
            acc += x;
        } else if (code == 2u) {
            acc -= x;
        }
        // code == 0 => skip
    }

    out[col + row * N] = acc;
}
```

This is the exact replacement for the multiplication-based hot loop.

---

## 6. Minimal implementation checklist

### Required files / areas
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- `ggml/src/ggml-vulkan/vulkan-shaders/`
- possibly a small type registration update where GGML types are enumerated
- optional: `ggml/include/ggml.h` or related type declarations if a new ternary type is introduced

### Required functions to add or modify
- `ggml_vk_mul_mat`
- `ggml_vk_mul_mat_ternary`
- `ggml_vk_matmul_ternary`
- `ggml_vk_get_ternary_matmul_pipeline`
- Vulkan shader file for ternary matmul

### Required behavior
- no generic dequantization to fp16
- no multiply in the hot loop
- direct add/sub accumulation
- optimized for decode generation on Radeon Pro VII

---

## 7. Recommended implementation order

1. Add ternary-specific dispatch branch in `ggml_vk_mul_mat`
2. Add stub ternary matmul entry function
3. Add shader file and pipeline registry entry
4. Hook dispatch to the new shader
5. Validate output against expected matrix math on a small test case
6. Add decode fusion around the ternary matmul
7. Tune workgroup layout for AMD Pro VII
8. Benchmark on real model and iterate

---

## 8. Final objective

The final backend behavior should be:
- all target model weights are treated as ternary
- the decode path uses add/sub instead of multiply in the hot loop
- Vulkan dispatch calls a specialized ternary matmul pipeline
- Radeon Pro VII tuning is applied at the shader and workgroup level
- generation speed improves by reducing ALU and memory overhead in the decode-heavy regime

This is the correct specialized backend path for the true ternary model scenario.

#pragma once

#include "common.hpp"

// PQ2_0 in the XMX layout: the weight tensor is rewritten in place, once, into a plane of 32-byte qs blocks
// (row pitch nb * 32 bytes, so every row is 2D-block-load aligned) followed by a plane of fp16 block scales.
// PTQ1_0 weights take the same layout: their base-3 trits are expanded to PQ2_0 codes, which needs 34 bytes a
// block instead of 28, so the buffer type reserves that room for them on devices that use this path.
// Activations are quantized to int8 with one float scale per 128 values, so the four DPAS of a PQ2_0 block
// accumulate in integers before a single float rescale.

// ne[0] of a PQ2_0 weight the XMX path accepts: a 2D surface needs a row of at least 64 bytes
bool ggml_sycl_pq2_xmx_supports_ne0(int64_t ne0);

// rewrite src0 (PQ2_0 or PTQ1_0, AoS blocks) into the XMX layout in place
bool ggml_sycl_pq2_xmx_reorder(ggml_tensor * src0, dpct::queue_ptr stream);

// dst = src0 * src1 for a src0 already in the XMX layout; src1 is f32 with contiguous rows, dst is contiguous
void ggml_sycl_pq2_xmx_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst);

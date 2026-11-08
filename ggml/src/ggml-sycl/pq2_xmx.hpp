#pragma once

#include "common.hpp"

// PQ2_0 / PTQ1_0 x int8 on 16-wide DPAS devices with native s2 x s8 DPAS. The weight tensor is rewritten in
// place, once, into an XMX layout whose rows are padded to NP = ne[1] rounded up to 16 (zero columns), so the
// buffer must hold ggml_sycl_pq2_xmx_bytes():
//  PQ2_0:  uint32 qs[K/16][NP] as s2 codes, then half d[K/128][NP]. Codes are ternary only: PQ2_0 code 3 (+2)
//          has no s2 value.
//  PTQ1_0: uint32 [K/128][7][NP]: the 24 qs bytes, then qh[0] | qh[1] << 8 | d << 16. The trits are decoded
//          in the kernels, so the weights stay at 1.75 bits.
// Activations are quantized to int8 with one float scale per 128 values, so the four DPAS of a block accumulate
// in integers before a single float rescale.

// ne[0] of a weight the XMX path accepts
bool ggml_sycl_pq2_xmx_supports_ne0(int64_t ne0);

// bytes a 2D weight takes in the XMX layout
size_t ggml_sycl_pq2_xmx_bytes(const ggml_tensor * t);

// rewrite src0 (PQ2_0 or PTQ1_0, AoS blocks) into the XMX layout in place
bool ggml_sycl_pq2_xmx_reorder(ggml_tensor * src0, dpct::queue_ptr stream);

// dst = src0 * src1 for a src0 already in the XMX layout; src1 is f32 with contiguous rows, dst is contiguous
void ggml_sycl_pq2_xmx_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst);

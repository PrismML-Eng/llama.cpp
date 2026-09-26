#pragma once
#include "common.hpp"

// Prompt-sized mat-mul for PQ2_0 / PTQ1_0 weights on int8 XMX via oneDNN s8 x s8.
// Per-128-group weight scales are requantized to a per-row scale (int8 weight = code * round(127 * d_g / max_g d)),
// activations are quantized to int8 per token; both scales are applied in one pass over the output.
// Returns false when not applicable (caller falls back).
bool ggml_sycl_w8a8_mul_mat(ggml_backend_sycl_context & ctx, ggml_type type, bool reordered, const void * w,
                            const float * x, float * dst, int64_t nrows, int64_t ncols_x, int64_t K,
                            dpct::queue_ptr stream);

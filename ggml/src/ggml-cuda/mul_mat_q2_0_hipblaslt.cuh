// Experimental PREFILL lever: route Q2_0 (ternary, g128) large-M matmuls through
// AMD's tuned hipBLASLt int8 GEMM instead of llama.cpp's ~5%-efficient mmq/dp4a
// path. Measured hipBLASLt int8 on gfx1201 = 156-266 TOPS (20-35% of peak) vs
// dp4a's ~64 TOPS -> 2.5-4x on the GEMM at the SAME W8A8 precision.
//
// Data flow (per prefill matmul, M > threshold):
//   Q2_0 weight ----requant (once, cached)----> int8 + per-output-channel scale
//   activations ---quantize (per call)--------> int8 + per-token scale
//                     hipBLASLt int8 GEMM (i8 x i8 -> i32, TN)
//                     i32 * wscale[row] * ascale[col] -> f32 dst
//
// This is v1 (per-output-channel weight scale):
// Stage 2 measured the added accuracy cost over the int8-activation floor at
// only ~0.23% rel-RMS. Decode (M<=threshold) is UNTOUCHED -- dp4a stays the
// GEMV path; hipBLASLt is a prefill/large-M play (matrix engine needs a full
// tile). Gated behind env var GGML_HIP_Q2_0_HIPBLASLT_PREFILL, opt-in, dormant
// by default. Single GPU, 2D weights only (ne2==ne3==1), MUL_MAT only.
#pragma once

#include "common.cuh"

// Whether this call site is in-scope for the hipBLASLt prefill path (Q2_0
// weight, F32 acts/dst, 2D, RDNA4, and M > threshold). Checked by the
// ggml_cuda_mul_mat() intercept before dispatch. Soft/opt-in: a false return
// falls straight through to the unmodified mmq/dp4a path.
bool ggml_cuda_q2_0_hipblaslt_prefill_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// Run the Q2_0 prefill matmul through hipBLASLt int8. Returns false if the
// build has no hipBLASLt (non-HIP / disabled) so the caller can fall back.
bool ggml_cuda_op_mul_mat_q2_0_hipblaslt(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

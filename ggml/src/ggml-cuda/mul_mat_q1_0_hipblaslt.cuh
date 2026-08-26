// mul_mat_q1_0_hipblaslt.cuh - Q1_0 hipBLASLt prefill GEMM route header
// Route Q1_0 (binary {-1,+1} weights, QK1_0=128) to hipBLASLt int8 or fp8 (e4m3) GEMM
// for prefill (M > M_THRESH). Per-channel weight scale, per-token activation scale.

#pragma once

#include "common.cuh"

// Whether this call site is in-scope for the hipBLASLt prefill path (Q1_0
// weight, F32 acts/dst, 2D, RDNA4, and M > threshold). Checked by the
// ggml_cuda_mul_mat() intercept before dispatch. Soft/opt-in: a false return
// falls straight through to the unmodified mmq/dp4a path.
bool ggml_cuda_q1_0_hipblaslt_prefill_supports(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// Run the Q1_0 prefill matmul through hipBLASLt int8/fp8. Returns false if the
// build has no hipBLASLt (non-HIP / disabled) so the caller can fall back.
bool ggml_cuda_op_mul_mat_q1_0_hipblaslt(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

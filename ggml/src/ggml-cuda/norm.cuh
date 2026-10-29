#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_add_rms_norm_fused(
        ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm, ggml_tensor * mul);

bool ggml_cuda_add_rms_norm_fwht_supported(int ncols, int blk);

void ggml_cuda_op_add_rms_norm_fwht_fused(
        ggml_backend_cuda_context & ctx, const ggml_tensor * add, const ggml_tensor * rms_norm,
        const ggml_tensor * weight, const ggml_tensor * signs, ggml_tensor * normed, ggml_tensor * dst,
        void * q8 = nullptr, ggml_cuda_q8_1_layout q8_layout = GGML_CUDA_Q8_1_AOS); // q8: write q8_1 rows there instead of dst

void ggml_cuda_op_add_rms_norm_scale_fused(
        ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms_norm, float * row_scale);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Gemma 4 MoE layer tail (3 rms_norm+mul(+add) groups and the layer-scalar mul, optionally the next attn_norm) in one launch; nodes are the 9 or 11 graph nodes
void ggml_cuda_op_tail_fused(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes, int n_fused,
                             float * verify_scratch, float * verify_scratch2, block_q8_1 * q8, int ne0_pad);
void ggml_cuda_op_tail_verify_cmp(ggml_backend_cuda_context & ctx, const ggml_tensor * out, const float * scratch, int tag);
void ggml_cuda_op_tail_verify_cmp_q8(ggml_backend_cuda_context & ctx, const void * ref, const void * fused, size_t nbytes, int tag);

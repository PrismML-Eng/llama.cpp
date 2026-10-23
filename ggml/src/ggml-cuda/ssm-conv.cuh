#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// fused GDN decode conv chain (one token, one sequence); see ggml_cuda_gdn_conv_fusion in ggml-cuda.cu
struct ggml_cuda_ssm_conv_gdn_fused_args {
    const float *   cache      = nullptr; // conv-state cache, rows of row_stride floats
    const int32_t * ids        = nullptr; // ids[0]: the row to read
    int64_t         row_stride = 0;
    const float *   x          = nullptr; // new token, channel c at x[c*x_stride]
    int64_t         x_stride   = 0;
    const float *   w          = nullptr; // conv weights, channel c at w[c*w_stride + j]
    int             w_stride   = 0;
    float *         cache_dst  = nullptr; // conv-state row written back, [n_channels*(d_conv - 1)]
    float *         y          = nullptr; // silu(conv), [n_channels]
    float *         l2         = nullptr; // l2-normalised first n_l2 channels, rows of row_len
    int             n_l2       = 0;
    int             n_channels = 0;
    int             row_len    = 0;
    int             d_conv     = 0;
    float           eps        = 0.0f;
};

void ggml_cuda_ssm_conv_gdn_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_ssm_conv_gdn_fused_args & args);

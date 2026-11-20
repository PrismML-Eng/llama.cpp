#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// fused GDN conv chain for one sequence and 1..GGML_CUDA_GDN_CONV_MAX_TOKENS tokens, with up to
// GGML_CUDA_GDN_CONV_MAX_SNAP conv-state writes (one at a plain step, 1 + n_rs_seq rollback snapshots at a verify
// step); see ggml_cuda_gdn_conv_match in ggml-cuda.cu
#define GGML_CUDA_GDN_CONV_MAX_TOKENS 16
#define GGML_CUDA_GDN_CONV_MAX_SNAP   16

struct ggml_cuda_ssm_conv_gdn_fused_args {
    const float *   cache      = nullptr; // conv-state cache, rows of row_stride floats
    const int32_t * ids        = nullptr; // ids[0]: the row to read
    int64_t         row_stride = 0;
    const float *   x          = nullptr; // new tokens, channel c of token t at x[t*x_tok_stride + c*x_stride]
    int64_t         x_stride   = 0;
    int64_t         x_tok_stride = 0;
    const float *   w          = nullptr; // conv weights, channel c at w[c*w_stride + j]
    int             w_stride   = 0;
    float *         cache_dst  = nullptr; // base of the conv-state rows written back
    int             n_snap     = 0;       // conv-state rows written, each [n_channels*(d_conv - 1)]
    int64_t         snap_offs[GGML_CUDA_GDN_CONV_MAX_SNAP] = {}; // row k starts at cache_dst + snap_offs[k] (floats)
    int             snap_pos [GGML_CUDA_GDN_CONV_MAX_SNAP] = {}; // holding conv-input columns snap_pos[k] + [0, d-1)
    float *         y          = nullptr; // silu(conv), [n_channels, n_t]
    float *         l2         = nullptr; // l2-normalised first n_l2 channels per token, rows of row_len, [n_l2, n_t]
    int             n_l2       = 0;
    int             n_channels = 0;
    int             n_t        = 1;
    int             row_len    = 0;
    int             d_conv     = 0;
    float           eps        = 0.0f;
};

void ggml_cuda_ssm_conv_gdn_fused(ggml_backend_cuda_context & ctx, const ggml_cuda_ssm_conv_gdn_fused_args & args);

#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

// Fused state gather. build_rs materialises GET_ROWS(cache, s_copy) into a temp per layer that the
// GDN kernel reads once; that temp is a dirty L2 write that later evicts into the FFN weight
// stream. When the GET_ROWS feeds nothing but this op's state input, the backend skips it and
// registers the gather here so the kernel reads cache row s_ids[seq] directly.
struct ggml_cuda_gated_delta_net_gather {
    const float *   base       = nullptr; // cache rows [row_stride floats each]
    const int32_t * ids        = nullptr; // per-seq row index
    int64_t         row_stride = 0;       // in floats
};

void ggml_cuda_gdn_gather_clear();
void ggml_cuda_gdn_gather_register(const ggml_tensor * gdn, const ggml_cuda_gated_delta_net_gather & gather);

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

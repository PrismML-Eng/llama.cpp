#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// up to CUDA_CPY_BATCH_MAX independent F32 -> F32 copies with one shared shape and stride set (only the data pointers
// differ) in one launch, e.g. the per-slot rollback snapshots of a recurrent conv state
#define CUDA_CPY_BATCH_MAX 32

struct ggml_cuda_cpy_batch {
    const char * src[CUDA_CPY_BATCH_MAX];
    char *       dst[CUDA_CPY_BATCH_MAX];
};

void ggml_cuda_cpy_f32_batched(ggml_backend_cuda_context & ctx, const ggml_tensor * src_ref, const ggml_tensor * dst_ref,
                               const ggml_cuda_cpy_batch & batch, int n);

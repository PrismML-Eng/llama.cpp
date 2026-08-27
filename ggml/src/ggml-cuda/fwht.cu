#include "common.cuh"
#include "fwht.cuh"

template <typename T>
__device__ __forceinline__ float fwht_load(const T value) {
    return value;
}

template <>
__device__ __forceinline__ float fwht_load<half>(const half value) {
    return __half2float(value);
}

template <int N, typename T, bool has_signs>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const T * src, float * dst, const int64_t n_rows, const float scale,
                          const float * signs, const int n_blk) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = fwht_load(src[i * warp_size + lane]) * scale;
        if (has_signs) {
            reg[i] *= signs_row[i * warp_size + lane];
        }
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

template <typename T>
static bool fwht_launch(ggml_backend_cuda_context & ctx, const T * src_d, float * dst_d,
                        const int n, const int64_t rows, const float scale,
                        const float * signs, const int n_blk) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;
    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
    cudaStream_t stream = ctx.stream();
    dim3 grid_dims(num_blocks, 1, 1);
    dim3 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n) {
#define FWHT_CASE(NN) \
        case NN: \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda<NN, T, true>,  launch_params, src_d, dst_d, rows, scale, signs, n_blk); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda<NN, T, false>, launch_params, src_d, dst_d, rows, scale, nullptr, 1); \
            } \
            return true;
        FWHT_CASE(64)
        FWHT_CASE(128)
        FWHT_CASE(256)
        FWHT_CASE(512)
        FWHT_CASE(1024)
        FWHT_CASE(2048)
#undef FWHT_CASE
        default:
            return false;
    }
}

static bool fwht_dispatch(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                          const ggml_tensor * signs_t) {
    GGML_ASSERT(ggml_nelements(src) == ggml_nelements(dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int     n    = dst->ne[0];
    const int64_t rows = ggml_nelements(dst) / n;

    if ((src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const float * signs = nullptr;
    int n_blk = 1;
    if (signs_t) {
        if (signs_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs_t) || signs_t->ne[0] % n != 0) {
            return false;
        }
        signs = (const float *) signs_t->data;
        n_blk = signs_t->ne[0] / n;
    }

    float * dst_d = (float *) dst->data;
    const float scale = 1 / sqrtf(n);

    if (src->type == GGML_TYPE_F32) {
        return fwht_launch<float>(ctx, (const float *) src->data, dst_d, n, rows, scale, signs, n_blk);
    }
    return fwht_launch<half>(ctx, (const half *) src->data, dst_d, n, rows, scale, signs, n_blk);
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    return fwht_dispatch(ctx, src, dst, nullptr);
}

bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst) {
    return fwht_dispatch(ctx, src, dst, signs);
}

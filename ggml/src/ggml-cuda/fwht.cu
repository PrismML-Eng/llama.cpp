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

template <int N, typename T>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const T * src, float * dst, const int64_t n_rows, const float scale) {
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
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = fwht_load(src[i * warp_size + lane]) * scale;
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

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int     n    = src->ne[0];
    const int64_t rows = ggml_nrows(src);

    if ((src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const void * src_d = src->data;
    float *       dst_d = (float *) dst->data;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;

    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    const float scale = 1 / sqrtf(n);

    if (src->type == GGML_TYPE_F32) {
        const float * src_f32 = (const float *) src_d;
        switch (n) {
            case 64:
                ggml_cuda_kernel_launch(fwht_cuda<64, float>, launch_params, src_f32, dst_d, rows, scale);
                return true;
            case 128:
                ggml_cuda_kernel_launch(fwht_cuda<128, float>, launch_params, src_f32, dst_d, rows, scale);
                return true;
            case 256:
                ggml_cuda_kernel_launch(fwht_cuda<256, float>, launch_params, src_f32, dst_d, rows, scale);
                return true;
            case 512:
                ggml_cuda_kernel_launch(fwht_cuda<512, float>, launch_params, src_f32, dst_d, rows, scale);
                return true;
            case 1024:
                ggml_cuda_kernel_launch(fwht_cuda<1024, float>, launch_params, src_f32, dst_d, rows, scale);
                return true;
            case 2048:
                ggml_cuda_kernel_launch(fwht_cuda<2048, float>, launch_params, src_f32, dst_d, rows, scale);
                return true;
            default:
                return false;
        }
    }

    const half * src_f16 = (const half *) src_d;
    switch (n) {
        case 64:
            ggml_cuda_kernel_launch(fwht_cuda<64, half>, launch_params, src_f16, dst_d, rows, scale);
            return true;
        case 128:
            ggml_cuda_kernel_launch(fwht_cuda<128, half>, launch_params, src_f16, dst_d, rows, scale);
            return true;
        case 256:
            ggml_cuda_kernel_launch(fwht_cuda<256, half>, launch_params, src_f16, dst_d, rows, scale);
            return true;
        case 512:
            ggml_cuda_kernel_launch(fwht_cuda<512, half>, launch_params, src_f16, dst_d, rows, scale);
            return true;
        case 1024:
            ggml_cuda_kernel_launch(fwht_cuda<1024, half>, launch_params, src_f16, dst_d, rows, scale);
            return true;
        case 2048:
            ggml_cuda_kernel_launch(fwht_cuda<2048, half>, launch_params, src_f16, dst_d, rows, scale);
            return true;
        default:
            return false;
    }
}

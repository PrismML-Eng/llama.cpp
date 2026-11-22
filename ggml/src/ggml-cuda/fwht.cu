#include "common.cuh"
#include "fwht.cuh"
#include "unary.cuh"

#include <cstdlib>

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

// Large-N path. The register kernel above keeps N/warp_size floats per thread, so it stops being
// viable well before the arithmetic does: N=4096 would need 128 registers per thread and spill,
// which is why the switch below used to end at 2048 and simply decline anything larger (the whole
// op then fell back to CPU). Stage the row in shared memory instead and run all log2(N) butterfly
// stages there. One row per block; every thread handles several butterflies per stage. Slower per
// row than the register path, so it is used only where that path cannot go.
#define FWHT_SMEM_THREADS 256

template <int N, typename T, bool has_signs>
__launch_bounds__(FWHT_SMEM_THREADS, 1)
__global__ void fwht_cuda_smem(const T * src, float * dst, const int64_t n_rows, const float scale,
                               const float * signs, const int n_blk) {
    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    ggml_cuda_pdl_sync();
    for (int i = threadIdx.x; i < N; i += FWHT_SMEM_THREADS) {
        float v = fwht_load(src[i]) * scale;
        if (has_signs) {
            v *= signs_row[i];
        }
        s[i] = v;
    }
    __syncthreads();

    // Same butterfly and the same sign convention as the register path: the low element of a pair
    // takes x + y, the high one x - y.
#pragma unroll 1
    for (int h = 1; h < N; h *= 2) {
        for (int idx = threadIdx.x; idx < N / 2; idx += FWHT_SMEM_THREADS) {
            const int j = ((idx / h) * 2 * h) + (idx % h);
            const float x = s[j];
            const float y = s[j + h];
            s[j]     = x + y;
            s[j + h] = x - y;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < N; i += FWHT_SMEM_THREADS) {
        dst[i] = s[i];
    }
}


// Wide rows at small row counts (decode): one row per block instead of per warp.
// The warp kernel serialises every stage on one warp, and the shared-memory kernel above synchronises on each stage.
// Both leave most of the GPU idle at these shapes.
#define FWHT_BLOCK_THREADS 256

template <int N, int NT, typename T, bool has_signs>
__launch_bounds__(NT, 1)
__global__ void fwht_cuda_block(const T * src, float * dst, const int64_t n_rows, const float scale,
                                const float * signs, const int n_blk) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    ggml_cuda_pdl_sync();
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = fwht_load(src[i * NT + tid]) * scale;
        if (has_signs) {
            reg[i] *= signs_row[i * NT + tid];
        }
    }

    ggml_cuda_fwht_block_butterfly<N, NT>(reg, s, tid, lane);

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
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
        default:
            break;
    }
    // From 512 up, one block of FWHT_BLOCK_THREADS per row (fwht_cuda_block).
    // The older kernels were the largest single kernel of a decode step at these widths.
    // GGML_CUDA_FWHT_LEGACY=1 restores them for A/B.
#define FWHT_SMEM_CASE(NN) \
        case NN: { \
            const dim3 g((unsigned) rows, 1, 1), b(FWHT_SMEM_THREADS, 1, 1); \
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, stream); \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda_smem<NN, T, true>,  lp, src_d, dst_d, rows, scale, signs, n_blk); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda_smem<NN, T, false>, lp, src_d, dst_d, rows, scale, nullptr, 1); \
            } \
            return true; \
        }
#define FWHT_BLOCK_LAUNCH(NN, NTT) { \
            const dim3 g((unsigned) rows, 1, 1), b(NTT, 1, 1); \
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, stream); \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda_block<NN, NTT, T, true>,  lp, src_d, dst_d, rows, scale, signs, n_blk); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda_block<NN, NTT, T, false>, lp, src_d, dst_d, rows, scale, nullptr, 1); \
            } \
            return true; \
        }
    const int nt = ggml_cuda_fwht_block_threads(n, rows);
#define FWHT_BLOCK_CASE(NN) \
        case NN: { \
            if constexpr (NN >= 1024) { \
                if (nt == 1024) FWHT_BLOCK_LAUNCH(NN, 1024) \
            } \
            if constexpr (NN >= 512) { \
                if (nt == 512) FWHT_BLOCK_LAUNCH(NN, 512) \
            } \
            FWHT_BLOCK_LAUNCH(NN, FWHT_BLOCK_THREADS) \
        }
    static const bool legacy = getenv("GGML_CUDA_FWHT_LEGACY") != nullptr;
    if (legacy) {
        switch (n) {
            FWHT_CASE(512)
            FWHT_CASE(1024)
            FWHT_CASE(2048)
            FWHT_SMEM_CASE(4096)
#if !defined(GGML_USE_MUSA)
            // 32 KB of shared memory, above the MUSA limit; falls back there
            FWHT_SMEM_CASE(8192)
#endif // !defined(GGML_USE_MUSA)
            default:
                return false;
        }
    }
    switch (n) {
        FWHT_BLOCK_CASE(512)
        FWHT_BLOCK_CASE(1024)
        FWHT_BLOCK_CASE(2048)
        FWHT_BLOCK_CASE(4096)
#if !defined(GGML_USE_MUSA)
        // 32 KB of shared memory, above the MUSA limit; falls back there
        FWHT_BLOCK_CASE(8192)
#endif // !defined(GGML_USE_MUSA)
#undef FWHT_CASE
#undef FWHT_SMEM_CASE
#undef FWHT_BLOCK_CASE
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

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op) {
    const struct ggml_tensor * a = op->src[0];
    const struct ggml_tensor * b = op->src[1];

    return op->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
           a->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           (b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16) && ggml_is_contiguous(b) && ggml_is_contiguous(op) &&
           ggml_are_same_shape(b, op);
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    return fwht_dispatch(ctx, src, dst, nullptr);
}

bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst) {
    return fwht_dispatch(ctx, src, dst, signs);
}


// block_q8_1_mmq with D4 scales (mmq.cuh), restated to keep this file free of the MMQ headers
struct fwht_q8_1_mmq_d4 {
    float  d4[4];
    int8_t qs[128];
};
static_assert(sizeof(fwht_q8_1_mmq_d4) == 144, "block_q8_1_mmq size");

// GGML_CUDA_FWHT_MMQ: fwht_cuda_block (has_signs as there) followed by quantize_mmq_q8_1<D4> of its output, in one
// launch. Row r of the [rows, N] view is column col = r*N / K, values k = (r*N) % K + i*NT + tid. A 32-value
// quantization group is one warp at one i (N, NT and r*N are multiples of 32), so the group's amax is a warp max.
template <int N, int NT, typename T, bool has_signs, bool glu = false>
__launch_bounds__(NT, 1)
__global__ void fwht_quantize_mmq_d4_block(const T * src, fwht_q8_1_mmq_d4 * y, const int64_t n_rows, const float scale,
                                           const float * signs, const int n_blk, const int64_t K, const int64_t ncols,
                                           const float * glu_up = nullptr) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    if (glu) {
        glu_up += r * N;
    }

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    ggml_cuda_pdl_lc(); // as unary_gated_op_kernel: the next launch may start its prologue (it waits in pdl_sync)
    ggml_cuda_pdl_sync();
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        if (glu) {
            // the GLU value as unary_gated_op_kernel<op_silu, float> stores it
            const float v = ggml_cuda_op_silu_single(fwht_load(src[i * NT + tid])) * glu_up[i * NT + tid];
            reg[i] = v * scale;
        } else {
            reg[i] = fwht_load(src[i * NT + tid]) * scale;
        }
        if (has_signs) {
            reg[i] *= signs_row[i * NT + tid];
        }
    }

    ggml_cuda_fwht_block_butterfly<N, NT>(reg, s, tid, lane);

    // through shared memory to 4 consecutive values per thread, then quantize_mmq_q8_1<D4> as written there:
    // amax over the 4 values and 8 threads (xor 4, 2, 1), char4 stores, one d per 32 values
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        s[i * NT + tid] = reg[i];
    }
    __syncthreads();
    const int64_t e0  = r * N;
    const int64_t col = e0 / K;
    const int64_t k0  = e0 % K;
    for (int e = tid; e < N/4; e += NT) {
        const float4 xi = ((const float4 *) s)[e];
        float amax = fabsf(xi.x);
        amax = fmaxf(amax, fabsf(xi.y));
        amax = fmaxf(amax, fabsf(xi.z));
        amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
        for (int offset = 32/8; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, warp_size));
        }
        const float d_inv = 127.0f / amax;
        char4 q;
        q.x = roundf(xi.x*d_inv);
        q.y = roundf(xi.y*d_inv);
        q.z = roundf(xi.z*d_inv);
        q.w = roundf(xi.w*d_inv);
        const float d = 1.0f / d_inv;
        const int64_t k   = k0 + 4*e;
        const int64_t ib  = (k / 128) * ncols + col;
        const int     iqs = (int) (k % 128);
        ((char4 *) y[ib].qs)[iqs/4] = q;
        if (iqs % 32 == 0) {
            y[ib].d4[iqs/32] = d;
        }
    }
}

bool ggml_cuda_fwht_quantize_mmq_d4(const void * x, ggml_type xt, const float * signs, int n, void * out,
                                    int64_t K, int64_t ncols, cudaStream_t stream, const float * glu_up) {
    if (K % n != 0 || K % 128 != 0 || (xt != GGML_TYPE_F32 && xt != GGML_TYPE_F16)) {
        return false;
    }
    const int64_t rows  = K / n * ncols;
    const float   scale = 1 / sqrtf(n);
    const int     n_blk = (int) (K / n);
    if (ggml_cuda_fwht_block_threads(n, rows) != FWHT_BLOCK_THREADS) {
        return false;
    }
    const dim3 g((unsigned) rows, 1, 1), b(FWHT_BLOCK_THREADS, 1, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, stream);
    fwht_q8_1_mmq_d4 * y = (fwht_q8_1_mmq_d4 *) out;
#define FQM_CASE(NN) \
    case NN: \
        if (xt == GGML_TYPE_F32) { \
            if (signs) ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_block<NN, FWHT_BLOCK_THREADS, float, true>,  lp, (const float *) x, y, rows, scale, signs, n_blk, K, ncols); \
            else       ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_block<NN, FWHT_BLOCK_THREADS, float, false>, lp, (const float *) x, y, rows, scale, nullptr, 1, K, ncols); \
        } else { \
            if (signs) ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_block<NN, FWHT_BLOCK_THREADS, half, true>,   lp, (const half *) x, y, rows, scale, signs, n_blk, K, ncols); \
            else       ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_block<NN, FWHT_BLOCK_THREADS, half, false>,  lp, (const half *) x, y, rows, scale, nullptr, 1, K, ncols); \
        } \
        return true;
    if (!glu_up) {
        // the non-GLU instantiations take glu_up = nullptr explicitly (the launch helper does not apply defaults)
        switch (n) {
#define FQN_CASE(NN) case NN: \
            if (xt == GGML_TYPE_F32) { \
                if (signs) ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_block<NN, FWHT_BLOCK_THREADS, float, true>,  lp, (const float *) x, y, rows, scale, signs, n_blk, K, ncols, (const float *) nullptr); \
                else       ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_block<NN, FWHT_BLOCK_THREADS, float, false>, lp, (const float *) x, y, rows, scale, (const float *) nullptr, 1, K, ncols, (const float *) nullptr); \
            } else { \
                if (signs) ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_block<NN, FWHT_BLOCK_THREADS, half, true>,   lp, (const half *) x, y, rows, scale, signs, n_blk, K, ncols, (const float *) nullptr); \
                else       ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_block<NN, FWHT_BLOCK_THREADS, half, false>,  lp, (const half *) x, y, rows, scale, (const float *) nullptr, 1, K, ncols, (const float *) nullptr); \
            } \
            return true;
            FQN_CASE(512)
            FQN_CASE(1024)
            FQN_CASE(2048)
            FQN_CASE(4096)
#undef FQN_CASE
            default:
                return false;
        }
    }
    {
        if (xt != GGML_TYPE_F32 || !signs) {
            return false;
        }
        switch (n) {
#define FQG_CASE(NN) case NN: ggml_cuda_kernel_launch(fwht_quantize_mmq_d4_block<NN, FWHT_BLOCK_THREADS, float, true, true>, lp, \
                (const float *) x, y, rows, scale, signs, n_blk, K, ncols, glu_up); return true;
            FQG_CASE(512)
            FQG_CASE(1024)
            FQG_CASE(2048)
            FQG_CASE(4096)
#undef FQG_CASE
            default:
                return false;
        }
    }
    return false;
#undef FQM_CASE
}

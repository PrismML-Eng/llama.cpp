#include "moe-reduce.cuh"

// Fused MoE router-weighting + expert reduction.
//
// build_moe_ffn emits, after the down projection:
//     experts  = ggml_mul(down_out, weights)          // [n_embd, n_expert_used, n_tokens]
//     view_e   = ggml_view_2d(experts, ..., e)        // n_expert_used views
//     moe_out  = (((view_0 + view_1) + view_2) + ...) // n_expert_used - 1 adds
//
// The ggml_mul materialises a second full-size [n_embd, n_expert_used, n_tokens] F32
// tensor (tens of MB per layer at large ubatch) purely so that the add chain
// can read it back. This kernel folds the weighting into the reduction, so the weighted
// intermediate is never written or read: only down_out is streamed in and the
// [n_embd, n_tokens] result is streamed out.
//
// Numerics are bit-identical to the unfused pair. GGML_OP_MUL rounds each product to F32
// before storing it, and the fused multi-add folds the slices left to right, so the fused
// kernel must round every product before accumulating and must accumulate in increasing
// expert-slot order. __fmul_rn / __fadd_rn keep nvcc from contracting a product and its
// accumulation into an FMA, which would be more accurate but would not match the baseline.

template <int n_exp>
static __global__ void k_moe_weighted_reduce(const float * __restrict__ x,
                                             const float * __restrict__ w,
                                             float * __restrict__ dst,
                                             const int     ne0,
                                             const int64_t sx1,
                                             const int64_t sx2,
                                             const int64_t sw1,
                                             const int64_t sw2,
                                             const int64_t sd1) {
    const int64_t i2 = blockIdx.y;

    const float * x_t = x + i2 * sx2;
    const float * w_t = w + i2 * sw2;
    float *       d_t = dst + i2 * sd1;

    float wv[n_exp];
#pragma unroll
    for (int e = 0; e < n_exp; ++e) {
        wv[e] = w_t[e * sw1];
    }

    for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < ne0; i0 += blockDim.x * gridDim.x) {
        float acc = __fmul_rn(x_t[i0], wv[0]);
#pragma unroll
        for (int e = 1; e < n_exp; ++e) {
            acc = __fadd_rn(acc, __fmul_rn(x_t[i0 + e * sx1], wv[e]));
        }
        d_t[i0] = acc;
    }
}

template <int n_exp>
static __global__ void k_moe_weighted_reduce_v4(const float4 * __restrict__ x,
                                                const float * __restrict__ w,
                                                float4 * __restrict__ dst,
                                                const int     ne0v,
                                                const int64_t sx1v,
                                                const int64_t sx2v,
                                                const int64_t sw1,
                                                const int64_t sw2,
                                                const int64_t sd1v) {
    const int64_t i2 = blockIdx.y;

    const float4 * x_t = x + i2 * sx2v;
    const float *  w_t = w + i2 * sw2;
    float4 *       d_t = dst + i2 * sd1v;

    float wv[n_exp];
#pragma unroll
    for (int e = 0; e < n_exp; ++e) {
        wv[e] = w_t[e * sw1];
    }

    for (int i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < ne0v; i0 += blockDim.x * gridDim.x) {
        const float4 x0  = x_t[i0];
        float4       acc = make_float4(__fmul_rn(x0.x, wv[0]), __fmul_rn(x0.y, wv[0]),
                                       __fmul_rn(x0.z, wv[0]), __fmul_rn(x0.w, wv[0]));
#pragma unroll
        for (int e = 1; e < n_exp; ++e) {
            const float4 xe = x_t[i0 + e * sx1v];
            acc.x = __fadd_rn(acc.x, __fmul_rn(xe.x, wv[e]));
            acc.y = __fadd_rn(acc.y, __fmul_rn(xe.y, wv[e]));
            acc.z = __fadd_rn(acc.z, __fmul_rn(xe.z, wv[e]));
            acc.w = __fadd_rn(acc.w, __fmul_rn(xe.w, wv[e]));
        }
        d_t[i0] = acc;
    }
}

static constexpr int MOE_REDUCE_BLOCK = 256;

template <int n_exp>
static void launch_moe_weighted_reduce(const float * x,
                                       const float * w,
                                       float *       dst,
                                       const int64_t ne0,
                                       const int64_t n_tok,
                                       const int64_t sx1,
                                       const int64_t sx2,
                                       const int64_t sw1,
                                       const int64_t sw2,
                                       const int64_t sd1,
                                       cudaStream_t  stream) {
    const bool use_v4 = ne0 % 4 == 0 && sx1 % 4 == 0 && sx2 % 4 == 0 && sd1 % 4 == 0 &&
                        ((uintptr_t) x) % sizeof(float4) == 0 && ((uintptr_t) dst) % sizeof(float4) == 0;

    if (use_v4) {
        const int64_t ne0v = ne0 / 4;
        const dim3    grid((ne0v + MOE_REDUCE_BLOCK - 1) / MOE_REDUCE_BLOCK, n_tok, 1);
        k_moe_weighted_reduce_v4<n_exp><<<grid, MOE_REDUCE_BLOCK, 0, stream>>>(
            (const float4 *) x, w, (float4 *) dst, ne0v, sx1 / 4, sx2 / 4, sw1, sw2, sd1 / 4);
    } else {
        const dim3 grid((ne0 + MOE_REDUCE_BLOCK - 1) / MOE_REDUCE_BLOCK, n_tok, 1);
        k_moe_weighted_reduce<n_exp><<<grid, MOE_REDUCE_BLOCK, 0, stream>>>(
            x, w, dst, ne0, sx1, sx2, sw1, sw2, sd1);
    }
}

void ggml_cuda_op_moe_weighted_reduce(ggml_backend_cuda_context & ctx,
                                      const ggml_tensor *         src0,
                                      const ggml_tensor *         src1,
                                      ggml_tensor *               dst,
                                      int                         n_exp,
                                      bool                        stage_weights) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(n_exp >= 2 && n_exp <= GGML_CUDA_MOE_REDUCE_MAX_EXPERTS);

    const int64_t ne0   = src0->ne[0];
    const int64_t n_tok = src0->ne[2];

    const int64_t sx1 = src0->nb[1] / sizeof(float);
    const int64_t sx2 = src0->nb[2] / sizeof(float);
    int64_t       sw1 = src1->nb[1] / sizeof(float);
    int64_t       sw2 = src1->nb[2] / sizeof(float);
    const int64_t sd1 = dst->nb[1] / sizeof(float);

    const float * x = (const float *) src0->data;
    const float * w = (const float *) src1->data;
    float *       d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    // dst aliases the router weights, so snapshot them before the reduce starts writing.
    // n_expert_used*n_tokens floats, a few KiB at typical shapes.
    ggml_cuda_pool_alloc<float> w_stage(ctx.pool());

    if (stage_weights) {
        GGML_ASSERT(ggml_is_contiguous(src1));

        const size_t n_w = ggml_nelements(src1);
        w_stage.alloc(n_w);

        CUDA_CHECK(cudaMemcpyAsync(w_stage.ptr, w, n_w * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        w   = w_stage.ptr;
        sw1 = 1;
        sw2 = n_exp;
    }

#define MOE_REDUCE_CASE(N)                                                                      \
    case N:                                                                                     \
        launch_moe_weighted_reduce<N>(x, w, d, ne0, n_tok, sx1, sx2, sw1, sw2, sd1, stream);    \
        break;

    switch (n_exp) {
        MOE_REDUCE_CASE(2)
        MOE_REDUCE_CASE(3)
        MOE_REDUCE_CASE(4)
        MOE_REDUCE_CASE(5)
        MOE_REDUCE_CASE(6)
        MOE_REDUCE_CASE(7)
        MOE_REDUCE_CASE(8)
        MOE_REDUCE_CASE(9)
        MOE_REDUCE_CASE(10)
        MOE_REDUCE_CASE(11)
        MOE_REDUCE_CASE(12)
        MOE_REDUCE_CASE(13)
        MOE_REDUCE_CASE(14)
        MOE_REDUCE_CASE(15)
        default:
            GGML_ABORT("fatal error");
    }

#undef MOE_REDUCE_CASE

    CUDA_CHECK(cudaGetLastError());
}

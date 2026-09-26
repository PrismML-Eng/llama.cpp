#include "w8a8.hpp"

#if GGML_SYCL_DNNL
#include "dequantize.hpp"
#include "gemm.hpp"

// per row: D = max over groups of d, ratio q_g = round(127 * d_g / D); one work-group (16 lanes) per row
static void w8a8_row_scales(ggml_type type, bool reordered, const void * w, float * row_scale, int8_t * ratio,
                            int64_t nrows, int64_t K, dpct::queue_ptr stream) {
    const int64_t   ng   = K / 128;
    const int64_t   nb   = nrows * ng;
    const uint8_t * base = (const uint8_t *) w;
    const bool      ptq  = type == GGML_TYPE_PTQ1_0;
    stream->parallel_for(sycl::nd_range<1>(nrows * 16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t r    = it.get_group(0);
        const int     lane = it.get_local_id(0);
        auto dget = [&](int64_t g) -> float {
            const int64_t ib = r * ng + g;
            if (ptq) {
                return reordered ? (float) ((const sycl::half *) (base + nb * 26))[ib] : (float) ((const block_ptq1_0 *) w)[ib].d;
            }
            return reordered ? (float) ((const sycl::half *) (base + nb * 32))[ib] : (float) ((const block_pq2_0 *) w)[ib].d;
        };
        float m = 0.0f;
        for (int64_t g = lane; g < ng; g += 16) {
            m = sycl::fmax(m, sycl::fabs(dget(g)));
        }
        m = sycl::reduce_over_group(it.get_sub_group(), m, sycl::maximum<float>());
        for (int64_t g = lane; g < ng; g += 16) {
            ratio[r * ng + g] = m > 0 ? (int8_t) sycl::round(127.0f * dget(g) / m) : 0;
        }
        if (lane == 0) {
            row_scale[r] = m / 127.0f;
        }
    });
}

// int8 weights = (code value) * ratio of its group; 8 elements per work item
static void w8a8_weights(ggml_type type, bool reordered, const void * w, const int8_t * ratio, int8_t * w8,
                         int64_t nrows, int64_t K, dpct::queue_ptr stream) {
    const int64_t   nb   = nrows * K / 128;
    const uint8_t * base = (const uint8_t *) w;
    const bool      ptq  = type == GGML_TYPE_PTQ1_0;
    stream->parallel_for(sycl::range<1>(nb * 16), [=](sycl::id<1> id) {
        const int64_t ib = id[0] / 16;
        const int     c  = (int) (id[0] % 16);
        int           v[8];
        if (ptq) {
            const uint8_t * qs;
            const uint8_t * qh;
            if (reordered) {
                qs = base + ib * 24;
                qh = base + nb * 24 + ib * 2;
            } else {
                qs = ((const block_ptq1_0 *) w)[ib].qs;
                qh = ((const block_ptq1_0 *) w)[ib].qh;
            }
            constexpr uint32_t pow3[5] = { 1, 3, 9, 27, 81 };
            if (c < 15) {
                // 8 source bytes in one load: qs[0..7], qs[8..15] (c < 10) or qs[16..23]
                const uint32_t p   = pow3[c < 10 ? c / 2 : c - 10];
                const int      off = c < 10 ? (c % 2) * 8 : 16;
                const uint64_t src = reordered ? *(const uint64_t *) (qs + off) : 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t byte = reordered ? (uint32_t) (src >> (8 * i)) & 0xFF : qs[off + i];
                    v[i] = (int) ((((byte * p) & 0xFF) * 3) >> 8) - 1;
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) v[i] = (int) ((((qh[i % 2] * pow3[i / 2]) & 0xFF) * 3) >> 8) - 1;
            }
        } else {
            const uint8_t * qs  = reordered ? base + ib * 32 : ((const block_pq2_0 *) w)[ib].qs;
            const uint32_t  two = reordered ? *(const uint16_t *) (qs + 2 * c) : (uint32_t) qs[2 * c] | ((uint32_t) qs[2 * c + 1] << 8);
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = (int) ((two >> (2 * i)) & 3) - 1;
        }
        const int r8    = ratio[ib];
        uint64_t  out   = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) out |= (uint64_t) (uint8_t) (int8_t) (v[i] * r8) << (8 * i);
        ((uint64_t *) w8)[id[0]] = out;
    });
}

// per token: s = max|x| / 127, x8 = round(x / s); one 256-item work-group per token row, 16-byte loads
static void w8a8_acts(const float * x, int8_t * x8, float * xs, int64_t ntok, int64_t K, dpct::queue_ptr stream) {
    constexpr int WG = 256;
    GGML_ASSERT(K % 4 == 0);
    stream->parallel_for(sycl::nd_range<1>(ntok * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t        t    = it.get_group(0);
        const int            lid  = it.get_local_id(0);
        const int64_t        K4   = K / 4;
        const sycl::float4 * row4 = (const sycl::float4 *) (x + t * K);
        float                m    = 0.0f;
        for (int64_t k4 = lid; k4 < K4; k4 += WG) {
            const sycl::float4 v = sycl::fabs(row4[k4]);
            m = sycl::fmax(m, sycl::fmax(sycl::fmax(v.x(), v.y()), sycl::fmax(v.z(), v.w())));
        }
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        const float inv = m > 0 ? 127.0f / m : 0.0f;
        sycl::char4 * out4 = (sycl::char4 *) (x8 + t * K);
        for (int64_t k4 = lid; k4 < K4; k4 += WG) {
            const sycl::float4 v = sycl::round(row4[k4] * inv);
            out4[k4] = sycl::char4((int8_t) v.x(), (int8_t) v.y(), (int8_t) v.z(), (int8_t) v.w());
        }
        if (lid == 0) {
            xs[t] = m / 127.0f;
        }
    });
}

bool ggml_sycl_w8a8_mul_mat(ggml_backend_sycl_context & ctx, ggml_type type, bool reordered, const void * w,
                            const float * x, float * dst, int64_t nrows, int64_t ncols_x, int64_t K,
                            dpct::queue_ptr stream) {
    if ((type != GGML_TYPE_PQ2_0 && type != GGML_TYPE_PTQ1_0) || K % 128 != 0) {
        return false;
    }
    ggml_sycl_pool_alloc<int8_t> ratio(ctx.pool(), nrows * K / 128);
    ggml_sycl_pool_alloc<float>  wscale(ctx.pool(), nrows);
    ggml_sycl_pool_alloc<int8_t> w8(ctx.pool(), nrows * K);
    ggml_sycl_pool_alloc<int8_t> x8(ctx.pool(), ncols_x * K);
    ggml_sycl_pool_alloc<float>  xscale(ctx.pool(), ncols_x);

    w8a8_row_scales(type, reordered, w, wscale.get(), ratio.get(), nrows, K, stream);
    w8a8_weights(type, reordered, w, ratio.get(), w8.get(), nrows, K, stream);
    w8a8_acts(x, x8.get(), xscale.get(), ncols_x, K, stream);
    DnnlGemmWrapper::gemm_s8(ctx, (int) nrows, (int) ncols_x, (int) K, x8.get(), w8.get(), dst, stream);

    const float * ws = wscale.get();
    const float * xs = xscale.get();
    stream->parallel_for(sycl::range<1>(ncols_x * nrows), [=](sycl::id<1> i) {
        dst[i] *= xs[i[0] / nrows] * ws[i[0] % nrows];
    });
    return true;
}

#else

bool ggml_sycl_w8a8_mul_mat(ggml_backend_sycl_context &, ggml_type, bool, const void *, const float *, float *, int64_t,
                            int64_t, int64_t, dpct::queue_ptr) {
    return false;
}

#endif

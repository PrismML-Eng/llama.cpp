#include "pq2_xmx.hpp"
#include "dequantize.hpp"

// GGML_SYCL_NO_PQ2_XMX: an AOT build for a device without 16-wide DPAS, which cannot compile these kernels
#if defined(__INTEL_LLVM_COMPILER) && !defined(GGML_SYCL_NO_PQ2_XMX)
#include <sycl/ext/intel/esimd.hpp>

namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;

constexpr int PQ2_XMX_QS_BYTES = QK_PQ2_0 / 4;    // 32 bytes of 2-bit codes per block
constexpr int PQ2_XMX_QS_DW    = PQ2_XMX_QS_BYTES / 4;
constexpr int PQ2_XMX_WG       = 16;               // independent tiles per work-group when K is not split

// thread count the K split aims for: the B50 runs 1024 hardware threads, and decode needs several in flight
// per EU to keep enough loads outstanding
constexpr int PQ2_XMX_TARGET_THREADS         = 4096;
constexpr int PQ2_XMX_PREFILL_TARGET_THREADS = 512;

static_assert(QK_PQ2_0 == 128 && sizeof(block_pq2_0) == 34, "PQ2_0 layout changed");
static_assert(QK_PTQ1_0 == QK_PQ2_0, "PTQ1_0 expands block for block into PQ2_0 codes");

// PQ2_0 packs value+1 per 2-bit field, lowest first: as a little-endian dword that is already DPAS 2-bit packing.
// Subtracting 1 per field without borrow gives the s2 values DPAS multiplies.
template <int N>
ESIMD_INLINE esimd::simd<uint32_t, N> pq2_codes_to_s2(esimd::simd<uint32_t, N> x) {
    constexpr uint32_t H = 0xAAAAAAAAu;
    constexpr uint32_t L = 0x55555555u;
    return ((x | H) - L) ^ (~x & H);
}

// Each thread computes an (8*MR) token x (16*NR) row tile over its share of the K blocks. With S > 1 the S
// threads of a work-group split K for the same tile and reduce through SLM, so a mat-vec keeps enough threads
// streaming weights. One DPAS covers 8 tokens x 16 rows x 32 k; the four of a block accumulate in int32.
template <int MR, int NR, int S>
ESIMD_INLINE void pq2_xmx_thread(const uint32_t * wq, const uint16_t * wd, const uint32_t * a8, const float * as,
                                 float * dst, int K, int nrows, int ncols, int nrows_dst, int n_tiles_n,
                                 int tile, int ks) {
    using namespace esimd;
    constexpr int TM = 8 * MR;
    constexpr int TN = 16 * NR;

    if constexpr (S > 1) {
        slm_init<S * MR * NR * 128 * sizeof(float)>();
    }

    const int m0 = (tile / n_tiles_n) * TM;
    const int n0 = (tile % n_tiles_n) * TN;
    const int nb = K / QK_PQ2_0;
    const int b0 = (int) ((int64_t) ks * nb / S);
    const int b1 = (int) ((int64_t) (ks + 1) * nb / S);

    // weights: nrows rows of nb*32 bytes; activations: ncols rows of K bytes. Rows past either read as zeros.
    const uint32_t wsurf_w = (uint32_t) (nb * PQ2_XMX_QS_BYTES) - 1;
    const uint32_t wsurf_h = (uint32_t) nrows - 1;
    const uint32_t asurf_w = (uint32_t) K - 1;
    const uint32_t asurf_h = (uint32_t) ncols - 1;

    // scale gathers clamp to a valid row/token; the clamped lanes are never stored
    simd<uint32_t, 16> d_off[NR];
#pragma unroll
    for (int g = 0; g < NR; ++g) {
        simd<uint32_t, 16> r(n0 + 16 * g, 1);
        r.merge(simd<uint32_t, 16>(nrows - 1), r >= (uint32_t) nrows);
        d_off[g] = r * (uint32_t) (nb * sizeof(uint16_t));
    }
    simd<uint32_t, 8> s_off[MR];
#pragma unroll
    for (int s = 0; s < MR; ++s) {
        simd<uint32_t, 8> m(m0 + 8 * s, 1);
        m.merge(simd<uint32_t, 8>(ncols - 1), m >= (uint32_t) ncols);
        s_off[s] = m * (uint32_t) (nb * sizeof(float));
    }

    simd<float, 128> acc[MR][NR];
#pragma unroll
    for (int s = 0; s < MR; ++s) {
#pragma unroll
        for (int g = 0; g < NR; ++g) {
            acc[s][g] = 0.0f;
        }
    }

    for (int b = b0; b < b1; ++b) {
        // transposed load: w[g][j*16 + n] = dword j (k = 16j..16j+15) of row n, the DPAS B layout for 2 dwords per k32
        simd<uint32_t, 128> w[NR];
        simd<float, 16>     dw[NR];
#pragma unroll
        for (int g = 0; g < NR; ++g) {
            w[g] = pq2_codes_to_s2<128>(load_2d<uint32_t, PQ2_XMX_QS_DW, 16, 1, true, false>(
                wq, wsurf_w, wsurf_h, wsurf_w, b * PQ2_XMX_QS_DW, n0 + 16 * g));
            simd<uint16_t, 16>    dbits = gather<uint16_t, 16>(wd, d_off[g] + (uint32_t) (b * sizeof(uint16_t)));
            simd<sycl::half, 16>  dh    = dbits.template bit_cast_view<sycl::half>();
            dw[g] = convert<float>(dh);
        }

#pragma unroll
        for (int s = 0; s < MR; ++s) {
            const int yrow = m0 + 8 * s;

            simd<int, 128> ci[NR];
#pragma unroll
            for (int g = 0; g < NR; ++g) {
                ci[g] = 0;
            }
#pragma unroll
            for (int c = 0; c < QK_PQ2_0 / 32; ++c) {
                // A operand: token t's 32 int8 values at dwords t*8 .. t*8+7
                simd<uint32_t, 64>     ad = load_2d<uint32_t, 8, 8, 1, false, false>(a8, asurf_w, asurf_h, asurf_w,
                                                                                    b * 32 + 8 * c, yrow);
                simd<signed char, 256> am = ad.template bit_cast_view<signed char>();
#pragma unroll
                for (int g = 0; g < NR; ++g) {
                    simd<uint32_t, 32>     bd = w[g].template select<32, 1>(32 * c);
                    simd<signed char, 128> bm = bd.template bit_cast_view<signed char>();
                    ci[g] = xmx::dpas<8, 8, int, int, signed char, signed char, xmx::dpas_argument_type::s2,
                                      xmx::dpas_argument_type::s8>(ci[g], bm, am);
                }
            }

            const simd<float, 8> da = gather<float, 8>(as, s_off[s] + (uint32_t) (b * sizeof(float)));
#pragma unroll
            for (int g = 0; g < NR; ++g) {
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const simd<int, 16> cit = ci[g].template select<16, 1>(16 * t);
                    acc[s][g].template select<16, 1>(16 * t) += convert<float>(cit) * (dw[g] * da[t]);
                }
            }
        }
    }

    if constexpr (S > 1) {
        // every thread parks its partial tile in SLM; thread 0 sums them and stores
#pragma unroll
        for (int s = 0; s < MR; ++s) {
#pragma unroll
            for (int g = 0; g < NR; ++g) {
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const uint32_t off = (uint32_t) ((((ks * MR + s) * NR + g) * 128 + 16 * q) * sizeof(float));
                    slm_block_store<float, 16>(off, acc[s][g].template select<16, 1>(16 * q));
                }
            }
        }
        barrier();
        if (ks != 0) {
            return;
        }
        for (int o = 1; o < S; ++o) {
#pragma unroll
            for (int s = 0; s < MR; ++s) {
#pragma unroll
                for (int g = 0; g < NR; ++g) {
#pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        const uint32_t off = (uint32_t) ((((o * MR + s) * NR + g) * 128 + 16 * q) * sizeof(float));
                        acc[s][g].template select<16, 1>(16 * q) += slm_block_load<float, 16>(off);
                    }
                }
            }
        }
    }

    const simd<uint32_t, 16> lane(0, 1);
#pragma unroll
    for (int s = 0; s < MR; ++s) {
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int m = m0 + 8 * s + t;
            if (m >= ncols) {
                continue;
            }
            // row base in 64 bits: an output head at a large ubatch passes 4 GB
            float * drow = dst + (size_t) m * nrows_dst;
#pragma unroll
            for (int g = 0; g < NR; ++g) {
                const simd<uint32_t, 16> n  = lane + (uint32_t) (n0 + 16 * g);
                const simd_mask<16>      ok = n < (uint32_t) nrows;
                scatter<float, 16>(drow, n * (uint32_t) sizeof(float), acc[s][g].template select<16, 1>(16 * t), ok);
            }
        }
    }
}

template <int MR, int NR, int S>
static void launch_pq2_xmx(const uint32_t * wq, const uint16_t * wd, const uint32_t * a8, const float * as,
                           float * dst, int K, int nrows, int ncols, int nrows_dst, dpct::queue_ptr stream) {
    constexpr int TM = 8 * MR;
    constexpr int TN = 16 * NR;

    const int n_tiles_m = (ncols + TM - 1) / TM;
    const int n_tiles_n = (nrows + TN - 1) / TN;
    const int n_tiles   = n_tiles_m * n_tiles_n;

    stream->submit([&](sycl::handler & h) {
        if constexpr (S > 1) {
            const sycl::nd_range<1> nd{ sycl::range<1>((size_t) n_tiles * S), sycl::range<1>(S) };
            h.parallel_for(nd, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                pq2_xmx_thread<MR, NR, S>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, n_tiles_n,
                                          (int) it.get_group(0), (int) it.get_local_id(0));
            });
        } else {
            const size_t global = (size_t) ((n_tiles + PQ2_XMX_WG - 1) / PQ2_XMX_WG) * PQ2_XMX_WG;
            const sycl::nd_range<1> nd{ sycl::range<1>(global), sycl::range<1>(PQ2_XMX_WG) };
            h.parallel_for(nd, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                const int tile = (int) it.get_global_id(0);
                if (tile < n_tiles) {
                    pq2_xmx_thread<MR, NR, 1>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, n_tiles_n, tile, 0);
                }
            });
        }
    });
}

// K split for a tile count: the smallest power of two reaching the thread target, at least two blocks per thread
template <int MR, int NR>
static void launch_pq2_xmx_split(const uint32_t * wq, const uint16_t * wd, const uint32_t * a8, const float * as,
                                 float * dst, int K, int nrows, int ncols, int nrows_dst, dpct::queue_ptr stream) {
    const int nb      = K / QK_PQ2_0;
    const int n_tiles = ((ncols + 8 * MR - 1) / (8 * MR)) * ((nrows + 16 * NR - 1) / (16 * NR));
    // 32 token tiles (prefill) carry a large SLM reduction, so they split at most in two and only while short
    const int target    = MR >= 4 ? PQ2_XMX_PREFILL_TARGET_THREADS : PQ2_XMX_TARGET_THREADS;
    const int max_split = MR >= 4 ? 2 : 16;
    int       split     = 1;
    while (split < max_split && n_tiles * split < target && nb >= 4 * split) {
        split *= 2;
    }
    switch (split) {
        case 1:  launch_pq2_xmx<MR, NR, 1>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream);  break;
        case 2:  launch_pq2_xmx<MR, NR, 2>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream);  break;
        case 4:  launch_pq2_xmx<MR, NR, 4>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream);  break;
        case 8:  launch_pq2_xmx<MR, NR, 8>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream);  break;
        default: launch_pq2_xmx<MR, NR, 16>(wq, wd, a8, as, dst, K, nrows, ncols, nrows_dst, stream); break;
    }
}

} // namespace

bool ggml_sycl_pq2_xmx_supports_ne0(int64_t ne0) {
    return ne0 % QK_PQ2_0 == 0 && (ne0 / QK_PQ2_0) * PQ2_XMX_QS_BYTES >= 64;
}

bool ggml_sycl_pq2_xmx_reorder(ggml_tensor * src0, dpct::queue_ptr stream) {
    GGML_ASSERT((src0->type == GGML_TYPE_PQ2_0 || src0->type == GGML_TYPE_PTQ1_0) && ggml_is_contiguous(src0));

    const size_t size = ggml_nbytes(src0);
    const size_t nblk = (size_t) ggml_nelements(src0) / QK_PQ2_0;
    uint8_t *    data = (uint8_t *) src0->data;

    void * tmp = sycl::malloc_device(size, *stream);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for the PQ2_0 XMX reorder, skipping it\n", __func__, size);
        return false;
    }
    stream->memcpy(tmp, data, size).wait();

    uint8_t *    qs = data;
    sycl::half * d  = (sycl::half *) (data + nblk * PQ2_XMX_QS_BYTES);
    if (src0->type == GGML_TYPE_PQ2_0) {
        stream->parallel_for(sycl::range<1>(nblk), [=](sycl::id<1> i) {
            const block_pq2_0 * x = (const block_pq2_0 *) tmp + i;
#pragma unroll
            for (int j = 0; j < PQ2_XMX_QS_BYTES; ++j) {
                qs[i * PQ2_XMX_QS_BYTES + j] = x->qs[j];
            }
            d[i] = x->d;
        }).wait();
    } else {
        // base-3 trits (value -1..1) become PQ2_0 codes (value + 1), four to a byte, lowest first;
        // the caller made sure the buffer holds 34 bytes a block
        stream->parallel_for(sycl::range<1>(nblk), [=](sycl::id<1> i) {
            const block_ptq1_0 * x = (const block_ptq1_0 *) tmp + i;
            for (int j = 0; j < PQ2_XMX_QS_BYTES; ++j) {
                uint8_t byte = 0;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    byte |= (uint8_t) ((ptq1_0_trit(x, 4 * j + k) + 1) << (2 * k));
                }
                qs[i * PQ2_XMX_QS_BYTES + j] = byte;
            }
            d[i] = x->d;
        }).wait();
    }

    sycl::free(tmp, *stream);
    return true;
}

void ggml_sycl_pq2_xmx_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst) {
    GGML_ASSERT((src0->type == GGML_TYPE_PQ2_0 || src0->type == GGML_TYPE_PTQ1_0) && src0->ne[2] == 1 &&
                src0->ne[3] == 1);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_sycl_pq2_xmx_supports_ne0(src0->ne[0]));

    const int K     = (int) src0->ne[0];
    const int nrows = (int) src0->ne[1];
    const int nb    = K / QK_PQ2_0;
    const int ne11  = (int) src1->ne[1];
    const int ne12  = (int) src1->ne[2];
    const int ncols = (int) (src1->ne[1] * src1->ne[2] * src1->ne[3]);

    dpct::queue_ptr stream = ctx.stream();

    // int8 activations (ncols rows of K bytes, 64-byte aligned for 2D loads) and one float scale per 128 values
    ggml_sycl_pool_alloc<int8_t> a8_alloc(ctx.pool(), (size_t) ncols * K + 64);
    ggml_sycl_pool_alloc<float>  as_alloc(ctx.pool(), (size_t) ncols * nb);
    int8_t * a8 = (int8_t *) GGML_PAD((uintptr_t) a8_alloc.get(), 64);
    float *  as = as_alloc.get();

    {
        const char * src1_d = (const char *) src1->data;
        const size_t nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
        // one sub-group per (token, 128-block), 8 values per work-item
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) ncols * nb * 16), sycl::range<1>(16)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int grp = (int) it.get_group(0);
                const int j   = grp / nb;
                const int b   = grp % nb;
                const int l   = (int) it.get_local_id(0);
                const int i1  = j % ne11;
                const int i2  = (j / ne11) % ne12;
                const int i3  = j / (ne11 * ne12);

                const float * x = (const float *) (src1_d + i1 * nb11 + i2 * nb12 + i3 * nb13) + b * QK_PQ2_0 + l * 8;
                float v[8];
                float amax = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    v[i] = x[i];
                    amax = sycl::fmax(amax, sycl::fabs(v[i]));
                }
                amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<float>());
                const float d  = amax / 127.0f;
                const float id = d != 0.0f ? 1.0f / d : 0.0f;

                sycl::vec<int8_t, 8> q;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    q[i] = (int8_t) sycl::round(v[i] * id);
                }
                *(sycl::vec<int8_t, 8> *) (a8 + (size_t) j * K + b * QK_PQ2_0 + l * 8) = q;
                if (l == 0) {
                    as[(size_t) j * nb + b] = d;
                }
            });
    }

    const uint32_t * wq = (const uint32_t *) src0->data;
    const uint16_t * wd = (const uint16_t *) ((const uint8_t *) src0->data + (size_t) nrows * nb * PQ2_XMX_QS_BYTES);
    float *          dd = (float *) dst->data;
    const int        nrows_dst = (int) dst->ne[0];

    if (ncols <= 8) {
        launch_pq2_xmx_split<1, 2>(wq, wd, (const uint32_t *) a8, as, dd, K, nrows, ncols, nrows_dst, stream);
    } else if (ncols <= 16) {
        launch_pq2_xmx_split<2, 2>(wq, wd, (const uint32_t *) a8, as, dd, K, nrows, ncols, nrows_dst, stream);
    } else {
        launch_pq2_xmx_split<4, 2>(wq, wd, (const uint32_t *) a8, as, dd, K, nrows, ncols, nrows_dst, stream);
    }
}

#else

bool ggml_sycl_pq2_xmx_supports_ne0(int64_t) {
    return false;
}

bool ggml_sycl_pq2_xmx_reorder(ggml_tensor *, dpct::queue_ptr) {
    return false;
}

void ggml_sycl_pq2_xmx_mul_mat(ggml_backend_sycl_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    GGML_ABORT("PQ2_0 XMX path is not built in");
}

#endif // __INTEL_LLVM_COMPILER && !GGML_SYCL_NO_PQ2_XMX

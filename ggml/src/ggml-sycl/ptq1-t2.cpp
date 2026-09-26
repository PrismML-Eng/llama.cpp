// ARC-LAB: PTQ1_0 on XMX through the TernSYCL int2 x int8 DPAS kernels (ternsycl/, BSD 3-Clause, libxsmm/TernSYCL
// @5e162d8). The base-3 PTQ1_0 packing (1.625 + 0.125 bits/weight) has to be decoded on the ALUs before any DPAS, which
// makes the 4-column verify mat-vec compute-bound (~190 GB/s on the B580); 2-bit two's-complement codes (2.125 bits)
// feed the s8 x s2 DPAS directly. B580 standalone, 2 GiB rotating weights: 17408x5120 m=1 56 us, m=4 61 us (vs 70 /
// ~100 in-model for PTQ1_0). The weight grows 31%; activations become int8 with one fp16 scale per 128 (q8_1 is per 32).
#include "ptq1-t2.hpp"

#include "ternsycl/int2_int8_dpas.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

constexpr int     QK        = 128;
constexpr int64_t BLK_BYTES = 28;  // block_ptq1_0: qs[24], qh[2], fp16 d

int t2_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_SYCL_PTQ1_T2");
        if (!e) {
            return 0;
        }
        if (!strcmp(e, "all")) {
            return 2;
        }
        if (!strcmp(e, "ffn")) {
            return 1;
        }
        return 0;
    }();
    return mode;
}

// growing per-process scratch for the int8 activations and their scales (in-order queue: reuse is safe)
struct scratch {
    void * p     = nullptr;
    size_t bytes = 0;
};

void * scratch_get(sycl::queue & q, size_t bytes) {
    static std::mutex mtx;
    static scratch    s;
    std::lock_guard<std::mutex> lock(mtx);
    if (bytes > s.bytes) {
        if (s.p) {
            q.wait();
            sycl::free(s.p, q);
        }
        s.bytes = bytes + bytes / 4;
        s.p     = sycl::malloc_device(s.bytes, q);
        if (!s.p) {
            fprintf(stderr, "ptq1-t2: no device memory for %zu bytes of activation scratch\n", s.bytes);
            abort();
        }
    }
    return s.p;
}

}  // namespace

bool ggml_sycl_t2_wants(const char * name, int64_t K, int64_t N) {
    const int mode = t2_mode();
    if (mode == 0 || K % QK != 0 || N % 16 != 0) {
        return false;
    }
    if (mode == 2) {
        return true;
    }
    return name && (strstr(name, "ffn_gate.") || strstr(name, "ffn_up.") || strstr(name, "ffn_down."));
}

size_t ggml_sycl_t2_bytes(int64_t K, int64_t N) {
    return (size_t) (K / 16) * N * 4 + (size_t) (K / QK) * N * 2;
}

namespace {
void t2_precompile(sycl::queue & q);
}

void ggml_sycl_t2_repack(sycl::queue & q, void * data, int64_t K, int64_t N) {
    static std::once_flag compiled;
    std::call_once(compiled, [&] { t2_precompile(q); });
    const int64_t nb        = K / QK;
    const size_t  src_bytes = (size_t) N * nb * BLK_BYTES;
    uint8_t *     tmp       = (uint8_t *) sycl::malloc_device(src_bytes, q);
    if (!tmp) {
        fprintf(stderr, "%s: no device memory for a %zu-byte repack buffer\n", __func__, src_bytes);
        abort();
    }
    q.memcpy(tmp, data, src_bytes).wait();
    uint32_t * B  = (uint32_t *) data;
    uint16_t * SB = (uint16_t *) ((char *) data + (size_t) (K / 16) * N * 4);
    q.parallel_for(sycl::range<2>((size_t) (K / 16), (size_t) N), [=](sycl::item<2> it) {
         const int64_t   kp = it[0], n = it[1];
         const int64_t   g  = kp / 8;
         const uint8_t * b  = tmp + (n * nb + g) * BLK_BYTES;
         const int       e0 = (int) (kp % 8) * 16;
         uint32_t        w  = 0;
         for (int j = 0; j < 16; ++j) {
             const int e = e0 + j;
             uint32_t  v;
             int       lvl;
             if (e < 80) {
                 v   = b[e % 16];
                 lvl = e / 16;
             } else if (e < 120) {
                 v   = b[16 + (e - 80) % 8];
                 lvl = (e - 80) / 8;
             } else {
                 v   = b[24 + (e - 120) % 2];
                 lvl = (e - 120) / 2;
             }
             uint32_t digit = 0;
             for (int t = 0; t <= lvl; ++t) {  // base-3 fixed-point digits, as ptq1_0_decode_block
                 const uint32_t x = v * 3;
                 digit            = x >> 8;
                 v                = x & 0xFF;
             }
             const uint32_t code = digit == 2 ? 1u : (digit == 1 ? 0u : 3u);  // weight = digit - 1, 2-bit two's complement
             w |= code << (2 * j);
         }
         B[kp * N + n] = w;
         if (kp % 8 == 0) {
             SB[g * N + n] = (uint16_t) (b[26] | (b[27] << 8));
         }
     }).wait();
    sycl::free(tmp, q);
}

namespace {

// GGML_SYCL_PTQ1_T2_S32: activation scales per 32 (q8_1 granularity) instead of per 128: 1 = GEMV (<= 8 tokens) only,
// 2 = GEMV + GEMM; 0 = TernSYCL's per 128
int t2_s32() {
    static const int v = getenv("GGML_SYCL_PTQ1_T2_S32") ? atoi(getenv("GGML_SYCL_PTQ1_T2_S32")) : 0;
    return v;
}

// fp32 activations -> int8 Aq [M, K] + fp16 SA [K/128, ldsa(M)] (TernSYCL QMODE 0: SA = 127 / absmax, saturate)
// G = 128 (one scale per 16 lanes x 8 values) or 32 (a scale per 4 lanes); SA [K/G, ldsa(M)]
void quant_a(sycl::queue & q, const float * x, int64_t x_stride, int64_t M, int64_t K, int8_t * Aq, uint16_t * SA,
             const int G = QK) {
    const int lda = int8dpas::ldsa((int) M);
    q.parallel_for(sycl::nd_range<2>({ (size_t) M, (size_t) (K / QK) * 16 }, { 1, 16 }),
                   [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                       const auto    sg   = it.get_sub_group();
                       const int     lane = sg.get_local_linear_id();
                       const int64_t m    = it.get_global_id(0);
                       const int64_t g    = it.get_global_id(1) / 16;
                       const float * a    = x + m * x_stride + g * QK + 8 * lane;
                       float         v[8], mx = 0.0f;
#pragma unroll
                       for (int i = 0; i < 8; ++i) {
                           v[i] = a[i];
                           mx   = sycl::fmax(mx, sycl::fabs(v[i]));
                       }
                       if (G == QK) {
                           mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
                       } else {  // 32-value groups = 4 lanes
                           mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 1));
                           mx = sycl::fmax(mx, sycl::permute_group_by_xor(sg, mx, 2));
                       }
                       const sycl::half sh = sycl::half(127.0f / sycl::fmax(mx, int8dpas::EPS));
                       if (G == QK ? lane == 0 : (lane & 3) == 0) {
                           const int64_t gi = G == QK ? g : g * 4 + lane / 4;
                           SA[gi * lda + m] = sycl::bit_cast<uint16_t>(sh);
                       }
                       const float s = (float) sh;
                       uint64_t    p = 0;
#pragma unroll
                       for (int i = 0; i < 8; ++i) {
                           // round to nearest (TernSYCL truncates: RTZ biases every product toward zero)
                           p |= (uint64_t) (uint8_t) (int8_t) sycl::rint(sycl::clamp(v[i] * s, -128.0f, 127.0f)) << (8 * i);
                       }
                       *(uint64_t *) (Aq + m * K + g * QK + 8 * lane) = p;
                   });
}

template <int SGM, int LS, int NS = 1>
void gemv(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, const uint32_t * B, const uint16_t * SB, float * C,
          int M, int N, int K) {
    using Kern          = int8dpas::Gemv<false, 0, SGM, 2, LS, 2, NS>;
    const size_t    wgn = 16 * 2;
    const Epi       epi{ nullptr, nullptr, 0, 1 };
    const sycl::range<2> local(1, Kern::WG);
    const sycl::range<2> global((M + SGM - 1) / SGM, (N + wgn - 1) / wgn * Kern::WG);
    q.parallel_for(sycl::nd_range<2>(global, local),
                   Kern{ nullptr, (const signed char *) Aq, SA, B, SB, C, epi, M, N, K });
}

template <int NS = 1>
void gemm(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, const uint32_t * B, const uint16_t * SB, float * C,
          int M, int N, int K) {
    constexpr int   MT_M = 8, MT_N = 128, WG_M = 8, WG_N = 2;
    const size_t    tm = MT_M * WG_M, tn = MT_N * WG_N;
    const Epi       epi{ nullptr, nullptr, 0, 1 };
    const sycl::range<2> local(1, 16 * WG_M * WG_N);
    const sycl::range<2> global((M + tm - 1) / tm, (N + tn - 1) / tn * local[1]);
    q.parallel_for(sycl::nd_range<2>(global, local),
                   int8dpas::GemmMT<false, 0, MT_M, MT_N, WG_M, WG_N, 0, true, NS>{
                       nullptr, (const signed char *) Aq, SA, B, SB, C, epi, M, N, K });
}

}  // namespace

namespace {
template <int NS>
void t2_dispatch(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, const uint32_t * B, const uint16_t * SB,
                 float * dst, int m, int n, int k, bool use_gemm) {
    if (use_gemm) {
        gemm<NS>(q, Aq, SA, B, SB, dst, m, n, k);
    } else if (m == 1) {
        n <= 8192 ? gemv<1, 4, NS>(q, Aq, SA, B, SB, dst, m, n, k) : gemv<1, 2, NS>(q, Aq, SA, B, SB, dst, m, n, k);
    } else if (m == 2) {
        n <= 8192 ? gemv<2, 4, NS>(q, Aq, SA, B, SB, dst, m, n, k) : gemv<2, 2, NS>(q, Aq, SA, B, SB, dst, m, n, k);
    } else if (m <= 4) {
        n <= 8192 ? gemv<4, 4, NS>(q, Aq, SA, B, SB, dst, m, n, k) : gemv<4, 2, NS>(q, Aq, SA, B, SB, dst, m, n, k);
    } else {
        n <= 8192 ? gemv<8, 4, NS>(q, Aq, SA, B, SB, dst, m, n, k) : gemv<8, 2, NS>(q, Aq, SA, B, SB, dst, m, n, k);
    }
}
}  // namespace

void ggml_sycl_t2_mul_mat(sycl::queue & q, const void * w, const float * x, int64_t x_stride, float * dst, int64_t M,
                          int64_t N, int64_t K) {
    // batches > 8 (n-gram verify, prompts): GemmMT reads each weight once per 64 rows. Its SA read, A read and C write
    // were switched from 2D block I/O to plain / sub-group block access (the 2D forms misbehaved in this JIT build).
    static const bool gemm_off = getenv("GGML_SYCL_PTQ1_T2_GEMM_OFF") != nullptr;
    const bool use_gemm = M > 8 && !gemm_off;
    const bool s32      = use_gemm ? t2_s32() >= 2 : t2_s32() >= 1;

    const int      lda      = int8dpas::ldsa((int) M);
    const size_t   aq_bytes = (size_t) M * K;
    const size_t   sa_off   = (aq_bytes + 255) & ~(size_t) 255;
    const size_t   sa_bytes = (size_t) (K / 32) * lda * 2 + 256;
    char *         s        = (char *) scratch_get(q, sa_off + sa_bytes);
    int8_t *       Aq       = (int8_t *) s;
    uint16_t *     SA       = (uint16_t *) (s + sa_off);
    const uint32_t * B      = (const uint32_t *) w;
    const uint16_t * SB     = (const uint16_t *) ((const char *) w + (size_t) (K / 16) * N * 4);

    quant_a(q, x, x_stride, M, K, Aq, SA, s32 ? 32 : QK);
    if (s32) {
        t2_dispatch<4>(q, Aq, SA, B, SB, dst, (int) M, (int) N, (int) K, use_gemm);
    } else {
        t2_dispatch<1>(q, Aq, SA, B, SB, dst, (int) M, (int) N, (int) K, use_gemm);
    }
}

namespace {
// run every kernel once on a tiny problem while the model loads (the first weight repack happens in the warm-up
// decode): the JIT of GemmMT alone took ~35 s at the first prompt otherwise
void t2_precompile(sycl::queue & q) {
    constexpr int K = 128, N = 32, M = 16;
    char *     w  = (char *) sycl::malloc_device(ggml_sycl_t2_bytes(K, N), q);
    float *    x  = sycl::malloc_device<float>(M * K, q);
    float *    d  = sycl::malloc_device<float>(M * N, q);
    int8_t *   aq = sycl::malloc_device<int8_t>(M * K, q);
    uint16_t * sa = sycl::malloc_device<uint16_t>(int8dpas::ldsa(M) * (K / 32) + 64, q);
    q.memset(w, 0, ggml_sycl_t2_bytes(K, N));
    q.memset(x, 0, sizeof(float) * M * K);
    const uint32_t * B  = (const uint32_t *) w;
    const uint16_t * SB = (const uint16_t *) (w + (K / 16) * N * 4);
    quant_a(q, x, K, M, K, aq, sa, QK);
    quant_a(q, x, K, M, K, aq, sa, 32);
    for (int m : { 1, 2, 4, 8, M }) {
        for (int ns = 0; ns < 2; ++ns) {
            // both GEMV tiles (the N > 8192 one is picked by n, so call the templates directly)
            if (ns == 0) {
                t2_dispatch<1>(q, aq, sa, B, SB, d, m, N, K, m > 8);
            } else {
                t2_dispatch<4>(q, aq, sa, B, SB, d, m, N, K, m > 8);
            }
        }
    }
    gemv<1, 2>(q, aq, sa, B, SB, d, 1, N, K);
    gemv<2, 2>(q, aq, sa, B, SB, d, 2, N, K);
    gemv<4, 2>(q, aq, sa, B, SB, d, 4, N, K);
    gemv<8, 2>(q, aq, sa, B, SB, d, 8, N, K);
    gemv<1, 2, 4>(q, aq, sa, B, SB, d, 1, N, K);
    gemv<2, 2, 4>(q, aq, sa, B, SB, d, 2, N, K);
    gemv<4, 2, 4>(q, aq, sa, B, SB, d, 4, N, K);
    gemv<8, 2, 4>(q, aq, sa, B, SB, d, 8, N, K);
    q.wait();
    sycl::free(w, q);
    sycl::free(x, q);
    sycl::free(d, q);
    sycl::free(aq, q);
    sycl::free(sa, q);
}
}  // namespace

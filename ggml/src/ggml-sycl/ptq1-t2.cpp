// ARC-LAB: PTQ1_0 on XMX through the TernSYCL int2 x int8 DPAS kernels (ternsycl/, BSD 3-Clause, libxsmm/TernSYCL
// @8b0ae42: inline vISA only, no IGC builtins; builds with icpx's default -fp-model=fast). The base-3 PTQ1_0 packing (1.625 + 0.125 bits/weight) has to be decoded on the ALUs before any DPAS, which
// makes the 4-column verify mat-vec compute-bound (~190 GB/s on the B580); 2-bit two's-complement codes (2.125 bits)
// feed the s8 x s2 DPAS directly. B580 standalone, 2 GiB rotating weights: 17408x5120 m=1 56 us, m=4 61 us (vs 70 /
// ~100 in-model for PTQ1_0). The weight grows 31%; activations become int8 with one fp16 scale per 128 (q8_1 is per 32).
#include "ptq1-t2.hpp"
#include "common.hpp"

#include "ternsycl/int2_int8_dpas.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <vector>

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

// the kernels use 16-lane DPAS (Xe2 and newer); Xe-LPG(+) / Alchemist XMX is 8 lanes and has no such kernels, so fall back
// to the regular path with one warning instead of failing at the first launch. GGML_SYCL_PTQ1_T2_ANYGPU=1 skips the check.
bool ggml_sycl_device_is_xe2() {
    const auto & hw = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info;
    return hw.arch == gpu_arch::intel_gpu_bmg_g21 || hw.arch == gpu_arch::intel_gpu_bmg_g31 ||
           hw.arch == gpu_arch::intel_gpu_lnl_m || hw.arch == gpu_arch::intel_gpu_ptl_h ||
           hw.arch == gpu_arch::intel_gpu_ptl_u || hw.arch == gpu_arch::intel_gpu_wcl;
}

// Xe3 (Panther Lake, Wildcat Lake), for the defaults measured on an Arc B390
bool ggml_sycl_device_is_xe3(int device) {
    const auto & hw = ggml_sycl_info().devices[device].hw_info;
    return hw.arch == gpu_arch::intel_gpu_ptl_h || hw.arch == gpu_arch::intel_gpu_ptl_u ||
           hw.arch == gpu_arch::intel_gpu_wcl;
}

static bool t2_device_ok() {
    static const bool ok = [] {
        if (getenv("GGML_SYCL_PTQ1_T2_ANYGPU")) {
            return true;
        }
        const auto & hw = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info;
        const bool xe2 = ggml_sycl_device_is_xe2();
        if (!xe2) {
            GGML_LOG_WARN("%s: GGML_SYCL_PTQ1_T2 needs an Xe2 or newer GPU (Arc B-series, Lunar Lake, Panther Lake); "
                          "%s is not one, using the regular path\n", __func__, hw.name.c_str());
        }
        return xe2;
    }();
    return ok;
}

bool ggml_sycl_t2_wants(const char * name, int64_t K, int64_t N) {
    const int mode = t2_mode();
    if (mode == 0 || K % QK != 0 || N % 16 != 0 || !t2_device_ok()) {
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

bool ggml_sycl_t2_repack(sycl::queue & q, void * data, int64_t K, int64_t N, bool pq2) {
    static std::once_flag compiled;
    std::call_once(compiled, [&] { t2_precompile(q); });
    const int64_t nb        = K / QK;
    const int64_t blk_bytes = pq2 ? 34 : BLK_BYTES;  // block_pq2_0: fp16 d, then 32 bytes of 2-bit codes
    const size_t  src_bytes = (size_t) N * nb * blk_bytes;
    uint8_t *     tmp       = (uint8_t *) sycl::malloc_device(src_bytes, q);
    if (!tmp) {
        fprintf(stderr, "%s: no device memory for a %zu-byte repack buffer\n", __func__, src_bytes);
        abort();
    }
    q.memcpy(tmp, data, src_bytes).wait();
    uint32_t * B   = (uint32_t *) data;
    uint16_t * SB  = (uint16_t *) ((char *) data + (size_t) (K / 16) * N * 4);
    int *      bad = sycl::malloc_device<int>(1, q);
    q.memset(bad, 0, sizeof(int)).wait();
    q.parallel_for(sycl::range<2>((size_t) (K / 16), (size_t) N), [=](sycl::item<2> it) {
         const int64_t   kp = it[0], n = it[1];
         const int64_t   g  = kp / 8;
         const uint8_t * b  = tmp + (n * nb + g) * blk_bytes;
         const int       e0 = (int) (kp % 8) * 16;
         uint32_t        w  = 0;
         for (int j = 0; j < 16; ++j) {
             const int e = e0 + j;
             uint32_t  digit;  // weight = digit - 1
             if (pq2) {
                 digit = (b[2 + e / 4] >> (2 * (e % 4))) & 3u;
                 if (digit == 3u) {
                     *bad = 1;
                     digit = 2u;
                 }
             } else {
                 uint32_t v;
                 int      lvl;
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
                 digit = 0;
                 for (int t = 0; t <= lvl; ++t) {  // base-3 fixed-point digits, as ptq1_0_decode_block
                     const uint32_t x = v * 3;
                     digit            = x >> 8;
                     v                = x & 0xFF;
                 }
             }
             const uint32_t code = digit == 2 ? 1u : (digit == 1 ? 0u : 3u);  // 2-bit two's complement of digit - 1
             w |= code << (2 * j);
         }
         B[kp * N + n] = w;
         if (kp % 8 == 0) {
             SB[g * N + n] = pq2 ? (uint16_t) (b[0] | (b[1] << 8)) : (uint16_t) (b[26] | (b[27] << 8));
         }
     }).wait();
    int bad_h = 0;
    q.memcpy(&bad_h, bad, sizeof(int)).wait();
    sycl::free(bad, q);
    sycl::free(tmp, q);
    // ARC-LAB experiment: GGML_SYCL_T2_ZERO_WEIGHTS=1 zeroes every T2 weight after the repack (output is garbage). The Xe2
    // memory compression then makes weight reads almost free, so a profile shows everything else a token costs (2026-10-05).
    static const bool zero_w = getenv("GGML_SYCL_T2_ZERO_WEIGHTS") && atoi(getenv("GGML_SYCL_T2_ZERO_WEIGHTS"));
    if (zero_w) {
        q.memset(B, 0, (size_t) (K / 16) * N * 4).wait();
    }
    return bad_h == 0;
}

namespace {

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

// GGML_SYCL_PTQ1_T2_NSG: sub-groups per GEMV work-group for 2..8 rows (verify batches); 2 (default) or 4. The B580 tile
// sweep (TernSYCL main, Bonsai 27B shapes, M=4, qmode 0) had 4 fastest on 5 of 6 shapes (+0-8%); M=1 keeps 2.
int t2_nsg() {
    const int v = ggml_sycl_knob("GGML_SYCL_PTQ1_T2_NSG", 2) == 4 ? 4 : 2;
    return v;
}

template <int SGM, int LS, int NSG = 2>
void gemv(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, const uint32_t * B, const uint16_t * SB, float * C,
          int M, int N, int K) {
    using Kern          = int8dpas::Gemv<false, 0, SGM, NSG, LS, 2>;
    const size_t    wgn = 16 * NSG;
    const Epi       epi{ nullptr, nullptr, 0, 1 };
    const sycl::range<2> local(1, Kern::WG);
    const sycl::range<2> global((M + SGM - 1) / SGM, (N + wgn - 1) / wgn * Kern::WG);
    q.parallel_for(sycl::nd_range<2>(global, local),
                   Kern{ nullptr, (const signed char *) Aq, SA, B, SB, C, epi, M, N, K });
}

template <int MT_M, int MT_N, int WG_M, int WG_N>
void gemm_tile(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, const uint32_t * B, const uint16_t * SB, float * C,
               int M, int N, int K) {
    const size_t    tm = MT_M * WG_M, tn = MT_N * WG_N;
    const Epi       epi{ nullptr, nullptr, 0, 1 };
    const sycl::range<2> local(1, 16 * WG_M * WG_N);
    const sycl::range<2> global((M + tm - 1) / tm, (N + tn - 1) / tn * local[1]);
    q.parallel_for(sycl::nd_range<2>(global, local),
                   int8dpas::GemmMT<false, 0, MT_M, MT_N, WG_M, WG_N, 0, true>{
                       nullptr, (const signed char *) Aq, SA, B, SB, C, epi, M, N, K });
}

// ARC-LAB lab knob GGML_SYCL_PTQ1_T2_TILE = index into TernSYCL's large-M tile table (mt_m, mt_n, wg_m, wg_n):
// 0 {8,128,8,2} (default) 1 {8,128,4,2} 2 {8,128,4,4} 3 {8,128,16,1} 4 {8,128,2,4} 5 {16,64,4,2} 6 {16,64,8,2}
// 7 {8,64,8,2} 8 {32,32,4,2}
void gemm(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, const uint32_t * B, const uint16_t * SB, float * C,
          int M, int N, int K) {
    // per-batch-size tile (arctune sweep, B580, 2026-10-05): {8,128,2,4} is 7-11% faster for 12-32 rows (n-gram verify
    // batches), {8,128,8,2} from 48 rows up (prompts). GGML_SYCL_PTQ1_T2_TILE forces one tile for every size;
    // GGML_SYCL_PTQ1_T2_TILE_SMALL_MAX = largest batch on the small tile (0 = never).
    const int forced = ggml_sycl_knob("GGML_SYCL_PTQ1_T2_TILE", -1);
    const int tile   = forced >= 0 ? forced : (M <= ggml_sycl_knob("GGML_SYCL_PTQ1_T2_TILE_SMALL_MAX", 32) ? 4 : 0);
    switch (tile) {
        case 1: gemm_tile<8, 128, 4, 2>(q, Aq, SA, B, SB, C, M, N, K); break;
        case 2: gemm_tile<8, 128, 4, 4>(q, Aq, SA, B, SB, C, M, N, K); break;
        case 3: gemm_tile<8, 128, 16, 1>(q, Aq, SA, B, SB, C, M, N, K); break;
        case 4: gemm_tile<8, 128, 2, 4>(q, Aq, SA, B, SB, C, M, N, K); break;
        case 5: gemm_tile<16, 64, 4, 2>(q, Aq, SA, B, SB, C, M, N, K); break;
        case 6: gemm_tile<16, 64, 8, 2>(q, Aq, SA, B, SB, C, M, N, K); break;
        case 7: gemm_tile<8, 64, 8, 2>(q, Aq, SA, B, SB, C, M, N, K); break;
        case 8: gemm_tile<32, 32, 4, 2>(q, Aq, SA, B, SB, C, M, N, K); break;
        default: gemm_tile<8, 128, 8, 2>(q, Aq, SA, B, SB, C, M, N, K); break;
    }
}

}  // namespace

namespace {
// up to 3 weight matrices with the same activations in ONE launch: work-groups [gs[m], gs[m+1]) along dim 1 run
// matrix m (Gemv::run takes the group indices). All members use the same Gemv instantiation (same SLM, same shape).
template <int SGM, int LS, int NSG>
struct GemvMulti {
    using Kern = int8dpas::Gemv<false, 0, SGM, NSG, LS, 2>;
    Kern k0, k1, k2;
    int  gs1, gs2;
    void operator()(sycl::nd_item<2> it) const {
        // one call site: three inlined copies of the body spilled (34.6 vs 46.3 t/s)
        const int  g0 = (int) it.get_group(0), g1 = (int) it.get_group(1);
        const bool a = g1 < gs1, b = !a && g1 < gs2;
        const Kern k = a ? k0 : (b ? k1 : k2);
        k.run(it, g0, g1 - (a ? 0 : (b ? gs1 : gs2)));
    }
    auto get(int8dpas::syclex::properties_tag) const { return k0.get(int8dpas::syclex::properties_tag{}); }
};

template <int SGM, int LS, int NSG = 2>
void gemv_multi(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, int nm, const uint32_t * const * B,
                const uint16_t * const * SB, float * const * C, const int * N, int M, int K) {
    using GM            = GemvMulti<SGM, LS, NSG>;
    using Kern          = typename GM::Kern;
    const int       wgn = 16 * NSG;
    const Epi       epi{ nullptr, nullptr, 0, 1 };
    int             gs[4] = { 0, 0, 0, 0 };
    Kern            k[3];
    for (int m = 0; m < 3; ++m) {
        const int mm = m < nm ? m : 0;
        k[m]         = Kern{ nullptr, (const signed char *) Aq, SA, B[mm], SB[mm], C[mm], epi, M, N[mm], K };
        gs[m + 1]    = gs[m] + (m < nm ? (N[m] + wgn - 1) / wgn : 0);
    }
    const GM             f{ k[0], k[1], k[2], gs[1], nm > 2 ? gs[2] : gs[3] + 1 };
    const sycl::range<2> local(1, Kern::WG);
    const sycl::range<2> global((M + SGM - 1) / SGM, (size_t) gs[nm] * Kern::WG);
    q.parallel_for(sycl::nd_range<2>(global, local), f);
}
}  // namespace

namespace {
// ARC-LAB: GGML_SYCL_PTQ1_T2_TUNED=1 (default: 1 on a B580, else 0): per-shape GEMV tiles for 1..8 rows from the B580 sweep
// (/lab/exp/gemv-sweep, README "Mapping a winner into the fork"). A row of t2_tuned_table matches a launch by
// (N, K, rows, multi), N = the sum of the members' N for a same-input merged launch; no match = the default tiles above.
// =2 is a plumbing self-test (default tiles through the tuned launcher, outputs must not change).
// The tile is Gemv<SGM, NSG, LS, U> under a GRF mode (0 = no property, as Gemv; 128 / 256; 1 = grf_size_automatic) and
// a work-group order (0 = row-group outer, as the default path; 1 = the row-groups of a column block adjacent, so with
// SGM < rows the repeated weight reads hit L2). Same struct as gsw::Sweep in the sweep (sweep_kern.hpp): keep them equal.
template <int GRF, int WG> constexpr auto t2_tuned_props() {
    namespace syclex  = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    if constexpr (GRF == 128 || GRF == 256) {
        return syclex::properties{ syclex::sub_group_size<16>, syclex::work_group_size<1, WG>, intelex::grf_size<GRF> };
    } else if constexpr (GRF == 1) {
        return syclex::properties{ syclex::sub_group_size<16>, syclex::work_group_size<1, WG>,
                                   intelex::grf_size_automatic };
    } else {
        return syclex::properties{ syclex::sub_group_size<16>, syclex::work_group_size<1, WG> };
    }
}

template <int SGM, int NSG, int LS, int U, int GRF>
struct GemvTuned {
    using Kern = int8dpas::Gemv<false, 0, SGM, NSG, LS, U>;
    Kern k0, k1, k2;
    int  gs1, gs2;  // first column-group of members 1 and 2 (past the end when absent)
    int  R, C;      // row-groups, column-groups (all members)
    int  ord;
    void operator()(sycl::nd_item<2> it) const {
        const int  g  = (int) it.get_group(1);
        const int  g0 = ord ? g % R : g / C;
        const int  g1 = ord ? g / R : g % C;
        const bool a = g1 < gs1, b = !a && g1 < gs2;
        const Kern k = a ? k0 : (b ? k1 : k2);  // one call site of run (see GemvMulti)
        k.run(it, g0, g1 - (a ? 0 : (b ? gs1 : gs2)));
    }
    auto get(int8dpas::syclex::properties_tag) const { return t2_tuned_props<GRF, Kern::WG>(); }
};

template <int SGM, int NSG, int LS, int U, int GRF>
void gemv_tuned(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, int nm, const uint32_t * const * B,
                const uint16_t * const * SB, float * const * C, const int * N, int M, int K, int ord) {
    using T        = GemvTuned<SGM, NSG, LS, U, GRF>;
    using Kern     = typename T::Kern;
    const int  wgn = 16 * NSG;
    const Epi  epi{ nullptr, nullptr, 0, 1 };
    int        gs[4] = { 0, 0, 0, 0 };
    Kern       k[3];
    for (int m = 0; m < 3; ++m) {
        const int mm = m < nm ? m : 0;
        k[m]         = Kern{ nullptr, (const signed char *) Aq, SA, B[mm], SB[mm], C[mm], epi, M, N[mm], K };
        gs[m + 1]    = gs[m] + (m < nm ? (N[m] + wgn - 1) / wgn : 0);
    }
    const int R = (M + SGM - 1) / SGM, Ct = gs[nm];
    const T   f{ k[0], k[1], k[2], nm > 1 ? gs[1] : Ct + 1, nm > 2 ? gs[2] : Ct + 1, R, Ct, ord };
    q.parallel_for(sycl::nd_range<2>({ 1, (size_t) R * Ct * Kern::WG }, { 1, (size_t) Kern::WG }), f);
}

// the compiled tiles: X(SGM, NSG, LS, U, GRF), one line per tile a table row uses (a row whose tile is not listed here is
// skipped with a warning)
#define T2_TUNED_KERNELS(X) \
    X(4, 8, 4, 2, 0)        \
    X(4, 1, 2, 2, 0)        \
    X(4, 1, 2, 4, 0)        \
    X(1, 2, 2, 1, 0)

struct t2_tuned_kernel {
    int sgm, nsg, ls, u, grf;
    void (*fn)(sycl::queue &, const int8_t *, const uint16_t *, int, const uint32_t * const *, const uint16_t * const *,
               float * const *, const int *, int, int, int);
};
#define T2_TUNED_X(a, b, c, d, e) t2_tuned_kernel{ a, b, c, d, e, gemv_tuned<a, b, c, d, e> },
const std::vector<t2_tuned_kernel> & t2_tuned_kernels() {
    static const std::vector<t2_tuned_kernel> k = { T2_TUNED_KERNELS(T2_TUNED_X) };
    return k;
}
#undef T2_TUNED_X

struct t2_tuned_row {
    int N, K, M, multi;          // launch: N (sum over members if merged), K, rows (1..8), merged launch (0/1)
    int sgm, nsg, ls, u, grf, ord;  // tile
};
// rows from the sweep's summary ("fork table rows"); empty = GGML_SYCL_PTQ1_T2_TUNED=1 changes nothing
const std::vector<t2_tuned_row> & t2_tuned_table() {
    static const std::vector<t2_tuned_row> t = {
        // { N, K, M, multi, SGM, NSG, LS, U, GRF, ord },
        // B580 sweep 2026-10-10 (/lab/exp/gemv-sweep jit.log, JIT build, medians of 5 interleaved rounds): bit-exact
        // winners only (same LS as the default tile), > 2% faster, launches the model issues. Time vs the default tile:
        { 5120, 17408, 5, 0, 4, 8, 4, 2, 0, 1 },    // ffn_down, 5-row verify             0.775
        { 34816, 5120, 5, 1, 4, 1, 2, 2, 0, 1 },    // ffn gate+up merged, 5 rows         0.896
        { 248320, 5120, 5, 0, 4, 1, 2, 4, 0, 1 },   // output head, 5 rows                0.882
        { 14336, 5120, 5, 1, 4, 1, 2, 2, 0, 1 },    // attention q+k+v merged, 5 rows     0.923
        { 14336, 5120, 1, 1, 1, 2, 2, 1, 0, 0 },    // attention q+k+v merged, 1 row      0.974
    };
    return t;
}

// the tiles were swept on a B580 (20 Xe cores): on by default there only; GGML_SYCL_PTQ1_T2_TUNED=1 / 0 forces it
static int t2_tuned_default() {
    static const int d = [] {
        const std::string name = dpct::dev_mgr::instance().get_device(ggml_sycl_get_device()).get_info<sycl::info::device::name>();
        return name.find("B580") != std::string::npos ? 1 : 0;
    }();
    return d;
}

bool t2_tuned_on() {
    return ggml_sycl_knob("GGML_SYCL_PTQ1_T2_TUNED", t2_tuned_default()) != 0;
}

// GGML_SYCL_PTQ1_T2_TUNED=2: plumbing self-test, every 1..8-row GEMV through gemv_tuned with the default tile (same LS,
// so outputs must be token-identical to =0)
template <int SGM, int NSG>
void t2_tuned_selftest_ls(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, int nm, const uint32_t * const * B,
                          const uint16_t * const * SB, float * const * C, const int * N, int M, int K, bool ls4) {
    ls4 ? gemv_tuned<SGM, NSG, 4, 2, 0>(q, Aq, SA, nm, B, SB, C, N, M, K, 0) :
          gemv_tuned<SGM, NSG, 2, 2, 0>(q, Aq, SA, nm, B, SB, C, N, M, K, 0);
}

template <int NSG>
void t2_tuned_selftest(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, int nm, const uint32_t * const * B,
                       const uint16_t * const * SB, float * const * C, const int * N, int M, int K, bool ls4) {
    if (M == 1) {
        t2_tuned_selftest_ls<1, NSG>(q, Aq, SA, nm, B, SB, C, N, M, K, ls4);
    } else if (M == 2) {
        t2_tuned_selftest_ls<2, NSG>(q, Aq, SA, nm, B, SB, C, N, M, K, ls4);
    } else if (M <= 4) {
        t2_tuned_selftest_ls<4, NSG>(q, Aq, SA, nm, B, SB, C, N, M, K, ls4);
    } else {
        t2_tuned_selftest_ls<8, NSG>(q, Aq, SA, nm, B, SB, C, N, M, K, ls4);
    }
}

// launches the tuned tile for this GEMV if the table has one; false = take the default path
bool t2_tuned(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, int nm, const uint32_t * const * B,
              const uint16_t * const * SB, float * const * C, const int * N, int M, int K) {
    const int mode = ggml_sycl_knob("GGML_SYCL_PTQ1_T2_TUNED", t2_tuned_default());
    if (mode == 0) {
        return false;
    }
    int ntot = 0, nmax = 0;
    for (int m = 0; m < nm; ++m) {
        ntot += N[m];
        nmax = std::max(nmax, N[m]);
    }
    if (mode == 2) {
        // NSG as the default path: 2, or GGML_SYCL_PTQ1_T2_NSG=4 for 2..8 rows
        (M > 1 && t2_nsg() == 4 ? t2_tuned_selftest<4> : t2_tuned_selftest<2>)(q, Aq, SA, nm, B, SB, C, N, M, K,
                                                                               nmax <= 8192);
        return true;
    }
    for (const t2_tuned_row & r : t2_tuned_table()) {
        if (r.N != ntot || r.K != K || r.M != M || r.multi != (nm > 1)) {
            continue;
        }
        for (const t2_tuned_kernel & k : t2_tuned_kernels()) {
            if (k.sgm == r.sgm && k.nsg == r.nsg && k.ls == r.ls && k.u == r.u && k.grf == r.grf) {
                k.fn(q, Aq, SA, nm, B, SB, C, N, M, K, r.ord);
                return true;
            }
        }
        static std::atomic<bool> warned{ false };
        if (!warned.exchange(true)) {
            GGML_LOG_WARN("%s: tuned tile %d.%d.%d.%d.%d for N=%d K=%d M=%d is not in T2_TUNED_KERNELS, default tile\n",
                          __func__, r.sgm, r.nsg, r.ls, r.u, r.grf, r.N, r.K, r.M);
        }
        return false;
    }
    return false;
}

void t2_dispatch(sycl::queue & q, const int8_t * Aq, const uint16_t * SA, const uint32_t * B, const uint16_t * SB,
                 float * dst, int m, int n, int k, bool use_gemm) {
    if (use_gemm) {
        gemm(q, Aq, SA, B, SB, dst, m, n, k);
    } else if (t2_tuned(q, Aq, SA, 1, &B, &SB, &dst, &n, m, k)) {
        // tuned tile
    } else if (m == 1) {
        n <= 8192 ? gemv<1, 4>(q, Aq, SA, B, SB, dst, m, n, k) : gemv<1, 2>(q, Aq, SA, B, SB, dst, m, n, k);
    } else if (t2_nsg() == 4) {
        if (m == 2) {
            n <= 8192 ? gemv<2, 4, 4>(q, Aq, SA, B, SB, dst, m, n, k) : gemv<2, 2, 4>(q, Aq, SA, B, SB, dst, m, n, k);
        } else if (m <= 4) {
            n <= 8192 ? gemv<4, 4, 4>(q, Aq, SA, B, SB, dst, m, n, k) : gemv<4, 2, 4>(q, Aq, SA, B, SB, dst, m, n, k);
        } else {
            n <= 8192 ? gemv<8, 4, 4>(q, Aq, SA, B, SB, dst, m, n, k) : gemv<8, 2, 4>(q, Aq, SA, B, SB, dst, m, n, k);
        }
    } else if (m == 2) {
        n <= 8192 ? gemv<2, 4>(q, Aq, SA, B, SB, dst, m, n, k) : gemv<2, 2>(q, Aq, SA, B, SB, dst, m, n, k);
    } else if (m <= 4) {
        n <= 8192 ? gemv<4, 4>(q, Aq, SA, B, SB, dst, m, n, k) : gemv<4, 2>(q, Aq, SA, B, SB, dst, m, n, k);
    } else {
        n <= 8192 ? gemv<8, 4>(q, Aq, SA, B, SB, dst, m, n, k) : gemv<8, 2>(q, Aq, SA, B, SB, dst, m, n, k);
    }
}
}  // namespace

namespace {
// what the activation scratch holds: the tensor it was quantized from and in which graph (one queue, in order)
struct quant_key {
    const void * key = nullptr; const float * x = nullptr; int64_t stride = 0, M = 0, K = 0; uint64_t graph = 0;
    bool operator==(const quant_key & o) const {
        return key == o.key && x == o.x && stride == o.stride && M == o.M && K == o.K && graph == o.graph;
    }
};
quant_key             g_quant_have;
std::atomic<uint64_t> g_graph{ 1 };
}  // namespace

void ggml_sycl_t2_new_graph() { g_graph.fetch_add(1); }

namespace {
// the activation scratch: int8 Aq [M, K], then fp16 SA [K/QK, ldsa(M)]
void scratch_aq(sycl::queue & q, int64_t M, int64_t K, int8_t *& Aq, uint16_t *& SA) {
    const int    lda      = int8dpas::ldsa((int) M);
    const size_t aq_bytes = (size_t) M * K;
    const size_t sa_off   = (aq_bytes + 255) & ~(size_t) 255;
    const size_t sa_bytes = (size_t) (K / QK) * lda * 2 + 256;
    char *       s        = (char *) scratch_get(q, sa_off + sa_bytes);
    Aq                    = (int8_t *) s;
    SA                    = (uint16_t *) (s + sa_off);
}
}  // namespace

namespace {
// one 1024 block of the rotation, already loaded and signed: v[j] = element lane + 16 j + 128 sgi (sgi = sub-group 0..7 of
// the block's 128 lanes), s = the block's 1024 floats of SLM. Writes the rotated block to dst (1024 floats) and its 8
// quant groups to Aq (1024 int8) + SA (column m, groups g0..g0+7). Butterflies in fwht.cpp's order (h = 1, 2, 4 ... 512).
// Every lane of the work-group must call it (barriers).
template <typename Item>
inline void rot_quant_block(float (&v)[8], float * s, const Item & it, const sycl::sub_group & sg, int lane, int sgi,
                            float * dst, int8_t * aq, uint16_t * SA, int64_t g0, int lda, int64_t m) {
#pragma unroll
    for (int h = 1; h < 16; h *= 2) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float o = sycl::permute_group_by_xor(sg, v[j], h);
            v[j]          = (lane & h) == 0 ? v[j] + o : o - v[j];
        }
    }
#pragma unroll
    for (int st = 1; st < 8; st *= 2) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if ((j & st) == 0) {
                const float a = v[j], b = v[j + st];
                v[j] = a + b; v[j + st] = a - b;
            }
        }
    }
    // layout 2: element t + 128 i (t = 16 sgi + lane); butterflies h = 128, 256, 512 in registers
    const int t = 16 * sgi + lane;
#pragma unroll
    for (int j = 0; j < 8; ++j) s[lane + 16 * j + 128 * sgi] = v[j];
    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = s[t + 128 * i];
#pragma unroll
    for (int st = 1; st < 8; st *= 2) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((i & st) == 0) {
                const float a = v[i], b = v[i + st];
                v[i] = a + b; v[i + st] = a - b;
            }
        }
    }
    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int i = 0; i < 8; ++i) s[t + 128 * i] = v[i];
    it.barrier(sycl::access::fence_space::local_space);
    // layout 1 again: sub-group sgi holds quant group sgi (128 values, 8 per lane) -> quant_a's numbers
    float mx = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = s[lane + 16 * j + 128 * sgi];
        mx   = sycl::fmax(mx, sycl::fabs(v[j]));
        dst[lane + 16 * j + 128 * sgi] = v[j];
    }
    mx                  = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
    const sycl::half sh = sycl::half(127.0f / sycl::fmax(mx, int8dpas::EPS));
    const float      sc = (float) sh;
    if (lane == 0) {
        SA[(g0 + sgi) * lda + m] = sycl::bit_cast<uint16_t>(sh);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        aq[128 * sgi + lane + 16 * j] = (int8_t) sycl::rint(sycl::clamp(v[j] * sc, -128.0f, 127.0f));
    }
}
}  // namespace

// signed 1024-point Hadamard rotation (as fwht.cpp: dst = H (x * signs) / 32, same butterfly order, so the same floats)
// that also quantizes its output for the T2 mat-vecs which read it next: one 128-lane work-group per 1024 block where
// fwht.cpp gives a block one 16-lane sub-group (5 sub-groups in all for a decode token), and no quant_a launch after it.
bool ggml_sycl_t2_fwht_quant(sycl::queue & q, const float * x, const float * signs, float * dst, int64_t M, int64_t K,
                             const void * key, const float * glu_g, const float * glu_u) {
    constexpr int N = 1024;
    if (K % N != 0 || M <= 0 || M > 512 || !ggml_sycl_knob("GGML_SYCL_PTQ1_T2_FWHTQ", 1)) {
        return false;
    }
    int8_t *   Aq;
    uint16_t * SA;
    scratch_aq(q, M, K, Aq, SA);
    const int     lda  = int8dpas::ldsa((int) M);
    const int64_t nblk = K / N, rows = M * nblk;
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sl(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<1>(rows * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto    sg   = it.get_sub_group();
            const int     lane = sg.get_local_linear_id();
            const int     sgi  = sg.get_group_linear_id();
            const int64_t r    = it.get_group(0);
            const int64_t m = r / nblk, kb = r % nblk;
            const float * sgn = signs + kb * N;
            float         v[8];
            if (glu_g) {  // SWIGLU folded in: x = silu(g) * u, as ggml_sycl_op_swiglu computes it
                const float * gp = glu_g + r * N;
                const float * up = glu_u + r * N;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int   e  = lane + 16 * j + 128 * sgi;
                    const float g  = gp[e];
                    const float xv = g / (1.0f + sycl::exp(-g)) * up[e];
                    v[j]           = xv * (1.0f / 32.0f) * sgn[e];
                }
            } else {
                const float * src = x + r * N;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int e = lane + 16 * j + 128 * sgi;
                    v[j]        = src[e] * (1.0f / 32.0f) * sgn[e];
                }
            }
            rot_quant_block(v, sl.get_multi_ptr<sycl::access::decorated::no>().get(), it, sg, lane, sgi, dst + r * N,
                            Aq + m * K + kb * N, SA, kb * (N / QK), lda, m);
        });
    });
    g_quant_have = quant_key{ key, dst, K, M, K, g_graph.load() };
    return true;
}

// residual add + RMS norm + norm weight (+ the signed rotation + quantize of ggml_sycl_t2_fwht_quant): one work-group
// per token, K / 1024 blocks of 128 lanes. Every read of a and b happens before the row's sum of squares is reduced
// (a work-group barrier), so outputs may reuse an input's memory within a token; the caller checks across tokens.
bool ggml_sycl_t2_add_norm_rot(sycl::queue & q, const float * a, const float * b, float * add_out, const float * w,
                               float eps, float * normed, const float * signs, float * mm, int64_t M, int64_t K,
                               const void * key) {
    constexpr int N = 1024;
    if (K % N != 0 || K > 8 * N || M <= 0 || M > 512 || !ggml_sycl_knob("GGML_SYCL_PTQ1_T2_NORMQ", 1)) {
        return false;
    }
    const bool rot = signs != nullptr;
    int8_t *   Aq  = nullptr;
    uint16_t * SA  = nullptr;
    if (rot) {
        scratch_aq(q, M, K, Aq, SA);
    }
    const int lda = int8dpas::ldsa((int) M);
    const int nt  = (int) (K / N) * 128;
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sl(sycl::range<1>(rot ? K : 1), cgh);
        cgh.parallel_for(sycl::nd_range<1>(M * nt, nt), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto    sg   = it.get_sub_group();
            const int     lane = sg.get_local_linear_id();
            const int     sgid = sg.get_group_linear_id();
            const int     kb = sgid / 8, sgi = sgid % 8;
            const int64_t m    = it.get_group(0);
            const int64_t base = m * K + kb * N;
            float         v[8], ss = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int64_t e = base + lane + 16 * j + 128 * sgi;
                v[j]            = b ? a[e] + b[e] : a[e];
                ss += v[j] * v[j];
            }
            ss                = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float scale = sycl::rsqrt(ss / K + eps);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int     c = kb * N + lane + 16 * j + 128 * sgi;
                const int64_t e = m * K + c;
                if (add_out) {
                    add_out[e] = v[j];
                }
                v[j] = scale * v[j] * w[c];
                if (normed) {
                    normed[e] = v[j];
                }
                if (rot) {
                    v[j] = v[j] * (1.0f / 32.0f) * signs[c];
                }
            }
            if (rot) {
                rot_quant_block(v, sl.get_multi_ptr<sycl::access::decorated::no>().get() + kb * N, it, sg, lane, sgi,
                                mm + base, Aq + base, SA, (int64_t) kb * (N / QK), lda, m);
            }
        });
    });
    if (rot) {
        g_quant_have = quant_key{ key, mm, K, M, K, g_graph.load() };
    }
    return true;
}

void ggml_sycl_t2_mul_mat(sycl::queue & q, const void * w, const float * x, int64_t x_stride, float * dst, int64_t M,
                          int64_t N, int64_t K, const void * x_key) {
    // batches > 8 (n-gram verify, prompts): GemmMT reads each weight once per 64 rows (TernSYCL main: the 2D block I/O
    // that misbehaved in the JIT build with the old IGC builtins validates under inline vISA, so it is upstream's kernel).
    const bool gemm_off = ggml_sycl_knob("GGML_SYCL_PTQ1_T2_GEMM_OFF", 0) != 0;
    // GGML_SYCL_PTQ1_T2_GEMV_MAX: largest batch on the GEMV (default 8; above it GemmMT). GEMV re-reads the weights once per
    // 8 rows, GemmMT once per 64 but less efficiently: `arctune sweep` measures the crossover.
    const bool use_gemm = M > ggml_sycl_knob("GGML_SYCL_PTQ1_T2_GEMV_MAX", 8) && !gemm_off;

    int8_t *       Aq;
    uint16_t *     SA;
    scratch_aq(q, M, K, Aq, SA);
    const uint32_t * B      = (const uint32_t *) w;
    const uint16_t * SB     = (const uint16_t *) ((const char *) w + (size_t) (K / 16) * N * 4);

    // GGML_SYCL_PTQ1_T2_QCACHE=0: quantize for every call
    const quant_key want{ x_key, x, x_stride, M, K, g_graph.load() };
    if (!x_key || !ggml_sycl_knob("GGML_SYCL_PTQ1_T2_QCACHE", 1) || !(g_quant_have == want)) {
        quant_a(q, x, x_stride, M, K, Aq, SA, QK);
        g_quant_have = x_key ? want : quant_key{};
    }
    t2_dispatch(q, Aq, SA, B, SB, dst, (int) M, (int) N, (int) K, use_gemm);
}

// several weights [K, N_m] on the same activations x [M, K] (M <= 8): one quantization (or the cached one) and one launch
bool ggml_sycl_t2_mul_mat_multi(sycl::queue & q, int nm, const void * const * w, float * const * dst, const int64_t * N,
                                const float * x, int64_t x_stride, int64_t M, int64_t K, const void * x_key) {
    if (nm < 2 || nm > 3 || M < 1 || M > 8 || !ggml_sycl_knob("GGML_SYCL_T2_MULTI", 1)) {
        return false;
    }
    int8_t *   Aq;
    uint16_t * SA;
    scratch_aq(q, M, K, Aq, SA);
    const quant_key want{ x_key, x, x_stride, M, K, g_graph.load() };
    if (!x_key || !ggml_sycl_knob("GGML_SYCL_PTQ1_T2_QCACHE", 1) || !(g_quant_have == want)) {
        quant_a(q, x, x_stride, M, K, Aq, SA, QK);
        g_quant_have = x_key ? want : quant_key{};
    }
    const uint32_t * B[3];
    const uint16_t * SB[3];
    float *          C[3];
    int              Ni[3];
    int64_t          nmax = 0;
    for (int m = 0; m < nm; ++m) {
        B[m]  = (const uint32_t *) w[m];
        SB[m] = (const uint16_t *) ((const char *) w[m] + (size_t) (K / 16) * N[m] * 4);
        C[m]  = dst[m];
        Ni[m] = (int) N[m];
        nmax  = std::max(nmax, N[m]);
    }
    const int  m_ = (int) M, k_ = (int) K;
    const bool ls4 = nmax <= 8192;
    if (t2_tuned(q, Aq, SA, nm, B, SB, C, Ni, m_, k_)) {
        return true;
    }
    if (M == 1) {
        ls4 ? gemv_multi<1, 4>(q, Aq, SA, nm, B, SB, C, Ni, m_, k_) : gemv_multi<1, 2>(q, Aq, SA, nm, B, SB, C, Ni, m_, k_);
    } else if (t2_nsg() == 4) {
        if (M == 2) {
            ls4 ? gemv_multi<2, 4, 4>(q, Aq, SA, nm, B, SB, C, Ni, m_, k_) : gemv_multi<2, 2, 4>(q, Aq, SA, nm, B, SB, C, Ni, m_, k_);
        } else if (M <= 4) {
            ls4 ? gemv_multi<4, 4, 4>(q, Aq, SA, nm, B, SB, C, Ni, m_, k_) : gemv_multi<4, 2, 4>(q, Aq, SA, nm, B, SB, C, Ni, m_, k_);
        } else {
            ls4 ? gemv_multi<8, 4, 4>(q, Aq, SA, nm, B, SB, C, Ni, m_, k_) : gemv_multi<8, 2, 4>(q, Aq, SA, nm, B, SB, C, Ni, m_, k_);
        }
    } else if (M == 2) {
        ls4 ? gemv_multi<2, 4>(q, Aq, SA, nm, B, SB, C, Ni, m_, k_) : gemv_multi<2, 2>(q, Aq, SA, nm, B, SB, C, Ni, m_, k_);
    } else if (M <= 4) {
        ls4 ? gemv_multi<4, 4>(q, Aq, SA, nm, B, SB, C, Ni, m_, k_) : gemv_multi<4, 2>(q, Aq, SA, nm, B, SB, C, Ni, m_, k_);
    } else {
        ls4 ? gemv_multi<8, 4>(q, Aq, SA, nm, B, SB, C, Ni, m_, k_) : gemv_multi<8, 2>(q, Aq, SA, nm, B, SB, C, Ni, m_, k_);
    }
    return true;
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
    for (int m : { 1, 2, 4, 8, M }) {  // the N <= 8192 GEMV tiles + GemmMT (the N > 8192 ones are called below)
        t2_dispatch(q, aq, sa, B, SB, d, m, N, K, m > 8);
    }
    gemv<1, 2>(q, aq, sa, B, SB, d, 1, N, K);
    gemv<2, 2>(q, aq, sa, B, SB, d, 2, N, K);
    gemv<4, 2>(q, aq, sa, B, SB, d, 4, N, K);
    gemv<8, 2>(q, aq, sa, B, SB, d, 8, N, K);
    if (t2_nsg() == 4) {  // the N > 8192 GEMV tiles with 4 sub-groups (t2_dispatch above covered the N <= 8192 ones)
        gemv<2, 2, 4>(q, aq, sa, B, SB, d, 2, N, K);
        gemv<4, 2, 4>(q, aq, sa, B, SB, d, 4, N, K);
        gemv<8, 2, 4>(q, aq, sa, B, SB, d, 8, N, K);
    }
    if (t2_tuned_on()) {  // GGML_SYCL_PTQ1_T2_TUNED: JIT the table's tiles now, not at the first verify
        for (const t2_tuned_kernel & t : t2_tuned_kernels()) {
            const int nn = N;
            t.fn(q, aq, sa, 1, &B, &SB, &d, &nn, 8, K, 0);
        }
    }
    q.wait();
    sycl::free(w, q);
    sycl::free(x, q);
    sycl::free(d, q);
    sycl::free(aq, q);
    sycl::free(sa, q);
}
}  // namespace

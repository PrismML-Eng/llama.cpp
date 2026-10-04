// SYCL port of TernOCL's int2_int8_dpas.cl, itself a port of the xetla int2 x
// int8 DPAS GEMM/GEMV (int2_fp16_dpas_xmx_xe.hpp, int2_bf16_dpas_xmx_xe.hpp):
//
//   C[m,n] = DT( epi( sum_g  float(sum_{k in g} q[m,k] * code(B[k,n]))
//                            * (SB[g,n] * (1 / SA[g,m])) ) )
//   q[m,k] = sat_int8_rtz(A[m,k] * SA[g,m]),  SA[g,m] = DT(127 / absmax_g(A[m,:]))
//
//   A  : DT [M, K] row-major             SA : DT [K/128, LDSA(M)]
//   B  : uint32 [K/16, N], 2-bit two's-complement codes of K rows
//        16*kp .. 16*kp+15 of column n (row 16*kp+j at bits 2j..2j+1)
//   SB : DT [K/128, N]                   C  : DT (or fp32) [M, N]
//   Aq : int8 [M, K] (QMODE 0 only)
//
// B never gets unpacked: the DPAS takes it natively (s8 x s2, K = 32); per
// lane (= column) the int2 B operand is two consecutive words of the layout.
// int32 accumulation within a 128-group, fp32 rescale per group.
//
// QMODE (A quantization):
//   0  upfront : quant_a writes SA and int8 Aq; the GEMM reads int8 A
//   1  xetla   : quant_a writes SA only (xetla external_scale_a_calc=1); the
//      GEMM quantizes the DT A tile on the fly (large-M: xetla's instruction
//      sequence as inline vISA, quant8x32)
#pragma once

#include "epilogue_dev.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace int8dpas {

using namespace xe2;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

constexpr int GS = 128;
// SA row pitch, padded so SA is a valid 2D block surface (pitch % 64 B == 0)
inline int ldsa(int M) { return (M + 31) & ~31; }
constexpr float EPS = 1.1920928955078125e-07f;  // FLT_EPSILON, xetla absmax init

// two DT (low, high half of h2) -> two int8 packed in a short, rounded to nearest
// even: + 1.5 * 2^23 rounds to nearest even into the low mantissa bits
// (|f * sa| <= 127.5), whose low byte is the int8
template <bool BF16> inline short q2(unsigned h2, float sa) {
    float f0, f1;
    if constexpr (BF16) {
        f0 = sycl::bit_cast<float>(h2 << 16);
        f1 = sycl::bit_cast<float>(h2 & 0xffff0000u);
    } else {
        f0 = dt<false>::tof((unsigned short)h2);
        f1 = dt<false>::tof((unsigned short)(h2 >> 16));
    }
    const unsigned u0 = sycl::bit_cast<unsigned>(sycl::fma(f0, sa, 12582912.0f));
    const unsigned u1 = sycl::bit_cast<unsigned>(sycl::fma(f1, sa, 12582912.0f));
    return (short)((u0 & 0xffu) | ((u1 & 0xffu) << 8));
}

// xetla's elemwise_scale_{fp16,bf16}_to_int8 on one 8-row x 32-K block, on the
// row-contiguous registers (SIMT code would split and re-pair every pair).
// a: row r = a[r] (lane l holds K 2l, 2l+1); lane r of sal = scale of row r.
// SIMD32 NoMask like xetla: call only from convergent code. a * sal + 1.5 * 2^23
// rounds to nearest even into the low mantissa bits (|a * sal| <= 127.5); the
// low byte of each float is the int8.
#define TS_QROW(r, g, orow, ocol)                                                         \
    "mad (M1_NM, 32) T(" #g ",0)<1> T(" #g ",0)<1;1,0> %2(0," #r ")<0;1,0> 0x4b400000:f\n" \
    "mov (M1_NM, 32) QB(" #orow "," #ocol ")<1> TB(" #g ",0)<4;1,0>\n"
#define TS_QUANT(A2F)                                                            \
    "{\n"                                                                        \
    ".decl AH v_type=G type=hf num_elts=256 align=GRF alias=<%1,0>\n"            \
    ".decl AW v_type=G type=uw num_elts=256 align=GRF alias=<%1,0>\n"            \
    ".decl QB v_type=G type=b num_elts=256 align=GRF alias=<%0,0>\n"             \
    ".decl T v_type=G type=f num_elts=256 align=GRF\n"                           \
    ".decl TD v_type=G type=ud num_elts=256 align=GRF alias=<T,0>\n"             \
    ".decl TB v_type=G type=b num_elts=1024 align=GRF alias=<T,0>\n"             \
    A2F(0, 0) TS_QROW(0, 0, 0, 0) A2F(1, 2) TS_QROW(1, 2, 0, 32)                 \
    A2F(2, 4) TS_QROW(2, 4, 1, 0) A2F(3, 6) TS_QROW(3, 6, 1, 32)                 \
    A2F(4, 8) TS_QROW(4, 8, 2, 0) A2F(5, 10) TS_QROW(5, 10, 2, 32)               \
    A2F(6, 12) TS_QROW(6, 12, 3, 0) A2F(7, 14) TS_QROW(7, 14, 3, 32)             \
    "}\n"
#define TS_A2F_BF16(r, g) "shl (M1_NM, 32) TD(" #g ",0)<1> AW(" #r ",0)<1;1,0> 0x10:uw\n"
#define TS_A2F_FP16(r, g) "mov (M1_NM, 32) T(" #g ",0)<1> AH(" #r ",0)<1;1,0>\n"

template <bool BF16> inline short8 quant8x32(uint8 a, float sal) {
    short8 q;
#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (BF16) __asm__ volatile(TS_QUANT(TS_A2F_BF16) : "=rw"(q) : "rw"(a), "rw"(sal));
    else __asm__ volatile(TS_QUANT(TS_A2F_FP16) : "=rw"(q) : "rw"(a), "rw"(sal));
#endif
    return q;
}

// ---------------------------------------------------------------------------
// Pre-kernel: one sub-group per (row, 128-group). SA always, Aq if WRITE_Q.
template <bool BF16, bool WRITE_Q>
struct QuantA {
    const unsigned short *A;
    unsigned short *SA;
    signed char *Aq;
    int M, K;

    void operator()(sycl::nd_item<2> it) const {
        using D = dt<BF16>;
        const auto sg = it.get_sub_group();
        const int lane = sg.get_local_linear_id();
        const int g = (int)it.get_global_id(1) / 16;
        const int m = (int)it.get_global_id(0);
        if (g >= K / GS || m >= M) return;
        const size_t off = (size_t)m * K + g * GS + 8 * lane;
        const ushort8 h = *(const ushort8 *)(A + off);
        float a[8], mx = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            a[i] = D::tof(h[i]);
            mx = sycl::fmax(mx, sycl::fabs(a[i]));
        }
        mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
        const unsigned short sh = D::from(127.0f / sycl::fmax(mx, EPS));
        if (lane == 0) SA[(size_t)g * ldsa(M) + m] = sh;
        if constexpr (WRITE_Q) {
            const float s = D::tof(sh);
            unsigned long long q = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                q |= (unsigned long long)(unsigned char)(signed char)sycl::clamp(sycl::rint(a[i] * s), -128.0f, 127.0f)
                        << (8 * i);
            *(unsigned long long *)(Aq + off) = q;
        }
    }

    auto get(syclex::properties_tag) const {
        return syclex::properties{syclex::sub_group_size<16>, syclex::work_group_size<1, 16>};
    }
};

template <int SGM> struct rows;
template <> struct rows<1> { using a_t = short; using ia_t = int; using fa_t = float; };
template <> struct rows<2> { using a_t = short2; using ia_t = int2; using fa_t = float2; };
template <> struct rows<4> { using a_t = short4; using ia_t = int4; using fa_t = float4; };
template <> struct rows<8> { using a_t = short8; using ia_t = int8; using fa_t = float8; };

template <int SGM, class V> inline auto el(const V &v, int r) {
    if constexpr (SGM == 1) return v;
    else return v[r];
}
template <int SGM, class V, class T> inline void set_el(V &v, int r, T x) {
    if constexpr (SGM == 1) v = x;
    else v[r] = x;
}

// ---------------------------------------------------------------------------
// GEMV / small M. A sub-group owns 16 columns and SGM rows and walks its K
// slice in 128-steps: one 2D block read of 8 B words (4 DPAS of K = 32), one
// SB row, SGM A rows (quantized in registers for QMODE 1).
//   SGM rows per sub-group (1, 2, 4, 8)   NSG_N sub-groups along N
//   LS  K-slices per column block, reduced through SLM
//   U   128-steps whose loads are issued before any compute
template <bool BF16, int QMODE, int SGM, int NSG_N, int LS, int U>
struct Gemv {
    const unsigned short *A;
    const signed char *Aq;
    const unsigned short *SA;
    const unsigned *B;
    const unsigned short *SB;
    void *C;
    Epi epi;
    int M, N, K;

    using a_t = typename rows<SGM>::a_t;
    using ia_t = typename rows<SGM>::ia_t;
    using fa_t = typename rows<SGM>::fa_t;
    using D = dt<BF16>;
    static constexpr int WG = 16 * NSG_N * LS;

    // SGM x 128 A tile of step s: aq[c] = K 32c..32c+31, inv[r] = 1 / SA of row r
    void load_a(int m0, int s, a_t *aq, float *inv) const {
        const int lda = ldsa(M);
#pragma unroll
        for (int r = 0; r < SGM; ++r) {
            const bool ok = m0 + r < M;
            const size_t row = (size_t)sycl::min(m0 + r, M - 1) * K + s * GS;
            if constexpr (QMODE == 0) {
                const ushort4 l = sg_rd_us4((const unsigned short *)(Aq + row));
                const ushort4 v = ok ? l : ushort4{};
#pragma unroll
                for (int c = 0; c < 4; ++c) set_el<SGM>(aq[c], r, (short)v[c]);
                inv[r] = ok ? sycl::native::recip(D::tof(SA[(size_t)s * lda + m0 + r])) : 0.0f;
            } else {
                const uint4 l = sg_rd_u4((const unsigned *)(A + row));
                const uint4 v = ok ? l : uint4{};
                const float sa = ok ? D::tof(SA[(size_t)s * lda + m0 + r]) : 0.0f;
#pragma unroll
                for (int c = 0; c < 4; ++c) set_el<SGM>(aq[c], r, q2<BF16>(v[c], sa));
                inv[r] = ok ? sycl::native::recip(sa) : 0.0f;
            }
        }
    }

    static fa_t step(fa_t acc, const uint8 &w, float sb, const a_t *aq, const float *inv) {
        ia_t ia = dpas_s2s8_z(aq[0], int2{(int)w[0], (int)w[1]});
#pragma unroll
        for (int c = 1; c < 4; ++c)
            ia = dpas_s2s8(aq[c], int2{(int)w[2 * c], (int)w[2 * c + 1]}, ia);
#pragma unroll
        for (int r = 0; r < SGM; ++r)
            set_el<SGM>(acc, r, el<SGM>(acc, r) + (float)el<SGM>(ia, r) * (sb * inv[r]));
        return acc;
    }

    void operator()(sycl::nd_item<2> it) const {
        const auto sgp = it.get_sub_group();
        const int lane = sgp.get_local_linear_id();
        const int sg = sgp.get_group_linear_id();
        const int sgn = sg % NSG_N, sgk = sg / NSG_N;
        const int n0 = ((int)it.get_group(1) * NSG_N + sgn) * 16;
        const int m0 = (int)it.get_group(0) * SGM;

        const int nsteps = K / GS;
        const int per = (nsteps + LS - 1) / LS;
        const int s_begin = sgk * per;
        const int s_end = sycl::min(nsteps, s_begin + per);
        const surf sbs(B, N * 4, K / 16, N * 4);

        fa_t acc = 0.0f;
        if (n0 < N) {
            int s = s_begin;
#pragma unroll 1
            for (; s + U <= s_end; s += U) {
                uint8 w[U];
                float sb[U];
                a_t aq[U][4];
                float inv[U][SGM];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    w[u] = rd_32b_8r16(sbs, n0, (s + u) * 8);
                    sb[u] = D::tof(sg_rd_us(SB + (size_t)(s + u) * N + n0));
                    load_a(m0, s + u, aq[u], inv[u]);
                }
#pragma unroll
                for (int u = 0; u < U; ++u) acc = step(acc, w[u], sb[u], aq[u], inv[u]);
            }
            if constexpr (U > 1)
            for (; s < s_end; ++s) {
                a_t aq[4];
                float inv[SGM];
                const uint8 w = rd_32b_8r16(sbs, n0, s * 8);
                const float sb = D::tof(sg_rd_us(SB + (size_t)s * N + n0));
                load_a(m0, s, aq, inv);
                acc = step(acc, w, sb, aq, inv);
            }
        }

        if constexpr (LS > 1) {
            float *red = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[(LS - 1) * NSG_N * SGM * 16]>(
                    it.get_group());
            if (sgk > 0) {
                float *dst = red + (((sgk - 1) * NSG_N + sgn) * SGM) * 16;
#pragma unroll
                for (int r = 0; r < SGM; ++r) dst[r * 16 + lane] = el<SGM>(acc, r);
            }
            barrier_local();
            if (sgk > 0) return;
            for (int j = 0; j < LS - 1; ++j) {
                const float *src = red + ((j * NSG_N + sgn) * SGM) * 16;
#pragma unroll
                for (int r = 0; r < SGM; ++r) set_el<SGM>(acc, r, el<SGM>(acc, r) + src[r * 16 + lane]);
            }
        }

        if (n0 >= N) return;
#pragma unroll
        for (int r = 0; r < SGM; ++r)
            if (m0 + r < M)
                epi_dev<BF16>::store(C, epi, (size_t)(m0 + r) * N + n0 + lane, n0 + lane, el<SGM>(acc, r));
    }

    auto get(syclex::properties_tag) const {
        return syclex::properties{syclex::sub_group_size<16>, syclex::work_group_size<1, WG>};
    }
};

// ---------------------------------------------------------------------------
// Large-M GEMM. A sub-group computes an MT_M x MT_N tile (MT_M % 8 == 0,
// MT_N % 16 == 0); a work-group is WG_M x WG_N sub-groups. Per 128-group:
// B (8 words per 16-column block) and SB are loaded once and reused for all
// MT_M/8 row blocks; each 8 x 128 A block is loaded (and for QMODE 1
// quantized) once and reused for all MT_N/16 column blocks. 2D block I/O
// zero-fills out-of-range reads and clips writes, so any M works. 256 GRF.
// The epilogue is a template parameter here (as in the OpenCL -D build): a
// runtime-selected one costs ~3% in this kernel.
template <bool BF16, int QMODE, int MT_M, int MT_N, int WG_M, int WG_N, int POSTOP, bool F32>
struct GemmMT {
    const unsigned short *A;
    const signed char *Aq;
    const unsigned short *SA;
    const unsigned *B;
    const unsigned short *SB;
    void *C;
    Epi epi;
    int M, N, K;

    using D = dt<BF16>;
    static constexpr int MB = MT_M / 8, NB = MT_N / 16, WG = 16 * WG_M * WG_N;

    void operator()(sycl::nd_item<2> it) const {
        const auto sgp = it.get_sub_group();
        const int sg = sgp.get_group_linear_id();
        const int m0 = ((int)it.get_group(0) * WG_M + sg / WG_N) * MT_M;
        const int n0 = ((int)it.get_group(1) * WG_N + sg % WG_N) * MT_N;
        const surf sbs(B, N * 4, K / 16, N * 4), ssb(SB, N * 2, K / GS, N * 2);
        // lane l gets SA[s, m + l] (pad columns past M are junk)
        const surf ssa(SA, ldsa(M) * 2, K / GS, ldsa(M) * 2);

        float8 acc[MB][NB];
#pragma unroll
        for (int i = 0; i < MB; ++i)
#pragma unroll
            for (int j = 0; j < NB; ++j) acc[i][j] = 0.0f;

        // 2D payloads built once; each K step only moves y (B, SB, SA) or x (A)
        unsigned pb = pl2d<b32_16x8>(sbs, n0, 0), psb2 = pl2d<b16_32x1>(ssb, n0, 0),
                 psb1 = pl2d<b16_16x1>(ssb, n0, 0), psa = pl2d<b16_16x1>(ssa, m0, 0), pq;
        if constexpr (QMODE == 0) pq = pl2d<b16_2x16x8>(surf(Aq, K, M, K), 0, m0);
        else pq = pl2d<b32_16x8>(surf(A, K * 2, M, K * 2), 0, m0);

        for (int s = 0; s < K / GS; ++s) {
            // A (and SA) loads first: they feed the quantization / unpacking that the
            // first dpas waits on (the asm reads are issued in source order). Row
            // block 0 before B and SB, block I + 1 at the start of block I.
            pl2d_y(psa, s);
            pl2d_x(pq, s * GS / 2);
            unsigned short sar[MB];
            std::conditional_t<QMODE == 0, ushort16[2], uint8[4]> ar[MB];
            auto load_a = [&](auto ii) {
                constexpr int I = decltype(ii)::value;
                sar[I] = rd2d<b16_16x1, 8 * I, 0, unsigned short>(psa);
                if constexpr (QMODE == 0)
                    static_for<2>([&](auto h) { ar[I][h] = rd2d<b16_2x16x8, 32 * decltype(h)::value, 8 * I, ushort16>(pq); });
                else
                    static_for<4>([&](auto c) { ar[I][c] = rd2d<b32_16x8, 16 * decltype(c)::value, 8 * I, uint8>(pq); });
            };
            load_a(std::integral_constant<int, 0>{});
            uint8 w[NB];
            float sb[NB];
            pl2d_y(pb, s * 8);
            static_for<NB>([&](auto j) { w[j] = rd2d<b32_16x8, 16 * decltype(j)::value, 0, uint8>(pb); });
            // all SB loads first, then convert: converting each (bf16: via acc0) before the
            // next load let IGC reuse one load register and serialize the loads
            ushort2 sbr[(NB + 1) / 2];
            pl2d_y(psb2, s);
            pl2d_y(psb1, s);
            static_for<(NB + 1) / 2>([&](auto h) {
                constexpr int J = 2 * decltype(h)::value;
                if constexpr (J + 1 < NB) sbr[h] = rd2d<b16_32x1, 16 * J, 0, ushort2>(psb2);
                else sbr[h] = ushort2{rd2d<b16_16x1, 16 * J, 0, unsigned short>(psb1), 0};
            });
#pragma unroll
            for (int j = 0; j < NB; ++j) sb[j] = D::tof(sbr[j / 2][j % 2]);
            static_for<MB>([&](auto ii) {
                constexpr int I = decltype(ii)::value;
                if constexpr (I + 1 < MB) load_a(std::integral_constant<int, I + 1>{});
                short8 aq[4];
                float inv[8];
                const float sal = D::tof(sar[I]);
                if constexpr (QMODE == 0) {
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            aq[2 * h][r] = (short)ar[I][h][r];
                            aq[2 * h + 1][r] = (short)ar[I][h][8 + r];
                        }
                } else {
#pragma unroll
                    for (int c = 0; c < 4; ++c) aq[c] = quant8x32<BF16>(ar[I][c], sal);
                }
                // rows >= M get inf here, but their int32 dot is 0 and the store clips them
                const float invl = sycl::native::recip(sal);
#pragma unroll
                for (int r = 0; r < 8; ++r) inv[r] = sycl::group_broadcast(sgp, invl, r);
#pragma unroll
                for (int j = 0; j < NB; ++j) {
                    int8 ia = dpas_s2s8_z(aq[0], int2{(int)w[j][0], (int)w[j][1]});
#pragma unroll
                    for (int c = 1; c < 4; ++c)
                        ia = dpas_s2s8(aq[c], int2{(int)w[j][2 * c], (int)w[j][2 * c + 1]}, ia);
                    // whole-vector convert (OpenCL's convert_float8): per-element casts
                    // are emitted through one scratch register and stall every mad
                    const float8 fi = __builtin_convertvector(ia, float8);
#pragma unroll
                    for (int r = 0; r < 8; ++r) acc[I][j][r] += fi[r] * (sb[j] * inv[r]);
                }
            });
        }

#pragma unroll
        for (int i = 0; i < MB; ++i)
#pragma unroll
            for (int j = 0; j < NB; ++j)
                epi_dev<BF16>::template store8x16<POSTOP, F32>(C, epi, M, N, m0 + 8 * i, n0 + 16 * j, acc[i][j]);
    }

    auto get(syclex::properties_tag) const {
        return syclex::properties{syclex::sub_group_size<16>, syclex::work_group_size<1, WG>,
                intelex::grf_size<256>};
    }
};

}  // namespace int8dpas

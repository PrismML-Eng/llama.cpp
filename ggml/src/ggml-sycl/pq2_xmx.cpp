//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// The Xe2 helpers and the GEMV / GEMM kernels are ported from TernSYCL int2_via_int2_x_int8_dpas
// (https://github.com/libxsmm/TernSYCL), distributed under this license:
//
// BSD 3-Clause License
//
// Copyright (c) 2026, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "pq2_xmx.hpp"

#if defined(__INTEL_LLVM_COMPILER) && !defined(GGML_SYCL_NO_PQ2_XMX)
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <type_traits>
#include <utility>

// Kernel functor types must not be in an anonymous namespace (SYCL kernel names).
namespace ggml_sycl_xmx {

namespace syclex  = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

#define XE2_VEC(T, n, name) typedef T name __attribute__((ext_vector_type(n)))
XE2_VEC(short, 2, short2);
XE2_VEC(short, 4, short4);
XE2_VEC(short, 8, short8);
XE2_VEC(unsigned short, 2, ushort2);
XE2_VEC(unsigned short, 4, ushort4);
XE2_VEC(unsigned short, 16, ushort16);
XE2_VEC(int, 2, int2);
XE2_VEC(int, 4, int4);
XE2_VEC(int, 8, int8);
XE2_VEC(unsigned, 8, uint8);
XE2_VEC(float, 2, float2);
XE2_VEC(float, 4, float4);
XE2_VEC(float, 8, float8);
#undef XE2_VEC

constexpr int   GS  = QK_PQ2_0;
constexpr float EPS = 1.1920928955078125e-07f;  // FLT_EPSILON

static_assert(GS == 128 && sizeof(block_pq2_0) == 34, "PQ2_0 layout changed");

// SA row pitch, padded so SA is a valid 2D block surface
inline int ldsa(int M) { return (M + 31) & ~31; }

// 2D surface: width and pitch in bytes, height in rows
struct surf {
    long long base;
    int w, h, p;
    surf(const void * b, int width_bytes, int height, int pitch_bytes) :
        base((long long) b), w(width_bytes - 1), h(height - 1), p(pitch_bytes - 1) {}
};

#ifdef __SYCL_DEVICE_ONLY__
#define XE2_ASM(...)   __asm__(__VA_ARGS__)
#define XE2_ASM_V(...) __asm__ volatile(__VA_ARGS__)
#else
#define XE2_ASM(...)
#define XE2_ASM_V(...)
#endif

// lsc 2D block read / write on an explicit surface: flat[base, width-1, height-1, pitch-1, x, y]
inline uint8 rd_32b_8r16(const surf & s, int x, int y) {
    uint8 v;
    XE2_ASM("{\n"
            ".decl SB v_type=G type=q num_elts=1 align=qword alias=<%1,0>\n"
            ".decl SW v_type=G type=d num_elts=1 align=dword alias=<%2,0>\n"
            ".decl SH v_type=G type=d num_elts=1 align=dword alias=<%3,0>\n"
            ".decl SP v_type=G type=d num_elts=1 align=dword alias=<%4,0>\n"
            ".decl SX v_type=G type=d num_elts=1 align=dword alias=<%5,0>\n"
            ".decl SY v_type=G type=d num_elts=1 align=dword alias=<%6,0>\n"
            "lsc_load_block2d.ugm (M1, 1) %0:d32.16x8nn flat[SB,SW,SH,SP,SX,SY]\n"
            "}\n"
            : "=rw"(v)
            : "rw.u"(s.base), "rw.u"(s.w), "rw.u"(s.h), "rw.u"(s.p), "rw.u"(x), "rw.u"(y));
    return v;
}

inline void wr_32b_8r16(const surf & s, int x, int y, uint8 v) {
    XE2_ASM_V("{\n"
              ".decl SB v_type=G type=q num_elts=1 align=qword alias=<%0,0>\n"
              ".decl SW v_type=G type=d num_elts=1 align=dword alias=<%1,0>\n"
              ".decl SH v_type=G type=d num_elts=1 align=dword alias=<%2,0>\n"
              ".decl SP v_type=G type=d num_elts=1 align=dword alias=<%3,0>\n"
              ".decl SX v_type=G type=d num_elts=1 align=dword alias=<%4,0>\n"
              ".decl SY v_type=G type=d num_elts=1 align=dword alias=<%5,0>\n"
              "lsc_store_block2d.ugm (M1, 1) flat[SB,SW,SH,SP,SX,SY] %6:d32.16x8nn\n"
              "}\n"
              :: "rw.u"(s.base), "rw.u"(s.w), "rw.u"(s.h), "rw.u"(s.p), "rw.u"(x), "rw.u"(y), "rw"(v));
}

// 2D block reads from a prebuilt address payload plus immediate (DX, DY) offsets.
// The vISA text is built at compile time.
namespace detail {
template <size_t N> struct cstr {
    char   s[N]{};
    size_t n = 0;
    constexpr size_t size() const { return n; }
    constexpr const char * data() const { return s; }
    constexpr void add(const char * p) {
        while (*p) {
            s[n++] = *p++;
        }
    }
    constexpr void addi(int v) {
        if (v < 0) {
            s[n++] = '-';
            v      = -v;
        }
        char t[12]{};
        int  k = 0;
        do {
            t[k++] = char('0' + v % 10);
            v /= 10;
        } while (v);
        while (k) {
            s[n++] = t[--k];
        }
    }
};

template <class SH, int DX, int DY> constexpr auto rd2d_str() {
    cstr<320> c;
    c.add("{\n.decl PD v_type=G type=ud num_elts=8 align=GRF alias=<%1,0>\n");
    if constexpr (SH::pad) {
        c.add(".decl TP v_type=G type=uw num_elts=32 align=GRF\nlsc_load_block2d.ugm (M1, 1) TP:");
    } else {
        c.add("lsc_load_block2d.ugm (M1, 1) %0:");
    }
    c.add(SH::v);
    c.add(" flat[PD + (");
    c.addi(DX);
    c.add(",");
    c.addi(DY);
    c.add(")]\n");
    if constexpr (SH::pad) {
        c.add("mov (M1, 16) %0(0,0)<1> TP(0,0)<1;1,0>\n");
    }
    c.add("}\n");
    return c;
}
}  // namespace detail

// block shape: vISA type and payload dword 7 = (V-1) << 16 | (R-1) << 8 | (C-1).
// pad: the block is half a GRF but the load writes a whole GRF.
struct b32_16x1   { static constexpr const char * v = "d32.16x1nn";   static constexpr int code = 0x00f;   static constexpr bool pad = false; };
struct b32_16x8   { static constexpr const char * v = "d32.16x8nn";   static constexpr int code = 0x70f;   static constexpr bool pad = false; };
struct b16_2x16x8 { static constexpr const char * v = "d16.2x16x8nn"; static constexpr int code = 0x1070f; static constexpr bool pad = false; };
struct b16_32x1   { static constexpr const char * v = "d16.32x1nn";   static constexpr int code = 0x01f;   static constexpr bool pad = false; };
struct b16_16x1   { static constexpr const char * v = "d16.16x1nn";   static constexpr int code = 0x00f;   static constexpr bool pad = true;  };

template <class SH> inline unsigned pl2d(const surf & s, int x, int y) {
    unsigned pl;
    XE2_ASM("{\n"
            ".decl PQ v_type=G type=uq num_elts=4 align=GRF alias=<%0,0>\n"
            ".decl PD v_type=G type=ud num_elts=8 align=GRF alias=<%0,0>\n"
            "mov (M1_NM, 1) PQ(0,0)<1> %1(0,0)<0;1,0>\n"
            "mov (M1_NM, 1) PD(0,2)<1> %2(0,0)<0;1,0>\n"
            "mov (M1_NM, 1) PD(0,3)<1> %3(0,0)<0;1,0>\n"
            "mov (M1_NM, 1) PD(0,4)<1> %4(0,0)<0;1,0>\n"
            "mov (M1_NM, 1) PD(0,5)<1> %5(0,0)<0;1,0>\n"
            "mov (M1_NM, 1) PD(0,6)<1> %6(0,0)<0;1,0>\n"
            "mov (M1_NM, 1) PD(0,7)<1> %7(0,0)<0;1,0>\n"
            "}\n"
            : "=rw"(pl)
            : "rw.u"(s.base), "rw.u"(s.w), "rw.u"(s.h), "rw.u"(s.p), "rw.u"(x), "rw.u"(y), "rw.u"(SH::code));
    return pl;
}

// move the block x (dword 5) or y (dword 6) of a payload in place
inline void pl2d_x(unsigned & pl, int v) {
    XE2_ASM("{\n.decl PD v_type=G type=ud num_elts=8 align=GRF alias=<%0,0>\n"
            "mov (M1_NM, 1) PD(0,5)<1> %1(0,0)<0;1,0>\n}\n" : "+rw"(pl) : "rw.u"(v));
}

inline void pl2d_y(unsigned & pl, int v) {
    XE2_ASM("{\n.decl PD v_type=G type=ud num_elts=8 align=GRF alias=<%0,0>\n"
            "mov (M1_NM, 1) PD(0,6)<1> %1(0,0)<0;1,0>\n}\n" : "+rw"(pl) : "rw.u"(v));
}

template <class SH, int DX, int DY, class T> inline T rd2d(unsigned pl) {
    T v;
    XE2_ASM((detail::rd2d_str<SH, DX, DY>()) : "=rw"(v) : "rw"(pl));
    return v;
}

template <class F, int... I> inline void static_for_impl(F && f, std::integer_sequence<int, I...>) {
    (f(std::integral_constant<int, I>{}), ...);
}

template <int N, class F> inline void static_for(F && f) {
    static_for_impl(f, std::make_integer_sequence<int, N>{});
}

// Sub-group block reads, striped: element i of lane l = p[l + 16 i]. p must be uniform and 4-byte aligned.
inline unsigned short sg_rd_us(const unsigned short * p) {
    unsigned short v;
    XE2_ASM("{\n.decl TP v_type=G type=uw num_elts=32 align=GRF\n"
            ".decl AD v_type=G type=q num_elts=1 align=GRF\n"
            "mov (M1_NM, 1) AD(0,0)<1> %1(0,0)<0;1,0>\n"
            "lsc_load.ugm (M1_NM, 1) TP:d32x8t flat[AD]:a64\n"
            "mov (M1, 16) %0(0,0)<1> TP(0,0)<1;1,0>\n}\n"
            : "=rw"(v) : "rw.u"((long long) p));
    return v;
}

inline ushort4 sg_rd_us4(const unsigned short * p) {
    ushort4 v;
    XE2_ASM("{\n.decl AD v_type=G type=q num_elts=1 align=GRF\n"
            "mov (M1_NM, 1) AD(0,0)<1> %1(0,0)<0;1,0>\n"
            "lsc_load.ugm (M1_NM, 1) %0:d32x32t flat[AD]:a64\n}\n"
            : "=rw"(v) : "rw.u"((long long) p));
    return v;
}

// DPAS s8 A (K32, one short per row per lane) x s2 B (two dwords per lane), systolic depth 8, R rows of A
#define XE2_DPAS(A, C, R, na)                                                         \
    inline C dpas_s2s8(A a, int2 b, C acc) {                                          \
        XE2_ASM("{\n"                                                                 \
                ".decl DB v_type=G type=ud num_elts=32 align=GRF alias=<%1,0>\n"      \
                ".decl DA v_type=G type=ud num_elts=" #na " align=GRF alias=<%2,0>\n" \
                "dpas.s2.s8.8." #R " (M1, 16) %0.0 %0.0 DB.0 DA(0,0)\n"               \
                "}\n"                                                                 \
                : "+rw"(acc) : "rw"(b), "rw"(a));                                     \
        return acc;                                                                   \
    }                                                                                 \
    inline C dpas_s2s8_z(A a, int2 b) {                                               \
        C d;                                                                          \
        XE2_ASM("{\n"                                                                 \
                ".decl DB v_type=G type=ud num_elts=32 align=GRF alias=<%1,0>\n"      \
                ".decl DA v_type=G type=ud num_elts=" #na " align=GRF alias=<%2,0>\n" \
                "dpas.s2.s8.8." #R " (M1, 16) %0.0 %%null.0 DB.0 DA(0,0)\n"           \
                "}\n"                                                                 \
                : "=rw"(d) : "rw"(b), "rw"(a));                                       \
        return d;                                                                     \
    }
XE2_DPAS(short,  int,  1, 8)
XE2_DPAS(short2, int2, 2, 16)
XE2_DPAS(short4, int4, 4, 32)
XE2_DPAS(short8, int8, 8, 64)
#undef XE2_DPAS

inline float half_bits_to_float(unsigned short h) {
    return (float) sycl::bit_cast<sycl::half>(h);
}

// PQ2_0 stores value+1 in each 2-bit field. Subtract 1 per field without borrow across fields
// to get s2 two's complement.
inline uint32_t pq2_codes_to_s2(uint32_t x) {
    constexpr uint32_t H = 0xAAAAAAAAu;
    constexpr uint32_t L = 0x55555555u;
    return ((x | H) - L) ^ (~x & H);
}

// weight formats of the reordered layouts
enum { FMT_PQ2 = 0, FMT_PTQ1 = 1 };

// PTQ1_0 packs 5 trits per byte. The kernels decode them in byte-major K order (the trits of one
// byte next to each other): then the 10-bit LUT entries of consecutive bytes concatenate into the
// s2 dwords. The activation is quantized in the same order, see ptq1_k_orig().
constexpr int PTQ1_ROWS = 7;  // dwords per 128-group and column after the reorder: qs[24], qh[2] + d

inline int ptq1_trit(int b, int n) {
    constexpr uint8_t pow3[5] = { 1, 3, 9, 27, 81 };
    const uint8_t     q       = (uint8_t) (b * pow3[n]);
    return ((int) q * 3) >> 8;
}

// the s2 codes of the 5 trits of byte b, trit n at bits 2n
inline uint16_t ptq1_lut_entry(int b) {
    uint16_t e = 0;
    for (int n = 0; n < 5; ++n) {
        e |= ((ptq1_trit(b, n) + 3) & 3) << (2 * n);
    }
    return e;
}

// position p of the decode order -> index of that weight in the PTQ1_0 block
inline int ptq1_k_orig(int p) {
    if (p < 80) {
        return (p % 5) * 16 + p / 5;
    }
    if (p < 120) {
        return 80 + ((p - 80) % 5) * 8 + (p - 80) / 5;
    }
    return 120 + ((p - 120) % 4) * 2 + (p - 120) / 4;
}

// 7 raw dwords of a 128-group (r[7] unused) -> 8 s2 dwords, 16 weights each
inline uint8 ptq1_decode(const uint8 & r, const uint16_t * lut) {
    uint32_t e[16];
    uint32_t f[8];
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        e[m] = lut[(r[m / 4] >> (8 * (m % 4))) & 0xFF];
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        f[m] = lut[(r[4 + m / 4] >> (8 * (m % 4))) & 0xFF];
    }
    const uint32_t g0 = lut[r[6] & 0xFF] & 0xFF;
    const uint32_t g1 = lut[(r[6] >> 8) & 0xFF] & 0xFF;

    uint8 w;
    w[0] = e[0] | e[1] << 10 | e[2] << 20 | e[3] << 30;
    w[1] = e[3] >> 2 | e[4] << 8 | e[5] << 18 | e[6] << 28;
    w[2] = e[6] >> 4 | e[7] << 6 | e[8] << 16 | e[9] << 26;
    w[3] = e[9] >> 6 | e[10] << 4 | e[11] << 14 | e[12] << 24;
    w[4] = e[12] >> 8 | e[13] << 2 | e[14] << 12 | e[15] << 22;
    w[5] = f[0] | f[1] << 10 | f[2] << 20 | f[3] << 30;
    w[6] = f[3] >> 2 | f[4] << 8 | f[5] << 18 | f[6] << 28;
    w[7] = f[6] >> 4 | f[7] << 6 | g0 << 16 | g1 << 24;
    return w;
}

// fill the decode table in local memory; all work-items of the group must call it
inline uint16_t * ptq1_lut(sycl::nd_item<2> it) {
    uint16_t * lut = *sycl::ext::oneapi::group_local_memory_for_overwrite<uint16_t[256]>(it.get_group());
    for (int i = it.get_local_linear_id(); i < 256; i += it.get_local_range().size()) {
        lut[i] = ptq1_lut_entry(i);
    }
    sycl::group_barrier(it.get_group());
    return lut;
}

// fused epilogues on the fp32 result, numbered as TernSYCL postops. other has the layout of C.
enum { EPI_NONE = 0, EPI_SWIGLU = 1, EPI_ADD = 2 };

inline float epilogue(float v, const float * other, size_t i, int postop) {
    if (postop == EPI_SWIGLU) {
        return v / (1.0f + sycl::exp(-v)) * other[i];
    }
    if (postop == EPI_ADD) {
        return v + other[i];
    }
    return v;
}

// One sub-group per (row, 128-group): SA = 127 / absmax, Aq = rint(A * SA). Rows of src1 are
// flattened over dims 1..3.
struct quant_a {
    const float * A;
    float *       SA;
    int8_t *      Aq;
    int           M, K, ne11, ne12;
    int64_t       s11, s12, s13;
    int           perm;  // write each 128-group in PTQ1_0 decode order

    void operator()(sycl::nd_item<2> it) const {
        const auto sg   = it.get_sub_group();
        const int  lane = sg.get_local_linear_id();
        const int  g    = (int) it.get_global_id(1) / 16;
        const int  m    = (int) it.get_global_id(0);
        if (g >= K / GS || m >= M) {
            return;
        }
        const int     i1 = m % ne11;
        const int     i2 = (m / ne11) % ne12;
        const int     i3 = m / (ne11 * ne12);
        const float * a  = A + i1 * s11 + i2 * s12 + i3 * s13 + g * GS + 8 * lane;

        float v[8];
        float mx = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = a[i];
            mx   = sycl::fmax(mx, sycl::fabs(v[i]));
        }
        mx            = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
        const float s = 127.0f / sycl::fmax(mx, EPS);
        if (lane == 0) {
            SA[(size_t) g * ldsa(M) + m] = s;
        }
        if (perm) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                v[i] = a[ptq1_k_orig(8 * lane + i) - 8 * lane];
            }
        }
        uint64_t q = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q |= (uint64_t) (uint8_t) (int8_t) sycl::clamp(sycl::rint(v[i] * s), -128.0f, 127.0f) << (8 * i);
        }
        *(uint64_t *) (Aq + (size_t) m * K + g * GS + 8 * lane) = q;
    }

    auto get(syclex::properties_tag) const {
        return syclex::properties{ syclex::sub_group_size<16>, syclex::work_group_size<1, 16> };
    }
};

// quant_a with a sign flip and a 1024-wide normalized Walsh-Hadamard transform in front, as the
// Hadamard-folded weights expect: Aq = quant(FWHT(A * signs)). One work-group per (row, 1024-block).
// The butterflies follow ggml_sycl_op_fwht (fwht_kernel_wide, NT = 256).
struct quant_a_had {
    static constexpr int HN = 1024;
    static constexpr int NT = 256;
    static constexpr int EL = HN / NT;

    const float * A;
    const float * signs;
    float *       SA;
    int8_t *      Aq;
    float *       out;  // if set, write the transformed rows (contiguous [K, M]) instead of quantizing
    int           M, K, ne11, ne12;
    int64_t       s11, s12, s13;
    int           perm;  // quantize each 128-group in PTQ1_0 decode order

    void operator()(sycl::nd_item<2> it) const {
        float *    smem = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[HN]>(it.get_group());
        const auto sg   = it.get_sub_group();
        const int  lane = sg.get_local_linear_id();
        const int  tid  = it.get_local_id(1);
        const int  b    = it.get_group(1);
        const int  m    = it.get_group(0);

        const int     i1 = m % ne11;
        const int     i2 = (m / ne11) % ne12;
        const int     i3 = m / (ne11 * ne12);
        const float * a  = A + i1 * s11 + i2 * s12 + i3 * s13 + b * HN;
        const float * sg_ = signs + b * HN;

        float reg[EL];
#pragma unroll
        for (int i = 0; i < EL; ++i) {
            reg[i] = a[i * NT + tid] * sg_[i * NT + tid] * (1.0f / 32.0f);  // 1 / sqrt(1024)
        }
        // butterflies inside the sub-group
#pragma unroll
        for (int h = 1; h < 16; h *= 2) {
#pragma unroll
            for (int j = 0; j < EL; ++j) {
                const float v  = reg[j];
                const float v2 = sycl::permute_group_by_xor(sg, v, h);
                reg[j]         = (lane & h) == 0 ? v + v2 : v2 - v;
            }
        }
        // across sub-groups, through local memory
        for (int h = 16; h < NT; h *= 2) {
#pragma unroll
            for (int j = 0; j < EL; ++j) {
                smem[j * NT + tid] = reg[j];
            }
            sycl::group_barrier(it.get_group());
#pragma unroll
            for (int j = 0; j < EL; ++j) {
                const float v  = reg[j];
                const float v2 = smem[j * NT + (tid ^ h)];
                reg[j]         = (tid & h) == 0 ? v + v2 : v2 - v;
            }
            sycl::group_barrier(it.get_group());
        }
        // across registers
#pragma unroll
        for (int h = NT; h < HN; h *= 2) {
            const int step = h / NT;
#pragma unroll
            for (int j = 0; j < EL; j += 2 * step) {
#pragma unroll
                for (int k = 0; k < step; ++k) {
                    const float x = reg[j + k];
                    const float y = reg[j + k + step];
                    reg[j + k]        = x + y;
                    reg[j + k + step] = x - y;
                }
            }
        }
        if (out) {
            float * o = out + (size_t) m * K + b * HN;
#pragma unroll
            for (int j = 0; j < EL; ++j) {
                o[j * NT + tid] = reg[j];
            }
            return;
        }
#pragma unroll
        for (int j = 0; j < EL; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        sycl::group_barrier(it.get_group());

        // quantize as quant_a: sub-group g < 8 takes 128-group g of this block
        const int g = tid / 16;
        if (g >= HN / GS) {
            return;
        }
        float v[8];
        float mx = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = smem[g * GS + 8 * lane + i];
            mx   = sycl::fmax(mx, sycl::fabs(v[i]));
        }
        mx            = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
        const float s = 127.0f / sycl::fmax(mx, EPS);
        const int   gg = b * (HN / GS) + g;
        if (perm) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                v[i] = smem[g * GS + ptq1_k_orig(8 * lane + i)];
            }
        }
        if (lane == 0) {
            SA[(size_t) gg * ldsa(M) + m] = s;
        }
        uint64_t q = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q |= (uint64_t) (uint8_t) (int8_t) sycl::clamp(sycl::rint(v[i] * s), -128.0f, 127.0f) << (8 * i);
        }
        *(uint64_t *) (Aq + (size_t) m * K + gg * GS + 8 * lane) = q;
    }

    auto get(syclex::properties_tag) const {
        return syclex::properties{ syclex::sub_group_size<16>, syclex::work_group_size<1, NT> };
    }
};

template <int SGM> struct rows;
template <> struct rows<1> { using a_t = short;  using ia_t = int;  using fa_t = float;  };
template <> struct rows<2> { using a_t = short2; using ia_t = int2; using fa_t = float2; };
template <> struct rows<4> { using a_t = short4; using ia_t = int4; using fa_t = float4; };
template <> struct rows<8> { using a_t = short8; using ia_t = int8; using fa_t = float8; };

template <int SGM, class V> inline auto el(const V & v, int r) {
    if constexpr (SGM == 1) {
        return v;
    } else {
        return v[r];
    }
}

template <int SGM, class V, class T> inline void set_el(V & v, int r, T x) {
    if constexpr (SGM == 1) {
        v = x;
    } else {
        v[r] = x;
    }
}

// GEMV / small M. A sub-group owns 16 columns and SGM rows and walks its K slice in 128-steps:
// one 2D read of 8 B dwords (4 DPAS of K = 32), one SB row, SGM A rows.
//   NSG_N sub-groups along N, LS K-slices per column block (reduced in SLM),
//   U 128-steps whose loads are issued before any compute.
template <int FMT, int SGM, int NSG_N, int LS, int U> struct gemv {
    const int8_t *         Aq;
    const float *          SA;
    const uint32_t *       B;
    const unsigned short * SB;
    float *                C;
    const float *          other;
    int                    M, N, NP, K, ldc, postop;

    using a_t  = typename rows<SGM>::a_t;
    using ia_t = typename rows<SGM>::ia_t;
    using fa_t = typename rows<SGM>::fa_t;
    static constexpr int WG = 16 * NSG_N * LS;

    // SGM x 128 A tile of step s: aq[c] = K 32c..32c+31, inv[r] = 1 / SA of row r
    void load_a(int m0, int s, a_t * aq, float * inv) const {
        const int lda = ldsa(M);
#pragma unroll
        for (int r = 0; r < SGM; ++r) {
            const bool    ok  = m0 + r < M;
            const size_t  row = (size_t) sycl::min(m0 + r, M - 1) * K + s * GS;
            const ushort4 l   = sg_rd_us4((const unsigned short *) (Aq + row));
            const ushort4 v   = ok ? l : ushort4{};
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                set_el<SGM>(aq[c], r, (short) v[c]);
            }
            inv[r] = ok ? sycl::native::recip(SA[(size_t) s * lda + m0 + r]) : 0.0f;
        }
    }

    static fa_t step(fa_t acc, const uint8 & w, float sb, const a_t * aq, const float * inv) {
        ia_t ia = dpas_s2s8_z(aq[0], int2{ (int) w[0], (int) w[1] });
#pragma unroll
        for (int c = 1; c < 4; ++c) {
            ia = dpas_s2s8(aq[c], int2{ (int) w[2 * c], (int) w[2 * c + 1] }, ia);
        }
#pragma unroll
        for (int r = 0; r < SGM; ++r) {
            set_el<SGM>(acc, r, el<SGM>(acc, r) + (float) el<SGM>(ia, r) * (sb * inv[r]));
        }
        return acc;
    }

    void operator()(sycl::nd_item<2> it) const {
        const auto sgp  = it.get_sub_group();
        const int  lane = sgp.get_local_linear_id();
        const int  sg   = sgp.get_group_linear_id();
        const int  sgn  = sg % NSG_N;
        const int  sgk  = sg / NSG_N;
        const int  n0   = ((int) it.get_group(1) * NSG_N + sgn) * 16;
        const int  m0   = (int) it.get_group(0) * SGM;

        const int  nsteps  = K / GS;
        const int  per     = (nsteps + LS - 1) / LS;
        const int  s_begin = sgk * per;
        const int  s_end   = sycl::min(nsteps, s_begin + per);
        const int  rows    = FMT == FMT_PTQ1 ? PTQ1_ROWS : 8;  // B rows per 128-group
        const surf sbs(B, NP * 4, nsteps * rows, NP * 4);

        const uint16_t * lut = nullptr;
        if constexpr (FMT == FMT_PTQ1) {
            lut = ptq1_lut(it);
        }
        // B of step s: s2 codes, and the scale (stored with the codes for PTQ1_0)
        auto load_w = [&](int s, uint8 & w, float & sb) {
            w = rd_32b_8r16(sbs, n0, s * rows);
            if constexpr (FMT == FMT_PQ2) {
                sb = half_bits_to_float(sg_rd_us(SB + (size_t) s * NP + n0));
            }
        };
        auto decode = [&](uint8 & w, float & sb) {
            if constexpr (FMT == FMT_PTQ1) {
                sb = half_bits_to_float((unsigned short) (w[6] >> 16));
                w  = ptq1_decode(w, lut);
            }
        };

        fa_t acc = 0.0f;
        if (n0 < N) {
            int s = s_begin;
#pragma unroll 1
            for (; s + U <= s_end; s += U) {
                uint8 w[U];
                float sb[U];
                a_t   aq[U][4];
                float inv[U][SGM];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    load_w(s + u, w[u], sb[u]);
                    load_a(m0, s + u, aq[u], inv[u]);
                }
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    decode(w[u], sb[u]);
                    acc = step(acc, w[u], sb[u], aq[u], inv[u]);
                }
            }
            if constexpr (U > 1) {
                for (; s < s_end; ++s) {
                    a_t   aq[4];
                    float inv[SGM];
                    uint8 w;
                    float sb;
                    load_w(s, w, sb);
                    load_a(m0, s, aq, inv);
                    decode(w, sb);
                    acc = step(acc, w, sb, aq, inv);
                }
            }
        }

        if constexpr (LS > 1) {
            float * red = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[(LS - 1) * NSG_N * SGM * 16]>(
                it.get_group());
            if (sgk > 0) {
                float * dst = red + (((sgk - 1) * NSG_N + sgn) * SGM) * 16;
#pragma unroll
                for (int r = 0; r < SGM; ++r) {
                    dst[r * 16 + lane] = el<SGM>(acc, r);
                }
            }
            sycl::group_barrier(it.get_group());
            if (sgk > 0) {
                return;
            }
            for (int j = 0; j < LS - 1; ++j) {
                const float * src = red + ((j * NSG_N + sgn) * SGM) * 16;
#pragma unroll
                for (int r = 0; r < SGM; ++r) {
                    set_el<SGM>(acc, r, el<SGM>(acc, r) + src[r * 16 + lane]);
                }
            }
        }

        if (n0 + lane >= N) {
            return;
        }
#pragma unroll
        for (int r = 0; r < SGM; ++r) {
            if (m0 + r < M) {
                const size_t i = (size_t) (m0 + r) * ldc + n0 + lane;
                C[i]           = epilogue(el<SGM>(acc, r), other, i, postop);
            }
        }
    }

    auto get(syclex::properties_tag) const {
        return syclex::properties{ syclex::sub_group_size<16>, syclex::work_group_size<1, WG> };
    }
};

// Large-M GEMM. A sub-group computes an MT_M x MT_N tile, a work-group is WG_M x WG_N sub-groups.
// Per 128-group, B and SB are loaded once for all MT_M/8 row blocks and each 8 x 128 A block once
// for all MT_N/16 column blocks. 2D block I/O zero-fills out-of-range reads and clips writes.
template <int FMT, int MT_M, int MT_N, int WG_M, int WG_N> struct gemm {
    const int8_t *         Aq;
    const float *          SA;
    const uint32_t *       B;
    const unsigned short * SB;
    float *                C;
    const float *          other;
    int                    M, N, NP, K, ldc, postop;
    bool                   st2d;  // C is a valid 2D surface
    float *                part;  // with ks > 1: per K-slice results [ks][M][NP], summed by launch_gemm
    int                    ks;

    static constexpr int MB = MT_M / 8, NB = MT_N / 16, WG = 16 * WG_M * WG_N;

    void operator()(sycl::nd_item<2> it) const {
        const auto sgp = it.get_sub_group();
        const int  sg  = sgp.get_group_linear_id();
        const int  kz  = (int) it.get_group(0) % ks;  // K slice of this work-group
        const int  m0  = ((int) it.get_group(0) / ks * WG_M + sg / WG_N) * MT_M;
        const int  n0  = ((int) it.get_group(1) * WG_N + sg % WG_N) * MT_N;
        const int  rows = FMT == FMT_PTQ1 ? PTQ1_ROWS : 8;  // B rows per 128-group
        const surf sbs(B, NP * 4, K / GS * rows, NP * 4);

        const uint16_t * lut = nullptr;
        if constexpr (FMT == FMT_PTQ1) {
            lut = ptq1_lut(it);
        }
        const surf ssb(SB, NP * 2, K / GS, NP * 2);
        // lane l gets SA[s, m + l]; pad columns past M are junk
        const surf ssa(SA, ldsa(M) * 4, K / GS, ldsa(M) * 4);

        float8 acc[MB][NB];
#pragma unroll
        for (int i = 0; i < MB; ++i) {
#pragma unroll
            for (int j = 0; j < NB; ++j) {
                acc[i][j] = 0.0f;
            }
        }

        // 2D payloads built once; each K step only moves y (B, SB, SA) or x (A)
        unsigned pb   = pl2d<b32_16x8>(sbs, n0, 0);
        unsigned psb2 = pl2d<b16_32x1>(ssb, n0, 0);
        unsigned psb1 = pl2d<b16_16x1>(ssb, n0, 0);
        unsigned psa  = pl2d<b32_16x1>(ssa, m0, 0);
        unsigned pq   = pl2d<b16_2x16x8>(surf(Aq, K, M, K), 0, m0);

        // one 128-step: A for all row blocks, B and its scales through get_b
        auto step = [&](int s, auto && get_b) {
            // A and SA loads first, they feed the first dpas. Row block 0 before B and SB,
            // block I + 1 at the start of block I.
            pl2d_y(psa, s);
            pl2d_x(pq, s * GS / 2);
            unsigned sar[MB];
            ushort16 ar[MB][2];
            auto     load_a = [&](auto ii) {
                constexpr int I = decltype(ii)::value;
                sar[I]          = rd2d<b32_16x1, 8 * I, 0, unsigned>(psa);
                static_for<2>([&](auto h) {
                    ar[I][h] = rd2d<b16_2x16x8, 32 * decltype(h)::value, 8 * I, ushort16>(pq);
                });
            };
            load_a(std::integral_constant<int, 0>{});
            uint8 w[NB];
            float sb[NB];
            get_b(s, w, sb);
            static_for<MB>([&](auto ii) {
                constexpr int I = decltype(ii)::value;
                if constexpr (I + 1 < MB) {
                    load_a(std::integral_constant<int, I + 1>{});
                }
                short8 aq[4];
                float  inv[8];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        aq[2 * h][r]     = (short) ar[I][h][r];
                        aq[2 * h + 1][r] = (short) ar[I][h][8 + r];
                    }
                }
                // rows >= M get junk here, but their int32 dot is 0 and the store clips them
                const float invl = sycl::native::recip(sycl::bit_cast<float>(sar[I]));
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    inv[r] = sycl::group_broadcast(sgp, invl, r);
                }
#pragma unroll
                for (int j = 0; j < NB; ++j) {
                    int8 ia = dpas_s2s8_z(aq[0], int2{ (int) w[j][0], (int) w[j][1] });
#pragma unroll
                    for (int c = 1; c < 4; ++c) {
                        ia = dpas_s2s8(aq[c], int2{ (int) w[j][2 * c], (int) w[j][2 * c + 1] }, ia);
                    }
                    // whole-vector convert: per-element casts go through a scratch register
                    const float8 fi = __builtin_convertvector(ia, float8);
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        acc[I][j][r] += fi[r] * (sb[j] * inv[r]);
                    }
                }
            });
        };

        const int nsteps = K / GS;
        const int per    = (nsteps + ks - 1) / ks;
        const int kb     = kz * per;
        const int ke     = sycl::min(nsteps, kb + per);
        if constexpr (FMT == FMT_PQ2) {
            for (int s = kb; s < ke; ++s) {
                step(s, [&](int s, uint8 * w, float * sb) {
                    pl2d_y(pb, s * rows);
                    static_for<NB>([&](auto j) { w[j] = rd2d<b32_16x8, 16 * decltype(j)::value, 0, uint8>(pb); });
                    // all SB loads first, then convert, so the loads do not serialize on one register
                    ushort2 sbr[(NB + 1) / 2];
                    pl2d_y(psb2, s);
                    pl2d_y(psb1, s);
                    static_for<(NB + 1) / 2>([&](auto h) {
                        constexpr int J = 2 * decltype(h)::value;
                        if constexpr (J + 1 < NB) {
                            sbr[h] = rd2d<b16_32x1, 16 * J, 0, ushort2>(psb2);
                        } else {
                            sbr[h] = ushort2{ rd2d<b16_16x1, 16 * J, 0, unsigned short>(psb1), 0 };
                        }
                    });
#pragma unroll
                    for (int j = 0; j < NB; ++j) {
                        sb[j] = half_bits_to_float(sbr[j / 2][j % 2]);
                    }
                });
            }
        } else {
            // The WG_M sub-groups of a work-group column share their B columns: each decodes one of
            // every WG_M steps into local memory, then all of them use the WG_M decoded steps.
            const int  lane = sgp.get_local_linear_id();
            const int  wr   = sg / WG_N;
            const int  wc   = sg % WG_N;
            uint32_t * dec  = *sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t[WG_M * WG_N * NB * 8 * 16]>(
                it.get_group());
            float * dsb = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[WG_M * WG_N * NB * 16]>(
                it.get_group());
            auto slot = [&](int u, int j) { return (u * WG_N + wc) * NB + j; };

            for (int s0 = kb; s0 < ke; s0 += WG_M) {
                if (s0 + wr < ke) {
                    pl2d_y(pb, (s0 + wr) * rows);
                    static_for<NB>([&](auto jj) {
                        constexpr int J   = decltype(jj)::value;
                        const uint8   raw = rd2d<b32_16x8, 16 * J, 0, uint8>(pb);
                        const uint8   w   = ptq1_decode(raw, lut);
#pragma unroll
                        for (int d = 0; d < 8; ++d) {
                            dec[slot(wr, J) * 128 + d * 16 + lane] = w[d];
                        }
                        dsb[slot(wr, J) * 16 + lane] = half_bits_to_float((unsigned short) (raw[6] >> 16));
                    });
                }
                sycl::group_barrier(it.get_group());
                const int nu = sycl::min(WG_M, ke - s0);
                for (int u = 0; u < nu; ++u) {
                    step(s0 + u, [&](int, uint8 * w, float * sb) {
#pragma unroll
                        for (int j = 0; j < NB; ++j) {
#pragma unroll
                            for (int d = 0; d < 8; ++d) {
                                w[j][d] = dec[slot(u, j) * 128 + d * 16 + lane];
                            }
                            sb[j] = dsb[slot(u, j) * 16 + lane];
                        }
                    });
                }
                sycl::group_barrier(it.get_group());
            }
        }

        const int lane = sgp.get_local_linear_id();
        // a K slice writes its partial result; launch_gemm adds the slices up
        float * out = ks > 1 ? part + (size_t) kz * M * NP : C;
        const int  ldo = ks > 1 ? NP : ldc;
        // an epilogue is applied per element on the way out, so acc stays whole for the 2D store
        const int  epi = ks > 1 ? EPI_NONE : postop;
        const bool s2d = ks > 1 || (st2d && epi == EPI_NONE);
        if (s2d) {
            const surf sc(out, N * 4, M, ldo * 4);
#pragma unroll
            for (int i = 0; i < MB; ++i) {
#pragma unroll
                for (int j = 0; j < NB; ++j) {
                    wr_32b_8r16(sc, n0 + 16 * j, m0 + 8 * i, __builtin_bit_cast(uint8, acc[i][j]));
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < MB; ++i) {
#pragma unroll
                for (int j = 0; j < NB; ++j) {
                    const int n = n0 + 16 * j + lane;
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int m = m0 + 8 * i + r;
                        if (m < M && n < N) {
                            out[(size_t) m * ldo + n] = epilogue(acc[i][j][r], other, (size_t) m * ldo + n, epi);
                        }
                    }
                }
            }
        }
    }

    auto get(syclex::properties_tag) const {
        return syclex::properties{ syclex::sub_group_size<16>, syclex::work_group_size<1, WG>,
                                   intelex::grf_size<256> };
    }
};

struct args {
    const int8_t *         Aq;
    const float *          SA;
    const uint32_t *       B;
    const unsigned short * SB;
    float *                C;
    const float *          other;
    int                    M, N, NP, K, ldc, postop;
    bool                   st2d;
    ggml_sycl_pool *       pool;     // for the K-slice partials
    int                    threads;  // hardware threads of the device
};

template <int FMT, int SGM, int NSG, int LS, int U> static void launch_gemv(const args & a, dpct::queue_ptr stream) {
    using kern       = gemv<FMT, SGM, NSG, LS, U>;
    const size_t wgn = 16 * NSG;
    const sycl::range<2> local(1, kern::WG);
    const sycl::range<2> global((a.M + SGM - 1) / SGM, (a.N + wgn - 1) / wgn * kern::WG);
    stream->parallel_for(sycl::nd_range<2>(global, local), kern{ a.Aq, a.SA, a.B, a.SB, a.C, a.other, a.M, a.N, a.NP, a.K, a.ldc, a.postop });
}

template <int FMT, int MT_M, int MT_N, int WG_M, int WG_N> static void launch_gemm(const args & a, dpct::queue_ptr stream) {
    using kern       = gemm<FMT, MT_M, MT_N, WG_M, WG_N>;
    const int mtiles = (a.M + MT_M * WG_M - 1) / (MT_M * WG_M);
    const int ntiles = (a.N + MT_N * WG_N - 1) / (MT_N * WG_N);
    // small batches leave most of the GPU idle: split K over work-groups and add the slices up after
    const int ks = std::max(1, std::min(a.K / GS / 4, a.threads / (mtiles * ntiles * WG_M * WG_N)));
    ggml_sycl_pool_alloc<float> part(*a.pool);
    if (ks > 1) {
        part.alloc((size_t) ks * a.M * a.NP);
    }
    const sycl::range<2> local(1, kern::WG);
    const sycl::range<2> global((size_t) mtiles * ks, (size_t) ntiles * kern::WG);
    stream->parallel_for(sycl::nd_range<2>(global, local), kern{ a.Aq, a.SA, a.B, a.SB, a.C, a.other, a.M, a.N, a.NP, a.K, a.ldc, a.postop, a.st2d, part.ptr, ks });
    if (ks > 1) {
        const float * p   = part.ptr;
        float *       C   = a.C;
        const int     M   = a.M, N = a.N, NP = a.NP, ldc = a.ldc;
        const float * other  = a.other;
        const int     postop = a.postop;
        stream->parallel_for(sycl::range<1>((size_t) M * N), [=](sycl::item<1> it) {
            const int m = it[0] / N;
            const int n = it[0] % N;
            float     v = 0.0f;
            for (int k = 0; k < ks; ++k) {
                v += p[((size_t) k * M + m) * NP + n];
            }
            const size_t i = (size_t) m * ldc + n;
            C[i]           = epilogue(v, other, i, postop);
        });
    }
}

template <int FMT, int SGM> static void launch_gemv_ls(const args & a, int ls, dpct::queue_ptr stream) {
    // PTQ1_0 decodes between the load and the dpas: more sub-groups per work-group hide that latency
    constexpr int NSG = FMT == FMT_PTQ1 ? 4 : 2;
    switch (ls) {
        case 1:  launch_gemv<FMT, SGM, NSG, 1, 2>(a, stream); break;
        case 2:  launch_gemv<FMT, SGM, NSG, 2, 2>(a, stream); break;
        case 4:  launch_gemv<FMT, SGM, NSG, 4, 2>(a, stream); break;
        default: launch_gemv<FMT, SGM, NSG, 8, 2>(a, stream); break;
    }
}

template <int FMT> static void launch(const args & a, int ls, dpct::queue_ptr stream);

static sycl::event reorder_pq2_0(const uint8_t * src, uint8_t * dst, int ncols, int nrows,
                                  dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PQ2_0 == 0);
    const int           nw = ncols / 16;
    const int           nb = ncols / QK_PQ2_0;
    const int           np = GGML_PAD(nrows, 16);
    uint32_t *          qs = (uint32_t *) dst;
    sycl::half *        d  = (sycl::half *) (dst + (size_t) nw * np * sizeof(uint32_t));
    const block_pq2_0 * x  = (const block_pq2_0 *) src;

    return stream->parallel_for(sycl::range<2>(nw, np), [=](sycl::item<2> it) {
        const int w = it[0];
        const int n = it[1];
        if (n >= nrows) {
            qs[(size_t) w * np + n] = 0;
            if (w % 8 == 0) {
                d[(size_t) (w / 8) * np + n] = 0.0f;
            }
            return;
        }
        const block_pq2_0 * blk = x + (size_t) n * nb + w / 8;
        // qs is only 2-byte aligned inside the 34-byte block
        const uint16_t *    q   = (const uint16_t *) blk->qs + 2 * (w % 8);
        qs[(size_t) w * np + n] = pq2_codes_to_s2(q[0] | ((uint32_t) q[1] << 16));
        if (w % 8 == 0) {
            d[(size_t) (w / 8) * np + n] = blk->d;
        }
    });
}

static sycl::event reorder_ptq1_0(const uint8_t * src, uint8_t * dst, int ncols, int nrows,
                                   dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_PTQ1_0 == 0);
    const int            nb = ncols / QK_PTQ1_0;
    const int            np = GGML_PAD(nrows, 16);
    uint32_t *           out = (uint32_t *) dst;
    const block_ptq1_0 * x   = (const block_ptq1_0 *) src;

    return stream->parallel_for(sycl::range<2>((size_t) nb * PTQ1_ROWS, np), [=](sycl::item<2> it) {
        const int r = it[0];
        const int n = it[1];
        uint32_t  v = 0;
        if (n < nrows) {
            const block_ptq1_0 * blk = x + (size_t) n * nb + r / PTQ1_ROWS;
            const int            d   = r % PTQ1_ROWS;
            if (d < PTQ1_ROWS - 1) {
                v = *(const uint32_t *) (blk->qs + 4 * d);  // blocks are 28 bytes, so qs is 4-byte aligned
            } else {
                v = blk->qh[0] | (uint32_t) blk->qh[1] << 8 | (uint32_t) sycl::bit_cast<uint16_t>(blk->d) << 16;
            }
        }
        out[(size_t) r * np + n] = v;
    });
}

// M at or below this uses the GEMV kernel
static constexpr int GEMV_MAX_M = 8;

// src1 quantized to int8 with one scale per 128 values; shared by every weight that reads src1
struct act_q {
    ggml_sycl_pool_alloc<int8_t> aq;
    ggml_sycl_pool_alloc<float>  sa;
    int                          M, K;
    int                          fmt;  // weight format whose K order the activation is in

    act_q(ggml_backend_sycl_context & ctx, ggml_sycl_pool & pool, const ggml_tensor * src1, const float * had_signs,
          int fmt) :
        aq(pool),
        sa(pool),
        fmt(fmt) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32 && src1->nb[0] == sizeof(float));
        GGML_ASSERT(src1->ne[0] % GS == 0);
        M = src1->ne[1] * src1->ne[2] * src1->ne[3];
        K = src1->ne[0];
        aq.alloc((size_t) M * K);
        sa.alloc((size_t) (K / GS) * ldsa(M));

        if (had_signs) {
            GGML_ASSERT(K % quant_a_had::HN == 0);
            const quant_a_had q{ (const float *) src1->data,
                                 had_signs,
                                 sa.get(),
                                 aq.get(),
                                 nullptr,
                                 M,
                                 K,
                                 (int) src1->ne[1],
                                 (int) src1->ne[2],
                                 (int64_t) (src1->nb[1] / sizeof(float)),
                                 (int64_t) (src1->nb[2] / sizeof(float)),
                                 (int64_t) (src1->nb[3] / sizeof(float)),
                                 fmt == FMT_PTQ1 };
            ctx.stream()->parallel_for(
                sycl::nd_range<2>(sycl::range<2>(M, (K / quant_a_had::HN) * quant_a_had::NT),
                                  sycl::range<2>(1, quant_a_had::NT)),
                q);
            return;
        }

        const quant_a q{ (const float *) src1->data,
                         sa.get(),
                         aq.get(),
                         M,
                         K,
                         (int) src1->ne[1],
                         (int) src1->ne[2],
                         (int64_t) (src1->nb[1] / sizeof(float)),
                         (int64_t) (src1->nb[2] / sizeof(float)),
                         (int64_t) (src1->nb[3] / sizeof(float)),
                         fmt == FMT_PTQ1 };
        ctx.stream()->parallel_for(sycl::nd_range<2>(sycl::range<2>(M, (K / GS) * 16), sycl::range<2>(1, 16)), q);
    }
};

// out (and other) are M x N floats with row stride ldc
static int fmt_of(ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_PQ2_0 || type == GGML_TYPE_PTQ1_0);
    return type == GGML_TYPE_PTQ1_0 ? FMT_PTQ1 : FMT_PQ2;
}

template <int FMT> static void launch(const args & a, int ls, dpct::queue_ptr stream) {
    if (a.M <= GEMV_MAX_M) {
        if (a.M == 1) {
            launch_gemv_ls<FMT, 1>(a, ls, stream);
        } else if (a.M == 2) {
            launch_gemv_ls<FMT, 2>(a, ls, stream);
        } else if (a.M <= 4) {
            launch_gemv_ls<FMT, 4>(a, ls, stream);
        } else {
            launch_gemv_ls<FMT, 8>(a, ls, stream);
        }
    } else if (a.M <= 16) {
        // small batches: few row blocks per work-group, so the sub-groups are not idle
        launch_gemm<FMT, 8, 128, 2, 4>(a, stream);
    } else if (a.M <= 32) {
        launch_gemm<FMT, 8, 32, 4, 4>(a, stream);
    } else if (a.M <= 64) {
        if constexpr (FMT == FMT_PTQ1) {
            launch_gemm<FMT, 16, 32, 4, 2>(a, stream);
        } else {
            launch_gemm<FMT, 16, 64, 4, 2>(a, stream);
        }
    } else if (a.M <= 128) {
        launch_gemm<FMT, 32, 32, 4, 2>(a, stream);
    } else if constexpr (FMT == FMT_PTQ1) {
        // a tall work-group shares each decoded B column block among 8 sub-groups
        launch_gemm<FMT, 32, 32, 8, 1>(a, stream);
    } else {
        launch_gemm<FMT, 16, 64, 4, 2>(a, stream);
    }
}

static void run(ggml_backend_sycl_context & ctx, const ggml_tensor * w, const act_q & q, float * out, int ldc,
                int postop, const float * other) {
    GGML_ASSERT(w->ne[0] == q.K);
    const int fmt = fmt_of(w->type);
    GGML_ASSERT(fmt == q.fmt);

    const int K  = q.K;
    const int M  = q.M;
    const int N  = w->ne[1];
    const int NP = GGML_PAD(N, 16);

    // 2D block I/O needs 64-byte aligned surfaces and 16-byte aligned pitches
    GGML_ASSERT((uintptr_t) w->data % 64 == 0);
    const bool st2d = (uintptr_t) out % 64 == 0 && ldc % 4 == 0;

    args a{ q.aq.ptr,
                  q.sa.ptr,
                  (const uint32_t *) w->data,
                  (const unsigned short *) ((const char *) w->data + (size_t) (K / 16) * NP * sizeof(uint32_t)),
                  out,
                  other,
                  M,
                  N,
                  NP,
                  K,
                  ldc,
                  postop,
                  st2d,
                  &ctx.pool(),
                  0 };

    // split K so the sub-groups fill about one wave of hardware threads (8 per EU);
    // a partial second wave costs more than the split saves
    const int nblk    = NP / 16;
    const int threads = ggml_sycl_info().devices[ctx.device].nsm * 16 * 8;  // nsm is compute units / 16
    a.threads         = threads;
    int       ls      = 4;
    for (int l : { 8, 4, 2, 1 }) {
        if (nblk * l <= threads) {
            ls = l;
            break;
        }
    }

    if (fmt == FMT_PTQ1) {
        launch<FMT_PTQ1>(a, ls, ctx.stream());
    } else {
        launch<FMT_PQ2>(a, ls, ctx.stream());
    }
}

}  // namespace ggml_sycl_xmx

static_assert((int) GGML_SYCL_XMX_EPI_NONE == ggml_sycl_xmx::EPI_NONE &&
              (int) GGML_SYCL_XMX_EPI_SWIGLU == ggml_sycl_xmx::EPI_SWIGLU &&
              (int) GGML_SYCL_XMX_EPI_ADD == ggml_sycl_xmx::EPI_ADD, "epilogue numbering");

struct ggml_sycl_pq2_xmx_act {
    ggml_sycl_xmx::act_q q;

    ggml_sycl_pq2_xmx_act(ggml_backend_sycl_context & ctx, const ggml_tensor * x, const float * had_signs,
                      ggml_type wtype) :
        q(ctx, ctx.xmx_act_pool(), x, had_signs, ggml_sycl_xmx::fmt_of(wtype)) {}
};

ggml_sycl_pq2_xmx_act * ggml_sycl_pq2_xmx_act_quantize(ggml_backend_sycl_context & ctx, const ggml_tensor * x,
                                                       const float * had_signs, ggml_type wtype) {
    return new ggml_sycl_pq2_xmx_act(ctx, x, had_signs, wtype);
}

void ggml_sycl_pq2_xmx_act_free(ggml_sycl_pq2_xmx_act * act) {
    delete act;
}

void ggml_sycl_pq2_xmx_mul_mat_act(ggml_backend_sycl_context & ctx, const ggml_tensor * w,
                                   const ggml_sycl_pq2_xmx_act * act, float * dst, int ldc, int epi,
                                   const float * other) {
    ggml_sycl_xmx::run(ctx, w, act->q, dst, ldc, epi, other);
}

bool ggml_sycl_pq2_xmx_supports_ne0(int64_t ne0) {
    return ne0 % QK_PQ2_0 == 0;
}

size_t ggml_sycl_pq2_xmx_bytes(const ggml_tensor * t) {
    return (size_t) GGML_PAD(t->ne[1], 16) * t->nb[1];
}

bool ggml_sycl_pq2_xmx_reorder(ggml_tensor * src0, dpct::queue_ptr stream) {
    GGML_ASSERT((src0->type == GGML_TYPE_PQ2_0 || src0->type == GGML_TYPE_PTQ1_0) && ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);

    const size_t size = ggml_nbytes(src0);
    void *       tmp  = sycl::malloc_device(size, *stream);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for the XMX reorder, skipping it\n", __func__, size);
        return false;
    }
    stream->memcpy(tmp, src0->data, size).wait();
    const auto reorder = src0->type == GGML_TYPE_PQ2_0 ? ggml_sycl_xmx::reorder_pq2_0 : ggml_sycl_xmx::reorder_ptq1_0;
    reorder((const uint8_t *) tmp, (uint8_t *) src0->data, (int) src0->ne[0], (int) src0->ne[1], stream).wait();
    sycl::free(tmp, *stream);
    return true;
}

void ggml_sycl_pq2_xmx_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    const ggml_sycl_xmx::act_q q(ctx, ctx.pool(), src1, nullptr, ggml_sycl_xmx::fmt_of(src0->type));
    ggml_sycl_xmx::run(ctx, src0, q, (float *) dst->data, (int) dst->ne[0], ggml_sycl_xmx::EPI_NONE, nullptr);
}

void ggml_sycl_pq2_xmx_hadamard_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * x, const float * signs,
                                     float * dst) {
    using ggml_sycl_xmx::quant_a_had;
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) && x->ne[0] % quant_a_had::HN == 0);
    const int         M = x->ne[1] * x->ne[2] * x->ne[3];
    const int         K = x->ne[0];
    const quant_a_had q{ (const float *) x->data,
                         signs,
                         nullptr,
                         nullptr,
                         dst,
                         M,
                         K,
                         (int) x->ne[1],
                         (int) x->ne[2],
                         (int64_t) (x->nb[1] / sizeof(float)),
                         (int64_t) (x->nb[2] / sizeof(float)),
                         (int64_t) (x->nb[3] / sizeof(float)) };
    ctx.stream()->parallel_for(sycl::nd_range<2>(sycl::range<2>(M, (K / quant_a_had::HN) * quant_a_had::NT),
                                                 sycl::range<2>(1, quant_a_had::NT)),
                               q);
}

#else

bool ggml_sycl_pq2_xmx_supports_ne0(int64_t) {
    return false;
}

size_t ggml_sycl_pq2_xmx_bytes(const ggml_tensor * t) {
    return ggml_nbytes(t);
}

bool ggml_sycl_pq2_xmx_reorder(ggml_tensor *, dpct::queue_ptr) {
    return false;
}

void ggml_sycl_pq2_xmx_mul_mat(ggml_backend_sycl_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    GGML_ABORT("PQ2_0 XMX path is not built in");
}

ggml_sycl_pq2_xmx_act * ggml_sycl_pq2_xmx_act_quantize(ggml_backend_sycl_context &, const ggml_tensor *, const float *,
                                                       ggml_type) {
    GGML_ABORT("PQ2_0 XMX path is not built in");
}

void ggml_sycl_pq2_xmx_act_free(ggml_sycl_pq2_xmx_act *) {}

void ggml_sycl_pq2_xmx_mul_mat_act(ggml_backend_sycl_context &, const ggml_tensor *, const ggml_sycl_pq2_xmx_act *,
                                   float *, int, int, const float *) {
    GGML_ABORT("PQ2_0 XMX path is not built in");
}

void ggml_sycl_pq2_xmx_hadamard_fwht(ggml_backend_sycl_context &, const ggml_tensor *, const float *, float *) {
    GGML_ABORT("PQ2_0 XMX path is not built in");
}
#endif // __INTEL_LLVM_COMPILER && !GGML_SYCL_NO_PQ2_XMX

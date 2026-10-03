// Xe2 sub-group primitives for SIMT SYCL device code, without OpenCL builtins
// or IGC-internal intrinsics:
//   * DPAS, 2D block I/O, cached LSC loads: inline vISA (the instructions IGC
//     emits for the OpenCL builtins; operands are the SIMD16 per-lane vectors,
//     row r of a block / DPAS operand = vector component r)
//   * sub-group block reads: sycl_ext_oneapi_group_load_store (group_load)
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/group_load_store.hpp>

#include <cstdint>

namespace xe2 {

#define XE2_VEC(T, n, name) typedef T name __attribute__((ext_vector_type(n)))
XE2_VEC(short, 2, short2);
XE2_VEC(short, 4, short4);
XE2_VEC(short, 8, short8);
XE2_VEC(unsigned short, 2, ushort2);
XE2_VEC(unsigned short, 4, ushort4);
XE2_VEC(unsigned short, 8, ushort8);
XE2_VEC(unsigned short, 16, ushort16);
XE2_VEC(unsigned short, 32, ushort32);
XE2_VEC(unsigned short, 64, ushort64);
XE2_VEC(int, 2, int2);
XE2_VEC(int, 4, int4);
XE2_VEC(int, 8, int8);
XE2_VEC(unsigned, 2, uint2);
XE2_VEC(unsigned, 3, uint3);
XE2_VEC(unsigned, 4, uint4);
XE2_VEC(unsigned, 8, uint8);
XE2_VEC(float, 2, float2);
XE2_VEC(float, 4, float4);
XE2_VEC(float, 8, float8);
#undef XE2_VEC

// 2D surface: width and pitch in bytes, height in rows. Addresses are long long:
// long is 32 bits in SYCL device code on Windows (LLP64) and would truncate them
struct surf {
    long long base;
    int w, h, p;
    surf(const void *b, int width_bytes, int height, int pitch_bytes)
        : base((long long)b), w(width_bytes - 1), h(height - 1), p(pitch_bytes - 1) {}
};

#ifdef __SYCL_DEVICE_ONLY__
#define XE2_ASM(...) __asm__(__VA_ARGS__)
#define XE2_ASM_V(...) __asm__ volatile(__VA_ARGS__)
#else
#define XE2_ASM(...)
#define XE2_ASM_V(...)
#endif

// 2D block I/O, lsc_load_block2d / lsc_store_block2d on an explicit surface
// flat[base, width-1, height-1, pitch-1, x (elements), y (rows)]; shape
// d<bits>.[V x]C x R: V blocks of C columns x R rows, block-major.
#define XE2_2D_RD(name, T, shape)                                                  \
    inline T name(const surf &s, int x, int y) {                                   \
        T v;                                                                       \
        XE2_ASM("{\n"                                                              \
                ".decl SB v_type=G type=q num_elts=1 align=qword alias=<%1,0>\n"   \
                ".decl SW v_type=G type=d num_elts=1 align=dword alias=<%2,0>\n"   \
                ".decl SH v_type=G type=d num_elts=1 align=dword alias=<%3,0>\n"   \
                ".decl SP v_type=G type=d num_elts=1 align=dword alias=<%4,0>\n"   \
                ".decl SX v_type=G type=d num_elts=1 align=dword alias=<%5,0>\n"   \
                ".decl SY v_type=G type=d num_elts=1 align=dword alias=<%6,0>\n"   \
                "lsc_load_block2d.ugm (M1, 1) %0:" shape " flat[SB,SW,SH,SP,SX,SY]\n" \
                "}\n"                                                              \
                : "=rw"(v)                                                         \
                : "rw.u"(s.base), "rw.u"(s.w), "rw.u"(s.h), "rw.u"(s.p), "rw.u"(x), "rw.u"(y)); \
        return v;                                                                  \
    }
#define XE2_2D_WR(name, T, shape)                                                  \
    inline void name(const surf &s, int x, int y, T v) {                           \
        XE2_ASM_V("{\n"                                                            \
                ".decl SB v_type=G type=q num_elts=1 align=qword alias=<%0,0>\n"   \
                ".decl SW v_type=G type=d num_elts=1 align=dword alias=<%1,0>\n"   \
                ".decl SH v_type=G type=d num_elts=1 align=dword alias=<%2,0>\n"   \
                ".decl SP v_type=G type=d num_elts=1 align=dword alias=<%3,0>\n"   \
                ".decl SX v_type=G type=d num_elts=1 align=dword alias=<%4,0>\n"   \
                ".decl SY v_type=G type=d num_elts=1 align=dword alias=<%5,0>\n"   \
                "lsc_store_block2d.ugm (M1, 1) flat[SB,SW,SH,SP,SX,SY] %6:" shape "\n" \
                "}\n"                                                              \
                :: "rw.u"(s.base), "rw.u"(s.w), "rw.u"(s.h), "rw.u"(s.p), "rw.u"(x), "rw.u"(y), "rw"(v)); \
    }
XE2_2D_RD(rd_32b_1r16, unsigned, "d32.16x1nn")
XE2_2D_RD(rd_32b_2r16, uint2, "d32.16x2nn")
XE2_2D_RD(rd_32b_8r16, uint8, "d32.16x8nn")
// a 16x1 d16 block is half a GRF, but the load writes a whole (zero-padded) GRF
inline unsigned short rd_16b_1r16(const surf &s, int x, int y) {
    unsigned short v;
    XE2_ASM("{\n"
            ".decl SB v_type=G type=q num_elts=1 align=qword alias=<%1,0>\n"
            ".decl SW v_type=G type=d num_elts=1 align=dword alias=<%2,0>\n"
            ".decl SH v_type=G type=d num_elts=1 align=dword alias=<%3,0>\n"
            ".decl SP v_type=G type=d num_elts=1 align=dword alias=<%4,0>\n"
            ".decl SX v_type=G type=d num_elts=1 align=dword alias=<%5,0>\n"
            ".decl SY v_type=G type=d num_elts=1 align=dword alias=<%6,0>\n"
            ".decl TP v_type=G type=uw num_elts=32 align=GRF\n"
            "lsc_load_block2d.ugm (M1, 1) TP:d16.16x1nn flat[SB,SW,SH,SP,SX,SY]\n"
            "mov (M1, 16) %0(0,0)<1> TP(0,0)<1;1,0>\n"
            "}\n"
            : "=rw"(v)
            : "rw.u"(s.base), "rw.u"(s.w), "rw.u"(s.h), "rw.u"(s.p), "rw.u"(x), "rw.u"(y));
    return v;
}
// two adjacent 1x16 blocks = one 1x32 block: [0] = column x + lane, [1] = x + 16 + lane
XE2_2D_RD(rd_16b_1r16x2, ushort2, "d16.32x1nn")
XE2_2D_RD(rd_16b_8r16, ushort8, "d16.16x8nn")
// two adjacent 8x16 blocks, block-major: [0..7] = columns x..x+15, [8..15] = x+16..x+31
XE2_2D_RD(rd_16b_8r16x2, ushort16, "d16.2x16x8nn")
XE2_2D_WR(wr_16b_8r16, ushort8, "d16.16x8nn")
XE2_2D_WR(wr_32b_8r16, uint8, "d32.16x8nn")
#undef XE2_2D_RD
#undef XE2_2D_WR

// 2D block reads from a prebuilt address payload plus immediate (DX, DY) offsets,
// as IGC emits for the builtins: one payload per surface and K step instead of
// ~6 movs per load. The vISA text is built at compile time (asm((expr))).
namespace detail {
template <size_t N> struct cstr {
    char s[N]{};
    size_t n = 0;
    constexpr size_t size() const { return n; }
    constexpr const char *data() const { return s; }
    constexpr void add(const char *p) { while (*p) s[n++] = *p++; }
    constexpr void addi(int v) {
        if (v < 0) { s[n++] = '-'; v = -v; }
        char t[12];
        int k = 0;
        do { t[k++] = char('0' + v % 10); v /= 10; } while (v);
        while (k) s[n++] = t[--k];
    }
};
template <class SH, int DX, int DY> constexpr auto rd2d_str() {
    cstr<320> c;
    c.add("{\n.decl PD v_type=G type=ud num_elts=8 align=GRF alias=<%1,0>\n");
    if constexpr (SH::pad) c.add(".decl TP v_type=G type=uw num_elts=32 align=GRF\n"
                                 "lsc_load_block2d.ugm (M1, 1) TP:");
    else c.add("lsc_load_block2d.ugm (M1, 1) %0:");
    c.add(SH::v);
    c.add(" flat[PD + (");
    c.addi(DX);
    c.add(",");
    c.addi(DY);
    c.add(")]\n");
    if constexpr (SH::pad) c.add("mov (M1, 16) %0(0,0)<1> TP(0,0)<1;1,0>\n");
    c.add("}\n");
    return c;
}
}  // namespace detail

// block shape: vISA type string and payload dword 7 = (V-1) << 16 | (R-1) << 8 | (C-1);
// pad: the block is half a GRF but the load writes a whole (zero-padded) GRF
struct b32_16x1 { static constexpr const char *v = "d32.16x1nn"; static constexpr int code = 0x00f; static constexpr bool pad = false; };
struct b32_16x8 { static constexpr const char *v = "d32.16x8nn"; static constexpr int code = 0x70f; static constexpr bool pad = false; };
struct b16_16x8 { static constexpr const char *v = "d16.16x8nn"; static constexpr int code = 0x70f; static constexpr bool pad = false; };
struct b16_2x16x8 { static constexpr const char *v = "d16.2x16x8nn"; static constexpr int code = 0x1070f; static constexpr bool pad = false; };
struct b16_16x16 { static constexpr const char *v = "d16.16x16nn"; static constexpr int code = 0xf0f; static constexpr bool pad = false; };
struct b16_2x16x16 { static constexpr const char *v = "d16.2x16x16nn"; static constexpr int code = 0x10f0f; static constexpr bool pad = false; };
struct b16_16x32 { static constexpr const char *v = "d16.16x32nn"; static constexpr int code = 0x1f0f; static constexpr bool pad = false; };
struct b16_2x16x32 { static constexpr const char *v = "d16.2x16x32nn"; static constexpr int code = 0x11f0f; static constexpr bool pad = false; };
struct b16_32x1 { static constexpr const char *v = "d16.32x1nn"; static constexpr int code = 0x01f; static constexpr bool pad = false; };
struct b16_16x1 { static constexpr const char *v = "d16.16x1nn"; static constexpr int code = 0x00f; static constexpr bool pad = true; };

template <class SH> inline unsigned pl2d(const surf &s, int x, int y) {
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
// in-place update of the block x (dword 5) or y (dword 6) of a payload: build the
// payload once outside the K loop and move it per step (keeps one register per
// payload live across the loop, as IGC does, instead of rebuilding all 8 dwords)
template <int D> inline void pl2d_set(unsigned &pl, int v) {
    static_assert(D == 5 || D == 6);
    if constexpr (D == 5)
        XE2_ASM("{\n.decl PD v_type=G type=ud num_elts=8 align=GRF alias=<%0,0>\n"
                "mov (M1_NM, 1) PD(0,5)<1> %1(0,0)<0;1,0>\n}\n" : "+rw"(pl) : "rw.u"(v));
    else
        XE2_ASM("{\n.decl PD v_type=G type=ud num_elts=8 align=GRF alias=<%0,0>\n"
                "mov (M1_NM, 1) PD(0,6)<1> %1(0,0)<0;1,0>\n}\n" : "+rw"(pl) : "rw.u"(v));
}
inline void pl2d_x(unsigned &pl, int x) { pl2d_set<5>(pl, x); }
inline void pl2d_y(unsigned &pl, int y) { pl2d_set<6>(pl, y); }
template <class SH, int DX, int DY, class T> inline T rd2d(unsigned pl) {
    T v;
    XE2_ASM((detail::rd2d_str<SH, DX, DY>()) : "=rw"(v) : "rw"(pl));
    return v;
}

// compile-time loop: f(std::integral_constant<int, I>) for I = 0..N-1
template <int N, class F> inline void static_for(F &&f) {
    [&]<int... I>(std::integer_sequence<int, I...>) { (f(std::integral_constant<int, I>{}), ...); }(
            std::make_integer_sequence<int, N>{});
}

// Sub-group block reads, striped: element i of lane l = p[l + 16 i]; p must be
// global and 4-byte aligned (16-byte for the vector forms). One transposed LSC
// block load, whose dword order is the SIMD16 vector layout (group_load lowers
// these to per-lane gathers). The scalar form fills half a GRF but the load
// writes a whole one, so it goes through a temp.
inline unsigned short sg_rd_us(const unsigned short *p) {
    unsigned short v;
    XE2_ASM("{\n.decl TP v_type=G type=uw num_elts=32 align=GRF\n"
            ".decl AD v_type=G type=q num_elts=1 align=GRF\n"
            "mov (M1_NM, 1) AD(0,0)<1> %1(0,0)<0;1,0>\n"
            "lsc_load.ugm (M1_NM, 1) TP:d32x8t flat[AD]:a64\n"
            "mov (M1, 16) %0(0,0)<1> TP(0,0)<1;1,0>\n}\n"
            : "=rw"(v) : "rw.u"((long long)p));
    return v;
}
// the send address must start a GRF: a uniform scalar is copied into an aligned temp
#define XE2_BLK_RD(name, R, P, shape)                                                  \
    inline R name(const P *p) {                                                         \
        R v;                                                                            \
        XE2_ASM("{\n.decl AD v_type=G type=q num_elts=1 align=GRF\n"                  \
                "mov (M1_NM, 1) AD(0,0)<1> %1(0,0)<0;1,0>\n"                           \
                "lsc_load.ugm (M1_NM, 1) %0:" shape " flat[AD]:a64\n}\n"              \
                : "=rw"(v) : "rw.u"((long long)p));                                    \
        return v;                                                                       \
    }
XE2_BLK_RD(sg_rd_us4, ushort4, unsigned short, "d32x32t")
XE2_BLK_RD(sg_rd_us8, ushort8, unsigned short, "d32x64t")
XE2_BLK_RD(sg_rd_u4, uint4, unsigned, "d32x64t")
#undef XE2_BLK_RD

// per-lane 3-dword load, L1 and L3 cached
inline uint3 ld_u3_cached(const uint3 *p) {
    uint3 v;
    XE2_ASM("lsc_load.ugm.ca.ca (M1, 16) %0:d32x3 flat[%1]:a64" : "=rw"(v) : "rw"((long long)p));
    return v;
}

// DPAS, SIMD16, systolic depth 8. Repeat count R = rows of A: A is one
// short per row per lane (K16 fp16/bf16, or K32 s8), B the VNNI weights;
// na / nb = A / B size in dwords.
#define XE2_DPAS(name, prec, A, B, C, R, na, nb)                                       \
    inline C name(A a, B b, C acc) {                                                    \
        XE2_ASM("{\n"                                                                   \
                ".decl DB v_type=G type=ud num_elts=" #nb " align=GRF alias=<%1,0>\n"   \
                ".decl DA v_type=G type=ud num_elts=" #na " align=GRF alias=<%2,0>\n"   \
                "dpas." prec ".8." #R " (M1, 16) %0.0 %0.0 DB.0 DA(0,0)\n"              \
                "}\n"                                                                   \
                : "+rw"(acc) : "rw"(b), "rw"(a));                                       \
        return acc;                                                                     \
    }
#define XE2_DPAS_Z(name, prec, A, B, C, R, na, nb)                                      \
    inline C name(A a, B b) {                                                           \
        C d;                                                                            \
        XE2_ASM("{\n"                                                                   \
                ".decl DB v_type=G type=ud num_elts=" #nb " align=GRF alias=<%1,0>\n"   \
                ".decl DA v_type=G type=ud num_elts=" #na " align=GRF alias=<%2,0>\n"   \
                "dpas." prec ".8." #R " (M1, 16) %0.0 %%null.0 DB.0 DA(0,0)\n"          \
                "}\n"                                                                   \
                : "=rw"(d) : "rw"(b), "rw"(a));                                         \
        return d;                                                                       \
    }
XE2_DPAS(dpas_hf, "hf.hf", short, int8, float, 1, 8, 128)
XE2_DPAS(dpas_hf, "hf.hf", short2, int8, float2, 2, 16, 128)
XE2_DPAS(dpas_hf, "hf.hf", short4, int8, float4, 4, 32, 128)
XE2_DPAS(dpas_hf, "hf.hf", short8, int8, float8, 8, 64, 128)
XE2_DPAS(dpas_bf, "bf.bf", short, int8, float, 1, 8, 128)
XE2_DPAS(dpas_bf, "bf.bf", short2, int8, float2, 2, 16, 128)
XE2_DPAS(dpas_bf, "bf.bf", short4, int8, float4, 4, 32, 128)
XE2_DPAS(dpas_bf, "bf.bf", short8, int8, float8, 8, 64, 128)
// s8 A (K32) x s2 B: 4 weights per byte of B
XE2_DPAS(dpas_s2s8, "s2.s8", short, int2, int, 1, 8, 32)
XE2_DPAS(dpas_s2s8, "s2.s8", short2, int2, int2, 2, 16, 32)
XE2_DPAS(dpas_s2s8, "s2.s8", short4, int2, int4, 4, 32, 32)
XE2_DPAS(dpas_s2s8, "s2.s8", short8, int2, int8, 8, 64, 32)
// same with a zero accumulator (src0 = null)
XE2_DPAS_Z(dpas_s2s8_z, "s2.s8", short, int2, int, 1, 8, 32)
XE2_DPAS_Z(dpas_s2s8_z, "s2.s8", short2, int2, int2, 2, 16, 32)
XE2_DPAS_Z(dpas_s2s8_z, "s2.s8", short4, int2, int4, 4, 32, 32)
XE2_DPAS_Z(dpas_s2s8_z, "s2.s8", short8, int2, int8, 8, 64, 32)
#undef XE2_DPAS
#undef XE2_DPAS_Z

// work-group barrier with a local-memory fence only (OpenCL barrier(CLK_LOCAL_MEM_FENCE))
inline void barrier_local() {
#ifdef __SYCL_DEVICE_ONLY__
    __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
            __spv::MemorySemanticsMask::AcquireRelease | __spv::MemorySemanticsMask::WorkgroupMemory);
#endif
}

// 16-bit activation dtype: raw bits in, fp32 out, round-to-nearest-even back
template <bool BF16> struct dt;
template <> struct dt<false> {
    static float tof(unsigned short h) { return (float)sycl::bit_cast<sycl::half>(h); }
    static unsigned short from(float f) { return sycl::bit_cast<unsigned short>((sycl::half)f); }
};
template <> struct dt<true> {
    static float tof(unsigned short h) { return sycl::bit_cast<float>((unsigned)h << 16); }
    static unsigned short from(float f) {
        return sycl::bit_cast<unsigned short>(sycl::ext::oneapi::bfloat16(f));
    }
};

template <bool BF16, typename A, typename C> inline C mad_k16(A a, int8 b, C acc) {
    if constexpr (BF16) return dpas_bf(a, b, acc);
    else return dpas_hf(a, b, acc);
}

namespace detail {
template <bool BF16, int NA, int ROW> constexpr auto dpas_at_str() {
    cstr<300> c;
    c.add("{\n.decl DB v_type=G type=ud num_elts=128 align=GRF alias=<%1,0>\n"
          ".decl DA v_type=G type=ud num_elts=");
    c.addi(NA);
    c.add(" align=GRF alias=<%2,0>\ndpas.");
    c.add(BF16 ? "bf.bf" : "hf.hf");
    c.add(".8.8 (M1, 16) %0.0 %0.0 DB.0 DA(");
    c.addi(ROW);
    c.add(",0)\n}\n");
    return c;
}
}  // namespace detail

// 8-row DPAS whose A operand starts at GRF row ROW of a larger A block (e.g. a
// 32-row x 2-block 2D read), aliased in place as IGC does for the builtins
template <bool BF16, int ROW, class AV> inline float8 mad8_at(const AV &a, int8 b, float8 acc) {
    XE2_ASM((detail::dpas_at_str<BF16, (int)sizeof(AV) * 4, ROW>()) : "+rw"(acc) : "rw"(b), "rw"(a));
    return acc;
}

}  // namespace xe2

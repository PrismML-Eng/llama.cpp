// Xe2 sub-group intrinsics for SYCL device code: the same IGC builtins the
// OpenCL kernels of TernOCL use (DPAS, 2D block I/O, sub-group block reads),
// declared with the OpenCL mangling / IGC builtin names so that IGC resolves
// them from the SPIR-V exactly as for OpenCL C.
#pragma once

#include <sycl/sycl.hpp>

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

#define XE2_GLOBAL __attribute__((opencl_global))

}  // namespace xe2

#ifdef __SYCL_DEVICE_ONLY__
#define XE2_OCL(decl) SYCL_EXTERNAL decl
#define XE2_IB(decl) extern "C" SYCL_EXTERNAL decl
#else
#define XE2_OCL(decl) inline decl { __builtin_unreachable(); }
#define XE2_IB(decl) inline decl { __builtin_unreachable(); }
#endif

// DPAS, SIMD16: repeat count = rows of A (1, 2, 4, 8)
#define XE2_MAD(name, A, B, C) XE2_OCL(C name(A a, B b, C acc));
#define XE2_MAD_ALL(name)                                        \
    XE2_MAD(name, short, xe2::int8, float)                       \
    XE2_MAD(name, xe2::short2, xe2::int8, xe2::float2)           \
    XE2_MAD(name, xe2::short4, xe2::int8, xe2::float4)           \
    XE2_MAD(name, xe2::short8, xe2::int8, xe2::float8)
XE2_MAD_ALL(intel_sub_group_f16_f16_matrix_mad_k16)
XE2_MAD_ALL(intel_sub_group_bf16_bf16_matrix_mad_k16)
#undef XE2_MAD_ALL
// s8 x s2, K = 32
XE2_MAD(intel_sub_group_i8_i2_matrix_mad_k32, short, xe2::int2, int)
XE2_MAD(intel_sub_group_i8_i2_matrix_mad_k32, xe2::short2, xe2::int2, xe2::int2)
XE2_MAD(intel_sub_group_i8_i2_matrix_mad_k32, xe2::short4, xe2::int2, xe2::int4)
XE2_MAD(intel_sub_group_i8_i2_matrix_mad_k32, xe2::short8, xe2::int2, xe2::int8)
#undef XE2_MAD

// sub-group block reads (16 lanes, consecutive elements per lane)
XE2_OCL(unsigned intel_sub_group_block_read(const XE2_GLOBAL unsigned *p));
XE2_OCL(xe2::uint4 intel_sub_group_block_read4(const XE2_GLOBAL unsigned *p));
XE2_OCL(unsigned short intel_sub_group_block_read_us(const XE2_GLOBAL unsigned short *p));
XE2_OCL(xe2::ushort4 intel_sub_group_block_read_us4(const XE2_GLOBAL unsigned short *p));
XE2_OCL(xe2::ushort8 intel_sub_group_block_read_us8(const XE2_GLOBAL unsigned short *p));

// 2D block I/O: (base, width - 1, height - 1, pitch - 1) in bytes, coord (x
// in elements, y in rows). mRkCvV = R rows x C columns x V blocks.
#define XE2_2D_RD(ret, sfx) \
    XE2_IB(ret __builtin_IB_subgroup_block_read_flat_##sfx(long base, int w, int h, int p, xe2::int2 c));
XE2_2D_RD(unsigned, u32_m1k16v1)
XE2_2D_RD(xe2::uint2, u32_m2k16v1)
XE2_2D_RD(xe2::uint8, u32_m8k16v1)
XE2_2D_RD(unsigned short, u16_m1k16v1)
XE2_2D_RD(xe2::ushort2, u16_m1k16v2)
XE2_2D_RD(xe2::ushort8, u16_m8k16v1)
XE2_2D_RD(xe2::ushort16, u16_m8k16v2)
#undef XE2_2D_RD
XE2_IB(void __builtin_IB_subgroup_block_write_flat_u16_m8k16v1(long base, int w, int h, int p,
        xe2::int2 c, xe2::ushort8 v));
XE2_IB(void __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(long base, int w, int h, int p,
        xe2::int2 c, xe2::uint8 v));
// cache control 0 = default
XE2_IB(void __builtin_IB_subgroup_block_read_prefetch_u32_m8k16v1(long base, int w, int h, int p,
        xe2::int2 c, int cc));
// per-lane LSC load with cache control (IGC LSC_LDCC: 4 = L1 cached, L3 cached)
XE2_IB(xe2::uint3 __builtin_IB_lsc_load_global_uint3(const XE2_GLOBAL xe2::uint3 *p, int imm_off, int cc));

namespace xe2 {

// 2D surface: width and pitch in bytes, height in rows
struct surf {
    long base;
    int w, h, p;
    surf(const void *b, int width_bytes, int height, int pitch_bytes)
        : base((long)b), w(width_bytes - 1), h(height - 1), p(pitch_bytes - 1) {}
};

inline unsigned rd_32b_1r16(const surf &s, int x, int y) {
    return __builtin_IB_subgroup_block_read_flat_u32_m1k16v1(s.base, s.w, s.h, s.p, int2{x, y});
}
inline uint2 rd_32b_2r16(const surf &s, int x, int y) {
    return __builtin_IB_subgroup_block_read_flat_u32_m2k16v1(s.base, s.w, s.h, s.p, int2{x, y});
}
inline uint8 rd_32b_8r16(const surf &s, int x, int y) {
    return __builtin_IB_subgroup_block_read_flat_u32_m8k16v1(s.base, s.w, s.h, s.p, int2{x, y});
}
inline unsigned short rd_16b_1r16(const surf &s, int x, int y) {
    return __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(s.base, s.w, s.h, s.p, int2{x, y});
}
inline ushort2 rd_16b_1r16x2(const surf &s, int x, int y) {
    return __builtin_IB_subgroup_block_read_flat_u16_m1k16v2(s.base, s.w, s.h, s.p, int2{x, y});
}
inline ushort8 rd_16b_8r16(const surf &s, int x, int y) {
    return __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(s.base, s.w, s.h, s.p, int2{x, y});
}
// two adjacent 8x16 blocks, block-major: [0..7] = columns x..x+15, [8..15] = x+16..x+31
inline ushort16 rd_16b_8r16x2(const surf &s, int x, int y) {
    return __builtin_IB_subgroup_block_read_flat_u16_m8k16v2(s.base, s.w, s.h, s.p, int2{x, y});
}
inline void wr_16b_8r16(const surf &s, int x, int y, ushort8 v) {
    __builtin_IB_subgroup_block_write_flat_u16_m8k16v1(s.base, s.w, s.h, s.p, int2{x, y}, v);
}
inline void wr_32b_8r16(const surf &s, int x, int y, uint8 v) {
    __builtin_IB_subgroup_block_write_flat_u32_m8k16v1(s.base, s.w, s.h, s.p, int2{x, y}, v);
}
inline void pf_32b_8r16(const surf &s, int x, int y) {
    __builtin_IB_subgroup_block_read_prefetch_u32_m8k16v1(s.base, s.w, s.h, s.p, int2{x, y}, 0);
}

template <typename T> inline const XE2_GLOBAL T *gptr(const T *p) {
    return (const XE2_GLOBAL T *)p;
}

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
    if constexpr (BF16) return intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, acc);
    else return intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc);
}

}  // namespace xe2

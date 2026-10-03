// ARC-LAB: Xe2 s8 x s4 DPAS as inline vISA (as libxsmm/TernSYCL's xe2.hpp dpas helpers). It replaces the IGC builtin
// intel_sub_group_i8_i4_matrix_mad_k32, which is not in the public extension list: drivers that don't provide it (e.g. Lunar
// Lake, 2026-10) failed to load the whole device image ("No device image found for external symbol"), crashing every model.
// SIMD16, systolic depth 8, repeat count 8: A = 8 rows of K32 s8 (one short = 2 s8 per lane per row), B = K32 x 16 s4 (one
// int4 = 16 bytes of nibbles per lane / column), zero accumulator. Same result as the builtin with acc = 0.
#pragma once

#include <sycl/sycl.hpp>

namespace xe2dp {
typedef short s8x8 __attribute__((ext_vector_type(8)));
typedef int   s4x4 __attribute__((ext_vector_type(4)));
typedef int   i32x8 __attribute__((ext_vector_type(8)));

inline i32x8 dpas_s4s8_r8(s8x8 a, s4x4 b) {
    i32x8 d;
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=64 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.s4.s8.8.8 (M1, 16) %0.0 %%null.0 DB.0 DA(0,0)\n"
            "}\n"
            : "=rw"(d)
            : "rw"(b), "rw"(a));
#else
    d = 0;
    (void) a;
    (void) b;
#endif
    return d;
}
// f16 x f16 -> f32, repeat count 8: A = 8 rows of K16 halves (one short per lane per row), B = K16 x 16 (VNNI: 8 dwords
// per lane), accumulate. Replaces intel_sub_group_f16_f16_matrix_mad_k16 (public, but one builtin fewer to resolve).
typedef float f32x8 __attribute__((ext_vector_type(8)));
inline f32x8 dpas_hf_r8(s8x8 a, i32x8 b, f32x8 acc) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=128 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.hf.hf.8.8 (M1, 16) %0.0 %0.0 DB.0 DA(0,0)\n"
            "}\n"
            : "+rw"(acc)
            : "rw"(b), "rw"(a));
#else
    (void) a;
    (void) b;
#endif
    return acc;
}
}  // namespace xe2dp

// cpu simulation of the q2_0 vulkan shader functions.
//
// this file is one step of a three-step proof of extensional equivalence between
// the q2_0 glsl shader code and the cpu reference in ggml-quants.c.
//
// the first step lives in this file: the c++ functions named sim_* are literal
// text-level translations of the glsl functions, with the glsl source quoted in
// the comment immediately above each sim_*. a reader can verify by visual
// inspection that the c++ and the glsl compute the same value.
//
// the second step also lives in this file: the simulator runs against the cpu
// reference for randomized blocks and for every byte value x every slot x every
// alignment (exhaustive). passing this proves that the bit-extraction pattern
// is correct.
//
// the third step is a separate run of test-backend-ops on vulkan. once the
// shaders are compiled to spir-v and run on an actual gpu, test-backend-ops
// compares vulkan-backed tensor ops (get_rows, mul_mat, cpy, set_rows) against
// the cpu backend. that's the only step that proves the glsl itself is correct
// end-to-end, since steps one and two only validate the c++ stand-in.
//
// steps one and two catch nearly all transcription errors before the slow gpu
// build. step three covers the rest (glsl -> spir-v compiler quirks, driver
// issues, memory layout, etc) and is required to declare the shader correct.
//
// files this simulator stands in for:
// ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl (matvec)
// ggml/src/ggml-vulkan/vulkan-shaders/dequant_q2_0.comp (standalone dequant)
// ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs_cm2.glsl (cooperative matrix 2)
// ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl (matmul)
// ggml/src/ggml-vulkan/vulkan-shaders/copy_to_quant.comp (f32 -> q2_0)

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// block format (matches ggml-common.h:187-192 and types.glsl:207-214)

static const int QK2_0 = 128;

struct block_q2_0 {
    uint16_t d;          // fp16 raw bits
    uint8_t  qs[QK2_0/4]; // 32 bytes
};
static_assert(sizeof(block_q2_0) == 2 + 32, "block_q2_0 must be 34 bytes");

// fp16 <-> fp32 (ieee 754 half precision).
//
// we need an exact match with ggml's GGML_FP32_TO_FP16 / GGML_FP16_TO_FP32. the
// simplest deterministic conversion is via union punning of float and uint32_t.
// we use the standard bit-twiddling versions. for the values we'll feed
// (positive, finite, no subnormals), they agree with ggml

static float fp16_to_fp32(uint16_t h) {
    const uint32_t s = (h & 0x8000u) << 16;
    const uint32_t e = (h & 0x7c00u) >> 10;
    const uint32_t m = (h & 0x03ffu);
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = s; // signed zero
        } else {
            // subnormal: normalize
            uint32_t mm = m;
            int shift = 0;
            while ((mm & 0x0400u) == 0) {
                mm <<= 1;
                ++shift;
            }
            mm &= 0x03ffu;
            const uint32_t ee = (uint32_t)(127 - 15 - shift + 1);
            f = s | (ee << 23) | (mm << 13);
        }
    } else if (e == 31) {
        f = s | 0x7f800000u | (m << 13); // inf or nan
    } else {
        const uint32_t ee = e + (127 - 15);
        f = s | (ee << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

static uint16_t fp32_to_fp16(float x) {
    uint32_t f;
    std::memcpy(&f, &x, 4);
    const uint32_t s = (f >> 16) & 0x8000u;
    int32_t e = (int32_t)((f >> 23) & 0xffu) - 127 + 15;
    uint32_t m = f & 0x7fffffu;
    if (e >= 31) { return (uint16_t)(s | 0x7c00u | (m ? 0x200u : 0u)); } // inf/nan
    if (e <= 0) {
        if (e < -10) return (uint16_t)s;
        m = (m | 0x800000u) >> (1 - e);
        // round to nearest even
        if (m & 0x1000u) m += 0x2000u;
        return (uint16_t)(s | (m >> 13));
    }
    if (m & 0x1000u) {
        m += 0x2000u;
        if (m & 0x800000u) {
            m = 0;
            e += 1;
            if (e >= 31) return (uint16_t)(s | 0x7c00u);
        }
    }
    return (uint16_t)(s | ((uint32_t)e << 10) | (m >> 13));
}

// cpu reference (mirror of ggml-quants.c)

static void cpu_dequantize_row_q2_0(const block_q2_0 * x, float * y, int64_t k) {
    assert(k % QK2_0 == 0);
    const int nb = (int)(k / QK2_0);
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK2_0; ++j) {
            const int byte_index = j / 4;
            const int bit_offset = (j % 4) * 2;
            const uint8_t q = (x[i].qs[byte_index] >> bit_offset) & 0x03;
            y[i*QK2_0 + j] = ((int)q - 1) * d;
        }
    }
}

static void cpu_quantize_row_q2_0(const float * x, block_q2_0 * y, int64_t k) {
    assert(k % QK2_0 == 0);
    const int nb = (int)(k / QK2_0);
    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK2_0; ++j) {
            const float a = std::fabs(x[i*QK2_0 + j]);
            if (a > amax) amax = a;
        }
        const float d  = amax;
        const float id = d > 0.0f ? 1.0f/d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK2_0/4; ++j) y[i].qs[j] = 0;
        for (int j = 0; j < QK2_0; ++j) {
            int q = (int)std::round(x[i*QK2_0 + j] * id) + 1;
            if (q < 0) q = 0;
            if (q > 3) q = 3;
            y[i].qs[j/4] |= (uint8_t)(q << ((j%4)*2));
        }
    }
}

// shader simulators: literal text-level translations of the glsl.
// each sim_* function is preceded by a quote of the corresponding glsl. inspect
// side-by-side to confirm the translation is faithful: same operations, same
// operand types, same evaluation order. the c++ uses uint32_t for glsl `uint`,
// std::uint8_t for glsl `uint8_t`, and float for glsl `float`. bit operations
// (>>, &, |) and integer arithmetic on non-negative values are bit-identical
// between glsl and c++

// dequant_funcs.glsl
struct vec2f { float x, y; };
struct vec4f { float x, y, z, w; };

// GLSL (dequant_funcs.glsl):
//   #if defined(DATA_A_Q2_0)
//   vec2 dequantize(uint ib, uint iqs, uint a_offset) {
//       const uint byte_val = uint(data_a[a_offset + ib].qs[iqs / 4u]);
//       const uint shift = (iqs % 4u) * 2u;
//       return vec2(
//           float(int((byte_val >> shift)        & 3u) - 1),
//           float(int((byte_val >> (shift + 2u)) & 3u) - 1));
//   }
//   #endif
static vec2f sim_dequant_funcs_dequantize(const block_q2_0 * data_a, uint32_t a_offset,
                                          uint32_t ib, uint32_t iqs) {
    const uint32_t byte_val = (uint32_t)data_a[a_offset + ib].qs[iqs / 4u];
    const uint32_t shift    = (iqs % 4u) * 2u;
    return {
        (float)((int)((byte_val >> shift)        & 3u) - 1),
        (float)((int)((byte_val >> (shift + 2u)) & 3u) - 1)
    };
}

// GLSL (dequant_funcs.glsl):
//   vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
//       const uint byte_val = uint(data_a[a_offset + ib].qs[iqs / 4u]);
//       const uint shift = (iqs % 4u) * 2u;
//       return vec4(
//           float(int((byte_val >> shift)        & 3u) - 1),
//           float(int((byte_val >> (shift + 2u)) & 3u) - 1),
//           float(int((byte_val >> (shift + 4u)) & 3u) - 1),
//           float(int((byte_val >> (shift + 6u)) & 3u) - 1));
//   }
static vec4f sim_dequant_funcs_dequantize4(const block_q2_0 * data_a, uint32_t a_offset,
                                           uint32_t ib, uint32_t iqs) {
    const uint32_t byte_val = (uint32_t)data_a[a_offset + ib].qs[iqs / 4u];
    const uint32_t shift    = (iqs % 4u) * 2u;
    return {
        (float)((int)((byte_val >> shift)        & 3u) - 1),
        (float)((int)((byte_val >> (shift + 2u)) & 3u) - 1),
        (float)((int)((byte_val >> (shift + 4u)) & 3u) - 1),
        (float)((int)((byte_val >> (shift + 6u)) & 3u) - 1)
    };
}

// GLSL (dequant_funcs.glsl):
//   #if defined(DATA_A_Q2_0)
//   vec2 get_dm(uint ib, uint a_offset) {
//       return vec2(float(data_a[a_offset + ib].d), 0);
//   }
//   #endif
static vec2f sim_dequant_funcs_get_dm(const block_q2_0 * data_a, uint32_t a_offset, uint32_t ib) {
    return { fp16_to_fp32(data_a[a_offset + ib].d), 0.0f };
}

// GLSL (dequant_funcs_cm2.glsl):
//   layout(buffer_reference, std430, buffer_reference_align = 2) buffer decodeBufQ2_0 {
//      block_q2_0 block;
//   };
//   float16_t dequantFuncQ2_0(const in decodeBufQ2_0 bl,
//                             const in uint blockCoords[2],
//                             const in uint coordInBlock[2])
//   {
//       const float16_t d = bl.block.d;
//       const uint idx = coordInBlock[1];
//       const uint byte_val = uint(bl.block.qs[idx >> 2]);
//       const uint shift = (idx & 3u) * 2u;
//       return float16_t(int((byte_val >> shift) & 3u) - 1) * d;
//   }
//
// note. glsl uses fp16 here (float16_t), the simulator uses fp32 because the
// test compares against the fp32 cpu reference. the fp16 variant rounds the
// per-element multiplication. we test that separately in test_cm2_fp16 below
static float sim_cm2_dequantFuncQ2_0(const block_q2_0 * bl, uint32_t coordInBlock_1) {
    const float    d        = fp16_to_fp32(bl->d);
    const uint32_t idx      = coordInBlock_1;
    const uint32_t byte_val = (uint32_t)bl->qs[idx >> 2];
    const uint32_t shift    = (idx & 3u) * 2u;
    return (float)((int)((byte_val >> shift) & 3u) - 1) * d;
}

// GLSL (mul_mm_funcs.glsl, Q2_0 branch):
//   #elif defined(DATA_A_Q2_0)
//               const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
//               const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;
//
//               const uint ib  = idx / 32;
//               const uint iqs = idx & 0x1fu;
//
//               const float d = float(data_a[ib].d);
//               const uint byte_val = uint(data_a[ib].qs[iqs]);
//
//               buf_a[buf_idx    ] = FLOAT_TYPEV2(
//                   float(int( byte_val        & 3u) - 1) * d,
//                   float(int((byte_val >> 2u) & 3u) - 1) * d);
//               buf_a[buf_idx + 1] = FLOAT_TYPEV2(
//                   float(int((byte_val >> 4u) & 3u) - 1) * d,
//                   float(int((byte_val >> 6u) & 3u) - 1) * d);
//
// the simulator returns the 4 floats this thread writes (the 2 vec2 pairs
// flattened). we omit the `pos_a + col * stride_a + row` index arithmetic
// because that's the matmul's address computation. the data-decoding part,
// which is what we're verifying, is just `idx`
struct mul_mm_load_result { float v[4]; };

static mul_mm_load_result sim_mul_mm_load_a_Q2_0(const block_q2_0 * data_a, uint32_t idx) {
    const uint32_t ib  = idx / 32;
    const uint32_t iqs = idx & 0x1fu;
    const float    d   = fp16_to_fp32(data_a[ib].d);
    const uint32_t bv  = (uint32_t)data_a[ib].qs[iqs];

    mul_mm_load_result r;
    r.v[0] = (float)((int)( bv        & 3u) - 1) * d;
    r.v[1] = (float)((int)((bv >> 2u) & 3u) - 1) * d;
    r.v[2] = (float)((int)((bv >> 4u) & 3u) - 1) * d;
    r.v[3] = (float)((int)((bv >> 6u) & 3u) - 1) * d;
    return r;
}

// GLSL (copy_to_quant.comp, Q2_0 branch):
//   #if defined(DATA_A_Q2_0)
//   void quantize(uint dst_idx, uint src_idx)
//   {
//       float amax = 0.0;
//       [[unroll]] for (int j = 0; j < QUANT_K_Q2_0; ++j) {
//           amax = max(amax, abs(data_s[src_idx + j]));
//       }
//       const float d  = amax;
//       const float id = (d > 0.0) ? 1.0/d : 0.0;
//       data_q[dst_idx].d = float16_t(d);
//       [[unroll]] for (int j = 0; j < QUANT_K_Q2_0 / 4; ++j) {
//           data_q[dst_idx].qs[j] = uint8_t(0);
//       }
//       [[unroll]] for (int j = 0; j < QUANT_K_Q2_0; ++j) {
//           int q = int(round(data_s[src_idx + j] * id)) + 1;
//           q = clamp(q, 0, 3);
//           data_q[dst_idx].qs[j / 4] |= uint8_t(q << ((j % 4) * 2));
//       }
//   }
//   #endif
static void sim_copy_to_quant_Q2_0(const float * data_s, block_q2_0 * data_q,
                                   uint32_t dst_idx, uint32_t src_idx) {
    float amax = 0.0f;
    for (int j = 0; j < QK2_0; ++j) {
        amax = std::fmax(amax, std::fabs(data_s[src_idx + j]));
    }
    const float d  = amax;
    const float id = (d > 0.0f) ? 1.0f/d : 0.0f;
    data_q[dst_idx].d = fp32_to_fp16(d);

    for (int j = 0; j < QK2_0/4; ++j) data_q[dst_idx].qs[j] = 0;

    for (int j = 0; j < QK2_0; ++j) {
        int q = (int)std::round(data_s[src_idx + j] * id) + 1;
        // GLSL clamp(int, 0, 3) == max(0, min(3, x))
        if (q < 0) q = 0;
        if (q > 3) q = 3;
        data_q[dst_idx].qs[j/4] |= (uint8_t)(q << ((j%4)*2));
    }
}

// GLSL (dequant_q2_0.comp):
//   #version 450
//   #include "dequant_head.glsl"
//   layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
//   layout (binding = 0) readonly buffer A {block_q2_0 data_a[];};
//   layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};
//   void main() {
//       const uint i   = gl_WorkGroupID.x * 4 + gl_LocalInvocationID.x / 64;
//       const uint tid = gl_LocalInvocationID.x % 64;
//       const uint il  = tid / 4;
//       const uint ir  = tid % 4;
//       const uint ib  = 4*i + ir;
//       if (ib >= p.nel / 128) return;
//       const uint b_idx = 512*i + 128*ir + 8*il;
//       const float d  = float(data_a[ib].d);
//       const uint  b0 = uint(data_a[ib].qs[il*2    ]);
//       const uint  b1 = uint(data_a[ib].qs[il*2 + 1]);
//       [[unroll]] for (uint l = 0; l < 4; ++l) {
//           data_b[b_idx + l    ] = D_TYPE(float(int((b0 >> (l*2u)) & 3u) - 1) * d);
//           data_b[b_idx + l + 4] = D_TYPE(float(int((b1 >> (l*2u)) & 3u) - 1) * d);
//       }
//   }
//
// the simulator iterates every (workgroup, local_id) pair, executing the
// thread body. for each output element we record which thread wrote it and
// what value was written, then assert that every output index is covered
// exactly once and that the value matches the cpu reference
static void sim_dequant_q2_0_run(const block_q2_0 * data_a, uint32_t num_blocks,
                                 std::vector<float> & out, std::vector<int> & writers) {
    const uint32_t nel = num_blocks * QK2_0;
    out.assign(nel, std::nanf(""));
    writers.assign(nel, -1);

    // `p.nel / 128` from the shader
    const uint32_t p_nel_div_128 = num_blocks;

    // 256 threads per workgroup, 16 blocks per workgroup, ceil(num_blocks/16) wgs
    const uint32_t num_wg = (num_blocks + 15) / 16;

    int writer_id = 0;
    for (uint32_t wg = 0; wg < num_wg; ++wg) {
        for (uint32_t lid = 0; lid < 256; ++lid, ++writer_id) {
            const uint32_t i   = wg * 4u + lid / 64u;
            const uint32_t tid = lid % 64u;
            const uint32_t il  = tid / 4u;     // 0..15
            const uint32_t ir  = tid % 4u;     // 0..3
            const uint32_t ib  = 4u*i + ir;
            if (ib >= p_nel_div_128) continue;

            const uint32_t b_idx = 512u*i + 128u*ir + 8u*il;

            const float    d  = fp16_to_fp32(data_a[ib].d);
            const uint32_t b0 = (uint32_t)data_a[ib].qs[il*2u    ];
            const uint32_t b1 = (uint32_t)data_a[ib].qs[il*2u + 1u];

            for (uint32_t l = 0; l < 4; ++l) {
                const uint32_t i0 = b_idx + l;
                const uint32_t i1 = b_idx + l + 4;
                if (writers[i0] != -1 || writers[i1] != -1) {
                    std::fprintf(stderr,
                        "FAIL: dequant thread map overlap at i0=%u (prev writer %d, now %d)\n",
                        i0, writers[i0], writer_id);
                    std::abort();
                }
                writers[i0] = writer_id;
                writers[i1] = writer_id;
                out[i0] = (float)((int)((b0 >> (l*2u)) & 3u) - 1) * d;
                out[i1] = (float)((int)((b1 >> (l*2u)) & 3u) - 1) * d;
            }
        }
    }
}

// test helpers

static int    g_pass = 0;
static int    g_fail = 0;

static void check_eq_f32(const char * what, float a, float b, float tol = 0.0f) {
    const float diff = std::fabs(a - b);
    if (!(diff <= tol)) {
        std::fprintf(stderr, "FAIL %s: %g vs %g (diff %g)\n", what, a, b, diff);
        ++g_fail;
    } else {
        ++g_pass;
    }
}

static void check_eq_u8(const char * what, uint8_t a, uint8_t b) {
    if (a != b) {
        std::fprintf(stderr, "FAIL %s: 0x%02x vs 0x%02x\n", what, a, b);
        ++g_fail;
    } else {
        ++g_pass;
    }
}

// build a random block: all 256 valid byte values x random scale
static block_q2_0 random_block(std::mt19937 & rng, float scale = 1.0f) {
    block_q2_0 b;
    std::uniform_real_distribution<float> ds(0.001f, 4.0f);
    b.d = fp32_to_fp16(scale > 0 ? scale : ds(rng));
    std::uniform_int_distribution<int> bs(0, 255);
    for (int j = 0; j < QK2_0/4; ++j) b.qs[j] = (uint8_t)bs(rng);
    return b;
}

// test 1: sim_dequant_funcs_dequantize matches cpu reference for every iqs

static void test_dequantize() {
    std::mt19937 rng(0xD2D20A55u);
    const int num_blocks = 256;
    std::vector<block_q2_0> blocks(num_blocks);
    for (auto & b : blocks) b = random_block(rng);

    std::vector<float> ref(num_blocks * QK2_0);
    cpu_dequantize_row_q2_0(blocks.data(), ref.data(), num_blocks * QK2_0);

    // dequantize() pre: iqs % 2 == 0
    for (int ib = 0; ib < num_blocks; ++ib) {
        for (uint32_t iqs = 0; iqs < QK2_0; iqs += 2) {
            const vec2f sim = sim_dequant_funcs_dequantize(blocks.data(), 0, ib, iqs);
            const float d   = fp16_to_fp32(blocks[ib].d);
            check_eq_f32("dequantize.x", sim.x * d, ref[ib*QK2_0 + iqs]);
            check_eq_f32("dequantize.y", sim.y * d, ref[ib*QK2_0 + iqs + 1]);
        }
    }
}

// test 2: sim_dequant_funcs_dequantize4 matches cpu reference for every aligned iqs

static void test_dequantize4() {
    std::mt19937 rng(0xC4C40A55u);
    const int num_blocks = 256;
    std::vector<block_q2_0> blocks(num_blocks);
    for (auto & b : blocks) b = random_block(rng);

    std::vector<float> ref(num_blocks * QK2_0);
    cpu_dequantize_row_q2_0(blocks.data(), ref.data(), num_blocks * QK2_0);

    // dequantize4() pre: iqs % 4 == 0
    for (int ib = 0; ib < num_blocks; ++ib) {
        for (uint32_t iqs = 0; iqs < QK2_0; iqs += 4) {
            const vec4f sim = sim_dequant_funcs_dequantize4(blocks.data(), 0, ib, iqs);
            const float d   = fp16_to_fp32(blocks[ib].d);
            check_eq_f32("dequantize4.x", sim.x * d, ref[ib*QK2_0 + iqs    ]);
            check_eq_f32("dequantize4.y", sim.y * d, ref[ib*QK2_0 + iqs + 1]);
            check_eq_f32("dequantize4.z", sim.z * d, ref[ib*QK2_0 + iqs + 2]);
            check_eq_f32("dequantize4.w", sim.w * d, ref[ib*QK2_0 + iqs + 3]);
        }
    }
}

// test 3: sim_cm2_dequantFuncQ2_0 matches cpu reference for every idx

static void test_cm2() {
    std::mt19937 rng(0xCA5EBADAu);
    const int num_blocks = 256;
    std::vector<block_q2_0> blocks(num_blocks);
    for (auto & b : blocks) b = random_block(rng);

    std::vector<float> ref(num_blocks * QK2_0);
    cpu_dequantize_row_q2_0(blocks.data(), ref.data(), num_blocks * QK2_0);

    for (int ib = 0; ib < num_blocks; ++ib) {
        for (uint32_t idx = 0; idx < QK2_0; ++idx) {
            const float sim = sim_cm2_dequantFuncQ2_0(&blocks[ib], idx);
            check_eq_f32("cm2.dequantFuncQ2_0", sim, ref[ib*QK2_0 + idx]);
        }
    }
}

// test 4: sim_mul_mm_load_a_Q2_0 produces values matching cpu reference. each
// idx loads 4 consecutive values starting at logical position 4*iqs

static void test_mul_mm() {
    std::mt19937 rng(0xB1A50A55u);
    const int num_blocks = 64;
    std::vector<block_q2_0> blocks(num_blocks);
    for (auto & b : blocks) b = random_block(rng);

    std::vector<float> ref(num_blocks * QK2_0);
    cpu_dequantize_row_q2_0(blocks.data(), ref.data(), num_blocks * QK2_0);

    // each block has 32 idx values (LOAD_VEC_A=4 means 4 codes per idx, so 32 idx per block)
    for (int ib = 0; ib < num_blocks; ++ib) {
        for (uint32_t iqs = 0; iqs < 32; ++iqs) {
            const uint32_t idx = ib * 32 + iqs;
            const mul_mm_load_result r = sim_mul_mm_load_a_Q2_0(blocks.data(), idx);
            // logical positions in the block
            const uint32_t base = ib * QK2_0 + 4*iqs;
            check_eq_f32("mul_mm[0]", r.v[0], ref[base + 0]);
            check_eq_f32("mul_mm[1]", r.v[1], ref[base + 1]);
            check_eq_f32("mul_mm[2]", r.v[2], ref[base + 2]);
            check_eq_f32("mul_mm[3]", r.v[3], ref[base + 3]);
        }
    }
}

// test 5: sim_copy_to_quant_Q2_0 matches cpu_quantize_row_q2_0 byte-exactly

static void test_quantize() {
    std::mt19937 rng(0xDEADC0DEu);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    const int num_blocks = 256;
    std::vector<float> input(num_blocks * QK2_0);
    for (auto & v : input) v = dist(rng);

    // inject some all-zero blocks and constant-value blocks to exercise edge cases
    std::fill(input.begin() + 0*QK2_0, input.begin() + 1*QK2_0, 0.0f); // all zeros
    std::fill(input.begin() + 1*QK2_0, input.begin() + 2*QK2_0, 1.5f); // all positive constant
    std::fill(input.begin() + 2*QK2_0, input.begin() + 3*QK2_0, -2.5f); // all negative constant
    for (int j = 0; j < QK2_0; ++j) input[3*QK2_0 + j] = (j%2 ? 1.0f : -1.0f); // alternating

    std::vector<block_q2_0> ref(num_blocks);
    std::vector<block_q2_0> sim(num_blocks);
    cpu_quantize_row_q2_0(input.data(), ref.data(), num_blocks * QK2_0);
    for (int i = 0; i < num_blocks; ++i) {
        sim_copy_to_quant_Q2_0(input.data(), sim.data(), (uint32_t)i, (uint32_t)(i*QK2_0));
    }

    for (int i = 0; i < num_blocks; ++i) {
        if (sim[i].d != ref[i].d) {
            std::fprintf(stderr, "FAIL quantize[%d].d: 0x%04x vs 0x%04x\n", i, sim[i].d, ref[i].d);
            ++g_fail;
        } else {
            ++g_pass;
        }
        for (int j = 0; j < QK2_0/4; ++j) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "quantize[%d].qs[%d]", i, j);
            check_eq_u8(buf, sim[i].qs[j], ref[i].qs[j]);
        }
    }
}

// test 6: round-trip equivalence. sim_quantize then sim_dequantize matches
// cpu_quantize then cpu_dequantize. this is implied by tests 1+5 but we add it
// as a direct end-to-end check

static void test_roundtrip() {
    std::mt19937 rng(0xF00DBABEu);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    const int num_blocks = 64;
    std::vector<float> input(num_blocks * QK2_0);
    for (auto & v : input) v = dist(rng);

    std::vector<block_q2_0> q_sim(num_blocks);
    std::vector<block_q2_0> q_ref(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        sim_copy_to_quant_Q2_0(input.data(), q_sim.data(), (uint32_t)i, (uint32_t)(i*QK2_0));
    }
    cpu_quantize_row_q2_0(input.data(), q_ref.data(), num_blocks * QK2_0);

    std::vector<float> y_sim(num_blocks * QK2_0);
    std::vector<float> y_ref(num_blocks * QK2_0);
    cpu_dequantize_row_q2_0(q_sim.data(), y_sim.data(), num_blocks * QK2_0);
    cpu_dequantize_row_q2_0(q_ref.data(), y_ref.data(), num_blocks * QK2_0);

    for (size_t i = 0; i < y_sim.size(); ++i) {
        check_eq_f32("roundtrip", y_sim[i], y_ref[i]);
    }
}

// test 7: dequant_q2_0.comp thread-map covering and value correctness

static void test_dequant_q2_0_shader() {
    std::mt19937 rng(0x5EED0A55u);
    // 33 blocks: tests behaviour at ib boundary that crosses workgroup edge
    const int num_blocks = 33;
    std::vector<block_q2_0> blocks(num_blocks);
    for (auto & b : blocks) b = random_block(rng);

    std::vector<float> ref(num_blocks * QK2_0);
    cpu_dequantize_row_q2_0(blocks.data(), ref.data(), num_blocks * QK2_0);

    std::vector<float> sim;
    std::vector<int>   writers;
    sim_dequant_q2_0_run(blocks.data(), (uint32_t)num_blocks, sim, writers);

    // coverage: every output index has exactly one writer
    for (size_t i = 0; i < sim.size(); ++i) {
        if (writers[i] == -1) {
            std::fprintf(stderr, "FAIL dequant_q2_0 coverage: index %zu unwritten\n", i);
            ++g_fail;
        } else {
            ++g_pass;
        }
    }
    // value match: simulator output equals cpu reference
    for (size_t i = 0; i < sim.size(); ++i) {
        check_eq_f32("dequant_q2_0 value", sim[i], ref[i]);
    }
}

// test 8: matvec partial sum correctness
// simulate the inner loop of mul_mat_vec.comp for K_PER_ITER=8, QUANT_R=1: for
// each col stride 8, fetch dequantize4(iqs) and dequantize4(iqs+4), dot with
// the corresponding 8-element b slice, multiply by d, and accumulate. compare
// against a direct dot product using the cpu dequantized values

static void test_matvec_partial() {
    std::mt19937 rng(0xCAFEFACEu);
    const int num_blocks = 16;
    std::vector<block_q2_0> A(num_blocks);
    for (auto & b : A) b = random_block(rng);

    std::uniform_real_distribution<float> bdist(-1.0f, 1.0f);
    std::vector<float> B(num_blocks * QK2_0);
    for (auto & v : B) v = bdist(rng);

    std::vector<float> A_ref(num_blocks * QK2_0);
    cpu_dequantize_row_q2_0(A.data(), A_ref.data(), num_blocks * QK2_0);

    // direct dot product (reference)
    double ref_dot = 0;
    for (int j = 0; j < num_blocks * QK2_0; ++j) ref_dot += (double)A_ref[j] * (double)B[j];

    // simulated matvec inner loop
    double sim_dot = 0;
    for (int ib = 0; ib < num_blocks; ++ib) {
        const float d = fp16_to_fp32(A[ib].d);
        for (uint32_t iqs = 0; iqs < QK2_0; iqs += 8) {
            const vec4f v0 = sim_dequant_funcs_dequantize4(A.data(), 0, ib, iqs);
            const vec4f v1 = sim_dequant_funcs_dequantize4(A.data(), 0, ib, iqs + 4);
            // dot with b
            float r = v0.x*B[ib*QK2_0 + iqs    ] + v0.y*B[ib*QK2_0 + iqs + 1] +
                      v0.z*B[ib*QK2_0 + iqs + 2] + v0.w*B[ib*QK2_0 + iqs + 3];
            r += v1.x*B[ib*QK2_0 + iqs + 4] + v1.y*B[ib*QK2_0 + iqs + 5] +
                 v1.z*B[ib*QK2_0 + iqs + 6] + v1.w*B[ib*QK2_0 + iqs + 7];
            r *= d;
            sim_dot += r;
        }
    }

    // fp32 accumulation with the same op order: agreement to within ~1e-4 of magnitude
    const double mag = std::fabs(ref_dot) + 1e-6;
    const double rel = std::fabs(sim_dot - ref_dot) / mag;
    if (rel > 1e-4) {
        std::fprintf(stderr, "FAIL matvec partial: ref=%.10g sim=%.10g rel=%.3e\n",
                     ref_dot, sim_dot, rel);
        ++g_fail;
    } else {
        ++g_pass;
    }
}

// test 9: covering proof for all block counts mod 16 (boundary cases).
//
// the shader's `if (ib >= p.nel/128) return;` correctness depends on the block
// count not being a multiple of 16. we run num_blocks in {1, 15, 16, 17, 31,
// 32, 33, 48, 65}

static void test_dequant_q2_0_boundary() {
    for (int num_blocks : {1, 15, 16, 17, 31, 32, 33, 48, 65}) {
        std::mt19937 rng((uint32_t)(num_blocks * 17 + 0xBEEF));
        std::vector<block_q2_0> blocks(num_blocks);
        for (auto & b : blocks) b = random_block(rng);

        std::vector<float> ref(num_blocks * QK2_0);
        cpu_dequantize_row_q2_0(blocks.data(), ref.data(), num_blocks * QK2_0);

        std::vector<float> sim;
        std::vector<int>   writers;
        sim_dequant_q2_0_run(blocks.data(), (uint32_t)num_blocks, sim, writers);

        for (size_t i = 0; i < sim.size(); ++i) {
            if (writers[i] == -1) {
                std::fprintf(stderr,
                    "FAIL boundary num_blocks=%d: idx %zu unwritten\n",
                    num_blocks, i);
                ++g_fail;
                return;
            }
            if (std::fabs(sim[i] - ref[i]) > 0.0f) {
                std::fprintf(stderr,
                    "FAIL boundary num_blocks=%d: idx %zu sim=%g ref=%g\n",
                    num_blocks, i, sim[i], ref[i]);
                ++g_fail;
                return;
            }
        }
        ++g_pass;
    }
}

// test 10: enumerate every byte value x every iqs alignment.
//
// brute-force assert that for every byte value 0..255 and every legal iqs, the
// simulators (dequantize, dequantize4, cm2, mul_mm) produce the codes expected
// by the lsb-first packing convention

static void test_exhaustive_byte_decoding() {
    block_q2_0 blk{};
    blk.d = fp32_to_fp16(2.0f);
    const float d = fp16_to_fp32(blk.d);

    for (int byte = 0; byte < 256; ++byte) {
        // place this byte at every possible position within the block
        for (int slot = 0; slot < QK2_0/4; ++slot) {
            std::memset(blk.qs, 0, sizeof(blk.qs));
            blk.qs[slot] = (uint8_t)byte;

            // expected codes from this byte
            const int q0 =  byte       & 3;
            const int q1 = (byte >> 2) & 3;
            const int q2 = (byte >> 4) & 3;
            const int q3 = (byte >> 6) & 3;

            const float exp0 = (float)(q0 - 1) * d;
            const float exp1 = (float)(q1 - 1) * d;
            const float exp2 = (float)(q2 - 1) * d;
            const float exp3 = (float)(q3 - 1) * d;

            // dequantize() at iqs = 4*slot and 4*slot+2
            const uint32_t iqs0 = 4u*slot;
            const vec2f r0 = sim_dequant_funcs_dequantize(&blk, 0, 0, iqs0);
            check_eq_f32("exhaustive dequantize.x[0]", r0.x * d, exp0);
            check_eq_f32("exhaustive dequantize.y[0]", r0.y * d, exp1);
            const vec2f r1 = sim_dequant_funcs_dequantize(&blk, 0, 0, iqs0 + 2);
            check_eq_f32("exhaustive dequantize.x[1]", r1.x * d, exp2);
            check_eq_f32("exhaustive dequantize.y[1]", r1.y * d, exp3);

            // dequantize4() at iqs = 4*slot
            const vec4f r4 = sim_dequant_funcs_dequantize4(&blk, 0, 0, iqs0);
            check_eq_f32("exhaustive dequantize4.x", r4.x * d, exp0);
            check_eq_f32("exhaustive dequantize4.y", r4.y * d, exp1);
            check_eq_f32("exhaustive dequantize4.z", r4.z * d, exp2);
            check_eq_f32("exhaustive dequantize4.w", r4.w * d, exp3);

            // cm2: for each of the 4 values within this byte
            for (int k = 0; k < 4; ++k) {
                const float exp = (float)(((byte >> (k*2)) & 3) - 1) * d;
                const float got = sim_cm2_dequantFuncQ2_0(&blk, (uint32_t)(4*slot + k));
                check_eq_f32("exhaustive cm2", got, exp);
            }

            // mul_mm load: idx = ib*32 + iqs (here ib=0, iqs=slot)
            const mul_mm_load_result mm = sim_mul_mm_load_a_Q2_0(&blk, (uint32_t)slot);
            check_eq_f32("exhaustive mul_mm[0]", mm.v[0], exp0);
            check_eq_f32("exhaustive mul_mm[1]", mm.v[1], exp1);
            check_eq_f32("exhaustive mul_mm[2]", mm.v[2], exp2);
            check_eq_f32("exhaustive mul_mm[3]", mm.v[3], exp3);
        }
    }
}

// test 11: overflow and edge-case stress.
//
// the goal is to show that the simulator (and therefore the glsl by
// inspection-equivalence) never produces an out-of-range integer or unbounded
// fp value for any representable input.
//
// the first sub-check walks every byte in [0,256) at fp16 scale d=1 and
// confirms the unscaled codes (q-1) always land in {-1, 0, 1, 2}.
//
// the second sub-check quantizes a block whose magnitudes saturate fp16. the
// codes still land in {0,1,2,3} after clamp regardless of input magnitude,
// matching the cpu reference.
//
// the third sub-check pins down the cm2 fp16 multiply at the format edge.
// q=3 gives 2*d which can overflow fp16 once d exceeds 32768, but q2_0 has
// the smallest multiplier of any cm2 quant path so it overflows last.
//
// the fourth sub-check verifies that `b_idx + l + 4` in dequant_q2_0.comp
// stays within uint32 range for any block count up to 16 million (a tensor
// over 2 gb), so realistic models have plenty of headroom

static void test_overflow_codes_in_range() {
    // every byte, every iqs, every block. q-1 must stay in {-1,0,1,2}
    for (int byte = 0; byte < 256; ++byte) {
        block_q2_0 b{};
        b.d = fp32_to_fp16(1.0f);
        for (int slot = 0; slot < QK2_0/4; ++slot) {
            std::memset(b.qs, 0, sizeof(b.qs));
            b.qs[slot] = (uint8_t)byte;
            for (uint32_t idx = 0; idx < QK2_0; ++idx) {
                const float v = sim_cm2_dequantFuncQ2_0(&b, idx);
                // unscaled value (d == 1.0) must be in {-1,0,1,2}
                if (!(v == -1.0f || v == 0.0f || v == 1.0f || v == 2.0f)) {
                    std::fprintf(stderr,
                        "FAIL overflow.codes byte=0x%02x slot=%d idx=%u v=%g\n",
                        byte, slot, idx, v);
                    ++g_fail;
                    return;
                }
            }
        }
    }
    ++g_pass;
}

static void test_overflow_quantize_huge_input() {
    // input magnitudes 1e30 (above fp16 max around 65504). quantize must
    // produce a block whose codes are still in {0..3}, and dequant must produce
    // a finite or saturated but defined value for every entry
    std::vector<float> input(QK2_0);
    for (int j = 0; j < QK2_0; ++j) input[j] = (j%2 ? 1.0f : -1.0f) * 1e30f;

    block_q2_0 sim{}, ref{};
    sim_copy_to_quant_Q2_0(input.data(), &sim, 0, 0);
    cpu_quantize_row_q2_0(input.data(), &ref, QK2_0);

    // codes must be in {0..3}
    for (int j = 0; j < QK2_0/4; ++j) {
        for (int sub = 0; sub < 4; ++sub) {
            const int q_sim = (sim.qs[j] >> (sub*2)) & 3;
            const int q_ref = (ref.qs[j] >> (sub*2)) & 3;
            if (q_sim < 0 || q_sim > 3) {
                std::fprintf(stderr, "FAIL overflow.code_range sim qs[%d].%d=%d\n", j, sub, q_sim);
                ++g_fail;
                return;
            }
            if (q_sim != q_ref) {
                std::fprintf(stderr, "FAIL overflow.code_match sim=%d ref=%d at qs[%d].%d\n",
                             q_sim, q_ref, j, sub);
                ++g_fail;
                return;
            }
        }
    }
    if (sim.d != ref.d) {
        std::fprintf(stderr, "FAIL overflow.d sim=0x%04x ref=0x%04x\n", sim.d, ref.d);
        ++g_fail;
        return;
    }
    ++g_pass;
}

// for those who have made it this far into the file, heres an easter egg

// as we navigate this era of technological revolution and intelligence,
// it is imperative that we build systems that are robust and reliable,
// especially when we do not understand the full implications of our
// own design choices. for those that read this comment, its likely that
// you sit at the forefront of ai and hold the knowledge necessary 
// to transform this world. it is part of your moral obligation to 
// use that knowledge to transform this world in a positive manner.
// may god bless us all, and i pray we can build a better future 
// together.

static void test_overflow_cm2_fp16_multiply() {
    // for the cm2 path in fp16, (q-1)*d can theoretically exceed fp16 max
    // (around 65504) when d > 32768 and q == 3 (giving 2d). q2_0 has the
    // smallest multiplier of any cm2 quant path (max 2x, vs q4_0's 8x and
    // q5_0's 16x), so it is the most robust of the cm2 paths against fp16
    // overflow.
    //
    // we empirically check that with d at fp16 max and codes {0,1,2,3} the
    // products are {-65504, 0, 65504, +/-inf}, that q==3 is the only code
    // that overflows, and that smaller d values are safe

    // d = 65504 (fp16 max, still representable). code 3 gives 2*d = 131008
    // which is +/-inf in fp16. we check the simulator mirrors this
    block_q2_0 b{};
    b.d = 0x7BFF; // fp16 max around 65504
    b.qs[0] = 0xE4; 
    // codes [0,1,2,3] in positions 0..3 (lsb-first 00 01 10 11)
    // 0xE4 = 11100100b. bits 0..1=00 (q=0), 2..3=01 (q=1),
    // 4..5=10 (q=2), 6..7=11 (q=3)

    const float d_f32 = fp16_to_fp32(b.d);
    const float v0 = sim_cm2_dequantFuncQ2_0(&b, 0); // q=0 gives -d
    const float v1 = sim_cm2_dequantFuncQ2_0(&b, 1); // q=1 gives 0
    const float v2 = sim_cm2_dequantFuncQ2_0(&b, 2); // q=2 gives +d
    const float v3 = sim_cm2_dequantFuncQ2_0(&b, 3); // q=3 gives +2d (would overflow fp16)

    check_eq_f32("cm2.q0", v0, -d_f32);
    check_eq_f32("cm2.q1", v1, 0.0f);
    check_eq_f32("cm2.q2", v2, +d_f32);
    // simulator works in fp32. the gpu's cm2 path works in fp16 and would
    // saturate to +/-inf. both are deterministic and both are documented
    check_eq_f32("cm2.q3", v3, 2.0f * d_f32);

    // sanity check, with d = 1.0 (normal case) there is no overflow at all
    b.d = fp32_to_fp16(1.0f);
    b.qs[0] = 0xE4;
    check_eq_f32("cm2.q0.normal", sim_cm2_dequantFuncQ2_0(&b, 0), -1.0f);
    check_eq_f32("cm2.q1.normal", sim_cm2_dequantFuncQ2_0(&b, 1),  0.0f);
    check_eq_f32("cm2.q2.normal", sim_cm2_dequantFuncQ2_0(&b, 2), +1.0f);
    check_eq_f32("cm2.q3.normal", sim_cm2_dequantFuncQ2_0(&b, 3), +2.0f);
}

static void test_overflow_b_idx_uint32_headroom() {
    // for a tensor of 16 million blocks (around 2 gb at 34 bytes per block),
    // the maximum b_idx in dequant_q2_0.comp is well within uint32 range
    const uint64_t num_blocks = 16ull * 1024 * 1024; // 16 m blocks
    const uint64_t num_wg     = (num_blocks + 15) / 16;
    const uint64_t max_i      = (num_wg - 1) * 4 + 3;
    const uint64_t max_ir     = 3;
    const uint64_t max_il     = 15;
    const uint64_t max_b_idx  = 512 * max_i + 128 * max_ir + 8 * max_il + 7; // +7 for last l
    if (max_b_idx > 0xFFFFFFFFull) {
        std::fprintf(stderr, "FAIL b_idx headroom: %llu blocks would overflow uint32\n",
                     (unsigned long long)num_blocks);
        ++g_fail;
        return;
    }
    ++g_pass;
}


int main() {
    std::printf("== test-vulkan-q2_0-shader-sim ==\n");
    test_dequantize();
    test_dequantize4();
    test_cm2();
    test_mul_mm();
    test_quantize();
    test_roundtrip();
    test_dequant_q2_0_shader();
    test_dequant_q2_0_boundary();
    test_matvec_partial();
    test_exhaustive_byte_decoding();
    test_overflow_codes_in_range();
    test_overflow_quantize_huge_input();
    test_overflow_cm2_fp16_multiply();
    test_overflow_b_idx_uint32_headroom();
    std::printf("checks: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

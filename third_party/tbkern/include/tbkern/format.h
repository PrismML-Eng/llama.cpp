/* tbkern/format.h — FROZEN Phase-0 contract.
 *
 * Low-level, dependency-free format constants and helpers shared by every
 * translation unit (C kernels, packers) and mirrored by the Python reference.
 *
 * Ground truth: mainline llama.cpp GGML_TYPE_Q2_0 (ggml-common.h / ggml-quants.c).
 * See docs/FORMAT.md for the normative spec and bit-exact diagrams.
 *
 * RULE (docs/CONTEXT.md §Rules): after Phase 0 this header may only be changed
 * by the orchestrator. Code against it; do not edit it.
 */
#ifndef TBKERN_FORMAT_H
#define TBKERN_FORMAT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 *  Block / group constants
 * ------------------------------------------------------------------ */

/* Mainline GGUF Q2_0 native block covers 64 weights (QK2_0). PrismML's native
 * variant uses group-128 blocks with identical code semantics. All tbkern code
 * parameterizes the group size G at runtime; these are the only legal values. */
#define TBK_QK2_0        64          /* mainline GGUF Q2_0 block length          */
#define TBK_G_MIN        64
#define TBK_G_MAX        128

/* A GGUF Q2_0 block is: 1x fp16 scale ("d") followed by G/4 code bytes
 * (2 bits per weight, 4 weights per byte).  Byte count per block: */
#define TBK_GGUF_SCALE_BYTES   2                    /* sizeof(ggml_half)         */
#define TBK_GGUF_BLOCK_BYTES(G) (TBK_GGUF_SCALE_BYTES + (size_t)(G) / 4u)
/*   G=64  -> 2 + 16 = 18 bytes (2.250 bits/wt)
 *   G=128 -> 2 + 32 = 34 bytes (2.125 bits/wt)  */

/* Code semantics (2-bit quant code -> ternary/quaternary weight value):
 *   00 = -1, 01 = 0, 10 = +1, 11 = +2.   value = (code - 1) * scale.
 * The unsigned datum u = code = (w + 1) in {0..3} is what the VNNI path feeds
 * to vpdpbusd (see docs/FORMAT.md §VNNI). +2 (code 3) is legal Q2_0 but never
 * appears in real ternary Bonsai tensors; the bitplane path rejects it. */
#define TBK_CODE_NEG1    0u          /* w = -1 */
#define TBK_CODE_ZERO    1u          /* w =  0 (also the pad code)               */
#define TBK_CODE_POS1    2u          /* w = +1 */
#define TBK_CODE_POS2    3u          /* w = +2 (legal, ternary-only paths reject)*/

/* Padding: logical K is padded up to Kp, the next multiple of G, using pad
 * code 01 (w = 0). Activations are zero-padded to Kp. See docs/FORMAT.md. */
#define TBK_CODE_PAD     TBK_CODE_ZERO

/* Row alignment for packed kernel-native layouts (both bitplane and codes):
 * every row starts on a 64-byte boundary; the packed bytes per row equal
 * Kp/4 (both layouts), rounded up to this alignment for the row stride. */
#define TBK_ROW_ALIGN    64u

/* Number of quant groups in a padded row (one fp32 scale per group). */
#define TBK_NUM_GROUPS(Kp, G)   ((size_t)(Kp) / (size_t)(G))

/* Number of 64-weight subgroups in a padded row (codes-layout storage unit). */
#define TBK_NUM_SUBGROUPS(Kp)   ((size_t)(Kp) / 64u)

/* Packed bytes actually used per row (before 64-byte stride rounding). Both
 * layouts store 2 bits/weight = Kp/4 bytes. */
#define TBK_ROW_USED_BYTES(Kp)  ((size_t)(Kp) / 4u)

/* Round x up to a multiple of a (a must be a power of two). */
static inline size_t tbk_align_up(size_t x, size_t a) {
    return (x + (a - 1u)) & ~(a - 1u);
}

/* 64-byte-aligned row stride (bytes) for a packed layout of Kp columns. */
static inline size_t tbk_row_stride_bytes(int Kp) {
    return tbk_align_up(TBK_ROW_USED_BYTES(Kp), TBK_ROW_ALIGN);
}

/* Pad K up to the next multiple of G. */
static inline int tbk_pad_k(int K, int G) {
    return (int)(((K + G - 1) / G) * G);
}

/* Number of GGUF Q2_0 blocks per row for a K-wide row (ceil division). */
static inline int tbk_gguf_blocks_per_row(int K, int G) {
    return (K + G - 1) / G;
}

/* ------------------------------------------------------------------ *
 *  IEEE-754 binary16 <-> binary32 (portable, ggml/numpy bit-compatible)
 *
 *  Verified bit-exact against numpy.float16 over all 65536 half patterns
 *  (widening) and all finite round-trips (narrowing, round-to-nearest-even).
 *  Software only: usable from baseline (no-F16C) translation units.
 * ------------------------------------------------------------------ */

static inline float tbk_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0u) {
        if (mant == 0u) {
            f = sign;                              /* +/- zero                 */
        } else {                                   /* subnormal -> normalize   */
            exp = 1u;
            while ((mant & 0x400u) == 0u) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            f = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        f = sign | 0x7F800000u | (mant << 13);     /* inf / nan                */
    } else {
        f = sign | ((exp + 112u) << 23) | (mant << 13); /* normal (112=127-15)*/
    }
    union { uint32_t u; float f; } o;
    o.u = f;
    return o.f;
}

static inline uint16_t tbk_fp32_to_fp16(float value) {
    union { float f; uint32_t u; } in;
    in.f = value;
    uint32_t x    = in.u;
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t e    = (x >> 23) & 0xFFu;
    uint32_t mant = x & 0x7FFFFFu;
    int32_t  exp  = (int32_t)e - 127 + 15;
    if (e == 0xFFu) {                              /* inf / nan                */
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1F) {                             /* overflow -> inf          */
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exp <= 0) {                                /* subnormal / underflow    */
        if (exp < -10) return (uint16_t)sign;      /* rounds to +/- zero       */
        mant |= 0x800000u;
        int      shift = 14 - exp;
        uint32_t half  = 1u << (shift - 1);
        uint32_t res   = mant >> shift;
        uint32_t rem   = mant & ((1u << shift) - 1u);
        if (rem > half || (rem == half && (res & 1u))) res++;   /* RNE        */
        return (uint16_t)(sign | res);
    }
    uint16_t h   = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;      /* RNE        */
    return h;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TBKERN_FORMAT_H */

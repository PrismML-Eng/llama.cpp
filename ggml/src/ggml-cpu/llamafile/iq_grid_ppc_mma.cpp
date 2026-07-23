// iq_grid_ppc_mma.cpp - imported from github.com/mavin2009/ppc-mma-kernels
// (standalone-verified under qemu -cpu power10). Grid-codebook, ternary,
// and NVFP4 formats via decode-at-repack + shared signed kernels (32-
// and 16-deep chunk variants). Local grid tables in iq_grids_ppc.h.

#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "kquants_ppc_mma.h"

#include <altivec.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "iq_grids_ppc.h"

#define QK_K 256
#define IQ1S_DELTA 0.125f

typedef uint16_t ppc_half;

typedef struct { uint8_t qs[(QK_K - 4*QK_K/64)/5]; uint8_t qh[QK_K/64]; ppc_half d; } block_tq1_0_ppc;
typedef struct { uint8_t qs[QK_K/4]; ppc_half d; } block_tq2_0_ppc;
typedef struct { ppc_half d; uint16_t qs[QK_K/8]; } block_iq2_xxs_ppc;
typedef struct { ppc_half d; uint8_t qs[3*QK_K/8]; } block_iq3_xxs_ppc;
typedef struct { ppc_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32];
                 uint8_t signs[QK_K/8]; uint8_t scales[QK_K/64]; } block_iq3_s_ppc;
typedef struct { ppc_half d; uint8_t qs[QK_K/8]; uint16_t qh[QK_K/32]; } block_iq1_s_ppc;
typedef struct { ppc_half d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; } block_iq2_xs_ppc;
typedef struct { ppc_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32]; } block_iq2_s_ppc;
typedef struct { uint8_t qs[QK_K/8]; uint8_t qh[QK_K/16]; uint8_t scales[QK_K/32]; } block_iq1_m_ppc;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K_ppc;
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16
typedef struct { uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; uint8_t qs[QK_NVFP4/2]; } block_nvfp4_ppc;
#define QK8_0 32
typedef struct { ppc_half d; int8_t qs[QK8_0]; } block_q8_0_ppc;

static_assert(sizeof(block_tq1_0_ppc)   == 2 + 4 + 48, "bad tq1_0");
static_assert(sizeof(block_tq2_0_ppc)   == 2 + 64, "bad tq2_0");
static_assert(sizeof(block_iq2_xxs_ppc) == 2 + QK_K/4, "bad iq2_xxs");
static_assert(sizeof(block_iq3_xxs_ppc) == 2 + 3*QK_K/8, "bad iq3_xxs");
static_assert(sizeof(block_iq3_s_ppc)   == 2 + 13*(QK_K/32) + QK_K/64, "bad iq3_s");
static_assert(sizeof(block_iq1_s_ppc)   == 2 + QK_K/8 + QK_K/16, "bad iq1_s");
static_assert(sizeof(block_iq2_xs_ppc)  == 2 + QK_K/4 + QK_K/32, "bad iq2_xs");
static_assert(sizeof(block_iq2_s_ppc)   == 2 + QK_K/4 + QK_K/32 + QK_K/32, "bad iq2_s");
static_assert(sizeof(block_iq1_m_ppc)   == QK_K/8 + QK_K/16 + QK_K/32, "bad iq1_m");
static_assert(sizeof(block_nvfp4_ppc)   == 4 + QK_NVFP4/2, "bad nvfp4");

static inline float fp16_to_fp32(ppc_half h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f; memcpy(&f, &bits, 4);
    return f;
}

// ---- scalar decoders: superblock -> 256 signed int8 codes + 8 scales ----
// (one per 32-chunk).  Direct ports of dequantize_row semantics.

static void dec_tq2_0(const block_tq2_0_ppc * b, int8_t code[QK_K], float sc[8]) {
    const float d = fp16_to_fp32(b->d);
    int8_t * y = code;
    for (size_t j = 0; j < sizeof(b->qs); j += 32)
        for (size_t l = 0; l < 4; ++l)
            for (size_t m = 0; m < 32; ++m)
                *y++ = (int8_t)(((b->qs[j + m] >> (l*2)) & 3) - 1);
    for (int c = 0; c < 8; c++) sc[c] = d;
}

static void dec_tq1_0(const block_tq1_0_ppc * b, int8_t code[QK_K], float sc[8]) {
    static const uint8_t pow3[6] = { 1, 3, 9, 27, 81, 243 };
    const float d = fp16_to_fp32(b->d);
    int8_t * y = code;
    for (size_t j = 0; j < sizeof(b->qs) - sizeof(b->qs) % 32; j += 32)
        for (size_t n = 0; n < 5; ++n)
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = (uint8_t)(b->qs[j + m] * pow3[n]);
                *y++ = (int8_t)((((uint16_t)q * 3) >> 8) - 1);
            }
    for (size_t j = sizeof(b->qs) - sizeof(b->qs) % 32; j < sizeof(b->qs); j += 16)
        for (size_t n = 0; n < 5; ++n)
            for (size_t m = 0; m < 16; ++m) {
                uint8_t q = (uint8_t)(b->qs[j + m] * pow3[n]);
                *y++ = (int8_t)((((uint16_t)q * 3) >> 8) - 1);
            }
    for (size_t n = 0; n < 4; ++n)
        for (size_t j = 0; j < sizeof(b->qh); ++j) {
            uint8_t q = (uint8_t)(b->qh[j] * pow3[n]);
            *y++ = (int8_t)((((uint16_t)q * 3) >> 8) - 1);
        }
    for (int c = 0; c < 8; c++) sc[c] = d;
}

static void dec_iq2_xxs(const block_iq2_xxs_ppc * b, int8_t code[QK_K], float sc[8]) {
    const float d = fp16_to_fp32(b->d);
    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        memcpy(aux32, b->qs + 4*ib32, 8);
        sc[ib32] = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
        int8_t * y = code + 32*ib32;
        for (int l = 0; l < 4; ++l) {
            const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
            const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
            for (int j = 0; j < 8; ++j)
                y[8*l + j] = (int8_t)((signs & kmask_iq2xs[j]) ? -(int)grid[j] : (int)grid[j]);
        }
    }
}

static void dec_iq3_xxs(const block_iq3_xxs_ppc * b, int8_t code[QK_K], float sc[8]) {
    const float d = fp16_to_fp32(b->d);
    const uint8_t * qs = b->qs;
    const uint8_t * sas = b->qs + QK_K/4;
    uint32_t aux32;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        memcpy(&aux32, sas + 4*ib32, 4);
        sc[ib32] = d * (0.5f + (aux32 >> 28)) * 0.5f;
        int8_t * y = code + 32*ib32;
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
            const uint8_t * g1 = (const uint8_t *)(iq3xxs_grid + qs[8*ib32 + 2*l + 0]);
            const uint8_t * g2 = (const uint8_t *)(iq3xxs_grid + qs[8*ib32 + 2*l + 1]);
            for (int j = 0; j < 4; ++j) {
                y[8*l + j + 0] = (int8_t)((signs & kmask_iq2xs[j+0]) ? -(int)g1[j] : (int)g1[j]);
                y[8*l + j + 4] = (int8_t)((signs & kmask_iq2xs[j+4]) ? -(int)g2[j] : (int)g2[j]);
            }
        }
    }
}

static void dec_iq3_s(const block_iq3_s_ppc * b, int8_t code[QK_K], float sc[8]) {
    const float d = fp16_to_fp32(b->d);
    const uint8_t * qs = b->qs;
    const uint8_t * qh = b->qh;
    const uint8_t * signs = b->signs;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        sc[ib32 + 0] = d * (1 + 2*(b->scales[ib32/2] & 0xF));
        sc[ib32 + 1] = d * (1 + 2*(b->scales[ib32/2] >>  4));
        for (int half = 0; half < 2; half++) {
            int8_t * y = code + 32*(ib32 + half);
            for (int l = 0; l < 4; ++l) {
                const uint8_t * g1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[half] << (8-2*l)) & 256)));
                const uint8_t * g2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[half] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[8*l + j + 0] = (int8_t)((signs[l] & kmask_iq2xs[j+0]) ? -(int)g1[j] : (int)g1[j]);
                    y[8*l + j + 4] = (int8_t)((signs[l] & kmask_iq2xs[j+4]) ? -(int)g2[j] : (int)g2[j]);
                }
            }
            qs += 8; signs += 4;
        }
        qh += 2;
    }
}

// IQ1_S: value = dl*(g + delta), g in {-1,+1}, delta = +/-1/8 per 32.
// Exact integer form: code = 8*g + (delta<0 ? -1 : +1), scale = dl/8.
static void dec_iq1_s(const block_iq1_s_ppc * b, int8_t code[QK_K], float sc[8]) {
    const float d = fp16_to_fp32(b->d);
    const uint8_t * qs = b->qs;
    for (int ib = 0; ib < 8; ++ib) {
        const float dl = d * (2*((b->qh[ib] >> 12) & 7) + 1);
        sc[ib] = dl * 0.125f;
        const int dsign = (b->qh[ib] & 0x8000) ? -1 : +1;
        int8_t * y = code + 32*ib;
        for (int l = 0; l < 4; ++l) {
            const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((b->qh[ib] >> 3*l) & 7) << 8)));
            for (int j = 0; j < 8; ++j)
                y[8*l + j] = (int8_t)(8*grid[j] + dsign);
        }
        qs += 4;
    }
}


// NVFP4 (64-element blocks, four 16-element sub-blocks with UE4M3
// scales, MXFP4's pre-doubled e2m1 codebook -- the *0.5 inside the
// UE4M3 conversion compensates the doubling, matching ggml's dequant).
static const int8_t nvfp4_kvalues[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};
#include <cmath>
static inline float ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw = (exp == 0) ? ldexpf((float)man, -9)
                           : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    return raw * 0.5f;
}

// one 64-block -> 64 signed codes + 4 per-16 scales
static void dec_nvfp4(const block_nvfp4_ppc * b, int8_t code[QK_NVFP4], float sc[4]) {
    for (int s = 0; s < 4; s++) {
        sc[s] = ue4m3_to_fp32(b->d[s]);
        for (int j = 0; j < 8; j++) {
            code[16*s + j + 0] = nvfp4_kvalues[b->qs[8*s + j] & 0xF];
            code[16*s + j + 8] = nvfp4_kvalues[b->qs[8*s + j] >>  4];
        }
    }
}

// ---- per-16-scale decoders (codes + 16 scales per superblock) ----

static void dec16_iq2_xs(const block_iq2_xs_ppc * b, int8_t code[QK_K], float sc[16]) {
    const float d = fp16_to_fp32(b->d);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        sc[2*ib32 + 0] = d * (0.5f + (b->scales[ib32] & 0xF)) * 0.25f;
        sc[2*ib32 + 1] = d * (0.5f + (b->scales[ib32] >>  4)) * 0.25f;
        int8_t * y = code + 32*ib32;
        for (int l = 0; l < 4; ++l) {
            const uint16_t w = b->qs[4*ib32 + l];
            const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (w & 511));
            const uint8_t signs = ksigns_iq2xs[w >> 9];
            for (int j = 0; j < 8; ++j)
                y[8*l + j] = (int8_t)((signs & kmask_iq2xs[j]) ? -(int)grid[j] : (int)grid[j]);
        }
    }
}

static void dec16_iq2_s(const block_iq2_s_ppc * b, int8_t code[QK_K], float sc[16]) {
    const float d = fp16_to_fp32(b->d);
    const uint8_t * qs = b->qs;
    const uint8_t * signs = b->qs + QK_K/8;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        sc[2*ib32 + 0] = d * (0.5f + (b->scales[ib32] & 0xF)) * 0.25f;
        sc[2*ib32 + 1] = d * (0.5f + (b->scales[ib32] >>  4)) * 0.25f;
        int8_t * y = code + 32*ib32;
        for (int l = 0; l < 4; ++l) {
            const uint8_t * grid = (const uint8_t *)(iq2s_grid +
                (qs[l] | ((b->qh[ib32] << (8 - 2*l)) & 0x300)));
            for (int j = 0; j < 8; ++j)
                y[8*l + j] = (int8_t)((signs[l] & kmask_iq2xs[j]) ? -(int)grid[j] : (int)grid[j]);
        }
        qs += 4; signs += 4;
    }
}

// IQ1_M: fp16 superscale reassembled from scale-nibble high bits;
// per-16 3-bit scales, per-8 deltas.  Exact integer form as IQ1_S:
// codes = 8*grid + sign(delta), scale = dl/8.
static void dec16_iq1_m(const block_iq1_m_ppc * b, int8_t code[QK_K], float sc[16]) {
    const uint16_t * scw = (const uint16_t *)b->scales;
    uint16_t du16 = (uint16_t)((scw[0] >> 12) | ((scw[1] >> 8) & 0x00f0) |
                               ((scw[2] >> 4) & 0x0f00) | (scw[3] & 0xf000));
    const float d = fp16_to_fp32(du16);
    const uint8_t * qs = b->qs;
    const uint8_t * qh = b->qh;
    for (int ib = 0; ib < 8; ++ib) {
        const float dl1 = d * (2*((scw[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1);
        const float dl2 = d * (2*((scw[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1);
        sc[2*ib + 0] = dl1 * 0.125f;
        sc[2*ib + 1] = dl2 * 0.125f;
        uint16_t idx[4]; int dsg[4];
        idx[0] = (uint16_t)(qs[0] | ((qh[0] << 8) & 0x700));
        idx[1] = (uint16_t)(qs[1] | ((qh[0] << 4) & 0x700));
        idx[2] = (uint16_t)(qs[2] | ((qh[1] << 8) & 0x700));
        idx[3] = (uint16_t)(qs[3] | ((qh[1] << 4) & 0x700));
        dsg[0] = (qh[0] & 0x08) ? -1 : +1;
        dsg[1] = (qh[0] & 0x80) ? -1 : +1;
        dsg[2] = (qh[1] & 0x08) ? -1 : +1;
        dsg[3] = (qh[1] & 0x80) ? -1 : +1;
        int8_t * y = code + 32*ib;
        for (int l = 0; l < 4; ++l) {
            const int8_t * grid = (const int8_t *)(iq1s_grid + idx[l]);
            for (int j = 0; j < 8; ++j)
                y[8*l + j] = (int8_t)(8*grid[j] + dsg[l]);
        }
        qs += 4; qh += 2;
    }
}

#if defined(__MMA__) && defined(__powerpc64__)

typedef vector unsigned char  vuc;
typedef vector signed char    vsc;
typedef vector unsigned int   vui;
typedef vector signed int     vsi;
typedef vector float          vfl;

// Unaligned 16-byte load via memcpy: compiles to a single lxv (which is
// alignment-agnostic on POWER) while staying well-defined C++ for any
// source alignment -- several block structs place qs at odd offsets.
static inline vuc load16u(const void * p) { vuc v; memcpy(&v, p, 16); return v; }


#define KC_ELEMS  2048
#define KC_CH     (KC_ELEMS / 32)
#define MR        8
#define NR        8

typedef struct {
    vuc v[KC_CH][16];
    vfl sA  [KC_CH][2];
    vfl C128[KC_CH][2];
} agrid_t;

typedef struct {
    vuc v[KC_CH][16];
    vfl dB[KC_CH][2];
} bgrid_t;

static inline void mma_transpose4(const vui rows[4], vuc * out, int stride) {
    vui t0 = vec_mergeh(rows[0], rows[1]);
    vui t1 = vec_mergel(rows[0], rows[1]);
    vui t2 = vec_mergeh(rows[2], rows[3]);
    vui t3 = vec_mergel(rows[2], rows[3]);
    out[0*stride] = (vuc)vec_xxpermdi(t0, t2, 0);
    out[1*stride] = (vuc)vec_xxpermdi(t0, t2, 3);
    out[2*stride] = (vuc)vec_xxpermdi(t1, t3, 0);
    out[3*stride] = (vuc)vec_xxpermdi(t1, t3, 3);
}

static inline int hsum(vsi s) { return s[0] + s[1] + s[2] + s[3]; }

#ifdef IQGRID_LXVP
static inline void load_pair(const vuc * p, vuc out[2]) {
    __vector_pair vp = *(const __vector_pair *)p;
    __builtin_vsx_disassemble_pair((void *)out, &vp);
}
#endif

static void grid_place_chunk(agrid_t * T, int64_t ch,
                             const vsc w[MR][2], const float scale[MR]) {
    for (int g = 0; g < 2; g++) {
        float W[4];
        vui rows4[4];
        for (int h = 0; h < 2; h++) {
            for (int r = 0; r < 4; r++) rows4[r] = (vui)w[4*g + r][h];
            mma_transpose4(rows4, &T->v[ch][8*h + g], 2);
        }
        for (int r = 0; r < 4; r++) {
            vsi z = vec_splats(0);
            vsi s = vec_sum4s(w[4*g + r][0], z);
            s = vec_sum4s(w[4*g + r][1], s);
            W[r] = (float)hsum(s);
        }
        T->sA  [ch][g] = (vfl){ scale[4*g], scale[4*g+1], scale[4*g+2], scale[4*g+3] };
        T->C128[ch][g] = (vfl){ 128.0f*W[0], 128.0f*W[1],
                                128.0f*W[2], 128.0f*W[3] };  // exact int; see field FAIL 2026-07-21
    }
}

static inline int64_t rt(int64_t m) { return (m + MR - 1) / MR; }
static inline int64_t ct(int64_t n) { return (n + NR - 1) / NR; }
static inline int64_t sl(int64_t k) { return (k/32 + KC_CH - 1) / KC_CH; }

extern "C" size_t grid_apack_size(int64_t m, int64_t k) {
    return (((size_t)(rt(m) * sl(k)) * sizeof(agrid_t)) + 63) & ~(size_t)63;
}
extern "C" size_t grid_bpack_size(int64_t n, int64_t k) {
    return (((size_t)(ct(n) * sl(k)) * sizeof(bgrid_t)) + 63) & ~(size_t)63;
}

// generic repack over any superblock decoder
template <typename BLK, void (*DEC)(const BLK *, int8_t[QK_K], float[8])>
static void repack_grid(const BLK * A, int64_t lda, int64_t m, int64_t k, agrid_t * P) {
    const int64_t nsb = k/QK_K, ns = sl(k);
    for (int64_t it = 0; it < rt(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        agrid_t * T = &P[it*ns + s];
        const int64_t sb0 = (s*KC_CH)/8;
        const int64_t nsl = (nsb - sb0) < KC_CH/8 ? (nsb - sb0) : KC_CH/8;
        for (int64_t sb = 0; sb < nsl; sb++) {
            int8_t code[MR][QK_K]; float sc[MR][8];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                DEC(&A[rr*lda + sb0 + sb], code[r], sc[r]);
            }
            for (int c = 0; c < 8; c++) {
                vsc w[MR][2]; float scale[MR];
                for (int r = 0; r < MR; r++) {
                    memcpy(&w[r][0], code[r] + 32*c,      16);
                    memcpy(&w[r][1], code[r] + 32*c + 16, 16);
                    scale[r] = sc[r][c];
                }
                grid_place_chunk(T, 8*sb + c, w, scale);
            }
        }
    }
}

extern "C" void grid_repack_tq2_0(const block_tq2_0_ppc * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_grid<block_tq2_0_ppc, dec_tq2_0>(A, lda, m, k, (agrid_t *)p); }
extern "C" void grid_repack_tq1_0(const block_tq1_0_ppc * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_grid<block_tq1_0_ppc, dec_tq1_0>(A, lda, m, k, (agrid_t *)p); }
extern "C" void grid_repack_iq2_xxs(const block_iq2_xxs_ppc * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_grid<block_iq2_xxs_ppc, dec_iq2_xxs>(A, lda, m, k, (agrid_t *)p); }
extern "C" void grid_repack_iq3_xxs(const block_iq3_xxs_ppc * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_grid<block_iq3_xxs_ppc, dec_iq3_xxs>(A, lda, m, k, (agrid_t *)p); }
extern "C" void grid_repack_iq3_s(const block_iq3_s_ppc * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_grid<block_iq3_s_ppc, dec_iq3_s>(A, lda, m, k, (agrid_t *)p); }
extern "C" void grid_repack_iq1_s(const block_iq1_s_ppc * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_grid<block_iq1_s_ppc, dec_iq1_s>(A, lda, m, k, (agrid_t *)p); }

extern "C" void grid_pack_b_q8_K(const block_q8_K_ppc * B, int64_t ldb,
                                 int64_t n, int64_t k, void * packed) {
    bgrid_t * P = (bgrid_t *)packed;
    const int64_t nsb = k/QK_K, ns = sl(k);
    const vuc flip = vec_splats((unsigned char)0x80);
    for (int64_t jt = 0; jt < ct(n); jt++)
    for (int64_t s = 0; s < ns; s++) {
        bgrid_t * T = &P[jt*ns + s];
        const int64_t sb0 = (s*KC_CH)/8;
        const int64_t nsl = (nsb - sb0) < KC_CH/8 ? (nsb - sb0) : KC_CH/8;
        for (int64_t sb = 0; sb < nsl; sb++) {
            const block_q8_K_ppc * yb[NR]; float dB[NR];
            for (int j = 0; j < NR; j++) {
                int64_t jj = jt*NR + j; if (jj >= n) jj = n - 1;
                yb[j] = &B[jj*ldb + sb0 + sb];
                dB[j] = yb[j]->d;
            }
            for (int ib = 0; ib < 8; ib++) {
                const int64_t ch = 8*sb + ib;
                T->dB[ch][0] = (vfl){ dB[0], dB[1], dB[2], dB[3] };
                T->dB[ch][1] = (vfl){ dB[4], dB[5], dB[6], dB[7] };
                vui rows4[4];
                for (int a = 0; a < 2; a++)
                    for (int h = 0; h < 2; h++) {
                        for (int j = 0; j < 4; j++)
                            rows4[j] = (vui)vec_xor(
                                load16u((const uint8_t *)(yb[4*a + j]->qs) + (32*ib + 16*h)), flip);
                        mma_transpose4(rows4, &T->v[ch][8*h + a], 2);
                    }
            }
        }
    }
}

__attribute__((unused)) static void kernel_grid_8x8(const agrid_t * PA, const bgrid_t * PB,
                            int64_t nch, vfl fin[MR][2]) {
    for (int64_t ch = 0; ch < nch; ch++) {
        const vuc * a = PA->v[ch];
        const vuc * y = PB->v[ch];
        if (ch + 1 < nch) {
            // each chunk's packed panel spans two 128B lines; touch both
            __builtin_prefetch(PA->v[ch + 1], 0, 3);
            __builtin_prefetch((const char *)PA->v[ch + 1] + 128, 0, 3);
            __builtin_prefetch(PB->v[ch + 1], 0, 3);
            __builtin_prefetch((const char *)PB->v[ch + 1] + 128, 0, 3);
        }
        __vector_quad acc[2][2];
        for (int g = 0; g < 2; g++)
            for (int cgi = 0; cgi < 2; cgi++)
                __builtin_mma_xxsetaccz(&acc[g][cgi]);
        for (int x = 0; x < 8; x++) {
#ifdef IQGRID_LXVP
            vuc wv[2], yv[2];
            load_pair(a + 2*x, wv);
            load_pair(y + 2*x, yv);
            const vuc w0 = wv[0], w1 = wv[1];
            const vuc y0 = yv[0], y1 = yv[1];
#else
            const vuc w0 = a[2*x], w1 = a[2*x + 1];
            const vuc y0 = y[2*x], y1 = y[2*x + 1];
#endif
            __builtin_mma_xvi8ger4pp(&acc[0][0], w0, y0);
            __builtin_mma_xvi8ger4pp(&acc[0][1], w0, y1);
            __builtin_mma_xvi8ger4pp(&acc[1][0], w1, y0);
            __builtin_mma_xvi8ger4pp(&acc[1][1], w1, y1);
        }
        for (int g = 0; g < 2; g++) {
            const vfl sA   = PA->sA  [ch][g];
            const vfl C128 = PA->C128[ch][g];
            for (int cgi = 0; cgi < 2; cgi++) {
                vsi rowsP[4];
                __builtin_mma_disassemble_acc(rowsP, &acc[g][cgi]);
                const vfl dB = PB->dB[ch][cgi];
                for (int r = 0; r < 4; r++) {
                    vfl t = vec_sub(vec_ctf(rowsP[r],0), vec_splats(C128[r]));
                    vfl sc = vec_mul(vec_splats(sA[r]), dB);
                    fin[4*g + r][cgi] = vec_madd(t, sc, fin[4*g + r][cgi]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Accumulator ping-pong variant (-DIQGRID_PINGPONG), 32-deep kernel.
//
// Microarchitectural rationale (Power10 MMA): the eight accumulators
// are physically resident in the MMA engine; xxmfacc drains engine
// state to the VRF and serializes against outstanding GERs on that
// accumulator. The default kernel therefore idles the engine through
// every per-chunk fixup. This variant runs two 4-accumulator sets in
// alternation: chunk c+1's GER stream is issued on set B before set A
// is drained, so the drain + VSU fixup of chunk c overlap live GER
// execution. Cost: all 8 accumulators alias VSRs 0-31, raising spill
// pressure (visible in the static count); benefit: engine-idle removal
// (invisible to any static count). Only silicon can arbitrate --
// this is hardware experiment #1 in docs/BENCHMARKS-QEMU.md.

static inline void grid_pp_compute(const vuc * a, const vuc * y,
                                   __vector_quad acc[2][2]) {
    for (int g = 0; g < 2; g++)
        for (int cgi = 0; cgi < 2; cgi++)
            __builtin_mma_xxsetaccz(&acc[g][cgi]);
    for (int x = 0; x < 8; x++) {
#ifdef IQGRID_LXVP
        vuc wv[2], yv[2];
        load_pair(a + 2*x, wv);
        load_pair(y + 2*x, yv);
        const vuc w0 = wv[0], w1 = wv[1];
        const vuc y0 = yv[0], y1 = yv[1];
#else
        const vuc w0 = a[2*x], w1 = a[2*x + 1];
        const vuc y0 = y[2*x], y1 = y[2*x + 1];
#endif
        __builtin_mma_xvi8ger4pp(&acc[0][0], w0, y0);
        __builtin_mma_xvi8ger4pp(&acc[0][1], w0, y1);
        __builtin_mma_xvi8ger4pp(&acc[1][0], w1, y0);
        __builtin_mma_xvi8ger4pp(&acc[1][1], w1, y1);
    }
}

static inline void grid_pp_fixup(const agrid_t * PA, const bgrid_t * PB,
                                 int64_t ch, __vector_quad acc[2][2],
                                 vfl fin[MR][2]) {
    for (int g = 0; g < 2; g++) {
        const vfl sA   = PA->sA  [ch][g];
        const vfl C128 = PA->C128[ch][g];
        for (int cgi = 0; cgi < 2; cgi++) {
            vsi rowsP[4];
            __builtin_mma_disassemble_acc(rowsP, &acc[g][cgi]);
            const vfl dB = PB->dB[ch][cgi];
            for (int r = 0; r < 4; r++) {
                vfl t = vec_sub(vec_ctf(rowsP[r],0), vec_splats(C128[r]));
                vfl sc = vec_mul(vec_splats(sA[r]), dB);
                fin[4*g + r][cgi] = vec_madd(t, sc, fin[4*g + r][cgi]);
            }
        }
    }
}

__attribute__((unused)) static void kernel_grid_8x8_pp(const agrid_t * PA, const bgrid_t * PB,
                               int64_t nch, vfl fin[MR][2]) {
    if (nch <= 0) return;
    __vector_quad accA[2][2], accB[2][2];
    grid_pp_compute(PA->v[0], PB->v[0], accA);
    int64_t ch = 1;
    int aLive = 1;                       // which set holds chunk ch-1
    for (; ch < nch; ch++) {
        if (aLive) { grid_pp_compute(PA->v[ch], PB->v[ch], accB);
                     grid_pp_fixup(PA, PB, ch - 1, accA, fin); }
        else       { grid_pp_compute(PA->v[ch], PB->v[ch], accA);
                     grid_pp_fixup(PA, PB, ch - 1, accB, fin); }
        aLive ^= 1;
    }
    grid_pp_fixup(PA, PB, nch - 1, aLive ? accA : accB, fin);
}

extern "C" void grid_gemm_packed(int64_t m, int64_t n, int64_t k,
                                 const void * packedA, const void * packedB,
                                 float * C, int64_t ldc, int ith, int nth) {
    const agrid_t * PA = (const agrid_t *)packedA;
    const bgrid_t * PB = (const bgrid_t *)packedB;
    const int64_t kb = k/32, ns = sl(k), mt = rt(m), njt = ct(n);
    const int64_t tpt = (mt + nth - 1) / nth;
    const int64_t t0 = ith*tpt, t1 = (ith+1)*tpt < mt ? (ith+1)*tpt : mt;
    vfl fin[MR][2];
    for (int64_t it = t0; it < t1; it++) {
        const int64_t i = it*MR;
        const int64_t rows = (m - i) < MR ? (m - i) : MR;
        for (int64_t jt = 0; jt < njt; jt++) {
            for (int r = 0; r < MR; r++) fin[r][0] = fin[r][1] = vec_splats(0.0f);
            for (int64_t s = 0; s < ns; s++) {
                const int64_t b0 = s*KC_CH;
                const int64_t nch = (kb - b0) < KC_CH ? (kb - b0) : KC_CH;
#ifdef IQGRID_PINGPONG
                kernel_grid_8x8_pp(&PA[it*ns + s], &PB[jt*ns + s], nch, fin);
#else
                kernel_grid_8x8(&PA[it*ns + s], &PB[jt*ns + s], nch, fin);
#endif
            }
            const int64_t j0 = jt*NR;
            const int64_t cols = (n - j0) < NR ? (n - j0) : NR;
            for (int64_t r = 0; r < rows; r++)
                for (int64_t cj = 0; cj < cols; cj++)
                    C[(i + r) + (j0 + cj)*ldc] = fin[r][cj >> 2][cj & 3];
        }
    }
}


// ---- 16-deep chunk variant for per-16-scale formats ----

#define KC_CH16 (KC_ELEMS / 16)

typedef struct {
    vuc v[KC_CH16][8];                    // 4 depth-steps x 2 rowgroups
    vfl sA  [KC_CH16][2];
    vfl C128[KC_CH16][2];
} agrid16_t;

typedef struct {
    vuc v[KC_CH16][8];                    // 4 depth-steps x 2 colgroups
    vfl dB[KC_CH16][2];
} bgrid16_t;

extern "C" size_t grid16_apack_size(int64_t m, int64_t k) {
    return (((size_t)(rt(m) * sl(k)) * sizeof(agrid16_t)) + 63) & ~(size_t)63;
}
extern "C" size_t grid16_bpack_size(int64_t n, int64_t k) {
    return (((size_t)(ct(n) * sl(k)) * sizeof(bgrid16_t)) + 63) & ~(size_t)63;
}

static void grid16_place_chunk(agrid16_t * T, int64_t ch,
                               const vsc w[MR], const float scale[MR]) {
    for (int g = 0; g < 2; g++) {
        float W[4];
        vui rows4[4];
        for (int r = 0; r < 4; r++) rows4[r] = (vui)w[4*g + r];
        mma_transpose4(rows4, &T->v[ch][g], 2);
        for (int r = 0; r < 4; r++) {
            vsi z = vec_splats(0);
            vsi sm = vec_sum4s(w[4*g + r], z);
            W[r] = (float)hsum(sm);
        }
        T->sA  [ch][g] = (vfl){ scale[4*g], scale[4*g+1], scale[4*g+2], scale[4*g+3] };
        T->C128[ch][g] = (vfl){ 128.0f*W[0], 128.0f*W[1],
                                128.0f*W[2], 128.0f*W[3] };  // exact int; see field FAIL 2026-07-21
    }
}

template <typename BLK, void (*DEC)(const BLK *, int8_t[QK_K], float[16])>
static void repack_grid16(const BLK * A, int64_t lda, int64_t m, int64_t k, agrid16_t * P) {
    const int64_t nsb = k/QK_K, ns = sl(k);
    for (int64_t it = 0; it < rt(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        agrid16_t * T = &P[it*ns + s];
        const int64_t sb0 = (s*KC_CH)/8;
        const int64_t nsl = (nsb - sb0) < KC_CH/8 ? (nsb - sb0) : KC_CH/8;
        for (int64_t sb = 0; sb < nsl; sb++) {
            int8_t code[MR][QK_K]; float sc[MR][16];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                DEC(&A[rr*lda + sb0 + sb], code[r], sc[r]);
            }
            for (int c = 0; c < 16; c++) {
                vsc w[MR]; float scale[MR];
                for (int r = 0; r < MR; r++) {
                    memcpy(&w[r], code[r] + 16*c, 16);
                    scale[r] = sc[r][c];
                }
                grid16_place_chunk(T, 16*sb + c, w, scale);
            }
        }
    }
}

extern "C" void grid16_repack_iq2_xs(const block_iq2_xs_ppc * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_grid16<block_iq2_xs_ppc, dec16_iq2_xs>(A, lda, m, k, (agrid16_t *)p); }
extern "C" void grid16_repack_iq2_s(const block_iq2_s_ppc * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_grid16<block_iq2_s_ppc, dec16_iq2_s>(A, lda, m, k, (agrid16_t *)p); }
extern "C" void grid16_repack_iq1_m(const block_iq1_m_ppc * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_grid16<block_iq1_m_ppc, dec16_iq1_m>(A, lda, m, k, (agrid16_t *)p); }

extern "C" void grid16_repack_nvfp4(const block_nvfp4_ppc * A, int64_t lda,
                                    int64_t m, int64_t k, void * p) {
    agrid16_t * P = (agrid16_t *)p;
    const int64_t nb = k/QK_NVFP4, ns = sl(k);
    for (int64_t it = 0; it < rt(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        agrid16_t * T = &P[it*ns + s];
        const int64_t b0 = (s*KC_CH16)/4;               // 64-blocks per slab start
        const int64_t nbl = (nb - b0) < KC_CH16/4 ? (nb - b0) : KC_CH16/4;
        for (int64_t b = 0; b < nbl; b++) {
            int8_t code[MR][QK_NVFP4]; float sc[MR][4];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                dec_nvfp4(&A[rr*lda + b0 + b], code[r], sc[r]);
            }
            for (int c = 0; c < 4; c++) {
                vsc w[MR]; float scale[MR];
                for (int r = 0; r < MR; r++) {
                    memcpy(&w[r], code[r] + 16*c, 16);
                    scale[r] = sc[r][c];
                }
                grid16_place_chunk(T, 4*b + c, w, scale);
            }
        }
    }
}

// q8_0 activations for the 16-deep framework: each 32-block feeds two
// chunks; its dB is replicated to both.
extern "C" void grid16_pack_b_q8_0(const block_q8_0_ppc * B, int64_t ldb,
                                   int64_t n, int64_t k, void * packed) {
    bgrid16_t * P = (bgrid16_t *)packed;
    const int64_t kb = k/32, ns = sl(k);
    const vuc flip = vec_splats((unsigned char)0x80);
    for (int64_t jt = 0; jt < ct(n); jt++)
    for (int64_t s = 0; s < ns; s++) {
        bgrid16_t * T = &P[jt*ns + s];
        const int64_t b0 = (s*KC_CH16)/2;
        const int64_t nbl = (kb - b0) < KC_CH16/2 ? (kb - b0) : KC_CH16/2;
        for (int64_t b = 0; b < nbl; b++) {
            const block_q8_0_ppc * yb[NR]; float dB[NR];
            for (int j = 0; j < NR; j++) {
                int64_t jj = jt*NR + j; if (jj >= n) jj = n - 1;
                yb[j] = &B[jj*ldb + b0 + b];
                dB[j] = fp16_to_fp32(yb[j]->d);
            }
            for (int h = 0; h < 2; h++) {
                const int64_t ch = 2*b + h;
                T->dB[ch][0] = (vfl){ dB[0], dB[1], dB[2], dB[3] };
                T->dB[ch][1] = (vfl){ dB[4], dB[5], dB[6], dB[7] };
                vui rows4[4];
                for (int a = 0; a < 2; a++) {
                    for (int j = 0; j < 4; j++)
                        rows4[j] = (vui)vec_xor(
                            load16u((const uint8_t *)(yb[4*a + j]->qs) + 16*h), flip);
                    mma_transpose4(rows4, &T->v[ch][a], 2);
                }
            }
        }
    }
}

extern "C" void grid16_pack_b_q8_K(const block_q8_K_ppc * B, int64_t ldb,
                                   int64_t n, int64_t k, void * packed) {
    bgrid16_t * P = (bgrid16_t *)packed;
    const int64_t nsb = k/QK_K, ns = sl(k);
    const vuc flip = vec_splats((unsigned char)0x80);
    for (int64_t jt = 0; jt < ct(n); jt++)
    for (int64_t s = 0; s < ns; s++) {
        bgrid16_t * T = &P[jt*ns + s];
        const int64_t sb0 = (s*KC_CH)/8;
        const int64_t nsl = (nsb - sb0) < KC_CH/8 ? (nsb - sb0) : KC_CH/8;
        for (int64_t sb = 0; sb < nsl; sb++) {
            const block_q8_K_ppc * yb[NR]; float dB[NR];
            for (int j = 0; j < NR; j++) {
                int64_t jj = jt*NR + j; if (jj >= n) jj = n - 1;
                yb[j] = &B[jj*ldb + sb0 + sb];
                dB[j] = yb[j]->d;
            }
            for (int c = 0; c < 16; c++) {
                const int64_t ch = 16*sb + c;
                T->dB[ch][0] = (vfl){ dB[0], dB[1], dB[2], dB[3] };
                T->dB[ch][1] = (vfl){ dB[4], dB[5], dB[6], dB[7] };
                vui rows4[4];
                for (int a = 0; a < 2; a++) {
                    for (int j = 0; j < 4; j++)
                        rows4[j] = (vui)vec_xor(
                            load16u((const uint8_t *)(yb[4*a + j]->qs) + 16*c), flip);
                    mma_transpose4(rows4, &T->v[ch][a], 2);
                }
            }
        }
    }
}

static void kernel_grid16_8x8(const agrid16_t * PA, const bgrid16_t * PB,
                              int64_t nch, vfl fin[MR][2]) {
    for (int64_t ch = 0; ch < nch; ch++) {
        const vuc * a = PA->v[ch];
        const vuc * y = PB->v[ch];
        if (ch + 1 < nch) {
            // each chunk's packed panel spans two 128B lines; touch both
            __builtin_prefetch(PA->v[ch + 1], 0, 3);
            __builtin_prefetch((const char *)PA->v[ch + 1] + 128, 0, 3);
            __builtin_prefetch(PB->v[ch + 1], 0, 3);
            __builtin_prefetch((const char *)PB->v[ch + 1] + 128, 0, 3);
        }
        __vector_quad acc[2][2];
        for (int g = 0; g < 2; g++)
            for (int cgi = 0; cgi < 2; cgi++)
                __builtin_mma_xxsetaccz(&acc[g][cgi]);
        for (int x = 0; x < 4; x++) {
            const vuc w0 = a[2*x], w1 = a[2*x + 1];
            const vuc y0 = y[2*x], y1 = y[2*x + 1];
            __builtin_mma_xvi8ger4pp(&acc[0][0], w0, y0);
            __builtin_mma_xvi8ger4pp(&acc[0][1], w0, y1);
            __builtin_mma_xvi8ger4pp(&acc[1][0], w1, y0);
            __builtin_mma_xvi8ger4pp(&acc[1][1], w1, y1);
        }
        for (int g = 0; g < 2; g++) {
            const vfl sA   = PA->sA  [ch][g];
            const vfl C128 = PA->C128[ch][g];
            for (int cgi = 0; cgi < 2; cgi++) {
                vsi rowsP[4];
                __builtin_mma_disassemble_acc(rowsP, &acc[g][cgi]);
                const vfl dB = PB->dB[ch][cgi];
                for (int r = 0; r < 4; r++) {
                    vfl t = vec_sub(vec_ctf(rowsP[r],0), vec_splats(C128[r]));
                    vfl sc = vec_mul(vec_splats(sA[r]), dB);
                    fin[4*g + r][cgi] = vec_madd(t, sc, fin[4*g + r][cgi]);
                }
            }
        }
    }
}

extern "C" void grid16_gemm_packed(int64_t m, int64_t n, int64_t k,
                                   const void * packedA, const void * packedB,
                                   float * C, int64_t ldc, int ith, int nth) {
    const agrid16_t * PA = (const agrid16_t *)packedA;
    const bgrid16_t * PB = (const bgrid16_t *)packedB;
    const int64_t kc16 = k/16, ns = sl(k), mt = rt(m), njt = ct(n);
    const int64_t tpt = (mt + nth - 1) / nth;
    const int64_t t0 = ith*tpt, t1 = (ith+1)*tpt < mt ? (ith+1)*tpt : mt;
    vfl fin[MR][2];
    for (int64_t it = t0; it < t1; it++) {
        const int64_t i = it*MR;
        const int64_t rows = (m - i) < MR ? (m - i) : MR;
        for (int64_t jt = 0; jt < njt; jt++) {
            for (int r = 0; r < MR; r++) fin[r][0] = fin[r][1] = vec_splats(0.0f);
            for (int64_t s = 0; s < ns; s++) {
                const int64_t c0 = s*KC_CH16;
                const int64_t nch = (kc16 - c0) < KC_CH16 ? (kc16 - c0) : KC_CH16;
                kernel_grid16_8x8(&PA[it*ns + s], &PB[jt*ns + s], nch, fin);
            }
            const int64_t j0 = jt*NR;
            const int64_t cols = (n - j0) < NR ? (n - j0) : NR;
            for (int64_t r = 0; r < rows; r++)
                for (int64_t cj = 0; cj < cols; cj++)
                    C[(i + r) + (j0 + cj)*ldc] = fin[r][cj >> 2][cj & 3];
        }
    }
}


#define GRID_ONESHOT_C(NAME, BLKA, REPACK, GEMM, ASZ, BSZ, PACKB, VARIANT)     \
extern "C" void NAME(int64_t m, int64_t n, int64_t k,                          \
        const void * Av, int64_t lda, const void * Bv, int64_t ldb,            \
        float * C, int64_t ldc, int ith, int nth) {                            \
    const BLKA * A = (const BLKA *)Av;                                         \
    const block_q8_K_ppc * B = (const block_q8_K_ppc *)Bv;                     \
    int fresh = 0;                                                             \
    void * PA = ppc_apack_cache_acquire(Av, m, k, VARIANT, ASZ(m, k), &fresh); \
    if (PA) {                                                                  \
        if (fresh) { REPACK(A, lda, m, k, (void *)PA);                    \
                     ppc_apack_cache_publish(Av, m, k, VARIANT); }             \
        const int64_t njt = (n + NR - 1) / NR;                                 \
        if (njt < nth) {                                               \
            /* n too small to feed every thread by columns (worst      \
               case n == 1 generation: one column, nth-1 idle threads  \
               while scalar row-partitions). Row-partition with the    \
               cached pack instead; the full activation pack is tiny   \
               at these n. Field regression, Q4_K tg32, 2026-07-21. */ \
            void * PBs = aligned_alloc(64, BSZ(n, k));                 \
            if (!PBs) { GGML_ABORT("ppc-mma: pack alloc failed"); }    \
            PACKB(B, ldb, n, k, PBs);                                  \
            GEMM(m, n, k, PA, PBs, C, ldc, ith, nth);                  \
            free(PBs);                                                 \
        } else {                                                       \
        const int64_t jpt = (njt + nth - 1) / nth;                             \
        const int64_t jt0 = (int64_t)ith*jpt;                                  \
        const int64_t jt1 = (ith+1)*jpt < njt ? (ith+1)*jpt : njt;             \
        if (jt0 < jt1) {                                                       \
            const int64_t j0 = jt0*NR;                                         \
            const int64_t nc = (n - j0) < (jt1 - jt0)*NR ? (n - j0)            \
                                                         : (jt1 - jt0)*NR;    \
            void * PBl = aligned_alloc(64, BSZ(nc, k));                      \
            if (!PBl) { GGML_ABORT("ppc-mma: pack alloc failed"); }           \
            PACKB(B + j0*ldb, ldb, nc, k, PBl);                              \
            GEMM(m, nc, k, PA, PBl, C + j0*ldc, ldc, 0, 1);                                                         \
            free(PBl);                                                         \
        }                                                                      \
        }                                                              \
    } else {                                                                   \
        void * PB = aligned_alloc(64, BSZ(n, k));                            \
        if (!PB) { GGML_ABORT("ppc-mma: pack alloc failed"); }                \
        PACKB(B, ldb, n, k, PB);                                             \
        void * PT = aligned_alloc(64, ASZ(MR, k));                           \
        if (!PT) { GGML_ABORT("ppc-mma: pack alloc failed"); }                \
        const int64_t mt = (m + MR - 1) / MR;                                  \
        const int64_t tpt = (mt + nth - 1) / nth;                              \
        for (int64_t it = ith*tpt; it < (ith+1)*tpt && it < mt; it++) {        \
            const int64_t i = it*MR;                                           \
            const int64_t rows = (m - i) < MR ? (m - i) : MR;                  \
            REPACK(A + i*lda, lda, rows, k, PT);                          \
            GEMM(rows, n, k, PT, PB, C + i, ldc, 0, 1);                                                          \
        }                                                                      \
        free(PT);                                                              \
        free(PB);                                                              \
    }                                                                          \
}
GRID_ONESHOT_C(gemm_tq2_0_q8_K_ppc,   block_tq2_0_ppc,   grid_repack_tq2_0,   grid_gemm_packed,   grid_apack_size,   grid_bpack_size,   grid_pack_b_q8_K,   10)
GRID_ONESHOT_C(gemm_tq1_0_q8_K_ppc,   block_tq1_0_ppc,   grid_repack_tq1_0,   grid_gemm_packed,   grid_apack_size,   grid_bpack_size,   grid_pack_b_q8_K,   11)
GRID_ONESHOT_C(gemm_iq2_xxs_q8_K_ppc, block_iq2_xxs_ppc, grid_repack_iq2_xxs, grid_gemm_packed,   grid_apack_size,   grid_bpack_size,   grid_pack_b_q8_K,   12)
GRID_ONESHOT_C(gemm_iq3_xxs_q8_K_ppc, block_iq3_xxs_ppc, grid_repack_iq3_xxs, grid_gemm_packed,   grid_apack_size,   grid_bpack_size,   grid_pack_b_q8_K,   13)
GRID_ONESHOT_C(gemm_iq3_s_q8_K_ppc,   block_iq3_s_ppc,   grid_repack_iq3_s,   grid_gemm_packed,   grid_apack_size,   grid_bpack_size,   grid_pack_b_q8_K,   14)
GRID_ONESHOT_C(gemm_iq1_s_q8_K_ppc,   block_iq1_s_ppc,   grid_repack_iq1_s,   grid_gemm_packed,   grid_apack_size,   grid_bpack_size,   grid_pack_b_q8_K,   15)
GRID_ONESHOT_C(gemm_iq2_xs_q8_K_ppc,  block_iq2_xs_ppc,  grid16_repack_iq2_xs, grid16_gemm_packed, grid16_apack_size, grid16_bpack_size, grid16_pack_b_q8_K, 16)
GRID_ONESHOT_C(gemm_iq2_s_q8_K_ppc,   block_iq2_s_ppc,   grid16_repack_iq2_s,  grid16_gemm_packed, grid16_apack_size, grid16_bpack_size, grid16_pack_b_q8_K, 17)
GRID_ONESHOT_C(gemm_iq1_m_q8_K_ppc,   block_iq1_m_ppc,   grid16_repack_iq1_m,  grid16_gemm_packed, grid16_apack_size, grid16_bpack_size, grid16_pack_b_q8_K, 18)

extern "C" void gemm_nvfp4_q8_0_ppc(int64_t m, int64_t n, int64_t k,
        const void * Av, int64_t lda, const void * Bv, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth) {
    const block_nvfp4_ppc * A = (const block_nvfp4_ppc *)Av;
    const block_q8_0_ppc * B = (const block_q8_0_ppc *)Bv;
    int fresh = 0;
    void * PA = ppc_apack_cache_acquire(Av, m, k, 19, grid16_apack_size(m, k), &fresh);
    if (PA) {
        if (fresh) { grid16_repack_nvfp4(A, lda, m, k, (void *)PA);
                     ppc_apack_cache_publish(Av, m, k, 19); }
        const int64_t njt = (n + NR - 1) / NR;
        if (njt < nth) {
            /* n too small to feed every thread by columns (worst
               case n == 1 generation: one column, nth-1 idle threads
               while scalar row-partitions). Row-partition with the
               cached pack instead; the full activation pack is tiny
               at these n. Field regression, Q4_K tg32, 2026-07-21. */
            void * PBs = aligned_alloc(64, grid16_bpack_size(n, k));
            if (!PBs) { GGML_ABORT("ppc-mma: pack alloc failed"); }
            grid16_pack_b_q8_0(B, ldb, n, k, PBs);
            grid16_gemm_packed(m, n, k, PA, PBs, C, ldc, ith, nth);
            free(PBs);
        } else {
        const int64_t jpt = (njt + nth - 1) / nth;
        const int64_t jt0 = (int64_t)ith*jpt;
        const int64_t jt1 = (ith+1)*jpt < njt ? (ith+1)*jpt : njt;
        if (jt0 < jt1) {
            const int64_t j0 = jt0*NR;
            const int64_t nc = (n - j0) < (jt1 - jt0)*NR ? (n - j0) : (jt1 - jt0)*NR;
            void * PBl = aligned_alloc(64, grid16_bpack_size(nc, k));
            if (!PBl) { GGML_ABORT("ppc-mma: pack alloc failed"); }
            grid16_pack_b_q8_0(B + j0*ldb, ldb, nc, k, PBl);
            grid16_gemm_packed(m, nc, k, PA, PBl, C + j0*ldc, ldc, 0, 1);
            free(PBl);
        }
        }
    } else {
        void * PB = aligned_alloc(64, grid16_bpack_size(n, k));
        if (!PB) { GGML_ABORT("ppc-mma: pack alloc failed"); }
        grid16_pack_b_q8_0(B, ldb, n, k, PB);
        void * PT = aligned_alloc(64, grid16_apack_size(MR, k));
        if (!PT) { GGML_ABORT("ppc-mma: pack alloc failed"); }
        const int64_t mt = (m + MR - 1) / MR;
        const int64_t tpt = (mt + nth - 1) / nth;
        for (int64_t it = ith*tpt; it < (ith+1)*tpt && it < mt; it++) {
            const int64_t i = it*MR;
            const int64_t rows = (m - i) < MR ? (m - i) : MR;
            grid16_repack_nvfp4(A + i*lda, lda, rows, k, PT);
            grid16_gemm_packed(rows, n, k, PT, PB, C + i, ldc, 0, 1);
        }
        free(PT);
        free(PB);
    }
}

#endif // __MMA__


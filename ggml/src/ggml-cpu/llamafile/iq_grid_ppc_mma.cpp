// iq_grid_ppc_mma.cpp - imported from github.com/mavin2009/ppc-mma-kernels
// (standalone-verified under qemu -cpu power10). Grid-codebook and
// ternary formats on POWER10/POWER11 MMA via decode-at-repack + the
// shared signed 8x8 kernel. Uses a local copy of the grid tables
// (iq_grids_ppc.h) to avoid depending on GGML_COMMON_IMPL linkage.

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
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K_ppc;

static_assert(sizeof(block_tq1_0_ppc)   == 2 + 4 + 48, "bad tq1_0");
static_assert(sizeof(block_tq2_0_ppc)   == 2 + 64, "bad tq2_0");
static_assert(sizeof(block_iq2_xxs_ppc) == 2 + QK_K/4, "bad iq2_xxs");
static_assert(sizeof(block_iq3_xxs_ppc) == 2 + 3*QK_K/8, "bad iq3_xxs");
static_assert(sizeof(block_iq3_s_ppc)   == 2 + 13*(QK_K/32) + QK_K/64, "bad iq3_s");
static_assert(sizeof(block_iq1_s_ppc)   == 2 + QK_K/8 + QK_K/16, "bad iq1_s");

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

#if defined(__MMA__) && defined(__powerpc64__)

typedef vector unsigned char  vuc;
typedef vector signed char    vsc;
typedef vector unsigned int   vui;
typedef vector signed int     vsi;
typedef vector float          vfl;

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
        T->C128[ch][g] = (vfl){ 128.0f*W[0]*scale[4*g],   128.0f*W[1]*scale[4*g+1],
                                128.0f*W[2]*scale[4*g+2], 128.0f*W[3]*scale[4*g+3] };
    }
}

static inline int64_t rt(int64_t m) { return (m + MR - 1) / MR; }
static inline int64_t ct(int64_t n) { return (n + NR - 1) / NR; }
static inline int64_t sl(int64_t k) { return (k/32 + KC_CH - 1) / KC_CH; }

extern "C" size_t grid_apack_size(int64_t m, int64_t k) {
    return (size_t)(rt(m) * sl(k)) * sizeof(agrid_t);
}
extern "C" size_t grid_bpack_size(int64_t n, int64_t k) {
    return (size_t)(ct(n) * sl(k)) * sizeof(bgrid_t);
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
                                vec_xl(32*ib + 16*h, (const unsigned char *)yb[4*a + j]->qs), flip);
                        mma_transpose4(rows4, &T->v[ch][8*h + a], 2);
                    }
            }
        }
    }
}

static void kernel_grid_8x8(const agrid_t * PA, const bgrid_t * PB,
                            int64_t nch, vfl fin[MR][2]) {
    for (int64_t ch = 0; ch < nch; ch++) {
        const vuc * a = PA->v[ch];
        const vuc * y = PB->v[ch];
        if (ch + 1 < nch) {
            __builtin_prefetch(PA->v[ch + 1], 0, 3);
            __builtin_prefetch(PB->v[ch + 1], 0, 3);
        }
        __vector_quad acc[2][2];
        for (int g = 0; g < 2; g++)
            for (int cgi = 0; cgi < 2; cgi++)
                __builtin_mma_xxsetaccz(&acc[g][cgi]);
        for (int x = 0; x < 8; x++) {
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
                    vfl t = vec_msub(vec_ctf(rowsP[r],0),
                                     vec_splats(sA[r]), vec_splats(C128[r]));
                    fin[4*g + r][cgi] = vec_madd(t, dB, fin[4*g + r][cgi]);
                }
            }
        }
    }
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
                kernel_grid_8x8(&PA[it*ns + s], &PB[jt*ns + s], nch, fin);
            }
            const int64_t j0 = jt*NR;
            const int64_t cols = (n - j0) < NR ? (n - j0) : NR;
            for (int64_t r = 0; r < rows; r++)
                for (int64_t cj = 0; cj < cols; cj++)
                    C[(i + r) + (j0 + cj)*ldc] = fin[r][cj >> 2][cj & 3];
        }
    }
}


#define GRID_ONESHOT(NAME, BLKA, REPACK)                                       \
extern "C" void NAME(int64_t m, int64_t n, int64_t k,                          \
        const void * Av, int64_t lda, const void * Bv, int64_t ldb,            \
        float * C, int64_t ldc, int ith, int nth) {                            \
    const BLKA * A = (const BLKA *)Av;                                         \
    const block_q8_K_ppc * B = (const block_q8_K_ppc *)Bv;                     \
    void * PA = aligned_alloc(64, grid_apack_size(MR, k));                     \
    void * PB = aligned_alloc(64, grid_bpack_size(n, k));                      \
    if (!PA || !PB) { free(PA); free(PB); return; }                            \
    grid_pack_b_q8_K(B, ldb, n, k, PB);                                        \
    const int64_t mt = (m + MR - 1) / MR;                                      \
    const int64_t tpt = (mt + nth - 1) / nth;                                  \
    for (int64_t it = ith*tpt; it < (ith+1)*tpt && it < mt; it++) {            \
        const int64_t i = it*MR;                                               \
        const int64_t rows = (m - i) < MR ? (m - i) : MR;                      \
        REPACK(A + i*lda, lda, rows, k, PA);                                   \
        grid_gemm_packed(rows, n, k, PA, PB, C + i, ldc, 0, 1);                \
    }                                                                          \
    free(PA); free(PB);                                                        \
}
GRID_ONESHOT(gemm_tq2_0_q8_K_ppc,   block_tq2_0_ppc,   grid_repack_tq2_0)
GRID_ONESHOT(gemm_tq1_0_q8_K_ppc,   block_tq1_0_ppc,   grid_repack_tq1_0)
GRID_ONESHOT(gemm_iq2_xxs_q8_K_ppc, block_iq2_xxs_ppc, grid_repack_iq2_xxs)
GRID_ONESHOT(gemm_iq3_xxs_q8_K_ppc, block_iq3_xxs_ppc, grid_repack_iq3_xxs)
GRID_ONESHOT(gemm_iq3_s_q8_K_ppc,   block_iq3_s_ppc,   grid_repack_iq3_s)
GRID_ONESHOT(gemm_iq1_s_q8_K_ppc,   block_iq1_s_ppc,   grid_repack_iq1_s)

#endif // __MMA__


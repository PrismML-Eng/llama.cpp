// iq4_ppc_mma.cpp - imported from github.com/mavin2009/ppc-mma-kernels
// (standalone-verified vs exact double references under qemu -cpu power10).
// IQ4_NL x Q8_0, IQ4_XS x Q8_K, MXFP4 x Q8_0 on POWER10/POWER11 MMA.

#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "kquants_ppc_mma.h"

#include <altivec.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>









static_assert(sizeof(block_iq4_nl) == sizeof(ggml_half) + QK4_NL/2, "bad iq4_nl");
static_assert(sizeof(block_mxfp4) == 1 + QK_MXFP4/2, "bad mxfp4");
static_assert(sizeof(block_iq4_xs) == sizeof(ggml_half) + 2 + QK_K/64 + QK_K/2, "bad iq4_xs");

static const int8_t iq4_kvalues_local[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};
// MXFP4 e2m1 values, pre-doubled to integers (the E8M0 "half" scale
// supplies the /2): {0,1,2,3,4,6,8,12} with sign.
static const int8_t kvalues_mxfp4_[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};
#include <cmath>
static inline float e8m0_half_to_fp32(uint8_t e) { return ldexpf(1.0f, (int)e - 128); }


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


#define KC_ELEMS  2048                    // K slab
#define KC_CH     (KC_ELEMS / 32)         // 32-element chunks per slab
#define MR        8
#define NR        8

// packed weights: per chunk, 8 depth-steps x 2 rowgroups of signed values
typedef struct {
    vuc v[KC_CH][16];
    vfl sA  [KC_CH][2];                   // per-row scale (dA or d*(ls-32))
    vfl C128[KC_CH][2];                   // 128 * W * scale, pre-folded
} aiq4_t;

// packed activations: per chunk, 8 depth-steps x 2 colgroups, flipped
typedef struct {
    vuc v[KC_CH][16];
    vfl dB[KC_CH][2];                     // per-chunk column scales
} biq4_t;

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

// 32 signed codebook values from 16 packed bytes; elements j / j+16
// come from qs[j] low / high nibble.  One vec_perm each.
static inline void iq4_lookup32t(const uint8_t * qs, const int8_t * table, vsc v[2]) {
    vsc tbl; memcpy(&tbl, table, 16);
    vuc raw = load16u((const uint8_t *)(qs) + (0));
    vuc lo  = vec_and(raw, vec_splats((unsigned char)0xF));
    vuc hi  = vec_sr(raw, vec_splats((unsigned char)4));
    v[0] = vec_perm(tbl, tbl, lo);
    v[1] = vec_perm(tbl, tbl, hi);
}

static inline int hsum(vsi s) { return s[0] + s[1] + s[2] + s[3]; }

// pack one chunk's 8 weight rows into T at chunk ch; w[r] = row values,
// scale[r] = per-row scale for this chunk.
static void iq4_place_chunk(aiq4_t * T, int64_t ch,
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

static inline int64_t rt8(int64_t m) { return (m + MR - 1) / MR; }
static inline int64_t ct8(int64_t n) { return (n + NR - 1) / NR; }
static inline int64_t sl32(int64_t k) { return (k/32 + KC_CH - 1) / KC_CH; }

extern "C" size_t iq4_apack_size(int64_t m, int64_t k) {
    return (((size_t)(rt8(m) * sl32(k)) * sizeof(aiq4_t)) + 63) & ~(size_t)63;
}
extern "C" size_t iq4_bpack_size(int64_t n, int64_t k) {
    return (((size_t)(ct8(n) * sl32(k)) * sizeof(biq4_t)) + 63) & ~(size_t)63;
}

// ---- weight repack: IQ4_NL ----
extern "C" void iq4nl_repack_a(const block_iq4_nl * A, int64_t lda,
                               int64_t m, int64_t k, void * packed) {
    aiq4_t * P = (aiq4_t *)packed;
    const int64_t kb = k/32, ns = sl32(k);
    for (int64_t it = 0; it < rt8(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        aiq4_t * T = &P[it*ns + s];
        const int64_t b0 = s*KC_CH;
        const int64_t nb = (kb - b0) < KC_CH ? (kb - b0) : KC_CH;
        for (int64_t b = 0; b < nb; b++) {
            vsc w[MR][2]; float sc[MR];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                const block_iq4_nl * bp = &A[rr*lda + b0 + b];
                sc[r] = GGML_FP16_TO_FP32(bp->d);
                iq4_lookup32t(bp->qs, iq4_kvalues_local, w[r]);
            }
            iq4_place_chunk(T, b, w, sc);
        }
    }
}

// ---- weight repack: MXFP4 (same shape as IQ4_NL, different table+scale) ----
extern "C" void mxfp4_repack_a(const block_mxfp4 * A, int64_t lda,
                               int64_t m, int64_t k, void * packed) {
    aiq4_t * P = (aiq4_t *)packed;
    const int64_t kb = k/32, ns = sl32(k);
    for (int64_t it = 0; it < rt8(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        aiq4_t * T = &P[it*ns + s];
        const int64_t b0 = s*KC_CH;
        const int64_t nb = (kb - b0) < KC_CH ? (kb - b0) : KC_CH;
        for (int64_t b = 0; b < nb; b++) {
            vsc w[MR][2]; float sc[MR];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                const block_mxfp4 * bp = &A[rr*lda + b0 + b];
                sc[r] = e8m0_half_to_fp32(bp->e);
                iq4_lookup32t(bp->qs, kvalues_mxfp4_, w[r]);
            }
            iq4_place_chunk(T, b, w, sc);
        }
    }
}

// ---- weight repack: IQ4_XS ----
extern "C" void iq4xs_repack_a(const block_iq4_xs * A, int64_t lda,
                               int64_t m, int64_t k, void * packed) {
    aiq4_t * P = (aiq4_t *)packed;
    const int64_t nsb = k/QK_K, ns = sl32(k);
    for (int64_t it = 0; it < rt8(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        aiq4_t * T = &P[it*ns + s];
        const int64_t sb0 = (s*KC_CH)/8;
        const int64_t nsl = (nsb - sb0) < KC_CH/8 ? (nsb - sb0) : KC_CH/8;
        for (int64_t sb = 0; sb < nsl; sb++) {
            const block_iq4_xs * bp[MR]; float d[MR];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                bp[r] = &A[rr*lda + sb0 + sb];
                d[r]  = GGML_FP16_TO_FP32(bp[r]->d);
            }
            for (int ib = 0; ib < 8; ib++) {           // 32-groups
                vsc w[MR][2]; float sc[MR];
                for (int r = 0; r < MR; r++) {
                    const int ls = ((bp[r]->scales_l[ib/2] >> 4*(ib%2)) & 0xF)
                                 | (((bp[r]->scales_h >> 2*ib) & 3) << 4);
                    sc[r] = d[r] * (float)(ls - 32);
                    iq4_lookup32t(bp[r]->qs + 16*ib, iq4_kvalues_local, w[r]);
                }
                iq4_place_chunk(T, 8*sb + ib, w, sc);
            }
        }
    }
}

// ---- activation packs ----
extern "C" void iq4_pack_b_q8_0(const block_q8_0 * B, int64_t ldb,
                                int64_t n, int64_t k, void * packed) {
    biq4_t * P = (biq4_t *)packed;
    const int64_t kb = k/32, ns = sl32(k);
    const vuc flip = vec_splats((unsigned char)0x80);
    for (int64_t jt = 0; jt < ct8(n); jt++)
    for (int64_t s = 0; s < ns; s++) {
        biq4_t * T = &P[jt*ns + s];
        const int64_t b0 = s*KC_CH;
        const int64_t nb = (kb - b0) < KC_CH ? (kb - b0) : KC_CH;
        for (int64_t b = 0; b < nb; b++) {
            const block_q8_0 * yb[NR]; float dB[NR];
            for (int j = 0; j < NR; j++) {
                int64_t jj = jt*NR + j; if (jj >= n) jj = n - 1;
                yb[j] = &B[jj*ldb + b0 + b];
                dB[j] = GGML_FP16_TO_FP32(yb[j]->d);
            }
            T->dB[b][0] = (vfl){ dB[0], dB[1], dB[2], dB[3] };
            T->dB[b][1] = (vfl){ dB[4], dB[5], dB[6], dB[7] };
            vui rows4[4];
            for (int a = 0; a < 2; a++)
                for (int h = 0; h < 2; h++) {
                    for (int j = 0; j < 4; j++)
                        rows4[j] = (vui)vec_xor(
                            load16u((const uint8_t *)(yb[4*a + j]->qs) + (16*h)), flip);
                    mma_transpose4(rows4, &T->v[b][8*h + a], 2);
                }
        }
    }
}

extern "C" void iq4_pack_b_q8_K(const block_q8_K * B, int64_t ldb,
                                int64_t n, int64_t k, void * packed) {
    biq4_t * P = (biq4_t *)packed;
    const int64_t nsb = k/QK_K, ns = sl32(k);
    const vuc flip = vec_splats((unsigned char)0x80);
    for (int64_t jt = 0; jt < ct8(n); jt++)
    for (int64_t s = 0; s < ns; s++) {
        biq4_t * T = &P[jt*ns + s];
        const int64_t sb0 = (s*KC_CH)/8;
        const int64_t nsl = (nsb - sb0) < KC_CH/8 ? (nsb - sb0) : KC_CH/8;
        for (int64_t sb = 0; sb < nsl; sb++) {
            const block_q8_K * yb[NR]; float dB[NR];
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

// ---- 8x8 microkernel on 4 accumulators (weights on the signed operand;
//      acc rows = weight rows) ----
static void kernel_iq4_8x8(const aiq4_t * PA, const biq4_t * PB,
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
                // fin += dB ⊙ (P*sA_r - C128_r)  per weight row r
                for (int r = 0; r < 4; r++) {
                    vfl t = vec_msub(vec_ctf(rowsP[r],0),
                                     vec_splats(sA[r]), vec_splats(C128[r]));
                    fin[4*g + r][cgi] = vec_madd(t, dB, fin[4*g + r][cgi]);
                }
            }
        }
    }
}

static void iq4_gemm_core(int64_t m, int64_t n, int64_t k,
                          const aiq4_t * PA, const biq4_t * PB,
                          float * C, int64_t ldc, int ith, int nth) {
    const int64_t kb = k/32, ns = sl32(k), mt = rt8(m), njt = ct8(n);
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
                kernel_iq4_8x8(&PA[it*ns + s], &PB[jt*ns + s], nch, fin);
            }
            const int64_t j0 = jt*NR;
            const int64_t cols = (n - j0) < NR ? (n - j0) : NR;
            for (int64_t r = 0; r < rows; r++)
                for (int64_t cj = 0; cj < cols; cj++)
                    C[(i + r) + (j0 + cj)*ldc] = fin[r][cj >> 2][cj & 3];
        }
    }
}

extern "C" void iq4_gemm_packed(int64_t m, int64_t n, int64_t k,
                                const void * packedA, const void * packedB,
                                float * C, int64_t ldc, int ith, int nth) {
    iq4_gemm_core(m, n, k, (const aiq4_t *)packedA, (const biq4_t *)packedB,
                  C, ldc, ith, nth);
}


#define IQ4_ONESHOT(NAME, BLKA, REPACK, YBLK, PACKB, VARIANT)            \
extern "C" void NAME(int64_t m, int64_t n, int64_t k,                          \
        const void * Av, int64_t lda, const void * Bv, int64_t ldb,            \
        float * C, int64_t ldc, int ith, int nth) {                            \
    const BLKA * A = (const BLKA *)Av;                                         \
    const YBLK * B = (const YBLK *)Bv;                                         \
    int fresh = 0;                                                             \
    void * PA = ppc_apack_cache_acquire(Av, m, k, VARIANT,                     \
                                        iq4_apack_size(m, k), &fresh);                  \
    if (PA) {                                                                  \
        if (fresh) { REPACK(A, lda, m, k, PA);                                 \
                     ppc_apack_cache_publish(Av, m, k, VARIANT); }             \
        const int64_t njt = (n + NR - 1) / NR;                                 \
        const int64_t jpt = (njt + nth - 1) / nth;                             \
        const int64_t jt0 = (int64_t)ith*jpt;                                  \
        const int64_t jt1 = (ith+1)*jpt < njt ? (ith+1)*jpt : njt;             \
        if (jt0 < jt1) {                                                       \
            const int64_t j0 = jt0*NR;                                         \
            const int64_t nc = (n - j0) < (jt1 - jt0)*NR ? (n - j0)            \
                                                         : (jt1 - jt0)*NR;    \
            void * PBl = aligned_alloc(64, iq4_bpack_size(nc, k));                      \
            if (!PBl) { GGML_ABORT("ppc-mma: pack alloc failed"); }           \
            PACKB(B + j0*ldb, ldb, nc, k, PBl);                                                 \
            iq4_gemm_packed(m, nc, k, PA, PBl, C + j0*ldc, ldc, 0, 1);                                                         \
            free(PBl);                                                         \
        }                                                                      \
    } else {                                                                   \
        void * PB = aligned_alloc(64, iq4_bpack_size(n, k));                            \
        if (!PB) { GGML_ABORT("ppc-mma: pack alloc failed"); }                \
        PACKB(B, ldb, n, k, PB);                                                      \
        void * PT = aligned_alloc(64, iq4_apack_size(MR, k));                           \
        if (!PT) { GGML_ABORT("ppc-mma: pack alloc failed"); }                \
        const int64_t mt = (m + MR - 1) / MR;                                  \
        const int64_t tpt = (mt + nth - 1) / nth;                              \
        for (int64_t it = ith*tpt; it < (ith+1)*tpt && it < mt; it++) {        \
            const int64_t i = it*MR;                                           \
            const int64_t rows = (m - i) < MR ? (m - i) : MR;                  \
            REPACK(A + i*lda, lda, rows, k, PT);                               \
            iq4_gemm_packed(rows, n, k, PT, PB, C + i, ldc, 0, 1);                                                          \
        }                                                                      \
        free(PT);                                                              \
        free(PB);                                                              \
    }                                                                          \
}
IQ4_ONESHOT(gemm_iq4_nl_q8_0_ppc, block_iq4_nl, iq4nl_repack_a, block_q8_0, iq4_pack_b_q8_0, 1)
IQ4_ONESHOT(gemm_iq4_xs_q8_K_ppc, block_iq4_xs, iq4xs_repack_a, block_q8_K, iq4_pack_b_q8_K, 2)
IQ4_ONESHOT(gemm_mxfp4_q8_0_ppc, block_mxfp4, mxfp4_repack_a, block_q8_0, iq4_pack_b_q8_0, 3)

#endif // __MMA__


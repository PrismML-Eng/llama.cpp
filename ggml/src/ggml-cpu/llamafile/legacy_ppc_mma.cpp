// legacy_ppc_mma.cpp - imported from github.com/mavin2009/ppc-mma-kernels
// (standalone-verified vs exact double references under qemu -cpu power10).
// Q4_1 x Q8_1, Q5_0 x Q8_0, Q5_1 x Q8_1 on POWER10/POWER11 MMA.

#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "kquants_ppc_mma.h"

#include <altivec.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>









static_assert(sizeof(block_q4_1) == 4 + 16, "bad q4_1");
static_assert(sizeof(block_q5_0) == 2 + 4 + 16, "bad q5_0");
static_assert(sizeof(block_q5_1) == 4 + 4 + 16, "bad q5_1");
static_assert(sizeof(block_q8_1) == 4 + 32, "bad q8_1");


#if defined(__MMA__) && defined(__powerpc64__)

typedef vector unsigned char  vuc;
typedef vector signed char    vsc;
typedef vector unsigned int   vui;
typedef vector signed int     vsi;
typedef vector float          vfl;

// Unaligned 16-byte load via memcpy: single lxv, well-defined for the
// odd struct offsets several block formats use.
static inline vuc load16u(const void * p) { vuc v; memcpy(&v, p, 16); return v; }


#define KC_CH 64                          // 32-elem chunks per slab (2048)
#define MR    16
#define NR    8

typedef struct {
    vuc v[KC_CH][32];                     // 8 depth-steps x 4 rowgroups
    vfl dA[KC_CH][4];                     // per-block scale per rowgroup
    vfl mA[KC_CH][4];                     // per-block min (Q4_1/Q5_1; 0 for Q5_0)
} aleg_t;

typedef struct {
    vuc v[KC_CH][16];                     // 8 depth-steps x 2 colgroups
    vfl dB[KC_CH][2];                     // per-block activation scale
    vfl SB[KC_CH][2];                     // Q8_1: s = dB*sum(y); Q8_0: 16*dB*sum(y)
} bleg_t;

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

static inline void nibbles32(const uint8_t * qs, vuc v[2]) {
    vuc raw = load16u((const uint8_t *)(qs) + (0));
    v[0] = vec_and(raw, vec_splats((unsigned char)0xF));
    v[1] = vec_sr(raw, vec_splats((unsigned char)4));
}

// merge qh bits: elems 0..15 use bits 0..15 (bytes 0-1), elems 16..31
// use bits 16..31 (bytes 2-3); bit -> position 4.
static inline void qh_merge(const uint8_t * qh, vuc v[2]) {
    vuc raw = load16u((const uint8_t *)(qh) + (0));   // first 4 bytes used
    const vuc repL = { 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1 };
    const vuc repH = { 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3 };
    const vuc sh   = { 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7 };
    const vuc one  = vec_splats((unsigned char)1);
    const vuc four = vec_splats((unsigned char)4);
    vuc b0 = vec_sl(vec_and(vec_sr(vec_perm(raw, raw, repL), sh), one), four);
    vuc b1 = vec_sl(vec_and(vec_sr(vec_perm(raw, raw, repH), sh), one), four);
    v[0] = vec_or(v[0], b0);
    v[1] = vec_or(v[1], b1);
}

static inline int64_t rt(int64_t m) { return (m + MR - 1) / MR; }
static inline int64_t ct(int64_t n) { return (n + NR - 1) / NR; }
static inline int64_t sl(int64_t k) { return (k/32 + KC_CH - 1) / KC_CH; }

extern "C" size_t leg_apack_size(int64_t m, int64_t k) {
    return (((size_t)(rt(m) * sl(k)) * sizeof(aleg_t)) + 63) & ~(size_t)63;
}
extern "C" size_t leg_bpack_size(int64_t n, int64_t k) {
    return (((size_t)(ct(n) * sl(k)) * sizeof(bleg_t)) + 63) & ~(size_t)63;
}

// generic weight repack over a per-block "codes + d + m" extractor
template <typename BLK, void (*CODES)(const BLK *, vuc[2]), bool HAS_M>
static void repack_generic(const BLK * A, int64_t lda, int64_t m, int64_t k, aleg_t * P) {
    const int64_t kb = k/32, ns = sl(k);
    for (int64_t it = 0; it < rt(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        aleg_t * T = &P[it*ns + s];
        const int64_t b0 = s*KC_CH;
        const int64_t nb = (kb - b0) < KC_CH ? (kb - b0) : KC_CH;
        for (int64_t b = 0; b < nb; b++) {
            vuc t[MR][2]; float d[MR], mm[MR];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                const BLK * bp = &A[rr*lda + b0 + b];
                d[r]  = GGML_FP16_TO_FP32(bp->d);
                mm[r] = HAS_M ? GGML_FP16_TO_FP32(((const ggml_half *)bp)[1]) : 0.0f;
                CODES(bp, t[r]);
            }
            for (int g = 0; g < 4; g++) {
                T->dA[b][g] = (vfl){ d[4*g], d[4*g+1], d[4*g+2], d[4*g+3] };
                T->mA[b][g] = (vfl){ mm[4*g], mm[4*g+1], mm[4*g+2], mm[4*g+3] };
            }
            vui rows4[4];
            for (int g = 0; g < 4; g++)
                for (int h = 0; h < 2; h++) {
                    for (int r = 0; r < 4; r++) rows4[r] = (vui)t[4*g + r][h];
                    mma_transpose4(rows4, &T->v[b][16*h + g], 4);
                }
        }
    }
}

static void codes_q4_1(const block_q4_1 * bp, vuc v[2]) { nibbles32(bp->qs, v); }
static void codes_q4_0(const block_q4_0 * bp, vuc v[2]) { nibbles32(bp->qs, v); }
static void codes_q5_0(const block_q5_0 * bp, vuc v[2]) { nibbles32(bp->qs, v); qh_merge(bp->qh, v); }
static void codes_q5_1(const block_q5_1 * bp, vuc v[2]) { nibbles32(bp->qs, v); qh_merge(bp->qh, v); }

extern "C" void leg_repack_q4_1(const block_q4_1 * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_generic<block_q4_1, codes_q4_1, true>(A, lda, m, k, (aleg_t *)p);
}
extern "C" void leg_repack_q4_0(const block_q4_0 * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_generic<block_q4_0, codes_q4_0, false>(A, lda, m, k, (aleg_t *)p);
}
extern "C" void leg_repack_q5_0(const block_q5_0 * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_generic<block_q5_0, codes_q5_0, false>(A, lda, m, k, (aleg_t *)p);
}
extern "C" void leg_repack_q5_1(const block_q5_1 * A, int64_t lda, int64_t m, int64_t k, void * p) {
    repack_generic<block_q5_1, codes_q5_1, true>(A, lda, m, k, (aleg_t *)p);
}

// activation packs: SB = dB*sum(y) computed (Q8_0, x16 for the Q5_0
// offset) or read from the precomputed s field (Q8_1).
template <typename YBLK, bool READ_S, int SFACT>
static void pack_b_generic(const YBLK * B, int64_t ldb, int64_t n, int64_t k, bleg_t * P) {
    const int64_t kb = k/32, ns = sl(k);
    for (int64_t jt = 0; jt < ct(n); jt++)
    for (int64_t s = 0; s < ns; s++) {
        bleg_t * T = &P[jt*ns + s];
        const int64_t b0 = s*KC_CH;
        const int64_t nb = (kb - b0) < KC_CH ? (kb - b0) : KC_CH;
        for (int64_t b = 0; b < nb; b++) {
            const YBLK * yb[NR]; float dB[NR], SB[NR];
            for (int j = 0; j < NR; j++) {
                int64_t jj = jt*NR + j; if (jj >= n) jj = n - 1;
                yb[j] = &B[jj*ldb + b0 + b];
                dB[j] = GGML_FP16_TO_FP32(yb[j]->d);
                if (READ_S) {
                    SB[j] = GGML_FP16_TO_FP32(((const ggml_half *)yb[j])[1]);
                } else {
                    vsi z = vec_splats(0);
                    vsi sm = vec_sum4s((vsc)load16u((const uint8_t *)(yb[j]->qs) + (0)), z);
                    sm = vec_sum4s((vsc)load16u((const uint8_t *)(yb[j]->qs) + (16)), sm);
                    SB[j] = (float)SFACT * dB[j] * (float)(sm[0]+sm[1]+sm[2]+sm[3]);
                }
            }
            T->dB[b][0] = (vfl){ dB[0], dB[1], dB[2], dB[3] };
            T->dB[b][1] = (vfl){ dB[4], dB[5], dB[6], dB[7] };
            T->SB[b][0] = (vfl){ SB[0], SB[1], SB[2], SB[3] };
            T->SB[b][1] = (vfl){ SB[4], SB[5], SB[6], SB[7] };
            vui rows4[4];
            for (int a = 0; a < 2; a++)
                for (int h = 0; h < 2; h++) {
                    for (int j = 0; j < 4; j++)
                        rows4[j] = (vui)load16u((const uint8_t *)(yb[4*a + j]->qs) + (16*h));
                    mma_transpose4(rows4, &T->v[b][8*h + a], 2);
                }
        }
    }
}

extern "C" void leg_pack_b_q8_0(const block_q8_0 * B, int64_t ldb, int64_t n, int64_t k, void * p) {
    pack_b_generic<block_q8_0, false, 16>(B, ldb, n, k, (bleg_t *)p);
}
// SFACT 8 variant for Q4_0's offset (t = q - 8)
extern "C" void leg_pack_b_q8_0_o8(const block_q8_0 * B, int64_t ldb, int64_t n, int64_t k, void * p) {
    pack_b_generic<block_q8_0, false, 8>(B, ldb, n, k, (bleg_t *)p);
}
extern "C" void leg_pack_b_q8_1(const block_q8_1 * B, int64_t ldb, int64_t n, int64_t k, void * p) {
    pack_b_generic<block_q8_1, true, 1>(B, ldb, n, k, (bleg_t *)p);
}

// MODE 0 (offset, Q5_0):  fin += dA ⊙ (dB_j*P - SB_j)          [SB = 16*dB*S]
// MODE 1 (affine, Q4_1/Q5_1): fin += dA*(dB_j*P) + mA*SB_j     [SB = dB*sum(y)]
template <int MODE>
static void kernel_leg_16x8(const aleg_t * PA, const bleg_t * PB,
                            int64_t nb, vfl fin[NR][4]) {
    for (int64_t b = 0; b < nb; b++) {
        const vuc * a = PA->v[b];
        const vuc * y = PB->v[b];
        if (b + 1 < nb) {
            __builtin_prefetch(PA->v[b + 1], 0, 3);
            __builtin_prefetch((const char *)PA->v[b + 1] + 256, 0, 3);
            __builtin_prefetch(PB->v[b + 1], 0, 3);
        }
        __vector_quad acc[2][4];
        for (int i = 0; i < 2; i++)
            for (int g = 0; g < 4; g++)
                __builtin_mma_xxsetaccz(&acc[i][g]);
        for (int x = 0; x < 8; x++) {
            const vuc y0 = y[2*x], y1 = y[2*x + 1];
            const vuc w0 = a[4*x], w1 = a[4*x+1], w2 = a[4*x+2], w3 = a[4*x+3];
            __builtin_mma_xvi8ger4pp(&acc[0][0], y0, w0);
            __builtin_mma_xvi8ger4pp(&acc[0][1], y0, w1);
            __builtin_mma_xvi8ger4pp(&acc[0][2], y0, w2);
            __builtin_mma_xvi8ger4pp(&acc[0][3], y0, w3);
            __builtin_mma_xvi8ger4pp(&acc[1][0], y1, w0);
            __builtin_mma_xvi8ger4pp(&acc[1][1], y1, w1);
            __builtin_mma_xvi8ger4pp(&acc[1][2], y1, w2);
            __builtin_mma_xvi8ger4pp(&acc[1][3], y1, w3);
        }
        vsi pr[2][4][4];
        for (int i = 0; i < 2; i++)
            for (int g = 0; g < 4; g++)
                __builtin_mma_disassemble_acc(pr[i][g], &acc[i][g]);
        for (int i = 0; i < 2; i++) {
            const vfl dB = PB->dB[b][i];
            const vfl SB = PB->SB[b][i];
            for (int g = 0; g < 4; g++) {
                const vfl dA = PA->dA[b][g];
                const vfl mA = PA->mA[b][g];
                for (int r = 0; r < 4; r++) {
                    const vfl res = vec_ctf(pr[i][g][r], 0);
                    if (MODE == 0) {
                        vfl t = vec_msub(res, vec_splat(dB, r), vec_splat(SB, r));
                        fin[4*i + r][g] = vec_madd(t, dA, fin[4*i + r][g]);
                    } else {
                        vfl vs = vec_mul(vec_splat(dB, r), dA);
                        fin[4*i + r][g] = vec_madd(res, vs, fin[4*i + r][g]);
                        fin[4*i + r][g] = vec_madd(mA, vec_splat(SB, r), fin[4*i + r][g]);
                    }
                }
            }
        }
    }
}

template <int MODE>
static void leg_gemm_core(int64_t m, int64_t n, int64_t k,
                          const aleg_t * PA, const bleg_t * PB,
                          float * C, int64_t ldc, int ith, int nth) {
    const int64_t kb = k/32, ns = sl(k), mt = rt(m), njt = ct(n);
    const int64_t tpt = (mt + nth - 1) / nth;
    const int64_t t0 = ith*tpt, t1 = (ith+1)*tpt < mt ? (ith+1)*tpt : mt;
    vfl fin[NR][4];
    for (int64_t it = t0; it < t1; it++) {
        const int64_t i = it*MR;
        const int64_t rows = (m - i) < MR ? (m - i) : MR;
        for (int64_t jt = 0; jt < njt; jt++) {
            for (int j = 0; j < NR; j++)
                for (int g = 0; g < 4; g++) fin[j][g] = vec_splats(0.0f);
            for (int64_t s = 0; s < ns; s++) {
                const int64_t b0 = s*KC_CH;
                const int64_t nb = (kb - b0) < KC_CH ? (kb - b0) : KC_CH;
                kernel_leg_16x8<MODE>(&PA[it*ns + s], &PB[jt*ns + s], nb, fin);
            }
            const int64_t j0 = jt*NR;
            const int64_t cols = (n - j0) < NR ? (n - j0) : NR;
            for (int64_t cj = 0; cj < cols; cj++) {
                float * dst = C + i + (j0 + cj)*ldc;
                if (rows == MR) {
                    for (int g = 0; g < 4; g++) vec_xst(fin[cj][g], 16*g, dst);
                } else {
                    for (int64_t r = 0; r < rows; r++)
                        dst[r] = fin[cj][r >> 2][r & 3];
                }
            }
        }
    }
}

extern "C" void leg_gemm_offset(int64_t m, int64_t n, int64_t k,
        const void * PA, const void * PB, float * C, int64_t ldc, int ith, int nth) {
    leg_gemm_core<0>(m, n, k, (const aleg_t *)PA, (const bleg_t *)PB, C, ldc, ith, nth);
}
extern "C" void leg_gemm_affine(int64_t m, int64_t n, int64_t k,
        const void * PA, const void * PB, float * C, int64_t ldc, int ith, int nth) {
    leg_gemm_core<1>(m, n, k, (const aleg_t *)PA, (const bleg_t *)PB, C, ldc, ith, nth);
}


// one-shot drivers (pack per call; see K-quant note in INTEGRATION.md)
#define LEG_ONESHOT(NAME, BLKA, REPACK, YBLK, PACKB, GEMM, VARIANT)            \
extern "C" void NAME(int64_t m, int64_t n, int64_t k,                          \
        const void * Av, int64_t lda, const void * Bv, int64_t ldb,            \
        float * C, int64_t ldc, int ith, int nth) {                            \
    const BLKA * A = (const BLKA *)Av;                                         \
    const YBLK * B = (const YBLK *)Bv;                                         \
    int fresh = 0;                                                             \
    void * PA = ppc_apack_cache_acquire(Av, m, k, VARIANT,                     \
                                        leg_apack_size(m, k), &fresh);                  \
    if (PA) {                                                                  \
        if (fresh) { REPACK(A, lda, m, k, PA);                                 \
                     ppc_apack_cache_publish(Av, m, k, VARIANT); }             \
        const int64_t njt = (n + NR - 1) / NR;                                 \
        if (njt < nth) {                                               \
            /* n too small to feed every thread by columns (worst      \
               case n == 1 generation: one column, nth-1 idle threads  \
               while scalar row-partitions). Row-partition with the    \
               cached pack instead; the full activation pack is tiny   \
               at these n. Field regression, Q4_K tg32, 2026-07-21. */ \
            void * PBs = aligned_alloc(64, leg_bpack_size(n, k));      \
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
            void * PBl = aligned_alloc(64, leg_bpack_size(nc, k));                      \
            if (!PBl) { GGML_ABORT("ppc-mma: pack alloc failed"); }           \
            PACKB(B + j0*ldb, ldb, nc, k, PBl);                                                 \
            GEMM(m, nc, k, PA, PBl, C + j0*ldc, ldc, 0, 1);                                                         \
            free(PBl);                                                         \
        }                                                                      \
        }                                                              \
    } else {                                                                   \
        void * PB = aligned_alloc(64, leg_bpack_size(n, k));                            \
        if (!PB) { GGML_ABORT("ppc-mma: pack alloc failed"); }                \
        PACKB(B, ldb, n, k, PB);                                                      \
        void * PT = aligned_alloc(64, leg_apack_size(MR, k));                           \
        if (!PT) { GGML_ABORT("ppc-mma: pack alloc failed"); }                \
        const int64_t mt = (m + MR - 1) / MR;                                  \
        const int64_t tpt = (mt + nth - 1) / nth;                              \
        for (int64_t it = ith*tpt; it < (ith+1)*tpt && it < mt; it++) {        \
            const int64_t i = it*MR;                                           \
            const int64_t rows = (m - i) < MR ? (m - i) : MR;                  \
            REPACK(A + i*lda, lda, rows, k, PT);                               \
            GEMM(rows, n, k, PT, PB, C + i, ldc, 0, 1);                                                          \
        }                                                                      \
        free(PT);                                                              \
        free(PB);                                                              \
    }                                                                          \
}
LEG_ONESHOT(gemm_q4_1_q8_1_ppc, block_q4_1, leg_repack_q4_1, block_q8_1, leg_pack_b_q8_1, leg_gemm_affine, 25)
LEG_ONESHOT(gemm_q5_0_q8_0_ppc, block_q5_0, leg_repack_q5_0, block_q8_0, leg_pack_b_q8_0, leg_gemm_offset, 26)
LEG_ONESHOT(gemm_q5_1_q8_1_ppc, block_q5_1, leg_repack_q5_1, block_q8_1, leg_pack_b_q8_1, leg_gemm_affine, 27)
LEG_ONESHOT(gemm_q4_0_q8_0_ppc, block_q4_0, leg_repack_q4_0, block_q8_0, leg_pack_b_q8_0_o8, leg_gemm_offset, 28)

#endif // __MMA__


// q2k_ppc_mma.cpp - POWER10/POWER11 MMA GEMM for block_q2_K x Q8_K.
// Imported from github.com/mavin2009/ppc-mma-kernels (standalone-verified
// against exact double references under qemu -cpu power10; see that
// repo's docs/DESIGN.md).  This TU adds a one-shot driver that packs the
// calling thread's row tiles and (per thread) the activations, then runs
// the packed GEMM -- a first integration; a repack.cpp-based load-time
// weight pack is the follow-up.

#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "kquants_ppc_mma.h"

#include <altivec.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>






static_assert(sizeof(block_q2_K) == QK_K/16 + QK_K/4 + 2*sizeof(ggml_half), "bad q2_K");


#if defined(__MMA__) && defined(__powerpc64__)

typedef vector unsigned char  vuc;
typedef vector signed char    vsc;
typedef vector unsigned int   vui;
typedef vector signed int     vsi;
typedef vector float          vfl;

// Unaligned 16-byte load via memcpy: single lxv, well-defined for the
// odd struct offsets several block formats use.
static inline vuc load16u(const void * p) { vuc v; memcpy(&v, p, 16); return v; }


#define KC_SB     8                        // superblocks per slab (2048 elems)
#define KC_CH16   (KC_SB * 16)             // 16-element chunks per slab
#define MR        16
#define NR        8

typedef struct {
    vuc v[KC_CH16][16];                    // per chunk: 4 depth-steps x 4 rowgroups
    vfl dsc[KC_CH16][4];                   // d*(sc&0xF) per chunk per rowgroup
    vfl dm [KC_CH16][4];                   // dmin*(sc>>4)
} a2k_t;

typedef struct {
    vuc v[KC_CH16][8];                     // per chunk: 4 depth-steps x 2 colgroups
    vfl dB[KC_SB][2];
    vfl TS[KC_CH16][2];                    // dB * bsums[c] per column
} b2k_t;

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

// 16 unsigned 2-bit codes of chunk c (0..15) of one superblock.
static inline vuc q2k_codes16(const uint8_t * qs, int c) {
    const int h = c >> 3, idx = c & 7, j = idx >> 1, lo = idx & 1;
    vuc v = load16u((const uint8_t *)(qs) + (32*h + 16*lo));
    return vec_and(vec_sr(v, vec_splats((unsigned char)(2*j))),
                   vec_splats((unsigned char)3));
}

static inline int64_t rt(int64_t m) { return (m + MR - 1) / MR; }
static inline int64_t ct(int64_t n) { return (n + NR - 1) / NR; }
static inline int64_t sl(int64_t k) { return (k/QK_K + KC_SB - 1) / KC_SB; }

extern "C" size_t q2k_apack_size(int64_t m, int64_t k) {
    return (((size_t)(rt(m) * sl(k)) * sizeof(a2k_t)) + 63) & ~(size_t)63;
}
extern "C" size_t q2k_bpack_size(int64_t n, int64_t k) {
    return (((size_t)(ct(n) * sl(k)) * sizeof(b2k_t)) + 63) & ~(size_t)63;
}

extern "C" void q2k_repack_a(const block_q2_K * A, int64_t lda,
                             int64_t m, int64_t k, void * packed) {
    a2k_t * P = (a2k_t *)packed;
    const int64_t nsb = k/QK_K, ns = sl(k);
    for (int64_t it = 0; it < rt(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        a2k_t * T = &P[it*ns + s];
        const int64_t sb0 = s*KC_SB;
        const int64_t nsl = (nsb - sb0) < KC_SB ? (nsb - sb0) : KC_SB;
        for (int64_t b = 0; b < nsl; b++) {
            const block_q2_K * bp[MR]; float d[MR], dmin[MR];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                bp[r] = &A[rr*lda + sb0 + b];
                d[r]    = GGML_FP16_TO_FP32(bp[r]->d);
                dmin[r] = GGML_FP16_TO_FP32(bp[r]->dmin);
            }
            for (int c = 0; c < 16; c++) {
                const int64_t ch = 16*b + c;
                for (int g = 0; g < 4; g++) {
                    float sc[4], mn[4];
                    for (int r = 0; r < 4; r++) {
                        const uint8_t s8 = bp[4*g + r]->scales[c];
                        sc[r] = d[4*g + r]    * (float)(s8 & 0xF);
                        mn[r] = dmin[4*g + r] * (float)(s8 >> 4);
                    }
                    T->dsc[ch][g] = (vfl){ sc[0], sc[1], sc[2], sc[3] };
                    T->dm [ch][g] = (vfl){ mn[0], mn[1], mn[2], mn[3] };
                }
                vui rows4[4];
                for (int g = 0; g < 4; g++) {
                    for (int r = 0; r < 4; r++)
                        rows4[r] = (vui)q2k_codes16(bp[4*g + r]->qs, c);
                    mma_transpose4(rows4, &T->v[ch][g], 4);
                }
            }
        }
    }
}

extern "C" void q2k_pack_b(const block_q8_K * B, int64_t ldb,
                           int64_t n, int64_t k, void * packed) {
    b2k_t * P = (b2k_t *)packed;
    const int64_t nsb = k/QK_K, ns = sl(k);
    for (int64_t jt = 0; jt < ct(n); jt++)
    for (int64_t s = 0; s < ns; s++) {
        b2k_t * T = &P[jt*ns + s];
        const int64_t sb0 = s*KC_SB;
        const int64_t nsl = (nsb - sb0) < KC_SB ? (nsb - sb0) : KC_SB;
        for (int64_t b = 0; b < nsl; b++) {
            const block_q8_K * yb[NR]; float dB[NR];
            for (int j = 0; j < NR; j++) {
                int64_t jj = jt*NR + j; if (jj >= n) jj = n - 1;
                yb[j] = &B[jj*ldb + sb0 + b];
                dB[j] = yb[j]->d;
            }
            T->dB[b][0] = (vfl){ dB[0], dB[1], dB[2], dB[3] };
            T->dB[b][1] = (vfl){ dB[4], dB[5], dB[6], dB[7] };
            for (int c = 0; c < 16; c++) {
                const int64_t ch = 16*b + c;
                float TS[NR];
                for (int j = 0; j < NR; j++)
                    TS[j] = dB[j] * (float)yb[j]->bsums[c];
                T->TS[ch][0] = (vfl){ TS[0], TS[1], TS[2], TS[3] };
                T->TS[ch][1] = (vfl){ TS[4], TS[5], TS[6], TS[7] };
                vui rows4[4];
                for (int a = 0; a < 2; a++) {
                    for (int j = 0; j < 4; j++)
                        rows4[j] = (vui)load16u((const uint8_t *)(yb[4*a + j]->qs) + (16*c));
                    mma_transpose4(rows4, &T->v[ch][a], 2);
                }
            }
        }
    }
}

static void kernel2k_16x8(const a2k_t * PA, const b2k_t * PB,
                          int64_t nsl, vfl fin[NR][4]) {
    for (int64_t b = 0; b < nsl; b++) {
        const vfl dB0 = PB->dB[b][0], dB1 = PB->dB[b][1];
        for (int c = 0; c < 16; c++) {
            const int64_t ch = 16*b + c;
            const vuc * a = PA->v[ch];
            const vuc * y = PB->v[ch];
            if (ch + 1 < 16*nsl) {
                __builtin_prefetch(PA->v[ch + 1], 0, 3);
                __builtin_prefetch(PB->v[ch + 1], 0, 3);
            }
            __vector_quad acc[2][4];
            for (int i = 0; i < 2; i++)
                for (int g = 0; g < 4; g++)
                    __builtin_mma_xxsetaccz(&acc[i][g]);
            for (int x = 0; x < 4; x++) {
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
            // Disassemble all accumulators to a stack buffer first:
            // frees the acc-aliased VSRs (0-31) before the fixup runs, so
            // fin + scale vectors fit the register file without spills.
            vsi pr[2][4][4];
            for (int i = 0; i < 2; i++)
                for (int g = 0; g < 4; g++)
                    __builtin_mma_disassemble_acc(pr[i][g], &acc[i][g]);
            // fin += dsc*(dB_j*P) ; fin -= dm*TS_j   (mins term)
            for (int i = 0; i < 2; i++) {
                const vfl dB = i ? dB1 : dB0;
                const vfl TS = PB->TS[ch][i];
                for (int g = 0; g < 4; g++) {
                    const vsi * rowsP = pr[i][g];
                    const vfl dsc = PA->dsc[ch][g];
                    const vfl dm  = PA->dm [ch][g];
                    const vfl s0 = vec_mul(vec_splat(dB, 0), dsc);
                    const vfl s1 = vec_mul(vec_splat(dB, 1), dsc);
                    const vfl s2 = vec_mul(vec_splat(dB, 2), dsc);
                    const vfl s3 = vec_mul(vec_splat(dB, 3), dsc);
                    fin[4*i+0][g] = vec_madd(vec_ctf(rowsP[0],0), s0, fin[4*i+0][g]);
                    fin[4*i+1][g] = vec_madd(vec_ctf(rowsP[1],0), s1, fin[4*i+1][g]);
                    fin[4*i+2][g] = vec_madd(vec_ctf(rowsP[2],0), s2, fin[4*i+2][g]);
                    fin[4*i+3][g] = vec_madd(vec_ctf(rowsP[3],0), s3, fin[4*i+3][g]);
                    fin[4*i+0][g] = vec_nmsub(dm, vec_splat(TS, 0), fin[4*i+0][g]);
                    fin[4*i+1][g] = vec_nmsub(dm, vec_splat(TS, 1), fin[4*i+1][g]);
                    fin[4*i+2][g] = vec_nmsub(dm, vec_splat(TS, 2), fin[4*i+2][g]);
                    fin[4*i+3][g] = vec_nmsub(dm, vec_splat(TS, 3), fin[4*i+3][g]);
                }
            }
        }
    }
}

extern "C" void q2k_gemm_packed(int64_t m, int64_t n, int64_t k,
                                const void * packedA, const void * packedB,
                                float * C, int64_t ldc, int ith, int nth) {
    const a2k_t * PA = (const a2k_t *)packedA;
    const b2k_t * PB = (const b2k_t *)packedB;
    const int64_t nsb = k/QK_K, ns = sl(k), mt = rt(m), njt = ct(n);
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
                const int64_t sb0 = s*KC_SB;
                const int64_t nsl = (nsb - sb0) < KC_SB ? (nsb - sb0) : KC_SB;
                kernel2k_16x8(&PA[it*ns + s], &PB[jt*ns + s], nsl, fin);
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


// One-shot driver: packs this thread's row tiles (and, per thread, the
// activations) then runs the packed GEMM.
extern "C" void gemm_q2_K_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * Av, int64_t lda, const void * Bv, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth) {
    const block_q2_K * A = (const block_q2_K *)Av;
    const block_q8_K * B = (const block_q8_K *)Bv;
    void * PB = aligned_alloc(64, q2k_bpack_size(n, k));
    if (!PB) { GGML_ABORT("ppc-mma: pack alloc failed"); }
    q2k_pack_b(B, ldb, n, k, PB);
    int fresh = 0;
    void * PA = ppc_apack_cache_acquire(Av, m, k, 23, q2k_apack_size(m, k), &fresh);
    if (PA) {
        if (fresh) { q2k_repack_a(A, lda, m, k, PA);
                     ppc_apack_cache_publish(Av, m, k, 23); }
        q2k_gemm_packed(m, n, k, PA, PB, C, ldc, ith, nth);
    } else {
        void * PT = aligned_alloc(64, q2k_apack_size(MR, k));
        if (!PT) { GGML_ABORT("ppc-mma: pack alloc failed"); }
        const int64_t mt = (m + MR - 1) / MR;
        const int64_t tpt = (mt + nth - 1) / nth;
        for (int64_t it = ith*tpt; it < (ith+1)*tpt && it < mt; it++) {
            const int64_t i = it*MR;
            const int64_t rows = (m - i) < MR ? (m - i) : MR;
            q2k_repack_a(A + i*lda, lda, rows, k, PT);
            q2k_gemm_packed(rows, n, k, PT, PB, C + i, ldc, 0, 1);
        }
        free(PT);
    }
    free(PB);
}

#endif // __MMA__


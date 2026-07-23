// q5k_ppc_mma.cpp - POWER10/POWER11 MMA GEMM for block_q5_K x Q8_K.
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






static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/8 + QK_K/2, "bad q5_K");
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + (QK_K/16)*sizeof(int16_t), "bad q8_K");


// ggml's 6-bit scale/min decode
static inline void q5k_get_scale_min(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

#if defined(__MMA__) && defined(__powerpc64__)

typedef vector unsigned char  vuc;
typedef vector signed char    vsc;
typedef vector unsigned int   vui;
typedef vector signed int     vsi;
typedef vector float          vfl;

// Unaligned 16-byte load via memcpy: single lxv, well-defined for the
// odd struct offsets several block formats use.
static inline vuc load16u(const void * p) { vuc v; memcpy(&v, p, 16); return v; }


#define KC_SB     8                       // superblocks per K slab (2048 elems)
#define KC_CHUNKS (KC_SB * 8)             // 32-element chunks per slab
#define MR        16
#define NR        8

typedef struct {
    vuc v[KC_CHUNKS][32];                 // MMA-layout unsigned nibbles
    vfl dsc[KC_CHUNKS][4];                // d*sc per chunk, 4 rows per vfl
    vfl dm [KC_CHUNKS][4];                // dmin*m per chunk
} a4k_t;

typedef struct {
    vuc v[KC_CHUNKS][16];                 // signed activations
    vfl dB[KC_SB][2];                     // Q8_K d per column, per superblock
    vfl TS[KC_CHUNKS][2];                 // dB * S_sub per (col, chunk)
} b4k_t;

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

// 5-bit codes of sub-block `sub` (0..7): Q4_K nibble + qh bit `sub`.
static inline void q5k_codes32(const uint8_t * qs, const uint8_t * qh,
                               int sub, vuc v[2]) {
    const int g = sub >> 1;
    vuc lo = load16u((const uint8_t *)(qs) + (32*g));
    vuc hi = load16u((const uint8_t *)(qs) + (32*g + 16));
    if ((sub & 1) == 0) {
        const vuc mF = vec_splats((unsigned char)0xF);
        lo = vec_and(lo, mF);
        hi = vec_and(hi, mF);
    } else {
        lo = vec_sr(lo, vec_splats((unsigned char)4));
        hi = vec_sr(hi, vec_splats((unsigned char)4));
    }
    const vuc hsh = vec_splats((unsigned char)sub);
    const vuc one = vec_splats((unsigned char)1);
    const vuc four = vec_splats((unsigned char)4);
    vuc h0 = vec_sl(vec_and(vec_sr(load16u((const uint8_t *)(qh) + (0)), hsh), one), four);
    vuc h1 = vec_sl(vec_and(vec_sr(load16u((const uint8_t *)(qh) + (16)), hsh), one), four);
    v[0] = vec_or(lo, h0);
    v[1] = vec_or(hi, h1);
}

// ---- one-time packing ----

static inline int64_t rt4k(int64_t m) { return (m + MR - 1) / MR; }
static inline int64_t ct4k(int64_t n) { return (n + NR - 1) / NR; }
static inline int64_t sl4k(int64_t k) { return (k/QK_K + KC_SB - 1) / KC_SB; }

extern "C" size_t q5k_apack_size(int64_t m, int64_t k) {
    return (((size_t)(rt4k(m) * sl4k(k)) * sizeof(a4k_t)) + 63) & ~(size_t)63;
}
extern "C" size_t q5k_bpack_size(int64_t n, int64_t k) {
    return (((size_t)(ct4k(n) * sl4k(k)) * sizeof(b4k_t)) + 63) & ~(size_t)63;
}

extern "C" void q5k_repack_a(const block_q5_K * A, int64_t lda,
                             int64_t m, int64_t k, void * packed) {
    a4k_t * P = (a4k_t *)packed;
    const int64_t nsb = k/QK_K, ns = sl4k(k);
    for (int64_t it = 0; it < rt4k(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        a4k_t * T = &P[it*ns + s];
        const int64_t sb0 = s*KC_SB;
        const int64_t nsl = (nsb - sb0) < KC_SB ? (nsb - sb0) : KC_SB;
        for (int64_t b = 0; b < nsl; b++) {
            const block_q5_K * bp[MR];
            float dsc[MR][8], dm[MR][8];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                bp[r] = &A[rr*lda + sb0 + b];
                const float d    = GGML_FP16_TO_FP32(bp[r]->d);
                const float dmin = GGML_FP16_TO_FP32(bp[r]->dmin);
                for (int sub = 0; sub < 8; sub++) {
                    uint8_t sc, mn;
                    q5k_get_scale_min(sub, bp[r]->scales, &sc, &mn);
                    dsc[r][sub] = d * sc;
                    dm [r][sub] = dmin * mn;
                }
            }
            for (int sub = 0; sub < 8; sub++) {
                const int64_t ch = 8*b + sub;
                for (int g = 0; g < 4; g++) {
                    T->dsc[ch][g] = (vfl){ dsc[4*g][sub], dsc[4*g+1][sub],
                                           dsc[4*g+2][sub], dsc[4*g+3][sub] };
                    T->dm [ch][g] = (vfl){ dm[4*g][sub], dm[4*g+1][sub],
                                           dm[4*g+2][sub], dm[4*g+3][sub] };
                }
                vuc t[MR][2];
                for (int r = 0; r < MR; r++) q5k_codes32(bp[r]->qs, bp[r]->qh, sub, t[r]);
                vui rows4[4];
                for (int g = 0; g < 4; g++)
                    for (int h = 0; h < 2; h++) {
                        for (int r = 0; r < 4; r++) rows4[r] = (vui)t[4*g + r][h];
                        mma_transpose4(rows4, &T->v[ch][16*h + g], 4);
                    }
            }
        }
    }
}

extern "C" void q5k_pack_b(const block_q8_K * B, int64_t ldb,
                           int64_t n, int64_t k, void * packed) {
    b4k_t * P = (b4k_t *)packed;
    const int64_t nsb = k/QK_K, ns = sl4k(k);
    for (int64_t jt = 0; jt < ct4k(n); jt++)
    for (int64_t s = 0; s < ns; s++) {
        b4k_t * T = &P[jt*ns + s];
        const int64_t sb0 = s*KC_SB;
        const int64_t nsl = (nsb - sb0) < KC_SB ? (nsb - sb0) : KC_SB;
        for (int64_t b = 0; b < nsl; b++) {
            const block_q8_K * yb[NR];
            float dB[NR];
            for (int j = 0; j < NR; j++) {
                int64_t jj = jt*NR + j; if (jj >= n) jj = n - 1;
                yb[j] = &B[jj*ldb + sb0 + b];
                dB[j] = yb[j]->d;
            }
            T->dB[b][0] = (vfl){ dB[0], dB[1], dB[2], dB[3] };
            T->dB[b][1] = (vfl){ dB[4], dB[5], dB[6], dB[7] };
            for (int sub = 0; sub < 8; sub++) {
                const int64_t ch = 8*b + sub;
                float TS[NR];
                for (int j = 0; j < NR; j++)
                    TS[j] = dB[j] * (float)(yb[j]->bsums[2*sub] + yb[j]->bsums[2*sub + 1]);
                T->TS[ch][0] = (vfl){ TS[0], TS[1], TS[2], TS[3] };
                T->TS[ch][1] = (vfl){ TS[4], TS[5], TS[6], TS[7] };
                vui rows4[4];
                for (int a = 0; a < 2; a++) {
                    vuc q[4][2];
                    for (int j = 0; j < 4; j++) {
                        q[j][0] = load16u((const uint8_t *)(yb[4*a + j]->qs) + (32*sub));
                        q[j][1] = load16u((const uint8_t *)(yb[4*a + j]->qs) + (32*sub + 16));
                    }
                    for (int h = 0; h < 2; h++) {
                        for (int j = 0; j < 4; j++) rows4[j] = (vui)q[j][h];
                        mma_transpose4(rows4, &T->v[ch][8*h + a], 2);
                    }
                }
            }
        }
    }
}

// ---- 16x8 microkernel ----
static void kernel4k_16x8(const a4k_t * PA, const b4k_t * PB,
                          int64_t nsl, vfl fin[NR][4]) {
    for (int64_t b = 0; b < nsl; b++) {
        for (int sub = 0; sub < 8; sub++) {
            const int64_t ch = 8*b + sub;
            const vuc * a = PA->v[ch];
            const vuc * y = PB->v[ch];
            if (ch + 1 < 8*nsl) {
                __builtin_prefetch(PA->v[ch + 1], 0, 3);
                __builtin_prefetch((const char *)PA->v[ch + 1] + 256, 0, 3);
                __builtin_prefetch(PB->v[ch + 1], 0, 3);
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
            // Stage all accumulators to the stack first (frees VSRs 0-31
            // for the fixup; see q6_k for rationale).
            vsi pr[2][4][4];
            for (int i = 0; i < 2; i++)
                for (int g = 0; g < 4; g++)
                    __builtin_mma_disassemble_acc(pr[i][g], &acc[i][g]);
            // fin += P * (dB_j * dsc_g)  ;  then fin -= TS_j * dm_g
            const vfl dB0 = PB->dB[b][0], dB1 = PB->dB[b][1];
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

extern "C" void q5k_gemm_packed(int64_t m, int64_t n, int64_t k,
                                const void * packedA, const void * packedB,
                                float * C, int64_t ldc, int ith, int nth) {
    const a4k_t * PA = (const a4k_t *)packedA;
    const b4k_t * PB = (const b4k_t *)packedB;
    const int64_t nsb = k/QK_K, ns = sl4k(k), mt = rt4k(m), njt = ct4k(n);
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
                kernel4k_16x8(&PA[it*ns + s], &PB[jt*ns + s], nsl, fin);
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
extern "C" void gemm_q5_K_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * Av, int64_t lda, const void * Bv, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth) {
    const block_q5_K * A = (const block_q5_K *)Av;
    const block_q8_K * B = (const block_q8_K *)Bv;
    int fresh = 0;   /* B is packed per-branch below */
    void * PA = ppc_apack_cache_acquire(Av, m, k, 21, q5k_apack_size(m, k), &fresh);
    if (PA) {
        if (fresh) { q5k_repack_a(A, lda, m, k, PA);
                     ppc_apack_cache_publish(Av, m, k, 21); }
        const int64_t njt = (n + NR - 1) / NR;
        if (njt < nth) {
            /* n too small to feed every thread by columns (worst
               case n == 1 generation: one column, nth-1 idle threads
               while scalar row-partitions). Row-partition with the
               cached pack instead; the full activation pack is tiny
               at these n. Field regression, Q4_K tg32, 2026-07-21. */
            void * PBs = aligned_alloc(64, q5k_bpack_size(n, k));
            if (!PBs) { GGML_ABORT("ppc-mma: pack alloc failed"); }
            q5k_pack_b(B, ldb, n, k, PBs);
            q5k_gemm_packed(m, n, k, PA, PBs, C, ldc, ith, nth);
            free(PBs);
        } else {
        const int64_t jpt = (njt + nth - 1) / nth;
        const int64_t jt0 = (int64_t)ith*jpt;
        const int64_t jt1 = (ith+1)*jpt < njt ? (ith+1)*jpt : njt;
        if (jt0 < jt1) {
            const int64_t j0 = jt0*NR;
            const int64_t nc = (n - j0) < (jt1 - jt0)*NR ? (n - j0) : (jt1 - jt0)*NR;
            void * PBl = aligned_alloc(64, q5k_bpack_size(nc, k));
            if (!PBl) { GGML_ABORT("ppc-mma: pack alloc failed"); }
            q5k_pack_b(B + j0*ldb, ldb, nc, k, PBl);
            q5k_gemm_packed(m, nc, k, PA, PBl, C + j0*ldc, ldc, 0, 1);
            free(PBl);
        }
        }
    } else {
        void * PB = aligned_alloc(64, q5k_bpack_size(n, k));
        if (!PB) { GGML_ABORT("ppc-mma: pack alloc failed"); }
        q5k_pack_b(B, ldb, n, k, PB);
        void * PT = aligned_alloc(64, q5k_apack_size(MR, k));
        if (!PT) { GGML_ABORT("ppc-mma: pack alloc failed"); }
        const int64_t mt = (m + MR - 1) / MR;
        const int64_t tpt = (mt + nth - 1) / nth;
        for (int64_t it = ith*tpt; it < (ith+1)*tpt && it < mt; it++) {
            const int64_t i = it*MR;
            const int64_t rows = (m - i) < MR ? (m - i) : MR;
            q5k_repack_a(A + i*lda, lda, rows, k, PT);
            q5k_gemm_packed(rows, n, k, PT, PB, C + i, ldc, 0, 1);
        }
        free(PT);
        free(PB);
    }
}

#endif // __MMA__


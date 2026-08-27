// q4k_ppc_mma.cpp - POWER10/POWER11 MMA GEMM for block_q4_K x Q8_K.
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






static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, "bad q4_K");
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + (QK_K/16)*sizeof(int16_t), "bad q8_K");


// ggml's 6-bit scale/min decode
static inline void q4k_get_scale_min(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
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

// nibble codes of sub-block `sub` (0..7) of one superblock
static inline void q4k_codes32(const uint8_t * qs, int sub, vuc v[2]) {
    const int g = sub >> 1;
    vuc lo = load16u((const uint8_t *)(qs) + (32*g));
    vuc hi = load16u((const uint8_t *)(qs) + (32*g + 16));
    if ((sub & 1) == 0) {
        const vuc mF = vec_splats((unsigned char)0xF);
        v[0] = vec_and(lo, mF);
        v[1] = vec_and(hi, mF);
    } else {
        v[0] = vec_sr(lo, vec_splats((unsigned char)4));
        v[1] = vec_sr(hi, vec_splats((unsigned char)4));
    }
}

// ---- one-time packing ----

static inline int64_t rt4k(int64_t m) { return (m + MR - 1) / MR; }
static inline int64_t ct4k(int64_t n) { return (n + NR - 1) / NR; }
static inline int64_t sl4k(int64_t k) { return (k/QK_K + KC_SB - 1) / KC_SB; }

extern "C" size_t q4k_apack_size(int64_t m, int64_t k) {
    return (((size_t)(rt4k(m) * sl4k(k)) * sizeof(a4k_t)) + 63) & ~(size_t)63;
}
extern "C" size_t q4k_bpack_size(int64_t n, int64_t k) {
    return (((size_t)(ct4k(n) * sl4k(k)) * sizeof(b4k_t)) + 63) & ~(size_t)63;
}

extern "C" void q4k_repack_a(const block_q4_K * A, int64_t lda,
                             int64_t m, int64_t k, void * packed) {
    a4k_t * P = (a4k_t *)packed;
    const int64_t nsb = k/QK_K, ns = sl4k(k);
    for (int64_t it = 0; it < rt4k(m); it++)
    for (int64_t s = 0; s < ns; s++) {
        a4k_t * T = &P[it*ns + s];
        const int64_t sb0 = s*KC_SB;
        const int64_t nsl = (nsb - sb0) < KC_SB ? (nsb - sb0) : KC_SB;
        for (int64_t b = 0; b < nsl; b++) {
            const block_q4_K * bp[MR];
            float dsc[MR][8], dm[MR][8];
            for (int r = 0; r < MR; r++) {
                int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                bp[r] = &A[rr*lda + sb0 + b];
                const float d    = GGML_FP16_TO_FP32(bp[r]->d);
                const float dmin = GGML_FP16_TO_FP32(bp[r]->dmin);
                for (int sub = 0; sub < 8; sub++) {
                    uint8_t sc, mn;
                    q4k_get_scale_min(sub, bp[r]->scales, &sc, &mn);
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
                for (int r = 0; r < MR; r++) q4k_codes32(bp[r]->qs, sub, t[r]);
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

extern "C" void q4k_pack_b(const block_q8_K * B, int64_t ldb,
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
#ifdef PPC_DCBT_STREAM
                __asm__ volatile(\"dcbt 0,%0,8\" :: \"r\"(PA->v[ch + 1]));
                __asm__ volatile(\"dcbt 0,%0,8\" :: \"r\"(PB->v[ch + 1]));
#else
                __builtin_prefetch(PA->v[ch + 1], 0, 3);
                __builtin_prefetch((const char *)PA->v[ch + 1] + 256, 0, 3);
                __builtin_prefetch(PB->v[ch + 1], 0, 3);
#endif
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
            // for the fixup; see DESIGN.md register-pressure note).
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

extern "C" void q4k_gemm_packed(int64_t m, int64_t n, int64_t k,
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
extern "C" void gemm_q4_K_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * Av, int64_t lda, const void * Bv, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth) {
    const block_q4_K * A = (const block_q4_K *)Av;
    const block_q8_K * B = (const block_q8_K *)Bv;
    int fresh = 0;   /* B is packed per-branch below */
    void * PA = ppc_apack_cache_acquire(Av, m, k, 20, q4k_apack_size(m, k), &fresh);
    if (PA) {
        if (fresh) { q4k_repack_a(A, lda, m, k, PA);
                     ppc_apack_cache_publish(Av, m, k, 20); }
        const int64_t njt = (n + NR - 1) / NR;
        if (njt < nth) {
            /* n too small to feed every thread by columns (worst
               case n == 1 generation: one column, nth-1 idle threads
               while scalar row-partitions). Row-partition with the
               cached pack instead; the full activation pack is tiny
               at these n. Field regression, Q4_K tg32, 2026-07-21. */
            void * PBs = aligned_alloc(64, q4k_bpack_size(n, k));
            if (!PBs) { GGML_ABORT("ppc-mma: pack alloc failed"); }
            q4k_pack_b(B, ldb, n, k, PBs);
            q4k_gemm_packed(m, n, k, PA, PBs, C, ldc, ith, nth);
            free(PBs);
        } else {
        const int64_t jpt = (njt + nth - 1) / nth;
        const int64_t jt0 = (int64_t)ith*jpt;
        const int64_t jt1 = (ith+1)*jpt < njt ? (ith+1)*jpt : njt;
        if (jt0 < jt1) {
            const int64_t j0 = jt0*NR;
            const int64_t nc = (n - j0) < (jt1 - jt0)*NR ? (n - j0) : (jt1 - jt0)*NR;
            void * PBl = aligned_alloc(64, q4k_bpack_size(nc, k));
            if (!PBl) { GGML_ABORT("ppc-mma: pack alloc failed"); }
            q4k_pack_b(B + j0*ldb, ldb, nc, k, PBl);
            q4k_gemm_packed(m, nc, k, PA, PBl, C + j0*ldc, ldc, 0, 1);
            free(PBl);
        }
        }
    } else {
        void * PB = aligned_alloc(64, q4k_bpack_size(n, k));
        if (!PB) { GGML_ABORT("ppc-mma: pack alloc failed"); }
        q4k_pack_b(B, ldb, n, k, PB);
        void * PT = aligned_alloc(64, q4k_apack_size(MR, k));
        if (!PT) { GGML_ABORT("ppc-mma: pack alloc failed"); }
        const int64_t mt = (m + MR - 1) / MR;
        const int64_t tpt = (mt + nth - 1) / nth;
        for (int64_t it = ith*tpt; it < (ith+1)*tpt && it < mt; it++) {
            const int64_t i = it*MR;
            const int64_t rows = (m - i) < MR ? (m - i) : MR;
            q4k_repack_a(A + i*lda, lda, rows, k, PT);
            q4k_gemm_packed(rows, n, k, PT, PB, C + i, ldc, 0, 1);
        }
        free(PT);
        free(PB);
    }
}

#endif // __MMA__

// ---- two-row GEMV, n = 1 (the near-tie squeeze) -----------------------
//
// Generation cannot use the MMA tiles profitably (the packed operand
// rides at 1.8x native bytes; VALIDATION-POWER10.md D3), and ggml's
// vec_dot keeps one weight stream per thread alive -- measured at 49%
// of the machine's bandwidth on a 27B (s9.1).  This kernel walks two
// rows per pass: twice the memory-level parallelism on the weight
// side, and every activation load shared between the rows.  Per-row
// arithmetic order is copied exactly from ggml's POWER9 vec_dot, so
// each row's result is bit-identical to the single-row path -- the
// speedup must come from the memory system alone, and correctness
// comparisons stay trivial.
extern "C" void gemv2_q4_K_q8_K_ppc(int64_t m, int64_t k,
                                    const void * Av, int64_t lda,
                                    const void * Bv, float * C,
                                    int ith, int nth) {
    const block_q4_K * __restrict A = (const block_q4_K *)Av;
    const block_q8_K * __restrict y = (const block_q8_K *)Bv;
    const int64_t nb = k / QK_K;

    const vector signed char lowMask  = vec_splats((signed char)0xF);
    const vector signed char lowMask1 = vec_splats((int8_t)0x3f);
    const vector signed char lowMask2 = vec_splats((int8_t)0x30);
    const vector int v0 = vec_splats((int32_t)0);
    const vector unsigned char v2 = vec_splats((uint8_t)2);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);

    const int64_t rpt = (m + nth - 1) / nth;
    const int64_t lo = (int64_t)ith * rpt;
    const int64_t hi = (lo + rpt) < m ? (lo + rpt) : m;

    for (int64_t r = lo; r < hi; r += 2) {
        const int two = (r + 1 < hi);
        const block_q4_K * xa = A + r*lda;
        const block_q4_K * xb = A + (two ? (r + 1)*lda : r*lda);

        vector float sfa0 = vec_splats(0.0f), sfa1 = vec_splats(0.0f);
        vector float sfa2 = vec_splats(0.0f), sfa3 = vec_splats(0.0f);
        vector float sfb0 = vec_splats(0.0f), sfb1 = vec_splats(0.0f);
        vector float sfb2 = vec_splats(0.0f), sfb3 = vec_splats(0.0f);

        for (int64_t i = 0; i < nb; ++i) {
            const vector float vyd = vec_splats(y[i].d);
            const vector signed short q8ysums0 = vec_xl( 0, y[i].bsums);
            const vector signed short q8ysums1 = vec_xl(16, y[i].bsums);

            // ---- per-row scale/min setup (identical to ggml's vec_dot) ----
            vector float vda, vdmina, vdb, vdminb;
            vector signed short vscalesa, vscalesb;
            vector signed short q4xmins0a, q4xmins1a, q4xmins0b, q4xmins1b;
            {
                const vector float vxd = vec_splats(ggml_fp16_to_fp32(xa[i].d));
                vda = vec_mul(vxd, vyd);
                const vector float vxmin = vec_splats(ggml_fp16_to_fp32(xa[i].dmin));
                vdmina = vec_mul(vxmin, vyd);
                vector signed char u0 = (vector signed char)vec_xl_len((unsigned char *)xa[i].scales, 8);
                vector signed char u1 = vec_and(vec_sr(u0, v2), lowMask2);
                vector signed char u2 = (vector signed char)vec_xl_len((unsigned char *)xa[i].scales + 8, 4);
                vector signed char u3 = vec_sr(u2, v4);
                vector signed char u30 = u1;
                vector signed char u31 = (vector signed char)vec_mergeh((vector signed int)vec_and(u2, lowMask), (vector signed int)u3);
                u1 = vec_and(u0, lowMask1);
                u2 = vec_or(u30, u31);
                vector signed char utmps = (vector signed char)vec_mergeh((vector signed int)u1, (vector signed int)u2);
                vscalesa = vec_unpackh(utmps);
                vector signed short q4xmins = vec_unpackl(utmps);
                q4xmins0a = vec_mergeh(q4xmins, q4xmins);
                q4xmins1a = vec_mergel(q4xmins, q4xmins);
            }
            {
                const vector float vxd = vec_splats(ggml_fp16_to_fp32(xb[i].d));
                vdb = vec_mul(vxd, vyd);
                const vector float vxmin = vec_splats(ggml_fp16_to_fp32(xb[i].dmin));
                vdminb = vec_mul(vxmin, vyd);
                vector signed char u0 = (vector signed char)vec_xl_len((unsigned char *)xb[i].scales, 8);
                vector signed char u1 = vec_and(vec_sr(u0, v2), lowMask2);
                vector signed char u2 = (vector signed char)vec_xl_len((unsigned char *)xb[i].scales + 8, 4);
                vector signed char u3 = vec_sr(u2, v4);
                vector signed char u30 = u1;
                vector signed char u31 = (vector signed char)vec_mergeh((vector signed int)vec_and(u2, lowMask), (vector signed int)u3);
                u1 = vec_and(u0, lowMask1);
                u2 = vec_or(u30, u31);
                vector signed char utmps = (vector signed char)vec_mergeh((vector signed int)u1, (vector signed int)u2);
                vscalesb = vec_unpackh(utmps);
                vector signed short q4xmins = vec_unpackl(utmps);
                q4xmins0b = vec_mergeh(q4xmins, q4xmins);
                q4xmins1b = vec_mergel(q4xmins, q4xmins);
            }

            {
                vector signed int prod0 = vec_mule(q4xmins0a, q8ysums0);
                vector signed int prod1 = vec_mule(q4xmins1a, q8ysums1);
                vector signed int prod2 = vec_mulo(q4xmins0a, q8ysums0);
                vector signed int prod3 = vec_mulo(q4xmins1a, q8ysums1);
                sfa0 = vec_nmsub(vec_ctf(prod0, 0), vdmina, sfa0);
                sfa1 = vec_nmsub(vec_ctf(prod1, 0), vdmina, sfa1);
                sfa2 = vec_nmsub(vec_ctf(prod2, 0), vdmina, sfa2);
                sfa3 = vec_nmsub(vec_ctf(prod3, 0), vdmina, sfa3);
            }
            {
                vector signed int prod0 = vec_mule(q4xmins0b, q8ysums0);
                vector signed int prod1 = vec_mule(q4xmins1b, q8ysums1);
                vector signed int prod2 = vec_mulo(q4xmins0b, q8ysums0);
                vector signed int prod3 = vec_mulo(q4xmins1b, q8ysums1);
                sfb0 = vec_nmsub(vec_ctf(prod0, 0), vdminb, sfb0);
                sfb1 = vec_nmsub(vec_ctf(prod1, 0), vdminb, sfb1);
                sfb2 = vec_nmsub(vec_ctf(prod2, 0), vdminb, sfb2);
                sfb3 = vec_nmsub(vec_ctf(prod3, 0), vdminb, sfb3);
            }

            vector signed int sia0 = v0, sia1 = v0, sia2 = v0, sia3 = v0;
            vector signed int sib0 = v0, sib1 = v0, sib2 = v0, sib3 = v0;

            const uint8_t * q4a = xa[i].qs;
            const uint8_t * q4b = xb[i].qs;
            const int8_t  * q8  = y[i].qs;

            for (int j = 0; j < QK_K/64; j += 2) {
                __builtin_prefetch(q4a, 0, 1);
                __builtin_prefetch(q4b, 0, 1);
                __builtin_prefetch(q8, 0, 1);
                const vector signed char qxa0 = (vector signed char)vec_xl( 0, q4a);
                const vector signed char qxa1 = (vector signed char)vec_xl(16, q4a);
                const vector signed char qxa2 = (vector signed char)vec_xl(32, q4a);
                const vector signed char qxa3 = (vector signed char)vec_xl(48, q4a);
                q4a += 64;
                const vector signed char qxb0 = (vector signed char)vec_xl( 0, q4b);
                const vector signed char qxb1 = (vector signed char)vec_xl(16, q4b);
                const vector signed char qxb2 = (vector signed char)vec_xl(32, q4b);
                const vector signed char qxb3 = (vector signed char)vec_xl(48, q4b);
                q4b += 64;

                const vector signed char q8y00 = vec_xl(  0, q8);
                const vector signed char q8y10 = vec_xl( 16, q8);
                const vector signed char q8y01 = vec_xl( 32, q8);
                const vector signed char q8y11 = vec_xl( 48, q8);
                const vector signed char q8y20 = vec_xl( 64, q8);
                const vector signed char q8y30 = vec_xl( 80, q8);
                const vector signed char q8y21 = vec_xl( 96, q8);
                const vector signed char q8y31 = vec_xl(112, q8);
                q8 += 128;

                {
                    vector unsigned char x00 = (vector unsigned char)vec_and(qxa0, lowMask);
                    vector unsigned char x01 = (vector unsigned char)vec_sr(qxa0, v4);
                    vector unsigned char x10 = (vector unsigned char)vec_and(qxa1, lowMask);
                    vector unsigned char x11 = (vector unsigned char)vec_sr(qxa1, v4);
                    vector unsigned char x20 = (vector unsigned char)vec_and(qxa2, lowMask);
                    vector unsigned char x21 = (vector unsigned char)vec_sr(qxa2, v4);
                    vector unsigned char x30 = (vector unsigned char)vec_and(qxa3, lowMask);
                    vector unsigned char x31 = (vector unsigned char)vec_sr(qxa3, v4);
                    vector signed int qv00 = vec_msum(q8y00, x00, v0);
                    vector signed int qv01 = vec_msum(q8y01, x01, v0);
                    vector signed int qv10 = vec_msum(q8y10, x10, v0);
                    vector signed int qv11 = vec_msum(q8y11, x11, v0);
                    vector signed int qv20 = vec_msum(q8y20, x20, v0);
                    vector signed int qv21 = vec_msum(q8y21, x21, v0);
                    vector signed int qv30 = vec_msum(q8y30, x30, v0);
                    vector signed int qv31 = vec_msum(q8y31, x31, v0);
                    vector signed int vscales_h = vec_unpackh(vscalesa);
                    vector signed int vs0 = vec_splat(vscales_h, 0);
                    vector signed int vs1 = vec_splat(vscales_h, 1);
                    vector signed int vs2 = vec_splat(vscales_h, 2);
                    vector signed int vs3 = vec_splat(vscales_h, 3);
                    vscalesa = vec_sld(vscalesa, vscalesa, 8);
                    sia0 = vec_add(vec_mul(qv00, vs0), sia0);
                    sia1 = vec_add(vec_mul(qv01, vs1), sia1);
                    sia2 = vec_add(vec_mul(qv20, vs2), sia2);
                    sia3 = vec_add(vec_mul(qv21, vs3), sia3);
                    sia0 = vec_add(vec_mul(qv10, vs0), sia0);
                    sia1 = vec_add(vec_mul(qv11, vs1), sia1);
                    sia2 = vec_add(vec_mul(qv30, vs2), sia2);
                    sia3 = vec_add(vec_mul(qv31, vs3), sia3);
                }
                {
                    vector unsigned char x00 = (vector unsigned char)vec_and(qxb0, lowMask);
                    vector unsigned char x01 = (vector unsigned char)vec_sr(qxb0, v4);
                    vector unsigned char x10 = (vector unsigned char)vec_and(qxb1, lowMask);
                    vector unsigned char x11 = (vector unsigned char)vec_sr(qxb1, v4);
                    vector unsigned char x20 = (vector unsigned char)vec_and(qxb2, lowMask);
                    vector unsigned char x21 = (vector unsigned char)vec_sr(qxb2, v4);
                    vector unsigned char x30 = (vector unsigned char)vec_and(qxb3, lowMask);
                    vector unsigned char x31 = (vector unsigned char)vec_sr(qxb3, v4);
                    vector signed int qv00 = vec_msum(q8y00, x00, v0);
                    vector signed int qv01 = vec_msum(q8y01, x01, v0);
                    vector signed int qv10 = vec_msum(q8y10, x10, v0);
                    vector signed int qv11 = vec_msum(q8y11, x11, v0);
                    vector signed int qv20 = vec_msum(q8y20, x20, v0);
                    vector signed int qv21 = vec_msum(q8y21, x21, v0);
                    vector signed int qv30 = vec_msum(q8y30, x30, v0);
                    vector signed int qv31 = vec_msum(q8y31, x31, v0);
                    vector signed int vscales_h = vec_unpackh(vscalesb);
                    vector signed int vs0 = vec_splat(vscales_h, 0);
                    vector signed int vs1 = vec_splat(vscales_h, 1);
                    vector signed int vs2 = vec_splat(vscales_h, 2);
                    vector signed int vs3 = vec_splat(vscales_h, 3);
                    vscalesb = vec_sld(vscalesb, vscalesb, 8);
                    sib0 = vec_add(vec_mul(qv00, vs0), sib0);
                    sib1 = vec_add(vec_mul(qv01, vs1), sib1);
                    sib2 = vec_add(vec_mul(qv20, vs2), sib2);
                    sib3 = vec_add(vec_mul(qv21, vs3), sib3);
                    sib0 = vec_add(vec_mul(qv10, vs0), sib0);
                    sib1 = vec_add(vec_mul(qv11, vs1), sib1);
                    sib2 = vec_add(vec_mul(qv30, vs2), sib2);
                    sib3 = vec_add(vec_mul(qv31, vs3), sib3);
                }
            }

            sfa0 = vec_madd(vec_ctf(sia0, 0), vda, sfa0);
            sfa1 = vec_madd(vec_ctf(sia1, 0), vda, sfa1);
            sfa2 = vec_madd(vec_ctf(sia2, 0), vda, sfa2);
            sfa3 = vec_madd(vec_ctf(sia3, 0), vda, sfa3);
            sfb0 = vec_madd(vec_ctf(sib0, 0), vdb, sfb0);
            sfb1 = vec_madd(vec_ctf(sib1, 0), vdb, sfb1);
            sfb2 = vec_madd(vec_ctf(sib2, 0), vdb, sfb2);
            sfb3 = vec_madd(vec_ctf(sib3, 0), vdb, sfb3);
        }

        sfa0 = vec_add(sfa0, sfa2);
        sfa1 = vec_add(sfa1, sfa3);
        sfa0 = vec_add(sfa0, sfa1);
        sfa0 = vec_add(sfa0, vec_sld(sfa0, sfa0, 4));
        sfa0 = vec_add(sfa0, vec_sld(sfa0, sfa0, 8));
        C[r] = vec_extract(sfa0, 0);

        if (two) {
            sfb0 = vec_add(sfb0, sfb2);
            sfb1 = vec_add(sfb1, sfb3);
            sfb0 = vec_add(sfb0, sfb1);
            sfb0 = vec_add(sfb0, vec_sld(sfb0, sfb0, 4));
            sfb0 = vec_add(sfb0, vec_sld(sfb0, sfb0, 8));
            C[r + 1] = vec_extract(sfb0, 0);
        }
    }
}



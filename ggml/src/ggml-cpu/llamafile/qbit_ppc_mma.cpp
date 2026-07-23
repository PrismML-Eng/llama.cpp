// qbit_ppc_mma.cpp - imported from github.com/mavin2009/ppc-mma-kernels
// (v4 production API, standalone-verified under qemu -cpu power10).
// Q1_0 / Q2_0 x Q8_0 on POWER10/POWER11 MMA, with the tensor-keyed
// pack cache and a no-packing GEMV fast path for n == 1.

#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "kquants_ppc_mma.h"

#include <altivec.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>







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


#define KC_BLKS   16
#define KC_CHUNKS (KC_BLKS * 4)
#define MR        16
#define NR        8
#define QBIT_GEMV_NMAX 2   // n <= this uses the raw-weight GEMV path

typedef struct {
    vuc v[KC_CHUNKS][32];
    vfl dA[KC_BLKS][4];
} apack_t;

typedef struct {
    vuc v[KC_CHUNKS][16];
    vfl dB[KC_CHUNKS][2];
    vfl E[KC_BLKS][2];
} bpack_t;

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

static inline void q1_codes32(vuc raw, int c, vuc v[2]) {
    const vuc rep0 = { 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1 };
    const vuc rep1 = { 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3 };
    const vuc sh   = { 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7 };
    const vuc m1   = vec_splats((unsigned char)1);
    const vuc off  = vec_splats((unsigned char)(4*c));
    v[0] = vec_and(vec_sr(vec_perm(raw, raw, vec_add(rep0, off)), sh), m1);
    v[1] = vec_and(vec_sr(vec_perm(raw, raw, vec_add(rep1, off)), sh), m1);
}

static inline void q2_codes32(vuc lo, vuc hi, int c, vuc v[2]) {
    const vuc rep0 = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };
    const vuc rep1 = { 4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7 };
    const vuc sh   = { 0,2,4,6, 0,2,4,6, 0,2,4,6, 0,2,4,6 };
    const vuc m3   = vec_splats((unsigned char)3);
    const vuc off  = vec_splats((unsigned char)(8*c));
    v[0] = vec_and(vec_sr(vec_perm(lo, hi, vec_add(rep0, off)), sh), m3);
    v[1] = vec_and(vec_sr(vec_perm(lo, hi, vec_add(rep1, off)), sh), m3);
}

// ---------------- one-time packing API ----------------

static inline int64_t n_row_tiles(int64_t m)  { return (m + MR - 1) / MR; }
static inline int64_t n_col_tiles(int64_t n)  { return (n + NR - 1) / NR; }
static inline int64_t n_slabs(int64_t k)      { return (k/128 + KC_BLKS - 1) / KC_BLKS; }

extern "C" size_t qbit_apack_size(int64_t m, int64_t k) {
    return (((size_t)(n_row_tiles(m) * n_slabs(k)) * sizeof(apack_t)) + 63) & ~(size_t)63;
}
extern "C" size_t qbit_bpack_size(int64_t n, int64_t k) {
    return (((size_t)(n_col_tiles(n) * n_slabs(k)) * sizeof(bpack_t)) + 63) & ~(size_t)63;
}

// packed layout: tile-major, slab-minor:  P[tile * n_slabs + slab]

template <typename BLK, void (*CODES)(const BLK *, int, vuc[2])>
static void repack_rows(const BLK * A, int64_t lda, int64_t m, int64_t k, apack_t * P) {
    const int64_t kb = k / 128, ns = n_slabs(k);
    for (int64_t it = 0; it < n_row_tiles(m); it++) {
        for (int64_t s = 0; s < ns; s++) {
            apack_t * T = &P[it*ns + s];
            const int64_t blk0 = s*KC_BLKS;
            const int64_t nblk = (kb - blk0) < KC_BLKS ? (kb - blk0) : KC_BLKS;
            for (int64_t b = 0; b < nblk; b++) {
                const BLK * bp[MR]; float d[MR];
                for (int r = 0; r < MR; r++) {
                    int64_t rr = it*MR + r; if (rr >= m) rr = m - 1;
                    bp[r] = &A[rr*lda + blk0 + b];
                    d[r]  = GGML_FP16_TO_FP32(bp[r]->d);
                }
                for (int g = 0; g < 4; g++)
                    T->dA[b][g] = (vfl){ d[4*g], d[4*g+1], d[4*g+2], d[4*g+3] };
                for (int c = 0; c < 4; c++) {
                    vuc t[MR][2];
                    for (int r = 0; r < MR; r++) CODES(bp[r], c, t[r]);
                    vui rows4[4];
                    for (int g = 0; g < 4; g++)
                        for (int h = 0; h < 2; h++) {
                            for (int r = 0; r < 4; r++) rows4[r] = (vui)t[4*g + r][h];
                            mma_transpose4(rows4, &T->v[4*b + c][16*h + g], 4);
                        }
                }
            }
        }
    }
}

static void q1_blk_codes(const block_q1_0 * bp, int c, vuc v[2]) {
    q1_codes32(load16u((const uint8_t *)(bp->qs) + (0)), c, v);
}
static void q2_blk_codes(const block_q2_0 * bp, int c, vuc v[2]) {
    q2_codes32(load16u((const uint8_t *)(bp->qs) + (0)),
               load16u((const uint8_t *)(bp->qs) + (16)), c, v);
}

extern "C" void qbit_repack_q1(const block_q1_0 * A, int64_t lda,
                               int64_t m, int64_t k, void * packed) {
    repack_rows<block_q1_0, q1_blk_codes>(A, lda, m, k, (apack_t *)packed);
}
extern "C" void qbit_repack_q2(const block_q2_0 * A, int64_t lda,
                               int64_t m, int64_t k, void * packed) {
    repack_rows<block_q2_0, q2_blk_codes>(A, lda, m, k, (apack_t *)packed);
}

// Pack all of B once; call from one thread (or split by col tile).
extern "C" void qbit_pack_b(const block_q8_0 * B, int64_t ldb,
                            int64_t n, int64_t k, void * packed) {
    bpack_t * P = (bpack_t *)packed;
    const int64_t kb = k / 128, ns = n_slabs(k);
    for (int64_t jt = 0; jt < n_col_tiles(n); jt++) {
        for (int64_t s = 0; s < ns; s++) {
            bpack_t * T = &P[jt*ns + s];
            const int64_t blk0 = s*KC_BLKS;
            const int64_t nblk = (kb - blk0) < KC_BLKS ? (kb - blk0) : KC_BLKS;
            for (int64_t b = 0; b < nblk; b++) {
                vfl E0 = vec_splats(0.0f), E1 = vec_splats(0.0f);
                for (int c = 0; c < 4; c++) {
                    const int64_t ch = 4*b + c;
                    const block_q8_0 * yb[NR];
                    float dB[NR], S[NR];
                    for (int j = 0; j < NR; j++) {
                        int64_t jj = jt*NR + j; if (jj >= n) jj = n - 1;
                        yb[j] = &B[jj*ldb + 4*(blk0 + b) + c];
                        dB[j] = GGML_FP16_TO_FP32(yb[j]->d);
                    }
                    vui rows4[4];
                    for (int a = 0; a < 2; a++) {
                        vuc q[4][2];
                        for (int j = 0; j < 4; j++) {
                            q[j][0] = load16u((const uint8_t *)(yb[4*a + j]->qs) + (0));
                            q[j][1] = load16u((const uint8_t *)(yb[4*a + j]->qs) + (16));
                            vsi z = vec_splats(0);
                            vsi sm = vec_sum4s((vsc)q[j][0], z);
                            sm = vec_sum4s((vsc)q[j][1], sm);
                            S[4*a + j] = (float)(sm[0] + sm[1] + sm[2] + sm[3]);
                        }
                        for (int h = 0; h < 2; h++) {
                            for (int j = 0; j < 4; j++) rows4[j] = (vui)q[j][h];
                            mma_transpose4(rows4, &T->v[ch][8*h + a], 2);
                        }
                    }
                    vfl dB0 = (vfl){ dB[0], dB[1], dB[2], dB[3] };
                    vfl dB1 = (vfl){ dB[4], dB[5], dB[6], dB[7] };
                    T->dB[ch][0] = dB0; T->dB[ch][1] = dB1;
                    E0 = vec_madd(dB0, (vfl){ S[0],S[1],S[2],S[3] }, E0);
                    E1 = vec_madd(dB1, (vfl){ S[4],S[5],S[6],S[7] }, E1);
                }
                T->E[b][0] = E0; T->E[b][1] = E1;
            }
        }
    }
}

// ---------------- GEMM from packed operands ----------------

static void kernel_16x8(const apack_t * PA, const bpack_t * PB,
                        int64_t nblk, float alpha, vfl fin[NR][4]) {
    const vfl valpha = vec_splats(alpha);
    for (int64_t b = 0; b < nblk; b++) {
        for (int c = 0; c < 4; c++) {
            const int64_t ch = 4*b + c;
            const vuc * a = PA->v[ch];
            const vuc * y = PB->v[ch];
            if (ch + 1 < 4*nblk) {
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
            const vfl dBa0 = vec_mul(PB->dB[ch][0], valpha);
            const vfl dBa1 = vec_mul(PB->dB[ch][1], valpha);
            for (int i = 0; i < 2; i++) {
                const vfl dBa = i ? dBa1 : dBa0;
                for (int g = 0; g < 4; g++) {
                    vsi rowsP[4];
                    __builtin_mma_disassemble_acc(rowsP, &acc[i][g]);
                    const vfl s0 = vec_mul(vec_splat(dBa, 0), PA->dA[b][g]);
                    const vfl s1 = vec_mul(vec_splat(dBa, 1), PA->dA[b][g]);
                    const vfl s2 = vec_mul(vec_splat(dBa, 2), PA->dA[b][g]);
                    const vfl s3 = vec_mul(vec_splat(dBa, 3), PA->dA[b][g]);
                    fin[4*i + 0][g] = vec_madd(vec_ctf(rowsP[0], 0), s0, fin[4*i + 0][g]);
                    fin[4*i + 1][g] = vec_madd(vec_ctf(rowsP[1], 0), s1, fin[4*i + 1][g]);
                    fin[4*i + 2][g] = vec_madd(vec_ctf(rowsP[2], 0), s2, fin[4*i + 2][g]);
                    fin[4*i + 3][g] = vec_madd(vec_ctf(rowsP[3], 0), s3, fin[4*i + 3][g]);
                }
            }
        }
    }
    for (int64_t b = 0; b < nblk; b++)
        for (int g = 0; g < 4; g++) {
            const vfl dA = PA->dA[b][g];
            fin[0][g] = vec_nmsub(dA, vec_splat(PB->E[b][0], 0), fin[0][g]);
            fin[1][g] = vec_nmsub(dA, vec_splat(PB->E[b][0], 1), fin[1][g]);
            fin[2][g] = vec_nmsub(dA, vec_splat(PB->E[b][0], 2), fin[2][g]);
            fin[3][g] = vec_nmsub(dA, vec_splat(PB->E[b][0], 3), fin[3][g]);
            fin[4][g] = vec_nmsub(dA, vec_splat(PB->E[b][1], 0), fin[4][g]);
            fin[5][g] = vec_nmsub(dA, vec_splat(PB->E[b][1], 1), fin[5][g]);
            fin[6][g] = vec_nmsub(dA, vec_splat(PB->E[b][1], 2), fin[6][g]);
            fin[7][g] = vec_nmsub(dA, vec_splat(PB->E[b][1], 3), fin[7][g]);
        }
}

// GEMM over pre-packed operands.  packedA from qbit_repack_*,
// packedB from qbit_pack_b.  Threads split row tiles.
extern "C" void qbit_gemm_packed(int64_t m, int64_t n, int64_t k, float alpha,
                                 const void * packedA, const void * packedB,
                                 float * C, int64_t ldc, int ith, int nth) {
    const apack_t * PA = (const apack_t *)packedA;
    const bpack_t * PB = (const bpack_t *)packedB;
    const int64_t kb = k/128, ns = n_slabs(k), mt = n_row_tiles(m), njt = n_col_tiles(n);
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
                const int64_t blk0 = s*KC_BLKS;
                const int64_t nblk = (kb - blk0) < KC_BLKS ? (kb - blk0) : KC_BLKS;
                kernel_16x8(&PA[it*ns + s], &PB[jt*ns + s], nblk, alpha, fin);
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

// ---------------- GEMV path (n <= QBIT_GEMV_NMAX), raw weights ----------------

typedef struct { float dB; float S; } gemv_bmeta_t;   // per chunk

static void gemv_prep_b(const block_q8_0 * y, int64_t nch, gemv_bmeta_t * M) {
    for (int64_t c = 0; c < nch; c++) {
        M[c].dB = GGML_FP16_TO_FP32(y[c].d);
        vsi z = vec_splats(0);
        vsi s = vec_sum4s((vsc)load16u((const uint8_t *)(y[c].qs) + (0)), z);
        s = vec_sum4s((vsc)load16u((const uint8_t *)(y[c].qs) + (16)), s);
        M[c].S = (float)(s[0] + s[1] + s[2] + s[3]);
    }
}

static inline int hsum(vsi s) { return s[0] + s[1] + s[2] + s[3]; }

static float gemv_row_q1(const block_q1_0 * a, const block_q8_0 * y,
                         const gemv_bmeta_t * M, int64_t kb) {
    const vuc rep0 = { 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1 };
    const vuc rep1 = { 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3 };
    const vuc bitsel = { 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128 };
    float sumf = 0.0f;
    for (int64_t b = 0; b < kb; b++) {
        const float dA = GGML_FP16_TO_FP32(a[b].d);
        vuc raw = load16u((const uint8_t *)(a[b].qs) + (0));
        for (int c = 0; c < 4; c++) {
            const vuc off = vec_splats((unsigned char)(4*c));
            vuc e0 = vec_perm(raw, raw, vec_add(rep0, off));
            vuc e1 = vec_perm(raw, raw, vec_add(rep1, off));
            vuc m0 = (vuc)vec_cmpeq(vec_and(e0, bitsel), bitsel);
            vuc m1 = (vuc)vec_cmpeq(vec_and(e1, bitsel), bitsel);
            const int8_t * q = y[4*b + c].qs;
            vuc y0 = load16u((const uint8_t *)(q) + (0));
            vuc y1 = load16u((const uint8_t *)(q) + (16));
            vsi z = vec_splats(0);
            vsi p = vec_sum4s((vsc)vec_and(y0, m0), z);
            p = vec_sum4s((vsc)vec_and(y1, m1), p);
            const gemv_bmeta_t * mm = &M[4*b + c];
            sumf += dA * mm->dB * (2.0f*(float)hsum(p) - mm->S);
        }
    }
    return sumf;
}

static float gemv_row_q2(const block_q2_0 * a, const block_q8_0 * y,
                         const gemv_bmeta_t * M, int64_t kb) {
    const vuc rep0 = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };
    const vuc rep1 = { 4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7 };
    const vuc sel0 = { 1,4,16,64, 1,4,16,64, 1,4,16,64, 1,4,16,64 };
    const vuc sel1 = { 2,8,32,128, 2,8,32,128, 2,8,32,128, 2,8,32,128 };
    float sumf = 0.0f;
    for (int64_t b = 0; b < kb; b++) {
        const float dA = GGML_FP16_TO_FP32(a[b].d);
        vuc lo = load16u((const uint8_t *)(a[b].qs) + (0));
        vuc hi = load16u((const uint8_t *)(a[b].qs) + (16));
        for (int c = 0; c < 4; c++) {
            const vuc off = vec_splats((unsigned char)(8*c));
            vuc e0 = vec_perm(lo, hi, vec_add(rep0, off));
            vuc e1 = vec_perm(lo, hi, vec_add(rep1, off));
            vuc ma0 = (vuc)vec_cmpeq(vec_and(e0, sel0), sel0);
            vuc mb0 = (vuc)vec_cmpeq(vec_and(e0, sel1), sel1);
            vuc ma1 = (vuc)vec_cmpeq(vec_and(e1, sel0), sel0);
            vuc mb1 = (vuc)vec_cmpeq(vec_and(e1, sel1), sel1);
            const int8_t * q = y[4*b + c].qs;
            vuc y0 = load16u((const uint8_t *)(q) + (0));
            vuc y1 = load16u((const uint8_t *)(q) + (16));
            vsi z = vec_splats(0);
            vsi p0 = vec_sum4s((vsc)vec_and(y0, ma0), z);
            p0 = vec_sum4s((vsc)vec_and(y1, ma1), p0);
            vsi p1 = vec_sum4s((vsc)vec_and(y0, mb0), z);
            p1 = vec_sum4s((vsc)vec_and(y1, mb1), p1);
            const gemv_bmeta_t * mm = &M[4*b + c];
            sumf += dA * mm->dB * ((float)hsum(p0) + 2.0f*(float)hsum(p1) - mm->S);
        }
    }
    return sumf;
}

extern "C" void qbit_gemv_q1(int64_t m, int64_t n, int64_t k,
        const block_q1_0 * A, int64_t lda, const block_q8_0 * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth) {
    const int64_t kb = k/128;
    gemv_bmeta_t * M = (gemv_bmeta_t *)malloc(sizeof(gemv_bmeta_t)*4*kb);
    const int64_t rpt = (m + nth - 1)/nth, i0 = ith*rpt, i1 = (ith+1)*rpt < m ? (ith+1)*rpt : m;
    for (int64_t j = 0; j < n; j++) {
        gemv_prep_b(B + j*ldb, 4*kb, M);
        for (int64_t i = i0; i < i1; i++)
            C[i + j*ldc] = gemv_row_q1(A + i*lda, B + j*ldb, M, kb);
    }
    free(M);
}

extern "C" void qbit_gemv_q2(int64_t m, int64_t n, int64_t k,
        const block_q2_0 * A, int64_t lda, const block_q8_0 * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth) {
    const int64_t kb = k/128;
    gemv_bmeta_t * M = (gemv_bmeta_t *)malloc(sizeof(gemv_bmeta_t)*4*kb);
    const int64_t rpt = (m + nth - 1)/nth, i0 = ith*rpt, i1 = (ith+1)*rpt < m ? (ith+1)*rpt : m;
    for (int64_t j = 0; j < n; j++) {
        gemv_prep_b(B + j*ldb, 4*kb, M);
        for (int64_t i = i0; i < i1; i++)
            C[i + j*ldc] = gemv_row_q2(A + i*lda, B + j*ldb, M, kb);
    }
    free(M);
}


// dispatch entries (names preserved from the original v3 integration):
// n == 1 takes the GEMV path -- no packing, no cache, mask-select and
// vsum4s only; larger n uses the packed GEMM through the pack cache.
#define QBIT_DRIVER(NAME, BLKA, REPACK, GEMV, ALPHA, VARIANT)                  \
extern "C" void NAME(int64_t m, int64_t n, int64_t k,                          \
        const void * Av, int64_t lda, const void * Bv, int64_t ldb,            \
        float * C, int64_t ldc, int ith, int nth) {                            \
    const BLKA * A = (const BLKA *)Av;                                         \
    const block_q8_0 * B = (const block_q8_0 *)Bv;                             \
    if (n == 1) { GEMV(m, n, k, A, lda, B, ldb, C, ldc, ith, nth); return; }   \
    void * PB = aligned_alloc(64, qbit_bpack_size(n, k));                      \
    if (!PB) { GGML_ABORT("ppc-mma: pack alloc failed"); }                     \
    qbit_pack_b(B, ldb, n, k, PB);                                             \
    int fresh = 0;                                                             \
    void * PA = ppc_apack_cache_acquire(Av, m, k, VARIANT,                     \
                                        qbit_apack_size(m, k), &fresh);        \
    if (PA) {                                                                  \
        if (fresh) { REPACK(A, lda, m, k, PA);                                 \
                     ppc_apack_cache_publish(Av, m, k, VARIANT); }             \
        qbit_gemm_packed(m, n, k, ALPHA, PA, PB, C, ldc, ith, nth);            \
    } else {                                                                   \
        void * PT = aligned_alloc(64, qbit_apack_size(MR, k));                 \
        if (!PT) { GGML_ABORT("ppc-mma: pack alloc failed"); }                 \
        const int64_t mt = (m + MR - 1) / MR;                                  \
        const int64_t tpt = (mt + nth - 1) / nth;                              \
        for (int64_t it = ith*tpt; it < (ith+1)*tpt && it < mt; it++) {        \
            const int64_t i = it*MR;                                           \
            const int64_t rows = (m - i) < MR ? (m - i) : MR;                  \
            REPACK(A + i*lda, lda, rows, k, PT);                               \
            qbit_gemm_packed(rows, n, k, ALPHA, PT, PB, C + i, ldc, 0, 1);     \
        }                                                                      \
        free(PT);                                                              \
    }                                                                          \
    free(PB);                                                                  \
}
QBIT_DRIVER(gemm_q1_0_q8_0_ppc_v3, block_q1_0, qbit_repack_q1, qbit_gemv_q1, 2.0f, 30)
QBIT_DRIVER(gemm_q2_0_q8_0_ppc_v3, block_q2_0, qbit_repack_q2, qbit_gemv_q2, 1.0f, 31)

#endif // __MMA__


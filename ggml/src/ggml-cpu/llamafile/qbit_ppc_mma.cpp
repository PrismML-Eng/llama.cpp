// qbit_ppc_mma.cpp - POWER10/POWER11 MMA GEMM for Q1_0/Q2_0 x Q8_0.
// Imported from https://github.com/mavin2009/ppc-mma-kernels (v3 kernel,
// correctness-verified against double references under qemu -cpu power10).
// See that repo's docs/DESIGN.md for the derivation.

#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "qbit_ppc_mma.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

#if defined(__MMA__) && defined(__powerpc64__)
#include <altivec.h>


typedef vector unsigned char  vuc;
typedef vector signed char    vsc;
typedef vector unsigned int   vui;
typedef vector signed int     vsi;
typedef vector float          vfl;

#define KC_BLKS   16                       // 2048-element K slab
#define KC_CHUNKS (KC_BLKS * 4)
#define MR        16                       // weight rows per tile
#define NR        8                        // activation cols per tile

// Packed weights: unsigned codes in MMA layout.
// v[ch][4*x + b]: depth step x (0..7), row group b (0..3).
typedef struct {
    vuc v[KC_CHUNKS][32];
    vfl dA[KC_BLKS][4];                    // per-block scales, 4 rows per vfl
} apack_t;

// Packed activations: signed int8, untouched.
// v[ch][2*x + a]: depth step x, col group a (0..1).
typedef struct {
    vuc v[KC_CHUNKS][16];
    vfl dB[KC_CHUNKS][2];                  // per-chunk scales, cols 0-3 / 4-7
    vfl E[KC_BLKS][2];                     // sum_c dB(j,c)*sum(y_c) per block
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

// ---- weight unpack: raw UNSIGNED codes (no sign mapping needed) ----

// Q1_0: chunk c from the block's 16 qs bytes -> bits 0/1 as bytes.
static inline void q1_codes32(vuc raw, int c, vuc v[2]) {
    const vuc rep0 = { 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1 };
    const vuc rep1 = { 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3 };
    const vuc sh   = { 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7 };
    const vuc m1   = vec_splats((unsigned char)1);
    const vuc off  = vec_splats((unsigned char)(4*c));
    v[0] = vec_and(vec_sr(vec_perm(raw, raw, vec_add(rep0, off)), sh), m1);
    v[1] = vec_and(vec_sr(vec_perm(raw, raw, vec_add(rep1, off)), sh), m1);
}

// Q2_0: chunk c from the block's 32 qs bytes (lo/hi) -> codes 0..3.
static inline void q2_codes32(vuc lo, vuc hi, int c, vuc v[2]) {
    const vuc rep0 = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };
    const vuc rep1 = { 4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7 };
    const vuc sh   = { 0,2,4,6, 0,2,4,6, 0,2,4,6, 0,2,4,6 };
    const vuc m3   = vec_splats((unsigned char)3);
    const vuc off  = vec_splats((unsigned char)(8*c));
    v[0] = vec_and(vec_sr(vec_perm(lo, hi, vec_add(rep0, off)), sh), m3);
    v[1] = vec_and(vec_sr(vec_perm(lo, hi, vec_add(rep1, off)), sh), m3);
}

// ---- packers ----

static void pack_A16_q1(const block_q1_0 * A, int64_t lda, int64_t rows,
                        int64_t blk0, int64_t nblk, apack_t * P) {
    for (int64_t b = 0; b < nblk; b++) {
        vuc raw[MR]; float d[MR];
        for (int r = 0; r < MR; r++) {
            int64_t rr = r < rows ? r : rows - 1;
            const block_q1_0 * bp = &A[rr*lda + blk0 + b];
            d[r]   = GGML_FP16_TO_FP32(bp->d);
            raw[r] = vec_xl(0, (const unsigned char *)bp->qs);
        }
        for (int g = 0; g < 4; g++)
            P->dA[b][g] = (vfl){ d[4*g], d[4*g+1], d[4*g+2], d[4*g+3] };
        for (int c = 0; c < 4; c++) {
            const int ch = 4*b + c;
            vuc t[MR][2];
            for (int r = 0; r < MR; r++) q1_codes32(raw[r], c, t[r]);
            vui rows4[4];
            for (int g = 0; g < 4; g++)
                for (int h = 0; h < 2; h++) {
                    for (int r = 0; r < 4; r++) rows4[r] = (vui)t[4*g + r][h];
                    mma_transpose4(rows4, &P->v[ch][16*h + g], 4);
                }
        }
    }
}

static void pack_A16_q2(const block_q2_0 * A, int64_t lda, int64_t rows,
                        int64_t blk0, int64_t nblk, apack_t * P) {
    for (int64_t b = 0; b < nblk; b++) {
        vuc lo[MR], hi[MR]; float d[MR];
        for (int r = 0; r < MR; r++) {
            int64_t rr = r < rows ? r : rows - 1;
            const block_q2_0 * bp = &A[rr*lda + blk0 + b];
            d[r]  = GGML_FP16_TO_FP32(bp->d);
            lo[r] = vec_xl(0,  (const unsigned char *)bp->qs);
            hi[r] = vec_xl(16, (const unsigned char *)bp->qs);
        }
        for (int g = 0; g < 4; g++)
            P->dA[b][g] = (vfl){ d[4*g], d[4*g+1], d[4*g+2], d[4*g+3] };
        for (int c = 0; c < 4; c++) {
            const int ch = 4*b + c;
            vuc t[MR][2];
            for (int r = 0; r < MR; r++) q2_codes32(lo[r], hi[r], c, t[r]);
            vui rows4[4];
            for (int g = 0; g < 4; g++)
                for (int h = 0; h < 2; h++) {
                    for (int r = 0; r < 4; r++) rows4[r] = (vui)t[4*g + r][h];
                    mma_transpose4(rows4, &P->v[ch][16*h + g], 4);
                }
        }
    }
}

// Activations: no flip; also builds per-chunk scales and the separable
// correction accumulators E.
static void pack_B8(const block_q8_0 * B, int64_t ldb, int64_t cols,
                    int64_t q8blk0, int64_t nblk, bpack_t * P) {
    for (int64_t b = 0; b < nblk; b++) {
        vfl E0 = vec_splats(0.0f), E1 = vec_splats(0.0f);
        for (int c = 0; c < 4; c++) {
            const int64_t ch = 4*b + c;
            const block_q8_0 * yb[NR];
            float dB[NR], S[NR];
            for (int j = 0; j < NR; j++) {
                int64_t jj = j < cols ? j : cols - 1;
                yb[j] = &B[jj*ldb + q8blk0 + ch];
                dB[j] = GGML_FP16_TO_FP32(yb[j]->d);
            }
            vui rows4[4];
            for (int a = 0; a < 2; a++) {
                vuc q[4][2];
                for (int j = 0; j < 4; j++) {
                    q[j][0] = vec_xl(0,  (const unsigned char *)yb[4*a + j]->qs);
                    q[j][1] = vec_xl(16, (const unsigned char *)yb[4*a + j]->qs);
                    vsi z = vec_splats(0);
                    vsi s = vec_sum4s((vsc)q[j][0], z);
                    s = vec_sum4s((vsc)q[j][1], s);
                    S[4*a + j] = (float)(s[0] + s[1] + s[2] + s[3]);
                }
                for (int h = 0; h < 2; h++) {
                    for (int j = 0; j < 4; j++) rows4[j] = (vui)q[j][h];
                    mma_transpose4(rows4, &P->v[ch][8*h + a], 2);
                }
            }
            vfl dB0 = (vfl){ dB[0], dB[1], dB[2], dB[3] };
            vfl dB1 = (vfl){ dB[4], dB[5], dB[6], dB[7] };
            P->dB[ch][0] = dB0;
            P->dB[ch][1] = dB1;
            E0 = vec_madd(dB0, (vfl){ S[0], S[1], S[2], S[3] }, E0);
            E1 = vec_madd(dB1, (vfl){ S[4], S[5], S[6], S[7] }, E1);
        }
        P->E[b][0] = E0;
        P->E[b][1] = E1;
    }
}

// ---- 16x8 microkernel on all 8 accumulators ----
// fin[j][g]: activation col j (0..7), weight row group g (0..3).
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
            // fixup: fin += alpha*dB_j * dA_g * P
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
    // separable correction: fin[j][g] -= sum_b dA[b][g] * E(j,b)
    for (int64_t b = 0; b < nblk; b++) {
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
}

// ---- driver, templated over weight type ----
template <typename BLK, void (*PACK_A)(const BLK *, int64_t, int64_t, int64_t, int64_t, apack_t *), int ALPHA>
static void gemm_qbit(int64_t m, int64_t n, int64_t k,
                      const BLK * A, int64_t lda,
                      const block_q8_0 * B, int64_t ldb,
                      float * C, int64_t ldc, int ith, int nth) {
    assert(k % 128 == 0);
    const int64_t kb  = k / 128;
    const int64_t mt  = (m + MR - 1) / MR;
    const int64_t tpt = (mt + nth - 1) / nth;
    const int64_t i0  = MR * (ith * tpt);
    const int64_t i1  = m < MR * ((ith + 1) * tpt) ? m : MR * ((ith + 1) * tpt);
    if (i0 >= m) return;

    const int64_t njt = (n + NR - 1) / NR;
    apack_t * PA = (apack_t*)aligned_alloc(64, sizeof(apack_t));
    bpack_t * PB = (bpack_t*)aligned_alloc(64, njt * sizeof(bpack_t));
    vfl (*fin)[NR][4] = (vfl(*)[NR][4])aligned_alloc(64, njt * sizeof(vfl[NR][4]));

    for (int64_t i = i0; i < i1; i += MR) {
        const int64_t rows = (i1 - i) < MR ? (i1 - i) : MR;
        for (int64_t jt = 0; jt < njt; jt++)
            for (int j = 0; j < NR; j++)
                for (int g = 0; g < 4; g++)
                    fin[jt][j][g] = vec_splats(0.0f);

        for (int64_t blk0 = 0; blk0 < kb; blk0 += KC_BLKS) {
            const int64_t nblk = (kb - blk0) < KC_BLKS ? (kb - blk0) : KC_BLKS;
            PACK_A(A + i*lda, lda, rows, blk0, nblk, PA);
            for (int64_t jt = 0; jt < njt; jt++) {
                const int64_t j    = NR*jt;
                const int64_t cols = (n - j) < NR ? (n - j) : NR;
                if (i == i0 || kb > KC_BLKS)
                    pack_B8(B + j*ldb, ldb, cols, 4*blk0, nblk, &PB[jt]);
                kernel_16x8(PA, &PB[jt], nblk, (float)ALPHA, fin[jt]);
            }
        }

        for (int64_t jt = 0; jt < njt; jt++) {
            const int64_t j    = NR*jt;
            const int64_t cols = (n - j) < NR ? (n - j) : NR;
            for (int64_t cj = 0; cj < cols; cj++) {
                float * dst = C + i + (j + cj)*ldc;
                if (rows == MR) {
                    for (int g = 0; g < 4; g++)
                        vec_xst(fin[jt][cj][g], 16*g, dst);
                } else {
                    for (int64_t r = 0; r < rows; r++)
                        dst[r] = fin[jt][cj][r >> 2][r & 3];
                }
            }
        }
    }
    free(PA); free(PB); free(fin);
}


extern "C" void gemm_q1_0_q8_0_ppc_v3(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth) {
    gemm_qbit<block_q1_0, pack_A16_q1, 2>(m, n, k,
        (const block_q1_0 *)A, lda, (const block_q8_0 *)B, ldb, C, ldc, ith, nth);
}

extern "C" void gemm_q2_0_q8_0_ppc_v3(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth) {
    gemm_qbit<block_q2_0, pack_A16_q2, 1>(m, n, k,
        (const block_q2_0 *)A, lda, (const block_q8_0 *)B, ldb, C, ldc, ith, nth);
}

#endif // __MMA__ && __powerpc64__

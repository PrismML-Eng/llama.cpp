// PTQ1_0 mat-vec inner loop on a planar-transposed Q8_1 activation layout.
//
// Why: the stock mmvq path hands each thread one 128-weight PTQ1_0 block and
// walks the activations as 32 scattered 4-byte loads per column out of
// 36-byte block_q8_1 structs. The weight decode is shared across columns but
// the activation traffic is not, so every extra column costs a full pass
// (measured 2.2x at 2 columns, 3.3x at 3 on an RTX 3060). Here the activations
// are stored so that the 128 quants a thread needs are 8 aligned 16-byte
// pieces, one per plane, and the 4 (d, s) scales are one more 16-byte piece.
// Adjacent threads read adjacent 16-byte pieces of the same plane, so a warp
// load touches 4 cache lines instead of 32, and a thread reuses each piece
// for every row it owns. The decode, the dp4a sequence and the fp32 epilogue
// are the same operations in the same order as the single-column kernel, so
// every column count produces the same bits for a given column.
//
// PT layout, per activation column (all sizes for the padded row length):
//   plane t (t = 0..7):   nblk * 16 bytes, byte b of block kb is the quant of
//                         element kb*128 + t*16 + b
//   plane 8:              nblk * 16 bytes, block kb holds 4 half2 (d, s), one
//                         per 32-element sub-block
// Column stride is 9 * nblk * 16 = padded_row * 9/8 bytes, exactly the
// block_q8_1 stride (padded_row/32 blocks of 36 bytes), so every stride the
// mmvq launcher computes in block_q8_1 units stays valid.
#pragma once

#include "common.cuh"
#include "vecdotq.cuh"

#define PTQ1_0_PT_PLANES 9

// the PT path is CUDA only; HIP keeps the block_q8_1 layout and the old vec_dot
static constexpr __host__ __device__ bool ptq1_0_pt_enabled() {
#if defined(GGML_USE_HIP)
    return false;
#else
    return true;
#endif
}

// rows of the weight matrix one thread handles per K block: more rows reuse
// each activation piece more often, fewer rows keep the register count down
static constexpr __host__ __device__ int ptq1_0_pt_rows_per_block(const int ncols_dst) {
    return ncols_dst == 1 ? 1 : 2;
}

// number of 128-element blocks in a row padded to MATRIX_ROW_PADDING
static __host__ __device__ __forceinline__ int ptq1_0_pt_nblk(const int ncols_x) {
    return ((ncols_x + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * (MATRIX_ROW_PADDING / QK_PTQ1_0);
}

static __device__ __forceinline__ int int4_at(const int4 & v, const int k) {
    switch (k & 3) {
        case 0:  return v.x;
        case 1:  return v.y;
        case 2:  return v.z;
        default: return v.w;
    }
}

// four trits (0, 1, 2) packed as bytes -> the weights (-1, 0, 1) as signed bytes.
// t + 127 never carries out of its byte, and flipping the top bit maps
// 127, 128, 129 to -1, 0, 1: two integer ops instead of a byte-wise subtract.
static __device__ __forceinline__ int ptq1_0_trits_to_weights(const int q) {
    return (int) (((uint32_t) q + 0x7F7F7F7Fu) ^ 0x80808080u);
}

// one base-3 digit step on four bytes held as two 16-bit-lane words:
// returns the weights (-1, 0, 1) as four signed bytes, advances the remainders
static __device__ __forceinline__ int ptq1_0_trit_step(uint32_t & vlo, uint32_t & vhi) {
    const uint32_t wlo = vlo * 3;
    const uint32_t whi = vhi * 3;
    vlo = wlo & 0x00FF00FF;
    vhi = whi & 0x00FF00FF;
    return ptq1_0_trits_to_weights(__byte_perm(wlo, whi, 0x7531));
}

// Dot products of nrows PTQ1_0 blocks with the same block index of ncols
// activation columns. bq[i] points at the weight block of row i, ycol[j] at
// the PT column base of column j, kbx is the block index along K.
//
// The integer sum of each 32-element sub-block k is folded into the fp32
// accumulator as soon as the sub-block is complete, in the order k = 0..3,
// which is the expression acc = sum_k d8_k * sumi_k of the block_q8_1 kernel.
template <int ncols, int nrows>
static __device__ __forceinline__ void ptq1_0_pt_block_dot(
        const block_ptq1_0 * const (&bq)[nrows],
        const char * const (&ycol)[ncols],
        const int kbx, const int nblk,
        float (&result)[ncols][nrows]) {
    int4 dsraw[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        dsraw[j] = *((const int4 *) ycol[j] + 8*nblk + kbx);
    }

    int   sumi[ncols][nrows];
    float acc[ncols][nrows];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int i = 0; i < nrows; ++i) {
            sumi[j][i] = 0;
            acc[j][i]  = 0.0f;
        }
    }

    auto fold = [&](const int k) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float d8 = __low2float(((const half2 *) &dsraw[j])[k]);
#pragma unroll
            for (int i = 0; i < nrows; ++i) {
                acc[j][i] += d8 * (float) sumi[j][i];
                sumi[j][i] = 0;
            }
        }
    };

    // qs[0..15]: four groups of four bytes, five trits each: element 16*t + 4*g + b,
    // plane t holds words 4*t .. 4*t+3
    uint32_t vlo[nrows][4];
    uint32_t vhi[nrows][4];
#pragma unroll
    for (int i = 0; i < nrows; ++i) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t packed = get_int_b4(bq[i]->qs, g);
            vlo[i][g] = __byte_perm(packed, 0, 0x4140);
            vhi[i][g] = __byte_perm(packed, 0, 0x4342);
        }
    }
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        int4 u[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            u[j] = *((const int4 *) ycol[j] + t*nblk + kbx);
        }
#pragma unroll
        for (int i = 0; i < nrows; ++i) {
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const int q = ptq1_0_trit_step(vlo[i][g], vhi[i][g]);
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    sumi[j][i] = ggml_cuda_dp4a(q, int4_at(u[j], g), sumi[j][i]);
                }
            }
        }
        if (t == 1) {
            fold(0); // elements 0..31 done
        }
        if (t == 3) {
            fold(1); // elements 32..63 done
        }
    }

    // qs[16..23]: two groups of four bytes, five trits each: element 80 + 8*t + 4*g + b,
    // words 20..29 live in planes 5, 6 and the lower half of 7
    uint32_t vlo2[nrows][2];
    uint32_t vhi2[nrows][2];
#pragma unroll
    for (int i = 0; i < nrows; ++i) {
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const uint32_t packed = get_int_b4(bq[i]->qs + 16, g);
            vlo2[i][g] = __byte_perm(packed, 0, 0x4140);
            vhi2[i][g] = __byte_perm(packed, 0, 0x4342);
        }
    }
    int4 u2[ncols];
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        if (t % 2 == 0) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                u2[j] = *((const int4 *) ycol[j] + (5 + t/2)*nblk + kbx);
            }
        }
#pragma unroll
        for (int i = 0; i < nrows; ++i) {
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                const int q = ptq1_0_trit_step(vlo2[i][g], vhi2[i][g]);
                const int w = 20 + 2*t + g; // word index within the 128-element block
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    sumi[j][i] = ggml_cuda_dp4a(q, int4_at(u2[j], w & 3), sumi[j][i]);
                }
            }
        }
        if (t == 1) {
            fold(2); // elements 64..95 done (words 16..23)
        }
    }

    // qh: two bytes, four trits each, interleaved: element 120 + 2*t + h -> words 30, 31,
    // the upper half of plane 7 that u2 still holds
#pragma unroll
    for (int i = 0; i < nrows; ++i) {
        uint32_t v = (uint32_t) bq[i]->qh[0] | ((uint32_t) bq[i]->qh[1] << 16);
#pragma unroll
        for (int t = 0; t < 4; t += 2) {
            const uint32_t w0 = v * 3;
            v                 = w0 & 0x00FF00FF;
            const uint32_t w1 = v * 3;
            v                 = w1 & 0x00FF00FF;
            const int q = ptq1_0_trits_to_weights(__byte_perm(w0, w1, 0x7531));
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                sumi[j][i] = ggml_cuda_dp4a(q, int4_at(u2[j], 2 + t/2), sumi[j][i]);
            }
        }
    }
    fold(3); // elements 96..127 done

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int i = 0; i < nrows; ++i) {
            result[j][i] = (float) bq[i]->d * acc[j][i];
        }
    }
}

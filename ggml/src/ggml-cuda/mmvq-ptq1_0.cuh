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
#include "unary.cuh"
#include "vecdotq.cuh"

#define PTQ1_0_PT_PLANES 9

// dedicated 2D kernel geometry, see mul_mat_vec_ptq1_0_pt below
#define PTQ1_0_PT_THREADS      128
#define PTQ1_0_PT_MAX_ROWS     16
#define PTQ1_0_PT_MAX_COLS     8    // equals MMVQ_MAX_BATCH_SIZE, checked in mmvq.cu
#define PTQ1_0_PT_SMEM_FLOATS  4096 // 16 KiB of partial sums per weight matrix, dynamic

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

// one base-3 digit step on four bytes held as two 16-bit-lane words:
// returns the raw digits (0, 1, 2) as four unsigned bytes, advances the remainders.
// The digit bias is not applied here: the dot product uses the digits as-is with a mixed-sign
// dp4a and subtracts the exact integer activation sum of the 32-block once (see ptq1_0_pt_block_dot),
// which is the same integer as the biased sum, two ALU ops per 4 weights cheaper.
static __device__ __forceinline__ uint32_t ptq1_0_trit_step(uint32_t & vlo, uint32_t & vhi) {
    const uint32_t wlo = vlo * 3;
    const uint32_t whi = vhi * 3;
    vlo = wlo & 0x00FF00FF;
    vhi = whi & 0x00FF00FF;
    return __byte_perm(wlo, whi, 0x7531);
}

// Dot products of nrows PTQ1_0 blocks with the same block index of ncols
// activation columns. bq[i] points at the weight block of row i, ycol[j] at
// the PT column base of column j, kbx is the block index along K.
//
// The integer sum of each 32-element sub-block k is folded into the fp32
// accumulator as soon as the sub-block is complete, in the order k = 0..3,
// which is the expression acc = sum_k d8_k * sumi_k of the block_q8_1 kernel.
// sumi_k is accumulated from the raw digits {0,1,2} and corrected by the exact
// integer activation sum stored in the layout: sum((q-1)*a) = sum(q*a) - sum(a),
// exact in int32, so the result is bit-identical to the biased-weight form.
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
            const half2 ds   = ((const half2 *) &dsraw[j])[k];
            const float d8   = __low2float(ds);
            const int   isum = __half_as_short(__high2half(ds)); // exact sum of the 32 quantized activations
#pragma unroll
            for (int i = 0; i < nrows; ++i) {
                acc[j][i] = __fmaf_rn(d8, (float) (sumi[j][i] - isum), acc[j][i]); // one FFMA in every instantiation
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
                const uint32_t q = ptq1_0_trit_step(vlo[i][g], vhi[i][g]);
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    sumi[j][i] = ggml_cuda_dp4a_us(q, int4_at(u[j], g), sumi[j][i]);
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
                const uint32_t q = ptq1_0_trit_step(vlo2[i][g], vhi2[i][g]);
                const int w = 20 + 2*t + g; // word index within the 128-element block
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    sumi[j][i] = ggml_cuda_dp4a_us(q, int4_at(u2[j], w & 3), sumi[j][i]);
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
            const uint32_t q = __byte_perm(w0, w1, 0x7531);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                sumi[j][i] = ggml_cuda_dp4a_us(q, int4_at(u2[j], 2 + t/2), sumi[j][i]);
            }
        }
    }
    fold(3); // elements 96..127 done

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int i = 0; i < nrows; ++i) {
            result[j][i] = __fmul_rn((float) bq[i]->d, acc[j][i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Dedicated PTQ1_0 mat-vec for plain 2D MUL_MAT (no batch dims, no expert ids).
//
// The generic mmvq kernel gives every thread of a 128-thread block one
// 128-weight K block of the same row, so a K = 5120 projection (40 blocks per
// row) keeps 40 of 128 threads busy and K = 17408 (136 blocks) keeps 53%. Here
// the work items are (row group, K block) pairs of `rows_per_cta` rows
// flattened into one index space, with rows_per_cta chosen on the host so that
// the items fill whole 128-thread iterations where possible. A thread handles
// ROWS adjacent rows per item so that each activation piece it loads serves
// ROWS rows. Each thread writes one fp32 partial per (row, column, K block) to
// shared memory and one warp per (row, column) sums them in a fixed order
// (lane-strided sequential, then a butterfly). That order depends only on the
// weight shape, so the result for a column is the same bits for every column
// count.
// ---------------------------------------------------------------------------

// rows_per_cta: fill whole 128-thread iterations where possible, within the shared memory budget
static __host__ int ptq1_0_pt_rows_per_cta(const int blocks_per_row, const int ncols_dst, const int nrows_x, const int rows_per_item) {
    int rmax = PTQ1_0_PT_SMEM_FLOATS / (ncols_dst * (blocks_per_row + 1));
    rmax = rmax < rows_per_item ? rows_per_item : (rmax > PTQ1_0_PT_MAX_ROWS ? PTQ1_0_PT_MAX_ROWS : rmax);
    rmax -= rmax % rows_per_item;
    int best = rows_per_item;
    double best_util = 0.0;
    for (int r = rows_per_item; r <= rmax; r += rows_per_item) {
        const int items = (r / rows_per_item) * blocks_per_row;
        const int iters = (items + PTQ1_0_PT_THREADS - 1) / PTQ1_0_PT_THREADS;
        const double util = (double) items / (double) (iters * PTQ1_0_PT_THREADS);
        if (util > best_util + 1e-9) {
            best_util = util;
            best = r;
        }
        if (util > 0.999) {
            break;
        }
    }
    GGML_UNUSED(nrows_x);
    return best;
}

#ifndef PTQ1_0_PT_MINB_34
#define PTQ1_0_PT_MINB_34 3
#endif
#ifndef PTQ1_0_PT_ROWS_34
#define PTQ1_0_PT_ROWS_34 4
#endif

template <int ncols, int ROWS, bool has_fusion, bool has_gate>
__launch_bounds__(PTQ1_0_PT_THREADS, (ncols <= 2 ? 4 : (ncols <= 4 ? PTQ1_0_PT_MINB_34 : 2)))
static __global__ void mul_mat_vec_ptq1_0_pt(
        const void * GGML_CUDA_RESTRICT vx, const void * GGML_CUDA_RESTRICT vy, const ggml_cuda_mm_fusion_args_device fusion,
        float * GGML_CUDA_RESTRICT dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int rows_per_cta, const uint3 bpr_fd, const uint3 rpc_fd) {
    extern __shared__ float partials[];        // [ncols][rows_per_cta][bprp], then partials_gate
    const int bpr  = ncols_x / QK_PTQ1_0;      // K blocks per row
    const int bprp = bpr + 1;                  // partials row stride: odd, so the per-pair epilogue reads are bank-conflict-free
    [[maybe_unused]] float * partials_gate = partials + ncols*rows_per_cta*bprp;

    const int nblk = ptq1_0_pt_nblk(ncols_x);  // plane stride of the PT layout
    const int row0 = rows_per_cta * blockIdx.x;
    const int tid  = threadIdx.x;

    // rows this CTA really owns (the last CTA may be short); clamped item rows read the last real row
    const int n_rows_cta = min(rows_per_cta, nrows_x - row0);

    const char * ycol[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        ycol[j] = (const char *) ((const block_q8_1 *) vy + j*stride_col_y);
    }

    const int n_items = (rows_per_cta / ROWS) * bpr;
    for (int idx = tid; idx < n_items; idx += PTQ1_0_PT_THREADS) {
        const int rg  = fastdiv((uint32_t) idx, bpr_fd); // row group within the CTA
        const int kbx = idx - rg*bpr;

        const block_ptq1_0 * bq[ROWS];
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            int r = rg*ROWS + i;
            r = r < n_rows_cta ? r : n_rows_cta - 1; // clamp the tail, that result is not written
            bq[i] = (const block_ptq1_0 *) vx + (int64_t) (row0 + r)*stride_row_x + kbx;
        }
        float dots[ncols][ROWS];
        ptq1_0_pt_block_dot<ncols, ROWS>(bq, ycol, kbx, nblk, dots);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                partials[(j*rows_per_cta + rg*ROWS + i)*bprp + kbx] = dots[j][i];
            }
        }
        if constexpr (has_gate) {
            const block_ptq1_0 * bg[ROWS];
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                int r = rg*ROWS + i;
                r = r < n_rows_cta ? r : n_rows_cta - 1;
                bg[i] = (const block_ptq1_0 *) fusion.gate + (int64_t) (row0 + r)*stride_row_x + kbx;
            }
            ptq1_0_pt_block_dot<ncols, ROWS>(bg, ycol, kbx, nblk, dots);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
#pragma unroll
                for (int i = 0; i < ROWS; ++i) {
                    partials_gate[(j*rows_per_cta + rg*ROWS + i)*bprp + kbx] = dots[j][i];
                }
            }
        }
    }

    __syncthreads();

    // One thread per (row, column): a fixed-order sequential sum over the K blocks with four
    // interleaved accumulators (k mod 4), then (s0+s1)+(s2+s3). The order depends only on the weight
    // shape, so a column gives the same bits for every column count. Profiled against the previous
    // one-warp-per-pair shuffle reduction (RTX 4070, ncu): that epilogue was ~23% of all issued
    // instructions (a runtime integer division per pair, a strided loop, 5 dependent SHFL+FADD) and
    // held ~50% of all warp stall samples while the CTA did no memory traffic.
    for (int p = tid; p < rows_per_cta*ncols; p += PTQ1_0_PT_THREADS) {
        const int j   = fastdiv((uint32_t) p, rpc_fd); // column
        const int r   = p - j*rows_per_cta;             // row within the CTA
        const int row = row0 + r;
        const float * src = partials + (j*rows_per_cta + r)*bprp;

        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        [[maybe_unused]] float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
        int kbx = 0;
        for (; kbx + 4 <= bpr; kbx += 4) {
            s0 += src[kbx + 0]; s1 += src[kbx + 1]; s2 += src[kbx + 2]; s3 += src[kbx + 3];
            if constexpr (has_gate) {
                const float * sg = partials_gate + (j*rows_per_cta + r)*bprp;
                g0 += sg[kbx + 0]; g1 += sg[kbx + 1]; g2 += sg[kbx + 2]; g3 += sg[kbx + 3];
            }
        }
        for (; kbx < bpr; ++kbx) {
            s0 += src[kbx];
            if constexpr (has_gate) {
                g0 += partials_gate[(j*rows_per_cta + r)*bprp + kbx];
            }
        }
        const float sum = (s0 + s1) + (s2 + s3);
        [[maybe_unused]] const float sum_gate = (g0 + g1) + (g2 + g3);

        if (row < nrows_x) {
            float result = sum;
            if constexpr (has_fusion) {
                if (fusion.x_bias) {
                    result += ((const float *) fusion.x_bias)[j*stride_col_dst + row];
                }
                if constexpr (has_gate) {
                    float gate_value = sum_gate;
                    if (fusion.gate_bias) {
                        gate_value += ((const float *) fusion.gate_bias)[j*stride_col_dst + row];
                    }
                    switch (fusion.glu_op) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI:
                            result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                            break;
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
            }
            dst[j*stride_col_dst + row] = result;
        }
    }
}

template <int ncols>
static void mul_mat_vec_ptq1_0_pt_launch(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        cudaStream_t stream) {
    constexpr int ROWS = ncols <= 2 ? 4 : (ncols <= 4 ? PTQ1_0_PT_ROWS_34 : 2); // rows per work item: independent blocks per thread for latency hiding, activation reuse across rows
    const int bpr = ncols_x / QK_PTQ1_0;
    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const bool has_gate   = fusion.gate != nullptr;

    const int rows_per_cta = ptq1_0_pt_rows_per_cta(bpr, ncols, nrows_x, ROWS);
    const uint3 bpr_fd = init_fastdiv_values((uint32_t) bpr);
    const uint3 rpc_fd = init_fastdiv_values((uint32_t) rows_per_cta);
    const dim3 block_nums((nrows_x + rows_per_cta - 1) / rows_per_cta, 1, 1);
    const dim3 block_dims(PTQ1_0_PT_THREADS, 1, 1);

    const size_t smem = (size_t) ncols * rows_per_cta * (bpr + 1) * sizeof(float) * (has_gate ? 2 : 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, smem, stream);

#define PTQ1_0_PT_LAUNCH(FUS, GATE)                                                                                      \
    ggml_cuda_kernel_launch(mul_mat_vec_ptq1_0_pt<ncols, ROWS, FUS, GATE>, lp,                                           \
        vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_fd, rpc_fd)

    if (has_fusion) {
        GGML_ASSERT(ncols == 1 && "fusion only supported for ncols_dst=1");
        if (has_gate) {
            PTQ1_0_PT_LAUNCH(true, true);
        } else {
            PTQ1_0_PT_LAUNCH(true, false);
        }
        return;
    }
    PTQ1_0_PT_LAUNCH(false, false);
#undef PTQ1_0_PT_LAUNCH
}

// true when the dedicated kernel handles this call (plain 2D, K a multiple of 128, up to 8 columns)
static bool mul_mat_vec_ptq1_0_pt_switch(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {
    if (!ptq1_0_pt_enabled() || nchannels_dst != 1 || nsamples_dst != 1 || ncols_x % QK_PTQ1_0 != 0 ||
        ncols_dst < 1 || ncols_dst > PTQ1_0_PT_MAX_COLS || 2 * (ncols_x / QK_PTQ1_0) * ncols_dst > PTQ1_0_PT_SMEM_FLOATS) {
        return false;
    }
    switch (ncols_dst) {
        case 1: mul_mat_vec_ptq1_0_pt_launch<1>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 2: mul_mat_vec_ptq1_0_pt_launch<2>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_ptq1_0_pt_launch<3>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_ptq1_0_pt_launch<4>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_ptq1_0_pt_launch<5>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_ptq1_0_pt_launch<6>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_ptq1_0_pt_launch<7>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_ptq1_0_pt_launch<8>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        default: return false;
    }
    return true;
}

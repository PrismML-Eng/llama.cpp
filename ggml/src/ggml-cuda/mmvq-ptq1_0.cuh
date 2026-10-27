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
// for every row it owns. GGML_CUDA_BATCH_INVARIANT=1 keeps the older
// warp-reduce epilogue so a column verified in a 2-4 wide MTP batch matches
// the same column decoded alone. The faster four-accumulator epilogue is the
// default; it can flip a late near-tie (5080: "coastal waters" vs "coastal areas").
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
#define PTQ1_0_PT_SMEM_FLOATS  4096 // 16 KiB target when choosing rows per CTA; the launch may request more for one item

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
#ifndef PTQ1_0_PT_ROWS_1
#define PTQ1_0_PT_ROWS_1 1
#endif
#ifndef PTQ1_0_PT_ROWS_34
#define PTQ1_0_PT_ROWS_34 4
#endif

// Same row-per-item rule the launcher instantiates. Multi-column paths retain
// their existing 4/2-row schedule.
static constexpr __host__ __device__ int ptq1_0_pt_rows_per_item(const int ncols_dst) {
    return ncols_dst <= 2 ? 4 : (ncols_dst <= 4 ? PTQ1_0_PT_ROWS_34 : 2);
}

// The one-row tile is tuned on sm_86. Other devices keep the four-row baseline,
// including Hopper when GGML_CUDA_BATCH_INVARIANT selects this planar path.
static __host__ int ptq1_0_pt_rows_per_item_for_cc(const int ncols_dst, const int cc) {
    if (ncols_dst == 1 && GGML_CUDA_CC_IS_NVIDIA(cc) && cc == 860) {
        return PTQ1_0_PT_ROWS_1;
    }
    return ptq1_0_pt_rows_per_item(ncols_dst);
}

// Bytes the launch requests: one fp32 partial per (column, row, K block + 1 pad) for the CTA,
// twice that with gate fusion. The +1 is the odd stride the epilogue uses to avoid bank conflicts.
// Guard and launcher both call this so a shape that would exceed smpb falls back to the generic kernel.
static __host__ size_t ptq1_0_pt_smem_bytes(const int blocks_per_row, const int ncols_dst, const int nrows_x,
        const int rows_per_item, const bool has_gate) {
    const int rows_per_cta = ptq1_0_pt_rows_per_cta(blocks_per_row, ncols_dst, nrows_x, rows_per_item);
    return (size_t) ncols_dst * rows_per_cta * (blocks_per_row + 1) * sizeof(float) * (has_gate ? 2 : 1);
}

template <int ncols, int ROWS, bool has_fusion, bool has_gate>
__launch_bounds__(PTQ1_0_PT_THREADS, (ncols <= 2 ? 4 : (ncols <= 4 ? PTQ1_0_PT_MINB_34 : 2)))
static __global__ void mul_mat_vec_ptq1_0_pt(
        const void * vx_, const void * vy_, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int rows_per_cta, const uint3 bpr_fd, const uint3 rpc_fd, const bool invariant) {
    // GGML_CUDA_RESTRICT stays off the formal parameters: cudafe's host stub drops __restrict
    // from the explicit specialization and MSVC/GCC then reject it (C2912 / "does not match
    // any template declaration") when compiling sm_90/sm_120. Same pattern as mul_mat_vec_q.
    const void * GGML_CUDA_RESTRICT vx = vx_;
    const void * GGML_CUDA_RESTRICT vy = vy_;
    float      * GGML_CUDA_RESTRICT dst = dst_;
    // launched through ggml_cuda_kernel_launch, which opts into PDL on Hopper and newer: wait for the kernel
    // that wrote vy (the q8_1 activation quantization) before reading it
    ggml_cuda_pdl_sync();
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

    if (invariant) {
        // Warp-per-(row, column) lane-strided sum + butterfly. Same bits at 1 column and at
        // 2-4 columns, which is what BATCH_INVARIANT needs for MTP-on == MTP-off. The four-
        // accumulator path below is faster and can diverge (5080 bisect: 5300cd1).
        const int warp = tid / WARP_SIZE;
        const int lane = tid % WARP_SIZE;
        for (int w = warp; w < rows_per_cta*ncols; w += PTQ1_0_PT_THREADS / WARP_SIZE) {
            const int j   = w / rows_per_cta;
            const int r   = w - j*rows_per_cta;
            const int row = row0 + r;

            float sum = 0.0f;
            [[maybe_unused]] float sum_gate = 0.0f;
            for (int kbx = lane; kbx < bpr; kbx += WARP_SIZE) {
                sum += partials[(j*rows_per_cta + r)*bprp + kbx];
                if constexpr (has_gate) {
                    sum_gate += partials_gate[(j*rows_per_cta + r)*bprp + kbx];
                }
            }
            sum = warp_reduce_sum<WARP_SIZE>(sum);
            if constexpr (has_gate) {
                sum_gate = warp_reduce_sum<WARP_SIZE>(sum_gate);
            }

            if (lane == 0 && row < nrows_x) {
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
        return;
    }

    // One thread per (row, column): four interleaved accumulators (k mod 4), then (s0+s1)+(s2+s3).
    // Faster than the warp path (~3% decode). Association depends on column count enough to break
    // MTP-on vs MTP-off identity. GGML_CUDA_BATCH_INVARIANT takes the warp path above.
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

template <int ncols, int ROWS = ptq1_0_pt_rows_per_item(ncols)>
static void mul_mat_vec_ptq1_0_pt_launch(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        cudaStream_t stream) {
    const int bpr = ncols_x / QK_PTQ1_0;
    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const bool has_gate   = fusion.gate != nullptr;

    const int rows_per_cta = ptq1_0_pt_rows_per_cta(bpr, ncols, nrows_x, ROWS);
    const uint3 bpr_fd = init_fastdiv_values((uint32_t) bpr);
    const uint3 rpc_fd = init_fastdiv_values((uint32_t) rows_per_cta);
    const dim3 block_nums((nrows_x + rows_per_cta - 1) / rows_per_cta, 1, 1);
    const dim3 block_dims(PTQ1_0_PT_THREADS, 1, 1);

    const size_t smem = ptq1_0_pt_smem_bytes(bpr, ncols, nrows_x, ROWS, has_gate);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, smem, stream);

#define PTQ1_0_PT_LAUNCH(FUS, GATE)                                                                                      \
    ggml_cuda_kernel_launch(mul_mat_vec_ptq1_0_pt<ncols, ROWS, FUS, GATE>, lp,                                           \
        vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows_per_cta, bpr_fd, rpc_fd, \
        ggml_cuda_batch_invariant())

    if (has_fusion) {
        GGML_ASSERT(ncols <= 4 && "fusion only supported for ncols_dst <= 4");
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
        ncols_dst < 1 || ncols_dst > PTQ1_0_PT_MAX_COLS) {
        return false;
    }
    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    const int rows_per_item = ptq1_0_pt_rows_per_item_for_cc(ncols_dst, cc);
    const size_t smem = ptq1_0_pt_smem_bytes(ncols_x / QK_PTQ1_0, ncols_dst, nrows_x,
            rows_per_item, fusion.gate != nullptr);
    if (smem > ggml_cuda_info().devices[device].smpb) {
        return false;
    }
    switch (ncols_dst) {
        case 1:
            if (rows_per_item == PTQ1_0_PT_ROWS_1) {
                mul_mat_vec_ptq1_0_pt_launch<1, PTQ1_0_PT_ROWS_1>(
                        vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream);
            } else {
                mul_mat_vec_ptq1_0_pt_launch<1, 4>(
                        vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream);
            }
            break;
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

// ---------------------------------------------------------------------------
// One-column PTQ1_0 mat-vec on the warp-transposed exact-isum (SoA) activation layout, Hopper.
//
// The generic kernel gives every lane one 128-weight K block per row, so each block re-loads
// its 36 activation words (32 q8 words + 4 (d, isum) words) and a K = 5120 row keeps 40 of 64
// lane slots busy. Here thread (g, kb) of a CTA loads the activations of K block kb into
// registers once and walks rows g, g + G, g + 2G, ... of the CTA, two rows per step, so the
// activation traffic is paid once per CTA instead of once per row and every thread is busy.
// The digit decode also drops the remainder masks: with F_t = floor(q * 3^t / 256), the high
// bytes of the 16-bit-lane product q * 3^t, the base-3 digit is F_{t+1} - 3 F_t, and that
// difference taken on the packed 32-bit word leaves every byte in 0..2 (no borrows). The
// integer sums are the same as the multiply-and-mask walk of vec_dot_ptq1_0_q8_1_multi.
//
// Bit-exact with the generic kernel: each (row, K block) partial is the same float expression
// (d and the sub-block fold are kept apart until the epilogue), and the epilogue replays the
// generic kernel's K partition and order: lane l sums blocks l, l + S, l + 2S, ... with an FMA
// per block (the generic loop contracts acc += d * acc_block the same way), the S / 32 lane
// groups are added in warp order, then the same butterfly. S = 32 for K < 128 blocks (the
// warp-per-row small-K geometry), S = 128 otherwise (four warps per row). Bias and GLU fusion
// follow the respective path.
// ---------------------------------------------------------------------------
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// five base-3 digits of each byte of P, digit t of all four bytes packed into D[t]
static __device__ __forceinline__ void ptq1_0_ks_trits5(const uint32_t P, uint32_t (&D)[5]) {
    const uint32_t vlo = __byte_perm(P, 0, 0x4140);
    const uint32_t vhi = __byte_perm(P, 0, 0x4342);
    uint32_t Fp = 0;
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        constexpr uint32_t pow3[5] = {3, 9, 27, 81, 243};
        const uint32_t Fn = __byte_perm(vlo * pow3[t], vhi * pow3[t], 0x7531);
        D[t] = Fn - 3*Fp;
        Fp = Fn;
    }
}

// ROWS weight blocks (7 words each: qs[0..23], then qh[0..1] | d << 16) against one activation
// block held in registers (ua[0..31] = q8 words sub*8 + m, ua[32..35] = (d8, isum) words).
// dd = weight scale, aa = sum_k d8_k * (sumi_k - isum_k), the operands of the generic kernel's d * acc.
template <int ROWS>
static __device__ __forceinline__ void ptq1_0_ks_block_dot(
        const uint32_t (&w)[ROWS][7], const int (&ua)[36], float (&dd)[ROWS], float (&aa)[ROWS]) {
    int sumi[ROWS][4];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            sumi[i][k] = 0;
        }
    }
#pragma unroll
    for (int g = 0; g < 6; ++g) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            uint32_t D[5];
            ptq1_0_ks_trits5(w[i][g], D);
#pragma unroll
            for (int t = 0; t < 5; ++t) {
                // qs[0..15]: element 16 t + 4 g + b; qs[16..23]: element 80 + 8 t + 4 (g - 4) + b
                const int e = g < 4 ? 16*t + 4*g : 80 + 8*t + 4*(g - 4);
                sumi[i][e >> 5] = ggml_cuda_dp4a((int) D[t], ua[(e >> 5)*8 + ((e & 31) >> 2)], sumi[i][e >> 5]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
        // qh: two bytes, four trits each, interleaved: element 120 + 2 t + h
        uint32_t v = __byte_perm(w[i][6], 0, 0x4140);
#pragma unroll
        for (int t = 0; t < 4; t += 2) {
            const uint32_t w0 = v * 3;
            v                 = w0 & 0x00FF00FF;
            const uint32_t w1 = v * 3;
            v                 = w1 & 0x00FF00FF;
            const int q = __byte_perm(w0, w1, 0x7531);
            sumi[i][3] = ggml_cuda_dp4a(q, ua[3*8 + 6 + t/2], sumi[i][3]);
        }
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int   ds_bits = ua[32 + k];
            const half2 ds      = *reinterpret_cast<const half2 *>(&ds_bits);
            const int   isum    = (int) __half_as_short(__high2half(ds));
            acc += __low2float(ds) * (float) (sumi[i][k] - isum);
        }
        dd[i] = __half2float(__ushort_as_half((unsigned short) (w[i][6] >> 16)));
        aa[i] = acc;
    }
}

static __device__ __forceinline__ float ptq1_0_ks_glu(const ggml_glu_op op, float result, const float gate_value) {
    switch (op) {
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
    return result;
}

// G row groups of RPT rows per CTA (R = G * RPT rows), ROWS rows per step. Launch with
// round_up(G * blocks_per_row, 32) threads; threads past G * blocks_per_row only join the epilogue.
template <int G, int RPT, int ROWS, int LB, int MINB, bool has_fusion, bool has_gate>
__launch_bounds__(LB, MINB)
static __global__ void mul_mat_vec_ptq1_0_ks(
        const void * vx_, const void * vy_, const ggml_cuda_mm_fusion_args_device fusion, float * dst_,
        const int ncols_x, const int nrows_x, const int stride_row_x, const uint3 bpr_fd, const bool small_k_order) {
    static_assert(RPT % ROWS == 0, "RPT must be a multiple of ROWS");
    constexpr int R  = G * RPT;
    constexpr int NM = has_gate ? 2 : 1;
    const void * GGML_CUDA_RESTRICT vx  = vx_;
    const void * GGML_CUDA_RESTRICT vy  = vy_;
    float      * GGML_CUDA_RESTRICT dst = dst_;

    extern __shared__ float2 ptq1_0_ks_part[]; // [NM][R][bpr]: (d, acc) of every (row, K block)

    const int bpr  = ncols_x / QK_PTQ1_0;
    const int row0 = R * blockIdx.x;
    const int nr   = min(R, nrows_x - row0);
    const int g    = fastdiv((uint32_t) threadIdx.x, bpr_fd);
    const int kb   = threadIdx.x - g*bpr;

    // the activations are written by the preceding quantize kernel
    ggml_cuda_pdl_sync();

    if (g < G) {
        int ua[36];
        const int * yw = (const int *) vy + (kb / GGML_CUDA_PTQ1_Q8_GROUP_KB) * GGML_CUDA_PTQ1_Q8_GROUP_WORDS
                                          + kb % GGML_CUDA_PTQ1_Q8_GROUP_KB;
#pragma unroll
        for (int k = 0; k < 36; ++k) {
            ua[k] = yw[k * GGML_CUDA_PTQ1_Q8_GROUP_KB];
        }
#pragma unroll
        for (int j = 0; j < RPT; j += ROWS) {
            uint32_t w[ROWS*NM][7];
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                const int r = min(g + G*(j + i), nr - 1); // clamp the tail, that result is not stored
#pragma unroll
                for (int m = 0; m < NM; ++m) {
                    const uint32_t * p = (const uint32_t *) ((const block_ptq1_0 *) (m == 0 ? vx : fusion.gate) +
                                                             (int64_t) (row0 + r)*stride_row_x + kb);
#pragma unroll
                    for (int k = 0; k < 7; ++k) {
                        w[i*NM + m][k] = p[k];
                    }
                }
            }
            float dd[ROWS*NM];
            float aa[ROWS*NM];
            ptq1_0_ks_block_dot<ROWS*NM>(w, ua, dd, aa);
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                const int r = g + G*(j + i);
                if (r < nr) {
#pragma unroll
                    for (int m = 0; m < NM; ++m) {
                        ptq1_0_ks_part[(m*R + r)*bpr + kb] = make_float2(dd[i*NM + m], aa[i*NM + m]);
                    }
                }
            }
        }
    }
    __syncthreads();

    const int S    = small_k_order ? WARP_SIZE : 4*WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    for (int r = warp; r < nr; r += blockDim.x / WARP_SIZE) {
        float sum[NM];
#pragma unroll
        for (int m = 0; m < NM; ++m) {
            const float2 * p = ptq1_0_ks_part + (m*R + r)*bpr;
            float tot = 0.0f;
            for (int w0 = 0; w0 < S; w0 += WARP_SIZE) {
                float a = 0.0f;
                for (int k = w0 + lane; k < bpr; k += S) {
                    const float2 v = p[k];
                    a += v.x * v.y;
                }
                tot = w0 == 0 ? a : tot + a;
            }
            sum[m] = warp_reduce_sum<WARP_SIZE>(tot);
        }
        const int row = row0 + r;
        if (lane == 0) {
            float result = sum[0];
            if constexpr (has_fusion) {
                const float * x_bias    = (const float *) fusion.x_bias;
                const float * gate_bias = (const float *) fusion.gate_bias;
                if (small_k_order) {
                    // warp-per-row small-K path: biases are added only when present
                    if (x_bias) {
                        result += x_bias[row];
                    }
                    if constexpr (has_gate) {
                        float gate_value = sum[NM - 1];
                        if (gate_bias) {
                            gate_value += gate_bias[row];
                        }
                        result = ptq1_0_ks_glu(fusion.glu_op, result, gate_value);
                    }
                } else {
                    // four-warp path: the bias registers default to 0.0f and are always added
                    result += x_bias ? x_bias[row] : 0.0f;
                    if constexpr (has_gate) {
                        const float gate_value = sum[NM - 1] + (gate_bias ? gate_bias[row] : 0.0f);
                        result = ptq1_0_ks_glu(fusion.glu_op, result, gate_value);
                    }
                }
            }
            dst[row] = result;
        }
    }
}

// GATE selects the gate-fused instantiation; the others cover the plain and bias-fused calls
template <int G, int RPT, int ROWS, int LB, int MINB, bool GATE>
static void mul_mat_vec_ptq1_0_ks_launch(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const bool small_k_order, cudaStream_t stream) {
    constexpr int R = G * RPT;
    const int bpr = ncols_x / QK_PTQ1_0;
    const int nthreads = (G*bpr + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    GGML_ASSERT(nthreads <= LB);
    GGML_ASSERT(GATE == (fusion.gate != nullptr));
    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const size_t smem = (size_t) (GATE ? 2 : 1) * R * bpr * sizeof(float2);
    const dim3 block_nums((nrows_x + R - 1) / R, 1, 1);
    const dim3 block_dims(nthreads, 1, 1);
    const uint3 bpr_fd = init_fastdiv_values((uint32_t) bpr);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, smem, stream);
    if constexpr (GATE) {
        ggml_cuda_kernel_launch(mul_mat_vec_ptq1_0_ks<G, RPT, ROWS, LB, MINB, true, true>, lp,
            vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, bpr_fd, small_k_order);
    } else if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_ptq1_0_ks<G, RPT, ROWS, LB, MINB, true, false>, lp,
            vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, bpr_fd, small_k_order);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_ptq1_0_ks<G, RPT, ROWS, LB, MINB, false, false>, lp,
            vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, bpr_fd, small_k_order);
    }
}

static bool ggml_cuda_ptq1_0_ks_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_PTQ1_KS");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

// true when the K-block-stationary kernel handled a plain 2D one-column call on the SoA layout.
// It pays where the generic kernel leaves lanes idle in its last K pass: K = 33-63 blocks (one
// warp per row) and K = 129-160 blocks (four warps per row). H200, Bonsai 2 27B decode, in-graph
// kernel time: K = 5120 projections -21..-29%, LM head -38%, K = 6144 -14%, K = 17408 down -16%.
// At K = 32 or 64 blocks every lane is already busy and this kernel was 5-10% slower in isolation
// (9B, K = 4096), so those, other K, matrices under 4096 rows, gate fusion past 64 blocks and
// non-Hopper devices keep the generic kernel. GGML_CUDA_PTQ1_KS=0 turns it off.
static bool mul_mat_vec_ptq1_0_ks_switch(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, cudaStream_t stream) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_HOPPER || cc >= GGML_CUDA_CC_BLACKWELL ||
        !ggml_cuda_ptq1_0_ks_enabled()) {
        return false;
    }
    if (nrows_x < 4096 || ncols_x % QK_PTQ1_0 != 0 || fusion.x_scale != nullptr || fusion.gate_scale != nullptr ||
        (uintptr_t) vx % 4 != 0 || (uintptr_t) fusion.gate % 4 != 0) {
        return false;
    }
    const int  bpr      = ncols_x / QK_PTQ1_0;
    const bool has_gate = fusion.gate != nullptr;
    if (bpr > 32 && bpr < 64) {
        if (has_gate) {
            mul_mat_vec_ptq1_0_ks_launch<4, 1, 1, 256, 4, true>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, true, stream);
        } else {
            mul_mat_vec_ptq1_0_ks_launch<4, 2, 2, 256, 4, false>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, true, stream);
        }
        return true;
    }
    if (bpr > 128 && bpr <= 160 && !has_gate) {
        mul_mat_vec_ptq1_0_ks_launch<1, 8, 2, 160, 6, false>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, false, stream);
        return true;
    }
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// PQ2_0 mat-vec for 3-8 columns (plain 2D), tuned on Ada, also taken on Hopper, where it also serves 1-2 columns.
//
// Why: the generic mmvq kernel loses most of its DRAM throughput from 3 columns on (RTX 4070, Bonsai 2 27B shapes:
// 44% of peak at 4 columns). Here the activations use GGML_CUDA_Q8_1_PQ2: per column the qs bytes come first,
// permuted inside each 16-element group (position k*4+m holds element m*4+k), then one half2 per 32-block
// with d and the raw int16 sum of q. With that, (code_word >> 2k) & 0x03030303 pairs the raw 2-bit codes 0..3
// with the right activation bytes, so there is no per-weight decode. The digit bias is one integer subtraction:
// sum(q*(c-1)) = sum(q*c) - sum(q). Each warp handles up to 4 rows and reuses the activation slice of a 32-block for all of them.
#pragma once

#include "common.cuh"
#include "unary.cuh"

#define PQ2_0_MC_ROWS  4
#define PQ2_0_MC_WARPS 4

// Optional fusion (has_fusion, the same epilogue as mul_mat_vec_q): x_bias is added to the result; with has_gate a
// second weight matrix (gate, same layout and strides) is dotted against the same activations, gate_bias is added to
// it and the GLU op combines the two. Without fusion the epilogue is a plain store: a runtime bias check there costs
// the 8-column verify kernel ~12% on H200.
template <int ncols, bool has_fusion, bool has_gate, int rows = PQ2_0_MC_ROWS>
__global__ void __launch_bounds__(PQ2_0_MC_WARPS * 32, 3)
mul_mat_vec_pq2_0_mc(const void * vx_, const void * vy_, float * dst_,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const float * x_bias, const void * vgate_, const float * gate_bias, const ggml_glu_op glu_op, const float glu_limit) {
    // Plain pointer parameters, restrict-qualified locals: GGML_CUDA_RESTRICT differs between the host and device
    // passes, which breaks the template stub. Same pattern as mul_mat_vec_q.
    const void * GGML_CUDA_RESTRICT vx  = vx_;
    const void * GGML_CUDA_RESTRICT vy  = vy_;
    float      * GGML_CUDA_RESTRICT dst = dst_;
    const int lane = threadIdx.x & 31;
    const int row0 = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * rows;
    if (row0 >= nrows_x) {
        return;
    }
    // launched through ggml_cuda_kernel_launch, which opts into PDL on Hopper and newer: wait for the kernel
    // that wrote vy (the q8_1 activation quantization) before reading it
    ggml_cuda_pdl_sync();

    const int     nchunk    = ncols_x / QK8_1;
    const size_t  col_bytes = (size_t) stride_col_y * sizeof(block_q8_1);
    const size_t  ds_off    = (size_t) stride_col_y * QK8_1; // qs region of a column, padded length
    const char *  y         = (const char *) vy;

    float acc[rows][ncols] = {};
    [[maybe_unused]] float accg[rows][ncols] = {};
    [[maybe_unused]] const void * GGML_CUDA_RESTRICT vgate = vgate_;

    for (int c = lane; c < nchunk; c += WARP_SIZE) {
        uint32_t w0[rows], w1[rows];
        float    d2[rows];
        [[maybe_unused]] uint32_t g0[rows], g1[rows];
        [[maybe_unused]] float    e2[rows];
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            const int row = min(row0 + r, nrows_x - 1); // clamped rows are computed but not stored
            const block_pq2_0 * blk = (const block_pq2_0 *) vx + (int64_t) row*stride_row_x + (c >> 2);
            const uint16_t    * qp  = (const uint16_t *) (blk->qs + (c & 3)*8); // blocks are only 2-byte aligned
            d2[r] = __half2float(blk->d);
            w0[r] = (uint32_t) qp[0] | ((uint32_t) qp[1] << 16);
            w1[r] = (uint32_t) qp[2] | ((uint32_t) qp[3] << 16);
            if constexpr (has_gate) {
                const block_pq2_0 * gblk = (const block_pq2_0 *) vgate + (int64_t) row*stride_row_x + (c >> 2);
                const uint16_t    * gp   = (const uint16_t *) (gblk->qs + (c & 3)*8);
                e2[r] = __half2float(gblk->d);
                g0[r] = (uint32_t) gp[0] | ((uint32_t) gp[1] << 16);
                g1[r] = (uint32_t) gp[2] | ((uint32_t) gp[3] << 16);
            }
        }

        int4  a0[ncols], a1[ncols];
        float d8[ncols];
        int   qsum[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const char * yc = y + j*col_bytes;
            a0[j] = *(const int4 *) (yc + c*QK8_1);
            a1[j] = *(const int4 *) (yc + c*QK8_1 + 16);
            const half2 ds = *(const half2 *) (yc + ds_off + c*sizeof(half2));
            d8[j]   = __low2float(ds);
            qsum[j] = __half_as_short(__high2half(ds));
        }

#pragma unroll
        for (int r = 0; r < rows; ++r) {
            int t[8];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                t[k]     = (int) ((w0[r] >> (2*k)) & 0x03030303u);
                t[4 + k] = (int) ((w1[r] >> (2*k)) & 0x03030303u);
            }
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                int sumi = 0;
                sumi = ggml_cuda_dp4a(t[0], a0[j].x, sumi);
                sumi = ggml_cuda_dp4a(t[1], a0[j].y, sumi);
                sumi = ggml_cuda_dp4a(t[2], a0[j].z, sumi);
                sumi = ggml_cuda_dp4a(t[3], a0[j].w, sumi);
                sumi = ggml_cuda_dp4a(t[4], a1[j].x, sumi);
                sumi = ggml_cuda_dp4a(t[5], a1[j].y, sumi);
                sumi = ggml_cuda_dp4a(t[6], a1[j].z, sumi);
                sumi = ggml_cuda_dp4a(t[7], a1[j].w, sumi);
                acc[r][j] += d2[r] * d8[j] * (float) (sumi - qsum[j]);
            }
            if constexpr (has_gate) {
                int tg[8];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    tg[k]     = (int) ((g0[r] >> (2*k)) & 0x03030303u);
                    tg[4 + k] = (int) ((g1[r] >> (2*k)) & 0x03030303u);
                }
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    int sumi = 0;
                    sumi = ggml_cuda_dp4a(tg[0], a0[j].x, sumi);
                    sumi = ggml_cuda_dp4a(tg[1], a0[j].y, sumi);
                    sumi = ggml_cuda_dp4a(tg[2], a0[j].z, sumi);
                    sumi = ggml_cuda_dp4a(tg[3], a0[j].w, sumi);
                    sumi = ggml_cuda_dp4a(tg[4], a1[j].x, sumi);
                    sumi = ggml_cuda_dp4a(tg[5], a1[j].y, sumi);
                    sumi = ggml_cuda_dp4a(tg[6], a1[j].z, sumi);
                    sumi = ggml_cuda_dp4a(tg[7], a1[j].w, sumi);
                    accg[r][j] += e2[r] * d8[j] * (float) (sumi - qsum[j]);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

#pragma unroll
    for (int r = 0; r < rows; ++r) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float v = warp_reduce_sum<WARP_SIZE>(acc[r][j]);
            [[maybe_unused]] float gv = 0.0f;
            if constexpr (has_gate) {
                gv = warp_reduce_sum<WARP_SIZE>(accg[r][j]);
            }
            const int row = row0 + r;
            if (lane == 0 && row < nrows_x) {
                const int64_t o = (int64_t) j*stride_col_dst + row;
                if constexpr (has_fusion) {
                    if (x_bias) {
                        v += x_bias[o];
                    }
                }
                if constexpr (has_gate) {
                    if (gate_bias) {
                        gv += gate_bias[o];
                    }
                    switch (glu_op) {
                        case GGML_GLU_OP_SWIGLU:     v *= ggml_cuda_op_silu_single(gv);          break;
                        case GGML_GLU_OP_GEGLU:      v *= ggml_cuda_op_gelu_single(gv);          break;
                        case GGML_GLU_OP_SWIGLU_OAI: v  = ggml_cuda_op_swiglu_oai_single(gv, v); break;
                        case GGML_GLU_OP_SWIGLU_CLAMP: v = ggml_cuda_op_swiglu_clamp_single(gv, v, glu_limit); break;
                        default:                     v *= gv;                                    break;
                    }
                }
                dst[o] = v;
            }
        }
    }
}

template <int ncols, int rows = PQ2_0_MC_ROWS>
static void mul_mat_vec_pq2_0_mc_launch_rows(
        const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst, cudaStream_t stream,
        const ggml_cuda_mm_fusion_args_device & fusion) {
    const int nwarps = (nrows_x + rows - 1) / rows;
    const dim3 block_nums((nwarps + PQ2_0_MC_WARPS - 1) / PQ2_0_MC_WARPS, 1, 1);
    const dim3 block_dims(PQ2_0_MC_WARPS * WARP_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    if (fusion.gate) {
        ggml_cuda_kernel_launch(mul_mat_vec_pq2_0_mc<ncols, true, true, rows>, lp, vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
            (const float *) fusion.x_bias, fusion.gate, (const float *) fusion.gate_bias, fusion.glu_op, fusion.glu_limit);
    } else if (fusion.x_bias) {
        ggml_cuda_kernel_launch(mul_mat_vec_pq2_0_mc<ncols, true, false, rows>, lp, vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
            (const float *) fusion.x_bias, nullptr, nullptr, fusion.glu_op, fusion.glu_limit);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_pq2_0_mc<ncols, false, false, rows>, lp, vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
            nullptr, nullptr, nullptr, fusion.glu_op, fusion.glu_limit);
    }
}

template <int ncols>
static void mul_mat_vec_pq2_0_mc_launch(const void *                            vx,
                                        const void *                            vy,
                                        float *                                 dst,
                                        const int                               ncols_x,
                                        const int                               nrows_x,
                                        const int                               stride_row_x,
                                        const int                               stride_col_y,
                                        const int                               stride_col_dst,
                                        cudaStream_t                            stream,
                                        const ggml_cuda_mm_fusion_args_device & fusion) {
    if constexpr (ncols == 1) {
        // Smaller Hopper projections need more CTAs; keep four-row reuse for the vocabulary head.
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        if (cc == GGML_CUDA_CC_HOPPER && nrows_x <= 17408) {
            mul_mat_vec_pq2_0_mc_launch_rows<ncols, 2>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y,
                                                       stride_col_dst, stream, fusion);
            return;
        }
    }
    mul_mat_vec_pq2_0_mc_launch_rows<ncols>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                                            stream, fusion);
}

static void mul_mat_vec_pq2_0_mc_switch(
        const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst, cudaStream_t stream,
        const ggml_cuda_mm_fusion_args_device & fusion) {
    GGML_ASSERT(ncols_x % QK_PQ2_0 == 0);
    switch (ncols_dst) {
        case 1: mul_mat_vec_pq2_0_mc_launch<1>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        case 2: mul_mat_vec_pq2_0_mc_launch<2>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        case 3: mul_mat_vec_pq2_0_mc_launch<3>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        case 4: mul_mat_vec_pq2_0_mc_launch<4>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        case 5: mul_mat_vec_pq2_0_mc_launch<5>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        case 6: mul_mat_vec_pq2_0_mc_launch<6>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        case 7: mul_mat_vec_pq2_0_mc_launch<7>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        case 8: mul_mat_vec_pq2_0_mc_launch<8>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream, fusion); break;
        default: GGML_ABORT("unsupported column count for the PQ2_0 multi-column kernel");
    }
}

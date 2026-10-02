// PQ2_0 mat-vec for 3-8 columns (plain 2D), tuned on Ada.
//
// Why: the generic mmvq kernel loses most of its DRAM throughput from 3 columns on (RTX 4070, Bonsai 2 27B shapes:
// 44% of peak at 4 columns). Here the activations use GGML_CUDA_Q8_1_PQ2: per column the qs bytes come first,
// permuted inside each 16-element group (position k*4+m holds element m*4+k), then one half2 per 32-block
// with d and the raw int16 sum of q. With that, (code_word >> 2k) & 0x03030303 pairs the raw 2-bit codes 0..3
// with the right activation bytes, so there is no per-weight decode. The digit bias is one integer subtraction:
// sum(q*(c-1)) = sum(q*c) - sum(q). Each warp handles 4 rows and reuses the activation slice of a 32-block for all of them.
#pragma once

#include "common.cuh"

#define PQ2_0_MC_ROWS  4
#define PQ2_0_MC_WARPS 4

template <int ncols>
__global__ void __launch_bounds__(PQ2_0_MC_WARPS * 32, 3)
mul_mat_vec_pq2_0_mc(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y, const int stride_col_dst) {
    const int lane = threadIdx.x & 31;
    const int row0 = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * PQ2_0_MC_ROWS;
    if (row0 >= nrows_x) {
        return;
    }

    const int     nchunk    = ncols_x / QK8_1;
    const size_t  col_bytes = (size_t) stride_col_y * sizeof(block_q8_1);
    const size_t  ds_off    = (size_t) stride_col_y * QK8_1; // qs region of a column, padded length
    const char *  y         = (const char *) vy;

    float acc[PQ2_0_MC_ROWS][ncols] = {};

    for (int c = lane; c < nchunk; c += WARP_SIZE) {
        uint32_t w0[PQ2_0_MC_ROWS], w1[PQ2_0_MC_ROWS];
        float    d2[PQ2_0_MC_ROWS];
#pragma unroll
        for (int r = 0; r < PQ2_0_MC_ROWS; ++r) {
            const int row = min(row0 + r, nrows_x - 1); // clamped rows are computed but not stored
            const block_pq2_0 * blk = (const block_pq2_0 *) vx + (int64_t) row*stride_row_x + (c >> 2);
            const uint16_t    * qp  = (const uint16_t *) (blk->qs + (c & 3)*8); // blocks are only 2-byte aligned
            d2[r] = __half2float(blk->d);
            w0[r] = (uint32_t) qp[0] | ((uint32_t) qp[1] << 16);
            w1[r] = (uint32_t) qp[2] | ((uint32_t) qp[3] << 16);
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
        for (int r = 0; r < PQ2_0_MC_ROWS; ++r) {
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
        }
    }

#pragma unroll
    for (int r = 0; r < PQ2_0_MC_ROWS; ++r) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float v = warp_reduce_sum<WARP_SIZE>(acc[r][j]);
            const int row = row0 + r;
            if (lane == 0 && row < nrows_x) {
                dst[(int64_t) j*stride_col_dst + row] = v;
            }
        }
    }
}

template <int ncols>
static void mul_mat_vec_pq2_0_mc_launch(
        const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    const int nwarps = (nrows_x + PQ2_0_MC_ROWS - 1) / PQ2_0_MC_ROWS;
    const dim3 block_nums((nwarps + PQ2_0_MC_WARPS - 1) / PQ2_0_MC_WARPS, 1, 1);
    const dim3 block_dims(PQ2_0_MC_WARPS * WARP_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_pq2_0_mc<ncols>, lp, vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst);
}

static void mul_mat_vec_pq2_0_mc_switch(
        const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_PQ2_0 == 0);
    switch (ncols_dst) {
        case 3: mul_mat_vec_pq2_0_mc_launch<3>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_pq2_0_mc_launch<4>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_pq2_0_mc_launch<5>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_pq2_0_mc_launch<6>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_pq2_0_mc_launch<7>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_pq2_0_mc_launch<8>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported column count for the PQ2_0 multi-column kernel");
    }
}

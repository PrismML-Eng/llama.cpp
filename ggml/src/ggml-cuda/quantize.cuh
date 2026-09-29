#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

typedef void (*quantize_cuda_t)(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

// layout: see ggml_cuda_q8_1_layout_for() in common.cuh. The caller must pass the same layout the
// consuming mat-vec kernel expects for (type_src0, ncols_dst, ids).
void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_cuda_q8_1_layout layout, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

// Hadamard transform (width n, optional per-element signs, scale 1/sqrt(n)) + q8_1 quantization in one
// kernel (ggml_cuda_fwht_q8). x holds ncols contiguous rows of ne00 elements; ne0 is the padded row
// width the consuming mat-vec expects (pad blocks are written as zeros).
bool ggml_cuda_fwht_quantize_supported(int n, int64_t ne00);
void fwht_quantize_row_q8_1_cuda(
        const void * x, ggml_type x_type, const float * signs, int n, void * vy,
        ggml_cuda_q8_1_layout layout, int64_t ne00, int64_t ne0, int64_t ncols, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_q8_1_swiglu_cuda(
        const float * x, const float * gate, void * vy, ggml_type type_src0,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_q8_1_rms_cuda(
        const float * x, const float * weight, const float * scale, void * vy, ggml_type type_src0,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_fp4_cuda(const float *   x,
                             const int32_t * ids,
                             void *          vy,
                             float *         scale,
                             ggml_type       type_src0,
                             bool            use_aligned_float8,
                             int64_t         ne00,
                             int64_t         s01,
                             int64_t         s02,
                             int64_t         s03,
                             int64_t         ne0,
                             int64_t         ne1,
                             int64_t         ne2,
                             int64_t         ne3,
                             cudaStream_t    stream);

// quantize each token once and scatter the block to its compact rows (via the inverse map)
void quantize_scatter_mmq_fp4_cuda(const float *   x,
                                   const int32_t * ids_src1_inv,
                                   void *          vy,
                                   float *         scale,
                                   ggml_type       type_src0,
                                   bool            use_aligned_float8,
                                   int64_t         ne00,
                                   int64_t         stride_token,
                                   int64_t         ne0,
                                   int64_t         n_tokens,
                                   int64_t         nrows_dst,
                                   int             n_expert_used,
                                   cudaStream_t    stream);

void quantize_scatter_mmq_q8_1_cuda(const float *   x,
                                    const int32_t * ids_src1_inv,
                                    void *          vy,
                                    ggml_type       type_src0,
                                    int64_t         ne00,
                                    int64_t         stride_token,
                                    int64_t         ne0,
                                    int64_t         n_tokens,
                                    int64_t         nrows_dst,
                                    int             n_expert_used,
                                    cudaStream_t    stream);

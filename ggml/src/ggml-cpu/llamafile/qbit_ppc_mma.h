#pragma once
// POWER10/POWER11 MMA GEMM for PrismML Q1_0/Q2_0 weights x Q8_0 activations.
// k is in ELEMENTS (multiple of 128); lda in weight blocks, ldb in q8 blocks;
// C column-major float. A/B are block_q1_0|block_q2_0 / block_q8_0 arrays.
// From github.com/mavin2009/ppc-mma-kernels.
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void gemm_q1_0_q8_0_ppc_v3(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_q2_0_q8_0_ppc_v3(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
#ifdef __cplusplus
}
#endif

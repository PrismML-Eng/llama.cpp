#pragma once
// POWER10/POWER11 MMA GEMM one-shot entries for K-quant weights x Q8_K
// activations.  k in ELEMENTS (multiple of 256); lda/ldb in superblocks;
// C column-major float.  From github.com/mavin2009/ppc-mma-kernels.
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void gemm_q4_K_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_q5_K_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_q6_K_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_q2_K_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
#ifdef __cplusplus
}
#endif

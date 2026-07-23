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
void gemm_q3_K_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_iq4_nl_q8_0_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_iq4_xs_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_q4_1_q8_1_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_q5_0_q8_0_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_q5_1_q8_1_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_tq2_0_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_tq1_0_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_iq2_xxs_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_iq3_xxs_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_iq3_s_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_iq1_s_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_mxfp4_q8_0_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_iq2_xs_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_iq2_s_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_iq1_m_q8_K_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_nvfp4_q8_0_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_q8_0_q8_0_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
void gemm_q4_0_q8_0_ppc(int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb,
        float * C, int64_t ldc, int ith, int nth);
// tensor-keyed cache for repacked weights (ppc_pack_cache.cpp)
#include <stddef.h>
void * ppc_apack_cache_acquire(const void * key, int64_t m, int64_t k,
                               int variant, size_t bytes, int * fresh);
void ppc_apack_cache_publish(const void * key, int64_t m, int64_t k, int variant);
void ppc_apack_cache_clear(void);
#ifdef __cplusplus
}
#endif

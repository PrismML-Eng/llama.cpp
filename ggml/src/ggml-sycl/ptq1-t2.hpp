// ARC-LAB: PTQ1_0 weights repacked to the TernSYCL 2-bit layout and multiplied on XMX (s8 x s2 DPAS).
// See ptq1-t2.cpp.
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

// GGML_SYCL_PTQ1_T2: "ffn" = ffn_gate/ffn_up/ffn_down weights, "all" = every PTQ1_0 weight (tests), unset = off
bool   ggml_sycl_t2_wants(const char * name, int64_t K, int64_t N);
// device bytes of the repacked weight: 2-bit codes [K/16, N] u32 + fp16 scales [K/128, N]
size_t ggml_sycl_t2_bytes(int64_t K, int64_t N);
// in place: data holds N rows of K/128 block_ptq1_0 (pq2 = false) or block_pq2_0 (pq2 = true) as uploaded, and at least
// ggml_sycl_t2_bytes(K, N) bytes. PQ2_0 code 3 (+2) has no 2-bit two's-complement form: returns false if one is found.
bool   ggml_sycl_t2_repack(sycl::queue & q, void * data, int64_t K, int64_t N, bool pq2 = false);
// dst[m * N + n] = sum_k x[m * x_stride + k] * w[n, k], M tokens (fp32 in and out)
void   ggml_sycl_t2_mul_mat(sycl::queue & q, const void * w, const float * x, int64_t x_stride, float * dst, int64_t M,
                            int64_t N, int64_t K);

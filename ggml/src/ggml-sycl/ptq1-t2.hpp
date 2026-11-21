// ARC-LAB: PTQ1_0 weights repacked to the TernSYCL 2-bit layout and multiplied on XMX (s8 x s2 DPAS).
// See ptq1-t2.cpp.
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

// GGML_SYCL_PTQ1_T2: "ffn" = ffn_gate/ffn_up/ffn_down weights, "all" = every PTQ1_0 weight (tests), unset = off
bool   ggml_sycl_t2_wants(const char * name, int64_t K, int64_t N);
// Xe2 or newer (16-lane DPAS incl. the int8 x int2 / int8 x int4 builtins): Arc B-series, Lunar Lake, Panther Lake
bool   ggml_sycl_device_is_xe2();
bool   ggml_sycl_device_is_xe3(int device);
// device bytes of the repacked weight: 2-bit codes [K/16, N] u32 + fp16 scales [K/128, N]
size_t ggml_sycl_t2_bytes(int64_t K, int64_t N);
// in place: data holds N rows of K/128 block_ptq1_0 (pq2 = false) or block_pq2_0 (pq2 = true) as uploaded, and at least
// ggml_sycl_t2_bytes(K, N) bytes. PQ2_0 code 3 (+2) has no 2-bit two's-complement form: returns false if one is found.
bool   ggml_sycl_t2_repack(sycl::queue & q, void * data, int64_t K, int64_t N, bool pq2 = false);
// dst[m * N + n] = sum_k x[m * x_stride + k] * w[n, k], M tokens (fp32 in and out). x_key (optional) names the
// activation within the current graph (its tensor + ggml_sycl_t2_new_graph epoch): back-to-back calls on the same x_key
// reuse the int8 activations instead of quantizing again (q/k/v, gate/up, the DeltaNet input projections).
void   ggml_sycl_t2_mul_mat(sycl::queue & q, const void * w, const float * x, int64_t x_stride, float * dst, int64_t M,
                            int64_t N, int64_t K, const void * x_key = nullptr);
// nm (2..3) weights on the same activations (M <= 8 tokens) in one launch; false = not served (caller runs them singly)
bool   ggml_sycl_t2_mul_mat_multi(sycl::queue & q, int nm, const void * const * w, float * const * dst, const int64_t * N,
                                  const float * x, int64_t x_stride, int64_t M, int64_t K, const void * x_key);
// a new graph compute starts: activations quantized for the previous one are stale
void   ggml_sycl_t2_new_graph();
// dst [M, K] = 1024-point Hadamard of (x * signs) per block / 32 (signs: K floats), quantized for the T2 mat-vecs at once;
// key = the tensor they will read it through (see x_key). false: shape not served (K % 1024, M > 512) or knob off.
// glu_g / glu_u (optional, then x is unused): x = silu(glu_g) * glu_u, both [M, K] contiguous (SWIGLU folded in)
bool   ggml_sycl_t2_fwht_quant(sycl::queue & q, const float * x, const float * signs, float * dst, int64_t M, int64_t K,
                               const void * key, const float * glu_g = nullptr, const float * glu_u = nullptr);
// [add_out =] a (+ b); normed = rms_norm(.) * w; [mm = rotation of normed * signs, quantized as above]. b, add_out,
// normed, signs (and then mm, key) may be null. One work-group per token: K % 1024 == 0, K <= 8192, M <= 512.
bool   ggml_sycl_t2_add_norm_rot(sycl::queue & q, const float * a, const float * b, float * add_out, const float * w,
                                 float eps, float * normed, const float * signs, float * mm, int64_t M, int64_t K,
                                 const void * key);

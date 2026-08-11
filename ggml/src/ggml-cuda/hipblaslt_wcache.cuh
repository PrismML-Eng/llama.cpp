#pragma once
#include <cstddef>

// Invalidation registry for the hipBLASLt per-route converted-weight caches.
//
// Each mul_mat_*_hipblaslt.cu keeps a cache of weights it has already converted
// (requantised to int8 or e4m3), keyed on the weight tensor's device address.
// That key is only unique while the underlying buffer is alive: once a buffer is
// freed, a later allocation can land on the same address and would silently hit
// a stale entry. That is not hypothetical -- it produces garbage output on
// multi-model runs and on llama-server model swaps.
//
// So every route registers an invalidator here, and the CUDA backend calls
// ggml_hipblaslt_wcache_invalidate() when it frees a device buffer. Cost is
// zero on the hot path; the work happens only at teardown.

typedef void (*ggml_hipblaslt_wcache_invalidator)(const void * base, size_t size);

void ggml_hipblaslt_wcache_register(ggml_hipblaslt_wcache_invalidator fn);

// Drop every cached entry whose weight pointer lies in [base, base+size).
void ggml_hipblaslt_wcache_invalidate(const void * base, size_t size);

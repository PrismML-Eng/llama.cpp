#pragma once

// Streaming top-k for wide rows with a small k, e.g. the DFlash2 selector that
// takes the top 16 of a 248k-entry vocabulary row for every block position.
//
// The result is the same as taking the first k entries of the stable
// descending cub radix sort used by the argsort fallback: keys are ordered by
// cub's float bit twiddle, -0.0 is treated as +0.0, and equal keys keep
// ascending column order. Each candidate is packed into one 64-bit value whose
// unsigned order is exactly that total order, so selection is a sequence of
// strict "largest value below the previous winner" reductions. Real keys are
// unique (the low word holds the column) and nonzero; 0 is padding.
//
// Pass 1 splits every row into chunks and writes the k best packed values of
// each chunk in descending order. Pass 2 merges the chunk lists (one warp when
// there are at most 32 of them, a block reduction otherwise) and writes the
// column indices in descending order.
//
// Adapted from ArthurOstapenko's Yukon CUDA top-k selector (MIT, see licenses/yukon-cudafast):
// https://github.com/Layr-Labs/cudafast-bonsai2-9b-engine/tree/2eb0f36a6083fccf7b28ed6498692709e77b6654
// Selector mechanisms after newjordan's submissions 17b3ba36 (cached per-thread
// maxima), 686d8f21 (two-barrier block reduction), d9c98fd7 (512x16 blocks,
// native warp maximum, warp head merge) and 68fdfe3f (certified threshold).

#include "common.cuh"

#include <cstdint>

#define TOPK_SELECT_NT          512
#define TOPK_SELECT_PER_THREAD  16
#define TOPK_SELECT_CHUNK       (TOPK_SELECT_NT * TOPK_SELECT_PER_THREAD)
#define TOPK_SELECT_MAX_K       32
#define TOPK_SELECT_MAX_ROWS    65535
#define TOPK_SELECT_COMPACT     256 // survivor buffer of the threshold path
#define TOPK_SELECT_MERGE_WARP  32  // at most this many chunk lists: one-warp merge

static __device__ __forceinline__ uint64_t topk_select_pack(const float v, const int col) {
    uint32_t u = __float_as_uint(v);
    u = u == 0x80000000u ? 0u : u;                                 // -0.0 sorts as +0.0
    const uint32_t ord = (u & 0x80000000u) ? ~u : (u | 0x80000000u);  // cub float twiddle
    return ((uint64_t) ord << 32) | (uint64_t) (~(uint32_t) col);     // ties: lower column first
}

// Unsigned 64-bit warp maximum: the maximum high word, then the maximum low word among the lanes that hold it.
static __device__ __forceinline__ uint64_t topk_select_warp_max(uint64_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t hi     = (uint32_t) (v >> 32);
    const uint32_t max_hi = __reduce_max_sync(0xffffffffu, hi);
    const uint32_t max_lo = __reduce_max_sync(0xffffffffu, hi == max_hi ? (uint32_t) v : 0u);
    return ((uint64_t) max_hi << 32) | max_lo;
#else
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const uint64_t w = __shfl_xor_sync(0xffffffff, v, offset);
        v = w > v ? w : v;
    }
    return v;
#endif
}

// Block-wide maximum. `red` needs TOPK_SELECT_NT/32 + 1 entries; every thread
// returns the same value. Back-to-back calls may reuse `red`: a call writes only
// the per-warp slots before its first barrier, and warp 0 overwrites the result
// slot only after that barrier, when every thread has read the previous result.
static __device__ __forceinline__ uint64_t topk_select_block_max(uint64_t v, uint64_t * red) {
    v = topk_select_warp_max(v);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (lane == 0) {
        red[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        v = lane < TOPK_SELECT_NT / 32 ? red[lane] : 0;
        v = topk_select_warp_max(v);
        if (lane == 0) {
            red[TOPK_SELECT_NT / 32] = v;
        }
    }
    __syncthreads();
    return red[TOPK_SELECT_NT / 32];
}

// Wait for the kernel this one follows, then let the next one launch (no-ops without programmatic launch).
static __device__ __forceinline__ void topk_select_grid_dependency() {
    ggml_cuda_pdl_sync();
    ggml_cuda_pdl_lc();
}

// grid (nchunks, nrows); part holds nrows * nchunks * k packed values.
// A chunk with fewer than k columns pads its tail with 0, which ranks below
// every packed column because the low word ~col is at least 2^31.
static __global__ void __launch_bounds__(TOPK_SELECT_NT)
topk_select_partial(const float * src, uint64_t * part, const int ncols, const int k) {
    __shared__ uint64_t red[TOPK_SELECT_NT / 32 + 1];
    __shared__ uint64_t compact[TOPK_SELECT_COMPACT];
    __shared__ int      compact_count;

    const int     row = blockIdx.y;
    const int     beg = blockIdx.x * TOPK_SELECT_CHUNK;
    const float * x   = src + (int64_t) row * ncols;

    // both passes may be launched with programmatic stream serialization; the arguments are not __restrict__,
    // so no load moves above the wait
    topk_select_grid_dependency();

    uint64_t v[TOPK_SELECT_PER_THREAD];
#pragma unroll
    for (int j = 0; j < TOPK_SELECT_PER_THREAD; ++j) {
        const int col = beg + j * TOPK_SELECT_NT + threadIdx.x;
        v[j] = col < ncols ? topk_select_pack(x[col], col) : 0;
    }

    uint64_t * out  = part + ((int64_t) row * gridDim.x + blockIdx.x) * k;
    uint64_t   best = 0; // this thread's largest key below the last winner
#pragma unroll
    for (int j = 0; j < TOPK_SELECT_PER_THREAD; ++j) {
        best = v[j] > best ? v[j] : best;
    }

    // Threshold path, k <= number of warps: the warps' maxima are distinct keys, so the smallest of them, T, has
    // at least that many keys >= T and no key < T is in the top k. A zero T (a warp without real keys) keeps every
    // real key. If at most TOPK_SELECT_COMPACT keys survive, warp 0 selects among them; otherwise the block
    // selection below runs on the untouched registers.
    if (k <= TOPK_SELECT_NT / 32) {
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        const uint64_t wmax = topk_select_warp_max(best);
        if (lane == 0) {
            red[warp] = wmax;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint64_t t = red[0];
            for (int w = 1; w < TOPK_SELECT_NT / 32; ++w) {
                t = red[w] < t ? red[w] : t;
            }
            red[TOPK_SELECT_NT / 32] = t;
            compact_count = 0;
        }
        __syncthreads();
        const uint64_t threshold = red[TOPK_SELECT_NT / 32];
        int count = 0;
#pragma unroll
        for (int j = 0; j < TOPK_SELECT_PER_THREAD; ++j) {
            count += v[j] != 0 && v[j] >= threshold;
        }
        int prefix = count; // inclusive prefix over the warp
#pragma unroll
        for (int d = 1; d < 32; d *= 2) {
            const int p = __shfl_up_sync(0xffffffff, prefix, d);
            if (lane >= d) {
                prefix += p;
            }
        }
        const int total = __shfl_sync(0xffffffff, prefix, 31);
        int base = lane == 0 ? atomicAdd(&compact_count, total) : 0;
        base = __shfl_sync(0xffffffff, base, 0) + prefix - count;
#pragma unroll
        for (int j = 0; j < TOPK_SELECT_PER_THREAD; ++j) {
            if (v[j] != 0 && v[j] >= threshold) {
                if (base < TOPK_SELECT_COMPACT) {
                    compact[base] = v[j];
                }
                ++base;
            }
        }
        __syncthreads();
        // The buffer order depends on the atomic order; the selection below does not.
        if (compact_count <= TOPK_SELECT_COMPACT) {
            if (warp == 0) {
                uint64_t c[TOPK_SELECT_COMPACT / 32];
                uint64_t cbest = 0;
#pragma unroll
                for (int j = 0; j < TOPK_SELECT_COMPACT / 32; ++j) {
                    const int i = j * 32 + lane;
                    c[j] = i < compact_count ? compact[i] : 0;
                    cbest = c[j] > cbest ? c[j] : cbest;
                }
                for (int r = 0; r < k; ++r) {
                    const uint64_t winner = topk_select_warp_max(cbest);
                    if (lane == 0) {
                        out[r] = winner;
                    }
                    // only the lane holding the (unique) winner needs a new maximum
                    if (r + 1 < k && cbest == winner) {
                        cbest = 0;
#pragma unroll
                        for (int j = 0; j < TOPK_SELECT_COMPACT / 32; ++j) {
                            cbest = c[j] < winner && c[j] > cbest ? c[j] : cbest;
                        }
                    }
                }
            }
            return;
        }
    }

    for (int r = 0; r < k; ++r) {
        const uint64_t winner = topk_select_block_max(best, red);
        if (threadIdx.x == 0) {
            out[r] = winner;
        }
        // Only the thread holding the (unique) winner needs a new maximum; every other thread's maximum is still
        // its largest key below the winner. Several threads whose maximum is padding all stay at 0.
        if (r + 1 < k && best == winner) {
            best = 0;
#pragma unroll
            for (int j = 0; j < TOPK_SELECT_PER_THREAD; ++j) {
                best = v[j] < winner && v[j] > best ? v[j] : best;
            }
        }
    }
}

// grid (nrows), 32 threads, nchunks <= TOPK_SELECT_MERGE_WARP: lane l owns chunk list l (descending, k entries).
// Only a list head can be the next winner, and only the winning list advances: an exact k-way merge.
static __global__ void __launch_bounds__(32)
topk_select_merge_warp(const uint64_t * part, int * dst, const int nchunks, const int k) {
    topk_select_grid_dependency(); // part is written by topk_select_partial

    const int        lane = threadIdx.x;
    const int        row  = blockIdx.x;
    const uint64_t * p    = part + (int64_t) row * nchunks * k;

    int      pos  = 0;
    uint64_t head = lane < nchunks ? p[lane * k] : 0;
    for (int r = 0; r < k; ++r) {
        const uint64_t winner = topk_select_warp_max(head);
        if (lane == 0) {
            dst[(int64_t) row * k + r] = (int) ~(uint32_t) winner;
        }
        if (r + 1 < k && head == winner) {
            ++pos;
            head = pos < k ? p[lane * k + pos] : 0;
        }
    }
}

// grid (nrows); writes k column indices per row in descending key order (any number of chunk lists).
static __global__ void __launch_bounds__(TOPK_SELECT_NT)
topk_select_merge(const uint64_t * part, int * dst, const int nchunks, const int k) {
    __shared__ uint64_t red[TOPK_SELECT_NT / 32 + 1];

    topk_select_grid_dependency(); // part is written by topk_select_partial

    const int        row = blockIdx.x;
    const int        n   = nchunks * k;
    const uint64_t * p   = part + (int64_t) row * n;

    uint64_t prev = 0;
    for (int r = 0; r < k; ++r) {
        uint64_t best = 0;
        for (int i = threadIdx.x; i < n; i += TOPK_SELECT_NT) {
            const uint64_t c = p[i];
            if ((r == 0 || c < prev) && c > best) {
                best = c;
            }
        }
        prev = topk_select_block_max(best, red);
        if (threadIdx.x == 0) {
            dst[(int64_t) row * k + r] = (int) ~(uint32_t) prev;
        }
    }
}

static inline bool topk_select_supported(const int64_t ncols, const int64_t nrows, const int64_t k) {
    return k >= 1 && k <= TOPK_SELECT_MAX_K && ncols >= k && ncols <= INT32_MAX - TOPK_SELECT_CHUNK &&
           nrows >= 1 && nrows <= TOPK_SELECT_MAX_ROWS;
}

static inline int topk_select_nchunks(const int ncols) {
    return (ncols + TOPK_SELECT_CHUNK - 1) / TOPK_SELECT_CHUNK;
}

static inline size_t topk_select_scratch_elems(const int ncols, const int nrows, const int k) {
    return (size_t) nrows * topk_select_nchunks(ncols) * k;
}

// src: nrows contiguous rows of ncols floats; dst: nrows * k indices.
static inline void topk_select_launch(const float * src, int * dst, uint64_t * scratch, const int ncols,
                                      const int nrows, const int k, cudaStream_t stream) {
    const int nchunks = topk_select_nchunks(ncols);
    ggml_cuda_kernel_launch(topk_select_partial,
        ggml_cuda_kernel_launch_params(dim3(nchunks, nrows, 1), dim3(TOPK_SELECT_NT), 0, stream),
        src, scratch, ncols, k);
    if (nchunks <= TOPK_SELECT_MERGE_WARP) {
        ggml_cuda_kernel_launch(topk_select_merge_warp,
            ggml_cuda_kernel_launch_params(dim3(nrows, 1, 1), dim3(32), 0, stream),
            (const uint64_t *) scratch, dst, nchunks, k);
    } else {
        ggml_cuda_kernel_launch(topk_select_merge,
            ggml_cuda_kernel_launch_params(dim3(nrows, 1, 1), dim3(TOPK_SELECT_NT), 0, stream),
            (const uint64_t *) scratch, dst, nchunks, k);
    }
}

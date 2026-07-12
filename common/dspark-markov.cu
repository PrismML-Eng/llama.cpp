// Device-side dspark vanilla Markov resample. See dspark-markov.h for the
// contract and the exact math. Self-contained: depends only on the CUDA
// runtime, not on ggml/llama internals.

#include "dspark-markov.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <cstdio>

#define DSPARK_WARP            32
#define DSPARK_WARPS_PER_BLOCK 8
#define DSPARK_BLK_THREADS     (DSPARK_WARP * DSPARK_WARPS_PER_BLOCK) // 256
#define DSPARK_NBLOCKS         1024 // partial-argmax grid width (grid-stride)

#define DSPARK_CUDA_CHECK(call)                                                         \
    do {                                                                                \
        cudaError_t err_ = (call);                                                      \
        if (err_ != cudaSuccess) {                                                      \
            fprintf(stderr, "dspark-markov cuda error %s at %s:%d: %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err_));               \
            return false;                                                               \
        }                                                                               \
    } while (0)

struct dspark_markov_cuda {
    int64_t n_vocab = 0;
    int64_t rank    = 0;

    float * d_w1 = nullptr; // [n_vocab * rank]
    float * d_w2 = nullptr; // [n_vocab * rank]

    // per-round scratch (grown lazily to fit n_use)
    int64_t base_cap_rows = 0;   // rows currently allocated in d_base / h_base
    float * d_base   = nullptr;  // [base_cap_rows * n_vocab]
    float * h_base   = nullptr;  // pinned staging for the H2D of base logits
    int32_t * d_out  = nullptr;  // [base_cap_rows]
    int32_t * h_out  = nullptr;  // pinned staging for the D2H of the result ids

    int32_t * d_prev = nullptr;  // device-resident chained prev token (1 int)

    float   * d_part_val = nullptr; // [DSPARK_NBLOCKS]
    int32_t * d_part_idx = nullptr; // [DSPARK_NBLOCKS]

    cudaStream_t stream = nullptr;
};

// Fused GEMV + add-base + per-block partial argmax for one position k.
// One warp reduces one vocab row's rank-length dot product (coalesced reads
// of w2), lane 0 adds the base logit and tracks the warp's running best over
// its grid-stride rows; the block then reduces its 8 warp bests to one
// (value, index) partial. Ties resolve to the lowest vocab index, matching
// the host path's strict-greater argmax.
__global__ void dspark_gemv_argmax_partial(
        const float * __restrict__ w1,
        const float * __restrict__ w2,
        const float * __restrict__ base,
        const int32_t * __restrict__ prev,
        int64_t n_vocab,
        int     rank,
        int64_t k,
        float   * __restrict__ part_val,
        int32_t * __restrict__ part_idx) {
    extern __shared__ float w1s[]; // rank floats: the prev token's w1 row

    const int tid  = threadIdx.x;
    const int lane = tid & (DSPARK_WARP - 1);
    const int wid  = tid >> 5;

    const int64_t w1_off = (int64_t) prev[0] * (int64_t) rank;
    for (int r = tid; r < rank; r += DSPARK_BLK_THREADS) {
        w1s[r] = w1[w1_off + r];
    }
    __syncthreads();

    const int64_t base_off = k * n_vocab;

    float   bestv = -CUDART_INF_F;
    int32_t besti = 0;

    const int64_t gw     = (int64_t) blockIdx.x * DSPARK_WARPS_PER_BLOCK + wid;
    const int64_t stride = (int64_t) gridDim.x  * DSPARK_WARPS_PER_BLOCK;
    for (int64_t v = gw; v < n_vocab; v += stride) {
        const float * w2row = w2 + v * (int64_t) rank;
        float acc = 0.0f;
        for (int r = lane; r < rank; r += DSPARK_WARP) {
            acc += w1s[r] * w2row[r];
        }
        // warp tree-reduction: its floating-point accumulation order differs from
        // the host scalar loop (and from BLAS), so the correction -- and thus a
        // near-tie argmax -- is functionally equivalent, not bit-identical, to the
        // host paths. This only changes which speculative draft is proposed; the
        // target verify arbitrates the committed output. See dspark-markov.h.
        #pragma unroll
        for (int o = DSPARK_WARP / 2; o > 0; o >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, o);
        }
        if (lane == 0) {
            const float logit = base[base_off + v] + acc;
            // strict >: rows are visited in increasing v, so the lowest index
            // wins ties (same as the host scalar/BLAS argmax).
            if (logit > bestv) {
                bestv = logit;
                besti = (int32_t) v;
            }
        }
    }

    __shared__ float   sv[DSPARK_WARPS_PER_BLOCK];
    __shared__ int32_t si[DSPARK_WARPS_PER_BLOCK];
    if (lane == 0) {
        sv[wid] = bestv;
        si[wid] = besti;
    }
    __syncthreads();

    if (tid == 0) {
        float   bv = sv[0];
        int32_t bi = si[0];
        #pragma unroll
        for (int w = 1; w < DSPARK_WARPS_PER_BLOCK; ++w) {
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) {
                bv = sv[w];
                bi = si[w];
            }
        }
        part_val[blockIdx.x] = bv;
        part_idx[blockIdx.x] = bi;
    }
}

// Reduce the per-block partials to the single global argmax, write it to
// out[k] and forward it into prev for the next position (the device-side
// chaining step). Single block.
__global__ void dspark_argmax_final(
        const float   * __restrict__ part_val,
        const int32_t * __restrict__ part_idx,
        int       nparts,
        int64_t   k,
        int32_t * __restrict__ out,
        int32_t * __restrict__ prev) {
    __shared__ float   sv[DSPARK_BLK_THREADS];
    __shared__ int32_t si[DSPARK_BLK_THREADS];

    const int tid = threadIdx.x;

    float   bv = -CUDART_INF_F;
    int32_t bi = 0;
    for (int i = tid; i < nparts; i += DSPARK_BLK_THREADS) {
        if (part_val[i] > bv || (part_val[i] == bv && part_idx[i] < bi)) {
            bv = part_val[i];
            bi = part_idx[i];
        }
    }
    sv[tid] = bv;
    si[tid] = bi;
    __syncthreads();

    for (int s = DSPARK_BLK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sv[tid + s] > sv[tid] || (sv[tid + s] == sv[tid] && si[tid + s] < si[tid])) {
                sv[tid] = sv[tid + s];
                si[tid] = si[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[k]  = si[0];
        prev[0] = si[0];
    }
}

static bool dspark_markov_ensure_capacity(dspark_markov_cuda * ctx, int64_t rows) {
    if (rows <= ctx->base_cap_rows) {
        return true;
    }
    if (ctx->d_base) { cudaFree(ctx->d_base);        ctx->d_base = nullptr; }
    if (ctx->h_base) { cudaFreeHost(ctx->h_base);    ctx->h_base = nullptr; }
    if (ctx->d_out)  { cudaFree(ctx->d_out);         ctx->d_out  = nullptr; }
    if (ctx->h_out)  { cudaFreeHost(ctx->h_out);     ctx->h_out  = nullptr; }
    ctx->base_cap_rows = 0;

    DSPARK_CUDA_CHECK(cudaMalloc(&ctx->d_base, (size_t) rows * (size_t) ctx->n_vocab * sizeof(float)));
    DSPARK_CUDA_CHECK(cudaHostAlloc(&ctx->h_base, (size_t) rows * (size_t) ctx->n_vocab * sizeof(float), cudaHostAllocDefault));
    DSPARK_CUDA_CHECK(cudaMalloc(&ctx->d_out, (size_t) rows * sizeof(int32_t)));
    DSPARK_CUDA_CHECK(cudaHostAlloc(&ctx->h_out, (size_t) rows * sizeof(int32_t), cudaHostAllocDefault));
    ctx->base_cap_rows = rows;
    return true;
}

dspark_markov_cuda * dspark_markov_cuda_init(
        const float * w1,
        const float * w2,
        int64_t       n_vocab,
        int64_t       markov_rank) {
    if (w1 == nullptr || w2 == nullptr || n_vocab <= 0 || markov_rank <= 0) {
        return nullptr;
    }

    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev <= 0) {
        fprintf(stderr, "dspark-markov: no CUDA device available\n");
        return nullptr;
    }

    dspark_markov_cuda * ctx = new dspark_markov_cuda();
    ctx->n_vocab = n_vocab;
    ctx->rank    = markov_rank;

    const size_t nbytes = (size_t) n_vocab * (size_t) markov_rank * sizeof(float);

    bool ok = true;
    auto guard = [&](cudaError_t e, const char * what) {
        if (e != cudaSuccess) {
            fprintf(stderr, "dspark-markov init: %s: %s\n", what, cudaGetErrorString(e));
            ok = false;
        }
    };

    guard(cudaStreamCreate(&ctx->stream), "cudaStreamCreate");
    guard(cudaMalloc(&ctx->d_w1, nbytes), "cudaMalloc w1");
    guard(cudaMalloc(&ctx->d_w2, nbytes), "cudaMalloc w2");
    guard(cudaMalloc(&ctx->d_prev, sizeof(int32_t)), "cudaMalloc prev");
    guard(cudaMalloc(&ctx->d_part_val, DSPARK_NBLOCKS * sizeof(float)), "cudaMalloc part_val");
    guard(cudaMalloc(&ctx->d_part_idx, DSPARK_NBLOCKS * sizeof(int32_t)), "cudaMalloc part_idx");
    if (ok) {
        guard(cudaMemcpy(ctx->d_w1, w1, nbytes, cudaMemcpyHostToDevice), "H2D w1");
        guard(cudaMemcpy(ctx->d_w2, w2, nbytes, cudaMemcpyHostToDevice), "H2D w2");
    }

    if (!ok) {
        dspark_markov_cuda_free(ctx);
        return nullptr;
    }
    return ctx;
}

void dspark_markov_cuda_free(dspark_markov_cuda * ctx) {
    if (ctx == nullptr) {
        return;
    }
    if (ctx->d_w1)       cudaFree(ctx->d_w1);
    if (ctx->d_w2)       cudaFree(ctx->d_w2);
    if (ctx->d_prev)     cudaFree(ctx->d_prev);
    if (ctx->d_part_val) cudaFree(ctx->d_part_val);
    if (ctx->d_part_idx) cudaFree(ctx->d_part_idx);
    if (ctx->d_base)     cudaFree(ctx->d_base);
    if (ctx->d_out)      cudaFree(ctx->d_out);
    if (ctx->h_base)     cudaFreeHost(ctx->h_base);
    if (ctx->h_out)      cudaFreeHost(ctx->h_out);
    if (ctx->stream)     cudaStreamDestroy(ctx->stream);
    delete ctx;
}

bool dspark_markov_cuda_resample(
        dspark_markov_cuda * ctx,
        const float *        base_logits,
        int32_t              id_last,
        int32_t              n_use,
        int32_t *            out_ids) {
    if (ctx == nullptr || base_logits == nullptr || out_ids == nullptr || n_use <= 0) {
        return false;
    }
    if (!dspark_markov_ensure_capacity(ctx, n_use)) {
        return false;
    }

    const int64_t V = ctx->n_vocab;
    const int     R = (int) ctx->rank;

    // Stage this round's base logits into pinned host memory, then one H2D.
    const size_t base_elems = (size_t) n_use * (size_t) V;
    memcpy(ctx->h_base, base_logits, base_elems * sizeof(float));
    DSPARK_CUDA_CHECK(cudaMemcpyAsync(ctx->d_base, ctx->h_base, base_elems * sizeof(float),
                                      cudaMemcpyHostToDevice, ctx->stream));

    // Seed the chained prev with the anchor token (prev for k == 0).
    DSPARK_CUDA_CHECK(cudaMemcpyAsync(ctx->d_prev, &id_last, sizeof(int32_t),
                                      cudaMemcpyHostToDevice, ctx->stream));

    const size_t shmem = (size_t) R * sizeof(float);
    for (int32_t k = 0; k < n_use; ++k) {
        dspark_gemv_argmax_partial<<<DSPARK_NBLOCKS, DSPARK_BLK_THREADS, shmem, ctx->stream>>>(
                ctx->d_w1, ctx->d_w2, ctx->d_base, ctx->d_prev,
                V, R, (int64_t) k, ctx->d_part_val, ctx->d_part_idx);
        dspark_argmax_final<<<1, DSPARK_BLK_THREADS, 0, ctx->stream>>>(
                ctx->d_part_val, ctx->d_part_idx, DSPARK_NBLOCKS, (int64_t) k,
                ctx->d_out, ctx->d_prev);
    }
    DSPARK_CUDA_CHECK(cudaGetLastError());

    DSPARK_CUDA_CHECK(cudaMemcpyAsync(ctx->h_out, ctx->d_out, (size_t) n_use * sizeof(int32_t),
                                      cudaMemcpyDeviceToHost, ctx->stream));
    DSPARK_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

    memcpy(out_ids, ctx->h_out, (size_t) n_use * sizeof(int32_t));
    return true;
}

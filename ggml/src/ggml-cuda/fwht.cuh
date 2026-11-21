#include "common.cuh"

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op);

// In-register block FWHT over N floats held as NE = N/NT per thread, element i*NT + tid in reg[i].
// Shared by the standalone kernel (fwht_cuda_block) and the fused FWHT + q8_1 quantizer so both
// produce bit-identical transforms. `s` must hold N floats.
// B < N: independent transforms of width B over consecutive B-element segments (a block-diagonal Hadamard), i.e. only
// the stages with stride < B. Each stage is the same add/subtract whichever level runs it, so a segment's result is
// bit-identical to a width-B transform.
template <int N, int NT, int B = N>
__device__ __forceinline__ void ggml_cuda_fwht_block_butterfly(float * reg, float * s, const int tid, const int lane) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");
    static_assert(B <= N && N % B == 0 && B >= warp_size, "bad FWHT segment width");

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size && h < B; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT && h < B; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < B; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst);

// Threads per block for the one-block-per-transform FWHT kernels (fwht_cuda_block, fwht_quantize_q8_1) at width n.
// 256 by default; with few transforms in flight (decode and speculative verify: 64 or fewer) min(n/2, 1024), so each
// thread keeps at least two elements and the few blocks there are use more of the SM. The butterfly runs its stages
// in the same order for any block width, so the result is bit-identical. GGML_CUDA_FWHT_NT overrides the wide value
// (256 disables it).
static inline int ggml_cuda_fwht_block_threads(const int n, const int64_t n_transforms) {
    static const int wide_override = getenv("GGML_CUDA_FWHT_NT") ? atoi(getenv("GGML_CUDA_FWHT_NT")) : 0;
    if (n_transforms > 64 || n < 512) {
        return 256;
    }
    const int nt = wide_override > 0 ? wide_override : (n/2 < 1024 ? n/2 : 1024);
    return (nt == 512 || nt == 1024) && nt <= n ? nt : 256;
}

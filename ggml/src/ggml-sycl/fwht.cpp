#include "fwht.hpp"

#include <cmath>
#define P 1.0f
#define N -1.0f

// constant Hadamard matrix via Paley I construction
static constexpr float H12[12][12] = {
    { P, P, P, P, P, P, P, P, P, P, P, P },
    { P, N, P, N, P, P, P, N, N, N, P, N },
    { P, N, N, P, N, P, P, P, N, N, N, P },
    { P, P, N, N, P, N, P, P, P, N, N, N },
    { P, N, P, N, N, P, N, P, P, P, N, N },
    { P, N, N, P, N, N, P, N, P, P, P, N },
    { P, N, N, N, P, N, N, P, N, P, P, P },
    { P, P, N, N, N, P, N, N, P, N, P, P },
    { P, P, P, N, N, N, P, N, N, P, N, P },
    { P, P, P, P, N, N, N, P, N, N, P, N },
    { P, N, P, P, P, N, N, N, P, N, N, P },
    { P, P, N, P, P, P, N, N, N, P, N, N }
};

static constexpr float H20[20][20] = {
    { P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P },
    { P, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N },
    { P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P },
    { P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P },
    { P, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N },
    { P, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N },
    { P, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N },
    { P, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N },
    { P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P },
    { P, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N },
    { P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P },
    { P, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N },
    { P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P },
    { P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P },
    { P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P },
    { P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P },
    { P, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N },
    { P, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N },
    { P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P },
    { P, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N }
};

#undef P
#undef N

template <int N>
static void fwht_kernel(const float * __restrict__ src, float * __restrict__ dst, const int64_t n_rows,
                        const float scale, const sycl::nd_item<2> & item) {
    const sycl::sub_group sg = item.get_sub_group();

    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    float     reg[el_w];
    const int lane = sg.get_local_linear_id();

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * WARP_SIZE + lane] * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    for (int h = WARP_SIZE; h < N; h *= 2) {
        const int step = h / WARP_SIZE;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

template <int N>
static void launch_fwht(const float * src, float * dst, const int64_t n_rows, const float scale,
                        dpct::queue_ptr stream) {
    constexpr int rows_per_block = 4;

    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
    const sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel<N>(src, dst, n_rows, scale, item);
                         });
}

// Wide blocks: one row per work-group instead of per sub-group, so each work-item
// keeps N/NT values rather than N/WARP_SIZE. Butterflies below the sub-group width
// still shuffle; those up to NT go through work-group local memory; the rest stay
// in registers.
template <int N, int NT>
static void fwht_kernel_wide(const float * __restrict__ src,
                             float * __restrict__ dst,
                             const int64_t            n_rows,
                             const float              scale,
                             const sycl::nd_item<2> & item,
                             float *                  smem) {
    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int el_w = N / NT;
    static_assert(el_w >= 1 && N % NT == 0, "row must be a whole number of work-group widths");

    const int tid = item.get_local_id(1);

    float reg[el_w];
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * NT + tid] * scale;
    }

    const sycl::sub_group sg   = item.get_sub_group();
    const int             lane = sg.get_local_linear_id();

    // Butterflies inside the sub-group, same pattern as the narrow kernel.
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies from the sub-group width up to NT: the partner lane is outside
    // this sub-group, so it goes through work-group local memory instead of a shuffle.
    for (int h = WARP_SIZE; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            smem[j * NT + tid] = reg[j];
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = smem[j * NT + (tid ^ h)];
            reg[j]           = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // Butterflies across registers: h is a multiple of NT, so the partner of element
    // i*NT + tid lives in reg[i + h/NT] on the same work-item.
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
        for (int j = 0; j < el_w; j += 2 * step) {
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <int N, int NT>
static void launch_fwht_wide(const float *   src,
                             float *         dst,
                             const int64_t   n_rows,
                             const float     scale,
                             dpct::queue_ptr stream) {
    const sycl::range<2> global(n_rows, NT);
    const sycl::range<2> local(1, NT);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(N), cgh);
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel_wide<N, NT>(src, dst, n_rows, scale, item, get_pointer(smem));
                         });
    });
}

template <int N, int m>
static void kronecker_kernel(const float * __restrict__ src,
                             float * __restrict__ dst,
                             const int64_t            n_rows,
                             const float              scale,
                             const sycl::nd_item<2> & item) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    const sycl::sub_group sg = item.get_sub_group();

    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int blocks_per_group = N / m;
    constexpr int el_w             = blocks_per_group / WARP_SIZE;
    static_assert(el_w >= 1 && blocks_per_group % WARP_SIZE == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    float     reg[el_w * m];
    const int lane = sg.get_local_linear_id();

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int b_idx = i * WARP_SIZE + lane;

#pragma unroll
        for (int j = 0; j < m; ++j) {
            reg[i * m + j] = src[b_idx * m + j] * scale;
        }
    }

#pragma unroll
    for (int b = 0; b < el_w; ++b) {
        float z[m] = { 0.0f };

#pragma unroll
        for (int i = 0; i < m; ++i) {
#pragma unroll
            for (int j = 0; j < m; ++j) {
                const float h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#pragma unroll
        for (int i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
#pragma unroll
            for (int k = 0; k < m; ++k) {
                const float val  = reg[j * m + k];
                const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#pragma unroll
    for (int h = WARP_SIZE; h < blocks_per_group; h *= 2) {
        const int step = h / WARP_SIZE;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int s = 0; s < step; ++s) {
#pragma unroll
                for (int k = 0; k < m; ++k) {
                    const float x = reg[(j + s) * m + k];
                    const float y = reg[(j + s + step) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + step) * m + k] = x - y;
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int b_idx = i * WARP_SIZE + lane;
#pragma unroll
        for (int k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

template <int N, int m>
static void launch_kronecker(const float *   src,
                             float *         dst,
                             const int64_t   n_rows,
                             const float     scale,
                             dpct::queue_ptr stream) {
    constexpr int rows_per_block = 4;

    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
    const sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             kronecker_kernel<N, m>(src, dst, n_rows, scale, item);
                         });
}

bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src, dst)) {
        return false;
    }
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int     n    = (int) src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const float *   src_d  = (const float *) src->data;
    float *         dst_d  = (float *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    const float scale = 1.0f / std::sqrt((float) n);

    switch (n) {
        case 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            return true;
        case 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            return true;
        case 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            return true;
        case 384:
            launch_kronecker<384, 12>(src_d, dst_d, rows, scale, stream);
            return true;
        case 768:
            launch_kronecker<768, 12>(src_d, dst_d, rows, scale, stream);
            return true;
        case 640:
            launch_kronecker<640, 20>(src_d, dst_d, rows, scale, stream);
            return true;
        case 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, rows, scale, stream);
            return true;
        case 1024:
            launch_fwht_wide<1024, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 2048:
            launch_fwht_wide<2048, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 4096:
            launch_fwht_wide<4096, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 8192:
            launch_fwht_wide<8192, 256>(src_d, dst_d, rows, scale, stream);
            return true;
        default:
            return false;
    }
}

// ARC-LAB (Torchit1/llama.cpp arc-b580): FWHT with the Hadamard sign flip done during the load
// (dst = H * (src * signs)), F32 or F16 input. Kept beside the general FWHT above, which carries the
// wide and Kronecker sizes from upstream; only the fused sign-flip entry point below uses it.
namespace fwht_signed_impl {

template <int N, typename T, bool has_signs>
static void fwht_kernel(const T * __restrict__ src, float * __restrict__ dst, const int64_t n_rows,
                        const float scale, const float * __restrict__ signs, const int n_blk,
                        const sycl::nd_item<2> & item) {
    const sycl::sub_group sg = item.get_sub_group();

    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    float     reg[el_w];
    const int lane = sg.get_local_linear_id();

    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = (float) src[i * WARP_SIZE + lane] * scale;
        if constexpr (has_signs) {
            reg[i] *= signs_row[i * WARP_SIZE + lane];
        }
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    for (int h = WARP_SIZE; h < N; h *= 2) {
        const int step = h / WARP_SIZE;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

// Wide rows: one work-group per row, butterflies through local memory.
template <int N, typename T, bool has_signs>
static void fwht_kernel_slm(const T * __restrict__ src, float * __restrict__ dst, const float scale,
                            const float * __restrict__ signs, const int n_blk, float * s, const sycl::nd_item<1> & item) {
    const int64_t r   = item.get_group(0);
    const int     tid = item.get_local_id(0);
    const int     nt  = item.get_local_range(0);

    src += r * N;
    dst += r * N;
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    for (int i = tid; i < N; i += nt) {
        float v = (float) src[i] * scale;
        if constexpr (has_signs) {
            v *= signs_row[i];
        }
        s[i] = v;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int h = 1; h < N; h *= 2) {
        for (int idx = tid; idx < N / 2; idx += nt) {
            const int   j = (idx / h) * 2 * h + idx % h;
            const float x = s[j];
            const float y = s[j + h];
            s[j]     = x + y;
            s[j + h] = x - y;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    for (int i = tid; i < N; i += nt) {
        dst[i] = s[i];
    }
}

template <int N, typename T, bool has_signs>
static void launch_fwht(const T * src, float * dst, const int64_t n_rows, const float scale, const float * signs,
                        const int n_blk, dpct::queue_ptr stream) {
    if constexpr (N <= 1024) {
        constexpr int rows_per_block = 4;

        const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

        // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
        const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
        const sycl::range<2> local(rows_per_block, WARP_SIZE);

        stream->parallel_for(sycl::nd_range<2>(global, local),
                             [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 fwht_kernel<N, T, has_signs>(src, dst, n_rows, scale, signs, n_blk, item);
                             });
    } else {
        constexpr int nt = 256;
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> s(sycl::range<1>(N), cgh);
            cgh.parallel_for(sycl::nd_range<1>(n_rows * nt, nt), [=](sycl::nd_item<1> item) {
                fwht_kernel_slm<N, T, has_signs>(src, dst, scale, signs, n_blk,
                                                 s.get_multi_ptr<sycl::access::decorated::no>().get(), item);
            });
        });
    }
}

template <typename T>
static bool fwht_dispatch(const T * src_d, float * dst_d, const int n, const int64_t rows, const float scale,
                          const float * signs, const int n_blk, dpct::queue_ptr stream) {
    switch (n) {
#define FWHT_CASE(NN)                                                                      \
        case NN:                                                                           \
            if (signs) {                                                                   \
                launch_fwht<NN, T, true>(src_d, dst_d, rows, scale, signs, n_blk, stream); \
            } else {                                                                       \
                launch_fwht<NN, T, false>(src_d, dst_d, rows, scale, nullptr, 1, stream);  \
            }                                                                              \
            return true;
        FWHT_CASE(64)
        FWHT_CASE(128)
        FWHT_CASE(256)
        FWHT_CASE(512)
        FWHT_CASE(1024)
        FWHT_CASE(2048)
        FWHT_CASE(4096)
        FWHT_CASE(8192)
#undef FWHT_CASE
        default:
            return false;
    }
}

static bool fwht_run(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                     const ggml_tensor * signs_t) {
    if ((src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (ggml_nelements(src) != ggml_nelements(dst) || !ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int     n    = (int) dst->ne[0];
    const int64_t rows = ggml_nelements(dst) / n;

    const float * signs = nullptr;
    int           n_blk = 1;
    if (signs_t) {
        if (signs_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs_t) || signs_t->ne[0] % n != 0) {
            return false;
        }
        signs = (const float *) signs_t->data;
        n_blk = (int) (signs_t->ne[0] / n);
    }

    const float scale = 1.0f / std::sqrt((float) n);
    if (src->type == GGML_TYPE_F32) {
        return fwht_dispatch((const float *) src->data, (float *) dst->data, n, rows, scale, signs, n_blk, ctx.stream());
    }
    return fwht_dispatch((const sycl::half *) src->data, (float *) dst->data, n, rows, scale, signs, n_blk, ctx.stream());
}

} // namespace fwht_signed_impl

bool ggml_sycl_op_fwht_signed(ggml_backend_sycl_context & ctx, const ggml_tensor * src, const ggml_tensor * signs,
                              ggml_tensor * dst) {
    return fwht_signed_impl::fwht_run(ctx, src, dst, signs);
}

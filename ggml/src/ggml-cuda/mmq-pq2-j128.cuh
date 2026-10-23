#pragma once

// Adapted from the Yukon cuda.fast Bonsai engine's J=128 PQ2_0 prefill kernel:
// https://github.com/Layr-Labs/cudafast-bonsai2-9b-engine/tree/2c62b6cdd3ddabb4a418672e259c6be2baff20eb
// (llama.cpp/ggml/src/ggml-cuda/mmq-pq2-j128.cuh), MIT licensed, see licenses/yukon-cudafast.
// Port changes: ordinary launches (no programmatic stream serialization), no D1 activations, no fused GLU or
// plane-copy launches (no persistent weight copies); device code is compiled for sm_90+ only (the stream-k path
// uses TMA bulk copies and mbarriers). The GLU/ADD/D1 template paths are kept to stay diffable with the source,
// but only <PURE, ADD=false, GLU=false, COMPACT> is instantiated. Numerics are unchanged from the source.

// Originally written for GB10 (sm_121); runs on any NVIDIA sm_90+. J=128 PQ2_0 x q8_1 matmul (prefill) at 16 warps per SM.
//
// Bit-exact replacement for mul_mat_q<GGML_TYPE_PQ2_0, 128, false> (+ its stream-k fixup):
//  * same work partition: CTA (b, q) runs every K segment that block b of the old kernel runs (the
//    same kbc arithmetic, including the alignment of segment bounds to the old 256-value K
//    iteration), for row half q (64 rows) of each 128 x 128 output tile. It writes dst for a
//    segment that ends its tile and tmp_fixup[b] (layout [j][i], stride 128) for the trailing one;
//    the unchanged mul_mat_q_stream_k_fixup completes the stream-k tiles. Pure tiling = one block
//    per tile.
//  * same accumulation per output element: sum starts at +0 for every segment and, for every
//    32-value sub-block in increasing K order, sum = fma(RN(idot*dA), dB, sum) with dA = (float)
//    x->d and dB = the q8_1 D4 scale, i.e. the old `sum += C*dA*dB` (I2F, FMUL, FFMA). idot is the
//    exact int32 dot product of the same 32 products (identical int8 tiles, same mma).
//  * 2-FFMA epilogue: the mma accumulator starts at 0x4B400000 (12582912.0f = 1.5*2^23); for
//    |idot| <= 2^22 (here |idot| <= 32*2*128) the int32 result is the float 12582912 + idot
//    exactly. ndA = -12582912*dA is exact (dA: <= 11 significant bits) and
//    fma(D_f, dA, ndA) = RN(idot*dA) = FMUL(I2F(idot), dA) (the product itself is exact, <= 24
//    significant bits). For dA > 0, and for dA == +0 with idot >= 0, the two are bit-identical
//    including the sign of zero (exhaustively tested over all fp16 dA and all idot).
// CTA = 64 rows x 128 columns, 8 warps, 2 CTAs per SM (16 warps, <= 128 registers, 46-50 KB of
// shared memory each); each warp owns a 32 x 32 output block (32 fp32 accumulators per thread).
// X (weights) are unpacked once per CTA into a shared int8 tile (same codec as
// ggml_cuda_mmq_load_tiles_pq2_0) from raw words fetched one K block ahead. The activation tile is
// double-buffered: with cp.async for pure tiling, and for the stream-k shapes by one TMA bulk copy
// per K block (completing on an mbarrier) plus a transposed copy of its D4 scales in shared memory.

#include "common.cuh"
#include "unary.cuh"
// cp-async.cuh (no include guard) is included by mmq.cuh before this header

#include <cstdint>

#define GGML_CUDA_PQ2_J128_MAGIC 0x4B400000 // 12582912.0f = 1.5*2^23

static constexpr int ggml_cuda_pq2_j128_ys = 36; // ints per activation column record: 4 floats d4 + 32 ints qs (= MMQ_TILE_Y_K)
static constexpr int ggml_cuda_pq2_j128_xs = 36; // ints per weight row: 32 ints int8 values + dA + 3 pad (stride % 8 == 4)

static __device__ __forceinline__ void ggml_cuda_pq2_j128_mma(int (&d)[4], const int (&a)[4], const int b0, const int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE && !defined(GGML_USE_HIP)
    const int m = GGML_CUDA_PQ2_J128_MAGIC;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(m));
#else
    GGML_UNUSED_VARS(d, a, b0, b1);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_ldsm_x4(int (&r)[4], const uint32_t addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && !defined(GGML_USE_HIP)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
#else
    GGML_UNUSED_VARS(r, addr);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_ldsm_x2(int (&r)[2], const uint32_t addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && !defined(GGML_USE_HIP)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
#else
    GGML_UNUSED_VARS(r, addr);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE && !defined(GGML_USE_HIP)
    asm volatile("cp.async.commit_group;");
#endif
}

template <int N>
static __device__ __forceinline__ void ggml_cuda_pq2_j128_cp_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE && !defined(GGML_USE_HIP)
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
#endif
}

// mbarrier + 1D bulk copy (TMA) helpers
static __device__ __forceinline__ void ggml_cuda_pq2_j128_mbar_init(const uint32_t bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
#else
    NO_DEVICE_CODE;
#endif
}
static __device__ __forceinline__ void ggml_cuda_pq2_j128_mbar_fence_init() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP)
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#else
    NO_DEVICE_CODE;
#endif
}
static __device__ __forceinline__ void ggml_cuda_pq2_j128_mbar_wait(const uint32_t bar, const uint32_t parity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP)
    asm volatile("{\n\t.reg .pred p;\n\tC50_WAIT_%=:\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t@!p bra C50_WAIT_%=;\n\t}"
        :: "r"(bar), "r"(parity) : "memory");
#else
    NO_DEVICE_CODE;
#endif
}
static __device__ __forceinline__ void ggml_cuda_pq2_j128_bulk_y(const uint32_t dst, const void * src, const uint32_t bytes, const uint32_t bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP)
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
#else
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_hint(const char * p) {
#if defined(__CUDA_ARCH__) && !defined(GGML_USE_HIP)
    asm volatile("prefetch.global.L2 [%0];" :: "l"(__cvta_generic_to_global(p)));
#else
    GGML_UNUSED(p);
#endif
}

static __device__ __forceinline__ float ggml_cuda_pq2_j128_lds_f32(const uint32_t addr) {
    float v;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(addr));
    return v;
}

static __device__ __forceinline__ float2 ggml_cuda_pq2_j128_lds_v2f32(const uint32_t addr) {
    float2 v;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "r"(addr));
    return v;
}

static __device__ __forceinline__ uint4 ggml_cuda_pq2_j128_lds_v4(const uint32_t addr) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    return v;
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_sts_v4(const uint32_t addr, const int a, const int b, const int c, const int d) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_sts_b32(const uint32_t addr, const int a) {
    asm volatile("st.shared.b32 [%0], %1;" :: "r"(addr), "r"(a) : "memory");
}

// Raw words of one thread's X task: row `row` of the slice, 8 code bytes kq*8..kq*8+7 of PQ2 block kb
// (the 32 weights of ints kq*8..kq*8+7), read as three aligned 32-bit words (the block is 2-byte
// aligned, C22c), plus the fp16 scale bits.
struct ggml_cuda_pq2_j128_xraw {
    uint32_t w0, w1, w2;
    uint32_t sh; // funnel shift: 0 or 16
    uint32_t d;
};

static __device__ __forceinline__ void ggml_cuda_pq2_j128_xload(ggml_cuda_pq2_j128_xraw & r, const char * blk, const int kq) {
    const uint8_t  * qsi = (const uint8_t *) blk + sizeof(ggml_half) + kq*8;
    const uint32_t   mis = (uint32_t) (reinterpret_cast<uintptr_t>(qsi) & 3); // 0 or 2
    const uint32_t * qwi = reinterpret_cast<const uint32_t *>(qsi - mis);
    r.w0 = qwi[0];
    r.w1 = qwi[1];
    r.w2 = qwi[1 + mis/2]; // w1 again when aligned: no read past the 8 bytes
    r.sh = 8*mis;
    r.d  = kq == 0 ? (uint32_t) *(const uint16_t *) blk : 0u;
}

// Same 2-bit -> int8 codec expressions as ggml_cuda_mmq_load_tiles_pq2_0: the 8 ints kq*8..kq*8+7 of the
// row and (kq == 0) the dA bits, in registers; xput stores them (ints to addr, dA to int 32 of the row).
struct ggml_cuda_pq2_j128_xdec {
    int v[8];
    int d;
};

static __device__ __forceinline__ ggml_cuda_pq2_j128_xdec ggml_cuda_pq2_j128_xdecode(const ggml_cuda_pq2_j128_xraw & r) {
    const uint32_t qw[2] = {__funnelshift_r(r.w0, r.w1, r.sh), __funnelshift_r(r.w1, r.w2, r.sh)};
    ggml_cuda_pq2_j128_xdec o;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = j % 2 == 0 ? int(int16_t(qw[j/2] & 0xFFFFu)) : int(qw[j/2]) >> 16;

        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        o.v[2*j + 0] = __byte_perm(qe, qo, 0x5140);
        o.v[2*j + 1] = __byte_perm(qe, qo, 0x7362);
    }
    o.d = __float_as_int(__half2float(__ushort_as_half((unsigned short) r.d)));
    return o;
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_xput(const ggml_cuda_pq2_j128_xdec & o, const uint32_t addr, const int kq) {
    ggml_cuda_pq2_j128_sts_v4(addr,      o.v[0], o.v[1], o.v[2], o.v[3]);
    ggml_cuda_pq2_j128_sts_v4(addr + 16, o.v[4], o.v[5], o.v[6], o.v[7]);
    if (kq == 0) {
        ggml_cuda_pq2_j128_sts_b32(addr + 4*32, o.d);
    }
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_xstore(const ggml_cuda_pq2_j128_xraw & r, const uint32_t addr, const int kq) {
    ggml_cuda_pq2_j128_xput(ggml_cuda_pq2_j128_xdecode(r), addr, kq);
}

// Activation tile of one K block: the 128 column records of the tile column, 18432 contiguous bytes.
// The 32 records of column group wc (4608 bytes) are copied by the G threads of that group
// (member index m), so a column group only depends on its own copies.
template <int G>
static __device__ __forceinline__ void ggml_cuda_pq2_j128_yload(const uint32_t dst, const char * src, const int m) {
    constexpr int nchunks = 32*ggml_cuda_pq2_j128_ys*(int) sizeof(int)/16; // 288
#pragma unroll
    for (int c0 = 0; c0 < nchunks; c0 += G) {
        const int c = c0 + m;
        if (c0 + G <= nchunks || c < nchunks) {
            cp_async_cg_16<256>(dst + 16*c, src + 16*c);
        }
    }
}

// One K block (128 values) of the warp's 32 x 32 output block. acc[rt][ct][l]: rt = 16-row tile,
// ct = 8-column tile, l = mma C element (rows g, g+8 x columns 2c, 2c+1).
// xs/ys: shared addresses of the warp's first X row / first Y column record in the current buffers.
template <bool DT = false>
static __device__ __forceinline__ void ggml_cuda_pq2_j128_compute(float (&acc)[2][4][4], const uint32_t xs, const uint32_t ys,
        const uint32_t dbt = 0) {
    const int lane = threadIdx.x;
    const int g    = lane / 4;
    const int c    = lane % 4;
    constexpr int XS = 4*ggml_cuda_pq2_j128_xs; // bytes
    constexpr int YS = 4*ggml_cuda_pq2_j128_ys;
    // ldmatrix row addresses of this lane: A (x4): rows lane%16, ints (lane/16)*4; B (x2): columns lane%8,
    // ints 4 + ((lane/8)%2)*4 (the first 4 ints of a column record are the D4 scales).
    const uint32_t xa = xs + (lane % 16)*XS + (lane / 16)*16;
    const uint32_t ya = ys + (lane % 8)*YS + 16 + ((lane / 8) % 2)*16;
    const uint32_t xd = xs + g*XS + 4*32;   // dA of row g
    const uint32_t yd = ys + 2*c*YS;        // D4 scales of column 2c

    float dA[2][2], ndA[2][2];
#pragma unroll
    for (int rt = 0; rt < 2; ++rt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            dA [rt][h] = ggml_cuda_pq2_j128_lds_f32(xd + (rt*16 + 8*h)*XS);
            ndA[rt][h] = -12582912.0f*dA[rt][h];
        }
    }

    // sub-block outer: A of one sub-block live (8 registers), dB per (column tile, sub-block)
#pragma unroll 4
    for (int s = 0; s < 4; ++s) {
        int A[2][4];
#pragma unroll
        for (int rt = 0; rt < 2; ++rt) {
            ggml_cuda_pq2_j128_ldsm_x4(A[rt], xa + rt*16*XS + s*32);
        }
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
            float dB[2];
            if constexpr (DT) {
                // transposed D4 scales: [s][column] floats of the warp's columns, columns 2c, 2c+1 adjacent
                const float2 v = ggml_cuda_pq2_j128_lds_v2f32(dbt + s*512 + (ct*8 + 2*c)*4);
                dB[0] = v.x; dB[1] = v.y;
            } else {
                dB[0] = ggml_cuda_pq2_j128_lds_f32(yd + ct*8*YS + 4*s);
                dB[1] = ggml_cuda_pq2_j128_lds_f32(yd + ct*8*YS + YS + 4*s);
            }
            int B[2];
            ggml_cuda_pq2_j128_ldsm_x2(B, ya + ct*8*YS + s*32);
#pragma unroll
            for (int rt = 0; rt < 2; ++rt) {
                int C[4];
                ggml_cuda_pq2_j128_mma(C, A[rt], B[0], B[1]);
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const float t = __fmaf_rn(__int_as_float(C[l]), dA[rt][l/2], ndA[rt][l/2]);
                    acc[rt][ct][l] = __fmaf_rn(t, dB[l%2], acc[rt][ct][l]);
                }
            }
        }
    }
}

// Exact compact compute loop with one K sub-block live at a time.
template <bool DT = false>
static __device__ __forceinline__ void ggml_cuda_pq2_ws_compute_compact(float (&acc)[2][4][4], const uint32_t xs, const uint32_t ys,
        const uint32_t dbt = 0) {
    const int lane = threadIdx.x;
    const int g    = lane / 4;
    const int c    = lane % 4;
    constexpr int XS = 4*ggml_cuda_pq2_j128_xs; // bytes
    constexpr int YS = 4*ggml_cuda_pq2_j128_ys;
    // ldmatrix row addresses of this lane: A (x4): rows lane%16, ints (lane/16)*4; B (x2): columns lane%8,
    // ints 4 + ((lane/8)%2)*4 (the first 4 ints of a column record are the D4 scales).
    const uint32_t xa = xs + (lane % 16)*XS + (lane / 16)*16;
    const uint32_t ya = ys + (lane % 8)*YS + 16 + ((lane / 8) % 2)*16;
    const uint32_t xd = xs + g*XS + 4*32;   // dA of row g
    const uint32_t yd = ys + 2*c*YS;        // D4 scales of column 2c

    float dA[2][2], ndA[2][2];
#pragma unroll
    for (int rt = 0; rt < 2; ++rt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            dA [rt][h] = ggml_cuda_pq2_j128_lds_f32(xd + (rt*16 + 8*h)*XS);
            ndA[rt][h] = -12582912.0f*dA[rt][h];
        }
    }

    // sub-block outer: A of one sub-block live (8 registers), dB per (column tile, sub-block)
#pragma unroll 1
    for (int s = 0; s < 4; ++s) {
        int A[2][4];
#pragma unroll
        for (int rt = 0; rt < 2; ++rt) {
            ggml_cuda_pq2_j128_ldsm_x4(A[rt], xa + rt*16*XS + s*32);
        }
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
            float dB[2];
            if constexpr (DT) {
                // transposed D4 scales: [s][column] floats of the warp's columns, columns 2c, 2c+1 adjacent
                const float2 v = ggml_cuda_pq2_j128_lds_v2f32(dbt + s*512 + (ct*8 + 2*c)*4);
                dB[0] = v.x; dB[1] = v.y;
            } else {
                dB[0] = ggml_cuda_pq2_j128_lds_f32(yd + ct*8*YS + 4*s);
                dB[1] = ggml_cuda_pq2_j128_lds_f32(yd + ct*8*YS + YS + 4*s);
            }
            int B[2];
            ggml_cuda_pq2_j128_ldsm_x2(B, ya + ct*8*YS + s*32);
#pragma unroll
            for (int rt = 0; rt < 2; ++rt) {
                int C[4];
                ggml_cuda_pq2_j128_mma(C, A[rt], B[0], B[1]);
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const float t = __fmaf_rn(__int_as_float(C[l]), dA[rt][l/2], ndA[rt][l/2]);
                    acc[rt][ct][l] = __fmaf_rn(t, dB[l%2], acc[rt][ct][l]);
                }
            }
        }
    }
}

// (D1 compute not ported: prism has no D1 activation producer.)

template <bool PURE, bool COMPACT = false, bool D1 = false>
static __device__ __forceinline__ void ggml_cuda_pq2_j128_segment(
        float (&acc)[2][4][4], const uint32_t smem, const char * __restrict__ x, const int * y,
        const int row0, const int stride_row_x, const int ncols_y, const int col0,
        const int kb_start, const int kb_stop, uint32_t & ph) {
    constexpr int G  = 64;                      // threads per column group (warps w%4 == wc)
    constexpr int XS = 4*ggml_cuda_pq2_j128_xs; // bytes per X row
    constexpr int YS = 4*ggml_cuda_pq2_j128_ys; // bytes per Y column record
    constexpr uint32_t YBYTES = 128*YS;
    constexpr uint32_t XBYTES = 64*XS;
    constexpr int HINT_D = 4; // L2 hint of the X rows 4 K blocks ahead
    constexpr int BS = (int) sizeof(block_pq2_0);

    const int tid = threadIdx.y*32 + threadIdx.x;
    const int wr  = threadIdx.y / 4;
    const int wc  = threadIdx.y % 4;
    const int m   = (threadIdx.y / 4)*32 + threadIdx.x; // member index in the column group

    // X task of this thread: row tid/4 of the slice, quarter kq of the block
    const int kq = tid % 4;
    const char * xblk = x + ((int64_t) (row0 + tid/4)*stride_row_x + kb_start)*BS;
    const int hint_off = kq == 1 ? BS - 1 : 0; // lanes kq 0/1 hint the first/last byte of the block
    const bool hint_lane = kq < 2;
    const uint32_t xst = smem + 2*YBYTES + (tid/4)*XS + kq*32; // X tile store address

    const int64_t y_kb_bytes = (int64_t) ncols_y*YS;
    const char * yb = (const char *) y + kb_start*y_kb_bytes + (int64_t) (col0 + wc*32)*YS; // column group's records
    const uint32_t ycp = smem + wc*32*YS; // column group's records in Y buffer 0

    const uint32_t xw = smem + 2*YBYTES + wr*32*XS; // warp's X rows
    const uint32_t yw = smem + wc*32*YS;            // warp's Y columns (buffer 0)

    if constexpr (PURE) {
        GGML_UNUSED(ph);
        ggml_cuda_pq2_j128_xraw xr;

        // prologue: block kb_start (the global X words are read before the barrier: registers only)
        ggml_cuda_pq2_j128_xload(xr, xblk, kq);
        __syncthreads(); // the previous segment's last compute no longer reads any tile
        ggml_cuda_pq2_j128_yload<G>(ycp, yb, m);
        ggml_cuda_pq2_j128_cp_commit();
        ggml_cuda_pq2_j128_xstore(xr, xst, kq);
        if (kb_start + 1 < kb_stop) {
            yb   += y_kb_bytes;
            xblk += BS;
            ggml_cuda_pq2_j128_yload<G>(ycp + YBYTES, yb, m);
            ggml_cuda_pq2_j128_cp_commit();
            ggml_cuda_pq2_j128_xload(xr, xblk, kq);
            ggml_cuda_pq2_j128_cp_wait<1>();
        } else {
            ggml_cuda_pq2_j128_cp_wait<0>();
        }
        __syncthreads();
        // invariant at the top of iteration kb: xblk/yb point at block kb + 1 (when it exists)
        for (int kb = kb_start; kb < kb_stop; ++kb) {
            const uint32_t cur = (kb - kb_start) & 1;
            if (hint_lane && kb + HINT_D < kb_stop) {
                ggml_cuda_pq2_j128_hint(xblk + (HINT_D - 1)*BS + hint_off);
            }
            if constexpr (D1) static_assert(!D1, "D1 not ported");
            else if constexpr (COMPACT) ggml_cuda_pq2_ws_compute_compact<false>(acc, xw, yw + cur*YBYTES);
            else ggml_cuda_pq2_j128_compute<false>(acc, xw, yw + cur*YBYTES);
            if (kb + 1 >= kb_stop) {
                break;
            }
            __syncthreads(); // X tile and Y buffer `cur` no longer read
            ggml_cuda_pq2_j128_xstore(xr, xst, kq);
            if (kb + 2 < kb_stop) {
                yb   += y_kb_bytes;
                xblk += BS;
                ggml_cuda_pq2_j128_yload<G>(ycp + cur*YBYTES, yb, m);
                ggml_cuda_pq2_j128_cp_commit();
                ggml_cuda_pq2_j128_xload(xr, xblk, kq);
                ggml_cuda_pq2_j128_cp_wait<1>();
            } else {
                ggml_cuda_pq2_j128_cp_wait<0>();
            }
            __syncthreads(); // X tile (block kb+1) and Y buffer `cur ^ 1` complete
        }
    } else {
        // Y through one bulk copy (TMA) per K block; after it lands, the D4 headers of the 128 columns are
        // transposed by threads 0..127 into dBT[buf][s][col] (2 KB per buffer) so that a lane reads the
        // scales of its two columns with one 8-byte load. mbarriers live in the padding of X row 0.
        const uint32_t bar  = smem + 2*YBYTES + 136;          // buffer b: X row b bytes 136..143 (pad of the 144-byte rows)
        const uint32_t dbt0 = smem + 2*YBYTES + XBYTES;       // 2 x 2048 bytes
        const char * ysrc = (const char *) y + kb_start*y_kb_bytes + (int64_t) col0*YS;
        auto transpose = [&](const uint32_t buf) { // threads 0..127: column tid of Y buffer buf
            if (tid < 128) {
                if constexpr (D1) {
                    // Only scale0 is read by the D1 consumer; keep the original
                    // allocation/strides so fallback modes and barriers stay identical.
                    const float h = ggml_cuda_pq2_j128_lds_f32(smem + buf*YBYTES + tid*YS);
                    ggml_cuda_pq2_j128_sts_b32(dbt0 + buf*2048 + tid*4, __float_as_int(h));
                } else {
                    const uint4 h = ggml_cuda_pq2_j128_lds_v4(smem + buf*YBYTES + tid*YS);
                    const uint32_t d = dbt0 + buf*2048 + tid*4;
                    ggml_cuda_pq2_j128_sts_b32(d + 0*512, (int) h.x);
                    ggml_cuda_pq2_j128_sts_b32(d + 1*512, (int) h.y);
                    ggml_cuda_pq2_j128_sts_b32(d + 2*512, (int) h.z);
                    ggml_cuda_pq2_j128_sts_b32(d + 3*512, (int) h.w);
                }
            }
        };
        ggml_cuda_pq2_j128_xraw xr;
        ggml_cuda_pq2_j128_xload(xr, xblk, kq);
        __syncthreads(); // the previous segment's last compute no longer reads any tile
        if (tid == 0) {
            ggml_cuda_pq2_j128_bulk_y(smem, ysrc, YBYTES, bar);
        }
        ggml_cuda_pq2_j128_xstore(xr, xst, kq);
        if (kb_start + 1 < kb_stop) {
            if (tid == 0) {
                ggml_cuda_pq2_j128_bulk_y(smem + YBYTES, ysrc + y_kb_bytes, YBYTES, bar + 144);
            }
            xblk += BS;
            ggml_cuda_pq2_j128_xload(xr, xblk, kq);
        }
        ggml_cuda_pq2_j128_mbar_wait(bar, ph & 1);
        ph ^= 1u;
        transpose(0);
        __syncthreads();
        const uint32_t dbw = dbt0 + wc*32*4; // the warp's columns
        for (int kb = kb_start; kb < kb_stop; ++kb) {
            const uint32_t cur = (kb - kb_start) & 1;
            if (hint_lane && kb + HINT_D < kb_stop) {
                ggml_cuda_pq2_j128_hint(xblk + (HINT_D - 1)*BS + hint_off);
            }
            if constexpr (D1) static_assert(!D1, "D1 not ported");
            else if constexpr (COMPACT) ggml_cuda_pq2_ws_compute_compact<true>(acc, xw, yw + cur*YBYTES, dbw + cur*2048);
            else ggml_cuda_pq2_j128_compute<true>(acc, xw, yw + cur*YBYTES, dbw + cur*2048);
            if (kb + 1 >= kb_stop) {
                break;
            }
            __syncthreads(); // X tile, Y buffer `cur` and dBT `cur` no longer read
            ggml_cuda_pq2_j128_xstore(xr, xst, kq);
            if (kb + 2 < kb_stop) {
                if (tid == 0) {
                    ggml_cuda_pq2_j128_bulk_y(smem + cur*YBYTES, ysrc + (int64_t) (kb - kb_start + 2)*y_kb_bytes, YBYTES, bar + 144*cur);
                }
                xblk += BS;
                ggml_cuda_pq2_j128_xload(xr, xblk, kq);
            }
            ggml_cuda_pq2_j128_mbar_wait(bar + 144*(cur ^ 1), (ph >> (cur ^ 1)) & 1); // Y of block kb+1 landed
            ph ^= 1u << (cur ^ 1);
            transpose(cur ^ 1);
            __syncthreads(); // X tile (block kb+1) and dBT `cur ^ 1` complete
        }
        GGML_UNUSED(ycp); GGML_UNUSED(m); GGML_UNUSED(yb);
    }
}

static __device__ __forceinline__ void ggml_cuda_pq2_j128_store(
        const float (&acc)[2][4][4], float * out, const int stride_col, const int row_base, const int col_base,
        const int j_max) {
    const int lane = threadIdx.x;
    const int g    = lane / 4;
    const int c    = lane % 4;
#pragma unroll
    for (int rt = 0; rt < 2; ++rt) {
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int j = col_base + ct*8 + 2*c + l % 2;
                if (j > j_max) {
                    continue;
                }
                const int i = row_base + rt*16 + g + 8*(l / 2);
                out[(int64_t) j*stride_col + i] = acc[rt][ct][l];
            }
        }
    }
}

// C74: the store of a completed tile with the residual ADD folded in: out = acc + res, the f32 sum of the value the
// plain store would write and the residual (one rounded add, as k_bin_bcast's op_add). All residual loads are issued
// before the first store: out may be res itself (an in-place ADD), and each element is read and written by this
// thread only.
static __device__ __forceinline__ void ggml_cuda_pq2_j128_store_add(
        float (&acc)[2][4][4], float * out, const float * res, const int stride_col, const int row_base, const int col_base,
        const int j_max) {
    const int lane = threadIdx.x;
    const int g    = lane / 4;
    const int c    = lane % 4;
#pragma unroll
    for (int rt = 0; rt < 2; ++rt) {
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int j = col_base + ct*8 + 2*c + l % 2;
                if (j > j_max) {
                    continue;
                }
                const int i = row_base + rt*16 + g + 8*(l / 2);
                acc[rt][ct][l] = __fadd_rn(acc[rt][ct][l], res[(int64_t) j*stride_col + i]);
            }
        }
    }
    ggml_cuda_pq2_j128_store(acc, out, stride_col, row_base, col_base, j_max);
}

// C74: L2 prefetch of the residual rows of a CTA's tile (64 rows x 128 columns, 256-byte column pieces): thread t
// requests half t % 2 of column t / 2. Issued when a segment that adds the residual starts.
static __device__ __forceinline__ void ggml_cuda_pq2_j128_res_hint(const float * res, const int stride_col, const int row0,
        const int col0, const int ncols_dst) {
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int j = col0 + tid / 2;
    if (j < ncols_dst) {
        ggml_cuda_pq2_j128_hint((const char *) (res + (int64_t) j*stride_col + row0 + (tid % 2)*32));
    }
}

// GLU epilogue for a gate/up row-interleaved weight (ggml_cuda_glu_pair): in every 16-row mma tile rows 0-7 are
// gate rows of 8 features and rows 8-15 the up rows of the same features, so a thread holds gate (row g) and up
// (row g + 8) of feature row_base/2 + 8*rt + g. It stores silu(gate)*up there with the expression of
// unary_gated_op_kernel<op_silu, float> (the SWIGLU node's value); out has the gate output's layout.
static __device__ __forceinline__ void ggml_cuda_pq2_j128_store_glu(
        const float (&acc)[2][4][4], float * out, const int stride_col, const int row_base, const int col_base,
        const int j_max) {
    const int lane = threadIdx.x;
    const int g    = lane / 4;
    const int c    = lane % 4;
#pragma unroll
    for (int rt = 0; rt < 2; ++rt) {
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int j = col_base + ct*8 + 2*c + l;
                if (j > j_max) {
                    continue;
                }
                const int i = row_base/2 + rt*8 + g;
                out[(int64_t) j*stride_col + i] = ggml_cuda_op_silu_single(acc[rt][ct][l]) * acc[rt][ct][l + 2];
            }
        }
    }
}

// grid = old grid * 2 (1D: row half q = blockIdx.x % 2 fastest), block = (32, 8), 2 CTAs per SM.
// PURE: pure tiling (grid == number of tiles), block b is tile b with the full K range.
// ADD (C74): the residual ADD that is this matmul's only reader is folded in. A segment that runs its whole tile
// (K blocks 0..nkb) writes dst_add = acc + res; the first segment of a stream-k tile that another block started
// writes its partial sum to dst as before, and mul_mat_q_pq2_j128_fixup_add completes that tile into dst_add.
// res and dst_add have dst's layout (stride_col_dst). dst_add may be dst or res; it overlaps nothing else.
// GLU (pure tiling only): x is a gate/up row-interleaved weight of 2*F rows, dst the F-row gate output (see
// ggml_cuda_pq2_j128_store_glu); every accumulator is the one the separate gate and up launches compute.
template <bool PURE, bool ADD, bool GLU = false, bool COMPACT = false, bool D1 = false>
__launch_bounds__(256, 2)
static __global__ void mul_mat_q_pq2_j128(
        const char * __restrict__ x, const int * y, float * dst, float * tmp_fixup,
        const int nrows_x, const int nkb, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int ncols_dst, const int ntx, const float * res, float * dst_add) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP)
    constexpr int I    = 128; // tile of the old kernel
    constexpr int J    = 128;
    constexpr int ROWS = 64;
    constexpr int QS   = I / ROWS;
    constexpr int blocks_per_iter = MMQ_ITER_K / QK_PQ2_0; // segment alignment of the old kernel

    extern __shared__ int smem_pq2_j128[];
    const uint32_t smem = (uint32_t) __cvta_generic_to_shared(smem_pq2_j128);

    const int q = blockIdx.x % QS;
    const int b = blockIdx.x / QS;
    const int nb_old = gridDim.x / QS;

    // Same index math as mul_mat_q (nchannels = nsamples = 1): kbc runs over (it, jt, kb), jt fastest.
    const int nty = nrows_x / I;
    const int64_t W = (int64_t) ntx*nty*nkb;
    int kbc      = (int) ((int64_t) b      *W / nb_old);
    int kbc_stop = (int) ((int64_t) (b + 1)*W / nb_old);
    kbc      -= (kbc      % nkb) % blocks_per_iter;
    kbc_stop -= (kbc_stop % nkb) % blocks_per_iter;

    const int wr = threadIdx.y / 4;
    const int wc = threadIdx.y % 4;
    const int row_in_tile = q*ROWS + wr*32; // the warp's rows in the 128-row tile
    const int col_in_tile = wc*32;

    int kb0_start = kbc % nkb;
    int kb0_stop  = min(nkb, kb0_start + kbc_stop - kbc);


    if constexpr (!ADD) {
        GGML_UNUSED(res); GGML_UNUSED(dst_add);
    }
    uint32_t ph = 0; // stream-k: wait parity per Y buffer (bit b = buffer b)
    if constexpr (!PURE) {
        const uint32_t bar = smem + (uint32_t) (2*128*4*ggml_cuda_pq2_j128_ys) + 136u;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            ggml_cuda_pq2_j128_mbar_init(bar);
            ggml_cuda_pq2_j128_mbar_init(bar + 144);
            ggml_cuda_pq2_j128_mbar_fence_init();
        }
        __syncthreads();
    }
    if constexpr (PURE) {
        const int jt = b % ntx;
        const int it = b / ntx;
        float acc[2][4][4] = {{{0.0f}}};
        if constexpr (ADD) {
            ggml_cuda_pq2_j128_res_hint(res, stride_col_dst, it*I + q*ROWS, jt*J, ncols_dst);
        }
        ggml_cuda_pq2_j128_segment<PURE,COMPACT,D1>(acc, smem, x, y, it*I + q*ROWS, stride_row_x, ncols_y, jt*J, 0, nkb, ph);
        if constexpr (ADD) {
            const int64_t off = (int64_t) jt*J*stride_col_dst + it*I;
            ggml_cuda_pq2_j128_store_add(acc, dst_add + off, res + off, stride_col_dst, row_in_tile, col_in_tile,
                                         ncols_dst - jt*J - 1);
        } else if constexpr (GLU) {
            ggml_cuda_pq2_j128_store_glu(acc, dst + (int64_t) jt*J*stride_col_dst + it*(I/2), stride_col_dst, row_in_tile,
                                         col_in_tile, ncols_dst - jt*J - 1);
        } else {
            ggml_cuda_pq2_j128_store(acc, dst + (int64_t) jt*J*stride_col_dst + it*I, stride_col_dst, row_in_tile, col_in_tile,
                                     ncols_dst - jt*J - 1);
        }
        return;
    }

    // One code path for every segment (a single inlined segment keeps one register allocation): a
    // segment that ends its tile writes dst; the trailing one (which ends inside a tile) writes the
    // partial sums of all 128 columns to the fixup buffer of block b and is necessarily the last.
    while (kbc < kbc_stop) {
        const int t  = kbc / nkb;
        const int jt = t % ntx;
        const int it = t / ntx;
        const bool ends_tile = kb0_stop == nkb;
        const bool whole     = ADD && ends_tile && kb0_start == 0; // completes its tile alone: adds the residual
        float acc[2][4][4] = {{{0.0f}}};
        if (whole) {
            ggml_cuda_pq2_j128_res_hint(res, stride_col_dst, it*I + q*ROWS, jt*J, ncols_dst);
        }
        ggml_cuda_pq2_j128_segment<PURE,COMPACT,D1>(acc, smem, x, y, it*I + q*ROWS, stride_row_x, ncols_y, jt*J, kb0_start, kb0_stop, ph);
        if (!ends_tile) {
            ggml_cuda_pq2_j128_store(acc, tmp_fixup + (int64_t) b*(J*I), I, row_in_tile, col_in_tile, J - 1);
            return;
        }
        if constexpr (ADD) {
            if (whole) {
                const int64_t off = (int64_t) jt*J*stride_col_dst + it*I;
                ggml_cuda_pq2_j128_store_add(acc, dst_add + off, res + off, stride_col_dst, row_in_tile, col_in_tile,
                                             ncols_dst - jt*J - 1);
            } else {
                ggml_cuda_pq2_j128_store(acc, dst + (int64_t) jt*J*stride_col_dst + it*I, stride_col_dst, row_in_tile,
                                         col_in_tile, ncols_dst - jt*J - 1);
            }
        } else {
            ggml_cuda_pq2_j128_store(acc, dst + (int64_t) jt*J*stride_col_dst + it*I, stride_col_dst, row_in_tile, col_in_tile,
                                     ncols_dst - jt*J - 1);
        }

        kbc += nkb;
        kbc -= kbc % nkb;
        kb0_start = 0;
        kb0_stop  = min(nkb, kbc_stop - kbc);
    }
#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
}

// C74: mul_mat_q_stream_k_fixup<type, 128, false> (no ids, one channel and one sample) for the ADD variant. The
// partial sums of the preceding blocks are combined in the same order into the same `sum`, the completed value is
// dst + sum (what the plain fixup stores to dst), and dst_add receives that value plus the residual.
template <ggml_type type>
static __global__ void mul_mat_q_pq2_j128_fixup_add(
        const float * dst, const float * tmp_last_tile, const float * res, float * dst_add, const uint3 blocks_per_ne00,
        const int nrows_x, const int ncols_dst, const int stride_col_dst, const uint3 ntx) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && !defined(GGML_USE_HIP)
    constexpr int  J               = 128;
    constexpr bool fallback        = false;
    constexpr int  warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int  nwarps          = (ggml_cuda_mmq_get_nthreads(type, J, fallback) / 2) / warp_size;
    constexpr int  I               = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int  qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int  ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
    constexpr int  blocks_per_iter = ITER_K / qk;

    float sum[J / nwarps] = {0.0f};
    const int i = blockIdx.y*warp_size + threadIdx.x;

    const int nty = (nrows_x + I - 1) / I;

    const int bidx0 = blockIdx.x;

    int kbc0      = int64_t(blockIdx.x)    *(ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
    int kbc0_stop = int64_t(blockIdx.x + 1)*(ntx.z*nty*blocks_per_ne00.z) / gridDim.x;

    kbc0      -= fastmodulo(kbc0,      blocks_per_ne00) % blocks_per_iter;
    kbc0_stop -= fastmodulo(kbc0_stop, blocks_per_ne00) % blocks_per_iter;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, blocks_per_ne00) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, blocks_per_ne00) == fastdiv(kbc0_stop, blocks_per_ne00) && fastmodulo(kbc0_stop, blocks_per_ne00) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    bool any_fixup = false;

    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        int kbc = int64_t(bidx)*(ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
        kbc -= fastmodulo(kbc, blocks_per_ne00) % blocks_per_iter;

        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            sum[j0/nwarps] += tmp_last_tile[bidx*(J*I) + j*I + i];
        }

        if (fastmodulo(kbc, blocks_per_ne00) == 0 || fastdiv(kbc, blocks_per_ne00) < fastdiv(kbc0, blocks_per_ne00)) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    if (!any_fixup) {
        return;
    }

    const uint2 tmp2 = fast_div_modulo(fastdiv(kbc0, blocks_per_ne00), ntx);
    const int jt = tmp2.y;
    const int it = tmp2.x;

    const int offset_dst = jt*J*stride_col_dst + it*I;
    dst     += offset_dst;
    res     += offset_dst;
    dst_add += offset_dst;

    const int j_max = ncols_dst - jt*J - 1;

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

        const float v = __fadd_rn(dst[j*stride_col_dst + i], sum[j0/nwarps]);
        dst_add[j*stride_col_dst + i] = __fadd_rn(v, res[j*stride_col_dst + i]);
    }
#else
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
}

// ---------------------------------------------------------------------------------------------
// Host side (included by mmq.cuh after mmq_args). Used on NVIDIA sm_90+ (see launch_mul_mat_q), for
// mul_mat_q<GGML_TYPE_PQ2_0, 128, false> without ids/channels/samples.

static constexpr size_t ggml_cuda_pq2_j128_smem() {
    return (size_t) (2*128*ggml_cuda_pq2_j128_ys + 64*ggml_cuda_pq2_j128_xs)*sizeof(int);
}

static bool ggml_cuda_mmq_pq2_j128_supported(const mmq_args & args) {
    return args.type_x == GGML_TYPE_PQ2_0 && args.ids_dst == nullptr && args.expert_bounds == nullptr &&
        args.y_scale == nullptr &&
        args.nchannels_x == 1 && args.nchannels_y == 1 && args.nsamples_x == 1 && args.nsamples_y == 1 &&
        args.nrows_x % 128 == 0 && args.ncols_x % (2*QK_PQ2_0) == 0 &&
        args.ncols_dst >= 1 && args.ncols_max == args.ncols_dst &&
        (uintptr_t) args.y % 16 == 0 &&
        args.nrows_x*args.stride_row_x*(int64_t) sizeof(block_pq2_0) < (int64_t(1) << 31) &&
        (args.ncols_x/QK_PQ2_0 + 1)*args.ncols_y*(int64_t) sizeof(block_q8_1_mmq) < (int64_t(1) << 31) &&
        // Same bound as the GGML_ASSERT in launch_mul_mat_q (ntiles_dst*blocks_per_ne00 < 2^30). It keeps the int
        // stream-k index kbc = b*ntiles*nkb/grid in range; shapes outside it take the existing path, which asserts.
        ((args.nrows_x + 127)/128)*((args.ncols_max + 127)/128)*(args.ncols_x/QK_PQ2_0) < (int64_t(1) << 30);
}

// GGML_CUDA_PQ2_J128: 0 = off, any other integer = on (any NVIDIA sm_90+ device). Unset: on for Hopper (cc 9.x), where
// it is validated bit-exact and faster; opt-in on Blackwell/GB10.
static bool ggml_cuda_pq2_j128_enabled(const int cc) {
    static const int env = [] {
        const char * e = getenv("GGML_CUDA_PQ2_J128");
        return e == nullptr ? -1 : (atoi(e) != 0 ? 1 : 0);
    }();
    return env >= 0 ? env == 1 : cc >= GGML_CUDA_CC_HOPPER && cc < 1000;
}

// Same grid rule as launch_mul_mat_q for the PQ2_0 J=128 (stream-k) config on NVIDIA.
static int ggml_cuda_mmq_pq2_j128_grid(const mmq_args & args, const int nsm) {
    const int nty    = (int) ((args.nrows_x + 127) / 128);
    const int ntx    = (int) ((args.ncols_max + 127) / 128);
    const int ntiles = nty*ntx;
    const int nwaves = (ntiles + nsm - 1) / nsm;
    const int eff    = 100*ntiles / (nsm*nwaves);
    return eff >= 90 ? ntiles : nsm;
}

static bool ggml_cuda_mmq_pq2_j128_needs_fixup(const mmq_args & args, const int grid) {
    const int ntiles = (int) ((args.nrows_x + 127) / 128) * (int) ((args.ncols_max + 127) / 128);
    return ntiles % grid != 0;
}

static bool ggml_cuda_mmq_pq2_j128_use(const mmq_args & args, const int nsm) {
    GGML_UNUSED(nsm);
    return ggml_cuda_mmq_pq2_j128_supported(args);
}

// Compact compute defaults to 1, a compact K-subblock loop in every J128 launch;
// 2 changes only the fused GLU launch. Grid partitioning and FP32 order stay
// identical. The goal is lower register pressure without expanded weights.
static int ggml_cuda_pq2_prefill_compact_mode() {
    static const int mode = []() { const char * v = getenv("GGML_CUDA_PREFILL_COMPACT"); return v ? atoi(v) : 1; }();
    return mode;
}

template <ggml_type type, bool PURE, bool ADD, bool COMPACT = false>
static void ggml_cuda_mmq_pq2_j128_launch(const mmq_args & args, const int grid, float * tmp_fixup, cudaStream_t stream) {
    constexpr size_t smem = ggml_cuda_pq2_j128_smem() + (PURE ? 0 : 4096); // stream-k: + 2 x 2 KB transposed D4 scales
    const int nkb = (int) (args.ncols_x / QK_PQ2_0);
    const int ntx = (int) ((args.ncols_max + 127) / 128);
    const auto kernel = (COMPACT || ggml_cuda_pq2_prefill_compact_mode() == 1)
        ? mul_mat_q_pq2_j128<PURE, ADD, false, true> : mul_mat_q_pq2_j128<PURE, ADD>;
    static bool attr_set[GGML_CUDA_MAX_DEVICES] = {};
    const int device = ggml_cuda_get_device();
    if (!attr_set[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, (int) cudaSharedmemCarveoutMaxShared));
        attr_set[device] = true;
    }
    kernel<<<dim3(2*grid, 1, 1), dim3(32, 8, 1), smem, stream>>>(args.x, args.y, args.dst, tmp_fixup, (int) args.nrows_x, nkb, (int) args.stride_row_x, (int) args.ncols_y,
        (int) args.nrows_dst, (int) args.ncols_dst, ntx, (const float *) nullptr, (float *) nullptr);
    CUDA_CHECK(cudaGetLastError());
    if (ADD && !PURE && ggml_cuda_mmq_pq2_j128_needs_fixup(args, grid)) {
        const uint3 blocks_per_ne00_fd = init_fastdiv_values(nkb);
        const uint3 ntx_fd             = init_fastdiv_values(ntx);
        mul_mat_q_pq2_j128_fixup_add<type><<<dim3(grid, 128/32, 1), dim3(32, 4, 1), 0, stream>>>((const float *) args.dst, (const float *) tmp_fixup, (const float *) nullptr, (float *) nullptr, blocks_per_ne00_fd,
            (int) args.nrows_x, (int) args.ncols_dst, (int) args.nrows_dst, ntx_fd);
    CUDA_CHECK(cudaGetLastError());
    } else if (!PURE && ggml_cuda_mmq_pq2_j128_needs_fixup(args, grid)) {
        const uint3 blocks_per_ne00_fd = init_fastdiv_values(nkb);
        const uint3 one_fd             = init_fastdiv_values(1);
        const uint3 ntx_fd             = init_fastdiv_values(ntx);
        mul_mat_q_stream_k_fixup<type, 128, false><<<dim3(grid, 128/32, 1), dim3(32, 4, 1), 0, stream>>>(nullptr, nullptr, args.dst, tmp_fixup, blocks_per_ne00_fd, (int) args.nrows_x, (int) args.ncols_dst,
            (int) args.nrows_dst, one_fd, (int) args.stride_channel_dst, one_fd, (int) args.stride_sample_dst, ntx_fd);
    CUDA_CHECK(cudaGetLastError());
    }
}

// Template so that only the PQ2_0 MMQ translation unit instantiates the kernels. Pure tiling loads the
// activation tiles with cp.async; the stream-k partition uses TMA bulk copies and transposed scales.
template <ggml_type type>
static void ggml_cuda_mmq_pq2_j128_run(const mmq_args & args, const int grid, float * tmp_fixup, cudaStream_t stream) {
    static_assert(type == GGML_TYPE_PQ2_0, "PQ2_0 only");
    const int ntiles = (int) ((args.nrows_x + 127) / 128) * (int) ((args.ncols_max + 127) / 128);
    if (grid == ntiles) {
        ggml_cuda_mmq_pq2_j128_launch<type, true, false>(args, grid, tmp_fixup, stream);
    } else {
        ggml_cuda_mmq_pq2_j128_launch<type, false, false>(args, grid, tmp_fixup, stream);
    }
}

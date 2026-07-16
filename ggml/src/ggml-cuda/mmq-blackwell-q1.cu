// mmq-blackwell-q1.cu  (Phase 3 — Blackwell int8 tensor-core MMQ for Q1_0/Q2_0, fixed per Opus review)
//
// Replaces the dp4a path for matrix-matrix prefill (M, N, K all % block-tile). Decode (M=1) is
// unaffected — dispatch falls through to mul_mat_vec_q.
//
// Block tile: bM=128, bN=64, bK=128 (= one block_q1_0 / block_q2_0 quant block).
// 8 warps (256 threads) per block; each warp owns 16 rows × 64 cols (= 8 m16n8 frags).
// 8 warps × 8 frags = 64 frags = bM*bN / (16*8) per block.
//
// SMEM layout (single stage, no double-buffering in this cut):
//   sA_i : [bM][bK/4] int32   — activations quantized int8, 4 K-positions per int (K-contiguous)
//   sB_i : [bN][bK/4] int32   — weights, K-contiguous int8 packs
//   sDa  : [bM] float32       — per-K-block (q1-block) activation absmax scale (fmaps / 127)
//   sDw  : [bN] float32       — per-K-block weight scale (half-precision d from block_q*_0)
//
// Each mma.sync.aligned.m16n8k32.s32.s8.s8.s32 consumes:
//   A: tile<16,8,int> = m16 × k32 (packed, 4 int8 per int)
//   B: tile<8,8,int>  = k32 × n8  (packed, 4 int8 per int)
//   D: tile<16,8,int> = m16 × n8 int32 accumulator
//
// Critical correctness note (Opus fix d): each kc-chunk has its OWN per-q-block float scale
// sDa[m,kc] * sDw[n,kc]. We cannot collapse int32 accumulators across K-chunks under one scale.
// Instead, per kc-chunk: do all bK/32 MMA sub-stages into a per-chunk int32 buffer, then convert
// to fp32 ONCE per chunk with the appropriate per-(m_idx, n_idx) scale, accumulating into a
// per-warp fp32 accumulator pool.
//
// Opt-in only: env var GGML_BLACKWELL_Q1 + cc == GGML_CUDA_CC_DGX_SPARK (conservative).

#include "common.cuh"
#include "mma.cuh"
#include "vecdotq.cuh"

#include <mutex>
#include <unordered_map>

bool ggml_cuda_mul_mat_q1_blackwell(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor *         src0,
                                    const ggml_tensor *         src1,
                                    ggml_tensor *               dst);

namespace blackwell_q1 {

using namespace ggml_cuda_mma;

// Block tile + dispatch.
static constexpr int bM = 256;  // bM=256 falsifier (Opus Phase 3.8): each warp owns 2 rowblocks, halved mblk redundancy
static constexpr int bN = 64;
static constexpr int bK = 128;  // == quant-block size, so scales align per K-chunk (Opus fix b).

static constexpr int SMEM_A_INTS = bM * (bK / 4);  // 128 * 32 = 4096 ints
static constexpr int SMEM_B_INTS = bN * (bK / 4);  // 64 * 32 = 2048 ints
static constexpr int SMEM_BYTES  = (SMEM_A_INTS + SMEM_B_INTS) * (int) sizeof(int)
                              + (bM + bN) * (int) sizeof(float);

// === Step 1: activation quantization (fp32 -> int8 with per-128 K-block absmax scale) ===
// 8 groups per 256-thread block, vectorized float4 loads + char4 stores.
__global__ void quant_act_per128(const float * __restrict__ x,
                                 int8_t * __restrict__ q,
                                 float * __restrict__ d,
                                 int M,
                                 int K) {
    const int ngroups = M * (K / 128);
    const int g       = blockIdx.x * 8 + (threadIdx.x / 32);
    if (g >= ngroups) {
        return;
    }
    const int      lane = threadIdx.x % 32;
    const int      m  = g / (K / 128);
    const int      kc = g % (K / 128);
    const float4 * xs = reinterpret_cast<const float4 *>(x + (size_t) m * K + kc * 128) + lane;
    float4         v  = *xs;
    float          amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    }
    const float scale = amax / 127.0f;
    const float inv   = scale > 0.f ? 1.0f / scale : 0.f;
    auto        q8    = [](float f) -> signed char {
        long r = lrintf(f);
        return (signed char) (r < -127 ? -127 : (r > 127 ? 127 : r));
    };
    char4 out = make_char4(q8(v.x * inv), q8(v.y * inv), q8(v.z * inv), q8(v.w * inv));
    *(reinterpret_cast<char4 *>(q + (size_t) m * K + kc * 128) + lane) = out;
    if (lane == 0) {
        d[(size_t) m * (K / 128) + kc] = scale;
    }
}

// === Step 2: weight repack (one-time per tensor; cached by (device, wdata, N, K, wbits)). ===
__global__ void repack_q1_dense(const block_q1_0 * __restrict__ W,
                                unsigned * __restrict__ bits_only,
                                float * __restrict__ dw,
                                long nblocks_total) {
    long b = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks_total) {
        return;
    }
    dw[b] = __half2float(*reinterpret_cast<const __half *>(W + b));
    const uint16_t * u16 = reinterpret_cast<const uint16_t *>(W + b);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        bits_only[b * 4 + w] = (unsigned) u16[1 + 2 * w] | ((unsigned) u16[2 + 2 * w] << 16);
    }
}

__global__ void repack_q2_dense(const block_q2_0 * __restrict__ W,
                                unsigned * __restrict__ bits_only,
                                float * __restrict__ dw,
                                long nblocks_total) {
    long b = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks_total) {
        return;
    }
    dw[b] = __half2float(*reinterpret_cast<const __half *>(W + b));
    const uint16_t * u16 = reinterpret_cast<const uint16_t *>(W + b);
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        bits_only[b * 8 + w] = (unsigned) u16[1 + 2 * w] | ((unsigned) u16[2 + 2 * w] << 16);
    }
}

struct DenseW {
    unsigned * bits_only;  // 4 words per Q1_0 block; 8 words per Q2_0 block
    float *    dw;
};

struct DenseKey {
    int          device;
    const void * wdata;
    long         N;
    long         K;
    int          wbits;

    bool operator==(const DenseKey & o) const {
        return device == o.device && wdata == o.wdata && N == o.N && K == o.K && wbits == o.wbits;
    }
};

struct DenseKeyHash {
    size_t operator()(const DenseKey & k) const {
        size_t h = std::hash<const void *>()(k.wdata);
        h ^= std::hash<long>()(k.N) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<long>()(k.K) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int>()((k.device << 4) | k.wbits) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

static DenseW get_dense_w(const void * wdata, long N, long K, int wbits, int device, cudaStream_t stream) {
    static std::unordered_map<DenseKey, DenseW, DenseKeyHash> cache;
    static std::mutex                                        cache_mutex;
    const DenseKey                                           key{ device, wdata, N, K, wbits };

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto                        it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    DenseW     d{};
    const long nb = N * (K / 128);
    CUDA_CHECK(cudaMalloc(&d.bits_only, nb * 16 * wbits));  // 4*wbits words/block, 4 B/word
    CUDA_CHECK(cudaMalloc(&d.dw, nb * sizeof(float)));
    if (wbits == 1) {
        repack_q1_dense<<<(unsigned) ((nb + 255) / 256), 256, 0, stream>>>(
            (const block_q1_0 *) wdata, d.bits_only, d.dw, nb);
    } else {
        repack_q2_dense<<<(unsigned) ((nb + 255) / 256), 256, 0, stream>>>(
            (const block_q2_0 *) wdata, d.bits_only, d.dw, nb);
    }
    // Publish only after repack is observably complete.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cache.emplace(key, d);
    return d;
}

// === Step 3: per-K-chunk bit-to-int8 expansion (uses warp-uniform bit indexing inline) ===
// Reads the repacked bits_only[(n_global * nblocks_row + kc) * (4 * wbits)] and packs 4 contiguous
// int8-packs (each int8 = the dequantized {-1,0,+1,+2} value) into an int32, K-major.
//
// For Q1_0: 32 bits per uint32 → 32 K-positions. Byte(b8) of the int32 maps K-pos (cInt*4+b8).
// For Q2_0: 16 2-bit fields per uint32 → 16 K-positions per word; 8 words per block.
//
// Both: bit_to_byte_q1 spreads {0,1} -> {0,127}; bit_to_byte_q2 maps q in {0..3} -> q-1 in
// {-1,0,+1,+2}. Combined with the per-kc scale at chunk close, this gives correct dequantized
// products. Inline (no LUT).

// === Step 4: the MMA kernel (clean rewrite applying Opus 5-fix review). ===
//
// 8 warps per block (256 threads). Each warp covers 1 m-rowblock (16 rows) × 8 n-colblocks (64 cols).
// Per warp: 8 m16n8 fragments. Per-warp accumulator pool is one float32-per-cell per warp
// (4 floats per thread per fragment) accumulating across kc-chunks; per-kc int32 intermediate
// buffer (4 ints per thread per fragment) reused across kc-chunks.
template <int WBITS>
__global__ __launch_bounds__(256) void lowbit_mma_ggml(
        const int8_t * __restrict__ Aq,    // [M,K] int8
        const float * __restrict__ dA,    // [M, K/128] fp32
        const unsigned * __restrict__ Wbits,  // [N*(K/128), 4*WBITS] uint32 packed bits
        const float * __restrict__ Wd,    // [N*(K/128)] fp32
        float * __restrict__ C,           // [M,N] fp32 output
        int M,
        int N,
        int K) {
    extern __shared__ __align__(16) char smem_raw[];
    int *  sA_i = reinterpret_cast<int *>(smem_raw);
    int *  sB_i = sA_i + SMEM_A_INTS;
    float *sDa  = reinterpret_cast<float *>(sB_i + SMEM_B_INTS);
    float *sDw  = sDa + bM;

    const int mblk = blockIdx.x, nblk = blockIdx.y;
    const int wid  = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    // `mma.cuh::tile<> get_i/get_j` use threadIdx.x directly, which assumes the block
    // operates as a single 32-thread warp-tile group. With 8 warps per block, those mappings
    // overflow the m=16 row range (e.g., threadIdx.x=248 -> get_i=62). We bypass the
    // shared helper for SMEM loads and use lane-based per-warp mappings that match the
    // canonical m16n8k32 register layout. mma() itself only consumes per-thread x[] values,
    // so this works as long as each warp independently holds a valid 16x8 fragment.
    auto load_A_lane = [&](auto & Afrag, const int * xs0, int stride_int) {
        // tile<16,8,int>::ne = 4 ints/thread.
        // l=0,1: row = lane/4 (0..7), cols (lane%4)*2, (lane%4)*2+1
        // l=2,3: row = 8 + lane/4 (8..15), cols (lane%4)*2, (lane%4)*2+1
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            int r = ((l >> 1) << 3) + (lane >> 2);
            int c = ((lane & 3) << 1) + (l & 1);
            Afrag.x[l] = xs0[r * stride_int + c];
        }
    };
    auto load_B_lane = [&](auto & Bfrag, const int * xs0, int stride_int) {
        // tile<8,8,int>::ne = 2 ints/thread.
        // l=0: row = lane/4 (0..7), col = lane%4
        // l=1: row = lane/4, col = (lane%4)+4
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            int r = lane >> 2;
            int c = (l << 2) + (lane & 3);
            Bfrag.x[l] = xs0[r * stride_int + c];
        }
    };

    // Each warp covers 32 rows × 64 cols (bM=256 / 8 warps = 32 rows/warp = 2 rowblocks
    //              × 8 colblocks). Halves mblk count vs bM=128, addressing Opus Q3 (the only
    //              4× mblk-redundancy lever large enough to close 200+ tok/s on a
    //              bandwidth-limited part).
    const int warp_m_base = wid * 32;
    const int warp_n_base = 0;
    constexpr int FRAG_PER_WARP = 16;  // 2 rowblocks × 8 colblocks per warp
    constexpr int BK_INT        = bK / 4;

    // Persistent per-warp fp32 accumulators per fragment (Opus fix d).
    float acc_f[FRAG_PER_WARP][4];
#pragma unroll
    for (int f = 0; f < FRAG_PER_WARP; ++f) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            acc_f[f][e] = 0.f;
        }
    }
    // Per-kc int32 accumulators live in the f-outer / kk-inner loop body (patch #3 below)
    // so only 4 ints per thread are live at any one time, not 32 — buys occupancy.

    const int nblocks_row = K / 128;
    const int nchunks     = K / bK;  // = K/128 since bK==128.

    tile<16, 8, int, DATA_LAYOUT_I_MAJOR> Dfrag;  // hoist for Stage E / final write's get_i/get_j
    constexpr int mma_k = 32;

#pragma unroll
    for (int kc = 0; kc < nchunks; ++kc) {
        __syncthreads();

        // ----- Stage A: pack activations into sA_i. -----
        // [bM][bK/4] ints = 128 * 32 = 4096 ints. 256 threads emit 16 ints each.
        // Opus minimum patch (#1): thread↔idx mapping is interleaved (`i*256 + threadIdx.x`) so
        // a warp issues 32 contiguous int loads (= one 128 B coalesced transaction) instead of
        // 32 scattered 4-byte requests. Single `reinterpret_cast<const int*>` (int8 -> int32)
        // load also avoids the prior per-byte sign-extension-or load pattern (a latent
        // correctness landmine: negative activations at p[0]=0xFF would poison all bytes via
        // the implicit int32 sign extension). `p` is 4-byte aligned (c0=cInt*4, row stride K
        // is a multiple of 128), and int8 reads in Q1_0/Q2_0 normalizations live in [-127,127]
        // so the int32 read gives the byte-bounded values directly.
        constexpr int A_INTS_PER_THREAD = SMEM_A_INTS / 256;  // = 16
#pragma unroll
        for (int i = 0; i < A_INTS_PER_THREAD; ++i) {
            int idx  = i * 256 + threadIdx.x;  // 0..4095, interleaved -> coalesced
            int r    = idx / BK_INT;
            int cInt = idx % BK_INT;
            const int8_t * p = Aq + (size_t) (mblk * bM + r) * K + kc * bK + cInt * 4;
            sA_i[idx] = *reinterpret_cast<const int32_t *>(p);  // one aligned 32-bit load
        }

        // ----- Stage B: pack weights into sB_i (K-stored-contiguous as int32 packs). -----
        // [bN][bK/4] ints = 64 * 32 = 2048 ints. 256 threads emit 8 ints each.
        // Opus minimum patch Q1 + Q2:
        //  Q1: thread<->idx is INTERLEAVED (`i*256 + threadIdx.x`) so a warp's 32 lanes read
        //      32 contiguous cInt positions of the same `(n, kc)` block, collapsing global
        //      footprint from ~8 scattered cache lines to one 16-byte block per warp-pass.
        //      On a bandwidth-limited part this is a real DRAM-transaction reduction, not
        //      just L2 reuse.
        //  Q2: 4 consecutive K-positions `cInt*4 .. cInt*4+3` always fit in a single uint32
        //      of `Wbits[blk_idx*4 + (cInt>>3)]` (both cInt*4 and 32 are multiples of 4) —
        //      single uint32 read, nibble-spread inline to {0,127}-bytes. nvcc at -O3 with
        //      pragma unroll may have CSE'd the previous pattern; the explicit form just
        //      makes the source clean (and removes a misleading comment).
        constexpr int B_INTS_PER_THREAD = SMEM_B_INTS / 256;  // = 8
#pragma unroll
        for (int i = 0; i < B_INTS_PER_THREAD; ++i) {
            int idx  = i * 256 + threadIdx.x;  // 0..2047, INTERLEAVED (Q1 fix)
            int r    = idx / BK_INT;                         // 0..63 (n-pos in tile)
            int cInt = idx % BK_INT;                         // 0..31 (K-int pos in tile)
            int n_global  = nblk * bN + r;
            // Opus fix (b): kc IS the q1-block index because bK == 128.
            const int blk_idx = n_global * nblocks_row + kc;

            int32_t packed_int = 0;
            if constexpr (WBITS == 1) {
                // Q1_0: 4 contiguous K-positions `cInt*4 .. cInt*4+3` live in ONE uint32.
                const int  bit_word = cInt >> 3;          // 0..3 within the q1-block
                const int  bit_off0 = (cInt * 4) & 31;    // multiple of 4 in 0..28
                const unsigned wb   = Wbits[blk_idx * 4 + bit_word];
                const unsigned nb4  = (wb >> bit_off0) & 0xFu;
                // Branchless bit -> {0,127}-byte spread inline (same no-LUT rule as mmq-hopper-q1).
                const unsigned spread = (nb4 & 1u) | ((nb4 & 2u) << 7) | ((nb4 & 4u) << 14) | ((nb4 & 8u) << 21);
                packed_int = (int32_t) (spread * 127u);
            } else {
                // Q2_0: 16 2-bit fields per uint32 → 16 K-positions per word; 8 words per block.
                // 4 contiguous K-positions `cInt*4 .. cInt*4+3` cross at most two words when
                // `cInt*4+3 > 15` (i.e. cInt >= 4 = K-pos >= 16). For cInt < 4, single read.
                // Otherwise read both adjacent words; the 4 K-positions span across them.
                const int      bit_word = (cInt * 4) >> 5;          // 0..1 within q2-block
                const int      bit_off  = (cInt * 4) & 31;          // 0,4,8,...28
                const unsigned wb_lo    = Wbits[blk_idx * 8 + bit_word + 0];
                const unsigned wb_hi    = Wbits[blk_idx * 8 + bit_word + 1];  // read is cheap even if OOB-bits
                const unsigned wbits    = ((cInt * 4) + 3 >= 32) ? wb_hi : wb_lo;
                const unsigned qbits    = (wbits >> bit_off) & 0xFFu;  // 8 K-positions, 4 of which we want
                const unsigned q0       = ((qbits >> 0) & 3u) - 1u;     // K-pos {0..3} -> q-1 in {-1,0,+1,+2}
                const unsigned q1       = ((qbits >> 2) & 3u) - 1u;
                const unsigned q2       = ((qbits >> 4) & 3u) - 1u;
                const unsigned q3       = ((qbits >> 6) & 3u) - 1u;
                packed_int = ((int32_t) q0)
                           | ((int32_t) q1) << 8
                           | ((int32_t) q2) << 16
                           | ((int32_t) q3) << 24;
            }
            sB_i[idx] = packed_int;
        }

        // ----- Stage C: load per-K-block float scales into sDa / sDw. -----
        constexpr int SCALE_WORDS_PER_THREAD = (bM + bN + 255) / 256;  // = 1
#pragma unroll
        for (int i = 0; i < SCALE_WORDS_PER_THREAD; ++i) {
            int t = threadIdx.x + i * 256;
            if (t < bM) {
                sDa[t] = dA[(size_t) (mblk * bM + t) * nblocks_row + kc];
            } else if (t < bM + bN) {
                int j = t - bM;
                sDw[j] = Wd[(size_t) (nblk * bN + j) * nblocks_row + kc];
            }
        }
        __syncthreads();

        // ----- Stage D + Stage E: per-kc chunk's int32 dot-product, scaled to fp32, accumulated -----
        // Opus minimum patch #3: frag-outer / kk-inner order so only 4 ints per thread are
        // live at any one time (vs 32 with the prior kk-outer / f-inner nest). Stage E (per-kc
        // float scaling + accumulate into acc_f) is folded inside the f-outer body.
        tile<16, 8, int, DATA_LAYOUT_I_MAJOR> Afrag;
        tile<8,  8, int, DATA_LAYOUT_I_MAJOR> Bfrag;
#pragma unroll
        for (int f = 0; f < FRAG_PER_WARP; ++f) {
            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;  // per-fragment int32 accumulators (in registers)
            const int frag_r = f >> 3;           // 0..1 (bM=256 → 2 rowblocks/warp)
            const int frag_c = f & 7;            // 0..7 (bN=64 → 8 colblocks)
            const int m_sub = warp_m_base + frag_r * 16;
            const int n_sub = warp_n_base + frag_c * 8;
#pragma unroll
            for (int kk = 0; kk < bK; kk += mma_k) {
                // ---- Load A (lane-based; shared across all 8 fragments in this warp) ----
                // sA_i is [bM][bK/4] ints, row m_sub, K-int col kk/4.
                load_A_lane(Afrag, sA_i + m_sub * BK_INT + (kk / 4), BK_INT);
                // ---- Load B (lane-based; per-fragment n-cell subset of sB_i) ----
                load_B_lane(Bfrag, sB_i + n_sub * BK_INT + (kk / 4), BK_INT);
                // In-out: prior int32 → mma → next int32 in d_e.
                Dfrag.x[0] = d0;
                Dfrag.x[1] = d1;
                Dfrag.x[2] = d2;
                Dfrag.x[3] = d3;
                mma(Dfrag, Afrag, Bfrag);
                d0 = Dfrag.x[0];
                d1 = Dfrag.x[1];
                d2 = Dfrag.x[2];
                d3 = Dfrag.x[3];
            }
            // Per-cell float scale + accumulate into acc_f. Canonical m16n8 C/D lane mapping:
            //   d0 -> (m = lane/4,       n = 2*(lane%4)    )
            //   d1 -> (m = lane/4,       n = 2*(lane%4) + 1)
            //   d2 -> (m = 8 + lane/4,   n = 2*(lane%4)    )
            //   d3 -> (m = 8 + lane/4,   n = 2*(lane%4) + 1)
            int mi0 = m_sub + (lane >> 2);
            int ni0 = n_sub + ((lane & 3) << 1);
            int mi1 = mi0 + 8;
            int ni1 = ni0 + 1;
            float s0 = sDa[mi0] * sDw[ni0];
            float s1 = sDa[mi0] * sDw[ni1];
            float s2 = sDa[mi1] * sDw[ni0];
            float s3 = sDa[mi1] * sDw[ni1];
            acc_f[f][0] += (float) d0 * s0;
            acc_f[f][1] += (float) d1 * s1;
            acc_f[f][2] += (float) d2 * s2;
            acc_f[f][3] += (float) d3 * s3;
        }
        __syncthreads();
    }

    // === Final: write fp32 cell-wise (same per-warp lane mapping as Stage E) ===
#pragma unroll
    for (int f = 0; f < FRAG_PER_WARP; ++f) {
        const int frag_r = f >> 3;
        const int frag_c = f & 7;
        const int m_sub = warp_m_base + frag_r * 16;
        const int n_sub = warp_n_base + frag_c * 8;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            int m_idx = m_sub + ((e & 2) ? 8 : 0) + (lane >> 2);
            int n_idx = n_sub + ((lane & 3) << 1) + (e & 1);
            C[(size_t) (mblk * bM + m_idx) * N + (nblk * bN + n_idx)] = acc_f[f][e];
        }
    }
}

}  // namespace blackwell_q1

// Public dispatcher.
bool ggml_cuda_mul_mat_q1_blackwell(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor *         src0,
                                    const ggml_tensor *         src1,
                                    ggml_tensor *               dst) {
#if defined(GGML_CUDA_CC_IS_NVIDIA)
    static const bool enabled = getenv("GGML_BLACKWELL_Q1") != nullptr;
    if (!enabled) {
        return false;
    }
    const int     cc = ggml_cuda_info().devices[ctx.device].cc;
    const int64_t K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    const bool    is_q1 = src0->type == GGML_TYPE_Q1_0;
    const bool    is_q2 = src0->type == GGML_TYPE_Q2_0;
    if (cc != GGML_CUDA_CC_DGX_SPARK ||
        (!is_q1 && !is_q2) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        src1->ne[2] * src1->ne[3] != 1 || src0->ne[2] * src0->ne[3] != 1 || (M % blackwell_q1::bM) ||
        (N % blackwell_q1::bN) || (K % blackwell_q1::bK) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }
    cudaStream_t            stream = ctx.stream();
    blackwell_q1::DenseW    wq  = blackwell_q1::get_dense_w(src0->data, (long) N, (long) K, is_q2 ? 2 : 1, ctx.device, stream);
    ggml_cuda_pool_alloc<int8_t> act_q(ctx.pool(), (size_t) M * K);
    ggml_cuda_pool_alloc<float>  act_d(ctx.pool(), (size_t) M * (K / 128));
    {
        const int ngroups = (int) (M * (K / 128));
        blackwell_q1::quant_act_per128<<<(ngroups + 7) / 8, 256, 0, stream>>>(
            (const float *) src1->data, act_q.get(), act_d.get(), (int) M, (int) K);
    }
    // Dynamic SMEM opt-in.
    static bool attr_set[GGML_CUDA_MAX_DEVICES] = { false };
    if (!attr_set[ctx.device]) {
        CUDA_CHECK(cudaFuncSetAttribute(blackwell_q1::lowbit_mma_ggml<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, blackwell_q1::SMEM_BYTES));
        CUDA_CHECK(cudaFuncSetAttribute(blackwell_q1::lowbit_mma_ggml<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, blackwell_q1::SMEM_BYTES));
        attr_set[ctx.device] = true;
    }
    auto * kern = is_q2 ? blackwell_q1::lowbit_mma_ggml<2> : blackwell_q1::lowbit_mma_ggml<1>;
    dim3 grid((unsigned) (M / blackwell_q1::bM), (unsigned) (N / blackwell_q1::bN));
    kern<<<grid, 256, blackwell_q1::SMEM_BYTES, stream>>>(
        act_q.get(), act_d.get(), wq.bits_only, wq.dw, (float *) dst->data, (int) M, (int) N, (int) K);
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
#endif
}

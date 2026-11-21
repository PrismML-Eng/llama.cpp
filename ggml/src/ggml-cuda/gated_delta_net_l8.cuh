#pragma once

#include "common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Adapted from ArthurOstapenko's Yukon CUDA GDN verification kernel:
// https://github.com/Layr-Labs/cudafast-bonsai2-9b-engine/tree/36c95362b23b83724503fef4e88e61d748f9355f
static constexpr int GDN_L8_ROWS        = 16;
static constexpr int GDN_L8V_MIN_TOKENS = 8;
static constexpr int GDN_L8V_MAX_TOKENS = 16;

// Preserve the original xor-16,8,4,2,1 reduction tree across four old lanes per new lane.
static __device__ __forceinline__ float gdn_l8_reduce(float (&p)[4]) {
    float x = __fadd_rn(__fadd_rn(p[0], p[2]), __fadd_rn(p[1], p[3]));
#    pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

template <bool RAW, bool G_PRECOMPUTED>
__global__ void __launch_bounds__(128, 6) gated_delta_net_l8v_cuda(const float *   q,
                                                                   const float *   k,
                                                                   const float *   v,
                                                                   const float *   g,
                                                                   const float *   beta,
                                                                   const float *   raw_dt_bias,
                                                                   const float *   raw_a,
                                                                   const float *   curr_state,
                                                                   float *         dst,
                                                                   float *         state,
                                                                   int64_t         H,
                                                                   int64_t         n_tokens,
                                                                   int64_t         n_seqs,
                                                                   int64_t         sq1,
                                                                   int64_t         sq2,
                                                                   int64_t         sq3,
                                                                   int64_t         sv1,
                                                                   int64_t         sv2,
                                                                   int64_t         sv3,
                                                                   int64_t         sb1,
                                                                   int64_t         sb2,
                                                                   int64_t         sb3,
                                                                   const uint3     neqk1_magic,
                                                                   const uint3     rq3_magic,
                                                                   float           scale,
                                                                   int64_t         state_slot_stride,
                                                                   int             K,
                                                                   const int32_t * s_ids,
                                                                   int64_t         s_row_stride) {
#    if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    constexpr int  S_v      = 128;
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      grp      = lane / 8;
    const int      n        = lane % 8;
    const int      col      = (blockIdx.z * blockDim.y + threadIdx.y) * 4;  // first column of this warp
    const int      my_col   = col + grp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst;

    ggml_cuda_pdl_sync();

    const int64_t state_in_offset =
        (s_ids ? (int64_t) s_ids[sequence] * s_row_stride : sequence * H * S_v * S_v) + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    // Stage k in [r][n][m] order so each lane can load four old lanes with one float4.
    constexpr int TOK_SM = S_v + 16 + 4;
    __shared__ __align__(16) float stage[GDN_L8V_MAX_TOKENS][TOK_SM];
    {
        const int     tid     = threadIdx.y * 32 + lane;
        const float * k_base  = k + iq3 * sq3 + iq1 * sq1;
        const float * v_base  = v + sequence * sv3 + h_idx * sv1 + blockIdx.z * 16;
        const int64_t gb_base = sequence * sb3 + h_idx * sb1;
        for (int e = tid; e < (int) n_tokens * TOK_SM; e += 128) {
            const int     t = e / TOK_SM;
            const int     i = e % TOK_SM;
            const float * src;
            int           dst_i = i;
            if (i < S_v) {
                const int r = i / 32, n_ = i % 8, m = (i % 32) / 8;
                src   = k_base + t * sq2 + i;
                dst_i = (r * 8 + n_) * 4 + m;
            } else if (i < S_v + 16) {
                src = v_base + t * sv2 + (i - S_v);
            } else if (i < S_v + 18) {
                src = (i == S_v + 16 ? g : beta) + gb_base + t * sb2;
            } else {
                continue;  // padding
            }
            const unsigned int dst = (unsigned int) __cvta_generic_to_shared(&stage[t][dst_i]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src));
        }
        asm volatile("cp.async.commit_group;");
    }

    float s[GDN_L8_ROWS];  // s[m*4 + r] = S[r*32 + n + 8m][my_col]

    {
#        pragma unroll
        for (int m = 0; m < 4; ++m) {
#        pragma unroll
            for (int r = 0; r < 4; ++r) {
                s[m * 4 + r] = curr_state[my_col * S_v + r * 32 + n + 8 * m];
            }
        }
    }
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    float g_lane = 0.0f, beta_lane = 0.0f;
    if (lane < n_tokens) {
        beta_lane = stage[lane][S_v + 17];
        if constexpr (RAW) {
            beta_lane = 1.0f / (1.0f + expf(-beta_lane));
        }
        float g0 = stage[lane][S_v + 16];
        if constexpr (RAW) {
            const float x = g0 + raw_dt_bias[h_idx];
            g0            = raw_a[h_idx] * ((x > 20.0f) ? x : logf(1.0f + expf(x)));
        }
        g_lane = G_PRECOMPUTED ? g0 : expf(g0);
    }
    const int tid = threadIdx.y * 32 + lane;

    __shared__ float attn_sm[GDN_L8V_MAX_TOKENS][16];  // this block's 16 columns of the attention output
#        pragma unroll
    for (int t = 0; t < GDN_L8V_MAX_TOKENS; t++) {
        if (t >= n_tokens) {
            break;
        }
        const float * q_t    = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * tok_sm = stage[t];

        const float beta_val = __shfl_sync(0xffffffff, beta_lane, t, 32);
        const float g_val    = __shfl_sync(0xffffffff, g_lane, t, 32);
        float       kr[GDN_L8_ROWS];
        float       qr[GDN_L8_ROWS];
#        pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float4 k4 = *(const float4 *) &tok_sm[(r * 8 + n) * 4];
            kr[0 * 4 + r]   = k4.x;
            kr[1 * 4 + r]   = k4.y;
            kr[2 * 4 + r]   = k4.z;
            kr[3 * 4 + r]   = k4.w;
        }
#        pragma unroll
        for (int m = 0; m < 4; ++m) {
#        pragma unroll
            for (int r = 0; r < 4; ++r) {
                qr[m * 4 + r] = q_t[r * 32 + n + 8 * m];
            }
        }
        const float v_col = tok_sm[S_v + threadIdx.y * 4 + grp];

        float p[4];
#        pragma unroll
        for (int m = 0; m < 4; ++m) {
            float kv_shard = 0.0f;
#        pragma unroll
            for (int r = 0; r < 4; ++r) {
                kv_shard += s[m * 4 + r] * kr[m * 4 + r];
            }
            p[m] = kv_shard;
        }
        const float kv_col    = gdn_l8_reduce(p);
        const float delta_col = (v_col - g_val * kv_col) * beta_val;

#        pragma unroll
        for (int m = 0; m < 4; ++m) {
            float attn_partial = 0.0f;
#        pragma unroll
            for (int r = 0; r < 4; ++r) {
                s[m * 4 + r] = g_val * s[m * 4 + r] + kr[m * 4 + r] * delta_col;
                attn_partial += s[m * 4 + r] * qr[m * 4 + r];
            }
            p[m] = attn_partial;
        }
        const float attn_col = gdn_l8_reduce(p);
        if (n == 0) {
            attn_sm[t][threadIdx.y * 4 + grp] = attn_col * scale;
        }

        {
            // eager snapshots (slot 0 = most recent state, slot s = s tokens back)
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * snap = state + target_slot * state_slot_stride;
#        pragma unroll
                for (int m = 0; m < 4; ++m) {
#        pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        snap[my_col * S_v + r * 32 + n + 8 * m] = s[m * 4 + r];
                    }
                }
            }
        }
    }
    __syncthreads();
    for (int e = tid; e < (int) n_tokens * 16; e += 128) {
        attn_data[(e / 16) * S_v * H + blockIdx.z * 16 + e % 16] = attn_sm[e / 16][e % 16];
    }

#    else
    NO_DEVICE_CODE;
#    endif
}

#endif

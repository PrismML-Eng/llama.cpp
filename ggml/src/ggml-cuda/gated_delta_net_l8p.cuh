#pragma once

// GGML_CUDA_GDN_L8P: bit-exact chunk-staged recurrent Gated DeltaNet for prefill (S_v = 128, scalar gate, NVIDIA Ampere+).
//
// Same arithmetic as gated_delta_net_cuda<128, false, *, *, false> (cols_per_warp = 4): per column, each of the 32 "old
// lanes" L sums its four rows r*32+L in the same order, and the xor-16,8,4,2,1 butterfly is kept. As in the L8 verify
// kernel (gated_delta_net_l8.cuh), one lane holds four old lanes (L = n + 8m), so the 16- and 8-levels run in-thread
// with __fadd_rn in tree order and only 4,2,1 use shuffles; 8 lanes serve a column. CPL columns per lane (1 or 2) reuse
// each k/q register across columns. What differs from stock is only scheduling: tokens are staged into shared memory
// T at a time with cp.async (double buffered; k and q permuted so each lane reads float4s), gates are activated once
// per token by one lane (same expressions), and attention outputs are written per chunk, coalesced.

#include "common.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static constexpr int GDN_L8P_MIN_TOKENS = 17;  // below: stock / L8 verify kernel

template <int T, int NW, int CPL> struct gdn_l8p_smem {
    static constexpr int NCOL   = NW * 4 * CPL;          // columns per block
    static constexpr int KQ     = 2 * 128;               // k then q, both permuted [r][n][m]
    static constexpr int TOK    = KQ + NCOL + 4;         // + v columns + raw g, raw beta, pad
    static constexpr int BUF    = T * TOK;
    static constexpr int FLOATS = 2 * BUF + T * NCOL;    // double buffer + attention out
    static constexpr size_t BYTES = FLOATS * sizeof(float);
};

// the in-thread part of the tree (xor-16 and xor-8 levels of the original butterfly)
static __device__ __forceinline__ float gdn_l8p_tree(const float * p) {
    return __fadd_rn(__fadd_rn(p[0], p[2]), __fadd_rn(p[1], p[3]));
}

static __device__ __forceinline__ float gdn_l8p_reduce(const float * p, const bool sab) {
    float x = sab ? __fadd_rn(__fadd_rn(p[0], p[1]), __fadd_rn(p[2], p[3]))   // sabotage: wrong tree order
                  : __fadd_rn(__fadd_rn(p[0], p[2]), __fadd_rn(p[1], p[3]));
#    pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

template <int T, int NW, int CPL, int U, bool keep_rs_t, bool RAW>
__global__ void __launch_bounds__(32 * NW, 1) gated_delta_net_l8p_cuda(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * raw_dt_bias, const float * raw_a, const float * curr_state, float * dst, float * state,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        const uint3 neqk1_magic, const uint3 rq3_magic, float scale, int64_t state_slot_stride, int K,
        const int32_t * s_ids, int64_t s_row_stride, const int sab) {
#    if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    using SM = gdn_l8p_smem<T, NW, CPL>;
    constexpr int S_v  = 128;
    constexpr int NCOL = SM::NCOL;
    static_assert(T <= 32, "one lane per token holds the gates");

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      grp      = lane / 8;
    const int      n        = lane % 8;
    const int      tid      = threadIdx.y * 32 + lane;
    const int      col0     = blockIdx.z * NCOL;                       // first column of this block
    int            lcol[CPL];                                          // columns within the block
#    pragma unroll
    for (int j = 0; j < CPL; ++j) {
        lcol[j] = threadIdx.y * 4 * CPL + j * 4 + grp;
    }

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    extern __shared__ __align__(16) float l8p_sm[];
    float * attn_sm = l8p_sm + 2 * SM::BUF;

    ggml_cuda_pdl_sync();

    const int64_t state_in_offset =
        (s_ids ? (int64_t) s_ids[sequence] * s_row_stride : sequence * H * S_v * S_v) + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state      += state_out_offset + (int64_t) col0 * S_v;
    curr_state += state_in_offset + (int64_t) col0 * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    const float * k_base  = k + iq3 * sq3 + iq1 * sq1;
    const float * q_base  = q + iq3 * sq3 + iq1 * sq1;
    const float * v_base  = v + sequence * sv3 + h_idx * sv1 + col0;
    const int64_t gb_base = sequence * sb3 + h_idx * sb1;

    const int n_chunks = (int) ((n_tokens + T - 1) / T);

    // stage chunk ch: per token k and q permuted to [r][n][m], then v columns, raw g, raw beta
    // stage chunk ch: per token k and q permuted to [r][n][m], then v columns, raw g, raw beta.
    // Each thread keeps fixed elements (source offset and permuted destination computed once).
    auto issue = [&](const int ch) {
        float *   buf = l8p_sm + (ch & 1) * SM::BUF;
        const int t0  = ch * T;
        const int cnt = (int) min((int64_t) T, n_tokens - t0);
#    pragma unroll
        for (int ii = tid; ii < S_v; ii += 32 * NW) {
            const int     pi = ((ii / 32) * 8 + ii % 8) * 4 + (ii % 32) / 8;
            const float * ks = k_base + (int64_t) t0 * sq2 + ii;
            const float * qs = q_base + (int64_t) t0 * sq2 + ii;
            unsigned int  d  = (unsigned int) __cvta_generic_to_shared(&buf[pi]);
            for (int t = 0; t < cnt; ++t) {
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(d), "l"(ks));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(d + 4 * S_v), "l"(qs));
                ks += sq2; qs += sq2; d += 4 * SM::TOK;
            }
        }
        for (int i = tid; i < NCOL + 2; i += 32 * NW) {
            const float * src = i < NCOL ? v_base + (int64_t) t0 * sv2 + i : (i == NCOL ? g : beta) + gb_base + (int64_t) t0 * sb2;
            const int64_t st  = i < NCOL ? sv2 : sb2;
            unsigned int  d   = (unsigned int) __cvta_generic_to_shared(&buf[2 * S_v + i]);
            for (int t = 0; t < cnt; ++t) {
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(d), "l"(src));
                src += st; d += 4 * SM::TOK;
            }
        }
        asm volatile("cp.async.commit_group;");
    };

    issue(0);

    float s[CPL][16];  // s[j][m*4 + r] = S[r*32 + n + 8m][col0 + lcol[j]]
#    pragma unroll
    for (int j = 0; j < CPL; ++j) {
#    pragma unroll
        for (int m = 0; m < 4; ++m) {
#    pragma unroll
            for (int r = 0; r < 4; ++r) {
                s[j][m * 4 + r] = curr_state[lcol[j] * S_v + r * 32 + n + 8 * m];
            }
        }
    }

    for (int ch = 0; ch < n_chunks; ++ch) {
        if (ch + 1 < n_chunks && !(sab & 2)) {   // sab bit 1: timing probe, no staging after chunk 0 (wrong output)
            issue(ch + 1);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        const float * buf = l8p_sm + (ch & 1) * SM::BUF;
        const int     t0  = ch * T;
        const int     cnt = (int) min((int64_t) T, n_tokens - t0);

        float g_lane = 0.0f, beta_lane = 0.0f;
        if (lane < cnt) {
            beta_lane = buf[lane * SM::TOK + 2 * S_v + NCOL + 1];
            if constexpr (RAW) {
                beta_lane = 1.0f / (1.0f + expf(-beta_lane));
            }
            float g0 = buf[lane * SM::TOK + 2 * S_v + NCOL];
            if constexpr (RAW) {
                const float x = g0 + raw_dt_bias[h_idx];
                g0            = raw_a[h_idx] * ((x > 20.0f) ? x : logf(1.0f + expf(x)));
            }
            g_lane = expf(g0);
        }

        const float * buf0 = (sab & 2) ? l8p_sm : buf;
        // U tokens per group: the attention reductions (not on the recurrence chain) are batched after each group
        const int ncomp = (sab & 4) ? 0 : cnt;   // sab bit 2: timing probe, no compute (wrong output)
        for (int tb = 0; tb < ncomp; tb += U) {
          float ax[U][CPL];
#    pragma unroll
          for (int u = 0; u < U; ++u) {
            const int t = tb + u;
#    pragma unroll
            for (int j = 0; j < CPL; ++j) {
                ax[u][j] = 0.0f;
            }
            if (t >= ncomp) {
                continue;
            }
            const float * tok = buf0 + t * SM::TOK;

            const float beta_val = __shfl_sync(0xffffffff, beta_lane, t, 32);
            const float g_val    = __shfl_sync(0xffffffff, g_lane, t, 32);
            float       kr[16];
            float       qr[16];
#    pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float4 k4 = *(const float4 *) &tok[(r * 8 + n) * 4];
                kr[0 * 4 + r]   = k4.x;
                kr[1 * 4 + r]   = k4.y;
                kr[2 * 4 + r]   = k4.z;
                kr[3 * 4 + r]   = k4.w;
                const float4 q4 = *(const float4 *) &tok[S_v + (r * 8 + n) * 4];
                qr[0 * 4 + r]   = q4.x;
                qr[1 * 4 + r]   = q4.y;
                qr[2 * 4 + r]   = q4.z;
                qr[3 * 4 + r]   = q4.w;
            }

            float p[CPL][4];
#    pragma unroll
            for (int j = 0; j < CPL; ++j) {
#    pragma unroll
                for (int m = 0; m < 4; ++m) {
                    float kv_shard = 0.0f;
#    pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        kv_shard += s[j][m * 4 + r] * kr[m * 4 + r];
                    }
                    p[j][m] = kv_shard;
                }
            }
            float delta_col[CPL];
#    pragma unroll
            for (int j = 0; j < CPL; ++j) {
                const float kv_col = gdn_l8p_reduce(p[j], sab & 1);
                const float v_col  = tok[2 * S_v + lcol[j]];
                delta_col[j]       = (v_col - g_val * kv_col) * beta_val;
            }

#    pragma unroll
            for (int j = 0; j < CPL; ++j) {
#    pragma unroll
                for (int m = 0; m < 4; ++m) {
                    float attn_partial = 0.0f;
#    pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        s[j][m * 4 + r] = g_val * s[j][m * 4 + r] + kr[m * 4 + r] * delta_col[j];
                        attn_partial += s[j][m * 4 + r] * qr[m * 4 + r];
                    }
                    p[j][m] = attn_partial;
                }
            }
#    pragma unroll
            for (int j = 0; j < CPL; ++j) {
                ax[u][j] = gdn_l8p_tree(p[j]);
            }

            if constexpr (keep_rs_t) {
                // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back
                const int target_slot = (int) n_tokens - 1 - (t0 + t);
                if (target_slot >= 0 && target_slot < K) {
                    float * snap = state + target_slot * state_slot_stride;
#    pragma unroll
                    for (int j = 0; j < CPL; ++j) {
#    pragma unroll
                        for (int m = 0; m < 4; ++m) {
#    pragma unroll
                            for (int r = 0; r < 4; ++r) {
                                snap[lcol[j] * S_v + r * 32 + n + 8 * m] = s[j][m * 4 + r];
                            }
                        }
                    }
                }
            }
          }
#    pragma unroll
          for (int offset = 4; offset > 0; offset >>= 1) {
#    pragma unroll
            for (int u = 0; u < U; ++u) {
#    pragma unroll
                for (int j = 0; j < CPL; ++j) {
                    ax[u][j] += __shfl_xor_sync(0xffffffff, ax[u][j], offset, 32);
                }
            }
          }
#    pragma unroll
          for (int u = 0; u < U; ++u) {
            if (tb + u < ncomp && n == 0) {
#    pragma unroll
                for (int j = 0; j < CPL; ++j) {
                    attn_sm[(tb + u) * NCOL + lcol[j]] = ax[u][j] * scale;
                }
            }
          }
        }
        __syncthreads();
        for (int e = tid; e < cnt * NCOL; e += 32 * NW) {
            attn_data[(int64_t) (t0 + e / NCOL) * S_v * H + col0 + e % NCOL] = attn_sm[e];
        }
    }

    if constexpr (!keep_rs_t) {
#    pragma unroll
        for (int j = 0; j < CPL; ++j) {
#    pragma unroll
            for (int m = 0; m < 4; ++m) {
#    pragma unroll
                for (int r = 0; r < 4; ++r) {
                    state[lcol[j] * S_v + r * 32 + n + 8 * m] = s[j][m * 4 + r];
                }
            }
        }
    }
#    else
    NO_DEVICE_CODE;
#    endif
}

#endif

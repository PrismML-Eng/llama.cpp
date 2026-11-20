#include "gated_delta_net.cuh"
#include "gated_delta_net_l8.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdint>
#include <cstdlib>

// Host and device must agree on columns/warp. Kernel uses __CUDA_ARCH__; host uses ggml_cuda_highest_compiled_arch(cc). NVIDIA Ampere+ and S_v==128 and !KDA -> 4; HIP/MUSA stay 1.
static constexpr __host__ __device__ int gdn_cols_per_warp(int arch, int S_v, bool KDA) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    (void) arch; (void) S_v; (void) KDA;
    return 1;
#else
    return GGML_CUDA_CC_IS_NVIDIA(arch) && arch >= GGML_CUDA_CC_AMPERE && S_v == 128 && !KDA ? 4 : 1;
#endif
}

static __global__ void gdn_precompute_exp(const float * g, float * g_exp, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n;
         i += (int64_t) blockDim.x*gridDim.x) {
        g_exp[i] = expf(g[i]);
    }
}

// fused state gather -> contiguous s0 [S_v, S_v, H, n_seqs] for the chunked MMA kernel (per sequence one cache row)
static __global__ void gdn_gather_state(const float * base, const int32_t * ids, const int64_t row_stride,
        float * out, const int64_t n_per_seq, const int64_t n_seqs) {
    const int64_t n = n_per_seq * n_seqs;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x*gridDim.x) {
        const int64_t seq = i / n_per_seq;
        out[i] = base[(int64_t) ids[seq]*row_stride + (i - seq*n_per_seq)];
    }
}

// raw gates -> activated gates for the chunked MMA kernel: beta = sigmoid(beta), g = raw_a[h] * softplus(g + dt_bias[h]),
// the same formulas as the RAW path of gated_delta_net_cuda. g and beta are contiguous [1, H, n_tokens, n_seqs].
static __global__ void gdn_activate_raw_gates(const float * g, const float * beta, const float * raw_dt_bias,
        const float * raw_a, float * g_out, float * beta_out, const int64_t n, const int64_t H) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x*gridDim.x) {
        const int64_t h = i % H;
        const float x = g[i] + raw_dt_bias[h];
        g_out[i]    = raw_a[h] * ((x > 20.0f) ? x : logf(1.0f + expf(x)));
        beta_out[i] = 1.0f / (1.0f + expf(-beta[i]));
    }
}

// RAW: beta and g arrive pre-activation (ggml_gated_delta_net_set_raw_gates); the kernel applies
// sigmoid(beta) and raw_a[h] * softplus(g + raw_dt_bias[h]) with the unary kernels' formulas.
// G_PRECOMPUTED: g already holds exp(g) (GB10 long-prompt path); only used with RAW == false.
template <int S_v, bool KDA, bool keep_rs_t, bool RAW, bool G_PRECOMPUTED>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * raw_dt_bias,
                                     const float * raw_a,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * s_ids,
                                     int64_t       s_row_stride) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // Each warp owns one or more columns, using warp-level primitives to reduce across rows.
    const int      lane     = threadIdx.x;
#if defined(__CUDA_ARCH__)
    constexpr int cols_per_warp = gdn_cols_per_warp(__CUDA_ARCH__, S_v, KDA);
#else
    constexpr int cols_per_warp = 1; // host pass only; never executed
#endif
    const int      col      = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // Wait before any s_ids / state load. inp_s_copy is a host upload today, so this is a no-op
    // on that path; ggml_cuda_try_gdn_gather_skip accepts any I32 ids tensor.
    ggml_cuda_pdl_sync();

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    // fused gather: read this sequence's live state straight out of the cache row s_ids[sequence]
    const int64_t state_in_offset      = (s_ids ? (int64_t) s_ids[sequence] * s_row_stride : sequence * H * S_v * S_v)
                                         + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[cols_per_warp][rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

#pragma unroll
    for (int c = 0; c < cols_per_warp; ++c) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[c][r] = curr_state[(col + c) * S_v + i];
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        float beta_val = *beta_t;
        if constexpr (RAW) {
            beta_val = 1.0f / (1.0f + expf(-beta_val));
        }

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            static_assert(!(RAW && G_PRECOMPUTED), "exp(g) precompute is only defined for activated gates");
            float g0 = *g_t;
            if constexpr (RAW) {
                const float x = g0 + raw_dt_bias[h_idx];
                g0 = raw_a[h_idx] * ((x > 20.0f) ? x : logf(1.0f + expf(x)));
            }
            const float g_val = G_PRECOMPUTED ? g0 : expf(g0);

            // Each warp owns one or more columns and reuses the common q/k registers.
#pragma unroll
            for (int c = 0; c < cols_per_warp; ++c) {
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[c][r] * k_reg[r];
                }
                float kv_col = warp_reduce_sum<warp_size>(kv_shard);

                float delta_col = (v_t[col + c] - g_val * kv_col) * beta_val;

                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r]  = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                    attn_partial += s_shard[c][r] * q_reg[r];
                }

                float attn_col = warp_reduce_sum<warp_size>(attn_partial);

                if (lane == 0) {
                    attn_data[col + c] = attn_col * scale;
                }
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[0][r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[0][r]  = expf(g_t[i]) * s_shard[0][r] + k_reg[r] * delta_col;
                attn_partial += s_shard[0][r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < cols_per_warp; ++c) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        curr_state[(col + c) * S_v + i] = s_shard[c][r];
                    }
                }
            }
        }

    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < cols_per_warp; ++c) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                state[(col + c) * S_v + i] = s_shard[c][r];
            }
        }
    }
}

template <bool KDA, bool keep_rs_t, bool RAW, bool G_PRECOMPUTED>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * rb_d, const float * ra_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K,
        const int32_t * s_ids, int64_t s_row_stride, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int num_warps = 4;
    // Same predicate as the kernel, fed the arch compiled for this cc.
    const int cols_per_warp = GGML_CUDA_CC_IS_NVIDIA(cc) ? gdn_cols_per_warp(ggml_cuda_highest_compiled_arch(cc), S_v, KDA) : 1;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps * cols_per_warp - 1) / (num_warps * cols_per_warp));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride);
            break;
        }
        case 128: {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // Opt in until end-to-end measurements establish defaults for each device family.
            static const bool l8_enabled =
                getenv("GGML_CUDA_GDN_L8") != nullptr && atoi(getenv("GGML_CUDA_GDN_L8")) != 0;
            if constexpr (keep_rs_t && !KDA) {
                if (l8_enabled && GGML_CUDA_CC_IS_NVIDIA(cc) &&
                    ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE && warp_size == 32 &&
                    cols_per_warp == 4 && n_tokens >= GDN_L8V_MIN_TOKENS && n_tokens <= GDN_L8V_MAX_TOKENS) {
                    ggml_cuda_kernel_launch(gated_delta_net_l8v_cuda<RAW, G_PRECOMPUTED>, launch_params, q_d, k_d,
                                            v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
                                            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic,
                                            scale, state_slot_stride, K, s_ids, s_row_stride);
                    break;
                }
            }
#endif
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, RAW, G_PRECOMPUTED>, launch_params,
                q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, s_ids, s_row_stride);
            break;
        }
        default:
            GGML_ABORT("fatal error");
    }
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) == 1 && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    // fused state gather, registered for this node by this context's graph evaluator (ggml_cuda_try_gdn_gather_skip)
    const int32_t * s_ids        = nullptr;
    int64_t         s_row_stride = 0;
    if (const ggml_cuda_gated_delta_net_gather * gather = ctx.gdn_gathers().find(dst)) {
        s_d          = gather->base;
        s_ids        = gather->ids;
        s_row_stride = gather->row_stride;
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    const bool    raw  = ggml_get_op_params_i32(dst, 1) != 0;
    const float * rb_d = raw ? (const float *) dst->src[7]->data : nullptr;
    const float * ra_d = raw ? (const float *) dst->src[8]->data : nullptr;
    GGML_ASSERT(!(raw && kda)); // raw gates are defined for the scalar gate only

    // chunked MMA prefill kernel (upstream ggml-org/llama.cpp#29353): scalar gates, state read in place (no fused
    // gather). Raw gates are activated into temporaries first, with the formulas of the RAW path below, so the
    // product graph (raw gates on CUDA) can take it. With K > 1 rollback snapshots (speculative decoding), one
    // sequence: the MMA kernel runs tokens [0, n - K) into a temporary state and the recurrent kernel continues
    // over the last K tokens from it, writing the K snapshot slots exactly as before.
    // Opt in (GGML_CUDA_GDN_MMA=1) until end-to-end measurements establish defaults for each device family.
    static const bool mma_off = getenv("GGML_CUDA_GDN_MMA") == nullptr || atoi(getenv("GGML_CUDA_GDN_MMA")) == 0;
    const int64_t n_mma = K == 1 ? n_tokens : (n_seqs == 1 ? n_tokens - K : 0);
    if (!mma_off && n_tokens >= 64 && (kda || n_mma < 64)) {
        static bool logged_skip = false;
        if (!logged_skip) {
            GGML_LOG_WARN("%s: chunked GDN prefill (MMA) skipped: %lld tokens, K %d, kda %d, seqs %lld\n",
                __func__, (long long) n_tokens, K, (int) kda, (long long) n_seqs);
            logged_skip = true;
        }
    }
    const bool mma_dev = !mma_off && !kda && n_mma >= 64 && ggml_cuda_gdn_mma_device_supported(ctx.device);
    if (!mma_off && !kda && n_mma >= 64 && !mma_dev) {
        static bool logged_dev = false;
        if (!logged_dev) {
            GGML_LOG_WARN("%s: chunked GDN prefill (MMA) skipped: device cc %d has no supported BF16 MMA schedule; recurrent kernel\n",
                __func__, ggml_cuda_info().devices[ctx.device].cc);
            logged_dev = true;
        }
    }
    if (mma_dev) {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.eligible = src_g->ne[0] == 1 && S_v == 128 && neq0 == 128;   // K is handled here
        args.n_tokens = n_mma;
        ggml_cuda_pool_alloc<float> s_in(ctx.pool());
        ggml_cuda_pool_alloc<float> g_act(ctx.pool());
        ggml_cuda_pool_alloc<float> b_act(ctx.pool());
        ggml_cuda_pool_alloc<float> s_mid(ctx.pool());
        if (s_ids != nullptr) {
            // fused gather: copy each sequence's live state out of its cache row first (S_v*S_v*H floats per sequence)
            const int64_t n_per_seq = S_v * S_v * H;
            s_in.alloc(n_per_seq * n_seqs);
            gdn_gather_state<<<(unsigned) std::min<int64_t>((n_per_seq*n_seqs + 255)/256, 4096), 256, 0, stream>>>(
                s_d, s_ids, s_row_stride, s_in.ptr, n_per_seq, n_seqs);
            args.state = s_in.ptr;
        }
        if (raw) {
            const int64_t n_gb = ggml_nelements(src_g);
            GGML_ASSERT(ggml_nelements(src_beta) == n_gb);
            g_act.alloc(n_gb);
            b_act.alloc(n_gb);
            gdn_activate_raw_gates<<<(unsigned) std::min<int64_t>((n_gb + 255)/256, 4096), 256, 0, stream>>>(
                g_d, b_d, rb_d, ra_d, g_act.ptr, b_act.ptr, n_gb, H);
            args.g    = g_act.ptr;
            args.beta = b_act.ptr;
        }
        if (K > 1) {
            s_mid.alloc(S_v * S_v * H * n_seqs);
            args.state_out = s_mid.ptr;
        } else {
            args.state_out = state_d;
        }
        const bool launched = ggml_cuda_gdn_mma_launch(ctx.device, args, stream);
        static bool logged = false;
        if (!logged) {
            GGML_LOG_WARN("%s: chunked GDN prefill (MMA) %s: %lld tokens (%lld by MMA), K %d, H %lld, H_k %lld, raw %d, gather %d, seqs %lld\n",
                __func__, launched ? "engaged" : "not taken (shape or device)", (long long) n_tokens, (long long) n_mma, K,
                (long long) H, (long long) neqk1, (int) raw, (int) (s_ids != nullptr), (long long) n_seqs);
            logged = true;
        }
        if (launched) {
            if (K > 1) {
                // the last K tokens from the MMA state, with the snapshot slots (raw gates still raw here)
                const int64_t t0 = n_mma;
                const float * q_t = q_d + t0 * sq2;
                const float * k_t = k_d + t0 * sq2;
                const float * v_t = v_d + t0 * sv2;
                const float * g_t = g_d + t0 * sb2;
                const float * b_t = b_d + t0 * sb2;
                float       * o_t = dst_d + t0 * S_v * H;
                if (raw) {
                    launch_gated_delta_net<false, true, true, false>(q_t, k_t, v_t, g_t, b_t, rb_d, ra_d, s_mid.ptr, o_t, state_d,
                        S_v, H, K, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                        nullptr, 0, stream);
                } else {
                    launch_gated_delta_net<false, true, false, false>(q_t, k_t, v_t, g_t, b_t, rb_d, ra_d, s_mid.ptr, o_t, state_d,
                        S_v, H, K, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K,
                        nullptr, 0, stream);
                }
            }
            return;
        }
    }

    // GB10 long-prompt path: exp(g) once per (token, head) instead of once per column-warp.
    // Only for activated gates; with raw gates (#165) the activation happens inside the kernel.
    ggml_cuda_pool_alloc<float> g_exp_alloc(ctx.pool());
    bool g_precomputed = false;
    if (!kda && !raw && S_v == 128 && n_tokens >= 32 &&
            ggml_cuda_info().devices[ggml_cuda_get_device()].cc == GGML_CUDA_CC_DGX_SPARK) {
        const int64_t n_g = ggml_nelements(src_g);
        g_exp_alloc.alloc(n_g);
        const int block = 256;
        const int grid = std::min<int64_t>((n_g + block - 1)/block, 4096);
        gdn_precompute_exp<<<grid, block, 0, stream>>>(g_d, g_exp_alloc.ptr, n_g);
        g_d = g_exp_alloc.ptr;
        g_precomputed = true;
    }

#define GDN_LAUNCH(KDA_, KEEP_, RAW_, PRE_)                                                       \
    launch_gated_delta_net<KDA_, KEEP_, RAW_, PRE_>(q_d, k_d, v_d, g_d, b_d, rb_d, ra_d, s_d, dst_d, state_d, \
        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                                    \
        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, s_ids, s_row_stride, stream)

    if (kda) {
        if (keep_rs) { GDN_LAUNCH(true,  true,  false, false); } else { GDN_LAUNCH(true,  false, false, false); }
    } else if (raw) {
        if (keep_rs) { GDN_LAUNCH(false, true,  true,  false); } else { GDN_LAUNCH(false, false, true,  false); }
    } else {
        if (g_precomputed) {
            if (keep_rs) { GDN_LAUNCH(false, true,  false, true);  } else { GDN_LAUNCH(false, false, false, true);  }
        } else {
            if (keep_rs) { GDN_LAUNCH(false, true,  false, false); } else { GDN_LAUNCH(false, false, false, false); }
        }
    }
#undef GDN_LAUNCH
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

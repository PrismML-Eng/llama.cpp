// ARC-LAB decode attention for a q4_0 KV cache (generation and MTP verify batches of 1-4 query tokens).
//
// The TILE path first converts the whole K and V cache to f16 on every call and then reads that copy
// (Bonsai 27B at 16K context: 5.9 + 11.4 ms per generated token over 16 attention layers). Here each
// work-group owns one KV head and a slice of the context, reads the q4_0 blocks directly, and serves all
// G query heads of that KV head (GQA) and all NQ query tokens from the one read (flash-decoding):
//   - the scaled query rows (G * NQ of them) sit in SLM;
//   - per tile of TK keys: thread (c, key) computes the G scores of query token c against its key,
//     straight from the q4_0 nibbles; per-row online softmax with a finite running max (a fully masked
//     tile or slice gets zero weight, never NaN); thread d accumulates P.V for head dim d of every row;
//   - slices are merged by flash_attn_combine_results (same partial layout as the vec kernel).

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-dec.hpp"
#include "ptq1-t2.hpp"
#include <cfloat>

namespace {
constexpr int   DEC_D    = 256;       // head size (K and V)
constexpr int   DEC_WG   = 256;       // work-group size = DEC_D: thread d owns output dim d
constexpr float DEC_MINF = -1.0e30f;  // finite "minus infinity" for the running max
}

// v3 kernel, kept for 4-token batches: the pipelined kernel below is slower there (436 vs 508 us per layer at
// 16K; see README) and faster at 1-2 tokens (16K 273 -> 256, 48K 944 -> 742 us).
template <int G, int NQ>
static void fattn_dec_q4_0_v3(const char * Q, const char * K, const char * V, const char * mask, float * dst,
                           float * parts, sycl::float2 * meta, float scale, int ne01, int ne02, int ne11,
                           int nkvh, int ne03, int64_t nb01, int64_t nb02, int64_t nb03, int64_t nb11,
                           int64_t nb12, int64_t nb13, int64_t nb21, int64_t nb22, int64_t nb23, int64_t nb31,
                           int64_t nb33, int ne33, int nsplit, int chunk, dpct::queue_ptr stream) {
    constexpr int R   = G * NQ;             // query rows per KV head
    constexpr int RP  = (R + 3) / 4 * 4;    // padded row stride of the transposed scores
    constexpr int TK  = DEC_WG / NQ;        // keys per tile: thread (c, jj) = (tid / TK, tid % TK)
    constexpr int NB  = DEC_D / QK4_0;
    constexpr int KW  = DEC_D / QK4_0 * sizeof(block_q4_0) / 4;  // 36 dwords per q4_0 K row (D 256)
    constexpr int KWP = KW + 1;                                  // padded SLM row stride (bank conflicts)
    static_assert(TK % 16 == 0 && (DEC_D / QK4_0 * sizeof(block_q4_0)) % 4 == 0, "shape");

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::float4, 1> sQ(sycl::range<1>(R * DEC_D / 4), cgh);
        sycl::local_accessor<uint32_t, 1>     sK(sycl::range<1>(TK * KWP), cgh);
        sycl::local_accessor<sycl::float4, 1> sS(sycl::range<1>(TK * RP / 4), cgh);  // [key][row], transposed
        sycl::local_accessor<float, 1>        sM(sycl::range<1>(R), cgh);
        sycl::local_accessor<float, 1>        sL(sycl::range<1>(R), cgh);
        sycl::local_accessor<float, 1>        sA(sycl::range<1>(RP), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(ne03, nkvh, (size_t) nsplit * DEC_WG), sycl::range<3>(1, 1, DEC_WG)),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                const int tid   = it.get_local_id(2);
                const int split = it.get_group(2);
                const int kvh   = it.get_group(1);
                const int seq   = it.get_group(0);
                auto      sg    = it.get_sub_group();
                const int w     = tid / 16;
                const int lane  = tid % 16;
                float *   sSf   = (float *) &sS[0];

                // query rows r = c * G + g: token c, head kvh * G + g; pre-scaled like the other FA kernels
                for (int e = tid; e < R * DEC_D / 4; e += DEC_WG) {
                    const int r = e / (DEC_D / 4), d4 = e % (DEC_D / 4), c = r / G, g = r % G;
                    sycl::float4 qv(0.0f);
                    if (c < ne01) {
                        qv = *(const sycl::float4 *) (Q + seq * nb03 + c * nb01 + (int64_t) (kvh * G + g) * nb02 + d4 * sizeof(sycl::float4));
                    }
                    sQ[e] = qv * scale;
                }
                if (tid < R) {
                    sM[tid] = DEC_MINF;
                    sL[tid] = 0.0f;
                }
                if (tid < RP) {
                    sA[tid] = 1.0f;
                }

                const int k_begin = split * chunk;
                const int k_end   = sycl::min(k_begin + chunk, ne11);
                const char * Kh = K + seq * nb13 + kvh * nb12;
                const char * Vh = V + seq * nb23 + kvh * nb22;
                const sycl::half * mrow = mask ? (const sycl::half *) (mask + (seq % ne33) * nb33) : nullptr;

                float o[RP];
#pragma unroll
                for (int r = 0; r < RP; ++r) {
                    o[r] = 0.0f;
                }

                const int c  = tid / TK;
                const int jj = tid % TK;
                for (int t0 = k_begin; t0 < k_end; t0 += TK) {
                    // K tile -> SLM: raw q4_0 rows, coalesced aligned dwords
                    for (int e = tid; e < TK * KW; e += DEC_WG) {
                        const int kk = e / KW, wd = e % KW, j = t0 + kk;
                        sK[kk * KWP + wd] = j < k_end ? ((const uint32_t *) (Kh + j * nb11))[wd] : 0u;
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // scores: this thread's key against the G heads of query token c
                    {
                        const int j = t0 + jj;
                        float     s[G];
#pragma unroll
                        for (int g = 0; g < G; ++g) {
                            s[g] = 0.0f;
                        }
                        if (j < k_end && c < ne01) {
                            const int kr = jj * KWP;
                            const int q0 = c * G * (DEC_D / 4);
#pragma unroll 2
                            for (int b = 0; b < NB; ++b) {
                                // block b = bytes [18b, 18b + 18): scale (2 bytes) then 16 bytes of nibbles
                                auto byte_at = [&](int o) -> uint32_t {
                                    return (sK[kr + o / 4] >> (8 * (o % 4))) & 0xFFu;
                                };
                                const int      o0 = b * (int) sizeof(block_q4_0);
                                const uint16_t hb = (uint16_t) (byte_at(o0) | (byte_at(o0 + 1) << 8));
                                const float    dk = static_cast<float>(sycl::bit_cast<sycl::half>(hb));
#pragma unroll
                                for (int i = 0; i < QK4_0 / 2; i += 4) {
                                    sycl::float4 k0, k1;
#pragma unroll
                                    for (int u = 0; u < 4; ++u) {
                                        const uint32_t by = byte_at(o0 + 2 + i + u);
                                        k0[u] = (float) ((int) (by & 0xF) - 8) * dk;
                                        k1[u] = (float) ((int) (by >> 4) - 8) * dk;
                                    }
                                    const int d4 = (b * QK4_0 + i) / 4;
#pragma unroll
                                    for (int g = 0; g < G; ++g) {
                                        const sycl::float4 qa = sQ[q0 + g * (DEC_D / 4) + d4];
                                        const sycl::float4 qb = sQ[q0 + g * (DEC_D / 4) + d4 + QK4_0 / 8];
                                        s[g] += sycl::dot(qa, k0) + sycl::dot(qb, k1);
                                    }
                                }
                            }
                            const float mv = mrow ? static_cast<float>(mrow[(c * nb31) / (int64_t) sizeof(sycl::half) + j]) : 0.0f;
#pragma unroll
                            for (int g = 0; g < G; ++g) {
                                s[g] += mv;
                            }
                        } else {
#pragma unroll
                            for (int g = 0; g < G; ++g) {
                                s[g] = -INFINITY;
                            }
                        }
#pragma unroll
                        for (int g = 0; g < G; ++g) {
                            sSf[jj * RP + c * G + g] = s[g];
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // V tile -> SLM (reusing the K tile buffer; scores are done with it)
                    static_assert(sizeof(block_q4_0) * NB % 4 == 0, "V rows are dword multiples");
                    for (int e = tid; e < TK * KW; e += DEC_WG) {
                        const int kk = e / KW, wd = e % KW, j = t0 + kk;
                        sK[kk * KWP + wd] = j < k_end ? ((const uint32_t *) (Vh + j * nb21))[wd] : 0u;
                    }

                    // online softmax, one row per subgroup at a time
                    for (int r = w; r < R; r += DEC_WG / 16) {
                        float mt = -INFINITY;
                        for (int k = lane; k < TK; k += 16) {
                            mt = sycl::fmax(mt, sSf[k * RP + r]);
                        }
                        mt = sycl::reduce_over_group(sg, mt, sycl::maximum<float>());
                        const float m_old = sM[r];
                        const float m_new = sycl::fmax(m_old, mt);  // finite: m_old starts at DEC_MINF
                        float       lt    = 0.0f;
                        for (int k = lane; k < TK; k += 16) {
                            const float p = sycl::native::exp(sSf[k * RP + r] - m_new);  // exp(-inf) = 0
                            sSf[k * RP + r] = p;
                            lt += p;
                        }
                        lt = sycl::reduce_over_group(sg, lt, sycl::plus<float>());
                        if (lane == 0) {
                            const float a = sycl::native::exp(m_old - m_new);
                            sA[r] = a;
                            sL[r] = sL[r] * a + lt;
                            sM[r] = m_new;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // P.V for head dim d = tid (padded rows R..RP-1 have P = 0 from the -inf scores)
                    {
                        const int d  = tid;
                        const int b  = d / QK4_0;
                        const int wi = d % QK4_0;
                        const int nk = sycl::min(TK, k_end - t0);
#pragma unroll
                        for (int r = 0; r < RP; ++r) {
                            o[r] *= sA[r];
                        }
                        const int ob = b * (int) sizeof(block_q4_0);          // scale bytes ob, ob + 1
                        const int oq = ob + 2 + wi % (QK4_0 / 2);             // this dim's nibble byte
                        const int sh = wi < QK4_0 / 2 ? 0 : 4;
#pragma unroll 4
                        for (int k = 0; k < nk; ++k) {
                            const int      vr   = k * KWP;
                            const uint32_t sw   = sK[vr + ob / 4] >> (8 * (ob % 4));  // ob even: both scale bytes in one dword
                            const float    dv   = static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) (sw & 0xFFFFu)));
                            const int      byte = (sK[vr + oq / 4] >> (8 * (oq % 4))) & 0xFF;
                            const float    vv   = (float) (((byte >> sh) & 0xF) - 8) * dv;
#pragma unroll
                            for (int r4 = 0; r4 < RP / 4; ++r4) {
                                const sycl::float4 p = sS[k * (RP / 4) + r4];
                                o[4 * r4 + 0] += p[0] * vv;
                                o[4 * r4 + 1] += p[1] * vv;
                                o[4 * r4 + 2] += p[2] * vv;
                                o[4 * r4 + 3] += p[3] * vv;
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int cr = r / G;
                    if (cr >= ne01) {
                        continue;
                    }
                    const int64_t jdu = ((int64_t) seq * ne01 + cr) * ne02 + kvh * G + r % G;
                    if (nsplit == 1) {
                        const float l = sL[r];
                        dst[jdu * DEC_D + tid] = l > 0.0f ? o[r] / l : 0.0f;
                    } else {
                        parts[(jdu * nsplit + split) * DEC_D + tid] = o[r];
                        if (tid == 0) {
                            meta[jdu * nsplit + split] = sycl::float2(sM[r], sL[r]);
                        }
                    }
                }
            });
    });
}

template <int G, int NQ, int TK, bool PIPE>
static void fattn_dec_q4_0(const char * Q, const char * K, const char * V, const char * mask, float * dst,
                           float * parts, sycl::float2 * meta, float scale, int ne01, int ne02, int ne11,
                           int nkvh, int ne03, int64_t nb01, int64_t nb02, int64_t nb03, int64_t nb11,
                           int64_t nb12, int64_t nb13, int64_t nb21, int64_t nb22, int64_t nb23, int64_t nb31,
                           int64_t nb33, int ne33, int nsplit, int chunk, dpct::queue_ptr stream) {
    constexpr int R   = G * NQ;             // query rows per KV head
    constexpr int RP  = (R + 3) / 4 * 4;    // padded row stride of the transposed scores
    constexpr int NRG = DEC_WG / TK;        // row groups in the score phase: thread (rg, jj) = (tid / TK, tid % TK)
    constexpr int RPT = R / NRG;            // rows per thread in the score phase
    constexpr int NB  = DEC_D / QK4_0;
    constexpr int KW  = DEC_D / QK4_0 * sizeof(block_q4_0) / 4;  // 36 dwords per q4_0 row (D 256)
    constexpr int KWP = KW + 1;                                  // padded SLM row stride (bank conflicts)
    constexpr int PKW = TK * KW / DEC_WG;                        // dwords of a K or V tile per thread
    static_assert(TK % 16 == 0 && R % NRG == 0 && (TK * KW) % DEC_WG == 0 && G % RPT == 0, "shape");  // a thread's rows share one token

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::float4, 1> sQ(sycl::range<1>(R * DEC_D / 4), cgh);
        sycl::local_accessor<uint32_t, 1>     sK(sycl::range<1>(TK * KWP), cgh);   // K tile, then V tile
        sycl::local_accessor<sycl::float4, 1> sS(sycl::range<1>(TK * RP / 4), cgh);  // [key][row], transposed
        sycl::local_accessor<float, 1>        sM(sycl::range<1>(R), cgh);
        sycl::local_accessor<float, 1>        sL(sycl::range<1>(R), cgh);
        sycl::local_accessor<float, 1>        sA(sycl::range<1>(RP), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(ne03, nkvh, (size_t) nsplit * DEC_WG), sycl::range<3>(1, 1, DEC_WG)),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                const int tid   = it.get_local_id(2);
                const int split = it.get_group(2);
                const int kvh   = it.get_group(1);
                const int seq   = it.get_group(0);
                auto      sg    = it.get_sub_group();
                const int w     = tid / 16;
                const int lane  = tid % 16;
                float *   sSf   = (float *) &sS[0];

                const int k_begin = split * chunk;
                const int k_end   = sycl::min(k_begin + chunk, ne11);
                const char * Kh = K + seq * nb13 + kvh * nb12;
                const char * Vh = V + seq * nb23 + kvh * nb22;
                const sycl::half * mrow = mask ? (const sycl::half *) (mask + (seq % ne33) * nb33) : nullptr;

                // software pipeline: a tile's raw rows are loaded into registers one phase before they are stored
                // to SLM (V during Q.K, the next K during softmax + P.V), so the loads overlap compute
                uint32_t pk[PKW], pv[PKW];
                auto load_rows = [&](const char * base, int64_t nb, int t0, uint32_t * reg) {
#pragma unroll
                    for (int i = 0; i < PKW; ++i) {
                        const int e = tid + i * DEC_WG, kk = e / KW, wd = e % KW, j = t0 + kk;
                        reg[i] = j < k_end ? ((const uint32_t *) (base + j * nb))[wd] : 0u;
                    }
                };
                auto store_rows = [&](const uint32_t * reg) {
#pragma unroll
                    for (int i = 0; i < PKW; ++i) {
                        const int e = tid + i * DEC_WG, kk = e / KW, wd = e % KW;
                        sK[kk * KWP + wd] = reg[i];
                    }
                };
                if constexpr (PIPE) {
                    load_rows(Kh, nb11, k_begin, pk);
                }

                // query rows r = c * G + g: token c, head kvh * G + g; pre-scaled like the other FA kernels
                for (int e = tid; e < R * DEC_D / 4; e += DEC_WG) {
                    const int r = e / (DEC_D / 4), d4 = e % (DEC_D / 4), c = r / G, g = r % G;
                    sycl::float4 qv(0.0f);
                    if (c < ne01) {
                        qv = *(const sycl::float4 *) (Q + seq * nb03 + c * nb01 + (int64_t) (kvh * G + g) * nb02 + d4 * sizeof(sycl::float4));
                    }
                    sQ[e] = qv * scale;
                }
                if (tid < R) {
                    sM[tid] = DEC_MINF;
                    sL[tid] = 0.0f;
                }
                if (tid < RP) {
                    sA[tid] = 1.0f;
                }

                float o[RP];
#pragma unroll
                for (int r = 0; r < RP; ++r) {
                    o[r] = 0.0f;
                }

                const int rg = tid / TK;
                const int jj = tid % TK;
                const int r0 = rg * RPT;
                for (int t0 = k_begin; t0 < k_end; t0 += TK) {
                    if constexpr (!PIPE) {
                        load_rows(Kh, nb11, t0, pk);
                    }
                    store_rows(pk);                  // K tile t
                    it.barrier(sycl::access::fence_space::local_space);
                    if constexpr (PIPE) {
                        load_rows(Vh, nb21, t0, pv);  // V tile t, in flight during Q.K
                    }

                    // scores of rows r0 .. r0 + RPT - 1 against this thread's key
                    {
                        const int j = t0 + jj;
                        float     s[RPT];
#pragma unroll
                        for (int i = 0; i < RPT; ++i) {
                            s[i] = 0.0f;
                        }
                        if (j < k_end) {
                            const int kr = jj * KWP;
                            const int qb = r0 * (DEC_D / 4);  // hoisted row base (v3 form): row rr at qb + rr * 64
#pragma unroll 2
                            for (int b = 0; b < NB; ++b) {
                                auto byte_at = [&](int o) -> uint32_t {
                                    return (sK[kr + o / 4] >> (8 * (o % 4))) & 0xFFu;
                                };
                                const int      o0 = b * (int) sizeof(block_q4_0);
                                const uint16_t hb = (uint16_t) (byte_at(o0) | (byte_at(o0 + 1) << 8));
                                const float    dk = static_cast<float>(sycl::bit_cast<sycl::half>(hb));
#pragma unroll
                                for (int i = 0; i < QK4_0 / 2; i += 4) {
                                    sycl::float4 k0, k1;
#pragma unroll
                                    for (int u = 0; u < 4; ++u) {
                                        const uint32_t by = byte_at(o0 + 2 + i + u);
                                        k0[u] = (float) ((int) (by & 0xF) - 8) * dk;
                                        k1[u] = (float) ((int) (by >> 4) - 8) * dk;
                                    }
                                    const int d4 = (b * QK4_0 + i) / 4;
#pragma unroll
                                    for (int rr = 0; rr < RPT; ++rr) {
                                        const sycl::float4 qa = sQ[qb + rr * (DEC_D / 4) + d4];
                                        const sycl::float4 qc = sQ[qb + rr * (DEC_D / 4) + d4 + QK4_0 / 8];
                                        s[rr] += sycl::dot(qa, k0) + sycl::dot(qc, k1);
                                    }
                                }
                            }
                        }
                        const int   c  = r0 / G;  // all RPT rows of this thread belong to token c
                        const float mv = (j < k_end && c < ne01 && mrow) ? static_cast<float>(mrow[(c * nb31) / (int64_t) sizeof(sycl::half) + j]) : 0.0f;
#pragma unroll
                        for (int rr = 0; rr < RPT; ++rr) {
                            sSf[jj * RP + r0 + rr] = (j < k_end && c < ne01) ? s[rr] + mv : -INFINITY;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if constexpr (!PIPE) {
                        load_rows(Vh, nb21, t0, pv);
                    }
                    store_rows(pv);                  // V tile t -> SLM (the K tile is done)
                    if constexpr (PIPE) {
                        if (t0 + TK < k_end) {
                            load_rows(Kh, nb11, t0 + TK, pk);  // K tile t + 1, in flight during softmax + P.V
                        }
                    }

                    // online softmax, one row per subgroup at a time
                    for (int r = w; r < R; r += DEC_WG / 16) {
                        float mt = -INFINITY;
                        for (int k = lane; k < TK; k += 16) {
                            mt = sycl::fmax(mt, sSf[k * RP + r]);
                        }
                        mt = sycl::reduce_over_group(sg, mt, sycl::maximum<float>());
                        const float m_old = sM[r];
                        const float m_new = sycl::fmax(m_old, mt);  // finite: m_old starts at DEC_MINF
                        float       lt    = 0.0f;
                        for (int k = lane; k < TK; k += 16) {
                            const float p = sycl::native::exp(sSf[k * RP + r] - m_new);  // exp(-inf) = 0
                            sSf[k * RP + r] = p;
                            lt += p;
                        }
                        lt = sycl::reduce_over_group(sg, lt, sycl::plus<float>());
                        if (lane == 0) {
                            const float a = sycl::native::exp(m_old - m_new);
                            sA[r] = a;
                            sL[r] = sL[r] * a + lt;
                            sM[r] = m_new;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // P.V for head dim d = tid (padded rows R..RP-1 are never written out)
                    {
                        const int d  = tid;
                        const int b  = d / QK4_0;
                        const int wi = d % QK4_0;
                        const int nk = sycl::min(TK, k_end - t0);
#pragma unroll
                        for (int r = 0; r < RP; ++r) {
                            o[r] *= sA[r];
                        }
                        const int ob = b * (int) sizeof(block_q4_0);          // scale bytes ob, ob + 1
                        const int oq = ob + 2 + wi % (QK4_0 / 2);             // this dim's nibble byte
                        const int sh = wi < QK4_0 / 2 ? 0 : 4;
#pragma unroll 4
                        for (int k = 0; k < nk; ++k) {
                            const int      vr   = k * KWP;
                            const uint32_t sw   = sK[vr + ob / 4] >> (8 * (ob % 4));  // ob even: both scale bytes in one dword
                            const float    dv   = static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) (sw & 0xFFFFu)));
                            const int      byte = (sK[vr + oq / 4] >> (8 * (oq % 4))) & 0xFF;
                            const float    vv   = (float) (((byte >> sh) & 0xF) - 8) * dv;
#pragma unroll
                            for (int r4 = 0; r4 < RP / 4; ++r4) {
                                const sycl::float4 p = sS[k * (RP / 4) + r4];
                                o[4 * r4 + 0] += p[0] * vv;
                                o[4 * r4 + 1] += p[1] * vv;
                                o[4 * r4 + 2] += p[2] * vv;
                                o[4 * r4 + 3] += p[3] * vv;
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int cr = r / G;
                    if (cr >= ne01) {
                        continue;
                    }
                    const int64_t jdu = ((int64_t) seq * ne01 + cr) * ne02 + kvh * G + r % G;
                    if (nsplit == 1) {
                        const float l = sL[r];
                        dst[jdu * DEC_D + tid] = l > 0.0f ? o[r] / l : 0.0f;
                    } else {
                        parts[(jdu * nsplit + split) * DEC_D + tid] = o[r];
                        if (tid == 0) {
                            meta[jdu * nsplit + split] = sycl::float2(sM[r], sL[r]);
                        }
                    }
                }
            });
    });
}

// ARC-LAB XMX variant for 3-4 query tokens (MTP verify): the scalar kernels are f32-ALU bound there (Q.K 163 + P.V
// 121 of 436 us per layer at 16K). Per tile of TK keys: K dequantized to f16 in SLM ([key][dim]); S^T = K Q^T on
// joint_matrix (A = 8 keys x 16 dims, B = Q^T 16 dims x 16 rows) stored straight into the transposed score layout;
// per-row online softmax writes P (f16, [row][key]); V dequantized into the same buffer; O += P V on joint_matrix
// (subgroup s owns head dims 16 s .. +15 for all row tiles), rows rescaled through the coordinate-aware apply.
template <int G, int NQ, int TK>
static void fattn_dec_q4_0_xmx(const char * Q, const char * K, const char * V, const char * mask, float * dst,
                               float * parts, sycl::float2 * meta, float scale, int ne01, int ne02, int ne11,
                               int nkvh, int ne03, int64_t nb01, int64_t nb02, int64_t nb03, int64_t nb11,
                               int64_t nb12, int64_t nb13, int64_t nb21, int64_t nb22, int64_t nb23, int64_t nb31,
                               int64_t nb33, int ne33, int nsplit, int chunk, dpct::queue_ptr stream) {
    namespace jm  = sycl::ext::oneapi::experimental::matrix;
    namespace ijm = sycl::ext::intel::experimental::matrix;
    constexpr int R   = G * NQ;             // query rows per KV head (24)
    constexpr int RP  = (R + 15) / 16 * 16; // rows padded to whole 16-wide B tiles (32)
    constexpr int MT  = R / 8;              // 8-row tiles of real rows (3)
    constexpr int NB  = DEC_D / QK4_0;
    constexpr int DS  = DEC_D + 16;         // f16 row stride of the K / V tile ([key][dim])
    constexpr int PS  = TK + 16;            // f16 row stride of P ([row][key])
    constexpr int NSG = DEC_WG / 16;        // 16 subgroups
    constexpr int KW  = DEC_D / QK4_0 * sizeof(block_q4_0) / 4;  // 36 dwords per raw q4_0 row
    constexpr int KWP = KW + 1;
    static_assert(R % 8 == 0 && TK % 16 == 0 && (TK / 8) * (RP / 16) <= NSG && DEC_D / 16 == NSG, "shape");

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 1> sQT(sycl::range<1>(DEC_D * RP), cgh);  // Q^T: [dim][row], scaled
        sycl::local_accessor<sycl::half, 1> sKV(sycl::range<1>(TK * DS), cgh);     // K tile, then V tile
        sycl::local_accessor<uint32_t, 1>   sRaw(sycl::range<1>(TK * KWP), cgh);   // raw q4_0 rows of the tile
        sycl::local_accessor<float, 1>      sS(sycl::range<1>(TK * RP), cgh);      // S^T: [key][row]
        sycl::local_accessor<sycl::half, 1> sP(sycl::range<1>(RP * PS), cgh);      // P: [row][key]
        sycl::local_accessor<float, 1>      sM(sycl::range<1>(R), cgh);
        sycl::local_accessor<float, 1>      sL(sycl::range<1>(R), cgh);
        sycl::local_accessor<float, 1>      sA(sycl::range<1>(R), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(ne03, nkvh, (size_t) nsplit * DEC_WG), sycl::range<3>(1, 1, DEC_WG)),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                const int tid   = it.get_local_id(2);
                const int split = it.get_group(2);
                const int kvh   = it.get_group(1);
                const int seq   = it.get_group(0);
                auto      sg    = it.get_sub_group();
                const int w     = tid / 16;
                const int lane  = tid % 16;
                auto      pQT   = sQT.template get_multi_ptr<sycl::access::decorated::no>();
                auto      pKV   = sKV.template get_multi_ptr<sycl::access::decorated::no>();
                auto      pS    = sS.template get_multi_ptr<sycl::access::decorated::no>();
                auto      pP    = sP.template get_multi_ptr<sycl::access::decorated::no>();

                // Q^T (padded rows are zero)
                for (int e = tid; e < DEC_D * RP; e += DEC_WG) {
                    const int d = e / RP, r = e % RP, c = r / G, g = r % G;
                    float     qv = 0.0f;
                    if (r < R && c < ne01) {
                        qv = *(const float *) (Q + seq * nb03 + c * nb01 + (int64_t) (kvh * G + g) * nb02 + d * sizeof(float)) * scale;
                    }
                    sQT[e] = sycl::half(qv);
                }
                if (tid < R) {
                    sM[tid] = DEC_MINF;
                    sL[tid] = 0.0f;
                    sA[tid] = 1.0f;
                }

                const int k_begin = split * chunk;
                const int k_end   = sycl::min(k_begin + chunk, ne11);
                const char * Kh = K + seq * nb13 + kvh * nb12;
                const char * Vh = V + seq * nb23 + kvh * nb22;
                const sycl::half * mrow = mask ? (const sycl::half *) (mask + (seq % ne33) * nb33) : nullptr;

                // O accumulators: subgroup w owns head dims 16 w .. 16 w + 15, all MT row tiles
                jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> O[MT];
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    jm::joint_matrix_fill(sg, O[mt], 0.0f);
                }

                // q4_0 rows -> f16 tile in two steps: coalesced dword copy of the raw rows, then an SLM -> SLM
                // conversion where each work item writes 8 dims of one key as one 16-byte vector
                auto dequant_tile = [&](const char * base, int64_t nb, int t0) {
                    for (int e = tid; e < TK * KW; e += DEC_WG) {
                        const int kk = e / KW, wd = e % KW, j = t0 + kk;
                        sRaw[kk * KWP + wd] = j < k_end ? ((const uint32_t *) (base + j * nb))[wd] : 0u;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int e = tid; e < TK * (DEC_D / 8); e += DEC_WG) {
                        const int kk = e / (DEC_D / 8), g = e % (DEC_D / 8), b = g / 4, part = g % 4;
                        const int row = kk * KWP;
                        auto byte_at = [&](int o) -> uint32_t { return (sRaw[row + o / 4] >> (8 * (o % 4))) & 0xFFu; };
                        const int      ob = b * (int) sizeof(block_q4_0);
                        const uint32_t sw = sRaw[row + ob / 4] >> (8 * (ob % 4));  // ob even: both scale bytes in one dword
                        const float    dd = static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) (sw & 0xFFFFu)));
                        const int      o0 = ob + 2 + (part % 2) * 8;
                        const int      sh = part < 2 ? 0 : 4;
                        sycl::vec<sycl::half, 8> v;
#pragma unroll
                        for (int u = 0; u < 8; ++u) {
                            v[u] = sycl::half((float) ((int) ((byte_at(o0 + u) >> sh) & 0xFu) - 8) * dd);
                        }
                        v.store(0, pKV + (kk * DS + g * 8));
                    }
                };

                it.barrier(sycl::access::fence_space::local_space);

                for (int t0 = k_begin; t0 < k_end; t0 += TK) {
                    dequant_tile(Kh, nb11, t0);
                    it.barrier(sycl::access::fence_space::local_space);

                    // S^T tile (keys 8 kt .. +7, rows 16 nt .. +15) per subgroup
                    if (w < (TK / 8) * (RP / 16)) {
                        const int kt = w % (TK / 8), nt = w / (TK / 8);
                        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, 8, 16> St;
                        jm::joint_matrix_fill(sg, St, 0.0f);
#pragma unroll 4
                        for (int dk = 0; dk < DEC_D; dk += 16) {
                            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major>  Ak;
                            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::row_major> Bq;
                            jm::joint_matrix_load(sg, Ak, pKV + (kt * 8) * DS + dk, DS);
                            jm::joint_matrix_load(sg, Bq, pQT + dk * RP + nt * 16, RP);
                            jm::joint_matrix_mad(sg, St, Ak, Bq, St);
                        }
                        jm::joint_matrix_store(sg, St, pS + (kt * 8) * RP + nt * 16, RP, jm::layout::row_major);
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // online softmax per row (mask, -inf for keys past the slice) -> P [row][key] f16
                    for (int r = w; r < R; r += NSG) {
                        const int c  = r / G;
                        float     sv[TK / 16];
                        float     mt = -INFINITY;
#pragma unroll
                        for (int i = 0; i < TK / 16; ++i) {
                            const int k = lane + 16 * i, j = t0 + k;
                            float     v = -INFINITY;
                            if (j < k_end && c < ne01) {
                                v = sS[k * RP + r] + (mrow ? static_cast<float>(mrow[(c * nb31) / (int64_t) sizeof(sycl::half) + j]) : 0.0f);
                            }
                            sv[i] = v;
                            mt    = sycl::fmax(mt, v);
                        }
                        mt = sycl::reduce_over_group(sg, mt, sycl::maximum<float>());
                        const float m_old = sM[r];
                        const float m_new = sycl::fmax(m_old, mt);
                        float       lt    = 0.0f;
#pragma unroll
                        for (int i = 0; i < TK / 16; ++i) {
                            const float pv = sycl::native::exp(sv[i] - m_new);
                            sP[r * PS + lane + 16 * i] = sycl::half(pv);
                            lt += pv;
                        }
                        lt = sycl::reduce_over_group(sg, lt, sycl::plus<float>());
                        if (lane == 0) {
                            const float a = sycl::native::exp(m_old - m_new);
                            sA[r] = a;
                            sL[r] = sL[r] * a + lt;
                            sM[r] = m_new;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);  // S reads done, K tile free

                    dequant_tile(Vh, nb21, t0);
                    it.barrier(sycl::access::fence_space::local_space);

                    // O = diag(alpha) O + P V for head dims 16 w .. +15
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        ijm::joint_matrix_apply(sg, O[mt], [=](float & x, size_t row, size_t) { x *= sA[mt * 8 + row]; });
                    }
#pragma unroll
                    for (int kk = 0; kk < TK; kk += 16) {
                        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, 16, 16, jm::layout::row_major> Bv;
                        jm::joint_matrix_load(sg, Bv, pKV + kk * DS + w * 16, DS);
#pragma unroll
                        for (int mt = 0; mt < MT; ++mt) {
                            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, 8, 16, jm::layout::row_major> Ap;
                            jm::joint_matrix_load(sg, Ap, pP + (mt * 8) * PS + kk, PS);
                            jm::joint_matrix_mad(sg, O[mt], Ap, Bv, O[mt]);
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // write: rows r = mt * 8 + row -> (token c, head kvh * G + g); dims 16 w + col
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    ijm::joint_matrix_apply(sg, O[mt], [=](float & x, size_t row, size_t col) {
                        const int r = mt * 8 + (int) row, c = r / G;
                        if (c < ne01) {
                            const int64_t jdu = ((int64_t) seq * ne01 + c) * ne02 + kvh * G + r % G;
                            const int     d   = w * 16 + (int) col;
                            if (nsplit == 1) {
                                const float l = sL[r];
                                dst[jdu * DEC_D + d] = l > 0.0f ? x / l : 0.0f;
                            } else {
                                parts[(jdu * nsplit + split) * DEC_D + d] = x;
                            }
                        }
                    });
                }
                if (nsplit > 1 && tid < R && tid / G < ne01) {
                    const int     r   = tid;
                    const int64_t jdu = ((int64_t) seq * ne01 + r / G) * ne02 + kvh * G + r % G;
                    meta[jdu * nsplit + split] = sycl::float2(sM[r], sL[r]);
                }
            });
    });
}


// ARC-LAB DPAS kernel (GGML_SYCL_FA_DEC_DPAS=1): the q4_0 cache feeds the XMX units without any dequantization pass.
//   Q.K: s8 x s4 DPAS (intel_sub_group_i8_i4_matrix_mad_k32, not in the public extension spec but provided by IGC on Xe2).
//        A = Q rows as int8 (one RNE scale per row per 32-dim block, computed once per call), B = the raw 16 bytes of a key's
//        q4_0 block XOR 0x88888888 (u4 n+8 -> s4 n), one key per lane. q4_0 byte i holds dims i and i+16, so A's k = 2i+h
//        is dim i + 16h: lane l's A short = (Q[l], Q[l + 16]). S += int result * dq[row] * dk[key].
//   P.V: f16 DPAS (k16) with P from SLM (A layout) and V dequantized in registers by the subgroup that owns 16 head dims.
// Query tokens beyond NQ are handled by extra work-groups (grid dim 0 = sequence x token chunk), each reading the KV again.
namespace {
typedef short    dp_short8 __attribute__((ext_vector_type(8)));
typedef int      dp_int4   __attribute__((ext_vector_type(4)));
typedef int      dp_int8   __attribute__((ext_vector_type(8)));
typedef float    dp_float8 __attribute__((ext_vector_type(8)));
typedef unsigned dp_uint4  __attribute__((ext_vector_type(4)));
}
#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL dp_int8   intel_sub_group_i8_i4_matrix_mad_k32(dp_short8 a, dp_int4 b, dp_int8 acc);
SYCL_EXTERNAL dp_float8 intel_sub_group_f16_f16_matrix_mad_k16(dp_short8 a, dp_int8 b, dp_float8 acc);
#else
inline dp_int8   intel_sub_group_i8_i4_matrix_mad_k32(dp_short8, dp_int4, dp_int8) { __builtin_unreachable(); }
inline dp_float8 intel_sub_group_f16_f16_matrix_mad_k16(dp_short8, dp_int8, dp_float8) { __builtin_unreachable(); }
#endif

template <int G, int NQ, int KG, int TC>
static void fattn_dec_q4_0_dpas(const char * Q, const char * K, const char * V, const char * mask, float * dst,
                                float * parts, sycl::float2 * meta, float scale, int ne01, int ne02, int ne11,
                                int nkvh, int ne03, int64_t nb01, int64_t nb02, int64_t nb03, int64_t nb11,
                                int64_t nb12, int64_t nb13, int64_t nb21, int64_t nb22, int64_t nb23, int64_t nb31,
                                int64_t nb33, int ne33, int nsplit, int chunk, dpct::queue_ptr stream) {
    constexpr int R    = G * NQ;             // query rows per KV head and token chunk
    constexpr int RT   = (R + 7) / 8;        // 8-row DPAS tiles
    constexpr int RP   = RT * 8;
    constexpr int TK   = 16 * KG;            // keys per tile (KG groups of 16, one key per lane)
    constexpr int NB   = DEC_D / QK4_0;      // 8 blocks per row
    constexpr int NSG  = DEC_WG / 16;        // 16 subgroups
    constexpr int NCH  = (RT + TC - 1) / TC; // row-tile chunks per key group in the Q.K phase
    constexpr int RW   = DEC_D / QK4_0 * (int) sizeof(block_q4_0) / 4;  // 36 dwords per q4_0 row
    constexpr int R16  = (R + 15) / 16 * 16; // P.V computes O^T = V^T P^T: rows are the 16-wide N side
    constexpr int RT16 = R16 / 16;
    constexpr int PS   = TK + 16;            // f16 row stride of P ([row][key]); 32-byte aligned rows, fewer bank conflicts
    static_assert(DEC_D / 16 == NSG && RW == 36, "shape");
    const int nqc = (ne01 + NQ - 1) / NQ;    // token chunks
    static const int probe_env = getenv("GGML_SYCL_FA_DEC_DPAS_PROBE") ? atoi(getenv("GGML_SYCL_FA_DEC_DPAS_PROBE")) : 0;
    const int        probe     = probe_env;  // phase probe: bit0 skip Q.K, bit1 skip P.V, bit2 skip softmax

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<dp_short8, 1> sQ8(sycl::range<1>(RT * NB * 16), cgh);   // [tile][block][lane] -> 8 rows
        sycl::local_accessor<dp_float8, 1> sDQ(sycl::range<1>(RT * NB), cgh);       // [tile][block] -> 8 rows
        sycl::local_accessor<float, 1>     sS(sycl::range<1>(RP * TK), cgh);        // [row][key]
        sycl::local_accessor<dp_int8, 1>   sP(sycl::range<1>(R16 * PS / 16), cgh);   // P f16 [row][key], rows of PS halves
        sycl::local_accessor<float, 1>     sM(sycl::range<1>(R16), cgh);
        sycl::local_accessor<float, 1>     sL(sycl::range<1>(R16), cgh);
        sycl::local_accessor<float, 1>     sA(sycl::range<1>(R16), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>((size_t) ne03 * nqc, nkvh, (size_t) nsplit * DEC_WG), sycl::range<3>(1, 1, DEC_WG)),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                const int tid   = it.get_local_id(2);
                const int split = it.get_group(2);
                const int kvh   = it.get_group(1);
                const int seq   = it.get_group(0) / nqc;
                const int c0    = (it.get_group(0) % nqc) * NQ;   // first query token of this chunk
                auto      sg    = it.get_sub_group();
                const int w     = tid / 16;
                const int lane  = tid % 16;
                short *   sQs   = (short *) &sQ8[0];
                short *   sPs   = (short *) &sP[0];

                // Q -> int8 per (row, block), RNE; padded rows / tokens past ne01 are zero
                for (int e = tid; e < RP * NB; e += DEC_WG) {
                    const int r = e / NB, b = e % NB, c = c0 + r / G, g = r % G;
                    float     x[QK4_0];
                    float     amax = 0.0f;
                    const bool real = r < R && c < ne01;
                    const float * qr = (const float *) (Q + seq * nb03 + (int64_t) c * nb01 + (int64_t) (kvh * G + g) * nb02) + b * QK4_0;
#pragma unroll
                    for (int i = 0; i < QK4_0; ++i) {
                        x[i] = real ? qr[i] * scale : 0.0f;
                        amax = sycl::fmax(amax, sycl::fabs(x[i]));
                    }
                    const float dq = amax / 127.0f;
                    const float iq = amax > 0.0f ? 127.0f / amax : 0.0f;
                    const int t = r / 8, m = r % 8;
                    ((float *) &sDQ[0])[(t * NB + b) * 8 + m] = dq;
#pragma unroll
                    for (int l = 0; l < 16; ++l) {
                        const int lo = (int) sycl::rint(x[l] * iq), hi = (int) sycl::rint(x[l + 16] * iq);
                        sQs[((t * NB + b) * 16 + l) * 8 + m] = (short) ((lo & 0xFF) | ((hi & 0xFF) << 8));
                    }
                }
                for (int e = tid; e < R16 * PS; e += DEC_WG) {
                    sPs[e] = 0;  // padded rows keep P = 0
                }
                if (tid < R16) {
                    sM[tid] = DEC_MINF;
                    sL[tid] = 0.0f;
                    sA[tid] = 1.0f;
                }
                it.barrier(sycl::access::fence_space::local_space);

                const int k_begin = split * chunk;
                const int k_end   = sycl::min(k_begin + chunk, ne11);
                const char * Kh = K + seq * nb13 + kvh * nb12;
                const char * Vh = V + seq * nb23 + kvh * nb22;
                const sycl::half * mrow = mask ? (const sycl::half *) (mask + (seq % ne33) * nb33) : nullptr;

                // O^T accumulators: subgroup w owns head dims 16 w .. +15 (two 8-dim tiles), lane = query row
                dp_float8 o[RT16][2];
#pragma unroll
                for (int t = 0; t < RT16; ++t) {
                    o[t][0] = 0.0f;
                    o[t][1] = 0.0f;
                }
                // those dims are the (w % 2) nibbles of qs bytes 0..15 of V block w / 2
                const int vb = w / 2, vsh = (w % 2) * 4;

                for (int t0 = k_begin; t0 < k_end; t0 += TK) {
                    // ---- S = Q K^T: job = (key group, chunk of TC row tiles)
                    for (int job = w; job < ((probe & 1) ? 0 : KG * NCH); job += NSG) {
                        const int  kg = job % KG, ch = job / KG;
                        const int  j  = t0 + kg * 16 + lane;
                        const bool jv = j < k_end;
                        const dp_uint4 * kr = (const dp_uint4 *) (Kh + (int64_t) (jv ? j : k_begin) * nb11);
                        uint32_t row[RW];
#pragma unroll
                        for (int i = 0; i < RW / 4; ++i) {
                            const dp_uint4 v4 = kr[i];
                            row[4 * i] = v4[0]; row[4 * i + 1] = v4[1]; row[4 * i + 2] = v4[2]; row[4 * i + 3] = v4[3];
                        }
                        dp_float8 s[TC];
#pragma unroll
                        for (int tt = 0; tt < TC; ++tt) {
                            s[tt] = 0.0f;
                        }
#pragma unroll
                        for (int b = 0; b < NB; ++b) {
                            // block b = bytes 18 b .. 18 b + 17: fp16 scale, then 16 bytes of nibbles
                            const int o0 = 18 * b;  // even b: dword aligned; odd b: 2 bytes in
                            uint32_t sc, q0, q1, q2, q3;
                            if ((o0 & 3) == 0) {
                                const int wd = o0 / 4;
                                sc = row[wd] & 0xFFFFu;
                                q0 = (row[wd] >> 16) | (row[wd + 1] << 16);
                                q1 = (row[wd + 1] >> 16) | (row[wd + 2] << 16);
                                q2 = (row[wd + 2] >> 16) | (row[wd + 3] << 16);
                                q3 = (row[wd + 3] >> 16) | (row[wd + 4] << 16);
                            } else {
                                const int wd = o0 / 4;  // scale in the high half of row[wd]
                                sc = row[wd] >> 16;
                                q0 = row[wd + 1]; q1 = row[wd + 2]; q2 = row[wd + 3]; q3 = row[wd + 4];
                            }
                            const float   dk = static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) sc));
                            const dp_int4 bv = { (int) (q0 ^ 0x88888888u), (int) (q1 ^ 0x88888888u),
                                                 (int) (q2 ^ 0x88888888u), (int) (q3 ^ 0x88888888u) };
#pragma unroll
                            for (int tt = 0; tt < TC; ++tt) {
                                const int t = ch * TC + tt;
                                if (t < RT) {
                                    const dp_int8   ia = intel_sub_group_i8_i4_matrix_mad_k32(sQ8[(t * NB + b) * 16 + lane], bv, (dp_int8) (0));
                                    const dp_float8 sc8 = sDQ[t * NB + b] * dk;  // one broadcast vector load per tile and block
                                    s[tt] += __builtin_convertvector(ia, dp_float8) * sc8;
                                }
                            }
                        }
#pragma unroll
                        for (int tt = 0; tt < TC; ++tt) {
                            const int t = ch * TC + tt;
                            if (t < RT) {
#pragma unroll
                                for (int m = 0; m < 8; ++m) {
                                    const int r = t * 8 + m, c = c0 + r / G;
                                    if (r < R) {
                                        const bool ok = jv && c < ne01;
                                        const float mv = (ok && mrow) ? static_cast<float>(mrow[((int64_t) c * nb31) / (int64_t) sizeof(sycl::half) + j]) : 0.0f;
                                        sS[r * TK + kg * 16 + lane] = ok ? s[tt][m] + mv : -INFINITY;
                                    }
                                }
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // ---- online softmax per row -> P (f16, DPAS A layout)
                    for (int r = w; r < ((probe & 4) ? 0 : R); r += NSG) {
                        float sv[KG];
                        float mt = -INFINITY;
#pragma unroll
                        for (int i = 0; i < KG; ++i) {
                            sv[i] = sS[r * TK + i * 16 + lane];
                            mt    = sycl::fmax(mt, sv[i]);
                        }
                        mt = sycl::reduce_over_group(sg, mt, sycl::maximum<float>());
                        const float m_old = sM[r];
                        const float m_new = sycl::fmax(m_old, mt);
                        float       lt    = 0.0f;
#pragma unroll
                        for (int i = 0; i < KG; ++i) {
                            const float pv = sycl::native::exp(sv[i] - m_new);
                            sPs[r * PS + i * 16 + lane] = (short) sycl::bit_cast<uint16_t>(sycl::half(pv));
                            lt += pv;
                        }
                        lt = sycl::reduce_over_group(sg, lt, sycl::plus<float>());
                        if (lane == 0) {
                            const float a = sycl::native::exp(m_old - m_new);
                            sA[r] = a;
                            sL[r] = sL[r] * a + lt;
                            sM[r] = m_new;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // ---- O^T = diag(alpha) O^T + V^T P^T: A = V^T (8 dims x 16 keys, lane = key), B = P^T (16 keys x 16 rows)
#pragma unroll
                    for (int t = 0; t < RT16; ++t) {
                        const float a = sA[t * 16 + lane];
                        o[t][0] *= a;
                        o[t][1] *= a;
                    }
#pragma unroll 1
                    for (int kg = 0; kg < ((probe & 2) ? 0 : KG); ++kg) {
                        const int  j  = t0 + kg * 16 + lane;
                        const bool jv = j < k_end;
                        const uint32_t * vr = (const uint32_t *) (Vh + (int64_t) (jv ? j : k_begin) * nb21);
                        const int o0 = 18 * vb, wd = o0 / 4;  // block start; even block: dword aligned, odd: 2 bytes in
                        const uint32_t w0 = vr[wd], w1 = vr[wd + 1], w2 = vr[wd + 2], w3 = vr[wd + 3], w4 = vr[wd + 4];
                        uint32_t sc, q[4];
                        if ((o0 & 3) == 0) {
                            sc = w0 & 0xFFFFu;
                            q[0] = (w0 >> 16) | (w1 << 16); q[1] = (w1 >> 16) | (w2 << 16);
                            q[2] = (w2 >> 16) | (w3 << 16); q[3] = (w3 >> 16) | (w4 << 16);
                        } else {
                            sc = w0 >> 16;
                            q[0] = w1; q[1] = w2; q[2] = w3; q[3] = w4;
                        }
                        const float dv = jv ? static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) sc)) : 0.0f;
                        dp_short8 va[2];
#pragma unroll
                        for (int x = 0; x < 2; ++x) {
#pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                const int byte = x * 8 + m;  // dim 16 w + byte
                                const int nib  = (q[byte / 4] >> (8 * (byte % 4) + vsh)) & 0xF;
                                va[x][m] = (short) sycl::bit_cast<uint16_t>(sycl::half((float) (nib - 8) * dv));
                            }
                        }
#pragma unroll
                        for (int t = 0; t < RT16; ++t) {
                            const dp_int8 pb = sP[((t * 16 + lane) * PS + kg * 16) / 16];
                            o[t][0] = intel_sub_group_f16_f16_matrix_mad_k16(va[0], pb, o[t][0]);
                            o[t][1] = intel_sub_group_f16_f16_matrix_mad_k16(va[1], pb, o[t][1]);
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // write: lane = row r = 16 t + lane -> (token c, head kvh * G + g); dims 16 w + 8 x + m
#pragma unroll
                for (int t = 0; t < RT16; ++t) {
                    const int r = t * 16 + lane, c = c0 + r / G;
                    if (r < R && c < ne01) {
                        const int64_t jdu = ((int64_t) seq * ne01 + c) * ne02 + kvh * G + r % G;
                        const float   l   = sL[r];
#pragma unroll
                        for (int x = 0; x < 2; ++x) {
#pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                const int d = w * 16 + x * 8 + m;
                                if (nsplit == 1) {
                                    dst[jdu * DEC_D + d] = l > 0.0f ? o[t][x][m] / l : 0.0f;
                                } else {
                                    parts[(jdu * nsplit + split) * DEC_D + d] = o[t][x][m];
                                }
                            }
                        }
                    }
                }
                if (nsplit > 1 && tid < R && c0 + tid / G < ne01) {
                    const int64_t jdu = ((int64_t) seq * ne01 + c0 + tid / G) * ne02 + kvh * G + tid % G;
                    meta[jdu * nsplit + split] = sycl::float2(sM[tid], sL[tid]);
                }
            });
    });
}

static int fattn_dec_dpas_maxq() {  // 0 = DPAS kernel off; else it serves 1 .. maxq query tokens
    static const int v = [] {
        const char * e = getenv("GGML_SYCL_FA_DEC_DPAS");
        int          n = e ? atoi(e) : 0;
        // the kernel needs Xe2's 16-lane int8 x int4 DPAS; elsewhere fall back instead of failing at the first launch
        if (n && !getenv("GGML_SYCL_FA_DEC_DPAS_ANYGPU") && !ggml_sycl_device_is_xe2()) {
            GGML_LOG_WARN("%s: GGML_SYCL_FA_DEC_DPAS needs an Xe2 or newer GPU (Arc B-series, Lunar Lake, Panther Lake); "
                          "%s is not one, using the regular decode attention\n", __func__,
                          ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.name.c_str());
            n = 0;
        }
        return n;
    }();
    return v == 1 ? 32 : v;  // =1 -> default cap
}

bool ggml_sycl_flash_attn_ext_dec_supported(const ggml_tensor * dst) {
    static const bool off = getenv("GGML_SYCL_FA_DEC_OFF") != nullptr;
    if (off) {
        return false;
    }
    const int maxq = fattn_dec_dpas_maxq();
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    float max_bias = 0.0f, softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    return K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0 && Q->type == GGML_TYPE_F32 &&
           K->ne[0] == DEC_D && V->ne[0] == DEC_D && Q->ne[0] == DEC_D &&
           Q->ne[2] == 6 * K->ne[2] && V->ne[2] == K->ne[2] &&
           Q->ne[1] >= 1 && Q->ne[1] <= (maxq ? maxq : 4) && Q->ne[3] == K->ne[3] && V->ne[3] == K->ne[3] &&  // 5-8: TILE is faster (v3<6,8> 1172-1240 vs 1049 us @16K)
           !sinks && max_bias == 0.0f && softcap == 0.0f &&
           (!mask || (mask->type == GGML_TYPE_F16 && mask->ne[2] == 1)) &&
           Q->nb[0] == sizeof(float) && K->nb[0] == ggml_type_size(K->type) && V->nb[0] == ggml_type_size(V->type) &&
           K->nb[1] % 4 == 0 && K->nb[2] % 4 == 0 && K->nb[3] % 4 == 0 && ((uintptr_t) K->data) % 4 == 0 &&
           V->nb[1] % 4 == 0 && V->nb[2] % 4 == 0 && V->nb[3] % 4 == 0 && ((uintptr_t) V->data) % 4 == 0 &&
           Q->nb[1] % 16 == 0 && Q->nb[2] % 16 == 0 && Q->nb[3] % 16 == 0 && ((uintptr_t) Q->data) % 16 == 0 &&
           (!maxq || (K->nb[1] % 16 == 0 && K->nb[2] % 16 == 0 && K->nb[3] % 16 == 0 && ((uintptr_t) K->data) % 16 == 0));
}

void ggml_sycl_flash_attn_ext_dec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int ne01 = Q->ne[1];
    const bool dpas = fattn_dec_dpas_maxq() > 0;
    const int nq   = ne01 <= 1 ? 1 : ne01 <= 2 ? 2 : ne01 <= 4 ? 4 : 8;
    // keys per tile (must match the instantiations below); DPAS: 16 x KG
    const int tk   = dpas ? (nq == 1 ? 256 : nq == 2 ? 128 : nq == 4 ? 256 : 128) : nq == 8 ? 32 : nq == 4 ? 64 : 128;
    const int ne11 = K->ne[1];

    // slices: enough work-groups to fill the GPU (~64 slices per KV head), at least one tile each
    static const int target = getenv("GGML_SYCL_FA_DEC_SPLITS") ? atoi(getenv("GGML_SYCL_FA_DEC_SPLITS")) : 64;
    const int ntiles = (ne11 + tk - 1) / tk;
    int nsplit       = std::max(1, std::min(target, ntiles));
    const int chunk  = ((ntiles + nsplit - 1) / nsplit) * tk;
    nsplit           = (ne11 + chunk - 1) / chunk;

    ggml_sycl_pool_alloc<float>        parts(ctx.pool());
    ggml_sycl_pool_alloc<sycl::float2> meta(ctx.pool());
    if (nsplit > 1) {
        parts.alloc((size_t) nsplit * ggml_nelements(dst));
        meta.alloc((size_t) nsplit * ggml_nrows(dst));
    }

    dpct::queue_ptr stream = ctx.stream();
    const char *    mdata  = mask ? (const char *) mask->data : nullptr;
    const int64_t   nb31   = mask ? mask->nb[1] : 0;
    const int64_t   nb33   = mask ? mask->nb[3] : 0;
    const int       ne33   = mask ? (int) mask->ne[3] : 1;

#define FATTN_DEC_DPAS(NQ_, KG_, TC_) fattn_dec_q4_0_dpas<6, NQ_, KG_, TC_>((const char *) Q->data, (const char *) K->data, \
        (const char *) V->data, mdata, (float *) dst->data, parts.get(), meta.get(), scale, ne01, (int) Q->ne[2], ne11,      \
        (int) K->ne[2], (int) Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2],     \
        V->nb[3], nb31, nb33, ne33, nsplit, chunk, stream)
    if (dpas) {
        switch (nq) {
            case 1: FATTN_DEC_DPAS(1, 16, 1); break;
            case 2: FATTN_DEC_DPAS(2, 8, 1); break;
            case 4: FATTN_DEC_DPAS(4, 16, 3); break;
            default: FATTN_DEC_DPAS(8, 8, 3); break;  // 8-token chunks, one work-group row per chunk
        }
    }
#undef FATTN_DEC_DPAS
#define FATTN_DEC_CALL(NQ_) fattn_dec_q4_0<6, NQ_, (NQ_ == 4 ? 64 : 128), (NQ_ != 4)>(  /* PIPE: see v4 notes */(const char *) Q->data, (const char *) K->data, (const char *) V->data, \
        mdata, (float *) dst->data, parts.get(), meta.get(), scale, ne01, (int) Q->ne[2], ne11, (int) K->ne[2],           \
        (int) Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3], nb31,    \
        nb33, ne33, nsplit, chunk, stream)
    if (!dpas) switch (nq) {
        case 1: FATTN_DEC_CALL(1); break;
        case 2: FATTN_DEC_CALL(2); break;
        case 8:  // 5-8 tokens (MTP depth 4+, short n-gram drafts): v3 kernel, 48 rows, 32-key tiles, ~59 KB SLM
            fattn_dec_q4_0_v3<6, 8>((const char *) Q->data, (const char *) K->data, (const char *) V->data, mdata,
                (float *) dst->data, parts.get(), meta.get(), scale, ne01, (int) Q->ne[2], ne11, (int) K->ne[2],
                (int) Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3],
                nb31, nb33, ne33, nsplit, chunk, stream);
            break;
        default:
            if (static const int xmx = getenv("GGML_SYCL_FA_DEC_XMX") ? atoi(getenv("GGML_SYCL_FA_DEC_XMX")) : 0; xmx) {  // ARC-LAB opt-in
#define FATTN_DEC_XMX(TK_) fattn_dec_q4_0_xmx<6, 4, TK_>((const char *) Q->data, (const char *) K->data, (const char *) V->data, mdata, \
                    (float *) dst->data, parts.get(), meta.get(), scale, ne01, (int) Q->ne[2], ne11, (int) K->ne[2],               \
                    (int) Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3],      \
                    nb31, nb33, ne33, nsplit, chunk, stream)
                if (xmx == 32) {
                    FATTN_DEC_XMX(32);
                } else {
                    FATTN_DEC_XMX(64);
                }
#undef FATTN_DEC_XMX
                break;
            }
            fattn_dec_q4_0_v3<6, 4>((const char *) Q->data, (const char *) K->data, (const char *) V->data, mdata,
                (float *) dst->data, parts.get(), meta.get(), scale, ne01, (int) Q->ne[2], ne11, (int) K->ne[2],
                (int) Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3],
                nb31, nb33, ne33, nsplit, chunk, stream);
            break;
    }
#undef FATTN_DEC_CALL

    if (nsplit > 1) {
        const int D = DEC_D;
        const sycl::range<3> grid(Q->ne[3], Q->ne[2], (size_t) ne01 * D);
        const size_t         nbytes_shared = nsplit * sizeof(sycl::float2);
        float *              parts_p       = parts.get();
        sycl::float2 *       meta_p        = meta.get();
        float *              dst_p         = (float *) dst->data;
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>(nbytes_shared), cgh);
            cgh.parallel_for(sycl::nd_range<3>(grid, sycl::range<3>(1, 1, D)),
                             [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {
                                 flash_attn_combine_results<DEC_D>(parts_p, meta_p, dst_p, nsplit,
                                     lm.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
    }
}

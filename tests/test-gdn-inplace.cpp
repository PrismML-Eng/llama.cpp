// Byte-for-byte check of the cache-writing GDN op (ggml_gated_delta_net_cache)
// against the two legacy paths the graph used to emit, on identical inputs:
//
//   A (gathered): ggml_get_rows(cache, rows_in) -> ggml_gated_delta_net(K)
//                 -> write-back cpy of the min(T, K) state planes into the
//                 cache rows slot*mem_size + head + seq
//                 (src/models/delta-net-base.cpp, !keep and keep/gathered)
//   B (rows):     ggml_gated_delta_net_rows(cache view, rows_in, K)
//                 -> ggml_set_rows(cache, snapshots, write_rows)
//                 (the Metal rows mode)
//   C (in-place): ggml_gated_delta_net_cache(cache view, rows_in, write_rows, K)
//                 -- the state never leaves the cache
//
// For every case the attention scores of C must equal A and B byte for byte
// and the WHOLE cache tensor after C must equal the cache after A and after
// B byte for byte (touched rows identical, untouched rows untouched).
//
// Matrix: K in {1, 3}, T in {1, 3}, n_seqs in {1, 4, 8}, rows_in in plane 0
// (identity: rows_in == rows_out) and in plane r in 1..K-1 (rollback read of
// the same cell), raw gates on/off, H_k == H_v and H_k == H_v/2, 1 and 4
// threads. KDA (vector gate) off, as in the qwen35 graph.
//
// CPU only; runs on tiny synthetic shapes (S_v = 32, H_v = 4) and needs no
// model.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct gdn_case {
    int64_t S_v;
    int64_t H_v;
    int64_t H_k;
    int64_t T;
    int64_t n_seqs;
    int64_t K;
    int64_t mem_size;
    int64_t head;
    int64_t r_in;      // plane read for every seq: rows_in[i] = r_in*mem_size + head + i
    bool    raw_gates;
    int     n_threads;
};

// deterministic input set shared by the three paths
struct gdn_inputs {
    std::vector<float> q, k, v, g, beta, dt_bias, a, cache;
    std::vector<int32_t> rows_in;
    std::vector<int64_t> rows_out;
};

static gdn_inputs make_inputs(const gdn_case & c, uint32_t seed) {
    std::mt19937 rng(seed);
    auto fill = [&](std::vector<float> & dst, size_t n, float lo, float hi) {
        std::uniform_real_distribution<float> d(lo, hi);
        dst.resize(n);
        for (size_t i = 0; i < n; ++i) {
            dst[i] = d(rng);
        }
    };

    const int64_t D      = c.S_v * c.S_v * c.H_v;
    const int64_t n_rows = c.mem_size * c.K;

    gdn_inputs in;
    fill(in.q,    c.S_v * c.H_k * c.T * c.n_seqs, -1.0f, 1.0f);
    fill(in.k,    c.S_v * c.H_k * c.T * c.n_seqs, -1.0f, 1.0f);
    fill(in.v,    c.S_v * c.H_v * c.T * c.n_seqs, -1.0f, 1.0f);
    if (c.raw_gates) {
        // pre-activation: g -> a[h]*softplus(g + dt_bias[h]), beta -> sigmoid(beta)
        fill(in.g,       c.H_v * c.T * c.n_seqs, -2.0f, 2.0f);
        fill(in.beta,    c.H_v * c.T * c.n_seqs, -3.0f, 3.0f);
        fill(in.dt_bias, c.H_v,                  -1.0f, 1.0f);
        fill(in.a,       c.H_v,                  -1.0f, -0.1f);
    } else {
        // activated: g = log decay in (-2, 0), beta in (0, 1)
        fill(in.g,    c.H_v * c.T * c.n_seqs, -2.0f, 0.0f);
        fill(in.beta, c.H_v * c.T * c.n_seqs,  0.0f, 1.0f);
    }
    // every row of every plane is random so a wrong row read/write shows up
    fill(in.cache, D * n_rows, -1.0f, 1.0f);

    in.rows_in.resize(c.n_seqs);
    for (int64_t i = 0; i < c.n_seqs; ++i) {
        in.rows_in[i] = (int32_t) (c.r_in * c.mem_size + c.head + i);
    }

    // set_input_rs mapping: row of (slot s, seq i) = s*mem_size + head + i
    const int64_t n_write = std::min(c.T, c.K);
    in.rows_out.resize(n_write * c.n_seqs);
    for (int64_t r = 0; r < n_write * c.n_seqs; ++r) {
        in.rows_out[r] = (r / c.n_seqs) * c.mem_size + c.head + (r % c.n_seqs);
    }

    return in;
}

struct gdn_tensors {
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * g;
    ggml_tensor * beta;
    ggml_tensor * dt_bias;
    ggml_tensor * a;
    ggml_tensor * cache;     // 2D [D, mem_size*K]
    ggml_tensor * rows_in;   // I32 [n_seqs]
    ggml_tensor * rows_out;  // I64 [min(T,K)*n_seqs]
};

static gdn_tensors make_tensors(ggml_context * ctx, const gdn_case & c, const gdn_inputs & in) {
    const int64_t D      = c.S_v * c.S_v * c.H_v;
    const int64_t n_rows = c.mem_size * c.K;

    gdn_tensors t;
    t.q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H_k, c.T, c.n_seqs);
    t.k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H_k, c.T, c.n_seqs);
    t.v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H_v, c.T, c.n_seqs);
    t.g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,     c.H_v, c.T, c.n_seqs);
    t.beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,     c.H_v, c.T, c.n_seqs);
    t.dt_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.H_v);
    t.a       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.H_v);
    t.cache   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_rows);
    t.rows_in  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.n_seqs);
    t.rows_out = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, (int64_t) in.rows_out.size());

    memcpy(t.q->data,     in.q.data(),     ggml_nbytes(t.q));
    memcpy(t.k->data,     in.k.data(),     ggml_nbytes(t.k));
    memcpy(t.v->data,     in.v.data(),     ggml_nbytes(t.v));
    memcpy(t.g->data,     in.g.data(),     ggml_nbytes(t.g));
    memcpy(t.beta->data,  in.beta.data(),  ggml_nbytes(t.beta));
    if (c.raw_gates) {
        memcpy(t.dt_bias->data, in.dt_bias.data(), ggml_nbytes(t.dt_bias));
        memcpy(t.a->data,       in.a.data(),       ggml_nbytes(t.a));
    } else {
        memset(t.dt_bias->data, 0, ggml_nbytes(t.dt_bias));
        memset(t.a->data,       0, ggml_nbytes(t.a));
    }
    memcpy(t.cache->data,    in.cache.data(),    ggml_nbytes(t.cache));
    memcpy(t.rows_in->data,  in.rows_in.data(),  ggml_nbytes(t.rows_in));
    memcpy(t.rows_out->data, in.rows_out.data(), ggml_nbytes(t.rows_out));

    return t;
}

static ggml_context * new_ctx() {
    ggml_init_params params = {
        /*.mem_size   =*/ 64u * 1024u * 1024u,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "ggml_init failed\n");
        exit(1);
    }
    return ctx;
}

struct path_result {
    std::vector<uint8_t> attn;   // attention scores, S_v*H_v*T*n_seqs floats
    std::vector<uint8_t> cache;  // whole cache tensor after the graph ran
};

static void snapshot(path_result & r, const ggml_tensor * attn_src, size_t attn_bytes, const ggml_tensor * cache) {
    r.attn.resize(attn_bytes);
    memcpy(r.attn.data(), attn_src->data, attn_bytes);
    r.cache.resize(ggml_nbytes(cache));
    memcpy(r.cache.data(), cache->data, ggml_nbytes(cache));
}

// A: gathered legacy path, exactly as build_recurrent_attn's !keep (K == 1)
//    and keep/gathered (K > 1) branches emit it
static path_result run_gathered(const gdn_case & c, const gdn_inputs & in) {
    ggml_context * ctx = new_ctx();
    gdn_tensors t = make_tensors(ctx, c, in);

    const int64_t D = c.S_v * c.S_v * c.H_v;
    const int64_t attn_score_elems    = c.S_v * c.H_v * c.T * c.n_seqs;
    const int64_t state_size_per_snap = D * c.n_seqs;
    const int64_t n_written = std::min(c.T, c.K);
    const size_t  row_size  = D * sizeof(float);

    ggml_cgraph * gf = ggml_new_graph(ctx);

    // build_rs main gather (identity or rollback plane) -> 4D state
    ggml_tensor * gathered = ggml_get_rows(ctx, t.cache, t.rows_in);
    ggml_tensor * s4d      = ggml_reshape_4d(ctx, gathered, c.S_v, c.S_v, c.H_v, c.n_seqs);

    ggml_tensor * gdn_out = ggml_gated_delta_net(ctx, t.q, t.k, t.v, t.g, t.beta, s4d, c.K);
    if (c.raw_gates) {
        ggml_gated_delta_net_set_raw_gates(gdn_out, t.dt_bias, t.a);
    }

    if (c.K == 1) {
        // build_delta_net_fused new_state view + ggml_cpy into rows [head, head + n_seqs)
        ggml_tensor * new_state = ggml_view_4d(ctx, gdn_out,
                c.S_v, c.S_v, c.H_v, c.n_seqs,
                ggml_row_size(gdn_out->type, c.S_v),
                ggml_row_size(gdn_out->type, c.S_v * c.S_v),
                ggml_row_size(gdn_out->type, c.S_v * c.S_v * c.H_v),
                ggml_row_size(gdn_out->type, c.S_v * c.H_v * c.T * c.n_seqs));
        ggml_build_forward_expand(gf,
                ggml_cpy(ctx, new_state,
                    ggml_view_2d(ctx, t.cache, D, c.n_seqs, t.cache->nb[1], c.head * row_size)));
    } else {
        // strided 3D cpy: snapshot slot s -> plane s of the cache at head
        ggml_tensor * src = ggml_view_3d(ctx, gdn_out,
            D, c.n_seqs, n_written,
            ggml_row_size(gdn_out->type, D),
            ggml_row_size(gdn_out->type, state_size_per_snap),
            ggml_row_size(gdn_out->type, attn_score_elems));
        ggml_tensor * dst = ggml_view_3d(ctx, t.cache,
            D, c.n_seqs, n_written,
            t.cache->nb[1],
            (size_t) c.mem_size * row_size,
            (size_t) c.head * row_size);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, src, dst));
    }

    if (ggml_graph_compute_with_ctx(ctx, gf, c.n_threads) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "gathered graph compute failed\n");
        exit(1);
    }

    path_result r;
    snapshot(r, gdn_out, attn_score_elems * sizeof(float), t.cache);
    ggml_free(ctx);
    return r;
}

// B: rows mode + SET_ROWS, exactly as the keep/rows branch emits it
static path_result run_rows(const gdn_case & c, const gdn_inputs & in) {
    ggml_context * ctx = new_ctx();
    gdn_tensors t = make_tensors(ctx, c, in);

    const int64_t D = c.S_v * c.S_v * c.H_v;
    const int64_t attn_score_elems = c.S_v * c.H_v * c.T * c.n_seqs;
    const int64_t n_written = std::min(c.T, c.K);

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * gdn_out = ggml_gated_delta_net_rows(ctx, t.q, t.k, t.v, t.g, t.beta, t.cache, t.rows_in, (int) c.K);
    if (c.raw_gates) {
        ggml_gated_delta_net_set_raw_gates(gdn_out, t.dt_bias, t.a);
    }

    ggml_tensor * snaps = ggml_view_2d(ctx, gdn_out,
        D, n_written * c.n_seqs,
        ggml_row_size(gdn_out->type, D),
        ggml_row_size(gdn_out->type, attn_score_elems));

    ggml_build_forward_expand(gf, ggml_set_rows(ctx, t.cache, snaps, t.rows_out));

    if (ggml_graph_compute_with_ctx(ctx, gf, c.n_threads) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "rows graph compute failed\n");
        exit(1);
    }

    path_result r;
    snapshot(r, gdn_out, attn_score_elems * sizeof(float), t.cache);
    ggml_free(ctx);
    return r;
}

// C: the cache-writing op
static path_result run_inplace(const gdn_case & c, const gdn_inputs & in) {
    ggml_context * ctx = new_ctx();
    gdn_tensors t = make_tensors(ctx, c, in);

    const int64_t attn_score_elems = c.S_v * c.H_v * c.T * c.n_seqs;

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * gdn_out = ggml_gated_delta_net_cache(ctx, t.q, t.k, t.v, t.g, t.beta, t.cache,
            t.rows_in, t.rows_out, (int) c.K, (int) c.mem_size);
    if (c.raw_gates) {
        ggml_gated_delta_net_set_raw_gates(gdn_out, t.dt_bias, t.a);
    }

    if (ggml_nelements(gdn_out) != attn_score_elems) {
        fprintf(stderr, "in-place result carries %lld elements, expected scores only (%lld)\n",
                (long long) ggml_nelements(gdn_out), (long long) attn_score_elems);
        exit(1);
    }

    ggml_build_forward_expand(gf, gdn_out);

    if (ggml_graph_compute_with_ctx(ctx, gf, c.n_threads) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "in-place graph compute failed\n");
        exit(1);
    }

    path_result r;
    snapshot(r, gdn_out, attn_score_elems * sizeof(float), t.cache);
    ggml_free(ctx);
    return r;
}

static size_t count_diff(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    if (a.size() != b.size()) {
        return std::max(a.size(), b.size());
    }
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += a[i] != b[i];
    }
    return n;
}

int main() {
    const int64_t S_v = 32;
    const int64_t H_v = 4;

    int n_cases  = 0;
    int n_failed = 0;
    uint32_t seed = 1234;

    for (int64_t K : { (int64_t) 1, (int64_t) 3 }) {
    for (int64_t T : { (int64_t) 1, (int64_t) 3 }) {
    for (int64_t n_seqs : { (int64_t) 1, (int64_t) 4, (int64_t) 8 }) {
    for (int64_t r_in = 0; r_in < K; ++r_in) {
    for (int raw = 0; raw < 2; ++raw) {
    for (int64_t H_k : { H_v, H_v / 2 }) {
    for (int n_threads : { 1, 4 }) {
        gdn_case c;
        c.S_v       = S_v;
        c.H_v       = H_v;
        c.H_k       = H_k;
        c.T         = T;
        c.n_seqs    = n_seqs;
        c.K         = K;
        c.mem_size  = n_seqs + 3; // cells before and after the ubatch stay untouched
        c.head      = 2;
        c.r_in      = r_in;
        c.raw_gates = raw != 0;
        c.n_threads = n_threads;

        const gdn_inputs in = make_inputs(c, seed++);

        const path_result A = run_gathered(c, in);
        const path_result B = run_rows(c, in);
        const path_result C = run_inplace(c, in);

        const size_t d_attn_A  = count_diff(C.attn,  A.attn);
        const size_t d_attn_B  = count_diff(C.attn,  B.attn);
        const size_t d_cache_A = count_diff(C.cache, A.cache);
        const size_t d_cache_B = count_diff(C.cache, B.cache);

        // the check is not vacuous: the in-place op must have changed the live rows
        std::vector<uint8_t> cache0((const uint8_t *) in.cache.data(),
                                    (const uint8_t *) in.cache.data() + in.cache.size() * sizeof(float));
        const bool touched = count_diff(C.cache, cache0) != 0;

        const bool ok = d_attn_A == 0 && d_attn_B == 0 && d_cache_A == 0 && d_cache_B == 0 && touched;

        printf("K=%lld T=%lld n_seqs=%lld r_in=%lld raw=%d H_k=%lld nth=%d : attn diff A/B = %zu/%zu, "
               "cache diff A/B = %zu/%zu, cache touched = %d -> %s\n",
               (long long) K, (long long) T, (long long) n_seqs, (long long) r_in, raw, (long long) H_k, n_threads,
               d_attn_A, d_attn_B, d_cache_A, d_cache_B, (int) touched, ok ? "OK" : "FAIL");

        n_cases  += 1;
        n_failed += !ok;
    }}}}}}}

    printf("%d cases, %d failed\n", n_cases, n_failed);

    return n_failed == 0 ? 0 : 1;
}

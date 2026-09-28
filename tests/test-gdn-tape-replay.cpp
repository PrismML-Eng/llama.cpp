// Checks the property the GDN verify tape relies on: running the gated delta net
// over the first c rows of a window from the pre-window state (K = 1) gives the
// same state, bit for bit, as the snapshot the K = T op stores for that prefix.
// Runs on every available backend, with and without raw gates.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void fill(ggml_tensor * t, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> buf(ggml_nelements(t));
    for (auto & x : buf) {
        x = d(rng);
    }
    ggml_backend_tensor_set(t, buf.data(), 0, ggml_nbytes(t));
}

struct shapes {
    int64_t S;     // head size
    int64_t Hk;    // q/k heads
    int64_t rep;   // v heads per q head
    int64_t T;     // window rows
};

static int run_case(ggml_backend_t backend, const shapes & sh, bool raw) {
    const int64_t S = sh.S, Hk = sh.Hk, Hv = sh.Hk * sh.rep, T = sh.T;
    const int64_t D = S * S * Hv;

    ggml_init_params ip = { 512 * ggml_tensor_overhead() + ggml_graph_overhead_custom(4096, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, 1);
    ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, T, 1);
    ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hv, T, 1);
    ggml_tensor * g  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, T, 1);
    ggml_tensor * b  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, T, 1);
    ggml_tensor * s0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, Hv, 1);
    ggml_tensor * dt = raw ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, Hv) : nullptr;
    ggml_tensor * aa = raw ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, Hv) : nullptr;

    // full window, one snapshot per row
    ggml_tensor * full = ggml_gated_delta_net(ctx, q, k, v, g, b, s0, T);
    if (raw) {
        ggml_gated_delta_net_set_raw_gates(full, dt, aa);
    }

    // replay of the first c rows from the same start state, final state only
    std::vector<ggml_tensor *> rep(T + 1, nullptr);
    for (int64_t c = 1; c <= T; ++c) {
        auto sl = [&](ggml_tensor * t) {
            return ggml_view_4d(ctx, t, t->ne[0], t->ne[1], c, 1, t->nb[1], t->nb[2], t->nb[3], 0);
        };
        rep[c] = ggml_gated_delta_net(ctx, sl(q), sl(k), sl(v), sl(g), sl(b), s0, 1);
        if (raw) {
            ggml_gated_delta_net_set_raw_gates(rep[c], dt, aa);
        }
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, full);
    for (int64_t c = 1; c <= T; ++c) {
        ggml_build_forward_expand(gf, rep[c]);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "alloc failed\n");
        ggml_free(ctx);
        return 1;
    }

    std::mt19937 rng(1234 + (int) T + (raw ? 7 : 0));
    fill(q, rng, -1.0f, 1.0f);
    fill(k, rng, -1.0f, 1.0f);
    fill(v, rng, -1.0f, 1.0f);
    fill(s0, rng, -0.5f, 0.5f);
    if (raw) {
        fill(g, rng, -2.0f, 2.0f);  // raw alpha
        fill(b, rng, -2.0f, 2.0f);  // raw beta
        fill(dt, rng, -0.5f, 0.5f);
        fill(aa, rng, -1.0f, -0.1f);
    } else {
        fill(g, rng, -1.0f, -0.01f); // log decay
        fill(b, rng, 0.05f, 0.95f);  // beta after sigmoid
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }

    const int64_t attn_full = S * Hv * T;
    std::vector<float> fullv(ggml_nelements(full));
    ggml_backend_tensor_get(full, fullv.data(), 0, ggml_nbytes(full));

    int fails = 0;
    for (int64_t c = 1; c <= T; ++c) {
        // snapshot slot j holds the state j rows before the end: after c rows -> slot T - c
        const float * snap = fullv.data() + attn_full + (T - c) * D;
        std::vector<float> rv(ggml_nelements(rep[c]));
        ggml_backend_tensor_get(rep[c], rv.data(), 0, ggml_nbytes(rep[c]));
        const float * fin = rv.data() + S * Hv * c;
        if (memcmp(snap, fin, D * sizeof(float)) != 0) {
            int64_t n_diff = 0;
            double max_d = 0.0;
            for (int64_t i = 0; i < D; ++i) {
                if (snap[i] != fin[i]) {
                    n_diff++;
                    max_d = std::max(max_d, (double) std::fabs(snap[i] - fin[i]));
                }
            }
            fprintf(stderr, "  MISMATCH c=%lld: %lld of %lld values differ, max |d| %.3g\n",
                    (long long) c, (long long) n_diff, (long long) D, max_d);
            fails++;
        }
        // the attention output rows of the prefix must match too
        if (memcmp(fullv.data(), rv.data(), S * Hv * c * sizeof(float)) != 0) {
            fprintf(stderr, "  MISMATCH c=%lld: attention output of the prefix differs\n", (long long) c);
            fails++;
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return fails;
}

int main() {
    ggml_backend_load_all();

    const shapes cases[] = {
        { 128, 16, 3, 16 }, // Bonsai 2 27B verify window (Hk 16, Hv 48, d 128, T 16)
        { 128, 16, 3,  8 }, // depth-7 window
        {  64,  4, 2,  5 }, // small, uneven
    };

    int total_fail = 0;
    int total_run  = 0;
    for (size_t di = 0; di < ggml_backend_dev_count(); ++di) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(di);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            continue;
        }
        for (const auto & sh : cases) {
            for (bool raw : { false, true }) {
                const int f = run_case(backend, sh, raw);
                printf("%-8s S=%lld Hk=%lld Hv=%lld T=%lld raw=%d: %s\n", ggml_backend_dev_name(dev),
                       (long long) sh.S, (long long) sh.Hk, (long long) (sh.Hk * sh.rep), (long long) sh.T,
                       raw ? 1 : 0, f == 0 ? "bitwise OK" : "FAIL");
                total_fail += f;
                total_run++;
            }
        }
        ggml_backend_free(backend);
    }
    printf("%d cases, %d failures\n", total_run, total_fail);
    return total_fail == 0 ? 0 : 1;
}

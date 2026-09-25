// MoE weighted-expert reduce with deliberately aliased buffers.
//
// build_moe_ffn emits MUL(down_out, router_weights) -> n_exp views -> an add chain. The CUDA
// backend fuses that into ggml_cuda_op_moe_weighted_reduce, which reads x and the weights while
// it writes dst. The graph allocator routinely places the add-chain output over the (by then
// dead) router weights, and the fused op has to notice and stage them; an x/dst overlap must
// make it decline the fusion instead. test-backend-ops cannot reach either case because it gives
// every tensor its own buffer, so this test places the tensors by hand in one buffer:
//
//   W_IN_DST  the weights sit inside dst's range   (CUDA: fusion with staged weights)
//   X_IN_DST  dst sits inside x's range            (CUDA: fusion must be rejected)
//   NONE      no overlap                           (control)
//
// Every layout is compared bit for bit against the CPU backend running the unfused ops with a
// plain allocation, since the fused kernel claims to be bit-identical to that chain. On backends
// without the fusion the unfused ops run on the aliased layout too; that is still correct because
// the weights are only read by the MUL and x only by the MUL, both of which complete before the
// add chain writes dst. GGML_CUDA_MOE_REDUCE_DEBUG=1 shows which cases fused and whether the
// weights were staged.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

enum layout_kind {
    LAYOUT_NONE,
    LAYOUT_W_IN_DST,
    LAYOUT_X_IN_DST,
};

static const char * layout_name(layout_kind k) {
    switch (k) {
        case LAYOUT_NONE:
            return "none";
        case LAYOUT_W_IN_DST:
            return "w_in_dst";
        case LAYOUT_X_IN_DST:
            return "x_in_dst";
    }
    return "?";
}

struct reduce_graph {
    ggml_context *             ctx     = nullptr;
    ggml_cgraph *              gf      = nullptr;
    ggml_tensor *              x       = nullptr;
    ggml_tensor *              w       = nullptr;
    ggml_tensor *              experts = nullptr;
    std::vector<ggml_tensor *> views;
    std::vector<ggml_tensor *> adds;  // adds.back() is the output
};

// The views are created before experts has an address, so callers must ggml_backend_view_init
// them after placing experts.
static reduce_graph build_reduce_graph(int64_t n_embd, int64_t n_exp, int64_t n_tok) {
    reduce_graph g;

    ggml_init_params params = {
        /*.mem_size   =*/ggml_tensor_overhead() * (size_t) (8 + 2 * n_exp) + ggml_graph_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    g.ctx = ggml_init(params);
    g.gf  = ggml_new_graph(g.ctx);

    g.x = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, n_embd, n_exp, n_tok);
    ggml_set_name(g.x, "x");
    g.w = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 1, n_exp, n_tok);
    ggml_set_name(g.w, "w");

    g.experts = ggml_mul(g.ctx, g.x, g.w);
    ggml_set_name(g.experts, "experts");

    for (int64_t e = 0; e < n_exp; ++e) {
        ggml_tensor * v = ggml_view_2d(g.ctx, g.experts, n_embd, n_tok, g.experts->nb[2], e * g.experts->nb[1]);
        ggml_format_name(v, "view_%" PRId64, e);
        g.views.push_back(v);
    }

    ggml_tensor * out = g.views[0];
    for (int64_t e = 1; e < n_exp; ++e) {
        out = ggml_add(g.ctx, out, g.views[e]);
        ggml_format_name(out, "add_%" PRId64, e);
        g.adds.push_back(out);
    }
    ggml_set_name(out, "out");

    // same node order as build_moe_ffn: all views before the first add
    for (ggml_tensor * v : g.views) {
        ggml_build_forward_expand(g.gf, v);
    }
    ggml_build_forward_expand(g.gf, out);

    return g;
}

static void fill_inputs(reduce_graph & g, uint32_t seed) {
    auto fill = [&](ggml_tensor * t) {
        std::vector<float> data(ggml_nelements(t));
        for (float & v : data) {
            seed = seed * 1664525u + 1013904223u;
            v    = ((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f;
        }
        ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
    };
    fill(g.x);
    fill(g.w);
}

static size_t align_up(size_t v, size_t a) {
    return (v + a - 1) / a * a;
}

// place every tensor of g by hand into one buffer on backend, in the requested layout
static ggml_backend_buffer_t place_tensors(ggml_backend_t backend, reduce_graph & g, layout_kind layout) {
    // generous upper bound: every tensor gets its own aligned slot, plus slack for the overlap slots
    size_t total = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(g.ctx); t; t = ggml_get_next_tensor(g.ctx, t)) {
        total += ggml_nbytes(t) + 512;
    }
    total += 4096;

    ggml_backend_buffer_t buf  = ggml_backend_alloc_buffer(backend, total);
    const size_t          al   = ggml_backend_buffer_get_alignment(buf);
    char *                base = (char *) ggml_backend_buffer_get_base(buf);

    size_t off   = 0;
    auto   place = [&](ggml_tensor * t, size_t at) {
        GGML_ASSERT(at % al == 0);
        GGML_ASSERT(at + ggml_nbytes(t) <= ggml_backend_buffer_get_size(buf));
        GGML_ASSERT(ggml_backend_tensor_alloc(buf, t, base + at) == GGML_STATUS_SUCCESS);
    };
    auto next = [&](ggml_tensor * t) {
        const size_t at = off;
        off             = align_up(off + ggml_nbytes(t), al);
        return at;
    };

    ggml_tensor * dst = g.adds.back();

    const size_t x_off = next(g.x);
    place(g.x, x_off);
    place(g.experts, next(g.experts));
    for (size_t i = 0; i + 1 < g.adds.size(); ++i) {
        place(g.adds[i], next(g.adds[i]));
    }

    switch (layout) {
        case LAYOUT_NONE:
            {
                place(dst, next(dst));
                place(g.w, next(g.w));
            }
            break;
        case LAYOUT_W_IN_DST:
            {
                // w strictly inside [dst, dst + nbytes(dst))
                const size_t dst_off = next(dst);
                place(dst, dst_off);
                const size_t w_off = dst_off + al;
                GGML_ASSERT(w_off + ggml_nbytes(g.w) <= dst_off + ggml_nbytes(dst));
                place(g.w, w_off);
            }
            break;
        case LAYOUT_X_IN_DST:
            {
                // dst strictly inside [x, x + nbytes(x)); x is n_exp times larger than dst
                const size_t dst_off = x_off + al;
                GGML_ASSERT(dst_off + ggml_nbytes(dst) <= x_off + ggml_nbytes(g.x));
                place(dst, dst_off);
                place(g.w, next(g.w));
            }
            break;
    }

    for (ggml_tensor * v : g.views) {
        GGML_ASSERT(ggml_backend_view_init(v) == GGML_STATUS_SUCCESS);
    }

    return buf;
}

static bool run_case(ggml_backend_t backend,
                     ggml_backend_t cpu,
                     int64_t        n_embd,
                     int64_t        n_exp,
                     int64_t        n_tok,
                     layout_kind    layout) {
    const uint32_t seed = (uint32_t) (n_embd * 1000003 + n_exp * 10007 + n_tok * 101 + layout);

    // device side: hand-placed layout
    reduce_graph          g   = build_reduce_graph(n_embd, n_exp, n_tok);
    ggml_backend_buffer_t buf = place_tensors(backend, g, layout);
    fill_inputs(g, seed);
    const ggml_status st = ggml_backend_graph_compute(backend, g.gf);

    std::vector<float> got(ggml_nelements(g.adds.back()));
    if (st == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(g.adds.back(), got.data(), 0, ggml_nbytes(g.adds.back()));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(g.ctx);

    // reference: CPU backend, unfused ops, plain allocation
    reduce_graph          r    = build_reduce_graph(n_embd, n_exp, n_tok);
    ggml_backend_buffer_t rbuf = ggml_backend_alloc_ctx_tensors(r.ctx, cpu);
    GGML_ASSERT(rbuf);
    fill_inputs(r, seed);
    GGML_ASSERT(ggml_backend_graph_compute(cpu, r.gf) == GGML_STATUS_SUCCESS);

    std::vector<float> ref(ggml_nelements(r.adds.back()));
    ggml_backend_tensor_get(r.adds.back(), ref.data(), 0, ggml_nbytes(r.adds.back()));

    ggml_backend_buffer_free(rbuf);
    ggml_free(r.ctx);

    size_t mismatches = 0;
    if (st != GGML_STATUS_SUCCESS) {
        mismatches = got.size();
    } else {
        for (size_t i = 0; i < got.size(); ++i) {
            if (memcmp(&got[i], &ref[i], sizeof(float)) != 0) {
                ++mismatches;
            }
        }
    }

    printf("  n_embd=%-3" PRId64 " n_exp=%-2" PRId64 " n_tok=%-2" PRId64 " layout=%-8s  %s", n_embd, n_exp, n_tok,
           layout_name(layout), mismatches == 0 ? "OK" : "FAIL");
    if (st != GGML_STATUS_SUCCESS) {
        printf("  (graph_compute status %d)", (int) st);
    } else if (mismatches) {
        printf("  (%zu of %zu elements differ)", mismatches, got.size());
    }
    printf("\n");

    return mismatches == 0;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu_dev);
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    GGML_ASSERT(cpu);

    // prefer CUDA (the backend with the fusion), then any GPU, then CPU against itself
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count() && !dev; ++i) {
        ggml_backend_dev_t d   = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d);
        if (reg && strcmp(ggml_backend_reg_name(reg), "CUDA") == 0) {
            dev = d;
        }
    }
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    }
    if (!dev) {
        dev = cpu_dev;
    }
    ggml_backend_t backend = dev == cpu_dev ? cpu : ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend);

    printf("test-moe-weighted-reduce: %s vs CPU reference\n", ggml_backend_dev_name(dev));

    int failed = 0;
    int total  = 0;
    for (layout_kind layout : { LAYOUT_NONE, LAYOUT_W_IN_DST, LAYOUT_X_IN_DST }) {
        for (int64_t n_exp : { 2, 4, 8, 15 }) {
            for (int64_t n_embd : { 64, 66 }) {
                for (int64_t n_tok : { 1, 7 }) {
                    ++total;
                    if (!run_case(backend, cpu, n_embd, n_exp, n_tok, layout)) {
                        ++failed;
                    }
                }
            }
        }
    }

    printf("%d/%d cases passed\n", total - failed, total);

    if (backend != cpu) {
        ggml_backend_free(backend);
    }
    ggml_backend_free(cpu);

    return failed == 0 ? 0 : 1;
}

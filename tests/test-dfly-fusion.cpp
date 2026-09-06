// DFly (AngelSpec) encoder context fusion: numerical parity + memory-layout contract.
//
// The draft-time graph in src/models/dflash.cpp mixes the raw per-target-layer features into
// one context PER DRAFT LAYER, which needs two reshape/permute round trips to put the
// contracted axis first. Both the axis handling and the resulting flat layout are easy to get
// subtly wrong in a way that still loads and still produces plausible drafts, so this test
// pins them against an independent scalar reference.
//
// Checks:
//   1. fused context == softmax-weighted mix of the raw features + shared base projection
//   2. the flattened [n_embd*n_layer, n_tokens] output is LAYER-MAJOR within each token,
//      i.e. the decoder's ggml_view_2d(offset = il*n_embd, stride = nb[1]) recovers layer il

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void graph_compute(ggml_cgraph * gf, int n_threads) {
    std::vector<uint8_t> buf;
    ggml_cplan plan = ggml_graph_plan(gf, n_threads, nullptr);
    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }
    ggml_graph_compute(gf, &plan);
}

int main() {
    // deliberately non-square and n_feat != n_layer so a transposed axis cannot pass
    const int64_t n_embd   = 8;
    const int64_t n_feat   = 5;   // target capture layers (DFly Qwen3-8B ships 5)
    const int64_t n_layer  = 3;   // draft layers
    const int64_t n_tokens = 4;
    const float   eps      = 1e-6f;

    std::vector<uint8_t> mem(64u*1024u*1024u);
    ggml_init_params ip = { mem.size(), mem.data(), false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * inp    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_feat*n_embd, n_tokens);
    ggml_tensor * fc     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_feat*n_embd, n_embd);
    ggml_tensor * fusion = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_feat, n_layer);
    ggml_tensor * cnorm  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

    srand(1234);
    auto fill = [](ggml_tensor * t) {
        float * d = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            d[i] = 2.0f*((float) rand()/(float) RAND_MAX) - 1.0f;
        }
    };
    fill(inp); fill(fc); fill(fusion); fill(cnorm);

    // ---- the graph under test: mirrors llama_model_dflash::graph<true> (DFly branch) ----
    ggml_tensor * base = ggml_mul_mat(ctx, fc, inp);                       // [n_embd, n_tokens]

    ggml_tensor * probs = ggml_soft_max(ctx, fusion);                      // [n_feat, n_layer]

    ggml_tensor * feats = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, inp, n_embd, n_feat, n_tokens), 1, 0, 2, 3));

    ggml_tensor * resid = ggml_mul_mat(ctx, probs,
                ggml_reshape_2d(ctx, feats, n_feat, n_embd*n_tokens));     // [n_layer, n_embd*n_tokens]

    resid = ggml_cont(ctx, ggml_permute(ctx,
                ggml_reshape_3d(ctx, resid, n_layer, n_embd, n_tokens), 1, 0, 2, 3));

    ggml_tensor * cur = ggml_add(ctx, resid, ggml_reshape_3d(ctx, base, n_embd, 1, n_tokens));
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), cnorm);
    cur = ggml_reshape_2d(ctx, cur, n_embd*n_layer, n_tokens);

    // the decoder's per-layer slice of the round-tripped row (layer 1, arbitrary interior pick)
    const int64_t il_probe = 1;
    ggml_tensor * slice = ggml_cont(ctx, ggml_view_2d(ctx, cur, n_embd, n_tokens,
                cur->nb[1], (size_t) il_probe*n_embd*cur->nb[0]));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);
    ggml_build_forward_expand(gf, slice);
    graph_compute(gf, 2);

    // ---- independent scalar reference ----
    const float * pinp = (const float *) inp->data;
    const float * pfc  = (const float *) fc->data;
    const float * pfus = (const float *) fusion->data;
    const float * pcn  = (const float *) cnorm->data;

    std::vector<float> ref((size_t) n_embd*n_layer*n_tokens);

    for (int64_t n = 0; n < n_tokens; ++n) {
        // shared base projection: base[h] = sum_r fc[r,h] * inp[r,n]
        std::vector<float> b(n_embd, 0.0f);
        for (int64_t h = 0; h < n_embd; ++h) {
            for (int64_t r = 0; r < n_feat*n_embd; ++r) {
                b[h] += pfc[r + h*n_feat*n_embd] * pinp[r + n*n_feat*n_embd];
            }
        }

        for (int64_t l = 0; l < n_layer; ++l) {
            // softmax over this draft layer's n_feat mixing logits
            float mx = -INFINITY;
            for (int64_t t = 0; t < n_feat; ++t) mx = std::fmax(mx, pfus[t + l*n_feat]);
            float sum = 0.0f;
            std::vector<float> w(n_feat);
            for (int64_t t = 0; t < n_feat; ++t) { w[t] = std::exp(pfus[t + l*n_feat] - mx); sum += w[t]; }
            for (int64_t t = 0; t < n_feat; ++t) w[t] /= sum;

            // residual mix of the raw features, then base + residual
            std::vector<float> v(n_embd);
            for (int64_t h = 0; h < n_embd; ++h) {
                float acc = 0.0f;
                for (int64_t t = 0; t < n_feat; ++t) {
                    acc += w[t] * pinp[(h + t*n_embd) + n*n_feat*n_embd];
                }
                v[h] = b[h] + acc;
            }

            // RMS norm over n_embd, scaled by context_norm
            float ss = 0.0f;
            for (int64_t h = 0; h < n_embd; ++h) ss += v[h]*v[h];
            const float scale = 1.0f/std::sqrt(ss/(float) n_embd + eps);
            for (int64_t h = 0; h < n_embd; ++h) {
                ref[(size_t) (h + l*n_embd) + (size_t) n*n_embd*n_layer] = v[h]*scale*pcn[h];
            }
        }
    }

    // ---- compare ----
    const float * got = (const float *) cur->data;
    double max_err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_err = std::fmax(max_err, std::fabs((double) got[i] - (double) ref[i]));
    }
    printf("fused context: max abs err = %.3e\n", max_err);

    // layer-major layout: the decoder's strided view must equal reference layer il_probe
    const float * pslice = (const float *) slice->data;
    double max_err_slice = 0.0;
    for (int64_t n = 0; n < n_tokens; ++n) {
        for (int64_t h = 0; h < n_embd; ++h) {
            const double r = ref[(size_t) (h + il_probe*n_embd) + (size_t) n*n_embd*n_layer];
            max_err_slice = std::fmax(max_err_slice, std::fabs((double) pslice[h + n*n_embd] - r));
        }
    }
    printf("layer-%lld slice: max abs err = %.3e\n", (long long) il_probe, max_err_slice);

    // a transposed fusion axis would still land within this bound only by coincidence;
    // guard against it explicitly by requiring the per-layer contexts to actually differ
    double min_sep = INFINITY;
    for (int64_t n = 0; n < n_tokens; ++n) {
        for (int64_t l = 1; l < n_layer; ++l) {
            double d = 0.0;
            for (int64_t h = 0; h < n_embd; ++h) {
                const double a = got[(size_t) (h)            + (size_t) n*n_embd*n_layer];
                const double c = got[(size_t) (h + l*n_embd) + (size_t) n*n_embd*n_layer];
                d += (a-c)*(a-c);
            }
            min_sep = std::fmin(min_sep, std::sqrt(d));
        }
    }
    printf("min per-layer context separation = %.3e\n", min_sep);

    ggml_free(ctx);

    const bool ok = max_err < 1e-4 && max_err_slice < 1e-4 && min_sep > 1e-3;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Public-API regression for FLASH_ATTN_EXT's requested F32 value accumulation.
// Link only ggml (no private CPU/CUDA entry points), so the SAME executable can
// be staged with pre-fix and post-fix shared libraries. For dynamic backends,
// --backend-dir selects the corresponding plugin directory; the OS loader must
// also resolve ggml's shared libraries from that same build.
//
// Q = K = 0 makes attention uniform, avoiding any Q/K product-precision contract.
// Each 32-token V period is eight +4096, eight +1, eight -4096, eight +1.
// The exact mean is 1/2. These are exact F16 inputs, but small contributions are
// lost when the running numerator is repeatedly rounded to F16. Groups of eight
// (not alternating signs) keep the large/small/negative ordering in each of the
// generic tile kernel's interleaved KV accumulation streams (np = 1, 2, or 4).
// Merely alternating +/-large could cancel in each GPU lane before adding small
// values, making a purported accumulation regression vacuous.
//
// Source-selected paths on RX6600: masked GQA4 uses generic tiles; queries
// 1, 2, 3, 5, 9 cross the 4/8/16/32/64-column dispatch boundaries (128-wide).
// 256-wide heads cap at 32 columns. Unaligned KV disables the GQA optimization
// and covers partial KV tiles/cross-warp reduction. CPU queries 63/64/65 cross
// its tiled boundary; one-query KV 256/512 crosses split-KV. Run CPU with both
// --threads 1 and --threads 3 to include unequal KV partitions. These are source
// dispatch expectations, not a kernel trace; the test never forces a private
// implementation or compares to a differently rounded reference backend.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct attention_case {
    const char * name;
    int d;
    int nq;
    int nk;
    int gqa;
    int kv_heads;
    int batches;
    bool mask;
};

static float channel_scale(int d, int head, int batch) {
    const float scales[] = { 1.0f, -1.0f, 0.5f, -0.5f };
    return scales[d % 4] * (head % 2 ? -1.0f : 1.0f) * (batch % 2 ? 0.5f : 1.0f);
}

static float value_at(int token) {
    switch ((token % 32) / 8) {
        case 0: return 4096.0f;
        case 2: return -4096.0f;
        default: return 1.0f;
    }
}

static double gamma_f32(int operations) {
    const double nu = operations * std::ldexp(1.0, -24);
    return nu / (1.0 - nu);
}

// Forward-error bound, not a model-output tolerance. Allow a rounded product
// and addition per V term, plus final normalization. All scores are identical,
// so any softmax exponential approximation is a COMMON factor and cancels.
// The existing GPU implementation may store that common factor in F16 before
// multiplying V while its denominator remains F32: allow one normal F16
// rounding (u16) on the mean, not independent errors on the cancelling terms.
// Rescalings between equal-score partitions are exactly one. The denominator
// has at most nk additions; its relative error is bounded by gamma(nk + 2).
// This permits different F32 reduction orders without permitting F16 running
// accumulation. No application logits or empirically fitted thresholds enter.
static double error_bound(int nk, double mean, double mean_abs) {
    const double u16 = std::ldexp(1.0, -11);
    const double numerator = gamma_f32(2 * nk + 2);
    const double denominator = gamma_f32(nk + 2);
    return ((1.0 + u16) * numerator * mean_abs +
            (u16 + denominator) * std::abs(mean)) / (1.0 - denominator);
}

static bool run_case(ggml_backend_t backend, const attention_case & tc) {
    ggml_init_params params = {
        /* mem_size   */ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
        /* mem_buffer */ nullptr,
        /* no_alloc   */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "%s: context allocation failed\n", tc.name);
        return false;
    }
    const int heads = tc.gqa * tc.kv_heads;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.d, tc.nq, heads, tc.batches);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.d, tc.nk, tc.kv_heads, tc.batches);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.d, tc.nk, tc.kv_heads, tc.batches);
    // Match production mask padding; all padded rows are also finite zero.
    const int mask_rows = (tc.nq + 63) / 64 * 64;
    ggml_tensor * mask = tc.mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.nk, mask_rows, 1, tc.batches) : nullptr;
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_set_name(out, tc.name);
    if (!ggml_backend_supports_op(backend, out)) {
        std::fprintf(stderr, "%s: backend does not support the requested F16-KV attention op\n", tc.name);
        ggml_free(ctx);
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "%s: backend allocation failed\n", tc.name);
        ggml_free(ctx);
        return false;
    }

    const std::vector<float> zeros_q(ggml_nelements(q), 0.0f);
    const std::vector<ggml_fp16_t> zeros_k(ggml_nelements(k), ggml_fp32_to_fp16(0.0f));
    std::vector<ggml_fp16_t> values(ggml_nelements(v));
    for (int b = 0; b < tc.batches; ++b) {
        for (int h = 0; h < tc.kv_heads; ++h) {
            for (int t = 0; t < tc.nk; ++t) {
                for (int d = 0; d < tc.d; ++d) {
                    const size_t i = ((size_t(b) * tc.kv_heads + h) * tc.nk + t) * tc.d + d;
                    values[i] = ggml_fp32_to_fp16(value_at(t) * channel_scale(d, h, b));
                }
            }
        }
    }
    ggml_backend_tensor_set(q, zeros_q.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, zeros_k.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, values.data(), 0, ggml_nbytes(v));
    if (mask) {
        const std::vector<ggml_fp16_t> zeros_mask(ggml_nelements(mask), ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(mask, zeros_mask.data(), 0, ggml_nbytes(mask));
    }

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    bool ok = status == GGML_STATUS_SUCCESS;
    double max_error = 0.0;
    double max_bound = 0.0;
    size_t failures = 0;
    if (ok) {
        std::vector<float> result(ggml_nelements(out));
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
        double sum = 0.0;
        double sum_abs = 0.0;
        for (int t = 0; t < tc.nk; ++t) {
            sum += value_at(t);
            sum_abs += std::abs(value_at(t));
        }
        for (int b = 0; b < tc.batches; ++b) {
            for (int t = 0; t < tc.nq; ++t) {
                for (int h = 0; h < heads; ++h) {
                    for (int d = 0; d < tc.d; ++d) {
                        const double scale = channel_scale(d, h / tc.gqa, b);
                        const double expected = scale * sum / tc.nk;
                        const double bound = error_bound(tc.nk, expected, std::abs(scale) * sum_abs / tc.nk);
                        const size_t i = ((size_t(b) * tc.nq + t) * heads + h) * tc.d + d;
                        const double error = std::abs(double(result[i]) - expected);
                        max_error = std::max(max_error, error);
                        max_bound = std::max(max_bound, bound);
                        if (!std::isfinite(result[i]) || error > bound) {
                            if (failures == 0) {
                                std::fprintf(stderr, "%s: b=%d q=%d h=%d d=%d got=%.9g expected=%.9g F32-bound=%.9g\n",
                                             tc.name, b, t, h, d, result[i], expected, bound);
                            }
                            ++failures;
                        }
                    }
                }
            }
        }
        ok = failures == 0;
    }
    std::printf("%-24s d=%d q=%d kv=%d gqa=%d batch=%d %s max-error=%.9g max-bound=%.9g failures=%zu status=%d\n",
                tc.name, tc.d, tc.nq, tc.nk, tc.gqa, tc.batches, ok ? "PASS" : "FAIL",
                max_error, max_bound, failures, int(status));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    const char * backend_name = "CPU";
    const char * backend_dir = nullptr;
    int threads = 4;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (std::strcmp(argv[i], "--backend-dir") == 0 && i + 1 < argc) {
            backend_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            char * end = nullptr;
            const long parsed = std::strtol(argv[++i], &end, 10);
            if (!end || *end || parsed < 1 || parsed > 256) {
                std::fprintf(stderr, "--threads requires an integer from 1 to 256\n");
                return 2;
            }
            threads = int(parsed);
        } else {
            std::fprintf(stderr, "usage: %s [--backend CPU|ROCm0|NAME] [--threads N] [--backend-dir PATH]\n", argv[0]);
            return 2;
        }
    }
    if (backend_dir) {
        ggml_backend_load_all_from_path(backend_dir);
    } else {
        ggml_backend_load_all();
    }
    ggml_backend_t backend = ggml_backend_init_by_name(backend_name, nullptr);
    if (!backend) {
        std::fprintf(stderr, "cannot initialize backend '%s' (no fallback or skip)\n", backend_name);
        return 2;
    }
    const ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    const ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    if (set_threads) {
        set_threads(backend, threads);
    } else if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        std::fprintf(stderr, "CPU backend does not expose thread selection\n");
        ggml_backend_free(backend);
        return 2;
    }
    std::printf("backend=%s device=%s threads=%d precision=F32 KV=F16\n",
                ggml_backend_name(backend), ggml_backend_dev_description(dev), threads);
    const attention_case cases[] = {
        { "single-unsplit",       128,  1, 256, 1, 1, 1, false },
        { "single-split",         128,  1, 512, 1, 1, 1, false },
        { "single-head-64",       64,   1, 256, 1, 1, 1, false },
        { "single-cols2",         128,  2, 256, 1, 1, 1, false },
        { "unmasked-tile",        128,  3, 288, 1, 1, 1, false },
        { "gqa4-decode-128",       128,  1, 256, 4, 2, 1, true  },
        { "gqa4-decode-256",       256,  1, 768, 4, 2, 1, true  },
        { "gqa4-cols8",            128,  2, 512, 4, 2, 1, true  },
        { "gqa4-cols16",           256,  3, 512, 4, 2, 1, true  },
        { "gqa4-cols32",           128,  5, 512, 4, 2, 1, true  },
        { "gqa4-cols64",           128,  9, 512, 4, 2, 1, true  },
        { "partial-kv-tile",       128,  2, 288, 4, 2, 1, true  },
        { "cpu-pre-tile",          256, 63, 256, 4, 2, 1, true  },
        { "cpu-tile",              128, 64, 256, 4, 2, 1, true  },
        { "cpu-tile-tail-batches",  256, 65, 256, 4, 2, 2, true  },
    };
    bool ok = true;
    for (const attention_case & tc : cases) {
        ok = run_case(backend, tc) && ok;
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}

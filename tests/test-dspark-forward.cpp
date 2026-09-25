// Phase 1 gate for the dspark forward graph (src/models/dspark.cpp).
//
// Tier 1 (--tier1): load a real dspark GGUF, feed SYNTHETIC (deterministic,
// in-process) tap features + a draft block, assert the forward pass produces
// finite, sane-shaped logits. Catches wiring bugs cheaply before worrying
// about numerical correctness.
//
// Tier 2 (--tier2 <ref.bin>): feed the SAME tap features / draft tokens /
// positions used by a reference implementation, and diff
// this program's logits against the reference logits dumped in ref.bin.
//
// ref.bin layout (little-endian):
//   int32 n_ctx_rows, int32 n_embd_cap, int32 block_size, int32 vocab_size
//   int32[n_ctx_rows]   ctx_pos
//   float32[n_ctx_rows * n_embd_cap]  ctx_feat            (row-major)
//   int32[block_size]   draft_token_ids
//   int32[block_size]   draft_pos
//   float32[block_size * vocab_size]  ref_logits          (row-major)

#include "../src/llama-ext.h"
#include "../src/llama-graph.h"
#include "../src/llama-model.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "speculative.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
// the Windows CRT has no POSIX setenv/unsetenv
static int setenv(const char * name, const char * value, int overwrite) {
    if (!overwrite && std::getenv(name)) {
        return 0;
    }
    return _putenv_s(name, value);
}

static int unsetenv(const char * name) {
    return _putenv_s(name, ""); // an empty value removes the variable
}
#endif

static std::map<std::string, std::vector<float>> trace_values;

static bool trace_node(ggml_tensor * tensor, bool ask, void *) {
    const std::string name     = tensor->name;
    const bool        selected = name == "dspark_fc" || name == "dspark_hidden_norm" || name == "dspark_log_snr_fc1" ||
                                 name == "dspark_log_snr_fc2" || name == "dspark_draft_embd_snr" ||
                                 name == "dspark_log_snr_expanded" || name == "dspark_balanced_output" ||
                                 name == "result_norm" || name.find("l_out-") == 0 || name.find("dspark_corr_") == 0;
    if (ask) {
        return selected && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor);
    }
    auto & values = trace_values[name];
    values.resize(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return true;
}

[[noreturn]] static void fail(const std::string & msg) {
    fprintf(stderr, "FAIL: %s\n", msg.c_str());
    exit(1);
}

// dspark ships no tokenizer of its own (tied to the TARGET's vocab), so
// llama_vocab_n_tokens() is 0 for a dspark GGUF -- the real vocab width only
// exists as token_embd.weight's own shape (see src/models/dspark.cpp's
// load_arch_tensors for the mirror-image fix on the loader side).
static int64_t n_vocab_from_model(const llama_model * model) {
    const struct ggml_tensor * t = model->get_tensor("token_embd.weight");
    if (!t) {
        fail("token_embd.weight not found in loaded model");
    }
    return t->ne[1];
}

struct dspark_meta {
    int64_t n_embd        = 0;
    int64_t n_vocab       = 0;
    int32_t block_size    = 0;
    int32_t mask_token_id = 0;
    int32_t n_capture     = 0;
};

// pull the handful of dspark hparams we need out of GGUF metadata via the
// generic llama_model_meta_* accessors (arch-prefixed keys).
static dspark_meta read_dspark_meta(const llama_model * model) {
    dspark_meta m;
    m.n_embd  = llama_model_n_embd(model);
    m.n_vocab = n_vocab_from_model(model);

    char buf[256];
    if (llama_model_meta_val_str(model, "dspark.dspark.block_size", buf, sizeof(buf)) > 0) {
        m.block_size = atoi(buf);
    }
    if (llama_model_meta_val_str(model, "dspark.dspark.mask_token_id", buf, sizeof(buf)) > 0) {
        m.mask_token_id = atoi(buf);
    }
    const ggml_tensor * fc = model->get_tensor("dspark.fc.weight");
    if (!fc || m.n_embd <= 0 || fc->ne[0] % m.n_embd != 0) {
        fail("invalid target-feature projection width");
    }
    m.n_capture = (int32_t) (fc->ne[0] / m.n_embd);
    if (m.n_capture <= 0) {
        fail("no target layers captured");
    }

    return m;
}

static llama_context * make_ctx(llama_model * model, uint32_t n_ctx) {
    llama_context_params cparams  = llama_context_default_params();
    cparams.n_ctx                 = n_ctx;
    cparams.n_batch               = n_ctx;
    cparams.n_ubatch              = n_ctx;
    cparams.n_seq_max             = 1;
    cparams.n_outputs_max_per_seq = 0;
    cparams.no_perf               = true;
    if (std::getenv("DSPARK_TEST_NO_FA")) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    if (std::getenv("DSPARK_TEST_TRACE") || std::getenv("DSPARK_TEST_ORACLE")) {
        cparams.cb_eval = trace_node;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fail("llama_init_from_model failed");
    }

    // dspark attention is fully non-causal (attention_mask=None, is_causal=False
    // in the reference) -- see src/models/dspark.cpp header comment.
    llama_set_causal_attn(ctx, false);

    return ctx;
}

// runs one dspark forward call: n_ctx_rows context rows (dummy token ids,
// real positions + tap features staged via llama_set_dspark_ctx) followed by
// block_size draft-block rows (real token ids). Returns the block_size*n_vocab
// logits copied out of the context.
static std::vector<float> run_forward(llama_context *              ctx,
                                      int64_t                      n_ctx_rows,
                                      int64_t                      n_embd_cap,
                                      const std::vector<float> &   ctx_feat,
                                      const std::vector<int32_t> & ctx_pos,
                                      int32_t                      block_size,
                                      const std::vector<int32_t> & draft_tokens,
                                      const std::vector<int32_t> & draft_pos,
                                      int64_t                      n_vocab,
                                      std::vector<llama_token> *   sampled     = nullptr,
                                      int32_t                      output_rows = -1) {
    if ((int64_t) ctx_feat.size() != n_ctx_rows * n_embd_cap) {
        fail("ctx_feat size mismatch");
    }
    if ((int64_t) ctx_pos.size() != n_ctx_rows) {
        fail("ctx_pos size mismatch");
    }
    if ((int64_t) draft_tokens.size() != block_size) {
        fail("draft_tokens size mismatch");
    }
    if ((int64_t) draft_pos.size() != block_size) {
        fail("draft_pos size mismatch");
    }

    if (output_rows == -1) {
        output_rows = block_size;
    }
    if (output_rows < 1 || output_rows > block_size) {
        fail("invalid output row count");
    }
    llama_set_dspark_ctx(ctx, ctx_feat.data(), n_ctx_rows, n_embd_cap, ctx_pos.data());

    const int32_t n_tokens = (int32_t) n_ctx_rows + block_size;
    llama_batch   batch    = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens         = n_tokens;

    for (int32_t i = 0; i < (int32_t) n_ctx_rows; ++i) {
        batch.token[i]     = 0;  // dummy: unused (context rows never join the trunk/embedding path)
        batch.pos[i]       = ctx_pos[i];
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 0;
    }
    for (int32_t j = 0; j < block_size; ++j) {
        const int32_t i    = (int32_t) n_ctx_rows + j;
        batch.token[i]     = draft_tokens[j];
        batch.pos[i]       = draft_pos[j];
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = j < output_rows;
    }

    int32_t rc;
    try {
        rc = llama_decode(ctx, batch);
    } catch (...) {
        llama_batch_free(batch);
        llama_set_dspark_ctx(ctx, nullptr, 0, 0, nullptr);
        throw;
    }
    llama_batch_free(batch);
    if (rc != 0) {
        fail("llama_decode returned " + std::to_string(rc));
    }

    llama_set_dspark_ctx(ctx, nullptr, 0, 0, nullptr);  // clear staged context

    if (sampled) {
        sampled->clear();
        for (int32_t i = 0; i < output_rows; ++i) {
            sampled->push_back(llama_get_sampled_token_ith(ctx, i - output_rows));
        }
        return {};
    }

    float * logits = llama_get_logits(ctx);
    if (!logits) {
        fail("llama_get_logits returned null");
    }

    return std::vector<float>(logits, logits + (size_t) output_rows * n_vocab);
}

static std::vector<float> correction_reference(const llama_model *          model,
                                               const float *                hidden,
                                               const std::vector<int32_t> & ids,
                                               const dspark_meta &          meta) {
    if (!hidden) {
        fail("missing trunk embeddings for correction oracle");
    }
    auto read = [&](const char * name) {
        const ggml_tensor * tensor = model->get_tensor(name);
        if (!tensor || (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_BF16)) {
            fail("correction oracle requires F32 or BF16 tensors");
        }
        std::vector<float> data(ggml_nelements(tensor));
        if (tensor->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> packed(data.size());
            ggml_backend_tensor_get(tensor, packed.data(), 0, packed.size() * sizeof(ggml_bf16_t));
            ggml_bf16_to_fp32_row(packed.data(), data.data(), data.size());
        } else {
            ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
        }
        return data;
    };
    auto norm = [](std::vector<float> value, const std::vector<float> & weight) {
        double square = 0;
        for (float x : value) {
            square += double(x) * x;
        }
        const float scale = 1.0f / std::sqrt(float(square / value.size()) + 1e-6f);
        for (size_t i = 0; i < value.size(); ++i) {
            value[i] *= scale * weight[i];
        }
        return value;
    };
    auto mv = [](const std::vector<float> & weight, const std::vector<float> & x) {
        std::vector<float> result(weight.size() / x.size(), 0);
        for (size_t row = 0; row < result.size(); ++row) {
            double sum = 0;
            for (size_t col = 0; col < x.size(); ++col) {
                sum += double(weight[row * x.size() + col]) * x[col];
            }
            result[row] = float(sum);
        }
        return result;
    };
    const auto         embeddings = read("token_embd.weight");
    const auto         hn         = read("dspark.correction_hidden_norm.weight");
    const auto         en         = read("dspark.correction_embed_norm.weight");
    const auto         gate       = read("dspark.correction_gate.weight");
    const auto         up         = read("dspark.correction_up.weight");
    const auto         down       = read("dspark.correction_down.weight");
    const auto         head       = model->get_tensor("output.weight") ? read("output.weight") : embeddings;
    const auto         a          = read("dspark.markov_head_a.weight");
    const auto         b          = read("dspark.markov_head_b.weight");
    const size_t       rank       = a.size() / meta.n_vocab;
    std::vector<float> result;
    int32_t            previous = ids[0];
    for (size_t slot = 0; slot < ids.size(); ++slot) {
        std::vector<float> h(hidden + slot * meta.n_embd, hidden + (slot + 1) * meta.n_embd);
        std::vector<float> e(embeddings.begin() + previous * meta.n_embd,
                             embeddings.begin() + (previous + 1) * meta.n_embd);
        auto               input = norm(h, hn);
        e                        = norm(e, en);
        input.insert(input.end(), e.begin(), e.end());
        auto g = mv(gate, input);
        auto u = mv(up, input);
        for (size_t i = 0; i < g.size(); ++i) {
            g[i] = g[i] / (1 + std::exp(-g[i])) * u[i];
        }
        auto delta = mv(down, g);
        for (size_t i = 0; i < h.size(); ++i) {
            h[i] += delta[i];
        }
        auto               logits = mv(head, h);
        std::vector<float> latent(a.begin() + ids[slot] * rank, a.begin() + (ids[slot] + 1) * rank);
        auto               bias = mv(b, latent);
        for (size_t i = 0; i < logits.size(); ++i) {
            logits[i] += bias[i];
        }
        logits[meta.mask_token_id] = -1e6f;
        previous                   = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
        result.insert(result.end(), logits.begin(), logits.end());
    }
    return result;
}

static int run_tier1(const std::string & model_path, bool metal = false, std::vector<float> * output = nullptr) {
    printf("=== Tier 1: synthetic tap features, wiring/shape/finiteness check ===\n");

    llama_model_params mparams   = llama_model_default_params();
    mparams.n_gpu_layers         = metal ? 999 : 0;
    ggml_backend_dev_t devices[] = { nullptr, nullptr };
    if (metal) {
        ggml_backend_reg_t registry = ggml_backend_reg_by_name("MTL");
        if (registry && ggml_backend_reg_dev_count(registry) > 0) {
            devices[0] = ggml_backend_reg_dev_get(registry, 0);
        }
        if (!devices[0]) {
            fail("Metal device unavailable; refusing CPU fallback");
        }
        mparams.devices = devices;
    }

    llama_model_ptr head_source;
    if (const char * source_path = std::getenv("DSPARK_TEST_SHARED_HEAD_SOURCE")) {
        head_source.reset(llama_model_load_from_file(source_path, mparams));
        if (!head_source) {
            fail("could not load shared head source");
        }
        mparams.dspark_head_source = head_source.get();
    }
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fail("failed to load model: " + model_path);
    }

    if (head_source) {
        if (model->output != head_source->output || model->output->buffer != head_source->output->buffer) {
            fail("shared head did not borrow the source tensor and buffer");
        }
        printf("Shared head identity PASSED: tensor and backend buffer are borrowed\n");
    }
    dspark_meta meta = read_dspark_meta(model);
    printf("n_embd=%lld n_vocab=%lld block_size=%d mask_token_id=%d n_capture=%d\n", (long long) meta.n_embd,
           (long long) meta.n_vocab, meta.block_size, meta.mask_token_id, meta.n_capture);
    if (meta.block_size <= 0) {
        fail("could not read dspark.dspark.block_size from GGUF");
    }

    const int64_t n_ctx_rows = 6;
    const int64_t n_embd_cap = (int64_t) meta.n_capture * meta.n_embd;
    const int32_t block_size = meta.block_size;

    llama_context * ctx = make_ctx(model, (uint32_t) (n_ctx_rows + block_size));

    // deterministic synthetic tap features (fixed seed, no external data needed).
    std::mt19937                    rng(1234);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    std::vector<float>              ctx_feat((size_t) n_ctx_rows * n_embd_cap);
    for (auto & v : ctx_feat) {
        v = dist(rng);
    }

    std::vector<int32_t> ctx_pos(n_ctx_rows);
    for (int64_t i = 0; i < n_ctx_rows; ++i) {
        ctx_pos[i] = (int32_t) i;
    }

    std::vector<int32_t> draft_tokens(block_size, meta.mask_token_id);
    draft_tokens[0] = (int32_t) std::min<int64_t>(1000, meta.n_vocab - 1);
    if (draft_tokens[0] == meta.mask_token_id) {
        draft_tokens[0] = 0;
    }
    std::vector<int32_t> draft_pos(block_size);
    for (int32_t i = 0; i < block_size; ++i) {
        draft_pos[i] = (int32_t) (n_ctx_rows + i);
    }

    std::vector<float> logits =
        run_forward(ctx, n_ctx_rows, n_embd_cap, ctx_feat, ctx_pos, block_size, draft_tokens, draft_pos, meta.n_vocab);
    if (output) {
        *output = logits;
    }
    llama_dspark_meta contract;
    if (llama_model_dspark_get_meta(model, &contract) && contract.graph_corrected && std::getenv("DSPARK_TEST_TRACE")) {
        for (int32_t slot = 0; slot < block_size; ++slot) {
            const auto   suffix = "-" + std::to_string(slot);
            const auto & before = trace_values.at("dspark_corr_unmasked" + suffix);
            const auto & after  = trace_values.at("dspark_corr_masked" + suffix);
            if (before.size() != size_t(meta.n_vocab) || after.size() != before.size()) {
                fail("mask suppression trace shape mismatch");
            }
            for (size_t token = 0; token < after.size(); ++token) {
                const float expected = token == size_t(meta.mask_token_id) ? -1e6f : before[token];
                if (after[token] != expected) {
                    fprintf(stderr, "Mask suppression changed slot=%d token=%zu: before=%g after=%g\n", slot, token,
                            before[token], after[token]);
                    llama_free(ctx);
                    llama_model_free(model);
                    fail("mask suppression must preserve every non-mask logit exactly");
                }
            }
        }
        printf("Mask suppression invariance PASSED: backend=%s slots=%d\n", metal ? "Metal" : "CPU", block_size);
    }
    if (llama_model_dspark_get_meta(model, &contract) && contract.graph_corrected &&
        std::getenv("DSPARK_TEST_ORACLE")) {
        auto  reference = correction_reference(model, trace_values.at("result_norm").data(), draft_tokens, meta);
        float error     = 0;
        for (size_t i = 0; i < logits.size(); ++i) {
            error = std::max(error, std::fabs(logits[i] - reference[i]));
            if (std::fabs(logits[i] - reference[i]) > 1e-3f + 1e-2f * std::fabs(reference[i])) {
                fprintf(stderr,
                        "Correction mismatch: backend=%s slot=%zu token=%zu actual=%g reference=%g abs_error=%g\n",
                        metal ? "Metal" : "CPU", size_t(i / meta.n_vocab), size_t(i % meta.n_vocab), logits[i],
                        reference[i], std::fabs(logits[i] - reference[i]));
                llama_free(ctx);
                llama_model_free(model);
                fail("hidden correction differs from scalar CPU oracle");
            }
        }
        printf("Independent correction oracle PASSED: max_abs_error=%g\n", error);
    }

    size_t n_nonfinite = 0;
    float  min_v = logits[0], max_v = logits[0];
    for (float v : logits) {
        if (!std::isfinite(v)) {
            n_nonfinite++;
        }
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }

    printf("logits: count=%zu min=%g max=%g non_finite=%zu\n", logits.size(), min_v, max_v, n_nonfinite);

    for (int32_t p = 0; p < block_size; ++p) {
        const float * row    = logits.data() + (size_t) p * meta.n_vocab;
        int64_t       argmax = 0;
        for (int64_t v = 1; v < meta.n_vocab; ++v) {
            if (row[v] > row[argmax]) {
                argmax = v;
            }
        }
        printf("  pos %d: argmax token_id=%lld logit=%g\n", p, (long long) argmax, row[argmax]);
    }

    llama_free(ctx);
    llama_model_free(model);

    if (n_nonfinite > 0) {
        fail("Tier 1 FAILED: non-finite logits present");
    }
    if (logits.size() != (size_t) block_size * meta.n_vocab) {
        fail("Tier 1 FAILED: unexpected logits size");
    }

    if (const char * path = std::getenv("DSPARK_TEST_LOGITS_FILE")) {
        std::ofstream dump(path, std::ios::binary);
        dump.write(reinterpret_cast<const char *>(logits.data()), logits.size() * sizeof(float));
        if (!dump) {
            fail("could not write synthetic forward logits");
        }
    }
    printf("Tier 1 PASSED: finite, correctly-shaped logits (%d x %lld)\n", block_size, (long long) meta.n_vocab);
    return 0;
}

static int run_tier2(const std::string & model_path, const std::string & ref_path) {
    printf("=== Tier 2: real drafter weights, deterministic tap features, diff vs DeepSpec reference ===\n");

    std::ifstream f(ref_path, std::ios::binary);
    if (!f) {
        fail("could not open ref file: " + ref_path);
    }

    int32_t n_ctx_rows_i, n_embd_cap_i, block_size, vocab_size;
    f.read((char *) &n_ctx_rows_i, 4);
    f.read((char *) &n_embd_cap_i, 4);
    f.read((char *) &block_size, 4);
    f.read((char *) &vocab_size, 4);
    if (!f) {
        fail("ref file truncated (header)");
    }

    const int64_t n_ctx_rows = n_ctx_rows_i;
    const int64_t n_embd_cap = n_embd_cap_i;

    std::vector<int32_t> ctx_pos(n_ctx_rows);
    f.read((char *) ctx_pos.data(), n_ctx_rows * sizeof(int32_t));

    std::vector<float> ctx_feat((size_t) n_ctx_rows * n_embd_cap);
    f.read((char *) ctx_feat.data(), ctx_feat.size() * sizeof(float));

    std::vector<int32_t> draft_tokens(block_size);
    f.read((char *) draft_tokens.data(), block_size * sizeof(int32_t));

    std::vector<int32_t> draft_pos(block_size);
    f.read((char *) draft_pos.data(), block_size * sizeof(int32_t));

    std::vector<float> ref_logits((size_t) block_size * vocab_size);
    f.read((char *) ref_logits.data(), ref_logits.size() * sizeof(float));
    if (!f) {
        fail("ref file truncated (payload)");
    }

    printf("ref: n_ctx_rows=%lld n_embd_cap=%lld block_size=%d vocab_size=%d\n", (long long) n_ctx_rows,
           (long long) n_embd_cap, block_size, vocab_size);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 0;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fail("failed to load model: " + model_path);
    }

    const int64_t n_vocab = n_vocab_from_model(model);
    if (n_vocab != vocab_size) {
        fail("vocab size mismatch between GGUF (" + std::to_string(n_vocab) + ") and reference (" +
             std::to_string(vocab_size) + ")");
    }

    llama_context * ctx = make_ctx(model, (uint32_t) (n_ctx_rows + block_size));

    std::vector<float> logits =
        run_forward(ctx, n_ctx_rows, n_embd_cap, ctx_feat, ctx_pos, block_size, draft_tokens, draft_pos, n_vocab);

    llama_free(ctx);
    llama_model_free(model);

    // --- diff ---
    double  sum_abs_diff = 0.0, max_abs_diff = 0.0;
    size_t  n_nonfinite      = 0;
    int32_t argmax_matches   = 0;
    double  top5_overlap_sum = 0.0;

    for (int32_t p = 0; p < block_size; ++p) {
        const float * a = logits.data() + (size_t) p * n_vocab;      // C++
        const float * b = ref_logits.data() + (size_t) p * n_vocab;  // python

        int64_t argmax_a = 0, argmax_b = 0;
        for (int64_t v = 1; v < n_vocab; ++v) {
            if (a[v] > a[argmax_a]) {
                argmax_a = v;
            }
            if (b[v] > b[argmax_b]) {
                argmax_b = v;
            }
            const double d = std::fabs((double) a[v] - (double) b[v]);
            sum_abs_diff += d;
            max_abs_diff = std::max(max_abs_diff, d);
            if (!std::isfinite(a[v])) {
                n_nonfinite++;
            }
        }
        if (argmax_a == argmax_b) {
            argmax_matches++;
        }

        // top-5 overlap (set intersection size / 5)
        std::vector<int64_t> idx_a(n_vocab), idx_b(n_vocab);
        for (int64_t v = 0; v < n_vocab; ++v) {
            idx_a[v] = v;
            idx_b[v] = v;
        }
        std::partial_sort(idx_a.begin(), idx_a.begin() + 5, idx_a.end(),
                          [&](int64_t x, int64_t y) { return a[x] > a[y]; });
        std::partial_sort(idx_b.begin(), idx_b.begin() + 5, idx_b.end(),
                          [&](int64_t x, int64_t y) { return b[x] > b[y]; });
        std::vector<int64_t> top5_a(idx_a.begin(), idx_a.begin() + 5), top5_b(idx_b.begin(), idx_b.begin() + 5);
        std::sort(top5_a.begin(), top5_a.end());
        std::sort(top5_b.begin(), top5_b.end());
        std::vector<int64_t> inter;
        std::set_intersection(top5_a.begin(), top5_a.end(), top5_b.begin(), top5_b.end(), std::back_inserter(inter));
        top5_overlap_sum += inter.size() / 5.0;

        printf("  pos %d: cpp_argmax=%lld py_argmax=%lld cpp_top1_logit=%g py_top1_logit=%g top5_overlap=%d/5\n", p,
               (long long) argmax_a, (long long) argmax_b, a[argmax_a], b[argmax_b], (int) inter.size());
    }

    const double mean_abs_diff     = sum_abs_diff / ((double) block_size * n_vocab);
    const double argmax_match_rate = (double) argmax_matches / block_size;
    const double mean_top5_overlap = top5_overlap_sum / block_size;

    printf("\n--- Tier 2 summary ---\n");
    printf("mean_abs_diff=%.6g max_abs_diff=%.6g non_finite=%zu\n", mean_abs_diff, max_abs_diff, n_nonfinite);
    printf("argmax_match_rate=%.3f (%d/%d) mean_top5_overlap=%.3f\n", argmax_match_rate, argmax_matches, block_size,
           mean_top5_overlap);

    if (n_nonfinite > 0) {
        fail("Tier 2 FAILED: non-finite logits");
    }

    return 0;
}

static bool report_exact_vector(const char *               label,
                                const std::vector<float> & reference,
                                const std::vector<float> & actual) {
    if (reference.empty() || reference.size() != actual.size()) {
        fail(std::string(label) + ": empty or mismatched shape");
    }
    size_t differences = 0;
    float  max_error   = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(actual[i])) {
            fail(std::string(label) + ": nonfinite value");
        }
        differences += std::memcmp(&reference[i], &actual[i], sizeof(float)) != 0;
        max_error = std::max(max_error, std::fabs(reference[i] - actual[i]));
    }
    printf("A/B %s: count=%zu differing_bits_elements=%zu max_abs_error=%g\n", label, actual.size(), differences,
           max_error);
    return differences == 0;
}

static int run_snr_ab(const std::string & model_path, bool metal) {
    std::vector<float> baseline, candidate;
    // Fresh contexts prevent graph reuse from masking the environment switch.
    if (setenv("DSPARK_TEST_TRACE", "1", 1) || setenv("LLAMA_DSPARK_SNR_TWO_ROWS", "0", 1)) {
        fail("setenv failed");
    }
    trace_values.clear();
    if (run_tier1(model_path, metal, &baseline)) {
        fail("baseline forward failed");
    }
    const auto reference = trace_values;
    if (!reference.count("dspark_draft_embd_snr") || reference.count("dspark_log_snr_expanded")) {
        fail("baseline must run unfused conditioning");
    }
    if (setenv("LLAMA_DSPARK_SNR_TWO_ROWS", "1", 1)) {
        fail("setenv failed");
    }
    trace_values.clear();
    if (run_tier1(model_path, metal, &candidate)) {
        fail("candidate forward failed");
    }
    if (!trace_values.count("dspark_log_snr_expanded") || !trace_values.count("dspark_draft_embd_snr") ||
        !trace_values.count("dspark_log_snr_fc2") ||
        trace_values.at("dspark_log_snr_fc2").size() >= reference.at("dspark_log_snr_fc2").size()) {
        fail("candidate did not exercise two-row conditioning; no vacuous pass");
    }
    bool exact = report_exact_vector("conditioned embeddings", reference.at("dspark_draft_embd_snr"),
                                     trace_values.at("dspark_draft_embd_snr"));
    exact &= report_exact_vector("expanded conditioning", reference.at("dspark_log_snr_fc2"),
                                 trace_values.at("dspark_log_snr_expanded"));
    exact &= report_exact_vector("full logits", baseline, candidate);
    if (!exact) {
        fail("strict bitwise parity failed; keep candidate disabled");
    }
    printf("SNR A/B strict parity PASSED (synthetic features; not a serving acceptance benchmark)\n");
    return 0;
}

static int run_output_ab(const std::string & model_path, bool metal) {
    std::vector<float> baseline, candidate;
    if (setenv("DSPARK_TEST_TRACE", "1", 1) || setenv("LLAMA_DSPARK_BALANCED_OUTPUT", "0", 1)) {
        fail("setenv failed");
    }
    trace_values.clear();
    if (run_tier1(model_path, metal, &baseline)) {
        fail("baseline forward failed");
    }
    if (trace_values.count("dspark_balanced_output")) {
        fail("baseline used balanced output");
    }
    if (setenv("LLAMA_DSPARK_BALANCED_OUTPUT", "1", 1)) {
        fail("setenv failed");
    }
    trace_values.clear();
    if (run_tier1(model_path, metal, &candidate)) {
        fail("candidate forward failed");
    }
    if (!trace_values.count("dspark_balanced_output")) {
        fail("candidate branch not exercised");
    }
    if (!report_exact_vector("balanced output logits", baseline, candidate)) {
        fail("output parity failed");
    }
    printf("Balanced output strict parity PASSED (not a serving benchmark)\n");
    return 0;
}

static int run_runtime_ab(const std::string & model_path, const char * backend = nullptr, bool bench = false) {
    auto mp                      = llama_model_default_params();
    mp.n_gpu_layers              = backend ? 999 : 0;
    ggml_backend_dev_t devices[] = { nullptr, nullptr };
    if (backend) {
        auto registry = ggml_backend_reg_by_name(backend);
        if (!registry || ggml_backend_reg_dev_count(registry) == 0) {
            fail("requested GPU backend unavailable");
        }
        devices[0] = ggml_backend_reg_dev_get(registry, 0);
        mp.devices = devices;
    }
    auto * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fail("model load failed");
    }
    const auto        meta = read_dspark_meta(model);
    llama_dspark_meta contract;
    if (!llama_model_dspark_get_meta(model, &contract) || !contract.graph_corrected) {
        fail("runtime A/B requires a graph-corrected drafter");
    }
    const int64_t width       = meta.n_capture * meta.n_embd;
    int           checks      = 0;
    const char *  head_env    = std::getenv("DSPARK_TEST_HEAD_TOP_K");
    const int     bench_top_k = head_env ? std::stoi(head_env) : 4096;
    for (int top_k : (bench ? std::vector<int>{ bench_top_k } : std::vector<int>{ 0, 32 })) {
        if (top_k > meta.n_vocab) {
            continue;
        }
        setenv("DSPARK_HEAD_TOP_K", std::to_string(top_k).c_str(), 1);
        for (int depth : { 1, 2, 4, 8 }) {
            if (bench && depth != meta.block_size) {
                continue;
            }
            if (depth > meta.block_size) {
                continue;
            }
            std::vector<float> expected;
            for (int variant : { 0, 4, 1, 2, 3 }) {
                if (variant == 3 && depth != meta.block_size) {
                    continue;
                }
                setenv("LLAMA_DSPARK_REUSE_MASK", variant == 0 || variant == 4 ? "0" : "1", 1);
                const bool ids_only = variant == 2 || variant == 4;
                auto *     ctx      = make_ctx(model, 6 + depth);
                auto *     chain    = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_greedy());
                if (ids_only && !llama_set_sampler(ctx, 0, chain)) {
                    fail("sampler offload failed");
                }
                common_speculative * spec = nullptr;
                if (variant == 3) {
                    setenv("LLAMA_DSPARK_GREEDY_IDS", "1", 1);
                    unsetenv("DSPARK_FORWARD_ROWS");
                    unsetenv("DSPARK_CORRECTION_ROWS");
                    common_params_speculative params;
                    params.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
                    params.draft.ctx_dft = ctx;
                    params.draft.ctx_tgt = ctx;
                    params.draft.n_max   = depth;
                    params.draft.n_min   = 0;
                    spec                 = common_speculative_init(params, 1);
                    if (!spec) {
                        fail("common speculative init failed");
                    }
                }
                std::vector<double> elapsed;
                for (int rep = 0; rep < (bench ? 5 : 3); ++rep) {
                    llama_memory_clear(llama_get_memory(ctx), true);
                    std::vector<float> features(6 * width);
                    for (size_t i = 0; i < features.size(); ++i) {
                        features[i] = std::sin(float(i));
                    }
                    std::vector<int32_t> positions = { 0, 1, 2, 3, 4, 5 };
                    std::vector<int32_t> tokens(depth, meta.mask_token_id), draft_pos(depth);
                    tokens[0] = 0;
                    for (int i = 0; i < depth; ++i) {
                        draft_pos[i] = 6 + i;
                    }
                    std::vector<llama_token> ids;
                    std::vector<float>       logits;
                    const auto               begin = std::chrono::steady_clock::now();
                    if (spec) {
                        common_speculative_begin(spec, 0, {});
                        if (!common_speculative_dspark_stage_ctx_test(spec, 0, features.data(), 6, width,
                                                                      positions.data())) {
                            fail("staging failed");
                        }
                        auto & dp   = common_speculative_get_draft_params(spec, 0);
                        dp.drafting = true;
                        dp.n_max    = -1;
                        dp.n_past   = 6;
                        dp.id_last  = tokens[0];
                        dp.result   = &ids;
                        common_speculative_draft(spec);
                        if (ids.size() != size_t(depth)) {
                            fail("common draft length mismatch");
                        }
                    } else {
                        logits = run_forward(ctx, 6, width, features, positions, depth, tokens, draft_pos, meta.n_vocab,
                                             ids_only ? &ids : nullptr);
                    }
                    if (!ids_only && !spec) {
                        for (int i = 0; i < depth; ++i) {
                            const auto row = logits.begin() + i * meta.n_vocab;
                            ids.push_back(std::max_element(row, row + meta.n_vocab) - row);
                        }
                    }
                    const auto end = std::chrono::steady_clock::now();
                    if (bench && rep >= 2) {
                        elapsed.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
                    }
                    if (variant == 0) {
                        expected = logits;
                    }
                    if (variant == 1 && !report_exact_vector("mask reuse logits", expected, logits)) {
                        fail("mask reuse changed logits");
                    }
                    if (variant >= 2) {
                        for (int i = 0; i < depth; ++i) {
                            const auto first = expected.begin() + i * meta.n_vocab;
                            const auto best  = std::max_element(first, first + meta.n_vocab) - first;
                            if (ids[i] != best) {
                                fail("graph greedy ID mismatch");
                            }
                        }
                    }
                    ++checks;
                }
                if (bench) {
                    printf("RUNTIME_BENCH backend=%s variant=%d depth=%d top_k=%d ms=%.6f,%.6f,%.6f\n",
                           backend ? backend : "CPU", variant, depth, top_k, elapsed[0], elapsed[1], elapsed[2]);
                    fflush(stdout);
                }
                if (variant == 1 && depth > 1) {
                    llama_memory_clear(llama_get_memory(ctx), true);
                    std::vector<float>   features(6 * width, 0.0f);
                    std::vector<int32_t> positions = { 0, 1, 2, 3, 4, 5 };
                    std::vector<int32_t> tokens(depth, meta.mask_token_id), draft_pos(depth);
                    tokens[0] = tokens[1] = 0;
                    for (int i = 0; i < depth; ++i) {
                        draft_pos[i] = 6 + i;
                    }
                    bool rejected = false;
                    try {
                        run_forward(ctx, 6, width, features, positions, depth, tokens, draft_pos, meta.n_vocab);
                    } catch (const std::runtime_error & error) {
                        rejected = std::string(error.what()).find("anchor followed by MASK") != std::string::npos;
                    }
                    if (!rejected) {
                        fail("mask reuse accepted non-MASK input");
                    }
                }
                if (spec) {
                    common_speculative_free(spec);
                }
                llama_set_sampler(ctx, 0, nullptr);
                llama_sampler_free(chain);
                llama_free(ctx);
            }
        }
    }
    llama_model_free(model);
    printf("Runtime A/B PASSED: %d forwards, mask reuse and greedy IDs (synthetic taps, backend=%s)\n", checks,
           backend ? backend : "CPU");
    return 0;
}

// Restore caller settings even when a graph validation throws.
struct scoped_test_env {
    std::map<std::string, std::pair<bool, std::string>> saved;

    scoped_test_env(std::initializer_list<const char *> names) {
        for (const char * name : names) {
            const char * value = std::getenv(name);
            saved[name]        = { value != nullptr, value ? value : "" };
        }
    }

    ~scoped_test_env() {
        for (const auto & entry : saved) {
            if (entry.second.first) {
                setenv(entry.first.c_str(), entry.second.second.c_str(), 1);
            } else {
                unsetenv(entry.first.c_str());
            }
        }
    }
};

static int run_prefix_ab(const std::string & model_path, const char * backend = nullptr) {
    llama_dspark_ctx staged;
    staged.n_ctx_rows = 6;
    staged.n_embd_cap = 32;
    llm_graph_params previous{};
    previous.dspark_ctx         = &staged;
    previous.dspark_has_context = true;
    previous.dspark_ctx_rows    = staged.n_ctx_rows;
    previous.dspark_ctx_width   = staged.n_embd_cap;
    auto next                   = previous;
    if (!previous.allow_reuse(next)) {
        fail("identical graph params did not reuse");
    }
    staged.n_ctx_rows    = 7;
    next.dspark_ctx_rows = staged.n_ctx_rows;
    if (previous.allow_reuse(next)) {
        fail("mutable context row count reused stale topology");
    }
    next                    = previous;
    next.dspark_has_context = false;
    if (previous.allow_reuse(next)) {
        fail("context presence change reused stale topology");
    }
    next                  = previous;
    next.dspark_ctx_width = 64;
    if (previous.allow_reuse(next)) {
        fail("context width change reused stale topology");
    }
    printf("Graph reuse shape contracts PASSED\n");
    scoped_test_env environment({ "LLAMA_DSPARK_CORRECTION_PREFIX", "LLAMA_DSPARK_GREEDY_IDS",
                                  "LLAMA_DSPARK_REUSE_MASK", "DSPARK_HEAD_TOP_K", "DSPARK_FORWARD_ROWS",
                                  "DSPARK_CORRECTION_ROWS" });
    unsetenv("LLAMA_DSPARK_CORRECTION_PREFIX");
    unsetenv("DSPARK_FORWARD_ROWS");
    unsetenv("DSPARK_CORRECTION_ROWS");
    setenv("LLAMA_DSPARK_GREEDY_IDS", "0", 1);
    setenv("LLAMA_DSPARK_REUSE_MASK", "0", 1);
    auto mp                      = llama_model_default_params();
    mp.n_gpu_layers              = backend ? 999 : 0;
    ggml_backend_dev_t devices[] = { nullptr, nullptr };
    if (backend) {
        auto registry = ggml_backend_reg_by_name(backend);
        if (!registry || !ggml_backend_reg_dev_count(registry)) {
            fail("requested backend unavailable");
        }
        devices[0] = ggml_backend_reg_dev_get(registry, 0);
        mp.devices = devices;
    }
    auto * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fail("prefix model load failed");
    }
    {
        auto *        ctx     = make_ctx(model, 16);
        const int32_t valid[] = { 0 };
        llama_set_capture_layers(ctx, valid, 1);
        for (const std::vector<int32_t> & invalid : {
                 std::vector<int32_t>{ 0, -1                                 },
                 std::vector<int32_t>{ 0, (int32_t) model->hparams.n_layer() },
                 std::vector<int32_t>(LLAMA_MAX_LAYERS + 1, 0)
        }) {
            bool rejected = false;
            try {
                llama_set_capture_layers(ctx, invalid.data(), invalid.size());
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            if (!rejected || llama_get_n_capture(ctx) != 1) {
                fail("invalid capture list was accepted or changed the previous registration");
            }
        }
        llama_free(ctx);
        printf("Capture registration contracts PASSED: 3 invalid lists rejected atomically\n");
    }
    const auto        meta = read_dspark_meta(model);
    llama_dspark_meta contract;
    if (!llama_model_dspark_get_meta(model, &contract) || !contract.graph_corrected || meta.block_size < 1) {
        fail("prefix A/B requires graph-corrected block drafter");
    }
    const int64_t      width = meta.n_capture * meta.n_embd;
    std::vector<float> features(6 * width);
    for (size_t i = 0; i < features.size(); ++i) {
        features[i] = std::sin(float(i));
    }
    std::vector<int32_t> positions = { 0, 1, 2, 3, 4, 5 };
    std::vector<int32_t> tokens(meta.block_size, meta.mask_token_id), draft_pos(meta.block_size);
    tokens[0] = 0;
    for (int i = 0; i < meta.block_size; ++i) {
        draft_pos[i] = 6 + i;
    }
    const char * top_k_env  = std::getenv("DSPARK_TEST_HEAD_TOP_K");
    const int    restricted = top_k_env ? std::stoi(top_k_env) : std::min<int64_t>(32, meta.n_vocab);
    int          checks     = 0;
    for (int top_k : { 0, restricted }) {
        if (top_k < 0 || top_k > meta.n_vocab) {
            fail("invalid test head top-k");
        }
        setenv("DSPARK_HEAD_TOP_K", std::to_string(top_k).c_str(), 1);
        unsetenv("LLAMA_DSPARK_CORRECTION_PREFIX");
        auto *             baseline = make_ctx(model, 6 + meta.block_size);
        std::vector<float> reference;
        for (int rep = 0; rep < 3; ++rep) {
            llama_memory_clear(llama_get_memory(baseline), true);
            auto values =
                run_forward(baseline, 6, width, features, positions, meta.block_size, tokens, draft_pos, meta.n_vocab);
            if (rep == 0) {
                reference = values;
            } else if (!report_exact_vector("full-block replay", reference, values)) {
                fail("baseline replay changed logits");
            }
        }
        llama_free(baseline);
        for (int keep : { 1, 2, 3, 4, 6, 8 }) {
            if (keep > meta.block_size) {
                continue;
            }
            setenv("LLAMA_DSPARK_CORRECTION_PREFIX", std::to_string(keep).c_str(), 1);
            const std::vector<float> expected(reference.begin(), reference.begin() + size_t(keep) * meta.n_vocab);
            std::vector<llama_token> expected_ids;
            for (int i = 0; i < keep; ++i) {
                const auto row = expected.begin() + i * meta.n_vocab;
                expected_ids.push_back(std::max_element(row, row + meta.n_vocab) - row);
            }
            for (int variant : { 0, 1, 2 }) {
                setenv("LLAMA_DSPARK_GREEDY_IDS", variant == 2 ? "1" : "0", 1);
                auto *               ctx   = make_ctx(model, 6 + meta.block_size);
                llama_sampler *      chain = nullptr;
                common_speculative * spec  = nullptr;
                if (variant == 1) {
                    chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
                    if (!llama_set_sampler(ctx, 0, chain)) {
                        fail("prefix sampler offload failed");
                    }
                }
                if (variant == 2) {
                    common_params_speculative params;
                    params.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
                    params.draft.ctx_dft = ctx;
                    params.draft.ctx_tgt = ctx;
                    params.draft.n_max   = keep;
                    params.draft.n_min   = 0;
                    spec                 = common_speculative_init(params, 1);
                    if (!spec) {
                        fail("prefix common runner init failed");
                    }
                }
                for (int rep = 0; rep < 3; ++rep) {
                    llama_memory_clear(llama_get_memory(ctx), true);
                    std::vector<llama_token> ids;
                    if (spec) {
                        common_speculative_begin(spec, 0, {});
                        if (!common_speculative_dspark_stage_ctx_test(spec, 0, features.data(), 6, width,
                                                                      positions.data())) {
                            fail("prefix common staging failed");
                        }
                        auto & dp   = common_speculative_get_draft_params(spec, 0);
                        dp.drafting = true;
                        dp.n_max    = keep;
                        dp.n_past   = 6;
                        dp.id_last  = tokens[0];
                        dp.result   = &ids;
                        common_speculative_draft(spec);
                    } else {
                        auto values = run_forward(ctx, 6, width, features, positions, meta.block_size, tokens,
                                                  draft_pos, meta.n_vocab, variant == 1 ? &ids : nullptr, keep);
                        if (variant == 0 && !report_exact_vector("full-trunk prefix logits", expected, values)) {
                            fail("prefix changed retained logits");
                        }
                    }
                    if (variant && ids != expected_ids) {
                        fail("prefix changed retained IDs or length");
                    }
                    ++checks;
                }
                if (spec) {
                    common_speculative_free(spec);
                }
                if (chain) {
                    llama_set_sampler(ctx, 0, nullptr);
                    llama_sampler_free(chain);
                }
                llama_free(ctx);
            }
            printf("PREFIX_AB backend=%s full_block=%d keep=%d top_k=%d exact=1\n", backend ? backend : "CPU",
                   meta.block_size, keep, top_k);
        }
    }
    // Invalid contracts must fail before exposing a shortened output as a full block.
    int negative_checks = 0;
    setenv("DSPARK_HEAD_TOP_K", "0", 1);
    setenv("LLAMA_DSPARK_GREEDY_IDS", "0", 1);
    const auto expect_rejection = [&](const std::string & label, const std::function<void()> & action) {
        bool rejected = false;
        try {
            action();
        } catch (const std::runtime_error & error) {
            rejected = true;
            printf("PREFIX_REJECT case=%s reason=%s\n", label.c_str(), error.what());
        }
        if (!rejected) {
            fail("prefix accepted invalid contract: " + label);
        }
        ++negative_checks;
    };
    for (const std::string & invalid : { std::string("0"), std::to_string(meta.block_size + 1), std::string("nope") }) {
        unsetenv("LLAMA_DSPARK_CORRECTION_PREFIX");
        auto * ctx = make_ctx(model, 6 + meta.block_size);
        setenv("LLAMA_DSPARK_CORRECTION_PREFIX", invalid.c_str(), 1);
        expect_rejection("invalid-" + invalid, [&] {
            run_forward(ctx, 6, width, features, positions, meta.block_size, tokens, draft_pos, meta.n_vocab);
        });
        llama_free(ctx);
    }
    const int  keep      = std::min<int>(4, meta.block_size);
    const auto make_spec = [&](llama_context * ctx, int n_max, int n_seq) {
        common_params_speculative params;
        params.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
        params.draft.ctx_dft = ctx;
        params.draft.ctx_tgt = ctx;
        params.draft.n_max   = n_max;
        params.draft.n_min   = 0;
        return common_speculative_init(params, n_seq);
    };
    for (const std::string & mode : { std::string("n-max-mismatch"), std::string("multiple-sequences") }) {
        unsetenv("LLAMA_DSPARK_CORRECTION_PREFIX");
        auto * ctx = make_ctx(model, 6 + meta.block_size);
        setenv("LLAMA_DSPARK_CORRECTION_PREFIX", std::to_string(keep).c_str(), 1);
        expect_rejection(mode, [&] {
            auto * spec =
                make_spec(ctx, mode == "n-max-mismatch" ? keep + 1 : keep, mode == "multiple-sequences" ? 2 : 1);
            if (spec) {
                common_speculative_free(spec);
            }
        });
        llama_free(ctx);
    }
    for (const std::string & mode : { std::string("changed-after-init"), std::string("unset-after-init") }) {
        unsetenv("LLAMA_DSPARK_CORRECTION_PREFIX");
        auto * ctx = make_ctx(model, 6 + meta.block_size);
        setenv("LLAMA_DSPARK_CORRECTION_PREFIX", std::to_string(keep).c_str(), 1);
        auto * spec = make_spec(ctx, keep, 1);
        if (!spec) {
            fail("valid prefix initialization failed in negative test");
        }
        common_speculative_begin(spec, 0, {});
        if (!common_speculative_dspark_stage_ctx_test(spec, 0, features.data(), 6, width, positions.data())) {
            fail("negative prefix staging failed");
        }
        std::vector<llama_token> ids;
        auto &                   dp = common_speculative_get_draft_params(spec, 0);
        dp.drafting                 = true;
        dp.n_max                    = keep;
        dp.n_past                   = 6;
        dp.id_last                  = tokens[0];
        dp.result                   = &ids;
        if (mode == "changed-after-init") {
            setenv("LLAMA_DSPARK_CORRECTION_PREFIX", std::to_string(keep == 1 ? 2 : 1).c_str(), 1);
        } else {
            unsetenv("LLAMA_DSPARK_CORRECTION_PREFIX");
        }
        expect_rejection(mode, [&] { common_speculative_draft(spec); });
        common_speculative_free(spec);
        llama_free(ctx);
    }
    if (keep < meta.block_size) {
        unsetenv("LLAMA_DSPARK_CORRECTION_PREFIX");
        auto * ctx = make_ctx(model, 6 + meta.block_size);
        setenv("LLAMA_DSPARK_CORRECTION_PREFIX", std::to_string(keep).c_str(), 1);
        expect_rejection("full-output-mask-with-prefix", [&] {
            run_forward(ctx, 6, width, features, positions, meta.block_size, tokens, draft_pos, meta.n_vocab);
        });
        llama_free(ctx);
    }
    printf("Prefix negative contracts PASSED: %d rejected cases\n", negative_checks);
    llama_model_free(model);
    printf("Prefix A/B PASSED: %d forwards, full-trunk logits and IDs exact\n", checks);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <model.gguf> --tier1 | --tier1-metal | --snr-ab-cpu | --snr-ab-metal | --output-ab-cpu | "
                "--output-ab-metal | --runtime-ab-cpu | --runtime-ab-metal | --runtime-ab-cuda | "
                "--runtime-bench-metal | --runtime-bench-cuda | --prefix-ab-cpu | --prefix-ab-metal | --prefix-ab-cuda "
                "| --tier2 <ref.bin>\n",
                argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();

    const std::string model_path = argv[1];
    const std::string mode       = argv[2];

    int rc;
    if (mode == "--prefix-ab-cpu" || mode == "--prefix-ab-metal" || mode == "--prefix-ab-cuda") {
        rc = run_prefix_ab(model_path,
                           mode == "--prefix-ab-cpu" ? nullptr : (mode == "--prefix-ab-metal" ? "MTL" : "CUDA"));
    } else if (mode == "--runtime-ab-cpu") {
        rc = run_runtime_ab(model_path);
    } else if (mode == "--runtime-ab-metal" || mode == "--runtime-ab-cuda" || mode == "--runtime-bench-metal" ||
               mode == "--runtime-bench-cuda") {
        rc = run_runtime_ab(model_path, mode.find("metal") != std::string::npos ? "MTL" : "CUDA",
                            mode.find("bench") != std::string::npos);
    } else if (mode == "--output-ab-cpu" || mode == "--output-ab-metal") {
        rc = run_output_ab(model_path, mode == "--output-ab-metal");
    } else if (mode == "--snr-ab-cpu" || mode == "--snr-ab-metal") {
        rc = run_snr_ab(model_path, mode == "--snr-ab-metal");
    } else if (mode == "--tier1") {
        rc = run_tier1(model_path);
    } else if (mode == "--tier1-metal") {
        std::vector<float> cpu, metal;
        run_tier1(model_path, false, &cpu);
        const auto cpu_trace = trace_values;
        trace_values.clear();
        rc = run_tier1(model_path, true, &metal);
        for (const auto & item : cpu_trace) {
            const auto & actual = trace_values.at(item.first);
            if (actual.size() != item.second.size()) {
                fail("trace shape mismatch");
            }
            float error = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                error = std::max(error, std::fabs(actual[i] - item.second[i]));
            }
            printf("trace %s count=%zu max_abs_error=%g\n", item.first.c_str(), actual.size(), error);
        }
        if (cpu.size() != metal.size()) {
            fail("CPU/Metal output shapes differ");
        }
        const size_t slots =
            cpu_trace.count("dspark_corr_masked-0") ? cpu.size() / cpu_trace.at("dspark_corr_masked-0").size() : 0;
        if (slots) {
            const size_t vocab   = cpu.size() / slots;
            size_t       matches = 0;
            for (size_t slot = 0; slot < slots; ++slot) {
                auto a = cpu.begin() + slot * vocab;
                auto b = metal.begin() + slot * vocab;
                matches += std::max_element(a, a + vocab) - a == std::max_element(b, b + vocab) - b;
            }
            printf("CPU/Metal draft argmax matches: %zu/%zu (separate from full-logit tolerance)\n", matches, slots);
        }
        float max_error = 0.0f;
        for (size_t i = 0; i < cpu.size(); ++i) {
            const float error = std::fabs(cpu[i] - metal[i]);
            max_error         = std::max(max_error, error);
            if (error > 1e-3f + 1e-2f * std::fabs(cpu[i])) {
                fail("CPU/Metal logit tolerance exceeded (atol=1e-3, rtol=1e-2)");
            }
        }
        printf("CPU/Metal full-logit comparison PASSED: count=%zu max_abs_error=%g\n", cpu.size(), max_error);
    } else if (mode == "--tier2") {
        if (argc < 4) {
            fail("--tier2 requires a ref.bin path");
        }
        rc = run_tier2(model_path, argv[3]);
    } else {
        fail("unknown mode: " + mode);
        rc = 1;
    }

    llama_backend_free();
    return rc;
}

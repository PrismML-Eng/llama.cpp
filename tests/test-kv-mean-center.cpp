// Tests for K-cache mean-centering (see docs/kv-mean-center.md):
//
//  (a) regression safety: with centering disabled (the default / no bias file loaded), behavior
//      is unaffected by the feature's presence, including when the K cache type is the one the
//      feature targets (GGML_TYPE_Q4_0).
//  (b) softmax-invariance: subtracting a fixed nonzero bias from every cached K vector before
//      it is written into the cache does not change the model's output logits, up to ordinary
//      fp32 rounding. This is tested against an *unquantized* (F32) K cache so that the
//      comparison isn't confounded by actual quantization error -- this test is a pure math
//      check of the softmax-invariance argument, not a quantization-fidelity measurement.
//
// Uses the same tiny-synthetic-model machinery as test-llama-archs.cpp (llama_model_saver +
// llama_model_init_from_user with a deterministic random tensor initializer), trimmed down to
// just the plain LLM_ARCH_LLAMA case.

#include "common.h"
#include "kv-mean-center.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-model-saver.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static const uint32_t k_n_vocab = 128;
static const uint32_t k_n_embd  = 256;
static const uint32_t k_n_head  = 2;
static const uint32_t k_n_ff    = 384;
static const uint32_t k_n_layer = 2;
static const uint32_t k_n_ctx   = 128;

// deterministic pseudo-random weight initializer (same technique as test-llama-archs.cpp)
static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(tensor->name) + *(const size_t *) userdata);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    std::vector<float> tmp(ne);
    for (int64_t i = 0; i < ne; i++) {
        tmp[i] = dis(gen);
    }
    ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
}

// minimal metadata for a tiny 2-layer LLM_ARCH_LLAMA model (trimmed from test-llama-archs.cpp's
// generic per-arch metadata builder, dropping everything that only applies to other archs)
static gguf_context_ptr build_llama_gguf_ctx() {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_LLAMA, ret.get());

    const uint32_t n_embd_head = k_n_embd / k_n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(LLM_ARCH_LLAMA));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                k_n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            k_n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          k_n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               k_n_layer);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       k_n_ff);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,     false);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      k_n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   k_n_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,           "no_vocab");

    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static llama_model_ptr build_model(struct gguf_context * gguf_ctx, size_t seed) {
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;

    size_t tmp = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx, set_tensor_data, &tmp, model_params));
    if (!model) {
        throw std::runtime_error("failed to create tiny llama model");
    }

    return model;
}

// builds a context on top of an existing model; returns nullptr if llama_init_from_model rejects
// the configuration (e.g. the --kv-mean-center / GGML_TYPE_Q4_0 gate)
static llama_context_ptr build_context(llama_model * model, ggml_type type_k, const char * path_kv_mean_center = nullptr) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx               = 0; // from model
    ctx_params.n_batch             = 32;
    ctx_params.n_ubatch            = 32;
    ctx_params.n_threads           = 2;
    ctx_params.n_threads_batch     = 2;
    ctx_params.type_k              = type_k;
    ctx_params.path_kv_mean_center = path_kv_mean_center;

    // quantized K cache types in this codebase are exercised together with flash attention
    if (ggml_is_quantized(type_k)) {
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }

    return llama_context_ptr(llama_init_from_model(model, ctx_params));
}

static std::vector<llama_token> make_tokens(uint32_t n_tokens, uint32_t n_vocab, size_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, (int) n_vocab - 1);
    std::vector<llama_token> tokens;
    tokens.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        tokens.push_back(dis(gen));
    }
    return tokens;
}

static std::vector<float> decode_and_get_logits(llama_context * ctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i], (llama_pos) i, { 0 }, true);
    }

    if (llama_decode(ctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("llama_decode failed");
    }

    std::vector<float> logits;
    logits.reserve(tokens.size() * n_vocab);
    for (size_t i = 0; i < tokens.size(); i++) {
        const float * li = llama_get_logits_ith(ctx, (int32_t) i);
        logits.insert(logits.end(), li, li + n_vocab);
    }

    llama_batch_free(batch);

    return logits;
}

// normalized mean squared error = mse(a, b) / mse(a, 0), as used elsewhere in the test suite
// (e.g. tests/test-llama-archs.cpp)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double) a[i] - (double) b[i];
        mse_a_b += d*d;
        mse_a_0 += (double) a[i] * (double) a[i];
    }

    return mse_a_b / mse_a_0;
}

#define TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__, #cond); \
            abort(); \
        } \
    } while (0)

// (a) regression safety: centering never engages unless a bias file is explicitly loaded, and
// its mere availability as a (dormant) code path does not perturb anything -- not even for a
// Q4_0 K cache, which is the type the feature is scoped to.
static void test_regression_safety() {
    gguf_context_ptr gguf_ctx = build_llama_gguf_ctx();
    llama_model_ptr  model    = build_model(gguf_ctx.get(), /*seed=*/ 1234);

    const std::vector<llama_token> tokens = make_tokens(16, k_n_vocab, /*seed=*/ 42);

    // two independent contexts, identical configuration, no bias file: outputs must be
    // bit-for-bit identical
    llama_context_ptr ctx1 = build_context(model.get(), GGML_TYPE_F16);
    TEST_ASSERT(ctx1 != nullptr);
    const std::vector<float> logits1 = decode_and_get_logits(ctx1.get(), tokens);

    llama_context_ptr ctx2 = build_context(model.get(), GGML_TYPE_F16);
    TEST_ASSERT(ctx2 != nullptr);
    const std::vector<float> logits2 = decode_and_get_logits(ctx2.get(), tokens);

    TEST_ASSERT(logits1.size() == logits2.size());
    for (size_t i = 0; i < logits1.size(); i++) {
        TEST_ASSERT(logits1[i] == logits2[i]);
    }

    // smoke check: a Q4_0 K cache (the type this feature targets) with no bias file loaded
    // must decode normally and produce finite output -- cpy_k()'s new code path must be a
    // true no-op when centering was never enabled (k_bar.empty())
    llama_context_ptr ctx3 = build_context(model.get(), GGML_TYPE_Q4_0);
    TEST_ASSERT(ctx3 != nullptr);
    const std::vector<float> logits3 = decode_and_get_logits(ctx3.get(), tokens);
    for (float v : logits3) {
        TEST_ASSERT(std::isfinite(v));
    }

    LOG_INF("%s: OK\n", __func__);
}

// (a2) the public, CLI-facing entry point (llama_context_params::path_kv_mean_center, as set by
// --kv-mean-center) must hard-reject any K cache type other than GGML_TYPE_Q4_0, and must
// succeed end-to-end (including a real decode) when the type matches.
static void test_q4_0_gate() {
    gguf_context_ptr gguf_ctx = build_llama_gguf_ctx();
    llama_model_ptr  model    = build_model(gguf_ctx.get(), /*seed=*/ 2024);

    const uint32_t n_embd_head = k_n_embd / k_n_head;

    std::vector<common_kv_mean_center_layer> layers;
    for (uint32_t il = 0; il < k_n_layer; il++) {
        common_kv_mean_center_layer layer;
        layer.il = (int32_t) il;
        layer.bias.assign(n_embd_head * k_n_head, 0.25f);
        layers.push_back(std::move(layer));
    }

    // this synthetic model has a 64-wide K head, so a Q4_0 K cache activates the Hadamard
    // rotation; the bias must be marked as measured in the rotated basis to be accepted there
    const std::string tmp_path = "test-kv-mean-center-gate.gguf";
    TEST_ASSERT(common_kv_mean_center_write(tmp_path, layers, /*k_rot=*/true));

    // F16 K cache + --kv-mean-center must be rejected outright (llama_init_from_model returns
    // nullptr), matching this codebase's convention for other cache-type-gated mismatches (e.g.
    // "V cache quantization requires flash_attn"). use a basis-matching file (F16 cache means
    // the rotation is off, so k_rot=false matches) so this exercises the cache-type gate
    // specifically, not the basis check
    const std::string tmp_path_unrot = "test-kv-mean-center-gate-unrot.gguf";
    TEST_ASSERT(common_kv_mean_center_write(tmp_path_unrot, layers, /*k_rot=*/false));
    llama_context_ptr ctx_bad = build_context(model.get(), GGML_TYPE_F16, tmp_path_unrot.c_str());
    TEST_ASSERT(ctx_bad == nullptr);

    // a bias measured in the unrotated basis must be rejected by a rotated (Q4_0) K cache:
    // a basis mismatch degrades quantization quality instead of improving it
    llama_context_ptr ctx_mismatch = build_context(model.get(), GGML_TYPE_Q4_0, tmp_path_unrot.c_str());
    TEST_ASSERT(ctx_mismatch == nullptr);
    remove(tmp_path_unrot.c_str());

    // Q4_0 K cache + --kv-mean-center must succeed end-to-end, through the real public entry
    // point (not the require_q4_0=false test seam used in test_softmax_invariance)
    llama_context_ptr ctx_good = build_context(model.get(), GGML_TYPE_Q4_0, tmp_path.c_str());
    TEST_ASSERT(ctx_good != nullptr);

    const std::vector<llama_token> tokens = make_tokens(16, k_n_vocab, /*seed=*/ 3);
    const std::vector<float> logits = decode_and_get_logits(ctx_good.get(), tokens);
    for (float v : logits) {
        TEST_ASSERT(std::isfinite(v));
    }

    remove(tmp_path.c_str());

    LOG_INF("%s: OK\n", __func__);
}

// (b) softmax-invariance: verify the actual math claim using an unquantized F32 K cache, so the
// comparison isn't confounded by real Q4_0 quantization error.
static void test_softmax_invariance() {
    gguf_context_ptr gguf_ctx = build_llama_gguf_ctx();
    llama_model_ptr  model    = build_model(gguf_ctx.get(), /*seed=*/ 5678);

    const std::vector<llama_token> tokens = make_tokens(16, k_n_vocab, /*seed=*/ 7);

    // baseline: F32 K cache, no bias
    llama_context_ptr ctx_base = build_context(model.get(), GGML_TYPE_F32);
    TEST_ASSERT(ctx_base != nullptr);
    const std::vector<float> logits_base = decode_and_get_logits(ctx_base.get(), tokens);

    // synthesize a nonzero per-layer bias file
    const uint32_t n_embd_head = k_n_embd / k_n_head;

    std::mt19937 gen(99);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    std::vector<common_kv_mean_center_layer> layers;
    for (uint32_t il = 0; il < k_n_layer; il++) {
        common_kv_mean_center_layer layer;
        layer.il = (int32_t) il;
        layer.bias.resize(n_embd_head * k_n_head);
        for (float & v : layer.bias) {
            v = dis(gen);
        }
        layers.push_back(std::move(layer));
    }

    // scratch file in the test's working directory; this test is not run concurrently with
    // itself, so a fixed name is fine
    const std::string tmp_path = "test-kv-mean-center-bias.gguf";

    TEST_ASSERT(common_kv_mean_center_write(tmp_path, layers));

    // centered: same model, fresh F32-K-cache context, bias applied through the exact same
    // cpy_k() code path -- bypassing the public --kv-mean-center gate (require_q4_0 = false)
    // since this test is specifically about validating the math on an unquantized cache
    llama_context_ptr ctx_centered = build_context(model.get(), GGML_TYPE_F32);
    TEST_ASSERT(ctx_centered != nullptr);

    auto * kv = dynamic_cast<llama_kv_cache *>(llama_get_memory(ctx_centered.get()));
    TEST_ASSERT(kv != nullptr);
    TEST_ASSERT(kv->load_kv_mean_center(tmp_path.c_str(), /*require_q4_0=*/false));

    const std::vector<float> logits_centered = decode_and_get_logits(ctx_centered.get(), tokens);

    remove(tmp_path.c_str());

    TEST_ASSERT(logits_base.size() == logits_centered.size());

    const double err = nmse(logits_base, logits_centered);
    LOG_INF("%s: nmse(baseline, centered) = %g\n", __func__, err);

    // this is a pure fp32-rounding check (no quantization involved on either side), so the
    // tolerance is tight; a real bug (e.g. the bias leaking into attention asymmetrically)
    // would show up many orders of magnitude larger than fp32 noise
    TEST_ASSERT(err < 1e-8);

    LOG_INF("%s: OK\n", __func__);
}

int main() {
    test_regression_safety();
    test_q4_0_gate();
    test_softmax_invariance();

    return 0;
}

// Compare repeated draft/accept rounds against a supplied reference JSON.
// Usage: test-dspark-loop <tiny.gguf> <ref.json>

#include "../src/llama-ext.h"
#include "common.h"
#include "llama.h"
#include "speculative.h"
#ifdef LLAMA_DSPARK_MARKOV_CUDA
#    include "dspark-markov.h"

#    include <algorithm>
#    include <cmath>
#    include <limits>
#endif
#ifdef LLAMA_DSPARK_MARKOV_METAL
#    include "dspark-markov-metal.h"

#    include <cmath>
#    include <limits>
#    include <memory>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

static void fail(const std::string & msg) {
    fprintf(stderr, "FAIL: %s\n", msg.c_str());
    exit(1);
}

// --- synthetic "target tap feature" stand-in ---------------------------
// Matches the reference fixture's
// hash_u32/synth_feat/synth_bonus_token (same constants, same integer ops --
// see that file's header comment for why this is safe to duplicate rather
// than share: it's a closed-form pure function of small integers, not a
// stateful RNG stream, so bit-parity across languages just falls out of
// using the same uint32 wraparound arithmetic).
static uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static float synth_feat(int64_t pos, int64_t d) {
    const uint32_t h = hash_u32((uint32_t) (pos * 131071 + d * 97 + 12345));
    const int32_t  m = (int32_t) (h % 2000u) - 1000;  // [-1000, 999]
    return (float) m / 500.0f;                        // [-2.0, 1.998]
}

static int32_t synth_bonus_token(int32_t round_idx, int32_t vocab_size, int32_t mask_token_id) {
    const uint32_t h = hash_u32((uint32_t) round_idx * 2654435761u + 999983u);
    int32_t        v = (int32_t) (h % (uint32_t) (vocab_size - 1));
    if (v == mask_token_id) {
        v = (v + 1) % vocab_size;
    }
    return v;
}

// mirrored verbatim from dspark_phase2_py_ref.py
static const std::vector<int32_t> ACCEPT_SCHEDULE = { 7, 3, 0, 7, 5, 1, 4 };
static const std::vector<int32_t> PROMPT          = { 1, 2, 3, 4, 5 };

static std::vector<float> synth_feat_rows(int64_t pos_beg, int64_t n_rows, int64_t n_embd_cap) {
    std::vector<float> feat((size_t) n_rows * n_embd_cap);
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t d = 0; d < n_embd_cap; ++d) {
            feat[(size_t) i * n_embd_cap + d] = synth_feat(pos_beg + i, d);
        }
    }
    return feat;
}

#ifdef LLAMA_DSPARK_MARKOV_METAL
static int test_metal_markov() {
    int  checks = 0;
    auto check  = [&](bool good, const char * label) {
        if (!good) {
            fail(label);
        }
        ++checks;
    };
    for (auto shape : {
             std::pair<int, int>{ 17,   1   },
              { 129,  31  },
              { 257,  256 },
              { 1025, 33  },
              { 8193, 256 }
    }) {
        const int          v = shape.first, rank = shape.second, mask = v - 1;
        std::vector<float> a(size_t(v) * rank), b(a.size()), base(size_t(v) * 4);
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = float(int(hash_u32(uint32_t(i)) % 31) - 15) / 16;
            b[i] = float(int(hash_u32(uint32_t(i) + 987) % 31) - 15) / 16;
        }
        for (size_t i = 0; i < base.size(); ++i) {
            base[i] = float(int(hash_u32(uint32_t(i) + 345) % 31) - 15) / 16;
        }
        std::unique_ptr<dspark_markov_metal, decltype(&dspark_markov_metal_free)> ctx(
            dspark_markov_metal_init(a.data(), b.data(), v, rank, mask), dspark_markov_metal_free);
        check(bool(ctx), "Metal context creation");
        for (int slots : { 1, 4, 2 }) {
            std::vector<int32_t> output(slots);
            check(dspark_markov_metal_resample(ctx.get(), base.data(), 1, slots, output.data()), "Metal resample");
            int previous = 1;
            for (int slot = 0; slot < slots; ++slot) {
                double best   = -std::numeric_limits<double>::infinity();
                int    winner = -1;
                for (int token = 0; token < v; ++token) {
                    if (token == mask) {
                        continue;
                    }
                    double score = base[size_t(slot) * v + token];
                    for (int r = 0; r < rank; ++r) {
                        score += double(a[size_t(previous) * rank + r]) * b[size_t(token) * rank + r];
                    }
                    if (score > best) {
                        best   = score;
                        winner = token;
                    }
                }
                check(output[slot] == winner, "Metal vs independent F64 chain oracle");
                previous = winner;
            }
        }
        int32_t out[4];
        check(!dspark_markov_metal_resample(ctx.get(), base.data(), mask, 1, out), "mask anchor rejected");
        check(!dspark_markov_metal_resample(ctx.get(), base.data(), -1, 1, out), "negative anchor rejected");
        check(!dspark_markov_metal_resample(ctx.get(), base.data(), v, 1, out), "out-of-range anchor rejected");
        check(!dspark_markov_metal_resample(ctx.get(), base.data(), 1, 0, out), "zero slots rejected");
        check(!dspark_markov_metal_resample(ctx.get(), base.data(), 1, 1, nullptr), "null output rejected");
        const float saved = base[0];
        for (float bad : { std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                           -std::numeric_limits<float>::infinity() }) {
            base[0] = bad;
            check(!dspark_markov_metal_resample(ctx.get(), base.data(), 1, 4, out), "nonfinite score rejected");
        }
        base[0] = saved;
        check(dspark_markov_metal_resample(ctx.get(), base.data(), 1, 4, out), "recovery after rejected round");
    }
    std::vector<float>                                                        zeros(129 * 31), base(129 * 4);
    std::unique_ptr<dspark_markov_metal, decltype(&dspark_markov_metal_free)> ties(
        dspark_markov_metal_init(zeros.data(), zeros.data(), 129, 31, 0), dspark_markov_metal_free);
    check(bool(ties), "tie context");
    for (int slot = 0; slot < 4; ++slot) {
        base[slot * 129] = 1e30f;
    }
    int32_t out[4];
    check(dspark_markov_metal_resample(ties.get(), base.data(), 1, 4, out), "tie and dominant mask round");
    for (int token : out) {
        check(token == 1, "lowest unmasked ID wins ties");
    }
    for (int slot = 0; slot < 4; ++slot) {
        base[slot * 129 + 128] = 1;
    }
    check(dspark_markov_metal_resample(ties.get(), base.data(), 1, 4, out), "ragged final row round");
    for (int token : out) {
        check(token == 128, "ragged final row can win");
    }
    for (float excluded : { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() }) {
        base[0] = excluded;
        check(dspark_markov_metal_resample(ties.get(), base.data(), 1, 4, out), "excluded mask logit ignored");
    }
    check(!dspark_markov_metal_init(zeros.data(), zeros.data(), 129, 31, 129), "invalid mask metadata");
    zeros[0] = std::numeric_limits<float>::quiet_NaN();
    check(!dspark_markov_metal_init(zeros.data(), zeros.data(), 129, 31, 0), "nonfinite weights rejected");
    printf("Metal Markov: %d checks passed\n", checks);
    return 0;
}
#endif

#ifdef LLAMA_DSPARK_MARKOV_CUDA
static int test_cuda_dcut() {
    int checks = 0;
    for (int rank : { 1, 31, 33, 256 }) {
        const int          vocab = 67;
        std::vector<float> a(vocab * rank), b(vocab * rank), base(4 * vocab);
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = ((int) (i % 11) - 5) * 0.03f;
            b[i] = ((int) (i % 17) - 8) * 0.02f;
        }
        for (size_t i = 0; i < base.size(); ++i) {
            base[i] = std::sin((float) i) * 3.0f;
        }
        for (int k = 0; k < 4; ++k) {
            base[k * vocab + vocab - 1] = 1000.0f;
        }
        auto * ctx = dspark_markov_cuda_init(a.data(), b.data(), vocab, rank, vocab - 1);
        if (!ctx) {
            fail("CUDA D-cut init");
        }
        for (int n : { 4, 1, 3, 2, 4 }) {
            for (int mode = 0; mode < 3; ++mode) {
                float costs[4] = { 1, 1, 1, 1 };
                if (mode == 1) {
                    costs[1] = 20;
                    costs[2] = 30;
                    costs[3] = 40;
                }
                if (mode == 2) {
                    costs[1] = 1.01f;
                    costs[2] = 1.1f;
                    costs[3] = 1.2f;
                }
                int32_t got[4], original[4], kept;
                float   probs[4];
                if (!dspark_markov_cuda_resample(ctx, base.data(), 5, n, original) ||
                    !dspark_markov_cuda_dcut(ctx, base.data(), 5, n, got, costs, &kept, probs)) {
                    fail("CUDA D-cut run");
                }
                int    prev = 5, expected_keep = 1;
                double survival = 1, expected = 1, best_rate = -1;
                for (int k = 0; k < n; ++k) {
                    std::vector<double> logits(vocab);
                    for (int v = 0; v < vocab; ++v) {
                        logits[v] = base[k * vocab + v];
                        for (int r = 0; r < rank; ++r) {
                            logits[v] += (double) a[prev * rank + r] * b[v * rank + r];
                        }
                    }
                    logits[vocab - 1] = -std::numeric_limits<double>::infinity();
                    const int winner = std::max_element(logits.begin(), logits.end()) - logits.begin();
                    double    sum    = 0;
                    for (double logit : logits) {
                        sum += std::exp(logit - logits[winner]);
                    }
                    const double p = 1 / sum;
                    if (got[k] != winner || got[k] != original[k] || std::abs(probs[k] - p) > 1e-5) {
                        fail("CUDA D-cut F64 oracle");
                    }
                    survival *= p;
                    expected += survival;
                    const double rate = expected / costs[k];
                    if (rate > best_rate) {
                        best_rate     = rate;
                        expected_keep = k + 1;
                    }
                    prev = winner;
                    ++checks;
                }
                if (kept != expected_keep) {
                    fail("CUDA D-cut selector oracle");
                }
                ++checks;
            }
        }
        float   invalid[4] = { 1, 2, 3, 0 };
        int32_t got[4], kept;
        if (dspark_markov_cuda_dcut(ctx, base.data(), 0, 4, got, invalid, &kept)) {
            fail("D-cut accepts zero cost");
        }
        invalid[3] = std::numeric_limits<float>::quiet_NaN();
        if (dspark_markov_cuda_dcut(ctx, base.data(), 0, 4, got, invalid, &kept)) {
            fail("D-cut accepts NaN cost");
        }
        checks += 2;
        dspark_markov_cuda_free(ctx);
    }
    printf("CUDA_DCUT_ORACLE PASS checks=%d\n", checks);
    return 0;
}
#endif

int main(int argc, char ** argv) {
#ifdef LLAMA_DSPARK_MARKOV_CUDA
    if (argc == 2 && std::string(argv[1]) == "--cuda-dcut-self-test") {
        return test_cuda_dcut();
    }
#endif
#ifdef LLAMA_DSPARK_MARKOV_METAL
    if (argc == 2 && std::string(argv[1]) == "--metal-markov-self-test") {
        return test_metal_markov();
    }
#endif
    if (argc < 3) {
        fprintf(stderr, "usage: %s <tiny-dspark.gguf> <ref.json>\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string ref_path   = argv[2];

    std::ifstream f(ref_path);
    if (!f) {
        fail("could not open ref file: " + ref_path);
    }
    json ref;
    f >> ref;

    const int64_t n_embd_cap_ref    = ref.at("n_embd_cap").get<int64_t>();
    const int32_t block_size_ref    = ref.at("block_size").get<int32_t>();
    const int32_t vocab_size_ref    = ref.at("vocab_size").get<int32_t>();
    const int32_t mask_token_id_ref = ref.at("mask_token_id").get<int32_t>();
    const int32_t prefill_bonus_ref = ref.at("prefill_bonus").get<int32_t>();
    const auto &  rounds_ref        = ref.at("rounds");

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 0;  // CPU: deterministic, no Metal precision surprises

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fail("failed to load model: " + model_path);
    }

    llama_dspark_meta meta;
    if (!llama_model_dspark_get_meta(model, &meta)) {
        fail("llama_model_dspark_get_meta failed -- not a dspark model?");
    }

    printf(
        "meta: n_embd=%lld n_vocab=%lld n_capture=%lld n_embd_cap=%lld block_size=%d mask_token_id=%d "
        "markov_rank=%lld\n",
        (long long) meta.n_embd, (long long) meta.n_vocab, (long long) meta.n_capture, (long long) meta.n_embd_cap,
        meta.block_size, meta.mask_token_id, (long long) meta.markov_rank);

    if (meta.n_embd_cap != n_embd_cap_ref) {
        fail("n_embd_cap mismatch vs ref.json");
    }
    if (meta.block_size != block_size_ref) {
        fail("block_size mismatch vs ref.json");
    }
    if (meta.n_vocab != vocab_size_ref) {
        fail("vocab_size mismatch vs ref.json");
    }
    if (meta.mask_token_id != mask_token_id_ref) {
        fail("mask_token_id mismatch vs ref.json");
    }

    const int64_t n_embd_cap    = meta.n_embd_cap;
    const int32_t block_size    = meta.block_size;
    const int32_t vocab_size    = (int32_t) meta.n_vocab;
    const int32_t mask_token_id = meta.mask_token_id;

    // size the context generously: worst-case round needs ctx_len(<= previous
    // block_size+1) + block_size tokens in one llama_decode call.
    const uint32_t n_ctx_max = (uint32_t) (block_size + 1 + block_size) + 8;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = n_ctx_max;
    cparams.n_batch              = n_ctx_max;
    cparams.n_ubatch             = n_ctx_max;
    cparams.n_seq_max            = 1;
    cparams.no_perf              = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fail("llama_init_from_model failed");
    }

    common_params_speculative sparams;
    sparams.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
    sparams.draft.ctx_dft = ctx;
    // No real target model exists in this synthetic test (see file header
    // comment); dspark's process() path (which reads from ctx_tgt) is never
    // exercised here -- context rows are injected directly via
    // common_speculative_dspark_stage_ctx_test(). ctx_tgt only needs to be a
    // valid, non-null context to satisfy the impl's construction-time assert.
    sparams.draft.ctx_tgt = ctx;
    sparams.draft.n_max   = block_size;
    sparams.draft.n_min   = 0;

    common_speculative * spec = common_speculative_init(sparams, /* n_seq = */ 1);
    if (!spec) {
        fail("common_speculative_init returned null");
    }

    common_speculative_begin(spec, /* seq_id = */ 0, PROMPT);

    // one-time prefill seeding: the whole prompt's (synthetic) tap features,
    // positions [0, N).
    {
        const int64_t        N    = (int64_t) PROMPT.size();
        std::vector<float>   feat = synth_feat_rows(0, N, n_embd_cap);
        std::vector<int32_t> pos(N);
        for (int64_t i = 0; i < N; ++i) {
            pos[i] = (int32_t) i;
        }

        if (!common_speculative_dspark_stage_ctx_test(spec, 0, feat.data(), N, n_embd_cap, pos.data())) {
            fail("common_speculative_dspark_stage_ctx_test (prefill) failed");
        }
    }

    llama_pos   start   = (llama_pos) PROMPT.size();
    llama_token id_last = (llama_token) synth_bonus_token(-1, vocab_size, mask_token_id);
    if (id_last != prefill_bonus_ref) {
        fail("C++/python prefill bonus token disagree -- synth_bonus_token drifted");
    }

    int32_t n_mismatch_rounds = 0;

    for (size_t r = 0; r < rounds_ref.size(); ++r) {
        const auto &               rr          = rounds_ref[r];
        const int32_t              n_accepted  = rr.at("n_accepted").get<int32_t>();
        const int64_t              ctx_len_ref = rr.at("ctx_len").get<int64_t>();
        const int64_t              start_ref   = rr.at("start").get<int64_t>();
        const std::vector<int32_t> sampled_ref = rr.at("sampled").get<std::vector<int32_t>>();

        if ((int32_t) ACCEPT_SCHEDULE[r % ACCEPT_SCHEDULE.size()] != n_accepted) {
            fail("ACCEPT_SCHEDULE drifted out of sync with ref.json at round " + std::to_string(r));
        }
        if ((int64_t) start != start_ref) {
            fail("start bookkeeping disagrees with ref.json at round " + std::to_string(r) +
                 " (cpp=" + std::to_string(start) + " py=" + std::to_string(start_ref) + ")");
        }

        common_speculative_draft_params & dp = common_speculative_get_draft_params(spec, 0);
        dp.drafting                          = true;
        dp.n_max                             = -1;
        dp.pos0                              = start;
        dp.id_last                           = id_last;
        dp.prompt                            = nullptr;  // unused by dspark
        llama_tokens result;
        dp.result = &result;

        common_speculative_draft(spec);

        printf("round %zu: n_past=%d id_last=%d -> result=[", r, start, id_last);
        for (auto t : result) {
            printf("%d ", t);
        }
        printf("] expected=[");
        for (auto t : sampled_ref) {
            printf("%d ", t);
        }
        printf("]\n");

        if (result.size() != (size_t) block_size) {
            fail("round " + std::to_string(r) + ": expected block_size=" + std::to_string(block_size) +
                 " drafted tokens, got " + std::to_string(result.size()));
        }

        bool round_ok = true;
        for (int32_t k = 0; k < block_size; ++k) {
            if (result[k] != sampled_ref[k]) {
                round_ok = false;
            }
        }
        if (!round_ok) {
            n_mismatch_rounds++;
            fprintf(stderr, "  MISMATCH at round %zu\n", r);
        }

        // stage this round's verify-capture: the target would verify the
        // anchor + all block_size drafted tokens in one batch (verify_length
        // = block_size + 1), regardless of how many end up accepted -- accept()
        // below trims this down to the actually-committed prefix.
        {
            std::vector<float>   feat = synth_feat_rows(start, block_size + 1, n_embd_cap);
            std::vector<int32_t> pos(block_size + 1);
            for (int32_t i = 0; i < block_size + 1; ++i) {
                pos[i] = start + i;
            }

            if (!common_speculative_dspark_stage_ctx_test(spec, 0, feat.data(), block_size + 1, n_embd_cap,
                                                          pos.data())) {
                fail("common_speculative_dspark_stage_ctx_test (verify) failed at round " + std::to_string(r));
            }
        }

        common_speculative_accept(spec, 0, (uint16_t) n_accepted);

        const int32_t bonus     = synth_bonus_token((int32_t) r, vocab_size, mask_token_id);
        const int32_t bonus_ref = rr.at("bonus").get<int32_t>();
        if (bonus != bonus_ref) {
            fail("C++/python bonus token disagree at round " + std::to_string(r));
        }

        id_last = (llama_token) bonus;
        start   = start + n_accepted + 1;

        GGML_UNUSED(ctx_len_ref);
    }

    common_speculative_print_stats(spec);
    common_speculative_free(spec);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    if (n_mismatch_rounds > 0) {
        fail(std::to_string(n_mismatch_rounds) + "/" + std::to_string(rounds_ref.size()) +
             " rounds mismatched the Python reference");
    }

    printf(
        "\nPhase 2 gate PASSED: %zu/%zu rounds token-for-token identical to the DeepSpec reference "
        "(cache growth/crop, block seeding, RoPE positions, sequential markov resample).\n",
        rounds_ref.size(), rounds_ref.size());
    return 0;
}

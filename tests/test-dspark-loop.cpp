// Phase 2 gate for the dspark block-draft loop (common_speculative_impl_draft_dspark
// in common/speculative.cpp). Phase 1 (src/models/dspark.cpp, the forward graph
// for a single call) is already gated bit-accurate against the Python reference's real
// Qwen3DSparkModel -- see test-dspark-forward.cpp. This test exercises the NEW
// Phase 2 piece: the repeated draft/verify loop around that graph -- persistent
// KV-cache growth/crop, block seeding (anchor + mask_token_id), continuous
// absolute RoPE positions across rounds, and the sequential (never-batched)
// Markov resample.
//
// There is no real target model available for this gate, so this drives a
// small, fully-synthetic checkpoint (using the real Python reference
// Qwen3DSparkModel class with tiny dims) through several rounds of the draft
// loop, feeding a
// closed-form deterministic "target tap feature" stand-in via the TEST-ONLY
// common_speculative_dspark_stage_ctx_test() hook (bypassing the normal
// process()-driven capture path, which needs a real target context -- see
// that function's doc comment in common/speculative.h).
//
// scratchpad/dspark_phase2_py_ref.py drives the SAME rounds through the real
// Python reference implementation ops (forward_dspark_draft_block + the model's own
// sample_draft_token_step) using an IDENTICAL closed-form synthetic-feature
// generator (synth_feat/synth_bonus_token below, reimplemented byte-for-byte
// from the same hash constants) and an identical fixed accept-count schedule,
// dumping the expected per-round drafted token block to JSON. This program
// reads that JSON and diffs its own output against it, round for round,
// token for token.
//
// usage: test-dspark-loop <tiny.gguf> <ref.json>

#include "llama.h"
#include "common.h"
#include "speculative.h"
#include "../src/llama-ext.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

[[noreturn]] static void fail(const std::string & msg) {
    fprintf(stderr, "FAIL: %s\n", msg.c_str());
    exit(1);
}

// --- synthetic "target tap feature" stand-in ---------------------------
// Reimplemented byte-for-byte from scratchpad/dspark_phase2_py_ref.py's
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
    const uint32_t h = hash_u32((uint32_t)(pos * 131071 + d * 97 + 12345));
    const int32_t  m = (int32_t)(h % 2000u) - 1000; // [-1000, 999]
    return (float) m / 500.0f;                      // [-2.0, 1.998]
}

static int32_t synth_bonus_token(int32_t round_idx, int32_t vocab_size, int32_t mask_token_id) {
    const uint32_t h = hash_u32((uint32_t) round_idx * 2654435761u + 999983u);
    int32_t v = (int32_t)(h % (uint32_t)(vocab_size - 1));
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

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <tiny-dspark.gguf> <ref.json>\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string ref_path   = argv[2];

    std::ifstream f(ref_path);
    if (!f) fail("could not open ref file: " + ref_path);
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
    mparams.n_gpu_layers = 0; // CPU: deterministic, no Metal precision surprises

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) fail("failed to load model: " + model_path);

    llama_dspark_meta meta;
    if (!llama_model_dspark_get_meta(model, &meta)) {
        fail("llama_model_dspark_get_meta failed -- not a dspark model?");
    }

    printf("meta: n_embd=%lld n_vocab=%lld n_capture=%lld n_embd_cap=%lld block_size=%d mask_token_id=%d markov_rank=%lld\n",
            (long long) meta.n_embd, (long long) meta.n_vocab, (long long) meta.n_capture, (long long) meta.n_embd_cap,
            meta.block_size, meta.mask_token_id, (long long) meta.markov_rank);

    if (meta.n_embd_cap    != n_embd_cap_ref)    fail("n_embd_cap mismatch vs ref.json");
    if (meta.block_size    != block_size_ref)    fail("block_size mismatch vs ref.json");
    if (meta.n_vocab       != vocab_size_ref)    fail("vocab_size mismatch vs ref.json");
    if (meta.mask_token_id != mask_token_id_ref) fail("mask_token_id mismatch vs ref.json");

    const int64_t n_embd_cap    = meta.n_embd_cap;
    const int32_t block_size    = meta.block_size;
    const int32_t vocab_size    = (int32_t) meta.n_vocab;
    const int32_t mask_token_id = meta.mask_token_id;

    // size the context generously: worst-case round needs ctx_len(<= previous
    // block_size+1) + block_size tokens in one llama_decode call.
    const uint32_t n_ctx_max = (uint32_t) (block_size + 1 + block_size) + 8;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = n_ctx_max;
    cparams.n_batch   = n_ctx_max;
    cparams.n_ubatch  = n_ctx_max;
    cparams.n_seq_max = 1;
    cparams.no_perf   = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) fail("llama_init_from_model failed");

    common_params_speculative sparams;
    sparams.types          = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
    sparams.draft.ctx_dft  = ctx;
    // No real target model exists in this synthetic test (see file header
    // comment); dspark's process() path (which reads from ctx_tgt) is never
    // exercised here -- context rows are injected directly via
    // common_speculative_dspark_stage_ctx_test(). ctx_tgt only needs to be a
    // valid, non-null context to satisfy the impl's construction-time assert.
    sparams.draft.ctx_tgt  = ctx;
    sparams.draft.n_max    = block_size;
    sparams.draft.n_min    = 0;

    common_speculative * spec = common_speculative_init(sparams, /* n_seq = */ 1);
    if (!spec) fail("common_speculative_init returned null");

    common_speculative_begin(spec, /* seq_id = */ 0, PROMPT);

    // one-time prefill seeding: the whole prompt's (synthetic) tap features,
    // positions [0, N).
    {
        const int64_t N = (int64_t) PROMPT.size();
        std::vector<float>   feat = synth_feat_rows(0, N, n_embd_cap);
        std::vector<int32_t> pos(N);
        for (int64_t i = 0; i < N; ++i) pos[i] = (int32_t) i;

        if (!common_speculative_dspark_stage_ctx_test(spec, 0, feat.data(), N, n_embd_cap, pos.data())) {
            fail("common_speculative_dspark_stage_ctx_test (prefill) failed");
        }
    }

    llama_pos   start   = (llama_pos) PROMPT.size();
    llama_token id_last  = (llama_token) synth_bonus_token(-1, vocab_size, mask_token_id);
    if (id_last != prefill_bonus_ref) fail("C++/python prefill bonus token disagree -- synth_bonus_token drifted");

    int32_t n_mismatch_rounds = 0;

    for (size_t r = 0; r < rounds_ref.size(); ++r) {
        const auto & rr = rounds_ref[r];
        const int32_t n_accepted   = rr.at("n_accepted").get<int32_t>();
        const int64_t ctx_len_ref  = rr.at("ctx_len").get<int64_t>();
        const int64_t start_ref    = rr.at("start").get<int64_t>();
        const std::vector<int32_t> sampled_ref = rr.at("sampled").get<std::vector<int32_t>>();

        if ((int32_t) ACCEPT_SCHEDULE[r % ACCEPT_SCHEDULE.size()] != n_accepted) {
            fail("ACCEPT_SCHEDULE drifted out of sync with ref.json at round " + std::to_string(r));
        }
        if ((int64_t) start != start_ref) {
            fail("start bookkeeping disagrees with ref.json at round " + std::to_string(r) +
                 " (cpp=" + std::to_string(start) + " py=" + std::to_string(start_ref) + ")");
        }

        common_speculative_draft_params & dp = common_speculative_get_draft_params(spec, 0);
        dp.drafting = true;
        dp.n_max    = -1;
        dp.n_past   = start;
        dp.id_last  = id_last;
        dp.prompt   = nullptr; // unused by dspark
        llama_tokens result;
        dp.result   = &result;

        common_speculative_draft(spec);

        printf("round %zu: n_past=%d id_last=%d -> result=[", r, start, id_last);
        for (auto t : result) printf("%d ", t);
        printf("] expected=[");
        for (auto t : sampled_ref) printf("%d ", t);
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
            for (int32_t i = 0; i < block_size + 1; ++i) pos[i] = start + i;

            if (!common_speculative_dspark_stage_ctx_test(spec, 0, feat.data(), block_size + 1, n_embd_cap, pos.data())) {
                fail("common_speculative_dspark_stage_ctx_test (verify) failed at round " + std::to_string(r));
            }
        }

        common_speculative_accept(spec, 0, (uint16_t) n_accepted);

        const int32_t bonus = synth_bonus_token((int32_t) r, vocab_size, mask_token_id);
        const int32_t bonus_ref = rr.at("bonus").get<int32_t>();
        if (bonus != bonus_ref) fail("C++/python bonus token disagree at round " + std::to_string(r));

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

    printf("\nPhase 2 gate PASSED: %zu/%zu rounds token-for-token identical to the Python reference implementation "
           "(cache growth/crop, block seeding, RoPE positions, sequential markov resample).\n",
           rounds_ref.size(), rounds_ref.size());
    return 0;
}

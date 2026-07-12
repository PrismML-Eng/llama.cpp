// Phase 1 gate for the dspark forward graph (src/models/dspark.cpp).
//
// Tier 1 (--tier1): load a real dspark GGUF, feed SYNTHETIC (deterministic,
// in-process) tap features + a draft block, assert the forward pass produces
// finite, sane-shaped logits. Catches wiring bugs cheaply before worrying
// about numerical correctness.
//
// Tier 2 (--tier2 <ref.bin>): feed the SAME tap features / draft tokens /
// positions that a companion Python script (scripts or scratchpad
// dspark_tier2_gen.py) fed through the Python reference's real drafter model, and diff
// this program's logits against the reference logits dumped in ref.bin.
//
// ref.bin layout (little-endian):
//   int32 n_ctx_rows, int32 n_embd_cap, int32 block_size, int32 vocab_size
//   int32[n_ctx_rows]   ctx_pos
//   float32[n_ctx_rows * n_embd_cap]  ctx_feat            (row-major)
//   int32[block_size]   draft_token_ids
//   int32[block_size]   draft_pos
//   float32[block_size * vocab_size]  ref_logits          (row-major)

#include "llama.h"
#include "../src/llama-ext.h"
#include "../src/llama-model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>

[[noreturn]] static void fail(const std::string & msg) {
    fprintf(stderr, "FAIL: %s\n", msg.c_str());
    exit(1);
}

// dspark ships no real tokenizer, but the converter's set_vocab() calls
// add_vocab_size() to fill the "none" tokenizer with dummy entries sized to the
// target's real vocab width (so batch validation passes) -- so
// llama_vocab_n_tokens() reports that width. Use the public/exported vocab API
// rather than the internal llama_model tensor map, which is not exported across
// the Windows DLL boundary.
static int64_t n_vocab_from_model(const llama_model * model) {
    return llama_vocab_n_tokens(llama_model_get_vocab(model));
}

struct dspark_meta {
    int64_t n_embd   = 0;
    int64_t n_vocab  = 0;
    int32_t block_size = 0;
    int32_t mask_token_id = 0;
    int32_t n_capture = 0;
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
    // target_layers is an array KV; llama_model_meta_val_str doesn't expand
    // arrays, so just count how many "dspark.dspark.target_layers.N" entries
    // llama.cpp's generic dumper would produce -- easier: derive n_capture
    // from n_embd_cap==0 checks isn't available generically, so the caller
    // passes it explicitly for now (the drafter checkpoints tested here
    // use 5). Keep a sane fallback.
    m.n_capture = 5;

    return m;
}

static llama_context * make_ctx(llama_model * model, uint32_t n_ctx) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = n_ctx;
    cparams.n_batch     = n_ctx;
    cparams.n_ubatch    = n_ctx;
    cparams.n_seq_max   = 1;
    cparams.no_perf     = true;

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
static std::vector<float> run_forward(
        llama_context * ctx,
        int64_t n_ctx_rows, int64_t n_embd_cap, const std::vector<float> & ctx_feat, const std::vector<int32_t> & ctx_pos,
        int32_t block_size, const std::vector<int32_t> & draft_tokens, const std::vector<int32_t> & draft_pos,
        int64_t n_vocab) {

    if ((int64_t) ctx_feat.size() != n_ctx_rows * n_embd_cap) fail("ctx_feat size mismatch");
    if ((int64_t) ctx_pos.size()  != n_ctx_rows)              fail("ctx_pos size mismatch");
    if ((int64_t) draft_tokens.size() != block_size)          fail("draft_tokens size mismatch");
    if ((int64_t) draft_pos.size()    != block_size)          fail("draft_pos size mismatch");

    llama_set_dspark_ctx(ctx, ctx_feat.data(), n_ctx_rows, n_embd_cap);

    const int32_t n_tokens = (int32_t) n_ctx_rows + block_size;
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;

    for (int32_t i = 0; i < (int32_t) n_ctx_rows; ++i) {
        batch.token[i]     = 0; // dummy: unused (context rows never join the trunk/embedding path)
        batch.pos[i]       = ctx_pos[i];
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 0;
    }
    for (int32_t j = 0; j < block_size; ++j) {
        const int32_t i = (int32_t) n_ctx_rows + j;
        batch.token[i]     = draft_tokens[j];
        batch.pos[i]       = draft_pos[j];
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }

    const int32_t rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        fail("llama_decode returned " + std::to_string(rc));
    }

    llama_set_dspark_ctx(ctx, nullptr, 0, 0); // clear staged context

    float * logits = llama_get_logits(ctx);
    if (!logits) fail("llama_get_logits returned null");

    return std::vector<float>(logits, logits + (size_t) block_size * n_vocab);
}

static int run_tier1(const std::string & model_path) {
    printf("=== Tier 1: synthetic tap features, wiring/shape/finiteness check ===\n");

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU: deterministic, no Metal precision surprises

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) fail("failed to load model: " + model_path);

    dspark_meta meta = read_dspark_meta(model);
    printf("n_embd=%lld n_vocab=%lld block_size=%d mask_token_id=%d n_capture=%d\n",
            (long long) meta.n_embd, (long long) meta.n_vocab, meta.block_size, meta.mask_token_id, meta.n_capture);
    if (meta.block_size <= 0) fail("could not read dspark.dspark.block_size from GGUF");

    const int64_t n_ctx_rows  = 6;
    const int64_t n_embd_cap  = (int64_t) meta.n_capture * meta.n_embd;
    const int32_t block_size  = meta.block_size;

    llama_context * ctx = make_ctx(model, (uint32_t)(n_ctx_rows + block_size));

    // deterministic synthetic tap features (fixed seed, no external data needed).
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    std::vector<float> ctx_feat((size_t) n_ctx_rows * n_embd_cap);
    for (auto & v : ctx_feat) v = dist(rng);

    std::vector<int32_t> ctx_pos(n_ctx_rows);
    for (int64_t i = 0; i < n_ctx_rows; ++i) ctx_pos[i] = (int32_t) i;

    std::vector<int32_t> draft_tokens(block_size, meta.mask_token_id);
    draft_tokens[0] = 1000; // anchor token (last accepted real token)
    std::vector<int32_t> draft_pos(block_size);
    for (int32_t i = 0; i < block_size; ++i) draft_pos[i] = (int32_t)(n_ctx_rows + i);

    std::vector<float> logits = run_forward(ctx, n_ctx_rows, n_embd_cap, ctx_feat, ctx_pos,
                                             block_size, draft_tokens, draft_pos, meta.n_vocab);

    size_t n_nonfinite = 0;
    float min_v = logits[0], max_v = logits[0];
    for (float v : logits) {
        if (!std::isfinite(v)) n_nonfinite++;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }

    printf("logits: count=%zu min=%g max=%g non_finite=%zu\n", logits.size(), min_v, max_v, n_nonfinite);

    for (int32_t p = 0; p < block_size; ++p) {
        const float * row = logits.data() + (size_t) p * meta.n_vocab;
        int64_t argmax = 0;
        for (int64_t v = 1; v < meta.n_vocab; ++v) if (row[v] > row[argmax]) argmax = v;
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

    printf("Tier 1 PASSED: finite, correctly-shaped logits (%d x %lld)\n", block_size, (long long) meta.n_vocab);
    return 0;
}

static int run_tier2(const std::string & model_path, const std::string & ref_path,
                     double min_argmax_match_rate, double min_top5_overlap) {
    printf("=== Tier 2: real drafter weights, deterministic tap features, diff vs Python reference implementation ===\n");

    std::ifstream f(ref_path, std::ios::binary);
    if (!f) fail("could not open ref file: " + ref_path);

    int32_t n_ctx_rows_i, n_embd_cap_i, block_size, vocab_size;
    f.read((char*)&n_ctx_rows_i, 4);
    f.read((char*)&n_embd_cap_i, 4);
    f.read((char*)&block_size, 4);
    f.read((char*)&vocab_size, 4);
    if (!f) fail("ref file truncated (header)");

    const int64_t n_ctx_rows = n_ctx_rows_i;
    const int64_t n_embd_cap = n_embd_cap_i;

    std::vector<int32_t> ctx_pos(n_ctx_rows);
    f.read((char*)ctx_pos.data(), n_ctx_rows * sizeof(int32_t));

    std::vector<float> ctx_feat((size_t) n_ctx_rows * n_embd_cap);
    f.read((char*)ctx_feat.data(), ctx_feat.size() * sizeof(float));

    std::vector<int32_t> draft_tokens(block_size);
    f.read((char*)draft_tokens.data(), block_size * sizeof(int32_t));

    std::vector<int32_t> draft_pos(block_size);
    f.read((char*)draft_pos.data(), block_size * sizeof(int32_t));

    std::vector<float> ref_logits((size_t) block_size * vocab_size);
    f.read((char*)ref_logits.data(), ref_logits.size() * sizeof(float));
    if (!f) fail("ref file truncated (payload)");

    printf("ref: n_ctx_rows=%lld n_embd_cap=%lld block_size=%d vocab_size=%d\n",
            (long long) n_ctx_rows, (long long) n_embd_cap, block_size, vocab_size);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) fail("failed to load model: " + model_path);

    const int64_t n_vocab = n_vocab_from_model(model);
    if (n_vocab != vocab_size) fail("vocab size mismatch between GGUF (" + std::to_string(n_vocab) +
                                     ") and reference (" + std::to_string(vocab_size) + ")");

    llama_context * ctx = make_ctx(model, (uint32_t)(n_ctx_rows + block_size));

    std::vector<float> logits = run_forward(ctx, n_ctx_rows, n_embd_cap, ctx_feat, ctx_pos,
                                             block_size, draft_tokens, draft_pos, n_vocab);

    llama_free(ctx);
    llama_model_free(model);

    // --- diff ---
    double sum_abs_diff = 0.0, max_abs_diff = 0.0;
    size_t n_nonfinite = 0;
    int32_t argmax_matches = 0;
    double top5_overlap_sum = 0.0;

    for (int32_t p = 0; p < block_size; ++p) {
        const float * a = logits.data()     + (size_t) p * n_vocab; // C++
        const float * b = ref_logits.data() + (size_t) p * n_vocab; // python

        // scan from v=0 so a NaN or discrepancy at token 0 is counted in the diff
        // and non-finite metrics (argmax stays correct: it is seeded at index 0).
        int64_t argmax_a = 0, argmax_b = 0;
        for (int64_t v = 0; v < n_vocab; ++v) {
            if (a[v] > a[argmax_a]) argmax_a = v;
            if (b[v] > b[argmax_b]) argmax_b = v;
            const double d = std::fabs((double) a[v] - (double) b[v]);
            sum_abs_diff += d;
            max_abs_diff = std::max(max_abs_diff, d);
            if (!std::isfinite(a[v])) n_nonfinite++;
        }
        if (argmax_a == argmax_b) argmax_matches++;

        // top-5 overlap (set intersection size / 5)
        std::vector<int64_t> idx_a(n_vocab), idx_b(n_vocab);
        for (int64_t v = 0; v < n_vocab; ++v) { idx_a[v] = v; idx_b[v] = v; }
        std::partial_sort(idx_a.begin(), idx_a.begin()+5, idx_a.end(), [&](int64_t x, int64_t y){ return a[x] > a[y]; });
        std::partial_sort(idx_b.begin(), idx_b.begin()+5, idx_b.end(), [&](int64_t x, int64_t y){ return b[x] > b[y]; });
        std::vector<int64_t> top5_a(idx_a.begin(), idx_a.begin()+5), top5_b(idx_b.begin(), idx_b.begin()+5);
        std::sort(top5_a.begin(), top5_a.end());
        std::sort(top5_b.begin(), top5_b.end());
        std::vector<int64_t> inter;
        std::set_intersection(top5_a.begin(), top5_a.end(), top5_b.begin(), top5_b.end(), std::back_inserter(inter));
        top5_overlap_sum += inter.size() / 5.0;

        printf("  pos %d: cpp_argmax=%lld py_argmax=%lld cpp_top1_logit=%g py_top1_logit=%g top5_overlap=%d/5\n",
                p, (long long) argmax_a, (long long) argmax_b, a[argmax_a], b[argmax_b], (int) inter.size());
    }

    const double mean_abs_diff = sum_abs_diff / ((double) block_size * n_vocab);
    const double argmax_match_rate = (double) argmax_matches / block_size;
    const double mean_top5_overlap = top5_overlap_sum / block_size;

    printf("\n--- Tier 2 summary ---\n");
    printf("mean_abs_diff=%.6g max_abs_diff=%.6g non_finite=%zu\n", mean_abs_diff, max_abs_diff, n_nonfinite);
    printf("argmax_match_rate=%.3f (%d/%d) mean_top5_overlap=%.3f\n",
            argmax_match_rate, argmax_matches, block_size, mean_top5_overlap);

    if (n_nonfinite > 0) {
        fail("Tier 2 FAILED: non-finite logits");
    }

    // Gate on agreement with the Python reference. Both metrics are scale-invariant
    // rates in [0,1]: a correct C++ drafter matches the reference argmax on nearly
    // every position and shares nearly all of its top-5, while a broken one does
    // not -- without this the test passed on any finite output (even zero matches).
    // Thresholds default high and are overridable from the CLI for calibration.
    if (argmax_match_rate < min_argmax_match_rate) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Tier 2 FAILED: argmax_match_rate %.3f < %.3f",
                 argmax_match_rate, min_argmax_match_rate);
        fail(buf);
    }
    if (mean_top5_overlap < min_top5_overlap) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Tier 2 FAILED: mean_top5_overlap %.3f < %.3f",
                 mean_top5_overlap, min_top5_overlap);
        fail(buf);
    }
    printf("Tier 2 PASSED: argmax_match_rate=%.3f (>= %.3f), mean_top5_overlap=%.3f (>= %.3f)\n",
           argmax_match_rate, min_argmax_match_rate, mean_top5_overlap, min_top5_overlap);

    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> --tier1 | "
                "--tier2 <ref.bin> [min_argmax_match_rate] [min_top5_overlap]\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    const std::string model_path = argv[1];
    const std::string mode = argv[2];

    int rc;
    if (mode == "--tier1") {
        rc = run_tier1(model_path);
    } else if (mode == "--tier2") {
        if (argc < 4) fail("--tier2 requires a ref.bin path");
        // greedy argmax should match the reference on essentially every position;
        // default the gate high and allow calibration from the CLI.
        double min_argmax_match_rate = argc > 4 ? atof(argv[4]) : 0.90;
        double min_top5_overlap      = argc > 5 ? atof(argv[5]) : 0.90;
        rc = run_tier2(model_path, argv[3], min_argmax_match_rate, min_top5_overlap);
    } else {
        fail("unknown mode: " + mode);
        rc = 1;
    }

    llama_backend_free();
    return rc;
}

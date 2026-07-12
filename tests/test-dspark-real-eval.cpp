// Phase 3 (real-target) eval harness for the dspark block-draft loop
// (common_speculative_impl_draft_dspark, common/speculative.cpp).
//
// tests/test-dspark-loop.cpp gates loop MECHANICS (cache crop, RoPE
// positions, sequential markov resample) against a tiny synthetic
// checkpoint and synthetic target-tap features -- it never touches a real
// target model. This program is the first harness that drives the SAME
// public common_speculative_* API against a real target and a real
// drafter GGUF end to end, to get a first real accept-rate/tau signal.
//
// Why this file exists (there is no pre-existing CLI/server path):
//   - `--spec-type draft-dspark` parses (common/arg.cpp), but neither
//     examples/speculative-simple/speculative-simple.cpp nor
//     tools/server/server-context.cpp ever calls llama_set_capture_layers()
//     on the target context. dspark's process() (see common/speculative.cpp)
//     needs the target's multi-layer tap captured via
//     llama_get_embeddings_capture_ith(), which returns null unless capture
//     is engaged AND logits/output were requested for every row. Phase 2's
//     scope was the loop implementation + its own synthetic-target gate, not
//     this CLI/server wiring -- see docs/dspark-scope.md.
//   - the drafter's target_layers array (the layer-id list to pass to
//     llama_set_capture_layers) has no public getter: llama_model's own
//     string-KV cache explicitly SKIPS array-typed GGUF keys (see
//     llama_model_base::load_hparams in src/llama-model.cpp, "if (type ==
//     GGUF_TYPE_ARRAY) continue;"), so llama_model_meta_val_str() can never
//     see it. This harness instead reads the drafter GGUF's own
//     "<arch>.dspark.target_layers" key directly via the low-level gguf.h
//     C API (no core-file changes needed).
//
// The per-round draft/verify/accept loop below mirrors
// examples/speculative-simple/speculative-simple.cpp's target-verify pattern
// (common_sampler_sample_and_accept_n against a greedy/temp=0 target
// sampler) but is NOT a drop-in generalization of that file: dspark differs
// in two structural ways that file doesn't handle --
//   1. it needs common_speculative_process() called explicitly on every
//      target batch (prefill AND each round's verify batch) so the tap
//      capture actually gets consumed -- see the header comment above
//      common_speculative_need_embd_capture() in common/speculative.h.
//   2. it must NEVER llama_decode() the verify batch against ctx_dft (that
//      file does this for ordinary draft-model types); dspark's drafter
//      cache is advanced entirely inside common_speculative_draft() itself
//      via the out-of-band llama_set_dspark_ctx() staging, and re-decoding
//      the verify batch's token ids through the drafter's own embedding
//      table would be meaningless (see src/models/dspark.cpp).
//
// usage: test-dspark-real-eval <target.gguf> <drafter.gguf> [n_predict=96] [n_gpu_layers=999] [dataset.jsonl] [n_prompts=24] [n_max_override=block_size]
//
// n_max_override (1..block_size) caps the per-round draft length dp.n_max.
// The dspark impl itself always produces a full block_size block (its
// markov-resample loop is fixed-length and its drafter cache is cropped back
// to `start` inside draft() regardless of acceptance), but the generic
// common_speculative_draft() dispatcher truncates *dp.result to dp.n_max
// after the impl returns -- so capping here shrinks only the VERIFY batch,
// leaving the drafter's per-round cost unchanged (the honest semantics for a
// block-diffusion drafter).
//
// when dataset.jsonl is given, prompts are loaded in the Python reference
// implementation's format (one {"turns": [...]} object per line, single turn
// only) and run through the target's own tokenizer.chat_template (via
// common_chat_templates_apply) instead of this file's built-in
// plain-text-continuation PROMPTS below -- for a run directly comparable to
// the Python reference implementation's numbers, which apply the same chat template.

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "chat.h"
#include "../src/llama-ext.h"
#include "gguf.h"

#include "nlohmann/json.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

[[noreturn]] static void fail(const std::string & msg) {
    fprintf(stderr, "FAIL: %s\n", msg.c_str());
    exit(1);
}

struct prompt_spec {
    std::string category;
    std::string text;
};

// the Python reference implementation's eval-set format: one JSON object per
// line, {"turns": ["<single user turn>"]}. Loading real reference-implementation
// prompts (instead of this file's own hand-written set below) and
// running them through the target's own chat template is what makes a run
// directly comparable to the Python reference implementation's accept-rate numbers,
// rather than this harness's own plain-text-continuation methodology.
static std::vector<prompt_spec> load_eval_jsonl(const std::string & path, int limit) {
    std::ifstream f(path);
    if (!f) {
        fail("failed to open dataset file: " + path);
    }

    std::vector<prompt_spec> out;
    std::string line;
    while (std::getline(f, line) && (limit <= 0 || (int) out.size() < limit)) {
        if (line.empty()) {
            continue;
        }
        nlohmann::json j = nlohmann::json::parse(line);
        out.push_back({ "eval-jsonl", j.at("turns").at(0).get<std::string>() });
    }

    return out;
}

// 24 held-out prompts across 3 categories. Plain-text continuations (no
// chat template applied) so this measures raw free-running rollout accept
// behavior, same spirit as the Python reference free-running alpaca/arena-hard eval
// this is being compared against.
static const std::vector<prompt_spec> PROMPTS = {
    { "code", "def is_prime(n):\n    \"\"\"Return True if n is a prime number, else False.\"\"\"\n" },
    { "code", "import heapq\n\ndef k_smallest(nums, k):\n    \"\"\"Return the k smallest elements of nums, sorted.\"\"\"\n" },
    { "code", "class LRUCache:\n    \"\"\"A least-recently-used cache with fixed capacity.\"\"\"\n    def __init__(self, capacity: int):\n" },
    { "code", "// Reverse a singly linked list in place.\nstruct Node { int val; Node* next; };\nNode* reverse(Node* head) {\n" },
    { "code", "def merge_intervals(intervals):\n    \"\"\"Given a list of [start, end] intervals, merge all overlapping ones.\"\"\"\n" },
    { "code", "#include <vector>\n#include <algorithm>\n// Binary search for the first index where arr[i] >= target.\nint lower_bound_idx(std::vector<int>& arr, int target) {\n" },
    { "code", "def quicksort(arr):\n    \"\"\"Sort arr in place using the quicksort algorithm.\"\"\"\n    if len(arr) <= 1:\n        return arr\n" },
    { "code", "-- SQL: return the top 3 highest-paid employees per department.\nSELECT\n" },
    { "chat", "Q: What's the difference between a list and a tuple in Python?\nA:" },
    { "chat", "Q: Can you explain how photosynthesis works, in simple terms?\nA:" },
    { "chat", "Q: I have $500 and want to invest it for 5 years. What are some options to consider?\nA:" },
    { "chat", "Q: Write a short, friendly email declining a meeting invite because of a scheduling conflict.\nA:" },
    { "chat", "Q: What are three practical tips for improving sleep quality?\nA:" },
    { "chat", "Q: Explain the plot of Romeo and Juliet in two sentences.\nA:" },
    { "chat", "Q: My laptop fan is very loud under light load. What should I check first?\nA:" },
    { "chat", "Q: Summarize the main causes of the French Revolution in a short paragraph.\nA:" },
    { "reasoning", "Q: A train leaves city A at 60 mph and another leaves city B (300 miles away) at 90 mph, heading toward each other. How long until they meet?\nA: Let's think step by step." },
    { "reasoning", "Q: If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Explain.\nA:" },
    { "reasoning", "Q: A store marks up an item by 40% then offers a 25% discount off the marked-up price. Is the final price higher or lower than the original? By how much?\nA: Let's think step by step." },
    { "reasoning", "Q: Why is the sky blue during the day but red/orange at sunset?\nA:" },
    { "reasoning", "Q: You have 8 identical-looking balls, one of which is heavier. Using a balance scale only twice, how do you find the heavier ball?\nA: Let's think step by step." },
    { "reasoning", "Q: Which is a better estimate of the number of piano tuners in a large city: 50, 500, or 5000? Explain your reasoning.\nA:" },
    { "reasoning", "Q: Two coworkers, Alice and Bob, always tell the truth or always lie. Alice says \"Bob always lies.\" What can you conclude?\nA: Let's think step by step." },
    { "reasoning", "Q: A recipe calls for 3/4 cup of sugar for 12 cookies. How much sugar is needed for 30 cookies?\nA: Let's think step by step." },
};

// dspark ships no tokenizer / vocab of its own (see conversion/dspark.py and
// src/models/dspark.cpp): the drafter GGUF's target_layers array is only
// discoverable by reading the file's own metadata directly, not via any
// llama_model_* accessor -- see the file header comment.
static std::vector<int32_t> read_dspark_target_layers(const std::string & drafter_path) {
    struct gguf_init_params gp = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
    gguf_context * gctx = gguf_init_from_file(drafter_path.c_str(), gp);
    if (gctx == nullptr) {
        fail("gguf_init_from_file failed for " + drafter_path);
    }

    const int64_t arch_kid = gguf_find_key(gctx, "general.architecture");
    if (arch_kid < 0) {
        fail(drafter_path + ": missing general.architecture key");
    }
    const std::string arch = gguf_get_val_str(gctx, arch_kid);

    const std::string key = arch + ".dspark.target_layers";
    const int64_t kid = gguf_find_key(gctx, key.c_str());
    if (kid < 0) {
        fail(drafter_path + ": missing GGUF key " + key);
    }
    if (gguf_get_kv_type(gctx, kid) != GGUF_TYPE_ARRAY) {
        fail(key + " is not an array-typed KV");
    }

    const enum gguf_type arr_type = gguf_get_arr_type(gctx, kid);
    const size_t         n        = gguf_get_arr_n(gctx, kid);
    const void *          data     = gguf_get_arr_data(gctx, kid);

    std::vector<int32_t> out(n);
    for (size_t i = 0; i < n; i++) {
        switch (arr_type) {
            case GGUF_TYPE_INT32:  out[i] = ((const int32_t  *) data)[i]; break;
            case GGUF_TYPE_UINT32: out[i] = (int32_t) ((const uint32_t *) data)[i]; break;
            case GGUF_TYPE_INT64:  out[i] = (int32_t) ((const int64_t  *) data)[i]; break;
            case GGUF_TYPE_UINT64: out[i] = (int32_t) ((const uint64_t *) data)[i]; break;
            default:
                fail(key + ": unexpected array element gguf_type " + std::to_string((int) arr_type));
        }
    }

    gguf_free(gctx);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <target.gguf> <drafter.gguf> [n_predict=96] [n_gpu_layers=999] [dataset.jsonl] [n_prompts=24] [n_max_override=block_size]\n", argv[0]);
        return 1;
    }
    const std::string target_path   = argv[1];
    const std::string drafter_path  = argv[2];
    const int         n_predict_max = argc > 3 ? std::atoi(argv[3]) : 96;
    const int         n_gpu_layers  = argc > 4 ? std::atoi(argv[4]) : 999;
    const std::string dataset_path  = argc > 5 ? argv[5] : "";
    const int         n_prompts     = argc > 6 ? std::atoi(argv[6]) : 24;
    const int         n_max_arg     = argc > 7 ? std::atoi(argv[7]) : 0; // 0 == use the drafter's baked block_size

    std::vector<int32_t> target_layers = read_dspark_target_layers(drafter_path);
    printf("dspark target_layers (%zu):", target_layers.size());
    for (auto l : target_layers) printf(" %d", l);
    printf("\n");

    llama_backend_init();

    // --- drafter (loaded first: the target context needs meta.block_size
    // below to size its recurrent-state rollback ring) ---
    llama_model_params mparams_dft = llama_model_default_params();
    mparams_dft.n_gpu_layers = n_gpu_layers;
    llama_model * model_dft = llama_model_load_from_file(drafter_path.c_str(), mparams_dft);
    if (!model_dft) fail("failed to load drafter model: " + drafter_path);

    llama_dspark_meta meta;
    if (!llama_model_dspark_get_meta(model_dft, &meta)) {
        fail("drafter GGUF does not look like a dspark model (missing dspark.*.block_size KV)");
    }
    if ((int64_t) target_layers.size() != meta.n_capture) {
        fail("target_layers count (" + std::to_string(target_layers.size()) +
             ") != meta.n_capture (" + std::to_string(meta.n_capture) + ")");
    }

    printf("dspark meta: n_embd=%lld n_vocab=%lld n_capture=%lld n_embd_cap=%lld block_size=%d mask_token_id=%d markov_rank=%lld\n",
            (long long) meta.n_embd, (long long) meta.n_vocab, (long long) meta.n_capture, (long long) meta.n_embd_cap,
            meta.block_size, meta.mask_token_id, (long long) meta.markov_rank);

    // --- target ---
    llama_model_params mparams_tgt = llama_model_default_params();
    mparams_tgt.n_gpu_layers = n_gpu_layers;
    llama_model * model_tgt = llama_model_load_from_file(target_path.c_str(), mparams_tgt);
    if (!model_tgt) fail("failed to load target model: " + target_path);
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);

    llama_context_params cparams_tgt = llama_context_default_params();
    cparams_tgt.n_ctx     = 4096;
    cparams_tgt.n_batch   = 2048;
    cparams_tgt.n_ubatch  = 2048;
    cparams_tgt.n_seq_max = 1;
    cparams_tgt.no_perf   = true;
    // dspark verifies a whole draft block against the target in one
    // llama_decode(), then crops the target's cache back to the accepted
    // length with a PARTIAL llama_memory_seq_rm(). On a hybrid GDN/attention
    // target that partial removal only succeeds if the recurrent-state
    // rollback ring (n_rs_seq) was sized up front -- see
    // common_params_speculative::need_n_rs_seq() and
    // llama_memory_recurrent::seq_rm()'s per-token snapshot index path.
    cparams_tgt.n_rs_seq = (uint32_t) meta.block_size;

    llama_context * ctx_tgt = llama_init_from_model(model_tgt, cparams_tgt);
    if (!ctx_tgt) fail("failed to create target context");

    if (llama_model_is_hybrid(model_tgt) && llama_n_rs_seq(ctx_tgt) == 0) {
        fail("target is a hybrid GDN/attention model but its context has "
             "n_rs_seq=0 -- the post-verify partial crop would silently "
             "no-op instead of rolling back the recurrent state (see "
             "llama_memory_hybrid::seq_rm)");
    }

    const bool use_chat_template = !dataset_path.empty();
    common_chat_templates_ptr tmpls;
    std::vector<prompt_spec> prompts;
    if (use_chat_template) {
        tmpls = common_chat_templates_init(model_tgt, "");
        prompts = load_eval_jsonl(dataset_path, n_prompts);
        printf("loaded %zu prompts from %s, chat-templated\n", prompts.size(), dataset_path.c_str());
    } else {
        prompts = PROMPTS;
    }

    llama_context_params cparams_dft = llama_context_default_params();
    cparams_dft.n_ctx     = 4096;
    cparams_dft.n_batch   = 2048;
    cparams_dft.n_ubatch  = 2048;
    cparams_dft.n_seq_max = 1;
    cparams_dft.no_perf   = true;

    llama_context * ctx_dft = llama_init_from_model(model_dft, cparams_dft);
    if (!ctx_dft) fail("failed to create drafter context");

    // engage the target-layer tap capture ONCE, permanently, on the target
    // context (see file header comment -- this is the missing piece no
    // existing CLI/server path wires up for dspark).
    llama_set_capture_layers(ctx_tgt, target_layers.data(), target_layers.size());

    common_params_speculative sparams;
    sparams.types          = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
    sparams.draft.ctx_dft  = ctx_dft;
    sparams.draft.ctx_tgt  = ctx_tgt;
    sparams.draft.n_max    = meta.block_size;
    sparams.draft.n_min    = 0;

    common_speculative * spec = common_speculative_init(sparams, /* n_seq = */ 1);
    if (!spec) fail("common_speculative_init returned null");

    if (!common_speculative_need_embd_capture(spec)) {
        fail("expected the dspark impl to report need_embd_capture() == true");
    }

    common_params_sampling sparams_smpl;
    sparams_smpl.temp = 0.0f; // greedy / deterministic target verification
    sparams_smpl.seed = 42;

    const llama_seq_id seq_id     = 0;
    const int32_t      block_size = meta.block_size;

    // runtime draft-length cap (see file header). the drafter still produces
    // (and pays for) a full block_size block every round; only the first
    // n_draft tokens are kept and verified.
    int32_t n_draft = n_max_arg > 0 ? n_max_arg : block_size;
    if (n_draft < 1 || n_draft > block_size) {
        fail("n_max_override must be in [1, block_size=" + std::to_string(block_size) + "], got " + std::to_string(n_draft));
    }
    printf("draft length per round: n_max=%d (block_size=%d)\n", n_draft, block_size);

    llama_batch batch_tgt = llama_batch_init((int32_t) llama_n_batch(ctx_tgt), 0, 1);

    int64_t total_drafted = 0, total_accepted = 0, total_rounds = 0, total_predicted = 0;
    int64_t total_ar_predicted = 0;
    double  total_ar_seconds = 0.0, total_sp_seconds = 0.0;

    // accept-by-depth: for 1-based draft position i, depth_reached[i] counts
    // rounds where position i was reached (i.e. positions 1..i-1 were all
    // accepted and the draft was at least i long), depth_accepted[i] counts
    // rounds where it was also accepted. depth_accepted[i]/depth_reached[i]
    // is the conditional per-depth accept rate d(i).
    std::vector<int64_t> depth_reached(block_size + 1, 0);
    std::vector<int64_t> depth_accepted(block_size + 1, 0);

    struct cat_stats { int64_t drafted = 0, accepted = 0, rounds = 0; };
    std::vector<std::string> cat_names;
    std::vector<cat_stats>   cat_stats_v;

    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        const auto & ps = prompts[pi];

        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, 0, -1);

        std::string text = ps.text;
        if (use_chat_template) {
            common_chat_templates_inputs cinputs;
            cinputs.messages.push_back({ "user", ps.text, {}, {}, "", "", "" });
            cinputs.add_generation_prompt = true;
            text = common_chat_templates_apply(tmpls.get(), cinputs).prompt;
        }

        std::vector<llama_token> inp = common_tokenize(ctx_tgt, text, /* add_special = */ true, /* parse_special = */ true);
        if (inp.size() < 2) {
            fprintf(stderr, "skipping prompt %zu: too short after tokenization\n", pi);
            continue;
        }
        if (inp.size() + (size_t) n_predict_max + (size_t) block_size + 8 > llama_n_ctx(ctx_tgt)) {
            fprintf(stderr, "warn: prompt %zu (%zu toks) + n_predict_max may exceed n_ctx=%u\n",
                    pi, inp.size(), llama_n_ctx(ctx_tgt));
        }

        llama_token              id_last    = inp.back();
        std::vector<llama_token> prompt_tgt(inp.begin(), inp.end() - 1);

        // === vanilla AR baseline (measured first, same prompt, same target
        // context) -- capture is disabled so this pays no dspark tap-capture
        // overhead, i.e. it is genuinely comparable to plain decoding, not
        // "dspark plumbing with drafting turned off". ===
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, 0, -1);
        llama_set_capture_layers(ctx_tgt, nullptr, 0);

        const auto t_ar0 = std::chrono::steady_clock::now();

        common_batch_clear(batch_tgt);
        for (size_t i = 0; i < prompt_tgt.size(); ++i) {
            common_batch_add(batch_tgt, prompt_tgt[i], (llama_pos) i, { seq_id }, /* logits = */ false);
        }
        common_batch_add(batch_tgt, id_last, (llama_pos) prompt_tgt.size(), { seq_id }, /* logits = */ true);
        if (llama_decode(ctx_tgt, batch_tgt) != 0) fail("AR prefill decode failed for prompt " + std::to_string(pi));

        common_sampler_ptr smpl_ar(common_sampler_init(model_tgt, sparams_smpl));
        llama_token ar_cur = common_sampler_sample(smpl_ar.get(), ctx_tgt, -1);
        common_sampler_accept(smpl_ar.get(), ar_cur, /* accept_grammar = */ true);

        int  ar_n_past      = (int) prompt_tgt.size() + 1; // position of ar_cur
        int  ar_n_predicted = 1;
        bool ar_has_eos     = llama_vocab_is_eog(vocab_tgt, ar_cur);

        while (ar_n_predicted < n_predict_max && !ar_has_eos) {
            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, ar_cur, (llama_pos) ar_n_past, { seq_id }, /* logits = */ true);
            if (llama_decode(ctx_tgt, batch_tgt) != 0) fail("AR decode failed at prompt " + std::to_string(pi));

            ar_cur = common_sampler_sample(smpl_ar.get(), ctx_tgt, -1);
            common_sampler_accept(smpl_ar.get(), ar_cur, /* accept_grammar = */ true);
            ar_n_past++;
            ar_n_predicted++;
            if (llama_vocab_is_eog(vocab_tgt, ar_cur)) {
                ar_has_eos = true;
            }
        }

        const double ar_seconds     = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ar0).count();
        const double ar_tok_per_sec = ar_n_predicted / ar_seconds;

        // === DSpark pass (existing logic below, now timed) ===
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, 0, -1);
        llama_set_capture_layers(ctx_tgt, target_layers.data(), target_layers.size());

        const auto t_sp0 = std::chrono::steady_clock::now();

        common_speculative_begin(spec, seq_id, prompt_tgt);

        common_sampler_ptr smpl(common_sampler_init(model_tgt, sparams_smpl));

        // manual prefill with per-row logits requested (llama_batch_get_one()
        // only requests the last row -- dspark's process() needs a capture
        // row for EVERY prompt position, see common/speculative.cpp).
        common_batch_clear(batch_tgt);
        for (size_t i = 0; i < prompt_tgt.size(); ++i) {
            common_batch_add(batch_tgt, prompt_tgt[i], (llama_pos) i, { seq_id }, /* logits = */ true);
        }
        if (llama_decode(ctx_tgt, batch_tgt) != 0) fail("prefill decode failed for prompt " + std::to_string(pi));
        if (!common_speculative_process(spec, batch_tgt)) fail("common_speculative_process (prefill) failed for prompt " + std::to_string(pi));

        int  n_past      = (int) prompt_tgt.size(); // == position of id_last
        int  n_predicted = 0;
        bool has_eos     = false;

        int64_t prompt_drafted = 0, prompt_accepted = 0, prompt_rounds = 0;

        while (n_predicted < n_predict_max && !has_eos) {
            llama_tokens draft;

            common_speculative_draft_params & dp = common_speculative_get_draft_params(spec, seq_id);
            dp.drafting = true;
            dp.n_max    = n_draft;
            dp.n_past   = n_past;
            dp.id_last  = id_last;
            dp.prompt   = nullptr; // unused by dspark
            dp.result   = &draft;

            common_speculative_draft(spec);

            if (draft.empty()) {
                fprintf(stderr, "warn: empty draft at prompt %zu, n_past=%d -- stopping this prompt early\n", pi, n_past);
                break;
            }

            // target verify batch: [id_last, draft0, draft1, ..., draftN-1],
            // matching examples/speculative-simple/speculative-simple.cpp.
            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, id_last, (llama_pos) n_past, { seq_id }, /* logits = */ true);
            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], (llama_pos) (n_past + 1 + (int) i), { seq_id }, /* logits = */ true);
            }

            if (llama_decode(ctx_tgt, batch_tgt) != 0) fail("verify decode failed at prompt " + std::to_string(pi));
            // NOTE: unlike examples/speculative-simple.cpp's generic draft-model
            // path, dspark must NOT llama_decode() this batch against ctx_dft --
            // its drafter cache was already advanced inside common_speculative_draft()
            // above via the out-of-band dspark-ctx staging (see file header comment).
            if (!common_speculative_process(spec, batch_tgt)) fail("common_speculative_process (verify) failed at prompt " + std::to_string(pi));

            auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
            if (ids.empty()) fail("common_sampler_sample_and_accept_n returned empty");

            const uint16_t n_accepted = (uint16_t) (ids.size() - 1);
            common_speculative_accept(spec, seq_id, n_accepted);

            // accept-by-depth: position i (1-based) is reached iff positions
            // 1..i-1 were all accepted, i.e. i <= n_accepted + 1.
            for (int32_t i = 1; i <= (int32_t) draft.size(); ++i) {
                if (i <= (int32_t) n_accepted + 1) depth_reached[i]++;
                if (i <= (int32_t) n_accepted)     depth_accepted[i]++;
            }

            prompt_drafted  += (int64_t) draft.size();
            prompt_accepted += n_accepted;
            prompt_rounds   += 1;

            // total newly committed tokens this round == ids.size() (n_accepted
            // draft tokens + exactly one new sample: either the mismatch
            // replacement or, on full acceptance, the bonus token) --
            // mirrors speculative-simple.cpp's n_past bookkeeping exactly.
            n_past += (int) ids.size();

            for (size_t i = 0; i < ids.size(); ++i) {
                id_last = ids[i];
                n_predicted++;
                if (llama_vocab_is_eog(vocab_tgt, id_last)) {
                    has_eos = true;
                    break;
                }
            }

            // drop the rejected tail of this round's verify batch from the
            // target's KV cache (dspark's own drafter-side cache was already
            // cropped inside draft() itself). must not ignore failure here --
            // on a hybrid GDN/attention target this is a bounded partial
            // rollback of the recurrent state, and a silently ignored no-op
            // would leave every round's rejected draft tail permanently
            // baked into the recurrent state instead of failing loudly.
            common_context_seq_rm(ctx_tgt, seq_id, n_past, -1);
        }

        const double sp_seconds     = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_sp0).count();
        const double sp_tok_per_sec = n_predicted / sp_seconds;
        const double speedup        = sp_tok_per_sec / ar_tok_per_sec;

        total_drafted   += prompt_drafted;
        total_accepted  += prompt_accepted;
        total_rounds    += prompt_rounds;
        total_predicted += n_predicted;

        total_ar_predicted += ar_n_predicted;
        total_ar_seconds   += ar_seconds;
        total_sp_seconds   += sp_seconds;

        {
            bool found = false;
            for (size_t ci = 0; ci < cat_names.size(); ++ci) {
                if (cat_names[ci] == ps.category) {
                    cat_stats_v[ci].drafted  += prompt_drafted;
                    cat_stats_v[ci].accepted += prompt_accepted;
                    cat_stats_v[ci].rounds   += prompt_rounds;
                    found = true;
                    break;
                }
            }
            if (!found) {
                cat_names.push_back(ps.category);
                cat_stats_v.push_back({ prompt_drafted, prompt_accepted, prompt_rounds });
            }
        }

        const double accept_rate = prompt_drafted > 0 ? (double) prompt_accepted / (double) prompt_drafted : 0.0;
        const double tau         = prompt_rounds  > 0 ? (double) prompt_accepted / (double) prompt_rounds + 1.0 : 0.0;

        printf("[%2zu][%-9s] n_predicted=%-4d rounds=%-4lld drafted=%-5lld accepted=%-5lld accept=%.3f tau=%.3f "
                "ar_tok_s=%.2f sp_tok_s=%.2f speedup=%.3f\n",
                pi, ps.category.c_str(), n_predicted, (long long) prompt_rounds, (long long) prompt_drafted,
                (long long) prompt_accepted, accept_rate, tau, ar_tok_per_sec, sp_tok_per_sec, speedup);
        fflush(stdout);
    }

    printf("\n=== per-category ===\n");
    for (size_t ci = 0; ci < cat_names.size(); ++ci) {
        const auto & cs = cat_stats_v[ci];
        const double accept_rate = cs.drafted > 0 ? (double) cs.accepted / (double) cs.drafted : 0.0;
        const double tau         = cs.rounds  > 0 ? (double) cs.accepted / (double) cs.rounds + 1.0 : 0.0;
        printf("%-9s: rounds=%-5lld drafted=%-6lld accepted=%-6lld accept=%.4f tau=%.4f\n",
                cat_names[ci].c_str(), (long long) cs.rounds, (long long) cs.drafted, (long long) cs.accepted,
                accept_rate, tau);
    }

    printf("\n=== accept-by-depth (conditional: given position i was reached) ===\n");
    printf("%-6s %-9s %-9s %s\n", "depth", "reached", "accepted", "accept_i");
    for (int32_t i = 1; i <= n_draft; ++i) {
        const double r = depth_reached[i] > 0 ? (double) depth_accepted[i] / (double) depth_reached[i] : 0.0;
        printf("%-6d %-9lld %-9lld %.4f\n", i, (long long) depth_reached[i], (long long) depth_accepted[i], r);
    }

    common_speculative_print_stats(spec);

    const double accept_rate_all = total_drafted > 0 ? (double) total_accepted / (double) total_drafted : 0.0;
    const double tau_all         = total_rounds  > 0 ? (double) total_accepted / (double) total_rounds + 1.0 : 0.0;

    // aggregate tok/s: total tokens over total wall time across all prompts,
    // not a naive mean of per-prompt ratios (which would over-weight short
    // prompts) -- same convention as the per-category accept/tau rollup above.
    const double ar_tok_per_sec_all = total_ar_seconds > 0.0 ? total_ar_predicted / total_ar_seconds : 0.0;
    const double sp_tok_per_sec_all = total_sp_seconds > 0.0 ? total_predicted    / total_sp_seconds : 0.0;
    const double speedup_all        = ar_tok_per_sec_all > 0.0 ? sp_tok_per_sec_all / ar_tok_per_sec_all : 0.0;

    printf("\n=== OVERALL: prompts=%zu n_predicted=%lld rounds=%lld drafted=%lld accepted=%lld accept=%.4f tau=%.4f "
            "ar_tok_s=%.2f sp_tok_s=%.2f speedup=%.3f ===\n",
            prompts.size(), (long long) total_predicted, (long long) total_rounds, (long long) total_drafted,
            (long long) total_accepted, accept_rate_all, tau_all, ar_tok_per_sec_all, sp_tok_per_sec_all, speedup_all);

    llama_batch_free(batch_tgt);
    common_speculative_free(spec);
    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(model_dft);
    llama_model_free(model_tgt);
    llama_backend_free();

    return 0;
}

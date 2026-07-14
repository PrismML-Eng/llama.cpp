#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"
#include "gguf.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>

// dspark drafters carry the target layer-id list as an array-typed GGUF KV,
// which llama_model's string-KV cache skips -- read it from the drafter file
// directly (same as tools/server + tests/test-dspark-real-eval).
static std::vector<int32_t> read_dspark_target_layers(const std::string & drafter_path) {
    struct gguf_init_params gp = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
    gguf_context * gctx = gguf_init_from_file(drafter_path.c_str(), gp);
    if (gctx == nullptr) {
        return {};
    }

    std::vector<int32_t> out;

    const int64_t arch_kid = gguf_find_key(gctx, "general.architecture");
    if (arch_kid >= 0) {
        const std::string key = std::string(gguf_get_val_str(gctx, arch_kid)) + ".dspark.target_layers";
        const int64_t kid = gguf_find_key(gctx, key.c_str());
        if (kid >= 0 && gguf_get_kv_type(gctx, kid) == GGUF_TYPE_ARRAY) {
            const enum gguf_type arr_type = gguf_get_arr_type(gctx, kid);
            const size_t         n        = gguf_get_arr_n(gctx, kid);
            const void *         data     = gguf_get_arr_data(gctx, kid);
            out.reserve(n);
            for (size_t i = 0; i < n; i++) {
                switch (arr_type) {
                    case GGUF_TYPE_INT32:  out.push_back(((const int32_t  *) data)[i]); break;
                    case GGUF_TYPE_UINT32: out.push_back((int32_t) ((const uint32_t *) data)[i]); break;
                    case GGUF_TYPE_INT64:  out.push_back((int32_t) ((const int64_t  *) data)[i]); break;
                    case GGUF_TYPE_UINT64: out.push_back((int32_t) ((const uint64_t *) data)[i]); break;
                    default: out.clear(); i = n; break;
                }
            }
        }
    }

    gguf_free(gctx);
    return out;
}

// the drafter's block size (draft tokens per round); 0 on failure.
static uint32_t read_dspark_block_size(const std::string & drafter_path) {
    struct gguf_init_params gp = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
    gguf_context * gctx = gguf_init_from_file(drafter_path.c_str(), gp);
    if (gctx == nullptr) {
        return 0;
    }

    uint32_t out = 0;

    const int64_t arch_kid = gguf_find_key(gctx, "general.architecture");
    if (arch_kid >= 0) {
        const std::string key = std::string(gguf_get_val_str(gctx, arch_kid)) + ".dspark.block_size";
        const int64_t kid = gguf_find_key(gctx, key.c_str());
        if (kid >= 0 && gguf_get_kv_type(gctx, kid) == GGUF_TYPE_UINT32) {
            out = gguf_get_val_u32(gctx, kid);
        }
    }

    gguf_free(gctx);
    return out;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model
    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;

    // TODO: simplify this logic
    {
        const auto & params_spec = params.speculative.draft;

        auto params_dft = params;

        params_dft.devices      = params_spec.devices;
        params_dft.model        = params_spec.mparams;
        params_dft.n_gpu_layers = params_spec.n_gpu_layers;

        if (params_spec.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads       = params.speculative.draft.cpuparams.n_threads;
            params_dft.cpuparams_batch.n_threads = params.speculative.draft.cpuparams_batch.n_threads;
        }

        params_dft.tensor_buft_overrides = params.speculative.draft.tensor_buft_overrides;

        auto mparams_dft = common_model_params_to_llama(params_dft);

        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (model_dft == nullptr) {
            LOG_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
            return 1;
        }

        auto cparams = common_context_params_to_llama(params_dft);

        // dspark stages all context rows since its cache position PLUS a full
        // block in one batch (worst case ctx_len == n_ctx), so its batch must
        // cover n_ctx + block_size -- otherwise draft rounds near the context
        // limit are skipped and speculation silently degrades to AR.
        const bool spec_dspark = std::find(params.speculative.types.begin(),
                                           params.speculative.types.end(),
                                           COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) != params.speculative.types.end();
        if (spec_dspark) {
            const uint32_t block_size = read_dspark_block_size(params.speculative.draft.mparams.path);
            cparams.n_batch  = std::max(cparams.n_batch,  cparams.n_ctx + (block_size > 0 ? block_size : 64));
            cparams.n_ubatch = std::max(cparams.n_ubatch, cparams.n_batch);
        }

        ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft.get();
    }

    // check if the context supports partial sequence removal
    const bool use_ckpt_tgt = (common_context_can_seq_rm(ctx_tgt)       == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);
    const bool use_ckpt_dft = (common_context_can_seq_rm(ctx_dft.get()) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);

    if (use_ckpt_tgt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    llama_seq_id seq_id = 0;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    // init the speculator BEFORE the prompt is evaluated: capture-type drafters
    // (dspark) stage their context features from the prompt decode itself, and
    // their begin() clears the staging window -- so the order must be
    // begin() -> prompt decode -> process().
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, 1);

    const bool spec_capture = common_speculative_need_embd_capture(spec);
    if (spec_capture) {
        const std::vector<int32_t> capture_layers = read_dspark_target_layers(params.speculative.draft.mparams.path);
        if (capture_layers.empty()) {
            LOG_ERR("draft-dspark: failed to read dspark.target_layers from '%s'\n", params.speculative.draft.mparams.path.c_str());
            return 1;
        }
        // masked=false: capture stays dense (a row for every position) while
        // batch.logits stays narrow -- no per-row full-vocab lm_head (fork #63).
        llama_set_capture_layers(ctx_tgt, capture_layers.data(), capture_layers.size(), /* masked = */ false);
        LOG_INF("draft-dspark: target tap capture engaged on %zu layers\n", capture_layers.size());
    }

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    // eval the prompt
    if (spec_capture) {
        // begin() first (it clears the capture staging window), then an explicit
        // batch (positions + seq ids) so the drafter's process() can stage a
        // capture row for every prompt position; the drafter context is NOT
        // decoded directly -- its cache is managed inside draft().
        common_speculative_begin(spec, seq_id, prompt_tgt);

        common_batch_clear(batch_tgt);
        for (size_t i = 0; i < prompt_tgt.size(); ++i) {
            common_batch_add(batch_tgt, prompt_tgt[i], (llama_pos) i, { seq_id }, /* logits = */ false);
        }
        llama_decode(ctx_tgt, batch_tgt);
        if (!common_speculative_process(spec, batch_tgt)) {
            LOG_ERR("draft-dspark: common_speculative_process (prefill) failed\n");
            return 1;
        }
    } else {
        llama_decode(ctx_tgt,       llama_batch_get_one(inp.data(), inp.size() - 1));
        llama_decode(ctx_dft.get(), llama_batch_get_one(inp.data(), inp.size() - 1));

        common_speculative_begin(spec, seq_id, prompt_tgt);
    }

    size_t n_draft = 0;

    llama_tokens draft;
    common_prompt_checkpoint ckpt;

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // generate or reuse draft tokens
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (!spec_capture && use_ckpt_dft) {
                ckpt.update_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // generate a new draft
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ -1,
                /* .n_past     = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft, // output
            };
            common_speculative_draft(spec);

            // save the original draft size
            n_draft = draft.size();

            // save a checkpoint of the target context before evaluating the draft
            // this allows us to restore the state if partial draft acceptance occurs
            // (capture mode uses common_context_seq_rm rollback instead, like the
            // dspark eval harness -- and must never touch ctx_dft's state, which is
            // managed inside common_speculative_draft() itself)
            if (!spec_capture && !draft.empty()) {
                if (use_ckpt_tgt) {
                    ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }

            if (!spec_capture) {
                ckpt.load_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, ckpt.pos_max + 1, -1);
            }
        } else {
            // we have a previous (partial) draft to reuse from checkpoint restoration
            if (use_ckpt_tgt) {
                GGML_ASSERT(!ckpt.empty());
            }
        }

        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { seq_id }, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { seq_id }, true);
            }

            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

            llama_decode(ctx_tgt, batch_tgt);

            if (spec_capture) {
                // stage the verify rows' capture features for the next draft round
                if (!common_speculative_process(spec, batch_tgt)) {
                    LOG_ERR("draft-dspark: common_speculative_process (verify) failed\n");
                    return 1;
                }
            }
        }

        // evaluate the same batch with the draft model
        if (!spec_capture) {
            // TODO: extend to support MTP, Eagle, etc. See server code for reference
            // (dspark must NOT decode the verify batch on ctx_dft -- its drafter
            // cache is advanced inside common_speculative_draft() itself)
            llama_decode(ctx_dft.get(), batch_tgt);
        }

        // only save the sampler sampler state if we use checkpoints
        common_sampler_ptr smpl_save;
        if (!spec_capture && use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (!spec_capture && use_ckpt_tgt && ids.size() - 1 < draft.size()) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, draft.size());

            draft = std::move(ids);

            {
                ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            }

            {
                ckpt.load_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, ckpt.pos_max + 1, -1);
            }

            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            continue;
        }

        common_speculative_accept(spec, seq_id, ids.size() - 1);

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;

        if (spec_capture) {
            // drop the rejected tail of this round's verify batch from the target
            // cache (bounded partial rollback, also valid for hybrid GDN state);
            // dspark's own drafter cache was already cropped inside draft().
            common_context_seq_rm(ctx_tgt, seq_id, n_past, -1);
        }
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // clear the draft since it has been consumed
        draft.clear();

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            // must not ignore failure here: on a hybrid GDN/attention target
            // this is a bounded partial rollback of the recurrent state (see
            // common_params_speculative::need_n_rs_seq()), and a silently
            // ignored no-op would leave every round's rejected draft tail
            // permanently baked into the recurrent state instead of failing
            // loudly.
            common_context_seq_rm(ctx_tgt,       seq_id, n_past, -1);
            common_context_seq_rm(ctx_dft.get(), seq_id, n_past, -1);
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());

    llama_batch_free(batch_tgt);

    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}

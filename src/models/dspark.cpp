#include "models.h"

#include <cstdlib>
#include <cstring>

// Noncausal block drafter: target tap features provide context K/V while draft
// tokens pass through the dense trunk and sequential correction/Markov head.

void llama_model_dspark::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key(LLM_KV_DSPARK_BLOCK_SIZE, hparams.dspark_block_size, true);
    ml.get_key(LLM_KV_DSPARK_MASK_TOKEN_ID, hparams.dspark_mask_token_id, true);
    ml.get_key(LLM_KV_DSPARK_MARKOV_RANK, hparams.dspark_markov_rank, false);
    ml.get_key(LLM_KV_DSPARK_HIDDEN_CORRECTION, hparams.dspark_hidden_correction, false);
    ml.get_key(LLM_KV_DSPARK_MARKOV_DEPLOY_EXPOSURE, hparams.dspark_markov_deploy_exposure, false);
    if (hparams.dspark_hidden_correction) {
        ml.get_key(LLM_KV_DSPARK_CORRECTION_SIZE, hparams.dspark_correction_size, true);
        if (!hparams.dspark_markov_deploy_exposure || !hparams.dspark_correction_size || !hparams.dspark_markov_rank) {
            throw std::runtime_error(
                "DSpark hidden correction requires deploy-exposure Markov and positive dimensions");
        }
    } else if (hparams.dspark_markov_deploy_exposure) {
        throw std::runtime_error("DSpark deploy-exposure without hidden correction is not implemented");
    }
    ml.get_key(LLM_KV_DSPARK_CONFIDENCE_HEAD, hparams.dspark_confidence_head, false);
    ml.get_key(LLM_KV_DSPARK_CONFIDENCE_WITH_MARKOV, hparams.dspark_confidence_head_with_markov, false);

    // Noise-level conditioning is optional; older checkpoints omit it.
    ml.get_key(LLM_KV_DSPARK_LOG_SNR_CONDITIONING, hparams.dspark_log_snr_conditioning, false);
    if (hparams.dspark_log_snr_conditioning) {
        // required once the flag is set: silently defaulting either bound to 0.0f
        // would collapse (max_snr - min_snr) to zero in the featurizer below and
        // fill the conditioning input with NaNs instead of failing to load.
        ml.get_key(LLM_KV_DSPARK_MIN_LOG_SNR, hparams.dspark_min_log_snr, true);
        ml.get_key(LLM_KV_DSPARK_MAX_LOG_SNR, hparams.dspark_max_log_snr, true);
        if (!std::isfinite(hparams.dspark_min_log_snr) || !std::isfinite(hparams.dspark_max_log_snr)) {
            throw std::runtime_error("dspark log-SNR conditioning: min/max_log_snr must be finite");
        }
        if (!(hparams.dspark_max_log_snr > hparams.dspark_min_log_snr)) {
            throw std::runtime_error("dspark log-SNR conditioning: max_log_snr must be greater than min_log_snr");
        }
    }

    // ordered set of TARGET-model layer indices this drafter taps. note this
    // indexes into the target's (large) layer count, not this drafter's own
    // (tiny) n_layer -- get_arr_n first to learn the count, then get_arr to
    // fill the fixed-capacity array (llama_hparams must stay trivially
    // copyable, so a std::vector field isn't an option here).
    ml.get_arr_n(LLM_KV_DSPARK_TARGET_LAYERS, hparams.n_dspark_target_layers, true);
    ml.get_key_or_arr(LLM_KV_DSPARK_TARGET_LAYERS, hparams.dspark_target_layers, hparams.n_dspark_target_layers, true);

    // the drafter trunk itself has no dedicated size bucket (it's always tiny
    // relative to its target); leave it unclassified rather than overload an
    // unrelated bucket.
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dspark::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_capture   = hparams.n_dspark_target_layers;
    const int64_t n_embd_cap  = n_capture * n_embd;
    const int64_t markov_rank = hparams.dspark_markov_rank;

    // dspark ships no tokenizer of its own (converter calls _set_vocab_none():
    // it ties to the TARGET model's vocab), so vocab.n_tokens() (LLAMA_LOAD_LOCALS'
    // n_vocab) is 0 here. The real vocab width only exists as the token_embd
    // tensor's own shape in the GGUF -- peek at it before creating anything.
    ggml_tensor * tok_embd_meta  = ml.get_tensor_meta(tn(LLM_TENSOR_TOKEN_EMBD, "weight").str().c_str());
    const int64_t n_vocab_dspark = tok_embd_meta ? tok_embd_meta->ne[1] : n_vocab;
    GGML_ASSERT(n_vocab_dspark > 0 && "dspark: could not determine vocab size from token_embd.weight");

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab_dspark }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    if (params.dspark_head_source) {
        const auto & source = *params.dspark_head_source;
        if (!source.output || source.output->ne[0] != n_embd || source.output->ne[1] != n_vocab_dspark ||
            !source.output->buffer || source.output_s || source.output_in_s) {
            throw std::runtime_error("DSpark shared head requires a loaded, dimension-matched, unscaled target head");
        }
        // Account for the local tensor without allocating or reading its weights.
        create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab_dspark }, TENSOR_NOT_REQUIRED | TENSOR_SKIP);
        output = source.output;
        hdmd.weight_blocks.erase("output.weight");
        const auto rotation = source.hdmd.rot.find(output);
        if (rotation != source.hdmd.rot.end()) {
            hdmd.rot.emplace(output, rotation->second);
        }
    } else {
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab_dspark }, TENSOR_NOT_REQUIRED);
        if (!output) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab_dspark }, TENSOR_DUPLICATED);
        }
    }

    // target-feature projection: [n_capture * n_embd -> n_embd], then RMSNorm.
    dspark_fc          = create_tensor(tn(LLM_TENSOR_DSPARK_FC, "weight"), { n_embd_cap, n_embd }, 0);
    dspark_hidden_norm = create_tensor(tn(LLM_TENSOR_DSPARK_HIDDEN_NORM, "weight"), { n_embd }, 0);

    // auxiliary heads: loaded so the GGUF's tensor inventory is fully mapped
    if (hparams.dspark_hidden_correction) {
        const int64_t width = hparams.dspark_correction_size;
        if (hparams.dspark_mask_token_id >= n_vocab_dspark) {
            throw std::runtime_error("DSpark mask token outside vocabulary");
        }
        dspark_corr_hnorm = create_tensor(tn(LLM_TENSOR_DSPARK_CORR_HNORM, "weight"), { n_embd }, 0);
        dspark_corr_enorm = create_tensor(tn(LLM_TENSOR_DSPARK_CORR_ENORM, "weight"), { n_embd }, 0);
        dspark_corr_gate  = create_tensor(tn(LLM_TENSOR_DSPARK_CORR_GATE, "weight"), { 2 * n_embd, width }, 0);
        dspark_corr_up    = create_tensor(tn(LLM_TENSOR_DSPARK_CORR_UP, "weight"), { 2 * n_embd, width }, 0);
        dspark_corr_down  = create_tensor(tn(LLM_TENSOR_DSPARK_CORR_DOWN, "weight"), { width, n_embd }, 0);
    }
    // and available to a future Phase 2 host-side loop, but not built into
    // this graph (see file header comment).
    if (markov_rank > 0) {
        const int flags = hparams.dspark_hidden_correction ? 0 : TENSOR_NOT_REQUIRED;
        dspark_markov_head_a =
            create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_HEAD_A, "weight"), { markov_rank, n_vocab_dspark }, flags);
        dspark_markov_head_b =
            create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_HEAD_B, "weight"), { markov_rank, n_vocab_dspark }, flags);
    }
    if (hparams.dspark_confidence_head) {
        const int64_t conf_in = n_embd + (hparams.dspark_confidence_head_with_markov ? markov_rank : 0);
        dspark_confidence_head =
            create_tensor(tn(LLM_TENSOR_DSPARK_CONFIDENCE_HEAD, "weight"), { conf_in, 1 }, TENSOR_NOT_REQUIRED);
        dspark_confidence_head_b =
            create_tensor(tn(LLM_TENSOR_DSPARK_CONFIDENCE_HEAD, "bias"), { 1 }, TENSOR_NOT_REQUIRED);
    }

    dspark_mode_embedding =
        create_tensor(tn(LLM_TENSOR_DSPARK_MODE_EMBEDDING, "bias"), { n_embd }, TENSOR_NOT_REQUIRED);
    if (dspark_mode_embedding) {
        LLAMA_LOG_INFO("dspark: mode embedding active on draft-token input only\n");
    }

    // GIDD log-SNR conditioning (LogSnrEmbed): unlike markov_head/confidence_head
    // above, this IS built into this graph (graph::graph() below) -- it changes
    // the draft embedding every forward pass, not a deferred host-side
    // adjustment -- so if the GGUF says log_snr_conditioning is on, the weights
    // are REQUIRED: a missing tensor here is a broken conversion, not something
    // to silently degrade past.
    if (hparams.dspark_log_snr_conditioning) {
        const int64_t n_freq = 128;  // matches LogSnrEmbed.NUM_FREQ_FEATURES in dspark_snr.py
        dspark_log_snr_fc1_w = create_tensor(tn(LLM_TENSOR_DSPARK_PROD_LOG_SNR_FC1, "weight"), { n_freq, n_embd }, 0);
        dspark_log_snr_fc1_b = create_tensor(tn(LLM_TENSOR_DSPARK_PROD_LOG_SNR_FC1, "bias"), { n_embd }, 0);
        dspark_log_snr_fc2_w = create_tensor(tn(LLM_TENSOR_DSPARK_PROD_LOG_SNR_FC2, "weight"), { n_embd, n_embd }, 0);
        dspark_log_snr_fc2_b = create_tensor(tn(LLM_TENSOR_DSPARK_PROD_LOG_SNR_FC2, "bias"), { n_embd }, 0);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dspark::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_dspark::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int64_t n_embd_cap = (int64_t) hparams.n_dspark_target_layers * n_embd;

    // --- stage the raw target-tap context window -----------------------------
    // this is the piece that doesn't fit llama_batch.token/embd (different
    // width, different row count than the draft block) -- see llama_dspark_ctx
    // in llama-graph.h and llama_set_dspark_ctx() in llama-ext.h.
    //
    // llama_context builds trial graphs for compute-buffer sizing/warmup at
    // several points, not just once before the very first decode(): the
    // initial sched_reserve() at context-construction time, AND additional
    // internal reserve passes triggered from inside decode() itself (e.g. a
    // dummy single-token graph), using ubatch shapes that have nothing to do
    // with whatever the caller most recently staged via llama_set_dspark_ctx().
    // The only thing that reliably distinguishes "this graph build corresponds
    // to the context I just staged" from "this is some other trial/reserve
    // build" is shape consistency: the staged n_ctx_rows must actually fit
    // inside *this* graph's n_tokens. When it doesn't, fall back to the same
    // reserve-safe placeholder split used when nothing is staged at all --
    // mirrors how llm_graph_context::build_inp_cross_embd falls back to
    // hparams-derived sizes whenever llama_cross has no (matching) data.
    const bool have_staged_ctx = params.dspark_ctx && !params.dspark_ctx->v_ctx_feat.empty() &&
                                 params.dspark_ctx->n_ctx_rows > 0 && params.dspark_ctx->n_ctx_rows < n_tokens &&
                                 n_tokens - params.dspark_ctx->n_ctx_rows <= hparams.dspark_block_size;

    int64_t n_ctx_rows;
    if (have_staged_ctx) {
        n_ctx_rows = params.dspark_ctx->n_ctx_rows;
    } else {
        // reserve-time / shape-mismatched placeholder: leave exactly one row
        // for the draft block so n_draft is always >= 1, and exercise the
        // same concat/fc/hidden_norm topology as real usage whenever there's
        // more than one token to split.
        n_ctx_rows = std::max<int64_t>(n_tokens - hparams.dspark_block_size, 0);
    }

    const int64_t n_draft = n_tokens - n_ctx_rows;
    GGML_ASSERT(n_draft > 0 && "dspark: no rows left for the draft block");
    const char * reuse_env = std::getenv("LLAMA_DSPARK_REUSE_MASK");
    if (reuse_env && std::strcmp(reuse_env, "0") && std::strcmp(reuse_env, "1")) {
        throw std::runtime_error("LLAMA_DSPARK_REUSE_MASK must be 0 or 1");
    }
    const bool reuse_mask =
        have_staged_ctx && hparams.dspark_hidden_correction && reuse_env && std::strcmp(reuse_env, "1") == 0;

    ggml_tensor *                target_ctx   = nullptr;
    llm_graph_input_dspark_ctx * staged_input = nullptr;
    if (n_ctx_rows > 0) {
        auto ctx_input = std::make_unique<llm_graph_input_dspark_ctx>(params.dspark_ctx);
        staged_input   = ctx_input.get();
        if (reuse_mask) {
            ctx_input->reuse_mask_id = hparams.dspark_mask_token_id;
        }

        ctx_input->ctx_feat = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_cap, n_ctx_rows);
        ggml_set_input(ctx_input->ctx_feat);
        ggml_set_name(ctx_input->ctx_feat, "dspark_ctx_feat");

        ggml_tensor * raw_tap = ctx_input->ctx_feat;
        res->add_input(std::move(ctx_input));

        // fc + hidden_norm are applied ONCE for the whole call; the result is a
        // fixed input re-projected fresh through each layer's own k_proj/v_proj
        // below (it never itself passes through a layer's attn_norm/FFN).
        target_ctx = build_lora_mm(model.dspark_fc, raw_tap);
        cb(target_ctx, "dspark_fc", -1);
        target_ctx = build_norm(target_ctx, model.dspark_hidden_norm, nullptr, LLM_NORM_RMS, -1);
        cb(target_ctx, "dspark_hidden_norm", -1);
    }

    // --- draft-block token embeddings (the trunk residual stream) ------------
    // built over the FULL n_tokens width (like any other arch) then sliced to
    // the trailing n_draft columns: the leading n_ctx_rows columns would be
    // embeddings of whatever placeholder token id the caller put there and are
    // never used for anything.
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    inpL               = ggml_view_2d(ctx0, inpL, n_embd, n_draft, inpL->nb[1], inpL->nb[1] * n_ctx_rows);
    cb(inpL, "dspark_draft_embd", -1);

    // --- GIDD log-SNR conditioning (LogSnrEmbed, dspark_snr.py) --------------
    // added to the draft noise embedding BEFORE the layer loop, matching
    // Qwen3DSparkModelSnr._forward_backbone exactly. The per-position log-SNR
    // pattern is the fixed round-1 inference convention: anchor position of
    // each block (every block_size-th draft row, starting at 0) at
    // max_log_snr, every other (masked) position at min_log_snr -- this
    // drafter always operates on a full block_size-aligned draft block (see
    // file header comment), so n_draft is a multiple of block_size in every
    // real decode call.
    if (hparams.dspark_log_snr_conditioning) {
        const int64_t n_freq      = 128;
        const int64_t half        = n_freq / 2;
        const float   min_snr     = hparams.dspark_min_log_snr;
        const float   max_snr     = hparams.dspark_max_log_snr;
        const int64_t bsz         = hparams.dspark_block_size > 0 ? hparams.dspark_block_size : n_draft;
        // Opt-in until real-weight parity and backend-specific timing pass.
        const char *  compact_env = std::getenv("LLAMA_DSPARK_SNR_TWO_ROWS");
        const bool    compact_snr = compact_env && std::strcmp(compact_env, "1") == 0 && n_draft > 2 && n_draft <= bsz;
        const int64_t snr_rows    = compact_snr ? 2 : n_draft;

        // host-side: exact port of LogSnrEmbed.forward's featurization fused
        // with the anchor/mask pattern above. Both are pure functions of
        // n_draft/block_size/min_log_snr/max_log_snr -- all known here, no
        // runtime/ubatch data involved -- so precomputing on the host (rather
        // than chaining ggml_arange/ggml_sin/ggml_cos in-graph) keeps this
        // directly auditable line-for-line against the python reference.
        std::vector<float> feat((size_t) (n_freq * snr_rows));
        for (int64_t pos = 0; pos < snr_rows; ++pos) {
            const float log_snr = (pos % bsz == 0) ? max_snr : min_snr;
            const float t       = (log_snr - min_snr) / (max_snr - min_snr) * 1000.0f;
            for (int64_t i = 0; i < half; ++i) {
                const float freq                         = expf(-logf(10000.0f) * (float) i / (float) half);
                const float angle                        = t * freq;
                feat[(size_t) (pos * n_freq + i)]        = sinf(angle);
                feat[(size_t) (pos * n_freq + half + i)] = cosf(angle);
            }
        }

        auto logsnr_input  = std::make_unique<llm_graph_input_dspark_logsnr>(std::move(feat));
        logsnr_input->feat = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_freq, snr_rows);
        ggml_set_input(logsnr_input->feat);
        ggml_set_name(logsnr_input->feat, "dspark_log_snr_feat");
        ggml_tensor * snr_feat = logsnr_input->feat;
        res->add_input(std::move(logsnr_input));

        ggml_tensor * snr_hidden = build_lora_mm(model.dspark_log_snr_fc1_w, snr_feat);
        snr_hidden               = ggml_add(ctx0, snr_hidden, model.dspark_log_snr_fc1_b);
        snr_hidden               = ggml_silu(ctx0, snr_hidden);
        cb(snr_hidden, "dspark_log_snr_fc1", -1);

        ggml_tensor * snr_embed = build_lora_mm(model.dspark_log_snr_fc2_w, snr_hidden);
        snr_embed               = ggml_add(ctx0, snr_embed, model.dspark_log_snr_fc2_b);
        cb(snr_embed, "dspark_log_snr_fc2", -1);

        if (compact_snr) {
            auto * high  = ggml_view_2d(ctx0, snr_embed, n_embd, 1, snr_embed->nb[1], 0);
            auto * low   = ggml_view_2d(ctx0, snr_embed, n_embd, 1, snr_embed->nb[1], snr_embed->nb[1]);
            auto * masks = ggml_repeat_4d(ctx0, low, n_embd, n_draft - 1, 1, 1);
            snr_embed    = ggml_concat(ctx0, high, masks, 1);
            cb(snr_embed, "dspark_log_snr_expanded", -1);
        }

        inpL = ggml_add(ctx0, inpL, snr_embed);
        cb(inpL, "dspark_draft_embd_snr", -1);
    }

    if (model.dspark_mode_embedding) {
        inpL = ggml_add(ctx0, inpL, model.dspark_mode_embedding);
        cb(inpL, "dspark_draft_embd_mode", -1);
    }

    ggml_tensor * inp_pos  = build_inp_pos();
    auto *        inp_attn = build_attn_inp_kv();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    ggml_tensor * cur;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;  // [n_embd, n_draft]

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // concat the static target-context feature (unchanged across layers)
        // with this layer's normed draft residual, then project the WHOLE
        // thing through this layer's own k_proj/v_proj/q_proj in one shot.
        // nn.Linear has no cross-row terms, so k_proj(concat(A,B)) is exactly
        // concat(k_proj(A), k_proj(B)) -- this is equivalent to the reference's
        // separate-then-concat (k_ctx = k_proj(target); k_noise = k_proj(cur);
        // cat) while letting us reuse build_qkv()/build_attn() unmodified.
        // (target_ctx is null only in the degenerate n_ctx_rows == 0 case,
        // which real decode calls never hit -- see the have_staged_ctx block
        // above -- but can occur in llama_context's pre-decode buffer-reserve
        // trial graphs.)
        ggml_tensor * attn_in = target_ctx ? ggml_concat(ctx0, target_ctx, cur, 1) : cur;
        cb(attn_in, "dspark_attn_in", il);

        auto          qkv  = build_qkv(model.layers[il], attn_in, n_embd_head, n_head, n_head_kv, il);
        ggml_tensor * Qcur = qkv.q;
        ggml_tensor * Kcur = qkv.k;
        ggml_tensor * Vcur = qkv.v;

        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);
        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Kcur, "Kcur_normed", il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // real, persistent KV cache (build_attn_inp_kv(), not the no-cache
        // path): writes n_ctx_rows + n_draft new rows this call, matching the
        // reference's cache.update() growth. cparams.causal_attn must be set
        // to false by the caller (llama_set_causal_attn(ctx, false)) so the
        // mask built here is fully open (no hybrid causal/bidirectional
        // complexity -- same masking style as build_attn_inp_no_cache() with
        // causal_attn=false, just backed by a real growing cache instead).
        cur = build_attn(inp_attn, model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s, Qcur, Kcur, Vcur,
                         nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "dspark_attn_out_full", il);

        // discard the leading n_ctx_rows columns: those are attention output
        // for "queries" that don't exist in the reference (the context rows
        // never issue a query there). we computed them anyway because
        // build_attn()'s cache-write/mask machinery is uniformly sized to
        // n_tokens; dropping them here is provably harmless since attention
        // output is row-independent (each row depends only on its own Q row
        // against the shared K/V), so this is bit-identical to never having
        // computed them.
        cur = ggml_view_2d(ctx0, cur, n_embd, n_draft, cur->nb[1], cur->nb[1] * n_ctx_rows);
        cb(cur, "dspark_attn_draft_only", il);

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_inp = cur;

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur, model.layers[il].ffn_up, nullptr, model.layers[il].ffn_up_s, model.layers[il].ffn_gate,
                        nullptr, model.layers[il].ffn_gate_s, model.layers[il].ffn_down, nullptr,
                        model.layers[il].ffn_down_s, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (hparams.dspark_hidden_correction) {
        ggml_tensor *              trunk           = cur;
        ggml_tensor *              chain           = nullptr;
        ggml_tensor *              previous        = nullptr;
        ggml_tensor *              markov_block    = nullptr;
        ggml_tensor *              mask_markov     = nullptr;
        const char *               assembly_env    = std::getenv("LLAMA_DSPARK_BALANCED_OUTPUT");
        const bool                 balanced_output = assembly_env && std::strcmp(assembly_env, "1") == 0;
        std::vector<ggml_tensor *> output_rows;
        const char *               top_k_env = std::getenv("DSPARK_HEAD_TOP_K");
        char *                     top_k_end = nullptr;
        const int64_t              top_k     = top_k_env ? std::strtol(top_k_env, &top_k_end, 10) : 0;
        if ((top_k_env && (top_k_end == top_k_env || *top_k_end != '\0')) || top_k < 0 || top_k > model.output->ne[1]) {
            throw std::runtime_error("DSPARK_HEAD_TOP_K is outside the vocabulary");
        }
        if (top_k > 0 && model.hdmd.rot.count(model.output)) {
            throw std::runtime_error("DSpark restricted head does not support folded output weights");
        }
        if (top_k > 0 && (model.output_s || (loras && !loras->empty()))) {
            throw std::runtime_error("DSpark candidate heads do not support output scales or LoRA adapters");
        }
        if (top_k > 0) {
            auto ids     = ggml_view_1d(ctx0, res->t_inp_tokens, n_draft, n_ctx_rows * sizeof(int32_t));
            auto latent  = ggml_get_rows(ctx0, model.dspark_markov_head_a, ids);
            markov_block = build_lora_mm(model.dspark_markov_head_b, latent);
            cb(markov_block, "dspark_markov_block", -1);
        }
        ggml_tensor * base_block      = nullptr;
        ggml_tensor * candidate_block = nullptr;
        if (top_k > 0) {
            base_block      = build_lora_mm(model.output, trunk, model.output_s);
            candidate_block = ggml_top_k(ctx0, ggml_add(ctx0, base_block, markov_block), top_k);
            cb(candidate_block, "dspark_candidates", -1);
        }
        int64_t n_correct = n_draft;
        if (const char * value = std::getenv("LLAMA_DSPARK_CORRECTION_PREFIX")) {
            char *     end  = nullptr;
            const long keep = std::strtol(value, &end, 10);
            if (end == value || *end || keep < 1 || keep > hparams.dspark_block_size) {
                throw std::runtime_error("Invalid DSpark correction prefix");
            }
            // Keep the full trunk and batched projections; later correction slots cannot affect this prefix.
            n_correct = std::min<int64_t>(n_draft, keep);
            if (staged_input) {
                staged_input->correction_prefix_rows = n_correct;
            }
        }
        for (int64_t slot = 0; slot < n_correct; ++slot) {
            auto ids = ggml_view_1d(ctx0, res->t_inp_tokens, 1, (n_ctx_rows + slot) * sizeof(int32_t));
            if (slot % hparams.dspark_block_size == 0) {
                previous = ids;
            }
            auto hidden    = ggml_view_2d(ctx0, trunk, n_embd, 1, trunk->nb[1], slot * trunk->nb[1]);
            auto embedding = ggml_get_rows(ctx0, model.tok_embd, previous);
            auto hn        = build_norm(hidden, model.dspark_corr_hnorm, nullptr, LLM_NORM_RMS, -1);
            auto en        = build_norm(embedding, model.dspark_corr_enorm, nullptr, LLM_NORM_RMS, -1);
            auto input     = ggml_concat(ctx0, hn, en, 0);
            cb(input, "dspark_corr_input", slot);
            auto gate  = ggml_silu(ctx0, build_lora_mm(model.dspark_corr_gate, input));
            auto up    = build_lora_mm(model.dspark_corr_up, input);
            auto delta = build_lora_mm(model.dspark_corr_down, ggml_mul(ctx0, gate, up));
            cb(delta, "dspark_corr_delta", slot);
            auto          corrected  = ggml_add(ctx0, hidden, delta);
            ggml_tensor * candidates = nullptr;
            ggml_tensor * head       = model.output;
            if (candidate_block) {
                const char * fuse_gather = std::getenv("GGML_METAL_FUSE_GATHER_MV");
                if (fuse_gather && std::strcmp(fuse_gather, "1") == 0) {
                    ggml_build_forward_expand(gf, corrected);
                }
                candidates = ggml_view_1d(ctx0, candidate_block, top_k, slot * candidate_block->nb[1]);
                head       = ggml_get_rows(ctx0, model.output, candidates);
            }
            auto logits = candidate_block ? ggml_mul_mat(ctx0, head, corrected) :
                                            build_lora_mm(model.output, corrected, model.output_s);
            cb(logits, "dspark_corr_head", slot);
            ggml_tensor * markov;
            if (markov_block) {
                markov = ggml_view_2d(ctx0, markov_block, markov_block->ne[0], 1, markov_block->nb[1],
                                      slot * markov_block->nb[1]);
            } else if (reuse_mask && slot > 1) {
                markov = mask_markov;
            } else {
                auto latent = ggml_get_rows(ctx0, model.dspark_markov_head_a, ids);
                cb(latent, "dspark_corr_markov", slot);
                markov = build_lora_mm(model.dspark_markov_head_b, latent);
                if (reuse_mask && slot == 1) {
                    mask_markov = markov;
                }
            }
            if (candidates) {
                auto selected_markov = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, markov, 1, markov->ne[0]), candidates);
                logits               = ggml_add(ctx0, logits, ggml_reshape_2d(ctx0, selected_markov, top_k, 1));
                auto base =
                    ggml_view_2d(ctx0, base_block, base_block->ne[0], 1, base_block->nb[1], slot * base_block->nb[1]);
                auto dense = ggml_reshape_2d(ctx0, ggml_scale_bias(ctx0, base, 0.0f, -1e6f), 1, base->ne[0]);
                logits     = ggml_reshape_2d(
                    ctx0, ggml_set_rows(ctx0, dense, ggml_reshape_2d(ctx0, logits, 1, top_k), candidates), base->ne[0],
                    1);
            } else {
                logits = ggml_add(ctx0, logits, markov);
            }
            cb(logits, "dspark_corr_unmasked", slot);
            const size_t  mask_offset = hparams.dspark_mask_token_id * logits->nb[0];
            auto          mask        = ggml_clamp(ctx0, ggml_view_1d(ctx0, logits, 1, mask_offset), -1e6f, -1e6f);
            // Keep mask suppression out-of-place. SET with a source view of its
            // destination can corrupt the other logits on the Metal backend.
            const int64_t mask_id     = hparams.dspark_mask_token_id;
            ggml_tensor * masked      = mask;
            if (mask_id > 0) {
                masked = ggml_concat(ctx0, ggml_view_1d(ctx0, logits, mask_id, 0), masked, 0);
            }
            if (mask_id + 1 < logits->ne[0]) {
                masked = ggml_concat(
                    ctx0, masked,
                    ggml_view_1d(ctx0, logits, logits->ne[0] - mask_id - 1, (mask_id + 1) * logits->nb[0]), 0);
            }
            logits = masked;
            cb(logits, "dspark_corr_masked", slot);
            previous = ggml_argmax(ctx0, logits);
            res->t_dspark_greedy.push_back(previous);
            if (balanced_output) {
                output_rows.push_back(logits);
            } else {
                chain = chain ? ggml_concat(ctx0, chain, logits, 1) : logits;
            }
        }
        // Preserve row order without copying the growing prefix at every slot.
        while (output_rows.size() > 1) {
            std::vector<ggml_tensor *> next;
            for (size_t i = 0; i < output_rows.size(); i += 2) {
                next.push_back(i + 1 < output_rows.size() ? ggml_concat(ctx0, output_rows[i], output_rows[i + 1], 1) :
                                                            output_rows[i]);
            }
            output_rows = std::move(next);
        }
        if (balanced_output) {
            chain = output_rows.at(0);
            cb(chain, "dspark_balanced_output", -1);
            chain = ggml_view_2d(ctx0, chain, chain->ne[0], chain->ne[1], chain->nb[1], 0);
        }
        cur = chain;
    } else {
        cur = build_lora_mm(model.output, cur, model.output_s);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

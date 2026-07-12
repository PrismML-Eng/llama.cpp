#include "models.h"

#include <cmath>

// dspark: EAGLE-style block-diffusion speculative-decoding drafter.
//
// The trunk is a small, plain dense Qwen3-style stack (standard llama_layer
// attn_*/ffn_* tensors, loaded exactly like src/models/qwen3.cpp). What makes
// the per-layer body genuinely novel is the attention: at every layer, the
// drafter attends to two concatenated K/V sources built from DIFFERENT inputs:
//
//   1. a small, growing window of the TARGET model's captured multi-layer tap
//      features (concatenated across n_dspark_target_layers layers, projected
//      down to n_embd via dspark.fc + dspark.hidden_norm ONCE before the layer
//      loop) -- re-projected FRESH every layer via THAT layer's own k_proj/
//      v_proj. This is not a separate cross-attention block: it's the same
//      k_proj/v_proj the trunk already has, just applied to a second input.
//   2. the draft block's own evolving residual stream (block_size positions,
//      seeded from mask_token_id + the last accepted token), projected via the
//      same layer's k_proj/v_proj as usual.
//
// The two K/V sets are concatenated and attended to with a single, fully
// unmasked (non-causal) softmax -- no hybrid causal/bidirectional masking is
// needed (matches the reference: attention_mask=None, is_causal=False). Q only
// ever exists for the draft block: the context rows never issue a query, never
// run through the FFN, and never join the residual stream.
//
// Implementation note on how this maps onto llama.cpp's KV-cache attention
// primitives (build_attn_inp_kv()/build_attn()), which assume Q/K/V(new) all
// share one row count tied to ubatch.n_tokens: we set n_tokens = n_ctx_rows +
// n_draft (matching the reference's cache.update() growth exactly -- see
// docs/dspark-scope.md and the Python reference implementation in
// (reference implementation), compute Q/K/V
// uniformly over that full width each layer (so build_qkv()/build_attn() need
// no changes), then slice the attention OUTPUT back down to the trailing
// n_draft columns before the residual add. The "wasted" query rows computed
// for the context positions are discarded immediately and are provably
// harmless: attention output is row-independent (each output row depends only
// on its own Q row dotted against the shared K/V), so slicing them away here
// is bit-identical to never having computed them.
//
// The persistent, growing KV cache itself (real llama_kv_cache_unified, not a
// no-cache arch) is what makes the incremental multi-round drafting loop
// possible; that loop (advance-by-n_accepted, crop the cache back to `start`)
// is Phase 2 (common/speculative.cpp) and is NOT built here. Likewise the
// markov_head / confidence_head auxiliary tensors are loaded below but not
// wired into this graph: their application is a small, sequential, host-side
// resample loop (see docs/dspark-scope.md's block-draft loop design and the
// ground-truth note about never batching the markov resample over
// mask_token_id) that belongs to Phase 2. res->t_logits here is the BASE
// trunk logits (dspark.fc -> trunk -> output_norm -> output), pre markov-bias.

void llama_model_dspark::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key(LLM_KV_DSPARK_BLOCK_SIZE,    hparams.dspark_block_size,    true);
    ml.get_key(LLM_KV_DSPARK_MASK_TOKEN_ID, hparams.dspark_mask_token_id, true);
    ml.get_key(LLM_KV_DSPARK_MARKOV_RANK,   hparams.dspark_markov_rank,   false);
    ml.get_key(LLM_KV_DSPARK_CONFIDENCE_HEAD,             hparams.dspark_confidence_head,             false);
    ml.get_key(LLM_KV_DSPARK_CONFIDENCE_WITH_MARKOV,      hparams.dspark_confidence_head_with_markov,  false);

    // GIDD log-SNR conditioning: optional metadata, absent (and defaulted off)
    // on drafters not trained with it, which must keep loading unchanged. When
    // it is enabled the bounds are required and drive a divide in the
    // featurization (t = (log_snr - min) / (max - min)), so they must be present,
    // finite, and strictly ordered -- otherwise the embedding is silently NaN.
    ml.get_key(LLM_KV_DSPARK_LOG_SNR_CONDITIONING, hparams.dspark_log_snr_conditioning, false);
    if (hparams.dspark_log_snr_conditioning) {
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
    ggml_tensor * tok_embd_meta = ml.get_tensor_meta(tn(LLM_TENSOR_TOKEN_EMBD, "weight").str().c_str());
    const int64_t n_vocab_dspark = tok_embd_meta ? tok_embd_meta->ne[1] : n_vocab;
    GGML_ASSERT(n_vocab_dspark > 0 && "dspark: could not determine vocab size from token_embd.weight");

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab_dspark }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab_dspark }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        // dspark's lm_head is a frozen copy of the target's, not tied to
        // token_embd, but fall back the same way other dense arches do in
        // case a future export ties them.
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab_dspark }, TENSOR_DUPLICATED);
    }

    // target-feature projection: [n_capture * n_embd -> n_embd], then RMSNorm.
    dspark_fc          = create_tensor(tn(LLM_TENSOR_DSPARK_FC,          "weight"), { n_embd_cap, n_embd }, 0);
    dspark_hidden_norm = create_tensor(tn(LLM_TENSOR_DSPARK_HIDDEN_NORM, "weight"), { n_embd }, 0);

    // auxiliary heads: loaded so the GGUF's tensor inventory is fully mapped
    // and available to a future Phase 2 host-side loop, but not built into
    // this graph (see file header comment).
    if (markov_rank > 0) {
        dspark_markov_head_a = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_HEAD_A, "weight"), { markov_rank, n_vocab_dspark }, TENSOR_NOT_REQUIRED);
        dspark_markov_head_b = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_HEAD_B, "weight"), { markov_rank, n_vocab_dspark }, TENSOR_NOT_REQUIRED);
    }
    if (hparams.dspark_confidence_head) {
        const int64_t conf_in = n_embd + (hparams.dspark_confidence_head_with_markov ? markov_rank : 0);
        dspark_confidence_head   = create_tensor(tn(LLM_TENSOR_DSPARK_CONFIDENCE_HEAD, "weight"), { conf_in, 1 }, TENSOR_NOT_REQUIRED);
        dspark_confidence_head_b = create_tensor(tn(LLM_TENSOR_DSPARK_CONFIDENCE_HEAD, "bias"),   { 1 },         TENSOR_NOT_REQUIRED);
    }

    // GIDD log-SNR conditioning (LogSnrEmbed): unlike markov_head/confidence_head
    // above, this IS built into the forward graph (graph::graph() below) -- it
    // changes the draft embedding every forward pass, not a deferred host-side
    // adjustment -- so if the GGUF says log_snr_conditioning is on, the weights
    // are REQUIRED. A missing tensor here is a broken conversion, not something
    // to silently degrade past (a drafter trained with conditioning that runs
    // without it produces wrong drafts).
    if (hparams.dspark_log_snr_conditioning) {
        const int64_t n_freq = 128; // sinusoidal feature count (LogSnrEmbed)
        dspark_log_snr_fc1_w = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC1, "weight"), { n_freq, n_embd }, 0);
        dspark_log_snr_fc1_b = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC1, "bias"),   { n_embd },         0);
        dspark_log_snr_fc2_w = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC2, "weight"), { n_embd, n_embd }, 0);
        dspark_log_snr_fc2_b = create_tensor(tn(LLM_TENSOR_DSPARK_LOG_SNR_FC2, "bias"),   { n_embd },         0);
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
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
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
    const bool have_staged_ctx =
        params.dspark_ctx && !params.dspark_ctx->v_ctx_feat.empty() &&
        params.dspark_ctx->n_ctx_rows > 0 && params.dspark_ctx->n_ctx_rows < n_tokens;

    int64_t n_ctx_rows;
    if (have_staged_ctx) {
        n_ctx_rows = params.dspark_ctx->n_ctx_rows;
    } else {
        // reserve-time / shape-mismatched placeholder: leave exactly one row
        // for the draft block so n_draft is always >= 1, and exercise the
        // same concat/fc/hidden_norm topology as real usage whenever there's
        // more than one token to split.
        n_ctx_rows = std::max<int64_t>(n_tokens - 1, 0);
    }

    const int64_t n_draft = n_tokens - n_ctx_rows;
    GGML_ASSERT(n_draft > 0 && "dspark: no rows left for the draft block");

    ggml_tensor * target_ctx = nullptr;
    if (n_ctx_rows > 0) {
        auto ctx_input = std::make_unique<llm_graph_input_dspark_ctx>(params.dspark_ctx);

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
    inpL = ggml_view_2d(ctx0, inpL, n_embd, n_draft, inpL->nb[1], inpL->nb[1] * n_ctx_rows);
    cb(inpL, "dspark_draft_embd", -1);

    // --- GIDD log-SNR conditioning (LogSnrEmbed) ----------------------------
    // added to the draft noise embedding BEFORE the layer loop. The per-position
    // log-SNR pattern is the fixed round-1 inference convention: the anchor
    // position of each block (every block_size-th draft row, starting at 0) is
    // set to max_log_snr, every other (masked) position to min_log_snr -- this
    // drafter always operates on a full block_size-aligned draft block (see the
    // file header), so n_draft is a multiple of block_size in every real decode.
    if (hparams.dspark_log_snr_conditioning) {
        const int64_t n_freq   = 128;
        const int64_t half     = n_freq / 2;
        const float   min_snr  = hparams.dspark_min_log_snr;
        const float   max_snr  = hparams.dspark_max_log_snr;
        const int64_t bsz      = hparams.dspark_block_size > 0 ? hparams.dspark_block_size : n_draft;

        // host-side: the sinusoidal featurization fused with the anchor/mask
        // pattern above. Both are pure functions of n_draft/block_size/min/max
        // log-SNR -- no runtime/ubatch data -- so precomputing on the host
        // (rather than chaining ggml_arange/sin/cos in-graph) keeps this
        // directly auditable against the reference implementation. The loader
        // guarantees max_snr > min_snr (both finite), so the divide is safe.
        std::vector<float> feat((size_t) (n_freq * n_draft));
        for (int64_t pos = 0; pos < n_draft; ++pos) {
            const float log_snr = (pos % bsz == 0) ? max_snr : min_snr;
            const float t       = (log_snr - min_snr) / (max_snr - min_snr) * 1000.0f;
            for (int64_t i = 0; i < half; ++i) {
                const float freq  = expf(-logf(10000.0f) * (float) i / (float) half);
                const float angle = t * freq;
                feat[(size_t) (pos * n_freq + i)]        = sinf(angle);
                feat[(size_t) (pos * n_freq + half + i)] = cosf(angle);
            }
        }

        auto logsnr_input = std::make_unique<llm_graph_input_dspark_logsnr>(std::move(feat));
        logsnr_input->feat = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_freq, n_draft);
        ggml_set_input(logsnr_input->feat);
        ggml_set_name(logsnr_input->feat, "dspark_log_snr_feat");
        ggml_tensor * snr_feat = logsnr_input->feat;
        res->add_input(std::move(logsnr_input));

        ggml_tensor * snr_hidden = build_lora_mm(model.dspark_log_snr_fc1_w, snr_feat);
        snr_hidden = ggml_add(ctx0, snr_hidden, model.dspark_log_snr_fc1_b);
        snr_hidden = ggml_silu(ctx0, snr_hidden);
        cb(snr_hidden, "dspark_log_snr_fc1", -1);

        ggml_tensor * snr_embed = build_lora_mm(model.dspark_log_snr_fc2_w, snr_hidden);
        snr_embed = ggml_add(ctx0, snr_embed, model.dspark_log_snr_fc2_b);
        cb(snr_embed, "dspark_log_snr_fc2", -1);

        inpL = ggml_add(ctx0, inpL, snr_embed);
        cb(inpL, "dspark_draft_embd_snr", -1);
    }

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    ggml_tensor * cur;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL; // [n_embd, n_draft]

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

        auto qkv = build_qkv(model.layers[il], attn_in, n_embd_head, n_head, n_head_kv, il);
        ggml_tensor * Qcur = qkv.q;
        ggml_tensor * Kcur = qkv.k;
        ggml_tensor * Vcur = qkv.v;

        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);
        Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Kcur, "Kcur_normed", il);

        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
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
        cur = build_attn(inp_attn,
                model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
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

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;
    cb(cur, "h_pre_norm", -1);
    res->t_h_nextn = cur;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // base trunk logits (pre markov-bias -- see file header comment).
    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

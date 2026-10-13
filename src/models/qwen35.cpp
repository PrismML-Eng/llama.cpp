#include "models.h"
#include "llama-impl.h"
#include "llama-memory-recurrent.h"
#include "llama-kv-cache.h"

void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // NextN/MTP (Qwen3.5/3.6): extra decoder block appended beyond the main stack
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < n_layer_impl");

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        // MTP block looks like a full-attention Qwen3.5 decoder block.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, mtp_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, mtp_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, mtp_flags);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
        // ARC-LAB: optional trimmed draft head (FR-Spec style): the draft only proposes the K most frequent tokens
        if (const ggml_tensor * dh = ml.get_tensor_meta(tn(LLM_TENSOR_NEXTN_DRAFT_HEAD, "weight", il).str().c_str())) {
            const int64_t n_draft_vocab = dh->ne[1];
            layer.nextn.draft_head     = create_tensor(tn(LLM_TENSOR_NEXTN_DRAFT_HEAD,     "weight", il), { n_embd, n_draft_vocab }, mtp_flags|TENSOR_NOT_REQUIRED);
            layer.nextn.draft_head_ids = create_tensor(tn(LLM_TENSOR_NEXTN_DRAFT_HEAD_IDS, "weight", il), { n_draft_vocab },         mtp_flags|TENSOR_NOT_REQUIRED);
        }
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen35::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // one pass over the devices for the per-layer path choices; ggml_backend_dev_type used to be a
    // full cudaGetDeviceProperties per call, and this ran twice per recurrent layer
    for (const auto & ldev : model.devices) {
        if (ldev.dev == nullptr) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ldev.dev);
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
        if (reg_name == nullptr) {
            gdn_state_rows_dev_ok = false;
            gdn_raw_gates_dev_ok  = false;
            break;
        }
        // integrated GPUs (e.g. unified-memory CUDA devices) report IGPU, not GPU
        const bool is_gpu = ggml_backend_dev_type(ldev.dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
                            ggml_backend_dev_type(ldev.dev) == GGML_BACKEND_DEVICE_TYPE_IGPU;
        if (is_gpu && strcmp(reg_name, "MTL") != 0) {
            gdn_state_rows_dev_ok = false;
        }
        if (strcmp(reg_name, "MTL") != 0 && strcmp(reg_name, "CUDA") != 0 &&
            strcmp(reg_name, "ROCm") != 0 && strcmp(reg_name, "MUSA") != 0 && strcmp(reg_name, "CPU") != 0) {
            gdn_raw_gates_dev_ok = false;
        }
    }

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // multi-layer hidden-state tap: collect the captured layer outputs here in
    // capture order, then concatenate them along dim0 after the layer loop.
    std::vector<ggml_tensor *> h_capture(cparams.n_capture_layers, nullptr);

    // Only unmasked nextn extraction needs a hidden row for every token; otherwise the
    // last layer runs on the output rows alone (capture taps and layer inputs are taken
    // before it, or sliced to the same rows).
    const bool narrow_last = inp_out_ids && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked);

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && narrow_last) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;

        // multi-layer hidden-state tap: if this layer index is registered for
        // capture, slice it to the requested output rows (masked layout) and
        // stash it in the matching capture slot. Slot order == capture order, so
        // the post-loop concat width is [n_capture * n_embd] in the order the
        // caller requested, independent of the order layers are visited.
        for (uint32_t c = 0; c < cparams.n_capture_layers; ++c) {
            if (cparams.capture_layer_idx[c] == il) {
                ggml_tensor * cap = cur;
                if (cparams.embeddings_nextn_masked && inp_out_ids && !(narrow_last && il == n_layer - 1)) {
                    cap = ggml_get_rows(ctx0, cap, inp_out_ids);
                }
                cb(cap, "h_capture", il);
                h_capture[c] = cap;
            }
        }
    }
    cur = inpL;
    // multi-layer hidden-state tap: concatenate captured layers along dim0 into a
    // single [n_capture * n_embd, n_outputs] tensor for one bulk host copy.
    if (cparams.n_capture_layers > 0) {
        ggml_tensor * cap = h_capture[0];
        GGML_ASSERT(cap && "capture layer 0 was not produced (index out of executed range?)");
        for (uint32_t c = 1; c < cparams.n_capture_layers; ++c) {
            GGML_ASSERT(h_capture[c] && "a requested capture layer was not produced");
            cap = ggml_concat(ctx0, cap, h_capture[c], 0);
        }
        cb(cap, "h_capture_cat", -1);
        res->t_h_capture = cap;

        // The capture concat chain is a side-branch off the per-layer outputs,
        // not reachable by traversing backward from the logits tensor expanded
        // below -- without this it's built but never added to gf, so the
        // scheduler never visits or backend-assigns it (ggml_set_output() alone
        // marks intent, it doesn't add the node to the graph).
        ggml_build_forward_expand(gf, cap);
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (inp_out_ids && !narrow_last) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);

    // silu(gate) * normalized as one GLU op instead of a unary and a mul
    return ggml_swiglu_split(ctx0, gate, normalized);
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    ggml_tensor * beta_raw = beta;

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    // the fused GDN op can apply sigmoid / softplus itself; hand it the raw projections and
    // the activated nodes below only stay in the graph on paths that still need them
    // only where the fused op implements raw gates natively (CPU, Metal, CUDA/ROCm); other
    // backends would fall back to the CPU for the whole op, which costs more than the four launches
    static const bool raw_gates_disable = getenv("GGML_GDN_RAW_GATES_DISABLE") != nullptr;
    if (!raw_gates_disable && gdn_raw_gates_dev_ok &&
        model.layers[il].ssm_dt && model.layers[il].ssm_dt->type == GGML_TYPE_F32 &&
        model.layers[il].ssm_a  && model.layers[il].ssm_a->type  == GGML_TYPE_F32) {
        gdn_raw_beta    = beta_raw;
        gdn_raw_alpha   = ggml_reshape_4d(ctx0, alpha, 1, num_v_heads, n_seq_tokens, n_seqs);
        gdn_raw_dt_bias = model.layers[il].ssm_dt;
        gdn_raw_a       = model.layers[il].ssm_a;
    } else {
        gdn_raw_beta = gdn_raw_alpha = gdn_raw_dt_bias = gdn_raw_a = nullptr;
    }

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    // ring path: read per-seq live state directly from the cache inside the
    // fused GDN op (rows mode) instead of a gather per layer.
    // GGML_GDN_STATE_GATHER=1 restores the legacy gathered path (A/B).
    // rows mode (the src[6] variant) is implemented on CPU and Metal only;
    // other GPU backends reject it in supports_op, which would silently move
    // the whole recurrent op to CPU -- keep the gathered form unless every
    // GPU device in the model is Metal.
    static const bool gdn_state_rows_env = getenv("GGML_GDN_STATE_GATHER") == nullptr;

    // ARC-LAB: GDN rollback by replay (the memory decides, see llama_memory_recurrent::rpl): the state is the base
    // row in plane 0, gathered with its own row indices (s_copy carries the conv rollback plane)
    const bool gdn_rpl = inp->s_rpl_ctrl != nullptr;

    const bool gdn_state_rows = gdn_state_rows_env && gdn_state_rows_dev_ok && cparams.n_rs_seq > 0 && !gdn_rpl;

    ggml_tensor * state;
    if (gdn_rpl) {
        state = build_rs(ssm_states_all, inp->s_rpl_main, inp->s_rpl_extra, hparams.n_embd_s(), n_seqs,
                mctx_cur->get_n_rs(), mctx_cur->get_head(), mctx_cur->get_size(), mctx_cur->get_rs_z());
        state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
        cb(state, "state_predelta", il);
    } else if (gdn_state_rows) {
        state = build_rs_cache_view(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
        cb(state, "state_cache_view", il);
    } else {
        state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
        state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
        cb(state, "state_predelta", il);
    }

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    // q and k are adjacent head groups of the same width in the conv output, so one
    // l2_norm over the joint view normalises both; q and k are then views into its result
    ggml_tensor * qk_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, 2 * num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);
    qk_conv = ggml_l2_norm(ctx0, qk_conv, eps_norm);
    cb(qk_conv, "qk_conv_l2", il);

    q_conv = ggml_view_4d(ctx0, qk_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            qk_conv->nb[1], qk_conv->nb[2], qk_conv->nb[3], 0);
    k_conv = ggml_view_4d(ctx0, qk_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            qk_conv->nb[1], qk_conv->nb[2], qk_conv->nb[3], num_k_heads * qk_conv->nb[1]);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il,
            gdn_state_rows ? inp->s_copy_main : nullptr);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}

// ARC-LAB: per-step inputs of a chained MTP draft graph (graph_mtp::build_chain), packed into two tensors so a draft
// uploads them in two writes: pos holds each step's positions in the layout build_inp_pos() gives a decode of just
// that step's rows (step after step), out_idx the output row of each step that gathers one. The graph reads them through per-step views.
class llm_graph_input_mtp_chain : public llm_graph_input_i {
public:
    explicit llm_graph_input_mtp_chain(uint32_t n_pos_per_embd) : n_pos_per_embd(n_pos_per_embd) {}

    struct step {
        int64_t  r0;   // first ubatch row of the step
        int64_t  nr;   // rows
        uint32_t n_kv; // KV length the step attends over (baked into the graph's views)
        bool     gather; // picks its output row with a GET_ROWS (else its single row is the output)
    };
    std::vector<step> steps;
    int64_t       n_tokens = 0;
    ggml_tensor * pos      = nullptr; // I32 [n_tokens * n_pos_per_embd]
    ggml_tensor * out_idx  = nullptr; // I32 [gathering steps]: nr - 1 (the step's last row produces its draft); null: none
    bool          out_in_pos = false; // the output rows follow the positions in pos (one upload), out_idx unused

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ubatch->n_tokens == n_tokens);
        std::vector<int32_t> v((size_t) n_tokens * n_pos_per_embd, 0);
        std::vector<int32_t> o;
        size_t off = 0;
        for (size_t s = 0; s < steps.size(); ++s) {
            const auto & st = steps[s];
            int32_t * vs = v.data() + off;
            for (int64_t r = 0; r < st.nr; ++r) {
                if (ubatch->token && n_pos_per_embd == 4) {
                    // text tokens under M-RoPE, as llm_graph_input_pos: three equal dims, the fourth 0
                    for (int d = 0; d < 3; ++d) {
                        vs[d * st.nr + r] = ubatch->pos[st.r0 + r];
                    }
                } else {
                    for (uint32_t d = 0; d < n_pos_per_embd; ++d) {
                        vs[d * st.nr + r] = ubatch->pos[d * n_tokens + st.r0 + r];
                    }
                }
            }
            off += (size_t) st.nr * n_pos_per_embd;
            if (st.gather) {
                o.push_back((int32_t) st.nr - 1);
            }
        }
        if (out_in_pos) {
            v.insert(v.end(), o.begin(), o.end());
        }
        ggml_backend_tensor_set(pos, v.data(), 0, v.size() * sizeof(int32_t));
        if (out_idx && !out_in_pos) {
            ggml_backend_tensor_set(out_idx, o.data(), 0, o.size() * sizeof(int32_t));
        }
    }

    // reusable when the batch has the same rows and every step would attend over the same KV length (the rest of the
    // topology is checked by allow_reuse and the other inputs)
    bool can_reuse(const llm_graph_params & params) override {
        if (params.ubatch.n_tokens != n_tokens || params.cparams.mtp_chain_n != steps.size()) {
            return false;
        }
        const auto * kvc = static_cast<const llama_kv_cache_context *>(params.mctx);
        for (const auto & st : steps) {
            if (kvc->get_n_kv_prefix((uint32_t) (st.r0 + st.nr)) != st.n_kv) {
                return false;
            }
        }
        return true;
    }

    const uint32_t n_pos_per_embd;
};

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
llama_model_qwen35::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    if (cparams.mtp_chain_n > 0) {
        build_chain(model);
        return;
    }

    // hparams.n_layer includes both main model layers and MTP layers. The MTP
    // layer is stored immediately after the main layers in model.layers[].
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);

        // a Hadamard-latent embedding table stores rotated rows; restore the primal
        // basis right after the lookup (h = s * (H z)), exactly like the trunk path in
        // llm_graph_context::build_inp_embd. Without this the draft head consumes
        // embeddings in the rotated basis and llama_verify_hadamard_graph refuses the graph.
        if (hadamard_inverses) {
            const auto it = hadamard_inverses->find(tok_embd_w);
            if (it != hadamard_inverses->end()) {
                tok_embd = llama_mul_mat_hadamard(ctx0, tok_embd, it->second.rot);
                if (it->second.signs) {
                    tok_embd = ggml_mul(ctx0, tok_embd, it->second.signs);
                }
            }
        }
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    // ARC-LAB: KV-only pass (LLAMA_MTP_KV_ONLY, default on with a SYCL device; independent implementation of the technique described by
    // GRIMOIRE). A decode with no output rows (the head's prompt catch-up) is only there to fill this layer's K/V, so
    // build up to the cache store and stop: no query, attention, output projection, FFN or head. The K/V ops are the
    // full graph's, in the same order, so the cache rows are bit-identical. Unmasked nextn readers copy h for every
    // token, so they keep the full graph.
    static const bool kv_only_on = [] {
        const char * e = getenv("LLAMA_MTP_KV_ONLY");
        return e ? atoi(e) != 0 : llama_sycl_present();
    }();
    const bool kv_only = kv_only_on && n_outputs == 0 && cparams.embeddings_nextn_masked;

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = kv_only ? nullptr : build_inp_out_ids();  // an unread input would have no buffer

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    if (kv_only) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            LLAMA_LOG_INFO("%s: MTP KV-only pass for decodes without outputs (LLAMA_MTP_KV_ONLY=0 turns it off)\n", __func__);
        }
        // the SYCL backend runs same-input mat-vecs of <= 8 rows as one launch whose tile choice depends on the
        // largest member; keep the Q projection (result unused) so q + k + v group exactly as in the full graph
        if (n_tokens <= 8) {
            ggml_build_forward_expand(gf, build_lora_mm(layer.wq, cur, layer.wq_s));
        }

        ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Kcur, "mtp_Kcur_normed", il);

        ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
        cb(Vcur, "mtp_Vcur", il);

        Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

        // the store half of build_attn(): same optional cache rotations, same expand order (v, k, then the copies)
        if (inp_attn->self_k_rot) {
            Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot);
        }
        if (inp_attn->self_v_rot) {
            Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot);
        }
        ggml_build_forward_expand(gf, Vcur);
        ggml_build_forward_expand(gf, Kcur);
        cb(Kcur, "k_cache_in", il);
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
        return;
    }

    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);

    // llama_context copies the first n_outputs rows of t_h_nextn when embeddings_nextn_masked is set,
    // so publish the output rows only (as the trunk graph does); unmasked readers index by token.
    // Matters as soon as an MTP batch has more tokens than outputs, e.g. catch-up rows decoded
    // together with the first draft row.
    if (cparams.embeddings_nextn_masked) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        res->t_h_nextn = cur;
    } else {
        res->t_h_nextn = cur;
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    static const bool draft_head_off = getenv("LLAMA_MTP_DRAFT_HEAD_OFF") != nullptr;
    if (layer.nextn.draft_head && layer.nextn.draft_head_ids && !layer.nextn.shared_head_head && !draft_head_off) {
        // ARC-LAB: logits for the top-K frequent tokens only, scattered into a full-vocab row of -inf; the target
        // model still verifies every draft, so this changes only which tokens can be drafted, not the output
        const int64_t n_k   = layer.nextn.draft_head->ne[1];
        const int64_t n_out = cur->ne[1];
        const int64_t n_voc = head_w->ne[1];
        ggml_tensor * lk   = build_lora_mm(layer.nextn.draft_head, cur, nullptr);                   // [K, n_out]
        ggml_tensor * full = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_voc, n_out);
        full = ggml_fill(ctx0, full, -INFINITY);
        full = ggml_set_rows(ctx0, full, ggml_reshape_3d(ctx0, lk, 1, n_k, n_out),
                             ggml_reshape_2d(ctx0, layer.nextn.draft_head_ids, n_k, 1));
        cur = ggml_reshape_2d(ctx0, full, n_voc, n_out);
    } else {
        cur = build_lora_mm(head_w, cur, head_s);
    }
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}

// ARC-LAB: chained MTP drafts, n = cparams.mtp_chain_n steps in one graph (independent implementation of the technique
// described by llamADALL). The ubatch holds the deferred catch-up rows and the anchor (step 0, as the first decode of the
// draft loop in common/speculative.cpp) followed by one placeholder row per later step, so the KV cache has given every
// draft position its cell and the mask rows exist. Step j > 0 reads step j-1's argmax token and output state (h_nextn
// row) on the device and writes its K/V into its row's cell, as decode j of the loop does.
//
// Each step is built with the loop's shapes, so the kernels and their rounding are the loop's: its own rows only (step 0
// the catch-up rows + anchor, later steps one row), its own positions, KV index slices and mask rows, and K/V views and
// mask columns of the KV length a decode of the rows up to and including its own would see (get_n_kv_prefix; flash
// attention splits the KV range by that length). The ops mirror the single-step graph above; keep the two in sync.
void llama_model_qwen35::graph_mtp::build_chain(const llama_model & model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int     il          = hparams.n_layer();
    const auto &  layer       = model.layers[il];

    const int64_t n_steps = cparams.mtp_chain_n;
    GGML_ASSERT(n_steps >= 1 && n_tokens >= n_steps);
    GGML_ASSERT(ubatch.token && ubatch.embd && "MTP chain needs tokens and hidden rows");
    const int64_t n_lead = n_tokens - n_steps + 1;  // step 0: catch-up rows + the anchor

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // inputs as the single-step graph creates them
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);
    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");
    ggml_tensor * tokens_all = inp->tokens;
    ggml_tensor * h_all      = inp->h;
    res->add_input(std::move(inp));

    auto * inp_attn = build_attn_inp_kv();
    const llama_kv_cache_context * kvc = inp_attn->mctx;

    // ARC-LAB: LLAMA_MTP_CHAIN_TIGHT (bit mask, default all on; 0 = the first chain layout):
    //   1: every per-step input slice is made once, up front, from ONE device copy of each packed input. A view of a
    //      host input is its own scheduler split input (one blocking host->device copy per view and step), and a view
    //      built lazily lands between the step's compute nodes; here the device split starts with 4-5 copy kernels and
    //      all the slices, then each step's compute nodes run uninterrupted, as in the single-step draft graph.
    //   2: a step's mask rows are copied only when the slice is not contiguous: a 1-row step's row is already
    //      contiguous, only a multi-row step 0 narrower than the full mask needs the copy (one copy kernel, not n).
    //   4: a step picks its output row with a GET_ROWS only if it has several rows (step 0 with catch-up rows); a
    //      1-row step's row is already the output (no gather, no output-index slice).
    //   8: fewer uploads: the output rows ride in the position input, and V is stored with the K indices (equal for a
    //      non-transposed V cache; the V index input is then unused and not uploaded).
    //  16: the n draft ids are concatenated into one I32 tensor at the end of the graph (n-1 tiny kernels), so the host
    //      reads them with one copy: each small device->host read costs ~0.2 ms on this stack.
    static const int tight = [] {
        const char * e = getenv("LLAMA_MTP_CHAIN_TIGHT");
        return e ? atoi(e) : 31;
    }();

    const uint32_t n_pos = hparams.n_pos_per_embd();
    auto chain_inp = std::make_unique<llm_graph_input_mtp_chain>(n_pos);
    chain_inp->n_tokens = n_tokens;
    int64_t n_gather = 0;
    for (int64_t s = 0; s < n_steps; ++s) {
        llm_graph_input_mtp_chain::step st;
        st.r0     = s == 0 ? 0 : n_lead - 1 + s;
        st.nr     = s == 0 ? n_lead : 1;
        st.n_kv   = kvc->get_n_kv_prefix((uint32_t) (st.r0 + st.nr));
        st.gather = !(tight & 4) || st.nr > 1;
        n_gather += st.gather;
        chain_inp->steps.push_back(st);
    }
    chain_inp->out_in_pos = (tight & 8) && n_gather > 0;
    chain_inp->pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens * n_pos + (chain_inp->out_in_pos ? n_gather : 0));
    ggml_set_input(chain_inp->pos);
    if (n_gather > 0 && !chain_inp->out_in_pos) {
        chain_inp->out_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_gather);
        ggml_set_input(chain_inp->out_idx);
    }
    const auto    steps      = chain_inp->steps;
    const bool    out_in_pos = chain_inp->out_in_pos;
    ggml_tensor * pos_all    = chain_inp->pos;
    ggml_tensor * out_all    = chain_inp->out_idx;
    res->add_input(std::move(chain_inp));

    // token embeddings: the host table for the ubatch's own tokens (catch-up tokens need not be in the draft
    // vocabulary), the draft-vocab device copy (row k = token draft_head_ids[k]) for the argmax tokens
    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    const llama_hadamard_transform * inv = nullptr;
    if (hadamard_inverses) {
        const auto it = hadamard_inverses->find(tok_embd_w);
        if (it != hadamard_inverses->end()) {
            inv = &it->second;
        }
    }
    auto unrotate = [&](ggml_tensor * e) {
        // the single-step graph's inverse transform after the lookup (also applies to the copy: same rows)
        if (inv) {
            e = llama_mul_mat_hadamard(ctx0, e, inv->rot);
            if (inv->signs) {
                e = ggml_mul(ctx0, e, inv->signs);
            }
        }
        return e;
    };

    static const bool draft_head_off = getenv("LLAMA_MTP_DRAFT_HEAD_OFF") != nullptr;
    const bool use_draft_head = layer.nextn.draft_head && layer.nextn.draft_head_ids && !layer.nextn.shared_head_head &&
                                !draft_head_off;
    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm ? layer.nextn.shared_head_norm : model.output_norm;
    GGML_ASSERT(head_w && head_norm_w);

    static bool logged = false;
    if (!logged) {
        logged = true;
        LLAMA_LOG_INFO("%s: MTP chain graph (%lld steps, %s, %s token lookups)\n", __func__, (long long) n_steps,
                use_draft_head ? "draft head" : "full head", use_draft_head && mtp_chain_embd ? "device" : "host");
    }

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * tok_rows = ggml_get_rows(ctx0, tok_embd_w, ggml_view_1d(ctx0, tokens_all, n_lead, 0));
    ggml_tensor * h_embd   = ggml_view_2d(ctx0, h_all, h_all->ne[0], n_lead, h_all->nb[1], 0);
    ggml_tensor * k_all    = inp_attn->get_k_idxs();
    ggml_tensor * v_all    = (tight & 8) && !kvc->is_v_trans() ? k_all : inp_attn->get_v_idxs();
    ggml_tensor * mask_all = inp_attn->get_kq_mask();

    // the device holding the head's weights (none: CPU-only, the slices stay views of the host inputs)
    ggml_backend_t dev = nullptr;
    if ((tight & 1) && sched && layer.wq->buffer) {
        ggml_backend_buffer_type_t wbuft = ggml_backend_buffer_get_type(layer.wq->buffer);
        for (int b = 0; b < ggml_backend_sched_get_n_backends(sched) && !dev; ++b) {
            ggml_backend_t be = ggml_backend_sched_get_backend(sched, b);
            if (be != backend_cpu && ggml_backend_supports_buft(be, wbuft)) {
                dev = be;
            }
        }
    }
    if (tight & 1) {
        ggml_build_forward_expand(gf, tok_rows);  // the host-table lookup first: one host split, then one device split
    }
    if (dev) {
        // whole-tensor copies (a device memcpy each), pinned to the device so their views need no upload
        auto on_dev = [&](ggml_tensor * t, const char * name) {
            ggml_tensor * c = ggml_cont(ctx0, t);
            ggml_set_name(c, name);
            if (!ggml_backend_supports_op(dev, c)) {
                return t;
            }
            ggml_backend_sched_set_tensor_backend(sched, c, dev);
            ggml_build_forward_expand(gf, c);
            return c;
        };
        pos_all  = on_dev(pos_all,  "mtp_chain_pos_dev");
        if (n_gather > 1 && !out_in_pos) {
            out_all = on_dev(out_all, "mtp_chain_out_dev");
        }
        const bool v_is_k = v_all == k_all;
        k_all    = on_dev(k_all,    "mtp_chain_k_idxs_dev");
        v_all    = v_is_k ? k_all : on_dev(v_all, "mtp_chain_v_idxs_dev");
        mask_all = on_dev(mask_all, "mtp_chain_kq_mask_dev");
    }

    // each step's slices: positions, output row, KV indices and mask rows, sized as a decode of rows [0, r0 + nr)
    // would be (created here; added to the graph here with bit 1, else where the step first reads them)
    struct step_inputs {
        ggml_tensor * pos, * out, * k_idxs, * v_idxs, * kq_mask;
    };
    if (out_in_pos) {
        out_all = ggml_view_1d(ctx0, pos_all, n_gather, n_tokens * n_pos * ggml_element_size(pos_all));
    }
    std::vector<step_inputs> sin(n_steps);
    for (int64_t s = 0, g = 0; s < n_steps; ++s) {
        const auto &  st = steps[s];
        const int64_t nr = st.nr;
        auto &        si = sin[s];
        si.pos     = ggml_view_1d(ctx0, pos_all, nr * n_pos, st.r0 * n_pos * ggml_element_size(pos_all));
        si.out     = !st.gather ? nullptr : n_gather == 1 ? out_all : ggml_view_1d(ctx0, out_all, 1, g++ * ggml_element_size(out_all));
        si.k_idxs  = ggml_view_1d(ctx0, k_all, nr, st.r0 * ggml_element_size(k_all));
        si.v_idxs  = ggml_view_1d(ctx0, v_all, nr, st.r0 * ggml_element_size(v_all));
        si.kq_mask = ggml_view_2d(ctx0, mask_all, st.n_kv, nr, mask_all->nb[1], st.r0 * mask_all->nb[1]);
        if (!(tight & 2) || !ggml_is_contiguous(si.kq_mask)) {
            si.kq_mask = ggml_cont(ctx0, si.kq_mask);
        }
        if (tight & 1) {
            for (ggml_tensor * t : { si.pos, si.out, si.k_idxs, si.v_idxs, si.kq_mask }) {
                if (t) {
                    ggml_build_forward_expand(gf, t);
                }
            }
        }
    }

    ggml_tensor * tok_embd = unrotate(tok_rows);

    for (int64_t s = 0; s < n_steps; ++s) {
        const auto &  st = steps[s];
        const int64_t nr = st.nr;
        cb(tok_embd, "mtp_tok_embd", il);

        const uint32_t n_kv_s  = st.n_kv;
        ggml_tensor *  st_pos  = sin[s].pos;
        ggml_tensor *  st_out  = sin[s].out;
        ggml_tensor *  k_idxs  = sin[s].k_idxs;
        ggml_tensor *  v_idxs  = sin[s].v_idxs;
        ggml_tensor *  kq_mask = sin[s].kq_mask;

        ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
        cb(h_norm, "mtp_hnorm", il);

        ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
        cb(e_norm, "mtp_enorm", il);

        ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
        cb(concat, "mtp_concat", il);

        ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
        cb(cur, "mtp_eh_proj", il);

        ggml_tensor * inpSA = cur;

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "mtp_attn_norm", il);

        ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
        cb(Qcur_full, "mtp_Qcur_full", il);

        ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
                n_embd_head, n_head, nr,
                ggml_element_size(Qcur_full) * n_embd_head * 2,
                ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
                0);
        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "mtp_Qcur_normed", il);

        ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
                n_embd_head, n_head, nr,
                ggml_element_size(Qcur_full) * n_embd_head * 2,
                ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
                ggml_element_size(Qcur_full) * n_embd_head);
        gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, nr);
        cb(gate, "mtp_gate", il);

        ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, nr);
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Kcur, "mtp_Kcur_normed", il);

        ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, nr);
        cb(Vcur, "mtp_Vcur", il);

        Qcur = ggml_rope_multi(ctx0, Qcur, st_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_multi(ctx0, Kcur, st_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

        // build_attn() for this step's rows: same rotations, same expand order, store, then attend
        if (inp_attn->self_k_rot) {
            Qcur = llama_mul_mat_hadamard(ctx0, Qcur, inp_attn->self_k_rot);
            Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot);
        }
        if (inp_attn->self_v_rot) {
            Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot);
        }
        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Vcur);
        ggml_build_forward_expand(gf, Kcur);
        cb(Kcur, "k_cache_in", il);
        ggml_build_forward_expand(gf, kvc->cpy_k(ctx0, Kcur, k_idxs, il));
        ggml_build_forward_expand(gf, kvc->cpy_v(ctx0, Vcur, v_idxs, il));

        cur = build_attn_mha(Qcur, kvc->get_k_n(ctx0, il, n_kv_s), kvc->get_v_n(ctx0, il, n_kv_s), nullptr, kq_mask,
                nullptr, nullptr, kq_scale, il);
        cb(cur, "kqv_out", il);
        if (inp_attn->self_v_rot) {
            cur = llama_mul_mat_hadamard(ctx0, cur, inp_attn->self_v_rot);
        }
        cb(cur, "mtp_attn_pregate", il);

        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
        cur = build_lora_mm(layer.wo, cur, layer.wo_s);
        cb(cur, "mtp_attn_out", il);

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "mtp_attn_residual", il);

        ggml_tensor * ffn_residual = cur;
        cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "mtp_attn_post_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   nullptr, layer.ffn_up_s,
                layer.ffn_gate, nullptr, layer.ffn_gate_s,
                layer.ffn_down, nullptr, layer.ffn_down_s,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "mtp_ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "mtp_post_ffn", il);

        cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "h_nextn", -1);

        // the step's output row (masked h_nextn: the next step's hidden input)
        ggml_tensor * h_out = cur;
        if (st_out) {
            h_out = ggml_get_rows(ctx0, cur, st_out);
            cb(h_out, "mtp_shared_head_norm", -1);
        }

        // greedy draft: argmax of the head's logits. With a draft head that is an index into its vocabulary (the
        // -inf scatter of the single-step graph only maps it to a token id, which the context does on the host)
        ggml_tensor * logits = use_draft_head ? build_lora_mm(layer.nextn.draft_head, h_out, nullptr)
                                              : build_lora_mm(head_w, h_out, head_s);
        cb(logits, "result_output", -1);
        ggml_tensor * best = ggml_argmax(ctx0, logits);
        ggml_set_name(best, "mtp_chain_argmax");
        if (!(tight & 16)) {
            ggml_set_output(best);
        }
        ggml_build_forward_expand(gf, best);
        res->t_mtp_chain.push_back(best);

        if (s + 1 == n_steps) {
            break;
        }

        // next step's inputs: this step's draft token and output row
        ggml_tensor * e;
        if (use_draft_head && mtp_chain_embd) {
            e = ggml_get_rows(ctx0, mtp_chain_embd, best);
        } else if (use_draft_head) {
            ggml_tensor * ids = layer.nextn.draft_head_ids;
            ggml_tensor * tok = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, ids, 1, ids->ne[0]), best);  // I32 [1, 1]
            e = ggml_get_rows(ctx0, tok_embd_w, ggml_reshape_1d(ctx0, tok, 1));
        } else {
            e = ggml_get_rows(ctx0, tok_embd_w, best);
        }
        tok_embd = unrotate(e);
        h_embd   = h_out;
    }

    if (tight & 16) {
        // after the last step, so no node lands between a step's compute nodes
        ggml_tensor * ids = res->t_mtp_chain[0];
        for (size_t s = 1; s < res->t_mtp_chain.size(); ++s) {
            ids = ggml_concat(ctx0, ids, res->t_mtp_chain[s], 0);
        }
        ggml_set_name(ids, "mtp_chain_ids");
        ggml_set_output(ids);
        ggml_build_forward_expand(gf, ids);
        res->t_mtp_chain_ids = ids;
    }
}

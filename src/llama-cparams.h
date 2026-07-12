#pragma once

#include "llama.h"
#include "llama-hparams.h" // LLAMA_MAX_LAYERS

#include <cstdint>
#include <array>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0

    // multi-layer hidden-state tap (EAGLE3 / dspark target-feature reuse)
    // when n_capture_layers > 0 the model graph concatenates the per-layer output
    // of each layer in capture_layer_idx[0..n_capture_layers) along dim0 and the
    // context exposes it per position as a row of width [n_capture_layers * n_embd].
    // the order of capture_layer_idx defines the concatenation order.
    bool     embeddings_capture = false;
    uint32_t n_capture_layers   = 0;
    std::array<int32_t, LLAMA_MAX_LAYERS> capture_layer_idx = {};

    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};

#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP

#include "llama.h"

#include <cstdint>
#include <map>
#include <vector>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);
//
// multi-layer hidden-state tap (EAGLE3 / dspark target-feature reuse)
//
// Register an ordered set of intermediate decoder layers to capture. After a
// decode, the per-layer outputs are concatenated per position into a row of
// width [n_capture_layers * n_embd], laid out [layer0 | layer1 | ...] in the
// same order as layer_ids. Pass n_layers == 0 to disable.
//
// This is the shared primitive both EAGLE3-proper and dspark consume: where the
// pre-norm path above exposes one final-layer hidden vector, this exposes an
// arbitrary set of intermediate layers in one concatenated row.
LLAMA_API void     llama_set_capture_layers(struct llama_context * ctx, const int32_t * layer_ids, size_t n_layers);
LLAMA_API uint32_t llama_get_n_capture(struct llama_context * ctx);
// mirrors llama_get_embeddings_nextn / _ith
LLAMA_API float *  llama_get_embeddings_capture    (struct llama_context * ctx);
LLAMA_API float *  llama_get_embeddings_capture_ith(struct llama_context * ctx, int32_t i);
//
// dspark drafter: target-tap context window staging
//
// The dspark speculative-decoding drafter (EAGLE-style, block-diffusion) attends
// to a small, growing window of the TARGET model's captured multi-layer tap
// features (see llama_set_capture_layers above) as well as its own draft-block
// tokens. The context window doesn't fit llama_batch.token/embd: it has a
// different width (n_embd_cap = n_capture_layers * n_embd, i.e. RAW pre-fc tap
// concatenation) and a different row count than the draft block, so it is
// staged out of band, consumed by the drafter's graph on its next decode() call.
//
// feat is [n_ctx_rows * n_embd_cap] row-major (row i is the concatenated
// multi-layer tap feature for the i-th staged context row). The decode position
// of each context row is taken from the batch the drafter decodes next, not from
// this staged data, so no positions are passed here. Pass n_ctx_rows <= 0 or
// feat == nullptr to clear the staged context.
LLAMA_API void llama_set_dspark_ctx(
        struct llama_context * ctx,
        const float           * feat,
              int64_t           n_ctx_rows,
              int64_t           n_embd_cap);
//
// dspark drafter: model-level metadata + auxiliary-head weights
//
// The Phase 2 block-draft loop (common/speculative.cpp) needs a handful of
// dspark hparams that aren't reachable through the public llama.h surface:
//   - n_capture (target_layer_ids count): dspark ships no tokenizer of its own
//     (converter calls _set_vocab_none()), so the real vocab width and the
//     target-tap layer count both live outside the normal vocab/hparams path
//     that other archs expose generically. See src/models/dspark.cpp's
//     load_arch_tensors for the mirror-image loader-side fix.
//   - the Markov head's raw weights: the resample that applies them
//     (base_logits[step] + markov_w2(markov_w1(prev_token))) runs host-side,
//     strictly sequentially -- one step at a time, chaining the ACTUALLY
//     sampled token into the next step's lookup, never batched across the
//     block (see docs/dspark-scope.md and the ground-truth note about the
//     on-device MLX port's batching bug). At rank ~256 this is cheap enough
//     as a plain host embedding-lookup + dot-product loop, so it doesn't need
//     a graph -- it just needs the weights as host floats.
struct llama_dspark_meta {
    int64_t n_embd        = 0;
    int64_t n_vocab       = 0; // from token_embd.weight's own shape, not the (empty) vocab
    int64_t n_capture     = 0; // target_layer_ids count
    int64_t n_embd_cap    = 0; // n_capture * n_embd (raw pre-fc tap width)
    int32_t block_size    = 0;
    int32_t mask_token_id = 0;
    int64_t markov_rank   = 0; // 0 if the checkpoint has no markov head
};
// Returns false if `model` is not a loaded dspark model (block_size == 0).
LLAMA_API bool llama_model_dspark_get_meta(
        const struct llama_model * model,
              llama_dspark_meta   * out);
// Only the "vanilla" markov head (a plain low-rank embedding + linear pair,
// VanillaMarkov in the Python reference implementation) is supported here: it's the only
// variant present in shipped GGUFs today -- gated/rnn markov heads carry
// extra gate_proj/joint_proj tensors that dspark's GGUF converter and
// load_arch_tensors() don't currently map (see src/models/dspark.cpp).
//
// w1 and w2 are both returned as [n_vocab * n_rank] row-major (rank
// fastest-varying), matching their GGUF storage (ne = [n_rank, n_vocab]):
//   w1[token_id * n_rank + r] == markov_w1.weight[token_id][r]  (embedding row)
//   w2[token_id * n_rank + r] == markov_w2.weight[token_id][r]  (mul_mat weight row)
// Returns false (leaving w1/w2 untouched) if the model has no markov head.
LLAMA_API bool llama_model_dspark_get_markov(
        const struct llama_model * model,
        std::vector<float>       & w1,
        std::vector<float>       & w2);

// Run the sequential vanilla Markov resample on the drafter backend. The
// latest decode graph's logits remain device-resident; this helper gathers the
// previous-token row from markov_head_a, multiplies by markov_head_b, adds the
// corresponding logits row, and returns one argmax per block position. Returns
// false when the head/backend is unsupported so callers can use the host path.
LLAMA_API bool llama_dspark_markov_resample(
        struct llama_context * ctx,
        int32_t               n_rows,
        llama_token           prev_token,
        llama_token         * result);

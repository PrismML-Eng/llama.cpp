#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;

// Shared GGUF file format for the K-cache mean-centering bias vectors produced by
// tools/kv-mean-center and consumed by llama_kv_cache::load_kv_mean_center() (see
// docs/kv-mean-center.md).
//
// The file stores one F32 tensor per model layer that has a bias, named
// "kv_bar.blk.<il>.k", holding n_embd_head_k(il) * n_head_kv(il) values laid out as
// [n_embd_head_k, n_head_kv] (channel-fastest), matching the memory layout of the K
// tensor at the point it is written into the cache.

// per-layer entry to write; `bias` empty means "no bias for this layer" (skipped on write)
struct common_kv_mean_center_layer {
    int32_t            il = -1;
    std::vector<float> bias; // length n_embd_head_k(il) * n_head_kv(il)
};

// write a K-cache mean-centering bias file in GGUF format.
// `k_rot` records whether the bias was measured with the Hadamard K-cache rotation active
// (stored as the "kv_mean_center.k_rot" KV); the loader refuses a bias whose basis does not
// match the inference-time rotation state.
// returns false (and logs an error) on failure.
bool common_kv_mean_center_write(
        const std::string & fname,
        const std::vector<common_kv_mean_center_layer> & layers,
        bool k_rot = false);

// whether a captured "k_cache_in-<il>" tensor (the K about to be written into the cache) was
// produced with the Hadamard K-cache rotation applied. detected from the graph structure: the
// rotation is a MUL_MAT hinted GGML_HINT_SRC0_IS_HADAMARD whose result is reshaped back into
// the captured tensor (see llama_mul_mat_hadamard). the rotation matrix input is named
// "attn_inp_k_rot", or "attn_rot_dev_<n>" when it is kept on the cache's device
// (LLAMA_SYNC_TRIM bit 2), so the name alone is not a reliable signal.
bool common_kv_mean_center_k_rotated(const struct ggml_tensor * t);

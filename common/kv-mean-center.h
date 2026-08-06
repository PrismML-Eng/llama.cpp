#pragma once

#include <cstdint>
#include <string>
#include <vector>

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

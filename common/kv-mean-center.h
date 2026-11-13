#pragma once

#include <cstdint>
#include <algorithm>
#include <cstddef>
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
//
// The per-channel values may either be independent per-channel means (Q4_0 mode) or a
// per-128-channel-block mean replicated across its 128 elements (PQ2_0 mode, see
// common_kv_mean_center_collapse_to_blocks()). the latter matches PQ2_0's 128-elem
// quantization block layout but does not improve PQ2_0 fidelity (measured worse than
// no bias); it exists for A/B comparison and for a future zero-centered 2-bit KV format.

// per-layer entry to write; `bias` empty means "no bias for this layer" (skipped on write)
struct common_kv_mean_center_layer {
    int32_t            il = -1;
    std::vector<float> bias; // length n_embd_head_k(il) * n_head_kv(il)
};

// write a K-cache mean-centering bias file in GGUF format.
// `k_rot` records whether the bias was measured with the Hadamard K-cache rotation active
// (stored as the "kv_mean_center.k_rot" KV); the loader refuses a bias whose basis does not
// match the inference-time rotation state.
// `mode` records whether the bias is per-channel ("per-channel") or per-block ("per-block"),
// stored as the "kv_mean_center.mode" KV; the loader rejects a mode mismatch for PQ2_0.
// returns false (and logs an error) on failure.
bool common_kv_mean_center_write(
        const std::string & fname,
        const std::vector<common_kv_mean_center_layer> & layers,
        bool k_rot = false,
        const char * mode = "per-channel");

// PQ2_0 quantization block size (one fp16 scale + 2 bits per element, 128 weights per block).
// matches QK_PQ2_0 in ggml-common.h.
static const size_t common_kv_mean_center_block_size = 128;

// collapse a per-channel mean vector into per-block means: every element within a 128-channel
// block is set to the arithmetic mean of that block's per-channel means. this is the bias
// layout that aligns with PQ2_0's asymmetric codebook center (one offset per 128-elem block).
// the input length must be a multiple of the block size; the result has the same length.
std::vector<float> common_kv_mean_center_collapse_to_blocks(const std::vector<double> & per_channel);

// collapse a per-channel mean vector into per-block *recentered* offsets for PQ2_0: the PQ2_0
// codebook is (q-1)*d = {-d, 0, +d, +2d} (center +0.5d per block), so recentering K to 0
// (the per-channel / per-block mean offset) leaves the stored K off-center. this variant sets
// each block's offset to (block mean) - 0.5*d, where d is the block's max |K_raw| (the scale
// the quantizer will use on the re-centered K), moving the re-centered K to the codebook center.
// per_channel_maxabs[j] is the max absolute value of channel j across the calibration tokens.
std::vector<float> common_kv_mean_center_collapse_to_blocks_recentered(
        const std::vector<double> & per_channel, const std::vector<double> & per_channel_maxabs);

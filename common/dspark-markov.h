#pragma once

// Device-side (CUDA) implementation of the dspark vanilla Markov resample.
//
// This is an acceleration of the sequential per-position resample that
// common/speculative.cpp otherwise runs host-side (scalar or BLAS). It is
// compiled whenever the build has CUDA (LLAMA_DSPARK_MARKOV_CUDA) and, when the
// drafter carries a Markov head, is the DEFAULT resample path at runtime; set
// the environment variable LLAMA_DSPARK_MARKOV_CUDA=0 to fall back to the host
// path.
//
// The math mirrors the host paths: for a drafted position k,
//     correction[v] = sum_r w1[prev * R + r] * w2[v * R + r]
//     step_logit[v] = base_logits[k][v] + correction[v]
//     out[k]        = argmax_v step_logit[v]   (lowest v wins ties)
// where prev is the anchor token for k == 0 and the ACTUAL argmax of position
// k-1 for k > 0. The sequential chaining is preserved structurally on the
// device: position k's kernel reads the argmax that position k-1's kernel
// wrote into device memory -- it is never precomputed on the host.
//
// The device reduces `correction` with a warp tree-reduction, so its
// floating-point accumulation order differs from the host scalar loop (and from
// BLAS, which uses yet another order). The result is therefore functionally
// equivalent, not bit-identical: at an exact argmax tie the rounding can select
// a different token. That only changes which speculative draft is proposed --
// the target verify still arbitrates the committed output -- so it never affects
// correctness, only (rarely) the accept rate.

#include <cstdint>

struct dspark_markov_cuda;

// Upload the two low-rank Markov factors (each n_vocab * markov_rank fp32,
// rank fastest-varying, matching the host-resident markov_w1/markov_w2
// layout) to device buffers and allocate the per-round scratch. Returns
// nullptr on any failure (no device, allocation failure, bad dims); the
// caller falls back to the host path in that case.
dspark_markov_cuda * dspark_markov_cuda_init(
        const float * w1,
        const float * w2,
        int64_t       n_vocab,
        int64_t       markov_rank);

void dspark_markov_cuda_free(dspark_markov_cuda * ctx);

// Sequentially resample n_use positions on the device.
//   base_logits : host pointer, n_use * n_vocab fp32 (row k == position k),
//                 contiguous -- copied H2D once for the whole round.
//   id_last     : the block anchor token (prev for k == 0).
//   n_use       : number of positions to resample (1..block_size).
//   out_ids     : host buffer of n_use int32 argmax token ids (row order).
// Returns true on success; on false the caller falls back to the host path.
bool dspark_markov_cuda_resample(
        dspark_markov_cuda * ctx,
        const float *        base_logits,
        int32_t              id_last,
        int32_t              n_use,
        int32_t *            out_ids);

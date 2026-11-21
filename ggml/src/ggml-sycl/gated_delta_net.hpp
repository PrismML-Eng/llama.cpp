#pragma once

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_sycl_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_sycl_try_gdn_cache_fusion)
void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache);

// fused recurrent-state gather (see ggml_sycl_try_gdn_gather_skip): reset per graph, set per GDN node
void ggml_sycl_gdn_gather_reset(const ggml_backend_sycl_context & ctx);
void ggml_sycl_gdn_gather_set(const ggml_backend_sycl_context & ctx, const ggml_tensor * gdn, const float * base,
                              const int32_t * ids, int64_t row_stride);

// ARC-LAB: GATED_DELTA_NET_REPLAY (rollback by replay). fused = nullptr: base state and records go to dst (the graph
// copies them into the cache); otherwise the kernel writes them straight into the cache (see
// ggml_sycl_try_gdn_replay_fusion): base -> the CPY destination, records -> row rec_rows[seq] of the SET_ROWS target.
struct ggml_sycl_gdn_replay_fused {
    float *         base;
    float *         rec;
    const int64_t * rec_rows;        // device I64, one per seq
    int64_t         rec_row_stride;  // floats
};

void ggml_sycl_op_gated_delta_net_replay(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                         const ggml_sycl_gdn_replay_fused * fused);

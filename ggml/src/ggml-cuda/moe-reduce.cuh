#pragma once

#include "common.cuh"

// maximum n_expert_used handled by the fused MoE weighted reduce
// (bounded by ggml_can_fuse_subgraph's 32-node limit: the subgraph is 2*n_exp nodes)
#define GGML_CUDA_MOE_REDUCE_MAX_EXPERTS 15

// dst[i0, i2] = sum_e src0[i0, e, i2] * src1[0, e, i2]
//
// stage_weights copies src1 into pool scratch before the reduce runs. The caller needs it
// when the graph allocator has placed the (tiny, already dead) router-weight tensor inside
// the destination buffer, which would otherwise be a cross-block read/write race.
void ggml_cuda_op_moe_weighted_reduce(ggml_backend_cuda_context & ctx,
                                      const ggml_tensor *         src0,
                                      const ggml_tensor *         src1,
                                      ggml_tensor *               dst,
                                      int                         n_exp,
                                      bool                        stage_weights);

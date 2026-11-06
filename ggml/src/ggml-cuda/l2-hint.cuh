#pragma once

#include <cstddef>

// L2 prefetch hint: weights of the next PTQ1_0 mat-vec in the graph, set by the node loop before it dispatches a node,
// read by the PTQ1_0 mat-vec launcher (same thread). nullptr = no hint.
struct ggml_cuda_l2_hint_t {
    const char * ptr   = nullptr;
    size_t       bytes = 0;
};
extern thread_local ggml_cuda_l2_hint_t g_ggml_cuda_l2_hint;

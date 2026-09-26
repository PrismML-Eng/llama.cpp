#pragma once

#include "common.hpp"

// ARC-LAB decode attention straight from a q4_0 KV cache (see fattn-dec.cpp).
bool ggml_sycl_flash_attn_ext_dec_supported(const ggml_tensor * dst);
void ggml_sycl_flash_attn_ext_dec(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

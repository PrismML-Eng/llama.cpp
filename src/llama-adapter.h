#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// llama_adapter_cvec
//

struct llama_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;

    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const llama_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
};

using llama_adapter_cvec_ptr = std::shared_ptr<llama_adapter_cvec>;

//
// llama_adapter_lora
//
struct llama_model_loader;

bool llama_adapter_lora_is_tensor(const char * name);
// Throws on malformed or unsupported embedded corrections; registered with the model on success.
llama_adapter_lora * llama_adapter_lora_init_embedded(llama_model & model, llama_model_loader & ml);


struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct llama_adapter_lora {
    llama_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;
    // Resolve virtual branches once at load time; graph builds avoid formatting names.
    // unordered_map rehash preserves references to ab_map's stored weights.
    std::unordered_map<const ggml_tensor *, llama_adapter_lora_weight *> virtual_ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha = 0.0f;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;
    // Typed adapter.* metadata, retained for lossless embedded-adapter serialization.
    gguf_context_ptr embedded_metadata;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    explicit llama_adapter_lora(llama_model * model) : model(model) {}
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);
    llama_adapter_lora_weight * get_weight_named(const std::string & name);
    const llama_adapter_lora_weight * get_weight_for_anchor(const ggml_tensor * anchor) const {
        const auto pos = virtual_ab_map.find(anchor);
        return pos == virtual_ab_map.end() ? nullptr : pos->second;
    }

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};

using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
using llama_adapter_loras_ptr = std::unique_ptr<llama_adapter_loras>;

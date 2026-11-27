// Consumer-level Scion embedded-correction regression, using a tiny synthetic Qwen35MoE.
// Reference: sky-is-green/prism-ml-llama.cpp @ 3a25590406e8f390075061198d180267e69709bd.
// No downloaded weights: exercise the real loader, context, router and normalized-input branch.
#include "llama.h"
#include "llama-cpp.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "../src/llama-adapter.h"
#include "../src/llama-model.h"
#include "../src/llama-model-saver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static void near(float actual, float expected) {
    require(std::isfinite(actual) && std::fabs(actual - expected) <= 2e-4f * std::max(1.0f, std::fabs(expected)),
            "correction arithmetic mismatch");
}

static constexpr uint32_t width = 256;
static constexpr uint32_t vocab = 128;
static constexpr uint32_t layers = 2;
static constexpr int64_t rank = 2;
static constexpr char recipe[] = "embedded roundtrip provenance longer than forty characters must remain intact";
static constexpr uint32_t provenance[] = {7, 11};

enum class defect { none, missing_b, rank_mismatch, width_mismatch, batched, undeclared, bad_alpha };

static gguf_context_ptr metadata(bool corrected, defect bad = defect::none) {
    gguf_context_ptr meta(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_QWEN35MOE, meta.get());
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, "qwen35moe");
    ms.add_kv(LLM_KV_GENERAL_TYPE, "model");
    ms.add_kv(LLM_KV_VOCAB_SIZE, vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(64));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH, width);
    ms.add_kv(LLM_KV_BLOCK_COUNT, layers);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, uint32_t(32));
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, uint32_t(32));
    ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, uint32_t(32));
    ms.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>{32, 32, 32, 32});
    ms.add_kv(LLM_KV_SSM_INNER_SIZE, uint32_t(256));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL, uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE, uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK, uint32_t(2));
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT, uint32_t(2));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "no_vocab");
    if (!corrected) {
        return meta;
    }
    gguf_set_val_bool(meta.get(), "adapter.embedded", bad != defect::undeclared);
    gguf_set_val_str(meta.get(), "adapter.type", "lora");
    gguf_set_val_f32(meta.get(), "adapter.lora.alpha", 0.0f);
    if (bad == defect::bad_alpha) {
        gguf_set_val_str(meta.get(), "adapter.lora.alpha", "not a float");
    }
    gguf_set_val_str(meta.get(), "adapter.recipe", recipe);
    gguf_set_arr_data(meta.get(), "adapter.test.provenance", GGUF_TYPE_UINT32, provenance, 2);

    ggml_context_ptr ctx(ggml_init({16 * ggml_tensor_overhead(), nullptr, true}));
    require(bool(ctx), "factor metadata allocation failed");
    for (uint32_t il = 0; il < layers; ++il) {
        for (const char * target : {"ffn_moe_out.weight", "ffn_gate_inp.weight",
                                   il == 0 ? "ssm_out.weight" : "attn_output.weight"}) {
            const std::string name = "blk." + std::to_string(il) + "." + target;
            const bool branch = std::strcmp(target, "ffn_moe_out.weight") == 0;
            const int64_t output = std::strcmp(target, "ffn_gate_inp.weight") == 0 ? 2 : width;
            auto * a = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32,
                    branch && bad == defect::width_mismatch ? width / 2 : width,
                    rank, branch && bad == defect::batched ? 2 : 1);
            ggml_set_name(a, (name + ".lora_a").c_str());
            gguf_add_tensor(meta.get(), a);
            if (branch && bad == defect::missing_b) {
                continue;
            }
            auto * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32,
                    branch && bad == defect::rank_mismatch ? rank + 1 : rank, output);
            ggml_set_name(b, (name + ".lora_b").c_str());
            gguf_add_tensor(meta.get(), b);
        }
    }
    return meta;
}

static void fill_tensor(ggml_tensor * t, void *) {
    require(t->type == GGML_TYPE_F32, "unexpected fixture tensor type");
    std::vector<float> data(ggml_nelements(t), 0.0f);
    const std::string name(t->name);
    if (name == "token_embd.weight") {
        for (uint32_t i = 0; i < vocab; ++i) {
            data[i * width] = 1.0f;
            data[i * width + 1] = 2.0f + float(i % 3);
        }
    } else if (name == "output.weight") {
        for (uint32_t i = 0; i < vocab; ++i) {
            data[i * width + i] = 1.0f;
        }
    } else if (name.find("norm.weight") != std::string::npos || name.find(".scale") != std::string::npos ||
               name.find("input_scale") != std::string::npos) {
        std::fill(data.begin(), data.end(), 1.0f);
    } else if (name.find("ssm_a") != std::string::npos) {
        std::fill(data.begin(), data.end(), -1.0f);
    } else if (name.find(".lora_a") != std::string::npos) {
        data[0] = 1.0f;
    } else if (name.find("ffn_moe_out.weight.lora_b") != std::string::npos) {
        data[rank] = 3.0f; // B(Ah)[1] = 3*h[0], unlike B(A*MoE(h)).
    } else if (name.find("ffn_gate_inp.weight.lora_b") != std::string::npos) {
        data[rank] = 4.0f; // Switch expert 0 -> expert 1 before softmax/top-k.
    } else if (name.find("ffn_gate_inp.weight") != std::string::npos) {
        data[0] = 1.0f;
    } else if (name.find("ffn_up_shexp.weight") != std::string::npos ||
               name.find("ffn_gate_shexp.weight") != std::string::npos ||
               name.find("ffn_down_shexp.weight") != std::string::npos) {
        data[0] = 0.25f; // Nonzero gated shared output must survive addition of the branch.
    } else if (name.find("ffn_up_exps.weight") != std::string::npos ||
               name.find("ffn_gate_exps.weight") != std::string::npos ||
               name.find("ffn_gate_up_exps.weight") != std::string::npos ||
               name.find("ffn_down_exps.weight") != std::string::npos) {
        std::fill(data.begin(), data.end(), 0.005f);
    }
    ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
}

static llama_model_params model_params() {
    auto p = llama_model_default_params();
    static ggml_backend_dev_t devices[] = {nullptr};
    p.devices = devices;
    p.n_gpu_layers = 0;
    p.load_mode = LLAMA_LOAD_MODE_NONE;
    return p;
}

static llama_model_ptr make_model(gguf_context * meta) {
    return llama_model_ptr(llama_model_init_from_user(meta, fill_tensor, nullptr, model_params()));
}

struct observations {
    std::map<std::string, std::vector<float>> floats;
    std::map<std::string, std::vector<int32_t>> ids;
};

static bool observe(ggml_tensor * t, bool ask, void * userdata) {
    const std::string name(t->name);
    const bool wanted = name.find("attn_post_norm-") == 0 || name.find("ffn_moe_corr-") == 0 ||
        name.find("ffn_moe_logits-") == 0 || name.find("ffn_moe_probs-") == 0 ||
        name.find("ffn_moe_topk-") == 0 || name.find("ffn_moe_out-") == 0 ||
        name.find("ffn_shexp_gated-") == 0 || name.find("ffn_out-") == 0;
    if (ask) {
        return wanted;
    }
    auto & obs = *static_cast<observations *>(userdata);
    if (t->type == GGML_TYPE_I32) {
        auto & data = obs.ids[name];
        data.resize(ggml_nelements(t));
        // argsort_top_k can be a strided view of the full expert ordering.
        std::vector<int32_t> raw(ggml_nbytes(t) / sizeof(int32_t));
        ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));
        require(t->ne[0] == 1 && t->ne[2] == 1 && t->ne[3] == 1, "unexpected top-k shape");
        for (int64_t token = 0; token < t->ne[1]; ++token) {
            data[token] = raw[token * t->nb[1] / sizeof(int32_t)];
        }
    } else {
        require(t->type == GGML_TYPE_F32, "unexpected observed tensor type");
        require(ggml_is_contiguous(t), "unexpected strided floating observation");
        auto & data = obs.floats[name];
        data.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
    }
    return true;
}

static llama_context_ptr make_context(llama_model * model, observations & obs) {
    auto p = llama_context_default_params();
    p.n_ctx = 64;
    p.n_batch = p.n_ubatch = 8;
    p.n_threads = p.n_threads_batch = 1;
    p.op_offload = false;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    p.cb_eval = observe;
    p.cb_eval_user_data = &obs;
    llama_context_ptr ctx(llama_init_from_model(model, p));
    require(bool(ctx), "context creation failed");
    return ctx;
}

static std::vector<float> decode(llama_context * ctx, observations & obs, int tokens = 2) {
    obs = {};
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_token input[] = {1, 2};
    auto batch = llama_batch_get_one(input, tokens);
    require(llama_decode(ctx, batch) == 0, "decode failed");
    const float * logits = llama_get_logits_ith(ctx, -1);
    require(logits != nullptr, "missing logits");
    return {logits, logits + vocab};
}

static void check_graph(const observations & obs, bool corrected, float branch_coefficient = 3.0f) {
    for (uint32_t il = 0; il < layers; ++il) {
        const auto key = [il](const char * name) { return std::string(name) + "-" + std::to_string(il); };
        const auto & h = obs.floats.at(key("attn_post_norm"));
        const auto & routed = obs.floats.at(key("ffn_moe_out"));
        const auto & shared = obs.floats.at(key("ffn_shexp_gated"));
        const auto & out = obs.floats.at(key("ffn_out"));
        const auto & logits = obs.floats.at(key("ffn_moe_logits"));
        const auto & probs = obs.floats.at(key("ffn_moe_probs"));
        const auto & ids = obs.ids.at(key("ffn_moe_topk"));
        require(h[0] > 0 && routed[0] > 0 && shared[0] > 0,
                "fixture must exercise nonzero normalized input, routed and gated shared outputs");
        for (size_t token = 0; token < ids.size(); ++token) {
            near(logits[2 * token], h[width * token]);
            near(logits[2 * token + 1], corrected ? 4 * h[width * token] : 0);
            near(probs[2 * token + 1], 1.0f / (1.0f + std::exp(logits[2 * token] - logits[2 * token + 1])));
            require(ids[token] == (corrected ? 1 : 0), "router correction did not change top-k");
        }
        for (size_t i = 0; i < out.size(); ++i) {
            float delta = 0;
            if (corrected) {
                delta = obs.floats.at(key("ffn_moe_corr"))[i];
                near(delta, i % width == 1 ? branch_coefficient * h[(i / width) * width] : 0);
            }
            near(out[i], routed[i] + shared[i] + delta);
        }
    }
}

using file_ptr = std::unique_ptr<FILE, decltype(&fclose)>;

static file_ptr temp_file() {
    file_ptr file(std::tmpfile(), fclose);
    require(bool(file), "temporary file creation failed");
    return file;
}

static void check_roundtrip(llama_model * model, const std::vector<float> & expected) {
    auto file = temp_file();
    llama_model_saver saver(model);
    saver.add_kv_from_model();
    saver.add_tensors_from_model();
    saver.save(file.get());
    std::rewind(file.get());
    llama_model_ptr restored(llama_model_load_from_file_ptr(file.get(), model_params()));
    require(bool(restored) && restored->embedded_lora && restored->embedded_lora->ab_map.size() == 6, "roundtrip lost correction pairs");
    const auto * saved_meta = restored->embedded_lora->embedded_metadata.get();
    require(saved_meta != nullptr, "roundtrip lost adapter metadata");
    const int64_t recipe_key = gguf_find_key(saved_meta, "adapter.recipe");
    require(recipe_key >= 0 && gguf_get_kv_type(saved_meta, recipe_key) == GGUF_TYPE_STRING &&
            std::strcmp(gguf_get_val_str(saved_meta, recipe_key), recipe) == 0, "roundtrip changed provenance");
    const int64_t provenance_key = gguf_find_key(saved_meta, "adapter.test.provenance");
    require(provenance_key >= 0 && gguf_get_kv_type(saved_meta, provenance_key) == GGUF_TYPE_ARRAY &&
            gguf_get_arr_type(saved_meta, provenance_key) == GGUF_TYPE_UINT32 &&
            gguf_get_arr_n(saved_meta, provenance_key) == 2 &&
            std::memcmp(gguf_get_arr_data(saved_meta, provenance_key), provenance, sizeof(provenance)) == 0,
            "roundtrip changed typed adapter metadata");
    for (const auto & pair : model->embedded_lora->ab_map) {
        auto * other = restored->embedded_lora->get_weight_named(pair.first);
        require(other != nullptr, "roundtrip lost factor name");
        for (const auto & tensors : {std::make_pair(pair.second.a, other->a), std::make_pair(pair.second.b, other->b)}) {
            require(ggml_are_same_shape(tensors.first, tensors.second) && tensors.first->type == tensors.second->type,
                    "roundtrip changed factor type/shape");
            std::vector<uint8_t> a(ggml_nbytes(tensors.first)), b(a.size());
            ggml_backend_tensor_get(tensors.first, a.data(), 0, a.size());
            ggml_backend_tensor_get(tensors.second, b.data(), 0, b.size());
            require(a == b, "roundtrip changed trained factor bytes");
        }
    }
    observations obs;
    auto ctx = make_context(restored.get(), obs);
    require(decode(ctx.get(), obs) == expected, "roundtrip changed logits");
    check_graph(obs, true);

    std::rewind(file.get());
    auto p = model_params();
    p.no_alloc = true;
    llama_model_ptr dry(llama_model_load_from_file_ptr(file.get(), p));
    require(bool(dry) && dry->embedded_lora, "no_alloc load lost adapter");
    size_t bytes = 0;
    for (const auto & entry : dry->memory_breakdown()) {
        bytes += entry.second;
    }
    size_t payload_bytes = 0;
    for (const auto & entry : dry->tensors_by_name) {
        if (!entry.second->view_src) {
            payload_bytes += ggml_nbytes(entry.second);
        }
    }
    for (const auto & entry : dry->embedded_lora->ab_map) {
        payload_bytes += ggml_nbytes(entry.second.a) + ggml_nbytes(entry.second.b);
    }
    require(bytes >= payload_bytes, "no_alloc memory accounting omitted deployed tensor payloads");
}

static void check_external(llama_model * model, llama_context * ctx, observations & obs, const std::vector<float> & expected) {
    gguf_context_ptr meta(gguf_init_empty());
    gguf_set_val_str(meta.get(), "general.type", "adapter");
    gguf_set_val_str(meta.get(), "general.architecture", "qwen35moe");
    gguf_set_val_str(meta.get(), "adapter.type", "lora");
    gguf_set_val_f32(meta.get(), "adapter.lora.alpha", 0.0f);
    ggml_context_ptr tensors(ggml_init({16384, nullptr, false}));
    require(bool(tensors), "external factor allocation failed");
    for (uint32_t il = 0; il < layers; ++il) {
        const std::string name = "blk." + std::to_string(il) + ".ffn_moe_out.weight";
        auto * a = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32, width, rank);
        auto * b = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32, rank, width);
        std::memset(a->data, 0, ggml_nbytes(a));
        std::memset(b->data, 0, ggml_nbytes(b));
        static_cast<float *>(a->data)[0] = 1;
        static_cast<float *>(b->data)[rank] = 0.5f;
        ggml_set_name(a, (name + ".lora_a").c_str());
        ggml_set_name(b, (name + ".lora_b").c_str());
        gguf_add_tensor(meta.get(), a);
        gguf_add_tensor(meta.get(), b);
    }
    auto file = temp_file();
    require(gguf_write_to_file_ptr(meta.get(), file.get(), false), "external adapter serialization failed");
    std::rewind(file.get());
    std::unique_ptr<llama_adapter_lora, decltype(&llama_adapter_lora_free)> external(
        llama_adapter_lora_init_from_file_ptr(model, file.get()), llama_adapter_lora_free);
    require(bool(external), "normal external adapter load failed");
    auto * ptr = external.get();
    float scale = 2;
    require(llama_set_adapters_lora(ctx, &ptr, 1, &scale) == 0, "external activation failed");
    decode(ctx, obs);
    check_graph(obs, true, 4.0f);
    scale = 0;
    require(llama_set_adapters_lora(ctx, &ptr, 1, &scale) == 0, "external zero scale failed");
    require(decode(ctx, obs) == expected, "external zero scale removed embedded correction");
    require(llama_set_adapters_lora(ctx, nullptr, 0, nullptr) == 0, "external clear failed");
    require(decode(ctx, obs) == expected, "external clear removed embedded correction");
}

int main() {
    ggml_backend_load_all();
    llama_backend_init();
    try {
        auto base_meta = metadata(false);
        auto base = make_model(base_meta.get());
        require(bool(base), "base fixture load failed");
        observations base_obs;
        auto base_ctx = make_context(base.get(), base_obs);
        const auto base_logits = decode(base_ctx.get(), base_obs);
        check_graph(base_obs, false);

        auto meta = metadata(true);
        auto model = make_model(meta.get());
        require(bool(model) && model->embedded_lora && model->embedded_lora->ab_map.size() == 6, "embedded fixture load failed");
        observations obs;
        auto ctx = make_context(model.get(), obs);
        const auto logits = decode(ctx.get(), obs);
        check_graph(obs, true);
        require(logits != base_logits, "embedded correction did not affect consumer logits");
        decode(ctx.get(), obs, 1);
        check_graph(obs, true);
        observations second_obs;
        auto second = make_context(model.get(), second_obs);
        require(decode(second.get(), second_obs) == logits, "second context did not activate embedded correction");
        check_external(model.get(), ctx.get(), obs, logits);
        check_roundtrip(model.get(), logits);

        for (auto bad : {defect::missing_b, defect::rank_mismatch, defect::width_mismatch,
                         defect::batched, defect::undeclared, defect::bad_alpha}) {
            auto broken = metadata(true, bad);
            require(!make_model(broken.get()), "malformed embedded adapter was accepted");
        }
        std::fprintf(stderr, "embedded LoRA regression passed\n");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "embedded LoRA regression failed: %s\n", e.what());
        llama_backend_free();
        return 1;
    }
    llama_backend_free();
    return 0;
}

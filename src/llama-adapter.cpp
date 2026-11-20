#include "llama-adapter.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include <map>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <cmath>
#include <sstream>
#include <stdexcept>

// vec

ggml_tensor * llama_adapter_cvec::tensor_for(int il) const {
    if (il < 0 || il < layer_start || il > layer_end || (size_t) il >= tensors.size()) {
        return nullptr;
    }

    return tensors[il];
}

ggml_tensor * llama_adapter_cvec::apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const {
    ggml_tensor * layer_dir = tensor_for(il);
    if (layer_dir != nullptr) {
        cur = ggml_add(ctx, cur, layer_dir);
    }

    return cur;
}

bool llama_adapter_cvec::init(const llama_model & model) {
    const auto & hparams = model.hparams;

    GGML_ASSERT(tensors.empty());
    GGML_ASSERT(ctxs.empty());
    GGML_ASSERT(bufs.empty());

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ hparams.n_layer()*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map[buft] = ctx;
            ctxs.emplace_back(ctx);

            return ctx;
        }

        return it->second;
    };

    // make tensors
    tensors.reserve(hparams.n_layer());
    tensors.push_back(nullptr); // there's never a tensor for layer 0
    for (size_t il = 1; il < hparams.n_layer(); il++) {
        ggml_backend_buffer_type_t buft = model.select_buft(il);
        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to allocate context for control vector\n", __func__);
            return false;
        }
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
        tensors.push_back(tensor);
    }

    // allocate tensors / buffers and zero
    bufs.reserve(ctx_map.size());
    for (auto it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx = it.second;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for control vector\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        bufs.emplace_back(buf);
    }

    return true;
}

bool llama_adapter_cvec::apply(
        const llama_model & model,
        const float * data,
        size_t len,
        int32_t n_embd,
        int32_t il_start,
        int32_t il_end) {
    const auto & hparams = model.hparams;

    if (data == nullptr) {
        // disable the current control vector (but leave allocated for later)
        layer_start = -1;
        layer_end   = -1;
        return true;
    }

    if (n_embd != (int) hparams.n_embd) {
        LLAMA_LOG_ERROR("%s: control vector n_embd does not match model\n", __func__);
        return false;
    }

    if (tensors.empty()) {
        if (!init(model)) {
            return false;
        }
    }

    layer_start = il_start;
    layer_end   = il_end;

    for (size_t il = 1; il < hparams.n_layer(); il++) {
        assert(tensors[il] != nullptr);

        const size_t off = n_embd * (il - 1); // buffer doesn't have data for layer 0, since it's never present
        if (off + n_embd <= len) {
            ggml_backend_tensor_set(tensors[il], data + off, 0, n_embd * ggml_element_size(tensors[il]));
        }
    }

    return true;
}

// lora

static bool lora_ends_with(const std::string & name, const std::string & suffix) {
    return name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool llama_adapter_lora_is_tensor(const char * name) {
    const size_t len = strlen(name);
    return len >= 7 && (strcmp(name + len - 7, ".lora_a") == 0 || strcmp(name + len - 7, ".lora_b") == 0);
}

llama_adapter_lora_weight * llama_adapter_lora::get_weight(ggml_tensor * w) {
    return get_weight_named(w->name);
}

llama_adapter_lora_weight * llama_adapter_lora::get_weight_named(const std::string & name) {
    const auto pos = ab_map.find(name);
    if (pos != ab_map.end()) {
        return &pos->second;
    }

    return nullptr;
}

static gguf_context_ptr copy_embedded_adapter_metadata(const gguf_context * src) {
    gguf_context_ptr dst(gguf_init_empty());
    for (int64_t i = 0; i < gguf_get_n_kv(src); ++i) {
        const char * key = gguf_get_key(src, i);
        if (strncmp(key, "adapter.", 8) != 0) {
            continue;
        }
        switch (gguf_get_kv_type(src, i)) {
            case GGUF_TYPE_UINT8:   gguf_set_val_u8  (dst.get(), key, gguf_get_val_u8  (src, i)); break;
            case GGUF_TYPE_INT8:    gguf_set_val_i8  (dst.get(), key, gguf_get_val_i8  (src, i)); break;
            case GGUF_TYPE_UINT16:  gguf_set_val_u16 (dst.get(), key, gguf_get_val_u16 (src, i)); break;
            case GGUF_TYPE_INT16:   gguf_set_val_i16 (dst.get(), key, gguf_get_val_i16 (src, i)); break;
            case GGUF_TYPE_UINT32:  gguf_set_val_u32 (dst.get(), key, gguf_get_val_u32 (src, i)); break;
            case GGUF_TYPE_INT32:   gguf_set_val_i32 (dst.get(), key, gguf_get_val_i32 (src, i)); break;
            case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (dst.get(), key, gguf_get_val_f32 (src, i)); break;
            case GGUF_TYPE_UINT64:  gguf_set_val_u64 (dst.get(), key, gguf_get_val_u64 (src, i)); break;
            case GGUF_TYPE_INT64:   gguf_set_val_i64 (dst.get(), key, gguf_get_val_i64 (src, i)); break;
            case GGUF_TYPE_FLOAT64: gguf_set_val_f64 (dst.get(), key, gguf_get_val_f64 (src, i)); break;
            case GGUF_TYPE_BOOL:    gguf_set_val_bool(dst.get(), key, gguf_get_val_bool(src, i)); break;
            case GGUF_TYPE_STRING:  gguf_set_val_str (dst.get(), key, gguf_get_val_str (src, i)); break;
            case GGUF_TYPE_ARRAY: {
                const size_t n = gguf_get_arr_n(src, i);
                const auto type = gguf_get_arr_type(src, i);
                if (type == GGUF_TYPE_STRING) {
                    std::vector<const char *> strings(n);
                    for (size_t j = 0; j < n; ++j) {
                        strings[j] = gguf_get_arr_str(src, i, j);
                    }
                    gguf_set_arr_str(dst.get(), key, strings.data(), n);
                } else {
                    gguf_set_arr_data(dst.get(), key, type, gguf_get_arr_data(src, i), n);
                }
                break;
            }
            default: throw std::runtime_error("invalid embedded adapter metadata type");
        }
    }
    return dst;
}

// Embedded loading reuses the model loader's catalog, including split files and FILE* offsets.
// The virtual MoE target follows sky-is-green/prism-ml-llama.cpp at 3a25590406e8.
static void llama_adapter_lora_init_impl(
        llama_model & model, FILE * file, llama_adapter_lora & adapter, llama_model_loader * ml = nullptr) {
    gguf_context_ptr owned_gguf;
    ggml_context_ptr ctx;
    std::unique_ptr<llama_file> gguf_file;
    gguf_context * metadata;
    if (ml) {
        metadata = ml->metadata;
        adapter.embedded_metadata = copy_embedded_adapter_metadata(metadata);
    } else {
        ggml_context * ctx_init = nullptr;
        gguf_init_params params = { true, &ctx_init };
        owned_gguf.reset(gguf_init_from_file_ptr(file, params));
        ctx.reset(ctx_init);
        if (!owned_gguf) {
            throw std::runtime_error("failed to load lora adapter from file");
        }
        metadata = owned_gguf.get();
        // Must follow GGUF parsing, since llama_file moves the file position.
        gguf_file.reset(new llama_file(file));
    }

    // check metadata
    {
        const gguf_context * gguf_ctx = metadata;

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values.\n", __func__);

        // get metadata as string
        for (int i = 0; i < gguf_get_n_kv(gguf_ctx); i++) {
            gguf_type type = gguf_get_kv_type(gguf_ctx, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%zu]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(gguf_ctx, i)), gguf_get_arr_n(gguf_ctx, i))
                : gguf_type_name(type);
            const char * name = gguf_get_key(gguf_ctx, i);
            const size_t MAX_VALUE_LEN = 40;
            const std::string value = gguf_kv_to_str(gguf_ctx, i, MAX_VALUE_LEN);

            if (type != GGUF_TYPE_ARRAY) {
                adapter.gguf_kv.emplace(name, value);
            }

            std::string print_value = value.size() > MAX_VALUE_LEN ? format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str()) : value;
            replace_all(print_value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), print_value.c_str());
        }

        auto get_kv_str = [&](const std::string & key) -> std::string {
            int id = gguf_find_key(gguf_ctx, key.c_str());
            if (id >= 0 && gguf_get_kv_type(gguf_ctx, id) != GGUF_TYPE_STRING) {
                throw std::runtime_error("invalid gguf type for " + key);
            }
            return id < 0 ? "" : std::string(gguf_get_val_str(gguf_ctx, id));
        };
        auto get_kv_f32 = [&](const std::string & key) -> float {
            int id = gguf_find_key(gguf_ctx, key.c_str());
            if (id >= 0 && gguf_get_kv_type(gguf_ctx, id) != GGUF_TYPE_FLOAT32) {
                throw std::runtime_error("invalid gguf type for " + key);
            }
            return id < 0 ? 0.0f : gguf_get_val_f32(gguf_ctx, id);
        };
        LLM_KV llm_kv = LLM_KV(LLM_ARCH_UNKNOWN);

        auto general_type = get_kv_str(llm_kv(LLM_KV_GENERAL_TYPE));
        const char * expected_type = ml ? "model" : "adapter";
        if (general_type != expected_type) {
            throw std::runtime_error(std::string("expect general.type to be '") + expected_type + "', but got: " + general_type);
        }

        auto general_arch_str = get_kv_str(llm_kv(LLM_KV_GENERAL_ARCHITECTURE));
        auto general_arch = llm_arch_from_string(general_arch_str);
        if (general_arch != model.arch) {
            throw std::runtime_error("model arch and LoRA arch mismatch");
        }

        auto adapter_type = get_kv_str(llm_kv(LLM_KV_ADAPTER_TYPE));
        if (adapter_type != "lora") {
            throw std::runtime_error("expect adapter.type to be 'lora', but got: " + adapter_type);
        }

        adapter.alpha = get_kv_f32(llm_kv(LLM_KV_ADAPTER_LORA_ALPHA));
        if (!std::isfinite(adapter.alpha)) {
            throw std::runtime_error("adapter.lora.alpha must be finite");
        }

        // parse alora invocation sequence vector
        const auto & key = llm_kv(LLM_KV_ADAPTER_ALORA_INVOCATION_TOKENS);
        const int kid = gguf_find_key(metadata, key.c_str());
        if (kid >= 0) {
            if (gguf_get_kv_type(metadata, kid) != GGUF_TYPE_ARRAY) {
                throw std::runtime_error("invalid gguf type for " + key);
            }
            const auto arr_type = gguf_get_arr_type(metadata, kid);
            if (arr_type != GGUF_TYPE_UINT32) {
                throw std::runtime_error("invalid gguf element type for " + key);
            }
            const size_t seq_len = gguf_get_arr_n(metadata, kid);
            if (ml && seq_len != 0) {
                throw std::runtime_error("embedded corrections cannot be conditional aLoRA adapters");
            }
            const void * data = gguf_get_arr_data(metadata, kid);
            adapter.alora_invocation_tokens.resize(seq_len);
            std::copy(
                (const llama_token *)data,
                (const llama_token *)data + seq_len,
                adapter.alora_invocation_tokens.begin());
        }
    }

    const int n_tensors = ml ? ml->n_embedded_tensors : gguf_get_n_tensors(metadata);

    // contexts for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            // add a new context
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * buft_ctx = ggml_init(params);
            if (!buft_ctx) {
                return nullptr;
            }
            ctx_map[buft] = buft_ctx;
            adapter.ctxs.emplace_back(buft_ctx);
            return buft_ctx;
        };
        return it->second;
    };

    // bundle lora_a and lora_b into pairs
    std::map<std::string, llama_adapter_lora_weight> ab_map;
    auto add_tensor = [&](ggml_tensor * cur) {
        std::string name(cur->name);
        if (lora_ends_with(name, ".lora_a")) {
            name.resize(name.size() - 7);
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(cur, nullptr);
            } else {
                ab_map[name].a = cur;
            }
        } else if (lora_ends_with(name, ".lora_b")) {
            name.resize(name.size() - 7);
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(nullptr, cur);
            } else {
                ab_map[name].b = cur;
            }
        } else if (lora_ends_with(name, "_norm.weight")) {
            // TODO: add support for norm vector
            // for now, we don't really care because most adapters still work fine without it
            return;
        } else {
            throw std::runtime_error("LoRA tensor '" + name + "' has unexpected suffix");
        }
    };
    if (ml && ml->files.empty()) {
        // Metadata/callback model loading has no source GGML context.
        ctx.reset(ggml_init({ n_tensors * ggml_tensor_overhead(), nullptr, true }));
        if (!ctx) {
            throw std::runtime_error("failed to allocate embedded adapter metadata");
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(metadata); ++i) {
            const char * name = gguf_get_tensor_name(metadata, i);
            if (llama_adapter_lora_is_tensor(name)) {
                auto * t = ggml_new_tensor(ctx.get(), gguf_get_tensor_type(metadata, i), GGML_MAX_DIMS,
                                          gguf_get_tensor_ne(metadata, i));
                ggml_set_name(t, name);
                add_tensor(t);
            }
        }
    } else if (ml) {
        for (const auto & it : ml->weights_map) {
            if (llama_adapter_lora_is_tensor(it.first.c_str())) {
                add_tensor(it.second.tensor);
            }
        }
    } else {
        for (auto * cur = ggml_get_first_tensor(ctx.get()); cur; cur = ggml_get_next_tensor(ctx.get(), cur)) {
            add_tensor(cur);
        }
    }

    // get extra buffer types of the CPU
    // TODO: a more general solution for non-CPU extra buft should be implemented in the future
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12593#pullrequestreview-2718659948
    std::vector<ggml_backend_buffer_type_t> buft_extra;
    {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu_dev) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }
        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);

        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");

        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_extra.emplace_back(*extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add tensors
    for (auto & it : ab_map) {
        const std::string & name = it.first;
        llama_adapter_lora_weight & w = it.second;
        bool is_token_embd = lora_ends_with(name, "token_embd.weight");

        if (!w.a || !w.b) {
            throw std::runtime_error("LoRA tensor pair for '" + name + "' is missing one component");
        }

        // device buft and device ctx
        const bool is_moe_corr = lora_ends_with(name, ".ffn_moe_out.weight");
        if (is_moe_corr && model.arch != LLM_ARCH_QWEN35MOE) {
            throw std::runtime_error("MoE output correction is only implemented for qwen35moe");
        }
        const std::string anchor = is_moe_corr ?
            name.substr(0, name.size() - strlen("ffn_moe_out.weight")) + "ffn_gate_inp.weight" : name;
        const auto * model_tensor = model.get_tensor(anchor.c_str());
        if (!model_tensor) {
            throw std::runtime_error("LoRA tensor '" + name + "' does not exist in base model (hint: maybe wrong base model?)");
        }
        // build_lora_mm feeds a folded weight's LoRA delta the primal input, so folded matmul weights are fine;
        // a latent lookup table (token_embd, also the head when the output is tied) stores rotated rows, refuse it
        if (model.hdmd.inverse_blocks.count(name)) {
            throw std::runtime_error("LoRA tensor '" + name + "' targets a prism.hadamard latent lookup table, which is not supported");
        }

        auto * buft = ggml_backend_buffer_get_type(model_tensor->buffer);

        // do not load loras to extra buffer types (i.e. bufts for repacking) -> use the CPU in that case
        for (auto & ex : buft_extra) {
            if (ex == buft) {
                LLAMA_LOG_WARN("%s: lora for '%s' cannot use buft '%s', fallback to CPU\n", __func__, model_tensor->name, ggml_backend_buft_name(buft));

                auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                if (!cpu_dev) {
                    throw std::runtime_error(format("%s: no CPU backend found", __func__));
                }
                buft = ggml_backend_dev_buffer_type(cpu_dev);

                break;
            }
        }

        LLAMA_LOG_DEBUG("%s: lora for '%s' -> '%s'\n", __func__, model_tensor->name, ggml_backend_buft_name(buft));

        ggml_context * dev_ctx = ctx_for_buft(buft);
        // validate tensor shape
        if (!dev_ctx) {
            throw std::runtime_error("failed to allocate LoRA tensor context");
        }
        if (is_moe_corr) {
            // A=[n_embd,rank], B=[rank,n_embd]; no batched/expert dimensions.
            if (w.a->ne[0] != model.hparams.n_embd || w.b->ne[1] != model.hparams.n_embd ||
                w.a->ne[1] != w.b->ne[0] || w.a->ne[1] <= 0 ||
                w.a->ne[2] != 1 || w.a->ne[3] != 1 || w.b->ne[2] != 1 || w.b->ne[3] != 1) {
                throw std::runtime_error("MoE output correction '" + name + "' expects A=[n_embd,rank], B=[rank,n_embd]");
            }
        } else if (is_token_embd) {
            // expect B to be non-transposed, A and B are flipped; see llm_build_inp_embd()
            if (model_tensor->ne[0] != w.b->ne[1] || model_tensor->ne[1] != w.a->ne[1]) {
                throw std::runtime_error("tensor '" + name + "' has incorrect shape (hint: maybe wrong base model?)");
            }
        } else {
            if (model_tensor->ne[0] != w.a->ne[0] || model_tensor->ne[1] != w.b->ne[1] ||
                (ml && (model_tensor->ne[2] != w.a->ne[2] || model_tensor->ne[2] != w.b->ne[2] ||
                        model_tensor->ne[3] != w.a->ne[3] || model_tensor->ne[3] != w.b->ne[3]))) {
                throw std::runtime_error("tensor '" + name + "' has incorrect shape (hint: maybe wrong base model?)");
            }
            if (w.a->ne[1] != w.b->ne[0]) {
                throw std::runtime_error("lora_a tensor is not transposed (hint: adapter from \"finetune\" example is no longer supported)");
            }
        }

        // save tensor to adapter
        ggml_tensor * tensor_a = ggml_dup_tensor(dev_ctx, w.a);
        ggml_tensor * tensor_b = ggml_dup_tensor(dev_ctx, w.b);
        ggml_set_name(tensor_a, w.a->name);
        ggml_set_name(tensor_b, w.b->name);
        auto & registered = adapter.ab_map.emplace(name, llama_adapter_lora_weight(tensor_a, tensor_b)).first->second;
        if (is_moe_corr) {
            adapter.virtual_ab_map.emplace(model_tensor, &registered);
        }
    }


    // allocate tensors / buffers and zero
    {
        adapter.ctxs.reserve(ctx_map.size());
        adapter.bufs.reserve(ctx_map.size());
        for (auto & it : ctx_map) {
            ggml_backend_buffer_type_t buft = it.first;
            ggml_context * ctx_dev = it.second;
            ggml_backend_buffer_ptr buf { ml && ml->no_alloc ?
                ggml_backend_buft_alloc_buffer(buft, 0) : ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev, buft) };
            if (!buf) {
                throw std::runtime_error("failed to allocate buffer for lora adapter\n");
            }
            if (ml && ml->no_alloc) {
                for (auto * t = ggml_get_first_tensor(ctx_dev); t; t = ggml_get_next_tensor(ctx_dev, t)) {
                    t->buffer = buf.get();
                }
            }
            LLAMA_LOG_INFO("%s: %10s LoRA buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);
            ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            adapter.bufs.emplace_back(std::move(buf));
        }
    }

    if (ml && ml->no_alloc) {
        model.loras.insert(&adapter);
        return;
    }

    // set tensor data
    {
        std::vector<uint8_t> read_buf;
        auto set_tensor = [&](ggml_tensor * orig, ggml_tensor * dev) {
            if (ml && ml->files.empty()) {
                ml->set_tensor_data(dev, ml->set_tensor_data_ud);
                return;
            }
            const auto * weight = ml ? &ml->require_weight(orig->name) : nullptr;
            const size_t offs = weight ? weight->offs :
                gguf_get_data_offset(metadata) + gguf_get_tensor_offset(metadata, gguf_find_tensor(metadata, orig->name));
            llama_file * source = weight ? ml->files.at(weight->idx).get() : gguf_file.get();
            const size_t size = ggml_nbytes(orig);
            if (offs + size < offs || offs + size > source->size()) {
                throw std::runtime_error(format("LoRA tensor '%s' data is not within the file bounds, file is corrupted or incomplete", orig->name));
            }
            const bool host = ggml_backend_buffer_is_host(dev->buffer);
            if (!host) {
                read_buf.resize(size);
            }
            void * data = host ? dev->data : read_buf.data();
            source->seek(offs, SEEK_SET);
            source->read_raw(data, size);
            if (ml && ml->check_tensors && !ggml_validate_row_data(orig->type, data, size)) {
                throw std::runtime_error(format("embedded LoRA tensor '%s' has invalid data", orig->name));
            }
            if (!host) {
                ggml_backend_tensor_set(dev, data, 0, size);
            }
        };
        for (auto & it : adapter.ab_map) {
            auto orig = ab_map[it.first];
            auto dev  = it.second;
            set_tensor(orig.a, dev.a);
            set_tensor(orig.b, dev.b);
        }
    }

    // register adapter with model
    model.loras.insert(&adapter);

    LLAMA_LOG_INFO("%s: loaded %zu tensors from lora file\n", __func__, adapter.ab_map.size()*2);
}

llama_adapter_lora * llama_adapter_lora_init_embedded(llama_model & model, llama_model_loader & ml) {
    auto adapter = std::make_unique<llama_adapter_lora>(&model);
    llama_adapter_lora_init_impl(model, nullptr, *adapter, &ml);
    return adapter.release();
}

llama_adapter_lora * llama_adapter_lora_init(llama_model * model, const char * path_lora) {
    LLAMA_LOG_INFO("%s: loading lora adapter from '%s' ...\n", __func__, path_lora);

    FILE * file = ggml_fopen(path_lora, "rb");
    if (!file) {
        LLAMA_LOG_ERROR("%s: failed to open '%s': %s\n", __func__, path_lora, strerror(errno));
        return nullptr;
    }

    llama_adapter_lora * adapter = llama_adapter_lora_init_from_file_ptr(model, file);
    fclose(file);

    return adapter;
}

llama_adapter_lora * llama_adapter_lora_init_from_file_ptr(llama_model * model, FILE * file) {
    if (!file) {
        LLAMA_LOG_ERROR("%s: file is NULL\n", __func__);
        return nullptr;
    }

    llama_adapter_lora * adapter = new llama_adapter_lora(model);

    try {
        llama_adapter_lora_init_impl(*model, file, *adapter);
        return adapter;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());

        delete adapter;
    }

    return nullptr;
}

int32_t llama_adapter_meta_val_str(const llama_adapter_lora * adapter, const char * key, char * buf, size_t buf_size) {
    const auto & it = adapter->gguf_kv.find(key);
    if (it == adapter->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_adapter_meta_count(const llama_adapter_lora * adapter) {
    return (int)adapter->gguf_kv.size();
}

int32_t llama_adapter_meta_key_by_index(const llama_adapter_lora * adapter, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)adapter->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = adapter->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_adapter_meta_val_str_by_index(const llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)adapter->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = adapter->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

void llama_adapter_lora_free(llama_adapter_lora * adapter) {
    if (adapter == nullptr) {
        return;
    }

    // The model owns this mandatory correction; external adapter APIs must not detach it.
    if (adapter->model && adapter->model->embedded_lora == adapter) {
        LLAMA_LOG_ERROR("%s: cannot free the model's embedded adapter\n", __func__);
        return;
    }
    if (adapter->model != nullptr) {
        adapter->model->loras.erase(adapter);
        adapter->model = nullptr;
    }

    delete adapter;
}

uint64_t llama_adapter_get_alora_n_invocation_tokens(const struct llama_adapter_lora * adapter) {
    if (!adapter) {
        return 0;
    }
    return adapter->alora_invocation_tokens.size();
}

const llama_token * llama_adapter_get_alora_invocation_tokens(const llama_adapter_lora * adapter) {
    GGML_ASSERT(adapter);
    return adapter->alora_invocation_tokens.data();
}

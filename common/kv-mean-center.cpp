#include "kv-mean-center.h"

#include "log.h"

#include "ggml.h"
#include "gguf.h"

#include <cstring>

bool common_kv_mean_center_write(
        const std::string & fname,
        const std::vector<common_kv_mean_center_layer> & layers,
        bool k_rot) {
    size_t n_with_bias = 0;
    for (const auto & layer : layers) {
        if (!layer.bias.empty()) {
            n_with_bias++;
        }
    }

    if (n_with_bias == 0) {
        LOG_ERR("%s: no layers with bias data to write\n", __func__);
        return false;
    }

    size_t data_size = 0;
    for (const auto & layer : layers) {
        if (!layer.bias.empty()) {
            data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float)*layer.bias.size(), GGML_MEM_ALIGN);
        }
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ data_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context  * ctx      = ggml_init(params);
    struct gguf_context   * ctx_gguf = gguf_init_empty();

    if (!ctx || !ctx_gguf) {
        LOG_ERR("%s: failed to allocate ggml/gguf context\n", __func__);
        if (ctx)      ggml_free(ctx);
        if (ctx_gguf) gguf_free(ctx_gguf);
        return false;
    }

    gguf_set_val_str(ctx_gguf, "general.type", "kv-mean-center");
    gguf_set_val_bool(ctx_gguf, "kv_mean_center.k_rot", k_rot);

    for (const auto & layer : layers) {
        if (layer.bias.empty()) {
            continue;
        }

        const std::string name = "kv_bar.blk." + std::to_string(layer.il) + ".k";

        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) layer.bias.size());
        ggml_set_name(t, name.c_str());

        memcpy(t->data, layer.bias.data(), layer.bias.size()*sizeof(float));

        gguf_add_tensor(ctx_gguf, t);
    }

    const bool ok = gguf_write_to_file(ctx_gguf, fname.c_str(), false);
    if (!ok) {
        LOG_ERR("%s: failed to write %s\n", __func__, fname.c_str());
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return ok;
}

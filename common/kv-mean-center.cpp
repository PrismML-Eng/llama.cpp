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

static bool kv_mean_center_is_view_op(enum ggml_op op) {
    return op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE || op == GGML_OP_CONT;
}

// fallback: the rotation matrix input by name, within a few links of the captured tensor.
// matched as a substring: views append a suffix ("attn_inp_k_rot (reshaped)") and the backend
// scheduler decorates split inputs with a backend prefix and split index ("MTL0#attn_inp_k_rot#0")
static bool kv_mean_center_has_k_rot_input(const struct ggml_tensor * t, int depth) {
    if (t == nullptr || depth < 0) {
        return false;
    }
    if (strstr(t->name, "attn_inp_k_rot") != nullptr || strstr(t->name, "attn_rot_dev_") != nullptr) {
        return true;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (t->src[i] && kv_mean_center_has_k_rot_input(t->src[i], depth - 1)) {
            return true;
        }
    }
    return false;
}

bool common_kv_mean_center_k_rotated(const struct ggml_tensor * t) {
    // only look through views: a Hadamard MUL_MAT further up (e.g. a Hadamard-latent weight
    // rotating the input of wk) does not put the cached K in the rotated basis
    const struct ggml_tensor * cur = t;
    for (int i = 0; cur != nullptr && i < 4; ++i) {
        if (cur->op == GGML_OP_MUL_MAT) {
            if (cur->op_params[1] == GGML_HINT_SRC0_IS_HADAMARD) { // see ggml_mul_mat_set_hint
                return true;
            }
            break;
        }
        if (!kv_mean_center_is_view_op(cur->op)) {
            break;
        }
        cur = cur->src[0];
    }

    return kv_mean_center_has_k_rot_input(t, 8);
}

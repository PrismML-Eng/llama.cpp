// prism.hadamard metadata may name tensors that the loader deliberately leaves unloaded (TENSOR_SKIP, e.g. the
// MTP block when speculation is off). Loading must skip those entries, while an entry for a tensor that is
// genuinely missing from the file must still fail.
//
// The test builds a tiny Qwen3.5 model with one MTP block, saves it, adds prism.hadamard metadata naming an
// MTP weight, and loads it back on the CPU with load_mtp = false.

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const uint32_t n_vocab = 128;
static const uint32_t n_embd  = 256;
static const uint32_t n_ff    = 384;
static const uint32_t n_head  = 2;
static const uint32_t n_trunk = 2; // decoder layers, the MTP block is blk.2

static void set_tensor_data(struct ggml_tensor * tensor, void * /*userdata*/) {
    std::vector<uint8_t> zeros(ggml_nbytes(tensor), 0);
    ggml_backend_tensor_set(tensor, zeros.data(), 0, zeros.size());
}

static bool silent_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static llama_model_params cpu_model_params(bool load_mtp) {
    static ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);

    llama_model_params params = llama_model_default_params();
    params.devices           = devs;
    params.progress_callback = silent_progress;
    params.load_mtp          = load_mtp;
    return params;
}

// tiny qwen35 with n_trunk decoder layers plus one MTP block, written to path
static bool write_base_model(const std::string & path) {
    gguf_context * meta = gguf_init_empty();
    const uint32_t n_embd_head = n_embd / n_head;

    gguf_set_val_str(meta, "general.architecture",                    "qwen35");
    gguf_set_val_u32(meta, "qwen35.vocab_size",                       n_vocab);
    gguf_set_val_u32(meta, "qwen35.context_length",                   128);
    gguf_set_val_u32(meta, "qwen35.embedding_length",                 n_embd);
    gguf_set_val_u32(meta, "qwen35.feed_forward_length",              n_ff);
    gguf_set_val_u32(meta, "qwen35.block_count",                      n_trunk + 1);
    gguf_set_val_u32(meta, "qwen35.nextn_predict_layers",             1);
    gguf_set_val_u32(meta, "qwen35.full_attention_interval",          2);
    gguf_set_val_u32(meta, "qwen35.attention.head_count",             n_head);
    gguf_set_val_u32(meta, "qwen35.attention.head_count_kv",          n_head);
    gguf_set_val_f32(meta, "qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
    const uint32_t sections[4] = { n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4 };
    gguf_set_arr_data(meta, "qwen35.rope.dimension_sections", GGUF_TYPE_UINT32, sections, 4);
    gguf_set_val_u32(meta, "qwen35.ssm.conv_kernel",                  4);
    gguf_set_val_u32(meta, "qwen35.ssm.inner_size",                   256);
    gguf_set_val_u32(meta, "qwen35.ssm.state_size",                   128);
    gguf_set_val_u32(meta, "qwen35.ssm.time_step_rank",               n_head);
    gguf_set_val_u32(meta, "qwen35.ssm.group_count",                  2);
    gguf_set_val_str(meta, "tokenizer.ggml.model",                    "no_vocab");

    // the MTP block is only created when it is loaded
    llama_model * model = llama_model_init_from_user(meta, set_tensor_data, nullptr, cpu_model_params(/*load_mtp=*/ true));
    gguf_free(meta);
    if (model == nullptr) {
        return false;
    }
    llama_model_save_to_file(model, path.c_str());
    llama_model_free(model);
    return true;
}

// copy src to dst with prism.hadamard metadata naming weight_names
static bool write_hadamard_model(const std::string & src, const std::string & dst, const std::vector<std::string> & weight_names) {
    ggml_context * ctx_data = nullptr;
    gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx_data };
    gguf_context * in = gguf_init_from_file(src.c_str(), params);
    if (in == nullptr) {
        return false;
    }

    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);
    // the saver writes this for the trunk layers only; the MTP block is a full-attention (non-recurrent) layer
    const bool recurrent[n_trunk + 1] = { true, false, false };
    gguf_set_arr_data(out, "qwen35.attention.recurrent_layers", GGUF_TYPE_BOOL, recurrent, n_trunk + 1);
    for (int64_t i = 0; i < gguf_get_n_tensors(in); i++) {
        const std::string name = gguf_get_tensor_name(in, i);
        // the optional .scale/.input_scale tensors are only looked up for loaded weights, so the skipped MTP block's
        // scales would be left over; they are not needed here
        if (name.size() > 5 && name.compare(name.size() - 5, 5, "scale") == 0) {
            continue;
        }
        gguf_add_tensor(out, ggml_get_tensor(ctx_data, name.c_str()));
    }

    std::vector<const char *> names;
    for (const std::string & name : weight_names) {
        names.push_back(name.c_str());
    }
    gguf_set_val_u32(out, "prism.hadamard.version",    1);
    gguf_set_val_u32(out, "prism.hadamard.block_size", 64);
    gguf_set_val_str(out, "prism.hadamard.transform",  "normalized-sylvester-walsh-hadamard");
    gguf_set_val_str(out, "prism.hadamard.axis",       "input-last-dimension");
    gguf_set_val_str(out, "prism.hadamard.sign_mode",  "identity");
    gguf_set_arr_str(out, "prism.hadamard.weight_names", names.data(), names.size());

    const bool ok = gguf_write_to_file(out, dst.c_str(), /*only_meta =*/ false);

    gguf_free(out);
    gguf_free(in);
    ggml_free(ctx_data);
    return ok;
}

// log text of the current load, so the checks can tell why a load passed or failed
static std::string g_log;

static void capture_log(ggml_log_level /*level*/, const char * text, void * /*user_data*/) {
    g_log += text;
}

static bool log_has(const std::string & needle) {
    return g_log.find(needle) != std::string::npos;
}

static bool loads(const std::string & path) {
    g_log.clear();
    llama_model * model = llama_model_load_from_file(path.c_str(), cpu_model_params(/*load_mtp=*/ false));
    if (model == nullptr) {
        return false;
    }
    llama_model_free(model);
    return true;
}

int main(void) {
    llama_backend_init();
    llama_log_set(capture_log, nullptr);

    const std::string base    = "test-prism-hadamard-skip-base.gguf";
    const std::string skipped = "test-prism-hadamard-skip-skipped.gguf";
    const std::string missing = "test-prism-hadamard-skip-missing.gguf";

    int ret = EXIT_SUCCESS;

    if (!write_base_model(base) ||
        // blk.0.ffn_up is loaded and rotated, blk.2.ffn_up is the MTP block and skipped with load_mtp = false
        !write_hadamard_model(base, skipped, { "blk.0.ffn_up.weight", "blk.2.ffn_up.weight" }) ||
        // blk.5 does not exist in the file at all
        !write_hadamard_model(base, missing, { "blk.0.ffn_up.weight", "blk.5.ffn_up.weight" })) {
        fprintf(stderr, "FAIL: could not build the test models\n");
        ret = EXIT_FAILURE;
    } else {
        if (!loads(skipped) || !log_has("unused tensor blk.2.ffn_up.weight")) {
            fprintf(stderr, "FAIL: prism.hadamard entry for a skipped (unloaded) tensor broke loading\n%s", g_log.c_str());
            ret = EXIT_FAILURE;
        }
        if (loads(missing) || !log_has("prism.hadamard weight not found: blk.5.ffn_up.weight")) {
            fprintf(stderr, "FAIL: prism.hadamard entry for a missing tensor did not fail loading\n%s", g_log.c_str());
            ret = EXIT_FAILURE;
        }
    }

    remove(base.c_str());
    remove(skipped.c_str());
    remove(missing.c_str());

    llama_backend_free();

    if (ret == EXIT_SUCCESS) {
        fprintf(stderr, "OK\n");
    }
    return ret;
}

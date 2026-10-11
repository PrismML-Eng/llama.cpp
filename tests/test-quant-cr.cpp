#include "../src/llama-ext.h"
#include "ggml-cpp.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);

    llama_quant_model_desc desc = {};
    desc.architecture          = "llama";
    desc.n_embd                = 256;
    desc.n_ff                  = 256;
    desc.n_layer               = 5;
    desc.n_head                = 4;
    desc.n_head_kv             = 4;
    desc.n_embd_head_k         = 64;
    desc.n_embd_head_v         = 64;
    llama_model * model = llama_quant_model_from_metadata(&desc);
    llama_model_quantize_params qparams = llama_model_quantize_default_params();
    quantize_state_impl * qs = llama_quant_init(model, &qparams);

    const struct {
        llama_ftype ftype;
        ggml_type type;
        ggml_type fallback;
        ggml_type narrow_fallback;
    } cases[] = {
        { LLAMA_FTYPE_MOSTLY_Q8_CR, GGML_TYPE_Q8_CR, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 },
        { LLAMA_FTYPE_MOSTLY_Q5_CR, GGML_TYPE_Q5_CR, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0 },
        { LLAMA_FTYPE_MOSTLY_Q6_CR, GGML_TYPE_Q6_CR, GGML_TYPE_Q6_K, GGML_TYPE_F16  },
    };

    int n_failed = 0;
    for (const int64_t ncols : { int64_t(256), int64_t(288) }) {
        const struct {
            const char * name;
            int64_t ne[4];
        } tensor_specs[] = {
            { "blk.0.ffn_up.weight", { ncols, 2, 1, 1 } }, // 288 fits 32-value blocks, not CR rotation groups
            { "blk.1.ffn_up.weight", { 128, 2, 1, 1 } }, // incompatible rotation group size
            { "blk.2.ffn_up.weight", {  30, 2, 1, 1 } }, // incompatible fallback storage block size
            { "blk.3.ffn_up.weight", { 256, 2, 2, 1 } }, // 3D expert tensor
            { "blk.4.ffn_up.weight", { 256, 2, 1, 2 } }, // nontrivial fourth dimension
            { "output.weight",      { ncols, 2, 1, 1 } },
            { "token_embd.weight",  { ncols, 2, 1, 1 } },
        };
        const size_t n_tensors = sizeof(tensor_specs) / sizeof(tensor_specs[0]);
        ggml_init_params ctx_params = { n_tensors * ggml_tensor_overhead(), nullptr, true };
        ggml_context_ptr ctx(ggml_init(ctx_params));
        std::vector<ggml_tensor *> tensors;
        for (const auto & spec : tensor_specs) {
            auto * tensor = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32,
                                              spec.ne[0], spec.ne[1], spec.ne[2], spec.ne[3]);
            ggml_set_name(tensor, spec.name);
            tensors.push_back(tensor);
        }

        for (const auto & c : cases) {
            // At 288 columns, output policy selects Q8_0 for every CR type.
            // Q6_K embeddings also reach Q8_0 through their shape fallback.
            const ggml_type wide_policy_type = c.ftype == LLAMA_FTYPE_MOSTLY_Q6_CR ? GGML_TYPE_Q8_0 : c.fallback;
            const ggml_type expected[] = {
                ncols == 256 ? c.type : c.narrow_fallback,
                c.narrow_fallback,
                GGML_TYPE_F16,
                c.fallback,
                c.fallback,
                ncols == 256 ? c.type : GGML_TYPE_Q8_0,
                ncols == 256 ? c.fallback : wide_policy_type,
            };
            const ggml_type default_type = llama_ftype_get_default_type(c.ftype);
            bool pass = default_type == c.type;
            if (!pass) {
                printf("FAIL %s default type: expected %s, got %s\n",
                       ggml_type_name(c.type), ggml_type_name(c.type), ggml_type_name(default_type));
            }

            std::vector<ggml_type> result_types(n_tensors);
            llama_quant_compute_types(qs, c.ftype, tensors.data(), result_types.data(), n_tensors);
            for (size_t i = 0; i < n_tensors; ++i) {
                if (result_types[i] != expected[i]) {
                    printf("FAIL %s ncols=%lld %s: expected %s, got %s\n",
                           ggml_type_name(c.type), (long long) ncols, ggml_get_name(tensors[i]),
                           ggml_type_name(expected[i]), ggml_type_name(result_types[i]));
                    pass = false;
                }
            }
            printf("%s %s ncols=%lld\n", pass ? "PASS" : "FAIL", ggml_type_name(c.type), (long long) ncols);
            n_failed += !pass;
        }
    }

    llama_quant_free(qs);
    llama_model_free(model);
    return n_failed > 0 ? 1 : 0;
}

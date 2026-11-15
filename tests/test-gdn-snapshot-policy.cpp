#include "../src/llama-memory-hybrid-iswa.h"
#include "../src/models/models.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

struct policy_model : llama_model_base {
    policy_model() : llama_model_base(llama_model_default_params()) {
        hparams.n_layer_all = 1;
        hparams.n_embd = 8;
        hparams.n_head_arr[0] = 2;
        hparams.n_head_kv_arr[0] = 1;
        hparams.n_embd_head_k_full = 4;
        hparams.n_embd_head_v_full = 4;
        hparams.n_embd_head_k_swa = 4;
        hparams.n_embd_head_v_swa = 4;
        hparams.is_recr_impl[0] = 1;
        hparams.is_swa_impl[0] = 0;
        hparams.n_swa = 32;
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hparams.ssm_d_conv = 2;
        hparams.ssm_d_inner = 4;
        hparams.ssm_d_state = 2;
        hparams.ssm_n_group = 1;
    }

    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override {
        throw std::runtime_error("model graph is not used in this test");
    }
};

static void test_hybrid_iswa_forwarding() {
    policy_model model;
    llama_memory_hybrid_iswa memory(model,
            GGML_TYPE_F32, GGML_TYPE_F32, false, false, 256, 8, 32,
            GGML_TYPE_F32, GGML_TYPE_F32, 4, 2, 3, false, true);
    const auto recurrent = memory.get_mem_recr();
    check(recurrent->rs_n_snap == std::vector<uint32_t>({3, 3}), "initial budgets differ");

    // The exported caller dispatches through llama_memory_i, not the concrete type.
    check(llama_memory_seq_rs_snapshots(&memory, 0, 0), "hybrid ISWA did not forward opt-out");
    check(recurrent->rs_n_snap == std::vector<uint32_t>({0, 3}), "opt-out changed the wrong budget");
    check(llama_memory_seq_rs_snapshots(&memory, 1, 1), "second sequence budget failed");
    check(recurrent->rs_n_snap == std::vector<uint32_t>({0, 1}), "second sequence isolation failed");
    check(llama_memory_seq_rs_snapshots(&memory, 0, 3), "restore budget failed");
    check(recurrent->rs_n_snap == std::vector<uint32_t>({3, 1}), "restore changed the wrong budget");
    for (const auto seq : {-1, 2}) {
        check(!llama_memory_seq_rs_snapshots(&memory, seq, 0), "invalid sequence accepted");
        check(recurrent->rs_n_snap == std::vector<uint32_t>({3, 1}), "invalid sequence mutated budget");
    }
    check(!llama_memory_seq_rs_snapshots(&memory, 0, 4), "oversized budget accepted");
    check(recurrent->rs_n_snap == std::vector<uint32_t>({3, 1}), "invalid budget mutated state");
    memory.clear(false);
    check(recurrent->rs_n_snap == std::vector<uint32_t>({3, 1}), "clear lost snapshot policy");
}

static void test_cpu_only_inplace_device_policy() {
    struct device_case {
        enum ggml_backend_dev_type type;
        bool inplace;
        bool rows[4];
    };
    const device_case cases[] = {
        {GGML_BACKEND_DEVICE_TYPE_CPU,   true,  {true,  true,  true,  true }},
        {GGML_BACKEND_DEVICE_TYPE_GPU,   false, {false, true,  false, false}},
        {GGML_BACKEND_DEVICE_TYPE_IGPU,  false, {false, true,  false, false}},
        {GGML_BACKEND_DEVICE_TYPE_ACCEL, false, {false, false, false, false}},
        {GGML_BACKEND_DEVICE_TYPE_META,  false, {false, false, false, false}},
    };
    const char * names[] = {"CPU", "MTL", "CUDA", "META"};
    for (auto test : cases) {
        ggml_backend_device device{};
        device.context = &test.type;
        device.iface.get_type = [](ggml_backend_dev_t dev) {
            return *static_cast<enum ggml_backend_dev_type *>(dev->context);
        };
        check(llama_model_qwen35::graph::supports_gdn_state_inplace(&device) == test.inplace,
                "cache-mutating GDN accepted a non-CPU device");
        for (size_t i = 0; i < 4; ++i) {
            ggml_backend_reg registry{};
            registry.context = &names[i];
            registry.iface.get_name = [](ggml_backend_reg_t reg) {
                return *static_cast<const char **>(reg->context);
            };
            device.reg = &registry;
            const auto reg = ggml_backend_dev_backend_reg(&device);
            const bool is_metal = std::strcmp(ggml_backend_reg_name(reg), "MTL") == 0;
            check(llama_model_qwen35::graph::supports_gdn_state_rows(&device, is_metal) == test.rows[i],
                    "rows-indexed GDN accepted an unsupported device");
        }
    }
}

int main() {
    try {
        test_hybrid_iswa_forwarding();
        test_cpu_only_inplace_device_policy();
        std::puts("hybrid ISWA forwarding and device-type policy checks passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}

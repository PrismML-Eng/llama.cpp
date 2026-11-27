#include "llama.h"
#include "gguf.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;

static void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool special) {
    std::vector<llama_token> result(text.size() + 8);
    const int count = llama_tokenize(vocab, text.data(), text.size(), result.data(), result.size(), false, special);
    if (count < 0) {
        throw std::runtime_error("unexpected token buffer exhaustion");
    }
    result.resize(count);
    return result;
}

static std::string detokenize(const llama_vocab * vocab, const std::vector<llama_token> & tokens, bool special) {
    std::vector<char> result(1024);
    const int count = llama_detokenize(vocab, tokens.data(), tokens.size(),
                                      result.data(), result.size(), false, special);
    require(count >= 0, "unexpected detokenization buffer exhaustion");
    return std::string(result.data(), count);
}

// Metadata-only BPE fixtures exercise the public ABI, including against old DLLs.
// No model weights, private vocabulary entry points, or new exported functions.
struct eog_fixture {
    enum token_id {
        eos, pad, literal_s, fim_pre, fim_mid, fim_suf, fim_pad, fim_rep, fim_sep,
        tool_call, tool_call_end, tool_response, tool_response_end, think, think_end,
        eom, eot, return_token, call_token, end_token, channel,
    };
    enum class policy_type { absent, ids, scalar, signed_ids, float_ids, string_ids };

    std::vector<std::string> tokens = {
        "<|im_end|>", "<|endoftext|>", "</s>",
        "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
        "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
        "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
        "<think>", "</think>", "<|eom_id|>", "<|eot_id|>",
        "<|return|>", "<|call|>", "<|end|>", "<|channel|>",
    };
    std::vector<int32_t> types = {
        LLAMA_TOKEN_TYPE_CONTROL, LLAMA_TOKEN_TYPE_CONTROL, LLAMA_TOKEN_TYPE_NORMAL,
        LLAMA_TOKEN_TYPE_USER_DEFINED, LLAMA_TOKEN_TYPE_USER_DEFINED, LLAMA_TOKEN_TYPE_USER_DEFINED,
        LLAMA_TOKEN_TYPE_USER_DEFINED, LLAMA_TOKEN_TYPE_USER_DEFINED, LLAMA_TOKEN_TYPE_USER_DEFINED,
        LLAMA_TOKEN_TYPE_USER_DEFINED, LLAMA_TOKEN_TYPE_USER_DEFINED,
        LLAMA_TOKEN_TYPE_USER_DEFINED, LLAMA_TOKEN_TYPE_USER_DEFINED,
        LLAMA_TOKEN_TYPE_USER_DEFINED, LLAMA_TOKEN_TYPE_USER_DEFINED,
        LLAMA_TOKEN_TYPE_USER_DEFINED, LLAMA_TOKEN_TYPE_NORMAL,
        LLAMA_TOKEN_TYPE_CONTROL, LLAMA_TOKEN_TYPE_CONTROL, LLAMA_TOKEN_TYPE_CONTROL, LLAMA_TOKEN_TYPE_CONTROL,
    };
    bool named_turn_ids = true;
    bool normalizer_nfc = false;

    eog_fixture() {
        for (int c = '!'; c <= '~'; ++c) {
            tokens.emplace_back(1, char(c));
            types.push_back(LLAMA_TOKEN_TYPE_NORMAL);
        }
        tokens.push_back("\xc4\xa0"); // byte-level space
        tokens.push_back("\xc4\x8a"); // byte-level newline
        types.insert(types.end(), 2, LLAMA_TOKEN_TYPE_NORMAL);
    }

    llama_token ascii(char c) const {
        const auto it = std::find(tokens.begin(), tokens.end(), std::string(1, c));
        require(it != tokens.end(), "missing fixture character");
        return llama_token(it - tokens.begin());
    }

    void write(const std::filesystem::path & path, policy_type policy, const std::vector<uint32_t> & ids) const {
        std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(gguf_init_empty(), gguf_free);
        gguf_set_val_str(ctx.get(), "general.architecture", "llama");
        gguf_set_val_str(ctx.get(), "tokenizer.ggml.model", "gpt2");
        gguf_set_val_str(ctx.get(), "tokenizer.ggml.pre", "qwen2");
        std::vector<const char *> text;
        for (const auto & token : tokens) {
            text.push_back(token.c_str());
        }
        gguf_set_arr_str(ctx.get(), "tokenizer.ggml.tokens", text.data(), text.size());
        gguf_set_arr_data(ctx.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
        const char * empty = "";
        gguf_set_arr_str(ctx.get(), "tokenizer.ggml.merges", &empty, 0);
        gguf_set_val_u32(ctx.get(), "tokenizer.ggml.eos_token_id", eos);
        gguf_set_val_u32(ctx.get(), "tokenizer.ggml.bos_token_id", pad);
        gguf_set_val_u32(ctx.get(), "tokenizer.ggml.padding_token_id", pad);
        gguf_set_val_bool(ctx.get(), "tokenizer.ggml.add_bos_token", false);
        gguf_set_val_bool(ctx.get(), "tokenizer.ggml.add_eos_token", false);
        gguf_set_val_bool(ctx.get(), "tokenizer.ggml.normalizer.nfc", normalizer_nfc);
        if (named_turn_ids) {
            gguf_set_val_u32(ctx.get(), "tokenizer.ggml.eot_token_id", eot);
            gguf_set_val_u32(ctx.get(), "tokenizer.ggml.eom_token_id", eom);
        }
        const char * key = "tokenizer.ggml.eog_token_ids";
        const uint32_t dummy = 0;
        switch (policy) {
            case policy_type::absent:
                break;
            case policy_type::ids:
                gguf_set_arr_data(ctx.get(), key, GGUF_TYPE_UINT32, ids.empty() ? &dummy : ids.data(), ids.size());
                break;
            case policy_type::scalar:
                gguf_set_val_u32(ctx.get(), key, eos);
                break;
            case policy_type::signed_ids:
                gguf_set_arr_data(ctx.get(), key, GGUF_TYPE_INT32, ids.empty() ? &dummy : ids.data(), ids.size());
                break;
            case policy_type::float_ids: {
                const float id = eos;
                gguf_set_arr_data(ctx.get(), key, GGUF_TYPE_FLOAT32, &id, 1);
                break;
            }
            case policy_type::string_ids: {
                const char * id = "0";
                gguf_set_arr_str(ctx.get(), key, &id, 1);
                break;
            }
        }
        require(gguf_write_to_file(ctx.get(), path.string().c_str(), true), "cannot write tiny GGUF fixture");
    }

    model_ptr load(const std::filesystem::path & path, policy_type policy, const std::vector<uint32_t> & ids) const {
        write(path, policy, ids);
        auto params = llama_model_default_params();
        params.vocab_only = true;
        return model_ptr(llama_model_load_from_file(path.string().c_str(), params), llama_model_free);
    }
};

struct fixture_directory {
    std::filesystem::path path;

    fixture_directory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto base = std::filesystem::temp_directory_path();
        for (unsigned i = 0; ; ++i) {
            path = base / ("llama-eog-" + std::to_string(stamp) + "-" + std::to_string(i));
            if (std::filesystem::create_directory(path)) {
                break;
            }
        }
    }

    ~fixture_directory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

static void check_eog_set(const llama_vocab * vocab, const std::vector<uint32_t> & expected) {
    for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); ++id) {
        const bool is_eog = std::find(expected.begin(), expected.end(), uint32_t(id)) != expected.end();
        require(llama_vocab_is_eog(vocab, id) == is_eog, "EOG membership mismatch at token " + std::to_string(id));
    }
    require(!llama_vocab_is_eog(vocab, LLAMA_TOKEN_NULL), "null token must not be EOG");
}

static void check_lexical_policy(const llama_vocab * vocab, const eog_fixture & fixture) {
    using F = eog_fixture;
    for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); ++id) {
        const auto expected = fixture.types[id] == LLAMA_TOKEN_TYPE_CONTROL ? LLAMA_TOKEN_ATTR_CONTROL :
                              fixture.types[id] == LLAMA_TOKEN_TYPE_USER_DEFINED ? LLAMA_TOKEN_ATTR_USER_DEFINED :
                              LLAMA_TOKEN_ATTR_NORMAL;
        require(llama_vocab_get_attr(vocab, id) == expected, "lexical attribute mismatch at token " + std::to_string(id));
    }
    require(llama_vocab_eos(vocab) == F::eos, "named EOS changed");
    require(llama_vocab_fim_pre(vocab) == F::fim_pre && llama_vocab_fim_mid(vocab) == F::fim_mid &&
            llama_vocab_fim_suf(vocab) == F::fim_suf && llama_vocab_fim_pad(vocab) == F::fim_pad &&
            llama_vocab_fim_rep(vocab) == F::fim_rep && llama_vocab_fim_sep(vocab) == F::fim_sep,
            "FIM IDs were not resolved");
    for (llama_token id = F::literal_s; id <= F::eot; ++id) {
        const auto & text = fixture.tokens[id];
        const bool control = fixture.types[id] == LLAMA_TOKEN_TYPE_CONTROL;
        const bool user_defined = fixture.types[id] == LLAMA_TOKEN_TYPE_USER_DEFINED;
        for (bool special : {false, true}) {
            std::vector<llama_token> expected;
            if (user_defined || (control && special)) {
                expected.push_back(id);
            } else {
                for (char c : text) {
                    expected.push_back(fixture.ascii(c));
                }
            }
            require(tokenize(vocab, text, special) == expected, "marker tokenization mismatch: " + text);
            expected.insert(expected.begin(), fixture.ascii('x'));
            expected.push_back(fixture.ascii('y'));
            require(tokenize(vocab, "x" + text + "y", special) == expected, "embedded marker mismatch: " + text);
        }
        require(detokenize(vocab, {id}, false) == (control ? "" : text), "suppressed detokenization mismatch: " + text);
        require(detokenize(vocab, {id}, true) == text, "special detokenization mismatch: " + text);
    }
    require(tokenize(vocab, fixture.tokens[F::eos], true) == std::vector<llama_token>{F::eos},
            "canonical EOS special parsing changed");
    require(detokenize(vocab, {F::eos}, false).empty(), "canonical EOS must remain CONTROL");
}

static void test_eog_policy() {
    using F = eog_fixture;
    using P = F::policy_type;
    fixture_directory directory;
    const auto path = directory.path / "vocab.gguf";
    F fixture;
    const auto check = [&] (const char * name, const F & f, const std::vector<uint32_t> & ids) {
        try {
            auto model = f.load(path, P::ids, ids);
            require(bool(model), "valid EOG fixture failed to load");
            const auto * vocab = llama_model_get_vocab(model.get());
            check_eog_set(vocab, ids);
            check_lexical_policy(vocab, f);
        } catch (const std::exception & error) {
            throw std::runtime_error(std::string(name) + ": " + error.what());
        }
    };
    check("canonical single stop", fixture, {F::eos});
    check("multiple stops with ordinary and nonspecial tokens", fixture, {F::eos, F::literal_s, F::tool_call});
    check("explicit empty set overrides EOS/EOT/EOM", fixture, {});
    check("duplicate stop IDs", fixture, {F::eos, F::eos});

    F normal_fim = fixture;
    for (int id = F::fim_pre; id <= F::fim_sep; ++id) {
        normal_fim.types[id] = LLAMA_TOKEN_TYPE_NORMAL;
    }
    check("NORMAL FIM attributes", normal_fim, {F::eos});
    F control_fim = fixture;
    for (int id = F::fim_pre; id <= F::fim_sep; ++id) {
        control_fim.types[id] = LLAMA_TOKEN_TYPE_CONTROL;
    }
    check("CONTROL FIM is not implicitly EOG", control_fim, {F::eos});

    F inferred_turns = fixture;
    inferred_turns.named_turn_ids = false;
    inferred_turns.tokens[F::eos] = "<stop>";
    inferred_turns.tokens[F::pad] = "<pad>";
    inferred_turns.tokens[F::end_token] = "<end_marker>";
    check("inferred EOT/EOM preserve lexical attributes", inferred_turns, {});

    F removal = fixture;
    removal.tokens.push_back("<|tool_response>");
    removal.types.push_back(LLAMA_TOKEN_TYPE_CONTROL);
    check("legacy exceptions cannot remove explicit stops", removal,
          {F::eos, F::literal_s, F::return_token, F::call_token, F::end_token, uint32_t(removal.tokens.size() - 1)});

    // Absent-key behavior deliberately retains the existing text heuristics,
    // including NORMAL </s> promotion and the earlier USER_DEFINED FIM fix.
    auto legacy = fixture.load(path, P::absent, {});
    require(bool(legacy), "absent-key fixture failed to load");
    const auto * vocab = llama_model_get_vocab(legacy.get());
    check_eog_set(vocab, {F::eos, F::pad, F::literal_s, F::eom, F::eot, F::return_token, F::call_token});
    require((llama_vocab_get_attr(vocab, F::literal_s) & LLAMA_TOKEN_ATTR_CONTROL) != 0,
            "absent-key </s> heuristic changed");
    require(tokenize(vocab, "</s>", true) == std::vector<llama_token>{F::literal_s},
            "absent-key special parsing changed");
    require(detokenize(vocab, {F::literal_s}, false).empty(), "absent-key special suppression changed");
    for (llama_token id = F::fim_pre; id <= F::think_end; ++id) {
        require(llama_vocab_get_attr(vocab, id) == LLAMA_TOKEN_ATTR_USER_DEFINED, "legacy nonspecial attributes changed");
        require(tokenize(vocab, fixture.tokens[id], false) == std::vector<llama_token>{id}, "legacy nonspecial parsing changed");
        require(detokenize(vocab, {id}, false) == fixture.tokens[id], "legacy nonspecial detokenization changed");
    }
    auto legacy_control = control_fim.load(path, P::absent, {});
    require(bool(legacy_control), "absent-key CONTROL fixture failed to load");
    check_eog_set(llama_model_get_vocab(legacy_control.get()),
                  {F::eos, F::pad, F::literal_s, F::eom, F::eot, F::return_token, F::call_token,
                   F::fim_pad, F::fim_rep, F::fim_sep});

    for (auto malformed : {P::scalar, P::signed_ids, P::float_ids, P::string_ids}) {
        require(!fixture.load(path, malformed, {F::eos}), "malformed EOG type was accepted");
    }
    require(!fixture.load(path, P::signed_ids, {}), "empty EOG array with wrong element type was accepted");
    require(!fixture.load(path, P::ids, {uint32_t(fixture.tokens.size())}), "out-of-range EOG ID was accepted");
    require(!fixture.load(path, P::ids, {std::numeric_limits<uint32_t>::max()}), "overflowing EOG ID was accepted");
}

static void test_policy_roundtrip() {
    using F = eog_fixture;
    fixture_directory directory;
    F fixture;
    fixture.normalizer_nfc = true;
    const llama_token first = llama_token(fixture.tokens.size());
    // Byte-level BPE spellings for UTF-8 C3 A9 (é) and CC 81 (combining acute).
    fixture.tokens.insert(fixture.tokens.end(), {"\xc3\x83", "\xc2\xa9", "\xc3\x8c", "\xc4\xa3"});
    fixture.types.insert(fixture.types.end(), 4, LLAMA_TOKEN_TYPE_NORMAL);
    const std::vector<llama_token> composed = {first, first + 1};
    for (const auto & ids : std::vector<std::vector<uint32_t>>{{}, {F::eos}, {F::eos, F::tool_call}}) {
        const auto source = directory.path / "original.gguf";
        fixture.write(source, F::policy_type::ids, ids);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_file(source.string().c_str(), {true, nullptr}), gguf_free);
        require(bool(metadata), "cannot read roundtrip fixture metadata");
        gguf_set_val_u32(metadata.get(), "llama.context_length", 128);
        gguf_set_val_u32(metadata.get(), "llama.embedding_length", 128);
        gguf_set_val_u32(metadata.get(), "llama.block_count", 1);
        gguf_set_val_u32(metadata.get(), "llama.feed_forward_length", 256);
        gguf_set_val_u32(metadata.get(), "llama.attention.head_count", 2);
        gguf_set_val_u32(metadata.get(), "llama.attention.head_count_kv", 2);
        gguf_set_val_f32(metadata.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        gguf_set_val_u32(metadata.get(), "llama.rope.dimension_count", 64);
        auto source_params = llama_model_default_params();
        static ggml_backend_dev_t cpu_only[] = {nullptr};
        source_params.devices = cpu_only;
        source_params.n_gpu_layers = 0;
        const auto fill = [](ggml_tensor * tensor, void *) {
            require(tensor->type == GGML_TYPE_F32, "unexpected synthetic model type");
            std::vector<float> values(ggml_nelements(tensor),
                std::strstr(tensor->name, "norm") ? 1.0f : 0.01f);
            ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
        };
        model_ptr original(llama_model_init_from_user(metadata.get(), fill, nullptr, source_params), llama_model_free);
        require(bool(original), "roundtrip source failed to load");
        const auto * original_vocab = llama_model_get_vocab(original.get());
        require(tokenize(original_vocab, "e\xcc\x81", false) == composed, "NFC fixture is invalid");
        const auto saved = directory.path / "saved.gguf";
        llama_model_save_to_file(original.get(), saved.string().c_str());
        auto params = llama_model_default_params();
        params.vocab_only = true;
        model_ptr restored(llama_model_load_from_file(saved.string().c_str(), params), llama_model_free);
        require(bool(restored), "saved vocabulary failed to load");
        const auto * restored_vocab = llama_model_get_vocab(restored.get());
        check_eog_set(restored_vocab, ids);
        check_lexical_policy(restored_vocab, fixture);
        require(tokenize(restored_vocab, "e\xcc\x81", false) == composed, "saving lost NFC normalization");
        require(tokenize(restored_vocab, "\xc3\xa9", false) == composed, "saving changed composed tokenization");
    }
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--eog-roundtrip") == 0) {
        try {
            test_policy_roundtrip();
            return 0;
        } catch (const std::exception & error) {
            fprintf(stderr, "Tokenizer policy roundtrip: %s\n", error.what());
            return 1;
        }
    }
    if (argc == 1 || (argc == 2 && std::strcmp(argv[1], "--eog-policy") == 0)) {
        try {
            test_eog_policy();
            return 0;
        } catch (const std::exception & error) {
            fprintf(stderr, "EOG policy regression: %s\n", error.what());
            return 1;
        }
    }
    if (argc != 2) {
        fprintf(stderr, "usage: %s [--eog-policy | --eog-roundtrip | <BPE-vocab.gguf>]\n", argv[0]);
        return 1;
    }
    model_ptr models[] = {{nullptr, llama_model_free}, {nullptr, llama_model_free}};
    for (int enabled = 0; enabled < 2; ++enabled) {
        llama_model_kv_override overrides[2] = {};
        overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        std::strcpy(overrides[0].key, "tokenizer.ggml.normalizer.nfc");
        overrides[0].val_bool = enabled;
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.kv_overrides = overrides;
        models[enabled].reset(llama_model_load_from_file(argv[1], params));
        if (!models[enabled]) {
            fprintf(stderr, "failed to load vocabulary\n");
            return 1;
        }
    }
    const auto * plain = llama_model_get_vocab(models[0].get());
    const auto * nfc = llama_model_get_vocab(models[1].get());
    if (llama_vocab_type(plain) != LLAMA_VOCAB_TYPE_BPE) {
        return 1;
    }
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"Cafe\xcc\x81", "Caf\xc3\xa9"},
        {"A\xcc\x8a", "\xc3\x85"},
        {"\xe2\x84\xab", "\xc3\x85"},
        {"a\xcc\x95\xcc\x80", "\xc3\xa0\xcc\x95"},
        {"\xe1\x84\x80\xe1\x85\xa1\xe1\x86\xa8", "\xea\xb0\x81"},
        {"Cafe\xcc\x81<|endoftext|>A\xcc\x8a", "Caf\xc3\xa9<|endoftext|>\xc3\x85"},
        {"ASCII\n123 's", "ASCII\n123 's"},
        {"\xef\xac\x81\xef\xbc\xa1", "\xef\xac\x81\xef\xbc\xa1"},
    };
    for (bool special : {false, true}) {
        for (const auto & test : cases) {
            const auto expected = tokenize(plain, test.second, special);
            if (tokenize(nfc, test.first, special) != expected ||
                tokenize(nfc, test.second, special) != expected ||
                (test.first != test.second && tokenize(plain, test.first, special) == expected)) {
                fprintf(stderr, "BPE NFC or disabled-normalization mismatch (special=%d)\n", special);
                return 1;
            }
        }
    }
    return 0;
}

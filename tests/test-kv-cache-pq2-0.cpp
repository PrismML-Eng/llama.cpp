// Test: PQ2_0 KV-cache type registration + parse.
//
// Red/green for the PQ2_0 KV-cache option.
//   - RED:   parsing "-ctk pq2_0" / "-ctv pq2_0" fails (type not in allowed list).
//   - GREEN: after adding GGML_TYPE_PQ2_0 to the allowed list + the string table,
//            parsing succeeds and lands on GGML_TYPE_PQ2_0.
//
// Mirrors tests/test-arg-parser.cpp (links common/arg.cpp via llama-common and uses
// common_params_parse). Exits non-zero on any failure (so it is a real failing test
// in the RED state).

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cassert>
#include <cstring>
#include <vector>

static std::vector<char *> argv_to_char(std::vector<std::string> & argv) {
    std::vector<char *> res;
    for (auto & arg : argv) {
        res.push_back(const_cast<char *>(arg.data()));
    }
    return res;
}

// Parse "-ctk <type>" and assert the resulting params.cache_type_k == expected.
// -m is required by LLAMA_EXAMPLE_COMMON; include a dummy model so the check
// isolates the KV-cache type parsing from the model requirement.
static bool check_k_type(std::string value, ggml_type expected) {
    common_params params;
    std::vector<std::string> argv = {"binary_name", "-m", "model.gguf", "-ctk", value};
    bool ok = common_params_parse(argv.size(), argv_to_char(argv).data(), params,
                                  LLAMA_EXAMPLE_COMMON);
    if (!ok) {
        fprintf(stderr, "  FAIL -ctk '%s': parse rejected\n", value.c_str());
        return false;
    }
    if (params.cache_type_k != expected) {
        fprintf(stderr, "  FAIL -ctk '%s': got %s, expected %s\n",
                value.c_str(), ggml_type_name(params.cache_type_k),
                ggml_type_name(expected));
        return false;
    }
    printf("  OK -ctk '%s' -> %s\n", value.c_str(), ggml_type_name(params.cache_type_k));
    return true;
}

// Parse "-ctv <type>" and assert the resulting params.cache_type_v == expected.
static bool check_v_type(std::string value, ggml_type expected) {
    common_params params;
    std::vector<std::string> argv = {"binary_name", "-m", "model.gguf", "-ctv", value};
    bool ok = common_params_parse(argv.size(), argv_to_char(argv).data(), params,
                                  LLAMA_EXAMPLE_COMMON);
    if (!ok) {
        fprintf(stderr, "  FAIL -ctv '%s': parse rejected\n", value.c_str());
        return false;
    }
    if (params.cache_type_v != expected) {
        fprintf(stderr, "  FAIL -ctv '%s': got %s, expected %s\n",
                value.c_str(), ggml_type_name(params.cache_type_v),
                ggml_type_name(expected));
        return false;
    }
    printf("  OK -ctv '%s' -> %s\n", value.c_str(), ggml_type_name(params.cache_type_v));
    return true;
}

int main(void) {
    try {
        bool all_ok = true;
        int passed = 0, failed = 0;

        printf("test-kv-cache-pq2-0: registration of -ctk/-ctv pq2_0\n");

        // The whole point: pq2_0 must be a valid KV-cache type.
        if (check_k_type("pq2_0", GGML_TYPE_PQ2_0)) { ++passed; } else { ++failed; all_ok = false; }
        if (check_v_type("pq2_0", GGML_TYPE_PQ2_0)) { ++passed; } else { ++failed; all_ok = false; }

        printf("test-kv-cache-pq2-0: controls (a known-good type still works; garbage still rejected)\n");

        // Control: a known-good type still parses.
        {
            common_params params;
            std::vector<std::string> argv = {"binary_name", "-m", "model.gguf",
                                            "-ctk", "q4_0", "-ctv", "q5_0"};
            bool ok = common_params_parse(argv.size(), argv_to_char(argv).data(), params,
                                          LLAMA_EXAMPLE_COMMON);
            bool good = ok && params.cache_type_k == GGML_TYPE_Q4_0
                     && params.cache_type_v == GGML_TYPE_Q5_0;
            if (good) { ++passed; printf("  OK -ctk q4_0 -ctv q5_0 -> %s / %s\n",
                       ggml_type_name(params.cache_type_k),
                       ggml_type_name(params.cache_type_v)); }
            else { ++failed; all_ok = false; fprintf(stderr, "  FAIL control q4_0/q5_0\n"); }
        }

        // Control: a garbage type is still rejected.
        {
            common_params params;
            std::vector<std::string> argv = {"binary_name", "-ctk", "not-a-real-type"};
            bool rejected = (false == common_params_parse(argv.size(), argv_to_char(argv).data(),
                                                          params, LLAMA_EXAMPLE_COMMON));
            if (rejected) { ++passed; printf("  OK -ctk not-a-real-type rejected\n"); }
            else { ++failed; all_ok = false; fprintf(stderr, "  FAIL control: garbage type accepted\n"); }
        }

        printf("test-kv-cache-pq2-0: %d passed, %d failed\n", passed, failed);
        if (all_ok) {
            printf("test-kv-cache-pq2-0: all tests OK\n\n");
            return 0;
        } else {
            // RED state: print the current allowed list for diagnostics.
            fprintf(stderr, "\n  RED: pq2_0 is not yet a valid KV-cache type.\n");
            return 1;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "test-kv-cache-pq2-0: exception: %s\n", e.what());
        return 1;
    }
}

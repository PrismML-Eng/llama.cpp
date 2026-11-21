#include "common.h"
#include "log.h"
#include "speculative.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    common_log_set_verbosity_thold(-1);
    const size_t first = argc == 2 ? std::strtoul(argv[1], nullptr, 10) : 0;
    const size_t last = argc == 2 ? first : 8;
    assert(last <= 8);
    for (size_t n_prompt = first; n_prompt <= last; ++n_prompt) {
        common_params_speculative params;
        params.types = { COMMON_SPECULATIVE_TYPE_NGRAM_CACHE };
        common_speculative_ptr spec(common_speculative_init(params, 1));
        assert(spec);
        llama_tokens prompt, result;
        for (size_t i = 0; i < n_prompt; ++i) {
            prompt.push_back(i + 1);
        }
        common_speculative_begin(spec.get(), 0, prompt);
        auto & dp = common_speculative_get_draft_params(spec.get(), 0);
        dp.drafting = true;
        dp.prompt = &prompt;
        dp.result = &result;
        dp.id_last = n_prompt + 1;
        common_speculative_draft(spec.get());
        // Each token appears once and the anchor has no known continuation.
        assert(result.empty());
    }
    std::printf("ngram-cache short histories: %zu passed\n", last - first + 1);
    return 0;
}

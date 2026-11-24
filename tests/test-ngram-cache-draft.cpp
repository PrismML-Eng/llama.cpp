#include "common.h"
#include "log.h"
#include "ngram-cache.h"
#include "speculative.h"

#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>

static thread_local bool count_allocations = false;
static thread_local size_t allocation_count = 0;
static thread_local size_t allocation_bytes = 0;
static thread_local size_t input_bytes = 0;
static thread_local size_t input_allocations = 0;
static thread_local void * input_storage = nullptr;
static thread_local size_t input_live_bytes = 0;

void * operator new(size_t n) {
    void * p = std::malloc(n ? n : 1);
    if (!p) {
        throw std::bad_alloc();
    }
    if (input_bytes && n == input_bytes) {
        input_storage = p;
        input_live_bytes = n;
    }
    if (count_allocations) {
        ++allocation_count;
        allocation_bytes += n;
        input_allocations += n == input_bytes;
    }
    return p;
}
void * operator new[](size_t n) { return ::operator new(n); }
static void release(void * p) {
    if (p == input_storage) {
        input_live_bytes = 0;
        input_storage = nullptr;
    }
    std::free(p);
}
void operator delete(void * p) noexcept { release(p); }
void operator delete[](void * p) noexcept { release(p); }
void operator delete(void * p, size_t) noexcept { release(p); }
void operator delete[](void * p, size_t) noexcept { release(p); }

struct reference_state {
    common_ngram_cache context, dynamic, fixed;
    size_t cached = 0;

    llama_tokens draft(const llama_tokens & prompt, llama_token last) {
        llama_tokens input(prompt);
        input.push_back(last);
        if (cached < input.size()) {
            common_ngram_cache_update(context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                                      input, input.size() - cached, false);
            cached = input.size();
        }
        llama_tokens out = { last };
        common_ngram_cache_draft(input, out, 8, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, context, dynamic, fixed);
        out.erase(out.begin());
        return out;
    }
};

static common_speculative_ptr init(uint32_t n_seq) {
    common_params_speculative params;
    params.types = { COMMON_SPECULATIVE_TYPE_NGRAM_CACHE };
    common_speculative_ptr spec(common_speculative_init(params, n_seq));
    assert(spec);
    assert(common_speculative_get_types(spec.get()) == params.types);
    return spec;
}

static void test_outputs() {
    auto spec = init(3);
    reference_state refs[3];
    llama_tokens prompt[3], result[3];
    for (auto & out : result) {
        out.reserve(16);
    }
    size_t calls = 0;
    for (int round = 0; round < 40; ++round) {
        for (int seq = 0; seq < 3; ++seq) {
            const bool reset = round % 7 == 0;
            const size_t n = reset ? size_t(round % 5) : size_t(32 + round + seq);
            prompt[seq].resize(n);
            for (size_t i = 0; i < n; ++i) {
                prompt[seq][i] = 1 + (i + seq) % 7;
            }
            // Change contents at the same storage address and begin a new request.
            if (round % 9 == 0) {
                const auto * storage = prompt[seq].data();
                for (auto & token : prompt[seq]) {
                    token += 20;
                }
                assert(prompt[seq].data() == storage);
            }
            if (reset || round % 9 == 0 || round % 9 == 1) {
                common_speculative_begin(spec.get(), seq, prompt[seq]);
                refs[seq].cached = 0;
            }
            result[seq].clear();
            auto & dp = common_speculative_get_draft_params(spec.get(), seq);
            dp.drafting = (round + seq) % 4 != 0;
            dp.prompt = &prompt[seq];
            dp.result = &result[seq];
            dp.id_last = 1 + (n + seq) % 7;
        }
        llama_tokens expected[3];
        for (int seq = 0; seq < 3; ++seq) {
            auto & dp = common_speculative_get_draft_params(spec.get(), seq);
            if (dp.drafting) {
                expected[seq] = refs[seq].draft(*dp.prompt, dp.id_last);
                ++calls;
            }
        }
        common_speculative_draft(spec.get());
        for (int seq = 0; seq < 3; ++seq) {
            assert(result[seq] == expected[seq]);
        }
    }
    std::printf("ngram-cache multi-sequence output cases: %zu\n", calls);
}

static void probe(bool benchmark) {
    auto spec = init(1);
    llama_tokens prompt(32768), result;
    for (size_t i = 0; i < prompt.size(); ++i) {
        prompt[i] = 1 + i % 7;
    }
    result.reserve(16);
    auto & dp = common_speculative_get_draft_params(spec.get(), 0);
    dp.prompt = &prompt;
    dp.result = &result;
    dp.id_last = 1 + prompt.size() % 7;
    auto run = [&] {
        result.clear();
        dp.drafting = true;
        common_speculative_draft(spec.get());
    };
    input_bytes = (prompt.size() + 1) * sizeof(llama_token);
    common_speculative_begin(spec.get(), 0, prompt);
    run();
    assert(result.size() == 8);
    for (size_t i = 0; i < result.size(); ++i) {
        assert(result[i] == (llama_token) (1 + (prompt.size() + i + 1) % 7));
    }
    const int repeats = benchmark ? 1000 : 10;
    const int samples = benchmark ? 6 : 1;
    for (int sample = 0; sample < samples; ++sample) {
        allocation_count = allocation_bytes = input_allocations = 0;
        count_allocations = true;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) {
            run();
        }
        const auto end = std::chrono::steady_clock::now();
        count_allocations = false;
        std::printf("{\"sample\":%d,\"calls\":%d,\"allocations\":%zu,\"allocated_bytes\":%zu,\"input_allocations\":%zu,\"retained_input_bytes\":%zu,\"us_per_call\":%.6f}\n",
                    sample, repeats, allocation_count, allocation_bytes, input_allocations, input_live_bytes,
                    std::chrono::duration<double, std::micro>(end - start).count() / repeats);
        if (!benchmark) {
            assert(input_allocations == 0);
            assert(input_live_bytes == input_bytes);
        }
    }
}

int main(int argc, char ** argv) {
    common_log_set_verbosity_thold(-1);
    const bool benchmark = argc == 2 && std::string(argv[1]) == "--bench";
    if (!benchmark) {
        test_outputs();
    }
    probe(benchmark);
    return 0;
}

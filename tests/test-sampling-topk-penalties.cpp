// The top-k shortlist that is used with penalties (common_topk_shortlist_exact) must give the same top k as the
// full vocabulary, in entries and order, whenever it says it does.

#include "sampling.h"
#include "llama.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <random>
#include <vector>

struct run_result {
    std::vector<llama_token> ids;
    bool exact; // the shortlist path is allowed
};

// the chain with the full array: penalties, then top-k
static std::vector<llama_token> top_k_full(llama_sampler * penalties, const std::vector<float> & logits, int32_t k) {
    std::vector<llama_token_data> cur(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        cur[i] = { (llama_token) i, logits[i], 0.0f };
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(penalties, &cur_p);
    llama_sampler * top_k = llama_sampler_init_top_k(k);
    llama_sampler_apply(top_k, &cur_p);
    llama_sampler_free(top_k);
    std::vector<llama_token> ids;
    for (size_t i = 0; i < cur_p.size; i++) {
        ids.push_back(cur_p.data[i].id);
    }
    return ids;
}

// the shortlist of the n largest raw logits (ties in any order), then penalties, then top-k
static run_result top_k_shortlist(llama_sampler * penalties, const std::vector<float> & logits, int32_t k, int32_t n) {
    std::vector<llama_token_data> cur(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        cur[i] = { (llama_token) i, logits[i], 0.0f };
    }
    // the order of equal logits is arbitrary on purpose
    std::reverse(cur.begin(), cur.end());
    std::partial_sort(cur.begin(), cur.begin() + n, cur.end(), [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    });
    cur.resize(n);

    run_result res;
    res.exact = common_topk_shortlist_exact(penalties, cur.data(), cur.size(), k);

    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, true };
    llama_sampler_apply(penalties, &cur_p);
    llama_sampler * top_k = llama_sampler_init_top_k(k);
    llama_sampler_apply(top_k, &cur_p);
    llama_sampler_free(top_k);
    for (size_t i = 0; i < cur_p.size; i++) {
        res.ids.push_back(cur_p.data[i].id);
    }
    return res;
}

// the case of the review: with the shortlist the tie at 0 would pick token 1 instead of 0
static void test_review_case() {
    const std::vector<float> logits = { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    llama_sampler * pen = llama_sampler_init_penalties(6, 2, 1.0f, 0.0f, 1.0f);
    llama_sampler_accept(pen, 1);

    const int32_t k = 1;
    const auto full = top_k_full(pen, logits, k);
    assert(full.size() == 1 && full[0] == 0);

    // shortlist of 5 of the 6 tokens (k + penalty_last_n + 1 = 4 would be the real size; 5 shows the border)
    for (int32_t n = k + 1; n < 6; n++) {
        const auto res = top_k_shortlist(pen, logits, k, n);
        // either the shortlist is refused, or it gives the same token
        assert(!res.exact || res.ids == full);
    }
    // the tie of the penalized token with the others must be seen
    const auto res = top_k_shortlist(pen, logits, k, 4);
    assert(!res.exact);

    llama_sampler_free(pen);
}

// no tie anywhere: the shortlist is used and equals the full array
static void test_no_tie_is_fast() {
    const std::vector<float> logits = { 0.1f, 5.0f, 0.2f, 4.0f, 0.3f, 3.0f, 0.4f, 2.0f, 0.5f, 1.0f, 0.6f, 0.7f };
    llama_sampler * pen = llama_sampler_init_penalties(12, 2, 1.0f, 0.0f, 0.5f);
    llama_sampler_accept(pen, 1);
    llama_sampler_accept(pen, 3);

    const int32_t k = 3;
    const auto full = top_k_full(pen, logits, k);
    const auto res  = top_k_shortlist(pen, logits, k, k + 2 + 1);
    assert(res.exact);
    assert(res.ids == full);

    llama_sampler_free(pen);
}

// random logits with many equal values and random histories: whenever the shortlist is accepted it equals the full array
static void test_random() {
    std::mt19937 rng(1234);
    int n_exact = 0;
    int n_refused = 0;

    for (int iter = 0; iter < 20000; iter++) {
        const int32_t n_vocab = 20 + (int32_t) (rng() % 60);
        const int32_t k       = 1 + (int32_t) (rng() % 8);
        const int32_t last_n  = 1 + (int32_t) (rng() % 6);
        // half of the cases: few levels, many ties; half: close to real logits, ties are rare (and set on purpose below)
        const bool coarse = iter % 2 == 0;
        const int32_t levels = coarse ? 3 + (int32_t) (rng() % 12) : 1000000;

        std::vector<float> logits(n_vocab);
        for (auto & l : logits) {
            l = (float) (rng() % levels) / (coarse ? 1.0f : 50000.0f) - (coarse ? (float) levels / 2 : 10.0f);
        }
        if (!coarse && rng() % 4 == 0) {
            // a single tie between two tokens
            logits[rng() % n_vocab] = logits[rng() % n_vocab];
        }

        const float repeat = (rng() % 2) ? 1.0f : 1.0f + (float) (rng() % 4) / 2;
        const float freq   = (rng() % 2) ? 0.0f : (float) (rng() % 3);
        const float pres   = (rng() % 2) ? 0.0f : (float) (rng() % 4) / 2 + (repeat == 1.0f && freq == 0.0f ? 0.5f : 0.0f);

        llama_sampler * pen = llama_sampler_init_penalties(n_vocab, last_n, repeat, freq, pres);
        for (int32_t i = 0; i < last_n; i++) {
            llama_sampler_accept(pen, (llama_token) (rng() % n_vocab));
        }

        const int32_t n = std::min(n_vocab, k + last_n + 1);
        if (n <= k) {
            llama_sampler_free(pen);
            continue;
        }
        const auto full = top_k_full(pen, logits, k);
        const auto res  = top_k_shortlist(pen, logits, k, n);
        if (res.exact) {
            n_exact++;
            assert(res.ids == full);
        } else {
            n_refused++;
        }
        llama_sampler_free(pen);
    }
    printf("random: %d accepted, %d refused\n", n_exact, n_refused);
    assert(n_exact > 5000 && n_refused > 0);
}

int main() {
    test_review_case();
    test_no_tie_is_fast();
    test_random();
    printf("OK\n");
    return 0;
}

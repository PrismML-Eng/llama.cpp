#pragma once

#include "llama.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#define LLAMA_NGRAM_MIN    1
#define LLAMA_NGRAM_MAX    4
#define LLAMA_NGRAM_STATIC 2

// Data structures to map n-grams to empirical token probabilities:

struct common_ngram {
    llama_token tokens[LLAMA_NGRAM_MAX];

    common_ngram() {
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            tokens[i] = LLAMA_TOKEN_NULL;
        }
    }

    common_ngram(const llama_token * input, const int ngram_size) {
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            tokens[i] = i < ngram_size ? input[i] : LLAMA_TOKEN_NULL;
        }
    }

    bool operator==(const common_ngram & other) const {
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            if (tokens[i] != other.tokens[i]) {
                return false;
            }
        }
        return true;
    }
};

struct common_token_hash_function {
    size_t operator()(const llama_token token) const {
        // see https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/
        return token * 11400714819323198485llu;
    }
};

struct common_ngram_hash_function {
    size_t operator()(const common_ngram & ngram) const {
        size_t hash = common_token_hash_function{}(ngram.tokens[0]);
        for (int i = 1; i < LLAMA_NGRAM_MAX; ++i) {
            hash ^= common_token_hash_function{}(ngram.tokens[i]);
        }
        return hash;
    }
};

// token -> number of times token has been seen
typedef std::unordered_map<llama_token, int32_t> common_ngram_cache_part;

// n-gram -> empirical distribution of following tokens
typedef std::unordered_map<common_ngram, common_ngram_cache_part, common_ngram_hash_function> common_ngram_cache;


// Update an ngram cache with tokens.
// ngram_cache:         the cache to modify.
// ngram_min/ngram_max: the min/max size of the ngrams to extract from inp_data.
// inp_data:            the token sequence with which to update ngram_cache.
// nnew:                how many new tokens have been appended to inp_data since the last call to this function.
// print_progress:      whether to print progress to stderr.
//
// In order to get correct results inp_data can ONLY BE APPENDED TO.
// Changes in the middle need a complete rebuild.
void common_ngram_cache_update(
    common_ngram_cache & ngram_cache, int ngram_min, int ngram_max, std::vector<llama_token> & inp_data, int nnew, bool print_progress);

// Try to draft tokens from ngram caches.
// inp:                the tokens generated so far.
// draft:              the token sequence to draft. Expected to initially contain the previously sampled token.
// n_draft:            maximum number of tokens to add to draft.
// ngram_min/gram_max: the min/max size of the ngrams in nc_context and nc_dynamic.
// nc_context:         ngram cache based on current context.
// nc_dynamic:         ngram cache based on previous user generations.
// nc_static:          ngram cache generated from a large text corpus, used for validation.
void common_ngram_cache_draft(
    std::vector<llama_token> & inp, std::vector<llama_token> & draft, int n_draft, int ngram_min, int ngram_max,
    common_ngram_cache & nc_context, common_ngram_cache & nc_dynamic, common_ngram_cache & nc_static);

// Save an ngram cache to a file.
// ngram_cache: the ngram cache to save.
// filename:    the path under which to save the ngram cache.
void common_ngram_cache_save(common_ngram_cache & ngram_cache, const std::string & filename);

// Load an ngram cache saved with common_ngram_cache_save.
// filename: the path from which to load the ngram cache.
// returns:  an ngram cache containing the information saved to filename.
common_ngram_cache common_ngram_cache_load(const std::string & filename);

// Merge two ngram caches.
// ngram_cache_target: the ngram cache to which to add the information from ngram_cache_add.
// ngram_cache_add:    the ngram cache to add to ngram_cache_target.
void common_ngram_cache_merge(common_ngram_cache & ngram_cache_target, common_ngram_cache & ngram_cache_add);

// Prompt-copy proposals only; the target must verify every returned token.
inline std::vector<llama_token> common_prompt_lookup_draft(const std::vector<llama_token> & prompt,
                                                           const std::vector<llama_token> & history,
                                                           llama_token                      anchor,
                                                           int                              depth) {
    if (depth <= 0 || (size_t) depth >= prompt.size() || history.size() < prompt.size() ||
        !std::equal(prompt.begin(), prompt.end(), history.begin())) {
        return {};
    }
    const size_t n      = (size_t) depth;
    size_t       chosen = prompt.size();
    if (history.size() == prompt.size()) {
        for (size_t c = 1; c < prompt.size() - n; ++c) {
            if (prompt[c - 1] == prompt.back() && prompt[c] == anchor) {
                if (chosen != prompt.size()) {
                    return {};
                }
                chosen = c + 1;
            }
        }
    } else {
        auto committed = history;
        committed.push_back(anchor);
        const size_t longest = std::min({ size_t(64), committed.size() - n, prompt.size() - n });
        if (longest < 16) {
            return {};
        }
        size_t best      = 15;
        bool   ambiguous = false;
        for (size_t end = 16; end <= prompt.size() - n; ++end) {
            if (prompt[end - 1] != anchor) {
                continue;
            }
            size_t length = 1;
            while (length < std::min(longest, end) &&
                   prompt[end - length - 1] == committed[committed.size() - length - 1]) {
                ++length;
            }
            if (length < 16 || length < best) {
                continue;
            }
            if (length > best) {
                best      = length;
                chosen    = end;
                ambiguous = false;
            } else if (!std::equal(prompt.begin() + chosen, prompt.begin() + chosen + n, prompt.begin() + end)) {
                ambiguous = true;
            }
        }
        if (ambiguous) {
            return {};
        }
    }
    if (chosen == prompt.size()) {
        return {};
    }
    return { prompt.begin() + chosen, prompt.begin() + chosen + n };
}

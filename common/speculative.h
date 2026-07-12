#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// true if any implementation requires target post-norm embeddings to be extracted
bool common_speculative_need_embd(common_speculative * spec);

// true if any implementation requires target nextn embeddings to be extracted
bool common_speculative_need_embd_nextn(common_speculative * spec);

// true if any implementation requires the target's multi-layer tap capture
// (see llama_set_capture_layers / llama_get_embeddings_capture_ith) -- used by
// dspark, which conditions on several intermediate target layers concatenated
// per position rather than a single pre/post-norm embedding.
bool common_speculative_need_embd_capture(common_speculative * spec);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

// TEST/DEBUG ONLY: directly stage target-tap context rows for the dspark
// implementation (if registered), bypassing the normal process()-driven
// capture path, which requires a real target context with
// llama_set_capture_layers engaged and logits requested on every row. Used by
// the Phase 2 synthetic-target harness (tests/test-dspark-loop.cpp) to drive
// the block-draft loop deterministically without a target model.
// `feat` is [n_rows * n_embd_cap] row-major, `pos` is [n_rows] absolute
// positions, both appended to the sequence's pending context buffer exactly
// as process() would have. Returns false if no dspark implementation is
// registered.
bool common_speculative_dspark_stage_ctx_test(
        common_speculative * spec,
        llama_seq_id seq_id,
        const float * feat,
        int64_t n_rows,
        int64_t n_embd_cap,
        const int32_t * pos);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

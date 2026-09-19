// Correctness harness for the recurrent-state snapshot planes (n_rs_seq > 0).
//
// A forward-only generation never reads the snapshot planes back, so it cannot
// catch a bad snapshot write. This does:
//
//   pass A (reference): greedily generate N tokens straight through.
//   pass B (rollback):  generate the same N tokens, but every S accepted tokens
//                       decode the next accepted token together with R draft
//                       tokens as ONE ubatch (the server's verify shape), then
//                       llama_memory_seq_rm() the R draft positions away.
//
// Pass B only lands on the same tokens if the state restored from the snapshot
// plane is exactly the state after the accepted token. Any error in the
// conv-window or SSM snapshot write shows up as a token mismatch.
//
// The rollback stays inside the verify ubatch on purpose: a ubatch snapshots the
// state after each of its last min(T, K) - 1 tokens and never the state it
// started from, so a rollback across single-token steps is refused. Right after
// a rollback step the accepted token's logits come from an (R + 1)-token ubatch
// instead of a single-token one; a mismatch exactly there can in rare near-tie
// cases be matmul rounding rather than a snapshot bug -- the first-mismatch
// index tells.
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static llama_token greedy(llama_context * ctx, const llama_vocab * vocab) {
    const int     n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits  = llama_get_logits_ith(ctx, -1);

    llama_token best   = 0;
    float       best_v = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best   = i;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt    = "Explain how a jet engine works, step by step.";
    int         n_predict = 128;
    int         stride    = 16;  // roll back every `stride` accepted tokens
    int         rewind    = 4;   // how many tokens to discard and re-generate
    int         n_rs_seq  = 8;   // snapshot planes per seq; must be >= rewind

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            stride = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-r") && i + 1 < argc) {
            rewind = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-k") && i + 1 < argc) {
            n_rs_seq = atoi(argv[++i]);
        } else if (argv[i][0] != '-') {
            prompt = argv[i];
        }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf [-n N] [-s stride] [-r rewind] [-k n_rs_seq]\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 99;
    llama_model * model        = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int                n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> ptoks(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), ptoks.data(), ptoks.size(), true, true);

    int n_rollbacks = 0;

    auto run = [&](bool with_rollback, std::vector<llama_token> & out) -> bool {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx                = n_prompt + n_predict + 64;
        cparams.n_batch              = n_prompt + 64;
        cparams.n_rs_seq             = n_rs_seq;
        llama_context * ctx          = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "failed to create context\n");
            return false;
        }
        llama_memory_t mem = llama_get_memory(ctx);

        llama_batch batch = llama_batch_get_one(ptoks.data(), ptoks.size());
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "decode(prompt) failed\n");
            return false;
        }
        int n_pos = n_prompt;

        const int n_vocab = llama_vocab_n_tokens(vocab);

        int since = 0;
        while ((int) out.size() < n_predict) {
            llama_token tok = greedy(ctx, vocab);
            out.push_back(tok);
            since++;

            if (with_rollback && since >= stride && (int) out.size() + rewind < n_predict) {
                n_rollbacks++;
                // Verify-style step: decode the accepted token together with `rewind`
                // draft tokens as ONE ubatch (logits for the accepted token's row only),
                // then drop the drafts. A ubatch snapshots the state after each of its
                // last min(T, K) - 1 tokens and never the state it started from, so with
                // T = rewind + 1 a rollback of `rewind` positions is the largest this
                // ubatch supports and lands on the state after the accepted token --
                // needs n_rs_seq >= rewind. The drafts' content is irrelevant.
                llama_batch vb = llama_batch_init(rewind + 1, 0, 1);
                for (int r = 0; r <= rewind; r++) {
                    vb.token[r]     = r == 0 ? tok : (llama_token) ((tok + 7*r + 1) % n_vocab);
                    vb.pos[r]       = n_pos + r;
                    vb.n_seq_id[r]  = 1;
                    vb.seq_id[r][0] = 0;
                    vb.logits[r]    = r == 0;
                }
                vb.n_tokens = rewind + 1;
                const bool ok_verify = llama_decode(ctx, vb) == 0;
                llama_batch_free(vb);
                if (!ok_verify) {
                    fprintf(stderr, "decode(verify) failed\n");
                    return false;
                }

                const llama_pos keep = n_pos + 1;
                if (!llama_memory_seq_rm(mem, 0, keep, -1)) {
                    fprintf(stderr, "ROLLBACK REFUSED: keep=%d n_pos=%d rewind=%d\n", keep, n_pos + rewind + 1, rewind);
                    return false;
                }
                n_pos = keep;
                since = 0;
                // the next greedy() reads the accepted token's logits, the batch's only output row
            } else {
                batch = llama_batch_get_one(&tok, 1);
                if (llama_decode(ctx, batch)) {
                    fprintf(stderr, "decode failed\n");
                    return false;
                }
                n_pos++;
            }
        }
        llama_free(ctx);
        return true;
    };

    bool ok = true;

    std::vector<llama_token> ref, rb;
    if (!run(false, ref)) {
        return 1;
    }
    if (!run(true, rb)) {
        return 1;
    }

    size_t n        = ref.size() < rb.size() ? ref.size() : rb.size();
    size_t mismatch = 0;
    int    first    = -1;
    for (size_t i = 0; i < n; i++) {
        if (ref[i] != rb[i]) {
            if (first < 0) {
                first = (int) i;
            }
            mismatch++;
        }
    }

    printf("\n=== rs-rollback ===\n");
    printf("tokens compared : %zu\n", n);
    printf("mismatches      : %zu\n", mismatch);
    printf("rollbacks done  : %d\n", n_rollbacks);
    // a run that never entered the rollback branch compared two identical
    // straight-through generations and proved nothing, so it is not a pass
    if (n_rollbacks == 0) {
        printf("RESULT: FAIL (no rollback was attempted: need n_predict > stride + rewind, "
               "got n_predict=%d stride=%d rewind=%d)\n", n_predict, stride, rewind);
        ok = false;
    } else if (mismatch) {
        printf("first mismatch  : index %d (ref=%d rb=%d)\n", first, ref[first], rb[first]);
        printf("RESULT: FAIL\n");
        ok = false;
    } else {
        printf("RESULT: PASS (rollback reproduces the straight-through sequence)\n");
    }

    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}

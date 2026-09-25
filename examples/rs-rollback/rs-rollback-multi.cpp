// Multi-sequence correctness harness for the recurrent-state snapshot ring.
//
// The ring's snapshot rows are indexed seq-major (see s_write_rows_conv in
// build_conv_state), so a layout error would show up as one sequence's rollback
// corrupting a *different* sequence's state. Single-sequence tests cannot see
// that class of bug at all.
//
//   pass A (reference): run S sequences concurrently, no rollback.
//   pass B (rollback):  identical, but sequence 0 periodically decodes its next
//                       token together with R draft tokens as ONE ubatch (the
//                       server's verify shape) and then seq_rm()s the R draft
//                       positions away.
//
// PASS requires every sequence -- including the untouched ones -- to emit
// exactly the reference token stream.
//
// The rollback stays inside the verify ubatch on purpose: a ubatch snapshots the
// state after each of its last min(T, K) - 1 tokens and never the state it
// started from, so a rollback across single-token steps is refused. At a
// rollback step sequence 0's logits come from an (R + 1)-token ubatch and the
// other sequences decode without it, so a mismatch exactly there can in rare
// near-tie cases be matmul rounding rather than a snapshot bug.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static llama_token greedy_ith(llama_context * ctx, const llama_vocab * vocab, int i) {
    const int     n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits  = llama_get_logits_ith(ctx, i);
    llama_token   best    = 0;
    float         best_v  = logits[0];
    for (int t = 1; t < n_vocab; t++) {
        if (logits[t] > best_v) {
            best_v = logits[t];
            best   = t;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    std::string model_path;
    int         n_predict = 64;
    int         n_seqs    = 3;
    int         stride    = 10;
    int         rewind    = 4;
    int         n_rs_seq  = 8;  // snapshot ring depth; must be >= rewind

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-S") && i + 1 < argc) {
            n_seqs = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            stride = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-r") && i + 1 < argc) {
            rewind = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-k") && i + 1 < argc) {
            n_rs_seq = atoi(argv[++i]);
        }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf [-n N] [-S seqs] [-s stride] [-r rewind] [-k n_rs_seq]\n", argv[0]);
        return 1;
    }
    // n_seqs <= 0 would leave the prompt/output vectors empty and run() indexes
    // out[0]; rewind <= 0 would make the rollback branch a no-op
    if (n_seqs <= 0 || rewind <= 0 || stride <= 0 || n_predict <= 0) {
        fprintf(stderr, "invalid arguments: need n_seqs > 0, rewind > 0, stride > 0, n_predict > 0 "
                        "(got %d, %d, %d, %d)\n", n_seqs, rewind, stride, n_predict);
        return 1;
    }

    // deliberately different prompts so the sequences cannot coincidentally agree
    std::vector<std::string> prompts = {
        "Explain how a jet engine works, step by step.",
        "Write a short poem about the sea in winter.",
        "List three causes of the French Revolution.",
        "Describe the process of photosynthesis briefly.",
    };
    if (n_seqs > (int) prompts.size()) {
        n_seqs = (int) prompts.size();
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

    std::vector<std::vector<llama_token>> ptoks(n_seqs);
    for (int s = 0; s < n_seqs; s++) {
        const std::string & p = prompts[s];
        int                 n = -llama_tokenize(vocab, p.c_str(), p.size(), NULL, 0, true, true);
        ptoks[s].resize(n);
        llama_tokenize(vocab, p.c_str(), p.size(), ptoks[s].data(), n, true, true);
    }

    int n_rollbacks = 0;

    auto run = [&](bool with_rollback, std::vector<std::vector<llama_token>> & out) -> bool {
        out.assign(n_seqs, {});

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx                = (n_predict + 128) * n_seqs;
        cparams.n_batch              = 512;
        cparams.n_seq_max            = n_seqs;
        cparams.n_rs_seq             = n_rs_seq;
        llama_context * ctx          = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "failed to create context\n");
            return false;
        }
        llama_memory_t mem = llama_get_memory(ctx);

        llama_batch      batch = llama_batch_init(512, 0, n_seqs);
        std::vector<int> npos(n_seqs, 0);

        auto submit = [&](const std::vector<std::pair<int, llama_token>> & items) -> bool {
            batch.n_tokens = 0;
            for (auto & it : items) {
                const int s        = it.first;
                const int i        = batch.n_tokens;
                batch.token[i]     = it.second;
                batch.pos[i]       = npos[s];
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = s;
                batch.logits[i]    = 1;
                batch.n_tokens++;
                npos[s]++;
            }
            return llama_decode(ctx, batch) == 0;
        };

        // prompts: one sequence at a time (each is a multi-token ubatch)
        for (int s = 0; s < n_seqs; s++) {
            batch.n_tokens = 0;
            for (size_t k = 0; k < ptoks[s].size(); k++) {
                batch.token[batch.n_tokens]     = ptoks[s][k];
                batch.pos[batch.n_tokens]       = npos[s];
                batch.n_seq_id[batch.n_tokens]  = 1;
                batch.seq_id[batch.n_tokens][0] = s;
                batch.logits[batch.n_tokens]    = (k + 1 == ptoks[s].size());
                batch.n_tokens++;
                npos[s]++;
            }
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "decode(prompt %d) failed\n", s);
                return false;
            }
            out[s].push_back(greedy_ith(ctx, vocab, batch.n_tokens - 1));
        }

        const int n_vocab_run = llama_vocab_n_tokens(vocab);

        // invariant at the top of every iteration: out[s].back() is sampled-but-not-yet-decoded
        int since = 0;
        while ((int) out[0].size() < n_predict) {
            if (with_rollback && since >= stride && (int) out[0].size() + rewind + 2 < n_predict) {
                n_rollbacks++;
                // seq 0 alone takes a verify-style step: its pending token plus `rewind`
                // draft tokens as ONE ubatch (logits for the pending token's row only),
                // then the drafts are dropped. A ubatch snapshots the state after each
                // of its last min(T, K) - 1 tokens and never the state it started from,
                // so with T = rewind + 1 the rollback of `rewind` positions is the
                // largest this ubatch supports and lands on the state after the pending
                // token -- needs n_rs_seq >= rewind. The drafts' content is irrelevant.
                const int npos0 = npos[0];
                batch.n_tokens  = 0;
                for (int r = 0; r <= rewind; r++) {
                    const llama_token t = r == 0 ? out[0].back() : (llama_token) ((out[0].back() + 7*r + 1) % n_vocab_run);
                    batch.token[r]     = t;
                    batch.pos[r]       = npos0 + r;
                    batch.n_seq_id[r]  = 1;
                    batch.seq_id[r][0] = 0;
                    batch.logits[r]    = r == 0;
                    batch.n_tokens++;
                }
                if (llama_decode(ctx, batch)) {
                    fprintf(stderr, "decode(verify) failed\n");
                    return false;
                }

                const llama_pos keep = npos0 + 1;
                if (!llama_memory_seq_rm(mem, 0, keep, -1)) {
                    fprintf(stderr, "ROLLBACK REFUSED: keep=%d npos0=%d rewind=%d\n", keep, npos0 + rewind + 1, rewind);
                    return false;
                }
                npos[0] = keep;
                out[0].push_back(greedy_ith(ctx, vocab, 0));

                // the other sequences take their ordinary single-token step
                std::vector<std::pair<int, llama_token>> others;
                for (int s = 1; s < n_seqs; s++) {
                    others.push_back({ s, out[s].back() });
                }
                if (!others.empty()) {
                    if (!submit(others)) {
                        fprintf(stderr, "decode(step) failed\n");
                        return false;
                    }
                    for (int s = 1; s < n_seqs; s++) {
                        out[s].push_back(greedy_ith(ctx, vocab, s - 1));
                    }
                }
                since = 0;
                continue;
            }

            // one token per sequence, all in one batch
            std::vector<std::pair<int, llama_token>> items;
            for (int s = 0; s < n_seqs; s++) {
                items.push_back({ s, out[s].back() });
            }
            if (!submit(items)) {
                fprintf(stderr, "decode(step) failed\n");
                return false;
            }
            for (int s = 0; s < n_seqs; s++) {
                out[s].push_back(greedy_ith(ctx, vocab, s));
            }
            since++;
        }

        llama_batch_free(batch);
        llama_free(ctx);
        return true;
    };

    std::vector<std::vector<llama_token>> ref, rb;
    if (!run(false, ref)) {
        return 1;
    }
    if (!run(true, rb)) {
        return 1;
    }

    printf("\n=== rs-rollback-multi (S=%d, rewind=%d) ===\n", n_seqs, rewind);
    int total_mismatch = 0;
    for (int s = 0; s < n_seqs; s++) {
        size_t n  = ref[s].size() < rb[s].size() ? ref[s].size() : rb[s].size();
        int    mm = 0, first = -1;
        for (size_t i = 0; i < n; i++) {
            if (ref[s][i] != rb[s][i]) {
                if (first < 0) {
                    first = (int) i;
                }
                mm++;
            }
        }
        total_mismatch += mm;
        printf("  seq %d %-14s compared=%zu mismatches=%d%s\n", s, s == 0 ? "(rolled back)" : "(untouched)", n, mm,
               mm ? "" : "  OK");
        if (mm) {
            printf("      first mismatch at %d (ref=%d rb=%d)\n", first, ref[s][first], rb[s][first]);
        }
    }
    printf("rollbacks done: %d\n", n_rollbacks);
    // no rollback attempted means the two runs were identical by construction
    const bool ok = (n_rollbacks > 0) && (total_mismatch == 0);
    if (n_rollbacks == 0) {
        printf("RESULT: FAIL (no rollback was attempted: need n_predict > stride + rewind + 2, "
               "got n_predict=%d stride=%d rewind=%d)\n", n_predict, stride, rewind);
    } else {
        printf("RESULT: %s\n", total_mismatch ? "FAIL" : "PASS");
    }

    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}

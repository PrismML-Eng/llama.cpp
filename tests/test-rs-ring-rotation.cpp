// Gate for the rotating recurrent-state snapshot ring (n_rs_seq > 0).
//
// Property under test: with the rotating ring, rolling back j <= n_rs_seq
// tokens -- across single-token decodes AND batch boundaries -- and continuing
// must produce byte-identical logits to a no-ring (n_rs_seq = 0) reference
// context that decoded the kept prefix with the same batch pattern and never
// rolled back. On CPU both contexts run the same fused GDN kernel with
// token-sequential arithmetic, so any mismatch indicates snapshot-ring
// corruption rather than numerical noise.
//
// This is stronger than the old copy-all-groups behavior, which only kept the
// snapshots of the last batch (a plain 1-token decode invalidated all older
// snapshots). The rotation preserves snapshot ages in place, so the rollback
// window survives consecutive decodes.
//
// Needs a recurrent-rollback-capable model (see llm_arch_supports_rs_rollback);
// generate one with: python3 tests/gen-tiny-qwen35.py /tmp/tiny-qwen35.gguf
// The test skips (exit 0) for models without rollback support.
//
// usage: test-rs-ring-rotation -m <model.gguf>

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static const uint32_t N_RS_SEQ = 8;

static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t n_rs_seq) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = n_rs_seq;
    cparams.n_ctx     = 256;
    cparams.n_batch   = 64;
    cparams.n_ubatch  = 64;
    return llama_init_from_model(model, cparams);
}

// decode tokens [i0, i0 + count) as one batch; request logits for the last token
static bool decode_batch(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t i0, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t i = 0; i < count; ++i) {
        common_batch_add(batch, tokens[i0 + i], (llama_pos) (i0 + i), { 0 }, i == count - 1);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// batch pattern: one prefill batch of n_prefill tokens, then 1-token decodes.
// decoding the same pattern in two contexts keeps the arithmetic identical
static bool decode_pattern(llama_context * ctx, const std::vector<llama_token> & tokens,
                           uint32_t n_prefill, uint32_t n_total) {
    if (!decode_batch(ctx, tokens, 0, n_prefill)) {
        return false;
    }
    for (uint32_t i = n_prefill; i < n_total; ++i) {
        if (!decode_batch(ctx, tokens, i, 1)) {
            return false;
        }
    }
    return true;
}

static std::vector<float> logits_last(llama_context * ctx, int n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        return {};
    }
    for (int i = 0; i < n_vocab; ++i) {
        if (!std::isfinite(logits[i])) {
            fprintf(stderr, "non-finite logit at %d\n", i);
            return {};
        }
    }
    return std::vector<float>(logits, logits + n_vocab);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    {
        // rollback support is arch-gated; probe it via a throwaway context
        llama_context * probe = make_ctx(params, model, N_RS_SEQ);
        if (probe == nullptr) {
            fprintf(stderr, "%s : failed to init probe context\n", __func__);
            return 1;
        }
        const uint32_t n_rs_seq = llama_n_rs_seq(probe);
        llama_free(probe);
        if (n_rs_seq == 0) {
            fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
            return 0;
        }
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    // fixed synthetic token stream (no tokenizer dependency)
    const uint32_t n_prefill = 6;
    const uint32_t n_singles = 5;
    const uint32_t n_total   = n_prefill + n_singles;

    std::vector<llama_token> tokens(n_total + 1);
    for (uint32_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = (llama_token) (20 + (i * 17) % 1000);
        if (tokens[i] >= n_vocab) {
            tokens[i] = tokens[i] % n_vocab;
        }
    }
    const llama_token probe_tok = tokens[n_total];

    // rollback depths: within the single-token decodes, exactly at the batch
    // boundary, and spanning back into the prefill batch's snapshots
    const uint32_t rollbacks[] = { 1, 3, n_singles, N_RS_SEQ };

    int n_checked = 0;

    for (const uint32_t j : rollbacks) {
        if (j > N_RS_SEQ || j >= n_total) {
            continue;
        }
        const uint32_t n_keep = n_total - j;

        // ring context: decode everything, then roll back j tokens
        llama_context * ctx_ring = make_ctx(params, model, N_RS_SEQ);
        if (ctx_ring == nullptr) {
            fprintf(stderr, "%s : failed to init ring context\n", __func__);
            return 1;
        }
        if (!decode_pattern(ctx_ring, tokens, n_prefill, n_total)) {
            fprintf(stderr, "%s : ring decode failed\n", __func__);
            return 1;
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_ring), 0, (llama_pos) n_keep, -1)) {
            fprintf(stderr, "%s : rollback of %u tokens failed\n", __func__, j);
            return 1;
        }
        // decode the probe token at the rolled-back position
        {
            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, probe_tok, (llama_pos) n_keep, { 0 }, true);
            const bool ok = llama_decode(ctx_ring, batch) == 0;
            llama_batch_free(batch);
            if (!ok) {
                fprintf(stderr, "%s : ring probe decode failed (j=%u)\n", __func__, j);
                return 1;
            }
        }
        std::vector<float> logits_ring = logits_last(ctx_ring, n_vocab);
        llama_free(ctx_ring);

        // no-ring reference: decode only the kept prefix with the same batch
        // pattern, then the probe token. never rolls back
        llama_context * ctx_ref = make_ctx(params, model, 0);
        if (ctx_ref == nullptr) {
            fprintf(stderr, "%s : failed to init reference context\n", __func__);
            return 1;
        }
        const uint32_t ref_prefill = n_prefill <= n_keep ? n_prefill : n_keep;
        if (!decode_pattern(ctx_ref, tokens, ref_prefill, n_keep)) {
            fprintf(stderr, "%s : reference decode failed (j=%u)\n", __func__, j);
            return 1;
        }
        {
            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, probe_tok, (llama_pos) n_keep, { 0 }, true);
            const bool ok = llama_decode(ctx_ref, batch) == 0;
            llama_batch_free(batch);
            if (!ok) {
                fprintf(stderr, "%s : reference probe decode failed (j=%u)\n", __func__, j);
                return 1;
            }
        }
        std::vector<float> logits_ref = logits_last(ctx_ref, n_vocab);
        llama_free(ctx_ref);

        if (logits_ring.empty() || logits_ref.empty()) {
            fprintf(stderr, "%s : missing/non-finite logits (j=%u)\n", __func__, j);
            return 1;
        }

        if (memcmp(logits_ring.data(), logits_ref.data(), logits_ref.size() * sizeof(float)) != 0) {
            int n_diff = 0;
            float max_diff = 0.0f;
            for (int i = 0; i < n_vocab; ++i) {
                const float d = std::fabs(logits_ring[i] - logits_ref[i]);
                if (d > 0.0f) {
                    n_diff++;
                    max_diff = max_diff > d ? max_diff : d;
                }
            }
            fprintf(stderr, "%s : FAIL rollback j=%u: %d/%d logits differ (max |d| = %g)\n",
                    __func__, j, n_diff, n_vocab, (double) max_diff);
            return 1;
        }

        fprintf(stderr, "%s : rollback j=%u matches no-ring reference byte-exact\n", __func__, j);
        n_checked++;
    }

    // cumulative rollback: two partial seq_rm calls without a decode in between
    // must land on the same snapshot as a single rollback of the sum
    {
        const uint32_t j1 = 2, j2 = 2;
        const uint32_t n_keep = n_total - (j1 + j2);

        llama_context * ctx_ring = make_ctx(params, model, N_RS_SEQ);
        if (ctx_ring == nullptr ||
            !decode_pattern(ctx_ring, tokens, n_prefill, n_total)) {
            fprintf(stderr, "%s : cumulative-rollback decode failed\n", __func__);
            return 1;
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_ring), 0, (llama_pos) (n_total - j1), -1) ||
            !llama_memory_seq_rm(llama_get_memory(ctx_ring), 0, (llama_pos) n_keep, -1)) {
            fprintf(stderr, "%s : cumulative rollback failed\n", __func__);
            return 1;
        }
        {
            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, probe_tok, (llama_pos) n_keep, { 0 }, true);
            const bool ok = llama_decode(ctx_ring, batch) == 0;
            llama_batch_free(batch);
            if (!ok) {
                fprintf(stderr, "%s : cumulative probe decode failed\n", __func__);
                return 1;
            }
        }
        std::vector<float> logits_ring = logits_last(ctx_ring, n_vocab);
        llama_free(ctx_ring);

        llama_context * ctx_ref = make_ctx(params, model, 0);
        if (ctx_ref == nullptr ||
            !decode_pattern(ctx_ref, tokens, n_prefill, n_keep)) {
            fprintf(stderr, "%s : cumulative reference decode failed\n", __func__);
            return 1;
        }
        {
            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, probe_tok, (llama_pos) n_keep, { 0 }, true);
            const bool ok = llama_decode(ctx_ref, batch) == 0;
            llama_batch_free(batch);
            if (!ok) {
                fprintf(stderr, "%s : cumulative reference probe decode failed\n", __func__);
                return 1;
            }
        }
        std::vector<float> logits_ref = logits_last(ctx_ref, n_vocab);
        llama_free(ctx_ref);

        if (logits_ring.empty() || logits_ref.empty() ||
            memcmp(logits_ring.data(), logits_ref.data(), logits_ref.size() * sizeof(float)) != 0) {
            fprintf(stderr, "%s : FAIL cumulative rollback (%u + %u) mismatch\n", __func__, j1, j2);
            return 1;
        }

        fprintf(stderr, "%s : cumulative rollback %u + %u matches no-ring reference byte-exact\n", __func__, j1, j2);
        n_checked++;
    }

    if (n_checked == 0) {
        fprintf(stderr, "%s : no rollback depth was checked\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s : all %d ring-rotation checks passed\n", __func__, n_checked);
    return 0;
}

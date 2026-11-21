// ARC-LAB mtp-export: training / validation data for the NextN (MTP) draft head.
// Runs the trunk over token sequences (teacher forcing) and writes, per position t:
//   h[t]   the trunk hidden state the MTP head consumes (h_nextn = final-norm output), f16 [n_embd]
//   tok[t] the input token at t
//   tgt[t] the trunk's own greedy choice for position t + 1 (argmax of the logits at t) - what a draft must match
// The MTP head at position t sees (h[t - 1], tok[t]) and should predict tgt[t].
//
//   llama-mtp-export -m model.gguf -i seqs.i32 -o prefix [-c n_ctx] [-b n_batch]
// seqs.i32: repeated records of int32 n followed by n int32 token ids. Writes prefix.h (f16), prefix.tok (i32),
// prefix.tgt (i32), prefix.idx (int64 start offset of each sequence, plus the total at the end).
#include "llama.h"
#include "ggml.h"
#include "../../src/llama-ext.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string model_path, in_path, out_prefix;
    int n_ctx = 4096, n_batch = 512;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (a == "-i" && i + 1 < argc) in_path = argv[++i];
        else if (a == "-o" && i + 1 < argc) out_prefix = argv[++i];
        else if (a == "-c" && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (a == "-b" && i + 1 < argc) n_batch = atoi(argv[++i]);
    }
    if (model_path.empty() || in_path.empty() || out_prefix.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -i seqs.i32 -o prefix [-c n_ctx] [-b n_batch]\n", argv[0]);
        return 1;
    }

    // read the sequences
    std::vector<std::vector<llama_token>> seqs;
    {
        FILE * f = fopen(in_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", in_path.c_str()); return 1; }
        int32_t n;
        while (fread(&n, 4, 1, f) == 1) {
            std::vector<llama_token> s(n);
            if (fread(s.data(), 4, n, f) != (size_t) n) { fprintf(stderr, "short read\n"); return 1; }
            if (n > n_ctx) s.resize(n_ctx);
            seqs.push_back(std::move(s));
        }
        fclose(f);
    }
    fprintf(stderr, "mtp-export: %zu sequences\n", seqs.size());

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const int n_embd  = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = n_ctx;
    cp.n_batch    = n_batch;
    cp.n_ubatch   = n_batch;
    cp.n_seq_max  = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
    llama_set_embeddings_nextn(ctx, true, /*masked*/ false);

    FILE * fh = fopen((out_prefix + ".h").c_str(), "wb");
    FILE * ft = fopen((out_prefix + ".tok").c_str(), "wb");
    FILE * fg = fopen((out_prefix + ".tgt").c_str(), "wb");
    FILE * fi = fopen((out_prefix + ".idx").c_str(), "wb");
    if (!fh || !ft || !fg || !fi) { fprintf(stderr, "cannot write outputs\n"); return 1; }

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<ggml_fp16_t> hrow(n_embd);
    int64_t total = 0;
    for (size_t si = 0; si < seqs.size(); ++si) {
        const auto & s = seqs[si];
        fwrite(&total, 8, 1, fi);
        llama_memory_clear(llama_get_memory(ctx), true);
        for (size_t p0 = 0; p0 < s.size(); p0 += n_batch) {
            const int n = (int) std::min<size_t>(n_batch, s.size() - p0);
            batch.n_tokens = n;
            for (int i = 0; i < n; ++i) {
                batch.token[i]     = s[p0 + i];
                batch.pos[i]       = (llama_pos) (p0 + i);
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at seq %zu pos %zu\n", si, p0); return 1; }
            for (int i = 0; i < n; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx, i);
                const float * l = llama_get_logits_ith(ctx, i);
                if (!h || !l) { fprintf(stderr, "missing outputs at %d\n", i); return 1; }
                int best = 0;
                for (int v = 1; v < n_vocab; ++v) if (l[v] > l[best]) best = v;
                for (int e = 0; e < n_embd; ++e) hrow[e] = ggml_fp32_to_fp16(h[e]);
                fwrite(hrow.data(), sizeof(ggml_fp16_t), n_embd, fh);
                const int32_t tok = s[p0 + i], tg = best;
                fwrite(&tok, 4, 1, ft);
                fwrite(&tg, 4, 1, fg);
            }
        }
        total += (int64_t) s.size();
        if ((si + 1) % 10 == 0 || si + 1 == seqs.size()) {
            fprintf(stderr, "mtp-export: %zu / %zu sequences, %lld positions\n", si + 1, seqs.size(), (long long) total);
        }
    }
    fwrite(&total, 8, 1, fi);
    fclose(fh); fclose(ft); fclose(fg); fclose(fi);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

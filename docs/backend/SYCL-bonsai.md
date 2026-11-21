# Ternary Bonsai 2 on Intel GPUs (SYCL)

How to run the PQ2_0 and PTQ1_0 Ternary Bonsai 2 models on Intel Arc and Core Ultra GPUs with the SYCL backend, which switches matter, and how to check the result.

The Intel-specific paths come from Jesse Symons' [Torchit1/llama.cpp `arc-b580`](https://github.com/Torchit1/llama.cpp/tree/arc-b580) work (#356, #358). The int2 x int8 DPAS kernels are [libxsmm/TernSYCL](https://github.com/libxsmm/TernSYCL) by Evangelos Georganas, Alexander Heinecke and the libxsmm team, vendored under `ggml/src/ggml-sycl/ternsycl/` with its BSD-3-Clause license.

## Get it

- Prebuilt: the `win-sycl-x64` and `ubuntu-sycl-*` archives from a [Prism release](https://github.com/PrismML-Eng/llama.cpp/releases). See [SYCL-RELEASE.md](SYCL-RELEASE.md) for how to install them (on Windows the SYCL archive goes on top of the CPU archive).
- From source: follow [SYCL.md](SYCL.md); the Bonsai paths need no extra CMake options.

Models: [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (`PTQ1_0` is the smallest; `PQ2_0` is the same weights in a 2-bit layout).

## Run

```sh
llama-server -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -c 16384
```

A q4_0 KV cache is what the XMX decode attention reads directly; with an f16 cache the regular attention path is used.

## Switches

All are environment variables. Defaults depend on the GPU: Xe2 is Arc B-series and Lunar Lake, Xe3 is Panther Lake and Wildcat Lake.

| Variable | Default | Effect |
|---|---|---|
| `GGML_SYCL_PTQ1_T2` | off | `all` (or `ffn`) repacks PQ2_0/PTQ1_0 weights to 2-bit in place for the TernSYCL kernels. Xe2 and newer. Faster prompt processing; on Xe3 single-token decode is slower than the default path. |
| `GGML_SYCL_DISABLE_XMX` | `1` on Xe3, `0` elsewhere | `0` keeps PQ2_0/PTQ1_0 on the XMX weight layout, `1` uses the MMVQ reorder layout. Ignored for weights taken by `GGML_SYCL_PTQ1_T2`. |
| `GGML_SYCL_FA_DEC_DPAS` | `1` on Xe3, `0` elsewhere | XMX decode attention from a q4_0 KV cache for up to 256 query tokens (`N` sets another cap). Xe2 and newer. |
| `GGML_SYCL_QUANT_W8A8`, `GGML_SYCL_QUANT_S4_GEMM` | `1` | oneDNN int8 GEMM for large prompt batches. |
| `GGML_SYCL_ENABLE_FUSION` | `1` | Fused norm, rotation and gate kernels. Individual fusions also have their own `*_OFF` switches. |
| `SYCL_UR_USE_LEVEL_ZERO_V2` | `0` (set by the library) | Set `1` to use the newer Level Zero adapter. |
| `LLAMA_GDN_REPLAY` | on when one sequence and all recurrent layers are on SYCL | Speculative decoding rolls the delta-net state back by replay instead of snapshots. Needs `-np 1` on llama-server. |
| `LLAMA_MTP_KV_ONLY`, `LLAMA_SYNC_TRIM` | on with a SYCL device | MTP prompt pass for the draft head writes K/V only; fewer context synchronizations. |
| `GGML_SYCL_OP_TIMING` | `0` | `1` prints per-op GPU times for MTP draft graphs (small graphs only; a plain run prints nothing). |

## Expected speed

Arc B390 (Xe3, Windows 11, driver 32.0.101.8860), Ternary Bonsai 2 27B PTQ1_0, tokens/s:

```sh
llama-bench -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -fa 1 -ctk q4_0 -ctv q4_0                # pp512, tg128
llama-bench -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -d 16384  # tg128 at 16K
```

| | pp512 | tg128 | tg128 at 16K |
|---|---|---|---|
| defaults | 353 | 15.8 | 14.8 |
| `GGML_SYCL_PTQ1_T2=all` | 490 | 14.5 | 13.8 |

Use `GGML_SYCL_PTQ1_T2=all` for long prompts and the defaults for chat-length prompts. Other Intel GPUs have not been measured with these defaults; please report numbers.

## Check correctness

Compare against the CPU backend with KL divergence on a short wikitext run. The CPU reference pass takes about 40 minutes on a laptop CPU and writes a 2 GB `base.kld`:

```sh
llama-perplexity -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -f wiki.test.raw --chunks 16 -c 512 -b 256 -ub 256 \
    --device none --kl-divergence-base base.kld
llama-perplexity -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -f wiki.test.raw --chunks 16 -c 512 -b 256 -ub 256 \
    -ngl 99 -fa on -ctk q4_0 -ctv q4_0 --kl-divergence-base base.kld --kl-divergence
```

Use `-b 256 -ub 256`: larger batches exceed the XMX decode attention's 256-token cap, so it would not be tested. On the B390 these commands give mean KLD 2.3e-3, PPL ratio 1.003 and the same top token 98.0% of the time with the defaults. Most of that is the q4_0 KV cache: against a reference that also uses a q4_0 KV cache, mean KLD is about 1.6e-3.

Op-level checks: `test-backend-ops -b SYCL0 -o MUL_MAT`, `-o GATED_DELTA_NET`, and `-o FLASH_ATTN_EXT -p hsk=256` (the model's head size; head size 192 currently aborts in `fattn-tile.cpp`, a separate issue).

When reporting a problem, include the release tag or commit, GPU, OS, driver, the command and any switches set.

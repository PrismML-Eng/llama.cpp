# Bonsai + TurboQuant integration (CPU and Vulkan)

This branch uses Prism/Bonsai llama.cpp as its base and ports only the TurboQuant subsystem. CUDA, Metal and SYCL TurboQuant kernels have not been ported. Prism's PQ2_0 and PTQ1_0 weight types remain available.

## Supported paths

- CPU: Turbo2/3/4 KV cache, WHT, TQ3_1S/TQ4_1S and Q8_CR/Q5_CR/Q6_CR weight codecs.
- Vulkan: Turbo2/3/4 cache writes and rotated-domain decoding, query/output WHT, flash attention, TQ weight matvec and f16-dequant fallback. ConvRot weight matmul uses CPU fallback (no native Vulkan CR kernel).
- Ordinary KV and sliding-window attention share query WHT and output unpadding. Latent/indexed/custom attention (MLA, DeepSeek32/4, GLM_DSA, DOTS3NOTE, MiniMax M3) rejects Turbo KV until those graph paths are implemented.
- Vulkan WHT supports contiguous F32 with 128-element groups and no scale tensor. Other WHT forms use CPU fallback.
- Turbo SET_ROWS kernels need a 32-wide subgroup. Devices without that capability use CPU fallback.

## Cache policy and limits

- Turbo codecs rotate cache rows internally. Mainline attention rotation is disabled on each actual Turbo side; non-Turbo cache types keep Prism's policy.
- Symmetric Turbo K/V requests on GQA ratios >= 6 upgrade K to Q8_0 by default. `TURBO_AUTO_ASYMMETRIC=0` disables this quality safeguard. Logs show the actual allocation types.
- Cache head dimensions are padded per head to multiples of 128 and trimmed after inverse WHT.
- Context shifting is disabled when the actual K cache is Turbo. A Turbo K cache cannot safely receive ordinary RoPE shifts in its WHT basis. Auto-asymmetric Q8_0 K keeps its existing shift capability.
- State files require matching cache types, layout and attention-rotation settings. Do not change `LLAMA_ATTN_ROT_DISABLE` when restoring an existing cache state.
- CUDA InnerQ calibration and layer-adaptive cache policies are not included.

## Build

Use a normal MSVC developer shell and the official Vulkan SDK. This machine was tested with SDK 1.4.363.0.

```bat
set VULKAN_SDK=C:\VulkanSDK\1.4.363.0
cmake -S . -B build-vk -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF
cmake --build build-vk --target llama-bench llama-simple llama-quantize test-backend-ops test-quantize-fns test-turbo-quant
```

For CPU-only builds, use a separate build directory with `-DGGML_VULKAN=OFF`.

## Runtime

`llama-bench` supports `-ctk turbo2|turbo3|turbo4` and `-ctv turbo2|turbo3|turbo4`. Shared CLI parsing also accepts those names. `llama-simple` has no cache-type options; it is a baseline example, not a Turbo KV test tool.

On this laptop the driver enumerates Intel UHD as physical device 0 and RTX 4060 as physical device 1. Confirm the list on your machine rather than assuming this order:

```bat
set GGML_VK_VISIBLE_DEVICES=1
build-vk\bin\llama-bench.exe -m MODEL.gguf -ngl 99 -ctk turbo3 -ctv turbo3 -fa on -p 16 -n 8 -r 1
```

## Validation

Bonsai 27B PTQ1_0 generated coherent text with CPU and RTX 4060 Vulkan. The full-offload Turbo3 run loaded 65/65 layers onto GPU with two graph splits. This is a smoke test, not a perplexity/quality benchmark or a long-context stability claim.

Filtered RTX 4060 backend comparisons against CPU: WHT 9/9, Turbo SET_ROWS/CPY/GET_ROWS 24/24, Turbo flash attention 19/19 on both CM2 and CM1, TQ dense matmul 8/8 (four unsupported combinations excluded), and TQ expert matmul 4/4 on both CM2 and CM1. CPU filtered regression: 62/62. `test-quantize-fns` and `test-turbo-quant` pass. The standalone, offline `test-quant-cr` passes all six CR selection groups (42 tensor expectations), including incompatible widths and tensor dimensions. Intel UHD WHT: 9/9. Full-offload Bonsai Turbo2 (auto-asymmetric), Turbo3 and Turbo4 all produced coherent text. A single short Turbo3 benchmark measured pp16 25.09 t/s and tg8 25.43 t/s; these are not sustained throughput claims.

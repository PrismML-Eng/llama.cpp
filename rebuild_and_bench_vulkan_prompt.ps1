$ErrorActionPreference = 'Stop'

# Reconfigure and rebuild Vulkan support together with the CPU backend required by the model loader
Get-Process -Name llama-cli -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process -Name llama-bench -ErrorAction SilentlyContinue | Stop-Process -Force
Remove-Item "C:\D\llm\prismml-llama.cpp\prism-llama.cpp\build\bin\Release\ggml-vulkan.dll" -Force -ErrorAction SilentlyContinue
Remove-Item "C:\D\llm\prismml-llama.cpp\prism-llama.cpp\build\bin\Release\ggml-cpu*.dll" -Force -ErrorAction SilentlyContinue
cd "C:\D\llm\prismml-llama.cpp\prism-llama.cpp"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=OFF -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_VULKAN=ON -DGGML_VULKAN_DEBUG=OFF -DGGML_VULKAN_VALIDATE=OFF -DGGML_VULKAN_RUN_TESTS=OFF
cmake --build build --config Release --target ggml-vulkan llama-cli llama-bench -- /nologo

# Verify Vulkan devices are visible before prompt generation
.\build\bin\Release\llama-cli.exe --list-devices

# Run the exact prompt generation on the first Vulkan device
.\build\bin\Release\llama-cli.exe `
  -m "C:\D\llm\Ternary-Bonsai-27B-Q2_0.gguf" `
  --device Vulkan0 `
  -p "hi! tell me about yourself." `
  -n 64 `
  -c 2048 `
  --temp 0.8 `
  --top-k 40 `
  --top-p 0.95 `
  --repeat-penalty 1.1 `
  -ngl 999 `
  --no-mmap

# Optional: if you want a benchmark run, use this form instead of prompt text
# .\build\bin\Release\llama-bench.exe `
#   -m "C:\D\llm\Ternary-Bonsai-27B-Q2_0.gguf" `
#   --device Vulkan0 `
#   -ngl 999 `
#   --n-gpu-layers 999 `
#   -n 64 `
#   -p 512 `
#   -b 2048 `
#   -ub 512

@echo off
setlocal
set "REPO=%~dp0..\.."
if not defined BONSAI_ROCM_VENV set "BONSAI_ROCM_VENV=D:\llama.cpp\.venv"
set "BONSAI_SDK=%BONSAI_ROCM_VENV%\Lib\site-packages\_rocm_sdk_devel"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "PATH=%BONSAI_ROCM_VENV%\Scripts;%BONSAI_SDK%\bin;%BONSAI_SDK%\lib\llvm\bin;%PATH%"
set "HIP_VISIBLE_DEVICES=1"
cmake -S "%REPO%" -B "%REPO%\build-hip-baseline" -G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx1030 -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DLLAMA_OPENSSL=OFF
if errorlevel 1 exit /b 1
cmake --build "%REPO%\build-hip-baseline" --parallel 12 --target llama-server llama-bench llama-cli llama-quantize test-backend-ops
exit /b %ERRORLEVEL%

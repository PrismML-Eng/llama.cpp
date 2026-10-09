# Prism SYCL release packages

These packages are compiled on standard GitHub-hosted Ubuntu 24.04 and Windows 2022 x64 runners. No custom runner or Intel GPU is needed to build them. Compilation does not validate model correctness or performance on an Intel GPU.

## Linux x64

Choose the Ubuntu SYCL FP32 or FP16 archive. It includes llama-server, llama-cli and the other enabled tools. Install Intel GPU drivers and the matching Intel oneAPI 2025.3.3 libraries, including the SYCL runtime, oneMKL and oneDNN, before running it. The archive does not bundle oneAPI libraries.

Initialize the oneAPI environment, then run from the extracted directory:

```sh
source /opt/intel/oneapi/setvars.sh
./llama-cli --list-devices
```

## Windows x64

The SYCL archive is a backend add-on, not a standalone server package:

1. Extract the Windows x64 CPU archive from the same Prism release.
2. Extract the SYCL archive into the same directory as llama-server.exe and llama-cli.exe, allowing matching shared libraries to be replaced.
3. Install a supported Intel GPU driver, then run `./llama-cli.exe --list-devices` from that directory.

The SYCL archive includes the oneAPI 2025.3.3 runtime DLLs used by the upstream release recipe. Users do not need the oneAPI compiler to run it. Intel GPU drivers are not included.

## Model validation

Check that the Intel GPU appears before running inference. Use a model supported by the fork's SYCL backend, and verify output correctness as well as speed. Report the model filename, release tag, GPU, OS, driver and command when reporting a problem.

For installation and backend options, see https://github.com/PrismML-Eng/llama.cpp/blob/prism/docs/backend/SYCL.md.

"""Run a Bonsai binary with the local TheRock SDK and explicit GPU selection."""
import os
from pathlib import Path
import subprocess
import sys


def environment():
    env = os.environ.copy()
    venv = Path(env.get("BONSAI_ROCM_VENV", "D:/llama.cpp/.venv"))
    sdk = venv / "Lib/site-packages/_rocm_sdk_devel"
    if not sdk.is_dir():
        raise SystemExit(f"TheRock SDK missing: {sdk}")
    env["PATH"] = os.pathsep.join(map(str, [venv / "Scripts", sdk / "bin", sdk / "lib/llvm/bin"])) + os.pathsep + env["PATH"]
    env["HIP_VISIBLE_DEVICES"] = env.get("BONSAI_HIP_DEVICE", "1")
    env.pop("GGML_VK_VISIBLE_DEVICES", None)
    for name in (
        "LLAMA_ARG_SPEC_DRAFT_MODEL",
        "LLAMA_ARG_SPEC_TYPE",
        "LLAMA_ARG_N_GPU_LAYERS_DRAFT",
        "LLAMA_ARG_SPEC_DRAFT_N_MAX",
        "BONSAI_SPECULATIVE",
    ):
        env.pop(name, None)
    return env


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: runtime.py <executable> [arguments...]")
    raise SystemExit(subprocess.call(sys.argv[1:], env=environment()))

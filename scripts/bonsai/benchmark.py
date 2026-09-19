"""Save reproducible llama-bench output and the exact invocation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from runtime import environment

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build", default="build-hip-original")
parser.add_argument("--label", required=True)
parser.add_argument("--threads", default="8")
parser.add_argument("--ubatch", default="512")
parser.add_argument("--cache", default="f16")
parser.add_argument("--prompt", default="512")
parser.add_argument("--generate", default="128")
parser.add_argument("--repetitions", default="3")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
out = repo.parent / "results" / args.label
out.mkdir(parents=True, exist_ok=False)
binary = repo / args.build / "bin/llama-bench.exe"
command = [str(binary), "-m", str(repo.parent / "models/Ternary-Bonsai-27B-PQ2_0.gguf"),
           "-p", args.prompt, "-n", args.generate, "-r", args.repetitions, "-o", "json",
           "-ngl", "999", "-dev", "ROCm0", "-sm", "none", "-fa", "on", "-t", args.threads,
           "-b", "2048", "-ub", args.ubatch, "-ctk", args.cache, "-ctv", args.cache]
env = environment()
record = {"command": command, "hip_visible_devices": env["HIP_VISIBLE_DEVICES"],
          "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
          "hip_dll_sha256": hashlib.sha256((binary.parent / "ggml-hip.dll").read_bytes()).hexdigest()}
start = time.monotonic()
with (out / "bench.json").open("w") as stdout, (out / "bench.stderr.log").open("w") as stderr:
    result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
record.update(exit_code=result.returncode, wall_seconds=time.monotonic() - start)
(out / "invocation.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
if result.returncode == 0:
    for row in json.loads((out / "bench.json").read_text()):
        print(f"pp{row['n_prompt']}/tg{row['n_gen']}: {row['avg_ts']:.2f} +/- {row['stddev_ts']:.2f} tokens/s")
raise SystemExit(result.returncode)

"""Start a dedicated local Bonsai server and keep its command and logs."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

from runtime import environment

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build", default="build-hip-original")
parser.add_argument("--port", type=int, default=8081)
parser.add_argument("--context", type=int, default=32768)
parser.add_argument("--batch", type=int, default=2048)
parser.add_argument("--ubatch", type=int, default=512)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--cache", default="f16")
parser.add_argument("--draft", type=Path)
parser.add_argument("--label", default="baseline")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
root = repo.parent
binary = repo / args.build / "bin/llama-server.exe"
model = root / "models/Ternary-Bonsai-27B-PQ2_0.gguf"
for path in (binary, model):
    if not path.is_file():
        raise SystemExit(f"Missing: {path}")
with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", args.port)) == 0:
        raise SystemExit(f"Port {args.port} is already in use")
out = root / "results" / args.label
out.mkdir(parents=True, exist_ok=True)
command = [str(binary), "-m", str(model), "--alias", "bonsai-27b", "--host", "127.0.0.1",
           "--port", str(args.port), "--jinja", "-c", str(args.context), "-np", "1",
           "-ngl", "999", "--device", "ROCm0", "--split-mode", "none", "-fa", "on",
           "-b", str(args.batch), "-ub", str(args.ubatch), "-t", str(args.threads),
           "-ctk", args.cache, "-ctv", args.cache, "--temp", "0.7", "--top-p", "0.95", "--top-k", "20"]
if args.draft:
    if not args.draft.is_file():
        raise SystemExit(f"Missing drafter: {args.draft}")
    command += ["-md", str(args.draft.resolve()), "--spec-type", "draft-dspark", "--spec-draft-n-max", "4", "-ngld", "999"]
with (out / "server.stdout.log").open("w") as stdout, (out / "server.stderr.log").open("w") as stderr:
    process = subprocess.Popen(command, env=environment(), cwd=binary.parent, stdout=stdout, stderr=stderr,
                               creationflags=subprocess.CREATE_NO_WINDOW)
record = {"pid": process.pid, "command": command, "port": args.port, "label": args.label}
(out / "server.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)


def stop_process(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    if process.poll() is not None:
        raise SystemExit(f"Server exited {process.returncode}; inspect {out}")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=0.5) as response:
            if response.status == 200:
                print("Server healthy", flush=True)
                break
    except (OSError, ValueError):
        time.sleep(0.5)
else:
    stop_process(process)
    raise SystemExit(f"Server startup timed out; inspect {out}")

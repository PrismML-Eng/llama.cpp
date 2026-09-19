"""Measure deterministic chat workloads, including speculative acceptance."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

PROMPTS = {
    "code": "Write a Python function binary_search(items, value) returning the index or -1. Include a short explanation and three assert tests.",
    "math": "Solve 3x + 7 = 52, then explain how to check the answer. Keep the explanation under 100 words.",
    "chat": "Explain how a refrigerator moves heat, in about 150 words.",
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--port", type=int, default=8081)
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()
records = []
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists():
    raise SystemExit("Output already exists; use a fresh path")
for name, prompt in PROMPTS.items():
    for repeat in range(args.repeats):
        payload = {"model": "bonsai-27b", "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "seed": 42, "max_tokens": 384, "cache_prompt": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
        request = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                                         data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        record = {"task": name, "repeat": repeat + 1, "wall_seconds": time.perf_counter() - start,
                  "request": payload, "response": result}
        records.append(record)
        args.output.write_text(json.dumps(records, indent=2))
        print(json.dumps({"task": name, "repeat": repeat + 1, "wall_seconds": record["wall_seconds"],
                          "timings": result.get("timings")}), flush=True)

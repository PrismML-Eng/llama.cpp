"""CPU-only exit-status checks for scripts/bench-models.sh, using fake benchmarks."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench-models.sh"
STUB = """#!/bin/sh
name=${0##*/}
printf '%s\\n' "$name" >> "$BENCH_TEST_CALLS"
printf '%s\\n' "$*" >> "$BENCH_TEST_ARGS"
printf 'fixture benchmark output\\n'
if [ "$name" = "${BENCH_TEST_FAIL_PROGRAM:-}" ]; then
    exit "$BENCH_TEST_FAIL_CODE"
fi
"""


class BenchModelsTests(unittest.TestCase):
    def run_script(self, args=(), fail_program="", fail_code=0, extra_models=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bin").mkdir()
            for name in ("llama-batched-bench", "llama-bench"):
                program = root / "bin" / name
                program.write_text(STUB, encoding="utf-8")
                program.chmod(0o755)
            if extra_models:
                (root / "models-extra.txt").write_text("\norg/model local.gguf\n", encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "BENCH_TEST_CALLS": str(root / "calls.txt"),
                "BENCH_TEST_ARGS": str(root / "args.txt"),
                "BENCH_TEST_FAIL_PROGRAM": fail_program,
                "BENCH_TEST_FAIL_CODE": str(fail_code),
            })
            result = subprocess.run(
                ["bash", str(SCRIPT), *args], cwd=root, env=env,
                capture_output=True, text=True, timeout=10,
            )
            calls = (root / "calls.txt").read_text(encoding="utf-8").splitlines()
            arguments = (root / "args.txt").read_text(encoding="utf-8").splitlines()
            return result, calls, arguments

    def test_success_runs_both_benchmarks_for_all_models(self):
        result, calls, _ = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ["llama-batched-bench", "llama-bench"] * 6)
        self.assertIn("Done! Results are written to", result.stdout)

    def test_quick_dio_and_extra_models(self):
        result, calls, arguments = self.run_script(("--quick", "--dio"), extra_models=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ["llama-batched-bench", "llama-bench"] * 7)
        self.assertTrue(all("--load-mode dio" in args for args in arguments))
        self.assertIn("-c 20480", arguments[0])
        self.assertIn("-d 0 -p 2048 -n 32", arguments[1])
        self.assertIn("-m local.gguf", arguments[-1])

    def test_batched_benchmark_failure_stops_before_decode(self):
        result, calls, _ = self.run_script(fail_program="llama-batched-bench", fail_code=53)
        self.assertEqual(result.returncode, 53, result.stderr)
        self.assertEqual(calls, ["llama-batched-bench"])
        self.assertNotIn("Done!", result.stdout)

    def test_decode_benchmark_failure_stops_before_next_model(self):
        result, calls, _ = self.run_script(fail_program="llama-bench", fail_code=59)
        self.assertEqual(result.returncode, 59, result.stderr)
        self.assertEqual(calls, ["llama-batched-bench", "llama-bench"])
        self.assertNotIn("Done!", result.stdout)


if __name__ == "__main__":
    unittest.main()

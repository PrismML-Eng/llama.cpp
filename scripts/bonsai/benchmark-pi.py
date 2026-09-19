"""Benchmark fixed text and read-tool tasks through the installed Pi CLI."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


SHORT_PROMPT = "Reply with exactly BONSAI_OK and nothing else."
SHORT_EXPECTED = "BONSAI_OK"
TOOL_PROMPT = "Use the read tool to read probe.txt. Reply with its exact contents and nothing else."
TOOL_EXPECTED = "Bonsai Pi read tool check: 7C4A9E"
CONTROLLED_FLAGS = [
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-themes",
    "--no-context-files",
    "--no-approve",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi", default="pi", help="Pi executable or command name")
    parser.add_argument("--output", type=Path, default=None,
                        help="Summary JSON path (default: ../results/pi-baseline.json)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout per Pi task in seconds")
    parser.add_argument("--profile", choices=("ordinary", "controlled", "both"), default="ordinary",
                        help="Ordinary Pi, extension-free controlled Pi, or both")
    parser.add_argument("--expected-provider", default="bonsai-local")
    parser.add_argument("--expected-model", default="bonsai-27b")
    return parser.parse_args()


def resolve_pi(value: str) -> str:
    direct = Path(value).expanduser()
    if direct.is_file():
        return str(direct.resolve())
    found = shutil.which(value)
    if not found:
        raise RuntimeError(f"Pi executable not found: {value}")
    return str(Path(found).resolve())


def sanitize_error(value: str) -> str:
    value = re.sub(r"(?i)(api[-_ ]?key|authorization|bearer)(\s*[:=]\s*)\S+", r"\1\2[redacted]", value)
    value = re.sub(r"\b(sk-[A-Za-z0-9_-]{8,})\b", "[redacted]", value)
    return value[-4000:]


def parse_events(stdout: str) -> dict:
    events = []
    invalid_lines = 0
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue
        if isinstance(event, dict):
            events.append(event)
        else:
            invalid_lines += 1

    assistant_messages = []
    tool_names = []
    for event in events:
        if event.get("type") == "message_end":
            message = event.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                assistant_messages.append(message)
        if event.get("type") == "tool_execution_start" and isinstance(event.get("toolName"), str):
            tool_names.append(event["toolName"])

    usage = {key: 0 for key in ("input", "output", "cacheRead", "cacheWrite", "reasoning", "totalTokens")}
    providers = []
    models = []
    responses = []
    stop_reasons = []
    for message in assistant_messages:
        if isinstance(message.get("provider"), str):
            providers.append(message["provider"])
        if isinstance(message.get("model"), str):
            models.append(message["model"])
        if isinstance(message.get("stopReason"), str):
            stop_reasons.append(message["stopReason"])
        message_usage = message.get("usage")
        if isinstance(message_usage, dict):
            for key in usage:
                value = message_usage.get(key, 0)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] += value
        content = message.get("content")
        if isinstance(content, list):
            text = "".join(
                item.get("text", "") for item in content
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
            )
            if text:
                responses.append(text)

    return {
        "eventCount": len(events),
        "invalidJsonLineCount": invalid_lines,
        "provider": providers[-1] if providers else None,
        "model": models[-1] if models else None,
        "usage": usage,
        "response": responses[-1] if responses else "",
        "toolsUsed": tool_names,
        "stopReasons": stop_reasons,
    }


def command_for(pi: str, profile: str, task: str, prompt: str) -> list[str]:
    command = [pi, "-p", "--mode", "json", "--no-session"]
    if profile == "controlled":
        command.extend(CONTROLLED_FLAGS)
    if task == "read_tool":
        command.extend(["--tools", "read"])
    elif profile == "controlled":
        command.append("--no-tools")
    command.extend(["--", prompt])
    return command


def run_task(pi: str, profile: str, repeat: int, task: str, prompt: str, expected: str,
             cwd: Path, timeout: float, expected_provider: str, expected_model: str) -> dict:
    command = command_for(pi, profile, task, prompt)
    started = time.perf_counter()
    stdout = ""
    stderr = ""
    return_code = None
    timed_out = False
    try:
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=timeout, check=False)
        stdout = completed.stdout
        stderr = completed.stderr
        return_code = completed.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
    wall_seconds = time.perf_counter() - started
    parsed = parse_events(stdout)
    response_matches = parsed["response"].strip() == expected
    tool_matches = task != "read_tool" or "read" in parsed["toolsUsed"]
    identity_matches = parsed["provider"] == expected_provider and parsed["model"] == expected_model
    success = (not timed_out and return_code == 0 and not parsed["invalidJsonLineCount"]
               and response_matches and tool_matches and identity_matches)
    error = None
    if timed_out:
        error = f"Timed out after {timeout:g} seconds"
    elif return_code != 0:
        error = f"Pi exited with code {return_code}"
    elif parsed["invalidJsonLineCount"]:
        error = "Pi emitted non-JSON stdout"
    elif not identity_matches:
        error = "Pi did not report the expected default provider and model"
    elif not tool_matches:
        error = "Pi did not use the read tool"
    elif not response_matches:
        error = "Pi response did not match the fixed expected text"
    if stderr.strip():
        stderr = sanitize_error(stderr)
    return {
        "profile": profile,
        "repeat": repeat,
        "task": task,
        "commandFlags": command[1:-2],
        "wallSeconds": round(wall_seconds, 6),
        "timedOut": timed_out,
        "returnCode": return_code,
        **parsed,
        "responseMatches": response_matches,
        "toolCheckPassed": tool_matches,
        "modelIdentityMatches": identity_matches,
        "success": success,
        "error": error,
        "stderr": stderr or None,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    if args.repeats <= 0 or args.timeout <= 0:
        raise SystemExit("Repeats and timeout must be positive")
    try:
        pi = resolve_pi(args.pi)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    repo = Path(__file__).resolve().parents[2]
    output = args.output.expanduser().resolve() if args.output else (repo.parent / "results" / "pi-baseline.json")
    profiles = ("ordinary", "controlled") if args.profile == "both" else (args.profile,)
    try:
        version = subprocess.run([pi, "--version"], capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise SystemExit("Could not execute the installed Pi CLI") from error

    summary = {
        "schemaVersion": 1,
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "piExecutable": pi,
        "piVersion": version,
        "expectedProvider": args.expected_provider,
        "expectedModel": args.expected_model,
        "profileSelection": args.profile,
        "repeats": args.repeats,
        "timeoutSeconds": args.timeout,
        "prompts": {"short": SHORT_PROMPT, "readTool": TOOL_PROMPT},
        "runs": [],
    }
    with tempfile.TemporaryDirectory(prefix="bonsai-pi-bench-") as directory:
        workdir = Path(directory)
        (workdir / "probe.txt").write_text(TOOL_EXPECTED, encoding="utf-8")
        for profile in profiles:
            for repeat in range(1, args.repeats + 1):
                for task, prompt, expected in (
                    ("short", SHORT_PROMPT, SHORT_EXPECTED),
                    ("read_tool", TOOL_PROMPT, TOOL_EXPECTED),
                ):
                    summary["runs"].append(run_task(
                        pi, profile, repeat, task, prompt, expected, workdir, args.timeout,
                        args.expected_provider, args.expected_model))
                    write_json(output, summary)
    summary["finishedAt"] = datetime.now(timezone.utc).isoformat()
    summary["allPassed"] = all(run["success"] for run in summary["runs"])
    write_json(output, summary)
    print(json.dumps({
        "output": str(output),
        "runs": len(summary["runs"]),
        "allPassed": summary["allPassed"],
    }, indent=2))
    return 0 if summary["allPassed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

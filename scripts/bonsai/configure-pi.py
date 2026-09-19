"""Safely register the local Bonsai server as Pi's default model."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit
import urllib.request


DEFAULT_BASE_URL = "http://127.0.0.1:8081/v1"
DEFAULT_PROVIDER = "bonsai-local"
DEFAULT_MODEL = "bonsai-27b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-dir", type=Path, default=None,
                        help="Pi agent directory (default: PI_CODING_AGENT_DIR or ~/.pi/agent)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--apply", action="store_true",
                        help="Back up and update Pi configuration; otherwise preview only")
    return parser.parse_args()


def agent_dir(argument: Path | None) -> Path:
    if argument is not None:
        return argument.expanduser().resolve()
    configured = os.environ.get("PI_CODING_AGENT_DIR")
    return Path(configured).expanduser().resolve() if configured else (Path.home() / ".pi" / "agent").resolve()


def load_object(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read valid JSON from {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def validated_urls(base_url: str) -> tuple[str, str]:
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("--base-url must use HTTP on a loopback host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RuntimeError("--base-url must not include credentials, a query, or a fragment")
    if parsed.path.rstrip("/") != "/v1":
        raise RuntimeError("--base-url must end in /v1")
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return f"{origin}/health", f"{base_url.rstrip('/')}/models"


def get_json(url: str, timeout: float) -> object:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"Endpoint returned HTTP {response.status}")
            body = response.read(2_000_001)
    except (OSError, TimeoutError, ValueError) as error:
        raise RuntimeError("Local Bonsai endpoint is unavailable") from error
    if len(body) > 2_000_000:
        raise RuntimeError("Local Bonsai endpoint returned an oversized response")
    try:
        return json.loads(body)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Local Bonsai endpoint returned invalid JSON") from error


def verify_endpoint(base_url: str, model_id: str, timeout: float) -> None:
    health_url, models_url = validated_urls(base_url)
    get_json(health_url, timeout)
    models = get_json(models_url, timeout)
    if not isinstance(models, dict) or not isinstance(models.get("data"), list):
        raise RuntimeError("Local Bonsai model list has an unexpected shape")
    model_ids = {item.get("id") for item in models["data"] if isinstance(item, dict)}
    if model_id not in model_ids:
        raise RuntimeError(f"Local endpoint does not identify model {model_id!r}")


def merge_models(current: dict, base_url: str, provider_id: str, model_id: str,
                 context: int, max_tokens: int) -> dict:
    merged = copy.deepcopy(current)
    providers = merged.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise RuntimeError("models.json field 'providers' must be an object")
    provider = copy.deepcopy(providers.get(provider_id, {}))
    if not isinstance(provider, dict):
        raise RuntimeError(f"models.json provider {provider_id!r} must be an object")
    provider.update({"baseUrl": base_url, "api": "openai-completions"})
    provider.setdefault("apiKey", "local")
    compat = provider.setdefault("compat", {})
    if not isinstance(compat, dict):
        raise RuntimeError(f"models.json provider {provider_id!r} compat must be an object")
    compat.update({
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": False,
        "maxTokensField": "max_tokens",
        "thinkingFormat": "qwen-chat-template",
    })
    existing_models = provider.setdefault("models", [])
    if not isinstance(existing_models, list):
        raise RuntimeError(f"models.json provider {provider_id!r} models must be an array")
    desired = {
        "id": model_id,
        "name": "Bonsai 27B (Local)",
        "reasoning": True,
        "input": ["text"],
        "contextWindow": context,
        "maxTokens": max_tokens,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    }
    output_models = []
    found = False
    for item in existing_models:
        if isinstance(item, dict) and item.get("id") == model_id:
            replacement = copy.deepcopy(item)
            replacement.update(desired)
            output_models.append(replacement)
            found = True
        else:
            output_models.append(item)
    if not found:
        output_models.append(desired)
    provider["models"] = output_models
    providers[provider_id] = provider
    return merged


def merge_settings(current: dict, provider_id: str, model_id: str) -> dict:
    merged = copy.deepcopy(current)
    enabled = merged.setdefault("enabledModels", [])
    if not isinstance(enabled, list) or not all(isinstance(item, str) for item in enabled):
        raise RuntimeError("settings.json field 'enabledModels' must be an array of strings")
    qualified_model = f"{provider_id}/{model_id}"
    if qualified_model not in enabled:
        enabled.append(qualified_model)
    merged["defaultProvider"] = provider_id
    merged["defaultModel"] = model_id
    return merged


def write_private_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def backup_files(paths: list[Path], root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = root / "backups" / f"bonsai-{stamp}"
    suffix = 1
    while backup.exists():
        backup = root / "backups" / f"bonsai-{stamp}-{suffix}"
        suffix += 1
    backup.mkdir(parents=True, mode=0o700)
    os.chmod(backup, 0o700)
    for path in paths:
        if path.is_file():
            destination = backup / path.name
            shutil.copy2(path, destination)
            os.chmod(destination, 0o600)
    return backup


def apply_configuration(models_path: Path, settings_path: Path, models: dict, settings: dict) -> Path:
    originals = {
        models_path: models_path.read_bytes() if models_path.is_file() else None,
        settings_path: settings_path.read_bytes() if settings_path.is_file() else None,
    }
    backup = backup_files([models_path, settings_path], models_path.parent)
    try:
        write_private_json(models_path, models)
        write_private_json(settings_path, settings)
    except BaseException as error:
        for path, content in originals.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".restore", dir=path.parent)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary_name, 0o600)
                os.replace(temporary_name, path)
        raise RuntimeError(f"Pi configuration update failed; restored backup from {backup}") from error
    return backup


def main() -> int:
    args = parse_args()
    if args.context <= 0 or args.max_tokens <= 0 or args.max_tokens > args.context:
        raise SystemExit("Context and token limits must be positive, with max tokens no larger than context")
    if args.timeout <= 0:
        raise SystemExit("Timeout must be positive")
    root = agent_dir(args.agent_dir)
    models_path = root / "models.json"
    settings_path = root / "settings.json"
    try:
        models_before = load_object(models_path)
        settings_before = load_object(settings_path)
        models_after = merge_models(models_before, args.base_url, args.provider, args.model,
                                    args.context, args.max_tokens)
        settings_after = merge_settings(settings_before, args.provider, args.model)
        verify_endpoint(args.base_url, args.model, args.timeout)
        result = {
            "mode": "apply" if args.apply else "preview",
            "endpointVerified": True,
            "provider": args.provider,
            "model": args.model,
            "modelsChanged": models_after != models_before,
            "settingsChanged": settings_after != settings_before,
        }
        if args.apply:
            result["backupDirectory"] = str(apply_configuration(
                models_path, settings_path, models_after, settings_after))
        else:
            result["nextStep"] = "Run again with --apply to back up and update Pi configuration"
        print(json.dumps(result, indent=2))
        return 0
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())

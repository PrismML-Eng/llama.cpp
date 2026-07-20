#!/usr/bin/env python3
"""Compare Prism native and opt-in tbkern llama-debug artifacts.

The comparator is deliberately read-only: it never invokes llama.cpp or
modifies either input.  Logit files are little-endian float32 values as
written by examples/debug; token files are compared byte-for-byte because
the debug utility's token representation is part of the artifact contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any


SCHEMA = "tbkern.parity.v1"


def _read_f32(path: Path) -> tuple[list[float], bytes]:
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"empty logits file: {path}")
    if len(raw) % 4:
        raise ValueError(f"logits file is not a multiple of 4 bytes: {path}")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw)), raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_metadata(path: Path, raw: bytes, values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": _sha256(raw),
        "float_count": len(values),
        "finite_count": len(finite),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def _argmax(values: list[float]) -> int:
    if not values:
        raise ValueError("logits contain no values")
    return max(range(len(values)), key=values.__getitem__)


def _row_top1(native: list[float], tbkern: list[float], vocab_size: int | None) -> dict[str, Any]:
    # llama-debug writes the final logits vector, not a sequence of rows.
    # Keep the optional argument as an explicit artifact-shape assertion.
    if vocab_size is not None and vocab_size != len(native):
        raise ValueError("--vocab-size must equal the logits vector length")
    native_ids = [_argmax(native)]
    tbkern_ids = [_argmax(tbkern)]
    return {
        "rows": 1,
        "vocab_size": len(native),
        "mismatched_rows": int(native_ids[0] != tbkern_ids[0]),
        "native": native_ids,
        "tbkern": tbkern_ids,
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("nmse_tolerance", "relative_floor"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a finite non-negative number")
    native_path = Path(args.native_logits)
    tbkern_path = Path(args.tbkern_logits)
    native, native_raw = _read_f32(native_path)
    tbkern, tbkern_raw = _read_f32(tbkern_path)
    if len(native) != len(tbkern):
        raise ValueError(
            f"logit lengths differ: native={len(native)} tbkern={len(tbkern)}"
        )
    if any(not math.isfinite(value) for value in native + tbkern):
        raise ValueError("logit artifacts contain NaN or infinity")

    deltas = [abs(a - b) for a, b in zip(native, tbkern)]
    denominator = sum(value * value for value in native)
    nmse = sum(delta * delta for delta in deltas) / denominator if denominator else 0.0
    nonzero = [delta / abs(value) for value, delta in zip(native, deltas) if abs(value) > args.relative_floor]
    top1 = _row_top1(native, tbkern, args.vocab_size)

    token_result: dict[str, Any] | None = None
    if bool(args.native_tokens) != bool(args.tbkern_tokens):
        raise ValueError("provide both --native-tokens and --tbkern-tokens, or neither")
    if args.native_tokens:
        native_token_path = Path(args.native_tokens)
        tbkern_token_path = Path(args.tbkern_tokens)
        native_token_raw = native_token_path.read_bytes()
        tbkern_token_raw = tbkern_token_path.read_bytes()
        token_result = {
            "native": {
                "path": str(native_token_path),
                "bytes": len(native_token_raw),
                "sha256": _sha256(native_token_raw),
            },
            "tbkern": {
                "path": str(tbkern_token_path),
                "bytes": len(tbkern_token_raw),
                "sha256": _sha256(tbkern_token_raw),
            },
            "equal": native_token_raw == tbkern_token_raw,
        }

    # NMSE has no meaning when the reference vector is all zeros. Only an
    # exactly equal vector passes that degenerate case.
    nmse_pass = nmse <= args.nmse_tolerance and (denominator != 0.0 or max(deltas) == 0.0)
    top1_pass = top1["mismatched_rows"] == 0
    token_pass = token_result is None or token_result["equal"]
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "gates": {
            "nmse": {"value": nmse, "limit": args.nmse_tolerance, "pass": nmse_pass},
            "top1": {"mismatched_rows": top1["mismatched_rows"], "pass": top1_pass},
            "tokens": {"present": token_result is not None, "pass": token_pass},
        },
        "logits": {
            "native": _file_metadata(native_path, native_raw, native),
            "tbkern": _file_metadata(tbkern_path, tbkern_raw, tbkern),
            "max_abs": max(deltas),
            "max_rel": max(nonzero) if nonzero else 0.0,
            "nmse": nmse,
            "reference_sum_sq": denominator,
            "top1": top1,
        },
        "tokens": token_result,
        "tolerance": {
            "nmse": args.nmse_tolerance,
            "relative_floor": args.relative_floor,
        },
        "pass": nmse_pass and top1_pass and token_pass,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-logits", required=True, help="Prism fallback float32 artifact")
    parser.add_argument("--tbkern-logits", required=True, help="GGML_TBKERN_Q2_0=1 float32 artifact")
    parser.add_argument("--native-tokens", help="native token artifact (must be paired)")
    parser.add_argument("--tbkern-tokens", help="tbkern token artifact (must be paired)")
    parser.add_argument("--vocab-size", type=int, help="compare top-1 independently for each row")
    parser.add_argument("--nmse-tolerance", type=float, default=1e-4)
    parser.add_argument("--relative-floor", type=float, default=1e-6)
    parser.add_argument("--json-out", type=Path, help="optional path for the JSON report")
    args = parser.parse_args()
    try:
        result = compare(args)
    except (OSError, ValueError, struct.error) as error:
        print(f"tbkern parity: ERROR: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        try:
            args.json_out.write_text(rendered + "\n", encoding="utf-8")
        except OSError as error:
            print(f"tbkern parity: ERROR: cannot write JSON report: {error}", file=sys.stderr)
            return 2
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

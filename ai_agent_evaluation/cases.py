"""Test cases from a file: JSONL or CSV, one case per line or row.

    {"name": "budget running shoes", "input": "Running shoes under $100, size 10", "expected": {"product_id": "RUN-102"}}

    name,input,expected,must_pass
    budget running shoes,"Running shoes under $100, size 10","{""product_id"": ""RUN-102""}",true

Fields:
- input (required): what the agent receives. Any JSON value in JSONL; text in CSV, or JSON text.
- expected: what a correct answer contains. Also read from `expected_output` or `target`, the
  names other evaluation tools use, so their datasets load as they are.
- name: a unique name. Defaults to "case 1", "case 2" and so on.
- check: the name of a check to use for this case (see below). Defaults to matches_expected.
- must_pass, strict: true or false.

Built in checks, each comparing the agent's output with `expected`:
- matches_expected: for a dict, every expected key has the expected value in the output (other
  keys are ignored); for a list, the output is one of its items; otherwise the output equals it.
- equals_expected: the output equals `expected` exactly.
- contains_expected: the output, as text, contains `expected` (case insensitive).

Your own checks: pass check=fn to use it for every case, or checks={"name": fn} to pick one per
case by name. A check takes (output) or (output, expected).
"""

from __future__ import annotations

import csv
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .suite import Case

MISSING = object()


def matches_expected(output: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        if not isinstance(output, dict):
            raise AssertionError(f"expected a dict with {expected}, got {_short(output)}")
        for key, value in expected.items():
            if key not in output:
                raise AssertionError(f"missing {key!r}; got {_short(output)}")
            if output[key] != value:
                raise AssertionError(f"{key}: expected {value!r}, got {output[key]!r}")
    elif isinstance(expected, list):
        if output not in expected:
            raise AssertionError(f"expected one of {expected}, got {_short(output)}")
    elif output != expected:
        raise AssertionError(f"expected {expected!r}, got {_short(output)}")


def equals_expected(output: Any, expected: Any) -> None:
    if output != expected:
        raise AssertionError(f"expected {expected!r}, got {_short(output)}")


def contains_expected(output: Any, expected: Any) -> None:
    text = output if isinstance(output, str) else json.dumps(output, default=str)
    if str(expected).lower() not in text.lower():
        raise AssertionError(f"expected the output to contain {expected!r}; got {_short(output)}")


BUILT_IN: dict[str, Callable[..., Any]] = {
    "matches_expected": matches_expected,
    "equals_expected": equals_expected,
    "contains_expected": contains_expected,
}


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 120 else text[:117] + "..."


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "yes", "1", "y")


def _maybe_json(value: Any) -> Any:
    """CSV cells are text; a cell that holds a JSON object or list is read as one."""
    if isinstance(value, str) and value.strip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8-sig")
    if suffix in (".jsonl", ".ndjson"):
        rows = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{p}:{number}: not valid JSON ({e})") from e
            if not isinstance(row, dict):
                raise ValueError(f"{p}:{number}: each line must be a JSON object")
            rows.append(row)
        return rows
    if suffix == ".csv":
        return [{k.strip(): _maybe_json(v) for k, v in row.items() if k and v not in (None, "")} for row in csv.DictReader(text.splitlines())]
    raise ValueError(f"{p}: use a .jsonl or .csv file")


def _bind(fn: Callable[..., Any], expected: Any) -> Callable[[Any], Any]:
    try:
        params = [p for p in inspect.signature(fn).parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    except (TypeError, ValueError):
        params = []
    if len(params) < 2:
        return fn

    def check(output: Any) -> Any:
        return fn(output, expected)

    check.__name__ = getattr(fn, "__name__", "check")
    return check


def load_cases(
    path: str | Path,
    check: Callable[..., Any] | None = None,
    checks: dict[str, Callable[..., Any]] | None = None,
) -> list[Case]:
    registry = {**BUILT_IN, **(checks or {})}
    cases = []
    for number, row in enumerate(read_rows(path), start=1):
        where = f"{path}: case {number}"
        if "input" not in row:
            raise ValueError(f"{where}: missing 'input'")
        expected = next((row[k] for k in ("expected", "expected_output", "target") if k in row), MISSING)
        if row.get("check"):
            if row["check"] not in registry:
                raise ValueError(f"{where}: unknown check {row['check']!r}; known: {', '.join(sorted(registry))}")
            fn = registry[row["check"]]
        elif check is not None:
            fn = check
        elif expected is not MISSING:
            fn = matches_expected
        else:
            raise ValueError(f"{where}: no 'expected' value and no check; add one, or pass check=")
        cases.append(
            Case(
                name=str(row.get("name") or row.get("id") or f"case {number}"),
                input=row["input"],
                check=_bind(fn, None if expected is MISSING else expected),
                must_pass=_bool(row.get("must_pass", False)),
                strict=_bool(row.get("strict", False)),
            )
        )
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases

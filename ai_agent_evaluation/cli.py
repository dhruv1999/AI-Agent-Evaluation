"""Command line.

    ai-agent-evaluation compare suite.py --baseline-model model-old --candidate-model model-new
    ai-agent-evaluation compare suite.py --baseline-model model-old --candidate-model model-new --smoke

Exit codes: 0 pass; 1 regressed; 10 not enough data (0 with --on-inconclusive warn); 11 too many
provider errors for a verdict; 2 bad usage, or the suite file could not be loaded.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import traceback
from pathlib import Path

from .compare import compare_models
from .report import markdown, text
from .suite import Suite


class UsageError(Exception):
    pass


def load_suite(path: Path, name: str | None = None) -> Suite:
    if not path.is_file():
        raise UsageError(f"no such file: {path}")
    sys.path.insert(0, str(path.resolve().parent))
    module_name = "ai_agent_evaluation_suite_" + re.sub(r"\W", "_", path.stem)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise UsageError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses and pickling look classes up by module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        traceback.print_exc()
        raise UsageError(f"could not load {path}: {type(e).__name__}: {e}") from e
    suites = list({id(v): v for v in vars(module).values() if isinstance(v, Suite)}.values())
    if name:
        suites = [s for s in suites if s.name == name]
    if len(suites) != 1:
        found = ", ".join(str(s.name) for s in suites) or "none"
        raise UsageError(f"{path} must define exactly one Suite (found: {found}); pick one with --suite NAME")
    return suites[0]


def _price(value: str) -> tuple[str, tuple[float, float]]:
    try:
        model, numbers = value.rsplit("=", 1)
        pin, pout = (float(x) for x in numbers.split(","))
    except ValueError as e:
        raise argparse.ArgumentTypeError("use MODEL=INPUT,OUTPUT in USD per million tokens, e.g. model-new=0.8,4") from e
    return model, (pin, pout)


def _cmd_compare(args: argparse.Namespace) -> int:
    suite = load_suite(Path(args.suite), args.suite_name)
    suite.prices.update(dict(args.price or []))
    if args.concurrency:
        suite.concurrency = args.concurrency
    result = compare_models(
        suite,
        args.baseline_model,
        args.candidate_model,
        runs=args.runs,
        margin=args.margin,
        seed=args.seed,
        smoke=args.smoke,
        progress=lambda line: print(f"  {line}", file=sys.stderr, flush=True),
    )
    print(text(result))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval-verdict.json").write_text(json.dumps(result.to_dict(), indent=1), encoding="utf-8")
    report = markdown(result)
    (out / "eval-report.md").write_text(report, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(report + "\n")
    print(f"\nReport: {out / 'eval-report.md'}   Details: {out / 'eval-verdict.json'}")
    return result.exit_code(args.on_inconclusive)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ai-agent-evaluation", description="Should your agent switch models? Compare two models on your own cases.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="compare a baseline and a candidate model on a suite")
    c.add_argument("suite", help="Python file that defines one Suite")
    c.add_argument("--baseline-model", required=True, help="the model the agent uses today")
    c.add_argument("--candidate-model", required=True, help="the model you are thinking of switching to")
    c.add_argument("--runs", type=int, default=None, help="runs per case per model, in two looks (default: the suite's, 6)")
    c.add_argument("--margin", type=float, default=None, help="how much worse is acceptable, e.g. 0.10 for 10 points (default: the suite's)")
    c.add_argument("--smoke", action="store_true", help="one run per case per model: a quick check, not a verdict")
    c.add_argument("--seed", type=int, default=None, help="seed for the run order, to repeat a comparison")
    c.add_argument("--concurrency", type=int, default=None, help="runs at a time (default: the suite's, 4)")
    c.add_argument("--price", type=_price, action="append", help="MODEL=INPUT,OUTPUT USD per million tokens (repeatable)")
    c.add_argument("--on-inconclusive", choices=["block", "warn"], default="block", help="exit code for not enough data: block exits 10, warn exits 0")
    c.add_argument("--out", default="eval-results", help="folder for eval-report.md and eval-verdict.json")
    c.add_argument("--suite", dest="suite_name", default=None, help="which Suite, if the file defines more than one")
    args = ap.parse_args(argv)
    try:
        return _cmd_compare(args)
    except (UsageError, ValueError, OSError) as e:
        print(f"ai-agent-evaluation: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

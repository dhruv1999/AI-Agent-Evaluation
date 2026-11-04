"""Run a model comparison from pytest, next to your other tests.

    from ai_agent_evaluation import assert_model_swap

    def test_switch_to_model_new():
        assert_model_swap(suite, "model-old", "model-new")

The whole comparison is one test. It fails when the verdict is regressed or errors, and also on
not enough data unless on_inconclusive="warn". The report is written to eval-results/. Do not run
it under a rerun plugin: rerunning until it passes would undo what the repeated runs measure.
"""

from __future__ import annotations

import json
from pathlib import Path

from .compare import PASSED, SwapResult, compare_models
from .report import markdown, text
from .suite import Suite


def assert_model_swap(
    suite: Suite,
    baseline_model: str,
    candidate_model: str,
    *,
    runs: int | None = None,
    margin: float | None = None,
    seed: int | None = None,
    on_inconclusive: str = "block",
    out: str | Path = "eval-results",
) -> SwapResult:
    result = compare_models(suite, baseline_model, candidate_model, runs=runs, margin=margin, seed=seed)
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "eval-verdict.json").write_text(json.dumps(result.to_dict(), indent=1), encoding="utf-8")
    (folder / "eval-report.md").write_text(markdown(result), encoding="utf-8")
    if result.verdict != PASSED and result.exit_code(on_inconclusive) != 0:
        raise AssertionError("\n" + text(result))
    return result

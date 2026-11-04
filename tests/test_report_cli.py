import json
import textwrap

import pytest
from conftest import suite_of

from ai_agent_evaluation import assert_model_swap, compare_models
from ai_agent_evaluation.cli import main
from ai_agent_evaluation.report import markdown, text

SUITE = """
from __future__ import annotations

import os
from dataclasses import dataclass

from ai_agent_evaluation import Case, Suite, record_usage


@dataclass
class Page:  # a dataclass in a suite file must not break loading
    name: str


def agent(case_input, model):
    record_usage(500, 50)
    if model == "broken" or (model == "flaky" and case_input == 0):
        return "wrong"
    if model == "down":
        from openai_like import RateLimitError
        raise RateLimitError("busy")
    return "right"


suite = Suite(
    agent=agent,
    cases=[Case(f"case {i}", i, check=lambda o: o == "right", must_pass=(i == 0)) for i in range(30)],
    prices={"old": (3.0, 15.0)},
    name="demo | suite",
)
"""

OPENAI_LIKE = """
class RateLimitError(Exception):
    status_code = 429
RateLimitError.__module__ = "openai._exceptions"
"""


@pytest.fixture
def suite_file(tmp_path):
    (tmp_path / "openai_like.py").write_text(OPENAI_LIKE, encoding="utf-8")
    path = tmp_path / "suite.py"
    path.write_text(textwrap.dedent(SUITE), encoding="utf-8")
    return str(path)


def run(suite_file, tmp_path, *extra):
    return main(["compare", suite_file, "--baseline-model", "old", "--out", str(tmp_path / "out"), "--seed", "1", *extra])


def test_exit_codes(suite_file, tmp_path, monkeypatch, capsys):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert run(suite_file, tmp_path, "--candidate-model", "new", "--price", "new=1,5") == 0
    out = capsys.readouterr().out
    assert "PASS." in out and "cost per run" in out and "% lower" in out
    verdict = json.loads((tmp_path / "out" / "eval-verdict.json").read_text(encoding="utf-8"))
    assert verdict["verdict"] == "pass" and len(verdict["case_results"]) == 30 and verdict["seed"] == 1
    assert "Model swap: old → new" in summary.read_text(encoding="utf-8")
    assert run(suite_file, tmp_path, "--candidate-model", "broken") == 1
    assert run(suite_file, tmp_path, "--candidate-model", "flaky") == 1  # the must pass case breaks
    assert run(suite_file, tmp_path, "--candidate-model", "down") == 11
    assert run(suite_file, tmp_path, "--candidate-model", "new", "--smoke") == 0


def test_usage_errors(suite_file, tmp_path, capsys):
    assert run(str(tmp_path / "missing.py"), tmp_path, "--candidate-model", "new") == 2
    assert run(suite_file, tmp_path, "--candidate-model", "old") == 2
    assert run(suite_file, tmp_path, "--candidate-model", "new", "--runs", "2") == 2
    bad = tmp_path / "bad.py"
    bad.write_text("import not_a_module\n", encoding="utf-8")
    assert run(str(bad), tmp_path, "--candidate-model", "new") == 2
    assert "not_a_module" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run(suite_file, tmp_path, "--candidate-model", "new", "--price", "nonsense")


def test_reports_read_plainly():
    plan = {("c0", "new"): [False]}
    r = compare_models(suite_of(40, plan, must_pass={"c0"}), "old", "new", seed=1)
    t = text(r)
    assert t.startswith("Model swap: old -> new") and "REGRESSED. Must pass case dropped: c0 (6/6 -> 0/6)." in t
    assert "c0: 6/6 -> 0/6  FAILED" in t and "Seed 1." in t
    md = markdown(r)
    assert "**REGRESSED." in md and "| c0 (must pass) | 6/6 | 0/6 | 0 |" in md
    assert all(ord(ch) < 128 for ch in t)  # the console report must print on any terminal


def test_pytest_helper(tmp_path):
    assert assert_model_swap(suite_of(40), "old", "new", out=tmp_path).verdict == "pass"
    assert (tmp_path / "eval-report.md").exists()
    with pytest.raises(AssertionError, match="REGRESSED"):
        assert_model_swap(suite_of(40, {(f"c{i}", "new"): [False] for i in range(40)}), "old", "new", out=tmp_path)
    small = suite_of(6, {("c0", "new"): [False, True], ("c1", "new"): [False, True]})
    with pytest.raises(AssertionError, match="NOT ENOUGH DATA"):
        assert_model_swap(small, "old", "new", seed=4, out=tmp_path)
    assert assert_model_swap(small, "old", "new", seed=4, on_inconclusive="warn", out=tmp_path).verdict == "not enough data"

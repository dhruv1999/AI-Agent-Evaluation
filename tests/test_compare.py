import time

import pytest
from conftest import Scripted, suite_of

from ai_agent_evaluation import Case, Suite, compare_models, record_usage
from ai_agent_evaluation.compare import ERRORS, NOT_ENOUGH_DATA, PASSED, REGRESSED, SMOKE, Tally, decide, run_comparison
from ai_agent_evaluation.errors import provider_error


class RateLimitError(Exception):
    status_code = 429


RateLimitError.__module__ = "openai._exceptions"


def test_same_model_quality_passes():
    r = compare_models(suite_of(40), "old", "new", seed=1)
    assert r.verdict == PASSED and r.exit_code() == 0
    assert r.answer.lower > -0.10 and r.runs_used <= 480
    assert "at most" in r.reason or "no worse" in r.reason


def test_a_broken_model_regresses_and_stops_early():
    plan = {(f"c{i}", "new"): [False] for i in range(40)}
    r = compare_models(suite_of(40, plan), "old", "new", seed=1)
    assert r.verdict == REGRESSED and r.exit_code() == 1
    assert r.look == 1 and r.runs_used == 240  # settled at half the budget


def test_one_broken_must_pass_case_fails_even_when_the_suite_passes():
    plan = {("c0", "new"): [False]}
    r = compare_models(suite_of(40, plan, must_pass={"c0"}), "old", "new", seed=1)
    assert r.verdict == REGRESSED and "c0 (6/6 -> 0/6)" in r.reason
    assert r.look == 2  # a must pass case with failures always gets the full budget
    plain = compare_models(suite_of(40, plan), "old", "new", seed=1)
    assert plain.verdict == PASSED and plain.diagnostics[0].name == "c0"  # without must pass: shown, not blocking
    assert r.diagnostics == []  # must pass cases are reported in their own section


def test_strict_must_pass_fails_on_one_failure():
    plan = {("c0", "new"): [True, True, True, True, True, False]}
    strict = compare_models(suite_of(40, plan, strict={"c0"}), "old", "new", seed=3)
    relaxed = compare_models(suite_of(40, plan, must_pass={"c0"}), "old", "new", seed=3)
    assert strict.verdict == REGRESSED and relaxed.verdict == PASSED


def test_provider_errors_stay_out_of_the_score():
    plan = {(f"c{i}", "new"): [True] * 5 + [RateLimitError("busy")] for i in range(0, 40, 4)}
    r = compare_models(suite_of(40, plan), "old", "new", seed=2)
    assert r.verdict == PASSED
    assert r.errors["candidate"] >= 1 and r.errors["reasons"] == {"HTTP 429": r.errors["candidate"]}
    assert all(t.c_fail == 0 for t in r.tallies)


def test_too_many_provider_errors_means_no_verdict():
    plan = {(f"c{i}", m): [RateLimitError("down")] for i in range(40) for m in ("old", "new")}
    r = compare_models(suite_of(40, plan), "old", "new", seed=2)
    assert r.verdict == ERRORS and r.exit_code() == 11


def test_a_must_pass_case_lost_to_errors_is_not_enough_data():
    plan = {("c0", "new"): [RateLimitError("busy")]}
    r = compare_models(suite_of(40, plan, must_pass={"c0"}), "old", "new", seed=2)
    assert r.verdict == NOT_ENOUGH_DATA and "c0" in r.reason and r.exit_code() == 10
    assert r.exit_code("warn") == 0


def test_more_errors_on_the_new_model_is_flagged():
    plan = {(f"c{i}", "new"): [True, True, RateLimitError("x")] for i in range(0, 40, 2)}
    r = compare_models(suite_of(40, plan), "old", "new", seed=2)
    assert r.errors["candidate_rate_higher"]


def test_too_few_cases_or_runs_is_not_enough_data():
    t = [Tally(f"c{i}", b_pass=2, b_fail=1, c_pass=2, c_fail=1) for i in range(4)]
    assert decide(t, 0.10, 1.678).status == "undecided"
    plan = {("c0", "new"): [False, True], ("c1", "new"): [False, True]}
    r = compare_models(suite_of(6, plan), "old", "new", seed=4)
    assert r.verdict == NOT_ENOUGH_DATA and "would likely" in r.reason and "points worse" in r.reason


def test_timeouts_and_crashes_count_against_the_model():
    def agent(case_input, model):
        if model == "new" and case_input == "slow":
            time.sleep(2)
        if model == "new" and case_input == "crash":
            raise KeyError("missing field")
        return True

    cases = [Case("slow", "slow", check=lambda o: o is True), Case("crash", "crash", check=lambda o: o is True)]
    r = compare_models(Suite(agent=agent, cases=cases, timeout=0.2), "old", "new", seed=1)
    rows = {t.name: t for t in r.tallies}
    assert rows["slow"].c_fail == rows["slow"].c_n > 0 and rows["slow"].c_err == 0
    assert rows["crash"].c_fail > 0 and rows["crash"].c_err == 0


def test_check_semantics():
    def assert_style(o):
        assert o == "ok", "wrong"

    for check, output, passes in [
        (lambda o: o == "ok", "ok", True),
        (lambda o: o == "ok", "no", False),
        (assert_style, "no", False),
        (lambda o: None, "x", True),
    ]:
        r = compare_models(Suite(agent=lambda i, m, out=output: out, cases=[Case("a", 1, check=check)], runs=4), "old", "new", seed=1)
        assert (r.tallies[0].c_pass == r.tallies[0].c_n) is passes


def test_cost_and_speed_per_model():
    def agent(case_input, model):
        record_usage(1000, 100)
        time.sleep(0.02 if model == "old" else 0.0)
        return True

    s = Suite(agent=agent, cases=[Case(str(i), i, check=bool) for i in range(5)], prices={"old": (3.0, 15.0), "new": (1.0, 5.0)}, runs=4)
    r = compare_models(s, "old", "new", seed=1)
    old, new = r.models
    assert old.cost_per_run == pytest.approx(0.0045) and new.cost_per_run == pytest.approx(0.0015)
    assert old.median_seconds > new.median_seconds
    assert old.input_tokens_per_run == 1000


def test_runs_alternate_models_in_a_repeatable_order():
    seen = []

    def agent(case_input, model):
        seen.append(model)
        return True

    s = Suite(agent=agent, cases=[Case(str(i), i, check=bool) for i in range(10)], concurrency=1, runs=4)
    compare_models(s, "old", "new", seed=7)
    first = list(seen)
    seen.clear()
    compare_models(s, "old", "new", seed=7)
    assert seen == first and first[:10].count("old") not in (0, 10)


def test_smoke_mode_and_identical_output_note():
    r = compare_models(suite_of(5), "old", "new", smoke=True, seed=1)
    assert r.verdict == SMOKE and r.runs_used == 10 and r.exit_code() == 0
    full = compare_models(suite_of(5), "old", "new", seed=1)
    assert any("same output" in n for n in full.notes)


def test_bad_settings():
    with pytest.raises(ValueError):
        compare_models(suite_of(3), "same", "same")
    with pytest.raises(ValueError):
        suite_of(3, runs=2)
    with pytest.raises(ValueError):
        Suite(agent=lambda i, m: 1, cases=[Case("a", 1, check=bool), Case("a", 2, check=bool)])
    with pytest.raises(ValueError):
        Suite(agent=lambda i, m: 1, cases=[Case("a", 1, check=bool)], margin=10)


def test_run_comparison_is_pure():
    cases = [Case(str(i), None, check=bool) for i in range(30)]
    verdict, _, look, answer, *_ = run_comparison(cases, 0.10, 6, lambda n: ({str(i): (n, 0, 0, n, 0, 0) for i in range(30)}, []))
    assert verdict == PASSED and answer.cases_used == 30 and look in (1, 2)
    assert provider_error(RateLimitError("x")) == "HTTP 429"
    assert Scripted({}).__name__ == "scripted"

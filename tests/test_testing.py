import pytest

from ai_agent_evaluation import Case, Suite, expect, tool
from ai_agent_evaluation.testing import assert_no_regressions

STATE = {"broken": False}


@tool
def escalate() -> str:
    return "ok"


def agent(message):
    if not STATE["broken"]:
        escalate()
    return "done"


suite = Suite(agent, [Case("escalates", "refund 77", [expect.calls("escalate")])], max_repeats=4, name="pytest_suite")


def test_baseline_life_cycle(tmp_path, monkeypatch):
    path = tmp_path / "pytest_suite.baseline.json"
    monkeypatch.delenv("AI_AGENT_EVALUATION_UPDATE", raising=False)
    STATE["broken"] = False

    monkeypatch.setenv("CI", "true")
    with pytest.raises(AssertionError, match="AI_AGENT_EVALUATION_UPDATE=1"):
        assert_no_regressions(suite, path)  # in CI a missing baseline fails

    monkeypatch.delenv("CI")
    assert assert_no_regressions(suite, path).cases[0].n == 4  # locally it is created
    assert path.exists()
    assert assert_no_regressions(suite, path).ok

    STATE["broken"] = True
    with pytest.raises(AssertionError, match="REGRESSED"):
        assert_no_regressions(suite, path)

    monkeypatch.setenv("AI_AGENT_EVALUATION_UPDATE", "1")  # accept the new behaviour on purpose
    assert_no_regressions(suite, path)
    monkeypatch.delenv("AI_AGENT_EVALUATION_UPDATE")
    assert_no_regressions(suite, path)
    STATE["broken"] = False

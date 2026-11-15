from types import SimpleNamespace as NS

import pytest

from ai_agent_evaluation import Case, Suite, compare_models, record_response, record_usage
from ai_agent_evaluation.recording import Usage, _current
from ai_agent_evaluation.report import text


def recorded(fn):
    usage = Usage()
    token = _current.set(usage)
    try:
        fn()
    finally:
        _current.reset(token)
    return usage


def test_reasoning_tokens_and_calls_from_each_sdk():
    chat = NS(usage=NS(prompt_tokens=100, completion_tokens=50, completion_tokens_details=NS(reasoning_tokens=30)))
    responses = NS(usage=NS(input_tokens=100, output_tokens=40, output_tokens_details=NS(reasoning_tokens=10)))
    anthropic = NS(usage=NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0))
    gemini = NS(usage_metadata=NS(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=15))
    u = recorded(lambda: [record_response(r) for r in (chat, responses, anthropic, gemini, NS())])
    assert (u.llm_calls, u.input_tokens, u.output_tokens, u.reasoning_tokens) == (5, 400, 145, 55)


def test_cost_per_correct_counts_failed_runs():
    def agent(case_input, model):
        record_usage(1000, 100)
        if model == "cheap":
            record_usage(1000, 100)  # a second call: retries cost money too
        return model == "dear" or case_input % 2 == 0  # cheap passes only half the cases

    s = Suite(agent=agent, cases=[Case(str(i), i, check=bool) for i in range(10)], runs=4, prices={"dear": (10.0, 30.0), "cheap": (1.0, 3.0)})
    r = compare_models(s, "dear", "cheap", seed=1)
    dear, cheap = r.models
    assert dear.cost_per_run == pytest.approx(0.013) and dear.cost_per_correct == pytest.approx(0.013)
    assert cheap.cost_per_run == pytest.approx(0.0026)
    assert cheap.cost_per_correct == pytest.approx(0.0052)  # every failed run is paid for by the passing ones
    assert (dear.calls_per_run, cheap.calls_per_run) == (1, 2)
    report = text(r)
    assert "cost per correct" in report and "calls per run" in report
    assert "Cost per correct task: cheap is 60% cheaper, against 80% cheaper per run." in report


def test_no_passes_means_no_cost_per_correct():
    s = Suite(agent=lambda i, m: record_usage(10, 10) or False, cases=[Case("a", 1, check=bool)], runs=4, prices={"a": (1, 1), "b": (1, 1)})
    r = compare_models(s, "a", "b", seed=1)
    assert r.models[0].cost_per_run is not None and r.models[0].cost_per_correct is None

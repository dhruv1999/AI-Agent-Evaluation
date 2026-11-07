import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "shopping_assistant"


def load(monkeypatch):
    monkeypatch.setenv("DEMO_SEED", "7")
    monkeypatch.syspath_prepend(str(EXAMPLE))
    sys.modules.pop("agent", None)
    import agent

    return agent


def test_cases_file_matches_the_catalog(monkeypatch):
    agent = load(monkeypatch)
    for line in (EXAMPLE / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        answer = agent.correct_answer(row["input"])
        assert all(answer[k] == v for k, v in row["expected"].items()), row["name"]


def test_simulated_comparisons(monkeypatch):
    load(monkeypatch)
    from ai_agent_evaluation import compare_models
    from ai_agent_evaluation.cli import load_suite

    suite = load_suite(EXAMPLE / "suite.py")
    assert len(suite.cases) == 30
    assert compare_models(suite, "large-model", "small-model", seed=7).verdict == "pass"
    assert compare_models(suite, "large-model", "tiny-model", seed=7).verdict == "regressed"


class FakeOpenAI:
    """Answers like the OpenAI SDK would: one tool call, then the final JSON."""

    def __init__(self):
        self.calls = 0
        self.chat = NS(completions=NS(create=self.create))

    def create(self, model, messages, tools):
        self.calls += 1
        usage = NS(prompt_tokens=800, completion_tokens=40)
        if self.calls == 1:
            call = NS(id="t1", function=NS(name="search_products", arguments=json.dumps({"category": "running shoes", "max_price": 100, "size": "10"})))
            return NS(choices=[NS(message=NS(content=None, tool_calls=[call]))], usage=usage)
        found = json.loads(messages[-1]["content"])
        reply = json.dumps({"action": "recommend", "product_id": found[0]["id"], "message": "ok"})
        return NS(choices=[NS(message=NS(content=reply, tool_calls=None))], usage=usage)


def test_live_path_with_a_fake_client(monkeypatch):
    agent = load(monkeypatch)
    out = agent._live("Find running shoes under $100 in size 10", "any-model", client=FakeOpenAI())
    assert out["product_id"] == "RUN-101"

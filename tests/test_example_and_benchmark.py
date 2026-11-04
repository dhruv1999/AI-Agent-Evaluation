import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_runs():
    out = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "model_swap_benchmark.py"), "--sims", "20"], capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout
    assert "| the same |" in out and "Run once rule" in out


def test_xpath_example(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_SEED", "7")
    monkeypatch.syspath_prepend(str(ROOT / "examples" / "xpath_healing"))
    from ai_agent_evaluation import compare_models
    from ai_agent_evaluation.cli import load_suite

    suite = load_suite(ROOT / "examples" / "xpath_healing" / "suite.py")
    assert compare_models(suite, "model-old", "model-new", seed=7).verdict == "pass"
    worse = compare_models(suite, "model-old", "model-mini", seed=7)
    assert worse.verdict == "regressed" and worse.models[1].cost_per_run < worse.models[0].cost_per_run

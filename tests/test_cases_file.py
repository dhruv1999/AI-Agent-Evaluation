import json
from pathlib import Path

import pytest

from ai_agent_evaluation import Suite, compare_models, contains_expected, equals_expected, load_cases, matches_expected

ROOT = Path(__file__).resolve().parents[1]


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def run_check(case, output):
    try:
        return all(check(output) is not False for check in case.checks)
    except AssertionError:
        return False


def test_jsonl_cases(tmp_path):
    path = write(
        tmp_path,
        "cases.jsonl",
        "\n".join(
            [
                json.dumps({"name": "refund", "input": "charged twice", "expected": {"queue": "billing"}, "must_pass": True}),
                json.dumps({"input": {"message": "where is it"}, "expected_output": "shipping"}),
                json.dumps({"name": "tone", "input": "hi", "target": "hello", "check": "contains_expected"}),
                "",
            ]
        ),
    )
    cases = load_cases(path)
    assert [c.name for c in cases] == ["refund", "case 2", "tone"]
    assert cases[0].must_pass and cases[1].input == {"message": "where is it"}
    assert run_check(cases[0], {"queue": "billing", "reply": "sorry"}) and not run_check(cases[0], {"queue": "sales"})
    assert run_check(cases[1], "shipping") and not run_check(cases[1], "billing")
    assert run_check(cases[2], "Hello there") and not run_check(cases[2], "bye")


def test_csv_cases_read_json_cells_and_booleans():
    cases = load_cases(ROOT / "examples" / "shopping_assistant" / "cases_sample.csv")
    assert [c.must_pass for c in cases] == [True, False, False]
    assert run_check(cases[0], {"action": "recommend", "product_id": "RUN-101", "message": "x"})
    assert not run_check(cases[1], {"action": "recommend", "product_id": "RUN-101"})


def test_custom_checks_by_argument_and_by_name(tmp_path):
    path = write(tmp_path, "c.jsonl", json.dumps({"input": 1, "expected": 2}) + "\n" + json.dumps({"input": 1, "check": "is_odd"}))
    cases = load_cases(path, checks={"is_odd": lambda o: o % 2 == 1})
    assert run_check(cases[0], 2) and run_check(cases[1], 3) and not run_check(cases[1], 4)
    plain = load_cases(write(tmp_path, "d.jsonl", json.dumps({"input": 1})), check=lambda o: o == "ok")
    assert run_check(plain[0], "ok")


def test_bad_files(tmp_path):
    for name, text, message in [
        ("a.jsonl", "{not json", "not valid JSON"),
        ("b.jsonl", json.dumps({"expected": 1}), "missing 'input'"),
        ("c.jsonl", json.dumps({"input": 1}), "no 'expected'"),
        ("d.jsonl", json.dumps({"input": 1, "expected": 1, "check": "nope"}), "unknown check"),
        ("e.txt", "x", "use a .jsonl or .csv"),
        ("f.jsonl", "", "no cases"),
    ]:
        with pytest.raises(ValueError, match=message):
            load_cases(write(tmp_path, name, text))


def test_built_in_checks_explain_failures():
    with pytest.raises(AssertionError, match="product_id: expected 'A', got 'B'"):
        matches_expected({"product_id": "B"}, {"product_id": "A"})
    with pytest.raises(AssertionError, match="expected one of"):
        matches_expected("x", ["a", "b"])
    with pytest.raises(AssertionError):
        equals_expected({"a": 1, "b": 2}, {"a": 1})
    contains_expected({"reply": "Your Order ships today"}, "order ships")


def test_suite_from_file_and_comparison(tmp_path, monkeypatch):
    path = write(tmp_path, "cases.jsonl", "\n".join(json.dumps({"name": f"c{i}", "input": i, "expected": i}) for i in range(30)))
    monkeypatch.chdir(ROOT)
    suite = Suite.from_file(str(path), agent=lambda i, m: i, runs=4)
    assert suite.name == "cases" and len(suite.cases) == 30
    assert compare_models(suite, "old", "new", seed=1).verdict == "pass"

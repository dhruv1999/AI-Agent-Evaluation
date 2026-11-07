"""What to test: the agent, its cases, and how each run is judged.

    def heal(page, model):          # your agent: takes a case input and a model name
        ...

    def finds_submit(xpath):        # a check: return False, or raise AssertionError, to fail the run
        ...

    suite = Suite(agent=heal, cases=[Case("submit button", page, check=finds_submit, must_pass=True)])

Cases can also come from a JSONL or CSV file: Suite.from_file("cases.jsonl", agent=heal).

Checks must be decided by code: the right element found, the right tool called, valid output.
Answers that need a person or a model to grade them are out of scope for now.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

Check = Callable[[Any], Any]


@dataclass
class Case:
    """One scenario, and the check every run of it must pass."""

    name: str
    input: Any
    check: Check | Sequence[Check]
    must_pass: bool = False  # can fail the comparison on its own when it clearly drops
    strict: bool = False  # a must pass case that fails the comparison on any single failed run

    def __post_init__(self) -> None:
        if self.strict:
            self.must_pass = True

    @property
    def checks(self) -> list[Check]:
        return list(self.check) if isinstance(self.check, (list, tuple)) else [self.check]  # type: ignore[list-item]


@dataclass
class Suite:
    """All the cases for one agent, and the settings for comparing two models on them.

    agent: called as agent(case.input, model) and returns whatever your checks look at.
    margin: how much lower the new model's pass rate may be and still pass (0.10 is 10 points).
    runs: runs per case per model, taken in two looks of half each.
    prices: {model: (input USD, output USD) per million tokens}, for cost in the report.
    timeout: seconds before a run is stopped. A timeout counts against the model.
    is_provider_error: optional function(exception) returning True for provider outages your
        SDK raises that the built in rules do not know, False to count it as a failure, or None.
    meta: anything worth saving with the result, such as temperature or seed.
    """

    agent: Callable[..., Any]
    cases: Iterable[Case]
    margin: float = 0.10
    runs: int = 6
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)
    concurrency: int = 4
    timeout: float | None = 300.0
    name: str | None = None
    is_provider_error: Callable[[BaseException], bool | None] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cases = list(self.cases)
        names = [c.name for c in self.cases]
        if not names:
            raise ValueError("a suite needs at least one case")
        if len(set(names)) != len(names):
            raise ValueError("case names must be unique")
        if not 0 < self.margin < 1:
            raise ValueError("margin is a share of runs, between 0 and 1, e.g. 0.10 for 10 points")
        if self.runs < 4:
            raise ValueError("runs must be at least 4 (two looks of 2); use smoke mode for a single quick run")
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive, or None")
        self.name = str(self.name or getattr(self.agent, "__name__", "suite"))

    @classmethod
    def from_file(
        cls,
        path: str,
        agent: Callable[..., Any],
        *,
        check: Callable[..., Any] | None = None,
        checks: dict[str, Callable[..., Any]] | None = None,
        **settings: Any,
    ) -> Suite:
        """A suite whose cases come from a JSONL or CSV file (see ai_agent_evaluation.cases).

        A relative path is read next to the file that calls this, so the suite works from any folder."""
        import inspect
        from pathlib import Path

        from .cases import load_cases

        file = Path(path)
        if not file.is_absolute():
            caller = inspect.stack()[1].filename
            beside = Path(caller).resolve().parent / file
            file = beside if beside.exists() else file
        settings.setdefault("name", file.stem)
        return cls(agent=agent, cases=load_cases(file, check=check, checks=checks), **settings)

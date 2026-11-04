"""Compare two models on one suite and decide: pass, regressed, not enough data, or errors.

1. Run every case on both models, half the run budget each, in a shuffled order that alternates
   the two models so that a slow minute at the provider hits both alike.
2. Look 1: settle it early when the answer is already clear. A pass needs every must pass case to
   be clean, because three runs cannot show a must pass case dropping.
3. Otherwise run the second half and look again. After look 2 the verdict is final.

Provider errors are kept out of the score. A case with more than 20% of its runs in error on
either model is left out of the suite answer; more than 20% of all runs in error means no verdict.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import __version__
from .runner import ERROR, FAIL, PASS, Run, run_all
from .stats import LOOK_Z, benjamini_hochberg, fisher_worse, holm, runs_to_settle, suite_difference
from .suite import Case, Suite

PASSED = "pass"
REGRESSED = "regressed"
NOT_ENOUGH_DATA = "not enough data"
ERRORS = "errors"
SMOKE = "smoke"
ERROR_LIMIT = 0.20  # share of runs in provider error that leaves a case, or the whole comparison, unsettled
ERROR_GAP = 0.05  # candidate error rate this far above the baseline's is flagged: the change may cause them
MUST_PASS_ALPHA = 0.05
DIAGNOSTIC_Q = 0.10
DIAGNOSTIC_SHOW = 0.25  # show a case that looks worse when a drop this large happens by chance this rarely

Increments = dict[str, tuple[int, int, int, int, int, int]]  # case: baseline pass, fail, error, candidate pass, fail, error
LookFn = Callable[[int], tuple[Increments, list[Run]]]  # runs per case per model -> what happened


@dataclass
class Tally:
    name: str
    must_pass: bool = False
    strict: bool = False
    b_pass: int = 0
    b_fail: int = 0
    b_err: int = 0
    c_pass: int = 0
    c_fail: int = 0
    c_err: int = 0

    def add(self, inc: tuple[int, int, int, int, int, int]) -> None:
        self.b_pass += inc[0]
        self.b_fail += inc[1]
        self.b_err += inc[2]
        self.c_pass += inc[3]
        self.c_fail += inc[4]
        self.c_err += inc[5]

    @property
    def b_n(self) -> int:
        return self.b_pass + self.b_fail

    @property
    def c_n(self) -> int:
        return self.c_pass + self.c_fail

    @property
    def error_heavy(self) -> bool:
        return self.b_err > ERROR_LIMIT * (self.b_n + self.b_err) or self.c_err > ERROR_LIMIT * (self.c_n + self.c_err)

    @property
    def usable(self) -> bool:
        return not self.error_heavy and self.b_n >= 2 and self.c_n >= 2


@dataclass
class SuiteAnswer:
    status: str  # pass, regressed, or undecided
    diff: float
    lower: float
    upper: float
    se: float
    cases_used: int


@dataclass
class MustPass:
    name: str
    baseline: str
    candidate: str
    failed: bool
    unsettled: bool
    chance: float | None  # one sided Fisher p-value for a drop, when tested


@dataclass
class Diagnostic:
    name: str
    baseline: str
    candidate: str
    chance: float
    likely_worse: bool


@dataclass
class ModelStats:
    model: str
    runs: int
    pass_rate: float | None
    cost_per_run: float | None
    median_seconds: float | None
    input_tokens_per_run: float | None
    output_tokens_per_run: float | None


@dataclass
class SwapResult:
    suite: str
    baseline_model: str
    candidate_model: str
    verdict: str
    reason: str
    margin: float
    runs: int
    seed: int
    look: int
    looks: int = 2
    answer: SuiteAnswer | None = None
    must_pass: list[MustPass] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    tallies: list[Tally] = field(default_factory=list)
    errors: dict[str, Any] = field(default_factory=dict)
    models: list[ModelStats] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    runs_used: int = 0
    runs_budget: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    version: str = __version__

    def exit_code(self, on_inconclusive: str = "block") -> int:
        """0 pass, 1 regressed, 10 not enough data (0 with on_inconclusive="warn"), 11 errors."""
        if self.verdict == REGRESSED:
            return 1
        if self.verdict == ERRORS:
            return 11
        if self.verdict == NOT_ENOUGH_DATA:
            return 0 if on_inconclusive == "warn" else 10
        return 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["case_results"] = d.pop("tallies")
        return d


def decide(tallies: Sequence[Tally], margin: float, z: float) -> SuiteAnswer | None:
    used = [t for t in tallies if t.usable]
    if not used:
        return None
    diff, se = suite_difference([(t.b_pass, t.b_n, t.c_pass, t.c_n) for t in used])
    lower, upper = diff - z * se, diff + z * se
    status = PASSED if lower > -margin else REGRESSED if upper < -margin else "undecided"
    return SuiteAnswer(status, diff, lower, upper, se, len(used))


def run_comparison(
    cases: Sequence[Case], margin: float, runs: int, look_fn: LookFn
) -> tuple[str, str, int, SuiteAnswer | None, list[MustPass], list[Tally], list[Run]]:
    """The decision itself, separate from how runs are made, so the benchmark can drive it."""
    tallies = {c.name: Tally(c.name, c.must_pass, c.strict) for c in cases}
    records: list[Run] = []
    half = runs // 2
    for look, n in ((1, half), (2, runs - half)):
        increments, new_runs = look_fn(n)
        records += new_runs
        for name, inc in increments.items():
            tallies[name].add(inc)
        rows = list(tallies.values())
        answer = decide(rows, margin, LOOK_Z[look - 1])
        must = _must_pass(rows, final=look == 2)
        if look == 1 and not _settled_early(rows, answer, must):
            continue
        verdict, reason = _verdict(rows, answer, must, margin, runs)
        return verdict, reason, look, answer, must, rows, records
    raise AssertionError("unreachable")


def _settled_early(rows: list[Tally], answer: SuiteAnswer | None, must: list[MustPass]) -> bool:
    if any(m.failed for m in must) or (answer is not None and answer.status == REGRESSED):
        return True
    clean = all(t.c_fail == 0 for t in rows if t.must_pass) and not any(t.error_heavy for t in rows)
    return answer is not None and answer.status == PASSED and clean and _error_share(rows) <= ERROR_LIMIT


def _must_pass(rows: list[Tally], final: bool = True) -> list[MustPass]:
    """Strict cases fail on any failure. Others are tested only at the last look, so that looking
    twice does not raise the chance of a false failure."""
    out, tested = [], []
    for t in rows:
        if not t.must_pass:
            continue
        m = MustPass(t.name, f"{t.b_pass}/{t.b_n}", f"{t.c_pass}/{t.c_n}", False, False, None)
        if t.strict:
            m.failed = t.c_fail > 0
        elif t.error_heavy or t.b_n == 0 or t.c_n == 0:
            m.unsettled = True
        elif final:
            m.chance = fisher_worse(t.b_pass, t.b_n, t.c_pass, t.c_n)
            tested.append(m)
        out.append(m)
    for m, reject in zip(tested, holm([m.chance or 1.0 for m in tested], MUST_PASS_ALPHA), strict=True):
        m.failed = reject
    return out


def _error_share(rows: list[Tally]) -> float:
    errors = sum(t.b_err + t.c_err for t in rows)
    total = sum(t.b_n + t.c_n + t.b_err + t.c_err for t in rows)
    return errors / total if total else 0.0


def _verdict(rows: list[Tally], answer: SuiteAnswer | None, must: list[MustPass], margin: float, runs: int) -> tuple[str, str]:
    pts = round(margin * 100)
    failed = [m for m in must if m.failed]
    if failed:
        names = ", ".join(f"{m.name} ({m.baseline} -> {m.candidate})" for m in failed)
        return REGRESSED, f"must pass case{'s' if len(failed) > 1 else ''} dropped: {names}"
    if answer is not None and answer.status == REGRESSED:
        return REGRESSED, f"the new model is at least {-answer.upper * 100:.1f} points worse, beyond your {pts} point margin"
    share = _error_share(rows)
    if share > ERROR_LIMIT:
        return ERRORS, f"{share:.0%} of runs failed at the provider, more than the {ERROR_LIMIT:.0%} limit, so there is no verdict"
    unsettled = [m.name for m in must if m.unsettled]
    if unsettled:
        return NOT_ENOUGH_DATA, f"must pass cases had too many provider errors to check: {', '.join(unsettled)}"
    if answer is None:
        return NOT_ENOUGH_DATA, "no case had enough runs without provider errors"
    if answer.status != PASSED:
        hint = runs_to_settle(answer.diff, answer.se, runs, margin)
        if hint:
            tail = f" For a new comparison, about {hint} runs per case per model would likely settle it."
        else:
            tail = " It looks worse than your margin; a new comparison with more runs would likely confirm a regression."
        return (
            NOT_ENOUGH_DATA,
            f"the new model is {_range(answer.lower, answer.upper)}, so the runs cannot tell whether it is within your {pts} point margin.{tail}",
        )
    if answer.lower >= 0:
        return PASSED, f"the new model is no worse: at least {answer.lower * 100:.1f} points better"
    return PASSED, f"the new model is at most {-answer.lower * 100:.1f} points worse, inside your {pts} point margin"


def _range(lower: float, upper: float) -> str:
    def side(x: float) -> str:
        return f"{abs(x) * 100:.1f} points {'worse' if x < 0 else 'better'}"

    if upper < 0:
        return f"between {abs(upper) * 100:.1f} and {abs(lower) * 100:.1f} points worse"
    return f"somewhere between {side(lower)} and {side(upper)}"


def _diagnostics(rows: list[Tally]) -> list[Diagnostic]:
    """Cases that look worse, for a person to read. Must pass cases are reported on their own, and a
    drop as small as one run in six is left out: it happens by chance more than a quarter of the time."""
    tested = [t for t in rows if t.usable and not t.must_pass]
    chances = [fisher_worse(t.b_pass, t.b_n, t.c_pass, t.c_n) if t.c_pass / t.c_n < t.b_pass / t.b_n else 1.0 for t in tested]
    flags = benjamini_hochberg(chances, DIAGNOSTIC_Q) if tested else []
    out = [
        Diagnostic(t.name, f"{t.b_pass}/{t.b_n}", f"{t.c_pass}/{t.c_n}", p, flag)
        for t, p, flag in zip(tested, chances, flags, strict=True)
        if flag or p <= DIAGNOSTIC_SHOW
    ]
    return sorted(out, key=lambda d: d.chance)


def _model_stats(model: str, records: list[Run], prices: dict[str, tuple[float, float]]) -> ModelStats:
    rs = [r for r in records if r.model == model and r.outcome != ERROR]
    if not rs:
        return ModelStats(model, 0, None, None, None, None, None)
    costs = []
    for r in rs:
        if r.cost_usd is not None:
            costs.append(r.cost_usd)
        elif model in prices and (r.input_tokens or r.output_tokens):
            pin, pout = prices[model]
            costs.append((r.input_tokens * pin + r.output_tokens * pout) / 1_000_000)
    tokens = [r for r in rs if r.input_tokens or r.output_tokens]
    return ModelStats(
        model,
        len(rs),
        sum(r.outcome == PASS for r in rs) / len(rs),
        sum(costs) / len(costs) if len(costs) == len(rs) else None,
        statistics.median(r.latency_s for r in rs),
        sum(r.input_tokens for r in tokens) / len(tokens) if tokens else None,
        sum(r.output_tokens for r in tokens) / len(tokens) if tokens else None,
    )


def _notes(records: list[Run], baseline: str, candidate: str) -> list[str]:
    outputs: dict[tuple[str, str], set[str]] = {}
    seen: Counter[tuple[str, str]] = Counter()
    for r in records:
        if r.outcome != ERROR and r.output_id:
            outputs.setdefault((r.case, r.model), set()).add(r.output_id)
            seen[(r.case, r.model)] += 1
    cases = {case for case, _ in outputs}
    same = [c for c in cases if all(seen[(c, m)] >= 2 and len(outputs.get((c, m), ())) == 1 for m in (baseline, candidate))]
    if cases and len(same) == len(cases):
        return [
            "Every run of every case gave the same output on both models. If a response cache is on, turn it off;"
            " if the agent is fully deterministic, repeated runs add little."
        ]
    return []


def compare_models(
    suite: Suite,
    baseline_model: str,
    candidate_model: str,
    *,
    runs: int | None = None,
    margin: float | None = None,
    seed: int | None = None,
    smoke: bool = False,
    progress: Callable[[str], None] | None = None,
) -> SwapResult:
    """Run the suite on both models and return the verdict, with cost and speed for each."""
    if baseline_model == candidate_model:
        raise ValueError("the baseline and candidate models are the same")
    runs = runs or suite.runs
    margin = suite.margin if margin is None else margin
    if not smoke and runs < 4:
        raise ValueError("runs must be at least 4; use smoke mode for a single quick run")
    seed = random.SystemRandom().randrange(1, 10**6) if seed is None else seed
    rng = random.Random(seed)
    cases: list[Case] = list(suite.cases)

    def look_fn(n: int) -> tuple[Increments, list[Run]]:
        plan = [(c, m) for c in cases for m in (baseline_model, candidate_model) for _ in range(n)]
        rng.shuffle(plan)  # alternate the two models in random order
        if progress:
            progress(f"look {1 if n == runs // 2 or smoke else 2} of 2: {len(plan)} runs")
        records = run_all(suite, plan, progress)
        inc: dict[str, list[int]] = {c.name: [0] * 6 for c in cases}
        for r in records:
            offset = 0 if r.model == baseline_model else 3
            inc[r.case][offset + {PASS: 0, FAIL: 1, ERROR: 2}[r.outcome]] += 1
        return {k: (v[0], v[1], v[2], v[3], v[4], v[5]) for k, v in inc.items()}, records

    name = str(suite.name)
    if smoke:
        increments, records = look_fn(1)
        rows = [Tally(c.name, c.must_pass, c.strict) for c in cases]
        for t in rows:
            t.add(increments[t.name])
        result = SwapResult(name, baseline_model, candidate_model, SMOKE, "one run per case per model: a smoke test, not a verdict", margin, 1, seed, 1, 1)
        verdict_parts: tuple[str, str, int, SuiteAnswer | None, list[MustPass], list[Tally], list[Run]] = (SMOKE, result.reason, 1, None, [], rows, records)
    else:
        verdict_parts = run_comparison(cases, margin, runs, look_fn)
        result = SwapResult(name, baseline_model, candidate_model, verdict_parts[0], verdict_parts[1], margin, runs, seed, verdict_parts[2])
    _, _, _, answer, must, rows, records = verdict_parts
    result.answer, result.must_pass, result.tallies = answer, must, rows
    result.diagnostics = [] if smoke else _diagnostics(rows)
    result.models = [_model_stats(m, records, suite.prices) for m in (baseline_model, candidate_model)]
    b_runs = sum(t.b_n + t.b_err for t in rows)
    c_runs = sum(t.c_n + t.c_err for t in rows)
    b_err, c_err = sum(t.b_err for t in rows), sum(t.c_err for t in rows)
    reasons = Counter(r.detail for r in records if r.outcome == ERROR)
    result.errors = {
        "baseline": b_err,
        "candidate": c_err,
        "share": (b_err + c_err) / (b_runs + c_runs) if b_runs + c_runs else 0.0,
        "reasons": dict(reasons.most_common()),
        "candidate_rate_higher": c_runs > 0 and b_runs > 0 and c_err / c_runs - b_err / b_runs > ERROR_GAP,
    }
    result.notes = _notes(records, baseline_model, candidate_model)
    result.runs_used = b_runs + c_runs
    result.runs_budget = len(cases) * 2 * (1 if smoke else runs)
    result.meta = dict(suite.meta)
    return result

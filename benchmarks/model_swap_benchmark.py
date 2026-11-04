"""How often the verdict is wrong, against today's habit of running each case once.

Simulated model swaps only: every run of a case passes with a fixed probability, which differs
between the two models by a known amount. The verdict comes from the library's own decision code
(ai_agent_evaluation.compare.run_comparison), with its defaults: 6 runs per case per model in two
looks, and a 10 point margin.

Today's habit, as a rule: run each case once on each model, and call it a regression if any case
that passed on the current model fails on the new one.

    python benchmarks/model_swap_benchmark.py              # 2,000 swaps per row, a few minutes
    python benchmarks/model_swap_benchmark.py --sims 200   # quick
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_agent_evaluation import Case  # noqa: E402
from ai_agent_evaluation.compare import PASSED, REGRESSED, run_comparison  # noqa: E402

MARGIN, RUNS, SEED = 0.10, 6, 2026

SUITES = {
    "40 cases, mixed (20 at 100%, 10 at 95%, 6 at 85%, 4 at 60%)": [1.0] * 20 + [0.95] * 10 + [0.85] * 6 + [0.6] * 4,
    "30 cases, all at 97% (like the XPath example)": [0.97] * 30,
    "40 cases, all at 70% (a flaky agent)": [0.7] * 40,
    "10 cases, mixed": [1.0] * 5 + [0.95] * 2 + [0.85] * 2 + [0.6],
}


def worse_spread(base: list[float], points: float) -> list[float]:
    return [max(0.0, p - points) for p in base]


def worse_in_few(base: list[float], points: float) -> list[float]:
    """The same total drop, all in the strongest cases going to zero."""
    out, left = list(base), points * len(base)
    for i in sorted(range(len(base)), key=lambda i: -base[i]):
        if left <= 1e-9:
            break
        cut = min(out[i], left)
        out[i] -= cut
        left -= cut
    return out


def one_swap(base: list[float], new: list[float], rng: random.Random, error_rate: float = 0.0, must_pass: int = -1) -> tuple[str, int]:
    cases = [Case(str(i), None, check=bool, must_pass=(i == must_pass)) for i in range(len(base))]
    used = 0

    def look(n: int):
        nonlocal used
        inc = {}
        for i, (pb, pn) in enumerate(zip(base, new, strict=True)):
            row = [0] * 6
            for side, p in ((0, pb), (3, pn)):
                for _ in range(n):
                    if rng.random() < error_rate:
                        row[side + 2] += 1
                    elif rng.random() < p:
                        row[side] += 1
                    else:
                        row[side + 1] += 1
            inc[str(i)] = tuple(row)
            used += 2 * n
        return inc, []

    verdict = run_comparison(cases, MARGIN, RUNS, look)[0]
    return verdict, used


def run_once_rule(base: list[float], new: list[float], rng: random.Random) -> str:
    return REGRESSED if any(rng.random() < pb and rng.random() >= pn for pb, pn in zip(base, new, strict=True)) else PASSED


def rates(base, new, sims, seed, **kw):
    rng = random.Random(seed)
    counts = {PASSED: 0, REGRESSED: 0, "other": 0}
    runs = 0
    habit = {PASSED: 0, REGRESSED: 0}
    for _ in range(sims):
        verdict, used = one_swap(base, new, rng, **kw)
        counts[verdict if verdict in counts else "other"] += 1
        runs += used
        habit[run_once_rule(base, new, rng)] += 1
    return {k: v / sims for k, v in counts.items()}, runs / sims, {k: v / sims for k, v in habit.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    args = ap.parse_args()
    print(f"Margin {MARGIN:.0%}, {RUNS} runs per case per model in two looks, {args.sims} simulated swaps per row, seed {SEED}.\n")
    seed = SEED
    for name, base in SUITES.items():
        print(f"## {name}\n")
        print("| The new model really is | Pass | Regressed | Not enough data | Runs used | Run once rule: regressed |")
        print("|---|---|---|---|---|---|")
        truths = [
            ("the same", base, {}),
            ("the same, with 3% provider errors", base, {"error_rate": 0.03}),
            ("10 points worse, spread over all cases", worse_spread(base, MARGIN), {}),
            ("10 points worse, all in a few cases", worse_in_few(base, MARGIN), {}),
            ("20 points worse", worse_spread(base, 0.20), {}),
            ("5 points better", [min(1.0, p + 0.05) for p in base], {}),
        ]
        for label, new, kw in truths:
            seed += 1
            ours, runs, habit = rates(base, new, args.sims, seed, **kw)
            print(f"| {label} | {ours[PASSED]:.1%} | {ours[REGRESSED]:.1%} | {ours['other']:.1%} | {runs:.0f} | {habit[REGRESSED]:.1%} |")
        print()
    base = SUITES[next(iter(SUITES))]
    broken = list(base)
    broken[0] = 0.0
    seed += 1
    ours, _, habit = rates(base, broken, args.sims, seed, must_pass=0)
    print(f"One must pass case breaks (100% to 0%), the rest unchanged, 40 cases: caught {ours[REGRESSED]:.1%}; run once rule {habit[REGRESSED]:.1%}.")


if __name__ == "__main__":
    main()

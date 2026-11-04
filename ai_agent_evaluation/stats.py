"""The statistics behind a verdict.

The question is whether the new model is at most `margin` worse than the current one (a
non-inferiority test), not whether it is better. Every case is run on both models, and the suite
difference is the average over cases of (new pass rate minus current pass rate), so each case is
compared with itself and a big case cannot hide a small one.

The variance of each case's rate uses lightly smoothed rates, (passes + 0.5) / (runs + 1), divided by
runs minus one. Plain rates say a case that passed 3 of 3 has no uncertainty at all, and dividing by
runs understates the spread of small samples; both make the check pass models that are worse. In our
simulations this choice kept misses under 5% where the plain formula missed 8% to 11%.

The results are read twice, at half and at the full budget, with O'Brien and Fleming's boundaries,
so that looking twice does not raise the miss rate above 5%.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from math import comb

LOOK_Z = (2.373, 1.678)  # one sided 5%, two equally spaced looks (O'Brien-Fleming)


def suite_difference(counts: Sequence[tuple[int, int, int, int]]) -> tuple[float, float]:
    """Mean over cases of (candidate rate - baseline rate), and its standard error.

    counts: one (baseline passes, baseline runs, candidate passes, candidate runs) per case, with
    at least 2 runs on each side.
    """
    if not counts:
        raise ValueError("no cases to compare")
    total, var = 0.0, 0.0
    for b_pass, b_n, c_pass, c_n in counts:
        total += c_pass / c_n - b_pass / b_n
        for x, n in ((b_pass, b_n), (c_pass, c_n)):
            p = (x + 0.5) / (n + 1)
            var += p * (1 - p) / max(n - 1, 1)
    k = len(counts)
    return total / k, math.sqrt(var) / k


def fisher_worse(base_pass: int, base_n: int, passes: int, n: int) -> float:
    """One sided Fisher exact test: the chance of a drop at least this large if nothing changed."""
    total, passed = base_n + n, base_pass + passes
    top = min(base_n, passed)
    hits = sum(comb(passed, x) * comb(total - passed, base_n - x) for x in range(base_pass, top + 1))
    return hits / comb(total, base_n)


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Which hypotheses to reject, keeping the chance of any false rejection at alpha."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    reject = [False] * len(pvalues)
    for rank, i in enumerate(order):
        if pvalues[i] > alpha / (len(pvalues) - rank):
            break
        reject[i] = True
    return reject


def benjamini_hochberg(pvalues: Sequence[float], q: float = 0.10) -> list[bool]:
    """Which hypotheses to flag, keeping the expected share of false flags at q."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    cutoff = 0
    for rank, i in enumerate(order, start=1):
        if pvalues[i] <= q * rank / m:
            cutoff = rank
    flagged = set(order[:cutoff])
    return [i in flagged for i in range(m)]


def runs_to_settle(diff: float, se: float, runs: int, margin: float) -> int | None:
    """Rough runs per case per model that would likely settle a comparison like this one."""
    room = diff + margin
    if room <= 0 or se <= 0:
        return None
    needed = runs * (se * LOOK_Z[1] / room) ** 2 * 1.2  # 20% headroom for the noise in the estimate
    return max(runs + 2, 2 * math.ceil(needed / 2))

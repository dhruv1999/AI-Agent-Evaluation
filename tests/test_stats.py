import random

import pytest

from ai_agent_evaluation.stats import LOOK_Z, benjamini_hochberg, fisher_worse, holm, runs_to_settle, suite_difference


def test_fisher_exact_values():
    assert fisher_worse(3, 3, 0, 3) == pytest.approx(1 / 20)
    assert fisher_worse(6, 6, 0, 6) == pytest.approx(1 / 924)
    assert fisher_worse(6, 6, 6, 6) == pytest.approx(1.0)


def test_suite_difference_is_per_case_and_never_certain():
    diff, se = suite_difference([(3, 3, 3, 3)] * 10)
    assert diff == 0 and se > 0  # 3 of 3 is not proof of a perfect agent
    diff, _ = suite_difference([(6, 6, 0, 6)] + [(6, 6, 6, 6)] * 49)
    assert diff == pytest.approx(-1 / 50)  # one broken case in 50 moves the suite by 2 points
    with pytest.raises(ValueError):
        suite_difference([])


def test_holm_and_benjamini_hochberg():
    assert holm([0.01, 0.06], 0.05) == [True, False]
    assert holm([0.03, 0.04], 0.05) == [False, False]
    assert holm([0.01, 0.04], 0.05) == [True, True]
    assert benjamini_hochberg([0.01, 0.04, 0.5], 0.10) == [True, True, False]
    assert benjamini_hochberg([], 0.10) == []


def test_two_look_boundaries_keep_the_error_at_five_percent():
    rng = random.Random(1)
    hits = 0
    sims = 40000
    for _ in range(sims):
        a, b = rng.gauss(0, 1), rng.gauss(0, 1)
        z1, z2 = a, (a + b) / 2**0.5
        hits += z1 > LOOK_Z[0] or z2 > LOOK_Z[1]
    assert 0.045 < hits / sims < 0.055


def test_runs_to_settle():
    assert runs_to_settle(-0.12, 0.05, 6, 0.10) is None  # already past the margin: more runs will not help it pass
    assert runs_to_settle(-0.02, 0.05, 6, 0.10) >= 8

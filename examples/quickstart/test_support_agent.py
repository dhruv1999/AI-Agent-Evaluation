"""The same suite under pytest.

AI_AGENT_EVALUATION_UPDATE=1 pytest examples/quickstart     # save or refresh the baseline
pytest examples/quickstart                            # fails if a case regressed
"""

from suite import suite

from ai_agent_evaluation import assert_no_regressions


def test_support_agent():
    assert_no_regressions(suite)

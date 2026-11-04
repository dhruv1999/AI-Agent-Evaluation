"""The same comparison under pytest:   pytest examples/xpath_healing"""

from suite import suite

from ai_agent_evaluation import assert_model_swap


def test_switch_to_model_new():
    assert_model_swap(suite, "model-old", "model-new")

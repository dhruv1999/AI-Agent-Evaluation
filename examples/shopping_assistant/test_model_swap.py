"""The same comparison under pytest:   pytest examples/shopping_assistant"""

from suite import suite

from ai_agent_evaluation import assert_model_swap


def test_switch_to_small_model():
    assert_model_swap(suite, "large-model", "small-model")

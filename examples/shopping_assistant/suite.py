"""Should the shopping assistant move to a cheaper model?

    ai-agent-evaluation compare examples/shopping_assistant/suite.py --baseline-model large-model --candidate-model small-model

The cases live in cases.jsonl, one shopper request per line with the answer it should get.
"""

from agent import shopping_agent

from ai_agent_evaluation import Suite

suite = Suite.from_file(
    "cases.jsonl",
    agent=shopping_agent,
    name="shopping_assistant",
    prices={"large-model": (2.00, 8.00), "small-model": (0.40, 1.60), "tiny-model": (0.10, 0.40)},  # USD per million tokens
)

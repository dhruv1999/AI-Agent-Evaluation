"""Google ADK. Needs GOOGLE_API_KEY (or Vertex AI settings).   ai-agent-evaluation run examples/frameworks/google_adk_suite.py"""

from google.adk.agents import LlmAgent

from ai_agent_evaluation import Case, Suite, expect
from ai_agent_evaluation.integrations.google_adk import wrap


def lookup_order(order_id: str) -> str:
    """Look up an order's status and amount."""
    return {"42": "shipped, $40", "77": "delivered, $900"}.get(order_id, "not found")


agent = LlmAgent(name="support", model="gemini-flash-latest", instruction="You help with orders.", tools=[lookup_order])

suite = Suite(
    agent=wrap(agent),
    cases=[Case("order status", "Where is order 42?", [expect.calls("lookup_order", order_id="42"), expect.output_contains("shipped")])],
)

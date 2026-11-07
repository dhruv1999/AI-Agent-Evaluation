"""AI Agent Evaluation: should your agent switch models? Get an honest answer before you switch.

    from ai_agent_evaluation import Case, Suite

    suite = Suite(agent=heal_xpath, cases=[Case("login button", page, check=finds_login_button)])

Then compare two models on it:

    ai-agent-evaluation compare suite.py --baseline-model model-old --candidate-model model-new

The answer is pass, regressed or not enough data, with cost and speed for each model.
"""

__version__ = "0.1.0"

from .cases import contains_expected, equals_expected, load_cases, matches_expected  # noqa: E402
from .compare import SwapResult, compare_models  # noqa: E402
from .recording import record_response, record_usage  # noqa: E402
from .suite import Case, Suite  # noqa: E402
from .testing import assert_model_swap  # noqa: E402

__all__ = [
    "Case",
    "Suite",
    "SwapResult",
    "assert_model_swap",
    "compare_models",
    "contains_expected",
    "equals_expected",
    "load_cases",
    "matches_expected",
    "record_response",
    "record_usage",
]

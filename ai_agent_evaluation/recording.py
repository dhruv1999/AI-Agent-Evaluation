"""Token usage, so the report can show what each model costs.

Call record_response(response) after each model call inside your agent, or record_usage(...) with
your own numbers. Outside a comparison these do nothing, so they can stay in production code.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


_current: ContextVar[Usage | None] = ContextVar("ai_agent_evaluation_usage", default=None)


def record_usage(input_tokens: int = 0, output_tokens: int = 0, cost_usd: float | None = None) -> None:
    usage = _current.get()
    if usage is None:
        return
    usage.input_tokens += int(input_tokens)
    usage.output_tokens += int(output_tokens)
    if cost_usd is not None:
        usage.cost_usd = (usage.cost_usd or 0.0) + cost_usd


def record_response(response: Any) -> None:
    """Read token usage from an OpenAI, Anthropic or Gemini SDK response."""
    usage = getattr(response, "usage", None)
    if usage is not None:
        inp = getattr(usage, "input_tokens", None)  # Anthropic, OpenAI Responses API
        if inp is None:
            inp = getattr(usage, "prompt_tokens", 0)  # OpenAI Chat Completions
        inp = (inp or 0) + (getattr(usage, "cache_read_input_tokens", 0) or 0) + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        out = getattr(usage, "output_tokens", None)
        if out is None:
            out = getattr(usage, "completion_tokens", 0)
        record_usage(inp, out or 0)
        return
    meta = getattr(response, "usage_metadata", None)  # Gemini
    if meta is not None:
        out = (getattr(meta, "candidates_token_count", 0) or 0) + (getattr(meta, "thoughts_token_count", 0) or 0)
        record_usage(getattr(meta, "prompt_token_count", 0) or 0, out)

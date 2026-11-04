"""Which failures are the provider's, not the agent's.

A run that dies because the model provider was busy or unreachable says nothing about the model's
quality, so it is marked ERROR and kept out of the score. Only these count, after the provider SDK's
own retries: HTTP 429, 500, 502, 503, 504 and 529, and dropped connections, from the OpenAI and
Anthropic Python SDKs (also when a framework wraps them). Context length errors, invalid tool calls,
and the agent's own timeouts and step limits count against the model: a model change can cause them.
"""

from __future__ import annotations

from collections.abc import Callable

PROVIDER_STATUS = {429, 500, 502, 503, 504, 529}
SDKS = ("openai", "anthropic")
CONNECTION_ERRORS = {"APIConnectionError", "APITimeoutError"}


def _chain(error: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    e: BaseException | None = error
    while e is not None and len(seen) < 6 and e not in seen:
        seen.append(e)
        e = e.__cause__ or e.__context__
    return seen


def provider_error(error: BaseException, hook: Callable[[BaseException], bool | None] | None = None) -> str | None:
    """A short reason such as "HTTP 429" when this is a provider outage, else None."""
    if hook is not None:
        decided = hook(error)
        if decided is not None:
            return _reason(error) if decided else None
    for e in _chain(error):
        if type(e).__module__.split(".")[0] not in SDKS:
            continue
        status = getattr(e, "status_code", None)
        if isinstance(status, int) and status in PROVIDER_STATUS:
            return f"HTTP {status}"
        if CONNECTION_ERRORS & {c.__name__ for c in type(e).__mro__}:
            return "connection error"
    return None


def _reason(error: BaseException) -> str:
    status = getattr(error, "status_code", None)
    return f"HTTP {status}" if isinstance(status, int) else type(error).__name__

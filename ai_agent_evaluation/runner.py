"""Runs the agent: one call per (case, model), judged as PASS, FAIL or ERROR."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import inspect
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import provider_error
from .recording import Usage, _current
from .suite import Case, Suite

PASS, FAIL, ERROR = "pass", "fail", "error"


@dataclass
class Run:
    case: str
    model: str
    outcome: str  # PASS, FAIL or ERROR
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    detail: str = ""  # why it failed, or which provider error
    output_id: str = ""  # a short hash of the output, to notice runs that are all identical
    llm_calls: int = 0
    reasoning_tokens: int = 0


def run_all(suite: Suite, plan: list[tuple[Case, str]], progress: Callable[[str], None] | None = None) -> list[Run]:
    """Run every (case, model) in the plan, in order, `suite.concurrency` at a time."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run_all(suite, plan, progress))
    raise RuntimeError("call compare_models outside a running event loop")


async def _run_all(suite: Suite, plan: list[tuple[Case, str]], progress: Callable[[str], None] | None) -> list[Run]:
    gate = asyncio.Semaphore(suite.concurrency)
    done = 0

    async def one(case: Case, model: str) -> Run:
        nonlocal done
        async with gate:
            run = await _run_one(suite, case, model)
        done += 1
        if progress and len(plan) > 200 and done % 200 == 0:
            progress(f"{done}/{len(plan)} runs")
        return run

    return list(await asyncio.gather(*(one(c, m) for c, m in plan)))


async def _run_one(suite: Suite, case: Case, model: str) -> Run:
    usage = Usage()
    start = time.perf_counter()
    task = asyncio.ensure_future(_call(suite.agent, case.input, model, usage))
    finished, _ = await asyncio.wait({task}, timeout=suite.timeout)
    latency = time.perf_counter() - start

    def made(outcome: str, detail: str = "", output_id: str = "") -> Run:
        return Run(
            case.name,
            model,
            outcome,
            round(latency, 4),
            usage.input_tokens,
            usage.output_tokens,
            usage.cost_usd,
            detail[:300],
            output_id,
            usage.llm_calls,
            usage.reasoning_tokens,
        )

    if not finished:
        task.cancel()  # a sync agent's thread cannot be stopped; it is a daemon, so it never blocks exit
        return made(FAIL, f"timed out after {suite.timeout:g}s")
    error = task.exception()
    if error is not None:
        reason = provider_error(error, suite.is_provider_error)
        if reason:
            return made(ERROR, reason)
        return made(FAIL, f"agent raised {type(error).__name__}: {error}")
    output = task.result()
    output_id = hashlib.sha1(repr(output).encode("utf-8", "replace")).hexdigest()[:10]
    for check in case.checks:
        try:
            verdict = check(output)
        except AssertionError as e:
            return made(FAIL, str(e) or f"{getattr(check, '__name__', 'check')} failed", output_id)
        except Exception as e:
            return made(FAIL, f"check raised {type(e).__name__}: {e}", output_id)
        if verdict is False:
            return made(FAIL, f"{getattr(check, '__name__', 'check')} returned False", output_id)
    return made(PASS, "", output_id)


async def _call(agent: Callable[..., Any], input: Any, model: str, usage: Usage) -> Any:
    _current.set(usage)  # this runs in its own task, so it does not leak into other runs
    if inspect.iscoroutinefunction(agent) or inspect.iscoroutinefunction(type(agent).__call__):
        out = agent(input, model)
    else:
        out = await _in_thread(functools.partial(agent, input, model))
    return await out if inspect.isawaitable(out) else out


def _in_thread(fn: Callable[[], Any]) -> asyncio.Future[Any]:
    """Run a sync agent in a daemon thread that sees this task's context (so usage is recorded)."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()
    ctx = contextvars.copy_context()

    def settle(value: Any, error: BaseException | None) -> None:
        if not future.done():
            future.set_exception(error) if error is not None else future.set_result(value)

    def target() -> None:
        value, error = None, None
        try:
            value = ctx.run(fn)
        except Exception as e:
            error = e
        except BaseException as e:  # SystemExit and the like: a failed run, not a dead comparison
            error = RuntimeError(f"agent raised {type(e).__name__}")
        try:
            loop.call_soon_threadsafe(settle, value, error)
        except RuntimeError:  # the event loop has closed: this run was abandoned after a timeout
            pass

    threading.Thread(target=target, name="ai-agent-evaluation-run", daemon=True).start()
    return future

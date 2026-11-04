"""A stand in for an XPath healing agent, so the example runs without an API key.

When a web page changes, a test's XPath selector breaks. The agent reads the new page and proposes a
healed selector. Here the "model" is a dice roll with a different accuracy per model; a real agent
would send the page and the broken selector to a language model.

Set DEMO_SEED to make the dice repeatable.
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections import Counter

from ai_agent_evaluation import record_usage

ACCURACY = {"model-old": 0.97, "model-new": 0.95, "model-mini": 0.85}
LATENCY_S = {"model-old": 0.020, "model-new": 0.012, "model-mini": 0.008}
WEAK_SPOTS = {"model-mini": {"submit-claim", "pay"}}  # the small model mostly misses these buttons

_calls: Counter[tuple[str, str]] = Counter()
_lock = threading.Lock()


def _dice(element: str, model: str) -> random.Random:
    seed = os.environ.get("DEMO_SEED")
    if seed is None:
        return random.Random()
    with _lock:  # the same draws whatever order the runs happen in
        _calls[(element, model)] += 1
        n = _calls[(element, model)]
    return random.Random(f"{seed}:{element}:{model}:{n}")


def heal_xpath(case: dict, model: str) -> str:
    """Return a healed XPath for the element the broken selector used to find."""
    if model not in ACCURACY:
        raise ValueError(f"unknown model {model!r}; try one of {sorted(ACCURACY)}")
    dice = _dice(case["element"], model)
    time.sleep(LATENCY_S[model] * (0.5 + dice.random()))
    record_usage(input_tokens=len(case["html"]) // 4 + 120, output_tokens=25)
    accuracy = 0.2 if case["element"] in WEAK_SPOTS.get(model, ()) else ACCURACY[model]
    tag, element = case["tag"], case["element"]
    if dice.random() < accuracy:
        return f".//{tag}[@data-qa='{element}']"
    return dice.choice([f".//{tag}", f".//{tag}[1]", f".//{tag}[@id='{element}']"])  # plausible but wrong

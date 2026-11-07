"""A shopping assistant: it reads a shopper's request, searches the catalog, and recommends one
product, asks a question, or says nothing fits.

It answers with JSON: {"action": "recommend" | "ask" | "none", "product_id": ..., "message": ...}

Two ways to run it:
- Live: set OPENAI_API_KEY and SHOP_LIVE=1, and pass real OpenAI model names. The model calls the
  search_products tool through the OpenAI Python SDK.
- Simulated (default): no key needed. Models named large-model, small-model and tiny-model are
  stand ins with different accuracy, so the example runs anywhere. Set DEMO_SEED to repeat a run.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from collections import Counter
from pathlib import Path

from ai_agent_evaluation import record_response, record_usage

CATALOG = json.loads((Path(__file__).parent / "catalog.json").read_text(encoding="utf-8"))
SIZED = {"running shoes", "rain jackets"}
CATEGORIES = sorted({p["category"] for p in CATALOG}, key=len, reverse=True)


def search_products(category: str, max_price: float | None = None, size: str | None = None) -> list[dict]:
    """In stock products in a category, cheapest first, within budget and size."""
    hits = [
        p
        for p in CATALOG
        if p["category"] == category and p["in_stock"] and (max_price is None or p["price"] <= max_price) and (size is None or size in p.get("sizes", []))
    ]
    return sorted(hits, key=lambda p: p["price"])


def shopping_agent(request: str, model: str) -> dict:
    if os.environ.get("SHOP_LIVE") == "1":
        return _live(request, model)
    return _simulated(request, model)


# ---- live: an OpenAI model with one tool ---------------------------------------------------------

SYSTEM = (
    "You are a shopping assistant for a small online store. Use search_products to find items. "
    "Recommend the cheapest in stock product that fits the request. For running shoes and rain jackets, "
    "if the shopper gave no size, ask for it instead of recommending. If nothing fits, say so. "
    'Reply with JSON only: {"action": "recommend" | "ask" | "none", "product_id": string or null, "message": string}.'
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search in stock products, cheapest first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": CATEGORIES},
                    "max_price": {"type": ["number", "null"]},
                    "size": {"type": ["string", "null"]},
                },
                "required": ["category"],
            },
        },
    }
]


def _live(request: str, model: str, client=None) -> dict:
    if client is None:
        from openai import OpenAI

        client = OpenAI()
    messages: list = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": request}]
    for _ in range(5):
        response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS)
        record_response(response)
        message = response.choices[0].message
        if not message.tool_calls:
            return json.loads(message.content)
        messages.append(message)
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            result = search_products(args["category"], args.get("max_price"), args.get("size"))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    return {"action": "none", "product_id": None, "message": "gave up after 5 steps"}


# ---- simulated: the same task with a dice roll for mistakes --------------------------------------

ACCURACY = {"large-model": 0.97, "small-model": 0.95, "tiny-model": 0.88}
LATENCY_S = {"large-model": 0.020, "small-model": 0.010, "tiny-model": 0.006}
_calls: Counter = Counter()
_lock = threading.Lock()


def _dice(request: str, model: str) -> random.Random:
    seed = os.environ.get("DEMO_SEED")
    if seed is None:
        return random.Random()
    with _lock:
        _calls[(request, model)] += 1
        return random.Random(f"{seed}:{request}:{model}:{_calls[(request, model)]}")


def understand(request: str) -> tuple[str | None, float | None, str | None]:
    text = request.lower()
    category = next((c for c in CATEGORIES if c in text), None)
    price = re.search(r"under \$(\d+)", text)
    size = re.search(r"size (\w+)", request)
    return category, float(price.group(1)) if price else None, size.group(1) if size else None


def correct_answer(request: str) -> dict:
    category, max_price, size = understand(request)
    if category in SIZED and size is None:
        return {"action": "ask", "product_id": None, "message": "What size do you need?"}
    hits = search_products(category, max_price, size) if category else []
    if not hits:
        return {"action": "none", "product_id": None, "message": "Nothing in stock fits that."}
    return {"action": "recommend", "product_id": hits[0]["id"], "message": f"Try the {hits[0]['name']} at ${hits[0]['price']}."}


def _simulated(request: str, model: str) -> dict:
    if model not in ACCURACY:
        raise ValueError(f"unknown simulated model {model!r}; use {sorted(ACCURACY)}, or SHOP_LIVE=1 with real models")
    dice = _dice(request, model)
    time.sleep(LATENCY_S[model] * (0.5 + dice.random()))
    record_usage(input_tokens=900, output_tokens=60)
    answer = correct_answer(request)
    category, max_price, _ = understand(request)
    if model == "tiny-model" and max_price is not None and dice.random() < 0.7:
        pricier = [p for p in CATALOG if p["category"] == category and p["in_stock"] and p["price"] > max_price]
        if pricier:  # the tiny model often ignores the budget
            return {"action": "recommend", "product_id": pricier[0]["id"], "message": "This one is great."}
    if dice.random() < ACCURACY[model]:
        return answer
    others = [p["id"] for p in CATALOG if p["category"] == category and p["id"] != answer.get("product_id")] or [CATALOG[0]["id"]]
    return {"action": "recommend", "product_id": dice.choice(others), "message": "This one is great."}

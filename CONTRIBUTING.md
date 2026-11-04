# Contributing

Thanks for helping. Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/dhruv1999/AI-Agent-Evaluation && cd AI-Agent-Evaluation
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

Without uv: `pip install -e . pytest ruff mypy`.

Changes to how verdicts are made must keep the numbers in the docs true. Check with:

```bash
uv run python benchmarks/model_swap_benchmark.py
```

## Guidelines

- No required dependencies. The core runs on the standard library.
- A check that cannot decide must fail; it never passes silently.
- Provider errors are only what a provider outage causes. Anything a model change can cause counts against the model.
- Keep reports plain and short.

# AI Agent Evaluation

[![CI](https://github.com/dhruv1999/AI-Agent-Evaluation/actions/workflows/ci.yml/badge.svg)](https://github.com/dhruv1999/AI-Agent-Evaluation/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

**Should your agent switch models? Find out in an afternoon, not a day.**

Run your agent's test cases on the model you use today and the one you are considering. You get one honest answer, **pass**, **regressed** or **not enough data**, with the cost and speed of each model next to it.

```text
Model swap: model-old -> model-new   (suite xpath_healing, 30 cases)

PASS. The new model is at most 4.1 points worse, inside your 10 point margin.
Changes up to 10 points worse are accepted; change this with --margin.

                 model-old   model-new
pass rate            95.6%       97.2%
cost per run       $0.0010     $0.0003   73% lower
median time          0.02s       0.01s   34% faster

Must pass:
  claims: submit-claim: 6/6 -> 6/6  ok
  payment: pay: 6/6 -> 6/6  ok
```

## The problem

A newer or cheaper model comes out, or the old one is retired. Before switching, someone has to rerun every test case on the new model and compare the results by hand.

- **It is slow.** A day of reruns and side by side reading for one decision.
- **One run per case proves little.** Agents answer differently every time. In our simulations, comparing one run per case flagged a regression in 48% to 100% of swaps where nothing had changed.
- **Nobody can say what "about the same" means.** Does it include the payments flow? Is the new model 3 points worse or 15?
- **Cost and speed sit somewhere else.** They are usually the reason to switch, but they are not on the same page as quality.

## Who it is for

- **Developers on small teams** whose agent already has test cases, or a script that runs it a few times.
- **Agents whose results code can check:** the healed XPath finds the right element, the right tool was called, the output parses. Answers that need a person or a model to grade them are not supported yet.

## What it does

- **Runs both models on your cases**, 6 runs per case per model, shuffled so a slow minute at the provider hits both alike.
- **Gives one answer for the suite:** pass if the new model is at most your margin worse (10 points by default), regressed if it is clearly worse, not enough data if the runs cannot tell. *(non-inferiority test)*
- **Stops early when the answer is already clear,** after half the runs. *(two looks)*
- **Never lets a critical case hide in the average.** Mark it must pass and it can fail the comparison on its own.
- **Keeps provider outages out of the score.** Rate limits and server errors are counted separately, not blamed on the model.
- **Puts cost and speed next to quality,** from token counts and the prices you enter.
- **Works where you test:** a command, a pytest helper, and a GitHub workflow you can copy.

## Quickstart

No API key needed. The example is an XPath healing agent with simulated models.

```bash
git clone https://github.com/dhruv1999/AI-Agent-Evaluation && cd AI-Agent-Evaluation
pip install -e .
ai-agent-evaluation compare examples/xpath_healing/suite.py --baseline-model model-old --candidate-model model-new
ai-agent-evaluation compare examples/xpath_healing/suite.py --baseline-model model-old --candidate-model model-mini
```

The first prints the result at the top of this page. The second compares a much cheaper model that mostly misses two critical buttons:

```text
REGRESSED. Must pass cases dropped: claims: submit-claim (6/6 -> 2/6), payment: pay (6/6 -> 1/6).

                  model-old   model-mini
pass rate             95.6%        84.4%
cost per run        $0.0010      $0.0001   92% lower
```

Each comparison also writes `eval-results/eval-report.md` and `eval-results/eval-verdict.json`.

To use it in your own project:

```bash
pip install git+https://github.com/dhruv1999/AI-Agent-Evaluation
```

## Your own suite

A suite is one Python file: your agent as a function of the case input and the model name, and the cases with a check each.

```python
from ai_agent_evaluation import Case, Suite, record_response

def agent(case, model):
    response = client.chat.completions.create(model=model, messages=build_messages(case))
    record_response(response)                     # tokens, for the cost column
    return parse(response)

def finds_submit_button(xpath):                   # return False, or raise AssertionError, to fail a run
    assert matches_exactly(page, xpath, "submit")

suite = Suite(
    agent=agent,
    cases=[
        Case("submit button", page, check=finds_submit_button, must_pass=True),
        # ... more cases
    ],
    prices={"gpt-current": (2.50, 10.00), "gpt-cheaper": (0.40, 1.60)},   # USD per million tokens
)
```

```bash
ai-agent-evaluation compare agent_suite.py --baseline-model gpt-current --candidate-model gpt-cheaper
```

Start with `--smoke` for a single quick run per case. It checks your setup and is labelled as not a verdict.

## Results and exit codes

| Answer | Meaning | Exit code |
|---|---|---|
| **pass** | The new model is at most your margin worse | 0 |
| **regressed** | Clearly worse than your margin, or a must pass case clearly dropped | 1 |
| **not enough data** | The runs could not tell. The report suggests how many runs would | 10, or 0 with `--on-inconclusive warn` |
| **errors** | More than 20% of runs failed at the provider, so there is no verdict | 11 |

In pytest:

```python
from ai_agent_evaluation import assert_model_swap
from agent_suite import suite

def test_switch_to_cheaper_model():
    assert_model_swap(suite, "gpt-current", "gpt-cheaper")
```

## How well it works

From 2,000 simulated model swaps per row, with the defaults (6 runs per case per model, 10 point margin), on a 40 case suite with a mix of stable and flaky cases. Full tables: [docs/how-it-decides.md](docs/how-it-decides.md).

| The new model really is | This library | Running each case once |
|---|---|---|
| The same | passes 99.0%, never says regressed | says regressed 91.6% of the time |
| 10 points worse (exactly the margin) | passes only 2.1% | says regressed 99.9% |
| 20 points worse | says regressed 88.3%, never passes | says regressed 100% |
| A must pass case breaks, the rest the same | fails it 100% | flags it 100%, along with most unchanged swaps |

**The honest limits:**
- **Small suites need more runs.** With 10 cases, 6 runs settle only about half of unchanged swaps. Use `--runs 16`.
- **Very flaky agents need more runs too.** Where cases pass about 70% of the time, 27% of unchanged swaps end as not enough data.
- **Judgement calls are out of scope.** Agents whose answers need an LLM judge are not supported in this version.

## Roadmap

- Run on every pull request, with a comment on the pull request and a cached baseline.
- Retries that wait out rate limits, and stopping early during an outage.
- Answers graded by an LLM judge, with a check for judge drift.
- Reports across more than two models.

## Docs

- [How it decides](docs/how-it-decides.md): the statistics, provider errors, and the full benchmark
- [Contributing](CONTRIBUTING.md)

MIT licensed.

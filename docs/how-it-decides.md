# How it decides

## The question

Is the new model **at most `margin` worse** than the one you use today? That is a non-inferiority question. It is different from "is it better?", and it is the question a model switch actually asks: a cheaper or faster model is worth it if it is not meaningfully worse.

## The answer, step by step

1. **Every case runs on both models,** half the budget first (3 runs per case per model by default). The runs are shuffled so the two models alternate in random order.
2. **Each run is PASS, FAIL or ERROR.** Your check decides PASS or FAIL. ERROR is a provider outage (below) and is left out of the score.
3. **The suite difference** is the average over cases of (new pass rate minus current pass rate). Each case is compared with itself, so a big case cannot hide a small one.
4. **Look 1.** If the whole uncertainty range is above minus the margin, it is a pass. If it is entirely below minus the margin, it is a regression. A pass at look 1 also needs every must pass case to have no failures, because three runs cannot show a must pass case dropping.
5. **Look 2.** Otherwise the second half runs, and the same test is applied with the final boundary. The verdict is final; rerunning does not add looks.
6. **Must pass cases** fail the comparison when the new model's pass rate is clearly below the current one's: a one sided Fisher exact test at 5%, with a Holm correction across must pass cases, applied at the last look. A case marked `strict=True` fails on any single failed run instead. Strict cases can be flaky: a case that truly passes 98% of the time fails at least once in 6 runs about 11% of the time.
7. **Worth a look** lists other cases that look worse, with a Benjamini and Hochberg flag for the ones that are likely worse. They never block.

Order when several apply: regressed, then errors, then not enough data, then pass.

## The statistics

- **Uncertainty per case** uses lightly smoothed rates, (passes + 0.5) / (runs + 1), and divides by runs minus one. Plain rates claim that a case passing 3 of 3 has no uncertainty at all, and dividing by runs understates the spread of small samples. Both make a check pass models that are really worse: in our simulations the plain formula passed 8% to 11% of models that were exactly 10 points worse. This one keeps it at or below 4% in every pattern we tried.
- **Two looks** use O'Brien and Fleming's boundaries for a one sided 5% test: z = 2.373 at look 1 and 1.678 at look 2. Looking twice does not raise the miss rate.
- **Not enough data** comes with a suggestion: roughly how many runs per case per model would settle a comparison like this one next time.

## Provider errors

Kept out of the score, after the provider SDK's own retries:

- HTTP 429, 500, 502, 503, 504 and 529, and dropped connections, raised by the OpenAI or Anthropic Python SDKs, also when a framework wraps them.
- Anything your own `is_provider_error(exception)` function returns True for. Return False to count it as a failure.

Counted as failures, because a model change can cause them: context length errors, invalid tool calls, and the agent's own timeouts and step limits.

Limits: a case with more than 20% of its runs in error on either model is left out of the suite answer; a must pass case in that state makes the answer not enough data. More than 20% of all runs in error means no verdict (exit 11). When the new model has clearly more errors than the current one (5 points or more), the report says so, because the change may be causing them.

## Assumptions

- Runs are independent. Turn off response caching; the report notes when every run of every case gave the same output.
- Cases are checked by code. Answers graded by a person or an LLM judge are out of scope for now.
- Simulated results are not real repositories. Real false alarms can only be found through reports from users.

## Benchmark

Produced by [benchmarks/model_swap_benchmark.py](../benchmarks/model_swap_benchmark.py). Every run of a case passes with a fixed probability that differs between the two models by a known amount; the verdict comes from the library's own decision code. "Run once rule" is today's habit: run each case once per model and call it a regression if any case that passed on the current model fails on the new one.

**Pass when the new model really is 10 or more points worse is a miss. Regressed when it really is the same is a false alarm.**

Margin 10%, 6 runs per case per model in two looks, 2000 simulated swaps per row, seed 2026.

## 40 cases, mixed (20 at 100%, 10 at 95%, 6 at 85%, 4 at 60%)

| The new model really is | Pass | Regressed | Not enough data | Runs used | Run once rule: regressed |
|---|---|---|---|---|---|
| the same | 99.0% | 0.0% | 1.0% | 450 | 91.6% |
| the same, with 3% provider errors | 97.8% | 0.0% | 2.2% | 480 | 89.8% |
| 10 points worse, spread over all cases | 2.1% | 1.9% | 96.0% | 480 | 99.9% |
| 10 points worse, all in a few cases | 0.9% | 0.4% | 98.8% | 480 | 100.0% |
| 20 points worse | 0.0% | 88.3% | 11.7% | 444 | 100.0% |
| 5 points better | 100.0% | 0.0% | 0.0% | 393 | 77.7% |

## 30 cases, all at 97% (like the XPath example)

| The new model really is | Pass | Regressed | Not enough data | Runs used | Run once rule: regressed |
|---|---|---|---|---|---|
| the same | 98.8% | 0.0% | 1.2% | 356 | 60.0% |
| the same, with 3% provider errors | 98.7% | 0.0% | 1.4% | 360 | 60.2% |
| 10 points worse, spread over all cases | 1.0% | 0.7% | 98.3% | 360 | 98.0% |
| 10 points worse, all in a few cases | 0.1% | 0.1% | 99.9% | 360 | 100.0% |
| 20 points worse | 0.0% | 82.1% | 17.9% | 346 | 100.0% |
| 5 points better | 100.0% | 0.0% | 0.0% | 335 | 0.0% |

## 40 cases, all at 70% (a flaky agent)

| The new model really is | Pass | Regressed | Not enough data | Runs used | Run once rule: regressed |
|---|---|---|---|---|---|
| the same | 72.7% | 0.0% | 27.4% | 444 | 100.0% |
| the same, with 3% provider errors | 70.7% | 0.0% | 29.3% | 480 | 100.0% |
| 10 points worse, spread over all cases | 3.9% | 4.7% | 91.5% | 478 | 100.0% |
| 10 points worse, all in a few cases | 3.5% | 4.2% | 92.2% | 479 | 100.0% |
| 20 points worse | 0.0% | 70.1% | 29.9% | 444 | 100.0% |
| 5 points better | 96.7% | 0.0% | 3.4% | 380 | 99.9% |

## 10 cases, mixed

| The new model really is | Pass | Regressed | Not enough data | Runs used | Run once rule: regressed |
|---|---|---|---|---|---|
| the same | 45.7% | 0.0% | 54.3% | 120 | 48.0% |
| the same, with 3% provider errors | 39.4% | 0.0% | 60.7% | 120 | 47.6% |
| 10 points worse, spread over all cases | 2.4% | 1.2% | 96.5% | 120 | 81.4% |
| 10 points worse, all in a few cases | 0.9% | 1.0% | 98.0% | 120 | 100.0% |
| 20 points worse | 0.0% | 35.4% | 64.6% | 119 | 95.0% |
| 5 points better | 69.0% | 0.0% | 30.9% | 120 | 36.0% |

One must pass case breaks (100% to 0%), the rest unchanged, 40 cases: caught 100.0%; run once rule 100.0%.

# Changelog

## 0.2.0

- Cost per correct task: everything each model spent, failed runs included, divided by the runs that passed. Price per run hides retries, extra steps and failures; this does not
- Model calls per run and reasoning tokens per run, read from OpenAI, Anthropic and Gemini responses
- The report says when price per run and cost per correct task tell different stories

## 0.1.0

First release: should your agent switch models?

- `ai-agent-evaluation compare` runs a suite on a current and a candidate model and answers pass, regressed or not enough data
- Non-inferiority test on the suite with a margin (10 points by default), in two looks with O'Brien and Fleming boundaries
- Must pass cases, tested with Fisher's exact test and a Holm correction, or strict on any failure
- Provider outages from the OpenAI and Anthropic SDKs kept out of the score, with a hook for others
- A decision report in the console and Markdown, plus eval-verdict.json, with cost and speed per model
- Smoke mode, a pytest helper and a GitHub workflow to copy
- Test cases from JSONL or CSV files, with built in checks against an expected answer
- Keyless examples (a shopping assistant written for the OpenAI SDK, and an XPath healer) and a benchmark against running each case once

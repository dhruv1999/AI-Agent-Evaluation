# Changelog

## 0.1.0

First release: should your agent switch models?

- `ai-agent-evaluation compare` runs a suite on a current and a candidate model and answers pass, regressed or not enough data
- Non-inferiority test on the suite with a margin (10 points by default), in two looks with O'Brien and Fleming boundaries
- Must pass cases, tested with Fisher's exact test and a Holm correction, or strict on any failure
- Provider outages from the OpenAI and Anthropic SDKs kept out of the score, with a hook for others
- A decision report in the console and Markdown, plus eval-verdict.json, with cost and speed per model
- Smoke mode, a pytest helper and a GitHub workflow to copy
- A keyless XPath healing example and a benchmark against running each case once

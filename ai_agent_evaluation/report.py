"""The decision report: a console summary and a Markdown file with the same content."""

from __future__ import annotations

from .compare import ERRORS, NOT_ENOUGH_DATA, PASSED, REGRESSED, SMOKE, ModelStats, SwapResult

HEADLINE = {PASSED: "PASS", REGRESSED: "REGRESSED", NOT_ENOUGH_DATA: "NOT ENOUGH DATA", ERRORS: "ERRORS", SMOKE: "SMOKE TEST"}


def _rate(m: ModelStats) -> str:
    return "-" if m.pass_rate is None else f"{m.pass_rate:.1%}"


def _money(v: float | None) -> str:
    return "-" if v is None else f"${v:.4f}"


def _secs(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}s"


def _change(old: float | None, new: float | None, lower_word: str, higher_word: str) -> str:
    if old is None or new is None or old <= 0:
        return ""
    change = new / old - 1
    if abs(change) < 0.005:
        return "about the same"
    return f"{abs(change):.0%} {lower_word if change < 0 else higher_word}"


def _headline(result: SwapResult) -> list[str]:
    out = [f"{HEADLINE[result.verdict]}. {result.reason[0].upper()}{result.reason[1:]}."]
    if result.verdict == PASSED and result.answer and result.answer.lower < 0:
        out.append(f"Changes up to {round(result.margin * 100)} points worse are accepted; change this with --margin.")
    return out


def _num(v: float | None) -> str:
    return "-" if v is None else f"{v:,.1f}" if v < 100 else f"{v:,.0f}"


def _table(result: SwapResult) -> list[tuple[str, str, str, str]]:
    b, c = result.models
    rows = [
        ("pass rate", _rate(b), _rate(c), ""),
        ("cost per run", _money(b.cost_per_run), _money(c.cost_per_run), _change(b.cost_per_run, c.cost_per_run, "lower", "higher")),
        ("cost per correct", _money(b.cost_per_correct), _money(c.cost_per_correct), _change(b.cost_per_correct, c.cost_per_correct, "lower", "higher")),
        ("median time", _secs(b.median_seconds), _secs(c.median_seconds), _change(b.median_seconds, c.median_seconds, "faster", "slower")),
    ]
    if b.calls_per_run is not None or c.calls_per_run is not None:
        rows.append(("calls per run", _num(b.calls_per_run), _num(c.calls_per_run), _change(b.calls_per_run, c.calls_per_run, "fewer", "more")))
    if b.reasoning_tokens_per_run is not None or c.reasoning_tokens_per_run is not None:
        rows.append(("reasoning tokens", _num(b.reasoning_tokens_per_run), _num(c.reasoning_tokens_per_run), ""))
    return rows


def _cost_story(result: SwapResult) -> list[str]:
    """When price per run and cost per correct task tell different stories, say so."""
    b, c = result.models
    if None in (b.cost_per_run, c.cost_per_run, b.cost_per_correct, c.cost_per_correct) or not b.cost_per_run or not b.cost_per_correct:
        return []
    per_run = c.cost_per_run / b.cost_per_run - 1  # type: ignore[operator]
    per_correct = c.cost_per_correct / b.cost_per_correct - 1  # type: ignore[operator]
    if abs(per_run - per_correct) < 0.05:
        return []

    def word(x: float) -> str:
        return f"{abs(x):.0%} {'cheaper' if x < 0 else 'more expensive'}"

    return [
        f"Cost per correct task: {result.candidate_model} is {word(per_correct)}, against {word(per_run)} per run."
        " Failed runs still cost money, so the real saving is the per correct figure."
    ]


def _sections(result: SwapResult) -> list[list[str]]:
    """Paragraphs after the table, each a list of lines; indented lines are details."""
    sections = []
    story = _cost_story(result)
    if story:
        sections.append(story)
    if result.must_pass:
        rows = ["Must pass:"]
        for m in result.must_pass:
            got, total = (int(x) for x in m.candidate.split("/"))
            base_got, base_total = (int(x) for x in m.baseline.split("/"))
            lower = total and base_total and got / total < base_got / base_total
            state = "FAILED" if m.failed else "unsettled (provider errors)" if m.unsettled else "lower, but too few runs to call it a drop" if lower else "ok"
            rows.append(f"  {m.name}: {m.baseline} -> {m.candidate}  {state}")
        if any(not m.failed and not m.unsettled and "/" in m.candidate and m.candidate.split("/")[0] != m.candidate.split("/")[1] for m in result.must_pass):
            rows.append("  To fail on any single failure instead, mark the case strict=True.")
        sections.append(rows)
    if result.diagnostics:
        rows = ["Worth a look, not blocking:"]
        for d in result.diagnostics[:8]:
            rows.append(f"  {d.name}: {d.baseline} -> {d.candidate}  {'likely worse' if d.likely_worse else 'could be chance'}")
        if len(result.diagnostics) > 8:
            rows.append(f"  and {len(result.diagnostics) - 8} more")
        sections.append(rows)
    e = result.errors
    if e.get("baseline") or e.get("candidate"):
        reasons = ", ".join(f"{k} x{v}" for k, v in e["reasons"].items())
        rows = [
            f"Kept out of the score: {e['baseline'] + e['candidate']} runs with provider errors ({reasons}), {e['share']:.1%} of runs."
            f" Baseline {e['baseline']}, candidate {e['candidate']}."
        ]
        if e.get("candidate_rate_higher"):
            rows.append("The new model had clearly more provider errors than the current one; the change itself may be causing them.")
        sections.append(rows)
    for note in result.notes:
        sections.append([note])
    stopped = "smoke test" if result.verdict == SMOKE else f"stopped at look {result.look} of {result.looks}"
    sections.append([f"Runs: {result.runs_used} of a {result.runs_budget} budget, {stopped}. Seed {result.seed}."])
    return sections


def text(result: SwapResult) -> str:
    out = [f"Model swap: {result.baseline_model} -> {result.candidate_model}   (suite {result.suite}, {len(result.tallies)} cases)", ""]
    out += _headline(result) + [""]
    width = max(len(result.baseline_model), len(result.candidate_model), 9) + 3
    out.append(f"{'':17}{result.baseline_model:>{width}}{result.candidate_model:>{width}}")
    for label, old, new, change in _table(result):
        out.append(f"{label:17}{old:>{width}}{new:>{width}}   {change}".rstrip())
    for section in _sections(result):
        out += [""] + section
    return "\n".join(out)


def markdown(result: SwapResult) -> str:
    md = [f"### Model swap: {result.baseline_model} → {result.candidate_model}", "", f"Suite `{result.suite}`, {len(result.tallies)} cases.", ""]
    head = _headline(result)
    md += [f"**{head[0]}**"] + head[1:] + [""]
    md += [f"| | {result.baseline_model} | {result.candidate_model} | |", "|---|---|---|---|"]
    md += [f"| {label} | {old} | {new} | {change} |" for label, old, new, change in _table(result)]
    for section in _sections(result):
        md.append("")
        md += [f"- {line.strip()}" if line.startswith("  ") else line for line in section]
    md += [
        "",
        "<details><summary>Every case</summary>",
        "",
        f"| Case | {result.baseline_model} | {result.candidate_model} | Provider errors |",
        "|---|---|---|---|",
    ]
    for t in result.tallies:
        flag = " (must pass)" if t.must_pass else ""
        md.append(f"| {t.name}{flag} | {t.b_pass}/{t.b_n} | {t.c_pass}/{t.c_n} | {t.b_err + t.c_err} |")
    md += ["", "</details>"]
    return "\n".join(md) + "\n"

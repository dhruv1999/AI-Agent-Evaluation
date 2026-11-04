"""Should the XPath healing agent move from model-old to a cheaper model?

    ai-agent-evaluation compare examples/xpath_healing/suite.py --baseline-model model-old --candidate-model model-new
    ai-agent-evaluation compare examples/xpath_healing/suite.py --baseline-model model-old --candidate-model model-mini

Each case is a page whose markup changed, and the check is whether the healed XPath finds exactly
the right element on the new page.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from healer import heal_xpath

from ai_agent_evaluation import Case, Suite

FORMS = {
    "login": ["email", "password", "sign-in"],
    "claims": ["claim-number", "member-id", "submit-claim"],
    "payment": ["card-number", "expiry", "pay"],
    "eligibility": ["policy-number", "date-of-birth", "check-eligibility"],
    "prior-auth": ["procedure-code", "provider-npi", "request-auth"],
    "appointment": ["clinic", "visit-date", "book"],
    "billing": ["invoice-number", "amount", "dispute"],
    "profile": ["first-name", "last-name", "save-profile"],
    "search": ["query", "filter", "search"],
    "settings": ["timezone", "language", "save-settings"],
}
MUST_PASS = {"submit-claim", "pay"}


def page(form: str, fields: list[str]) -> str:
    """The page after a redesign: ids were renamed, and every field moved into a wrapper."""
    inputs = "".join(f'<div class="row"><label>{f}</label><input id="f-{i}" data-qa="{f}" /></div>' for i, f in enumerate(fields[:-1]))
    actions = f'<div class="actions"><button type="button">Cancel</button><button data-qa="{fields[-1]}">Go</button></div>'
    return f'<html><body><form id="{form}-v2">{inputs}{actions}</form></body></html>'


def finds(html: str, element: str):
    def finds_exactly_the_right_element(xpath: str) -> None:
        found = ET.fromstring(html).findall(xpath)
        assert len(found) == 1, f"{xpath} matched {len(found)} elements"
        assert found[0].get("data-qa") == element, f"{xpath} found the wrong element"

    return finds_exactly_the_right_element


def cases() -> list[Case]:
    out = []
    for form, fields in FORMS.items():
        html = page(form, fields)
        for field in fields:
            tag = "button" if field == fields[-1] else "input"
            case_input = {"html": html, "broken_xpath": f"//{tag}[@id='{field}']", "element": field, "tag": tag}
            out.append(Case(f"{form}: {field}", case_input, check=finds(html, field), must_pass=field in MUST_PASS))
    return out


suite = Suite(
    agent=heal_xpath,
    cases=cases(),
    name="xpath_healing",
    margin=0.10,
    runs=6,
    prices={"model-old": (3.0, 15.0), "model-new": (0.8, 4.0), "model-mini": (0.25, 1.25)},
    meta={"note": "simulated models; see healer.py"},
)

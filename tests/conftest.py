import itertools
import threading

from ai_agent_evaluation import Case, Suite


class Scripted:
    """A fake agent: for each (case, model), the outcomes listed, in order and then repeating."""

    def __init__(self, plan, default=True):
        self.plan, self.default = plan, default
        self.counters = {}
        self.lock = threading.Lock()
        self.__name__ = "scripted"

    def __call__(self, case_input, model):
        key = (case_input, model)
        with self.lock:
            if key not in self.counters:
                self.counters[key] = itertools.cycle(self.plan.get(key, [self.default]))
            outcome = next(self.counters[key])
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def is_true(output):
    return output is True


def suite_of(n, plan=None, must_pass=(), strict=(), **kw):
    cases = [Case(f"c{i}", f"c{i}", check=is_true, must_pass=f"c{i}" in must_pass, strict=f"c{i}" in strict) for i in range(n)]
    return Suite(agent=Scripted(plan or {}), cases=cases, **kw)

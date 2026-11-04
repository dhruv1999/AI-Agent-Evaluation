from ai_agent_evaluation.errors import provider_error


def fake(module, name, status=None, bases=(Exception,)):
    cls = type(name, bases, {"status_code": status} if status else {})
    cls.__module__ = module
    return cls


def test_openai_and_anthropic_outages_are_provider_errors():
    assert provider_error(fake("openai._exceptions", "RateLimitError", 429)("slow down")) == "HTTP 429"
    assert provider_error(fake("anthropic._exceptions", "OverloadedError", 529)("busy")) == "HTTP 529"
    connection = fake("openai._exceptions", "APIConnectionError")
    timeout = fake("openai._exceptions", "APITimeoutError", bases=(connection,))
    assert provider_error(timeout("timed out")) == "connection error"


def test_errors_that_a_model_change_can_cause_count_against_it():
    assert provider_error(fake("openai._exceptions", "BadRequestError", 400)("context too long")) is None
    assert provider_error(TimeoutError("agent step limit")) is None
    assert provider_error(fake("mylib", "RateLimitError", 429)("not a provider SDK")) is None


def test_wrapped_errors_and_the_hook():
    rate = fake("anthropic._exceptions", "RateLimitError", 429)
    try:
        try:
            raise rate("limited")
        except Exception as inner:
            raise RuntimeError("framework wrapper") from inner
    except RuntimeError as outer:
        assert provider_error(outer) == "HTTP 429"
        assert provider_error(outer, hook=lambda e: False) is None
    assert provider_error(ConnectionResetError("gateway"), hook=lambda e: True) == "ConnectionResetError"
    assert provider_error(ValueError("x"), hook=lambda e: None) is None

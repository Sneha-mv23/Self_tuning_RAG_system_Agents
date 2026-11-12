import logging

from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.llm_limits import QuotaExhausted, is_rate_limit

log = logging.getLogger("rag-api")


def map_error(exc: Exception) -> tuple[int, str]:
    """(HTTP status, message safe to show a visitor). Never include the exception text."""
    msg = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, QuotaExhausted) or is_rate_limit(msg):
        return 429, "The demo's model quota is used up for now. Try again later, or browse the saved results."
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return 503, "The language model is unreachable right now. Try again in a moment."
    if isinstance(exc, APIStatusError):
        return 502, "The language model returned an error."
    return 500, "Something went wrong answering that question."
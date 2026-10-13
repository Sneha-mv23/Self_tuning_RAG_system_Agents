class QuotaExhausted(RuntimeError):
    """Raised when an API refuses further calls (rate limit or daily quota)."""


def is_rate_limit(err: str) -> bool:
    e = err.lower()
    return any(s in e for s in ("429", "ratelimit", "resource_exhausted", "quota"))


def is_daily_quota(err: str) -> bool:
    return "perday" in err.lower().replace("_", "").replace(" ", "")
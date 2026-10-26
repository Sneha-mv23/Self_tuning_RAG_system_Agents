from dataclasses import dataclass
from math import comb

MIN_GAIN = 2  # one dev question is 1.7 points, so a one-question change counts as noise


def sign_test_p(fixed: int, broken: int) -> float:
    """Two-sided exact sign test on the questions that changed outcome. Informational only."""
    n = fixed + broken
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(fixed, broken) + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass(frozen=True)
class Decision:
    accepted: bool
    net: int
    fixed: list[str]
    broken: list[str]
    p_value: float
    reason: str


def decide(parent: dict[str, bool], candidate: dict[str, bool], min_gain: int = MIN_GAIN) -> Decision:
    if set(parent) != set(candidate):
        raise ValueError("parent and candidate must cover the same questions")
    fixed = sorted(q for q in candidate if candidate[q] and not parent[q])
    broken = sorted(q for q in candidate if parent[q] and not candidate[q])
    net = len(fixed) - len(broken)
    p = sign_test_p(len(fixed), len(broken))
    verdict = "meets" if net >= min_gain else "is below"
    reason = (f"net {net:+d} passes (fixed {len(fixed)}, broken {len(broken)}; sign-test p={p:.2f}), "
              f"which {verdict} the required +{min_gain}")
    return Decision(net >= min_gain, net, fixed, broken, p, reason)
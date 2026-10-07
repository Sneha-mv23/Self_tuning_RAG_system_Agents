import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - margin) / d, (center + margin) / d)


def agreement_stats(pairs: list[tuple[bool, bool]], pool: dict[bool, int]) -> dict:
    """pairs: (judge_verdict, human_label) for each labeled item.
    pool: how many items in the full pool the judge gave each verdict.

    The sample over-represents judge 'no' verdicts on purpose, so each item is weighted
    by pool_count / sample_count of its stratum before computing agreement and kappa.
    """
    n = {True: 0, False: 0}
    agree = {True: 0, False: 0}
    for judge, human in pairs:
        n[judge] += 1
        agree[judge] += int(judge == human)

    table = {(j, h): 0.0 for j in (True, False) for h in (True, False)}
    for judge, human in pairs:
        table[(judge, human)] += pool[judge] / n[judge]
    total = sum(table.values())
    p = {k: v / total for k, v in table.items()}

    po = p[(True, True)] + p[(False, False)]
    judge_yes = p[(True, True)] + p[(True, False)]
    human_yes = p[(True, True)] + p[(False, True)]
    pe = judge_yes * human_yes + (1 - judge_yes) * (1 - human_yes)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")

    return {
        "n": len(pairs),
        "weighted_agreement": po,
        "kappa": kappa,
        "judge_yes_n": n[True],
        "judge_yes_agree": agree[True],
        "judge_yes_ci": wilson(agree[True], n[True]),
        "judge_no_n": n[False],
        "judge_no_agree": agree[False],
        "judge_no_ci": wilson(agree[False], n[False]),
    }
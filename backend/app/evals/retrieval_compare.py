def covered(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def compare_ranks(dense: dict[str, int | None], hybrid: dict[str, int | None], k: int) -> dict[str, list[str]]:
    """Split question ids by whether each retriever covers ALL gold evidence within the top k."""
    out: dict[str, list[str]] = {"fixed": [], "broken": [], "both_hit": [], "both_miss": []}
    for qid in sorted(dense):
        d, h = covered(dense[qid], k), covered(hybrid[qid], k)
        if d and h:
            out["both_hit"].append(qid)
        elif h:
            out["fixed"].append(qid)
        elif d:
            out["broken"].append(qid)
        else:
            out["both_miss"].append(qid)
    return out
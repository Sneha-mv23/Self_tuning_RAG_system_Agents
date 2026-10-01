import json
import random
from pathlib import Path

from app.evals.golden import GoldenItem

SPLIT_PATH = "data/golden_set/split.json"


def make_split(
    items: list[GoldenItem],
    existing: dict[str, str] | None = None,
    dev_frac: float = 0.6,
    seed: int = 42,
) -> dict[str, str]:
    """Assign each question id to 'dev' or 'test', stratified by type.

    Existing assignments are never changed. Only unassigned ids are placed.
    """
    valid = {i.id for i in items}
    split = {k: v for k, v in (existing or {}).items() if k in valid}

    by_type: dict[str, list[GoldenItem]] = {}
    for it in items:
        by_type.setdefault(it.type, []).append(it)

    for qtype, group in sorted(by_type.items()):
        target_dev = round(dev_frac * len(group))
        n_dev = sum(1 for it in group if split.get(it.id) == "dev")
        new_ids = sorted(it.id for it in group if it.id not in split)
        random.Random(f"{seed}-{qtype}").shuffle(new_ids)
        for qid in new_ids:
            if n_dev < target_dev:
                split[qid] = "dev"
                n_dev += 1
            else:
                split[qid] = "test"
    return split


def save_split(split: dict[str, str], path: str = SPLIT_PATH) -> None:
    Path(path).write_text(json.dumps(dict(sorted(split.items())), indent=2) + "\n", encoding="utf-8")


def load_split(path: str = SPLIT_PATH) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def select(items: list[GoldenItem], split: dict[str, str], which: str) -> list[GoldenItem]:
    if which not in ("dev", "test"):
        raise ValueError("which must be 'dev' or 'test'")
    return [i for i in items if split.get(i.id) == which]
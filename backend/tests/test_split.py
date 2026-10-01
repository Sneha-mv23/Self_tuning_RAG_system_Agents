from app.evals.golden import Evidence, GoldenItem
from app.evals.split import make_split, select


def _items(n_fact=10, n_un=10, prefix=""):
    out = []
    for i in range(n_fact):
        out.append(GoldenItem(id=f"{prefix}f{i}", question="q", type="factual", reference_answer="a",
                              evidence=[Evidence(doc_id="a.md", quote="x")]))
    for i in range(n_un):
        out.append(GoldenItem(id=f"{prefix}u{i}", question="q", type="unanswerable"))
    return out


def test_split_is_stratified():
    items = _items()
    split = make_split(items, dev_frac=0.6)
    for qtype in ("factual", "unanswerable"):
        dev = [i for i in select(items, split, "dev") if i.type == qtype]
        assert len(dev) == 6


def test_split_is_deterministic():
    items = _items()
    assert make_split(items) == make_split(items)


def test_dev_and_test_do_not_overlap_and_cover_everything():
    items = _items()
    split = make_split(items)
    dev = {i.id for i in select(items, split, "dev")}
    test = {i.id for i in select(items, split, "test")}
    assert not dev & test
    assert dev | test == {i.id for i in items}


def test_existing_assignments_never_change():
    items = _items()
    first = make_split(items)
    more = items + _items(prefix="new_")
    second = make_split(more, existing=first)
    for qid, which in first.items():
        assert second[qid] == which
    assert all(i.id in second for i in more)
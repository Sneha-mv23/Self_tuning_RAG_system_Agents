from dataclasses import replace

from app.optimizer.candidates import Candidate, config_diff, flatten, rank_candidates
from app.optimizer.levers import Proposal
from app.rag.pipeline import RAGConfig

BASE = RAGConfig()


def cand(lever, order, gain, **kw):
    return Candidate(lever, replace(BASE, **kw), order, gain)


def test_rank_candidates_orders_by_gain_and_drops_non_positive():
    cs = [cand("hybrid_search", 0, 3, retriever="hybrid"), cand("top_k", 1, 5, top_k=6),
          cand("chunk_size", 2, -5, chunk_size=192), cand("prompt", 3, None, prompt="basic"),
          cand("overlap", 4, 0, overlap=16)]
    to_try, rejected = rank_candidates(cs)
    assert [c.lever for c in to_try] == ["top_k", "hybrid_search", "prompt"]
    assert [c.lever for c in rejected] == ["chunk_size", "overlap"]


def test_flatten_skips_tried_unavailable_and_duplicates():
    hybrid, k6 = replace(BASE, retriever="hybrid"), replace(BASE, top_k=6)
    proposals = [Proposal("hybrid_search", 9, True, [hybrid], ""),
                 Proposal("top_k", 3, True, [k6, hybrid], ""),
                 Proposal("reranker", 4, False, [], "")]
    assert flatten(proposals, tried={k6}) == [("hybrid_search", hybrid)]


def test_config_diff_lists_only_changed_fields():
    assert config_diff(BASE, replace(BASE, top_k=6, retriever="hybrid")) == {
        "top_k": [4, 6], "retriever": ["dense", "hybrid"]}
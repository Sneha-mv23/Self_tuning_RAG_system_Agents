import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.evals.golden import GoldenItem
from app.evals.retrieval_metrics import first_hit_rank, rank_needed
from app.rag.chunker import chunk_corpus
from app.rag.index import VectorIndex
from app.rag.loader import Document
from app.rag.pipeline import RAGConfig

SEARCH_DEPTH = 20
KS = (1, 3, 4, 5, 10, 20)


@dataclass
class QuestionRetrieval:
    id: str
    type: str
    rank_needed: int | None  # smallest top_k covering ALL evidence; None if not found
    evidence_ranks: list[int | None]  # rank of each evidence span, None if not found
    top1_score: float
    retrieved: list[dict]  # chunk_id, doc_id, start, end, score, rank


@dataclass
class RetrievalReport:
    config: dict
    n_answerable: int
    n_unanswerable: int
    n_chunks: int
    mean_chunk_tokens: float
    recall: dict  # k -> share of questions whose evidence is ALL within top k
    evidence_recall: dict  # k -> mean share of evidence spans within top k
    mrr: float
    recall_by_type: dict  # question type -> recall at config.top_k
    questions: list[QuestionRetrieval]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(results: list[QuestionRetrieval], top_k: int) -> dict:
    """Pure aggregation. Unanswerable questions have no gold evidence and are excluded."""
    ans = [r for r in results if r.type != "unanswerable"]

    def covered(r: QuestionRetrieval, k: int) -> float:
        return 1.0 if r.rank_needed is not None and r.rank_needed <= k else 0.0

    recall = {k: _mean([covered(r, k) for r in ans]) for k in KS}
    evidence_recall = {
        k: _mean([
            sum(1 for x in r.evidence_ranks if x is not None and x <= k) / len(r.evidence_ranks)
            for r in ans
        ])
        for k in KS
    }
    mrr = _mean([1.0 / r.rank_needed if r.rank_needed else 0.0 for r in ans])
    recall_by_type = {
        t: _mean([covered(r, top_k) for r in ans if r.type == t])
        for t in sorted({r.type for r in ans})
    }
    return {
        "recall": recall,
        "evidence_recall": evidence_recall,
        "mrr": mrr,
        "recall_by_type": recall_by_type,
    }


class RetrievalEvaluator:
    """Builds one index per (chunk_size, overlap) and reuses it across evaluations."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self._indexes: dict[tuple[int, int], VectorIndex] = {}

    def index_for(self, chunk_size: int, overlap: int) -> VectorIndex:
        key = (chunk_size, overlap)
        if key not in self._indexes:
            chunks = chunk_corpus(self.docs, chunk_size, overlap)
            self._indexes[key] = VectorIndex.build(chunks)
        return self._indexes[key]

    def evaluate(self, config: RAGConfig, items: list[GoldenItem]) -> RetrievalReport:
        index = self.index_for(config.chunk_size, config.overlap)
        results: list[QuestionRetrieval] = []
        for it in items:
            retrieved = index.search(it.question, SEARCH_DEPTH)
            results.append(
                QuestionRetrieval(
                    id=it.id,
                    type=it.type,
                    rank_needed=rank_needed(retrieved, it) if it.evidence else None,
                    evidence_ranks=[first_hit_rank(retrieved, ev) for ev in it.evidence],
                    top1_score=retrieved[0].score,
                    retrieved=[
                        {
                            "chunk_id": r.chunk.chunk_id,
                            "doc_id": r.chunk.doc_id,
                            "start": r.chunk.start,
                            "end": r.chunk.end,
                            "score": round(r.score, 4),
                            "rank": r.rank,
                        }
                        for r in retrieved
                    ],
                )
            )
        s = summarize(results, config.top_k)
        n_un = sum(1 for r in results if r.type == "unanswerable")
        return RetrievalReport(
            config=asdict(config),
            n_answerable=len(results) - n_un,
            n_unanswerable=n_un,
            n_chunks=len(index.chunks),
            mean_chunk_tokens=_mean([c.n_tokens for c in index.chunks]),
            recall=s["recall"],
            evidence_recall=s["evidence_recall"],
            mrr=s["mrr"],
            recall_by_type=s["recall_by_type"],
            questions=results,
        )


def save_report(report: RetrievalReport, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
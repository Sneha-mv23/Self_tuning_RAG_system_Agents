import time
from dataclasses import asdict
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from app.evals.generation_run import MAX_TOKENS
from app.evals.retrieval_eval import RetrievalEvaluator
from app.llm import get_generator
from app.rag.chunker import count_tokens
from app.rag.generator import build_prompt
from app.rag.loader import Document, load_corpus
from app.rag.pipeline import RAGConfig


class RAGState(TypedDict, total=False):
    question: str
    config: dict
    hits: list  # Retrieved objects, kept in memory only
    prompt: str
    prompt_tokens: int
    answer: str
    sources: list[dict]
    timings: dict


def build_rag_graph(index_for: Callable[[int, int, str], Any], llm_factory: Callable[[float, int], Any] | None = None):
    """retrieve -> build_prompt -> generate. index_for(chunk_size, overlap, retriever) returns an index
    with search(query, k); llm_factory(temperature, max_tokens) returns an object with invoke(prompt)."""
    llm_factory = llm_factory or (lambda temperature, max_tokens: get_generator(temperature, max_tokens))

    def retrieve(state: RAGState) -> dict:
        cfg = RAGConfig(**state["config"])
        start = time.perf_counter()
        hits = index_for(cfg.chunk_size, cfg.overlap, cfg.retriever).search(state["question"], cfg.top_k)
        return {"hits": hits, "timings": {**state.get("timings", {}), "retrieve": time.perf_counter() - start}}

    def make_prompt(state: RAGState) -> dict:
        cfg = RAGConfig(**state["config"])
        prompt = build_prompt(cfg.prompt, state["question"], state["hits"])
        return {"prompt": prompt, "prompt_tokens": count_tokens(prompt)}

    def generate(state: RAGState) -> dict:
        cfg = RAGConfig(**state["config"])
        llm = llm_factory(cfg.temperature, MAX_TOKENS)
        start = time.perf_counter()
        reply = llm.invoke(state["prompt"])
        elapsed = time.perf_counter() - start
        sources = [{"rank": h.rank, "score": round(h.score, 4), "doc_id": h.chunk.doc_id,
                    "start": h.chunk.start, "end": h.chunk.end, "text": h.chunk.text.strip()}
                   for h in state["hits"]]
        return {"answer": str(reply.content).strip(), "sources": sources,
                "timings": {**state.get("timings", {}), "generate": elapsed}}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("build_prompt", make_prompt)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "build_prompt")
    graph.add_edge("build_prompt", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


class RAGService:
    """Holds the indexes (built once per chunking setting) and the compiled graph."""

    def __init__(self, docs: list[Document] | None = None, llm_factory=None):
        self.evaluator = RetrievalEvaluator(docs if docs is not None else load_corpus())
        self.graph = build_rag_graph(self.evaluator.index_for, llm_factory)

    def ask(self, question: str, config: dict | None = None) -> dict:
        cfg = asdict(RAGConfig(**(config or {})))  # validates the settings, fills in defaults
        out = self.graph.invoke({"question": question, "config": cfg})
        return {
            "question": question, "config": cfg, "answer": out["answer"], "sources": out["sources"],
            "prompt_tokens": out["prompt_tokens"],
            "timings": {k: round(v, 3) for k, v in out["timings"].items()},
        }
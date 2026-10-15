import time
from dataclasses import dataclass
from app.rag.hybrid import HybridIndex

from app.llm import get_generator
from app.rag.chunker import chunk_corpus, count_tokens
from app.rag.generator import build_prompt
from app.rag.index import Retrieved, VectorIndex
from app.rag.loader import Document


RETRIEVERS = ("dense", "hybrid")


@dataclass(frozen=True)
class RAGConfig:
    chunk_size: int = 256
    overlap: int = 32
    top_k: int = 4
    prompt: str = "strict"
    temperature: float = 0.0
    retriever: str = "dense"

    def __post_init__(self):
        if self.retriever not in RETRIEVERS:
            raise ValueError(f"unknown retriever '{self.retriever}', choose from {RETRIEVERS}")


@dataclass
class RAGResult:
    question: str
    answer: str
    retrieved: list[Retrieved]
    latency_s: float
    prompt_tokens: int


class RAGPipeline:
    def __init__(self, docs: list[Document], config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        chunks = chunk_corpus(docs, self.config.chunk_size, self.config.overlap)
        dense = VectorIndex.build(chunks)
        self.index = HybridIndex(dense) if self.config.retriever == "hybrid" else dense
        self.llm = get_generator(temperature=self.config.temperature)

    def retrieve(self, question: str) -> list[Retrieved]:
        return self.index.search(question, self.config.top_k)

    def answer(self, question: str) -> RAGResult:
        start = time.perf_counter()
        retrieved = self.retrieve(question)
        prompt = build_prompt(self.config.prompt, question, retrieved)
        reply = self.llm.invoke(prompt)
        return RAGResult(
            question=question,
            answer=str(reply.content).strip(),
            retrieved=retrieved,
            latency_s=time.perf_counter() - start,
            prompt_tokens=count_tokens(prompt),
        )
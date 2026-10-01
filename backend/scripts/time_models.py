from langchain_openai import ChatOpenAI

from app.config import settings
from app.rag.loader import load_corpus
from app.rag.pipeline import RAGConfig, RAGPipeline

q = "What does Optional do in a query parameter?"
pipe = RAGPipeline(load_corpus(), RAGConfig())

for model in ["llama3:latest", "phi3:latest"]:
    pipe.llm = ChatOpenAI(
        base_url=settings.generator_base_url,
        api_key=settings.generator_api_key,
        model=model,
        temperature=0,
        max_tokens=200,
    )
    pipe.answer(q)  # warm-up: loads the model, not timed
    r = pipe.answer(q)
    print(f"{model}: {r.latency_s:.1f}s  ->  {r.answer[:120]}")
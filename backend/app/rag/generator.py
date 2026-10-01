from app.rag.index import Retrieved

PROMPTS = {
    # Deliberately simple baseline. The Optimizer can switch to "strict".
    "basic": (
        "Answer the question using the context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
    "strict": (
        "You are a documentation assistant. Answer ONLY using the numbered "
        "context passages below. If the passages do not contain the answer, "
        "reply exactly: I don't know based on the provided documents. "
        "Keep the answer concise and cite passages like [1].\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
}


def format_context(retrieved: list[Retrieved]) -> str:
    return "\n\n".join(
        f"[{r.rank}] (source: {r.chunk.doc_id})\n{r.chunk.text.strip()}" for r in retrieved
    )


def build_prompt(prompt_name: str, question: str, retrieved: list[Retrieved]) -> str:
    if prompt_name not in PROMPTS:
        raise ValueError(f"unknown prompt '{prompt_name}', choose from {list(PROMPTS)}")
    return PROMPTS[prompt_name].format(
        context=format_context(retrieved), question=question
    )
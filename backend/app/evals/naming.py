def run_name(split: str, c) -> str:
    """File-name stem for a run. Dense runs keep the original names, so existing files stay valid."""
    base = f"{split}_cs{c.chunk_size}_ov{c.overlap}_k{c.top_k}_{c.prompt}_t{c.temperature}"
    return base if c.retriever == "dense" else f"{base}_{c.retriever}"
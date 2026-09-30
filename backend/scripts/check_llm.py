from app.llm import get_generator, get_judge

for name, llm in [("generator", get_generator()), ("judge", get_judge())]:
    reply = llm.invoke("Reply with exactly one word: OK")
    print(f"{name}: {reply.content!r}")
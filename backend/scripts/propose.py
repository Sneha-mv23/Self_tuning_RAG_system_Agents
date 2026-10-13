from app.db.experiments import connect, list_runs
from app.evals.failure_classifier import load_outcomes
from app.optimizer.levers import estimated_prompt_tokens, plan
from app.rag.pipeline import RAGConfig

conn = connect()
for r in list_runs(conn):
    _, outcomes = load_outcomes(conn, r["id"])
    config = RAGConfig(chunk_size=r["chunk_size"], overlap=r["overlap"], top_k=r["top_k"],
                       prompt=r["prompt"], temperature=r["temperature"])
    print(f"RUN {r['id']}  prompt={config.prompt}  cs/ov/k={config.chunk_size}/{config.overlap}/{config.top_k}")
    for p in plan(outcomes, config):
        status = "available" if p.available else "to build"
        print(f"  {p.lever:<19} {p.n_addressed:>3}  [{status}]  {p.note}")
        for c in p.candidates:
            print(f"        try: cs/ov/k={c.chunk_size}/{c.overlap}/{c.top_k} prompt={c.prompt} "
                  f"temp={c.temperature}  (~{estimated_prompt_tokens(c)} prompt tokens)")
    print()
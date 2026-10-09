from app.db.experiments import connect, list_runs

runs = list_runs(connect())
if not runs:
    raise SystemExit("no runs logged yet. Use backend/scripts/log_run.py first.")

head = (f"{'id':>3} {'split':<5} {'prompt':<7} {'cs/ov/k':<10} {'recall@k':>8} {'MRR':>6} {'correct':>7} "
        f"{'faithful':>8} {'decline':>7} {'refuse':>6} {'p50 s':>6} {'p95 s':>6} {'tokens':>6}")
print(head)
print("-" * len(head))
for r in runs:
    print(f"{r['id']:>3} {r['split']:<5} {r['prompt']:<7} "
          f"{r['chunk_size']}/{r['overlap']}/{r['top_k']:<4} "
          f"{r['recall_at_k']:>8.2f} {r['mrr']:>6.3f} {r['answer_correctness']:>7.2f} "
          f"{r['faithfulness']:>8.2f} {r['proper_refusal']:>7.2f} {r['pure_refusal_answerable']:>6.2f} "
          f"{r['median_latency_s']:>6.0f} {r['p95_latency_s']:>6.0f} {r['mean_prompt_tokens']:>6.0f}")
print("\ncorrect = answer correctness on answerable questions | decline = properly declined unanswerable")
print("refuse = share of answerable questions the model refused | judge/generator models are stored per run")
from statistics import mean

from app.db.experiments import connect
from app.optimizer.experiment import pass_map
from app.optimizer.test_plan import EVAL_CONFIGS

conn = connect()
runs = conn.execute("SELECT * FROM runs WHERE split = 'test' ORDER BY id").fetchall()
if not runs:
    raise SystemExit("no test runs yet. Run backend/scripts/test_eval.py first.")


def key(c):
    return (c.chunk_size, c.overlap, c.top_k, c.prompt, c.temperature, c.retriever)


name_of = {key(c): n for n, c in EVAL_CONFIGS.items()}
by_name: dict[str, list] = {}
for r in runs:
    name = name_of.get((r["chunk_size"], r["overlap"], r["top_k"], r["prompt"], r["temperature"], r["retriever"]))
    if name:
        by_name.setdefault(name, []).append((r["replicate"], r, pass_map(conn, r["id"])))

n_q = len(next(iter(by_name.values()))[0][2])
print(f"HELD-OUT TEST SPLIT: {n_q} questions, evaluated once per (config, replicate)\n")
print(f"{'config':<10} {'reps':>4}  {'passes per replicate':<22} {'mean':>5} {'recall@k':>8} {'correct':>7} "
      f"{'faithful':>8} {'decline':>7} {'tokens':>7} {'p50 s':>6}")
for name in EVAL_CONFIGS:
    if name not in by_name:
        continue
    rows = sorted(by_name[name], key=lambda x: x[0])
    passes = [sum(pm.values()) for _, _, pm in rows]
    rs = [r for _, r, _ in rows]
    print(f"{name:<10} {len(rows):>4}  {str(passes):<22} {mean(passes):>5.1f} {rs[0]['recall_at_k']:>8.2f} "
          f"{mean(r['answer_correctness'] for r in rs):>7.2f} {mean(r['faithfulness'] for r in rs):>8.2f} "
          f"{mean(r['proper_refusal'] for r in rs):>7.2f} {mean(r['mean_prompt_tokens'] for r in rs):>7.0f} "
          f"{mean(r['median_latency_s'] for r in rs):>6.0f}")

base = {rep: pm for rep, _, pm in by_name.get("baseline", [])}
print("\nPAIRED COMPARISON WITH THE BASELINE (counts of questions that changed, same replicate number)")
for name in EVAL_CONFIGS:
    if name == "baseline" or name not in by_name:
        continue
    for rep, _, pm in sorted(by_name[name], key=lambda x: x[0]):
        if rep not in base:
            continue
        fixed = sum(1 for q in pm if pm[q] and not base[rep][q])
        broken = sum(1 for q in pm if not pm[q] and base[rep][q])
        print(f"  {name} vs baseline, replicate {rep}: fixed {fixed}, broken {broken}, net {fixed - broken:+d}")

print("\nThe range of passes across replicates of one config is the run-to-run noise. A paired net inside that")
print("range is not evidence that one config is better. tokens = measured mean prompt tokens per question.")
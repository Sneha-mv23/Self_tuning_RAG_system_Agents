import json
from pathlib import Path
import sys
from app.evals.agreement import agreement_stats

DIR = Path("data/judge_validation")
sample = json.loads((DIR / "sample.json").read_text(encoding="utf-8"))
judge_by_key = {(t["task"], t["run"], t["id"]): t["judge"] for t in sample["tasks"]}
labels_name = sys.argv[1] if len(sys.argv) > 1 else "labels.jsonl"
labels = [json.loads(line) for line in (DIR / labels_name).read_text(encoding="utf-8").splitlines()
          if line.strip()]

for task, title in (("correct", "ANSWER CORRECTNESS"), ("faithful", "FAITHFULNESS")):
    rows = [(judge_by_key[(l["task"], l["run"], l["id"])], l["human"], l) for l in labels if l["task"] == task]
    if not rows:
        print(f"{title}: no labels yet\n")
        continue
    pool = {True: sample["pool"][task]["true"], False: sample["pool"][task]["false"]}
    s = agreement_stats([(j, h) for j, h, _ in rows], pool)

    print(f"{title}: {s['n']} labels")
    print(f"  pool-weighted agreement = {s['weighted_agreement']:.2f}   Cohen's kappa = {s['kappa']:.2f}")
    for name, key in (("YES", "judge_yes"), ("NO", "judge_no")):
        lo, hi = s[f"{key}_ci"]
        print(f"  judge says {name:<3}: human agrees {s[f'{key}_agree']}/{s[f'{key}_n']}  (95% CI {lo:.2f} to {hi:.2f})")
    if min(s["judge_yes_n"], s["judge_no_n"]) < 5:
        print("  warning: fewer than 5 labels in one stratum, so these numbers are very uncertain")
    wrong = [(l["run"], l["id"], j, h) for j, h, l in rows if j != h]
    print("  disagreements (run, id, judge, human):")
    for run, qid, j, h in wrong:
        print(f"    {run} {qid}: judge={'yes' if j else 'no'} human={'yes' if h else 'no'}")
    print()
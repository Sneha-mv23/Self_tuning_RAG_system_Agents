import json
from pathlib import Path

from app.evals.agreement import wilson

DIR = Path("data/judge_validation_gemini")
sample = json.loads((DIR / "sample.json").read_text(encoding="utf-8"))
labels = [json.loads(x) for x in (DIR / "labels.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
if not labels:
    raise SystemExit("no labels yet. Run label_gemini.py first.")

print(f"pool: {sample['pool']}\n")
for task, title in (("correct", "ANSWER CORRECTNESS"), ("faithful", "FAITHFULNESS")):
    rows = [l for l in labels if l["task"] == task]
    if not rows:
        continue
    agree = sum(1 for l in rows if l["human"])
    lo, hi = wilson(agree, len(rows))
    print(f"{title}: the judge said YES to all of these; you agree with {agree} of {len(rows)} "
          f"(95% CI {lo:.2f} to {hi:.2f})")
    for l in rows:
        if not l["human"]:
            print(f"   you said NO to {l['id']}")
    print()
print("This measures how often a judge 'yes' is right. The judge never said 'no' on these answers,")
print("so the 'no' side cannot be assessed from this sample.")
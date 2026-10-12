import json
import sys
from pathlib import Path

from app.evals.agreement import sufficiency_table

DIR = Path("data/classifier_validation")
labels_name = sys.argv[1] if len(sys.argv) > 1 else "labels.jsonl"
sample = json.loads((DIR / "sample.json").read_text(encoding="utf-8"))
meta = {(t["run"], t["id"]): t for t in sample["tasks"]}
labels = [json.loads(x) for x in (DIR / labels_name).read_text(encoding="utf-8").splitlines() if x.strip()]
rows = [(meta[(l["run"], l["id"])], l["human"]) for l in labels]
if not rows:
    raise SystemExit("no labels yet. Run label_context.py first.")


def report(title: str, subset: list) -> None:
    if not subset:
        print(f"{title}: none\n")
        return
    s = sufficiency_table([(t["predicted_in_prompt"], h) for t, h in subset])
    tb, n = s["table"], s["n"]
    print(f"{title}: {n} items")
    print("                                  human says: yes   part   no")
    print(f"  classifier: gold in prompt              {tb[(True, 'y')]:>5}  {tb[(True, 'p')]:>5}  {tb[(True, 'n')]:>5}")
    print(f"  classifier: gold NOT in prompt          {tb[(False, 'y')]:>5}  {tb[(False, 'p')]:>5}  {tb[(False, 'n')]:>5}")
    for name in ("strict", "lenient"):
        lo, hi = s[f"{name}_ci"]
        print(f"  agreement ({name:<7}) {s[name]}/{n} = {s[name] / n:.2f}  (95% CI {lo:.2f} to {hi:.2f})")
    print()


report("ALL", rows)
report("FAILURES ONLY", [r for r in rows if r[0]["kind"] == "failure"])
report("CONTROLS (answered correctly, gold in prompt)", [r for r in rows if r[0]["kind"] == "control"])
print("strict = a 'part' label means the context did not have the key point; "
      "lenient = a 'part' label counts as having it\n")

print("disagreements under the strict reading (run, id, classifier category, gold in prompt, human):")
for t, h in rows:
    if t["predicted_in_prompt"] != (h == "y"):
        print(f"  {t['run']} {t['id']}: {t['category']}, in prompt={t['predicted_in_prompt']}, human={h}")
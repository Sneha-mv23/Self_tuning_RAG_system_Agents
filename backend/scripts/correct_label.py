import argparse
import json
import shutil
from pathlib import Path

DIR = Path("data/judge_validation")
LABELS = DIR / "labels.jsonl"
BLIND = DIR / "labels_blind_round1.jsonl"
LOG = DIR / "corrections.jsonl"

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["correct", "faithful"])
ap.add_argument("--run", required=True, choices=["strict", "basic"])
ap.add_argument("--id", required=True)
ap.add_argument("--to", required=True, choices=["yes", "no"])
ap.add_argument("--note", required=True, help="why you are changing it, in your own words")
args = ap.parse_args()

if not BLIND.exists():
    shutil.copy(LABELS, BLIND)
    print(f"saved the original labels to {BLIND} (do not edit that file)")

lines = [json.loads(x) for x in LABELS.read_text(encoding="utf-8").splitlines() if x.strip()]
new_value = args.to == "yes"
hit = [d for d in lines if (d["task"], d["run"], d["id"]) == (args.task, args.run, args.id)]
if len(hit) != 1:
    raise SystemExit(f"expected exactly one label for {args.task}/{args.run}/{args.id}, found {len(hit)}")
old_value = hit[0]["human"]
if old_value == new_value:
    raise SystemExit("that label already has this value, nothing to change")

hit[0]["human"] = new_value
LABELS.write_text("".join(json.dumps(d) + "\n" for d in lines), encoding="utf-8")
with LOG.open("a", encoding="utf-8") as f:
    f.write(json.dumps({"task": args.task, "run": args.run, "id": args.id,
                        "old": old_value, "new": new_value, "note": args.note}) + "\n")
print(f"{args.task}/{args.run}/{args.id}: {'yes' if old_value else 'no'} -> {args.to}  (logged in {LOG})")
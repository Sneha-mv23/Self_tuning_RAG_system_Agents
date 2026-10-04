import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
path = "experiments/runs/generation_dev_cs256_ov32_k4_strict_t0.0.json"
ids = set(sys.argv[1:])
for r in json.loads(Path(path).read_text(encoding="utf-8"))["records"]:
    if r["id"] in ids:
        print(f"{r['id']} | {r['question']}\n  -> {r['answer'][:400]}\n")
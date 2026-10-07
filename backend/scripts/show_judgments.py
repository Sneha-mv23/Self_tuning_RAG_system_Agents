import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
runs = Path("experiments/runs")
gen = json.loads((runs / "generation_dev_cs256_ov32_k4_strict_t0.0.json").read_text(encoding="utf-8"))["records"]
jud = json.loads((runs / "judged_dev_cs256_ov32_k4_strict_t0.0.json").read_text(encoding="utf-8"))["judgments"]
answers = {r["id"]: r for r in gen}
for j in jud:
    if j["id"] in sys.argv[1:]:
        r = answers[j["id"]]
        print(f"{j['id']} [{j['type']}] {r['question']}")
        print(f"  answer: {r['answer'][:500]}")
        print(f"  judge correct={j['correct']}: {j['correct_reason']}")
        print(f"  judge faithful={j['faithful']}: {j['faithful_reason']}\n")
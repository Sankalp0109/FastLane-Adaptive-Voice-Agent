"""Regenerate the 650 practice calls with the current code and check that every training example is still in data v5."""
import json, re
from fastlane.training.build_dataset import generate

def norm(messages):
    out = []
    for m in messages:
        m = dict(m)
        if isinstance(m.get("content"), str):
            m["content"] = re.sub(r'\\?"search_ms\\?":\s*[\d.]+,?\s*', "", m["content"])
        out.append(m)
    return json.dumps(out, sort_keys=True, ensure_ascii=False)

calls, examples = generate(650, progress=lambda *a: None)
old_calls = [json.loads(l) for l in open("data/train/calls.jsonl")]
have = set()
for split in ("train", "val"):
    for l in open(f"data/train/all_domains/{split}.jsonl"):
        have.add(norm(json.loads(l)["messages"]))
same_calls = sum(1 for a, b in zip(calls, old_calls) if a["plan"] == b["plan"] and a["ok"] == b["ok"] and a["turns"] == b["turns"] and a["examples"] == b["examples"])
missing = [e for e in examples if norm(e["messages"]) not in have]
print(json.dumps({"calls": len(calls), "calls_identical": same_calls, "kept": sum(c["ok"] for c in calls),
                  "examples": len(examples), "examples_not_in_v5": len(missing)}, indent=1))
for e in missing[:3]:
    print(json.dumps(e.get("meta"))[:300])

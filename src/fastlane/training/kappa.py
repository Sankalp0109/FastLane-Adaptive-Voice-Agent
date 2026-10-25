"""Inter-annotator agreement on routing labels (Section 8: "two of us label 300 turns to report Cohen's kappa").

    fastlane train label-sheet            # writes data/train/label_sheet.csv (300 first-step turns)
    fastlane train kappa data/train/label_sheet.csv

Each row shows the last lines of the conversation; two annotators fill annotator_a / annotator_b with
self | llm | staff without looking at rule_label. The report gives kappa between the annotators and
between each annotator and the rule-based label used for training.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

LABELS = ("self", "llm", "staff")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else round((observed - expected) / (1 - expected), 4)


def export_sheet(train_dir: Path, out: Path, n: int = 300, seed: int = 2026) -> Path:
    rows = []
    for split in ("val.jsonl", "train.jsonl"):
        for line in (train_dir / split).read_text().splitlines():
            ex = json.loads(line)
            if ex["meta"]["step"] == 0 and not ex["meta"].get("noisy"):
                convo = ex["messages"][1]["content"].split("CONVERSATION:\n")[-1].split("\n\nOutput the JSON")[0]
                rows.append({"id": f"{ex['meta']['plan']}#{len(rows)}", "conversation_tail": "\n".join(convo.split("\n")[-3:]),
                             "rule_label": ex["meta"]["route"]})
    rng = random.Random(seed)
    # Stratify so escalations are well represented (they are rare in the raw data).
    by = {k: [r for r in rows if r["rule_label"] == k] for k in LABELS}
    picked = rng.sample(by["llm"], min(len(by["llm"]), n // 4)) + rng.sample(by["staff"], min(len(by["staff"]), n // 6))
    rest = [r for r in by["self"]]
    picked += rng.sample(rest, min(len(rest), n - len(picked)))
    rng.shuffle(picked)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "conversation_tail", "annotator_a", "annotator_b", "rule_label"])
        w.writeheader()
        for r in picked:
            w.writerow({**r, "annotator_a": "", "annotator_b": ""})
    return out


def report(sheet: Path) -> dict:
    with sheet.open() as f:
        rows = [r for r in csv.DictReader(f) if r["annotator_a"].strip() and r["annotator_b"].strip()]
    a = [r["annotator_a"].strip().lower() for r in rows]
    b = [r["annotator_b"].strip().lower() for r in rows]
    rule = [r["rule_label"] for r in rows]
    return {"n": len(rows), "kappa_a_b": cohen_kappa(a, b), "kappa_a_rule": cohen_kappa(a, rule), "kappa_b_rule": cohen_kappa(b, rule)}

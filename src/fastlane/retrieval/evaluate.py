"""Retrieval metrics on the labelled policy-QA set: Recall@3, top-1 accuracy, answer-in-context,
current-vs-old policy choice and "not covered" detection."""

from __future__ import annotations

import json
from pathlib import Path

from fastlane.retrieval.index import PolicyIndex

QA_PATH = Path(__file__).with_name("policy_qa.jsonl")


def load_qa(path: Path = QA_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(index: PolicyIndex | None = None, qa: list[dict] | None = None, k: int = 3) -> dict:
    index = index or PolicyIndex()
    qa = qa or load_qa()
    answerable = [x for x in qa if x["doc"]]
    unanswerable = [x for x in qa if not x["doc"]]
    hits3 = top1 = answer_found = covered_ok = 0
    version_ok = version_n = 0
    failures = []
    for x in answerable:
        r = index.search(x["q"], k=k)
        docs = ranked = [res["doc_id"] for res in r["results"]]
        hit = x["doc"] in ranked[:k]
        hits3 += hit
        top1 += bool(docs) and docs[0] == x["doc"]
        covered_ok += r["covered"]
        text = " ".join(res["text"] for res in r["results"]).lower()
        answer_found += bool(r["covered"]) and x["answer"].lower() in text
        if x["kind"] in ("current", "old"):
            version_n += 1
            version_ok += bool(docs) and docs[0] == x["doc"]
        if not hit or not r["covered"]:
            failures.append({"q": x["q"], "want": x["doc"], "got": ranked, "covered": r["covered"]})
    not_covered_ok = 0
    for x in unanswerable:
        r = index.search(x["q"], k=k)
        not_covered_ok += not r["covered"]
        if r["covered"]:
            failures.append({"q": x["q"], "want": None, "got": [res["doc_id"] for res in r["results"]], "covered": True})
    n = len(answerable)
    return {
        "n_answerable": n,
        "n_not_covered": len(unanswerable),
        f"recall@{k}": round(hits3 / n, 3),
        "top1_accuracy": round(top1 / n, 3),
        "answer_in_context": round(answer_found / n, 3),
        "covered_when_answerable": round(covered_ok / n, 3),
        "current_policy_choice": round(version_ok / version_n, 3) if version_n else None,
        "not_covered_detection": round(not_covered_ok / len(unanswerable), 3) if unanswerable else None,
        "failures": failures,
    }

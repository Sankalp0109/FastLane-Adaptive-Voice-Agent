"""Build the LoRA training data (Table 5).

    fastlane train data --plans 700

Writes data/train/<adapter>/{train,val}.jsonl with chat-format examples ({"messages": [...]}: system,
user, previous steps of the same turn, and the target assistant JSON as the last message), plus
stats.json. Two adapters: "all_domains" (FastLane) and "movies_buses" (A8, no restaurant calls).
Train/val are split by call (not by example) so no call contributes to both.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from fastlane.config import DATA_DIR, SEED, SYSTEMS, WORLD_DB
from fastlane.retrieval.index import PolicyIndex
from fastlane.training.dialogues import Planner, run_plan
from fastlane.training.noise import noisy_user_content
from fastlane.world.db import World

TRAIN_DIR = DATA_DIR / "train"

KIND_WEIGHTS = {
    "book": 28, "dates": 12, "sold_out": 8, "change_mind": 10, "modify": 6, "cancel": 8, "policy": 10, "usual": 5,
    "returning_cancel": 3, "red_flag": 5, "rule_group": 2, "rule_deposit": 2, "rule_colleague": 2, "rule_no_pax": 2,
}
DOMAIN_ONLY = {"rule_deposit": "restaurants", "rule_no_pax": "buses"}
# Prompt variants the adapter is used with: tool lists and context differ, so it must see them all.
CONFIG_WEIGHTS = {"fastlane": 60, "A2": 15, "A4": 10, "A6": 5, "A7": 10}
NEEDS_HISTORY = {"usual", "returning_cancel"}


def generate(n_plans: int, seed: int = SEED + 9001, world_path: Path = WORLD_DB, progress=print) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    world = World.fresh(world_path)
    planner = Planner(world, seed=seed + 1)
    index = PolicyIndex()
    kinds, kw = zip(*KIND_WEIGHTS.items())
    configs, cw = zip(*CONFIG_WEIGHTS.items())
    calls, examples = [], []
    attempts = 0
    while len(calls) < n_plans and attempts < n_plans * 4:
        attempts += 1
        kind = rng.choices(kinds, kw)[0]
        domain = DOMAIN_ONLY.get(kind) or rng.choice(["movies", "buses", "restaurants"])
        cfg_name = rng.choices(configs, cw)[0]
        if kind in NEEDS_HISTORY and cfg_name == "A7":
            cfg_name = "fastlane"
        plan = planner.plan(domain, kind)
        if plan is None:
            continue
        result = run_plan(plan, SYSTEMS[cfg_name], world_path, index, seed=rng.randrange(1 << 30))
        calls.append({"plan": plan.plan_id, "domain": domain, "kind": kind, "config": cfg_name, "ok": result["ok"],
                      "error": result["error"], "turns": result["turns"], "examples": len(result["examples"])})
        examples += result["examples"]
        if len(calls) % 100 == 0:
            progress(f"{len(calls)} calls, {sum(c['ok'] for c in calls)} kept, {len(examples)} examples")
    return calls, examples


def split_and_write(calls: list[dict], examples: list[dict], out_dir: Path, domains: set[str], seed: int,
                    noise_share: float = 0.3, val_share: float = 0.1, route_upsample: int = 3) -> dict:
    rng = random.Random(seed)
    kept_plans = sorted({c["plan"] for c in calls if c["ok"] and c["domain"] in domains})
    rng.shuffle(kept_plans)
    n_val = max(1, int(len(kept_plans) * val_share))
    val_plans = set(kept_plans[:n_val])
    train, val = [], []
    for ex in examples:
        if ex["meta"]["domain"] not in domains:
            continue
        (val if ex["meta"]["plan"] in val_plans else train).append(ex)
    # Speech-noise copies of ~30% of training examples (caller lines mis-transcribed, targets unchanged).
    noisy = []
    for ex in train:
        if rng.random() < noise_share:
            msgs = [dict(m) for m in ex["messages"]]
            msgs[1]["content"] = noisy_user_content(msgs[1]["content"], rng)
            if msgs[1]["content"] != ex["messages"][1]["content"]:
                noisy.append({"messages": msgs, "meta": {**ex["meta"], "noisy": True}})
    train += noisy
    # Escalations are ~5% of steps; repeat them so the adapter does not learn to never pass a turn on.
    # Validation keeps the natural mix so its numbers stay honest.
    train += [ex for ex in train if ex["meta"]["route"] != "self" for _ in range(route_upsample - 1)]
    rng.shuffle(train)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    stats = {
        "calls_kept": len(kept_plans), "val_calls": n_val, "train_examples": len(train), "val_examples": len(val),
        "noisy_copies": len(noisy),
        "routes_train": dict(Counter(e["meta"]["route"] for e in train)),
        "kinds_train": dict(Counter(e["meta"]["kind"] for e in train)),
        "configs_train": dict(Counter(e["meta"]["config"] for e in train)),
        "domains_train": dict(Counter(e["meta"]["domain"] for e in train)),
        "approx_tokens_per_example": round(sum(len(json.dumps(e["messages"])) for e in train) / max(1, len(train)) / 4),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def build(n_plans: int = 700, seed: int = SEED + 9001) -> dict:
    calls, examples = generate(n_plans, seed)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    (TRAIN_DIR / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n")
    report = {
        "calls": len(calls), "kept": sum(c["ok"] for c in calls),
        "kept_by_kind": {k: f"{sum(c['ok'] for c in calls if c['kind'] == k)}/{sum(1 for c in calls if c['kind'] == k)}" for k in KIND_WEIGHTS},
        "all_domains": split_and_write(calls, examples, TRAIN_DIR / "all_domains", {"movies", "buses", "restaurants"}, seed),
        "movies_buses": split_and_write(calls, examples, TRAIN_DIR / "movies_buses", {"movies", "buses"}, seed),
    }
    (TRAIN_DIR / "report.json").write_text(json.dumps(report, indent=2))
    return report

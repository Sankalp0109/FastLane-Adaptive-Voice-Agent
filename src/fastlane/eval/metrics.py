"""Aggregate per-call records into the Table 7 metrics, per system."""

from __future__ import annotations

import math
from collections import defaultdict

from fastlane.config import SYSTEMS

# USD per 1M tokens (input, output), for the cost estimate. Local models cost 0.
# Fill in the provider's current list price before reporting costs; free-tier use is billed at 0.
PRICES: dict[str, tuple[float, float]] = {}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(centre - half, 3), round(centre + half, 3))


def _rate(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


def summarize_system(rows: list[dict]) -> dict:
    system = rows[0]["system"]
    cfg = SYSTEMS.get(system)
    n = len(rows)
    succ = sum(bool(r["success"]) for r in rows)
    turns = [t for r in rows for t in r["turns"]]
    model_ms = [t["model_ms"] for t in turns]
    lookup_ms = [t["model_ms"] + sum(x["ms"] for x in t["tools"]) for t in turns if any(x["tool"] == "search_policy" for x in t["tools"]) or t.get("policy_prefetched")]
    large_turns = [t for t in turns if any(c["role"] in ("large", "router") for c in t["model_calls"])]
    large_calls = sum(1 for t in turns for c in t["model_calls"] if c["role"] in ("large", "router"))
    tokens: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    cost = 0.0
    throttle_s = 0.0
    for t in turns:
        for c in t["model_calls"]:
            tokens[c["role"]][0] += c["prompt_tokens"]
            tokens[c["role"]][1] += c["completion_tokens"]
            pin, pout = PRICES.get(c["model"], (0.0, 0.0))
            cost += (c["prompt_tokens"] * pin + c["completion_tokens"] * pout) / 1e6
            throttle_s += c["throttle_ms"] / 1000

    out = {
        "system": system,
        "description": cfg.description if cfg else "",
        "calls": n,
        "success": _rate(succ, n),
        "success_ci95": wilson(succ, n),
        "turns": len(turns),
        "model_ms_p50": percentile(model_ms, 0.5),
        "model_ms_p95": percentile(model_ms, 0.95),
        "lookup_turn_ms_p95": percentile(lookup_ms, 0.95),
        "large_turn_rate": _rate(len(large_turns), len(turns)),
        "large_calls_per_call": round(large_calls / n, 2) if n else None,
        "tokens_per_call": {role: [round(a / n), round(b / n)] for role, (a, b) in tokens.items()},
        "cost_per_call_usd": round(cost / n, 5) if n else None,
        "throttle_s_per_call": round(throttle_s / n, 1) if n else None,
        "errors": sum(1 for r in rows if r.get("error")),
    }

    # Router: predicted hard = the turn went to the large model or staff.
    if cfg is None or cfg.router in ("small", "large_prompt"):
        tp = sum(1 for t in turns if t["hard"] and t["handled_by"] in ("large", "staff"))
        fp = sum(1 for t in turns if not t["hard"] and t["handled_by"] in ("large", "staff"))
        fn = sum(1 for t in turns if t["hard"] and t["handled_by"] not in ("large", "staff"))
        out["router_precision"] = _rate(tp, tp + fp)
        out["router_hard_recall"] = _rate(tp, tp + fn)
        out["hard_turns"] = tp + fn
        reasons: dict[str, int] = defaultdict(int)
        for t in turns:
            if t["escalation_reason"]:
                reasons[t["escalation_reason"].split(":")[0]] += 1
        out["escalation_reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:8])

    # Safety.
    writes = [w for r in rows for w in r["writes"]]
    out["unsafe_write_calls"] = _rate(sum(1 for r in rows if r["unsafe_writes"]), n)
    out["unconfirmed_writes"] = sum(1 for w in writes if not w["confirmed"])
    out["rule_violating_writes"] = sum(1 for w in writes if w["violations"])
    red = [r for r in rows if r["category"] == "red_flag"]
    out["missed_handoffs"] = f"{sum(1 for r in red if not r['handoff'])}/{len(red)}"

    # Tool use.
    tool_calls = sum(r["tool_calls"] for r in rows)
    invalid = sum(r["tool_errors"].get("invalid_args", 0) + r["tool_errors"].get("unknown_tool", 0) for r in rows)
    out["invalid_tool_call_rate"] = _rate(invalid, tool_calls)
    injected = [r for r in rows if r.get("injected")]
    out["injected_error_success"] = _rate(sum(bool(r["success"]) for r in injected), len(injected))

    # Retrieval and policy answers.
    el = [r["early_lookup"] for r in rows]
    out["early_lookup"] = {
        "started": sum(e["started"] for e in el), "reused": sum(e["reused"] for e in el),
        "wasted": sum(e["wasted"] for e in el), "cache_hits": sum(e["cache_hits"] for e in el),
        "saved_ms_total": round(sum(e["saved_ms"] for e in el), 1),
    }
    answered = [r for r in rows if r.get("answer_ok") is not None]
    out["policy_answer_accuracy"] = _rate(sum(bool(r["answer_ok"]) for r in answered), len(answered))

    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        by_cat[r["category"]][0] += bool(r["success"])
        by_cat[r["category"]][1] += 1
    out["success_by_category"] = {k: f"{a}/{b}" for k, (a, b) in sorted(by_cat.items())}
    by_dom: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        by_dom[r["domain"]][0] += bool(r["success"])
        by_dom[r["domain"]][1] += 1
    out["success_by_domain"] = {k: f"{a}/{b}" for k, (a, b) in sorted(by_dom.items())}
    return out


def summarize(rows: list[dict]) -> list[dict]:
    by_system: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_system[r["system"]].append(r)
    order = list(SYSTEMS)
    summaries = [summarize_system(by_system[s]) for s in sorted(by_system, key=lambda s: order.index(s) if s in order else 99)]
    base = next((s for s in summaries if s["system"] == "B1"), None)
    if base and base["model_ms_p95"]:
        for s in summaries:
            if s["model_ms_p95"] is not None:
                s["p95_vs_B1"] = round(1 - s["model_ms_p95"] / base["model_ms_p95"], 3)
            if s["success"] is not None and base["success"] is not None:
                s["success_vs_B1"] = round(s["success"] - base["success"], 3)
    return summaries

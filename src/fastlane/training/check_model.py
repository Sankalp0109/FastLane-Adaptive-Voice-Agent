"""Offline step accuracy of a small model on held-out validation steps.

    fastlane train check --model ollama:fastlane-qwen2.5-3b
    fastlane train check --model ollama:qwen2.5:3b --full-prompt     # untuned baseline (A1-style prompt)
    fastlane train check --rescore results/adapter_v2/rows_v2_all267.jsonl   # re-score saved rows, no model calls

Each validation example is the exact input the small model saw in a generated call; we compare the model's
JSON with the target: parses, route, tool name, tool arguments (exact), and latency. Escalation precision and
recall come from the route (self vs llm/staff). No large model or simulated caller is involved.

Dates count as equal when they name the same day ("October 24" and "2026-10-24"): the gateway resolves both
to the same search, so the difference does not change what the agent does.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from datetime import date
from pathlib import Path

from fastlane.config import SYSTEMS, TODAY
from fastlane.gateway.dates import resolve_date
from fastlane.models import prompts
from fastlane.models.parsing import escalation_score, parse_small
from fastlane.models.registry import client_for

CANON = {"date", "time", "quantity", "party_size", "domain", "option_id", "booking_id", "name", "phone", "movie", "cinema",
         "seat_category", "origin", "destination", "restaurant", "expression", "query", "passengers"}


def _day(text: str) -> str | None:
    try:
        return date.fromisoformat(text.strip()).isoformat()
    except ValueError:
        r = resolve_date(text, TODAY)
        return r.date.isoformat() if r.date else None


def _args_equal(got: dict, want: dict) -> bool:
    """Exact on the arguments that decide the action; ignores extra optional keys the model adds."""
    for k, v in want.items():
        if k not in CANON:
            continue
        g = got.get(k)
        if k == "date" and isinstance(v, str) and isinstance(g, str):
            if _day(v) is None or _day(v) != _day(g):
                return False
        elif isinstance(v, str) and isinstance(g, str):
            if v.strip().lower() != g.strip().lower():
                return False
        elif isinstance(v, (int, float)) and isinstance(g, (int, float, str)):
            try:
                if float(g) != float(v):
                    return False
            except ValueError:
                return False
        elif v != g:
            return False
    return True


def _full_prompt(example: dict) -> list[dict]:
    """Swap the compact system prompt for the full instruction prompt (for the untuned baseline)."""
    msgs = [dict(m) for m in example["messages"][:-1]]
    tools_block = msgs[0]["content"].split("Tools:\n", 1)[1].split("\n\nReply with", 1)[0]
    cfg = SYSTEMS["A1"]
    specs_text = prompts.small_system(cfg, [])  # instruction text without tools
    head, tail = specs_text.split("Tools:\n", 1)
    msgs[0]["content"] = head + "Tools:\n" + tools_block + tail[tail.find("\n\n"):]
    return msgs


def check(model: str, val_path: Path, n: int | None = None, full_prompt: bool = False, seed: int = 0,
          save_rows: Path | None = None, where=None) -> dict:
    """where: optional filter on example meta (e.g. lambda m: m["domain"] == "restaurants")."""
    rows = [json.loads(l) for l in val_path.read_text().splitlines() if l.strip()]
    if where is not None:
        rows = [r for r in rows if where(r["meta"])]
    if n and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    client = client_for(model)
    per_row = []
    for ex in rows:
        target = parse_small(ex["messages"][-1]["content"])
        msgs = _full_prompt(ex) if full_prompt else ex["messages"][:-1]
        r = client.chat(msgs, max_tokens=350, json_mode=True, logprobs=True)
        got = parse_small(r.text)
        per_row.append({"meta": ex["meta"], "parsed": got.ok, "want_route": target.route, "got_route": got.route,
                        "want_tool": target.tool, "got_tool": got.tool, "latency_ms": round(r.latency_ms, 1),
                        "escalation_score": round(escalation_score(r, got), 3),
                        "want_date_slot": target.slots.get("date"), "got_date_slot": got.slots.get("date")})
    if save_rows is not None:
        save_rows.parent.mkdir(parents=True, exist_ok=True)
        save_rows.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in per_row) + "\n")
    return summarise(per_row, model, "full" if full_prompt else "compact")


def summarise(per_row: list[dict], model: str = "", prompt: str = "") -> dict:
    """Metrics from per-step rows (fresh from check() or loaded from a --save-rows file)."""
    confusion: Counter = Counter()
    stats: Counter = Counter()
    misses = []
    for x in per_row:
        want_route, got_route, want_tool, got_tool = x["want_route"], x["got_route"], x["want_tool"], x["got_tool"]
        confusion[(want_route, got_route)] += 1
        stats["route_ok"] += got_route == want_route
        stats["parsed"] += x.get("parsed", True)
        x["args_ok"] = None
        if want_route == "self":
            stats["self_steps"] += 1
            stats["tool_name_ok"] += (want_tool or {}).get("name") == (got_tool or {}).get("name")
            if want_tool and got_tool and got_tool["name"] == want_tool["name"]:
                x["args_ok"] = _args_equal(got_tool["args"], want_tool["args"])
                stats["tool_steps"] += 1
                stats["tool_args_ok"] += x["args_ok"]
                if not x["args_ok"] and len(misses) < 8:
                    misses.append({"want": want_tool, "got": got_tool})
    esc = lambda r: r != "self"  # noqa: E731
    tp = sum(v for (t, g), v in confusion.items() if esc(t) and esc(g))
    fp = sum(v for (t, g), v in confusion.items() if not esc(t) and esc(g))
    fn = sum(v for (t, g), v in confusion.items() if esc(t) and not esc(g))
    n = len(per_row)
    lat = sorted(x["latency_ms"] for x in per_row)
    return {
        "model": model, "prompt": prompt, "examples": n,
        "json_ok": round(stats["parsed"] / n, 3),
        "route_accuracy": round(stats["route_ok"] / n, 3),
        "escalation_precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "escalation_recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "tool_choice_accuracy": round(stats["tool_name_ok"] / stats["self_steps"], 3) if stats["self_steps"] else None,
        "tool_args_exact": round(stats["tool_args_ok"] / stats["tool_steps"], 3) if stats["tool_steps"] else None,
        "latency_ms_p50": round(statistics.median(lat)), "latency_ms_p95": round(lat[int(0.95 * (n - 1))]),
        "route_confusion": {f"{t}->{g}": v for (t, g), v in sorted(confusion.items())},
        "arg_misses": misses,
    }

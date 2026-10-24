"""Parse the small model's JSON turn and compute its escalation score."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from fastlane.models.base import LLMResponse

ROUTES = ("self", "llm", "staff")


@dataclass
class SmallTurn:
    route: str = "self"
    reason: str = ""
    slots: dict[str, Any] = field(default_factory=dict)
    tool: dict | None = None  # {"name": ..., "args": {...}}
    say: str = ""
    ok: bool = True  # False when the output could not be parsed
    raw: str = ""


def _first_json_object(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_small(text: str | None) -> SmallTurn:
    raw = text or ""
    obj = _first_json_object(raw)
    if obj is None:
        # A bare escalation keyword is also accepted (the proposal's ESCALATE_LLM / ESCALATE_STAFF).
        if "ESCALATE_STAFF" in raw:
            return SmallTurn(route="staff", reason="escalate_staff", raw=raw)
        if "ESCALATE_LLM" in raw:
            return SmallTurn(route="llm", reason="escalate_llm", raw=raw)
        return SmallTurn(route="llm", reason="unparseable output", ok=False, raw=raw)
    route = str(obj.get("route", "self")).lower().strip()
    route = {"escalate_llm": "llm", "large": "llm", "escalate_staff": "staff", "human": "staff"}.get(route, route)
    if route not in ROUTES:
        route = "llm"
    tool = obj.get("tool")
    if isinstance(tool, dict) and tool.get("name"):
        args = tool.get("args") or tool.get("arguments") or {}
        tool = {"name": str(tool["name"]), "args": args if isinstance(args, dict) else {}}
    else:
        tool = None
    slots = obj.get("slots") if isinstance(obj.get("slots"), dict) else {}
    return SmallTurn(route=route, reason=str(obj.get("reason") or ""), slots=slots, tool=tool,
                     say=str(obj.get("say") or "").strip(), raw=raw)


def escalation_score(resp: LLMResponse, parsed: SmallTurn) -> float:
    """P(route != self) from the log-probabilities at the route value, when available.

    The route value is the first word token after '"route": "'. Its top alternatives give the
    probabilities of self/llm/staff. Without log-probabilities the score is 0 or 1.
    """
    hard = 0.0 if parsed.route == "self" else 1.0
    if not resp.top_logprobs:
        return hard
    for alts in resp.top_logprobs:
        probs = {"self": 0.0, "llm": 0.0, "staff": 0.0}
        for token, lp in alts.items():
            t = token.strip().strip('"').lower()
            if not t:
                continue
            for route in probs:
                if route.startswith(t) or t.startswith(route):
                    probs[route] += math.exp(lp)
                    break
        total = sum(probs.values())
        if total > 0.5:  # this position is the route value
            return round((probs["llm"] + probs["staff"]) / total, 4)
    return hard

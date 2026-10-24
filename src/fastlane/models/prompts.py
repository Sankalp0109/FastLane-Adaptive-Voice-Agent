"""System prompts and context builders for the small model, the large model and the B3 router.

The small-model prompt is also the prompt used in the LoRA training data (Phase 6), so the tuned
model sees at inference exactly the format it was trained on. Keep it short: every token adds
latency on the small model.
"""

from __future__ import annotations

import json

from fastlane.config import NOW, SystemConfig

WORLD_LINE = (
    "You are FastLane, an automated phone booking assistant for four cinemas in Hyderabad (Galaxy Kukatpally, "
    "Starlight Gachibowli, Regal Ameerpet, Moonbeam Secunderabad), buses between Hyderabad and Bengaluru, "
    "Vijayawada and Chennai, and the three Tandoor Tales restaurants (Jubilee Hills, Banjara Hills, Hitech City). "
    f"Now: {NOW.strftime('%A %d %B %Y, %H:%M')}."
)

STYLE = (
    "Speak like a polite Indian call-centre agent: short, clear sentences (one or two), ask for one missing "
    "detail at a time, say prices in rupees. Never invent shows, buses, tables, prices or policies: use tools."
)

RULES = """Booking rules:
- Movies: at most 10 tickets per booking; changes and cancellations only up to 4 hours before the show; Rs 30 per ticket cancellation fee.
- Buses: at most 6 seats; name, age and gender for every passenger; changes up to 12 hours before departure; refund 90% (72h+), 75% (24-72h), 50% (12-24h), 25% (4-12h), no cancellation within 4 hours.
- Restaurants: party limits Jubilee Hills 12, Banjara Hills 10, Hitech City 8; deposit Rs 300/guest for 8+ (Rs 250/guest for 6+ at Hitech City), never waived by you; changes and cancellations up to 2 hours before; Banjara Hills closed Mondays.
- Before any booking, change or cancellation, read back the full details and get a clear yes from the caller.
- Changes and cancellations need the phone number used for the booking; never act on someone else's booking.
- Payments, refund disputes, complaints, out-of-scope requests and groups above limits go to staff."""

GATEWAY_NOTE = (
    "The booking system enforces the rules: a create/modify/cancel first returns a 'readback' that must be read to "
    "the caller word for word; call the same tool again with the same arguments after the caller says yes. "
    "Never write your own read-back or confirmation: when the caller has chosen an option, call the tool to get the read-back."
)

SMALL_FORMAT = """Reply with ONE JSON object and nothing else:
{"route": "self" | "llm" | "staff", "reason": "<why, if not self>", "slots": {<details the caller gave this turn>}, "tool": {"name": "<tool>", "args": {...}} or null, "say": "<words to speak, or empty when calling a tool>"}
route "llm" = hard turn for the expert model: unclear dates, sold-out or no availability, caller changed an earlier choice, policy or menu questions, anything you are unsure about.
route "staff" = disputes about money already paid (double charge, missing refund), complaints, out-of-scope requests, group above limits, deposit waiver, identity cannot be verified, caller asks for a human.
Questions about refund RULES or fees ("how much refund if I cancel...") are policy questions: use search_policy or route "llm", not "staff".
slots keys: domain, intent, caller_name, date, time, quantity, movie, cinema, seat_category, origin, destination, bus_type, berth_preference, passengers, restaurant, party_size, special_request, booking_id.
"tool" must be one of the tools listed above, or null. After a tool call you will see its result; then output the next JSON.
When a create/modify/cancel result contains "readback", that is TEXT TO SPEAK, not a tool: put it in "say" word for word with "tool": null."""


def compact_tools(specs: list[dict]) -> str:
    """One line per tool: name(args) - description. Much shorter than JSON schemas."""
    lines = []
    for s in specs:
        props = s["input_schema"].get("properties", {})
        required = set(s["input_schema"].get("required", []))
        args = ", ".join(f"{k}{'' if k in required else '?'}" for k in props)
        lines.append(f"- {s['name']}({args}): {s['description']}")
    return "\n".join(lines)


COMPACT_FORMAT = ('Reply with one JSON object: {"route": "self"|"llm"|"staff", "reason": "", "slots": {}, '
                  '"tool": {"name": "", "args": {}} or null, "say": ""}.')


def small_system(cfg: SystemConfig, specs: list[dict], compact: bool = False) -> str:
    """compact=True: the short prompt the LoRA model is trained with (it has learned the format and
    the routing rules, so only the world, the tools and the output shape are needed)."""
    if compact:
        return "\n\n".join([WORLD_LINE, "Tools:\n" + compact_tools(specs), COMPACT_FORMAT])
    parts = [WORLD_LINE, STYLE]
    if cfg.rules_in_prompt:
        parts.append(RULES)
    if cfg.enforce_rules:
        parts.append(GATEWAY_NOTE)
    parts += ["Tools:\n" + compact_tools(specs), SMALL_FORMAT]
    return "\n\n".join(parts)


def large_system(cfg: SystemConfig) -> str:
    parts = [WORLD_LINE, STYLE]
    if cfg.rules_in_prompt:
        parts.append(RULES)
    if cfg.enforce_rules:
        parts.append(GATEWAY_NOTE)
    if cfg.large_reasoning:
        parts.append("Before you reply, work through the steps: what the caller wants, which facts you still need "
                     "(use tools), which rules apply, and what the best available option is.")
    else:
        parts.append("Reply directly without long deliberation.")
    if cfg.router == "small":
        parts.append("You are handling one hard turn of a call that a smaller assistant is running. Finish this turn "
                     "(use tools as needed) and reply to the caller; control then returns to the smaller assistant.")
    parts.append("If the caller needs a human (payments, complaints, out-of-scope, limits), call handoff_to_staff with a short summary.")
    return "\n\n".join(parts)


ROUTER_SYSTEM = (
    "You route turns of a booking phone call. Answer EASY if a small assistant can handle the caller's latest turn "
    "(greetings, giving a detail, simple yes/no, simple search or booking). Answer HARD for relative or festival dates, "
    "sold-out or alternatives, changed choices, policy/menu/refund questions, rule-breaking or complaints, or anything "
    "needing several steps. Reply with one word: EASY or HARD."
)


def context_block(record_text: str | None, history: dict | None, policy: dict | None) -> str:
    parts = []
    if record_text is not None:
        parts.append("BOOKING RECORD:\n" + record_text)
    if history and history.get("returning_caller"):
        h = {k: history[k] for k in ("name", "usual") if k in history}
        h["recent"] = history.get("recent", [])[:3]
        h["upcoming"] = [
            {k: b.get(k) for k in ("booking_id", "domain", "movie", "cinema", "route_id", "restaurant", "spoken_date", "spoken_time", "quantity", "party_size")
             if b.get(k) is not None}
            for b in history.get("upcoming", [])
        ]
        parts.append("CALLER HISTORY (calling number):\n" + json.dumps(h, ensure_ascii=False))
    if policy and policy.get("results"):
        top = policy["results"][0]
        tag = "" if policy.get("covered") else " (weak match - may not answer the question)"
        parts.append(f"POSSIBLY RELEVANT POLICY{tag}: [{top['doc']} / {top['section']}] {top['text'][:700]}")
    return "\n\n".join(parts)


def transcript(turns: list[tuple[str, str]], limit: int | None) -> str:
    """turns: [(speaker, text)], speaker in {'Caller', 'Agent'}; keep the last `limit` lines."""
    selected = turns if limit is None else turns[-limit:]
    return "\n".join(f"{who}: {text}" for who, text in selected)


def compact_result(result: dict, max_chars: int = 1600) -> str:
    text = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= max_chars else text[:max_chars] + "...(truncated)"

"""Task success: does the final booking database match the script's gold state? (exact field match)

The set of bookings the call created or changed (across the whole database, so touching someone
else's booking is caught) must match one of the script's acceptable outcomes one-to-one. Hand-off
and policy-answer expectations must also hold.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any

from fastlane.world.db import World


def booking_views(world: World) -> dict[str, dict]:
    return {r["id"]: world.describe_booking(r["id"]) for r in world.q("SELECT id FROM bookings")}


def changed_bookings(before: dict[str, dict], after: dict[str, dict]) -> list[dict]:
    return [v for bid, v in sorted(after.items()) if bid not in before or before[bid] != v]


def field_matches(key: str, expected: Any, view: dict) -> bool:
    if key == "passenger_first_names":
        got = sorted(str(p.get("name") or "").split()[0].lower() for p in view.get("passengers", []) if p.get("name"))
        return got == expected
    actual = view.get(key)
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def view_matches(expected: dict, view: dict) -> bool:
    return all(field_matches(k, v, view) for k, v in expected.items())


def outcome_matches(outcome: list[dict], changed: list[dict]) -> bool:
    if len(outcome) != len(changed):
        return False
    if not outcome:
        return True
    return any(all(view_matches(e, v) for e, v in zip(outcome, perm)) for perm in permutations(changed))


def mismatch_report(outcome: list[dict], changed: list[dict]) -> list[str]:
    """Human-readable differences against the first outcome (for debugging failed calls)."""
    notes = []
    if len(outcome) != len(changed):
        notes.append(f"expected {len(outcome)} changed booking(s), got {len(changed)}")
    for e in outcome:
        best = min(changed, key=lambda v: sum(not field_matches(k, x, v) for k, x in e.items()), default=None)
        if best is None:
            continue
        diffs = [f"{k}: want {x!r} got {best.get(k) if k != 'passenger_first_names' else '...'}" for k, x in e.items() if not field_matches(k, x, best)]
        if diffs:
            notes.append("; ".join(diffs))
    return notes


_SPACES = dict.fromkeys(map(ord, "\u00a0\u2007\u2009\u200a\u202f\u205f\u3000"), " ")
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")


def normalise_text(text: str) -> str:
    """LLMs write '4\u202fhours' and 'non\u2011AC'; compare on plain spaces and hyphens."""
    return " ".join(text.translate(_SPACES).translate(_DASHES).replace("₹", "rs ").lower().split())


def score_call(gold: dict, changed: list[dict], agent_text: str, handoff: dict | None) -> dict:
    outcomes = gold.get("outcomes", [[]])
    outcome_ok = any(outcome_matches(o, changed) for o in outcomes)
    want_handoff = gold.get("handoff")
    handoff_ok = want_handoff is None or bool(handoff) == want_handoff
    keys = gold.get("answer_any") or []
    text = normalise_text(agent_text)
    answer_ok = not keys or any(normalise_text(k) in text for k in keys)
    return {
        "success": outcome_ok and handoff_ok and answer_ok,
        "outcome_ok": outcome_ok,
        "handoff_ok": handoff_ok,
        "answer_ok": answer_ok if keys else None,
        "mismatch": [] if outcome_ok else mismatch_report(outcomes[0], changed),
    }

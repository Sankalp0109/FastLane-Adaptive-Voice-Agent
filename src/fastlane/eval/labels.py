"""Rule-based easy/hard labels for caller turns.

Used for router precision/recall in evaluation and for the routing targets of the training data
(Phase 6). A turn is hard if what the caller says needs work beyond a lookup and a template:
relative/festival dates, a changed choice, a policy question, a red flag or a rule-breaking
request, or if the turn ran into no availability.
"""

from __future__ import annotations

import re

HARD_MOMENTS = {"relative_date", "festival_date", "change_mind", "policy_question", "red_flag", "rule_break", "sold_out"}

PATTERNS = {
    "relative_date": r"\b(next|coming|following)\s+(mon|tues|wednes|thurs|fri|satur|sun)day\b|\b(this|next)\s+weekend\b|\bday after tomorrow\b|\bweek from\b|\bin \d+ days\b",
    "festival_date": r"\b(diwali|deepavali|dussehra|dasara|dussera|bathukamma|dhanteras|bhai dooj|karva chauth|christmas|naraka)\b",
    "change_mind": r"\b(actually|instead|change (it|that|to)|make it|rather|on second thought|no wait|switch to)\b",
    "policy_question": r"\b(refund|cancellation fee|allowed|can i (bring|take|carry)|deposit|policy|halal|allerg|nut[- ]free|nuts|luggage|parking|wheelchair|3d|alcohol|open on|closed on|jain|pets?|dog|certificate|how much is)\b",
    "red_flag": r"\b(charged twice|overcharged|complain|complaint|rude|manager|human|real person|lost (my|a)|left my|flight|train tickets?|hotel|wedding|charter|private screening|money back|not arrived|hair in|dangerous)\b|\b(1[0-9]|[2-9][0-9])\s+(people|seats|tickets|guests)\b",
    "rule_break": r"\b(don'?t ask for (any|a) deposit|skip the deposit|no deposit|colleague'?s|my friend'?s booking|full refund|all in one booking|sort out the rest|tell the conductor)\b",
}
_COMPILED = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}


def text_moments(text: str) -> set[str]:
    return {k for k, rx in _COMPILED.items() if rx.search(text)}


def is_hard(text: str, sim_moments: list[str] | None = None, no_availability: bool = False) -> tuple[bool, list[str]]:
    moments = text_moments(text) | (set(sim_moments or []) & HARD_MOMENTS)
    if no_availability:
        moments.add("sold_out")
    return bool(moments), sorted(moments)

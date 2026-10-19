"""Global constants and the system presets compared in the experiments (Table 6)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Literal

# The practice world has a fixed "today" so relative dates ("next Friday") have one right answer.
TODAY = date(2026, 10, 15)  # Thursday; Dussehra (Oct 20) and Diwali (Nov 8) fall in the window
NOW = datetime(2026, 10, 15, 11, 0)  # calls happen at 11:00 IST on TODAY
BOOKING_WINDOW_DAYS = 30  # inventory exists for TODAY .. TODAY + 29

# Festival calendar of the practice world (also published in docs/festival_calendar.md).
# Keys are canonical names; FESTIVAL_ALIASES maps spoken variants onto them.
FESTIVALS: dict[str, date] = {
    "Saddula Bathukamma": date(2026, 10, 18),
    "Dussehra": date(2026, 10, 20),
    "Karva Chauth": date(2026, 10, 29),
    "Dhanteras": date(2026, 11, 6),
    "Naraka Chaturdashi": date(2026, 11, 7),
    "Diwali": date(2026, 11, 8),
    "Govardhan Puja": date(2026, 11, 9),
    "Bhai Dooj": date(2026, 11, 10),
    "Children's Day": date(2026, 11, 14),
    "Christmas": date(2026, 12, 25),
}
FESTIVAL_ALIASES: dict[str, str] = {
    "bathukamma": "Saddula Bathukamma",
    "dasara": "Dussehra",
    "dussera": "Dussehra",
    "vijayadashami": "Dussehra",
    "vijaya dashami": "Dussehra",
    "deepavali": "Diwali",
    "divali": "Diwali",
    "lakshmi puja": "Diwali",
    "laxmi puja": "Diwali",
    "choti diwali": "Naraka Chaturdashi",
    "chhoti diwali": "Naraka Chaturdashi",
    "naraka chaturdasi": "Naraka Chaturdashi",
    "bhaiya dooj": "Bhai Dooj",
    "bhau beej": "Bhai Dooj",
    "karwa chauth": "Karva Chauth",
    "xmas": "Christmas",
}
HISTORY_DAYS = 365  # synthetic past bookings span the year before TODAY
CITY = "Hyderabad"
SEED = 2026

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
WORLD_DB = DATA_DIR / "world.db"
DOCS_DIR = PACKAGE_DIR / "retrieval" / "docs"

Domain = Literal["movies", "buses", "restaurants"]
DOMAINS: tuple[Domain, ...] = ("movies", "buses", "restaurants")


@dataclass(frozen=True)
class SystemConfig:
    """One agent configuration. Every baseline/variant is FastLane with flags changed."""

    name: str
    description: str
    # Which models exist and who decides the route for each turn.
    small_model: Literal["lora", "prompted", "none"] = "lora"
    large_model: bool = True
    router: Literal["small", "large_prompt", "always_small", "always_large"] = "small"
    adapter: Literal["all_domains", "movies_buses"] = "all_domains"
    # Gateway and tools.
    date_tools: bool = True  # resolve_date + find_alternatives available (A2 turns off)
    enforce_rules: bool = True  # rules + read-back checked in code (A3 turns off)
    rules_in_prompt: bool = False  # booking rules written into the system prompt
    # Context and retrieval.
    booking_record: bool = True  # structured record shared between models (A4 turns off)
    large_reasoning: bool = True  # step-by-step reasoning on hard turns (A5 turns off)
    early_lookup: bool = True  # policy lookup starts on partial transcripts (A6 turns off)
    caller_history: bool = True  # past bookings of returning callers (A7 turns off)
    # Speed-ups (Section 6).
    templates: bool = True  # template read-backs and goodbyes
    tool_retry: bool = True  # one retry on a failed tool call
    tags: tuple[str, ...] = field(default_factory=tuple)


FASTLANE = SystemConfig(
    name="fastlane",
    description="Fine-tuned small model answers easy turns and passes hard ones to the large model or staff.",
    tags=("ours",),
)

SYSTEMS: dict[str, SystemConfig] = {
    "fastlane": FASTLANE,
    "B1": replace(
        FASTLANE,
        name="B1",
        description="Large model only; rules in the prompt.",
        small_model="none",
        router="always_large",
        enforce_rules=False,
        rules_in_prompt=True,
        templates=False,
        tags=("baseline",),
    ),
    "B2": replace(
        FASTLANE,
        name="B2",
        description="Small model only, no fine-tuning; large model never used.",
        small_model="prompted",
        large_model=False,
        router="always_small",
        rules_in_prompt=True,
        tags=("baseline",),
    ),
    "B3": replace(
        FASTLANE,
        name="B3",
        description="Large model prompted to route each turn easy/hard before it is answered.",
        router="large_prompt",
        tags=("baseline",),
    ),
    "A1": replace(FASTLANE, name="A1", description="No fine-tuning; small model prompted.", small_model="prompted", rules_in_prompt=True, tags=("variant",)),
    "A2": replace(FASTLANE, name="A2", description="No date and alternative tools.", date_tools=False, tags=("variant",)),
    "A3": replace(FASTLANE, name="A3", description="No safety checks; rules only in the prompt, no read-back.", enforce_rules=False, rules_in_prompt=True, tags=("variant",)),
    "A4": replace(FASTLANE, name="A4", description="No booking record; models see chat history only.", booking_record=False, tags=("variant",)),
    "A5": replace(FASTLANE, name="A5", description="No step-by-step reasoning on hard turns.", large_reasoning=False, tags=("variant",)),
    "A6": replace(FASTLANE, name="A6", description="No early lookup.", early_lookup=False, tags=("variant",)),
    "A7": replace(FASTLANE, name="A7", description="No caller history.", caller_history=False, tags=("variant",)),
    "A8": replace(FASTLANE, name="A8", description="Adapter trained on movies and buses only; tested on restaurants.", adapter="movies_buses", tags=("variant",)),
}


def get_system(name: str) -> SystemConfig:
    try:
        return SYSTEMS[name]
    except KeyError:
        raise KeyError(f"unknown system {name!r}; choose from {', '.join(SYSTEMS)}") from None

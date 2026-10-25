"""Code tools for relative dates and spoken times (the `resolve_date` tool).

Conventions of the practice world (also in docs/booking_faq.md):
* "this Friday" / "coming Friday" / "Friday" = the next Friday strictly after today.
* "next Friday" = the Friday of next calendar week (weeks run Monday-Sunday). When that
  differs from the coming Friday the result is flagged ambiguous so the read-back can confirm it.
* "this weekend" = the coming Saturday (Sunday given as an alternative).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from fastlane.config import BOOKING_WINDOW_DAYS, FESTIVAL_ALIASES, FESTIVALS

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAY_ABBR = {w[:3]: i for i, w in enumerate(WEEKDAYS)} | {"tues": 1, "wed": 2, "thur": 3, "thurs": 3}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
_MONTH_ABBR = {m[:3]: i + 1 for i, m in enumerate(MONTHS)} | {"sept": 9}

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty five": 45, "forty": 40, "fifty": 50, "a": 1, "an": 1, "couple of": 2,
}
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
    "thirtieth": 30,
}


@dataclass
class ResolvedDate:
    date: date | None
    expression: str
    explanation: str
    ambiguous: bool = False
    alternatives: list[date] = field(default_factory=list)
    in_window: bool = True
    time_hint: str | None = None  # e.g. "evening" from "tonight"

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat() if self.date else None,
            "weekday": self.date.strftime("%A") if self.date else None,
            "spoken": spoken_date(self.date) if self.date else None,
            "explanation": self.explanation,
            "ambiguous": self.ambiguous,
            "alternatives": [d.isoformat() for d in self.alternatives],
            "in_booking_window": self.in_window,
            "time_hint": self.time_hint,
        }


def spoken_date(d: date) -> str:
    """'Friday, 16 October' - the form used in read-backs."""
    return f"{d.strftime('%A')}, {d.day} {d.strftime('%B')}"


def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"[,.!?]", " ", text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", text)
    # "twenty third" -> "23", "thirty first" -> "31"
    def tens_ordinal(m: re.Match) -> str:
        return str(_NUMBER_WORDS[m.group(1)] + _ORDINAL_WORDS[m.group(2)])

    text = re.sub(r"\b(twenty|thirty)[\s-]+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)\b", tens_ordinal, text)
    for word, n in _ORDINAL_WORDS.items():
        text = re.sub(rf"\b{word}\b", str(n), text)
    return re.sub(r"\s+", " ", text).strip()


def _weekday_index(token: str) -> int | None:
    token = token.lower()
    if token in WEEKDAYS:
        return WEEKDAYS.index(token)
    return _WEEKDAY_ABBR.get(token)


def _month_index(token: str) -> int | None:
    token = token.lower()
    if token in MONTHS:
        return MONTHS.index(token) + 1
    return _MONTH_ABBR.get(token)


def _coming(today: date, weekday: int) -> date:
    """Next occurrence strictly after today."""
    days = (weekday - today.weekday()) % 7 or 7
    return today + timedelta(days=days)


def _next_week(today: date, weekday: int) -> date:
    monday_next_week = today - timedelta(days=today.weekday()) + timedelta(days=7)
    return monday_next_week + timedelta(days=weekday)


def _festival(text: str) -> tuple[str, date] | None:
    for alias, canonical in sorted(FESTIVAL_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if alias in text:
            return canonical, FESTIVALS[canonical]
    for name, d in FESTIVALS.items():
        if name.lower() in text:
            return name, d
    return None


def _day_month(day: int, month: int, today: date, year: int | None = None) -> date | None:
    year = year or today.year
    try:
        d = date(year, month, day)
    except ValueError:
        return None
    if year == today.year and d < today and (today - d).days > 7:
        d = date(year + 1, month, day)  # "5th January" said in October means next year
    return d


def resolve_date(expression: str, today: date) -> ResolvedDate:
    raw = expression
    text = _normalise(expression)
    window_end = today + timedelta(days=BOOKING_WINDOW_DAYS - 1)

    def done(d: date | None, why: str, **kw) -> ResolvedDate:
        r = ResolvedDate(d, raw, why, **kw)
        if d is not None:
            r.in_window = today <= d <= window_end
        return r

    if not text:
        return done(None, "empty expression")

    # Festivals, with "day before/after" and "eve".
    fest = _festival(text)
    if fest:
        name, d = fest
        if re.search(r"\b(day before|eve)\b", text):
            return done(d - timedelta(days=1), f"day before {name} ({d.isoformat()})")
        if re.search(r"\bday after\b", text):
            return done(d + timedelta(days=1), f"day after {name} ({d.isoformat()})")
        return done(d, f"{name} falls on {d.isoformat()} in this year's calendar")

    time_hint = None
    if re.search(r"\btonight\b", text):
        time_hint = "evening"
    elif m := re.search(r"\b(morning|afternoon|evening|night)\b", text):
        time_hint = m.group(1)

    # "in 3 days", "3 days from now", "a week from today"
    m = re.search(r"\b(?:in\s+)?(\d+|[a-z]+(?: of)?)\s+(day|days|week|weeks)\b(?:\s+(?:from\s+(?:now|today)|later|time))?", text)
    if m and (m.group(1).isdigit() or m.group(1) in _NUMBER_WORDS) and re.search(r"\bin\b|from|later|time", text):
        n = int(m.group(1)) if m.group(1).isdigit() else _NUMBER_WORDS[m.group(1)]
        days = n * (7 if m.group(2).startswith("week") else 1)
        return done(today + timedelta(days=days), f"{n} {m.group(2)} from today", time_hint=time_hint)

    if re.search(r"\b(today|tonight|now)\b", text):
        return done(today, "today", time_hint=time_hint)
    if re.search(r"\bday after tomorrow\b|\bday after tmrw\b", text):
        return done(today + timedelta(days=2), "day after tomorrow", time_hint=time_hint)
    if re.search(r"\b(tomorrow|tmrw|tomorow|kal)\b", text):
        return done(today + timedelta(days=1), "tomorrow", time_hint=time_hint)

    # Explicit dates win over weekday words ("Friday 23rd October"): 2026-10-23, 23/10, 23-10-2026, "23 october", "october 23", bare "23".
    if m := re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text):
        d = _day_month(int(m.group(3)), int(m.group(2)), today, int(m.group(1)))
        return done(d, "explicit date") if d else done(None, "invalid date")
    if m := re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", text):
        year = int(m.group(3)) if m.group(3) else None
        if year is not None and year < 100:
            year += 2000
        d = _day_month(int(m.group(1)), int(m.group(2)), today, year)  # Indian order: day/month
        return done(d, "explicit date (day/month)") if d else done(None, "invalid date")
    for m in re.finditer(r"\b(\d{1,2})\s*(?:of\s+)?([a-z]+)\b", text):
        if month := _month_index(m.group(2)):
            d = _day_month(int(m.group(1)), month, today)
            return done(d, "explicit date") if d else done(None, "invalid date")
    for m in re.finditer(r"\b([a-z]+)\s+(\d{1,2})\b", text):
        if month := _month_index(m.group(1)):
            d = _day_month(int(m.group(2)), month, today)
            return done(d, "explicit date") if d else done(None, "invalid date")
    # Weekends.
    if re.search(r"\bnext weekend\b", text):
        sat = _next_week(today, 5) if today.weekday() < 5 else _coming(today, 5) + timedelta(days=7)
        return done(sat, "Saturday of next weekend", alternatives=[sat + timedelta(days=1)], ambiguous=True)
    if re.search(r"\b(this|coming)? ?weekend\b", text):
        sat = today if today.weekday() == 5 else _coming(today, 5)
        return done(sat, "Saturday of this weekend", alternatives=[sat + timedelta(days=1)], ambiguous=True)
    if re.search(r"\bnext week\b", text) and not any(w in text for w in WEEKDAYS):
        mon = _next_week(today, 0)
        return done(None, f"next week runs {mon.isoformat()} to {(mon + timedelta(days=6)).isoformat()}; ask which day",
                    ambiguous=True, alternatives=[mon + timedelta(days=i) for i in range(7)])

    # "Friday the 23rd": a day of the month beats the weekday word.
    if m := re.search(r"\b(?:mon|tues|wednes|thurs|fri|satur|sun)day\s+(?:the\s+)?(\d{1,2})\b", text):
        day = int(m.group(1))
        if 1 <= day <= 31:
            month, year = today.month, today.year
            if day < today.day:
                month, year = (1, year + 1) if month == 12 else (month + 1, year)
            d = _day_month(day, month, today, year)
            if d:
                wd = re.match(r"\w+", m.group(0)).group(0)
                note = "" if d.strftime("%A").lower() == wd else f" ({d.isoformat()} is a {d.strftime('%A')}, not a {wd.title()}: confirm)"
                return done(d, "day of month with weekday" + note, ambiguous=bool(note))

    # Weekdays with modifiers.
    m = next(
        (x for x in re.finditer(r"\b(next|this|coming|upcoming|following)?\s*([a-z]+day|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b", text)
         if _weekday_index(x.group(2)) is not None),
        None,
    )
    if m:
        wd = _weekday_index(m.group(2))
        modifier = m.group(1)
        coming = _coming(today, wd)
        if modifier == "next":
            nxt = _next_week(today, wd)
            if nxt != coming:
                return done(nxt, f"{WEEKDAYS[wd].title()} of next week (weeks run Monday to Sunday)",
                            ambiguous=True, alternatives=[coming], time_hint=time_hint)
            return done(nxt, f"{WEEKDAYS[wd].title()} of next week", time_hint=time_hint)
        if modifier == "following":
            return done(coming + timedelta(days=7), f"the {WEEKDAYS[wd].title()} after the coming one", time_hint=time_hint)
        return done(coming, f"the coming {WEEKDAYS[wd].title()}", time_hint=time_hint)

    if m := re.search(r"\b(?:on\s+)?(?:the\s+)?(\d{1,2})\b", text):
        day = int(m.group(1))
        if 1 <= day <= 31:
            month, year = today.month, today.year
            if day < today.day:
                month, year = (1, year + 1) if month == 12 else (month + 1, year)
            d = _day_month(day, month, today, year)
            if d:
                return done(d, "day of month; assumed the next such date")

    return done(None, f"could not understand {raw!r} as a date")


# ------------------------------------------------------------------- times

_PART_OF_DAY = {
    "morning": (360, 719),
    "noon": (720, 780),
    "afternoon": (720, 1019),
    "lunch": (720, 900),
    "evening": (1020, 1259),
    "dinner": (1140, 1380),
    "night": (1200, 1439),
    "late night": (1290, 1439),
    "matinee": (720, 960),
    "first show": (540, 720),
    "second show": (1200, 1439),
}


def parse_time(text: str) -> tuple[int, int] | None:
    """Parse a spoken or written time into a (from_min, to_min) window.

    Exact times give a zero-width window ("8:30 pm" -> (1230, 1230)). Parts of day give a range.
    Hours 1-11 with no am/pm are read as pm unless the text says morning ("8" -> 20:00), which
    matches how callers talk about evening shows, night buses and dinner. Two-digit 24-hour times
    ("08:00", "10:00", "22:30") are taken literally.
    """
    if re.fullmatch(r"\s*([01]\d|2[0-3]):([0-5]\d)(:\d{2})?\s*", text):
        # Two-digit HH:MM is 24-hour ("10:00" is 10 AM): this is the format tools are given.
        h, m = int(text.strip()[:2]), int(text.strip()[3:5])
        return h * 60 + m, h * 60 + m
    t = text.lower().strip().replace(".", ":")
    t = re.sub(r"\s+", " ", t)
    if not t:
        return None
    for phrase in sorted(_PART_OF_DAY, key=len, reverse=True):
        if re.search(rf"\b{phrase}\b", t) and not re.search(r"\d", t):
            return _PART_OF_DAY[phrase]
    m = re.search(r"\bhalf past (\w+)\b", t)
    if m:
        h = int(m.group(1)) if m.group(1).isdigit() else _NUMBER_WORDS.get(m.group(1))
        if h is None:
            return None
        t = re.sub(r"half past \w+", f"{h}:30", t)
    m = re.search(r"\bquarter past (\w+)\b", t)
    if m:
        h = int(m.group(1)) if m.group(1).isdigit() else _NUMBER_WORDS.get(m.group(1))
        if h is None:
            return None
        t = re.sub(r"quarter past \w+", f"{h}:15", t)
    for word, n in sorted(_NUMBER_WORDS.items(), key=lambda kv: -len(kv[0])):
        if len(word) > 2:
            t = re.sub(rf"\b{word}\b", str(n), t)
    t = re.sub(r"\b(\d{1,2}) (15|30|45)\b", r"\1:\2", t)  # "eight thirty" -> "8:30"
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a m|p m|o'?clock|hrs|hours)?\b", t)
    if not m:
        return None
    raw_hour, mins = m.group(1), int(m.group(2) or 0)
    h = int(raw_hour)
    suffix = (m.group(3) or "").replace(" ", "")
    if h > 23 or mins > 59:
        return None
    if suffix == "pm" and h < 12:
        h += 12
    elif suffix == "am" and h == 12:
        h = 0
    elif suffix not in ("am", "pm", "hrs", "hours") and 1 <= h <= 11 and not raw_hour.startswith("0"):
        if not re.search(r"\b(morning|breakfast)\b", t):
            h += 12
    minute = h * 60 + mins
    return minute, minute

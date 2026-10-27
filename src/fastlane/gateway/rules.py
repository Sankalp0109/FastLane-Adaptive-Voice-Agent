"""Booking rules enforced in code (Table 4). The policy documents describe the same rules in prose;
tests in tests/test_rules.py keep the two in step.

Each check returns a list of Violation; an empty list means the action is allowed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# --- movies
MOVIE_MAX_TICKETS = 10
MOVIE_CHANGE_CUTOFF_HOURS = 4  # cancel or change up to 4 h before the show
MOVIE_CANCEL_FEE_PER_TICKET = 30
MOVIE_BOOKING_CLOSES_MIN = 15  # sales stop 15 min before the show

# --- buses (current policy, effective 2026-09-01)
BUS_MAX_SEATS = 6
BUS_BOOKING_CLOSES_MIN = 60
BUS_MODIFY_CUTOFF_HOURS = 12
BUS_REFUND_TABLE = [  # (hours before departure >=, refund %)
    (72, 90),
    (24, 75),
    (12, 50),
    (4, 25),
]
BUS_NO_CANCEL_HOURS = 4
BUS_POLICY_EFFECTIVE = date(2026, 9, 1)
# Superseded policy (bookings made before 2026-09-01): kept only in docs/bus_refund_policy_2025.md.
BUS_REFUND_TABLE_OLD = [(48, 85), (12, 60), (4, 25)]

# --- restaurants
RESTAURANT_CHANGE_CUTOFF_HOURS = 2
RESTAURANT_DEPOSIT_REFUND_HOURS = 24
RESTAURANT_BOOKING_CLOSES_MIN = 30
SEATING_MIN = 90


@dataclass(frozen=True)
class Violation:
    code: str
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


def hours_until(when: datetime, now: datetime) -> float:
    return (when - now).total_seconds() / 3600


# ------------------------------------------------------------------ movies

def check_movie_booking(quantity: int, show_start: datetime, now: datetime) -> list[Violation]:
    v = []
    if quantity < 1:
        v.append(Violation("bad_quantity", "At least one ticket is needed."))
    if quantity > MOVIE_MAX_TICKETS:
        v.append(Violation("seat_limit", f"At most {MOVIE_MAX_TICKETS} tickets can be booked in one booking; larger groups go to staff."))
    if show_start - now < timedelta(minutes=MOVIE_BOOKING_CLOSES_MIN):
        v.append(Violation("sales_closed", "Booking for this show has closed."))
    return v


def check_movie_change(show_start: datetime, now: datetime) -> list[Violation]:
    if hours_until(show_start, now) < MOVIE_CHANGE_CUTOFF_HOURS:
        return [Violation("cancellation_cutoff", f"Movie tickets cannot be cancelled or changed within {MOVIE_CHANGE_CUTOFF_HOURS} hours of the show.")]
    return []


def movie_refund(amount: int, quantity: int) -> int:
    return max(0, amount - MOVIE_CANCEL_FEE_PER_TICKET * quantity)


# ------------------------------------------------------------------- buses

def check_bus_booking(quantity: int, passengers: list[dict] | None, depart: datetime, now: datetime) -> list[Violation]:
    v = []
    if quantity < 1:
        v.append(Violation("bad_quantity", "At least one seat is needed."))
    if quantity > BUS_MAX_SEATS:
        v.append(Violation("seat_limit", f"At most {BUS_MAX_SEATS} seats per booking."))
    if depart - now < timedelta(minutes=BUS_BOOKING_CLOSES_MIN):
        v.append(Violation("sales_closed", "Booking for this bus has closed."))
    pax = passengers or []
    if len(pax) != quantity:
        v.append(Violation("passenger_details", f"Name, age and gender are needed for each of the {quantity} passengers."))
    for p in pax:
        if not str(p.get("name", "")).strip():
            v.append(Violation("passenger_details", "Every passenger needs a name."))
        age = p.get("age")
        if not isinstance(age, int) or not 1 <= age <= 110:
            v.append(Violation("passenger_details", f"Passenger {p.get('name') or '?'} needs a valid age."))
        if p.get("gender") not in ("M", "F", "O"):
            v.append(Violation("passenger_details", f"Passenger {p.get('name') or '?'} needs a gender (M/F/O)."))
    return v


_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                       "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}


def said_numbers(text: str) -> set[int]:
    """Numbers a caller said, as digits ("42") or words ("forty-two", "forty two")."""
    t = text.lower().replace("-", " ")
    out = {int(n) for n in re.findall(r"\b\d{1,3}\b", t)}
    words = re.findall(r"[a-z]+", t)
    for i, w in enumerate(words):
        if w in _UNITS:
            out.add(_UNITS[w])
        elif w in _TENS:
            nxt = words[i + 1] if i + 1 < len(words) else ""
            out.add(_TENS[w] + (_UNITS[nxt] if nxt in _UNITS and _UNITS[nxt] < 10 else 0))
    return out


def check_passenger_source(passengers: list[dict], heard: str, known: list[dict], requested: int | None) -> list[Violation]:
    """Passenger details must come from the caller (or their own booking history), never be guessed, and the
    caller's requested seat count must be covered. `heard` is everything the caller has said in this call."""
    v = []
    if requested and len(passengers) < requested:
        v.append(Violation("passenger_count", f"The caller asked for {requested} seats but details were given for "
                                              f"{len(passengers)}; ask for the name, age and gender of every passenger."))
    words = set(re.findall(r"[a-z]+", heard.lower()))
    numbers = said_numbers(heard)
    known_names = {str(k.get("name", "")).lower().split()[0] for k in known if k.get("name")}
    for pax in passengers:
        first = (str(pax.get("name", "")).lower().split() or [""])[0]
        from_history = first in known_names and any(k.get("age") == pax.get("age") for k in known
                                                    if str(k.get("name", "")).lower().split()[:1] == [first])
        if from_history:
            continue
        if first not in words:
            v.append(Violation("passenger_unheard", f"The caller has not given a passenger named {pax.get('name')!r}; "
                                                    "ask for each passenger's name, age and gender, do not guess."))
        elif pax.get("age") not in numbers:
            v.append(Violation("passenger_unheard", f"The caller has not said {pax.get('name')}'s age; ask for it, do not guess."))
    return v


# --- grounding: what is booked must have come up in the call (pilot v3 / voice demo: a cinema and a time
# nobody had mentioned were put into a booking)
_GENERIC_WORDS = {"cinemas", "cinema", "multiplex", "talkies", "tandoor", "tales", "hills", "city", "the"}
_HOUR_WORDS = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
_TIME_SPAN = re.compile(
    rf"\b(?:(?:half past|quarter past|quarter to)\s+(?:\d{{1,2}}|{_HOUR_WORDS})"
    rf"|(?:\d{{1,2}}|{_HOUR_WORDS})(?:[:.]\d{{2}}|\s+(?:fifteen|thirty|forty[- ]five))?\s*"
    r"(?:a\.?\s?m\b\.?|p\.?\s?m\b\.?|o'?\s?clock|hrs)"
    r"|\d{1,2}:\d{2})"
    r"(?:\s+(?:in the (?:morning|afternoon|evening|night)|at night))?",
    re.IGNORECASE)
_PARTS_OF_DAY = re.compile(r"\b(morning|afternoon|evening|night|noon|lunch|dinner)\b", re.IGNORECASE)


def times_said(text: str, parts_of_day: bool = True) -> list[tuple[int, int]]:
    """Every time window mentioned in free text ("the 7 PM show", "half past seven", "19:00", "evening").
    A time without am/pm ("10 o'clock") may be either half of the day."""
    from fastlane.gateway.dates import parse_time

    out: list[tuple[int, int]] = []
    for m in _TIME_SPAN.finditer(text):
        span = m.group(0)
        w = parse_time(span)
        if w is None:
            continue
        out.append(w)
        if not re.search(r"[ap]\.?\s?m\b|morning|afternoon|evening|night", span, re.IGNORECASE) and not re.fullmatch(r"\d{2}:\d{2}", span):
            out.append(((w[0] + 720) % 1440, (w[1] + 720) % 1440))  # "10 o'clock": 10 AM or 10 PM
    if parts_of_day:
        for m in _PARTS_OF_DAY.finditer(text):
            w = parse_time(m.group(1).lower())
            if w is not None:
                out.append(w)
    return out


def _words(text: str) -> set[str]:
    toks = re.findall(r"[a-z]+", text.lower())
    return set(toks) | {a + b for a, b in zip(toks, toks[1:])}  # "hi tech" also as "hitech"


def check_grounded(caller: str, agent: str, venue: str | None, venue_aliases: list[str], minute: int | None,
                   what: str) -> list[Violation]:
    """The venue and the time of a new booking must have been said by the caller or offered by the agent earlier in
    the call (`caller`, `agent`: everything each side said before this request). `what` names the venue kind."""
    from difflib import SequenceMatcher

    v = []
    said = _words(caller) | _words(agent)
    if venue:
        names = {w for w in re.findall(r"[a-z]+", venue.lower()) if w not in _GENERIC_WORDS} | {a.lower() for a in venue_aliases}
        heard = any(n in said or (len(n) >= 5 and any(SequenceMatcher(None, n, w).ratio() >= 0.8 for w in said if len(w) >= 4))
                    for n in names)
        if not heard:
            v.append(Violation("not_requested", f"The caller has not asked for {venue}; ask which {what} they want, do not choose one."))
    if minute is not None:
        windows = times_said(caller) + times_said(agent, parts_of_day=False)
        if not any(lo <= minute <= hi for lo, hi in windows):
            hhmm = f"{minute // 60:02d}:{minute % 60:02d}"
            v.append(Violation("not_requested", f"The time {hhmm} was neither asked for by the caller nor offered to them; "
                                                f"ask which time they want or offer the available times."))
    return v


def bus_refund_percent(depart: datetime, now: datetime, booked_on: date | None = None) -> int | None:
    """Refund % for cancelling now; None means cancellation is not allowed."""
    h = hours_until(depart, now)
    if h < BUS_NO_CANCEL_HOURS:
        return None
    table = BUS_REFUND_TABLE_OLD if booked_on is not None and booked_on < BUS_POLICY_EFFECTIVE else BUS_REFUND_TABLE
    for threshold, pct in table:
        if h >= threshold:
            return pct
    return None


def check_bus_cancel(depart: datetime, now: datetime) -> list[Violation]:
    if bus_refund_percent(depart, now) is None:
        return [Violation("cancellation_cutoff", f"Bus tickets cannot be cancelled within {BUS_NO_CANCEL_HOURS} hours of departure.")]
    return []


def check_bus_change(depart: datetime, now: datetime) -> list[Violation]:
    if hours_until(depart, now) < BUS_MODIFY_CUTOFF_HOURS:
        return [Violation("change_cutoff", f"Bus tickets cannot be changed within {BUS_MODIFY_CUTOFF_HOURS} hours of departure; cancellation rules apply instead.")]
    return []


# -------------------------------------------------------------- restaurants

def check_restaurant_booking(party: int, start: datetime, now: datetime, rest: dict) -> list[Violation]:
    v = []
    if party < 1:
        v.append(Violation("bad_quantity", "Party size must be at least 1."))
    if party > rest["max_party"]:
        v.append(Violation("party_size_limit", f"{rest['name']} takes phone bookings for up to {rest['max_party']} guests; larger groups are handled by our events team."))
    if start - now < timedelta(minutes=RESTAURANT_BOOKING_CLOSES_MIN):
        v.append(Violation("sales_closed", "Tables must be booked at least 30 minutes ahead."))
    if rest["closed_weekday"] is not None and start.weekday() == rest["closed_weekday"]:
        v.append(Violation("closed", f"{rest['name']} is closed on {start.strftime('%A')}s."))
    m = start.hour * 60 + start.minute
    in_lunch = rest["lunch_open"] <= m <= rest["lunch_last_seating"]
    in_dinner = rest["dinner_open"] <= m <= rest["dinner_last_seating"]
    if not (in_lunch or in_dinner):
        v.append(Violation("outside_hours", "Seatings start between 12:00-14:30 for lunch and 19:00 to the last dinner seating."))
    if m % 30:
        v.append(Violation("bad_time", "Seatings start on the hour or half hour."))
    return v


def restaurant_deposit(party: int, rest: dict) -> int:
    return party * rest["deposit_per_guest"] if party >= rest["deposit_min_party"] else 0


def check_restaurant_change(start: datetime, now: datetime) -> list[Violation]:
    if hours_until(start, now) < RESTAURANT_CHANGE_CUTOFF_HOURS:
        return [Violation("change_cutoff", f"Reservations cannot be changed or cancelled within {RESTAURANT_CHANGE_CUTOFF_HOURS} hours; please call the branch.")]
    return []


def restaurant_deposit_refund(deposit: int, start: datetime, now: datetime) -> int:
    return deposit if hours_until(start, now) >= RESTAURANT_DEPOSIT_REFUND_HOURS else 0

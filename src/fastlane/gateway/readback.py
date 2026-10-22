"""Read-backs: what the agent must say before any booking change, and how we know the caller agreed.

The gateway refuses a create/modify/cancel unless (1) the agent's previous utterance read back the
key details of exactly that action and (2) the caller's next utterance confirmed it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from fastlane.gateway.dates import spoken_date

_SMALL_NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def spoken_time(hhmm: str) -> str:
    h, m = map(int, hhmm.split(":"))
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {suffix}" if m else f"{h12} {suffix}"


def number_forms(n: int) -> list[str]:
    forms = [str(n)]
    if 0 <= n < len(_SMALL_NUMBERS):
        forms.append(_SMALL_NUMBERS[n])
    return forms


def date_forms(iso: str) -> list[str]:
    d = date.fromisoformat(iso)
    month = d.strftime("%B").lower()
    return [
        f"{d.day} {month}", f"{d.day}th {month}", f"{d.day}st {month}", f"{d.day}nd {month}", f"{d.day}rd {month}",
        f"{month} {d.day}", f"{d.day} {month[:3]}", f"{month[:3]} {d.day}", f"{d.day}/{d.month}", iso,
        f"{d.day} of {month}",
    ]


def time_forms(hhmm: str) -> list[str]:
    h, m = map(int, hhmm.split(":"))
    h12 = h % 12 or 12
    forms = [hhmm, f"{h12}:{m:02d}", spoken_time(hhmm).lower()]
    if m == 0:
        forms += [f"{h12} {'am' if h < 12 else 'pm'}", f"{h12} o'clock", f"{h12}pm" if h >= 12 else f"{h12}am"]
    if m == 30:
        forms += [f"half past {h12}", f"{_SMALL_NUMBERS[h12] if h12 < len(_SMALL_NUMBERS) else h12} thirty"]
    return forms


@dataclass
class Readback:
    text: str
    keys: dict[str, list[str]] = field(default_factory=dict)  # detail -> acceptable spoken forms

    def delivered_in(self, agent_text: str) -> bool:
        """True if every key detail appears (in some form) in what the agent said."""
        said = _norm(agent_text)
        return all(any(_norm(f) in said for f in forms) for forms in self.keys.values())

    def missing_in(self, agent_text: str) -> list[str]:
        said = _norm(agent_text)
        return [k for k, forms in self.keys.items() if not any(_norm(f) in said for f in forms)]


def _norm(text: str) -> str:
    text = text.lower().replace("₹", " rs ")
    text = re.sub(r"[^a-z0-9:/' ]+", " ", text)
    return " " + re.sub(r"\s+", " ", text).strip() + " "


def _venue_forms(label: str) -> list[str]:
    words = label.split()
    return [label, words[0], words[-1], " ".join(words[-2:])]


def movie_booking(movie: str, cinema: str, iso_date: str, hhmm: str, quantity: int, category: str,
                  amount: int, name: str, phone: str) -> Readback:
    when = f"{spoken_date(date.fromisoformat(iso_date))} at {spoken_time(hhmm)}"
    text = (f"Just to confirm: {quantity} {category} ticket{'s' if quantity > 1 else ''} for {movie} at {cinema}, "
            f"{when}, total ₹{amount}, under {name}, phone ending {phone[-4:]}. Shall I book it?")
    return Readback(text, {
        "movie": [movie, movie.split()[-1]],
        "venue": _venue_forms(cinema),
        "date": date_forms(iso_date),
        "time": time_forms(hhmm),
        "quantity": number_forms(quantity),
    })


def bus_booking(origin: str, destination: str, iso_date: str, hhmm: str, bus_type: str, passengers: list[dict],
                amount: int, phone: str) -> Readback:
    when = f"{spoken_date(date.fromisoformat(iso_date))}, departing {spoken_time(hhmm)}"
    names = ", ".join(f"{p['name']} ({p.get('age')})" for p in passengers)
    n = len(passengers)
    text = (f"Just to confirm: {n} seat{'s' if n > 1 else ''} on the {bus_type.replace('_', ' ').replace('nonac', 'non-AC').replace('ac ', 'AC ')} bus "
            f"from {origin} to {destination}, {when}, for {names}, total ₹{amount}, phone ending {phone[-4:]}. Shall I book it?")
    return Readback(text, {
        "route": [destination],
        "date": date_forms(iso_date),
        "time": time_forms(hhmm),
        "quantity": number_forms(n),
        "passengers": [passengers[0]["name"].split()[0]] if passengers else [""],
    })


def table_booking(restaurant: str, iso_date: str, hhmm: str, party: int, name: str, phone: str,
                  deposit: int, special_request: str | None) -> Readback:
    when = f"{spoken_date(date.fromisoformat(iso_date))} at {spoken_time(hhmm)}"
    extra = f", note: {special_request}" if special_request else ""
    dep = f" A deposit of ₹{deposit} is needed to hold the table; we will SMS a payment link." if deposit else ""
    text = (f"Just to confirm: a table for {party} at {restaurant}, {when}, under {name}, phone ending {phone[-4:]}{extra}.{dep} "
            f"Shall I book it?")
    area = restaurant.replace("Tandoor Tales ", "")
    keys = {
        "venue": [restaurant, area, area.split()[0]],
        "date": date_forms(iso_date),
        "time": time_forms(hhmm),
        "party": number_forms(party),
    }
    if deposit:
        keys["deposit"] = ["deposit"]
    return Readback(text, keys)


def cancellation(booking_id: str, summary: str, refund_note: str) -> Readback:
    spaced = " ".join(booking_id)
    text = f"Just to confirm: cancel booking {spaced} - {summary}. {refund_note} Shall I go ahead and cancel?"
    return Readback(text, {"cancel": ["cancel"], "booking": [booking_id, spaced, summary.split(",")[0]]})


def modification(booking_id: str, before: str, after: str, note: str = "") -> Readback:
    spaced = " ".join(booking_id)
    text = f"Just to confirm: change booking {spaced} from {before} to {after}.{(' ' + note) if note else ''} Shall I make the change?"
    return Readback(text, {"change": ["change", "move", "modify", "update"], "after": [after.split(",")[0]]})


# ----------------------------------------------------------- confirmation

_YES = [
    "yes", "yeah", "yep", "yup", "ya", "yaa", "haan", "haa", "ha ji", "haanji", "ji haan", "correct", "right",
    "that's right", "thats right", "confirm", "confirmed", "go ahead", "please do", "do it", "book it", "sure",
    "ok", "okay", "okk", "fine", "perfect", "theek hai", "thik hai", "sahi hai", "done", "proceed", "absolutely",
    "of course", "cancel it", "please cancel", "make the change", "sounds good", "alright", "all right", "avunu", "sari",
]
_NO = [
    "no", "nope", "nahi", "nahin", "na", "not", "don't", "dont", "wait", "hold on", "change", "actually", "wrong",
    "instead", "cancel that", "stop", "leave it", "one minute", "ek minute", "but", "kaadu", "vaddu",
]


def confirmation(text: str) -> bool | None:
    """True = clear yes, False = no / wants a change, None = unclear (ask again)."""
    t = " " + re.sub(r"[^a-z' ]+", " ", text.lower()) + " "
    t = re.sub(r"\s+", " ", t)
    has_yes = any(f" {w} " in t for w in _YES)
    has_no = any(f" {w} " in t for w in _NO)
    # "no, that's correct" / "no no, go ahead" are rare; any no-word blocks the change (safer).
    if has_no:
        return False
    if has_yes:
        return True
    return None

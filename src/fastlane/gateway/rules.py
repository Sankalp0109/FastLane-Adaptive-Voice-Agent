"""Booking rules enforced in code (Table 4). The policy documents describe the same rules in prose;
tests in tests/test_rules.py keep the two in step.

Each check returns a list of Violation; an empty list means the action is allowed.
"""

from __future__ import annotations

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

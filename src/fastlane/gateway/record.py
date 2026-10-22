"""The booking record: a typed, per-call record of what the caller wants.

It is passed to whichever model replies next (small, large or staff) so details survive a switch.
Variant A4 removes it from the model context; the gateway still keeps it for scoring.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Intent = Literal["book", "modify", "cancel", "info"]

# Fields whose change after being set counts as "the caller changed an earlier choice".
TRACKED_FIELDS = (
    "domain", "date", "time", "quantity", "movie", "cinema", "seat_category", "origin", "destination",
    "bus_type", "restaurant", "party_size", "option_id",
)


class Passenger(BaseModel):
    name: str
    age: int | None = None
    gender: Literal["M", "F", "O"] | None = None

    @field_validator("gender", mode="before")
    @classmethod
    def _gender(cls, v):
        if v is None:
            return None
        g = str(v).strip().lower()
        return {"m": "M", "male": "M", "man": "M", "f": "F", "female": "F", "woman": "F", "o": "O", "other": "O"}.get(g)


class BookingRecord(BaseModel):
    domain: Literal["movies", "buses", "restaurants"] | None = None
    intent: Intent | None = None
    caller_name: str | None = None
    caller_phone: str | None = None  # number the call comes from
    contact_phone: str | None = None  # number given for the booking, if different
    # Shared
    date: str | None = None  # ISO
    time: str | None = None  # HH:MM
    quantity: int | None = None
    option_id: str | None = None  # chosen inventory option
    booking_id: str | None = None  # existing booking (modify/cancel) or the one just created
    identity_verified: bool = False
    # Movies
    movie: str | None = None
    cinema: str | None = None
    seat_category: str | None = None
    # Buses
    origin: str | None = None
    destination: str | None = None
    bus_type: str | None = None
    berth_preference: str | None = None
    passengers: list[Passenger] = Field(default_factory=list)
    # Restaurants
    restaurant: str | None = None
    party_size: int | None = None
    special_request: str | None = None
    # Read-back / progress
    readback_status: Literal["none", "pending", "delivered", "confirmed", "rejected"] = "none"
    pending_readback: str | None = None
    completed_actions: list[str] = Field(default_factory=list)
    changed_fields: list[str] = Field(default_factory=list)  # fields the caller changed after setting
    notes: list[str] = Field(default_factory=list)

    def set(self, track: bool = True, **fields: Any) -> list[str]:
        """Update fields; return the tracked fields whose earlier value was overwritten.

        track=False for updates that come from the system (tool results), not from the caller."""
        changed = []
        for key, value in fields.items():
            if value is None or not hasattr(self, key):
                continue
            old = getattr(self, key)
            if track and key in TRACKED_FIELDS and old not in (None, [], "") and old != value:
                changed.append(key)
                if key not in self.changed_fields:
                    self.changed_fields.append(key)
            setattr(self, key, value)
        return changed

    def render(self) -> str:
        """Compact one-line-per-field text for model prompts (only filled fields)."""
        skip = {"changed_fields", "notes", "completed_actions", "passengers", "pending_readback"}
        lines = []
        for key, value in self.model_dump().items():
            if key in skip or value in (None, [], "", False) or (key == "readback_status" and value == "none"):
                continue
            lines.append(f"{key}: {value}")
        if self.passengers:
            pax = "; ".join(f"{p.name} ({p.age or '?'}, {p.gender or '?'})" for p in self.passengers)
            lines.append(f"passengers: {pax}")
        if self.pending_readback:
            lines.append(f"pending_readback: {self.pending_readback}")
        if self.completed_actions:
            lines.append(f"done: {', '.join(self.completed_actions)}")
        if self.changed_fields:
            lines.append(f"caller_changed: {', '.join(self.changed_fields)}")
        if self.notes:
            lines.append(f"notes: {' | '.join(self.notes[-3:])}")
        return "\n".join(lines) if lines else "(empty)"

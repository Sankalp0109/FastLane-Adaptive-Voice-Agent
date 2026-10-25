"""Tool argument schemas and descriptions shared by every model.

The JSON schemas generated here are what the small and large models see, so descriptions are
written for a model reading them, not for developers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Args(BaseModel):
    model_config = ConfigDict(extra="ignore")


DomainT = Literal["movies", "buses", "restaurants"]


class ResolveDateArgs(_Args):
    expression: str = Field(description="What the caller said, e.g. 'next Friday', 'day before Diwali', '23rd'.")


class SearchInventoryArgs(_Args):
    domain: DomainT
    date: str = Field(description="ISO date YYYY-MM-DD, or the caller's own words for the date ('next Friday', '10th November'), which are resolved for you.")
    time: str | None = Field(None, description="Preferred time 'HH:MM' (24h) or a part of day like 'evening'.")
    quantity: int | None = Field(None, description="Tickets (movies) or seats (buses).")
    movie: str | None = None
    cinema: str | None = Field(None, description="Cinema name or area, e.g. 'Galaxy' or 'Kukatpally'.")
    seat_category: str | None = Field(None, description="classic, prime or recliner.")
    origin: str | None = Field(None, description="Bus origin city.")
    destination: str | None = Field(None, description="Bus destination city.")
    bus_type: str | None = Field(None, description="e.g. 'AC sleeper', 'non-AC seater', or just 'sleeper'.")
    restaurant: str | None = Field(None, description="Branch name or area, e.g. 'Jubilee Hills'. Omit to search all branches.")
    party_size: int | None = Field(None, description="Guests (restaurants).")


class FindAlternativesArgs(SearchInventoryArgs):
    option_id: str | None = Field(None, description="The option the caller wanted, if one was found.")


class PassengerArgs(_Args):
    name: str
    age: int
    gender: str = Field(description="M, F or O")

    @field_validator("gender")
    @classmethod
    def _gender(cls, v: str) -> str:
        g = str(v).strip().lower()
        if g in ("m", "male", "man", "boy", "gents", "mr"):
            return "M"
        if g in ("f", "female", "woman", "girl", "ladies", "mrs", "ms"):
            return "F"
        if g in ("o", "other", "others", "non-binary", "nonbinary"):
            return "O"
        raise ValueError("gender must be M, F or O")


class CreateBookingArgs(_Args):
    domain: DomainT
    option_id: str = Field(description="option_id from search_inventory or find_alternatives.")
    name: str = Field(description="Name the booking is under.")
    phone: str | None = Field(None, description="10-digit contact number; defaults to the number the caller is calling from.")
    quantity: int | None = Field(None, description="Tickets (movies) or seats (buses).")
    seat_category: str | None = Field(None, description="Movies only, if not already in the option_id.")
    passengers: list[PassengerArgs] | None = Field(None, description="Buses: one entry per seat with name, age, gender.")
    seat_preference: Literal["lower", "upper", "window"] | None = None
    party_size: int | None = Field(None, description="Restaurants: number of guests.")
    special_request: str | None = Field(None, description="Restaurants: dietary needs, occasion, seating wish.")
    waive_deposit: bool = Field(False, description="Deposits can only be waived by staff. Always leave false.")


class ModifyBookingArgs(_Args):
    booking_id: str
    phone: str = Field(description="Phone number the booking was made with, as told by the caller.")
    new_option_id: str | None = Field(None, description="New show/trip/time option from a search, if the time/date/venue changes.")
    quantity: int | None = None
    seat_category: str | None = None
    party_size: int | None = None
    passengers: list[PassengerArgs] | None = None
    special_request: str | None = None


class CancelBookingArgs(_Args):
    booking_id: str
    phone: str = Field(description="Phone number the booking was made with, as told by the caller.")


class LookupBookingArgs(_Args):
    phone: str = Field(description="Phone number the booking was made with.")
    booking_id: str | None = Field(None, description="Booking reference like 'MV7K2Q9'. Omit to list bookings of the calling number.")


class CallerHistoryArgs(_Args):
    phone: str | None = Field(None, description="Defaults to the number the caller is calling from.")


class SearchPolicyArgs(_Args):
    query: str = Field(description="The caller's question in plain words, e.g. 'refund if I cancel bus 10 hours before'.")
    domain: DomainT | None = None


class SendSmsArgs(_Args):
    message: str
    phone: str | None = Field(None, description="Defaults to the caller's number. Only the caller or a verified booking's number is allowed.")


class HandoffArgs(_Args):
    reason: Literal["payment", "complaint", "out_of_scope", "large_group", "repeated_failure", "identity", "caller_request", "other"]
    summary: str = Field(description="Two or three sentences for the staff member: who, what they want, what was done.")


TOOLS: dict[str, tuple[type[_Args], str]] = {
    "resolve_date": (ResolveDateArgs, "Turn a spoken date ('next Friday', 'Diwali', 'the 23rd') into an ISO date, e.g. to tell the caller which day it is. Searches accept the caller's words directly."),
    "search_inventory": (SearchInventoryArgs, "Find shows, bus trips or restaurant tables with live availability and option_ids."),
    "find_alternatives": (FindAlternativesArgs, "When the wanted show/bus/table is full, find the closest available alternatives (other times, venues, categories, days)."),
    "create_booking": (CreateBookingArgs, "Create a booking. The first call returns a read-back that must be read to the caller; call again with the same arguments after the caller says yes."),
    "modify_booking": (ModifyBookingArgs, "Change an existing booking (time, date, venue, quantity, passengers). Needs the booking's phone number and a confirmed read-back."),
    "cancel_booking": (CancelBookingArgs, "Cancel an existing booking. Needs the booking's phone number and a confirmed read-back. Returns the refund."),
    "lookup_booking": (LookupBookingArgs, "Look up a booking by reference and phone number, or list the calling number's bookings."),
    "get_caller_history": (CallerHistoryArgs, "Past and upcoming bookings of the calling number, with their usual choices."),
    "search_policy": (SearchPolicyArgs, "Search fares, refund rules, FAQs, menus and the festival calendar. Answer policy questions only from its results."),
    "send_sms": (SendSmsArgs, "Send an SMS to the caller (e.g. directions, menu link). Booking confirmations are sent automatically."),
    "handoff_to_staff": (HandoffArgs, "Transfer the call to a human: payments/refunds disputes, complaints, out-of-scope requests, groups above limits, identity problems, or repeated tool failures."),
}

WRITE_TOOLS = {"create_booking", "modify_booking", "cancel_booking"}


def _strip_titles(node):
    """Drop pydantic's auto-generated 'title' keys: they repeat the property names and cost tokens."""
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if not (k == "title" and isinstance(v, str))}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


def tool_specs(names: list[str]) -> list[dict]:
    """Provider-neutral specs: name, description, JSON schema of the arguments."""
    out = []
    for name in names:
        model, description = TOOLS[name]
        out.append({"name": name, "description": description, "input_schema": _strip_titles(model.model_json_schema())})
    return out

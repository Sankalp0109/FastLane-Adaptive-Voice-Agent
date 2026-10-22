"""Fixed responses that need no model call (Section 6 speed-ups: template read-backs and goodbyes)."""

from __future__ import annotations

import re

GREETING = ("Namaste, this is FastLane, an automated booking assistant for movies, buses and Tandoor Tales restaurants. "
            "How can I help you today?")

HANDOFF = "I'm connecting you to a member of our staff now. I've passed on the details, so you won't need to repeat them."

FALLBACK = "Sorry, I didn't quite get that. Could you please say it once more?"

GOODBYE = "Thank you for calling FastLane. Have a nice day!"

_BYE = re.compile(r"\b(bye|goodbye|that'?s all|that is all|nothing else|no thanks|thank you|thanks|dhanyavaad|shukriya)\b", re.I)
_MORE = re.compile(r"\b(also|and|another|one more|book|cancel|change|what|how|can|could|is|are|do|does|want|need|\?)\b", re.I)


def is_goodbye(text: str) -> bool:
    """A closing turn with no new request ("ok thank you, bye")."""
    return bool(_BYE.search(text)) and not _MORE.search(_BYE.sub("", text))


def spell_ref(booking_id: str) -> str:
    return " ".join(booking_id)


def done(action: str, booking: dict, extra: dict | None = None) -> str:
    ref = spell_ref(booking["booking_id"])
    if action == "cancel_booking":
        refund = (extra or {}).get("refund_amount")
        money = f" A refund of ₹{refund} will reach your original payment method in 5 to 7 working days." if refund else ""
        return f"Done, booking {ref} is cancelled.{money} You'll get an SMS shortly. Anything else?"
    if booking.get("status") == "pending_deposit":
        return (f"Your table is held, reference {ref}. Please pay the deposit of ₹{(extra or {}).get('deposit_due') or booking.get('amount')} "
                f"using the SMS link within 2 hours to confirm it. Anything else?")
    seats = booking.get("seats")
    seat_txt = f" Seats {', '.join(seats)}." if seats else ""
    verb = "updated" if action == "modify_booking" else "confirmed"
    return f"Done, your booking is {verb}. Reference {ref}.{seat_txt} You'll get an SMS shortly. Anything else?"


def handoff_summary(record_text: str, last_caller: str, reason: str) -> str:
    return f"Reason: {reason}. Caller said: \"{last_caller[:200]}\". Record: {record_text.replace(chr(10), '; ')[:400]}"

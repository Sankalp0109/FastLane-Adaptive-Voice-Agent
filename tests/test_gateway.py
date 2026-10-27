import json
from dataclasses import replace
from datetime import datetime

import pytest

from fastlane.config import FASTLANE, SYSTEMS
from fastlane.gateway.gateway import Gateway
from fastlane.gateway.readback import confirmation

CALLER = "9876500001"


def confirm(gw: Gateway, tool: str, args: dict, answer: str = "yes, go ahead") -> dict:
    """Call a write tool, read the read-back out, have the caller answer, call again."""
    first = gw.call(tool, args)
    assert first["error"] == "needs_readback", first
    gw.end_turn(first["readback"])
    gw.begin_turn(answer)
    return gw.call(tool, args)


def first_available(result: dict, category: str | None = None) -> str:
    for show in result["results"]:
        for o in show["options"]:
            if o["available"] and (category is None or o["category"] == category):
                return o["option_id"]
    raise AssertionError("nothing available")


def reads_out_results(g: Gateway) -> Gateway:
    """Most gateway tests are about other rules: let every search result count as read out to the caller, as an
    agent would, so the grounding rule (venue/time must have come up in the call) is satisfied."""
    call = g.call

    def call_and_offer(name, args):
        r = call(name, args)
        if name in ("search_inventory", "find_alternatives") and r.get("ok"):
            g.offered.append(json.dumps(r))
        return r

    g.call = call_and_offer
    return g


@pytest.fixture
def gw(world):
    g = Gateway(world, CALLER)
    g.begin_turn("hello")
    return reads_out_results(g)


def movie_option(gw, date="2026-10-22", qty=2):
    r = gw.call("search_inventory", {"domain": "movies", "date": date, "movie": "Rocket Raja", "quantity": qty})
    return first_available(r, "prime")


# ------------------------------------------------------------------ read-back gate

def test_write_needs_confirmed_readback(gw, world):
    args = {"domain": "movies", "option_id": movie_option(gw), "name": "Ravi Kumar", "quantity": 2}
    r = gw.call("create_booking", args)
    assert r["error"] == "needs_readback" and "Rocket Raja" in r["readback"]
    assert world.q("SELECT * FROM bookings WHERE phone=?", (CALLER,)) == []
    gw.end_turn(r["readback"])
    gw.begin_turn("haan, theek hai")
    r = gw.call("create_booking", args)
    assert r["ok"] and r["booking"]["quantity"] == 2
    event = gw.write_events()[0]
    assert event.confirmed and not event.violations
    assert gw.unsafe_writes() == []
    assert world.one("SELECT COUNT(*) n FROM sms_log WHERE phone=?", (CALLER,))["n"] == 1


def test_readback_must_actually_be_spoken(gw):
    args = {"domain": "movies", "option_id": movie_option(gw), "name": "Ravi", "quantity": 2}
    gw.call("create_booking", args)
    gw.end_turn("Sure, booking it now!")  # did not read the details back
    gw.begin_turn("yes")
    assert gw.call("create_booking", args)["error"] == "needs_readback"


def test_caller_correction_cancels_pending(gw):
    args = {"domain": "movies", "option_id": movie_option(gw), "name": "Ravi", "quantity": 2}
    r = confirm(gw, "create_booking", args, answer="no wait, make it three tickets")
    assert r["error"] == "needs_readback"
    assert gw.record.readback_status == "pending"


def test_confirmation_is_for_exact_arguments(gw):
    args = {"domain": "movies", "option_id": movie_option(gw), "name": "Ravi", "quantity": 2}
    r = gw.call("create_booking", args)
    gw.end_turn(r["readback"])
    gw.begin_turn("yes")
    assert gw.call("create_booking", {**args, "quantity": 3})["error"] == "needs_readback"


@pytest.mark.parametrize("text,expected", [
    ("yes please", True), ("haan ji", True), ("correct, go ahead", True), ("ok", True),
    ("no", False), ("wait, change the time", False), ("yes but make it 3", False), ("hmm", None),
])
def test_confirmation_detector(text, expected):
    assert confirmation(text) is expected


# ------------------------------------------------------------------ domain rules

def test_movie_seat_limit(gw):
    args = {"domain": "movies", "option_id": movie_option(gw), "name": "Ravi", "quantity": 11}
    r = gw.call("create_booking", args)
    assert r["error"] == "rule_violation" and r["violations"][0]["code"] == "seat_limit"


def test_movie_cancellation_cutoff(gw, world):
    # A seeded booking for a show later today, inside the 4-hour cut-off (now is 11:00).
    b = world.one(
        """SELECT b.id, b.phone FROM bookings b JOIN shows s ON s.id=b.show_id
           WHERE s.date='2026-10-15' AND s.start_min < 900 AND b.domain='movies'"""
    )
    assert b is not None, "seed should plant a booking inside the cut-off"
    r = gw.call("cancel_booking", {"booking_id": b["id"], "phone": b["phone"]})
    assert r["error"] == "rule_violation" and r["violations"][0]["code"] == "cancellation_cutoff"


def test_identity_mismatch_is_refused_without_leaking(gw, world):
    b = world.one("SELECT id, phone FROM bookings WHERE domain='buses' LIMIT 1")
    r = gw.call("cancel_booking", {"booking_id": b["id"], "phone": "9000000000"})
    assert r["error"] == "not_found_or_mismatch"
    r2 = gw.call("lookup_booking", {"booking_id": "MVZZZZZ", "phone": "9000000000"})
    assert r2["error"] == "not_found_or_mismatch" and r["message"].startswith(r2["message"][:30])
    assert world.one("SELECT status FROM bookings WHERE id=?", (b["id"],))["status"] == "confirmed"


def test_bus_cancel_refund_and_release(world):
    b = world.one(
        """SELECT b.id, b.phone, b.amount, b.trip_id FROM bookings b JOIN bus_trips t ON t.id=b.trip_id
           WHERE t.date >= '2026-10-19' ORDER BY t.date LIMIT 1"""
    )
    gw = Gateway(world, b["phone"])
    gw.begin_turn("please cancel my bus ticket")
    r = confirm(gw, "cancel_booking", {"booking_id": b["id"], "phone": b["phone"]}, answer="yes cancel it")
    assert r["ok"] and r["refund_amount"] == b["amount"] * 90 // 100
    assert world.one("SELECT COUNT(*) n FROM bus_seats_taken WHERE booking_id=?", (b["id"],))["n"] == 0
    assert world.describe_booking(b["id"])["status"] == "cancelled"


def test_bus_needs_passenger_details(gw):
    r = gw.call("search_inventory", {"domain": "buses", "date": "2026-10-22", "destination": "Vijayawada", "quantity": 2})
    option = next(t["option_id"] for t in r["results"] if t["available"])
    r = gw.call("create_booking", {"domain": "buses", "option_id": option, "name": "Ravi", "quantity": 2,
                                   "passengers": [{"name": "Ravi", "age": 34, "gender": "M"}]})
    assert r["error"] == "rule_violation" and "passenger" in r["message"]
    gw.end_turn("Please tell me the name, age and gender of each passenger.")
    gw.begin_turn("Ravi Kumar, 34, male, and Lakshmi Kumar, thirty-one, female.")
    pax = [{"name": "Ravi Kumar", "age": 34, "gender": "M"}, {"name": "Lakshmi Kumar", "age": 31, "gender": "F"}]
    r = confirm(gw, "create_booking", {"domain": "buses", "option_id": option, "name": "Ravi Kumar", "passengers": pax})
    assert r["ok"] and [p["name"] for p in r["booking"]["passengers"]] == ["Ravi Kumar", "Lakshmi Kumar"]


def _free_table_option(gw, party, date="2026-10-21", restaurant="jubilee hills"):
    r = gw.call("search_inventory", {"domain": "restaurants", "date": date, "time": "dinner", "restaurant": restaurant, "party_size": party})
    return r["results"][0]["times"][0]["option_id"]


def test_large_party_needs_deposit_and_cannot_skip_it(gw, world):
    option = _free_table_option(gw, 9)
    r = gw.call("create_booking", {"domain": "restaurants", "option_id": option, "name": "Asha", "party_size": 9, "waive_deposit": True})
    assert r["error"] == "rule_violation" and r["violations"][0]["code"] == "deposit_waived"
    r = confirm(gw, "create_booking", {"domain": "restaurants", "option_id": option, "name": "Asha", "party_size": 9})
    assert r["ok"] and r["booking"]["status"] == "pending_deposit" and r["deposit_due"] == 9 * 300
    sms = world.one("SELECT body FROM sms_log WHERE phone=? ORDER BY id DESC", (CALLER,))["body"]
    assert "deposit" in sms


def test_party_size_limit(gw):
    option = _free_table_option(gw, 4)
    r = gw.call("create_booking", {"domain": "restaurants", "option_id": option, "name": "Asha", "party_size": 13})
    assert r["error"] == "rule_violation" and r["violations"][0]["code"] == "party_size_limit"


def test_modify_movie_moves_seats(world):
    b = world.one(
        """SELECT b.id, b.phone, b.show_id, b.seat_category, b.quantity FROM bookings b JOIN shows s ON s.id=b.show_id
           WHERE s.date >= '2026-10-17' AND b.domain='movies' LIMIT 1"""
    )
    gw = Gateway(world, b["phone"])
    gw.begin_turn("can I move my booking to another show")
    view = world.describe_booking(b["id"])
    r = gw.call("search_inventory", {"domain": "movies", "date": view["date"], "movie": view["movie"], "quantity": b["quantity"]})
    new = next(o["option_id"] for s in r["results"] for o in s["options"]
               if o["available"] and s["show_id"] != b["show_id"])
    r = confirm(gw, "modify_booking", {"booking_id": b["id"], "phone": b["phone"], "new_option_id": new})
    assert r["ok"] and r["booking"]["show_id"] == new.split(":")[0]
    assert world.one("SELECT COUNT(*) n FROM show_seats_taken WHERE show_id=? AND booking_id=?", (b["show_id"], b["id"]))["n"] == 0


def test_modify_not_executed_until_confirmed(world):
    b = world.one("SELECT id, phone, party_size FROM bookings WHERE domain='restaurants' AND res_date >= '2026-10-18' LIMIT 1")
    before = world.describe_booking(b["id"])
    gw = Gateway(world, b["phone"])
    gw.begin_turn("make it one more person")
    r = gw.call("modify_booking", {"booking_id": b["id"], "phone": b["phone"], "party_size": b["party_size"] + 1})
    assert r["error"] in ("needs_readback", "no_availability")
    assert world.describe_booking(b["id"]) == before  # dry run rolled back


# ------------------------------------------------------------------ variants

def test_without_enforcement_writes_go_through_but_are_logged_unsafe(world):
    gw = Gateway(world, CALLER, config=SYSTEMS["A3"])
    gw.begin_turn("book it")
    option = _free_table_option(gw, 9)
    r = gw.call("create_booking", {"domain": "restaurants", "option_id": option, "name": "Asha", "party_size": 9, "waive_deposit": True})
    assert r["ok"] and r["booking"]["status"] == "confirmed"  # deposit skipped
    unsafe = gw.unsafe_writes()
    assert len(unsafe) == 1 and "deposit_waived" in unsafe[0].violations and unsafe[0].confirmed is False


def test_unenforced_identity_mismatch_is_logged(world):
    b = world.one("SELECT id FROM bookings WHERE domain='buses' LIMIT 1")
    gw = Gateway(world, CALLER, config=SYSTEMS["A3"])
    gw.begin_turn("cancel my friend's booking")
    r = gw.call("cancel_booking", {"booking_id": b["id"], "phone": CALLER})
    assert r["ok"]
    assert "identity_mismatch" in gw.unsafe_writes()[0].violations


def test_tool_sets_follow_config(world):
    assert "resolve_date" not in Gateway(world, CALLER, config=SYSTEMS["A2"]).tool_names
    assert "find_alternatives" not in Gateway(world, CALLER, config=SYSTEMS["A2"]).tool_names
    assert "get_caller_history" not in Gateway(world, CALLER, config=SYSTEMS["A7"]).tool_names
    specs = Gateway(world, CALLER).specs()
    assert len(specs) == 11 and json.dumps(specs)
    r = Gateway(world, CALLER, config=SYSTEMS["A2"]).call("resolve_date", {"expression": "next friday"})
    assert r["error"] == "unknown_tool"


def test_injected_failure_then_retry(world):
    gw = Gateway(world, CALLER, inject_failures={"search_inventory": 2})
    gw.begin_turn("hi")
    args = {"domain": "movies", "date": "2026-10-22"}
    assert gw.call("search_inventory", args)["error"] == "system_error"
    assert gw.call("search_inventory", args)["error"] == "system_error"
    assert gw.consecutive_failures == 2
    assert gw.call("search_inventory", args)["ok"]
    assert gw.consecutive_failures == 0


def test_invalid_args_reported(gw):
    r = gw.call("search_inventory", {"domain": "trains", "date": "2026-10-22"})
    assert r["error"] == "invalid_args"
    # With the date tools, searches take the caller's words and resolve them in code ...
    r = gw.call("search_inventory", {"domain": "movies", "date": "next friday"})
    assert r["ok"] and r["resolved_date"]["date"] == "2026-10-23" and gw.record.date == "2026-10-23"
    r = gw.call("search_inventory", {"domain": "movies", "date": "next week"})
    assert r["error"] == "ambiguous_date" and len(r["alternatives"]) == 7


def test_words_for_dates_need_the_date_tools(world):
    gw = Gateway(world, CALLER, config=SYSTEMS["A2"])
    gw.begin_turn("hi")
    r = gw.call("search_inventory", {"domain": "movies", "date": "next friday"})
    assert r["error"] == "invalid_args" and "YYYY-MM-DD" in r["message"]


def test_caller_history_for_returning_caller(world):
    phone = next(iter(world.profiles()))
    gw = Gateway(world, phone)
    gw.begin_turn("hi it's me again")
    r = gw.call("get_caller_history", {})
    assert r["ok"] and r["returning_caller"] and r["recent"]
    assert gw.call("get_caller_history", {"phone": "9000000000"})["error"] == "identity"


def test_handoff_ends_tool_use(gw, world):
    r = gw.call("handoff_to_staff", {"reason": "payment", "summary": "Caller disputes a charge."})
    assert r["ok"] and gw.handoff["reason"] == "payment"
    assert gw.call("search_inventory", {"domain": "movies", "date": "2026-10-22"})["error"] == "call_transferred"
    assert world.one("SELECT COUNT(*) n FROM handoffs")["n"] == 1


def test_sales_close_before_show(world):
    gw = Gateway(world, CALLER, now=datetime(2026, 10, 15, 20, 20))
    gw.begin_turn("tickets for the 8:30 show tonight")
    r = gw.call("search_inventory", {"domain": "movies", "date": "2026-10-15", "time": "20:30"})
    assert all(s["time"] != "20:30" for s in r["results"])


def test_record_tracks_caller_changes_only():
    from fastlane.gateway.record import BookingRecord

    rec = BookingRecord()
    rec.set(track=False, movie="Kaala Samudram")
    rec.set(track=False, movie="Rocket Raja")
    assert rec.changed_fields == []
    assert rec.set(movie="Shadow Protocol") == ["movie"]


def test_fastlane_enforces_rules():
    assert FASTLANE.enforce_rules and not replace(FASTLANE, enforce_rules=False).enforce_rules


def test_passenger_gender_words_are_normalised(gw):
    r = gw.call("search_inventory", {"domain": "buses", "date": "2026-10-22", "destination": "Chennai", "quantity": 1})
    option = next(t["option_id"] for t in r["results"] if t["available"])
    gw.begin_turn("Sindhu Goud, forty, female")
    r = gw.call("create_booking", {"domain": "buses", "option_id": option, "name": "Sindhu",
                                   "passengers": [{"name": "Sindhu Goud", "age": 40, "gender": "female"}]})
    assert r["error"] == "needs_readback"
    r = gw.call("create_booking", {"domain": "buses", "option_id": option, "name": "Sindhu",
                                   "passengers": [{"name": "Sindhu Goud", "age": 40, "gender": "alien"}]})
    assert r["error"] == "invalid_args"


def test_bus_passengers_must_come_from_the_caller(gw):
    """Pilot v3 failures: the small model invented passengers ("Caller", 40) and booked 1 of 3 requested seats."""
    gw.begin_turn("I want 3 seats to Vijayawada on the 22nd, put it in my name Kavya Tiwari, 30.")
    r = gw.call("search_inventory", {"domain": "buses", "date": "2026-10-22", "destination": "Vijayawada", "quantity": 3})
    option = next(t["option_id"] for t in r["results"] if t["available"])
    r = gw.call("create_booking", {"domain": "buses", "option_id": option, "name": "Kavya Tiwari",
                                   "passengers": [{"name": "Kavya Tiwari", "age": 30, "gender": "F"}]})
    assert r["error"] == "rule_violation" and {v["code"] for v in r["violations"]} == {"passenger_count"}
    pax = [{"name": "Kavya Tiwari", "age": 30, "gender": "F"}, {"name": "Caller", "age": 40, "gender": "M"},
           {"name": "Fatima Khan", "age": 52, "gender": "F"}]
    r = gw.call("create_booking", {"domain": "buses", "option_id": option, "name": "Kavya Tiwari", "passengers": pax})
    codes = [v["code"] for v in r["violations"]]
    assert codes == ["passenger_unheard", "passenger_unheard"]  # "Caller" never said; Fatima's name not said either
    gw.begin_turn("Okay, the others are Fatima Khan, fifty-two, female and Jagadeesh Rao, 41, male.")
    pax[1] = {"name": "Jagadeesh Rao", "age": 41, "gender": "M"}
    r = gw.call("create_booking", {"domain": "buses", "option_id": option, "name": "Kavya Tiwari", "passengers": pax})
    assert r["error"] == "needs_readback"
    pax[2] = {"name": "Fatima Khan", "age": 25, "gender": "F"}  # an age the caller never said
    r = gw.call("create_booking", {"domain": "buses", "option_id": option, "name": "Kavya Tiwari", "passengers": pax})
    assert [v["code"] for v in r["violations"]] == ["passenger_unheard"] and "age" in r["violations"][0]["message"]


def test_said_numbers():
    from fastlane.gateway.rules import said_numbers

    assert {42, 31, 7} <= said_numbers("she is 42, he is thirty-one and the kid is seven")
    assert 45 in said_numbers("forty five") and 20 in said_numbers("twenty")


def test_booking_must_be_for_what_came_up_in_the_call(world):
    """Voice demo: the small model searched with a cinema nobody had named and dropped the caller's 7 PM, and the
    read-back offered Galaxy at 10 AM. Venue and time must have been said by the caller or offered by the agent."""
    g = Gateway(world, CALLER)
    g.begin_turn("Two prime tickets for Rocket Raja at Regal Ameerpet on the 22nd, the evening show.")
    args = {"domain": "movies", "date": "2026-10-22", "movie": "Rocket Raja", "quantity": 2, "seat_category": "prime"}
    other = next(x for x in g.call("search_inventory", args)["results"] if "Regal" not in x["cinema"])
    opt = next(o["option_id"] for o in other["options"] if o["available"])
    r = g.call("create_booking", {"domain": "movies", "option_id": opt, "name": "Ravi", "quantity": 2})
    assert r["error"] == "rule_violation" and any(v["code"] == "not_requested" and "has not asked for" in v["message"] for v in r["violations"])
    regal = g.call("search_inventory", {**args, "cinema": "Regal"})["results"] + g.call("search_inventory", {**args, "cinema": "Regal", "time": "evening"})["results"]
    morning = next((s for s in regal if int(s["time"][:2]) < 12), None)
    if morning is not None:  # right cinema, but a morning show when the caller asked for the evening
        opt = next(o["option_id"] for o in morning["options"] if o["available"])
        r = g.call("create_booking", {"domain": "movies", "option_id": opt, "name": "Ravi", "quantity": 2})
        assert [v["code"] for v in r["violations"]] == ["not_requested"] and "neither asked for" in r["violations"][0]["message"]
    evening = next(s for s in regal if 17 <= int(s["time"][:2]) <= 20)
    opt = next(o["option_id"] for o in evening["options"] if o["available"])
    assert g.call("create_booking", {"domain": "movies", "option_id": opt, "name": "Ravi", "quantity": 2})["error"] == "needs_readback"


def test_offered_alternative_counts_but_a_rejected_readback_does_not(world):
    g = Gateway(world, CALLER)
    g.begin_turn("A table for 4 at Jubilee Hills on the 21st please.")
    g.end_turn("Jubilee Hills has tables at 7:30 PM and 9 PM on the 21st. Which would you like?")
    g.begin_turn("The first one.")
    opt = "tandoor-jubilee@2026-10-21T19:30"
    r = g.call("create_booking", {"domain": "restaurants", "option_id": opt, "name": "Asha", "party_size": 4})
    assert r["error"] == "needs_readback"
    g.end_turn(r["readback"])  # the read-back itself does not count as an offer
    assert "7:30" not in " ".join(g.offered) or "Jubilee Hills has tables" in " ".join(g.offered)
    bad = "tandoor-jubilee@2026-10-21T21:30"
    g.begin_turn("No, that's wrong.")
    r = g.call("create_booking", {"domain": "restaurants", "option_id": bad, "name": "Asha", "party_size": 4})
    assert r["error"] == "rule_violation" and r["violations"][0]["code"] == "not_requested"

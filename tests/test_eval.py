"""Test-set generation, gold achievability, scoring, runner and metrics."""

import json
from collections import Counter

import pytest

from fastlane.config import SYSTEMS
from fastlane.eval.labels import is_hard
from fastlane.eval.metrics import percentile, summarize, wilson
from fastlane.eval.runner import run_call, sample_calls
from fastlane.eval.scenarios import MIX, _Builder, setup_world
from fastlane.eval.score import booking_views, changed_bookings, score_call
from fastlane.gateway.gateway import Gateway
from fastlane.models.scripted import ScriptedClient
from fastlane.world.db import World


@pytest.fixture(scope="module")
def scripts(world_path):
    w = World.fresh(world_path)
    out = _Builder(w).build()
    w.close()
    return out


def test_category_mix(scripts):
    assert len(scripts) == 360
    counts = Counter((s.domain, s.category) for s in scripts)
    for category, per_domain in MIX.items():
        for domain, n in zip(("movies", "buses", "restaurants"), per_domain):
            assert counts[(domain, category)] == n, (domain, category)
    assert sum(1 for s in scripts if s.inject_failure) == 36
    assert len({s.call_id for s in scripts}) == 360


def test_build_is_deterministic(world_path, scripts):
    w = World.fresh(world_path)
    again = _Builder(w).build()
    assert [s.to_json() for s in again] == [s.to_json() for s in scripts]


def _parse_pax(text: str) -> list[dict]:
    g = {"male": "M", "female": "F", "other": "O"}
    out = []
    for part in text.split(";"):
        name, age, gender = [x.strip() for x in part.split(",")]
        out.append({"name": name, "age": int(age), "gender": g[gender]})
    return out


def _option(world: World, e: dict) -> str:
    if e["domain"] == "movies":
        h, m = map(int, e["time"].split(":"))
        row = world.one(
            """SELECT s.id FROM shows s JOIN screens sc ON sc.id=s.screen_id
               WHERE s.movie_id=? AND sc.cinema_id=? AND s.date=? AND s.start_min=?""",
            (e["movie_id"], e["cinema_id"], e["date"], h * 60 + m))
        return f"{row['id']}:{e['seat_category']}"
    if e["domain"] == "buses":
        h, m = map(int, e["time"].split(":"))
        row = world.one("SELECT id FROM bus_trips WHERE route_id=? AND date=? AND depart_min=? AND bus_type=?",
                        (e["route_id"], e["date"], h * 60 + m, e["bus_type"]))
        return row["id"]
    return f"{e['restaurant_id']}@{e['date']}T{e['time']}"


def oracle(world: World, script) -> None:
    """Apply the script's first gold outcome directly through the gateway (no model)."""
    gw = Gateway(world, script.phone, config=SYSTEMS["A3"])
    gw.begin_turn("oracle")
    pax = None
    for key, value in script.facts.items():
        if key.startswith("passengers"):
            pax = _parse_pax(value)
    for e in script.gold["outcomes"][0]:
        if e.get("status") == "cancelled":
            r = gw.call("cancel_booking", {"booking_id": e["booking_id"], "phone": script.phone})
        elif "booking_id" in e:
            args = {"booking_id": e["booking_id"], "phone": script.phone, "new_option_id": _option(world, e)}
            if e["domain"] == "restaurants":
                args["party_size"] = e["party_size"]
            r = gw.call("modify_booking", args)
        else:
            args = {"domain": e["domain"], "option_id": _option(world, e), "name": script.caller_name}
            if e["domain"] == "movies":
                args["quantity"] = e["quantity"]
            elif e["domain"] == "buses":
                args["passengers"] = pax
            else:
                args["party_size"] = e["party_size"]
            r = gw.call("create_booking", args)
        assert r.get("ok"), (script.call_id, r)


def test_every_gold_outcome_is_achievable(world_path, scripts):
    for s in scripts:
        w = World.fresh(world_path)
        setup_world(w, s)
        before = booking_views(w)
        oracle(w, s)
        changed = changed_bookings(before, booking_views(w))
        text = " ".join(s.gold.get("answer_any") or [])
        handoff = {"reason": "other"} if s.gold.get("handoff") else None
        result = score_call(s.gold, changed, text, handoff)
        assert result["success"], (s.call_id, result["mismatch"])
        w.close()


def test_setup_ids_match_scripts(world_path, scripts):
    for s in [x for x in scripts if x.setup]:
        w = World.fresh(world_path)
        ids = setup_world(w, s)
        ref = s.facts.get("booking reference")
        if ref:
            assert ref in ids, s.call_id
        w.close()


def test_doing_nothing_fails_booking_calls(world_path, scripts):
    s = next(x for x in scripts if x.category == "simple" and not x.setup)
    w = World.fresh(world_path)
    assert score_call(s.gold, [], "", None)["success"] is False
    red = next(x for x in scripts if x.category == "red_flag")
    assert score_call(red.gold, [], "", None)["success"] is False  # missed hand-off
    assert score_call(red.gold, [], "", {"reason": "complaint"})["success"] is True
    w.close()


def test_wrong_booking_fails(world_path, scripts):
    s = next(x for x in scripts if x.domain == "movies" and x.category == "simple" and not x.setup)
    wrong = dict(s.gold["outcomes"][0][0], quantity=s.gold["outcomes"][0][0]["quantity"] + 1)
    result = score_call(s.gold, [wrong], "", None)
    assert not result["success"] and "quantity" in result["mismatch"][-1]


def test_hard_labels():
    assert is_hard("can we do next Friday instead")[0]
    assert is_hard("what's the refund if I cancel")[0]
    assert is_hard("I want to talk to a manager")[0]
    assert not is_hard("two tickets please")[0]
    assert is_hard("ok", no_availability=True)[1] == ["sold_out"]


def test_sample_is_stratified(scripts):
    sample = sample_calls(scripts, 24)
    assert len(sample) == 24
    assert len({(c.domain, c.category) for c in sample}) == 24


def test_metric_helpers():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi


def test_runner_end_to_end_with_scripted_models(world_path, scripts):
    """One red-flag call through the real runner with scripted small model and caller."""
    s = next(x for x in scripts if x.category == "red_flag" and not x.inject_failure)
    small = ScriptedClient(lambda m, t: json.dumps({"route": "staff", "reason": "complaint", "slots": {}, "tool": None, "say": ""}))
    caller = ScriptedClient(lambda m, t: json.dumps({"say": "Okay, thank you, bye.", "done": True}), name="caller")
    rec = run_call(s, "fastlane", 0, small, ScriptedClient([], name="large"), caller, world_path=world_path)
    assert rec["error"] is None and rec["success"] and rec["handoff"]["reason"] == "complaint"
    assert rec["turns"][0]["handled_by"] == "staff"
    summary = summarize([rec])[0]
    assert summary["success"] == 1.0 and summary["missed_handoffs"] == "0/1"


def test_runner_scores_a_booking_call(world_path, scripts):
    """A simple booking played by a scripted 'perfect' small model through the read-back gate."""
    s = next(x for x in scripts if x.domain == "restaurants" and x.category == "simple" and not x.setup and not x.inject_failure
             and x.gold["outcomes"][0][0]["status"] == "confirmed")
    e = s.gold["outcomes"][0][0]
    args = {"domain": "restaurants", "option_id": _option(World.fresh(world_path), e), "name": s.caller_name, "party_size": e["party_size"]}
    small = ScriptedClient([json.dumps({"route": "self", "slots": {}, "tool": {"name": "create_booking", "args": args}, "say": ""})])
    replies = iter(["Yes, that's correct.", "Thank you, bye."])
    caller = ScriptedClient(lambda m, t: json.dumps({"say": next(replies), "done": False}), name="caller")
    rec = run_call(s, "fastlane", 0, small, ScriptedClient([], name="large"), caller, world_path=world_path)
    assert rec["error"] is None and rec["success"], rec["mismatch"]
    assert rec["writes"] and rec["writes"][0]["confirmed"] is True and rec["unsafe_writes"] == 0


def test_answer_keywords_survive_llm_unicode():
    gold = {"outcomes": [[]], "handoff": False, "answer_any": ["4 hours", "non-ac"]}
    assert score_call(gold, [], "only up to 4 hours before; the non‑AC bus", None)["answer_ok"] is True

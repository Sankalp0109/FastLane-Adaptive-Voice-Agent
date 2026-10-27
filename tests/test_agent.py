"""Orchestrator tests with scripted models: every routing path, for every kind of system."""

import json
import math

import pytest

from fastlane.config import SYSTEMS
from fastlane.agent.orchestrator import Agent
from fastlane.gateway.gateway import Gateway
from fastlane.models.base import LLMResponse, Usage
from fastlane.models.parsing import escalation_score, parse_small
from fastlane.models.scripted import ScriptedClient
from fastlane.retrieval.index import PolicyIndex

CALLER = "9876500001"


@pytest.fixture(scope="module")
def index():
    return PolicyIndex()


def small_json(route="self", tool=None, args=None, say="", slots=None, reason=""):
    out = {"route": route, "reason": reason, "slots": slots or {}, "tool": {"name": tool, "args": args or {}} if tool else None, "say": say}
    return json.dumps(out)


def free_option(world, date="2026-10-22", movie="Rocket Raja", qty=2):
    g = Gateway(world, CALLER)
    r = g.call("search_inventory", {"domain": "movies", "date": date, "movie": movie, "quantity": qty, "seat_category": "prime"})
    return next(o["option_id"] for s in r["results"] for o in s["options"] if o["available"])


def make(world, index, system="fastlane", small=None, large=None, **kw):
    return Agent(world, CALLER, SYSTEMS[system], small=small, large=large, policy_index=index, raise_errors=True, **kw)


def test_easy_booking_needs_no_large_model(world, index):
    opt = free_option(world)
    search = {"domain": "movies", "date": "2026-10-22", "movie": "Rocket Raja", "quantity": 2, "seat_category": "prime"}
    create = {"domain": "movies", "option_id": opt, "name": "Ravi Kumar", "quantity": 2}
    small = ScriptedClient([
        small_json(tool="search_inventory", args=search, slots={"domain": "movies", "movie": "Rocket Raja", "quantity": 2}),
        small_json(tool="create_booking", args=create),
    ])
    large = ScriptedClient([], name="large")
    agent = make(world, index, small=small, large=large)
    agent.greet()
    show = Gateway(world, CALLER).inv.show(opt.split(":")[0])
    when = f"{show['start_min'] // 60 % 12 or 12}:{show['start_min'] % 60:02d} {'PM' if show['start_min'] >= 720 else 'AM'}"
    readback = agent.respond(f"Two prime tickets for Rocket Raja at {show['cinema']} on the 22nd at {when}, name Ravi Kumar")
    assert readback.startswith("Just to confirm") and "Rocket Raja" in readback
    done = agent.respond("yes, go ahead")
    assert done.startswith("Done") and "Reference" in done
    assert agent.traces[-1].handled_by == "template" and not agent.traces[-1].model_calls
    assert large.calls == []
    assert len(world.q("SELECT id FROM bookings WHERE phone=?", (CALLER,))) == 1
    assert agent.gw.unsafe_writes() == []
    assert agent.traces[1].model_ms == 200.0  # two small steps x 100 ms


def test_small_routes_hard_turn_to_large(world, index):
    small = ScriptedClient([small_json(route="llm", reason="relative date")])
    large = ScriptedClient([{"tool": "resolve_date", "args": {"expression": "next Friday"}}, "Next Friday is the 23rd of October. Which film?"], name="large")
    agent = make(world, index, small=small, large=large)
    agent.greet()
    reply = agent.respond("I want to watch something next Friday")
    tr = agent.traces[-1]
    assert reply.startswith("Next Friday") and tr.handled_by == "large" and "relative date" in tr.escalation_reason
    assert [t.tool for t in tr.tools] == ["resolve_date"] and tr.tools[0].by == "large"
    # Large model saw the booking record and the tools.
    assert large.calls[0]["tools"] and "BOOKING RECORD" in large.calls[0]["messages"][-1]["content"]
    assert large.calls[0]["reasoning"] == "medium"


def test_no_availability_triggers_large(world, index):
    sold_out = {"domain": "movies", "date": "2026-10-16", "time": "20:30", "movie": "Kaala Samudram", "cinema": "Galaxy", "quantity": 2}
    small = ScriptedClient([small_json(tool="search_inventory", args=sold_out)])
    large = ScriptedClient(["That show is sold out; the 5 PM show at Galaxy has prime seats. Shall I check those?"], name="large")
    agent = make(world, index, small=small, large=large)
    agent.greet()
    agent.respond("Kaala Samudram tomorrow 8:30 at Galaxy, two tickets")
    tr = agent.traces[-1]
    assert tr.handled_by == "large" and tr.escalation_reason == "no availability"
    assert "Work already done" in large.calls[0]["messages"][-1]["content"]


def test_staff_route_hands_off(world, index):
    small = ScriptedClient([small_json(route="staff", reason="payment dispute")])
    agent = make(world, index, small=small, large=ScriptedClient([], name="large"))
    agent.greet()
    reply = agent.respond("You charged me twice for my bus ticket!")
    assert "staff" in reply and agent.ended and agent.gw.handoff["reason"] == "payment"


def test_changed_choice_triggers_large(world, index):
    small = ScriptedClient([
        small_json(say="Which cinema would you like?", slots={"domain": "movies", "movie": "Rocket Raja", "quantity": 2}),
        small_json(say="Sure.", slots={"quantity": 4}),
    ])
    large = ScriptedClient(["Okay, four tickets instead of two. Which cinema?"], name="large")
    agent = make(world, index, small=small, large=large)
    agent.greet()
    agent.respond("two tickets for Rocket Raja")
    assert agent.traces[-1].handled_by == "small"
    agent.respond("actually make it four")
    assert agent.traces[-1].handled_by == "large" and "quantity" in agent.traces[-1].escalation_reason


def test_spelling_variants_are_not_changes(world, index):
    small = ScriptedClient([
        small_json(say="Which date?", slots={"movie": "Kaala Samudram"}),
        small_json(say="Which cinema?", slots={"movie": "kala samudram", "date": "next friday"}),
    ])
    agent = make(world, index, small=small, large=ScriptedClient([], name="large"))
    agent.greet()
    agent.respond("Kaala Samudram please")
    agent.respond("Kala Samudram, next Friday")
    assert agent.traces[-1].handled_by == "small" and agent.gw.record.changed_fields == []


def test_repeated_tool_failure_goes_to_staff(world, index):
    args = {"domain": "movies", "date": "2026-10-22"}
    small = ScriptedClient([small_json(tool="search_inventory", args=args)])
    agent = make(world, index, small=small, large=ScriptedClient([], name="large"), inject_failures={"search_inventory": 2})
    agent.greet()
    reply = agent.respond("what's showing on the 22nd")
    assert "staff" in reply and agent.gw.handoff["reason"] == "repeated_failure"
    assert [t.error for t in agent.traces[-1].tools][:2] == ["system_error", "system_error"]


def test_single_failure_is_retried(world, index):
    args = {"domain": "movies", "date": "2026-10-22"}
    small = ScriptedClient([small_json(tool="search_inventory", args=args), small_json(say="Rocket Raja and others are showing.")])
    agent = make(world, index, small=small, large=ScriptedClient([], name="large"), inject_failures={"search_inventory": 1})
    agent.greet()
    assert agent.respond("what's showing on the 22nd") == "Rocket Raja and others are showing."
    assert [t.ok for t in agent.traces[-1].tools] == [False, True]


def test_unparseable_small_output_escalates(world, index):
    small = ScriptedClient(["Sure! I can help with that."])
    large = ScriptedClient(["Happy to help. Which film?"], name="large")
    agent = make(world, index, small=small, large=large)
    agent.greet()
    agent.respond("hi I want movie tickets")
    assert agent.traces[-1].escalation_reason == "unparseable small output"


def test_goodbye_template(world, index):
    agent = make(world, index, small=ScriptedClient([]), large=ScriptedClient([], name="large"))
    agent.greet()
    assert agent.respond("ok thank you, bye") == "Thank you for calling FastLane. Have a nice day!"
    assert agent.ended and not agent.traces[-1].model_calls


def test_b1_uses_only_large_model_with_full_history(world, index):
    large = ScriptedClient(["Which film would you like?", "And how many tickets?"], name="large")
    agent = make(world, index, system="B1", large=large)
    agent.greet()
    agent.respond("I want movie tickets")
    agent.respond("Rocket Raja")
    assert all(t.handled_by == "large" for t in agent.traces[1:])
    roles = [m["role"] for m in large.calls[1]["messages"]]
    assert roles.count("user") == 2 and "Booking rules" in large.calls[1]["messages"][0]["content"]


def test_b1_writes_without_gate_are_counted(world, index):
    opt = free_option(world)
    large = ScriptedClient([{"tool": "create_booking", "args": {"domain": "movies", "option_id": opt, "name": "Ravi", "quantity": 2}}, "Booked!"], name="large")
    agent = make(world, index, system="B1", large=large)
    agent.greet()
    agent.respond("book two Rocket Raja tickets, any show")
    unsafe = agent.gw.unsafe_writes()
    assert len(unsafe) == 1 and unsafe[0].confirmed is False  # no read-back was given


def test_b2_never_calls_large(world, index):
    small = ScriptedClient([small_json(route="llm", say="Let me check. Which date?")])
    agent = make(world, index, system="B2", small=small)
    agent.greet()
    assert agent.respond("tickets for next friday") == "Let me check. Which date?"
    assert agent.large is None and agent.traces[-1].handled_by == "small"


def test_b3_asks_large_router_each_turn(world, index):
    small = ScriptedClient([small_json(say="Which film?")])
    large = ScriptedClient(["EASY", "HARD", "Next Friday is 23 October."], name="large")
    agent = make(world, index, system="B3", small=small, large=large)
    agent.greet()
    agent.respond("I want movie tickets")
    assert agent.traces[-1].handled_by == "small" and agent.traces[-1].model_calls[0].role == "router"
    agent.respond("for next Friday")
    assert agent.traces[-1].handled_by == "large"


def test_a4_has_no_record_in_context(world, index):
    small = ScriptedClient([small_json(say="Which film?"), small_json(say="How many?")])
    agent = make(world, index, system="A4", small=small, large=ScriptedClient([], name="large"))
    agent.greet()
    agent.respond("movie tickets please")
    agent.respond("Rocket Raja")
    user = small.calls[-1]["messages"][1]["content"]
    assert "BOOKING RECORD" not in user and "Namaste" in user  # full transcript, from the greeting


def test_a5_large_answers_without_reasoning(world, index):
    small = ScriptedClient([small_json(route="llm")])
    large = ScriptedClient(["Sure."], name="large")
    agent = make(world, index, system="A5", small=small, large=large)
    agent.greet()
    agent.respond("complicated thing")
    assert large.calls[0]["reasoning"] == "low"


def test_returning_caller_history_in_context(world, index):
    phone = next(iter(world.profiles()))
    small = ScriptedClient([small_json(say="Welcome back!")])
    agent = Agent(world, phone, SYSTEMS["fastlane"], small=small, large=ScriptedClient([], name="large"), policy_index=index, raise_errors=True)
    agent.greet()
    agent.respond("hi, same as last time please")
    assert "CALLER HISTORY" in small.calls[0]["messages"][1]["content"]
    small7 = ScriptedClient([small_json(say="Hello!")])
    agent7 = Agent(world, phone, SYSTEMS["A7"], small=small7, large=ScriptedClient([], name="large"), policy_index=index, raise_errors=True)
    agent7.greet()
    agent7.respond("hi, same as last time please")
    assert "CALLER HISTORY" not in small7.calls[0]["messages"][1]["content"]


def test_early_lookup_puts_policy_in_context(world, index):
    small = ScriptedClient([small_json(say="Pets are not allowed on our buses, except service dogs.")])
    agent = make(world, index, small=small, large=ScriptedClient([], name="large"))
    agent.greet()
    agent.respond("can I take my dog on the bus to Bengaluru?", partials=["can I take my dog", "can I take my dog on the bus to"])
    assert agent.traces[-1].policy_prefetched
    assert "POSSIBLY RELEVANT POLICY" in small.calls[0]["messages"][1]["content"]


# ------------------------------------------------------------------ parsing

def test_parse_small_variants():
    assert parse_small('```json\n{"route": "self", "say": "Hi"}\n```').say == "Hi"
    assert parse_small("ESCALATE_STAFF").route == "staff"
    assert parse_small("blah").ok is False
    p = parse_small('noise {"route": "llm", "tool": {"name": "resolve_date", "args": {"expression": "Diwali"}}} trailing')
    assert p.route == "llm" and p.tool["args"]["expression"] == "Diwali"


def test_escalation_score_from_logprobs():
    resp = LLMResponse('{"route": "self"}', usage=Usage(), top_logprobs=[
        {"{\"": 0.0}, {"route": 0.0}, {"\":": 0.0}, {" \"": 0.0},
        {"self": math.log(0.7), "llm": math.log(0.2), "staff": math.log(0.1)},
    ])
    assert escalation_score(resp, parse_small(resp.text)) == pytest.approx(0.3, abs=1e-3)
    assert escalation_score(LLMResponse('{"route": "llm"}'), parse_small('{"route": "llm"}')) == 1.0


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_every_system_completes_a_call(world, index, system):
    def small_fn(messages, tools):
        return small_json(say="Which film would you like?")

    def large_fn(messages, tools):
        if messages[0]["content"].startswith("You route turns"):
            return "HARD" if "Friday" in messages[-1]["content"] else "EASY"
        return "Which show time suits you?"

    cfg = SYSTEMS[system]
    small = ScriptedClient(small_fn) if cfg.small_model != "none" else None
    large = ScriptedClient(large_fn, name="large") if cfg.large_model else None
    agent = Agent(world, CALLER, cfg, small=small, large=large, policy_index=index, raise_errors=True)
    agent.greet()
    for line in ["I want movie tickets", "next Friday please", "thanks, bye"]:
        agent.respond(line)
    assert agent.ended
    assert [t.turn for t in agent.traces] == [0, 1, 2, 3]
    handled = {t.handled_by for t in agent.traces[1:]}
    if cfg.router == "always_large":
        assert handled <= {"large", "template"}
    if not cfg.large_model:
        assert "large" not in handled


def test_large_recovers_from_provider_tool_validation_error(world, index):
    calls = {"n": 0}

    def large_fn(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Error code: 400 - Tool call validation failed: parameters for tool create_booking did not match schema")
        assert "rejected" in messages[-1]["content"]
        return "Could you tell me each passenger's gender?"

    agent = make(world, index, small=ScriptedClient([small_json(route="llm")]), large=ScriptedClient(large_fn, name="large"))
    agent.greet()
    assert agent.respond("book two seats") == "Could you tell me each passenger's gender?"


def test_write_args_fill_the_record(world, index):
    large = ScriptedClient([{"tool": "create_booking", "args": {"domain": "buses", "option_id": "TR99999", "name": "Uday Goud",
                                                                "passengers": [{"name": "Uday Goud", "age": 49, "gender": "male"}]}},
                            "Which bus would you like?"], name="large")
    agent = make(world, index, small=ScriptedClient([small_json(route="llm")]), large=large)
    agent.greet()
    agent.respond("book for Uday Goud, 49, male")
    rec = agent.gw.record
    assert rec.caller_name == "Uday Goud" and rec.passengers[0].gender == "M"


def test_fallback_client_switches_on_rate_limit():
    from fastlane.models.openai_compat import FallbackClient, RateLimited

    class Limited:
        name, max_wait_s, calls = "groq:x", None, 0

        def chat(self, messages, **kw):
            self.calls += 1
            raise RateLimited(300, "tokens per day")

    primary, backup = Limited(), ScriptedClient(["from backup", "again backup"], name="gemini:y")
    fb = FallbackClient(primary, backup)
    assert fb.chat([{"role": "user", "content": "hi"}]).text == "from backup"
    assert fb.chat([{"role": "user", "content": "hi"}]).text == "again backup"
    assert primary.calls == 1 and fb.backup_calls == 2  # primary not retried while blocked


@pytest.mark.parametrize("text,bye", [
    ("ok thank you, bye", True), ("No, that's all. Thank you!", True), ("Fine, leave it then. Bye.", True),
    ("Siddharth Menon, 44, male, that's all", False), ("thanks, can you also book a table", False),
])
def test_goodbye_detection(text, bye):
    from fastlane.agent.templates import is_goodbye

    assert is_goodbye(text) is bye



def test_spoken_date_slot_is_resolved_into_the_record(world, index):
    small = ScriptedClient([small_json(say="Which cinema?", slots={"movie": "Rocket Raja", "date": "next Friday"})])
    agent = make(world, index, small=small, large=ScriptedClient([], name="large"))
    agent.greet()
    agent.respond("Rocket Raja next Friday")
    assert agent.gw.record.date == "2026-10-23" and agent.traces[-1].handled_by == "small"


def test_numeric_policy_answer_goes_to_large(world, index):
    """Pilot v3: the small model read the bus refund table wrongly; amounts are now the large model's job."""
    q = {"query": "refund if I cancel 10 hours before departure", "domain": "buses"}
    small = ScriptedClient([small_json(tool="search_policy", args=q), small_json(say="You get 90% back.")])
    large = ScriptedClient(["Cancelling 4 to 12 hours before departure refunds 25% of the fare."], name="large")
    agent = make(world, index, small=small, large=large)
    agent.greet()
    reply = agent.respond("How much refund if I cancel my bus ticket 10 hours before departure?")
    tr = agent.traces[-1]
    assert "25%" in reply and tr.handled_by == "large" and tr.escalation_reason == "policy answer with numbers"


def test_policy_answer_without_numbers_stays_small(world, index):
    q = {"query": "are subtitles available", "domain": "movies"}
    small = ScriptedClient([small_json(tool="search_policy", args=q), small_json(say="Yes, all non-English films have English subtitles.")])
    agent = make(world, index, small=small, large=ScriptedClient([], name="large"))
    agent.greet()
    agent.respond("Are subtitles there for Telugu films?")
    assert agent.traces[-1].handled_by == "small"


def test_fallback_client_waits_for_primary_when_backup_fails():
    from fastlane.models.openai_compat import FallbackClient, RateLimited

    class LimitedOnce:
        name, max_wait_s, calls = "groq:x", None, 0

        def chat(self, messages, **kw):
            self.calls += 1
            if self.max_wait_s is not None:
                raise RateLimited(30, "tokens per minute")
            return ScriptedClient(["from primary after waiting"]).chat(messages)

    class Slow:
        name = "gemini:y"

        def chat(self, messages, **kw):
            raise TimeoutError("backup too slow")

    primary = LimitedOnce()
    fb = FallbackClient(primary, Slow())
    r = fb.chat([{"role": "user", "content": "hi"}])
    assert r.text == "from primary after waiting" and r.throttle_ms >= 0
    assert primary.calls == 2 and primary.max_wait_s == 10.0  # waiting allowed only for that one call

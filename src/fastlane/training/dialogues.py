"""Generated practice calls for training data ("practice system" rows of Table 5).

Both sides are code: a rule-based caller that reveals details when asked, and an oracle agent that
knows the right next step. The call runs through the REAL orchestrator and gateway, with the oracle
plugged in as the small and the large model. Every small-model call is recorded as
(exact input messages, ideal JSON output), so training data matches inference byte for byte.

Routing targets follow the proposal: the small model handles easy turns (including relative dates and
policy questions when it has the date and policy tools), passes changed choices, "my usual" turns,
refused details and dates-without-a-date-tool to the large model, and sends red flags, deposit
waivers and over-limit groups to staff. No availability and tool errors are escalated by the
orchestrator in code. Only calls that end in the planned final state are kept.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date

from fastlane.agent.orchestrator import Agent
from fastlane.config import SEED, SYSTEMS, TODAY, SystemConfig
from fastlane.eval.runner import stt_partials
from fastlane.eval.scenarios import CallScript, _Builder, load_test_calls, setup_world
from fastlane.eval.score import booking_views, changed_bookings, outcome_matches
from fastlane.gateway.dates import resolve_date
from fastlane.gateway.readback import spoken_time
from fastlane.gateway.tools import WRITE_TOOLS
from fastlane.models.base import LLMResponse, ToolCall, Usage
from fastlane.retrieval.index import PolicyIndex
from fastlane.training import phrases as P
from fastlane.world.db import World, fmt_time

SLOT_ORDER = {
    "movies": ["movie", "date", "cinema", "time", "seat_category", "quantity", "name"],
    "buses": ["destination", "date", "time", "passengers", "name"],
    "restaurants": ["restaurant", "date", "time", "party_size", "name"],
}
STAFF_EVENTS = {"red_flag", "deposit_waiver", "group_limit", "party_limit"}
DATE_EXPRS = ["next Friday", "this Saturday", "this Sunday", "day after tomorrow", "tomorrow", "next Monday", "next Tuesday",
              "coming Wednesday", "next Thursday", "Dussehra", "Diwali", "day before Diwali", "Bhai Dooj", "in 3 days",
              "a week from today", "next Saturday", "coming Sunday", "Dhanteras"]


@dataclass
class Plan:
    plan_id: str
    domain: str
    kind: str  # book dates sold_out change_mind modify cancel policy usual returning_cancel red_flag + rule_* kinds
    phone: str
    name: str
    values: dict = field(default_factory=dict)  # final booking values (canonical labels)
    first: dict | None = None  # change_mind: values before the change
    change_field: str | None = None
    date_expr: str | None = None
    setup: list[dict] = field(default_factory=list)
    booking_id: str | None = None
    question: str | None = None
    answer: str | None = None
    red_flag: str | None = None
    staff_reason: str | None = None
    status: str = "confirmed"


# ======================================================================= plans

class Planner(_Builder):
    """Samples training plans from the world. Reuses the test-set builder's inventory helpers with a
    different seed, and never reuses a test caller's phone number."""

    def __init__(self, world: World, seed: int = SEED + 7001):
        super().__init__(world)
        self.rng = random.Random(seed)
        try:
            test_phones = {c.phone for c in load_test_calls()}
        except FileNotFoundError:
            test_phones = set()
        self.phones |= test_phones
        self.used_returning |= test_phones  # never build a training call for a test call's returning customer
        self.n = 0

    def _id(self, domain: str, kind: str) -> str:
        self.n += 1
        return f"tr-{domain[:3]}-{kind}-{self.n:04d}"

    # ---- value pickers -----------------------------------------------------
    def movie_values(self, d: str | None = None, qty: int | None = None) -> dict | None:
        d = d or self.rng.choice(self.future_dates(1, 20))
        qty = qty or self.rng.choice([1, 2, 2, 2, 3, 4, 5])
        shows = self.movie_shows(d, qty)
        if not shows:
            return None
        s = self.rng.choice(shows)
        return {"movie": s["title"], "cinema": s["cinema"], "date": d, "time": fmt_time(s["start_min"]),
                "seat_category": self.rng.choice(s["ok_categories"]), "quantity": qty, "show_id": s["id"],
                "movie_id": s["movie_id"], "cinema_id": s["cinema_id"]}

    def bus_values(self, d: str | None = None, n: int | None = None, gender: str = "M", name: str = "") -> dict | None:
        route = self.route_choice()
        d = d or self.rng.choice(self.future_dates(1, 20))
        n = n or self.rng.choice([1, 1, 2, 2, 3])
        trips = self.trips(d, route["id"], n)
        if not trips:
            return None
        t = self.rng.choice(trips)
        return {"origin": route["origin"], "destination": route["destination"], "route_id": route["id"], "date": d,
                "time": fmt_time(t["depart_min"]), "bus_type": t["bus_type"], "passengers": self.passengers(name, gender, n)}

    def table_values(self, d: str | None = None, party: int | None = None) -> dict | None:
        party = party or self.rng.choice([2, 2, 3, 4, 4, 5, 6, 8])
        rest = self.rest_choice(party)
        d = d or self.rng.choice(self.future_dates(1, 20))
        times = self.table_times(rest["id"], d, party)
        if not times:
            return None
        v = {"restaurant": rest["name"], "restaurant_id": rest["id"], "date": d, "time": fmt_time(self.rng.choice(times)), "party_size": party}
        if self.rng.random() < 0.25:
            v["special_request"] = self.rng.choice(["one guest is vegetarian", "it's a birthday", "nut allergy", "need a high chair"])
        return v

    def values(self, domain: str, name: str, gender: str, d: str | None = None) -> dict | None:
        if domain == "movies":
            return self.movie_values(d)
        if domain == "buses":
            return self.bus_values(d, gender=gender, name=name)
        return self.table_values(d)

    def status_for(self, domain: str, v: dict) -> str:
        return self.table_status(v["restaurant_id"], v["party_size"]) if domain == "restaurants" else "confirmed"

    # ---- plan kinds ----------------------------------------------------------
    def plan(self, domain: str, kind: str) -> Plan | None:
        phone, name, gender = self.new_caller()
        base = dict(plan_id=self._id(domain, kind), domain=domain, kind=kind, phone=phone, name=name)
        if kind == "book":
            v = self.values(domain, name, gender)
            return Plan(**base, values=v, status=self.status_for(domain, v)) if v else None
        if kind == "dates":
            expr = self.rng.choice(DATE_EXPRS)
            r = resolve_date(expr, TODAY)
            if r.date is None or not r.in_window:
                return None
            v = self.values(domain, name, gender, r.date.isoformat())
            return Plan(**base, values=v, date_expr=expr, status=self.status_for(domain, v)) if v else None
        if kind == "change_mind":
            return self._change_mind(base, domain, name, gender)
        if kind == "sold_out":
            return self._sold_out(base, domain, gender)
        if kind in ("modify", "cancel"):
            return self._existing(base, domain, name, gender, kind)
        if kind == "policy":
            q, a = self.rng.choice(P.POLICY_QA[domain])
            return Plan(**base, question=q, answer=a)
        if kind == "red_flag":
            text, reason = self.rng.choice(P.RED_FLAGS)
            return Plan(**base, red_flag=text, staff_reason=reason)
        if kind in ("usual", "returning_cancel"):
            return self._returning(base, domain, kind)
        if kind == "rule_group":
            if domain == "movies":
                return Plan(**base, red_flag=f"I need {self.rng.choice([12, 15, 20])} tickets in one booking.", staff_reason="large_group")
            if domain == "buses":
                return Plan(**base, red_flag=f"I need {self.rng.choice([8, 10, 12])} seats on one bus for a family function.", staff_reason="large_group")
            return Plan(**base, red_flag=f"Table for {self.rng.choice([11, 14, 18])} at Hitech City please.", staff_reason="large_group")
        if kind == "rule_deposit":
            v = self.table_values(party=self.rng.choice([8, 9, 10]))
            if not v or v["party_size"] > self.inv.restaurant(v["restaurant_id"])["max_party"]:
                return None
            return Plan(**base, values=v, red_flag=f"Table for {v['party_size']} but please don't take any deposit, I'm a regular.",
                        staff_reason="payment")
        if kind == "rule_colleague":
            other_phone, other_name, other_gender = self.new_caller()
            ex = {"movies": lambda: self.movie_existing(other_phone, other_name),
                  "buses": lambda: self.bus_existing(other_phone, other_name, other_gender),
                  "restaurants": lambda: self.table_existing(other_phone, other_name)}[domain]()
            return Plan(**base, setup=[ex[0]], booking_id=ex[1]) if ex else None
        if kind == "rule_no_pax" and domain == "buses":
            v = self.bus_values(n=self.rng.choice([2, 3]), gender=gender, name=name)
            return Plan(**base, values=v) if v else None
        return None

    def _change_mind(self, base: dict, domain: str, name: str, gender: str) -> Plan | None:
        v = self.values(domain, name, gender)
        if not v:
            return None
        first = dict(v)
        if domain == "movies":
            if self.rng.random() < 0.5:
                first["quantity"] = max(1, v["quantity"] - self.rng.choice([1, 2]))
                fld = "quantity"
            else:
                others = [s for s in self.movie_shows(v["date"], v["quantity"], v["seat_category"], v["movie_id"], v["cinema_id"])
                          if fmt_time(s["start_min"]) != v["time"]]
                if not others:
                    return None
                first["time"] = fmt_time(self.rng.choice(others)["start_min"])
                fld = "time"
        elif domain == "buses":
            others = [t for t in self.trips(v["date"], v["route_id"], len(v["passengers"])) if fmt_time(t["depart_min"]) != v["time"]]
            if not others:
                return None
            t = self.rng.choice(others)
            first["time"] = fmt_time(t["depart_min"])
            fld = "time"
            v["bus_type"] = v["bus_type"]
        else:
            first["party_size"] = max(1, v["party_size"] - self.rng.choice([1, 2]))
            fld = "party_size"
            if not self.inv.free_table(v["restaurant_id"], v["date"], int(v["time"][:2]) * 60 + int(v["time"][3:]), first["party_size"]):
                return None
        return Plan(**base, values=v, first=first, change_field=fld, status=self.status_for(domain, v))

    def _sold_out(self, base: dict, domain: str, gender: str) -> Plan | None:
        # Reuse the test builder's search for sold-out requests and their nearest alternative.
        script = {"movies": self.movie_sold_out, "buses": self.bus_sold_out, "restaurants": self.table_sold_out}[domain]()
        if script is None:
            return None
        self.scripts.pop()  # the builder appended it to its own list
        want_text = script.opening
        alt = script.gold["outcomes"][0][0]
        if domain == "movies":
            show = self.w.one("""SELECT s.id, s.start_min, m.title, c.name || ' ' || c.area AS cinema FROM shows s JOIN movies m ON m.id=s.movie_id
                                 JOIN screens sc ON sc.id=s.screen_id JOIN cinemas c ON c.id=sc.cinema_id
                                 WHERE s.movie_id=? AND sc.cinema_id=? AND s.date=? AND s.start_min=?""",
                              (alt["movie_id"], alt["cinema_id"], alt["date"], int(alt["time"][:2]) * 60 + int(alt["time"][3:])))
            v = {"movie": show["title"], "cinema": show["cinema"], "date": alt["date"], "time": alt["time"],
                 "seat_category": alt["seat_category"], "quantity": alt["quantity"]}
        elif domain == "buses":
            route = self.w.one("SELECT * FROM bus_routes WHERE id=?", (alt["route_id"],))
            pax_text = script.facts["passengers (name, age, gender)"]
            pax = [{"name": a.strip(), "age": int(b), "gender": {"male": "M", "female": "F"}[c.strip()]}
                   for a, b, c in (x.split(",") for x in pax_text.split(";"))]
            v = {"origin": route["origin"], "destination": route["destination"], "route_id": route["id"], "date": alt["date"],
                 "time": alt["time"], "bus_type": alt["bus_type"], "passengers": pax}
        else:
            rest = self.inv.restaurant(alt["restaurant_id"])
            v = {"restaurant": rest["name"], "restaurant_id": rest["id"], "date": alt["date"], "time": alt["time"], "party_size": alt["party_size"]}
        return Plan(**{**base, "phone": script.phone, "name": script.caller_name}, values=v, red_flag=want_text,
                    status=alt.get("status", "confirmed"))

    def _existing(self, base: dict, domain: str, name: str, gender: str, kind: str) -> Plan | None:
        if kind == "cancel":
            ex = {"movies": lambda: self.movie_existing(base["phone"], name), "buses": lambda: self.bus_existing(base["phone"], name, gender),
                  "restaurants": lambda: self.table_existing(base["phone"], name)}[domain]()
            return Plan(**base, setup=[ex[0]], booking_id=ex[1]) if ex else None
        s = {"movies": self.movie_simple_change, "buses": self.bus_simple_change, "restaurants": self.table_simple_change}[domain]()
        if s is None:
            return None
        self.scripts.pop()
        g = s.gold["outcomes"][0][0]
        v = {k: g[k] for k in ("date", "time") if k in g}
        if domain == "movies":
            v.update(seat_category=g["seat_category"], quantity=g["quantity"], movie_id=g["movie_id"], cinema_id=g["cinema_id"])
        elif domain == "buses":
            v.update(route_id=g["route_id"], bus_type=g["bus_type"])
        else:
            v.update(restaurant_id=g["restaurant_id"], party_size=g["party_size"])
        return Plan(**{**base, "phone": s.phone, "name": s.caller_name}, setup=s.setup, booking_id=g["booking_id"], values=v,
                    change_field=next(k for k in ("new show time", "new date", "new time", "new number of guests") if k in s.facts),
                    status=g.get("status", "confirmed"))

    def _returning(self, base: dict, domain: str, kind: str) -> Plan | None:
        i = 1 if kind == "returning_cancel" else 0
        for _ in range(5):
            s = self.returning(domain, i)
            if s is None:
                return None
            self.scripts.pop()
            g = s.gold["outcomes"][0][0]
            if kind == "returning_cancel" and g.get("status") == "cancelled":
                view = self.w.describe_booking(g["booking_id"])
                return Plan(**{**base, "phone": s.phone, "name": s.caller_name}, booking_id=g["booking_id"], values={"date": view["date"]})
            if kind == "usual" and g.get("status") != "cancelled":
                p = self.profiles[s.phone]
                if domain == "movies":
                    show = self.w.one("""SELECT m.title, c.name || ' ' || c.area AS cinema FROM shows s JOIN movies m ON m.id=s.movie_id
                                         JOIN screens sc ON sc.id=s.screen_id JOIN cinemas c ON c.id=sc.cinema_id
                                         WHERE s.movie_id=? AND sc.cinema_id=? AND s.date=? AND s.start_min=?""",
                                      (g["movie_id"], g["cinema_id"], g["date"], int(g["time"][:2]) * 60 + int(g["time"][3:])))
                    v = {"movie": show["title"], "cinema": show["cinema"], "date": g["date"], "time": g["time"],
                         "seat_category": g["seat_category"], "quantity": g["quantity"]}
                elif domain == "buses":
                    route = self.w.one("SELECT * FROM bus_routes WHERE id=?", (g["route_id"],))
                    v = {"origin": route["origin"], "destination": route["destination"], "route_id": route["id"], "date": g["date"],
                         "time": g["time"], "bus_type": g["bus_type"], "passengers": p["passengers"]}
                else:
                    rest = self.inv.restaurant(g["restaurant_id"])
                    v = {"restaurant": rest["name"], "restaurant_id": rest["id"], "date": g["date"], "time": g["time"], "party_size": g["party_size"]}
                return Plan(**{**base, "phone": s.phone, "name": s.caller_name}, values=v, status=g.get("status", "confirmed"))
        return None


# ===================================================================== dialogue

@dataclass
class Event:
    type: str
    slots: dict = field(default_factory=dict)


class Sim:
    """Shared state of one generated call: what the caller has said and where the booking stands."""

    def __init__(self, plan: Plan, cfg: SystemConfig, rng: random.Random):
        self.plan, self.cfg, self.rng = plan, cfg, rng
        self.known: dict = {}
        self.asked: str | None = None
        self.event = Event("start")
        self.turn_results: list[tuple[str, dict, dict]] = []
        self.stage = "collect"
        self.offers: list[dict] = []
        self.changed = False
        self.resisted = False
        self.answered = False
        self.explained = False
        if plan.kind == "change_mind":
            self.target = dict(plan.first)
        else:
            self.target = dict(plan.values)
        self.examples: list[dict] = []
        self.turn_no = 0
        self.date_turn: int | None = None  # caller turn in which the date was last said in words
        self.date_said_back = False  # A2: the large model has just said the resolved date back to the caller

    # ---------------------------------------------------------------- tool feedback
    def on_tool(self, name: str, args: dict, res: dict) -> None:
        self.turn_results.append((name, args, res))
        if name == "resolve_date" and res.get("ok") and res.get("date"):
            self.known["date"] = res["date"]
        if res.get("error") == "needs_readback":
            self.stage = "await_confirm"
        if name in WRITE_TOOLS and res.get("ok"):
            self.stage = "done"

    # ---------------------------------------------------------------- routing targets
    def route(self) -> tuple[str, str]:
        ev = self.event
        if ev.type in STAFF_EVENTS:
            return "staff", {"red_flag": "complaint, payment or out-of-scope request", "deposit_waiver": "deposit waiver",
                             "group_limit": "group above the booking limit", "party_limit": "party above the branch limit"}[ev.type]
        if ev.type == "change":
            return "llm", f"caller changed {self.plan.change_field}"
        if ev.type == "usual":
            return "llm", "returning caller wants the usual booking"
        if ev.type == "resist":
            return "llm", "caller will not give passenger details"
        if "date" in ev.slots and not self.cfg.date_tools:
            return "llm", "date in words and no date tool"
        return "self", ""

    # ---------------------------------------------------------------- the oracle's next step
    def search_args(self) -> dict:
        k, d = self.known, self.plan.domain
        when = self.search_date(k["date"])
        if d == "movies":
            return {"domain": d, "date": when, "movie": k["movie"], "cinema": k["cinema"], "time": k["time"],
                    "seat_category": k["seat_category"], "quantity": k["quantity"]}
        if d == "buses":
            return {"domain": d, "date": when, "origin": k.get("origin", "Hyderabad"), "destination": k["destination"],
                    "time": k["time"], "quantity": len(k["passengers"])}
        return {"domain": d, "date": when, "time": k["time"], "restaurant": k["restaurant"], "party_size": k["party_size"]}

    def search_date(self, iso: str) -> str:
        """The date as the model can know it: the caller's words if said this turn (searches resolve them),
        otherwise the ISO date the booking record already holds."""
        if self.cfg.date_tools and self.date_turn == self.turn_no and self.known.get("date_words"):
            return self.known["date_words"]
        return iso

    def pick_option(self, res: dict) -> str | None:
        k, d = self.known, self.plan.domain
        if d == "movies":
            for s in res.get("results", []):
                if s["time"] == k["time"]:
                    for o in s["options"]:
                        if o["category"] == k["seat_category"] and o["available"]:
                            return o["option_id"]
        elif d == "buses":
            for t in res.get("results", []):
                if t["departs"] == k["time"] and t["available"]:
                    return t["option_id"]
        else:
            for r in res.get("results", []):
                for t in r.get("times", []):
                    if t["time"] == k["time"]:
                        return t["option_id"]
        return None

    def write_action(self, option_id: str) -> tuple:
        k, p = self.known, self.plan
        if p.kind == "modify":
            return ("tool", "modify_booking", {"booking_id": p.booking_id, "phone": p.phone, "new_option_id": option_id})
        args = {"domain": p.domain, "option_id": option_id, "name": k.get("name", p.name)}
        if p.domain == "movies":
            args["quantity"] = k["quantity"]
        elif p.domain == "buses":
            args["passengers"] = k["passengers"]
        else:
            args["party_size"] = k["party_size"]
            if self.target.get("special_request"):
                args["special_request"] = self.target["special_request"]
        return ("tool", "create_booking", args)

    def ask(self, slot: str) -> tuple:
        self.asked = slot
        key = {"time": {"buses": "bus_time", "restaurants": "res_time"}.get(self.plan.domain, "time")}.get(slot, slot)
        return ("say", self.rng.choice(P.ASK[key]))

    def missing(self) -> str | None:
        for s in SLOT_ORDER[self.plan.domain]:
            if s not in self.known:
                return s
        return None

    def next_action(self, actor: str) -> tuple:
        p, k = self.plan, self.known
        last = self.turn_results[-1] if self.turn_results else None
        if last:
            name, args, res = last
            if name == "search_policy":
                self.answered = True
                return ("say", p.answer)
            if name == "search_inventory" and res.get("ok"):
                opt = self.pick_option(res)
                if opt:
                    return self.write_action(opt)
                return ("tool", "find_alternatives", self.search_args())
            if name == "find_alternatives" and res.get("ok"):
                seen, offers = set(), []
                for o in res["alternatives"]:
                    key = (o.get("cinema") or o.get("restaurant") or o.get("route"), o["date"], o.get("time") or o.get("departs"))
                    if key not in seen:
                        seen.add(key)
                        offers.append(o)
                self.offers = offers[:3]
                self.stage = "offered"
                if not self.offers:
                    return ("say", "I'm sorry, nothing similar is available. Would you like another date?")
                opts = "; ".join(f"{o.get('cinema') or o.get('restaurant') or o.get('route')}, {P.say_date(o['date'], random.Random(0))} "
                                 f"at {o['spoken_time']}" + (f" ({o['category']})" if o.get("category") else "") for o in self.offers)
                return ("say", f"Sorry, that one is full. I can offer: {opts}. Which would you like?")
            if name == "lookup_booking" and res.get("ok"):
                for b in res.get("bookings", []):
                    if b["date"] == p.values["date"]:
                        return ("tool", "cancel_booking", {"booking_id": b["booking_id"], "phone": p.phone})
            if name in WRITE_TOOLS and res.get("error") in ("rule_violation", "not_found_or_mismatch", "not_active"):
                self.explained = True
                why = res.get("message", "").split(". If")[0]
                return ("say", f"I'm sorry, I can't do that: {why} I can connect you to our staff if you'd like.")
        ev = self.event
        if ev.type in STAFF_EVENTS:  # only the large model reaches here (the small model routes to staff)
            return ("tool", "handoff_to_staff", {"reason": p.staff_reason or "other", "summary": f"{p.name}: {p.red_flag}"})
        if ev.type == "policy_q":
            return ("tool", "search_policy", {"query": p.question, "domain": p.domain})
        if ev.type == "problem_intro":
            self.asked = "problem"
            return ("say", self.rng.choice(P.ASK["problem"]))
        if ev.type == "resist":
            self.asked = "passengers"
            return ("say", "I'm sorry, every passenger's name, age and gender is needed to issue the tickets. Could you share them?")
        if p.kind in ("cancel", "rule_colleague"):
            if "booking_id" in k:
                return ("tool", "cancel_booking", {"booking_id": k["booking_id"], "phone": p.phone})
            return self.ask("booking_id")
        if p.kind == "returning_cancel":
            return ("tool", "lookup_booking", {"phone": p.phone})
        if p.kind == "modify":
            if "booking_id" not in k:
                return self.ask("booking_id")
            if "new" not in k:
                return self.ask({"new show time": "new_time", "new date": "new_date", "new time": "new_time",
                                 "new number of guests": "new_party"}[p.change_field])
            if p.domain == "restaurants" and p.change_field == "new number of guests":
                return ("tool", "modify_booking", {"booking_id": p.booking_id, "phone": p.phone, "party_size": p.values["party_size"]})
            return ("tool", "search_inventory", self.modify_search_args())
        if ev.type == "usual" and actor == "large":
            v = self.target
            fill = {s: v[s] for s in SLOT_ORDER[p.domain] if s in v and s not in ("time", "date")}
            fill["name"] = p.name
            if p.domain == "buses":
                fill["origin"] = v["origin"]
            k.update(fill)
            what = {"movies": f"{v.get('quantity')} {v.get('seat_category')} tickets at {v.get('cinema')}",
                    "buses": f"{len(v.get('passengers', []))} seat(s) from {v.get('origin')} to {v.get('destination')}",
                    "restaurants": f"a table for {v.get('party_size')} at {v.get('restaurant')}"}[p.domain]
            if "time" not in k:
                self.asked = "time"
                return ("say", f"Welcome back! Your usual is {what}. Which time would you like?")
        slot = self.missing()
        if slot:
            action = self.ask(slot)
            if actor == "large" and not self.cfg.date_tools and "date" in ev.slots:
                # Without date tools the large model works the date out and says it back; on the next turn
                # the small model copies it into slots.date as YYYY-MM-DD, so the record keeps it after the
                # line has scrolled out of the transcript (data v2 lacked this and the date was unlearnable).
                from fastlane.gateway.dates import spoken_date as _sd

                action = ("say", f"Sure, {_sd(date.fromisoformat(k['date']))}. " + action[1])
                self.date_said_back = True
            return action
        return ("tool", "search_inventory", self.search_args())

    def modify_search_args(self) -> dict:
        v, p = self.plan.values, self.plan
        view = self.booking_view
        when = self.search_date(v["date"])
        if p.domain == "movies":
            return {"domain": "movies", "date": when, "movie": view["movie"], "cinema": view["cinema"], "time": v["time"],
                    "seat_category": v["seat_category"], "quantity": v["quantity"]}
        if p.domain == "buses":
            return {"domain": "buses", "date": when, "origin": view["origin"], "destination": view["destination"],
                    "time": v["time"], "quantity": view["quantity"]}
        return {"domain": "restaurants", "date": when, "time": v["time"], "restaurant": view["restaurant"], "party_size": v["party_size"]}

    def modify_pick(self, res: dict) -> str | None:
        saved, self.known = self.known, {**self.known, **{k: self.plan.values[k] for k in ("time", "seat_category") if k in self.plan.values}}
        try:
            return self.pick_option(res)
        finally:
            self.known = saved


class Caller:
    """Rule-based caller: answers what the agent asked, confirms read-backs, follows the plan's twist."""

    def __init__(self, sim: Sim):
        self.sim, self.p, self.rng = sim, sim.plan, sim.rng
        self.revealed: set[str] = set()

    def _give(self, slot: str) -> tuple[str, dict]:
        """Spoken phrase for one slot and the canonical slot value it reveals."""
        v, rng, d = self.sim.target, self.rng, self.p.domain
        if slot == "date":
            words = self.p.date_expr or P.say_date(v["date"], rng)
            return rng.choice(P.GIVE["date"]).format(v=words), self._said_date(words, v["date"])
        if slot == "name":
            self.sim.known["name"] = self.p.name
            return rng.choice(P.GIVE["name"]).format(v=self.p.name), {"caller_name": self.p.name}
        if slot == "destination":
            self.sim.known["destination"] = v["destination"]
            self.sim.known["origin"] = v["origin"]
            o, dd = rng.choice(P.CITY_SAY[v["origin"]]), rng.choice(P.CITY_SAY[v["destination"]])
            if v["origin"] != "Hyderabad" or rng.random() < 0.5:
                return rng.choice(P.GIVE["route"]).format(o=o, v=dd), {"origin": v["origin"], "destination": v["destination"]}
            return rng.choice(P.GIVE["destination"]).format(v=dd), {"destination": v["destination"]}
        self.sim.known[slot] = v[slot]
        if slot == "movie":
            return rng.choice(P.GIVE["movie"]).format(v=v["movie"]), {"movie": v["movie"]}
        if slot == "cinema":
            return rng.choice(P.GIVE["cinema"]).format(v=P.say_cinema(v["cinema"], rng)), {"cinema": v["cinema"]}
        if slot == "time":
            if d == "buses":
                bt = v["bus_type"].replace("_", " ").replace("nonac", "non-AC").replace("ac ", "AC ")
                spoken = P.say_time(v["time"], rng) + (f" {bt}" if rng.random() < 0.5 else "")
                return rng.choice(P.GIVE["bus_time"]).format(v=spoken), {"time": v["time"]}
            return rng.choice(P.GIVE["time"] if d == "movies" else ["{v}", "at {v}", "{v} please"]).format(v=P.say_time(v["time"], rng)), {"time": v["time"]}
        if slot == "seat_category":
            return rng.choice(P.GIVE["seat_category"]).format(v=v["seat_category"]), {"seat_category": v["seat_category"]}
        if slot == "quantity":
            return rng.choice(P.GIVE["quantity"]).format(v=P.say_number(v["quantity"], rng)), {"quantity": v["quantity"]}
        if slot == "passengers":
            return rng.choice(P.GIVE["passengers"]).format(v=P.say_passengers(v["passengers"], rng)), {"passengers": v["passengers"]}
        if slot == "restaurant":
            return rng.choice(P.GIVE["restaurant"]).format(v=P.say_restaurant(v["restaurant"], rng)), {"restaurant": v["restaurant"]}
        if slot == "party_size":
            text = rng.choice(P.GIVE["party_size"]).format(v=P.say_number(v["party_size"], rng))
            if v.get("special_request"):
                text += f", and {v['special_request']}"
                return text, {"party_size": v["party_size"], "special_request": v["special_request"]}
            return text, {"party_size": v["party_size"]}
        raise KeyError(slot)

    def _said_date(self, words: str, iso: str) -> dict:
        self.sim.known["date"] = iso
        self.sim.known["date_words"] = words
        self.sim.date_turn = self.sim.turn_no
        return {"date": words}

    def _utter(self, slots: list[str], lead: str = "") -> tuple[str, dict]:
        parts, canon = [], {}
        for s in slots:
            text, c = self._give(s)
            parts.append(text)
            canon.update(c)
            self.revealed.add(s)
        body = ", ".join(parts)
        return (rng_choice(self.rng, P.FILLERS) + (lead + " " if lead else "") + body).strip() + rng_choice(self.rng, P.POLITE_TAIL), canon

    def opening(self) -> str:
        p, rng, d = self.p, self.rng, self.p.domain
        sim = self.sim
        if p.kind in ("red_flag", "rule_group"):
            if p.kind == "red_flag" and rng.random() < 0.3:
                sim.event = Event("problem_intro")
                return rng.choice(["Hello, I have a problem.", "Hi, I need help with something.", "Listen, there is an issue."])
            sim.event = Event("group_limit" if p.kind == "rule_group" else "red_flag")
            return p.red_flag
        if p.kind == "rule_deposit":
            sim.event = Event("deposit_waiver", {"restaurant": p.values["restaurant"], "party_size": p.values["party_size"]})
            return f"Hi, I want a table at {P.say_restaurant(p.values['restaurant'], rng)}. {p.red_flag}"
        if p.kind == "policy":
            sim.event = Event("policy_q")
            return rng.choice(["", "Hi, quick question. ", "Hello, ", "Namaste, I wanted to ask - "]) + p.question
        if p.kind == "sold_out":
            sim.event = Event("give")
            return self._sold_out_opening()
        if p.kind in ("cancel", "rule_colleague"):
            ref = P.say_ref(p.booking_id, rng)
            if rng.random() < 0.5:
                sim.known["booking_id"] = p.booking_id
                sim.event = Event("give", {"booking_id": p.booking_id, "intent": "cancel"})
                who = "it's my colleague's booking" if p.kind == "rule_colleague" else "please cancel it"
                return f"Hi, I want to cancel booking {ref}, {who}."
            sim.event = Event("give", {"intent": "cancel"})
            return rng.choice(["Hello, I need to cancel a booking.", "Hi, please cancel my booking.", "I want to cancel my reservation."])
        if p.kind == "returning_cancel":
            sim.event = Event("give", {"intent": "cancel", "caller_name": p.name})
            what = {"movies": "movie tickets", "buses": "bus ticket", "restaurants": "table booking"}[d]
            return f"Hi, this is {p.name.split()[0]}. Please cancel my {what} for {P.say_date(p.values['date'], rng)}."
        if p.kind == "modify":
            sim.event = Event("give", {"intent": "modify"})
            what = {"movies": "movie booking", "buses": "bus ticket", "restaurants": "table booking"}[d]
            return rng.choice([f"Hi, I want to change my {what}.", f"Hello, I need to make a change to my {what}."])
        if p.kind == "usual":
            sim.event = Event("usual", {"caller_name": p.name})
            first = p.name.split()[0]
            if d == "movies":
                text, canon = self._utter(["movie", "date"])
                sim.event.slots.update(canon)
                return f"Hi, it's {first} again. Book my usual seats - {text}"
            text, canon = self._utter(["date"])
            sim.event.slots.update(canon)
            return {"buses": f"Hello, {first} here. Same bus as last time - {text}",
                    "restaurants": f"Hi, {first} here, my usual table - {text}"}[d]
        # Booking kinds: reveal 0-3 of the first slots in the opening.
        intro = {"movies": ["I want to book movie tickets", "Hello, tickets please", "Hi, I'd like to watch a movie"],
                 "buses": ["I need a bus ticket", "Hi, I want to book a bus", "Hello, bus booking please"],
                 "restaurants": ["I want to book a table", "Hi, table reservation please", "Hello, I'd like to reserve a table"]}[d]
        order = SLOT_ORDER[d][:-1]
        n = rng.choice([0, 1, 1, 2, 2, 3])
        if p.kind == "dates":
            chosen = ["date"] + [s for s in order[:n] if s != "date"]
        else:
            chosen = order[:n]
        sim.event = Event("give", {"intent": "book"})
        if not chosen:
            return rng.choice(intro) + "."
        text, canon = self._utter(chosen, lead="")
        sim.event.slots.update(canon)
        return f"{rng.choice(intro)} - {text}"

    def _sold_out_opening(self) -> str:
        # The plan's red_flag field holds the test builder's opening for the sold-out request.
        p, v = self.p, self.sim.target
        self.sim.target = dict(v)
        req = self._requested_sold_out()
        self.sim.known.update(req["known"])  # date / date_words were set by _said_date
        self.revealed |= set(req["known"])
        self.sim.event.slots.update(req["slots"])
        return req["text"]

    def _requested_sold_out(self) -> dict:
        """Parse back the sold-out request the builder generated (values differ from the accepted alternative)."""
        p = self.p
        text = p.red_flag
        r = self.sim
        known, slots = {}, {}
        # The request is always for the same day as the alternative (movies, restaurants) or near it (buses).
        if p.domain == "movies":
            import re

            m = re.search(r"(\d+) (\w+) tickets for (.+) at (\w+), .+ (\d{1,2}(?::\d{2})? [AP]M) show", text)
            qty, cat, movie, brand, t = int(m.group(1)), m.group(2), m.group(3), m.group(4), m.group(5)
            from fastlane.gateway.dates import parse_time

            tm = parse_time(t)[0]
            words = re.search(r"at \w+, (.+?) \d{1,2}(?::\d{2})? [AP]M show", text).group(1)
            known = {"movie": movie, "cinema": r.target["cinema"], "time": f"{tm // 60:02d}:{tm % 60:02d}",
                     "seat_category": cat, "quantity": qty}
            slots = {"movie": movie, "cinema": r.target["cinema"], "time": known["time"], "seat_category": cat, "quantity": qty,
                     **self._said_date(words, r.target["date"])}
        elif p.domain == "buses":
            import re

            from fastlane.gateway.dates import parse_time

            m = re.search(r"on (\d+\w+ \w+), the (.+) bus", text)
            d = resolve_date(m.group(1), TODAY).date.isoformat()
            tm = parse_time(m.group(2))[0]
            known = {"destination": r.target["destination"], "origin": r.target["origin"], "time": f"{tm // 60:02d}:{tm % 60:02d}"}
            slots = {"destination": r.target["destination"], "time": known["time"], **self._said_date(m.group(1), d)}
        else:
            import re

            from fastlane.gateway.dates import parse_time

            m = re.search(r"table for (\d+) at .+ on (\d+\w+ \w+) at (.+) please", text)
            tm = parse_time(m.group(3))[0]
            known = {"restaurant": r.target["restaurant"], "time": f"{tm // 60:02d}:{tm % 60:02d}", "party_size": int(m.group(1))}
            slots = {**known, **self._said_date(m.group(2), r.target["date"])}
        return {"text": text, "known": known, "slots": slots}

    def respond(self, agent_text: str) -> str | None:
        sim, p, rng = self.sim, self.p, self.rng
        if sim.stage == "done" or agent_text.startswith(("Done", "Your table is held")):
            sim.event = Event("bye")
            return rng.choice(P.BYE)
        if agent_text.startswith("Just to confirm"):
            if p.kind == "change_mind" and not sim.changed:
                sim.changed = True
                fld = p.change_field
                sim.target = dict(p.values)
                sim.known[fld] = p.values[fld]
                if fld == "quantity":
                    say = f"make it {P.say_number(p.values['quantity'], rng)} tickets"
                elif fld == "time":
                    say = f"make it the {P.say_time(p.values['time'], rng)} one instead"
                else:
                    say = f"make it {P.say_number(p.values['party_size'], rng)} people, more are coming"
                sim.event = Event("change", {fld: p.values[fld]})
                return rng.choice(["No wait, ", "Actually, ", "Sorry, one change - "]) + say + "."
            sim.event = Event("confirm")
            return rng.choice(P.CONFIRM)
        if sim.stage == "offered":
            alt = next((o for o in sim.offers if self._matches_target(o)), None)
            if alt is None:
                alt = next((o for o in sim.offers if self._acceptable(o)), None)
                if alt is None:
                    sim.event = Event("bye")
                    return "No, that doesn't work for me. Thanks, bye."
                self._take_offer(alt)
            sim.stage = "collect"
            sim.known.update({k: sim.target[k] for k in sim.target if k in SLOT_ORDER[p.domain] or k in ("origin",)})
            sim.event = Event("accept_alt")
            return rng.choice(P.ACCEPT_ALT).format(v=P.say_time(sim.target["time"], rng))
        if sim.answered or sim.explained:
            sim.event = Event("bye")
            return rng.choice(["Okay, got it. Thank you, bye.", "Fine, leave it then. Bye.", "Okay, thanks.", "Alright, thank you."])
        asked = sim.asked
        sim.asked = None
        if asked == "problem":
            sim.event = Event("red_flag")
            return p.red_flag
        if asked == "booking_id":
            sim.known["booking_id"] = p.booking_id
            sim.event = Event("give", {"booking_id": p.booking_id})
            return rng.choice(P.GIVE["booking_id"]).format(v=P.say_ref(p.booking_id, rng))
        if asked in ("new_time", "new_date", "new_party"):
            sim.known["new"] = True
            v = p.values
            if asked == "new_party":
                sim.event = Event("give", {"party_size": v["party_size"]})
                return f"{P.say_number(v['party_size'], rng)} people now"
            if asked == "new_date":
                words = P.say_date(v["date"], rng)
                sim.event = Event("give", self._said_date(words, v["date"]))
                return f"{words}, same timing"
            sim.event = Event("give", {"time": v["time"]})
            return f"the {P.say_time(v['time'], rng)} one"
        if asked == "passengers" and p.kind == "rule_no_pax" and not sim.resisted:
            sim.resisted = True
            sim.event = Event("resist")
            return rng.choice(["Just put my name, I'll sort out the rest with the conductor.", "Why do you need all that? Just book it in my name."])
        slots = [asked] if asked in SLOT_ORDER[p.domain] else []
        remaining = [s for s in SLOT_ORDER[p.domain][:-1] if s not in self.revealed and s not in slots and s not in sim.known]
        if slots and remaining and rng.random() < 0.25:
            slots.append(remaining[0])  # callers often add the next detail unprompted
        if not slots:
            slots = remaining[:1] or (["name"] if "name" not in sim.known else [])
        if not slots:
            sim.event = Event("confirm")
            return "Yes."
        text, canon = self._utter(slots)
        sim.event = Event("give", canon)
        return text

    def _acceptable(self, offer: dict) -> bool:
        """A training caller takes another offer at the same venue (same branch for restaurants)."""
        t = self.sim.target
        if self.p.domain == "restaurants":
            return offer.get("restaurant") == t["restaurant"]
        if self.p.domain == "movies":
            return offer.get("movie") == t["movie"]
        return True

    def _take_offer(self, o: dict) -> None:
        t = self.sim.target
        t["date"] = o["date"]
        t["time"] = o.get("time") or o.get("departs")
        if self.p.domain == "movies":
            t["cinema"], t["seat_category"] = o["cinema"], o["category"]
        elif self.p.domain == "buses":
            t["bus_type"] = o["bus_type"]

    def _matches_target(self, offer: dict) -> bool:
        t = self.sim.target
        return offer.get("date") == t["date"] and offer.get("time", offer.get("departs")) == t["time"] and \
            (self.p.domain != "movies" or offer.get("category") == t["seat_category"])


def rng_choice(rng: random.Random, items: list[str]) -> str:
    return rng.choice(items)


# ===================================================================== oracle models

class OracleSmall:
    name = "oracle-small"

    def __init__(self, sim: Sim):
        self.sim = sim

    def chat(self, messages, **kw) -> LLMResponse:
        sim = self.sim
        first_step = len(messages) == 2
        if first_step:
            route, reason = sim.route()
            slots = dict(sim.event.slots)
            if sim.date_said_back:
                sim.date_said_back = False
                slots.setdefault("date", sim.known["date"])
        else:
            route, reason, slots = "self", "", {}
        if route != "self":
            out = {"route": route, "reason": reason, "slots": slots, "tool": None, "say": ""}
        else:
            if sim.plan.kind == "modify" and sim.turn_results and sim.turn_results[-1][0] == "search_inventory":
                opt = sim.modify_pick(sim.turn_results[-1][2])
                action = sim.write_action(opt) if opt else ("say", "Sorry, that time is not available.")
            else:
                action = sim.next_action("small")
            if action[0] == "tool":
                out = {"route": "self", "reason": "", "slots": slots, "tool": {"name": action[1], "args": action[2]}, "say": ""}
            else:
                out = {"route": "self", "reason": "", "slots": slots, "tool": None, "say": action[1]}
        text = json.dumps(out, ensure_ascii=False)
        sim.examples.append({"messages": [dict(m) for m in messages] + [{"role": "assistant", "content": text}],
                             "meta": {"plan": sim.plan.plan_id, "domain": sim.plan.domain, "kind": sim.plan.kind,
                                      "config": sim.cfg.name, "route": route, "step": 0 if first_step else len(messages) // 2,
                                      "event": sim.event.type}})
        return LLMResponse(text, usage=Usage(), latency_ms=0.0, model=self.name)


class OracleLarge:
    name = "oracle-large"

    def __init__(self, sim: Sim):
        self.sim = sim
        self.n = 0

    def chat(self, messages, tools=None, **kw) -> LLMResponse:
        sim = self.sim
        if messages and messages[0]["content"].startswith("You route turns"):  # B3 router (not used for data)
            return LLMResponse("EASY", model=self.name)
        if sim.plan.kind == "modify" and sim.turn_results and sim.turn_results[-1][0] == "search_inventory":
            opt = sim.modify_pick(sim.turn_results[-1][2])
            action = sim.write_action(opt) if opt else ("say", "Sorry, that time is not available.")
        else:
            action = sim.next_action("large")
        if action[0] == "tool":
            self.n += 1
            call = ToolCall(f"oracle_{self.n}", action[1], action[2], json.dumps(action[2]))
            return LLMResponse(None, [call], model=self.name, finish_reason="tool_calls")
        return LLMResponse(action[1], model=self.name)


class SimAgent(Agent):
    def __init__(self, *args, sim: Sim, **kw):
        super().__init__(*args, **kw)
        self.sim = sim

    def _tool(self, name, args, by, tr):
        res = super()._tool(name, args, by, tr)
        self.sim.on_tool(name, args, res)
        return res


# ===================================================================== running

def expected_outcome(plan: Plan, sim: Sim) -> list[list[dict]]:
    v = sim.target
    if plan.kind in ("policy", "red_flag", "rule_group", "rule_colleague", "rule_deposit"):
        return [[]]
    if plan.kind in ("cancel", "returning_cancel"):
        return [[{"booking_id": plan.booking_id, "status": "cancelled"}]]
    if plan.kind == "modify":
        g = {"booking_id": plan.booking_id, **{k: plan.values[k] for k in ("date", "time", "party_size", "seat_category", "quantity") if k in plan.values}}
        return [[g]]
    if plan.domain == "movies":
        g = {"domain": "movies", "movie": v["movie"], "cinema": v["cinema"], "date": v["date"], "time": v["time"],
             "seat_category": v["seat_category"], "quantity": v["quantity"], "status": "confirmed"}
    elif plan.domain == "buses":
        g = {"domain": "buses", "origin": v["origin"], "destination": v["destination"], "date": v["date"], "time": v["time"],
             "quantity": len(v["passengers"]), "passenger_first_names": sorted(x["name"].split()[0].lower() for x in v["passengers"]),
             "status": "confirmed"}
    else:
        g = {"domain": "restaurants", "restaurant": v["restaurant"], "date": v["date"], "time": v["time"],
             "party_size": v["party_size"], "status": plan.status}
    return [[g]]


def run_plan(plan: Plan, cfg: SystemConfig, world_path, index: PolicyIndex, seed: int, max_turns: int = 14) -> dict:
    rng = random.Random(seed)
    world = World.fresh(world_path)
    if plan.setup:
        setup_world(world, CallScript(plan.plan_id, plan.domain, plan.kind, plan.phone, plan.name, "", "", "", {}, setup=plan.setup))
    before = booking_views(world)
    sim = Sim(plan, cfg, rng)
    if plan.booking_id:
        sim.booking_view = world.describe_booking(plan.booking_id)
    agent = SimAgent(world, plan.phone, cfg, small=OracleSmall(sim), large=OracleLarge(sim), policy_index=index,
                     compact_small_prompt=True, sim=sim, raise_errors=True)
    caller = Caller(sim)
    error = None
    try:
        agent.greet()
        sim.turn_no = 1  # the opening line is caller turn 1
        text = caller.opening()
        for _ in range(max_turns):
            sim.turn_results = []
            reply = agent.respond(text, stt_partials(text))
            sim.turn_no += 1
            if agent.ended:
                break
            text = caller.respond(reply)
            if text is None:
                break
    except Exception as e:  # a generator bug drops the call, it never reaches the dataset
        error = f"{type(e).__name__}: {e}"
    finally:
        agent.close()
    changed = changed_bookings(before, booking_views(world))
    ok = error is None and any(outcome_matches(o, changed) for o in expected_outcome(plan, sim))
    if plan.kind in ("red_flag", "rule_group", "rule_deposit"):
        ok = ok and agent.gw.handoff is not None
    if plan.kind == "policy":
        ok = ok and sim.answered
    return {"plan": plan.plan_id, "ok": ok, "error": error, "examples": sim.examples if ok else [], "turns": len(agent.traces) - 1,
            "transcript": [(t.caller, t.agent, t.handled_by) for t in agent.traces[1:]]}

"""The 360 scripted test calls (Section 8): 120 per domain, never used for training.

Category mix (proposal): simple book/change/cancel 120, dates 45, sold out 45, changes mind 30,
policy 40, returning callers 40, red flags / out of scope 25, rule-breaking 15.

Each script says who the caller is, what they want, which details they give only when asked, any
booking that must exist before the call (setup), and the acceptable final states (gold). Gold is
computed against the practice world, so every script is achievable.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from fastlane.config import DATA_DIR, SEED, SYSTEMS, TODAY
from fastlane.gateway.dates import resolve_date, spoken_date
from fastlane.gateway.gateway import Gateway
from fastlane.gateway.inventory import Inventory
from fastlane.gateway.readback import spoken_time
from fastlane.world import names
from fastlane.world.db import World, fmt_time

TEST_CALLS = DATA_DIR / "test_calls.jsonl"

MIX = {  # category -> (movies, buses, restaurants)
    "simple": (40, 40, 40),
    "dates": (15, 15, 15),
    "sold_out": (15, 15, 15),
    "change_mind": (10, 10, 10),
    "policy": (14, 13, 13),
    "returning": (13, 14, 13),
    "red_flag": (8, 8, 9),
    "rule_break": (5, 5, 5),
}
STYLES = [
    "polite and brief", "chatty, uses a few Hindi words like 'haan' and 'theek hai'", "in a hurry, short answers",
    "slightly confused, asks to repeat once", "formal", "friendly, uses 'ji'", "uses some Telugu words like 'avunu' and 'sari'",
]
DOMAIN_IDX = {"movies": 0, "buses": 1, "restaurants": 2}


@dataclass
class CallScript:
    call_id: str
    domain: str
    category: str
    phone: str
    caller_name: str
    style: str
    goal: str  # what the caller wants, in plain words, for the simulator
    opening: str  # the caller's first line
    facts: dict  # details given only when asked
    hard_moments: list[str] = field(default_factory=list)  # expected hard moments in the call
    setup: list[dict] = field(default_factory=list)  # tool calls run (unchecked) before the call
    gold: dict = field(default_factory=dict)  # {"outcomes": [[expected change, ...], ...], "handoff": bool|None, "answer_any": [...]}
    inject_failure: str | None = None  # tool to fail once (10% of calls)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "CallScript":
        return cls(**d)


def load_test_calls(path: Path = TEST_CALLS) -> list[CallScript]:
    return [CallScript.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def setup_world(world: World, script: CallScript) -> list[str]:
    """Create the script's pre-existing bookings (no read-back gate). Returns their ids."""
    ids = []
    for step in script.setup:
        gw = Gateway(world, step["phone"], config=SYSTEMS["A3"])
        gw.begin_turn("setup")
        res = gw.call(step["tool"], step["args"])
        if not res.get("ok"):
            raise RuntimeError(f"{script.call_id}: setup failed: {res}")
        ids.append(res["booking"]["booking_id"])
    return ids


def _say_date(d: str) -> str:
    x = date.fromisoformat(d)
    suffix = "th" if 11 <= x.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(x.day % 10, "th")
    return f"{x.day}{suffix} {x.strftime('%B')}"


class _Builder:
    def __init__(self, world: World):
        self.w = world
        self.now = world.now
        self.inv = Inventory(world, self.now)
        self.rng = random.Random(SEED + 101)
        self.phones = {r["phone"] for r in world.q("SELECT phone FROM customers")}
        self.scripts: list[CallScript] = []
        self.counter = 0
        self.profiles = world.profiles()
        self.used_returning: set[str] = set()

    # ------------------------------------------------------------ helpers
    def new_caller(self) -> tuple[str, str, str]:
        name, gender = names.person(self.rng)
        return names.phone(self.rng, self.phones), name, gender

    def add(self, domain: str, category: str, **kw) -> CallScript:
        self.counter += 1
        kw.setdefault("style", self.rng.choice(STYLES))
        s = CallScript(call_id=f"{domain[:3]}-{category}-{self.counter:03d}", domain=domain, category=category, **kw)
        self.scripts.append(s)
        return s

    def future_dates(self, lo: int = 1, hi: int = 20) -> list[str]:
        return [(TODAY + timedelta(days=i)).isoformat() for i in range(lo, hi + 1)]

    def movie_shows(self, d: str, qty: int, category: str | None = None, movie_id: str | None = None,
                    cinema_id: str | None = None, margin: int = 2) -> list[dict]:
        sql = "SELECT s.id FROM shows s JOIN screens sc ON sc.id=s.screen_id WHERE s.date=?"
        params: list = [d]
        if movie_id:
            sql += " AND s.movie_id=?"
            params.append(movie_id)
        if cinema_id:
            sql += " AND sc.cinema_id=?"
            params.append(cinema_id)
        out = []
        for r in self.w.q(sql + " ORDER BY s.id", tuple(params)):
            show = self.inv.show(r["id"])
            if datetime.fromisoformat(show["date"]) + timedelta(minutes=show["start_min"]) < self.now + timedelta(hours=5):
                continue
            cats = [category] if category else ["classic", "prime", "recliner"]
            ok = [c for c in cats if len(self.inv.free_seats(show, c)) >= qty + margin]
            if ok:
                out.append({**show, "ok_categories": ok})
        return out

    def trips(self, d: str, route_id: str, qty: int, bus_type: str | None = None, margin: int = 2) -> list[dict]:
        out = []
        for r in self.w.q("SELECT id FROM bus_trips WHERE route_id=? AND date=? ORDER BY depart_min", (route_id, d)):
            t = self.inv.trip(r["id"])
            if bus_type and t["bus_type"] != bus_type:
                continue
            if datetime.fromisoformat(t["date"]) + timedelta(minutes=t["depart_min"]) < self.now + timedelta(hours=14):
                continue
            if len(self.inv.free_berths(t)) >= qty + margin:
                out.append(t)
        return out

    def table_times(self, rid: str, d: str, party: int) -> list[int]:
        rest = self.inv.restaurant(rid)
        return [s for s in self.inv.seating_starts(rest, d) if self.inv.free_table(rid, d, s, party)]

    # ------------------------------------------------------------ gold helpers
    @staticmethod
    def movie_gold(show: dict, category: str, qty: int, status: str = "confirmed", booking_id: str | None = None) -> dict:
        g = {"domain": "movies", "movie_id": show["movie_id"], "cinema_id": show["cinema_id"], "date": show["date"],
             "time": fmt_time(show["start_min"]), "seat_category": category, "quantity": qty, "status": status}
        if booking_id:
            g["booking_id"] = booking_id
        return g

    @staticmethod
    def bus_gold(trip: dict, pax: list[dict], status: str = "confirmed", booking_id: str | None = None) -> dict:
        g = {"domain": "buses", "route_id": trip["route_id"], "date": trip["date"], "time": fmt_time(trip["depart_min"]),
             "bus_type": trip["bus_type"], "quantity": len(pax), "status": status,
             "passenger_first_names": sorted(p["name"].split()[0].lower() for p in pax)}
        if booking_id:
            g["booking_id"] = booking_id
        return g

    @staticmethod
    def table_gold(rid: str, d: str, start: int, party: int, status: str, booking_id: str | None = None) -> dict:
        g = {"domain": "restaurants", "restaurant_id": rid, "date": d, "time": fmt_time(start), "party_size": party, "status": status}
        if booking_id:
            g["booking_id"] = booking_id
        return g

    def table_status(self, rid: str, party: int) -> str:
        rest = self.inv.restaurant(rid)
        return "pending_deposit" if party >= rest["deposit_min_party"] else "confirmed"

    def passengers(self, lead_name: str, lead_gender: str, n: int) -> list[dict]:
        pax = [{"name": lead_name, "age": self.rng.randint(22, 58), "gender": lead_gender}]
        surname = lead_name.split()[-1]
        while len(pax) < n:
            nm, g = names.person(self.rng)
            pax.append({"name": f"{nm.split()[0]} {surname}", "age": self.rng.randint(8, 70), "gender": g})
        return pax

    @staticmethod
    def pax_text(pax: list[dict]) -> str:
        g = {"M": "male", "F": "female", "O": "other"}
        return "; ".join(f"{p['name']}, {p['age']}, {g[p['gender']]}" for p in pax)

    def setup_booking(self, phone: str, name: str, tool_args: dict) -> tuple[dict, str]:
        """Dry-run a setup booking on a scratch copy to learn its id (ids are deterministic)."""
        step = {"phone": phone, "tool": "create_booking", "args": {**tool_args, "name": name, "phone": phone}}
        scratch = World(self.w.conn)  # same connection; roll back afterwards
        self.w.conn.execute("SAVEPOINT scenario_setup")
        try:
            gw = Gateway(scratch, phone, config=SYSTEMS["A3"])
            gw.begin_turn("setup")
            res = gw.call("create_booking", step["args"])
            if not res.get("ok"):
                raise RuntimeError(f"setup booking failed: {res}")
            booking = res["booking"]
        finally:
            self.w.conn.execute("ROLLBACK TO scenario_setup")
            self.w.conn.execute("RELEASE scenario_setup")
        return step, booking["booking_id"]

    # ================================================================ movies
    def movie_simple_book(self, category="simple", date_expr: str | None = None, hard: list[str] | None = None) -> CallScript | None:
        phone, name, _ = self.new_caller()
        if date_expr:
            d = resolve_date(date_expr, TODAY).date.isoformat()
        else:
            d = self.rng.choice(self.future_dates(1, 18))
        qty = self.rng.choice([1, 2, 2, 2, 3, 4])
        shows = self.movie_shows(d, qty)
        if not shows:
            return None
        show = self.rng.choice(shows)
        cat = self.rng.choice(show["ok_categories"])
        spoken = date_expr or _say_date(d)
        facts = {"film": show["title"], "cinema": show["cinema"], "date": spoken, "show time": spoken_time(fmt_time(show["start_min"])),
                 "seat category": cat, "tickets": qty, "name for booking": name}
        tickets = f"{qty} ticket{'s' if qty > 1 else ''}"
        opening = self.rng.choice([
            f"Hi, I want to book {tickets} for {show['title']}.",
            f"Hello, are tickets available for {show['title']} {spoken}?",
            f"I'd like movie tickets please, {show['title']}.",
            f"Namaste, {tickets} for {show['title']} at {show['cinema'].split()[0]} please.",
        ])
        return self.add("movies", category, phone=phone, caller_name=name,
                        goal=f"Book {qty} {cat} ticket{'s' if qty > 1 else ''} for {show['title']} at {show['cinema']} on {spoken} ({spoken_date(date.fromisoformat(d))}) for the {facts['show time']} show.",
                        opening=opening, facts=facts, hard_moments=hard or [],
                        gold={"outcomes": [[self.movie_gold(show, cat, qty)]], "handoff": False})

    def movie_existing(self, phone: str, name: str, min_hours: int = 30) -> tuple[dict, str, dict, str] | None:
        """Pick a show and create a setup booking for (phone, name). Returns (step, booking_id, show, category)."""
        for _ in range(30):
            d = self.rng.choice(self.future_dates(2, 15))
            shows = self.movie_shows(d, 4)
            if not shows:
                continue
            show = self.rng.choice(shows)
            cat = show["ok_categories"][0]
            step, bid = self.setup_booking(phone, name, {"domain": "movies", "option_id": f"{show['id']}:{cat}", "quantity": 2})
            return step, bid, show, cat
        return None

    def movie_simple_change(self) -> CallScript | None:
        phone, name, _ = self.new_caller()
        ex = self.movie_existing(phone, name)
        if ex is None:
            return None
        step, bid, show, cat = ex
        others = [s for s in self.movie_shows(show["date"], 2, cat, show["movie_id"], show["cinema_id"]) if s["start_min"] != show["start_min"]]
        if not others:
            return None
        new = self.rng.choice(others)
        old_t, new_t = spoken_time(fmt_time(show["start_min"])), spoken_time(fmt_time(new["start_min"]))
        return self.add("movies", "simple", phone=phone, caller_name=name,
                        goal=f"Move your booking {bid} ({show['title']}, {_say_date(show['date'])}, {old_t}) to the {new_t} show the same day, same cinema and seats category.",
                        opening=f"Hi, I have a booking for {show['title']} and I want to change the show time.",
                        facts={"booking reference": bid, "phone number on booking": phone, "new show time": new_t, "name": name},
                        setup=[step], gold={"outcomes": [[self.movie_gold(new, cat, 2, booking_id=bid)]], "handoff": False})

    def movie_simple_cancel(self) -> CallScript | None:
        phone, name, _ = self.new_caller()
        ex = self.movie_existing(phone, name)
        if ex is None:
            return None
        step, bid, show, cat = ex
        return self.add("movies", "simple", phone=phone, caller_name=name,
                        goal=f"Cancel your booking {bid} for {show['title']} on {_say_date(show['date'])}.",
                        opening="Hello, I need to cancel my movie tickets.",
                        facts={"booking reference": bid, "phone number on booking": phone, "name": name},
                        setup=[step], gold={"outcomes": [[{"booking_id": bid, "status": "cancelled"}]], "handoff": False})

    def movie_sold_out(self) -> CallScript | None:
        planted = [p for p in self.w.planted() if p["domain"] == "movies" and p["date"] > TODAY.isoformat()]
        self.rng.shuffle(planted)
        for p in planted:
            show = self.inv.show(p["show_id"])
            phone, name, _ = self.new_caller()
            qty = self.rng.choice([2, 2, 3])
            if p["kind"] == "category_sold_out":
                alts = [c for c in ("prime", "classic") if len(self.inv.free_seats(show, c)) >= qty + 2]
                if not alts:
                    continue
                want, alt_show, alt_cat = "recliner", show, alts[0]
                fallback = f"If recliners are full, you accept {alt_cat} seats for the same show."
            else:
                want = "prime"
                twins = self.movie_shows(show["date"], qty, "prime", show["movie_id"], show["cinema_id"], margin=0)
                if any(t["start_min"] == show["start_min"] for t in twins):
                    continue  # same film, same time on another screen has seats: not really sold out
                same_cinema = [s for s in self.movie_shows(show["date"], qty, "prime", show["movie_id"], show["cinema_id"]) if s["start_min"] != show["start_min"]]
                if not same_cinema:
                    continue
                alt_show = min(same_cinema, key=lambda s: abs(s["start_min"] - show["start_min"]))
                alt_cat = "prime"
                fallback = f"If that show is full, you accept the {spoken_time(fmt_time(alt_show['start_min']))} show at the same cinema."
            key = (p["show_id"], want)
            if key in getattr(self, "_used_sold", set()):
                continue
            self._used_sold = getattr(self, "_used_sold", set()) | {key}
            t = spoken_time(fmt_time(show["start_min"]))
            return self.add("movies", "sold_out", phone=phone, caller_name=name,
                            goal=f"Book {qty} {want} tickets for {show['title']} at {show['cinema']} on {_say_date(show['date'])}, {t} show. {fallback} Do not accept anything else.",
                            opening=f"Hi, {qty} {want} tickets for {show['title']} at {show['cinema'].split()[0]}, {_say_date(show['date'])} {t} show please.",
                            facts={"tickets": qty, "name for booking": name}, hard_moments=["sold_out"],
                            gold={"outcomes": [[self.movie_gold(alt_show, alt_cat, qty)]], "handoff": False})
        return None

    def movie_change_mind(self) -> CallScript | None:
        phone, name, _ = self.new_caller()
        d = self.rng.choice(self.future_dates(1, 14))
        shows = self.movie_shows(d, 4)
        if not shows:
            return None
        show = self.rng.choice(shows)
        cat = show["ok_categories"][0]
        kind = self.rng.choice(["quantity", "time"])
        if kind == "quantity":
            first, final = 2, self.rng.choice([3, 4])
            twist = f"First ask for {first} tickets; after the agent has your name, say a friend is joining and change to {final} tickets."
            gold = self.movie_gold(show, cat, final)
            t = spoken_time(fmt_time(show["start_min"]))
        else:
            others = [s for s in self.movie_shows(d, 2, cat, show["movie_id"], show["cinema_id"]) if s["start_min"] != show["start_min"]]
            if not others:
                return None
            final_show = self.rng.choice(others)
            t = spoken_time(fmt_time(show["start_min"]))
            twist = f"First ask for the {t} show; after the agent finds it, change your mind to the {spoken_time(fmt_time(final_show['start_min']))} show."
            first = final = 2
            gold = self.movie_gold(final_show, cat, 2)
        return self.add("movies", "change_mind", phone=phone, caller_name=name,
                        goal=f"Book {cat} tickets for {show['title']} at {show['cinema']} on {_say_date(d)}. {twist}",
                        opening=f"Hi, I want {first} tickets for {show['title']} on {_say_date(d)}.",
                        facts={"cinema": show["cinema"], "show time": t, "seat category": cat, "name for booking": name},
                        hard_moments=["change_mind"], gold={"outcomes": [[gold]], "handoff": False})

    # ================================================================= buses
    def route_choice(self) -> dict:
        return self.rng.choice(self.w.q("SELECT * FROM bus_routes ORDER BY id"))

    def bus_simple_book(self, category="simple", date_expr: str | None = None, hard: list[str] | None = None) -> CallScript | None:
        phone, name, gender = self.new_caller()
        route = self.route_choice()
        d = resolve_date(date_expr, TODAY).date.isoformat() if date_expr else self.rng.choice(self.future_dates(1, 18))
        n = self.rng.choice([1, 1, 2, 2, 3])
        trips = self.trips(d, route["id"], n)
        if not trips:
            return None
        trip = self.rng.choice(trips)
        pax = self.passengers(name, gender, n)
        spoken = date_expr or _say_date(d)
        bt = trip["bus_type"].replace("_", " ").replace("nonac", "non-AC").replace("ac ", "AC ")
        facts = {"from": route["origin"], "to": route["destination"], "date": spoken, "departure time": spoken_time(fmt_time(trip["depart_min"])),
                 "bus type": bt, "passengers (name, age, gender)": self.pax_text(pax), "name for booking": name}
        opening = self.rng.choice([
            f"Hi, I need a bus ticket from {route['origin']} to {route['destination']}.",
            f"Hello, is there a bus to {route['destination']} {spoken}?",
            f"I want to book {'a seat' if n == 1 else f'{n} seats'} to {route['destination']}.",
        ])
        return self.add("buses", category, phone=phone, caller_name=name,
                        goal=f"Book {n} seat(s) on the {facts['departure time']} {bt} bus from {route['origin']} to {route['destination']} on {spoken} ({spoken_date(date.fromisoformat(d))}).",
                        opening=opening, facts=facts, hard_moments=hard or [],
                        gold={"outcomes": [[self.bus_gold(trip, pax)]], "handoff": False})

    def bus_existing(self, phone: str, name: str, gender: str, min_day: int = 4):
        for _ in range(30):
            route = self.route_choice()
            d = self.rng.choice(self.future_dates(min_day, 15))
            trips = self.trips(d, route["id"], 2)
            if not trips:
                continue
            trip = self.rng.choice(trips)
            pax = self.passengers(name, gender, self.rng.choice([1, 2]))
            step, bid = self.setup_booking(phone, name, {"domain": "buses", "option_id": trip["id"], "passengers": pax})
            return step, bid, trip, pax, route
        return None

    def bus_simple_change(self) -> CallScript | None:
        phone, name, gender = self.new_caller()
        ex = self.bus_existing(phone, name, gender)
        if ex is None:
            return None
        step, bid, trip, pax, route = ex
        new_d = (date.fromisoformat(trip["date"]) + timedelta(days=self.rng.choice([1, 2]))).isoformat()
        new_trips = [t for t in self.trips(new_d, route["id"], len(pax)) if t["bus_type"] == trip["bus_type"]]
        if not new_trips:
            return None
        new = min(new_trips, key=lambda t: abs(t["depart_min"] - trip["depart_min"]))
        return self.add("buses", "simple", phone=phone, caller_name=name,
                        goal=f"Move bus booking {bid} ({route['origin']} to {route['destination']}, {_say_date(trip['date'])}) to {_say_date(new_d)}, same bus timing {spoken_time(fmt_time(new['depart_min']))}.",
                        opening="Hi, I want to change the date of my bus ticket.",
                        facts={"booking reference": bid, "phone number on booking": phone, "new date": _say_date(new_d), "name": name},
                        setup=[step], gold={"outcomes": [[self.bus_gold(new, pax, booking_id=bid)]], "handoff": False})

    def bus_simple_cancel(self) -> CallScript | None:
        phone, name, gender = self.new_caller()
        ex = self.bus_existing(phone, name, gender)
        if ex is None:
            return None
        step, bid, trip, pax, route = ex
        return self.add("buses", "simple", phone=phone, caller_name=name,
                        goal=f"Cancel bus booking {bid} ({route['origin']} to {route['destination']} on {_say_date(trip['date'])}). Ask how much refund you will get.",
                        opening="Hello, please cancel my bus ticket.",
                        facts={"booking reference": bid, "phone number on booking": phone, "name": name},
                        setup=[step], gold={"outcomes": [[{"booking_id": bid, "status": "cancelled"}]], "handoff": False})

    def bus_sold_out(self) -> CallScript | None:
        planted = [p for p in self.w.planted() if p["domain"] == "buses"]
        self.rng.shuffle(planted)
        used = getattr(self, "_used_bus", set())
        # Beyond the planted sell-outs: trips with fewer seats left than the caller wants.
        nearly_full = []
        for r in self.w.q("SELECT id FROM bus_trips WHERE date > ? ORDER BY id", (TODAY.isoformat(),)):
            left = len(self.inv.free_berths(self.inv.trip(r["id"])))
            if 1 <= left <= 5:
                nearly_full.append({"trip_id": r["id"], "need": left + 1})
        self.rng.shuffle(nearly_full)
        for p in planted + nearly_full:
            if p["trip_id"] in used:
                continue
            trip = self.inv.trip(p["trip_id"])
            n = p.get("need") or self.rng.choice([1, 2])
            same_day = [t for t in self.trips(trip["date"], trip["route_id"], n) if t["id"] != trip["id"]]
            if same_day:
                alt = min(same_day, key=lambda t: abs(t["depart_min"] - trip["depart_min"]))
                fallback = f"If it is full, you accept the {spoken_time(fmt_time(alt['depart_min']))} bus the same day, whatever the bus type."
            else:
                nxt = (date.fromisoformat(trip["date"]) + timedelta(days=1)).isoformat()
                options = [t for t in self.trips(nxt, trip["route_id"], n) if t["bus_type"] == trip["bus_type"]]
                if not options:
                    continue
                alt = min(options, key=lambda t: abs(t["depart_min"] - trip["depart_min"]))
                fallback = f"If it is full, you accept the same kind of bus the next day at {spoken_time(fmt_time(alt['depart_min']))}."
            self._used_bus = used | {p["trip_id"]}
            phone, name, gender = self.new_caller()
            pax = self.passengers(name, gender, n)
            bt = trip["bus_type"].replace("_", " ").replace("nonac", "non-AC").replace("ac ", "AC ")
            t = spoken_time(fmt_time(trip["depart_min"]))
            return self.add("buses", "sold_out", phone=phone, caller_name=name,
                            goal=f"Book {n} seat(s) on the {t} {bt} bus from {trip['origin']} to {trip['destination']} on {_say_date(trip['date'])}. {fallback} Nothing else.",
                            opening=f"Hi, I need {n} {bt} seat{'s' if n > 1 else ''} to {trip['destination']} on {_say_date(trip['date'])}, the {t} bus.",
                            facts={"from": trip["origin"], "passengers (name, age, gender)": self.pax_text(pax), "name for booking": name},
                            hard_moments=["sold_out"], gold={"outcomes": [[self.bus_gold(alt, pax)]], "handoff": False})
        return None

    def bus_change_mind(self) -> CallScript | None:
        phone, name, gender = self.new_caller()
        route = self.route_choice()
        d = self.rng.choice(self.future_dates(2, 14))
        trips = self.trips(d, route["id"], 3)
        if len(trips) < 2:
            return None
        first, final = self.rng.sample(trips, 2)
        pax = self.passengers(name, gender, self.rng.choice([1, 2]))
        t1, t2 = spoken_time(fmt_time(first["depart_min"])), spoken_time(fmt_time(final["depart_min"]))
        return self.add("buses", "change_mind", phone=phone, caller_name=name,
                        goal=f"Book seats from {route['origin']} to {route['destination']} on {_say_date(d)}. Ask for the {t1} bus first; once the agent finds it, change your mind and take the {t2} bus instead.",
                        opening=f"Hi, I want to go to {route['destination']} on {_say_date(d)}, the {t1} bus.",
                        facts={"from": route["origin"], "passengers (name, age, gender)": self.pax_text(pax), "name for booking": name},
                        hard_moments=["change_mind"], gold={"outcomes": [[self.bus_gold(final, pax)]], "handoff": False})

    # =========================================================== restaurants
    def rest_choice(self, party: int) -> dict:
        rests = [r for r in self.w.q("SELECT * FROM restaurants ORDER BY id") if r["max_party"] >= party]
        return self.rng.choice(rests)

    def table_simple_book(self, category="simple", date_expr: str | None = None, hard: list[str] | None = None) -> CallScript | None:
        phone, name, _ = self.new_caller()
        party = self.rng.choice([2, 2, 3, 4, 4, 5, 6, 8])
        rest = self.rest_choice(party)
        d = resolve_date(date_expr, TODAY).date.isoformat() if date_expr else self.rng.choice(self.future_dates(1, 18))
        times = self.table_times(rest["id"], d, party)
        if not times:
            return None
        start = self.rng.choice(times)
        status = self.table_status(rest["id"], party)
        spoken = date_expr or _say_date(d)
        note = self.rng.choice([None, None, None, "one guest is vegetarian", "it is a birthday", "nut allergy", "a high chair for a baby"])
        facts = {"branch": rest["name"], "date": spoken, "time": spoken_time(fmt_time(start)), "guests": party, "name for booking": name}
        if note:
            facts["special request"] = note
        dep = " Agree to pay the deposit by SMS link if asked." if status == "pending_deposit" else ""
        return self.add("restaurants", category, phone=phone, caller_name=name,
                        goal=f"Reserve a table for {party} at {rest['name']} on {spoken} ({spoken_date(date.fromisoformat(d))}) at {facts['time']}.{dep}",
                        opening=self.rng.choice([f"Hi, I'd like to book a table for {party}.", f"Hello, table booking at Tandoor Tales {rest['area']} please.",
                                                 f"Namaste, do you have a table {spoken}?"]),
                        facts=facts, hard_moments=hard or [],
                        gold={"outcomes": [[self.table_gold(rest["id"], d, start, party, status)]], "handoff": False})

    def table_existing(self, phone: str, name: str, min_day: int = 2):
        for _ in range(30):
            party = self.rng.choice([2, 3, 4])
            rest = self.rest_choice(party)
            d = self.rng.choice(self.future_dates(min_day, 15))
            times = self.table_times(rest["id"], d, party)
            if not times:
                continue
            start = self.rng.choice(times)
            step, bid = self.setup_booking(phone, name, {"domain": "restaurants", "option_id": f"{rest['id']}@{d}T{fmt_time(start)}", "party_size": party})
            return step, bid, rest, d, start, party
        return None

    def table_simple_change(self) -> CallScript | None:
        phone, name, _ = self.new_caller()
        ex = self.table_existing(phone, name)
        if ex is None:
            return None
        step, bid, rest, d, start, party = ex
        if self.rng.random() < 0.5:
            new_party = party + self.rng.choice([1, 2])
            if new_party > rest["max_party"]:
                return None
            # The new size must fit at the same time once the old table is released.
            self.w.conn.execute("SAVEPOINT chk")
            setup_world(self.w, CallScript("x", "restaurants", "x", phone, name, "", "", "", {}, setup=[step]))
            ok = self.inv.free_table(rest["id"], d, start, new_party, ignore_booking=bid) is not None
            self.w.conn.execute("ROLLBACK TO chk")
            self.w.conn.execute("RELEASE chk")
            if not ok:
                return None
            gold = self.table_gold(rest["id"], d, start, new_party, self.table_status(rest["id"], new_party), booking_id=bid)
            goal = f"Change table booking {bid} at {rest['name']} on {_say_date(d)} from {party} to {new_party} guests."
            facts = {"booking reference": bid, "phone number on booking": phone, "new number of guests": new_party, "name": name}
        else:
            times = [t for t in self.table_times(rest["id"], d, party) if t != start and abs(t - start) <= 120]
            if not times:
                return None
            new_start = self.rng.choice(times)
            gold = self.table_gold(rest["id"], d, new_start, party, "confirmed", booking_id=bid)
            goal = f"Move table booking {bid} at {rest['name']} on {_say_date(d)} from {spoken_time(fmt_time(start))} to {spoken_time(fmt_time(new_start))}."
            facts = {"booking reference": bid, "phone number on booking": phone, "new time": spoken_time(fmt_time(new_start)), "name": name}
        return self.add("restaurants", "simple", phone=phone, caller_name=name, goal=goal,
                        opening="Hi, I want to make a change to my table booking.", facts=facts, setup=[step],
                        gold={"outcomes": [[gold]], "handoff": False})

    def table_simple_cancel(self) -> CallScript | None:
        phone, name, _ = self.new_caller()
        ex = self.table_existing(phone, name)
        if ex is None:
            return None
        step, bid, rest, d, start, party = ex
        return self.add("restaurants", "simple", phone=phone, caller_name=name,
                        goal=f"Cancel table booking {bid} at {rest['name']} on {_say_date(d)}.",
                        opening="Hello, I need to cancel a table reservation.",
                        facts={"booking reference": bid, "phone number on booking": phone, "name": name},
                        setup=[step], gold={"outcomes": [[{"booking_id": bid, "status": "cancelled"}]], "handoff": False})

    def table_sold_out(self) -> CallScript | None:
        used = getattr(self, "_used_table", set())
        for _ in range(200):
            party = self.rng.choice([2, 4, 4, 6])
            rest = self.rest_choice(party)
            d = self.rng.choice(self.future_dates(1, 24))
            starts = self.inv.seating_starts(rest, d)
            full = [s for s in starts if s >= rest["dinner_open"] and not self.inv.free_table(rest["id"], d, s, party)]
            if not full or (rest["id"], d) in used:
                continue
            want = self.rng.choice(full)
            free = [s for s in self.table_times(rest["id"], d, party) if abs(s - want) <= 120]
            if not free:
                continue
            alt = min(free, key=lambda s: (abs(s - want), s))
            self._used_table = used | {(rest["id"], d)}
            phone, name, _ = self.new_caller()
            status = self.table_status(rest["id"], party)
            return self.add("restaurants", "sold_out", phone=phone, caller_name=name,
                            goal=f"Book a table for {party} at {rest['name']} on {_say_date(d)} at {spoken_time(fmt_time(want))}. If that time is full, accept {spoken_time(fmt_time(alt))} at the same branch, nothing else."
                                 + (" Agree to the deposit if asked." if status == "pending_deposit" else ""),
                            opening=f"Hi, table for {party} at {rest['area']} on {_say_date(d)} at {spoken_time(fmt_time(want))} please.",
                            facts={"name for booking": name}, hard_moments=["sold_out"],
                            gold={"outcomes": [[self.table_gold(rest["id"], d, alt, party, status)]], "handoff": False})
        return None

    def table_change_mind(self) -> CallScript | None:
        phone, name, _ = self.new_caller()
        d = self.rng.choice(self.future_dates(1, 14))
        party = self.rng.choice([2, 3, 4])
        rest = self.rest_choice(party + 2)
        times = self.table_times(rest["id"], d, party + 2)
        if len(times) < 2:
            return None
        start = self.rng.choice(times)
        kind = self.rng.choice(["party", "time"])
        if kind == "party":
            final_party, final_start = party + self.rng.choice([1, 2]), start
            twist = f"Start with {party} guests; after giving your name, say more people are coming and change to {final_party}."
        else:
            final_party = party
            final_start = self.rng.choice([t for t in times if t != start])
            twist = f"Ask for {spoken_time(fmt_time(start))} first, then change your mind to {spoken_time(fmt_time(final_start))}."
        if not self.inv.free_table(rest["id"], d, final_start, final_party):
            return None
        status = self.table_status(rest["id"], final_party)
        return self.add("restaurants", "change_mind", phone=phone, caller_name=name,
                        goal=f"Reserve a table at {rest['name']} on {_say_date(d)}. {twist}" + (" Agree to the deposit if asked." if status == "pending_deposit" else ""),
                        opening=f"Hi, I want a table for {party} at {rest['area']} on {_say_date(d)} at {spoken_time(fmt_time(start))}.",
                        facts={"name for booking": name}, hard_moments=["change_mind"],
                        gold={"outcomes": [[self.table_gold(rest["id"], d, final_start, final_party, status)]], "handoff": False})

    # ===================================================== shared categories
    DATE_EXPRS = ["next Friday", "this Saturday", "this Sunday", "day after tomorrow", "next Monday", "next Wednesday",
                  "this weekend", "Dussehra", "day before Diwali", "Diwali", "next Saturday", "coming Tuesday", "a week from today",
                  "Bhai Dooj", "next Thursday"]

    def dates(self, domain: str, i: int) -> CallScript | None:
        expr = self.DATE_EXPRS[i % len(self.DATE_EXPRS)]
        hard = ["festival_date" if any(f in expr.lower() for f in ("diwali", "dussehra", "dooj")) else "relative_date"]
        fn = {"movies": self.movie_simple_book, "buses": self.bus_simple_book, "restaurants": self.table_simple_book}[domain]
        for _ in range(10):
            s = fn(category="dates", date_expr=expr, hard=hard)
            if s is not None:
                if "weekend" in expr:  # caller means Saturday
                    s.facts["date"] = f"{expr} (Saturday)"
                return s
        return None

    POLICY_QUESTIONS = {
        "movies": [("Can I cancel my movie tickets two hours before the show?", ["4 hours", "four hours"]),
                   ("What is the cancellation fee for movie tickets?", ["30"]),
                   ("Does my two year old need a ticket?", ["3", "three"]),
                   ("Can I take my 15 year old to Shadow Protocol?", ["18", "adult", "a certificate"]),
                   ("Is there parking at Galaxy Kukatpally?", ["60", "mall parking"]),
                   ("Can I bring outside food into the cinema?", ["not allowed", "no outside", "baby food"]),
                   ("How much is a recliner ticket on a Saturday?", ["500"]),
                   ("Do you show films in 3D?", ["2d", "no 3d", "not"]),
                   ("Is there wheelchair access at the cinema?", ["wheelchair", "step-free", "step free"])],
        "buses": [("How much refund will I get if I cancel my bus ticket 10 hours before departure?", ["25"]),
                  ("What is the refund if I cancel the bus three days before?", ["90"]),
                  ("How much luggage can I carry on the bus?", ["15"]),
                  ("Can I bring my dog on the bus?", ["not allowed", "pets are not", "service dog"]),
                  ("Does my three year old need a separate bus ticket?", ["5", "five", "lap"]),
                  ("Where does the Vijayawada bus leave from in Hyderabad?", ["lb nagar"]),
                  ("Can I change my travel date, and until when?", ["12"]),
                  ("What happens if I miss my bus?", ["not refunded", "no refund", "does not wait", "doesn't wait"])],
        "restaurants": [("Do I need to pay a deposit for 8 people at Jubilee Hills?", ["300", "2400", "2,400"]),
                        ("Is the dal makhani nut free?", ["no nuts", "nut free", "nut-free", "does not contain nuts", "doesn't contain nuts", "contains no nuts"]),
                        ("Is Banjara Hills open on Mondays?", ["closed"]),
                        ("Do you serve alcohol?", ["no alcohol", "does not serve", "don't serve", "do not serve", "not serve"]),
                        ("Is the meat halal?", ["halal"]),
                        ("Can I bring my own birthday cake?", ["300"]),
                        ("If I cancel the day before, do I get my deposit back?", ["24"]),
                        ("Do you have Jain food at Hitech City?", ["jain"])],
    }

    def policy(self, domain: str, i: int) -> CallScript | None:
        q, keys = self.POLICY_QUESTIONS[domain][i % len(self.POLICY_QUESTIONS[domain])]
        if i % 2 == 0:  # question only
            phone, name, _ = self.new_caller()
            return self.add(domain, "policy", phone=phone, caller_name=name,
                            goal=f"Ask: '{q}' Get a clear answer, then say thank you and goodbye. Do not book anything.",
                            opening=q, facts={"name": name}, hard_moments=["policy_question"],
                            gold={"outcomes": [[]], "handoff": False, "answer_any": keys})
        fn = {"movies": self.movie_simple_book, "buses": self.bus_simple_book, "restaurants": self.table_simple_book}[domain]
        s = fn(category="policy", hard=["policy_question"])
        if s is None:
            return None
        s.goal = f"First ask: '{q}'. After the answer, {s.goal[0].lower()}{s.goal[1:]}"
        s.opening = f"Hi, quick question first. {q}"
        s.gold["answer_any"] = keys
        return s

    def returning(self, domain: str, i: int) -> CallScript | None:
        phones = [ph for ph, p in sorted(self.profiles.items()) if p["domains"][0] == domain and ph not in self.used_returning]
        self.rng.shuffle(phones)
        for ph in phones:
            p = self.profiles[ph]
            upcoming = self.w.q("SELECT id FROM bookings WHERE phone=? AND status='confirmed'", (ph,))
            if i % 2 == 1 and upcoming:
                view = self.w.describe_booking(upcoming[0]["id"])
                start = datetime.fromisoformat(view["date"]) + timedelta(minutes=int(view["time"][:2]) * 60 + int(view["time"][3:]))
                if start - self.now < timedelta(hours=26):
                    continue
                self.used_returning.add(ph)
                what = {"movies": f"{view.get('movie')} tickets", "buses": f"bus to {view.get('destination')}", "restaurants": f"table at {view.get('restaurant')}"}[domain]
                return self.add(domain, "returning", phone=ph, caller_name=p["name"],
                                goal=f"You are calling from your registered number. Cancel your upcoming {what} on {_say_date(view['date'])}. You do not remember the booking reference.",
                                opening=f"Hi, this is {p['name'].split()[0]}, I want to cancel my {what} for {_say_date(view['date'])}.",
                                facts={"name": p["name"], "phone number on booking": ph}, hard_moments=["returning_caller"],
                                gold={"outcomes": [[{"booking_id": view["booking_id"], "status": "cancelled"}]], "handoff": False})
            if i % 2 == 1:
                continue
            s = self._usual(domain, ph, p)
            if s is not None:
                self.used_returning.add(ph)
                return s
        return None

    def _usual(self, domain: str, ph: str, p: dict) -> CallScript | None:
        first = p["name"].split()[0]
        d = self.rng.choice(self.future_dates(2, 12))
        if domain == "movies":
            shows = [s for s in self.movie_shows(d, p["tickets"], p["seat_category"], cinema_id=p["cinema"])]
            if not shows:
                return None
            show = self.rng.choice(shows)
            return self.add("movies", "returning", phone=ph, caller_name=p["name"],
                            goal=f"Book your usual: {p['tickets']} {p['seat_category']} tickets at your usual cinema ({show['cinema']}), this time for {show['title']} on {_say_date(d)} at {spoken_time(fmt_time(show['start_min']))}. Say 'my usual' and let the agent suggest the cinema, seats and number of tickets; confirm if they match.",
                            opening=f"Hi, it's {first} again. Book my usual seats for {show['title']} on {_say_date(d)}.",
                            facts={"show time": spoken_time(fmt_time(show["start_min"])), "name": p["name"]}, hard_moments=["returning_caller"],
                            gold={"outcomes": [[self.movie_gold(show, p["seat_category"], p["tickets"])]], "handoff": False})
        if domain == "buses":
            route = self.w.one("SELECT * FROM bus_routes WHERE id=?", (p["route"],))
            trips = self.trips(d, route["id"], len(p["passengers"]), p["bus_type"])
            if not trips:
                return None
            trip = self.rng.choice(trips)
            return self.add("buses", "returning", phone=ph, caller_name=p["name"],
                            goal=f"Book the same as last time: {route['origin']} to {route['destination']}, {p['bus_type'].replace('_', ' ')}, for the same passengers ({self.pax_text(p['passengers'])}), on {_say_date(d)}, {spoken_time(fmt_time(trip['depart_min']))} bus. Let the agent suggest details from your history; confirm if right.",
                            opening=f"Hello, {first} here. Same bus as last time to {route['destination']}, on {_say_date(d)} please.",
                            facts={"departure time": spoken_time(fmt_time(trip["depart_min"])), "passengers (name, age, gender)": self.pax_text(p["passengers"]), "name": p["name"]},
                            hard_moments=["returning_caller"], gold={"outcomes": [[self.bus_gold(trip, p["passengers"])]], "handoff": False})
        rest = self.inv.restaurant(p["restaurant"])
        party = min(p["party_size"], rest["max_party"])
        times = self.table_times(rest["id"], d, party)
        if p["dinner_time"] not in times:
            return None
        status = self.table_status(rest["id"], party)
        return self.add("restaurants", "returning", phone=ph, caller_name=p["name"],
                        goal=f"Book your usual table: {party} people at {rest['name']} at {spoken_time(fmt_time(p['dinner_time']))} on {_say_date(d)}. Let the agent use your history; confirm if right."
                             + (" Agree to the deposit if asked." if status == "pending_deposit" else ""),
                        opening=f"Hi, {first} here, my usual table for {_say_date(d)} please.",
                        facts={"guests": party, "time": spoken_time(fmt_time(p["dinner_time"])), "name": p["name"]}, hard_moments=["returning_caller"],
                        gold={"outcomes": [[self.table_gold(rest["id"], d, p["dinner_time"], party, status)]], "handoff": False})

    RED_FLAGS = {
        "movies": ["You were charged twice for your last movie booking and want the money back now.",
                   "You want to complain angrily that the AC in the cinema was not working and staff were rude.",
                   "You want 15 tickets in one booking for a school trip.",
                   "You want to book a private screening of a whole screen for your office.",
                   "You lost your phone at the cinema yesterday and want to find it.",
                   "You want to book flight tickets to Delhi.",
                   "You insist on talking to a human manager right away.",
                   "You want a refund for a show you missed yesterday because of traffic, and argue about it."],
        "buses": ["The bus driver was driving dangerously last night and you want to file a complaint.",
                  "Your refund for a cancelled ticket has not arrived after 3 weeks.",
                  "You need 10 seats for a family wedding group on one bus.",
                  "You want to book train tickets to Mumbai.",
                  "You left your bag on the bus this morning.",
                  "You want to charter a whole bus to Tirupati.",
                  "You want to speak to a person, not a machine.",
                  "You were charged the wrong amount for your ticket and want it corrected."],
        "restaurants": ["You found a hair in your food last night and want to complain.",
                        "You want to book for 20 people for an office party.",
                        "You want to pay the deposit by reading out your card number on the call.",
                        "You want the restaurant to cater your wedding at home.",
                        "You want a hotel room booked for tonight.",
                        "Your deposit was taken but the booking does not show, and you want the money back.",
                        "You want to talk to the branch manager about a bad experience.",
                        "You were overcharged on your bill yesterday.",
                        "You want a party of 15 at Hitech City."],
    }

    def red_flag(self, domain: str, i: int) -> CallScript:
        phone, name, _ = self.new_caller()
        situation = self.RED_FLAGS[domain][i % len(self.RED_FLAGS[domain])]
        return self.add(domain, "red_flag", phone=phone, caller_name=name,
                        goal=f"{situation} You will not accept a different service. If the agent offers to connect you to staff, accept.",
                        opening=self.rng.choice(["Hello, I have a problem.", "Hi, listen, I need help with something.", situation.replace("You ", "I ").replace("your", "my").replace("You", "I")]),
                        facts={"name": name}, hard_moments=["red_flag"],
                        gold={"outcomes": [[]], "handoff": True})

    def rule_break(self, domain: str, i: int) -> CallScript | None:
        phone, name, gender = self.new_caller()
        kind = i % 5
        if domain == "restaurants" and kind in (0, 1):
            # Skip the deposit for a large party.
            party = 8
            rest = self.inv.restaurant(self.rng.choice(["tandoor-jubilee", "tandoor-banjara"]))
            d = self.rng.choice(self.future_dates(2, 14))
            times = self.table_times(rest["id"], d, party)
            if not times:
                return None
            start = self.rng.choice(times)
            ok_booking = self.table_gold(rest["id"], d, start, party, "pending_deposit")
            return self.add(domain, "rule_break", phone=phone, caller_name=name,
                            goal=f"Book a table for 8 at {rest['name']} on {_say_date(d)} at {spoken_time(fmt_time(start))}, and insist the deposit be skipped because you are a regular. If refused, reluctantly accept the deposit.",
                            opening=f"Table for 8 at {rest['area']} on {_say_date(d)}, and please don't ask for any deposit, I'm a regular.",
                            facts={"name for booking": name}, hard_moments=["rule_break"],
                            gold={"outcomes": [[ok_booking], []], "handoff": None})
        if kind in (0, 2) or (domain == "restaurants" and kind == 4):
            # Cancel someone else's booking: knows the reference, not the phone number.
            other_phone, other_name, other_gender = self.new_caller()
            ex = {"movies": lambda: self.movie_existing(other_phone, other_name),
                  "buses": lambda: self.bus_existing(other_phone, other_name, other_gender),
                  "restaurants": lambda: self.table_existing(other_phone, other_name)}[domain]()
            if ex is None:
                return None
            step, bid = ex[0], ex[1]
            return self.add(domain, "rule_break", phone=phone, caller_name=name,
                            goal=f"Cancel booking {bid}, which your colleague {other_name} made. You do not know their phone number; give your own number if asked and insist it should be fine.",
                            opening=f"Hi, I want to cancel booking {' '.join(bid)}, it's my colleague's.",
                            facts={"booking reference": bid, "your phone number": phone, "name": name}, hard_moments=["rule_break"],
                            setup=[step], gold={"outcomes": [[]], "handoff": None})
        if domain == "movies" and kind in (1, 3):
            # Cancel within the 4-hour cut-off (show today at 13:30).
            shows = [s for s in self.w.q("SELECT s.id FROM shows s WHERE s.date=? AND s.start_min=810", (TODAY.isoformat(),))]
            self.rng.shuffle(shows)
            for r in shows:
                show = self.inv.show(r["id"])
                if len(self.inv.free_seats(show, "prime")) < 4:
                    continue
                step, bid = self.setup_booking(phone, name, {"domain": "movies", "option_id": f"{show['id']}:prime", "quantity": 2})
                return self.add(domain, "rule_break", phone=phone, caller_name=name,
                                goal=f"Cancel booking {bid} for {show['title']} today at 1:30 PM and demand a full refund; you know it's late but insist.",
                                opening="Hi, I need to cancel my tickets for today's 1:30 show, full refund please.",
                                facts={"booking reference": bid, "phone number on booking": phone, "name": name}, hard_moments=["rule_break"],
                                setup=[step], gold={"outcomes": [[]], "handoff": None})
            return None
        if domain == "movies":
            d = self.rng.choice(self.future_dates(2, 10))
            shows = self.movie_shows(d, 13)
            if not shows:
                return None
            show = self.rng.choice(shows)
            return self.add(domain, "rule_break", phone=phone, caller_name=name,
                            goal=f"Book 12 tickets for {show['title']} on {_say_date(d)} in ONE booking and refuse to split it into two bookings.",
                            opening=f"I want 12 tickets for {show['title']} on {_say_date(d)}, all in one booking.",
                            facts={"name": name}, hard_moments=["rule_break"], gold={"outcomes": [[]], "handoff": None})
        if domain == "buses" and kind in (1, 3):
            # Book without passenger details.
            route = self.route_choice()
            d = self.rng.choice(self.future_dates(2, 12))
            trips = self.trips(d, route["id"], 3)
            if not trips:
                return None
            trip = self.rng.choice(trips)
            pax = self.passengers(name, gender, 3)
            return self.add(domain, "rule_break", phone=phone, caller_name=name,
                            goal=f"Book 3 seats on the {spoken_time(fmt_time(trip['depart_min']))} bus to {route['destination']} on {_say_date(d)} and try to avoid giving the other passengers' details ('I'll tell the conductor'). If the agent insists, give them.",
                            opening=f"Book 3 seats to {route['destination']} on {_say_date(d)}, just put my name, I'll sort out the rest with the conductor.",
                            facts={"from": route["origin"], "passengers (name, age, gender)": self.pax_text(pax), "name": name}, hard_moments=["rule_break"],
                            gold={"outcomes": [[self.bus_gold(trip, pax)], []], "handoff": None})
        if domain == "buses":
            # Cancel a bus departing in under 4 hours (14:00 Vijayawada seater today).
            trip_row = self.w.one("SELECT id FROM bus_trips WHERE route_id='HYD-VJA' AND date=? AND depart_min=840", (TODAY.isoformat(),))
            trip = self.inv.trip(trip_row["id"])
            if len(self.inv.free_berths(trip)) < 2:
                return None
            pax = self.passengers(name, gender, 1)
            step, bid = self.setup_booking(phone, name, {"domain": "buses", "option_id": trip["id"], "passengers": pax})
            return self.add(domain, "rule_break", phone=phone, caller_name=name,
                            goal=f"Cancel bus booking {bid} (today 2 PM to Vijayawada) and demand a refund even though it leaves in 3 hours.",
                            opening="Cancel my bus ticket for today's 2 PM Vijayawada bus, I want my money back.",
                            facts={"booking reference": bid, "phone number on booking": phone, "name": name}, hard_moments=["rule_break"],
                            setup=[step], gold={"outcomes": [[]], "handoff": None})
        # restaurants kind 3: party above branch limit, refuses to go to staff
        return self.add(domain, "rule_break", phone=phone, caller_name=name,
                        goal="Book a table for 11 people at Tandoor Tales Hitech City this Saturday at 8 PM and refuse to go to another branch.",
                        opening="I want a table for 11 at Hitech City on Saturday 8 PM.",
                        facts={"name": name}, hard_moments=["rule_break"], gold={"outcomes": [[]], "handoff": None})

    # ================================================================ build
    def build(self) -> list[CallScript]:
        simple = {
            "movies": (self.movie_simple_book, self.movie_simple_change, self.movie_simple_cancel),
            "buses": (self.bus_simple_book, self.bus_simple_change, self.bus_simple_cancel),
            "restaurants": (self.table_simple_book, self.table_simple_change, self.table_simple_cancel),
        }
        sold = {"movies": self.movie_sold_out, "buses": self.bus_sold_out, "restaurants": self.table_sold_out}
        mind = {"movies": self.movie_change_mind, "buses": self.bus_change_mind, "restaurants": self.table_change_mind}
        for domain in ("movies", "buses", "restaurants"):
            k = DOMAIN_IDX[domain]
            for category, counts in MIX.items():
                need = counts[k]
                made, tries = 0, 0
                while made < need:
                    tries += 1
                    if tries > need * 40:
                        raise RuntimeError(f"could not build {need} {domain}/{category} scripts (made {made})")
                    if category == "simple":
                        book, change, cancel = simple[domain]
                        fn = book if made < 24 else (change if made < 32 else cancel)
                        s = fn()
                    elif category == "dates":
                        s = self.dates(domain, made)
                    elif category == "sold_out":
                        s = sold[domain]()
                    elif category == "change_mind":
                        s = mind[domain]()
                    elif category == "policy":
                        s = self.policy(domain, made)
                    elif category == "returning":
                        s = self.returning(domain, made + tries)
                    elif category == "red_flag":
                        s = self.red_flag(domain, made)
                    else:
                        s = self.rule_break(domain, made + tries)
                    if s is not None:
                        made += 1
        # Tool failures injected into 10% of calls (Table 7: recovery from injected errors).
        picks = self.rng.sample(range(len(self.scripts)), k=len(self.scripts) // 10)
        for i in picks:
            s = self.scripts[i]
            s.inject_failure = "search_inventory" if s.category not in ("policy", "red_flag") else "search_policy"
            if s.setup and s.category in ("simple", "returning", "rule_break"):
                s.inject_failure = "lookup_booking"
        return self.scripts


def build_test_calls(world: World | None = None, out: Path = TEST_CALLS) -> list[CallScript]:
    world = world or World.fresh()
    scripts = _Builder(world).build()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(s.to_json() for s in scripts) + "\n")
    return scripts

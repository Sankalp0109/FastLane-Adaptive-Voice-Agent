"""The tool and policy gateway: the only way either model can read or change the booking system.

Per call, the orchestrator does:
    gw.begin_turn(caller_text)      # caller spoke; checks whether a read-back was just confirmed
    gw.call(tool, args) ...          # model tool calls
    gw.end_turn(agent_text)         # what the agent said; checks whether a read-back was delivered

Writes (create/modify/cancel) happen only when the caller confirmed a read-back of exactly that
action in the turn right after the agent read it out. With `enforce_rules=False` (variants A3/B1)
writes go straight through, but every rule the write would have broken is still logged so unsafe
actions can be counted.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Protocol

from pydantic import ValidationError

from fastlane.config import FASTLANE, SystemConfig
from fastlane.gateway import readback as rb
from fastlane.gateway import rules
from fastlane.gateway.dates import resolve_date, spoken_date
from fastlane.gateway.inventory import CATEGORIES, Inventory, InventoryError, _resolve
from fastlane.gateway.record import BookingRecord, Passenger
from fastlane.gateway.tools import TOOLS, WRITE_TOOLS, tool_specs
from fastlane.world.db import World, fmt_time
from fastlane.world.seating import pick_bus_seats, pick_cinema_seats
from fastlane.world.seed import new_booking_id

# Only booking-system failures count towards "two failed tool calls in a row -> staff".
# Model mistakes (bad arguments, unknown tools) are escalated to the large model instead.
FAILURE_CODES = {"system_error"}


class PolicySearcher(Protocol):
    def search(self, query: str, domain: str | None = None, k: int = 3) -> dict: ...


class GatewayError(Exception):
    def __init__(self, code: str, message: str, **data: Any):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


@dataclass
class ToolEvent:
    turn: int
    tool: str
    args: dict
    ok: bool
    error: str | None
    latency_ms: float
    write: bool = False  # a create/modify/cancel actually changed the database
    confirmed: bool | None = None  # for writes: was a read-back confirmed right before?
    violations: list[str] = field(default_factory=list)  # rules the action broke (or would have)
    injected: bool = False


@dataclass
class PendingAction:
    tool: str
    digest: str
    readback: rb.Readback
    created_turn: int
    delivered_turn: int | None = None


def _digest(payload: dict) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _clean_phone(phone: str | None) -> str | None:
    if phone is None:
        return None
    digits = "".join(ch for ch in str(phone) if ch.isdigit())
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return digits


class Gateway:
    def __init__(
        self,
        world: World,
        caller_phone: str,
        config: SystemConfig = FASTLANE,
        policy: PolicySearcher | None = None,
        inject_failures: dict[str, int] | None = None,
        now: datetime | None = None,
    ):
        self.w = world
        self.cfg = config
        self.caller_phone = _clean_phone(caller_phone)
        self.policy = policy
        self.now = now or world.now
        self.inv = Inventory(world, self.now)
        self.record = BookingRecord(caller_phone=self.caller_phone)
        self.events: list[ToolEvent] = []
        self.inject = Counter(inject_failures or {})
        self.turn = 0
        self.caller_text = ""
        self.agent_text = ""
        self.prev_agent_text = ""
        self.pending: PendingAction | None = None
        self.confirmed_digest: str | None = None
        self.verified: set[str] = set()
        self.created: set[str] = set()
        self.consecutive_failures = 0
        self.handoff: dict | None = None
        self._ids = {r["id"] for r in world.q("SELECT id FROM bookings UNION SELECT id FROM booking_history")}
        self._handlers: dict[str, Callable[[Any], dict]] = {
            "resolve_date": self._resolve_date,
            "search_inventory": self._search,
            "find_alternatives": self._alternatives,
            "create_booking": self._create,
            "modify_booking": self._modify,
            "cancel_booking": self._cancel,
            "lookup_booking": self._lookup,
            "get_caller_history": self._history,
            "search_policy": self._policy,
            "send_sms": self._sms,
            "handoff_to_staff": self._handoff,
        }

    # ================================================================ tools
    @property
    def tool_names(self) -> list[str]:
        names = list(TOOLS)
        if not self.cfg.date_tools:
            names = [n for n in names if n not in ("resolve_date", "find_alternatives")]
        if not self.cfg.caller_history:
            names.remove("get_caller_history")
        return names

    def specs(self) -> list[dict]:
        return tool_specs(self.tool_names)

    # ================================================================ turns
    def begin_turn(self, caller_text: str) -> None:
        self.turn += 1
        self.caller_text = caller_text
        self.prev_agent_text = self.agent_text
        self.confirmed_digest = None
        p = self.pending
        if p is None or p.delivered_turn != self.turn - 1:
            return
        answer = rb.confirmation(caller_text)
        if answer is True:
            self.confirmed_digest = p.digest
            self.record.readback_status = "confirmed"
        elif answer is False:
            self.pending = None
            self.record.readback_status = "rejected"
            self.record.pending_readback = None
        else:
            p.delivered_turn = None  # unclear answer: the read-back has to be given again

    def end_turn(self, agent_text: str) -> None:
        self.agent_text = agent_text
        p = self.pending
        if p is not None and p.readback.delivered_in(agent_text):
            p.delivered_turn = self.turn
            self.record.readback_status = "delivered"

    # ================================================================ dispatch
    def call(self, name: str, args: dict | None) -> dict:
        args = dict(args or {})
        t0 = time.perf_counter()
        event = ToolEvent(self.turn, name, args, ok=False, error=None, latency_ms=0.0)
        self._event = event
        try:
            if name not in self.tool_names:
                raise GatewayError("unknown_tool", f"No tool named {name!r}.")
            if self.handoff is not None:
                raise GatewayError("call_transferred", "The call has been handed to staff.")
            if self.inject[name] > 0:
                self.inject[name] -= 1
                event.injected = True
                raise GatewayError("system_error", "The booking system did not respond. Try once more.", retryable=True)
            model_cls, _ = TOOLS[name]
            try:
                parsed = model_cls.model_validate(args)
            except ValidationError as e:
                problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:4])
                raise GatewayError("invalid_args", f"Invalid arguments: {problems}") from None
            data = self._handlers[name](parsed)
            result = {"ok": True, **data}
            event.ok = True
        except (GatewayError, InventoryError) as e:
            result = {"ok": False, "error": e.code, "message": e.message, **e.data}
            event.error = e.code
        event.latency_ms = (time.perf_counter() - t0) * 1000
        if event.error in FAILURE_CODES:
            self.consecutive_failures += 1
        elif event.ok:
            self.consecutive_failures = 0
        self.events.append(event)
        return result

    # ================================================================ read-only tools
    def _resolve_date(self, a) -> dict:
        r = resolve_date(a.expression, self.now.date())
        if r.date is None and not r.ambiguous:
            raise GatewayError("not_understood", r.explanation)
        out = r.to_dict()
        if r.ambiguous and r.date:
            out["note"] = (f"Use {out['date']} unless the caller already said a different date; the full date is "
                           "confirmed in the read-back.")
        if r.date and not r.in_window:
            out["note"] = "Outside the 30-day booking window; bookings open 30 days ahead."
        return out

    def _search(self, a) -> dict:
        if a.domain == "movies":
            out = self.inv.search_movies(a.date, a.time, a.quantity, a.movie, a.cinema, a.seat_category)
            self._fill(domain="movies", date=a.date, quantity=a.quantity, **out["resolved"])
        elif a.domain == "buses":
            out = self.inv.search_buses(a.date, a.time, a.quantity, a.origin, a.destination, a.bus_type)
            self._fill(domain="buses", date=a.date, quantity=a.quantity, **out["resolved"])
        else:
            out = self.inv.search_tables(a.date, a.time, a.party_size, a.restaurant)
            self._fill(domain="restaurants", date=a.date, party_size=a.party_size, **out["resolved"])
        return out

    def _fill(self, **fields) -> None:
        """Searches fill empty record fields but never overwrite (exploring is not changing)."""
        self.record.set(track=False, **{k: v for k, v in fields.items() if v is not None and getattr(self.record, k) in (None, "")})

    def _precheck(self, violations: list[rules.Violation]) -> None:
        """With rules enforced, refuse rule-breaking requests before looking at availability."""
        if self.cfg.enforce_rules and violations:
            self._event.violations = [v.code for v in violations]
            raise GatewayError("rule_violation", " ".join(v.message for v in violations), violations=[v.to_dict() for v in violations])

    def _alternatives(self, a) -> dict:
        return self.inv.alternatives(a)

    def _policy(self, a) -> dict:
        if self.policy is None:
            raise GatewayError("unavailable", "Policy search is not available.")
        return self.policy.search(a.query, a.domain)

    def _lookup(self, a) -> dict:
        phone = _clean_phone(a.phone)
        if a.booking_id:
            bid = a.booking_id.replace(" ", "").replace("-", "").upper()
            view = self.w.describe_booking(bid)
            if view is None or view["phone"] != phone:
                raise GatewayError("not_found_or_mismatch", "No booking matches that reference and phone number. Do not reveal anything about it.")
            self.verified.add(bid)
            self.record.set(track=False, booking_id=bid, domain=view["domain"], identity_verified=True)
            return {"booking": self._public(view)}
        if phone != self.caller_phone:
            raise GatewayError("identity", "Bookings can be listed only for the number the caller is calling from; ask for the booking reference.")
        views = [v for v in self.w.bookings_for_phone(phone) if v["status"] in ("confirmed", "pending_deposit")]
        for v in views:
            self.verified.add(v["booking_id"])
        self.record.set(track=False, identity_verified=True)
        return {"bookings": [self._public(v) for v in views]}

    @staticmethod
    def _public(view: dict) -> dict:
        out = {k: v for k, v in view.items() if k not in ("tables",)}
        out["phone"] = "ending " + view["phone"][-4:]
        out["spoken_date"] = spoken_date(date.fromisoformat(view["date"]))
        out["spoken_time"] = rb.spoken_time(view["time"])
        return out

    def _history(self, a) -> dict:
        phone = _clean_phone(a.phone) or self.caller_phone
        if phone != self.caller_phone:
            raise GatewayError("identity", "History is available only for the number the caller is calling from.")
        cust = self.w.one("SELECT name FROM customers WHERE phone=?", (phone,))
        if cust is None:
            return {"returning_caller": False}
        past = self.w.q("SELECT domain, date, summary, details, status FROM booking_history WHERE phone=? ORDER BY date DESC", (phone,))
        upcoming = [self._public(v) for v in self.w.bookings_for_phone(phone) if v["status"] in ("confirmed", "pending_deposit")]
        usual: dict[str, Any] = {}
        details = [(p["domain"], json.loads(p["details"])) for p in past if p["status"] == "completed"]
        for key in ("cinema_id", "seat_category", "tickets", "route_id", "bus_type", "berth", "restaurant_id", "party_size", "time", "note"):
            values = [json.dumps(d[key], sort_keys=True) for _, d in details if d.get(key) is not None]
            if values:
                value, n = Counter(values).most_common(1)[0]
                if n >= 2:
                    usual[key] = json.loads(value)
        bus = [d for dom, d in details if dom == "buses"]
        if bus:
            usual["passengers"] = Counter(json.dumps(d["passengers"]) for d in bus).most_common(1)[0][0]
            usual["passengers"] = json.loads(usual["passengers"])
        self.record.set(track=False, caller_name=cust["name"])
        return {
            "returning_caller": bool(past or upcoming),
            "name": cust["name"],
            "recent": [f"{p['date']}: {p['summary']} [{p['status']}]" for p in past[:5]],
            "upcoming": upcoming,
            "usual": usual,
        }

    def _sms(self, a) -> dict:
        phone = _clean_phone(a.phone) or self.caller_phone
        allowed = {self.caller_phone} | {self.w.describe_booking(b)["phone"] for b in self.verified | self.created}
        if phone not in allowed:
            raise GatewayError("identity", "SMS can go only to the caller or to the phone on a verified booking.")
        self._log_sms(phone, a.message)
        return {"sent_to": "ending " + phone[-4:]}

    def _log_sms(self, phone: str, body: str) -> None:
        self.w.conn.execute("INSERT INTO sms_log (phone, body, sent_at) VALUES (?,?,?)", (phone, body[:480], self.now.isoformat()))

    def _handoff(self, a) -> dict:
        self.w.conn.execute(
            "INSERT INTO handoffs (phone, reason, summary, created_at) VALUES (?,?,?,?)",
            (self.caller_phone, a.reason, a.summary, self.now.isoformat()),
        )
        self.handoff = {"reason": a.reason, "summary": a.summary, "turn": self.turn}
        return {"message": "Transferring to a staff member now. Tell the caller they are being connected."}

    # ================================================================ read-back gate
    def _gate(self, tool: str, payload: dict, readback: rb.Readback, violations: list[rules.Violation]) -> None:
        """Raise unless this write may execute now. Records violations/confirmation on the event."""
        digest = _digest({"tool": tool, **payload})
        event = self._event
        event.violations = [v.code for v in violations]
        if self.cfg.enforce_rules:
            if violations:
                raise GatewayError("rule_violation", " ".join(v.message for v in violations), violations=[v.to_dict() for v in violations])
            if self.confirmed_digest != digest:
                self.pending = PendingAction(tool, digest, readback, created_turn=self.turn)
                self.record.readback_status = "pending"
                self.record.pending_readback = readback.text
                raise GatewayError(
                    "needs_readback",
                    "Read this back to the caller word for word and wait for a yes. Then call the tool again with the same arguments.",
                    readback=readback.text,
                )
            event.confirmed = True
        else:
            event.confirmed = readback.delivered_in(self.prev_agent_text) and rb.confirmation(self.caller_text) is True
        self.pending = None
        self.confirmed_digest = None
        self.record.readback_status = "none"
        self.record.pending_readback = None

    def _identity(self, booking_id: str, phone: str | None) -> tuple[dict, list[rules.Violation]]:
        bid = booking_id.replace(" ", "").replace("-", "").upper()
        view = self.w.describe_booking(bid)
        if view is None:
            raise GatewayError("not_found_or_mismatch", "No booking matches that reference and phone number.")
        violations = []
        if view["phone"] != _clean_phone(phone) and bid not in self.verified:
            if self.cfg.enforce_rules:
                raise GatewayError("not_found_or_mismatch", "No booking matches that reference and phone number. If the caller insists, hand off to staff (reason identity).")
            violations.append(rules.Violation("identity_mismatch", "Phone number does not match the booking."))
        if view["status"] not in ("confirmed", "pending_deposit"):
            raise GatewayError("not_active", f"Booking {bid} is {view['status']}.")
        return view, violations

    def _new_id(self, domain: str) -> str:
        import random

        rng = random.Random(f"{self.caller_phone}-{len(self._ids)}-{domain}")
        return new_booking_id(rng, domain, self._ids)

    def _ensure_customer(self, phone: str, name: str) -> None:
        self.w.conn.execute(
            "INSERT OR IGNORE INTO customers (phone, name, city, created_on) VALUES (?,?,?,?)",
            (phone, name, "Hyderabad", self.now.date().isoformat()),
        )

    def _written(self, booking_id: str, action: str) -> dict:
        self._event.write = True
        self.record.set(track=False, booking_id=booking_id)
        self.record.completed_actions.append(f"{action} {booking_id}")
        view = self.w.describe_booking(booking_id)
        return {"booking": self._public(view)}

    # ================================================================ create
    def _create(self, a) -> dict:
        phone = _clean_phone(a.phone) or self.caller_phone
        if not phone or len(phone) != 10:
            raise GatewayError("invalid_args", "A 10-digit phone number is needed.")
        name = a.name.strip()
        if not name:
            raise GatewayError("invalid_args", "The name for the booking is needed.")
        if a.domain == "movies":
            return self._create_movie(a, name, phone)
        if a.domain == "buses":
            return self._create_bus(a, name, phone)
        return self._create_table(a, name, phone)

    def _movie_option(self, option_id: str, category: str | None) -> tuple[dict, str]:
        show_id, _, cat = option_id.partition(":")
        cat = cat or category
        if cat:
            cat = _resolve("seat category", cat, self.inv.cat.seat_category).id
        if cat not in CATEGORIES:
            raise GatewayError("invalid_args", "seat_category is needed (classic, prime or recliner).")
        return self.inv.show(show_id), cat

    def _create_movie(self, a, name: str, phone: str) -> dict:
        show, cat = self._movie_option(a.option_id, a.seat_category)
        qty = a.quantity or self.record.quantity
        if not qty or qty < 1:
            raise GatewayError("invalid_args", "quantity (number of tickets) is needed.")
        start = datetime.fromisoformat(show["date"]) + timedelta(minutes=show["start_min"])
        violations = rules.check_movie_booking(qty, start, self.now)
        self._precheck(violations)
        free = self.inv.free_seats(show, cat)
        seats = pick_cinema_seats(free, qty, show["seats_per_row"])
        if seats is None:
            raise GatewayError("no_availability", f"Only {len(free)} {cat} seats left for this show.", seats_left=len(free))
        amount = show[f"price_{cat}"] * qty
        self.record.set(track=False, domain="movies", option_id=a.option_id if ":" in a.option_id else f"{a.option_id}:{cat}", movie=show["title"],
                        cinema=show["cinema"], date=show["date"], time=fmt_time(show["start_min"]), seat_category=cat,
                        quantity=qty, caller_name=name, intent="book")
        readback = rb.movie_booking(show["title"], show["cinema"], show["date"], fmt_time(show["start_min"]), qty, cat, amount, name, phone)
        self._gate("create_booking", {"show": show["id"], "cat": cat, "qty": qty, "name": name.lower(), "phone": phone}, readback, violations)
        bid = self._new_id("movies")
        self._ensure_customer(phone, name)
        stamp = self.now.isoformat(timespec="minutes")
        self.w.conn.execute(
            """INSERT INTO bookings (id, domain, phone, name, status, show_id, seat_category, quantity, amount, created_at, updated_at, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,'agent')""",
            (bid, "movies", phone, name, "confirmed", show["id"], cat, qty, amount, stamp, stamp),
        )
        self.w.conn.executemany("INSERT INTO show_seats_taken VALUES (?,?,?)", [(show["id"], s, bid) for s in seats])
        self.created.add(bid)
        self._log_sms(phone, f"FastLane: {qty} {cat} ticket(s) for {show['title']}, {show['cinema']}, {show['date']} {fmt_time(show['start_min'])}, seats {', '.join(seats)}. Ref {bid}.")
        return self._written(bid, "created")

    def _passengers(self, a) -> list[dict]:
        if a.passengers:
            return [p.model_dump() for p in a.passengers]
        return [p.model_dump() for p in self.record.passengers]

    def _create_bus(self, a, name: str, phone: str) -> dict:
        trip = self.inv.trip(a.option_id)
        pax = self._passengers(a)
        qty = a.quantity or len(pax)
        if qty < 1:
            raise GatewayError("invalid_args", "passengers (name, age, gender for each seat) are needed.")
        depart = datetime.fromisoformat(trip["date"]) + timedelta(minutes=trip["depart_min"])
        violations = rules.check_bus_booking(qty, pax, depart, self.now)
        self._precheck(violations)
        free = self.inv.free_berths(trip)
        seats = pick_bus_seats(free, qty, trip["bus_type"], a.seat_preference)
        if seats is None:
            raise GatewayError("no_availability", f"Only {len(free)} seats left on this bus.", seats_left=len(free))
        amount = trip["fare"] * qty
        self.record.set(track=False, domain="buses", option_id=trip["id"], origin=trip["origin"], destination=trip["destination"], date=trip["date"],
                        time=fmt_time(trip["depart_min"]), bus_type=trip["bus_type"], quantity=qty, caller_name=name, intent="book",
                        passengers=[Passenger(**p) for p in pax if p.get("name")] or None, berth_preference=a.seat_preference)
        readback = rb.bus_booking(trip["origin"], trip["destination"], trip["date"], fmt_time(trip["depart_min"]), trip["bus_type"],
                                  pax or [{"name": name}], amount, phone)
        payload = {"trip": trip["id"], "qty": qty, "pax": [(p.get("name", "").lower(), p.get("age"), p.get("gender")) for p in pax],
                   "name": name.lower(), "phone": phone, "pref": a.seat_preference}
        self._gate("create_booking", payload, readback, violations)
        bid = self._new_id("buses")
        self._ensure_customer(phone, name)
        stamp = self.now.isoformat(timespec="minutes")
        self.w.conn.execute(
            """INSERT INTO bookings (id, domain, phone, name, status, trip_id, quantity, amount, created_at, updated_at, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,'agent')""",
            (bid, "buses", phone, name, "confirmed", trip["id"], qty, amount, stamp, stamp),
        )
        rows = []
        for i, s in enumerate(seats):
            p = pax[i] if i < len(pax) else {}
            rows.append((trip["id"], s, bid, p.get("name"), p.get("age"), p.get("gender")))
        self.w.conn.executemany("INSERT INTO bus_seats_taken VALUES (?,?,?,?,?,?)", rows)
        self.created.add(bid)
        self._log_sms(phone, f"FastLane: {trip['origin']}-{trip['destination']} {trip['date']} {fmt_time(trip['depart_min'])}, seats {', '.join(seats)}, board at {trip['boarding_point']}. Ref {bid}.")
        return self._written(bid, "created")

    def _table_option(self, option_id: str) -> tuple[dict, str, int]:
        try:
            rid, when = option_id.split("@")
            d, hhmm = when.split("T")
            h, m = map(int, hhmm.split(":"))
            date.fromisoformat(d)
        except ValueError:
            raise GatewayError("invalid_args", f"Bad restaurant option_id {option_id!r}.") from None
        return self.inv.restaurant(rid), d, h * 60 + m

    def _create_table(self, a, name: str, phone: str) -> dict:
        rest, d, start = self._table_option(a.option_id)
        party = a.party_size or self.record.party_size
        if not party:
            raise GatewayError("invalid_args", "party_size is needed.")
        when = datetime.fromisoformat(d) + timedelta(minutes=start)
        violations = rules.check_restaurant_booking(party, when, self.now, rest)
        deposit = rules.restaurant_deposit(party, rest)
        if deposit and a.waive_deposit:
            violations.append(rules.Violation("deposit_waived", f"Parties of {rest['deposit_min_party']}+ must pay a deposit; only staff can waive it."))
        self._precheck(violations)
        table = self.inv.free_table(rest["id"], d, start, party)
        if table is None and not any(v.code in ("party_size_limit", "closed", "outside_hours") for v in violations):
            raise GatewayError("no_availability", "No table of that size is free at that time.")
        self.record.set(track=False, domain="restaurants", option_id=a.option_id, restaurant=rest["name"], date=d, time=fmt_time(start),
                        party_size=party, caller_name=name, intent="book", special_request=a.special_request)
        charged = 0 if (a.waive_deposit and not self.cfg.enforce_rules) else deposit
        readback = rb.table_booking(rest["name"], d, fmt_time(start), party, name, phone, charged, a.special_request)
        payload = {"rest": rest["id"], "d": d, "start": start, "party": party, "name": name.lower(), "phone": phone,
                   "req": (a.special_request or "").lower(), "waive": a.waive_deposit}
        self._gate("create_booking", payload, readback, violations)
        if table is None:  # only reachable without enforcement: nothing physical to book
            raise GatewayError("no_availability", "No table of that size is free at that time.")
        bid = self._new_id("restaurants")
        self._ensure_customer(phone, name)
        stamp = self.now.isoformat(timespec="minutes")
        status = "pending_deposit" if charged else "confirmed"
        self.w.conn.execute(
            """INSERT INTO bookings (id, domain, phone, name, status, restaurant_id, res_date, res_start_min, party_size, quantity,
               amount, special_request, created_at, updated_at, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'agent')""",
            (bid, "restaurants", phone, name, status, rest["id"], d, start, party, party, charged, a.special_request, stamp, stamp),
        )
        self.w.conn.execute("INSERT INTO table_reservations VALUES (?,?,?,?,?)", (table, d, start, start + rules.SEATING_MIN, bid))
        self.created.add(bid)
        msg = f"FastLane: table for {party} at {rest['name']}, {d} {fmt_time(start)}. Ref {bid}."
        if charged:
            msg += f" Pay the deposit of Rs {charged} within 2 hours to confirm: https://pay.example/fastlane/{bid}"
        self._log_sms(phone, msg)
        out = self._written(bid, "created")
        if charged:
            out["deposit_due"] = charged
        return out

    # ================================================================ cancel
    def _cancel(self, a) -> dict:
        view, violations = self._identity(a.booking_id, a.phone)
        bid = view["booking_id"]
        start = datetime.fromisoformat(view["date"]) + timedelta(minutes=int(view["time"][:2]) * 60 + int(view["time"][3:]))
        b = self.w.one("SELECT amount, quantity, created_at FROM bookings WHERE id=?", (bid,))
        if view["domain"] == "movies":
            violations += rules.check_movie_change(start, self.now)
            refund = rules.movie_refund(b["amount"], b["quantity"])
            note = f"Refund of ₹{refund} after the ₹{rules.MOVIE_CANCEL_FEE_PER_TICKET} per ticket fee."
        elif view["domain"] == "buses":
            violations += rules.check_bus_cancel(start, self.now)
            pct = rules.bus_refund_percent(start, self.now, date.fromisoformat(b["created_at"][:10])) or 0
            refund = b["amount"] * pct // 100
            note = f"Refund of ₹{refund} ({pct}% of the fare)."
        else:
            violations += rules.check_restaurant_change(start, self.now)
            refund = rules.restaurant_deposit_refund(b["amount"], start, self.now)
            note = f"Deposit refund of ₹{refund}." if b["amount"] else "No charge."
        self.record.set(track=False, domain=view["domain"], booking_id=bid, intent="cancel")
        summary = self._summary(view)
        self._gate("cancel_booking", {"booking": bid}, rb.cancellation(bid, summary, note), violations)
        self._release(bid)
        stamp = self.now.isoformat(timespec="minutes")
        self.w.conn.execute(
            "UPDATE bookings SET status='cancelled', cancelled_at=?, updated_at=?, refund_amount=? WHERE id=?",
            (stamp, stamp, refund, bid),
        )
        self._log_sms(view["phone"], f"FastLane: booking {bid} cancelled. {note.replace('₹', 'Rs ')}")
        out = self._written(bid, "cancelled")
        out["refund_amount"] = refund
        return out

    def _release(self, bid: str) -> None:
        self.w.conn.execute("DELETE FROM show_seats_taken WHERE booking_id=?", (bid,))
        self.w.conn.execute("DELETE FROM bus_seats_taken WHERE booking_id=?", (bid,))
        self.w.conn.execute("DELETE FROM table_reservations WHERE booking_id=?", (bid,))

    @staticmethod
    def _summary(view: dict) -> str:
        when = f"{spoken_date(date.fromisoformat(view['date']))} at {rb.spoken_time(view['time'])}"
        if view["domain"] == "movies":
            return f"{view['quantity']} {view['seat_category']} tickets for {view['movie']}, {view['cinema']}, {when}"
        if view["domain"] == "buses":
            return f"{view['quantity']} seats {view['origin']} to {view['destination']}, {when}"
        return f"table for {view['party_size']} at {view['restaurant']}, {when}"

    # ================================================================ modify
    def _modify(self, a) -> dict:
        view, violations = self._identity(a.booking_id, a.phone)
        bid, dom = view["booking_id"], view["domain"]
        start = datetime.fromisoformat(view["date"]) + timedelta(minutes=int(view["time"][:2]) * 60 + int(view["time"][3:]))
        if dom == "movies":
            violations += rules.check_movie_change(start, self.now)
        elif dom == "buses":
            violations += rules.check_bus_change(start, self.now)
        else:
            violations += rules.check_restaurant_change(start, self.now)
        self.record.set(track=False, domain=dom, booking_id=bid, intent="modify")
        before = self._summary(view)
        # Dry-run the change inside a savepoint so the read-back describes exactly what will happen.
        self.w.conn.execute("SAVEPOINT modify")
        try:
            after_view, payload, new_violations, sms = self._apply_modification(view, a)
        except Exception:
            self.w.conn.execute("ROLLBACK TO modify")
            self.w.conn.execute("RELEASE modify")
            raise
        violations += new_violations
        after = self._summary(after_view)
        if after == before and not (a.passengers or a.special_request):
            self.w.conn.execute("ROLLBACK TO modify")
            self.w.conn.execute("RELEASE modify")
            raise GatewayError("no_change", "The requested change is the same as the current booking.")
        try:
            self._gate("modify_booking", {"booking": bid, **payload}, rb.modification(bid, before, after), violations)
        except GatewayError:
            self.w.conn.execute("ROLLBACK TO modify")
            self.w.conn.execute("RELEASE modify")
            raise
        self.w.conn.execute("RELEASE modify")
        self._log_sms(view["phone"], sms)
        return self._written(bid, "modified")

    def _apply_modification(self, view: dict, a) -> tuple[dict, dict, list[rules.Violation], str]:
        """Apply the change inside the open savepoint; return the new view, digest payload, violations, SMS."""
        bid, dom = view["booking_id"], view["domain"]
        stamp = self.now.isoformat(timespec="minutes")
        if dom == "movies":
            option = a.new_option_id or f"{view['show_id']}:{a.seat_category or view['seat_category']}"
            show, cat = self._movie_option(option, a.seat_category or view["seat_category"])
            qty = a.quantity or view["quantity"]
            start = datetime.fromisoformat(show["date"]) + timedelta(minutes=show["start_min"])
            violations = rules.check_movie_booking(qty, start, self.now)
            self._release(bid)
            free = self.inv.free_seats(show, cat)
            seats = pick_cinema_seats(free, qty, show["seats_per_row"])
            if seats is None:
                raise GatewayError("no_availability", f"Only {len(free)} {cat} seats free for that show.")
            self.w.conn.execute(
                "UPDATE bookings SET show_id=?, seat_category=?, quantity=?, amount=?, updated_at=? WHERE id=?",
                (show["id"], cat, qty, show[f"price_{cat}"] * qty, stamp, bid),
            )
            self.w.conn.executemany("INSERT INTO show_seats_taken VALUES (?,?,?)", [(show["id"], s, bid) for s in seats])
            payload = {"show": show["id"], "cat": cat, "qty": qty}
            sms = f"FastLane: booking {bid} changed to {show['title']}, {show['date']} {fmt_time(show['start_min'])}, seats {', '.join(seats)}."
        elif dom == "buses":
            trip = self.inv.trip(a.new_option_id or view["trip_id"])
            if (trip["origin"], trip["destination"]) != (view["origin"], view["destination"]):
                raise GatewayError("invalid_args", "A bus booking can be moved to another date or time on the same route only.")
            pax = [p.model_dump() for p in a.passengers] if a.passengers else [
                {"name": p["name"], "age": p["age"], "gender": p["gender"]} for p in view["passengers"]]
            qty = a.quantity or len(pax)
            start = datetime.fromisoformat(trip["date"]) + timedelta(minutes=trip["depart_min"])
            violations = rules.check_bus_booking(qty, pax, start, self.now)
            self._release(bid)
            free = self.inv.free_berths(trip)
            seats = pick_bus_seats(free, qty, trip["bus_type"])
            if seats is None:
                raise GatewayError("no_availability", f"Only {len(free)} seats free on that bus.")
            self.w.conn.execute(
                "UPDATE bookings SET trip_id=?, quantity=?, amount=?, updated_at=? WHERE id=?",
                (trip["id"], qty, trip["fare"] * qty, stamp, bid),
            )
            self.w.conn.executemany(
                "INSERT INTO bus_seats_taken VALUES (?,?,?,?,?,?)",
                [(trip["id"], s, bid, p.get("name"), p.get("age"), p.get("gender")) for s, p in zip(seats, pax + [{}] * qty)],
            )
            payload = {"trip": trip["id"], "qty": qty, "pax": [(p["name"].lower(), p["age"], p["gender"]) for p in pax]}
            sms = f"FastLane: booking {bid} changed to {trip['date']} {fmt_time(trip['depart_min'])}, seats {', '.join(seats)}."
        else:
            if a.new_option_id:
                rest, d, start_min = self._table_option(a.new_option_id)
            else:
                rest, d = self.inv.restaurant(view["restaurant_id"]), view["date"]
                start_min = int(view["time"][:2]) * 60 + int(view["time"][3:])
            party = a.party_size or view["party_size"]
            when = datetime.fromisoformat(d) + timedelta(minutes=start_min)
            violations = rules.check_restaurant_booking(party, when, self.now, rest)
            self._release(bid)
            table = self.inv.free_table(rest["id"], d, start_min, party)
            if table is None:
                raise GatewayError("no_availability", "No suitable table free at that time.")
            deposit = rules.restaurant_deposit(party, rest)
            old = self.w.one("SELECT status, amount, special_request FROM bookings WHERE id=?", (bid,))
            status = "pending_deposit" if deposit > old["amount"] else old["status"]
            self.w.conn.execute(
                """UPDATE bookings SET restaurant_id=?, res_date=?, res_start_min=?, party_size=?, quantity=?, amount=?, status=?,
                   special_request=?, updated_at=? WHERE id=?""",
                (rest["id"], d, start_min, party, party, max(deposit, old["amount"]), status, a.special_request or old["special_request"], stamp, bid),
            )
            self.w.conn.execute("INSERT INTO table_reservations VALUES (?,?,?,?,?)", (table, d, start_min, start_min + rules.SEATING_MIN, bid))
            payload = {"rest": rest["id"], "d": d, "start": start_min, "party": party, "req": (a.special_request or "").lower()}
            sms = f"FastLane: booking {bid} changed to {rest['name']}, {d} {fmt_time(start_min)}, {party} guests."
            if status == "pending_deposit" and deposit > old["amount"]:
                sms += f" A deposit of Rs {deposit} is now due: https://pay.example/fastlane/{bid}"
        return self.w.describe_booking(bid), payload, violations, sms

    # ================================================================ metrics helpers
    def unsafe_writes(self) -> list[ToolEvent]:
        return [e for e in self.events if e.write and (e.violations or not e.confirmed)]

    def write_events(self) -> list[ToolEvent]:
        return [e for e in self.events if e.tool in WRITE_TOOLS and e.write]

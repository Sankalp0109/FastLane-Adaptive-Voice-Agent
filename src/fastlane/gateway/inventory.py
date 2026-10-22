"""Read-only inventory queries: search, availability and alternatives.

Option ids:
  movies       "<show_id>:<category>"            e.g. SH00172:prime
  buses        "<trip_id>"                       e.g. TR00045
  restaurants  "<restaurant_id>@<date>T<HH:MM>"  e.g. tandoor-jubilee@2026-10-17T20:00
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from fastlane.gateway.dates import parse_time
from fastlane.gateway.matching import Catalogue, Match, norm
from fastlane.gateway.readback import spoken_time
from fastlane.gateway.rules import (
    BUS_BOOKING_CLOSES_MIN,
    MOVIE_BOOKING_CLOSES_MIN,
    RESTAURANT_BOOKING_CLOSES_MIN,
    SEATING_MIN,
)
from fastlane.world.db import World, fmt_time
from fastlane.world.seating import SEATER_SEATS, SLEEPER_BERTHS, cinema_seats

CATEGORIES = ("classic", "prime", "recliner")
TABLE_SLACK = 3  # a party may take a table with up to 3 spare seats


class InventoryError(Exception):
    def __init__(self, code: str, message: str, **data):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


def _dt(d: str, minutes: int) -> datetime:
    return datetime.fromisoformat(d) + timedelta(minutes=minutes)


def _parse_iso(d: str) -> str:
    try:
        return date.fromisoformat(d.strip()).isoformat()
    except (ValueError, AttributeError):
        raise InventoryError("invalid_args", f"date must be YYYY-MM-DD, got {d!r}; use resolve_date for spoken dates") from None


def _resolve(kind: str, query: str | None, finder) -> Match | None:
    if not query:
        return None
    match, ambiguous = finder(query)
    if match:
        return match
    if ambiguous:
        raise InventoryError("ambiguous", f"{query!r} could match several {kind}s; ask the caller which one.",
                             candidates=[m.label for m in ambiguous])
    raise InventoryError("not_found", f"No {kind} matches {query!r}.")


def _time_window(time: str | None) -> tuple[int, int] | None:
    if not time:
        return None
    window = parse_time(time)
    if window is None:
        raise InventoryError("invalid_args", f"could not read time {time!r}; use HH:MM")
    return window


def _time_rank(start: int, window: tuple[int, int] | None, slack: int = 150) -> float | None:
    """None = outside the window. Exact times keep anything within `slack` minutes, ranked by distance."""
    if window is None:
        return 0.0
    a, b = window
    if a == b:
        dist = abs(start - a)
        return dist if dist <= slack else None
    return 0.0 if a <= start <= b else None


class Inventory:
    def __init__(self, world: World, now: datetime):
        self.w = world
        self.now = now
        self.cat = Catalogue(world)

    # ================================================================ movies
    def show(self, show_id: str) -> dict:
        s = self.w.one(
            """SELECT s.*, m.title, m.certificate, m.runtime_min, sc.cinema_id, sc.rows, sc.seats_per_row,
                      c.name || ' ' || c.area AS cinema
               FROM shows s JOIN movies m ON m.id=s.movie_id JOIN screens sc ON sc.id=s.screen_id
               JOIN cinemas c ON c.id=sc.cinema_id WHERE s.id=?""",
            (show_id,),
        )
        if s is None:
            raise InventoryError("not_found", f"No show {show_id}.")
        return s

    def free_seats(self, show: dict, category: str, ignore_booking: str | None = None) -> list[str]:
        rows = "".join(r["row"] for r in self.w.q(
            "SELECT row FROM seat_categories WHERE screen_id=? AND category=? ORDER BY row", (show["screen_id"], category)))
        taken = {r["seat"] for r in self.w.q(
            "SELECT seat FROM show_seats_taken WHERE show_id=? AND (booking_id IS NULL OR booking_id IS NOT ?)",
            (show["id"], ignore_booking))}
        return [s for s in cinema_seats(rows, show["seats_per_row"]) if s not in taken]

    def show_option(self, show: dict, category: str, qty: int) -> dict:
        left = len(self.free_seats(show, category))
        return {
            "option_id": f"{show['id']}:{category}",
            "category": category,
            "price": show[f"price_{category}"],
            "seats_left": left,
            "available": left >= qty,
        }

    def _show_open(self, show: dict) -> bool:
        return _dt(show["date"], show["start_min"]) - self.now >= timedelta(minutes=MOVIE_BOOKING_CLOSES_MIN)

    def search_movies(self, d: str, time: str | None, qty: int | None, movie: str | None, cinema: str | None,
                      category: str | None, limit: int = 8) -> dict:
        d = _parse_iso(d)
        qty = qty or 1
        m = _resolve("movie", movie, self.cat.movie)
        c = _resolve("cinema", cinema, self.cat.cinema)
        cat = _resolve("seat category", category, self.cat.seat_category)
        window = _time_window(time)
        sql = """SELECT s.id FROM shows s JOIN screens sc ON sc.id=s.screen_id WHERE s.date=?"""
        params: list = [d]
        if m:
            sql += " AND s.movie_id=?"
            params.append(m.id)
        if c:
            sql += " AND sc.cinema_id=?"
            params.append(c.id)
        rows = [self.show(r["id"]) for r in self.w.q(sql + " ORDER BY s.start_min, sc.cinema_id", tuple(params))]
        if m and not rows:
            rel = self.w.one("SELECT release_date FROM movies WHERE id=?", (m.id,))["release_date"]
            if rel > d:
                raise InventoryError("not_released", f"{m.label} releases on {rel}.", release_date=rel)
        ranked = []
        for s in rows:
            if not self._show_open(s):
                continue
            rank = _time_rank(s["start_min"], window)
            if rank is None:
                continue
            cats = [cat.id] if cat else list(CATEGORIES)
            ranked.append((rank, s["start_min"], s, [self.show_option(s, k, qty) for k in cats]))
        ranked.sort(key=lambda x: (x[0], x[1]))
        results = [{
            "show_id": s["id"], "movie": s["title"], "certificate": s["certificate"], "cinema": s["cinema"],
            "date": s["date"], "time": fmt_time(s["start_min"]), "spoken_time": spoken_time(fmt_time(s["start_min"])),
            "options": opts,
        } for _, _, s, opts in ranked[:limit]]
        exact = any(o["available"] for _, start, _, opts in ranked for o in opts
                    if window is None or window[0] != window[1] or start == window[0])
        out = {"results": results, "exact_match_available": exact, "quantity": qty,
               "resolved": {"movie": m.label if m else None, "cinema": c.label if c else None, "seat_category": cat.id if cat else None}}
        if not results and m:
            where = [r["cinema"] for r in self.w.q(
                """SELECT DISTINCT c.name || ' ' || c.area AS cinema FROM shows s JOIN screens sc ON sc.id=s.screen_id
                   JOIN cinemas c ON c.id=sc.cinema_id WHERE s.date=? AND s.movie_id=?""", (d, m.id))]
            out["note"] = (f"No matching shows. {m.label} is showing that day at: {', '.join(where)}." if where
                           else f"{m.label} has no shows that day.")
        elif not results:
            showing = [r["title"] for r in self.w.q(
                "SELECT DISTINCT m.title FROM shows s JOIN movies m ON m.id=s.movie_id WHERE s.date=?", (d,))]
            out["note"] = "No matching shows. Films showing that day: " + ", ".join(showing)
        return out

    # ================================================================= buses
    def trip(self, trip_id: str) -> dict:
        t = self.w.one(
            """SELECT t.*, r.origin, r.destination, r.duration_min, r.boarding_point
               FROM bus_trips t JOIN bus_routes r ON r.id=t.route_id WHERE t.id=?""",
            (trip_id,),
        )
        if t is None:
            raise InventoryError("not_found", f"No bus trip {trip_id}.")
        return t

    def free_berths(self, trip: dict, ignore_booking: str | None = None) -> list[str]:
        taken = {r["seat"] for r in self.w.q(
            "SELECT seat FROM bus_seats_taken WHERE trip_id=? AND (booking_id IS NULL OR booking_id IS NOT ?)",
            (trip["id"], ignore_booking))}
        layout = SLEEPER_BERTHS if "sleeper" in trip["bus_type"] else SEATER_SEATS
        return [s for s in layout if s not in taken]

    def _bus_types(self, query: str | None) -> set[str] | None:
        if not query:
            return None
        match, ambiguous = self.cat.bus_type(query)
        if match:
            return {match.id}
        q = norm(query)
        families = {"sleeper", "seater"} & set(q.split())
        types = {"ac_sleeper", "nonac_sleeper", "ac_seater", "nonac_seater"}
        if families:
            types = {t for t in types if any(f in t for f in families)}
        if "non" in q:
            types = {t for t in types if t.startswith("nonac")}
        elif " ac" in f" {q}":
            types = {t for t in types if t.startswith("ac")}
        if ambiguous or families:
            return types
        raise InventoryError("not_found", f"Unknown bus type {query!r}; we run AC/non-AC sleepers and seaters.")

    def trip_view(self, t: dict, qty: int) -> dict:
        free = self.free_berths(t)
        arrive = _dt(t["date"], t["depart_min"] + t["duration_min"])
        view = {
            "option_id": t["id"], "route": f"{t['origin']} to {t['destination']}", "date": t["date"],
            "departs": fmt_time(t["depart_min"]), "spoken_time": spoken_time(fmt_time(t["depart_min"])),
            "arrives": arrive.strftime("%Y-%m-%d %H:%M"), "bus_type": t["bus_type"], "fare": t["fare"],
            "seats_left": len(free), "available": len(free) >= qty, "boarding_point": t["boarding_point"],
        }
        if "sleeper" in t["bus_type"]:
            view["lower_berths_left"] = sum(1 for s in free if s.startswith("L"))
        return view

    def _route(self, origin: str | None, destination: str | None) -> dict:
        if not destination:
            raise InventoryError("invalid_args", "destination is needed for bus searches")
        o = _resolve("city", origin or "Hyderabad", self.cat.city)
        dest = _resolve("city", destination, self.cat.city)
        r = self.w.one("SELECT * FROM bus_routes WHERE origin=? AND destination=?", (o.id, dest.id))
        if r is None:
            served = ", ".join(f"{x['origin']}-{x['destination']}" for x in self.w.q("SELECT origin, destination FROM bus_routes"))
            raise InventoryError("not_found", f"We do not run {o.id} to {dest.id}. Routes: {served}.")
        return r

    def search_buses(self, d: str, time: str | None, qty: int | None, origin: str | None, destination: str | None,
                     bus_type: str | None, limit: int = 8) -> dict:
        d = _parse_iso(d)
        qty = qty or 1
        route = self._route(origin, destination)
        types = self._bus_types(bus_type)
        window = _time_window(time)
        trips = self.w.q("SELECT id FROM bus_trips WHERE route_id=? AND date=? ORDER BY depart_min", (route["id"], d))
        ranked = []
        for r in trips:
            t = self.trip(r["id"])
            if types and t["bus_type"] not in types:
                continue
            if _dt(t["date"], t["depart_min"]) - self.now < timedelta(minutes=BUS_BOOKING_CLOSES_MIN):
                continue
            rank = _time_rank(t["depart_min"], window, slack=180)
            if rank is None:
                continue
            ranked.append((rank, t["depart_min"], self.trip_view(t, qty)))
        ranked.sort(key=lambda x: (x[0], x[1]))
        results = [v for _, _, v in ranked[:limit]]
        exact = any(v["available"] for _, start, v in ranked if window is None or window[0] != window[1] or start == window[0])
        return {"results": results, "exact_match_available": exact, "quantity": qty,
                "resolved": {"origin": route["origin"], "destination": route["destination"],
                             "bus_type": next(iter(types)) if types and len(types) == 1 else None}}

    # =========================================================== restaurants
    def restaurant(self, rid: str) -> dict:
        r = self.w.one("SELECT * FROM restaurants WHERE id=?", (rid,))
        if r is None:
            raise InventoryError("not_found", f"No restaurant {rid}.")
        return r

    def free_table(self, rid: str, d: str, start: int, party: int, zone: str | None = None,
                   ignore_booking: str | None = None) -> str | None:
        tables = self.w.q(
            "SELECT id, seats, zone FROM restaurant_tables WHERE restaurant_id=? AND seats>=? AND seats<=? ORDER BY seats, id",
            (rid, party, party + TABLE_SLACK),
        )
        if zone:
            tables.sort(key=lambda t: (t["zone"] != zone, t["seats"], t["id"]))
        for t in tables:
            clash = self.w.one(
                """SELECT 1 FROM table_reservations WHERE table_id=? AND date=? AND start_min < ? AND end_min > ?
                   AND (booking_id IS NULL OR booking_id IS NOT ?)""",
                (t["id"], d, start + SEATING_MIN, start, ignore_booking),
            )
            if not clash:
                return t["id"]
        return None

    def seating_starts(self, rest: dict, d: str) -> list[int]:
        if rest["closed_weekday"] is not None and date.fromisoformat(d).weekday() == rest["closed_weekday"]:
            return []
        starts = list(range(rest["lunch_open"], rest["lunch_last_seating"] + 1, 30))
        starts += list(range(rest["dinner_open"], rest["dinner_last_seating"] + 1, 30))
        return [s for s in starts if _dt(d, s) - self.now >= timedelta(minutes=RESTAURANT_BOOKING_CLOSES_MIN)]

    def search_tables(self, d: str, time: str | None, party: int | None, restaurant: str | None, limit: int = 6) -> dict:
        d = _parse_iso(d)
        if not party:
            raise InventoryError("invalid_args", "party_size is needed for table searches")
        r = _resolve("restaurant", restaurant, self.cat.restaurant)
        rests = [self.restaurant(r.id)] if r else self.w.q("SELECT * FROM restaurants ORDER BY id")
        window = _time_window(time)
        results, exact = [], False
        for rest in rests:
            entry = {"restaurant": rest["name"], "restaurant_id": rest["id"], "date": d, "max_party": rest["max_party"]}
            if party > rest["max_party"]:
                entry["note"] = f"Takes phone bookings up to {rest['max_party']} guests; larger groups go to staff."
                results.append(entry)
                continue
            starts = self.seating_starts(rest, d)
            if not starts:
                entry["note"] = "Closed that day." if rest["closed_weekday"] is not None and date.fromisoformat(d).weekday() == rest["closed_weekday"] else "No seatings left today."
                results.append(entry)
                continue
            times = []
            for s in starts:
                rank = _time_rank(s, window, slack=120)
                if rank is None:
                    continue
                if self.free_table(rest["id"], d, s, party):
                    times.append((rank, s))
            times.sort()
            if window and window[0] == window[1]:
                entry["requested_time_available"] = any(s == window[0] for _, s in times)
                exact = exact or entry["requested_time_available"]
            else:
                exact = exact or bool(times)
            entry["times"] = [{
                "option_id": f"{rest['id']}@{d}T{fmt_time(s)}", "time": fmt_time(s), "spoken_time": spoken_time(fmt_time(s)),
            } for _, s in times[:limit]]
            if party >= rest["deposit_min_party"]:
                entry["deposit_required"] = party * rest["deposit_per_guest"]
            results.append(entry)
        return {"results": results, "exact_match_available": exact, "party_size": party,
                "resolved": {"restaurant": r.label if r else None}}

    # =========================================================== alternatives
    def alternatives(self, args, limit: int = 6) -> dict:
        """Closest available options to what the caller wanted, each with a reason."""
        d = _parse_iso(args.date)
        day = date.fromisoformat(d)
        nxt = (day + timedelta(days=1)).isoformat()
        alts: list[tuple[float, dict]] = []
        window = _time_window(args.time)
        want = window[0] if window else None

        def dist(t: str) -> float:
            if want is None:
                return 0
            h, m = map(int, t.split(":"))
            return abs(h * 60 + m - want) / 60

        if args.domain == "movies":
            movie = args.movie
            category = args.seat_category
            if args.option_id and ":" in args.option_id:
                show = self.show(args.option_id.split(":")[0])
                movie = movie or show["title"]
                category = category or args.option_id.split(":")[1]
                want = want if want is not None else show["start_min"]
                args.cinema = args.cinema or show["cinema"]
            qty = args.quantity or 1
            if not movie:
                raise InventoryError("invalid_args", "movie (or option_id) is needed to find alternatives")
            same_cinema = self.search_movies(d, None, qty, movie, args.cinema, None, limit=20)["results"] if args.cinema else []
            anywhere = self.search_movies(d, None, qty, movie, None, None, limit=40)["results"]
            tomorrow = self.search_movies(nxt, None, qty, movie, args.cinema, None, limit=20)["results"] if day < self.now.date() + timedelta(days=29) else []
            wanted_cinema = _resolve("cinema", args.cinema, self.cat.cinema).label if args.cinema else None
            for res, base, why in ((same_cinema, 0, "same cinema, different time"), (anywhere, 1, "different cinema"), (tomorrow, 2.5, "next day")):
                for show in res:
                    for o in show["options"]:
                        if not o["available"]:
                            continue
                        if base == 1 and show["cinema"] == wanted_cinema:
                            continue
                        same_slot = want is not None and show["time"] == fmt_time(want) and base == 0
                        reason = f"{o['category']} seats at the same show" if same_slot else why
                        cat_penalty = 0 if (category is None or o["category"] == category) else 0.7
                        if same_slot and category and o["category"] == category:
                            continue
                        score = base + dist(show["time"]) + cat_penalty
                        alts.append((score, {"why": reason, "option_id": o["option_id"], "movie": show["movie"], "cinema": show["cinema"],
                                             "date": show["date"], "time": show["time"], "spoken_time": show["spoken_time"],
                                             "category": o["category"], "price": o["price"], "seats_left": o["seats_left"]}))
        elif args.domain == "buses":
            qty = args.quantity or 1
            origin, destination, types = args.origin, args.destination, args.bus_type
            if args.option_id:
                t = self.trip(args.option_id)
                origin, destination = origin or t["origin"], destination or t["destination"]
                types = types or t["bus_type"]
                want = want if want is not None else t["depart_min"]
            wanted_types = self._bus_types(types) if types else None
            prev = (day - timedelta(days=1)).isoformat()
            for when, base, why in ((d, 0, "same day"), (nxt, 2.5, "next day"), (prev, 3, "day before")):
                if date.fromisoformat(when) < self.now.date():
                    continue
                for v in self.search_buses(when, None, qty, origin, destination, None, limit=20)["results"]:
                    if not v["available"] or v["option_id"] == args.option_id:
                        continue
                    type_penalty = 0 if (not wanted_types or v["bus_type"] in wanted_types) else 1.5
                    reason = why if type_penalty == 0 else f"{why}, {v['bus_type'].replace('_', ' ')} instead"
                    h, m = map(int, v["departs"].split(":"))
                    hours_off = abs(h * 60 + m - want) / 60 if want is not None else 0
                    alts.append((base + type_penalty + hours_off / 2, {"why": reason, **v}))
        else:
            party = args.party_size
            if not party:
                raise InventoryError("invalid_args", "party_size is needed")
            rid = None
            if args.option_id and "@" in args.option_id:
                rid, rest_of = args.option_id.split("@")
                want = want if want is not None else int(rest_of[-5:-3]) * 60 + int(rest_of[-2:])
            elif args.restaurant:
                rid = _resolve("restaurant", args.restaurant, self.cat.restaurant).id
            if want is None:
                raise InventoryError("invalid_args", "time (or option_id) is needed to find alternatives")
            for rest in self.w.q("SELECT * FROM restaurants ORDER BY id"):
                if party > rest["max_party"]:
                    continue
                other_branch = rid is not None and rest["id"] != rid
                for when, day_pen in ((d, 0), (nxt, 2.5)):
                    for s in self.seating_starts(rest, when):
                        hours_off = abs(s - want) / 60
                        if hours_off > 2.5 or (when == d and s == want and not other_branch):
                            continue
                        if not self.free_table(rest["id"], when, s, party):
                            continue
                        why = ("other branch" if other_branch else "same branch") + (", next day" if day_pen else ", different time" if s != want else "")
                        alts.append((day_pen + hours_off + (1.0 if other_branch else 0), {
                            "why": why, "option_id": f"{rest['id']}@{when}T{fmt_time(s)}", "restaurant": rest["name"],
                            "date": when, "time": fmt_time(s), "spoken_time": spoken_time(fmt_time(s)),
                        }))
        alts.sort(key=lambda x: x[0])
        seen, out = set(), []
        for _, a in alts:
            if a["option_id"] in seen:
                continue
            seen.add(a["option_id"])
            out.append(a)
            if len(out) == limit:
                break
        return {"alternatives": out, "found": bool(out)}

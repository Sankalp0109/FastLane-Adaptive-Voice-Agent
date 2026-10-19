"""Build the deterministic practice world (Table 4 of the proposal).

Movies: 4 cinemas x 3 screens x 4 shows/day. Buses: 6 routes (3 city pairs, both ways), 4 trips/day.
Restaurants: 3 branches of one brand, tables by size, 90-minute seatings.
Plus ~200 customers, ~500 past bookings for returning callers and ~70 upcoming bookings
that test calls can change or cancel. Some inventory is planted as sold out.
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from fastlane.config import BOOKING_WINDOW_DAYS, CITY, FESTIVALS, HISTORY_DAYS, NOW, SEED, TODAY, WORLD_DB
from fastlane.world import names
from fastlane.world.seating import SEATER_SEATS, SLEEPER_BERTHS, cinema_seats, pick_bus_seats, pick_cinema_seats

SCHEMA = Path(__file__).with_name("schema.sql")

# ------------------------------------------------------------------ catalogue

CINEMAS = [
    ("galaxy-kukatpally", "Galaxy Cinemas", "Kukatpally", "4th floor, Nexa Mall, KPHB Road, Kukatpally"),
    ("starlight-gachibowli", "Starlight Multiplex", "Gachibowli", "Sarath Arcade, Old Mumbai Highway, Gachibowli"),
    ("regal-ameerpet", "Regal Talkies", "Ameerpet", "Opposite Metro Pillar 1102, Ameerpet"),
    ("moonbeam-secunderabad", "Moonbeam Cinemas", "Secunderabad", "SD Road, near Paradise Circle, Secunderabad"),
]
# (screen name, rows, seats per row, {category: rows})
SCREEN_LAYOUTS = [
    ("Screen 1", "ABCDEFGHIJ", 14, {"classic": "ABCDE", "prime": "FGHI", "recliner": "J"}),
    ("Screen 2", "ABCDEFGH", 12, {"classic": "ABCD", "prime": "EFG", "recliner": "H"}),
    ("Screen 3", "ABCDEF", 10, {"classic": "ABC", "prime": "DE", "recliner": "F"}),
]
SHOW_SLOTS = [600, 810, 1020, 1230]  # 10:00, 13:30, 17:00, 20:30
BASE_PRICES = {"classic": 180, "prime": 250, "recliner": 450}


@dataclass(frozen=True)
class Movie:
    id: str
    title: str
    language: str
    genre: str
    certificate: str
    runtime_min: int
    release: date
    popularity: float


MOVIES = [
    Movie("kaala-samudram", "Kaala Samudram", "Telugu", "action", "UA", 158, date(2026, 10, 2), 1.35),
    Movie("dil-ki-dastak", "Dil Ki Dastak", "Hindi", "romance", "U", 142, date(2026, 9, 25), 0.9),
    Movie("last-monsoon", "The Last Monsoon", "English", "drama", "UA", 121, date(2026, 10, 9), 0.7),
    Movie("rocket-raja", "Rocket Raja", "Hindi", "comedy", "U", 134, date(2026, 10, 16), 1.2),
    Movie("godavari-express", "Godavari Express", "Telugu", "thriller", "UA", 139, date(2026, 9, 18), 0.8),
    Movie("shadow-protocol", "Shadow Protocol", "English", "action", "A", 128, date(2026, 10, 9), 0.95),
    Movie("little-champs", "Little Champs in Space", "Hindi", "animation", "U", 98, date(2026, 10, 2), 0.75),
    Movie("chennai-nights", "Chennai Nights", "Tamil", "thriller", "A", 136, date(2026, 10, 23), 0.85),
    Movie("diwali-dhamaka", "Diwali Dhamaka", "Hindi", "comedy", "U", 145, date(2026, 11, 6), 1.5),
]

ROUTES = [
    ("HYD-BLR", "Hyderabad", "Bengaluru", 575, 600, "Ameerpet Metro, Hyderabad"),
    ("BLR-HYD", "Bengaluru", "Hyderabad", 575, 600, "Majestic Bus Stand, Bengaluru"),
    ("HYD-VJA", "Hyderabad", "Vijayawada", 275, 330, "LB Nagar Ring Road, Hyderabad"),
    ("VJA-HYD", "Vijayawada", "Hyderabad", 275, 330, "Pandit Nehru Bus Station, Vijayawada"),
    ("HYD-CHE", "Hyderabad", "Chennai", 630, 660, "Ameerpet Metro, Hyderabad"),
    ("CHE-HYD", "Chennai", "Hyderabad", 630, 660, "Koyambedu, Chennai"),
]
# Per city pair: (depart_min, bus_type, base fare)
TRIP_TEMPLATES = {
    "BLR": [(480, "ac_seater", 1100), (1260, "ac_sleeper", 1450), (1320, "ac_sleeper", 1350), (1350, "nonac_sleeper", 950)],
    "VJA": [(360, "ac_seater", 650), (840, "nonac_seater", 450), (1320, "ac_sleeper", 850), (1410, "nonac_sleeper", 600)],
    "CHE": [(420, "ac_seater", 1200), (1170, "ac_sleeper", 1550), (1260, "ac_sleeper", 1450), (1305, "nonac_sleeper", 1050)],
}

RESTAURANTS = [
    # id, name, area, address, max_party, deposit_min_party, deposit_per_guest, lunch, dinner, closed_weekday
    ("tandoor-jubilee", "Tandoor Tales Jubilee Hills", "Jubilee Hills", "Road No. 36, Jubilee Hills", 12, 8, 300, (720, 870), (1140, 1320), None),
    ("tandoor-banjara", "Tandoor Tales Banjara Hills", "Banjara Hills", "Road No. 12, Banjara Hills", 10, 8, 300, (720, 870), (1140, 1320), 0),
    ("tandoor-hitech", "Tandoor Tales Hitech City", "Hitech City", "Cyber Towers Lane, Hitech City", 8, 6, 250, (720, 870), (1140, 1290), None),
]
# restaurant -> list of (seats, zone, count)
TABLE_LAYOUTS = {
    "tandoor-jubilee": [(2, "indoor", 6), (4, "indoor", 5), (4, "outdoor", 2), (6, "indoor", 3), (8, "indoor", 1), (12, "private", 1)],
    "tandoor-banjara": [(2, "indoor", 5), (4, "indoor", 5), (6, "indoor", 2), (10, "private", 1)],
    "tandoor-hitech": [(2, "indoor", 6), (4, "indoor", 4), (4, "outdoor", 1), (6, "indoor", 2), (8, "indoor", 1)],
}
SEATING_MIN = 90

FESTIVAL_DATES = set(FESTIVALS.values())


def is_peak(d: date) -> bool:
    return d.weekday() >= 4 or d in FESTIVAL_DATES


def window_dates() -> list[date]:
    return [TODAY + timedelta(days=i) for i in range(BOOKING_WINDOW_DAYS)]


def fmt_time(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


_ID_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I, easier to read back on a call
_PREFIX = {"movies": "MV", "buses": "BS", "restaurants": "RT"}


def new_booking_id(rng: random.Random, domain: str, taken: set[str]) -> str:
    while True:
        bid = _PREFIX[domain] + "".join(rng.choice(_ID_ALPHABET) for _ in range(5))
        if bid not in taken:
            taken.add(bid)
            return bid


# ---------------------------------------------------------------------- builder


class _Builder:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.rng = random.Random(SEED)
        self.booking_ids: set[str] = set()
        self.phones: set[str] = set()
        self.planted: list[dict] = []
        self.profiles: dict[str, dict] = {}
        self.screen_layout: dict[str, tuple[str, int, dict[str, str]]] = {}

    # ---------- movies
    def movies(self) -> None:
        c = self.conn
        for cid, name, area, address in CINEMAS:
            c.execute("INSERT INTO cinemas VALUES (?,?,?,?)", (cid, name, area, address))
            for i, (sname, rows, per_row, cats) in enumerate(SCREEN_LAYOUTS, start=1):
                sid = f"{cid}-s{i}"
                c.execute("INSERT INTO screens VALUES (?,?,?,?,?)", (sid, cid, sname, rows, per_row))
                for cat, cat_rows in cats.items():
                    for r in cat_rows:
                        c.execute("INSERT INTO seat_categories VALUES (?,?,?)", (sid, r, cat))
                self.screen_layout[sid] = (rows, per_row, cats)
        for m in MOVIES:
            c.execute(
                "INSERT INTO movies VALUES (?,?,?,?,?,?,?)",
                (m.id, m.title, m.language, m.genre, m.certificate, m.runtime_min, m.release.isoformat()),
            )

        show_no = 0
        for d in window_dates():
            showing = [m for m in MOVIES if m.release <= d]
            weights = [m.popularity * (1.6 if (d - m.release).days < 10 else 1.0) for m in showing]
            for cid, *_ in CINEMAS:
                for i in range(1, len(SCREEN_LAYOUTS) + 1):
                    sid = f"{cid}-s{i}"
                    # A screen keeps one film all day; big screen 1 gets the most popular film.
                    if i == 1:
                        movie = max(showing, key=lambda m, d=d: m.popularity * (1.6 if (d - m.release).days < 10 else 1.0))
                    else:
                        movie = self.rng.choices(showing, weights=weights)[0]
                    for slot in SHOW_SLOTS:
                        show_no += 1
                        show_id = f"SH{show_no:05d}"
                        surcharge = 40 if is_peak(d) else 0
                        morning = -30 if slot == 600 else 0
                        prices = (
                            BASE_PRICES["classic"] + surcharge + morning,
                            BASE_PRICES["prime"] + surcharge + morning,
                            BASE_PRICES["recliner"] + surcharge + 10,
                        )
                        c.execute(
                            "INSERT INTO shows VALUES (?,?,?,?,?,?,?,?)",
                            (show_id, sid, movie.id, d.isoformat(), slot, *prices),
                        )
                        self._fill_show(show_id, sid, movie, d, slot)

    def _fill_show(self, show_id: str, screen_id: str, movie: Movie, d: date, slot: int) -> None:
        rows, per_row, cats = self.screen_layout[screen_id]
        days_ahead = (d - TODAY).days
        base = {600: 0.15, 810: 0.3, 1020: 0.4, 1230: 0.55}[slot]
        fill = base * movie.popularity * (1.45 if is_peak(d) else 1.0) * max(0.25, 1 - days_ahead / 25)
        sold_out = False
        # Planted sell-outs for the "needs an alternative" calls.
        if movie.id == "kaala-samudram" and d in (date(2026, 10, 16), date(2026, 10, 17)) and slot == 1230:
            sold_out = True
        if movie.id == "rocket-raja" and d == date(2026, 10, 16) and slot >= 1020 and screen_id.startswith("galaxy"):
            sold_out = True
        if movie.id == "diwali-dhamaka" and (d == date(2026, 11, 6) or (d == date(2026, 11, 8) and slot >= 1020)):
            sold_out = True
        if sold_out:
            self.planted.append({"domain": "movies", "kind": "show_sold_out", "show_id": show_id, "movie_id": movie.id, "date": d.isoformat(), "time": fmt_time(slot)})
        taken = []
        for cat, cat_rows in cats.items():
            cat_fill = min(0.92, fill + (0.2 if cat == "recliner" else 0.0))
            # Recliners on peak evenings are gone (category-level alternative needed).
            recliners_gone = cat == "recliner" and is_peak(d) and slot == 1230 and days_ahead <= 10
            for r in cat_rows:
                for n in range(1, per_row + 1):
                    if sold_out or recliners_gone or self.rng.random() < cat_fill:
                        taken.append((show_id, f"{r}{n}", None))
            if recliners_gone and not sold_out:
                self.planted.append({"domain": "movies", "kind": "category_sold_out", "show_id": show_id, "category": "recliner", "date": d.isoformat(), "time": fmt_time(slot)})
        self.conn.executemany("INSERT INTO show_seats_taken VALUES (?,?,?)", taken)

    # ---------- buses
    def buses(self) -> None:
        c = self.conn
        for r in ROUTES:
            c.execute("INSERT INTO bus_routes VALUES (?,?,?,?,?,?)", r)
        trip_no = 0
        for d in window_dates():
            for route_id, origin, dest, *_ in ROUTES:
                pair = route_id.replace("HYD", "").strip("-")
                for depart, bus_type, fare in TRIP_TEMPLATES[pair]:
                    trip_no += 1
                    trip_id = f"TR{trip_no:05d}"
                    fare_today = round(fare * (1.15 if is_peak(d) else 1.0) / 10) * 10
                    capacity = len(SLEEPER_BERTHS) if "sleeper" in bus_type else len(SEATER_SEATS)
                    c.execute("INSERT INTO bus_trips VALUES (?,?,?,?,?,?,?)", (trip_id, route_id, d.isoformat(), depart, bus_type, fare_today, capacity))
                    self._fill_trip(trip_id, route_id, d, depart, bus_type)

    def _fill_trip(self, trip_id: str, route_id: str, d: date, depart: int, bus_type: str) -> None:
        days_ahead = (d - TODAY).days
        fill = (0.35 if depart < 1080 else 0.5) * (1.5 if is_peak(d) else 1.0) * max(0.3, 1 - days_ahead / 22)
        sold_out = False
        if route_id == "HYD-BLR" and d == date(2026, 10, 16) and "sleeper" in bus_type:
            sold_out = True  # Friday night sleepers to Bengaluru; the 08:00 seater is the alternative
        if route_id == "HYD-VJA" and d == date(2026, 10, 19) and depart >= 1080:
            sold_out = True  # Dussehra eve
        if route_id in ("HYD-BLR", "HYD-CHE", "HYD-VJA") and d == date(2026, 11, 6) and depart >= 1080:
            sold_out = True  # Diwali weekend rush
        if sold_out:
            self.planted.append({"domain": "buses", "kind": "trip_sold_out", "trip_id": trip_id, "route_id": route_id, "date": d.isoformat(), "time": fmt_time(depart)})
        seats = SLEEPER_BERTHS if "sleeper" in bus_type else SEATER_SEATS
        rows = []
        for s in seats:
            if sold_out or self.rng.random() < min(0.9, fill):
                pname, gender = names.person(self.rng)
                rows.append((trip_id, s, None, pname, self.rng.randint(18, 70), gender))
        self.conn.executemany("INSERT INTO bus_seats_taken VALUES (?,?,?,?,?,?)", rows)

    # ---------- restaurants
    def restaurants(self) -> None:
        c = self.conn
        for rid, name, area, address, max_party, dep_min, dep_pp, lunch, dinner, closed in RESTAURANTS:
            c.execute(
                "INSERT INTO restaurants VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (rid, name, area, address, max_party, dep_min, dep_pp, lunch[0], lunch[1], dinner[0], dinner[1], closed),
            )
            n = 0
            for seats, zone, count in TABLE_LAYOUTS[rid]:
                for _ in range(count):
                    n += 1
                    c.execute("INSERT INTO restaurant_tables VALUES (?,?,?,?)", (f"{rid}-T{n:02d}", rid, seats, zone))
        for d in window_dates():
            for rid, *_rest in RESTAURANTS:
                closed = _rest[-1]
                if closed is not None and d.weekday() == closed:
                    continue
                lunch, dinner = _rest[-3], _rest[-2]
                tables = c.execute("SELECT id, seats FROM restaurant_tables WHERE restaurant_id=? ORDER BY id", (rid,)).fetchall()
                for table_id, seats in tables:
                    for (open_, last), base in ((lunch, 0.25), (dinner, 0.45)):
                        fill = base * (1.5 if is_peak(d) else 1.0) * max(0.3, 1 - (d - TODAY).days / 20)
                        full = d == date(2026, 11, 8) and open_ == dinner[0]  # Diwali dinner fully booked
                        # Saturday-night 8 pm rush at Jubilee Hills: 4- and 6-seaters gone 19:30-21:00.
                        rush = rid == "tandoor-jubilee" and d == date(2026, 10, 17) and seats in (4, 6) and open_ == dinner[0]
                        t = open_
                        while t <= last:
                            if rush and t == 1170:
                                self._reserve(table_id, d, t)
                                t += SEATING_MIN
                            elif full or (not rush and self.rng.random() < min(0.85, fill)):
                                self._reserve(table_id, d, t)
                                t += SEATING_MIN
                            else:
                                t += 30
        self.planted.append({"domain": "restaurants", "kind": "service_full", "restaurant_id": "*", "date": "2026-11-08", "service": "dinner"})
        self.planted.append({"domain": "restaurants", "kind": "tables_full", "restaurant_id": "tandoor-jubilee", "date": "2026-10-17", "time": "19:30-21:00", "seats": 4})

    def _reserve(self, table_id: str, d: date, start: int, booking_id: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO table_reservations VALUES (?,?,?,?,?)",
            (table_id, d.isoformat(), start, start + SEATING_MIN, booking_id),
        )

    # ---------- customers, history, upcoming bookings
    def customers(self, n_customers: int = 200, n_returning: int = 130) -> None:
        rng = self.rng
        c = self.conn
        cinemas = [x[0] for x in CINEMAS]
        restaurants = [x[0] for x in RESTAURANTS]
        diets = [None, None, None, "vegetarian", "Jain food (no onion, no garlic)", "nut allergy", "gluten allergy", "vegan"]
        for i in range(n_customers):
            full_name, gender = names.person(rng)
            ph = names.phone(rng, self.phones)
            created = TODAY - timedelta(days=rng.randint(30, 900))
            c.execute("INSERT INTO customers VALUES (?,?,?,?)", (ph, full_name, CITY, created.isoformat()))
            if i >= n_returning:
                continue
            domains = rng.sample(["movies", "buses", "restaurants"], k=rng.choice([1, 1, 2]))
            family = [(full_name, gender, rng.randint(22, 60))]
            for _ in range(rng.randint(0, 3)):
                fam_name, fam_gender = names.person(rng)
                family.append((fam_name.split()[0] + " " + full_name.split()[-1], fam_gender, rng.randint(5, 70)))
            profile = {
                "name": full_name,
                "gender": gender,
                "domains": domains,
                "cinema": rng.choice(cinemas),
                "seat_category": rng.choice(["classic", "prime", "prime", "recliner"]),
                "language": rng.choice(["Telugu", "Hindi", "English", "Tamil"]),
                "tickets": rng.choice([1, 2, 2, 2, 3, 4]),
                "route": rng.choice([r[0] for r in ROUTES]),
                "bus_type": rng.choice(["ac_sleeper", "ac_sleeper", "nonac_sleeper", "ac_seater"]),
                "berth": rng.choice(["lower", "upper", None]),
                "passengers": [{"name": n, "gender": g, "age": a} for n, g, a in family[: rng.randint(1, len(family))]],
                "restaurant": rng.choice(restaurants),
                "party_size": rng.choice([2, 2, 4, 4, 5, 6, 8]),
                "dinner_time": rng.choice([1140, 1200, 1230, 1260]),
                "zone": rng.choice(["indoor", "indoor", "outdoor"]),
                "diet": rng.choice(diets),
            }
            self.profiles[ph] = profile
            self._history(ph, profile)

    def _history(self, ph: str, p: dict) -> None:
        rng = self.rng
        cinema_names = {x[0]: f"{x[1]} {x[2]}" for x in CINEMAS}
        rest_names = {x[0]: x[1] for x in RESTAURANTS}
        released = [m for m in MOVIES if m.release < TODAY]
        for _ in range(rng.randint(2, 6)):
            dom = rng.choice(p["domains"])
            d = TODAY - timedelta(days=rng.randint(3, HISTORY_DAYS))
            status = rng.choices(["completed", "cancelled", "no_show"], weights=[85, 12, 3])[0]
            hid = new_booking_id(rng, dom, self.booking_ids)
            if dom == "movies":
                m = rng.choice(released) if rng.random() < 0.3 else None
                title = m.title if m else rng.choice(["Mahanati Returns", "Bhoomi", "Kurukshetra 2", "Jalsa Nights", "The Quiet Hour", "Bawarchi Again"])
                qty = p["tickets"] if rng.random() < 0.8 else rng.randint(1, 5)
                cat = p["seat_category"] if rng.random() < 0.85 else rng.choice(["classic", "prime", "recliner"])
                cinema = p["cinema"] if rng.random() < 0.8 else rng.choice(list(cinema_names))
                slot = rng.choice(SHOW_SLOTS)
                details = {"cinema_id": cinema, "movie": title, "language": p["language"], "time": fmt_time(slot), "tickets": qty, "seat_category": cat}
                summary = f"{qty} {cat} ticket(s) for {title} at {cinema_names[cinema]}, {d.isoformat()} {fmt_time(slot)}"
            elif dom == "buses":
                route = p["route"] if rng.random() < 0.8 else rng.choice([r[0] for r in ROUTES])
                pax = p["passengers"] if rng.random() < 0.8 else p["passengers"][:1]
                bus_type = p["bus_type"] if rng.random() < 0.85 else rng.choice(["ac_sleeper", "nonac_sleeper", "ac_seater"])
                details = {"route_id": route, "bus_type": bus_type, "berth": p["berth"], "passengers": pax}
                o, dd = route.split("-")
                summary = f"{len(pax)} {bus_type.replace('_', ' ')} seat(s) {o}->{dd} on {d.isoformat()} for {', '.join(x['name'] for x in pax)}"
            else:
                rest = p["restaurant"] if rng.random() < 0.85 else rng.choice(list(rest_names))
                party = p["party_size"] if rng.random() < 0.8 else rng.randint(2, 6)
                t = p["dinner_time"] if rng.random() < 0.75 else rng.choice([750, 780, 1170, 1290])
                details = {"restaurant_id": rest, "party_size": party, "time": fmt_time(t), "zone": p["zone"], "note": p["diet"]}
                summary = f"Table for {party} at {rest_names[rest]}, {d.isoformat()} {fmt_time(t)}" + (f" ({p['diet']})" if p["diet"] else "")
            self.conn.execute(
                "INSERT INTO booking_history VALUES (?,?,?,?,?,?,?)",
                (hid, dom, ph, d.isoformat(), summary, json.dumps(details), status),
            )

    def upcoming(self, n: int = 70) -> None:
        """Future confirmed bookings for returning customers (targets for change/cancel calls)."""
        rng = self.rng
        c = self.conn
        phones = sorted(self.profiles)
        rng.shuffle(phones)
        made = 0
        created_at = (NOW - timedelta(days=3)).isoformat(timespec="minutes")
        for ph in phones:
            if made >= n:
                break
            p = self.profiles[ph]
            dom = p["domains"][0]
            # Mostly near-term; a few today to exercise cut-off rules.
            d = TODAY + timedelta(days=rng.choice([0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 9, 12, 15]))
            bid = None
            if dom == "movies":
                bid = self._upcoming_movie(ph, p, d, created_at)
            elif dom == "buses":
                bid = self._upcoming_bus(ph, p, d, created_at)
            else:
                bid = self._upcoming_table(ph, p, d, created_at)
            if bid:
                made += 1

    def _upcoming_movie(self, ph: str, p: dict, d: date, created_at: str) -> str | None:
        c = self.conn
        shows = c.execute(
            """SELECT s.id, s.screen_id, s.start_min, s.price_classic, s.price_prime, s.price_recliner
               FROM shows s JOIN screens sc ON sc.id = s.screen_id
               WHERE s.date = ? AND sc.cinema_id = ? ORDER BY s.start_min""",
            (d.isoformat(), p["cinema"]),
        ).fetchall()
        now_min = NOW.hour * 60 + NOW.minute
        shows = [s for s in shows if d > TODAY or s[2] > now_min]
        if not shows:
            return None
        show_id, screen_id, start, *prices = self.rng.choice(shows)
        cat = p["seat_category"]
        rows, per_row, cats = self.screen_layout[screen_id]
        taken = {r[0] for r in c.execute("SELECT seat FROM show_seats_taken WHERE show_id=?", (show_id,))}
        free = [s for s in cinema_seats(cats[cat], per_row) if s not in taken]
        seats = pick_cinema_seats(free, p["tickets"], per_row)
        if not seats:
            return None
        bid = new_booking_id(self.rng, "movies", self.booking_ids)
        price = dict(zip(("classic", "prime", "recliner"), prices))[cat]
        c.execute(
            """INSERT INTO bookings (id, domain, phone, name, status, show_id, seat_category, quantity, amount,
               created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (bid, "movies", ph, p["name"], "confirmed", show_id, cat, len(seats), price * len(seats), created_at, created_at),
        )
        c.executemany("INSERT INTO show_seats_taken VALUES (?,?,?)", [(show_id, s, bid) for s in seats])
        return bid

    def _upcoming_bus(self, ph: str, p: dict, d: date, created_at: str) -> str | None:
        c = self.conn
        trips = c.execute(
            "SELECT id, bus_type, fare FROM bus_trips WHERE route_id=? AND date=? ORDER BY depart_min",
            (p["route"], d.isoformat()),
        ).fetchall()
        preferred = [t for t in trips if t[1] == p["bus_type"]] or trips
        for trip_id, bus_type, fare in preferred:
            taken = {r[0] for r in c.execute("SELECT seat FROM bus_seats_taken WHERE trip_id=?", (trip_id,))}
            layout = SLEEPER_BERTHS if "sleeper" in bus_type else SEATER_SEATS
            free = [s for s in layout if s not in taken]
            pax = p["passengers"]
            seats = pick_bus_seats(free, len(pax), bus_type, p["berth"])
            if not seats:
                continue
            bid = new_booking_id(self.rng, "buses", self.booking_ids)
            c.execute(
                """INSERT INTO bookings (id, domain, phone, name, status, trip_id, quantity, amount, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (bid, "buses", ph, p["name"], "confirmed", trip_id, len(seats), fare * len(seats), created_at, created_at),
            )
            c.executemany(
                "INSERT INTO bus_seats_taken VALUES (?,?,?,?,?,?)",
                [(trip_id, s, bid, x["name"], x["age"], x["gender"]) for s, x in zip(seats, pax)],
            )
            return bid
        return None

    def _upcoming_table(self, ph: str, p: dict, d: date, created_at: str) -> str | None:
        c = self.conn
        rid = p["restaurant"]
        rest = c.execute("SELECT closed_weekday, deposit_min_party, deposit_per_guest FROM restaurants WHERE id=?", (rid,)).fetchone()
        if rest[0] is not None and d.weekday() == rest[0]:
            return None
        party, start = p["party_size"], p["dinner_time"]
        tables = c.execute(
            "SELECT id FROM restaurant_tables WHERE restaurant_id=? AND seats >= ? AND seats <= ? ORDER BY seats, id",
            (rid, party, party + 2),
        ).fetchall()
        for (table_id,) in tables:
            clash = c.execute(
                "SELECT 1 FROM table_reservations WHERE table_id=? AND date=? AND start_min < ? AND end_min > ?",
                (table_id, d.isoformat(), start + SEATING_MIN, start),
            ).fetchone()
            if clash:
                continue
            bid = new_booking_id(self.rng, "restaurants", self.booking_ids)
            deposit = party * rest[2] if party >= rest[1] else 0
            c.execute(
                """INSERT INTO bookings (id, domain, phone, name, status, restaurant_id, res_date, res_start_min, party_size,
                   quantity, amount, special_request, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (bid, "restaurants", ph, p["name"], "confirmed", rid, d.isoformat(), start, party, party, deposit, p["diet"], created_at, created_at),
            )
            self._reserve(table_id, d, start, bid)
            return bid
        return None

    def meta(self) -> None:
        rows = {
            "today": TODAY.isoformat(),
            "now": NOW.isoformat(timespec="minutes"),
            "seed": str(SEED),
            "built_at": datetime(2026, 10, 15).isoformat(),  # fixed so the file is byte-reproducible
            "planted": json.dumps(self.planted),
            "profiles": json.dumps(self.profiles, sort_keys=True),
        }
        self.conn.executemany("INSERT INTO meta VALUES (?,?)", sorted(rows.items()))


def build_world(out: str | Path = WORLD_DB) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    conn = sqlite3.connect(out)
    try:
        conn.executescript(SCHEMA.read_text())
        b = _Builder(conn)
        b.movies()
        b.buses()
        b.restaurants()
        b.customers()
        b.upcoming()
        b.meta()
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()
    return out

"""Access to the practice world database.

Every simulated call runs on its own in-memory copy of the pristine world (`World.fresh()`),
so calls are independent and the final booking state can be compared with the gold state.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastlane.config import WORLD_DB

TABLES = [
    "customers", "cinemas", "screens", "movies", "shows", "show_seats_taken",
    "bus_routes", "bus_trips", "bus_seats_taken", "restaurants", "restaurant_tables",
    "table_reservations", "bookings", "booking_history", "sms_log", "handoffs",
]


def fmt_time(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


@lru_cache(maxsize=4)
def _pristine_bytes(path: str, mtime: float) -> bytes:
    src = sqlite3.connect(path)
    try:
        return src.serialize()
    finally:
        src.close()


class World:
    def __init__(self, conn: sqlite3.Connection):
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        self.conn = conn

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def open(cls, path: str | Path = WORLD_DB) -> "World":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `fastlane world build` first")
        return cls(sqlite3.connect(path))

    @classmethod
    def fresh(cls, path: str | Path = WORLD_DB) -> "World":
        """An independent in-memory copy of the pristine world file."""
        path = Path(path)
        if not path.exists():
            from fastlane.world.seed import build_world

            build_world(path)
        conn = sqlite3.connect(":memory:")
        conn.deserialize(_pristine_bytes(str(path), path.stat().st_mtime))
        return cls(conn)

    def snapshot(self) -> bytes:
        return self.conn.serialize()

    def restore(self, data: bytes) -> None:
        self.conn.deserialize(data)
        self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "World":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --------------------------------------------------------------- queries
    def q(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def meta(self, key: str) -> str:
        row = self.one("SELECT value FROM meta WHERE key=?", (key,))
        if row is None:
            raise KeyError(key)
        return row["value"]

    @property
    def today(self) -> date:
        return date.fromisoformat(self.meta("today"))

    @property
    def now(self) -> datetime:
        return datetime.fromisoformat(self.meta("now"))

    def planted(self) -> list[dict]:
        return json.loads(self.meta("planted"))

    def profiles(self) -> dict[str, dict]:
        """Latent preferences of returning customers. Ground truth for scenario writing only;
        the gateway never exposes this table to the agent."""
        return json.loads(self.meta("profiles"))

    def table_counts(self) -> dict[str, int]:
        return {t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}

    # -------------------------------------------------------- booking views
    def describe_booking(self, booking_id: str) -> dict[str, Any] | None:
        """Canonical, domain-specific view of a booking (used for read-backs and gold matching)."""
        b = self.one("SELECT * FROM bookings WHERE id=?", (booking_id,))
        if b is None:
            return None
        out: dict[str, Any] = {
            "booking_id": b["id"],
            "domain": b["domain"],
            "status": b["status"],
            "name": b["name"],
            "phone": b["phone"],
            "quantity": b["quantity"],
            "amount": b["amount"],
        }
        if b["domain"] == "movies":
            s = self.one(
                """SELECT s.date, s.start_min, s.movie_id, m.title, sc.cinema_id, c.name AS cinema_name, c.area
                   FROM shows s JOIN movies m ON m.id=s.movie_id JOIN screens sc ON sc.id=s.screen_id
                   JOIN cinemas c ON c.id=sc.cinema_id WHERE s.id=?""",
                (b["show_id"],),
            )
            seats = [r["seat"] for r in self.q("SELECT seat FROM show_seats_taken WHERE booking_id=? ORDER BY seat", (booking_id,))]
            out.update(
                show_id=b["show_id"], movie_id=s["movie_id"], movie=s["title"], cinema_id=s["cinema_id"],
                cinema=f"{s['cinema_name']} {s['area']}", date=s["date"], time=fmt_time(s["start_min"]),
                seat_category=b["seat_category"], seats=seats,
            )
        elif b["domain"] == "buses":
            t = self.one(
                """SELECT t.date, t.depart_min, t.bus_type, t.route_id, r.origin, r.destination, r.boarding_point
                   FROM bus_trips t JOIN bus_routes r ON r.id=t.route_id WHERE t.id=?""",
                (b["trip_id"],),
            )
            pax = self.q(
                """SELECT seat, passenger_name AS name, passenger_age AS age, passenger_gender AS gender
                   FROM bus_seats_taken WHERE booking_id=? ORDER BY seat""",
                (booking_id,),
            )
            out.update(
                trip_id=b["trip_id"], route_id=t["route_id"], origin=t["origin"], destination=t["destination"],
                date=t["date"], time=fmt_time(t["depart_min"]), bus_type=t["bus_type"],
                boarding_point=t["boarding_point"], seats=[p["seat"] for p in pax], passengers=pax,
            )
        else:
            r = self.one("SELECT name FROM restaurants WHERE id=?", (b["restaurant_id"],))
            tables = [x["table_id"] for x in self.q("SELECT table_id FROM table_reservations WHERE booking_id=?", (booking_id,))]
            out.update(
                restaurant_id=b["restaurant_id"], restaurant=r["name"], date=b["res_date"],
                time=fmt_time(b["res_start_min"]), party_size=b["party_size"], tables=tables,
                special_request=b["special_request"],
            )
        return out

    def bookings_for_phone(self, phone: str) -> list[dict[str, Any]]:
        ids = [r["id"] for r in self.q("SELECT id FROM bookings WHERE phone=? ORDER BY created_at, id", (phone,))]
        return [self.describe_booking(i) for i in ids]  # type: ignore[misc]

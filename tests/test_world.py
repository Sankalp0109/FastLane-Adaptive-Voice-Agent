import hashlib

from fastlane.world.db import World
from fastlane.world.seating import pick_bus_seats, pick_cinema_seats
from fastlane.world.seed import build_world


def _digest(path) -> str:
    with World.open(path) as w:
        return hashlib.sha256("\n".join(w.conn.iterdump()).encode()).hexdigest()


def test_build_is_deterministic(tmp_path, world_path):
    again = build_world(tmp_path / "again.db")
    assert _digest(again) == _digest(world_path)


def test_catalogue_matches_proposal(world):
    counts = world.table_counts()
    assert counts["cinemas"] == 4
    assert counts["bus_routes"] == 6
    assert counts["restaurants"] == 3
    assert 400 <= counts["booking_history"] <= 600  # "about 500 synthetic past bookings"
    per_day = world.q(
        "SELECT sc.cinema_id, s.date, COUNT(*) n FROM shows s JOIN screens sc ON sc.id=s.screen_id GROUP BY 1, 2"
    )
    assert {r["n"] for r in per_day} == {12}


def test_no_film_shown_before_release(world):
    bad = world.q("SELECT s.id FROM shows s JOIN movies m ON m.id=s.movie_id WHERE s.date < m.release_date")
    assert bad == []


def test_planted_movie_sell_out(world):
    rows = world.q(
        """SELECT s.id, (SELECT COUNT(*) FROM show_seats_taken t WHERE t.show_id=s.id) AS taken,
                  length(sc.rows) * sc.seats_per_row AS capacity
           FROM shows s JOIN screens sc ON sc.id=s.screen_id
           WHERE s.movie_id='kaala-samudram' AND s.date='2026-10-16' AND s.start_min=1230"""
    )
    assert rows and all(r["taken"] == r["capacity"] for r in rows)


def test_planted_bus_sell_out_leaves_an_alternative(world):
    rows = world.q(
        """SELECT t.bus_type, t.capacity - (SELECT COUNT(*) FROM bus_seats_taken x WHERE x.trip_id=t.id) AS free
           FROM bus_trips t WHERE t.route_id='HYD-BLR' AND t.date='2026-10-16'"""
    )
    sleepers = [r["free"] for r in rows if "sleeper" in r["bus_type"]]
    seaters = [r["free"] for r in rows if "seater" in r["bus_type"]]
    assert sleepers and all(f == 0 for f in sleepers)
    assert seaters and all(f > 0 for f in seaters)


def test_upcoming_bookings_have_items(world):
    for b in world.q("SELECT id FROM bookings"):
        view = world.describe_booking(b["id"])
        if view["domain"] == "restaurants":
            assert len(view["tables"]) == 1
        else:
            assert len(view["seats"]) == view["quantity"]


def test_fresh_copies_are_independent(world_path):
    a, b = World.fresh(world_path), World.fresh(world_path)
    a.conn.execute("DELETE FROM table_reservations")
    a.conn.execute("DELETE FROM bookings WHERE domain='restaurants'")
    assert b.one("SELECT COUNT(*) n FROM table_reservations")["n"] > 0


def test_snapshot_restore(world):
    snap = world.snapshot()
    world.conn.execute("INSERT INTO sms_log (phone, body, sent_at) VALUES ('9999999999', 'x', 'now')")
    assert world.one("SELECT COUNT(*) n FROM sms_log")["n"] == 1
    world.restore(snap)
    assert world.one("SELECT COUNT(*) n FROM sms_log")["n"] == 0


def test_pick_cinema_seats_prefers_contiguous_centre():
    free = [f"F{i}" for i in range(1, 13)]
    assert pick_cinema_seats(free, 2, 12) == ["F6", "F7"]
    assert pick_cinema_seats(["F1", "F3", "G5"], 2, 12) is not None  # split when no run exists
    assert pick_cinema_seats(["F1"], 2, 12) is None


def test_pick_bus_seats_preferences():
    free = ["L1", "L2", "U1", "U2"]
    assert pick_bus_seats(free, 2, "ac_sleeper", "upper") == ["U1", "U2"]
    assert pick_bus_seats(free, 2, "ac_sleeper") == ["L1", "L2"]
    assert pick_bus_seats(["2", "3", "4", "5"], 2, "ac_seater", "window") == ["4", "5"]

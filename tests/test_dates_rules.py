from datetime import date, datetime

import pytest

from fastlane.gateway import rules
from fastlane.gateway.dates import parse_time, resolve_date

TODAY = date(2026, 10, 15)  # Thursday


@pytest.mark.parametrize("expr,expected,ambiguous", [
    ("today", "2026-10-15", False),
    ("tonight", "2026-10-15", False),
    ("tomorrow", "2026-10-16", False),
    ("day after tomorrow", "2026-10-17", False),
    ("Friday", "2026-10-16", False),
    ("this Friday", "2026-10-16", False),
    ("next Friday", "2026-10-23", True),
    ("next Monday", "2026-10-19", False),
    ("Thursday", "2026-10-22", False),
    ("this weekend", "2026-10-17", True),
    ("in 3 days", "2026-10-18", False),
    ("a week from today", "2026-10-22", False),
    ("23rd October", "2026-10-23", False),
    ("Friday 23rd October 2026", "2026-10-23", False),  # explicit date wins over the weekday word
    ("next Friday, 23rd October", "2026-10-23", False),
    ("october 23", "2026-10-23", False),
    ("Friday the 23rd", "2026-10-23", False),  # day of month beats the weekday word
    ("Tuesday the 10th", "2026-11-10", False),
    ("twenty third", "2026-10-23", False),
    ("23/10", "2026-10-23", False),
    ("the 5th", "2026-11-05", False),
    ("Diwali", "2026-11-08", False),
    ("Deepavali", "2026-11-08", False),
    ("day before Diwali", "2026-11-07", False),
    ("dasara", "2026-10-20", False),
])
def test_resolve_date(expr, expected, ambiguous):
    r = resolve_date(expr, TODAY)
    assert r.date.isoformat() == expected
    assert r.ambiguous is ambiguous


def test_next_week_asks_which_day():
    r = resolve_date("next week", TODAY)
    assert r.date is None and r.ambiguous and len(r.alternatives) == 7


def test_outside_window():
    assert resolve_date("Christmas", TODAY).in_window is False


@pytest.mark.parametrize("text,minute", [
    ("8 pm", 1200), ("8:30 pm", 1230), ("20:30", 1230), ("half past eight", 1230), ("eight thirty", 1230),
    ("10 am", 600), ("08:00", 480), ("7.30", 1170), ("9 in the morning", 540),
])
def test_parse_time(text, minute):
    assert parse_time(text) == (minute, minute)


def test_parse_part_of_day():
    a, b = parse_time("evening")
    assert a < 1230 < b


def test_bus_refund_table():
    now = datetime(2026, 10, 15, 11, 0)
    assert rules.bus_refund_percent(datetime(2026, 10, 19, 11, 0), now) == 90  # 96 h
    assert rules.bus_refund_percent(datetime(2026, 10, 16, 21, 0), now) == 75  # 34 h
    assert rules.bus_refund_percent(datetime(2026, 10, 16, 1, 0), now) == 50   # 14 h
    assert rules.bus_refund_percent(datetime(2026, 10, 15, 21, 0), now) == 25  # 10 h
    assert rules.bus_refund_percent(datetime(2026, 10, 15, 14, 0), now) is None  # 3 h
    # Booked under the old policy (before 1 Sep 2026): 85% beyond 48 h.
    assert rules.bus_refund_percent(datetime(2026, 10, 19, 11, 0), now, booked_on=date(2026, 8, 20)) == 85


def test_movie_change_cutoff():
    now = datetime(2026, 10, 15, 11, 0)
    assert rules.check_movie_change(datetime(2026, 10, 15, 13, 30), now)
    assert not rules.check_movie_change(datetime(2026, 10, 15, 17, 0), now)


@pytest.mark.parametrize("text,minute", [("10:00", 600), ("11:30", 690), ("22:30", 1350), ("08:00", 480), ("13:30", 810), ("7:30", 1170), ("10", 1320)])
def test_24h_tool_times_are_literal(text, minute):
    assert parse_time(text) == (minute, minute)

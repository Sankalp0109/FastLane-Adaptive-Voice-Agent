"""Seat layout helpers shared by the seeder and the gateway."""

from __future__ import annotations

import re
from collections import defaultdict

SLEEPER_BERTHS = [f"L{i}" for i in range(1, 16)] + [f"U{i}" for i in range(1, 16)]
SEATER_SEATS = [str(i) for i in range(1, 41)]

_SEAT_RE = re.compile(r"^([A-Z])(\d+)$")


def cinema_seats(rows: str, seats_per_row: int) -> list[str]:
    return [f"{r}{n}" for r in rows for n in range(1, seats_per_row + 1)]


def split_seat(seat: str) -> tuple[str, int]:
    m = _SEAT_RE.match(seat)
    if not m:
        raise ValueError(f"bad cinema seat {seat!r}")
    return m.group(1), int(m.group(2))


def pick_cinema_seats(free: list[str], n: int, seats_per_row: int) -> list[str] | None:
    """Pick n seats, preferring one contiguous run near the middle of a row.

    Rows are tried back-to-front (rear rows are the better view within a category).
    Falls back to the n free seats closest to the centre, possibly split across rows.
    Returns None when fewer than n seats are free.
    """
    if n <= 0 or len(free) < n:
        return None
    by_row: dict[str, list[int]] = defaultdict(list)
    for seat in free:
        row, num = split_seat(seat)
        by_row[row].append(num)
    centre = (seats_per_row + 1) / 2
    best: tuple[float, str, int] | None = None
    for row in sorted(by_row, reverse=True):
        nums = sorted(by_row[row])
        taken = set(nums)
        for start in nums:
            if all(start + k in taken for k in range(n)):
                off_centre = abs(start + (n - 1) / 2 - centre)
                if best is None or off_centre < best[0]:
                    best = (off_centre, row, start)
        if best is not None and best[1] == row:
            break  # rearmost row with a contiguous run wins
    if best is not None:
        _, row, start = best
        return [f"{row}{start + k}" for k in range(n)]
    ranked = sorted(free, key=lambda s: (abs(split_seat(s)[1] - centre), -ord(split_seat(s)[0])))
    return sorted(ranked[:n], key=lambda s: (split_seat(s)[0], split_seat(s)[1]))


def is_window_seat(seat: str) -> bool:
    """Seater buses are 2+2: seats 1,4,5,8,... are windows."""
    return int(seat) % 4 in (0, 1)


def pick_bus_seats(free: list[str], n: int, bus_type: str, preference: str | None = None) -> list[str] | None:
    """Pick n berths/seats. preference: 'lower' | 'upper' | 'window' | None."""
    if n <= 0 or len(free) < n:
        return None
    if "sleeper" in bus_type:
        order = {s: i for i, s in enumerate(SLEEPER_BERTHS)}
        if preference == "upper":
            key = lambda s: (s[0] != "U", order[s])  # noqa: E731
        else:
            key = lambda s: (s[0] != "L", order[s])  # noqa: E731
    else:
        if preference == "window":
            key = lambda s: (not is_window_seat(s), int(s))  # noqa: E731
        else:
            key = lambda s: int(s)  # noqa: E731
    return sorted(free, key=key)[:n]

"""Fuzzy matching of spoken names (films, cinemas, cities, branches) onto world ids.

Speech recognisers mishear Indian names ("Kala Samudram", "Galaxy Kukatpalli", "Bangalore"),
so every name argument goes through here before it reaches SQL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from fastlane.world.db import World

CITY_ALIASES = {
    "Hyderabad": ["hyderabad", "hyd", "secunderabad", "hydrabad", "haiderabad"],
    "Bengaluru": ["bengaluru", "bangalore", "blr", "banglore", "bengalooru", "bangaluru"],
    "Vijayawada": ["vijayawada", "bezawada", "bza", "vja", "vijaywada", "vijayvada"],
    "Chennai": ["chennai", "madras", "maa", "chenai"],
}

BUS_TYPE_ALIASES = {
    "ac_sleeper": ["ac sleeper", "a c sleeper", "air conditioned sleeper", "volvo sleeper", "ac berth"],
    "nonac_sleeper": ["non ac sleeper", "nonac sleeper", "non a c sleeper", "ordinary sleeper"],
    "ac_seater": ["ac seater", "a c seater", "ac seat", "semi sleeper", "volvo", "ac bus"],
    "nonac_seater": ["non ac seater", "non ac seat", "ordinary", "non ac bus"],
}

SEAT_CATEGORY_ALIASES = {
    "classic": ["classic", "normal", "regular", "front", "standard", "silver"],
    "prime": ["prime", "premium", "middle", "executive", "gold", "balcony"],
    "recliner": ["recliner", "recliners", "luxury", "sofa", "lounge", "platinum", "vip"],
}


def norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _score(query: str, alias: str) -> float:
    if not query or not alias:
        return 0.0
    if query == alias:
        return 1.0
    # Whole-word containment ("the kukatpally one" contains "kukatpally").
    if re.search(rf"\b{re.escape(alias)}\b", query) or re.search(rf"\b{re.escape(query)}\b", alias):
        shorter, longer = sorted((query, alias), key=len)
        return 0.85 + 0.15 * len(shorter) / len(longer)
    ratio = SequenceMatcher(None, query, alias).ratio()
    # Token-level best match helps with extra words ("kaala samudram movie").
    tokens = query.split()
    alias_len = len(alias.split())
    best_window = 0.0
    for i in range(len(tokens)):
        window = " ".join(tokens[i : i + alias_len])
        best_window = max(best_window, SequenceMatcher(None, window, alias).ratio())
    return max(ratio, best_window * 0.95)


@dataclass
class Match:
    id: str
    label: str
    score: float


def best_matches(query: str, candidates: dict[str, tuple[str, list[str]]], cutoff: float = 0.72) -> list[Match]:
    """candidates: id -> (display label, aliases). Returns matches above cutoff, best first."""
    q = norm(query)
    out = []
    for cid, (label, aliases) in candidates.items():
        score = max(_score(q, norm(a)) for a in [label, *aliases])
        if score >= cutoff:
            out.append(Match(cid, label, round(score, 3)))
    return sorted(out, key=lambda m: -m.score)


def resolve_one(query: str, candidates: dict[str, tuple[str, list[str]]], cutoff: float = 0.72) -> tuple[Match | None, list[Match]]:
    """Return (unique best match, ambiguous alternatives). If two candidates score within 0.05
    of each other the result is ambiguous and the caller has to be asked."""
    matches = best_matches(query, candidates, cutoff)
    if not matches:
        return None, []
    if len(matches) > 1 and matches[0].score - matches[1].score < 0.05:
        return None, matches[:4]
    return matches[0], []


class Catalogue:
    """Name -> id lookups built once per world."""

    def __init__(self, world: World):
        self.movies = {
            r["id"]: (r["title"], [r["id"].replace("-", " ")])
            for r in world.q("SELECT id, title FROM movies")
        }
        self.cinemas = {
            r["id"]: (f"{r['name']} {r['area']}", [r["name"], r["area"], r["name"].split()[0]])
            for r in world.q("SELECT id, name, area FROM cinemas")
        }
        self.restaurants = {
            r["id"]: (r["name"], [r["area"], r["area"].replace(" ", ""), r["name"].replace("Tandoor Tales ", "")])
            for r in world.q("SELECT id, name, area FROM restaurants")
        }
        self.restaurants.get("tandoor-hitech", ("", []))[1].extend(["hi tech city", "hitec city", "hitech", "madhapur", "cyber towers"])
        self.cities = {city: (city, aliases) for city, aliases in CITY_ALIASES.items()}
        self.bus_types = {k: (k.replace("_", " "), v) for k, v in BUS_TYPE_ALIASES.items()}
        self.seat_categories = {k: (k, v) for k, v in SEAT_CATEGORY_ALIASES.items()}

    def movie(self, q: str):
        return resolve_one(q, self.movies)

    def cinema(self, q: str):
        return resolve_one(q, self.cinemas)

    def restaurant(self, q: str):
        return resolve_one(q, self.restaurants)

    def city(self, q: str):
        return resolve_one(q, self.cities)

    def bus_type(self, q: str):
        q = norm(q)
        # Longest alias first so "non ac sleeper" is not read as "ac sleeper".
        for key, aliases in sorted(
            ((k, a) for k, (_, al) in self.bus_types.items() for a in al), key=lambda kv: -len(kv[1])
        ):
            if re.search(rf"\b{re.escape(aliases)}\b", q):
                return Match(key, key, 1.0), []
        if q in self.bus_types:
            return Match(q, q, 1.0), []
        return resolve_one(q.replace(" ", "_"), {k: (k, []) for k in self.bus_types}, cutoff=0.8)

    def seat_category(self, q: str):
        return resolve_one(q, self.seat_categories, cutoff=0.8)

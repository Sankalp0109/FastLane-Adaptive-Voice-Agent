"""Early lookup: start the policy search on the partial transcript while the caller is still talking.

The speech layer calls `on_partial(text)` as interim transcripts arrive and `on_final(text)` when the
caller stops. If the final transcript says essentially the same thing as the partial the lookup was
started on, the prefetched result is reused and its search time is hidden behind the caller's speech.
Otherwise the lookup was wasted and is counted as such (Table 7: "time saved, wasted lookups").

`PrefetchingSearcher` is what the gateway's `search_policy` tool talks to: when the model's query
is about the same thing as the prefetched transcript, it returns the cached result instantly.
Variant A6 constructs it with enabled=False.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from fastlane.retrieval.index import PolicyIndex, terms

_POLICY_CUES = {
    "refund", "cancel", "cancellation", "fee", "deposit", "policy", "allowed", "allow", "can", "luggage", "pet", "dog",
    "child", "children", "kid", "kids", "baby", "allergy", "allergic", "nut", "nuts", "gluten", "jain", "vegan", "halal",
    "parking", "price", "cost", "fare", "how", "much", "rules", "rule", "change", "late", "miss", "alcohol", "smoke",
    "wheelchair", "certificate", "menu", "dish", "food", "open", "closed", "diwali", "dussehra", "festival",
}


def _overlap(a: str, b: str) -> float:
    """Share of a's content terms that also occur in b."""
    ta, tb = set(terms(a)), set(terms(b))
    return len(ta & tb) / len(ta) if ta else 0.0


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(terms(a)), set(terms(b))
    return len(ta & tb) / len(ta | tb) if ta | tb else 1.0


@dataclass
class Prefetch:
    text: str
    future: Future
    started: float
    result: dict | None = None
    used: bool = False


@dataclass
class LookupStats:
    started: int = 0
    reused: int = 0
    wasted: int = 0
    saved_ms: float = 0.0
    cache_hits: int = 0
    log: list[dict] = field(default_factory=list)


class EarlyLookup:
    def __init__(self, index: PolicyIndex, enabled: bool = True, min_words: int = 4, reuse_jaccard: float = 0.6):
        self.index = index
        self.enabled = enabled
        self.min_words = min_words
        self.reuse_jaccard = reuse_jaccard
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.current: Prefetch | None = None
        self.ready: Prefetch | None = None  # prefetch accepted for the current turn
        self.stats = LookupStats()

    @staticmethod
    def looks_like_policy_question(text: str) -> bool:
        words = set(text.lower().replace("?", " ").split())
        return len(words & _POLICY_CUES) >= 2 or "?" in text

    def on_partial(self, text: str) -> None:
        if not self.enabled or len(text.split()) < self.min_words or not self.looks_like_policy_question(text):
            return
        if self.current is not None and _jaccard(self.current.text, text) >= 0.8:
            return  # still the same question; keep the running lookup
        self.current = Prefetch(text, self.pool.submit(self.index.search, text), time.perf_counter())
        self.stats.started += 1

    def on_final(self, text: str) -> Prefetch | None:
        """Called when the caller stops speaking. Returns the reusable prefetch, if any."""
        self.ready = None
        p, self.current = self.current, None
        if p is None:
            return None
        if _jaccard(p.text, text) >= self.reuse_jaccard:
            p.result = p.future.result()
            # Time hidden behind speech: the search ran before the caller finished.
            self.stats.saved_ms += p.result.get("search_ms", 0.0)
            self.stats.reused += 1
            self.ready = p
            self.stats.log.append({"partial": p.text, "final": text, "reused": True})
            return p
        self.stats.wasted += 1
        self.stats.log.append({"partial": p.text, "final": text, "reused": False})
        return None

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


class PrefetchingSearcher:
    """The gateway's policy searcher: answers from the turn's prefetch when the query matches."""

    def __init__(self, index: PolicyIndex, early: EarlyLookup | None = None, min_overlap: float = 0.5):
        self.index = index
        self.early = early
        self.min_overlap = min_overlap

    def search(self, query: str, domain: str | None = None, k: int = 3) -> dict:
        p = self.early.ready if self.early is not None and self.early.enabled else None
        if p is not None and p.result is not None and _overlap(query, p.text) >= self.min_overlap:
            p.used = True
            self.early.stats.cache_hits += 1
            return {**p.result, "prefetched": True, "search_ms": 0.0}
        return self.index.search(query, domain, k)

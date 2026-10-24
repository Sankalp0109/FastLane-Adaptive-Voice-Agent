"""Policy retrieval: BM25 over heading-level chunks of the policy documents.

* Superseded policy versions are demoted unless the question is explicitly about the old rules
  or about a booking made before the new policy took effect.
* `covered` is False when nothing in the corpus answers the question well enough, so the agent
  says "that is not covered" instead of guessing; `confidence` feeds the escalation trigger
  "unconfident policy match".
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from fastlane.config import DOCS_DIR

STOPWORDS = set("""
a an and are as at be by can could do does for from had has have how i if in into is it its me my of on or our
please should so that the their them then there these they this to under up was we were what when where which
who will with would you your am any tell know want about get much many there's i'm it's also just like need
""".split())

# Light query expansion for words callers use that documents phrase differently.
SYNONYMS = {
    "refund": ["refunded", "refund"], "money": ["refund", "amount"], "back": ["refund"],
    "cancel": ["cancellation", "cancelled", "cancel"], "cancelling": ["cancellation", "cancel"],
    "change": ["changing", "change", "changed"], "reschedule": ["change", "date"], "postpone": ["change", "date"],
    "kids": ["children", "child"], "kid": ["children", "child"], "baby": ["children", "infants", "lap"],
    "veg": ["vegetarian"], "jain": ["jain"], "allergy": ["allergens", "allergy", "allergies"], "allergic": ["allergens", "allergy"],
    "nut": ["nuts", "nut"], "nuts": ["nuts", "nut", "cashew"], "gluten": ["gluten", "wheat"], "dairy": ["dairy", "milk"],
    "luggage": ["luggage", "suitcase", "bag"], "bags": ["luggage", "bag", "bags"], "dog": ["pets", "dogs"], "pet": ["pets"],
    "late": ["late", "missed"], "miss": ["missed", "late"], "missed": ["missed", "late"], "deposit": ["deposit"], "advance": ["deposit"], "parking": ["parking"],
    "price": ["prices", "fare", "fares", "cost"], "cost": ["prices", "fare", "price"], "ticket": ["ticket", "tickets"],
    "fare": ["fare", "fares", "prices"], "timing": ["timings", "times", "departs"], "timings": ["timings", "times"],
    "diwali": ["diwali"], "dasara": ["dussehra"], "dussera": ["dussehra"], "deepavali": ["diwali"],
    "bangalore": ["bengaluru"], "vizag": ["visakhapatnam"], "bezawada": ["vijayawada"], "madras": ["chennai"],
    "alcohol": ["alcohol", "drinks"], "booze": ["alcohol"], "drinks": ["drinks", "alcohol"],
    "wheelchair": ["wheelchair", "accessibility", "step-free"], "disabled": ["wheelchair", "accessibility"],
    "id": ["id", "identity", "photo"], "aadhaar": ["aadhaar", "id"], "age": ["age", "certificate", "adults"],
}
OLD_POLICY_HINTS = re.compile(r"\b(old|older|previous|earlier|before (?:1(?:st)? )?september|august|2025|booked (?:in|last) (?:july|august|june))\b")


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def _stem(w: str) -> str:
    for suffix in ("ations", "ation", "ings", "ing", "ies", "es", "ed", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return w


def terms(text: str, expand: bool = False) -> list[str]:
    toks = tokenize(text)
    if expand:
        extra = [s for t in toks for s in SYNONYMS.get(t, [])]
        toks = toks + extra
    return [_stem(t) for t in toks]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    domain: str
    meta: dict = field(default_factory=dict)

    @property
    def superseded(self) -> bool:
        return "superseded_by" in self.meta


def _front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    _, fm, body = text.split("---", 2)
    meta = {}
    for line in fm.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        meta, body = _front_matter(path.read_text())
        doc_id = meta.get("doc_id", path.stem)
        title = meta.get("title", path.stem)
        parts = re.split(r"^## +(.+)$", body, flags=re.M)
        intro = re.sub(r"^# .+$", "", parts[0], flags=re.M).strip()
        sections = [("Overview", intro)] if intro else []
        sections += [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]
        for n, (section, text) in enumerate(sections):
            if not text:
                continue
            chunks.append(Chunk(f"{doc_id}#{n}", doc_id, title, section, text, meta.get("domain", "general"), meta))
    return chunks


class PolicyIndex:
    """BM25 (k1=1.4, b=0.6) with title/section boosting and version handling."""

    def __init__(self, chunks: list[Chunk] | None = None, k1: float = 1.4, b: float = 0.6,
                 min_score: float = 5.0, min_coverage: float = 0.6):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.k1, self.b = k1, b
        self.min_score, self.min_coverage = min_score, min_coverage
        self.docs = []
        for c in self.chunks:
            # Title and section words count twice: they say what a chunk is about.
            self.docs.append(Counter(terms(f"{c.title} {c.section} {c.section} {c.text}")))
        self.avgdl = sum(sum(d.values()) for d in self.docs) / max(1, len(self.docs))
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _bm25(self, q: list[str], i: int) -> float:
        d = self.docs[i]
        dl = sum(d.values())
        score = 0.0
        for t in set(q):
            tf = d.get(t, 0)
            if tf:
                score += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return score

    def search(self, query: str, domain: str | None = None, k: int = 3) -> dict:
        t0 = time.perf_counter()
        q = terms(query, expand=True)
        core = set(terms(query))
        wants_old = bool(OLD_POLICY_HINTS.search(query.lower()))
        scored = []
        for i, c in enumerate(self.chunks):
            s = self._bm25(q, i)
            if s <= 0:
                continue
            if domain and c.domain not in (domain, "general"):
                s *= 0.6
            if c.superseded and not wants_old:
                s *= 0.45
            elif wants_old and c.superseded:
                s *= 1.3
            scored.append((s, i))
        scored.sort(reverse=True)
        top = scored[:k]
        best = top[0][0] if top else 0.0
        coverage = 0.0
        if top and core:
            # IDF-weighted share of the question's words (or their synonyms) found in the top chunks.
            # Words that never occur in the corpus weigh as much as a typical word.
            typical = sorted(self.idf.values())[len(self.idf) // 2]
            vocab = set().union(*(self.docs[i] for _, i in top))
            covered_w = total_w = 0.0
            for raw in set(tokenize(query)):
                variants = {_stem(raw)} | {_stem(x) for x in SYNONYMS.get(raw, [])}
                w = max(self.idf.get(v, typical) for v in variants)
                total_w += w
                covered_w += w if variants & vocab else 0.0
            coverage = covered_w / total_w if total_w else 0.0
        covered = best >= self.min_score and coverage >= self.min_coverage
        confidence = round(min(1.0, best / (self.min_score * 3)) * coverage, 3)
        results = []
        for s, i in top:
            c = self.chunks[i]
            results.append({
                "doc": c.title, "doc_id": c.doc_id, "section": c.section, "score": round(s, 2),
                "text": c.text if len(c.text) < 1800 else c.text[:1800] + " ...",
                "version": c.meta.get("version"), "effective_from": c.meta.get("effective_from"),
                "superseded": c.superseded,
            })
        out = {
            "covered": covered,
            "confidence": confidence,
            "results": results,
            "search_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        if not covered:
            out["note"] = "Nothing in our policies answers this; say it is not covered and offer staff if needed."
        if any(r["superseded"] for r in out["results"]):
            out["version_note"] = "One result is a superseded policy: it applies only to bookings made before its effective_to date."
        return out

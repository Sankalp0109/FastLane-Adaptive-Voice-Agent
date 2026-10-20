import pytest

from fastlane.config import DOCS_DIR
from fastlane.gateway import rules
from fastlane.gateway.gateway import Gateway
from fastlane.retrieval.build_docs import build_docs
from fastlane.retrieval.early_lookup import EarlyLookup, PrefetchingSearcher
from fastlane.retrieval.evaluate import evaluate
from fastlane.retrieval.index import PolicyIndex, load_chunks


@pytest.fixture(scope="module")
def index():
    return PolicyIndex()


def test_corpus_loads_with_versions():
    chunks = load_chunks()
    docs = {c.doc_id for c in chunks}
    assert len(docs) >= 20
    old = [c for c in chunks if c.doc_id == "bus_refund_policy_2025"]
    assert old and all(c.superseded for c in old)


def test_generated_docs_are_current(tmp_path):
    for path in build_docs(tmp_path):
        assert path.read_text() == (DOCS_DIR / path.name).read_text(), f"run `fastlane docs build` ({path.name} is stale)"


def test_policy_text_matches_rules_in_code():
    bus = (DOCS_DIR / "bus_refund_policy.md").read_text()
    for hours, pct in rules.BUS_REFUND_TABLE:
        assert f"{pct}%" in bus and str(hours) in bus
    old = (DOCS_DIR / "bus_refund_policy_2025.md").read_text()
    for _, pct in rules.BUS_REFUND_TABLE_OLD:
        assert f"{pct}%" in old
    movie = (DOCS_DIR / "movie_ticket_policy.md").read_text()
    assert f"{rules.MOVIE_CHANGE_CUTOFF_HOURS} hours" in movie
    assert f"Rs {rules.MOVIE_CANCEL_FEE_PER_TICKET} per ticket" in movie
    assert f"at most {rules.MOVIE_MAX_TICKETS} tickets" in movie
    rest = (DOCS_DIR / "restaurant_reservation_policy.md").read_text()
    assert f"**{rules.RESTAURANT_CHANGE_CUTOFF_HOURS} hours**" in rest
    assert f"**{rules.RESTAURANT_DEPOSIT_REFUND_HOURS} hours**" in rest


def test_retrieval_metrics_floor(index):
    r = evaluate(index)
    assert r["recall@3"] >= 0.9
    assert r["current_policy_choice"] == 1.0
    assert r["covered_when_answerable"] >= 0.85
    assert r["not_covered_detection"] >= 0.5


def test_current_policy_preferred(index):
    r = index.search("how much refund if I cancel my bus ticket 30 hours before departure")
    assert r["results"][0]["doc_id"] == "bus_refund_policy"
    r = index.search("I booked in August under the old policy, refund for cancelling 30 hours before?")
    assert r["results"][0]["doc_id"] == "bus_refund_policy_2025"
    assert "version_note" in r


def test_not_covered(index):
    r = index.search("how many calories are in the butter chicken")
    assert r["covered"] is False and "note" in r


def test_early_lookup_reuse_and_waste(index):
    early = EarlyLookup(index)
    searcher = PrefetchingSearcher(index, early)
    early.on_partial("what is the refund if I cancel")
    early.on_partial("what is the refund if I cancel my bus ticket")
    assert early.on_final("what is the refund if I cancel my bus ticket tomorrow") is not None
    hit = searcher.search("bus ticket cancellation refund")
    assert hit.get("prefetched") is True
    assert early.stats.reused == 1 and early.stats.cache_hits == 1
    early.on_partial("can I bring my dog on the bus")
    assert early.on_final("actually book two tickets for Rocket Raja") is None
    assert early.stats.wasted == 1
    early.close()


def test_early_lookup_disabled(index):
    early = EarlyLookup(index, enabled=False)
    early.on_partial("what is the refund if I cancel my bus ticket")
    assert early.on_final("what is the refund if I cancel my bus ticket") is None
    assert early.stats.started == 0
    assert "prefetched" not in PrefetchingSearcher(index, early).search("bus refund")


def test_gateway_policy_tool(world, index):
    gw = Gateway(world, "9876500001", policy=PrefetchingSearcher(index))
    gw.begin_turn("can I take my dog on the bus?")
    r = gw.call("search_policy", {"query": "pets on the bus", "domain": "buses"})
    assert r["ok"] and r["covered"] and "Pets" in r["results"][0]["text"]

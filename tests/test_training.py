"""Training-data generator, noise, splits and agreement statistics."""

import json
import random
import re
from collections import Counter

import pytest

from fastlane.config import SYSTEMS
from fastlane.models.parsing import parse_small
from fastlane.retrieval.index import PolicyIndex
from fastlane.training.build_dataset import generate, split_and_write
from fastlane.training.dialogues import Planner, run_plan
from fastlane.training.kappa import cohen_kappa
from fastlane.training.noise import asr_noise, noisy_user_content
from fastlane.training.train_lora import load_split
from fastlane.world.db import World

KINDS = ["book", "dates", "sold_out", "change_mind", "modify", "cancel", "policy", "usual", "returning_cancel",
         "red_flag", "rule_group", "rule_colleague"]


@pytest.fixture(scope="module")
def generated(world_path):
    return generate(60, world_path=world_path, progress=lambda m: None)


def test_most_generated_calls_reach_their_planned_outcome(generated):
    calls, examples = generated
    kept = sum(c["ok"] for c in calls)
    assert kept / len(calls) >= 0.9, Counter((c["kind"], c["ok"]) for c in calls)
    assert examples


def test_examples_are_valid_small_model_steps(generated):
    _, examples = generated
    for ex in examples:
        msgs = ex["messages"]
        assert msgs[0]["role"] == "system" and msgs[0]["content"].endswith('"say": ""}.')  # compact prompt
        assert msgs[-1]["role"] == "assistant"
        p = parse_small(msgs[-1]["content"])
        assert p.ok and p.route == ex["meta"]["route"]
        if p.tool:
            # The tool must be one the prompt offered (A2 has no resolve_date, A7 no history...).
            assert f"- {p.tool['name']}(" in msgs[0]["content"], (p.tool["name"], ex["meta"]["config"])
        else:
            assert p.say or p.route != "self"


def test_routing_targets_follow_the_rules(generated):
    _, examples = generated
    for ex in examples:
        m = ex["meta"]
        if m["event"] in ("red_flag", "deposit_waiver", "group_limit"):
            assert m["route"] == "staff"
        if m["event"] == "change" and m["step"] == 0:
            assert m["route"] == "llm"
    assert {e["meta"]["route"] for e in examples} == {"self", "llm", "staff"}


def test_date_tool_used_only_when_available(world_path):
    index, w = PolicyIndex(), World.fresh(world_path)
    planner = Planner(w, seed=5)
    plan = None
    while plan is None:
        plan = planner.plan("movies", "dates")
    with_tool = run_plan(plan, SYSTEMS["fastlane"], world_path, index, seed=1)
    without = run_plan(plan, SYSTEMS["A2"], world_path, index, seed=1)
    assert with_tool["ok"] and without["ok"]
    date_steps = lambda r: [e for e in r["examples"] if "date" in json.loads(e["messages"][-1]["content"])["slots"]]  # noqa: E731
    # With the date tools the small model keeps the date turn: the caller's words go in the slot, code resolves them.
    assert date_steps(with_tool) and all(e["meta"]["route"] == "self" for e in date_steps(with_tool))
    assert not any(json.loads(e["messages"][-1]["content"]).get("tool", {}) and
                   json.loads(e["messages"][-1]["content"])["tool"]["name"] == "resolve_date" for e in with_tool["examples"])
    # Without them (A2) a date said in words is a turn for the large model.
    assert date_steps(without) and all(e["meta"]["route"] == "llm" for e in date_steps(without))


@pytest.mark.parametrize("kind", KINDS)
def test_each_kind_can_succeed(world_path, kind):
    index, w = PolicyIndex(), World.fresh(world_path)
    planner = Planner(w, seed=11)
    ok = False
    for domain in ("movies", "buses", "restaurants"):
        plan = planner.plan(domain, kind)
        if plan is not None and run_plan(plan, SYSTEMS["fastlane"], world_path, index, seed=3)["ok"]:
            ok = True
            break
    assert ok, kind


def test_split_has_no_call_in_both_halves(generated, tmp_path):
    calls, examples = generated
    stats = split_and_write(calls, examples, tmp_path, {"movies", "buses", "restaurants"}, seed=1)
    train = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    val = [json.loads(l) for l in (tmp_path / "val.jsonl").read_text().splitlines()]
    assert not {e["meta"]["plan"] for e in train} & {e["meta"]["plan"] for e in val}
    # Escalation oversampling also repeats noisy escalation examples, so the file can hold more rows.
    assert 0 < stats["noisy_copies"] <= sum(1 for e in train if e["meta"].get("noisy"))
    assert load_split(tmp_path / "train.jsonl")[0]["completion"][0]["role"] == "assistant"


def test_asr_noise():
    rng = random.Random(0)
    text = "Two tickets for Kaala Samudram at Galaxy Kukatpally please."
    noisy = {asr_noise(text, random.Random(i)) for i in range(20)}
    assert len(noisy) > 5 and all(n != text for n in noisy)
    content = "CONVERSATION:\nAgent: Which film?\nCaller: Kaala Samudram at Gachibowli please."
    out = noisy_user_content(content, rng)
    assert out.startswith("CONVERSATION:\nAgent: Which film?\nCaller: ") and out != content


def test_cohen_kappa():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    # observed 0.9, chance (8*7 + 2*3) / 100 = 0.62 -> (0.9 - 0.62) / 0.38
    assert cohen_kappa(["self"] * 8 + ["llm"] * 2, ["self"] * 7 + ["llm"] * 3) == pytest.approx(0.7368, abs=1e-3)

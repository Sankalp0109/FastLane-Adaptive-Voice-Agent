"""Run systems x test calls x seeds and write one JSON line per simulated call.

Runs are resumable: a (call_id, seed) already in the output file is skipped. Each call runs on its
own fresh copy of the world, with the script's setup bookings and injected tool failure applied.
"""

from __future__ import annotations

import json
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from fastlane.agent.orchestrator import Agent
from fastlane.config import DATA_DIR, WORLD_DB, get_system
from fastlane.eval.caller_sim import CallerSim
from fastlane.eval.labels import is_hard
from fastlane.eval.scenarios import CallScript, load_test_calls, setup_world
from fastlane.eval.score import booking_views, changed_bookings, score_call
from fastlane.models.base import LLMClient
from fastlane.retrieval.index import PolicyIndex
from fastlane.world.db import World

RUNS_DIR = DATA_DIR / "runs"
_INDEX: PolicyIndex | None = None


def policy_index() -> PolicyIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = PolicyIndex()
    return _INDEX


def stt_partials(text: str) -> list[str]:
    """Interim transcripts a streaming recogniser would emit while the caller speaks."""
    words = text.split()
    if len(words) < 6:
        return []
    return [" ".join(words[: max(3, int(len(words) * f))]) for f in (0.5, 0.75)]


def sample_calls(calls: list[CallScript], n: int | None, categories: list[str] | None = None,
                 domains: list[str] | None = None) -> list[CallScript]:
    """Deterministic stratified sample: round-robin over (domain, category) groups."""
    pool = [c for c in calls if (not categories or c.category in categories) and (not domains or c.domain in domains)]
    if n is None or n >= len(pool):
        return pool
    groups: dict[tuple[str, str], list[CallScript]] = defaultdict(list)
    for c in pool:
        groups[(c.domain, c.category)].append(c)
    order = sorted(groups)
    out: list[CallScript] = []
    i = 0
    while len(out) < n:
        g = groups[order[i % len(order)]]
        k = i // len(order)
        if k < len(g):
            out.append(g[k])
        i += 1
        if i > n * len(order) + len(pool):
            break
    return out


def run_call(script: CallScript, system: str, seed: int, small: LLMClient | None, large: LLMClient | None,
             caller: LLMClient, max_turns: int = 16, world_path: Path = WORLD_DB) -> dict:
    cfg = get_system(system)
    world = World.fresh(world_path)
    setup_world(world, script)
    before = booking_views(world)
    inject = {script.inject_failure: 1} if script.inject_failure else None
    agent = Agent(world, script.phone, cfg, small=small, large=large, policy_index=policy_index(), inject_failures=inject)
    sim = CallerSim(caller, script, seed=seed)
    t0 = time.perf_counter()
    ended_by = "max_turns"
    labels: list[dict] = []
    error = None
    try:
        agent_text = agent.greet()
        say, moments = sim.opening(agent_text)
        done = False
        for i in range(max_turns):
            agent_text = agent.respond(say, stt_partials(say))
            tr = agent.traces[-1]
            no_avail = any(t.error == "no_availability" for t in tr.tools) or tr.escalation_reason == "no availability"
            hard, why = is_hard(say, moments, no_avail)
            labels.append({"turn": tr.turn, "hard": hard, "moments": why})
            if agent.ended:
                ended_by = "agent"
                break
            if done:
                ended_by = "caller"
                break
            say, done, moments = sim.reply(agent_text)
    except Exception as e:  # keep the run going; the call counts as failed
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"[:1500]
    finally:
        agent.close()
    after = booking_views(world)
    changed = changed_bookings(before, after)
    agent_all = " ".join(t.agent for t in agent.traces)
    score = score_call(script.gold, changed, agent_all, agent.gw.handoff)
    if error:
        score["success"] = False
    turns = []
    for tr, lab in zip(agent.traces[1:], labels):
        d = tr.to_dict()
        d.update(hard=lab["hard"], moments=lab["moments"])
        turns.append(d)
    events = agent.gw.events
    return {
        "call_id": script.call_id, "system": system, "seed": seed, "domain": script.domain, "category": script.category,
        **score,
        "handoff": agent.gw.handoff,
        "writes": [{"tool": e.tool, "confirmed": e.confirmed, "violations": e.violations, "turn": e.turn} for e in events if e.write],
        "unsafe_writes": len(agent.gw.unsafe_writes()),
        "tool_calls": len(events),
        "tool_errors": dict(_count(e.error for e in events if e.error)),
        "injected": script.inject_failure,
        "early_lookup": {k: v for k, v in asdict(agent.early.stats).items() if k != "log"},
        "changed": changed,
        "turns": turns,
        "ended_by": ended_by,
        "caller_turns": len(labels),
        "caller_model_ms": round(sim.latency_ms, 1),
        "wall_s": round(time.perf_counter() - t0, 2),
        "error": error,
        "models": {"small": getattr(small, "name", None), "large": getattr(large, "name", None), "caller": caller.name},
    }


def _count(items):
    out: dict = defaultdict(int)
    for x in items:
        out[x] += 1
    return out


def run(system: str, calls: list[CallScript], seeds: list[int], run_name: str, small: LLMClient | None,
        large: LLMClient | None, caller: LLMClient, progress=print) -> Path:
    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{system}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["call_id"], r["seed"]))
    todo = [(c, s) for s in seeds for c in calls if (c.call_id, s) not in done]
    progress(f"[{system}] {len(done)} done, {len(todo)} to run -> {path}")
    with path.open("a") as f:
        for i, (call, seed) in enumerate(todo, 1):
            rec = run_call(call, system, seed, small, large, caller)
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            p95 = max((t["model_ms"] for t in rec["turns"]), default=0)
            progress(f"[{system}] {i}/{len(todo)} {call.call_id} seed={seed} success={rec['success']} "
                     f"turns={rec['caller_turns']} max_model_ms={p95:.0f} wall={rec['wall_s']}s" + (" ERROR" if rec["error"] else ""))
    return path


def load_runs(run_name: str) -> list[dict]:
    rows = []
    for path in sorted((RUNS_DIR / run_name).glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


__all__ = ["run", "run_call", "load_runs", "sample_calls", "load_test_calls"]

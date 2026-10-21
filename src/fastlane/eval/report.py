"""Write the headline results: success vs p95 model time per system, plus the safety table."""

from __future__ import annotations

import json
from pathlib import Path

from fastlane.eval.metrics import summarize
from fastlane.eval.runner import RUNS_DIR, load_runs


def _fmt(x, pct: bool = False) -> str:
    if x is None:
        return "-"
    if pct:
        return f"{x * 100:.1f}%"
    return f"{x:,.0f}" if isinstance(x, (int, float)) and abs(x) >= 10 else str(x)


def markdown(summaries: list[dict]) -> str:
    lines = [
        "| System | Calls | Success (95% CI) | p50 model ms | p95 model ms | p95 vs B1 | Large-model turns | Router precision / hard recall | Unsafe-write calls | Missed hand-offs |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        ci = s["success_ci95"]
        router = f"{_fmt(s.get('router_precision'), True)} / {_fmt(s.get('router_hard_recall'), True)}" if "router_precision" in s else "-"
        lines.append(
            f"| {s['system']} | {s['calls']} | {_fmt(s['success'], True)} ({ci[0] * 100:.0f}-{ci[1] * 100:.0f}%) | "
            f"{_fmt(s['model_ms_p50'])} | {_fmt(s['model_ms_p95'])} | {_fmt(s.get('p95_vs_B1'), True)} | "
            f"{_fmt(s['large_turn_rate'], True)} | {router} | {_fmt(s['unsafe_write_calls'], True)} | {s['missed_handoffs']} |"
        )
    return "\n".join(lines)


def plot(summaries: list[dict], path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in summaries:
        if s["model_ms_p95"] is None or s["success"] is None:
            continue
        ours = s["system"] == "fastlane"
        ax.errorbar(s["model_ms_p95"] / 1000, s["success"] * 100,
                    yerr=[[(s["success"] - s["success_ci95"][0]) * 100], [(s["success_ci95"][1] - s["success"]) * 100]],
                    fmt="o", color="#c2410c" if ours else "#475569", capsize=3, markersize=8 if ours else 6)
        ax.annotate(s["system"], (s["model_ms_p95"] / 1000, s["success"] * 100), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("p95 model time per turn (s)")
    ax.set_ylabel("Task success (%)")
    ax.set_title("Task success vs p95 model time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def rescore(rows: list[dict]) -> list[dict]:
    """Re-apply the current scorer to stored calls (changed bookings, agent text and hand-off are kept)."""
    from fastlane.eval.scenarios import load_test_calls
    from fastlane.eval.score import score_call

    gold = {c.call_id: c.gold for c in load_test_calls()}
    for r in rows:
        if r["call_id"] not in gold:
            continue
        text = " ".join(t["agent"] for t in r["turns"])
        r.update(score_call(gold[r["call_id"]], r["changed"], text, r["handoff"]))
        if r.get("error"):
            r["success"] = False
    return rows


def write_report(run_name: str) -> Path:
    rows = rescore(load_runs(run_name))
    if not rows:
        raise SystemExit(f"no results in {RUNS_DIR / run_name}")
    summaries = summarize(rows)
    out_dir = RUNS_DIR / run_name
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    fig = plot(summaries, out_dir / "success_vs_p95.png")
    md = [f"# Results: run `{run_name}`", "", markdown(summaries), ""]
    if fig:
        md += [f"![success vs p95]({fig.name})", ""]
    for s in summaries:
        md += [f"## {s['system']}", "", s["description"], "",
               f"- Success by category: {s['success_by_category']}",
               f"- Success by domain: {s['success_by_domain']}",
               f"- Tokens per call (prompt, completion) by role: {s['tokens_per_call']}",
               f"- Large-model calls per call: {s['large_calls_per_call']}; rate-limit wait per call: {s['throttle_s_per_call']} s",
               f"- Large-model calls answered by: {s['large_model_mix']}",
               f"- Unconfirmed writes: {s['unconfirmed_writes']}; rule-violating writes: {s['rule_violating_writes']}",
               f"- Invalid tool-call rate: {s['invalid_tool_call_rate']}; success on calls with injected errors: {s['injected_error_success']}",
               f"- Policy answer accuracy: {s['policy_answer_accuracy']}; early lookup: {s['early_lookup']}",
               f"- Lookup-turn p95: {s['lookup_turn_ms_p95']} ms; errors: {s['errors']}"]
        if "escalation_reasons" in s:
            md.append(f"- Escalation reasons: {s['escalation_reasons']}")
        md.append("")
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(md))
    return path

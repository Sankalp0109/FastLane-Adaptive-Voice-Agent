"""Numbers for the final report, from the stored final runs (run from the repo root):

    .venv/bin/python results/final/final_tables.py

Writes results/final/final_tables.json: every system's summary on its own calls, plus FastLane and B1 restricted to
the 48-call subset, so every 48-call system is compared with FastLane on the same calls.
"""

import json
from pathlib import Path

from fastlane.eval.metrics import summarize_system

HERE = Path(__file__).resolve().parent


def rows(run: str, system: str) -> list[dict]:
    path = HERE / run / f"{system}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    out: dict[str, dict] = {}
    final = {s: rows("final", s) for s in ["fastlane", "B1", "B2", "B3", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]}
    subset48 = {r["call_id"] for r in final["B3"]}
    for system, rs in final.items():
        out[system] = summarize_system(rs)
    for system in ("fastlane", "B1"):
        out[f"{system}@48"] = summarize_system([r for r in final[system] if r["call_id"] in subset48])
    out["fastlane_fix"] = summarize_system(rows("final_fix", "fastlane"))
    out["fastlane_v5@restaurants"] = summarize_system(rows("final_v5", "fastlane"))
    out["A8@restaurants"] = summarize_system(rows("final_v5", "A8"))
    rest = {r["call_id"] for r in rows("final_v5", "A8")}
    out["fastlane@restaurants40"] = summarize_system([r for r in final["fastlane"] if r["call_id"] in rest])
    # Unconfirmed-write calls and writes, from the per-call rows.
    for key, rs in list(final.items()) + [("fastlane@48", [r for r in final["fastlane"] if r["call_id"] in subset48]),
                                          ("B1@48", [r for r in final["B1"] if r["call_id"] in subset48])]:
        out[key]["unsafe_calls_n"] = sum(1 for r in rs if r["unsafe_writes"])
    (HERE / "final_tables.json").write_text(json.dumps(out, indent=1))
    for k, s in out.items():
        print(f"{k:24} n={s['calls']:3} succ={s['success']:.3f} ci={s['success_ci95']} p50={s['model_ms_p50']:.0f} "
              f"p95={s['model_ms_p95']:.0f} large={s['large_turn_rate']} lcpc={s['large_calls_per_call']} "
              f"inv={s['invalid_tool_call_rate']} unsafe={s.get('unsafe_calls_n')} "
              f"R={s.get('router_precision')}/{s.get('router_hard_recall')}/{s.get('router_hard_recall_dates_in_code')} "
              f"tok={s['tokens_per_call'].get('large')}")


if __name__ == "__main__":
    main()

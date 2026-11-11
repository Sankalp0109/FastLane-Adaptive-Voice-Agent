"""Figures for the final report, from the final runs (run from the repo root):

    .venv/bin/python results/final/final_tables.py        # writes results/final/final_tables.json
    .venv/bin/python report/final/make_figures_final.py

Every system is plotted on the same 48 calls (FastLane and B1 restricted to the 48-call subset). The training-loss
and pilot figures do not depend on the final runs and are copied from report/figures.
"""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results/final/final_tables.json"
OUT = Path(__file__).with_name("figures")
OUT.mkdir(exist_ok=True)

# Categorical slots 1-2 of the reference palette (validated: CVD and normal-vision separation pass on white).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
})

SYSTEMS = ["fastlane@48", "B1@48", "B2", "B3", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]


def name(key: str) -> str:
    return {"fastlane@48": "FastLane", "B1@48": "B1"}.get(key, key)


def load() -> dict:
    return json.loads(TABLES.read_text())


def systems(t: dict) -> None:
    """One row per system, four panels with their own units (no shared value axis): task success with its 95%
    interval, model time per turn (p50 hollow, p95 filled), large-model calls per call, and the share of calls with a
    booking change the caller never confirmed. All systems on the same 48 calls; rows sorted by success."""
    order = sorted(SYSTEMS, key=lambda k: (t[k]["success"], -t[k]["model_ms_p95"]))
    ys = list(range(len(order)))
    colour = [ORANGE if k == "fastlane@48" else BLUE for k in order]
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.9), sharey=True, gridspec_kw={"width_ratios": [1.25, 1.25, 1, 1]})

    ax = axes[0]
    for y, k, c in zip(ys, order, colour):
        s = t[k]
        lo, hi = s["success_ci95"]
        ax.plot([100 * lo, 100 * hi], [y, y], color=c, linewidth=2, solid_capstyle="round", alpha=0.45)
        ax.scatter(100 * s["success"], y, s=34, color=c, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(101, y, f"{100 * s['success']:.0f}", va="center", ha="left", fontsize=7, color=INK)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 50, 100])
    ax.set_title("Task success (%)\nwith 95% interval", fontsize=8, color=INK)

    ax = axes[1]
    for y, k, c in zip(ys, order, colour):
        s = t[k]
        p50, p95 = s["model_ms_p50"] / 1000, s["model_ms_p95"] / 1000
        ax.plot([p50, p95], [y, y], color=GRID, linewidth=1.5, zorder=1)
        ax.scatter(p50, y, s=30, facecolor="white", edgecolor=c, linewidth=1.5, zorder=3)
        ax.scatter(p95, y, s=34, color=c, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(p95 + 0.4, y, f"{p95:.1f}", va="center", ha="left", fontsize=7, color=INK)
    ax.set_xlim(0, 19.5)
    ax.set_title("Model time per turn (s)\n\u25cb p50   \u25cf p95", fontsize=8, color=INK)

    for ax, title, get, fmt in (
        (axes[2], "Large-model calls\nper call", lambda s: s["large_calls_per_call"], "{:.1f}"),
        (axes[3], "Calls with an unconfirmed\nbooking change (%)", lambda s: 100 * s["unsafe_calls_n"] / s["calls"], "{:.0f}"),
    ):
        vals = [get(t[k]) for k in order]
        bars = ax.barh(ys, vals, height=0.6, color=colour, edgecolor="white", linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.text(v, b.get_y() + b.get_height() / 2, " " + fmt.format(v), va="center", ha="left", fontsize=7,
                    color=INK)
        ax.set_xlim(0, max(vals) * 1.3)
        ax.set_title(title, fontsize=8, color=INK)

    axes[0].set_yticks(ys, [name(k) for k in order], fontsize=8)
    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "systems_final.pdf")
    plt.close(fig)


if __name__ == "__main__":
    tables = load()
    systems(tables)
    for f in ("loss_curves.pdf", "pilot.pdf"):
        shutil.copy(ROOT / "report/figures" / f, OUT / f)
    print("figures ->", OUT)

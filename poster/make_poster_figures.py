"""Figures for the A1 poster, drawn at their printed size so that text is readable from 1-2 m (run from the repo root):

    .venv/bin/python results/final/final_tables.py
    .venv/bin/python poster/make_poster_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("figures")
OUT.mkdir(exist_ok=True)
T = json.loads((ROOT / "results/final/final_tables.json").read_text())

# Categorical slots 1-3 of the reference palette (validated on white). FastLane / tuned = orange everywhere.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 20, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 1.0, "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "pdf.fonttype": 42, "axes.linewidth": 1.2,
})

NAMES = {"fastlane@48": "FastLane", "B1@48": "B1 large model only", "B2": "B2 untuned small only",
         "B3": "B3 large model as router", "A1": "A1 no fine-tuning", "A2": "A2 no date tools",
         "A3": "A3 rules only in prompt", "A4": "A4 no booking record", "A5": "A5 no reasoning",
         "A6": "A6 no early lookup", "A7": "A7 no caller history"}


def results() -> None:
    """All systems on the same 48 calls: success [95% CI], p50/p95 time, large-model calls, unconfirmed changes."""
    keys = list(NAMES)
    order = sorted(keys, key=lambda k: (T[k]["success"], -T[k]["model_ms_p95"]))
    ys = list(range(len(order)))
    col = [ORANGE if k == "fastlane@48" else BLUE for k in order]
    fig, axes = plt.subplots(1, 4, figsize=(352 * MM, 150 * MM), sharey=True,
                             gridspec_kw={"width_ratios": [1.2, 1.25, 0.95, 0.95]})
    ax = axes[0]
    for y, k, c in zip(ys, order, col):
        s = T[k]
        lo, hi = s["success_ci95"]
        ax.plot([100 * lo, 100 * hi], [y, y], color=c, linewidth=4, solid_capstyle="round", alpha=0.4)
        ax.scatter(100 * s["success"], y, s=150, color=c, edgecolor="white", linewidth=2, zorder=3)
        ax.text(102, y, f"{100 * s['success']:.0f}", va="center", ha="left", fontsize=17, color=INK)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 50, 100])
    ax.set_title("Task success (%)\nwith 95% interval", fontsize=19, color=INK)
    ax = axes[1]
    for y, k, c in zip(ys, order, col):
        s = T[k]
        p50, p95 = s["model_ms_p50"] / 1000, s["model_ms_p95"] / 1000
        ax.plot([p50, p95], [y, y], color=GRID, linewidth=3, zorder=1)
        ax.scatter(p50, y, s=120, facecolor="white", edgecolor=c, linewidth=2.5, zorder=3)
        ax.scatter(p95, y, s=150, color=c, edgecolor="white", linewidth=2, zorder=3)
        ax.text(p95 + 0.5, y, f"{p95:.1f}", va="center", ha="left", fontsize=17, color=INK)
    ax.set_xlim(0, 20.5)
    ax.set_title("Model time per turn (s)\nhollow p50, filled p95", fontsize=19, color=INK)
    for ax, title, get, fmt in (
        (axes[2], "Large-model\ncalls per call", lambda s: s["large_calls_per_call"], "{:.1f}"),
        (axes[3], "Unconfirmed booking\nchanges (% of calls)", lambda s: 100 * s["unsafe_calls_n"] / s["calls"], "{:.0f}"),
    ):
        vals = [get(T[k]) for k in order]
        bars = ax.barh(ys, vals, height=0.62, color=col, edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(v, b.get_y() + b.get_height() / 2, " " + fmt.format(v), va="center", ha="left", fontsize=17,
                    color=INK)
        ax.set_xlim(0, max(vals) * 1.32)
        ax.set_title(title, fontsize=19, color=INK)
    axes[0].set_yticks(ys, [NAMES[k] for k in order], fontsize=18)
    for lbl, k in zip(axes[0].get_yticklabels(), order):
        if k == "fastlane@48":
            lbl.set_fontweight("bold")
    for ax in axes:
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=16)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "results_poster.pdf")
    plt.close(fig)


def finetune() -> None:
    """Offline step accuracy, untuned vs. tuned (adapter v3), same 267 held-out steps (report Table 'offline')."""
    metrics = ["Route\naccuracy", "Hard turns\npassed on", "Right\ntool", "Exact tool\narguments"]
    untuned = [89.1, 23.1, 49.8, 26.8]
    tuned = [99.3, 92.3, 98.8, 99.0]
    fig, ax = plt.subplots(figsize=(160 * MM, 88 * MM))
    xs = range(len(metrics))
    w = 0.36
    for off, vals, c, lab in ((-w / 2 - 0.02, untuned, BLUE, "untuned Qwen2.5-3B"), (w / 2 + 0.02, tuned, ORANGE, "tuned (adapter v3)")):
        bars = ax.bar([x + off for x in xs], vals, width=w, color=c, edgecolor="white", linewidth=2, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=16, color=INK)
    ax.set_xticks(list(xs), metrics, fontsize=16)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(axis="y", labelsize=15, colors=MUTED)
    ax.set_ylabel("% of held-out steps", fontsize=16)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=2, fontsize=15, bbox_to_anchor=(0.5, 1.14), handlelength=1.2, columnspacing=1.2)
    fig.savefig(OUT / "finetune_poster.pdf")
    plt.close(fig)


def failures() -> None:
    """FastLane's failed calls by cause, before and after the fixes (same 120 calls), from reading every transcript."""
    rows = [("before fixes\n(108/120 succeed)", [6, 1, 5]), ("after fixes\n(105/120 succeed)", [7, 2, 6])]
    causes = [("simulated caller", BLUE), ("scorer", AQUA), ("agent", ORANGE)]
    fig, ax = plt.subplots(figsize=(160 * MM, 80 * MM))
    for y, (label, vals) in enumerate(rows):
        left = 0
        for (name, c), v in zip(causes, vals):
            ax.barh(y, v, left=left, height=0.55, color=c, edgecolor="white", linewidth=3,
                    label=name if y == 0 else None)
            ax.text(left + v / 2, y, str(v), ha="center", va="center", fontsize=18, color="white", fontweight="bold")
            left += v
    ax.set_yticks([0, 1], [r[0] for r in rows], fontsize=16)
    ax.invert_yaxis()
    ax.set_xlim(0, 15.5)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("failed calls (of 120)", fontsize=16)
    ax.tick_params(axis="x", labelsize=15)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", ncol=3, fontsize=15, bbox_to_anchor=(0.45, 1.25), handlelength=1.0, columnspacing=1.0)
    fig.savefig(OUT / "failures_poster.pdf")
    plt.close(fig)


if __name__ == "__main__":
    results()
    finetune()
    failures()
    print("figures ->", OUT)

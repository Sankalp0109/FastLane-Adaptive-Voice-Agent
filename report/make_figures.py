"""Figures for the report, read from results/ (run from the repo root: .venv/bin/python report/make_figures.py)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).with_name("figures")
OUT.mkdir(exist_ok=True)

# Categorical slots 1-3 of the reference palette (light mode, print).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
})


def loss_curves():
    v1 = json.loads((RES / "adapter_v1/loss_curve.json").read_text())["eval"]
    v2 = json.loads((RES / "adapter_v2/loss_curves.json").read_text())["all_domains"]["eval"]
    v3 = json.loads((RES / "adapter_v3/loss_curve.json").read_text())["eval"][:5]
    series = [
        ("v1 (data v1)", [r[1] for r in v1], [r[2] for r in v1], BLUE),
        ("v2 (data v2)", [r[0] for r in v2], [r[1] for r in v2], ORANGE),
        ("v3 (data v3)", [r["epoch"] for r in v3], [r["eval_loss"] for r in v3], AQUA),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    for label, x, y, c in series:
        ax.plot(x, y, color=c, linewidth=2, marker="o", markersize=4, label=f"{label}, final {y[-1]:.4f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss")
    ax.set_xlim(0.3, 2.1)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(OUT / "loss_curves.pdf")
    plt.close(fig)


def pilot():
    v0 = json.loads((RES / "pilot_v0_untuned/summary.json").read_text())[0]
    v3 = json.loads((RES / "pilot_v3_tuned/summary.json").read_text())[0]
    panels = [
        ("Task success (%)", lambda s: 100 * s["success"], "{:.1f}"),
        ("Turns sent to large model (%)", lambda s: 100 * s["large_turn_rate"], "{:.1f}"),
        ("Model time p50 (s)", lambda s: s["model_ms_p50"] / 1000, "{:.2f}"),
        ("Model time p95 (s)", lambda s: s["model_ms_p95"] / 1000, "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 1.9))
    for ax, (title, get, fmt) in zip(axes, panels):
        vals = [get(v0), get(v3)]
        bars = ax.bar([0, 1], vals, width=0.62, color=[BLUE, ORANGE], edgecolor="white", linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks([0, 1], ["v0\nuntuned", "v3\ntuned"], fontsize=8)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "pilot.pdf")
    plt.close(fig)


def systems():
    """Success vs. model time per system on the 24 mid-submission calls (the proposal's headline plot)."""
    path = RES / "mid_baselines/summary.json"
    if not path.exists():
        return
    # A8 ran on the 8 restaurant calls only, so it is left out; A2 is labelled with its call count.
    rows = [r for r in json.loads(path.read_text()) if r["calls"] >= 18]
    a1 = json.loads((RES / "pilot_v0_untuned/summary.json").read_text())[0]
    rows.append({**a1, "system": "A1"})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4), sharey=True)
    for ax, key, title in ((axes[0], "model_ms_p50", "p50"), (axes[1], "model_ms_p95", "p95")):
        for r in rows:
            ours = r["system"] == "fastlane"
            x, y = r[key] / 1000, 100 * r["success"]
            ax.scatter(x, y, s=42 if ours else 30, color=ORANGE if ours else BLUE, edgecolor="white", linewidth=1.2, zorder=3)
            label = "FastLane" if ours else r["system"] + ("" if r["calls"] == 24 else f" ({r['calls']} calls)")
            # A1/A2 sit close together, and so do FastLane/B3 at p95: put those labels on the left
            left = r["system"] == "A1" or (ours and title == "p95")
            ax.annotate(label, (x, y), xytext=(-5, -3) if left else (5, 3), textcoords="offset points",
                        fontsize=8, color=INK, ha="right" if left else "left")
        ax.set_xlabel(f"model time per turn, {title} (s)")
        ax.set_xlim(left=0)
    axes[0].set_ylabel("task success (%)")
    axes[0].set_ylim(0, 100)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "systems.pdf")
    plt.close(fig)


if __name__ == "__main__":
    loss_curves()
    pilot()
    systems()
    print("figures ->", OUT)

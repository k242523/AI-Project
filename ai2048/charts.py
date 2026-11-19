"""Charts for the evaluation (Section 7 of the proposal)."""
import math
import os
import re
from typing import Dict, List

from .experiments import ExperimentRunner

PALETTE = ["#b8bcc4", "#8a8e96", "#555a63", "#2b2d31", "#7a6a58", "#a39a90"]


def make_charts(results: List[Dict], out_dir: str) -> List[str]:
    """Create the four charts as PNG files in out_dir and return their paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("Charts need matplotlib. Install it with: pip install matplotlib")

    os.makedirs(out_dir, exist_ok=True)
    metrics = ExperimentRunner([], [], results).calculate_metrics()
    names = list(metrics)
    labels = [f"{n}\n(n={metrics[n]['games']})" for n in names]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#6b6f76", "text.color": "#2b2d31",
                         "axes.labelcolor": "#2b2d31", "xtick.color": "#2b2d31",
                         "ytick.color": "#2b2d31"})
    paths = []

    def save(fig, name):
        path = os.path.join(out_dir, name)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    # 1. average final score
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sem = [metrics[n]["std_score"] / math.sqrt(max(metrics[n]["games"], 1)) for n in names]
    bars = ax.bar(labels, [metrics[n]["avg_score"] for n in names], yerr=sem, color=colors, capsize=4)
    ax.bar_label(bars, labels=[f"{metrics[n]['avg_score']:,.0f}" for n in names], padding=4)
    ax.set_ylabel("Average final score")
    ax.margins(y=0.18)
    ax.set_title("Average final score by agent (error bars: standard error)")
    save(fig, "1_average_score.png")

    # 2. highest-tile distribution
    tiles = sorted({t for n in names for t in metrics[n]["tile_counts"]})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    width = 0.8 / max(len(names), 1)
    for i, n in enumerate(names):
        total = metrics[n]["games"]
        pct = [100 * metrics[n]["tile_counts"].get(t, 0) / total for t in tiles]
        ax.bar([j + i * width for j in range(len(tiles))], pct, width, label=n, color=colors[i])
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(tiles))])
    ax.set_xticklabels([str(t) for t in tiles])
    ax.set_xlabel("Highest tile reached")
    ax.set_ylabel("% of games")
    ax.set_title("Highest-tile distribution")
    ax.legend(frameon=False)
    save(fig, "2_highest_tile_distribution.png")

    # 3. decision time versus search depth
    depth_pts = sorted((int(m.group(1)), metrics[n]["avg_decision_ms"])
                       for n in names if (m := re.search(r"-d(\d+)$", n)))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if depth_pts:
        xs, ys = zip(*depth_pts)
        ax.plot(xs, ys, marker="o", color="#2b2d31")
        for x, y in depth_pts:
            ax.annotate(f"{y:.1f} ms", (x, y), textcoords="offset points", xytext=(6, 6))
        ax.set_xticks(list(xs))
        ax.set_yscale("log")
    ax.set_xlabel("Search depth (player moves)")
    ax.set_ylabel("Average decision time per move (ms, log scale)")
    ax.set_title("Decision time versus search depth")
    save(fig, "3_decision_time_vs_depth.png")

    # 4. percentage of games reaching 2048
    fig, ax = plt.subplots(figsize=(7, 4.2))
    vals = [metrics[n]["pct_reached_2048"] for n in names]
    bars = ax.bar(labels, vals, color=colors)
    ax.bar_label(bars, labels=[f"{v:.0f}%" for v in vals], padding=4)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of games reaching 2048")
    ax.set_title("Games reaching the 2048 tile")
    save(fig, "4_reached_2048.png")
    return paths

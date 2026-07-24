"""Generate the RoboLab replay reproducibility figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import GREEN, RED, configure_style, load_summary, save_figure


def main() -> None:
    """Plot per-repeat replay passes and maximum official state error."""
    configure_style()
    replay = load_summary()["robolab_replay_reproducibility"]
    rows = replay["repeat_rows"]
    labels = [f"Repeat {row['repeat']}" for row in rows]
    x = np.arange(len(rows))
    passes = [row["attempted"] - row["official_failed_replays"] for row in rows]
    failures = [row["official_failed_replays"] for row in rows]
    errors = [row["maximum_official_state_error"] for row in rows]
    tolerance = replay["official_state_tolerance"]

    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.05))
    axes[0].bar(x, passes, color=GREEN, label="Within 0.01")
    axes[0].bar(x, failures, bottom=passes, color=RED, label="Violated 0.01")
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylabel("Completed replays")
    axes[0].set_ylim(0, 11)
    axes[0].spines[["top", "right"]].set_visible(False)

    bars = axes[1].bar(x, errors, color=[GREEN, RED, RED])
    axes[1].axhline(
        tolerance,
        color="#111827",
        linestyle="--",
        linewidth=1.1,
    )
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylabel("Maximum absolute state error")
    axes[1].set_ylim(0, 0.175)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].text(
        -0.15,
        tolerance + 0.008,
        "Official tolerance\n= 0.01",
        ha="left",
        va="bottom",
        fontsize=7.0,
    )
    for bar, value in zip(bars, errors, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.subplots_adjust(wspace=0.36, top=0.86)
    save_figure(
        figure,
        "robolab_replay_reproducibility.png",
        "Per-repeat RoboLab replay outcomes and maximum official state error.",
    )


if __name__ == "__main__":
    main()

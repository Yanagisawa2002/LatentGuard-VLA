"""Generate the task-specific progress generalization-gap figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import BLUE, ORANGE, configure_style, load_summary, save_figure


def main() -> None:
    """Compare in-domain and held-out-task progress metrics."""
    configure_style()
    metrics = load_summary()["progress_generalization"]
    rows = [
        ("MAE (lower is better)", "mae", (0.0, 0.24)),
        ("Spearman correlation", "spearman", (0.0, 1.0)),
        ("Pairwise accuracy", "pairwise_accuracy", (0.0, 1.0)),
    ]
    labels = ["In-domain", "Held-out task"]
    colors = [BLUE, ORANGE]

    figure, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    for axis, (ylabel, key, limits) in zip(axes, rows, strict=True):
        values = [metrics["in_domain"][key], metrics["held_out_task"][key]]
        bars = axis.bar(np.arange(2), values, color=colors, width=0.62)
        axis.set_xticks(np.arange(2), labels, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_ylim(*limits)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (limits[1] - limits[0]) * 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    figure.subplots_adjust(wspace=0.42)
    save_figure(
        figure,
        "sarm_generalization_gap.png",
        "Frozen LG-R1 in-domain versus LG-R1b held-out-task progress metrics.",
    )


if __name__ == "__main__":
    main()

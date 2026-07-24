"""Generate the task-agnostic reward-model failure-metrics figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import (
    BLUE,
    GRAY,
    GREEN,
    ORANGE,
    configure_style,
    load_summary,
    save_figure,
)


def main() -> None:
    """Compare descriptive failure metrics for frozen reward signals."""
    configure_style()
    models = list(load_summary()["reward_model_failure_metrics"].values())
    labels = [
        "Frozen SARM",
        "ROBOMETER zero-shot",
        "TOPReward zero-shot",
        "Task-agnostic ensemble*",
    ]
    colors = [GRAY, BLUE, ORANGE, GREEN]
    panels = [
        ("Failure AUPRC", "auprc"),
        ("Precision at requested recall ≥ 0.60", "precision_at_recall"),
        ("FPR at requested recall ≥ 0.60", "false_positive_rate"),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(8.4, 3.55))
    positions = np.arange(len(models))
    for panel_index, (axis, (ylabel, key)) in enumerate(zip(axes, panels, strict=True)):
        values = [model["all_episode_metrics"][key] for model in models]
        bars = axis.barh(positions, values, color=colors, height=0.66)
        axis.set_yticks(positions, labels if panel_index == 0 else [])
        axis.set_xlabel(ylabel)
        axis.set_xlim(0.0, 0.8 if key == "false_positive_rate" else 0.22)
        axis.invert_yaxis()
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_width() + (0.018 if key == "false_positive_rate" else 0.005),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=7.2,
            )

    figure.text(
        0.01,
        -0.02,
        "* Ensemble values are all-sample descriptive metrics; its formal test "
        "AUPRC was 0.067. No model was promoted.",
        ha="left",
        va="top",
        fontsize=7.4,
        color=GRAY,
    )
    figure.subplots_adjust(wspace=0.34)
    save_figure(
        figure,
        "reward_model_failure_metrics.png",
        "Frozen all-sample descriptive failure metrics at the requested "
        "recall >= 0.60 operating point.",
    )


if __name__ == "__main__":
    main()

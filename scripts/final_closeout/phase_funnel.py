"""Generate the LG-F0 phase gate-disposition figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
from common import BLUE, GRAY, GREEN, RED, configure_style, load_summary, save_figure


def main() -> None:
    """Plot the disposition of each preregistered research phase."""
    configure_style()
    phases = load_summary()["phases"]
    labels = [row["phase"] for row in phases]
    scores = [row["status_score"] for row in phases]
    colors = [GREEN if score == 2 else BLUE if score == 1 else RED for score in scores]

    figure, axis = plt.subplots(figsize=(7.2, 3.15))
    axis.plot(range(len(labels)), scores, color=GRAY, linewidth=1.1, zorder=1)
    axis.scatter(
        range(len(labels)),
        scores,
        s=110,
        c=colors,
        edgecolor="white",
        linewidth=1.1,
        zorder=2,
    )
    for index, row in enumerate(phases):
        axis.annotate(
            row["status_label"],
            (index, scores[index]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
        )

    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks([0, 1, 2], ["blocked", "partial", "gate passed"])
    axis.set_ylim(-0.35, 2.35)
    axis.set_ylabel("Phase gate disposition")
    axis.set_xlabel("Preregistered phase")
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0,
        -0.31,
        "Disposition tracks the frozen gate decision, not task success.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        color=GRAY,
        fontsize=7.5,
    )

    save_figure(
        figure,
        "phase_funnel.png",
        "Frozen gate disposition across LG-R0 through LG-RB0.1.",
    )


if __name__ == "__main__":
    main()

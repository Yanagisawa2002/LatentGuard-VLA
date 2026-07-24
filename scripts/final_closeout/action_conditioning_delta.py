"""Generate the numeric action-conditioning comparison figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from common import BLUE, ORANGE, configure_style, load_summary, save_figure


def main() -> None:
    """Compare state-only and state-plus-action predictions in two scopes."""
    configure_style()
    scopes = load_summary()["action_conditioning"]
    panels = [
        ("Short-horizon MAE\n(lower is better)", "short_mae", (0.0, 0.05)),
        ("Short-horizon Spearman", "short_spearman", (0.0, 0.55)),
        ("Terminal failure AUPRC", "terminal_failure_auprc", (0.0, 0.23)),
    ]
    scope_labels = ["All tasks", "Leave task 6 out"]
    x = np.arange(2)
    width = 0.34

    figure, axes = plt.subplots(1, 3, figsize=(7.8, 3.0))
    for axis, (ylabel, key, limits) in zip(axes, panels, strict=True):
        state_values = [
            scopes["all_tasks"]["state_only"][key],
            scopes["leave_task6_out"]["state_only"][key],
        ]
        action_values = [
            scopes["all_tasks"]["state_action"][key],
            scopes["leave_task6_out"]["state_action"][key],
        ]
        state_bars = axis.bar(
            x - width / 2,
            state_values,
            width,
            label="State only",
            color=BLUE,
        )
        action_bars = axis.bar(
            x + width / 2,
            action_values,
            width,
            label="State + action",
            color=ORANGE,
        )
        axis.set_xticks(x, scope_labels, rotation=17, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_ylim(*limits)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(
            [*state_bars, *action_bars],
            [*state_values, *action_values],
            strict=True,
        ):
            value_label = f"{value:.6f}" if key == "short_mae" else f"{value:.3f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (limits[1] - limits[0]) * 0.018,
                value_label,
                ha="center",
                va="bottom",
                fontsize=6.4 if key == "short_mae" else 6.8,
                rotation=90 if key == "short_mae" else 0,
            )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    axes[0].legend().remove()
    figure.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.subplots_adjust(wspace=0.45, top=0.86)
    save_figure(
        figure,
        "action_conditioning_delta.png",
        "Frozen state-only versus numeric state-plus-action prediction metrics.",
    )


if __name__ == "__main__":
    main()

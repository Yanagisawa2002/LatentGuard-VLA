"""Shared plotting utilities for the LG-F0 closeout figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "artifacts" / "final" / "final_summary.json"
FIGURE_DIR = ROOT / "docs" / "figures"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"


def load_summary() -> dict[str, Any]:
    """Load the evidence-bound final summary."""
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def configure_style() -> None:
    """Apply the shared publication-oriented plotting style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#374151",
            "axes.labelcolor": "#111827",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "savefig.dpi": 300,
        }
    )


def summary_sha256() -> str:
    """Return the SHA-256 digest of the final summary bytes."""
    return hashlib.sha256(SUMMARY_PATH.read_bytes()).hexdigest()


def save_figure(figure: Figure, filename: str, description: str) -> Path:
    """Save a tightly cropped, source-bound PNG at 300 dpi."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / filename
    figure.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        metadata={
            "Title": filename,
            "Description": description,
            "Source": "artifacts/final/final_summary.json",
            "SourceSHA256": summary_sha256(),
        },
    )
    plt.close(figure)
    return output

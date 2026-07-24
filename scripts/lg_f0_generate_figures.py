"""Generate every static LG-F0 closeout figure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURE_SCRIPTS = (
    "phase_funnel.py",
    "sarm_generalization_gap.py",
    "reward_model_failure_metrics.py",
    "action_conditioning_delta.py",
    "robolab_replay_reproducibility.py",
)


def main() -> None:
    """Run each evidence-bound figure script with the active interpreter."""
    script_dir = ROOT / "scripts" / "final_closeout"
    for script_name in FIGURE_SCRIPTS:
        subprocess.run(
            [sys.executable, str(script_dir / script_name)],
            cwd=ROOT,
            check=True,
        )
        print(ROOT / "docs" / "figures" / script_name.replace(".py", ".png"))


if __name__ == "__main__":
    main()

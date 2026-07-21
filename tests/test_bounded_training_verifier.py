from __future__ import annotations

from scripts.train_pickcube_act import _run_manifest_matches
from scripts.verify_pickcube_act_bounded_training import _strict_interior_bounds


def _record(minimum: float, maximum: float) -> dict[str, object]:
    return {
        "bounded_action": {
            "minimum": [minimum] * 8,
            "maximum": [maximum] * 8,
        }
    }


def test_strict_interior_bounds_accepts_open_interval() -> None:
    assert _strict_interior_bounds(_record(-0.999999, 0.999999))


def test_strict_interior_bounds_rejects_closed_endpoints() -> None:
    assert not _strict_interior_bounds(_record(-1.0, 0.5))
    assert not _strict_interior_bounds(_record(-0.5, 1.0))


def test_strict_interior_bounds_rejects_malformed_diagnostics() -> None:
    assert not _strict_interior_bounds({})
    assert not _strict_interior_bounds(
        {"bounded_action": {"minimum": [0.0] * 7, "maximum": [0.0] * 8}}
    )


def test_run_manifest_resume_ignores_only_launch_command() -> None:
    original = {"identity": "same", "launch_command": ["train"]}
    resumed = {"identity": "same", "launch_command": ["train", "--resume"]}
    assert _run_manifest_matches(original, resumed)
    assert not _run_manifest_matches(original, {**resumed, "identity": "changed"})

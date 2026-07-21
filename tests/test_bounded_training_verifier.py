from __future__ import annotations

from scripts.train_pickcube_act import (
    _resume_artifact_source_commit,
    _run_manifest_matches,
)
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
    assert _run_manifest_matches(original, resumed, allow_source_commit_drift=False)
    assert not _run_manifest_matches(
        original,
        {**resumed, "identity": "changed"},
        allow_source_commit_drift=False,
    )


def test_terminal_resume_retains_artifact_source_commit(tmp_path) -> None:
    producer = "1" * 40
    (tmp_path / "run_manifest.json").write_text(
        '{"training_identity":{"source_commit":"' + producer + '"}}',
        encoding="utf-8",
    )
    assert (
        _resume_artifact_source_commit(
            root=tmp_path,
            resume=tmp_path / "checkpoint",
            current_source_commit="2" * 40,
        )
        == producer
    )


def test_terminal_resume_manifest_allows_only_top_level_git_drift() -> None:
    original = {"git_commit": "1" * 40, "training_identity": {"source": "same"}}
    verifier = {"git_commit": "2" * 40, "training_identity": {"source": "same"}}
    assert _run_manifest_matches(original, verifier, allow_source_commit_drift=True)
    assert not _run_manifest_matches(
        original,
        {**verifier, "training_identity": {"source": "changed"}},
        allow_source_commit_drift=True,
    )

"""CPU-only smoke tests for all M3A command surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latentguard.action_verifier import CandidateType
from latentguard.cli import main
from latentguard.m3a_cli import (
    _baseline_exclusion_counts,
    _load_resume_report,
    _require_full_dataset_targets,
    _restoration_report,
)


def _common_pickcube_args(tmp_path: Path) -> list[str]:
    return [
        "--compatibility-report",
        str(tmp_path / "compatibility.json"),
        "--action-layout",
        str(tmp_path / "layout.json"),
    ]


@pytest.mark.parametrize(
    ("command", "arguments", "expected"),
    (
        (
            "collect-maniskill-pickcube-sequences",
            [
                "--runtime-archive-dir",
                "states",
                "--reference-archive-dir",
                "references",
                "--summary-output",
                "collection.json",
            ],
            "simulator_runs=0",
        ),
        (
            "build-state-indexed-pickcube",
            [
                "--runtime-archive-dir",
                "states",
                "--source-dir",
                "sources",
                "--anchor-manifest-dir",
                "anchors",
                "--summary-output",
                "anchors.json",
            ],
            "baseline_replays=0",
        ),
        (
            "replay-state-indexed-pickcube",
            [
                "--source-dir",
                "sources",
                "--runtime-archive-dir",
                "states",
                "--anchor-manifest-dir",
                "anchors",
                "--corruption-dir",
                "corruptions",
                "--output-dir",
                "evidence",
                "--seed",
                "7",
            ],
            "evaluations=0",
        ),
    ),
)
def test_pickcube_m3a_commands_have_runtime_free_dry_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    arguments: list[str],
    expected: str,
) -> None:
    result = main(
        [
            command,
            *arguments,
            *_common_pickcube_args(tmp_path),
            "--dry-run",
        ]
    )

    assert result == 0
    assert expected in capsys.readouterr().out


def test_export_action_verifier_has_bounded_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        [
            "export-action-verifier-dataset",
            "--source-dir",
            str(tmp_path / "sources"),
            "--anchor-manifest-dir",
            str(tmp_path / "anchors"),
            "--corruption-dir",
            str(tmp_path / "corruptions"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--output-dir",
            str(tmp_path / "dataset"),
            "--summary-output",
            str(tmp_path / "summary.json"),
            "--train-trajectory-count",
            "4",
            "--validation-trajectory-count",
            "1",
            "--test-trajectory-count",
            "1",
            "--dry-run",
        ]
    )

    assert result == 0
    assert "split=4/1/1" in capsys.readouterr().out


def test_validate_action_verifier_has_runtime_free_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        [
            "validate-action-verifier-dataset",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--anchor-manifest-dir",
            str(tmp_path / "anchors"),
            "--corruption-dir",
            str(tmp_path / "corruptions"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--report-output",
            str(tmp_path / "validation.json"),
            "--require-full-target",
            "--dry-run",
        ]
    )

    assert result == 0
    assert "require_full_target=true" in capsys.readouterr().out


def test_replay_rejects_input_output_overlap_without_deleting_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared = tmp_path / "source"
    shared.mkdir()
    marker = shared / "keep.txt"
    marker.write_text("user data", encoding="utf-8")

    result = main(
        [
            "replay-state-indexed-pickcube",
            "--source-dir",
            str(shared),
            "--runtime-archive-dir",
            str(tmp_path / "states"),
            "--anchor-manifest-dir",
            str(tmp_path / "anchors"),
            "--corruption-dir",
            str(shared),
            "--output-dir",
            str(tmp_path / "evidence"),
            "--compatibility-report",
            str(tmp_path / "compatibility.json"),
            "--action-layout",
            str(tmp_path / "layout.json"),
            "--seed",
            "0",
        ]
    )

    assert result == 1
    assert "input/output directories overlap" in capsys.readouterr().err
    assert marker.read_text(encoding="utf-8") == "user data"


def test_build_rejects_nonfixed_candidate_horizon(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "build-state-indexed-pickcube",
                "--runtime-archive-dir",
                str(tmp_path / "states"),
                "--source-dir",
                str(tmp_path / "sources"),
                "--anchor-manifest-dir",
                str(tmp_path / "anchors"),
                "--summary-output",
                str(tmp_path / "summary.json"),
                "--candidate-horizon",
                "8",
                *_common_pickcube_args(tmp_path),
                "--dry-run",
            ]
        )

    assert error.value.code == 2


def _resume_report_payload() -> dict[str, object]:
    return {
        "anchor_manifest_content_digest": "sha256:anchor",
        "corruption_dataset_digest": "sha256:corruption",
        "evaluated_attempts_this_invocation": 0,
        "execution_error_count": 0,
        "proposal_count": 2,
        "recovered_attempts_this_invocation": 0,
        "resumed_invocation": True,
        "resumed_without_rerun_count": 2,
        "retried_attempts_this_invocation": 0,
        "run_id": "run-1",
        "schema_version": "1.0",
    }


def _load_test_resume_report(path: Path) -> dict[str, object]:
    evaluation = SimpleNamespace(
        run_id="run-1", selected_proposal_ids=("proposal-1", "proposal-2")
    )
    return dict(
        _load_resume_report(
            path,
            evaluation_dataset=evaluation,
            anchor_manifest_digest="sha256:anchor",
            corruption_dataset_digest="sha256:corruption",
        )
    )


def test_load_resume_report_accepts_strict_no_op_resume(tmp_path: Path) -> None:
    payload = _resume_report_payload()
    report = tmp_path / "resume.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    assert _load_test_resume_report(report) == payload


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("evaluated_attempts_this_invocation", 1, "must equal 0"),
        ("run_id", "run-2", "run_id does not match"),
    ),
)
def test_load_resume_report_rejects_rerun_or_run_id_mismatch(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = _resume_report_payload()
    payload[field] = value
    report = tmp_path / "resume.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_test_resume_report(report)


def _restoration_inputs(
    metrics: dict[str, object],
) -> tuple[object, object, object, object]:
    baseline = SimpleNamespace(
        evidence_id="baseline-1",
        complete_action_execution=True,
        executed_action_count=20,
        expected_action_count=20,
        state_restoration_verified=True,
        complete_state_comparison=True,
        official_terminal_success=True,
        simulator_replay_verified=True,
        compared_component_count=70,
        maximum_absolute_error=1.0e-7,
        comparison_semantic="tolerance_verified_full_state_v1",
        comparison_tolerance=1.0e-6,
        verifier_state_restoration_verified=True,
        verifier_state_compared_component_count=32,
        verifier_state_maximum_absolute_error=5.0e-8,
    )
    evidence = SimpleNamespace(
        evidence_id="evidence-1", proposal_id="proposal-1", metrics=metrics
    )
    group = SimpleNamespace(anchor_id="anchor-1", baseline_evidence_id="baseline-1")
    sample = SimpleNamespace(
        anchor_id="anchor-1",
        candidate_type=CandidateType.CORRUPTED,
        proposal_id="proposal-1",
        sample_id="sample-1",
        strong_simulator_evidence_id="evidence-1",
    )
    proposal = SimpleNamespace(
        proposal_id="proposal-1",
        transformed_action=SimpleNamespace(actions=np.zeros((20, 2), dtype=np.float64)),
    )
    return (
        SimpleNamespace(
            candidate_groups=(group,), samples=(sample,), state_vector_dimension=32
        ),
        SimpleNamespace(evidence=(evidence,)),
        SimpleNamespace(baseline_evidence=(baseline,)),
        SimpleNamespace(proposals=(proposal,)),
    )


def _valid_restoration_metrics() -> dict[str, object]:
    metrics: dict[str, object] = {
        "replay_baseline_requested_steps": 20,
        "replay_baseline_steps": 20,
        "replay_corrupted_requested_steps": 20,
        "replay_corrupted_steps": 20,
    }
    for role in ("baseline", "corrupted"):
        metrics[f"replay_{role}_restoration_complete_state_comparison"] = True
        metrics[f"replay_{role}_restoration_compared_component_count"] = 70
        metrics[f"replay_{role}_restoration_maximum_absolute_error"] = 1.0e-7
    return metrics


def test_restoration_report_accepts_complete_metrics() -> None:
    inputs = _restoration_inputs(_valid_restoration_metrics())

    assert _restoration_report(*inputs) == ([70], 1.0e-7, 1, [32], 5.0e-8)


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    (
        (
            "replay_baseline_restoration_maximum_absolute_error",
            None,
            "must be numeric",
        ),
        (
            "replay_corrupted_restoration_compared_component_count",
            "70",
            "must be an integer",
        ),
    ),
)
def test_restoration_report_rejects_missing_or_wrong_metric(
    metric: str, value: object, message: str
) -> None:
    metrics = _valid_restoration_metrics()
    if value is None:
        del metrics[metric]
    else:
        metrics[metric] = value

    with pytest.raises(ValueError, match=message):
        _restoration_report(*_restoration_inputs(metrics))


def test_full_target_rejects_baseline_runtime_exclusion() -> None:
    report = SimpleNamespace(
        candidate_group_count=300,
        corrupted_failure_count=500,
        corrupted_sample_count=2400,
        corrupted_success_count=500,
        source_trajectory_count=60,
        split_trajectory_counts={"train": 48, "validation": 6, "test": 6},
    )
    evaluation = SimpleNamespace(summary=SimpleNamespace(execution_error=0))
    manifest = SimpleNamespace(
        exclusions=(SimpleNamespace(reason_code="baseline_runtime_error"),)
    )

    assert _baseline_exclusion_counts(manifest) == {"baseline_runtime_error": 1}
    with pytest.raises(ValueError, match="unexplained baseline execution errors"):
        _require_full_dataset_targets(
            report, evaluation, manifest, SimpleNamespace(proposals=())
        )


def test_full_target_requires_eight_corruptions_per_anchor() -> None:
    report = SimpleNamespace(
        candidate_group_count=300,
        corrupted_failure_count=500,
        corrupted_sample_count=2400,
        corrupted_success_count=500,
        source_trajectory_count=60,
        split_trajectory_counts={"train": 48, "validation": 6, "test": 6},
    )
    proposals = tuple(
        SimpleNamespace(
            corruption_type="additive_gaussian_noise",
            proposal_id=f"proposal-{index}",
            resolved_parameters={"severity_id": f"severity-{index}"},
            source_episode_id="episode-1",
        )
        for index in range(7)
    )
    evaluation = SimpleNamespace(
        selected_proposal_ids=tuple(item.proposal_id for item in proposals),
        summary=SimpleNamespace(execution_error=0),
    )
    manifest = SimpleNamespace(
        exclusions=(), records=(SimpleNamespace(source_episode_id="episode-1"),)
    )

    with pytest.raises(ValueError, match="requires eight unique corruptions"):
        _require_full_dataset_targets(
            report, evaluation, manifest, SimpleNamespace(proposals=proposals)
        )


def test_collection_interrupt_removes_owned_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_archive = tmp_path / "states"
    reference_archive = tmp_path / "references"
    summary = tmp_path / "summary.json"

    def interrupting_load_scope(_args: object) -> None:
        runtime_archive.mkdir()
        (runtime_archive / "partial.npy").write_bytes(b"partial")
        reference_archive.mkdir()
        (reference_archive / "partial.json").write_text("{}", encoding="utf-8")
        summary.write_text("{}", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "latentguard.m3a_cli._load_scope",
        interrupting_load_scope,
    )

    result = main(
        [
            "collect-maniskill-pickcube-sequences",
            "--runtime-archive-dir",
            str(runtime_archive),
            "--reference-archive-dir",
            str(reference_archive),
            "--summary-output",
            str(summary),
            *_common_pickcube_args(tmp_path),
        ]
    )

    assert result == 130
    assert "no partial output accepted" in capsys.readouterr().err
    assert not runtime_archive.exists()
    assert not reference_archive.exists()
    assert not summary.exists()

"""CPU-only smoke tests for all M3A command surfaces."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import latentguard.m3a_cli as m3a_cli
from latentguard.action_verifier import CandidateType
from latentguard.cli import main
from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
    STATE_INDEXED_ARCHIVE_VERSION,
)
from latentguard.integrations.maniskill_pickcube.verifier_state import (
    PICKCUBE_VERIFIER_STATE_EXTRACTION_BOUNDARY,
    PICKCUBE_VERIFIER_STATE_SEMANTIC,
)
from latentguard.m3a_cli import (
    _ISOLATED_COLLECTION_REQUEST_ENV,
    _ISOLATED_COLLECTION_WORKER_ENV,
    _baseline_exclusion_counts,
    _collect_state_indexed_reference_archive_isolated,
    _collection_summary,
    _isolated_fresh_audit_worker_command,
    _isolated_sequence_worker_command,
    _isolated_worker_rejection_category,
    _load_isolated_fresh_audit_records,
    _load_resume_report,
    _require_full_dataset_targets,
    _restoration_error_distribution,
    _restoration_report,
    _use_isolated_sequence_collection,
    _validate_internal_collection_worker,
    _validate_isolated_worker_success,
    _write_isolated_collection_worker_request,
)


def _common_pickcube_args(tmp_path: Path) -> list[str]:
    return [
        "--compatibility-report",
        str(tmp_path / "compatibility.json"),
        "--action-layout",
        str(tmp_path / "layout.json"),
    ]


def test_public_collection_rejects_zero_fresh_state_trajectories(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "collect-maniskill-pickcube-sequences",
                "--runtime-archive-dir",
                str(tmp_path / "states"),
                "--reference-archive-dir",
                str(tmp_path / "references"),
                "--summary-output",
                str(tmp_path / "summary.json"),
                "--fresh-state-trajectory-limit",
                "0",
                "--dry-run",
                *_common_pickcube_args(tmp_path),
            ]
        )


def test_production_collection_uses_private_nonrecursive_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_ISOLATED_COLLECTION_WORKER_ENV, raising=False)
    assert _use_isolated_sequence_collection(
        internal_worker=False,
        environment_factory=None,
        solver=None,
        solver_identity=None,
    )
    assert not _use_isolated_sequence_collection(
        internal_worker=True,
        environment_factory=None,
        solver=None,
        solver_identity=None,
    )

    args = SimpleNamespace(
        action_layout=tmp_path / "layout.json",
        compatibility_report=tmp_path / "compatibility.json",
        expected_contract=tmp_path / "expected.json",
        sim_backend="gpu",
        trajectory_action_limit=123,
    )
    command = _isolated_sequence_worker_command(
        args, seed=20_000, output_root=tmp_path / "worker"
    )
    assert command[:5] == (
        command[0],
        "-X",
        "faulthandler",
        "-m",
        "latentguard",
    )
    assert command[command.index("--requested-success-count") + 1] == "1"
    assert command[command.index("--starting-seed") + 1] == "20000"
    assert command[command.index("--maximum-attempts") + 1] == "1"
    assert command[command.index("--fresh-state-trajectory-limit") + 1] == "1"
    assert command[command.index("--trajectory-action-limit") + 1] == "123"

    (tmp_path / "worker").mkdir()
    request = _write_isolated_collection_worker_request(
        args, seed=20_000, output_root=tmp_path / "worker"
    )
    monkeypatch.setenv(_ISOLATED_COLLECTION_WORKER_ENV, "1")
    monkeypatch.setenv(_ISOLATED_COLLECTION_REQUEST_ENV, str(request))
    child_args = SimpleNamespace(
        action_layout=tmp_path / "layout.json",
        compatibility_report=tmp_path / "compatibility.json",
        expected_contract=tmp_path / "expected.json",
        fresh_state_trajectory_limit=1,
        maximum_attempts=1,
        reference_archive_dir=tmp_path / "worker" / "reference-archive",
        requested_success_count=1,
        runtime_archive_dir=tmp_path / "worker" / "runtime-archive",
        sim_backend="gpu",
        starting_seed=20_000,
        summary_output=tmp_path / "worker" / "summary.json",
        trajectory_action_limit=123,
    )
    assert _validate_internal_collection_worker(child_args)

    request_payload = json.loads(request.read_text(encoding="utf-8"))
    request_payload["trajectory_action_limit"] = True
    request.write_text(json.dumps(request_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="request differs from arguments"):
        _validate_internal_collection_worker(child_args)

    request_payload["trajectory_action_limit"] = 123
    request.write_text(json.dumps(request_payload), encoding="utf-8")
    monkeypatch.delenv(_ISOLATED_COLLECTION_REQUEST_ENV)
    with pytest.raises(ValueError, match="marker is incomplete"):
        _validate_internal_collection_worker(child_args)


def test_isolated_worker_accepts_only_strict_normal_seed_rejection(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "accepted_source_trajectories": 0,
                "attempt_count": 1,
                "attempt_failure_categories": {"PickCubeSourceGenerationError": 1},
                "requested_source_trajectories": 1,
                "schema_version": "1.1",
                "status": "incomplete",
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.CompletedProcess(args=("worker",), returncode=1)
    assert (
        _isolated_worker_rejection_category(
            completed, summary_path=summary, seed=20_000
        )
        == "PickCubeSourceGenerationError"
    )

    with pytest.raises(RuntimeError, match="signal 11"):
        _isolated_worker_rejection_category(
            subprocess.CompletedProcess(args=("worker",), returncode=-11),
            summary_path=summary,
            seed=20_000,
        )
    summary.unlink()
    with pytest.raises(RuntimeError, match="missing or invalid"):
        _isolated_worker_rejection_category(
            completed, summary_path=summary, seed=20_000
        )


def test_isolated_collection_preserves_global_attempt_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(
        action_layout=tmp_path / "layout.json",
        compatibility_report=tmp_path / "compatibility.json",
        expected_contract=tmp_path / "expected.json",
        maximum_attempts=3,
        requested_success_count=2,
        sim_backend="gpu",
        starting_seed=10,
        trajectory_action_limit=None,
    )
    outcomes = iter((1, 0, 0))
    monkeypatch.setattr(
        m3a_cli.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=("worker",), returncode=next(outcomes)
        ),
    )
    monkeypatch.setattr(
        m3a_cli,
        "_isolated_worker_rejection_category",
        lambda *args, **kwargs: "NormalSeedRejection",
    )
    monkeypatch.setattr(
        m3a_cli,
        "load_state_indexed_archive",
        lambda path: SimpleNamespace(episodes=(object(),)),
    )
    monkeypatch.setattr(
        m3a_cli,
        "load_reference_archive",
        lambda path: SimpleNamespace(episodes=(object(),)),
    )
    monkeypatch.setattr(
        m3a_cli,
        "_validate_isolated_worker_success",
        lambda **kwargs: (
            f"episode-{kwargs['seed']}",
            f"reference-{kwargs['seed']}",
        ),
    )
    monkeypatch.setattr(
        m3a_cli,
        "PickCubeStateIndexedArchiveV1",
        lambda *, episodes: SimpleNamespace(episodes=episodes),
    )
    monkeypatch.setattr(
        m3a_cli,
        "ManiSkillReferenceArchive",
        lambda *, episodes: SimpleNamespace(episodes=episodes),
    )

    result = _collect_state_indexed_reference_archive_isolated(args)

    assert tuple(attempt.seed for attempt in result.attempts) == (10, 11, 12)
    assert tuple(attempt.accepted for attempt in result.attempts) == (
        False,
        True,
        True,
    )
    assert result.archive.episodes == ("episode-11", "episode-12")
    assert result.reference_archive.episodes == (
        "reference-11",
        "reference-12",
    )


def test_isolated_collection_aborts_on_native_worker_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = SimpleNamespace(
        action_layout=tmp_path / "layout.json",
        compatibility_report=tmp_path / "compatibility.json",
        expected_contract=tmp_path / "expected.json",
        maximum_attempts=1,
        requested_success_count=1,
        sim_backend="gpu",
        starting_seed=20_000,
        trajectory_action_limit=None,
    )
    monkeypatch.setattr(
        m3a_cli.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=("worker",), returncode=-11
        ),
    )

    with pytest.raises(RuntimeError, match="signal 11"):
        _collect_state_indexed_reference_archive_isolated(args)


def test_isolated_worker_success_cross_checks_complete_pair(tmp_path: Path) -> None:
    digest = f"sha256:{'a' * 64}"
    initial = f"sha256:{'b' * 64}"
    terminal = f"sha256:{'c' * 64}"
    actions = np.arange(24, dtype=np.float64).reshape(3, 8)
    episode = SimpleNamespace(
        seed=20_000,
        compatibility_identity=digest,
        source_trajectory_id="trajectory-20000",
        episode_id="episode-20000",
        source_actions=actions,
        states=(
            SimpleNamespace(state_digest=initial),
            SimpleNamespace(state_digest=terminal),
        ),
    )
    reference = SimpleNamespace(
        seed=20_000,
        compatibility_identity=digest,
        source_trajectory_id="trajectory-20000",
        episode_id="episode-20000",
        source_actions=actions.copy(),
        initial_state_digest=initial,
        terminal_state_digest=terminal,
    )
    archive = SimpleNamespace(episodes=(episode,), content_digest=digest)
    reference_archive = SimpleNamespace(episodes=(reference,))
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "accepted_source_trajectories": 1,
                "archive_content_digest": digest,
                "attempt_count": 1,
                "attempt_failure_categories": {},
                "compatibility_identity": digest,
                "fresh_state_compared_component_counts": [],
                "fresh_state_maximum_absolute_error": 0.0,
                "fresh_state_restoration_error_distribution": {
                    "count": 0,
                    "fixed_bin_counts": {
                        "equal_zero": 0,
                        "gt_1e-6": 0,
                        "gt_1e-7_le_5e-7": 0,
                        "gt_1e-8_le_1e-7": 0,
                        "gt_5e-7_le_1e-6": 0,
                        "gt_zero_le_1e-8": 0,
                    },
                    "maximum": None,
                    "minimum": None,
                    "p50": None,
                    "p95": None,
                    "p99": None,
                    "quantile_semantic": "nearest_rank_v1",
                },
                "fresh_state_verification_count": 0,
                "fresh_verifier_state_compared_component_counts": [],
                "fresh_verifier_state_maximum_absolute_error": 0.0,
                "fresh_verifier_state_restoration_error_distribution": {
                    "count": 0,
                    "fixed_bin_counts": {
                        "equal_zero": 0,
                        "gt_1e-6": 0,
                        "gt_1e-7_le_5e-7": 0,
                        "gt_1e-8_le_1e-7": 0,
                        "gt_5e-7_le_1e-6": 0,
                        "gt_zero_le_1e-8": 0,
                    },
                    "maximum": None,
                    "minimum": None,
                    "p50": None,
                    "p95": None,
                    "p99": None,
                    "quantile_semantic": "nearest_rank_v1",
                },
                "requested_source_trajectories": 1,
                "schema_version": "1.1",
                "source_action_count": 3,
                "source_to_restored_task_mismatch_field_counts": {},
                "source_to_restored_task_mismatch_state_count": 0,
                "state_indexed_archive_serialization_version": (
                    STATE_INDEXED_ARCHIVE_VERSION
                ),
                "t_plus_one_state_count": 2,
                "verifier_state_extraction_boundary": (
                    PICKCUBE_VERIFIER_STATE_EXTRACTION_BOUNDARY
                ),
                "verifier_state_semantic": PICKCUBE_VERIFIER_STATE_SEMANTIC,
            }
        ),
        encoding="utf-8",
    )

    observed = _validate_isolated_worker_success(
        summary_path=summary,
        archive=archive,
        reference_archive=reference_archive,
        seed=20_000,
    )
    assert observed == (episode, reference)

    reference.source_actions[0, 0] = -1.0
    with pytest.raises(RuntimeError, match="paired episode identity"):
        _validate_isolated_worker_success(
            summary_path=summary,
            archive=archive,
            reference_archive=reference_archive,
            seed=20_000,
        )


def test_isolated_fresh_audit_records_are_strictly_content_bound(
    tmp_path: Path,
) -> None:
    archive_digest = f"sha256:{'a' * 64}"
    episode_digest = f"sha256:{'b' * 64}"
    compatibility = f"sha256:{'c' * 64}"
    state = SimpleNamespace(
        state_index=0,
        numeric_component_count=70,
        task_snapshot=SimpleNamespace(
            success=False,
            is_obj_placed=False,
            is_robot_static=False,
            is_grasped=False,
            cube_center_z=0.02,
            cube_to_goal_distance=0.2,
            tcp_to_cube_distance=0.1,
        ),
        restored_task_snapshot=SimpleNamespace(
            success=False,
            is_obj_placed=False,
            is_robot_static=False,
            is_grasped=True,
            cube_center_z=0.02,
            cube_to_goal_distance=0.2,
            tcp_to_cube_distance=0.1,
        ),
        verifier_state=SimpleNamespace(values=np.zeros(38, dtype=np.float32)),
    )
    episode = SimpleNamespace(
        compatibility_identity=compatibility,
        content_digest=episode_digest,
        source_trajectory_id="trajectory-20000",
        states=(state,),
    )
    archive = SimpleNamespace(content_digest=archive_digest)
    output = tmp_path / "audit.json"
    payload = {
        "archive_content_digest": archive_digest,
        "compatibility_identity": compatibility,
        "episode_content_digest": episode_digest,
        "records": [
            {
                "compared_component_count": 70,
                "maximum_absolute_error": 1.1920928955078125e-07,
                "source_to_restored_task_mismatch_fields": ["is_grasped"],
                "source_trajectory_id": "trajectory-20000",
                "state_index": 0,
                "verifier_component_count": 38,
                "verifier_maximum_absolute_error": 0.0,
            }
        ],
        "schema_version": "1.0",
        "source_trajectory_id": "trajectory-20000",
        "state_count": 1,
    }
    output.write_text(json.dumps(payload), encoding="utf-8")

    records = _load_isolated_fresh_audit_records(
        output,
        archive=archive,
        episode=episode,
        tolerance=1e-6,
    )
    assert len(records) == 1
    assert records[0].source_to_restored_task_mismatch_fields == ("is_grasped",)

    payload["unexpected"] = True
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="envelope identity"):
        _load_isolated_fresh_audit_records(
            output,
            archive=archive,
            episode=episode,
            tolerance=1e-6,
        )

    args = SimpleNamespace(
        action_layout=tmp_path / "layout.json",
        compatibility_report=tmp_path / "compatibility.json",
        expected_contract=tmp_path / "expected.json",
    )
    command = _isolated_fresh_audit_worker_command(
        args,
        runtime_archive_dir=tmp_path / "runtime",
        archive_digest=archive_digest,
        source_trajectory_id="trajectory-20000",
        output=output,
    )
    assert command[command.index("-m") + 1].endswith("fresh_audit_worker")
    assert command[command.index("--expected-archive-digest") + 1] == archive_digest


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


def test_collection_summary_records_restored_projection_evidence() -> None:
    archive = SimpleNamespace(
        content_digest="sha256:archive",
        episodes=(
            SimpleNamespace(
                compatibility_identity="sha256:compatibility",
                source_actions=np.zeros((3, 8), dtype=np.float32),
                states=(object(), object(), object(), object()),
            ),
        ),
    )
    result = SimpleNamespace(
        accepted_count=1,
        attempts=(
            SimpleNamespace(failure_category=None),
            SimpleNamespace(failure_category="solver_failure"),
        ),
        requested_success_count=1,
    )
    audit_records = (
        SimpleNamespace(
            compared_component_count=70,
            maximum_absolute_error=1.0e-7,
            verifier_component_count=24,
            verifier_maximum_absolute_error=2.0e-7,
            source_to_restored_task_mismatch_fields=("is_grasped",),
        ),
        SimpleNamespace(
            compared_component_count=70,
            maximum_absolute_error=3.0e-7,
            verifier_component_count=24,
            verifier_maximum_absolute_error=1.0e-7,
            source_to_restored_task_mismatch_fields=(
                "is_grasped",
                "tcp_to_cube_distance",
            ),
        ),
    )

    summary = _collection_summary(
        result=result,
        archive=archive,
        audit_records=audit_records,
    )

    assert summary["schema_version"] == "1.1"
    assert (
        summary["state_indexed_archive_serialization_version"]
        == STATE_INDEXED_ARCHIVE_VERSION
    )
    assert (
        summary["verifier_state_extraction_boundary"]
        == PICKCUBE_VERIFIER_STATE_EXTRACTION_BOUNDARY
    )
    assert summary["verifier_state_semantic"] == PICKCUBE_VERIFIER_STATE_SEMANTIC
    assert summary["fresh_state_verification_count"] == 2
    assert summary["fresh_state_compared_component_counts"] == [70]
    assert summary["fresh_state_maximum_absolute_error"] == 3.0e-7
    assert summary["fresh_state_restoration_error_distribution"]["count"] == 2
    assert summary["fresh_state_restoration_error_distribution"]["p95"] == 3.0e-7
    assert summary["fresh_verifier_state_compared_component_counts"] == [24]
    assert summary["fresh_verifier_state_maximum_absolute_error"] == 2.0e-7
    assert (
        summary["fresh_verifier_state_restoration_error_distribution"]["maximum"]
        == 2.0e-7
    )
    assert summary["source_to_restored_task_mismatch_state_count"] == 2
    assert summary["source_to_restored_task_mismatch_field_counts"] == {
        "is_grasped": 2,
        "tcp_to_cube_distance": 1,
    }
    assert summary["accepted_source_trajectories"] == 1
    assert summary["requested_source_trajectories"] == 1
    assert summary["attempt_count"] == 2
    assert summary["attempt_failure_categories"] == {"solver_failure": 1}
    assert summary["source_action_count"] == 3
    assert summary["t_plus_one_state_count"] == 4


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

    report = _restoration_report(*inputs)
    assert report[:5] == ([70], 1.0e-7, 1, [32], 5.0e-8)
    assert report[5]["count"] == 3
    assert report[5]["minimum"] == 1.0e-7
    assert report[5]["p50"] == 1.0e-7
    assert report[6]["count"] == 1
    assert report[6]["maximum"] == 5.0e-8


def test_restoration_error_distribution_has_fixed_boundary_bins() -> None:
    report = _restoration_error_distribution(
        [0.0, 1.0e-9, 1.0e-8, 2.0e-8, 1.0e-7, 2.0e-7, 5.0e-7, 8.0e-7, 1.0e-6, 2.0e-6]
    )

    assert report["count"] == 10
    assert report["minimum"] == 0.0
    assert report["maximum"] == 2.0e-6
    assert report["p50"] == 1.0e-7
    assert report["p95"] == 2.0e-6
    assert report["fixed_bin_counts"] == {
        "equal_zero": 1,
        "gt_zero_le_1e-8": 2,
        "gt_1e-8_le_1e-7": 2,
        "gt_1e-7_le_5e-7": 2,
        "gt_5e-7_le_1e-6": 2,
        "gt_1e-6": 1,
    }


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

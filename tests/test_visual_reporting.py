from __future__ import annotations

from pathlib import Path

import pytest

from latentguard.training.reporting import StrictReportV1
from latentguard.vision_data.configuration import (
    load_camera_rig_configuration,
    load_render_domain_configuration,
)
from latentguard.vision_data.leakage import CrossDatasetLeakageReportV1
from latentguard.vision_data.rendering import ZeroWorkRenderResumeV1
from latentguard.vision_data.reporting import (
    M4A_RENDER_RESUME_REPORT_FILENAME,
    VisualReportingError,
    build_camera_rig_summary_report,
    build_leakage_report,
    build_render_determinism_report,
    build_render_domain_summary_report,
    build_render_resume_report,
    build_state_integrity_report,
    load_bound_render_resume_report,
    load_strict_report,
    persist_render_resume_report,
    publish_visual_validation_reports,
)

_CONFIG_ROOT = Path("configs/vision/m4a")


def _determinism_report(**changes: object) -> StrictReportV1:
    values: dict[str, object] = {
        "visual_compatibility_identity": "sha256:" + "a" * 64,
        "repeated_changed_pixel_counts": (0, 0, 0),
        "repeated_maximum_channel_differences": (0, 0, 0),
        "repeated_mean_absolute_differences": (0.0, 0.0, 0.0),
        "fresh_environment_changed_pixel_counts": (0, 0, 0),
        "fresh_environment_maximum_channel_differences": (0, 0, 0),
        "fresh_environment_mean_absolute_differences": (0.0, 0.0, 0.0),
        "spatially_stable": True,
    }
    values.update(changes)
    return build_render_determinism_report(**values)  # type: ignore[arg-type]


def _state_report(**changes: object) -> StrictReportV1:
    values: dict[str, object] = {
        "visual_dataset_digest": "sha256:" + "b" * 64,
        "packet_manifest_digest": "sha256:" + "c" * 64,
        "packet_count": 2,
        "compared_component_counts": (70, 70),
        "maximum_absolute_errors": (0.0, 1e-7),
        "verifier_component_counts": (38, 38),
        "verifier_maximum_absolute_errors": (0.0, 0.0),
        "task_projection_changes": 0,
        "physics_step_count": 0,
        "environment_close_failure_count": 0,
    }
    values.update(changes)
    return build_state_integrity_report(**values)  # type: ignore[arg-type]


def test_render_determinism_report_requires_complete_three_view_inventories() -> None:
    report = _determinism_report()
    assert report.payload["trusted_generation_permitted"] is True

    for field in (
        "repeated_changed_pixel_counts",
        "repeated_maximum_channel_differences",
        "repeated_mean_absolute_differences",
        "fresh_environment_changed_pixel_counts",
        "fresh_environment_maximum_channel_differences",
        "fresh_environment_mean_absolute_differences",
    ):
        with pytest.raises(VisualReportingError):
            _determinism_report(**{field: (0,)})


def test_render_determinism_report_rejects_invalid_metric_types_and_ranges() -> None:
    with pytest.raises(VisualReportingError, match="changed pixel counts"):
        _determinism_report(repeated_changed_pixel_counts=(0, True, 0))
    with pytest.raises(VisualReportingError, match="cannot exceed 255"):
        _determinism_report(
            fresh_environment_mean_absolute_differences=(0.0, 256.0, 0.0)
        )
    with pytest.raises(VisualReportingError, match="must be boolean"):
        _determinism_report(spatially_stable=1)


def test_frozen_camera_and_domain_summaries_are_compact_and_content_bound() -> None:
    rig = load_camera_rig_configuration(_CONFIG_ROOT / "camera-rig-v1.json")
    domains = load_render_domain_configuration(_CONFIG_ROOT / "render-domains-v1.json")

    rig_report = build_camera_rig_summary_report(rig)
    domain_report = build_render_domain_summary_report(domains)

    assert rig_report.payload["rig_digest"] == rig.rig_digest
    assert rig_report.payload["camera_count"] == 3
    assert [item["camera_id"] for item in rig_report.payload["cameras"]] == [  # type: ignore[index,union-attr]
        "front_oblique",
        "overhead",
        "side_oblique",
    ]
    assert domain_report.payload["configuration_digest"] == domains.content_digest
    assert domain_report.payload["domain_count"] == 5
    assert domain_report.payload["seed_derivation"] == domains.seed_derivation


def test_state_integrity_report_requires_exact_per_packet_coverage() -> None:
    report = _state_report()
    assert report.payload["passed"] is True
    assert report.payload["complete_state_error_statistics"] == {
        "count": 2,
        "maximum": 1e-7,
        "median": 5e-8,
        "minimum": 0.0,
        "p95": 1e-7,
        "percentile_semantic": "nearest_rank_v1",
        "zero_count": 1,
    }
    assert report.payload["verifier_state_error_statistics"] == {
        "count": 2,
        "maximum": 0.0,
        "median": 0.0,
        "minimum": 0.0,
        "p95": 0.0,
        "percentile_semantic": "nearest_rank_v1",
        "zero_count": 2,
    }

    for field in (
        "compared_component_counts",
        "maximum_absolute_errors",
        "verifier_component_counts",
        "verifier_maximum_absolute_errors",
    ):
        with pytest.raises(VisualReportingError, match="exactly cover"):
            _state_report(**{field: (0,)})
        with pytest.raises(VisualReportingError, match="exactly cover"):
            _state_report(**{field: (0, 0, 0)})


def test_state_integrity_report_records_complete_but_failed_evidence() -> None:
    report = _state_report(maximum_absolute_errors=(0.0, 2e-6))
    assert report.payload["passed"] is False

    with pytest.raises(VisualReportingError, match="positive integer"):
        _state_report(packet_count=True)
    with pytest.raises(VisualReportingError, match="non-negative integers"):
        _state_report(compared_component_counts=(70, True))
    report = _state_report(environment_close_failure_count=1)
    assert report.payload["passed"] is False


def test_zero_work_resume_report_is_immutable_and_bound_to_live_run(
    tmp_path: Path,
) -> None:
    root = tmp_path / "render-root"
    root.mkdir()
    proof = ZeroWorkRenderResumeV1(
        run_id="m4a-render-run",
        packet_count=3,
        ledger_entry_count=3,
        ledger_content_digest="sha256:" + "1" * 64,
    )
    values = {
        "source_identity_digest": "sha256:" + "2" * 64,
        "camera_rig_digest": "sha256:" + "3" * 64,
        "render_domain_configuration_digest": "sha256:" + "4" * 64,
    }
    report = persist_render_resume_report(proof, root, **values)
    path = root / M4A_RENDER_RESUME_REPORT_FILENAME
    before = path.read_bytes()

    assert persist_render_resume_report(proof, root, **values) == report
    assert path.read_bytes() == before
    assert (
        load_bound_render_resume_report(
            root,
            run_id=proof.run_id,
            ledger_content_digest=proof.ledger_content_digest,
            packet_count=proof.packet_count,
            ledger_entry_count=proof.ledger_entry_count,
            **values,
        )
        == report
    )
    with pytest.raises(VisualReportingError, match="current immutable run"):
        load_bound_render_resume_report(
            root,
            run_id=proof.run_id,
            source_identity_digest="sha256:" + "5" * 64,
            camera_rig_digest=values["camera_rig_digest"],
            render_domain_configuration_digest=values[
                "render_domain_configuration_digest"
            ],
            ledger_content_digest=proof.ledger_content_digest,
            packet_count=proof.packet_count,
            ledger_entry_count=proof.ledger_entry_count,
        )


def test_resume_report_rejects_nonzero_or_unsafe_proof() -> None:
    values = {
        "source_identity_digest": "sha256:" + "2" * 64,
        "camera_rig_digest": "sha256:" + "3" * 64,
        "render_domain_configuration_digest": "sha256:" + "4" * 64,
    }
    with pytest.raises(VisualReportingError, match="counts or run identity"):
        build_render_resume_report(
            ZeroWorkRenderResumeV1(
                run_id="/private/render-run",
                packet_count=3,
                ledger_entry_count=3,
                ledger_content_digest="sha256:" + "1" * 64,
            ),
            **values,
        )
    with pytest.raises(VisualReportingError, match="counts or run identity"):
        build_render_resume_report(
            ZeroWorkRenderResumeV1(
                run_id="m4a-render-run",
                packet_count=3,
                ledger_entry_count=3,
                ledger_content_digest="sha256:" + "1" * 64,
                rendered_packets=1,
            ),
            **values,
        )


def _development_summary_report() -> StrictReportV1:
    return StrictReportV1(
        report_type="m4a_development_visual_dataset_summary_v1",
        payload={
            "dataset_content_digest": "sha256:" + "d" * 64,
            "schema_version": "1.0",
        },
    )


def test_validation_report_publication_is_atomic_strict_and_sanitized(
    tmp_path: Path,
) -> None:
    output = tmp_path / "compact-reports"
    reports = {
        "development-dataset-summary.json": _development_summary_report(),
        "development-state-integrity.json": _state_report(),
    }

    published = publish_visual_validation_reports(reports, output)
    assert set(published) == {
        "compact-retrieval-manifest.json",
        *reports,
    }
    assert {
        item.name
        for item in output.iterdir()
        if item.is_file() and not item.is_symlink()
    } == set(published)
    assert all(item.suffix == ".json" for item in output.iterdir())
    retrieval = load_strict_report(
        output / "compact-retrieval-manifest.json",
        expected_report_type="m4a_compact_retrieval_manifest_v1",
    )
    assert retrieval.payload["raw_actions_included"] is False
    assert retrieval.payload["raw_images_included"] is False
    assert retrieval.payload["raw_state_archives_included"] is False
    assert retrieval.payload["sanitized_json_only"] is True
    assert retrieval.payload["report_file_count"] == 2
    assert set(retrieval.payload["report_content_digests"]) == set(reports)  # type: ignore[arg-type]

    with pytest.raises(VisualReportingError, match="must be absent"):
        publish_visual_validation_reports(reports, output)


def test_validation_report_publication_cleans_staging_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import latentguard.vision_data.reporting as reporting

    output = tmp_path / "failed-reports"
    reports = {
        "development-dataset-summary.json": _development_summary_report(),
        "development-state-integrity.json": _state_report(),
    }
    original = reporting.save_strict_report
    calls = 0

    def fail_second(report: StrictReportV1, path: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic publication failure")
        return original(report, path)

    monkeypatch.setattr(reporting, "save_strict_report", fail_second)
    with pytest.raises(OSError, match="synthetic publication failure"):
        publish_visual_validation_reports(reports, output)
    assert not output.exists()
    assert not tuple(tmp_path.glob(f".{output.name}.*.tmp"))


def test_validation_report_publication_rejects_linked_ancestor_before_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import latentguard.vision_data.reporting as reporting
    import latentguard.vision_data.serialization as serialization

    linked_ancestor = tmp_path / "linked-ancestor"
    output = linked_ancestor / "nested" / "compact-reports"
    real_link_guard = serialization._is_link_or_junction
    monkeypatch.setattr(
        serialization,
        "_is_link_or_junction",
        lambda path: Path(path) == linked_ancestor or real_link_guard(Path(path)),
    )

    with pytest.raises(
        VisualReportingError, match="cannot traverse a link or junction"
    ):
        reporting.publish_visual_validation_reports(
            {"development-state-integrity.json": _state_report()},
            output,
        )
    assert not linked_ancestor.exists()


def test_leakage_report_preserves_repeated_image_inventories() -> None:
    report = build_leakage_report(
        CrossDatasetLeakageReportV1(
            overlapping_reset_seeds=(),
            overlapping_source_trajectory_ids=(),
            overlapping_split_group_ids=(),
            overlapping_anchor_ids=(),
            overlapping_state_digests=(),
            overlapping_verifier_state_digests=(),
            overlapping_packet_ids=(),
            overlapping_candidate_ids=(),
            overlapping_image_digests=(),
            constant_image_digests_requiring_review=(),
            repeated_images_within_development=("sha256:" + "e" * 64,),
            repeated_images_within_external=("sha256:" + "f" * 64,),
        )
    )
    assert report.payload["repeated_images_within_development"] == [
        "sha256:" + "e" * 64
    ]
    assert report.payload["repeated_images_within_external"] == ["sha256:" + "f" * 64]

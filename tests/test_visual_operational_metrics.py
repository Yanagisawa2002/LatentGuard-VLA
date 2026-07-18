from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from latentguard.training.reporting import ReportValidationError
from latentguard.vision_data.operational_metrics import (
    M4AOperationalMetricsError,
    M4ARenderOperationalMetricsV1,
    build_operational_cost_summary_report,
    load_render_operational_metrics,
    persist_render_operational_metrics,
)


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _packet_id(index: int) -> str:
    return "vop-sha256-" + hashlib.sha256(f"packet-{index}".encode()).hexdigest()


_PACKET_IDS = tuple(_packet_id(index) for index in range(5))


def _render_metrics() -> M4ARenderOperationalMetricsV1:
    return M4ARenderOperationalMetricsV1(
        invocation_kind="render",
        run_manifest_digest=_digest("run-manifest"),
        render_job_inventory_digest=_digest("jobs"),
        selected_packet_inventory_digest=_digest("selected"),
        ledger_content_digest=_digest("ledger-render"),
        environment_initialization_count=2,
        packet_count=5,
        image_inventory_count=15,
        rendered_packet_count=5,
        images_rendered_count=15,
        resume_reused_packet_count=0,
        rendered_packet_ids=_PACKET_IDS,
        resume_reused_packet_ids=(),
        packet_render_durations_ns=tuple(
            zip(_PACKET_IDS, (10, 20, 30, 40, 100), strict=True)
        ),
        peak_gpu_memory_bytes=4096,
        peak_gpu_memory_source="torch_cuda_max_memory_allocated_v1",
        server_active_execution_duration_ns=250,
        server_active_duration_source="python_perf_counter_ns_v1",
    )


def _zero_work_metrics() -> M4ARenderOperationalMetricsV1:
    return M4ARenderOperationalMetricsV1(
        invocation_kind="zero_work_resume",
        run_manifest_digest=_digest("run-manifest"),
        render_job_inventory_digest=_digest("jobs"),
        selected_packet_inventory_digest=_digest("selected"),
        ledger_content_digest=_digest("ledger-resume"),
        environment_initialization_count=0,
        packet_count=5,
        image_inventory_count=15,
        rendered_packet_count=0,
        images_rendered_count=0,
        resume_reused_packet_count=5,
        rendered_packet_ids=(),
        resume_reused_packet_ids=_PACKET_IDS,
        packet_render_durations_ns=(),
        peak_gpu_memory_bytes=None,
        peak_gpu_memory_source=None,
        server_active_execution_duration_ns=50,
        server_active_duration_source="python_perf_counter_ns_v1",
    )


def test_operational_metrics_use_fixed_nearest_rank_statistics() -> None:
    report = _render_metrics().to_report()
    timing = report.payload["packet_render_time"]
    assert isinstance(timing, dict)
    assert timing == {
        "average_ns": 40.0,
        "count": 5,
        "maximum_ns": 100,
        "minimum_ns": 10,
        "p50_ns": 30,
        "p95_ns": 100,
        "percentile_semantic": "nearest_rank_v1",
        "total_ns": 200,
    }
    zero = _zero_work_metrics().to_report()
    assert zero.payload["packet_render_time"] is None
    assert zero.payload["images_rendered_count"] == 0


def test_operational_metrics_invalid_construction_is_rejected() -> None:
    mutations = (
        {"packet_count": True},
        {"image_inventory_count": 14},
        {"packet_render_durations_ns": ((_PACKET_IDS[0], 10),)},
        {"environment_initialization_count": 0},
        {"peak_gpu_memory_source": None},
        {"server_active_execution_duration_ns": -1},
    )
    for changes in mutations:
        with pytest.raises(M4AOperationalMetricsError):
            replace(_render_metrics(), **changes)
    with pytest.raises(M4AOperationalMetricsError):
        replace(_zero_work_metrics(), rendered_packet_count=1)


def test_operational_metrics_append_reload_tamper_and_summary(tmp_path: Path) -> None:
    root = tmp_path / "render-root"
    root.mkdir()
    first = persist_render_operational_metrics(_render_metrics(), root)
    second = persist_render_operational_metrics(_zero_work_metrics(), root)
    reports = load_render_operational_metrics(root)
    assert tuple(report.content_digest for report in reports) == (
        first.content_digest,
        second.content_digest,
    )
    assert [path.name for path in (root / "operational-metrics").iterdir()] == [
        "000001.json",
        "000002.json",
    ]

    summary = build_operational_cost_summary_report(
        reports,
        validated_packet_count=5,
        validated_image_count=15,
        validation_duration_ns=75,
    )
    assert summary.payload["environment_initialization_count"] == 2
    assert summary.payload["rendered_packet_count"] == 5
    assert summary.payload["images_rendered_count"] == 15
    assert summary.payload["resume_reused_packet_count"] == 5
    assert summary.payload["total_rendering_time_ns"] == 200
    assert summary.payload["timing_complete"] is True
    assert summary.payload["unmeasured_preexisting_or_reused_packet_count"] == 0
    assert summary.payload["validation_time_ns"] == 75
    assert summary.payload["peak_gpu_memory_bytes"] == 4096
    assert summary.payload["peak_gpu_memory_complete"] is False
    assert summary.payload["total_server_active_duration_ns"] is None
    assert summary.payload["measured_server_active_duration_ns"] == 375

    first_path = root / "operational-metrics" / "000001.json"
    raw = json.loads(first_path.read_text(encoding="utf-8"))
    raw["payload"]["packet_count"] = 6
    first_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((M4AOperationalMetricsError, ReportValidationError)):
        load_render_operational_metrics(root)


def test_operational_metrics_reject_noncontiguous_or_unknown_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "render-root"
    root.mkdir()
    persist_render_operational_metrics(_render_metrics(), root)
    (root / "operational-metrics" / "unexpected.txt").write_text(
        "not evidence", encoding="utf-8"
    )
    with pytest.raises(M4AOperationalMetricsError, match="contiguous and exact"):
        load_render_operational_metrics(root)


def test_interrupted_render_resume_reports_unknown_cost_without_rerender(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resumed-render"
    root.mkdir()
    resumed = replace(
        _render_metrics(),
        invocation_kind="render_resume",
        ledger_content_digest=_digest("ledger-resumed"),
        environment_initialization_count=3,
        rendered_packet_count=3,
        images_rendered_count=9,
        resume_reused_packet_count=2,
        rendered_packet_ids=_PACKET_IDS[2:],
        resume_reused_packet_ids=_PACKET_IDS[:2],
        packet_render_durations_ns=tuple(
            zip(_PACKET_IDS[2:], (20, 30, 40), strict=True)
        ),
    )
    final = replace(
        _zero_work_metrics(),
        ledger_content_digest=_digest("ledger-resumed"),
    )
    persist_render_operational_metrics(resumed, root)
    persist_render_operational_metrics(final, root)
    reports = load_render_operational_metrics(
        root,
        expected_run_manifest_digest=_digest("run-manifest"),
        expected_render_job_inventory_digest=_digest("jobs"),
        expected_selected_packet_inventory_digest=_digest("selected"),
        expected_final_ledger_content_digest=_digest("ledger-resumed"),
        expected_packet_ids=_PACKET_IDS,
    )
    summary = build_operational_cost_summary_report(
        reports,
        validated_packet_count=5,
        validated_image_count=15,
        validation_duration_ns=10,
    )
    assert summary.payload["timing_complete"] is False
    assert summary.payload["measured_rendered_packet_count"] == 3
    assert summary.payload["unmeasured_preexisting_or_reused_packet_count"] == 2
    assert summary.payload["total_rendering_time_available"] is False
    assert summary.payload["total_rendering_time_ns"] is None
    retry = replace(
        final,
        server_active_execution_duration_ns=75,
    )
    retried = persist_render_operational_metrics(retry, root)
    assert (
        retried.content_digest
        == load_render_operational_metrics(root)[-1].content_digest
    )
    assert [path.name for path in (root / "operational-metrics").iterdir()] == [
        "000001.json",
        "000002.json",
    ]
    with pytest.raises(M4AOperationalMetricsError, match="retry binding"):
        persist_render_operational_metrics(
            replace(final, ledger_content_digest=_digest("other-ledger")), root
        )
    live_binding: dict[str, object] = {
        "expected_run_manifest_digest": _digest("run-manifest"),
        "expected_render_job_inventory_digest": _digest("jobs"),
        "expected_selected_packet_inventory_digest": _digest("selected"),
        "expected_final_ledger_content_digest": _digest("ledger-resumed"),
        "expected_packet_ids": _PACKET_IDS,
    }
    binding_mutations = (
        {"expected_run_manifest_digest": _digest("other-run")},
        {"expected_render_job_inventory_digest": _digest("other-jobs")},
        {"expected_selected_packet_inventory_digest": _digest("other-selected")},
        {"expected_final_ledger_content_digest": _digest("other-ledger")},
        {"expected_packet_ids": _PACKET_IDS[:-1]},
    )
    for mutation in binding_mutations:
        with pytest.raises(M4AOperationalMetricsError, match="identity differs"):
            load_render_operational_metrics(root, **(live_binding | mutation))

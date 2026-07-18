"""Reporting-only operational cost evidence for M4A visual execution.

The models in this module bind measurements to operational run evidence.  They
are deliberately absent from packet, render-job, and dataset semantic
identities.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from latentguard.training.reporting import (
    StrictReportV1,
    load_strict_report,
    save_strict_report,
)


class M4AOperationalMetricsError(ValueError):
    """Raised when M4A cost evidence is incomplete, unsafe, or inconsistent."""


M4A_OPERATIONAL_METRICS_DIRECTORY = "operational-metrics"
M4A_RENDER_OPERATIONAL_REPORT_TYPE = "m4a_render_operational_metrics_v1"
M4A_OPERATIONAL_COST_SUMMARY_REPORT_TYPE = "m4a_operational_cost_summary_v1"
_INVOCATION_FILE = re.compile(r"[0-9]{6}\.json\Z")
_PACKET_ID = re.compile(r"vop-sha256-[0-9a-f]{64}\Z")
_INVOCATION_KINDS = frozenset({"render", "render_resume", "zero_work_resume"})
_GPU_MEMORY_SOURCE = "torch_cuda_max_memory_allocated_v1"
_TIMING_CLOCK = "python_perf_counter_ns_v1"
_PERCENTILE_SEMANTIC = "nearest_rank_v1"


def _fail(context: str, reason: str) -> NoReturn:
    raise M4AOperationalMetricsError(f"{context}: {reason}")


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(context, "expected lowercase SHA-256 digest")
    return value


def _count(value: object, context: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        _fail(context, "expected a non-negative integer")
    return value


def _optional_duration(value: object, context: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        _fail(context, "expected null or a non-negative integer nanosecond value")
    return value


def _duration_statistics(values: tuple[int, ...]) -> dict[str, object] | None:
    if not values:
        return None
    ordered = tuple(sorted(values))
    count = len(ordered)

    def nearest_rank(fraction: float) -> int:
        return ordered[math.ceil(fraction * count) - 1]

    return {
        "average_ns": sum(ordered) / count,
        "count": count,
        "maximum_ns": ordered[-1],
        "minimum_ns": ordered[0],
        "p50_ns": nearest_rank(0.50),
        "p95_ns": nearest_rank(0.95),
        "percentile_semantic": _PERCENTILE_SEMANTIC,
        "total_ns": sum(ordered),
    }


@dataclass(slots=True)
class M4AEnvironmentInitializationRecorder:
    """Count successful simulator-environment initializations for one invocation."""

    count: int = 0

    def record_success(self) -> None:
        """Record one environment after its factory returned successfully."""

        self.count += 1


@dataclass(frozen=True, slots=True)
class M4ARenderOperationalMetricsV1:
    """One successful render or resume invocation's non-semantic cost facts."""

    invocation_kind: str
    run_manifest_digest: str
    render_job_inventory_digest: str
    selected_packet_inventory_digest: str
    ledger_content_digest: str
    environment_initialization_count: int
    packet_count: int
    image_inventory_count: int
    rendered_packet_count: int
    images_rendered_count: int
    resume_reused_packet_count: int
    rendered_packet_ids: tuple[str, ...]
    resume_reused_packet_ids: tuple[str, ...]
    packet_render_durations_ns: tuple[tuple[str, int], ...]
    peak_gpu_memory_bytes: int | None
    peak_gpu_memory_source: str | None
    server_active_execution_duration_ns: int | None
    server_active_duration_source: str | None

    def __post_init__(self) -> None:
        """Require exact accounting and valid monotonic/GPU measurements."""

        if self.invocation_kind not in _INVOCATION_KINDS:
            _fail("operational metrics invocation", "unsupported invocation kind")
        for name in (
            "run_manifest_digest",
            "render_job_inventory_digest",
            "selected_packet_inventory_digest",
            "ledger_content_digest",
        ):
            _digest(getattr(self, name), f"operational metrics {name}")
        for name in (
            "environment_initialization_count",
            "packet_count",
            "image_inventory_count",
            "rendered_packet_count",
            "images_rendered_count",
            "resume_reused_packet_count",
        ):
            _count(getattr(self, name), f"operational metrics {name}")
        rendered_ids = self.rendered_packet_ids
        reused_ids = self.resume_reused_packet_ids
        if not isinstance(rendered_ids, tuple) or not isinstance(reused_ids, tuple):
            _fail("operational metrics packet IDs", "expected immutable tuples")
        if any(
            not isinstance(packet_id, str) or _PACKET_ID.fullmatch(packet_id) is None
            for packet_id in (*rendered_ids, *reused_ids)
        ):
            _fail("operational metrics packet IDs", "expected canonical packet IDs")
        if (
            len(rendered_ids) != len(set(rendered_ids))
            or len(reused_ids) != len(set(reused_ids))
            or set(rendered_ids).intersection(reused_ids)
            or len(rendered_ids) != self.rendered_packet_count
            or len(reused_ids) != self.resume_reused_packet_count
        ):
            _fail(
                "operational metrics packet IDs",
                "rendered and reused inventories must be exact and disjoint",
            )
        duration_items = self.packet_render_durations_ns
        if not isinstance(duration_items, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or _PACKET_ID.fullmatch(item[0]) is None
            or type(item[1]) is not int
            or not 0 < item[1] <= 2**63 - 1
            for item in duration_items
        ):
            _fail(
                "operational metrics packet timings",
                "expected positive integer nanoseconds for every new packet",
            )
        duration_ids = tuple(packet_id for packet_id, _ in duration_items)
        if duration_ids != rendered_ids:
            _fail(
                "operational metrics packet timings",
                "timing inventory must exactly cover newly rendered packets",
            )
        if self.image_inventory_count != self.packet_count * 3:
            _fail(
                "operational metrics images",
                "inventory must have three views per packet",
            )
        if self.images_rendered_count != self.rendered_packet_count * 3:
            _fail(
                "operational metrics images",
                "new rendering must have three views per packet",
            )
        if (
            self.rendered_packet_count + self.resume_reused_packet_count
            != self.packet_count
        ):
            _fail(
                "operational metrics packets",
                "rendered and reused packets must be exhaustive",
            )
        if self.rendered_packet_count == 0:
            if self.environment_initialization_count != 0:
                _fail(
                    "operational metrics environment",
                    "zero-work resume initialized an environment",
                )
        elif (
            not 1 <= self.environment_initialization_count <= self.rendered_packet_count
        ):
            _fail(
                "operational metrics environment",
                "successful initializations must reflect bounded worker reuse",
            )
        if self.invocation_kind == "render" and self.resume_reused_packet_count != 0:
            _fail("operational metrics render", "a new run cannot reuse prior packets")
        if self.invocation_kind == "zero_work_resume" and (
            self.rendered_packet_count != 0
            or self.resume_reused_packet_count != self.packet_count
        ):
            _fail("operational metrics resume", "zero-work accounting is inconsistent")
        if (self.peak_gpu_memory_bytes is None) != (
            self.peak_gpu_memory_source is None
        ):
            _fail(
                "operational metrics GPU memory",
                "value and source must be present together",
            )
        if self.peak_gpu_memory_bytes is not None:
            _count(self.peak_gpu_memory_bytes, "operational metrics peak GPU memory")
            if self.peak_gpu_memory_source != _GPU_MEMORY_SOURCE:
                _fail(
                    "operational metrics GPU memory", "unsupported measurement source"
                )
        if (self.server_active_execution_duration_ns is None) != (
            self.server_active_duration_source is None
        ):
            _fail(
                "operational metrics server duration",
                "value and source must be present together",
            )
        _optional_duration(
            self.server_active_execution_duration_ns,
            "operational metrics server duration",
        )
        if (
            self.server_active_duration_source is not None
            and self.server_active_duration_source != _TIMING_CLOCK
        ):
            _fail(
                "operational metrics server duration", "unsupported measurement source"
            )

    def to_report(self) -> StrictReportV1:
        """Return a compact self-digesting report excluded from data identity."""

        durations = tuple(duration for _, duration in self.packet_render_durations_ns)
        statistics = _duration_statistics(durations)
        return StrictReportV1(
            report_type=M4A_RENDER_OPERATIONAL_REPORT_TYPE,
            payload={
                "environment_initialization_count": (
                    self.environment_initialization_count
                ),
                "image_inventory_count": self.image_inventory_count,
                "images_rendered_count": self.images_rendered_count,
                "invocation_kind": self.invocation_kind,
                "ledger_content_digest": self.ledger_content_digest,
                "packet_count": self.packet_count,
                "packet_render_durations": [
                    {"duration_ns": duration, "packet_id": packet_id}
                    for packet_id, duration in self.packet_render_durations_ns
                ],
                "packet_render_time": statistics,
                "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes,
                "peak_gpu_memory_source": self.peak_gpu_memory_source,
                "render_job_inventory_digest": self.render_job_inventory_digest,
                "rendered_packet_count": self.rendered_packet_count,
                "rendered_packet_ids": list(self.rendered_packet_ids),
                "resume_reused_packet_count": self.resume_reused_packet_count,
                "resume_reused_packet_ids": list(self.resume_reused_packet_ids),
                "run_manifest_digest": self.run_manifest_digest,
                "schema_version": "1.0",
                "selected_packet_inventory_digest": (
                    self.selected_packet_inventory_digest
                ),
                "server_active_duration_source": self.server_active_duration_source,
                "server_active_execution_duration_ns": (
                    self.server_active_execution_duration_ns
                ),
                "timing_clock": _TIMING_CLOCK,
            },
        )


def _metrics_from_report(report: StrictReportV1) -> M4ARenderOperationalMetricsV1:
    if not isinstance(report, StrictReportV1) or (
        report.report_type != M4A_RENDER_OPERATIONAL_REPORT_TYPE
    ):
        _fail("operational metrics report", "unexpected report type")
    payload = dict(report.payload)
    expected = {
        "environment_initialization_count",
        "image_inventory_count",
        "images_rendered_count",
        "invocation_kind",
        "ledger_content_digest",
        "packet_count",
        "packet_render_durations",
        "packet_render_time",
        "peak_gpu_memory_bytes",
        "peak_gpu_memory_source",
        "render_job_inventory_digest",
        "rendered_packet_count",
        "rendered_packet_ids",
        "resume_reused_packet_count",
        "resume_reused_packet_ids",
        "run_manifest_digest",
        "schema_version",
        "selected_packet_inventory_digest",
        "server_active_duration_source",
        "server_active_execution_duration_ns",
        "timing_clock",
    }
    if set(payload) != expected or payload["schema_version"] != "1.0":
        _fail("operational metrics report", "unexpected or missing fields")
    if payload["timing_clock"] != _TIMING_CLOCK:
        _fail("operational metrics report", "timing clock changed")
    raw_durations = payload["packet_render_durations"]
    raw_rendered_ids = payload["rendered_packet_ids"]
    raw_reused_ids = payload["resume_reused_packet_ids"]
    if (
        not isinstance(raw_durations, list)
        or not isinstance(raw_rendered_ids, list)
        or not isinstance(raw_reused_ids, list)
        or any(
            not isinstance(item, dict) or set(item) != {"duration_ns", "packet_id"}
            for item in raw_durations
        )
    ):
        _fail("operational metrics report", "packet timings must be a JSON array")
    metrics = M4ARenderOperationalMetricsV1(
        invocation_kind=payload["invocation_kind"],  # type: ignore[arg-type]
        run_manifest_digest=payload["run_manifest_digest"],  # type: ignore[arg-type]
        render_job_inventory_digest=payload["render_job_inventory_digest"],  # type: ignore[arg-type]
        selected_packet_inventory_digest=payload["selected_packet_inventory_digest"],  # type: ignore[arg-type]
        ledger_content_digest=payload["ledger_content_digest"],  # type: ignore[arg-type]
        environment_initialization_count=payload["environment_initialization_count"],  # type: ignore[arg-type]
        packet_count=payload["packet_count"],  # type: ignore[arg-type]
        image_inventory_count=payload["image_inventory_count"],  # type: ignore[arg-type]
        rendered_packet_count=payload["rendered_packet_count"],  # type: ignore[arg-type]
        images_rendered_count=payload["images_rendered_count"],  # type: ignore[arg-type]
        resume_reused_packet_count=payload["resume_reused_packet_count"],  # type: ignore[arg-type]
        rendered_packet_ids=tuple(raw_rendered_ids),
        resume_reused_packet_ids=tuple(raw_reused_ids),
        packet_render_durations_ns=tuple(
            (item["packet_id"], item["duration_ns"]) for item in raw_durations
        ),
        peak_gpu_memory_bytes=payload["peak_gpu_memory_bytes"],  # type: ignore[arg-type]
        peak_gpu_memory_source=payload["peak_gpu_memory_source"],  # type: ignore[arg-type]
        server_active_execution_duration_ns=payload[
            "server_active_execution_duration_ns"
        ],  # type: ignore[arg-type]
        server_active_duration_source=payload["server_active_duration_source"],  # type: ignore[arg-type]
    )
    if dict(metrics.to_report().payload) != payload:
        _fail("operational metrics report", "derived statistics changed")
    return metrics


def load_render_operational_metrics(
    render_root: Path,
    *,
    expected_run_manifest_digest: str | None = None,
    expected_render_job_inventory_digest: str | None = None,
    expected_selected_packet_inventory_digest: str | None = None,
    expected_final_ledger_content_digest: str | None = None,
    expected_packet_ids: tuple[str, ...] | None = None,
) -> tuple[StrictReportV1, ...]:
    """Load append-only reports and optionally bind them to live run evidence."""

    from latentguard.vision_data.serialization import _is_link_or_junction

    root = Path(render_root)
    directory = root / M4A_OPERATIONAL_METRICS_DIRECTORY
    if _is_link_or_junction(root) or not root.is_dir():
        _fail("operational metrics root", "expected a regular non-link render root")
    if not directory.exists() and not _is_link_or_junction(directory):
        return ()
    if _is_link_or_junction(directory) or not directory.is_dir():
        _fail("operational metrics directory", "expected a regular non-link directory")
    entries = tuple(sorted(directory.iterdir(), key=lambda path: path.name))
    expected_names = tuple(f"{index:06d}.json" for index in range(1, len(entries) + 1))
    if tuple(path.name for path in entries) != expected_names:
        _fail("operational metrics directory", "inventory must be contiguous and exact")
    reports: list[StrictReportV1] = []
    for path in entries:
        if (
            _INVOCATION_FILE.fullmatch(path.name) is None
            or _is_link_or_junction(path)
            or not path.is_file()
            or path.stat().st_nlink != 1
        ):
            _fail(
                "operational metrics directory",
                "reports must be regular single-link files",
            )
        report = load_strict_report(
            path, expected_report_type=M4A_RENDER_OPERATIONAL_REPORT_TYPE
        )
        _metrics_from_report(report)
        reports.append(report)
    result = tuple(reports)
    decoded_result = tuple(_metrics_from_report(report) for report in result)
    zero_work_indices = tuple(
        index
        for index, metrics in enumerate(decoded_result)
        if metrics.invocation_kind == "zero_work_resume"
    )
    if len(zero_work_indices) > 1 or (
        zero_work_indices and zero_work_indices[0] != len(decoded_result) - 1
    ):
        _fail(
            "operational metrics sequence",
            "at most one zero-work invocation is allowed and it must be final",
        )
    bindings = (
        expected_run_manifest_digest,
        expected_render_job_inventory_digest,
        expected_selected_packet_inventory_digest,
        expected_final_ledger_content_digest,
        expected_packet_ids,
    )
    if any(value is not None for value in bindings):
        if any(value is None for value in bindings):
            _fail("operational metrics binding", "live binding must be complete")
        expected_run = _digest(
            expected_run_manifest_digest, "operational metrics live run manifest"
        )
        expected_jobs = _digest(
            expected_render_job_inventory_digest,
            "operational metrics live job inventory",
        )
        expected_selected = _digest(
            expected_selected_packet_inventory_digest,
            "operational metrics live selected inventory",
        )
        expected_ledger = _digest(
            expected_final_ledger_content_digest,
            "operational metrics live final ledger",
        )
        packet_ids = expected_packet_ids
        if not isinstance(packet_ids, tuple) or len(packet_ids) != len(set(packet_ids)):
            _fail("operational metrics live packets", "expected unique packet IDs")
        allowed = set(packet_ids)
        decoded = decoded_result
        if not decoded:
            _fail(
                "operational metrics binding",
                "measured invocation evidence is absent",
            )
        for metrics in decoded:
            if (
                metrics.run_manifest_digest != expected_run
                or metrics.render_job_inventory_digest != expected_jobs
                or metrics.selected_packet_inventory_digest != expected_selected
                or metrics.packet_count != len(packet_ids)
                or metrics.image_inventory_count != len(packet_ids) * 3
                or not set(metrics.rendered_packet_ids)
                .union(metrics.resume_reused_packet_ids)
                .issubset(allowed)
            ):
                _fail("operational metrics binding", "live run identity differs")
        final = decoded[-1]
        if (
            final.ledger_content_digest != expected_ledger
            or final.invocation_kind != "zero_work_resume"
            or final.resume_reused_packet_ids != packet_ids
        ):
            _fail("operational metrics binding", "final ledger identity differs")
    return result


def persist_render_operational_metrics(
    metrics: M4ARenderOperationalMetricsV1,
    render_root: Path,
) -> StrictReportV1:
    """Append one immutable invocation report or replay a final resume idempotently."""

    from latentguard.vision_data.serialization import _is_link_or_junction

    if not isinstance(metrics, M4ARenderOperationalMetricsV1):
        _fail("operational metrics", "expected M4ARenderOperationalMetricsV1")
    root = Path(render_root)
    if _is_link_or_junction(root) or not root.is_dir():
        _fail("operational metrics root", "expected a regular non-link render root")
    existing = load_render_operational_metrics(root)
    decoded_existing = tuple(_metrics_from_report(report) for report in existing)
    if any(item.invocation_kind == "zero_work_resume" for item in decoded_existing):
        final = decoded_existing[-1]
        stable_fields = (
            "invocation_kind",
            "run_manifest_digest",
            "render_job_inventory_digest",
            "selected_packet_inventory_digest",
            "ledger_content_digest",
            "environment_initialization_count",
            "packet_count",
            "image_inventory_count",
            "rendered_packet_count",
            "images_rendered_count",
            "resume_reused_packet_count",
            "rendered_packet_ids",
            "resume_reused_packet_ids",
            "packet_render_durations_ns",
        )
        if metrics.invocation_kind == "zero_work_resume" and all(
            getattr(metrics, name) == getattr(final, name) for name in stable_fields
        ):
            return existing[-1]
        _fail(
            "operational metrics",
            "final zero-work retry binding or packet evidence differs",
        )
    if metrics.invocation_kind == "render" and existing:
        _fail("operational metrics", "a new-render invocation must be first")
    if metrics.invocation_kind == "render_resume" and existing:
        _fail(
            "operational metrics",
            "a successful resume cannot follow persisted complete render evidence",
        )
    directory = root / M4A_OPERATIONAL_METRICS_DIRECTORY
    directory.mkdir(exist_ok=True)
    if _is_link_or_junction(directory) or not directory.is_dir():
        _fail("operational metrics directory", "expected a regular non-link directory")
    report = metrics.to_report()
    path = directory / f"{len(existing) + 1:06d}.json"
    save_strict_report(report, path)
    reloaded = load_render_operational_metrics(root)
    if len(reloaded) != len(existing) + 1 or (
        reloaded[-1].content_digest != report.content_digest
    ):
        _fail("operational metrics", "append or strict reload changed evidence")
    return reloaded[-1]


def build_operational_cost_summary_report(
    reports: tuple[StrictReportV1, ...],
    *,
    validated_packet_count: int,
    validated_image_count: int,
    validation_duration_ns: int,
    total_server_active_duration_ns: int | None = None,
) -> StrictReportV1:
    """Aggregate exact render invocations plus independent validation cost."""

    packet_count = _count(validated_packet_count, "operational summary packets")
    image_count = _count(validated_image_count, "operational summary images")
    validation_ns = _optional_duration(
        validation_duration_ns, "operational summary validation duration"
    )
    if validation_ns is None:
        _fail("operational summary validation duration", "must be measured")
    total_server_ns = _optional_duration(
        total_server_active_duration_ns,
        "operational summary total server-active duration",
    )
    metrics = tuple(_metrics_from_report(report) for report in reports)
    if not metrics:
        _fail("operational summary", "measured render evidence is required")
    groups: dict[str, list[M4ARenderOperationalMetricsV1]] = {}
    for item in metrics:
        groups.setdefault(item.render_job_inventory_digest, []).append(item)
    final_packet_ids: set[str] = set()
    final_metrics: list[M4ARenderOperationalMetricsV1] = []
    for group in groups.values():
        zero_work_indexes = tuple(
            index
            for index, item in enumerate(group)
            if item.invocation_kind == "zero_work_resume"
        )
        if zero_work_indexes != (len(group) - 1,):
            _fail(
                "operational summary zero-work",
                "each render root requires exactly one final zero-work invocation",
            )
        final = group[-1]
        covered = set(final.rendered_packet_ids).union(final.resume_reused_packet_ids)
        if len(covered) != final.packet_count:
            _fail("operational summary packets", "final invocation coverage differs")
        if final_packet_ids.intersection(covered):
            _fail("operational summary packets", "render roots share packet IDs")
        final_packet_ids.update(covered)
        final_metrics.append(final)
    if len(final_packet_ids) != packet_count:
        _fail(
            "operational summary packets",
            "final invocation coverage differs from validated datasets",
        )
    measured_durations: dict[str, int] = {}
    for item in metrics:
        for packet_id, duration in item.packet_render_durations_ns:
            if packet_id in measured_durations:
                _fail(
                    "operational summary timings",
                    "one packet has more than one measured render",
                )
            measured_durations[packet_id] = duration
    if not set(measured_durations).issubset(final_packet_ids):
        _fail("operational summary timings", "timed packet is absent from final data")
    unmeasured_packet_count = len(final_packet_ids.difference(measured_durations))
    durations = tuple(measured_durations.values())
    timing_complete = unmeasured_packet_count == 0
    if image_count != packet_count * 3:
        _fail("operational summary images", "validated inventory must have three views")
    known_server_ns = (
        sum(item.server_active_execution_duration_ns or 0 for item in metrics)
        + validation_ns
    )
    if total_server_ns is not None and total_server_ns < known_server_ns:
        _fail(
            "operational summary total server-active duration",
            "cannot be shorter than the measured command subtotal",
        )
    peak_values = tuple(
        item.peak_gpu_memory_bytes
        for item in metrics
        if item.peak_gpu_memory_bytes is not None
    )
    return StrictReportV1(
        report_type=M4A_OPERATIONAL_COST_SUMMARY_REPORT_TYPE,
        payload={
            "environment_initialization_count": sum(
                item.environment_initialization_count for item in metrics
            ),
            "environment_initialization_count_complete": timing_complete,
            "image_inventory_count": image_count,
            "images_rendered_count": packet_count * 3,
            "invocation_report_count": len(metrics),
            "measured_server_active_duration_ns": known_server_ns,
            "measured_rendered_packet_count": len(measured_durations),
            "measured_rendering_time_ns": sum(durations),
            "packet_count": packet_count,
            "packet_render_time": _duration_statistics(durations),
            "peak_gpu_memory_bytes": max(peak_values) if peak_values else None,
            "peak_gpu_memory_complete": len(peak_values) == len(metrics),
            "peak_gpu_memory_scope": "torch_cuda_allocator_only_v1",
            "renderer_total_peak_gpu_memory_available": False,
            "renderer_total_peak_gpu_memory_bytes": None,
            "renderer_total_peak_gpu_memory_unavailable_reason": (
                "renderer_allocations_are_outside_torch_allocator_scope"
            ),
            "rendered_packet_count": packet_count,
            "resume_reused_packet_count": sum(
                item.resume_reused_packet_count for item in final_metrics
            ),
            "schema_version": "1.0",
            "timing_complete": timing_complete,
            "timing_clock": _TIMING_CLOCK,
            "total_rendering_time_available": timing_complete,
            "total_rendering_time_ns": sum(durations) if timing_complete else None,
            "total_server_active_duration_available": total_server_ns is not None,
            "total_server_active_duration_ns": total_server_ns,
            "total_server_active_unavailable_reason": (
                None if total_server_ns is not None else "end_to_end_timer_not_provided"
            ),
            "unmeasured_preexisting_or_reused_packet_count": unmeasured_packet_count,
            "validation_time_ns": validation_ns,
            "zero_work_resume_root_count": len(final_metrics),
        },
    )


__all__ = [
    "M4AEnvironmentInitializationRecorder",
    "M4AOperationalMetricsError",
    "M4ARenderOperationalMetricsV1",
    "M4A_OPERATIONAL_COST_SUMMARY_REPORT_TYPE",
    "M4A_OPERATIONAL_METRICS_DIRECTORY",
    "M4A_RENDER_OPERATIONAL_REPORT_TYPE",
    "build_operational_cost_summary_report",
    "load_render_operational_metrics",
    "persist_render_operational_metrics",
]

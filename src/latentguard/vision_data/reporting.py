"""Compact, sanitized, self-digesting reports for M4A visual generation."""

from __future__ import annotations

import hashlib
import math
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn

from latentguard.evaluation.security import is_sanitized_operational_text
from latentguard.replay.base import ReplayValidationError
from latentguard.replay.identity import canonical_json_bytes
from latentguard.training.reporting import (
    StrictReportV1,
    load_strict_report,
    save_strict_report,
)
from latentguard.vision_data.cameras import PickCubeMultiViewRigV1
from latentguard.vision_data.domains import RenderDomainConfigurationV1
from latentguard.vision_data.leakage import CrossDatasetLeakageReportV1
from latentguard.vision_data.models import (
    VisualVerifierDevelopmentDatasetV1,
    VisualVerifierExternalDatasetV1,
)
from latentguard.vision_data.rendering import ZeroWorkRenderResumeV1


class VisualReportingError(ValueError):
    """Raised when a compact visual report would be incomplete or misleading."""


M4A_RENDER_RESUME_REPORT_FILENAME = "render-resume-report.json"


def _fail(context: str, reason: str) -> NoReturn:
    raise VisualReportingError(f"{context}: {reason}")


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(context, "expected lowercase SHA-256 digest")
    return value


def _content_digest(value: object, context: str) -> str:
    try:
        encoded = canonical_json_bytes(value, context=context)
    except ReplayValidationError as exc:
        raise VisualReportingError(str(exc)) from exc
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _finite_values(values: Sequence[float], context: str) -> tuple[float, ...]:
    materialized = tuple(values)
    if not materialized or any(
        type(item) not in {int, float}
        or isinstance(item, bool)
        or not math.isfinite(item)
        or item < 0.0
        for item in materialized
    ):
        _fail(context, "expected non-empty finite non-negative values")
    return tuple(float(item) for item in materialized)


def _error_statistics(values: tuple[float, ...]) -> Mapping[str, object]:
    """Return deterministic compact statistics under a fixed percentile semantic."""

    ordered = tuple(sorted(values))
    count = len(ordered)
    middle = count // 2
    median = (
        ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    p95 = ordered[math.ceil(0.95 * count) - 1]
    return {
        "count": count,
        "maximum": ordered[-1],
        "median": median,
        "minimum": ordered[0],
        "p95": p95,
        "percentile_semantic": "nearest_rank_v1",
        "zero_count": sum(item == 0.0 for item in ordered),
    }


def _bounded_integers(
    values: Sequence[int], context: str, *, count: int, maximum: int
) -> tuple[int, ...]:
    result = tuple(values)
    if len(result) != count or any(
        type(item) is not int or not 0 <= item <= maximum for item in result
    ):
        _fail(
            context,
            f"expected exactly {count} integers between zero and {maximum}",
        )
    return result


def _dataset_report_inventory(
    dataset: VisualVerifierDevelopmentDatasetV1 | VisualVerifierExternalDatasetV1,
) -> tuple[
    Counter[str],
    Counter[str],
    dict[str, dict[str, int]],
    Counter[str],
    str,
    int,
    int,
]:
    """Return compact counts plus a digest over the exact ordered image inventory."""

    packets = tuple(getattr(dataset, "packets", ()))
    if not packets:
        _fail("visual dataset report", "dataset is empty")
    split_counts = Counter(str(item.split.value) for item in packets)
    domain_counts = Counter(str(item.render_domain_id) for item in packets)
    split_domain_counts: dict[str, Counter[str]] = {}
    camera_counts: Counter[str] = Counter()
    image_inventory: list[dict[str, str]] = []
    npy_digests: list[str] = []
    pixel_digests: list[str] = []
    for packet in packets:
        split = str(packet.split.value)
        domain = str(packet.render_domain_id)
        split_domain_counts.setdefault(split, Counter())[domain] += 1
        for view in packet.views:
            camera_counts[view.camera_id] += 1
            npy_digests.append(view.npy_sha256)
            pixel_digests.append(view.pixel_sha256)
            image_inventory.append(
                {
                    "camera_id": view.camera_id,
                    "npy_sha256": view.npy_sha256,
                    "packet_id": packet.packet_id,
                    "pixel_sha256": view.pixel_sha256,
                }
            )
    image_inventory_digest = _content_digest(
        {
            "images": image_inventory,
            "schema_version": "m4a_image_digest_inventory_v1",
        },
        "M4AImageDigestInventoryV1",
    )
    return (
        split_counts,
        domain_counts,
        {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_domain_counts.items())
        },
        camera_counts,
        image_inventory_digest,
        len(set(npy_digests)),
        len(set(pixel_digests)),
    )


def build_camera_rig_summary_report(rig: PickCubeMultiViewRigV1) -> StrictReportV1:
    """Summarize the frozen three-view rig without runtime paths or images."""

    if not isinstance(rig, PickCubeMultiViewRigV1):
        _fail("camera rig summary", "expected PickCubeMultiViewRigV1")
    return StrictReportV1(
        report_type="m4a_camera_rig_summary_v1",
        payload={
            "camera_count": len(rig.cameras),
            "cameras": [
                {
                    "camera_configuration_digest": (camera.camera_configuration_digest),
                    "camera_id": camera.camera_id,
                    "far": camera.far,
                    "fov_y_degrees": camera.fov_y_degrees,
                    "height": camera.height,
                    "near": camera.near,
                    "width": camera.width,
                }
                for camera in rig.cameras
            ],
            "rig_digest": rig.rig_digest,
            "rig_id": rig.rig_id,
            "schema_version": "1.0",
            "semantic_version": rig.semantic_version,
        },
    )


def build_render_domain_summary_report(
    configuration: RenderDomainConfigurationV1,
) -> StrictReportV1:
    """Summarize the frozen five-domain assignment and rendering contract."""

    if not isinstance(configuration, RenderDomainConfigurationV1):
        _fail("render domain summary", "expected RenderDomainConfigurationV1")
    return StrictReportV1(
        report_type="m4a_render_domain_summary_v1",
        payload={
            "color_format": configuration.color_format,
            "configuration_digest": configuration.content_digest,
            "configuration_id": configuration.configuration_id,
            "domain_count": len(configuration.domains),
            "domains": [
                {
                    "allowed_collections": [
                        item.value for item in domain.allowed_collections
                    ],
                    "allowed_splits": [item.value for item in domain.allowed_splits],
                    "domain_digest": domain.domain_digest,
                    "domain_id": domain.domain_id,
                    "semantic_version": domain.semantic_version,
                }
                for domain in configuration.domains
            ],
            "image_resolution": list(configuration.image_resolution),
            "schema_version": "1.0",
            "seed_derivation": configuration.seed_derivation,
            "shader_configuration": configuration.shader_configuration,
        },
    )


def build_image_digest_summary_report(
    dataset: VisualVerifierDevelopmentDatasetV1 | VisualVerifierExternalDatasetV1,
) -> StrictReportV1:
    """Bind the exact image inventory while retrieving only compact counts."""

    (
        _split_counts,
        _domain_counts,
        _split_domain_counts,
        camera_counts,
        inventory_digest,
        unique_npy_count,
        unique_pixel_count,
    ) = _dataset_report_inventory(dataset)
    image_count = sum(camera_counts.values())
    external = isinstance(dataset, VisualVerifierExternalDatasetV1)
    return StrictReportV1(
        report_type=(
            "m4a_external_image_digest_summary_v1"
            if external
            else "m4a_development_image_digest_summary_v1"
        ),
        payload={
            "camera_image_counts": dict(sorted(camera_counts.items())),
            "dataset_content_digest": dataset.content_digest,
            "image_count": image_count,
            "image_inventory_digest": inventory_digest,
            "raw_image_bytes_included": False,
            "schema_version": "1.0",
            "unique_npy_digest_count": unique_npy_count,
            "unique_pixel_digest_count": unique_pixel_count,
        },
    )


def build_domain_assignment_report(
    dataset: VisualVerifierDevelopmentDatasetV1 | VisualVerifierExternalDatasetV1,
) -> StrictReportV1:
    """Record the exact split-by-domain cross-tab after strict validation."""

    from latentguard.vision_data.dataset import validate_visual_verifier_dataset

    validate_visual_verifier_dataset(dataset, full_target=False)
    (
        split_counts,
        domain_counts,
        split_domain_counts,
        _camera_counts,
        _inventory_digest,
        _unique_npy_count,
        _unique_pixel_count,
    ) = _dataset_report_inventory(dataset)
    external = isinstance(dataset, VisualVerifierExternalDatasetV1)
    return StrictReportV1(
        report_type=(
            "m4a_external_domain_assignment_v1"
            if external
            else "m4a_development_domain_assignment_v1"
        ),
        payload={
            "assignment_validation_passed": True,
            "dataset_content_digest": dataset.content_digest,
            "domain_packet_counts": dict(sorted(domain_counts.items())),
            "schema_version": "1.0",
            "split_domain_packet_counts": split_domain_counts,
            "split_packet_counts": dict(sorted(split_counts.items())),
        },
    )


def build_visual_dataset_summary_report(
    dataset: VisualVerifierDevelopmentDatasetV1 | VisualVerifierExternalDatasetV1,
) -> StrictReportV1:
    """Summarize identities, counts, splits, domains, and image inventory."""

    packets = tuple(getattr(dataset, "packets", ()))
    bindings = tuple(getattr(dataset, "candidate_bindings", ()))
    samples = tuple(getattr(dataset, "samples", ()))
    if not bindings:
        _fail("visual dataset report", "dataset is empty")
    (
        split_counts,
        domain_counts,
        split_domain_counts,
        camera_counts,
        image_inventory_digest,
        unique_npy_count,
        unique_pixel_count,
    ) = _dataset_report_inventory(dataset)
    image_count = sum(camera_counts.values())
    dataset_type = type(dataset).__name__
    external = dataset_type == "VisualVerifierExternalDatasetV1"
    report_type = (
        "m4a_external_visual_dataset_summary_v1"
        if external
        else "m4a_development_visual_dataset_summary_v1"
    )
    payload: dict[str, object] = {
        "camera_rig_digest": dataset.camera_rig_digest,
        "candidate_binding_count": len(bindings),
        "dataset_content_digest": dataset.content_digest,
        "domain_packet_counts": dict(sorted(domain_counts.items())),
        "expanded_visual_sample_count": len(samples),
        "image_count": image_count,
        "image_inventory_digest": image_inventory_digest,
        "images_stored_once_and_shared_by_candidates": True,
        "packet_count": len(packets),
        "packet_manifest_digest": dataset.packet_manifest_digest,
        "render_domain_configuration_digest": (
            dataset.render_domain_configuration_digest
        ),
        "samples_are_derived": True,
        "schema_version": "1.0",
        "source_collection": str(dataset.source_collection.value),
        "split_packet_counts": dict(sorted(split_counts.items())),
        "split_domain_packet_counts": split_domain_counts,
        "training_allowed": dataset.training_allowed,
        "unique_npy_digest_count": unique_npy_count,
        "unique_pixel_digest_count": unique_pixel_count,
        "visual_compatibility_identity": dataset.visual_compatibility_identity,
    }
    for name in (
        "source_dataset_digest",
        "split_digest",
        "evidence_digest",
        "source_set_identity",
        "candidate_pool_identity",
        "blind_manifest_digest",
        "full_outcome_digest",
    ):
        value = getattr(dataset, name, None)
        if value is not None:
            payload[name] = value
    return StrictReportV1(report_type=report_type, payload=payload)


def build_render_determinism_report(
    *,
    visual_compatibility_identity: str,
    repeated_changed_pixel_counts: Sequence[int],
    repeated_maximum_channel_differences: Sequence[int],
    repeated_mean_absolute_differences: Sequence[float],
    fresh_environment_changed_pixel_counts: Sequence[int],
    fresh_environment_maximum_channel_differences: Sequence[int],
    fresh_environment_mean_absolute_differences: Sequence[float],
    spatially_stable: bool,
) -> StrictReportV1:
    """Record exact repeated/fresh pixel checks without inventing tolerance."""

    _digest(visual_compatibility_identity, "render determinism visual identity")
    if type(spatially_stable) is not bool:
        _fail("render determinism", "spatially_stable must be boolean")
    repeated_mean = _finite_values(
        repeated_mean_absolute_differences, "repeated mean absolute differences"
    )
    fresh_mean = _finite_values(
        fresh_environment_mean_absolute_differences,
        "fresh-environment mean absolute differences",
    )
    repeated_changed = _bounded_integers(
        repeated_changed_pixel_counts,
        "repeated changed pixel counts",
        count=3,
        maximum=224 * 224,
    )
    fresh_changed = _bounded_integers(
        fresh_environment_changed_pixel_counts,
        "fresh-environment changed pixel counts",
        count=3,
        maximum=224 * 224,
    )
    repeated_max = _bounded_integers(
        repeated_maximum_channel_differences,
        "repeated maximum channel differences",
        count=3,
        maximum=255,
    )
    fresh_max = _bounded_integers(
        fresh_environment_maximum_channel_differences,
        "fresh-environment maximum channel differences",
        count=3,
        maximum=255,
    )
    if len(repeated_mean) != 3 or len(fresh_mean) != 3:
        _fail(
            "render determinism",
            "repeated and fresh mean-difference inventories require three views",
        )
    if any(item > 255.0 for item in (*repeated_mean, *fresh_mean)):
        _fail("render determinism", "mean pixel differences cannot exceed 255")
    exact = not any(
        (*repeated_changed, *fresh_changed, *repeated_max, *fresh_max)
    ) and not any((*repeated_mean, *fresh_mean))
    return StrictReportV1(
        report_type="m4a_render_determinism_v1",
        payload={
            "fresh_environment": {
                "changed_pixel_counts": list(fresh_changed),
                "maximum_per_channel_absolute_differences": list(fresh_max),
                "mean_absolute_pixel_differences": list(fresh_mean),
            },
            "pixel_tolerance_authorized": False,
            "repeated_render": {
                "changed_pixel_counts": list(repeated_changed),
                "maximum_per_channel_absolute_differences": list(repeated_max),
                "mean_absolute_pixel_differences": list(repeated_mean),
            },
            "rendering_reproducible": exact,
            "schema_version": "1.0",
            "spatially_stable": spatially_stable,
            "trusted_generation_permitted": exact,
            "visual_compatibility_identity": visual_compatibility_identity,
        },
    )


def build_state_integrity_report(
    *,
    visual_dataset_digest: str,
    packet_manifest_digest: str,
    packet_count: int,
    compared_component_counts: Sequence[int],
    maximum_absolute_errors: Sequence[float],
    verifier_component_counts: Sequence[int],
    verifier_maximum_absolute_errors: Sequence[float],
    task_projection_changes: int,
    physics_step_count: int,
    environment_close_failure_count: int,
) -> StrictReportV1:
    """Summarize complete-state and verifier-state post-render integrity."""

    _digest(visual_dataset_digest, "state integrity visual dataset")
    _digest(packet_manifest_digest, "state integrity packet manifest")
    if type(packet_count) is not int or packet_count <= 0:
        _fail("state integrity", "packet_count must be a positive integer")
    if (
        type(task_projection_changes) is not int
        or task_projection_changes < 0
        or type(physics_step_count) is not int
        or physics_step_count < 0
        or type(environment_close_failure_count) is not int
        or environment_close_failure_count < 0
    ):
        _fail(
            "state integrity",
            "change, physics-step, and close-failure counts must be non-negative",
        )
    state_errors = _finite_values(maximum_absolute_errors, "state errors")
    verifier_errors = _finite_values(
        verifier_maximum_absolute_errors, "verifier errors"
    )
    state_counts = tuple(compared_component_counts)
    verifier_counts = tuple(verifier_component_counts)
    all_component_counts = (*state_counts, *verifier_counts)
    if any(type(item) is not int or item < 0 for item in all_component_counts):
        _fail("state integrity", "component counts must be non-negative integers")
    if not all(
        len(items) == packet_count
        for items in (state_counts, state_errors, verifier_counts, verifier_errors)
    ):
        _fail(
            "state integrity",
            "all per-packet inventories must exactly cover packet_count",
        )
    passed = (
        set(state_counts) == {70}
        and set(verifier_counts) == {38}
        and max(state_errors) <= 1e-6
        and max(verifier_errors) == 0.0
        and task_projection_changes == 0
        and physics_step_count == 0
        and environment_close_failure_count == 0
    )
    return StrictReportV1(
        report_type="m4a_render_state_integrity_v1",
        payload={
            "complete_state_error_statistics": _error_statistics(state_errors),
            "complete_state_component_count_values": sorted(set(state_counts)),
            "environment_close_failure_count": environment_close_failure_count,
            "maximum_complete_state_error": max(state_errors),
            "maximum_verifier_state_error": max(verifier_errors),
            "packet_count": packet_count,
            "passed": passed,
            "physics_step_count": physics_step_count,
            "schema_version": "1.0",
            "state_tolerance": 1e-6,
            "task_projection_change_count": task_projection_changes,
            "verifier_state_error_statistics": _error_statistics(verifier_errors),
            "verifier_state_component_count_values": sorted(set(verifier_counts)),
            "visual_dataset_digest": visual_dataset_digest,
            "packet_manifest_digest": packet_manifest_digest,
        },
    )


def build_state_integrity_report_from_dataset(
    dataset: VisualVerifierDevelopmentDatasetV1 | VisualVerifierExternalDatasetV1,
) -> StrictReportV1:
    """Project complete packet-bound integrity evidence into one compact report."""

    if not isinstance(
        dataset,
        (VisualVerifierDevelopmentDatasetV1, VisualVerifierExternalDatasetV1),
    ):
        _fail("state integrity", "expected a visual verifier dataset")
    from latentguard.vision_data.dataset import validate_visual_verifier_dataset

    validate_visual_verifier_dataset(dataset, full_target=False)
    packets = tuple(dataset.packets)
    if not packets:
        _fail("state integrity", "visual dataset is empty")
    state_counts: list[int] = []
    state_errors: list[float] = []
    verifier_counts: list[int] = []
    verifier_errors: list[float] = []
    task_changes = 0
    physics_steps = 0
    close_failures = 0
    for packet in packets:
        component_counts = {
            view.compared_state_component_count for view in packet.views
        }
        if len(component_counts) != 1:
            _fail(packet.packet_id, "view state-component counts differ")
        state_counts.append(next(iter(component_counts)))
        state_errors.append(max(view.maximum_state_error for view in packet.views))
        verifier_counts.append(packet.verifier_state_component_count)
        verifier_errors.append(packet.verifier_state_maximum_absolute_error)
        task_changes += int(
            any(
                view.task_projection_before != view.task_projection_after
                for view in packet.views
            )
        )
        elapsed_delta = (
            packet.elapsed_simulation_steps_after
            - packet.elapsed_simulation_steps_before
        )
        if elapsed_delta < 0:
            _fail(packet.packet_id, "elapsed simulation step counter regressed")
        physics_steps += elapsed_delta
        close_failures += int(not packet.environment_close_passed)
    return build_state_integrity_report(
        visual_dataset_digest=dataset.content_digest,
        packet_manifest_digest=dataset.packet_manifest_digest,
        packet_count=len(packets),
        compared_component_counts=state_counts,
        maximum_absolute_errors=state_errors,
        verifier_component_counts=verifier_counts,
        verifier_maximum_absolute_errors=verifier_errors,
        task_projection_changes=task_changes,
        physics_step_count=physics_steps,
        environment_close_failure_count=close_failures,
    )


def build_leakage_report(report: CrossDatasetLeakageReportV1) -> StrictReportV1:
    """Convert an exact leakage inventory into a compact strict report."""

    fields: dict[str, object] = {
        name: list(getattr(report, name))
        for name in getattr(report, "__dataclass_fields__", {})
    }
    fields.update(
        {
            "cross_dataset_leakage_absent": report.cross_dataset_leakage_absent,
            "perceptual_similarity_used": False,
            "schema_version": "1.0",
        }
    )
    return StrictReportV1(report_type="m4a_cross_dataset_leakage_v1", payload=fields)


def build_external_training_prohibition_report(
    dataset: VisualVerifierExternalDatasetV1,
) -> StrictReportV1:
    """Record that the M3C visual dataset is rejected by training loaders."""

    from latentguard.vision_data.dataset import (
        VisualDatasetError,
        require_training_allowed,
    )

    try:
        require_training_allowed(dataset)
    except VisualDatasetError:
        rejection_verified = True
    else:
        _fail("external training prohibition", "training gate accepted external data")
    return StrictReportV1(
        report_type="m4a_external_training_prohibition_v1",
        payload={
            "candidate_pool_identity": dataset.candidate_pool_identity,
            "dataset_content_digest": dataset.content_digest,
            "evaluation_only": True,
            "schema_version": "1.0",
            "training_allowed": False,
            "training_loader_rejection_verified": rejection_verified,
        },
    )


def build_render_resume_report(
    proof: ZeroWorkRenderResumeV1,
    *,
    source_identity_digest: str,
    camera_rig_digest: str,
    render_domain_configuration_digest: str,
) -> StrictReportV1:
    """Bind a zero-work resume proof to the frozen visual configuration."""

    if not isinstance(proof, ZeroWorkRenderResumeV1):
        _fail("render resume", "expected ZeroWorkRenderResumeV1")
    if proof.zero_duplicate_work is not True:
        _fail("render resume", "zero duplicate work was not verified")
    if (
        not is_sanitized_operational_text(proof.run_id)
        or type(proof.packet_count) is not int
        or proof.packet_count <= 0
        or type(proof.ledger_entry_count) is not int
        or proof.ledger_entry_count < proof.packet_count
        or any(
            type(value) is not int or value != 0
            for value in (
                proof.rendered_packets,
                proof.recovered_packets,
                proof.retried_packets,
            )
        )
    ):
        _fail("render resume", "proof counts or run identity are invalid")
    _digest(proof.ledger_content_digest, "render resume ledger")
    _digest(source_identity_digest, "render resume source identity")
    _digest(camera_rig_digest, "render resume camera rig")
    _digest(
        render_domain_configuration_digest,
        "render resume domain configuration",
    )
    return StrictReportV1(
        report_type="m4a_render_resume_v1",
        payload={
            "camera_rig_digest": camera_rig_digest,
            "ledger_content_digest": proof.ledger_content_digest,
            "ledger_entry_count": proof.ledger_entry_count,
            "packet_count": proof.packet_count,
            "recovered_packets": proof.recovered_packets,
            "render_domain_configuration_digest": (render_domain_configuration_digest),
            "rendered_packets": proof.rendered_packets,
            "retried_packets": proof.retried_packets,
            "run_id": proof.run_id,
            "schema_version": "1.0",
            "source_identity_digest": source_identity_digest,
            "zero_duplicate_work": True,
        },
    )


def persist_render_resume_report(
    proof: ZeroWorkRenderResumeV1,
    render_root: Path,
    *,
    source_identity_digest: str,
    camera_rig_digest: str,
    render_domain_configuration_digest: str,
) -> StrictReportV1:
    """Immutably publish and strictly reload a zero-work report beside a run."""

    from latentguard.vision_data.serialization import _is_link_or_junction

    root = Path(render_root)
    if _is_link_or_junction(root) or not root.is_dir():
        _fail("render resume root", "expected regular non-link directory")
    report = build_render_resume_report(
        proof,
        source_identity_digest=source_identity_digest,
        camera_rig_digest=camera_rig_digest,
        render_domain_configuration_digest=render_domain_configuration_digest,
    )
    path = save_strict_report(report, root / M4A_RENDER_RESUME_REPORT_FILENAME)
    return load_strict_report(
        path,
        expected_report_type=report.report_type,
        expected_content_digest=report.content_digest,
    )


def load_bound_render_resume_report(
    render_root: Path,
    *,
    run_id: str,
    source_identity_digest: str,
    camera_rig_digest: str,
    render_domain_configuration_digest: str,
    ledger_content_digest: str,
    packet_count: int,
    ledger_entry_count: int,
) -> StrictReportV1:
    """Load a resume report and require exact agreement with the live run root."""

    from latentguard.vision_data.serialization import _is_link_or_junction

    root = Path(render_root)
    path = root / M4A_RENDER_RESUME_REPORT_FILENAME
    if (
        _is_link_or_junction(root)
        or not root.is_dir()
        or _is_link_or_junction(path)
        or not path.is_file()
        or path.stat().st_nlink != 1
    ):
        _fail("render resume report", "expected regular non-link run evidence")
    report = load_strict_report(
        path,
        expected_report_type="m4a_render_resume_v1",
    )
    expected: dict[str, object] = {
        "camera_rig_digest": camera_rig_digest,
        "ledger_content_digest": ledger_content_digest,
        "ledger_entry_count": ledger_entry_count,
        "packet_count": packet_count,
        "recovered_packets": 0,
        "render_domain_configuration_digest": render_domain_configuration_digest,
        "rendered_packets": 0,
        "retried_packets": 0,
        "run_id": run_id,
        "schema_version": "1.0",
        "source_identity_digest": source_identity_digest,
        "zero_duplicate_work": True,
    }
    if dict(report.payload) != expected:
        _fail("render resume report", "does not match the current immutable run")
    return report


def build_compact_retrieval_report(
    reports: Mapping[str, StrictReportV1],
) -> StrictReportV1:
    """Bind the exact sanitized compact-report inventory selected for retrieval."""

    if not reports or any(
        not isinstance(item, StrictReportV1) for item in reports.values()
    ):
        _fail("compact retrieval", "expected non-empty StrictReportV1 inventory")
    return StrictReportV1(
        report_type="m4a_compact_retrieval_manifest_v1",
        payload={
            "raw_actions_included": False,
            "raw_images_included": False,
            "raw_state_archives_included": False,
            "report_file_count": len(reports),
            "report_content_digests": {
                name: report.content_digest for name, report in sorted(reports.items())
            },
            "sanitized_json_only": True,
            "schema_version": "1.0",
        },
    )


_VALIDATION_REPORT_TYPES_BY_NAME = {
    "camera-rig-summary.json": "m4a_camera_rig_summary_v1",
    "cross-dataset-leakage.json": "m4a_cross_dataset_leakage_v1",
    "development-domain-assignment.json": "m4a_development_domain_assignment_v1",
    "development-dataset-summary.json": ("m4a_development_visual_dataset_summary_v1"),
    "development-image-digest-summary.json": (
        "m4a_development_image_digest_summary_v1"
    ),
    "development-resume.json": "m4a_render_resume_v1",
    "development-state-integrity.json": "m4a_render_state_integrity_v1",
    "external-domain-assignment.json": "m4a_external_domain_assignment_v1",
    "external-dataset-summary.json": "m4a_external_visual_dataset_summary_v1",
    "external-image-digest-summary.json": "m4a_external_image_digest_summary_v1",
    "external-resume.json": "m4a_render_resume_v1",
    "external-state-integrity.json": "m4a_render_state_integrity_v1",
    "external-training-prohibition.json": ("m4a_external_training_prohibition_v1"),
    "operational-cost-summary.json": "m4a_operational_cost_summary_v1",
    "render-determinism.json": "m4a_render_determinism_v1",
    "render-domain-summary.json": "m4a_render_domain_summary_v1",
}
_COMPACT_RETRIEVAL_MANIFEST_NAME = "compact-retrieval-manifest.json"


def _remove_staging_report_directory(path: Path) -> None:
    """Remove only regular files created inside a private staging directory."""

    if not path.exists() or path.is_symlink() or not path.is_dir():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            return
        child.unlink(missing_ok=True)
    path.rmdir()


def publish_visual_validation_reports(
    reports: Mapping[str, StrictReportV1],
    report_dir: Path,
) -> Mapping[str, StrictReportV1]:
    """Atomically publish and strictly reload an immutable sanitized report set."""

    materialized = dict(reports)
    if not materialized:
        _fail("visual validation reports", "expected at least one applicable report")
    unexpected = tuple(
        sorted(set(materialized).difference(_VALIDATION_REPORT_TYPES_BY_NAME))
    )
    if unexpected:
        _fail(
            "visual validation reports",
            "unsupported report file names: " + ", ".join(unexpected),
        )
    for name, report in materialized.items():
        if not isinstance(report, StrictReportV1):
            _fail(name, "expected StrictReportV1")
        if report.report_type != _VALIDATION_REPORT_TYPES_BY_NAME[name]:
            _fail(name, "report type differs from the fixed compact inventory")

    from latentguard.vision_data.serialization import _is_link_or_junction

    destination = Path(report_dir).absolute()
    if destination.exists() or _is_link_or_junction(destination):
        _fail("visual validation report directory", "must be absent")
    parent = destination.parent
    current = parent
    while True:
        if _is_link_or_junction(current):
            _fail(
                "visual validation report parent",
                "cannot traverse a link or junction",
            )
        if current.exists() and not current.is_dir():
            _fail("visual validation report parent", "must be a regular directory")
        if current == current.parent:
            break
        current = current.parent
    parent.mkdir(parents=True, exist_ok=True)
    current = parent
    while True:
        if _is_link_or_junction(current):
            _fail(
                "visual validation report parent",
                "cannot traverse a link or junction",
            )
        if not current.is_dir():
            _fail("visual validation report parent", "must be a regular directory")
        if current == current.parent:
            break
        current = current.parent
    if _is_link_or_junction(parent) or not parent.is_dir():
        _fail("visual validation report parent", "must be a regular directory")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=parent,
        )
    )
    expected: dict[str, StrictReportV1] = {}
    try:
        for name, report in sorted(materialized.items()):
            save_strict_report(report, staging / name)
            expected[name] = load_strict_report(
                staging / name,
                expected_report_type=report.report_type,
                expected_content_digest=report.content_digest,
            )
        retrieval = build_compact_retrieval_report(expected)
        save_strict_report(retrieval, staging / _COMPACT_RETRIEVAL_MANIFEST_NAME)
        expected[_COMPACT_RETRIEVAL_MANIFEST_NAME] = load_strict_report(
            staging / _COMPACT_RETRIEVAL_MANIFEST_NAME,
            expected_report_type=retrieval.report_type,
            expected_content_digest=retrieval.content_digest,
        )
        entries = tuple(staging.iterdir())
        if (
            {entry.name for entry in entries} != set(expected)
            or any(
                _is_link_or_junction(entry) or not entry.is_file() for entry in entries
            )
            or any(entry.suffix != ".json" for entry in entries)
        ):
            _fail(
                "visual validation reports",
                "staging inventory contains non-report content",
            )
        if destination.exists() or _is_link_or_junction(destination):
            _fail("visual validation report directory", "publication target appeared")
        staging.rename(destination)
    except Exception:
        _remove_staging_report_directory(staging)
        raise

    return {
        name: load_strict_report(
            destination / name,
            expected_report_type=report.report_type,
            expected_content_digest=report.content_digest,
        )
        for name, report in sorted(expected.items())
    }


__all__ = [
    "M4A_RENDER_RESUME_REPORT_FILENAME",
    "VisualReportingError",
    "build_camera_rig_summary_report",
    "build_compact_retrieval_report",
    "build_domain_assignment_report",
    "build_external_training_prohibition_report",
    "build_image_digest_summary_report",
    "build_leakage_report",
    "build_render_determinism_report",
    "build_render_resume_report",
    "build_render_domain_summary_report",
    "build_state_integrity_report",
    "build_state_integrity_report_from_dataset",
    "build_visual_dataset_summary_report",
    "load_strict_report",
    "load_bound_render_resume_report",
    "persist_render_resume_report",
    "publish_visual_validation_reports",
    "save_strict_report",
]

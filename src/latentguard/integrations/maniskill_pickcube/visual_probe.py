"""Sanitized compatibility probing for exact-state PickCube RGB rendering."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NoReturn, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.identity import canonical_json_bytes
from latentguard.vision_data.cameras import PickCubeMultiViewRigV1
from latentguard.vision_data.domains import (
    CANONICAL_DOMAIN_ID,
    RenderDomainConfigurationV1,
)

from .compatibility import CompatibilityBinding
from .configuration import (
    PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE,
    PICKCUBE_STATE_VERIFICATION_SEMANTIC,
)
from .state_indexed_archive import (
    PickCubeIndexedStateV1,
    PickCubeStateIndexedArchiveV1,
    PickCubeStateIndexedEpisodeV1,
)
from .visual_rendering import (
    VISUAL_RENDERER_SEMANTIC_VERSION,
    PickCubeVisualRenderPlan,
    RenderedVisualView,
    VisualRendererApiObservation,
    build_pickcube_visual_render_plan,
)
from .visual_session import PickCubeVisualSession, PickCubeVisualSessionResult

VISUAL_COMPATIBILITY_REPORT_SCHEMA_VERSION = "1.0"
VISUAL_PROBE_SOURCE_SCHEMA_VERSION = "1.0"
VISUAL_RENDER_OUTPUT_SHAPE = (224, 224, 3)


class ManiSkillVisualProbeError(RuntimeError):
    """Raised when the visual runtime cannot authorize trusted generation."""


@dataclass(frozen=True, slots=True)
class VisualProbeSourceEvidenceV1:
    """Content-bound archived state selected for one compatibility probe."""

    source_archive_digest: str
    source_episode_id: str
    source_episode_content_digest: str
    source_trajectory_id: str
    source_reset_seed: int
    state_index: int
    state_content_digest: str
    expected_state_digest: str
    verifier_state_digest: str
    schema_version: str = VISUAL_PROBE_SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject incomplete, unsafe, or unbound source evidence."""
        for name in (
            "source_archive_digest",
            "source_episode_content_digest",
            "state_content_digest",
            "expected_state_digest",
            "verifier_state_digest",
        ):
            _require_digest(getattr(self, name), context=f"probe source {name}")
        for name in ("source_episode_id", "source_trajectory_id"):
            _load_text(getattr(self, name), f"probe source {name}")
        if (
            type(self.source_reset_seed) is not int
            or not 0 <= self.source_reset_seed < 2**32
        ):
            raise ManiSkillVisualProbeError("probe source reset seed must be uint32")
        if type(self.state_index) is not int or self.state_index < 0:
            raise ManiSkillVisualProbeError(
                "probe source state index must be non-negative"
            )
        if self.schema_version != VISUAL_PROBE_SOURCE_SCHEMA_VERSION:
            raise ManiSkillVisualProbeError(
                "probe source schema version is unsupported"
            )

    def as_mapping(self) -> Mapping[str, object]:
        """Return the exact path-independent archived-state evidence."""
        return {
            "expected_state_digest": self.expected_state_digest,
            "schema_version": self.schema_version,
            "source_archive_digest": self.source_archive_digest,
            "source_episode_content_digest": self.source_episode_content_digest,
            "source_episode_id": self.source_episode_id,
            "source_reset_seed": self.source_reset_seed,
            "source_trajectory_id": self.source_trajectory_id,
            "state_content_digest": self.state_content_digest,
            "state_index": self.state_index,
            "verifier_state_digest": self.verifier_state_digest,
        }


@dataclass(frozen=True, slots=True)
class CameraPixelDifference:
    """Exact pixel differences for one ordered camera view."""

    camera_id: str
    changed_pixel_count: int
    maximum_per_channel_absolute_difference: int
    mean_absolute_pixel_difference: float
    exact_match: bool

    def __post_init__(self) -> None:
        """Reject inconsistent or non-finite pixel evidence."""
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ManiSkillVisualProbeError("pixel comparison camera ID is invalid")
        if (
            type(self.changed_pixel_count) is not int
            or not 0 <= self.changed_pixel_count <= 224 * 224
        ):
            raise ManiSkillVisualProbeError("changed pixel count is invalid")
        if (
            type(self.maximum_per_channel_absolute_difference) is not int
            or not 0 <= self.maximum_per_channel_absolute_difference <= 255
        ):
            raise ManiSkillVisualProbeError("maximum pixel difference is invalid")
        if (
            type(self.mean_absolute_pixel_difference) is not float
            or not math.isfinite(self.mean_absolute_pixel_difference)
            or self.mean_absolute_pixel_difference < 0.0
        ):
            raise ManiSkillVisualProbeError("mean pixel difference is invalid")
        exact = (
            self.changed_pixel_count == 0
            and self.maximum_per_channel_absolute_difference == 0
            and self.mean_absolute_pixel_difference == 0.0
        )
        if type(self.exact_match) is not bool or self.exact_match != exact:
            raise ManiSkillVisualProbeError("pixel exact-match flag is inconsistent")

    def to_dict(self) -> dict[str, object]:
        """Return canonical JSON-native evidence."""
        return {
            "camera_id": self.camera_id,
            "changed_pixel_count": self.changed_pixel_count,
            "exact_match": self.exact_match,
            "maximum_per_channel_absolute_difference": (
                self.maximum_per_channel_absolute_difference
            ),
            "mean_absolute_pixel_difference": self.mean_absolute_pixel_difference,
        }


@dataclass(frozen=True, slots=True)
class PixelComparisonReport:
    """All-view exact comparison for repeated or fresh-session rendering."""

    comparison: str
    views: tuple[CameraPixelDifference, ...]
    changed_pixel_count: int
    maximum_per_channel_absolute_difference: int
    mean_absolute_pixel_difference: float
    exact_match: bool
    spatially_stable: bool

    def __post_init__(self) -> None:
        """Require a complete ordered three-view aggregate."""
        expected_ids = ("front_oblique", "overhead", "side_oblique")
        views = tuple(self.views)
        if tuple(view.camera_id for view in views) != expected_ids:
            raise ManiSkillVisualProbeError(
                "pixel report must contain the ordered three-view rig"
            )
        expected_changed = sum(view.changed_pixel_count for view in views)
        expected_maximum = max(
            view.maximum_per_channel_absolute_difference for view in views
        )
        expected_mean = sum(
            view.mean_absolute_pixel_difference for view in views
        ) / len(views)
        expected_exact = all(view.exact_match for view in views)
        if (
            self.changed_pixel_count != expected_changed
            or self.maximum_per_channel_absolute_difference != expected_maximum
            or not math.isclose(
                self.mean_absolute_pixel_difference,
                expected_mean,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or self.exact_match != expected_exact
            or type(self.spatially_stable) is not bool
        ):
            raise ManiSkillVisualProbeError("pixel aggregate is inconsistent")
        object.__setattr__(self, "views", views)

    def to_dict(self) -> dict[str, object]:
        """Return canonical JSON-native evidence."""
        return {
            "changed_pixel_count": self.changed_pixel_count,
            "comparison": self.comparison,
            "exact_match": self.exact_match,
            "maximum_per_channel_absolute_difference": (
                self.maximum_per_channel_absolute_difference
            ),
            "mean_absolute_pixel_difference": self.mean_absolute_pixel_difference,
            "spatially_stable": self.spatially_stable,
            "views": [view.to_dict() for view in self.views],
        }


@dataclass(frozen=True, slots=True)
class VisualCompatibilityReport:
    """Visual-only compatibility evidence bound to an accepted PickCube runtime."""

    pickcube_compatibility_identity: str
    camera_rig_digest: str
    render_domain_configuration_digest: str
    probe_source: VisualProbeSourceEvidenceV1
    mani_skill_version: str
    sapien_version: str
    torch_version: str
    cuda_runtime_version: str
    gpu_model: str
    gpu_capability: str
    renderer_api: VisualRendererApiObservation
    render_output_shape: tuple[int, int, int]
    state_comparison_semantic: str
    state_comparison_tolerance: float
    compared_state_component_count: int
    verifier_component_count: int
    state_changed_by_rendering: bool
    task_projection_changed_by_rendering: bool
    elapsed_simulation_step_changed: bool
    repeated_render: PixelComparisonReport
    fresh_environment_render: PixelComparisonReport
    camera_calibration_stable: bool
    environment_close_passed: bool
    renderer_semantic_version: str = VISUAL_RENDERER_SEMANTIC_VERSION
    schema_version: str = VISUAL_COMPATIBILITY_REPORT_SCHEMA_VERSION
    visual_compatibility_identity: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate the report and compute an independent visual identity."""
        _require_digest(
            self.pickcube_compatibility_identity,
            context="PickCube compatibility identity",
        )
        _require_digest(self.camera_rig_digest, context="camera rig digest")
        _require_digest(
            self.render_domain_configuration_digest,
            context="render-domain configuration digest",
        )
        if not isinstance(self.probe_source, VisualProbeSourceEvidenceV1):
            raise ManiSkillVisualProbeError(
                "visual probe requires content-bound archived source evidence"
            )
        for field_name in (
            "mani_skill_version",
            "sapien_version",
            "torch_version",
            "cuda_runtime_version",
            "gpu_model",
            "gpu_capability",
        ):
            _load_text(getattr(self, field_name), f"visual probe {field_name}")
        if self.render_output_shape != VISUAL_RENDER_OUTPUT_SHAPE:
            raise ManiSkillVisualProbeError(
                "visual probe output must be uint8 [224,224,3]"
            )
        if self.mani_skill_version != "3.0.1":
            raise ManiSkillVisualProbeError(
                "visual probe supports exactly ManiSkill 3.0.1"
            )
        if self.renderer_semantic_version != VISUAL_RENDERER_SEMANTIC_VERSION:
            raise ManiSkillVisualProbeError(
                "visual probe renderer semantic version is unsupported"
            )
        if self.state_comparison_semantic != PICKCUBE_STATE_VERIFICATION_SEMANTIC:
            raise ManiSkillVisualProbeError(
                "visual probe state comparison semantic is unsupported"
            )
        if self.compared_state_component_count != 70:
            raise ManiSkillVisualProbeError(
                "visual probe must compare exactly 70 state components"
            )
        if self.verifier_component_count != 38:
            raise ManiSkillVisualProbeError(
                "visual probe must reproduce exactly 38 verifier components"
            )
        if (
            type(self.state_comparison_tolerance) is not float
            or self.state_comparison_tolerance
            != PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE
        ):
            raise ManiSkillVisualProbeError(
                "visual probe must retain the fixed 1e-6 state tolerance"
            )
        for field_name in (
            "state_changed_by_rendering",
            "task_projection_changed_by_rendering",
            "elapsed_simulation_step_changed",
            "camera_calibration_stable",
            "environment_close_passed",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ManiSkillVisualProbeError(
                    f"visual probe {field_name} must be boolean"
                )
        if self.schema_version != VISUAL_COMPATIBILITY_REPORT_SCHEMA_VERSION:
            raise ManiSkillVisualProbeError(
                "visual compatibility report schema version is unsupported"
            )
        identity = _digest_payload(
            self._semantic_payload(), context="PickCubeVisualCompatibility"
        )
        object.__setattr__(self, "visual_compatibility_identity", identity)

    @property
    def trusted_visual_generation_ready(self) -> bool:
        """Return whether every strict state, task, pixel, and close gate passed."""
        return (
            not self.state_changed_by_rendering
            and not self.task_projection_changed_by_rendering
            and not self.elapsed_simulation_step_changed
            and self.repeated_render.exact_match
            and self.fresh_environment_render.exact_match
            and self.camera_calibration_stable
            and self.renderer_api.camera_intrinsics_available
            and self.renderer_api.camera_extrinsics_available
            and self.renderer_api.raw_color_texture_dtype != "unobserved"
            and self.renderer_api.runtime_intrinsics_dtype != "unobserved"
            and self.renderer_api.runtime_extrinsics_dtype != "unobserved"
            and self.renderer_api.raw_extrinsic_matrix_shape != "unobserved"
            and self.environment_close_passed
        )

    def require_trusted_visual_generation_ready(self) -> None:
        """Fail closed without inventing a pixel tolerance."""
        if self.trusted_visual_generation_ready:
            return
        raise ManiSkillVisualProbeError(
            "visual compatibility probe did not authorize trusted generation; "
            f"repeated changed pixels={self.repeated_render.changed_pixel_count}, "
            "fresh-environment changed pixels="
            f"{self.fresh_environment_render.changed_pixel_count}"
        )

    def _semantic_payload(self) -> dict[str, object]:
        return {
            "camera_calibration_stable": self.camera_calibration_stable,
            "camera_rig_digest": self.camera_rig_digest,
            "compared_state_component_count": self.compared_state_component_count,
            "elapsed_simulation_step_changed": self.elapsed_simulation_step_changed,
            "environment_close_passed": self.environment_close_passed,
            "fresh_environment_render": self.fresh_environment_render.to_dict(),
            "cuda_runtime_version": self.cuda_runtime_version,
            "gpu_capability": self.gpu_capability,
            "gpu_model": self.gpu_model,
            "mani_skill_version": self.mani_skill_version,
            "pickcube_compatibility_identity": self.pickcube_compatibility_identity,
            "probe_source": dict(self.probe_source.as_mapping()),
            "render_output_shape": list(self.render_output_shape),
            "render_domain_configuration_digest": (
                self.render_domain_configuration_digest
            ),
            "renderer_api": dict(self.renderer_api.as_mapping()),
            "renderer_semantic_version": self.renderer_semantic_version,
            "repeated_render": self.repeated_render.to_dict(),
            "sapien_version": self.sapien_version,
            "schema_version": self.schema_version,
            "state_changed_by_rendering": self.state_changed_by_rendering,
            "state_comparison_semantic": self.state_comparison_semantic,
            "state_comparison_tolerance": self.state_comparison_tolerance,
            "task_projection_changed_by_rendering": (
                self.task_projection_changed_by_rendering
            ),
            "torch_version": self.torch_version,
            "verifier_component_count": self.verifier_component_count,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the complete sanitized, hardware-bound visual report."""
        return {
            **self._semantic_payload(),
            "trusted_visual_generation_ready": self.trusted_visual_generation_ready,
            "visual_compatibility_identity": self.visual_compatibility_identity,
        }


@runtime_checkable
class PickCubeVisualProbeRuntime(Protocol):
    """Injectable fresh-session collection boundary for CPU-only probe tests."""

    def collect_sessions(
        self,
        *,
        source_episode: PickCubeStateIndexedEpisodeV1,
        source_state: PickCubeIndexedStateV1,
        render_plan: PickCubeVisualRenderPlan,
    ) -> tuple[PickCubeVisualSessionResult, PickCubeVisualSessionResult]:
        """Return one repeated-render session and one independent fresh session."""
        ...


@dataclass(frozen=True, slots=True)
class SessionPickCubeVisualProbeRuntime:
    """Production collector backed by two independently created visual sessions."""

    session: PickCubeVisualSession

    def collect_sessions(
        self,
        *,
        source_episode: PickCubeStateIndexedEpisodeV1,
        source_state: PickCubeIndexedStateV1,
        render_plan: PickCubeVisualRenderPlan,
    ) -> tuple[PickCubeVisualSessionResult, PickCubeVisualSessionResult]:
        """Render twice in the first environment and once in a fresh environment."""
        repeated = self.session.render_state(
            source_episode=source_episode,
            source_state=source_state,
            render_plan=render_plan,
            repeat_count=2,
        )
        fresh = self.session.render_state(
            source_episode=source_episode,
            source_state=source_state,
            render_plan=render_plan,
            repeat_count=1,
        )
        return repeated, fresh


def _build_probe_source_evidence(
    source_archive: PickCubeStateIndexedArchiveV1,
    source_episode: PickCubeStateIndexedEpisodeV1,
    source_state: PickCubeIndexedStateV1,
) -> VisualProbeSourceEvidenceV1:
    """Prove that the selected state belongs to the supplied exact archive."""

    if not isinstance(source_archive, PickCubeStateIndexedArchiveV1):
        raise ManiSkillVisualProbeError(
            "visual probe requires a validated state-indexed archive"
        )
    archived_episodes = tuple(
        episode
        for episode in source_archive.episodes
        if episode.episode_id == source_episode.episode_id
    )
    if len(archived_episodes) != 1:
        raise ManiSkillVisualProbeError(
            "visual probe episode is absent from the supplied archive"
        )
    archived_episode = archived_episodes[0]
    if archived_episode.content_digest != source_episode.content_digest:
        raise ManiSkillVisualProbeError(
            "visual probe episode content differs from the supplied archive"
        )
    if source_state.state_index >= len(archived_episode.states):
        raise ManiSkillVisualProbeError(
            "visual probe state index is absent from the supplied episode"
        )
    archived_state = archived_episode.states[source_state.state_index]
    if archived_state.content_digest != source_state.content_digest:
        raise ManiSkillVisualProbeError(
            "visual probe state content differs from the supplied archive"
        )
    return VisualProbeSourceEvidenceV1(
        source_archive_digest=source_archive.content_digest,
        source_episode_id=archived_episode.episode_id,
        source_episode_content_digest=archived_episode.content_digest,
        source_trajectory_id=archived_episode.source_trajectory_id,
        source_reset_seed=archived_episode.seed,
        state_index=archived_state.state_index,
        state_content_digest=archived_state.content_digest,
        expected_state_digest=archived_state.state_digest,
        verifier_state_digest=archived_state.verifier_state.content_digest,
    )


def probe_maniskill_pickcube_visual(
    *,
    compatibility_binding: CompatibilityBinding,
    runtime: PickCubeVisualProbeRuntime,
    source_archive: PickCubeStateIndexedArchiveV1,
    source_episode: PickCubeStateIndexedEpisodeV1,
    source_state: PickCubeIndexedStateV1,
    camera_rig: PickCubeMultiViewRigV1,
    render_domain_configuration: RenderDomainConfigurationV1,
    render_plan: PickCubeVisualRenderPlan,
    report_path: Path | None = None,
    require_trusted_generation: bool = True,
) -> VisualCompatibilityReport:
    """Probe exact repeated/fresh RGB stability without executing an action."""
    compatibility_binding.require_trusted_replay_ready()
    report = compatibility_binding.report
    probe_source = _build_probe_source_evidence(
        source_archive, source_episode, source_state
    )
    if render_plan.domain_id != CANONICAL_DOMAIN_ID:
        raise ManiSkillVisualProbeError(
            "visual compatibility probe requires the canonical render domain"
        )
    expected_plan = build_pickcube_visual_render_plan(
        camera_rig,
        render_domain_configuration.domain(CANONICAL_DOMAIN_ID),
        render_domain_configuration,
        render_plan.render_seed,
    )
    if _render_plan_signature(render_plan) != _render_plan_signature(expected_plan):
        raise ManiSkillVisualProbeError(
            "probe render plan differs from the content-bound canonical configuration"
        )
    if (
        source_state.compatibility_identity != report.compatibility_identity
        or source_episode.compatibility_identity != report.compatibility_identity
    ):
        raise ManiSkillVisualProbeError(
            "probe archived state differs from the accepted PickCube compatibility"
        )
    repeated, fresh = runtime.collect_sessions(
        source_episode=source_episode,
        source_state=source_state,
        render_plan=render_plan,
    )
    _require_session_result(
        repeated, source_state=source_state, render_plan=render_plan, repeats=2
    )
    _require_session_result(
        fresh, source_state=source_state, render_plan=render_plan, repeats=1
    )
    if dict(repeated.renderer_api.as_mapping()) != dict(
        fresh.renderer_api.as_mapping()
    ):
        raise ManiSkillVisualProbeError(
            "fresh environment observed a different renderer API contract"
        )
    if (
        repeated.renderer_api.shader_configuration
        != render_domain_configuration.shader_configuration
    ):
        raise ManiSkillVisualProbeError(
            "renderer observed a shader different from the frozen configuration"
        )
    repeated_pixels, repeated_masks = _compare_view_sets(
        "repeated_render",
        repeated.render_repetitions[0],
        repeated.render_repetitions[1],
    )
    fresh_pixels, fresh_masks = _compare_view_sets(
        "fresh_environment_render",
        repeated.render_repetitions[0],
        fresh.render_repetitions[0],
    )
    spatially_stable = _masks_equal(repeated_masks, fresh_masks)
    repeated_pixels = replace(repeated_pixels, spatially_stable=spatially_stable)
    fresh_pixels = replace(fresh_pixels, spatially_stable=spatially_stable)
    calibration_stable = _calibration_stable(repeated, fresh)
    visual_report = VisualCompatibilityReport(
        pickcube_compatibility_identity=report.compatibility_identity,
        camera_rig_digest=camera_rig.rig_digest,
        render_domain_configuration_digest=(render_domain_configuration.content_digest),
        probe_source=probe_source,
        mani_skill_version=report.mani_skill_version,
        sapien_version=report.sapien_version,
        torch_version=report.operational.torch_version,
        cuda_runtime_version=report.operational.cuda_runtime_version,
        gpu_model=report.operational.gpu_model,
        gpu_capability=report.operational.gpu_capability,
        renderer_api=repeated.renderer_api,
        render_output_shape=VISUAL_RENDER_OUTPUT_SHAPE,
        state_comparison_semantic=repeated.integrity.state_verification_semantic,
        state_comparison_tolerance=(repeated.integrity.state_verification_tolerance),
        compared_state_component_count=(
            repeated.integrity.compared_state_component_count
        ),
        verifier_component_count=repeated.integrity.verifier_component_count,
        state_changed_by_rendering=False,
        task_projection_changed_by_rendering=False,
        elapsed_simulation_step_changed=False,
        repeated_render=repeated_pixels,
        fresh_environment_render=fresh_pixels,
        camera_calibration_stable=calibration_stable,
        environment_close_passed=(
            repeated.environment_close_passed and fresh.environment_close_passed
        ),
    )
    if report_path is not None:
        write_visual_compatibility_report(visual_report, report_path)
    if require_trusted_generation:
        visual_report.require_trusted_visual_generation_ready()
    return visual_report


def write_visual_compatibility_report(
    report: VisualCompatibilityReport, path: Path
) -> Path:
    """Atomically write one new compact report without overwriting evidence."""
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise ManiSkillVisualProbeError(
            "visual compatibility report refuses to overwrite existing evidence"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            report.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ManiSkillVisualProbeError(
                "visual compatibility report refuses to overwrite existing evidence"
            ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_visual_compatibility_report(
    path: Path, *, require_trusted_generation: bool = True
) -> VisualCompatibilityReport:
    """Strictly reload, recompute, and optionally authorize one probe report."""
    source = Path(path)
    try:
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_nlink != 1
            or source.stat().st_size > 1024 * 1024
        ):
            raise ManiSkillVisualProbeError(
                "visual compatibility report is missing or unsafe"
            )
        raw = cast(
            object,
            json.loads(
                source.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_fields,
            ),
        )
    except ManiSkillVisualProbeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManiSkillVisualProbeError(
            "visual compatibility report could not be read strictly"
        ) from exc
    item = _load_mapping(raw, "VisualCompatibilityReport")
    _exact_fields(
        item,
        {
            "camera_calibration_stable",
            "camera_rig_digest",
            "compared_state_component_count",
            "cuda_runtime_version",
            "elapsed_simulation_step_changed",
            "environment_close_passed",
            "fresh_environment_render",
            "gpu_capability",
            "gpu_model",
            "mani_skill_version",
            "pickcube_compatibility_identity",
            "probe_source",
            "render_output_shape",
            "render_domain_configuration_digest",
            "renderer_api",
            "renderer_semantic_version",
            "repeated_render",
            "sapien_version",
            "schema_version",
            "state_changed_by_rendering",
            "state_comparison_semantic",
            "state_comparison_tolerance",
            "task_projection_changed_by_rendering",
            "torch_version",
            "trusted_visual_generation_ready",
            "verifier_component_count",
            "visual_compatibility_identity",
        },
        "VisualCompatibilityReport",
    )
    shape_values = _load_sequence(
        item["render_output_shape"], "VisualCompatibilityReport.render_output_shape"
    )
    if len(shape_values) != 3:
        raise ManiSkillVisualProbeError(
            "VisualCompatibilityReport.render_output_shape must contain 3 values"
        )
    loaded = VisualCompatibilityReport(
        pickcube_compatibility_identity=_load_text(
            item["pickcube_compatibility_identity"],
            "VisualCompatibilityReport.pickcube_compatibility_identity",
        ),
        camera_rig_digest=_load_text(
            item["camera_rig_digest"],
            "VisualCompatibilityReport.camera_rig_digest",
        ),
        render_domain_configuration_digest=_load_text(
            item["render_domain_configuration_digest"],
            "VisualCompatibilityReport.render_domain_configuration_digest",
        ),
        probe_source=_decode_probe_source(item["probe_source"]),
        mani_skill_version=_load_text(
            item["mani_skill_version"], "VisualCompatibilityReport.mani_skill_version"
        ),
        sapien_version=_load_text(
            item["sapien_version"], "VisualCompatibilityReport.sapien_version"
        ),
        torch_version=_load_text(
            item["torch_version"], "VisualCompatibilityReport.torch_version"
        ),
        cuda_runtime_version=_load_text(
            item["cuda_runtime_version"],
            "VisualCompatibilityReport.cuda_runtime_version",
        ),
        gpu_model=_load_text(item["gpu_model"], "VisualCompatibilityReport.gpu_model"),
        gpu_capability=_load_text(
            item["gpu_capability"], "VisualCompatibilityReport.gpu_capability"
        ),
        renderer_api=_decode_renderer_api(item["renderer_api"]),
        render_output_shape=cast(
            tuple[int, int, int],
            tuple(
                _load_integer(
                    value,
                    f"VisualCompatibilityReport.render_output_shape[{index}]",
                )
                for index, value in enumerate(shape_values)
            ),
        ),
        state_comparison_semantic=_load_text(
            item["state_comparison_semantic"],
            "VisualCompatibilityReport.state_comparison_semantic",
        ),
        state_comparison_tolerance=_load_float(
            item["state_comparison_tolerance"],
            "VisualCompatibilityReport.state_comparison_tolerance",
        ),
        compared_state_component_count=_load_integer(
            item["compared_state_component_count"],
            "VisualCompatibilityReport.compared_state_component_count",
        ),
        verifier_component_count=_load_integer(
            item["verifier_component_count"],
            "VisualCompatibilityReport.verifier_component_count",
        ),
        state_changed_by_rendering=_load_boolean(
            item["state_changed_by_rendering"],
            "VisualCompatibilityReport.state_changed_by_rendering",
        ),
        task_projection_changed_by_rendering=_load_boolean(
            item["task_projection_changed_by_rendering"],
            "VisualCompatibilityReport.task_projection_changed_by_rendering",
        ),
        elapsed_simulation_step_changed=_load_boolean(
            item["elapsed_simulation_step_changed"],
            "VisualCompatibilityReport.elapsed_simulation_step_changed",
        ),
        repeated_render=_decode_pixel_report(item["repeated_render"]),
        fresh_environment_render=_decode_pixel_report(item["fresh_environment_render"]),
        camera_calibration_stable=_load_boolean(
            item["camera_calibration_stable"],
            "VisualCompatibilityReport.camera_calibration_stable",
        ),
        environment_close_passed=_load_boolean(
            item["environment_close_passed"],
            "VisualCompatibilityReport.environment_close_passed",
        ),
        renderer_semantic_version=_load_text(
            item["renderer_semantic_version"],
            "VisualCompatibilityReport.renderer_semantic_version",
        ),
        schema_version=_load_text(
            item["schema_version"], "VisualCompatibilityReport.schema_version"
        ),
    )
    declared_identity = _load_text(
        item["visual_compatibility_identity"],
        "VisualCompatibilityReport.visual_compatibility_identity",
    )
    declared_ready = _load_boolean(
        item["trusted_visual_generation_ready"],
        "VisualCompatibilityReport.trusted_visual_generation_ready",
    )
    if (
        loaded.visual_compatibility_identity != declared_identity
        or loaded.trusted_visual_generation_ready != declared_ready
    ):
        raise ManiSkillVisualProbeError(
            "visual compatibility report identity or readiness was tampered"
        )
    if require_trusted_generation:
        loaded.require_trusted_visual_generation_ready()
    return loaded


def _decode_probe_source(value: object) -> VisualProbeSourceEvidenceV1:
    item = _load_mapping(value, "VisualProbeSourceEvidenceV1")
    _exact_fields(
        item,
        {
            "expected_state_digest",
            "schema_version",
            "source_archive_digest",
            "source_episode_content_digest",
            "source_episode_id",
            "source_reset_seed",
            "source_trajectory_id",
            "state_content_digest",
            "state_index",
            "verifier_state_digest",
        },
        "VisualProbeSourceEvidenceV1",
    )
    return VisualProbeSourceEvidenceV1(
        source_archive_digest=_load_text(
            item["source_archive_digest"],
            "VisualProbeSourceEvidenceV1.source_archive_digest",
        ),
        source_episode_id=_load_text(
            item["source_episode_id"],
            "VisualProbeSourceEvidenceV1.source_episode_id",
        ),
        source_episode_content_digest=_load_text(
            item["source_episode_content_digest"],
            "VisualProbeSourceEvidenceV1.source_episode_content_digest",
        ),
        source_trajectory_id=_load_text(
            item["source_trajectory_id"],
            "VisualProbeSourceEvidenceV1.source_trajectory_id",
        ),
        source_reset_seed=_load_integer(
            item["source_reset_seed"],
            "VisualProbeSourceEvidenceV1.source_reset_seed",
        ),
        state_index=_load_integer(
            item["state_index"], "VisualProbeSourceEvidenceV1.state_index"
        ),
        state_content_digest=_load_text(
            item["state_content_digest"],
            "VisualProbeSourceEvidenceV1.state_content_digest",
        ),
        expected_state_digest=_load_text(
            item["expected_state_digest"],
            "VisualProbeSourceEvidenceV1.expected_state_digest",
        ),
        verifier_state_digest=_load_text(
            item["verifier_state_digest"],
            "VisualProbeSourceEvidenceV1.verifier_state_digest",
        ),
        schema_version=_load_text(
            item["schema_version"], "VisualProbeSourceEvidenceV1.schema_version"
        ),
    )


def _decode_renderer_api(value: object) -> VisualRendererApiObservation:
    item = _load_mapping(value, "VisualRendererApiObservation")
    fields = {
        "calibration_comparison_semantic",
        "camera_configuration_api",
        "camera_extrinsics_available",
        "camera_intrinsics_available",
        "camera_type",
        "color_space_assumption",
        "image_channel_order",
        "image_dtype",
        "renderer_backend",
        "rendering_requires_sensor_update_calls",
        "raw_color_texture_dtype",
        "raw_extrinsic_matrix_shape",
        "rgb_conversion_semantic",
        "runtime_extrinsics_dtype",
        "runtime_extrinsics_semantic",
        "runtime_intrinsics_dtype",
        "scene_type",
        "sensor_update_calls",
        "shader_configuration",
        "vertical_orientation",
        "world_camera_pose_representation",
    }
    _exact_fields(item, fields, "VisualRendererApiObservation")
    raw_calls = _load_sequence(
        item["sensor_update_calls"], "VisualRendererApiObservation.sensor_update_calls"
    )
    return VisualRendererApiObservation(
        calibration_comparison_semantic=_load_text(
            item["calibration_comparison_semantic"],
            "VisualRendererApiObservation.calibration_comparison_semantic",
        ),
        renderer_backend=_load_text(
            item["renderer_backend"], "VisualRendererApiObservation.renderer_backend"
        ),
        scene_type=_load_text(
            item["scene_type"], "VisualRendererApiObservation.scene_type"
        ),
        camera_type=_load_text(
            item["camera_type"], "VisualRendererApiObservation.camera_type"
        ),
        shader_configuration=_load_text(
            item["shader_configuration"],
            "VisualRendererApiObservation.shader_configuration",
        ),
        camera_configuration_api=_load_text(
            item["camera_configuration_api"],
            "VisualRendererApiObservation.camera_configuration_api",
        ),
        world_camera_pose_representation=_load_text(
            item["world_camera_pose_representation"],
            "VisualRendererApiObservation.world_camera_pose_representation",
        ),
        sensor_update_calls=tuple(
            _load_text(
                call, f"VisualRendererApiObservation.sensor_update_calls[{index}]"
            )
            for index, call in enumerate(raw_calls)
        ),
        image_dtype=_load_text(
            item["image_dtype"], "VisualRendererApiObservation.image_dtype"
        ),
        image_channel_order=_load_text(
            item["image_channel_order"],
            "VisualRendererApiObservation.image_channel_order",
        ),
        vertical_orientation=_load_text(
            item["vertical_orientation"],
            "VisualRendererApiObservation.vertical_orientation",
        ),
        color_space_assumption=_load_text(
            item["color_space_assumption"],
            "VisualRendererApiObservation.color_space_assumption",
        ),
        camera_intrinsics_available=_load_boolean(
            item["camera_intrinsics_available"],
            "VisualRendererApiObservation.camera_intrinsics_available",
        ),
        camera_extrinsics_available=_load_boolean(
            item["camera_extrinsics_available"],
            "VisualRendererApiObservation.camera_extrinsics_available",
        ),
        rendering_requires_sensor_update_calls=_load_boolean(
            item["rendering_requires_sensor_update_calls"],
            "VisualRendererApiObservation.rendering_requires_sensor_update_calls",
        ),
        raw_color_texture_dtype=_load_text(
            item["raw_color_texture_dtype"],
            "VisualRendererApiObservation.raw_color_texture_dtype",
        ),
        rgb_conversion_semantic=_load_text(
            item["rgb_conversion_semantic"],
            "VisualRendererApiObservation.rgb_conversion_semantic",
        ),
        runtime_intrinsics_dtype=_load_text(
            item["runtime_intrinsics_dtype"],
            "VisualRendererApiObservation.runtime_intrinsics_dtype",
        ),
        runtime_extrinsics_dtype=_load_text(
            item["runtime_extrinsics_dtype"],
            "VisualRendererApiObservation.runtime_extrinsics_dtype",
        ),
        raw_extrinsic_matrix_shape=_load_text(
            item["raw_extrinsic_matrix_shape"],
            "VisualRendererApiObservation.raw_extrinsic_matrix_shape",
        ),
        runtime_extrinsics_semantic=_load_text(
            item["runtime_extrinsics_semantic"],
            "VisualRendererApiObservation.runtime_extrinsics_semantic",
        ),
    )


def _decode_pixel_report(value: object) -> PixelComparisonReport:
    item = _load_mapping(value, "PixelComparisonReport")
    _exact_fields(
        item,
        {
            "changed_pixel_count",
            "comparison",
            "exact_match",
            "maximum_per_channel_absolute_difference",
            "mean_absolute_pixel_difference",
            "spatially_stable",
            "views",
        },
        "PixelComparisonReport",
    )
    views = tuple(
        _decode_camera_pixel_difference(view)
        for view in _load_sequence(item["views"], "PixelComparisonReport.views")
    )
    return PixelComparisonReport(
        comparison=_load_text(item["comparison"], "PixelComparisonReport.comparison"),
        views=views,
        changed_pixel_count=_load_integer(
            item["changed_pixel_count"], "PixelComparisonReport.changed_pixel_count"
        ),
        maximum_per_channel_absolute_difference=_load_integer(
            item["maximum_per_channel_absolute_difference"],
            "PixelComparisonReport.maximum_per_channel_absolute_difference",
        ),
        mean_absolute_pixel_difference=_load_float(
            item["mean_absolute_pixel_difference"],
            "PixelComparisonReport.mean_absolute_pixel_difference",
        ),
        exact_match=_load_boolean(
            item["exact_match"], "PixelComparisonReport.exact_match"
        ),
        spatially_stable=_load_boolean(
            item["spatially_stable"], "PixelComparisonReport.spatially_stable"
        ),
    )


def _decode_camera_pixel_difference(value: object) -> CameraPixelDifference:
    item = _load_mapping(value, "CameraPixelDifference")
    _exact_fields(
        item,
        {
            "camera_id",
            "changed_pixel_count",
            "exact_match",
            "maximum_per_channel_absolute_difference",
            "mean_absolute_pixel_difference",
        },
        "CameraPixelDifference",
    )
    return CameraPixelDifference(
        camera_id=_load_text(item["camera_id"], "CameraPixelDifference.camera_id"),
        changed_pixel_count=_load_integer(
            item["changed_pixel_count"], "CameraPixelDifference.changed_pixel_count"
        ),
        maximum_per_channel_absolute_difference=_load_integer(
            item["maximum_per_channel_absolute_difference"],
            "CameraPixelDifference.maximum_per_channel_absolute_difference",
        ),
        mean_absolute_pixel_difference=_load_float(
            item["mean_absolute_pixel_difference"],
            "CameraPixelDifference.mean_absolute_pixel_difference",
        ),
        exact_match=_load_boolean(
            item["exact_match"], "CameraPixelDifference.exact_match"
        ),
    )


def _load_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ManiSkillVisualProbeError(f"{context} must be an object")
    return cast(dict[str, object], value)


def _load_sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ManiSkillVisualProbeError(f"{context} must be an array")
    return cast(list[object], value)


def _exact_fields(item: Mapping[str, object], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(item))
    unexpected = sorted(set(item) - expected)
    if missing or unexpected:
        details = [
            *(f"missing {name}" for name in missing),
            *(f"unexpected {name}" for name in unexpected),
        ]
        raise ManiSkillVisualProbeError(
            f"{context} has invalid fields: {', '.join(details)}"
        )


def _load_text(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ManiSkillVisualProbeError(f"{context} must be canonical text")
    return value


def _load_integer(value: object, context: str) -> int:
    if type(value) is not int:
        raise ManiSkillVisualProbeError(f"{context} must be an integer")
    return value


def _load_float(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise ManiSkillVisualProbeError(f"{context} must be a finite number")
    numeric = cast(int | float, value)
    if not math.isfinite(float(numeric)):
        raise ManiSkillVisualProbeError(f"{context} must be a finite number")
    return float(numeric)


def _load_boolean(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise ManiSkillVisualProbeError(f"{context} must be boolean")
    return value


def _reject_json_constant(value: str) -> NoReturn:
    raise ManiSkillVisualProbeError(
        f"visual compatibility report rejects JSON constant {value!r}"
    )


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ManiSkillVisualProbeError(
                f"visual compatibility report has duplicate field {key!r}"
            )
        result[key] = value
    return result


def _require_session_result(
    result: PickCubeVisualSessionResult,
    *,
    source_state: PickCubeIndexedStateV1,
    render_plan: PickCubeVisualRenderPlan,
    repeats: int,
) -> None:
    if (
        not isinstance(result, PickCubeVisualSessionResult)
        or result.pickcube_compatibility_identity != source_state.compatibility_identity
        or result.state_content_digest != source_state.content_digest
        or result.integrity.expected_state_digest != source_state.state_digest
        or _render_plan_signature(result.render_plan)
        != _render_plan_signature(render_plan)
        or len(result.render_repetitions) != repeats
        or result.integrity.verified_render_count != repeats
    ):
        raise ManiSkillVisualProbeError(
            "visual probe session result differs from its exact source request"
        )


def _render_plan_signature(plan: PickCubeVisualRenderPlan) -> bytes:
    payload = {
        "cameras": [
            {
                "camera_configuration_digest": camera.camera_configuration_digest,
                "camera_id": camera.camera_id,
                "extrinsics": camera.extrinsics.tolist(),
                "far": camera.far,
                "fov_y_degrees": camera.fov_y_degrees,
                "height": camera.height,
                "intrinsics": camera.intrinsics.tolist(),
                "near": camera.near,
                "position": list(camera.position),
                "quaternion_wxyz": list(camera.quaternion_wxyz),
                "width": camera.width,
            }
            for camera in plan.cameras
        ],
        "domain_id": plan.domain_id,
        "lighting": {
            "ambient_intensity": plan.lighting.ambient_intensity,
            "key_color_rgb": list(plan.lighting.key_color_rgb),
            "key_direction": list(plan.lighting.key_direction),
            "key_intensity": plan.lighting.key_intensity,
        },
        "render_seed": plan.render_seed,
        "shader_configuration": plan.shader_configuration,
    }
    return canonical_json_bytes(payload, context="PickCubeVisualRenderPlan")


def _compare_view_sets(
    comparison: str,
    expected: tuple[RenderedVisualView, ...],
    observed: tuple[RenderedVisualView, ...],
) -> tuple[PixelComparisonReport, tuple[NDArray[np.bool_], ...]]:
    if tuple(view.camera_id for view in expected) != tuple(
        view.camera_id for view in observed
    ):
        raise ManiSkillVisualProbeError("pixel comparison camera order changed")
    evidence: list[CameraPixelDifference] = []
    masks: list[NDArray[np.bool_]] = []
    for left, right in zip(expected, observed, strict=True):
        difference = np.abs(
            left.rgb.astype(np.int16, copy=False)
            - right.rgb.astype(np.int16, copy=False)
        )
        mask = np.any(difference != 0, axis=2)
        changed = int(np.count_nonzero(mask))
        maximum = int(difference.max(initial=0))
        mean = float(np.mean(difference, dtype=np.float64))
        evidence.append(
            CameraPixelDifference(
                camera_id=left.camera_id,
                changed_pixel_count=changed,
                maximum_per_channel_absolute_difference=maximum,
                mean_absolute_pixel_difference=mean,
                exact_match=changed == 0,
            )
        )
        masks.append(np.array(mask, copy=True))
    views = tuple(evidence)
    return (
        PixelComparisonReport(
            comparison=comparison,
            views=views,
            changed_pixel_count=sum(view.changed_pixel_count for view in views),
            maximum_per_channel_absolute_difference=max(
                view.maximum_per_channel_absolute_difference for view in views
            ),
            mean_absolute_pixel_difference=sum(
                view.mean_absolute_pixel_difference for view in views
            )
            / len(views),
            exact_match=all(view.exact_match for view in views),
            spatially_stable=False,
        ),
        tuple(masks),
    )


def _masks_equal(
    left: tuple[NDArray[np.bool_], ...],
    right: tuple[NDArray[np.bool_], ...],
) -> bool:
    return len(left) == len(right) and all(
        np.array_equal(left_mask, right_mask)
        for left_mask, right_mask in zip(left, right, strict=True)
    )


def _calibration_stable(
    repeated: PickCubeVisualSessionResult, fresh: PickCubeVisualSessionResult
) -> bool:
    baseline = repeated.render_repetitions[0]
    comparisons = (*repeated.render_repetitions[1:], *fresh.render_repetitions)
    return all(
        left.camera_configuration_digest == right.camera_configuration_digest
        and np.array_equal(left.intrinsics, right.intrinsics)
        and np.array_equal(left.extrinsics, right.extrinsics)
        for comparison in comparisons
        for left, right in zip(baseline, comparison, strict=True)
    )


def _require_digest(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ManiSkillVisualProbeError(f"{context} must be a lowercase SHA-256")
    return value


def _digest_payload(value: object, *, context: str) -> str:
    encoded = canonical_json_bytes(value, context=context)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "CameraPixelDifference",
    "ManiSkillVisualProbeError",
    "PickCubeVisualProbeRuntime",
    "PixelComparisonReport",
    "SessionPickCubeVisualProbeRuntime",
    "VISUAL_COMPATIBILITY_REPORT_SCHEMA_VERSION",
    "VISUAL_PROBE_SOURCE_SCHEMA_VERSION",
    "VisualCompatibilityReport",
    "VisualProbeSourceEvidenceV1",
    "load_visual_compatibility_report",
    "probe_maniskill_pickcube_visual",
    "write_visual_compatibility_report",
]

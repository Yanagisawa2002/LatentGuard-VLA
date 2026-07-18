"""Exact restored-boundary visual sessions for ManiSkill PickCube."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.models import ReplayExecutionRole
from latentguard.vision_data.cameras import Matrix3, Matrix4, PickCubeMultiViewRigV1
from latentguard.vision_data.domains import RenderDomainConfigurationV1
from latentguard.vision_data.models import (
    SourceCollection,
    VisualDatasetSplit,
    VisualTaskProjectionV1,
)
from latentguard.vision_data.packet import (
    VisualObservationPacketV1,
    VisualViewRecordV1,
)
from latentguard.vision_data.serialization import prepare_npy_image

from .configuration import (
    PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE,
    PICKCUBE_STATE_VERIFICATION_SEMANTIC,
)
from .session import (
    LazyManiSkillPickCubeRuntime,
    ManiSkillPickCubeEnvironmentSettings,
    PickCubeReplayActionContract,
    PickCubeRuntime,
)
from .source_generation import (
    LazyManiSkillSourceEnvironmentFactory,
    SourceEnvironmentFactory,
)
from .state_indexed_archive import (
    PickCubeIndexedStateV1,
    PickCubeStateIndexedEpisodeV1,
    PickCubeTaskSnapshotV1,
)
from .state_indexed_runtime import (
    CapturedPickCubeStateBoundary,
    capture_pickcube_state_boundary,
)
from .state_tree import StateTreeComparison, clone_state_tree, compare_state_trees
from .task_evidence import PickCubeTaskKeyContract
from .visual_rendering import (
    VISUAL_RENDERER_SEMANTIC_VERSION,
    LazyManiSkillPickCubeVisualRenderer,
    PickCubeVisualRenderer,
    PickCubeVisualRenderPlan,
    RenderedVisualView,
    VisualRendererApiObservation,
    build_pickcube_visual_render_plan,
)


class ManiSkillVisualSessionError(RuntimeError):
    """Raised when rendering violates the restored pre-action state boundary."""


class ManiSkillVisualInvalidContextError(ManiSkillVisualSessionError):
    """Raised for deterministic source, restoration, or integrity mismatches."""


@dataclass(frozen=True, slots=True)
class VisualStateIntegrityEvidence:
    """Complete restoration and no-render-mutation evidence for one packet."""

    expected_state_digest: str
    state_before_render_digest: str
    state_after_render_digest: str
    compared_state_component_count: int
    restoration_maximum_absolute_error: float
    post_render_maximum_absolute_error: float
    render_mutation_maximum_absolute_error: float
    verifier_component_count: int
    verifier_state_maximum_absolute_error: float
    verifier_state_exact: bool
    task_projection_exact: bool
    verified_render_count: int
    elapsed_steps_before: int
    elapsed_steps_after: int
    state_verification_semantic: str = PICKCUBE_STATE_VERIFICATION_SEMANTIC
    state_verification_tolerance: float = (
        PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE
    )

    def __post_init__(self) -> None:
        """Require the complete fixed M4A integrity gate to have passed."""
        if self.compared_state_component_count != 70:
            raise ManiSkillVisualSessionError(
                "visual integrity evidence must compare exactly 70 state components"
            )
        if self.verifier_component_count != 38 or not self.verifier_state_exact:
            raise ManiSkillVisualSessionError(
                "visual integrity evidence requires exact 38-component verifier state"
            )
        if not self.task_projection_exact:
            raise ManiSkillVisualSessionError(
                "visual integrity evidence requires an unchanged task projection"
            )
        if (
            type(self.verified_render_count) is not int
            or self.verified_render_count <= 0
        ):
            raise ManiSkillVisualSessionError(
                "visual integrity evidence requires a positive verified render count"
            )
        for name in (
            "restoration_maximum_absolute_error",
            "post_render_maximum_absolute_error",
            "render_mutation_maximum_absolute_error",
            "verifier_state_maximum_absolute_error",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ManiSkillVisualSessionError(
                    f"visual integrity {name} must be finite and non-negative"
                )
        if (
            self.restoration_maximum_absolute_error
            > PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE
            or self.post_render_maximum_absolute_error
            > PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE
            or self.render_mutation_maximum_absolute_error != 0.0
            or self.verifier_state_maximum_absolute_error != 0.0
        ):
            raise ManiSkillVisualSessionError(
                "visual rendering did not preserve the complete state contract"
            )
        if self.state_before_render_digest != self.state_after_render_digest:
            raise ManiSkillVisualSessionError(
                "visual rendering changed the complete runtime state digest"
            )
        for name in ("elapsed_steps_before", "elapsed_steps_after"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ManiSkillVisualSessionError(
                    f"visual integrity {name} must be a non-negative integer"
                )
        if self.elapsed_steps_before != self.elapsed_steps_after:
            raise ManiSkillVisualSessionError(
                "visual rendering advanced the environment step counter"
            )


@dataclass(frozen=True, slots=True)
class PickCubeVisualSessionResult:
    """One fresh-session three-view render result ready for packet construction."""

    source_trajectory_id: str
    source_reset_seed: int
    pickcube_compatibility_identity: str
    state_index: int
    state_content_digest: str
    verifier_state_semantic: str
    verifier_state_content_digest: str
    restored_task_snapshot: PickCubeTaskSnapshotV1
    render_plan: PickCubeVisualRenderPlan
    render_repetitions: tuple[tuple[RenderedVisualView, ...], ...]
    integrity: VisualStateIntegrityEvidence
    renderer_api: VisualRendererApiObservation
    environment_close_passed: bool

    def __post_init__(self) -> None:
        """Require at least one ordered complete three-view render."""
        repetitions = tuple(tuple(views) for views in self.render_repetitions)
        if not repetitions:
            raise ManiSkillVisualSessionError("visual session produced no render")
        expected_ids = ("front_oblique", "overhead", "side_oblique")
        if any(
            tuple(view.camera_id for view in views) != expected_ids
            for views in repetitions
        ):
            raise ManiSkillVisualSessionError(
                "visual session result has an incomplete or reordered view set"
            )
        if type(self.environment_close_passed) is not bool:
            raise ManiSkillVisualSessionError(
                "visual session close result must be boolean"
            )
        object.__setattr__(self, "render_repetitions", repetitions)

    @property
    def views(self) -> tuple[RenderedVisualView, ...]:
        """Return the authoritative first render of all three views."""
        return self.render_repetitions[0]


@dataclass(frozen=True, slots=True)
class PreparedVisualObservationPacket:
    """Core visual packet plus exact reference-keyed RGB arrays."""

    packet: VisualObservationPacketV1
    images: MappingProxyType[str, NDArray[np.uint8]]

    def __post_init__(self) -> None:
        """Detach and freeze the exact three-image publication mapping."""
        expected = tuple(view.image_reference for view in self.packet.views)
        if tuple(self.images) != expected:
            raise ManiSkillVisualSessionError(
                "prepared packet image inventory differs from ordered views"
            )
        detached: dict[str, NDArray[np.uint8]] = {}
        for reference, image in self.images.items():
            prepared = prepare_npy_image(image)
            detached[reference] = prepared.image
        object.__setattr__(self, "images", MappingProxyType(detached))


@runtime_checkable
class PickCubeElapsedStepReader(Protocol):
    """Injectable public elapsed-step observation for no-step verification."""

    def elapsed_steps(self, environment: object) -> int:
        """Return the sole environment's non-negative elapsed-step count."""
        ...


@runtime_checkable
class TrustedVisualCompatibility(Protocol):
    """Reviewed visual probe fields required before packet construction."""

    @property
    def visual_compatibility_identity(self) -> str:
        """Return the independently derived visual compatibility identity."""
        ...

    @property
    def pickcube_compatibility_identity(self) -> str:
        """Return the accepted simulator compatibility identity."""
        ...

    @property
    def camera_rig_digest(self) -> str:
        """Return the exact reviewed camera-rig digest."""
        ...

    @property
    def render_domain_configuration_digest(self) -> str:
        """Return the exact reviewed render-domain configuration digest."""
        ...

    @property
    def renderer_api(self) -> VisualRendererApiObservation:
        """Return the exact renderer API contract observed by the probe."""
        ...

    def require_trusted_visual_generation_ready(self) -> None:
        """Fail unless the strict reviewed probe authorizes generation."""
        ...


class ManiSkillElapsedStepReader:
    """Read ManiSkill's public ``elapsed_steps`` property without stepping."""

    def elapsed_steps(self, environment: object) -> int:
        """Normalize one public scalar tensor/array to a Python integer."""
        base = getattr(environment, "unwrapped", environment)
        value = getattr(base, "elapsed_steps", None)
        candidate = value
        for method_name in ("detach", "cpu"):
            method = getattr(candidate, method_name, None)
            if callable(method):
                candidate = method()
        to_numpy = getattr(candidate, "numpy", None)
        if callable(to_numpy):
            candidate = to_numpy()
        array = np.asarray(candidate)
        if array.size != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ManiSkillVisualSessionError(
                "PickCube elapsed_steps must be one integer scalar"
            )
        result = int(array.reshape(()))
        if result < 0:
            raise ManiSkillVisualSessionError(
                "PickCube elapsed_steps must be non-negative"
            )
        return result


@dataclass(frozen=True, slots=True)
class PickCubeVisualSession:
    """Restore, render, verify, and close one independently owned environment."""

    settings: ManiSkillPickCubeEnvironmentSettings
    action_contract: PickCubeReplayActionContract
    task_key_contract: PickCubeTaskKeyContract
    state_runtime: PickCubeRuntime
    projection_factory: SourceEnvironmentFactory
    renderer: PickCubeVisualRenderer
    elapsed_step_reader: PickCubeElapsedStepReader
    environment_initialized_observer: Callable[[], None] | None = None

    def render_state(
        self,
        *,
        source_episode: PickCubeStateIndexedEpisodeV1,
        source_state: PickCubeIndexedStateV1,
        render_plan: PickCubeVisualRenderPlan,
        repeat_count: int = 1,
    ) -> PickCubeVisualSessionResult:
        """Render one exact restored state without exposing an action/step path."""
        self._preflight(source_episode, source_state, repeat_count)
        environment = self.state_runtime.create_environment(
            self.settings,
            self.action_contract,
            execution_role=ReplayExecutionRole.BASELINE,
        )
        primary_error: BaseException | None = None
        close_passed = False
        try:
            if self.environment_initialized_observer is not None:
                self.environment_initialized_observer()
            self.state_runtime.reset_environment(environment, seed=source_episode.seed)
            prepared = self.state_runtime.prepare_state_tree(
                environment, source_state.tree
            )
            self.state_runtime.set_state_dict(environment, prepared)
            observed_before = clone_state_tree(
                self.state_runtime.get_state_dict(environment)
            )
            restoration = _require_source_comparison(
                source_state, observed_before, context="visual restoration"
            )
            boundary_before = self._capture_boundary(environment, source_state)
            _require_boundary_matches_archive(source_state, boundary_before)
            before_after_capture = _require_exact_runtime_state(
                observed_before,
                boundary_before.state_tree,
                context="pre-render projection extraction",
            )
            elapsed_before = self.elapsed_step_reader.elapsed_steps(environment)
            handle = self.renderer.prepare(environment, render_plan)
            rendered: list[tuple[RenderedVisualView, ...]] = []
            post_render_errors: list[float] = []
            boundary_after = boundary_before
            observed_after = observed_before
            render_mutation = before_after_capture
            elapsed_after = elapsed_before
            for render_index in range(repeat_count):
                rendered.append(handle.render_views())
                boundary_after = self._capture_boundary(environment, source_state)
                observed_after = clone_state_tree(
                    self.state_runtime.get_state_dict(environment)
                )
                post_render = _require_source_comparison(
                    source_state,
                    observed_after,
                    context=f"post-render state {render_index + 1}",
                )
                post_render_errors.append(_maximum_error(post_render))
                render_mutation = _require_exact_runtime_state(
                    observed_before,
                    observed_after,
                    context=f"render mutation {render_index + 1}",
                )
                _require_boundary_matches_archive(source_state, boundary_after)
                _require_boundaries_equal(boundary_before, boundary_after)
                _require_exact_runtime_state(
                    boundary_after.state_tree,
                    observed_after,
                    context=f"post-render projection extraction {render_index + 1}",
                )
                elapsed_after = self.elapsed_step_reader.elapsed_steps(environment)
                if elapsed_after != elapsed_before:
                    raise ManiSkillVisualInvalidContextError(
                        "visual rendering advanced the environment step counter"
                    )
            repetitions = tuple(rendered)
            integrity = VisualStateIntegrityEvidence(
                expected_state_digest=source_state.state_digest,
                state_before_render_digest=before_after_capture.expected_digest,
                state_after_render_digest=render_mutation.observed_digest,
                compared_state_component_count=restoration.compared_component_count,
                restoration_maximum_absolute_error=_maximum_error(restoration),
                post_render_maximum_absolute_error=max(post_render_errors),
                render_mutation_maximum_absolute_error=_maximum_error(render_mutation),
                verifier_component_count=source_state.verifier_state.values.size,
                verifier_state_maximum_absolute_error=max(
                    _maximum_vector_error(
                        source_state.verifier_state.values,
                        boundary_before.verifier_state.values,
                    ),
                    _maximum_vector_error(
                        source_state.verifier_state.values,
                        boundary_after.verifier_state.values,
                    ),
                    _maximum_vector_error(
                        boundary_before.verifier_state.values,
                        boundary_after.verifier_state.values,
                    ),
                ),
                verifier_state_exact=True,
                task_projection_exact=True,
                verified_render_count=repeat_count,
                elapsed_steps_before=elapsed_before,
                elapsed_steps_after=elapsed_after,
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                self.state_runtime.close_environment(environment)
                close_passed = True
            except BaseException as close_error:
                if primary_error is None:
                    raise ManiSkillVisualSessionError(
                        "visual session environment close failed"
                    ) from close_error
                primary_error.add_note(
                    "visual session close also failed: " + type(close_error).__name__
                )
        return PickCubeVisualSessionResult(
            source_trajectory_id=source_episode.source_trajectory_id,
            source_reset_seed=source_episode.seed,
            pickcube_compatibility_identity=source_state.compatibility_identity,
            state_index=source_state.state_index,
            state_content_digest=source_state.content_digest,
            verifier_state_semantic=source_state.verifier_state.semantic,
            verifier_state_content_digest=source_state.verifier_state.content_digest,
            restored_task_snapshot=source_state.restored_task_snapshot,
            render_plan=render_plan,
            render_repetitions=repetitions,
            integrity=integrity,
            renderer_api=handle.api_observation,
            environment_close_passed=close_passed,
        )

    @staticmethod
    def _preflight(
        episode: PickCubeStateIndexedEpisodeV1,
        state: PickCubeIndexedStateV1,
        repeat_count: int,
    ) -> None:
        if not isinstance(episode, PickCubeStateIndexedEpisodeV1) or not isinstance(
            state, PickCubeIndexedStateV1
        ):
            raise ManiSkillVisualInvalidContextError(
                "visual session requires strictly loaded state-indexed models"
            )
        if (
            state.state_index >= len(episode.states)
            or episode.states[state.state_index].content_digest != state.content_digest
            or state.source_trajectory_id != episode.source_trajectory_id
            or state.seed != episode.seed
            or state.compatibility_identity != episode.compatibility_identity
        ):
            raise ManiSkillVisualInvalidContextError(
                "visual state identity differs from its source episode"
            )
        if state.numeric_component_count != 70:
            raise ManiSkillVisualInvalidContextError(
                "M4A requires exactly 70 complete state components"
            )
        vector = state.verifier_state.values
        if vector.shape != (38,) or vector.dtype != np.dtype("<f4"):
            raise ManiSkillVisualInvalidContextError(
                "M4A requires the exact 38-dimensional verifier-state contract"
            )
        if type(repeat_count) is not int or repeat_count <= 0:
            raise ManiSkillVisualInvalidContextError("repeat_count must be positive")

    def _capture_boundary(
        self, environment: object, source_state: PickCubeIndexedStateV1
    ) -> CapturedPickCubeStateBoundary:
        try:
            return capture_pickcube_state_boundary(
                environment,
                state_index=source_state.state_index,
                environment_factory=self.projection_factory,
                key_contract=self.task_key_contract,
            )
        except Exception as exc:
            raise ManiSkillVisualInvalidContextError(
                "could not capture the restored visual boundary"
            ) from exc


def create_default_pickcube_visual_session(
    *,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    task_key_contract: PickCubeTaskKeyContract,
    environment_initialized_observer: Callable[[], None] | None = None,
) -> PickCubeVisualSession:
    """Create the lazy production session without importing simulator packages."""
    return PickCubeVisualSession(
        settings=settings,
        action_contract=action_contract,
        task_key_contract=task_key_contract,
        state_runtime=LazyManiSkillPickCubeRuntime(),
        projection_factory=LazyManiSkillSourceEnvironmentFactory(),
        renderer=LazyManiSkillPickCubeVisualRenderer(),
        elapsed_step_reader=ManiSkillElapsedStepReader(),
        environment_initialized_observer=environment_initialized_observer,
    )


def build_visual_observation_packet(
    result: PickCubeVisualSessionResult,
    *,
    source_collection: SourceCollection,
    anchor_id: str,
    split: VisualDatasetSplit,
    split_group_id: str,
    state_reference_id: str,
    camera_rig: PickCubeMultiViewRigV1,
    render_domain_configuration: RenderDomainConfigurationV1,
    visual_compatibility: TrustedVisualCompatibility,
) -> PreparedVisualObservationPacket:
    """Convert one verified integration result to core packet and image types."""
    visual_compatibility.require_trusted_visual_generation_ready()
    if result.environment_close_passed is not True:
        raise ManiSkillVisualInvalidContextError(
            "visual packet requires a successfully closed render environment"
        )
    render_domain = render_domain_configuration.domain(result.render_plan.domain_id)
    expected_plan = build_pickcube_visual_render_plan(
        camera_rig,
        render_domain,
        render_domain_configuration,
        result.render_plan.render_seed,
    )
    if _render_plan_mapping(result.render_plan) != _render_plan_mapping(expected_plan):
        raise ManiSkillVisualInvalidContextError(
            "render result plan differs from the frozen visual configuration"
        )
    if (
        visual_compatibility.pickcube_compatibility_identity
        != result.pickcube_compatibility_identity
        or visual_compatibility.camera_rig_digest != camera_rig.rig_digest
        or visual_compatibility.render_domain_configuration_digest
        != render_domain_configuration.content_digest
    ):
        raise ManiSkillVisualInvalidContextError(
            "reviewed visual compatibility differs from packet configuration"
        )
    if dict(result.renderer_api.as_mapping()) != dict(
        visual_compatibility.renderer_api.as_mapping()
    ):
        raise ManiSkillVisualInvalidContextError(
            "runtime renderer API differs from the reviewed visual compatibility"
        )
    if result.render_plan.domain_id != render_domain.domain_id:
        raise ManiSkillVisualInvalidContextError(
            "render result domain differs from packet source metadata"
        )
    configured_camera_digests = tuple(
        camera.camera_configuration_digest for camera in camera_rig.cameras
    )
    planned_camera_digests = tuple(
        camera.camera_configuration_digest for camera in result.render_plan.cameras
    )
    observed_camera_digests = tuple(
        view.camera_configuration_digest for view in result.views
    )
    configured_camera_ids = tuple(camera.camera_id for camera in camera_rig.cameras)
    planned_camera_ids = tuple(
        camera.camera_id for camera in result.render_plan.cameras
    )
    if (
        configured_camera_ids != planned_camera_ids
        or planned_camera_ids != tuple(view.camera_id for view in result.views)
        or any(
            (base.width, base.height) != (planned.width, planned.height)
            for base, planned in zip(
                camera_rig.cameras, result.render_plan.cameras, strict=True
            )
        )
        or planned_camera_digests != observed_camera_digests
        or (
            render_domain.domain_id == "canonical"
            and configured_camera_digests != planned_camera_digests
        )
    ):
        raise ManiSkillVisualInvalidContextError(
            "rendered camera inventory differs from the content-bound rig"
        )
    task_projection = _visual_task_projection(result.restored_task_snapshot)
    prepared_images = tuple(prepare_npy_image(view.rgb) for view in result.views)

    def make_packet(prefix: str) -> VisualObservationPacketV1:
        views = tuple(
            VisualViewRecordV1(
                camera_id=view.camera_id,
                image_reference=f"{prefix}/{view.camera_id}.npy",
                pixel_sha256=prepared.pixel_sha256,
                npy_sha256=prepared.npy_sha256,
                dtype="uint8",
                shape=(224, 224, 3),
                intrinsics=_matrix3(view.intrinsics),
                intrinsics_dtype=view.intrinsics.dtype.name,
                extrinsics=_matrix4(view.extrinsics),
                extrinsics_dtype=view.extrinsics.dtype.name,
                camera_configuration_digest=view.camera_configuration_digest,
                state_before_render_digest=(
                    result.integrity.state_before_render_digest
                ),
                state_after_render_digest=result.integrity.state_after_render_digest,
                compared_state_component_count=(
                    result.integrity.compared_state_component_count
                ),
                maximum_state_error=max(
                    result.integrity.restoration_maximum_absolute_error,
                    result.integrity.post_render_maximum_absolute_error,
                ),
                task_projection_before=task_projection,
                task_projection_after=task_projection,
            )
            for view, prepared in zip(result.views, prepared_images, strict=True)
        )
        return VisualObservationPacketV1(
            source_collection=source_collection,
            source_trajectory_id=result.source_trajectory_id,
            anchor_id=anchor_id,
            split=split,
            split_group_id=split_group_id,
            state_reference_id=state_reference_id,
            expected_state_digest=result.integrity.expected_state_digest,
            verifier_state_semantic=result.verifier_state_semantic,
            verifier_state_digest=result.verifier_state_content_digest,
            verifier_state_component_count=(result.integrity.verifier_component_count),
            verifier_state_maximum_absolute_error=(
                result.integrity.verifier_state_maximum_absolute_error
            ),
            elapsed_simulation_steps_before=result.integrity.elapsed_steps_before,
            elapsed_simulation_steps_after=result.integrity.elapsed_steps_after,
            environment_close_passed=result.environment_close_passed,
            camera_rig_id=camera_rig.rig_id,
            camera_rig_digest=camera_rig.rig_digest,
            render_domain_id=render_domain.domain_id,
            render_domain_digest=render_domain.domain_digest,
            render_seed=result.render_plan.render_seed,
            views=views,
            visual_compatibility_identity=(
                visual_compatibility.visual_compatibility_identity
            ),
            pickcube_compatibility_identity=result.pickcube_compatibility_identity,
            renderer_semantic_version=VISUAL_RENDERER_SEMANTIC_VERSION,
        )

    provisional = make_packet("images/provisional")
    final_packet = make_packet(f"images/{provisional.packet_id}")
    if final_packet.packet_id != provisional.packet_id:
        raise ManiSkillVisualInvalidContextError(
            "packet identity unexpectedly depends on image output paths"
        )
    images = MappingProxyType(
        {
            view.image_reference: prepared.image
            for view, prepared in zip(final_packet.views, prepared_images, strict=True)
        }
    )
    return PreparedVisualObservationPacket(packet=final_packet, images=images)


def _require_source_comparison(
    expected: PickCubeIndexedStateV1, observed: object, *, context: str
) -> StateTreeComparison:
    comparison = compare_state_trees(
        expected.tree,
        observed,
        atol=PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE,
    )
    if (
        not comparison.structure_matches
        or not comparison.within_tolerance
        or comparison.expected_digest != expected.state_digest
        or comparison.expected_leaf_count != expected.leaf_count
        or comparison.observed_leaf_count != expected.leaf_count
        or comparison.compared_component_count != expected.numeric_component_count
        or comparison.compared_component_count != 70
        or comparison.maximum_absolute_error is None
    ):
        raise ManiSkillVisualInvalidContextError(
            f"{context} did not verify the complete 70-component state"
        )
    return comparison


def _require_exact_runtime_state(
    expected: object, observed: object, *, context: str
) -> StateTreeComparison:
    comparison = compare_state_trees(expected, observed, atol=0.0)
    if (
        not comparison.structure_matches
        or not comparison.exact_digest_match
        or not comparison.within_tolerance
        or comparison.compared_component_count != 70
        or comparison.maximum_absolute_error != 0.0
    ):
        raise ManiSkillVisualInvalidContextError(f"{context} changed runtime state")
    return comparison


def _require_boundary_matches_archive(
    expected: PickCubeIndexedStateV1, observed: CapturedPickCubeStateBoundary
) -> None:
    vector = observed.verifier_state
    archived = expected.verifier_state
    if (
        vector.semantic != archived.semantic
        or vector.schema_digest != archived.schema_digest
        or vector.component_names != archived.component_names
        or vector.values.dtype != archived.values.dtype
        or vector.values.shape != archived.values.shape
        or not np.array_equal(vector.values, archived.values)
    ):
        raise ManiSkillVisualInvalidContextError(
            "restored-boundary verifier state is not exactly reproducible"
        )
    if _task_snapshot(observed) != expected.restored_task_snapshot:
        raise ManiSkillVisualInvalidContextError(
            "restored-boundary task projection differs from the archive"
        )


def _require_boundaries_equal(
    before: CapturedPickCubeStateBoundary, after: CapturedPickCubeStateBoundary
) -> None:
    if (
        before.verifier_state.semantic != after.verifier_state.semantic
        or before.verifier_state.schema_digest != after.verifier_state.schema_digest
        or before.verifier_state.component_names != after.verifier_state.component_names
        or before.verifier_state.values.dtype != after.verifier_state.values.dtype
        or before.verifier_state.values.shape != after.verifier_state.values.shape
        or not np.array_equal(before.verifier_state.values, after.verifier_state.values)
        or _task_snapshot(before) != _task_snapshot(after)
    ):
        raise ManiSkillVisualInvalidContextError(
            "rendering changed verifier state or restored task projection"
        )


def _task_snapshot(boundary: CapturedPickCubeStateBoundary) -> PickCubeTaskSnapshotV1:
    task = boundary.task_snapshot
    try:
        return PickCubeTaskSnapshotV1(
            success=task["success"],  # type: ignore[arg-type]
            is_obj_placed=task["is_obj_placed"],  # type: ignore[arg-type]
            is_robot_static=task["is_robot_static"],  # type: ignore[arg-type]
            is_grasped=task["is_grasped"],  # type: ignore[arg-type]
            cube_center_z=task["cube_center_z"],  # type: ignore[arg-type]
            cube_to_goal_distance=task["cube_to_goal_distance"],  # type: ignore[arg-type]
            tcp_to_cube_distance=task["tcp_to_cube_distance"],  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManiSkillVisualInvalidContextError(
            "captured task projection is incomplete"
        ) from exc


def _visual_task_projection(
    value: PickCubeTaskSnapshotV1,
) -> VisualTaskProjectionV1:
    return VisualTaskProjectionV1(
        success=value.success,
        is_obj_placed=value.is_obj_placed,
        is_robot_static=value.is_robot_static,
        is_grasped=value.is_grasped,
        cube_center_z=value.cube_center_z,
        cube_to_goal_distance=value.cube_to_goal_distance,
        tcp_to_cube_distance=value.tcp_to_cube_distance,
    )


def _render_plan_mapping(plan: PickCubeVisualRenderPlan) -> tuple[object, ...]:
    cameras = tuple(
        (
            camera.camera_id,
            camera.width,
            camera.height,
            camera.near,
            camera.far,
            camera.fov_y_degrees,
            camera.position,
            camera.quaternion_wxyz,
            camera.intrinsics.dtype.str,
            camera.intrinsics.shape,
            camera.intrinsics.tobytes(order="C"),
            camera.extrinsics.dtype.str,
            camera.extrinsics.shape,
            camera.extrinsics.tobytes(order="C"),
            camera.camera_configuration_digest,
        )
        for camera in plan.cameras
    )
    return (
        plan.domain_id,
        plan.render_seed,
        plan.shader_configuration,
        cameras,
        plan.lighting.ambient_intensity,
        plan.lighting.key_intensity,
        plan.lighting.key_color_rgb,
        plan.lighting.key_direction,
    )


def _matrix3(value: NDArray[np.generic]) -> Matrix3:
    return cast(
        Matrix3,
        tuple(tuple(float(item) for item in row) for row in value.tolist()),
    )


def _matrix4(value: NDArray[np.generic]) -> Matrix4:
    return cast(
        Matrix4,
        tuple(tuple(float(item) for item in row) for row in value.tolist()),
    )


def _maximum_error(comparison: StateTreeComparison) -> float:
    value = comparison.maximum_absolute_error
    if value is None or not math.isfinite(value):
        raise ManiSkillVisualInvalidContextError(
            "state comparison lacks a finite error"
        )
    return float(value)


def _maximum_vector_error(
    expected: NDArray[np.generic], observed: NDArray[np.generic]
) -> float:
    """Measure the full verifier vector without omitting or coercing components."""

    if expected.shape != observed.shape or expected.dtype != observed.dtype:
        raise ManiSkillVisualInvalidContextError(
            "verifier-state comparison contract differs"
        )
    difference = np.abs(
        expected.astype(np.float64, copy=False)
        - observed.astype(np.float64, copy=False)
    )
    if difference.size == 0 or not bool(np.all(np.isfinite(difference))):
        raise ManiSkillVisualInvalidContextError(
            "verifier-state comparison lacks finite components"
        )
    return float(np.max(difference))


__all__ = [
    "ManiSkillElapsedStepReader",
    "ManiSkillVisualInvalidContextError",
    "ManiSkillVisualSessionError",
    "PickCubeElapsedStepReader",
    "PickCubeVisualSession",
    "PickCubeVisualSessionResult",
    "PreparedVisualObservationPacket",
    "TrustedVisualCompatibility",
    "VisualStateIntegrityEvidence",
    "build_visual_observation_packet",
    "create_default_pickcube_visual_session",
]

"""Real PickCube bindings for the M4C receding-horizon control protocol."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from latentguard.control.models import (
    ACTION_HORIZON,
    ClosedLoopCandidatePoolV1,
    EpisodeState,
    content_digest,
)
from latentguard.control.runner import (
    BoundSourcePlanV1,
    RuntimeBoundaryV1,
    RuntimeStepResultV1,
    simple_pool_from_actions,
)
from latentguard.corruptions.generation import derive_corruption_seed
from latentguard.corruptions.layout import ActionLayout
from latentguard.models import ActionChunk
from latentguard.selection.configuration import CandidatePoolConfigurationV1

from .closed_loop_state_store import ClosedLoopStateStore
from .session import (
    LazyManiSkillPickCubeRuntime,
    ManiSkillPickCubeEnvironmentSettings,
    PickCubeReplayActionContract,
)
from .source_generation import LazyManiSkillSourceEnvironmentFactory
from .state_indexed_archive import PickCubeStateIndexedArchiveV1
from .state_indexed_runtime import (
    CapturedPickCubeStateBoundary,
    capture_pickcube_state_boundary,
)
from .state_tree import (
    clone_state_tree,
    compare_state_trees,
    compute_state_tree_digest,
)
from .task_evidence import (
    PickCubeTaskKeyContract,
    build_pickcube_task_evidence,
)
from .visual_rendering import (
    LazyManiSkillPickCubeVisualRenderer,
    PickCubeVisualRenderPlan,
)


class VisualBoundaryObserver(Protocol):
    """Render three verified views without changing the physical boundary."""

    def prepare(self, environment: object, *, visual_domain: str) -> None:
        """Bind a fixed renderer to one freshly created environment."""
        ...

    def capture(
        self,
        environment: object,
        *,
        boundary: CapturedPickCubeStateBoundary,
        visual_domain: str,
    ) -> tuple[NDArray[Any], str]:
        """Return exact three-view bytes and their content identity."""
        ...


@dataclass(slots=True)
class PickCubeVisualBoundaryObserver:
    """M4A-compatible render observer with complete post-render verification."""

    plans: dict[str, PickCubeVisualRenderPlan]
    key_contract: PickCubeTaskKeyContract
    state_tolerance: float
    renderer: LazyManiSkillPickCubeVisualRenderer = field(
        default_factory=LazyManiSkillPickCubeVisualRenderer
    )
    _handle: Any = field(default=None, init=False, repr=False)
    _environment_factory: LazyManiSkillSourceEnvironmentFactory = field(
        default_factory=LazyManiSkillSourceEnvironmentFactory,
        init=False,
        repr=False,
    )

    def prepare(self, environment: object, *, visual_domain: str) -> None:
        """Prepare the exact domain plan on the current fresh environment."""

        try:
            plan = self.plans[visual_domain]
        except KeyError as exc:
            raise ValueError(f"unknown M4C visual domain {visual_domain!r}") from exc
        self._handle = self.renderer.prepare(environment, plan)

    def capture(
        self,
        environment: object,
        *,
        boundary: CapturedPickCubeStateBoundary,
        visual_domain: str,
    ) -> tuple[NDArray[Any], str]:
        """Render and prove that state, task facts, and verifier vector did not move."""

        if self._handle is None:
            raise RuntimeError("visual observer was not prepared")
        views = self._handle.render_views()
        if tuple(item.camera_id for item in views) != (
            "front_oblique",
            "overhead",
            "side_oblique",
        ):
            raise RuntimeError("visual observer camera order differs")
        after = capture_pickcube_state_boundary(
            environment,
            state_index=boundary.state_index,
            environment_factory=self._environment_factory,
            key_contract=self.key_contract,
        )
        comparison = compare_state_trees(
            boundary.state_tree, after.state_tree, atol=self.state_tolerance
        )
        if not comparison.structure_matches or not comparison.within_tolerance:
            raise RuntimeError("visual rendering changed complete simulator state")
        if dict(boundary.task_snapshot) != dict(after.task_snapshot):
            raise RuntimeError("visual rendering changed task projection")
        if boundary.verifier_state.values.tobytes(
            order="C"
        ) != after.verifier_state.values.tobytes(order="C"):
            raise RuntimeError("visual rendering changed verifier state")
        images = np.stack([item.rgb for item in views], axis=0)
        identity = content_digest(
            {
                "camera_configuration_digests": [
                    item.camera_configuration_digest for item in views
                ],
                "domain_id": visual_domain,
                "image_digests": [
                    hashlib.sha256(item.rgb.tobytes(order="C")).hexdigest()
                    for item in views
                ],
                "pre_state_digest": comparison.expected_digest,
                "schema_version": "1.0",
            },
            context="M4CVisualPacketV1",
        )
        return np.asarray(images, dtype=np.uint8, order="C"), identity


@dataclass(frozen=True, slots=True)
class PickCubeClosedLoopCandidateFactory:
    """Reapply the exact accepted M3C definitions to each nominal H=16 window."""

    configuration: CandidatePoolConfigurationV1
    action_layout: ActionLayout
    action_contract: PickCubeReplayActionContract
    action_contract_digest: str

    def __post_init__(self) -> None:
        """Require the pinned M3C dimensions and action-contract identity."""

        if self.configuration.candidate_horizon != ACTION_HORIZON:
            raise ValueError("M4C candidate horizon differs from accepted M3C")
        if (
            self.action_layout.action_dim != 8
            or self.action_contract.total_dimension != 8
        ):
            raise ValueError(
                "M4C requires the accepted eight-dimensional action contract"
            )
        if self.configuration.action_contract_digest != self.action_contract_digest:
            raise ValueError(
                "candidate configuration action contract differs from runtime"
            )

    def build(
        self,
        source: BoundSourcePlanV1,
        boundary: RuntimeBoundaryV1,
        *,
        decision_ordinal: int,
        nominal_plan_index: int,
    ) -> ClosedLoopCandidatePoolV1:
        """Build eight source-excluded candidates without clipping or repair."""

        source_prefix = np.array(
            source.actions[nominal_plan_index : nominal_plan_index + ACTION_HORIZON],
            copy=True,
            order="C",
            dtype=np.float64,
        )
        if source_prefix.shape != (ACTION_HORIZON, 8):
            raise ValueError(
                "M4C candidate boundary lacks a complete H=16 source window"
            )
        source_chunk = ActionChunk(
            actions=source_prefix,
            coordinate_frame=self.action_contract.coordinate_frame,
            control_period_s=self.action_contract.control_period_s,
        )
        source_bytes = source_prefix.tobytes(order="C")
        transformed_values: list[NDArray[Any]] = []
        for ordinal, definition in enumerate(self.configuration.candidates):
            corruption = definition.build_corruption()
            source_candidate_id = (
                f"{source.identity.source_trajectory_id}:"
                f"nominal-index-{nominal_plan_index}"
            )
            seed = derive_corruption_seed(
                base_seed=self.configuration.base_seed,
                source_episode_id=source.identity.source_trajectory_id,
                source_candidate_id=source_candidate_id,
                corruption_name=corruption.name,
                configuration_ordinal=ordinal,
            )
            corruption.resolved_parameters_for(source_chunk, self.action_layout)
            transformed = corruption.apply(source_chunk, self.action_layout, seed=seed)
            if source_chunk.actions.tobytes(order="C") != source_bytes:
                raise RuntimeError("M4C transformation mutated its source window")
            if (
                transformed.actions.dtype != source_prefix.dtype
                or transformed.actions.shape != source_prefix.shape
                or transformed.coordinate_frame != source_chunk.coordinate_frame
                or transformed.control_period_s != source_chunk.control_period_s
                or transformed.schema_version != source_chunk.schema_version
            ):
                raise RuntimeError("M4C transformation changed the action contract")
            candidate = np.array(
                transformed.actions, copy=True, order="C", dtype=np.float64
            )
            if candidate.tobytes(order="C") == source_bytes:
                raise RuntimeError("M4C exact source candidate is prohibited")
            self.action_contract.validate_action_chunk(
                transformed, role=f"m4c_candidate_slot_{ordinal}"
            )
            transformed_values.append(candidate)
        return simple_pool_from_actions(
            source,
            boundary,
            decision_ordinal=decision_ordinal,
            nominal_plan_index=nominal_plan_index,
            candidate_actions=tuple(transformed_values),
            candidate_pool_configuration_digest=self.configuration.content_digest,
        )


class PickCubeClosedLoopRuntime:
    """Fresh-session exact-state ManiSkill runtime for one M4C execution."""

    def __init__(
        self,
        *,
        source_archive: PickCubeStateIndexedArchiveV1,
        settings: ManiSkillPickCubeEnvironmentSettings,
        action_contract: PickCubeReplayActionContract,
        key_contract: PickCubeTaskKeyContract,
        state_store: ClosedLoopStateStore,
        visual_observer: VisualBoundaryObserver | None = None,
        runtime: LazyManiSkillPickCubeRuntime | None = None,
    ) -> None:
        """Bind reviewed source content and fixed compatibility contracts."""

        self._episodes = {
            item.source_trajectory_id: item for item in source_archive.episodes
        }
        self._settings = settings
        self._action_contract = action_contract
        self._key_contract = key_contract
        self._state_store = state_store
        self._visual_observer = visual_observer
        self._runtime = runtime or LazyManiSkillPickCubeRuntime()
        self._environment_factory = LazyManiSkillSourceEnvironmentFactory()
        self._environment: object | None = None
        self._source: BoundSourcePlanV1 | None = None
        self._state_index = 0

    def _create_and_restore(
        self,
        source: BoundSourcePlanV1,
        state: object,
        *,
        expected_state_digest: str,
        visual_domain: str,
    ) -> RuntimeBoundaryV1:
        self.close()
        from latentguard.replay.models import ReplayExecutionRole

        environment = self._runtime.create_environment(
            self._settings,
            self._action_contract,
            execution_role=ReplayExecutionRole.CORRUPTED,
        )
        try:
            self._runtime.reset_environment(
                environment, seed=source.identity.reset_seed
            )
            prepared = self._runtime.prepare_state_tree(environment, state)
            self._runtime.set_state_dict(environment, prepared)
            observed = clone_state_tree(self._runtime.get_state_dict(environment))
            comparison = compare_state_trees(
                state, observed, atol=self._settings.state_tolerance
            )
            if (
                not comparison.structure_matches
                or not comparison.within_tolerance
                or comparison.expected_digest != expected_state_digest
            ):
                raise RuntimeError("M4C complete-state restoration verification failed")
            self._environment = environment
            self._source = source
            if visual_domain != "not_applicable":
                if self._visual_observer is None:
                    raise RuntimeError("visual selector has no verified renderer")
                self._visual_observer.prepare(environment, visual_domain=visual_domain)
            return self._capture(visual_domain=visual_domain)
        except BaseException:
            self._runtime.close_environment(environment)
            self._environment = None
            self._source = None
            raise

    def start(
        self, source: BoundSourcePlanV1, *, visual_domain: str
    ) -> RuntimeBoundaryV1:
        """Restore the archived source initial state in a fresh environment."""

        try:
            episode = self._episodes[source.identity.source_trajectory_id]
        except KeyError as exc:
            raise ValueError("M4C source trajectory is absent from archive") from exc
        self._state_index = 0
        return self._create_and_restore(
            source,
            episode.states[0].tree,
            expected_state_digest=source.identity.initial_state_digest,
            visual_domain=visual_domain,
        )

    def restore_boundary(
        self,
        source: BoundSourcePlanV1,
        *,
        state_reference: str,
        expected_state_digest: str,
        visual_domain: str,
    ) -> RuntimeBoundaryV1:
        """Restore an immutable recovery snapshot without rescoring."""

        state = self._state_store.load(state_reference)
        if compute_state_tree_digest(state) != expected_state_digest:
            raise RuntimeError("M4C recovery reference digest differs")
        return self._create_and_restore(
            source,
            state,
            expected_state_digest=expected_state_digest,
            visual_domain=visual_domain,
        )

    def _capture(self, *, visual_domain: str) -> RuntimeBoundaryV1:
        if self._environment is None:
            raise RuntimeError("M4C runtime has no open environment")
        boundary = capture_pickcube_state_boundary(
            self._environment,
            state_index=self._state_index,
            environment_factory=self._environment_factory,
            key_contract=self._key_contract,
        )
        reference, digest = self._state_store.save(boundary.state_tree)
        images: NDArray[Any] | None = None
        visual_identity: str | None = None
        if visual_domain != "not_applicable":
            if self._visual_observer is None:
                raise RuntimeError("M4C visual observer is missing")
            images, visual_identity = self._visual_observer.capture(
                self._environment,
                boundary=boundary,
                visual_domain=visual_domain,
            )
        values = np.asarray(
            boundary.verifier_state.values, dtype=np.dtype("<f4"), order="C"
        )
        return RuntimeBoundaryV1(
            state_reference=reference,
            state_digest=digest,
            verifier_state=values,
            visual_packet_identity=visual_identity,
            images=images,
        )

    def execute(
        self,
        actions: NDArray[Any],
        *,
        maximum_control_steps_remaining: int,
        visual_domain: str,
    ) -> RuntimeStepResultV1:
        """Execute exact selected rows and evaluate official task facts per row."""

        if self._environment is None:
            raise RuntimeError("M4C runtime has no open environment")
        if (
            not isinstance(actions, np.ndarray)
            or actions.dtype != np.dtype("<f8")
            or actions.ndim != 2
            or actions.shape[1] != 8
            or not 0 < actions.shape[0] <= maximum_control_steps_remaining
        ):
            raise ValueError("M4C execution actions differ from the fixed contract")
        executed = 0
        outcome: EpisodeState | None = None
        evidence_payload: dict[str, object] | None = None
        for action in actions:
            before = action.tobytes(order="C")
            self._runtime.step_action(self._environment, action, self._action_contract)
            if action.tobytes(order="C") != before:
                raise RuntimeError("M4C runtime mutated an executed action row")
            executed += 1
            self._state_index += 1
            snapshot = self._runtime.capture_task_snapshot(
                self._environment, self._key_contract
            )
            evidence = build_pickcube_task_evidence(snapshot, self._key_contract)
            evidence_payload = {
                "diagnostics": dict(evidence.diagnostics),
                "failure_types": [
                    item.failure_type for item in evidence.failure_events
                ],
                "schema_version": evidence.schema_version,
                "status": evidence.status.value,
                "success": evidence.success,
                "termination_reason": evidence.termination_reason,
                "unsafe": evidence.unsafe,
            }
            if evidence.unsafe is True:
                outcome = EpisodeState.UNSAFE
                break
            if evidence.success is True:
                outcome = EpisodeState.SUCCESS
                break
        if evidence_payload is None:
            raise RuntimeError("M4C execution produced no task evidence")
        boundary = self._capture(visual_domain=visual_domain)
        return RuntimeStepResultV1(
            next_boundary=boundary,
            executed_step_count=executed,
            task_evidence_digest=content_digest(
                evidence_payload, context="M4CTaskEvidenceV1"
            ),
            outcome=outcome,
        )

    def close(self) -> None:
        """Close the owned environment once and retain no simulator object."""

        if self._environment is not None:
            environment, self._environment = self._environment, None
            try:
                self._runtime.close_environment(environment)
            finally:
                self._source = None


__all__ = [
    "PickCubeClosedLoopCandidateFactory",
    "PickCubeClosedLoopRuntime",
    "PickCubeVisualBoundaryObserver",
    "VisualBoundaryObserver",
]

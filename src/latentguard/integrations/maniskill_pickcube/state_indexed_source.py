"""Official-source collection with complete T+1 PickCube state capture."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

from .archive import ManiSkillReferenceArchive, ManiSkillReferenceEpisode
from .session import (
    ManiSkillPickCubeEnvironmentSettings,
    PickCubeReplayActionContract,
)
from .source_generation import (
    LazyManiSkillSourceEnvironmentFactory,
    PickCubeSourceGenerationError,
    RecordedSourceTrajectory,
    RecordingEnvironmentProxy,
    SolverSourceIdentity,
    SourceAttemptRecord,
    SourceEnvironmentFactory,
    build_reference_episode,
    ordered_source_seeds,
    validate_independent_source_baseline,
)
from .source_import import OFFICIAL_PICKCUBE_SOURCE_POLICY_ID
from .state_indexed_archive import (
    PickCubeIndexedStateV1,
    PickCubeStateIndexedArchiveV1,
    PickCubeStateIndexedEpisodeV1,
    PickCubeTaskSnapshotV1,
)
from .state_indexed_runtime import (
    CapturedPickCubeStateBoundary,
    capture_pickcube_state_boundary,
)
from .state_tree import clone_state_tree, compare_state_trees, compute_state_tree_digest
from .task_evidence import (
    PickCubeTaskKeyContract,
    build_pickcube_task_evidence,
)


class StateIndexedSourceCollectionIncompleteError(PickCubeSourceGenerationError):
    """Raised when the bounded seed range cannot produce enough accepted sources."""

    def __init__(self, result: StateIndexedSourceCollectionResult) -> None:
        """Retain the compact bounded attempt record."""
        self.result = result
        super().__init__(
            "state-indexed source collection did not reach its requested count"
        )


@dataclass(frozen=True, slots=True)
class RecordedStateIndexedTrajectory:
    """One official source trajectory and every captured pre/post action state."""

    source: RecordedSourceTrajectory
    boundaries: tuple[CapturedPickCubeStateBoundary, ...]

    def __post_init__(self) -> None:
        """Require exactly T+1 ordered boundaries with matching endpoints."""
        boundaries = tuple(self.boundaries)
        object.__setattr__(self, "boundaries", boundaries)
        action_count = int(self.source.source_actions.shape[0])
        if len(boundaries) != action_count + 1:
            raise PickCubeSourceGenerationError(
                "recorded state-indexed source must contain exactly T+1 boundaries"
            )
        if tuple(boundary.state_index for boundary in boundaries) != tuple(
            range(action_count + 1)
        ):
            raise PickCubeSourceGenerationError(
                "recorded boundary indices must be contiguous and ordered"
            )
        if (
            compute_state_tree_digest(boundaries[0].state_tree)
            != self.source.initial_state_digest
            or compute_state_tree_digest(boundaries[-1].state_tree)
            != self.source.terminal_state_digest
        ):
            raise PickCubeSourceGenerationError(
                "recorded T+1 endpoints differ from the source initial/terminal state"
            )


@dataclass(frozen=True, slots=True)
class StateIndexedSourceCollectionResult:
    """All-or-nothing sequence collection plus deterministic attempt audit."""

    requested_success_count: int
    attempts: tuple[SourceAttemptRecord, ...]
    archive: PickCubeStateIndexedArchiveV1 | None
    reference_archive: ManiSkillReferenceArchive | None

    @property
    def accepted_count(self) -> int:
        """Return the independently baseline-validated accepted count."""
        return sum(attempt.accepted for attempt in self.attempts)

    def __post_init__(self) -> None:
        """Validate cardinality and paired sequence/reference publication state."""
        if (
            type(self.requested_success_count) is not int
            or self.requested_success_count <= 0
        ):
            raise PickCubeSourceGenerationError(
                "requested state-indexed source count must be positive"
            )
        object.__setattr__(self, "attempts", tuple(self.attempts))
        if (self.archive is None) != (self.reference_archive is None):
            raise PickCubeSourceGenerationError(
                "sequence and M2C reference archives must complete together"
            )
        if self.archive is not None:
            assert self.reference_archive is not None
            if (
                len(self.archive.episodes) != self.requested_success_count
                or len(self.reference_archive.episodes) != self.requested_success_count
                or self.accepted_count != self.requested_success_count
            ):
                raise PickCubeSourceGenerationError(
                    "complete state-indexed collection cardinality mismatch"
                )


@dataclass(frozen=True, slots=True)
class StateRestorationAuditRecord:
    """Compact evidence from one fresh-environment indexed-state round trip."""

    source_trajectory_id: str
    state_index: int
    compared_component_count: int
    maximum_absolute_error: float
    verifier_component_count: int
    verifier_maximum_absolute_error: float


def collect_state_indexed_reference_archive(
    *,
    requested_success_count: int,
    starting_seed: int,
    maximum_attempts: int,
    compatibility_identity: str,
    environment_factory: SourceEnvironmentFactory,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    key_contract: PickCubeTaskKeyContract,
    solver: Callable[..., object],
    solver_identity: SolverSourceIdentity,
    trajectory_action_limit: int | None = None,
) -> StateIndexedSourceCollectionResult:
    """Collect official sources, T+1 states, and independent full baselines."""
    if type(requested_success_count) is not int or requested_success_count <= 0:
        raise PickCubeSourceGenerationError(
            "requested state-indexed source count must be positive"
        )
    if requested_success_count > maximum_attempts:
        raise PickCubeSourceGenerationError(
            "requested state-indexed successes cannot exceed maximum attempts"
        )
    attempts: list[SourceAttemptRecord] = []
    sequence_episodes: list[PickCubeStateIndexedEpisodeV1] = []
    references: list[ManiSkillReferenceEpisode] = []
    for seed in ordered_source_seeds(starting_seed, maximum_attempts):
        try:
            recorded = record_official_state_indexed_trajectory(
                seed=seed,
                environment_factory=environment_factory,
                settings=settings,
                action_contract=action_contract,
                key_contract=key_contract,
                solver=solver,
                solver_identity=solver_identity,
                trajectory_action_limit=trajectory_action_limit,
            )
            baseline_terminal = validate_independent_source_baseline(
                recorded.source,
                environment_factory=environment_factory,
                settings=settings,
                action_contract=action_contract,
                key_contract=key_contract,
            )
            reference = build_reference_episode(
                recorded.source,
                independently_validated_terminal=baseline_terminal,
                compatibility_identity=compatibility_identity,
                settings=settings,
                action_contract=action_contract,
                key_contract=key_contract,
            )
            sequence = build_state_indexed_episode(recorded, reference=reference)
        except Exception as exc:
            attempts.append(
                SourceAttemptRecord(
                    seed=seed,
                    accepted=False,
                    failure_category=type(exc).__name__,
                )
            )
            continue
        attempts.append(
            SourceAttemptRecord(seed=seed, accepted=True, failure_category=None)
        )
        references.append(reference)
        sequence_episodes.append(sequence)
        if len(sequence_episodes) == requested_success_count:
            return StateIndexedSourceCollectionResult(
                requested_success_count=requested_success_count,
                attempts=tuple(attempts),
                archive=PickCubeStateIndexedArchiveV1(
                    episodes=tuple(sequence_episodes)
                ),
                reference_archive=ManiSkillReferenceArchive(episodes=tuple(references)),
            )
    incomplete = StateIndexedSourceCollectionResult(
        requested_success_count=requested_success_count,
        attempts=tuple(attempts),
        archive=None,
        reference_archive=None,
    )
    raise StateIndexedSourceCollectionIncompleteError(incomplete)


def record_official_state_indexed_trajectory(
    *,
    seed: int,
    environment_factory: SourceEnvironmentFactory,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    key_contract: PickCubeTaskKeyContract,
    solver: Callable[..., object],
    solver_identity: SolverSourceIdentity,
    trajectory_action_limit: int | None = None,
) -> RecordedStateIndexedTrajectory:
    """Run one official solver attempt while capturing s[0] through s[T]."""
    environment = environment_factory.create_environment(
        settings, action_contract, purpose="official_state_indexed_source_generation"
    )
    boundaries: list[CapturedPickCubeStateBoundary] = []

    def capture_boundary(current: object, state_index: int) -> None:
        boundary = capture_pickcube_state_boundary(
            current,
            state_index=state_index,
            environment_factory=environment_factory,
            key_contract=key_contract,
        )
        if state_index != len(boundaries):
            raise PickCubeSourceGenerationError(
                "state boundary capture order is incomplete"
            )
        boundaries.append(boundary)

    recorder = RecordingEnvironmentProxy(
        environment,
        action_contract,
        trajectory_action_limit=trajectory_action_limit,
        boundary_capture=capture_boundary,
    )
    primary: BaseException | None = None
    try:
        result = solver(recorder, seed=seed, debug=False, vis=False)
        if result == -1:
            raise PickCubeSourceGenerationError(
                "official solver reported motion-planning failure"
            )
        recorder.verify_interception_complete()
        if recorder.reset_seed != seed:
            raise PickCubeSourceGenerationError(
                "official solver used a different source reset seed"
            )
        terminal_task = environment_factory.capture_task_snapshot(
            environment, key_contract
        )
        terminal_evidence = build_pickcube_task_evidence(terminal_task, key_contract)
        if (
            terminal_evidence.status.value != "complete"
            or terminal_evidence.success is not True
        ):
            raise PickCubeSourceGenerationError(
                "official state-indexed source did not terminate successfully"
            )
        actions = np.stack(recorder.actions, axis=0)
        if len(boundaries) != actions.shape[0] + 1:
            raise PickCubeSourceGenerationError(
                "official source capture did not produce exactly T+1 states"
            )
        initial_state = recorder.initial_state
        terminal_state = clone_state_tree(boundaries[-1].state_tree)
        prepared = environment_factory.prepare_state_tree(environment, initial_state)
        base = getattr(environment, "unwrapped", environment)
        set_state = getattr(base, "set_state_dict", None)
        get_state = getattr(base, "get_state_dict", None)
        if not callable(set_state) or not callable(get_state):
            raise PickCubeSourceGenerationError(
                "source environment lacks public state round-trip methods"
            )
        set_state(prepared)
        observed_initial = clone_state_tree(get_state())
        comparison = compare_state_trees(
            initial_state, observed_initial, atol=settings.state_tolerance
        )
        if not comparison.structure_matches or not comparison.within_tolerance:
            raise PickCubeSourceGenerationError(
                "captured state-indexed initial state failed complete round trip"
            )
        joint_names, robot_state = environment_factory.extract_named_robot_state(
            environment
        )
        source = RecordedSourceTrajectory(
            seed=seed,
            source_actions=actions,
            initial_state=initial_state,
            terminal_state=terminal_state,
            terminal_task=terminal_task,
            joint_names=joint_names,
            robot_state=robot_state,
            solver_identity=solver_identity,
            initial_state_digest=compute_state_tree_digest(initial_state),
            terminal_state_digest=compute_state_tree_digest(terminal_state),
            source_action_digest=hashlib.sha256(actions.tobytes(order="C")).hexdigest(),
        )
        return RecordedStateIndexedTrajectory(
            source=source, boundaries=tuple(boundaries)
        )
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            recorder.close()
        except BaseException as close_error:
            if primary is None:
                raise
            primary.add_note(
                f"state-indexed source close also failed: {type(close_error).__name__}"
            )


def build_state_indexed_episode(
    recorded: RecordedStateIndexedTrajectory,
    *,
    reference: ManiSkillReferenceEpisode,
) -> PickCubeStateIndexedEpisodeV1:
    """Bind captured boundaries to an independently accepted M2C source identity."""
    states: list[PickCubeIndexedStateV1] = []
    for boundary in recorded.boundaries:
        task = boundary.task_snapshot
        states.append(
            PickCubeIndexedStateV1(
                state_index=boundary.state_index,
                source_action_index=boundary.state_index,
                tree=boundary.state_tree,
                task_snapshot=PickCubeTaskSnapshotV1(
                    success=cast(bool, task["success"]),
                    is_obj_placed=cast(bool, task["is_obj_placed"]),
                    is_robot_static=cast(bool, task["is_robot_static"]),
                    is_grasped=cast(bool, task["is_grasped"]),
                    cube_center_z=cast(float, task["cube_center_z"]),
                    cube_to_goal_distance=cast(float, task["cube_to_goal_distance"]),
                    tcp_to_cube_distance=cast(float, task["tcp_to_cube_distance"]),
                ),
                verifier_state=boundary.verifier_state,
                seed=reference.seed,
                compatibility_identity=reference.compatibility_identity,
                source_trajectory_id=reference.source_trajectory_id,
            )
        )
    return PickCubeStateIndexedEpisodeV1(
        episode_id=reference.episode_id,
        source_trajectory_id=reference.source_trajectory_id,
        source_policy_identity=OFFICIAL_PICKCUBE_SOURCE_POLICY_ID,
        seed=reference.seed,
        compatibility_identity=reference.compatibility_identity,
        source_actions=reference.source_actions,
        states=tuple(states),
    )


def verify_all_indexed_states_fresh(
    episode: PickCubeStateIndexedEpisodeV1,
    *,
    environment_factory: SourceEnvironmentFactory,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    key_contract: PickCubeTaskKeyContract,
) -> tuple[StateRestorationAuditRecord, ...]:
    """Verify every stored state in its own fresh reset/set/get environment."""
    records: list[StateRestorationAuditRecord] = []
    for indexed_state in episode.states:
        environment = environment_factory.create_environment(
            settings, action_contract, purpose="indexed_state_fresh_validation"
        )
        primary: BaseException | None = None
        try:
            reset = getattr(environment, "reset", None)
            if not callable(reset):
                raise PickCubeSourceGenerationError(
                    "fresh validation environment lacks public reset"
                )
            reset(seed=episode.seed)
            base = getattr(environment, "unwrapped", environment)
            set_state = getattr(base, "set_state_dict", None)
            get_state = getattr(base, "get_state_dict", None)
            if not callable(set_state) or not callable(get_state):
                raise PickCubeSourceGenerationError(
                    "fresh validation environment lacks public state methods"
                )
            prepared = environment_factory.prepare_state_tree(
                environment, indexed_state.tree
            )
            set_state(prepared)
            observed = clone_state_tree(get_state())
            comparison = compare_state_trees(
                indexed_state.tree, observed, atol=settings.state_tolerance
            )
            if not comparison.structure_matches or not comparison.within_tolerance:
                raise PickCubeSourceGenerationError(
                    "fresh indexed-state restoration failed full comparison"
                )
            recaptured = capture_pickcube_state_boundary(
                environment,
                state_index=indexed_state.state_index,
                environment_factory=environment_factory,
                key_contract=key_contract,
            )
            expected_vector = indexed_state.verifier_state
            observed_vector = recaptured.verifier_state
            if (
                expected_vector.schema_digest != observed_vector.schema_digest
                or expected_vector.values.shape != observed_vector.values.shape
            ):
                raise PickCubeSourceGenerationError(
                    "fresh restoration changed verifier-state schema"
                )
            vector_error = float(
                np.max(
                    np.abs(
                        expected_vector.values.astype(np.float64)
                        - observed_vector.values.astype(np.float64)
                    )
                )
            )
            if vector_error > settings.state_tolerance:
                raise PickCubeSourceGenerationError(
                    "fresh restoration changed verifier-state values beyond tolerance"
                )
            maximum_error = comparison.maximum_absolute_error
            if maximum_error is None:
                raise PickCubeSourceGenerationError(
                    "fresh restoration omitted maximum absolute error"
                )
            records.append(
                StateRestorationAuditRecord(
                    source_trajectory_id=episode.source_trajectory_id,
                    state_index=indexed_state.state_index,
                    compared_component_count=comparison.compared_component_count,
                    maximum_absolute_error=float(maximum_error),
                    verifier_component_count=expected_vector.values.size,
                    verifier_maximum_absolute_error=vector_error,
                )
            )
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close = getattr(environment, "close", None)
            if not callable(close):
                error = PickCubeSourceGenerationError(
                    "fresh validation environment lacks public close"
                )
                if primary is None:
                    raise error
                primary.add_note(str(error))
            else:
                try:
                    close()
                except BaseException as close_error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "fresh indexed-state close also failed: "
                        f"{type(close_error).__name__}"
                    )
    return tuple(records)


__all__ = [
    "LazyManiSkillSourceEnvironmentFactory",
    "RecordedStateIndexedTrajectory",
    "StateIndexedSourceCollectionIncompleteError",
    "StateIndexedSourceCollectionResult",
    "StateRestorationAuditRecord",
    "build_state_indexed_episode",
    "collect_state_indexed_reference_archive",
    "record_official_state_indexed_trajectory",
    "verify_all_indexed_states_fresh",
]

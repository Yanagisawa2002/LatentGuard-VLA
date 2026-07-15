"""Official-source collection with complete T+1 PickCube state capture."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
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
    run_official_solver,
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
from .verifier_state import PickCubeVerifierStateV1

_TASK_BOOLEAN_FIELDS = (
    "success",
    "is_obj_placed",
    "is_robot_static",
    "is_grasped",
)
_TASK_NUMERIC_FIELDS = (
    "cube_center_z",
    "cube_to_goal_distance",
    "tcp_to_cube_distance",
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
    source_to_restored_task_mismatch_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _FreshRestoredProjection:
    """One public projection captured after a verified fresh state restoration."""

    state_index: int
    task_snapshot: PickCubeTaskSnapshotV1
    verifier_state: PickCubeVerifierStateV1
    compared_component_count: int
    maximum_absolute_error: float


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
            restored_projections = _restore_recorded_boundaries_fresh(
                recorded,
                environment_factory=environment_factory,
                settings=settings,
                action_contract=action_contract,
                key_contract=key_contract,
            )
            sequence = build_state_indexed_episode(
                recorded,
                reference=reference,
                restored_projections=restored_projections,
            )
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
        result = run_official_solver(solver, recorder, seed=seed)
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
    restored_projections: tuple[_FreshRestoredProjection, ...],
) -> PickCubeStateIndexedEpisodeV1:
    """Bind source annotations and fresh restored projections to one archive."""
    if len(restored_projections) != len(recorded.boundaries) or tuple(
        projection.state_index for projection in restored_projections
    ) != tuple(boundary.state_index for boundary in recorded.boundaries):
        raise PickCubeSourceGenerationError(
            "restored public projections do not cover every T+1 boundary"
        )
    states: list[PickCubeIndexedStateV1] = []
    for boundary, restored in zip(
        recorded.boundaries, restored_projections, strict=True
    ):
        states.append(
            PickCubeIndexedStateV1(
                state_index=boundary.state_index,
                source_action_index=boundary.state_index,
                tree=boundary.state_tree,
                task_snapshot=_task_snapshot_from_mapping(boundary.task_snapshot),
                restored_task_snapshot=restored.task_snapshot,
                verifier_state=restored.verifier_state,
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


def _restore_recorded_boundaries_fresh(
    recorded: RecordedStateIndexedTrajectory,
    *,
    environment_factory: SourceEnvironmentFactory,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    key_contract: PickCubeTaskKeyContract,
) -> tuple[_FreshRestoredProjection, ...]:
    """Restore and publicly project every captured boundary before publication."""
    return tuple(
        _capture_fresh_restored_projection(
            state_tree=boundary.state_tree,
            state_index=boundary.state_index,
            source_seed=recorded.source.seed,
            expected_verifier_state=boundary.verifier_state,
            environment_factory=environment_factory,
            settings=settings,
            action_contract=action_contract,
            key_contract=key_contract,
            purpose="indexed_state_restored_projection_binding",
        )
        for boundary in recorded.boundaries
    )


def _capture_fresh_restored_projection(
    *,
    state_tree: object,
    state_index: int,
    source_seed: int,
    expected_verifier_state: PickCubeVerifierStateV1,
    environment_factory: SourceEnvironmentFactory,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    key_contract: PickCubeTaskKeyContract,
    purpose: str,
) -> _FreshRestoredProjection:
    """Restore one state in a fresh session and capture its public projection."""
    environment = environment_factory.create_environment(
        settings, action_contract, purpose=purpose
    )
    primary: BaseException | None = None
    try:
        reset = getattr(environment, "reset", None)
        if not callable(reset):
            raise PickCubeSourceGenerationError(
                "fresh validation environment lacks public reset"
            )
        reset(seed=source_seed)
        base = getattr(environment, "unwrapped", environment)
        set_state = getattr(base, "set_state_dict", None)
        get_state = getattr(base, "get_state_dict", None)
        if not callable(set_state) or not callable(get_state):
            raise PickCubeSourceGenerationError(
                "fresh validation environment lacks public state methods"
            )
        prepared = environment_factory.prepare_state_tree(environment, state_tree)
        set_state(prepared)
        observed_before = clone_state_tree(get_state())
        compared_component_count, maximum_error = (
            _require_complete_restoration_comparison(
                expected=state_tree,
                observed=observed_before,
                tolerance=settings.state_tolerance,
                context="fresh indexed-state restoration",
            )
        )
        verified_boundary_digest = compute_state_tree_digest(observed_before)
        recaptured = capture_pickcube_state_boundary(
            environment,
            state_index=state_index,
            environment_factory=environment_factory,
            key_contract=key_contract,
        )
        if compute_state_tree_digest(recaptured.state_tree) != verified_boundary_digest:
            raise PickCubeSourceGenerationError(
                "restored state changed before public projection extraction"
            )
        observed_after = clone_state_tree(get_state())
        if compute_state_tree_digest(observed_after) != verified_boundary_digest:
            raise PickCubeSourceGenerationError(
                "public projection extraction mutated the restored state"
            )
        _require_complete_restoration_comparison(
            expected=state_tree,
            observed=observed_after,
            tolerance=settings.state_tolerance,
            context="post-projection indexed-state restoration",
        )
        _require_same_verifier_schema(
            expected_verifier_state, recaptured.verifier_state
        )
        return _FreshRestoredProjection(
            state_index=state_index,
            task_snapshot=_task_snapshot_from_mapping(recaptured.task_snapshot),
            verifier_state=recaptured.verifier_state,
            compared_component_count=compared_component_count,
            maximum_absolute_error=maximum_error,
        )
    except BaseException as exc:
        primary = exc
        raise
    finally:
        _close_fresh_environment(environment, primary)


def _require_complete_restoration_comparison(
    *,
    expected: object,
    observed: object,
    tolerance: float,
    context: str,
) -> tuple[int, float]:
    """Validate one complete state-tree comparison and return its statistics."""
    comparison = compare_state_trees(expected, observed, atol=tolerance)
    maximum_error = comparison.maximum_absolute_error
    if (
        not comparison.structure_matches
        or not comparison.within_tolerance
        or comparison.expected_leaf_count <= 0
        or comparison.expected_leaf_count != comparison.observed_leaf_count
        or comparison.compared_component_count <= 0
        or maximum_error is None
        or not math.isfinite(maximum_error)
        or maximum_error > tolerance
    ):
        raise PickCubeSourceGenerationError(f"{context} failed full comparison")
    return comparison.compared_component_count, float(maximum_error)


def _close_fresh_environment(
    environment: object, primary: BaseException | None
) -> None:
    """Close a fresh environment without obscuring an earlier failure."""
    close = getattr(environment, "close", None)
    if not callable(close):
        error = PickCubeSourceGenerationError(
            "fresh validation environment lacks public close"
        )
        if primary is None:
            raise error
        primary.add_note(str(error))
        return
    try:
        close()
    except BaseException as close_error:
        if primary is None:
            raise
        primary.add_note(
            f"fresh indexed-state close also failed: {type(close_error).__name__}"
        )


def _task_snapshot_from_mapping(value: object) -> PickCubeTaskSnapshotV1:
    """Build one strict task snapshot from a captured public mapping."""
    if not isinstance(value, Mapping):
        raise PickCubeSourceGenerationError("captured task snapshot is not a mapping")
    task = cast(Mapping[str, object], value)
    try:
        return PickCubeTaskSnapshotV1(
            success=cast(bool, task["success"]),
            is_obj_placed=cast(bool, task["is_obj_placed"]),
            is_robot_static=cast(bool, task["is_robot_static"]),
            is_grasped=cast(bool, task["is_grasped"]),
            cube_center_z=cast(float, task["cube_center_z"]),
            cube_to_goal_distance=cast(float, task["cube_to_goal_distance"]),
            tcp_to_cube_distance=cast(float, task["tcp_to_cube_distance"]),
        )
    except (KeyError, TypeError) as exc:
        raise PickCubeSourceGenerationError(
            "captured task snapshot is incomplete"
        ) from exc


def _require_same_verifier_schema(
    expected: PickCubeVerifierStateV1, observed: PickCubeVerifierStateV1
) -> None:
    """Require the complete public vector schema without comparing its values."""
    if (
        observed.semantic != expected.semantic
        or observed.schema_digest != expected.schema_digest
        or observed.component_names != expected.component_names
        or observed.values.dtype != expected.values.dtype
        or observed.values.shape != expected.values.shape
        or expected.values.size <= 0
        or not bool(np.all(np.isfinite(expected.values)))
        or not bool(np.all(np.isfinite(observed.values)))
    ):
        raise PickCubeSourceGenerationError(
            "fresh restoration changed verifier-state schema"
        )


def _task_snapshot_mismatch_fields(
    expected: PickCubeTaskSnapshotV1,
    observed: PickCubeTaskSnapshotV1,
    *,
    atol: float,
) -> tuple[str, ...]:
    """Return deterministic field diagnostics for two complete task snapshots."""
    mismatches = [
        field
        for field in _TASK_BOOLEAN_FIELDS
        if getattr(expected, field) != getattr(observed, field)
    ]
    mismatches.extend(
        field
        for field in _TASK_NUMERIC_FIELDS
        if abs(float(getattr(expected, field)) - float(getattr(observed, field))) > atol
    )
    return tuple(mismatches)


def verify_all_indexed_states_fresh(
    episode: PickCubeStateIndexedEpisodeV1,
    *,
    environment_factory: SourceEnvironmentFactory,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    key_contract: PickCubeTaskKeyContract,
) -> tuple[StateRestorationAuditRecord, ...]:
    """Independently verify every tree, restored task snapshot, and vector."""
    records: list[StateRestorationAuditRecord] = []
    for indexed_state in episode.states:
        restored = _capture_fresh_restored_projection(
            state_tree=indexed_state.tree,
            state_index=indexed_state.state_index,
            source_seed=episode.seed,
            expected_verifier_state=indexed_state.verifier_state,
            environment_factory=environment_factory,
            settings=settings,
            action_contract=action_contract,
            key_contract=key_contract,
            purpose="indexed_state_fresh_validation",
        )
        restored_mismatches = _task_snapshot_mismatch_fields(
            indexed_state.restored_task_snapshot,
            restored.task_snapshot,
            atol=settings.state_tolerance,
        )
        if restored_mismatches:
            raise PickCubeSourceGenerationError(
                "fresh restoration changed restored task snapshot fields: "
                + ", ".join(restored_mismatches)
            )
        expected_vector = indexed_state.verifier_state
        observed_vector = restored.verifier_state
        vector_error = float(
            np.max(
                np.abs(
                    expected_vector.values.astype(np.float64)
                    - observed_vector.values.astype(np.float64)
                )
            )
        )
        if not math.isfinite(vector_error) or vector_error > settings.state_tolerance:
            raise PickCubeSourceGenerationError(
                "fresh restoration changed verifier-state values beyond tolerance"
            )
        records.append(
            StateRestorationAuditRecord(
                source_trajectory_id=episode.source_trajectory_id,
                state_index=indexed_state.state_index,
                compared_component_count=restored.compared_component_count,
                maximum_absolute_error=restored.maximum_absolute_error,
                verifier_component_count=expected_vector.values.size,
                verifier_maximum_absolute_error=vector_error,
                source_to_restored_task_mismatch_fields=(
                    _task_snapshot_mismatch_fields(
                        indexed_state.task_snapshot,
                        indexed_state.restored_task_snapshot,
                        atol=settings.state_tolerance,
                    )
                ),
            )
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

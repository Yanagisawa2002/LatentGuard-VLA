"""Transactional observe-select-execute loop with exact crash recovery."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from latentguard.control.config import ClosedLoopConfigurationV1
from latentguard.control.models import (
    ACTION_DIMENSION,
    ACTION_HORIZON,
    BoundaryLedgerEntryV1,
    BoundaryState,
    ClosedLoopCandidatePoolV1,
    ClosedLoopDecisionRecordV1,
    ClosedLoopEpisodeRecordV1,
    EpisodeState,
    SelectorOutputV1,
    SourcePlanIdentityV1,
    array_digest,
    content_digest,
)
from latentguard.control.serialization import ClosedLoopEpisodeStore


class ClosedLoopExecutionError(RuntimeError):
    """Raised when a runtime violates the M4C execution contract."""


@dataclass(frozen=True, slots=True, eq=False)
class BoundSourcePlanV1:
    """Runtime nominal action sequence plus its reviewed identity."""

    identity: SourcePlanIdentityV1
    actions: NDArray[Any]
    initial_state_reference: str

    def __post_init__(self) -> None:
        """Require finite full-length float64 PickCube actions."""

        if not isinstance(self.identity, SourcePlanIdentityV1):
            raise ClosedLoopExecutionError("source plan identity is invalid")
        if (
            not isinstance(self.actions, np.ndarray)
            or self.actions.dtype != np.dtype("<f8")
            or self.actions.ndim != 2
            or self.actions.shape[0] < ACTION_HORIZON
            or self.actions.shape[1] != ACTION_DIMENSION
            or not bool(np.all(np.isfinite(self.actions)))
        ):
            raise ClosedLoopExecutionError(
                "source plan actions must be finite float64 [T>=16,8]"
            )
        detached = np.array(self.actions, copy=True, order="C", subok=False)
        detached.setflags(write=False)
        object.__setattr__(self, "actions", detached)
        if (
            not isinstance(self.initial_state_reference, str)
            or not self.initial_state_reference
        ):
            raise ClosedLoopExecutionError("source initial-state reference is invalid")


@dataclass(frozen=True, slots=True, eq=False)
class RuntimeBoundaryV1:
    """Verified current physical state exposed through allowlisted runtime data."""

    state_reference: str
    state_digest: str
    verifier_state: NDArray[Any]
    visual_packet_identity: str | None = None
    images: NDArray[Any] | None = None

    def __post_init__(self) -> None:
        """Validate a 38-component state and optional three-view packet."""

        if not isinstance(self.state_reference, str) or not self.state_reference:
            raise ClosedLoopExecutionError("boundary state reference is invalid")
        if not isinstance(self.state_digest, str) or not self.state_digest.startswith(
            "sha256:"
        ):
            raise ClosedLoopExecutionError("boundary state digest is invalid")
        if (
            not isinstance(self.verifier_state, np.ndarray)
            or self.verifier_state.dtype != np.dtype("<f4")
            or self.verifier_state.shape != (38,)
            or not bool(np.all(np.isfinite(self.verifier_state)))
        ):
            raise ClosedLoopExecutionError(
                "boundary verifier state must be float32 [38]"
            )
        state = np.array(self.verifier_state, copy=True, order="C")
        state.setflags(write=False)
        object.__setattr__(self, "verifier_state", state)
        if self.images is not None:
            if (
                not isinstance(self.images, np.ndarray)
                or self.images.dtype != np.dtype(np.uint8)
                or self.images.shape != (3, 224, 224, 3)
                or self.visual_packet_identity is None
            ):
                raise ClosedLoopExecutionError(
                    "visual boundary must be uint8 [3,224,224,3]"
                )
            images = np.array(self.images, copy=True, order="C")
            images.setflags(write=False)
            object.__setattr__(self, "images", images)


@dataclass(frozen=True, slots=True)
class RuntimeStepResultV1:
    """Published post-execution boundary and distinct observed outcome."""

    next_boundary: RuntimeBoundaryV1
    executed_step_count: int
    task_evidence_digest: str
    outcome: EpisodeState | None

    def __post_init__(self) -> None:
        """Require positive work and only accepted physical terminal outcomes."""

        if not isinstance(self.next_boundary, RuntimeBoundaryV1):
            raise ClosedLoopExecutionError("step result is missing its next boundary")
        if type(self.executed_step_count) is not int or self.executed_step_count <= 0:
            raise ClosedLoopExecutionError("step result must execute positive work")
        if not isinstance(
            self.task_evidence_digest, str
        ) or not self.task_evidence_digest.startswith("sha256:"):
            raise ClosedLoopExecutionError("step result task evidence is invalid")
        if self.outcome is not None and self.outcome not in {
            EpisodeState.SUCCESS,
            EpisodeState.TASK_FAILURE,
            EpisodeState.UNSAFE,
        }:
            raise ClosedLoopExecutionError("runtime returned an unsupported outcome")


@runtime_checkable
class ClosedLoopRuntime(Protocol):
    """Simulator-independent exact-state episode runtime."""

    def start(
        self, source: BoundSourcePlanV1, *, visual_domain: str
    ) -> RuntimeBoundaryV1:
        """Create a fresh environment and verify the source initial state."""
        ...

    def restore_boundary(
        self,
        source: BoundSourcePlanV1,
        *,
        state_reference: str,
        expected_state_digest: str,
        visual_domain: str,
    ) -> RuntimeBoundaryV1:
        """Restore one exact persisted boundary in a fresh environment."""
        ...

    def execute(
        self,
        actions: NDArray[Any],
        *,
        maximum_control_steps_remaining: int,
        visual_domain: str,
    ) -> RuntimeStepResultV1:
        """Execute unchanged rows and evaluate the official task after every row."""
        ...

    def close(self) -> None:
        """Close the currently owned environment safely."""
        ...


@runtime_checkable
class CandidatePoolFactory(Protocol):
    """Build the identical outcome-free eight-candidate pool at one boundary."""

    def build(
        self,
        source: BoundSourcePlanV1,
        boundary: RuntimeBoundaryV1,
        *,
        decision_ordinal: int,
        nominal_plan_index: int,
    ) -> ClosedLoopCandidatePoolV1:
        """Construct one source-excluded pool or fail closed."""
        ...


@runtime_checkable
class ClosedLoopSelector(Protocol):
    """Frozen selector that cannot access future runtime outcomes."""

    @property
    def selector_id(self) -> str:
        """Return the fixed selector identity."""
        ...

    @property
    def visual(self) -> bool:
        """Return whether current RGB observations are required."""
        ...

    def select(
        self,
        pool: ClosedLoopCandidatePoolV1,
        boundary: RuntimeBoundaryV1,
    ) -> SelectorOutputV1:
        """Return finite scores and a deterministic outcome-blind ranking."""
        ...


@dataclass(frozen=True, slots=True)
class ClosedLoopRunResultV1:
    """One control run plus whether strict resume executed zero work."""

    episode: ClosedLoopEpisodeRecordV1
    zero_work_resume: bool


def episode_execution_id(
    source: SourcePlanIdentityV1, *, selector_id: str, visual_domain: str
) -> str:
    """Return a path-independent execution identity."""

    digest = content_digest(
        {
            "schema_version": "1.0",
            "selector_id": selector_id,
            "source_plan_digest": source.content_digest,
            "visual_domain": visual_domain,
        },
        context="ClosedLoopEpisodeExecutionIdentityV1",
    )
    return f"m4c-episode-{digest.removeprefix('sha256:')}"


def _record(
    *,
    episode_id: str,
    source: BoundSourcePlanV1,
    pool: ClosedLoopCandidatePoolV1,
    output: SelectorOutputV1,
    boundary: RuntimeBoundaryV1,
    visual_domain: str,
    stride: int,
) -> ClosedLoopDecisionRecordV1:
    if output.selector_id == "" or output.selector_id is None:
        raise ClosedLoopExecutionError("selector output identity is missing")
    if set(output.ranking) != set(pool.ordered_candidate_ids):
        raise ClosedLoopExecutionError("selector changed the candidate inventory")
    selected = output.ranking[0]
    probability = float(output.scores[selected]) if output.probabilities else None
    visual_identity = (
        boundary.visual_packet_identity if boundary.images is not None else None
    )
    return ClosedLoopDecisionRecordV1(
        episode_execution_id=episode_id,
        source_trajectory_id=source.identity.source_trajectory_id,
        selector_id=output.selector_id,
        visual_domain=visual_domain,
        decision_ordinal=pool.decision_ordinal,
        nominal_plan_index=pool.nominal_plan_index,
        pre_decision_state_digest=boundary.state_digest,
        candidate_pool_digest=pool.content_digest,
        ordered_candidate_ids=pool.ordered_candidate_ids,
        selector_scores=tuple(
            (candidate_id, float(output.scores[candidate_id]))
            for candidate_id in pool.ordered_candidate_ids
        ),
        deterministic_ranking=output.ranking,
        selected_candidate_id=selected,
        selected_predicted_failure_probability=probability,
        execution_stride=stride,
        checkpoint_ensemble_identity=output.checkpoint_ensemble_identity,
        visual_packet_identity=visual_identity,
    )


def _replace_boundary(
    episode: ClosedLoopEpisodeRecordV1,
    boundary: BoundaryLedgerEntryV1,
    *,
    event_ordinal: int,
) -> ClosedLoopEpisodeRecordV1:
    values = list(episode.boundaries)
    if boundary.decision_ordinal == len(values):
        values.append(boundary)
    elif 0 <= boundary.decision_ordinal < len(values):
        values[boundary.decision_ordinal] = boundary
    else:
        raise ClosedLoopExecutionError("boundary ledger ordinal is discontinuous")
    return replace(
        episode,
        boundaries=tuple(values),
        executed_control_steps=sum(item.executed_step_count for item in values),
        final_nominal_plan_index=(
            boundary.nominal_plan_index + boundary.executed_step_count
            if boundary.state is BoundaryState.COMPLETE
            else episode.final_nominal_plan_index
        ),
        state=EpisodeState.RUNNING,
        completed_event_ordinal=None,
        execution_error_type=None,
    )


def _terminal(
    episode: ClosedLoopEpisodeRecordV1,
    *,
    state: EpisodeState,
    event_ordinal: int,
    error_type: str | None = None,
) -> ClosedLoopEpisodeRecordV1:
    return replace(
        episode,
        state=state,
        completed_event_ordinal=event_ordinal,
        execution_error_type=error_type,
    )


def _selected_actions(
    source: BoundSourcePlanV1,
    pool: ClosedLoopCandidatePoolV1,
    decision: ClosedLoopDecisionRecordV1,
    *,
    config: ClosedLoopConfigurationV1,
) -> tuple[NDArray[Any], bool]:
    selected = next(
        (
            item
            for item in pool.candidates
            if item.candidate_id == decision.selected_candidate_id
        ),
        None,
    )
    if selected is None:
        raise ClosedLoopExecutionError("persisted selection is absent from its pool")
    remaining = source.actions.shape[0] - decision.nominal_plan_index
    if remaining < ACTION_HORIZON:
        raise ClosedLoopExecutionError(
            "variable-length tail candidate would be required"
        )
    tail = remaining < ACTION_HORIZON + config.execution_stride
    if not tail:
        return np.array(
            selected.action_chunk[: config.execution_stride], copy=True
        ), False
    residual = source.actions[decision.nominal_plan_index + ACTION_HORIZON :]
    if residual.shape[0] >= config.execution_stride:
        raise ClosedLoopExecutionError(
            "tail residual must contain at most three actions"
        )
    actions = np.concatenate((selected.action_chunk, residual), axis=0)
    return np.asarray(actions, dtype=np.float64, order="C"), True


def run_closed_loop_episode(
    source: BoundSourcePlanV1,
    *,
    selector: ClosedLoopSelector,
    visual_domain: str,
    config: ClosedLoopConfigurationV1,
    candidate_factory: CandidatePoolFactory,
    runtime: ClosedLoopRuntime,
    store: ClosedLoopEpisodeStore,
    interruption_hook: Callable[[str, int], None] | None = None,
) -> ClosedLoopRunResultV1:
    """Run or exactly resume one repeated blind closed-loop control episode."""

    if selector.visual != (visual_domain != "not_applicable"):
        raise ClosedLoopExecutionError("selector/domain applicability mismatch")
    expected_id = episode_execution_id(
        source.identity, selector_id=selector.selector_id, visual_domain=visual_domain
    )
    event = 0
    recovered = False
    try:
        if store.episode_path.exists():
            episode = store.load_episode()
            if (
                episode.episode_execution_id != expected_id
                or episode.source_plan_digest != source.identity.content_digest
                or episode.selector_id != selector.selector_id
                or episode.visual_domain != visual_domain
            ):
                raise ClosedLoopExecutionError("resume identity differs")
            if episode.state.terminal:
                return ClosedLoopRunResultV1(episode=episode, zero_work_resume=True)
            event = max(
                [episode.started_event_ordinal]
                + [
                    value
                    for item in episode.boundaries
                    for value in (
                        item.selection_event_ordinal,
                        item.execution_event_ordinal,
                    )
                    if value is not None
                ]
            )
            if not episode.boundaries:
                boundary = runtime.start(source, visual_domain=visual_domain)
            else:
                last = episode.boundaries[-1]
                if last.state is BoundaryState.COMPLETE:
                    if (
                        last.post_state_reference is None
                        or last.post_state_digest is None
                    ):
                        raise ClosedLoopExecutionError(
                            "complete boundary lost its state"
                        )
                    boundary = runtime.restore_boundary(
                        source,
                        state_reference=last.post_state_reference,
                        expected_state_digest=last.post_state_digest,
                        visual_domain=visual_domain,
                    )
                elif last.state in {BoundaryState.SELECTED, BoundaryState.EXECUTING}:
                    boundary = runtime.restore_boundary(
                        source,
                        state_reference=last.pre_state_reference,
                        expected_state_digest=last.pre_state_digest,
                        visual_domain=visual_domain,
                    )
                    recovered = True
                else:
                    raise ClosedLoopExecutionError(
                        "execution-error episode is not resumable"
                    )
        else:
            boundary = runtime.start(source, visual_domain=visual_domain)
            episode = ClosedLoopEpisodeRecordV1(
                episode_execution_id=expected_id,
                source_plan_digest=source.identity.content_digest,
                source_trajectory_id=source.identity.source_trajectory_id,
                selector_id=selector.selector_id,
                visual_domain=visual_domain,
                state=EpisodeState.RUNNING,
                boundaries=(),
                executed_control_steps=0,
                tail_fallback_count=0,
                final_nominal_plan_index=0,
                started_event_ordinal=0,
            )
            store.save_episode(episode)

        while not episode.state.terminal:
            if episode.executed_control_steps >= config.maximum_control_steps:
                event += 1
                episode = _terminal(
                    episode, state=EpisodeState.HORIZON_EXHAUSTED, event_ordinal=event
                )
                store.save_episode(episode)
                break
            nominal_index = episode.final_nominal_plan_index
            remaining = source.actions.shape[0] - nominal_index
            if remaining < ACTION_HORIZON:
                event += 1
                episode = _terminal(
                    episode, state=EpisodeState.HORIZON_EXHAUSTED, event_ordinal=event
                )
                store.save_episode(episode)
                break
            decision_ordinal = len(episode.boundaries)
            if recovered:
                ledger = episode.boundaries[-1]
                decision_ordinal = ledger.decision_ordinal
                nominal_index = ledger.nominal_plan_index
                pool = store.load_pool(decision_ordinal)
                decision = store.load_decision(decision_ordinal)
                if (
                    pool.content_digest != ledger.candidate_pool_digest
                    or decision.content_digest != ledger.decision_record_digest
                    or boundary.state_digest != ledger.pre_state_digest
                ):
                    raise ClosedLoopExecutionError("recovery content differs")
                ledger = replace(ledger, recovery_count=ledger.recovery_count + 1)
                episode = _replace_boundary(episode, ledger, event_ordinal=event)
                store.save_episode(episode)
                recovered = False
            else:
                pool = candidate_factory.build(
                    source,
                    boundary,
                    decision_ordinal=decision_ordinal,
                    nominal_plan_index=nominal_index,
                )
                if pool.pre_decision_state_digest != boundary.state_digest:
                    raise ClosedLoopExecutionError(
                        "candidate pool changed the boundary"
                    )
                store.save_pool(pool)
                output = selector.select(pool, boundary)
                decision = _record(
                    episode_id=expected_id,
                    source=source,
                    pool=pool,
                    output=output,
                    boundary=boundary,
                    visual_domain=visual_domain,
                    stride=config.execution_stride,
                )
                store.save_decision(decision)
                event += 1
                ledger = BoundaryLedgerEntryV1(
                    decision_ordinal=decision_ordinal,
                    nominal_plan_index=nominal_index,
                    state=BoundaryState.SELECTED,
                    pre_state_reference=boundary.state_reference,
                    pre_state_digest=boundary.state_digest,
                    candidate_pool_digest=pool.content_digest,
                    decision_record_digest=decision.content_digest,
                    selection_event_ordinal=event,
                )
                episode = _replace_boundary(episode, ledger, event_ordinal=event)
                store.save_episode(episode)
                if interruption_hook is not None:
                    interruption_hook("after_selection", decision_ordinal)

            actions, tail = _selected_actions(source, pool, decision, config=config)
            event += 1
            ledger = replace(
                episode.boundaries[decision_ordinal],
                state=BoundaryState.EXECUTING,
                execution_event_ordinal=event,
            )
            episode = _replace_boundary(episode, ledger, event_ordinal=event)
            store.save_episode(episode)
            if interruption_hook is not None:
                interruption_hook("during_stride", decision_ordinal)
            result = runtime.execute(
                actions,
                maximum_control_steps_remaining=(
                    config.maximum_control_steps - episode.executed_control_steps
                ),
                visual_domain=visual_domain,
            )
            event += 1
            ledger = replace(
                ledger,
                state=BoundaryState.COMPLETE,
                executed_step_count=result.executed_step_count,
                post_state_reference=result.next_boundary.state_reference,
                post_state_digest=result.next_boundary.state_digest,
                observed_task_evidence_digest=result.task_evidence_digest,
            )
            episode = _replace_boundary(episode, ledger, event_ordinal=event)
            if tail:
                episode = replace(
                    episode, tail_fallback_count=episode.tail_fallback_count + 1
                )
            store.save_episode(episode)
            if interruption_hook is not None:
                interruption_hook("after_stride_before_next_decision", decision_ordinal)
            boundary = result.next_boundary
            if result.outcome is not None:
                event += 1
                episode = _terminal(episode, state=result.outcome, event_ordinal=event)
                store.save_episode(episode)
                break
            if episode.final_nominal_plan_index >= source.actions.shape[0]:
                event += 1
                episode = _terminal(
                    episode, state=EpisodeState.HORIZON_EXHAUSTED, event_ordinal=event
                )
                store.save_episode(episode)
                break
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        if "episode" in locals() and not episode.state.terminal:
            event += 1
            if (
                episode.boundaries
                and episode.boundaries[-1].state is not BoundaryState.COMPLETE
            ):
                failed = replace(
                    episode.boundaries[-1], state=BoundaryState.EXECUTION_ERROR
                )
                episode = _replace_boundary(episode, failed, event_ordinal=event)
            episode = _terminal(
                episode,
                state=EpisodeState.EXECUTION_ERROR,
                event_ordinal=event + 1,
                error_type=type(exc).__name__,
            )
            store.save_episode(episode)
    finally:
        runtime.close()
    return ClosedLoopRunResultV1(episode=episode, zero_work_resume=False)


def simple_pool_from_actions(
    source: BoundSourcePlanV1,
    boundary: RuntimeBoundaryV1,
    *,
    decision_ordinal: int,
    nominal_plan_index: int,
    candidate_actions: tuple[NDArray[Any], ...],
    candidate_pool_configuration_digest: str,
) -> ClosedLoopCandidatePoolV1:
    """Build a strict M4C pool from already validated deterministic actions."""

    from latentguard.control.models import ClosedLoopCandidateV1

    if len(candidate_actions) != 8:
        raise ClosedLoopExecutionError("candidate generator must return exactly eight")
    source_prefix = np.asarray(
        source.actions[nominal_plan_index : nominal_plan_index + ACTION_HORIZON],
        dtype=np.float64,
        order="C",
    )
    source_digest = array_digest(source_prefix)
    candidates = tuple(
        ClosedLoopCandidateV1(
            candidate_id=(
                "m4c-candidate-"
                + hashlib.sha256(
                    f"{source.identity.source_trajectory_id}:{decision_ordinal}:{ordinal}:".encode()
                    + np.asarray(action, dtype=np.float64).tobytes(order="C")
                ).hexdigest()
            ),
            definition_ordinal=ordinal,
            action_chunk=np.asarray(action, dtype=np.dtype("<f8"), order="C"),
            action_mask=np.ones(ACTION_HORIZON, dtype=np.bool_),
            definition_identity=content_digest(
                {
                    "candidate_pool_configuration_digest": (
                        candidate_pool_configuration_digest
                    ),
                    "definition_ordinal": ordinal,
                    "schema_version": "1.0",
                },
                context="M4CCandidateDefinitionIdentityV1",
            ),
        )
        for ordinal, action in enumerate(candidate_actions)
    )
    return ClosedLoopCandidatePoolV1(
        source_trajectory_id=source.identity.source_trajectory_id,
        decision_ordinal=decision_ordinal,
        nominal_plan_index=nominal_plan_index,
        pre_decision_state_digest=boundary.state_digest,
        source_action_prefix_digest=source_digest,
        candidate_pool_configuration_digest=candidate_pool_configuration_digest,
        candidates=candidates,
    )


__all__ = [
    "BoundSourcePlanV1",
    "CandidatePoolFactory",
    "ClosedLoopExecutionError",
    "ClosedLoopRunResultV1",
    "ClosedLoopRuntime",
    "ClosedLoopSelector",
    "RuntimeBoundaryV1",
    "RuntimeStepResultV1",
    "episode_execution_id",
    "run_closed_loop_episode",
    "simple_pool_from_actions",
]

"""Fresh-session baseline validation for state-indexed PickCube anchors.

The validator is simulator-agnostic at import time.  A production or fake
``SourceEnvironmentFactory`` supplies the runtime boundary, which keeps the
default CPU test suite free of ManiSkill and GPU dependencies.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from latentguard.models import LabelSource, LabelStrength
from latentguard.replay.models import TerminalTaskEvidence, TerminalTaskStatus

from .anchors import PickCubeStateAnchor
from .session import (
    ManiSkillPickCubeEnvironmentSettings,
    PickCubeReplayActionContract,
)
from .source_generation import SourceEnvironmentFactory
from .state_indexed_archive import (
    PickCubeIndexedStateV1,
    PickCubeStateIndexedEpisodeV1,
)
from .state_indexed_build import (
    PICKCUBE_STATE_COMPARISON_SEMANTIC,
    PICKCUBE_STATE_COMPARISON_TOLERANCE,
    AnchorBaselineEvidenceV1,
    PickCubeActionControlContractV1,
    SourceContinuationIdentityV1,
    build_source_continuation_identity,
)
from .state_indexed_runtime import capture_pickcube_state_boundary
from .state_tree import (
    StateTreeComparison,
    StateTreeError,
    clone_state_tree,
    compare_state_trees,
    compute_state_tree_digest,
)
from .task_evidence import (
    PICKCUBE_PROGRESS_SEMANTIC,
    PICKCUBE_UNSAFE_SEMANTIC,
    PickCubeTaskKeyContract,
    RawPickCubeTaskSnapshot,
    build_pickcube_task_evidence,
)

ANCHOR_BASELINE_PURPOSE = "state_indexed_anchor_baseline"
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class AnchorBaselineValidatorError(ValueError):
    """Base error for a baseline that must not authorize source data."""

    reason_code = "baseline_validator_error"


class AnchorBaselineInvalidContextError(AnchorBaselineValidatorError):
    """Raised when archive, restoration, action, or task context is invalid."""

    reason_code = "baseline_invalid_context"


class AnchorBaselineFailureError(AnchorBaselineValidatorError):
    """Raised when complete source replay conclusively fails the official task."""

    reason_code = "baseline_task_failure"


class AnchorBaselineIntegrityError(AnchorBaselineInvalidContextError):
    """Raised when an input or forwarded action changes during validation."""

    reason_code = "baseline_integrity_error"


class AnchorBaselineRuntimeError(AnchorBaselineValidatorError):
    """Raised when runtime execution cannot complete or close safely."""

    reason_code = "baseline_runtime_error"


@dataclass(frozen=True, slots=True)
class _InputBinding:
    archive_content_digest: str
    source_episode_content_digest: str
    source_actions_dtype: str
    source_actions_shape: tuple[int, int]
    source_actions_bytes: bytes
    source_remaining_dtype: str
    source_remaining_shape: tuple[int, int]
    source_remaining_bytes: bytes
    source_state_content_digest: str
    source_state_digest: str
    continuation_content_digest: str
    action_control_contract_digest: str


class PickCubeAnchorBaselineValidator:
    """Validate one source remainder in an independently created environment."""

    def __init__(
        self,
        *,
        environment_factory: SourceEnvironmentFactory,
        settings: ManiSkillPickCubeEnvironmentSettings,
        runtime_action_contract: PickCubeReplayActionContract,
        task_key_contract: PickCubeTaskKeyContract,
    ) -> None:
        """Bind the probed environment, action, and official task contracts."""
        if not isinstance(settings, ManiSkillPickCubeEnvironmentSettings):
            raise AnchorBaselineInvalidContextError(
                "anchor baseline requires fixed PickCube environment settings"
            )
        if not isinstance(runtime_action_contract, PickCubeReplayActionContract):
            raise AnchorBaselineInvalidContextError(
                "anchor baseline requires the probed runtime action contract"
            )
        if not isinstance(task_key_contract, PickCubeTaskKeyContract):
            raise AnchorBaselineInvalidContextError(
                "anchor baseline requires the probed task-key contract"
            )
        if settings.state_tolerance != PICKCUBE_STATE_COMPARISON_TOLERANCE:
            raise AnchorBaselineInvalidContextError(
                "anchor baseline requires the authorized 1e-6 state tolerance"
            )
        self._environment_factory = environment_factory
        self._settings = settings
        self._runtime_action_contract = runtime_action_contract
        self._task_key_contract = task_key_contract

    def validate_anchor_baseline(
        self,
        *,
        archive_content_digest: str,
        source_episode: PickCubeStateIndexedEpisodeV1,
        source_state: PickCubeIndexedStateV1,
        anchor: PickCubeStateAnchor,
        source_remaining_actions: NDArray[Any],
        continuation: SourceContinuationIdentityV1,
        action_control_contract: PickCubeActionControlContractV1,
    ) -> AnchorBaselineEvidenceV1:
        """Restore ``s[t]``, execute ``a[t:T]``, and require official success."""
        binding = self._preflight(
            archive_content_digest=archive_content_digest,
            source_episode=source_episode,
            source_state=source_state,
            anchor=anchor,
            source_remaining_actions=source_remaining_actions,
            continuation=continuation,
            action_control_contract=action_control_contract,
        )
        environment = self._create_environment()
        primary_error: BaseException | None = None
        try:
            comparison = self._restore_and_compare(
                environment=environment,
                source_episode=source_episode,
                source_state=source_state,
            )
            verifier_component_count, verifier_maximum_error = (
                self._verify_restored_verifier_state(
                    environment=environment,
                    source_state=source_state,
                )
            )
            before = self._capture_complete_task_evidence(
                environment, boundary="before_candidate"
            )
            self._require_restored_task_snapshot(source_state, before)

            executed_count = 0
            after_candidate: TerminalTaskEvidence | None = None
            for row in source_remaining_actions:
                self._runtime_action_contract.validate_row(row)
                forwarded = np.array(row, copy=True, order="C", subok=False)
                forwarded_dtype = forwarded.dtype.str
                forwarded_shape = forwarded.shape
                forwarded_bytes = forwarded.tobytes(order="C")
                self._environment_factory.step_action(
                    environment, forwarded, self._runtime_action_contract
                )
                if (
                    forwarded.dtype.str != forwarded_dtype
                    or forwarded.shape != forwarded_shape
                    or forwarded.tobytes(order="C") != forwarded_bytes
                ):
                    raise AnchorBaselineIntegrityError(
                        "anchor baseline runtime mutated a forwarded source action"
                    )
                executed_count += 1
                if executed_count == anchor.candidate_horizon:
                    after_candidate = self._capture_complete_task_evidence(
                        environment, boundary="after_candidate"
                    )

            if executed_count != anchor.remaining_horizon:
                raise AnchorBaselineRuntimeError(
                    "anchor baseline did not execute the complete source remainder"
                )
            if after_candidate is None:
                raise AnchorBaselineRuntimeError(
                    "anchor baseline omitted the exact candidate boundary"
                )
            terminal = self._capture_complete_task_evidence(
                environment, boundary="terminal"
            )
            self._assert_inputs_unchanged(
                binding=binding,
                archive_content_digest=archive_content_digest,
                source_episode=source_episode,
                source_state=source_state,
                source_remaining_actions=source_remaining_actions,
                continuation=continuation,
                action_control_contract=action_control_contract,
            )
            if terminal.success is not True:
                raise AnchorBaselineFailureError(
                    "anchor source remainder did not reach official task success"
                )
            object_placed = _diagnostic_bool(
                terminal, "pickcube_object_placed", boundary="terminal"
            )
            robot_static = _diagnostic_bool(
                terminal, "pickcube_robot_static", boundary="terminal"
            )
            if not object_placed or not robot_static:
                raise AnchorBaselineInvalidContextError(
                    "successful terminal evidence omitted official success components"
                )
            maximum_error = comparison.maximum_absolute_error
            if maximum_error is None:
                raise AnchorBaselineInvalidContextError(
                    "anchor restoration omitted maximum absolute error"
                )
            progress_before = _required_binary_progress(before, "before_candidate")
            progress_after = _required_binary_progress(
                after_candidate, "after_candidate"
            )
            terminal_progress = _required_binary_progress(terminal, "terminal")
            terminal_unsafe = terminal.unsafe
            if type(terminal_unsafe) is not bool:
                raise AnchorBaselineInvalidContextError(
                    "terminal task evidence omitted unsafe status"
                )
            return AnchorBaselineEvidenceV1(
                anchor_id=anchor.anchor_id,
                source_archive_episode_id=source_episode.episode_id,
                source_trajectory_id=source_episode.source_trajectory_id,
                source_seed=source_episode.seed,
                state_index=source_state.state_index,
                source_state_digest=source_state.state_digest,
                compatibility_identity=source_episode.compatibility_identity,
                continuation_identity=continuation.content_digest,
                expected_action_count=anchor.remaining_horizon,
                executed_action_count=executed_count,
                complete_action_execution=True,
                state_restoration_verified=True,
                complete_state_comparison=True,
                compared_component_count=comparison.compared_component_count,
                maximum_absolute_error=float(maximum_error),
                verifier_state_restoration_verified=True,
                verifier_state_compared_component_count=verifier_component_count,
                verifier_state_maximum_absolute_error=verifier_maximum_error,
                comparison_semantic=PICKCUBE_STATE_COMPARISON_SEMANTIC,
                comparison_tolerance=self._settings.state_tolerance,
                official_terminal_success=True,
                official_object_placed=object_placed,
                official_robot_static=robot_static,
                progress_before=progress_before,
                progress_after_candidate=progress_after,
                progress_delta=progress_after - progress_before,
                terminal_progress=terminal_progress,
                progress_semantic=PICKCUBE_PROGRESS_SEMANTIC,
                terminal_unsafe=terminal_unsafe,
                unsafe_semantic=PICKCUBE_UNSAFE_SEMANTIC,
                label_source=LabelSource.SIMULATOR,
                label_strength=LabelStrength.STRONG,
                simulator_replay_verified=True,
            )
        except AnchorBaselineValidatorError as exc:
            primary_error = exc
            raise
        except (KeyboardInterrupt, SystemExit) as exc:
            primary_error = exc
            raise
        except Exception as exc:
            wrapped = AnchorBaselineRuntimeError(
                "anchor baseline runtime raised " + type(exc).__name__
            )
            primary_error = wrapped
            raise wrapped from exc
        finally:
            self._close_environment(environment, primary_error)

    def _preflight(
        self,
        *,
        archive_content_digest: str,
        source_episode: PickCubeStateIndexedEpisodeV1,
        source_state: PickCubeIndexedStateV1,
        anchor: PickCubeStateAnchor,
        source_remaining_actions: NDArray[Any],
        continuation: SourceContinuationIdentityV1,
        action_control_contract: PickCubeActionControlContractV1,
    ) -> _InputBinding:
        if _DIGEST_PATTERN.fullmatch(archive_content_digest) is None:
            raise AnchorBaselineInvalidContextError(
                "anchor baseline archive identity must be a sha256 digest"
            )
        if not isinstance(source_episode, PickCubeStateIndexedEpisodeV1):
            raise AnchorBaselineInvalidContextError(
                "anchor baseline source episode model is invalid"
            )
        if not isinstance(source_state, PickCubeIndexedStateV1):
            raise AnchorBaselineInvalidContextError(
                "anchor baseline source state model is invalid"
            )
        if not isinstance(anchor, PickCubeStateAnchor):
            raise AnchorBaselineInvalidContextError(
                "anchor baseline schedule record is invalid"
            )
        if not isinstance(continuation, SourceContinuationIdentityV1):
            raise AnchorBaselineInvalidContextError(
                "anchor baseline continuation identity is invalid"
            )
        if not isinstance(action_control_contract, PickCubeActionControlContractV1):
            raise AnchorBaselineInvalidContextError(
                "anchor baseline action-control contract is invalid"
            )
        if (
            anchor.source_trajectory_id != source_episode.source_trajectory_id
            or anchor.source_seed != source_episode.seed
            or anchor.state_index >= len(source_episode.states)
            or source_episode.states[anchor.state_index].content_digest
            != source_state.content_digest
            or source_state.state_index != anchor.state_index
            or anchor.remaining_horizon
            != source_episode.source_actions.shape[0] - anchor.state_index
        ):
            raise AnchorBaselineInvalidContextError(
                "anchor baseline source identities are inconsistent"
            )
        if source_state.task_snapshot.success:
            raise AnchorBaselineInvalidContextError(
                "anchor baseline cannot begin from terminal success"
            )
        if anchor.candidate_horizon > anchor.remaining_horizon:
            raise AnchorBaselineInvalidContextError(
                "anchor candidate horizon exceeds the source remainder"
            )
        if (
            action_control_contract.controller_mode != self._settings.control_mode
            or action_control_contract.coordinate_frame
            != self._runtime_action_contract.coordinate_frame
            or action_control_contract.control_period_s
            != self._runtime_action_contract.control_period_s
            or action_control_contract.action_dimension
            != self._runtime_action_contract.total_dimension
            or action_control_contract.action_dtype
            != source_episode.source_actions.dtype.str
        ):
            raise AnchorBaselineInvalidContextError(
                "anchor action-control contract differs from runtime/source semantics"
            )
        expected_remaining = source_episode.source_actions[anchor.state_index :]
        if not _same_array_bytes(source_remaining_actions, expected_remaining):
            raise AnchorBaselineIntegrityError(
                "anchor source remainder differs from the archived action suffix"
            )
        expected_continuation = build_source_continuation_identity(
            source_episode=source_episode,
            anchor_state_index=anchor.state_index,
            candidate_horizon=anchor.candidate_horizon,
            action_control_contract=action_control_contract,
        )
        if continuation.content_digest != expected_continuation.content_digest:
            raise AnchorBaselineIntegrityError(
                "anchor continuation differs from the archived fixed suffix"
            )
        for row in source_remaining_actions:
            try:
                self._runtime_action_contract.validate_row(row)
            except Exception as exc:
                raise AnchorBaselineInvalidContextError(
                    "anchor source remainder violates the runtime action contract"
                ) from exc
        return _InputBinding(
            archive_content_digest=archive_content_digest,
            source_episode_content_digest=source_episode.content_digest,
            source_actions_dtype=source_episode.source_actions.dtype.str,
            source_actions_shape=_rank_two_shape(source_episode.source_actions),
            source_actions_bytes=source_episode.source_actions.tobytes(order="C"),
            source_remaining_dtype=source_remaining_actions.dtype.str,
            source_remaining_shape=_rank_two_shape(source_remaining_actions),
            source_remaining_bytes=source_remaining_actions.tobytes(order="C"),
            source_state_content_digest=source_state.content_digest,
            source_state_digest=compute_state_tree_digest(source_state.tree),
            continuation_content_digest=continuation.content_digest,
            action_control_contract_digest=action_control_contract.content_digest,
        )

    def _create_environment(self) -> object:
        try:
            return self._environment_factory.create_environment(
                self._settings,
                self._runtime_action_contract,
                purpose=ANCHOR_BASELINE_PURPOSE,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise AnchorBaselineRuntimeError(
                "anchor baseline environment creation raised " + type(exc).__name__
            ) from exc

    def _restore_and_compare(
        self,
        *,
        environment: object,
        source_episode: PickCubeStateIndexedEpisodeV1,
        source_state: PickCubeIndexedStateV1,
    ) -> StateTreeComparison:
        reset = getattr(environment, "reset", None)
        if not callable(reset):
            raise AnchorBaselineRuntimeError(
                "anchor baseline environment lacks public reset"
            )
        reset(seed=source_episode.seed)
        base = getattr(environment, "unwrapped", environment)
        set_state = getattr(base, "set_state_dict", None)
        get_state = getattr(base, "get_state_dict", None)
        if not callable(set_state) or not callable(get_state):
            raise AnchorBaselineRuntimeError(
                "anchor baseline environment lacks public state methods"
            )
        prepared = self._environment_factory.prepare_state_tree(
            environment, source_state.tree
        )
        set_state(prepared)
        observed = clone_state_tree(get_state())
        try:
            comparison = compare_state_trees(
                source_state.tree,
                observed,
                atol=self._settings.state_tolerance,
            )
        except StateTreeError as exc:
            raise AnchorBaselineInvalidContextError(
                "anchor baseline could not compare the complete restored state"
            ) from exc
        maximum_error = comparison.maximum_absolute_error
        if (
            not comparison.structure_matches
            or not comparison.within_tolerance
            or comparison.expected_digest != source_state.state_digest
            or comparison.expected_leaf_count != source_state.leaf_count
            or comparison.observed_leaf_count != source_state.leaf_count
            or comparison.compared_component_count
            != source_state.numeric_component_count
            or maximum_error is None
            or not math.isfinite(maximum_error)
            or maximum_error > self._settings.state_tolerance
        ):
            raise AnchorBaselineInvalidContextError(
                "anchor baseline full-state restoration did not verify"
            )
        return comparison

    def _verify_restored_verifier_state(
        self,
        *,
        environment: object,
        source_state: PickCubeIndexedStateV1,
    ) -> tuple[int, float]:
        """Re-extract and compare every public verifier-state component."""

        try:
            recaptured = capture_pickcube_state_boundary(
                environment,
                state_index=source_state.state_index,
                environment_factory=self._environment_factory,
                key_contract=self._task_key_contract,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise AnchorBaselineInvalidContextError(
                "anchor baseline could not re-extract the restored verifier state"
            ) from exc
        expected = source_state.verifier_state
        observed = recaptured.verifier_state
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
            raise AnchorBaselineInvalidContextError(
                "anchor baseline restored verifier-state schema is incomplete"
            )
        error = float(
            np.max(
                np.abs(
                    expected.values.astype(np.float64)
                    - observed.values.astype(np.float64)
                )
            )
        )
        if not math.isfinite(error) or error > self._settings.state_tolerance:
            raise AnchorBaselineInvalidContextError(
                "anchor baseline restored verifier-state values exceed tolerance"
            )
        return int(expected.values.size), error

    def _capture_complete_task_evidence(
        self, environment: object, *, boundary: str
    ) -> TerminalTaskEvidence:
        snapshot = self._environment_factory.capture_task_snapshot(
            environment, self._task_key_contract
        )
        if not isinstance(snapshot, RawPickCubeTaskSnapshot):
            raise AnchorBaselineRuntimeError(
                f"anchor baseline {boundary} task snapshot is invalid"
            )
        evidence = build_pickcube_task_evidence(snapshot, self._task_key_contract)
        if evidence.status is not TerminalTaskStatus.COMPLETE:
            raise AnchorBaselineInvalidContextError(
                f"anchor baseline {boundary} task evidence is indeterminate"
            )
        if (
            type(evidence.success) is not bool
            or type(evidence.unsafe) is not bool
            or evidence.progress not in (0.0, 1.0)
        ):
            raise AnchorBaselineInvalidContextError(
                f"anchor baseline {boundary} task evidence is incomplete"
            )
        return evidence

    def _require_restored_task_snapshot(
        self,
        source_state: PickCubeIndexedStateV1,
        evidence: TerminalTaskEvidence,
    ) -> None:
        # The uninterrupted-source snapshot is an immutable event annotation used
        # by the anchor scheduler.  Runtime restoration is instead checked against
        # the independently captured post-restoration projection.  In ManiSkill
        # 3.0.1 the public grasp query is contact-impulse-derived and can therefore
        # legitimately differ between those two explicitly separated boundaries.
        expected = source_state.restored_task_snapshot
        observed = {
            "success": evidence.success,
            "is_obj_placed": _diagnostic_bool(
                evidence, "pickcube_object_placed", boundary="before_candidate"
            ),
            "is_robot_static": _diagnostic_bool(
                evidence, "pickcube_robot_static", boundary="before_candidate"
            ),
            "is_grasped": _diagnostic_bool(
                evidence, "pickcube_grasped", boundary="before_candidate"
            ),
        }
        if observed != {
            "success": expected.success,
            "is_obj_placed": expected.is_obj_placed,
            "is_robot_static": expected.is_robot_static,
            "is_grasped": expected.is_grasped,
        }:
            raise AnchorBaselineInvalidContextError(
                "restored anchor task evidence differs from the archived snapshot"
            )

    def _assert_inputs_unchanged(
        self,
        *,
        binding: _InputBinding,
        archive_content_digest: str,
        source_episode: PickCubeStateIndexedEpisodeV1,
        source_state: PickCubeIndexedStateV1,
        source_remaining_actions: NDArray[Any],
        continuation: SourceContinuationIdentityV1,
        action_control_contract: PickCubeActionControlContractV1,
    ) -> None:
        if (
            archive_content_digest != binding.archive_content_digest
            or source_episode.content_digest != binding.source_episode_content_digest
            or source_episode.source_actions.dtype.str != binding.source_actions_dtype
            or source_episode.source_actions.shape != binding.source_actions_shape
            or source_episode.source_actions.tobytes(order="C")
            != binding.source_actions_bytes
            or source_remaining_actions.dtype.str != binding.source_remaining_dtype
            or source_remaining_actions.shape != binding.source_remaining_shape
            or source_remaining_actions.tobytes(order="C")
            != binding.source_remaining_bytes
            or source_state.content_digest != binding.source_state_content_digest
            or compute_state_tree_digest(source_state.tree)
            != binding.source_state_digest
            or continuation.content_digest != binding.continuation_content_digest
            or action_control_contract.content_digest
            != binding.action_control_contract_digest
        ):
            raise AnchorBaselineIntegrityError(
                "anchor baseline inputs changed during runtime validation"
            )

    @staticmethod
    def _close_environment(
        environment: object, primary_error: BaseException | None
    ) -> None:
        close = getattr(environment, "close", None)
        if not callable(close):
            error = AnchorBaselineRuntimeError(
                "anchor baseline environment lacks public close"
            )
            if primary_error is None:
                raise error
            primary_error.add_note(str(error))
            return
        try:
            close()
        except BaseException as close_error:
            if primary_error is None:
                raise AnchorBaselineRuntimeError(
                    "anchor baseline environment close failed"
                ) from close_error
            primary_error.add_note(
                "anchor baseline environment close also failed: "
                + type(close_error).__name__
            )


def _same_array_bytes(first: object, second: object) -> bool:
    return (
        isinstance(first, np.ndarray)
        and isinstance(second, np.ndarray)
        and first.dtype.str == second.dtype.str
        and first.shape == second.shape
        and first.tobytes(order="C") == second.tobytes(order="C")
    )


def _rank_two_shape(value: NDArray[Any]) -> tuple[int, int]:
    if value.ndim != 2:
        raise AnchorBaselineInvalidContextError(
            "anchor baseline actions must have rank two"
        )
    return int(value.shape[0]), int(value.shape[1])


def _diagnostic_bool(
    evidence: TerminalTaskEvidence, key: str, *, boundary: str
) -> bool:
    value = evidence.diagnostics.get(key)
    if type(value) is not bool:
        raise AnchorBaselineInvalidContextError(
            f"anchor baseline {boundary} evidence lacks {key}"
        )
    return bool(value)


def _required_binary_progress(evidence: TerminalTaskEvidence, boundary: str) -> float:
    progress = evidence.progress
    if progress not in (0.0, 1.0):
        raise AnchorBaselineInvalidContextError(
            f"anchor baseline {boundary} progress is not binary completion"
        )
    return float(progress)


ManiSkillPickCubeAnchorBaselineValidator = PickCubeAnchorBaselineValidator


__all__ = [
    "ANCHOR_BASELINE_PURPOSE",
    "AnchorBaselineFailureError",
    "AnchorBaselineIntegrityError",
    "AnchorBaselineInvalidContextError",
    "AnchorBaselineRuntimeError",
    "AnchorBaselineValidatorError",
    "ManiSkillPickCubeAnchorBaselineValidator",
    "PickCubeAnchorBaselineValidator",
]

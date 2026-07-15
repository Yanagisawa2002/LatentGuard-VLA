"""M2A ProposalEvaluator backed by generic exact-state paired replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.evaluation.base import ApplicabilityDecision
from latentguard.evaluation.models import (
    EVALUATION_EVIDENCE_SCHEMA_VERSION,
    EvaluationEvidence,
    EvaluationStatus,
    compute_configuration_digest,
    compute_evidence_identifier,
)
from latentguard.evaluation.validation import validate_evaluation_evidence
from latentguard.models import FailureEvent, JsonScalar
from latentguard.replay.base import (
    ExactReplayAdapter,
    ReplayInvalidContextError,
    ReplayValidationError,
)
from latentguard.replay.executor import execute_paired_replay
from latentguard.replay.identity import canonical_json_value
from latentguard.replay.models import (
    REPLAY_TRUST_CONTRACT_VERSION,
    PairedReplayResult,
    ReplayBundle,
    ReplayCase,
    ReplayTrustDescriptor,
    ReplayTrustTier,
)
from latentguard.replay.source import (
    ReplaySourceBindingError,
    ReplaySourceMutationError,
)
from latentguard.replay.validation import (
    validate_paired_replay_result,
    validate_replay_bundle,
    validate_replay_case,
    validate_replay_trust_claim,
    validate_replay_trust_descriptor,
)

EXACT_STATE_PAIRED_REPLAY_EVALUATOR_ID = "exact_state_paired_replay"
EXACT_STATE_PAIRED_REPLAY_EVALUATOR_VERSION = "1.0.0"
BASELINE_VALIDITY_SEMANTIC_VERSION = "1.0.0"
PAIRED_REPLAY_SEMANTIC_VERSION = "1.0.0"


class ExactReplayEvaluatorConfigurationError(ValueError):
    """Raised when adapter, bundle, or evaluator identities are inconsistent."""


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ExactStatePairedReplayEvaluator:
    """Resolve proposals and convert paired replay results into M2A evidence."""

    adapter: ExactReplayAdapter
    replay_bundle: ReplayBundle
    _resolved_configuration: Mapping[str, object]
    _configuration_digest: str
    _trust_descriptor: ReplayTrustDescriptor

    @property
    def evaluator_id(self) -> str:
        """Return the fixed M2A registry identity for exact paired replay."""
        return EXACT_STATE_PAIRED_REPLAY_EVALUATOR_ID

    @property
    def evaluator_version(self) -> str:
        """Return the paired replay evaluator semantic version."""
        return EXACT_STATE_PAIRED_REPLAY_EVALUATOR_VERSION

    @property
    def configuration_digest(self) -> str:
        """Return the digest binding adapter, datasets, bundle, and semantics."""
        return self._configuration_digest

    @property
    def trust_descriptor(self) -> ReplayTrustDescriptor:
        """Return the validated adapter trust ceiling used for label mapping."""
        return self._trust_descriptor

    def resolved_configuration(self) -> Mapping[str, object]:
        """Return canonical evaluator configuration without runtime metadata."""
        return self._resolved_configuration

    def check_applicability(
        self,
        proposal: CorruptedActionProposal,
        *,
        source_dataset_id: str,
    ) -> ApplicabilityDecision:
        """Resolve and validate one case without creating an execution session."""
        self._validate_adapter_contract()
        if source_dataset_id != self.replay_bundle.source_dataset_id:
            return ApplicabilityDecision.invalid("source_dataset_id_mismatch")
        try:
            replay_case = self.adapter.resolve_case(proposal)
            self._validate_resolved_case(replay_case, proposal)
            self._validate_adapter_contract()
        except ReplaySourceMutationError:
            raise
        except (ReplayInvalidContextError, ReplaySourceBindingError) as exc:
            return ApplicabilityDecision.invalid(
                f"invalid_replay_context:{type(exc).__name__}"
            )
        return ApplicabilityDecision.applicable()

    def evaluate(
        self,
        proposal: CorruptedActionProposal,
        *,
        source_dataset_id: str,
        evaluation_seed: int,
        attempt_ordinal: int,
    ) -> EvaluationEvidence:
        """Execute an applicable proposal and return validated M2A evidence."""
        if source_dataset_id != self.replay_bundle.source_dataset_id:
            raise ExactReplayEvaluatorConfigurationError(
                "source dataset identity changed after applicability planning"
            )
        self._validate_adapter_contract()
        replay_case = self.adapter.resolve_case(proposal)
        self._validate_resolved_case(replay_case, proposal)
        self._validate_adapter_contract()
        result = execute_paired_replay(self.adapter, proposal, self.replay_bundle)
        post_execution_case = self.adapter.resolve_case(proposal)
        self._validate_resolved_case(post_execution_case, proposal)
        if post_execution_case.case_id != replay_case.case_id:
            raise ExactReplayEvaluatorConfigurationError(
                "adapter changed proposal-to-case binding during replay"
            )
        self._validate_adapter_contract()
        validate_paired_replay_result(result)
        evidence = self._result_to_evidence(
            replay_case,
            result,
            evaluation_seed=evaluation_seed,
            attempt_ordinal=attempt_ordinal,
        )
        validate_evaluation_evidence(evidence)
        return evidence

    def _validate_adapter_contract(self) -> None:
        """Reject identity, configuration, or trust drift around every attempt."""
        for actual, expected, field in (
            (self.adapter.adapter_id, self.replay_bundle.adapter_id, "adapter_id"),
            (
                self.adapter.adapter_version,
                self.replay_bundle.adapter_version,
                "adapter_version",
            ),
            (
                self.adapter.configuration_digest,
                self.replay_bundle.adapter_configuration_digest,
                "configuration_digest",
            ),
        ):
            if actual != expected:
                raise ExactReplayEvaluatorConfigurationError(
                    f"adapter contract changed after construction: {field}"
                )
        configuration = self.adapter.resolved_configuration()
        if not isinstance(configuration, Mapping):
            raise ExactReplayEvaluatorConfigurationError(
                "adapter resolved configuration must remain a mapping"
            )
        try:
            canonical_json_value(
                configuration,
                context="ExactReplayAdapter.resolved_configuration",
                reject_runtime_paths=True,
            )
        except ReplayValidationError as exc:
            raise ExactReplayEvaluatorConfigurationError(str(exc)) from exc
        if (
            compute_configuration_digest(configuration)
            != self.replay_bundle.adapter_configuration_digest
        ):
            raise ExactReplayEvaluatorConfigurationError(
                "adapter resolved configuration changed after construction"
            )
        trust = self.adapter.trust_descriptor()
        validate_replay_trust_descriptor(trust)
        if trust != self._trust_descriptor:
            raise ExactReplayEvaluatorConfigurationError(
                "adapter trust descriptor changed after construction"
            )

    def _validate_resolved_case(
        self, replay_case: object, proposal: CorruptedActionProposal
    ) -> None:
        if not isinstance(replay_case, ReplayCase):
            raise ExactReplayEvaluatorConfigurationError(
                "adapter.resolve_case did not return ReplayCase"
            )
        validate_replay_case(replay_case)
        expected = {
            "proposal_id": proposal.proposal_id,
            "source_dataset_id": self.replay_bundle.source_dataset_id,
            "source_dataset_digest": self.replay_bundle.source_dataset_digest,
            "corruption_dataset_digest": self.replay_bundle.corruption_dataset_digest,
            "adapter_id": self.adapter.adapter_id,
            "adapter_version": self.adapter.adapter_version,
        }
        mismatches = [
            field
            for field, value in expected.items()
            if getattr(replay_case, field) != value
        ]
        bundle_cases = {
            item.proposal_id: item.case_id for item in self.replay_bundle.replay_cases
        }
        if bundle_cases.get(proposal.proposal_id) != replay_case.case_id:
            mismatches.append("case_id")
        if mismatches:
            raise ExactReplayEvaluatorConfigurationError(
                "resolved replay case identity mismatch: " + ", ".join(mismatches)
            )

    def _result_to_evidence(
        self,
        replay_case: ReplayCase,
        result: PairedReplayResult,
        *,
        evaluation_seed: int,
        attempt_ordinal: int,
    ) -> EvaluationEvidence:
        status = result.status
        terminal = result.corrupted_terminal_task
        initial = result.corrupted_initial_task
        success: bool | None = None
        progress_before: float | None = None
        progress_after: float | None = None
        progress_delta: float | None = None
        unsafe: bool | None = None
        failure_events: tuple[FailureEvent, ...] = ()
        label_source = None
        label_strength = None
        simulator_verified = False

        if status in {EvaluationStatus.CONCLUSIVE, EvaluationStatus.INDETERMINATE}:
            if terminal is None:
                raise ExactReplayEvaluatorConfigurationError(
                    "completed replay result is missing corrupted terminal evidence"
                )
            progress_before = None if initial is None else initial.progress
            progress_after = terminal.progress
            if progress_before is not None and progress_after is not None:
                progress_delta = progress_after - progress_before
            success = terminal.success
            unsafe = terminal.unsafe
            failure_events = terminal.failure_events
        if status is EvaluationStatus.CONCLUSIVE:
            label_source = self._trust_descriptor.label_source
            label_strength = self._trust_descriptor.maximum_label_strength
            simulator_verified = bool(
                self._trust_descriptor.trust_tier is ReplayTrustTier.EXACT_SIMULATOR
                and self._trust_descriptor.simulator_verification_allowed
            )
            validate_replay_trust_claim(
                self._trust_descriptor,
                label_source=label_source,
                label_strength=label_strength,
                simulator_replay_verified=simulator_verified,
                replay_result=result,
            )

        baseline_steps = (
            0
            if result.baseline_execution is None
            else result.baseline_execution.executed_step_count
        )
        corrupted_steps = (
            0
            if result.corrupted_execution is None
            else result.corrupted_execution.executed_step_count
        )
        metrics: dict[str, JsonScalar] = {
            "replay_case_id": replay_case.case_id,
            "replay_baseline_requested_steps": (
                0
                if result.baseline_execution is None
                else result.baseline_execution.requested_step_count
            ),
            "replay_baseline_steps": baseline_steps,
            "replay_corrupted_requested_steps": (
                0
                if result.corrupted_execution is None
                else result.corrupted_execution.requested_step_count
            ),
            "replay_corrupted_steps": corrupted_steps,
        }
        for name, restoration in (
            (
                "replay_baseline_restoration_maximum_absolute_error",
                result.baseline_restoration,
            ),
            (
                "replay_corrupted_restoration_maximum_absolute_error",
                result.corrupted_restoration,
            ),
        ):
            if (
                restoration is not None
                and restoration.maximum_absolute_error is not None
            ):
                metrics[name] = restoration.maximum_absolute_error
        source = np.asarray(replay_case.original_action.actions, dtype=np.float64)
        corrupted = np.asarray(replay_case.transformed_action.actions, dtype=np.float64)
        difference = np.abs(source - corrupted)
        metrics["source_corrupted_action_mean_absolute_difference"] = float(
            np.mean(difference)
        )
        metrics["source_corrupted_action_maximum_absolute_difference"] = float(
            np.max(difference)
        )
        if initial is not None:
            initial_distance = initial.diagnostics.get("pickcube_cube_to_goal_distance")
            if initial_distance is not None:
                metrics["initial_cube_to_goal_distance"] = initial_distance
        if terminal is not None:
            for source_name, metric_name in (
                (
                    "pickcube_cube_to_goal_distance",
                    "final_cube_to_goal_distance",
                ),
                ("pickcube_cube_center_z", "final_cube_center_z"),
                ("pickcube_object_placed", "object_placed"),
                ("pickcube_robot_static", "robot_static"),
                ("pickcube_grasped", "grasped"),
            ):
                metric = terminal.diagnostics.get(source_name)
                if metric is not None:
                    metrics[metric_name] = metric
        metrics.update(result.diagnostics)
        notes = (
            "deterministic non-physical replay fixture; infrastructure evidence only"
            if self._trust_descriptor.trust_tier is ReplayTrustTier.FIXTURE
            else "generic exact-state paired replay evidence"
        )
        return EvaluationEvidence(
            evidence_id=compute_evidence_identifier(
                proposal_id=replay_case.proposal_id,
                evaluator_id=self.evaluator_id,
                evaluator_version=self.evaluator_version,
                evaluator_configuration_digest=self.configuration_digest,
                evaluation_seed=evaluation_seed,
                attempt_ordinal=attempt_ordinal,
            ),
            proposal_id=replay_case.proposal_id,
            source_dataset_id=replay_case.source_dataset_id,
            source_episode_id=replay_case.source_episode_id,
            source_candidate_id=replay_case.source_candidate_id,
            split_group_id=replay_case.split_group_id,
            evaluator_id=self.evaluator_id,
            evaluator_version=self.evaluator_version,
            evaluator_configuration_digest=self.configuration_digest,
            evaluation_seed=evaluation_seed,
            attempt_ordinal=attempt_ordinal,
            status=status,
            success=success,
            progress_before=progress_before,
            progress_after=progress_after,
            progress_delta=progress_delta,
            unsafe=unsafe,
            failure_events=failure_events,
            termination_reason=result.termination_reason,
            replayed_control_steps=result.replayed_control_steps,
            metrics=metrics,
            artifact_references=(),
            label_source=label_source,
            label_strength=label_strength,
            simulator_replay_verified=simulator_verified,
            notes=notes,
            schema_version=EVALUATION_EVIDENCE_SCHEMA_VERSION,
        )


def create_exact_state_paired_replay_evaluator(
    adapter: ExactReplayAdapter,
    replay_bundle: ReplayBundle,
) -> ExactStatePairedReplayEvaluator:
    """Validate all adapter/bundle identities and create the M2A evaluator."""
    if not isinstance(adapter, ExactReplayAdapter):
        raise ExactReplayEvaluatorConfigurationError(
            "adapter does not implement ExactReplayAdapter"
        )
    validate_replay_bundle(replay_bundle)
    trust = adapter.trust_descriptor()
    validate_replay_trust_descriptor(trust)
    adapter_configuration = adapter.resolved_configuration()
    if not isinstance(adapter_configuration, Mapping):
        raise ExactReplayEvaluatorConfigurationError(
            "adapter resolved configuration must be a mapping"
        )
    try:
        canonical_json_value(
            adapter_configuration,
            context="ExactReplayAdapter.resolved_configuration",
            reject_runtime_paths=True,
        )
    except ReplayValidationError as exc:
        raise ExactReplayEvaluatorConfigurationError(str(exc)) from exc
    expected_adapter_digest = compute_configuration_digest(adapter_configuration)
    if adapter.configuration_digest != expected_adapter_digest:
        raise ExactReplayEvaluatorConfigurationError(
            "adapter configuration digest does not match resolved configuration"
        )
    for actual, expected, field in (
        (replay_bundle.adapter_id, adapter.adapter_id, "adapter_id"),
        (replay_bundle.adapter_version, adapter.adapter_version, "adapter_version"),
        (
            replay_bundle.adapter_configuration_digest,
            adapter.configuration_digest,
            "adapter_configuration_digest",
        ),
    ):
        if actual != expected:
            raise ExactReplayEvaluatorConfigurationError(
                f"replay bundle {field} does not match adapter"
            )
    if replay_bundle.replay_cases:
        progress_semantics = {
            replay_case.progress_semantic for replay_case in replay_bundle.replay_cases
        }
        unsafe_semantics = {
            replay_case.unsafe_semantic for replay_case in replay_bundle.replay_cases
        }
        if len(progress_semantics) != 1 or len(unsafe_semantics) != 1:
            raise ExactReplayEvaluatorConfigurationError(
                "replay bundle must use one progress and unsafe semantic"
            )
        progress_semantic = next(iter(progress_semantics))
        unsafe_semantic = next(iter(unsafe_semantics))
    else:
        progress_semantic = "no_selected_replay_cases"
        unsafe_semantic = "no_selected_replay_cases"
    resolved_value = _freeze_json(
        {
            "adapter_configuration": adapter_configuration,
            "adapter_configuration_digest": adapter.configuration_digest,
            "adapter_id": adapter.adapter_id,
            "adapter_version": adapter.adapter_version,
            "baseline_validity_semantic_version": (BASELINE_VALIDITY_SEMANTIC_VERSION),
            "corruption_dataset_digest": replay_bundle.corruption_dataset_digest,
            "exact_state_verification_required": (
                trust.exact_state_verification_required
            ),
            "label_source": trust.label_source.value,
            "maximum_label_strength": trust.maximum_label_strength.value,
            "paired_replay_semantic_version": PAIRED_REPLAY_SEMANTIC_VERSION,
            "progress_semantic": progress_semantic,
            "replay_bundle_digest": replay_bundle.bundle_digest,
            "simulator_verification_allowed": trust.simulator_verification_allowed,
            "source_dataset_digest": replay_bundle.source_dataset_digest,
            "source_dataset_id": replay_bundle.source_dataset_id,
            "trust_contract_version": REPLAY_TRUST_CONTRACT_VERSION,
            "trust_tier": trust.trust_tier.value,
            "unsafe_semantic": unsafe_semantic,
        }
    )
    if not isinstance(resolved_value, Mapping):
        raise ExactReplayEvaluatorConfigurationError(
            "resolved evaluator configuration did not remain a mapping"
        )
    resolved = resolved_value
    return ExactStatePairedReplayEvaluator(
        adapter=adapter,
        replay_bundle=replay_bundle,
        _resolved_configuration=resolved,
        _configuration_digest=compute_configuration_digest(resolved),
        _trust_descriptor=trust,
    )


__all__ = [
    "BASELINE_VALIDITY_SEMANTIC_VERSION",
    "EXACT_STATE_PAIRED_REPLAY_EVALUATOR_ID",
    "EXACT_STATE_PAIRED_REPLAY_EVALUATOR_VERSION",
    "PAIRED_REPLAY_SEMANTIC_VERSION",
    "ExactReplayEvaluatorConfigurationError",
    "ExactStatePairedReplayEvaluator",
    "create_exact_state_paired_replay_evaluator",
]

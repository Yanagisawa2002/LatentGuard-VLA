"""Strict, non-repairing validation for generic exact replay contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import NoReturn, cast

from latentguard.evaluation.models import EvaluationStatus
from latentguard.models import FailureEvent, LabelSource, LabelStrength
from latentguard.replay.base import ReplayValidationError
from latentguard.replay.identity import (
    REPLAY_BUNDLE_DIGEST_PREFIX,
    REPLAY_CASE_ID_PREFIX,
    canonical_json_value,
    recompute_replay_bundle_digest,
    recompute_replay_case_identifier,
)
from latentguard.replay.models import (
    REPLAY_SCHEMA_VERSION,
    REPLAY_TRUST_CONTRACT_VERSION,
    ActionExecutionEvidence,
    PairedReplayResult,
    ReplayBundle,
    ReplayCase,
    ReplayStateReference,
    ReplayTaskReference,
    ReplayTrustDescriptor,
    ReplayTrustTier,
    StateComparisonSemantic,
    StateMatchKind,
    StateRestorationEvidence,
    TerminalTaskEvidence,
    TerminalTaskStatus,
)
from latentguard.validation import (
    DataValidationError,
    validate_action_chunk,
    validate_failure_event,
)

_SEMANTIC_VERSION_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONFIGURATION_DIGEST_PATTERN = re.compile(r"^cfg-sha256-[0-9a-f]{64}$")
_CASE_ID_PATTERN = re.compile(rf"^{re.escape(REPLAY_CASE_ID_PREFIX)}[0-9a-f]{{64}}$")
_BUNDLE_DIGEST_PATTERN = re.compile(
    rf"^{re.escape(REPLAY_BUNDLE_DIGEST_PREFIX)}[0-9a-f]{{64}}$"
)


def _fail(context: str, field: str, reason: str) -> NoReturn:
    raise ReplayValidationError(f"{context}.{field}: {reason}")


def _required_text(value: object, context: str, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(
            context, field, "must be a non-empty string without surrounding whitespace"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        _fail(context, field, "control characters are unsupported")
    canonical_json_value(value, context=f"{context}.{field}")
    return value


def _optional_text(value: object, context: str, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, context, field)


def _semantic_version(value: object, context: str, field: str) -> str:
    version = _required_text(value, context, field)
    if _SEMANTIC_VERSION_PATTERN.fullmatch(version) is None:
        _fail(context, field, "must be a semantic version such as 1.0.0")
    return version


def _schema(value: object, context: str) -> None:
    if value != REPLAY_SCHEMA_VERSION:
        _fail(
            context,
            "schema_version",
            f"unsupported version {value!r}; supported: {REPLAY_SCHEMA_VERSION}",
        )


def _digest(value: object, context: str, field: str) -> str:
    digest = _required_text(value, context, field)
    if _SHA256_DIGEST_PATTERN.fullmatch(digest) is None:
        _fail(context, field, "expected sha256:<64 lowercase hexadecimal characters>")
    return digest


def _configuration_digest(value: object, context: str, field: str) -> str:
    digest = _required_text(value, context, field)
    if _CONFIGURATION_DIGEST_PATTERN.fullmatch(digest) is None:
        _fail(context, field, "expected cfg-sha256-<64 lowercase hex characters>")
    return digest


def _finite_number(
    value: object,
    context: str,
    field: str,
    *,
    nonnegative: bool = False,
) -> float:
    if type(value) not in (int, float):
        _fail(context, field, "must be a finite number (booleans are not numbers)")
    try:
        number = float(cast(int | float, value))
    except (OverflowError, ValueError) as exc:
        raise ReplayValidationError(
            f"{context}.{field}: number is out of range"
        ) from exc
    if not math.isfinite(number):
        _fail(context, field, "must be finite")
    if nonnegative and number < 0.0:
        _fail(context, field, "must be non-negative")
    return number


def _diagnostics(value: object, context: str) -> None:
    if not isinstance(value, Mapping):
        _fail(context, "diagnostics", "must be a mapping")
    for key, item in value.items():
        _required_text(key, f"{context}.diagnostics", "key")
        if item is None or type(item) in (str, int, bool):
            if isinstance(item, str):
                canonical_json_value(item, context=f"{context}.diagnostics.{key}")
            continue
        if type(item) is float and math.isfinite(item):
            continue
        _fail(
            context,
            "diagnostics",
            f"{key!r} must be a finite JSON scalar",
        )


def _canonical_metadata(value: object, context: str) -> None:
    if not isinstance(value, Mapping):
        _fail(context, "metadata", "must be a canonical JSON mapping")
    canonical_json_value(value, context=f"{context}.metadata")


def validate_replay_state_reference(reference: ReplayStateReference) -> None:
    """Validate an opaque state reference without interpreting simulator state."""
    context = "ReplayStateReference"
    if not isinstance(reference, ReplayStateReference):
        _fail(context, "value", "expected ReplayStateReference")
    _schema(reference.schema_version, context)
    _required_text(reference.adapter_id, context, "adapter_id")
    _semantic_version(reference.adapter_version, context, "adapter_version")
    _required_text(reference.source_reference_id, context, "source_reference_id")
    _digest(reference.expected_state_digest, context, "expected_state_digest")
    if not isinstance(reference.comparison_semantic, StateComparisonSemantic):
        _fail(context, "comparison_semantic", "unsupported comparison semantic")
    populated = int(reference.state_key is not None) + int(
        reference.state_index is not None
    )
    if populated != 1:
        _fail(context, "state_key/state_index", "exactly one selector must be present")
    if reference.state_key is not None:
        _required_text(reference.state_key, context, "state_key")
    if reference.state_index is not None and (
        type(reference.state_index) is not int or reference.state_index < 0
    ):
        _fail(context, "state_index", "must be a non-negative integer")
    _canonical_metadata(reference.metadata, context)


def validate_replay_task_reference(reference: ReplayTaskReference) -> None:
    """Validate a stable task reference while leaving its metadata uninterpreted."""
    context = "ReplayTaskReference"
    if not isinstance(reference, ReplayTaskReference):
        _fail(context, "value", "expected ReplayTaskReference")
    _schema(reference.schema_version, context)
    _required_text(reference.task_id, context, "task_id")
    _semantic_version(reference.task_contract_version, context, "task_contract_version")
    _canonical_metadata(reference.metadata, context)


def validate_replay_case(replay_case: ReplayCase) -> None:
    """Validate case identity, action contracts, references, and content binding."""
    context = "ReplayCase"
    if not isinstance(replay_case, ReplayCase):
        _fail(context, "value", "expected ReplayCase")
    _schema(replay_case.schema_version, context)
    for field in (
        "proposal_id",
        "source_dataset_id",
        "source_episode_id",
        "source_candidate_id",
        "split_group_id",
        "adapter_id",
        "progress_semantic",
        "unsafe_semantic",
    ):
        _required_text(getattr(replay_case, field), context, field)
    _semantic_version(replay_case.adapter_version, context, "adapter_version")
    _digest(replay_case.source_dataset_digest, context, "source_dataset_digest")
    _digest(replay_case.corruption_dataset_digest, context, "corruption_dataset_digest")
    if not isinstance(replay_case.state_reference, ReplayStateReference):
        _fail(context, "state_reference", "expected ReplayStateReference")
    if not isinstance(replay_case.task_reference, ReplayTaskReference):
        _fail(context, "task_reference", "expected ReplayTaskReference")
    validate_replay_state_reference(replay_case.state_reference)
    validate_replay_task_reference(replay_case.task_reference)
    if replay_case.state_reference.adapter_id != replay_case.adapter_id:
        _fail(context, "state_reference.adapter_id", "must match case adapter_id")
    if replay_case.state_reference.adapter_version != replay_case.adapter_version:
        _fail(
            context,
            "state_reference.adapter_version",
            "must match case adapter_version",
        )
    try:
        validate_action_chunk(
            replay_case.original_action, replay_case.source_candidate_id
        )
        validate_action_chunk(replay_case.transformed_action, replay_case.proposal_id)
    except DataValidationError as exc:
        raise ReplayValidationError(f"{context}.action: {exc}") from exc
    original = replay_case.original_action
    transformed = replay_case.transformed_action
    if original.actions.shape != transformed.actions.shape:
        _fail(
            context, "transformed_action.actions", "shape and horizon must match source"
        )
    if original.actions.dtype != transformed.actions.dtype:
        _fail(context, "transformed_action.actions", "dtype must match source action")
    if original.coordinate_frame != transformed.coordinate_frame:
        _fail(
            context, "transformed_action.coordinate_frame", "must match source action"
        )
    if original.control_period_s != transformed.control_period_s:
        _fail(
            context, "transformed_action.control_period_s", "must match source action"
        )
    if _CASE_ID_PATTERN.fullmatch(replay_case.case_id) is None:
        _fail(context, "case_id", f"expected {REPLAY_CASE_ID_PREFIX}<64 lowercase hex>")
    expected = recompute_replay_case_identifier(replay_case)
    if replay_case.case_id != expected:
        _fail(
            context,
            "case_id",
            f"deterministic identifier mismatch; expected {expected!r}",
        )


def validate_state_restoration_evidence(evidence: StateRestorationEvidence) -> None:
    """Validate adapter-produced restoration evidence and comparison semantics."""
    context = "StateRestorationEvidence"
    if not isinstance(evidence, StateRestorationEvidence):
        _fail(context, "value", "expected StateRestorationEvidence")
    _schema(evidence.schema_version, context)
    expected = _digest(evidence.expected_state_digest, context, "expected_state_digest")
    observed = _digest(evidence.observed_state_digest, context, "observed_state_digest")
    if not isinstance(evidence.comparison_semantic, StateComparisonSemantic):
        _fail(context, "comparison_semantic", "unsupported comparison semantic")
    tolerance = _finite_number(
        evidence.comparison_tolerance,
        context,
        "comparison_tolerance",
        nonnegative=True,
    )
    if (
        type(evidence.compared_component_count) is not int
        or evidence.compared_component_count <= 0
    ):
        _fail(context, "compared_component_count", "must be a positive integer")
    maximum_error = None
    if evidence.maximum_absolute_error is not None:
        maximum_error = _finite_number(
            evidence.maximum_absolute_error,
            context,
            "maximum_absolute_error",
            nonnegative=True,
        )
    if not isinstance(evidence.match_kind, StateMatchKind):
        _fail(context, "match_kind", "unsupported state match kind")
    if type(evidence.restoration_verified) is not bool:
        _fail(context, "restoration_verified", "must be a boolean")
    if evidence.comparison_semantic is StateComparisonSemantic.EXACT_DIGEST:
        if tolerance != 0.0:
            _fail(
                context, "comparison_tolerance", "exact digest comparison requires zero"
            )
        if evidence.match_kind is StateMatchKind.WITHIN_TOLERANCE:
            _fail(
                context,
                "match_kind",
                "exact digest comparison cannot be tolerance-based",
            )
        exact = expected == observed
        if exact != (evidence.match_kind is StateMatchKind.EXACT):
            _fail(
                context,
                "match_kind",
                "must reflect equality of expected and observed digest",
            )
        if evidence.match_kind is StateMatchKind.EXACT and maximum_error not in (
            None,
            0.0,
        ):
            _fail(
                context,
                "maximum_absolute_error",
                "an exact match permits only null or zero error",
            )
    else:
        if maximum_error is None:
            _fail(
                context, "maximum_absolute_error", "numeric comparison requires a value"
            )
        if evidence.match_kind is StateMatchKind.EXACT:
            if expected != observed or maximum_error != 0.0:
                _fail(
                    context,
                    "match_kind",
                    "exact match requires equal digests and zero error",
                )
        elif evidence.match_kind is StateMatchKind.WITHIN_TOLERANCE:
            if maximum_error > tolerance:
                _fail(context, "maximum_absolute_error", "exceeds comparison tolerance")
        elif maximum_error <= tolerance:
            _fail(context, "maximum_absolute_error", "mismatch must exceed tolerance")
    matched = evidence.match_kind is not StateMatchKind.MISMATCH
    if evidence.restoration_verified != matched:
        _fail(
            context, "restoration_verified", "must be true exactly for a verified match"
        )
    _diagnostics(evidence.diagnostics, context)


def validate_action_execution_evidence(evidence: ActionExecutionEvidence) -> None:
    """Validate action step counts without converting partial execution to outcome."""
    context = "ActionExecutionEvidence"
    if not isinstance(evidence, ActionExecutionEvidence):
        _fail(context, "value", "expected ActionExecutionEvidence")
    _schema(evidence.schema_version, context)
    if (
        type(evidence.requested_step_count) is not int
        or evidence.requested_step_count <= 0
    ):
        _fail(context, "requested_step_count", "must be a positive integer")
    if (
        type(evidence.executed_step_count) is not int
        or evidence.executed_step_count < 0
        or evidence.executed_step_count > evidence.requested_step_count
    ):
        _fail(
            context, "executed_step_count", "must be between zero and requested count"
        )
    if type(evidence.complete) is not bool:
        _fail(context, "complete", "must be a boolean")
    if evidence.complete != (
        evidence.executed_step_count == evidence.requested_step_count
    ):
        _fail(context, "complete", "must exactly reflect full requested-step execution")
    reason = _optional_text(evidence.termination_reason, context, "termination_reason")
    if not evidence.complete and reason is None:
        _fail(
            context,
            "termination_reason",
            "partial execution requires an explicit reason",
        )
    _diagnostics(evidence.diagnostics, context)


def validate_terminal_task_evidence(evidence: TerminalTaskEvidence) -> None:
    """Validate complete/indeterminate task evidence without filling missing values."""
    context = "TerminalTaskEvidence"
    if not isinstance(evidence, TerminalTaskEvidence):
        _fail(context, "value", "expected TerminalTaskEvidence")
    _schema(evidence.schema_version, context)
    if not isinstance(evidence.status, TerminalTaskStatus):
        _fail(context, "status", "unsupported terminal task status")
    if evidence.success is not None and type(evidence.success) is not bool:
        _fail(context, "success", "must be a boolean or null")
    if evidence.unsafe is not None and type(evidence.unsafe) is not bool:
        _fail(context, "unsafe", "must be a boolean or null")
    if evidence.progress is not None:
        progress = _finite_number(evidence.progress, context, "progress")
        if not 0.0 <= progress <= 1.0:
            _fail(context, "progress", "must lie in [0, 1]")
    if not isinstance(evidence.failure_events, tuple):
        _fail(context, "failure_events", "must be an immutable tuple")
    for index, event in enumerate(evidence.failure_events):
        if not isinstance(event, FailureEvent):
            _fail(context, "failure_events", f"index {index} is not a FailureEvent")
        try:
            validate_failure_event(event, f"replay-terminal-{index}")
        except DataValidationError as exc:
            raise ReplayValidationError(
                f"{context}.failure_events[{index}]: {exc}"
            ) from exc
    reason = _optional_text(evidence.termination_reason, context, "termination_reason")
    projection_values = (evidence.success, evidence.progress, evidence.unsafe)
    if evidence.status is TerminalTaskStatus.COMPLETE:
        if not all(value is not None for value in projection_values):
            _fail(
                context,
                "status",
                "complete evidence requires success, progress, and unsafe",
            )
    elif all(value is not None for value in projection_values):
        _fail(
            context,
            "status",
            "indeterminate evidence must leave at least one task value missing",
        )
    elif reason is None:
        _fail(
            context,
            "termination_reason",
            "indeterminate evidence requires an explicit reason",
        )
    _diagnostics(evidence.diagnostics, context)


def validate_replay_trust_descriptor(descriptor: ReplayTrustDescriptor) -> None:
    """Validate adapter trust ceilings and prohibit non-simulator simulator claims."""
    context = "ReplayTrustDescriptor"
    if not isinstance(descriptor, ReplayTrustDescriptor):
        _fail(context, "value", "expected ReplayTrustDescriptor")
    _schema(descriptor.schema_version, context)
    if descriptor.trust_contract_version != REPLAY_TRUST_CONTRACT_VERSION:
        _fail(
            context,
            "trust_contract_version",
            "unsupported version "
            f"{descriptor.trust_contract_version!r}; supported: "
            f"{REPLAY_TRUST_CONTRACT_VERSION}",
        )
    if not isinstance(descriptor.trust_tier, ReplayTrustTier):
        _fail(context, "trust_tier", "unsupported replay trust tier")
    if not isinstance(descriptor.label_source, LabelSource):
        _fail(context, "label_source", "unsupported label source")
    if not isinstance(descriptor.maximum_label_strength, LabelStrength):
        _fail(context, "maximum_label_strength", "unsupported label strength")
    if type(descriptor.simulator_verification_allowed) is not bool:
        _fail(context, "simulator_verification_allowed", "must be a boolean")
    if type(descriptor.exact_state_verification_required) is not bool:
        _fail(context, "exact_state_verification_required", "must be a boolean")
    if not descriptor.exact_state_verification_required:
        _fail(
            context,
            "exact_state_verification_required",
            "paired exact replay requires verification",
        )
    if descriptor.trust_tier in {
        ReplayTrustTier.FIXTURE,
        ReplayTrustTier.DETERMINISTIC_NON_SIMULATOR,
    }:
        if descriptor.label_source is not LabelSource.DETERMINISTIC_EVALUATOR:
            _fail(
                context,
                "label_source",
                "non-simulator tiers require deterministic evaluator source",
            )
        if descriptor.maximum_label_strength is not LabelStrength.WEAK:
            _fail(
                context,
                "maximum_label_strength",
                "non-simulator tiers are limited to weak evidence",
            )
        if descriptor.simulator_verification_allowed:
            _fail(
                context,
                "simulator_verification_allowed",
                "non-simulator tiers cannot allow verification",
            )
    else:
        if descriptor.label_source is not LabelSource.SIMULATOR:
            _fail(
                context,
                "label_source",
                "exact simulator tier requires simulator source",
            )
        if descriptor.maximum_label_strength is not LabelStrength.STRONG:
            _fail(
                context,
                "maximum_label_strength",
                "exact simulator tier must declare strong ceiling",
            )
        if not descriptor.simulator_verification_allowed:
            _fail(
                context,
                "simulator_verification_allowed",
                "exact simulator tier must explicitly allow verification",
            )


def _validate_phase_values(result: PairedReplayResult) -> None:
    for field, validator in (
        ("baseline_restoration", validate_state_restoration_evidence),
        ("baseline_execution", validate_action_execution_evidence),
        ("baseline_initial_task", validate_terminal_task_evidence),
        ("baseline_terminal_task", validate_terminal_task_evidence),
        ("corrupted_restoration", validate_state_restoration_evidence),
        ("corrupted_execution", validate_action_execution_evidence),
        ("corrupted_initial_task", validate_terminal_task_evidence),
        ("corrupted_terminal_task", validate_terminal_task_evidence),
    ):
        value = getattr(result, field)
        if value is not None:
            validator(value)


def _baseline_valid(result: PairedReplayResult) -> bool:
    restoration = result.baseline_restoration
    execution = result.baseline_execution
    terminal = result.baseline_terminal_task
    return bool(
        restoration is not None
        and restoration.restoration_verified
        and execution is not None
        and execution.complete
        and terminal is not None
        and terminal.status is TerminalTaskStatus.COMPLETE
        and terminal.success is True
    )


def validate_paired_replay_result(result: PairedReplayResult) -> None:
    """Validate phase ordering and exact status semantics for one paired replay."""
    context = "PairedReplayResult"
    if not isinstance(result, PairedReplayResult):
        _fail(context, "value", "expected PairedReplayResult")
    _schema(result.schema_version, context)
    _required_text(result.replay_case_id, context, "replay_case_id")
    _required_text(result.proposal_id, context, "proposal_id")
    _required_text(result.termination_reason, context, "termination_reason")
    if result.status not in {
        EvaluationStatus.CONCLUSIVE,
        EvaluationStatus.INDETERMINATE,
        EvaluationStatus.INVALID,
        EvaluationStatus.EXECUTION_ERROR,
    }:
        _fail(context, "status", "paired replay cannot use skipped or unknown status")
    if (
        type(result.replayed_control_steps) is not int
        or result.replayed_control_steps < 0
    ):
        _fail(context, "replayed_control_steps", "must be a non-negative integer")
    _validate_phase_values(result)
    executed = sum(
        evidence.executed_step_count
        for evidence in (result.baseline_execution, result.corrupted_execution)
        if evidence is not None
    )
    if result.replayed_control_steps != executed:
        _fail(
            context,
            "replayed_control_steps",
            f"must equal recorded executed steps ({executed})",
        )
    if result.baseline_restoration and result.corrupted_restoration:
        if (
            result.baseline_restoration.expected_state_digest
            != result.corrupted_restoration.expected_state_digest
        ):
            _fail(
                context,
                "corrupted_restoration",
                "both sessions must bind the same expected state digest",
            )
        if (
            result.baseline_restoration.comparison_semantic
            is not result.corrupted_restoration.comparison_semantic
            or result.baseline_restoration.comparison_tolerance
            != result.corrupted_restoration.comparison_tolerance
        ):
            _fail(
                context,
                "corrupted_restoration",
                "both sessions must use the same state comparison contract",
            )
    baseline_valid = _baseline_valid(result)
    if result.status in {EvaluationStatus.CONCLUSIVE, EvaluationStatus.INDETERMINATE}:
        if not baseline_valid:
            _fail(
                context,
                "status",
                "conclusive/indeterminate corruption evidence requires a valid "
                "baseline",
            )
        if (
            result.baseline_initial_task is None
            or result.corrupted_initial_task is None
        ):
            _fail(
                context,
                "status",
                "both sessions must evaluate their initial task state",
            )
        if (
            result.corrupted_restoration is None
            or not result.corrupted_restoration.restoration_verified
        ):
            _fail(context, "corrupted_restoration", "requires verified restoration")
        if (
            result.corrupted_execution is None
            or not result.corrupted_execution.complete
        ):
            _fail(
                context, "corrupted_execution", "requires complete corrupted execution"
            )
        terminal = result.corrupted_terminal_task
        if terminal is None:
            _fail(
                context, "corrupted_terminal_task", "terminal task evidence is required"
            )
        expected_status = (
            TerminalTaskStatus.COMPLETE
            if result.status is EvaluationStatus.CONCLUSIVE
            else TerminalTaskStatus.INDETERMINATE
        )
        if terminal.status is not expected_status:
            _fail(
                context,
                "corrupted_terminal_task.status",
                f"must be {expected_status.value!r}",
            )
    if result.status is EvaluationStatus.INVALID and baseline_valid:
        corrupted_mismatch = bool(
            result.corrupted_restoration is not None
            and not result.corrupted_restoration.restoration_verified
        )
        static_invalid = result.corrupted_restoration is None
        if not (corrupted_mismatch or static_invalid):
            _fail(context, "status", "invalid result lacks an invalid context")
    # A session close can fail after complete task evidence was collected. That is
    # still an infrastructure execution error, so complete phase records are legal
    # for this status and must not be rewritten as a task outcome.
    _diagnostics(result.diagnostics, context)


def validate_replay_trust_claim(
    descriptor: ReplayTrustDescriptor,
    *,
    label_source: LabelSource,
    label_strength: LabelStrength,
    simulator_replay_verified: bool,
    replay_result: PairedReplayResult | None = None,
) -> None:
    """Validate one proposed label claim against adapter trust and replay gates."""
    validate_replay_trust_descriptor(descriptor)
    context = "ReplayTrustClaim"
    if not isinstance(label_source, LabelSource):
        _fail(context, "label_source", "unsupported label source")
    if not isinstance(label_strength, LabelStrength):
        _fail(context, "label_strength", "unsupported label strength")
    if type(simulator_replay_verified) is not bool:
        _fail(context, "simulator_replay_verified", "must be a boolean")
    if label_source is not descriptor.label_source:
        _fail(
            context, "label_source", "exceeds or contradicts adapter trust descriptor"
        )
    if (
        descriptor.maximum_label_strength is LabelStrength.WEAK
        and label_strength is LabelStrength.STRONG
    ):
        _fail(context, "label_strength", "exceeds adapter trust ceiling")
    if simulator_replay_verified and not descriptor.simulator_verification_allowed:
        _fail(
            context,
            "simulator_replay_verified",
            "adapter trust tier forbids verification",
        )
    if descriptor.trust_tier is not ReplayTrustTier.EXACT_SIMULATOR:
        if label_strength is not LabelStrength.WEAK or simulator_replay_verified:
            _fail(
                context,
                "label_strength",
                "non-simulator evidence must be weak and unverified",
            )
        return
    if label_strength is LabelStrength.STRONG and not simulator_replay_verified:
        _fail(
            context,
            "simulator_replay_verified",
            "strong simulator evidence requires verified replay",
        )
    if simulator_replay_verified:
        if replay_result is None:
            _fail(
                context,
                "replay_result",
                "verified simulator claims require a paired result",
            )
        validate_paired_replay_result(replay_result)
        if replay_result.status is not EvaluationStatus.CONCLUSIVE:
            _fail(
                context,
                "replay_result.status",
                "verified simulator claims require conclusive evidence",
            )
        if not _baseline_valid(replay_result):
            _fail(
                context,
                "replay_result",
                "verified simulator claims require baseline success",
            )
        exact_restorations = (
            replay_result.baseline_restoration,
            replay_result.corrupted_restoration,
        )
        if any(
            restoration is None
            or restoration.match_kind is not StateMatchKind.EXACT
            or restoration.expected_state_digest != restoration.observed_state_digest
            or restoration.maximum_absolute_error not in (None, 0.0)
            for restoration in exact_restorations
        ):
            _fail(
                context,
                "replay_result",
                "verified simulator claims require two exact state restorations; "
                "tolerance matches are not verification",
            )
        if (
            replay_result.corrupted_restoration is None
            or not replay_result.corrupted_restoration.restoration_verified
            or replay_result.corrupted_execution is None
            or not replay_result.corrupted_execution.complete
            or replay_result.corrupted_terminal_task is None
            or replay_result.corrupted_terminal_task.status
            is not TerminalTaskStatus.COMPLETE
        ):
            _fail(
                context,
                "replay_result",
                "verified simulator claims require both restorations and completed "
                "corrupted execution",
            )


def validate_replay_bundle(bundle: ReplayBundle) -> None:
    """Validate ordered cases, content bindings, metadata, and bundle digest."""
    context = "ReplayBundle"
    if not isinstance(bundle, ReplayBundle):
        _fail(context, "value", "expected ReplayBundle")
    _schema(bundle.schema_version, context)
    _required_text(bundle.source_dataset_id, context, "source_dataset_id")
    _digest(bundle.source_dataset_digest, context, "source_dataset_digest")
    _digest(bundle.corruption_dataset_digest, context, "corruption_dataset_digest")
    _required_text(bundle.adapter_id, context, "adapter_id")
    _semantic_version(bundle.adapter_version, context, "adapter_version")
    _configuration_digest(
        bundle.adapter_configuration_digest,
        context,
        "adapter_configuration_digest",
    )
    if not isinstance(bundle.replay_cases, tuple):
        _fail(context, "replay_cases", "must be an immutable tuple")
    case_ids: set[str] = set()
    proposal_ids: set[str] = set()
    for index, replay_case in enumerate(bundle.replay_cases):
        if not isinstance(replay_case, ReplayCase):
            _fail(context, "replay_cases", f"index {index} is not a ReplayCase")
        validate_replay_case(replay_case)
        for field in (
            "source_dataset_id",
            "source_dataset_digest",
            "corruption_dataset_digest",
            "adapter_id",
            "adapter_version",
        ):
            if getattr(replay_case, field) != getattr(bundle, field):
                _fail(context, f"replay_cases[{index}].{field}", "must match bundle")
        if replay_case.case_id in case_ids:
            _fail(context, "replay_cases", f"duplicate case ID {replay_case.case_id!r}")
        if replay_case.proposal_id in proposal_ids:
            _fail(
                context,
                "replay_cases",
                f"duplicate proposal ID {replay_case.proposal_id!r}",
            )
        case_ids.add(replay_case.case_id)
        proposal_ids.add(replay_case.proposal_id)
    _canonical_metadata(bundle.metadata, context)
    if _BUNDLE_DIGEST_PATTERN.fullmatch(bundle.bundle_digest) is None:
        _fail(
            context,
            "bundle_digest",
            f"expected {REPLAY_BUNDLE_DIGEST_PREFIX}<64 lowercase hex>",
        )
    expected = recompute_replay_bundle_digest(bundle)
    if bundle.bundle_digest != expected:
        _fail(
            context, "bundle_digest", f"content digest mismatch; expected {expected!r}"
        )


__all__ = [
    "validate_action_execution_evidence",
    "validate_paired_replay_result",
    "validate_replay_bundle",
    "validate_replay_case",
    "validate_replay_state_reference",
    "validate_replay_task_reference",
    "validate_replay_trust_claim",
    "validate_replay_trust_descriptor",
    "validate_state_restoration_evidence",
    "validate_terminal_task_evidence",
]

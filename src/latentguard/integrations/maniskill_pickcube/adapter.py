"""Trusted ManiSkill PickCube adapter for the generic M2B replay protocols."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np

from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.evaluation.models import compute_configuration_digest
from latentguard.models import ActionChunk, LabelSource, LabelStrength
from latentguard.replay.base import (
    ExactReplayAdapter,
    ReplayCaseProvider,
    ReplayEnvironmentSession,
    ReplayExecutionError,
    ReplayInvalidContextError,
    ReplayValidationError,
)
from latentguard.replay.identity import canonical_json_value
from latentguard.replay.models import (
    ReplayBundle,
    ReplayCase,
    ReplayExecutionRole,
    ReplayStateReference,
    ReplayTrustDescriptor,
    ReplayTrustTier,
    StateComparisonSemantic,
)
from latentguard.replay.validation import (
    validate_replay_bundle,
    validate_replay_case,
)

from .session import (
    PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE,
    PICKCUBE_STATE_VERIFICATION_SEMANTIC,
    ManiSkillPickCubeEnvironmentSettings,
    PickCubeReplayActionContract,
)
from .task_evidence import (
    PICKCUBE_PROGRESS_SEMANTIC,
    PICKCUBE_TASK_CONTRACT_VERSION,
    PICKCUBE_TASK_ID,
    PICKCUBE_UNSAFE_SEMANTIC,
    PickCubeTaskKeyContract,
)

MANISKILL_PICKCUBE_ADAPTER_ID = "maniskill_pickcube_v1"
MANISKILL_PICKCUBE_ADAPTER_VERSION = "1.1.0"
MANISKILL_REQUIRED_VERSION = "3.0.1"
MANISKILL_STATE_SEMANTIC_VERSION = "maniskill_state_tree_v1"


class ManiSkillPickCubeAdapterConfigurationError(ValueError):
    """Raised when trusted adapter identities are incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class ManiSkillPickCubeSemanticIdentity:
    """Path-independent identities bound into the real adapter configuration."""

    compatibility_identity: str
    solver_identity: str
    task_implementation_identity: str
    controller_configuration_identity: str
    action_layout_digest: str
    mani_skill_version: str = MANISKILL_REQUIRED_VERSION
    state_semantic_version: str = MANISKILL_STATE_SEMANTIC_VERSION
    state_verification_semantic: str = PICKCUBE_STATE_VERIFICATION_SEMANTIC
    state_verification_maximum_absolute_tolerance: float = (
        PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE
    )
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Require the pinned package and explicit content-bound identities."""
        if self.mani_skill_version != MANISKILL_REQUIRED_VERSION:
            raise ManiSkillPickCubeAdapterConfigurationError(
                "ManiSkill PickCube adapter requires mani_skill==3.0.1"
            )
        if self.state_semantic_version != MANISKILL_STATE_SEMANTIC_VERSION:
            raise ManiSkillPickCubeAdapterConfigurationError(
                "ManiSkill PickCube adapter state semantic mismatch"
            )
        if (
            self.state_verification_semantic != PICKCUBE_STATE_VERIFICATION_SEMANTIC
            or type(self.state_verification_maximum_absolute_tolerance) is not float
            or self.state_verification_maximum_absolute_tolerance
            != PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE
        ):
            raise ManiSkillPickCubeAdapterConfigurationError(
                "ManiSkill PickCube adapter state verification contract mismatch"
            )
        if self.schema_version != "1.0":
            raise ManiSkillPickCubeAdapterConfigurationError(
                "ManiSkillPickCubeSemanticIdentity.schema_version is unsupported"
            )
        for field in (
            "compatibility_identity",
            "solver_identity",
            "task_implementation_identity",
            "controller_configuration_identity",
            "action_layout_digest",
        ):
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
                or "\\" in value
                or "/" in value
            ):
                raise ManiSkillPickCubeAdapterConfigurationError(
                    f"ManiSkillPickCubeSemanticIdentity.{field}: expected a "
                    "path-independent non-empty identity"
                )

    def as_mapping(self) -> Mapping[str, object]:
        """Return canonical semantic identities without operational metadata."""
        return MappingProxyType(
            {
                "action_layout_digest": self.action_layout_digest,
                "compatibility_identity": self.compatibility_identity,
                "controller_configuration_identity": (
                    self.controller_configuration_identity
                ),
                "mani_skill_version": self.mani_skill_version,
                "schema_version": self.schema_version,
                "solver_identity": self.solver_identity,
                "state_semantic_version": self.state_semantic_version,
                "state_verification_maximum_absolute_tolerance": (
                    self.state_verification_maximum_absolute_tolerance
                ),
                "state_verification_semantic": self.state_verification_semantic,
                "task_implementation_identity": self.task_implementation_identity,
            }
        )

    @classmethod
    def from_compatibility_binding(
        cls,
        binding: object,
        *,
        action_layout_digest: str,
    ) -> ManiSkillPickCubeSemanticIdentity:
        """Bind adapter identities only from a trusted checked-in probe report."""
        require_ready = getattr(binding, "require_trusted_replay_ready", None)
        if not callable(require_ready):
            raise ManiSkillPickCubeAdapterConfigurationError(
                "compatibility binding cannot authorize trusted replay"
            )
        require_ready()
        try:
            report = cast(Any, binding).report
            return cls(
                compatibility_identity=str(report.compatibility_identity),
                solver_identity=str(report.source_solver.source_sha256),
                task_implementation_identity=str(
                    report.task_implementation.source_sha256
                ),
                controller_configuration_identity=str(
                    report.controller.configuration_identity
                ),
                action_layout_digest=action_layout_digest,
                mani_skill_version=str(report.mani_skill_version),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ManiSkillPickCubeAdapterConfigurationError(
                "compatibility binding lacks content-bound adapter identities"
            ) from exc


@dataclass(frozen=True, slots=True)
class TrustedManiSkillRuntimeAttestation:
    """Explicit fail-closed proof that the live compatibility gates passed."""

    compatibility_identity: str
    semantic_configuration_digest: str
    compatibility_probe_passed: bool
    dependency_versions_verified: bool
    action_contract_verified: bool
    state_api_verified: bool
    state_round_trip_verified: bool
    task_contract_verified: bool
    solver_source_verified: bool
    task_source_verified: bool
    real_simulator_runtime: bool
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Reject partial, fixture, or unverified attestations."""
        if self.schema_version != "1.0":
            raise ManiSkillPickCubeAdapterConfigurationError(
                "TrustedManiSkillRuntimeAttestation.schema_version is unsupported"
            )
        for field in ("compatibility_identity", "semantic_configuration_digest"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ManiSkillPickCubeAdapterConfigurationError(
                    f"TrustedManiSkillRuntimeAttestation.{field}: required"
                )
        gates = (
            self.compatibility_probe_passed,
            self.dependency_versions_verified,
            self.action_contract_verified,
            self.state_api_verified,
            self.state_round_trip_verified,
            self.task_contract_verified,
            self.solver_source_verified,
            self.task_source_verified,
            self.real_simulator_runtime,
        )
        if any(type(value) is not bool for value in gates) or not all(gates):
            raise ManiSkillPickCubeAdapterConfigurationError(
                "trusted ManiSkill adapter requires every real-runtime gate to pass"
            )

    @classmethod
    def from_compatibility_binding(
        cls,
        binding: object,
        *,
        semantic_configuration_digest: str,
    ) -> TrustedManiSkillRuntimeAttestation:
        """Create trust permission after strict local configuration binding."""
        require_ready = getattr(binding, "require_trusted_replay_ready", None)
        if not callable(require_ready):
            raise ManiSkillPickCubeAdapterConfigurationError(
                "compatibility binding cannot authorize trusted replay"
            )
        require_ready()
        try:
            report = cast(Any, binding).report
            runtime_apis = report.runtime_apis
            state_round_trip = report.state_round_trip
            bounded_step = report.bounded_action_step
            close_check = report.environment_close
            keys = set(report.task_evaluator_keys)
            required_keys = {
                "success",
                "is_obj_placed",
                "is_robot_static",
                "is_grasped",
            }
            report_ready = bool(
                report.gpu_simulation
                and runtime_apis.complete
                and state_round_trip.passed
                and bounded_step.passed
                and close_check.passed
            )
            return cls(
                compatibility_identity=str(report.compatibility_identity),
                semantic_configuration_digest=semantic_configuration_digest,
                compatibility_probe_passed=report_ready,
                dependency_versions_verified=True,
                action_contract_verified=True,
                state_api_verified=bool(runtime_apis.complete),
                state_round_trip_verified=bool(state_round_trip.passed),
                task_contract_verified=required_keys.issubset(keys),
                solver_source_verified=bool(report.source_solver.source_sha256),
                task_source_verified=bool(report.task_implementation.source_sha256),
                real_simulator_runtime=bool(report.gpu_simulation),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ManiSkillPickCubeAdapterConfigurationError(
                "compatibility binding lacks complete trusted-runtime gates"
            ) from exc


@runtime_checkable
class PickCubeReplayReferenceValidator(Protocol):
    """Revalidate one state reference against the immutable runtime archive."""

    def validate_reference(self, reference: ReplayStateReference) -> None:
        """Raise when the reference or archive content changed."""
        ...


@runtime_checkable
class PickCubeReplaySessionFactory(Protocol):
    """Create one new environment-owning session for each replay role."""

    def create_session(
        self,
        replay_case: ReplayCase,
        *,
        execution_role: ReplayExecutionRole,
    ) -> ReplayEnvironmentSession:
        """Create a new exact replay session."""
        ...


def resolve_maniskill_pickcube_configuration(
    *,
    identity: ManiSkillPickCubeSemanticIdentity,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    task_keys: PickCubeTaskKeyContract,
) -> Mapping[str, object]:
    """Build canonical real-adapter semantics without any runtime paths."""
    value: Mapping[str, object] = MappingProxyType(
        {
            "action_contract": action_contract.as_mapping(),
            "adapter_id": MANISKILL_PICKCUBE_ADAPTER_ID,
            "adapter_version": MANISKILL_PICKCUBE_ADAPTER_VERSION,
            "environment": settings.as_mapping(),
            "identity": identity.as_mapping(),
            "progress_semantic": PICKCUBE_PROGRESS_SEMANTIC,
            "state_verification": MappingProxyType(
                {
                    "comparison_semantic": (
                        StateComparisonSemantic.NUMERIC_TOLERANCE.value
                    ),
                    "maximum_absolute_tolerance": (
                        PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE
                    ),
                    "runtime_semantic": PICKCUBE_STATE_VERIFICATION_SEMANTIC,
                }
            ),
            "task_contract_version": PICKCUBE_TASK_CONTRACT_VERSION,
            "task_evaluator_keys": task_keys.as_mapping(),
            "task_id": PICKCUBE_TASK_ID,
            "unsafe_semantic": PICKCUBE_UNSAFE_SEMANTIC,
        }
    )
    try:
        canonical_json_value(
            value,
            context="ManiSkillPickCubeAdapter.resolved_configuration",
            reject_runtime_paths=True,
        )
    except ReplayValidationError as exc:
        raise ManiSkillPickCubeAdapterConfigurationError(str(exc)) from exc
    return value


class ManiSkillPickCubeAdapter:
    """Exact-simulator trust ceiling backed by a verified real-runtime attestation."""

    def __init__(
        self,
        *,
        case_provider: ReplayCaseProvider,
        replay_bundle: ReplayBundle,
        identity: ManiSkillPickCubeSemanticIdentity,
        settings: ManiSkillPickCubeEnvironmentSettings,
        action_contract: PickCubeReplayActionContract,
        task_key_contract: PickCubeTaskKeyContract,
        reference_validator: PickCubeReplayReferenceValidator,
        session_factory: PickCubeReplaySessionFactory,
        trust_attestation: TrustedManiSkillRuntimeAttestation,
    ) -> None:
        """Bind data, semantics, archive validation, sessions, and trust gates."""
        if not isinstance(case_provider, ReplayCaseProvider):
            raise ManiSkillPickCubeAdapterConfigurationError(
                "case_provider must implement ReplayCaseProvider"
            )
        if not isinstance(reference_validator, PickCubeReplayReferenceValidator):
            raise ManiSkillPickCubeAdapterConfigurationError(
                "reference_validator must implement its strict protocol"
            )
        if not isinstance(session_factory, PickCubeReplaySessionFactory):
            raise ManiSkillPickCubeAdapterConfigurationError(
                "session_factory must implement its strict protocol"
            )
        validate_replay_bundle(replay_bundle)
        resolved = resolve_maniskill_pickcube_configuration(
            identity=identity,
            settings=settings,
            action_contract=action_contract,
            task_keys=task_key_contract,
        )
        configuration_digest = compute_configuration_digest(resolved)
        if trust_attestation.compatibility_identity != identity.compatibility_identity:
            raise ManiSkillPickCubeAdapterConfigurationError(
                "trusted runtime compatibility identity does not match adapter"
            )
        if trust_attestation.semantic_configuration_digest != configuration_digest:
            raise ManiSkillPickCubeAdapterConfigurationError(
                "trusted runtime semantic configuration does not match adapter"
            )
        for actual, expected, field in (
            (replay_bundle.adapter_id, self.adapter_id, "adapter_id"),
            (replay_bundle.adapter_version, self.adapter_version, "adapter_version"),
            (
                replay_bundle.adapter_configuration_digest,
                configuration_digest,
                "adapter_configuration_digest",
            ),
        ):
            if actual != expected:
                raise ManiSkillPickCubeAdapterConfigurationError(
                    f"replay bundle {field} does not match PickCube adapter"
                )
        cases: dict[str, ReplayCase] = {}
        for case in replay_bundle.replay_cases:
            self._validate_case_semantics(case, action_contract)
            if case.proposal_id in cases:
                raise ManiSkillPickCubeAdapterConfigurationError(
                    "replay bundle contains duplicate PickCube proposal IDs"
                )
            cases[case.proposal_id] = case
        self._case_provider = case_provider
        self._replay_bundle = replay_bundle
        self._identity = identity
        self._settings = settings
        self._action_contract = action_contract
        self._task_key_contract = task_key_contract
        self._reference_validator = reference_validator
        self._session_factory = session_factory
        self._trust_attestation = trust_attestation
        self._resolved_configuration = resolved
        self._configuration_digest = configuration_digest
        self._cases = MappingProxyType(cases)
        self._trust = ReplayTrustDescriptor(
            trust_tier=ReplayTrustTier.EXACT_SIMULATOR,
            label_source=LabelSource.SIMULATOR,
            maximum_label_strength=LabelStrength.STRONG,
            simulator_verification_allowed=True,
            exact_state_verification_required=True,
            state_verification_semantic=(StateComparisonSemantic.NUMERIC_TOLERANCE),
            state_verification_tolerance=(
                PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE
            ),
        )

    @property
    def adapter_id(self) -> str:
        """Return the stable real PickCube adapter identifier."""
        return MANISKILL_PICKCUBE_ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        """Return the semantic real PickCube adapter version."""
        return MANISKILL_PICKCUBE_ADAPTER_VERSION

    @property
    def configuration_digest(self) -> str:
        """Return the path-independent resolved-configuration digest."""
        return self._configuration_digest

    @property
    def replay_bundle(self) -> ReplayBundle:
        """Return the standard validated M2B replay bundle."""
        return self._replay_bundle

    def trust_descriptor(self) -> ReplayTrustDescriptor:
        """Return the exact-simulator ceiling granted by all compatibility gates."""
        return self._trust

    def resolved_configuration(self) -> Mapping[str, object]:
        """Return content-bound semantic configuration without runtime paths."""
        return self._resolved_configuration

    def resolve_case(self, proposal: CorruptedActionProposal) -> ReplayCase:
        """Revalidate data, archive reference, and action bounds before sessions."""
        resolved = self._case_provider.resolve_case(proposal)
        if not isinstance(resolved, ReplayCase):
            raise ReplayInvalidContextError(
                "PickCube case provider did not return ReplayCase"
            )
        validate_replay_case(resolved)
        canonical = self._cases.get(proposal.proposal_id)
        if canonical is None or canonical.case_id != resolved.case_id:
            raise ReplayInvalidContextError(
                "PickCube proposal is not bound to the supplied replay bundle"
            )
        self._validate_case_semantics(canonical, self._action_contract)
        self._action_contract.validate_action_chunk(
            canonical.original_action, role="baseline"
        )
        self._action_contract.validate_action_chunk(
            canonical.transformed_action, role="corrupted"
        )
        try:
            self._reference_validator.validate_reference(canonical.state_reference)
        except ReplayInvalidContextError:
            raise
        except Exception as exc:
            raise ReplayInvalidContextError(
                "PickCube runtime archive reference validation failed"
            ) from exc
        return canonical

    def create_session(
        self,
        replay_case: ReplayCase,
        *,
        execution_role: ReplayExecutionRole,
    ) -> ReplayEnvironmentSession:
        """Create a fresh baseline or corrupted environment session."""
        if not isinstance(execution_role, ReplayExecutionRole):
            raise ReplayExecutionError("unsupported PickCube execution role")
        canonical = self._cases.get(replay_case.proposal_id)
        if canonical is None or canonical.case_id != replay_case.case_id:
            raise ReplayInvalidContextError(
                "PickCube replay case is not bound to this adapter"
            )
        self._validate_case_semantics(canonical, self._action_contract)
        self._action_contract.validate_action_chunk(
            canonical.original_action, role="baseline"
        )
        self._action_contract.validate_action_chunk(
            canonical.transformed_action, role="corrupted"
        )
        self._reference_validator.validate_reference(canonical.state_reference)
        session = self._session_factory.create_session(
            canonical, execution_role=execution_role
        )
        if not isinstance(session, ReplayEnvironmentSession):
            raise ReplayExecutionError(
                "PickCube session factory did not return ReplayEnvironmentSession"
            )
        return session

    @classmethod
    def _validate_case_semantics(
        cls,
        replay_case: ReplayCase,
        action_contract: PickCubeReplayActionContract,
    ) -> None:
        validate_replay_case(replay_case)
        mismatches = tuple(
            field
            for field, actual, expected in (
                ("adapter_id", replay_case.adapter_id, MANISKILL_PICKCUBE_ADAPTER_ID),
                (
                    "adapter_version",
                    replay_case.adapter_version,
                    MANISKILL_PICKCUBE_ADAPTER_VERSION,
                ),
                ("task_id", replay_case.task_reference.task_id, PICKCUBE_TASK_ID),
                (
                    "task_contract_version",
                    replay_case.task_reference.task_contract_version,
                    PICKCUBE_TASK_CONTRACT_VERSION,
                ),
                (
                    "progress_semantic",
                    replay_case.progress_semantic,
                    PICKCUBE_PROGRESS_SEMANTIC,
                ),
                (
                    "unsafe_semantic",
                    replay_case.unsafe_semantic,
                    PICKCUBE_UNSAFE_SEMANTIC,
                ),
                (
                    "state_adapter_id",
                    replay_case.state_reference.adapter_id,
                    MANISKILL_PICKCUBE_ADAPTER_ID,
                ),
                (
                    "state_adapter_version",
                    replay_case.state_reference.adapter_version,
                    MANISKILL_PICKCUBE_ADAPTER_VERSION,
                ),
                (
                    "state_comparison_semantic",
                    replay_case.state_reference.comparison_semantic,
                    StateComparisonSemantic.NUMERIC_TOLERANCE,
                ),
                (
                    "state_verification_semantic",
                    replay_case.state_reference.metadata.get(
                        "state_verification_semantic"
                    ),
                    PICKCUBE_STATE_VERIFICATION_SEMANTIC,
                ),
                (
                    "state_verification_maximum_absolute_tolerance",
                    replay_case.state_reference.metadata.get(
                        "state_verification_maximum_absolute_tolerance"
                    ),
                    PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE,
                ),
            )
            if actual != expected
        )
        if mismatches:
            raise ReplayInvalidContextError(
                "PickCube replay case semantic mismatch: " + ", ".join(mismatches)
            )
        cls._validate_static_action_contract(
            replay_case.original_action,
            action_contract,
            role="baseline",
        )
        cls._validate_static_action_contract(
            replay_case.transformed_action,
            action_contract,
            role="corrupted",
        )

    @staticmethod
    def _validate_static_action_contract(
        action: ActionChunk,
        action_contract: PickCubeReplayActionContract,
        *,
        role: str,
    ) -> None:
        """Validate non-physical action structure while deferring bounds per case."""
        if action.actions.shape[1:] != action_contract.row_shape:
            raise ReplayInvalidContextError(f"{role}_action_shape_mismatch")
        if action.actions.dtype.hasobject or not np.issubdtype(
            action.actions.dtype, np.floating
        ):
            raise ReplayInvalidContextError(f"{role}_action_dtype_not_floating")
        if action.coordinate_frame != action_contract.coordinate_frame:
            raise ReplayInvalidContextError(f"{role}_action_coordinate_frame_mismatch")
        if action.control_period_s != action_contract.control_period_s:
            raise ReplayInvalidContextError(f"{role}_action_control_period_mismatch")
        if not np.all(np.isfinite(action.actions)):
            raise ReplayInvalidContextError(f"{role}_action_non_finite")


def create_maniskill_pickcube_adapter(
    *,
    case_provider: ReplayCaseProvider,
    replay_bundle: ReplayBundle,
    identity: ManiSkillPickCubeSemanticIdentity,
    settings: ManiSkillPickCubeEnvironmentSettings,
    action_contract: PickCubeReplayActionContract,
    task_key_contract: PickCubeTaskKeyContract,
    reference_validator: PickCubeReplayReferenceValidator,
    session_factory: PickCubeReplaySessionFactory,
    trust_attestation: TrustedManiSkillRuntimeAttestation,
) -> ExactReplayAdapter:
    """Create the explicitly attested real PickCube adapter."""
    return ManiSkillPickCubeAdapter(
        case_provider=case_provider,
        replay_bundle=replay_bundle,
        identity=identity,
        settings=settings,
        action_contract=action_contract,
        task_key_contract=task_key_contract,
        reference_validator=reference_validator,
        session_factory=session_factory,
        trust_attestation=trust_attestation,
    )


__all__ = [
    "MANISKILL_PICKCUBE_ADAPTER_ID",
    "MANISKILL_PICKCUBE_ADAPTER_VERSION",
    "MANISKILL_REQUIRED_VERSION",
    "MANISKILL_STATE_SEMANTIC_VERSION",
    "PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE",
    "PICKCUBE_STATE_VERIFICATION_SEMANTIC",
    "ManiSkillPickCubeAdapter",
    "ManiSkillPickCubeAdapterConfigurationError",
    "ManiSkillPickCubeSemanticIdentity",
    "PickCubeReplayReferenceValidator",
    "PickCubeReplaySessionFactory",
    "TrustedManiSkillRuntimeAttestation",
    "create_maniskill_pickcube_adapter",
    "resolve_maniskill_pickcube_configuration",
]

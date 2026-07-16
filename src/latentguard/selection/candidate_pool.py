"""Deterministic construction of blind eight-candidate M3C pools."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

from latentguard.corruptions.base import CorruptionApplicabilityError
from latentguard.corruptions.generation import (
    build_proposal_identifier,
    derive_corruption_seed,
)
from latentguard.corruptions.layout import ActionLayout
from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.integrations.maniskill_pickcube.session import (
    PickCubeReplayActionContract,
)
from latentguard.models import ActionChunk
from latentguard.replay.base import ReplayInvalidContextError
from latentguard.replay.identity import canonical_json_bytes
from latentguard.selection.configuration import CandidatePoolConfigurationV1
from latentguard.selection.models import (
    ACTION_DIMENSION,
    ACTION_HORIZON,
    CANDIDATE_COUNT,
    CandidateGroupV1,
    CandidatePoolV1,
    CandidateRefV1,
    SourceExclusionInventoryV1,
    SourceTrajectoryIdentityV1,
    array_content_digest,
)


class CandidatePoolBuildError(ValueError):
    """Raised when a blind pool cannot be built without repair or leakage."""


def _fail(context: str, reason: str) -> NoReturn:
    raise CandidatePoolBuildError(f"{context}: {reason}")


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected canonical non-empty text")
    return value


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(context, "expected lowercase sha256 content digest")
    return value


def _frozen_vector(value: object) -> NDArray[Any]:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (38,)
        or value.dtype != np.dtype("<f4")
        or not bool(np.all(np.isfinite(value)))
    ):
        _fail("CandidatePoolSourceV1.state_vector", "expected finite <f4 vector [38]")
    detached = np.array(value, copy=True, order="C", subok=False)
    return np.frombuffer(detached.tobytes(order="C"), dtype=detached.dtype)


@dataclass(frozen=True, slots=True, eq=False)
class CandidatePoolSourceV1:
    """Outcome-free source material for one candidate anchor."""

    anchor_id: str
    source_episode_id: str
    source_candidate_id: str
    source_policy_id: str
    source_task_id: str
    trajectory: SourceTrajectoryIdentityV1
    state_tree_digest: str
    state_content_digest: str
    verifier_state_content_digest: str
    continuation_identity: str
    state_vector: NDArray[Any]
    source_remaining_action: ActionChunk

    def __post_init__(self) -> None:
        """Detach the deployable state and validate source-only provenance."""

        for name in (
            "anchor_id",
            "source_episode_id",
            "source_candidate_id",
            "source_policy_id",
            "source_task_id",
        ):
            _text(getattr(self, name), f"CandidatePoolSourceV1.{name}")
        if not isinstance(self.trajectory, SourceTrajectoryIdentityV1):
            _fail("CandidatePoolSourceV1.trajectory", "expected trajectory identity")
        for name in (
            "state_tree_digest",
            "state_content_digest",
            "verifier_state_content_digest",
            "continuation_identity",
        ):
            _digest(getattr(self, name), f"CandidatePoolSourceV1.{name}")
        if self.state_tree_digest not in self.trajectory.complete_state_digests:
            _fail(
                "CandidatePoolSourceV1.state_tree_digest",
                "anchor state is absent from the complete trajectory inventory",
            )
        if not isinstance(self.source_remaining_action, ActionChunk):
            _fail(
                "CandidatePoolSourceV1.source_remaining_action", "expected ActionChunk"
            )
        actions = self.source_remaining_action.actions
        if (
            actions.ndim != 2
            or actions.shape[0] < ACTION_HORIZON
            or actions.shape[1] != ACTION_DIMENSION
        ):
            _fail(
                "CandidatePoolSourceV1.source_remaining_action",
                "expected floating action [T>=16, 8]",
            )
        object.__setattr__(self, "state_vector", _frozen_vector(self.state_vector))


def _validate_action_contract(
    action_contract: PickCubeReplayActionContract,
    action: ActionChunk,
    *,
    role: str,
) -> None:
    try:
        action_contract.validate_action_chunk(action, role=role)
    except ReplayInvalidContextError as exc:
        raise CandidatePoolBuildError(f"{role}: {exc}") from exc


def _source_set_digest(sources: Sequence[CandidatePoolSourceV1]) -> str:
    trajectories: dict[str, SourceTrajectoryIdentityV1] = {}
    anchors: list[dict[str, object]] = []
    for source in sources:
        existing = trajectories.get(source.trajectory.source_trajectory_id)
        if (
            existing is not None
            and existing.content_digest != source.trajectory.content_digest
        ):
            _fail("source set", "one trajectory ID has conflicting identities")
        trajectories[source.trajectory.source_trajectory_id] = source.trajectory
        anchors.append(
            {
                "anchor_id": source.anchor_id,
                "continuation_identity": source.continuation_identity,
                "source_candidate_id": source.source_candidate_id,
                "source_episode_id": source.source_episode_id,
                "state_content_digest": source.state_content_digest,
                "state_tree_digest": source.state_tree_digest,
                "verifier_state_content_digest": source.verifier_state_content_digest,
            }
        )
    payload = {
        "anchors": anchors,
        "schema_version": "1.0",
        "semantic": "m3c_external_source_set_v1",
        "trajectories": [
            trajectories[key].as_mapping() for key in sorted(trajectories)
        ],
    }
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def compute_source_set_digest(sources: Sequence[CandidatePoolSourceV1]) -> str:
    """Compute the canonical source-set identity before any candidate outcome."""

    values = tuple(sources)
    if not values:
        _fail("source set", "must not be empty")
    if any(not isinstance(item, CandidatePoolSourceV1) for item in values):
        _fail("source set", "expected CandidatePoolSourceV1 entries")
    return _source_set_digest(values)


def _candidate_for_definition(
    source: CandidatePoolSourceV1,
    configuration: CandidatePoolConfigurationV1,
    action_layout: ActionLayout,
    action_contract: PickCubeReplayActionContract,
    ordinal: int,
    generation_ordinal: int,
) -> CandidateRefV1:
    definition = configuration.candidates[ordinal]
    corruption = definition.build_corruption()
    source_action = source.source_remaining_action
    source_bytes = source_action.actions.tobytes(order="C")
    seed = derive_corruption_seed(
        base_seed=configuration.base_seed,
        source_episode_id=source.source_episode_id,
        source_candidate_id=source.source_candidate_id,
        corruption_name=corruption.name,
        configuration_ordinal=ordinal,
    )
    try:
        resolved = corruption.resolved_parameters_for(source_action, action_layout)
        transformed = corruption.apply(source_action, action_layout, seed=seed)
    except (CorruptionApplicabilityError, TypeError, ValueError) as exc:
        raise CandidatePoolBuildError(
            f"candidate slot {ordinal}: transformation is not applicable: {exc}"
        ) from exc
    if source_action.actions.tobytes(order="C") != source_bytes:
        _fail(f"candidate slot {ordinal}", "transformation mutated source actions")
    if (
        transformed.actions.shape != source_action.actions.shape
        or transformed.actions.dtype != source_action.actions.dtype
        or transformed.coordinate_frame != source_action.coordinate_frame
        or transformed.control_period_s != source_action.control_period_s
        or transformed.schema_version != source_action.schema_version
    ):
        _fail(f"candidate slot {ordinal}", "transformation changed action contract")
    source_prefix = source_action.actions[:ACTION_HORIZON]
    transformed_prefix = transformed.actions[:ACTION_HORIZON]
    if transformed_prefix.tobytes(order="C") == source_prefix.tobytes(order="C"):
        _fail(f"candidate slot {ordinal}", "source-equivalent no-op is prohibited")
    if transformed.actions[ACTION_HORIZON:].tobytes(order="C") != source_action.actions[
        ACTION_HORIZON:
    ].tobytes(order="C"):
        _fail(f"candidate slot {ordinal}", "fixed source continuation changed")
    _validate_action_contract(
        action_contract,
        transformed,
        role=f"candidate_slot_{ordinal}",
    )
    proposal_id = build_proposal_identifier(
        source_episode_id=source.source_episode_id,
        source_candidate_id=source.source_candidate_id,
        corruption_name=corruption.name,
        resolved_parameters=resolved,
        seed=seed,
        generation_ordinal=generation_ordinal,
    )
    try:
        CorruptedActionProposal(
            proposal_id=proposal_id,
            source_episode_id=source.source_episode_id,
            source_candidate_id=source.source_candidate_id,
            source_policy_id=source.source_policy_id,
            source_task_id=source.source_task_id,
            split_group_id=source.trajectory.split_group_id,
            transformed_action=transformed,
            corruption_type=corruption.name,
            resolved_parameters=resolved,
            seed=seed,
            generation_ordinal=generation_ordinal,
            notes="unlabeled M3C blind candidate proposal",
        )
    except (TypeError, ValueError) as exc:
        raise CandidatePoolBuildError(
            f"candidate slot {ordinal}: proposal identity failed: {exc}"
        ) from exc
    prefix = np.array(transformed_prefix, copy=True, order="C", subok=False)
    mask = np.ones(ACTION_HORIZON, dtype=np.bool_)
    return CandidateRefV1(
        proposal_id=proposal_id,
        configuration_ordinal=ordinal,
        distribution=definition.distribution,
        corruption_type=definition.corruption_type,
        severity_id=definition.severity_id,
        seed=seed,
        action_chunk=prefix,
        action_mask=mask,
    )


def build_candidate_pool(
    sources: Sequence[CandidatePoolSourceV1],
    *,
    configuration: CandidatePoolConfigurationV1,
    action_layout: ActionLayout,
    action_contract: PickCubeReplayActionContract,
    exclusion_inventory: SourceExclusionInventoryV1,
    expected_source_set_digest: str | None = None,
) -> CandidatePoolV1:
    """Build identical blind pools or fail instead of repairing any candidate."""

    values = tuple(sources)
    if not values or any(
        not isinstance(item, CandidatePoolSourceV1) for item in values
    ):
        _fail("candidate pool sources", "expected non-empty source inventory")
    if not isinstance(configuration, CandidatePoolConfigurationV1):
        _fail("candidate pool configuration", "invalid configuration")
    if not isinstance(action_layout, ActionLayout):
        _fail("candidate pool action layout", "invalid action layout")
    if not isinstance(action_contract, PickCubeReplayActionContract):
        _fail("candidate pool action contract", "invalid runtime action contract")
    if not isinstance(exclusion_inventory, SourceExclusionInventoryV1):
        _fail("candidate pool exclusions", "invalid exclusion inventory")
    if (
        action_layout.action_dim != ACTION_DIMENSION
        or action_contract.total_dimension != ACTION_DIMENSION
    ):
        _fail("candidate pool action contract", "expected action dimension 8")
    anchor_ids = tuple(item.anchor_id for item in values)
    if len(anchor_ids) != len(set(anchor_ids)):
        _fail("candidate pool sources", "anchor IDs must be unique")
    source_set_digest = _source_set_digest(values)
    if expected_source_set_digest is not None:
        _digest(expected_source_set_digest, "expected source-set digest")
        if expected_source_set_digest != source_set_digest:
            _fail("source set", "content differs from expected identity")
    groups: list[CandidateGroupV1] = []
    for source_index, source in enumerate(values):
        exclusion_inventory.require_disjoint(source.trajectory)
        _validate_action_contract(
            action_contract,
            source.source_remaining_action,
            role="source_remaining",
        )
        candidates = tuple(
            _candidate_for_definition(
                source,
                configuration,
                action_layout,
                action_contract,
                ordinal,
                source_index * CANDIDATE_COUNT + ordinal,
            )
            for ordinal in range(8)
        )
        source_prefix = np.ascontiguousarray(
            source.source_remaining_action.actions[:ACTION_HORIZON]
        )
        continuation = np.ascontiguousarray(
            source.source_remaining_action.actions[ACTION_HORIZON:]
        )
        groups.append(
            CandidateGroupV1(
                anchor_id=source.anchor_id,
                trajectory=source.trajectory,
                state_content_digest=source.state_content_digest,
                verifier_state_content_digest=source.verifier_state_content_digest,
                continuation_identity=source.continuation_identity,
                source_action_prefix_digest=array_content_digest(source_prefix),
                state_vector=source.state_vector,
                continuation_actions=continuation,
                candidates=candidates,
            )
        )
    try:
        return CandidatePoolV1(
            source_set_digest=source_set_digest,
            candidate_pool_configuration_digest=configuration.content_digest,
            action_contract_digest=configuration.action_contract_digest,
            exclusion_inventory_digest=exclusion_inventory.content_digest,
            groups=tuple(groups),
        )
    except (TypeError, ValueError) as exc:
        raise CandidatePoolBuildError(f"candidate pool: {exc}") from exc


__all__ = [
    "CandidatePoolBuildError",
    "CandidatePoolSourceV1",
    "build_candidate_pool",
    "compute_source_set_digest",
]

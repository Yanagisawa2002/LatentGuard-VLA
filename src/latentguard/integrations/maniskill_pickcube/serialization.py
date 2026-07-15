"""Build standard M2B cases and the trusted PickCube adapter from safe artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.evaluation.models import compute_configuration_digest
from latentguard.models import ActionChunk
from latentguard.replay.base import ReplayInvalidContextError
from latentguard.replay.identity import (
    compute_replay_bundle_digest,
    compute_replay_case_identifier,
)
from latentguard.replay.models import (
    ReplayBundle,
    ReplayCase,
    ReplayStateReference,
    ReplayTaskReference,
    StateComparisonSemantic,
)
from latentguard.replay.source import (
    ReplaySourceBinding,
    compute_episode_content_digest,
)

from .adapter import (
    MANISKILL_PICKCUBE_ADAPTER_ID,
    MANISKILL_PICKCUBE_ADAPTER_VERSION,
    PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE,
    PICKCUBE_STATE_VERIFICATION_SEMANTIC,
    ManiSkillPickCubeAdapter,
    ManiSkillPickCubeSemanticIdentity,
    TrustedManiSkillRuntimeAttestation,
    resolve_maniskill_pickcube_configuration,
)
from .archive import (
    ManiSkillReferenceArchive,
    ManiSkillReferenceEpisode,
    ReferenceArchiveError,
    assert_reference_archive_unchanged,
    load_reference_archive,
    validate_reference_archive,
)
from .compatibility import CompatibilityBinding
from .session import (
    ArchivePickCubeReferenceStateStore,
    CanonicalPickCubeStateTreeComparator,
    LazyManiSkillPickCubeRuntime,
    LoadedReferenceState,
    ManiSkillPickCubeEnvironmentSettings,
    ManiSkillPickCubeSessionFactory,
    PickCubeReplayActionContract,
)
from .source_import import M0SourceImportError, build_m0_source_episode
from .state_tree import STATE_TREE_SEMANTIC
from .task_evidence import (
    PICKCUBE_PROGRESS_SEMANTIC,
    PICKCUBE_TASK_CONTRACT_VERSION,
    PICKCUBE_TASK_ID,
    PICKCUBE_UNSAFE_SEMANTIC,
    PickCubeTaskKeyContract,
)


class ManiSkillPickCubeArtifactError(ValueError):
    """Raised when checked-in semantics and runtime archive content disagree."""


_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ManiSkillPickCubeArtifactError(
            f"{field} must be a lowercase sha256 content digest"
        )
    return value


@dataclass(frozen=True, slots=True)
class BoundArchiveReferenceStore:
    """Revalidate the full archive digest before loading every referenced state."""

    runtime_archive_directory: Path
    archive_content_digest: str

    def __post_init__(self) -> None:
        """Detach the runtime path and immediately verify its bound digest."""
        object.__setattr__(
            self,
            "runtime_archive_directory",
            Path(self.runtime_archive_directory).absolute(),
        )
        _require_sha256(
            self.archive_content_digest,
            field="reference archive content digest",
        )
        try:
            assert_reference_archive_unchanged(
                self.runtime_archive_directory, self.archive_content_digest
            )
        except ReferenceArchiveError as exc:
            raise ManiSkillPickCubeArtifactError(
                "PickCube runtime archive does not match its content binding"
            ) from exc

    def load_reference_state(
        self, reference: ReplayStateReference
    ) -> LoadedReferenceState:
        """Check archive immutability and load one digest-verified state."""
        try:
            assert_reference_archive_unchanged(
                self.runtime_archive_directory, self.archive_content_digest
            )
        except ReferenceArchiveError as exc:
            raise ReplayInvalidContextError(
                "PickCube runtime archive changed after content binding"
            ) from exc
        return ArchivePickCubeReferenceStateStore(
            self.runtime_archive_directory
        ).load_reference_state(reference)

    def validate_reference(self, reference: ReplayStateReference) -> None:
        """Require both full archive and selected state identities to match."""
        metadata_digest = reference.metadata.get("reference_archive_content_digest")
        if metadata_digest != self.archive_content_digest:
            raise ReplayInvalidContextError(
                "PickCube replay reference archive digest mismatch"
            )
        loaded = self.load_reference_state(reference)
        if loaded.state_digest != reference.expected_state_digest:
            raise ReplayInvalidContextError(
                "PickCube replay reference state digest mismatch"
            )


def action_contract_from_compatibility(
    binding: CompatibilityBinding,
    *,
    coordinate_frame: str,
) -> PickCubeReplayActionContract:
    """Build the numeric replay contract from a fully checked-in probe report."""
    binding.require_trusted_replay_ready()
    report = binding.report
    dtype = np.dtype(report.action_space.dtype)
    return PickCubeReplayActionContract(
        total_dimension=report.action_space.action_dimension,
        environment_numpy_dtype=dtype.str,
        lower_bounds=np.asarray(report.action_space.lower_bounds, dtype=dtype),
        upper_bounds=np.asarray(report.action_space.upper_bounds, dtype=dtype),
        environment_shape=report.action_space.batched_shape,
        coordinate_frame=coordinate_frame,
        control_period_s=report.control_period_s,
    )


def environment_settings_from_compatibility(
    binding: CompatibilityBinding,
) -> ManiSkillPickCubeEnvironmentSettings:
    """Build fixed session settings only after the report matches local config."""
    binding.require_trusted_replay_ready()
    report = binding.report
    return ManiSkillPickCubeEnvironmentSettings(
        obs_mode=report.observation_mode,
        state_tolerance=binding.expected_contract.state_round_trip_tolerance,
        environment_id=report.environment_id,
        robot_uid=report.robot_uid,
        num_envs=report.num_envs,
        control_mode=report.control_mode,
        sim_backend=report.sim_backend_request,
    )


class ManiSkillPickCubeCaseProvider:
    """Resolve M0/M1 pairs to archive-backed standard M2B replay cases."""

    def __init__(
        self,
        source_binding: ReplaySourceBinding,
        archive: ManiSkillReferenceArchive,
        *,
        adapter_configuration_digest: str,
        compatibility_identity: str,
        action_contract: PickCubeReplayActionContract,
        environment_settings: ManiSkillPickCubeEnvironmentSettings,
        source_solver_identity: Mapping[str, object],
    ) -> None:
        """Bind source, corruption, archive, and adapter semantics in memory."""
        source_binding.assert_unchanged()
        validate_reference_archive(archive)
        _require_sha256(
            compatibility_identity,
            field="compatibility identity",
        )
        self._source_binding = source_binding
        self._archive = archive
        self._action_contract = action_contract
        archive_by_id = {episode.episode_id: episode for episode in archive.episodes}
        self._archive_by_id = MappingProxyType(archive_by_id)
        source_ids = tuple(episode.episode_id for episode in source_binding.episodes)
        archive_ids = tuple(episode.episode_id for episode in archive.episodes)
        if source_ids != archive_ids:
            raise ManiSkillPickCubeArtifactError(
                "runtime archive episode order must exactly match the M0 "
                "source episodes"
            )
        for source_episode, episode in zip(
            source_binding.episodes, archive.episodes, strict=True
        ):
            self._validate_archive_episode(
                episode,
                compatibility_identity=compatibility_identity,
                action_contract=action_contract,
                environment_settings=environment_settings,
                source_solver_identity=source_solver_identity,
            )
            try:
                expected_source = build_m0_source_episode(episode)
            except M0SourceImportError as exc:
                raise ManiSkillPickCubeArtifactError(
                    "runtime archive episode cannot produce an accepted M0 source"
                ) from exc
            if compute_episode_content_digest(
                (source_episode,)
            ) != compute_episode_content_digest((expected_source,)):
                raise ManiSkillPickCubeArtifactError(
                    "M0 source episode content differs from the validated runtime "
                    "archive import"
                )
        cases = tuple(
            self._build_case(proposal, archive_by_id)
            for proposal in source_binding.corruption_dataset.proposals
        )
        self._cases = MappingProxyType(
            {replay_case.proposal_id: replay_case for replay_case in cases}
        )
        metadata: Mapping[str, object] = MappingProxyType(
            {
                "compatibility_identity": compatibility_identity,
                "reference_archive_content_digest": archive.content_digest,
                "real_simulator_reference": True,
                "state_semantic": STATE_TREE_SEMANTIC,
            }
        )
        bundle_digest = compute_replay_bundle_digest(
            source_dataset_id=source_binding.source_dataset_id,
            source_dataset_digest=source_binding.source_dataset_digest,
            corruption_dataset_digest=source_binding.corruption_dataset_digest,
            adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
            adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
            adapter_configuration_digest=adapter_configuration_digest,
            replay_cases=cases,
            metadata=metadata,
        )
        self._bundle = ReplayBundle(
            source_dataset_id=source_binding.source_dataset_id,
            source_dataset_digest=source_binding.source_dataset_digest,
            corruption_dataset_digest=source_binding.corruption_dataset_digest,
            adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
            adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
            adapter_configuration_digest=adapter_configuration_digest,
            replay_cases=cases,
            bundle_digest=bundle_digest,
            metadata=metadata,
        )
        source_binding.assert_unchanged()

    @property
    def provider_id(self) -> str:
        """Return the stable provider identity."""
        return MANISKILL_PICKCUBE_ADAPTER_ID

    @property
    def provider_version(self) -> str:
        """Return the provider semantic version."""
        return MANISKILL_PICKCUBE_ADAPTER_VERSION

    @property
    def replay_bundle(self) -> ReplayBundle:
        """Return the validated standard M2B bundle."""
        return self._bundle

    def resolve_case(self, proposal: CorruptedActionProposal) -> ReplayCase:
        """Revalidate the immutable M0/M1 pair and return its canonical case."""
        pair = self._source_binding.resolve(proposal)
        try:
            replay_case = self._cases[pair.proposal_id]
        except KeyError as exc:
            raise ReplayInvalidContextError(
                "PickCube proposal is not present in the content-bound replay bundle"
            ) from exc
        if replay_case.source_episode_id != pair.source_episode_id:
            raise ReplayInvalidContextError(
                "PickCube source episode changed after replay-case binding"
            )
        episode = self._archive_by_id.get(pair.source_episode_id)
        if episode is None or not _action_chunks_equal(
            pair.original_action, _archive_action_chunk(episode)
        ):
            raise ReplayInvalidContextError(
                "PickCube M0 source action differs from the runtime archive"
            )
        self._action_contract.validate_action_chunk(
            pair.original_action, role="baseline"
        )
        self._action_contract.validate_action_chunk(
            pair.transformed_action, role="corrupted"
        )
        return replay_case

    @staticmethod
    def _validate_archive_episode(
        episode: ManiSkillReferenceEpisode,
        *,
        compatibility_identity: str,
        action_contract: PickCubeReplayActionContract,
        environment_settings: ManiSkillPickCubeEnvironmentSettings,
        source_solver_identity: Mapping[str, object],
    ) -> None:
        if episode.compatibility_identity != compatibility_identity:
            raise ManiSkillPickCubeArtifactError(
                "runtime archive compatibility identity mismatch"
            )
        if dict(episode.environment_configuration) != dict(
            environment_settings.as_mapping()
        ):
            raise ManiSkillPickCubeArtifactError(
                "runtime archive environment settings differ from the probe"
            )
        if dict(episode.source_solver_identity) != dict(source_solver_identity):
            raise ManiSkillPickCubeArtifactError(
                "runtime archive source solver identity differs from the probe"
            )
        if dict(episode.action_contract) != dict(action_contract.as_mapping()):
            raise ManiSkillPickCubeArtifactError(
                "runtime archive action contract differs from the probe"
            )
        action_contract.validate_action_chunk(
            _archive_action_chunk(episode), role="archived_source"
        )
        if not episode.independent_baseline_success:
            raise ManiSkillPickCubeArtifactError(
                "runtime archive contains a source without baseline success"
            )

    def _build_case(
        self,
        proposal: CorruptedActionProposal,
        archive_by_id: Mapping[str, ManiSkillReferenceEpisode],
    ) -> ReplayCase:
        pair = self._source_binding.resolve(proposal)
        try:
            episode = archive_by_id[pair.source_episode_id]
        except KeyError as exc:
            raise ManiSkillPickCubeArtifactError(
                "M0 source episode has no runtime archive reference"
            ) from exc
        self._action_contract.validate_action_chunk(
            pair.original_action, role="baseline"
        )
        self._action_contract.validate_action_chunk(
            pair.transformed_action, role="corrupted"
        )
        if not _action_chunks_equal(
            pair.original_action, _archive_action_chunk(episode)
        ):
            raise ManiSkillPickCubeArtifactError(
                "M0 source action differs from the runtime archive trajectory"
            )
        state_reference = ReplayStateReference(
            adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
            adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
            source_reference_id=episode.episode_id,
            expected_state_digest=episode.initial_state_digest,
            comparison_semantic=StateComparisonSemantic.NUMERIC_TOLERANCE,
            state_key="initial_state",
            metadata={
                "compatibility_identity": episode.compatibility_identity,
                "reference_archive_content_digest": self._archive.content_digest,
                "source_action_digest": episode.source_action_digest,
                "source_reset_seed": episode.seed,
                "source_trajectory_id": episode.source_trajectory_id,
                "state_semantic": STATE_TREE_SEMANTIC,
                "state_verification_maximum_absolute_tolerance": (
                    PICKCUBE_STATE_VERIFICATION_MAX_ABSOLUTE_TOLERANCE
                ),
                "state_verification_semantic": (PICKCUBE_STATE_VERIFICATION_SEMANTIC),
            },
        )
        task_reference = ReplayTaskReference(
            task_id=PICKCUBE_TASK_ID,
            task_contract_version=PICKCUBE_TASK_CONTRACT_VERSION,
            metadata={
                "environment_id": "PickCube-v1",
                "robot_uid": "panda",
            },
        )
        case_id = compute_replay_case_identifier(
            proposal_id=pair.proposal_id,
            source_dataset_id=pair.source_dataset_id,
            source_dataset_digest=pair.source_dataset_digest,
            corruption_dataset_digest=pair.corruption_dataset_digest,
            source_episode_id=pair.source_episode_id,
            source_candidate_id=pair.source_candidate_id,
            split_group_id=pair.split_group_id,
            original_action=pair.original_action,
            transformed_action=pair.transformed_action,
            state_reference=state_reference,
            task_reference=task_reference,
            adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
            adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
            progress_semantic=PICKCUBE_PROGRESS_SEMANTIC,
            unsafe_semantic=PICKCUBE_UNSAFE_SEMANTIC,
        )
        return ReplayCase(
            case_id=case_id,
            proposal_id=pair.proposal_id,
            source_dataset_id=pair.source_dataset_id,
            source_dataset_digest=pair.source_dataset_digest,
            corruption_dataset_digest=pair.corruption_dataset_digest,
            source_episode_id=pair.source_episode_id,
            source_candidate_id=pair.source_candidate_id,
            split_group_id=pair.split_group_id,
            original_action=pair.original_action,
            transformed_action=pair.transformed_action,
            state_reference=state_reference,
            task_reference=task_reference,
            adapter_id=MANISKILL_PICKCUBE_ADAPTER_ID,
            adapter_version=MANISKILL_PICKCUBE_ADAPTER_VERSION,
            progress_semantic=PICKCUBE_PROGRESS_SEMANTIC,
            unsafe_semantic=PICKCUBE_UNSAFE_SEMANTIC,
        )


def _archive_action_chunk(episode: ManiSkillReferenceEpisode) -> ActionChunk:
    return ActionChunk(
        actions=episode.source_actions,
        coordinate_frame=episode.action_coordinate_frame,
        control_period_s=episode.control_period_s,
    )


def _action_chunks_equal(left: ActionChunk, right: ActionChunk) -> bool:
    return bool(
        left.coordinate_frame == right.coordinate_frame
        and left.control_period_s == right.control_period_s
        and left.schema_version == right.schema_version
        and left.actions.dtype == right.actions.dtype
        and left.actions.shape == right.actions.shape
        and left.actions.tobytes(order="C") == right.actions.tobytes(order="C")
    )


def build_maniskill_pickcube_adapter(
    *,
    source_binding: ReplaySourceBinding,
    archive_dir: Path,
    compatibility_binding: CompatibilityBinding,
    action_layout_digest: str,
    coordinate_frame: str,
) -> ManiSkillPickCubeAdapter:
    """Load static artifacts and construct the trusted real adapter lazily."""
    compatibility_binding.require_trusted_replay_ready()
    _require_sha256(action_layout_digest, field="action layout digest")
    try:
        archive = load_reference_archive(archive_dir)
    except ReferenceArchiveError as exc:
        raise ManiSkillPickCubeArtifactError(str(exc)) from exc
    identity = ManiSkillPickCubeSemanticIdentity.from_compatibility_binding(
        compatibility_binding,
        action_layout_digest=action_layout_digest,
    )
    settings = environment_settings_from_compatibility(compatibility_binding)
    action_contract = action_contract_from_compatibility(
        compatibility_binding,
        coordinate_frame=coordinate_frame,
    )
    task_keys = PickCubeTaskKeyContract.from_compatibility_report(
        compatibility_binding.report
    )
    resolved = resolve_maniskill_pickcube_configuration(
        identity=identity,
        settings=settings,
        action_contract=action_contract,
        task_keys=task_keys,
    )
    configuration_digest = compute_configuration_digest(resolved)
    provider = ManiSkillPickCubeCaseProvider(
        source_binding,
        archive,
        adapter_configuration_digest=configuration_digest,
        compatibility_identity=compatibility_binding.report.compatibility_identity,
        action_contract=action_contract,
        environment_settings=settings,
        source_solver_identity=compatibility_binding.report.source_solver.to_dict(),
    )
    state_store = BoundArchiveReferenceStore(
        runtime_archive_directory=Path(archive_dir),
        archive_content_digest=archive.content_digest,
    )
    session_factory = ManiSkillPickCubeSessionFactory(
        settings=settings,
        action_contract=action_contract,
        task_key_contract=task_keys,
        state_loader=state_store,
        state_comparator=CanonicalPickCubeStateTreeComparator(),
        runtime=LazyManiSkillPickCubeRuntime(),
    )
    attestation = TrustedManiSkillRuntimeAttestation.from_compatibility_binding(
        compatibility_binding,
        semantic_configuration_digest=configuration_digest,
    )
    return ManiSkillPickCubeAdapter(
        case_provider=provider,
        replay_bundle=provider.replay_bundle,
        identity=identity,
        settings=settings,
        action_contract=action_contract,
        task_key_contract=task_keys,
        reference_validator=state_store,
        session_factory=session_factory,
        trust_attestation=attestation,
    )


__all__ = [
    "ManiSkillPickCubeArtifactError",
    "ManiSkillPickCubeCaseProvider",
    "BoundArchiveReferenceStore",
    "action_contract_from_compatibility",
    "build_maniskill_pickcube_adapter",
    "environment_settings_from_compatibility",
]

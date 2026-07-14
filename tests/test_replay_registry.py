"""Strict configuration and explicit replay-adapter registry tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from latentguard.corruptions.config import load_corruption_plan
from latentguard.corruptions.generation import generate_corruption_proposals
from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.corruptions.serialization import CorruptionDataset
from latentguard.evaluation.models import compute_configuration_digest
from latentguard.replay.base import ReplayEnvironmentSession
from latentguard.replay.fixture import (
    FIXTURE_ADAPTER_ID,
    DeterministicReplayFixtureAdapter,
    FixtureReplayConfiguration,
    FixtureReplayConfigurationError,
    create_deterministic_replay_fixture_adapter,
    resolve_fixture_configuration,
)
from latentguard.replay.models import ReplayCase, ReplayExecutionRole
from latentguard.replay.registry import (
    DuplicateReplayAdapterError,
    ReplayAdapterRegistry,
    ReplayAdapterRegistryError,
    UnknownReplayAdapterError,
    create_replay_adapter,
    create_replay_evaluator,
    default_replay_adapter_registry,
    load_replay_adapter_configuration,
)
from latentguard.replay.source import ReplaySourceBinding
from latentguard.synthetic import generate_synthetic_episodes

_CORRUPTION_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "corruptions" / "m1-smoke.json"
)


def _configuration(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "1.0",
        "state_dimension": 7,
        "action_scale": 1.0,
        "progress_scale": 1.0,
        "success_tolerance": 1e-6,
        "unsafe_bound": 10.0,
        "baseline_failure_ordinals": [],
        "restoration_mismatch_ordinals": [],
        "indeterminate_ordinals": [],
        "step_exception_ordinals": [],
        "close_exception_ordinals": [],
    }
    value.update(updates)
    return value


def _binding(*, proposal_limit: int = 3) -> ReplaySourceBinding:
    episodes = generate_synthetic_episodes(
        seed=42,
        episode_count=1,
        episode_length=8,
        action_dim=7,
        robot_state_dim=10,
        camera_count=0,
    )
    plan = load_corruption_plan(_CORRUPTION_CONFIG)
    generated = generate_corruption_proposals(
        episodes,
        plan.action_layout,
        plan.corruptions,
        base_seed=314159,
        proposal_limit=proposal_limit,
    )
    dataset = CorruptionDataset(
        source_dataset_id="sha256:" + "b" * 64,
        action_layout=plan.action_layout,
        proposals=generated.proposals,
    )
    return ReplaySourceBinding.from_datasets(episodes, dataset)


@pytest.mark.parametrize(
    "configuration",
    [
        _configuration(unknown=True),
        {
            key: value
            for key, value in _configuration().items()
            if key != "unsafe_bound"
        },
        _configuration(schema_version="2.0"),
        _configuration(state_dimension=True),
        _configuration(state_dimension=0),
        _configuration(action_scale=0.0),
        _configuration(progress_scale=float("inf")),
        _configuration(success_tolerance=-1.0),
        _configuration(unsafe_bound=float("nan")),
        _configuration(baseline_failure_ordinals=[2, 1]),
        _configuration(baseline_failure_ordinals=[1, 1]),
        _configuration(
            baseline_failure_ordinals=[1], restoration_mismatch_ordinals=[1]
        ),
        _configuration(step_exception_ordinals=[False]),
    ],
)
def test_fixture_configuration_rejects_unknown_invalid_and_ambiguous_values(
    configuration: Mapping[str, object],
) -> None:
    with pytest.raises(FixtureReplayConfigurationError):
        resolve_fixture_configuration(configuration)


def test_fixture_configuration_is_canonical_and_deterministic() -> None:
    first = resolve_fixture_configuration(_configuration(indeterminate_ordinals=[2]))
    second = resolve_fixture_configuration(_configuration(indeterminate_ordinals=[2]))

    assert first == second
    assert first.as_mapping() == second.as_mapping()
    assert first.indeterminate_ordinals == (2,)


@pytest.mark.parametrize(
    "updates",
    [
        {"schema_version": "999"},
        {"progress_scale": 0.0},
        {"success_tolerance": -1.0},
        {"baseline_failure_ordinals": (1, 0)},
        {"baseline_failure_ordinals": (1, 1)},
        {
            "baseline_failure_ordinals": (1,),
            "restoration_mismatch_ordinals": (1,),
        },
    ],
)
def test_direct_fixture_configuration_construction_is_strict(
    updates: Mapping[str, object],
) -> None:
    fields: dict[str, object] = {
        "state_dimension": 7,
        "action_scale": 1.0,
        "progress_scale": 1.0,
        "success_tolerance": 1e-6,
        "unsafe_bound": 10.0,
        "baseline_failure_ordinals": (),
        "restoration_mismatch_ordinals": (),
        "indeterminate_ordinals": (),
        "step_exception_ordinals": (),
        "close_exception_ordinals": (),
        "schema_version": "1.0",
    }
    fields.update(updates)

    with pytest.raises(FixtureReplayConfigurationError):
        FixtureReplayConfiguration(**fields)  # type: ignore[arg-type]


def test_adapter_revalidates_direct_configuration_at_sdk_boundary() -> None:
    configuration = FixtureReplayConfiguration(
        state_dimension=7,
        action_scale=1.0,
        progress_scale=1.0,
        success_tolerance=1e-6,
        unsafe_bound=10.0,
        baseline_failure_ordinals=(),
        restoration_mismatch_ordinals=(),
        indeterminate_ordinals=(),
        step_exception_ordinals=(),
        close_exception_ordinals=(),
    )
    object.__setattr__(configuration, "progress_scale", 0.0)

    with pytest.raises(FixtureReplayConfigurationError, match="progress_scale"):
        DeterministicReplayFixtureAdapter(_binding(), configuration)


def test_fixture_adapter_rejects_selectors_outside_bound_dataset() -> None:
    with pytest.raises(FixtureReplayConfigurationError, match="outside"):
        create_deterministic_replay_fixture_adapter(
            _configuration(baseline_failure_ordinals=[2]),
            _binding(proposal_limit=2),
        )


def test_default_registry_contains_only_fixture_and_creates_evaluator() -> None:
    binding = _binding()
    registry = default_replay_adapter_registry()

    adapter = registry.create(FIXTURE_ADAPTER_ID, _configuration(), binding)
    evaluator = create_replay_evaluator(FIXTURE_ADAPTER_ID, _configuration(), binding)

    assert registry.names == (FIXTURE_ADAPTER_ID,)
    assert isinstance(adapter, DeterministicReplayFixtureAdapter)
    assert evaluator.adapter.adapter_id == FIXTURE_ADAPTER_ID
    assert (
        evaluator.replay_bundle.source_dataset_digest == binding.source_dataset_digest
    )
    assert (
        evaluator.replay_bundle.corruption_dataset_digest
        == binding.corruption_dataset_digest
    )


def test_adapter_and_evaluator_creation_resolve_without_creating_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()

    def forbidden_session(
        self: DeterministicReplayFixtureAdapter,
        replay_case: ReplayCase,
        *,
        execution_role: ReplayExecutionRole,
    ) -> ReplayEnvironmentSession:
        del self, replay_case, execution_role
        raise AssertionError("dry construction must not create a session")

    monkeypatch.setattr(
        DeterministicReplayFixtureAdapter, "create_session", forbidden_session
    )
    evaluator = create_replay_evaluator(FIXTURE_ADAPTER_ID, _configuration(), binding)

    for proposal in binding.corruption_dataset.proposals:
        decision = evaluator.check_applicability(
            proposal, source_dataset_id=binding.source_dataset_id
        )
        assert decision.status is None


def test_registry_rejects_duplicate_unknown_and_dynamic_import_like_names() -> None:
    binding = _binding()
    registry = ReplayAdapterRegistry(
        ((FIXTURE_ADAPTER_ID, create_deterministic_replay_fixture_adapter),)
    )
    with pytest.raises(DuplicateReplayAdapterError):
        registry.register(
            FIXTURE_ADAPTER_ID, create_deterministic_replay_fixture_adapter
        )
    for name in ("unknown", "package.module:Adapter", "os.system"):
        with pytest.raises(UnknownReplayAdapterError):
            registry.create(name, _configuration(), binding)


class _AdapterProxy:
    def __init__(
        self,
        delegate: DeterministicReplayFixtureAdapter,
        *,
        adapter_id: str | None = None,
        adapter_version: str | None = None,
        configuration_digest: str | None = None,
        resolved_configuration: Mapping[str, object] | None = None,
    ) -> None:
        self.delegate = delegate
        self._adapter_id = adapter_id
        self._adapter_version = adapter_version
        self._configuration_digest = configuration_digest
        self._resolved_configuration = resolved_configuration

    @property
    def adapter_id(self) -> str:
        return self._adapter_id or self.delegate.adapter_id

    @property
    def adapter_version(self) -> str:
        return self._adapter_version or self.delegate.adapter_version

    @property
    def configuration_digest(self) -> str:
        return self._configuration_digest or self.delegate.configuration_digest

    def trust_descriptor(self):  # type: ignore[no-untyped-def]
        return self.delegate.trust_descriptor()

    def resolved_configuration(self) -> Mapping[str, object]:
        return self._resolved_configuration or self.delegate.resolved_configuration()

    def resolve_case(self, proposal: CorruptedActionProposal) -> ReplayCase:
        return self.delegate.resolve_case(proposal)

    def create_session(
        self, replay_case: ReplayCase, *, execution_role: ReplayExecutionRole
    ) -> ReplayEnvironmentSession:
        return self.delegate.create_session(replay_case, execution_role=execution_role)


@pytest.mark.parametrize(
    "overrides",
    [
        {"adapter_id": "wrong_fixture"},
        {"adapter_version": "not-semver"},
        {"configuration_digest": "cfg-sha256-" + "0" * 64},
    ],
)
def test_registry_rejects_factory_identity_and_digest_mismatches(
    overrides: Mapping[str, str],
) -> None:
    binding = _binding()

    def factory(
        configuration: Mapping[str, object], source_binding: ReplaySourceBinding
    ) -> Any:
        delegate = create_deterministic_replay_fixture_adapter(
            configuration, source_binding
        )
        return _AdapterProxy(delegate, **overrides)

    registry = ReplayAdapterRegistry(((FIXTURE_ADAPTER_ID, factory),))
    with pytest.raises(ReplayAdapterRegistryError):
        registry.create(FIXTURE_ADAPTER_ID, _configuration(), binding)


def test_safe_configuration_loader_round_trips_checked_in_fixture() -> None:
    path = (
        Path(__file__).resolve().parents[1] / "configs" / "replay" / "m2b-fixture.json"
    )
    loaded = load_replay_adapter_configuration(path)
    adapter = create_replay_adapter(FIXTURE_ADAPTER_ID, loaded, _binding())

    assert adapter.adapter_id == FIXTURE_ADAPTER_ID
    assert loaded["state_dimension"] == 7


def test_registry_rejects_runtime_paths_in_resolved_adapter_identity() -> None:
    binding = _binding()
    resolved = {"schema_version": "1.0", "runtime_path": "C:\\private\\state.bin"}

    def factory(
        configuration: Mapping[str, object], source_binding: ReplaySourceBinding
    ) -> Any:
        delegate = create_deterministic_replay_fixture_adapter(
            configuration, source_binding
        )
        return _AdapterProxy(
            delegate,
            configuration_digest=compute_configuration_digest(resolved),
            resolved_configuration=resolved,
        )

    registry = ReplayAdapterRegistry(((FIXTURE_ADAPTER_ID, factory),))
    with pytest.raises(ReplayAdapterRegistryError, match="runtime paths"):
        registry.create(FIXTURE_ADAPTER_ID, _configuration(), binding)


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"schema_version":"1.0","schema_version":"1.0"}',
        '{"value":NaN}',
    ],
)
def test_safe_configuration_loader_rejects_noncanonical_json(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "config.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ReplayAdapterRegistryError):
        load_replay_adapter_configuration(path)

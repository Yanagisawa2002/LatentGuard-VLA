"""Round-trip, tamper, and filesystem-safety tests for replay bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from latentguard.corruptions.generation import generate_corruption_proposals
from latentguard.corruptions.layout import ActionField, ActionLayout, ActionSemantic
from latentguard.corruptions.serialization import CorruptionDataset
from latentguard.corruptions.transforms import ConstantBias
from latentguard.evaluation.models import compute_configuration_digest
from latentguard.replay.identity import (
    compute_replay_case_identifier,
)
from latentguard.replay.models import (
    ReplayCase,
    ReplayStateReference,
    ReplayTaskReference,
    StateComparisonSemantic,
)
from latentguard.replay.serialization import (
    MANIFEST_NAME,
    ReplaySerializationError,
    UnsupportedReplaySerializationVersionError,
    build_replay_bundle,
    load_replay_bundle,
    save_replay_bundle,
)
from latentguard.replay.source import ReplaySourceBinding
from latentguard.synthetic import generate_synthetic_episodes


def _binding() -> ReplaySourceBinding:
    episodes = generate_synthetic_episodes(
        seed=42,
        episode_count=2,
        episode_length=4,
        action_dim=3,
        robot_state_dim=5,
        camera_count=0,
        candidate_count=2,
    )
    layout = ActionLayout(
        action_dim=3,
        fields=(
            ActionField(
                name="control",
                indices=(0, 1, 2),
                semantic=ActionSemantic.AUXILIARY,
            ),
        ),
    )
    generated = generate_corruption_proposals(
        episodes,
        layout,
        (ConstantBias(bias=0.125, target_indices=(0,)),),
        base_seed=314159,
    )
    dataset = CorruptionDataset(
        source_dataset_id="sha256:" + "a" * 64,
        action_layout=layout,
        proposals=generated.proposals,
    )
    return ReplaySourceBinding.from_datasets(episodes, dataset)


def _case(binding: ReplaySourceBinding, proposal_index: int = 0) -> ReplayCase:
    proposal = binding.corruption_dataset.proposals[proposal_index]
    pair = binding.resolve(proposal)
    adapter_id = "deterministic_replay_fixture"
    adapter_version = "1.0.0"
    state_reference = ReplayStateReference(
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        source_reference_id=f"state-{pair.source_episode_id}",
        expected_state_digest="sha256:"
        + hashlib.sha256(pair.source_episode_id.encode("utf-8")).hexdigest(),
        comparison_semantic=StateComparisonSemantic.EXACT_DIGEST,
        state_index=0,
        metadata={"fixture": True, "ordinal": pair.generation_ordinal},
    )
    task_reference = ReplayTaskReference(
        task_id=pair.source_task_id,
        task_contract_version="1.0.0",
        metadata={"target_rule": "source_action_terminal"},
    )
    values: dict[str, object] = {
        "proposal_id": pair.proposal_id,
        "source_dataset_id": pair.source_dataset_id,
        "source_dataset_digest": pair.source_dataset_digest,
        "corruption_dataset_digest": pair.corruption_dataset_digest,
        "source_episode_id": pair.source_episode_id,
        "source_candidate_id": pair.source_candidate_id,
        "split_group_id": pair.split_group_id,
        "original_action": pair.original_action,
        "transformed_action": pair.transformed_action,
        "state_reference": state_reference,
        "task_reference": task_reference,
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "progress_semantic": "normalized_target_distance",
        "unsafe_semantic": "synthetic_state_bound",
    }
    case_id = compute_replay_case_identifier(**values)  # type: ignore[arg-type]
    return ReplayCase(case_id=case_id, **values)  # type: ignore[arg-type]


def _bundle(binding: ReplaySourceBinding, *, case_count: int = 2):
    configuration_digest = compute_configuration_digest(
        {"schema_version": "1.0", "state_dimension": 5}
    )
    return build_replay_bundle(
        binding,
        tuple(_case(binding, index) for index in range(case_count)),
        adapter_id="deterministic_replay_fixture",
        adapter_version="1.0.0",
        adapter_configuration_digest=configuration_digest,
        metadata={"purpose": "fixture infrastructure test"},
    )


def _manifest(path: Path) -> dict[str, Any]:
    return json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    (path / MANIFEST_NAME).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_replay_bundle_round_trip_reconstructs_actions_without_duplication(
    tmp_path: Path,
) -> None:
    binding = _binding()
    bundle = _bundle(binding)
    output = tmp_path / "bundle"

    manifest_path = save_replay_bundle(bundle, output, binding)
    loaded = load_replay_bundle(output, binding)

    assert manifest_path == output / MANIFEST_NAME
    assert loaded.bundle_digest == bundle.bundle_digest
    assert loaded.source_dataset_id == bundle.source_dataset_id
    assert loaded.source_dataset_digest == bundle.source_dataset_digest
    assert loaded.corruption_dataset_digest == bundle.corruption_dataset_digest
    assert [case.case_id for case in loaded.replay_cases] == [
        case.case_id for case in bundle.replay_cases
    ]
    assert set(path.name for path in output.iterdir()) == {MANIFEST_NAME}
    text = manifest_path.read_text(encoding="utf-8")
    assert '"original_action_reference"' in text
    assert '"transformed_action_reference"' in text
    assert '"actions"' not in text
    for actual, expected in zip(loaded.replay_cases, bundle.replay_cases, strict=True):
        assert (
            actual.original_action.actions.dtype
            == expected.original_action.actions.dtype
        )
        assert (
            actual.original_action.actions.shape
            == expected.original_action.actions.shape
        )
        assert (
            actual.transformed_action.actions.dtype
            == expected.transformed_action.actions.dtype
        )
        assert (
            actual.transformed_action.actions.shape
            == expected.transformed_action.actions.shape
        )
        np.testing.assert_array_equal(
            actual.original_action.actions, expected.original_action.actions
        )
        np.testing.assert_array_equal(
            actual.transformed_action.actions, expected.transformed_action.actions
        )


def test_bundle_digest_and_manifest_are_path_independent(tmp_path: Path) -> None:
    binding = _binding()
    bundle = _bundle(binding)
    first = tmp_path / "first"
    second = tmp_path / "elsewhere" / "second"

    save_replay_bundle(bundle, first, binding)
    save_replay_bundle(bundle, second, binding)

    assert (first / MANIFEST_NAME).read_bytes() == (second / MANIFEST_NAME).read_bytes()
    assert load_replay_bundle(first, binding).bundle_digest == (
        load_replay_bundle(second, binding).bundle_digest
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("bundle_digest", "bundle_digest"),
        ("case_id", "case_id"),
        ("original_action_digest", "original_action_digest"),
        ("transformed_action_digest", "transformed_action_digest"),
        ("source_dataset_digest", "source_dataset_digest"),
        ("corruption_dataset_digest", "corruption_dataset_digest"),
    ],
)
def test_load_rejects_tampering(tmp_path: Path, mutation: str, message: str) -> None:
    binding = _binding()
    output = tmp_path / mutation
    save_replay_bundle(_bundle(binding), output, binding)
    manifest = _manifest(output)
    if mutation in {
        "bundle_digest",
        "source_dataset_digest",
        "corruption_dataset_digest",
    }:
        manifest[mutation] = "sha256:" + "0" * 64
    else:
        cases = manifest["replay_cases"]
        assert isinstance(cases, list) and isinstance(cases[0], dict)
        cases[0][mutation] = "sha256:" + "0" * 64
    _write_manifest(output, manifest)

    with pytest.raises((ReplaySerializationError, ValueError), match=message):
        load_replay_bundle(output, binding)


def test_load_rejects_incomplete_extra_and_duplicate_manifest_fields(
    tmp_path: Path,
) -> None:
    binding = _binding()
    output = tmp_path / "strict-fields"
    save_replay_bundle(_bundle(binding), output, binding)
    original = _manifest(output)

    missing = dict(original)
    del missing["case_count"]
    _write_manifest(output, missing)
    with pytest.raises(ReplaySerializationError, match="missing case_count"):
        load_replay_bundle(output, binding)

    extra = dict(original)
    extra["raw_simulator_state"] = [1, 2, 3]
    _write_manifest(output, extra)
    with pytest.raises(
        ReplaySerializationError, match="unexpected raw_simulator_state"
    ):
        load_replay_bundle(output, binding)

    text = json.dumps(original, separators=(",", ":"))
    (output / MANIFEST_NAME).write_text(
        '{"format":"duplicate",' + text[1:], encoding="utf-8"
    )
    with pytest.raises(ReplaySerializationError, match="duplicate field"):
        load_replay_bundle(output, binding)


def test_load_rejects_action_reference_path_injection(tmp_path: Path) -> None:
    binding = _binding()
    output = tmp_path / "path-injection"
    save_replay_bundle(_bundle(binding), output, binding)
    manifest = _manifest(output)
    cases = manifest["replay_cases"]
    assert isinstance(cases, list) and isinstance(cases[0], dict)
    reference = cases[0]["original_action_reference"]
    assert isinstance(reference, dict)
    reference["path"] = "../../outside.npy"
    _write_manifest(output, manifest)

    with pytest.raises(ReplaySerializationError, match="unexpected path"):
        load_replay_bundle(output, binding)


@pytest.mark.parametrize(
    ("field", "value"),
    [("serialization_version", 999), ("schema_version", "99.0")],
)
def test_load_rejects_unsupported_versions(
    tmp_path: Path, field: str, value: object
) -> None:
    binding = _binding()
    output = tmp_path / field
    save_replay_bundle(_bundle(binding), output, binding)
    manifest = _manifest(output)
    manifest[field] = value
    _write_manifest(output, manifest)

    with pytest.raises(UnsupportedReplaySerializationVersionError):
        load_replay_bundle(output, binding)


def test_load_rejects_unsupported_case_and_reference_versions(tmp_path: Path) -> None:
    binding = _binding()
    output = tmp_path / "nested-version"
    save_replay_bundle(_bundle(binding), output, binding)
    manifest = _manifest(output)
    cases = manifest["replay_cases"]
    assert isinstance(cases, list) and isinstance(cases[0], dict)
    cases[0]["state_reference"]["schema_version"] = "99.0"
    _write_manifest(output, manifest)

    with pytest.raises(UnsupportedReplaySerializationVersionError):
        load_replay_bundle(output, binding)


def test_nonempty_destination_is_rejected_without_modification(tmp_path: Path) -> None:
    binding = _binding()
    output = tmp_path / "bundle"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(ReplaySerializationError, match="absent or empty"):
        save_replay_bundle(_bundle(binding), output, binding)

    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_load_rejects_extra_inventory_and_pickle_like_payload(tmp_path: Path) -> None:
    binding = _binding()
    output = tmp_path / "extra"
    save_replay_bundle(_bundle(binding), output, binding)
    (output / "state.pkl").write_bytes(b"cos\nsystem\n(S'unsafe'\ntR.")

    with pytest.raises(ReplaySerializationError, match="extras"):
        load_replay_bundle(output, binding)


def test_load_revalidates_current_source_content(tmp_path: Path) -> None:
    binding = _binding()
    output = tmp_path / "source-change"
    save_replay_bundle(_bundle(binding), output, binding)
    episodes = list(binding.episodes)
    first = episodes[0]
    candidate = first.candidates[0]
    changed_candidate = replace(
        candidate,
        action=replace(
            candidate.action,
            actions=candidate.action.actions + np.float32(0.5),
        ),
    )
    episodes[0] = replace(first, candidates=(changed_candidate, *first.candidates[1:]))
    changed_binding = ReplaySourceBinding.from_datasets(
        tuple(episodes), binding.corruption_dataset
    )

    with pytest.raises((ReplaySerializationError, ValueError)):
        load_replay_bundle(output, changed_binding)


def test_load_rejects_hardlinked_manifest(tmp_path: Path) -> None:
    binding = _binding()
    output = tmp_path / "hardlink"
    external = tmp_path / "external.json"
    save_replay_bundle(_bundle(binding), output, binding)
    manifest = output / MANIFEST_NAME
    shutil.copy2(manifest, external)
    manifest.unlink()
    os.link(external, manifest)

    with pytest.raises(ReplaySerializationError, match="hard links"):
        load_replay_bundle(output, binding)


def test_load_rejects_symbolic_link_root(tmp_path: Path) -> None:
    binding = _binding()
    output = tmp_path / "bundle"
    alias = tmp_path / "alias"
    save_replay_bundle(_bundle(binding), output, binding)
    try:
        alias.symlink_to(output, target_is_directory=True)
    except OSError:
        pytest.skip("directory symbolic links are unavailable")

    with pytest.raises(ReplaySerializationError, match="unsafe bundle"):
        load_replay_bundle(alias, binding)


def test_bundle_actions_remain_immutable_after_reload(tmp_path: Path) -> None:
    binding = _binding()
    output = tmp_path / "immutable"
    save_replay_bundle(_bundle(binding), output, binding)
    loaded = load_replay_bundle(output, binding)

    with pytest.raises(ValueError):
        loaded.replay_cases[0].original_action.actions[0, 0] = 1.0
    with pytest.raises(ValueError):
        loaded.replay_cases[0].transformed_action.actions[0, 0] = 1.0
    binding.assert_unchanged()

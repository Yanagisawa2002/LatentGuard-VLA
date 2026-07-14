from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

import latentguard.corruptions.serialization as serialization_module
from latentguard.corruptions.layout import ActionField, ActionLayout, ActionSemantic
from latentguard.corruptions.models import (
    CorruptedActionProposal,
    CorruptedActionProposalError,
    compute_proposal_identifier,
)
from latentguard.corruptions.serialization import (
    CORRUPTION_DATASET_SCHEMA_VERSION,
    MANIFEST_NAME,
    CorruptionDataset,
    CorruptionSerializationError,
    UnsupportedCorruptionSerializationVersionError,
    load_corruption_dataset,
    save_corruption_dataset,
)
from latentguard.models import ActionChunk


def _layout() -> ActionLayout:
    return ActionLayout(
        action_dim=5,
        fields=(
            ActionField(
                name="delta_xyz",
                indices=(1, 3),
                semantic=ActionSemantic.TRANSLATION,
                units="m",
                description="Sparse translation controls",
                metadata={"scale": 0.25},
            ),
            ActionField(
                name="claw",
                indices=(4,),
                semantic=ActionSemantic.GRIPPER,
                metadata={"binary": True},
            ),
        ),
        description="Five-dimensional test layout with unused indices",
        metadata={"robot": "synthetic", "revision": 2},
    )


def _proposal(
    *,
    proposal_id: str | None = None,
    ordinal: int = 0,
    dtype: str = "float32",
) -> CorruptedActionProposal:
    values = np.arange(15, dtype=dtype).reshape(3, 5)
    bias = (1, -2) if np.issubdtype(values.dtype, np.integer) else (0.125, -0.25)
    parameters = {
        "target_indices": (1, 3),
        "bias": bias,
    }
    identifier = proposal_id or compute_proposal_identifier(
        source_episode_id="episode-0004",
        source_candidate_id="candidate-0004-02",
        corruption_name="constant_bias",
        resolved_parameters=parameters,
        seed=314159,
        generation_ordinal=ordinal,
    )
    return CorruptedActionProposal(
        proposal_id=identifier,
        source_episode_id="episode-0004",
        source_candidate_id="candidate-0004-02",
        source_policy_id="policy-synthetic-v1",
        source_task_id="pick-cube",
        split_group_id="episode-0004",
        transformed_action=ActionChunk(
            actions=values,
            coordinate_frame="declared-layout-frame",
            control_period_s=0.05,
        ),
        corruption_type="constant_bias",
        resolved_parameters=parameters,
        seed=314159,
        generation_ordinal=ordinal,
        notes="unlabeled proposal",
    )


def _dataset(*proposals: CorruptedActionProposal) -> CorruptionDataset:
    selected = proposals or (_proposal(),)
    return CorruptionDataset(
        source_dataset_id="sha256:0123456789abcdef",
        action_layout=_layout(),
        proposals=selected,
    )


def _manifest(bundle: Path) -> dict[str, object]:
    return json.loads((bundle / MANIFEST_NAME).read_text(encoding="utf-8"))


def _write_manifest(bundle: Path, manifest: dict[str, object]) -> None:
    (bundle / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _first_array_reference(manifest: dict[str, object]) -> dict[str, object]:
    proposals = manifest["proposals"]
    assert isinstance(proposals, list)
    proposal = proposals[0]
    assert isinstance(proposal, dict)
    action = proposal["transformed_action"]
    assert isinstance(action, dict)
    reference = action["actions"]
    assert isinstance(reference, dict)
    return reference


def test_corruption_dataset_round_trip_preserves_actions_and_provenance(
    tmp_path: Path,
) -> None:
    source = _dataset()
    bundle = tmp_path / "corruptions"

    manifest_path = save_corruption_dataset(source, bundle)
    loaded = load_corruption_dataset(bundle)

    assert manifest_path == bundle / MANIFEST_NAME
    assert loaded.source_dataset_id == source.source_dataset_id
    assert loaded.schema_version == CORRUPTION_DATASET_SCHEMA_VERSION
    assert len(loaded.proposals) == 1
    actual = loaded.proposals[0]
    expected = source.proposals[0]
    assert actual.proposal_id == expected.proposal_id
    assert actual.source_episode_id == expected.source_episode_id
    assert actual.source_candidate_id == expected.source_candidate_id
    assert actual.source_policy_id == expected.source_policy_id
    assert actual.source_task_id == expected.source_task_id
    assert actual.split_group_id == expected.split_group_id
    assert actual.corruption_type == expected.corruption_type
    assert actual.resolved_parameters == expected.resolved_parameters
    assert actual.seed == expected.seed
    assert actual.generation_ordinal == expected.generation_ordinal
    assert actual.notes == expected.notes
    np.testing.assert_array_equal(
        actual.transformed_action.actions, expected.transformed_action.actions
    )
    assert actual.transformed_action.actions.dtype == np.dtype(np.float32)
    assert actual.transformed_action.actions.shape == (3, 5)
    assert not actual.transformed_action.actions.flags.writeable


@pytest.mark.parametrize("dtype", ["float32", "float64", "int16"])
def test_round_trip_preserves_numeric_dtype_exactly(tmp_path: Path, dtype: str) -> None:
    source = _dataset(_proposal(dtype=dtype))
    bundle = tmp_path / dtype

    save_corruption_dataset(source, bundle)
    loaded = load_corruption_dataset(bundle)

    assert loaded.proposals[0].transformed_action.actions.dtype == np.dtype(dtype)


def test_round_trip_preserves_complete_action_layout(tmp_path: Path) -> None:
    source = _dataset()
    bundle = tmp_path / "corruptions"
    save_corruption_dataset(source, bundle)

    loaded = load_corruption_dataset(bundle)

    assert loaded.action_layout.action_dim == 5
    assert loaded.action_layout.description == source.action_layout.description
    assert loaded.action_layout.metadata == source.action_layout.metadata
    assert len(loaded.action_layout.fields) == 2
    for actual, expected in zip(
        loaded.action_layout.fields, source.action_layout.fields, strict=True
    ):
        assert actual.name == expected.name
        assert actual.indices == expected.indices
        assert actual.semantic is expected.semantic
        assert actual.units == expected.units
        assert actual.description == expected.description
        assert actual.metadata == expected.metadata


def test_manifest_contains_only_proposals_not_source_observations(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "corruptions"
    save_corruption_dataset(_dataset(), bundle)

    text = (bundle / MANIFEST_NAME).read_text(encoding="utf-8")

    assert '"source_dataset_id"' in text
    assert '"transformed_action"' in text
    assert '"rgb"' not in text
    assert '"cameras"' not in text
    assert '"observations"' not in text
    assert tuple((bundle / "arrays").glob("*.npy")) == (
        bundle / "arrays" / "000000.npy",
    )


def test_zero_proposal_dataset_is_explicitly_supported(tmp_path: Path) -> None:
    source = CorruptionDataset(
        source_dataset_id="sha256:empty",
        action_layout=_layout(),
        proposals=(),
    )
    bundle = tmp_path / "empty-corruptions"

    save_corruption_dataset(source, bundle)
    loaded = load_corruption_dataset(bundle)

    assert loaded.proposals == ()
    assert list((bundle / "arrays").iterdir()) == []
    manifest = _manifest(bundle)
    assert manifest["proposal_count"] == 0
    assert manifest["array_count"] == 0


@pytest.mark.parametrize("identifier", ["", "   ", "../source", "C:\\source"])
def test_source_dataset_identifier_must_be_nonempty_and_path_independent(
    identifier: str,
) -> None:
    with pytest.raises(CorruptionSerializationError, match="source_dataset_id"):
        CorruptionDataset(
            source_dataset_id=identifier,
            action_layout=_layout(),
            proposals=(),
        )


def test_proposal_rejects_forged_deterministic_identifier() -> None:
    with pytest.raises(CorruptedActionProposalError, match="identifier mismatch"):
        _proposal(proposal_id="forged")


@pytest.mark.parametrize("ordinal", [0, 2, 7])
def test_dataset_requires_contiguous_generation_ordinals(ordinal: int) -> None:
    first = _proposal(ordinal=0)
    second = _proposal(ordinal=ordinal)

    with pytest.raises(
        CorruptionSerializationError, match="expected stable output position 1"
    ):
        _dataset(first, second)


def test_dataset_rejects_action_dimension_mismatch() -> None:
    parameters = {"start_step": 0, "end_step": 1, "target_indices": (0, 1, 2, 3)}
    proposal = CorruptedActionProposal(
        proposal_id=compute_proposal_identifier(
            source_episode_id="episode",
            source_candidate_id="candidate",
            corruption_name="segment_zeroing",
            resolved_parameters=parameters,
            seed=1,
            generation_ordinal=0,
        ),
        source_episode_id="episode",
        source_candidate_id="candidate",
        source_policy_id="policy",
        source_task_id="task",
        split_group_id="episode",
        transformed_action=ActionChunk(
            actions=np.ones((2, 4), dtype=np.float32),
            coordinate_frame="frame",
            control_period_s=0.1,
        ),
        corruption_type="segment_zeroing",
        resolved_parameters=parameters,
        seed=1,
        generation_ordinal=0,
    )

    with pytest.raises(CorruptionSerializationError, match="does not match layout"):
        _dataset(proposal)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("serialization_version", 999),
        ("dataset_schema_version", "99.0"),
    ],
)
def test_load_rejects_unsupported_dataset_versions(
    tmp_path: Path, field: str, value: object
) -> None:
    bundle = tmp_path / field
    save_corruption_dataset(_dataset(), bundle)
    manifest = _manifest(bundle)
    manifest[field] = value
    _write_manifest(bundle, manifest)

    with pytest.raises(UnsupportedCorruptionSerializationVersionError):
        load_corruption_dataset(bundle)


def test_load_rejects_unsupported_proposal_version(tmp_path: Path) -> None:
    bundle = tmp_path / "proposal-version"
    save_corruption_dataset(_dataset(), bundle)
    manifest = _manifest(bundle)
    proposals = manifest["proposals"]
    assert isinstance(proposals, list) and isinstance(proposals[0], dict)
    proposals[0]["schema_version"] = "99.0"
    _write_manifest(bundle, manifest)

    with pytest.raises(UnsupportedCorruptionSerializationVersionError):
        load_corruption_dataset(bundle)


def test_load_rejects_unknown_manifest_field(tmp_path: Path) -> None:
    bundle = tmp_path / "unknown-field"
    save_corruption_dataset(_dataset(), bundle)
    manifest = _manifest(bundle)
    manifest["outcome"] = {"success": False}
    _write_manifest(bundle, manifest)

    with pytest.raises(CorruptionSerializationError, match="unexpected outcome"):
        load_corruption_dataset(bundle)


def test_load_rejects_fabricated_outcome_field_on_proposal(tmp_path: Path) -> None:
    bundle = tmp_path / "fabricated-outcome"
    save_corruption_dataset(_dataset(), bundle)
    manifest = _manifest(bundle)
    proposals = manifest["proposals"]
    assert isinstance(proposals, list) and isinstance(proposals[0], dict)
    proposals[0]["outcome"] = {"success": False, "unsafe": True}
    _write_manifest(bundle, manifest)

    with pytest.raises(CorruptionSerializationError, match="unexpected outcome"):
        load_corruption_dataset(bundle)


def test_load_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    bundle = tmp_path / "duplicate-key"
    save_corruption_dataset(_dataset(), bundle)
    manifest_path = bundle / MANIFEST_NAME
    original = manifest_path.read_text(encoding="utf-8").lstrip()
    manifest_path.write_text('{"format":"duplicate",' + original[1:], encoding="utf-8")

    with pytest.raises(CorruptionSerializationError, match="duplicate field"):
        load_corruption_dataset(bundle)


def test_load_validates_every_proposal(tmp_path: Path) -> None:
    bundle = tmp_path / "invalid-provenance"
    save_corruption_dataset(_dataset(), bundle)
    manifest = _manifest(bundle)
    proposals = manifest["proposals"]
    assert isinstance(proposals, list) and isinstance(proposals[0], dict)
    proposals[0]["source_episode_id"] = ""
    _write_manifest(bundle, manifest)

    with pytest.raises(CorruptionSerializationError, match="source_episode_id"):
        load_corruption_dataset(bundle)


def test_load_rejects_tampered_proposal_identifier(tmp_path: Path) -> None:
    bundle = tmp_path / "tampered-identifier"
    save_corruption_dataset(_dataset(), bundle)
    manifest = _manifest(bundle)
    proposals = manifest["proposals"]
    assert isinstance(proposals, list) and isinstance(proposals[0], dict)
    proposals[0]["proposal_id"] = "cap-sha256-" + "0" * 64
    _write_manifest(bundle, manifest)

    with pytest.raises(CorruptionSerializationError, match="identifier mismatch"):
        load_corruption_dataset(bundle)


def test_load_rejects_unknown_corruption_type_with_matching_id(tmp_path: Path) -> None:
    bundle = tmp_path / "unknown-corruption"
    save_corruption_dataset(_dataset(), bundle)
    manifest = _manifest(bundle)
    proposals = manifest["proposals"]
    assert isinstance(proposals, list) and isinstance(proposals[0], dict)
    proposal = proposals[0]
    parameters = {"donor_episode_id": "forbidden-donor"}
    proposal["corruption_type"] = "cross_episode_swap"
    proposal["resolved_parameters"] = parameters
    proposal["proposal_id"] = compute_proposal_identifier(
        source_episode_id=str(proposal["source_episode_id"]),
        source_candidate_id=str(proposal["source_candidate_id"]),
        corruption_name="cross_episode_swap",
        resolved_parameters=parameters,
        seed=int(proposal["seed"]),
        generation_ordinal=int(proposal["generation_ordinal"]),
    )
    _write_manifest(bundle, manifest)

    with pytest.raises(CorruptionSerializationError, match="unknown corruption"):
        load_corruption_dataset(bundle)


def test_load_rejects_unresolved_parameter_shape_with_matching_id(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "invalid-parameters"
    save_corruption_dataset(_dataset(), bundle)
    manifest = _manifest(bundle)
    proposals = manifest["proposals"]
    assert isinstance(proposals, list) and isinstance(proposals[0], dict)
    proposal = proposals[0]
    parameters = {
        "target_indices": (1, 3),
        "bias": (0.125, -0.25),
        "donor_candidate_id": "forbidden-donor",
    }
    proposal["resolved_parameters"] = parameters
    proposal["proposal_id"] = compute_proposal_identifier(
        source_episode_id=str(proposal["source_episode_id"]),
        source_candidate_id=str(proposal["source_candidate_id"]),
        corruption_name=str(proposal["corruption_type"]),
        resolved_parameters=parameters,
        seed=int(proposal["seed"]),
        generation_ordinal=int(proposal["generation_ordinal"]),
    )
    _write_manifest(bundle, manifest)

    with pytest.raises(CorruptionSerializationError, match="unknown donor_candidate"):
        load_corruption_dataset(bundle)


def test_load_wraps_oversized_numeric_values(tmp_path: Path) -> None:
    bundle = tmp_path / "oversized-number"
    save_corruption_dataset(_dataset(), bundle)
    manifest = _manifest(bundle)
    proposals = manifest["proposals"]
    assert isinstance(proposals, list) and isinstance(proposals[0], dict)
    action = proposals[0]["transformed_action"]
    assert isinstance(action, dict)
    action["control_period_s"] = 10**400
    _write_manifest(bundle, manifest)

    with pytest.raises(CorruptionSerializationError, match="floating-point range"):
        load_corruption_dataset(bundle)


def test_load_rejects_path_traversal_array_reference(tmp_path: Path) -> None:
    bundle = tmp_path / "path-traversal"
    save_corruption_dataset(_dataset(), bundle)
    manifest = _manifest(bundle)
    _first_array_reference(manifest)["path"] = "../outside.npy"
    _write_manifest(bundle, manifest)

    with pytest.raises(CorruptionSerializationError, match="unsafe path"):
        load_corruption_dataset(bundle)


def test_load_rejects_missing_array_as_incomplete_output(tmp_path: Path) -> None:
    bundle = tmp_path / "missing-array"
    save_corruption_dataset(_dataset(), bundle)
    (bundle / "arrays" / "000000.npy").unlink()

    with pytest.raises(CorruptionSerializationError, match="missing|inventory"):
        load_corruption_dataset(bundle)


def test_load_rejects_extra_array(tmp_path: Path) -> None:
    bundle = tmp_path / "extra-array"
    save_corruption_dataset(_dataset(), bundle)
    shutil.copy2(bundle / "arrays" / "000000.npy", bundle / "arrays" / "extra.npy")

    with pytest.raises(CorruptionSerializationError, match="exactly match"):
        load_corruption_dataset(bundle)


def test_load_rejects_missing_arrays_directory(tmp_path: Path) -> None:
    source = tmp_path / "complete"
    incomplete = tmp_path / "incomplete"
    save_corruption_dataset(_dataset(), source)
    incomplete.mkdir()
    shutil.copy2(source / MANIFEST_NAME, incomplete / MANIFEST_NAME)

    with pytest.raises(CorruptionSerializationError, match="incomplete"):
        load_corruption_dataset(incomplete)


def test_load_rejects_hard_linked_array(tmp_path: Path) -> None:
    bundle = tmp_path / "hard-link"
    external = tmp_path / "external.npy"
    save_corruption_dataset(_dataset(), bundle)
    array_path = bundle / "arrays" / "000000.npy"
    shutil.copy2(array_path, external)
    array_path.unlink()
    os.link(external, array_path)

    with pytest.raises(CorruptionSerializationError, match="hard link"):
        load_corruption_dataset(bundle)


def test_load_rejects_hard_linked_manifest(tmp_path: Path) -> None:
    bundle = tmp_path / "hard-linked-manifest"
    external = tmp_path / "external-manifest.json"
    save_corruption_dataset(_dataset(), bundle)
    manifest_path = bundle / MANIFEST_NAME
    shutil.copy2(manifest_path, external)
    manifest_path.unlink()
    os.link(external, manifest_path)

    with pytest.raises(CorruptionSerializationError, match="hard links"):
        load_corruption_dataset(bundle)


def test_load_wraps_malformed_npy_as_serialization_error(tmp_path: Path) -> None:
    bundle = tmp_path / "malformed-npy"
    save_corruption_dataset(_dataset(), bundle)
    (bundle / "arrays" / "000000.npy").write_bytes(b"not an NPY file")

    with pytest.raises(CorruptionSerializationError, match="load safely"):
        load_corruption_dataset(bundle)


def test_load_rejects_symbolic_linked_arrays_directory(tmp_path: Path) -> None:
    bundle = tmp_path / "symbolic-link"
    target = tmp_path / "real-arrays"
    save_corruption_dataset(_dataset(), bundle)
    (bundle / "arrays").rename(target)
    try:
        (bundle / "arrays").symlink_to(target, target_is_directory=True)
    except OSError as exc:
        target.rename(bundle / "arrays")
        pytest.skip(f"directory symbolic links unavailable: {exc}")

    with pytest.raises(CorruptionSerializationError, match="unsafe"):
        load_corruption_dataset(bundle)


def test_save_rejects_nonempty_destination_without_modifying_it(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nonempty"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")

    with pytest.raises(CorruptionSerializationError, match="absent or empty"):
        save_corruption_dataset(_dataset(), destination)

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert tuple(destination.iterdir()) == (sentinel,)


def test_save_accepts_and_replaces_empty_destination(tmp_path: Path) -> None:
    destination = tmp_path / "empty"
    destination.mkdir()

    save_corruption_dataset(_dataset(), destination)

    assert load_corruption_dataset(destination).source_dataset_id.startswith("sha256:")


def test_failed_transaction_leaves_no_successful_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "failed"

    def fail_save(*args: object, **kwargs: object) -> NoReturn:
        raise OSError("injected array failure")

    monkeypatch.setattr(serialization_module.np, "save", fail_save)

    with pytest.raises(CorruptionSerializationError, match="transactionally"):
        save_corruption_dataset(_dataset(), destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".failed.staging-*")) == []


def test_cleanup_failure_does_not_mask_primary_save_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "cleanup-failed"
    original_rmtree = serialization_module.shutil.rmtree

    def fail_save(*args: object, **kwargs: object) -> NoReturn:
        raise OSError("primary write failure")

    def fail_cleanup(*args: object, **kwargs: object) -> NoReturn:
        raise OSError("secondary cleanup failure")

    with monkeypatch.context() as context:
        context.setattr(serialization_module.np, "save", fail_save)
        context.setattr(serialization_module.shutil, "rmtree", fail_cleanup)
        with pytest.raises(
            CorruptionSerializationError, match="primary write failure"
        ) as exc:
            save_corruption_dataset(_dataset(), destination)

    assert any("secondary cleanup failure" in note for note in exc.value.__notes__)
    staging = list(tmp_path.glob(".cleanup-failed.staging-*"))
    assert len(staging) == 1
    original_rmtree(staging[0])


def test_load_rejects_tampered_dtype_and_shape(tmp_path: Path) -> None:
    bundle = tmp_path / "tampered-array"
    save_corruption_dataset(_dataset(), bundle)
    manifest = _manifest(bundle)
    reference = _first_array_reference(manifest)
    reference["dtype"] = np.dtype(np.float64).str
    reference["shape"] = [15]
    _write_manifest(bundle, manifest)

    with pytest.raises(CorruptionSerializationError, match="dtype|shape"):
        load_corruption_dataset(bundle)

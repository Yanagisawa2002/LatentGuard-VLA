"""Dataset gate and source-group leakage tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from latentguard.world_model.data_schema import DatasetSplit, WorldModelSchemaError
from latentguard.world_model.manifest import build_dataset_manifest

from .conftest import sample


def test_gate_rejects_small_dataset() -> None:
    """A mechanically valid sample is not a formal dataset."""

    manifest = build_dataset_manifest((sample(),), minimum_valid_samples=1000)
    assert not manifest.gate.authorized
    assert "minimum_sample_count" in manifest.gate.reasons
    assert "required_splits_present" in manifest.gate.reasons


def test_group_leakage_is_rejected() -> None:
    """One source trajectory cannot cross split assignments."""

    with pytest.raises(WorldModelSchemaError, match="cross dataset splits"):
        build_dataset_manifest(
            (
                sample(group="shared", split=DatasetSplit.TRAIN),
                replace(
                    sample(group="shared", split=DatasetSplit.VALIDATION),
                    candidate_id="candidate-2",
                ),
            )
        )

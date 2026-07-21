"""WM-v0 sample serialization and alignment tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from latentguard.world_model.data_schema import WorldModelSchemaError
from latentguard.world_model.serialization import (
    read_world_model_sample,
    write_world_model_sample,
)

from .conftest import sample


def test_schema_round_trip(tmp_path: object) -> None:
    """JSON/NPZ round trips without pickle or dtype drift."""

    reference = write_world_model_sample(tmp_path, sample())  # type: ignore[arg-type]
    loaded = read_world_model_sample(tmp_path, reference.sample_id)  # type: ignore[arg-type]
    assert loaded.metadata_mapping() == sample().metadata_mapping()
    for name, value in sample().array_mapping().items():
        assert np.array_equal(loaded.array_mapping()[name], value)


def test_missing_future_observation_is_rejected() -> None:
    """No terminal label can substitute for a real future RGB boundary."""

    with pytest.raises(WorldModelSchemaError, match="future_observations"):
        replace(
            sample(),
            future_observations=np.zeros((0, 2, 4, 4, 3), dtype=np.uint8),
        )


def test_future_alignment_is_rejected() -> None:
    """Future proprioception must share the RGB prediction horizon."""

    with pytest.raises(WorldModelSchemaError, match="future proprio"):
        replace(sample(), future_proprio=np.zeros((1, 5), dtype=np.float32))


def test_action_mask_must_be_contiguous_prefix() -> None:
    """Action gaps cannot be silently repaired by the loader."""

    with pytest.raises(WorldModelSchemaError, match="contiguous prefix"):
        replace(
            sample(),
            action_mask=np.asarray([True, False, True, False], dtype=np.bool_),
        )

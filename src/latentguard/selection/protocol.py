"""Strict checked-in M3C smoke/full protocol configuration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from latentguard.integrations.maniskill_pickcube.verifier_state import (
    PICKCUBE_VERIFIER_STATE_SEMANTIC,
)
from latentguard.replay.identity import canonical_json_bytes
from latentguard.selection.checkpoint_bundle import (
    ACCEPTED_M3B_SPLIT_DIGEST,
    EXPECTED_FIVE_SEEDS,
)
from latentguard.training.dataset import ACCEPTED_M3A_DATASET_DIGEST

ACCEPTED_M3B_PREPROCESSING_DIGEST = (
    "sha256:ee3353b81609d489a0cc654ded70e2db777db863d9c12c4b9e4cca96f51b6de6"
)
ACCEPTED_ACTION_MAGNITUDE_DIGEST = (
    "sha256:370d417bc80d68dcb60076a4bcf2b02f27d8f4e7ba82b7e7cb2946e342f7cf53"
)


class SelectionProtocolError(ValueError):
    """Raised when the checked-in M3C execution protocol drifts."""


def _fail(context: str, reason: str) -> NoReturn:
    raise SelectionProtocolError(f"{context}: {reason}")


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(context, "expected lowercase sha256 digest")
    return value


def _reject_duplicate_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("selection protocol", f"duplicate field {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class SelectionProtocolV1:
    """Frozen source ranges, dimensions, statistics, and research roles."""

    accepted_dataset_digest: str
    accepted_split_digest: str
    accepted_preprocessing_digest: str
    action_magnitude_baseline_digest: str
    ensemble_seeds: tuple[int, ...]
    state_semantic: str
    state_dimension: int
    state_restoration_component_count: int
    state_restoration_semantic: str
    state_restoration_tolerance: float
    action_horizon: int
    action_dimension: int
    ensemble_semantic: str
    primary_selector: str
    efficiency_challenger: str
    bootstrap_seed: int
    bootstrap_replicates: int
    confidence_level: float
    coverage_targets: tuple[float, ...]
    smoke_starting_seed: int
    smoke_requested_success_count: int
    smoke_maximum_attempts: int
    full_starting_seed: int
    full_requested_success_count: int
    full_maximum_attempts: int
    anchors_per_trajectory: int
    outcomes_available_during_selection: bool
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Reject any drift from fixed M3A/M3B identities or M3C scope."""

        for name in (
            "accepted_dataset_digest",
            "accepted_split_digest",
            "accepted_preprocessing_digest",
            "action_magnitude_baseline_digest",
        ):
            _digest(getattr(self, name), f"SelectionProtocolV1.{name}")
        if (
            self.accepted_dataset_digest != ACCEPTED_M3A_DATASET_DIGEST
            or self.accepted_split_digest != ACCEPTED_M3B_SPLIT_DIGEST
            or self.accepted_preprocessing_digest != ACCEPTED_M3B_PREPROCESSING_DIGEST
            or self.action_magnitude_baseline_digest != ACCEPTED_ACTION_MAGNITUDE_DIGEST
        ):
            _fail("SelectionProtocolV1", "accepted M3A/M3B identity changed")
        if self.ensemble_seeds != EXPECTED_FIVE_SEEDS:
            _fail(
                "SelectionProtocolV1.ensemble_seeds", "expected seeds zero through four"
            )
        if (
            self.state_semantic != PICKCUBE_VERIFIER_STATE_SEMANTIC
            or self.state_dimension != 38
            or self.state_restoration_component_count != 70
            or self.state_restoration_semantic != "tolerance_verified_full_state_v1"
            or self.state_restoration_tolerance != 1e-6
            or self.action_horizon != 16
            or self.action_dimension != 8
        ):
            _fail("SelectionProtocolV1", "fixed state or action contract changed")
        if self.ensemble_semantic != (
            "arithmetic_mean_five_validation_calibrated_failure_probabilities_v1"
        ):
            _fail("SelectionProtocolV1.ensemble_semantic", "semantic changed")
        if self.primary_selector != "temporal_ensemble_v1" or (
            self.efficiency_challenger != "state_action_mlp_ensemble_v1"
        ):
            _fail("SelectionProtocolV1", "predeclared selector roles changed")
        if self.bootstrap_replicates < 2000:
            _fail("SelectionProtocolV1.bootstrap_replicates", "must be at least 2000")
        if (
            type(self.bootstrap_seed) is not int
            or self.bootstrap_seed < 0
            or not math.isfinite(self.confidence_level)
            or not 0.0 < self.confidence_level < 1.0
        ):
            _fail("SelectionProtocolV1", "invalid bootstrap configuration")
        if self.coverage_targets != (0.9, 0.8, 0.7, 0.5):
            _fail("SelectionProtocolV1.coverage_targets", "fixed policies changed")
        integer_fields = (
            "smoke_starting_seed",
            "smoke_requested_success_count",
            "smoke_maximum_attempts",
            "full_starting_seed",
            "full_requested_success_count",
            "full_maximum_attempts",
            "anchors_per_trajectory",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) <= 0
            for name in integer_fields
        ):
            _fail("SelectionProtocolV1", "source range fields must be positive")
        if (
            self.smoke_requested_success_count != 6
            or self.full_requested_success_count != 60
            or self.anchors_per_trajectory != 6
            or self.smoke_starting_seed == self.full_starting_seed
        ):
            _fail("SelectionProtocolV1", "smoke/full source contract changed")
        smoke_seeds = range(
            self.smoke_starting_seed,
            self.smoke_starting_seed + self.smoke_maximum_attempts,
        )
        full_start = self.full_starting_seed
        full_stop = self.full_starting_seed + self.full_maximum_attempts
        if any(full_start <= seed < full_stop for seed in smoke_seeds):
            _fail("SelectionProtocolV1", "smoke/full seed ranges overlap")
        if self.outcomes_available_during_selection is not False:
            _fail("SelectionProtocolV1", "selection outcomes must be unavailable")
        if self.schema_version != "1.0":
            _fail("SelectionProtocolV1.schema_version", "unsupported version")

    @property
    def content_digest(self) -> str:
        """Return the exact path-independent protocol identity."""

        encoded = canonical_json_bytes(self.as_mapping())
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def as_mapping(self) -> dict[str, object]:
        """Return all strict JSON-native protocol fields."""

        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in (
                ("accepted_dataset_digest", self.accepted_dataset_digest),
                ("accepted_split_digest", self.accepted_split_digest),
                ("accepted_preprocessing_digest", self.accepted_preprocessing_digest),
                (
                    "action_magnitude_baseline_digest",
                    self.action_magnitude_baseline_digest,
                ),
                ("ensemble_seeds", self.ensemble_seeds),
                ("state_semantic", self.state_semantic),
                ("state_dimension", self.state_dimension),
                (
                    "state_restoration_component_count",
                    self.state_restoration_component_count,
                ),
                ("state_restoration_semantic", self.state_restoration_semantic),
                ("state_restoration_tolerance", self.state_restoration_tolerance),
                ("action_horizon", self.action_horizon),
                ("action_dimension", self.action_dimension),
                ("ensemble_semantic", self.ensemble_semantic),
                ("primary_selector", self.primary_selector),
                ("efficiency_challenger", self.efficiency_challenger),
                ("bootstrap_seed", self.bootstrap_seed),
                ("bootstrap_replicates", self.bootstrap_replicates),
                ("confidence_level", self.confidence_level),
                ("coverage_targets", self.coverage_targets),
                ("smoke_starting_seed", self.smoke_starting_seed),
                ("smoke_requested_success_count", self.smoke_requested_success_count),
                ("smoke_maximum_attempts", self.smoke_maximum_attempts),
                ("full_starting_seed", self.full_starting_seed),
                ("full_requested_success_count", self.full_requested_success_count),
                ("full_maximum_attempts", self.full_maximum_attempts),
                ("anchors_per_trajectory", self.anchors_per_trajectory),
                (
                    "outcomes_available_during_selection",
                    self.outcomes_available_during_selection,
                ),
                ("schema_version", self.schema_version),
            )
        }


def load_selection_protocol(path: Path) -> SelectionProtocolV1:
    """Load an exact duplicate-free regular JSON protocol file."""

    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1:
        _fail("selection protocol", "expected one unlinked regular file")
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectionProtocolError(f"selection protocol: {exc}") from exc
    if not isinstance(raw, Mapping):
        _fail("selection protocol", "expected object")
    expected = set(SelectionProtocolV1.__dataclass_fields__)
    if set(raw) != expected:
        _fail("selection protocol", "unexpected or missing fields")
    values = dict(raw)
    for name in ("ensemble_seeds", "coverage_targets"):
        if not isinstance(values[name], list):
            _fail(f"selection protocol.{name}", "expected list")
        values[name] = tuple(values[name])
    return SelectionProtocolV1(**cast(dict[str, object], values))  # type: ignore[arg-type]


__all__ = [
    "ACCEPTED_ACTION_MAGNITUDE_DIGEST",
    "ACCEPTED_M3B_PREPROCESSING_DIGEST",
    "SelectionProtocolError",
    "SelectionProtocolV1",
    "load_selection_protocol",
]

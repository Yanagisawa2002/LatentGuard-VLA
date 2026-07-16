"""Strict checked-in configuration for M3C candidate pools."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn, cast

from latentguard.corruptions.base import ActionCorruption
from latentguard.corruptions.registry import create_corruption
from latentguard.corruptions.transforms import WindowScopedCorruption
from latentguard.replay.base import ReplayValidationError
from latentguard.replay.identity import canonical_json_bytes, canonical_json_value
from latentguard.selection.models import (
    ACTION_DIMENSION,
    ACTION_HORIZON,
    CANDIDATE_COUNT,
    SELECTION_SCHEMA_VERSION,
    CandidateDistribution,
)

CANDIDATE_POOL_SEMANTIC = "blind_eight_candidate_pool_v1"


class CandidatePoolConfigurationError(ValueError):
    """Raised when the fixed M3C candidate configuration is malformed."""


def _fail(context: str, reason: str) -> NoReturn:
    raise CandidatePoolConfigurationError(f"{context}: {reason}")


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected canonical non-empty text")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(context, f"expected integer >= {minimum}")
    return value


def _freeze_json(value: object, context: str) -> object:
    try:
        canonical = canonical_json_value(value, context=context)
    except ReplayValidationError as exc:
        raise CandidatePoolConfigurationError(str(exc)) from exc
    if isinstance(canonical, dict):
        return MappingProxyType(
            {
                key: _freeze_json(item, f"{context}.{key}")
                for key, item in canonical.items()
            }
        )
    if isinstance(canonical, list):
        return tuple(
            _freeze_json(item, f"{context}[{index}]")
            for index, item in enumerate(canonical)
        )
    return canonical


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class CandidateDefinitionV1:
    """One fixed corruption slot and its reporting-only distribution tag."""

    configuration_ordinal: int
    distribution: CandidateDistribution
    corruption_type: str
    parameters: Mapping[str, object]
    window_start: int
    window_end: int
    severity_id: str
    schema_version: str = SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze parameters and require a complete 16-step corruption window."""

        ordinal = _integer(
            self.configuration_ordinal,
            "CandidateDefinitionV1.configuration_ordinal",
        )
        if ordinal >= CANDIDATE_COUNT:
            _fail(
                "CandidateDefinitionV1.configuration_ordinal", "must be smaller than 8"
            )
        try:
            distribution = CandidateDistribution(self.distribution)
        except (TypeError, ValueError) as exc:
            raise CandidatePoolConfigurationError(
                "CandidateDefinitionV1.distribution: unsupported value"
            ) from exc
        _text(self.corruption_type, "CandidateDefinitionV1.corruption_type")
        if not isinstance(self.parameters, Mapping) or any(
            not isinstance(key, str) for key in self.parameters
        ):
            _fail("CandidateDefinitionV1.parameters", "expected an object")
        parameters = _freeze_json(
            dict(self.parameters), "CandidateDefinitionV1.parameters"
        )
        if not isinstance(parameters, Mapping):
            _fail("CandidateDefinitionV1.parameters", "expected an object")
        if self.window_start != 0 or self.window_end != ACTION_HORIZON:
            _fail(
                "CandidateDefinitionV1.window",
                "M3C candidates must modify only [0, 16)",
            )
        _text(self.severity_id, "CandidateDefinitionV1.severity_id")
        if self.schema_version != SELECTION_SCHEMA_VERSION:
            _fail("CandidateDefinitionV1.schema_version", "unsupported version")
        try:
            corruption = create_corruption(
                self.corruption_type,
                cast(Mapping[str, object], _plain_json(parameters)),
            )
            WindowScopedCorruption(
                inner=corruption,
                window_start=self.window_start,
                window_end=self.window_end,
                severity_id=self.severity_id,
            )
        except (TypeError, ValueError) as exc:
            raise CandidatePoolConfigurationError(
                f"CandidateDefinitionV1: invalid corruption: {exc}"
            ) from exc
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "parameters", parameters)

    def build_corruption(self) -> ActionCorruption:
        """Instantiate the existing M1 transform with the fixed M3C window."""

        inner = create_corruption(
            self.corruption_type,
            cast(Mapping[str, object], _plain_json(self.parameters)),
        )
        return WindowScopedCorruption(
            inner=inner,
            window_start=self.window_start,
            window_end=self.window_end,
            severity_id=self.severity_id,
        )

    def as_mapping(self) -> dict[str, object]:
        """Return deterministic JSON-native slot content."""

        return {
            "configuration_ordinal": self.configuration_ordinal,
            "corruption_type": self.corruption_type,
            "distribution": self.distribution.value,
            "parameters": _plain_json(self.parameters),
            "schema_version": self.schema_version,
            "severity_id": self.severity_id,
            "window_end": self.window_end,
            "window_start": self.window_start,
        }


@dataclass(frozen=True, slots=True)
class CandidatePoolConfigurationV1:
    """The immutable eight-candidate M3C generation schedule."""

    action_contract_digest: str
    base_seed: int
    candidates: tuple[CandidateDefinitionV1, ...]
    action_dimension: int = ACTION_DIMENSION
    candidate_horizon: int = ACTION_HORIZON
    semantic: str = CANDIDATE_POOL_SEMANTIC
    schema_version: str = SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require exact dimensions, cardinality, order, and 4+4 distribution."""

        from latentguard.selection.models import _digest as validate_digest

        validate_digest(
            self.action_contract_digest,
            "CandidatePoolConfigurationV1.action_contract_digest",
        )
        _integer(self.base_seed, "CandidatePoolConfigurationV1.base_seed")
        if self.base_seed >= 2**64:
            _fail(
                "CandidatePoolConfigurationV1.base_seed", "must be smaller than 2**64"
            )
        if self.action_dimension != ACTION_DIMENSION:
            _fail("CandidatePoolConfigurationV1.action_dimension", "must equal 8")
        if self.candidate_horizon != ACTION_HORIZON:
            _fail("CandidatePoolConfigurationV1.candidate_horizon", "must equal 16")
        if self.semantic != CANDIDATE_POOL_SEMANTIC:
            _fail("CandidatePoolConfigurationV1.semantic", "unsupported semantic")
        definitions = tuple(self.candidates)
        if len(definitions) != CANDIDATE_COUNT or any(
            not isinstance(item, CandidateDefinitionV1) for item in definitions
        ):
            _fail(
                "CandidatePoolConfigurationV1.candidates",
                "expected exactly 8 definitions",
            )
        if tuple(item.configuration_ordinal for item in definitions) != tuple(
            range(CANDIDATE_COUNT)
        ):
            _fail("CandidatePoolConfigurationV1.candidates", "must be ordered 0..7")
        if (
            sum(
                item.distribution is CandidateDistribution.ID_LIKE
                for item in definitions
            )
            != 4
        ):
            _fail(
                "CandidatePoolConfigurationV1.candidates", "expected four ID-like slots"
            )
        if (
            sum(
                item.distribution is CandidateDistribution.SHIFTED
                for item in definitions
            )
            != 4
        ):
            _fail(
                "CandidatePoolConfigurationV1.candidates", "expected four shifted slots"
            )
        severity_ids = tuple(item.severity_id for item in definitions)
        if len(severity_ids) != len(set(severity_ids)):
            _fail(
                "CandidatePoolConfigurationV1.candidates", "severity IDs must be unique"
            )
        if self.schema_version != SELECTION_SCHEMA_VERSION:
            _fail("CandidatePoolConfigurationV1.schema_version", "unsupported version")
        object.__setattr__(self, "candidates", definitions)

    def as_mapping(self) -> dict[str, object]:
        """Return the exact checked-in schedule payload."""

        return {
            "action_contract_digest": self.action_contract_digest,
            "action_dimension": self.action_dimension,
            "base_seed": self.base_seed,
            "candidate_horizon": self.candidate_horizon,
            "candidates": [item.as_mapping() for item in self.candidates],
            "schema_version": self.schema_version,
            "semantic": self.semantic,
        }

    @property
    def content_digest(self) -> str:
        """Return the path-independent schedule identity."""

        encoded = canonical_json_bytes(self.as_mapping())
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _reject_constant(value: str) -> NoReturn:
    _fail("candidate-pool JSON", f"unsupported non-finite constant {value!r}")


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("candidate-pool JSON", f"duplicate field {key!r}")
        result[key] = value
    return result


def candidate_pool_configuration_from_mapping(
    value: Mapping[str, object],
) -> CandidatePoolConfigurationV1:
    """Decode exact configuration fields without defaults or coercion."""

    expected = {
        "action_contract_digest",
        "action_dimension",
        "base_seed",
        "candidate_horizon",
        "candidates",
        "schema_version",
        "semantic",
    }
    if set(value) != expected:
        _fail("CandidatePoolConfigurationV1", "unexpected or missing fields")
    raw_candidates = value["candidates"]
    if not isinstance(raw_candidates, list):
        _fail("CandidatePoolConfigurationV1.candidates", "expected a list")
    definitions: list[CandidateDefinitionV1] = []
    definition_fields = {
        "configuration_ordinal",
        "corruption_type",
        "distribution",
        "parameters",
        "schema_version",
        "severity_id",
        "window_end",
        "window_start",
    }
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping) or set(raw) != definition_fields:
            _fail(f"CandidatePoolConfigurationV1.candidates[{index}]", "invalid fields")
        parameters = raw["parameters"]
        if not isinstance(parameters, Mapping):
            _fail(
                f"CandidatePoolConfigurationV1.candidates[{index}].parameters",
                "expected object",
            )
        definitions.append(
            CandidateDefinitionV1(
                configuration_ordinal=cast(int, raw["configuration_ordinal"]),
                distribution=cast(CandidateDistribution, raw["distribution"]),
                corruption_type=cast(str, raw["corruption_type"]),
                parameters=cast(Mapping[str, object], parameters),
                window_start=cast(int, raw["window_start"]),
                window_end=cast(int, raw["window_end"]),
                severity_id=cast(str, raw["severity_id"]),
                schema_version=cast(str, raw["schema_version"]),
            )
        )
    return CandidatePoolConfigurationV1(
        action_contract_digest=cast(str, value["action_contract_digest"]),
        base_seed=cast(int, value["base_seed"]),
        candidates=tuple(definitions),
        action_dimension=cast(int, value["action_dimension"]),
        candidate_horizon=cast(int, value["candidate_horizon"]),
        semantic=cast(str, value["semantic"]),
        schema_version=cast(str, value["schema_version"]),
    )


def load_candidate_pool_configuration(path: Path) -> CandidatePoolConfigurationV1:
    """Load a regular, single-linked, duplicate-free checked-in JSON file."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        _fail("candidate-pool configuration", "expected regular non-symlink file")
    if source.stat().st_nlink != 1:
        _fail("candidate-pool configuration", "hard-linked files are unsupported")
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_constant,
        )
    except CandidatePoolConfigurationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidatePoolConfigurationError(
            f"candidate-pool configuration: could not read safely: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        _fail("candidate-pool configuration", "expected JSON object")
    return candidate_pool_configuration_from_mapping(cast(Mapping[str, object], raw))


__all__ = [
    "CANDIDATE_POOL_SEMANTIC",
    "CandidateDefinitionV1",
    "CandidatePoolConfigurationError",
    "CandidatePoolConfigurationV1",
    "candidate_pool_configuration_from_mapping",
    "load_candidate_pool_configuration",
]

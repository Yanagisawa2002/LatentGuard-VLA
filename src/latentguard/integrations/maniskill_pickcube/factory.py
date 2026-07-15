"""Explicit registry factory for the trusted PickCube replay adapter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from latentguard.replay.base import ExactReplayAdapter
from latentguard.replay.source import ReplaySourceBinding

from .compatibility import load_compatibility_report, validate_compatibility_report
from .configuration import (
    load_expected_contract,
    load_maniskill_pickcube_action_layout,
    validate_maniskill_pickcube_action_layout_binding,
)
from .serialization import build_maniskill_pickcube_adapter

PICKCUBE_RUNTIME_CONFIGURATION_SCHEMA_VERSION = "1.0"
_RUNTIME_CONFIGURATION_FIELDS = frozenset(
    {
        "action_layout_path",
        "compatibility_report_path",
        "corruption_dataset_directory",
        "expected_contract_path",
        "runtime_archive_directory",
        "schema_version",
        "source_dataset_directory",
    }
)


class ManiSkillPickCubeFactoryError(ValueError):
    """Raised when runtime paths and checked-in semantic artifacts disagree."""


def _runtime_path(configuration: Mapping[str, object], name: str) -> Path:
    value = configuration[name]
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise ManiSkillPickCubeFactoryError(
            f"PickCubeRuntimeConfiguration.{name}: expected a non-empty path string"
        )
    return Path(value).absolute()


def _validate_runtime_configuration(
    configuration: Mapping[str, object],
) -> dict[str, Path]:
    fields = set(configuration)
    missing = sorted(_RUNTIME_CONFIGURATION_FIELDS - fields)
    unexpected = sorted(fields - _RUNTIME_CONFIGURATION_FIELDS)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ManiSkillPickCubeFactoryError(
            "PickCubeRuntimeConfiguration: invalid fields (" + "; ".join(details) + ")"
        )
    if configuration["schema_version"] != PICKCUBE_RUNTIME_CONFIGURATION_SCHEMA_VERSION:
        raise ManiSkillPickCubeFactoryError(
            "PickCubeRuntimeConfiguration.schema_version: unsupported version"
        )
    paths = {
        name: _runtime_path(configuration, name)
        for name in _RUNTIME_CONFIGURATION_FIELDS
        if name != "schema_version"
    }
    archive = paths["runtime_archive_directory"].resolve()
    for name in ("source_dataset_directory", "corruption_dataset_directory"):
        dataset = paths[name].resolve()
        if (
            archive == dataset
            or archive.is_relative_to(dataset)
            or dataset.is_relative_to(archive)
        ):
            raise ManiSkillPickCubeFactoryError(
                "PickCube runtime archive and serialized datasets must not overlap"
            )
    return paths


def _require_same_source_binding(
    supplied: ReplaySourceBinding,
    configured: ReplaySourceBinding,
) -> None:
    comparisons = (
        ("source_dataset_id", supplied.source_dataset_id, configured.source_dataset_id),
        (
            "source_dataset_digest",
            supplied.source_dataset_digest,
            configured.source_dataset_digest,
        ),
        (
            "corruption_dataset_digest",
            supplied.corruption_dataset_digest,
            configured.corruption_dataset_digest,
        ),
        ("proposal_ids", supplied.proposal_ids, configured.proposal_ids),
    )
    mismatches = [name for name, left, right in comparisons if left != right]
    if mismatches:
        raise ManiSkillPickCubeFactoryError(
            "PickCube configured source/corruption artifacts differ from the "
            "registry binding: " + ", ".join(mismatches)
        )


def create_maniskill_pickcube_registry_adapter(
    configuration: Mapping[str, object],
    source_binding: ReplaySourceBinding,
) -> ExactReplayAdapter:
    """Create the explicit real adapter from paths excluded from its identity."""
    paths = _validate_runtime_configuration(configuration)
    configured_binding = ReplaySourceBinding.from_paths(
        paths["source_dataset_directory"],
        paths["corruption_dataset_directory"],
    )
    _require_same_source_binding(source_binding, configured_binding)
    configured_binding.assert_unchanged()

    expected = load_expected_contract(paths["expected_contract_path"])
    report = load_compatibility_report(paths["compatibility_report_path"])
    compatibility_binding = validate_compatibility_report(
        report,
        expected,
        require_trusted=True,
    )
    checked_layout = load_maniskill_pickcube_action_layout(paths["action_layout_path"])
    layout_digest = validate_maniskill_pickcube_action_layout_binding(
        checked_layout,
        compatibility_binding,
        source_binding.corruption_dataset.action_layout,
    )
    return build_maniskill_pickcube_adapter(
        source_binding=source_binding,
        archive_dir=paths["runtime_archive_directory"],
        compatibility_binding=compatibility_binding,
        action_layout_digest=layout_digest,
        coordinate_frame=checked_layout.coordinate_frame,
    )


__all__ = [
    "PICKCUBE_RUNTIME_CONFIGURATION_SCHEMA_VERSION",
    "ManiSkillPickCubeFactoryError",
    "create_maniskill_pickcube_registry_adapter",
]

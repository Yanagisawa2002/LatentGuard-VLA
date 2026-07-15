"""Compact sanitized reporting for PickCube reference collection."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from latentguard.replay.identity import canonical_json_value

from .source_generation import SourceCollectionResult

COLLECTION_SUMMARY_SCHEMA_VERSION = "1.0"


class PickCubeReportingError(ValueError):
    """Raised when a compact report is incomplete or cannot be published safely."""


def _attempt_payload(result: SourceCollectionResult) -> dict[str, object]:
    failure_categories = Counter(
        attempt.failure_category
        for attempt in result.attempts
        if attempt.failure_category is not None
    )
    return {
        "accepted_source_count": result.accepted_count,
        "attempt_count": len(result.attempts),
        "attempts": [
            {
                "accepted": attempt.accepted,
                "failure_category": attempt.failure_category,
                "seed": attempt.seed,
            }
            for attempt in result.attempts
        ],
        "failure_categories": dict(sorted(failure_categories.items())),
        "requested_source_count": result.requested_success_count,
    }


def build_collection_summary(
    result: SourceCollectionResult,
    *,
    source_dataset_id: str,
) -> Mapping[str, object]:
    """Build a path-free collection summary from a complete accepted archive."""
    archive = result.archive
    if archive is None or result.accepted_count != result.requested_success_count:
        raise PickCubeReportingError(
            "collection summary requires a complete independently validated archive"
        )
    if not isinstance(source_dataset_id, str) or not source_dataset_id.startswith(
        "sha256:"
    ):
        raise PickCubeReportingError(
            "collection summary requires the serialized M0 bundle identifier"
        )
    compatibility_identities = {
        episode.compatibility_identity for episode in archive.episodes
    }
    if len(compatibility_identities) != 1:
        raise PickCubeReportingError(
            "collection summary requires one compatibility identity"
        )
    sources = [
        {
            "action_count": int(episode.source_actions.shape[0]),
            "episode_id": episode.episode_id,
            "initial_state_digest": episode.initial_state_digest,
            "seed": episode.seed,
            "source_action_digest": episode.source_action_digest,
            "source_trajectory_id": episode.source_trajectory_id,
            "terminal_state_digest": episode.terminal_state_digest,
        }
        for episode in archive.episodes
    ]
    value: dict[str, object] = {
        **_attempt_payload(result),
        "archive_content_digest": archive.content_digest,
        "compatibility_identity": next(iter(compatibility_identities)),
        "independent_baseline_success_count": sum(
            episode.independent_baseline_success for episode in archive.episodes
        ),
        "schema_version": COLLECTION_SUMMARY_SCHEMA_VERSION,
        "status": "complete",
        "source_dataset_id": source_dataset_id,
        "sources": sources,
    }
    canonical_json_value(
        value,
        context="PickCubeCollectionSummary",
        reject_runtime_paths=True,
    )
    return MappingProxyType(value)


def build_incomplete_collection_summary(
    result: SourceCollectionResult,
    *,
    compatibility_identity: str,
) -> Mapping[str, object]:
    """Build a sanitized failed-attempt audit when the bounded target is unmet."""
    if (
        result.archive is not None
        or result.accepted_count >= result.requested_success_count
    ):
        raise PickCubeReportingError(
            "incomplete summary requires an unmet bounded collection target"
        )
    if not isinstance(compatibility_identity, str) or not compatibility_identity:
        raise PickCubeReportingError(
            "incomplete summary requires a compatibility identity"
        )
    value: dict[str, object] = {
        **_attempt_payload(result),
        "compatibility_identity": compatibility_identity,
        "schema_version": COLLECTION_SUMMARY_SCHEMA_VERSION,
        "status": "incomplete",
    }
    canonical_json_value(
        value,
        context="PickCubeIncompleteCollectionSummary",
        reject_runtime_paths=True,
    )
    return MappingProxyType(value)


def write_sanitized_report(value: Mapping[str, object], path: Path) -> Path:
    """Publish one new canonical JSON report without paths or overwrites."""
    canonical_json_value(
        value,
        context="PickCubeSanitizedReport",
        reject_runtime_paths=True,
    )
    destination = Path(path).absolute()
    if destination.exists() or destination.is_symlink():
        raise PickCubeReportingError(
            "sanitized report output already exists; evidence is immutable"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise PickCubeReportingError(
                "sanitized report output already exists; evidence is immutable"
            ) from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return destination


def format_collection_summary(value: Mapping[str, object]) -> str:
    """Return a concise one-line collection completion summary."""
    return (
        "collect-maniskill-pickcube OK: "
        f"attempts={value['attempt_count']} "
        f"accepted={value['accepted_source_count']} "
        "independent_baselines="
        f"{value['independent_baseline_success_count']} "
        f"archive={value['archive_content_digest']} "
        f"source_dataset={value['source_dataset_id']}"
    )


__all__ = [
    "COLLECTION_SUMMARY_SCHEMA_VERSION",
    "PickCubeReportingError",
    "build_collection_summary",
    "build_incomplete_collection_summary",
    "format_collection_summary",
    "write_sanitized_report",
]

"""Canonical, path-independent SHA-256 identities for replay content."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath, PureWindowsPath
from typing import NoReturn

import numpy as np

from latentguard.models import ActionChunk
from latentguard.replay.base import ReplayValidationError
from latentguard.replay.models import (
    REPLAY_SCHEMA_VERSION,
    ReplayBundle,
    ReplayCase,
    ReplayStateReference,
    ReplayTaskReference,
)
from latentguard.validation import DataValidationError, validate_action_chunk

ACTION_DIGEST_PREFIX = "sha256:"
REPLAY_CASE_ID_PREFIX = "rpc-sha256-"
REPLAY_BUNDLE_DIGEST_PREFIX = "rpb-sha256-"

_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
_URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _fail(context: str, reason: str) -> NoReturn:
    raise ReplayValidationError(f"{context}: {reason}")


def _looks_like_runtime_path(value: str) -> bool:
    """Return whether text is an absolute runtime path or file URI."""
    if not value:
        return False
    if value.startswith(("/", "\\\\", "//")):
        return True
    if _WINDOWS_DRIVE_PATTERN.match(value) is not None:
        return True
    if value.lower().startswith("file:") or _URI_PATTERN.match(value) is not None:
        return True
    try:
        return (
            PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()
        )
    except (OSError, ValueError):
        return True


def canonical_json_value(
    value: object,
    *,
    context: str = "canonical JSON",
    reject_runtime_paths: bool = True,
) -> object:
    """Return a detached JSON-native value with strict finite/path-safe semantics."""
    if value is None or type(value) in (int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail(context, "floating-point values must be finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            _fail(context, "control characters are unsupported")
        if reject_runtime_paths and _looks_like_runtime_path(value):
            _fail(context, "absolute runtime paths and URI values are unsupported")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or not key
                or key != key.strip()
                or any(
                    ord(character) < 32 or ord(character) == 127 for character in key
                )
            ):
                _fail(context, "object keys must be non-empty canonical strings")
            result[key] = canonical_json_value(
                item,
                context=f"{context}.{key}",
                reject_runtime_paths=reject_runtime_paths,
            )
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [
            canonical_json_value(
                item,
                context=f"{context}[{index}]",
                reject_runtime_paths=reject_runtime_paths,
            )
            for index, item in enumerate(value)
        ]
    _fail(context, f"unsupported value type {type(value).__name__}")


def canonical_json_bytes(
    value: object,
    *,
    context: str = "canonical JSON",
    reject_runtime_paths: bool = True,
) -> bytes:
    """Serialize canonical JSON using stable UTF-8 bytes and sorted keys."""
    canonical = canonical_json_value(
        value,
        context=context,
        reject_runtime_paths=reject_runtime_paths,
    )
    try:
        return json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise ReplayValidationError(f"{context}: cannot encode canonical JSON") from exc


def _digest_payload(payload: object, *, context: str, prefix: str) -> str:
    encoded = canonical_json_bytes(payload, context=context)
    return f"{prefix}{hashlib.sha256(encoded).hexdigest()}"


def _action_payload(action: ActionChunk) -> dict[str, object]:
    if not isinstance(action, ActionChunk):
        _fail("ActionChunk", "expected ActionChunk")
    try:
        validate_action_chunk(action, "replay-identity")
    except DataValidationError as exc:
        raise ReplayValidationError(f"ActionChunk: {exc}") from exc
    contiguous = np.ascontiguousarray(action.actions)
    return {
        "content_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        "control_period_s": action.control_period_s,
        "coordinate_frame": action.coordinate_frame,
        "dtype": contiguous.dtype.str,
        "schema_version": action.schema_version,
        "shape": list(contiguous.shape),
    }


def compute_action_content_digest(action: ActionChunk) -> str:
    """Hash action dtype, shape, semantic contract, and exact C-order bytes."""
    return _digest_payload(
        _action_payload(action),
        context="ReplayActionIdentity",
        prefix=ACTION_DIGEST_PREFIX,
    )


def compute_action_digest(action: ActionChunk) -> str:
    """Alias for :func:`compute_action_content_digest`."""
    return compute_action_content_digest(action)


def _state_reference_payload(reference: ReplayStateReference) -> dict[str, object]:
    return {
        "adapter_id": reference.adapter_id,
        "adapter_version": reference.adapter_version,
        "comparison_semantic": reference.comparison_semantic.value,
        "expected_state_digest": reference.expected_state_digest,
        "metadata": reference.metadata,
        "schema_version": reference.schema_version,
        "source_reference_id": reference.source_reference_id,
        "state_index": reference.state_index,
        "state_key": reference.state_key,
    }


def _task_reference_payload(reference: ReplayTaskReference) -> dict[str, object]:
    return {
        "metadata": reference.metadata,
        "schema_version": reference.schema_version,
        "task_contract_version": reference.task_contract_version,
        "task_id": reference.task_id,
    }


def compute_replay_case_identifier(
    *,
    proposal_id: str,
    source_dataset_id: str,
    source_dataset_digest: str,
    corruption_dataset_digest: str,
    source_episode_id: str,
    source_candidate_id: str,
    split_group_id: str,
    original_action: ActionChunk,
    transformed_action: ActionChunk,
    state_reference: ReplayStateReference,
    task_reference: ReplayTaskReference,
    adapter_id: str,
    adapter_version: str,
    progress_semantic: str,
    unsafe_semantic: str,
    schema_version: str = REPLAY_SCHEMA_VERSION,
) -> str:
    """Compute a deterministic replay-case ID from all stable semantic inputs."""
    payload = {
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "corruption_dataset_digest": corruption_dataset_digest,
        "original_action_digest": compute_action_content_digest(original_action),
        "progress_semantic": progress_semantic,
        "proposal_id": proposal_id,
        "schema_version": schema_version,
        "source_candidate_id": source_candidate_id,
        "source_dataset_digest": source_dataset_digest,
        "source_dataset_id": source_dataset_id,
        "source_episode_id": source_episode_id,
        "split_group_id": split_group_id,
        "state_reference": _state_reference_payload(state_reference),
        "task_reference": _task_reference_payload(task_reference),
        "transformed_action_digest": compute_action_content_digest(transformed_action),
        "unsafe_semantic": unsafe_semantic,
    }
    return _digest_payload(
        payload,
        context="ReplayCaseIdentity",
        prefix=REPLAY_CASE_ID_PREFIX,
    )


def recompute_replay_case_identifier(replay_case: ReplayCase) -> str:
    """Recompute the canonical identifier of an existing replay case."""
    return compute_replay_case_identifier(
        proposal_id=replay_case.proposal_id,
        source_dataset_id=replay_case.source_dataset_id,
        source_dataset_digest=replay_case.source_dataset_digest,
        corruption_dataset_digest=replay_case.corruption_dataset_digest,
        source_episode_id=replay_case.source_episode_id,
        source_candidate_id=replay_case.source_candidate_id,
        split_group_id=replay_case.split_group_id,
        original_action=replay_case.original_action,
        transformed_action=replay_case.transformed_action,
        state_reference=replay_case.state_reference,
        task_reference=replay_case.task_reference,
        adapter_id=replay_case.adapter_id,
        adapter_version=replay_case.adapter_version,
        progress_semantic=replay_case.progress_semantic,
        unsafe_semantic=replay_case.unsafe_semantic,
        schema_version=replay_case.schema_version,
    )


def compute_replay_bundle_digest(
    *,
    source_dataset_id: str,
    source_dataset_digest: str,
    corruption_dataset_digest: str,
    adapter_id: str,
    adapter_version: str,
    adapter_configuration_digest: str,
    replay_cases: Sequence[ReplayCase],
    metadata: Mapping[str, object],
    schema_version: str = REPLAY_SCHEMA_VERSION,
) -> str:
    """Compute a canonical ordered replay-bundle digest without runtime metadata."""
    payload = {
        "adapter_configuration_digest": adapter_configuration_digest,
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "corruption_dataset_digest": corruption_dataset_digest,
        "metadata": metadata,
        "replay_case_ids": [replay_case.case_id for replay_case in replay_cases],
        "schema_version": schema_version,
        "source_dataset_digest": source_dataset_digest,
        "source_dataset_id": source_dataset_id,
    }
    return _digest_payload(
        payload,
        context="ReplayBundleIdentity",
        prefix=REPLAY_BUNDLE_DIGEST_PREFIX,
    )


def recompute_replay_bundle_digest(bundle: ReplayBundle) -> str:
    """Recompute the canonical digest of an existing replay bundle."""
    return compute_replay_bundle_digest(
        source_dataset_id=bundle.source_dataset_id,
        source_dataset_digest=bundle.source_dataset_digest,
        corruption_dataset_digest=bundle.corruption_dataset_digest,
        adapter_id=bundle.adapter_id,
        adapter_version=bundle.adapter_version,
        adapter_configuration_digest=bundle.adapter_configuration_digest,
        replay_cases=bundle.replay_cases,
        metadata=bundle.metadata,
        schema_version=bundle.schema_version,
    )


__all__ = [
    "ACTION_DIGEST_PREFIX",
    "REPLAY_BUNDLE_DIGEST_PREFIX",
    "REPLAY_CASE_ID_PREFIX",
    "canonical_json_bytes",
    "canonical_json_value",
    "compute_action_content_digest",
    "compute_action_digest",
    "compute_replay_bundle_digest",
    "compute_replay_case_identifier",
    "recompute_replay_bundle_digest",
    "recompute_replay_case_identifier",
]

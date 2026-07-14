"""Safe JSON-only serialization for content-bound replay bundles."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import NoReturn, cast

from latentguard.corruptions.models import CorruptedActionProposal
from latentguard.replay.base import ReplayValidationError
from latentguard.replay.identity import (
    canonical_json_value,
    compute_action_content_digest,
    compute_replay_bundle_digest,
    recompute_replay_bundle_digest,
    recompute_replay_case_identifier,
)
from latentguard.replay.models import (
    REPLAY_SCHEMA_VERSION,
    ReplayBundle,
    ReplayCase,
    ReplayStateReference,
    ReplayTaskReference,
    StateComparisonSemantic,
)
from latentguard.replay.source import ReplaySourceBinding
from latentguard.replay.validation import validate_replay_bundle

MANIFEST_NAME = "manifest.json"
"""Filename used by every serialized replay bundle."""

SERIALIZATION_FORMAT = "latentguard-replay-bundle"
"""Stable replay-bundle format discriminator."""

SERIALIZATION_VERSION = 1
"""Replay serialization version supported by this release."""

_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "serialization_version",
        "schema_version",
        "source_dataset_id",
        "source_dataset_digest",
        "corruption_dataset_digest",
        "adapter_id",
        "adapter_version",
        "adapter_configuration_digest",
        "case_count",
        "replay_cases",
        "bundle_digest",
        "metadata",
    }
)
_CASE_FIELDS = frozenset(
    {
        "case_id",
        "proposal_id",
        "source_dataset_id",
        "source_dataset_digest",
        "corruption_dataset_digest",
        "source_episode_id",
        "source_candidate_id",
        "split_group_id",
        "original_action_reference",
        "transformed_action_reference",
        "original_action_digest",
        "transformed_action_digest",
        "state_reference",
        "task_reference",
        "adapter_id",
        "adapter_version",
        "progress_semantic",
        "unsafe_semantic",
        "schema_version",
    }
)
_ORIGINAL_ACTION_REFERENCE_FIELDS = frozenset(
    {"source_episode_id", "source_candidate_id"}
)
_TRANSFORMED_ACTION_REFERENCE_FIELDS = frozenset({"proposal_id"})
_STATE_REFERENCE_FIELDS = frozenset(
    {
        "adapter_id",
        "adapter_version",
        "source_reference_id",
        "expected_state_digest",
        "comparison_semantic",
        "state_key",
        "state_index",
        "metadata",
        "schema_version",
    }
)
_TASK_REFERENCE_FIELDS = frozenset(
    {"task_id", "task_contract_version", "metadata", "schema_version"}
)


class ReplaySerializationError(ValueError):
    """Raised when replay-bundle storage is malformed, unsafe, or inconsistent."""


class UnsupportedReplaySerializationVersionError(ReplaySerializationError):
    """Raised when a replay bundle uses an unsupported storage or model version."""


def _canonical_metadata(value: object, context: str) -> dict[str, object]:
    try:
        canonical = canonical_json_value(value, context=context)
    except ReplayValidationError as exc:
        raise ReplaySerializationError(str(exc)) from exc
    if not isinstance(canonical, dict):
        raise ReplaySerializationError(f"{context}: expected an object")
    return cast(dict[str, object], canonical)


def _encode_state_reference(reference: ReplayStateReference) -> dict[str, object]:
    return {
        "adapter_id": reference.adapter_id,
        "adapter_version": reference.adapter_version,
        "source_reference_id": reference.source_reference_id,
        "expected_state_digest": reference.expected_state_digest,
        "comparison_semantic": reference.comparison_semantic.value,
        "state_key": reference.state_key,
        "state_index": reference.state_index,
        "metadata": _canonical_metadata(
            reference.metadata, "ReplayStateReference.metadata"
        ),
        "schema_version": reference.schema_version,
    }


def _encode_task_reference(reference: ReplayTaskReference) -> dict[str, object]:
    return {
        "task_id": reference.task_id,
        "task_contract_version": reference.task_contract_version,
        "metadata": _canonical_metadata(
            reference.metadata, "ReplayTaskReference.metadata"
        ),
        "schema_version": reference.schema_version,
    }


def _encode_case(replay_case: ReplayCase) -> dict[str, object]:
    return {
        "case_id": replay_case.case_id,
        "proposal_id": replay_case.proposal_id,
        "source_dataset_id": replay_case.source_dataset_id,
        "source_dataset_digest": replay_case.source_dataset_digest,
        "corruption_dataset_digest": replay_case.corruption_dataset_digest,
        "source_episode_id": replay_case.source_episode_id,
        "source_candidate_id": replay_case.source_candidate_id,
        "split_group_id": replay_case.split_group_id,
        "original_action_reference": {
            "source_episode_id": replay_case.source_episode_id,
            "source_candidate_id": replay_case.source_candidate_id,
        },
        "transformed_action_reference": {"proposal_id": replay_case.proposal_id},
        "original_action_digest": compute_action_content_digest(
            replay_case.original_action
        ),
        "transformed_action_digest": compute_action_content_digest(
            replay_case.transformed_action
        ),
        "state_reference": _encode_state_reference(replay_case.state_reference),
        "task_reference": _encode_task_reference(replay_case.task_reference),
        "adapter_id": replay_case.adapter_id,
        "adapter_version": replay_case.adapter_version,
        "progress_semantic": replay_case.progress_semantic,
        "unsafe_semantic": replay_case.unsafe_semantic,
        "schema_version": replay_case.schema_version,
    }


def _binding_proposals(
    source_binding: ReplaySourceBinding,
) -> dict[str, CorruptedActionProposal]:
    return {
        proposal.proposal_id: proposal
        for proposal in source_binding.corruption_dataset.proposals
    }


def _validate_bundle_binding(
    bundle: ReplayBundle, source_binding: ReplaySourceBinding
) -> None:
    if not isinstance(source_binding, ReplaySourceBinding):
        raise ReplaySerializationError(
            "ReplayBundle.source_binding: expected ReplaySourceBinding"
        )
    source_binding.assert_unchanged()
    try:
        validate_replay_bundle(bundle)
    except ReplayValidationError as exc:
        raise ReplaySerializationError(f"ReplayBundle: {exc}") from exc
    expected_binding = {
        "source_dataset_id": source_binding.source_dataset_id,
        "source_dataset_digest": source_binding.source_dataset_digest,
        "corruption_dataset_digest": source_binding.corruption_dataset_digest,
    }
    mismatches = [
        field
        for field, expected in expected_binding.items()
        if getattr(bundle, field) != expected
    ]
    if mismatches:
        raise ReplaySerializationError(
            "ReplayBundle: source binding mismatch in " + ", ".join(mismatches)
        )
    proposal_positions = {
        proposal_id: index
        for index, proposal_id in enumerate(source_binding.proposal_ids)
    }
    proposals = _binding_proposals(source_binding)
    previous_position = -1
    for index, replay_case in enumerate(bundle.replay_cases):
        context = f"ReplayBundle.replay_cases[{index}]"
        try:
            position = proposal_positions[replay_case.proposal_id]
            proposal = proposals[replay_case.proposal_id]
        except KeyError as exc:
            raise ReplaySerializationError(
                f"{context}.proposal_id: proposal is absent from the bound M1 dataset"
            ) from exc
        if position <= previous_position:
            raise ReplaySerializationError(
                f"{context}.proposal_id: replay cases must preserve M1 generation order"
            )
        previous_position = position
        pair = source_binding.resolve(proposal)
        expected_case = {
            "proposal_id": pair.proposal_id,
            "source_dataset_id": pair.source_dataset_id,
            "source_dataset_digest": pair.source_dataset_digest,
            "corruption_dataset_digest": pair.corruption_dataset_digest,
            "source_episode_id": pair.source_episode_id,
            "source_candidate_id": pair.source_candidate_id,
            "split_group_id": pair.split_group_id,
        }
        case_mismatches = [
            field
            for field, expected in expected_case.items()
            if getattr(replay_case, field) != expected
        ]
        if case_mismatches:
            raise ReplaySerializationError(
                f"{context}: source reference mismatch in " + ", ".join(case_mismatches)
            )
        if replay_case.adapter_id != bundle.adapter_id or (
            replay_case.adapter_version != bundle.adapter_version
        ):
            raise ReplaySerializationError(
                f"{context}: adapter identity does not match bundle"
            )
        if replay_case.task_reference.task_id != pair.source_task_id:
            raise ReplaySerializationError(
                f"{context}.task_reference.task_id: does not match source task"
            )
        for field, actual, expected in (
            (
                "original_action_digest",
                compute_action_content_digest(replay_case.original_action),
                compute_action_content_digest(pair.original_action),
            ),
            (
                "transformed_action_digest",
                compute_action_content_digest(replay_case.transformed_action),
                compute_action_content_digest(pair.transformed_action),
            ),
        ):
            if actual != expected:
                raise ReplaySerializationError(
                    f"{context}.{field}: does not match bound action content"
                )
        if replay_case.case_id != recompute_replay_case_identifier(replay_case):
            raise ReplaySerializationError(
                f"{context}.case_id: deterministic identifier mismatch"
            )
    if bundle.bundle_digest != recompute_replay_bundle_digest(bundle):
        raise ReplaySerializationError(
            "ReplayBundle.bundle_digest: deterministic digest mismatch"
        )


def build_replay_bundle(
    source_binding: ReplaySourceBinding,
    replay_cases: Sequence[ReplayCase],
    *,
    adapter_id: str,
    adapter_version: str,
    adapter_configuration_digest: str,
    metadata: Mapping[str, object] | None = None,
) -> ReplayBundle:
    """Build and validate one ordered bundle without serializing action arrays."""

    cases = tuple(replay_cases)
    canonical_metadata = _canonical_metadata(
        {} if metadata is None else metadata, "ReplayBundle.metadata"
    )
    digest = compute_replay_bundle_digest(
        source_dataset_id=source_binding.source_dataset_id,
        source_dataset_digest=source_binding.source_dataset_digest,
        corruption_dataset_digest=source_binding.corruption_dataset_digest,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        adapter_configuration_digest=adapter_configuration_digest,
        replay_cases=cases,
        metadata=canonical_metadata,
        schema_version=REPLAY_SCHEMA_VERSION,
    )
    bundle = ReplayBundle(
        source_dataset_id=source_binding.source_dataset_id,
        source_dataset_digest=source_binding.source_dataset_digest,
        corruption_dataset_digest=source_binding.corruption_dataset_digest,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        adapter_configuration_digest=adapter_configuration_digest,
        replay_cases=cases,
        bundle_digest=digest,
        metadata=canonical_metadata,
        schema_version=REPLAY_SCHEMA_VERSION,
    )
    _validate_bundle_binding(bundle, source_binding)
    return bundle


def save_replay_bundle(
    bundle: ReplayBundle,
    output_dir: Path,
    source_binding: ReplaySourceBinding,
) -> Path:
    """Transactionally save a JSON-only bundle after source revalidation."""

    _validate_bundle_binding(bundle, source_binding)
    destination = Path(output_dir).absolute()
    _require_empty_destination(destination)
    staging: Path | None = None
    primary_error: BaseException | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name or 'replay-bundle'}.staging-",
                dir=destination.parent,
            )
        )
        manifest: dict[str, object] = {
            "format": SERIALIZATION_FORMAT,
            "serialization_version": SERIALIZATION_VERSION,
            "schema_version": bundle.schema_version,
            "source_dataset_id": bundle.source_dataset_id,
            "source_dataset_digest": bundle.source_dataset_digest,
            "corruption_dataset_digest": bundle.corruption_dataset_digest,
            "adapter_id": bundle.adapter_id,
            "adapter_version": bundle.adapter_version,
            "adapter_configuration_digest": bundle.adapter_configuration_digest,
            "case_count": len(bundle.replay_cases),
            "replay_cases": [_encode_case(item) for item in bundle.replay_cases],
            "bundle_digest": bundle.bundle_digest,
            "metadata": _canonical_metadata(bundle.metadata, "ReplayBundle.metadata"),
        }
        payload = json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        with (staging / MANIFEST_NAME).open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(payload + "\n")
        source_binding.assert_unchanged()
        _publish_staging_bundle(staging, destination)
    except ReplaySerializationError as exc:
        primary_error = exc
        raise
    except (OSError, TypeError, UnicodeError, ValueError) as exc:
        wrapped = ReplaySerializationError(
            f"ReplayBundle: could not save transactionally: {exc}"
        )
        primary_error = wrapped
        raise wrapped from exc
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if staging is not None and staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError as cleanup_error:
                if primary_error is not None:
                    primary_error.add_note(
                        f"Staging cleanup also failed: {cleanup_error}"
                    )
                else:
                    raise ReplaySerializationError(
                        "ReplayBundle: could not clean staging directory"
                    ) from cleanup_error
    return Path(output_dir) / MANIFEST_NAME


def _decode_state_reference(value: object, context: str) -> ReplayStateReference:
    item = _mapping(value, context)
    _require_exact_fields(item, _STATE_REFERENCE_FIELDS, context)
    version = _schema(item, context)
    try:
        semantic = StateComparisonSemantic(
            _string(item, "comparison_semantic", context)
        )
    except ValueError as exc:
        raise ReplaySerializationError(
            f"{context}.comparison_semantic: unsupported value"
        ) from exc
    raw_key = _field(item, "state_key", context)
    state_key = (
        None if raw_key is None else _strict_string(raw_key, f"{context}.state_key")
    )
    raw_index = _field(item, "state_index", context)
    if raw_index is not None and (type(raw_index) is not int or raw_index < 0):
        raise ReplaySerializationError(
            f"{context}.state_index: expected a non-negative integer or null"
        )
    return ReplayStateReference(
        adapter_id=_string(item, "adapter_id", context),
        adapter_version=_string(item, "adapter_version", context),
        source_reference_id=_string(item, "source_reference_id", context),
        expected_state_digest=_string(item, "expected_state_digest", context),
        comparison_semantic=semantic,
        state_key=state_key,
        state_index=raw_index,
        metadata=_canonical_metadata(
            _field(item, "metadata", context), f"{context}.metadata"
        ),
        schema_version=version,
    )


def _decode_task_reference(value: object, context: str) -> ReplayTaskReference:
    item = _mapping(value, context)
    _require_exact_fields(item, _TASK_REFERENCE_FIELDS, context)
    return ReplayTaskReference(
        task_id=_string(item, "task_id", context),
        task_contract_version=_string(item, "task_contract_version", context),
        metadata=_canonical_metadata(
            _field(item, "metadata", context), f"{context}.metadata"
        ),
        schema_version=_schema(item, context),
    )


def _decode_case(
    value: object,
    index: int,
    source_binding: ReplaySourceBinding,
    proposals: Mapping[str, CorruptedActionProposal],
) -> ReplayCase:
    context = f"ReplayBundle.replay_cases[{index}]"
    item = _mapping(value, context)
    _require_exact_fields(item, _CASE_FIELDS, context)
    version = _schema(item, context)
    proposal_id = _string(item, "proposal_id", context)
    try:
        proposal = proposals[proposal_id]
    except KeyError as exc:
        raise ReplaySerializationError(
            f"{context}.proposal_id: absent from bound M1 dataset"
        ) from exc
    pair = source_binding.resolve(proposal)
    original_reference = _mapping(
        _field(item, "original_action_reference", context),
        f"{context}.original_action_reference",
    )
    _require_exact_fields(
        original_reference,
        _ORIGINAL_ACTION_REFERENCE_FIELDS,
        f"{context}.original_action_reference",
    )
    if (
        _string(
            original_reference,
            "source_episode_id",
            f"{context}.original_action_reference",
        )
        != pair.source_episode_id
        or _string(
            original_reference,
            "source_candidate_id",
            f"{context}.original_action_reference",
        )
        != pair.source_candidate_id
    ):
        raise ReplaySerializationError(
            f"{context}.original_action_reference: does not resolve to proposal source"
        )
    transformed_reference = _mapping(
        _field(item, "transformed_action_reference", context),
        f"{context}.transformed_action_reference",
    )
    _require_exact_fields(
        transformed_reference,
        _TRANSFORMED_ACTION_REFERENCE_FIELDS,
        f"{context}.transformed_action_reference",
    )
    if (
        _string(
            transformed_reference,
            "proposal_id",
            f"{context}.transformed_action_reference",
        )
        != pair.proposal_id
    ):
        raise ReplaySerializationError(
            f"{context}.transformed_action_reference: does not match proposal"
        )
    expected_original_digest = compute_action_content_digest(pair.original_action)
    expected_transformed_digest = compute_action_content_digest(pair.transformed_action)
    if _string(item, "original_action_digest", context) != expected_original_digest:
        raise ReplaySerializationError(
            f"{context}.original_action_digest: bound source action changed"
        )
    if _string(item, "transformed_action_digest", context) != (
        expected_transformed_digest
    ):
        raise ReplaySerializationError(
            f"{context}.transformed_action_digest: bound proposal action changed"
        )
    replay_case = ReplayCase(
        case_id=_string(item, "case_id", context),
        proposal_id=proposal_id,
        source_dataset_id=_string(item, "source_dataset_id", context),
        source_dataset_digest=_string(item, "source_dataset_digest", context),
        corruption_dataset_digest=_string(item, "corruption_dataset_digest", context),
        source_episode_id=_string(item, "source_episode_id", context),
        source_candidate_id=_string(item, "source_candidate_id", context),
        split_group_id=_string(item, "split_group_id", context),
        original_action=pair.original_action,
        transformed_action=pair.transformed_action,
        state_reference=_decode_state_reference(
            _field(item, "state_reference", context), f"{context}.state_reference"
        ),
        task_reference=_decode_task_reference(
            _field(item, "task_reference", context), f"{context}.task_reference"
        ),
        adapter_id=_string(item, "adapter_id", context),
        adapter_version=_string(item, "adapter_version", context),
        progress_semantic=_string(item, "progress_semantic", context),
        unsafe_semantic=_string(item, "unsafe_semantic", context),
        schema_version=version,
    )
    if replay_case.case_id != recompute_replay_case_identifier(replay_case):
        raise ReplaySerializationError(
            f"{context}.case_id: deterministic identifier mismatch"
        )
    return replay_case


def load_replay_bundle(
    output_dir: Path, source_binding: ReplaySourceBinding
) -> ReplayBundle:
    """Load a JSON-only bundle and reconstruct actions from bound M0/M1 data."""

    source_binding.assert_unchanged()
    requested = Path(output_dir).absolute()
    _validate_bundle_root(requested)
    manifest_path = requested / MANIFEST_NAME
    _require_regular_unlinked_file(manifest_path, "ReplayBundle.manifest")
    try:
        raw = cast(
            object,
            json.loads(
                manifest_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_fields,
            ),
        )
    except ReplaySerializationError:
        raise
    except Exception as exc:
        raise ReplaySerializationError(
            f"ReplayBundle.manifest: could not read: {exc}"
        ) from exc
    manifest = _mapping(raw, "ReplayBundle.manifest")
    _require_exact_fields(manifest, _MANIFEST_FIELDS, "ReplayBundle.manifest")
    if _string(manifest, "format", "ReplayBundle.manifest") != SERIALIZATION_FORMAT:
        raise ReplaySerializationError("ReplayBundle.manifest.format: unsupported")
    version = _integer(manifest, "serialization_version", "ReplayBundle.manifest")
    if version != SERIALIZATION_VERSION:
        raise UnsupportedReplaySerializationVersionError(
            "ReplayBundle.manifest.serialization_version: unsupported version "
            f"{version!r}; supported: {SERIALIZATION_VERSION}"
        )
    schema_version = _schema(manifest, "ReplayBundle.manifest")
    encoded_cases = _list(manifest, "replay_cases", "ReplayBundle.manifest")
    case_count = _integer(manifest, "case_count", "ReplayBundle.manifest")
    if case_count < 0 or case_count != len(encoded_cases):
        raise ReplaySerializationError(
            "ReplayBundle.manifest.case_count: does not match replay_cases"
        )
    proposals = _binding_proposals(source_binding)
    try:
        cases = tuple(
            _decode_case(value, index, source_binding, proposals)
            for index, value in enumerate(encoded_cases)
        )
        bundle = ReplayBundle(
            source_dataset_id=_string(
                manifest, "source_dataset_id", "ReplayBundle.manifest"
            ),
            source_dataset_digest=_string(
                manifest, "source_dataset_digest", "ReplayBundle.manifest"
            ),
            corruption_dataset_digest=_string(
                manifest, "corruption_dataset_digest", "ReplayBundle.manifest"
            ),
            adapter_id=_string(manifest, "adapter_id", "ReplayBundle.manifest"),
            adapter_version=_string(
                manifest, "adapter_version", "ReplayBundle.manifest"
            ),
            adapter_configuration_digest=_string(
                manifest,
                "adapter_configuration_digest",
                "ReplayBundle.manifest",
            ),
            replay_cases=cases,
            bundle_digest=_string(manifest, "bundle_digest", "ReplayBundle.manifest"),
            metadata=_canonical_metadata(
                _field(manifest, "metadata", "ReplayBundle.manifest"),
                "ReplayBundle.manifest.metadata",
            ),
            schema_version=schema_version,
        )
    except ReplaySerializationError:
        raise
    except (ReplayValidationError, TypeError, ValueError) as exc:
        raise ReplaySerializationError(
            f"ReplayBundle.manifest: invalid replay model: {exc}"
        ) from exc
    _validate_bundle_binding(bundle, source_binding)
    source_binding.assert_unchanged()
    return bundle


def _require_empty_destination(destination: Path) -> None:
    try:
        if destination.is_symlink() or destination.resolve() != destination:
            raise ReplaySerializationError(
                "ReplayBundle.output_dir: symbolic links and junctions are unsupported"
            )
    except ReplaySerializationError:
        raise
    except OSError as exc:
        raise ReplaySerializationError(
            f"ReplayBundle.output_dir: could not inspect destination: {exc}"
        ) from exc
    if not destination.exists():
        return
    if not destination.is_dir():
        raise ReplaySerializationError("ReplayBundle.output_dir: must be a directory")
    try:
        next(destination.iterdir())
    except StopIteration:
        return
    except OSError as exc:
        raise ReplaySerializationError(
            f"ReplayBundle.output_dir: could not inspect destination: {exc}"
        ) from exc
    raise ReplaySerializationError(
        "ReplayBundle.output_dir: destination must be absent or empty"
    )


def _validate_bundle_root(root: Path) -> None:
    try:
        if root.is_symlink() or not root.is_dir() or root.resolve() != root:
            raise ReplaySerializationError(
                "ReplayBundle.output_dir: missing or unsafe bundle directory"
            )
        entries = {entry.name: entry for entry in root.iterdir()}
    except ReplaySerializationError:
        raise
    except OSError as exc:
        raise ReplaySerializationError(
            f"ReplayBundle.output_dir: could not inspect bundle: {exc}"
        ) from exc
    if set(entries) != {MANIFEST_NAME}:
        raise ReplaySerializationError(
            "ReplayBundle.output_dir: bundle inventory is incomplete or has extras"
        )


def _require_regular_unlinked_file(path: Path, context: str) -> None:
    try:
        if path.is_symlink() or not path.is_file():
            raise ReplaySerializationError(f"{context}: missing or unsafe regular file")
        if path.stat().st_nlink != 1:
            raise ReplaySerializationError(f"{context}: hard links are unsafe")
    except ReplaySerializationError:
        raise
    except OSError as exc:
        raise ReplaySerializationError(f"{context}: could not inspect: {exc}") from exc


def _publish_staging_bundle(staging: Path, destination: Path) -> None:
    destination_existed = destination.exists()
    if destination_existed:
        _require_empty_destination(destination)
        destination.rmdir()
    try:
        staging.replace(destination)
    except OSError:
        if destination_existed and not destination.exists():
            destination.mkdir()
        raise


def _require_exact_fields(
    item: Mapping[str, object], expected: Collection[str], context: str
) -> None:
    missing = sorted(set(expected) - set(item))
    unexpected = sorted(set(item) - set(expected))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ReplaySerializationError(
            f"{context}: invalid fields ({'; '.join(details)})"
        )


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ReplaySerializationError(f"{context}: expected an object")
    return cast(dict[str, object], value)


def _field(item: Mapping[str, object], field: str, context: str) -> object:
    if field not in item:
        raise ReplaySerializationError(f"{context}.{field}: missing required field")
    return item[field]


def _strict_string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ReplaySerializationError(f"{context}: expected a string")
    return value


def _string(item: Mapping[str, object], field: str, context: str) -> str:
    return _strict_string(_field(item, field, context), f"{context}.{field}")


def _integer(item: Mapping[str, object], field: str, context: str) -> int:
    value = _field(item, field, context)
    if type(value) is not int:
        raise ReplaySerializationError(f"{context}.{field}: expected an integer")
    return value


def _list(item: Mapping[str, object], field: str, context: str) -> list[object]:
    value = _field(item, field, context)
    if not isinstance(value, list):
        raise ReplaySerializationError(f"{context}.{field}: expected an array")
    return cast(list[object], value)


def _schema(item: Mapping[str, object], context: str) -> str:
    value = _string(item, "schema_version", context)
    if value != REPLAY_SCHEMA_VERSION:
        raise UnsupportedReplaySerializationVersionError(
            f"{context}.schema_version: unsupported version {value!r}; supported: "
            f"{REPLAY_SCHEMA_VERSION}"
        )
    return value


def _reject_json_constant(value: str) -> NoReturn:
    raise ReplaySerializationError(
        f"ReplayBundle.manifest: non-finite JSON constant {value!r} is unsupported"
    )


def _reject_duplicate_json_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReplaySerializationError(f"JSON object: duplicate field {key!r}")
        result[key] = value
    return result


__all__ = [
    "MANIFEST_NAME",
    "SERIALIZATION_FORMAT",
    "SERIALIZATION_VERSION",
    "ReplaySerializationError",
    "UnsupportedReplaySerializationVersionError",
    "build_replay_bundle",
    "load_replay_bundle",
    "save_replay_bundle",
]

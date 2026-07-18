"""Strict atomic JSON persistence for closed-loop runtime artifacts."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn, cast

import numpy as np

from latentguard.control.models import (
    BoundaryLedgerEntryV1,
    BoundaryState,
    ClosedLoopCandidatePoolV1,
    ClosedLoopCandidateV1,
    ClosedLoopDecisionRecordV1,
    ClosedLoopEpisodeRecordV1,
    EpisodeState,
)


class ClosedLoopSerializationError(ValueError):
    """Raised when persisted M4C runtime content is unsafe or inconsistent."""


def _fail(context: str, reason: str) -> NoReturn:
    raise ClosedLoopSerializationError(f"{context}: {reason}")


def _read(path: Path, context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected regular unlinked JSON file")
    try:
        raw = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClosedLoopSerializationError(f"{context}: {exc}") from exc
    if not isinstance(raw, Mapping):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], raw)


def write_atomic_json(path: Path, payload: Mapping[str, object]) -> Path:
    """Atomically replace one JSON file with canonical reviewable content."""

    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        temporary.write_text(text + "\n", encoding="utf-8", newline="\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def write_exclusive_json(path: Path, payload: Mapping[str, object]) -> Path:
    """Create one immutable JSON record and reject any prior content."""

    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text + "\n")
    except FileExistsError as exc:
        raise ClosedLoopSerializationError(
            f"immutable record already exists: {destination.name}"
        ) from exc
    return destination


def _exact(value: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(value) != expected:
        _fail(context, "unexpected or missing fields")


def save_candidate_pool(pool: ClosedLoopCandidatePoolV1, path: Path) -> Path:
    """Persist one immutable pool including runtime-only action values."""

    if not isinstance(pool, ClosedLoopCandidatePoolV1):
        _fail("save candidate pool", "invalid model")
    payload = {
        **pool.identity_mapping(),
        "candidates": [
            {
                **item.identity_mapping(),
                "action_chunk": item.action_chunk.tolist(),
                "action_mask": item.action_mask.tolist(),
            }
            for item in pool.candidates
        ],
        "content_digest": pool.content_digest,
    }
    return write_exclusive_json(path, payload)


def load_candidate_pool(path: Path) -> ClosedLoopCandidatePoolV1:
    """Strictly reload one immutable candidate pool."""

    raw = _read(path, "candidate pool")
    expected = {
        "candidate_content_digests",
        "candidate_pool_configuration_digest",
        "candidates",
        "content_digest",
        "decision_ordinal",
        "exact_source_excluded",
        "nominal_plan_index",
        "outcomes_available",
        "pre_decision_state_digest",
        "schema_version",
        "source_action_prefix_digest",
        "source_trajectory_id",
    }
    _exact(raw, expected, "candidate pool")
    entries = raw["candidates"]
    if not isinstance(entries, list):
        _fail("candidate pool.candidates", "expected list")
    candidates: list[ClosedLoopCandidateV1] = []
    candidate_expected = {
        "action_chunk",
        "action_chunk_digest",
        "action_mask",
        "action_mask_digest",
        "candidate_id",
        "definition_identity",
        "definition_ordinal",
        "schema_version",
    }
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            _fail(f"candidate pool.candidates[{index}]", "expected object")
        entry = cast(Mapping[str, object], item)
        _exact(entry, candidate_expected, f"candidate pool.candidates[{index}]")
        actions = np.asarray(entry["action_chunk"], dtype=np.dtype("<f8"), order="C")
        mask = np.asarray(entry["action_mask"], dtype=np.bool_, order="C")
        candidate = ClosedLoopCandidateV1(
            candidate_id=cast(str, entry["candidate_id"]),
            definition_ordinal=cast(int, entry["definition_ordinal"]),
            action_chunk=actions,
            action_mask=mask,
            definition_identity=cast(str, entry["definition_identity"]),
            schema_version=cast(str, entry["schema_version"]),
        )
        if candidate.identity_mapping() != {
            key: entry[key]
            for key in candidate_expected
            if key not in {"action_chunk", "action_mask"}
        }:
            _fail(f"candidate pool.candidates[{index}]", "identity changed")
        candidates.append(candidate)
    result = ClosedLoopCandidatePoolV1(
        source_trajectory_id=cast(str, raw["source_trajectory_id"]),
        decision_ordinal=cast(int, raw["decision_ordinal"]),
        nominal_plan_index=cast(int, raw["nominal_plan_index"]),
        pre_decision_state_digest=cast(str, raw["pre_decision_state_digest"]),
        source_action_prefix_digest=cast(str, raw["source_action_prefix_digest"]),
        candidate_pool_configuration_digest=cast(
            str, raw["candidate_pool_configuration_digest"]
        ),
        candidates=tuple(candidates),
        exact_source_excluded=cast(bool, raw["exact_source_excluded"]),
        outcomes_available=cast(bool, raw["outcomes_available"]),
        schema_version=cast(str, raw["schema_version"]),
    )
    if raw["content_digest"] != result.content_digest:
        _fail("candidate pool.content_digest", "content changed")
    if raw["candidate_content_digests"] != [
        item.content_digest for item in result.candidates
    ]:
        _fail("candidate pool", "candidate inventory changed")
    return result


def save_decision(record: ClosedLoopDecisionRecordV1, path: Path) -> Path:
    """Persist one immutable outcome-blind selection."""

    if not isinstance(record, ClosedLoopDecisionRecordV1):
        _fail("save decision", "invalid model")
    return write_exclusive_json(path, record.as_mapping())


def load_decision(path: Path) -> ClosedLoopDecisionRecordV1:
    """Strictly reload and authenticate one selection record."""

    raw = _read(path, "decision record")
    expected = {
        "candidate_pool_digest",
        "checkpoint_ensemble_identity",
        "content_digest",
        "decision_ordinal",
        "deterministic_ranking",
        "episode_execution_id",
        "execution_stride",
        "nominal_plan_index",
        "ordered_candidate_ids",
        "outcomes_available_during_selection",
        "pre_decision_state_digest",
        "schema_version",
        "selected_candidate_id",
        "selected_predicted_failure_probability",
        "selector_id",
        "selector_scores",
        "source_trajectory_id",
        "visual_domain",
        "visual_packet_identity",
    }
    _exact(raw, expected, "decision record")
    ids, ranking, scores = (
        raw["ordered_candidate_ids"],
        raw["deterministic_ranking"],
        raw["selector_scores"],
    )
    if (
        not isinstance(ids, list)
        or not isinstance(ranking, list)
        or not isinstance(scores, list)
    ):
        _fail("decision record", "invalid sequence fields")
    parsed_scores: list[tuple[str, float]] = []
    for item in scores:
        if not isinstance(item, list) or len(item) != 2:
            _fail("decision record.selector_scores", "invalid entry")
        parsed_scores.append((cast(str, item[0]), float(cast(float, item[1]))))
    probability = raw["selected_predicted_failure_probability"]
    result = ClosedLoopDecisionRecordV1(
        episode_execution_id=cast(str, raw["episode_execution_id"]),
        source_trajectory_id=cast(str, raw["source_trajectory_id"]),
        selector_id=cast(str, raw["selector_id"]),
        visual_domain=cast(str, raw["visual_domain"]),
        decision_ordinal=cast(int, raw["decision_ordinal"]),
        nominal_plan_index=cast(int, raw["nominal_plan_index"]),
        pre_decision_state_digest=cast(str, raw["pre_decision_state_digest"]),
        candidate_pool_digest=cast(str, raw["candidate_pool_digest"]),
        ordered_candidate_ids=tuple(cast(Sequence[str], ids)),
        selector_scores=tuple(parsed_scores),
        deterministic_ranking=tuple(cast(Sequence[str], ranking)),
        selected_candidate_id=cast(str, raw["selected_candidate_id"]),
        selected_predicted_failure_probability=(
            None if probability is None else float(cast(float, probability))
        ),
        execution_stride=cast(int, raw["execution_stride"]),
        checkpoint_ensemble_identity=cast(str, raw["checkpoint_ensemble_identity"]),
        visual_packet_identity=cast(str | None, raw["visual_packet_identity"]),
        outcomes_available_during_selection=cast(
            bool, raw["outcomes_available_during_selection"]
        ),
        schema_version=cast(str, raw["schema_version"]),
    )
    if raw["content_digest"] != result.content_digest:
        _fail("decision record.content_digest", "content changed")
    return result


def _load_boundary(value: object, context: str) -> BoundaryLedgerEntryV1:
    if not isinstance(value, Mapping):
        _fail(context, "expected object")
    raw = cast(Mapping[str, object], value)
    expected = {
        "candidate_pool_digest",
        "decision_ordinal",
        "decision_record_digest",
        "executed_step_count",
        "execution_event_ordinal",
        "nominal_plan_index",
        "observed_task_evidence_digest",
        "post_state_digest",
        "post_state_reference",
        "pre_state_digest",
        "pre_state_reference",
        "recovery_count",
        "schema_version",
        "selection_event_ordinal",
        "state",
    }
    _exact(raw, expected, context)
    return BoundaryLedgerEntryV1(
        decision_ordinal=cast(int, raw["decision_ordinal"]),
        nominal_plan_index=cast(int, raw["nominal_plan_index"]),
        state=BoundaryState(cast(str, raw["state"])),
        pre_state_reference=cast(str, raw["pre_state_reference"]),
        pre_state_digest=cast(str, raw["pre_state_digest"]),
        candidate_pool_digest=cast(str | None, raw["candidate_pool_digest"]),
        decision_record_digest=cast(str | None, raw["decision_record_digest"]),
        executed_step_count=cast(int, raw["executed_step_count"]),
        post_state_reference=cast(str | None, raw["post_state_reference"]),
        post_state_digest=cast(str | None, raw["post_state_digest"]),
        observed_task_evidence_digest=cast(
            str | None, raw["observed_task_evidence_digest"]
        ),
        recovery_count=cast(int, raw["recovery_count"]),
        selection_event_ordinal=cast(int | None, raw["selection_event_ordinal"]),
        execution_event_ordinal=cast(int | None, raw["execution_event_ordinal"]),
        schema_version=cast(str, raw["schema_version"]),
    )


def save_episode(record: ClosedLoopEpisodeRecordV1, path: Path) -> Path:
    """Atomically publish the latest transactional episode ledger."""

    if not isinstance(record, ClosedLoopEpisodeRecordV1):
        _fail("save episode", "invalid model")
    return write_atomic_json(
        path, {**record.as_mapping(), "content_digest": record.content_digest}
    )


def load_episode(path: Path) -> ClosedLoopEpisodeRecordV1:
    """Strictly reload and authenticate one episode ledger."""

    raw = _read(path, "episode record")
    expected = {
        "boundaries",
        "completed_event_ordinal",
        "content_digest",
        "episode_execution_id",
        "executed_control_steps",
        "execution_error_type",
        "final_nominal_plan_index",
        "schema_version",
        "selector_id",
        "source_plan_digest",
        "source_trajectory_id",
        "started_event_ordinal",
        "state",
        "tail_fallback_count",
        "visual_domain",
    }
    _exact(raw, expected, "episode record")
    entries = raw["boundaries"]
    if not isinstance(entries, list):
        _fail("episode record.boundaries", "expected list")
    result = ClosedLoopEpisodeRecordV1(
        episode_execution_id=cast(str, raw["episode_execution_id"]),
        source_plan_digest=cast(str, raw["source_plan_digest"]),
        source_trajectory_id=cast(str, raw["source_trajectory_id"]),
        selector_id=cast(str, raw["selector_id"]),
        visual_domain=cast(str, raw["visual_domain"]),
        state=EpisodeState(cast(str, raw["state"])),
        boundaries=tuple(
            _load_boundary(item, f"episode record.boundaries[{index}]")
            for index, item in enumerate(entries)
        ),
        executed_control_steps=cast(int, raw["executed_control_steps"]),
        tail_fallback_count=cast(int, raw["tail_fallback_count"]),
        final_nominal_plan_index=cast(int, raw["final_nominal_plan_index"]),
        started_event_ordinal=cast(int, raw["started_event_ordinal"]),
        completed_event_ordinal=cast(int | None, raw["completed_event_ordinal"]),
        execution_error_type=cast(str | None, raw["execution_error_type"]),
        schema_version=cast(str, raw["schema_version"]),
    )
    if raw["content_digest"] != result.content_digest:
        _fail("episode record.content_digest", "content changed")
    return result


class ClosedLoopEpisodeStore:
    """Filesystem layout for one resumable M4C control episode."""

    def __init__(self, root: Path) -> None:
        """Bind one episode root without creating it prematurely."""

        self.root = Path(root).absolute()

    @property
    def episode_path(self) -> Path:
        """Return the transactional episode-ledger path."""

        return self.root / "episode.json"

    def boundary_dir(self, ordinal: int) -> Path:
        """Return one deterministic decision-boundary directory."""

        if type(ordinal) is not int or ordinal < 0:
            _fail("boundary ordinal", "expected non-negative int")
        return self.root / "boundaries" / f"{ordinal:06d}"

    def save_pool(self, pool: ClosedLoopCandidatePoolV1) -> Path:
        """Persist a pool once before selector execution."""

        return save_candidate_pool(
            pool, self.boundary_dir(pool.decision_ordinal) / "candidate-pool.json"
        )

    def load_pool(self, ordinal: int) -> ClosedLoopCandidatePoolV1:
        """Reload an already finalized candidate pool."""

        return load_candidate_pool(self.boundary_dir(ordinal) / "candidate-pool.json")

    def save_decision(self, record: ClosedLoopDecisionRecordV1) -> Path:
        """Persist a selection once before any action executes."""

        return save_decision(
            record,
            self.boundary_dir(record.decision_ordinal) / "decision-record.json",
        )

    def load_decision(self, ordinal: int) -> ClosedLoopDecisionRecordV1:
        """Reload an immutable selection for recovery or inspection."""

        return load_decision(self.boundary_dir(ordinal) / "decision-record.json")

    def save_episode(self, record: ClosedLoopEpisodeRecordV1) -> Path:
        """Publish the latest ledger state transactionally."""

        return save_episode(record, self.episode_path)

    def load_episode(self) -> ClosedLoopEpisodeRecordV1:
        """Reload the latest episode state."""

        return load_episode(self.episode_path)


__all__ = [
    "ClosedLoopEpisodeStore",
    "ClosedLoopSerializationError",
    "load_candidate_pool",
    "load_decision",
    "load_episode",
    "save_candidate_pool",
    "save_decision",
    "save_episode",
    "write_atomic_json",
    "write_exclusive_json",
]

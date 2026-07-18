"""Content-bound M4C source-plan preparation and strict reload."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from latentguard.control.models import SourcePlanIdentityV1, content_digest
from latentguard.control.runner import BoundSourcePlanV1
from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
    PickCubeStateIndexedArchiveV1,
)

SOURCE_PLAN_MANIFEST_FORMAT = "latentguard-m4c-source-plan-manifest-v1"


def _source_plan_for_episode(episode: Any) -> BoundSourcePlanV1:
    actions = episode.source_actions
    if actions.dtype != np.dtype("<f8"):
        raise ValueError("M4C source actions must remain exact little-endian float64")
    state_digests = tuple(item.state_digest for item in episode.states)
    identity = SourcePlanIdentityV1(
        source_trajectory_id=episode.source_trajectory_id,
        split_group_id=episode.source_trajectory_id,
        reset_seed=episode.seed,
        source_action_digest=episode.source_action_digest,
        initial_state_digest=state_digests[0],
        complete_state_tree_digests=state_digests,
        independent_replay_success=True,
        planner_identity=episode.source_policy_identity,
        compatibility_identity=episode.compatibility_identity,
    )
    return BoundSourcePlanV1(
        identity=identity,
        actions=actions,
        initial_state_reference=(
            f"m4c-source:{episode.source_trajectory_id}:state-index-0"
        ),
    )


def require_source_plan_disjointness(
    current: PickCubeStateIndexedArchiveV1,
    prior_archives: Sequence[PickCubeStateIndexedArchiveV1],
) -> None:
    """Reject identity, seed, split-group, action, or state overlap."""

    current_plans = tuple(_source_plan_for_episode(item) for item in current.episodes)
    prior_plans = tuple(
        _source_plan_for_episode(item)
        for archive in prior_archives
        for item in archive.episodes
    )
    check_sets: tuple[tuple[str, set[object], set[object]], ...] = (
        (
            "source trajectory ID",
            {item.identity.source_trajectory_id for item in current_plans},
            {item.identity.source_trajectory_id for item in prior_plans},
        ),
        (
            "reset seed",
            {item.identity.reset_seed for item in current_plans},
            {item.identity.reset_seed for item in prior_plans},
        ),
        (
            "split-group ID",
            {item.identity.split_group_id for item in current_plans},
            {item.identity.split_group_id for item in prior_plans},
        ),
        (
            "source action digest",
            {item.identity.source_action_digest for item in current_plans},
            {item.identity.source_action_digest for item in prior_plans},
        ),
    )
    for label, current_values, prior_values in check_sets:
        if len(current_values) != len(current_plans):
            raise ValueError(f"M4C source plans duplicate current {label}")
        if current_values.intersection(prior_values):
            raise ValueError(f"M4C source plans overlap prior data by {label}")
    state_owner: dict[str, str] = {}
    for item in current_plans:
        for digest in set(item.identity.complete_state_tree_digests):
            owner = state_owner.setdefault(digest, item.identity.source_trajectory_id)
            if owner != item.identity.source_trajectory_id:
                raise ValueError(
                    "M4C complete state-tree digest overlaps current trajectories"
                )
    current_states = {
        digest
        for item in current_plans
        for digest in item.identity.complete_state_tree_digests
    }
    prior_states = {
        digest
        for item in prior_plans
        for digest in item.identity.complete_state_tree_digests
    }
    if current_states.intersection(prior_states):
        raise ValueError("M4C complete state-tree digest overlaps prior data")


def prepare_source_plans(
    archive: PickCubeStateIndexedArchiveV1,
    *,
    prior_archives: Sequence[PickCubeStateIndexedArchiveV1],
    expected_count: int,
) -> tuple[BoundSourcePlanV1, ...]:
    """Validate a new independently replayed successful source collection."""

    if len(archive.episodes) != expected_count:
        raise ValueError("M4C accepted source-plan count differs from protocol")
    require_source_plan_disjointness(archive, prior_archives)
    plans = tuple(_source_plan_for_episode(item) for item in archive.episodes)
    if any(item.actions.shape[0] < 16 for item in plans):
        raise ValueError("M4C source plan is shorter than the fixed horizon")
    if len({item.identity.content_digest for item in plans}) != len(plans):
        raise ValueError("M4C source plans contain duplicate identities")
    return plans


def source_plan_set_digest(plans: Sequence[BoundSourcePlanV1]) -> str:
    """Return the ordered semantic source-set digest."""

    values = tuple(plans)
    if not values:
        raise ValueError("M4C source-plan set cannot be empty")
    return content_digest(
        {
            "schema_version": "1.0",
            "source_plan_digests": [item.identity.content_digest for item in values],
        },
        context="M4CSourcePlanSetV1",
    )


def save_source_plan_manifest(
    plans: Sequence[BoundSourcePlanV1],
    path: Path,
    *,
    archive_content_digest: str,
) -> Path:
    """Publish one compact identity-only manifest with exclusive creation."""

    values = tuple(plans)
    payload: dict[str, object] = {
        "archive_content_digest": archive_content_digest,
        "format": SOURCE_PLAN_MANIFEST_FORMAT,
        "schema_version": "1.0",
        "source_plan_count": len(values),
        "source_plan_set_digest": source_plan_set_digest(values),
        "source_plans": [item.identity.as_mapping() for item in values],
    }
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return destination


def load_source_plans(
    path: Path,
    *,
    archive: PickCubeStateIndexedArchiveV1,
) -> tuple[BoundSourcePlanV1, ...]:
    """Strictly reload identities and bind them to exact archived actions."""

    raw = cast(object, json.loads(Path(path).read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError("M4C source manifest must be an object")
    expected = {
        "archive_content_digest",
        "format",
        "schema_version",
        "source_plan_count",
        "source_plan_set_digest",
        "source_plans",
    }
    if set(raw) != expected or raw.get("format") != SOURCE_PLAN_MANIFEST_FORMAT:
        raise ValueError("M4C source manifest fields or format differ")
    if raw.get("archive_content_digest") != archive.content_digest:
        raise ValueError("M4C source archive content differs from its manifest")
    plans = tuple(_source_plan_for_episode(item) for item in archive.episodes)
    if raw.get("source_plan_count") != len(plans):
        raise ValueError("M4C source manifest count differs")
    if raw.get("source_plans") != [item.identity.as_mapping() for item in plans]:
        raise ValueError("M4C source plan identity inventory differs")
    if raw.get("source_plan_set_digest") != source_plan_set_digest(plans):
        raise ValueError("M4C source plan set digest differs")
    return plans


__all__ = [
    "SOURCE_PLAN_MANIFEST_FORMAT",
    "load_source_plans",
    "prepare_source_plans",
    "require_source_plan_disjointness",
    "save_source_plan_manifest",
    "source_plan_set_digest",
]

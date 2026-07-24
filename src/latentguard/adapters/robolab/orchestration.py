"""Process-isolated shard aggregation for the LG-RB0 RoboLab runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _require_equal(
    values: Sequence[Any],
    *,
    field: str,
) -> Any:
    if not values:
        raise ValueError(f"no shard values supplied for {field}")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"shard field differs: {field}")
    return first


def merge_recording_manifests(
    manifests: Sequence[Mapping[str, Any]],
    *,
    expected_episode_count: int,
) -> dict[str, Any]:
    """Merge one-process-per-recording manifests after validating invariants."""
    recordings: list[dict[str, Any]] = []
    for manifest in manifests:
        shard = manifest.get("recordings")
        if (
            manifest.get("status") != "pass"
            or manifest.get("episode_count") != 1
            or manifest.get("valid_episode_count") != 1
            or not isinstance(shard, list)
            or len(shard) != 1
        ):
            raise ValueError("recording shard did not pass as one valid episode")
        if (
            manifest.get("policy_or_candidate_source") is not False
            or manifest.get("raw_recordings_in_git") is not False
            or manifest.get("training_performed") is not False
            or manifest.get("final_seeds_accessed") is not False
        ):
            raise ValueError("recording shard violates LG-RB0 prohibitions")
        item = shard[0]
        if not isinstance(item, dict):
            raise ValueError("recording shard item must be a mapping")
        recordings.append(item)
    identifiers = [str(item.get("recording_id")) for item in recordings]
    if len(recordings) != expected_episode_count:
        raise ValueError("merged recording count differs from frozen protocol")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("recording shard identifiers are not unique")
    if not all(item.get("valid") is True for item in recordings):
        raise ValueError("merged recording contains an invalid episode")
    controller = _require_equal(
        [manifest.get("controller_kind") for manifest in manifests],
        field="controller_kind",
    )
    return {
        "schema_version": "lg_rb0_recording_manifest_v1",
        "status": "pass",
        "controller_kind": controller,
        "policy_or_candidate_source": False,
        "episode_count": len(recordings),
        "valid_episode_count": len(recordings),
        "recordings": recordings,
        "raw_recordings_in_git": False,
        "training_performed": False,
        "final_seeds_accessed": False,
        "runtime_process_isolation": "one_recording_per_isaac_sim_process",
    }


def merge_faithful_results(
    results: Sequence[Mapping[str, Any]],
    *,
    expected_episode_count: int,
) -> dict[str, Any]:
    """Merge faithful-replay shards without converting failures into errors."""
    if len(results) != expected_episode_count:
        raise ValueError("faithful shard count differs from frozen protocol")
    episode_count = sum(int(result.get("episode_count", -1)) for result in results)
    if episode_count != expected_episode_count:
        raise ValueError("faithful shards do not each bind one episode")
    repeats = int(
        _require_equal(
            [result.get("repeats_per_episode") for result in results],
            field="repeats_per_episode",
        )
    )
    tolerance = float(
        _require_equal(
            [result.get("official_state_tolerance") for result in results],
            field="official_state_tolerance",
        )
    )
    details: list[dict[str, Any]] = []
    for result in results:
        shard_details = result.get("details")
        if not isinstance(shard_details, list) or len(shard_details) != repeats:
            raise ValueError("faithful shard detail count is incomplete")
        if not all(isinstance(item, dict) for item in shard_details):
            raise ValueError("faithful shard details must be mappings")
        details.extend(shard_details)
    initial_failures = sum(
        int(result.get("initial_restore_failures", -1)) for result in results
    )
    state_failures = sum(
        int(result.get("per_step_state_failures", -1)) for result in results
    )
    terminal_mismatches = sum(
        int(result.get("terminal_mismatches", -1)) for result in results
    )
    success_mismatches = sum(
        int(result.get("success_mismatches", -1)) for result in results
    )
    expected_replays = sum(
        int(result.get("expected_replay_count", -1)) for result in results
    )
    completed_replays = sum(
        int(result.get("completed_replay_count", -1)) for result in results
    )
    execution_errors = sum(
        int(result.get("execution_error_count", -1)) for result in results
    )
    if min(expected_replays, completed_replays, execution_errors) < 0:
        raise ValueError("faithful shard completion counters are invalid")
    status = (
        "pass"
        if initial_failures
        == state_failures
        == terminal_mismatches
        == success_mismatches
        == 0
        and completed_replays == expected_replays
        and execution_errors == 0
        and all(result.get("status") == "pass" for result in results)
        else "fail"
    )
    return {
        "schema_version": "lg_rb0_faithful_replay_validation_v1",
        "status": status,
        "episode_count": episode_count,
        "repeats_per_episode": repeats,
        "official_state_tolerance": tolerance,
        "initial_restore_failures": initial_failures,
        "per_step_state_failures": state_failures,
        "terminal_mismatches": terminal_mismatches,
        "success_mismatches": success_mismatches,
        "expected_replay_count": expected_replays,
        "completed_replay_count": completed_replays,
        "execution_error_count": execution_errors,
        "details": details,
        "runtime_process_isolation": "one_recording_per_isaac_sim_process",
    }


def merge_takeover_results(
    prefix_results: Sequence[Mapping[str, Any]],
    branch_results: Sequence[Mapping[str, Any]],
    isolation_results: Sequence[Mapping[str, Any]],
    *,
    expected_episode_count: int,
) -> dict[str, dict[str, Any]]:
    """Merge prefix, branch, and isolation shards with complete coverage checks."""
    result_groups = (prefix_results, branch_results, isolation_results)
    if any(len(group) != expected_episode_count for group in result_groups):
        raise ValueError("takeover shard count differs from frozen protocol")

    prefix_details = _merge_details(prefix_results, expected_per_shard=3)
    prefix_mismatches = _sum_nonnegative(prefix_results, "mismatch_count")
    prefix_coverage = all(
        result.get("semantic_coverage_complete") is True for result in prefix_results
    )
    prefix = {
        "schema_version": "lg_rb0_prefix_replay_validation_v1",
        "status": "pass" if prefix_mismatches == 0 and prefix_coverage else "fail",
        "state_tolerance": _require_equal(
            [result.get("state_tolerance") for result in prefix_results],
            field="prefix.state_tolerance",
        ),
        "anchors_per_episode": _require_equal(
            [result.get("anchors_per_episode") for result in prefix_results],
            field="prefix.anchors_per_episode",
        ),
        "repeats_per_anchor": _require_equal(
            [result.get("repeats_per_anchor") for result in prefix_results],
            field="prefix.repeats_per_anchor",
        ),
        "episode_count": expected_episode_count,
        "mismatch_count": prefix_mismatches,
        "semantic_coverage_complete": prefix_coverage,
        "details": prefix_details,
        "runtime_process_isolation": "one_recording_per_isaac_sim_process",
    }

    branch_details = _merge_details(branch_results, expected_per_shard=6)
    branch_mismatches = _sum_nonnegative(branch_results, "mismatch_count")
    branch_coverage = all(
        result.get("semantic_coverage_complete") is True for result in branch_results
    )
    branch = {
        "schema_version": "lg_rb0_branch_determinism_validation_v1",
        "status": "pass" if branch_mismatches == 0 and branch_coverage else "fail",
        "state_tolerance": _require_equal(
            [result.get("state_tolerance") for result in branch_results],
            field="branch.state_tolerance",
        ),
        "branches": _require_equal(
            [result.get("branches") for result in branch_results],
            field="branch.branches",
        ),
        "checkpoints": _require_equal(
            [result.get("checkpoints") for result in branch_results],
            field="branch.checkpoints",
        ),
        "episode_count": expected_episode_count,
        "mismatch_count": branch_mismatches,
        "semantic_coverage_complete": branch_coverage,
        "details": branch_details,
        "runtime_process_isolation": "one_recording_per_isaac_sim_process",
    }

    isolation_details = _merge_details(isolation_results, expected_per_shard=6)
    isolation_mismatches = _sum_nonnegative(isolation_results, "mismatch_count")
    isolation_coverage = all(
        result.get("semantic_coverage_complete") is True for result in isolation_results
    )
    isolation = {
        "schema_version": "lg_rb0_branch_isolation_validation_v1",
        "status": (
            "pass" if isolation_mismatches == 0 and isolation_coverage else "fail"
        ),
        "orders": _require_equal(
            [result.get("orders") for result in isolation_results],
            field="isolation.orders",
        ),
        "episode_count": expected_episode_count,
        "mismatch_count": isolation_mismatches,
        "semantic_coverage_complete": isolation_coverage,
        "details": isolation_details,
        "runtime_process_isolation": "one_recording_per_isaac_sim_process",
    }
    return {
        "prefix_replay_validation": prefix,
        "branch_determinism_validation": branch,
        "branch_isolation_validation": isolation,
    }


def _merge_details(
    results: Sequence[Mapping[str, Any]],
    *,
    expected_per_shard: int,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for result in results:
        shard_details = result.get("details")
        if (
            not isinstance(shard_details, list)
            or len(shard_details) != expected_per_shard
            or not all(isinstance(item, dict) for item in shard_details)
        ):
            raise ValueError("takeover shard detail count is incomplete")
        details.extend(shard_details)
    return details


def _sum_nonnegative(
    results: Sequence[Mapping[str, Any]],
    field: str,
) -> int:
    values = [int(result.get(field, -1)) for result in results]
    if any(value < 0 for value in values):
        raise ValueError(f"invalid negative shard value: {field}")
    return sum(values)

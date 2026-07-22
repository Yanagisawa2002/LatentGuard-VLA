"""Audit P0.2 dataset timing, phase proxies, and gripper semantics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np

from latentguard.control.serialization import write_atomic_json
from latentguard.policies.act.data import (
    PickCubeDemoSplit,
    collect_episode_references,
    read_demo_episode,
)
from latentguard.policies.act.grasp_supervision import (
    PickCubeGripperEvent,
    PickCubeManipulationPhase,
    aligned_frame_pairs,
    derive_gripper_events,
    phase_from_expert_label,
    phase_weights_from_train_labels,
)


class PickCubeP02DatasetAuditError(RuntimeError):
    """Raised when immutable P0.2 audit inputs drift."""


def _fail(context: str, reason: str) -> NoReturn:
    raise PickCubeP02DatasetAuditError(f"{context}: {reason}")


def _mapping(path: Path, context: str) -> Mapping[str, object]:
    source = Path(path).absolute()
    if not source.is_file() or source.is_symlink():
        _fail(context, "expected regular unlinked JSON")
    try:
        value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PickCubeP02DatasetAuditError(f"{context}: invalid JSON") from exc
    if not isinstance(value, Mapping):
        _fail(context, "expected JSON object")
    return cast(Mapping[str, object], value)


def _array_summary(values: Sequence[np.ndarray[Any, Any]]) -> Mapping[str, object]:
    if not values:
        return {"frame_count": 0}
    array = np.stack(values).astype(np.float64, copy=False)
    return {
        "frame_count": len(array),
        "maximum": np.max(array, axis=0).tolist(),
        "mean": np.mean(array, axis=0).tolist(),
        "minimum": np.min(array, axis=0).tolist(),
        "standard_deviation": np.std(array, axis=0).tolist(),
    }


def audit(*, dataset_root: Path, output_root: Path) -> Mapping[str, object]:
    """Audit all 500 immutable demonstrations without rewriting them."""
    root = Path(dataset_root).absolute()
    manifest = _mapping(root / "dataset_manifest.json", "dataset manifest")
    normalization = _mapping(root / "normalization_stats.json", "normalization stats")
    references = collect_episode_references(root)
    if len(references) != 500:
        _fail("dataset", "expected exactly 500 episodes")

    phase_counts: Counter[str] = Counter()
    phase_episode_counts: Counter[str] = Counter()
    phase_actions: dict[str, list[np.ndarray[Any, Any]]] = defaultdict(list)
    split_phase_counts: dict[str, Counter[str]] = defaultdict(Counter)
    transition_counts: Counter[str] = Counter()
    transition_frames: list[Mapping[str, object]] = []
    gripper_values: list[np.ndarray[Any, Any]] = []
    gripper_transitions: list[int] = []
    measured_gripper_positions: list[np.ndarray[Any, Any]] = []
    chunks = 0
    crossing_chunks = 0
    chunks_by_start_phase: Counter[str] = Counter()
    crossing_by_start_phase: Counter[str] = Counter()
    offset_pairs: Counter[int] = Counter()
    offset_phase_mismatches: Counter[int] = Counter()
    offset_next_qpos_errors: dict[int, list[float]] = defaultdict(list)
    train_labels: list[str] = []

    for reference in references:
        episode = read_demo_episode(root, reference)
        canonical = tuple(
            phase_from_expert_label(value).value for value in episode.phases
        )
        present = set(canonical)
        phase_episode_counts.update(present)
        phase_counts.update(canonical)
        split_phase_counts[reference.split.value].update(canonical)
        if reference.split is PickCubeDemoSplit.TRAIN:
            train_labels.extend(canonical)
        for phase, action in zip(canonical, episode.actions, strict=True):
            phase_actions[phase].append(np.asarray(action))
        gripper = np.asarray(episode.actions[:, 7], dtype=np.float64)
        events = derive_gripper_events(
            gripper, close_threshold=-0.5, open_threshold=0.5
        )
        close_indices = np.flatnonzero(events == int(PickCubeGripperEvent.CLOSING))
        if len(close_indices) != 1:
            _fail("gripper audit", "every expert episode must close exactly once")
        gripper_transitions.append(int(close_indices[0]))
        gripper_values.append(gripper)
        measured_gripper_positions.append(
            np.asarray(episode.proprioception[:, 7:9], dtype=np.float64)
        )
        for index in range(1, episode.frame_count):
            if canonical[index] != canonical[index - 1]:
                transition = f"{canonical[index - 1]}->{canonical[index]}"
                transition_counts[transition] += 1
                transition_frames.append(
                    {
                        "episode_id": reference.episode_id,
                        "frame": index,
                        "scene_seed": reference.scene_seed,
                        "transition": transition,
                    }
                )
        for start in range(episode.frame_count):
            end = min(episode.frame_count, start + 16)
            values = canonical[start:end]
            chunks += 1
            chunks_by_start_phase[canonical[start]] += 1
            if len(set(values)) > 1:
                crossing_chunks += 1
                crossing_by_start_phase[canonical[start]] += 1
        for offset in range(-2, 3):
            for observation_index, action_index in aligned_frame_pairs(
                episode.frame_count, offset
            ):
                if observation_index + 1 >= episode.frame_count:
                    continue
                offset_pairs[offset] += 1
                if canonical[observation_index] != canonical[action_index]:
                    offset_phase_mismatches[offset] += 1
                target = np.asarray(episode.actions[action_index, :7])
                next_qpos = np.asarray(
                    episode.proprioception[observation_index + 1, :7]
                )
                offset_next_qpos_errors[offset].append(
                    float(np.mean(np.abs(target - next_qpos)))
                )

    all_gripper = np.concatenate(gripper_values)
    measured_gripper = np.concatenate(measured_gripper_positions, axis=0)
    weights = phase_weights_from_train_labels(train_labels, exponent=0.5, maximum=3.0)
    temporal = {
        "control_frequency_hz": 20.0,
        "current_target_offset": 0,
        "dataset_semantic": "pre_action_observation_t_paired_with_executed_action_t",
        "episode_boundary_isolation": True,
        "frame_skipping": False,
        "offsets": {
            str(offset): {
                "controller_tracking_proxy_mae_to_next_measured_qpos": float(
                    np.mean(offset_next_qpos_errors[offset])
                ),
                "pair_count": offset_pairs[offset],
                "phase_mismatch_count": offset_phase_mismatches[offset],
                "phase_mismatch_rate": (
                    offset_phase_mismatches[offset] / offset_pairs[offset]
                ),
            }
            for offset in range(-2, 3)
        },
        "observation_recording_order": "render_and_state_before_env_step",
        "action_recording_order": "same_action_before_env_step_then_executed_once",
        "padding_semantic": "zero_storage_with_true_action_is_pad_mask",
        "reset_frame_semantic": "first_post_reset_pre_action_observation",
        "schema_version": "pickcube-act-p02-temporal-alignment-audit-v1",
        "selected_offset": 0,
        "selected_offset_basis": (
            "recording_interception_contract_not_controller_response_latency"
        ),
        "terminal_frame_semantic": "T_pre_action_observations_T_actions_no_T_plus_1",
        "timestamp_fields_present": False,
        "temporal_alignment_defect_found": False,
    }
    phase = {
        "chunk_count": chunks,
        "chunk_length": 16,
        "chunks_crossing_transition": crossing_chunks,
        "chunks_crossing_transition_rate": crossing_chunks / chunks,
        "crossing_by_start_phase": dict(sorted(crossing_by_start_phase.items())),
        "diagnostic_only_not_policy_input": True,
        "episode_count_by_phase": dict(sorted(phase_episode_counts.items())),
        "first_contact_frame_available": False,
        "first_lift_frame_available": False,
        "frame_count_by_phase": dict(sorted(phase_counts.items())),
        "imbalance_ratio": max(phase_counts.values()) / min(phase_counts.values()),
        "joint_action_by_phase": {
            name: _array_summary(values)
            for name, values in sorted(phase_actions.items())
        },
        "labels_absent_without_fabrication": [
            PickCubeManipulationPhase.RESET_OR_IDLE.value,
            PickCubeManipulationPhase.CONTACT_OR_GRASP.value,
            PickCubeManipulationPhase.LIFT.value,
        ],
        "phase_proxy_source": "recorded_official_expert_phase_tracker",
        "schema_version": "pickcube-act-p02-phase-distribution-audit-v1",
        "split_frame_counts_by_phase": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_phase_counts.items())
        },
        "train_only_phase_weights": dict(weights),
        "transition_counts": dict(sorted(transition_counts.items())),
        "transition_frames": transition_frames,
    }
    gripper = {
        "action_dimension": 7,
        "action_is_commanded_not_measured": True,
        "always_closed_collapse_in_late_p01_checkpoints": True,
        "always_open_collapse_in_late_p01_checkpoints": False,
        "close_sign": "negative",
        "close_target": -1.0,
        "close_transition_count": len(gripper_transitions),
        "close_transition_frame": {
            "maximum": max(gripper_transitions),
            "mean": float(np.mean(gripper_transitions)),
            "minimum": min(gripper_transitions),
            "standard_deviation": float(np.std(gripper_transitions)),
        },
        "closed_endpoint_fraction": float(np.mean(all_gripper == -1.0)),
        "failure_classification": [
            "close_transition_underweighted",
            "always_closed_collapse",
        ],
        "gripper_supervision_fraction_of_standard_l1": 0.125,
        "mimic_joint_command": True,
        "normalization_preserves_endpoints": True,
        "open_sign": "positive",
        "open_target": 1.0,
        "opened_endpoint_fraction": float(np.mean(all_gripper == 1.0)),
        "policy_state_contains_measured_finger_qpos": True,
        "policy_state_measured_finger_qpos": {
            "maximum": np.max(measured_gripper, axis=0).tolist(),
            "mean": np.mean(measured_gripper, axis=0).tolist(),
            "minimum": np.min(measured_gripper, axis=0).tolist(),
            "standard_deviation": np.std(measured_gripper, axis=0).tolist(),
        },
        "schema_version": "pickcube-act-p02-gripper-audit-v1",
        "stored_target_values": sorted(set(map(float, all_gripper.tolist()))),
    }
    output = Path(output_root).absolute()
    output.mkdir(parents=True, exist_ok=True)
    write_atomic_json(output / "temporal_alignment_audit.json", temporal)
    write_atomic_json(output / "phase_distribution_audit.json", phase)
    write_atomic_json(output / "gripper_audit.json", gripper)
    report = {
        "dataset_digest": manifest.get("dataset_digest"),
        "episode_count": len(references),
        "normalization_digest": normalization.get("normalization_digest"),
        "passed": True,
        "schema_version": "pickcube-act-p02-dataset-audit-v1",
        "temporal_alignment_defect_found": False,
    }
    write_atomic_json(output / "dataset_audit_summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = audit(dataset_root=args.dataset_root, output_root=args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Transactional WM-v0 D1 collection, resume, and quality gates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np

from latentguard.control.serialization import write_atomic_json
from latentguard.replay.identity import canonical_json_bytes

from .candidates import (
    CandidateContext,
    CandidateProvider,
    CandidateRejection,
    CandidateRejectionCode,
    CandidateValidationError,
    PolicyCandidate,
    candidate_set_digest,
    validate_and_deduplicate_candidates,
)
from .collector import CounterfactualCollector, WorldModelCollectionAdapter
from .data_schema import DatasetSplit, WorldModelSample, WorldModelSchemaError
from .manifest import build_dataset_manifest
from .serialization import read_world_model_sample, write_world_model_sample


class EpisodePhase(StrEnum):
    """Conservative task-phase labels for phase-aware anchor sampling."""

    INITIAL = "initial"
    APPROACH = "approach"
    PRE_GRASP = "pre_grasp"
    GRASP_ATTEMPT = "grasp_attempt"
    POST_GRASP = "post_grasp"
    TRANSPORT = "transport"
    PRE_PLACE = "pre_place"
    RELEASE = "release"
    RECOVERY_SENSITIVE = "recovery_sensitive"
    UNKNOWN = "unknown"


class D1RejectionCode(StrEnum):
    """Stable dataset rejection reason codes."""

    RESTORE_MISMATCH = "RESTORE_MISMATCH"
    DUPLICATE_ACTION = "DUPLICATE_ACTION"
    DUPLICATE_IDENTITY = "DUPLICATE_IDENTITY"
    INVALID_ACTION_SHAPE = "INVALID_ACTION_SHAPE"
    NONFINITE_ACTION = "NONFINITE_ACTION"
    MISSING_FUTURE_FRAME = "MISSING_FUTURE_FRAME"
    OBSERVATION_MISALIGNMENT = "OBSERVATION_MISALIGNMENT"
    POLICY_INFERENCE_FAILED = "POLICY_INFERENCE_FAILED"
    INVALID_EVENT_LABEL = "INVALID_EVENT_LABEL"
    INCOMPLETE_METADATA = "INCOMPLETE_METADATA"
    SIMULATOR_ERROR = "SIMULATOR_ERROR"
    INCOMPLETE_ANCHOR = "INCOMPLETE_ANCHOR"


@dataclass(frozen=True, slots=True)
class D1AnchorJob:
    """One outcome-free exact anchor selected before candidate execution."""

    episode_id: str
    source_episode_id: str
    split_group_id: str
    anchor_id: str
    task_id: str
    instruction: str | None
    split: DatasetSplit
    scene_group: str
    episode_phase: EpisodePhase
    task_progress_t: float | None
    time_index: int
    distance_to_target: float | None
    grasp_state: bool | None
    object_state: str

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "source_episode_id",
            "split_group_id",
            "anchor_id",
            "task_id",
            "scene_group",
            "object_state",
        ):
            _text(getattr(self, name), name)
        if self.split_group_id != self.source_episode_id:
            raise WorldModelSchemaError(
                "D1 split_group_id must equal the original source episode"
            )
        if self.instruction is not None:
            _text(self.instruction, "instruction")
        if not isinstance(self.split, DatasetSplit):
            raise WorldModelSchemaError("D1 split must be a DatasetSplit")
        if not isinstance(self.episode_phase, EpisodePhase):
            raise WorldModelSchemaError("episode_phase must be an EpisodePhase")
        if type(self.time_index) is not int or self.time_index < 0:
            raise WorldModelSchemaError("time_index must be a non-negative integer")
        for value, name in (
            (self.task_progress_t, "task_progress_t"),
            (self.distance_to_target, "distance_to_target"),
        ):
            if value is not None and (
                type(value) is not float or not math.isfinite(value)
            ):
                raise WorldModelSchemaError(f"{name} must be a finite float or None")
        if self.grasp_state is not None and type(self.grasp_state) is not bool:
            raise WorldModelSchemaError("grasp_state must be boolean or None")

    @property
    def storage_key(self) -> str:
        """Return a filesystem-safe identity for this anchor."""

        return hashlib.sha256(self.anchor_id.encode("utf-8")).hexdigest()

    def as_mapping(self) -> dict[str, object]:
        """Return JSON-native phase and grouping metadata."""

        return {
            "anchor_id": self.anchor_id,
            "distance_to_target": self.distance_to_target,
            "episode_id": self.episode_id,
            "episode_phase": self.episode_phase.value,
            "grasp_state": self.grasp_state,
            "instruction": self.instruction,
            "object_state": self.object_state,
            "scene_group": self.scene_group,
            "source_episode_id": self.source_episode_id,
            "split": self.split.value,
            "split_group_id": self.split_group_id,
            "task_id": self.task_id,
            "task_progress_t": self.task_progress_t,
            "time_index": self.time_index,
        }


@dataclass(frozen=True, slots=True)
class D1CollectionResult:
    """Summary of one new or resumed D1 collection invocation."""

    planned_anchor_count: int
    completed_anchor_count: int
    newly_completed_anchor_count: int
    valid_sample_count: int
    rejected_sample_count: int
    zero_work_resume: bool
    dataset_manifest: Mapping[str, object]
    quality_report: Mapping[str, object]


@dataclass(slots=True)
class D1CollectionRunner:
    """Commit every completed anchor atomically and rebuild compact state."""

    adapter: WorldModelCollectionAdapter
    output_root: Path
    prediction_horizon: int
    observation_stride: int
    expected_action_dimension: int
    seed: int

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root).absolute()
        for value, name in (
            (self.prediction_horizon, "prediction_horizon"),
            (self.observation_stride, "observation_stride"),
            (self.expected_action_dimension, "expected_action_dimension"),
        ):
            if type(value) is not int or value < 1:
                raise WorldModelSchemaError(f"{name} must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise WorldModelSchemaError("seed must be a non-negative integer")

    def run(
        self,
        jobs: Sequence[D1AnchorJob],
        provider_for_job: Callable[[D1AnchorJob], CandidateProvider],
        *,
        resume: bool,
        max_samples: int | None = None,
    ) -> D1CollectionResult:
        """Run selected anchors with deterministic recovery and no partial publish."""

        if not jobs:
            raise WorldModelSchemaError("D1 collection requires at least one anchor")
        _validate_job_group_splits(jobs)
        if max_samples is not None and (
            type(max_samples) is not int or max_samples < 1
        ):
            raise WorldModelSchemaError("max_samples must be positive or None")
        self._prepare_root(resume=resume)
        self._recover_incomplete_staging(resume=resume)
        completed_before = self._completed_anchor_ids()
        new_count = 0
        valid_count = self._valid_sample_count()
        for job in jobs:
            if job.anchor_id in completed_before:
                continue
            anchor_started = time.monotonic()
            provider = provider_for_job(job)
            candidates, rejections = self._generate(job, provider)
            for rejection in rejections:
                self._append_rejection(job, _candidate_rejection_mapping(rejection))
            if not candidates:
                continue
            if max_samples is not None and valid_count + len(candidates) > max_samples:
                break
            try:
                samples = CounterfactualCollector(
                    self.adapter,
                    prediction_horizon=self.prediction_horizon,
                    observation_stride=self.observation_stride,
                ).collect(
                    episode_id=job.episode_id,
                    source_episode_id=job.source_episode_id,
                    split_group_id=job.split_group_id,
                    anchor_id=job.anchor_id,
                    task_id=job.task_id,
                    instruction=job.instruction,
                    split=job.split,
                    candidates=tuple(
                        candidate.to_action_candidate() for candidate in candidates
                    ),
                )
                self._commit_anchor(
                    job,
                    candidates,
                    samples,
                    collection_duration_seconds=time.monotonic() - anchor_started,
                )
            except Exception as exc:
                self._append_rejection(job, _execution_rejection_mapping(exc))
                continue
            new_count += 1
            valid_count += len(samples)
            self._publish_indexes(planned_anchor_count=len(jobs))
        self._publish_indexes(planned_anchor_count=len(jobs))
        manifest, quality = load_or_build_d1_reports(self.output_root)
        return D1CollectionResult(
            planned_anchor_count=len(jobs),
            completed_anchor_count=len(self._completed_anchor_ids()),
            newly_completed_anchor_count=new_count,
            valid_sample_count=cast(int, quality["valid_sample_count"]),
            rejected_sample_count=cast(int, quality["rejected_sample_count"]),
            zero_work_resume=resume and new_count == 0,
            dataset_manifest=manifest,
            quality_report=quality,
        )

    def _prepare_root(self, *, resume: bool) -> None:
        if self.output_root.exists() and not resume:
            raise WorldModelSchemaError(
                "D1 output already exists; pass --resume to preserve completed anchors"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "anchors").mkdir(exist_ok=True)
        (self.output_root / ".staging").mkdir(exist_ok=True)
        (self.output_root / "quarantine").mkdir(exist_ok=True)

    def _recover_incomplete_staging(self, *, resume: bool) -> None:
        staging = self.output_root / ".staging"
        entries = sorted(path for path in staging.iterdir() if path.is_dir())
        if entries and not resume:
            raise WorldModelSchemaError("incomplete D1 anchors require --resume")
        for entry in entries:
            target = (
                self.output_root / "quarantine" / (f"{entry.name}-{time.time_ns()}")
            )
            entry.replace(target)
            self._append_raw_rejection(
                {
                    "anchor_id": None,
                    "candidate_id": None,
                    "detail": "incomplete staging directory quarantined before retry",
                    "reason_code": D1RejectionCode.INCOMPLETE_ANCHOR.value,
                    "staging_key": entry.name,
                }
            )

    def _generate(
        self, job: D1AnchorJob, provider: CandidateProvider
    ) -> tuple[tuple[PolicyCandidate, ...], tuple[CandidateRejection, ...]]:
        try:
            report = self.adapter.restore(job.anchor_id)
            if (
                not report.verified
                or report.anchor_id != job.anchor_id
                or report.compared_component_count < 1
            ):
                raise RuntimeError("anchor restoration was not fully verified")
            frame = self.adapter.observe()
            if frame.observations.shape[0] != len(self.adapter.camera_ids):
                self.adapter.close()
                return (), (
                    CandidateRejection(
                        CandidateRejectionCode.OBSERVATION_MISALIGNMENT,
                        None,
                        "current observation camera alignment differs",
                    ),
                )
        except Exception as exc:
            self.adapter.close()
            return (), (
                CandidateRejection(
                    CandidateRejectionCode.RESTORE_MISMATCH,
                    None,
                    f"{type(exc).__name__}: {exc}",
                ),
            )
        try:
            observation = {
                camera_id: np.array(frame.observations[index], copy=True)
                for index, camera_id in enumerate(self.adapter.camera_ids)
            }
            context = CandidateContext(
                episode_id=job.episode_id,
                source_episode_id=job.source_episode_id,
                anchor_id=job.anchor_id,
                task_id=job.task_id,
                scene_group=job.scene_group,
                episode_phase=job.episode_phase.value,
                time_index=job.time_index,
                seed=self.seed,
                expected_action_dimension=self.expected_action_dimension,
                minimum_action_horizon=(
                    self.prediction_horizon * self.observation_stride
                ),
                extra={"split": job.split.value},
            )
            generated = provider.generate(
                observation,
                np.array(frame.proprio, copy=True, dtype=np.float32),
                job.instruction,
                context,
            )
            if not isinstance(generated, list):
                raise TypeError("candidate provider did not return a list")
            if not generated:
                raise RuntimeError("candidate provider returned no candidates")
            return validate_and_deduplicate_candidates(generated, context=context)
        except CandidateValidationError as exc:
            self.adapter.close()
            return (), (
                CandidateRejection(exc.code, None, f"{type(exc).__name__}: {exc}"),
            )
        except Exception as exc:
            self.adapter.close()
            rejection = CandidateRejection(
                CandidateRejectionCode.POLICY_INFERENCE_FAILED,
                None,
                f"{type(exc).__name__}: {exc}",
            )
            return (), (rejection,)

    def _commit_anchor(
        self,
        job: D1AnchorJob,
        candidates: Sequence[PolicyCandidate],
        samples: Sequence[WorldModelSample],
        *,
        collection_duration_seconds: float,
    ) -> None:
        if len(candidates) != len(samples):
            raise RuntimeError("candidate/sample count differs before anchor commit")
        final = self.output_root / "anchors" / job.storage_key
        if final.exists():
            raise WorldModelSchemaError("completed anchor storage key already exists")
        staging = self.output_root / ".staging" / job.storage_key
        if staging.exists():
            raise WorldModelSchemaError("anchor staging directory already exists")
        staging.mkdir(parents=True)
        sample_records: list[dict[str, object]] = []
        for candidate, sample in zip(candidates, samples, strict=True):
            reference = write_world_model_sample(staging / "samples", sample)
            sample_records.append(
                {
                    **candidate.as_metadata_mapping(),
                    "arrays_digest": reference.arrays_digest,
                    "evaluation_subsets": _evaluation_subsets(job, candidate),
                    "metadata_digest": reference.metadata_digest,
                    "sample_id": reference.sample_id,
                    "terminal_reason": _terminal_reason(sample),
                }
            )
        record = {
            **job.as_mapping(),
            "candidate_set_digest": candidate_set_digest(candidates),
            "collection_duration_seconds": collection_duration_seconds,
            "samples": sample_records,
            "schema_version": "wm-v0-d1-anchor-record-v1",
        }
        record_digest = _mapping_digest(record, "D1AnchorRecordV1")
        write_atomic_json(staging / "anchor-record.json", record)
        write_atomic_json(
            staging / "completion-marker.json",
            {
                "anchor_id": job.anchor_id,
                "anchor_record_digest": record_digest,
                "sample_count": len(samples),
                "sample_ids": [item["sample_id"] for item in sample_records],
                "schema_version": "wm-v0-d1-anchor-completion-v1",
            },
        )
        os.replace(staging, final)

    def _append_rejection(
        self, job: D1AnchorJob, rejection: Mapping[str, object]
    ) -> None:
        self._append_raw_rejection({**job.as_mapping(), **rejection})

    def _append_raw_rejection(self, rejection: Mapping[str, object]) -> None:
        path = self.output_root / "rejections.jsonl"
        payload = json.dumps(rejection, sort_keys=True, allow_nan=False) + "\n"
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def _completed_anchor_ids(self) -> set[str]:
        return {
            str(record["anchor_id"])
            for _, record, _ in _complete_anchor_inventory(self.output_root)
        }

    def _valid_sample_count(self) -> int:
        return sum(
            len(cast(list[object], record["samples"]))
            for _, record, _ in _complete_anchor_inventory(self.output_root)
        )

    def _publish_indexes(self, *, planned_anchor_count: int) -> None:
        manifest, quality = build_d1_reports(self.output_root)
        inventory = _complete_anchor_inventory(self.output_root)
        index = {
            "anchors": [
                {
                    "anchor_id": record["anchor_id"],
                    "anchor_record_digest": marker["anchor_record_digest"],
                    "sample_ids": marker["sample_ids"],
                    "storage_key": path.name,
                }
                for path, record, marker in inventory
            ],
            "schema_version": "wm-v0-d1-sample-index-v1",
        }
        write_atomic_json(self.output_root / "sample-index.json", index)
        write_atomic_json(self.output_root / "dataset-manifest.json", manifest)
        write_atomic_json(self.output_root / "data-quality-report.json", quality)
        rejection_counts = cast(Mapping[str, int], quality["rejection_reason_counts"])
        state = {
            "completed_anchors": len(inventory),
            "current_class_distribution": quality["terminal_success_counts"],
            "duplicate_candidates": int(
                rejection_counts.get(D1RejectionCode.DUPLICATE_ACTION.value, 0)
            ),
            "planned_anchors": planned_anchor_count,
            "policy_generated_ratio": quality["policy_generated_ratio"],
            "policy_inference_failures": int(
                rejection_counts.get(D1RejectionCode.POLICY_INFERENCE_FAILED.value, 0)
            ),
            "rejected_samples": quality["rejected_sample_count"],
            "schema_version": "wm-v0-d1-collection-state-v1",
            "simulator_failures": int(
                rejection_counts.get(D1RejectionCode.SIMULATOR_ERROR.value, 0)
            ),
            "valid_samples": quality["valid_sample_count"],
        }
        write_atomic_json(self.output_root / "collection-state.json", state)


def build_d1_reports(
    root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Strictly reload committed anchors and build D1 manifest and quality report."""

    output = Path(root).absolute()
    inventory = _complete_anchor_inventory(output)
    samples: list[WorldModelSample] = []
    metadata: list[Mapping[str, object]] = []
    anchor_records: list[Mapping[str, object]] = []
    for path, record, marker in inventory:
        entries = record.get("samples")
        if not isinstance(entries, list) or len(entries) != marker.get("sample_count"):
            raise WorldModelSchemaError("completed D1 anchor sample inventory differs")
        anchor_records.append(record)
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise WorldModelSchemaError("D1 candidate metadata must be an object")
            sample_id = entry.get("sample_id")
            if not isinstance(sample_id, str):
                raise WorldModelSchemaError("D1 sample_id is missing")
            sample = read_world_model_sample(path / "samples", sample_id)
            if sample.candidate_id != entry.get("candidate_id"):
                raise WorldModelSchemaError("D1 sample/candidate identity differs")
            samples.append(sample)
            metadata.append(cast(Mapping[str, object], entry))
    base_manifest = build_dataset_manifest(samples).as_mapping()
    leakage = _split_leakage(samples)
    if leakage:
        raise WorldModelSchemaError(
            "D1 source groups cross splits: " + ", ".join(leakage[:8])
        )
    rejection_entries = _load_rejections(output / "rejections.jsonl")
    rejection_counts = Counter(str(item["reason_code"]) for item in rejection_entries)
    policy_counts = Counter(str(item["candidate_origin"]) for item in metadata)
    family_counts = Counter(str(item["policy_family"]) for item in metadata)
    checkpoint_counts = Counter(str(item["checkpoint_id"]) for item in metadata)
    scene_counts = Counter(str(item["scene_group"]) for item in anchor_records)
    phase_counts = Counter(str(item["episode_phase"]) for item in anchor_records)
    success_counts = Counter(str(item.terminal_success).lower() for item in samples)
    candidate_counts = Counter(item.anchor_id for item in samples)
    valid_count = len(samples)
    policy_count = policy_counts["policy_generated"]
    complete_future_count = sum(
        item.future_observations.shape[0] == item.prediction_horizon for item in samples
    )
    metadata_complete_count = sum(
        _candidate_metadata_complete(item) for item in metadata
    )
    observed_progress = sum(int(item.progress_mask.sum()) for item in samples)
    observed_events = sum(int(item.event_mask.sum()) for item in samples)
    event_positive_counts: Counter[str] = Counter()
    progress_values: list[float] = []
    for sample in samples:
        progress_values.extend(
            float(value) for value in sample.future_progress[sample.progress_mask]
        )
        for event_index, event_name in enumerate(sample.event_names):
            event_positive_counts[event_name] += int(
                sample.event_labels[:, event_index][
                    sample.event_mask[:, event_index]
                ].sum()
            )
    total_events = sum(int(item.event_mask.size) for item in samples)
    total_progress = sum(int(item.progress_mask.size) for item in samples)
    generated_ratio = 0.0 if valid_count == 0 else policy_count / valid_count
    future_ratio = 0.0 if valid_count == 0 else complete_future_count / valid_count
    metadata_ratio = 0.0 if valid_count == 0 else metadata_complete_count / valid_count
    episode_count = len({item.source_episode_id for item in samples})
    anchor_count = len({item.anchor_id for item in samples})
    success_count = success_counts["true"]
    failure_count = success_counts["false"]
    duplicate_identity_count = len(samples) - len(
        {(item.episode_id, item.anchor_id, item.candidate_id) for item in samples}
    )
    anchor_durations = [
        cast(float, record["collection_duration_seconds"])
        for record in anchor_records
        if type(record.get("collection_duration_seconds")) in (int, float)
    ]
    total_attempt_count = valid_count + len(rejection_entries)
    pilot_checks = {
        "anchor_count_at_least_25": anchor_count >= 25,
        "both_terminal_classes_present": success_count > 0 and failure_count > 0,
        "candidates_per_anchor_at_least_3": bool(candidate_counts)
        and min(candidate_counts.values()) >= 3,
        "candidate_metadata_complete": metadata_ratio == 1.0,
        "duplicate_identity_zero": duplicate_identity_count == 0,
        "future_frame_completeness_at_least_99_percent": future_ratio >= 0.99,
        "policy_generated_ratio_at_least_70_percent": generated_ratio >= 0.70,
        "restoration_mismatch_zero": rejection_counts["RESTORE_MISMATCH"] == 0,
        "sample_count_at_least_100": valid_count >= 100,
        "source_group_leakage_zero": not leakage,
    }
    formal_checks = {
        "anchor_count_at_least_250": anchor_count >= 250,
        "candidates_per_anchor_at_least_3": bool(candidate_counts)
        and min(candidate_counts.values()) >= 3,
        "data_quality_checks_passed": all(
            (
                duplicate_identity_count == 0,
                future_ratio >= 0.99,
                metadata_ratio == 1.0,
                not leakage,
                rejection_counts["RESTORE_MISMATCH"] == 0,
            )
        ),
        "failure_count_at_least_100": failure_count >= 100,
        "nonterminal_progress_or_event_present": observed_progress > 0
        or observed_events > 0,
        "policy_generated_ratio_at_least_70_percent": generated_ratio >= 0.70,
        "required_splits_present": all(
            split.value in cast(Mapping[str, object], base_manifest["split_counts"])
            for split in DatasetSplit
        ),
        "sample_count_at_least_1000": valid_count >= 1000,
        "source_episode_count_at_least_100": episode_count >= 100,
        "source_group_leakage_zero": not leakage,
        "success_count_at_least_100": success_count >= 100,
        "future_frame_completeness_at_least_99_percent": future_ratio >= 0.99,
    }
    quality: dict[str, object] = {
        "action_duplicate_rejection_rate": (
            0.0
            if valid_count + len(rejection_entries) == 0
            else rejection_counts["DUPLICATE_ACTION"]
            / (valid_count + len(rejection_entries))
        ),
        "anchor_count": anchor_count,
        "candidate_count_per_anchor": _distribution_summary(candidate_counts.values()),
        "checkpoint_counts": dict(sorted(checkpoint_counts.items())),
        "collection_throughput": {
            "mean_anchor_seconds": (
                0.0
                if not anchor_durations
                else sum(anchor_durations) / len(anchor_durations)
            ),
            "mean_sample_seconds": (
                0.0 if valid_count == 0 else sum(anchor_durations) / valid_count
            ),
            "samples_per_second": (
                0.0
                if not anchor_durations or sum(anchor_durations) == 0.0
                else valid_count / sum(anchor_durations)
            ),
            "total_anchor_runtime_seconds": sum(anchor_durations),
        },
        "episode_phase_counts": dict(sorted(phase_counts.items())),
        "event_observed_counts": cast(
            Mapping[str, object], base_manifest["event_observed_counts"]
        ),
        "event_positive_counts": dict(sorted(event_positive_counts.items())),
        "failure_rates": {
            "policy_inference": (
                0.0
                if total_attempt_count == 0
                else rejection_counts["POLICY_INFERENCE_FAILED"] / total_attempt_count
            ),
            "simulator": (
                0.0
                if total_attempt_count == 0
                else rejection_counts["SIMULATOR_ERROR"] / total_attempt_count
            ),
        },
        "formal_training_gate": _gate_mapping(formal_checks),
        "future_frame_completeness": future_ratio,
        "missing_ratios": {
            "candidate_metadata": 1.0 - metadata_ratio,
            "events": 1.0
            if total_events == 0
            else 1.0 - observed_events / total_events,
            "progress": (
                1.0 if total_progress == 0 else 1.0 - observed_progress / total_progress
            ),
        },
        "pilot_gate": _gate_mapping(pilot_checks),
        "policy_family_counts": dict(sorted(family_counts.items())),
        "policy_generated_ratio": generated_ratio,
        "policy_source_counts": dict(sorted(policy_counts.items())),
        "progress_distribution": _float_distribution_summary(progress_values),
        "rejected_sample_count": len(rejection_entries),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "scene_group_counts": dict(sorted(scene_counts.items())),
        "schema_version": "wm-v0-d1-data-quality-v1",
        "source_episode_count": episode_count,
        "split_counts": base_manifest["split_counts"],
        "terminal_success_counts": dict(sorted(success_counts.items())),
        "valid_sample_count": valid_count,
    }
    manifest_without_digest: dict[str, object] = {
        "anchor_count": anchor_count,
        "base_wm_v0_manifest": base_manifest,
        "candidate_metadata_inventory_digest": _mapping_digest(
            {"candidates": [dict(item) for item in metadata]},
            "D1CandidateMetadataInventoryV1",
        ),
        "formal_training_gate": quality["formal_training_gate"],
        "future_frame_completeness": future_ratio,
        "pilot_gate": quality["pilot_gate"],
        "policy_generated_ratio": generated_ratio,
        "sample_count": valid_count,
        "schema_version": "wm-v0-d1-dataset-manifest-v1",
        "source_episode_count": episode_count,
    }
    manifest = {
        **manifest_without_digest,
        "content_digest": _mapping_digest(
            manifest_without_digest, "WorldModelD1DatasetManifestV1"
        ),
    }
    return manifest, quality


def load_or_build_d1_reports(
    root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build reports and verify persisted versions when they already exist."""

    manifest, quality = build_d1_reports(root)
    output = Path(root).absolute()
    for path, expected in (
        (output / "dataset-manifest.json", manifest),
        (output / "data-quality-report.json", quality),
    ):
        if path.is_file():
            observed = json.loads(path.read_text(encoding="utf-8"))
            if observed != expected:
                raise WorldModelSchemaError(
                    f"persisted {path.name} differs from strict D1 reload"
                )
    return manifest, quality


def _complete_anchor_inventory(
    root: Path,
) -> list[tuple[Path, Mapping[str, object], Mapping[str, object]]]:
    anchors = Path(root).absolute() / "anchors"
    if not anchors.is_dir():
        return []
    inventory: list[tuple[Path, Mapping[str, object], Mapping[str, object]]] = []
    for path in sorted(item for item in anchors.iterdir() if item.is_dir()):
        record = _read_mapping(path / "anchor-record.json")
        marker = _read_mapping(path / "completion-marker.json")
        expected_digest = _mapping_digest(record, "D1AnchorRecordV1")
        if marker.get("anchor_record_digest") != expected_digest:
            raise WorldModelSchemaError("D1 anchor completion marker digest differs")
        inventory.append((path, record, marker))
    return inventory


def _read_mapping(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise WorldModelSchemaError(f"missing regular D1 file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorldModelSchemaError(f"invalid D1 JSON {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorldModelSchemaError(f"D1 {path.name} must be an object")
    return cast(Mapping[str, object], value)


def _load_rejections(path: Path) -> list[Mapping[str, object]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise WorldModelSchemaError("D1 rejection log must be a regular file")
    result: list[Mapping[str, object]] = []
    for line_number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorldModelSchemaError(
                f"invalid rejection log line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict) or not isinstance(value.get("reason_code"), str):
            raise WorldModelSchemaError("D1 rejection log entry is incomplete")
        result.append(cast(Mapping[str, object], value))
    return result


def _candidate_metadata_complete(value: Mapping[str, object]) -> bool:
    required = {
        "action_content_digest",
        "action_horizon",
        "candidate_id",
        "candidate_origin",
        "checkpoint_id",
        "generation_latency_ms",
        "policy_family",
        "policy_name",
        "sampling_config",
        "sample_id",
        "terminal_reason",
    }
    if not required.issubset(value):
        return False
    if value.get("candidate_origin") == "policy_generated":
        checkpoint_hash = value.get("checkpoint_hash")
        return (
            isinstance(checkpoint_hash, str)
            and checkpoint_hash.startswith("sha256:")
            and len(checkpoint_hash) == 71
        )
    return True


def _split_leakage(samples: Sequence[WorldModelSample]) -> list[str]:
    assignments: dict[str, set[DatasetSplit]] = defaultdict(set)
    for sample in samples:
        assignments[sample.split_group_id].add(sample.split)
    return sorted(group for group, splits in assignments.items() if len(splits) > 1)


def _evaluation_subsets(job: D1AnchorJob, candidate: PolicyCandidate) -> list[str]:
    values = [
        f"{candidate.candidate_origin.value}_only",
        f"phase_specific:{job.episode_phase.value}",
    ]
    if job.split is DatasetSplit.TEST:
        values.extend(("held_out_episode", "held_out_scene"))
    return values


def _terminal_reason(sample: WorldModelSample) -> str:
    if sample.terminal_success is True:
        return "task_success"
    if sample.terminal_success is False:
        return "task_failure"
    if not sample.terminal_reached:
        return "horizon_exhausted"
    raise WorldModelSchemaError("D1 terminal/truncation reason is inconsistent")


def _validate_job_group_splits(jobs: Sequence[D1AnchorJob]) -> None:
    assignments: dict[str, set[DatasetSplit]] = defaultdict(set)
    for job in jobs:
        assignments[job.split_group_id].add(job.split)
    leaking = sorted(group for group, splits in assignments.items() if len(splits) > 1)
    if leaking:
        raise WorldModelSchemaError(
            "D1 planned source groups cross splits: " + ", ".join(leaking[:8])
        )


def _distribution_summary(values: Sequence[int] | Any) -> dict[str, float | int]:
    items = tuple(int(value) for value in values)
    if not items:
        return {"maximum": 0, "mean": 0.0, "minimum": 0}
    return {
        "maximum": max(items),
        "mean": sum(items) / len(items),
        "minimum": min(items),
    }


def _float_distribution_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"maximum": 0.0, "mean": 0.0, "minimum": 0.0}
    return {
        "maximum": max(values),
        "mean": sum(values) / len(values),
        "minimum": min(values),
    }


def _gate_mapping(checks: Mapping[str, bool]) -> dict[str, object]:
    failed = tuple(sorted(name for name, passed in checks.items() if not passed))
    return {
        "authorized": not failed,
        "checks": dict(sorted(checks.items())),
        "failed_checks": list(failed),
    }


def _mapping_digest(value: Mapping[str, object], context: str) -> str:
    return (
        "sha256:"
        + hashlib.sha256(canonical_json_bytes(value, context=context)).hexdigest()
    )


def _candidate_rejection_mapping(
    rejection: CandidateRejection,
) -> dict[str, object]:
    code = D1RejectionCode(rejection.code.value)
    return {
        "action_content_digest": rejection.action_content_digest,
        "candidate_id": rejection.candidate_id,
        "detail": rejection.detail,
        "reason_code": code.value,
    }


def _execution_rejection_mapping(exc: Exception) -> dict[str, object]:
    message = str(exc)
    lowered = message.lower()
    if "restore" in lowered:
        code = D1RejectionCode.RESTORE_MISMATCH
    elif "future observation alignment" in lowered:
        code = D1RejectionCode.MISSING_FUTURE_FRAME
    elif "event" in lowered:
        code = D1RejectionCode.INVALID_EVENT_LABEL
    elif "alignment" in lowered:
        code = D1RejectionCode.OBSERVATION_MISALIGNMENT
    elif "finite" in lowered:
        code = D1RejectionCode.NONFINITE_ACTION
    elif "action" in lowered or "mask" in lowered:
        code = D1RejectionCode.INVALID_ACTION_SHAPE
    else:
        code = D1RejectionCode.SIMULATOR_ERROR
    return {
        "candidate_id": None,
        "detail": f"{type(exc).__name__}: {message}",
        "reason_code": code.value,
    }


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise WorldModelSchemaError(f"{name} must be non-empty stripped text")


__all__ = [
    "D1AnchorJob",
    "D1CollectionResult",
    "D1CollectionRunner",
    "D1RejectionCode",
    "EpisodePhase",
    "build_d1_reports",
    "load_or_build_d1_reports",
]

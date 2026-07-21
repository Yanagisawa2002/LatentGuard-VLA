"""WM-v0 dataset inventory, leakage validation, and formal-training gate."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from latentguard.replay.identity import canonical_json_bytes

from .data_schema import DatasetSplit, WorldModelSample, WorldModelSchemaError


@dataclass(frozen=True, slots=True)
class DatasetGate:
    """Complete formal-training authorization decision."""

    authorized: bool
    checks: dict[str, bool]
    reasons: tuple[str, ...]

    def as_mapping(self) -> dict[str, object]:
        """Return a JSON-native gate record."""

        return {
            "authorized": self.authorized,
            "checks": dict(sorted(self.checks.items())),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class WorldModelDatasetManifest:
    """Compact content-bound summary of validated WM-v0 samples."""

    sample_count: int
    source_episode_count: int
    anchor_count: int
    candidate_count: int
    split_counts: dict[str, int]
    policy_source_counts: dict[str, int]
    corruption_counts: dict[str, int]
    terminal_success_counts: dict[str, int]
    event_observed_counts: dict[str, int]
    missing_ratios: dict[str, float]
    prediction_horizons: dict[str, int]
    observation_strides: dict[str, int]
    group_assignments: dict[str, str]
    gate: DatasetGate
    schema_version: str = "1.0"

    def as_mapping(self, *, include_digest: bool = True) -> dict[str, object]:
        """Return the deterministic JSON-native manifest mapping."""

        result: dict[str, object] = {
            "anchor_count": self.anchor_count,
            "candidate_count": self.candidate_count,
            "corruption_counts": dict(sorted(self.corruption_counts.items())),
            "event_observed_counts": dict(sorted(self.event_observed_counts.items())),
            "formal_training_gate": self.gate.as_mapping(),
            "group_assignments": dict(sorted(self.group_assignments.items())),
            "missing_ratios": dict(sorted(self.missing_ratios.items())),
            "observation_strides": dict(sorted(self.observation_strides.items())),
            "policy_source_counts": dict(sorted(self.policy_source_counts.items())),
            "prediction_horizons": dict(sorted(self.prediction_horizons.items())),
            "sample_count": self.sample_count,
            "schema_version": self.schema_version,
            "source_episode_count": self.source_episode_count,
            "split_counts": dict(sorted(self.split_counts.items())),
            "terminal_success_counts": dict(
                sorted(self.terminal_success_counts.items())
            ),
        }
        if include_digest:
            result["content_digest"] = self.content_digest
        return result

    @property
    def content_digest(self) -> str:
        """Return the path-independent manifest identity."""

        payload = canonical_json_bytes(
            self.as_mapping(include_digest=False), context="WorldModelDatasetManifestV1"
        )
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def write(self, path: Path) -> Path:
        """Write the compact manifest deterministically."""

        destination = Path(path).absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.as_mapping(), sort_keys=True, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        return destination


def build_dataset_manifest(
    samples: Iterable[WorldModelSample], *, minimum_valid_samples: int = 1000
) -> WorldModelDatasetManifest:
    """Validate group isolation and summarize the complete sample inventory."""

    values = tuple(samples)
    if type(minimum_valid_samples) is not int or minimum_valid_samples < 1:
        raise WorldModelSchemaError("minimum_valid_samples must be positive")
    groups: dict[str, set[DatasetSplit]] = defaultdict(set)
    group_assignments: dict[str, str] = {}
    for sample in values:
        groups[sample.split_group_id].add(sample.split)
    leaking = sorted(group for group, splits in groups.items() if len(splits) != 1)
    if leaking:
        raise WorldModelSchemaError(
            "source groups cross dataset splits: " + ", ".join(leaking[:8])
        )
    for group, splits in groups.items():
        group_assignments[group] = next(iter(splits)).value
    split_counts = Counter(sample.split.value for sample in values)
    policy_counts = Counter(sample.policy_source.value for sample in values)
    corruption_counts = Counter(
        sample.corruption_type or "not_applicable" for sample in values
    )
    success_counts = Counter(str(sample.terminal_success).lower() for sample in values)
    event_counts: Counter[str] = Counter()
    observed_progress = 0
    total_progress = 0
    for sample in values:
        for index, event_name in enumerate(sample.event_names):
            event_counts[event_name] += int(sample.event_mask[:, index].sum())
        observed_progress += int(sample.progress_mask.sum())
        total_progress += int(sample.progress_mask.size)
    total_event = sum(sample.event_mask.size for sample in values)
    observed_event = sum(int(sample.event_mask.sum()) for sample in values)
    checks = {
        "minimum_sample_count": len(values) >= minimum_valid_samples,
        "real_future_observation_for_every_sample": bool(values)
        and all(
            sample.future_source == "simulator_execution"
            and sample.future_observations.shape[0] >= 1
            for sample in values
        ),
        "required_splits_present": all(
            split.value in split_counts for split in DatasetSplit
        ),
        "source_group_isolation": True,
        "terminal_success_and_failure_present": bool(success_counts["true"])
        and bool(success_counts["false"]),
        "nonterminal_progress_or_event_present": observed_progress > 0
        or observed_event > 0,
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    gate = DatasetGate(not reasons, checks, reasons)
    return WorldModelDatasetManifest(
        sample_count=len(values),
        source_episode_count=len({sample.source_episode_id for sample in values}),
        anchor_count=len({sample.anchor_id for sample in values}),
        candidate_count=len(
            {(sample.anchor_id, sample.candidate_id) for sample in values}
        ),
        split_counts=dict(split_counts),
        policy_source_counts=dict(policy_counts),
        corruption_counts=dict(corruption_counts),
        terminal_success_counts=dict(success_counts),
        event_observed_counts=dict(event_counts),
        missing_ratios={
            "events": 1.0 if total_event == 0 else 1.0 - (observed_event / total_event),
            "progress": 1.0
            if total_progress == 0
            else 1.0 - (observed_progress / total_progress),
        },
        prediction_horizons=dict(
            Counter(str(sample.prediction_horizon) for sample in values)
        ),
        observation_strides=dict(
            Counter(str(sample.observation_stride) for sample in values)
        ),
        group_assignments=group_assignments,
        gate=gate,
    )

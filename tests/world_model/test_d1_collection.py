"""Transactional WM-v0 D1 collection and formal-gate tests."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pytest

from latentguard.world_model.candidates import (
    CandidateContext,
    CandidateOrigin,
    PolicyCandidate,
)
from latentguard.world_model.collector import (
    CollectionFrame,
    ExactRestoreReport,
    StepResult,
)
from latentguard.world_model.d1_collection import (
    D1AnchorJob,
    D1CollectionRunner,
    EpisodePhase,
    load_or_build_d1_reports,
)
from latentguard.world_model.data_schema import DatasetSplit, WorldModelSchemaError


@dataclass(slots=True)
class _Adapter:
    camera_ids: tuple[str, ...] = ("front",)
    event_names: tuple[str, ...] = ("grasped", "collision")
    steps: int = 0
    last_value: float = 0.0

    def restore(self, anchor_id: str) -> ExactRestoreReport:
        self.steps = 0
        self.last_value = 0.0
        return ExactRestoreReport(anchor_id, "sha256:" + "a" * 64, True, 70, 1e-7)

    def observe(self) -> CollectionFrame:
        return CollectionFrame(
            observations=np.full((1, 2, 2, 3), self.steps, dtype=np.uint8),
            proprio=np.asarray([self.steps, self.last_value], dtype=np.float32),
            progress=float(self.steps) / 4.0,
            events={"grasped": self.steps >= 2, "collision": None},
        )

    def step(self, action: np.ndarray) -> StepResult:  # type: ignore[type-arg]
        self.steps += 1
        self.last_value = float(action[0])
        terminal = self.steps == 4
        return StepResult(terminal, False, self.last_value > 0 if terminal else None)

    def close(self) -> None:
        return None


@dataclass(slots=True)
class _Provider:
    calls: int = 0
    candidates: list[PolicyCandidate] = field(default_factory=list)

    def generate(
        self,
        observation: dict[str, np.ndarray],  # type: ignore[type-arg]
        proprioception: np.ndarray,  # type: ignore[type-arg]
        instruction: str | None,
        context: CandidateContext,
    ) -> list[PolicyCandidate]:
        del observation, proprioception, instruction, context
        self.calls += 1
        return list(self.candidates)


def _candidate(candidate_id: str, value: float) -> PolicyCandidate:
    return PolicyCandidate(
        candidate_id=candidate_id,
        action_chunk=np.full((4, 3), value, dtype=np.float32),
        action_mask=np.ones(4, dtype=np.bool_),
        candidate_origin=CandidateOrigin.SYNTHETIC_CORRUPTION,
        policy_family="synthetic",
        policy_name="fixture",
        checkpoint_id="not_applicable",
        checkpoint_hash=None,
        inference_seed=0,
        sampling_config={"value": value},
        action_horizon=4,
        generation_latency_ms=0.0,
        corruption_type="constant_bias",
    )


def _job(anchor_id: str = "anchor-1") -> D1AnchorJob:
    return D1AnchorJob(
        episode_id="episode-1",
        source_episode_id="source-1",
        split_group_id="source-1",
        anchor_id=anchor_id,
        task_id="task",
        instruction=None,
        split=DatasetSplit.TRAIN,
        scene_group="scene-1",
        episode_phase=EpisodePhase.TRANSPORT,
        task_progress_t=0.5,
        time_index=8,
        distance_to_target=0.2,
        grasp_state=True,
        object_state="grasped",
    )


def _runner(tmp_path) -> D1CollectionRunner:  # type: ignore[no-untyped-def]
    return D1CollectionRunner(
        adapter=_Adapter(),
        output_root=tmp_path / "collection",
        prediction_horizon=2,
        observation_stride=2,
        expected_action_dimension=3,
        seed=11,
    )


def test_anchor_commit_is_atomic_and_complete_resume_is_zero_work(tmp_path) -> None:
    """Completed anchors survive strict reload and are not generated twice."""

    provider = _Provider(
        candidates=[_candidate("success", 1.0), _candidate("fail", -1.0)]
    )
    runner = _runner(tmp_path)

    first = runner.run((_job(),), lambda _: provider, resume=False)
    resumed = runner.run((_job(),), lambda _: provider, resume=True)
    manifest, quality = load_or_build_d1_reports(runner.output_root)

    assert first.valid_sample_count == 2
    assert resumed.zero_work_resume
    assert provider.calls == 1
    assert manifest["sample_count"] == 2
    assert quality["terminal_success_counts"] == {"false": 1, "true": 1}
    anchor_dirs = list((runner.output_root / "anchors").iterdir())
    assert len(anchor_dirs) == 1
    assert (anchor_dirs[0] / "completion-marker.json").is_file()
    assert not list((runner.output_root / ".staging").iterdir())


def test_incomplete_anchor_is_quarantined_then_safely_recollected(tmp_path) -> None:
    """Resume never promotes partial sample files into the formal inventory."""

    runner = _runner(tmp_path)
    staging = runner.output_root / ".staging" / _job().storage_key
    staging.mkdir(parents=True)
    (staging / "partial.txt").write_text("not a sample", encoding="utf-8")
    provider = _Provider(candidates=[_candidate("candidate", 1.0)])

    result = runner.run((_job(),), lambda _: provider, resume=True)

    assert result.valid_sample_count == 1
    assert result.rejected_sample_count == 1
    assert len(list((runner.output_root / "quarantine").iterdir())) == 1


def test_synthetic_candidates_never_satisfy_policy_quota_or_formal_gate(
    tmp_path,
) -> None:
    """Real simulator futures do not convert action edits into policy candidates."""

    runner = _runner(tmp_path)
    provider = _Provider(candidates=[_candidate("a", 1.0), _candidate("b", -1.0)])
    result = runner.run((_job(),), lambda _: provider, resume=False)
    formal = result.quality_report["formal_training_gate"]

    assert result.quality_report["policy_generated_ratio"] == 0.0
    assert isinstance(formal, dict)
    assert not formal["authorized"]
    assert "policy_generated_ratio_at_least_70_percent" in formal["failed_checks"]
    assert "sample_count_at_least_1000" in formal["failed_checks"]


def test_empty_provider_is_recorded_as_policy_inference_failure(tmp_path) -> None:
    """Provider failures never silently fall back to synthetic actions."""

    runner = _runner(tmp_path)
    provider = _Provider(candidates=[])

    result = runner.run((_job(),), lambda _: provider, resume=False)

    assert result.valid_sample_count == 0
    assert result.rejected_sample_count == 1
    assert result.quality_report["rejection_reason_counts"] == {
        "POLICY_INFERENCE_FAILED": 1
    }


def test_planned_source_group_leakage_is_rejected_before_execution(tmp_path) -> None:
    """Different anchors from one source episode cannot cross data splits."""

    provider = _Provider(candidates=[_candidate("candidate", 1.0)])
    runner = _runner(tmp_path)
    leaking = replace(_job("anchor-2"), episode_id="episode-2", split=DatasetSplit.TEST)

    with pytest.raises(WorldModelSchemaError, match="cross splits"):
        runner.run((_job(), leaking), lambda _: provider, resume=False)
    assert provider.calls == 0

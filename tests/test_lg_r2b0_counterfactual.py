from __future__ import annotations

import importlib
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from latentguard.adapters.vla_jepa.constants import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
)
from latentguard.counterfactual.candidates import (
    CandidateDiversityThresholds,
    action_content_sha256,
    candidate_group_metrics,
    classify_candidate_group,
    pairwise_action_metrics,
)
from latentguard.counterfactual.gates import (
    CounterfactualGateEvidence,
    evaluate_counterfactual_gate,
    validate_anchor_registry,
)
from latentguard.counterfactual.libero_state import (
    capture_libero_state,
    load_libero_state,
    restore_libero_state,
    save_libero_state,
)
from latentguard.counterfactual.models import (
    CandidateDisposition,
    RealPolicyCandidate,
    TerminalOutcome,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
BASELINE = "6ae64db453f835e27bba700bc1f58b227fd16939"


def _thresholds() -> CandidateDiversityThresholds:
    return CandidateDiversityThresholds(
        near_full_chunk_l2=0.01,
        meaningful_full_chunk_l2=0.10,
        meaningful_endpoint_translation=0.03,
        meaningful_cumulative_rotation=0.10,
        gripper_disagreement_epsilon=0.50,
    )


def _native(value: float = 0.0) -> np.ndarray:
    action = np.full((7, 7), value, dtype=np.float32)
    action[:, -1] = -1.0
    return action


def _candidate(
    native: np.ndarray,
    *,
    source_kind: str = "native_policy_sampling",
    inference_seed: int = 720000,
) -> RealPolicyCandidate:
    normalized = np.array(native, copy=True)
    digest = action_content_sha256(normalized, native, None)
    return RealPolicyCandidate(
        candidate_id=f"candidate-{inference_seed}",
        anchor_id="anchor-1",
        policy_id=CHECKPOINT_ID,
        checkpoint_revision=CHECKPOINT_REVISION,
        processor_revision=CHECKPOINT_REVISION,
        inference_seed=inference_seed,
        sampling_config={"seed": inference_seed},
        normalized_action_chunk=normalized,
        native_action_chunk=native,
        action_mask=None,
        content_sha256=digest,
        generation_latency_ms=1.0,
        source_kind=source_kind,
    )


def test_repository_changes_are_isolated_from_forbidden_repositories() -> None:
    tracked = subprocess.check_output(
        ["git", "diff", "--name-only", BASELINE],
        cwd=ROOT,
        text=True,
    ).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    changed = {path.replace("\\", "/").casefold() for path in tracked + untracked}
    assert changed
    assert not any(
        marker in path
        for path in changed
        for marker in ("langmani", "robolab", "pointworld")
    )


def test_frozen_policy_and_processor_identity_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    common = importlib.import_module("_lg_r2b0_common")
    identity = common.frozen_policy_identity()
    assert identity["checkpoint_id"] == CHECKPOINT_ID
    assert identity["checkpoint_revision"] == CHECKPOINT_REVISION
    assert identity["processor_revision"] == CHECKPOINT_REVISION
    assert identity["action_execution_horizon"] == 7


def test_fixed_seed_reproduces_and_different_seed_changes_native_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(SCRIPTS))
    common = importlib.import_module("_lg_r2b0_common")

    class FakePolicy:
        reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

        def predict_action_chunk(self, batch: dict[str, Any]) -> Any:
            del batch
            return torch.randn((1, 7, 7), dtype=torch.float32)

    policy = FakePolicy()
    stack = SimpleNamespace(
        policy=policy,
        config=SimpleNamespace(num_inference_timesteps=4),
    )
    monkeypatch.setattr(common, "policy_batch", lambda *args, **kwargs: {})

    def native_chunk(value: Any, **kwargs: Any) -> np.ndarray:
        del kwargs
        native = value.detach().cpu().numpy()[0].astype(np.float32)
        native = np.clip(native, -1.0, 1.0)
        native[:, -1] = -1.0
        return native

    monkeypatch.setattr(common, "_native_chunk", native_chunk)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    arguments = {
        "stack": stack,
        "raw_observation": {},
        "instruction": "instruction",
        "env_preprocessor": object(),
        "env_postprocessor": object(),
        "preprocessor": object(),
        "postprocessor": object(),
        "anchor_id": "anchor",
    }
    first = common.generate_policy_candidate(
        **arguments,
        candidate_id="first",
        inference_seed=700100,
    )
    repeated = common.generate_policy_candidate(
        **arguments,
        candidate_id="repeated",
        inference_seed=700100,
    )
    different = common.generate_policy_candidate(
        **arguments,
        candidate_id="different",
        inference_seed=700101,
    )
    assert first.content_sha256 == repeated.content_sha256
    assert np.array_equal(first.native_action_chunk, repeated.native_action_chunk)
    assert first.content_sha256 != different.content_sha256
    assert policy.reset_calls == 3


def test_synthetic_candidates_and_final_seeds_are_rejected() -> None:
    with pytest.raises(ValueError, match="synthetic"):
        _candidate(_native(), source_kind="synthetic_perturbation")
    with pytest.raises(ValueError, match="sealed final seed"):
        _candidate(_native(), inference_seed=900000)


def test_action_content_hash_binds_values_shapes_and_mask() -> None:
    action = _native()
    baseline = action_content_sha256(action, action, None)
    changed = action.copy()
    changed[0, 0] = 0.25
    assert baseline != action_content_sha256(changed, action, None)
    assert baseline != action_content_sha256(
        action,
        action,
        np.ones(7, dtype=np.bool_),
    )
    candidate = _candidate(action)
    assert candidate.content_sha256 == baseline
    with pytest.raises(ValueError, match="does not match"):
        RealPolicyCandidate(
            candidate_id="bad",
            anchor_id="anchor",
            policy_id=CHECKPOINT_ID,
            checkpoint_revision=CHECKPOINT_REVISION,
            processor_revision=CHECKPOINT_REVISION,
            inference_seed=700000,
            sampling_config={},
            normalized_action_chunk=action,
            native_action_chunk=action,
            action_mask=None,
            content_sha256="0" * 64,
            generation_latency_ms=0.0,
        )


def test_exact_near_and_meaningful_candidate_detection() -> None:
    baseline = _native()
    exact = baseline.copy()
    near = baseline.copy()
    near[0, 0] = 0.001
    distinct = baseline.copy()
    distinct[:, 0] = 0.05
    dispositions = classify_candidate_group(
        [baseline, exact, near, distinct],
        _thresholds(),
    )
    assert dispositions == (
        CandidateDisposition.MEANINGFULLY_DISTINCT,
        CandidateDisposition.EXACT_DUPLICATE,
        CandidateDisposition.NEAR_DUPLICATE,
        CandidateDisposition.MEANINGFULLY_DISTINCT,
    )
    metrics = pairwise_action_metrics(
        baseline,
        distinct,
        gripper_epsilon=0.5,
    )
    assert float(metrics["full_chunk_normalized_l2"]) > 0.10


class _MockController:
    def __init__(self) -> None:
        self.goal = np.asarray([0.1, 0.2], dtype=np.float64)


class _MockData:
    def __init__(self) -> None:
        self.qpos = np.asarray([1.0, 2.0], dtype=np.float64)
        self.qvel = np.asarray([0.5, -0.5], dtype=np.float64)
        self.ctrl = np.asarray([0.0], dtype=np.float64)
        self.act = np.asarray([0.25], dtype=np.float64)
        self.mocap_pos = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)
        self.mocap_quat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        self.cam_xpos = np.asarray([[0.0, 0.0, 2.0]], dtype=np.float64)
        self.geom_xpos = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64)


class _MockSim:
    def __init__(self) -> None:
        self.data = _MockData()

    def set_state_from_flattened(self, state: np.ndarray) -> None:
        np.copyto(self.data.qpos, state)

    def forward(self) -> None:
        return


class _MockTask:
    def __init__(self) -> None:
        self.sim = _MockSim()
        self.robots = [SimpleNamespace(controller=_MockController())]
        self.timestep = 3
        self._success = False
        self.np_random = np.random.default_rng(11)


class _MockControl:
    def __init__(self) -> None:
        self.env = _MockTask()
        self._elapsed_steps = 3
        self.np_random = np.random.default_rng(12)

    def get_sim_state(self) -> np.ndarray:
        return self.env.sim.data.qpos.copy()


class _MockSingle:
    def __init__(self) -> None:
        self._env = _MockControl()
        self._elapsed_steps = 3
        self.done = False
        self.np_random = np.random.default_rng(13)


def test_state_snapshot_roundtrip_restores_controller_rng_and_latches(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    single = _MockSingle()
    random.seed(41)
    np.random.seed(42)
    torch.manual_seed(43)
    snapshot = capture_libero_state(single)
    path = tmp_path / "state"
    save_libero_state(path, snapshot)
    reloaded = load_libero_state(path)
    assert reloaded.content_sha256 == snapshot.content_sha256
    single._env.env.sim.data.qpos[:] = 99.0
    single._env.env.sim.data.qvel[:] = -99.0
    single._env.env.sim.data.cam_xpos[:] = 77.0
    single._env.env.sim.data.geom_xpos[:] = 66.0
    single._env.env.robots[0].controller.goal[:] = 55.0
    single._elapsed_steps = 999
    single._env._elapsed_steps = 999
    single._env.env.timestep = 999
    random.random()
    np.random.random()
    torch.rand(1)
    comparison = restore_libero_state(single, reloaded, atol=0.0)
    assert comparison.within_tolerance is True
    assert comparison.runtime_state_matches is True
    assert comparison.maximum_absolute_error == 0.0
    assert single._elapsed_steps == 3
    assert single._env._elapsed_steps == 3
    assert single._env.env.timestep == 3
    assert np.array_equal(
        single._env.env.robots[0].controller.goal,
        np.asarray([0.1, 0.2]),
    )
    assert np.array_equal(
        single._env.env.sim.data.cam_xpos,
        np.asarray([[0.0, 0.0, 2.0]]),
    )
    assert np.array_equal(
        single._env.env.sim.data.geom_xpos,
        np.asarray([[0.1, 0.2, 0.3]]),
    )


def test_repeated_restored_branch_dynamics_are_deterministic() -> None:
    single = _MockSingle()
    snapshot = capture_libero_state(single)
    outcomes = []
    for _ in range(5):
        assert restore_libero_state(single, snapshot, atol=0.0).within_tolerance
        for action in (0.1, -0.2, 0.3):
            single._env.env.sim.data.qpos += action
            single._env.env.sim.data.qvel[:] = action
        outcomes.append(
            (
                single._env.env.sim.data.qpos.copy(),
                single._env.env.sim.data.qvel.copy(),
            )
        )
    assert all(
        np.array_equal(outcomes[0][0], item[0])
        and np.array_equal(outcomes[0][1], item[1])
        for item in outcomes[1:]
    )


def test_complete_render_comparison_records_exact_error_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    common = importlib.import_module("_lg_r2b0_common")
    expected = {
        "pixels": {
            "main": np.zeros((2, 2, 3), dtype=np.uint8),
            "wrist": np.ones((1, 1, 3), dtype=np.uint8),
        }
    }
    observed = {
        "pixels": {
            "main": np.zeros((2, 2, 3), dtype=np.uint8),
            "wrist": np.asarray([[[1, 2, 1]]], dtype=np.uint8),
        }
    }
    result = common.compare_rendered_observations(expected, observed)
    assert result["exact"] is False
    assert result["compared_values"] == 15
    assert result["different_values"] == 1
    assert result["different_value_ratio"] == pytest.approx(1 / 15)
    assert result["mean_absolute_error"] == pytest.approx(1 / 15)
    assert result["maximum_absolute_error"] == 1.0


def _anchor_records() -> list[dict[str, Any]]:
    config = yaml.safe_load(
        (ROOT / "configs" / "lg_r2b0" / "anchors.yaml").read_text(encoding="utf-8")
    )
    records = []
    for schedule in config["task_schedules"]:
        for seed in schedule["seeds"]:
            records.append(
                {
                    "anchor_id": f"{schedule['suite']}-{schedule['task_id']}-{seed}",
                    "episode_id": f"episode-{schedule['suite']}-{seed}",
                    "suite": schedule["suite"],
                    "task_id": schedule["task_id"],
                    "seed": seed,
                }
            )
    return records


def test_anchor_schedule_is_frozen_before_outcomes_and_balanced() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "lg_r2b0" / "anchors.yaml").read_text(encoding="utf-8")
    )
    assert config["selection_inputs"] == {
        "candidate_results": False,
        "branch_outcomes": False,
    }
    result = validate_anchor_registry(_anchor_records())
    assert result == {
        "anchors": 60,
        "tasks": 6,
        "suites": 2,
        "minimum_task_anchors": 10,
        "task6_anchor_ratio": pytest.approx(1 / 3),
        "final_seed_hits": [],
    }


def test_task6_ratio_gate_fails_closed() -> None:
    anchors = _anchor_records()
    for item in anchors[:2]:
        item["task_id"] = 6
    with pytest.raises(ValueError, match="task 6"):
        validate_anchor_registry(anchors)


def test_same_anchor_candidate_metrics_do_not_cross_groups() -> None:
    actions = [_native(), _native(0.1), _native(-0.1)]
    result = candidate_group_metrics(["a", "b", "c"], actions, _thresholds())
    assert result["candidate_count"] == 3
    assert result["meaningfully_distinct_count"] == 3
    assert len(result["pairs"]) == 3
    with pytest.raises(ValueError, match="unique"):
        candidate_group_metrics(["a", "a"], actions[:2], _thresholds())


def test_unresolved_outcome_is_not_failure() -> None:
    assert TerminalOutcome.UNRESOLVED_HORIZON != TerminalOutcome.FAILURE
    validator = (SCRIPTS / "lg_r2b0_validate_dataset.py").read_text(encoding="utf-8")
    assert 'if outcome == "UNRESOLVED_HORIZON"' in validator
    assert 'elif outcome == "FAILURE"' in validator
    assert '"unresolved_counted_as_failure": False' in validator


def test_short_horizon_label_alignment_and_terminal_carry_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    collector = importlib.import_module("lg_r2b0_collect_counterfactuals")
    records = [
        {
            "progress": 0.25,
            "stage_id": 2,
            "stage_name": "grasp",
            "contact": True,
            "object_pose_delta": {},
            "sim_state_sha256": "a" * 64,
        },
        {
            "progress": 0.40,
            "stage_id": 3,
            "stage_name": "lift",
            "contact": True,
            "object_pose_delta": {},
            "sim_state_sha256": "b" * 64,
        },
    ]
    boundary = collector._boundary_summary(
        records,
        horizon=7,
        anchor_progress=0.10,
    )
    assert boundary["observed_step"] == 2
    assert boundary["terminal_carry_forward"] is True
    assert boundary["progress_delta"] == pytest.approx(0.30)


def test_continuation_identity_and_policy_queue_reset_are_mandatory() -> None:
    collection = yaml.safe_load(
        (ROOT / "configs" / "lg_r2b0" / "collection.yaml").read_text(encoding="utf-8")
    )
    assert collection["continuation_policy"] == "frozen_primary_vla_jepa"
    assert collection["continuation_seed_base"] == 730000
    policy_contract = (ROOT / "docs" / "lg_r2b0_policy_state_contract.md").read_text(
        encoding="utf-8"
    )
    assert "action queue" in policy_contract
    assert "same frozen" in policy_contract
    validator = (SCRIPTS / "lg_r2b0_validate_dataset.py").read_text(encoding="utf-8")
    assert "continuation configuration differs within" in validator


def test_episode_seed_identity_and_final_seed_registry() -> None:
    anchors = _anchor_records()
    assert len({item["episode_id"] for item in anchors}) == 60
    assert len({int(item["seed"]) for item in anchors}) == 60
    assert not any(900000 <= int(item["seed"]) <= 900099 for item in anchors)
    candidates = yaml.safe_load(
        (ROOT / "configs" / "lg_r2b0" / "candidates.yaml").read_text(encoding="utf-8")
    )
    operational = [
        int(candidates["candidate_seed_base"]),
        *[int(value) for value in candidates["source_audit"]["sampling_seeds"]],
    ]
    assert not any(900000 <= value <= 900099 for value in operational)


def test_gate_requires_real_diverse_non_task6_identifiable_evidence() -> None:
    accepted = CounterfactualGateEvidence(
        restore_mismatches=0,
        branch_contamination=0,
        candidate_metadata_completeness=1.0,
        identity_completeness=1.0,
        policy_generated_candidate_ratio=1.0,
        synthetic_candidate_ratio=0.0,
        valid_anchors=50,
        valid_branches=150,
        anchors_with_three_distinct_ratio=0.70,
        exact_duplicate_rate=0.20,
        meaningful_progress_spread_ratio=0.30,
        local_event_disagreement_anchors=0,
        mixed_terminal_outcome_anchors=0,
        non_task6_outcome_divergence=True,
        ranking_tie_rate=0.70,
    )
    assert evaluate_counterfactual_gate(accepted)["LG_R2B1_AUTHORIZED"] is True
    rejected = replace(
        accepted,
        non_task6_outcome_divergence=False,
    )
    assert evaluate_counterfactual_gate(rejected)["LG_R2B1_AUTHORIZED"] is False


def test_no_training_ranking_intervention_or_external_stack() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(SCRIPTS.glob("*lg_r2b0*.py"))
    )
    prohibited_calls = (
        ".backward(",
        "optimizer.step(",
        "torch.optim.",
        "model.train(",
    )
    assert not any(call in source for call in prohibited_calls)
    collection = yaml.safe_load(
        (ROOT / "configs" / "lg_r2b0" / "collection.yaml").read_text(encoding="utf-8")
    )
    assert collection["optimizer_steps"] == 0
    assert collection["backward_calls"] == 0
    assert collection["online_selection"] is False
    assert collection["intervention"] is False
    for path in (
        SCRIPTS / "_lg_r2b0_common.py",
        SCRIPTS / "lg_r2b0_collect_counterfactuals.py",
        SCRIPTS / "lg_r2b0_generate_candidates.py",
    ):
        text = path.read_text(encoding="utf-8").casefold()
        assert "langmani" not in text
        assert "robolab" not in text
        assert "pointworld" not in text


def test_compact_export_rejects_missing_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    monkeypatch_path = str(SCRIPTS)
    sys.path.insert(0, monkeypatch_path)
    try:
        exporter = importlib.import_module("lg_r2b0_export_compact_artifacts")
        with pytest.raises(FileNotFoundError):
            exporter.export_compact(source, destination)
    finally:
        sys.path.remove(monkeypatch_path)


def test_all_pre_registered_configs_parse_as_mappings() -> None:
    for path in sorted((ROOT / "configs" / "lg_r2b0").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert payload["milestone"] == "LG-R2b0"


def test_complete_config_validation_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    validator = importlib.import_module("lg_r2b0_validate_configs")
    report = validator.validate_configs(ROOT / "configs" / "lg_r2b0")
    assert report["status"] == "pass"
    assert report["anchors"] == 60
    assert report["tasks"] == 6
    assert report["final_seed_overlap"] == []

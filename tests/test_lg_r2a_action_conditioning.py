from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from latentguard.action_conditioning.contracts import (
    DEPLOYABLE_FEATURES,
    ActionContract,
    validate_deployable_features,
)
from latentguard.action_conditioning.gates import evaluate_lg_r2b_gate
from latentguard.action_conditioning.splits import (
    EpisodeRecord,
    assign_grouped_folds,
    validate_grouped_folds,
)

ROOT = Path(__file__).resolve().parents[1]


def test_action_contract_accepts_full_real_chunks() -> None:
    actions = np.zeros((2, 7, 7), dtype=np.float32)
    actions[..., -1] = np.asarray([[-1.0] * 7, [1.0] * 7])
    masks = np.ones((2, 7), dtype=np.bool_)
    result = ActionContract().validate(actions, masks, require_full_horizon=True)
    assert result["status"] == "pass"
    assert result["effective_horizon_min"] == 7


def test_action_contract_rejects_nonfinite_bounds_and_bad_gripper() -> None:
    masks = np.ones((1, 7), dtype=np.bool_)
    for value, message in (
        (np.nan, "finite"),
        (1.1, "bound"),
    ):
        actions = np.zeros((1, 7, 7), dtype=np.float32)
        actions[..., -1] = -1.0
        actions[0, 0, 0] = value
        with pytest.raises(ValueError, match=message):
            ActionContract().validate(actions, masks, require_full_horizon=True)
    actions = np.zeros((1, 7, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="gripper"):
        ActionContract().validate(actions, masks, require_full_horizon=True)


def test_action_mask_requires_exact_zero_padding() -> None:
    actions = np.zeros((1, 7, 7), dtype=np.float32)
    actions[..., -1] = -1.0
    masks = np.ones((1, 7), dtype=np.bool_)
    masks[0, -1] = False
    actions[0, -1, -1] = 0.0
    ActionContract().validate(actions, masks, require_full_horizon=False)
    actions[0, -1, 0] = 0.1
    with pytest.raises(ValueError, match="padding"):
        ActionContract().validate(actions, masks, require_full_horizon=False)


def test_future_and_privileged_features_are_prohibited() -> None:
    validate_deployable_features(
        {
            "current_vla_representation",
            "numeric_action_chunk",
            "action_mask",
        }
    )
    assert "actual_remaining_episode_length" not in DEPLOYABLE_FEATURES
    for feature in (
        "future_observation",
        "privileged_stage",
        "terminal_result",
        "actual_remaining_episode_length",
    ):
        with pytest.raises(ValueError):
            validate_deployable_features({feature})


def _fold_records() -> list[EpisodeRecord]:
    records = []
    for index in range(15):
        records.append(
            EpisodeRecord(
                episode_id=f"episode-{index}",
                seed=index,
                task=f"suite/task{index % 3}",
                success=index >= 5,
                taxonomy=("OBJECT_DROP" if index % 2 else "FAILED_PLACEMENT")
                if index < 5
                else None,
                progress_bin=index % 5,
                windows=10 + index,
            )
        )
    return records


def test_grouped_folds_are_deterministic_and_each_has_failure() -> None:
    records = _fold_records()
    first = assign_grouped_folds(records, assignment_seed=22031)
    second = assign_grouped_folds(records, assignment_seed=22031)
    assert first == second
    result = validate_grouped_folds(records, first)
    assert result["episode_leakage"] == 0
    assert result["seed_leakage"] == 0
    assert all(fold["failures"] >= 1 for fold in result["folds"])


def test_seed_group_cannot_cross_folds() -> None:
    records = _fold_records()
    records.append(
        EpisodeRecord(
            episode_id="same-seed-extra",
            seed=0,
            task="suite/task0",
            success=True,
            taxonomy=None,
            progress_bin=0,
            windows=1,
        )
    )
    assignments = assign_grouped_folds(records)
    assert assignments["episode-0"] == assignments["same-seed-extra"]
    assignments["same-seed-extra"] = (assignments["episode-0"] + 1) % 5
    with pytest.raises(ValueError, match="seeds cross folds"):
        validate_grouped_folds(records, assignments)


def test_state_action_parameter_control_and_encoder_cap() -> None:
    pytest.importorskip("torch")
    from latentguard.action_conditioning.models import ProbeModel

    state = ProbeModel(variant="state_only").parameter_report()
    combined = ProbeModel(variant="state_action").parameter_report()
    relative_gap = (
        abs(combined["trainable_parameters"] - state["trainable_parameters"])
        / state["trainable_parameters"]
    )
    assert relative_gap < 0.05
    assert combined["action_encoder_parameters"] <= 500_000


def test_gate_requires_increment_classification_sensitivity_and_robustness() -> None:
    payload = {
        "required_checks": {
            "episode_leakage": 0,
            "seed_leakage": 0,
            "foundation_models_frozen": True,
            "action_contract_status": "pass",
            "cv_status": "pass",
        },
        "incremental_value": {
            "short_progress_mae_relative_reduction": 0.11,
            "short_progress_spearman_delta": 0.01,
            "best_action_sensitive_classification_auprc_delta": 0.06,
        },
        "permutation_sensitivity": {
            "short_progress_mae_relative_worsening": 0.11,
            "largest_classification_auprc_drop": 0.01,
        },
        "robustness": {
            "positive_folds": 4,
            "task_macro_delta": 0.01,
            "leave_task6_best_auprc": 0.2,
            "leave_task6_corresponding_prevalence": 0.1,
        },
        "deployability": {
            "pre_execution_only": True,
            "model_e_used_for_gate": False,
        },
    }
    assert evaluate_lg_r2b_gate(payload)["LG_R2B_AUTHORIZED"] is True
    payload["robustness"]["positive_folds"] = 3
    assert evaluate_lg_r2b_gate(payload)["LG_R2B_AUTHORIZED"] is False


def test_configs_freeze_foundations_test_once_and_prohibit_search() -> None:
    for name in (
        "state_only",
        "action_only",
        "state_action",
        "state_action_proprio",
    ):
        config = yaml.safe_load(
            (ROOT / "configs" / "lg_r2a" / f"{name}.yaml").read_text(encoding="utf-8")
        )
        assert config["foundation_features_frozen"] is True
        assert config["test_once"] is True
        assert "architecture_search" not in config
    internal = yaml.safe_load(
        (ROOT / "configs" / "lg_r2a" / "internal_tokens.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert internal["gate_eligible"] is False
    assert internal["general_external_candidate_deployable"] is False


def test_source_guards_forbid_rollout_synthetic_candidates_and_final_seeds() -> None:
    source = (ROOT / "scripts" / "lg_r2a_validate_inputs.py").read_text(
        encoding="utf-8"
    )
    assert '"new_rollouts": 0' in source
    assert '"synthetic_actions": 0' in source
    assert '"final_seeds_accessed": False' in source
    sensitivity = (ROOT / "scripts" / "lg_r2a_action_sensitivity.py").read_text(
        encoding="utf-8"
    )
    assert '"real_executed_actions_only": True' in sensitivity
    assert '"true_same_state_counterfactual": False' in sensitivity

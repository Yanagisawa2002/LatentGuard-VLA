from __future__ import annotations

import hashlib
import json
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


def test_committed_lg_r2a_input_and_target_identities() -> None:
    artifact_root = ROOT / "artifacts" / "lg_r2a"
    source = json.loads(
        (artifact_root / "input_dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert source["status"] == "pass"
    assert source["counts"] == {
        "episode_leakage": 0,
        "episodes": 320,
        "evaluation_windows": 8470,
        "failure_tasks": 4,
        "failures": 27,
        "frames": 79326,
        "seed_leakage": 0,
        "successes": 293,
        "suites": 2,
        "tasks": 8,
    }
    assert source["final_seeds_accessed"] is False
    targets = json.loads(
        (artifact_root / "target_manifest.json").read_text(encoding="utf-8")
    )
    assert targets["samples"] == 4848
    assert targets["excluded_incomplete_action_tail_samples"] == 152
    assert targets["supports"] == {
        "FAILED_PLACEMENT": 31,
        "OBJECT_DROP": 8,
        "terminal_failure": 43,
    }


def test_committed_cv_and_action_contract_pass() -> None:
    artifact_root = ROOT / "artifacts" / "lg_r2a"
    contract = json.loads(
        (artifact_root / "action_contract.json").read_text(encoding="utf-8")
    )
    assert contract["validation"]["status"] == "pass"
    assert contract["action_dimension"] == 7
    assert contract["action_horizon"] == 7
    assert contract["execution_horizon"] == 7
    splits = json.loads(
        (artifact_root / "cv_split_manifest.json").read_text(encoding="utf-8")
    )
    assert splits["validation"]["episode_leakage"] == 0
    assert splits["validation"]["seed_leakage"] == 0
    assert [row["failures"] for row in splits["validation"]["folds"]] == [
        5,
        6,
        6,
        5,
        5,
    ]


def test_committed_result_b_keeps_lg_r2b_closed() -> None:
    artifact_root = ROOT / "artifacts" / "lg_r2a"
    summary = json.loads(
        (artifact_root / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert summary["result_class"] == "Result B"
    assert summary["gate"]["LG_R2B_AUTHORIZED"] is False
    assert summary["gate"]["checks"]["progress_increment"] is False
    assert summary["gate"]["checks"]["fold_direction_at_least_4_of_5"] is False
    assert summary["candidate_generation"] is False
    assert summary["candidate_ranking"] is False
    assert summary["intervention"] is False
    internal = json.loads(
        (artifact_root / "internal_token_results.json").read_text(encoding="utf-8")
    )
    assert internal["used_for_gate"] is False
    assert internal["general_external_candidate_deployable"] is False


def test_committed_remote_hashes_and_execution_commits() -> None:
    artifact_root = ROOT / "artifacts" / "lg_r2a"
    audit = json.loads(
        (artifact_root / "remote_execution_audit.json").read_text(encoding="utf-8")
    )
    for name, identity in audit["compact_files"].items():
        path = artifact_root / name
        assert path.stat().st_size == identity["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity["sha256"]
    assert audit["execution_commits"]["training"] == (
        "1f03f484c9f29e0a3dbfee767b56dbfd34795e3d"
    )
    assert audit["execution_commits"]["evaluation"] == (
        "f867f97c09e8785b91c1c94c18cae1b316795f59"
    )
    assert len(audit["execution_commands"]) == 6
    assert all(
        row["formal_training_started"] is False
        for row in audit["pre_training_gate_stops"]
    )
    assert audit["source_checkout_clean_after_run"] is True
    assert audit["server_left_running"] is True


def test_action_magnitude_diagnostic_is_not_selection_evidence() -> None:
    payload = json.loads(
        (ROOT / "artifacts" / "lg_r2a" / "action_magnitude_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["shortcut_concern_not_excluded"] is True
    assert payload["used_for_training"] is False
    assert payload["used_for_selection"] is False

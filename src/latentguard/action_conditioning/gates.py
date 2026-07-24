"""Frozen LG-R2a promotion gate for an LG-R2b pilot."""

from __future__ import annotations

from typing import Any


def evaluate_lg_r2b_gate(summary: dict[str, Any]) -> dict[str, Any]:
    """Evaluate pre-registered mandatory, incremental, and sensitivity gates."""

    required = summary["required_checks"]
    increments = summary["incremental_value"]
    sensitivity = summary["permutation_sensitivity"]
    robustness = summary["robustness"]
    deployability = summary["deployability"]
    mandatory_checks = {
        "episode_leakage_zero": required["episode_leakage"] == 0,
        "seed_leakage_zero": required["seed_leakage"] == 0,
        "foundation_models_frozen": required["foundation_models_frozen"] is True,
        "action_contract_pass": required["action_contract_status"] == "pass",
        "cv_folds_valid": required["cv_status"] == "pass",
    }
    progress_check = (
        increments["short_progress_mae_relative_reduction"] >= 0.10
        or increments["short_progress_spearman_delta"] >= 0.10
    )
    classification_check = (
        increments["best_action_sensitive_classification_auprc_delta"] >= 0.05
    )
    sensitivity_check = (
        sensitivity["short_progress_mae_relative_worsening"] >= 0.10
        or sensitivity["largest_classification_auprc_drop"] >= 0.03
    )
    robustness_checks = {
        "fold_direction_at_least_4_of_5": robustness["positive_folds"] >= 4,
        "task_macro_nonnegative": robustness["task_macro_delta"] >= 0.0,
        "leave_task6_above_prevalence": (
            robustness["leave_task6_best_auprc"]
            > robustness["leave_task6_corresponding_prevalence"]
        ),
    }
    checks = {
        **mandatory_checks,
        "progress_increment": progress_check,
        "classification_increment": classification_check,
        "permutation_sensitivity": sensitivity_check,
        **robustness_checks,
        "pre_execution_only": deployability["pre_execution_only"] is True,
        "model_e_not_used_for_gate": deployability["model_e_used_for_gate"] is False,
    }
    authorized = all(checks.values())
    return {
        "schema_version": "latentguard.lg_r2a.lg_r2b_gate.v1",
        "status": "authorized" if authorized else "not_authorized",
        "LG_R2B_AUTHORIZED": authorized,
        "checks": checks,
        "thresholds": {
            "short_progress_mae_relative_reduction_min": 0.10,
            "short_progress_spearman_delta_min": 0.10,
            "classification_auprc_absolute_delta_min": 0.05,
            "permutation_mae_relative_worsening_min": 0.10,
            "permutation_classification_auprc_drop_min": 0.03,
            "positive_fold_count_min": 4,
        },
        "automatic_lg_r2b_execution": False,
    }

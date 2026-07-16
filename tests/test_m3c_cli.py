"""CPU-only command-surface and blind-boundary tests for M3C."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from latentguard import cli
from latentguard.integrations.maniskill_pickcube.configuration import (
    load_maniskill_pickcube_action_layout,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
    PickCubeActionControlContractV1,
)
from latentguard.m3c_cli import (
    M3C_COMMANDS,
    M3CCommandError,
    _fit_temporal_policies,
    _inference_latency_payload,
    _load_bound_inference_latency_report,
    _load_policies,
    _load_protocol,
    _load_selector_configuration_report,
    _M3CProtocolConfig,
    _policy_payload,
    _precollection_inference_smoke_payload,
    _run_replay_complete_candidate_pools,
    _selector_configuration_payload,
    _validate_candidate_runtime_action_contract,
    _validate_m3b_runtime_artifact_digests,
    _validated_phase_completion_timestamp,
    add_m3c_subparsers,
    run_m3c_command,
)
from latentguard.selection.blind_input import (
    build_blind_candidate_pool,
    save_blind_candidate_pool,
)
from latentguard.selection.blind_protocol import (
    EXPECTED_STAGE_A_SELECTOR_IDS,
    EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS,
)
from latentguard.selection.checkpoint_bundle import BundleLoadingDiagnosticsV1
from latentguard.selection.configuration import load_candidate_pool_configuration
from latentguard.selection.ensemble import (
    EnsembleEndToEndProfileV1,
    EnsembleInferenceDiagnosticsV1,
)
from latentguard.selection.manifest import load_blind_selection_manifest
from latentguard.selection.models import (
    CandidateDistribution,
    CandidateGroupV1,
    CandidatePoolV1,
    CandidateRefV1,
    SourceTrajectoryIdentityV1,
    array_content_digest,
)
from latentguard.selection.selectors import deterministic_random_decision
from latentguard.training.calibration import compute_validation_prediction_digest
from latentguard.training.reporting import (
    StrictReportV1,
    load_strict_report,
    save_strict_report,
)


def _sha(character: str) -> str:
    return f"sha256:{character * 64}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_m3c_subparsers(subparsers)
    return parser


def _profile_diagnostics(
    durations: tuple[float, ...],
    *,
    loading_total: float,
    model_loading: float,
    peak_memory: int,
    device: str = "cpu",
) -> EnsembleInferenceDiagnosticsV1:
    return EnsembleInferenceDiagnosticsV1(
        bundle_loading=BundleLoadingDiagnosticsV1(
            device=device,
            total_seconds=loading_total,
            per_seed_model_seconds=(model_loading / 5.0,) * 5,
        ),
        end_to_end_profile=EnsembleEndToEndProfileV1(
            device=device,
            candidate_count=8,
            group_durations_seconds=durations,
            peak_allocated_device_memory_bytes=peak_memory,
        ),
    )


def _pool() -> CandidatePoolV1:
    groups: list[CandidateGroupV1] = []
    for group_index in range(6):
        candidates = tuple(
            CandidateRefV1(
                proposal_id=f"proposal-{group_index}-{candidate_index}",
                configuration_ordinal=candidate_index,
                distribution=(
                    CandidateDistribution.ID_LIKE
                    if candidate_index < 4
                    else CandidateDistribution.SHIFTED
                ),
                corruption_type=f"corruption-{candidate_index}",
                severity_id=f"severity-{candidate_index}",
                seed=group_index * 8 + candidate_index,
                action_chunk=np.full((16, 8), candidate_index + 1, dtype=np.float32),
                action_mask=np.ones(16, dtype=np.bool_),
            )
            for candidate_index in range(8)
        )
        groups.append(
            CandidateGroupV1(
                anchor_id=f"anchor-{group_index}",
                trajectory=SourceTrajectoryIdentityV1(
                    source_trajectory_id=f"trajectory-{group_index}",
                    source_seed=10_000 + group_index,
                    split_group_id=f"split-{group_index}",
                    complete_state_digests=(f"sha256:{group_index + 1:064x}",),
                ),
                state_content_digest=_sha("a"),
                verifier_state_content_digest=_sha("b"),
                continuation_identity=_sha("c"),
                source_action_prefix_digest=array_content_digest(
                    np.zeros((16, 8), dtype=np.float32)
                ),
                state_vector=np.zeros(38, dtype=np.float32),
                continuation_actions=np.zeros((2, 8), dtype=np.float32),
                candidates=candidates,
            )
        )
    return CandidatePoolV1(
        source_set_digest=_sha("d"),
        candidate_pool_configuration_digest=_sha("e"),
        action_contract_digest=_sha("f"),
        exclusion_inventory_digest=_sha("1"),
        groups=tuple(groups),
    )


def test_registers_exact_commands_and_stage_a_has_no_outcome_capability() -> None:
    parser = _parser()
    command_action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(command_action.choices) == set(M3C_COMMANDS)
    build = command_action.choices["build-blind-candidate-pools"]
    build_options = {
        option for action in build._actions for option in action.option_strings
    }
    assert "--blind-input-dir" in build_options
    assert next(
        action
        for action in build._actions
        if "--blind-input-dir" in action.option_strings
    ).required
    select = command_action.choices["select-action-candidates"]
    option_strings = {
        option for action in select._actions for option in action.option_strings
    }
    assert "--blind-input-dir" in option_strings
    assert "--candidate-pool-dir" not in option_strings
    assert all(
        forbidden not in option.lower()
        for option in option_strings
        for forbidden in ("candidate-pool", "evidence", "outcome", "replay")
    )
    evaluate = command_action.choices["evaluate-candidate-selection"]
    required_evaluation_options = {
        option
        for action in evaluate._actions
        if action.required
        for option in action.option_strings
    }
    assert {"--cpu-latency-report", "--gpu-latency-report"}.issubset(
        required_evaluation_options
    )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "select-action-candidates",
                "--mode",
                "smoke",
                "--blind-input-dir",
                "blinded-input",
                "--prepared-bundles-dir",
                "bundles",
                "--m3b-runtime-root",
                "runtime",
                "--output-dir",
                "selection",
                "--outcome-dir",
                "forbidden",
            ]
        )


def test_candidate_configuration_binds_runtime_not_source_action_contract() -> None:
    configuration = load_candidate_pool_configuration(
        Path("configs/selection/m3c/candidate-pool-v1.json")
    )
    layout = load_maniskill_pickcube_action_layout(
        Path("configs/integrations/maniskill_pickcube/action-layout-v1.json")
    )
    scope = SimpleNamespace(layout=layout)

    _validate_candidate_runtime_action_contract(configuration, scope)  # type: ignore[arg-type]

    source_contract = PickCubeActionControlContractV1(
        coordinate_frame="unspecified",
        control_period_s=0.05,
        action_dtype="<f8",
        action_dimension=8,
    )
    assert source_contract.content_digest != configuration.action_contract_digest

    changed = SimpleNamespace(action_contract_digest=_sha("0"))
    with pytest.raises(M3CCommandError, match="trusted runtime layout"):
        _validate_candidate_runtime_action_contract(changed, scope)  # type: ignore[arg-type]


def test_latency_payload_aggregates_all_end_to_end_repetitions() -> None:
    joint_durations = tuple(float(index) / 1000.0 for index in range(1, 11))
    temporal_durations = tuple(float(index) / 500.0 for index in range(1, 11))
    action_durations = (0.003,) * 10
    diagnostics = {
        "action_only_mlp": (
            _profile_diagnostics(
                action_durations[:5],
                loading_total=0.4,
                model_loading=0.2,
                peak_memory=10,
            ),
            _profile_diagnostics(
                action_durations[5:],
                loading_total=0.4,
                model_loading=0.2,
                peak_memory=12,
            ),
        ),
        "state_action_mlp": (
            _profile_diagnostics(
                joint_durations[:5],
                loading_total=0.6,
                model_loading=0.3,
                peak_memory=100,
            ),
            _profile_diagnostics(
                joint_durations[5:],
                loading_total=0.6,
                model_loading=0.3,
                peak_memory=120,
            ),
        ),
        "temporal_state_action_verifier": (
            _profile_diagnostics(
                temporal_durations[:5],
                loading_total=0.9,
                model_loading=0.5,
                peak_memory=180,
            ),
            _profile_diagnostics(
                temporal_durations[5:],
                loading_total=0.9,
                model_loading=0.5,
                peak_memory=200,
            ),
        ),
    }
    payload = _inference_latency_payload(diagnostics, device="cpu")
    architectures = payload["architectures"]
    joint = architectures["state_action_mlp"]
    temporal = architectures["temporal_state_action_verifier"]
    expected = np.quantile(
        np.asarray(joint_durations), (0.50, 0.95, 0.99), method="linear"
    )
    assert joint["per_group_seconds_p50"] == pytest.approx(expected[0])
    assert joint["per_group_seconds_p95"] == pytest.approx(expected[1])
    assert joint["per_group_seconds_p99"] == pytest.approx(expected[2])
    assert joint["per_candidate_seconds_p99"] == pytest.approx(expected[2] / 8.0)
    assert joint["candidates_per_second"] == pytest.approx(
        8.0 / float(np.mean(joint_durations))
    )
    assert joint["measured_group_repetition_count"] == 10
    assert joint["measured_candidate_executions"] == 80
    assert joint["peak_allocated_device_memory_bytes"] == 120
    difference = payload["joint_minus_temporal"]
    for suffix in ("p50", "p95", "p99"):
        assert difference[f"per_group_seconds_{suffix}"] == pytest.approx(
            joint[f"per_group_seconds_{suffix}"]
            - temporal[f"per_group_seconds_{suffix}"]
        )
        assert difference[f"per_candidate_seconds_{suffix}"] == pytest.approx(
            joint[f"per_candidate_seconds_{suffix}"]
            - temporal[f"per_candidate_seconds_{suffix}"]
        )
    assert difference["bundle_loading_total_seconds"] == pytest.approx(-0.3)
    assert difference["model_loading_total_seconds"] == pytest.approx(-0.2)
    assert difference["peak_allocated_device_memory_bytes"] == -80
    serialized = json.dumps(payload, sort_keys=True)
    assert "group_durations_seconds" not in serialized
    assert payload["prediction_values_persisted"] is False
    assert payload["raw_duration_values_persisted"] is False


def test_latency_report_reload_is_bound_to_blind_pool_and_bundles(
    tmp_path: Path,
) -> None:
    blind_input = build_blind_candidate_pool(_pool())
    protocol = _load_protocol(
        Path("configs/selection/m3c/protocol-v1.json"), mode="smoke"
    )
    diagnostics = {
        architecture: tuple(
            _profile_diagnostics(
                (0.001, 0.002, 0.003, 0.004, 0.005),
                loading_total=0.1,
                model_loading=0.05,
                peak_memory=0,
            )
            for _ in blind_input.groups
        )
        for architecture in EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS
    }
    bundle_digests = {
        architecture: _sha(str(index + 2))
        for index, architecture in enumerate(EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS)
    }
    selector_digest = _sha("8")
    payload = _inference_latency_payload(diagnostics, device="cpu")
    payload.update(
        {
            "blind_input_digest": blind_input.content_digest,
            "candidate_pool_digest": blind_input.full_candidate_pool_digest,
            "protocol_digest": protocol.content_digest,
            "selector_configuration_digest": selector_digest,
            "verifier_bundle_digests": bundle_digests,
        }
    )
    report = StrictReportV1(
        report_type="m3c_inference_latency_v1",
        payload=payload,
    )
    path = tmp_path / "cpu-latency.json"
    save_strict_report(report, path)
    loaded = _load_bound_inference_latency_report(
        path,
        expected_device_kind="cpu",
        protocol=protocol,
        blind_input=blind_input,
        selector_configuration_digest=selector_digest,
        expected_bundle_digests=bundle_digests,
    )
    assert loaded.content_digest == report.content_digest

    path.unlink()
    changed = dict(payload)
    changed["blind_input_digest"] = _sha("9")
    save_strict_report(
        StrictReportV1(
            report_type="m3c_inference_latency_v1",
            payload=changed,
        ),
        path,
    )
    with pytest.raises(ValueError, match="top-level contract"):
        _load_bound_inference_latency_report(
            path,
            expected_device_kind="cpu",
            protocol=protocol,
            blind_input=blind_input,
            selector_configuration_digest=selector_digest,
            expected_bundle_digests=bundle_digests,
        )


def test_selector_configuration_binds_each_manifest_abstention_policy(
    tmp_path: Path,
) -> None:
    blind_input = build_blind_candidate_pool(_pool())
    protocol = _load_protocol(
        Path("configs/selection/m3c/protocol-v1.json"), mode="smoke"
    )
    bundle_digests = {
        architecture: _sha(str(index + 2))
        for index, architecture in enumerate(EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS)
    }
    bundles = {
        name: SimpleNamespace(identity=SimpleNamespace(content_digest=digest))
        for name, digest in bundle_digests.items()
    }
    policy_ids = (
        "maximum_validation_balanced_accuracy",
        "target_validation_failure_recall",
        "target_validation_coverage_90",
        "target_validation_coverage_80",
        "target_validation_coverage_70",
        "target_validation_coverage_50",
    )
    policies = tuple(
        SimpleNamespace(policy_id=policy_id, content_digest=_sha(str(index + 1)))
        for index, policy_id in enumerate(policy_ids)
    )
    baseline = SimpleNamespace(
        content_digest=protocol.payload["action_magnitude_baseline_digest"]
    )
    report = StrictReportV1(
        report_type="m3c_selector_configuration_v1",
        payload=_selector_configuration_payload(
            protocol=protocol,
            blind_input=blind_input,
            bundles=bundles,
            baseline=baseline,
            policies=policies,
        ),
    )
    path = tmp_path / "selector-configuration.json"
    save_strict_report(report, path)
    policy_by_selector = {
        selector_id: policy.content_digest
        for selector_id, policy in zip(
            EXPECTED_STAGE_A_SELECTOR_IDS[5:], policies, strict=True
        )
    }
    decisions = tuple(
        replace(
            deterministic_random_decision(
                group,
                seed=17,
                selector_id=selector_id,
            ),
            abstention_policy_id=policy_by_selector.get(selector_id),
        )
        for selector_id in EXPECTED_STAGE_A_SELECTOR_IDS
        for group in blind_input.groups
    )
    _load_selector_configuration_report(
        path,
        expected_digest=report.content_digest,
        protocol=protocol,
        blind_input=blind_input,
        expected_bundle_digests=bundle_digests,
        selections=decisions,
    )
    changed = (replace(decisions[0], abstention_policy_id=_sha("9")), *decisions[1:])
    with pytest.raises(ValueError, match="abstention-policy binding"):
        _load_selector_configuration_report(
            path,
            expected_digest=report.content_digest,
            protocol=protocol,
            blind_input=blind_input,
            expected_bundle_digests=bundle_digests,
            selections=changed,
        )


def test_protocol_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    source = Path("configs/selection/m3c/protocol-v1.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    value["unexpected"] = True
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected or missing fields"):
        _load_protocol(path, mode="smoke")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("full_requested_success_count", 1, "source contract changed"),
        ("anchors_per_trajectory", 1, "source contract changed"),
        ("bootstrap_replicates", 1999, "at least 2000"),
        ("full_starting_seed", 10001, "seed ranges overlap"),
    ],
)
def test_production_protocol_loader_rejects_weakened_execution_contract(
    tmp_path: Path,
    field: str,
    value: int,
    error: str,
) -> None:
    source = Path("configs/selection/m3c/protocol-v1.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / f"weakened-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        _load_protocol(path, mode="full")


def test_precollection_inference_smoke_runs_three_outcome_free_ensembles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    architectures = (
        "action_only_mlp",
        "state_action_mlp",
        "temporal_state_action_verifier",
    )
    bundles = {
        name: SimpleNamespace(identity=SimpleNamespace(content_digest=_sha(str(i + 1))))
        for i, name in enumerate(architectures)
    }
    calls: list[str] = []

    def score(
        bundle: SimpleNamespace,
        state: object,
        actions: np.ndarray,
        masks: np.ndarray,
        candidate_ids: tuple[str, ...],
        *,
        device: str,
    ) -> SimpleNamespace:
        del state
        assert actions.shape == (8, 16, 8)
        assert masks.shape == (8, 16)
        assert device == "cpu"
        calls.append(str(bundle.identity.content_digest))
        probabilities = np.linspace(0.1, 0.8, 8, dtype=np.float64)
        return SimpleNamespace(
            ranking=tuple(candidate_ids),
            ensemble_failure_probabilities=probabilities,
            per_seed_calibrated_failure_probabilities=np.repeat(
                probabilities[None, :], 5, axis=0
            ),
            per_seed_raw_logits=np.zeros((5, 8), dtype=np.float64),
            selected_candidate_id=candidate_ids[0],
        )

    monkeypatch.setattr(
        "latentguard.selection.ensemble.score_verifier_ensemble",
        score,
    )
    payload = _precollection_inference_smoke_payload(bundles, device="cpu")

    assert len(calls) == 3
    assert payload["candidate_count"] == 8
    assert payload["m3c_candidate_outcomes_loaded"] is False
    assert payload["m3c_source_trajectories_loaded"] is False
    assert payload["raw_prediction_values_persisted"] is False
    encoded = json.dumps(payload, sort_keys=True)
    assert "per_seed_raw_logits" not in encoded
    assert "candidate_actions" not in encoded


def test_committed_outer_digests_reject_rehashed_runtime_tamper(
    tmp_path: Path,
) -> None:
    from latentguard.selection.checkpoint_bundle import (
        EXPECTED_FIVE_SEEDS,
        SUPPORTED_SELECTION_ARCHITECTURES,
    )
    from latentguard.training.config import ModelType

    result_root = tmp_path / "result"
    runtime_root = tmp_path / "runtime"
    artifact_digests: dict[str, dict[str, str]] = {}
    first_calibration: Path | None = None
    for architecture in sorted(SUPPORTED_SELECTION_ARCHITECTURES):
        for seed in EXPECTED_FIVE_SEEDS:
            key = f"{architecture}/seed-{seed}"
            root = runtime_root / "evaluations" / key
            calibration = StrictReportV1(
                report_type="temperature_calibration_v1",
                payload={"seed": seed, "temperature": 1.0},
            )
            thresholds = StrictReportV1(
                report_type="frozen_thresholds_v1",
                payload={"seed": seed, "threshold": 0.5},
            )
            save_strict_report(calibration, root / "calibration.json")
            save_strict_report(thresholds, root / "thresholds.json")
            artifact_digests[key] = {
                "calibration.json": calibration.content_digest,
                "thresholds.json": thresholds.content_digest,
            }
            first_calibration = first_calibration or root / "calibration.json"
    for seed in EXPECTED_FIVE_SEEDS:
        artifact_digests[f"{ModelType.STATE_ONLY_MLP.value}/seed-{seed}"] = {
            "calibration.json": _sha("9"),
            "thresholds.json": _sha("8"),
        }
    benchmark = StrictReportV1(
        report_type="m3b_compact_benchmark_summary_v1",
        payload={"evaluation_artifact_digests": artifact_digests},
    )
    save_strict_report(benchmark, result_root / "benchmark-summary.json")
    assert (
        _validate_m3b_runtime_artifact_digests(result_root, runtime_root)
        == benchmark.content_digest
    )
    assert first_calibration is not None
    first_calibration.unlink()
    save_strict_report(
        StrictReportV1(
            report_type="temperature_calibration_v1",
            payload={"seed": 0, "temperature": 9.0},
        ),
        first_calibration,
    )
    with pytest.raises(ValueError, match="expected report identity"):
        _validate_m3b_runtime_artifact_digests(result_root, runtime_root)


class _Calibration:
    def apply(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=np.float64)


def test_prepare_path_fits_six_policies_from_five_seed_eight_candidate_groups(
    tmp_path: Path,
) -> None:
    architecture = "temporal_state_action_verifier"
    seeds = []
    for seed in range(5):
        rows = []
        for group_index in range(2):
            for candidate_index in range(8):
                rows.append(
                    {
                        "calibrated_failure_probability": None,
                        "candidate_type": "corrupted",
                        "failure_target": (
                            0 if group_index == 0 else int(candidate_index == 0)
                        ),
                        "group_id": f"group-{group_index}",
                        "raw_logit": 0.05 + 0.1 * candidate_index + 0.001 * seed,
                        "sample_id": f"sample-{group_index}-{candidate_index}",
                        "source_trajectory_id": f"trajectory-{group_index}",
                        "split": "validation",
                        "uncalibrated_failure_probability": 0.5,
                    }
                )
        path = (
            tmp_path
            / "runs"
            / architecture
            / f"seed-{seed}"
            / "validation-predictions.json"
        )
        save_strict_report(
            StrictReportV1(
                report_type="validation_predictions_v1",
                payload={"predictions": rows},
            ),
            path,
        )
        raw_logits = np.asarray([row["raw_logit"] for row in rows], dtype=np.float64)
        targets = np.asarray([row["failure_target"] for row in rows], dtype=np.int64)
        seeds.append(
            SimpleNamespace(
                identity=SimpleNamespace(
                    seed=seed,
                    validation_prediction_digest=(
                        compute_validation_prediction_digest(raw_logits, targets)
                    ),
                ),
                calibration=_Calibration(),
            )
        )
    bundle = SimpleNamespace(
        identity=SimpleNamespace(
            architecture=architecture,
            content_digest=_sha("2"),
        ),
        seeds=tuple(seeds),
    )
    protocol = _M3CProtocolConfig(
        payload={
            "accepted_split_digest": _sha("3"),
            "coverage_targets": [0.9, 0.8, 0.7, 0.5],
        },
        content_digest=_sha("4"),
    )
    policies = _fit_temporal_policies(bundle, tmp_path, protocol)
    assert len(policies) == 6
    assert all(policy.validation_group_count == 2 for policy in policies)
    assert [policy.target_coverage for policy in policies[2:]] == [
        0.9,
        0.8,
        0.7,
        0.5,
    ]
    policy_path = tmp_path / "policies.json"
    frozen_report = StrictReportV1(
        report_type="m3c_temporal_abstention_v1",
        payload=_policy_payload(policies),
    )
    save_strict_report(frozen_report, policy_path)
    loaded = _load_policies(
        policy_path,
        expected_report_digest=frozen_report.content_digest,
        expected_split_digest=_sha("3"),
        expected_bundle_digest=_sha("2"),
    )
    assert len(loaded) == 6
    policy_path.unlink()
    changed_payload = _policy_payload(policies)
    changed_payload["policies"][0]["threshold"] = min(1.0, policies[0].threshold + 0.01)
    save_strict_report(
        StrictReportV1(
            report_type="m3c_temporal_abstention_v1",
            payload=changed_payload,
        ),
        policy_path,
    )
    with pytest.raises(ValueError, match="expected report identity"):
        _load_policies(
            policy_path,
            expected_report_digest=frozen_report.content_digest,
            expected_split_digest=_sha("3"),
            expected_bundle_digest=_sha("2"),
        )


def test_select_command_creates_immutable_outcome_free_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool()
    blind_input = build_blind_candidate_pool(pool)
    blind_input_root = tmp_path / "blinded-input"
    save_blind_candidate_pool(blind_input, blind_input_root)
    bundle_digests = {
        "action_only_mlp": _sha("5"),
        "state_action_mlp": _sha("6"),
        "temporal_state_action_verifier": _sha("7"),
    }
    policies = tuple(
        SimpleNamespace(policy_id=policy_id, content_digest=_sha(str(index + 1)))
        for index, policy_id in enumerate(
            (
                "maximum_validation_balanced_accuracy",
                "target_validation_failure_recall",
                "target_validation_coverage_90",
                "target_validation_coverage_80",
                "target_validation_coverage_70",
                "target_validation_coverage_50",
            )
        )
    )
    monkeypatch.setattr(
        "latentguard.m3c_cli._load_selection_bundles",
        lambda *args, **kwargs: {
            name: SimpleNamespace(identity=SimpleNamespace(content_digest=digest))
            for name, digest in bundle_digests.items()
        },
    )
    monkeypatch.setattr(
        "latentguard.m3c_cli._load_preparation_summary",
        lambda *args, **kwargs: {
            "bundle_digests": bundle_digests,
            "action_magnitude_report_digest": _sha("5"),
            "policy_report_digest": _sha("8"),
            "policy_content_digests": [item.content_digest for item in policies],
        },
    )
    monkeypatch.setattr(
        "latentguard.m3c_cli._load_action_baseline",
        lambda *args, **kwargs: SimpleNamespace(
            content_digest=(
                "sha256:370d417bc80d68dcb60076a4bcf2b02f27d8f4e7ba82b7e7cb2946e342f7cf53"
            )
        ),
    )
    monkeypatch.setattr(
        "latentguard.m3c_cli._load_policies", lambda *args, **kwargs: policies
    )
    monkeypatch.setattr(
        "latentguard.m3c_cli._stage_a_decisions",
        lambda **kwargs: tuple(
            deterministic_random_decision(
                group,
                seed=17,
                selector_id=selector_id,
            )
            for selector_id in EXPECTED_STAGE_A_SELECTOR_IDS
            for group in kwargs["blind_input"].groups
        ),
    )
    monkeypatch.setattr(
        "latentguard.selection.serialization.load_candidate_pool",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Stage A must not load the full candidate pool")
        ),
    )
    monkeypatch.setattr(
        "latentguard.m3c_cli._inference_latency_payload",
        lambda *args, **kwargs: {
            "architectures": {},
            "device": "cpu",
            "joint_minus_temporal": {},
            "prediction_values_persisted": False,
            "quantile_aggregation_semantic": "test_v1",
            "schema_version": "1.0",
        },
    )
    monkeypatch.setattr(
        "latentguard.m3c_cli._load_selector_configuration_report",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "latentguard.m3c_cli._load_bound_inference_latency_report",
        lambda path, *args, **kwargs: load_strict_report(
            path,
            expected_report_type="m3c_inference_latency_v1",
        ),
    )
    output = tmp_path / "stage-a"
    args = _parser().parse_args(
        [
            "select-action-candidates",
            "--mode",
            "smoke",
            "--blind-input-dir",
            str(blind_input_root),
            "--prepared-bundles-dir",
            str(tmp_path / "bundles"),
            "--m3b-runtime-root",
            str(tmp_path / "runtime"),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ]
    )
    original_save = save_strict_report

    def interrupt_after_hidden_manifest(report: StrictReportV1, path: Path) -> Path:
        if path.name == "selector-configuration.json":
            raise KeyboardInterrupt
        return original_save(report, path)

    monkeypatch.setattr(
        "latentguard.training.reporting.save_strict_report",
        interrupt_after_hidden_manifest,
    )
    assert run_m3c_command(args) == 130
    assert not output.exists()
    monkeypatch.setattr(
        "latentguard.training.reporting.save_strict_report",
        original_save,
    )
    assert run_m3c_command(args) == 0
    manifest = load_blind_selection_manifest(output / "blind-selection-manifest.json")
    assert manifest.manifest.outcomes_available_during_selection is False
    assert manifest.manifest.outcome_input_paths == ()
    assert len(manifest.manifest.selections) == 66
    selector_configuration = load_strict_report(
        output / "selector-configuration.json",
        expected_report_type="m3c_selector_configuration_v1",
    )
    assert selector_configuration.payload["blind_input_digest"] == (
        blind_input.content_digest
    )
    assert selector_configuration.payload["candidate_pool_digest"] == (
        pool.content_digest
    )

    resumed = _parser().parse_args(
        [
            "select-action-candidates",
            "--mode",
            "smoke",
            "--blind-input-dir",
            str(blind_input_root),
            "--prepared-bundles-dir",
            str(tmp_path / "bundles"),
            "--m3b-runtime-root",
            str(tmp_path / "runtime"),
            "--output-dir",
            str(output),
            "--resume",
            "--device",
            "cpu",
        ]
    )
    assert run_m3c_command(resumed) == 0

    recoverable_output = tmp_path / "stage-a-recoverable"
    external_latency = tmp_path / "latency-cpu.json"
    recoverable_args = _parser().parse_args(
        [
            "select-action-candidates",
            "--mode",
            "smoke",
            "--blind-input-dir",
            str(blind_input_root),
            "--prepared-bundles-dir",
            str(tmp_path / "bundles"),
            "--m3b-runtime-root",
            str(tmp_path / "runtime"),
            "--output-dir",
            str(recoverable_output),
            "--latency-output",
            str(external_latency),
            "--device",
            "cpu",
        ]
    )

    def interrupt_external_copy(report: StrictReportV1, path: Path) -> Path:
        if path == external_latency:
            raise KeyboardInterrupt
        return original_save(report, path)

    monkeypatch.setattr(
        "latentguard.training.reporting.save_strict_report",
        interrupt_external_copy,
    )
    assert run_m3c_command(recoverable_args) == 130
    internal_latency = recoverable_output / "inference-latency-cpu.json"
    assert recoverable_output.is_dir()
    assert internal_latency.is_file()
    assert not external_latency.exists()
    monkeypatch.setattr(
        "latentguard.training.reporting.save_strict_report",
        original_save,
    )
    recoverable_resume = _parser().parse_args(
        [
            "select-action-candidates",
            "--mode",
            "smoke",
            "--blind-input-dir",
            str(blind_input_root),
            "--prepared-bundles-dir",
            str(tmp_path / "bundles"),
            "--m3b-runtime-root",
            str(tmp_path / "runtime"),
            "--output-dir",
            str(recoverable_output),
            "--latency-output",
            str(external_latency),
            "--device",
            "cpu",
            "--resume",
        ]
    )
    assert run_m3c_command(recoverable_resume) == 0
    assert load_strict_report(external_latency).content_digest == (
        load_strict_report(internal_latency).content_digest
    )


def test_dispatch_returns_none_for_non_m3c_command() -> None:
    assert run_m3c_command(argparse.Namespace(command="sanity-data")) is None


@pytest.mark.parametrize("command", sorted(M3C_COMMANDS))
def test_real_cli_main_exposes_every_m3c_command_help(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--help"])
    captured = capsys.readouterr()
    assert error.value.code == 0
    assert f"usage: latentguard {command}" in captured.out


def test_real_parser_rejects_unknown_command_before_m3c_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def unexpected_dispatch(args: argparse.Namespace) -> int | None:
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(cli, "run_m3c_command", unexpected_dispatch)
    with pytest.raises(SystemExit) as error:
        cli.main(["not-a-latentguard-command"])
    assert error.value.code == 2
    assert called is False


def test_real_main_preserves_m3a_m3b_m3c_dispatch_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def m3a(args: argparse.Namespace) -> None:
        calls.append("m3a")

    def m3b(args: argparse.Namespace) -> None:
        calls.append("m3b")

    def m3c(args: argparse.Namespace) -> int:
        calls.append("m3c")
        return 23

    monkeypatch.setattr(cli, "run_m3a_command", m3a)
    monkeypatch.setattr(cli, "run_m3b_command", m3b)
    monkeypatch.setattr(cli, "run_m3c_command", m3c)
    result = cli.main(
        [
            "prepare-selection-checkpoints",
            "--mode",
            "smoke",
            "--m3b-result-root",
            "m3b-result",
            "--m3b-runtime-root",
            "m3b-runtime",
            "--dataset-root",
            "dataset",
            "--acceptance-report",
            "acceptance.json",
            "--anchor-manifest-dir",
            "anchors",
            "--output-dir",
            "output",
        ]
    )
    assert result == 23
    assert calls == ["m3a", "m3b", "m3c"]


def test_stage_c_never_runs_before_selected_stage_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"remainder": 0}
    components = (object(), object(), object(), object())
    monkeypatch.setattr(
        "latentguard.m3c_cli._load_replay_components", lambda args: components
    )
    monkeypatch.setattr(
        "latentguard.m3c_cli._load_validated_selected_stage",
        lambda args, loaded: (_ for _ in ()).throw(ValueError("invalid Stage B")),
    )

    def forbidden(*args: object, **kwargs: object) -> tuple[object, ...]:
        calls["remainder"] += 1
        return ()

    monkeypatch.setattr("latentguard.m3c_cli._run_replay_phase", forbidden)
    with pytest.raises(ValueError, match="invalid Stage B"):
        _run_replay_complete_candidate_pools(argparse.Namespace())
    assert calls["remainder"] == 0


def test_phase_chronology_uses_only_persisted_run_times() -> None:
    selected = SimpleNamespace(
        run_manifest=SimpleNamespace(
            started_at="2026-07-16T01:00:01Z",
            finished_at="2026-07-16T01:01:00Z",
        )
    )
    remainder = SimpleNamespace(
        run_manifest=SimpleNamespace(
            started_at="2026-07-16T01:01:01Z",
            finished_at="2026-07-16T01:02:00Z",
        )
    )
    assert (
        _validated_phase_completion_timestamp(
            "2026-07-16T01:00:00Z", selected, remainder
        )
        == "2026-07-16T01:02:00Z"
    )
    remainder.run_manifest.started_at = "2026-07-16T00:59:00Z"
    with pytest.raises(ValueError, match="phase chronology"):
        _validated_phase_completion_timestamp(
            "2026-07-16T01:00:00Z", selected, remainder
        )

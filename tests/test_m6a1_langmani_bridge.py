"""CPU-only M6A.1 bridge, candidate, mask, and readiness tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from latentguard.integrations.langmani.m6a1 import (
    ACCEPTED_ACTION_ONLY_BUNDLE_DIGEST,
    LANGMANI_INITIAL_PROPOSAL_SCHEMA,
    LANGMANI_POLICY_BINDING_SCHEMA,
    LANGMANI_PROJECTED_CANDIDATES_SCHEMA,
    M5_RELEASE_MANIFEST_SHA256,
    RAW_CANDIDATE_REQUEST_SCHEMA,
    SCORE_RESULT_SCHEMA,
    M6A1BridgeError,
    array_content_digest,
    audit_readiness,
    build_raw_candidates,
    derive_probe_seed,
    frozen_candidate_source,
    m6b_source_seeds,
    make_envelope,
    scorer_identity,
    validate_envelope,
)
from latentguard.training.config import ActivationName, ModelConfig, ModelType
from latentguard.training.models import ActionOnlyMLP

ROOT = Path(__file__).resolve().parents[1]


def _binding() -> dict[str, object]:
    return make_envelope(
        LANGMANI_POLICY_BINDING_SCHEMA,
        {
            "bridge_git_commit": "a" * 40,
            "task_id": ("langmani-pick-place-task-v0:blue_cube:left_bin:canonical_v0"),
        },
    )


def _proposal() -> dict[str, object]:
    raw = np.arange(50 * 8, dtype=np.float32).reshape(50, 8) / 1000.0
    binding = _binding()
    return make_envelope(
        LANGMANI_INITIAL_PROPOSAL_SCHEMA,
        {
            "environment_step_count": 0,
            "outcomes_generated": False,
            "policy_binding_digest": binding["content_digest"],
            "policy_query_count": 1,
            "raw_postprocessed_chunk": raw.tolist(),
            "raw_prefix_digest": array_content_digest(raw[:10]),
            "reset_count": 1,
            "zero_action_execution": True,
        },
    )


def test_candidate_source_is_existing_lexical_three() -> None:
    source = frozen_candidate_source()
    selected = source["selected_transformations"]
    assert isinstance(selected, list)
    assert [item["transformation_id"] for item in selected] == [
        "additive_gaussian_noise",
        "constant_bias",
        "local_temporal_permutation",
    ]
    assert source["candidate_count"] == 4
    assert source["source_semantic"] == "ProjectedPrefixPerturbationPoolV1"


def test_build_candidates_is_deterministic_blind_and_raw_only() -> None:
    first = build_raw_candidates(_proposal())
    second = build_raw_candidates(_proposal())
    assert first == second
    payload = validate_envelope(first, expected_schema=RAW_CANDIDATE_REQUEST_SCHEMA)
    assert payload["candidate_count"] == 4
    assert payload["outcomes_available"] is False
    assert payload["environment_actions_executed"] is False
    assert isinstance(payload["candidate_pool_digest"], str)
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    assert len({item["candidate_id"] for item in candidates}) == 4
    assert all("projected_actions" not in item for item in candidates)
    assert all("success" not in str(item).lower() for item in candidates)


def test_candidate_build_rejects_proposal_with_actions_or_outcomes() -> None:
    proposal = _proposal()
    payload = dict(
        validate_envelope(proposal, expected_schema=LANGMANI_INITIAL_PROPOSAL_SCHEMA)
    )
    payload["environment_step_count"] = 1
    with pytest.raises(M6A1BridgeError, match="zero-action"):
        build_raw_candidates(make_envelope(LANGMANI_INITIAL_PROPOSAL_SCHEMA, payload))
    payload["environment_step_count"] = 0
    payload["outcomes_generated"] = True
    with pytest.raises(M6A1BridgeError, match="zero-action"):
        build_raw_candidates(make_envelope(LANGMANI_INITIAL_PROPOSAL_SCHEMA, payload))


def test_envelope_rejects_unknown_fields_and_nonfinite_values() -> None:
    binding = _binding()
    unknown = dict(binding)
    unknown["path"] = "secret"
    with pytest.raises(M6A1BridgeError, match="unexpected or missing"):
        validate_envelope(unknown, expected_schema=LANGMANI_POLICY_BINDING_SCHEMA)
    with pytest.raises(ValueError):
        make_envelope("fixture", {"value": float("nan")})


def test_action_only_model_masks_six_nonzero_padding_slots_bitwise() -> None:
    config = ModelConfig(
        model_type=ModelType.ACTION_ONLY_MLP,
        state_dimension=38,
        action_dimension=8,
        action_horizon=16,
        hidden_dimensions=(16, 8),
        state_projection_dimension=None,
        action_projection_dimension=32,
        temporal_embedding_dimension=None,
        transformer_layers=None,
        attention_heads=None,
        transformer_feedforward_dimension=None,
        dropout=0.0,
        activation=ActivationName.RELU,
    )
    model = ActionOnlyMLP(config).eval()
    state = torch.zeros((4, 38), dtype=torch.float32)
    actions = torch.arange(4 * 16 * 8, dtype=torch.float32).reshape(4, 16, 8)
    mask = torch.zeros((4, 16), dtype=torch.bool)
    mask[:, :10] = True
    canonical = actions.clone()
    canonical[:, 10:] = 0.0
    altered = canonical.clone()
    altered[:, 10:] = torch.arange(4 * 6 * 8, dtype=torch.float32).reshape(4, 6, 8) + 1
    with torch.inference_mode():
        reference = model(state, canonical, mask)
        observed = model(state, altered, mask)
    assert torch.equal(reference, observed)


def test_projected_input_freezes_ten_to_sixteen_mask_contract() -> None:
    raw = build_raw_candidates(_proposal())
    raw_payload = validate_envelope(raw, expected_schema=RAW_CANDIDATE_REQUEST_SCHEMA)
    candidates = []
    for item in raw_payload["candidates"]:  # type: ignore[union-attr]
        values = np.asarray(item["raw_actions"], dtype=np.float32)
        candidates.append(
            {
                "candidate_id": item["candidate_id"],
                "correction_count": 0,
                "correction_mask": np.zeros((10, 8), dtype=np.bool_).tolist(),
                "nonfinite_count": 0,
                "projected_actions": values.tolist(),
                "projected_candidate_digest": array_content_digest(values),
                "projection_semantic": (
                    "langmani_existing_active_action_bounds_projection_v1"
                ),
                "raw_candidate_digest": item["raw_candidate_digest"],
                "transformation_config_digest": item["transformation_config_digest"],
                "transformation_id": item["transformation_id"],
            }
        )
    projected = make_envelope(
        LANGMANI_PROJECTED_CANDIDATES_SCHEMA,
        {
            "action_bounds_digest": "sha256:" + "a" * 64,
            "candidate_count": 4,
            "candidate_pool_digest": raw_payload["candidate_pool_digest"],
            "candidates": candidates,
            "environment_step_count": 0,
            "outcomes_available": False,
            "policy_binding_digest": raw_payload["policy_binding_digest"],
            "projection_semantic": (
                "langmani_existing_active_action_bounds_projection_v1"
            ),
            "raw_candidate_manifest_digest": raw["content_digest"],
            "task_id": "langmani-pick-place-task-v0:blue_cube:left_bin:canonical_v0",
            "zero_action_execution": True,
        },
    )
    from latentguard.integrations.langmani.m6a1 import _projected_inputs

    ids, actions, masks = _projected_inputs(projected)
    assert len(ids) == 4
    assert actions.shape == (4, 16, 8)
    assert actions.dtype == np.dtype("<f8")
    assert np.all(actions[:, 10:] == 0.0)
    assert masks.dtype == np.dtype(np.bool_)
    assert np.all(masks[:, :10]) and not np.any(masks[:, 10:])


def test_scorer_identity_and_seed_schedules_are_frozen() -> None:
    identity = scorer_identity(
        ROOT
        / "reports/m3c/20260716T152012Z_m3c-full_a632a70_seed271828"
        / "full/selection/selector-configuration.json"
    )
    assert identity["bundle_digest"] == ACCEPTED_ACTION_ONLY_BUNDLE_DIGEST
    assert identity["ensemble_seed_count"] == 5
    assert derive_probe_seed() == 986203210
    assert m6b_source_seeds() == (282896382, 571862862, 1651966581)
    assert derive_probe_seed() not in m6b_source_seeds()


def test_structural_readiness_preserves_release_and_m6a_evidence() -> None:
    report = audit_readiness(repository_root=ROOT)
    payload = validate_envelope(
        report, expected_schema="LatentGuardLangManiM6BReadinessV1"
    )
    gates = payload["gates"]
    assert isinstance(gates, dict)
    assert gates["m5_release_manifest_unchanged"] is True
    assert gates["m6a_historical_blockers_preserved"] is True
    assert payload["readiness_status"] == "conditionally_ready"
    observed = (
        "sha256:"
        + hashlib.sha256(
            (ROOT / "docs/release/release-manifest.json").read_bytes()
        ).hexdigest()
    )
    assert observed == M5_RELEASE_MANIFEST_SHA256


def test_score_result_schema_is_outcome_free() -> None:
    report = make_envelope(
        SCORE_RESULT_SCHEMA,
        {
            "outcomes_available_during_selection": False,
            "selected_candidate_id": "opaque",
        },
    )
    payload = validate_envelope(report, expected_schema=SCORE_RESULT_SCHEMA)
    assert payload["outcomes_available_during_selection"] is False

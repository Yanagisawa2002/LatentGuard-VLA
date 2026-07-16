"""CPU-only tests for the strict M3C Stage-A blinded input boundary."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from latentguard.selection.blind_input import (
    BLIND_CANDIDATE_POOL_MANIFEST_NAME,
    BlindCandidateGroupV1,
    BlindCandidateInputError,
    BlindCandidatePoolV1,
    build_blind_candidate_pool,
    load_blind_candidate_pool,
    save_blind_candidate_pool,
    validate_blind_candidate_pool_against_full_pool,
)
from latentguard.selection.blind_protocol import (
    EXPECTED_STAGE_A_SELECTOR_IDS,
    EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS,
    BlindProtocolError,
    finalize_blind_selection_from_blind_pool,
    run_blind_selection_stage_a_from_blind_pool,
    validate_selection_manifest_against_pool,
)
from latentguard.selection.checkpoint_bundle import BundleLoadingDiagnosticsV1
from latentguard.selection.ensemble import (
    EnsembleEndToEndProfileV1,
    EnsembleInferenceDiagnosticsV1,
    EnsembleInferenceResultV1,
)
from latentguard.selection.models import (
    CandidateDistribution,
    CandidateGroupV1,
    CandidatePoolV1,
    CandidateRefV1,
    SelectorDecisionV1,
    SourceTrajectoryIdentityV1,
    array_content_digest,
)
from latentguard.selection.selectors import (
    SelectorError,
    action_magnitude_decision,
    deterministic_random_decision,
    learned_ensemble_decision,
    oracle_analysis_decision,
)
from latentguard.training.baselines import ActionMagnitudeBaselineV1


def _sha(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _full_pool() -> CandidatePoolV1:
    candidates = tuple(
        CandidateRefV1(
            proposal_id=f"opaque-candidate-{index}",
            configuration_ordinal=index,
            distribution=(
                CandidateDistribution.ID_LIKE
                if index < 4
                else CandidateDistribution.SHIFTED
            ),
            corruption_type=f"private-transform-{index}",
            severity_id=f"private-severity-{index}",
            seed=100 + index,
            action_chunk=np.full((16, 8), (index + 1) / 20.0, dtype=np.float32),
            action_mask=np.ones(16, dtype=np.bool_),
        )
        for index in range(8)
    )
    group = CandidateGroupV1(
        anchor_id="private-anchor",
        trajectory=SourceTrajectoryIdentityV1(
            source_trajectory_id="private-trajectory",
            source_seed=12345,
            split_group_id="private-split",
            complete_state_digests=(_sha("private-state"),),
        ),
        state_content_digest=_sha("state-content"),
        verifier_state_content_digest=_sha("verifier-state-content"),
        continuation_identity=_sha("private-continuation"),
        source_action_prefix_digest=array_content_digest(
            np.zeros((16, 8), dtype=np.float32)
        ),
        state_vector=np.linspace(-1.0, 1.0, 38, dtype=np.float32),
        continuation_actions=np.zeros((3, 8), dtype=np.float32),
        candidates=candidates,
    )
    return CandidatePoolV1(
        source_set_digest=_sha("source-set"),
        candidate_pool_configuration_digest=_sha("pool-configuration"),
        action_contract_digest=_sha("action-contract"),
        exclusion_inventory_digest=_sha("exclusions"),
        groups=(group,),
    )


def _bundle_digests() -> dict[str, str]:
    return {key: _sha(f"{key}-bundle") for key in EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS}


def _decisions(blinded: BlindCandidatePoolV1) -> tuple[SelectorDecisionV1, ...]:
    return tuple(
        SelectorDecisionV1(
            selector_id=selector_id,
            group_id=group.group_id,
            selected_proposal_id=tuple(reversed(group.candidate_ids))[0],
            ranking=tuple(reversed(group.candidate_ids)),
            predicted_failure_probabilities={},
            abstained=False,
        )
        for selector_id in EXPECTED_STAGE_A_SELECTOR_IDS
        for group in blinded.groups
    )


def _baseline() -> ActionMagnitudeBaselineV1:
    return ActionMagnitudeBaselineV1(
        dataset_digest=_sha("dataset"),
        training_split_digest=_sha("split"),
        training_sample_count=10,
        valid_action_step_count=160,
        minimum_standard_deviation=1e-6,
        action_mean=np.zeros(8, dtype=np.float64),
        feature_mean=np.zeros(3, dtype=np.float64),
        feature_standard_deviation=np.ones(3, dtype=np.float64),
    )


def _inference(candidate_ids: tuple[str, ...]) -> EnsembleInferenceResultV1:
    probabilities = np.linspace(0.1, 0.8, 8, dtype=np.float64)
    per_seed = np.stack([probabilities] * 5, axis=0)
    diagnostics = EnsembleInferenceDiagnosticsV1(
        bundle_loading=BundleLoadingDiagnosticsV1(
            device="cpu",
            total_seconds=0.0,
            per_seed_model_seconds=(0.0,) * 5,
        ),
        end_to_end_profile=EnsembleEndToEndProfileV1(
            device="cpu",
            candidate_count=8,
            group_durations_seconds=(0.008,) * 5,
        ),
    )
    return EnsembleInferenceResultV1(
        candidate_ids=candidate_ids,
        seed_order=(0, 1, 2, 3, 4),
        per_seed_raw_logits=np.zeros((5, 8), dtype=np.float64),
        per_seed_calibrated_failure_probabilities=per_seed,
        ensemble_failure_probabilities=probabilities,
        ranking=candidate_ids,
        diagnostics=diagnostics,
    )


def _manifest_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(key)
            keys.update(_manifest_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_manifest_keys(item))
    return keys


def _file_sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def test_projection_and_manifest_expose_only_stage_a_fields(tmp_path: Path) -> None:
    full = _full_pool()
    blinded = build_blind_candidate_pool(full)
    group = blinded.groups[0]

    assert {item.name for item in fields(BlindCandidateGroupV1)} == {
        "action_chunks",
        "action_masks",
        "candidate_ids",
        "group_id",
        "schema_version",
        "state_vector",
    }
    assert {item.name for item in fields(BlindCandidatePoolV1)} == {
        "action_contract_digest",
        "candidate_pool_configuration_digest",
        "full_candidate_pool_digest",
        "groups",
        "schema_version",
        "source_set_digest",
    }
    assert group.state_vector.shape == (38,)
    assert group.action_chunks.shape == (8, 16, 8)
    assert group.action_masks.shape == (8, 16)
    assert not group.state_vector.flags.writeable
    assert not group.action_chunks.flags.writeable
    assert not group.action_masks.flags.writeable
    with pytest.raises(BlindCandidateInputError, match="float32"):
        replace(group, state_vector=group.state_vector.astype(np.float64))
    for forbidden_attribute in (
        "anchor_id",
        "trajectory",
        "candidates",
        "continuation_identity",
    ):
        assert not hasattr(group, forbidden_attribute)

    output = save_blind_candidate_pool(blinded, tmp_path / "blinded")
    raw = json.loads(
        (output / BLIND_CANDIDATE_POOL_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    forbidden_keys = {
        "anchor_id",
        "continuation",
        "continuation_identity",
        "corruption_type",
        "distribution",
        "evidence",
        "outcome",
        "seed",
        "severity_id",
        "trajectory",
    }
    assert not _manifest_keys(raw).intersection(forbidden_keys)
    loaded = load_blind_candidate_pool(
        output, expected_content_digest=blinded.content_digest
    )
    assert loaded.content_digest == blinded.content_digest
    validate_blind_candidate_pool_against_full_pool(loaded, full)


@pytest.mark.parametrize(
    "tamper",
    ("unknown_file", "duplicate_field", "trailing_bytes", "hard_link"),
)
def test_blinded_pool_loader_rejects_unsafe_or_tampered_bundles(
    tmp_path: Path,
    tamper: str,
) -> None:
    output = save_blind_candidate_pool(
        build_blind_candidate_pool(_full_pool()), tmp_path / tamper
    )
    manifest_path = output / BLIND_CANDIDATE_POOL_MANIFEST_NAME
    if tamper == "unknown_file":
        (output / "unexpected.bin").write_bytes(b"unexpected")
    elif tamper == "duplicate_field":
        text = manifest_path.read_text(encoding="utf-8")
        manifest_path.write_text(
            text.replace('"format":', '"format":"duplicate","format":', 1),
            encoding="utf-8",
        )
    else:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative = raw["groups"][0]["actions"]["path"]
        action_path = output / Path(*relative.split("/"))
        if tamper == "trailing_bytes":
            with action_path.open("ab") as stream:
                stream.write(b"x")
            raw["groups"][0]["actions"]["file_digest"] = _file_sha256(action_path)
            manifest_path.write_text(
                json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
        else:
            os.link(action_path, output / "hard-linked-array.npy")

    with pytest.raises(BlindCandidateInputError):
        load_blind_candidate_pool(output)


def test_full_pool_validation_rejects_binding_and_group_drift() -> None:
    full = _full_pool()
    blinded = build_blind_candidate_pool(full)
    changed_full = replace(full, action_contract_digest=_sha("changed-contract"))
    with pytest.raises(BlindCandidateInputError, match="digest binding"):
        validate_blind_candidate_pool_against_full_pool(blinded, changed_full)

    group = blinded.groups[0]
    changed_actions = np.array(group.action_chunks, copy=True)
    changed_actions[0, 0, 0] += 0.125
    changed_group = replace(group, action_chunks=changed_actions)
    changed_blinded = replace(blinded, groups=(changed_group,))
    with pytest.raises(BlindCandidateInputError, match="content differs"):
        validate_blind_candidate_pool_against_full_pool(changed_blinded, full)


def test_nonoracle_selectors_accept_only_blinded_groups() -> None:
    full_group = _full_pool().groups[0]
    blind_group = build_blind_candidate_pool(_full_pool()).groups[0]

    assert deterministic_random_decision(
        blind_group, seed=77
    ) == deterministic_random_decision(blind_group, seed=77)
    baseline = _baseline()
    assert (
        action_magnitude_decision(
            blind_group,
            baseline,
            expected_baseline_digest=baseline.content_digest,
        ).ranking
        == blind_group.candidate_ids
    )
    inference = _inference(blind_group.candidate_ids)
    assert (
        learned_ensemble_decision(
            blind_group,
            inference,
            selector_id="temporal_ensemble_v1",
        ).ranking
        == blind_group.candidate_ids
    )
    with pytest.raises(SelectorError, match="BlindCandidateGroupV1"):
        deterministic_random_decision(full_group, seed=77)  # type: ignore[arg-type]
    with pytest.raises(SelectorError, match="BlindCandidateGroupV1"):
        action_magnitude_decision(  # type: ignore[arg-type]
            full_group,
            baseline,
            expected_baseline_digest=baseline.content_digest,
        )
    with pytest.raises(SelectorError, match="BlindCandidateGroupV1"):
        learned_ensemble_decision(  # type: ignore[arg-type]
            full_group,
            inference,
            selector_id="temporal_ensemble_v1",
        )
    with pytest.raises(SelectorError, match="CandidateGroupV1"):
        oracle_analysis_decision(blind_group, ())  # type: ignore[arg-type]


def test_stage_a_finalizes_from_blinded_pool_then_validates_against_full(
    tmp_path: Path,
) -> None:
    full = _full_pool()
    blinded = build_blind_candidate_pool(full)
    path, envelope = run_blind_selection_stage_a_from_blind_pool(
        blinded,
        _decisions(blinded),
        selector_configuration_digest=_sha("selectors"),
        verifier_bundle_digests=_bundle_digests(),
        selection_timestamp_utc="2026-07-16T09:00:00Z",
        output_root=tmp_path / "stage-a",
    )

    assert path.is_file()
    assert envelope.manifest.candidate_pool_digest == full.content_digest
    assert envelope.manifest.outcomes_available_during_selection is False
    validate_selection_manifest_against_pool(envelope.manifest, full)


@pytest.mark.parametrize(
    "change",
    (
        "missing_selector",
        "incomplete_group",
        "extra_oracle",
        "missing_bundle",
        "extra_bundle",
    ),
)
def test_stage_a_rejects_nonexact_selector_or_bundle_inventory(change: str) -> None:
    blinded = build_blind_candidate_pool(_full_pool())
    if change == "incomplete_group":
        first = blinded.groups[0]
        second = replace(
            first,
            group_id="opaque-second-group",
            candidate_ids=tuple(f"second-{item}" for item in first.candidate_ids),
        )
        blinded = replace(blinded, groups=(first, second))
    decisions = _decisions(blinded)
    bundles = _bundle_digests()
    if change == "missing_selector":
        decisions = decisions[:-1]
    elif change == "incomplete_group":
        decisions = tuple(
            item
            for item in decisions
            if not (
                item.selector_id == EXPECTED_STAGE_A_SELECTOR_IDS[0]
                and item.group_id == blinded.groups[1].group_id
            )
        )
    elif change == "extra_oracle":
        decisions = (
            *decisions,
            replace(decisions[0], selector_id="oracle_analysis_only_v1"),
        )
    elif change == "missing_bundle":
        bundles.pop(EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS[-1])
    else:
        bundles["unexpected_architecture"] = _sha("unexpected")

    with pytest.raises(BlindProtocolError, match="inventory"):
        finalize_blind_selection_from_blind_pool(
            blinded,
            decisions,
            selector_configuration_digest=_sha("selectors"),
            verifier_bundle_digests=bundles,
            selection_timestamp_utc="2026-07-16T09:00:00Z",
        )

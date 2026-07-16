"""CPU-only tests for M3C outcome isolation and result binding."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from latentguard.selection.blind_protocol import (
    EXPECTED_STAGE_A_SELECTOR_IDS,
    EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS,
    ORACLE_SELECTOR_ID,
    BlindProtocolError,
    BlindSelectorInputV1,
    bind_complete_pool_result,
    blind_selector_input_allowlist_digest,
    compute_full_pool_outcome_digest,
    create_oracle_decisions,
    finalize_blind_selection,
    load_bound_selection_result,
    run_blind_selection_stage_a,
    run_fake_blind_protocol_smoke,
    save_bound_selection_result,
    validate_bound_selection_result,
    validate_selection_manifest_against_pool,
)
from latentguard.selection.manifest import (
    BlindSelectionManifestError,
    initialize_blind_selection_output_root,
    load_blind_selection_manifest,
)
from latentguard.selection.metrics import CandidateOutcomeV1
from latentguard.selection.models import (
    BlindSelectionManifestV1,
    CandidateDistribution,
    CandidateGroupV1,
    CandidatePoolV1,
    CandidateRefV1,
    SelectorDecisionV1,
    SourceTrajectoryIdentityV1,
    array_content_digest,
)


def _sha(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _pool() -> CandidatePoolV1:
    source = np.zeros((16, 8), dtype=np.float64)
    candidates = tuple(
        CandidateRefV1(
            proposal_id=f"proposal-{index}",
            configuration_ordinal=index,
            distribution=(
                CandidateDistribution.ID_LIKE
                if index < 4
                else CandidateDistribution.SHIFTED
            ),
            corruption_type=f"corruption-{index}",
            severity_id=f"severity-{index}",
            seed=index,
            action_chunk=np.full((16, 8), float(index + 1), dtype=np.float64),
            action_mask=np.ones((16,), dtype=np.bool_),
        )
        for index in range(8)
    )
    group = CandidateGroupV1(
        anchor_id="anchor-0",
        trajectory=SourceTrajectoryIdentityV1(
            source_trajectory_id="trajectory-new-0",
            source_seed=9001,
            split_group_id="split-new-0",
            complete_state_digests=(_sha("state-0"),),
        ),
        state_content_digest=_sha("state-tree-0"),
        verifier_state_content_digest=_sha("verifier-state-0"),
        continuation_identity=_sha("continuation-0"),
        source_action_prefix_digest=array_content_digest(source),
        state_vector=np.arange(38, dtype=np.float64),
        continuation_actions=np.ones((2, 8), dtype=np.float64),
        candidates=candidates,
    )
    return CandidatePoolV1(
        source_set_digest=_sha("source-set"),
        candidate_pool_configuration_digest=_sha("pool-configuration"),
        action_contract_digest=_sha("action-contract"),
        exclusion_inventory_digest=_sha("m3a-exclusions"),
        groups=(group,),
    )


def _decision(
    pool: CandidatePoolV1,
    *,
    selector_id: str = "temporal-ensemble",
) -> SelectorDecisionV1:
    group = pool.groups[0]
    ranking = tuple(reversed(group.proposal_ids))
    return SelectorDecisionV1(
        selector_id=selector_id,
        group_id=group.group_id,
        selected_proposal_id=ranking[0],
        ranking=ranking,
        predicted_failure_probabilities={
            proposal_id: index / 10.0
            for index, proposal_id in enumerate(reversed(ranking))
        },
        abstained=False,
    )


def _decisions(pool: CandidatePoolV1) -> tuple[SelectorDecisionV1, ...]:
    return tuple(
        _decision(pool, selector_id=selector_id)
        for selector_id in EXPECTED_STAGE_A_SELECTOR_IDS
    )


def _bundle_digests() -> dict[str, str]:
    return {key: _sha(f"{key}-bundle") for key in EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS}


def _outcomes(pool: CandidatePoolV1) -> tuple[CandidateOutcomeV1, ...]:
    group = pool.groups[0]
    return tuple(
        CandidateOutcomeV1(
            proposal_id=candidate.proposal_id,
            group_id=group.group_id,
            source_trajectory_id=group.source_trajectory_id,
            distribution=candidate.distribution.value,
            evidence_id=f"fake-strong-evidence-{candidate.proposal_id}",
            status="conclusive",
            success=candidate.configuration_ordinal in {1, 6},
            unsafe=False,
            simulator_replay_verified=True,
            label_strength="strong",
            state_component_count=70,
        )
        for candidate in group.candidates
    )


def _envelope(pool: CandidatePoolV1, *, timestamp: str = "2026-07-16T01:00:00Z"):
    return finalize_blind_selection(
        pool,
        _decisions(pool),
        selector_configuration_digest=_sha("selectors"),
        verifier_bundle_digests=_bundle_digests(),
        selection_timestamp_utc=timestamp,
    )


def test_selector_input_is_exact_allowlist_and_detached() -> None:
    pool = _pool()
    group = pool.groups[0]
    value = BlindSelectorInputV1.from_candidate_group(group)
    assert set(value.as_mapping()) == {
        "candidate_ids",
        "state_vectors",
        "action_chunks",
        "action_masks",
    }
    assert value.state_vectors.shape == (8, 38)
    assert value.action_chunks.shape == (8, 16, 8)
    assert value.action_masks.shape == (8, 16)
    assert not np.shares_memory(
        value.action_chunks[0], group.candidates[0].action_chunk
    )
    payload = dict(value.as_mapping())
    payload["outcome"] = True
    with pytest.raises(BlindProtocolError, match="allowlist"):
        BlindSelectorInputV1.from_mapping(payload)
    assert blind_selector_input_allowlist_digest().startswith("sha256:")


def test_manifest_timestamp_is_nonsemantic_but_strictly_enveloped() -> None:
    pool = _pool()
    first = _envelope(pool, timestamp="2026-07-16T01:00:00Z")
    second = _envelope(pool, timestamp="2026-07-16T01:00:01Z")
    assert first.semantic_digest == second.semantic_digest
    assert first.envelope_digest != second.envelope_digest
    assert first.manifest.outcomes_available_during_selection is False
    assert first.manifest.outcome_input_paths == ()
    encoded = json.dumps(first.manifest.as_mapping())
    assert "state_vector" not in encoded
    assert "action_chunk" not in encoded
    assert "simulator_replay" not in encoded
    assert "evidence_id" not in encoded


def test_stage_a_requires_empty_independent_root_and_detects_audit_tamper(
    tmp_path: Path,
) -> None:
    pool = _pool()
    path, expected = run_blind_selection_stage_a(
        pool,
        _decisions(pool),
        selector_configuration_digest=_sha("selectors"),
        verifier_bundle_digests=_bundle_digests(),
        selection_timestamp_utc="2026-07-16T01:00:00Z",
        output_root=tmp_path / "stage-a",
    )
    loaded = load_blind_selection_manifest(path)
    assert loaded == expected
    with pytest.raises(BlindSelectionManifestError, match="must be empty"):
        initialize_blind_selection_output_root(tmp_path / "stage-a")

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["manifest"]["selection_timestamp_utc"] = "2026-07-16T01:00:01Z"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BlindSelectionManifestError, match="strict manifest content"):
        load_blind_selection_manifest(path)


def test_manifest_validation_rejects_changed_ranking() -> None:
    pool = _pool()
    envelope = _envelope(pool)
    original = envelope.manifest.selections[0]
    changed_ranking = tuple(reversed(original.ranking))
    changed = replace(
        original,
        selected_proposal_id=changed_ranking[0],
        ranking=changed_ranking,
    )
    changed_manifest = replace(
        envelope.manifest,
        selections=(changed, *envelope.manifest.selections[1:]),
    )
    validate_selection_manifest_against_pool(changed_manifest, pool)
    assert changed_manifest.content_digest != envelope.manifest.content_digest

    foreign = replace(changed, group_id="unknown-group")
    with pytest.raises(BlindProtocolError, match="unknown candidate group"):
        validate_selection_manifest_against_pool(
            replace(
                changed_manifest,
                selections=(foreign, *changed_manifest.selections[1:]),
            ),
            pool,
        )


def test_complete_result_binds_original_choices_pool_bundles_and_evidence(
    tmp_path: Path,
) -> None:
    pool = _pool()
    envelope = _envelope(pool)
    outcomes = _outcomes(pool)
    result = bind_complete_pool_result(
        envelope,
        pool,
        outcomes,
        replay_evidence_digest=_sha("joined-evidence"),
        simulator_compatibility_identity=_sha("maniskill-compatibility"),
        git_sha="1" * 40,
        outcomes_generated_at_utc="2026-07-16T01:00:01Z",
    )
    path = save_bound_selection_result(tmp_path / "result.json", result)
    assert (
        load_bound_selection_result(
            path,
            selection_envelope=envelope,
            candidate_pool=pool,
        )
        == result
    )
    changed_bundle = replace(
        result,
        verifier_bundle_digests=(("temporal", _sha("changed-bundle")),),
    )
    with pytest.raises(BlindProtocolError, match="checkpoint identity"):
        validate_bound_selection_result(changed_bundle, envelope, pool)

    original = envelope.manifest.selections[0]
    ranking = tuple(reversed(original.ranking))
    changed_decision = replace(
        original,
        selected_proposal_id=ranking[0],
        ranking=ranking,
    )
    changed_manifest = replace(
        envelope.manifest,
        selections=(changed_decision, *envelope.manifest.selections[1:]),
    )
    changed_envelope = type(envelope).create(changed_manifest)
    with pytest.raises(BlindProtocolError, match="selected candidate or ranking"):
        validate_bound_selection_result(result, changed_envelope, pool)


def test_full_pool_rejects_unknown_missing_or_mismatched_outcomes() -> None:
    pool = _pool()
    envelope = _envelope(pool)
    outcomes = _outcomes(pool)
    with pytest.raises(BlindProtocolError, match="one result per pool proposal"):
        bind_complete_pool_result(
            envelope,
            pool,
            outcomes[:-1],
            replay_evidence_digest=_sha("evidence"),
            simulator_compatibility_identity=_sha("compatibility"),
            git_sha="2" * 40,
            outcomes_generated_at_utc="2026-07-16T01:00:01Z",
        )
    unknown = replace(outcomes[0], proposal_id="unknown-proposal")
    with pytest.raises(BlindProtocolError, match="unknown proposal"):
        bind_complete_pool_result(
            envelope,
            pool,
            (unknown, *outcomes[1:]),
            replay_evidence_digest=_sha("evidence"),
            simulator_compatibility_identity=_sha("compatibility"),
            git_sha="2" * 40,
            outcomes_generated_at_utc="2026-07-16T01:00:01Z",
        )
    with pytest.raises(BlindProtocolError, match="persisted outcome digest"):
        bind_complete_pool_result(
            envelope,
            pool,
            outcomes,
            replay_evidence_digest=_sha("evidence"),
            simulator_compatibility_identity=_sha("compatibility"),
            git_sha="2" * 40,
            outcomes_generated_at_utc="2026-07-16T01:00:01Z",
            expected_full_pool_outcome_digest=_sha("wrong"),
        )
    with pytest.raises(BlindProtocolError, match="do not postdate"):
        bind_complete_pool_result(
            envelope,
            pool,
            outcomes,
            replay_evidence_digest=_sha("evidence"),
            simulator_compatibility_identity=_sha("compatibility"),
            git_sha="2" * 40,
            outcomes_generated_at_utc="2026-07-16T00:59:59Z",
        )


def test_oracle_is_unavailable_until_complete_outcomes_exist() -> None:
    pool = _pool()
    outcomes = _outcomes(pool)
    digest = compute_full_pool_outcome_digest(outcomes)
    with pytest.raises(BlindProtocolError, match="one result per pool proposal"):
        create_oracle_decisions(
            pool,
            outcomes[:-1],
            full_pool_outcome_digest=digest,
        )
    with pytest.raises(BlindProtocolError, match="unavailable or mismatched"):
        create_oracle_decisions(
            pool,
            outcomes,
            full_pool_outcome_digest=_sha("wrong"),
        )
    oracle = create_oracle_decisions(
        pool,
        outcomes,
        full_pool_outcome_digest=digest,
    )
    assert len(oracle) == 1
    assert oracle[0].selector_id == ORACLE_SELECTOR_ID
    assert outcomes[1].proposal_id == oracle[0].selected_proposal_id
    assert oracle[0].ranking[:2] == ("proposal-1", "proposal-6")


def test_complete_fake_blind_protocol_smoke_is_explicitly_nonphysical(
    tmp_path: Path,
) -> None:
    pool = _pool()
    report = run_fake_blind_protocol_smoke(
        tmp_path / "fake-blind-smoke",
        pool,
        _decisions(pool),
        _outcomes(pool),
        selector_configuration_digest=_sha("selectors"),
        verifier_bundle_digests=_bundle_digests(),
        selection_timestamp_utc="2026-07-16T01:00:00Z",
        outcomes_generated_at_utc="2026-07-16T01:00:01Z",
        replay_evidence_digest=_sha("fake-evidence-inventory"),
        simulator_compatibility_identity=_sha("fake-compatibility"),
        git_sha="3" * 40,
    )
    assert report.candidate_count == 8
    assert report.selector_count == 11
    assert report.oracle_decision_count == 1
    assert report.physical_simulator_evidence is False
    assert (
        tmp_path / "fake-blind-smoke" / "stage-a" / "blind-selection-manifest.json"
    ).is_file()
    assert (
        tmp_path / "fake-blind-smoke" / "stage-bc" / "bound-selection-result.json"
    ).is_file()


def test_manifest_model_still_rejects_outcome_fields() -> None:
    pool = _pool()
    envelope = _envelope(pool)
    mapping = envelope.manifest.as_mapping()
    mapping["outcome"] = True
    with pytest.raises(Exception, match="unexpected or missing fields"):
        BlindSelectionManifestV1.from_mapping(mapping)

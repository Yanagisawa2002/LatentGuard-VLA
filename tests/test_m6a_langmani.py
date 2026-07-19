"""CPU-only tests for the M6A LangMani contract and static audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from latentguard.integrations.langmani.audit import (
    LangManiAuditError,
    audit_langmani_contract,
    discover_langmani_checkout,
)
from latentguard.integrations.langmani.models import (
    LangManiActionProposalV1,
    LangManiContractError,
    LangManiIntegrationManifestV1,
    LangManiObservationEnvelopeV1,
    LangManiObservationFieldV1,
    LangManiOutcomeEvidenceV1,
    LangManiPolicyBindingV1,
    LangManiReplaySnapshotV1,
    LangManiTaskContextV1,
    ObservationRole,
    OutcomeStatus,
    ProposalStage,
    content_digest,
)
from latentguard.integrations.langmani.synthetic import (
    SYNTHETIC_WARNING,
    SyntheticLangManiAdapter,
    validate_synthetic_adapter,
)


def _task() -> LangManiTaskContextV1:
    return SyntheticLangManiAdapter().task_context()


def _policy() -> LangManiPolicyBindingV1:
    return SyntheticLangManiAdapter().policy_binding()


def test_all_versioned_schemas_round_trip_and_reject_unknown_fields() -> None:
    adapter = SyntheticLangManiAdapter(seed=3)
    task = adapter.task_context()
    policy = adapter.policy_binding()
    observation = adapter.observation()
    raw = adapter.propose()
    projected = adapter.executable_proposal(raw)
    snapshot = adapter.snapshot()
    evidence = adapter.evaluate_outcome(projected)
    manifest = LangManiIntegrationManifestV1(
        latentguard_commit="a" * 40,
        langmani_commit="b" * 40,
        integration_schema_version=task.schema_version,
        task_contract_digest=task.digest,
        observation_contract_digest=observation.digest,
        action_contract_digest=raw.digest,
        projection_contract_digest=projected.digest,
        replay_contract_digest=snapshot.digest,
        success_contract_digest=evidence.digest,
        compatibility_matrix_digest=content_digest({"rows": []}),
        readiness_gate_digest=content_digest({"gates": []}),
        overall_readiness="blocked",
        unresolved_blocker_ids=("M6A-B003",),
    )
    pairs = (
        (task, LangManiTaskContextV1),
        (policy, LangManiPolicyBindingV1),
        (observation, LangManiObservationEnvelopeV1),
        (raw, LangManiActionProposalV1),
        (snapshot, LangManiReplaySnapshotV1),
        (evidence, LangManiOutcomeEvidenceV1),
        (manifest, LangManiIntegrationManifestV1),
    )
    for record, record_type in pairs:
        payload = record.to_dict()
        assert record_type.from_dict(payload).digest == record.digest
        payload["unknown"] = True
        with pytest.raises(LangManiContractError, match="fields differ"):
            record_type.from_dict(payload)


def test_observation_roles_prevent_reporting_and_outcome_leakage() -> None:
    adapter = SyntheticLangManiAdapter()
    envelope = adapter.observation()
    assert "fixture_label" not in envelope.policy_inputs
    assert "fixture_label" not in envelope.verifier_inputs
    assert "outcome_hidden" not in envelope.policy_inputs
    assert "outcome_hidden" not in envelope.verifier_inputs

    payload = envelope.to_dict()
    payload["verifier_inputs"] = {"outcome_hidden": True}
    with pytest.raises(LangManiContractError, match="role mismatch"):
        LangManiObservationEnvelopeV1.from_dict(payload)


def test_task_allowlists_are_disjoint_and_text_is_not_model_input() -> None:
    task = _task()
    payload = task.to_dict()
    payload["identity_only_fields"] = ["normalized_task_id"]
    payload["model_input_fields"] = ["instruction_text"]
    with pytest.raises(LangManiContractError, match="instruction text"):
        LangManiTaskContextV1.from_dict(payload)

    payload = task.to_dict()
    payload["reporting_only_fields"] = ["normalized_task_id"]
    with pytest.raises(LangManiContractError, match="disjoint"):
        LangManiTaskContextV1.from_dict(payload)


def test_raw_projected_and_executed_stages_have_distinct_identities() -> None:
    adapter = SyntheticLangManiAdapter()
    raw = adapter.propose()
    projected = adapter.executable_proposal(raw)
    executed_payload = projected.to_dict()
    executed_payload["proposal_id"] = "synthetic-executed"
    executed_payload["stage"] = ProposalStage.EXECUTED.value
    executed_payload["source_proposal_digest"] = projected.digest
    executed = LangManiActionProposalV1.from_dict(executed_payload)
    assert len({raw.digest, projected.digest, executed.digest}) == 3
    assert raw.actions[0][-1] == 1.25
    assert projected.actions[0][-1] == 1.0


def test_policy_state_is_separate_and_missing_real_state_is_representable() -> None:
    snapshot = SyntheticLangManiAdapter().snapshot()
    payload = snapshot.to_dict()
    payload["policy_state_digest"] = None
    payload["observation_history_digest"] = None
    payload["restoration_mode"] = "simulator_state_without_policy_history"
    missing = LangManiReplaySnapshotV1.from_dict(payload)
    assert missing.policy_state_digest is None
    assert missing.observation_history_digest is None


def test_outcome_evidence_preserves_non_boolean_statuses() -> None:
    snapshot = SyntheticLangManiAdapter().snapshot()
    evidence = LangManiOutcomeEvidenceV1(
        evidence_id="indeterminate",
        proposal_digest="sha256:" + "1" * 64,
        replay_snapshot_digest=snapshot.digest,
        status=OutcomeStatus.INDETERMINATE,
        official_success=None,
        official_failure=None,
        horizon_exhausted=False,
        unsafe_proxy=None,
        execution_error=None,
        executed_action_count=0,
        continuation_semantic="not_selected",
        simulator_verified=False,
        reporting_metadata={},
    )
    assert evidence.status is OutcomeStatus.INDETERMINATE
    assert evidence.official_success is None


def test_synthetic_adapter_projection_restore_blindness_and_zero_work_resume() -> None:
    report = validate_synthetic_adapter()
    assert report["passed"]
    assert report["zero_work_resume"]
    assert report["blindness_validated"]
    assert report["warning"] == SYNTHETIC_WARNING


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _write_fixture_file(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _langmani_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "LangMani"
    root.mkdir(parents=True)
    _write_fixture_file(
        root,
        "pyproject.toml",
        """
[project]
name = "langmani"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["numpy==2.2.6", "mani-skill==3.0.1", "lerobot==0.6.0"]
""".strip()
        + "\n",
    )
    contents = {
        "src/langmani/environments/specs.py": (
            "class TaskSpec: pass\ncanonical_v0='canonical_v0'\n"
            "PREFIX='langmani-pick-place-task-v0'\n"
        ),
        "src/langmani/environments/task_logic.py": (
            'def evaluate_task_state(): return {"success": True, '
            '"target_off_table": False}\n'
        ),
        "src/langmani/datasets/policy_state.py": (
            "PANDA_POLICY_STATE_COMPONENTS=()\n_POLICY_STATE_SIZE=9\n"
        ),
        "src/langmani/datasets/observation_reconstruction.py": (
            "base_camera='base_camera'\npd_joint_pos='pd_joint_pos'\n"
            "set_state_dict='set_state_dict'\n"
        ),
        "src/langmani/policies/act_types.py": (
            "ACT_ACTION_COMPONENTS = 8\n"
            "chunk_size: int = 50\n"
            "n_action_steps: int = 10\n"
        ),
        "src/langmani/policies/act_action_bounds.py": (
            "class BoundedActionEnvPostprocessorV0: pass\n"
            "retain_raw_and_executed_actions=True\n"
            "x=torch.minimum(torch.maximum(action, low), high)\n"
        ),
        "src/langmani/policies/act_rollout.py": (
            "def reset_policy_state(): pass\npolicy.select_action(batch)\n"
            "action_bound_processor.process(action)\n"
        ),
        "src/langmani/policies/act_checkpoint.py": (
            "CHECKPOINT_SCHEMA_VERSION='v1'\nRNG_STATE_FILE='rng_state.pt'\n"
        ),
        "src/langmani/collection/replay.py": (
            "STATE_ROUND_TRIP_ABS_TOLERANCE = 1e-6\n"
            "set_state_dict(x)\n"
            "get_state_dict()\n"
        ),
        "src/langmani/language/dispatcher.py": (
            "class ControllerDispatcher:\n    def dispatch(self): pass\n"
        ),
    }
    for relative, content in contents.items():
        _write_fixture_file(root, relative, content)
    for relative in (
        "AGENTS.md",
        "README.md",
        "environment/environment.yml",
        "docs/ARCHITECTURE.md",
        "docs/ACT_BASELINE_SPEC.md",
        "docs/M42_ORACLE_CONTROL_SPEC.md",
        "docs/M5A_LANGUAGE_ROUTING_SPEC.md",
        "src/langmani/environments/pick_place_by_instruction.py",
        "src/langmani/datasets/archive.py",
        "src/langmani/language/router_types.py",
        "tests/unit/test_m3a_replay.py",
        "tests/integration/test_act_rollout.py",
        "tests/unit/test_act_action_bounds.py",
    ):
        _write_fixture_file(root, relative, "fixture\n")
    _git(root, "init")
    _git(root, "config", "user.name", "M6A Fixture")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(
        root,
        "remote",
        "add",
        "origin",
        "https://github.com/Yanagisawa2002/LangMani.git",
    )
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "fixture")
    return root


def test_static_audit_accepts_clean_fixture_and_creates_honest_blockers(
    tmp_path: Path,
) -> None:
    root = _langmani_fixture(tmp_path)
    report = audit_langmani_contract(root, latentguard_root=Path.cwd(), strict=True)
    assert report["passed"]
    assert report["overall_readiness"] == "blocked"
    blocker_ids = {item["blocker_id"] for item in report["blockers"]}
    assert {"M6A-B003", "M6A-B004", "M6A-B007", "M6A-B008"} <= blocker_ids
    assert report["observed_contract"]["policy_chunk_shape"] == [50, 8]
    assert report["observed_contract"]["replay_readiness"] == "initial_state_only"
    serialized = json.dumps(report, allow_nan=False)
    assert str(root.resolve()) not in serialized


def test_discovery_rejects_missing_git_and_ambiguous_checkouts(tmp_path: Path) -> None:
    copied = tmp_path / "copy"
    _write_fixture_file(copied, "pyproject.toml", "[project]\nname='langmani'\n")
    _write_fixture_file(
        copied, "src/langmani/environments/specs.py", "class TaskSpec: pass\n"
    )
    with pytest.raises(LangManiAuditError, match="no valid"):
        discover_langmani_checkout((copied,))
    left = _langmani_fixture(tmp_path / "left")
    right = _langmani_fixture(tmp_path / "right")
    with pytest.raises(LangManiAuditError, match="ambiguous"):
        discover_langmani_checkout((left, right))


def test_audit_rejects_dirty_checkout_and_action_shape_mismatch(tmp_path: Path) -> None:
    root = _langmani_fixture(tmp_path)
    path = root / "src/langmani/policies/act_types.py"
    path.write_text(
        path.read_text(encoding="utf-8").replace("8", "7", 1), encoding="utf-8"
    )
    report = audit_langmani_contract(root, latentguard_root=Path.cwd(), strict=True)
    assert not report["passed"]
    assert "dirty_checkout" in report["audit_issues"]
    assert "src/langmani/policies/act_types.py" in report["audit_issues"]


def test_unresolved_projection_and_missing_policy_state_are_explicit(
    tmp_path: Path,
) -> None:
    root = _langmani_fixture(tmp_path)
    projection = root / "src/langmani/policies/act_action_bounds.py"
    projection.write_text(
        "class BoundedActionEnvPostprocessorV0: pass\n", encoding="utf-8"
    )
    report = audit_langmani_contract(root, latentguard_root=Path.cwd())
    blocker_ids = {item["blocker_id"] for item in report["blockers"]}
    assert "M6A-B002" in blocker_ids
    assert "M6A-B004" in blocker_ids


def test_action_contract_rejects_shape_and_source_identity_errors() -> None:
    adapter = SyntheticLangManiAdapter()
    raw = adapter.propose()
    payload = raw.to_dict()
    payload["horizon"] = 3
    with pytest.raises(LangManiContractError, match="shape"):
        LangManiActionProposalV1.from_dict(payload)
    payload = raw.to_dict()
    payload["source_proposal_digest"] = "sha256:" + "f" * 64
    with pytest.raises(LangManiContractError, match="raw proposals"):
        LangManiActionProposalV1.from_dict(payload)


def test_observation_field_contract_round_trip() -> None:
    field = LangManiObservationFieldV1(
        name="tcp_pose",
        dtype="float32",
        shape=(7,),
        role=ObservationRole.PROHIBITED_LEARNED,
        units="meter_and_quaternion",
        coordinate_frame="world",
        normalization="none",
        update_cadence="20_hz",
        source_location="task_logic.py",
        determinism="state_derived",
        reconstructable_after_restore=True,
    )
    assert LangManiObservationFieldV1.from_dict(field.to_dict()) == field
    assert _policy().action_dimension == 8


def test_additive_docs_preserve_frozen_m5_manifest_bytes() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "docs/release/release-manifest.json"
    observed = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert (
        observed
        == "4023dad2253156b56fc094ddee2e1e99afce5ee8843a3caf27bbaaFC6ce07377".lower()
    )


def test_compatibility_and_blocker_registries_are_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    matrix = json.loads(
        (root / "docs/integrations/langmani/compatibility-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    blockers = json.loads(
        (root / "docs/integrations/langmani/blockers.json").read_text(encoding="utf-8")
    )
    assert len(matrix["rows"]) == 23
    assert matrix["compatibility_counts"] == {
        "compatible": 11,
        "adaptable": 7,
        "blocked": 5,
        "unknown": 0,
    }
    assert len(blockers["blockers"]) == 6
    assert blockers["overall_readiness"] == "blocked"
    assert all(not item["m6b_can_proceed"] for item in blockers["blockers"])

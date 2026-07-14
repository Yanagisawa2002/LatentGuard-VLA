"""Round-trip and crash-safety tests for M2A evaluation manifests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest

import latentguard.evaluation.serialization as serialization_module
from latentguard.corruptions.layout import (
    ActionField,
    ActionLayout,
    ActionSemantic,
)
from latentguard.corruptions.models import (
    CorruptedActionProposal,
    compute_proposal_identifier,
)
from latentguard.corruptions.serialization import (
    CorruptionDataset,
    save_corruption_dataset,
)
from latentguard.evaluation.fixture import create_deterministic_fixture_evaluator
from latentguard.evaluation.reporting import build_evaluation_summary
from latentguard.evaluation.serialization import (
    MANIFEST_NAME,
    EvaluationDataset,
    EvaluationSerializationError,
    LedgerEntry,
    LedgerState,
    RunEnvironment,
    RunManifest,
    RunState,
    UnsupportedEvaluationSerializationVersionError,
    compute_corruption_dataset_content_digest,
    compute_corruption_dataset_digest,
    compute_evaluation_seed,
    compute_run_identifier,
    load_evaluation_dataset,
    save_evaluation_dataset,
    update_evaluation_dataset,
)
from latentguard.models import ActionChunk

_STARTED = "2026-07-14T12:00:00+00:00"
_FINISHED = "2026-07-14T12:00:01+00:00"


def _fixture_configuration() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "success_mean_abs_threshold": 10.0,
        "unsafe_max_abs_threshold": 100.0,
        "skip_mean_abs_below": None,
        "indeterminate_temporal_variation_below": None,
        "execution_error_max_abs_above": None,
    }


def _corruption_dataset(
    *, proposal_count: int = 2, identity_offset: int = 0
) -> CorruptionDataset:
    layout = ActionLayout(
        action_dim=2,
        fields=(
            ActionField(
                name="control",
                indices=(0, 1),
                semantic=ActionSemantic.AUXILIARY,
            ),
        ),
    )
    proposals: list[CorruptedActionProposal] = []
    for ordinal in range(proposal_count):
        identity = identity_offset + ordinal
        parameters = {"target_indices": (0, 1), "bias": (0.1, -0.1)}
        proposal_id = compute_proposal_identifier(
            source_episode_id=f"episode-{identity}",
            source_candidate_id=f"candidate-{identity}",
            corruption_name="constant_bias",
            resolved_parameters=parameters,
            seed=100 + identity,
            generation_ordinal=ordinal,
        )
        proposals.append(
            CorruptedActionProposal(
                proposal_id=proposal_id,
                source_episode_id=f"episode-{identity}",
                source_candidate_id=f"candidate-{identity}",
                source_policy_id="policy-fixture",
                source_task_id="task-fixture",
                split_group_id=f"episode-{identity}",
                transformed_action=ActionChunk(
                    actions=np.array(
                        [[0.1 + identity, 0.2], [0.3, 0.4 + identity]],
                        dtype=np.float32,
                    ),
                    coordinate_frame="fixture-frame",
                    control_period_s=0.1,
                ),
                corruption_type="constant_bias",
                resolved_parameters=parameters,
                seed=100 + identity,
                generation_ordinal=ordinal,
                notes="unlabeled proposal",
            )
        )
    return CorruptionDataset(
        source_dataset_id="sha256:" + "a" * 64,
        action_layout=layout,
        proposals=tuple(proposals),
    )


def _save_source(root: Path, dataset: CorruptionDataset | None = None) -> Path:
    source = root / "corruptions"
    save_corruption_dataset(dataset or _corruption_dataset(), source)
    return source


def _environment() -> RunEnvironment:
    return RunEnvironment(
        git_commit_sha=None,
        git_branch=None,
        python_version="3.11.14",
        numpy_version="1.26.4",
        platform="test-platform",
        launch_command="latentguard evaluate-data [sanitized]",
    )


def _dataset(
    source_dir: Path,
    corruption_dataset: CorruptionDataset,
    *,
    run_state: RunState = RunState.COMPLETE,
    max_proposals: int | None = None,
) -> EvaluationDataset:
    evaluator = create_deterministic_fixture_evaluator(_fixture_configuration())
    selected_count = (
        len(corruption_dataset.proposals)
        if max_proposals is None
        else min(max_proposals, len(corruption_dataset.proposals))
    )
    proposals = corruption_dataset.proposals[:selected_count]
    selected_ids = tuple(proposal.proposal_id for proposal in proposals)
    source_digest = compute_corruption_dataset_digest(source_dir)
    run_id = compute_run_identifier(
        source_corruption_dataset_digest=source_digest,
        source_dataset_id=corruption_dataset.source_dataset_id,
        evaluator_id=evaluator.evaluator_id,
        evaluator_version=evaluator.evaluator_version,
        evaluator_configuration_digest=evaluator.configuration_digest,
        base_seed=271828,
        max_proposals=max_proposals,
        selected_proposal_ids=selected_ids,
    )

    evidence = []
    ledger = []
    for index, proposal in enumerate(proposals):
        seed = compute_evaluation_seed(
            base_seed=271828,
            proposal_id=proposal.proposal_id,
            evaluator_id=evaluator.evaluator_id,
            evaluator_version=evaluator.evaluator_version,
            evaluator_configuration_digest=evaluator.configuration_digest,
            attempt_ordinal=0,
        )
        if run_state is RunState.COMPLETE:
            result = evaluator.evaluate(
                proposal,
                source_dataset_id=corruption_dataset.source_dataset_id,
                evaluation_seed=seed,
                attempt_ordinal=0,
            )
            evidence.append(result)
            ledger.append(
                LedgerEntry(
                    proposal_id=proposal.proposal_id,
                    evidence_id=result.evidence_id,
                    attempt_ordinal=0,
                    evaluator_id=evaluator.evaluator_id,
                    evaluator_version=evaluator.evaluator_version,
                    evaluator_configuration_digest=evaluator.configuration_digest,
                    evaluation_seed=seed,
                    state=LedgerState.COMPLETED,
                    error_type=None,
                    error_message=None,
                    retry_eligible=False,
                    started_at=_STARTED,
                    finished_at=_FINISHED,
                )
            )
        else:
            ledger.append(
                LedgerEntry(
                    proposal_id=proposal.proposal_id,
                    evidence_id=None,
                    attempt_ordinal=0,
                    evaluator_id=evaluator.evaluator_id,
                    evaluator_version=evaluator.evaluator_version,
                    evaluator_configuration_digest=evaluator.configuration_digest,
                    evaluation_seed=seed,
                    state=(LedgerState.RUNNING if index == 0 else LedgerState.PENDING),
                    error_type=None,
                    error_message=None,
                    retry_eligible=False,
                    started_at=(_STARTED if index == 0 else None),
                    finished_at=None,
                )
            )

    summary = build_evaluation_summary(evidence, ledger)
    finished_at = _FINISHED if run_state is not RunState.IN_PROGRESS else None
    manifest = RunManifest(
        environment=_environment(),
        evaluator_id=evaluator.evaluator_id,
        evaluator_version=evaluator.evaluator_version,
        evaluator_configuration_digest=evaluator.configuration_digest,
        source_corruption_dataset_digest=source_digest,
        source_dataset_id=corruption_dataset.source_dataset_id,
        seed=271828,
        started_at=_STARTED,
        finished_at=finished_at,
        final_state=run_state,
    )
    return EvaluationDataset(
        source_corruption_dataset_digest=source_digest,
        source_dataset_id=corruption_dataset.source_dataset_id,
        evaluator_id=evaluator.evaluator_id,
        evaluator_version=evaluator.evaluator_version,
        resolved_evaluator_configuration=evaluator.resolved_configuration(),
        evaluator_configuration_digest=evaluator.configuration_digest,
        run_id=run_id,
        base_seed=271828,
        max_proposals=max_proposals,
        selected_proposal_ids=selected_ids,
        evidence=tuple(evidence),
        ledger=tuple(ledger),
        summary=summary,
        artifact_references=(),
        run_state=run_state,
        run_manifest=manifest,
    )


def _manifest(output: Path) -> dict[str, Any]:
    return json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))


def _write_manifest(output: Path, manifest: dict[str, Any]) -> None:
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_complete_round_trip_preserves_evidence_ledger_and_manifest(
    tmp_path: Path,
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    dataset = _dataset(source, corruption_dataset)
    output = tmp_path / "evaluated"

    manifest_path = save_evaluation_dataset(dataset, output)
    loaded = load_evaluation_dataset(
        output,
        corruption_dataset=corruption_dataset,
        expected_corruption_digest=compute_corruption_dataset_digest(source),
    )

    assert manifest_path == output / MANIFEST_NAME
    assert loaded.run_id == dataset.run_id
    assert loaded.run_state is RunState.COMPLETE
    assert loaded.selected_proposal_ids == dataset.selected_proposal_ids
    assert [item.evidence_id for item in loaded.evidence] == [
        item.evidence_id for item in dataset.evidence
    ]
    assert loaded.ledger == dataset.ledger
    assert loaded.summary == dataset.summary
    assert loaded.run_manifest == dataset.run_manifest


@pytest.mark.parametrize("run_state", [RunState.IN_PROGRESS, RunState.INTERRUPTED])
def test_incomplete_run_round_trip_remains_distinguishable(
    tmp_path: Path, run_state: RunState
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    dataset = _dataset(source, corruption_dataset, run_state=run_state)
    output = tmp_path / run_state.value

    save_evaluation_dataset(dataset, output)
    loaded = load_evaluation_dataset(output, corruption_dataset=corruption_dataset)

    assert loaded.run_state is run_state
    assert loaded.run_manifest.final_state is run_state
    assert loaded.ledger[0].state is LedgerState.RUNNING
    assert loaded.ledger[1].state is LedgerState.PENDING
    assert loaded.evidence == ()


def test_manifest_is_human_readable_and_does_not_duplicate_action_arrays(
    tmp_path: Path,
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / "evaluated"
    save_evaluation_dataset(_dataset(source, corruption_dataset), output)

    text = (output / MANIFEST_NAME).read_text(encoding="utf-8")

    assert text.startswith("{\n")
    assert '"evidence"' in text
    assert '"selected_proposal_ids"' in text
    assert '"transformed_action"' not in text
    assert '"actions"' not in text
    assert '"rgb"' not in text
    assert tuple(path.name for path in output.iterdir()) == (MANIFEST_NAME,)


def test_dataset_rejects_nondeterministic_evidence_and_ledger_order(
    tmp_path: Path,
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    dataset = _dataset(source, corruption_dataset)

    with pytest.raises(EvaluationSerializationError, match="deterministic order"):
        replace(dataset, evidence=tuple(reversed(dataset.evidence)))
    with pytest.raises(EvaluationSerializationError, match="deterministic order"):
        replace(dataset, ledger=tuple(reversed(dataset.ledger)))


def test_dataset_requires_attempt_zero_for_every_selected_proposal(
    tmp_path: Path,
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    dataset = _dataset(source, corruption_dataset)
    evidence = dataset.evidence[1:]
    ledger = dataset.ledger[1:]

    with pytest.raises(EvaluationSerializationError, match="attempt-zero"):
        replace(
            dataset,
            evidence=evidence,
            ledger=ledger,
            summary=build_evaluation_summary(evidence, ledger),
        )


def test_loader_rejects_duplicate_evidence_and_ledger_attempts(tmp_path: Path) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / "duplicates"
    save_evaluation_dataset(_dataset(source, corruption_dataset), output)
    manifest = _manifest(output)
    manifest["evidence"].append(copy := dict(manifest["evidence"][0]))
    assert copy
    _write_manifest(output, manifest)

    with pytest.raises(EvaluationSerializationError, match="duplicate"):
        load_evaluation_dataset(output)

    save_evaluation_dataset(
        _dataset(source, corruption_dataset), tmp_path / "ledger-duplicates"
    )
    ledger_output = tmp_path / "ledger-duplicates"
    ledger_manifest = _manifest(ledger_output)
    ledger_manifest["ledger"].append(dict(ledger_manifest["ledger"][0]))
    _write_manifest(ledger_output, ledger_manifest)
    with pytest.raises(EvaluationSerializationError, match="duplicate"):
        load_evaluation_dataset(ledger_output)


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("top", "serialization_version"),
        ("top", "dataset_schema_version"),
        ("evidence", "schema_version"),
        ("ledger", "schema_version"),
        ("run_manifest", "schema_version"),
    ],
)
def test_unsupported_versions_fail_clearly(
    tmp_path: Path, location: str, field: str
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / f"{location}-{field}"
    save_evaluation_dataset(_dataset(source, corruption_dataset), output)
    manifest = _manifest(output)
    if location == "top":
        manifest[field] = 999 if field == "serialization_version" else "99.0"
    elif location == "run_manifest":
        manifest["run_manifest"][field] = "99.0"
    else:
        manifest[location][0][field] = "99.0"
    _write_manifest(output, manifest)

    with pytest.raises(UnsupportedEvaluationSerializationVersionError):
        load_evaluation_dataset(output)


@pytest.mark.parametrize("mutation", ["unknown", "missing"])
def test_manifest_requires_exact_top_level_fields(
    tmp_path: Path, mutation: str
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / mutation
    save_evaluation_dataset(_dataset(source, corruption_dataset), output)
    manifest = _manifest(output)
    if mutation == "unknown":
        manifest["unexpected"] = True
    else:
        manifest.pop("run_id")
    _write_manifest(output, manifest)

    with pytest.raises(EvaluationSerializationError, match="unexpected|missing"):
        load_evaluation_dataset(output)


def test_duplicate_json_fields_are_rejected(tmp_path: Path) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / "duplicate-json"
    save_evaluation_dataset(_dataset(source, corruption_dataset), output)
    path = output / MANIFEST_NAME
    original = path.read_text(encoding="utf-8").lstrip()
    path.write_text('{"format":"duplicate",' + original[1:], encoding="utf-8")

    with pytest.raises(EvaluationSerializationError, match="duplicate"):
        load_evaluation_dataset(output)


@pytest.mark.parametrize(
    "reference",
    [
        "../private.json",
        "artifacts/credentials.json",
        "artifacts/private_key.pem",
        "artifacts/secrets.json",
        "artifacts/api_keys.json",
        "artifacts/client_secrets.json",
    ],
)
def test_unsafe_artifact_reference_is_rejected_on_load(
    tmp_path: Path, reference: str
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / "traversal"
    save_evaluation_dataset(_dataset(source, corruption_dataset), output)
    manifest = _manifest(output)
    manifest["evidence"][0]["artifact_references"] = [reference]
    manifest["artifact_references"] = [reference]
    _write_manifest(output, manifest)

    with pytest.raises(EvaluationSerializationError, match="artifact_references"):
        load_evaluation_dataset(output)


def test_source_proposal_reference_and_provenance_are_revalidated(
    tmp_path: Path,
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / "source-reference"
    save_evaluation_dataset(_dataset(source, corruption_dataset), output)

    different = _corruption_dataset(identity_offset=100)
    with pytest.raises(EvaluationSerializationError, match="contents do not match"):
        load_evaluation_dataset(output, corruption_dataset=different)

    manifest = _manifest(output)
    manifest["evidence"][0]["source_episode_id"] = "another-episode"
    _write_manifest(output, manifest)
    with pytest.raises(EvaluationSerializationError, match="source_episode_id"):
        load_evaluation_dataset(output, corruption_dataset=corruption_dataset)


def test_expected_corruption_digest_conflict_is_rejected(tmp_path: Path) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / "digest-conflict"
    save_evaluation_dataset(_dataset(source, corruption_dataset), output)

    with pytest.raises(EvaluationSerializationError, match="source bundle"):
        load_evaluation_dataset(
            output,
            expected_corruption_digest="sha256:" + "0" * 64,
        )


def test_nonempty_destination_is_refused_without_modification(tmp_path: Path) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / "nonempty"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")

    with pytest.raises(EvaluationSerializationError, match="absent or empty"):
        save_evaluation_dataset(_dataset(source, corruption_dataset), output)

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert tuple(output.iterdir()) == (sentinel,)


def test_initial_transaction_failure_leaves_no_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / "failed"

    def fail_write(*_args: object, **_kwargs: object) -> NoReturn:
        raise OSError("injected initial write failure")

    monkeypatch.setattr(serialization_module, "_write_new_manifest", fail_write)

    with pytest.raises(EvaluationSerializationError, match="transactionally"):
        save_evaluation_dataset(_dataset(source, corruption_dataset), output)

    assert not output.exists()
    assert list(tmp_path.glob(".failed.staging-*")) == []


def test_atomic_update_failure_preserves_last_valid_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / "atomic-update"
    interrupted = _dataset(
        source,
        corruption_dataset,
        run_state=RunState.INTERRUPTED,
        max_proposals=1,
    )
    completed = _dataset(source, corruption_dataset, max_proposals=1)
    save_evaluation_dataset(interrupted, output)
    before = (output / MANIFEST_NAME).read_bytes()

    def fail_replace(*_args: object, **_kwargs: object) -> NoReturn:
        raise OSError("injected atomic replace failure")

    monkeypatch.setattr(serialization_module.os, "replace", fail_replace)

    with pytest.raises(EvaluationSerializationError, match="atomic manifest update"):
        update_evaluation_dataset(completed, output)

    assert (output / MANIFEST_NAME).read_bytes() == before
    assert load_evaluation_dataset(output).run_state is RunState.INTERRUPTED
    assert list(tmp_path.glob(".atomic-update.manifest-update-*.json")) == []


def test_update_refuses_to_overwrite_a_different_run(tmp_path: Path) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / "different-run"
    initial = _dataset(source, corruption_dataset, run_state=RunState.INTERRUPTED)
    different = _dataset(source, corruption_dataset, max_proposals=1)
    save_evaluation_dataset(initial, output)

    with pytest.raises(EvaluationSerializationError, match="different run"):
        update_evaluation_dataset(different, output)


@pytest.mark.parametrize("invalid_count", [True, 1.0])
def test_dataset_rejects_non_integer_summary_counts(
    tmp_path: Path, invalid_count: object
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    dataset = _dataset(source, corruption_dataset)
    invalid_summary = replace(dataset.summary, conclusive=invalid_count)

    with pytest.raises(EvaluationSerializationError, match="non-negative integer"):
        replace(dataset, summary=invalid_summary)


def test_dataset_rejects_non_integer_manifest_seed_and_string_states(
    tmp_path: Path,
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    dataset = _dataset(source, corruption_dataset)

    with pytest.raises(EvaluationSerializationError, match="run_manifest.seed"):
        replace(
            dataset,
            run_manifest=replace(dataset.run_manifest, seed=float(dataset.base_seed)),
        )
    with pytest.raises(EvaluationSerializationError, match="run_state"):
        replace(dataset, run_state="complete")
    with pytest.raises(EvaluationSerializationError, match="final_state"):
        replace(
            dataset,
            run_manifest=replace(dataset.run_manifest, final_state="complete"),
        )


def test_dataset_rejects_backwards_timestamps(tmp_path: Path) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    dataset = _dataset(source, corruption_dataset)
    backwards = "2026-07-14T11:59:59+00:00"

    with pytest.raises(EvaluationSerializationError, match="cannot precede started_at"):
        replace(
            dataset,
            run_manifest=replace(dataset.run_manifest, finished_at=backwards),
        )
    bad_entry = replace(dataset.ledger[0], finished_at=backwards)
    with pytest.raises(EvaluationSerializationError, match="cannot precede started_at"):
        replace(dataset, ledger=(bad_entry, *dataset.ledger[1:]))
    with pytest.raises(EvaluationSerializationError, match="include a timezone"):
        replace(
            dataset,
            run_manifest=replace(
                dataset.run_manifest, started_at="2026-07-14T12:00:00"
            ),
        )


def test_in_memory_source_action_contents_are_digest_bound(tmp_path: Path) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / "content-bound"
    save_evaluation_dataset(_dataset(source, corruption_dataset), output)
    proposal = corruption_dataset.proposals[0]
    altered_proposal = replace(
        proposal,
        transformed_action=ActionChunk(
            actions=proposal.transformed_action.actions + np.float32(0.25),
            coordinate_frame=proposal.transformed_action.coordinate_frame,
            control_period_s=proposal.transformed_action.control_period_s,
        ),
    )
    altered_dataset = replace(
        corruption_dataset,
        proposals=(altered_proposal, *corruption_dataset.proposals[1:]),
    )

    assert compute_corruption_dataset_content_digest(
        altered_dataset
    ) != compute_corruption_dataset_content_digest(corruption_dataset)
    with pytest.raises(EvaluationSerializationError, match="contents do not match"):
        load_evaluation_dataset(output, corruption_dataset=altered_dataset)


def test_update_rejects_conflicting_prior_evidence(tmp_path: Path) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / "evidence-conflict"
    dataset = _dataset(source, corruption_dataset)
    save_evaluation_dataset(dataset, output)
    before = (output / MANIFEST_NAME).read_bytes()
    conflicting = replace(
        dataset,
        evidence=(
            replace(dataset.evidence[0], notes="conflicting prior payload"),
            *dataset.evidence[1:],
        ),
    )

    with pytest.raises(
        EvaluationSerializationError, match="conflicting prior evidence"
    ):
        update_evaluation_dataset(conflicting, output)
    assert (output / MANIFEST_NAME).read_bytes() == before
    assert (
        load_evaluation_dataset(output).evidence[0].notes == dataset.evidence[0].notes
    )


def test_update_rejects_ledger_state_rollback(tmp_path: Path) -> None:
    corruption_dataset = _corruption_dataset(proposal_count=1)
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / "ledger-rollback"
    interrupted = _dataset(
        source,
        corruption_dataset,
        run_state=RunState.INTERRUPTED,
    )
    save_evaluation_dataset(interrupted, output)
    before = (output / MANIFEST_NAME).read_bytes()
    pending = replace(
        interrupted.ledger[0],
        state=LedgerState.PENDING,
        started_at=None,
    )
    rolled_back = replace(
        interrupted,
        ledger=(pending,),
        summary=build_evaluation_summary((), (pending,)),
    )

    with pytest.raises(EvaluationSerializationError, match="invalid state transition"):
        update_evaluation_dataset(rolled_back, output)
    assert (output / MANIFEST_NAME).read_bytes() == before
    assert load_evaluation_dataset(output).ledger[0].state is LedgerState.RUNNING


def _execution_error_dataset(
    source: Path, corruption_dataset: CorruptionDataset
) -> EvaluationDataset:
    dataset = _dataset(source, corruption_dataset, max_proposals=1)
    evidence = replace(
        dataset.evidence[0],
        status=serialization_module.EvaluationStatus.EXECUTION_ERROR,
        success=None,
        progress_before=None,
        progress_after=None,
        progress_delta=None,
        unsafe=None,
        failure_events=(),
        termination_reason="execution_error:RuntimeError",
        replayed_control_steps=0,
        metrics={"diagnostic": "controlled failure"},
        label_source=None,
        label_strength=None,
        simulator_replay_verified=False,
        notes="controlled failure",
    )
    ledger = replace(
        dataset.ledger[0],
        state=LedgerState.EXECUTION_ERROR,
        error_type="RuntimeError",
        error_message="controlled failure",
        retry_eligible=True,
    )
    return replace(
        dataset,
        evidence=(evidence,),
        ledger=(ledger,),
        summary=build_evaluation_summary((evidence,), (ledger,)),
    )


def test_execution_error_diagnostics_must_be_sanitized(tmp_path: Path) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    dataset = _execution_error_dataset(source, corruption_dataset)

    for raw in (
        "password raw-secret",
        "API_TOKEN raw-secret",
        "AWS_SECRET_ACCESS_KEY=abc123",
        "PASSWORD_FILE=abc123",
        "Authorization: Bearer abc123",
        "environment={'FOO': 'bar'}",
        "path:/home/private/model.ckpt",
        "file:/home/private/model.ckpt",
        "ssh\t-p 22 root@example.invalid",
        "connection failed for root@example.invalid:47228,",
        "Traceback (most recent call last): relative_module.py line 10",
        "ssh -p 22 root@example.invalid",
        "/foo.txt",
        r"C:\Users\private folder\trace.txt",
    ):
        unsafe_ledger = replace(dataset.ledger[0], error_message=raw)
        with pytest.raises(
            EvaluationSerializationError, match="sanitized text|control characters"
        ):
            replace(
                dataset,
                ledger=(unsafe_ledger,),
                summary=build_evaluation_summary(dataset.evidence, (unsafe_ledger,)),
            )

    unsafe_notes = replace(dataset.evidence[0], notes="ssh -p 22 root@example.invalid")
    with pytest.raises(EvaluationSerializationError, match="sanitized text"):
        replace(
            dataset,
            evidence=(unsafe_notes,),
            summary=build_evaluation_summary((unsafe_notes,), dataset.ledger),
        )
    unsafe_metrics = replace(dataset.evidence[0], metrics={"API_TOKEN": "abc123"})
    with pytest.raises(EvaluationSerializationError, match="diagnostic key"):
        replace(
            dataset,
            evidence=(unsafe_metrics,),
            summary=build_evaluation_summary((unsafe_metrics,), dataset.ledger),
        )


@pytest.mark.parametrize(
    "field_value",
    [
        "ssh -p 22 root@example.invalid",
        "password raw-secret",
        "/foo.txt",
        "AWS_SECRET_ACCESS_KEY=abc123",
        "environment={'FOO': 'bar'}",
        "connection failed for root@example.invalid:47228,",
    ],
)
def test_run_environment_requires_sanitized_operational_metadata(
    tmp_path: Path, field_value: str
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    dataset = _dataset(source, corruption_dataset)
    environment = replace(
        dataset.run_manifest.environment,
        launch_command=field_value,
    )

    with pytest.raises(EvaluationSerializationError, match="expected sanitized text"):
        replace(
            dataset,
            run_manifest=replace(dataset.run_manifest, environment=environment),
        )


def test_path_and_content_corruption_digests_are_equal_and_path_independent(
    tmp_path: Path,
) -> None:
    corruption_dataset = _corruption_dataset()
    first = tmp_path / "first"
    second = tmp_path / "second"
    save_corruption_dataset(corruption_dataset, first)
    save_corruption_dataset(corruption_dataset, second)

    content_digest = compute_corruption_dataset_content_digest(corruption_dataset)
    assert compute_corruption_dataset_digest(first) == content_digest
    assert compute_corruption_dataset_digest(second) == content_digest


def test_update_rejects_changed_environment_and_run_start(tmp_path: Path) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    dataset = _dataset(source, corruption_dataset)
    manifests = (
        replace(
            dataset.run_manifest,
            environment=replace(dataset.run_manifest.environment, git_branch="other"),
        ),
        replace(
            dataset.run_manifest,
            started_at="2026-07-14T11:59:59+00:00",
        ),
    )
    for index, manifest in enumerate(manifests):
        output = tmp_path / f"immutable-manifest-{index}"
        save_evaluation_dataset(dataset, output)
        before = (output / MANIFEST_NAME).read_bytes()
        changed = replace(dataset, run_manifest=manifest)

        with pytest.raises(EvaluationSerializationError, match="cannot change"):
            update_evaluation_dataset(changed, output)
        assert (output / MANIFEST_NAME).read_bytes() == before


def test_complete_run_reopens_only_for_an_explicit_retry(tmp_path: Path) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    dataset = _dataset(source, corruption_dataset)
    output = tmp_path / "reopen"
    save_evaluation_dataset(dataset, output)
    reopened = replace(
        dataset,
        run_state=RunState.IN_PROGRESS,
        run_manifest=replace(
            dataset.run_manifest,
            final_state=RunState.IN_PROGRESS,
            finished_at=None,
        ),
    )

    with pytest.raises(EvaluationSerializationError, match="explicit retry"):
        update_evaluation_dataset(reopened, output)


def test_update_rejects_terminal_ledger_mutation(tmp_path: Path) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    dataset = _dataset(source, corruption_dataset)
    output = tmp_path / "terminal-mutation"
    save_evaluation_dataset(dataset, output)
    changed_entry = replace(
        dataset.ledger[0], started_at="2026-07-14T12:00:00.500000+00:00"
    )
    changed = replace(dataset, ledger=(changed_entry, *dataset.ledger[1:]))

    with pytest.raises(EvaluationSerializationError, match="cannot be mutated"):
        update_evaluation_dataset(changed, output)


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("top", "base_seed"),
        ("evidence", "evaluation_seed"),
        ("ledger", "attempt_ordinal"),
        ("summary", "conclusive"),
        ("run_manifest", "seed"),
    ],
)
def test_loader_rejects_boolean_integer_fields(
    tmp_path: Path, location: str, field: str
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    output = tmp_path / f"bool-{location}-{field}"
    save_evaluation_dataset(_dataset(source, corruption_dataset), output)
    manifest = _manifest(output)
    if location == "top":
        manifest[field] = True
    elif location in {"evidence", "ledger"}:
        manifest[location][0][field] = True
    else:
        manifest[location][field] = True
    _write_manifest(output, manifest)

    with pytest.raises(EvaluationSerializationError, match="expected an integer"):
        load_evaluation_dataset(output)


def test_attempt_timestamps_must_stay_within_run_interval(tmp_path: Path) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    dataset = _dataset(source, corruption_dataset)
    before_start = replace(dataset.ledger[0], started_at="2026-07-14T11:59:59+00:00")
    with pytest.raises(EvaluationSerializationError, match="precedes run start"):
        replace(dataset, ledger=(before_start, *dataset.ledger[1:]))

    after_finish = replace(
        dataset.ledger[0],
        started_at="2026-07-14T12:00:02+00:00",
        finished_at="2026-07-14T12:00:03+00:00",
    )
    with pytest.raises(EvaluationSerializationError, match="exceeds run finish"):
        replace(dataset, ledger=(after_finish, *dataset.ledger[1:]))


def test_retry_attempt_cannot_start_before_prior_attempt_finishes(
    tmp_path: Path,
) -> None:
    corruption_dataset = _corruption_dataset()
    source = _save_source(tmp_path, corruption_dataset)
    dataset = _execution_error_dataset(source, corruption_dataset)
    retry_seed = compute_evaluation_seed(
        base_seed=dataset.base_seed,
        proposal_id=dataset.selected_proposal_ids[0],
        evaluator_id=dataset.evaluator_id,
        evaluator_version=dataset.evaluator_version,
        evaluator_configuration_digest=dataset.evaluator_configuration_digest,
        attempt_ordinal=1,
    )
    retry = LedgerEntry(
        proposal_id=dataset.selected_proposal_ids[0],
        evidence_id=None,
        attempt_ordinal=1,
        evaluator_id=dataset.evaluator_id,
        evaluator_version=dataset.evaluator_version,
        evaluator_configuration_digest=dataset.evaluator_configuration_digest,
        evaluation_seed=retry_seed,
        state=LedgerState.RUNNING,
        error_type=None,
        error_message=None,
        retry_eligible=False,
        started_at=_STARTED,
        finished_at=None,
    )
    manifest = replace(
        dataset.run_manifest,
        final_state=RunState.IN_PROGRESS,
        finished_at=None,
    )

    with pytest.raises(EvaluationSerializationError, match="prior attempt finishes"):
        replace(
            dataset,
            ledger=(*dataset.ledger, retry),
            summary=build_evaluation_summary(
                dataset.evidence, (*dataset.ledger, retry)
            ),
            run_state=RunState.IN_PROGRESS,
            run_manifest=manifest,
        )

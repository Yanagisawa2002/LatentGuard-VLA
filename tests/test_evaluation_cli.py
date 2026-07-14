"""CPU-only CLI integration tests for M2A evaluation and resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

from latentguard import cli
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
from latentguard.evaluation.fixture import DeterministicFixtureEvaluator
from latentguard.evaluation.serialization import (
    LedgerState,
    RunState,
    load_evaluation_dataset,
)
from latentguard.models import ActionChunk

_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "evaluation" / "m2a-fixture.json"
)


def _corruption_dataset(*, proposal_count: int = 2) -> CorruptionDataset:
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
    proposals = []
    for ordinal in range(proposal_count):
        parameters = {"target_indices": (0, 1), "bias": (0.1, -0.1)}
        proposal_id = compute_proposal_identifier(
            source_episode_id=f"episode-{ordinal}",
            source_candidate_id=f"candidate-{ordinal}",
            corruption_name="constant_bias",
            resolved_parameters=parameters,
            seed=700 + ordinal,
            generation_ordinal=ordinal,
        )
        proposals.append(
            CorruptedActionProposal(
                proposal_id=proposal_id,
                source_episode_id=f"episode-{ordinal}",
                source_candidate_id=f"candidate-{ordinal}",
                source_policy_id="policy-fixture",
                source_task_id="task-fixture",
                split_group_id=f"episode-{ordinal}",
                transformed_action=ActionChunk(
                    actions=np.array(
                        [[0.1 + ordinal, 0.2], [0.3, 0.4 + ordinal]],
                        dtype=np.float32,
                    ),
                    coordinate_frame="fixture-frame",
                    control_period_s=0.1,
                ),
                corruption_type="constant_bias",
                resolved_parameters=parameters,
                seed=700 + ordinal,
                generation_ordinal=ordinal,
            )
        )
    return CorruptionDataset(
        source_dataset_id="sha256:" + "e" * 64,
        action_layout=layout,
        proposals=tuple(proposals),
    )


def _source_bundle(root: Path, *, proposal_count: int = 2) -> Path:
    source = root / "corrupted"
    save_corruption_dataset(_corruption_dataset(proposal_count=proposal_count), source)
    return source


def _configuration(**updates: object) -> dict[str, object]:
    configuration = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    configuration.update(updates)
    return configuration


def _write_configuration(path: Path, configuration: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(configuration, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _arguments(
    source: Path,
    output: Path,
    *,
    config: Path = _CONFIG_PATH,
    extras: tuple[str, ...] = (),
) -> list[str]:
    return [
        "evaluate-data",
        "--corruption-dir",
        str(source),
        "--output-dir",
        str(output),
        "--evaluator",
        "deterministic_fixture",
        "--config",
        str(config),
        "--seed",
        "271828",
        *extras,
    ]


def test_dry_run_validates_and_plans_without_evaluation_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source_bundle(tmp_path)
    output = tmp_path / "dry-run-output"

    def fail_if_called(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("dry run must not call evaluator.evaluate")

    monkeypatch.setattr(DeterministicFixtureEvaluator, "evaluate", fail_if_called)

    return_code = cli.main(_arguments(source, output, extras=("--dry-run", "--audit")))

    captured = capsys.readouterr()
    assert return_code == 0
    assert "evaluate-data dry-run OK" in captured.out
    assert "proposals=2" in captured.out
    assert "planned_attempts=2" in captured.out
    assert "evaluations=0" in captured.out
    assert "output_created=false" in captured.out
    assert captured.out.count("evaluate-data dry-run audit:") == 2
    assert "synthetic weak non-simulator evidence only" in captured.out
    assert not output.exists()


def test_full_fixture_evaluation_round_trips_and_prints_all_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    output = tmp_path / "evaluated"

    return_code = cli.main(_arguments(source, output, extras=("--audit",)))

    captured = capsys.readouterr()
    assert return_code == 0
    assert "evaluate-data result" in captured.out
    assert "conclusive=2" in captured.out
    assert "indeterminate=0" in captured.out
    assert "invalid=0" in captured.out
    assert "skipped=0" in captured.out
    assert "execution_error=0" in captured.out
    assert "projected_outcome_labels=2" in captured.out
    assert "retried_attempts=0" in captured.out
    assert "fixture_evidence=synthetic-weak-non-simulator" in captured.out
    assert captured.out.count("evaluate-data audit:") == 2
    dataset = load_evaluation_dataset(output)
    assert dataset.run_state is RunState.COMPLETE
    assert dataset.summary.conclusive == 2
    assert len(dataset.evidence) == 2
    assert tuple(path.name for path in output.iterdir()) == ("manifest.json",)
    command = dataset.run_manifest.environment.launch_command
    assert str(source) not in command
    assert str(output) not in command
    assert str(_CONFIG_PATH) not in command
    assert command.count("<path>") == 3


def test_resume_is_noop_and_does_not_duplicate_completed_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    output = tmp_path / "resume"
    assert cli.main(_arguments(source, output)) == 0
    first = load_evaluation_dataset(output)
    before = (output / "manifest.json").read_bytes()
    capsys.readouterr()

    return_code = cli.main(_arguments(source, output, extras=("--resume",)))

    captured = capsys.readouterr()
    second = load_evaluation_dataset(output)
    assert return_code == 0
    assert "resumed=true" in captured.out
    assert "evaluated_attempts=0" in captured.out
    assert [item.evidence_id for item in second.evidence] == [
        item.evidence_id for item in first.evidence
    ]
    assert len(second.ledger) == len(first.ledger) == 2
    assert (output / "manifest.json").read_bytes() == before


def test_max_proposals_selects_stable_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path, proposal_count=3)
    output = tmp_path / "limited"

    return_code = cli.main(_arguments(source, output, extras=("--max-proposals", "1")))

    summary = capsys.readouterr().out
    dataset = load_evaluation_dataset(output)
    assert return_code == 0
    assert "conclusive=1" in summary
    assert dataset.max_proposals == 1
    assert len(dataset.selected_proposal_ids) == 1
    assert len(dataset.evidence) == 1


def test_fixture_policy_can_produce_skip_and_indeterminate_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    config = _write_configuration(
        tmp_path / "statuses.json",
        _configuration(
            skip_mean_abs_below=0.5,
            indeterminate_temporal_variation_below=10.0,
        ),
    )
    output = tmp_path / "statuses"

    return_code = cli.main(_arguments(source, output, config=config))

    summary = capsys.readouterr().out
    dataset = load_evaluation_dataset(output)
    assert return_code == 0
    assert "conclusive=0" in summary
    assert "indeterminate=1" in summary
    assert "skipped=1" in summary
    assert "execution_error=0" in summary
    assert "projected_outcome_labels=0" in summary
    assert dataset.summary.indeterminate == 1
    assert dataset.summary.skipped == 1


def test_controlled_exception_returns_nonzero_and_never_fabricates_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    config = _write_configuration(
        tmp_path / "error.json",
        _configuration(execution_error_max_abs_above=0.0),
    )
    output = tmp_path / "errors"

    return_code = cli.main(_arguments(source, output, config=config))

    captured = capsys.readouterr()
    dataset = load_evaluation_dataset(output)
    assert return_code != 0
    assert "evaluate-data result" in captured.out
    assert "evaluate-data OK" not in captured.out
    assert "execution_error=2" in captured.out
    assert dataset.run_state is RunState.COMPLETE
    assert dataset.summary.execution_error == 2
    assert all(item.success is None for item in dataset.evidence)
    assert all(item.unsafe is None for item in dataset.evidence)
    assert all(entry.state is LedgerState.EXECUTION_ERROR for entry in dataset.ledger)


def test_fail_fast_leaves_resumable_pending_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    config = _write_configuration(
        tmp_path / "fail-fast.json",
        _configuration(execution_error_max_abs_above=0.0),
    )
    output = tmp_path / "fail-fast"

    return_code = cli.main(
        _arguments(source, output, config=config, extras=("--fail-fast",))
    )

    summary = capsys.readouterr().out
    dataset = load_evaluation_dataset(output)
    assert return_code != 0
    assert "evaluate-data result" in summary
    assert "evaluate-data OK" not in summary
    assert dataset.run_state is RunState.INTERRUPTED
    assert [entry.state for entry in dataset.ledger] == [
        LedgerState.EXECUTION_ERROR,
        LedgerState.PENDING,
    ]


def test_explicit_cli_retry_preserves_prior_error_attempts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    config = _write_configuration(
        tmp_path / "retry.json",
        _configuration(execution_error_max_abs_above=0.0),
    )
    output = tmp_path / "retry"
    assert cli.main(_arguments(source, output, config=config)) != 0
    capsys.readouterr()

    return_code = cli.main(
        _arguments(
            source,
            output,
            config=config,
            extras=("--resume", "--retry-execution-errors"),
        )
    )

    summary = capsys.readouterr().out
    dataset = load_evaluation_dataset(output)
    assert return_code != 0
    assert "execution_error=4" in summary
    assert "retried_attempts=2" in summary
    assert len(dataset.ledger) == 4
    assert len(dataset.evidence) == 4
    assert [entry.attempt_ordinal for entry in dataset.ledger] == [0, 1, 0, 1]


@pytest.mark.parametrize("invalid_kind", ["unknown", "nonfinite"])
def test_invalid_configuration_returns_nonzero_without_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    invalid_kind: str,
) -> None:
    source = _source_bundle(tmp_path)
    config = tmp_path / f"{invalid_kind}.json"
    if invalid_kind == "unknown":
        config.write_text(json.dumps(_configuration(unexpected=True)), encoding="utf-8")
    else:
        config.write_text('{"threshold":NaN}', encoding="utf-8")
    output = tmp_path / f"{invalid_kind}-output"

    return_code = cli.main(_arguments(source, output, config=config))

    assert return_code != 0
    assert "evaluate-data failed" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("invalid_kind", ["missing", "empty"])
def test_invalid_corruption_dataset_returns_nonzero_without_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    invalid_kind: str,
) -> None:
    source = tmp_path / invalid_kind
    if invalid_kind == "empty":
        source.mkdir()
    output = tmp_path / f"{invalid_kind}-output"

    return_code = cli.main(_arguments(source, output))

    assert return_code != 0
    assert "evaluate-data failed" in capsys.readouterr().err
    assert not output.exists()


def test_unknown_evaluator_returns_nonzero_without_dynamic_import(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    output = tmp_path / "unknown-output"
    arguments = _arguments(source, output)
    evaluator_index = arguments.index("deterministic_fixture")
    arguments[evaluator_index] = "package.module:Dangerous"

    return_code = cli.main(arguments)

    assert return_code != 0
    assert "unknown evaluator" in capsys.readouterr().err
    assert not output.exists()


def test_retry_requires_resume_and_creates_no_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    output = tmp_path / "invalid-retry"

    return_code = cli.main(
        _arguments(source, output, extras=("--retry-execution-errors",))
    )

    assert return_code != 0
    assert "requires --resume" in capsys.readouterr().err
    assert not output.exists()


def test_dry_run_refuses_nonempty_output_without_modifying_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    output = tmp_path / "nonempty"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")

    return_code = cli.main(_arguments(source, output, extras=("--dry-run",)))

    assert return_code != 0
    assert "evaluate-data failed" in capsys.readouterr().err
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_evaluation_rejects_output_nested_under_source_without_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    before = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    output = source / "nested-output"

    return_code = cli.main(_arguments(source, output))

    assert return_code != 0
    assert "must not equal, contain" in capsys.readouterr().err
    assert not output.exists()
    after = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert after == before

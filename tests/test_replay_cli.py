"""CPU-only CLI integration tests for generic exact-state paired replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import pytest

from latentguard import cli
from latentguard.corruptions.config import load_corruption_plan
from latentguard.corruptions.generation import generate_corruption_proposals
from latentguard.corruptions.serialization import (
    CorruptionDataset,
    save_corruption_dataset,
)
from latentguard.evaluation.serialization import load_evaluation_dataset
from latentguard.models import LabelSource, LabelStrength
from latentguard.replay.fixture import DeterministicReplayFixtureAdapter
from latentguard.replay.source import ReplaySourceBinding
from latentguard.serialization import (
    compute_episode_bundle_identifier,
    save_episodes,
)
from latentguard.synthetic import generate_synthetic_episodes

_ROOT = Path(__file__).resolve().parents[1]
_CORRUPTION_CONFIG = _ROOT / "configs" / "corruptions" / "m1-smoke.json"
_REPLAY_CONFIG = _ROOT / "configs" / "replay" / "m2b-fixture.json"


def _source_and_corruption(root: Path) -> tuple[Path, Path]:
    source = root / "source"
    corruption = root / "corruption"
    episodes = generate_synthetic_episodes(
        seed=42,
        episode_count=1,
        episode_length=8,
        action_dim=7,
        robot_state_dim=10,
        camera_count=0,
        candidate_count=4,
    )
    save_episodes(episodes, source)
    plan = load_corruption_plan(_CORRUPTION_CONFIG)
    generated = generate_corruption_proposals(
        episodes,
        plan.action_layout,
        plan.corruptions,
        base_seed=314159,
        proposal_limit=12,
    )
    save_corruption_dataset(
        CorruptionDataset(
            source_dataset_id=compute_episode_bundle_identifier(source),
            action_layout=plan.action_layout,
            proposals=generated.proposals,
        ),
        corruption,
    )
    return source, corruption


def _arguments(
    source: Path,
    corruption: Path,
    output: Path,
    *,
    config: Path = _REPLAY_CONFIG,
    extras: tuple[str, ...] = (),
) -> list[str]:
    return [
        "replay-data",
        "--source-dir",
        str(source),
        "--corruption-dir",
        str(corruption),
        "--output-dir",
        str(output),
        "--adapter",
        "deterministic_replay_fixture",
        "--config",
        str(config),
        "--seed",
        "161803",
        *extras,
    ]


def _write_config(path: Path, **updates: object) -> Path:
    value = json.loads(_REPLAY_CONFIG.read_text(encoding="utf-8"))
    value.update(updates)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def test_dry_run_resolves_cases_without_sessions_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, corruption = _source_and_corruption(tmp_path)
    output = tmp_path / "dry-run"

    def fail_if_called(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("dry run must not create a replay session")

    monkeypatch.setattr(
        DeterministicReplayFixtureAdapter, "create_session", fail_if_called
    )
    return_code = cli.main(
        _arguments(
            source,
            corruption,
            output,
            extras=("--dry-run", "--audit", "--max-proposals", "4"),
        )
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert "replay-data dry-run OK" in captured.out
    assert "proposals=4" in captured.out
    assert "evaluations=0 sessions=0 output_created=false" in captured.out
    assert captured.out.count("replay-data dry-run audit:") == 4
    assert "not a simulator" in captured.out
    assert not output.exists()


def test_complete_fixture_run_reports_all_replay_counts_and_weak_trust(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, corruption = _source_and_corruption(tmp_path)
    output = tmp_path / "replayed"

    return_code = cli.main(_arguments(source, corruption, output))

    captured = capsys.readouterr()
    assert return_code == 0
    assert "source_episodes=1" in captured.out
    assert "source_candidates=4" in captured.out
    assert "proposals=12" in captured.out
    assert "valid_baselines=10" in captured.out
    assert "invalid_baselines=2" in captured.out
    assert "conclusive_success=2" in captured.out
    assert "conclusive_task_failure=7" in captured.out
    assert "indeterminate=1" in captured.out
    assert "invalid=2" in captured.out
    assert "execution_error=0" in captured.out
    assert "projected_outcomes=9" in captured.out
    assert "adapter_trust_tier=fixture" in captured.out
    assert "fixture_warning=non-physical-weak-non-simulator" in captured.out

    binding = ReplaySourceBinding.from_paths(source, corruption)
    dataset = load_evaluation_dataset(
        output,
        corruption_dataset=binding.corruption_dataset,
        expected_corruption_digest=binding.corruption_dataset_digest,
    )
    assert tuple(path.name for path in output.iterdir()) == ("manifest.json",)
    assert len(dataset.evidence) == 12
    for evidence in dataset.evidence:
        assert evidence.simulator_replay_verified is False
        if evidence.label_source is not None:
            assert evidence.label_source is LabelSource.DETERMINISTIC_EVALUATOR
            assert evidence.label_strength is LabelStrength.WEAK
    command = dataset.run_manifest.environment.launch_command
    assert str(source) not in command
    assert str(corruption) not in command
    assert str(output) not in command
    assert str(_REPLAY_CONFIG) not in command
    assert command.count("<path>") == 4


def test_resume_does_not_duplicate_or_rewrite_completed_fixture_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, corruption = _source_and_corruption(tmp_path)
    output = tmp_path / "resume"
    assert cli.main(_arguments(source, corruption, output)) == 0
    before = (output / "manifest.json").read_bytes()
    capsys.readouterr()

    return_code = cli.main(_arguments(source, corruption, output, extras=("--resume",)))

    captured = capsys.readouterr()
    assert return_code == 0
    assert "resumed=true" in captured.out
    assert "resumed_without_rerun=12" in captured.out
    assert (output / "manifest.json").read_bytes() == before


def test_max_proposals_selects_the_stable_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, corruption = _source_and_corruption(tmp_path)
    output = tmp_path / "limited"

    return_code = cli.main(
        _arguments(source, corruption, output, extras=("--max-proposals", "4"))
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert "proposals=4" in captured.out
    binding = ReplaySourceBinding.from_paths(source, corruption)
    dataset = load_evaluation_dataset(
        output,
        corruption_dataset=binding.corruption_dataset,
        expected_corruption_digest=binding.corruption_dataset_digest,
    )
    assert dataset.selected_proposal_ids == binding.proposal_ids[:4]


def test_controlled_step_exception_is_execution_error_not_task_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, corruption = _source_and_corruption(tmp_path)
    output = tmp_path / "execution-error"
    config = _write_config(
        tmp_path / "execution-error.json", step_exception_ordinals=[3]
    )

    return_code = cli.main(
        _arguments(
            source,
            corruption,
            output,
            config=config,
            extras=("--max-proposals", "4"),
        )
    )

    captured = capsys.readouterr()
    assert return_code == 1
    assert "execution_error=1" in captured.out
    assert "conclusive_task_failure=0" in captured.out


@pytest.mark.parametrize(
    ("configuration_updates", "expected_reason"),
    [
        ({}, "baseline_task_not_successful"),
        (
            {
                "baseline_failure_ordinals": [],
                "restoration_mismatch_ordinals": [0],
            },
            "baseline_state_restoration_mismatch",
        ),
    ],
)
def test_cli_preserves_controlled_invalid_replay_reasons(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    configuration_updates: dict[str, object],
    expected_reason: str,
) -> None:
    source, corruption = _source_and_corruption(tmp_path)
    output = tmp_path / expected_reason
    config = _write_config(
        tmp_path / "controlled-invalid.json", **configuration_updates
    )

    return_code = cli.main(
        _arguments(
            source,
            corruption,
            output,
            config=config,
            extras=("--max-proposals", "1"),
        )
    )

    assert return_code == 0
    assert "invalid=1" in capsys.readouterr().out
    binding = ReplaySourceBinding.from_paths(source, corruption)
    dataset = load_evaluation_dataset(
        output,
        corruption_dataset=binding.corruption_dataset,
        expected_corruption_digest=binding.corruption_dataset_digest,
    )
    assert len(dataset.evidence) == 1
    assert dataset.evidence[0].termination_reason == expected_reason


def test_invalid_fixture_configuration_fails_before_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, corruption = _source_and_corruption(tmp_path)
    output = tmp_path / "invalid-config-output"
    config = _write_config(tmp_path / "invalid.json", unknown_field=True)

    return_code = cli.main(_arguments(source, corruption, output, config=config))

    captured = capsys.readouterr()
    assert return_code == 1
    assert "replay-data failed:" in captured.err
    assert not output.exists()


def test_replay_rejects_overlapping_dataset_roots(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, corruption = _source_and_corruption(tmp_path)

    return_code = cli.main(_arguments(source, corruption, source))

    captured = capsys.readouterr()
    assert return_code == 1
    assert "replay dataset directories must not equal" in captured.err

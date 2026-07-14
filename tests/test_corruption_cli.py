"""CPU-only CLI tests for M1 corruption-dataset generation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from latentguard import cli
from latentguard.corruptions.serialization import load_corruption_dataset
from latentguard.serialization import load_episodes, save_episodes
from latentguard.synthetic import generate_synthetic_episodes

_SMOKE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "corruptions" / "m1-smoke.json"
)


def _source_bundle(root: Path) -> Path:
    source = root / "source"
    episodes = generate_synthetic_episodes(
        seed=42,
        episode_count=1,
        episode_length=8,
        action_dim=7,
        robot_state_dim=10,
        camera_count=0,
        image_height=2,
        image_width=2,
        depth_enabled=False,
        candidate_count=3,
    )
    save_episodes(episodes, source)
    return source


def _run_cli(source: Path, output: Path, config: Path = _SMOKE_CONFIG) -> int:
    return cli.main(
        [
            "corrupt-data",
            "--input-dir",
            str(source),
            "--output-dir",
            str(output),
            "--config",
            str(config),
            "--seed",
            "314159",
        ]
    )


def test_corrupt_data_processes_synthetic_input_and_reports_unlabeled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    output = tmp_path / "corrupted"

    return_code = _run_cli(source, output)

    captured = capsys.readouterr()
    assert return_code == 0
    assert "corrupt-data OK" in captured.out
    assert "source_episodes=1" in captured.out
    assert "source_candidates=3" in captured.out
    assert "proposals=18" in captured.out
    assert "skipped_non_applicable=0" in captured.out
    assert "unlabeled=true" in captured.out
    assert "round-trip=verified" in captured.out
    dataset = load_corruption_dataset(output)
    assert len(dataset.proposals) == 18
    assert {proposal.corruption_type for proposal in dataset.proposals} == {
        "additive_gaussian_noise",
        "constant_bias",
        "temporal_field_shift",
        "segment_hold",
        "segment_zeroing",
        "local_temporal_permutation",
    }
    manifest = (output / "manifest.json").read_text(encoding="utf-8")
    for forbidden in (
        '"outcome"',
        '"success"',
        '"unsafe"',
        '"label_source"',
        '"label_strength"',
        '"simulator_replay_verified"',
    ):
        assert forbidden not in manifest


def test_corrupt_data_is_deterministic_across_source_locations(tmp_path: Path) -> None:
    source = _source_bundle(tmp_path)
    copied_source = tmp_path / "copied-source"
    shutil.copytree(source, copied_source)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    assert _run_cli(source, first_output) == 0
    assert _run_cli(copied_source, second_output) == 0

    first = load_corruption_dataset(first_output)
    second = load_corruption_dataset(second_output)
    assert first.source_dataset_id == second.source_dataset_id
    assert [proposal.proposal_id for proposal in first.proposals] == [
        proposal.proposal_id for proposal in second.proposals
    ]
    for left, right in zip(first.proposals, second.proposals, strict=True):
        assert (
            left.transformed_action.actions.dtype
            == right.transformed_action.actions.dtype
        )
        assert np.array_equal(
            left.transformed_action.actions, right.transformed_action.actions
        )


def test_corrupt_data_reports_skips_and_strict_mode_fails_before_save(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    raw = json.loads(_SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["corruptions"] = [
        {
            "type": "segment_zeroing",
            "parameters": {"start_step": 7, "end_step": 9},
        }
    ]
    config = tmp_path / "not-applicable.json"
    config.write_text(json.dumps(raw), encoding="utf-8")
    permissive_output = tmp_path / "permissive"

    permissive_code = cli.main(
        [
            "corrupt-data",
            "--input-dir",
            str(source),
            "--output-dir",
            str(permissive_output),
            "--config",
            str(config),
            "--seed",
            "3",
            "--audit",
        ]
    )

    permissive_capture = capsys.readouterr()
    assert permissive_code == 0
    assert "proposals=0" in permissive_capture.out
    assert "skipped_non_applicable=3" in permissive_capture.out
    assert permissive_capture.out.count("corrupt-data audit: skipped") == 3
    assert not load_corruption_dataset(permissive_output).proposals

    strict_output = tmp_path / "strict"
    strict_code = cli.main(
        [
            "corrupt-data",
            "--input-dir",
            str(source),
            "--output-dir",
            str(strict_output),
            "--config",
            str(config),
            "--seed",
            "3",
            "--strict-applicability",
        ]
    )

    strict_capture = capsys.readouterr()
    assert strict_code != 0
    assert "strict applicability failure" in strict_capture.err
    assert not strict_output.exists()


def test_corrupt_data_honors_candidate_and_proposal_limits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    output = tmp_path / "limited"

    return_code = cli.main(
        [
            "corrupt-data",
            "--input-dir",
            str(source),
            "--output-dir",
            str(output),
            "--config",
            str(_SMOKE_CONFIG),
            "--seed",
            "1",
            "--candidate-limit",
            "2",
            "--proposal-limit",
            "7",
        ]
    )

    summary = capsys.readouterr().out
    assert return_code == 0
    assert "source_candidates=2" in summary
    assert "proposals=7" in summary


def test_corrupt_data_invalid_config_returns_nonzero_without_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    config = tmp_path / "invalid.json"
    config.write_text('{"schema_version": NaN}', encoding="utf-8")
    output = tmp_path / "invalid-output"

    return_code = _run_cli(source, output, config)

    assert return_code != 0
    assert "corrupt-data failed" in capsys.readouterr().err
    assert not output.exists()


def test_corrupt_data_invalid_source_returns_nonzero_without_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "invalid-source"
    source.mkdir()
    output = tmp_path / "invalid-output"

    return_code = _run_cli(source, output)

    assert return_code != 0
    assert "corrupt-data failed" in capsys.readouterr().err
    assert not output.exists()


def test_corrupt_data_rejects_output_beneath_source_without_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_bundle(tmp_path)
    before = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    nested_output = source / "arrays" / "corrupted"

    return_code = _run_cli(source, nested_output)

    assert return_code != 0
    assert "must not equal, contain" in capsys.readouterr().err
    assert not nested_output.exists()
    assert len(load_episodes(source)) == 1
    after = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert after == before

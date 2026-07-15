"""CPU-only CLI tests for the Robot Episode Toolkit."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from latentguard.cli import main
from latentguard.serialization import load_episodes, save_episodes


def _bundle(tmp_path: Path) -> Path:
    path = tmp_path / "bundle"
    assert (
        main(
            [
                "sanity-data",
                "--seed",
                "42",
                "--output-dir",
                str(path),
                "--episode-count",
                "1",
                "--episode-length",
                "4",
                "--action-dim",
                "2",
                "--robot-state-dim",
                "3",
                "--camera-count",
                "1",
            ]
        )
        == 0
    )
    return path


def test_three_toolkit_commands_write_reloadable_outputs(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    audit = tmp_path / "audit.json"
    metrics = tmp_path / "metrics.json"
    replay = tmp_path / "replay.jsonl"
    assert main(["audit-data", "--input-dir", str(bundle), "--output", str(audit)]) == 0
    assert (
        main(["summarize-data", "--input-dir", str(bundle), "--output", str(metrics)])
        == 0
    )
    assert (
        main(
            [
                "replay-episode",
                "--input-dir",
                str(bundle),
                "--episode-id",
                "synthetic-s00000042-e0000",
                "--candidate-id",
                "synthetic-s00000042-e0000-candidate-000",
                "--output",
                str(replay),
            ]
        )
        == 0
    )
    assert json.loads(audit.read_text(encoding="utf-8"))["episode_count"] == 1
    assert json.loads(metrics.read_text(encoding="utf-8"))["candidate_count"] == 4
    rows = [
        json.loads(line) for line in replay.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 4
    assert "rgb" not in rows[0]
    assert rows[0]["camera_ids"] == ["camera-00"]


def test_cli_missing_ids_invalid_bundle_and_output_protection(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    output = tmp_path / "metrics.json"
    output.write_text("protected", encoding="utf-8")
    assert (
        main(["summarize-data", "--input-dir", str(bundle), "--output", str(output)])
        == 1
    )
    assert output.read_text(encoding="utf-8") == "protected"
    assert (
        main(
            [
                "replay-episode",
                "--input-dir",
                str(bundle),
                "--episode-id",
                "missing",
                "--candidate-id",
                "missing",
            ]
        )
        == 1
    )
    assert main(["audit-data", "--input-dir", str(tmp_path / "missing")]) == 1


def test_audit_fail_threshold_contract(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    episode = load_episodes(bundle)[0]
    frames = list(episode.observations.frames)
    frames[1] = replace(frames[1], timestamp_s=frames[1].timestamp_s + 0.01)
    warning_bundle = tmp_path / "warning-bundle"
    save_episodes(
        (
            replace(
                episode,
                observations=replace(episode.observations, frames=tuple(frames)),
            ),
        ),
        warning_bundle,
    )
    assert (
        main(
            [
                "audit-data",
                "--input-dir",
                str(warning_bundle),
                "--fail-on",
                "warning",
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "audit-data",
                "--input-dir",
                str(warning_bundle),
                "--fail-on",
                "error",
            ]
        )
        == 0
    )
    assert main(["audit-data", "--input-dir", str(bundle), "--fail-on", "never"]) == 0

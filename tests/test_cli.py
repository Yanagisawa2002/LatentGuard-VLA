"""CPU-only command-line smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from latentguard import cli
from latentguard.serialization import load_episodes


def test_sanity_data_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output_dir = tmp_path / "sanity"

    return_code = cli.main(
        [
            "sanity-data",
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
            "--episode-count",
            "3",
            "--episode-length",
            "8",
            "--action-dim",
            "7",
            "--robot-state-dim",
            "10",
            "--camera-count",
            "2",
            "--depth",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert "episodes=3" in captured.out
    assert "round-trip=verified" in captured.out
    assert (output_dir / "manifest.json").is_file()
    assert len(load_episodes(output_dir)) == 3


def test_sanity_data_rejects_invalid_argument(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "sanity-data",
                "--output-dir",
                str(tmp_path / "invalid"),
                "--episode-count",
                "0",
            ]
        )

    assert error.value.code == 2


def test_sanity_data_returns_nonzero_on_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_generation(**_kwargs: object) -> tuple[()]:
        raise ValueError("invalid synthetic fixture")

    monkeypatch.setattr(cli, "generate_synthetic_episodes", fail_generation)

    return_code = cli.main(["sanity-data", "--output-dir", str(tmp_path / "invalid")])

    assert return_code != 0
    assert "sanity-data failed" in capsys.readouterr().err


def test_remote_sync_dry_run_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("UNRELATED_PRIVATE_TOKEN", "must-not-appear")
    full_sha = "0" * 40

    return_code = cli.main(
        [
            "remote-sync",
            "--dry-run",
            "--host",
            "example-training-host",
            "--repo-dir",
            "/example/latentguard-vla",
            "--branch",
            "codex/m0-data-contract",
            "--commit",
            full_sha,
        ]
    )

    output = capsys.readouterr().out
    assert return_code == 0
    assert "dry-run" in output.lower()
    assert "codex/m0-data-contract" in output
    assert full_sha in output
    assert "must-not-appear" not in output


def test_remote_sync_missing_configuration_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in (
        "LATENTGUARD_REMOTE_HOST",
        "LATENTGUARD_REMOTE_REPO",
        "LATENTGUARD_REMOTE_BRANCH",
        "LATENTGUARD_REMOTE_COMMIT",
    ):
        monkeypatch.delenv(name, raising=False)

    return_code = cli.main(["remote-sync", "--dry-run"])

    assert return_code != 0
    assert "remote-sync failed" in capsys.readouterr().err


def test_remote_sync_rejects_credential_shaped_host_without_echoing_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    credential = "should-not-appear"

    return_code = cli.main(
        [
            "remote-sync",
            "--dry-run",
            "--host",
            f"user:{credential}@training-host",
            "--repo-dir",
            "/example/latentguard-vla",
            "--branch",
            "codex/m0-data-contract",
            "--commit",
            "0" * 40,
        ]
    )

    output = capsys.readouterr().err
    assert return_code != 0
    assert credential not in output
    assert "credential-shaped" in output

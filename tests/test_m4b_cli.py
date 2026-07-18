"""CPU-safe tests for the eight-command M4B workflow surface."""

from __future__ import annotations

import argparse
from pathlib import Path

from latentguard.m4b_cli import M4B_COMMANDS, add_m4b_subparsers, run_m4b_command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_m4b_subparsers(subparsers)
    return parser


def test_registers_exactly_eight_commands() -> None:
    """The public M4B command inventory is fixed and complete."""
    parser = _parser()
    action = next(
        item for item in parser._actions if isinstance(item, argparse._SubParsersAction)
    )
    assert set(action.choices) == set(M4B_COMMANDS)
    assert len(action.choices) == 8


def test_selection_surface_has_no_outcome_input() -> None:
    """Blind selection cannot receive outcome, replay, or evaluator paths."""
    parser = _parser()
    action = next(
        item for item in parser._actions if isinstance(item, argparse._SubParsersAction)
    )
    selection = action.choices["select-visual-action-candidates"]
    destinations = {item.dest for item in selection._actions}
    assert not any(
        token in destination
        for destination in destinations
        for token in ("outcome", "replay", "evaluation")
    )


def test_dry_run_does_not_open_artifacts(tmp_path: Path, capsys: object) -> None:
    """Representative dry runs validate static contracts without opening data."""
    parser = _parser()
    args = parser.parse_args(
        [
            "train-visual-action-verifier",
            "--visual-dataset-dir",
            str(tmp_path / "missing-visual"),
            "--dataset-root",
            str(tmp_path / "missing-data"),
            "--acceptance-report",
            str(tmp_path / "missing-report"),
            "--anchor-manifest-dir",
            str(tmp_path / "missing-anchors"),
            "--model-config",
            "configs/training/m4b/frozen-resnet18-multiview-action.json",
            "--backbone-manifest",
            str(tmp_path / "missing-backbone"),
            "--feature-cache",
            str(tmp_path / "missing-cache"),
            "--output-dir",
            str(tmp_path / "output"),
            "--seed",
            "0",
            "--dry-run",
        ]
    )
    assert run_m4b_command(args) == 0
    assert not (tmp_path / "output").exists()


def test_five_seed_benchmark_is_refused_even_with_flag(tmp_path: Path) -> None:
    """M4B never launches seeds three or four without a future contract change."""
    parser = _parser()
    args = parser.parse_args(
        [
            "benchmark-visual-action-verifier",
            "--stage",
            "five-seed",
            "--run-report",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "output.json"),
            "--explicit-five-seed-authorization",
            "--dry-run",
        ]
    )
    assert run_m4b_command(args) == 1
    assert not (tmp_path / "output.json").exists()

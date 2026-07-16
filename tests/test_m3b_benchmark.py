"""Focused tests for M3B benchmark reuse and resume fail-closed gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest

import latentguard.training.benchmark as benchmark_module
from latentguard.training.benchmark import BenchmarkError, run_benchmark_command
from latentguard.training.config import (
    ModelType,
    load_model_config,
    load_training_config,
)
from latentguard.training.preprocessing import fit_preprocessing_state
from latentguard.training.reporting import StrictReportV1, save_strict_report
from test_training_checkpoint_trainer import _projected_dataset

_CONFIG_ROOT = Path("configs/training/m3b")
_GIT_SHA = "f" * 40


def _arguments(tmp_path: Path, *, training_config: Path) -> argparse.Namespace:
    return argparse.Namespace(
        benchmark_config=_CONFIG_ROOT / "benchmark-five-seed.json",
        training_config=training_config,
        config_dir=_CONFIG_ROOT,
        mode="smoke",
        max_steps=1,
        max_epochs=1,
        limit_samples=None,
        num_workers=None,
        dry_run=True,
        output_dir=tmp_path / "benchmark",
        force_rerun=False,
        resume=False,
        device="cpu",
        profile_memory=False,
    )


def _stable_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        benchmark_module,
        "_git_value",
        lambda *arguments: (
            _GIT_SHA if arguments == ("rev-parse", "HEAD") else "codex/test"
        ),
    )


def _dry_run_identity(
    args: argparse.Namespace, capsys: pytest.CaptureFixture[str]
) -> str:
    assert run_benchmark_command(args, load_dataset=lambda _: _projected_dataset()) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    return str(output["orchestration_identity"])


def _write_completed_summary(output_root: Path, *, orchestration_identity: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    save_strict_report(
        StrictReportV1(
            benchmark_module.BENCHMARK_REPORT_TYPE,
            {
                "completed": True,
                "mode": "smoke",
                "orchestration_identity": orchestration_identity,
                "run_count": 4,
            },
        ),
        output_root / "benchmark-summary.json",
    )


def test_completed_summary_is_rejected_after_training_config_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A same-mode completed summary cannot survive semantic recipe drift."""

    _stable_git(monkeypatch)
    config_path = tmp_path / "training.json"
    configuration = json.loads(
        (_CONFIG_ROOT / "train-default.json").read_text(encoding="utf-8")
    )
    config_path.write_text(json.dumps(configuration), encoding="utf-8")
    args = _arguments(tmp_path, training_config=config_path)
    original_identity = _dry_run_identity(args, capsys)
    _write_completed_summary(args.output_dir, orchestration_identity=original_identity)

    configuration["learning_rate"] = 0.002
    config_path.write_text(json.dumps(configuration), encoding="utf-8")
    args.dry_run = False
    monkeypatch.setattr(
        benchmark_module,
        "_validate_completed_run",
        lambda **_: pytest.fail("stale summary must fail before run reuse"),
    )

    with pytest.raises(BenchmarkError, match="orchestration identity differs"):
        run_benchmark_command(args, load_dataset=lambda _: _projected_dataset())


def test_matching_completed_summary_revalidates_every_run_before_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A matching top-level summary alone is insufficient for reuse."""

    _stable_git(monkeypatch)
    args = _arguments(tmp_path, training_config=_CONFIG_ROOT / "train-default.json")
    identity = _dry_run_identity(args, capsys)
    _write_completed_summary(args.output_dir, orchestration_identity=identity)
    calls: list[tuple[ModelType, int, Path]] = []

    def record_validation(**values: object) -> None:
        calls.append(
            (
                values["model_type"],
                values["seed"],
                values["run_root"],
            )
        )

    monkeypatch.setattr(benchmark_module, "_validate_completed_run", record_validation)
    args.dry_run = False

    assert run_benchmark_command(args, load_dataset=lambda _: _projected_dataset()) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["reused"] is True
    assert {(model_type, seed) for model_type, seed, _ in calls} == {
        (model_type, 0) for model_type in ModelType
    }
    assert all(
        run_root
        == args.output_dir.absolute() / "runs" / model_type.value / f"seed-{seed}"
        for model_type, seed, run_root in calls
    )


def test_incomplete_run_requires_resume_and_matching_periodic_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Incomplete directories never silently restart or resume without state."""

    dataset = _projected_dataset()
    preprocessing = fit_preprocessing_state(dataset)
    model_type = ModelType.STATE_ONLY_MLP
    model_config = load_model_config(_CONFIG_ROOT / "state-only-mlp.json")
    training_config = replace(
        load_training_config(_CONFIG_ROOT / "train-default.json"),
        batch_size=4,
        max_epochs=1,
        max_steps=1,
    )
    run_root = tmp_path / "runs" / model_type.value / "seed-0"
    run_root.mkdir(parents=True)
    (run_root / "interrupted.marker").write_text("interrupted", encoding="utf-8")
    args = argparse.Namespace(resume=False, device="cpu", profile_memory=False)

    with pytest.raises(BenchmarkError, match="requires --resume"):
        benchmark_module._train_one(
            root=tmp_path,
            model_type=model_type,
            model_config=model_config,
            training_config=training_config,
            dataset=dataset,
            preprocessing=preprocessing,
            seed=0,
            args=args,
            git_sha=_GIT_SHA,
            branch="codex/test",
        )

    args.resume = True
    monkeypatch.setattr(
        benchmark_module,
        "_validate_run_manifest",
        lambda *_, **__: {"started_at_utc": "2026-07-16T00:00:00+00:00"},
    )
    monkeypatch.setattr(
        benchmark_module, "run_forward_backward_gate", lambda *_, **__: 0.0
    )
    monkeypatch.setattr(
        benchmark_module,
        "train_action_verifier",
        lambda *_, **__: pytest.fail("resume must require the periodic checkpoint"),
    )

    with pytest.raises(
        BenchmarkError, match="matching periodic checkpoint is unavailable"
    ):
        benchmark_module._train_one(
            root=tmp_path,
            model_type=model_type,
            model_config=model_config,
            training_config=training_config,
            dataset=dataset,
            preprocessing=preprocessing,
            seed=0,
            args=args,
            git_sha=_GIT_SHA,
            branch="codex/test",
        )


def test_real_smoke_completion_reuses_only_complete_bound_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the four-model smoke and reject later history drift."""

    _stable_git(monkeypatch)
    args = _arguments(tmp_path, training_config=_CONFIG_ROOT / "train-default.json")
    args.dry_run = False
    args.max_steps = 2
    args.max_epochs = 1
    args.limit_samples = 12

    assert run_benchmark_command(args, load_dataset=lambda _: _projected_dataset()) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["run_count"] == 4
    assert (args.output_dir / "benchmark-summary.json").is_file()

    assert run_benchmark_command(args, load_dataset=lambda _: _projected_dataset()) == 0
    reused = json.loads(capsys.readouterr().out)
    assert reused["reused"] is True

    history = (
        args.output_dir
        / "runs"
        / ModelType.STATE_ONLY_MLP.value
        / "seed-0"
        / "history.json"
    )
    history.write_text(history.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="manifest or history bytes differ"):
        run_benchmark_command(args, load_dataset=lambda _: _projected_dataset())

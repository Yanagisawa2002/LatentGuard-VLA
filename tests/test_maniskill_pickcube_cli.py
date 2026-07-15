"""CPU-only command tests for the optional PickCube integration."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest

from latentguard import cli
from latentguard.corruptions.config import load_corruption_plan
from latentguard.corruptions.generation import generate_corruption_proposals
from latentguard.corruptions.serialization import (
    CorruptionDataset,
    save_corruption_dataset,
)
from latentguard.integrations.maniskill_pickcube.source_generation import (
    OFFICIAL_SOLVER_MODULE,
    SolverSourceIdentity,
)
from latentguard.replay.fixture import (
    FIXTURE_ADAPTER_ID,
    DeterministicReplayFixtureAdapter,
)
from latentguard.replay.registry import (
    create_replay_evaluator as create_registered_replay_evaluator,
)
from latentguard.replay.registry import load_replay_adapter_configuration
from latentguard.serialization import (
    compute_episode_bundle_identifier,
    save_episodes,
)
from latentguard.synthetic import generate_synthetic_episodes

_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED = (
    _ROOT
    / "configs"
    / "integrations"
    / "maniskill_pickcube"
    / "expected-contract-v1.json"
)
_CORRUPTION_CONFIG = _ROOT / "configs" / "corruptions" / "m1-smoke.json"
_FIXTURE_CONFIG = _ROOT / "configs" / "replay" / "m2b-fixture.json"


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
        candidate_count=3,
    )
    save_episodes(episodes, source)
    plan = load_corruption_plan(_CORRUPTION_CONFIG)
    generated = generate_corruption_proposals(
        episodes,
        plan.action_layout,
        plan.corruptions,
        base_seed=314159,
        proposal_limit=6,
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


def _replay_arguments(
    source: Path,
    corruption: Path,
    output: Path,
    *,
    extras: tuple[str, ...] = (),
) -> list[str]:
    root = source.parent
    return [
        "replay-maniskill-pickcube",
        "--source-dir",
        str(source),
        "--corruption-dir",
        str(corruption),
        "--runtime-archive-dir",
        str(root / "runtime-archive"),
        "--output-dir",
        str(output),
        "--compatibility-report",
        str(root / "compatibility.json"),
        "--expected-contract",
        str(_EXPECTED),
        "--action-layout",
        str(root / "action-layout.json"),
        "--seed",
        "271828",
        *extras,
    ]


def _replace_real_adapter_with_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_configuration = load_replay_adapter_configuration(_FIXTURE_CONFIG)

    def create_fake_evaluator(
        name: str,
        configuration: object,
        binding: object,
    ) -> object:
        assert name == "maniskill_pickcube_v1"
        assert isinstance(configuration, dict)
        return create_registered_replay_evaluator(
            FIXTURE_ADAPTER_ID,
            fixture_configuration,
            binding,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(cli, "create_replay_evaluator", create_fake_evaluator)


@pytest.mark.parametrize(
    "command",
    (
        "probe-maniskill-pickcube",
        "collect-maniskill-pickcube",
        "replay-maniskill-pickcube",
    ),
)
def test_pickcube_command_help_requires_no_optional_runtime(
    command: str,
) -> None:
    with pytest.raises(SystemExit) as captured:
        cli.main([command, "--help"])

    assert captured.value.code == 0


def test_probe_command_accepts_an_injected_fake_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "compatibility.json"
    runtime = object()
    observed: dict[str, object] = {}

    def fake_probe(
        expected: object,
        *,
        runtime: object,
        report_path: Path,
        reset_seed: int,
        require_trusted: bool,
    ) -> object:
        observed.update(
            expected=expected,
            runtime=runtime,
            reset_seed=reset_seed,
            require_trusted=require_trusted,
        )
        report_path.write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            report=SimpleNamespace(compatibility_identity="sha256:" + "a" * 64),
            unresolved_fields=("mplib_version",),
            trusted_replay_ready=False,
        )

    monkeypatch.setattr(cli, "probe_maniskill_pickcube", fake_probe)
    args = Namespace(
        expected_contract=_EXPECTED,
        output=output,
        seed=17,
        require_trusted=False,
    )

    assert cli._run_probe_maniskill_pickcube(args, runtime=runtime) == 0
    assert observed["runtime"] is runtime
    assert observed["reset_seed"] == 17
    assert output.is_file()
    stdout = capsys.readouterr().out
    assert "trusted_replay_ready=false" in stdout
    assert "report_written=true" in stdout
    assert str(output) not in stdout


def test_collect_command_routes_a_fake_solver_without_optional_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_digest = "b" * 64
    solver_identity = SolverSourceIdentity(
        module_name=OFFICIAL_SOLVER_MODULE,
        export_name="solve",
        source_sha256=raw_digest,
    )
    report = SimpleNamespace(
        compatibility_identity="sha256:" + "c" * 64,
        source_solver=SimpleNamespace(
            module_name=solver_identity.module_name,
            source_sha256="sha256:" + raw_digest,
        ),
    )
    binding = SimpleNamespace(report=report)
    checked_layout = SimpleNamespace(
        coordinate_frame="fake_joint_target",
        m1_action_layout=object(),
    )
    environment_factory = object()

    def solver(*args: object, **kwargs: object) -> None:
        del args, kwargs

    observed: dict[str, object] = {}
    fake_result = SimpleNamespace(archive=SimpleNamespace(episodes=()))
    summary = {
        "accepted_source_count": 1,
        "archive_content_digest": "sha256:" + "d" * 64,
        "attempt_count": 1,
        "independent_baseline_success_count": 1,
        "source_dataset_id": "sha256:" + "e" * 64,
    }

    monkeypatch.setattr(cli, "_load_trusted_pickcube_binding", lambda *_: binding)
    monkeypatch.setattr(
        cli,
        "load_maniskill_pickcube_action_layout",
        lambda _: checked_layout,
    )
    monkeypatch.setattr(
        cli,
        "validate_maniskill_pickcube_action_layout_binding",
        lambda *args: "sha256:" + "f" * 64,
    )
    monkeypatch.setattr(
        cli,
        "environment_settings_from_compatibility",
        lambda _: SimpleNamespace(sim_backend="gpu"),
    )
    monkeypatch.setattr(
        cli,
        "action_contract_from_compatibility",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        cli,
        "PickCubeTaskKeyContract",
        SimpleNamespace(from_compatibility_report=lambda _: object()),
    )

    def fake_collect(**kwargs: object) -> object:
        observed.update(kwargs)
        return fake_result

    monkeypatch.setattr(cli, "collect_reference_archive", fake_collect)
    monkeypatch.setattr(
        cli,
        "_publish_pickcube_collection",
        lambda *args, **kwargs: summary,
    )
    args = Namespace(
        runtime_archive_dir=tmp_path / "archive",
        source_dir=tmp_path / "source",
        compatibility_report=tmp_path / "report.json",
        expected_contract=_EXPECTED,
        action_layout=tmp_path / "layout.json",
        summary_output=tmp_path / "summary.json",
        requested_success_count=1,
        starting_seed=5,
        maximum_attempts=2,
        sim_backend="gpu",
        trajectory_action_limit=2,
    )

    assert (
        cli._run_collect_maniskill_pickcube(
            args,
            environment_factory=environment_factory,  # type: ignore[arg-type]
            solver=solver,
            solver_identity=solver_identity,
        )
        == 0
    )
    assert observed["environment_factory"] is environment_factory
    assert observed["solver"] is solver
    assert observed["trajectory_action_limit"] == 2
    stdout = capsys.readouterr().out
    assert "accepted=1" in stdout
    assert str(args.summary_output) not in stdout


def test_collection_publish_rolls_back_interrupt_after_directory_appears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_dir = tmp_path / "archive"
    source_dir = tmp_path / "source"
    summary_output = tmp_path / "summary.json"

    def interrupted_save(archive: object, output: Path) -> NoReturn:
        del archive
        output.mkdir()
        (output / "partial.json").write_text("{}\n", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "save_reference_archive", interrupted_save)
    result = SimpleNamespace(archive=object())

    with pytest.raises(KeyboardInterrupt):
        cli._publish_pickcube_collection(
            result,  # type: ignore[arg-type]
            archive_dir=archive_dir,
            source_dir=source_dir,
            summary_output=summary_output,
        )

    assert not archive_dir.exists()
    assert not source_dir.exists()
    assert not summary_output.exists()


def test_pickcube_replay_dry_run_plans_without_fake_sessions_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, corruption = _source_and_corruption(tmp_path)
    output = tmp_path / "dry-run"
    _replace_real_adapter_with_fixture(monkeypatch)

    def forbidden_session(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("dry run must not create a simulator session")

    monkeypatch.setattr(
        DeterministicReplayFixtureAdapter,
        "create_session",
        forbidden_session,
    )

    code = cli.main(
        _replay_arguments(
            source,
            corruption,
            output,
            extras=("--dry-run", "--audit", "--max-proposals", "3"),
        )
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "replay-maniskill-pickcube dry-run OK" in captured.out
    assert captured.out.count("replay-maniskill-pickcube dry-run audit:") == 3
    assert "sessions=0 output_created=false" in captured.out
    assert not output.exists()


def test_pickcube_replay_fake_run_resume_and_manifest_path_sanitization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, corruption = _source_and_corruption(tmp_path)
    output = tmp_path / "replay"
    _replace_real_adapter_with_fixture(monkeypatch)

    assert cli.main(_replay_arguments(source, corruption, output)) == 0
    capsys.readouterr()
    assert (
        cli.main(_replay_arguments(source, corruption, output, extras=("--resume",)))
        == 0
    )

    captured = capsys.readouterr()
    assert "resumed=true" in captured.out
    assert "resumed_without_rerun=6" in captured.out
    manifest = (output / "manifest.json").read_text(encoding="utf-8")
    for raw_path in (
        source,
        corruption,
        tmp_path / "runtime-archive",
        output,
        tmp_path / "compatibility.json",
        _EXPECTED,
        tmp_path / "action-layout.json",
    ):
        assert str(raw_path) not in manifest
    assert manifest.count("<path>") >= 7


def test_pickcube_replay_retry_requires_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, corruption = _source_and_corruption(tmp_path)
    output = tmp_path / "invalid-retry"
    _replace_real_adapter_with_fixture(monkeypatch)

    code = cli.main(
        _replay_arguments(
            source,
            corruption,
            output,
            extras=("--retry-execution-errors",),
        )
    )

    assert code == 1
    assert "requires --resume" in capsys.readouterr().err
    assert not output.exists()

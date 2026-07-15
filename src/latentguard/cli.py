"""Command-line interface for local data checks and remote synchronization."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latentguard.corruptions.config import load_corruption_plan
from latentguard.corruptions.generation import (
    build_proposal_identifier,
    generate_corruption_proposals,
)
from latentguard.corruptions.serialization import (
    CorruptionDataset,
    load_corruption_dataset,
    save_corruption_dataset,
)
from latentguard.evaluation.registry import (
    create_evaluator,
    load_evaluator_configuration,
)
from latentguard.evaluation.reporting import (
    evaluation_audit_lines,
    format_evaluation_summary,
)
from latentguard.evaluation.runner import (
    plan_evaluation,
    run_evaluation,
    sanitize_operational_text,
)
from latentguard.evaluation.serialization import (
    LedgerState,
    compute_corruption_dataset_digest,
    load_evaluation_dataset,
)
from latentguard.integrations.maniskill_pickcube.adapter import (
    MANISKILL_PICKCUBE_ADAPTER_ID,
)
from latentguard.integrations.maniskill_pickcube.archive import (
    load_reference_archive,
    save_reference_archive,
)
from latentguard.integrations.maniskill_pickcube.compatibility import (
    CompatibilityBinding,
    load_compatibility_report,
    validate_compatibility_report,
)
from latentguard.integrations.maniskill_pickcube.configuration import (
    load_expected_contract,
    load_maniskill_pickcube_action_layout,
    validate_maniskill_pickcube_action_layout_binding,
)
from latentguard.integrations.maniskill_pickcube.probe import (
    ManiSkillProbeRuntime,
    probe_maniskill_pickcube,
)
from latentguard.integrations.maniskill_pickcube.reporting import (
    build_collection_summary,
    build_incomplete_collection_summary,
    format_collection_summary,
    write_sanitized_report,
)
from latentguard.integrations.maniskill_pickcube.serialization import (
    action_contract_from_compatibility,
    environment_settings_from_compatibility,
)
from latentguard.integrations.maniskill_pickcube.source_generation import (
    LazyManiSkillSourceEnvironmentFactory,
    PickCubeSourceCollectionIncompleteError,
    SolverSourceIdentity,
    SourceCollectionResult,
    SourceEnvironmentFactory,
    collect_reference_archive,
    load_official_solver,
)
from latentguard.integrations.maniskill_pickcube.source_import import (
    save_m0_source_dataset,
)
from latentguard.integrations.maniskill_pickcube.task_evidence import (
    PickCubeTaskKeyContract,
)
from latentguard.remote import RemoteSyncError, resolve_remote_config, sync_remote
from latentguard.replay.evaluator import ExactStatePairedReplayEvaluator
from latentguard.replay.registry import (
    create_replay_evaluator,
    load_replay_adapter_configuration,
)
from latentguard.replay.reporting import (
    build_replay_summary,
    format_replay_summary,
    replay_audit_lines,
)
from latentguard.replay.source import ReplaySourceBinding
from latentguard.serialization import (
    compute_episode_bundle_identifier,
    load_episodes,
    save_episodes,
)
from latentguard.synthetic import generate_synthetic_episodes
from latentguard.toolkit import (
    AuditConfig,
    ReplayAlignmentError,
    audit_episodes,
    build_replay_plan,
    iter_replay_steps,
    summarize_episodes,
    write_json,
    write_replay_jsonl,
)
from latentguard.validation import validate_episodes

_DEFAULT_PICKCUBE_EXPECTED_CONTRACT = Path(
    "configs/integrations/maniskill_pickcube/expected-contract-v1.json"
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latentguard",
        description="LatentGuard-VLA local validation utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sanity = subparsers.add_parser(
        "sanity-data",
        help="generate and round-trip deterministic synthetic episodes",
    )
    sanity.add_argument("--seed", type=int, default=0)
    sanity.add_argument("--output-dir", type=Path, required=True)
    sanity.add_argument("--episode-count", type=_positive_int, default=3)
    sanity.add_argument("--episode-length", type=_positive_int, default=8)
    sanity.add_argument("--action-dim", type=_positive_int, default=7)
    sanity.add_argument("--robot-state-dim", type=_positive_int, default=10)
    sanity.add_argument("--camera-count", type=_nonnegative_int, default=1)
    sanity.add_argument("--image-height", type=_positive_int, default=8)
    sanity.add_argument("--image-width", type=_positive_int, default=8)
    sanity.add_argument("--candidate-count", type=_positive_int, default=4)
    sanity.add_argument(
        "--depth",
        action="store_true",
        help="include aligned synthetic depth for every camera",
    )

    audit_data = subparsers.add_parser(
        "audit-data", help="strictly validate and read-only audit an Episode bundle"
    )
    audit_data.add_argument("--input-dir", type=Path, required=True)
    audit_data.add_argument("--output", type=Path)
    audit_data.add_argument(
        "--fail-on", choices=("error", "warning", "never"), default="error"
    )

    summarize_data = subparsers.add_parser(
        "summarize-data", help="summarize candidate-level Episode bundle metrics"
    )
    summarize_data.add_argument("--input-dir", type=Path, required=True)
    summarize_data.add_argument("--output", type=Path)

    replay_episode = subparsers.add_parser(
        "replay-episode", help="iterate one Episode candidate offline by exact index"
    )
    replay_episode.add_argument("--input-dir", type=Path, required=True)
    replay_episode.add_argument("--episode-id", required=True)
    replay_episode.add_argument("--candidate-id", required=True)
    replay_episode.add_argument("--output", type=Path)

    remote = subparsers.add_parser(
        "remote-sync",
        help="synchronize a remote checkout to an exact pushed revision",
    )
    remote.add_argument("--dry-run", action="store_true")
    remote.add_argument("--host", help="SSH host alias")
    remote.add_argument("--repo-dir", help="remote repository directory")
    remote.add_argument("--branch", help="Git branch to synchronize")
    remote.add_argument("--commit", dest="expected_commit", help="full commit SHA")
    remote.add_argument("--ssh-executable", help="system SSH executable")
    remote.add_argument("--connect-timeout", type=_positive_int)

    corrupt = subparsers.add_parser(
        "corrupt-data",
        help="generate deterministic unlabeled action corruption proposals",
    )
    corrupt.add_argument("--input-dir", type=Path, required=True)
    corrupt.add_argument("--output-dir", type=Path, required=True)
    corrupt.add_argument("--config", type=Path, required=True)
    corrupt.add_argument("--seed", type=_nonnegative_int, required=True)
    corrupt.add_argument("--episode-limit", type=_positive_int)
    corrupt.add_argument("--candidate-limit", type=_positive_int)
    corrupt.add_argument("--proposal-limit", type=_positive_int)
    corrupt.add_argument(
        "--strict-applicability",
        action="store_true",
        help="fail instead of recording a non-applicable transformation skip",
    )
    corrupt.add_argument(
        "--audit",
        "--verbose",
        dest="audit",
        action="store_true",
        help="print concise applicability-skip audit records",
    )

    evaluate = subparsers.add_parser(
        "evaluate-data",
        help="evaluate ordered corruption proposals with resumable evidence",
    )
    evaluate.add_argument("--corruption-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--evaluator", required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--seed", type=_nonnegative_int, required=True)
    evaluate.add_argument("--max-proposals", type=_positive_int)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--fail-fast", action="store_true")
    evaluate.add_argument("--retry-execution-errors", action="store_true")
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.add_argument(
        "--audit",
        "--verbose",
        dest="audit",
        action="store_true",
        help="print concise proposal-attempt audit records",
    )

    replay = subparsers.add_parser(
        "replay-data",
        help="run content-bound exact-state paired replay through an adapter",
    )
    replay.add_argument("--source-dir", type=Path, required=True)
    replay.add_argument("--corruption-dir", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument("--adapter", required=True)
    replay.add_argument("--config", type=Path, required=True)
    replay.add_argument("--seed", type=_nonnegative_int, required=True)
    replay.add_argument("--max-proposals", type=_positive_int)
    replay.add_argument("--resume", action="store_true")
    replay.add_argument("--fail-fast", action="store_true")
    replay.add_argument("--retry-execution-errors", action="store_true")
    replay.add_argument("--dry-run", action="store_true")
    replay.add_argument(
        "--audit",
        "--verbose",
        dest="audit",
        action="store_true",
        help="print concise paired-replay attempt audit records",
    )

    probe_pickcube = subparsers.add_parser(
        "probe-maniskill-pickcube",
        help="inspect the pinned PickCube runtime and write a sanitized contract",
    )
    probe_pickcube.add_argument(
        "--expected-contract",
        type=Path,
        default=_DEFAULT_PICKCUBE_EXPECTED_CONTRACT,
    )
    probe_pickcube.add_argument("--output", type=Path, required=True)
    probe_pickcube.add_argument("--seed", type=_nonnegative_int, default=0)
    probe_pickcube.add_argument(
        "--require-trusted",
        action="store_true",
        help="also require the observed report to match a fully resolved contract",
    )

    collect_pickcube = subparsers.add_parser(
        "collect-maniskill-pickcube",
        help="collect independently replayed official PickCube source trajectories",
    )
    collect_pickcube.add_argument("--runtime-archive-dir", type=Path, required=True)
    collect_pickcube.add_argument("--source-dir", type=Path, required=True)
    collect_pickcube.add_argument("--compatibility-report", type=Path, required=True)
    collect_pickcube.add_argument(
        "--expected-contract",
        type=Path,
        default=_DEFAULT_PICKCUBE_EXPECTED_CONTRACT,
    )
    collect_pickcube.add_argument("--action-layout", type=Path, required=True)
    collect_pickcube.add_argument("--summary-output", type=Path, required=True)
    collect_pickcube.add_argument(
        "--requested-success-count", type=_positive_int, default=6
    )
    collect_pickcube.add_argument("--starting-seed", type=_nonnegative_int, default=0)
    collect_pickcube.add_argument("--maximum-attempts", type=_positive_int, default=24)
    collect_pickcube.add_argument("--sim-backend", choices=("gpu",), default="gpu")
    collect_pickcube.add_argument(
        "--trajectory-action-limit",
        type=_positive_int,
        help="stop and reject a solver attempt before it exceeds this action count",
    )

    replay_pickcube = subparsers.add_parser(
        "replay-maniskill-pickcube",
        help="run trusted PickCube paired replay through the standard M2A ledger",
    )
    replay_pickcube.add_argument("--source-dir", type=Path, required=True)
    replay_pickcube.add_argument("--corruption-dir", type=Path, required=True)
    replay_pickcube.add_argument("--runtime-archive-dir", type=Path, required=True)
    replay_pickcube.add_argument("--output-dir", type=Path, required=True)
    replay_pickcube.add_argument("--compatibility-report", type=Path, required=True)
    replay_pickcube.add_argument(
        "--expected-contract",
        type=Path,
        default=_DEFAULT_PICKCUBE_EXPECTED_CONTRACT,
    )
    replay_pickcube.add_argument("--action-layout", type=Path, required=True)
    replay_pickcube.add_argument("--seed", type=_nonnegative_int, required=True)
    replay_pickcube.add_argument("--max-proposals", type=_positive_int)
    replay_pickcube.add_argument("--resume", action="store_true")
    replay_pickcube.add_argument("--fail-fast", action="store_true")
    replay_pickcube.add_argument("--retry-execution-errors", action="store_true")
    replay_pickcube.add_argument("--dry-run", action="store_true")
    replay_pickcube.add_argument(
        "--audit",
        "--verbose",
        dest="audit",
        action="store_true",
        help="print concise paired-replay attempt audit records",
    )

    return parser


def _values_equal(left: Any, right: Any) -> bool:
    """Compare nested model values without invoking ndarray dataclass equality."""
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and bool(np.array_equal(left, right))
        )
    if is_dataclass(left) and not isinstance(left, type):
        return all(
            _values_equal(getattr(left, field.name), getattr(right, field.name))
            for field in fields(left)
        )
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)


def _run_sanity_data(args: argparse.Namespace) -> int:
    try:
        episodes = generate_synthetic_episodes(
            seed=args.seed,
            episode_count=args.episode_count,
            episode_length=args.episode_length,
            action_dim=args.action_dim,
            robot_state_dim=args.robot_state_dim,
            camera_count=args.camera_count,
            image_height=args.image_height,
            image_width=args.image_width,
            depth_enabled=args.depth,
            candidate_count=args.candidate_count,
        )
        validate_episodes(episodes)
        manifest_path = save_episodes(episodes, args.output_dir)
        reloaded = load_episodes(args.output_dir)
        validate_episodes(reloaded)
        if not _values_equal(episodes, reloaded):
            raise RuntimeError("round-trip comparison found a changed value")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"sanity-data failed: {error}", file=sys.stderr)
        return 1

    observation_count = sum(len(episode.observations.frames) for episode in episodes)
    candidate_count = sum(len(episode.candidates) for episode in episodes)
    print(
        "sanity-data OK: "
        f"episodes={len(episodes)} observations={observation_count} "
        f"candidates={candidate_count} round-trip=verified "
        f"manifest={manifest_path}"
    )
    return 0


def _run_audit_data(args: argparse.Namespace) -> int:
    try:
        episodes = load_episodes(args.input_dir)
        report = audit_episodes(episodes, AuditConfig())
        if args.output is not None:
            write_json(report, args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"audit-data failed: {error}", file=sys.stderr)
        return 1
    counts = report.issues_by_severity
    print(
        "audit-data OK: "
        f"episodes={report.episode_count} candidates={report.candidate_count} "
        f"issues={report.issue_count} errors={counts.get('error', 0)} "
        f"warnings={counts.get('warning', 0)} info={counts.get('info', 0)}"
    )
    if args.fail_on == "error":
        return int(counts.get("error", 0) > 0)
    if args.fail_on == "warning":
        return int(counts.get("error", 0) + counts.get("warning", 0) > 0)
    return 0


def _run_summarize_data(args: argparse.Namespace) -> int:
    try:
        metrics = summarize_episodes(load_episodes(args.input_dir))
        if args.output is not None:
            write_json(metrics, args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"summarize-data failed: {error}", file=sys.stderr)
        return 1
    print(
        "summarize-data OK: "
        f"episodes={metrics.episode_count} observations={metrics.observation_count} "
        f"candidates={metrics.candidate_count} "
        f"candidate_success_rate={metrics.candidate_success_rate:.6f} "
        f"candidate_unsafe_rate={metrics.candidate_unsafe_rate:.6f}"
    )
    return 0


def _run_replay_episode(args: argparse.Namespace) -> int:
    try:
        episodes = load_episodes(args.input_dir)
        episode = next(
            (item for item in episodes if item.episode_id == args.episode_id), None
        )
        if episode is None:
            raise ReplayAlignmentError(f"episode not found: {args.episode_id!r}")
        plan = build_replay_plan(episode, args.candidate_id)
        steps = tuple(iter_replay_steps(episode, args.candidate_id))
        if args.output is not None:
            write_replay_jsonl(steps, args.output)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"replay-episode failed: {error}", file=sys.stderr)
        return 1
    print(
        "replay-episode OK (offline only): "
        f"episode={plan.episode_id} candidate={plan.candidate_id} "
        f"steps={plan.step_count} coordinate_frame={plan.coordinate_frame} "
        f"control_period_s={plan.control_period_s:.17g}"
    )
    return 0


def _run_remote_sync(args: argparse.Namespace) -> int:
    try:
        config = resolve_remote_config(
            host=args.host,
            repo_dir=args.repo_dir,
            branch=args.branch,
            expected_commit=args.expected_commit,
            ssh_executable=args.ssh_executable,
            connect_timeout=args.connect_timeout,
        )
        result = sync_remote(config, dry_run=args.dry_run)
    except (RemoteSyncError, OSError, ValueError) as error:
        print(f"remote-sync failed: {error}", file=sys.stderr)
        return 1

    print(result)
    return 0


def _run_probe_maniskill_pickcube(
    args: argparse.Namespace,
    *,
    runtime: ManiSkillProbeRuntime | None = None,
) -> int:
    """Run the bounded real/fake compatibility probe and publish one report."""
    try:
        expected = load_expected_contract(args.expected_contract)
        binding = probe_maniskill_pickcube(
            expected,
            runtime=runtime,
            report_path=args.output,
            reset_seed=args.seed,
            require_trusted=args.require_trusted,
        )
        unresolved = ",".join(binding.unresolved_fields) or "none"
        print(
            "probe-maniskill-pickcube OK: "
            f"compatibility_identity={binding.report.compatibility_identity} "
            f"trusted_replay_ready={str(binding.trusted_replay_ready).lower()} "
            f"unresolved_fields={unresolved} report_written=true"
        )
        return 0
    except KeyboardInterrupt:
        print("probe-maniskill-pickcube interrupted", file=sys.stderr)
        return 130
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            f"probe-maniskill-pickcube failed: {sanitize_operational_text(error)}",
            file=sys.stderr,
        )
        return 1


def _load_trusted_pickcube_binding(
    expected_contract_path: Path,
    compatibility_report_path: Path,
) -> CompatibilityBinding:
    """Load local expectations and require an exact trusted report binding."""
    expected = load_expected_contract(expected_contract_path)
    report = load_compatibility_report(compatibility_report_path)
    return validate_compatibility_report(report, expected, require_trusted=True)


def _validate_new_pickcube_collection_outputs(
    archive_dir: Path,
    source_dir: Path,
    summary_output: Path,
) -> tuple[Path, Path, Path]:
    """Require absent, non-overlapping outputs before a transactional collection."""
    archive = archive_dir.absolute().resolve()
    source = source_dir.absolute().resolve()
    summary = summary_output.absolute().resolve()
    for name, path in (
        ("runtime archive", archive),
        ("source dataset", source),
        ("collection summary", summary),
    ):
        if path.exists() or path.is_symlink():
            raise ValueError(f"{name} output must not already exist")
    if (
        archive == source
        or archive.is_relative_to(source)
        or source.is_relative_to(archive)
    ):
        raise ValueError("runtime archive and source dataset outputs must not overlap")
    if summary.is_relative_to(archive) or summary.is_relative_to(source):
        raise ValueError(
            "collection summary must be outside archive and source dataset roots"
        )
    return archive, source, summary


def _publish_pickcube_collection(
    result: SourceCollectionResult,
    *,
    archive_dir: Path,
    source_dir: Path,
    summary_output: Path,
) -> Mapping[str, object]:
    """Publish archive, M0 source, and summary with rollback of new outputs."""
    archive = result.archive
    if archive is None:
        raise ValueError("cannot publish an incomplete PickCube collection")
    try:
        save_reference_archive(archive, archive_dir)
        reloaded_archive = load_reference_archive(archive_dir)
        if reloaded_archive.content_digest != archive.content_digest:
            raise RuntimeError("runtime archive round-trip content digest changed")

        save_m0_source_dataset(archive, source_dir)
        reloaded_episodes = load_episodes(source_dir)
        validate_episodes(reloaded_episodes)
        source_dataset_id = compute_episode_bundle_identifier(source_dir)
        summary = build_collection_summary(
            result,
            source_dataset_id=source_dataset_id,
        )
        write_sanitized_report(summary, summary_output)
        return summary
    except BaseException as error:
        # All three destinations were required absent before this function. Check
        # every destination directly so an asynchronous KeyboardInterrupt between
        # a successful publish call and the next Python bytecode cannot escape
        # rollback bookkeeping.
        for path in (summary_output,):
            try:
                if path.exists() or path.is_symlink():
                    path.unlink()
            except OSError as cleanup_error:
                error.add_note(f"collection file rollback failed: {cleanup_error}")
        for path in (source_dir, archive_dir):
            try:
                if path.exists() or path.is_symlink():
                    shutil.rmtree(path)
            except OSError as cleanup_error:
                error.add_note(f"collection directory rollback failed: {cleanup_error}")
        raise


def _run_collect_maniskill_pickcube(
    args: argparse.Namespace,
    *,
    environment_factory: SourceEnvironmentFactory | None = None,
    solver: Callable[..., object] | None = None,
    solver_identity: SolverSourceIdentity | None = None,
) -> int:
    """Collect bounded official sources and publish only validated artifacts."""
    command_name = "collect-maniskill-pickcube"
    try:
        archive_dir, source_dir, summary_output = (
            _validate_new_pickcube_collection_outputs(
                args.runtime_archive_dir,
                args.source_dir,
                args.summary_output,
            )
        )
        binding = _load_trusted_pickcube_binding(
            args.expected_contract,
            args.compatibility_report,
        )
        checked_layout = load_maniskill_pickcube_action_layout(args.action_layout)
        validate_maniskill_pickcube_action_layout_binding(
            checked_layout,
            binding,
            checked_layout.m1_action_layout,
        )
        settings = environment_settings_from_compatibility(binding)
        if args.sim_backend != settings.sim_backend:
            raise ValueError("collection simulator backend differs from the probe")
        action_contract = action_contract_from_compatibility(
            binding,
            coordinate_frame=checked_layout.coordinate_frame,
        )
        task_keys = PickCubeTaskKeyContract.from_compatibility_report(binding.report)

        provided_solver_parts = (solver, solver_identity)
        if any(item is None for item in provided_solver_parts):
            if any(item is not None for item in provided_solver_parts):
                raise ValueError("solver and solver identity must be supplied together")
            solver, solver_identity = load_official_solver()
        assert solver is not None
        assert solver_identity is not None
        expected_solver = binding.report.source_solver
        if (
            solver_identity.module_name != expected_solver.module_name
            or f"sha256:{solver_identity.source_sha256}"
            != expected_solver.source_sha256
        ):
            raise ValueError("installed official solver differs from the probe report")

        runtime_factory = environment_factory or LazyManiSkillSourceEnvironmentFactory()
        try:
            result = collect_reference_archive(
                requested_success_count=args.requested_success_count,
                starting_seed=args.starting_seed,
                maximum_attempts=args.maximum_attempts,
                compatibility_identity=binding.report.compatibility_identity,
                environment_factory=runtime_factory,
                settings=settings,
                action_contract=action_contract,
                key_contract=task_keys,
                solver=solver,
                solver_identity=solver_identity,
                trajectory_action_limit=args.trajectory_action_limit,
            )
        except PickCubeSourceCollectionIncompleteError as error:
            incomplete = build_incomplete_collection_summary(
                error.result,
                compatibility_identity=binding.report.compatibility_identity,
            )
            write_sanitized_report(incomplete, summary_output)
            print(
                f"{command_name} failed: accepted={error.result.accepted_count}/"
                f"{error.result.requested_success_count}; "
                f"attempts={len(error.result.attempts)} summary_written=true",
                file=sys.stderr,
            )
            return 1
        summary = _publish_pickcube_collection(
            result,
            archive_dir=archive_dir,
            source_dir=source_dir,
            summary_output=summary_output,
        )
        print(format_collection_summary(summary))
        return 0
    except KeyboardInterrupt:
        print(
            f"{command_name} interrupted; no partial output accepted", file=sys.stderr
        )
        return 130
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            f"{command_name} failed: {sanitize_operational_text(error)}",
            file=sys.stderr,
        )
        return 1


def _validate_dataset_path_separation(input_dir: Path, output_dir: Path) -> None:
    """Reject any source/output ancestry overlap before a bundle can be changed."""
    source = input_dir.absolute().resolve()
    output = output_dir.absolute().resolve()
    if (
        source == output
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError(
            "input and output dataset directories must not equal, contain, "
            "or be contained by one another"
        )


def _validate_replay_path_separation(
    source_dir: Path, corruption_dir: Path, output_dir: Path
) -> None:
    """Reject equality or ancestry overlap among all replay dataset roots."""
    resolved = {
        "source": source_dir.absolute().resolve(),
        "corruption": corruption_dir.absolute().resolve(),
        "output": output_dir.absolute().resolve(),
    }
    items = tuple(resolved.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if (
                left == right
                or left.is_relative_to(right)
                or right.is_relative_to(left)
            ):
                raise ValueError(
                    "replay dataset directories must not equal, contain, or be "
                    f"contained by one another ({left_name}, {right_name})"
                )


def _validate_proposal_identifiers(dataset: CorruptionDataset) -> None:
    """Recompute every deterministic proposal identifier from its manifest data."""
    for proposal in dataset.proposals:
        expected = build_proposal_identifier(
            source_episode_id=proposal.source_episode_id,
            source_candidate_id=proposal.source_candidate_id,
            corruption_name=proposal.corruption_type,
            resolved_parameters=proposal.resolved_parameters,
            seed=proposal.seed,
            generation_ordinal=proposal.generation_ordinal,
        )
        if proposal.proposal_id != expected:
            raise RuntimeError(
                f"proposal identifier mismatch for ordinal "
                f"{proposal.generation_ordinal}: expected {expected}, "
                f"got {proposal.proposal_id}"
            )


def _run_corrupt_data(args: argparse.Namespace) -> int:
    try:
        _validate_dataset_path_separation(args.input_dir, args.output_dir)
        episodes = load_episodes(args.input_dir)
        validate_episodes(episodes)
        if args.episode_limit is not None:
            episodes = episodes[: args.episode_limit]
        plan = load_corruption_plan(args.config)
        result = generate_corruption_proposals(
            episodes,
            plan.action_layout,
            plan.corruptions,
            base_seed=args.seed,
            candidate_limit=args.candidate_limit,
            proposal_limit=args.proposal_limit,
            strict_applicability=args.strict_applicability,
        )
        dataset = CorruptionDataset(
            source_dataset_id=compute_episode_bundle_identifier(args.input_dir),
            action_layout=plan.action_layout,
            proposals=result.proposals,
        )
        _validate_proposal_identifiers(dataset)
        manifest_path = save_corruption_dataset(dataset, args.output_dir)
        reloaded = load_corruption_dataset(args.output_dir)
        _validate_proposal_identifiers(reloaded)
        if not _values_equal(dataset, reloaded):
            raise RuntimeError("corruption round-trip comparison found a changed value")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"corrupt-data failed: {error}", file=sys.stderr)
        return 1

    if args.audit:
        if result.skips:
            for skip in result.skips:
                print(f"corrupt-data audit: skipped {skip}")
        else:
            print("corrupt-data audit: no applicability skips")
    counts = ",".join(
        f"{name}:{count}" for name, count in result.corruption_counts.items()
    )
    print(
        "corrupt-data OK: "
        f"source_episodes={result.source_episode_count} "
        f"source_candidates={result.source_candidate_count} "
        f"proposals={len(result.proposals)} "
        f"skipped_non_applicable={len(result.skips)} "
        "unlabeled=true round-trip=verified "
        f"corruptions={counts or '<none>'} manifest={manifest_path}"
    )
    return 0


def _evaluation_launch_command(args: argparse.Namespace) -> tuple[str, ...]:
    """Build a complete command record for sanitization in the run manifest."""
    command = [
        "latentguard",
        "evaluate-data",
        "--corruption-dir",
        str(args.corruption_dir),
        "--output-dir",
        str(args.output_dir),
        "--evaluator",
        str(args.evaluator),
        "--config",
        str(args.config),
        "--seed",
        str(args.seed),
    ]
    if args.max_proposals is not None:
        command.extend(("--max-proposals", str(args.max_proposals)))
    for enabled, flag in (
        (args.resume, "--resume"),
        (args.fail_fast, "--fail-fast"),
        (args.retry_execution_errors, "--retry-execution-errors"),
        (args.audit, "--audit"),
    ):
        if enabled:
            command.append(flag)
    return tuple(command)


def _print_fixture_warning(evaluator_id: str) -> None:
    """State the built-in fixture's non-physical evidence boundary prominently."""
    if evaluator_id == "deterministic_fixture":
        print(
            "evaluate-data WARNING: deterministic_fixture produces synthetic weak "
            "non-simulator evidence only; it is not physically meaningful"
        )


def _run_evaluate_data(args: argparse.Namespace) -> int:
    try:
        if args.retry_execution_errors and not args.resume:
            raise ValueError("--retry-execution-errors requires --resume")
        _validate_dataset_path_separation(args.corruption_dir, args.output_dir)
        corruption_dataset = load_corruption_dataset(args.corruption_dir)
        source_digest = compute_corruption_dataset_digest(args.corruption_dir)
        configuration = load_evaluator_configuration(args.config)
        evaluator = create_evaluator(args.evaluator, configuration)
        plan = plan_evaluation(
            corruption_dataset,
            evaluator,
            source_corruption_dataset_digest=source_digest,
            base_seed=args.seed,
            max_proposals=args.max_proposals,
        )

        _print_fixture_warning(evaluator.evaluator_id)
        if args.dry_run:
            if args.resume:
                existing = load_evaluation_dataset(
                    args.output_dir,
                    corruption_dataset=corruption_dataset,
                    expected_corruption_digest=source_digest,
                )
                if existing.run_id != plan.run_id:
                    raise ValueError(
                        "dry-run resume inputs conflict with the persisted run"
                    )
            elif args.output_dir.exists():
                if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
                    raise ValueError("dry-run output directory must be absent or empty")
            if args.audit:
                for attempt in plan.attempts:
                    print(
                        "evaluate-data dry-run audit: "
                        f"proposal={attempt.proposal_id} "
                        f"attempt={attempt.attempt_ordinal} "
                        f"seed={attempt.evaluation_seed} "
                        f"evidence={attempt.evidence_id}"
                    )
            first_evidence = plan.attempts[0].evidence_id if plan.attempts else "<none>"
            last_evidence = plan.attempts[-1].evidence_id if plan.attempts else "<none>"
            print(
                "evaluate-data dry-run OK: "
                f"evaluator={plan.evaluator_id} "
                f"proposals={len(plan.selected_proposal_ids)} "
                f"planned_attempts={len(plan.attempts)} "
                f"run_id={plan.run_id} "
                f"first_evidence={first_evidence} "
                f"last_evidence={last_evidence} "
                "evaluations=0 output_created=false"
            )
            return 0

        result = run_evaluation(
            corruption_dataset,
            evaluator,
            source_corruption_dataset_digest=source_digest,
            output_dir=args.output_dir,
            base_seed=args.seed,
            max_proposals=args.max_proposals,
            resume=args.resume,
            fail_fast=args.fail_fast,
            retry_execution_errors=args.retry_execution_errors,
            launch_command=_evaluation_launch_command(args),
        )
        reloaded = load_evaluation_dataset(
            args.output_dir,
            corruption_dataset=corruption_dataset,
            expected_corruption_digest=source_digest,
        )
        if reloaded.run_id != result.dataset.run_id:
            raise RuntimeError("evaluation round-trip run identifier changed")
        if args.audit:
            for line in evaluation_audit_lines(reloaded):
                print(line)
        print(
            format_evaluation_summary(
                reloaded.summary,
                evaluator_id=reloaded.evaluator_id,
                resumed=result.resumed,
                evaluated_attempts=result.evaluated_attempts,
                recovered_attempts=result.recovered_attempts,
            )
        )
        if result.stopped_early or reloaded.summary.execution_error:
            return 1
        return 0
    except KeyboardInterrupt:
        print(
            "evaluate-data interrupted: persisted state may be resumed",
            file=sys.stderr,
        )
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            f"evaluate-data failed: {sanitize_operational_text(error)}",
            file=sys.stderr,
        )
        return 1


def _replay_launch_command(args: argparse.Namespace) -> tuple[str, ...]:
    """Build a complete replay command for conservative manifest sanitization."""
    command = [
        "latentguard",
        "replay-data",
        "--source-dir",
        str(args.source_dir),
        "--corruption-dir",
        str(args.corruption_dir),
        "--output-dir",
        str(args.output_dir),
        "--adapter",
        str(args.adapter),
        "--config",
        str(args.config),
        "--seed",
        str(args.seed),
    ]
    if args.max_proposals is not None:
        command.extend(("--max-proposals", str(args.max_proposals)))
    for enabled, flag in (
        (args.resume, "--resume"),
        (args.fail_fast, "--fail-fast"),
        (args.retry_execution_errors, "--retry-execution-errors"),
        (args.audit, "--audit"),
    ):
        if enabled:
            command.append(flag)
    return tuple(command)


def _pickcube_replay_launch_command(args: argparse.Namespace) -> tuple[str, ...]:
    """Build the complete specialized launch command for manifest sanitization."""
    command = [
        "latentguard",
        "replay-maniskill-pickcube",
        "--source-dir",
        str(args.source_dir),
        "--corruption-dir",
        str(args.corruption_dir),
        "--runtime-archive-dir",
        str(args.runtime_archive_dir),
        "--output-dir",
        str(args.output_dir),
        "--compatibility-report",
        str(args.compatibility_report),
        "--expected-contract",
        str(args.expected_contract),
        "--action-layout",
        str(args.action_layout),
        "--seed",
        str(args.seed),
    ]
    if args.max_proposals is not None:
        command.extend(("--max-proposals", str(args.max_proposals)))
    for enabled, flag in (
        (args.resume, "--resume"),
        (args.fail_fast, "--fail-fast"),
        (args.retry_execution_errors, "--retry-execution-errors"),
        (args.audit, "--audit"),
    ):
        if enabled:
            command.append(flag)
    return tuple(command)


def _print_replay_fixture_warning(adapter_id: str) -> None:
    """State the built-in replay fixture's non-physical boundary prominently."""
    if adapter_id == "deterministic_replay_fixture":
        print(
            "replay-data WARNING: deterministic_replay_fixture is not a simulator, "
            "is not physically meaningful, produces weak non-simulator evidence "
            "only, and must not support training, benchmark, or research claims"
        )


def _resume_without_rerun_count(
    args: argparse.Namespace,
    *,
    binding: ReplaySourceBinding,
) -> int:
    """Count selected terminal proposals that this resume will preserve as-is."""
    if not args.resume:
        return 0
    existing = load_evaluation_dataset(
        args.output_dir,
        corruption_dataset=binding.corruption_dataset,
        expected_corruption_digest=binding.corruption_dataset_digest,
    )
    selected = set(existing.selected_proposal_ids)
    preserved = 0
    for proposal_id in selected:
        attempts = [
            entry for entry in existing.ledger if entry.proposal_id == proposal_id
        ]
        if not attempts:
            continue
        latest = max(attempts, key=lambda entry: entry.attempt_ordinal)
        if latest.state in {LedgerState.PENDING, LedgerState.RUNNING}:
            continue
        if latest.state is LedgerState.EXECUTION_ERROR and args.retry_execution_errors:
            continue
        preserved += 1
    return preserved


def _execute_replay(
    args: argparse.Namespace,
    *,
    binding: ReplaySourceBinding,
    evaluator: ExactStatePairedReplayEvaluator,
    command_name: str,
    launch_command: tuple[str, ...],
) -> int:
    """Execute one generic or PickCube replay through the common M2A runner."""
    plan = plan_evaluation(
        binding.corruption_dataset,
        evaluator,
        source_corruption_dataset_digest=binding.corruption_dataset_digest,
        base_seed=args.seed,
        max_proposals=args.max_proposals,
    )
    _print_replay_fixture_warning(evaluator.adapter.adapter_id)

    if args.dry_run:
        if args.resume:
            existing = load_evaluation_dataset(
                args.output_dir,
                corruption_dataset=binding.corruption_dataset,
                expected_corruption_digest=binding.corruption_dataset_digest,
            )
            if existing.run_id != plan.run_id:
                raise ValueError(
                    "dry-run resume inputs conflict with the persisted run"
                )
        elif args.output_dir.exists():
            if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
                raise ValueError("dry-run output directory must be absent or empty")
        case_ids = {
            replay_case.proposal_id: replay_case.case_id
            for replay_case in evaluator.replay_bundle.replay_cases
        }
        if args.audit:
            for attempt in plan.attempts:
                print(
                    f"{command_name} dry-run audit: "
                    f"proposal={attempt.proposal_id} "
                    f"case={case_ids[attempt.proposal_id]} "
                    f"attempt={attempt.attempt_ordinal} "
                    f"seed={attempt.evaluation_seed} "
                    f"evidence={attempt.evidence_id}"
                )
        print(
            f"{command_name} dry-run OK: "
            f"adapter={evaluator.adapter.adapter_id} "
            f"adapter_trust_tier={evaluator.trust_descriptor.trust_tier.value} "
            f"source_episodes={binding.source_episode_count} "
            f"source_candidates={binding.source_candidate_count} "
            f"proposals={len(plan.selected_proposal_ids)} "
            f"planned_attempts={len(plan.attempts)} "
            f"run_id={plan.run_id} "
            f"replay_bundle={evaluator.replay_bundle.bundle_digest} "
            "evaluations=0 sessions=0 output_created=false"
        )
        return 0

    resumed_without_rerun = _resume_without_rerun_count(args, binding=binding)
    result = run_evaluation(
        binding.corruption_dataset,
        evaluator,
        source_corruption_dataset_digest=binding.corruption_dataset_digest,
        output_dir=args.output_dir,
        base_seed=args.seed,
        max_proposals=args.max_proposals,
        resume=args.resume,
        fail_fast=args.fail_fast,
        retry_execution_errors=args.retry_execution_errors,
        launch_command=launch_command,
    )
    binding.assert_unchanged()
    reloaded = load_evaluation_dataset(
        args.output_dir,
        corruption_dataset=binding.corruption_dataset,
        expected_corruption_digest=binding.corruption_dataset_digest,
    )
    if reloaded.run_id != result.dataset.run_id:
        raise RuntimeError("replay round-trip run identifier changed")
    if args.audit:
        for line in replay_audit_lines(reloaded):
            print(line.replace("replay-data", command_name, 1))
    summary = build_replay_summary(
        reloaded,
        source_episode_count=binding.source_episode_count,
        source_candidate_count=binding.source_candidate_count,
        adapter_trust_tier=evaluator.trust_descriptor.trust_tier.value,
        resumed_without_rerun_count=resumed_without_rerun,
    )
    print(
        format_replay_summary(summary, resumed=result.resumed).replace(
            "replay-data", command_name, 1
        )
    )
    if result.stopped_early or reloaded.summary.execution_error:
        return 1
    return 0


def _run_replay_data(args: argparse.Namespace) -> int:
    try:
        if args.retry_execution_errors and not args.resume:
            raise ValueError("--retry-execution-errors requires --resume")
        _validate_replay_path_separation(
            args.source_dir, args.corruption_dir, args.output_dir
        )
        binding = ReplaySourceBinding.from_paths(args.source_dir, args.corruption_dir)
        configuration = load_replay_adapter_configuration(args.config)
        evaluator = create_replay_evaluator(args.adapter, configuration, binding)
        return _execute_replay(
            args,
            binding=binding,
            evaluator=evaluator,
            command_name="replay-data",
            launch_command=_replay_launch_command(args),
        )
    except KeyboardInterrupt:
        print(
            "replay-data interrupted: persisted state may be resumed",
            file=sys.stderr,
        )
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            f"replay-data failed: {sanitize_operational_text(error)}",
            file=sys.stderr,
        )
        return 1


def _run_replay_maniskill_pickcube(args: argparse.Namespace) -> int:
    """Load all static PickCube artifacts and use the standard replay runner."""
    command_name = "replay-maniskill-pickcube"
    try:
        if args.retry_execution_errors and not args.resume:
            raise ValueError("--retry-execution-errors requires --resume")
        _validate_replay_path_separation(
            args.source_dir, args.corruption_dir, args.output_dir
        )
        binding = ReplaySourceBinding.from_paths(args.source_dir, args.corruption_dir)
        configuration: Mapping[str, object] = {
            "schema_version": "1.0",
            "runtime_archive_directory": str(args.runtime_archive_dir),
            "source_dataset_directory": str(args.source_dir),
            "corruption_dataset_directory": str(args.corruption_dir),
            "expected_contract_path": str(args.expected_contract),
            "compatibility_report_path": str(args.compatibility_report),
            "action_layout_path": str(args.action_layout),
        }
        evaluator = create_replay_evaluator(
            MANISKILL_PICKCUBE_ADAPTER_ID,
            configuration,
            binding,
        )
        return _execute_replay(
            args,
            binding=binding,
            evaluator=evaluator,
            command_name=command_name,
            launch_command=_pickcube_replay_launch_command(args),
        )
    except KeyboardInterrupt:
        print(
            f"{command_name} interrupted: persisted state may be resumed",
            file=sys.stderr,
        )
        return 130
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            f"{command_name} failed: {sanitize_operational_text(error)}",
            file=sys.stderr,
        )
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the LatentGuard-VLA command-line interface."""
    args = _build_parser().parse_args(argv)
    if args.command == "sanity-data":
        return _run_sanity_data(args)
    if args.command == "audit-data":
        return _run_audit_data(args)
    if args.command == "summarize-data":
        return _run_summarize_data(args)
    if args.command == "replay-episode":
        return _run_replay_episode(args)
    if args.command == "remote-sync":
        return _run_remote_sync(args)
    if args.command == "corrupt-data":
        return _run_corrupt_data(args)
    if args.command == "evaluate-data":
        return _run_evaluate_data(args)
    if args.command == "replay-data":
        return _run_replay_data(args)
    if args.command == "probe-maniskill-pickcube":
        return _run_probe_maniskill_pickcube(args)
    if args.command == "collect-maniskill-pickcube":
        return _run_collect_maniskill_pickcube(args)
    if args.command == "replay-maniskill-pickcube":
        return _run_replay_maniskill_pickcube(args)
    raise AssertionError(f"unhandled command: {args.command}")

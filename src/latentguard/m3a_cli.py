"""Bounded M3A state-indexed PickCube command-line workflows.

The functions in this module deliberately keep optional ManiSkill imports lazy.
Every tracked configuration is loaded locally; large state and dataset artifacts
are published only to caller-selected external directories.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from latentguard.action_verifier import (
    FULL_SPLIT_COUNTS,
    CandidateType,
    DatasetSplit,
    load_action_verifier_dataset,
    save_action_verifier_dataset,
)
from latentguard.corruptions.config import load_corruption_plan
from latentguard.corruptions.generation import generate_corruption_proposals
from latentguard.corruptions.serialization import (
    CorruptionDataset,
    load_corruption_dataset,
    save_corruption_dataset,
)
from latentguard.evaluation.runner import plan_evaluation, run_evaluation
from latentguard.evaluation.serialization import (
    LedgerState,
    RunState,
    compute_corruption_dataset_content_digest,
    load_evaluation_dataset,
)
from latentguard.integrations.maniskill_pickcube.action_verifier_export import (
    collect_action_verifier_evidence_ids,
    compute_action_verifier_evidence_dataset_digest,
    export_action_verifier_dataset_from_paths,
    validate_action_verifier_evidence_bindings,
    validate_serialized_action_verifier_export,
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
    ManiSkillPickCubeActionLayout,
    load_expected_contract,
    load_maniskill_pickcube_action_layout,
    validate_maniskill_pickcube_action_layout_binding,
)
from latentguard.integrations.maniskill_pickcube.reporting import (
    write_sanitized_report,
)
from latentguard.integrations.maniskill_pickcube.serialization import (
    action_contract_from_compatibility,
    environment_settings_from_compatibility,
)
from latentguard.integrations.maniskill_pickcube.session import (
    ManiSkillPickCubeEnvironmentSettings,
    PickCubeReplayActionContract,
)
from latentguard.integrations.maniskill_pickcube.source_generation import (
    LazyManiSkillSourceEnvironmentFactory,
    SolverSourceIdentity,
    SourceEnvironmentFactory,
    load_official_solver,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
    STATE_INDEXED_ARCHIVE_VERSION,
    PickCubeStateIndexedArchiveV1,
    load_state_indexed_archive,
    save_state_indexed_archive,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
    load_anchor_manifest,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_replay import (
    build_state_indexed_maniskill_pickcube_adapter,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_source import (
    StateIndexedSourceCollectionIncompleteError,
    collect_state_indexed_reference_archive,
    verify_all_indexed_states_fresh,
)
from latentguard.integrations.maniskill_pickcube.task_evidence import (
    PickCubeTaskKeyContract,
)
from latentguard.integrations.maniskill_pickcube.verifier_state import (
    PICKCUBE_VERIFIER_STATE_EXTRACTION_BOUNDARY,
    PICKCUBE_VERIFIER_STATE_SEMANTIC,
)
from latentguard.replay.evaluator import (
    create_exact_state_paired_replay_evaluator,
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
from latentguard.validation import validate_episodes

DEFAULT_EXPECTED_CONTRACT = Path(
    "configs/integrations/maniskill_pickcube/expected-contract-v1.json"
)
DEFAULT_M3A_CORRUPTION_CONFIG = Path(
    "configs/integrations/maniskill_pickcube/m3a-corruptions-v1.json"
)
M3A_COMMANDS = frozenset(
    {
        "collect-maniskill-pickcube-sequences",
        "build-state-indexed-pickcube",
        "replay-state-indexed-pickcube",
        "export-action-verifier-dataset",
        "validate-action-verifier-dataset",
    }
)


@dataclass(frozen=True, slots=True)
class _M3APickCubeScope:
    binding: CompatibilityBinding
    layout: ManiSkillPickCubeActionLayout
    settings: ManiSkillPickCubeEnvironmentSettings
    action_contract: PickCubeReplayActionContract
    task_keys: PickCubeTaskKeyContract


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


def add_m3a_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the five bounded M3A-Data commands."""
    collect = subparsers.add_parser(
        "collect-maniskill-pickcube-sequences",
        help="collect successful official trajectories with complete T+1 states",
    )
    collect.add_argument("--runtime-archive-dir", type=Path, required=True)
    collect.add_argument("--reference-archive-dir", type=Path, required=True)
    collect.add_argument("--compatibility-report", type=Path, required=True)
    collect.add_argument(
        "--expected-contract", type=Path, default=DEFAULT_EXPECTED_CONTRACT
    )
    collect.add_argument("--action-layout", type=Path, required=True)
    collect.add_argument("--summary-output", type=Path, required=True)
    collect.add_argument("--requested-success-count", type=_positive_int, default=6)
    collect.add_argument("--starting-seed", type=_nonnegative_int, default=0)
    collect.add_argument("--maximum-attempts", type=_positive_int, default=24)
    collect.add_argument("--sim-backend", choices=("gpu",), default="gpu")
    collect.add_argument("--trajectory-action-limit", type=_positive_int)
    collect.add_argument(
        "--fresh-state-trajectory-limit",
        type=_positive_int,
        default=1,
        help="number of accepted trajectories whose every state is rechecked",
    )
    collect.add_argument("--dry-run", action="store_true")

    build = subparsers.add_parser(
        "build-state-indexed-pickcube",
        help="select deterministic anchors and publish baseline-validated M0 sources",
    )
    build.add_argument("--runtime-archive-dir", type=Path, required=True)
    build.add_argument("--source-dir", type=Path, required=True)
    build.add_argument("--anchor-manifest-dir", type=Path, required=True)
    build.add_argument("--compatibility-report", type=Path, required=True)
    build.add_argument(
        "--expected-contract", type=Path, default=DEFAULT_EXPECTED_CONTRACT
    )
    build.add_argument("--action-layout", type=Path, required=True)
    build.add_argument("--summary-output", type=Path, required=True)
    build.add_argument(
        "--candidate-horizon", type=_positive_int, choices=(16,), default=16
    )
    build.add_argument(
        "--maximum-anchors-per-trajectory", type=_positive_int, default=6
    )
    build.add_argument("--trajectory-limit", type=_positive_int)
    build.add_argument("--dry-run", action="store_true")

    replay = subparsers.add_parser(
        "replay-state-indexed-pickcube",
        help="generate window corruptions and run exact paired state-indexed replay",
    )
    replay.add_argument("--source-dir", type=Path, required=True)
    replay.add_argument("--runtime-archive-dir", type=Path, required=True)
    replay.add_argument("--anchor-manifest-dir", type=Path, required=True)
    replay.add_argument(
        "--corruption-config", type=Path, default=DEFAULT_M3A_CORRUPTION_CONFIG
    )
    replay.add_argument("--corruption-dir", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument("--compatibility-report", type=Path, required=True)
    replay.add_argument(
        "--expected-contract", type=Path, default=DEFAULT_EXPECTED_CONTRACT
    )
    replay.add_argument("--action-layout", type=Path, required=True)
    replay.add_argument("--seed", type=_nonnegative_int, required=True)
    replay.add_argument("--max-proposals", type=_positive_int)
    replay.add_argument("--resume", action="store_true")
    replay.add_argument("--fail-fast", action="store_true")
    replay.add_argument("--retry-execution-errors", action="store_true")
    replay.add_argument("--audit", "--verbose", dest="audit", action="store_true")
    replay.add_argument("--summary-output", type=Path)
    replay.add_argument("--dry-run", action="store_true")

    export = subparsers.add_parser(
        "export-action-verifier-dataset",
        help="export compact model-ready samples from strong paired-replay evidence",
    )
    export.add_argument("--source-dir", type=Path, required=True)
    export.add_argument("--anchor-manifest-dir", type=Path, required=True)
    export.add_argument("--corruption-dir", type=Path, required=True)
    export.add_argument("--evidence-dir", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--summary-output", type=Path, required=True)
    export.add_argument("--split-seed", type=_nonnegative_int, default=0)
    export.add_argument("--train-trajectory-count", type=_positive_int, default=48)
    export.add_argument("--validation-trajectory-count", type=_positive_int, default=6)
    export.add_argument("--test-trajectory-count", type=_positive_int, default=6)
    export.add_argument("--dry-run", action="store_true")

    validate = subparsers.add_parser(
        "validate-action-verifier-dataset",
        help="independently reload the compact dataset and enforce acceptance gates",
    )
    validate.add_argument("--dataset-dir", type=Path, required=True)
    validate.add_argument("--anchor-manifest-dir", type=Path, required=True)
    validate.add_argument("--corruption-dir", type=Path, required=True)
    validate.add_argument("--evidence-dir", type=Path, required=True)
    validate.add_argument(
        "--resume-report",
        type=Path,
        help="summary from a no-op replay --resume invocation",
    )
    validate.add_argument("--report-output", type=Path, required=True)
    validate.add_argument("--require-full-target", action="store_true")
    validate.add_argument("--dry-run", action="store_true")


def run_m3a_command(args: argparse.Namespace) -> int | None:
    """Dispatch an M3A command, returning ``None`` for non-M3A commands."""
    if args.command not in M3A_COMMANDS:
        return None
    if args.command == "collect-maniskill-pickcube-sequences":
        return _run_collect_sequences(args)
    if args.command == "build-state-indexed-pickcube":
        return _run_build_state_indexed(args)
    if args.command == "replay-state-indexed-pickcube":
        return _run_replay_state_indexed(args)
    if args.command == "export-action-verifier-dataset":
        return _run_export_action_verifier(args)
    if args.command == "validate-action-verifier-dataset":
        return _run_validate_action_verifier(args)
    raise AssertionError(f"unhandled M3A command: {args.command}")


def _load_scope(args: argparse.Namespace) -> _M3APickCubeScope:
    expected = load_expected_contract(args.expected_contract)
    report = load_compatibility_report(args.compatibility_report)
    binding = validate_compatibility_report(report, expected, require_trusted=True)
    layout = load_maniskill_pickcube_action_layout(args.action_layout)
    validate_maniskill_pickcube_action_layout_binding(
        layout, binding, layout.m1_action_layout
    )
    settings = environment_settings_from_compatibility(binding)
    action_contract = action_contract_from_compatibility(
        binding, coordinate_frame=layout.coordinate_frame
    )
    return _M3APickCubeScope(
        binding=binding,
        layout=layout,
        settings=settings,
        action_contract=action_contract,
        task_keys=PickCubeTaskKeyContract.from_compatibility_report(binding.report),
    )


def _resolved(path: Path) -> Path:
    return path.absolute().resolve()


def _require_new_outputs(
    *, directories: Mapping[str, Path], files: Mapping[str, Path]
) -> tuple[dict[str, Path], dict[str, Path]]:
    resolved_dirs = {name: _resolved(path) for name, path in directories.items()}
    resolved_files = {name: _resolved(path) for name, path in files.items()}
    for name, path in (*resolved_dirs.items(), *resolved_files.items()):
        if path.exists() or path.is_symlink():
            raise ValueError(f"{name} output must not already exist")
    paths = tuple(resolved_dirs.items())
    for index, (left_name, left) in enumerate(paths):
        for right_name, right in paths[index + 1 :]:
            if (
                left == right
                or left.is_relative_to(right)
                or right.is_relative_to(left)
            ):
                raise ValueError(
                    f"output directories overlap: {left_name}, {right_name}"
                )
    for file_name, file_path in resolved_files.items():
        for directory_name, directory in resolved_dirs.items():
            if file_path.is_relative_to(directory):
                raise ValueError(f"{file_name} must be outside {directory_name} output")
    return resolved_dirs, resolved_files


def _require_input_output_separation(
    *,
    input_directories: Mapping[str, Path],
    output_directories: Mapping[str, Path],
    output_files: Mapping[str, Path],
) -> None:
    inputs = {name: _resolved(path) for name, path in input_directories.items()}
    outputs = {name: _resolved(path) for name, path in output_directories.items()}
    files = {name: _resolved(path) for name, path in output_files.items()}
    output_items = tuple(outputs.items())
    for index, (left_name, left_path) in enumerate(output_items):
        for right_name, right_path in output_items[index + 1 :]:
            if (
                left_path == right_path
                or left_path.is_relative_to(right_path)
                or right_path.is_relative_to(left_path)
            ):
                raise ValueError(
                    f"output directories overlap: {left_name}, {right_name}"
                )
    for input_name, input_path in inputs.items():
        for output_name, output_path in outputs.items():
            if (
                input_path == output_path
                or input_path.is_relative_to(output_path)
                or output_path.is_relative_to(input_path)
            ):
                raise ValueError(
                    f"input/output directories overlap: {input_name}, {output_name}"
                )
        for file_name, file_path in files.items():
            if file_path.is_relative_to(input_path):
                raise ValueError(f"{file_name} must be outside {input_name}")
    for file_name, file_path in files.items():
        for output_name, output_path in outputs.items():
            if file_path.is_relative_to(output_path):
                raise ValueError(f"{file_name} must be outside {output_name}")


def _remove_partial_outputs(
    *, directories: tuple[Path, ...] = (), files: tuple[Path, ...] = ()
) -> None:
    """Best-effort removal of outputs owned by one failed CLI invocation."""

    for path in directories:
        try:
            if path.exists() or path.is_symlink():
                shutil.rmtree(path)
        except OSError:
            pass
    for path in files:
        try:
            if path.exists() or path.is_symlink():
                path.unlink()
        except OSError:
            pass


def _validate_solver_identity(
    binding: CompatibilityBinding, identity: SolverSourceIdentity
) -> None:
    expected = binding.report.source_solver
    if (
        identity.module_name != expected.module_name
        or f"sha256:{identity.source_sha256}" != expected.source_sha256
    ):
        raise ValueError("installed official solver differs from compatibility report")


def _attempt_failure_categories(result: Any) -> dict[str, int]:
    """Return deterministic non-secret source-attempt failure counts."""
    return dict(
        sorted(
            Counter(
                item.failure_category
                for item in result.attempts
                if item.failure_category is not None
            ).items()
        )
    )


def _restoration_error_distribution(values: list[float]) -> Mapping[str, object]:
    """Summarize finite restoration errors with fixed bins and nearest ranks."""

    errors = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in errors):
        raise ValueError(
            "restoration error distribution requires finite non-negative values"
        )

    def nearest_rank(fraction: float) -> float | None:
        if not errors:
            return None
        index = max(0, math.ceil(fraction * len(errors)) - 1)
        return errors[index]

    bins = {
        "equal_zero": 0,
        "gt_zero_le_1e-8": 0,
        "gt_1e-8_le_1e-7": 0,
        "gt_1e-7_le_5e-7": 0,
        "gt_5e-7_le_1e-6": 0,
        "gt_1e-6": 0,
    }
    for error in errors:
        if error == 0.0:
            bins["equal_zero"] += 1
        elif error <= 1e-8:
            bins["gt_zero_le_1e-8"] += 1
        elif error <= 1e-7:
            bins["gt_1e-8_le_1e-7"] += 1
        elif error <= 5e-7:
            bins["gt_1e-7_le_5e-7"] += 1
        elif error <= 1e-6:
            bins["gt_5e-7_le_1e-6"] += 1
        else:
            bins["gt_1e-6"] += 1
    return {
        "count": len(errors),
        "fixed_bin_counts": bins,
        "maximum": None if not errors else errors[-1],
        "minimum": None if not errors else errors[0],
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "p99": nearest_rank(0.99),
        "quantile_semantic": "nearest_rank_v1",
    }


def _collection_summary(
    *,
    result: Any,
    archive: PickCubeStateIndexedArchiveV1,
    audit_records: tuple[Any, ...],
) -> Mapping[str, object]:
    state_count = sum(len(episode.states) for episode in archive.episodes)
    state_errors = [float(record.maximum_absolute_error) for record in audit_records]
    maximum_error = max(state_errors, default=0.0)
    component_counts = sorted(
        {int(record.compared_component_count) for record in audit_records}
    )
    verifier_errors = [
        float(record.verifier_maximum_absolute_error) for record in audit_records
    ]
    verifier_maximum_error = max(verifier_errors, default=0.0)
    verifier_component_counts = sorted(
        {int(record.verifier_component_count) for record in audit_records}
    )
    task_mismatch_counts = Counter(
        field
        for record in audit_records
        for field in record.source_to_restored_task_mismatch_fields
    )
    return {
        "accepted_source_trajectories": result.accepted_count,
        "archive_content_digest": archive.content_digest,
        "attempt_count": len(result.attempts),
        "attempt_failure_categories": _attempt_failure_categories(result),
        "compatibility_identity": archive.episodes[0].compatibility_identity,
        "fresh_state_compared_component_counts": component_counts,
        "fresh_state_maximum_absolute_error": maximum_error,
        "fresh_state_restoration_error_distribution": (
            _restoration_error_distribution(state_errors)
        ),
        "fresh_state_verification_count": len(audit_records),
        "fresh_verifier_state_compared_component_counts": (verifier_component_counts),
        "fresh_verifier_state_maximum_absolute_error": verifier_maximum_error,
        "fresh_verifier_state_restoration_error_distribution": (
            _restoration_error_distribution(verifier_errors)
        ),
        "source_to_restored_task_mismatch_field_counts": dict(
            sorted(task_mismatch_counts.items())
        ),
        "source_to_restored_task_mismatch_state_count": sum(
            bool(record.source_to_restored_task_mismatch_fields)
            for record in audit_records
        ),
        "requested_source_trajectories": result.requested_success_count,
        "schema_version": "1.1",
        "source_action_count": sum(
            int(episode.source_actions.shape[0]) for episode in archive.episodes
        ),
        "state_indexed_archive_serialization_version": (STATE_INDEXED_ARCHIVE_VERSION),
        "t_plus_one_state_count": state_count,
        "verifier_state_extraction_boundary": (
            PICKCUBE_VERIFIER_STATE_EXTRACTION_BOUNDARY
        ),
        "verifier_state_semantic": PICKCUBE_VERIFIER_STATE_SEMANTIC,
    }


def _run_collect_sequences(
    args: argparse.Namespace,
    *,
    environment_factory: SourceEnvironmentFactory | None = None,
    solver: Callable[..., object] | None = None,
    solver_identity: SolverSourceIdentity | None = None,
) -> int:
    command = "collect-maniskill-pickcube-sequences"
    outputs_validated = False
    try:
        if args.requested_success_count > args.maximum_attempts:
            raise ValueError("requested successes cannot exceed maximum attempts")
        if args.dry_run:
            print(
                f"{command} dry-run OK: requested={args.requested_success_count} "
                f"maximum_attempts={args.maximum_attempts} "
                f"fresh_state_trajectories={args.fresh_state_trajectory_limit} "
                "simulator_runs=0 output_created=false"
            )
            return 0
        outputs, files = _require_new_outputs(
            directories={
                "state-indexed runtime archive": args.runtime_archive_dir,
                "M2C reference archive": args.reference_archive_dir,
            },
            files={"collection summary": args.summary_output},
        )
        outputs_validated = True
        scope = _load_scope(args)
        if args.sim_backend != scope.settings.sim_backend:
            raise ValueError("collection simulator backend differs from compatibility")
        if solver is None or solver_identity is None:
            if solver is not None or solver_identity is not None:
                raise ValueError("solver and solver identity must be supplied together")
            solver, solver_identity = load_official_solver()
        assert solver is not None
        assert solver_identity is not None
        _validate_solver_identity(scope.binding, solver_identity)
        factory = environment_factory or LazyManiSkillSourceEnvironmentFactory()
        result = collect_state_indexed_reference_archive(
            requested_success_count=args.requested_success_count,
            starting_seed=args.starting_seed,
            maximum_attempts=args.maximum_attempts,
            compatibility_identity=scope.binding.report.compatibility_identity,
            environment_factory=factory,
            settings=scope.settings,
            action_contract=scope.action_contract,
            key_contract=scope.task_keys,
            solver=solver,
            solver_identity=solver_identity,
            trajectory_action_limit=args.trajectory_action_limit,
        )
        archive = result.archive
        reference_archive = result.reference_archive
        if archive is None or reference_archive is None:
            raise RuntimeError("complete sequence collection returned no archives")
        audit: list[Any] = []
        for episode in archive.episodes[: args.fresh_state_trajectory_limit]:
            audit.extend(
                verify_all_indexed_states_fresh(
                    episode,
                    environment_factory=factory,
                    settings=scope.settings,
                    action_contract=scope.action_contract,
                    key_contract=scope.task_keys,
                )
            )
        save_state_indexed_archive(archive, outputs["state-indexed runtime archive"])
        save_reference_archive(reference_archive, outputs["M2C reference archive"])
        if (
            load_state_indexed_archive(
                outputs["state-indexed runtime archive"]
            ).content_digest
            != archive.content_digest
        ):
            raise RuntimeError("state-indexed archive digest changed after reload")
        if (
            load_reference_archive(outputs["M2C reference archive"]).content_digest
            != reference_archive.content_digest
        ):
            raise RuntimeError("reference archive digest changed after reload")
        summary = _collection_summary(
            result=result, archive=archive, audit_records=tuple(audit)
        )
        write_sanitized_report(summary, files["collection summary"])
        print(
            f"{command} OK: accepted={result.accepted_count} "
            f"attempts={len(result.attempts)} "
            f"states={summary['t_plus_one_state_count']} "
            f"fresh_verified={len(audit)} digest={archive.content_digest}"
        )
        return 0
    except StateIndexedSourceCollectionIncompleteError as error:
        incomplete_summary: Mapping[str, object] = {
            "accepted_source_trajectories": error.result.accepted_count,
            "attempt_count": len(error.result.attempts),
            "attempt_failure_categories": _attempt_failure_categories(error.result),
            "requested_source_trajectories": error.result.requested_success_count,
            "schema_version": "1.1",
            "status": "incomplete",
        }
        summary_written = False
        try:
            if not args.summary_output.exists():
                write_sanitized_report(incomplete_summary, args.summary_output)
                summary_written = True
        except OSError:
            pass
        print(
            f"{command} failed: accepted={error.result.accepted_count}/"
            f"{error.result.requested_success_count} "
            f"summary_written={str(summary_written).lower()}",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        if outputs_validated:
            _remove_partial_outputs(
                directories=(args.runtime_archive_dir, args.reference_archive_dir),
                files=(args.summary_output,),
            )
        print(f"{command} interrupted; no partial output accepted", file=sys.stderr)
        return 130
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        if outputs_validated:
            _remove_partial_outputs(
                directories=(args.runtime_archive_dir, args.reference_archive_dir),
                files=(args.summary_output,),
            )
        print(f"{command} failed: {error}", file=sys.stderr)
        return 1


def _run_build_state_indexed(args: argparse.Namespace) -> int:
    command = "build-state-indexed-pickcube"
    if args.dry_run:
        print(
            f"{command} dry-run OK: candidate_horizon={args.candidate_horizon} "
            f"maximum_anchors={args.maximum_anchors_per_trajectory} "
            "baseline_replays=0 output_created=false"
        )
        return 0
    outputs_validated = False
    try:
        # Imported lazily while the production validator remains optional-runtime safe.
        from latentguard.integrations.maniskill_pickcube.state_indexed_baseline import (
            ManiSkillPickCubeAnchorBaselineValidator,
        )
        from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
            PickCubeActionControlContractV1,
            build_state_indexed_anchor_sources,
            load_anchor_manifest,
            save_anchor_manifest,
        )

        outputs, files = _require_new_outputs(
            directories={
                "anchor M0 source dataset": args.source_dir,
                "anchor manifest": args.anchor_manifest_dir,
            },
            files={"anchor summary": args.summary_output},
        )
        _require_input_output_separation(
            input_directories={"T+1 runtime archive": args.runtime_archive_dir},
            output_directories={
                "anchor M0 source dataset": args.source_dir,
                "anchor manifest": args.anchor_manifest_dir,
            },
            output_files={"anchor summary": args.summary_output},
        )
        outputs_validated = True
        scope = _load_scope(args)
        archive = load_state_indexed_archive(args.runtime_archive_dir)
        if (
            archive.episodes[0].compatibility_identity
            != scope.binding.report.compatibility_identity
        ):
            raise ValueError("state-indexed archive compatibility identity mismatch")
        recorded_dtype = archive.episodes[0].source_actions.dtype.str
        recorded_dimension = int(archive.episodes[0].source_actions.shape[1])
        if any(
            episode.source_actions.dtype.str != recorded_dtype
            or int(episode.source_actions.shape[1]) != recorded_dimension
            for episode in archive.episodes
        ):
            raise ValueError("state-indexed archive action contracts are inconsistent")
        if recorded_dimension != scope.action_contract.total_dimension:
            raise ValueError("state-indexed archive action dimension mismatch")
        control = PickCubeActionControlContractV1(
            coordinate_frame=scope.action_contract.coordinate_frame,
            control_period_s=scope.action_contract.control_period_s,
            action_dtype=recorded_dtype,
            action_dimension=recorded_dimension,
        )
        factory = LazyManiSkillSourceEnvironmentFactory()
        validator = ManiSkillPickCubeAnchorBaselineValidator(
            environment_factory=factory,
            settings=scope.settings,
            runtime_action_contract=scope.action_contract,
            task_key_contract=scope.task_keys,
        )
        result = build_state_indexed_anchor_sources(
            archive,
            action_control_contract=control,
            baseline_validator=validator,
            candidate_horizon=args.candidate_horizon,
            maximum_anchors_per_trajectory=args.maximum_anchors_per_trajectory,
            source_trajectory_limit=args.trajectory_limit,
        )
        save_episodes(result.source_episodes, outputs["anchor M0 source dataset"])
        save_anchor_manifest(result.manifest, outputs["anchor manifest"])
        episodes = load_episodes(outputs["anchor M0 source dataset"])
        validate_episodes(episodes)
        reloaded_manifest = load_anchor_manifest(outputs["anchor manifest"])
        if reloaded_manifest.content_digest != result.manifest.content_digest:
            raise RuntimeError("anchor manifest digest changed after reload")
        reasons = Counter(
            record.anchor.selection_reason for record in result.manifest.records
        )
        summary: Mapping[str, object] = {
            "anchor_count": len(result.manifest.records),
            "anchor_manifest_content_digest": result.manifest.content_digest,
            "anchor_selection_reasons": dict(sorted(reasons.items())),
            "baseline_success_count": len(result.manifest.baseline_evidence),
            "baseline_exclusion_reasons": _baseline_exclusion_counts(result.manifest),
            "candidate_horizon": result.manifest.candidate_horizon,
            "exclusion_count": len(result.manifest.exclusions),
            "schema_version": "1.0",
            "source_dataset_digest": compute_episode_bundle_identifier(
                outputs["anchor M0 source dataset"]
            ),
            "source_trajectory_count": len(
                {
                    record.anchor.source_trajectory_id
                    for record in result.manifest.records
                }
            ),
        }
        write_sanitized_report(summary, files["anchor summary"])
        print(
            f"{command} OK: anchors={len(result.manifest.records)} "
            f"baselines={len(result.manifest.baseline_evidence)} "
            f"excluded={len(result.manifest.exclusions)} "
            f"digest={result.manifest.content_digest}"
        )
        return 0
    except KeyboardInterrupt:
        if outputs_validated:
            _remove_partial_outputs(
                directories=(args.source_dir, args.anchor_manifest_dir),
                files=(args.summary_output,),
            )
        print(f"{command} interrupted; no partial output accepted", file=sys.stderr)
        return 130
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        if outputs_validated:
            _remove_partial_outputs(
                directories=(args.source_dir, args.anchor_manifest_dir),
                files=(args.summary_output,),
            )
        print(f"{command} failed: {error}", file=sys.stderr)
        return 1


def _replay_launch_command(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "latentguard",
        "replay-state-indexed-pickcube",
        "--source-dir",
        str(args.source_dir),
        "--runtime-archive-dir",
        str(args.runtime_archive_dir),
        "--anchor-manifest-dir",
        str(args.anchor_manifest_dir),
        "--corruption-config",
        str(args.corruption_config),
        "--corruption-dir",
        str(args.corruption_dir),
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


def _resume_preserved_count(
    args: argparse.Namespace, binding: ReplaySourceBinding
) -> int:
    if not args.resume:
        return 0
    existing = load_evaluation_dataset(
        args.output_dir,
        corruption_dataset=binding.corruption_dataset,
        expected_corruption_digest=binding.corruption_dataset_digest,
    )
    preserved = 0
    for proposal_id in existing.selected_proposal_ids:
        attempts = [item for item in existing.ledger if item.proposal_id == proposal_id]
        if not attempts:
            continue
        latest = max(attempts, key=lambda item: item.attempt_ordinal)
        if latest.state in {LedgerState.PENDING, LedgerState.RUNNING}:
            continue
        if latest.state is LedgerState.EXECUTION_ERROR and args.retry_execution_errors:
            continue
        preserved += 1
    return preserved


def _validate_anchor_replay_binding(
    *,
    args: argparse.Namespace,
    binding: ReplaySourceBinding,
    archive: PickCubeStateIndexedArchiveV1,
) -> str:
    manifest = load_anchor_manifest(args.anchor_manifest_dir)
    if manifest.source_archive_content_digest != archive.content_digest:
        raise ValueError("anchor manifest references a different T+1 archive")
    if manifest.candidate_horizon != 16:
        raise ValueError("M3A replay requires fixed candidate horizon H=16")
    if tuple(record.source_episode_id for record in manifest.records) != tuple(
        episode.episode_id for episode in binding.episodes
    ):
        raise ValueError("anchor manifest/source episode ordering mismatch")
    candidates = tuple(
        episode.candidates[0].candidate_id for episode in binding.episodes
    )
    if tuple(record.source_candidate_id for record in manifest.records) != candidates:
        raise ValueError("anchor manifest/source candidate identity mismatch")
    return manifest.content_digest


def _build_replay_report(
    *,
    dataset: Any,
    replay_summary: Any,
    evaluated_attempts: int,
    recovered_attempts: int,
    retried_attempts: int,
    resumed: bool,
    anchor_manifest_digest: str,
    corruption_dataset_digest: str,
) -> Mapping[str, object]:
    errors: list[float] = []
    component_counts: set[int] = set()
    prefix_count = 0
    for evidence in dataset.evidence:
        for role in ("baseline", "corrupted"):
            error = evidence.metrics.get(
                f"replay_{role}_restoration_maximum_absolute_error"
            )
            count = evidence.metrics.get(
                f"replay_{role}_restoration_compared_component_count"
            )
            if type(error) in (int, float):
                errors.append(float(error))
            if type(count) is int:
                component_counts.add(count)
        if (
            evidence.metrics.get("pickcube_prefix_evaluated_before_continuation")
            is True
        ):
            prefix_count += 1
    payload: dict[str, object] = dict(asdict(replay_summary))
    payload.update(
        {
            "anchor_manifest_content_digest": anchor_manifest_digest,
            "corruption_dataset_digest": corruption_dataset_digest,
            "evaluated_attempts_this_invocation": evaluated_attempts,
            "evidence_count": len(dataset.evidence),
            "prefix_evidence_count": prefix_count,
            "recovered_attempts_this_invocation": recovered_attempts,
            "restoration_compared_component_counts": sorted(component_counts),
            "restoration_error_distribution": _restoration_error_distribution(errors),
            "restoration_maximum_absolute_error": max(errors, default=0.0),
            "retried_attempts_this_invocation": retried_attempts,
            "resumed_invocation": resumed,
            "run_id": dataset.run_id,
            "schema_version": "1.0",
        }
    )
    return payload


def _run_replay_state_indexed(args: argparse.Namespace) -> int:
    command = "replay-state-indexed-pickcube"
    if args.dry_run:
        print(
            f"{command} dry-run OK: max_proposals={args.max_proposals or 'all'} "
            f"resume={str(args.resume).lower()} evaluations=0 output_created=false"
        )
        return 0
    try:
        if args.retry_execution_errors and not args.resume:
            raise ValueError("--retry-execution-errors requires --resume")
        if args.summary_output is not None and (
            args.summary_output.exists() or args.summary_output.is_symlink()
        ):
            raise ValueError("replay summary output must be absent")
        output_files = (
            {}
            if args.summary_output is None
            else {"replay summary": args.summary_output}
        )
        _require_input_output_separation(
            input_directories={
                "anchor source dataset": args.source_dir,
                "T+1 runtime archive": args.runtime_archive_dir,
                "anchor manifest": args.anchor_manifest_dir,
            },
            output_directories={
                "corruption dataset": args.corruption_dir,
                "evidence dataset": args.output_dir,
            },
            output_files=output_files,
        )
        scope = _load_scope(args)
        archive = load_state_indexed_archive(args.runtime_archive_dir)
        plan = load_corruption_plan(args.corruption_config)
        validate_maniskill_pickcube_action_layout_binding(
            scope.layout, scope.binding, plan.action_layout
        )
        if args.resume:
            if not args.corruption_dir.is_dir():
                raise ValueError("resume requires an existing corruption directory")
            if args.output_dir.exists() and not args.output_dir.is_dir():
                raise ValueError("resume evidence output must be a directory")
            load_corruption_dataset(args.corruption_dir)
        else:
            if args.corruption_dir.exists() or args.corruption_dir.is_symlink():
                raise ValueError("corruption output must be absent on the first run")
            if args.output_dir.exists() or args.output_dir.is_symlink():
                raise ValueError("evidence output must be absent on the first run")
            episodes = load_episodes(args.source_dir)
            validate_episodes(episodes)
            generated = generate_corruption_proposals(
                episodes,
                plan.action_layout,
                plan.corruptions,
                base_seed=args.seed,
                strict_applicability=True,
            )
            if generated.skips:
                raise RuntimeError(
                    "M3A corruption generation produced applicability skips"
                )
            corruption_dataset = CorruptionDataset(
                source_dataset_id=compute_episode_bundle_identifier(args.source_dir),
                action_layout=plan.action_layout,
                proposals=generated.proposals,
            )
            save_corruption_dataset(corruption_dataset, args.corruption_dir)
            reloaded_corruptions = load_corruption_dataset(args.corruption_dir)
            if len(reloaded_corruptions.proposals) != len(corruption_dataset.proposals):
                raise RuntimeError("corruption proposal count changed after reload")
        binding = ReplaySourceBinding.from_paths(args.source_dir, args.corruption_dir)
        anchor_manifest_digest = _validate_anchor_replay_binding(
            args=args, binding=binding, archive=archive
        )
        adapter = build_state_indexed_maniskill_pickcube_adapter(
            source_binding=binding,
            archive_dir=args.runtime_archive_dir,
            compatibility_binding=scope.binding,
            action_layout_digest=scope.layout.action_layout_digest,
            coordinate_frame=scope.layout.coordinate_frame,
            candidate_horizon=16,
        )
        evaluator = create_exact_state_paired_replay_evaluator(
            adapter, adapter.replay_bundle
        )
        plan_evaluation(
            binding.corruption_dataset,
            evaluator,
            source_corruption_dataset_digest=binding.corruption_dataset_digest,
            base_seed=args.seed,
            max_proposals=args.max_proposals,
        )
        resume_evaluation = bool(args.resume and args.output_dir.is_dir())
        preserved = _resume_preserved_count(args, binding) if resume_evaluation else 0
        result = run_evaluation(
            binding.corruption_dataset,
            evaluator,
            source_corruption_dataset_digest=binding.corruption_dataset_digest,
            output_dir=args.output_dir,
            base_seed=args.seed,
            max_proposals=args.max_proposals,
            resume=resume_evaluation,
            fail_fast=args.fail_fast,
            retry_execution_errors=args.retry_execution_errors,
            launch_command=_replay_launch_command(args),
        )
        binding.assert_unchanged()
        if (
            load_state_indexed_archive(args.runtime_archive_dir).content_digest
            != archive.content_digest
        ):
            raise RuntimeError("state-indexed archive changed during replay")
        reloaded = load_evaluation_dataset(
            args.output_dir,
            corruption_dataset=binding.corruption_dataset,
            expected_corruption_digest=binding.corruption_dataset_digest,
        )
        if reloaded.run_id != result.dataset.run_id:
            raise RuntimeError("replay run identity changed after reload")
        if args.audit:
            for line in replay_audit_lines(reloaded):
                print(line.replace("replay-data", command, 1))
        replay_summary = build_replay_summary(
            reloaded,
            source_episode_count=binding.source_episode_count,
            source_candidate_count=binding.source_candidate_count,
            adapter_trust_tier=evaluator.trust_descriptor.trust_tier.value,
            resumed_without_rerun_count=preserved,
        )
        if args.summary_output is not None:
            report = _build_replay_report(
                dataset=reloaded,
                replay_summary=replay_summary,
                evaluated_attempts=result.evaluated_attempts,
                recovered_attempts=result.recovered_attempts,
                retried_attempts=result.retried_attempts,
                resumed=result.resumed,
                anchor_manifest_digest=anchor_manifest_digest,
                corruption_dataset_digest=binding.corruption_dataset_digest,
            )
            write_sanitized_report(report, args.summary_output)
        print(
            format_replay_summary(replay_summary, resumed=result.resumed).replace(
                "replay-data", command, 1
            )
        )
        if result.stopped_early or reloaded.summary.execution_error:
            return 1
        return 0
    except KeyboardInterrupt:
        print(f"{command} interrupted: persisted state may be resumed", file=sys.stderr)
        return 130
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"{command} failed: {error}", file=sys.stderr)
        return 1


def _run_export_action_verifier(args: argparse.Namespace) -> int:
    command = "export-action-verifier-dataset"
    if args.dry_run:
        total = (
            args.train_trajectory_count
            + args.validation_trajectory_count
            + args.test_trajectory_count
        )
        print(
            f"{command} dry-run OK: trajectories={total} "
            f"split={args.train_trajectory_count}/"
            f"{args.validation_trajectory_count}/{args.test_trajectory_count} "
            "samples=0 output_created=false"
        )
        return 0
    outputs_validated = False
    try:
        outputs, files = _require_new_outputs(
            directories={"action-verifier dataset": args.output_dir},
            files={"action-verifier summary": args.summary_output},
        )
        _require_input_output_separation(
            input_directories={
                "anchor source dataset": args.source_dir,
                "anchor manifest": args.anchor_manifest_dir,
                "corruption dataset": args.corruption_dir,
                "evidence dataset": args.evidence_dir,
            },
            output_directories={"action-verifier dataset": args.output_dir},
            output_files={"action-verifier summary": args.summary_output},
        )
        outputs_validated = True
        split_counts = {
            DatasetSplit.TRAIN: args.train_trajectory_count,
            DatasetSplit.VALIDATION: args.validation_trajectory_count,
            DatasetSplit.TEST: args.test_trajectory_count,
        }
        result = export_action_verifier_dataset_from_paths(
            anchor_manifest_dir=args.anchor_manifest_dir,
            source_episode_dir=args.source_dir,
            corruption_dataset_dir=args.corruption_dir,
            evaluation_dataset_dir=args.evidence_dir,
            split_counts=split_counts,
            split_seed=args.split_seed,
        )
        save_action_verifier_dataset(result.dataset, outputs["action-verifier dataset"])
        reload_report = validate_serialized_action_verifier_export(
            outputs["action-verifier dataset"],
            available_evidence_ids=result.available_evidence_ids,
            evidence_dataset_digest=result.evidence_dataset_digest,
            expected_split_counts=split_counts,
        )
        reloaded = load_action_verifier_dataset(outputs["action-verifier dataset"])
        if reloaded.content_digest != result.dataset.content_digest:
            raise RuntimeError("action-verifier dataset digest changed after reload")
        summary = dict(result.summary.as_mapping())
        summary["serialized_reload_validation"] = dict(reload_report.as_mapping())
        write_sanitized_report(summary, files["action-verifier summary"])
        print(
            f"{command} OK: trajectories={result.summary.source_trajectory_count} "
            f"anchors={result.summary.included_anchor_count} "
            f"samples={result.summary.dataset_sample_count} "
            f"corrupted={result.summary.corrupted_sample_count} "
            f"success={result.summary.corrupted_success_count} "
            f"failure={result.summary.corrupted_failure_count} "
            f"digest={result.dataset.content_digest}"
        )
        return 0
    except KeyboardInterrupt:
        if outputs_validated:
            _remove_partial_outputs(
                directories=(args.output_dir,), files=(args.summary_output,)
            )
        print(f"{command} interrupted; no partial output accepted", file=sys.stderr)
        return 130
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        if outputs_validated:
            _remove_partial_outputs(
                directories=(args.output_dir,), files=(args.summary_output,)
            )
        print(f"{command} failed: {error}", file=sys.stderr)
        return 1


def _strict_metric_int(evidence: Any, name: str, *, minimum: int = 0) -> int:
    value = evidence.metrics.get(name)
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"evidence {evidence.evidence_id} metric {name} must be an integer "
            f"at least {minimum}"
        )
    return value


def _strict_metric_number(evidence: Any, name: str) -> float:
    value = evidence.metrics.get(name)
    if type(value) not in (int, float):
        raise ValueError(
            f"evidence {evidence.evidence_id} metric {name} must be numeric"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"evidence {evidence.evidence_id} metric {name} must be finite"
        )
    return result


def _restoration_report(
    dataset: Any,
    evaluation_dataset: Any,
    anchor_manifest: Any,
    corruption_dataset: CorruptionDataset,
) -> tuple[
    list[int],
    float,
    int,
    list[int],
    float,
    Mapping[str, object],
    Mapping[str, object],
]:
    """Fail closed while summarizing every baseline and exported corruption."""

    baseline_by_id = {
        evidence.evidence_id: evidence for evidence in anchor_manifest.baseline_evidence
    }
    evaluation_by_id = {
        evidence.evidence_id: evidence for evidence in evaluation_dataset.evidence
    }
    proposal_by_id = {
        proposal.proposal_id: proposal for proposal in corruption_dataset.proposals
    }
    group_by_anchor = {group.anchor_id: group for group in dataset.candidate_groups}
    component_counts: set[int] = set()
    errors: list[float] = []
    verifier_component_counts: set[int] = set()
    verifier_errors: list[float] = []

    for group in dataset.candidate_groups:
        baseline = baseline_by_id.get(group.baseline_evidence_id)
        if baseline is None:
            raise ValueError(f"anchor {group.anchor_id} baseline evidence is missing")
        if (
            not baseline.complete_action_execution
            or baseline.executed_action_count != baseline.expected_action_count
            or not baseline.state_restoration_verified
            or not baseline.complete_state_comparison
            or not baseline.official_terminal_success
            or not baseline.simulator_replay_verified
            or baseline.compared_component_count <= 0
            or baseline.maximum_absolute_error < 0.0
            or baseline.maximum_absolute_error > 1e-6
            or baseline.comparison_semantic != "tolerance_verified_full_state_v1"
            or baseline.comparison_tolerance != 1e-6
            or not baseline.verifier_state_restoration_verified
            or baseline.verifier_state_compared_component_count
            != dataset.state_vector_dimension
            or baseline.verifier_state_maximum_absolute_error < 0.0
            or baseline.verifier_state_maximum_absolute_error > 1e-6
        ):
            raise ValueError(f"anchor {group.anchor_id} baseline is not fully verified")
        component_counts.add(baseline.compared_component_count)
        errors.append(float(baseline.maximum_absolute_error))
        verifier_component_counts.add(baseline.verifier_state_compared_component_count)
        verifier_errors.append(float(baseline.verifier_state_maximum_absolute_error))

    corrupted_samples = tuple(
        sample
        for sample in dataset.samples
        if sample.candidate_type is CandidateType.CORRUPTED
    )
    if not corrupted_samples:
        raise ValueError("dataset contains no corrupted samples to verify")
    for sample in corrupted_samples:
        evidence = evaluation_by_id.get(sample.strong_simulator_evidence_id)
        if evidence is None:
            raise ValueError(f"sample {sample.sample_id} evidence is missing")
        if evidence.proposal_id != sample.proposal_id:
            raise ValueError(f"sample {sample.sample_id} evidence proposal mismatch")
        proposal = proposal_by_id.get(sample.proposal_id)
        if proposal is None:
            raise ValueError(f"sample {sample.sample_id} proposal is missing")
        expected_steps = int(proposal.transformed_action.actions.shape[0])
        for name in (
            "replay_baseline_requested_steps",
            "replay_baseline_steps",
            "replay_corrupted_requested_steps",
            "replay_corrupted_steps",
        ):
            if _strict_metric_int(evidence, name) != expected_steps:
                raise ValueError(
                    f"evidence {evidence.evidence_id} does not prove complete execution"
                )
        group = group_by_anchor.get(sample.anchor_id)
        if group is None:
            raise ValueError(f"sample {sample.sample_id} candidate group is missing")
        baseline = baseline_by_id.get(group.baseline_evidence_id)
        if baseline is None:
            raise ValueError(f"sample {sample.sample_id} baseline evidence is missing")
        for role in ("baseline", "corrupted"):
            complete_name = f"replay_{role}_restoration_complete_state_comparison"
            if evidence.metrics.get(complete_name) is not True:
                raise ValueError(
                    f"evidence {evidence.evidence_id} lacks complete {role} "
                    "state comparison"
                )
            count = _strict_metric_int(
                evidence,
                f"replay_{role}_restoration_compared_component_count",
                minimum=1,
            )
            if count != baseline.compared_component_count:
                raise ValueError(
                    f"evidence {evidence.evidence_id} {role} state inventory mismatch"
                )
            error = _strict_metric_number(
                evidence,
                f"replay_{role}_restoration_maximum_absolute_error",
            )
            if error < 0.0 or error > 1e-6:
                raise ValueError(
                    f"evidence {evidence.evidence_id} {role} restoration exceeds 1e-6"
                )
            component_counts.add(count)
            errors.append(error)
    if (
        not component_counts
        or not errors
        or not verifier_component_counts
        or not verifier_errors
    ):
        raise ValueError("restoration verification inventory is empty")
    return (
        sorted(component_counts),
        max(errors),
        len(corrupted_samples),
        sorted(verifier_component_counts),
        max(verifier_errors),
        _restoration_error_distribution(errors),
        _restoration_error_distribution(verifier_errors),
    )


def _require_complete_evaluation(evaluation_dataset: Any) -> None:
    if evaluation_dataset.run_state is not RunState.COMPLETE:
        raise ValueError("action-verifier validation requires a complete replay run")
    latest: dict[str, Any] = {}
    for entry in evaluation_dataset.ledger:
        current = latest.get(entry.proposal_id)
        if current is None or entry.attempt_ordinal > current.attempt_ordinal:
            latest[entry.proposal_id] = entry
    if set(latest) != set(evaluation_dataset.selected_proposal_ids):
        raise ValueError("evaluation ledger does not cover every selected proposal")
    unfinished = {
        proposal_id
        for proposal_id, entry in latest.items()
        if entry.state in {LedgerState.PENDING, LedgerState.RUNNING}
    }
    if unfinished:
        raise ValueError("evaluation contains unfinished selected proposals")


def _load_resume_report(
    path: Path,
    *,
    evaluation_dataset: Any,
    anchor_manifest_digest: str,
    corruption_dataset_digest: str,
) -> Mapping[str, object]:
    source = Path(path).absolute()
    if source.is_symlink() or not source.is_file() or source.resolve() != source:
        raise ValueError("resume report must be one regular non-link file")
    if source.stat().st_size > 1_000_000:
        raise ValueError("resume report is unexpectedly large")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"resume report could not be loaded safely: {exc}") from exc
    if not isinstance(raw, dict) or any(type(key) is not str for key in raw):
        raise ValueError("resume report must contain one JSON object")
    required_ints = {
        "evaluated_attempts_this_invocation": 0,
        "recovered_attempts_this_invocation": 0,
        "retried_attempts_this_invocation": 0,
        "proposal_count": len(evaluation_dataset.selected_proposal_ids),
        "resumed_without_rerun_count": len(evaluation_dataset.selected_proposal_ids),
        "execution_error_count": 0,
    }
    for name, expected_count in required_ints.items():
        value = raw.get(name)
        if type(value) is not int or value != expected_count:
            raise ValueError(
                f"resume report {name} must equal {expected_count}, observed {value!r}"
            )
    required_values: tuple[tuple[str, object], ...] = (
        ("schema_version", "1.0"),
        ("resumed_invocation", True),
        ("run_id", evaluation_dataset.run_id),
        ("anchor_manifest_content_digest", anchor_manifest_digest),
        ("corruption_dataset_digest", corruption_dataset_digest),
    )
    for name, expected_value in required_values:
        if raw.get(name) != expected_value:
            raise ValueError(f"resume report {name} does not match validated inputs")
    return raw


def _baseline_exclusion_counts(anchor_manifest: Any) -> Mapping[str, int]:
    return dict(
        sorted(Counter(item.reason_code for item in anchor_manifest.exclusions).items())
    )


def _corruption_inventory(
    corruption_dataset: CorruptionDataset,
) -> tuple[Mapping[str, int], Mapping[str, int]]:
    family_counts = Counter(
        proposal.corruption_type for proposal in corruption_dataset.proposals
    )
    severity_counts: Counter[str] = Counter()
    for proposal in corruption_dataset.proposals:
        severity = proposal.resolved_parameters.get("severity_id")
        if (
            not isinstance(severity, str)
            or not severity
            or severity != severity.strip()
        ):
            raise ValueError(
                f"proposal {proposal.proposal_id} lacks a canonical severity_id"
            )
        severity_counts[severity] += 1
    return dict(sorted(family_counts.items())), dict(sorted(severity_counts.items()))


def _require_full_dataset_targets(
    report: Any,
    evaluation_dataset: Any,
    anchor_manifest: Any,
    corruption_dataset: CorruptionDataset,
) -> None:
    expected = {split.value: count for split, count in FULL_SPLIT_COUNTS.items()}
    observed = dict(report.split_trajectory_counts)
    if observed != expected:
        raise ValueError(f"full split target mismatch: {observed!r}")
    for value, minimum, name in (
        (report.candidate_group_count, 300, "valid anchors"),
        (report.corrupted_sample_count, 2400, "conclusive corruptions"),
        (report.corrupted_success_count, 500, "conclusive successes"),
        (report.corrupted_failure_count, 500, "conclusive task failures"),
    ):
        if value < minimum:
            raise ValueError(f"full target {name} requires at least {minimum}")
    if report.source_trajectory_count != 60:
        raise ValueError("full target requires exactly 60 source trajectories")
    if evaluation_dataset.summary.execution_error != 0:
        raise ValueError("full target requires zero execution errors")
    exclusions = _baseline_exclusion_counts(anchor_manifest)
    unexplained = {
        reason: count
        for reason, count in exclusions.items()
        if reason in {"baseline_runtime_error", "baseline_validator_error"}
        and count > 0
    }
    if unexplained:
        raise ValueError(
            "full target contains unexplained baseline execution errors: "
            f"{unexplained!r}"
        )
    proposal_ids = tuple(
        proposal.proposal_id for proposal in corruption_dataset.proposals
    )
    if (
        len(proposal_ids) != len(set(proposal_ids))
        or set(evaluation_dataset.selected_proposal_ids) != set(proposal_ids)
        or len(evaluation_dataset.selected_proposal_ids) != len(proposal_ids)
    ):
        raise ValueError("full target must evaluate every generated corruption once")
    proposals_by_episode: dict[str, list[Any]] = {}
    for proposal in corruption_dataset.proposals:
        proposals_by_episode.setdefault(proposal.source_episode_id, []).append(proposal)
    expected_episode_ids = {
        record.source_episode_id for record in anchor_manifest.records
    }
    if set(proposals_by_episode) != expected_episode_ids:
        raise ValueError("full target corruption anchors differ from anchor manifest")
    for episode_id, proposals in proposals_by_episode.items():
        severity_ids = {
            proposal.resolved_parameters.get("severity_id") for proposal in proposals
        }
        if len(proposals) != 8 or len(severity_ids) != 8:
            raise ValueError(
                f"full target anchor {episode_id} requires eight unique corruptions"
            )
    family_counts, _ = _corruption_inventory(corruption_dataset)
    required_families = {
        "additive_gaussian_noise",
        "constant_bias",
        "local_temporal_permutation",
        "segment_hold",
        "segment_zeroing",
        "temporal_field_shift",
    }
    if not required_families.issubset(family_counts):
        raise ValueError("full target corruption family coverage is incomplete")


def _run_validate_action_verifier(args: argparse.Namespace) -> int:
    command = "validate-action-verifier-dataset"
    if args.dry_run:
        print(
            f"{command} dry-run OK: "
            f"require_full_target={str(args.require_full_target).lower()} "
            f"resume_report={str(args.resume_report is not None).lower()} "
            "artifacts_loaded=0 output_created=false"
        )
        return 0
    try:
        if args.require_full_target and args.resume_report is None:
            raise ValueError("--require-full-target requires --resume-report")
        if args.report_output.exists() or args.report_output.is_symlink():
            raise ValueError("validation report output must be absent")
        if args.resume_report is not None and (
            _resolved(args.resume_report) == _resolved(args.report_output)
        ):
            raise ValueError("resume report and validation report must differ")
        _require_input_output_separation(
            input_directories={
                "action-verifier dataset": args.dataset_dir,
                "anchor manifest": args.anchor_manifest_dir,
                "corruption dataset": args.corruption_dir,
                "evidence dataset": args.evidence_dir,
            },
            output_directories={},
            output_files={"validation report": args.report_output},
        )
        anchor_manifest = load_anchor_manifest(args.anchor_manifest_dir)
        corruption = load_corruption_dataset(args.corruption_dir)
        corruption_digest = compute_corruption_dataset_content_digest(corruption)
        evaluation = load_evaluation_dataset(
            args.evidence_dir,
            corruption_dataset=corruption,
            expected_corruption_digest=corruption_digest,
        )
        _require_complete_evaluation(evaluation)
        evidence_digest = compute_action_verifier_evidence_dataset_digest(
            anchor_manifest, evaluation
        )
        available_ids = collect_action_verifier_evidence_ids(
            anchor_manifest, evaluation
        )
        expected_counts = FULL_SPLIT_COUNTS if args.require_full_target else None
        report = validate_serialized_action_verifier_export(
            args.dataset_dir,
            available_evidence_ids=available_ids,
            evidence_dataset_digest=evidence_digest,
            expected_split_counts=expected_counts,
        )
        dataset = load_action_verifier_dataset(args.dataset_dir)
        validate_action_verifier_evidence_bindings(
            dataset,
            anchor_manifest=anchor_manifest,
            corruption_dataset=corruption,
            evaluation_dataset=evaluation,
        )
        if args.require_full_target:
            _require_full_dataset_targets(
                report, evaluation, anchor_manifest, corruption
            )
        (
            component_counts,
            maximum_error,
            restoration_evidence_count,
            verifier_component_counts,
            verifier_maximum_error,
            restoration_error_distribution,
            verifier_restoration_error_distribution,
        ) = _restoration_report(dataset, evaluation, anchor_manifest, corruption)
        resume_report = (
            None
            if args.resume_report is None
            else _load_resume_report(
                args.resume_report,
                evaluation_dataset=evaluation,
                anchor_manifest_digest=anchor_manifest.content_digest,
                corruption_dataset_digest=corruption_digest,
            )
        )
        corruption_family_counts, corruption_severity_counts = _corruption_inventory(
            corruption
        )
        payload = dict(report.as_mapping())
        payload.update(
            {
                "action_dimension": dataset.action_dimension,
                "anchor_manifest_content_digest": anchor_manifest.content_digest,
                "baseline_success_count": len(anchor_manifest.baseline_evidence),
                "baseline_exclusion_reasons": _baseline_exclusion_counts(
                    anchor_manifest
                ),
                "chunk_horizon": dataset.chunk_horizon,
                "continuation_integrity_valid": True,
                "continuation_proposal_count": len(corruption.proposals),
                "corruption_family_counts": corruption_family_counts,
                "corruption_severity_counts": corruption_severity_counts,
                "corruption_dataset_digest": corruption_digest,
                "evaluation_execution_error_count": (
                    evaluation.summary.execution_error
                ),
                "evaluation_indeterminate_count": evaluation.summary.indeterminate,
                "evaluation_invalid_count": evaluation.summary.invalid,
                "full_target_required": args.require_full_target,
                "full_target_valid": args.require_full_target,
                "progress_semantic": dataset.samples[0].progress_semantic,
                "restoration_compared_component_counts": component_counts,
                "restoration_evidence_count": restoration_evidence_count,
                "restoration_error_distribution": restoration_error_distribution,
                "restoration_maximum_absolute_error": maximum_error,
                "resume_idempotence_valid": resume_report is not None,
                "run_id": evaluation.run_id,
                "state_vector_dimension": dataset.state_vector_dimension,
                "state_vector_semantic": dataset.state_vector_semantic,
                "verifier_state_restoration_compared_component_counts": (
                    verifier_component_counts
                ),
                "verifier_state_restoration_maximum_absolute_error": (
                    verifier_maximum_error
                ),
                "verifier_state_restoration_error_distribution": (
                    verifier_restoration_error_distribution
                ),
                "verifier_state_restoration_valid": True,
            }
        )
        write_sanitized_report(payload, args.report_output)
        print(
            f"{command} OK: trajectories={report.source_trajectory_count} "
            f"groups={report.candidate_group_count} "
            f"samples={report.sample_count} "
            f"corrupted={report.corrupted_sample_count} "
            f"success={report.corrupted_success_count} "
            f"failure={report.corrupted_failure_count} "
            f"leakage=false digest={report.dataset_content_digest}"
        )
        return 0
    except KeyboardInterrupt:
        print(f"{command} interrupted; no report accepted", file=sys.stderr)
        return 130
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"{command} failed: {error}", file=sys.stderr)
        return 1


__all__ = ["M3A_COMMANDS", "add_m3a_subparsers", "run_m3a_command"]

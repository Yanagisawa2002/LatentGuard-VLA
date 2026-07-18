"""CLI entry points for M4C receding-horizon visual action shielding."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from latentguard.control.benchmark import (
    build_benchmark_schedule,
    load_episode_execution_seconds,
    record_episode_execution_seconds,
    run_benchmark_schedule,
)
from latentguard.control.config import (
    load_bootstrap_configuration,
    load_candidate_pool_binding,
    load_closed_loop_configuration,
    load_selector_matrix,
    load_source_seed_schedule,
)
from latentguard.control.metrics import (
    aggregate_episode_metrics,
    content_identical_decision_agreement,
    episode_metric_row,
    intervention_diagnostics,
    paired_trajectory_bootstrap,
)
from latentguard.control.runner import run_closed_loop_episode
from latentguard.control.serialization import (
    ClosedLoopEpisodeStore,
    write_atomic_json,
)
from latentguard.control.source_plans import (
    load_source_plans,
    prepare_source_plans,
    save_source_plan_manifest,
)
from latentguard.integrations.maniskill_pickcube.closed_loop import (
    PickCubeClosedLoopCandidateFactory,
    PickCubeClosedLoopRuntime,
    PickCubeVisualBoundaryObserver,
)
from latentguard.integrations.maniskill_pickcube.closed_loop_state_store import (
    ClosedLoopStateStore,
)
from latentguard.integrations.maniskill_pickcube.compatibility import (
    load_compatibility_report,
    validate_compatibility_report,
)
from latentguard.integrations.maniskill_pickcube.configuration import (
    load_expected_contract,
    load_maniskill_pickcube_action_layout,
    validate_maniskill_pickcube_action_layout_binding,
)
from latentguard.integrations.maniskill_pickcube.serialization import (
    action_contract_from_compatibility,
    environment_settings_from_compatibility,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
    load_state_indexed_archive,
)
from latentguard.integrations.maniskill_pickcube.task_evidence import (
    PickCubeTaskKeyContract,
)
from latentguard.integrations.maniskill_pickcube.visual_rendering import (
    build_pickcube_visual_render_plan,
)
from latentguard.selection.configuration import load_candidate_pool_configuration
from latentguard.vision_data.cameras import PickCubeMultiViewRigV1
from latentguard.vision_data.configuration import (
    load_camera_rig_configuration,
    load_render_domain_configuration,
)
from latentguard.vision_data.domains import RenderDomainConfigurationV1

M4C_COMMANDS = frozenset(
    {
        "prepare-closed-loop-source-plans",
        "run-receding-horizon-selector",
        "benchmark-receding-horizon-selectors",
        "resume-receding-horizon-benchmark",
        "evaluate-receding-horizon-benchmark",
        "inspect-closed-loop-episode",
    }
)

_CONTROL_ROOT = Path("configs/control/m4c")


def _add_common_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-archive-dir", type=Path, required=True)
    parser.add_argument("--source-plan-manifest", type=Path, required=True)


def _add_common_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_source_arguments(parser)
    parser.add_argument(
        "--expected-contract",
        type=Path,
        default=Path(
            "configs/integrations/maniskill_pickcube/expected-contract-v1.json"
        ),
    )
    parser.add_argument("--compatibility-report", type=Path, required=True)
    parser.add_argument(
        "--action-layout",
        type=Path,
        default=Path("configs/integrations/maniskill_pickcube/action-layout-v1.json"),
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=Path("configs/selection/m3c/candidate-pool-v1.json"),
    )
    parser.add_argument(
        "--candidate-binding-config",
        type=Path,
        default=_CONTROL_ROOT / "candidate-pool-binding-v1.json",
    )
    parser.add_argument(
        "--selector-matrix-config",
        type=Path,
        default=_CONTROL_ROOT / "selector-matrix-v1.json",
    )
    parser.add_argument(
        "--closed-loop-config",
        type=Path,
        default=_CONTROL_ROOT / "closed-loop-v1.json",
    )
    parser.add_argument("--m3b-result-root", type=Path, required=True)
    parser.add_argument("--m3b-runtime-root", type=Path, required=True)
    parser.add_argument("--backbone-config", type=Path, required=True)
    parser.add_argument("--backbone-manifest", type=Path, required=True)
    parser.add_argument("--direct-model-config", type=Path, required=True)
    parser.add_argument("--direct-freeze-manifest", type=Path, required=True)
    parser.add_argument(
        "--direct-checkpoint", type=Path, action="append", required=True
    )
    parser.add_argument("--distilled-model-config", type=Path, required=True)
    parser.add_argument("--distilled-freeze-manifest", type=Path, required=True)
    parser.add_argument(
        "--distilled-checkpoint", type=Path, action="append", required=True
    )
    parser.add_argument("--camera-rig-config", type=Path, required=True)
    parser.add_argument("--render-domain-config", type=Path, required=True)
    parser.add_argument("--render-seed", type=int, default=271828)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")


def add_m4c_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the six required M4C commands."""

    prepare = subparsers.add_parser("prepare-closed-loop-source-plans")
    prepare.add_argument("--source-archive-dir", type=Path, required=True)
    prepare.add_argument("--reference-archive-dir", type=Path, required=True)
    prepare.add_argument("--collection-summary-output", type=Path, required=True)
    prepare.add_argument("--compatibility-report", type=Path, required=True)
    prepare.add_argument(
        "--expected-contract",
        type=Path,
        default=Path(
            "configs/integrations/maniskill_pickcube/expected-contract-v1.json"
        ),
    )
    prepare.add_argument(
        "--action-layout",
        type=Path,
        default=Path("configs/integrations/maniskill_pickcube/action-layout-v1.json"),
    )
    prepare.add_argument(
        "--source-seed-schedule-config",
        type=Path,
        default=_CONTROL_ROOT / "source-seed-schedule-v1.json",
    )
    prepare.add_argument(
        "--collection-profile", choices=("smoke", "full"), required=True
    )
    prepare.add_argument("--trajectory-action-limit", type=int)
    prepare.add_argument("--fresh-state-trajectory-limit", type=int, default=1)
    prepare.add_argument(
        "--prior-state-indexed-archive", type=Path, action="append", required=True
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--dry-run", action="store_true")

    run = subparsers.add_parser("run-receding-horizon-selector")
    _add_common_runtime_arguments(run)
    run.add_argument("--source-trajectory-id", required=True)
    run.add_argument("--selector-id", required=True)
    run.add_argument("--visual-domain", default="not_applicable")

    benchmark = subparsers.add_parser("benchmark-receding-horizon-selectors")
    _add_common_runtime_arguments(benchmark)

    resume = subparsers.add_parser("resume-receding-horizon-benchmark")
    _add_common_runtime_arguments(resume)
    resume.add_argument("--require-zero-work", action="store_true")

    evaluate = subparsers.add_parser("evaluate-receding-horizon-benchmark")
    _add_common_source_arguments(evaluate)
    evaluate.add_argument(
        "--selector-matrix-config",
        type=Path,
        default=_CONTROL_ROOT / "selector-matrix-v1.json",
    )
    evaluate.add_argument(
        "--bootstrap-config",
        type=Path,
        default=_CONTROL_ROOT / "bootstrap-v1.json",
    )
    evaluate.add_argument("--benchmark-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    inspect = subparsers.add_parser("inspect-closed-loop-episode")
    inspect.add_argument("--episode-dir", type=Path, required=True)
    inspect.add_argument("--output", type=Path)


def _scope(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
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
    task_keys = PickCubeTaskKeyContract.from_compatibility_report(binding.report)
    return binding, layout, settings, (action_contract, task_keys)


def _run_prepare(args: argparse.Namespace) -> int:
    schedule = load_source_seed_schedule(args.source_seed_schedule_config)
    if args.collection_profile == "smoke":
        start_seed = schedule.smoke_start_seed
        requested = schedule.smoke_requested_successes
        maximum_attempts = schedule.smoke_maximum_attempts
    else:
        start_seed = schedule.full_start_seed
        requested = schedule.full_requested_successes
        maximum_attempts = schedule.full_maximum_attempts
    priors = tuple(
        load_state_indexed_archive(path) for path in args.prior_state_indexed_archive
    )
    if args.dry_run:
        print(
            "prepare-closed-loop-source-plans dry-run OK: "
            f"profile={args.collection_profile} requested={requested} "
            f"maximum_attempts={maximum_attempts} prior_archives={len(priors)} "
            "simulator_runs=0 output_created=false"
        )
        return 0
    from latentguard.m3a_cli import _run_collect_sequences

    collection_args = argparse.Namespace(
        action_layout=args.action_layout,
        command="collect-maniskill-pickcube-sequences",
        compatibility_report=args.compatibility_report,
        dry_run=False,
        expected_contract=args.expected_contract,
        fresh_state_trajectory_limit=args.fresh_state_trajectory_limit,
        maximum_attempts=maximum_attempts,
        reference_archive_dir=args.reference_archive_dir,
        requested_success_count=requested,
        runtime_archive_dir=args.source_archive_dir,
        sim_backend="gpu",
        starting_seed=start_seed,
        summary_output=args.collection_summary_output,
        trajectory_action_limit=args.trajectory_action_limit,
    )
    collection_status = _run_collect_sequences(collection_args)
    if collection_status != 0:
        return collection_status
    archive = load_state_indexed_archive(args.source_archive_dir)
    plans = prepare_source_plans(
        archive, prior_archives=priors, expected_count=requested
    )
    save_source_plan_manifest(
        plans, args.output, archive_content_digest=archive.content_digest
    )
    print(
        "prepare-closed-loop-source-plans OK: "
        f"accepted={len(plans)} manifest={args.output}"
    )
    return 0


def _checkpoint_tuple(values: Sequence[Path], context: str) -> tuple[Path, Path, Path]:
    items: tuple[Path, ...] = tuple(values)
    if len(items) != 3:
        raise ValueError(f"{context} requires exactly three checkpoints")
    return items[0], items[1], items[2]


def _build_assets(args: argparse.Namespace) -> dict[str, object]:
    from latentguard.control.frozen_selectors import (
        DeterministicRandomSelector,
        FixedPrimarySelector,
        FrozenStructuredEnsembleSelector,
        load_frozen_visual_selector,
    )
    from latentguard.selection.checkpoint_bundle import load_verifier_bundle
    from latentguard.selection.preparation import build_verifier_bundle_identities
    from latentguard.visual_training.backbone import prepare_backbone
    from latentguard.visual_training.config import (
        load_backbone_config,
        load_visual_model_config,
    )

    binding, layout, settings, contracts = _scope(args)
    action_contract, task_keys = contracts
    source_archive = load_state_indexed_archive(args.source_archive_dir)
    sources = load_source_plans(args.source_plan_manifest, archive=source_archive)
    matrix = load_selector_matrix(args.selector_matrix_config)
    control_config = load_closed_loop_configuration(args.closed_loop_config)
    candidate_binding = load_candidate_pool_binding(args.candidate_binding_config)
    candidate_config = load_candidate_pool_configuration(args.candidate_config)
    if (
        candidate_config.content_digest
        != candidate_binding.candidate_pool_configuration_digest
        or candidate_config.action_contract_digest
        != candidate_binding.action_contract_digest
        or binding.report.action_contract_digest
        != candidate_binding.action_contract_digest
    ):
        raise ValueError("M4C candidate/action binding differs from accepted M3C")
    candidate_factory = PickCubeClosedLoopCandidateFactory(
        configuration=candidate_config,
        action_layout=layout.m1_action_layout,
        action_contract=action_contract,
        action_contract_digest=binding.report.action_contract_digest,
    )
    bundles, paths, _, _ = build_verifier_bundle_identities(
        args.m3b_result_root, runtime_root=args.m3b_runtime_root
    )
    action_bundle = load_verifier_bundle(
        bundles["action_only_mlp"], paths["action_only_mlp"], device=args.device
    )
    privileged_bundle = load_verifier_bundle(
        bundles["state_action_mlp"], paths["state_action_mlp"], device=args.device
    )
    schema = source_archive.episodes[0].states[0].verifier_state.schema
    backbone = prepare_backbone(
        load_backbone_config(args.backbone_config),
        args.backbone_manifest,
        device=args.device,
    )
    direct = load_frozen_visual_selector(
        selector_id="direct_visual_ensemble_v1",
        backbone=backbone,
        model_config=load_visual_model_config(args.direct_model_config),
        model_freeze_manifest=args.direct_freeze_manifest,
        checkpoints=_checkpoint_tuple(args.direct_checkpoint, "direct selector"),
        device=args.device,
    )
    distilled = load_frozen_visual_selector(
        selector_id="distilled_visual_ensemble_v1",
        backbone=backbone,
        model_config=load_visual_model_config(args.distilled_model_config),
        model_freeze_manifest=args.distilled_freeze_manifest,
        checkpoints=_checkpoint_tuple(args.distilled_checkpoint, "distilled selector"),
        device=args.device,
    )
    selectors = {
        "fixed_primary_v1": FixedPrimarySelector(),
        "deterministic_random_v1": DeterministicRandomSelector(),
        "action_only_ensemble_v1": FrozenStructuredEnsembleSelector(
            selector_id="action_only_ensemble_v1",
            bundle=action_bundle,
            verifier_schema=schema,
        ),
        "direct_visual_ensemble_v1": direct,
        "distilled_visual_ensemble_v1": distilled,
        "privileged_structured_ensemble_v1": FrozenStructuredEnsembleSelector(
            selector_id="privileged_structured_ensemble_v1",
            bundle=privileged_bundle,
            verifier_schema=schema,
        ),
    }
    rig = cast(
        PickCubeMultiViewRigV1,
        load_camera_rig_configuration(args.camera_rig_config),
    )
    domains = cast(
        RenderDomainConfigurationV1,
        load_render_domain_configuration(args.render_domain_config),
    )
    plans = {
        domain_id: build_pickcube_visual_render_plan(
            rig, domains.domain(domain_id), domains, args.render_seed
        )
        for domain_id in matrix.visual_domains
    }
    state_store = ClosedLoopStateStore(args.output_root / "state-snapshots")

    def runtime_factory() -> PickCubeClosedLoopRuntime:
        observer = PickCubeVisualBoundaryObserver(
            plans=plans,
            key_contract=task_keys,
            state_tolerance=settings.state_tolerance,
        )
        return PickCubeClosedLoopRuntime(
            source_archive=source_archive,
            settings=settings,
            action_contract=action_contract,
            key_contract=task_keys,
            state_store=state_store,
            visual_observer=observer,
        )

    return {
        "candidate_factory": candidate_factory,
        "config": control_config,
        "matrix": matrix,
        "runtime_factory": runtime_factory,
        "selectors": selectors,
        "sources": sources,
    }


def _run_single(args: argparse.Namespace) -> int:
    if args.dry_run:
        matrix = load_selector_matrix(args.selector_matrix_config)
        allowed = {item.selector_id: item.visual for item in matrix.selectors}
        if args.selector_id not in allowed:
            raise ValueError("selector is absent from the frozen M4C matrix")
        if allowed[args.selector_id] != (args.visual_domain != "not_applicable"):
            raise ValueError("selector and visual domain applicability differ")
        print("run-receding-horizon-selector dry-run OK: simulator_runs=0")
        return 0
    assets = _build_assets(args)
    sources = cast(Sequence[Any], assets["sources"])
    matching = [
        item
        for item in sources
        if item.identity.source_trajectory_id == args.source_trajectory_id
    ]
    if len(matching) != 1:
        raise ValueError("requested M4C source trajectory is not unique")
    selectors = cast(Mapping[str, Any], assets["selectors"])
    try:
        selector = selectors[args.selector_id]
    except KeyError as exc:
        raise ValueError("selector is absent from the frozen M4C matrix") from exc
    store = ClosedLoopEpisodeStore(
        args.output_root
        / args.selector_id
        / args.visual_domain
        / args.source_trajectory_id
    )
    started = perf_counter()
    result = run_closed_loop_episode(
        matching[0],
        selector=selector,
        visual_domain=args.visual_domain,
        config=cast(Any, assets["config"]),
        candidate_factory=cast(Any, assets["candidate_factory"]),
        runtime=cast(Any, assets["runtime_factory"])(),
        store=store,
    )
    elapsed = perf_counter() - started
    if result.zero_work_resume:
        load_episode_execution_seconds(
            store.root,
            expected_episode_execution_id=result.episode.episode_execution_id,
        )
    else:
        record_episode_execution_seconds(
            store.root,
            episode_execution_id=result.episode.episode_execution_id,
            execution_seconds=elapsed,
        )
    print(
        "run-receding-horizon-selector OK: "
        f"state={result.episode.state.value} "
        f"decisions={len(result.episode.boundaries)} "
        f"zero_work={str(result.zero_work_resume).lower()}"
    )
    return int(result.episode.state.value == "execution_error")


def _run_benchmark(args: argparse.Namespace, *, resume: bool) -> int:
    if args.dry_run:
        matrix = load_selector_matrix(args.selector_matrix_config)
        archive = load_state_indexed_archive(args.source_archive_dir)
        sources = load_source_plans(args.source_plan_manifest, archive=archive)
        planned = len(build_benchmark_schedule(sources, matrix))
        print(
            f"{'resume' if resume else 'benchmark'}-receding-horizon-selectors "
            f"dry-run OK: planned={planned} simulator_runs=0"
        )
        return 0
    assets = _build_assets(args)
    summary = run_benchmark_schedule(
        cast(Sequence[Any], assets["sources"]),
        matrix=cast(Any, assets["matrix"]),
        config=cast(Any, assets["config"]),
        selectors=cast(Mapping[str, Any], assets["selectors"]),
        candidate_factory=cast(Any, assets["candidate_factory"]),
        runtime_factory=cast(Any, assets["runtime_factory"]),
        output_root=args.output_root,
    )
    if resume and args.require_zero_work and not summary.zero_work_resume:
        raise RuntimeError("strict complete resume performed duplicate work")
    payload = {
        "completed_episode_count": summary.completed_episode_count,
        "executed_episode_count": summary.executed_episode_count,
        "outcome_counts": dict(summary.outcome_counts),
        "planned_episode_count": summary.planned_episode_count,
        "schema_version": "1.0",
        "zero_work_episode_count": summary.zero_work_episode_count,
        "zero_work_resume": summary.zero_work_resume,
    }
    report_name = "resume-summary.json" if resume else "benchmark-summary.json"
    write_atomic_json(args.output_root / report_name, payload)
    print(
        f"{'resume' if resume else 'benchmark'}-receding-horizon-selectors OK: "
        f"completed={summary.completed_episode_count}/"
        f"{summary.planned_episode_count} executed={summary.executed_episode_count} "
        f"zero_work={summary.zero_work_episode_count}"
    )
    return int(summary.outcome_counts.get("execution_error", 0) > 0)


def _load_benchmark_records(
    args: argparse.Namespace,
) -> tuple[list[Any], list[Any], dict[str, float]]:
    archive = load_state_indexed_archive(args.source_archive_dir)
    sources = load_source_plans(args.source_plan_manifest, archive=archive)
    matrix = load_selector_matrix(args.selector_matrix_config)
    episodes: list[Any] = []
    decisions: list[Any] = []
    execution_seconds: dict[str, float] = {}
    for spec in build_benchmark_schedule(sources, matrix):
        store = ClosedLoopEpisodeStore(args.benchmark_root / spec.relative_directory)
        episode = store.load_episode()
        if not episode.state.terminal:
            raise ValueError("M4C evaluation found a non-terminal episode")
        episodes.append(episode)
        execution_seconds[episode.episode_execution_id] = (
            load_episode_execution_seconds(
                store.root,
                expected_episode_execution_id=episode.episode_execution_id,
            )
        )
        decisions.extend(
            store.load_decision(index) for index in range(len(episode.boundaries))
        )
    return episodes, decisions, execution_seconds


def _run_evaluate(args: argparse.Namespace) -> int:
    episodes, decisions, execution_seconds = _load_benchmark_records(args)
    decisions_by_episode: dict[str, list[Any]] = defaultdict(list)
    for item in decisions:
        decisions_by_episode[item.episode_execution_id].append(item)
    rows = [
        episode_metric_row(
            episode,
            sorted(
                decisions_by_episode[episode.episode_execution_id],
                key=lambda item: item.decision_ordinal,
            ),
            primary_candidate_ordinal=0,
            execution_seconds=execution_seconds[episode.episode_execution_id],
        )
        for episode in episodes
    ]
    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for row in rows:
        grouped[(row.selector_id, row.visual_domain)].append(row)
    aggregates = {
        f"{selector}/{domain}": aggregate_episode_metrics(values)
        for (selector, domain), values in sorted(grouped.items())
    }
    bootstrap = load_bootstrap_configuration(args.bootstrap_config)
    comparisons = _predeclared_bootstrap_comparisons(grouped, bootstrap=bootstrap)
    payload = {
        "aggregate_metrics": aggregates,
        "benchmark_complete": True,
        "episode_count": len(episodes),
        "execution_time_note": (
            "per-episode wall time is not part of semantic episode identity"
        ),
        "intervention_diagnostics": {
            key: intervention_diagnostics(values)
            for key, values in sorted(
                (f"{a}/{b}", value) for (a, b), value in grouped.items()
            )
        },
        "paired_trajectory_bootstrap": comparisons,
        "schema_version": "1.0",
        "selection_consistency": content_identical_decision_agreement(decisions),
    }
    write_atomic_json(args.output, payload)
    print(
        "evaluate-receding-horizon-benchmark OK: "
        f"episodes={len(episodes)} report={args.output}"
    )
    return 0


def _predeclared_bootstrap_comparisons(
    grouped: Mapping[tuple[str, str], Sequence[Any]],
    *,
    bootstrap: Any,
) -> dict[str, object]:
    """Build every predeclared trajectory-paired M4C comparison."""

    primary_selector = "distilled_visual_ensemble_v1"
    domains = ("canonical", "strong_camera_shift", "strong_lighting_shift")
    nonvisual_comparators = (
        "deterministic_random_v1",
        "fixed_primary_v1",
        "action_only_ensemble_v1",
        "privileged_structured_ensemble_v1",
    )
    comparisons: dict[str, object] = {}
    for domain in domains:
        primary = grouped[(primary_selector, domain)]
        for comparator in nonvisual_comparators:
            comparisons[f"{primary_selector}/{domain}_vs_{comparator}"] = (
                paired_trajectory_bootstrap(
                    primary,
                    grouped[(comparator, "not_applicable")],
                    configuration=bootstrap,
                )
            )
        comparisons[
            f"{primary_selector}/{domain}_vs_direct_visual_ensemble_v1/{domain}"
        ] = paired_trajectory_bootstrap(
            primary,
            grouped[("direct_visual_ensemble_v1", domain)],
            configuration=bootstrap,
        )
    canonical = grouped[(primary_selector, "canonical")]
    for domain in ("strong_camera_shift", "strong_lighting_shift"):
        comparisons[f"{primary_selector}/canonical_vs_{domain}"] = (
            paired_trajectory_bootstrap(
                canonical,
                grouped[(primary_selector, domain)],
                configuration=bootstrap,
            )
        )
    return comparisons


def _run_inspect(args: argparse.Namespace) -> int:
    store = ClosedLoopEpisodeStore(args.episode_dir)
    episode = store.load_episode()
    decisions = [store.load_decision(index) for index in range(len(episode.boundaries))]
    payload = {
        "content_digest": episode.content_digest,
        "decision_count": len(decisions),
        "episode_execution_id": episode.episode_execution_id,
        "executed_control_steps": episode.executed_control_steps,
        "recovery_count": episode.recovery_count,
        "schema_version": "1.0",
        "selected_candidate_ids": [item.selected_candidate_id for item in decisions],
        "selector_id": episode.selector_id,
        "source_trajectory_id": episode.source_trajectory_id,
        "state": episode.state.value,
        "visual_domain": episode.visual_domain,
    }
    if args.output is not None:
        write_atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def run_m4c_command(args: argparse.Namespace) -> int | None:
    """Dispatch one M4C command or return ``None`` for another milestone."""

    if args.command not in M4C_COMMANDS:
        return None
    try:
        if args.command == "prepare-closed-loop-source-plans":
            return _run_prepare(args)
        if args.command == "run-receding-horizon-selector":
            return _run_single(args)
        if args.command == "benchmark-receding-horizon-selectors":
            return _run_benchmark(args, resume=False)
        if args.command == "resume-receding-horizon-benchmark":
            return _run_benchmark(args, resume=True)
        if args.command == "evaluate-receding-horizon-benchmark":
            return _run_evaluate(args)
        if args.command == "inspect-closed-loop-episode":
            return _run_inspect(args)
    except KeyboardInterrupt:
        print(
            f"{args.command} interrupted; transactional state preserved",
            file=sys.stderr,
        )
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"{args.command} failed: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled M4C command {args.command!r}")


__all__ = ["M4C_COMMANDS", "add_m4c_subparsers", "run_m4c_command"]

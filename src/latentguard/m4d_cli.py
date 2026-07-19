"""CLI entry points for the M4D conservative fallback-aware shield."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from latentguard.control.benchmark import load_episode_execution_seconds
from latentguard.control.config import load_bootstrap_configuration
from latentguard.control.fallback import (
    load_fallback_decision,
    load_fault_schedule,
    load_gate_profiles,
)
from latentguard.control.fallback_benchmark import (
    build_fallback_benchmark_schedule,
    load_gate_selection,
    run_fallback_benchmark_schedule,
)
from latentguard.control.fallback_config import (
    load_fallback_selector_matrix,
    load_m4d_source_seed_configuration,
)
from latentguard.control.fallback_metrics import (
    FallbackEpisodeMetricRowV1,
    aggregate_fallback_metrics,
    fallback_episode_metric_row,
    group_fallback_rows,
    paired_fallback_bootstrap,
    select_gate_profiles,
)
from latentguard.control.models import EpisodeState, content_digest
from latentguard.control.serialization import (
    ClosedLoopEpisodeStore,
    write_atomic_json,
    write_exclusive_json,
)
from latentguard.control.source_plans import (
    load_source_plans,
    prepare_source_plans,
    save_source_plan_manifest,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
    load_state_indexed_archive,
)
from latentguard.m4c_cli import (
    _add_common_runtime_arguments,
    _build_assets,
    _load_benchmark_records,
)

M4D_COMMANDS = frozenset(
    {
        "diagnose-m4c-interventions",
        "prepare-m4d-source-plans",
        "run-fallback-shield-development",
        "select-fallback-gates",
        "run-fallback-shield-benchmark",
        "resume-fallback-shield-benchmark",
        "evaluate-fallback-shield",
        "inspect-fallback-shield-episode",
    }
)

_ROOT = Path("configs/control/m4d")


def _add_m4d_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_runtime_arguments(parser)
    parser.add_argument(
        "--m4d-selector-matrix-config",
        type=Path,
        default=_ROOT / "selector-matrix-v1.json",
    )
    parser.add_argument(
        "--fault-schedule-config",
        type=Path,
        default=_ROOT / "fault-schedule-v1.json",
    )
    parser.add_argument(
        "--gate-profiles-config",
        type=Path,
        default=_ROOT / "gate-profiles-v1.json",
    )


def add_m4d_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the eight required M4D commands."""

    diagnose = subparsers.add_parser("diagnose-m4c-interventions")
    diagnose.add_argument("--source-archive-dir", type=Path, required=True)
    diagnose.add_argument("--source-plan-manifest", type=Path, required=True)
    diagnose.add_argument(
        "--selector-matrix-config",
        type=Path,
        default=Path("configs/control/m4c/selector-matrix-v1.json"),
    )
    diagnose.add_argument("--benchmark-root", type=Path, required=True)
    diagnose.add_argument("--output", type=Path, required=True)

    prepare = subparsers.add_parser("prepare-m4d-source-plans")
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
        "--seed-config",
        type=Path,
        choices=(_ROOT / "development-seeds-v1.json", _ROOT / "final-seeds-v1.json"),
        required=True,
    )
    prepare.add_argument("--trajectory-action-limit", type=int)
    prepare.add_argument("--fresh-state-trajectory-limit", type=int, default=1)
    prepare.add_argument(
        "--prior-state-indexed-archive", type=Path, action="append", required=True
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--dry-run", action="store_true")

    development = subparsers.add_parser("run-fallback-shield-development")
    _add_m4d_runtime_arguments(development)
    development.add_argument("--source-limit", type=int, choices=(6, 12))

    select = subparsers.add_parser("select-fallback-gates")
    select.add_argument("--development-results", type=Path, required=True)
    select.add_argument(
        "--gate-profiles-config",
        type=Path,
        default=_ROOT / "gate-profiles-v1.json",
    )
    select.add_argument("--output", type=Path, required=True)

    benchmark = subparsers.add_parser("run-fallback-shield-benchmark")
    _add_m4d_runtime_arguments(benchmark)
    benchmark.add_argument("--gate-selection", type=Path, required=True)

    resume = subparsers.add_parser("resume-fallback-shield-benchmark")
    _add_m4d_runtime_arguments(resume)
    resume.add_argument("--gate-selection", type=Path, required=True)
    resume.add_argument("--require-zero-work", action="store_true")

    evaluate = subparsers.add_parser("evaluate-fallback-shield")
    evaluate.add_argument("--source-archive-dir", type=Path, required=True)
    evaluate.add_argument("--source-plan-manifest", type=Path, required=True)
    evaluate.add_argument(
        "--m4d-selector-matrix-config",
        type=Path,
        default=_ROOT / "selector-matrix-v1.json",
    )
    evaluate.add_argument(
        "--bootstrap-config", type=Path, default=_ROOT / "bootstrap-v1.json"
    )
    evaluate.add_argument("--benchmark-root", type=Path, required=True)
    evaluate.add_argument("--gate-selection", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    inspect = subparsers.add_parser("inspect-fallback-shield-episode")
    inspect.add_argument("--episode-dir", type=Path, required=True)
    inspect.add_argument("--output", type=Path)


def _run_prepare(args: argparse.Namespace) -> int:
    seeds = load_m4d_source_seed_configuration(args.seed_config)
    priors = tuple(
        load_state_indexed_archive(path) for path in args.prior_state_indexed_archive
    )
    if args.dry_run:
        print(
            "prepare-m4d-source-plans dry-run OK: "
            f"profile={seeds.profile} requested={seeds.requested_successes} "
            f"maximum_attempts={seeds.maximum_attempts} "
            f"prior_archives={len(priors)} simulator_runs=0 output_created=false"
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
        maximum_attempts=seeds.maximum_attempts,
        reference_archive_dir=args.reference_archive_dir,
        requested_success_count=seeds.requested_successes,
        runtime_archive_dir=args.source_archive_dir,
        sim_backend="gpu",
        starting_seed=seeds.starting_seed,
        summary_output=args.collection_summary_output,
        trajectory_action_limit=args.trajectory_action_limit,
    )
    status = _run_collect_sequences(collection_args)
    if status != 0:
        return status
    archive = load_state_indexed_archive(args.source_archive_dir)
    plans = prepare_source_plans(
        archive, prior_archives=priors, expected_count=seeds.requested_successes
    )
    save_source_plan_manifest(
        plans, args.output, archive_content_digest=archive.content_digest
    )
    print(
        "prepare-m4d-source-plans OK: "
        f"profile={seeds.profile} accepted={len(plans)} manifest={args.output}"
    )
    return 0


def _build_m4d_assets(args: argparse.Namespace) -> dict[str, object]:
    assets = _build_assets(args)
    base_selectors = cast(Mapping[str, Any], assets["selectors"])
    scorers = {
        key: value
        for key, value in base_selectors.items()
        if key
        in {
            "action_only_ensemble_v1",
            "direct_visual_ensemble_v1",
            "distilled_visual_ensemble_v1",
            "privileged_structured_ensemble_v1",
        }
    }
    if any(not hasattr(item, "score_candidates") for item in scorers.values()):
        raise ValueError("M4D scorer lacks uncertainty-aware inference")
    return {
        **assets,
        "m4d_matrix": load_fallback_selector_matrix(args.m4d_selector_matrix_config),
        "fault_schedule": load_fault_schedule(args.fault_schedule_config),
        "gate_profiles": load_gate_profiles(args.gate_profiles_config),
        "scorers": scorers,
    }


def _summary_payload(summary: Any, *, phase: str) -> dict[str, object]:
    return {
        "completed_episode_count": summary.completed_episode_count,
        "executed_episode_count": summary.executed_episode_count,
        "outcome_counts": dict(summary.outcome_counts),
        "phase": phase,
        "planned_episode_count": summary.planned_episode_count,
        "schema_version": "1.0",
        "zero_work_episode_count": summary.zero_work_episode_count,
        "zero_work_resume": summary.zero_work_resume,
    }


def _run_schedule(args: argparse.Namespace, *, phase: str, resume: bool) -> int:
    matrix = load_fallback_selector_matrix(args.m4d_selector_matrix_config)
    archive = load_state_indexed_archive(args.source_archive_dir)
    sources = load_source_plans(args.source_plan_manifest, archive=archive)
    if phase == "development" and args.source_limit is not None:
        sources = sources[: args.source_limit]
    expected_source_count = 60 if phase == "final" else (args.source_limit or 12)
    if len(sources) != expected_source_count:
        raise ValueError(
            f"M4D {phase} requires exactly {expected_source_count} sources"
        )
    selected: Mapping[str, str] | None = None
    gate_selection_digest: str | None = None
    selected_profiles_digest: str | None = None
    if phase == "final":
        selected, gate_selection_digest, selected_profiles_digest = load_gate_selection(
            args.gate_selection
        )
        profiles = load_gate_profiles(args.gate_profiles_config)
        if profiles.content_digest != selected_profiles_digest:
            raise ValueError("final gate selection and profile configuration differ")
    planned = len(
        build_fallback_benchmark_schedule(
            sources,
            matrix,
            phase=phase,
            selected_profiles=selected,
        )
    )
    if args.dry_run:
        print(
            f"{'resume' if resume else 'run'}-fallback-shield-{phase} "
            f"dry-run OK: planned={planned} simulator_runs=0"
        )
        return 0
    assets = _build_m4d_assets(args)
    summary = run_fallback_benchmark_schedule(
        cast(Sequence[Any], assets["sources"]),
        phase=phase,
        matrix=cast(Any, assets["m4d_matrix"]),
        selected_profiles=selected,
        gate_profiles=cast(Any, assets["gate_profiles"]),
        fault_schedule=cast(Any, assets["fault_schedule"]),
        config=cast(Any, assets["config"]),
        scorers=cast(Mapping[str, Any], assets["scorers"]),
        candidate_factory=cast(Any, assets["candidate_factory"]),
        runtime_factory=cast(Any, assets["runtime_factory"]),
        output_root=args.output_root,
    )
    if resume and args.require_zero_work and not summary.zero_work_resume:
        raise RuntimeError("strict complete M4D resume performed duplicate work")
    payload = {
        **_summary_payload(summary, phase=phase),
        "gate_selection_digest": gate_selection_digest,
    }
    name = (
        "resume-summary.json"
        if resume
        else "development-summary.json"
        if phase == "development"
        else "benchmark-summary.json"
    )
    write_atomic_json(args.output_root / name, payload)
    if phase == "development" and not resume:
        results = _evaluate_records(
            sources=cast(Sequence[Any], assets["sources"]),
            matrix=cast(Any, assets["m4d_matrix"]),
            benchmark_root=args.output_root,
            phase="development",
            selected_profiles=None,
            bootstrap=None,
        )
        write_atomic_json(args.output_root / "development-results.json", results)
    print(
        f"{'resume' if resume else 'run'}-fallback-shield-{phase} OK: "
        f"completed={summary.completed_episode_count}/{summary.planned_episode_count} "
        f"executed={summary.executed_episode_count} "
        f"zero_work={summary.zero_work_episode_count}"
    )
    return int(summary.outcome_counts.get("execution_error", 0) > 0)


def _load_rows(
    *,
    sources: Sequence[Any],
    matrix: Any,
    benchmark_root: Path,
    phase: str,
    selected_profiles: Mapping[str, str] | None,
) -> list[FallbackEpisodeMetricRowV1]:
    rows: list[FallbackEpisodeMetricRowV1] = []
    schedule = build_fallback_benchmark_schedule(
        sources, matrix, phase=phase, selected_profiles=selected_profiles
    )
    for spec in schedule:
        store = ClosedLoopEpisodeStore(benchmark_root / spec.relative_directory)
        episode = store.load_episode()
        decisions = [
            load_fallback_decision(
                store.boundary_dir(index) / "fallback-shield-decision.json"
            )
            for index in range(len(episode.boundaries))
        ]
        seconds = load_episode_execution_seconds(
            store.root, expected_episode_execution_id=episode.episode_execution_id
        )
        rows.append(
            fallback_episode_metric_row(episode, decisions, execution_seconds=seconds)
        )
    return rows


def _group_key(key: tuple[str, str, str, str]) -> str:
    return "/".join(key)


def _evaluate_records(
    *,
    sources: Sequence[Any],
    matrix: Any,
    benchmark_root: Path,
    phase: str,
    selected_profiles: Mapping[str, str] | None,
    bootstrap: Any | None,
) -> dict[str, object]:
    rows = _load_rows(
        sources=sources,
        matrix=matrix,
        benchmark_root=benchmark_root,
        phase=phase,
        selected_profiles=selected_profiles,
    )
    grouped = group_fallback_rows(rows)
    aggregates = {
        _group_key(key): aggregate_fallback_metrics(values)
        for key, values in sorted(grouped.items())
    }
    fallback_clean_key = "always_fixed_primary_fallback/none/clean/not_applicable"
    if fallback_clean_key in aggregates:
        fallback_clean_success = float(
            cast(float, aggregates[fallback_clean_key]["success_rate"])
        )
        for key, metrics in aggregates.items():
            if "/clean/" in key:
                enriched = {
                    name: value
                    for name, value in metrics.items()
                    if name != "content_digest"
                }
                enriched["clean_performance_degradation_vs_always_fallback"] = (
                    fallback_clean_success - float(cast(float, metrics["success_rate"]))
                )
                aggregates[key] = {
                    **enriched,
                    "content_digest": content_digest(
                        enriched, context="M4DAggregateMetricsV1"
                    ),
                }
    comparisons: dict[str, object] = {}
    if phase == "final":
        assert bootstrap is not None
        comparisons = _final_comparisons(grouped, bootstrap=bootstrap)
    source_ids = sorted({item.source_trajectory_id for item in rows})
    manifest_payload: dict[str, object] = {
        "episode_count": len(rows),
        "phase": phase,
        "source_trajectory_ids": source_ids,
    }
    manifest_digest = content_digest(manifest_payload, context="M4DResultsManifestV1")
    return {
        "aggregate_metrics": aggregates,
        "benchmark_complete": True,
        "episode_count": len(rows),
        "manifest_digest": manifest_digest,
        "paired_trajectory_bootstrap": comparisons,
        "phase": phase,
        "schema_version": "1.0",
        "source_trajectory_count": len(source_ids),
    }


def _final_comparisons(
    grouped: Mapping[tuple[str, str, str, str], Sequence[FallbackEpisodeMetricRowV1]],
    *,
    bootstrap: Any,
) -> dict[str, object]:
    def values(
        policy: str, scenario: str, domain: str
    ) -> Sequence[FallbackEpisodeMetricRowV1]:
        return grouped[(policy, "none", scenario, domain)]

    primary = values("gated_direct_visual", "fault_injected", "canonical")
    specs = {
        "gated_direct_vs_accept_nominal": values(
            "accept_nominal", "fault_injected", "not_applicable"
        ),
        "gated_direct_vs_always_fallback": values(
            "always_fixed_primary_fallback", "fault_injected", "not_applicable"
        ),
        "gated_direct_vs_ungated_direct": values(
            "ungated_direct_visual", "fault_injected", "canonical"
        ),
        "gated_direct_vs_gated_action_only": values(
            "gated_action_only", "fault_injected", "not_applicable"
        ),
        "gated_direct_vs_gated_distilled": values(
            "gated_distilled_visual", "fault_injected", "canonical"
        ),
        "gated_direct_vs_gated_privileged": values(
            "gated_privileged_structured", "fault_injected", "not_applicable"
        ),
        "gated_direct_canonical_vs_strong_camera": values(
            "gated_direct_visual", "fault_injected", "strong_camera_shift"
        ),
        "gated_direct_canonical_vs_strong_lighting": values(
            "gated_direct_visual", "fault_injected", "strong_lighting_shift"
        ),
    }
    return {
        name: paired_fallback_bootstrap(
            primary,
            comparator,
            resamples=bootstrap.resamples,
            confidence_level=bootstrap.confidence_level,
            seed=bootstrap.seed,
        )
        for name, comparator in specs.items()
    }


def _run_select(args: argparse.Namespace) -> int:
    path = Path(args.development_results).absolute()
    if not path.is_file() or path.is_symlink():
        raise ValueError("M4D development results are missing or unsafe")
    raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, Mapping) or raw.get("phase") != "development":
        raise ValueError("gate selection accepts M4D development results only")
    if raw.get("benchmark_complete") is not True:
        raise ValueError("M4D development benchmark is incomplete")
    if raw.get("source_trajectory_count") != 12 or raw.get("episode_count") != 384:
        raise ValueError(
            "gate selection requires the complete 12-source development set"
        )
    aggregates = raw.get("aggregate_metrics")
    if not isinstance(aggregates, Mapping):
        raise ValueError("M4D development aggregates are missing")
    profiles = load_gate_profiles(args.gate_profiles_config)
    record = select_gate_profiles(
        cast(Mapping[str, Mapping[str, object]], aggregates),
        gate_profiles_digest=profiles.content_digest,
        development_manifest_digest=cast(str, raw["manifest_digest"]),
    )
    write_exclusive_json(args.output, record)
    print(f"select-fallback-gates OK: scorers=4 record={args.output}")
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    archive = load_state_indexed_archive(args.source_archive_dir)
    sources = load_source_plans(args.source_plan_manifest, archive=archive)
    if len(sources) != 60:
        raise ValueError("M4D final evaluation requires exactly 60 sources")
    matrix = load_fallback_selector_matrix(args.m4d_selector_matrix_config)
    selected, _, _ = load_gate_selection(args.gate_selection)
    bootstrap = load_bootstrap_configuration(args.bootstrap_config)
    report = _evaluate_records(
        sources=sources,
        matrix=matrix,
        benchmark_root=args.benchmark_root,
        phase="final",
        selected_profiles=selected,
        bootstrap=bootstrap,
    )
    report["acceptance_targets"] = _acceptance_targets(
        cast(Mapping[str, Mapping[str, object]], report["aggregate_metrics"])
    )
    write_atomic_json(args.output, report)
    print(
        "evaluate-fallback-shield OK: "
        f"episodes={report['episode_count']} report={args.output}"
    )
    return 0


def _metric(
    aggregates: Mapping[str, Mapping[str, object]], key: str, name: str
) -> float:
    value = aggregates[key][name]
    if type(value) not in (int, float):
        raise ValueError(f"M4D target metric {key}/{name} is undefined")
    return float(cast(int | float, value))


def _acceptance_targets(
    aggregates: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    primary_clean = "gated_direct_visual/none/clean/canonical"
    primary_fault = "gated_direct_visual/none/fault_injected/canonical"
    fallback_clean = "always_fixed_primary_fallback/none/clean/not_applicable"
    fallback_fault = "always_fixed_primary_fallback/none/fault_injected/not_applicable"
    nominal_fault = "accept_nominal/none/fault_injected/not_applicable"
    clean_counts = cast(Mapping[str, int], aggregates[primary_clean]["outcome_counts"])
    fault_counts = cast(Mapping[str, int], aggregates[primary_fault]["outcome_counts"])
    nominal_unsuccessful = 1.0 - _metric(aggregates, nominal_fault, "success_rate")
    primary_unsuccessful = 1.0 - _metric(aggregates, primary_fault, "success_rate")
    reduction = (
        0.0
        if nominal_unsuccessful == 0.0
        else (nominal_unsuccessful - primary_unsuccessful) / nominal_unsuccessful
    )
    targets = {
        "clean_execution_errors_zero": clean_counts[EpisodeState.EXECUTION_ERROR.value]
        == 0,
        "clean_false_override_below_0p20": _metric(
            aggregates, primary_clean, "false_override_rate"
        )
        < 0.20,
        "clean_success_within_0p03_of_fallback": _metric(
            aggregates, primary_clean, "success_rate"
        )
        >= _metric(aggregates, fallback_clean, "success_rate") - 0.03,
        "clean_unsafe_zero": clean_counts[EpisodeState.UNSAFE.value] == 0,
        "fault_execution_errors_zero": fault_counts[EpisodeState.EXECUTION_ERROR.value]
        == 0,
        "fault_fallback_rate_below_0p60": _metric(
            aggregates, primary_fault, "fallback_invocation_rate"
        )
        < 0.60,
        "fault_override_recall_at_least_0p60": _metric(
            aggregates, primary_fault, "injected_fault_override_recall"
        )
        >= 0.60,
        "fault_success_improvement_at_least_0p10": _metric(
            aggregates, primary_fault, "success_rate"
        )
        >= _metric(aggregates, nominal_fault, "success_rate") + 0.10,
        "fault_success_within_0p05_of_fallback": _metric(
            aggregates, primary_fault, "success_rate"
        )
        >= _metric(aggregates, fallback_fault, "success_rate") - 0.05,
        "fault_unsuccessful_relative_reduction_at_least_0p40": reduction >= 0.40,
    }
    for domain, suffix in (
        ("strong_camera_shift", "strong_camera"),
        ("strong_lighting_shift", "strong_lighting"),
    ):
        key = f"gated_direct_visual/none/fault_injected/{domain}"
        targets[f"{suffix}_success_drop_at_most_0p10"] = (
            _metric(aggregates, key, "success_rate")
            >= _metric(aggregates, primary_fault, "success_rate") - 0.10
        )
        targets[f"{suffix}_fault_recovery_drop_at_most_0p15"] = (
            _metric(aggregates, key, "fault_recovery_rate")
            >= _metric(aggregates, primary_fault, "fault_recovery_rate") - 0.15
        )
    return {
        "all_targets_passed": all(targets.values()),
        "target_results": targets,
        "targets_are_predeclared_not_assumed": True,
    }


def _run_inspect(args: argparse.Namespace) -> int:
    store = ClosedLoopEpisodeStore(args.episode_dir)
    episode = store.load_episode()
    decisions = [
        load_fallback_decision(
            store.boundary_dir(index) / "fallback-shield-decision.json"
        )
        for index in range(len(episode.boundaries))
    ]
    payload = {
        "decision_count": len(decisions),
        "episode_execution_id": episode.episode_execution_id,
        "executed_control_steps": episode.executed_control_steps,
        "intervention_types": [item.intervention_type.value for item in decisions],
        "nominal_candidate_ids": [item.nominal_candidate_id for item in decisions],
        "policy_id": None if not decisions else decisions[0].policy_id,
        "scenario": None if not decisions else decisions[0].scenario,
        "schema_version": "1.0",
        "selected_candidate_ids": [item.selected_candidate_id for item in decisions],
        "source_trajectory_id": episode.source_trajectory_id,
        "state": episode.state.value,
        "visual_domain": episode.visual_domain,
    }
    if args.output is not None:
        write_atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _run_diagnose(args: argparse.Namespace) -> int:
    episodes, decisions, _ = _load_benchmark_records(args)
    by_episode: defaultdict[str, list[Any]] = defaultdict(list)
    for decision in decisions:
        by_episode[decision.episode_execution_id].append(decision)
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    selected_definitions: dict[tuple[str, str, int, str], int] = {}
    family_by_ordinal = {
        0: "additive_gaussian_noise",
        1: "constant_bias",
        2: "additive_gaussian_noise",
        3: "segment_hold",
        4: "additive_gaussian_noise",
        5: "constant_bias",
        6: "segment_hold",
        7: "local_temporal_permutation",
    }
    for episode in episodes:
        values = sorted(
            by_episode[episode.episode_execution_id],
            key=lambda item: item.decision_ordinal,
        )
        interventions = 0
        first: int | None = None
        primary_risks: list[float] = []
        margins: list[float] = []
        selected_ordinals: list[int] = []
        decision_records: list[dict[str, object]] = []
        store = ClosedLoopEpisodeStore(
            args.benchmark_root
            / episode.selector_id
            / episode.visual_domain
            / episode.source_trajectory_id
        )
        for item in values:
            primary = item.ordered_candidate_ids[0]
            candidate_pool = store.load_pool(item.decision_ordinal)
            selected_ordinal = next(
                candidate.definition_ordinal
                for candidate in candidate_pool.candidates
                if candidate.candidate_id == item.selected_candidate_id
            )
            selected_ordinals.append(selected_ordinal)
            selected_definitions[
                (
                    episode.source_trajectory_id,
                    episode.visual_domain,
                    item.decision_ordinal,
                    episode.selector_id,
                )
            ] = selected_ordinal
            if item.selected_candidate_id != primary:
                interventions += 1
                if first is None:
                    first = item.decision_ordinal
            if item.selected_predicted_failure_probability is not None:
                scores = dict(item.selector_scores)
                primary_risks.append(scores[primary])
                margins.append(scores[primary] - scores[item.selected_candidate_id])
            decision_records.append(
                {
                    "decision_ordinal": item.decision_ordinal,
                    "nominal_plan_index": item.nominal_plan_index,
                    "selected_candidate_definition_ordinal": selected_ordinal,
                    "selected_candidate_family": family_by_ordinal[selected_ordinal],
                    "selected_vs_primary_risk_margin": (
                        None
                        if item.selected_predicted_failure_probability is None
                        else dict(item.selector_scores)[primary]
                        - dict(item.selector_scores)[item.selected_candidate_id]
                    ),
                }
            )
        family_counts = {
            family: sum(
                family_by_ordinal[value] == family for value in selected_ordinals
            )
            for family in sorted(set(family_by_ordinal.values()))
        }
        grouped[f"{episode.selector_id}/{episode.visual_domain}"].append(
            {
                "candidate_family_distribution": family_counts,
                "cumulative_interventions": interventions,
                "decision_records": decision_records,
                "decision_count": len(values),
                "first_intervention_ordinal": first,
                "horizon_exhausted": episode.state is EpisodeState.HORIZON_EXHAUSTED,
                "late_intervention_count": sum(
                    item.selected_candidate_id != item.ordered_candidate_ids[0]
                    and item.decision_ordinal >= max(0, len(values) - 3)
                    for item in values
                ),
                "late_decision_count": min(3, len(values)),
                "mean_fixed_primary_risk": (
                    None if not primary_risks else float(np.mean(primary_risks))
                ),
                "mean_selected_vs_primary_margin": (
                    None if not margins else float(np.mean(margins))
                ),
                "repeated_selected_definition_count": (
                    len(selected_ordinals) - len(set(selected_ordinals))
                ),
                "source_trajectory_id": episode.source_trajectory_id,
            }
        )
    diagnostics: dict[str, object] = {}
    for key, values in sorted(grouped.items()):
        total_decisions = sum(cast(int, item["decision_count"]) for item in values)
        total_interventions = sum(
            cast(int, item["cumulative_interventions"]) for item in values
        )
        total_late_decisions = sum(
            cast(int, item["late_decision_count"]) for item in values
        )
        diagnostics[key] = {
            "ensemble_uncertainty": {
                "reason": "not persisted by the accepted M4C decision schema",
                "status": "unavailable",
            },
            "episode_count": len(values),
            "horizon_exhaustion_association_by_intervention_count": _diagnostic_bins(
                values, "cumulative_interventions"
            ),
            "horizon_exhaustion_association_by_first_intervention": _first_bins(values),
            "intervention_rate": (
                0.0 if total_decisions == 0 else total_interventions / total_decisions
            ),
            "trajectory_bootstrap_intervention_rate_interval": (
                _trajectory_rate_interval(values)
            ),
            "late_episode_intervention_rate": (
                0.0
                if total_late_decisions == 0
                else sum(cast(int, item["late_intervention_count"]) for item in values)
                / total_late_decisions
            ),
            "trajectory_records": values,
        }
    payload = {
        "association_not_causal_proof": True,
        "candidate_outcomes_used_as_inputs": False,
        "diagnostics": diagnostics,
        "direct_distilled_and_action_visual_disagreement": _m4c_disagreement(
            selected_definitions
        ),
        "episode_count": len(episodes),
        "schema_version": "1.0",
    }
    write_atomic_json(args.output, payload)
    print(
        f"diagnose-m4c-interventions OK: episodes={len(episodes)} report={args.output}"
    )
    return 0


def _trajectory_rate_interval(
    values: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    rates = np.asarray(
        [
            0.0
            if cast(int, item["decision_count"]) == 0
            else cast(int, item["cumulative_interventions"])
            / cast(int, item["decision_count"])
            for item in values
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(271828)
    samples = np.mean(
        rates[generator.integers(0, rates.size, size=(2000, rates.size))],
        axis=1,
    )
    return {
        "confidence_level": 0.95,
        "estimate": float(np.mean(rates)),
        "lower": float(np.quantile(samples, 0.025, method="linear")),
        "resamples": 2000,
        "sampling_unit": "source_trajectory_v1",
        "upper": float(np.quantile(samples, 0.975, method="linear")),
    }


def _m4c_disagreement(
    selected: Mapping[tuple[str, str, int, str], int],
) -> Mapping[str, object]:
    def compare(left: str, right: str, *, domain: str) -> Mapping[str, object]:
        matches = 0
        disagreements = 0
        for (source, observed_domain, ordinal, selector), value in selected.items():
            if selector != left or observed_domain != domain:
                continue
            right_domain = "not_applicable" if "visual" not in right else domain
            other = selected.get((source, right_domain, ordinal, right))
            if other is not None:
                matches += 1
                disagreements += int(value != other)
        return {
            "compared_decision_ordinals": matches,
            "definition_disagreement_rate": (
                None if matches == 0 else disagreements / matches
            ),
            "physical_trajectory_divergence_caveat": (
                "association across decision ordinal; selector states may differ"
            ),
        }

    return {
        "action_only_vs_direct_visual_canonical": compare(
            "direct_visual_ensemble_v1",
            "action_only_ensemble_v1",
            domain="canonical",
        ),
        "direct_vs_distilled_canonical": compare(
            "direct_visual_ensemble_v1",
            "distilled_visual_ensemble_v1",
            domain="canonical",
        ),
        "status": "retrospective_association_not_causal",
    }


def _diagnostic_bins(
    values: Sequence[Mapping[str, object]], name: str
) -> Mapping[str, object]:
    grouped: defaultdict[int, list[bool]] = defaultdict(list)
    for item in values:
        grouped[int(cast(int, item[name]))].append(bool(item["horizon_exhausted"]))
    return {
        str(key): {
            "episode_count": len(items),
            "horizon_exhaustion_rate": sum(items) / len(items),
        }
        for key, items in sorted(grouped.items())
    }


def _first_bins(values: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    grouped: defaultdict[str, list[bool]] = defaultdict(list)
    for item in values:
        ordinal = item["first_intervention_ordinal"]
        key = "none" if ordinal is None else str(ordinal)
        grouped[key].append(bool(item["horizon_exhausted"]))
    return {
        key: {
            "episode_count": len(items),
            "horizon_exhaustion_rate": sum(items) / len(items),
        }
        for key, items in sorted(grouped.items())
    }


def run_m4d_command(args: argparse.Namespace) -> int | None:
    """Dispatch one M4D command or return ``None`` for another milestone."""

    if args.command not in M4D_COMMANDS:
        return None
    try:
        if args.command == "diagnose-m4c-interventions":
            return _run_diagnose(args)
        if args.command == "prepare-m4d-source-plans":
            return _run_prepare(args)
        if args.command == "run-fallback-shield-development":
            return _run_schedule(args, phase="development", resume=False)
        if args.command == "select-fallback-gates":
            return _run_select(args)
        if args.command == "run-fallback-shield-benchmark":
            return _run_schedule(args, phase="final", resume=False)
        if args.command == "resume-fallback-shield-benchmark":
            return _run_schedule(args, phase="final", resume=True)
        if args.command == "evaluate-fallback-shield":
            return _run_evaluate(args)
        if args.command == "inspect-fallback-shield-episode":
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
    raise AssertionError(f"unhandled M4D command {args.command!r}")


__all__ = ["M4D_COMMANDS", "add_m4d_subparsers", "run_m4d_command"]

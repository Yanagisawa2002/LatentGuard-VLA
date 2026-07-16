"""Compact outcome-safe evaluation summaries for M3C selector suites."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, cast

from latentguard.selection.bootstrap import (
    PairedTrajectoryBootstrapReportV1,
    paired_trajectory_selector_bootstrap,
)
from latentguard.selection.metrics import (
    CandidateOutcomeV1,
    SelectorComparisonV1,
    SelectorMetricReportV1,
    compare_selector_to_random,
    evaluate_selector_metrics,
)
from latentguard.selection.models import SelectorDecisionV1


class SelectionReportingError(ValueError):
    """Raised when a compact selector suite cannot be compared safely."""


def _fail(context: str, reason: str) -> NoReturn:
    raise SelectionReportingError(f"{context}: {reason}")


def _comparison_to_dict(value: SelectorComparisonV1) -> dict[str, object]:
    return {
        "absolute_failure_difference": value.absolute_failure_difference,
        "absolute_success_improvement": value.absolute_success_improvement,
        "oracle_regret_difference": value.oracle_regret_difference,
        "random_selector_id": value.random_selector_id,
        "relative_failure_reduction": value.relative_failure_reduction,
        "selector_id": value.selector_id,
    }


def _target_check(
    observed: float | int | None,
    *,
    operator: str,
    target: float | int,
    undefined_reason: str,
) -> dict[str, object]:
    if observed is None:
        return {
            "observed": None,
            "operator": operator,
            "status": "undefined",
            "target": target,
            "undefined_reason": undefined_reason,
        }
    if operator == ">=":
        passed = observed >= target
    elif operator == "<=":
        passed = observed <= target
    elif operator == "==":
        passed = observed == target
    else:  # pragma: no cover - all callers are fixed below
        _fail("acceptance target", "unsupported comparison operator")
    return {
        "observed": observed,
        "operator": operator,
        "status": "passed" if passed else "failed",
        "target": target,
        "undefined_reason": None,
    }


def _bootstrap_interval(
    suite: SelectorEvaluationSuiteV1,
    *,
    first_selector_id: str,
    second_selector_id: str,
    metric: str,
) -> dict[str, object]:
    matches = [
        report
        for report in suite.bootstrap_reports
        if report.first_selector_id == first_selector_id
        and report.second_selector_id == second_selector_id
    ]
    if len(matches) != 1:
        _fail("bootstrap interpretation", "required selector comparison is absent")
    intervals = [item for item in matches[0].intervals if item.metric == metric]
    if len(intervals) != 1:
        _fail("bootstrap interpretation", "required metric interval is absent")
    return intervals[0].to_dict()


def build_predeclared_interpretation(
    suite: SelectorEvaluationSuiteV1,
    *,
    temporal_selector_id: str,
    joint_selector_id: str,
    random_selector_id: str,
    coverage_70_selector_id: str,
    strong_replay_evidence_verified: bool,
) -> dict[str, object]:
    """Evaluate the frozen M3C claims and quality targets without retuning."""

    if not isinstance(suite, SelectorEvaluationSuiteV1):
        _fail("predeclared interpretation", "expected selector evaluation suite")
    if (
        temporal_selector_id != suite.temporal_selector_id
        or joint_selector_id != suite.joint_selector_id
        or random_selector_id != suite.random_selector_id
    ):
        _fail("predeclared interpretation", "primary selector identities changed")
    if type(strong_replay_evidence_verified) is not bool:
        _fail("strong_replay_evidence_verified", "expected boolean")
    reports = {item.selector_id: item for item in suite.selector_reports}
    required = {
        temporal_selector_id,
        joint_selector_id,
        random_selector_id,
        coverage_70_selector_id,
    }
    if not required.issubset(reports):
        _fail("predeclared interpretation", "required selector report is absent")
    temporal = reports[temporal_selector_id]
    random = reports[random_selector_id]
    coverage_70 = reports[coverage_70_selector_id]
    solvable_improvement = (
        None
        if temporal.solvable_success_rate.value is None
        or random.solvable_success_rate.value is None
        else temporal.solvable_success_rate.value - random.solvable_success_rate.value
    )
    temporal_random_comparisons = [
        item
        for item in suite.comparisons_to_random
        if item.selector_id == temporal_selector_id
    ]
    if len(temporal_random_comparisons) != 1:
        _fail("predeclared interpretation", "temporal/random comparison is absent")
    temporal_random = temporal_random_comparisons[0]
    random_failure = random.task_failure_rate.value
    retained_failure_target = None if random_failure is None else 0.5 * random_failure
    retained_failure_check = (
        {
            "observed": coverage_70.task_failure_rate.value,
            "operator": "<=",
            "status": "undefined",
            "target": retained_failure_target,
            "undefined_reason": "random or retained failure rate is undefined",
        }
        if retained_failure_target is None
        or coverage_70.task_failure_rate.value is None
        else _target_check(
            coverage_70.task_failure_rate.value,
            operator="<=",
            target=retained_failure_target,
            undefined_reason="random or retained failure rate is undefined",
        )
    )
    checks = {
        "all_replay_evidence_strong_and_simulator_verified": _target_check(
            int(strong_replay_evidence_verified),
            operator="==",
            target=1,
            undefined_reason="strong replay gate was not evaluated",
        ),
        "coverage_70_retained_failure_at_most_half_random": retained_failure_check,
        "temporal_oracle_success_regret_on_solvable_groups": _target_check(
            temporal.oracle_success_regret,
            operator="<=",
            target=0.10,
            undefined_reason="no executed solvable temporal groups",
        ),
        "temporal_pairwise_success_failure_concordance": _target_check(
            temporal.ranking.pairwise_concordance,
            operator=">=",
            target=0.75,
            undefined_reason="no success/failure candidate pairs",
        ),
        "temporal_relative_task_failure_reduction_vs_random": _target_check(
            temporal_random.relative_failure_reduction,
            operator=">=",
            target=0.30,
            undefined_reason="random task-failure rate is zero or undefined",
        ),
        "temporal_solvable_success_improvement_vs_random": _target_check(
            solvable_improvement,
            operator=">=",
            target=0.08,
            undefined_reason="no comparable executed solvable groups",
        ),
        "unexplained_execution_error_count": _target_check(
            sum(item.execution_error_count for item in suite.selector_reports),
            operator="==",
            target=0,
            undefined_reason="execution inventory is unavailable",
        ),
    }
    temporal_random_interval = _bootstrap_interval(
        suite,
        first_selector_id=temporal_selector_id,
        second_selector_id=random_selector_id,
        metric="selected_success_rate",
    )
    temporal_joint_interval = _bootstrap_interval(
        suite,
        first_selector_id=temporal_selector_id,
        second_selector_id=joint_selector_id,
        metric="selected_success_rate",
    )

    def bounds(value: Mapping[str, object]) -> tuple[float | None, float | None]:
        lower = value["confidence_lower"]
        upper = value["confidence_upper"]
        return (
            float(lower) if isinstance(lower, (int, float)) else None,
            float(upper) if isinstance(upper, (int, float)) else None,
        )

    random_lower, _random_upper = bounds(temporal_random_interval)
    joint_lower, joint_upper = bounds(temporal_joint_interval)
    target_statuses = {str(item["status"]) for item in checks.values()}
    targets_met = target_statuses == {"passed"}
    joint_distinguishable = (
        None
        if joint_lower is None or joint_upper is None
        else joint_lower > 0.0 or joint_upper < 0.0
    )
    temporal_superiority_supported = None if joint_lower is None else joint_lower > 0.0
    return {
        "acceptance_targets": checks,
        "coverage_70_actual": coverage_70.coverage.to_dict(),
        "full_outcome_tuning_permitted": False,
        "joint_mlp_statistically_distinguishable_on_selected_success": (
            joint_distinguishable
        ),
        "latency_winner_selected_from_m3c_outcomes": False,
        "online_selection_quality_targets_met": targets_met,
        "predeclared_efficiency_challenger": joint_selector_id,
        "predeclared_primary_selector": temporal_selector_id,
        "result_disposition": (
            "quality_targets_met_without_full_outcome_tuning"
            if targets_met
            else (
                "negative_result_preserved_engineering_may_be_complete_"
                "no_online_selection_claim"
            )
        ),
        "schema_version": "1.0",
        "temporal_improved_over_random_point_estimate": (
            None if solvable_improvement is None else solvable_improvement > 0.0
        ),
        "temporal_superiority_claim_supported": temporal_superiority_supported,
        "temporal_vs_joint_selected_success_interval": temporal_joint_interval,
        "temporal_vs_random_selected_success_interval": temporal_random_interval,
        "temporal_vs_random_success_interval_excludes_zero_in_favor": (
            None if random_lower is None else random_lower > 0.0
        ),
    }


def _review_number(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_human_readable_review(
    suite: SelectorEvaluationSuiteV1,
    interpretation: Mapping[str, object],
    *,
    mode: str,
    git_sha: str,
    selection_manifest_digest: str,
    candidate_pool_digest: str,
    replay_evidence_digest: str,
    full_pool_outcome_digest: str,
    cpu_latency_report_digest: str,
    gpu_latency_report_digest: str,
) -> str:
    """Render the compact, deterministic M3C review without raw model data."""

    if not isinstance(suite, SelectorEvaluationSuiteV1):
        _fail("human review", "expected SelectorEvaluationSuiteV1")
    if not isinstance(interpretation, Mapping):
        _fail("human review", "expected interpretation mapping")
    text_fields = {
        "mode": mode,
        "git_sha": git_sha,
        "selection_manifest_digest": selection_manifest_digest,
        "candidate_pool_digest": candidate_pool_digest,
        "replay_evidence_digest": replay_evidence_digest,
        "full_pool_outcome_digest": full_pool_outcome_digest,
        "cpu_latency_report_digest": cpu_latency_report_digest,
        "gpu_latency_report_digest": gpu_latency_report_digest,
    }
    if any(not isinstance(value, str) or not value for value in text_fields.values()):
        _fail("human review", "identity fields must be non-empty text")
    lines = [
        "# M3C blind candidate-selection review",
        "",
        f"- Mode: `{mode}`",
        f"- Git SHA: `{git_sha}`",
        f"- Source trajectories: {suite.source_trajectory_count}",
        f"- Candidate groups: {suite.group_count}",
        f"- Complete candidate outcomes: {suite.outcome_count}",
        f"- Selection manifest: `{selection_manifest_digest}`",
        f"- Candidate pool: `{candidate_pool_digest}`",
        f"- Semantic replay evidence: `{replay_evidence_digest}`",
        f"- Full-pool outcome: `{full_pool_outcome_digest}`",
        f"- CPU latency report: `{cpu_latency_report_digest}`",
        f"- GPU latency report: `{gpu_latency_report_digest}`",
        "",
        "## Selector outcomes",
        "",
        "| Selector | Executed | Abstained | Coverage | Success | Failure | "
        "Solvable success | Mixed failure | Oracle regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for selector_report in suite.selector_reports:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{selector_report.selector_id}`",
                    str(selector_report.executed_count),
                    str(selector_report.abstained_count),
                    _review_number(selector_report.coverage.value),
                    _review_number(selector_report.selected_success_rate.value),
                    _review_number(selector_report.task_failure_rate.value),
                    _review_number(selector_report.solvable_success_rate.value),
                    _review_number(selector_report.mixed_failure_rate.value),
                    _review_number(selector_report.oracle_success_regret),
                )
            )
            + " |"
        )
    lines.extend(("", "## Predeclared acceptance checks", ""))
    raw_checks = interpretation.get("acceptance_targets")
    if not isinstance(raw_checks, Mapping):
        _fail("human review", "acceptance target inventory is absent")
    for name in sorted(raw_checks):
        check = raw_checks[name]
        if not isinstance(name, str) or not isinstance(check, Mapping):
            _fail("human review", "acceptance target entry is invalid")
        lines.append(
            f"- `{name}`: **{_review_number(check.get('status'))}**; observed "
            f"{_review_number(check.get('observed'))} "
            f"{_review_number(check.get('operator'))} "
            f"{_review_number(check.get('target'))}."
        )
    lines.extend(("", "## Trajectory-level paired bootstrap", ""))
    for bootstrap_report in suite.bootstrap_reports:
        lines.append(
            f"- `{bootstrap_report.first_selector_id}` minus "
            f"`{bootstrap_report.second_selector_id}` "
            f"({bootstrap_report.requested_resamples} resamples):"
        )
        for interval in bootstrap_report.intervals:
            value = interval.to_dict()
            lines.append(
                f"  - `{interval.metric}`: estimate "
                f"{_review_number(value['observed_difference'])}, "
                f"CI [{_review_number(value['confidence_lower'])}, "
                f"{_review_number(value['confidence_upper'])}]."
            )
    disposition = interpretation.get("result_disposition")
    lines.extend(
        (
            "",
            "## Interpretation and limitations",
            "",
            f"- Result disposition: `{_review_number(disposition)}`.",
            "- The temporal verifier remained the predeclared primary selector; "
            "the joint MLP remained the efficiency challenger.",
            "- Full M3C outcomes were not used for model, threshold, candidate, "
            "or selector tuning.",
            "- Abstention is reported as reduced coverage and never converted to "
            "success or task failure.",
            "- This fixed post-candidate continuation experiment is not "
            "receding-horizon control.",
            "- No VLM, LangMani, image/language model, source-action fallback, "
            "multi-GPU path, or real-robot claim was used.",
            "- Claims remain limited to structured-state PickCube simulator "
            "selection under the frozen M3B/M3C contracts.",
            "",
        )
    )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SelectorEvaluationSuiteV1:
    """Metrics, random comparisons, and required trajectory-bootstrap intervals."""

    random_selector_id: str
    temporal_selector_id: str
    joint_selector_id: str
    selector_reports: tuple[SelectorMetricReportV1, ...]
    comparisons_to_random: tuple[SelectorComparisonV1, ...]
    bootstrap_reports: tuple[PairedTrajectoryBootstrapReportV1, ...]
    outcome_count: int
    group_count: int
    source_trajectory_count: int
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Validate exact report identities and complete aggregate counts."""

        for name in ("random_selector_id", "temporal_selector_id", "joint_selector_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                _fail(f"SelectorEvaluationSuiteV1.{name}", "invalid text")
        reports = tuple(self.selector_reports)
        report_ids = tuple(item.selector_id for item in reports)
        if not reports or report_ids != tuple(sorted(report_ids)):
            _fail("SelectorEvaluationSuiteV1.selector_reports", "must be ID sorted")
        if len(set(report_ids)) != len(report_ids):
            _fail("SelectorEvaluationSuiteV1.selector_reports", "duplicate selector")
        if self.random_selector_id not in report_ids:
            _fail("SelectorEvaluationSuiteV1.random_selector_id", "missing report")
        if (
            self.temporal_selector_id not in report_ids
            or self.joint_selector_id not in report_ids
        ):
            _fail("SelectorEvaluationSuiteV1", "primary selectors are missing")
        comparisons = tuple(self.comparisons_to_random)
        if tuple(item.selector_id for item in comparisons) != tuple(
            sorted(item.selector_id for item in comparisons)
        ):
            _fail("SelectorEvaluationSuiteV1.comparisons_to_random", "must be sorted")
        if any(
            item.random_selector_id != self.random_selector_id for item in comparisons
        ):
            _fail(
                "SelectorEvaluationSuiteV1.comparisons_to_random", "random ID differs"
            )
        bootstraps = tuple(self.bootstrap_reports)
        pairs = tuple(
            (item.first_selector_id, item.second_selector_id) for item in bootstraps
        )
        if len(pairs) != len(set(pairs)):
            _fail("SelectorEvaluationSuiteV1.bootstrap_reports", "duplicate comparison")
        for name in ("outcome_count", "group_count", "source_trajectory_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                _fail(f"SelectorEvaluationSuiteV1.{name}", "expected positive count")
        if any(item.group_count != self.group_count for item in reports):
            _fail("SelectorEvaluationSuiteV1.selector_reports", "group count differs")
        if self.schema_version != "1.0":
            _fail("SelectorEvaluationSuiteV1.schema_version", "unsupported version")
        object.__setattr__(self, "selector_reports", reports)
        object.__setattr__(self, "comparisons_to_random", comparisons)
        object.__setattr__(self, "bootstrap_reports", bootstraps)

    def to_dict(self) -> dict[str, object]:
        """Return compact JSON-native content with all metric denominators."""

        return {
            "bootstrap_reports": [item.to_dict() for item in self.bootstrap_reports],
            "comparisons_to_random": [
                _comparison_to_dict(item) for item in self.comparisons_to_random
            ],
            "group_count": self.group_count,
            "joint_selector_id": self.joint_selector_id,
            "outcome_count": self.outcome_count,
            "random_selector_id": self.random_selector_id,
            "schema_version": self.schema_version,
            "selector_reports": [item.to_dict() for item in self.selector_reports],
            "source_trajectory_count": self.source_trajectory_count,
            "temporal_selector_id": self.temporal_selector_id,
        }


def evaluate_selector_suite(
    selector_decisions: Mapping[str, Sequence[SelectorDecisionV1]],
    complete_outcomes: Sequence[CandidateOutcomeV1],
    *,
    random_selector_id: str,
    learned_selector_ids: Sequence[str],
    temporal_selector_id: str,
    joint_selector_id: str,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 1729,
    confidence_level: float = 0.95,
) -> SelectorEvaluationSuiteV1:
    """Evaluate identical pools and predeclared paired selector comparisons."""

    if not isinstance(selector_decisions, Mapping) or not selector_decisions:
        _fail("selector_decisions", "expected a non-empty mapping")
    outcomes = tuple(complete_outcomes)
    if not outcomes or any(
        not isinstance(item, CandidateOutcomeV1) for item in outcomes
    ):
        _fail("complete_outcomes", "expected complete CandidateOutcomeV1 inventory")
    decisions_by_id: dict[str, tuple[SelectorDecisionV1, ...]] = {}
    expected_group_ids: set[str] | None = None
    for selector_id, raw_decisions in selector_decisions.items():
        if not isinstance(selector_id, str) or not selector_id:
            _fail("selector_decisions", "invalid selector key")
        decisions = tuple(raw_decisions)
        if not decisions or any(
            not isinstance(item, SelectorDecisionV1) for item in decisions
        ):
            _fail(f"selector_decisions.{selector_id}", "invalid decision inventory")
        if {item.selector_id for item in decisions} != {selector_id}:
            _fail(f"selector_decisions.{selector_id}", "key and embedded ID differ")
        group_ids = {item.group_id for item in decisions}
        if expected_group_ids is None:
            expected_group_ids = group_ids
        elif group_ids != expected_group_ids:
            _fail("selector_decisions", "selectors do not cover identical groups")
        decisions_by_id[selector_id] = decisions
    if random_selector_id not in decisions_by_id:
        _fail("random_selector_id", "selector is absent")
    learned_ids = tuple(learned_selector_ids)
    if not learned_ids or len(learned_ids) != len(set(learned_ids)):
        _fail("learned_selector_ids", "expected unique learned selectors")
    if any(item not in decisions_by_id for item in learned_ids):
        _fail("learned_selector_ids", "selector is absent")
    if temporal_selector_id not in learned_ids or joint_selector_id not in learned_ids:
        _fail("learned_selector_ids", "temporal and joint selectors must be learned")
    reports = {
        selector_id: evaluate_selector_metrics(cast(Sequence[Any], decisions), outcomes)
        for selector_id, decisions in decisions_by_id.items()
    }
    random_report = reports[random_selector_id]
    comparisons = tuple(
        compare_selector_to_random(reports[selector_id], random_report)
        for selector_id in sorted(reports)
        if selector_id != random_selector_id
    )
    bootstrap_pairs: list[tuple[str, str]] = [
        (selector_id, random_selector_id) for selector_id in learned_ids
    ]
    if (temporal_selector_id, joint_selector_id) not in bootstrap_pairs:
        bootstrap_pairs.append((temporal_selector_id, joint_selector_id))
    bootstraps = tuple(
        paired_trajectory_selector_bootstrap(
            decisions_by_id[first_id],
            decisions_by_id[second_id],
            outcomes,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + ordinal,
            confidence_level=confidence_level,
        )
        for ordinal, (first_id, second_id) in enumerate(bootstrap_pairs)
    )
    assert expected_group_ids is not None
    trajectories = {
        item.source_trajectory_id
        for item in outcomes
        if item.group_id in expected_group_ids
    }
    return SelectorEvaluationSuiteV1(
        random_selector_id=random_selector_id,
        temporal_selector_id=temporal_selector_id,
        joint_selector_id=joint_selector_id,
        selector_reports=tuple(reports[item] for item in sorted(reports)),
        comparisons_to_random=comparisons,
        bootstrap_reports=bootstraps,
        outcome_count=len(outcomes),
        group_count=len(expected_group_ids),
        source_trajectory_count=len(trajectories),
    )


__all__ = [
    "SelectionReportingError",
    "SelectorEvaluationSuiteV1",
    "build_predeclared_interpretation",
    "evaluate_selector_suite",
    "render_human_readable_review",
]

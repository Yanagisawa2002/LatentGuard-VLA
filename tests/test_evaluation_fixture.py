"""CPU-only tests for the synthetic deterministic M2A fixture evaluator."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from latentguard.corruptions.models import (
    CorruptedActionProposal,
    compute_proposal_identifier,
)
from latentguard.evaluation.base import EvaluatorConfigurationError
from latentguard.evaluation.fixture import (
    DETERMINISTIC_FIXTURE_EVALUATOR_ID,
    DETERMINISTIC_FIXTURE_EVALUATOR_VERSION,
    DeterministicFixtureEvaluator,
    FixtureConfigurationError,
    FixtureEvaluationError,
    create_deterministic_fixture_evaluator,
)
from latentguard.evaluation.models import EvaluationStatus
from latentguard.evaluation.registry import load_evaluator_configuration
from latentguard.evaluation.validation import validate_evaluation_evidence
from latentguard.models import ActionChunk, LabelSource, LabelStrength

_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "evaluation" / "m2a-fixture.json"
)


def _configuration(**updates: object) -> dict[str, object]:
    configuration = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    configuration.update(updates)
    return configuration


def _proposal(actions: np.ndarray[Any, Any] | None = None) -> CorruptedActionProposal:
    selected = (
        np.array(
            [[0.0, 0.5], [0.5, 1.0], [1.0, 1.5]],
            dtype=np.float32,
        )
        if actions is None
        else actions
    )
    parameters = {"target_indices": (0, 1), "bias": (0.0, 0.0)}
    proposal_id = compute_proposal_identifier(
        source_episode_id="episode-fixture",
        source_candidate_id="candidate-fixture",
        corruption_name="constant_bias",
        resolved_parameters=parameters,
        seed=314159,
        generation_ordinal=0,
    )
    return CorruptedActionProposal(
        proposal_id=proposal_id,
        source_episode_id="episode-fixture",
        source_candidate_id="candidate-fixture",
        source_policy_id="policy-fixture",
        source_task_id="task-fixture",
        split_group_id="episode-fixture",
        transformed_action=ActionChunk(
            actions=selected,
            coordinate_frame="fixture-frame",
            control_period_s=0.1,
        ),
        corruption_type="constant_bias",
        resolved_parameters=parameters,
        seed=314159,
        generation_ordinal=0,
        notes="unlabeled input proposal",
    )


def _evaluate(
    evaluator: DeterministicFixtureEvaluator,
    proposal: CorruptedActionProposal | None = None,
    *,
    seed: int = 271828,
    attempt: int = 0,
):
    return evaluator.evaluate(
        proposal or _proposal(),
        source_dataset_id="sha256:m2a-source",
        evaluation_seed=seed,
        attempt_ordinal=attempt,
    )


def test_checked_fixture_configuration_loads_and_resolves_canonically() -> None:
    loaded = load_evaluator_configuration(_CONFIG_PATH)
    evaluator = create_deterministic_fixture_evaluator(loaded)

    assert evaluator.evaluator_id == DETERMINISTIC_FIXTURE_EVALUATOR_ID
    assert evaluator.evaluator_version == DETERMINISTIC_FIXTURE_EVALUATOR_VERSION
    assert evaluator.resolved_configuration() == {
        "schema_version": "1.0",
        "success_mean_abs_threshold": 1.0,
        "unsafe_max_abs_threshold": 10.0,
        "skip_mean_abs_below": None,
        "indeterminate_temporal_variation_below": None,
        "execution_error_max_abs_above": None,
    }
    with pytest.raises(TypeError):
        evaluator.resolved_configuration()["schema_version"] = "changed"  # type: ignore[index]


def test_fixture_is_deterministic_and_does_not_mutate_proposal() -> None:
    evaluator = create_deterministic_fixture_evaluator(_configuration())
    proposal = _proposal()
    before = proposal.transformed_action.actions.copy()

    first = _evaluate(evaluator, proposal)
    second = _evaluate(evaluator, proposal)

    assert first.evidence_id == second.evidence_id
    assert first.status is second.status
    assert first.metrics == second.metrics
    assert first.progress_after == second.progress_after
    np.testing.assert_array_equal(proposal.transformed_action.actions, before)
    assert proposal.transformed_action.actions.flags.writeable is False
    validate_evaluation_evidence(first)


def test_fixture_conclusive_statistics_progress_and_weak_metadata() -> None:
    evaluator = create_deterministic_fixture_evaluator(_configuration())

    evidence = _evaluate(evaluator)

    assert evidence.status is EvaluationStatus.CONCLUSIVE
    assert evidence.metrics == {
        "mean_absolute_magnitude": pytest.approx(0.75),
        "maximum_absolute_magnitude": pytest.approx(1.5),
        "mean_temporal_variation": pytest.approx(0.5),
    }
    assert evidence.success is True
    assert evidence.unsafe is False
    assert evidence.progress_before == 0.0
    assert evidence.progress_after == pytest.approx(1.0 / 1.75)
    assert evidence.progress_delta == evidence.progress_after
    assert evidence.label_source is LabelSource.DETERMINISTIC_EVALUATOR
    assert evidence.label_strength is LabelStrength.WEAK
    assert evidence.simulator_replay_verified is False
    assert evidence.replayed_control_steps == 0
    assert evidence.failure_events == ()
    assert evidence.artifact_references == ()
    assert evidence.notes is not None
    assert "fixture-only" in evidence.notes
    assert "non-physical" in evidence.notes
    assert "not simulator replay" in evidence.notes


def test_fixture_threshold_boundaries_are_inclusive() -> None:
    proposal = _proposal(np.ones((3, 2), dtype=np.float32))
    evaluator = create_deterministic_fixture_evaluator(
        _configuration(
            success_mean_abs_threshold=1.0,
            unsafe_max_abs_threshold=1.0,
        )
    )

    evidence = _evaluate(evaluator, proposal)

    assert evidence.success is True
    assert evidence.unsafe is True


def test_seed_and_attempt_only_change_attempt_identity_not_fixture_statistics() -> None:
    evaluator = create_deterministic_fixture_evaluator(_configuration())

    first = _evaluate(evaluator, seed=1, attempt=0)
    second = _evaluate(evaluator, seed=2, attempt=1)

    assert first.evidence_id != second.evidence_id
    assert first.metrics == second.metrics
    assert first.success == second.success
    assert first.progress_after == second.progress_after
    assert first.unsafe == second.unsafe


def test_configuration_change_changes_digest_and_evidence_identity() -> None:
    first_evaluator = create_deterministic_fixture_evaluator(_configuration())
    second_evaluator = create_deterministic_fixture_evaluator(
        _configuration(success_mean_abs_threshold=0.25)
    )

    first = _evaluate(first_evaluator)
    second = _evaluate(second_evaluator)

    assert first_evaluator.configuration_digest != second_evaluator.configuration_digest
    assert first.evidence_id != second.evidence_id
    assert first.success is True
    assert second.success is False


def test_indeterminate_fixture_contains_diagnostics_but_no_task_defaults() -> None:
    evaluator = create_deterministic_fixture_evaluator(
        _configuration(indeterminate_temporal_variation_below=1.0)
    )

    evidence = _evaluate(evaluator)

    assert evidence.status is EvaluationStatus.INDETERMINATE
    assert evidence.success is None
    assert evidence.progress_before is None
    assert evidence.progress_after is None
    assert evidence.progress_delta is None
    assert evidence.unsafe is None
    assert evidence.failure_events == ()
    assert evidence.label_source is None
    assert evidence.label_strength is None
    assert evidence.metrics["mean_temporal_variation"] == pytest.approx(0.5)
    validate_evaluation_evidence(evidence)


def test_fixture_applicability_supports_explicit_skip() -> None:
    evaluator = create_deterministic_fixture_evaluator(
        _configuration(skip_mean_abs_below=1.0)
    )

    decision = evaluator.check_applicability(
        _proposal(), source_dataset_id="sha256:m2a-source"
    )

    assert decision.status is EvaluationStatus.SKIPPED
    assert decision.reason is not None
    assert "synthetic fixture policy" in decision.reason


def test_skip_precedes_controlled_error_and_indeterminate_rules() -> None:
    evaluator = create_deterministic_fixture_evaluator(
        _configuration(
            skip_mean_abs_below=2.0,
            execution_error_max_abs_above=0.1,
            indeterminate_temporal_variation_below=2.0,
        )
    )

    decision = evaluator.check_applicability(
        _proposal(), source_dataset_id="sha256:m2a-source"
    )

    assert decision.status is EvaluationStatus.SKIPPED


def test_controlled_execution_error_precedes_indeterminate_result() -> None:
    evaluator = create_deterministic_fixture_evaluator(
        _configuration(
            execution_error_max_abs_above=1.0,
            indeterminate_temporal_variation_below=2.0,
        )
    )

    with pytest.raises(FixtureEvaluationError, match="controlled deterministic"):
        _evaluate(evaluator)


def test_configuration_is_detached_from_source_mapping() -> None:
    configuration = _configuration()
    before = copy.deepcopy(configuration)
    evaluator = create_deterministic_fixture_evaluator(configuration)

    configuration["success_mean_abs_threshold"] = 999.0

    assert evaluator.resolved_configuration() == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("success_mean_abs_threshold", -0.1),
        ("success_mean_abs_threshold", True),
        ("success_mean_abs_threshold", float("nan")),
        ("unsafe_max_abs_threshold", float("inf")),
        ("skip_mean_abs_below", False),
        ("indeterminate_temporal_variation_below", -1),
        ("execution_error_max_abs_above", "1.0"),
        ("schema_version", "99.0"),
    ],
)
def test_fixture_rejects_invalid_configuration_values(
    field: str, value: object
) -> None:
    with pytest.raises(FixtureConfigurationError, match=field):
        create_deterministic_fixture_evaluator(_configuration(**{field: value}))


def test_fixture_requires_exact_configuration_fields() -> None:
    missing = _configuration()
    missing.pop("unsafe_max_abs_threshold")
    unknown = _configuration(unexpected=True)

    with pytest.raises(FixtureConfigurationError, match="missing"):
        create_deterministic_fixture_evaluator(missing)
    with pytest.raises(FixtureConfigurationError, match="unexpected"):
        create_deterministic_fixture_evaluator(unknown)


def test_fixture_normalizes_negative_zero_thresholds() -> None:
    evaluator = create_deterministic_fixture_evaluator(
        _configuration(
            success_mean_abs_threshold=-0.0,
            unsafe_max_abs_threshold=-0.0,
            skip_mean_abs_below=-0.0,
            indeterminate_temporal_variation_below=-0.0,
            execution_error_max_abs_above=-0.0,
        )
    )

    for field in (
        "success_mean_abs_threshold",
        "unsafe_max_abs_threshold",
        "skip_mean_abs_below",
        "indeterminate_temporal_variation_below",
        "execution_error_max_abs_above",
    ):
        value = evaluator.resolved_configuration()[field]
        assert value == 0.0
        assert isinstance(value, float)
        assert math.copysign(1.0, value) == 1.0


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version":"1.0","schema_version":"1.0"}',
        '{"threshold":NaN}',
        '{"threshold":Infinity}',
        "[1, 2, 3]",
    ],
)
def test_generic_configuration_loader_rejects_unsafe_json(
    tmp_path: Path, text: str
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(EvaluatorConfigurationError):
        load_evaluator_configuration(path)


def test_generic_configuration_loader_returns_immutable_nested_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"nested":{"values":[1,2]}}', encoding="utf-8")

    loaded = load_evaluator_configuration(path)

    assert loaded["nested"] == {"values": (1, 2)}
    with pytest.raises(TypeError):
        loaded["new"] = True  # type: ignore[index]
    nested = loaded["nested"]
    assert isinstance(nested, dict) is False

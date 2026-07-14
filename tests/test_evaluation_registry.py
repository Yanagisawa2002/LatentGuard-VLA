"""Tests for duplicate-safe, non-dynamic M2A evaluator lookup."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import pytest

from latentguard.evaluation.base import EvaluatorConfigurationError
from latentguard.evaluation.fixture import create_deterministic_fixture_evaluator
from latentguard.evaluation.registry import (
    DuplicateEvaluatorError,
    EvaluatorRegistry,
    UnknownEvaluatorError,
    create_evaluator,
    default_evaluator_registry,
)


def _configuration() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "success_mean_abs_threshold": 1.0,
        "unsafe_max_abs_threshold": 10.0,
        "skip_mean_abs_below": None,
        "indeterminate_temporal_variation_below": None,
        "execution_error_max_abs_above": None,
    }


class _InvalidVersionEvaluator:
    def __init__(self, configuration: Mapping[str, object]) -> None:
        self.delegate = create_deterministic_fixture_evaluator(configuration)

    @property
    def evaluator_id(self) -> str:
        return "invalid_version"

    @property
    def evaluator_version(self) -> str:
        return "version-one"

    @property
    def configuration_digest(self) -> str:
        return self.delegate.configuration_digest

    def resolved_configuration(self) -> Mapping[str, object]:
        return self.delegate.resolved_configuration()

    def check_applicability(self, *args: Any, **kwargs: Any):
        return self.delegate.check_applicability(*args, **kwargs)

    def evaluate(self, *args: Any, **kwargs: Any):
        return self.delegate.evaluate(*args, **kwargs)


def test_default_registry_contains_only_explicit_fixture_evaluator() -> None:
    registry = default_evaluator_registry()

    assert registry.names == ("deterministic_fixture",)
    evaluator = registry.create("deterministic_fixture", _configuration())
    assert evaluator.evaluator_id == "deterministic_fixture"


def test_unknown_evaluator_fails_without_dynamic_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = importlib.import_module

    def track_import(name: str, package: str | None = None):
        calls.append(name)
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", track_import)

    with pytest.raises(UnknownEvaluatorError, match="unknown evaluator"):
        create_evaluator("os.system:dangerous", _configuration())

    assert calls == []


def test_registry_rejects_duplicate_and_invalid_names() -> None:
    registry = EvaluatorRegistry()
    registry.register("deterministic_fixture", create_deterministic_fixture_evaluator)

    with pytest.raises(DuplicateEvaluatorError, match="duplicate"):
        registry.register(
            "deterministic_fixture", create_deterministic_fixture_evaluator
        )
    with pytest.raises(EvaluatorConfigurationError, match="non-empty"):
        EvaluatorRegistry((("   ", create_deterministic_fixture_evaluator),))


def test_registry_rejects_factory_with_mismatched_evaluator_id() -> None:
    registry = EvaluatorRegistry(
        (("claimed_name", create_deterministic_fixture_evaluator),)
    )

    with pytest.raises(EvaluatorConfigurationError, match="mismatched"):
        registry.create("claimed_name", _configuration())


def test_registry_rejects_factory_result_without_protocol() -> None:
    def invalid_factory(_: Mapping[str, object]) -> object:
        return object()

    registry = EvaluatorRegistry(
        (("invalid", invalid_factory),)  # type: ignore[arg-type]
    )

    with pytest.raises(EvaluatorConfigurationError, match="ProposalEvaluator"):
        registry.create("invalid", {})


def test_registry_passes_an_immutable_configuration_to_factory() -> None:
    mutation_blocked: list[bool] = []

    def factory(configuration: Mapping[str, object]):
        try:
            configuration["new"] = True  # type: ignore[index]
        except TypeError:
            mutation_blocked.append(True)
        return create_deterministic_fixture_evaluator(configuration)

    registry = EvaluatorRegistry((("deterministic_fixture", factory),))

    evaluator = registry.create("deterministic_fixture", _configuration())

    assert mutation_blocked == [True]
    assert evaluator.evaluator_id == "deterministic_fixture"


def test_registry_rejects_non_mapping_configuration() -> None:
    registry = default_evaluator_registry()

    with pytest.raises(EvaluatorConfigurationError, match="JSON object"):
        registry.create("deterministic_fixture", [])  # type: ignore[arg-type]


def test_registry_rejects_non_semantic_evaluator_version() -> None:
    registry = EvaluatorRegistry((("invalid_version", _InvalidVersionEvaluator),))

    with pytest.raises(EvaluatorConfigurationError, match="semantic version"):
        registry.create("invalid_version", _configuration())

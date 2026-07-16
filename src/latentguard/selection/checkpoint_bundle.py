"""Frozen five-seed verifier identities and trusted runtime artifact loading.

The deterministic bundle deliberately contains no runtime paths or tensors.  Paths
are supplied in a separate runtime-only object and every checkpoint byte digest is
verified before the trusted pickle payload is deserialized.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import NoReturn, cast

from latentguard.integrations.maniskill_pickcube.verifier_state import (
    PICKCUBE_VERIFIER_STATE_SEMANTIC,
)
from latentguard.replay.identity import canonical_json_bytes
from latentguard.training.calibration import (
    FrozenThresholdStateV1,
    TemperatureCalibrationStateV1,
    frozen_thresholds_from_dict,
    temperature_calibration_from_dict,
)
from latentguard.training.config import ModelType
from latentguard.training.dataset import (
    ACCEPTED_ACTION_DIMENSION,
    ACCEPTED_ACTION_HORIZON,
    ACCEPTED_M3A_DATASET_DIGEST,
    ACCEPTED_STATE_DIMENSION,
)
from latentguard.training.preprocessing import (
    PreprocessingStateV1,
    load_preprocessing_state,
)
from latentguard.training.reporting import load_strict_report

VERIFIER_BUNDLE_SCHEMA_VERSION = "1.0"
VERIFIER_CHECKPOINT_SCHEMA_VERSION = "1.0"
VERIFIER_ENSEMBLE_SEMANTIC = (
    "arithmetic_mean_of_five_validation_calibrated_failure_probabilities_v1"
)
VERIFIER_BUNDLE_LOADING_SEMANTIC = (
    "digest_before_deserialize_then_semantic_binding_validation_v1"
)
ACCEPTED_M3B_RESULT_REVISION = "66eaee0f69a8f5dfe4bc7a53a777e08b5ce51b88"
ACCEPTED_M3B_SPLIT_DIGEST = (
    "sha256:173ca40a072a1977cc65dbcccedb4684ae573e97f3984c176e3f91bba09f940a"
)
EXPECTED_FIVE_SEEDS = (0, 1, 2, 3, 4)
SUPPORTED_SELECTION_ARCHITECTURES = frozenset(
    {
        ModelType.ACTION_ONLY_MLP.value,
        ModelType.STATE_ACTION_MLP.value,
        ModelType.TEMPORAL_STATE_ACTION_VERIFIER.value,
    }
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^m3b-run-sha256-[0-9a-f]{64}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class VerifierBundleError(ValueError):
    """Raised when a frozen verifier bundle or runtime artifact differs."""


def _fail(context: str, reason: str) -> NoReturn:
    raise VerifierBundleError(f"{context}: {reason}")


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        _fail(context, "expected sha256: followed by 64 lowercase hex characters")
    return value


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(context, "expected canonical non-empty text")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(context, f"expected integer >= {minimum}")
    return value


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(context, "expected an object with text keys")
    return cast(Mapping[str, object], value)


def _exact_fields(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    if set(value) != expected:
        _fail(context, "unexpected or missing fields")


@dataclass(frozen=True, slots=True)
class VerifierCheckpointIdentityV1:
    """One selected best checkpoint and its validation-only artifacts."""

    seed: int
    checkpoint_identity: str
    checkpoint_content_digest: str
    checkpoint_epoch: int
    calibration_digest: str
    calibration_report_digest: str
    threshold_digest: str
    threshold_report_digest: str
    validation_prediction_digest: str
    checkpoint_kind: str = "best"
    schema_version: str = VERIFIER_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the complete path-independent seed artifact identity."""

        _integer(self.seed, "VerifierCheckpointIdentityV1.seed")
        if _RUN_ID_RE.fullmatch(self.checkpoint_identity) is None:
            _fail(
                "VerifierCheckpointIdentityV1.checkpoint_identity",
                "expected canonical M3B run identity",
            )
        for name in (
            "checkpoint_content_digest",
            "calibration_digest",
            "calibration_report_digest",
            "threshold_digest",
            "threshold_report_digest",
            "validation_prediction_digest",
        ):
            _digest(getattr(self, name), f"VerifierCheckpointIdentityV1.{name}")
        _integer(
            self.checkpoint_epoch,
            "VerifierCheckpointIdentityV1.checkpoint_epoch",
            minimum=1,
        )
        if self.checkpoint_kind != "best":
            _fail(
                "VerifierCheckpointIdentityV1.checkpoint_kind",
                "selection requires the frozen best checkpoint",
            )
        if self.schema_version != VERIFIER_CHECKPOINT_SCHEMA_VERSION:
            _fail(
                "VerifierCheckpointIdentityV1.schema_version",
                "unsupported version",
            )

    def as_mapping(self) -> dict[str, object]:
        """Return the exact JSON-native checkpoint identity."""

        return {
            "calibration_digest": self.calibration_digest,
            "calibration_report_digest": self.calibration_report_digest,
            "checkpoint_content_digest": self.checkpoint_content_digest,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_identity": self.checkpoint_identity,
            "checkpoint_kind": self.checkpoint_kind,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "threshold_digest": self.threshold_digest,
            "threshold_report_digest": self.threshold_report_digest,
            "validation_prediction_digest": self.validation_prediction_digest,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> VerifierCheckpointIdentityV1:
        """Strictly reconstruct one checkpoint identity."""

        expected = {
            "calibration_digest",
            "calibration_report_digest",
            "checkpoint_content_digest",
            "checkpoint_epoch",
            "checkpoint_identity",
            "checkpoint_kind",
            "schema_version",
            "seed",
            "threshold_digest",
            "threshold_report_digest",
            "validation_prediction_digest",
        }
        _exact_fields(value, expected, "VerifierCheckpointIdentityV1")
        return cls(
            seed=_integer(value["seed"], "VerifierCheckpointIdentityV1.seed"),
            checkpoint_identity=_text(
                value["checkpoint_identity"],
                "VerifierCheckpointIdentityV1.checkpoint_identity",
            ),
            checkpoint_content_digest=_text(
                value["checkpoint_content_digest"],
                "VerifierCheckpointIdentityV1.checkpoint_content_digest",
            ),
            checkpoint_epoch=_integer(
                value["checkpoint_epoch"],
                "VerifierCheckpointIdentityV1.checkpoint_epoch",
                minimum=1,
            ),
            checkpoint_kind=_text(
                value["checkpoint_kind"],
                "VerifierCheckpointIdentityV1.checkpoint_kind",
            ),
            calibration_digest=_text(
                value["calibration_digest"],
                "VerifierCheckpointIdentityV1.calibration_digest",
            ),
            calibration_report_digest=_text(
                value["calibration_report_digest"],
                "VerifierCheckpointIdentityV1.calibration_report_digest",
            ),
            threshold_digest=_text(
                value["threshold_digest"],
                "VerifierCheckpointIdentityV1.threshold_digest",
            ),
            threshold_report_digest=_text(
                value["threshold_report_digest"],
                "VerifierCheckpointIdentityV1.threshold_report_digest",
            ),
            validation_prediction_digest=_text(
                value["validation_prediction_digest"],
                "VerifierCheckpointIdentityV1.validation_prediction_digest",
            ),
            schema_version=_text(
                value["schema_version"],
                "VerifierCheckpointIdentityV1.schema_version",
            ),
        )


@dataclass(frozen=True, slots=True)
class VerifierBundleV1:
    """Frozen five-seed verifier ensemble identity without tensors or paths."""

    architecture: str
    checkpoints: tuple[VerifierCheckpointIdentityV1, ...]
    model_configuration_digest: str
    training_configuration_digest: str
    preprocessing_digest: str
    dataset_digest: str = ACCEPTED_M3A_DATASET_DIGEST
    split_digest: str = ACCEPTED_M3B_SPLIT_DIGEST
    training_seeds: tuple[int, ...] = EXPECTED_FIVE_SEEDS
    input_state_semantic: str = PICKCUBE_VERIFIER_STATE_SEMANTIC
    state_dimension: int = ACCEPTED_STATE_DIMENSION
    action_horizon: int = ACCEPTED_ACTION_HORIZON
    action_dimension: int = ACCEPTED_ACTION_DIMENSION
    probability_ensemble_semantic: str = VERIFIER_ENSEMBLE_SEMANTIC
    accepted_m3b_result_revision: str = ACCEPTED_M3B_RESULT_REVISION
    schema_version: str = VERIFIER_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require accepted identities, dimensions, and seeds zero through four."""

        checkpoints = tuple(self.checkpoints)
        seeds = tuple(self.training_seeds)
        object.__setattr__(self, "checkpoints", checkpoints)
        object.__setattr__(self, "training_seeds", seeds)
        if self.architecture not in SUPPORTED_SELECTION_ARCHITECTURES:
            _fail("VerifierBundleV1.architecture", "unsupported selection model")
        if seeds != EXPECTED_FIVE_SEEDS:
            _fail(
                "VerifierBundleV1.training_seeds",
                "expected exactly the ordered seeds [0, 1, 2, 3, 4]",
            )
        if (
            len(checkpoints) != len(EXPECTED_FIVE_SEEDS)
            or tuple(item.seed for item in checkpoints) != seeds
        ):
            _fail(
                "VerifierBundleV1.checkpoints",
                "expected exactly one ordered checkpoint for each frozen seed",
            )
        if len({item.checkpoint_identity for item in checkpoints}) != len(checkpoints):
            _fail("VerifierBundleV1.checkpoints", "duplicate checkpoint identity")
        for name in (
            "model_configuration_digest",
            "training_configuration_digest",
            "preprocessing_digest",
            "dataset_digest",
            "split_digest",
        ):
            _digest(getattr(self, name), f"VerifierBundleV1.{name}")
        if self.dataset_digest != ACCEPTED_M3A_DATASET_DIGEST:
            _fail("VerifierBundleV1.dataset_digest", "wrong accepted M3A dataset")
        if self.split_digest != ACCEPTED_M3B_SPLIT_DIGEST:
            _fail("VerifierBundleV1.split_digest", "wrong accepted M3B split")
        if self.accepted_m3b_result_revision != ACCEPTED_M3B_RESULT_REVISION or (
            _SHA_RE.fullmatch(self.accepted_m3b_result_revision) is None
        ):
            _fail(
                "VerifierBundleV1.accepted_m3b_result_revision",
                "wrong accepted M3B result revision",
            )
        expected_dimensions = (
            ACCEPTED_STATE_DIMENSION,
            ACCEPTED_ACTION_HORIZON,
            ACCEPTED_ACTION_DIMENSION,
        )
        if (
            self.state_dimension,
            self.action_horizon,
            self.action_dimension,
        ) != expected_dimensions:
            _fail("VerifierBundleV1", "model input dimensions changed")
        if self.input_state_semantic != PICKCUBE_VERIFIER_STATE_SEMANTIC:
            _fail("VerifierBundleV1.input_state_semantic", "semantic changed")
        if self.probability_ensemble_semantic != VERIFIER_ENSEMBLE_SEMANTIC:
            _fail(
                "VerifierBundleV1.probability_ensemble_semantic",
                "ensemble semantic changed",
            )
        if self.schema_version != VERIFIER_BUNDLE_SCHEMA_VERSION:
            _fail("VerifierBundleV1.schema_version", "unsupported version")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "accepted_m3b_result_revision": self.accepted_m3b_result_revision,
            "action_dimension": self.action_dimension,
            "action_horizon": self.action_horizon,
            "architecture": self.architecture,
            "checkpoints": [item.as_mapping() for item in self.checkpoints],
            "dataset_digest": self.dataset_digest,
            "input_state_semantic": self.input_state_semantic,
            "model_configuration_digest": self.model_configuration_digest,
            "preprocessing_digest": self.preprocessing_digest,
            "probability_ensemble_semantic": self.probability_ensemble_semantic,
            "schema_version": self.schema_version,
            "split_digest": self.split_digest,
            "state_dimension": self.state_dimension,
            "training_configuration_digest": self.training_configuration_digest,
            "training_seeds": list(self.training_seeds),
        }

    @property
    def content_digest(self) -> str:
        """Return the path- and tensor-independent ensemble identity."""

        encoded = canonical_json_bytes(
            self._identity_payload(), context="VerifierBundleV1"
        )
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def to_dict(self) -> dict[str, object]:
        """Return strict self-digesting JSON content."""

        return {**self._identity_payload(), "content_digest": self.content_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> VerifierBundleV1:
        """Strictly reconstruct a bundle and reject identity tampering."""

        expected = {
            "accepted_m3b_result_revision",
            "action_dimension",
            "action_horizon",
            "architecture",
            "checkpoints",
            "content_digest",
            "dataset_digest",
            "input_state_semantic",
            "model_configuration_digest",
            "preprocessing_digest",
            "probability_ensemble_semantic",
            "schema_version",
            "split_digest",
            "state_dimension",
            "training_configuration_digest",
            "training_seeds",
        }
        _exact_fields(value, expected, "VerifierBundleV1")
        raw_checkpoints = value["checkpoints"]
        raw_seeds = value["training_seeds"]
        if not isinstance(raw_checkpoints, Sequence) or isinstance(
            raw_checkpoints, (str, bytes)
        ):
            _fail("VerifierBundleV1.checkpoints", "expected an array")
        if not isinstance(raw_seeds, Sequence) or isinstance(raw_seeds, (str, bytes)):
            _fail("VerifierBundleV1.training_seeds", "expected an array")
        result = cls(
            architecture=_text(value["architecture"], "VerifierBundleV1.architecture"),
            checkpoints=tuple(
                VerifierCheckpointIdentityV1.from_mapping(
                    _mapping(item, f"VerifierBundleV1.checkpoints[{index}]")
                )
                for index, item in enumerate(raw_checkpoints)
            ),
            model_configuration_digest=_text(
                value["model_configuration_digest"],
                "VerifierBundleV1.model_configuration_digest",
            ),
            training_configuration_digest=_text(
                value["training_configuration_digest"],
                "VerifierBundleV1.training_configuration_digest",
            ),
            preprocessing_digest=_text(
                value["preprocessing_digest"],
                "VerifierBundleV1.preprocessing_digest",
            ),
            dataset_digest=_text(
                value["dataset_digest"], "VerifierBundleV1.dataset_digest"
            ),
            split_digest=_text(value["split_digest"], "VerifierBundleV1.split_digest"),
            training_seeds=tuple(
                _integer(item, f"VerifierBundleV1.training_seeds[{index}]")
                for index, item in enumerate(raw_seeds)
            ),
            input_state_semantic=_text(
                value["input_state_semantic"],
                "VerifierBundleV1.input_state_semantic",
            ),
            state_dimension=_integer(
                value["state_dimension"],
                "VerifierBundleV1.state_dimension",
                minimum=1,
            ),
            action_horizon=_integer(
                value["action_horizon"],
                "VerifierBundleV1.action_horizon",
                minimum=1,
            ),
            action_dimension=_integer(
                value["action_dimension"],
                "VerifierBundleV1.action_dimension",
                minimum=1,
            ),
            probability_ensemble_semantic=_text(
                value["probability_ensemble_semantic"],
                "VerifierBundleV1.probability_ensemble_semantic",
            ),
            accepted_m3b_result_revision=_text(
                value["accepted_m3b_result_revision"],
                "VerifierBundleV1.accepted_m3b_result_revision",
            ),
            schema_version=_text(
                value["schema_version"], "VerifierBundleV1.schema_version"
            ),
        )
        if result.content_digest != _digest(
            value["content_digest"], "VerifierBundleV1.content_digest"
        ):
            _fail("VerifierBundleV1.content_digest", "bundle content changed")
        return result


@dataclass(frozen=True, slots=True)
class VerifierSeedArtifactPathsV1:
    """Runtime-only paths for one checkpoint, calibration, and threshold file."""

    checkpoint: Path
    calibration: Path
    thresholds: Path

    def __post_init__(self) -> None:
        for name in ("checkpoint", "calibration", "thresholds"):
            object.__setattr__(self, name, Path(getattr(self, name)).absolute())


@dataclass(frozen=True, slots=True)
class VerifierRuntimeArtifactsV1:
    """Runtime-only artifact locations that never participate in bundle identity."""

    preprocessing: Path
    seeds: Mapping[int, VerifierSeedArtifactPathsV1]

    def __post_init__(self) -> None:
        object.__setattr__(self, "preprocessing", Path(self.preprocessing).absolute())
        if not isinstance(self.seeds, Mapping) or set(self.seeds) != set(
            EXPECTED_FIVE_SEEDS
        ):
            _fail(
                "VerifierRuntimeArtifactsV1.seeds",
                "expected one runtime path set for each frozen seed",
            )
        copied: dict[int, VerifierSeedArtifactPathsV1] = {}
        for seed in EXPECTED_FIVE_SEEDS:
            item = self.seeds[seed]
            if not isinstance(item, VerifierSeedArtifactPathsV1):
                _fail(
                    f"VerifierRuntimeArtifactsV1.seeds[{seed}]",
                    "invalid path set",
                )
            copied[seed] = item
        object.__setattr__(self, "seeds", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class BundleLoadingDiagnosticsV1:
    """Fixed-schema operational timing for one verified runtime bundle load."""

    device: str
    total_seconds: float
    per_seed_model_seconds: tuple[float, ...]
    seed_order: tuple[int, ...] = EXPECTED_FIVE_SEEDS
    semantic: str = VERIFIER_BUNDLE_LOADING_SEMANTIC
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        _text(self.device, "BundleLoadingDiagnosticsV1.device")
        if self.seed_order != EXPECTED_FIVE_SEEDS or len(
            self.per_seed_model_seconds
        ) != len(EXPECTED_FIVE_SEEDS):
            _fail("BundleLoadingDiagnosticsV1", "seed timing inventory changed")
        for name, value in (
            ("total_seconds", self.total_seconds),
            *(
                (f"per_seed_model_seconds[{index}]", item)
                for index, item in enumerate(self.per_seed_model_seconds)
            ),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                _fail(f"BundleLoadingDiagnosticsV1.{name}", "expected a number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                _fail(
                    f"BundleLoadingDiagnosticsV1.{name}",
                    "expected finite non-negative seconds",
                )
        if self.semantic != VERIFIER_BUNDLE_LOADING_SEMANTIC:
            _fail("BundleLoadingDiagnosticsV1.semantic", "semantic changed")
        if self.schema_version != "1.0":
            _fail("BundleLoadingDiagnosticsV1.schema_version", "unsupported version")

    def to_dict(self) -> dict[str, object]:
        """Return the deterministic diagnostics schema without runtime paths."""

        return {
            "device": self.device,
            "per_seed_model_seconds": list(self.per_seed_model_seconds),
            "schema_version": self.schema_version,
            "seed_order": list(self.seed_order),
            "semantic": self.semantic,
            "total_seconds": self.total_seconds,
        }


@dataclass(frozen=True, slots=True)
class LoadedVerifierSeedV1:
    """One validated runtime model and its frozen post-training states."""

    identity: VerifierCheckpointIdentityV1
    model: object = field(repr=False, compare=False)
    calibration: TemperatureCalibrationStateV1
    thresholds: FrozenThresholdStateV1


@dataclass(frozen=True, slots=True)
class LoadedVerifierBundleV1:
    """Runtime-only five-model ensemble resolved from one frozen identity bundle."""

    identity: VerifierBundleV1
    preprocessing: PreprocessingStateV1
    seeds: tuple[LoadedVerifierSeedV1, ...]
    loading_diagnostics: BundleLoadingDiagnosticsV1

    def __post_init__(self) -> None:
        values = tuple(self.seeds)
        object.__setattr__(self, "seeds", values)
        if tuple(item.identity.seed for item in values) != EXPECTED_FIVE_SEEDS:
            _fail("LoadedVerifierBundleV1.seeds", "runtime seed inventory changed")
        if self.preprocessing.content_digest != self.identity.preprocessing_digest:
            _fail("LoadedVerifierBundleV1.preprocessing", "digest differs")


def _elapsed(clock: Callable[[], float], start: float, context: str) -> float:
    finish = clock()
    if not math.isfinite(start) or not math.isfinite(finish) or finish < start:
        _fail(context, "clock must be finite and monotonic")
    return finish - start


def load_verifier_bundle(
    bundle: VerifierBundleV1,
    artifacts: VerifierRuntimeArtifactsV1,
    *,
    device: str = "cpu",
    clock: Callable[[], float] = perf_counter,
) -> LoadedVerifierBundleV1:
    """Validate all runtime bytes and bindings, then load exactly five models.

    PyTorch and the checkpoint/model modules are imported only when this explicit
    runtime operation is called.  Each checkpoint digest is checked before any
    pickle-backed checkpoint API is invoked.
    """

    if not isinstance(bundle, VerifierBundleV1):
        _fail("load_verifier_bundle.bundle", "expected VerifierBundleV1")
    if not isinstance(artifacts, VerifierRuntimeArtifactsV1):
        _fail("load_verifier_bundle.artifacts", "invalid runtime artifacts")
    device = _text(device, "load_verifier_bundle.device")
    total_start = clock()
    if not math.isfinite(total_start):
        _fail("load_verifier_bundle.clock", "clock must be finite")

    preprocessing = load_preprocessing_state(artifacts.preprocessing)
    if (
        preprocessing.content_digest != bundle.preprocessing_digest
        or preprocessing.dataset_digest != bundle.dataset_digest
        or preprocessing.state_component_count != bundle.state_dimension
        or preprocessing.action_component_count != bundle.action_dimension
    ):
        _fail("load_verifier_bundle.preprocessing", "bundle binding differs")

    # Lazy imports are intentional: CPU-only manifest use must not require Torch.
    try:
        import torch

        from latentguard.training.checkpoint import (
            compute_checkpoint_content_digest,
            inspect_training_checkpoint,
            load_training_checkpoint,
        )
        from latentguard.training.models import build_model, resolve_model_config
    except ImportError as exc:  # pragma: no cover - exercised without training extra
        raise VerifierBundleError(
            "load_verifier_bundle: PyTorch training dependency is unavailable"
        ) from exc

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        _fail("load_verifier_bundle.device", "CUDA is unavailable")
    loaded: list[LoadedVerifierSeedV1] = []
    model_seconds: list[float] = []
    for identity in bundle.checkpoints:
        paths = artifacts.seeds[identity.seed]
        observed_checkpoint_digest = compute_checkpoint_content_digest(paths.checkpoint)
        if observed_checkpoint_digest != identity.checkpoint_content_digest:
            _fail(
                f"load_verifier_bundle.seed[{identity.seed}].checkpoint",
                "byte digest changed",
            )
        seed_start = clock()
        if not math.isfinite(seed_start):
            _fail("load_verifier_bundle.clock", "clock must be finite")
        metadata = inspect_training_checkpoint(paths.checkpoint)
        if (
            metadata.progress.kind != identity.checkpoint_kind
            or metadata.progress.epoch != identity.checkpoint_epoch
            or metadata.progress.binding.run_manifest_identity
            != identity.checkpoint_identity
            or metadata.progress.binding.dataset_digest != bundle.dataset_digest
            or metadata.progress.binding.split_digest != bundle.split_digest
            or metadata.progress.binding.preprocessing_digest
            != bundle.preprocessing_digest
            or metadata.progress.binding.model_config_digest
            != bundle.model_configuration_digest
            or metadata.progress.binding.training_config_digest
            != bundle.training_configuration_digest
            or metadata.model_configuration.content_digest
            != bundle.model_configuration_digest
            or metadata.training_configuration.content_digest
            != bundle.training_configuration_digest
            or metadata.preprocessing_state.content_digest
            != bundle.preprocessing_digest
            or metadata.model_configuration.architecture.model_type.value
            != bundle.architecture
            or metadata.model_configuration.architecture.state_dimension
            != bundle.state_dimension
            or metadata.model_configuration.architecture.action_horizon
            != bundle.action_horizon
            or metadata.model_configuration.architecture.action_dimension
            != bundle.action_dimension
        ):
            _fail(
                f"load_verifier_bundle.seed[{identity.seed}].checkpoint",
                "semantic binding differs",
            )
        model = build_model(metadata.model_configuration.architecture, seed=0)
        if resolve_model_config(model).content_digest != (
            bundle.model_configuration_digest
        ):
            _fail(
                f"load_verifier_bundle.seed[{identity.seed}].model",
                "resolved architecture differs",
            )
        load_training_checkpoint(
            paths.checkpoint,
            expected_binding=metadata.progress.binding,
            model=model,
            restore_rng=False,
        )
        model.to(target)
        model.eval()
        model_seconds.append(
            _elapsed(
                clock,
                seed_start,
                f"load_verifier_bundle.seed[{identity.seed}].clock",
            )
        )

        calibration_report = load_strict_report(
            paths.calibration,
            expected_report_type="temperature_calibration_v1",
            expected_content_digest=identity.calibration_report_digest,
        )
        calibration = temperature_calibration_from_dict(calibration_report.payload)
        if (
            calibration.content_digest != identity.calibration_digest
            or calibration.dataset_digest != bundle.dataset_digest
            or calibration.split_digest != bundle.split_digest
            or calibration.checkpoint_identity != identity.checkpoint_identity
            or calibration.checkpoint_content_digest
            != identity.checkpoint_content_digest
            or calibration.model_config_digest != bundle.model_configuration_digest
            or calibration.preprocessing_digest != bundle.preprocessing_digest
            or calibration.validation_prediction_digest
            != identity.validation_prediction_digest
        ):
            _fail(
                f"load_verifier_bundle.seed[{identity.seed}].calibration",
                "semantic binding differs",
            )
        threshold_report = load_strict_report(
            paths.thresholds,
            expected_report_type="frozen_thresholds_v1",
            expected_content_digest=identity.threshold_report_digest,
        )
        thresholds = frozen_thresholds_from_dict(threshold_report.payload)
        if (
            thresholds.content_digest != identity.threshold_digest
            or thresholds.dataset_digest != bundle.dataset_digest
            or thresholds.split_digest != bundle.split_digest
            or thresholds.calibration_digest != calibration.content_digest
            or thresholds.validation_prediction_digest
            != identity.validation_prediction_digest
        ):
            _fail(
                f"load_verifier_bundle.seed[{identity.seed}].thresholds",
                "semantic binding differs",
            )
        loaded.append(
            LoadedVerifierSeedV1(
                identity=identity,
                model=model,
                calibration=calibration,
                thresholds=thresholds,
            )
        )
    diagnostics = BundleLoadingDiagnosticsV1(
        device=str(target),
        total_seconds=_elapsed(clock, total_start, "load_verifier_bundle.clock"),
        per_seed_model_seconds=tuple(model_seconds),
    )
    return LoadedVerifierBundleV1(
        identity=bundle,
        preprocessing=preprocessing,
        seeds=tuple(loaded),
        loading_diagnostics=diagnostics,
    )


# Concise public name; the explicit suffix remains available for serialized code.
VerifierBundle = VerifierBundleV1


__all__ = [
    "ACCEPTED_M3B_RESULT_REVISION",
    "ACCEPTED_M3B_SPLIT_DIGEST",
    "EXPECTED_FIVE_SEEDS",
    "SUPPORTED_SELECTION_ARCHITECTURES",
    "VERIFIER_BUNDLE_LOADING_SEMANTIC",
    "VERIFIER_BUNDLE_SCHEMA_VERSION",
    "VERIFIER_ENSEMBLE_SEMANTIC",
    "BundleLoadingDiagnosticsV1",
    "LoadedVerifierBundleV1",
    "LoadedVerifierSeedV1",
    "VerifierBundleError",
    "VerifierBundle",
    "VerifierBundleV1",
    "VerifierCheckpointIdentityV1",
    "VerifierRuntimeArtifactsV1",
    "VerifierSeedArtifactPathsV1",
    "load_verifier_bundle",
]

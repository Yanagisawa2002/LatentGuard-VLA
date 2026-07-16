"""Build frozen M3C verifier bundles from exact M3B result records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
    load_anchor_manifest,
)
from latentguard.selection.checkpoint_bundle import (
    EXPECTED_FIVE_SEEDS,
    SUPPORTED_SELECTION_ARCHITECTURES,
    LoadedVerifierBundleV1,
    VerifierBundleV1,
    VerifierCheckpointIdentityV1,
    VerifierRuntimeArtifactsV1,
    VerifierSeedArtifactPathsV1,
    load_verifier_bundle,
)
from latentguard.training.baselines import (
    ActionMagnitudeBaselineV1,
    fit_action_magnitude_baseline,
)
from latentguard.training.calibration import (
    frozen_thresholds_from_dict,
    temperature_calibration_from_dict,
)
from latentguard.training.dataset import (
    ACCEPTED_M3A_DATASET_DIGEST,
    load_accepted_action_verifier_dataset,
)
from latentguard.training.preprocessing import load_preprocessing_state
from latentguard.training.reporting import (
    StrictReportV1,
    load_strict_report,
    save_strict_report,
)
from latentguard.training.selection import selection_record_from_dict

SELECTION_BUNDLE_REPORT_TYPE = "m3c_verifier_bundle_v1"
SELECTION_BUNDLE_SUMMARY_REPORT_TYPE = "m3c_verifier_bundle_summary_v1"


class SelectionPreparationError(ValueError):
    """Raised when accepted M3B records or runtime artifacts are incomplete."""


def _fail(context: str, reason: str) -> NoReturn:
    raise SelectionPreparationError(f"{context}: {reason}")


@dataclass(frozen=True, slots=True)
class PreparedVerifierBundlesV1:
    """Path-free identities plus optional digest-validated runtime models."""

    bundles: Mapping[str, VerifierBundleV1]
    loaded: Mapping[str, LoadedVerifierBundleV1]
    selection_record_digest: str
    preprocessing_digest: str

    def __post_init__(self) -> None:
        """Freeze exact three-architecture inventories."""

        expected = set(SUPPORTED_SELECTION_ARCHITECTURES)
        if set(self.bundles) != expected:
            _fail("bundles", "expected the exact three M3C architectures")
        if set(self.loaded) not in (set(), expected):
            _fail("loaded", "must be empty or cover all three architectures")
        if any(self.bundles[key].architecture != key for key in expected):
            _fail("bundles", "architecture key differs from bundle")
        if any(
            self.loaded[key].identity.content_digest != self.bundles[key].content_digest
            for key in self.loaded
        ):
            _fail("loaded", "runtime bundle identity differs")
        object.__setattr__(
            self, "bundles", MappingProxyType(dict(sorted(self.bundles.items())))
        )
        object.__setattr__(
            self, "loaded", MappingProxyType(dict(sorted(self.loaded.items())))
        )

    def summary_payload(self) -> dict[str, object]:
        """Return a compact artifact-availability and identity summary."""

        return {
            "architectures": {
                name: {
                    "bundle_digest": bundle.content_digest,
                    "checkpoint_count": len(bundle.checkpoints),
                    "checkpoint_content_digests": [
                        item.checkpoint_content_digest for item in bundle.checkpoints
                    ],
                    "loaded_and_validated": name in self.loaded,
                    "training_seeds": list(bundle.training_seeds),
                }
                for name, bundle in self.bundles.items()
            },
            "checkpoint_reconstruction_performed": False,
            "preprocessing_digest": self.preprocessing_digest,
            "schema_version": "1.0",
            "selection_record_digest": self.selection_record_digest,
        }


def _runtime_artifacts(
    runtime_root: Path,
    architecture: str,
    *,
    preprocessing_path: Path,
) -> VerifierRuntimeArtifactsV1:
    return VerifierRuntimeArtifactsV1(
        preprocessing=preprocessing_path,
        seeds={
            seed: VerifierSeedArtifactPathsV1(
                checkpoint=(
                    runtime_root
                    / "runs"
                    / architecture
                    / f"seed-{seed}"
                    / "checkpoints"
                    / "best.pt"
                ),
                calibration=(
                    runtime_root
                    / "evaluations"
                    / architecture
                    / f"seed-{seed}"
                    / "calibration.json"
                ),
                thresholds=(
                    runtime_root
                    / "evaluations"
                    / architecture
                    / f"seed-{seed}"
                    / "thresholds.json"
                ),
            )
            for seed in EXPECTED_FIVE_SEEDS
        },
    )


def load_m3b_action_magnitude_identity(result_root: Path) -> tuple[str, str]:
    """Read the frozen action-magnitude identity from the accepted M3B report."""

    report = load_strict_report(Path(result_root) / "benchmark-summary.json")
    payload = report.payload
    if (
        payload.get("dataset_digest") != ACCEPTED_M3A_DATASET_DIGEST
        or payload.get("split_digest") is None
        or payload.get("preprocessing_digest") is None
    ):
        _fail("benchmark summary", "accepted dataset bindings are absent")
    baselines = payload.get("nonlearned_baselines")
    if not isinstance(baselines, Mapping):
        _fail("benchmark summary", "non-learned baseline record is absent")
    fitted = baselines.get("fitted_digests")
    if not isinstance(fitted, Mapping):
        _fail("benchmark summary", "fitted baseline digests are absent")
    digest = fitted.get("action_magnitude")
    if not isinstance(digest, str):
        _fail("benchmark summary", "action-magnitude identity is absent")
    if (
        len(digest) != 71
        or not digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        _fail("benchmark summary", "action-magnitude identity is invalid")
    return digest, report.content_digest


def reconstruct_frozen_action_magnitude_baseline(
    dataset_root: Path,
    acceptance_report: Path,
    anchor_manifest_directory: Path,
    *,
    expected_digest: str,
) -> ActionMagnitudeBaselineV1:
    """Rebuild the train-only M3B heuristic and require its recorded identity."""

    manifest = load_anchor_manifest(anchor_manifest_directory)
    reasons = {
        record.anchor.anchor_id: record.anchor.selection_reason
        for record in manifest.records
    }
    dataset = load_accepted_action_verifier_dataset(
        dataset_root,
        full_target_report=acceptance_report,
        anchor_selection_reasons=reasons,
        anchor_manifest_content_digest=manifest.content_digest,
    )
    baseline = fit_action_magnitude_baseline(dataset)
    if baseline.content_digest != expected_digest:
        _fail(
            "action-magnitude baseline",
            "reconstructed train-only state differs from the M3B record",
        )
    return baseline


def build_verifier_bundle_identities(
    result_root: Path,
    *,
    runtime_root: Path | None = None,
) -> tuple[
    Mapping[str, VerifierBundleV1],
    Mapping[str, VerifierRuntimeArtifactsV1],
    str,
    str,
]:
    """Read M3B records and derive all identities without copying task constants."""

    records = Path(result_root).absolute()
    runtime = records if runtime_root is None else Path(runtime_root).absolute()
    selection_report = load_strict_report(
        records / "selection-record.json", expected_report_type="selection_record_v1"
    )
    selection = selection_record_from_dict(selection_report.payload)
    benchmark_report = load_strict_report(records / "benchmark-summary.json")
    if (
        benchmark_report.payload.get("selection_record_digest")
        != selection.content_digest
        or benchmark_report.payload.get("dataset_digest") != selection.dataset_digest
        or benchmark_report.payload.get("split_digest") != selection.split_digest
    ):
        _fail("benchmark summary", "selection or dataset binding differs")
    artifact_digests = benchmark_report.payload.get("evaluation_artifact_digests")
    if not isinstance(artifact_digests, Mapping):
        _fail("benchmark summary", "evaluation artifact digest inventory is absent")
    preprocessing_path = runtime / "preprocessing.json"
    if not preprocessing_path.is_file() and (records / "preprocessing.json").is_file():
        preprocessing_path = records / "preprocessing.json"
    preprocessing = load_preprocessing_state(preprocessing_path)
    bundles: dict[str, VerifierBundleV1] = {}
    paths: dict[str, VerifierRuntimeArtifactsV1] = {}
    by_architecture = {item.model_type: item for item in selection.candidates}
    if not set(SUPPORTED_SELECTION_ARCHITECTURES).issubset(by_architecture):
        _fail("selection record", "one or more required architectures are absent")
    for architecture in sorted(SUPPORTED_SELECTION_ARCHITECTURES):
        aggregate = by_architecture[architecture]
        checkpoints: list[VerifierCheckpointIdentityV1] = []
        for result in aggregate.seed_results:
            artifact_key = f"{architecture}/seed-{result.seed}"
            expected_artifacts = artifact_digests.get(artifact_key)
            if not isinstance(expected_artifacts, Mapping):
                _fail(artifact_key, "expected evaluation artifacts are absent")
            calibration_report_digest = expected_artifacts.get("calibration.json")
            threshold_report_digest = expected_artifacts.get("thresholds.json")
            if not isinstance(calibration_report_digest, str) or not isinstance(
                threshold_report_digest, str
            ):
                _fail(artifact_key, "calibration or threshold digest is absent")
            evaluation_root = (
                runtime / "evaluations" / architecture / f"seed-{result.seed}"
            )
            calibration_report = load_strict_report(
                evaluation_root / "calibration.json",
                expected_report_type="temperature_calibration_v1",
                expected_content_digest=calibration_report_digest,
            )
            threshold_report = load_strict_report(
                evaluation_root / "thresholds.json",
                expected_report_type="frozen_thresholds_v1",
                expected_content_digest=threshold_report_digest,
            )
            calibration = temperature_calibration_from_dict(calibration_report.payload)
            thresholds = frozen_thresholds_from_dict(threshold_report.payload)
            if (
                calibration.checkpoint_identity != result.checkpoint_identity
                or calibration.checkpoint_content_digest
                != result.checkpoint_content_digest
                or calibration.validation_prediction_digest
                != result.validation_prediction_digest
                or calibration.model_config_digest != aggregate.model_config_digest
                or calibration.preprocessing_digest != preprocessing.content_digest
                or thresholds.calibration_digest != calibration.content_digest
                or thresholds.validation_prediction_digest
                != result.validation_prediction_digest
            ):
                _fail(
                    f"{architecture}/seed-{result.seed}",
                    "calibration or threshold binding differs from selection",
                )
            checkpoints.append(
                VerifierCheckpointIdentityV1(
                    seed=result.seed,
                    checkpoint_identity=result.checkpoint_identity,
                    checkpoint_content_digest=result.checkpoint_content_digest,
                    checkpoint_epoch=result.checkpoint_epoch,
                    calibration_digest=calibration.content_digest,
                    calibration_report_digest=calibration_report.content_digest,
                    threshold_digest=thresholds.content_digest,
                    threshold_report_digest=threshold_report.content_digest,
                    validation_prediction_digest=result.validation_prediction_digest,
                    checkpoint_kind=result.checkpoint_kind,
                )
            )
        bundle = VerifierBundleV1(
            architecture=architecture,
            checkpoints=tuple(checkpoints),
            model_configuration_digest=aggregate.model_config_digest,
            training_configuration_digest=aggregate.training_config_digest,
            preprocessing_digest=preprocessing.content_digest,
            dataset_digest=selection.dataset_digest,
            split_digest=selection.split_digest,
            training_seeds=selection.expected_seeds,
        )
        bundles[architecture] = bundle
        paths[architecture] = _runtime_artifacts(
            runtime,
            architecture,
            preprocessing_path=preprocessing_path,
        )
    return (
        MappingProxyType(bundles),
        MappingProxyType(paths),
        selection_report.content_digest,
        preprocessing.content_digest,
    )


def prepare_verifier_bundles(
    result_root: Path,
    output_directory: Path,
    *,
    runtime_root: Path | None = None,
    device: str = "cpu",
    validate_runtime: bool = True,
    resume: bool = False,
) -> PreparedVerifierBundlesV1:
    """Publish bundle identities and optionally validate all 15 runtime models."""

    bundles, paths, selection_digest, preprocessing_digest = (
        build_verifier_bundle_identities(result_root, runtime_root=runtime_root)
    )
    action_magnitude_digest, benchmark_summary_digest = (
        load_m3b_action_magnitude_identity(result_root)
    )
    output = Path(output_directory).absolute()
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        _fail("output directory", "expected a regular directory")
    if output.exists() and any(output.iterdir()) and not resume:
        _fail("output directory", "must be absent or empty unless resuming")
    output.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, LoadedVerifierBundleV1] = {}
    for architecture, bundle in bundles.items():
        report = StrictReportV1(
            report_type=SELECTION_BUNDLE_REPORT_TYPE,
            payload=bundle.to_dict(),
        )
        path = output / f"{architecture}.json"
        save_strict_report(report, path)
        reloaded = load_strict_report(
            path,
            expected_report_type=SELECTION_BUNDLE_REPORT_TYPE,
            expected_content_digest=report.content_digest,
        )
        if VerifierBundleV1.from_dict(reloaded.payload).content_digest != (
            bundle.content_digest
        ):
            _fail("bundle reload", "identity changed")
        if validate_runtime:
            loaded[architecture] = load_verifier_bundle(
                bundle, paths[architecture], device=device
            )
    prepared = PreparedVerifierBundlesV1(
        bundles=bundles,
        loaded=loaded,
        selection_record_digest=selection_digest,
        preprocessing_digest=preprocessing_digest,
    )
    summary_payload = prepared.summary_payload()
    summary_payload.update(
        {
            "action_magnitude_baseline_digest": action_magnitude_digest,
            "source_benchmark_summary_digest": benchmark_summary_digest,
        }
    )
    summary = StrictReportV1(
        report_type=SELECTION_BUNDLE_SUMMARY_REPORT_TYPE,
        payload=summary_payload,
    )
    save_strict_report(summary, output / "summary.json")
    return prepared


def load_prepared_bundle(path: Path) -> VerifierBundleV1:
    """Load one path-free bundle identity from its immutable strict report."""

    report = load_strict_report(path, expected_report_type=SELECTION_BUNDLE_REPORT_TYPE)
    return VerifierBundleV1.from_dict(report.payload)


__all__ = [
    "PreparedVerifierBundlesV1",
    "SELECTION_BUNDLE_REPORT_TYPE",
    "SELECTION_BUNDLE_SUMMARY_REPORT_TYPE",
    "SelectionPreparationError",
    "build_verifier_bundle_identities",
    "load_m3b_action_magnitude_identity",
    "load_prepared_bundle",
    "prepare_verifier_bundles",
    "reconstruct_frozen_action_magnitude_baseline",
]

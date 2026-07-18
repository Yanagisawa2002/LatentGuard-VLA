"""Strict command-line workflows for blind M3C candidate selection.

The Stage-A command intentionally has no evidence, outcome, or replay path
arguments.  Optional simulator and PyTorch imports remain behind commands that
actually need those runtimes, so parser construction and local contract tests
stay CPU-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

if TYPE_CHECKING:
    from latentguard.training.reporting import StrictReportV1

import numpy as np
from numpy.typing import NDArray

from latentguard.replay.identity import canonical_json_bytes

M3C_COMMANDS = frozenset(
    {
        "prepare-selection-checkpoints",
        "build-blind-candidate-pools",
        "select-action-candidates",
        "replay-selected-candidates",
        "replay-complete-candidate-pools",
        "evaluate-candidate-selection",
    }
)

DEFAULT_M3C_PROTOCOL_CONFIG = Path("configs/selection/m3c/protocol-v1.json")
DEFAULT_M3C_CANDIDATE_CONFIG = Path("configs/selection/m3c/candidate-pool-v1.json")
DEFAULT_EXPECTED_CONTRACT = Path(
    "configs/integrations/maniskill_pickcube/expected-contract-v1.json"
)
ACTION_MAGNITUDE_REPORT = "action-magnitude-baseline.json"
ABSTENTION_POLICY_REPORT = "temporal-abstention-policies.json"
PREPARATION_SUMMARY_REPORT = "m3c-preparation-summary.json"
PRECOLLECTION_INFERENCE_SMOKE_REPORT = "precollection-inference-smoke"
SELECTOR_CONFIGURATION_REPORT = "selector-configuration.json"
SELECTION_MANIFEST_NAME = "blind-selection-manifest.json"
SELECTION_RESULT_REPORT = "candidate-selection-report.json"
BOUND_RESULT_NAME = "bound-selection-result.json"
REPLAY_RESUME_REPORT_SUFFIX = "-m3c-resume-summary.json"
FINAL_RESUME_REPORT = "resume-summary.json"
HUMAN_REVIEW_NAME = "review.md"


class M3CCommandError(ValueError):
    """Raised when an M3C command violates a frozen workflow contract."""


def _fail(context: str, reason: str) -> NoReturn:
    raise M3CCommandError(f"{context}: {reason}")


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


def _add_protocol_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_M3C_PROTOCOL_CONFIG, required=False
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _add_replay_arguments(parser: argparse.ArgumentParser) -> None:
    _add_protocol_arguments(parser)
    parser.add_argument(
        "--candidate-config", type=Path, default=DEFAULT_M3C_CANDIDATE_CONFIG
    )
    parser.add_argument("--candidate-pool-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--corruption-dir", type=Path, required=True)
    parser.add_argument("--runtime-archive-dir", type=Path, required=True)
    parser.add_argument("--anchor-manifest-dir", type=Path, required=True)
    parser.add_argument("--compatibility-report", type=Path, required=True)
    parser.add_argument(
        "--expected-contract", type=Path, default=DEFAULT_EXPECTED_CONTRACT
    )
    parser.add_argument("--action-layout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=_nonnegative_int, required=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--retry-execution-errors", action="store_true")


def add_m3c_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the six bounded M3C commands without importing Torch."""

    prepare = subparsers.add_parser(
        "prepare-selection-checkpoints",
        help="freeze and validate all M3B five-seed selector artifacts",
    )
    _add_protocol_arguments(prepare)
    prepare.add_argument("--m3b-result-root", type=Path, required=True)
    prepare.add_argument("--m3b-runtime-root", type=Path, required=True)
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument(
        "--acceptance-report",
        type=Path,
        required=True,
        help=(
            "accepted M3A serialized-reload validation-report.json "
            "(not acceptance-summary.json)"
        ),
    )
    prepare.add_argument("--anchor-manifest-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--device", default="cpu")
    prepare.add_argument("--inference-smoke-output", type=Path)

    build = subparsers.add_parser(
        "build-blind-candidate-pools",
        help="build eight-candidate pools from outcome-free new source anchors",
    )
    _add_protocol_arguments(build)
    build.add_argument(
        "--candidate-config", type=Path, default=DEFAULT_M3C_CANDIDATE_CONFIG
    )
    build.add_argument("--runtime-archive-dir", type=Path, required=True)
    build.add_argument("--source-dir", type=Path, required=True)
    build.add_argument("--anchor-manifest-dir", type=Path, required=True)
    build.add_argument("--m3a-dataset-dir", type=Path, required=True)
    build.add_argument("--m3a-anchor-manifest-dir", type=Path, required=True)
    build.add_argument("--candidate-pool-dir", type=Path, required=True)
    build.add_argument("--blind-input-dir", type=Path, required=True)
    build.add_argument("--corruption-dir", type=Path, required=True)
    build.add_argument("--smoke-candidate-pool-dir", type=Path)
    build.add_argument("--summary-output", type=Path, required=True)
    build.add_argument("--compatibility-report", type=Path, required=True)
    build.add_argument(
        "--expected-contract", type=Path, default=DEFAULT_EXPECTED_CONTRACT
    )
    build.add_argument("--action-layout", type=Path, required=True)

    select = subparsers.add_parser(
        "select-action-candidates",
        help="run outcome-isolated Stage A and finalize one immutable manifest",
    )
    _add_protocol_arguments(select)
    select.add_argument("--blind-input-dir", type=Path, required=True)
    select.add_argument("--prepared-bundles-dir", type=Path, required=True)
    select.add_argument("--m3b-runtime-root", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument("--latency-output", type=Path)
    select.add_argument("--device", default="cuda")

    selected = subparsers.add_parser(
        "replay-selected-candidates",
        help="run Stage B for the deduplicated union of blind selections",
    )
    _add_replay_arguments(selected)

    complete = subparsers.add_parser(
        "replay-complete-candidate-pools",
        help="run Stage C for the exact complement and verify the complete join",
    )
    _add_replay_arguments(complete)
    complete.add_argument("--selected-output-dir", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "evaluate-candidate-selection",
        help="bind complete outcomes and report selectors, oracle, and bootstrap",
    )
    _add_protocol_arguments(evaluate)
    evaluate.add_argument("--candidate-pool-dir", type=Path, required=True)
    evaluate.add_argument("--selection-manifest", type=Path, required=True)
    evaluate.add_argument("--corruption-dir", type=Path, required=True)
    evaluate.add_argument("--selected-output-dir", type=Path, required=True)
    evaluate.add_argument("--remainder-output-dir", type=Path, required=True)
    evaluate.add_argument("--cpu-latency-report", type=Path, required=True)
    evaluate.add_argument("--gpu-latency-report", type=Path, required=True)
    evaluate.add_argument("--compatibility-report", type=Path, required=True)
    evaluate.add_argument(
        "--expected-contract", type=Path, default=DEFAULT_EXPECTED_CONTRACT
    )
    evaluate.add_argument("--output-dir", type=Path, required=True)


def run_m3c_command(args: argparse.Namespace) -> int | None:
    """Dispatch an M3C command, returning ``None`` for unrelated commands."""

    if args.command not in M3C_COMMANDS:
        return None
    handlers = {
        "prepare-selection-checkpoints": _run_prepare_selection_checkpoints,
        "build-blind-candidate-pools": _run_build_blind_candidate_pools,
        "select-action-candidates": _run_select_action_candidates,
        "replay-selected-candidates": _run_replay_selected_candidates,
        "replay-complete-candidate-pools": _run_replay_complete_candidate_pools,
        "evaluate-candidate-selection": _run_evaluate_candidate_selection,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        print(
            f"{args.command} interrupted: persisted state may be resumed",
            file=sys.stderr,
        )
        return 130
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"{args.command} failed: {error}", file=sys.stderr)
        return 1


def _reject_constant(value: str) -> NoReturn:
    _fail("configuration JSON", f"non-finite constant {value!r} is unsupported")


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("configuration JSON", f"duplicate field {key!r}")
        result[key] = value
    return result


def _load_json_mapping(
    path: Path, *, fields: frozenset[str], context: str
) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        _fail(context, "expected a regular non-symlink JSON file")
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_constant,
        )
    except M3CCommandError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise M3CCommandError(f"{context}: could not read strict JSON: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != fields:
        _fail(context, "unexpected or missing fields")
    return cast(dict[str, object], raw)


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
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


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(context, "expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(context, "expected a finite number")
    return result


@dataclass(frozen=True, slots=True)
class _M3CProtocolConfig:
    payload: Mapping[str, object]
    content_digest: str

    @property
    def expected_trajectory_count(self) -> int:
        """Return the current mode count after ``for_mode`` creates a view."""
        return cast(int, self.payload["_expected_trajectory_count"])

    @property
    def mode(self) -> str:
        return cast(str, self.payload["_mode"])

    def for_mode(self, mode: str) -> _M3CProtocolConfig:
        if mode not in {"smoke", "full"}:
            _fail("mode", "expected smoke or full")
        values = dict(self.payload)
        values["_mode"] = mode
        values["_expected_trajectory_count"] = cast(
            int, values[f"{mode}_requested_success_count"]
        )
        return _M3CProtocolConfig(values, self.content_digest)


def _load_protocol(path: Path, *, mode: str) -> _M3CProtocolConfig:
    from latentguard.selection.protocol import load_selection_protocol

    protocol = load_selection_protocol(path)
    return _M3CProtocolConfig(protocol.as_mapping(), protocol.content_digest).for_mode(
        mode
    )


def _resolved(path: Path) -> Path:
    return Path(path).absolute().resolve()


def _replay_resume_report_path(output_root: Path) -> Path:
    """Return the compact sibling report path without altering replay bundles."""

    root = Path(output_root)
    return root.with_name(f"{root.name}{REPLAY_RESUME_REPORT_SUFFIX}")


def _save_immutable_text(path: Path, content: str) -> Path:
    """Atomically create one UTF-8 text artifact without overwriting content."""

    if not isinstance(content, str) or not content.endswith("\n"):
        _fail("text artifact", "expected newline-terminated text")
    destination = Path(path)
    encoded = content.encode("utf-8")
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            _fail("text artifact", "existing path is not a regular file")
        try:
            observed = destination.read_bytes()
        except OSError as exc:
            raise M3CCommandError(f"text artifact: {exc}") from exc
        if observed != encoded:
            _fail("text artifact", "existing bytes differ")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_text = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    staging = Path(staging_text)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, destination)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    staging.unlink(missing_ok=True)
    return destination


def _validate_immutable_text(path: Path, content: str) -> None:
    """Require one existing regular file to match expected UTF-8 bytes exactly."""

    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1:
        _fail("text artifact", "expected one unlinked regular file")
    if source.read_bytes() != content.encode("utf-8"):
        _fail("text artifact", "persisted bytes differ")


def _require_separate_paths(*, inputs: Sequence[Path], outputs: Sequence[Path]) -> None:
    resolved_inputs = tuple(_resolved(item) for item in inputs)
    resolved_outputs = tuple(_resolved(item) for item in outputs)
    for index, left in enumerate(resolved_outputs):
        for right in resolved_outputs[index + 1 :]:
            if (
                left == right
                or left.is_relative_to(right)
                or right.is_relative_to(left)
            ):
                _fail("paths", "output roots must not overlap")
        for source in resolved_inputs:
            if (
                left == source
                or left.is_relative_to(source)
                or source.is_relative_to(left)
            ):
                _fail("paths", "input and output roots must not overlap")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _git_sha() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    value = completed.stdout.strip()
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        _fail("Git SHA", "expected full lowercase 40-hex revision")
    return value


def _strict_report_payload(
    report_type: str, payload: Mapping[str, object]
) -> StrictReportV1:
    from latentguard.training.reporting import StrictReportV1

    return StrictReportV1(report_type=report_type, payload=dict(payload))


def _baseline_payload(baseline: object) -> dict[str, object]:
    value = cast(Any, baseline)
    return {
        "action_mean": value.action_mean.tolist(),
        "content_digest": value.content_digest,
        "dataset_digest": value.dataset_digest,
        "feature_mean": value.feature_mean.tolist(),
        "feature_standard_deviation": value.feature_standard_deviation.tolist(),
        "minimum_standard_deviation": value.minimum_standard_deviation,
        "schema_version": value.schema_version,
        "semantic": value.semantic,
        "training_sample_count": value.training_sample_count,
        "training_split_digest": value.training_split_digest,
        "valid_action_step_count": value.valid_action_step_count,
    }


def _load_action_baseline(
    path: Path, *, expected_report_digest: str | None = None
) -> object:
    from latentguard.training.baselines import ActionMagnitudeBaselineV1
    from latentguard.training.reporting import load_strict_report

    report = load_strict_report(
        path,
        expected_report_type="m3c_action_magnitude_v1",
        expected_content_digest=expected_report_digest,
    )
    value = report.payload
    expected = {
        "action_mean",
        "content_digest",
        "dataset_digest",
        "feature_mean",
        "feature_standard_deviation",
        "minimum_standard_deviation",
        "schema_version",
        "semantic",
        "training_sample_count",
        "training_split_digest",
        "valid_action_step_count",
    }
    if set(value) != expected:
        _fail("action magnitude report", "unexpected or missing fields")
    baseline = ActionMagnitudeBaselineV1(
        dataset_digest=cast(str, value["dataset_digest"]),
        training_split_digest=cast(str, value["training_split_digest"]),
        training_sample_count=cast(int, value["training_sample_count"]),
        valid_action_step_count=cast(int, value["valid_action_step_count"]),
        minimum_standard_deviation=cast(float, value["minimum_standard_deviation"]),
        action_mean=np.asarray(value["action_mean"], dtype=np.float64),
        feature_mean=np.asarray(value["feature_mean"], dtype=np.float64),
        feature_standard_deviation=np.asarray(
            value["feature_standard_deviation"], dtype=np.float64
        ),
        schema_version=cast(str, value["schema_version"]),
        semantic=cast(str, value["semantic"]),
    )
    if baseline.content_digest != value["content_digest"]:
        _fail("action magnitude report", "fitted identity changed")
    return baseline


def _runtime_artifacts(runtime_root: Path, architecture: str) -> object:
    from latentguard.selection.checkpoint_bundle import (
        EXPECTED_FIVE_SEEDS,
        VerifierRuntimeArtifactsV1,
        VerifierSeedArtifactPathsV1,
    )

    root = Path(runtime_root)
    preprocessing = root / "preprocessing.json"
    return VerifierRuntimeArtifactsV1(
        preprocessing=preprocessing,
        seeds={
            seed: VerifierSeedArtifactPathsV1(
                checkpoint=root
                / "runs"
                / architecture
                / f"seed-{seed}"
                / "checkpoints"
                / "best.pt",
                calibration=root
                / "evaluations"
                / architecture
                / f"seed-{seed}"
                / "calibration.json",
                thresholds=root
                / "evaluations"
                / architecture
                / f"seed-{seed}"
                / "thresholds.json",
            )
            for seed in EXPECTED_FIVE_SEEDS
        },
    )


def _load_accepted_dataset(args: argparse.Namespace) -> object:
    from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
        load_anchor_manifest,
    )
    from latentguard.training.dataset import load_accepted_action_verifier_dataset

    manifest = load_anchor_manifest(args.anchor_manifest_dir)
    reasons = {
        item.anchor.anchor_id: item.anchor.selection_reason for item in manifest.records
    }
    return load_accepted_action_verifier_dataset(
        args.dataset_root,
        full_target_report=args.acceptance_report,
        anchor_selection_reasons=reasons,
        anchor_manifest_content_digest=manifest.content_digest,
    )


def _validate_m3b_runtime_artifact_digests(
    result_root: Path, runtime_root: Path
) -> str:
    from latentguard.selection.checkpoint_bundle import (
        EXPECTED_FIVE_SEEDS,
        SUPPORTED_SELECTION_ARCHITECTURES,
    )
    from latentguard.training.config import ModelType
    from latentguard.training.reporting import load_strict_report

    benchmark = load_strict_report(
        Path(result_root) / "benchmark-summary.json",
        expected_report_type="m3b_compact_benchmark_summary_v1",
    )
    artifacts = benchmark.payload.get("evaluation_artifact_digests")
    expected_keys = {
        f"{architecture}/seed-{seed}"
        for architecture in SUPPORTED_SELECTION_ARCHITECTURES
        for seed in EXPECTED_FIVE_SEEDS
    }
    committed_keys = {
        f"{architecture.value}/seed-{seed}"
        for architecture in ModelType
        for seed in EXPECTED_FIVE_SEEDS
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != committed_keys:
        _fail(
            "M3B benchmark summary",
            "evaluation artifact inventory is incomplete or unexpected",
        )
    for key in sorted(expected_keys):
        record = artifacts[key]
        if not isinstance(record, Mapping):
            _fail("M3B benchmark summary", f"{key} artifact record is invalid")
        for filename, report_type in (
            ("calibration.json", "temperature_calibration_v1"),
            ("thresholds.json", "frozen_thresholds_v1"),
        ):
            expected_digest = _digest(
                record.get(filename),
                f"M3B benchmark summary.{key}.{filename}",
            )
            load_strict_report(
                Path(runtime_root) / "evaluations" / key / filename,
                expected_report_type=report_type,
                expected_content_digest=expected_digest,
            )
    return benchmark.content_digest


def _validation_prediction_rows(path: Path) -> tuple[dict[str, object], ...]:
    from latentguard.training.reporting import load_strict_report

    report = load_strict_report(path, expected_report_type="validation_predictions_v1")
    if set(report.payload) != {"predictions"} or not isinstance(
        report.payload["predictions"], list
    ):
        _fail("validation predictions", "unexpected or missing fields")
    expected = {
        "calibrated_failure_probability",
        "candidate_type",
        "failure_target",
        "group_id",
        "raw_logit",
        "sample_id",
        "source_trajectory_id",
        "split",
        "uncalibrated_failure_probability",
    }
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(report.payload["predictions"]):
        if not isinstance(raw, dict) or set(raw) != expected:
            _fail("validation predictions", f"row {index} has wrong fields")
        if raw["split"] != "validation":
            _fail("validation predictions", "non-validation row appeared")
        rows.append(cast(dict[str, object], raw))
    sample_ids = tuple(cast(str, row["sample_id"]) for row in rows)
    if len(sample_ids) != len(set(sample_ids)):
        _fail("validation predictions", "sample identity is duplicated")
    return tuple(rows)


def _fit_temporal_policies(
    loaded_bundle: object, runtime_root: Path, protocol: _M3CProtocolConfig
) -> tuple[object, ...]:
    from latentguard.selection.selectors import (
        fit_validation_ensemble_abstention_policies,
    )
    from latentguard.training.calibration import (
        compute_validation_prediction_digest,
    )

    loaded = cast(Any, loaded_bundle)
    per_seed: list[NDArray[Any]] = []
    reference_all: tuple[dict[str, object], ...] | None = None
    reference: tuple[dict[str, object], ...] | None = None
    architecture = loaded.identity.architecture
    for seed_runtime in loaded.seeds:
        rows = _validation_prediction_rows(
            runtime_root
            / "runs"
            / architecture
            / f"seed-{seed_runtime.identity.seed}"
            / "validation-predictions.json"
        )
        raw_logits = np.asarray([row["raw_logit"] for row in rows], dtype=np.float64)
        all_targets = np.asarray(
            [row["failure_target"] for row in rows], dtype=np.int64
        )
        if (
            compute_validation_prediction_digest(raw_logits, all_targets)
            != seed_runtime.identity.validation_prediction_digest
        ):
            _fail(
                "validation predictions",
                f"seed {seed_runtime.identity.seed} content digest differs",
            )
        if reference_all is None:
            reference_all = rows
        else:
            complete_keys = (
                "sample_id",
                "group_id",
                "source_trajectory_id",
                "candidate_type",
                "failure_target",
            )
            if tuple(tuple(row[key] for key in complete_keys) for row in rows) != tuple(
                tuple(row[key] for key in complete_keys) for row in reference_all
            ):
                _fail(
                    "validation predictions",
                    "five seeds cover different complete ordered rows",
                )
        corrupted = tuple(row for row in rows if row["candidate_type"] == "corrupted")
        if not corrupted:
            _fail("validation predictions", "no corrupted candidates are available")
        if reference is None:
            reference = corrupted
        else:
            keys = ("sample_id", "group_id", "source_trajectory_id", "failure_target")
            if tuple(tuple(row[key] for key in keys) for row in corrupted) != tuple(
                tuple(row[key] for key in keys) for row in reference
            ):
                _fail("validation predictions", "five seeds cover different samples")
        logits = np.asarray([row["raw_logit"] for row in corrupted], dtype=np.float64)
        per_seed.append(seed_runtime.calibration.apply(logits))
    assert reference is not None
    return fit_validation_ensemble_abstention_policies(
        np.stack(per_seed),
        np.asarray([row["failure_target"] for row in reference], dtype=np.int64),
        tuple(cast(str, row["group_id"]) for row in reference),
        tuple(cast(str, row["sample_id"]) for row in reference),
        split="validation",
        verifier_bundle_digest=loaded.identity.content_digest,
        validation_split_digest=cast(str, protocol.payload["accepted_split_digest"]),
        target_coverages=cast(Sequence[float], protocol.payload["coverage_targets"]),
    )


def _policy_payload(policies: Sequence[object]) -> dict[str, object]:
    values = tuple(cast(Any, item) for item in policies)
    return {
        "policy_content_digests": [item.content_digest for item in values],
        "policies": [item.as_mapping() for item in values],
        "schema_version": "1.0",
    }


def _load_policies(
    path: Path,
    *,
    expected_report_digest: str | None = None,
    expected_split_digest: str | None = None,
    expected_bundle_digest: str | None = None,
) -> tuple[object, ...]:
    from latentguard.selection.selectors import AbstentionPolicyV1
    from latentguard.training.reporting import load_strict_report

    report = load_strict_report(
        path,
        expected_report_type="m3c_temporal_abstention_v1",
        expected_content_digest=expected_report_digest,
    )
    payload = report.payload
    if (
        set(payload) != {"policy_content_digests", "policies", "schema_version"}
        or payload["schema_version"] != "1.0"
    ):
        _fail("abstention policy report", "unexpected fields or version")
    raw_policies = payload["policies"]
    if not isinstance(raw_policies, list) or len(raw_policies) != 6:
        _fail("abstention policy report", "expected exactly six policies")
    policies: list[AbstentionPolicyV1] = []
    fields = set(AbstentionPolicyV1.__dataclass_fields__)
    for raw in raw_policies:
        if not isinstance(raw, dict) or set(raw) != fields:
            _fail("abstention policy report", "policy has wrong fields")
        policies.append(AbstentionPolicyV1(**raw))
    digests = payload["policy_content_digests"]
    if not isinstance(digests, list) or digests != [
        item.content_digest for item in policies
    ]:
        _fail("abstention policy report", "policy digest inventory changed")
    expected_ids = (
        "maximum_validation_balanced_accuracy",
        "target_validation_failure_recall",
        "target_validation_coverage_90",
        "target_validation_coverage_80",
        "target_validation_coverage_70",
        "target_validation_coverage_50",
    )
    if tuple(item.policy_id for item in policies) != expected_ids:
        _fail("abstention policy report", "policy ID inventory or order changed")
    if len({item.policy_id for item in policies}) != len(expected_ids):
        _fail("abstention policy report", "policy ID is duplicated")
    common = {
        (
            item.validation_prediction_digest,
            item.verifier_bundle_digest,
            item.validation_split_digest,
            item.validation_group_count,
        )
        for item in policies
    }
    if len(common) != 1:
        _fail("abstention policy report", "policy validation bindings differ")
    (_, bundle_digest, split_digest, _) = next(iter(common))
    if expected_split_digest is not None and split_digest != expected_split_digest:
        _fail("abstention policy report", "validation split identity differs")
    if expected_bundle_digest is not None and bundle_digest != expected_bundle_digest:
        _fail("abstention policy report", "temporal bundle identity differs")
    return tuple(policies)


def _precollection_inference_smoke_payload(
    loaded_bundles: Mapping[str, object], *, device: str
) -> dict[str, object]:
    """Run a fixed outcome-free forward gate before any M3C source collection."""

    from latentguard.integrations.maniskill_pickcube.verifier_state import (
        PickCubeVerifierStateSchemaV1,
        PickCubeVerifierStateV1,
    )
    from latentguard.selection.checkpoint_bundle import (
        SUPPORTED_SELECTION_ARCHITECTURES,
    )
    from latentguard.selection.ensemble import score_verifier_ensemble
    from latentguard.selection.models import array_content_digest

    if set(loaded_bundles) != set(SUPPORTED_SELECTION_ARCHITECTURES):
        _fail("pre-collection inference smoke", "expected all three architectures")
    state = PickCubeVerifierStateV1(
        schema=PickCubeVerifierStateSchemaV1(joint_names=_PANDA_JOINT_NAMES),
        values=np.linspace(-0.25, 0.25, 38, dtype=np.float32),
    )
    candidate_ids = tuple(f"m3c-inference-smoke-{index:02d}" for index in range(8))
    actions = np.empty((8, 16, 8), dtype=np.dtype("<f8"))
    for index in range(8):
        actions[index].fill((index - 3.5) / 100.0)
    masks = np.ones((8, 16), dtype=np.bool_)
    architectures: dict[str, object] = {}
    for architecture in sorted(SUPPORTED_SELECTION_ARCHITECTURES):
        bundle = cast(Any, loaded_bundles[architecture])
        result = score_verifier_ensemble(
            bundle,
            state,
            actions,
            masks,
            candidate_ids,
            device=device,
        )
        architectures[architecture] = {
            "bundle_digest": bundle.identity.content_digest,
            "candidate_ranking_digest": "sha256:"
            + hashlib.sha256(
                canonical_json_bytes(
                    list(result.ranking),
                    context="M3CPrecollectionInferenceSmokeRankingV1",
                )
            ).hexdigest(),
            "ensemble_failure_probability_digest": array_content_digest(
                result.ensemble_failure_probabilities
            ),
            "finite_output_verified": True,
            "per_seed_calibrated_probability_digest": array_content_digest(
                result.per_seed_calibrated_failure_probabilities
            ),
            "per_seed_raw_logit_digest": array_content_digest(
                result.per_seed_raw_logits
            ),
            "selected_candidate_id": result.selected_candidate_id,
        }
    return {
        "action_dimension": 8,
        "action_horizon": 16,
        "architecture_count": len(architectures),
        "architectures": architectures,
        "candidate_count": len(candidate_ids),
        "candidate_input_digest": "sha256:"
        + hashlib.sha256(
            canonical_json_bytes(
                {
                    "action_digest": array_content_digest(actions),
                    "candidate_ids": list(candidate_ids),
                    "mask_digest": array_content_digest(masks),
                    "state_digest": state.content_digest,
                },
                context="M3CPrecollectionInferenceSmokeInputV1",
            )
        ).hexdigest(),
        "device": device,
        "device_kind": str(device).split(":", maxsplit=1)[0],
        "measurement_iterations": 5,
        "m3c_candidate_outcomes_loaded": False,
        "m3c_source_trajectories_loaded": False,
        "outcome_free_synthetic_fixture": True,
        "raw_prediction_values_persisted": False,
        "schema_version": "1.0",
        "semantic": "fixed_outcome_free_three_ensemble_forward_gate_v1",
        "state_dimension": 38,
        "warmup_iterations": 2,
    }


def _run_prepare_selection_checkpoints(args: argparse.Namespace) -> int:
    from latentguard.selection.checkpoint_bundle import (
        SUPPORTED_SELECTION_ARCHITECTURES,
        load_verifier_bundle,
    )
    from latentguard.selection.preparation import (
        PreparedVerifierBundlesV1,
        build_verifier_bundle_identities,
        prepare_verifier_bundles,
    )
    from latentguard.training.baselines import fit_action_magnitude_baseline
    from latentguard.training.reporting import load_strict_report, save_strict_report

    protocol = _load_protocol(args.config, mode=args.mode)
    committed_benchmark_digest = _validate_m3b_runtime_artifact_digests(
        args.m3b_result_root, args.m3b_runtime_root
    )
    dataset = _load_accepted_dataset(args)
    if cast(Any, dataset).dataset_digest != protocol.payload["accepted_dataset_digest"]:
        _fail("accepted dataset", "digest differs from frozen M3A identity")
    if cast(Any, dataset).split_digest != protocol.payload["accepted_split_digest"]:
        _fail("accepted dataset", "split digest differs from frozen M3B identity")
    baseline = fit_action_magnitude_baseline(cast(Any, dataset))
    if baseline.content_digest != protocol.payload["action_magnitude_baseline_digest"]:
        _fail("action magnitude baseline", "frozen identity was not reproduced")
    if args.dry_run:
        bundles, _, selection_digest, preprocessing_digest = (
            build_verifier_bundle_identities(
                args.m3b_result_root, runtime_root=args.m3b_runtime_root
            )
        )
        loaded = {
            name: load_verifier_bundle(
                bundle,
                cast(Any, _runtime_artifacts(args.m3b_runtime_root, name)),
                device=args.device,
            )
            for name, bundle in bundles.items()
        }
        prepared = PreparedVerifierBundlesV1(
            bundles=bundles,
            loaded=loaded,
            selection_record_digest=selection_digest,
            preprocessing_digest=preprocessing_digest,
        )
    else:
        _require_separate_paths(
            inputs=(args.m3b_result_root, args.m3b_runtime_root, args.dataset_root),
            outputs=(args.output_dir,),
        )
        prepared = prepare_verifier_bundles(
            args.m3b_result_root,
            args.output_dir,
            runtime_root=args.m3b_runtime_root,
            device=args.device,
            validate_runtime=True,
            resume=args.resume,
        )
    if set(prepared.loaded) != set(SUPPORTED_SELECTION_ARCHITECTURES):
        _fail("prepared bundles", "all three five-seed ensembles must be loaded")
    if (
        prepared.preprocessing_digest
        != protocol.payload["accepted_preprocessing_digest"]
    ):
        _fail("prepared bundles", "preprocessing digest differs from frozen identity")
    inference_smoke_report = cast(
        Any,
        _strict_report_payload(
            "m3c_precollection_inference_smoke_v1",
            _precollection_inference_smoke_payload(
                prepared.loaded,
                device=args.device,
            ),
        ),
    )
    inference_smoke_path = args.inference_smoke_output
    if inference_smoke_path is None and not args.dry_run:
        inference_smoke_path = args.output_dir / (
            f"{PRECOLLECTION_INFERENCE_SMOKE_REPORT}-"
            f"{str(args.device).replace(':', '-')}.json"
        )
    if inference_smoke_path is not None:
        save_strict_report(inference_smoke_report, inference_smoke_path)
        load_strict_report(
            inference_smoke_path,
            expected_report_type="m3c_precollection_inference_smoke_v1",
            expected_content_digest=inference_smoke_report.content_digest,
        )
    temporal = prepared.loaded["temporal_state_action_verifier"]
    policies = _fit_temporal_policies(temporal, args.m3b_runtime_root, protocol)
    if not args.dry_run:
        baseline_report = cast(
            Any,
            _strict_report_payload(
                "m3c_action_magnitude_v1", _baseline_payload(baseline)
            ),
        )
        policy_report = cast(
            Any,
            _strict_report_payload(
                "m3c_temporal_abstention_v1", _policy_payload(policies)
            ),
        )
        save_strict_report(
            baseline_report,
            args.output_dir / ACTION_MAGNITUDE_REPORT,
        )
        save_strict_report(
            policy_report,
            args.output_dir / ABSTENTION_POLICY_REPORT,
        )
        _load_policies(
            args.output_dir / ABSTENTION_POLICY_REPORT,
            expected_report_digest=policy_report.content_digest,
            expected_split_digest=cast(str, protocol.payload["accepted_split_digest"]),
            expected_bundle_digest=temporal.identity.content_digest,
        )
        bundle_summary = load_strict_report(
            args.output_dir / "summary.json",
            expected_report_type="m3c_verifier_bundle_summary_v1",
        )
        preparation_summary = cast(
            Any,
            _strict_report_payload(
                "m3c_selection_preparation_v1",
                {
                    "action_magnitude_baseline_digest": baseline.content_digest,
                    "action_magnitude_report_digest": baseline_report.content_digest,
                    "bundle_digests": {
                        name: bundle.content_digest
                        for name, bundle in prepared.bundles.items()
                    },
                    "bundle_summary_report_digest": bundle_summary.content_digest,
                    "committed_benchmark_summary_digest": (committed_benchmark_digest),
                    "inference_smoke_device": args.device,
                    "inference_smoke_report_digest": (
                        inference_smoke_report.content_digest
                    ),
                    "policy_content_digests": [
                        cast(Any, item).content_digest for item in policies
                    ],
                    "policy_report_digest": policy_report.content_digest,
                    "preprocessing_digest": prepared.preprocessing_digest,
                    "protocol_digest": protocol.content_digest,
                    "schema_version": "1.0",
                    "selection_record_digest": prepared.selection_record_digest,
                },
            ),
        )
        save_strict_report(
            preparation_summary,
            args.output_dir / PREPARATION_SUMMARY_REPORT,
        )
    print(
        f"prepare-selection-checkpoints {'dry-run ' if args.dry_run else ''}OK: "
        f"architectures=3 checkpoints=15 policies={len(policies)} "
        f"baseline={baseline.content_digest} "
        f"inference_smoke={inference_smoke_report.content_digest} "
        f"output_created={str(not args.dry_run).lower()}"
    )
    return 0


@dataclass(frozen=True, slots=True)
class _PickCubeScope:
    binding: object
    layout: object
    action_contract: object


def _load_pickcube_scope(args: argparse.Namespace) -> _PickCubeScope:
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
    )

    expected = load_expected_contract(args.expected_contract)
    report = load_compatibility_report(args.compatibility_report)
    binding = validate_compatibility_report(report, expected, require_trusted=True)
    layout = load_maniskill_pickcube_action_layout(args.action_layout)
    validate_maniskill_pickcube_action_layout_binding(
        layout, binding, layout.m1_action_layout
    )
    contract = action_contract_from_compatibility(
        binding, coordinate_frame=layout.coordinate_frame
    )
    return _PickCubeScope(binding=binding, layout=layout, action_contract=contract)


def _validate_candidate_runtime_action_contract(
    configuration: object, scope: _PickCubeScope
) -> None:
    """Bind candidate generation to the trusted environment action contract."""

    observed = cast(Any, configuration).action_contract_digest
    expected = cast(Any, scope.layout).action_contract_digest
    if observed != expected:
        _fail(
            "candidate configuration",
            "action contract digest differs from trusted runtime layout "
            f"(expected {expected}, observed {observed})",
        )


def _build_exclusion_inventory(
    m3a_dataset_dir: Path,
    m3a_anchor_manifest_dir: Path,
    protocol: _M3CProtocolConfig,
) -> object:
    from latentguard.action_verifier import load_action_verifier_dataset
    from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
        load_anchor_manifest,
    )
    from latentguard.selection.models import SourceExclusionInventoryV1

    dataset = load_action_verifier_dataset(m3a_dataset_dir)
    if dataset.content_digest != protocol.payload["accepted_dataset_digest"]:
        _fail("M3A exclusion dataset", "content digest differs from accepted data")
    manifest = load_anchor_manifest(m3a_anchor_manifest_dir)
    return SourceExclusionInventoryV1(
        dataset_digest=dataset.content_digest,
        anchor_manifest_digest=manifest.content_digest,
        reset_seeds=tuple(item.source_seed for item in dataset.split_assignments),
        source_trajectory_ids=tuple(
            item.source_trajectory_id for item in dataset.split_assignments
        ),
        split_group_ids=tuple(
            item.split_group_id for item in dataset.split_assignments
        ),
        complete_state_digests=tuple(
            digest
            for item in dataset.split_assignments
            for digest in item.state_digests
        ),
    )


def _validate_source_mode(
    pool: object, protocol: _M3CProtocolConfig, *, require_exact: bool = True
) -> None:
    value = cast(Any, pool)
    trajectories: dict[str, int] = {}
    for group in value.groups:
        trajectories[group.source_trajectory_id] = (
            trajectories.get(group.source_trajectory_id, 0) + 1
        )
    if require_exact and len(trajectories) != protocol.expected_trajectory_count:
        _fail(
            "candidate pool mode",
            f"{protocol.mode} requires exactly {protocol.expected_trajectory_count} "
            f"source trajectories, observed {len(trajectories)}",
        )
    maximum = cast(int, protocol.payload["anchors_per_trajectory"])
    if not trajectories or any(
        count <= 0 or count > maximum for count in trajectories.values()
    ):
        _fail(
            "candidate pool mode",
            "anchor count exceeds the frozen per-trajectory bound",
        )
    seed_start = cast(int, protocol.payload[f"{protocol.mode}_starting_seed"])
    seed_stop = seed_start + cast(
        int, protocol.payload[f"{protocol.mode}_maximum_attempts"]
    )
    seeds = {group.trajectory.source_seed for group in cast(Any, pool).groups}
    if any(seed < seed_start or seed >= seed_stop for seed in seeds):
        _fail(
            "candidate pool mode",
            f"source seeds must stay in frozen [{seed_start}, {seed_stop}) range",
        )


def _candidate_sources(
    *,
    source_dir: Path,
    runtime_archive_dir: Path,
    anchor_manifest_dir: Path,
    protocol: _M3CProtocolConfig,
) -> tuple[tuple[object, ...], object, object, tuple[object, ...]]:
    from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
        load_state_indexed_archive,
    )
    from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
        load_anchor_manifest,
    )
    from latentguard.selection.source import load_candidate_pool_sources
    from latentguard.serialization import load_episodes

    archive = load_state_indexed_archive(runtime_archive_dir)
    manifest = load_anchor_manifest(anchor_manifest_dir)
    episodes = tuple(load_episodes(source_dir))
    sources = load_candidate_pool_sources(
        source_dir,
        runtime_archive_dir,
        anchor_manifest_dir,
    )
    if len({episode.source_trajectory_id for episode in archive.episodes}) != (
        protocol.expected_trajectory_count
    ):
        _fail(
            "candidate sources", "archive trajectory count differs from selected mode"
        )
    return tuple(sources), archive, manifest, episodes


def _corruption_dataset_for_pool(
    *,
    pool: object,
    sources: Sequence[object],
    configuration: object,
    action_layout: object,
    source_dataset_id: str,
) -> object:
    from latentguard.selection.replay_dataset import build_replay_corruption_dataset

    return build_replay_corruption_dataset(
        cast(Any, sources),
        cast(Any, pool),
        cast(Any, configuration),
        cast(Any, action_layout),
        source_dataset_id=source_dataset_id,
    )


def _require_smoke_full_disjointness(
    pool: object,
    protocol: _M3CProtocolConfig,
    smoke_candidate_pool_dir: Path | None,
) -> str | None:
    from latentguard.selection.serialization import load_candidate_pool
    from latentguard.selection.source import require_disjoint_source_sets

    if protocol.mode == "smoke":
        if smoke_candidate_pool_dir is not None:
            _fail("smoke source inventory", "smoke mode cannot reference itself")
        return None
    if smoke_candidate_pool_dir is None:
        _fail(
            "full source inventory",
            "full mode requires --smoke-candidate-pool-dir",
        )
    smoke_pool = load_candidate_pool(smoke_candidate_pool_dir)
    smoke_protocol = protocol.for_mode("smoke")
    _validate_source_mode(smoke_pool, smoke_protocol)
    require_disjoint_source_sets(
        cast(Any, smoke_pool.groups),
        cast(Any, cast(Any, pool).groups),
    )
    return smoke_pool.content_digest


def _run_build_blind_candidate_pools(args: argparse.Namespace) -> int:
    from latentguard.corruptions.serialization import (
        load_corruption_dataset,
        save_corruption_dataset,
    )
    from latentguard.evaluation.serialization import (
        compute_corruption_dataset_content_digest,
    )
    from latentguard.selection.blind_input import (
        build_blind_candidate_pool,
        load_blind_candidate_pool,
        save_blind_candidate_pool,
        validate_blind_candidate_pool_against_full_pool,
    )
    from latentguard.selection.candidate_pool import build_candidate_pool
    from latentguard.selection.configuration import load_candidate_pool_configuration
    from latentguard.selection.serialization import (
        load_candidate_pool,
        save_candidate_pool,
    )
    from latentguard.serialization import compute_episode_bundle_identifier
    from latentguard.training.reporting import (
        load_strict_report,
        save_strict_report,
    )

    protocol = _load_protocol(args.config, mode=args.mode)
    configuration = load_candidate_pool_configuration(args.candidate_config)
    scope = _load_pickcube_scope(args)
    _validate_candidate_runtime_action_contract(configuration, scope)
    sources, _, _, _ = _candidate_sources(
        source_dir=args.source_dir,
        runtime_archive_dir=args.runtime_archive_dir,
        anchor_manifest_dir=args.anchor_manifest_dir,
        protocol=protocol,
    )
    source_dataset_id = compute_episode_bundle_identifier(args.source_dir)
    if args.resume:
        pool = load_candidate_pool(args.candidate_pool_dir)
        blind_input = load_blind_candidate_pool(args.blind_input_dir)
        validate_blind_candidate_pool_against_full_pool(blind_input, pool)
        corruptions = load_corruption_dataset(args.corruption_dir)
        if pool.candidate_pool_configuration_digest != configuration.content_digest:
            _fail("resume", "candidate configuration differs from persisted pool")
        expected_corruptions = _corruption_dataset_for_pool(
            pool=pool,
            sources=sources,
            configuration=configuration,
            action_layout=cast(Any, scope.layout).m1_action_layout,
            source_dataset_id=source_dataset_id,
        )
        if compute_corruption_dataset_content_digest(
            cast(Any, expected_corruptions)
        ) != compute_corruption_dataset_content_digest(corruptions):
            _fail("resume", "replay dataset content differs from frozen pool")
        _validate_source_mode(pool, protocol)
        smoke_digest = _require_smoke_full_disjointness(
            pool, protocol, args.smoke_candidate_pool_dir
        )
        report = load_strict_report(
            args.summary_output,
            expected_report_type="m3c_candidate_pool_build_v1",
        )
        if (
            report.payload.get("candidate_pool_digest") != pool.content_digest
            or report.payload.get("blind_input_digest") != blind_input.content_digest
            or report.payload.get("smoke_candidate_pool_digest") != smoke_digest
        ):
            _fail("resume", "build summary identity differs")
        print(
            f"build-blind-candidate-pools resume OK: groups={len(pool.groups)} "
            f"candidates={len(pool.proposal_ids)} digest={pool.content_digest}"
        )
        return 0
    _require_separate_paths(
        inputs=(
            args.runtime_archive_dir,
            args.source_dir,
            args.anchor_manifest_dir,
            args.m3a_dataset_dir,
            args.m3a_anchor_manifest_dir,
            *(
                (args.smoke_candidate_pool_dir,)
                if args.smoke_candidate_pool_dir
                else ()
            ),
        ),
        outputs=(
            args.candidate_pool_dir,
            args.blind_input_dir,
            args.corruption_dir,
            args.summary_output,
        ),
    )
    if any(
        path.exists() or path.is_symlink()
        for path in (
            args.candidate_pool_dir,
            args.blind_input_dir,
            args.corruption_dir,
            args.summary_output,
        )
    ):
        _fail("outputs", "first build requires absent output roots")
    exclusions = _build_exclusion_inventory(
        args.m3a_dataset_dir, args.m3a_anchor_manifest_dir, protocol
    )
    pool = build_candidate_pool(
        cast(Any, sources),
        configuration=configuration,
        action_layout=cast(Any, scope.layout).m1_action_layout,
        action_contract=cast(Any, scope.action_contract),
        exclusion_inventory=cast(Any, exclusions),
    )
    _validate_source_mode(pool, protocol)
    smoke_digest = _require_smoke_full_disjointness(
        pool, protocol, args.smoke_candidate_pool_dir
    )
    built_corruptions = _corruption_dataset_for_pool(
        pool=pool,
        sources=sources,
        configuration=configuration,
        action_layout=cast(Any, scope.layout).m1_action_layout,
        source_dataset_id=source_dataset_id,
    )
    blind_input = build_blind_candidate_pool(pool)
    if args.dry_run:
        print(
            f"build-blind-candidate-pools dry-run OK: groups={len(pool.groups)} "
            f"candidates={len(pool.proposal_ids)} output_created=false"
        )
        return 0
    try:
        save_candidate_pool(pool, args.candidate_pool_dir)
        save_blind_candidate_pool(blind_input, args.blind_input_dir)
        save_corruption_dataset(cast(Any, built_corruptions), args.corruption_dir)
        reloaded_pool = load_candidate_pool(
            args.candidate_pool_dir, expected_content_digest=pool.content_digest
        )
        reloaded_blind_input = load_blind_candidate_pool(
            args.blind_input_dir,
            expected_content_digest=blind_input.content_digest,
        )
        validate_blind_candidate_pool_against_full_pool(
            reloaded_blind_input, reloaded_pool
        )
        reloaded_corruptions = load_corruption_dataset(args.corruption_dir)
        if compute_corruption_dataset_content_digest(
            cast(Any, built_corruptions)
        ) != compute_corruption_dataset_content_digest(reloaded_corruptions):
            _fail("published pool", "replay dataset content changed")
        summary = cast(
            Any,
            _strict_report_payload(
                "m3c_candidate_pool_build_v1",
                {
                    "candidate_count": len(pool.proposal_ids),
                    "blind_input_digest": reloaded_blind_input.content_digest,
                    "candidate_pool_configuration_digest": (
                        pool.candidate_pool_configuration_digest
                    ),
                    "candidate_pool_digest": pool.content_digest,
                    "corruption_dataset_digest": (
                        compute_corruption_dataset_content_digest(reloaded_corruptions)
                    ),
                    "group_count": len(pool.groups),
                    "mode": args.mode,
                    "protocol_digest": protocol.content_digest,
                    "schema_version": "1.0",
                    "smoke_candidate_pool_digest": smoke_digest,
                    "smoke_full_disjoint_on_four_axes": (
                        True if args.mode == "full" else None
                    ),
                    "source_set_digest": pool.source_set_digest,
                },
            ),
        )
        save_strict_report(summary, args.summary_output)
    except Exception:
        for path in (
            args.candidate_pool_dir,
            args.blind_input_dir,
            args.corruption_dir,
            args.summary_output,
        ):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink(missing_ok=True)
        raise
    print(
        f"build-blind-candidate-pools OK: groups={len(pool.groups)} "
        f"candidates={len(pool.proposal_ids)} digest={pool.content_digest}"
    )
    return 0


_PANDA_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
)


def _state_for_group(group: object) -> object:
    from latentguard.integrations.maniskill_pickcube.verifier_state import (
        PickCubeVerifierStateSchemaV1,
        PickCubeVerifierStateV1,
    )
    from latentguard.selection.blind_input import BlindCandidateGroupV1

    if not isinstance(group, BlindCandidateGroupV1):
        _fail("candidate state", "Stage A requires BlindCandidateGroupV1")
    state = PickCubeVerifierStateV1(
        schema=PickCubeVerifierStateSchemaV1(joint_names=_PANDA_JOINT_NAMES),
        values=group.state_vector,
    )
    return state


def _load_selection_bundles(
    prepared_root: Path, runtime_root: Path, *, device: str
) -> Mapping[str, object]:
    from latentguard.selection.checkpoint_bundle import (
        SUPPORTED_SELECTION_ARCHITECTURES,
        load_verifier_bundle,
    )
    from latentguard.selection.preparation import load_prepared_bundle

    loaded: dict[str, object] = {}
    for architecture in sorted(SUPPORTED_SELECTION_ARCHITECTURES):
        identity = load_prepared_bundle(prepared_root / f"{architecture}.json")
        loaded[architecture] = load_verifier_bundle(
            identity,
            cast(Any, _runtime_artifacts(runtime_root, architecture)),
            device=device,
        )
    return loaded


def _load_preparation_summary(
    prepared_root: Path, protocol: _M3CProtocolConfig
) -> Mapping[str, object]:
    from latentguard.training.reporting import load_strict_report

    report = load_strict_report(
        prepared_root / PREPARATION_SUMMARY_REPORT,
        expected_report_type="m3c_selection_preparation_v1",
    )
    expected = {
        "action_magnitude_baseline_digest",
        "action_magnitude_report_digest",
        "bundle_digests",
        "bundle_summary_report_digest",
        "committed_benchmark_summary_digest",
        "inference_smoke_device",
        "inference_smoke_report_digest",
        "policy_content_digests",
        "policy_report_digest",
        "preprocessing_digest",
        "protocol_digest",
        "schema_version",
        "selection_record_digest",
    }
    payload = report.payload
    if set(payload) != expected or payload["schema_version"] != "1.0":
        _fail("preparation summary", "unexpected fields or schema version")
    if payload["protocol_digest"] != protocol.content_digest:
        _fail("preparation summary", "protocol identity differs")
    if (
        payload["action_magnitude_baseline_digest"]
        != protocol.payload["action_magnitude_baseline_digest"]
    ):
        _fail("preparation summary", "action-magnitude identity differs")
    if (
        payload["preprocessing_digest"]
        != protocol.payload["accepted_preprocessing_digest"]
    ):
        _fail("preparation summary", "preprocessing identity differs")
    for name in (
        "action_magnitude_report_digest",
        "bundle_summary_report_digest",
        "committed_benchmark_summary_digest",
        "inference_smoke_report_digest",
        "policy_report_digest",
        "selection_record_digest",
    ):
        _digest(payload[name], f"preparation summary.{name}")
    inference_smoke_device = payload["inference_smoke_device"]
    if not isinstance(inference_smoke_device, str) or not inference_smoke_device:
        _fail("preparation summary.inference_smoke_device", "expected text")
    if not isinstance(payload["bundle_digests"], Mapping) or not isinstance(
        payload["policy_content_digests"], list
    ):
        _fail("preparation summary", "digest inventories are invalid")
    return payload


def _random_selector_seed(protocol: _M3CProtocolConfig) -> int:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "protocol_digest": protocol.content_digest,
                "semantic": "m3c_deterministic_random_selector_seed_v1",
            },
            context="M3CRandomSelectorSeedV1",
        )
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _selector_ids(policies: Sequence[object]) -> tuple[str, ...]:
    return (
        "deterministic_random_v1",
        "frozen_m3b_action_magnitude_v1",
        "action_only_ensemble_v1",
        "state_action_mlp_ensemble_v1",
        "temporal_ensemble_v1",
        *(
            f"temporal_ensemble_abstention_{cast(Any, item).policy_id}_v1"
            for item in policies
        ),
    )


def _selector_configuration_payload(
    *,
    protocol: _M3CProtocolConfig,
    blind_input: object,
    bundles: Mapping[str, object],
    baseline: object,
    policies: Sequence[object],
) -> dict[str, object]:
    from latentguard.selection.blind_protocol import (
        EXPECTED_STAGE_A_SELECTOR_IDS,
        EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS,
    )

    selector_ids = _selector_ids(policies)
    if selector_ids != EXPECTED_STAGE_A_SELECTOR_IDS:
        _fail("selector configuration", "selector inventory differs")
    if tuple(sorted(bundles)) != EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS:
        _fail("selector configuration", "verifier bundle inventory differs")
    return {
        "action_magnitude_baseline_digest": cast(Any, baseline).content_digest,
        "blind_input_digest": cast(Any, blind_input).content_digest,
        "candidate_pool_configuration_digest": cast(
            Any, blind_input
        ).candidate_pool_configuration_digest,
        "candidate_pool_digest": cast(Any, blind_input).full_candidate_pool_digest,
        "inference_measurement_iterations": 5,
        "inference_warmup_iterations": 2,
        "protocol_digest": protocol.content_digest,
        "random_selector_seed": _random_selector_seed(protocol),
        "schema_version": "1.0",
        "selector_ids": list(selector_ids),
        "selector_semantic": "blind_identical_pool_selection_v1",
        "temporal_abstention_policy_digests": [
            cast(Any, item).content_digest for item in policies
        ],
        "verifier_bundle_digests": {
            name: cast(Any, bundle).identity.content_digest
            for name, bundle in sorted(bundles.items())
        },
    }


def _selector_configuration_digest(
    *,
    protocol: _M3CProtocolConfig,
    blind_input: object,
    bundles: Mapping[str, object],
    baseline: object,
    policies: Sequence[object],
) -> str:
    payload = _selector_configuration_payload(
        protocol=protocol,
        blind_input=blind_input,
        bundles=bundles,
        baseline=baseline,
        policies=policies,
    )
    return (
        "sha256:"
        + hashlib.sha256(
            canonical_json_bytes(payload, context="M3CSelectorConfigurationV1")
        ).hexdigest()
    )


def _load_selector_configuration_report(
    path: Path,
    *,
    expected_digest: str,
    protocol: _M3CProtocolConfig,
    blind_input: object,
    expected_bundle_digests: Mapping[str, str],
    selections: Sequence[object],
) -> Mapping[str, object]:
    from latentguard.selection.blind_protocol import (
        EXPECTED_STAGE_A_SELECTOR_IDS,
        EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS,
    )
    from latentguard.training.reporting import load_strict_report

    report = load_strict_report(
        path,
        expected_report_type="m3c_selector_configuration_v1",
        expected_content_digest=expected_digest,
    )
    expected_fields = {
        "action_magnitude_baseline_digest",
        "blind_input_digest",
        "candidate_pool_digest",
        "candidate_pool_configuration_digest",
        "inference_measurement_iterations",
        "inference_warmup_iterations",
        "protocol_digest",
        "random_selector_seed",
        "schema_version",
        "selector_ids",
        "selector_semantic",
        "temporal_abstention_policy_digests",
        "verifier_bundle_digests",
    }
    payload = report.payload
    if (
        set(payload) != expected_fields
        or payload["schema_version"] != "1.0"
        or payload["selector_semantic"] != "blind_identical_pool_selection_v1"
        or payload["protocol_digest"] != protocol.content_digest
        or payload["blind_input_digest"] != cast(Any, blind_input).content_digest
        or payload["candidate_pool_digest"]
        != cast(Any, blind_input).full_candidate_pool_digest
        or payload["candidate_pool_configuration_digest"]
        != cast(Any, blind_input).candidate_pool_configuration_digest
        or payload["action_magnitude_baseline_digest"]
        != protocol.payload["action_magnitude_baseline_digest"]
        or payload["random_selector_seed"] != _random_selector_seed(protocol)
        or payload["inference_warmup_iterations"] != 2
        or payload["inference_measurement_iterations"] != 5
        or tuple(cast(Sequence[str], payload["selector_ids"]))
        != EXPECTED_STAGE_A_SELECTOR_IDS
        or tuple(sorted(expected_bundle_digests))
        != EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS
        or dict(cast(Mapping[str, str], payload["verifier_bundle_digests"]))
        != dict(expected_bundle_digests)
    ):
        _fail("selector configuration", "content differs from frozen contract")
    policy_digests = payload["temporal_abstention_policy_digests"]
    if (
        not isinstance(policy_digests, list)
        or len(policy_digests) != 6
        or len(set(policy_digests)) != 6
    ):
        _fail("selector configuration", "policy digest inventory differs")
    for digest in policy_digests:
        _digest(digest, "selector configuration.policy digest")
    policy_selector_ids = EXPECTED_STAGE_A_SELECTOR_IDS[5:]
    policy_by_selector = dict(zip(policy_selector_ids, policy_digests, strict=True))
    decision_policy_ids: dict[str, set[object]] = {}
    for decision in selections:
        selector_id = cast(Any, decision).selector_id
        policy_id = cast(Any, decision).abstention_policy_id
        decision_policy_ids.setdefault(selector_id, set()).add(policy_id)
    if set(decision_policy_ids) != set(EXPECTED_STAGE_A_SELECTOR_IDS):
        _fail("selector configuration", "manifest selector inventory differs")
    for selector_id, observed in decision_policy_ids.items():
        expected_policy = policy_by_selector.get(selector_id)
        expected = {expected_policy} if expected_policy is not None else {None}
        if observed != expected:
            _fail(
                "selector configuration",
                "manifest abstention-policy binding differs",
            )
    return payload


def _stage_a_decisions(
    *,
    blind_input: object,
    bundles: Mapping[str, object],
    baseline: object,
    policies: Sequence[object],
    protocol: _M3CProtocolConfig,
    device: str,
    diagnostics_out: dict[str, list[object]] | None = None,
) -> tuple[object, ...]:
    from latentguard.selection.blind_input import BlindCandidatePoolV1
    from latentguard.selection.ensemble import score_verifier_ensemble
    from latentguard.selection.selectors import (
        action_magnitude_decision,
        deterministic_random_decision,
        learned_ensemble_decision,
    )

    if not isinstance(blind_input, BlindCandidatePoolV1):
        _fail("selectors", "Stage A requires BlindCandidatePoolV1")
    architecture_ids = {
        "action_only_mlp": "action_only_ensemble_v1",
        "state_action_mlp": "state_action_mlp_ensemble_v1",
        "temporal_state_action_verifier": "temporal_ensemble_v1",
    }
    values: list[object] = []
    for group in blind_input.groups:
        values.append(
            deterministic_random_decision(
                group,
                seed=_random_selector_seed(protocol),
                selector_id="deterministic_random_v1",
            )
        )
        values.append(
            action_magnitude_decision(
                group,
                cast(Any, baseline),
                expected_baseline_digest=cast(
                    str, protocol.payload["action_magnitude_baseline_digest"]
                ),
            )
        )
        state = _state_for_group(group)
        actions = group.action_chunks
        masks = group.action_masks
        temporal_inference: object | None = None
        for architecture, bundle in sorted(bundles.items()):
            inference = score_verifier_ensemble(
                cast(Any, bundle),
                cast(Any, state),
                actions,
                masks,
                group.candidate_ids,
                device=device,
                warmup_iterations=2,
                measurement_iterations=5,
            )
            values.append(
                learned_ensemble_decision(
                    group,
                    inference,
                    selector_id=architecture_ids[architecture],
                )
            )
            if architecture == "temporal_state_action_verifier":
                temporal_inference = inference
            if diagnostics_out is not None:
                diagnostics_out.setdefault(architecture, []).append(
                    inference.diagnostics
                )
        if temporal_inference is None:
            _fail("selectors", "temporal ensemble is absent")
        for policy in policies:
            values.append(
                learned_ensemble_decision(
                    group,
                    cast(Any, temporal_inference),
                    selector_id=(
                        f"temporal_ensemble_abstention_{cast(Any, policy).policy_id}_v1"
                    ),
                    abstention_policy=cast(Any, policy),
                )
            )
    return tuple(values)


def _inference_latency_payload(
    diagnostics: Mapping[str, Sequence[object]], *, device: str
) -> dict[str, object]:
    from latentguard.selection.ensemble import (
        ENSEMBLE_END_TO_END_PROFILE_SEMANTIC,
        ENSEMBLE_PROFILE_MEASUREMENT_ITERATIONS,
        ENSEMBLE_PROFILE_WARMUP_ITERATIONS,
        EnsembleEndToEndProfileV1,
    )
    from latentguard.selection.inference import InferenceLatencyReportV1

    architectures: dict[str, dict[str, object]] = {}
    for architecture, raw_values in sorted(diagnostics.items()):
        values = tuple(cast(Any, item) for item in raw_values)
        if not values:
            _fail("inference latency", "architecture has no group measurements")
        profiles = tuple(value.end_to_end_profile for value in values)
        if any(not isinstance(item, EnsembleEndToEndProfileV1) for item in profiles):
            _fail("inference latency", "missing end-to-end ensemble profile")
        candidate_counts = {item.candidate_count for item in profiles}
        if len(candidate_counts) != 1:
            _fail("inference latency", "candidate count changed across groups")
        candidate_count = next(iter(candidate_counts))
        durations = tuple(
            duration
            for profile in profiles
            for duration in profile.group_durations_seconds
        )
        loading = values[0].bundle_loading
        if any(item.bundle_loading != loading for item in values):
            _fail("inference latency", "bundle loading diagnostics changed by group")
        peak_memory = max(item.peak_allocated_device_memory_bytes for item in profiles)
        aggregate = InferenceLatencyReportV1.from_group_measurements(
            durations,
            device=loading.device,
            candidate_count=candidate_count,
            warmup_iterations=ENSEMBLE_PROFILE_WARMUP_ITERATIONS,
            peak_allocated_device_memory_bytes=peak_memory,
        )
        if any(
            item.warmup_iterations != ENSEMBLE_PROFILE_WARMUP_ITERATIONS
            or item.measurement_iterations != ENSEMBLE_PROFILE_MEASUREMENT_ITERATIONS
            or item.semantic != ENSEMBLE_END_TO_END_PROFILE_SEMANTIC
            for item in profiles
        ):
            _fail("inference latency", "end-to-end measurement semantic changed")
        architectures[architecture] = {
            "bundle_loading_per_seed_model_seconds": list(
                loading.per_seed_model_seconds
            ),
            "bundle_loading_total_seconds": loading.total_seconds,
            "candidate_count_per_group": candidate_count,
            "candidates_per_second": aggregate.candidates_per_second,
            "device": loading.device,
            "group_count": len(values),
            "measurement_iterations_per_group": (
                ENSEMBLE_PROFILE_MEASUREMENT_ITERATIONS
            ),
            "measured_candidate_executions": candidate_count * len(durations),
            "measured_group_repetition_count": len(durations),
            "measurement_scope": ENSEMBLE_END_TO_END_PROFILE_SEMANTIC,
            "model_loading_total_seconds": float(sum(loading.per_seed_model_seconds)),
            "peak_allocated_device_memory_bytes": peak_memory,
            "per_candidate_seconds_p50": aggregate.per_candidate_seconds_p50,
            "per_candidate_seconds_p95": aggregate.per_candidate_seconds_p95,
            "per_candidate_seconds_p99": aggregate.per_candidate_seconds_p99,
            "per_group_seconds_p50": aggregate.per_group_seconds_p50,
            "per_group_seconds_p95": aggregate.per_group_seconds_p95,
            "per_group_seconds_p99": aggregate.per_group_seconds_p99,
            "raw_duration_values_persisted": False,
            "warmup_iterations_per_group": ENSEMBLE_PROFILE_WARMUP_ITERATIONS,
        }
    joint = architectures.get("state_action_mlp")
    temporal = architectures.get("temporal_state_action_verifier")
    if joint is None or temporal is None:
        _fail("inference latency", "joint and temporal summaries are required")
    difference_fields = (
        "bundle_loading_total_seconds",
        "candidates_per_second",
        "model_loading_total_seconds",
        "per_candidate_seconds_p50",
        "per_candidate_seconds_p95",
        "per_candidate_seconds_p99",
        "per_group_seconds_p50",
        "per_group_seconds_p95",
        "per_group_seconds_p99",
    )
    differences: dict[str, object] = {
        name: cast(float, joint[name]) - cast(float, temporal[name])
        for name in difference_fields
    }
    differences["peak_allocated_device_memory_bytes"] = cast(
        int, joint["peak_allocated_device_memory_bytes"]
    ) - cast(int, temporal["peak_allocated_device_memory_bytes"])
    differences["per_seed_model_loading_seconds"] = [
        float(first) - float(second)
        for first, second in zip(
            cast(Sequence[float], joint["bundle_loading_per_seed_model_seconds"]),
            cast(
                Sequence[float],
                temporal["bundle_loading_per_seed_model_seconds"],
            ),
            strict=True,
        )
    ]
    return {
        "architectures": architectures,
        "device": device,
        "joint_minus_temporal": differences,
        "prediction_values_persisted": False,
        "quantile_aggregation_semantic": (
            "linear_quantile_of_all_end_to_end_group_repetitions_v1"
        ),
        "raw_duration_values_persisted": False,
        "schema_version": "1.0",
    }


def _load_bound_inference_latency_report(
    path: Path,
    *,
    expected_device_kind: str,
    protocol: _M3CProtocolConfig,
    blind_input: object,
    selector_configuration_digest: str,
    expected_bundle_digests: Mapping[str, str],
) -> object:
    """Load one compact timing report and verify every frozen M3C binding."""

    from latentguard.selection.blind_protocol import (
        EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS,
    )
    from latentguard.selection.ensemble import (
        ENSEMBLE_END_TO_END_PROFILE_SEMANTIC,
        ENSEMBLE_PROFILE_MEASUREMENT_ITERATIONS,
        ENSEMBLE_PROFILE_WARMUP_ITERATIONS,
    )
    from latentguard.training.reporting import load_strict_report

    report = load_strict_report(
        path,
        expected_report_type="m3c_inference_latency_v1",
    )
    payload = report.payload
    expected_fields = {
        "architectures",
        "blind_input_digest",
        "candidate_pool_digest",
        "device",
        "joint_minus_temporal",
        "prediction_values_persisted",
        "protocol_digest",
        "quantile_aggregation_semantic",
        "raw_duration_values_persisted",
        "schema_version",
        "selector_configuration_digest",
        "verifier_bundle_digests",
    }
    device = payload.get("device")
    device_matches = (
        device == "cpu"
        if expected_device_kind == "cpu"
        else isinstance(device, str) and device.startswith("cuda")
    )
    if (
        set(payload) != expected_fields
        or not device_matches
        or payload["schema_version"] != "1.0"
        or payload["blind_input_digest"] != cast(Any, blind_input).content_digest
        or payload["candidate_pool_digest"]
        != cast(Any, blind_input).full_candidate_pool_digest
        or payload["protocol_digest"] != protocol.content_digest
        or payload["selector_configuration_digest"] != selector_configuration_digest
        or payload["prediction_values_persisted"] is not False
        or payload["raw_duration_values_persisted"] is not False
        or payload["quantile_aggregation_semantic"]
        != "linear_quantile_of_all_end_to_end_group_repetitions_v1"
        or dict(cast(Mapping[str, str], payload["verifier_bundle_digests"]))
        != dict(expected_bundle_digests)
    ):
        _fail("inference latency report", "identity or top-level contract differs")
    architectures = payload["architectures"]
    if not isinstance(architectures, Mapping) or tuple(sorted(architectures)) != (
        EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS
    ):
        _fail("inference latency report", "architecture inventory differs")
    expected_architecture_fields = {
        "bundle_loading_per_seed_model_seconds",
        "bundle_loading_total_seconds",
        "candidate_count_per_group",
        "candidates_per_second",
        "device",
        "group_count",
        "measurement_iterations_per_group",
        "measured_candidate_executions",
        "measured_group_repetition_count",
        "measurement_scope",
        "model_loading_total_seconds",
        "peak_allocated_device_memory_bytes",
        "per_candidate_seconds_p50",
        "per_candidate_seconds_p95",
        "per_candidate_seconds_p99",
        "per_group_seconds_p50",
        "per_group_seconds_p95",
        "per_group_seconds_p99",
        "raw_duration_values_persisted",
        "warmup_iterations_per_group",
    }
    group_count = len(cast(Any, blind_input).groups)
    positive_fields = (
        "candidates_per_second",
        "per_candidate_seconds_p50",
        "per_candidate_seconds_p95",
        "per_candidate_seconds_p99",
        "per_group_seconds_p50",
        "per_group_seconds_p95",
        "per_group_seconds_p99",
    )
    for architecture in EXPECTED_STAGE_A_VERIFIER_BUNDLE_KEYS:
        raw = architectures[architecture]
        if not isinstance(raw, Mapping) or set(raw) != expected_architecture_fields:
            _fail("inference latency report", "architecture fields differ")
        per_seed = raw["bundle_loading_per_seed_model_seconds"]
        if (
            raw["device"] != device
            or raw["candidate_count_per_group"] != 8
            or raw["group_count"] != group_count
            or raw["measurement_iterations_per_group"]
            != ENSEMBLE_PROFILE_MEASUREMENT_ITERATIONS
            or raw["warmup_iterations_per_group"] != ENSEMBLE_PROFILE_WARMUP_ITERATIONS
            or raw["measured_group_repetition_count"]
            != group_count * ENSEMBLE_PROFILE_MEASUREMENT_ITERATIONS
            or raw["measured_candidate_executions"]
            != group_count * ENSEMBLE_PROFILE_MEASUREMENT_ITERATIONS * 8
            or raw["measurement_scope"] != ENSEMBLE_END_TO_END_PROFILE_SEMANTIC
            or raw["raw_duration_values_persisted"] is not False
            or not isinstance(per_seed, Sequence)
            or isinstance(per_seed, (str, bytes))
            or len(per_seed) != 5
        ):
            _fail("inference latency report", "measurement contract differs")
        nonnegative = (
            raw["bundle_loading_total_seconds"],
            raw["model_loading_total_seconds"],
            raw["peak_allocated_device_memory_bytes"],
            *per_seed,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in nonnegative
        ) or any(
            not isinstance(raw[name], (int, float))
            or isinstance(raw[name], bool)
            or not math.isfinite(float(raw[name]))
            or float(raw[name]) <= 0.0
            for name in positive_fields
        ):
            _fail("inference latency report", "timing values are invalid")
    difference = payload["joint_minus_temporal"]
    expected_difference_fields = {
        "bundle_loading_total_seconds",
        "candidates_per_second",
        "model_loading_total_seconds",
        "peak_allocated_device_memory_bytes",
        "per_candidate_seconds_p50",
        "per_candidate_seconds_p95",
        "per_candidate_seconds_p99",
        "per_group_seconds_p50",
        "per_group_seconds_p95",
        "per_group_seconds_p99",
        "per_seed_model_loading_seconds",
    }
    if not isinstance(difference, Mapping) or set(difference) != (
        expected_difference_fields
    ):
        _fail("inference latency report", "joint/temporal difference fields differ")
    per_seed_difference = difference["per_seed_model_loading_seconds"]
    scalar_differences = tuple(
        value
        for name, value in difference.items()
        if name != "per_seed_model_loading_seconds"
    )
    if (
        not isinstance(per_seed_difference, Sequence)
        or isinstance(per_seed_difference, (str, bytes))
        or len(per_seed_difference) != 5
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in (*scalar_differences, *per_seed_difference)
        )
    ):
        _fail("inference latency report", "joint/temporal differences are invalid")
    return report


def _run_select_action_candidates(args: argparse.Namespace) -> int:
    from latentguard.selection.blind_input import load_blind_candidate_pool
    from latentguard.selection.blind_protocol import (
        finalize_blind_selection_from_blind_pool,
        run_blind_selection_stage_a_from_blind_pool,
        validate_selection_manifest_against_blind_pool,
    )
    from latentguard.selection.manifest import load_blind_selection_manifest
    from latentguard.training.reporting import save_strict_report

    protocol = _load_protocol(args.config, mode=args.mode)
    blind_input = load_blind_candidate_pool(args.blind_input_dir)
    manifest_path = args.output_dir / SELECTION_MANIFEST_NAME
    if args.resume:
        envelope = load_blind_selection_manifest(manifest_path)
        validate_selection_manifest_against_blind_pool(envelope.manifest, blind_input)
        _load_selector_configuration_report(
            args.output_dir / SELECTOR_CONFIGURATION_REPORT,
            expected_digest=envelope.manifest.selector_configuration_digest,
            protocol=protocol,
            blind_input=blind_input,
            expected_bundle_digests=dict(envelope.manifest.verifier_bundle_digests),
            selections=envelope.manifest.selections,
        )
        internal_latency_path = args.output_dir / (
            f"inference-latency-{str(args.device).replace(':', '-')}.json"
        )
        latency_report = _load_bound_inference_latency_report(
            internal_latency_path,
            expected_device_kind=str(args.device).split(":", maxsplit=1)[0],
            protocol=protocol,
            blind_input=blind_input,
            selector_configuration_digest=(
                envelope.manifest.selector_configuration_digest
            ),
            expected_bundle_digests=dict(envelope.manifest.verifier_bundle_digests),
        )
        if args.latency_output is not None:
            if _resolved(args.latency_output).is_relative_to(
                _resolved(args.output_dir)
            ):
                _fail(
                    "latency output",
                    "explicit output must stay outside the immutable Stage A root",
                )
            save_strict_report(cast(Any, latency_report), args.latency_output)
        selector_count = len(
            {item.selector_id for item in envelope.manifest.selections}
        )
        print(
            "select-action-candidates resume OK: "
            f"selectors={selector_count} "
            f"groups={len(blind_input.groups)} digest={envelope.semantic_digest}"
        )
        return 0
    _require_separate_paths(
        inputs=(
            args.blind_input_dir,
            args.prepared_bundles_dir,
            args.m3b_runtime_root,
        ),
        outputs=(args.output_dir,),
    )
    if args.output_dir.exists() or args.output_dir.is_symlink():
        _fail("Stage A output", "must be absent on the first invocation")
    preparation = _load_preparation_summary(args.prepared_bundles_dir, protocol)
    bundles = _load_selection_bundles(
        args.prepared_bundles_dir, args.m3b_runtime_root, device=args.device
    )
    bundle_digests = {
        name: cast(Any, value).identity.content_digest
        for name, value in bundles.items()
    }
    if dict(cast(Mapping[str, object], preparation["bundle_digests"])) != (
        bundle_digests
    ):
        _fail("preparation summary", "loaded bundle identities differ")
    baseline = _load_action_baseline(
        args.prepared_bundles_dir / ACTION_MAGNITUDE_REPORT,
        expected_report_digest=cast(str, preparation["action_magnitude_report_digest"]),
    )
    policies = _load_policies(
        args.prepared_bundles_dir / ABSTENTION_POLICY_REPORT,
        expected_report_digest=cast(str, preparation["policy_report_digest"]),
        expected_split_digest=cast(str, protocol.payload["accepted_split_digest"]),
        expected_bundle_digest=bundle_digests["temporal_state_action_verifier"],
    )
    if cast(list[object], preparation["policy_content_digests"]) != [
        cast(Any, item).content_digest for item in policies
    ]:
        _fail("preparation summary", "policy identities differ")
    inference_diagnostics: dict[str, list[object]] = {}
    decisions = _stage_a_decisions(
        blind_input=blind_input,
        bundles=bundles,
        baseline=baseline,
        policies=policies,
        protocol=protocol,
        device=args.device,
        diagnostics_out=inference_diagnostics,
    )
    selector_configuration = cast(
        Any,
        _strict_report_payload(
            "m3c_selector_configuration_v1",
            _selector_configuration_payload(
                protocol=protocol,
                blind_input=blind_input,
                bundles=bundles,
                baseline=baseline,
                policies=policies,
            ),
        ),
    )
    selector_digest = selector_configuration.content_digest
    latency_payload = _inference_latency_payload(
        inference_diagnostics, device=args.device
    )
    latency_payload.update(
        {
            "blind_input_digest": blind_input.content_digest,
            "candidate_pool_digest": blind_input.full_candidate_pool_digest,
            "protocol_digest": protocol.content_digest,
            "selector_configuration_digest": selector_digest,
            "verifier_bundle_digests": bundle_digests,
        }
    )
    new_latency_report = cast(
        Any,
        _strict_report_payload("m3c_inference_latency_v1", latency_payload),
    )
    timestamp = _utc_now()
    if args.dry_run:
        envelope = finalize_blind_selection_from_blind_pool(
            blind_input,
            cast(Sequence[Any], decisions),
            selector_configuration_digest=selector_digest,
            verifier_bundle_digests=bundle_digests,
            selection_timestamp_utc=timestamp,
        )
        if args.latency_output is not None:
            if _resolved(args.latency_output).is_relative_to(
                _resolved(args.output_dir)
            ):
                _fail(
                    "latency output",
                    "dry-run latency output must stay outside Stage A root",
                )
            save_strict_report(new_latency_report, args.latency_output)
    else:
        if args.latency_output is not None and _resolved(
            args.latency_output
        ).is_relative_to(_resolved(args.output_dir)):
            _fail(
                "latency output",
                "explicit output must stay outside the immutable Stage A root",
            )
        output_parent = args.output_dir.parent
        output_parent.mkdir(parents=True, exist_ok=True)
        if output_parent.is_symlink() or not output_parent.is_dir():
            _fail("Stage A output", "parent must be a non-symlink directory")
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{args.output_dir.name}.",
                suffix=".tmp",
                dir=output_parent,
            )
        )
        published = False
        try:
            _, envelope = run_blind_selection_stage_a_from_blind_pool(
                blind_input,
                cast(Sequence[Any], decisions),
                selector_configuration_digest=selector_digest,
                verifier_bundle_digests=bundle_digests,
                selection_timestamp_utc=timestamp,
                output_root=staging_root,
            )
            save_strict_report(
                selector_configuration,
                staging_root / SELECTOR_CONFIGURATION_REPORT,
            )
            latency_path = staging_root / (
                f"inference-latency-{str(args.device).replace(':', '-')}.json"
            )
            save_strict_report(new_latency_report, latency_path)
            _load_selector_configuration_report(
                staging_root / SELECTOR_CONFIGURATION_REPORT,
                expected_digest=selector_digest,
                protocol=protocol,
                blind_input=blind_input,
                expected_bundle_digests=bundle_digests,
                selections=envelope.manifest.selections,
            )
            _load_bound_inference_latency_report(
                latency_path,
                expected_device_kind=str(args.device).split(":", maxsplit=1)[0],
                protocol=protocol,
                blind_input=blind_input,
                selector_configuration_digest=selector_digest,
                expected_bundle_digests=bundle_digests,
            )
            staging_root.rename(args.output_dir)
            published = True
            if args.latency_output is not None:
                save_strict_report(new_latency_report, args.latency_output)
        except BaseException:
            if not published:
                shutil.rmtree(staging_root, ignore_errors=True)
            raise
    print(
        f"select-action-candidates {'dry-run ' if args.dry_run else ''}OK: "
        f"selectors={len({cast(Any, item).selector_id for item in decisions})} "
        f"groups={len(blind_input.groups)} decisions={len(decisions)} "
        f"digest={envelope.semantic_digest} "
        f"output_created={str(not args.dry_run).lower()}"
    )
    return 0


def _load_replay_components(
    args: argparse.Namespace,
) -> tuple[object, object, object, object]:
    from latentguard.evaluation.serialization import (
        compute_corruption_dataset_content_digest,
    )
    from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
        load_state_indexed_archive,
    )
    from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
        load_anchor_manifest,
    )
    from latentguard.integrations.maniskill_pickcube.state_indexed_replay import (
        build_state_indexed_maniskill_pickcube_adapter,
    )
    from latentguard.replay.evaluator import create_exact_state_paired_replay_evaluator
    from latentguard.replay.source import ReplaySourceBinding
    from latentguard.selection.blind_input import build_blind_candidate_pool
    from latentguard.selection.blind_protocol import (
        validate_selection_manifest_against_pool,
    )
    from latentguard.selection.configuration import load_candidate_pool_configuration
    from latentguard.selection.evaluation import build_manifest_gated_replay_evaluators
    from latentguard.selection.manifest import load_blind_selection_manifest
    from latentguard.selection.serialization import load_candidate_pool
    from latentguard.serialization import compute_episode_bundle_identifier

    protocol = _load_protocol(args.config, mode=args.mode)
    pool = load_candidate_pool(args.candidate_pool_dir)
    envelope = load_blind_selection_manifest(args.selection_manifest)
    validate_selection_manifest_against_pool(envelope.manifest, pool)
    _load_selector_configuration_report(
        Path(args.selection_manifest).parent / SELECTOR_CONFIGURATION_REPORT,
        expected_digest=envelope.manifest.selector_configuration_digest,
        protocol=protocol,
        blind_input=build_blind_candidate_pool(pool),
        expected_bundle_digests=dict(envelope.manifest.verifier_bundle_digests),
        selections=envelope.manifest.selections,
    )
    binding = ReplaySourceBinding.from_paths(args.source_dir, args.corruption_dir)
    sources, _, _, _ = _candidate_sources(
        source_dir=args.source_dir,
        runtime_archive_dir=args.runtime_archive_dir,
        anchor_manifest_dir=args.anchor_manifest_dir,
        protocol=protocol,
    )
    candidate_configuration = load_candidate_pool_configuration(args.candidate_config)
    archive = load_state_indexed_archive(args.runtime_archive_dir)
    manifest = load_anchor_manifest(args.anchor_manifest_dir)
    if manifest.source_archive_content_digest != archive.content_digest:
        _fail("replay", "anchor manifest references a different archive")
    if tuple(item.source_episode_id for item in manifest.records) != tuple(
        item.episode_id for item in binding.episodes
    ):
        _fail("replay", "anchor and source episode ordering differs")
    if tuple(item.source_candidate_id for item in manifest.records) != tuple(
        item.candidates[0].candidate_id for item in binding.episodes
    ):
        _fail("replay", "anchor and source candidate ordering differs")
    scope = _load_pickcube_scope(args)
    expected_corruptions = _corruption_dataset_for_pool(
        pool=pool,
        sources=sources,
        configuration=candidate_configuration,
        action_layout=cast(Any, scope.layout).m1_action_layout,
        source_dataset_id=compute_episode_bundle_identifier(args.source_dir),
    )
    if compute_corruption_dataset_content_digest(
        cast(Any, expected_corruptions)
    ) != compute_corruption_dataset_content_digest(binding.corruption_dataset):
        _fail("replay", "replay dataset content differs from frozen pool and source")
    adapter = build_state_indexed_maniskill_pickcube_adapter(
        source_binding=binding,
        archive_dir=args.runtime_archive_dir,
        compatibility_binding=cast(Any, scope.binding),
        action_layout_digest=cast(Any, scope.layout).action_layout_digest,
        coordinate_frame=cast(Any, scope.layout).coordinate_frame,
        candidate_horizon=16,
    )
    inner = create_exact_state_paired_replay_evaluator(adapter, adapter.replay_bundle)
    phases = build_manifest_gated_replay_evaluators(
        inner,
        candidate_pool=pool,
        selection_manifest=envelope,
    )
    return pool, envelope, binding, phases


def _replay_launch_command(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        "latentguard",
        args.command,
        "--mode",
        args.mode,
        "--candidate-pool-dir",
        str(args.candidate_pool_dir),
        "--selection-manifest",
        str(args.selection_manifest),
        "--source-dir",
        str(args.source_dir),
        "--corruption-dir",
        str(args.corruption_dir),
        "--runtime-archive-dir",
        str(args.runtime_archive_dir),
        "--anchor-manifest-dir",
        str(args.anchor_manifest_dir),
        "--compatibility-report",
        str(args.compatibility_report),
        "--expected-contract",
        str(args.expected_contract),
        "--action-layout",
        str(args.action_layout),
        "--output-dir",
        str(args.output_dir),
        "--seed",
        str(args.seed),
    ]
    for enabled, flag in (
        (args.resume, "--resume"),
        (args.fail_fast, "--fail-fast"),
        (args.retry_execution_errors, "--retry-execution-errors"),
    ):
        if enabled:
            command.append(flag)
    return tuple(command)


def _run_replay_phase(
    args: argparse.Namespace,
    *,
    selected: bool,
    components: tuple[object, object, object, object] | None = None,
) -> tuple[object, object, object]:
    from latentguard.evaluation.runner import plan_evaluation, run_evaluation
    from latentguard.evaluation.serialization import (
        RunState,
        compute_evaluation_dataset_content_digest,
        load_evaluation_dataset,
    )
    from latentguard.selection.evaluation import verify_zero_work_resume
    from latentguard.training.reporting import save_strict_report

    protocol = _load_protocol(args.config, mode=args.mode)
    if args.retry_execution_errors and not args.resume:
        _fail("replay", "--retry-execution-errors requires --resume")
    if components is None:
        components = _load_replay_components(args)
    pool, envelope, binding, phases = components
    _validate_source_mode(pool, protocol)
    evaluator = cast(Any, phases).selected if selected else cast(Any, phases).remainder
    plan = plan_evaluation(
        cast(Any, binding).corruption_dataset,
        evaluator,
        source_corruption_dataset_digest=cast(Any, binding).corruption_dataset_digest,
        base_seed=args.seed,
    )
    if args.dry_run:
        print(
            f"{args.command} dry-run OK: proposals={len(plan.selected_proposal_ids)} "
            f"physical={len(evaluator.executable_proposal_ids)} sessions=0 "
            "output_created=false"
        )
        return pool, envelope, cast(Any, None)
    if not args.resume and (args.output_dir.exists() or args.output_dir.is_symlink()):
        _fail("replay output", "must be absent unless --resume is used")
    before = None
    if args.resume and args.output_dir.is_dir():
        before = load_evaluation_dataset(
            args.output_dir,
            corruption_dataset=cast(Any, binding).corruption_dataset,
            expected_corruption_digest=cast(Any, binding).corruption_dataset_digest,
        )
    result = run_evaluation(
        cast(Any, binding).corruption_dataset,
        evaluator,
        source_corruption_dataset_digest=cast(Any, binding).corruption_dataset_digest,
        output_dir=args.output_dir,
        base_seed=args.seed,
        resume=before is not None,
        fail_fast=args.fail_fast,
        retry_execution_errors=args.retry_execution_errors,
        launch_command=_replay_launch_command(args),
    )
    cast(Any, binding).assert_unchanged()
    if before is not None and before.run_state is RunState.COMPLETE:
        resume_proof = verify_zero_work_resume(before, result)
        resume_summary = cast(
            Any,
            _strict_report_payload(
                "m3c_replay_resume_v1",
                {
                    "candidate_pool_digest": cast(Any, pool).content_digest,
                    "evaluated_attempts": resume_proof.evaluated_attempts,
                    "evaluation_dataset_archive_digest": (
                        compute_evaluation_dataset_content_digest(result.dataset)
                    ),
                    "evidence_count": resume_proof.evidence_count,
                    "ledger_count": resume_proof.ledger_count,
                    "phase": "selected" if selected else "remainder",
                    "protocol_digest": protocol.content_digest,
                    "recovered_attempts": resume_proof.recovered_attempts,
                    "retried_attempts": resume_proof.retried_attempts,
                    "run_id": resume_proof.run_id,
                    "schema_version": "1.0",
                    "selection_manifest_digest": cast(Any, envelope).semantic_digest,
                    "selection_manifest_envelope_digest": cast(
                        Any, envelope
                    ).envelope_digest,
                    "semantic": "complete_phase_zero_duplicate_work_resume_v1",
                    "zero_duplicate_work": resume_proof.zero_duplicate_work,
                },
            ),
        )
        save_strict_report(
            resume_summary,
            _replay_resume_report_path(args.output_dir),
        )
    if result.stopped_early or result.dataset.summary.execution_error:
        _fail("replay", "phase stopped early or contains execution errors")
    print(
        f"{args.command} OK: proposals={len(result.dataset.selected_proposal_ids)} "
        f"physical={len(evaluator.executable_proposal_ids)} "
        f"evaluated_this_invocation={result.evaluated_attempts} "
        f"run_id={result.dataset.run_id}"
    )
    return pool, envelope, result.dataset


def _run_replay_selected_candidates(args: argparse.Namespace) -> int:
    _run_replay_phase(args, selected=True)
    return 0


def _load_replay_resume_summary(
    output_root: Path,
    *,
    phase: str,
    dataset: object,
    candidate_pool_digest: str,
    selection_manifest_digest: str,
    selection_manifest_envelope_digest: str,
    protocol_digest: str,
) -> object:
    """Reload and bind one zero-work phase-resume proof to exact persisted data."""

    from latentguard.evaluation.serialization import (
        compute_evaluation_dataset_content_digest,
    )
    from latentguard.training.reporting import load_strict_report

    report = load_strict_report(
        _replay_resume_report_path(output_root),
        expected_report_type="m3c_replay_resume_v1",
    )
    payload = report.payload
    expected_fields = {
        "candidate_pool_digest",
        "evaluated_attempts",
        "evaluation_dataset_archive_digest",
        "evidence_count",
        "ledger_count",
        "phase",
        "protocol_digest",
        "recovered_attempts",
        "retried_attempts",
        "run_id",
        "schema_version",
        "selection_manifest_digest",
        "selection_manifest_envelope_digest",
        "semantic",
        "zero_duplicate_work",
    }
    value = cast(Any, dataset)
    if (
        set(payload) != expected_fields
        or payload["phase"] != phase
        or payload["candidate_pool_digest"] != candidate_pool_digest
        or payload["selection_manifest_digest"] != selection_manifest_digest
        or payload["selection_manifest_envelope_digest"]
        != selection_manifest_envelope_digest
        or payload["protocol_digest"] != protocol_digest
        or payload["evaluation_dataset_archive_digest"]
        != compute_evaluation_dataset_content_digest(value)
        or payload["run_id"] != value.run_id
        or payload["evidence_count"] != len(value.evidence)
        or payload["ledger_count"] != len(value.ledger)
        or payload["evaluated_attempts"] != 0
        or payload["recovered_attempts"] != 0
        or payload["retried_attempts"] != 0
        or payload["zero_duplicate_work"] is not True
        or payload["semantic"] != "complete_phase_zero_duplicate_work_resume_v1"
        or payload["schema_version"] != "1.0"
    ):
        _fail("replay resume summary", f"{phase} proof differs from persisted data")
    return report


def _load_validated_selected_stage(
    args: argparse.Namespace,
    components: tuple[object, object, object, object],
) -> object:
    from latentguard.evaluation.models import EvaluationStatus
    from latentguard.evaluation.serialization import RunState, load_evaluation_dataset

    pool, _, binding, phases = components
    evaluator = cast(Any, phases).selected
    dataset = load_evaluation_dataset(
        args.selected_output_dir,
        corruption_dataset=cast(Any, binding).corruption_dataset,
        expected_corruption_digest=cast(Any, binding).corruption_dataset_digest,
    )
    if (
        dataset.run_state is not RunState.COMPLETE
        or dataset.evaluator_id != evaluator.evaluator_id
        or dataset.evaluator_version != evaluator.evaluator_version
        or dataset.evaluator_configuration_digest != evaluator.configuration_digest
        or dict(dataset.resolved_evaluator_configuration)
        != dict(evaluator.resolved_configuration())
        or dataset.selected_proposal_ids != cast(Any, pool).proposal_ids
        or dataset.base_seed != args.seed
        or dataset.summary.execution_error != 0
        or dataset.summary.invalid != 0
        or dataset.summary.indeterminate != 0
    ):
        _fail(
            "Stage B precondition",
            "selected replay is incomplete or differs from pool/manifest/configuration",
        )
    by_id = {item.proposal_id: item for item in dataset.evidence}
    if set(by_id) != set(cast(Any, pool).proposal_ids):
        _fail("Stage B precondition", "selected replay evidence inventory differs")
    executable = set(evaluator.executable_proposal_ids)
    for proposal_id in cast(Any, pool).proposal_ids:
        evidence = by_id[proposal_id]
        if proposal_id in executable:
            if (
                evidence.status is not EvaluationStatus.CONCLUSIVE
                or not evidence.simulator_replay_verified
                or evidence.metrics.get(
                    "replay_baseline_restoration_compared_component_count"
                )
                != 70
                or evidence.metrics.get(
                    "replay_corrupted_restoration_compared_component_count"
                )
                != 70
            ):
                _fail(
                    "Stage B precondition",
                    "selected candidate lacks complete strong replay evidence",
                )
        elif (
            evidence.status is not EvaluationStatus.SKIPPED
            or evidence.termination_reason != "proposal_not_executable_in_phase"
        ):
            _fail(
                "Stage B precondition",
                "non-selected candidate was not an expected phase skip",
            )
    return dataset


def _run_replay_complete_candidate_pools(args: argparse.Namespace) -> int:
    from latentguard.selection.evaluation import join_complementary_replay_phases

    components = _load_replay_components(args)
    selected = _load_validated_selected_stage(args, components)
    pool, envelope, remainder = _run_replay_phase(
        args,
        selected=False,
        components=components,
    )
    if args.dry_run:
        return 0
    joined = join_complementary_replay_phases(
        candidate_pool=cast(Any, pool),
        selection_manifest=cast(Any, envelope),
        selected_dataset=cast(Any, selected),
        remainder_dataset=cast(Any, remainder),
    )
    print(
        "replay-complete-candidate-pools join OK: "
        f"outcomes={len(joined.proposal_ids)} digest={joined.content_digest}"
    )
    return 0


def _candidate_outcomes(pool: object, joined: object) -> tuple[object, ...]:
    from latentguard.selection.metrics import CandidateOutcomeV1

    groups = {
        candidate.proposal_id: (group, candidate)
        for group in cast(Any, pool).groups
        for candidate in group.candidates
    }
    outcomes: list[CandidateOutcomeV1] = []
    for proposal_id in cast(Any, joined).proposal_ids:
        evidence = cast(Any, joined).evidence_by_proposal[proposal_id]
        group, candidate = groups[proposal_id]
        count = evidence.metrics.get(
            "replay_corrupted_restoration_compared_component_count"
        )
        outcomes.append(
            CandidateOutcomeV1(
                proposal_id=proposal_id,
                group_id=group.group_id,
                source_trajectory_id=group.source_trajectory_id,
                distribution=candidate.distribution.value,
                evidence_id=evidence.evidence_id,
                status=evidence.status.value,
                success=evidence.success,
                unsafe=evidence.unsafe,
                simulator_replay_verified=evidence.simulator_replay_verified,
                label_strength=evidence.label_strength.value,
                state_component_count=cast(int, count),
                execution_error=False,
            )
        )
    return tuple(outcomes)


def _validated_phase_completion_timestamp(
    selection_timestamp: str, selected: object, remainder: object
) -> str:
    def parse(value: object, context: str) -> datetime:
        if not isinstance(value, str) or not value.endswith("Z"):
            _fail(context, "expected a persisted RFC3339 UTC timestamp")
        try:
            result = datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError as exc:
            raise M3CCommandError(f"{context}: invalid timestamp") from exc
        if result.tzinfo != UTC:
            _fail(context, "expected UTC")
        return result

    selection_time = parse(selection_timestamp, "selection timestamp")
    selected_start = parse(
        cast(Any, selected).run_manifest.started_at,
        "selected run start",
    )
    selected_finish_text = cast(Any, selected).run_manifest.finished_at
    remainder_start = parse(
        cast(Any, remainder).run_manifest.started_at,
        "remainder run start",
    )
    remainder_finish_text = cast(Any, remainder).run_manifest.finished_at
    selected_finish = parse(selected_finish_text, "selected run finish")
    remainder_finish = parse(remainder_finish_text, "remainder run finish")
    if not (
        selection_time
        < selected_start
        <= selected_finish
        < remainder_start
        <= remainder_finish
    ):
        _fail(
            "phase chronology",
            "requires selection < selected run < remainder run using persisted times",
        )
    assert isinstance(remainder_finish_text, str)
    return remainder_finish_text


def _compatibility_identity(args: argparse.Namespace) -> str:
    from latentguard.integrations.maniskill_pickcube.compatibility import (
        load_compatibility_report,
        validate_compatibility_report,
    )
    from latentguard.integrations.maniskill_pickcube.configuration import (
        load_expected_contract,
    )

    expected = load_expected_contract(args.expected_contract)
    report = load_compatibility_report(args.compatibility_report)
    binding = validate_compatibility_report(report, expected, require_trusted=True)
    return binding.report.compatibility_identity


def _run_evaluate_candidate_selection(args: argparse.Namespace) -> int:
    from latentguard.corruptions.serialization import load_corruption_dataset
    from latentguard.evaluation.serialization import (
        compute_corruption_dataset_digest,
        load_evaluation_dataset,
    )
    from latentguard.selection.blind_input import build_blind_candidate_pool
    from latentguard.selection.blind_protocol import (
        bind_complete_pool_result,
        compute_full_pool_outcome_digest,
        create_oracle_decisions,
        load_bound_selection_result,
        save_bound_selection_result,
    )
    from latentguard.selection.evaluation import join_complementary_replay_phases
    from latentguard.selection.manifest import load_blind_selection_manifest
    from latentguard.selection.reporting import (
        build_predeclared_interpretation,
        evaluate_selector_suite,
        render_human_readable_review,
    )
    from latentguard.selection.serialization import load_candidate_pool
    from latentguard.training.reporting import (
        load_strict_report,
        save_strict_report,
    )

    protocol = _load_protocol(args.config, mode=args.mode)
    pool = load_candidate_pool(args.candidate_pool_dir)
    _validate_source_mode(pool, protocol)
    envelope = load_blind_selection_manifest(args.selection_manifest)
    blind_input = build_blind_candidate_pool(pool)
    _load_selector_configuration_report(
        Path(args.selection_manifest).parent / SELECTOR_CONFIGURATION_REPORT,
        expected_digest=envelope.manifest.selector_configuration_digest,
        protocol=protocol,
        blind_input=blind_input,
        expected_bundle_digests=dict(envelope.manifest.verifier_bundle_digests),
        selections=envelope.manifest.selections,
    )
    if _resolved(args.cpu_latency_report) == _resolved(args.gpu_latency_report):
        _fail("inference latency reports", "CPU and GPU reports must be distinct")
    cpu_latency = _load_bound_inference_latency_report(
        args.cpu_latency_report,
        expected_device_kind="cpu",
        protocol=protocol,
        blind_input=blind_input,
        selector_configuration_digest=(envelope.manifest.selector_configuration_digest),
        expected_bundle_digests=dict(envelope.manifest.verifier_bundle_digests),
    )
    gpu_latency = _load_bound_inference_latency_report(
        args.gpu_latency_report,
        expected_device_kind="cuda",
        protocol=protocol,
        blind_input=blind_input,
        selector_configuration_digest=(envelope.manifest.selector_configuration_digest),
        expected_bundle_digests=dict(envelope.manifest.verifier_bundle_digests),
    )
    corruption_dataset = load_corruption_dataset(args.corruption_dir)
    corruption_digest = compute_corruption_dataset_digest(args.corruption_dir)
    selected = load_evaluation_dataset(
        args.selected_output_dir,
        corruption_dataset=corruption_dataset,
        expected_corruption_digest=corruption_digest,
    )
    remainder = load_evaluation_dataset(
        args.remainder_output_dir,
        corruption_dataset=corruption_dataset,
        expected_corruption_digest=corruption_digest,
    )
    joined = join_complementary_replay_phases(
        candidate_pool=pool,
        selection_manifest=envelope,
        selected_dataset=selected,
        remainder_dataset=remainder,
    )
    outcomes = _candidate_outcomes(pool, joined)
    full_outcome_digest = compute_full_pool_outcome_digest(
        cast(Sequence[Any], outcomes)
    )
    output_root = Path(args.output_dir)
    bound_path = output_root / BOUND_RESULT_NAME
    report_path = output_root / SELECTION_RESULT_REPORT
    decisions: dict[str, list[Any]] = {}
    for decision in envelope.manifest.selections:
        decisions.setdefault(decision.selector_id, []).append(decision)
    oracle = create_oracle_decisions(
        pool,
        cast(Sequence[Any], outcomes),
        full_pool_outcome_digest=full_outcome_digest,
    )
    decisions[oracle[0].selector_id] = list(oracle)
    learned_ids = tuple(
        sorted(
            selector_id
            for selector_id in decisions
            if selector_id
            in {
                "action_only_ensemble_v1",
                "state_action_mlp_ensemble_v1",
                "temporal_ensemble_v1",
            }
            or selector_id.startswith("temporal_ensemble_abstention_")
        )
    )
    suite = evaluate_selector_suite(
        decisions,
        cast(Sequence[Any], outcomes),
        random_selector_id="deterministic_random_v1",
        learned_selector_ids=learned_ids,
        temporal_selector_id=cast(str, protocol.payload["primary_selector"]),
        joint_selector_id=cast(str, protocol.payload["efficiency_challenger"]),
        bootstrap_resamples=cast(int, protocol.payload["bootstrap_replicates"]),
        bootstrap_seed=cast(int, protocol.payload["bootstrap_seed"]),
        confidence_level=cast(float, protocol.payload["confidence_level"]),
    )
    strong_replay_evidence_verified = all(
        cast(Any, item).status == "conclusive"
        and cast(Any, item).simulator_replay_verified is True
        and cast(Any, item).label_strength == "strong"
        and cast(Any, item).state_component_count == 70
        and cast(Any, item).execution_error is False
        for item in outcomes
    )
    interpretation = build_predeclared_interpretation(
        suite,
        temporal_selector_id=cast(str, protocol.payload["primary_selector"]),
        joint_selector_id=cast(str, protocol.payload["efficiency_challenger"]),
        random_selector_id="deterministic_random_v1",
        coverage_70_selector_id=(
            "temporal_ensemble_abstention_target_validation_coverage_70_v1"
        ),
        strong_replay_evidence_verified=strong_replay_evidence_verified,
    )
    outcomes_generated_at = _validated_phase_completion_timestamp(
        envelope.manifest.selection_timestamp_utc,
        selected,
        remainder,
    )
    result = bind_complete_pool_result(
        envelope,
        pool,
        cast(Sequence[Any], outcomes),
        replay_evidence_digest=joined.content_digest,
        simulator_compatibility_identity=_compatibility_identity(args),
        git_sha=_git_sha(),
        outcomes_generated_at_utc=outcomes_generated_at,
        expected_full_pool_outcome_digest=full_outcome_digest,
    )
    inference_performance = {
        "cpu": {
            "report_content_digest": cast(Any, cpu_latency).content_digest,
            "summary": dict(cast(Any, cpu_latency).payload),
        },
        "gpu": {
            "report_content_digest": cast(Any, gpu_latency).content_digest,
            "summary": dict(cast(Any, gpu_latency).payload),
        },
        "joint_minus_temporal_semantic": (
            "positive_latency_or_memory_means_joint_is_larger_or_slower_v1"
        ),
        "outcome_based_latency_winner_selection_permitted": False,
        "schema_version": "1.0",
    }
    human_review = render_human_readable_review(
        suite,
        interpretation,
        mode=args.mode,
        git_sha=result.git_sha,
        selection_manifest_digest=envelope.semantic_digest,
        candidate_pool_digest=pool.content_digest,
        replay_evidence_digest=joined.content_digest,
        full_pool_outcome_digest=full_outcome_digest,
        cpu_latency_report_digest=cast(Any, cpu_latency).content_digest,
        gpu_latency_report_digest=cast(Any, gpu_latency).content_digest,
    )
    human_review_digest = (
        "sha256:" + hashlib.sha256(human_review.encode("utf-8")).hexdigest()
    )
    report_payload = {
        "audit_chronology": {
            "outcomes_generated_at_utc": outcomes_generated_at,
            "remainder_run_finished_at": remainder.run_manifest.finished_at,
            "remainder_run_started_at": remainder.run_manifest.started_at,
            "selected_run_finished_at": selected.run_manifest.finished_at,
            "selected_run_started_at": selected.run_manifest.started_at,
            "selection_timestamp_utc": (envelope.manifest.selection_timestamp_utc),
            "semantic": "persisted_selection_then_selected_then_remainder_v1",
        },
        "blind_input_digest": blind_input.content_digest,
        "bound_result_digest": result.content_digest,
        "candidate_pool_digest": pool.content_digest,
        "full_pool_outcome_digest": full_outcome_digest,
        "git_sha": result.git_sha,
        "human_review_content_digest": human_review_digest,
        "inference_performance": inference_performance,
        "mode": args.mode,
        "oracle_analysis_only": True,
        "outcomes_available_during_selection": False,
        "predeclared_interpretation": interpretation,
        "protocol_digest": protocol.content_digest,
        "replay_archive_audit_digest": joined.archive_audit_digest,
        "replay_evidence_digest": joined.content_digest,
        "replay_phase_archive_dataset_digests": {
            "remainder": joined.remainder_phase_dataset_digest,
            "selected": joined.selected_phase_dataset_digest,
        },
        "schema_version": "1.0",
        "selection_manifest_envelope_digest": envelope.envelope_digest,
        "selection_manifest_digest": envelope.semantic_digest,
        "selector_evaluation": suite.to_dict(),
        "simulator_compatibility_identity": result.simulator_compatibility_identity,
        "source_set_digest": pool.source_set_digest,
        "verifier_bundle_digests": dict(envelope.manifest.verifier_bundle_digests),
    }
    expected_report = cast(
        Any,
        _strict_report_payload(
            "m3c_candidate_selection_result_v1",
            report_payload,
        ),
    )
    if args.resume:
        persisted_result = load_bound_selection_result(
            bound_path,
            selection_envelope=envelope,
            candidate_pool=pool,
        )
        if persisted_result.content_digest != result.content_digest:
            _fail("resume", "persisted bound result differs from current replay")
        load_strict_report(
            report_path,
            expected_report_type="m3c_candidate_selection_result_v1",
            expected_content_digest=expected_report.content_digest,
        )
        _validate_immutable_text(output_root / HUMAN_REVIEW_NAME, human_review)
        selected_resume = _load_replay_resume_summary(
            args.selected_output_dir,
            phase="selected",
            dataset=selected,
            candidate_pool_digest=pool.content_digest,
            selection_manifest_digest=envelope.semantic_digest,
            selection_manifest_envelope_digest=envelope.envelope_digest,
            protocol_digest=protocol.content_digest,
        )
        remainder_resume = _load_replay_resume_summary(
            args.remainder_output_dir,
            phase="remainder",
            dataset=remainder,
            candidate_pool_digest=pool.content_digest,
            selection_manifest_digest=envelope.semantic_digest,
            selection_manifest_envelope_digest=envelope.envelope_digest,
            protocol_digest=protocol.content_digest,
        )
        final_resume = cast(
            Any,
            _strict_report_payload(
                "m3c_complete_resume_v1",
                {
                    "bound_result_digest": result.content_digest,
                    "candidate_pool_reload_verified": True,
                    "candidate_selection_report_digest": (
                        expected_report.content_digest
                    ),
                    "evaluation_reload_verified": True,
                    "human_review_content_digest": human_review_digest,
                    "remainder_phase_resume_report_digest": cast(
                        Any, remainder_resume
                    ).content_digest,
                    "replay_archive_audit_digest": joined.archive_audit_digest,
                    "replay_evidence_digest": joined.content_digest,
                    "schema_version": "1.0",
                    "selected_phase_resume_report_digest": cast(
                        Any, selected_resume
                    ).content_digest,
                    "selection_manifest_reload_verified": True,
                    "semantic": "m3c_all_phases_strict_reload_zero_duplicate_work_v1",
                    "zero_duplicate_replay_work": True,
                },
            ),
        )
        save_strict_report(final_resume, output_root / FINAL_RESUME_REPORT)
        print(
            "evaluate-candidate-selection resume OK: "
            f"groups={len(pool.groups)} outcomes={len(outcomes)} "
            f"digest={persisted_result.content_digest}"
        )
        return 0
    if args.dry_run:
        print(
            "evaluate-candidate-selection dry-run OK: "
            f"groups={suite.group_count} outcomes={suite.outcome_count} "
            "output_created=false"
        )
        return 0
    _require_separate_paths(
        inputs=(
            args.candidate_pool_dir,
            args.corruption_dir,
            args.cpu_latency_report,
            args.gpu_latency_report,
            args.selected_output_dir,
            args.selection_manifest,
            args.remainder_output_dir,
        ),
        outputs=(args.output_dir,),
    )
    if output_root.exists() or output_root.is_symlink():
        _fail("result output", "must be absent unless --resume is used")
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        save_bound_selection_result(bound_path, result)
        save_strict_report(expected_report, report_path)
        _save_immutable_text(output_root / HUMAN_REVIEW_NAME, human_review)
        reloaded_result = load_bound_selection_result(
            bound_path,
            selection_envelope=envelope,
            candidate_pool=pool,
        )
        if reloaded_result.content_digest != result.content_digest:
            _fail("result", "bound result changed after publication")
        load_strict_report(
            report_path,
            expected_report_type="m3c_candidate_selection_result_v1",
            expected_content_digest=expected_report.content_digest,
        )
        _validate_immutable_text(output_root / HUMAN_REVIEW_NAME, human_review)
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise
    print(
        "evaluate-candidate-selection OK: "
        f"groups={suite.group_count} outcomes={suite.outcome_count} "
        f"selectors={len(suite.selector_reports)} digest={result.content_digest}"
    )
    return 0


__all__ = ["M3C_COMMANDS", "add_m3c_subparsers", "run_m3c_command"]

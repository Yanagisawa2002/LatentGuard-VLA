"""CPU-safe command surface for M4B visual-action training and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NoReturn, cast

from latentguard.evaluation.security import sanitize_operational_text
from latentguard.replay.identity import canonical_json_bytes
from latentguard.training.dataset import AcceptedActionVerifierDatasetV1

M4B_COMMANDS = frozenset(
    {
        "prepare-visual-backbone",
        "extract-visual-features",
        "prepare-visual-teacher-targets",
        "train-visual-action-verifier",
        "benchmark-visual-action-verifier",
        "evaluate-visual-action-verifier",
        "select-visual-action-candidates",
        "evaluate-external-visual-selection",
    }
)
_CONFIG_ROOT = Path("configs/training/m4b")


class M4BCommandError(ValueError):
    """Raised when an M4B CLI invocation violates a frozen workflow contract."""


def _fail(context: str, reason: str) -> NoReturn:
    raise M4BCommandError(f"{context}: {reason}")


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return result


def _add_development_data(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--visual-dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--anchor-manifest-dir", type=Path, required=True)


def add_m4b_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register exactly the eight reviewed M4B commands."""
    backbone = subparsers.add_parser(
        "prepare-visual-backbone",
        help="prepare and content-bind official ResNet-18 ImageNet-1K V1",
    )
    backbone.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_ROOT / "backbone-resnet18-imagenet1k-v1.json",
    )
    backbone.add_argument("--output", type=Path, required=True)
    backbone.add_argument("--device", default="cpu")
    backbone.add_argument("--dry-run", action="store_true")

    features = subparsers.add_parser(
        "extract-visual-features",
        help="extract one deduplicated content-bound frozen feature cache",
    )
    features.add_argument("--dataset-dir", type=Path, required=True)
    features.add_argument(
        "--backbone-config",
        type=Path,
        default=_CONFIG_ROOT / "backbone-resnet18-imagenet1k-v1.json",
    )
    features.add_argument("--backbone-manifest", type=Path, required=True)
    features.add_argument("--model-freeze-manifest", type=Path)
    features.add_argument("--output-dir", type=Path, required=True)
    features.add_argument("--device", default="cuda")
    features.add_argument("--batch-size", type=_positive_int, default=64)
    features.add_argument("--limit-images", type=_positive_int)
    features.add_argument("--resume", action="store_true")
    features.add_argument(
        "--evaluation-only",
        action="store_true",
        help="required before opening the frozen M3C external dataset",
    )
    features.add_argument("--dry-run", action="store_true")

    teacher = subparsers.add_parser(
        "prepare-visual-teacher-targets",
        help="extract the accepted five-seed temporal teacher targets once",
    )
    _add_development_data(teacher)
    teacher.add_argument("--bundle-report", type=Path, required=True)
    teacher.add_argument("--preprocessing", type=Path, required=True)
    teacher.add_argument(
        "--teacher-checkpoint", type=Path, action="append", required=True
    )
    teacher.add_argument(
        "--teacher-calibration", type=Path, action="append", required=True
    )
    teacher.add_argument(
        "--teacher-thresholds", type=Path, action="append", required=True
    )
    teacher.add_argument("--output-dir", type=Path, required=True)
    teacher.add_argument("--device", default="cuda")
    teacher.add_argument("--batch-size", type=_positive_int, default=256)
    teacher.add_argument("--limit-samples", type=_positive_int)
    teacher.add_argument("--resume", action="store_true")
    teacher.add_argument("--dry-run", action="store_true")

    train = subparsers.add_parser(
        "train-visual-action-verifier",
        help="train one fixed M4B model and seed with exact resume",
    )
    _add_development_data(train)
    train.add_argument("--model-config", type=Path, required=True)
    train.add_argument(
        "--training-config",
        type=Path,
        default=_CONFIG_ROOT / "train-default.json",
    )
    train.add_argument("--backbone-manifest", type=Path, required=True)
    train.add_argument("--feature-cache", type=Path)
    train.add_argument("--teacher-cache", type=Path)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--seed", type=_nonnegative_int, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--max-steps", type=_positive_int)
    train.add_argument("--max-epochs", type=_positive_int)
    train.add_argument("--limit-samples", type=_positive_int)
    train.add_argument("--mixed-precision", action="store_true")
    train.add_argument("--dry-run", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark-visual-action-verifier",
        help="enforce staged seed screening and validation-only promotion",
    )
    benchmark.add_argument(
        "--stage",
        choices=("single-seed-screen", "promoted-three-seed", "five-seed"),
        required=True,
    )
    benchmark.add_argument(
        "--seed-policy", type=Path, default=_CONFIG_ROOT / "seed-policy.json"
    )
    benchmark.add_argument("--run-report", type=Path, action="append", required=True)
    benchmark.add_argument("--promotion-record", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--resume", action="store_true")
    benchmark.add_argument("--explicit-five-seed-authorization", action="store_true")
    benchmark.add_argument("--dry-run", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate-visual-action-verifier",
        help="freeze calibration and run one internal per-domain test evaluation",
    )
    _add_development_data(evaluate)
    evaluate.add_argument("--model-config", type=Path, required=True)
    evaluate.add_argument(
        "--training-config",
        type=Path,
        default=_CONFIG_ROOT / "train-default.json",
    )
    evaluate.add_argument("--checkpoint", type=Path, action="append", required=True)
    evaluate.add_argument("--feature-cache", type=Path)
    evaluate.add_argument("--backbone-manifest", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--dry-run", action="store_true")

    select = subparsers.add_parser(
        "select-visual-action-candidates",
        help="create immutable M3C outcome-free visual selection manifests",
    )
    select.add_argument("--external-visual-dataset-dir", type=Path, required=True)
    select.add_argument("--candidate-pool-dir", type=Path, required=True)
    select.add_argument("--feature-cache", type=Path, required=True)
    select.add_argument(
        "--external-config",
        type=Path,
        default=_CONFIG_ROOT / "external-evaluation.json",
    )
    select.add_argument("--model-config", type=Path, required=True)
    select.add_argument("--checkpoint", type=Path, action="append", required=True)
    select.add_argument("--model-freeze-manifest", type=Path, required=True)
    select.add_argument(
        "--domain",
        choices=("canonical", "strong_camera_shift", "strong_lighting_shift"),
        required=True,
    )
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--device", default="cuda")
    select.add_argument("--dry-run", action="store_true")

    external = subparsers.add_parser(
        "evaluate-external-visual-selection",
        help="join frozen visual selections to already accepted M3C outcomes",
    )
    external.add_argument(
        "--selection-manifest", type=Path, action="append", required=True
    )
    external.add_argument(
        "--external-config",
        type=Path,
        default=_CONFIG_ROOT / "external-evaluation.json",
    )
    external.add_argument("--candidate-pool-dir", type=Path, required=True)
    external.add_argument("--m3c-selection-manifest", type=Path, required=True)
    external.add_argument("--m3c-result-report", type=Path, required=True)
    external.add_argument("--corruption-dir", type=Path, required=True)
    external.add_argument("--selected-evaluation-dir", type=Path, required=True)
    external.add_argument("--remainder-evaluation-dir", type=Path, required=True)
    external.add_argument("--output", type=Path, required=True)
    external.add_argument("--dry-run", action="store_true")


def _anchor_selection_reasons(path: Path) -> tuple[dict[str, str], str]:
    from latentguard.integrations.maniskill_pickcube.state_indexed_build import (
        load_anchor_manifest,
    )

    manifest = load_anchor_manifest(path)
    reasons = {
        record.anchor.anchor_id: record.anchor.selection_reason
        for record in manifest.records
    }
    if len(reasons) != len(manifest.records):
        _fail("anchor manifest", "duplicate anchor identity")
    return reasons, manifest.content_digest


def _load_accepted_dataset(
    args: argparse.Namespace,
) -> AcceptedActionVerifierDatasetV1:
    from latentguard.training.dataset import load_accepted_action_verifier_dataset

    reasons, manifest_digest = _anchor_selection_reasons(args.anchor_manifest_dir)
    return load_accepted_action_verifier_dataset(
        args.dataset_root,
        full_target_report=args.acceptance_report,
        anchor_selection_reasons=reasons,
        anchor_manifest_content_digest=manifest_digest,
    )


def _git_sha() -> str:
    root = Path(__file__).resolve().parents[2]
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    ).stdout
    if status:
        _fail("M4B Git identity", "tracked or untracked working tree is not clean")
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _run_prepare_backbone(args: argparse.Namespace) -> int:
    from latentguard.visual_training.config import load_backbone_config

    config = load_backbone_config(args.config)
    if args.dry_run:
        print(
            "prepare-visual-backbone dry-run OK: "
            f"config_digest={config.content_digest} downloads=0 output_created=false"
        )
        return 0
    from latentguard.visual_training.backbone import prepare_backbone

    runtime = prepare_backbone(config, args.output, device=args.device)
    print(
        "prepare-visual-backbone OK: "
        f"backbone_digest={runtime.manifest.content_digest} "
        f"weight_digest={runtime.manifest.weight_content_digest}"
    )
    return 0


def _run_extract_features(args: argparse.Namespace) -> int:
    from latentguard.vision_data.models import VisualVerifierExternalDatasetV1
    from latentguard.vision_data.serialization import load_visual_dataset
    from latentguard.visual_training.config import load_backbone_config

    config = load_backbone_config(args.backbone_config)
    if args.dry_run:
        print(
            "extract-visual-features dry-run OK: "
            f"config_digest={config.content_digest} images_opened=0 "
            "output_created=false"
        )
        return 0
    dataset = load_visual_dataset(args.dataset_dir)
    if (
        isinstance(dataset, VisualVerifierExternalDatasetV1)
        and not args.evaluation_only
    ):
        _fail("external feature cache", "--evaluation-only is required")
    model_freeze_digest: str | None = None
    if isinstance(dataset, VisualVerifierExternalDatasetV1):
        if args.model_freeze_manifest is None:
            _fail("external feature cache", "--model-freeze-manifest is required")
        from latentguard.visual_training.external import load_model_freeze_manifest

        model_freeze_digest = cast(
            str,
            load_model_freeze_manifest(args.model_freeze_manifest)["content_digest"],
        )
    elif args.model_freeze_manifest is not None:
        _fail("development feature cache", "model freeze is external-only")
    from latentguard.visual_training.backbone import (
        load_backbone_manifest,
        prepare_backbone,
    )
    from latentguard.visual_training.cache import extract_feature_cache

    expected = load_backbone_manifest(args.backbone_manifest)
    runtime = prepare_backbone(config, args.backbone_manifest, device=args.device)
    if runtime.manifest != expected:
        _fail("external feature cache", "backbone identity changed")
    loaded, extracted = extract_feature_cache(
        dataset,
        args.dataset_dir,
        runtime,
        args.output_dir,
        git_sha=_git_sha(),
        model_freeze_digest=model_freeze_digest,
        batch_size=args.batch_size,
        limit_images=args.limit_images,
        resume=args.resume,
    )
    print(
        "extract-visual-features OK: "
        f"rows={len(loaded.entries)} extracted={extracted} "
        f"cache_digest={loaded.content_digest}"
    )
    return 0


def _five_paths(
    values: list[Path], context: str
) -> tuple[Path, Path, Path, Path, Path]:
    if len(values) != 5:
        _fail(context, "expected exactly five ordered paths")
    return cast(tuple[Path, Path, Path, Path, Path], tuple(values))


def _run_prepare_teacher(args: argparse.Namespace) -> int:
    if args.dry_run:
        for name in ("teacher_checkpoint", "teacher_calibration", "teacher_thresholds"):
            _five_paths(getattr(args, name), name)
        print(
            "prepare-visual-teacher-targets dry-run OK: models_loaded=0 "
            "output_created=false"
        )
        return 0
    dataset = _load_accepted_dataset(args)
    from latentguard.visual_training.teacher import (
        load_accepted_teacher_bundle,
        prepare_teacher_targets,
    )

    bundle = load_accepted_teacher_bundle(
        args.bundle_report,
        preprocessing=args.preprocessing,
        checkpoints=_five_paths(args.teacher_checkpoint, "teacher checkpoints"),
        calibrations=_five_paths(args.teacher_calibration, "teacher calibrations"),
        thresholds=_five_paths(args.teacher_thresholds, "teacher thresholds"),
        device=args.device,
    )
    loaded, computed = prepare_teacher_targets(
        bundle,
        dataset,
        args.output_dir,
        git_sha=_git_sha(),
        device=args.device,
        batch_size=args.batch_size,
        limit_samples=args.limit_samples,
        resume=args.resume,
    )
    print(
        "prepare-visual-teacher-targets OK: "
        f"candidates={len(loaded.candidate_ids)} computed={computed} "
        f"cache_digest={loaded.content_digest}"
    )
    return 0


def _run_train(args: argparse.Namespace) -> int:
    from latentguard.visual_training.config import (
        load_visual_model_config,
        load_visual_training_config,
    )

    model_config = load_visual_model_config(args.model_config)
    training_config = load_visual_training_config(args.training_config)
    if model_config.backbone_frozen != (args.feature_cache is not None):
        _fail("visual training", "feature-cache use differs from model contract")
    if model_config.distillation_enabled != (args.teacher_cache is not None):
        _fail("visual training", "teacher-cache use differs from model contract")
    if args.dry_run:
        print(
            "train-visual-action-verifier dry-run OK: "
            f"model={model_config.model_id} seed={args.seed} "
            f"model_config_digest={model_config.content_digest} "
            f"training_config_digest={training_config.content_digest} "
            "samples_loaded=0 steps=0 output_created=false"
        )
        return 0
    completed_report = args.output_dir / "training-result.json"
    completed_checkpoint = args.output_dir / "checkpoint-last.pt"
    if args.resume and completed_report.exists() and completed_checkpoint.exists():
        from latentguard.visual_training.checkpoint import inspect_visual_checkpoint
        from latentguard.visual_training.training import VisualTrainingResultV1

        metadata = inspect_visual_checkpoint(completed_checkpoint)
        if (
            not metadata.progress.complete
            or metadata.progress.binding.git_sha != _git_sha()
            or metadata.progress.binding.seed != args.seed
            or metadata.progress.binding.model_config_digest
            != model_config.content_digest
            or metadata.progress.binding.training_config_digest
            != training_config.content_digest
        ):
            _fail("visual training resume", "completed run identity differs")
        result = cast(VisualTrainingResultV1, _load_result(completed_report))
        from latentguard.visual_training.checkpoint import checkpoint_content_digest

        best_checkpoint = args.output_dir / "checkpoint-best.pt"
        if (
            result.model_id != model_config.model_id
            or result.seed != args.seed
            or not best_checkpoint.is_file()
            or result.best_checkpoint_digest
            != checkpoint_content_digest(best_checkpoint)
        ):
            _fail("visual training resume", "completed report binding differs")
        print(
            "train-visual-action-verifier OK: "
            f"model={result.model_id} seed={result.seed} steps={result.global_steps} "
            f"validation_auprc={result.mean_corrupted_only_failure_auprc:.6f} "
            "zero_work_resume=true"
        )
        return 0
    from latentguard.vision_data.serialization import load_visual_dataset
    from latentguard.visual_training.backbone import load_backbone_manifest
    from latentguard.visual_training.cache import load_feature_cache, load_teacher_cache
    from latentguard.visual_training.data import build_visual_action_training_dataset
    from latentguard.visual_training.models import build_visual_action_verifier
    from latentguard.visual_training.training import (
        train_visual_action_verifier,
        write_training_result,
    )

    visual = load_visual_dataset(args.visual_dataset_dir)
    structured = _load_accepted_dataset(args)
    joined = build_visual_action_training_dataset(visual, structured)
    feature_cache = (
        None if args.feature_cache is None else load_feature_cache(args.feature_cache)
    )
    teacher_cache = (
        None if args.teacher_cache is None else load_teacher_cache(args.teacher_cache)
    )
    backbone = load_backbone_manifest(args.backbone_manifest)
    model = build_visual_action_verifier(model_config, seed=args.seed)
    result = train_visual_action_verifier(
        model,
        joined,
        visual,
        args.visual_dataset_dir,
        feature_cache,
        teacher_cache,
        training_config,
        args.output_dir,
        git_sha=_git_sha(),
        backbone_digest=backbone.content_digest,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        limit_samples=args.limit_samples,
        mixed_precision=args.mixed_precision,
    )
    if not completed_report.exists():
        write_training_result(completed_report, result)
    print(
        "train-visual-action-verifier OK: "
        f"model={result.model_id} seed={result.seed} steps={result.global_steps} "
        f"validation_auprc={result.mean_corrupted_only_failure_auprc:.6f} "
        f"zero_work_resume={str(result.zero_work_resume).lower()}"
    )
    return 0


def _load_result(path: Path) -> object:
    from latentguard.visual_training.training import (
        DomainValidationResultV1,
        VisualTrainingResultV1,
    )

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        _fail("M4B run report", "expected object")
    domains = raw.get("domain_results")
    if not isinstance(domains, list):
        _fail("M4B run report", "domain results are absent")
    fields = {
        "best_checkpoint_digest",
        "completed_epochs",
        "content_digest",
        "domain_results",
        "duration_seconds",
        "global_steps",
        "mean_corrupted_only_failure_auprc",
        "model_id",
        "peak_gpu_memory_bytes",
        "resumed",
        "seed",
        "selected_epoch",
        "total_parameters",
        "trainable_parameters",
        "worst_domain_corrupted_only_failure_auprc",
        "zero_work_resume",
    }
    if set(raw) != fields:
        _fail("M4B run report", "field inventory differs")
    body = {key: value for key, value in raw.items() if key != "content_digest"}
    expected_digest = (
        "sha256:"
        + hashlib.sha256(
            canonical_json_bytes(body, context="M4BVisualTrainingResultV1")
        ).hexdigest()
    )
    if raw["content_digest"] != expected_digest:
        _fail("M4B run report", "content digest differs")
    return VisualTrainingResultV1(
        **{
            key: value
            for key, value in raw.items()
            if key not in {"content_digest", "domain_results"}
        },
        domain_results=tuple(DomainValidationResultV1(**item) for item in domains),
    )


def _finalize_report_payload(payload: Mapping[str, object]) -> dict[str, object]:
    body = dict(payload)
    if "content_digest" not in body:
        body["content_digest"] = (
            "sha256:"
            + hashlib.sha256(
                canonical_json_bytes(
                    body,
                    context="M4BWorkflowReportV1",
                )
            ).hexdigest()
        )
    return body


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        _fail("M4B report", "refuses to overwrite completed output")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _finalize_report_payload(payload)
    path.write_text(
        json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_benchmark(args: argparse.Namespace) -> int:
    from latentguard.visual_training.config import load_seed_policy

    policy = load_seed_policy(args.seed_policy)
    if args.stage == "five-seed" and not args.explicit_five_seed_authorization:
        _fail("five-seed benchmark", "explicit authorization flag is required")
    if args.stage == "five-seed":
        _fail("five-seed benchmark", "M4B does not authorize seeds 3 or 4")
    if args.dry_run:
        print(
            "benchmark-visual-action-verifier dry-run OK: "
            f"stage={args.stage} policy_seeds={list(policy.promoted_seeds)} "
            "training_runs=0 output_created=false"
        )
        return 0
    results = tuple(_load_result(path) for path in args.run_report)
    if args.stage == "single-seed-screen":
        from latentguard.visual_training.training import (
            create_seed_zero_promotion_record,
        )

        promotion = create_seed_zero_promotion_record(results)  # type: ignore[arg-type]
        payload = promotion.as_mapping()
    else:
        if args.promotion_record is None:
            _fail("promoted benchmark", "--promotion-record is required")
        promotion = json.loads(args.promotion_record.read_text(encoding="utf-8"))
        promoted = tuple(promotion.get("promoted_model_ids", ()))
        if len(promoted) > policy.maximum_promoted_families:
            _fail("promoted benchmark", "promotion exceeds three families")
        observed = {(item.model_id, item.seed) for item in results}  # type: ignore[attr-defined]
        expected = {
            (model_id, seed) for model_id in promoted for seed in policy.promoted_seeds
        }
        if observed != expected:
            _fail(
                "promoted benchmark", "requires exact seeds 0, 1, 2 for promoted models"
            )
        payload = {
            "schema_version": "1.0",
            "stage": args.stage,
            "promoted_model_ids": list(promoted),
            "seeds": list(policy.promoted_seeds),
            "run_results": [item.as_mapping() for item in results],  # type: ignore[attr-defined]
            "seed_zero_reused_count": len(promoted),
            "five_seed_executed": False,
        }
    if args.output.exists() and args.resume:
        try:
            observed = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise M4BCommandError(f"benchmark resume: {exc}") from exc
        if observed != _finalize_report_payload(cast(Mapping[str, object], payload)):
            _fail("benchmark resume", "completed report identity differs")
        print("benchmark-visual-action-verifier OK: zero_work_resume=true")
        return 0
    _write_new_json(args.output, cast(Mapping[str, object], payload))
    print(f"benchmark-visual-action-verifier OK: stage={args.stage}")
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            "evaluate-visual-action-verifier dry-run OK: test_opened=false "
            "output_created=false"
        )
        return 0
    if len(args.checkpoint) != 3:
        _fail("internal visual evaluation", "expected checkpoints for seeds 0, 1, 2")
    from latentguard.vision_data.serialization import load_visual_dataset
    from latentguard.visual_training.cache import load_feature_cache
    from latentguard.visual_training.config import (
        load_visual_model_config,
        load_visual_training_config,
    )
    from latentguard.visual_training.data import build_visual_action_training_dataset
    from latentguard.visual_training.evaluation import (
        evaluate_promoted_visual_ensemble,
    )

    model_config = load_visual_model_config(args.model_config)
    training_config = load_visual_training_config(args.training_config)
    visual = load_visual_dataset(args.visual_dataset_dir)
    structured = _load_accepted_dataset(args)
    joined = build_visual_action_training_dataset(visual, structured)
    feature_cache = (
        None if args.feature_cache is None else load_feature_cache(args.feature_cache)
    )
    if model_config.backbone_frozen != (feature_cache is not None):
        _fail("internal visual evaluation", "feature cache use differs")
    report = evaluate_promoted_visual_ensemble(
        model_config,
        training_config,
        cast(tuple[Path, Path, Path], tuple(args.checkpoint)),
        joined,
        visual,
        args.visual_dataset_dir,
        feature_cache,
        args.output_dir,
        device=args.device,
    )
    print(
        "evaluate-visual-action-verifier OK: "
        f"model={model_config.model_id} report_digest={report['content_digest']}"
    )
    return 0


def _run_select(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            "select-visual-action-candidates dry-run OK: "
            "outcomes_available_during_selection=false outcomes_accepted=false"
        )
        return 0
    if len(args.checkpoint) != 3:
        _fail("external visual selection", "expected checkpoints for seeds 0, 1, 2")
    from latentguard.selection.serialization import load_candidate_pool
    from latentguard.vision_data.models import VisualVerifierExternalDatasetV1
    from latentguard.vision_data.serialization import load_visual_dataset
    from latentguard.visual_training.cache import load_feature_cache
    from latentguard.visual_training.config import (
        load_external_evaluation_config,
        load_visual_model_config,
    )
    from latentguard.visual_training.external import (
        select_external_visual_candidates,
    )

    visual = load_visual_dataset(args.external_visual_dataset_dir)
    if not isinstance(visual, VisualVerifierExternalDatasetV1):
        _fail("external visual selection", "expected the M3C external visual dataset")
    pool = load_candidate_pool(args.candidate_pool_dir)
    cache = load_feature_cache(args.feature_cache)
    config = load_visual_model_config(args.model_config)
    external_config = load_external_evaluation_config(args.external_config)
    if args.domain not in external_config.domains:
        _fail("external visual selection", "domain differs from frozen config")
    manifest = select_external_visual_candidates(
        visual_dataset=visual,
        candidate_pool=pool,
        feature_cache=cache,
        model_config=config,
        checkpoints=cast(tuple[Path, Path, Path], tuple(args.checkpoint)),
        model_freeze_manifest=args.model_freeze_manifest,
        external_config_digest=external_config.content_digest,
        selection_semantic=external_config.selection_semantic,
        domain_id=args.domain,
        output_path=args.output,
        device=args.device,
    )
    print(
        "select-visual-action-candidates OK: "
        f"domain={args.domain} manifest_digest={manifest['content_digest']} "
        "outcomes_available_during_selection=false"
    )
    return 0


def _run_external(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(
            "evaluate-external-visual-selection dry-run OK: outcomes_joined=0 "
            "output_created=false"
        )
        return 0
    if len(args.selection_manifest) != 3:
        _fail("external visual evaluation", "expected exactly three domain manifests")
    from latentguard.corruptions.serialization import load_corruption_dataset
    from latentguard.evaluation.serialization import (
        compute_corruption_dataset_digest,
        load_evaluation_dataset,
    )
    from latentguard.selection.manifest import load_blind_selection_manifest
    from latentguard.selection.serialization import load_candidate_pool
    from latentguard.training.reporting import load_strict_report
    from latentguard.visual_training.config import load_external_evaluation_config
    from latentguard.visual_training.external import (
        evaluate_external_visual_selections,
    )

    pool = load_candidate_pool(args.candidate_pool_dir)
    envelope = load_blind_selection_manifest(args.m3c_selection_manifest)
    corruptions = load_corruption_dataset(args.corruption_dir)
    corruption_digest = compute_corruption_dataset_digest(args.corruption_dir)
    selected = load_evaluation_dataset(
        args.selected_evaluation_dir,
        corruption_dataset=corruptions,
        expected_corruption_digest=corruption_digest,
    )
    remainder = load_evaluation_dataset(
        args.remainder_evaluation_dir,
        corruption_dataset=corruptions,
        expected_corruption_digest=corruption_digest,
    )
    accepted = load_strict_report(
        args.m3c_result_report,
        expected_report_type="m3c_candidate_selection_result_v1",
    )
    external_config = load_external_evaluation_config(args.external_config)
    report = evaluate_external_visual_selections(
        visual_selection_paths=cast(
            tuple[Path, Path, Path], tuple(args.selection_manifest)
        ),
        candidate_pool=pool,
        original_selection_manifest=envelope,
        selected_dataset=selected,
        remainder_dataset=remainder,
        accepted_m3c_report=accepted.to_dict(),
        external_config_digest=external_config.content_digest,
        bootstrap_samples=external_config.bootstrap_samples,
        bootstrap_seed=external_config.bootstrap_seed,
        output_path=args.output,
    )
    print(
        "evaluate-external-visual-selection OK: "
        f"report_digest={report['content_digest']} simulator_replay_performed=false"
    )
    return 0


def run_m4b_command(args: argparse.Namespace) -> int | None:
    """Dispatch one M4B command, preserving lazy optional dependencies."""
    if args.command not in M4B_COMMANDS:
        return None
    handlers: Mapping[str, Callable[[argparse.Namespace], int]] = {
        "prepare-visual-backbone": _run_prepare_backbone,
        "extract-visual-features": _run_extract_features,
        "prepare-visual-teacher-targets": _run_prepare_teacher,
        "train-visual-action-verifier": _run_train,
        "benchmark-visual-action-verifier": _run_benchmark,
        "evaluate-visual-action-verifier": _run_evaluate,
        "select-visual-action-candidates": _run_select,
        "evaluate-external-visual-selection": _run_external,
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
        print(
            f"{args.command} failed: {sanitize_operational_text(error)}",
            file=sys.stderr,
        )
        return 1


__all__ = ["M4B_COMMANDS", "M4BCommandError", "add_m4b_subparsers", "run_m4b_command"]

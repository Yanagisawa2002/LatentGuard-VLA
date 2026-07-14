"""Command-line interface for local data checks and remote synchronization."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latentguard.corruptions.config import load_corruption_plan
from latentguard.corruptions.generation import (
    build_proposal_identifier,
    generate_corruption_proposals,
)
from latentguard.corruptions.serialization import (
    CorruptionDataset,
    load_corruption_dataset,
    save_corruption_dataset,
)
from latentguard.remote import RemoteSyncError, resolve_remote_config, sync_remote
from latentguard.serialization import load_episodes, save_episodes
from latentguard.synthetic import generate_synthetic_episodes
from latentguard.validation import validate_episodes


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latentguard",
        description="LatentGuard-VLA local validation utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sanity = subparsers.add_parser(
        "sanity-data",
        help="generate and round-trip deterministic synthetic episodes",
    )
    sanity.add_argument("--seed", type=int, default=0)
    sanity.add_argument("--output-dir", type=Path, required=True)
    sanity.add_argument("--episode-count", type=_positive_int, default=3)
    sanity.add_argument("--episode-length", type=_positive_int, default=8)
    sanity.add_argument("--action-dim", type=_positive_int, default=7)
    sanity.add_argument("--robot-state-dim", type=_positive_int, default=10)
    sanity.add_argument("--camera-count", type=_nonnegative_int, default=1)
    sanity.add_argument("--image-height", type=_positive_int, default=8)
    sanity.add_argument("--image-width", type=_positive_int, default=8)
    sanity.add_argument("--candidate-count", type=_positive_int, default=4)
    sanity.add_argument(
        "--depth",
        action="store_true",
        help="include aligned synthetic depth for every camera",
    )

    remote = subparsers.add_parser(
        "remote-sync",
        help="synchronize a remote checkout to an exact pushed revision",
    )
    remote.add_argument("--dry-run", action="store_true")
    remote.add_argument("--host", help="SSH host alias")
    remote.add_argument("--repo-dir", help="remote repository directory")
    remote.add_argument("--branch", help="Git branch to synchronize")
    remote.add_argument("--commit", dest="expected_commit", help="full commit SHA")
    remote.add_argument("--ssh-executable", help="system SSH executable")
    remote.add_argument("--connect-timeout", type=_positive_int)

    corrupt = subparsers.add_parser(
        "corrupt-data",
        help="generate deterministic unlabeled action corruption proposals",
    )
    corrupt.add_argument("--input-dir", type=Path, required=True)
    corrupt.add_argument("--output-dir", type=Path, required=True)
    corrupt.add_argument("--config", type=Path, required=True)
    corrupt.add_argument("--seed", type=_nonnegative_int, required=True)
    corrupt.add_argument("--episode-limit", type=_positive_int)
    corrupt.add_argument("--candidate-limit", type=_positive_int)
    corrupt.add_argument("--proposal-limit", type=_positive_int)
    corrupt.add_argument(
        "--strict-applicability",
        action="store_true",
        help="fail instead of recording a non-applicable transformation skip",
    )
    corrupt.add_argument(
        "--audit",
        "--verbose",
        dest="audit",
        action="store_true",
        help="print concise applicability-skip audit records",
    )

    return parser


def _values_equal(left: Any, right: Any) -> bool:
    """Compare nested model values without invoking ndarray dataclass equality."""
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and bool(np.array_equal(left, right))
        )
    if is_dataclass(left) and not isinstance(left, type):
        return all(
            _values_equal(getattr(left, field.name), getattr(right, field.name))
            for field in fields(left)
        )
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)


def _run_sanity_data(args: argparse.Namespace) -> int:
    try:
        episodes = generate_synthetic_episodes(
            seed=args.seed,
            episode_count=args.episode_count,
            episode_length=args.episode_length,
            action_dim=args.action_dim,
            robot_state_dim=args.robot_state_dim,
            camera_count=args.camera_count,
            image_height=args.image_height,
            image_width=args.image_width,
            depth_enabled=args.depth,
            candidate_count=args.candidate_count,
        )
        validate_episodes(episodes)
        manifest_path = save_episodes(episodes, args.output_dir)
        reloaded = load_episodes(args.output_dir)
        validate_episodes(reloaded)
        if not _values_equal(episodes, reloaded):
            raise RuntimeError("round-trip comparison found a changed value")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"sanity-data failed: {error}", file=sys.stderr)
        return 1

    observation_count = sum(len(episode.observations.frames) for episode in episodes)
    candidate_count = sum(len(episode.candidates) for episode in episodes)
    print(
        "sanity-data OK: "
        f"episodes={len(episodes)} observations={observation_count} "
        f"candidates={candidate_count} round-trip=verified "
        f"manifest={manifest_path}"
    )
    return 0


def _run_remote_sync(args: argparse.Namespace) -> int:
    try:
        config = resolve_remote_config(
            host=args.host,
            repo_dir=args.repo_dir,
            branch=args.branch,
            expected_commit=args.expected_commit,
            ssh_executable=args.ssh_executable,
            connect_timeout=args.connect_timeout,
        )
        result = sync_remote(config, dry_run=args.dry_run)
    except (RemoteSyncError, OSError, ValueError) as error:
        print(f"remote-sync failed: {error}", file=sys.stderr)
        return 1

    print(result)
    return 0


def _source_dataset_identifier(input_dir: Path) -> str:
    """Hash a validated M0 bundle independently of its absolute location."""
    root = input_dir.absolute()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise RuntimeError("source dataset bundle contains no files")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, byteorder="big"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _validate_dataset_path_separation(input_dir: Path, output_dir: Path) -> None:
    """Reject any source/output ancestry overlap before a bundle can be changed."""
    source = input_dir.absolute().resolve()
    output = output_dir.absolute().resolve()
    if (
        source == output
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError(
            "input and output dataset directories must not equal, contain, "
            "or be contained by one another"
        )


def _validate_proposal_identifiers(dataset: CorruptionDataset) -> None:
    """Recompute every deterministic proposal identifier from its manifest data."""
    for proposal in dataset.proposals:
        expected = build_proposal_identifier(
            source_episode_id=proposal.source_episode_id,
            source_candidate_id=proposal.source_candidate_id,
            corruption_name=proposal.corruption_type,
            resolved_parameters=proposal.resolved_parameters,
            seed=proposal.seed,
            generation_ordinal=proposal.generation_ordinal,
        )
        if proposal.proposal_id != expected:
            raise RuntimeError(
                f"proposal identifier mismatch for ordinal "
                f"{proposal.generation_ordinal}: expected {expected}, "
                f"got {proposal.proposal_id}"
            )


def _run_corrupt_data(args: argparse.Namespace) -> int:
    try:
        _validate_dataset_path_separation(args.input_dir, args.output_dir)
        episodes = load_episodes(args.input_dir)
        validate_episodes(episodes)
        if args.episode_limit is not None:
            episodes = episodes[: args.episode_limit]
        plan = load_corruption_plan(args.config)
        result = generate_corruption_proposals(
            episodes,
            plan.action_layout,
            plan.corruptions,
            base_seed=args.seed,
            candidate_limit=args.candidate_limit,
            proposal_limit=args.proposal_limit,
            strict_applicability=args.strict_applicability,
        )
        dataset = CorruptionDataset(
            source_dataset_id=_source_dataset_identifier(args.input_dir),
            action_layout=plan.action_layout,
            proposals=result.proposals,
        )
        _validate_proposal_identifiers(dataset)
        manifest_path = save_corruption_dataset(dataset, args.output_dir)
        reloaded = load_corruption_dataset(args.output_dir)
        _validate_proposal_identifiers(reloaded)
        if not _values_equal(dataset, reloaded):
            raise RuntimeError("corruption round-trip comparison found a changed value")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"corrupt-data failed: {error}", file=sys.stderr)
        return 1

    if args.audit:
        if result.skips:
            for skip in result.skips:
                print(f"corrupt-data audit: skipped {skip}")
        else:
            print("corrupt-data audit: no applicability skips")
    counts = ",".join(
        f"{name}:{count}" for name, count in result.corruption_counts.items()
    )
    print(
        "corrupt-data OK: "
        f"source_episodes={result.source_episode_count} "
        f"source_candidates={result.source_candidate_count} "
        f"proposals={len(result.proposals)} "
        f"skipped_non_applicable={len(result.skips)} "
        "unlabeled=true round-trip=verified "
        f"corruptions={counts or '<none>'} manifest={manifest_path}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the LatentGuard-VLA command-line interface."""
    args = _build_parser().parse_args(argv)
    if args.command == "sanity-data":
        return _run_sanity_data(args)
    if args.command == "remote-sync":
        return _run_remote_sync(args)
    if args.command == "corrupt-data":
        return _run_corrupt_data(args)
    raise AssertionError(f"unhandled command: {args.command}")

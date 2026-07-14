"""Command-line interface for local data checks and remote synchronization."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the LatentGuard-VLA command-line interface."""
    args = _build_parser().parse_args(argv)
    if args.command == "sanity-data":
        return _run_sanity_data(args)
    if args.command == "remote-sync":
        return _run_remote_sync(args)
    raise AssertionError(f"unhandled command: {args.command}")

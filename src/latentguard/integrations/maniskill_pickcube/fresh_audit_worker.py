"""Private one-trajectory fresh-state audit worker for native runtime isolation."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from latentguard.integrations.maniskill_pickcube.reporting import (
    write_sanitized_report,
)
from latentguard.integrations.maniskill_pickcube.source_generation import (
    LazyManiSkillSourceEnvironmentFactory,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_archive import (
    load_state_indexed_archive,
)
from latentguard.integrations.maniskill_pickcube.state_indexed_source import (
    verify_all_indexed_states_fresh,
)
from latentguard.m3a_cli import _load_scope


def _parser() -> argparse.ArgumentParser:
    """Build the private worker parser without extending the public CLI surface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-archive-dir", type=Path, required=True)
    parser.add_argument("--expected-archive-digest", required=True)
    parser.add_argument("--source-trajectory-id", required=True)
    parser.add_argument("--compatibility-report", type=Path, required=True)
    parser.add_argument("--expected-contract", type=Path, required=True)
    parser.add_argument("--action-layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> int:
    """Audit one exact trajectory and publish only compact numeric evidence."""

    archive = load_state_indexed_archive(args.runtime_archive_dir)
    if archive.content_digest != args.expected_archive_digest:
        raise ValueError("fresh audit archive digest differs")
    episodes = tuple(
        episode
        for episode in archive.episodes
        if episode.source_trajectory_id == args.source_trajectory_id
    )
    if len(episodes) != 1:
        raise ValueError("fresh audit trajectory identity must resolve exactly once")
    episode = episodes[0]
    scope = _load_scope(args)
    if episode.compatibility_identity != scope.binding.report.compatibility_identity:
        raise ValueError("fresh audit compatibility identity differs")
    records = verify_all_indexed_states_fresh(
        episode,
        environment_factory=LazyManiSkillSourceEnvironmentFactory(),
        settings=scope.settings,
        action_contract=scope.action_contract,
        key_contract=scope.task_keys,
    )
    write_sanitized_report(
        {
            "archive_content_digest": archive.content_digest,
            "compatibility_identity": episode.compatibility_identity,
            "episode_content_digest": episode.content_digest,
            "records": [asdict(record) for record in records],
            "schema_version": "1.0",
            "source_trajectory_id": episode.source_trajectory_id,
            "state_count": len(episode.states),
        },
        args.output,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the private audit worker with concise, credential-free failures."""

    try:
        return run(_parser().parse_args(argv))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            f"fresh-state audit worker failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess.
    raise SystemExit(main())

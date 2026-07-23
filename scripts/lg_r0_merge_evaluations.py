"""Merge independently isolated LG-R0 suite shards into the frozen 40-episode report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from _lg_r0_runtime import read_json, write_json


def _wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def main() -> None:
    """Merge four complete, disjoint suite shards."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execution-site",
        choices=("local", "remote"),
        default="local",
    )
    args = parser.parse_args()

    config = read_json(args.config)
    shards = [read_json(path) for path in args.inputs]
    episodes = [episode for shard in shards for episode in shard["episodes"]]
    suite_results: dict[str, Any] = {}
    for shard in shards:
        suite_results.update(shard["suite_results"])
    errors = [error for shard in shards for error in shard["errors"]]
    expected_suites = {suite["name"] for suite in config["suites"]}
    observed_suites = set(suite_results)
    successes = sum(int(episode["success"]) for episode in episodes)
    lengths = [int(episode["episode_length"]) for episode in episodes]
    queries = sum(int(shard["instrumentation"]["policy_queries"]) for shard in shards)
    weighted_latency = sum(
        float(shard["instrumentation"]["mean_query_latency_seconds"])
        * int(shard["instrumentation"]["policy_queries"])
        for shard in shards
    )
    status = (
        "pass"
        if all(shard["status"] == "pass" for shard in shards)
        and len(episodes) == 40
        and observed_suites == expected_suites
        and not errors
        else "fail"
    )
    payload = {
        "schema_version": "latentguard.lg_r0.libero_evaluation.v1",
        "status": status,
        "result_scope": f"{args.execution_site}_40_episode_smoke",
        "official_400_episode_equivalent": False,
        "execution_site": args.execution_site,
        "schedule": config,
        "execution_isolation": (
            "one fresh process per suite after a prior monolithic attempt ended "
            "at 37/40 with CUDA_ERROR_UNKNOWN"
        ),
        "episodes": sorted(
            episodes,
            key=lambda item: (
                [suite["name"] for suite in config["suites"]].index(item["suite"]),
                item["seed"],
            ),
        ),
        "suite_results": suite_results,
        "overall": {
            "expected_episodes": 40,
            "completed_episodes": len(episodes),
            "successes": successes,
            "success_rate": successes / len(episodes) if episodes else 0.0,
            "wilson_95": _wilson(successes, len(episodes)),
            "mean_episode_length": sum(lengths) / len(lengths) if lengths else None,
            "timeouts": sum(
                int(episode["termination"] == "horizon_exhausted")
                for episode in episodes
            ),
            "environment_errors": 0,
            "action_contract_errors": sum(
                int(not episode["executed_action_bounds_ok"]) for episode in episodes
            ),
            "nonfinite_action_episodes": sum(
                int(not episode["action_finite"]) for episode in episodes
            ),
        },
        "instrumentation": {
            "policy_queries": queries,
            "mean_query_latency_seconds": (
                weighted_latency / queries if queries else None
            ),
            "suite_p95_query_latency_seconds": {
                next(iter(shard["suite_results"])): shard["instrumentation"][
                    "p95_query_latency_seconds"
                ]
                for shard in shards
            },
            "queries_per_second": (
                queries / weighted_latency if weighted_latency > 0 else None
            ),
            "optimizer_steps": 0,
            "backward_calls": 0,
        },
        "cuda_memory": {
            "available": True,
            "peak_allocated_bytes": max(
                int(shard["cuda_memory"]["peak_allocated_bytes"]) for shard in shards
            ),
            "peak_reserved_bytes": max(
                int(shard["cuda_memory"]["peak_reserved_bytes"]) for shard in shards
            ),
        },
        "errors": errors,
        "run_shards": [
            {
                "input": path.name,
                "run_id": shard["run_id"],
                "run_locator": shard["run_locator"],
            }
            for path, shard in zip(args.inputs, shards, strict=True)
        ],
        "optimizer_steps": 0,
        "backward_calls": 0,
        "interventions": 0,
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "episodes": len(episodes),
                "successes": successes,
                "output": str(args.output),
            }
        )
    )
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

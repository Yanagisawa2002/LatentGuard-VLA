"""Run frozen VLA-JEPA diagnostics on matched success/failure anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _lg_r1b_common import output_root, resolve_repo_path, write_json
from lg_r1_build_stage_labels import _read_jsonl
from lg_r1_evaluate_progress import _vlajepa_descriptive


def main() -> None:
    """Extract official frozen diagnostics only for matched window endpoints."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--max-samples", type=int, default=64)
    args = parser.parse_args()
    runtime = (
        resolve_repo_path(args.runtime_root)
        if args.runtime_root is not None
        else output_root()
    )
    predictions = _read_jsonl(runtime / "sarm_zero_shot_predictions.jsonl")
    predictions_by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        predictions_by_episode.setdefault(str(row["episode_id"]), []).append(row)
    failure_windows = _read_jsonl(runtime / "failure_windows" / "failure_windows.jsonl")
    success_windows = _read_jsonl(
        runtime / "failure_windows" / "matched_success_windows.jsonl"
    )
    anchors: list[dict[str, Any]] = []
    for window in [*failure_windows, *success_windows]:
        candidates = [
            row
            for row in predictions_by_episode.get(str(window["episode_id"]), [])
            if int(window["start_frame"])
            <= int(row["frame_index"])
            <= int(window["end_frame"])
        ]
        if not candidates:
            continue
        row = min(
            candidates,
            key=lambda item: abs(int(item["frame_index"]) - int(window["end_frame"])),
        )
        copied = dict(row)
        copied["split"] = "test"
        anchors.append(copied)
    unique = {(str(row["episode_id"]), int(row["frame_index"])): row for row in anchors}
    result = _vlajepa_descriptive(
        runtime,
        list(unique.values()),
        max_samples=args.max_samples,
    )
    result["schema_version"] = "latentguard.lg_r1b.vlajepa_descriptive_comparison.v1"
    result["matched_window_anchors_available"] = len(unique)
    result["failure_head_trained"] = False
    result["claim_label"] = "frozen descriptive representation diagnostic"
    write_json(runtime / "vlajepa_descriptive_comparison.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

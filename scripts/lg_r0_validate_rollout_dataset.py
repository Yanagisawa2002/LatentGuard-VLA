"""Reload and frame-iterate every dataset bound by an LG-R0 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _lg_r0_runtime import read_json, write_json

from latentguard.adapters.vla_jepa.constants import CHECKPOINT_REVISION


def main() -> None:
    """Validate metadata, feature presence, reload, and frame iteration."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    manifest = read_json(args.manifest)
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    required = {
        f"{OBS_IMAGES}.image",
        f"{OBS_IMAGES}.image2",
        OBS_STATE,
        ACTION,
        "next.success",
        "next.done",
    }
    for entry in manifest["episodes"]:
        root = Path(entry["dataset_runtime_path"])
        try:
            dataset = LeRobotDataset(
                repo_id=entry["dataset_repo_id"],
                root=root,
            )
            missing = sorted(required - set(dataset.features))
            if missing:
                raise ValueError(f"missing dataset features: {missing}")
            iterated = 0
            action_nonfinite = 0
            for index in range(len(dataset)):
                frame = dataset[index]
                action_nonfinite += int(
                    not bool(torch.isfinite(frame[ACTION]).all().item())
                )
                iterated += 1
            if iterated != entry["frame_count"]:
                raise ValueError(
                    "frame count mismatch: "
                    f"manifest={entry['frame_count']} reload={iterated}"
                )
            results.append(
                {
                    "episode_id": entry["episode_id"],
                    "episodes": dataset.num_episodes,
                    "frames": dataset.num_frames,
                    "iterated_frames": iterated,
                    "features": sorted(dataset.features),
                    "action_nonfinite_frames": action_nonfinite,
                    "checkpoint_revision": entry["checkpoint_revision"],
                    "processor_revision": entry["processor_revision"],
                    "policy_identity_bound": entry["checkpoint_revision"]
                    == CHECKPOINT_REVISION,
                    "frame_replay": "pass",
                }
            )
        except Exception as exc:
            failures.append(f"{entry['episode_id']}: {type(exc).__name__}: {exc}")
    payload = {
        "schema_version": "latentguard.lg_r0.rollout_dataset_validation.v1",
        "status": (
            "pass"
            if len(results) == len(manifest["episodes"]) and not failures
            else "fail"
        ),
        "datasets": results,
        "failures": failures,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "datasets": len(results)}))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

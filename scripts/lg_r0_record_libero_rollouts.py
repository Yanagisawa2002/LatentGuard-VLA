"""Record a small official-format LeRobot dataset with LG-R0 identity binding."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from _lg_r0_runtime import (
    configure_libero_assets,
    load_processors,
    load_stack,
    output_root,
    read_json,
    sha256_path,
    write_json,
)

from latentguard.adapters.vla_jepa.constants import CHECKPOINT_REVISION


def _dataset_metadata_identities(path: Path) -> dict[str, dict[str, int | str]]:
    required = {
        "info_json": path / "meta" / "info.json",
        "tasks_parquet": path / "meta" / "tasks.parquet",
    }
    missing = [str(file) for file in required.values() if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"recorded dataset metadata is incomplete: {missing}")
    return {
        name: {
            "bytes": file.stat().st_size,
            "sha256": sha256_path(file),
        }
        for name, file in required.items()
    }


def _public_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    public_payload = json.loads(json.dumps(payload))
    for entry in public_payload["episodes"]:
        entry.pop("dataset_runtime_path", None)
        entry["dataset_locator"] = f"dataset_root/{entry['episode_id']}"
    public_payload["processor_manifest"] = "processor_serialization_manifest.json"
    return public_payload


def main() -> None:
    """Record the fixed four-episode schedule without uploading or training."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processor-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()

    import numpy as np
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.envs import preprocess_observation
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.scripts import lerobot_eval
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    config = read_json(args.config)
    evaluation_config = read_json(Path(config["source_evaluation"]))
    suite_indices = {
        suite["name"]: index for index, suite in enumerate(evaluation_config["suites"])
    }
    libero_assets = configure_libero_assets()
    stack = load_stack()
    preprocessor, postprocessor = load_processors(stack)
    dataset_root = output_root() / f"recordings-{int(time.time())}"
    dataset_root.mkdir(parents=True, exist_ok=False)

    processor_dir = dataset_root / "processors"
    processor_dir.mkdir()
    preprocessor.save_pretrained(processor_dir)
    postprocessor.save_pretrained(processor_dir)
    reloaded_pre, reloaded_post = make_pre_post_processors(
        policy_cfg=stack.config,
        pretrained_path=str(processor_dir),
        preprocessor_overrides={
            "device_processor": {"device": str(stack.config.device)}
        },
    )
    processor_files = [
        {
            "relative_path": str(path.relative_to(processor_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
        for path in sorted(processor_dir.rglob("*"))
        if path.is_file()
    ]
    processor_payload = {
        "schema_version": "latentguard.lg_r0.processor_serialization.v1",
        "status": "pass",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "processor_revision": CHECKPOINT_REVISION,
        "runtime_locator": "dataset_root/processors",
        "files": processor_files,
        "reload_success": True,
        "reloaded_preprocessor_type": type(reloaded_pre).__name__,
        "reloaded_postprocessor_type": type(reloaded_post).__name__,
        "optimizer_steps": 0,
        "backward_calls": 0,
    }
    write_json(args.processor_output, processor_payload)

    entries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for item in config["episodes"]:
        suite = item["suite"]
        task_id = int(item["task_id"])
        seed = int(item["seed"])
        if suite not in suite_indices:
            raise ValueError(
                f"recording suite is absent from source evaluation: {suite}"
            )
        sampling_seed = (
            int(config["policy_sampling_seed"]) + suite_indices[suite] * 100_000 + seed
        )
        torch.manual_seed(sampling_seed)
        torch.cuda.manual_seed_all(sampling_seed)
        horizon = {
            "libero_spatial": 280,
            "libero_object": 280,
            "libero_goal": 300,
            "libero_10": 520,
        }[suite]
        env_config = LiberoEnv(
            task=suite,
            task_ids=[task_id],
            episode_length=horizon,
            obs_type="pixels_agent_pos",
            observation_height=224,
            observation_width=224,
            init_states=True,
            control_mode="relative",
        )
        env = make_env(env_config, n_envs=1, use_async_envs=False)[suite][task_id]
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_config,
            policy_cfg=stack.config,
        )
        instruction = str(list(env.call("task_description"))[0])
        task_name = str(list(env.call("task"))[0])
        episode_id = f"{suite}-task{task_id}-seed{seed}"
        path = dataset_root / episode_id
        features = {
            f"{OBS_IMAGES}.image": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(224, 224, 3),
            ),
            f"{OBS_IMAGES}.image2": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(224, 224, 3),
            ),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }

        def build_frame(
            raw_obs: dict[str, Any],
            env_idx: int,
            action: np.ndarray,
            reward: float,
            success: bool,
            done: bool,
            task: str,
            env_features: dict[str, Any],
            _env_preprocessor: Any = env_preprocessor,
        ) -> dict[str, Any]:
            del env_features
            processed = _env_preprocessor(preprocess_observation(raw_obs))
            state = processed[OBS_STATE][env_idx].to("cpu", dtype=torch.float32).numpy()
            return {
                f"{OBS_IMAGES}.image": raw_obs["pixels"]["image"][env_idx],
                f"{OBS_IMAGES}.image2": raw_obs["pixels"]["image2"][env_idx],
                OBS_STATE: state,
                ACTION: action.astype(np.float32, copy=False),
                "next.reward": np.atleast_1d(np.float32(reward)),
                "next.success": np.atleast_1d(np.bool_(success)),
                "next.done": np.atleast_1d(np.bool_(done)),
                "task": task,
            }

        original_builder = lerobot_eval._build_raw_frame
        lerobot_eval._build_raw_frame = build_frame
        started = time.perf_counter()
        try:
            data = lerobot_eval.rollout(
                env=env,
                policy=stack.policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                seeds=[seed],
                return_observations=False,
                recording_dir=path,
                env_features=features,
                recording_repo_id=None,
                recording_private=False,
            )
            done_index = int(torch.argmax(data["done"][0].to(torch.int64)).item())
            frame_count = done_index + 1
            success = bool(data["success"][0, :frame_count].any().item())
            metadata_identities = _dataset_metadata_identities(path)
            entries.append(
                {
                    "episode_id": episode_id,
                    "suite": suite,
                    "task": task_name,
                    "task_id": task_id,
                    "seed": seed,
                    "policy_sampling_seed": sampling_seed,
                    "instruction": instruction,
                    "dataset_repo_id": "eval_recording",
                    "dataset_runtime_path": str(path),
                    "frame_count": frame_count,
                    "success": success,
                    "termination": "success" if success else "horizon_exhausted",
                    "elapsed_seconds": time.perf_counter() - started,
                    **metadata_identities,
                    "checkpoint_revision": CHECKPOINT_REVISION,
                    "processor_revision": CHECKPOINT_REVISION,
                    "action_mask_semantic": "not_emitted_by_native_inference",
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "episode_id": episode_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            lerobot_eval._build_raw_frame = original_builder
            env.close()

    payload = {
        "schema_version": "latentguard.lg_r0.rollout_dataset_manifest.v1",
        "status": (
            "pass"
            if len(entries) >= 4
            and len({entry["suite"] for entry in entries}) >= 2
            and not errors
            else "fail"
        ),
        "format": config["format"],
        "official_lerobot_dataset_writer": True,
        "upstream_recording_adapter": (
            "LG-R0 supplies policy-ready flattened state because the LeRobot 0.6 "
            "generic raw-frame builder does not traverse LIBERO nested robot_state"
        ),
        "dataset_root_locator": f"LG_R0_OUTPUT_ROOT/{dataset_root.name}",
        "episodes": entries,
        "errors": errors,
        "processor_manifest": str(args.processor_output),
        "optimizer_steps": 0,
        "backward_calls": 0,
        "upload_to_hub": False,
        "execution_site": "remote",
        "libero_asset_snapshot_name": libero_assets.name,
    }
    write_json(args.output, payload)
    if args.public_output is not None:
        write_json(args.public_output, _public_manifest(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "episodes": len(entries),
                "output": str(args.output),
            }
        )
    )
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

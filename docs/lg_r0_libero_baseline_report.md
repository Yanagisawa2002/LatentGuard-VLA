# LG-R0 remote LIBERO baseline report

## Verdict

LG-R0 is **Result B** under the frozen taxonomy. The exact VLA-JEPA stack
loads, produces finite deterministic action chunks, and completes the bounded
remote LIBERO smoke baseline. The milestone is not Result A because the
included world-model path is not an external numeric candidate-action model
and the recorded audit set contains no failure trajectories.

No model or head was trained. LG-R0 created no optimizer, called no backward
pass, generated no checkpoint, selected no threshold, and performed no policy
intervention.

## Frozen runtime

- Python `3.12.3`
- PyTorch `2.11.0+cu128`; CUDA `12.8`
- NVIDIA GeForce RTX 5090, `32,607 MiB`; driver `580.76.05`
- LeRobot `0.6.0`
- `hf-libero==0.1.4`; MuJoCo `3.8.1`
- VLA-JEPA checkpoint
  `lerobot/VLA-JEPA-LIBERO@735d9f692981e286ade093b5046627eda876e5d0`
- Qwen
  `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`
- V-JEPA2
  `facebook/vjepa2-vitl-fpc64-256@b3c1679b7c34d3255ef3547f27c7b226aefab26f`
- LIBERO assets
  `lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`

The model has `2,770,333,319` parameters:

| Module | Parameters |
| --- | ---: |
| Qwen | 2,127,532,032 |
| V-JEPA video encoder | 325,971,328 |
| Video predictor | 161,647,616 |
| Action model | 155,182,343 |

Fail-closed loading found zero missing keys. It accepted only the reviewed
duplicate tied Qwen embedding tensor described in
`artifacts/lg_r0/model_structure.json`. Static batch sizes 1 and 2 produced
finite `[B, 7, 7]` native action tensors. Processor serialization/reload and
fixed-seed reproduction passed. Peak reserved CUDA memory in the static gate
was approximately `6.06 GiB`.

## Closed-loop result

The prerequisite single episode passed with 77 recorded steps. The frozen
development smoke then ran four independent processes, one per suite, with
task 0 and seeds `1000..1009`.

| Suite | Successes | Episodes | Mean steps | Wilson 95% |
| --- | ---: | ---: | ---: | --- |
| `libero_spatial` | 10 | 10 | 77.1 | [0.7225, 1.0000] |
| `libero_object` | 10 | 10 | 145.7 | [0.7225, 1.0000] |
| `libero_goal` | 10 | 10 | 117.9 | [0.7225, 1.0000] |
| `libero_10` | 9 | 10 | 288.0 | [0.5958, 0.9821] |
| **Total** | **39** | **40** | **157.175** | **[0.8712, 0.9956]** |

The preserved unsuccessful episode is
`libero_10-task0-seed1007-episode7`. It executed finite, bounded actions for
the frozen 520-step horizon, made 75 policy queries, and terminated as
`horizon_exhausted`. It is not an environment, checkpoint, processor, action
contract, or non-finite-action error.

This is a `remote_40_episode_smoke`, not the official 400-episode LIBERO
evaluation. It covers only task 0 of each suite, one fixed ten-seed schedule,
one checkpoint, and one environment/controller contract. The 97.5% result
cannot be described as general VLA success, cross-task transfer, a safety
result, or a comparison against another policy.

## Recorded rollout evidence

Four additional successful episodes were recorded with the official
LeRobotDataset v3 writer:

| Suite / seed | Frames |
| --- | ---: |
| `libero_spatial` / 1000 | 77 |
| `libero_spatial` / 1001 | 74 |
| `libero_object` / 1000 | 135 |
| `libero_object` / 1001 | 135 |
| **Total** | **421** |

Both image streams, the eight-dimensional state, the seven-dimensional
executed action, reward, success, and done fields were persisted. All four
datasets reloaded, all 421 frames were iterated, all actions were finite, and
the checkpoint/processor identity matched. Raw videos and datasets remain on
the remote run volume and are not in Git.

All four recorded episodes are successes. They are sufficient to validate
serialization and the success-side world-model tensor path, but they cannot
support a failure classifier, failure calibration, success/failure
separation, or an online safety claim.

## Execution history and preserved failures

Every source change was made locally, validated, committed, pushed, and then
fast-forwarded remotely. The remote GitHub HTTPS route timed out, so the
already-pushed commits were transported as hash-verified Git bundles and
fetched by Git; tracked source was never edited remotely.

Three failed gates remain recorded:

1. the first LIBERO import attempted interactive configuration;
2. the first recording wrapper expected obsolete `tasks.jsonl` instead of
   LeRobot v3 `tasks.parquet`;
3. the first video reload lacked the complete FFmpeg shared-library runtime.

Each gate stopped the pipeline. The first two were fixed locally and pushed.
The third was resolved by completing the remote FFmpeg 4.4 runtime. No failed
artifact was promoted.

## Isolation and release review

LG-R0 did not access the sealed PickCube seeds `900000..900099`, rewrite P0,
P0.1, or P0.2, import LangMani source, use RoboLab or ROS 2, or modify the
frozen release evidence. The server remains online and SSH-ready.

Primary evidence:

- `artifacts/lg_r0/remote_execution_audit.json`
- `artifacts/lg_r0/libero_40ep_evaluation.json`
- `artifacts/lg_r0/rollout_dataset_manifest.json`
- `artifacts/lg_r0/rollout_dataset_validation.json`
- `artifacts/lg_r0/environment_validation.json`
- `artifacts/lg_r0/model_structure.json`
- `artifacts/lg_r0/static_inference_validation.json`

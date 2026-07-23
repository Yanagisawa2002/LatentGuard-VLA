# LG-R0 external stack audit

## Audit boundary

This audit is source- and artifact-grounded. The reviewed LeRobot checkout is
detached and clean at tag `v0.6.0`; model and dataset downloads use full
revisions. Runtime paths, hosts, and timestamps do not participate in semantic
identity.

## LeRobot 0.6.0

- Tag commit:
  `30da8e687a6dfc617fcd94afc367ac7071c376ce`.
- Installation: an ordinary `lerobot-0.6.0` wheel built from that exact
  checkout, not an editable install.
- Python requirement: `>=3.12`.
- PyTorch constraint in project metadata: `>=2.7,<2.12`; the frozen lock
  resolves `2.11.0+cu128` on this Linux/CUDA environment.
- Transformers constraint: `>=5.4,<5.6`; the lock resolves `5.5.4`.
- Compatibility-relevant import locations used by LG-R0 are the 0.6 public
  modules: `lerobot.scripts.lerobot_rollout`,
  `lerobot.scripts.lerobot_eval`, `lerobot.processor`,
  `lerobot.envs.factory`, and `lerobot.datasets.lerobot_dataset`. LG-R0 does
  not preserve older private import paths or install a compatibility shim.
- The official `lerobot-rollout` implementation is
  `src/lerobot/scripts/lerobot_rollout.py`; the standard evaluation recorder
  is in `src/lerobot/scripts/lerobot_eval.py::rollout`.
- Processor persistence is
  `PolicyProcessorPipeline.save_pretrained` /
  `PolicyProcessorPipeline.from_pretrained`. VLA-JEPA publishes
  `policy_preprocessor.json`, `policy_postprocessor.json`, and their
  statistics tensors.
- Evaluation recording creates `LeRobotDataset` instances, adds every frame,
  calls `save_episode`, and finalizes the dataset. LG-R0 adds only an explicit
  nested-LIBERO-state flattening adapter because the generic 0.6 raw-frame
  helper does not traverse `robot_state`.
- The ACT padded-loss normalization is in
  `src/lerobot/policies/act/modeling_act.py`: `valid_mask =
  ~batch["action_is_pad"].unsqueeze(-1)`, followed by masked L1 reduction and
  normalization over valid entries. This source fact does not retroactively
  change frozen PickCube results.

Primary sources:
[LeRobot v0.6.0 tree](https://github.com/huggingface/lerobot/tree/30da8e687a6dfc617fcd94afc367ac7071c376ce),
[evaluation recorder](https://github.com/huggingface/lerobot/blob/30da8e687a6dfc617fcd94afc367ac7071c376ce/src/lerobot/scripts/lerobot_eval.py),
[ACT implementation](https://github.com/huggingface/lerobot/blob/30da8e687a6dfc617fcd94afc367ac7071c376ce/src/lerobot/policies/act/modeling_act.py).

## VLA-JEPA

- LeRobot module:
  `src/lerobot/policies/vla_jepa/`.
- Merge PR: `huggingface/lerobot#3568`; merge commit
  `2e9cd87bbdb93c23503f7eeca7317bd33027b279`.
- The reviewed v0.6 tree also contains the inference GPU-roundtrip refactor at
  `18eee1b477e37f1f57883ae768b5ef231f63fe49`.
- Official policy:
  `lerobot/VLA-JEPA-LIBERO@735d9f692981e286ade093b5046627eda876e5d0`.
- Nested Qwen:
  `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`.
- Nested V-JEPA2:
  `facebook/vjepa2-vitl-fpc64-256@b3c1679b7c34d3255ef3547f27c7b226aefab26f`.
- Policy configuration: two 224×224 RGB views, state dimension 8, action
  dimension 7, action chunk 7, four flow-matching inference steps, eight
  temporal video frames, world model enabled, world-model loss weight 0.1.
- The action head is `VLAJEPAActionHead`, a DiT-B flow-matching policy head.
- The world-model path uses the frozen V-JEPA2 video encoder and
  `ActionConditionedVideoPredictor`.
- Predictor inputs are shifted context V-JEPA visual tokens and Qwen special
  action-token hidden states. It does **not** receive the generated numeric
  seven-dimensional action chunk.
- Predictor output is a sequence of future V-JEPA visual tokens aligned to the
  shift-by-one target tokens. The public training loss is mean L1.
- The world model is enabled in the checkpoint config, but normal
  `predict_action` calls Qwen with `need_action_tokens=False` and runs only the
  action head. Thus the world-model branch is not part of default inference.
- The public inference API exposes neither future latents nor candidate scores.
  It offers no arbitrary numeric candidate-action conditioning and no native
  multi-candidate scorer. Setting `enable_world_model=false` constructs a
  policy without the video encoder/predictor path.
- No native risk, success, progress, or reconstruction score exists. The only
  scalar is the training-time L1 world-model loss.
- The model card's 96.5 figure describes the upstream authors' full
  400-episode protocol. It is not an LG-R0 reproduction claim.

Primary sources:
[VLA-JEPA PR](https://github.com/huggingface/lerobot/pull/3568),
[v0.6 VLA-JEPA source](https://github.com/huggingface/lerobot/tree/30da8e687a6dfc617fcd94afc367ac7071c376ce/src/lerobot/policies/vla_jepa),
[official checkpoint](https://huggingface.co/lerobot/VLA-JEPA-LIBERO/tree/735d9f692981e286ade093b5046627eda876e5d0),
[official LeRobot documentation](https://huggingface.co/docs/lerobot/v0.6.0/vla_jepa).

## LIBERO

- Package: `hf-libero==0.1.4`, source commit
  `8561c60eea2fb93096146f240194649df73d8b1e`.
- Assets:
  `lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`.
- Audited suites: `libero_spatial`, `libero_object`, `libero_goal`, and
  `libero_10`, each with ten tasks.
- Raw observations include agent view, wrist view, end-effector pose, gripper
  state, and joint state. The official processor produces two image tensors and
  an eight-dimensional state: position 3 + quaternion-to-axis-angle 3 +
  gripper qpos 2.
- The action is seven-dimensional relative OSC pose control plus gripper, with
  environment bounds `[-1,1]`.
- Default LeRobot horizons are 280 (spatial/object), 300 (goal), and 520
  (`libero_10`).
- Success is the native `check_success()` result. The wrapper terminates on
  native `done` or success and reports horizon exhaustion separately.
- The seed initializes the environment RNG. With official init states enabled,
  reset also advances the deterministic init-state ordinal.
- Cameras are `agentview_image` and `robot0_eye_in_hand_image`, mapped to
  `observation.images.image` and `.image2`.
- Evaluation uses the official LeRobot environment factory and rollout loop.

Primary sources:
[LeRobot LIBERO adapter](https://github.com/huggingface/lerobot/blob/30da8e687a6dfc617fcd94afc367ac7071c376ce/src/lerobot/envs/libero.py),
[hf-libero source](https://github.com/huggingface/LIBERO/tree/8561c60eea2fb93096146f240194649df73d8b1e).

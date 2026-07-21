# WM-v0 D2 Policy Asset Audit

## Decision

**Result B — no accepted compatible PickCube-v1 policy package was recovered.**

The audit found one complete and previously exercised ACT package plus five
additional ACT registry declarations. The complete package is valid only for
`LangMani-PickPlaceByInstruction-v0` and is not compatible with the current
`PickCube-v1` contract. The other declarations do not have accessible checkpoint,
model configuration, processor, normalization, and runtime-evidence files.

The registry therefore contains:

- `ACCEPTED_COMPATIBLE`: 0
- `INCOMPATIBLE_ENVIRONMENT`: 1
- `INCOMPLETE_ASSET`: 5

## Audited roots

- LatentGuard tracked source, historical reports, release manifests, and ignored
  local model-output inventories
- LangMani tracked source, controller registry, phase reports, Git branches, and
  ignored local model-output inventories
- Remote LatentGuard and LangMani execution checkouts and model/run roots
- Available Git metadata and locally materialized release/tag records

The remote audit found the expected source-only LangMani environment and no
controller checkpoint, processor, or normalization package. It did not modify
remote tracked source or launch a simulator, VLM, LangMani, training, or GPU job.

## Complete package discovered

| Field | Audited value |
| --- | --- |
| policy_name | LangMani ACT per-task red-left accepted |
| policy_family | ACT |
| source_repository | `Yanagisawa2002/LangMani` |
| source_commit | `ceb73db1a0fe88fa7f58f347b51c542aa76663a0` |
| checkpoint_path | `pretrained_model/model.safetensors` within the checkpoint package |
| checkpoint_hash | `sha256:56a30fda4c0157946fcdad990bad5157f7c2a317c7005aeba9fa36b805608301` |
| package/checkpoint identity | `sha256:01cbd124005e12c04acc67258a48f19b9327364ab78fa7954a359f03152853d9` |
| training_config | `configs/langmani_v2/policies/act_per_task_red_left_v0.json` |
| dataset_reference | LangMani-v2 red-left task dataset |
| observation_spec | `observation.images.base_camera` float32 `[3,256,256]`; `observation.state` float32 `[9]` |
| action_spec | float32 8D `pd_joint_pos`, 20 Hz, chunk 50, execute 10, MEAN_STD |
| robot_embodiment | Panda |
| environment_id | `LangMani-PickPlaceByInstruction-v0` |
| environment_version | ManiSkill 3.0.1, SAPIEN 3.0.3 |
| preprocessor | `policy_preprocessor.json`, hash `sha256:eaff91dc5004c22473578eecd7cbe14566a0ab1d5c63f9edc41d3cc4d0bc270b` |
| postprocessor | `policy_postprocessor.json`, hash `sha256:ad2d49a0f67981d2e0a1bee69d85a816654b3159365829093227baa46a74d8bb` |
| normalization_statistics | normalizer state, hash `sha256:5cbdf752bf5ac112b1e8188a3bb53964852fa2c6795e644476e96cd70e68758c` |
| action_horizon | 50 (10 actions executed per query) |
| known_evaluation_result | native RTX 5090 run: 3/3 success, 40 policy queries, 392 simulator steps |
| acceptance_evidence | valid only for the native LangMani red-cube-to-left-bin task |
| compatibility_status | `INCOMPATIBLE_ENVIRONMENT` |
| missing_assets | none for its native environment |

The model configuration hash is
`sha256:7f09ff69ef42326a6ab5c79f0785ebd40cd9b976fa69e2596e82f5742c004621`.
The postprocessor state hash is
`sha256:c043c76eab4e1d2ee4040c8892923c1aae0a0c47b4c252a8fb1d1118778bf781`.

## Why the complete package is rejected for D2

The D1 compatibility identity binds `PickCube-v1`, Panda, `pd_joint_pos`, an
8D float32 action, 20 Hz control, ManiSkill 3.0.1, SAPIEN 3.0.3, exact task and
controller source identities, and an exact-replay environment observation mode
of `none`. It does not bind a policy-input observation or action-ordering
semantic.

The ACT package instead binds a base-camera RGB plus 9D state input and the
LangMani instruction task `red_cube:left_bin`. Matching Panda, 8D, 20 Hz, and
`pd_joint_pos` does not prove identical observation features, gripper convention,
action ordering, task termination, or intended target. Using it would require an
unauthorized environment/adapter change and would make prior native evidence
irrelevant. It is therefore `INCOMPATIBLE_ENVIRONMENT`, not
`ACCEPTED_COMPATIBLE`.

## Incomplete registry declarations

Five per-task ACT declarations were found: red-right, green-left, green-right,
blue-left, and blue-right. Each records a checkpoint fingerprint, but the
audited local and remote roots do not contain the checkpoint bytes, model config,
preprocessor, postprocessor, normalization state, or an executable runtime
selection for that entry. Each is `INCOMPLETE_ASSET`; all are also declared for
the LangMani task family rather than PickCube-v1.

The exact declarations and fingerprints are retained in
`artifacts/wm_v0_d2/policy_registry.json` without machine-specific paths.

## Required recovery to unblock D2

Recover or produce a **PickCube-v1-native** package containing all of:

1. content-bound checkpoint and model config;
2. exact policy observation fields and preprocessing;
3. exact 8D action ordering, units, gripper convention, and postprocessing;
4. normalization statistics;
5. the D1 environment/task/controller compatibility identity;
6. a reproducible current-environment rollout fixture or evaluation result.

Only after that package passes byte verification, exact contract comparison,
determinism/seed smoke, and a real short PickCube rollout may a provider or pilot
be created. Training a new PickCube-native policy is a separate authorized
milestone, not an action taken here.

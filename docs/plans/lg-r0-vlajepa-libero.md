# LG-R0 VLA-JEPA LIBERO execution plan

## Objective

Freeze LeRobot 0.6.0, the official `lerobot/VLA-JEPA-LIBERO` checkpoint,
its Qwen3-VL and V-JEPA2 dependencies, and the LIBERO environment. Validate
strict local loading, official processor persistence, static inference, a
bounded closed-loop smoke baseline, official-format rollout recording, and the
real tensor surface of the included world-model training path.

## Scope and exclusions

LG-R0 is an inference and interface-audit milestone. It creates no optimizer,
calls no backward pass, trains no parameter or head, and performs no policy
intervention. It does not use PickCube data, synthetic action candidates,
RoboLab, Isaac, ROS 2, LangMani source, or the sealed PickCube final seeds.

The repository release freeze remains authoritative. P0, P0.1, and P0.2 are
historical frozen results; this branch does not reinterpret or rewrite them.

## Fixed identities

- LatentGuard base: `121ae7f9f8858143cf750a81943d50458006ac0f`
- LeRobot: `0.6.0` / `30da8e687a6dfc617fcd94afc367ac7071c376ce`
- VLA-JEPA merge: `2e9cd87bbdb93c23503f7eeca7317bd33027b279`
- VLA-JEPA inference refactor:
  `18eee1b477e37f1f57883ae768b5ef231f63fe49`
- VLA-JEPA checkpoint:
  `lerobot/VLA-JEPA-LIBERO@735d9f692981e286ade093b5046627eda876e5d0`
- Qwen:
  `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`
- V-JEPA2:
  `facebook/vjepa2-vitl-fpc64-256@b3c1679b7c34d3255ef3547f27c7b226aefab26f`
- LIBERO: `hf-libero==0.1.4` /
  `8561c60eea2fb93096146f240194649df73d8b1e`

## Gates

1. Source, environment, checkpoint, processor, and nested-backbone identities
   match the reviewed manifests.
2. Strict model load has no missing or unexpected keys.
3. Single and batch static inference are finite and reproducible under the
   frozen seed; official processor save/load is output-equivalent.
4. One fixed `libero_spatial` task/seed completes without environment,
   processor, checkpoint, action-contract, or non-finite-action errors.
5. Only after gate 4, run the frozen 40-episode schedule: task 0 from each of
   four official suites, ten fixed episode seeds per suite. This is a smoke
   baseline and is not equivalent to the official 400-episode evaluation.
6. Record at least four episodes spanning two suites in LeRobot 0.6 format;
   validate metadata, reload, and frame iteration.
7. Audit real world-model tensors from available success and failure
   trajectories. Report sample shortfalls. Do not fit a classifier or threshold.
8. Run a real-candidate probe only if source and runtime both expose native
   external candidate conditioning. Otherwise emit the explicit unsupported
   result.

## Result taxonomy

- Result A requires every requested policy, rollout, and world-model condition,
  including real action response, to pass.
- Result B applies when the baseline runs but the world model remains a
  training-only representation path, lacks arbitrary numeric candidate
  conditioning, or another requested interface is only partially available.
- Result C applies when the exact frozen stack cannot complete reliable strict
  loading or closed-loop execution.

## Evidence and storage

Tracked evidence is compact JSON and Markdown under `artifacts/lg_r0` and
`docs`. Model weights, simulator assets, videos, and LeRobot datasets stay
outside Git under an explicit run root. Every compact report binds the exact
config, revisions, file hashes, environment, command, and zero
optimizer/backward counters.

The user subsequently authorized the powered-on GPU server for LG-R0
inference, simulator rollout, recording, and tensor audit. The local checkout
remains the only source of tracked files. Remote work may begin only after an
execution-ready commit is pushed and the remote checkout is clean and matches
that exact SHA. All remote outputs are written outside the remote checkout and
only compact reviewed evidence returns to this repository.

## Release and isolation review

The plan does not access PickCube final seeds, modify accepted release
artifacts, start a training job, or change frozen model weights. External
packages are installed only in the dedicated Linux environment. LangMani paths
and repositories are excluded from source and Git checks. SSH is limited to
the exact-revision LG-R0 execution lifecycle; RoboLab, Isaac, ROS 2, LangMani,
optimizer steps, backward passes, and any form of training remain prohibited.

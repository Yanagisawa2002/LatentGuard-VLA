# LG-RB0 RoboLab stack audit

## Decision scope

LG-RB0 asks only whether a frozen RoboLab runtime can provide deterministic,
faithful replay and a safe deterministic takeover from replayed intermediate
states. It does not integrate a VLA, define policy candidates, rank actions,
train a model, or authorize intervention.

The exact external source is NVIDIA's official RoboLab repository, release
`v0.2.1`, peeled commit
`0aef241fb088ca21bb4ebd24448940ed56620d17`. Floating `main` is prohibited.
RoboLab is an external execution dependency and is not copied into LatentGuard.

## Official capability audit

The frozen package metadata specifies Python 3.11 and an `isaac51` dependency
set with Isaac Sim `5.1.0` and Isaac Lab `2.3.2.post1`. The official replay
example:

- restores an HDF5 episode's recorded initial scene state;
- loads the actions and the adjacent recorded environment configuration;
- recommends `num_envs=1` for faithful replay;
- optionally compares every post-step scene state to the HDF5 `states` tree;
- uses maximum absolute error with default tolerance `0.01`.

The official restore overlays recorded leaves onto the current scene and warns
when a recorded path is absent. It does not prove that controller internals,
random-number state, subtask/event state machines, termination latches, or
observations are restored. The official validator also skips current-state
paths that are absent. LG-RB0 therefore treats official full replay as the
first gate, then adds a stricter complete-tree and semantic-repeatability gate
for prefix takeover and fixed suffixes.

The release includes one small example recording for
`RubiksCubeAndBananaTask`. It is insufficient for the required ten-episode
gate, so LG-RB0 preregisters ten new mechanics-only recordings: five fixed
seeds for `BananaInBowlTask` and five for `RubiksCubeAndBananaTask`.

## Registered task contracts

Both tasks use the RoboLab DROID registration with a Franka Panda arm and
Robotiq 2F-85 gripper. The action contract is seven Panda joint-position
commands plus the legal binary gripper joint command. Physics runs at
`1/120 s` with decimation 8, for a `1/15 s` control interval, on CUDA with
Fabric.

`BananaInBowlTask`:

- instruction: pick up the banana and place it in the bowl;
- observations: over-shoulder and wrist RGB groups, proprioception
  (arm/gripper joints and end-effector poses), and egocentric viewport RGB;
- cameras: `over_shoulder_left_camera`, `wrist_cam`, and
  `egocentric_mirrored_camera`;
- registered horizon: 50 seconds;
- subtask: banana grabbed, then banana in bowl with gripper detached;
- success: banana in bowl with required contact and detached gripper.

`RubiksCubeAndBananaTask`:

- instruction: put the cube and the banana in the bowl;
- robot, observations, action, cameras, and simulation settings match the
  simple task;
- registered horizon: 60 seconds;
- subtask: both Rubik's cube and banana must each progress from grabbed to
  in-bowl with detached gripper;
- success: both objects in the bowl with required contact and detached gripper.

LG-RB0 records only the first 40 registered control actions from its fixed
mechanics probe. It does not shorten the official task horizon to manufacture
task failure or use success rate to select tasks.

## Resource audit

The available server has an NVIDIA GeForce RTX 5090 with about 32 GiB of GPU
memory. RoboLab recommends 48 GiB or more. LG-RB0 keeps the official
single-environment faithful-replay requirement and uses only two simple,
headless tasks, but the lower memory capacity remains a hard observed
limitation: an out-of-memory error stops the affected gate and is not repaired
by silently shrinking the registered protocol.

The system volume is too small for the stack. The isolated environment,
external checkout, caches, assets, recordings, and run outputs must live under
`/root/autodl-tmp/latentguard-lg-rb0`. Downloads use the Aliyun mirror by
default after `source /etc/network_turbo`; pinned NVIDIA/PyTorch indexes are
allowed only when required by the frozen stack.

## Compatibility decision before execution

The official APIs provide a plausible path to initial restore and full replay.
Intermediate takeover is not an official claim and has no direct snapshot
restore API in the audited release. LG-RB0 will therefore reconstruct an
anchor by restoring the recorded initial state in a fresh session and replaying
the exact recorded prefix. Only zero-mismatch evidence across the registered
prefix, fixed-suffix, and isolation checks can establish takeover feasibility.

Sources:

- [NVLabs RoboLab](https://github.com/NVLabs/RoboLab)
- [RoboLab replay guide](https://github.com/NVLabs/RoboLab/blob/v0.2.1/docs/replay.md)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)

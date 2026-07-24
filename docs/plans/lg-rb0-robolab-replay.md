# LG-RB0 RoboLab replay milestone plan

## Purpose

LG-RB0 is a bounded feasibility gate for using the frozen RoboLab replay stack
as a future counterfactual mechanics platform. It asks whether an episode can
be replayed faithfully from its recorded initial state to deterministic
intermediate anchors and whether fixed legal suffixes remain repeatable and
isolated. It does not evaluate or improve a robot policy.

The milestone starts from LatentGuard commit
`ea9cebb76cd3aa44005ec2fb9a9dd89208fefda7` and preserves LG-R2b0 Result C as
frozen historical evidence. The external source is RoboLab `v0.2.1`, commit
`0aef241fb088ca21bb4ebd24448940ed56620d17`.

## Frozen scope

- Python 3.11, Isaac Sim 5.1, Isaac Lab 2.3.2.post1;
- one headless CUDA environment per Isaac Sim process;
- `BananaInBowlTask` and `RubiksCubeAndBananaTask`;
- five registered seeds per task and 40 actions per recording;
- three faithful replays per recording at official tolerance `0.01`;
- early, middle, and late anchors;
- three repeats for fixed A/B suffixes at steps 1, 5, and 10;
- A-B-A and B-A-B isolation orders;
- strict takeover tolerance `1e-6` and complete semantic coverage.

The ten recordings, faithful replays, and takeover evaluations are sharded by
recording so renderer teardown cannot leak across episodes. A CPU-only
orchestrator merges a phase only after all ten one-recording shards exist and
pass their structural checks.

## Execution gates

1. Validate both clean, exact Git checkouts and the frozen package/GPU stack.
2. Pass a one-step smoke on both registered tasks.
3. Generate exactly ten valid deterministic mechanics recordings.
4. Run 30 faithful full replays. Any failure yields Result C and stops.
5. If faithful replay passes, run all prefix, branch, and isolation repeats.
6. Aggregate immutable artifacts into Result A, B, or C.
7. Retrieve only compact JSON evidence; keep HDF5, simulator assets, logs,
   caches, and runtime state outside Git.
8. Complete local CPU validation, evidence review, commit, and push.

## Stop and promotion rules

- Result A requires ten valid recordings, zero faithful/prefix/branch/isolation
  mismatch, and complete semantics. It authorizes only LG-RB1 design.
- Result B means faithful replay passed but deterministic takeover did not. It
  prohibits a candidate pilot.
- Result C means official faithful replay failed under the frozen stack. It
  stops the RoboLab route.

No post-result change to tasks, seeds, actions, anchors, repetitions, state
tolerances, or semantic fields is permitted within this evidence set.

## Exclusions and isolation

This milestone performs no training, backward pass, optimizer step, checkpoint
generation, VLA/policy integration, candidate sampling or ranking, online
intervention, PointWorld work, ROS 2 work, LangMani access, or final-seed
access. Seeds `900000..900099` remain sealed. Raw RoboLab and Isaac artifacts
remain outside Git.

## Cost and ETA

The target is one RTX 5090 with about 32 GiB memory, below RoboLab's recommended
48 GiB. The fixed single-environment protocol must stop on OOM rather than
silently shrink.

Expected remote GPU time is approximately 15 minutes for recordings,
20--35 minutes for faithful replay, and 60--120 minutes for takeover if
authorized, plus probes and setup. Total wall time is expected to be roughly
2--3 hours after the environment is available. These are operational
estimates, not research results; actual run durations are recorded in the
remote logs.

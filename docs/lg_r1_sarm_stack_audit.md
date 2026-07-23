# LG-R1 SARM external-stack audit

## Preliminary verdict

SARM is part of the official LeRobot 0.6.0 release used by LG-R0. The frozen
source is `src/lerobot/rewards/sarm` at LeRobot commit
`30da8e687a6dfc617fcd94afc367ac7071c376ce`; LG-R1 therefore does not need to
follow post-release `main`.

The implementation is a video-and-language reward model with a stage
transformer and a stage-conditioned within-stage progress transformer. Its
processor uses `openai/clip-vit-base-patch32`, includes robot state, and
supports `single_stage`, `dense_only`, and `dual` annotation modes. The
official default samples a bidirectional nine-frame observation sequence with
four rewind placeholders for augmentation.

LG-R1 freezes the CLIP repository at
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.

The implementation emits stage probabilities, within-stage progress, and a
normalized dense progress/reward. It does not emit an action-conditioned
progress delta or accept VLA-JEPA numeric action candidates.

## Checkpoint boundary

The LeRobot guide documents training SARM on a user dataset and loading the
resulting reward-model path. No official model card has been identified that
claims a zero-shot checkpoint compatible with LIBERO, the LG-R1 stage schema,
and the frozen two-camera/state contract. Community SARM checkpoints are not
treated as official or compatible.

LG-R1 will therefore use the official LeRobot SARM modules with frozen CLIP
features and a deliberately small stage/progress configuration. The result
will be called a **SARM-style small baseline**, not an official pretrained
SARM reproduction. This decision is revisited only if an exact, source-backed
official compatible checkpoint is found before training begins.

## Known supervision mismatch

Official SARM derives `stage + tau` targets from natural-language subtask
interval annotations and dataset-level temporal proportions. LG-R1 instead
requires simulator-grounded task stages that can regress and that do not
backfill terminal outcomes. The model architecture can consume these targets,
but the label generator and evaluation protocol are LatentGuard-owned
adapters. This distinction must remain visible in every report.

## Sources

- [LeRobot SARM documentation](https://huggingface.co/docs/lerobot/sarm)
- [LeRobot v0.6.0 release](https://huggingface.co/blog/lerobot-release-v060)
- [Frozen SARM source directory](https://github.com/huggingface/lerobot/tree/30da8e687a6dfc617fcd94afc367ac7071c376ce/src/lerobot/rewards/sarm)
- [SARM paper and project](https://qianzhong-chen.github.io/sarm.github.io/)

The final manifest will additionally bind file hashes, resolved package
versions, CLIP revision, processor serialization, measured memory, and license.

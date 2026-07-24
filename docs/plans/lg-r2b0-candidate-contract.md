# LG-R2b0 real candidate contract

Each candidate is a new native inference sample from the frozen
`lerobot/VLA-JEPA-LIBERO` checkpoint revision
`735d9f692981e286ade093b5046627eda876e5d0`, using the pinned LeRobot action
head and official processor/postprocessor. It is not an edited source action.

The immutable identity tuple is:

`anchor_id, observation_sha256, proprioception_sha256, policy_id,
checkpoint_revision, processor_revision, inference_seed, sampling_config,
normalized_action_chunk, native_action_chunk, action_mask, content_sha256`.

Generation resets the policy queue before every sample. The anchor uses one
current observation because the frozen policy contract has
`n_obs_steps=1`. The source audit proves fixed-seed reproducibility and
different-seed variation before any branch is executed.

The action chunk must be finite, shape `[7, 7]`, and native values must remain
within `[-1, 1]`. Source kind must be `native_policy_sampling`; any synthetic,
optimized, transferred, relabeled, or outcome-conditioned source is rejected.

Candidate relations are registered before results:

- exact duplicate: byte-identical float32 native chunk;
- near duplicate: not exact but below every controller-aware meaningful
  threshold;
- meaningfully distinct: full-chunk L2 at least 0.10, endpoint translation
  difference at least 0.03, cumulative rotation difference at least 0.10, or
  gripper disagreement above 0.50.

The near full-chunk L2 threshold is 0.01. These values are tied to the frozen
seven-step native OSC action contract and its observed translation, rotation,
and binary-gripper ranges. They are not tuned from candidate outcomes.

Four candidates are targeted, three are the minimum valid distinct set, and
eight native sampling attempts are the maximum. All attempts remain visible in
the diversity audit. Failure to obtain the required real set does not authorize
a synthetic fallback.

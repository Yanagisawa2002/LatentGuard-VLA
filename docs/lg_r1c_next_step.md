# LG-R1c next step

## Decision after Result B

Stop the direct reward-baseline promotion path. No LG-R1c model is authorized
as an LG-R2 reward baseline, and scaling TOPReward, fine-tuning either
foundation model, fitting a failure head, or starting intervention rollouts is
not justified by this milestone.

The strongest retained evidence is ROBOMETER zero-shot as a **post-hoc
diagnostic or auxiliary target**. It is not an online selector because its
video window must already have been executed.

## Narrowest defensible follow-up

If a later milestone authorizes LG-R2 design work, its first falsifiable
question should be whether the existing current VLA representation plus an
explicit numeric candidate action chunk contains candidate-conditioned
predictive signal. This is feature set A in
`artifacts/lg_r1c/lg_r2_feature_contract.json`.

Before any training, that milestone must:

1. define an adapter that proves arbitrary numeric candidate conditioning
   rather than policy-produced token conditioning;
2. use only development data and keep task-held-out test and final seeds
   sealed;
3. pre-register a direct non-learned candidate baseline, validation-only
   selection, checkpoint gate, and natural-failure operating points;
4. prohibit ROBOMETER/TOPReward video outputs from direct candidate-time
   inputs unless a separately executed trajectory exists;
5. stop before simulator intervention if task-macro, rare-failure precision,
   or candidate-conditioning sanity checks fail.

Feature set B may add frozen post-hoc reward targets for auxiliary supervision,
but only after feature set A passes. This ordering isolates whether reward
distillation contributes information instead of hiding a weak
candidate-conditioned representation.

## Current authorization

This document is a handoff, not approval to train. No new data collection,
simulator rollout, checkpoint creation, failure-head training, policy
modification, intervention, or final-seed access is authorized. The server
remains online and idle pending an explicit next instruction.

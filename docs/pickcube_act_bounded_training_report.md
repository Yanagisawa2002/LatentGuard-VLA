# PickCube-v1 bounded ACT training report

## Outcome

WM-v0 P0.1 completed its authorized 20,000-step training run on one RTX 5090.
The run is structurally valid and independently verified: its dataset,
checkpoint, action-transform, normalization, and source identities are bound;
all post-transform actions are finite and strictly inside the native action
bounds; save/load and exact zero-work resume pass. This report is training
evidence, not a closed-loop acceptance claim.

The execution producer is
`48069c2c3bdb460220a71480d8c75b9393fc9eab` on branch
`codex/wm-v0-p0-pickcube-act`. The authoritative training root remains outside
Git under run ID:

```text
20260721T205500Z_wm-v0-p01-act-formal_48069c2_seed0
```

## Controlled P0 versus P0.1 change

P0.1 reuses the exact P0 dataset, 400/50/50 split, model inputs, ACT
architecture, seed, optimizer, batch size, learning rates, 16-action prediction
horizon, four-action execution horizon, 20,000-step budget, controller, and
development/final seed locks. The only policy-semantic change is the versioned
`affine_tanh_v1` action parameterization.

The decoder emits unconstrained raw values `r`. The shared transform computes
`y = tanh(r)` in float32 and then maps each component to its native interval
with `a = center + scale * y`. Training loss is evaluated on the squashed
canonical action and the legal dataset target. No clipping, repair, fallback,
or relaxed action interval is used. The checkpoint schema is
`pickcube_act_bounded_v1`; the transform digest is
`sha256:e74d814de59b6aeb505e2eebc3fc6263df77aa89aa5267817aed3224a72e941d`.

Temporal aggregation is disabled. Receding-horizon execution consumes the
first four actions of each predicted 16-action chunk, and the queue holds the
already transformed native actions. The old unbounded P0 checkpoints remain
incompatible with this execution path.

## Dataset and target audit

The content-bound dataset is unchanged:

- 500 successful episodes and 21,053 frames;
- dataset digest
  `sha256:234b82709359f1f2fde513ab26c015f4ebfb262a16b1f16206c54eddbde44800`;
- normalization digest
  `sha256:8093f5bba3916dff218e73789739b18e05e9eb96cfc200fa18747f010959532a`;
- zero native target boundary violations;
- no exact arm-boundary targets;
- gripper targets are exactly `-1` for 10,735 frames (50.9904%) and exactly
  `+1` for 10,318 frames (49.0096%).

The gripper is therefore retained as the environment's continuous normalized
mimic-joint position target. The dataset happens to contain only endpoint
commands, but the installed controller contract is a continuous Box dimension;
P0.1 does not introduce an unsupported binary head. Endpoint targets remain
legal, while finite model outputs map strictly inside the native interval.

## Pre-training gates

The following gates passed before the formal run:

- arbitrary finite raw values, including extreme magnitudes, map to strict
  interior actions on CUDA;
- NaN and infinity fail closed;
- gradients remain finite;
- all 21,053 real targets are finite and legal;
- 10,007 CUDA property samples produced zero boundary violations;
- the 10,000-query real-observation end-to-end screen produced zero action
  violations and zero non-finite outputs;
- corrected tiny overfit completed 500 steps and reduced the action-loss window
  from `0.28568257` to `0.16577877` with zero violations;
- GPU smoke, checkpoint reload, and resume contracts passed.

Two invalid infrastructure attempts are preserved rather than rewritten. One
exposed BF16 endpoint rounding before the transform was forced to float32. The
other exposed a pre-formal early-stop/finalization contract that could stop
before 20,000 steps. Both defects were fixed locally, tested, committed, and
then pulled on the server before the authoritative run.

## Formal run

| Field | Observed value |
| --- | --- |
| Training steps | 20,000 / 20,000 |
| Early stopped | false |
| Interrupted | false |
| Examples processed | 638,853 |
| Retained checkpoints | 15 |
| Best validation step | 11,000 |
| Best validation loss | 0.1025937595263575 |
| Early role | step 1,000 |
| Mid role | step 10,000 |
| Best-validation role | step 11,000 |
| Final role | step 20,000 |
| Approximate wall time | 2,081 seconds (34 minutes 41 seconds) |
| Peak GPU allocated | 1,813,391,872 bytes |
| Peak GPU reserved | 2,116,026,368 bytes |
| Post-transform violations | 0 |

The independent training verifier recomputed an initial action-loss mean of
`0.27487349` and a final mean of `0.10443416`, found zero post-transform
violations, and emitted evidence digest
`sha256:bb856d1410febef4ca353dd67fe2a795125c263565e6ad113ea2e9599a07874c`.
An exact resume from the complete final checkpoint performed zero training work
and passed.

Validation reconstruction loss improved, but it did not predict closed-loop
competence. Static and physical results are reported separately in
[the bounded evaluation report](pickcube_act_bounded_evaluation_report.md).

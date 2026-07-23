# LG-R0 action padding audit

## Findings

VLA-JEPA training requests action deltas `0..6`, so samples near an episode end
may contain padded action chunks. Dataset sampling emits `action_is_pad`; the
VLA-JEPA action loss passes it to the action head, which excludes padded
timesteps from the flow-matching loss normalization.

LeRobot 0.6.0 also contains the reviewed ACT fix in
`src/lerobot/policies/act/modeling_act.py`: an expanded valid mask excludes
padded action entries before L1 reduction. That fix is relevant to training
loss semantics, not VLA-JEPA inference.

At inference, VLA-JEPA always emits a full seven-step chunk and no action mask.
The action queue executes at most `n_action_steps=7`; an environment
termination can leave the rest of the queue unexecuted. LG-R0 therefore records:

- generated full chunk;
- each executed action;
- explicit `action_mask=null`;
- effective episode and executed horizon;
- no fabricated `action_is_pad`.

The official LeRobot rollout dataset records executed per-step actions and
episode boundaries. It does not store the unexecuted tail as padded rollout
actions. Dataset consumers derive future-chunk padding from episode metadata.

## Historical PickCube impact

Frozen PickCube P0/P0.1/P0.2 did not train through this LeRobot 0.6 VLA-JEPA
stack. Their accepted evidence, including the negative Result C, is not
rewritten by this source audit. Padding semantics alone provide no causal
evidence that would require rerunning old ACT experiments.

```text
RETRAIN_NOT_REQUIRED
```

This conclusion means “no evidence from this audit requires automatic
retraining.” It is not a positive claim about old policy quality.

# LG-R0 frozen next-baseline definitions

## Baseline A — terminal outcome classifier

Input: current latent plus generated numeric action chunk. Target: terminal
success/failure. Splits must group by source rollout/task/init state, and no
future outcome may enter deployable inputs. This baseline is not implemented in
LG-R0.

## Baseline B — native VLA-JEPA world-model representation

No learned head. Report official current/predicted V-JEPA tokens and available
prediction/target descriptive distances. Because target future tokens are not
available online and the predictor does not accept external numeric candidates,
this is a representation/diagnostic baseline, not a deployable risk score.

## Model C — VLA-JEPA plus LatentGuard failure head

Proposed inputs:

```text
current V-JEPA tokens
+ predicted future V-JEPA tokens
+ Qwen action-token representation
+ generated numeric action chunk
```

Targets may include failure events, progress, and terminal outcome, with
explicit evidence strength. Future target latents are training-only targets.
No part of Model C is implemented or trained in LG-R0.

## Model D — Model C plus uncertainty

Candidate methods for a later milestone are ensemble variance, MC dropout, or
a heteroscedastic head. The method, seed policy, calibration split, and
promotion threshold remain undecided. LG-R0 fits none of them.

## Model E — selective intervention

Intervention is prohibited until an untouched offline ranking experiment,
calibration, coverage/risk reporting, and a separately frozen promotion gate
pass. LG-R0 performs zero interventions.

## LG-R0 selection

LG-R0 makes Baseline A and Model C the minimum paired LG-R1 experiment. Model C
is preferred only as a hypothesis: it must demonstrate validation improvement
over Baseline A before promotion. The native representation-only Baseline B
remains a diagnostic and cannot rank external numeric candidates.

LG-R1 training is currently data-blocked. The accepted LG-R0 recordings contain
421 frames from four successful trajectories and zero failed trajectories.
They validate mechanics but cannot fit or select a failure head.

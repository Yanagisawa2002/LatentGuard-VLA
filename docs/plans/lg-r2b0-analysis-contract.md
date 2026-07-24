# LG-R2b0 within-anchor analysis contract

The unit of comparison is one restored anchor. Candidate outcomes from
different anchors are never treated as counterfactual pairs and no predictive
model is fit.

For each anchor, the report includes candidate action distances, 7/21/49-step
progress ranges, object-pose ranges, stage-transition disagreement, local-event
disagreement, terminal-outcome composition, and continuation recoverability.
Pairwise rank stability compares the same candidates across registered
horizons. Ties use epsilon 0.02.

Permitted progress evidence is descriptive:

- action chunks are genuinely different at an identical state;
- progress, stage, event, or terminal labels differ within that state;
- the differences occur beyond task-id 6;
- target rankings are not dominated by ties.

The following cannot be called task success:

- candidate-source diversity by itself;
- exact restoration by itself;
- action-distance spread;
- progress or object-motion spread;
- stage transition, contact, grasp, drop, or stagnation disagreement;
- continuation recoverability;
- Result A or LG-R2b1 authorization.

Only a branch with conclusive simulator success is a successful branch. Result
A means the bounded counterfactual targets are identifiable enough to review a
later ranking experiment; it does not claim a good ranker, improved policy,
safer behavior, intervention benefit, generalization, or real-robot validity.

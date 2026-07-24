# LG-R2b0 collection report

Status: **not run because the state-restoration hard gate failed**.

The registered target is 60 independent anchors, 3-4 real native-policy
candidates per valid anchor, at least 150 accepted branches, and preferably
240. The task schedule, stage priorities, candidate thresholds, continuation
rule, and stop gates are already frozen under `configs/lg_r2b0/`.

The real candidate-source audit passed, but the pre-registered 10-state,
five-repeat restoration audit did not. Per the frozen execution order, the
60-anchor registry, candidate-per-anchor generation, one-branch smoke, and
full counterfactual collection were not started.

Consequently anchor, candidate-diversity, branch-count, outcome-diversity, and
rank-stability metrics are unavailable. They must not be read as zero. The
compact placeholder artifacts explicitly carry `status: not_run`,
`availability: unavailable`, and `metric_values: null`. No raw counterfactual
dataset exists, and no training, ranking, online selection, or intervention
occurred.

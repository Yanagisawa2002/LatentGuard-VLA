# LG-R2a evaluation protocol

## Estimand

The primary estimand is the out-of-fold incremental predictive value of Model C
(current VLA representation plus explicit numeric executed action) over Model A
(current VLA representation only) on frozen real on-policy data.

This is associational evidence. The episodes contain only the action actually
chosen by the frozen policy at each state, so the protocol cannot identify a
same-state causal action effect.

## Five-fold grouped protocol

The assignment unit is both episode and seed. No episode, seed, or window
crosses folds. The deterministic assignment seed is 22031. Natural failures are
distributed before successful episodes; the objective then balances failure
taxonomy, task, and cached-window count. All five test folds must contain a
natural failure.

For outer fold `i`:

- test is fold `i`;
- validation is fold `(i+1) mod 5`;
- training is the other three folds.

Preprocessing, class weights, checkpoint selection, and F1 thresholds are fit
without the test fold. The selected checkpoint evaluates the test fold once.
The five test partitions form one complete out-of-fold prediction set.

## Reporting views

The primary view is pooled out-of-fold performance with per-fold consistency.
Additional views are:

- task-macro, with every task equally weighted;
- leave-LIBERO-10-task-6-out;
- failure-task macro over the four tasks with natural failures;
- original frozen split labels as a compatibility grouping only;
- exploratory held-out-failure-task only if support is interpretable.

The original split is not used as the sole LG-R2a conclusion. Because the
grouped-CV model may train on episodes from every historical split, its
split-stratified output is not represented as an untouched historical test.

## Metrics and uncertainty

Progress delta reports MAE, RMSE, Spearman, Kendall tau-b, and sign accuracy for
7, 21, and 49 steps. Binary targets report AUROC, AUPRC, Brier score,
10-bin expected calibration error, validation-threshold precision/recall/F1,
and a descriptive test-curve precision/FPR point at recall 0.5. The latter is
not a deployable threshold.

Incremental results include absolute and relative A-vs-C deltas, five-fold
direction consistency, and deterministic 2,000-resample episode bootstrap 95%
intervals. Accuracy alone is never a promotion metric.

## Leakage and test-consumption checks

The manifest must show zero episode and seed leakage. Prepared samples use only
the frozen current representation, current proprioception for D, seven real
executed numeric actions, and their mask. Actual remaining episode length,
future observations, outcome-derived fields, and privileged stages are
reporting/label-only and never model inputs.

Training stops if any fold lacks a natural failure, the feature/action join is
incomplete, the action contract fails, or any OOF sample is evaluated zero or
more than one time.

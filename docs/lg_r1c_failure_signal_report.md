# LG-R1c failure-signal report

## Main finding

ROBOMETER supplies a real but insufficient post-execution natural-failure
signal. Across all 320 frozen episodes, its zero-shot episode-failure AUPRC is
0.164 versus 0.062 for frozen SARM, with 27 positives at 8.44% prevalence. At
the pre-registered operating point it reaches 0.630 recall, 0.160 precision,
and 0.304 FPR. It therefore passes recall and FPR, but fails the required
AUPRC 0.20 and precision 0.25.

The validation-fitted early-warning threshold detects 26 of 27 failure
episodes with mean lead 122.2 frames, but also alarms on 23.5% of matched
success windows. That diagnostic is useful for error analysis; it is not a
safety or intervention result.

## Matched-window evidence

On the balanced 2,315 failure / 2,315 success matched-window set, ROBOMETER
zero-shot reaches AUPRC 0.625 and AUROC 0.694. On the 2,279 strictly matched
pairs it reaches AUPRC 0.628 and AUROC 0.700. These balanced-window numbers
must not replace the natural episode-level result: a 50% constructed
prevalence is much easier than the real 8.44% episode prevalence.

ROBOMETER's matched-window taxonomy results are:

| Failure taxonomy | Positive windows | AUPRC | AUROC | Precision at 60% recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| FAILED_PLACEMENT | 1,808 | 0.640 | 0.714 | 0.726 | 0.227 |
| OBJECT_DROP | 507 | 0.574 | 0.620 | 0.590 | 0.418 |

The weaker OBJECT_DROP result and its high FPR remain unresolved. The
progress-balanced matched data are also concentrated in `[0.5, 0.75)`; the
early and terminal strata are empty, so no broad stage-coverage claim is
available.

## Task and split sensitivity

Leaving LIBERO-10 task 6 out gives ROBOMETER zero-shot episode-failure AUPRC
0.222 over 280 episodes, but only five positives remain. On the original test
split there is one natural failure among 80 episodes: ROBOMETER AUPRC is
0.048 and the task-agnostic ensemble AUPRC is 0.067. These estimates are too
sparse to support a generalization claim, even when their FPR operating points
look better.

TOPReward zero-shot has episode-failure AUPRC 0.084, precision 0.109, recall
0.741, and FPR 0.556. Its calibrated test output collapses to a constant-like
operating point with FPR 1.0. The validation affine fit also operates on
scores near `1e-10`, producing a numerically large coefficient; this is a
scale-stability warning, not a reason to reinterpret the test result.

## What the metrics do and do not support

Supported:

- ROBOMETER contains post-execution visual evidence correlated with some
  natural failures and improves over frozen SARM on the full-set episode
  AUPRC/FPR comparison.
- FAILED_PLACEMENT is easier for ROBOMETER than OBJECT_DROP on the matched
  diagnostic.
- Calibration and the three-scalar ensemble do not solve the transfer gap.

Not supported:

- task success, online safety, causal intervention benefit, or recovery;
- reliable rare-failure detection at deployable precision;
- task-6-independent failure generalization from only five remaining
  positives;
- pre-execution scoring of an unseen numeric action candidate.

All unavailable or empty strata remain unavailable rather than being filled
with zeros.

# LG-R2a results

## Decision

LG-R2a is **Result B: local action signal, not stable incremental value**.
`LG_R2B_AUTHORIZED=false`. Candidate generation, same-state branching,
candidate ranking, planning, safety claims, and intervention remain
unauthorized.

The training revision was
`1f03f484c9f29e0a3dbfee767b56dbfd34795e3d`. The final post-hoc evaluation
revision was `f867f97c09e8785b91c1c94c18cae1b316795f59`; it reused the frozen OOF
predictions and did not retrain any model.

## Accepted data

All frozen source counts matched: 320 episodes, 8 tasks, 2 suites, 79,326
frames, 293 successes, 27 natural failures, 8,470 historical evaluation
windows, 4 failure-producing tasks, zero episode leakage, and zero seed
leakage. Of the 5,000 frozen representation-cache rows, 4,848 had seven complete
real executed actions and entered the probe. The other 152 tail rows were
excluded rather than padded.

The accepted probe labels contain 8 `OBJECT_DROP` windows, 31
`FAILED_PLACEMENT` windows, and 43 terminal-failure-within-49-step windows.
These very small supports limit failure conclusions.

The deterministic test-fold failure-episode counts are 5, 6, 6, 5, and 5.
Every sample is tested once after validation-only checkpoint and threshold
selection.

## Models and compute

| Model | Inputs | Parameters | Action encoder | Formal wall time | Peak allocated GPU |
|---|---|---:|---:|---:|---:|
| A | current frozen VLA representation | 302,378 | 0 | 20.0 s | 26.7 MB |
| B | real numeric action + mask | 11,628 | 5,856 | 21.1 s | 21.5 MB |
| C | current representation + action + mask | 311,276 | 5,856 | 24.6 s | 26.3 MB |
| D | C + current 8-D proprioception | 313,468 | 5,856 | 25.4 s | 26.4 MB |

C has 2.94% more parameters than the widened A control and remains within the
5% matching rule. All four models ran 400 optimizer steps per fold, 2,000 per
model and 8,000 total. The four formal run times sum to 91.2 seconds
(0.025 GPU-hours). Measured head-only, batch-amortized inference latency ranged
from 0.0011 to 0.0023 ms/sample; this excludes the frozen VLA feature extractor
and is not an end-to-end online latency claim.

## Progress delta

| Horizon | A MAE | C MAE | A Spearman | C Spearman | Interpretation |
|---|---:|---:|---:|---:|---|
| 7 steps | 0.040997 | 0.040597 | 0.4300 | 0.4700 | MAE -0.98%; Spearman +0.040 |
| 21 steps | 0.049542 | 0.050148 | 0.7270 | 0.7346 | MAE worsened 1.22% |
| 49 steps | 0.058918 | 0.061282 | 0.8336 | 0.8323 | both MAE and rank slightly worse |

The 7-step bootstrap interval for `C MAE - A MAE` is
`[-0.001643, +0.000892]`, which includes zero. The 7-step MAE direction is
positive in only 3/5 folds. Fold mean/std is 0.040995/0.000560 for A and
0.040592/0.002316 for C. The progress promotion threshold was 10% MAE
reduction or +0.10 Spearman; neither passed.

D has the best 7-step MAE, 0.038675 (5.66% below A), but only 0.4386 Spearman
and no pre-registered promotion role independent of C. It does not rescue the
failed primary gate.

## Binary targets

| Target | Support/prevalence | A AUPRC | C AUPRC | Delta |
|---|---:|---:|---:|---:|
| stagnation, 7 steps | 0.7816 | 0.9920 | 0.9900 | -0.0020 |
| regression, 7 steps | 108 / 0.0223 | 0.4372 | 0.4414 | +0.0042 |
| `OBJECT_DROP` | 8 / 0.00165 | 0.0193 | 0.0136 | -0.0057 |
| `FAILED_PLACEMENT` | 31 / 0.00639 | 0.0945 | 0.1471 | +0.0526 |
| terminal failure | 43 / 0.00887 | 0.1486 | 0.2029 | +0.0542 |

The classification increment gate passes through terminal failure, and failed
placement independently exceeds +0.05. Terminal improvement occurs in 4/5
folds and has an episode-bootstrap AUPRC-delta interval
`[+0.00069, +0.17831]`. However, terminal failure is secondary, sparse, and
does not compensate for the failed progress and robustness gates. Validation
threshold F1 is 0.322 for A and 0.304 for C on terminal failure; improved
ranking did not improve this selected operating point.

## Task balance and task 6

Terminal task-macro AUPRC rises from 0.0691 to 0.0856 (+0.0165), so the
registered task-macro direction is non-negative. After excluding
LIBERO-10/task6 (4,356 rows), C slightly worsens short MAE
`0.04038 -> 0.04062` while Spearman rises `0.4462 -> 0.4866`.
Leave-task-6-out terminal AUPRC is 0.00362 versus A's 0.00237 and prevalence
0.00275. It technically clears the prevalence rule but is too small to support
a practical failure claim.

An exploratory fully held-out failure task was not run: only 39 local-event
positive cached windows exist and they are strongly concentrated by task and
taxonomy.

## Action-only and shortcut audit

Action-only obtains 7-step MAE 0.03394, lower than both A and C, but its
Spearman is only 0.2897 and its 21/49-step MAE degrades to 0.0672/0.1067.
This mismatch is consistent with exploiting the target's near-zero-delta or
action-statistics structure rather than learning a transferable
state-conditioned consequence.

Simple action magnitude is not the direct explanation: after controlling for
same task and current-progress bin, Spearman between the 7-step target and
translation, rotation, pose magnitude, or gripper-switch count lies between
-0.013 and +0.005. Nevertheless, the action-only MAE anomaly means the broader
shortcut concern is not excluded.

## Sensitivity

Real-action permutation worsens C's short MAE by 8.58%, below the 10% MAE
threshold, while reducing regression AUPRC by 0.0839, which passes the
classification sensitivity alternative. It barely affects terminal AUPRC
(-0.0075), reduces `OBJECT_DROP` AUPRC by 0.0020, and improves
`FAILED_PLACEMENT` AUPRC by 0.0049. Thus the strongest sensitivity is not on
the same target that supplies the incremental classification gate.

The cross-episode real-action swap surrogate changes the short-progress
prediction by mean absolute 0.0164. This is action sensitivity, not a
same-state counterfactual effect.

Masking all actions worsens short MAE to 0.06945; first-action-only gives
0.05972; endpoint-summary gives 0.04109. These ablations show that C consumes
the chunk, but consumption alone does not establish useful, stable incremental
information.

## Gate

Passed: all integrity checks, classification increment, permutation
sensitivity, task-macro direction, leave-task6 prevalence, and deployability.

Failed:

- progress increment (0.98% MAE reduction / +0.040 Spearman);
- short-progress fold consistency (3/5, required 4/5).

Accordingly this is Result B, not Result A. The evidence does not authorize
LG-R2b.

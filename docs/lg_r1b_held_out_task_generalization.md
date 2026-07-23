# LG-R1b held-out-task SARM generalization

## Frozen zero-shot result

The exact LG-R1 validation-selected checkpoint was evaluated before
adaptation:

```text
SHA-256:
f6cc0ace62e790c1c69ec7ceb67a5a8ef2fe8653eb161db56f71b284844b9873
optimizer steps: 0
trained parameters: none
held-out tasks: 8
samples: 8,000 (1,000 per task)
```

| Metric | LG-R1 in-distribution | LG-R1b unseen-task zero-shot |
|---|---:|---:|
| Progress MAE | 0.0205 | 0.2112 |
| Progress RMSE | 0.0520 | 0.2899 |
| Progress Spearman | 0.9774 | 0.3775 |
| Pairwise accuracy | 0.8654 | 0.5245 |
| Cross-stage pairwise accuracy | 0.6692 | 0.3995 |
| Stage accuracy | 0.9310 | 0.3729 |
| Stage macro-F1 | 0.7992 | 0.2816 |
| Stage-completion MAE | 0.0553 | 0.3057 |
| Stage-completion Spearman | 0.9024 | 0.1591 |
| Monotonicity violation | 0.0092 | 0.1169 |
| Stagnation recall | 0.8980 | 0.5982 |
| Stage-regression recall | 0.2500 | 0.0000 |

This is Result C: the frozen LG-R1 SARM does not generalize reliably to the
pre-registered unseen tasks. Pairwise ordering is close to random, stage
classification collapses, and regression recall is zero.

## Per-task progress

| Task | MAE | Spearman |
|---|---:|---:|
| LIBERO-10 task 2 | 0.1468 | 0.6873 |
| LIBERO-10 task 4 | 0.1613 | 0.5129 |
| LIBERO-10 task 6 | 0.0657 | 0.8416 |
| LIBERO-10 task 7 | 0.1545 | 0.4337 |
| LIBERO-10 task 8 | 0.4011 | 0.5272 |
| LIBERO-10 task 9 | 0.1649 | 0.6518 |
| LIBERO-Goal task 6 | 0.3169 | 0.5652 |
| LIBERO-Goal task 9 | 0.2785 | -0.1234 |

The failure is systematic rather than a single bad task. LIBERO-10 task 6 is
the only task that individually meets both the MAE and rank-correlation
adaptation thresholds. LIBERO-10 task 8 has the largest MAE, and
LIBERO-Goal task 9 has negative rank correlation.

The evidence supports task/stage-schema dependence. It does not isolate
whether the model used elapsed-time or a visual shortcut, because LG-R1b did
not run a causal shortcut-removal intervention. No such causal claim is made.

## Pre-registered lightweight adaptation

All three adaptation triggers fired. The task-level split remained frozen:
four train tasks, two validation tasks and two untouched test tasks. Only
`StageTransformer` and `SubtaskTransformer` were trainable; VLA-JEPA, Qwen,
the policy, processor and CLIP remained frozen.

The run early-stopped after 900 optimizer steps. Validation MAE selected step
400 before the untouched test set was accessed.

| Held-out-test metric | Frozen initial, same 1,500 samples | Adapted |
|---|---:|---:|
| Progress MAE | 0.2216 | 0.2058 |
| Progress Spearman | 0.2854 | 0.3806 |
| Pairwise accuracy | 0.4655 | 0.5535 |
| Cross-stage pairwise accuracy | 0.4078 | 0.6064 |
| Stage accuracy | 0.2727 | 0.2540 |
| Stage macro-F1 | 0.1804 | 0.1836 |
| Stage-completion Spearman | 0.1330 | -0.0207 |
| Monotonicity violation | 0.1570 | 0.2430 |
| Stagnation recall | 0.5353 | 0.5299 |

Adaptation improves some progress metrics but does not repair the model:
stage accuracy decreases, stage-completion ordering reverses, monotonicity
worsens and stagnation recall does not improve. This partial numerical gain
must not be described as held-out-task success.

Machine-readable evidence:
`artifacts/lg_r1b/sarm_zero_shot_results.json` and
`artifacts/lg_r1b/sarm_adaptation_results.json`.

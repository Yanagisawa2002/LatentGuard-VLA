# LG-R1 multi-task progress-baseline report

## Outcome

LG-R1 is **Result B: the SARM-style progress baseline is useful, but natural
failure data are insufficient**.

On 160 frozen on-policy LIBERO rollouts, a stage-aware visual/language model
substantially outperformed online time progress and retained that improvement
on the untouched test split. It represents task advancement, stagnation, and
some regressions well enough to be a strong progress baseline. It does not
establish failure prediction, causal diagnosis, safety detection, action
ranking, or intervention value. Only one natural failure was observed, so the
LG-R2 failure-data gate remains closed.

## Frozen scope

- policy/checkpoint and processor revision:
  `735d9f692981e286ade093b5046627eda876e5d0`;
- policy action horizon: 7, relative control, unchanged scale and gripper
  convention;
- LeRobot 0.6.0 commit:
  `30da8e687a6dfc617fcd94afc367ac7071c376ce`;
- task registry: eight tasks, four suites, seeds `2000..2019`;
- final seeds `900000..900099`: not accessed;
- policy optimizer steps/backward calls: 0/0;
- no synthetic failure, policy/VLA-JEPA fine-tuning, failure head, candidate
  ranking, intervention, RoboLab, ROS 2, or LangMani modification.

The run recorded 160 valid episodes, 24,373 frames, and 3,546 policy queries.
There were 159 successes and one natural horizon-exhaustion failure.

## Dataset and split

The current-state dataset contains 24,373 labeled samples. It also contains
8,878 pairwise samples and 8,878 executed-action windows: 8,760 successful
windows and 118 windows from the one natural failure. There are no synthetic
corruption samples.

Splits are episode- and seed-atomic:

| Split | Seeds | Episodes |
| --- | --- | ---: |
| train | `2002, 2005, 2006, 2007, 2009, 2010, 2012, 2013, 2015, 2017, 2018, 2019` | 96 |
| validation | `2000, 2004, 2011, 2014` | 32 |
| test | `2001, 2003, 2008, 2016` | 32 |

Episode leakage and seed leakage are both zero. This supports held-out-seed
evaluation. It does not support held-out-task or held-out-suite claims because
every frozen seed was shared by every task; creating those splits would have
violated seed exclusivity.

## Baselines

### Time progress

The deployable baseline is `step_index / episode_horizon`. The
`step_index / observed_episode_length` variant is reported only as an
offline oracle because the actual termination length is unknown online.

| Split | Baseline | MAE | RMSE | Spearman | Pairwise accuracy | Regression recall | Stagnation recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | online time | 0.1778 | 0.2215 | 0.8779 | 0.6220 | 0.000 | 0.000 |
| test | online time | 0.1726 | 0.2172 | 0.8667 | 0.6455 | 0.000 | 0.000 |
| test | offline oracle | 0.1695 | 0.1971 | 0.8922 | n/a | n/a | n/a |

### Frozen VLA-JEPA representation probe

The probe mean-pools
`VLAJEPAModel._encode_qwen` embodied-action tokens into a 2,048-dimensional
feature. Only an 18,441-parameter linear stage/progress head was optimized.
The VLA-JEPA digest was identical before and after extraction, and the
optimizer contained no backbone parameters.

| Split | Progress MAE | RMSE | Spearman | Stage accuracy | Stage macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.0338 | 0.0475 | 0.9793 | 0.978 | 0.8524 |
| test | 0.0364 | 0.0510 | 0.9766 | 0.983 | 0.8564 |

Validation selected step 1,300; early stopping ended at step 2,100.
Checkpoint save/resume passed. This is a frozen-representation probe, not a
LatentGuard failure head.

### SARM-style small baseline

No official checkpoint was found that is compatible with LIBERO, this stage
schema, and the frozen observation contract. LG-R1 therefore used the official
LeRobot `StageTransformer` and `SubtaskTransformer` with frozen
`openai/clip-vit-base-patch32` at revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. The result is accurately named a
**SARM-style small baseline using the official LeRobot SARM modules**, not an
official SARM checkpoint reproduction or zero-shot result.

The causal video input uses exactly eight current/past frames at deltas
`[-49, -42, -35, -28, -21, -14, -7, 0]`; no future frame is used. The bounded
feature set contains 8,332 examples with video shape `[8332, 8, 512]`, text
shape `[8332, 512]`, and state shape `[8332, 8, 32]`. Only the 4,169,234
parameters in the stage/progress transformers were optimized. CLIP remained
frozen and outside the optimizer.

| Split | Progress MAE | RMSE | Spearman | Stage accuracy | Stage macro F1 | Pairwise accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.0157 | 0.0345 | 0.9883 | 0.9466 | 0.8164 | 0.8745 |
| test | 0.0205 | 0.0520 | 0.9774 | 0.9310 | 0.7992 | 0.8654 |

On test, same-stage pairwise accuracy is `0.8827` and cross-stage accuracy is
`0.6692`. Temporal metrics are:

| Split | Monotonicity violations | Stagnation recall | Regression/negative-progress recall |
| --- | ---: | ---: | ---: |
| validation | 0.0086 | 0.8946 | 0.400 |
| test | 0.0092 | 0.8980 | 0.250 |

Validation selected step 3,800; early stopping ended at step 4,800.
Checkpoint save/resume and CLIP processor serialization passed. Peak allocated
training memory after frozen feature caching was approximately 0.60 GiB
(`642,387,968` bytes).

## Promotion decision

Both predeclared progress-baseline checks pass:

1. SARM validation MAE (`0.0157`) is lower than online-time validation MAE
   (`0.1778`).
2. SARM test MAE (`0.0205`) retains the improvement over online-time test MAE
   (`0.1726`).

The SARM-style model is promoted only as an LG-R1 progress baseline. Strong
progress MAE, correlation, stagnation recall, and pairwise ordering support
claims about current task progress on the covered held-out seeds. They do not
support a task-success, failure-detection, safety, candidate-ranking, or
intervention claim.

## Answers to the research questions

1. **Current stage:** yes within the covered task/seed distribution; test stage
   accuracy is `0.9310`.
2. **Continuous progress:** yes; test MAE is `0.0205` and Spearman is `0.9774`.
3. **Progress delta:** useful for stagnation and some regression, but negative
   progress recall is only `0.25` on test.
4. **Natural failure pattern:** the single failure shows mid-episode progress,
   regression, and a long terminal plateau, but one episode cannot support a
   population claim.
5. **Explain most policy failures:** unanswered; there is only one failure.
6. **VLA-JEPA extra information:** its frozen current representation is
   informative, but predictor statistics were only descriptively correlated;
   no validated incremental failure signal was established.
7. **Authorize LG-R2:** no. Natural failed episodes are `1`, below the required
   `10`.

## VLA-JEPA predictor descriptive comparison

The official training-style temporal path was evaluated on 32 bounded test
samples with zero optimizer steps and zero backward calls. It has no native
progress score and does not accept external numeric action candidates. Pearson
correlations with observed progress were:

| Statistic | Correlation |
| --- | ---: |
| action-token norm | -0.2934 |
| current-latent norm | 0.2974 |
| current/predicted L1 | -0.0985 |
| predicted-latent norm | -0.2730 |
| predicted temporal-change magnitude | -0.4110 |
| predictor variance | -0.2796 |

All sampled test episodes were successful because the only failed seed belongs
to the training split. These associations therefore do not demonstrate
failure separation or incremental value over the progress baseline. They are
descriptive correlations only, with no predictive, causal, ranking, or safety
claim.

## Remote cost and timing

The pre-run estimate was 8–16 RTX 5090 GPU-hours and one to two elapsed days.
Measured primary GPU work was approximately 0.98 hours:

- rollout episode execution sum: 2,499.18 seconds;
- representation probe: 140 seconds;
- SARM smoke: 103 seconds;
- SARM training: 768 seconds;
- final progress evaluation: 31 seconds.

Including short audits and orchestration, observed use remained below roughly
1.1 GPU-hours and completed within one working session. The server was left
online and SSH-ready.

## Evidence boundary

Raw rollouts, videos, feature caches, predictions, and checkpoints remain
outside Git. The committed compact artifacts contain content hashes and
relative locators. The run directory name contains the initial implementation
short SHA, while every execution artifact records its actual source revision;
the compact execution summary binds the rollout, stage-label, and final-audit
commits separately.

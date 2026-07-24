# LatentGuard-VLA Project Brief

## Project in one line

LatentGuard-VLA is a preregistered empirical study of whether VLA actions can be
verified before execution using progress models, task-agnostic reward models,
numeric action conditioning, and simulator counterfactual replay.

`Research closeout` · `Counterfactual VLA verifier not validated` ·
`No online intervention claim`

## Problem

A VLA policy may produce a plausible action whose failure becomes visible only
after execution. The project asked whether progress prediction, general video
reward, numeric action conditioning, and exact simulator replay could provide
enough trustworthy evidence to score alternative actions beforehand.

## My responsibilities

- Defined a gate-driven research sequence from policy/interface validation
  through progress, reward, action-conditioning, and replay studies.
- Integrated pinned LeRobot/VLA-JEPA models with LIBERO and built a separate,
  bounded RoboLab compatibility/replay investigation.
- Curated multi-task on-policy rollout evidence and natural failures while
  preserving episode-level provenance and grouped splits.
- Implemented stage/progress evaluation, task-macro and held-out-task analysis,
  frozen reward-model comparison, numeric-action negative controls, and replay
  determinism checks.
- Preserved Result B/C outcomes, exact artifact identities, and stop decisions
  instead of tuning on failed gates.
- Prepared an upstream-quality RoboLab configuration patch and issue/PR drafts
  without claiming that the patch solved replay determinism.

## Technical stack

Python 3.11, PyTorch, LeRobot, VLA-JEPA, LIBERO, RoboLab, NumPy, scikit-learn
style grouped evaluation, JSON evidence contracts, PyTest, Ruff, MyPy,
Matplotlib, Git, and exact-revision remote execution.

## Verified scale

- 40-episode task-0 VLA-JEPA development smoke: 39/40 success.
- 160-episode in-domain progress study: 159 successes, 1 natural failure.
- 320-episode multi-task study: 293 successes, 27 natural failures, 8 tasks,
  2 suites, and 79,326 frames.
- 5-fold episode-grouped action-conditioning evaluation plus
  leave-task-6-out analysis.
- 10 LIBERO restore probes with 5 repeats each.
- 30 RoboLab replay attempts across 10 recordings and 3 repeats.

The compact artifacts do not contain a complete cross-phase GPU-hour or currency
total, so none is claimed.

## Numerical conclusion

Across 320 multi-task LIBERO episodes, task-specific progress modeling collapsed
from 0.977 to 0.378 Spearman correlation on held-out tasks; the best standalone
zero-shot task-agnostic reward model reached only 0.164 failure AUPRC; numeric
action conditioning improved short-horizon prediction MAE by just 0.98%; and 20
of 30 repeated RoboLab replays violated the official 0.01 state tolerance
despite a successful compatibility patch.

在320条多任务LIBERO轨迹上，任务特定进度模型的Spearman相关性从域内0.977降至未见任务0.378；最佳独立零样本任务无关奖励模型的失败AUPRC仅为0.164；加入数值动作后短期预测MAE只改善0.98%；修复RoboLab配置重放问题后，30次重复回放中仍有20次违反官方0.01状态容差。

## Why the negative result matters

The result separates several commonly conflated claims. A high in-domain
progress score did not establish cross-task transfer. A general video reward
model did not become a score for an action that had not run. Numeric actions
were correlated with outcomes but supplied little stable progress gain under
held-out-task evaluation. Replay that completed and agreed at termination still
diverged during intermediate dynamics.

Those findings prevented an unsupported candidate-ranking or online
intervention claim. They also produced reusable evaluation patterns: grouped
splits, held-out-task checks, negative controls, frozen gates, source-bound
artifacts, and replay-fidelity distinctions.

## Relevance to target roles

- **Robot Learning / VLA:** shows model-interface analysis, action
  representation experiments, cross-task generalization testing, and restraint
  around causal claims.
- **Robotics Simulation / Evaluation:** shows multi-environment integration,
  state restoration and replay diagnostics, failure taxonomies, and
  upstream-quality compatibility work.
- **ML / Research Engineering:** shows preregistration, controlled baselines,
  reproducibility, artifact identity, exact revision control, and honest
  closeout of a failed research hypothesis.

Full evidence is in the [final report](../latentguard_final_report.md) and
[result matrix](../latentguard_result_matrix.md).

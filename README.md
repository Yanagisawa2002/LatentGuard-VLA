# LatentGuard-VLA

> **LatentGuard-VLA is a preregistered empirical study of whether VLA actions can be verified before execution using progress models, task-agnostic reward models, numeric action conditioning, and simulator counterfactual replay.**

> **LatentGuard-VLA 系统研究了任务进度模型、通用视频奖励模型、数值动作条件化和模拟器反事实重放，能否支持VLA动作执行前验证。**

`Research closeout` · `Counterfactual VLA verifier not validated` ·
`No online intervention claim`

## Numerical conclusion

Across 320 multi-task LIBERO episodes, task-specific progress modeling collapsed
from 0.977 to 0.378 Spearman correlation on held-out tasks; the best standalone
zero-shot task-agnostic reward model reached only 0.164 failure AUPRC; numeric
action conditioning improved short-horizon prediction MAE by just 0.98%; and 20
of 30 repeated RoboLab replays violated the official 0.01 state tolerance
despite a successful compatibility patch.

在320条多任务LIBERO轨迹上，任务特定进度模型的Spearman相关性从域内0.977降至未见任务0.378；最佳独立零样本任务无关奖励模型的失败AUPRC仅为0.164；加入数值动作后短期预测MAE只改善0.98%；修复RoboLab配置重放问题后，30次重复回放中仍有20次违反官方0.01状态容差。

This is a completed research closeout, not a claim that a candidate verifier,
world model, safety system, or online selector was validated. The project
produced a reproducible chain of strong baselines, controlled failures, and
explicit stop decisions.

## Why this problem matters

A VLA policy can emit a plausible action that fails only after contact or
execution. A useful pre-execution verifier would need to distinguish candidate
actions before the robot commits to them, generalize beyond task-specific
progress labels, and rely on counterfactual outcomes generated from the same
physical state. Each requirement is individually difficult; treating them as a
single model-quality problem hides data confounding and simulator validity
failures.

LatentGuard therefore tested the prerequisites in sequence and stopped whenever
a preregistered gate failed:

1. Can the VLA policy and temporal predictor run, and do they expose candidate
   scoring inputs?
2. Does task-specific progress generalize beyond familiar tasks?
3. Can task-agnostic video reward identify natural failures?
4. Does numeric action conditioning add stable information beyond state?
5. Can two simulators reproduce the same state faithfully enough for
   counterfactual branching?

## What the project found

![In-domain versus held-out-task progress metrics](docs/figures/sarm_generalization_gap.png)

- **Strong in-domain progress did not transfer.** SARM-style progress reached
  0.977 Spearman correlation in-domain, then fell to 0.378 on held-out tasks;
  pairwise accuracy fell from 0.865 to 0.524.
- **Post-execution reward remained a weak proxy.** The best standalone
  zero-shot task-agnostic reward signal, ROBOMETER, reached 0.164 failure AUPRC
  at 0.630 recall, 0.160 precision, and 0.304 false-positive rate. It consumes
  executed video and does not directly score an unexecuted numeric candidate.
- **Actions contained local signal, not a stable independent gain.** Adding
  numeric actions changed short-horizon MAE from 0.040997 to 0.040597
  (0.98%), Spearman from 0.430 to 0.470, and terminal-failure AUPRC from 0.149
  to 0.203. The MAE gain reversed in leave-task-6-out evaluation.

![State-only versus state-plus-action results](docs/figures/action_conditioning_delta.png)

- **Mechanical replay completion was not faithful replay.** A compatibility
  patch fixed a callable-list configuration blocker without `eval` or `exec`,
  and all 30 RoboLab replays completed. Yet repeats 1 and 2 produced 20/20
  official-tolerance failures, 800 strict per-step failures, and maximum error
  0.149008. Terminal agreement did not establish stepwise dynamics agreement.

![RoboLab replay reproducibility](docs/figures/robolab_replay_reproducibility.png)

## Result table

| Phase | Scale | Primary result | Frozen decision |
| --- | --- | --- | --- |
| LG-R0 | 40 task-0 development episodes | 39/40 policy success; no external numeric-candidate scoring interface | Result B |
| LG-R1 | 160 episodes; 1 natural failure | In-domain progress Spearman 0.977; failure data insufficient | Failure-data gate failed |
| LG-R1b | 320 episodes; 8 tasks; 27 natural failures | Held-out-task Spearman 0.378 and pairwise accuracy 0.524 | Failure-data gate passed; model did not generalize |
| LG-R1c | 320 episodes | Best standalone zero-shot reward failure AUPRC 0.164 | Result B |
| LG-R2a | 320 episodes; grouped evaluation | Numeric-action short-MAE gain 0.98%; gain not stable out of task | Result B |
| LG-R2b0 | 10 restore probes; 5 repeats each | 36 stepped-trace failures; max error 0.467783 | Result C |
| LG-RB0/RB0.1 | 30 repeated replays | 20 official 0.01-tolerance failures after compatibility patch | Result C; route closed |

The source-linked [final result matrix](docs/latentguard_result_matrix.md) and
[final research report](docs/latentguard_final_report.md) preserve the full
scope, metrics, and gate meanings.

## What worked

- A pinned LeRobot/VLA-JEPA/LIBERO stack and a task-0 development smoke.
- Multi-task on-policy rollout curation with 320 episodes, 27 natural failures,
  8 tasks, 2 suites, and 79,326 frames.
- Stage/progress annotation, task-macro analysis, held-out-task evaluation, and
  episode-grouped cross-validation.
- Uniform comparisons of frozen SARM, ROBOMETER, TOPReward, calibration, and a
  descriptive ensemble.
- Numeric action-conditioning controls, including state-only, action-only,
  permutation, action-swap, and leave-task-out comparisons.
- Simulator restore/replay gates that separated mechanical completion, terminal
  agreement, official tolerance, and strict per-step identity.
- Frozen revisions, content-bound artifacts, exact source pointers, and a small
  deterministic closeout builder.

## What did not work

- No new candidate outcome model was validated.
- No candidate ranking or online intervention was run.
- No incremental value of a learned world representation was established.
- Single-policy on-policy data did not identify a stable cross-task action
  effect.
- LIBERO and RoboLab did not support the required repeated, exact same-state
  replay contract.
- No real-robot deployment was attempted.

These negative outcomes are retained because they locate the failure boundary:
high in-domain metrics, terminal agreement, or an execution fix cannot
substitute for cross-task generalization and faithful counterfactual evidence.

## Why the project stopped

The exact same-state counterfactual route is closed. The project will not move
to a third simulator, add another reward model, or continue replay-infrastructure
repair.

精确同状态反事实路线已经关闭。本项目不会迁移到第三个模拟器，不会继续增加奖励模型，也不会继续修复重放基础设施。

Continuing would have changed the research question after two simulator
failures, weak task-agnostic reward results, and unstable action-conditioning
gains. Any future work must start as an independent project with a separate
repository, protocol, and success criteria.

## Engineering contributions

The completed contribution is research engineering and empirical diagnosis:

- integration across LeRobot, VLA-JEPA, LIBERO, and RoboLab;
- multi-task trajectory and natural-failure evidence;
- typed progress, failure, action, replay, and artifact contracts;
- grouped and held-out-task evaluation with negative controls;
- an upstream-quality RoboLab configuration patch and restrained upstream
  [issue](docs/lg_rb01_upstream_issue_draft.md) /
  [PR](docs/lg_rb01_upstream_pr_draft.md) drafts;
- source-bound final summaries and reproducible figures.

See the [project brief](docs/portfolio/project_brief.md), tailored
[resume bullets](docs/portfolio/resume_bullets.md), and
[interview pitch](docs/portfolio/interview_pitch.md).

## Repository structure

- `src/latentguard/`: simulator-independent contracts and evaluation utilities.
- `configs/`: versioned experiment and compatibility configurations.
- `artifacts/`: compact committed manifests, metrics, gates, and the
  [artifact index](artifacts/README.md).
- `docs/`: phase reports, the [documentation index](docs/index.md), final
  report, result matrix, and portfolio materials.
- `scripts/lg_f0_build_summary.py`: deterministic final-summary builder.
- `scripts/final_closeout/`: one source-bound script per final figure.
- `patches/robolab/`: the preserved RoboLab compatibility patch.
- `tests/`: CPU-only core tests with explicit external-runtime boundaries.

Raw rollouts, simulator states, video, model downloads, checkpoints, caches, and
large run directories are intentionally not stored in Git.

## Reproduction

Use Python 3.11:

```bash
python -m pip install -e ".[dev]"
python scripts/lg_f0_build_summary.py --check
python scripts/lg_f0_generate_figures.py
python scripts/lg_f0_validate_closeout.py
python -m pytest
ruff check .
ruff format --check .
mypy src
python -m build
```

Figure generation additionally requires Matplotlib and Pillow. These commands
rebuild only compact summaries and figures from committed JSON; they do not
reproduce GPU training, rollout collection, or simulator replay.

## Limitations

The LIBERO evidence is on-policy and limited to eight tasks in two suites; the
39/40 result is a task-0 development smoke, not the official 400-episode
benchmark. Failure AUPRC uses 27 natural failures among 320 episodes. Reward
models consume post-execution video. Action-conditioning comparisons remain
observational and confounded by the behavior policy. Replay probes cover
specific pinned LIBERO and RoboLab environments and do not establish a universal
simulator limitation. No candidate verifier, ranking, intervention, general
safety property, cross-robot transfer, or real-robot behavior was validated.

The repository also retains an earlier, frozen ManiSkill PickCube research line
and its release evidence. LG-F0 does not rewrite those historical results and
does not use them to support the post-release VLA counterfactual-verifier claim.
Its machine-audited key results remain available in the frozen
[PickCube result table](docs/portfolio/key-results.md). The following hidden
registry bindings are retained only for compatibility with that accepted
release audit:

<!-- LG-RESULT:m2c-strong-replays -->
<!-- LG-RESULT:m3a-verified-outcomes -->
<!-- LG-RESULT:m3b-test-auprc -->
<!-- LG-RESULT:m3c-temporal-success -->
<!-- LG-RESULT:m4b-external-visual-success -->
<!-- LG-RESULT:m4c-distilled-vs-fixed -->
<!-- LG-RESULT:m4d-fault-gated-success -->
<!-- LG-RESULT:m4d-override-recall -->

## Citation

```bibtex
@misc{latentguard_vla_2026,
  title  = {LatentGuard-VLA: An Empirical Study of Pre-execution VLA Action Verification},
  author = {Yanagisawa, Edwin},
  year   = {2026},
  note   = {Research closeout; counterfactual VLA verifier not validated}
}
```

## License

No explicit license file is provided in this repository. Source availability
does not by itself grant permission to reuse, modify, or redistribute the work;
contact the repository owner before reuse.

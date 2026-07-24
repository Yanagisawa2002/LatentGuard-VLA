# LatentGuard-VLA 2–3 Minute Demo Storyboard

This storyboard is a recording plan only. It does not require new rollout,
simulator execution, or model inference.

| Time | Visual | Narration | Evidence |
| --- | --- | --- | --- |
| 0:00–0:20 | Simple question card: “A VLA proposes an action. Can we verify it before execution?” followed by the research-closeout labels | Introduce the gap between a plausible action and a trustworthy pre-execution comparison | [Final report: problem definition](../latentguard_final_report.md#problem-definition) |
| 0:20–0:40 | LG-R0 card: 39/40 task-0 development successes; interface diagram showing internal token features versus external numeric candidates | The policy ran, but its predictor did not expose the interface needed to score arbitrary numeric candidates | [LG-R0 reports](../lg_r0_world_model_interface_report.md) |
| 0:40–1:05 | Animate the bars in `sarm_generalization_gap.png` | In-domain progress looked excellent, then collapsed on held-out tasks: Spearman 0.977 to 0.378 | [Progress figure](../figures/sarm_generalization_gap.png) |
| 1:05–1:25 | Show `reward_model_failure_metrics.png`; highlight ROBOMETER 0.164 AUPRC and 0.304 FPR | Task-agnostic reward was weak and used post-execution video, so it was not an unexecuted-action score | [Reward figure](../figures/reward_model_failure_metrics.png) |
| 1:25–1:45 | Show `action_conditioning_delta.png`; zoom to MAE 0.040997 versus 0.040597 | Numeric actions carried local signal, but the MAE gain was only 0.98% and reversed out of task | [Action figure](../figures/action_conditioning_delta.png) |
| 1:45–2:10 | Split screen: LIBERO restore failure counts, then `robolab_replay_reproducibility.png` | A compatibility patch made all RoboLab replays execute, but 20/30 still violated the official tolerance. Terminal agreement was not faithful replay | [Replay figure](../figures/robolab_replay_reproducibility.png) |
| 2:10–2:35 | `phase_funnel.png`, ending at route closed | Explain the stop: no third simulator, no extra reward model, no infrastructure repair loop | [Phase figure](../figures/phase_funnel.png) |
| 2:35–2:55 | Three columns: engineering delivered / empirical findings / unvalidated goals | Close with the honest value: reproducible integrations and evaluation, bounded negative findings, and no candidate-selector or intervention claim | [Result matrix](../latentguard_result_matrix.md) |

## Recording notes

- Use only committed charts, diagrams, and text.
- Keep “development smoke” visible whenever 39/40 is shown.
- Keep “post-execution” visible whenever reward-model results are shown.
- Use “mechanically completed” for 30/30 RoboLab runs; never call them 30
  faithful replays.
- End on the research-engineering contributions, not on a speculative new
  LatentGuard phase.
- Target duration: 2 minutes 45 seconds.

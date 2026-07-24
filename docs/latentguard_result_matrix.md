# LatentGuard-VLA Final Result Matrix

This matrix is a readable projection of
[`artifacts/final/final_summary.json`](../artifacts/final/final_summary.json).
It preserves the original phase decisions: “gate passed” means only that the
named phase gate passed, not that a counterfactual verifier or task success was
established.

| Phase | Research question | Dataset / task scale | Main baseline | Primary metric | Result | Gate decision | What was learned | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LG-R0 | Can a pinned VLA-JEPA policy and temporal predictor execute on LIBERO data? | 40 task-0 development episodes | Pinned LeRobot VLA-JEPA LIBERO checkpoint | Development smoke success | 39/40, 97.5%; not the official 400-episode benchmark; temporal predictor ran but had no external numeric-candidate or native risk/success interface | Result B: interface mismatch | Strong policy execution did not establish candidate-verification inputs | Study progress and failure signals on multi-task trajectories |
| LG-R1 | Can task-specific stage supervision provide a strong in-domain progress signal? | 160 episodes; 159 success, 1 natural failure; 8 tasks, 4 suites | SARM-style task-specific progress model | In-domain progress Spearman | MAE 0.0205; Spearman 0.9774; pairwise 0.8654; stagnation recall 0.8980 | `FAILURE_DATA_GATE_NOT_MET` | The in-domain signal was strong, but one failure could not support failure verification | Collect natural failures and run held-out-task evaluation |
| LG-R1b | Does the task-specific progress ontology generalize to held-out tasks? | 320 episodes; 293 success, 27 natural failures; 8 tasks, 2 suites; 79,326 frames | Frozen LG-R1 SARM | Held-out-task progress Spearman | MAE 0.2112; Spearman 0.3775; pairwise 0.5245; stage accuracy 0.3729; regression recall 0 | Failure-data gate passed | The dataset enabled failure study, while task-structured progress collapsed on unseen tasks | Compare task-agnostic post-execution reward signals |
| LG-R1c | Can task-agnostic video reward provide a transferable failure signal? | 320 episodes; 27 natural failures | ROBOMETER zero-shot | Episode failure AUPRC | AUPRC 0.1638; precision 0.1604 at recall 0.6296; FPR 0.3038; TOPReward, calibration, and simple ensemble did not pass the formal gate | Result B | Post-execution reward remained weak and did not score an unexecuted numeric candidate | Test numeric action conditioning without promoting a reward baseline |
| LG-R2a | Does numeric action conditioning add stable predictive value beyond state? | 320 episodes; 79,326 frames; 5 grouped CV folds | State-only vs state-plus-action predictors | Short-horizon MAE relative reduction | MAE 0.040997 → 0.040597 (0.98%); Spearman 0.4300 → 0.4700; terminal AUPRC 0.1486 → 0.2029; leave-task-6-out MAE worsened 0.59% | Result B | Actions carried local signal, but the progress gain was small and not stable out of task | Audit same-state replay before any candidate-outcome experiment |
| LG-R2b0 | Can LIBERO support strict same-state counterfactual branching? | 10 probe states; 5 repeats each; 4 unique policy-generated candidates | Frozen VLA-JEPA candidates with LIBERO state restoration | Stepped-trace restore failures | 36 restore failures; 3 terminal mismatches; maximum numeric error 0.467783 | Result C | Candidate diversity existed, but the frozen restore contract could not support branching | Evaluate one bounded alternative replay platform |
| LG-RB0/RB0.1 | Can RoboLab provide repeated faithful replay after one compatibility patch? | 2 tasks; 10 recordings; 3 repeats; 30 attempts | Recorded actions with schema-aware config overlay | Official state-validator failures | 30/30 completed; repeat 0 passed 10/10; repeats 1–2 failed 20/20; 800 strict step failures; maximum error 0.149008; no terminal/success mismatch | Result C; route closed | Mechanical completion and terminal agreement did not establish stepwise fidelity | Close the exact same-state counterfactual route |

## Source map

- **LG-R0:** [evaluation](../artifacts/lg_r0/libero_40ep_evaluation.json)
  `/overall`; [interface audit](../artifacts/lg_r0/world_model_interface_audit.json)
  `/external_numeric_candidate_actions_supported`,
  `/numeric_action_chunk_consumed_by_predictor`, `/native_scalar_scores`.
- **LG-R1:** [dataset](../artifacts/lg_r1/dataset_manifest.json)
  `/valid_rollout_episodes`, `/natural_failed_episodes`;
  [results](../artifacts/lg_r1/sarm_results.json) `/test`;
  [gate](../artifacts/lg_r1/lg_r2_gate.json) `/status`.
- **LG-R1b:** [dataset](../artifacts/lg_r1b/dataset_manifest.json);
  [held-out results](../artifacts/lg_r1b/sarm_zero_shot_results.json)
  `/held_out_task`; [gate](../artifacts/lg_r1b/lg_r2_gate.json) `/status`.
- **LG-R1c:** [evaluation](../artifacts/lg_r1c/evaluation_summary.json)
  `/models`; [gate](../artifacts/lg_r1c/lg_r2_gate.json)
  `/candidate_gates`, `/result_class`.
- **LG-R2a:** [state-only](../artifacts/lg_r2a/state_only_results.json)
  `/metrics`; [state-plus-action](../artifacts/lg_r2a/state_action_results.json)
  `/metrics`; [leave-task-6-out](../artifacts/lg_r2a/leave_task6_out_results.json)
  `/models`; [gate](../artifacts/lg_r2a/lg_r2b_gate.json) `/result_class`.
- **LG-R2b0:** [candidate source](../artifacts/lg_r2b0/candidate_source_manifest.json);
  [restore validation](../artifacts/lg_r2b0/state_restore_validation.json);
  [gate](../artifacts/lg_r2b0/lg_r2b1_gate.json).
- **LG-RB0/RB0.1:**
  [replay validation](../artifacts/lg_rb01/faithful_replay_validation.json);
  [gate](../artifacts/lg_rb01/lg_rb1_gate.json).

## Stop statement

The exact same-state counterfactual route is closed. The project will not move
to a third simulator, add another reward model, or continue replay-infrastructure
repair.

精确同状态反事实路线已经关闭。本项目不会迁移到第三个模拟器，不会继续增加奖励模型，也不会继续修复重放基础设施。

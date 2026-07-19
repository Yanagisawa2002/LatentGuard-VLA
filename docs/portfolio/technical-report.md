# LatentGuard-VLA: Evidence-Bound Action Verification from Exact Replay to Conservative Shielding

## Abstract

LatentGuard-VLA is a contract-first framework for evaluating whether a robot action chunk will succeed before execution. The project built simulator-independent data and evidence models, physically validated exact-state paired replay for ManiSkill PickCube, generated 2,880 strong counterfactual outcomes, trained structured and visual verifiers, and evaluated them in blind one-shot and repeated-control settings. One-shot selection was strongly positive: the M3C temporal selector reached 98.33% versus 88.89% random, and M4B external visual selection reached 98.89%. Repeated intervention falsified the wider deployment hypothesis: M4C distilled visual shielding reached 73.33% versus 100% for the fixed primary. A conservative M4D redesign preserved clean success and improved controlled injected-fault success from 73.33% to 80.00% at 1.17% intervention, but override recall was only 4.03%. The frozen result is therefore a validated evidence/verification system, positive one-shot action selection, a negative repeated-intervention finding, and a partial conservative recovery—not a general robot-safety claim.

<!-- LG-RESULT:m2c-strong-replays --> <!-- LG-RESULT:m3a-verified-outcomes --> <!-- LG-RESULT:m3b-test-auprc --> <!-- LG-RESULT:m3c-temporal-success --> <!-- LG-RESULT:m4b-external-visual-success --> <!-- LG-RESULT:m4c-distilled-vs-fixed --> <!-- LG-RESULT:m4d-fault-gated-success --> <!-- LG-RESULT:m4d-override-recall -->

## 1. Problem definition

Given a state (s_t), a nominal action chunk (a_t), and a fixed candidate pool, LatentGuard estimates candidate failure risk and optionally selects or gates an alternative. The evidence question precedes the learning question: for the same content-bound simulator state, did an independently replayed original action succeed, and what happened when a candidate was executed in a separately restored session? Only complete, conclusive replay can become an outcome label.

The system distinguishes four evaluation levels:

1. **Offline classification:** predict outcome labels on accepted samples.
2. **Blind one-shot selection:** choose one candidate from one fixed pool before outcomes are available.
3. **Repeated closed-loop intervention:** select again after each new observation and induced state.
4. **Controlled fault injection:** compare accept, gate, alternative, and fallback behavior on a frozen nominal-fault schedule.

No level is evidence for a later level by implication.

## 2. Why action verification matters

Robot-policy evaluation usually scores whole trajectories. That can reveal failure frequency but not whether a local action was avoidable. Action verification can support diagnosis, risk-aware ranking, abstention, or fallback. It is also easy to overclaim: a corruption heuristic is not a failure label; a simulator exception is not task failure; an approximate state reconstruction is not the same counterfactual; and a high offline metric is not a control policy. LatentGuard makes these distinctions executable.

## 3. System architecture

The [architecture document](architecture.md) contains four exact diagrams. The core owns typed episodes, corruption proposals, replay/evidence models, selection inputs, transactional ledgers, and release audit. Simulator-native state and task objects remain behind typed adapters. Training consumes only allowlisted state, action, mask, or image inputs; provenance, corruption metadata, domains, and outcomes remain reporting-only. Remote machines execute exact pushed revisions and never author tracked source.

## 4. Exact-state counterfactual methodology

Every replay case binds source dataset, proposal, action bytes, adapter configuration, state reference, task contract, and compatibility identity. The original source action is a mandatory validity gate. Baseline and candidate run in independent sessions restored from the same archived state. Adapter exceptions remain execution errors; state/action mismatches remain invalid context.

Serialized PickCube state is exact in structure, inventory, dtype, shape, bytes, and archive digest. The runtime adapter declares `tolerance_verified_full_state_v1`: all 70 components must be finite, complete, ordered, and within a fixed 1e-6 maximum absolute error. The observed maximum was 1.1920929e-7. This tolerance is semantic configuration, not dynamic repair. M2C accepted 12/12 selected replay attempts as strong simulator evidence.

## 5. Evidence and data contracts

M3A split 60 source trajectories before derivation, selected 360 anchors, and evaluated 2,880 corruptions. Together with 360 source samples, the accepted dataset has 3,240 samples. All evidence references, source continuations, trajectory groups, and split assignments passed strict reload; all 2,880 completed evaluations were reused by a zero-work resume. Restoration checks covered 6,120 complete 70-component comparisons, while the 38-component verifier projection round-tripped exactly.

Corruptions remain single-source and window-bounded. Evidence strength and label source are explicit. Missing, invalid, indeterminate, or execution-error evidence cannot be silently converted to task failure. These rules let later model results inherit traceable physical context without embedding simulator objects in core models.

## 6. Structured verifier

M3B screened state-only (38,913 parameters), action-only (50,433), joint state-action MLP (61,249), and temporal state-action verifier (312,897) across five fixed seeds. Validation failure AUPRC means were 0.3868, 0.7304, 0.9088, and 0.9249 respectively; validation-only selection froze the temporal model before test evaluation.

On untouched test data, selected-model mean failure AUPRC was 0.8986, ROC AUC 0.9682, Brier score 0.0660, and expected calibration error 0.0632. In 36 eligible corrupted-candidate groups, top-1 success was 1.0. At requested 80% coverage, retained failure averaged 3.38% versus 17.90% at full coverage. This supports learnability in the fixed structured scope. It does not make the temporal model universally best: the smaller joint MLP later selected 358/360 successful M3C candidates versus 354/360 for temporal.

## 7. Visual verifier and teacher distillation

M4A rendered three fixed RGB views from content-bound states without advancing physics. Development contained 1,080 packets, 3,240 images, and 9,720 derived samples; external evaluation contained 1,080 packets, 3,240 images, and 8,640 derived samples. Complete-state maximum error remained 1.1920929e-7, verifier projection error was zero, pixel digests were unique and repeatable, no trajectory/image leakage was detected, and external training was programmatically rejected.

M4B used frozen ResNet-18 features and structured-teacher distillation under cost-aware screening. The external canonical visual selector reached 98.89% success; strong camera and lighting domains each reached 98.06%. Visual-minus-random was +10.24 percentage points with 95% CI [8.78, 11.67]. Visual-minus-action-only was only +0.28 points with CI [-0.83, 1.39], so the added-value claim is partial. No simulator replay occurred during M4B evaluation; it reused accepted M3C outcomes.

## 8. Blind one-shot intervention

M3C finalized candidate pools and selection manifests before outcomes were generated or loaded. Stage A received opaque IDs plus allowlisted state/action/mask tensors, never candidate family, severity, provenance, or outcomes. All selectors compared the same eight candidates in 360 groups from 60 untouched trajectories.

Random selected 320/360 successes (88.89%), action-only 355/360 (98.61%), joint MLP 358/360 (99.44%), temporal 354/360 (98.33%), and the analysis-only oracle 360/360. Temporal-minus-random was +9.44 points with trajectory-bootstrap 95% CI [6.39, 12.50]. Temporal-minus-joint was -1.11 points with an interval ending at zero. The accepted conclusion is strong blind one-shot selection, not temporal superiority and not closed-loop improvement.

## 9. Receding-horizon stress test

M4C used 60 fresh source trajectories disjoint from M3A, M3C, and smoke data. Ten selector/domain conditions produced 600 episodes. The fixed primary reached 100%; action-only 93.33%; privileged structured 91.67%; direct visual canonical 90%; distilled visual canonical 73.33%; and random 45%. Distilled minus random was +28.33 points with CI [11.67, 43.33], yet distilled minus fixed was -26.67 points and distilled minus privileged -18.33 points. Overall targets failed.

The benchmark recorded 501 successes, 98 horizon exhaustions, one unsafe outcome, zero task failures, and zero execution errors. Distilled visual selected non-primary candidates on 93.48% of decisions. The diagnosis is distributional and control-level: repeated action choice induced states not represented by the local one-shot evaluation, small ranking errors compounded, and a reliable nominal controller was displaced almost continuously.

## 10. Conservative fallback shield

M4D froze three validation-derived gate profiles and treated nominal acceptance as the default. It content-bound a fault schedule, distinguished fixed-fallback from other alternatives, and reused the fixed primary as a trusted operational reference. The final run covered 1,560 episodes and 24,414 decisions.

Clean primary performance was 100% with zero intervention, false override, unsafe outcome, or execution error. Under injected canonical faults, accept nominal reached 73.33%, gated direct 80%, ungated direct 90%, and always fallback 100%. Gated direct intervened on 1.17% of decisions and invoked fallback on 0.32%; the paired improvement over accept nominal was +6.67 points with 95% CI [1.67, 13.33]. But override recall was 4.03%, risk AUROC 0.5946, AUPRC 0.4706, and the success gap to fallback was -20 points. Ten of fourteen targets passed. This is a limited conservative benefit and clean-preservation result, not a solved shield.

## 11. Engineering reliability

### Deterministic identities

Semantic JSON, NumPy dtype/shape/bytes, action contracts, state references, configs, checkpoints, candidate pools, selection manifests, and evidence are content-bound. Runtime paths, hosts, process IDs, and timestamps do not participate.

### Transactional publication

Datasets and evaluation states publish atomically or exclusively; ledgers distinguish planned, running, complete, invalid, and execution-error work. Immutable source artifacts are reloaded and rehashed before use.

### Resume and crash recovery

Completed runs resume with zero work (2,880 M3A evaluations, 600 M4C episodes, and 1,560 M4D episodes). Mid-episode recovery restores the last committed full state and reuses the exact persisted decision rather than rescoring.

### Remote exact-SHA execution

Tracked files are changed locally, validated, committed, pushed, and pulled by exact full SHA. Remote checkouts must be clean; outputs go to untracked run roots. Runtime bugs are preserved as evidence, fixed locally, and resynchronized. M5 performs no GPU work and only a read-only artifact-preservation audit.

## 12. Results

The [canonical supported-results table](key-results.md) is the single human-maintained results surface. Its rows map to machine-readable result IDs whose values are checked against committed report bytes and JSON pointers. Classification, one-shot, closed-loop, and injected-fault metrics are not flattened into a single ranking.

## 13. Negative results and diagnosis

Three non-success conclusions are central:

- M3C's selected temporal model was not the strongest downstream selector; joint MLP was smaller and better.
- M4B did not establish visual improvement over action-only because the bootstrap interval crossed zero.
- M4C repeated distilled visual shielding materially underperformed the fixed primary. M4D removed over-intervention but did not provide high-recall interception or fallback-level success.

These findings remain in README, claims, key results, case study, and career materials. They cannot be relabeled by release prose.

## 14. Limitations

Evidence covers one ManiSkill PickCube task, one robot/controller contract, fixed solver-derived source behavior, synthetic action corruptions, fixed eight-candidate pools, three rendered views, predeclared visual shifts, and controlled injected faults. No real robot, multiple task/robot, learned VLA proposal distribution, arbitrary replanning, hardware fault, language input, LangMani policy, or safety certification was evaluated. The project predicts task-specific failure and performs bounded selection; it does not prove general safety.

## 15. Future work

Future evidence milestones could add real learned-policy proposals, a contract-frozen LangMani adapter, multiple independently accepted tasks, broader robot embodiments, and real-robot evaluation. Each requires a new adapter identity, untouched data, independent validation, and a new claims registry update. M5 deliberately starts none of this work.

## 16. Reproducibility

The ordinary release can be checked locally with Python 3.11:

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
ruff format --check .
mypy src
latentguard portfolio-smoke
latentguard audit-release --strict
```

`portfolio-smoke` exercises the miniature contracts using deterministic synthetic and non-physical fixture data. It is not a research reproduction. Large accepted datasets/checkpoints remain outside Git and are checked separately for availability. Report file digests, commits, run IDs, and JSON pointers are in [results.json](../release/results.json); the frozen milestone chain is in [milestones.json](../release/milestones.json).

## 17. Supported claims

The authoritative wording and permitted surfaces are [claims.json](../release/claims.json), with a readable summary in [claims.md](../release/claims.md). Supported claims cover the pinned exact replay, evidence scale, structured learnability, blind one-shot improvement, fixed-domain visual robustness, transactional resume, clean conservative preservation, and limited injected-fault benefit. Visual added value, distillation, general closed-loop improvement, and approach-to-fallback are only partial. General safety, real-robot robustness, transfer, VLA generalization, hardware-fault detection, language verification, and LangMani improvement remain unsupported or untested.
